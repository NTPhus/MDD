"""Local speech and Qwen inference, with bounded CUDA/CPU device policies."""
from __future__ import annotations
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REPEAT_REQUEST = "Sorry, I didn't catch that. Could you say it again, please?"


class UnrecognizedSpeech(ValueError):
    """ASR completed but produced no usable words, rather than a model failure."""


def native_condition(text, reference, phones):
    from scripts.cascade_bridge import predicted_condition, word_key
    words = text.split()
    annotations = {}
    for index, word in enumerate(words):
        annotations[word_key(word)] = [p for p, i in zip(reference["canonical_phones"], reference["canonical_word_indices"]) if i == index]
    row = {"reference_text": text, "annotation_info": annotations}
    ref = {**reference, "word_intervals": [{"index": i, "text": w} for i, w in enumerate(words)]}
    return predicted_condition(row, ref, phones)


def grounded_coaching(text, finished, comparison):
    from scripts.published_feedback_metrics import parse_prediction
    from scripts.cascade_bridge import word_key
    if not finished:
        raise ValueError("The coaching response reached its token limit.")
    parsed = parse_prediction(text)
    flagged = {word_key(w["text"]) for w in comparison["words"] if w["status"] == "difference"}
    if any(word_key(key) not in flagged for key in parsed):
        raise ValueError("The coaching response included a word without matching acoustic evidence.")
    return parsed


def conversation_history(request):
    """Use the same actual conversation for completion and the spoken reply."""
    from mdd.speech_topics import TOPICS
    history = list(request.get('history', []))
    topic_id = request.get('topic_id')
    if topic_id:
        history.insert(0, {'role': 'assistant', 'content': TOPICS[topic_id]['opening']})
    return history


def dialogue_request(request, transcript):
    from mdd.speech_topics import TOPICS
    from scripts.conversation_state import cafe_state, state_note
    state = None if request.get('needs_clarification') is True else cafe_state(request, transcript)
    if state is not None:
        return {**request, 'model_id': 'qwen-base', 'generation_profile': 'conversation',
                'conversation_state': state,
                'messages': [{'role': 'system', 'content': state_note(state)},
                             *list(request.get('history', [])), {'role': 'user', 'content': transcript}],
                'max_tokens': 96, 'timeout': 45.}
    if request.get('needs_clarification') is True:
        system = ("Some words in the latest speech transcript are unclear. Ask the user to say it again. "
                  "Do not guess the intended words, quote unclear words, correct grammar, or greet.")
        examples = []
    else:
        system = ("Continue the English conversation from the LAST user message, using the recent turns to remember what was said. "
                  "Answer the user's actual question first, including a brief, honest answer when they ask about you; do not claim personal experiences. "
                  "Then add a specific reaction or ask at most one short, open-ended follow-up that builds on a concrete detail and can gently expand the topic. "
                  "Avoid generic questions such as 'How about you?' when the user has already asked you something. "
                  "Use one or two short sentences; do not force a question when it would feel unnatural. "
                  "Do not greet again, repeat earlier questions, restate their sentence, or correct grammar. "
                  "Follow a new subject when the user changes it. The first exchanges are style examples; actual conversation follows.")
        examples = [{'role': 'user', 'content': 'I repair bicycles, mostly old ones.'},
                    {'role': 'assistant', 'content': 'What problems do you run into most often?'},
                    {'role': 'user', 'content': 'The brakes can be hard to adjust.'},
                    {'role': 'assistant', 'content': 'Do you test them one step at a time to find the cause?'},
                    {'role': 'user', 'content': 'I finally fixed it.'},
                    {'role': 'assistant', 'content': 'That must be a relief after all that effort.'}]
    topic_id = request.get('topic_id')
    if topic_id:
        topic = TOPICS[topic_id]
        if topic_id == 'food':
            system += (" Role-play a friendly cafe server. The learner is a customer practising an order; "
                       "acknowledge the items they name and continue the order naturally. Never refuse to help with an order "
                       "or claim you cannot assist with ordering. Do not invent menu availability or prices.")
            examples = [{'role': 'user', 'content': 'Could I have a tea, please?'},
                        {'role': 'assistant', 'content': 'Of course! What size would you like?'},
                        {'role': 'user', 'content': 'A medium tea, please.'},
                        {'role': 'assistant', 'content': 'Certainly. Is that for here or to go?'}]
        else:
            system += (f" Speaking scenario: {topic['title']}. Play the role of {topic['role']}; "
                       "keep the conversation natural and follow the learner if they change the subject.")
    # The authored opening has already been spoken. Reinserting it before
    # every real exchange anchors the tiny model to an old welcome/question.
    history = list(request.get('history', [])) or conversation_history(request)
    if topic_id:
        system += f"\nCurrent scenario: {TOPICS[topic_id]['title']}. Stay in the scenario while it fits the learner's turn."
    turn = transcript if not request.get('reference_text') else 'I just practised reading: ' + request['reference_text']
    return {**request, 'model_id': 'qwen-base', 'generation_profile': 'conversation', 'example_message_count': len(examples),
            'messages': [{'role': 'system', 'content': system}, *examples,
                         *history, {'role': 'user', 'content': turn}], 'max_tokens': 160, 'timeout': 75.}


def follow_up(text, topic_id, turns, *, transcript='', history=()):
    """Describe Qwen's reply without adding a question or advancing a script."""
    from mdd.speech_topics import TOPICS, guidance
    questions = [q.strip() for q in re.findall(r'[^.!?\n]*\?', text)]
    question = questions[-1] if questions else None
    source = 'qwen' if question else 'none'
    hint = {'question': question, 'keywords': [], 'starter': ''}
    if not topic_id or not question:
        return hint, source

    # Catalog hints are optional assistance, not the next scripted turn. Only
    # show them when they match what Qwen actually asked. Empty hints clear the
    # old topic card after a clarification or a learner-led change of subject.
    normalized = lambda value: re.sub(r'\W+', ' ', value.lower()).strip()
    stopwords = {'a', 'an', 'the', 'i', 'you', 'your', 'it', 'its', 'that', 'this', 'there',
                 'what', 'how', 'why', 'where', 'which', 'when', 'who', 'would', 'could', 'can',
                 'will', 'shall', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'to',
                 'of', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'but', 'so', 'about',
                 'as', 'if', 'like', 'get', 'me', 'my', 'we', 'our', 'have', 'has', 'had'}
    def tokens(value):
        return set(re.findall(r'[a-z]+', value.lower().replace('favorite', 'favourite'))) - stopwords
    question_tokens = tokens(question)
    steps = TOPICS[topic_id]['steps']
    scores = [len(question_tokens & tokens(' '.join([step['question'], *step['keywords'], step['starter']])))
              for step in steps]
    exact = next((index for index, step in enumerate(steps)
                  if normalized(question) == normalized(step['question'])), None)
    matched = exact if exact is not None else (max(range(len(steps)), key=scores.__getitem__)
                                               if max(scores, default=0) >= 2 else None)
    if matched is not None:
        hint = {**guidance(topic_id, matched), 'question': question}
    return hint, source


class LocalSpeechEngine:
    def __init__(self):
        self.asr = self.processor = None
        self.whisper_asr = None
        self.crottc_pronunciation = None
        self.bert_recovery = None
        from mdd.demo_inference import LocalInference
        from mdd.demo_model_registry import speech_device_policy
        self.phonemes = LocalInference(ROOT, threads=2, device=speech_device_policy('phoneme'))

    def load_asr(self):
        """Load the selected ASR once; no audio or inference is performed."""
        from mdd.demo_model_registry import speech_asr_selection
        selected = speech_asr_selection(ROOT)
        if selected.get('provider') == 'whisper':
            from scripts.whisper_asr import WhisperASR
            if getattr(self, 'whisper_asr', None) is None:
                self.whisper_asr = WhisperASR(ROOT, cpu_threads=2, device=selected.get('requested_device', 'cpu'))
            self.whisper_asr.load()
            self.asr = self.processor = None
            self.asr_backend = 'whisper'
            self.asr_model_id = selected['id']
            return
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
        self.asr_backend = 'wav2vec2'
        self.whisper_asr = None
        if self.asr is None or getattr(self, 'asr_identity', None) != selected['sha256']:
            path, expected = selected['path'], selected['sha256']
            if hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest() != expected:
                raise ValueError("ASR weights failed their checksum check")
            self.processor = Wav2Vec2Processor.from_pretrained(str(path), local_files_only=True)
            inference_config = Wav2Vec2Config.from_pretrained(str(path), local_files_only=True)
            inference_config.apply_spec_augment = False
            inference_config.mask_time_prob = inference_config.mask_feature_prob = 0.
            self.asr = Wav2Vec2ForCTC.from_pretrained(str(path), local_files_only=True,
                config=inference_config, attn_implementation="eager").cpu().eval().requires_grad_(False)
            self.asr_identity = expected
            self.asr_model_id = selected['id']

    def _phoneme_model(self):
        from mdd.demo_model_registry import pronunciation_selection
        selected = pronunciation_selection(ROOT, getattr(self, '_pronunciation_preference', None))
        if selected['provider'] == 'crottc-if':
            return {**selected, 'kind': 'perceived_phones', 'experimental': True,
                    'label': 'CROTTC-IF'}
        checkpoint = ROOT / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint"
        completed = json.loads((checkpoint.parent / "completed.json").read_text(encoding="utf-8"))
        if completed.get("status") != "completed":
            raise ValueError("The phoneme checkpoint is incomplete")
        return {"id": "cascade-B", "label": "wav2vec2 B", "kind": "phoneme",
                "experimental": False, "path": checkpoint}

    def _crottc_engine(self):
        from scripts.crottc_pronunciation import CROTTCPronunciation
        from mdd.demo_model_registry import speech_device_policy
        if getattr(self, 'crottc_pronunciation', None) is None:
            self.crottc_pronunciation = CROTTCPronunciation(ROOT, timeout=45., device=speech_device_policy('crottc'))
        return self.crottc_pronunciation

    def _load_pronunciation(self, cancelled=None):
        model = self._phoneme_model()
        if model.get('provider') == 'crottc-if':
            self._crottc_engine().load(cancelled)
        else:
            self.phonemes._get_bundle(model)
        return model

    def close(self):
        cancelled = getattr(self, '_pronunciation_cancelled', None)
        if cancelled is not None:
            cancelled.set()
        engine = getattr(self, 'crottc_pronunciation', None)
        if engine is not None:
            engine.close()
        engine = getattr(self, 'whisper_asr', None)
        if engine is not None:
            engine.close()

    def _prime_dictionaries(self, cancelled):
        from mdd import demo_reference, chat_pronunciation, pronunciation_assessment
        from mdd.phonetic_candidates import _phone_buckets
        path = (ROOT / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict').resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Dictionary is outside the workspace')
        info = path.stat()
        identity = (str(path), info.st_mtime_ns, info.st_size)
        loaders = (
            lambda: demo_reference._load(*identity),
            lambda: chat_pronunciation._load(*identity),
            lambda: pronunciation_assessment._read_dictionary(*identity, demo_reference.DICTIONARY_SHA256),
            lambda: _phone_buckets(*identity),
        )
        for load in loaders:
            if cancelled.is_set(): return
            if not load(): raise ValueError('The verified pronunciation dictionary is unavailable')

    def warmup(self, cancelled, qwen):
        """Load current models and verified lexicons without forwards or generation."""
        from mdd.demo_model_registry import recovery_selection
        timings = {}
        recovery = recovery_selection(ROOT)
        def load_recovery():
            if recovery['provider'] == 'bert' and getattr(self, 'asr_backend', 'wav2vec2') != 'whisper':
                from scripts.bert_transcript_recovery import BertTranscriptRecovery
                if self.bert_recovery is None:
                    self.bert_recovery = BertTranscriptRecovery(ROOT)
                self.bert_recovery._load(cancelled)
        stages = (
            ('qwen_load_sec', lambda: qwen.load('qwen-base')),
            ('asr_load_sec', self.load_asr),
            ('phoneme_load_sec', lambda: self._load_pronunciation(cancelled)),
            ('dictionary_load_sec', lambda: self._prime_dictionaries(cancelled)),
            ('recovery_load_sec', load_recovery),
        )
        for name, load in stages:
            if cancelled.is_set():
                return {'finish_reason': 'cancelled', 'status': 'cancelled', 'warmup_timings': timings}
            started = time.monotonic()
            load()
            timings[name] = round(time.monotonic() - started, 3)
        if cancelled.is_set():
            return {'finish_reason': 'cancelled', 'status': 'cancelled', 'warmup_timings': timings}
        return {'finish_reason': 'stop', 'status': 'ready', 'warmup_timings': timings,
                'asr_model_id': self.asr_model_id, 'phoneme_model_id': self._phoneme_model()['id'],
                'recovery_provider': recovery['provider'] if getattr(self, 'asr_backend', 'wav2vec2') != 'whisper' else None,
                'qwen_runtime': dict(qwen.runtime),
                'asr_runtime': ({'device': self.whisper_asr.device, 'compute_type': self.whisper_asr.compute_type,
                                 'requested_device': self.whisper_asr.requested_device,
                                 'fallback_reason': self.whisper_asr.fallback_reason}
                                if getattr(self, 'whisper_asr', None) is not None else {'device': 'cpu'}),
                'phoneme_runtime': (dict(self.crottc_pronunciation.runtime or {})
                                    if getattr(self, 'crottc_pronunciation', None) is not None
                                    else dict(getattr(self.phonemes, 'runtime_info', {}))),
                'load_only': True}

    def transcribe(self, waveform):
        # Timings from a previous recording must never label this recording.
        self.asr_word_intervals = []
        self.load_asr()
        if getattr(self, 'asr_backend', 'wav2vec2') == 'whisper':
            result = self.whisper_asr.transcribe(waveform)
            text = result['text'].strip()
            if not text:
                raise UnrecognizedSpeech('No usable English transcript was recognized')
            if len(text) > 1000:
                raise ValueError('No usable English transcript was recognized')
            self.asr_word_intervals = result['word_intervals']
            self.asr_word_timing = result.get('word_timing', 'whisper_native')
            self.asr_raw_transcript = result.get('raw_text', text)
            self.asr_runtime = {key: result[key] for key in
                ('model_id', 'revision', 'device', 'compute_type', 'requested_device', 'fallback_reason',
                 'word_timing', 'text_normalization') if key in result}
            return text
        import torch
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.asr(**inputs).logits
        ids = logits.argmax(-1)
        text = self.processor.batch_decode(ids)[0].strip().lower()
        text = re.sub(r"\bi\b", "I", text)
        text = text[:1].upper() + text[1:]
        if not text:
            raise UnrecognizedSpeech('No usable English transcript was recognized')
        if len(text) > 1000:
            raise ValueError("No usable English transcript was recognized")
        from mdd.phonetic_candidates import word_intervals
        vocabulary = self.processor.tokenizer.get_vocab()
        self.asr_word_intervals = word_intervals(ids[0].tolist(), {value: key for key, value in vocabulary.items()},
            self.processor.tokenizer.pad_token_id, self.processor.tokenizer.word_delimiter_token, len(waveform) / 16000)
        self.asr_word_timing = 'approximate_asr_ctc'
        self.asr_raw_transcript = text
        self.asr_runtime = {'model_id': self.asr_model_id, 'device': 'cpu', 'compute_type': 'float32',
                            'word_timing': self.asr_word_timing}
        return text

    def analyze(self, audio, reference_text=None):
        from mdd.demo_reference import dictionary_reference
        from mdd.chat_reference import reference_resolution
        from mdd.chat_pronunciation import target_words as reference_target_words
        from mdd.demo_feedback import ARPABET_IPA
        from mdd.demo_inference import canonical_comparison
        from mdd.acoustic_evidence import pack_evidence
        from mdd.chat_assessment import assessment_fields
        timings = {}
        stage_started = time.monotonic()
        provided_reference = reference_text is not None
        waveform, audio_info = self.phonemes._audio(audio)
        timings['audio_decode_sec'] = round(time.monotonic() - stage_started, 3)
        stage_started = time.monotonic()
        transcript = self.transcribe(waveform)
        timings['asr_sec'] = round(time.monotonic() - stage_started, 3)
        from mdd.speech_fluency import assess_fluency
        fluency = assess_fluency(waveform, transcript, getattr(self, 'asr_word_intervals', []))
        fluency['source'] = ('whisper_word_timestamps' if getattr(self, 'asr_backend', 'wav2vec2') == 'whisper'
                             else 'approximate_asr_ctc_word_timestamps')
        fluency['measured_transcript'] = transcript
        stage_started = time.monotonic()
        reference_text = reference_text or transcript
        reference, missing = dictionary_reference(ROOT, reference_text)
        target_words = reference_target_words(ROOT, reference_text, reference)
        phoneme_model = self._phoneme_model()
        timings['reference_setup_sec'] = round(time.monotonic() - stage_started, 3)
        stage_started = time.monotonic()
        resolution = reference_resolution(ROOT, reference_text)
        if provided_reference and reference is not None:
            resolution['status'] = 'provided'
        timings['reference_resolution_sec'] = round(time.monotonic() - stage_started, 3)
        result = {"transcript": transcript, "reference_text": reference_text, "audio": audio_info,
                'timings': timings, 'fluency': fluency,
                "reference_source": "reading_prompt" if provided_reference else "asr",
                "reference_resolution": resolution,
                "missing_words": missing, "phone_symbols": ARPABET_IPA,
                "reference": reference, 'target_words': target_words,
                "asr_model": getattr(self, 'asr_model_id', 'facebook/wav2vec2-base-960h'), "phoneme_model": phoneme_model['id'],
                'pronunciation_backend': phoneme_model.get('provider', 'wav2vec2-b'),
                "asr_backend": getattr(self, 'asr_backend', 'wav2vec2'),
                "asr_runtime": getattr(self, 'asr_runtime', {}),
                "asr_raw_transcript": getattr(self, 'asr_raw_transcript', transcript),
                }
        # Capture this recording's event and model instance before starting a
        # thread. A later turn must never replace its cancellation ownership.
        cancellation_event = getattr(self, '_pronunciation_cancelled', None)
        crottc_engine = self._crottc_engine() if phoneme_model.get('provider') == 'crottc-if' else None
        def score_sounds():
            scores, measured = {}, {}
            stage_started = time.monotonic()
            if phoneme_model.get('provider') == 'crottc-if':
                crottc_engine.load(cancellation_event)
                evidence = crottc_engine.infer(waveform, cancelled=cancellation_event)
            else:
                evidence = self.phonemes.posterior_evidence(phoneme_model, audio)
            measured['phoneme_sec'] = round(time.monotonic() - stage_started, 3)
            stage_started = time.monotonic()
            if not evidence['phonemes']:
                raise ValueError('The phoneme model did not recognize speech')
            raw_comparison = canonical_comparison(reference_text, reference, [item['phone'] for item in evidence['phonemes']])
            scores = assessment_fields(ROOT, reference_text, evidence, confirmed=provided_reference,
                                       raw_comparison=raw_comparison)
            if not scores.get('target_words'):
                scores['target_words'] = target_words
            measured['assessment_sec'] = round(time.monotonic() - stage_started, 3)
            stage_started = time.monotonic()
            from mdd.phonetic_candidates import sound_candidates
            if evidence.get('provider') == 'crottc-if':
                neighbours = {'status': 'unavailable', 'words': [], 'diagnostic_only': True,
                              'reason': 'perceived_phone_timestamps_unavailable', 'timing': 'unavailable'}
            else:
                neighbours = sound_candidates(ROOT, getattr(self, 'asr_word_intervals', []), evidence['phonemes'], transcript)
                neighbours['timing'] = getattr(self, 'asr_word_timing', 'approximate_asr_ctc')
            measured['phonetic_candidates_sec'] = round(time.monotonic() - stage_started, 3)
            return {**scores, 'phonemes': evidence['phonemes'], '_acoustic_evidence': pack_evidence(evidence),
                    'phonetic_candidates': neighbours, 'pronunciation_runtime': evidence.get('runtime', {}),
                    '_pronunciation_timings': measured}
        if (phoneme_model.get('provider') == 'crottc-if' and getattr(self, '_defer_pronunciation', False)
                and getattr(self, 'asr_backend', 'wav2vec2') == 'whisper'):
            from concurrent.futures import Future
            future = Future()
            self._pending_pronunciation = future
            def score_background():
                try:
                    future.set_result(score_sounds())
                except Exception as error:
                    future.set_exception(error)
            threading.Thread(target=score_background, daemon=True, name='chat-pronunciation').start()
            result.update(phonemes=[], comparison=None, assessment={'status': 'pending'},
                          sound_candidates={'status': 'pending', 'words': []})
        else:
            result.update(score_sounds())
            result['timings'].update(result.pop('_pronunciation_timings'))
        return result

    def _finish_pronunciation(self, result, cancelled, emit):
        """Join optional CPU analysis after publishing the GPU conversation reply."""
        future = getattr(self, '_pending_pronunciation', None)
        if future is None:
            return
        from concurrent.futures import TimeoutError as Pending
        emit({'type': 'stage', 'stage': 'analyzing'})
        try:
            while not cancelled.is_set():
                try:
                    fields = future.result(timeout=.1)
                    result.setdefault('timings', {}).update(fields.pop('_pronunciation_timings', {}))
                    result.update(fields)
                    return
                except Pending:
                    if future.done():
                        raise
                    continue
        except Exception as error:
            result.update(comparison=None, phonemes=[],
                assessment={'status': 'unavailable', 'reason': 'pronunciation_inference_failed'},
                sound_candidates={'status': 'unavailable', 'words': [], 'count': 0,
                                  'reason': 'pronunciation_inference_failed'},
                pronunciation_runtime={'status': 'unavailable', 'error_type': type(error).__name__})
        finally:
            self._pending_pronunciation = None

    def _use_bert_reference(self, result):
        """Compare the recovered sentence with the original, retained CTC evidence.

        The recovered text is a model inference, never a user confirmation.
        Rebuilding the reference changes neither audio nor acoustic predictions;
        the assessment keeps its existing abstention/calibration policy.
        """
        from scripts.transcript_normalization import valid_text
        interpretation = result.get('interpretation')
        if (result.get('reference_source') != 'asr' or not isinstance(interpretation, dict)
                or interpretation.get('provider') != 'bert'
                or interpretation.get('status') not in {'accepted', 'unchanged'}
                or interpretation.get('original') != result.get('transcript')
                or interpretation.get('reference_confirmed') is not False
                or not valid_text(interpretation.get('text'))):
            return result
        from mdd.acoustic_evidence import unpack_evidence
        from mdd.chat_assessment import assessment_fields
        from mdd.chat_pronunciation import target_words
        from mdd.chat_reference import reference_resolution
        from mdd.demo_inference import canonical_comparison
        from mdd.demo_reference import dictionary_reference
        started = time.monotonic()
        text = interpretation['text'].strip()
        evidence = unpack_evidence(result.get('_acoustic_evidence'))
        reference, missing = dictionary_reference(ROOT, text)
        comparison = canonical_comparison(text, reference, [item['phone'] for item in result['phonemes']])
        fields = assessment_fields(ROOT, text, evidence, confirmed=False, raw_comparison=comparison)
        if not fields.get('target_words'):
            fields['target_words'] = target_words(ROOT, text, reference)
        resolution = reference_resolution(ROOT, text)
        resolution.update(source='bert_inferred', reference_confirmed=False)
        if not missing:
            resolution['status'] = 'inferred'
        result.update(reference_text=text, reference_source='bert_inferred', reference=reference,
                      reference_resolution=resolution, missing_words=list(missing), **fields)
        result['sound_candidates'].update(source='bert_inferred', reference_confirmed=False, provisional=True)
        result['notice'] = 'Sound suggestions compare the original recording with the recovered English sentence.'
        interpretation['applied_as_pronunciation_reference'] = True
        result.setdefault('timings', {})['reference_reassessment_sec'] = round(time.monotonic() - started, 3)
        return result

    def run(self, request, cancelled, emit, qwen):
        import base64
        started = time.monotonic()
        emit({"type": "stage", "stage": "transcribing"})
        self._pending_pronunciation = None
        self._defer_pronunciation = True
        self._pronunciation_cancelled = cancelled
        self._pronunciation_preference = request.get('pronunciation_backend')
        if self._pronunciation_preference == 'crottc-if':
            # Release the comparison model before loading a second acoustic
            # encoder. Release unused CUDA cache for CROTTC's memory gate.
            self.phonemes._cached = self.phonemes._cache_key = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        elif self._pronunciation_preference == 'wav2vec2-b' and getattr(self, 'crottc_pronunciation', None) is not None:
            self.crottc_pronunciation.close()
        try:
            result = self.analyze(base64.b64decode(request["audio_base64"], validate=True), request.get("reference_text"))
        except UnrecognizedSpeech:
            if cancelled.is_set():
                return {'finish_reason': 'cancelled'}
            # No transcript is not a failed model load. Ask for a new recording
            # without inventing words, a pronunciation result or grammar advice.
            speech = {'transcript': '', 'reference_text': '', 'reference_source': 'asr',
                      'pronunciation_reference_ready': False, 'comparison': None, 'missing_words': [],
                      'transcript_clarity': {'needs_clarification': True, 'reason': 'unrecognized_speech',
                                             'interpreted': False, 'requires_confirmation': False},
                      'assessment': {'status': 'unavailable', 'reason': 'unrecognized_speech'},
                      'sound_candidates': {'status': 'unavailable', 'reason': 'unrecognized_speech', 'words': []},
                      'rewrite': {'status': 'not_needed'}, 'correction': {'status': 'not_needed'},
                      'coaching': {'status': 'unavailable', 'feedback': {}},
                      'spoken_response': REPEAT_REQUEST, 'reply_ready': True}
            emit({'type': 'stage', 'stage': 'generating'})
            emit({'type': 'chunk', 'text': REPEAT_REQUEST})
            emit({'type': 'speech', 'speech': speech})
            return {'finish_reason': 'stop', 'reply_source': 'clarification_fallback', 'reply_model_used': False,
                    'fallback_reason': 'unrecognized_speech', 'reply_generation_attempts': 0, 'reply_attempts': [],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0}, 'reply_model_id': 'qwen-base',
                    'input': 'audio', 'language': 'en-US', 'feedback_order': 'reply_first',
                    'latency_sec': round(time.monotonic() - started, 3)}
        finally:
            self._defer_pronunciation = False
        if request.get('recovery_enabled') is True and not request.get('reference_text'):
            # Publish progress without letting the UI use the raw target for
            # word marks or sentence voice while the selected recovery is pending.
            emit({'type': 'speech', 'speech': {**result, 'interpretation': {'status': 'pending'},
                                              'pronunciation_reference_ready': False}})
            emit({'type': 'stage', 'stage': 'correcting'})
            if request.get('recovery_provider') == 'bert':
                from scripts.bert_transcript_recovery import BertTranscriptRecovery
                if getattr(self, 'bert_recovery', None) is None:
                    self.bert_recovery = BertTranscriptRecovery(ROOT)
                result['interpretation'] = self.bert_recovery.recover(result['transcript'], cancelled,
                    phonetic_candidates=result.get('phonetic_candidates'), history=conversation_history(request))
            else:
                from scripts.transcript_recovery import recover
                result['interpretation'] = recover(result['transcript'], qwen, cancelled,
                    history=conversation_history(request))
            result.setdefault('timings', {})['interpretation_sec'] = result['interpretation'].get('latency_sec', 0.)
            if cancelled.is_set(): return {'finish_reason': 'cancelled'}
            if request.get('recovery_provider') == 'bert':
                result = self._use_bert_reference(result)
                # Word rescoring can leave fluent-looking fragments or split
                # names into dictionary syllables. Recover the meaning from the
                # whole ORIGINAL transcript before dialogue, independently of
                # the retained BERT pronunciation target and acoustic evidence.
                from scripts.sentence_recovery import needs_recovery, recover as recover_sentence
                if needs_recovery(result['transcript'], result['interpretation'],
                                  missing_words=result.get('missing_words', [])):
                    result['conversation_recovery'] = {'status': 'pending', 'original': result['transcript'],
                                                       'reference_confirmed': False}
                    emit({'type': 'speech', 'speech': {**result, 'pronunciation_reference_ready': False}})
                    result['conversation_recovery'] = recover_sentence(result['transcript'], qwen, cancelled,
                        history=conversation_history(request))
                    result['timings']['sentence_recovery_sec'] = result['conversation_recovery'].get('latency_sec', 0.)
                    if cancelled.is_set(): return {'finish_reason': 'cancelled'}
        elif request.get('normalization_enabled') is True and not request.get('reference_text'):
            from scripts.transcript_normalization import normalize
            # Keep all acoustic references/posteriors and the raw transcript.
            # This result is only a conservative text view for conversation.
            emit({'type': 'speech', 'speech': {**result, 'normalization': {'status': 'pending'}}})
            emit({'type': 'stage', 'stage': 'correcting'})
            result['normalization'] = normalize(result['transcript'], qwen, cancelled,
                                                 history=conversation_history(request))
            if cancelled.is_set(): return {'finish_reason': 'cancelled'}
        result['pronunciation_reference_ready'] = True
        try:
            return self._respond(request, result, cancelled, emit, qwen, started)
        finally:
            if getattr(self, '_pending_pronunciation', None) is not None:
                cancelled.set()
                self._pending_pronunciation = None

    def regenerate(self, request, cancelled, emit, qwen):
        """Respond to a server-confirmed edit using the existing recording result.

        This path never calls ASR, the acoustic model, or audio decoding. The
        service keeps the private original posterior in its own job snapshot.
        """
        started = time.monotonic()
        result = deepcopy(request.get('speech'))
        if (not isinstance(result, dict) or result.get('reference_source') != 'user_confirmed'
                or result.get('conversation_text') != request.get('dialogue_text')):
            raise ValueError('A server-confirmed conversation edit is required')
        for key in ('spoken_response', 'follow_up', 'guidance', 'coaching', 'rewrite', 'correction'):
            result.pop(key, None)
        view = {key: value for key, value in request.items() if key != 'speech'}
        view.update(reference_source='user_confirmed', reference_text=None, rewrite_enabled=True)
        return self._respond(view, result, cancelled, emit, qwen, started)

    def _respond(self, request, result, cancelled, emit, qwen, started):
        from scripts.l2arctic_plus_data import messages
        from scripts.conversation_context import prepare_dialogue_context
        requested_source = request.get('reference_source')
        if result.get('reference_source') != 'bert_inferred' or requested_source not in {None, 'asr'}:
            result["reference_source"] = (requested_source or result.get('reference_source')
                                          or ("reading_prompt" if request.get("reference_text") else "asr"))
        # Pronunciation provenance and dialogue provenance are independent: a
        # BERT-derived acoustic target is still an ASR interpretation for Qwen.
        dialogue_source = 'asr' if result['reference_source'] == 'bert_inferred' else result['reference_source']
        interpretation = result.get('interpretation') if dialogue_source == 'asr' else None
        sentence = result.get('conversation_recovery')
        if (dialogue_source == 'asr' and isinstance(sentence, dict)
                and sentence.get('status') in {'accepted', 'unchanged'}
                and sentence.get('original') == result['transcript']
                and sentence.get('reference_confirmed') is False):
            interpretation = sentence
        if isinstance(interpretation, dict) and interpretation.get('status') in {'accepted', 'unchanged'}:
            from mdd.chat_reference import reference_resolution
            # A copied malformed spelling is not evidence of recovered meaning.
            # This dictionary check routes clarification independently of the
            # selected pronunciation reference and never changes the evidence.
            interpretation['unresolved_words'] = reference_resolution(ROOT, interpretation['text']).get('missing_words', [])
        prepared = prepare_dialogue_context(request, result['transcript'], dialogue_text=request.get('dialogue_text'),
                                            source=dialogue_source, missing_words=result.get('missing_words', []),
                                            normalized_text=result.get('normalization', {}).get('text')
                                                if dialogue_source == 'asr' else None,
                                            interpretation=interpretation)
        request, dialogue_text = prepared['request'], prepared['text']
        result['transcript_clarity'] = prepared['clarity']
        needs_clarification = prepared['clarity']['needs_clarification']
        if prepared['request']['conversation_source'] in {'user_confirmed', 'asr_normalized', 'asr_interpreted'}:
            result['conversation_text'] = dialogue_text
        result['conversation_source'] = prepared['request']['conversation_source']
        result.pop('spoken_response', None)
        reference = result.pop("reference", None)
        topic_id = request.get('topic_id')
        turn_count = len(request.get('history', [])) // 2 + 1
        if topic_id:
            from mdd.speech_topics import guidance
            result['guidance'] = guidance(topic_id, turn_count)
        context = conversation_history(request)
        rewrite_enabled = request.get('rewrite_enabled') is True and not request.get('reference_text')
        result['reply_ready'] = False
        result['rewrite'] = {'status': 'pending' if rewrite_enabled else 'not_needed', 'mode': 'grammar_and_natural',
                             'original': dialogue_text}
        result['correction'] = {'status': 'not_needed' if request.get('reference_text') or rewrite_enabled else 'pending', 'mode': 'completion_only',
                                'original': dialogue_text}
        emit({"type": "speech", "speech": {**result, "coaching": {"status": "pending"}}})
        if cancelled.is_set():
            return {"finish_reason": "cancelled"}
        # Dialogue always uses base Qwen; the selected original/LoRA model is the pronunciation coach.
        reply_request = dialogue_request(request, dialogue_text)
        regenerated = request.get('operation') == 'regenerate_reference'
        from scripts.conversation_reply import ReplyWithoutEcho, REFERENCE_APOLOGY, MODEL_APOLOGIES
        regeneration_note = (
            '\nThe user has corrected a misunderstood transcript. Reply to this corrected meaning using the preceding conversation. '
            'Do not repeat or quote the user sentence. Do not apologize: the application supplies the apology separately.')
        if regenerated:
            reply_request['messages'][0]['content'] += regeneration_note
        chunks = []
        def reply_event(value):
            if value["type"] == "chunk" and not cancelled.is_set():
                first = not chunks
                chunks.append(value["text"])
                # Keep model-contribution checks on the body alone. An authored
                # apology must never make an empty/echo-only model reply pass.
                emit({**value, 'text': REFERENCE_APOLOGY + ' ' + value['text']}
                     if regenerated and first else value)
        from scripts.conversation_policy import GroundedReply
        state = reply_request.get('conversation_state')
        policy = GroundedReply(state, lambda text: reply_event({'type': 'chunk', 'text': text}),
                               transcript=dialogue_text, history=context, needs_clarification=needs_clarification)
        candidates = list(dict.fromkeys([dialogue_text, result['transcript']]))
        for candidate in tuple(candidates):
            candidates.extend(re.split(r'(?<=[.!?])\s+|\n+', candidate))
        # Replaying a previous welcome/question is another conversation loop.
        # Exclude short social acknowledgements that can naturally recur.
        for message in context:
            if message.get('role') == 'assistant':
                candidates.extend(sentence for sentence in re.split(r'(?<=[.!?])\s+|\n+', message['content'])
                                  if len(re.findall(r"\b[\w’']+\b", sentence)) >= 5)
        omitted_model_phrases = MODEL_APOLOGIES if regenerated else ()
        reply_stream = ReplyWithoutEcho(candidates, policy.push, omit_phrases=omitted_model_phrases,
                                        preserve_short_exchanges=not regenerated, embedded_echoes=regenerated)
        def model_reply_event(value):
            if value['type'] == 'chunk':
                reply_stream.push(value['text'])
        emit({"type": "stage", "stage": "generating"})
        if cancelled.is_set():
            return {'finish_reason': 'cancelled'}
        if needs_clarification:
            # Unresolved recognition is an application state, not a request for
            # another model guess. Publish the repeat prompt before coaching.
            reply_event({'type': 'chunk', 'text': REPEAT_REQUEST})
            reply_meta = {'finish_reason': 'stop', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
                          'reply_source': 'clarification_fallback', 'reply_model_used': False,
                          'fallback_reason': 'unclear_transcript', 'generation_latency_sec': 0.}
        else:
            reply_meta = qwen.generate(reply_request, cancelled, model_reply_event)
        if cancelled.is_set():
            return {"finish_reason": "cancelled"}
        reply_stream.finish()
        policy.finish()
        attempts = [] if needs_clarification else [{'raw_text': reply_stream.raw_text, 'omitted_echoes': list(reply_stream.removed),
                     'blocked_questions': list(policy.blocked),
                     'finish_reason': reply_meta.get('finish_reason')}]
        omitted = list(reply_stream.removed)
        blocked = list(policy.blocked)
        # An echoed statement is a generation failure, not a reason to make the
        # learner repeat a clear recording. Try one fresh contextual reply first.
        if not needs_clarification and ((not ''.join(chunks).strip() and omitted) or policy.needs_retry):
            retry_messages = [dict(message) for message in reply_request['messages']]
            if state:
                retry_messages[0]['content'] += '\nAsk only the missing detail specified above. Do not greet or ask for the order again.'
            else:
                retry_messages[0]['content'] = (
                    'You are talking with a friend in English. Use recent turns as context and reply directly to the last user message. '
                    'Answer the user\'s actual question first, briefly and honestly; do not claim personal experiences. If the meaning is clear, add one specific reaction or ask at most one open-ended follow-up grounded in a detail they shared. '
                    'Avoid generic questions and do not ask the same thing again. If some words are unclear, ask one short question about them. '
                    'Do not greet, thank, repeat their words, or restart earlier questions.')
                if regenerated:
                    retry_messages[0]['content'] += regeneration_note
            retry_request = {**reply_request, 'messages': retry_messages, 'max_tokens': 96, 'timeout': 35.}
            retry_policy = GroundedReply(state, lambda text: reply_event({'type': 'chunk', 'text': text}),
                                         transcript=dialogue_text, history=context, needs_clarification=needs_clarification)
            retry_stream = ReplyWithoutEcho([*candidates, *chunks], retry_policy.push,
                                           omit_phrases=omitted_model_phrases,
                                           preserve_short_exchanges=not regenerated, embedded_echoes=regenerated)
            try:
                retry_meta = qwen.generate(retry_request, cancelled, lambda value:
                    retry_stream.push(value['text']) if value['type'] == 'chunk' else None)
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                retry_stream.finish()
                retry_policy.finish()
                attempts.append({'raw_text': retry_stream.raw_text, 'omitted_echoes': list(retry_stream.removed),
                                 'blocked_questions': list(retry_policy.blocked),
                                 'finish_reason': retry_meta.get('finish_reason')})
                omitted.extend(retry_stream.removed)
                blocked.extend(retry_policy.blocked)
                reply_meta.update(retry_meta)
                if ''.join(chunks).strip(): reply_meta['reply_source'] = 'qwen_retry'
            except Exception:
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                attempts.append({'raw_text': retry_stream.raw_text, 'omitted_echoes': list(retry_stream.removed),
                                 'blocked_questions': list(retry_policy.blocked),
                                 'finish_reason': 'error'})
                omitted.extend(retry_stream.removed)
                blocked.extend(retry_policy.blocked)
        reply_meta.update(raw_reply_text=reply_stream.raw_text, omitted_leading_echoes=omitted,
                          omitted_echoes=omitted, blocked_questions=blocked, conversation_state=state,
                          reply_attempts=attempts, reply_generation_attempts=len(attempts))
        if not ''.join(chunks).strip():
            if needs_clarification:
                reply_event({'type': 'chunk', 'text': 'Could you say that another way? I could not understand the last part clearly.'})
                reply_meta['reply_source'] = 'clarification_fallback'
                reply_meta['fallback_reason'] = 'unclear_transcript_no_valid_model_reply'
            else:
                reply_event({'type': 'chunk', 'text': "I'm having trouble continuing this conversation right now."})
                reply_meta['reply_source'] = 'conversation_unavailable'
                reply_meta['fallback_reason'] = 'no_valid_model_reply'
        hint, source = follow_up(''.join(chunks), topic_id, turn_count, transcript=dialogue_text, history=context)
        if needs_clarification:
            source = 'application'
        result['follow_up'] = {'question': hint['question'], 'source': source}
        if topic_id: result['guidance'] = hint
        reply_meta['follow_up_source'] = source
        spoken = "".join(chunks).strip()
        if regenerated and spoken:
            spoken = REFERENCE_APOLOGY + ' ' + spoken
            reply_meta['reference_apology'] = {'text': REFERENCE_APOLOGY, 'source': 'application'}
        # The conversation voice reads the reply only. Sound estimates and tips
        # stay in the optional UI panel; individual examples require a click.
        result['spoken_response'] = spoken[:5000]
        result['reply_ready'] = True
        result['coaching'] = {'status': 'pending'}
        reply_latency = round(time.monotonic() - started, 3)
        emit({"type": "speech", "speech": result})
        # Publish the immutable reply before optional feedback uses the same
        # local worker. UI/voice can continue while the collapse is pending.
        correction_started = time.monotonic()
        if not request.get('reference_text') and not rewrite_enabled:
            from scripts.sentence_correction import correction_request, parse_correction
            emit({'type': 'stage', 'stage': 'correcting'})
            if cancelled.is_set(): return {'finish_reason': 'cancelled'}
            correction_chunks = []
            try:
                correction_meta = qwen.generate(correction_request(result['transcript'], history=context), cancelled,
                    lambda value: correction_chunks.append(value['text']) if value['type'] == 'chunk' else None)
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                result['correction'] = parse_correction(''.join(correction_chunks), correction_meta.get('finish_reason') == 'stop', result['transcript'], history=context)
            except Exception:
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                result['correction'] = {'status': 'unavailable', 'mode': 'completion_only', 'original': result['transcript'], 'model_id': 'qwen-base',
                    'source': 'qwen', 'message': 'No supported completion was available. Your wording is kept, and the conversation continues.'}
            emit({'type': 'speech', 'speech': {**result, 'coaching': {'status': 'pending'}}})
        correction_latency = round(time.monotonic() - correction_started, 3)
        if cancelled.is_set():
            return {'finish_reason': 'cancelled'}
        rewrite_started = time.monotonic()
        if rewrite_enabled and request.get('grammar_model_id') == 'qwen-grammar':
            from scripts.grammar_inference import rewrite
            emit({'type':'stage','stage':'correcting'})
            result['rewrite'] = rewrite(qwen,dialogue_text,cancelled,history=context)
            if cancelled.is_set(): return {'finish_reason':'cancelled'}
            emit({'type':'speech','speech':{**result,'coaching':{'status':'pending'}}})
        elif rewrite_enabled:
            from scripts.sentence_rewrite import rewrite_request, parse_rewrite
            emit({'type': 'stage', 'stage': 'correcting'})
            if cancelled.is_set(): return {'finish_reason': 'cancelled'}
            rewrite_chunks = []
            try:
                rewrite_meta = qwen.generate(rewrite_request(dialogue_text, history=context), cancelled,
                    lambda value: rewrite_chunks.append(value['text']) if value['type'] == 'chunk' else None)
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                result['rewrite'] = parse_rewrite(''.join(rewrite_chunks), rewrite_meta.get('finish_reason') == 'stop',
                                                  dialogue_text, history=context)
            except Exception:
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                result['rewrite'] = parse_rewrite('', False, dialogue_text)
            emit({'type': 'speech', 'speech': {**result, 'coaching': {'status': 'pending'}}})
        rewrite_latency = round(time.monotonic() - rewrite_started, 3)
        if cancelled.is_set():
            return {'finish_reason': 'cancelled'}
        self._finish_pronunciation(result, cancelled, emit)
        if cancelled.is_set():
            return {'finish_reason': 'cancelled'}
        coaching_started = time.monotonic()
        coaching = {'status': 'not_needed', 'feedback': {}, 'model_id': 'qwen-trained',
                    'message': 'No words were flagged by the acoustic comparison for Qwen to explain.'}
        comparison = result['comparison']
        has_differences = comparison is not None and any(word.get('status') == 'difference' for word in comparison.get('words', []))
        # Qwen LoRA explains only word differences already flagged by the
        # acoustic comparison. It does not replace or alter that evidence.
        if has_differences and reference is None:
            coaching = {'status': 'unavailable', 'feedback': {}, 'model_id': 'qwen-trained',
                        'message': 'An English pronunciation reference is unavailable, so Qwen LoRA was not asked to guess.'}
        elif has_differences:
            try:
                condition = native_condition(result['reference_text'], reference, [p['phone'] for p in result['phonemes']])
                if condition.get('input_alignment_status') == 'ok':
                    emit({'type': 'stage', 'stage': 'coaching'})
                    if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                    response = []
                    meta = qwen.generate({**request, 'model_id': 'qwen-trained', 'messages': messages(condition), 'max_tokens': 512, 'timeout': 110.}, cancelled,
                        lambda value: response.append(value['text']) if value['type'] == 'chunk' else None)
                    if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                    feedback = grounded_coaching(''.join(response), meta['finish_reason'] == 'stop', comparison)
                    coaching = {'status': 'ready', 'feedback': feedback, 'model_id': 'qwen-trained',
                                'message': 'Qwen LoRA suggestions grounded in the flagged sound differences.'}
                else:
                    coaching = {'status': 'unavailable', 'feedback': {}, 'model_id': 'qwen-trained',
                                'message': 'The word alignment was incomplete, so Qwen LoRA was not asked to guess.'}
            except Exception:
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                coaching = {'status': 'unavailable', 'feedback': {},
                            'model_id': 'qwen-trained',
                            'message': 'Qwen LoRA could not produce grounded feedback. The acoustic comparison remains available.'}
        coaching_latency = round(time.monotonic() - coaching_started, 3)
        result['coaching'] = coaching
        emit({'type': 'speech', 'speech': result})
        if cancelled.is_set():
            return {'finish_reason': 'cancelled'}
        return {**reply_meta, "reply_model_id": "qwen-base",
                "coaching_model_id": result.get('coaching', {}).get('model_id'),
                'reply_latency_sec': reply_latency, 'feedback_order': 'reply_first',
                'correction_model_id': 'qwen-base', 'correction_status': result['correction']['status'], 'correction_latency_sec': correction_latency,
                'completion_mode': 'not_needed' if rewrite_enabled else 'completion_only', 'dialogue_policy': 'natural_conversation',
                'rewrite_mode': 'grammar_and_natural' if rewrite_enabled else 'not_needed',
                'rewrite_status': result['rewrite']['status'], 'rewrite_latency_sec': rewrite_latency,
                'rewrite_model_id': result['rewrite'].get('model_id', 'qwen-base'),
                'coaching_latency_sec': coaching_latency,
                'conversation_source': result['conversation_source'],
                'pronunciation_reference_source': result['reference_source'],
                'interpretation_status': result.get('interpretation', {}).get('status', 'not_requested'),
                'interpretation_latency_sec': result.get('interpretation', {}).get('latency_sec', 0.),
                'conversation_recovery_status': result.get('conversation_recovery', {}).get('status', 'not_requested'),
                'conversation_recovery_latency_sec': result.get('conversation_recovery', {}).get('latency_sec', 0.),
                'regenerated_from_reference': request.get('operation') == 'regenerate_reference',
                'topic_id': topic_id,
                "input": "audio", "language": "en-US", "latency_sec": round(time.monotonic() - started, 3)}
