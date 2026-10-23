"""Local ASR + frozen phoneme B + Qwen English reply/coaching, inference only."""
from __future__ import annotations
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


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
    state = cafe_state(request, transcript)
    if state is not None:
        return {**request, 'model_id': 'qwen-base', 'generation_profile': 'conversation',
                'conversation_state': state,
                'messages': [{'role': 'system', 'content': state_note(state)},
                             *list(request.get('history', [])), {'role': 'user', 'content': transcript}],
                'max_tokens': 96, 'timeout': 45.}
    if request.get('needs_clarification') is True:
        system = ("Some words in the latest speech transcript are unclear. Ask ONE brief English clarification question. "
                  "Do not guess the intended words, complete or correct the sentence, or greet. "
                  "The example shows the style; respond to the actual last message.")
        examples = [{'role': 'user', 'content': 'The green flerm was narg.'},
                    {'role': 'assistant', 'content': 'Could you explain what you mean by the last part?'}]
    else:
        system = ("Continue the English conversation from the LAST user message. Use earlier turns for context. "
                  "Add a specific new reaction, answer, or relevant question. One or two short sentences; a question is optional. "
                  "Do not greet again, repeat earlier questions, restate their sentence, or correct grammar. "
                  "Follow a new subject. The first exchanges are style examples; actual conversation follows.")
        examples = [{'role': 'user', 'content': 'I repair bicycles, mostly old ones.'},
                    {'role': 'assistant', 'content': 'What problems do you run into most often?'},
                    {'role': 'user', 'content': 'The brakes can be hard to adjust.'},
                    {'role': 'assistant', 'content': 'Do you test them one step at a time to find the cause?'},
                    {'role': 'user', 'content': 'I finally fixed it.'},
                    {'role': 'assistant', 'content': 'That must be a relief after all that effort.'}]
    # The authored opening has already been spoken. Reinserting it before
    # every real exchange anchors the tiny model to an old welcome/question.
    history = list(request.get('history', [])) or conversation_history(request)
    topic_id = request.get('topic_id')
    if topic_id:
        topic = TOPICS[topic_id]
        system += f"\nInitial topic: {topic['title']}. Follow the current message even if the subject changes."
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
        self.bert_recovery = None
        from mdd.demo_inference import LocalInference
        self.phonemes = LocalInference(ROOT, threads=2)

    def transcribe(self, waveform):
        import torch
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
        from mdd.demo_model_registry import asr_selection
        selected = asr_selection(ROOT)
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
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.asr(**inputs).logits
        ids = logits.argmax(-1)
        text = self.processor.batch_decode(ids)[0].strip().lower()
        text = re.sub(r"\bi\b", "I", text)
        text = text[:1].upper() + text[1:]
        if not text or len(text) > 1000:
            raise ValueError("No usable English transcript was recognized")
        from mdd.phonetic_candidates import word_intervals
        vocabulary = self.processor.tokenizer.get_vocab()
        self.asr_word_intervals = word_intervals(ids[0].tolist(), {value: key for key, value in vocabulary.items()},
            self.processor.tokenizer.pad_token_id, self.processor.tokenizer.word_delimiter_token, len(waveform) / 16000)
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
        stage_started = time.monotonic()
        reference_text = reference_text or transcript
        reference, missing = dictionary_reference(ROOT, reference_text)
        target_words = reference_target_words(ROOT, reference_text, reference)
        checkpoint = ROOT / "runs/cascade-l2arctic-plus-seed42/B/best_checkpoint"
        completed = json.loads((checkpoint.parent / "completed.json").read_text(encoding="utf-8"))
        if completed.get("status") != "completed":
            raise ValueError("The phoneme checkpoint is incomplete")
        timings['reference_setup_sec'] = round(time.monotonic() - stage_started, 3)
        stage_started = time.monotonic()
        evidence = self.phonemes.posterior_evidence({"id": "cascade-B", "label": "wav2vec2 B", "kind": "phoneme",
                                                   "experimental": False, "path": checkpoint}, audio)
        timings['phoneme_sec'] = round(time.monotonic() - stage_started, 3)
        stage_started = time.monotonic()
        if not evidence["phonemes"]:
            raise ValueError("The phoneme model did not recognize speech")
        raw_comparison = canonical_comparison(reference_text, reference, [item['phone'] for item in evidence['phonemes']])
        assessment = assessment_fields(ROOT, reference_text, evidence, confirmed=provided_reference,
                                       raw_comparison=raw_comparison)
        if not assessment.get('target_words'):
            assessment['target_words'] = target_words
        timings['assessment_sec'] = round(time.monotonic() - stage_started, 3)
        stage_started = time.monotonic()
        resolution = reference_resolution(ROOT, reference_text)
        if provided_reference and reference is not None:
            resolution['status'] = 'provided'
        timings['reference_resolution_sec'] = round(time.monotonic() - stage_started, 3)
        stage_started = time.monotonic()
        from mdd.phonetic_candidates import sound_candidates
        neighbours = sound_candidates(ROOT, getattr(self, 'asr_word_intervals', []), evidence['phonemes'], transcript)
        timings['phonetic_candidates_sec'] = round(time.monotonic() - stage_started, 3)
        return {"transcript": transcript, "reference_text": reference_text, "audio": audio_info,
                'timings': timings,
                "reference_source": "reading_prompt" if provided_reference else "asr",
                "reference_resolution": resolution,
                "phonetic_candidates": neighbours,
                "missing_words": missing, "phone_symbols": ARPABET_IPA,
                "phonemes": evidence["phonemes"], "reference": reference,
                '_acoustic_evidence': pack_evidence(evidence), **assessment,
                "asr_model": getattr(self, 'asr_model_id', 'facebook/wav2vec2-base-960h'), "phoneme_model": "cascade-B",
                }

    def run(self, request, cancelled, emit, qwen):
        import base64
        started = time.monotonic()
        emit({"type": "stage", "stage": "transcribing"})
        result = self.analyze(base64.b64decode(request["audio_base64"], validate=True), request.get("reference_text"))
        if request.get('recovery_enabled') is True and not request.get('reference_text'):
            # A conversational interpretation is never an acoustic reference.
            # Keep raw ASR, posteriors and pronunciation assessment untouched.
            emit({'type': 'speech', 'speech': {**result, 'interpretation': {'status': 'pending'}}})
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
        elif request.get('normalization_enabled') is True and not request.get('reference_text'):
            from scripts.transcript_normalization import normalize
            # Keep all acoustic references/posteriors and the raw transcript.
            # This result is only a conservative text view for conversation.
            emit({'type': 'speech', 'speech': {**result, 'normalization': {'status': 'pending'}}})
            emit({'type': 'stage', 'stage': 'correcting'})
            result['normalization'] = normalize(result['transcript'], qwen, cancelled,
                                                 history=conversation_history(request))
            if cancelled.is_set(): return {'finish_reason': 'cancelled'}
        return self._respond(request, result, cancelled, emit, qwen, started)

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
        result["reference_source"] = request.get('reference_source') or ("reading_prompt" if request.get("reference_text") else "asr")
        interpretation = result.get('interpretation') if result['reference_source'] == 'asr' else None
        if isinstance(interpretation, dict) and interpretation.get('status') in {'accepted', 'unchanged'}:
            from mdd.chat_reference import reference_resolution
            # A copied malformed spelling is not evidence of recovered meaning.
            # This dictionary check only routes clarification; it never relabels audio.
            interpretation['unresolved_words'] = reference_resolution(ROOT, interpretation['text']).get('missing_words', [])
        prepared = prepare_dialogue_context(request, result['transcript'], dialogue_text=request.get('dialogue_text'),
                                            source=result['reference_source'], missing_words=result.get('missing_words', []),
                                            normalized_text=result.get('normalization', {}).get('text')
                                                if result['reference_source'] == 'asr' else None,
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
        reply_meta = qwen.generate(reply_request, cancelled, model_reply_event)
        if cancelled.is_set():
            return {"finish_reason": "cancelled"}
        reply_stream.finish()
        policy.finish()
        attempts = [{'raw_text': reply_stream.raw_text, 'omitted_echoes': list(reply_stream.removed),
                     'blocked_questions': list(policy.blocked),
                     'finish_reason': reply_meta.get('finish_reason')}]
        omitted = list(reply_stream.removed)
        blocked = list(policy.blocked)
        # An echoed statement is a generation failure, not a reason to make the
        # learner repeat a clear recording. Try one fresh contextual reply first.
        if (not ''.join(chunks).strip() and omitted) or policy.needs_retry:
            retry_messages = [dict(message) for message in reply_request['messages']]
            if state:
                retry_messages[0]['content'] += '\nAsk only the missing detail specified above. Do not greet or ask for the order again.'
            else:
                retry_messages[0]['content'] = (
                    'You are talking with a friend in English. Reply directly to the last user message. '
                    'If some words are unclear, ask one short question about what they mean. '
                    'Otherwise answer or ask a relevant follow-up. Do not greet, thank, repeat their words, or restart earlier questions.')
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
        coaching_started = time.monotonic()
        coaching = {'status': 'unavailable', 'feedback': {}, 'message': 'Review the available sound estimates below. You can edit the transcript if words were misheard.'}
        comparison = result['comparison']
        has_differences = comparison is not None and any(word.get('status') == 'difference' for word in comparison.get('words', []))
        # The experimental posterior feature validates word-level detection,
        # not a precise substitution diagnosis. Keep the legacy sound coach
        # limited to its original acoustic evidence condition.
        if has_differences and reference is not None and 'assessment' not in result:
            try:
                condition = native_condition(result['reference_text'], reference, [p['phone'] for p in result['phonemes']])
                if condition.get('input_alignment_status') == 'ok':
                    emit({'type': 'stage', 'stage': 'coaching'})
                    if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                    response = []
                    meta = qwen.generate({**request, 'messages': messages(condition), 'max_tokens': 512, 'timeout': 110.}, cancelled,
                        lambda value: response.append(value['text']) if value['type'] == 'chunk' else None)
                    if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                    feedback = grounded_coaching(''.join(response), meta['finish_reason'] == 'stop', comparison)
                    coaching = {'status': 'ready', 'feedback': feedback, 'message': 'AI practice suggestions based on estimated sound differences.'}
            except Exception:
                if cancelled.is_set(): return {'finish_reason': 'cancelled'}
                coaching = {'status': 'unavailable', 'feedback': {},
                            'message': 'The coach did not produce grounded feedback. Review the sound estimates below or try a shorter recording.'}
            coaching['model_id'] = request['model_id']
        coaching_latency = round(time.monotonic() - coaching_started, 3)
        result['coaching'] = coaching
        emit({'type': 'speech', 'speech': result})
        if cancelled.is_set():
            return {'finish_reason': 'cancelled'}
        return {**reply_meta, "reply_model_id": "qwen-base", "coaching_model_id": request["model_id"],
                'reply_latency_sec': reply_latency, 'feedback_order': 'reply_first',
                'correction_model_id': 'qwen-base', 'correction_status': result['correction']['status'], 'correction_latency_sec': correction_latency,
                'completion_mode': 'not_needed' if rewrite_enabled else 'completion_only', 'dialogue_policy': 'natural_conversation',
                'rewrite_mode': 'grammar_and_natural' if rewrite_enabled else 'not_needed',
                'rewrite_status': result['rewrite']['status'], 'rewrite_latency_sec': rewrite_latency,
                'rewrite_model_id': result['rewrite'].get('model_id', 'qwen-base'),
                'coaching_latency_sec': coaching_latency,
                'conversation_source': result['conversation_source'],
                'interpretation_status': result.get('interpretation', {}).get('status', 'not_requested'),
                'interpretation_latency_sec': result.get('interpretation', {}).get('latency_sec', 0.),
                'regenerated_from_reference': request.get('operation') == 'regenerate_reference',
                'topic_id': topic_id,
                "input": "audio", "language": "en-US", "latency_sec": round(time.monotonic() - started, 3)}
