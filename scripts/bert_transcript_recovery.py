"""Bounded masked-LM rescoring of independently supported ASR word candidates.

BERT never receives a reading answer and never generates a replacement sentence.
It compares the original span with a small set of dictionary spelling/sound
neighbours. Scores are language-model preferences, not acoustic probabilities or
proof of what somebody said. Accepted changes remain unconfirmed interpretations;
the caller must retain the original recording, transcript and sound reference.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from scripts.transcript_normalization import WORDS, format_only, signature, valid_text
from scripts.transcript_recovery import (
    FUNCTION_WORDS, NON_NAMES, NUMBER_WORDS, _negations, recovery_reason,
)

VERSION = 'bert_masked_recovery_v1'
MODEL_ID = 'google-bert/bert-base-uncased'
REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
MODEL_FILES = {
    'model.safetensors': '68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3',
    'config.json': '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20',
    'tokenizer.json': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
    'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
    'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
}
MAX_SPANS = 6
MAX_CANDIDATES = 9  # Includes the unchanged original.
MAX_SEQUENCE = 96
MAX_PIECES = 8
BATCH_SIZE = 2
# Initial conservative engineering settings, not thresholds calibrated on test
# labels. The evidence exposes their effect so a validation set can tune them.
MIN_GAIN = 1.0
MIN_MARGIN = .5
SPELLING_PENALTY = .25


def _cancelled(event):
    return event is not None and event.is_set()


def _base(original):
    text = format_only(original) if valid_text(original) else ''
    return {'version': VERSION, 'provider': 'bert', 'model_id': MODEL_ID,
            'revision': REVISION, 'device': 'cpu', 'model_loaded': False,
            'original': original, 'text': text, 'status': 'unavailable',
            'changed': text != original, 'requires_confirmation': False,
            'source': 'format_rules', 'reference_confirmed': False,
            'lexical_preservation': True, 'reason': 'model_unavailable',
            'score_kind': 'mean_whole_word_mask_log_score',
            'score_is_acoustic_probability': False, 'evidence': []}


def _plain(value):
    return value.casefold().replace('’', "'")


def _protected(word, index):
    key = _plain(word)
    return (key in FUNCTION_WORDS or key in NUMBER_WORDS or bool(_negations([key]))
            or any(character.isdigit() for character in key)
            or (index > 0 and word.istitle() and key not in NON_NAMES))


def _dictionary(root):
    # Reuse the pronunciation system's pinned, SHA-verified CMUdict loader.
    from mdd.pronunciation_assessment import _dictionary as load
    return load(root)


def _phone_rows(value, tokens):
    """Only use candidate rows bound to this exact raw ASR token position."""
    if (not isinstance(value, dict) or value.get('status') != 'available'
            or value.get('diagnostic_only') is not True):
        return {}
    result = {}
    for row in value.get('words', []):
        if not isinstance(row, dict):
            continue
        index = row.get('index')
        if (isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < len(tokens)
                or not isinstance(row.get('asr_word'), str)
                or _plain(row['asr_word']).strip('.,!?;:') != _plain(tokens[index].group())):
            continue
        phones = row.get('heard_phones')
        if (not isinstance(phones, list) or len(phones) > 24
                or not all(isinstance(phone, str) and re.fullmatch('[A-Z]+', phone)
                           for phone in phones)):
            continue
        result[index] = row
    return result


def _spelling_candidates(word, dictionary, cancelled=None):
    """A bounded same-initial edit search; sound neighbours cover other initials.

    CMUdict already contains many proper names. Contextual scores alone cannot
    make them confirmed English words, and the candidate set is deliberately
    limited rather than letting BERT invent whatever sounds fluent.
    """
    from mdd.chat_reference import _distance
    if not re.fullmatch('[a-z]{3,24}', word):
        return []
    limit = 1 if len(word) <= 4 else 2 if len(word) <= 7 else 3
    ranked = []
    for index, candidate in enumerate(dictionary):
        if index % 1024 == 0 and _cancelled(cancelled):
            raise InterruptedError('cancelled')
        if (candidate == word or not candidate.startswith(word[0])
                or abs(len(candidate) - len(word)) > limit
                or not re.fullmatch('[a-z]{3,24}', candidate)):
            continue
        distance = _distance(word, candidate, limit)
        if distance <= limit:
            similarity = SequenceMatcher(None, word, candidate, autojunk=False).ratio()
            ranked.append((distance, -similarity, candidate))
    return [word for _, _, word in sorted(ranked)[:5]]


def _phone_distance(heard, variants):
    from mdd.chat_reference import _distance
    if not heard or not variants:
        return None
    return min(_distance(heard, phones, max(len(heard), len(phones)))
               for phones, _ in variants)


def _distance_chars(before, after):
    from mdd.chat_reference import _distance
    left, right = _plain(before).replace(' ', ''), _plain(after).replace(' ', '')
    return _distance(left, right, max(len(left), len(right)))


def _candidate(original, text, *, origin, phone_distance=None):
    return {'text': text, 'origin': origin,
            'char_edit_distance': _distance_chars(original, text),
            'phone_edit_distance': phone_distance}


def _proposals(transcript, dictionary, phonetic_candidates=None, *, cancelled=None):
    tokens = list(WORDS.finditer(transcript))
    phone_rows = _phone_rows(phonetic_candidates, tokens)
    proposals, spelling_attempts = [], 0
    for index, token in enumerate(tokens):
        if _cancelled(cancelled):
            raise InterruptedError('cancelled')
        word, key = token.group(), _plain(token.group())
        if not _protected(word, index) and re.fullmatch('[a-z]{3,24}', key):
            row = phone_rows.get(index, {})
            heard = row.get('heard_phones', [])
            original_distance = _phone_distance(heard, dictionary.get(key))
            alternatives = {}
            if key not in dictionary:
                # Bound retrieval too, including unknown strings for which no
                # candidate is found. A long garbage transcript must not scan
                # the full lexicon once per word before the span limit applies.
                if spelling_attempts < MAX_SPANS:
                    spelling_attempts += 1
                    for candidate in _spelling_candidates(key, dictionary, cancelled):
                        alternatives[candidate] = _candidate(word, candidate, origin='dictionary_spelling')
                # Split only exactly preserved characters, never insert words.
                for position in range(2, len(key) - 1):
                    left, right = key[:position], key[position:]
                    if left in dictionary and right in dictionary:
                        candidate = f'{left} {right}'
                        alternatives[candidate] = _candidate(word, candidate, origin='dictionary_boundary')
            for item in row.get('candidates', [])[:5]:
                if not isinstance(item, dict):
                    continue
                candidate = item.get('word')
                if (not isinstance(candidate, str) or candidate == key
                        or candidate not in dictionary
                        or not re.fullmatch('[a-z]{3,24}', candidate)):
                    continue
                distance = _phone_distance(heard, dictionary.get(candidate))
                # Known words are not grammar-corrected because BERT prefers a
                # fluent alternative. They need a stronger independent phone
                # preference first; these phones are still noisy model output.
                if key in dictionary and (original_distance is None or distance is None
                                          or original_distance - distance < 2):
                    continue
                alternatives[candidate] = _candidate(word, candidate, origin='phonetic_dictionary',
                                                      phone_distance=distance)
            safe = []
            for candidate, item in alternatives.items():
                replaced = transcript[:token.start()] + candidate + transcript[token.end():]
                if recovery_reason(transcript, replaced) is None:
                    safe.append(item)
            if safe:
                # Preserve several spelling and sound alternatives when either
                # retrieval source returns many candidates.
                safe.sort(key=lambda item: (item['char_edit_distance'],
                          item['phone_edit_distance'] if item['phone_edit_distance'] is not None else 99,
                          item['text']))
                proposals.append({'start': token.start(), 'end': token.end(), 'index': index,
                    'original': word, 'known_word': key in dictionary,
                    'reason': 'phonetic_word' if key in dictionary else 'unknown_word',
                    'candidates': [_candidate(word, word, origin='original',
                        phone_distance=original_distance), *safe[:MAX_CANDIDATES - 1]]})
        if index + 1 < len(tokens):
            following = tokens[index + 1]
            # A dictionary join is a proposal, not an unconditional rule. In
            # particular "to day care" must not become "today care" merely
            # because the joined spelling exists.
            next_key = _plain(following.group())
            joined = key + next_key
            if (transcript[token.end():following.start()].isspace()
                    and re.fullmatch('[a-z]{4,24}', joined) and joined in dictionary
                    and not (index > 0 and word.istitle() and key not in NON_NAMES)
                    and not (following.group().istitle() and next_key not in NON_NAMES)):
                original = transcript[token.start():following.end()]
                replaced = transcript[:token.start()] + joined + transcript[following.end():]
                if recovery_reason(transcript, replaced) is None:
                    proposals.append({'start': token.start(), 'end': following.end(), 'index': index,
                        'original': original, 'known_word': False, 'reason': 'dictionary_boundary',
                        'candidates': [_candidate(original, original, origin='original'),
                                       _candidate(original, joined, origin='dictionary_boundary')]})
    priority = {'unknown_word': 0, 'dictionary_boundary': 1, 'phonetic_word': 2}
    return sorted(proposals, key=lambda row: (priority[row['reason']], row['start']))[:MAX_SPANS]


def _choose(proposal, scores):
    """Return transparent, conservative ranking evidence for one finite set."""
    ranked = []
    for candidate in proposal['candidates']:
        score = scores.get(candidate['text'])
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        selection = float(score) - SPELLING_PENALTY * candidate['char_edit_distance']
        ranked.append({**candidate, 'masked_lm_mean_log_score': round(float(score), 5),
                       'selection_score': round(selection, 5)})
    ranked.sort(key=lambda row: (-row['selection_score'], row['text']))
    original = next((row for row in ranked if row['origin'] == 'original'), None)
    evidence = {'index': proposal['index'], 'original': proposal['original'],
                'reason': proposal['reason'], 'candidates': ranked, 'selected': None,
                'accepted': False, 'decision': 'unscorable_candidates',
                'margin_vs_original': None, 'margin_vs_runnerup': None}
    if original is None or len(ranked) < 2:
        return None, evidence
    best = ranked[0]
    if best['origin'] == 'original':
        evidence['decision'] = 'original_preferred'
        return None, evidence
    gain = best['selection_score'] - original['selection_score']
    margin = best['selection_score'] - ranked[1]['selection_score']
    evidence.update(margin_vs_original=round(gain, 5), margin_vs_runnerup=round(margin, 5))
    minimum_gain = 2.0 if proposal['known_word'] else MIN_GAIN
    if gain < minimum_gain:
        evidence['decision'] = 'insufficient_context_gain'
    elif margin < MIN_MARGIN:
        evidence['decision'] = 'competing_candidates'
    else:
        evidence.update(selected=best['text'], accepted=True, decision='supported_context_preference')
        return best['text'], evidence
    return None, evidence


class BertTranscriptRecovery:
    """One lazy, CPU-only BERT instance per inference worker; no API/downloads."""

    def __init__(self, root, *, model_dir=None, max_edits=3):
        self.root = Path(root).resolve()
        self.model_dir = Path(model_dir) if model_dir is not None else self.root / 'models/pretrained/bert-base-uncased'
        self.max_edits = max(1, min(int(max_edits), 3))
        self.tokenizer = self.model = None
        self._load_lock = threading.Lock()

    def _verify(self, cancelled=None):
        manifest = json.loads((self.model_dir / 'setup_manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('status') not in {'downloaded', 'ready'}
                or manifest.get('model_id') != MODEL_ID or manifest.get('revision') != REVISION):
            raise ValueError('Pinned local BERT manifest is required')
        for name, expected_hash in MODEL_FILES.items():
            path = self.model_dir / name
            metadata = manifest.get('files', {}).get(name, {})
            if (metadata.get('sha256') != expected_hash
                    or path.stat().st_size != metadata.get('bytes')):
                raise ValueError('Pinned local BERT file is required')
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    if _cancelled(cancelled):
                        raise InterruptedError('cancelled')
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError('Local BERT file checksum mismatch')

    def _load(self, cancelled):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if _cancelled(cancelled):
                raise InterruptedError('cancelled')
            self._verify(cancelled)
            if _cancelled(cancelled):
                raise InterruptedError('cancelled')
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True,
                trust_remote_code=False, use_fast=True)
            model = AutoModelForMaskedLM.from_pretrained(str(self.model_dir),
                local_files_only=True, trust_remote_code=False, dtype=torch.float32,
                low_cpu_mem_usage=True, attn_implementation='eager').to('cpu')
            model.eval()
            model.requires_grad_(False)
            self.tokenizer, self.model = tokenizer, model

    def _score_candidates(self, text, start, stop, candidates, cancelled, *, history=()):
        """Whole-span masking, grouped by wordpiece count, without token leakage.

        All pieces of a candidate are hidden together. Each distinct candidate
        length uses one model row, so a nine-candidate set does not imply nine
        full-sentence forwards. Only the target logits are normalized. This is
        a bounded contextual score, not full-sentence PLL from the cited paper.
        """
        self._load(cancelled)
        if _cancelled(cancelled):
            raise InterruptedError('cancelled')
        import torch
        tokenizer = self.tokenizer
        prefix = tokenizer.encode(text[:start], add_special_tokens=False)
        suffix = tokenizer.encode(text[stop:], add_special_tokens=False)
        groups = {}
        candidate_ids = {}
        for candidate in candidates:
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if (not 1 <= len(pieces) <= MAX_PIECES
                    or any(piece in {tokenizer.unk_token_id, tokenizer.mask_token_id,
                                     tokenizer.cls_token_id, tokenizer.sep_token_id,
                                     tokenizer.pad_token_id} for piece in pieces)):
                continue
            candidate_ids[candidate] = pieces
            groups.setdefault(len(pieces), []).append(candidate)
        if not groups:
            return {}
        # BERT consumes ordinary background prose, not chat instructions. Keep
        # at most one previous actual turn, separated from the current sentence.
        previous = ''
        for row in reversed(list(history)[-2:]):
            if (isinstance(row, dict) and row.get('role') in {'user', 'assistant'}
                    and isinstance(row.get('content'), str)):
                previous = row['content'][-240:]
                break
        history_ids = tokenizer.encode(previous, add_special_tokens=False)[-20:] if previous else []
        rows = []
        for length in groups:
            available = MAX_SEQUENCE - length - 2
            # Prefer the local sentence to history; retain neighbours on both
            # sides even for long recordings instead of truncating the target.
            right = suffix[:available // 2]
            left = prefix[-(available - len(right)):]
            right = suffix[:available - len(left)]
            spare = available - len(left) - len(right)
            background = history_ids[-max(0, spare - 1):] if spare > 1 else []
            head = [tokenizer.cls_token_id]
            if background:
                head += background + [tokenizer.sep_token_id]
            head += left
            ids = head + [tokenizer.mask_token_id] * length + right + [tokenizer.sep_token_id]
            rows.append((length, ids, len(head)))
        scores = {}
        for offset in range(0, len(rows), BATCH_SIZE):
            if _cancelled(cancelled):
                raise InterruptedError('cancelled')
            batch = rows[offset:offset + BATCH_SIZE]
            width = max(len(ids) for _, ids, _ in batch)
            ids = torch.tensor([ids + [tokenizer.pad_token_id] * (width - len(ids))
                                for _, ids, _ in batch], dtype=torch.long, device='cpu')
            attention = (ids != tokenizer.pad_token_id).long()
            with torch.inference_mode():
                output = self.model(input_ids=ids, attention_mask=attention)
                for row_index, (length, _, target_start) in enumerate(batch):
                    # Normalize the masked positions only, keeping temporary
                    # allocations small on machines also running ASR and Qwen.
                    logs = output.logits[row_index, target_start:target_start + length].float().log_softmax(-1)
                    for candidate in groups[length]:
                        values = candidate_ids[candidate]
                        score = logs[torch.arange(length), torch.tensor(values)].mean().item()
                        scores[candidate] = float(score)
            del output, ids, attention
        if _cancelled(cancelled):
            raise InterruptedError('cancelled')
        return scores

    def recover(self, transcript, cancelled, *, phonetic_candidates=None, history=()):
        started = time.monotonic()
        result = _base(transcript)
        if not valid_text(transcript):
            result['reason'] = 'invalid_text'
        elif _cancelled(cancelled):
            result['reason'] = 'cancelled'
        else:
            try:
                dictionary = _dictionary(self.root)
                if not dictionary:
                    result['reason'] = 'verified_dictionary_unavailable'
                else:
                    proposals = _proposals(transcript, dictionary, phonetic_candidates, cancelled=cancelled)
                    edits, evidence = [], []
                    for proposal in proposals:
                        if _cancelled(cancelled):
                            raise InterruptedError('cancelled')
                        if any(proposal['start'] < end and proposal['end'] > start for start, end, _ in edits):
                            continue
                        scores = self._score_candidates(transcript, proposal['start'], proposal['end'],
                            [item['text'] for item in proposal['candidates']], cancelled, history=history)
                        selected, detail = _choose(proposal, scores)
                        evidence.append(detail)
                        if selected is not None:
                            edits.append((proposal['start'], proposal['end'], selected))
                        if len(edits) >= self.max_edits:
                            break
                    text = transcript
                    for start, end, candidate in sorted(edits, reverse=True):
                        text = text[:start] + candidate + text[end:]
                    # A final whole-utterance guard also catches interactions
                    # between independently safe, non-overlapping edits.
                    reason = recovery_reason(transcript, text)
                    if reason is not None:
                        result.update(reason=reason, evidence=evidence)
                    else:
                        text = format_only(text)
                        lexical_changed = signature(transcript) != signature(text)
                        unresolved_proposals = any(item['reason'] == 'unknown_word' and not item['accepted']
                                                   for item in evidence)
                        status = ('accepted' if edits else 'ambiguous' if unresolved_proposals else 'unchanged')
                        result.update(text=text, status=status, changed=text != transcript,
                            requires_confirmation=lexical_changed or status == 'ambiguous',
                            source='bert_inferred' if edits else 'format_rules',
                            lexical_preservation=not lexical_changed,
                            reason='candidate_ambiguity' if status == 'ambiguous' else None,
                            evidence=evidence, evaluated_spans=len(evidence),
                            candidate_limit=MAX_CANDIDATES, span_limit=MAX_SPANS,
                            max_edits=self.max_edits,
                            thresholds={'min_gain': MIN_GAIN, 'min_margin': MIN_MARGIN,
                                        'known_word_min_gain': 2.0, 'spelling_penalty': SPELLING_PENALTY})
            except InterruptedError:
                result = {**_base(transcript), 'reason': 'cancelled'}
            except Exception:
                # No exception details, prompts or path information leak into
                # the public speech job; raw ASR remains available to the UI.
                result['reason'] = 'cancelled' if _cancelled(cancelled) else 'model_unavailable'
        if _cancelled(cancelled):
            result = {**_base(transcript), 'reason': 'cancelled'}
        result['model_loaded'] = self.model is not None
        result['latency_sec'] = round(time.monotonic() - started, 3)
        return result
