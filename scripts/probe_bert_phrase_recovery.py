"""Offline experiment: infer short noisy ASR spans with the installed BERT.

No target sentence, phrase lookup table, audio or API is supplied to inference.
This is an experiment, not a replacement for the demo's production recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from scripts.bert_transcript_recovery import BertTranscriptRecovery, _dictionary
from scripts.transcript_normalization import WORDS, format_only
from scripts.transcript_recovery import FUNCTION_WORDS, NUMBER_WORDS, _negations

VERSION = 'bert_phrase_probe_v1'
BEAM = 16
MAX_SPANS = 4
MAX_ROUNDS = 2
PROTECTED_WORDS = FUNCTION_WORDS | frozenset(
    'to of in on at by for from with about into over under up down out off as so '
    'than then here there now yes okay ok hi oh please'.split())
PRONOUNS = frozenset('i me my mine you your yours he him his she her hers it its '
                     'we us our ours they them their theirs'.split())


def similar(left, right):
    # LCS handles repeated letters without greedy SequenceMatcher alignment.
    left, right = (re.sub('[^a-z]', '', value.lower()) for value in (left, right))
    row = [0] * (len(right) + 1)
    for a in left:
        previous = row
        row = [0]
        for index, b in enumerate(right):
            row.append(previous[index] + 1 if a == b else max(previous[index + 1], row[-1]))
    return 2 * row[-1] / (len(left) + len(right)) if left or right else 0.


def eligible(original, candidate, dictionary):
    """Bound lexical change; this cannot establish acoustic or semantic truth."""
    old, new = WORDS.findall(original.lower()), WORDS.findall(candidate.lower())
    if not old or not new or len(new) > 2 or similar(original, candidate) < .5:
        return False
    if any(word not in dictionary for word in new):
        return False
    if any(word in NUMBER_WORDS or any(c.isdigit() for c in word) for word in old + new):
        return False
    if _negations(old) != _negations(new):
        return False
    if [word for word in old if word in PRONOUNS] != [word for word in new if word in PRONOUNS]:
        return False
    if len(old) == 1 and old[0] in PROTECTED_WORDS:
        return False
    if len(old) == 2 and old[-1] in dictionary:
        return False
    return True


def spans(text, dictionary):
    words = list(WORDS.finditer(text))
    unknown = [i for i, word in enumerate(words) if word[0].lower() not in dictionary
               and re.fullmatch('[a-z]+', word[0].lower())
               and not (i and word[0].istitle())]
    if not unknown:
        return []
    result = []
    # Single-word repair first; reconsider preceding two-word spans next round.
    for i in unknown:
        if len(words[i][0]) >= 3:
            result.append((words[i].start(), words[i].end()))
    for i, word in enumerate(words):
        if (len(word[0]) == 2 and word[0].lower() in dictionary
                and word[0].lower() not in PROTECTED_WORDS | NUMBER_WORDS
                and not word[0][0].isupper()):
            result.append((word.start(), word.end()))
    if result:
        return list(dict.fromkeys(result))[:MAX_SPANS]
    for i in unknown:
        if i and not words[i - 1][0][0].isupper():
            result.append((words[i - 1].start(), words[i].end()))
    return list(dict.fromkeys(result))[:MAX_SPANS]


class PhraseProbe:
    def __init__(self, root=ROOT):
        self.bert = BertTranscriptRecovery(root)
        self.dictionary = _dictionary(root)
        self.cancelled = threading.Event()
        self.forwards = 0

    def logits(self, text):
        import torch
        tokenizer = self.bert.tokenizer
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=96)
        positions = (encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero().flatten()
        if not len(positions):
            return []
        with torch.inference_mode():
            output = self.bert.model(**encoded)
            self.forwards += 1
            return output.logits[0, positions].float().log_softmax(-1)

    def top_words(self, logs, limit):
        tokenizer = self.bert.tokenizer
        values, ids = logs.topk(min(256, logs.shape[-1]))
        result = []
        for value, token_id in zip(values.tolist(), ids.tolist()):
            word = tokenizer.convert_ids_to_tokens(token_id)
            if re.fullmatch('[a-z]+', word) and word in self.dictionary:
                result.append((word, value))
                if len(result) >= limit:
                    break
        return result

    def proposals(self, text, start, stop):
        mask = self.bert.tokenizer.mask_token
        original = text[start:stop]
        candidates = {}
        one = self.logits(text[:start] + mask + text[stop:])
        for word, value in self.top_words(one[0], 48):
            if eligible(original, word, self.dictionary):
                candidates[word] = value
        two = self.logits(text[:start] + mask + ' ' + mask + text[stop:])
        # Reverse beam: both words come from model vocabulary, not a phrase list.
        for second, second_score in self.top_words(two[1], BEAM):
            conditional = self.logits(text[:start] + mask + ' ' + second + text[stop:])
            for first, first_score in self.top_words(conditional[0], 32):
                phrase = first + ' ' + second
                if eligible(original, phrase, self.dictionary):
                    candidates[phrase] = (first_score + second_score) / 2
        return [word for word, _ in sorted(candidates.items(), key=lambda row: row[1], reverse=True)[:10]]

    def score(self, text, start, stop, candidate):
        """Mean whole-word pseudo log likelihood within the proposed span.

        Unlike independent masks, each masked word can attend to other candidate
        words. These scores are uncalibrated language preferences, not acoustics.
        """
        import torch
        tokenizer = self.bert.tokenizer
        full = text[:start] + candidate + text[stop:]
        encoded = tokenizer(full, return_tensors='pt', return_offsets_mapping=True,
                            truncation=True, max_length=96)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        logs = []
        for word in WORDS.finditer(candidate):
            left, right = start + word.start(), start + word.end()
            positions = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
            if not positions:
                return None
            ids = encoded['input_ids'].clone()
            ids[0, positions] = tokenizer.mask_token_id
            with torch.inference_mode():
                prediction = self.bert.model(input_ids=ids, attention_mask=encoded['attention_mask'])
                self.forwards += 1
                distribution = prediction.logits[0, positions].float().log_softmax(-1)
                targets = encoded['input_ids'][0, positions]
                logs.append(distribution[torch.arange(len(positions)), targets].mean().item())
        return sum(logs) / len(logs) if logs else None

    def recover(self, text):
        current, rounds = text, []
        for index in range(MAX_ROUNDS):
            evidence, edits = [], []
            targets = spans(current, self.dictionary)
            if not targets:
                break
            self.bert._load(self.cancelled)
            for start, stop in targets:
                if any(start < end and stop > begin for begin, end, _ in edits):
                    continue
                original = current[start:stop]
                options = list(dict.fromkeys([original] + self.proposals(current, start, stop)))
                scores = {value: self.score(current, start, stop, value) for value in options}
                ranked = sorted(((value, score) for value, score in scores.items() if score is not None),
                                key=lambda row: row[1], reverse=True)
                best, best_score = ranked[0] if ranked else (original, None)
                gain = best_score - scores[original] if best_score is not None and scores[original] is not None else 0.
                margin = best_score - ranked[1][1] if len(ranked) > 1 else 0.
                accepted = best != original and gain >= 1. and margin >= .35
                evidence.append({'original': original, 'start': start, 'end': stop,
                                 'candidates': [{'text': value, 'score': score} for value, score in ranked],
                                 'selected': best if accepted else None, 'gain': gain,
                                 'margin': margin, 'accepted': accepted})
                if accepted:
                    edits.append((start, stop, best))
            for start, stop, replacement in sorted(edits, reverse=True):
                current = current[:start] + replacement + current[stop:]
            rounds.append({'round': index + 1, 'evidence': evidence, 'text': current})
            if not edits:
                break
        return finish_result(text, current, rounds, self.dictionary)


def finish_result(original, proposal, rounds, dictionary):
    unresolved = [word for word in WORDS.findall(proposal) if word.lower() not in dictionary]
    # Partial fluency is insufficient: keep unclear proposals in diagnostics,
    # and retain the original sentence for the caller's repeat/clarify path.
    selected = original if unresolved else proposal
    status = ('ambiguous' if unresolved else 'unchanged'
              if format_only(selected) == format_only(original) else 'inferred')
    return {'original': original, 'text': format_only(selected), 'proposal_text': proposal,
            'rounds': rounds, 'status': status, 'unresolved_words': unresolved,
            'needs_clarification': bool(unresolved), 'reference_confirmed': False,
            'score_is_acoustic_probability': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < len(args.text) <= 500:
        parser.error('--text must contain 1–500 characters')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    probe = PhraseProbe()
    result = probe.recover(args.text)
    record = {'version': VERSION, 'model_id': 'google-bert/bert-base-uncased',
              'revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
              'input': args.text, 'target_supplied': False, 'audio_inference': False,
              'training': False, 'external_api': False, 'device': 'cpu',
              'forwards': probe.forwards, 'elapsed_sec': round(time.monotonic() - started, 3),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'result': result,
              'limits': {'rounds': MAX_ROUNDS, 'spans_per_round': MAX_SPANS, 'reverse_beam': BEAM},
              'production_enabled': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('elapsed_sec', 'forwards', 'production_enabled')}))
    print(json.dumps({key: result[key] for key in ('text', 'status', 'unresolved_words')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
