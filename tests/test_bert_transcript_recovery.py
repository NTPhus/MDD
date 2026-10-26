"""Finite, unconfirmed BERT repairs never become an audio reference."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from scripts import bert_transcript_recovery as bert


def lexicon(*words):
    return {word: ((('AA',), ('AA1',)),) for word in words}


class BertRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cancelled = threading.Event()
        self.engine = bert.BertTranscriptRecovery(ROOT)

    def recover(self, text, dictionary, scores=None, **kwargs):
        def score(text, start, stop, candidates, event, *, history=()):
            return {candidate: (scores or {}).get(candidate, -20.) for candidate in candidates}
        with patch.object(bert, '_dictionary', return_value=dictionary), \
             patch.object(self.engine, '_score_candidates', side_effect=score):
            return self.engine.recover(text, self.cancelled, **kwargs)

    def test_spelling_repair_is_unconfirmed_and_raw_recording_text_survives(self):
        raw = 'my fravorit drink is tea'
        phones = {'status': 'available', 'diagnostic_only': True, 'words': []}
        before = deepcopy(phones)
        result = self.recover(raw, lexicon('favorite', 'drink', 'tea'),
                              {'fravorit': -8., 'favorite': -2.}, phonetic_candidates=phones)
        self.assertEqual(result['text'], 'My favorite drink is tea')
        self.assertEqual(result['original'], raw)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['source'], 'bert_inferred')
        self.assertTrue(result['requires_confirmation'])
        self.assertFalse(result['reference_confirmed'])
        self.assertFalse(result['lexical_preservation'])
        self.assertFalse(result['score_is_acoustic_probability'])
        self.assertEqual(phones, before)
        self.assertEqual(result['provider'], 'bert')
        self.assertEqual(result['model_id'], bert.MODEL_ID)
        self.assertEqual(result['device'], 'cpu')
        self.assertGreater(result['evidence'][0]['margin_vs_original'], 1.)

    def test_real_learner_grammar_is_not_automatically_polished(self):
        raw = 'Yesterday I usually wakes up and I no brush teeth'
        with patch.object(bert, '_dictionary', return_value=lexicon(
                'yesterday', 'usually', 'wakes', 'up', 'brush', 'teeth')), \
             patch.object(self.engine, '_load') as load:
            result = self.engine.recover(raw, self.cancelled)
        self.assertEqual(result['status'], 'unchanged')
        self.assertEqual(result['text'], raw)
        self.assertFalse(result['requires_confirmation'])
        self.assertEqual(result['revision'], bert.REVISION)
        self.assertFalse(result['model_loaded'])
        load.assert_not_called()

    def test_dictionary_boundary_changes_require_a_contextual_preference(self):
        dictionary = lexicon('work', 'soft', 'ware', 'software', 'engineer', 'to', 'day', 'today', 'care')
        result = self.recover('I work as a soft ware engineer', dictionary,
                              {'soft ware': -7., 'software': -2.})
        self.assertEqual(result['text'], 'I work as a software engineer')
        self.assertEqual(result['status'], 'accepted')
        self.assertTrue(result['lexical_preservation'])
        # Existing lexical-preservation rules already allow "soft ware".
        self.assertFalse(result['requires_confirmation'])
        kept = self.recover('I go to day care', dictionary, {'to day': -1., 'today': -5.})
        self.assertEqual(kept['text'], 'I go to day care')
        self.assertEqual(kept['status'], 'unchanged')

    def test_no_number_negation_name_or_inflection_edit_enters_candidates(self):
        raw = "I do not want two coffees with Alice who wakes"
        dictionary = lexicon('want', 'coffees', 'with', 'alice', 'alicia', 'who', 'wakes', 'wake', 'three', 'tea')
        phone_rows = []
        for index, word in enumerate(raw.split()):
            phone_rows.append({'index': index, 'asr_word': word,
                'heard_phones': ['AA', 'BB', 'CC'],
                'candidates': [{'word': candidate} for candidate in dictionary]})
        proposals = bert._proposals(raw, dictionary,
            {'status': 'available', 'diagnostic_only': True, 'words': phone_rows})
        for proposal in proposals:
            for candidate in proposal['candidates'][1:]:
                changed = raw[:proposal['start']] + candidate['text'] + raw[proposal['end']:]
                self.assertIsNone(bert.recovery_reason(raw, changed))
                self.assertNotIn(proposal['original'].lower(), {'not', 'two', 'alice', 'wakes'})

    def test_known_word_needs_independent_sound_support_not_fluency_alone(self):
        dictionary = {'dreaming': ((('D', 'R', 'IY', 'M', 'IH', 'NG'), ()),),
                      'streaming': ((('S', 'T', 'R', 'IY', 'M', 'IH', 'NG'), ()),)}
        self.assertEqual(bert._proposals('I am dreaming', dictionary), [])
        phone_rows = {'status': 'available', 'diagnostic_only': True, 'words': [{
            'index': 2, 'asr_word': 'dreaming', 'heard_phones': ['S', 'T', 'R', 'IY', 'M', 'IH', 'NG'],
            'candidates': [{'word': 'streaming'}]}]}
        self.assertEqual(bert._proposals('I am dreaming', dictionary, phone_rows), [])
        proposals = bert._proposals('I am dreaming', dictionary, phone_rows,
                                    allow_known_word_edits=True)
        self.assertEqual(len(proposals), 1)
        self.assertTrue(proposals[0]['known_word'])
        selected, evidence = bert._choose(proposals[0], {'dreaming': -3., 'streaming': -1.5})
        self.assertIsNone(selected)  # Stronger 2-nat gate for dictionary words.
        self.assertEqual(evidence['decision'], 'insufficient_context_gain')
        selected, _ = bert._choose(proposals[0], {'dreaming': -8., 'streaming': -1.5})
        self.assertEqual(selected, 'streaming')

    def test_sound_rows_must_match_original_token_position_and_dictionary(self):
        dictionary = lexicon('dreaming', 'streaming')
        for value in [None, {'status': 'available', 'diagnostic_only': True, 'words': None},
            {'status': 'available', 'diagnostic_only': False, 'words': []},
            {'status': 'available', 'diagnostic_only': True, 'words': [
                {'index': True, 'asr_word': 'dreaming', 'heard_phones': []},
                {'index': 2, 'asr_word': 'different', 'heard_phones': [], 'candidates': []}]}]:
            with self.subTest(value=value):
                self.assertEqual(bert._proposals('I am dreaming', dictionary, value), [])

    def test_competing_candidates_abstain_instead_of_making_fluent_guess(self):
        result = self.recover('my fravorit drink', lexicon('favorite', 'favorites', 'drink'),
            {'fravorit': -10., 'favorite': -2., 'favorites': -2.1})
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['text'], 'My fravorit drink')
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(result['evidence'][0]['decision'], 'competing_candidates')

    def test_gain_gate_and_nonfinite_model_values_do_not_force_repairs(self):
        result = self.recover('my fravorit drink', lexicon('favorite', 'drink'),
            {'fravorit': -2., 'favorite': -1.})
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['text'], 'My fravorit drink')
        self.assertEqual(result['evidence'][0]['decision'], 'insufficient_context_gain')
        for bad in [float('nan'), float('inf'), None]:
            result = self.recover('my fravorit drink', lexicon('favorite', 'drink'),
                {'fravorit': -8., 'favorite': bad})
            self.assertEqual(result['text'], 'My fravorit drink')
            self.assertEqual(result['status'], 'ambiguous')

    def test_retrieval_is_bounded_even_when_every_unknown_has_no_candidate(self):
        raw = ' '.join('zzzz' + chr(97 + index // 26) + chr(97 + index % 26) for index in range(80))
        with patch.object(bert, '_spelling_candidates', return_value=[]) as retrieve:
            self.assertEqual(bert._proposals(raw, lexicon('favorite')), [])
        self.assertEqual(retrieve.call_count, bert.MAX_SPANS)

    def test_cancellation_during_retrieval_is_observed(self):
        self.cancelled.set()
        with self.assertRaises(InterruptedError):
            bert._spelling_candidates('favorite', lexicon('favorite'), self.cancelled)
        with self.assertRaises(InterruptedError):
            bert._proposals('my fravorit drink', lexicon('favorite'), cancelled=self.cancelled)

    def test_cancelled_missing_or_failed_models_leave_raw_text_available(self):
        self.cancelled.set()
        with patch.object(bert, '_dictionary') as dictionary:
            result = self.engine.recover('my fravorit', self.cancelled)
        dictionary.assert_not_called()
        self.assertEqual(result['reason'], 'cancelled')
        self.assertEqual(result['text'], 'My fravorit')
        self.cancelled.clear()
        with patch.object(bert, '_dictionary', return_value={}):
            self.assertEqual(self.engine.recover('my fravorit', self.cancelled)['reason'], 'verified_dictionary_unavailable')
        with patch.object(bert, '_dictionary', return_value=lexicon('favorite')), \
             patch.object(self.engine, '_score_candidates', side_effect=RuntimeError('PRIVATE')):
            result = self.engine.recover('my fravorit', self.cancelled)
        self.assertEqual(result['reason'], 'model_unavailable')
        self.assertEqual(result['text'], 'My fravorit')
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_late_cancellation_discards_otherwise_accepted_changes(self):
        def score(*args, **kwargs):
            self.cancelled.set()
            return {'fravorit': -8., 'favorite': -1.}
        with patch.object(bert, '_dictionary', return_value=lexicon('favorite')), \
             patch.object(self.engine, '_score_candidates', side_effect=score):
            result = self.engine.recover('my fravorit', self.cancelled)
        self.assertEqual(result['reason'], 'cancelled')
        self.assertEqual(result['text'], 'My fravorit')
        self.assertEqual(result['evidence'], [])

    def test_invalid_text_never_loads_model(self):
        with patch.object(self.engine, '_load') as load:
            result = self.engine.recover('x' * 1001, self.cancelled)
        self.assertEqual(result['reason'], 'invalid_text')
        load.assert_not_called()

    def test_manifest_and_model_files_are_checked_against_trusted_pins(self):
        content = b'known official model bytes'
        digest = hashlib.sha256(content).hexdigest()
        manifest = {'status': 'downloaded', 'model_id': bert.MODEL_ID,
            'revision': bert.REVISION, 'files': {'weights': {'sha256': digest, 'bytes': len(content)}}}
        files = {'weights': Mock(), 'setup_manifest.json': Mock()}
        files['weights'].stat.return_value = SimpleNamespace(st_size=len(content))
        files['weights'].open.side_effect = lambda mode: io.BytesIO(content)
        files['setup_manifest.json'].read_text.return_value = json.dumps(manifest)
        folder = MagicMock()
        folder.__truediv__.side_effect = lambda name: files[name]
        self.engine.model_dir = folder
        with patch.object(bert, 'MODEL_FILES', {'weights': digest}):
            self.engine._verify()
            files['weights'].open.side_effect = lambda mode: io.BytesIO(b'x' * len(content))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.engine._verify()


class BertMaskedScoringTests(unittest.TestCase):
    """A tiny fake network verifies masking without loading BERT weights."""

    def test_whole_span_masks_hide_all_candidate_pieces_and_group_lengths(self):
        import torch
        class Tokenizer:
            cls_token_id, sep_token_id, mask_token_id, pad_token_id, unk_token_id = 101, 102, 103, 0, 100
            mapping = {'odd': [4, 5], 'new': [6], 'pair': [6, 9], 'unknown': [100]}
            def encode(self, text, *, add_special_tokens):
                return self.mapping.get(text.strip(), [7] * len(text.split()))
        seen = []
        def network(**kwargs):
            ids = kwargs['input_ids']; seen.append(ids.clone())
            logits = torch.zeros((*ids.shape, 128))
            logits[:, :, 6] = 4.; logits[:, :, 9] = 2.
            logits[:, :, 4] = logits[:, :, 5] = 1.
            return SimpleNamespace(logits=logits)
        engine = bert.BertTranscriptRecovery(ROOT)
        engine.tokenizer, engine.model = Tokenizer(), network
        text = 'left odd right'
        scores = engine._score_candidates(text, 5, 8,
            ['odd', 'new', 'pair', 'unknown'], threading.Event())
        self.assertEqual(set(scores), {'odd', 'new', 'pair'})
        self.assertGreater(scores['new'], scores['pair'])
        self.assertGreater(scores['pair'], scores['odd'])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape[0], 2)  # One row per candidate length.
        for row in seen[0]:
            self.assertTrue(set(row.tolist()).issubset({0, 7, 101, 102, 103}))
        self.assertEqual(sorted((row == 103).sum().item() for row in seen[0]), [1, 2])
        self.assertTrue(all(math.isfinite(value) for value in scores.values()))

    def test_long_context_is_bounded_without_truncating_masked_target(self):
        import torch
        class Tokenizer:
            cls_token_id, sep_token_id, mask_token_id, pad_token_id, unk_token_id = 101, 102, 103, 0, 100
            def encode(self, text, *, add_special_tokens):
                return [7] * len(text.split())
        seen = []
        def network(**kwargs):
            ids = kwargs['input_ids']; seen.append(ids.clone())
            return SimpleNamespace(logits=torch.zeros((*ids.shape, 128)))
        engine = bert.BertTranscriptRecovery(ROOT)
        engine.tokenizer, engine.model = Tokenizer(), network
        text = 'left ' * 100 + 'odd' + ' right' * 100
        start = len('left ' * 100)
        engine._score_candidates(text, start, start + 3, ['odd'], threading.Event(),
            history=[{'role': 'assistant', 'content': 'earlier ' * 80}])
        self.assertLessEqual(seen[0].shape[1], bert.MAX_SEQUENCE)
        self.assertEqual((seen[0] == 103).sum().item(), 1)


if __name__ == '__main__':
    unittest.main()
