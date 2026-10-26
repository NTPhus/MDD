"""Perceived sequence alignment, provenance and reference reassessment contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src')]
from mdd.phone_sequence_assessment import align_variants, assess_sequence
from mdd.acoustic_evidence import pack_evidence, unpack_evidence
from mdd.chat_assessment import assessment_fields


def evidence(phones):
    return {'provider': 'crottc-if', 'model_id': 'crottc-if-l2-arctic',
            'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64,
            'duration_sec': 2., 'phonemes': [{'phone': phone} for phone in phones],
            'word_timing': 'unavailable'}


class PhoneLatticeTests(unittest.TestCase):
    def test_legitimate_dictionary_variant_is_selected_globally(self):
        path, distance = align_variants([[['R', 'IY', 'D'], ['R', 'EH', 'D']], [['IH', 'T']]],
                                        ['R', 'EH', 'D', 'IH', 'T'])
        self.assertEqual(distance, 0)
        self.assertTrue(all(event['kind'] == 'match' for event in path))
        self.assertEqual({event['variant_index'] for event in path if event['word_index'] == 0}, {1})

    def test_repeated_perceived_phones_are_not_ctc_collapsed(self):
        path, distance = align_variants([[['B', 'UW', 'K']], [['K', 'IY', 'P']]],
                                        ['B', 'UW', 'K', 'K', 'IY', 'P'])
        self.assertEqual(distance, 0)
        self.assertEqual(len(path), 6)

    def test_edits_retain_phone_type_and_reference_word(self):
        path, distance = align_variants([[['K', 'AE', 'T']], [['IH', 'Z']]], ['K', 'EH', 'T', 'Z'])
        edits = [item for item in path if item['kind'] != 'match']
        self.assertEqual(distance, 2)
        self.assertEqual([(item['kind'], item['word_index']) for item in edits],
                         [('substitution', 0), ('deletion', 1)])
        self.assertEqual(edits[0]['observed_phone'], 'EH')

    def test_insertions_at_sentence_boundaries_are_bounded(self):
        path, distance = align_variants([[['K', 'AE', 'T']]], ['AH', 'K', 'AE', 'T', 'S'])
        self.assertEqual(distance, 2)
        edits = [item for item in path if item['kind'] == 'insertion']
        self.assertEqual(len(edits), 2)
        self.assertTrue(all(item['word_index'] == 0 for item in edits))

    def test_invalid_or_oversized_graph_is_rejected_before_allocation(self):
        for choices, observed in (([], ['T']), ([[[]]], ['T']), ([[['T']]], []),
                                   ([[['T']]] * 201, ['T'])):
            with self.subTest(choices=len(choices)), self.assertRaises(ValueError):
                align_variants(choices, observed)

    def test_lattice_optimum_matches_exhaustive_sentence_variants(self):
        from itertools import product
        from mdd.metrics import align_phones
        choices = [[['R', 'IY', 'D'], ['R', 'EH', 'D']], [['AH'], ['EY']], [['K', 'AE', 'T']]]
        for observed in (['R', 'EH', 'D', 'EY', 'K', 'AE', 'T'],
                         ['AH', 'R', 'IY', 'D', 'K', 'EH', 'T', 'S'],
                         ['R', 'IY', 'D', 'AH', 'K', 'K', 'AE', 'T']):
            expected = min(sum(item['kind'] != 'match' for item in align_phones(
                [phone for word in variant for phone in word], observed)) for variant in product(*choices))
            with self.subTest(observed=observed):
                _, actual = align_variants(choices, observed)
                self.assertEqual(actual, expected)


class SequenceEvidenceTests(unittest.TestCase):
    def test_storage_retains_official_phones_without_invented_posteriors_or_timing(self):
        original = evidence(['K', 'AE', 'T'])
        original['unresolved_labels'] = ['err']
        packed = pack_evidence(original)
        restored = unpack_evidence(packed)
        self.assertEqual(restored['phonemes'], original['phonemes'])
        self.assertEqual(restored['unresolved_labels'], ['err'])
        self.assertEqual(restored['provider'], 'crottc-if')
        self.assertNotIn('log_probs', packed)
        self.assertNotIn('data', packed)
        self.assertNotIn('start_sec', restored['phonemes'][0])

    def test_malformed_identity_duration_phone_and_time_are_rejected(self):
        changes = ({'checkpoint_sha256': 'wrong'}, {'duration_sec': 31},
                   {'phonemes': [{'phone': '<unk>'}]},
                   {'phonemes': [{'phone': 'K', 'start_sec': 1., 'end_sec': .5}]})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                pack_evidence({**evidence(['K']), **change})

    def variants(self):
        return [{'text': 'cat', 'available': True, 'variants': [
            {'phones': ['K', 'AE', 'T'], 'pronunciation_phones': ['K', 'AE1', 'T']}]}]

    def test_review_hints_never_claim_calibrated_errors_or_borrow_b_calibration(self):
        with patch('mdd.pronunciation_assessment.accepted_variants', return_value=self.variants()), \
                patch('mdd.chat_pronunciation.target_words', return_value=[]), \
                patch('mdd.chat_assessment.load_calibration') as old_calibration:
            fields = assessment_fields(Path('.'), 'cat', evidence(['K', 'EH', 'T']), confirmed=True)
        old_calibration.assert_not_called()
        self.assertEqual(fields['assessment']['status'], 'ready')
        self.assertFalse(fields['assessment']['calibration']['applied'])
        self.assertEqual(fields['comparison']['words'][0]['status'], 'uncertain')
        self.assertEqual(fields['comparison']['events'], [])
        self.assertEqual(fields['sound_candidates']['count'], 1)
        self.assertTrue(fields['sound_candidates']['diagnostic_only'])
        self.assertEqual(fields['sound_candidates']['words'][0]['events'][0]['source'],
                         'crottc_perceived_sequence_edit')

    def test_unknown_labels_and_unknown_reference_abstain(self):
        for variants, changes in ((self.variants(), {'unresolved_labels': ['err']}),
                                  ([{'text': 'cat', 'available': False, 'variants': []}], {})):
            with self.subTest(changes=changes), \
                    patch('mdd.pronunciation_assessment.accepted_variants', return_value=variants):
                fields = assess_sequence(Path('.'), 'cat', {**evidence(['K', 'AE', 'T']), **changes})
            self.assertEqual(fields['assessment']['status'], 'unavailable')
            self.assertEqual(fields['sound_candidates']['count'], 0)

    def test_reassessment_changes_reference_only_and_keeps_fluency_and_model(self):
        from mdd.chat_reference import reassess_reference
        acoustic = evidence(['K', 'AE', 'T'])
        speech = {'transcript': 'cat', 'phonemes': acoustic['phonemes'],
                  '_acoustic_evidence': pack_evidence(acoustic),
                  'phoneme_model': 'crottc-if-l2-arctic', 'fluency': {'status': 'ready'},
                  'assessment': {'method': 'crottc_if_variant_sequence_review_v1'}}
        before = deepcopy(speech)
        with patch('mdd.chat_reference.demo_reference.dictionary_reference', return_value=(
                {'prompt_text': 'cat', 'canonical_phones': ['K', 'AE', 'T'], 'canonical_word_indices': [0, 0, 0]}, [])), \
                patch('mdd.pronunciation_assessment.accepted_variants', return_value=self.variants()), \
                patch('mdd.chat_pronunciation.target_words', return_value=[]):
            changed = reassess_reference(Path('.'), speech, 'cat')
        self.assertEqual(changed['reference_source'], 'user_confirmed')
        self.assertEqual(changed['_acoustic_evidence'], before['_acoustic_evidence'])
        self.assertEqual(changed['fluency'], before['fluency'])
        self.assertEqual(changed['phoneme_model'], before['phoneme_model'])
        self.assertEqual(speech, before)


if __name__ == '__main__':
    unittest.main()
