"""Uncalibrated candidates must be reviewable without claiming a sound is wrong."""
from copy import deepcopy
import unittest

from src.mdd.chat_diagnostics import sound_candidates


class ChatDiagnosticsTests(unittest.TestCase):
    def assessment(self):
        return {'status': 'ready', 'calibration': {'applied': False},
                'words': [{'text': 'with', 'status': 'uncertain', 'start_sec': 0., 'end_sec': .4},
                          {'text': 'me', 'status': 'uncertain', 'start_sec': .4, 'end_sec': .8}],
                'raw_events': [{'kind': 'substitution', 'word_index': 0, 'expected_phone': 'TH', 'observed_phone': 'T'}]}

    def test_unconfirmed_asr_shows_provisional_candidates_without_changing_uncertain_decisions(self):
        assessment = self.assessment(); before = deepcopy(assessment)
        result = sound_candidates(assessment, 'with me')
        self.assertEqual(result['status'], 'available')
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['words'][0]['text'], 'with')
        self.assertTrue(result['provisional'])
        self.assertFalse(result['reference_confirmed'])
        self.assertEqual(result['source'], 'asr')
        self.assertEqual(assessment, before)
        self.assertTrue(all(word['status'] == 'uncertain' for word in assessment['words']))
        self.assertIn('may mishear', result['message'])
        self.assertNotIn('correct', result['words'][0])

    def test_confirmed_candidates_are_separate_from_uncertain_decisions_and_keep_original_input(self):
        assessment = self.assessment(); before = deepcopy(assessment)
        result = sound_candidates(assessment, 'with me', reference_confirmed=True)
        self.assertEqual(assessment, before)
        self.assertTrue(result['diagnostic_only'])
        self.assertFalse(result['provisional'])
        self.assertEqual(result['source'], 'confirmed_reference')
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['words'][0]['text'], 'with')
        self.assertTrue(result['words'][0]['events'][0]['tentative'])
        self.assertTrue(all(word['status'] == 'uncertain' for word in assessment['words']))
        self.assertNotIn('correct', result['words'][0])

    def test_no_candidates_does_not_certify_correct_pronunciation(self):
        for confirmed in (False, True):
            with self.subTest(confirmed=confirmed):
                result = sound_candidates({**self.assessment(), 'raw_events': []}, 'with me', reference_confirmed=confirmed)
                self.assertEqual(result['status'], 'available')
                self.assertEqual(result['count'], 0)
                self.assertEqual(result['reason'], 'no_greedy_difference')
                self.assertEqual(result['provisional'], not confirmed)
                self.assertIn('does not certify', result['message'])

    def test_deletions_and_insertions_stay_tentative_not_validated_phone_diagnoses(self):
        assessment = self.assessment()
        assessment['raw_events'].extend([
            {'kind': 'deletion', 'word_index': 0, 'expected_phone': 'W', 'observed_phone': None},
            {'kind': 'insertion', 'word_index': 1, 'expected_phone': None, 'observed_phone': 'S'}])
        result = sound_candidates(assessment, 'with me', reference_confirmed=True)
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(result['words'][0]['events']), 2)
        self.assertTrue(all(event['tentative'] for word in result['words'] for event in word['events']))

    def test_invalid_reference_alignment_or_indices_never_shift_an_error_to_another_word(self):
        for change in ({'status': 'unavailable'}, {'words': [{'text': 'wrong'}]}, {'raw_events': [
                {'kind': 'substitution', 'word_index': 8}, {'kind': 'substitution', 'word_index': True}]}):
            with self.subTest(change=change):
                result = sound_candidates({**self.assessment(), **change}, 'with me', reference_confirmed=True)
                self.assertEqual(result['words'], [])

    def test_mismatched_word_text_is_unavailable_rather_than_a_no_difference_result(self):
        assessment = self.assessment()
        assessment['words'][0]['text'] = 'wrong'
        result = sound_candidates(assessment, 'with me', reference_confirmed=True)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reason'], 'reference_word_mismatch')
        self.assertNotIn('No sound difference', result['message'])

    def test_unknown_reference_and_alignment_failure_keep_explicit_reasons(self):
        for reason, missing in [('reference_contains_unknown_words', ['fravorit']),
                                ('ctc_alignment_impossible', []),
                                ('reference_phone_budget_exceeded', [])]:
            with self.subTest(reason=reason):
                assessment = {'status': 'unavailable', 'reason': reason, 'missing_words': missing}
                result = sound_candidates(assessment, 'my fravorit', reference_confirmed=True)
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['reason'], reason)
                self.assertEqual(result['missing_words'], missing)
                self.assertNotEqual(result['reason'], 'no_greedy_difference')

    def test_unknown_asr_reference_is_unavailable_not_hidden_behind_confirmation(self):
        result = sound_candidates({'status': 'unavailable', 'reason': 'reference_contains_unknown_words',
            'missing_words': ['fravorit']}, 'my fravorit')
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reason'], 'reference_contains_unknown_words')
        self.assertEqual(result['missing_words'], ['fravorit'])
        self.assertTrue(result['provisional'])
        self.assertEqual(result['source'], 'asr')
        self.assertNotIn('confirmed words', result['message'])

    def test_reference_confirmation_and_failed_calibration_are_independent(self):
        assessment = self.assessment()
        assessment['words'][0]['reason'] = 'calibration_unavailable'
        result = sound_candidates(assessment, 'with me', reference_confirmed=True)
        self.assertEqual(result['status'], 'available')
        self.assertEqual(result['reason'], 'tentative_differences')
        self.assertEqual(result['count'], 1)
        self.assertFalse(assessment['calibration']['applied'])


if __name__ == '__main__':
    unittest.main()
