"""Synthetic posterior evidence exercises CTC topology and safe abstention."""
from copy import deepcopy
import hashlib
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.mdd import demo_reference
from src.mdd.pronunciation_assessment import (METHOD, SCHEMA, _read_dictionary,
    accepted_variants, assess, score_words)


LEXICON = b'''the DH AH0
the(2) DH IY0
the(3) DH IY1
read R EH1 D
read(2) R IY1 D
wind W IH1 N D
wind(2) W AY1 N D
up AH1 P
aa AA1 AA0
a AH0
x AA1
x(2) IY1 AA0
join AA1 P
join(2) IY1 T
'''
PHONES = ['[PAD]', 'DH', 'AH', 'IY', 'R', 'EH', 'D', 'W', 'IH', 'N', 'AY', 'AA', 'P', 'B', 'T']


class PronunciationAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict'
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(LEXICON)
        checksum = patch.object(demo_reference, 'DICTIONARY_SHA256', hashlib.sha256(LEXICON).hexdigest())
        checksum.start()
        self.addCleanup(checksum.stop)
        self.addCleanup(_read_dictionary.cache_clear)

    def evidence(self, phones, *, frames=None):
        if frames is None:
            frames = ['[PAD]']
            for phone in phones:
                frames.extend([phone, phone, '[PAD]'])
        probabilities = np.full((len(frames), len(PHONES)), .01 / (len(PHONES) - 1))
        for index, phone in enumerate(frames):
            probabilities[index, PHONES.index(phone)] = .99
        return {'log_probs': np.log(probabilities), 'id_to_phone': dict(enumerate(PHONES)), 'blank_id': 0,
                'duration_sec': len(frames) / 50., 'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64}

    def policy(self, **changes):
        return {'schema': SCHEMA, 'schema_version': 1, 'method': METHOD, 'verified': True,
                'eligible': True, 'status': 'eligible', 'validation': {'status': 'verified', 'split': 'calibration'},
                'accept_threshold': -1., 'error_threshold': 1., 'checkpoint_sha256': 'a' * 64,
                'vocabulary_sha256': 'b' * 64, 'dictionary_sha256': demo_reference.DICTIONARY_SHA256, **changes}

    def test_all_verified_variants_preserve_first_and_deduplicate_stress_only_variants(self):
        variants = accepted_variants(self.root, '“THE,” read wind missing')
        self.assertEqual(variants[0]['text'], '“THE,”')
        self.assertEqual(variants[0]['variants'], [
            {'phones': ['DH', 'AH'], 'pronunciation_phones': ['DH', 'AH0']},
            {'phones': ['DH', 'IY'], 'pronunciation_phones': ['DH', 'IY0']}])
        self.assertEqual(len(variants[1]['variants']), 2)
        self.assertEqual(len(variants[2]['variants']), 2)
        self.assertFalse(variants[3]['available'])
        self.assertEqual(variants[3]['variants'], [])

    def test_legitimate_nonprimary_variants_are_selected_by_acoustics_not_flagged_as_edits(self):
        evidence = self.evidence('DH IY R IY D W AY N D'.split())
        result = score_words(self.root, 'the read wind', evidence)
        self.assertEqual(result['status'], 'ready', result)
        self.assertEqual([word['dictionary_variant_index'] for word in result['target_words']], [1, 1, 1])
        self.assertEqual(result['raw_events'], [])
        self.assertTrue(result['alignment']['variant_compatible'])
        self.assertTrue(all(word['score_eligible'] and word['error_score'] < -1 for word in result['words']))
        self.assertEqual([word['status'] for word in result['words']], ['uncertain'] * 3)

    def test_reference_confirmed_and_verified_calibration_are_both_required(self):
        evidence = self.evidence(['AH', 'P'])
        for calibration, confirmed, expected in ((None, True, 'uncertain'), (self.policy(verified=False), True, 'uncertain'),
                                                (self.policy(), False, 'uncertain'), (self.policy(), True, 'match')):
            with self.subTest(confirmed=confirmed, calibration=calibration):
                result = assess(self.root, 'up', evidence, calibration, confirmed)
                self.assertEqual(result['words'][0]['status'], expected)

    def test_competitor_margin_supplies_continuous_evidence_and_calibrated_difference(self):
        good = assess(self.root, 'up', self.evidence(['AH', 'P']), self.policy(), True)
        bad = assess(self.root, 'up', self.evidence(['AH', 'B']), self.policy(), True)
        self.assertEqual(bad['status'], 'ready', bad)
        self.assertTrue(bad['words'][0]['score_eligible'])
        self.assertGreater(bad['words'][0]['error_score'], good['words'][0]['error_score'])
        self.assertGreater(bad['words'][0]['error_score'], 1.)
        self.assertEqual(bad['words'][0]['status'], 'difference')
        self.assertEqual(bad['comparison']['events'][0]['expected_phone'], 'P')
        self.assertEqual(bad['comparison']['events'][0]['observed_phone'], 'B')
        self.assertTrue(bad['comparison']['events'][0]['tentative'])

    def test_uncalibrated_hard_substitution_is_uncertain_and_published_events_empty(self):
        result = assess(self.root, 'up', self.evidence(['AH', 'B']), reference_confirmed=True)
        self.assertEqual(result['words'][0]['status'], 'uncertain')
        self.assertTrue(result['raw_events'])
        self.assertEqual(result['comparison']['events'], [])
        self.assertEqual(result['comparison']['counts']['substitution'], 0)

    def test_calibration_identity_or_eligibility_failure_cannot_make_claims(self):
        evidence = self.evidence(['AH', 'B'])
        for changes in ({'eligible': False}, {'status': 'ineligible'}, {'checkpoint_sha256': 'c' * 64},
                        {'vocabulary_sha256': 'c' * 64}, {'dictionary_sha256': 'c' * 64},
                        {'accept_threshold': 2., 'error_threshold': 1.}, {'error_threshold': float('nan')},
                        {'validation': {'status': 'failed', 'split': 'calibration'}}):
            with self.subTest(changes=changes):
                result = assess(self.root, 'up', evidence, self.policy(**changes), True)
                self.assertFalse(result['calibration']['applied'])
                self.assertEqual(result['words'][0]['status'], 'uncertain')

    def test_calibration_abstention_band_is_preserved(self):
        evidence = self.evidence(['AH', 'B'])
        score = score_words(self.root, 'up', evidence)['words'][0]['error_score']
        result = assess(self.root, 'up', evidence, self.policy(accept_threshold=score - 1, error_threshold=score + 1), True)
        self.assertEqual(result['words'][0]['status'], 'uncertain')
        self.assertEqual(result['words'][0]['reason'], 'calibration_abstention_band')

    def test_ctc_repeated_phone_and_word_boundary_require_a_blank(self):
        for text, phone in (('aa', 'AA'), ('a a', 'AH')):
            with self.subTest(text=text):
                impossible = assess(self.root, text, self.evidence([], frames=[phone, phone]))
                self.assertEqual(impossible['status'], 'unavailable')
                self.assertEqual(impossible['reason'], 'ctc_alignment_impossible')
                possible = assess(self.root, text, self.evidence([], frames=[phone, '[PAD]', phone]))
                self.assertEqual(possible['status'], 'ready', possible)
                self.assertTrue(possible['alignment']['variant_compatible'])

    def test_viterbi_score_equals_exhaustive_ctc_path_max_on_tiny_variant_lattice(self):
        probabilities = np.array([[.3, .6, .1], [.5, .2, .3], [.1, .7, .2]])
        evidence = {'log_probs': np.log(probabilities), 'id_to_phone': {'0': '[PAD]', '1': 'AA', '2': 'IY'},
                    'blank_id': 0, 'duration_sec': .06}
        expected = -np.inf
        for sequence in product(range(3), repeat=3):
            decoded, previous = [], None
            for token in sequence:
                if token != previous and token:
                    decoded.append(token)
                previous = token
            if tuple(decoded) in ((1,), (2, 1)):
                expected = max(expected, sum(evidence['log_probs'][index, token] for index, token in enumerate(sequence)))
        result = assess(self.root, 'x', evidence)
        self.assertEqual(result['status'], 'ready', result)
        self.assertAlmostEqual(result['alignment']['ctc_viterbi_log_prob'], expected)

    def test_variant_branches_cannot_be_spliced_into_an_invalid_pronunciation(self):
        result = assess(self.root, 'join', self.evidence(['AA', 'T']))
        self.assertIn(result['target_words'][0]['phones'], [['AA', 'P'], ['IY', 'T']])
        self.assertNotEqual(result['target_words'][0]['phones'], ['AA', 'T'])

    def test_insertions_deletions_and_all_blank_audio_are_explicit_abstentions(self):
        for phones in (['AH'], ['AH', 'T', 'P'], []):
            with self.subTest(phones=phones):
                evidence = self.evidence(phones) if phones else self.evidence([], frames=['[PAD]'] * 12)
                result = assess(self.root, 'up', evidence, self.policy(), True)
                self.assertEqual(result['status'], 'ready', result)
                self.assertFalse(result['words'][0]['score_eligible'])
                self.assertEqual(result['words'][0]['status'], 'uncertain')
                self.assertEqual(result['comparison']['events'], [])

    def test_unknown_words_or_unverified_dictionary_do_not_trigger_g2p_or_errors(self):
        result = assess(self.root, 'fravorit', self.evidence(['AH', 'P']))
        self.assertEqual(result['reason'], 'reference_contains_unknown_words')
        self.assertEqual(result['words'][0]['status'], 'unavailable')
        self.path.write_bytes(b'up MADEUP\n')
        result = assess(self.root, 'up', self.evidence(['AH', 'P']))
        self.assertEqual(result['reason'], 'verified_dictionary_unavailable')
        self.assertEqual(result['comparison']['events'], [])

    def test_unrepresented_legitimate_dictionary_variant_prevents_word_decision(self):
        evidence = self.evidence(['DH', 'AH'])
        # Replacing one vocabulary label makes a legitimate CMU variant unusable.
        evidence['id_to_phone'][PHONES.index('IY')] = '[OTHER]'
        result = assess(self.root, 'the', evidence, self.policy(), True)
        self.assertEqual(result['status'], 'ready', result)
        self.assertFalse(result['words'][0]['score_eligible'])
        self.assertEqual(result['words'][0]['reason'], 'incomplete_dictionary_variant_coverage')

    def test_invalid_or_over_budget_evidence_abstains_without_exceptions(self):
        evidence = self.evidence(['AH', 'P'])
        for value in (None, {**evidence, 'log_probs': [[0., 0.]]},
                      {**evidence, 'log_probs': np.zeros((2001, len(PHONES)))},
                      {**evidence, 'log_probs': [[float('nan')] * len(PHONES)]},
                      {**evidence, 'blank_id': True}, {**evidence, 'duration_sec': -1}):
            result = assess(self.root, 'up', value)
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(result['comparison']['events'], [])
        result = assess(self.root, 'aa ' * 151, evidence)
        self.assertEqual(result['status'], 'unavailable')

    def test_evidence_calibration_and_cached_reference_are_not_mutated(self):
        evidence, policy = self.evidence(['AH', 'P']), self.policy()
        before = deepcopy(evidence)
        policy_before = deepcopy(policy)
        result = assess(self.root, 'up', evidence, policy, True)
        np.testing.assert_array_equal(evidence['log_probs'], before['log_probs'])
        self.assertEqual(evidence['id_to_phone'], before['id_to_phone'])
        self.assertEqual(policy, policy_before)
        result['target_words'][0]['phones'][0] = 'B'
        self.assertEqual(accepted_variants(self.root, 'up')[0]['variants'][0]['phones'], ['AH', 'P'])


if __name__ == '__main__':
    unittest.main()
