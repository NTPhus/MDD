"""Retained posteriors are bounded, identity-bound, and never public chat data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib
import base64

import numpy as np

from src.mdd.acoustic_evidence import pack_evidence, unpack_evidence
from src.mdd.chat_assessment import load_calibration, CALIBRATION_PATH, METHOD
from src.mdd.demo_reference import DICTIONARY_SHA256


def evidence():
    return {'log_probs': np.log(np.array([[.1, .8, .1], [.9, .05, .05]], dtype=np.float32)),
            'id_to_phone': {0: '[PAD]', 1: 'W', 2: 'TH'}, 'blank_id': 0, 'duration_sec': .4,
            'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64}


class EvidenceTests(unittest.TestCase):
    def test_server_roundtrip_preserves_original_posteriors_and_identity(self):
        original = evidence()
        stored = json.loads(json.dumps(pack_evidence(original)))
        restored = unpack_evidence(stored)
        np.testing.assert_array_equal(restored['log_probs'], original['log_probs'])
        self.assertEqual(restored['id_to_phone'], original['id_to_phone'])
        self.assertEqual(restored['checkpoint_sha256'], original['checkpoint_sha256'])

    def test_invalid_shapes_nonfinite_and_compression_bombs_are_rejected(self):
        for matrix in (np.zeros((2001, 3)), np.zeros((2, 257)), np.array([[0, np.nan, -1]])):
            with self.assertRaises(ValueError):
                pack_evidence({**evidence(), 'log_probs': matrix})
        stored = pack_evidence(evidence())
        for change in ({'shape': [True, 3]}, {'shape': [2, 4]}, {'blank_id': 9},
                       {'checkpoint_sha256': 'not a digest'}, {'duration_sec': 31},
                       {'data': base64.b64encode(zlib.compress(b'x' * 1000000)).decode()},
                       {'data': stored['data'] + 'junk'}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                unpack_evidence({**stored, **change})

    def test_malformed_codec_metadata_and_trailing_compressed_members_fail_closed(self):
        stored = pack_evidence(evidence())
        compressed = base64.b64decode(stored['data'])
        for value in ({}, {**stored, 'id_to_phone': {'0': '[PAD]'}},
                      {**stored, 'duration_sec': float('nan')}, {**stored, 'duration_sec': '0.4'},
                      {**stored, 'data': 'not-base64!'},
                      {**stored, 'data': base64.b64encode(compressed[:-1]).decode()},
                      {**stored, 'data': base64.b64encode(compressed + zlib.compress(b'extra')).decode()}):
            with self.subTest(keys=list(value)), self.assertRaises(ValueError):
                unpack_evidence(value)


class CalibrationProofTests(unittest.TestCase):
    """A small complete fixture exercises the same full-validation proof gates.

    Two utterances contain 40 labelled words each, enough to satisfy the actual
    registered 10-correct/5-error decision minima without ML or audio inference.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / CALIBRATION_PATH
        self.config_path = self.root / 'configs/pronunciation_calibration.json'
        self.scorer_path = self.root / 'src/mdd/pronunciation_assessment.py'
        self.protocol_path = self.root / 'data/protocol/fixture/protocol.json'
        self.splits_path = self.root / 'data/protocol/fixture/splits.json'
        self.annotations_path = self.root / 'data/processed/fixture/annotations.jsonl'
        self.write(self.scorer_path, '# Synthetic scorer identity, not executable inference.\n', raw=True)
        self.write(self.protocol_path, {'protocol': 'fixture', 'frozen': True})
        self.splits = {'joint_validation_ids': ['HJK/u1', 'PNV/u2'], 'joint_train_ids': ['LXC/train'],
                       'cascade_test_ids': ['NJS/test']}
        self.write(self.splits_path, self.splits)
        self.write(self.annotations_path, '{"utterance_id":"HJK/u1"}\n{"utterance_id":"PNV/u2"}\n', raw=True)
        policy = {'positive_class': 'manually annotated word has at least one substitution, deletion, or addition',
                  'max_false_positive_rate': .1, 'max_false_negative_rate': .2, 'min_coverage': .5,
                  'min_correct_decisions': 10, 'min_error_decisions': 5, 'grid_points': 31,
                  'objective': 'maximize decision coverage subject to registered constraints'}
        self.config = {'method': METHOD, 'no_training': True, 'no_llm_or_tts': True,
                       'threshold_policy': policy, 'fit_speakers': ['HJK', 'LXC'], 'heldout_speakers': ['PNV'],
                       'protocol_dir': 'data/protocol/fixture', 'annotations': 'data/processed/fixture/annotations.jsonl',
                       'protocol_sha256': self.digest(self.protocol_path), 'splits_sha256': self.digest(self.splits_path),
                       'annotations_sha256': self.digest(self.annotations_path)}
        records = [{'utterance_id': 'HJK/u1', 'speaker_id': 'HJK', 'role': 'fit'},
                   {'utterance_id': 'PNV/u2', 'speaker_id': 'PNV', 'role': 'heldout'}]
        metrics = {'tp': 9, 'fp': 1, 'tn': 19, 'fn': 1, 'gold_correct': 25, 'gold_error': 15,
                   'gold_labelled_words': 40, 'decided_words': 30, 'abstained_words': 10,
                   'correct_abstained': 5, 'error_abstained': 5, 'false_positive_rate': .05,
                   'false_negative_rate': .1, 'coverage': .75}
        self.artifact = {'schema': 'mdd-pronunciation-calibration-v1', 'schema_version': 1,
                         'method': METHOD, 'method_sha256': self.digest(self.scorer_path),
                         'status': 'eligible', 'eligible': True, 'verified': False,
                         'dictionary_sha256': DICTIONARY_SHA256,
                         'checkpoint_sha256': 'a' * 64, 'vocabulary_sha256': 'b' * 64,
                         'protocol_sha256': self.config['protocol_sha256'],
                         'splits_sha256': self.config['splits_sha256'],
                         'annotations_sha256': self.config['annotations_sha256'],
                         'accept_threshold': -2., 'error_threshold': 2.,
                         'validation': {'status': 'verified', 'split': 'calibration', 'failures': [],
                                        'registered_policy': deepcopy(policy), 'fit_speakers': ['HJK', 'LXC'],
                                        'heldout_speakers': ['PNV'], 'selection_mode': 'full-validation',
                                        'fit_utterances': 1, 'heldout_utterances': 1,
                                        'fit_metrics': deepcopy(metrics), 'heldout_metrics': deepcopy(metrics)}}
        self.plan = {'mode': 'full-validation', 'frozen_before_inference': True, 'test_audio_or_gold_used': False,
                     'records': records, 'identity': {key: self.artifact[key] for key in (
                         'method', 'method_sha256', 'checkpoint_sha256', 'vocabulary_sha256', 'dictionary_sha256',
                         'protocol_sha256', 'splits_sha256', 'annotations_sha256')}}
        self.refresh_selection_hash()
        self.persist()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write(path, value, *, raw=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value, sort_keys=True), encoding='utf-8')

    def refresh_selection_hash(self):
        value = hashlib.sha256(json.dumps(self.plan['records'], sort_keys=True, ensure_ascii=False,
                                         separators=(',', ':')).encode()).hexdigest()
        self.plan['selection_sha256'] = value
        self.artifact['validation']['selection_sha256'] = value

    def persist(self):
        self.write(self.config_path, self.config)
        self.write(self.path, self.artifact)
        self.write(self.path.parent / 'selection.json', self.plan)

    def test_complete_full_validation_proof_enables_only_in_memory_verified_flag(self):
        result = load_calibration(self.root, evidence())
        self.assertIsNotNone(result)
        self.assertTrue(result['verified'])
        self.assertFalse(json.loads(self.path.read_text(encoding='utf-8'))['verified'])
        self.assertEqual(result['validation']['fit_metrics']['coverage'], .75)

    def test_invalid_schema_thresholds_and_identity_never_enable_a_policy(self):
        original = deepcopy(self.artifact)
        for change in ({'schema': 'other'}, {'schema_version': 0}, {'method': 'other'}, {'eligible': False},
                       {'status': 'ineligible'}, {'checkpoint_sha256': 'c' * 64},
                       {'vocabulary_sha256': 'c' * 64}, {'dictionary_sha256': 'c' * 64},
                       {'accept_threshold': True}, {'accept_threshold': 2.}, {'accept_threshold': 5.},
                       {'error_threshold': None}, {'error_threshold': float('nan')},
                       {'error_threshold': float('inf')}):
            with self.subTest(change=change):
                self.artifact = {**deepcopy(original), **change}
                self.persist()
                self.assertIsNone(load_calibration(self.root, evidence()))

    def test_changed_scorer_protocol_splits_or_annotations_invalidate_proof(self):
        for source in (self.scorer_path, self.protocol_path, self.splits_path, self.annotations_path):
            original = source.read_text(encoding='utf-8')
            with self.subTest(source=source.name):
                source.write_text(original + '\n# changed source identity\n', encoding='utf-8')
                self.assertIsNone(load_calibration(self.root, evidence()))
                source.write_text(original, encoding='utf-8')

    def test_persisted_verified_flag_cannot_bypass_pilot_or_unfrozen_selection(self):
        original = deepcopy(self.plan)
        self.artifact['verified'] = True
        for change in ({'mode': 'pilot'}, {'frozen_before_inference': False}, {'test_audio_or_gold_used': True},
                       {'selection_sha256': 'c' * 64}, {'identity': {'checkpoint_sha256': 'c' * 64}}):
            with self.subTest(change=change):
                self.plan = {**deepcopy(original), **change}
                self.persist()
                self.assertIsNone(load_calibration(self.root, evidence()))

    def test_test_and_train_overlap_are_rejected_even_after_all_hashes_are_updated(self):
        original = deepcopy(self.splits)
        for key in ('cascade_test_ids', 'joint_train_ids'):
            with self.subTest(split=key):
                self.splits = deepcopy(original)
                self.splits[key].append('PNV/u2')
                self.write(self.splits_path, self.splits)
                digest = self.digest(self.splits_path)
                self.config['splits_sha256'] = self.artifact['splits_sha256'] = digest
                self.plan['identity']['splits_sha256'] = digest
                self.persist()
                self.assertIsNone(load_calibration(self.root, evidence()))

    def test_every_validation_id_is_required_once_with_correct_speaker_role(self):
        original = deepcopy(self.plan['records'])
        for records in ([original[0]], [original[0], original[0]],
                        [original[0], {**original[1], 'speaker_id': 'HJK'}]):
            with self.subTest(records=records):
                self.plan['records'] = deepcopy(records)
                self.refresh_selection_hash()
                self.persist()
                self.assertIsNone(load_calibration(self.root, evidence()))

    def test_relaxed_or_failed_registered_policy_and_test_split_are_rejected(self):
        original = deepcopy(self.artifact['validation'])
        for change in ({'registered_policy': {**original['registered_policy'], 'max_false_positive_rate': .99}},
                       {'status': 'failed'}, {'split': 'test'}, {'failures': ['heldout_gate_failed']},
                       {'fit_speakers': ['PNV']}, {'heldout_speakers': ['HJK']}, {'fit_utterances': 2}):
            with self.subTest(change=change):
                self.artifact['validation'] = {**deepcopy(original), **change}
                self.persist()
                self.assertIsNone(load_calibration(self.root, evidence()))

    def test_nonfinite_coverage_invalid_counts_and_rate_count_inconsistency_fail_closed(self):
        original = deepcopy(self.artifact['validation'])
        for role in ('fit_metrics', 'heldout_metrics'):
            for change in ({'coverage': float('nan')}, {'coverage': .9}, {'coverage': 1.1},
                           {'tp': -1}, {'tp': True}, {'decided_words': 31}, {'gold_correct': 26},
                           {'abstained_words': 9}, {'false_positive_rate': .01},
                           {'false_negative_rate': .01}, {'false_positive_rate': .11}):
                with self.subTest(role=role, change=change):
                    self.artifact['validation'] = deepcopy(original)
                    self.artifact['validation'][role].update(change)
                    self.persist()
                    self.assertIsNone(load_calibration(self.root, evidence()))

    def test_registered_coverage_and_minimum_decision_counts_apply_even_with_consistent_rates(self):
        baseline = deepcopy(self.artifact['validation']['fit_metrics'])
        cases = [
            # Plenty of total decisions, but fewer than ten correct-word decisions.
            {'tn': 9, 'fp': 0, 'tp': 20, 'fn': 0, 'gold_correct': 10, 'gold_error': 30,
             'decided_words': 29, 'abstained_words': 11, 'correct_abstained': 1, 'error_abstained': 10,
             'false_positive_rate': 0., 'false_negative_rate': 0., 'coverage': 29 / 40},
            # High coverage does not excuse fewer than five error-word decisions.
            {'tn': 30, 'fp': 0, 'tp': 4, 'fn': 0, 'gold_correct': 30, 'gold_error': 10,
             'decided_words': 34, 'abstained_words': 6, 'correct_abstained': 0, 'error_abstained': 6,
             'false_positive_rate': 0., 'false_negative_rate': 0., 'coverage': 34 / 40},
            # Good decided FPR/FNR with too much abstention still fails coverage.
            {'gold_correct': 60, 'gold_error': 40, 'gold_labelled_words': 100,
             'abstained_words': 70, 'correct_abstained': 40, 'error_abstained': 30, 'coverage': .3},
        ]
        for change in cases:
            with self.subTest(change=change):
                self.artifact['validation']['fit_metrics'] = {**baseline, **change}
                self.persist()
                self.assertIsNone(load_calibration(self.root, evidence()))

    def test_missing_or_malformed_local_proof_never_raises_or_uses_old_flags(self):
        self.artifact['verified'] = True
        original_config, original_plan = deepcopy(self.config), deepcopy(self.plan)
        original_artifact = deepcopy(self.artifact)
        for kind in ('missing_config', 'missing_policy', 'missing_records', 'missing_identity',
                     'missing_metric', 'invalid_json', 'nonobject_artifact'):
            with self.subTest(kind=kind):
                self.config, self.plan, self.artifact = deepcopy(original_config), deepcopy(original_plan), deepcopy(original_artifact)
                if kind == 'missing_policy':
                    self.config.pop('threshold_policy')
                elif kind == 'missing_records':
                    self.plan.pop('records')
                elif kind == 'missing_identity':
                    self.plan.pop('identity')
                elif kind == 'missing_metric':
                    self.artifact['validation']['heldout_metrics'].pop('coverage')
                self.persist()
                if kind == 'missing_config':
                    self.config_path.unlink()
                elif kind == 'invalid_json':
                    self.path.write_text('{broken', encoding='utf-8')
                elif kind == 'nonobject_artifact':
                    self.path.write_text('[]', encoding='utf-8')
                self.assertIsNone(load_calibration(self.root, evidence()))


if __name__ == '__main__':
    unittest.main()
