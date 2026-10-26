"""BERT chooses conversation/pronunciation text while retaining acoustic evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from mdd.demo_model_registry import recovery_selection
from scripts.local_speech_engine import LocalSpeechEngine


class RecoverySelectionTests(unittest.TestCase):
    def test_only_a_complete_pinned_install_selects_bert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(recovery_selection(root)['provider'], 'qwen')
            model = root / 'models/pretrained/bert-base-uncased'
            model.mkdir(parents=True)
            files = {}
            for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors'):
                (model / name).write_bytes(b'fixture')
                files[name] = {'bytes': 7}
            manifest = {'model_id': 'google-bert/bert-base-uncased', 'revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
                        'status': 'downloading', 'files': files}
            path = model / 'setup_manifest.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(recovery_selection(root)['provider'], 'qwen')
            manifest['status'] = 'downloaded'
            path.write_text(json.dumps(manifest))
            self.assertEqual(recovery_selection(root)['provider'], 'bert')
            (model / 'model.safetensors').write_bytes(b'partial')
            self.assertEqual(recovery_selection(root)['provider'], 'bert')  # Same size; loader verifies SHA.
            (model / 'model.safetensors').write_bytes(b'cut')
            self.assertEqual(recovery_selection(root)['provider'], 'qwen')


class BertRoutingTests(unittest.TestCase):
    def run_case(self, *, reading=False):
        engine = object.__new__(LocalSpeechEngine)
        evidence = {'transcript': 'My fravorit is tea', 'reference_text': 'My fravorit is tea',
                    'reference': None, 'missing_words': ['fravorit'], 'timings': {},
                    'phonetic_candidates': {'words': [{'index': 1, 'candidates': [{'word': 'favorite'}]}]},
                    '_acoustic_evidence': {'posterior': 'private'}}
        engine.analyze = Mock(return_value=deepcopy(evidence))
        engine._respond = Mock(side_effect=lambda request, result, *args: result)
        inferred = {'original': evidence['transcript'], 'text': 'My favorite is tea.', 'status': 'accepted',
                    'latency_sec': .25, 'source': 'bert_inferred', 'reference_confirmed': False}
        engine._use_bert_reference = Mock(side_effect=lambda result: {
            **result, 'reference_text': inferred['text'], 'reference_source': 'bert_inferred'})
        events, cancelled, qwen = [], threading.Event(), Mock()
        with patch('scripts.bert_transcript_recovery.BertTranscriptRecovery') as model:
            model.return_value.recover.return_value = inferred
            result = engine.run({'audio_base64': 'YWJj', 'history': [], 'recovery_enabled': True,
                                 'recovery_provider': 'bert',
                                 'reference_text': 'My favorite is tea.' if reading else None},
                                cancelled, events.append, qwen)
            if reading:
                model.assert_not_called()
                engine._use_bert_reference.assert_not_called()
            else:
                model.return_value.recover.assert_called_once_with(evidence['transcript'], cancelled,
                    phonetic_candidates=evidence['phonetic_candidates'], history=[])
                self.assertEqual(result['interpretation'], inferred)
                self.assertEqual(result['timings']['interpretation_sec'], .25)
                engine._use_bert_reference.assert_called_once()
        qwen.generate.assert_not_called()
        self.assertEqual(result['transcript'], evidence['transcript'])
        self.assertEqual(result['reference_text'], evidence['reference_text'] if reading else inferred['text'])
        self.assertEqual(result['_acoustic_evidence'], evidence['_acoustic_evidence'])
        return result

    def test_server_selected_bert_uses_audio_candidates_without_qwen_recovery(self):
        self.run_case()

    def test_reading_prompt_bypasses_bert(self):
        self.assertNotIn('interpretation', self.run_case(reading=True))


if __name__ == '__main__':
    unittest.main()
