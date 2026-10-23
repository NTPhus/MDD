"""Guard the paid-task budget and preserve existing voice-cache receipts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_ai33_opening import MeasuredClient, cache_state, main
from scripts.ai33_tts import AI33Error, AI33TTS


class OpeningBenchmarkTests(unittest.TestCase):
    def test_at_most_one_submission_even_after_a_failed_http_call(self):
        client = MeasuredClient.__new__(MeasuredClient)
        client.posts, client.events = 0, []
        with patch.object(AI33TTS, '_request_once', side_effect=AI33Error('connection failed')) as request:
            with self.assertRaises(AI33Error):
                client._request_once('POST', '/v3/text-to-speech', {'text': 'Authored opening'})
            with self.assertRaisesRegex(AI33Error, 'budget'):
                client._request_once('POST', '/v3/text-to-speech', {'text': 'Authored opening'})
            self.assertEqual(request.call_count, 1)
        self.assertEqual(client.posts, 1)
        self.assertEqual(client.events[0]['stage'], 'submit_task')
        self.assertNotIn('Authored opening', json.dumps(client.events))

    def test_cache_inspection_preserves_invalid_existing_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = folder / 'receipt.json'
            client = SimpleNamespace(cache_dir=folder, speed=1,
                _cache_request=lambda *args, **kwargs: (None, receipt, None, None))
            self.assertEqual(cache_state(client, 'Hi!'), 'not_submitted')
            receipt.write_text('{"task_id":"existing-task"}')
            self.assertEqual(cache_state(client, 'Hi!'), 'task_receipt')
            original = '{"unknown":"keep this receipt"}'
            receipt.write_text(original)
            with self.assertRaises(AI33Error):
                cache_state(client, 'Hi!')
            self.assertEqual(receipt.read_text(), original)

    def test_real_measurement_requires_an_explicit_run_flag(self):
        with patch('scripts.benchmark_ai33_opening.run_probe') as run:
            with self.assertRaises(SystemExit) as caught:
                main([])
            self.assertEqual(caught.exception.code, 2)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
