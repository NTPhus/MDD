"""The web service owns one cancellable inference prewarm task."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.demo_chat import ChatService


class ChatPrewarmTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.service = ChatService(self.folder.name)
        self.addCleanup(self.service.close)

    def test_repeated_prewarm_uses_the_same_owned_worker_without_jobs(self):
        runner = Mock()
        runner.warmup.return_value = {'qwen_runtime': {'device': 'cuda:0'}}
        with patch.object(self.service, '_module', return_value=runner):
            task = self.service.prewarm()
            self.assertIs(self.service.prewarm(), task)
            task.join(2)
        self.assertFalse(task.is_alive())
        runner.warmup.assert_called_once_with(cancelled=self.service._prewarm_cancelled)
        self.assertEqual(self.service._prewarm_state['status'], 'ready')
        self.assertEqual(self.service._prewarm_state['qwen_device'], 'cuda:0')
        self.assertEqual(len(self.service._jobs), 0)
        self.assertIs(self.service._runner, runner)

    def test_close_cancels_pending_prewarm_and_prevents_late_restart(self):
        entered = threading.Event()
        runner = Mock()
        def prepare(*, cancelled):
            entered.set()
            if not cancelled.wait(2):
                raise AssertionError('close did not cancel model warmup')
            return {}
        runner.warmup.side_effect = prepare
        runner.shutdown.side_effect = lambda: self.assertTrue(self.service._prewarm_cancelled.is_set())
        with patch.object(self.service, '_module', return_value=runner):
            task = self.service.prewarm()
            self.assertTrue(entered.wait(1))
            self.service.close()
            task.join(2)
            self.assertIsNone(self.service.prewarm())
        self.assertFalse(task.is_alive())
        runner.warmup.assert_called_once()
        self.assertEqual(self.service._prewarm_state['status'], 'cancelled')

    def test_failed_background_load_leaves_normal_inference_available(self):
        runner = Mock()
        runner.warmup.side_effect = RuntimeError('fixture model failure')
        with patch.object(self.service, '_module', return_value=runner):
            task = self.service.prewarm()
            task.join(2)
        self.assertEqual(self.service._prewarm_state['status'], 'unavailable')
        self.assertFalse(self.service._closed)
        self.assertIs(self.service._runner, runner)


if __name__ == '__main__':
    unittest.main()
