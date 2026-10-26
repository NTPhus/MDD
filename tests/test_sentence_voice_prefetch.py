"""Background sentence voice is idempotent, bounded and independent of reply TTS."""
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.demo_chat import ChatService, ChatJob, ChatError


class SentencePrefetchTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.service = ChatService(Path(self.folder.name), capability_loader=lambda: {
            'tts_configured': True, 'english_voice_configured': True})
        self.audio = self.service.cache_dir / 'fixture.mp3'
        self.audio.parent.mkdir(parents=True)
        self.audio.write_bytes(b'ID3fixture')
        self.job = ChatJob('qwen-base', status='correcting', text='', speech={
            'transcript': 'I usually wake up early.', 'reference_text': 'I usually wake up early.',
            'reply_ready': False, 'rewrite': {'status': 'pending'}})
        self.service._jobs[self.job.identifier] = self.job
        self.payload = {'job_id': self.job.identifier, 'locale': 'en-US', 'target': 'sentence'}

    def release_slot(self, target):
        slot = self.service._sentence_voice_slot if target == 'sentence' else self.service._voice_slot
        self.assertTrue(slot.acquire(timeout=3), 'Voice worker did not release its own slot')
        slot.release()

    def test_prefetch_before_reply_is_idempotent_and_does_not_block_bot_voice(self):
        entered_sentence, entered_reply, release = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        calls = []
        def synthesize(text, **options):
            calls.append(text)
            (entered_sentence if text == self.job.speech['reference_text'] else entered_reply).set()
            release.wait(3)
            return self.audio
        self.service.tts_factory = lambda: SimpleNamespace(synthesize=synthesize)
        first = self.service.voice(self.payload)
        self.assertTrue(entered_sentence.wait(1))
        self.assertEqual(first['sentence_voice']['status'], 'generating')
        self.service.voice(self.payload)
        self.assertEqual(calls, ['I usually wake up early.'])
        with self.assertRaises(ChatError):
            self.service.voice({'job_id': self.job.identifier, 'locale': 'en-US'})
        with self.service._lock:
            self.job.text = 'What helps you get up early?'
            self.job.speech['reply_ready'] = True
        self.service.voice({'job_id': self.job.identifier, 'locale': 'en-US'})
        self.assertTrue(entered_reply.wait(1), 'Sentence prefetch occupied the reply voice slot')
        pending = self.service.poll(self.job.identifier)
        self.assertEqual(pending['sentence_voice']['status'], 'generating')
        self.assertEqual(pending['voice']['status'], 'generating')
        release.set()
        self.release_slot('sentence'); self.release_slot('reply')
        done = self.service.poll(self.job.identifier)
        self.assertEqual(done['sentence_voice']['status'], 'ready')
        self.assertEqual(done['voice']['status'], 'ready')
        self.service.voice(self.payload)
        self.assertEqual(len(calls), 2, 'Listen re-submitted already prefetched audio')

    def test_another_sentence_is_bounded_while_reply_slot_remains_free(self):
        self.assertTrue(self.service._sentence_voice_slot.acquire(blocking=False))
        self.addCleanup(self.service._sentence_voice_slot.release)
        with self.assertRaises(ChatError) as caught:
            self.service.voice(self.payload)
        self.assertEqual(caught.exception.code, 'chat_voice_busy')
        self.assertTrue(self.service._voice_slot.acquire(blocking=False))
        self.service._voice_slot.release()

    def test_cancelled_generation_discards_late_sentence_and_releases_slot(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def synthesize(*args, **options):
            entered.set(); release.wait(3); return self.audio
        self.service.tts_factory = lambda: SimpleNamespace(synthesize=synthesize)
        self.service.voice(self.payload)
        self.assertTrue(entered.wait(1))
        self.job.cancelled.set(); self.job.status = 'cancelled'
        release.set(); self.release_slot('sentence')
        self.assertEqual(self.job.sentence_voice['status'], 'cancelled')
        self.assertEqual(self.service._voices, {})

    def test_sentence_needs_recognized_text_before_prefetch(self):
        self.job.speech = None
        with self.assertRaises(ChatError) as caught:
            self.service.voice(self.payload)
        self.assertEqual(caught.exception.code, 'chat_not_ready')

    def test_pending_bert_reference_cannot_synthesize_raw_transcript(self):
        calls = []
        self.service.tts_factory = lambda: SimpleNamespace(synthesize=lambda text, **options: calls.append(text))
        self.job.speech['pronunciation_reference_ready'] = False
        with self.assertRaises(ChatError) as caught:
            self.service.voice(self.payload)
        self.assertEqual(caught.exception.code, 'sentence_reference_pending')
        self.assertEqual(calls, [])
        self.assertEqual(self.job.sentence_voice['status'], 'not_requested')

    def test_sentence_synthesis_uses_ready_bert_reference_instead_of_raw_asr(self):
        calls = []
        recovered = 'I usually wake up early.'
        self.job.speech.update(transcript='I usuly wake up erly', reference_text=recovered,
                              reference_source='bert_inferred', pronunciation_reference_ready=True)
        def synthesize(text, **options):
            calls.append(text)
            return self.audio
        self.service.tts_factory = lambda: SimpleNamespace(synthesize=synthesize)
        self.service.voice(self.payload)
        self.release_slot('sentence')
        self.assertEqual(calls, [recovered])
        self.assertEqual(self.job.sentence_voice_text, recovered)


if __name__ == '__main__':
    unittest.main()
