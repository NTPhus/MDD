"""Load-only startup warms existing model objects without changing speech outputs."""
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.local_speech_engine import LocalSpeechEngine
from scripts import local_chat_process


class SpeechPrewarmTests(unittest.TestCase):
    def setUp(self):
        self.order=[]
        self.event=threading.Event()
        self.engine=object.__new__(LocalSpeechEngine)
        self.engine.bert_recovery=None
        self.engine.asr_model_id='fixture-asr'
        self.engine.load_asr=Mock(side_effect=lambda:self.order.append('asr'))
        self.engine._phoneme_model=Mock(return_value={'id':'cascade-B','kind':'phoneme'})
        self.engine.phonemes=SimpleNamespace(_get_bundle=Mock(side_effect=lambda model:self.order.append('phoneme')),
            posterior_evidence=Mock(side_effect=AssertionError('No audio inference during prewarm')))
        self.engine._prime_dictionaries=Mock(side_effect=lambda cancelled:self.order.append('dictionary'))
        self.engine.analyze=Mock(side_effect=AssertionError('No audio analysis during prewarm'))
        self.qwen=SimpleNamespace(load=Mock(side_effect=lambda model:self.order.append('qwen')),
            runtime={'device':'cuda:0'},generate=Mock(side_effect=AssertionError('No generation during prewarm')))
        selection=patch('mdd.demo_model_registry.recovery_selection',return_value={'provider':'qwen','ready':False})
        self.selection=selection.start();self.addCleanup(selection.stop)

    def test_models_load_qwen_first_and_no_generation_or_audio_forward_runs(self):
        result=self.engine.warmup(self.event,self.qwen)
        self.assertEqual(self.order,['qwen','asr','phoneme','dictionary'])
        self.qwen.load.assert_called_once_with('qwen-base')
        self.assertEqual(result['status'],'ready')
        self.assertTrue(result['load_only'])
        self.assertEqual(result['qwen_runtime'],{'device':'cuda:0'})
        self.assertEqual(set(result['warmup_timings']),{'qwen_load_sec','asr_load_sec','phoneme_load_sec',
            'dictionary_load_sec','recovery_load_sec'})
        self.qwen.generate.assert_not_called();self.engine.analyze.assert_not_called()
        self.engine.phonemes.posterior_evidence.assert_not_called()

    def test_pre_cancelled_warmup_and_mid_load_cancellation_skip_later_models(self):
        self.event.set()
        self.assertEqual(self.engine.warmup(self.event,self.qwen)['finish_reason'],'cancelled')
        self.qwen.load.assert_not_called()
        self.event.clear()
        self.qwen.load.side_effect=lambda model:self.event.set()
        self.assertEqual(self.engine.warmup(self.event,self.qwen)['finish_reason'],'cancelled')
        self.engine.load_asr.assert_not_called()
        self.engine.phonemes._get_bundle.assert_not_called()

    def test_only_selected_bert_is_loaded_and_reuses_the_existing_instance(self):
        self.selection.return_value={'provider':'bert','ready':True}
        bert=SimpleNamespace(_load=Mock(side_effect=lambda event:self.order.append('bert')))
        with patch('scripts.bert_transcript_recovery.BertTranscriptRecovery',return_value=bert) as constructor:
            self.engine.warmup(self.event,self.qwen)
            self.engine.warmup(self.event,self.qwen)
        constructor.assert_called_once_with(ROOT)
        self.assertIs(self.engine.bert_recovery,bert)
        self.assertEqual(bert._load.call_count,2)
        self.assertEqual(self.order,['qwen','asr','phoneme','dictionary','bert']*2)

    def test_asr_load_helper_preserves_checksum_cpu_eval_and_disabled_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            weights=b'test-only ASR weights'
            (path/'model.safetensors').write_bytes(weights)
            expected=hashlib.sha256(weights).hexdigest()
            config=SimpleNamespace()
            network=Mock()
            network.cpu.return_value=network;network.eval.return_value=network;network.requires_grad_.return_value=network
            config_class=SimpleNamespace(from_pretrained=Mock(return_value=config))
            model_class=SimpleNamespace(from_pretrained=Mock(return_value=network))
            processor_class=SimpleNamespace(from_pretrained=Mock(return_value=Mock()))
            fake=SimpleNamespace(Wav2Vec2Config=config_class,Wav2Vec2ForCTC=model_class,
                                 Wav2Vec2Processor=processor_class)
            engine=object.__new__(LocalSpeechEngine);engine.asr=engine.processor=None
            with patch.dict(sys.modules,{'transformers':fake}),patch('mdd.demo_model_registry.speech_asr_selection',
                    return_value={'id':'fixture-asr','path':path,'sha256':expected}):
                engine.load_asr();engine.load_asr()
                self.assertIs(engine.asr,network)
                self.assertEqual(model_class.from_pretrained.call_count,1)
                self.assertFalse(config.apply_spec_augment)
                self.assertEqual((config.mask_time_prob,config.mask_feature_prob),(0.,0.))
                network.cpu.assert_called_once_with();network.eval.assert_called_once_with()
                network.requires_grad_.assert_called_once_with(False);network.assert_not_called()
                engine.asr=None
                (path/'model.safetensors').write_bytes(b'changed weights')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    engine.load_asr()
                self.assertEqual(model_class.from_pretrained.call_count,1)


class ProcessPrewarmTests(unittest.TestCase):
    def test_warmup_uses_same_worker_request_and_preserves_owned_cancel_event(self):
        event=threading.Event()
        with patch.object(local_chat_process,'_request',return_value={'status':'ready'}) as request:
            result=local_chat_process.warmup(event)
        self.assertEqual(result,{'status':'ready'})
        self.assertEqual(request.call_args.args[0],{'type':'warmup','model_id':'qwen-base'})
        self.assertIs(request.call_args.kwargs['cancelled'],event)
        self.assertEqual(local_chat_process._WARMUPS,set())

    def test_shutdown_cancels_queued_warmup_before_it_can_restart_worker(self):
        entered=threading.Event();event=threading.Event();result=[]
        original=local_chat_process._request
        def request(*args,**kwargs):
            entered.set()
            return original(*args,**kwargs)
        with patch.object(local_chat_process,'_PROCESS',None),patch.object(local_chat_process,'_WARMUPS',set()), \
                patch.object(local_chat_process,'_shutdown_worker') as shutdown, \
                patch.object(local_chat_process,'_start') as start, \
                patch.object(local_chat_process,'_request',side_effect=request):
            with local_chat_process._LOCK:
                thread=threading.Thread(target=lambda:result.append(local_chat_process.warmup(event)))
                thread.start()
                self.assertTrue(entered.wait(1))
                local_chat_process.shutdown()
                self.assertTrue(event.is_set())
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]['finish_reason'],'cancelled')
        start.assert_not_called();shutdown.assert_called_once_with()

    def test_stale_shutdown_epoch_never_starts_a_worker(self):
        with patch.object(local_chat_process,'_SHUTDOWN_EPOCH',10), \
                patch.object(local_chat_process,'_start') as start:
            result=local_chat_process._request({'type':'warmup','model_id':'qwen-base'},5.,
                on_text=lambda value:None,cancelled=threading.Event(),shutdown_epoch=9)
        self.assertEqual(result['finish_reason'],'cancelled')
        start.assert_not_called()


if __name__=='__main__':unittest.main()
