import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.asr_vi_data import normalized,prompt_split,audit
from scripts import asr_vi_data
from scripts.grammar_inference import rewrite
from mdd.demo_model_registry import asr_selection,grammar_selection

class DatasetTests(unittest.TestCase):
    def test_prepared_bytes_match_manifest_digests_on_windows(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'protocol';source.mkdir()
            (source/'splits.json').write_text('{"source_train_ids":[]}')
            (source/'manifest.jsonl').write_text('')
            texts={}
            for first in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                for second in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    text='HELLO '+first+second+' WORLD'
                    texts.setdefault(prompt_split(text),text)
            for speaker,split in [('T1','train'),('T2','train'),('V','validation'),('E','test')]:
                folder=root/'corpus'/speaker
                (folder/'transcript').mkdir(parents=True);(folder/'wav').mkdir()
                (folder/'transcript/a.txt').write_text(texts[split])
                (folder/'wav/a.wav').write_bytes((speaker+texts[split]).encode())
            config={'corpus_root':'corpus','dataset_dir':'prepared','source_protocol':'protocol','seed':42,
                    'train_speakers':['T1','T2'],'validation_speakers':['V'],'test_speakers':['E'],
                    'source_replay_fraction':.25,'max_train_seconds':8}
            with patch.object(asr_vi_data,'ROOT',root),patch('soundfile.info',return_value=SimpleNamespace(frames=16000,samplerate=16000,channels=1)):
                expected=asr_vi_data.prepare(config)
                parts,actual=asr_vi_data.load(config)
                self.assertEqual(actual,expected);self.assertEqual(len(parts['train']),2)
                self.assertNotIn(b'\r\n',(root/'prepared/train.jsonl').read_bytes())
    def test_prompt_holdout_ignores_punctuation_case_and_whitespace(self):
        self.assertEqual(prompt_split("It WAS difficult, to get there!"),prompt_split('it was difficult to get there'))
        self.assertEqual(normalized("I'm ready."),"I'M READY")
        for bad in ['Tôi nói tiếng Anh','I have 3 cats','']:
            with self.subTest(bad=bad),self.assertRaises(ValueError):normalized(bad)

    def test_audit_rejects_speaker_prompt_or_audio_leaks(self):
        parts={split:[{'speaker_id':split,'text':split,'audio_sha256':split}]
               for split in ['train','validation','test']}
        audit(parts)
        for key in ['speaker_id','text','audio_sha256']:
            value=parts['test'][0][key];parts['test'][0][key]=parts['train'][0][key]
            with self.subTest(key=key),self.assertRaises(ValueError):audit(parts)
            parts['test'][0][key]=value

class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def write(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def test_asr_rejects_incomplete_or_nonimproving_runs(self):
        run='runs/asr-vietnamese-english-v1/'
        self.write(run+'best_checkpoint/model.safetensors',{})
        self.assertFalse(asr_selection(self.root)['trained'])
        for status,improved,expected in [('started',True,False),('completed',False,False),('completed',True,True)]:
            self.write(run+'training_provenance.json',{'status':status,'results':{'validation_improved':improved,'trained_sha256':'digest'}})
            self.assertEqual(asr_selection(self.root)['trained'],expected)
    def test_grammar_requires_matching_completed_adapter_validation(self):
        run='runs/grammar-qwen3-06b-wi-jfleg-v1/'
        self.write(run+'adapter/adapter_model.safetensors',{})
        self.write(run+'training_provenance.json',{'status':'completed','adapter_sha256':'good',
                   'effective_config':{'dataset_kind':'published_grammar_fluency'}})
        self.assertFalse(grammar_selection(self.root)['ready'])
        self.write(run+'demo_validation.json',{'status':'passed','adapter_sha256':'wrong'})
        self.assertFalse(grammar_selection(self.root)['ready'])
        self.write(run+'demo_validation.json',{'status':'passed','adapter_sha256':'good'})
        self.assertTrue(grammar_selection(self.root)['ready'])

class GrammarAdapterTests(unittest.TestCase):
    @staticmethod
    def engine(outputs):
        qwen=Mock()
        def generate(request,cancelled,emit):
            task=json.loads(request['messages'][-1]['content'])['task']
            text,finish_reason=outputs[task]
            emit({'type':'chunk','text':json.dumps({task:text})})
            return {'finish_reason':finish_reason}
        qwen.generate.side_effect=generate
        return qwen

    def test_two_training_format_tasks_keep_corrections_out_of_dialogue(self):
        qwen=Mock();seen=[]
        def generate(request,cancelled,emit):
            seen.append(request)
            task=json.loads(request['messages'][-1]['content'])['task']
            emit({'type':'chunk','text':json.dumps({task:'I usually wake up at seven am.'})})
            return {'finish_reason':'stop'}
        qwen.generate.side_effect=generate
        result=rewrite(qwen,'I usually wakes up at seven am',threading.Event())
        self.assertEqual(result['model_id'],'qwen-grammar');self.assertEqual(result['status'],'ready')
        self.assertTrue(result['grammar_changed'])
        self.assertEqual([r['model_id'] for r in seen],['qwen-grammar','qwen-grammar'])
        self.assertTrue(all('generation_profile' not in r for r in seen))
        self.assertEqual(result['feedback_state'],'suggestions_available')

    def test_guard_rejection_and_other_task_budget_failure_leave_no_ready_output(self):
        qwen=self.engine({'grammar_corrected':('I like tea.','stop'),
                          'natural':('I like coffee.','length')})
        result=rewrite(qwen,'I like coffee.',threading.Event())
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['feedback_state'],'unavailable')
        self.assertFalse(result['changed']);self.assertFalse(result['partial'])
        self.assertEqual(result['grammar_corrected'],'');self.assertEqual(result['natural'],'')
        self.assertEqual(result['field_reasons'],{
            'grammar_corrected':'unsupported_edit','natural':'generation_budget_exhausted'})
        self.assertEqual(result['task_failures'],{'natural':'generation_budget_exhausted'})
        self.assertNotEqual(result['explanation'],'No wording change was suggested.')

    def test_partial_output_retains_only_surviving_edits_and_their_explanation(self):
        qwen=self.engine({'grammar_corrected':('I usually drink tea.','stop'),
                          'natural':('I usually wake up at seven am.','stop')})
        result=rewrite(qwen,'I usually wakes up at seven am.',threading.Event())
        self.assertEqual(result['status'],'ready');self.assertTrue(result['partial'])
        self.assertEqual(result['feedback_state'],'partial')
        self.assertFalse(result['grammar_changed']);self.assertEqual(result['grammar_corrected'],'')
        self.assertTrue(result['natural_changed']);self.assertTrue(result['changed'])
        self.assertIn('wakes',result['explanation']);self.assertIn('wake',result['explanation'])
        self.assertNotIn('drink',result['explanation'])
        self.assertEqual(result['field_reasons'],{'grammar_corrected':'unsupported_edit'})

    def test_only_discarded_task_format_edits_do_not_count_as_a_change(self):
        qwen=self.engine({'grammar_corrected':('I like coffee.','length'),
                          'natural':('I like coffee','stop')})
        result=rewrite(qwen,'I like coffee',threading.Event())
        self.assertEqual(result['feedback_state'],'partial');self.assertTrue(result['partial'])
        self.assertFalse(result['changed']);self.assertFalse(result['grammar_format_changed'])
        self.assertEqual(result['explanation'],'No wording change was suggested.')

    def test_unchanged_result_is_not_a_grammar_or_meaning_verdict(self):
        for source in ['I like coffee.','I flubble coffee.']:
            with self.subTest(source=source):
                qwen=self.engine({task:(source,'stop') for task in ['grammar_corrected','natural']})
                result=rewrite(qwen,source,threading.Event())
                self.assertEqual(result['status'],'ready')
                self.assertEqual(result['feedback_state'],'no_change_suggested')
                self.assertFalse(result['partial']);self.assertFalse(result['changed'])
                self.assertEqual(result['field_reasons'],{})
                self.assertEqual(result['explanation'],'No wording change was suggested.')

    def test_format_only_changes_are_recorded_without_claiming_a_wording_correction(self):
        qwen=self.engine({task:('I like coffee.','stop') for task in ['grammar_corrected','natural']})
        result=rewrite(qwen,'i like coffee',threading.Event())
        self.assertEqual(result['feedback_state'],'no_change_suggested')
        self.assertTrue(result['changed']);self.assertTrue(result['grammar_format_changed'])
        self.assertFalse(result['grammar_changed'])

    def test_long_transcript_budget_is_bounded_and_keeps_the_timeout(self):
        for count,expected in [(40,320),(41,512)]:
            with self.subTest(words=count):
                source=' '.join(['coffee']*count)
                qwen=self.engine({task:(source,'stop') for task in ['grammar_corrected','natural']})
                rewrite(qwen,source,threading.Event())
                for call in qwen.generate.call_args_list:
                    self.assertEqual(call.args[0]['max_tokens'],expected)
                    self.assertEqual(call.args[0]['timeout'],45.)
    def test_invalid_tasks_are_unavailable_instead_of_successful_unchanged_output(self):
        qwen=Mock()
        qwen.generate.side_effect=lambda request,cancelled,emit:(emit({'type':'chunk','text':'wrong format'}) or {'finish_reason':'stop'})
        result=rewrite(qwen,'I like coffee',threading.Event())
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['grammar_status'],'unavailable');self.assertEqual(result['natural_status'],'unavailable')
        self.assertFalse(result['changed'])
    def test_cancellation_never_starts_the_second_task(self):
        qwen=Mock();cancelled=threading.Event()
        def generate(request,event,emit):
            event.set();return {'finish_reason':'cancelled'}
        qwen.generate.side_effect=generate
        self.assertIsNone(rewrite(qwen,'I like coffee',cancelled));self.assertEqual(qwen.generate.call_count,1)

if __name__=='__main__':unittest.main()
