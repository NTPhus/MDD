"""Owned localhost jobs with explicitly local Kokoro; no external API calls."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.benchmark_demo_latency import LocalDemoClient, NoRedirect, encode_audio


class Client(LocalDemoClient):
    def voice(self, job, target='reply'):
        if target not in {'reply','sentence'}: raise ValueError('Invalid voice target')
        data=json.dumps({'job_id':job,'provider':'kokoro','voice_id':'af_heart','speed':1.,
                         'locale':'en-US','target':target}).encode()
        request=urllib.request.Request(self.base+'/api/chat/voice',data=data,
            headers={'Content-Type':'application/json','Origin':self.base})
        with self.opener.open(request,timeout=20) as response:return json.load(response)


def main():
    out=ROOT/'data/generated/gpu-inference-v1/live';out.mkdir(parents=True,exist_ok=True)
    client=Client('http://127.0.0.1:8765')
    deadline=time.monotonic()+90
    while True:
        config=client.request('/api/chat/config')
        if config['inference_warmup']['status'] in {'ready','unavailable'}:break
        if time.monotonic()>deadline:raise TimeoutError('Prewarm')
        time.sleep(.25)
    assert config['inference_warmup']['status']=='ready',config['inference_warmup']
    assert config['inference_warmup']['qwen_device']=='cuda:0'
    assert config['inference_warmup']['asr_runtime']['device']=='cuda'
    assert config['inference_warmup']['phoneme_runtime']['device']=='cuda:0'
    manifest=json.loads((ROOT/'data/generated/asr-backends-20261023/manifest.json').read_text(encoding='utf-8'))
    ids=['user-road','user-stella']
    cases=[next(r for r in manifest['cases'] if r['id']==identifier) for identifier in ids]
    cases.append(manifest['cases'][0])
    results={'status':'running','config':{k:config[k] for k in ('inference_warmup','speech_device_policy','default_voice_provider')},
             'jobs':[],'gpu_samples':[],'external_api_calls':0,'training_started':False}
    stop=threading.Event();start=time.monotonic()
    def monitor():
        while not stop.is_set():
            try:
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                    timeout=3,text=True,creationflags=flags).strip()
                memory,usage=map(float,raw.split(','))
                results['gpu_samples'].append({'elapsed_sec':round(time.monotonic()-start,3),'used_mib':memory,'utilization_percent':usage})
            except (OSError,ValueError,subprocess.SubprocessError):pass
            stop.wait(.2)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        for index,case in enumerate(cases):
            model='crottc-if-l2-arctic' if index==2 else 'cascade-B'
            encoded,audio=encode_audio(ROOT/case['audio'])
            before=time.monotonic();reply_sec=voice_sec=None;voice_sent=False
            job=client.request('/api/chat/speech',{'model_id':'qwen-base','pronunciation_model_id':model,
                'request_id':uuid.uuid4().hex,'history':[],'audio_base64':encoded})
            while True:
                speech=job.get('speech') or {}
                if speech.get('reply_ready') is True and reply_sec is None:reply_sec=round(time.monotonic()-before,3)
                # Deliberately synthesize while grammar/phone processing is still active.
                if reply_sec is not None and not voice_sent:
                    job=client.voice(job['id']);voice_sent=True
                voice=job.get('voice') or {}
                if voice.get('status')=='ready' and voice_sec is None:voice_sec=round(time.monotonic()-before,3)
                if job['status'] in {'error','cancelled'}:raise RuntimeError(job.get('error'))
                if voice.get('status')=='error':raise RuntimeError(voice)
                if job['status']=='ready' and voice.get('status')=='ready':break
                if time.monotonic()-before>150:raise TimeoutError('GPU live job')
                time.sleep(.15);job=client.request('/api/chat/jobs/'+job['id'])
            speech=job['speech'];runtime=speech['pronunciation_runtime']
            assert speech['asr_runtime']['device']=='cuda',speech['asr_runtime']
            assert job['metadata']['qwen_runtime']['device']=='cuda:0',job['metadata']
            assert voice['provider']=='kokoro'
            assert voice['timing']['effective_device']=='cuda',voice
            if model=='cascade-B':assert runtime['device']=='cuda:0',runtime
            else:
                assert runtime['requested_device']=='auto',runtime
                assert runtime['device'] in {'cpu','cuda:0'},runtime
            sentence_voice=None
            if case['id']=='user-stella':
                # The long reference previously exceeded Kokoro's 768 MiB arena.
                job=client.voice(job['id'],target='sentence')
                deadline=time.monotonic()+45
                while job['sentence_voice']['status']=='generating':
                    if time.monotonic()>deadline:raise TimeoutError('Long sentence voice')
                    time.sleep(.15);job=client.request('/api/chat/jobs/'+job['id'])
                sentence_voice=job['sentence_voice']
                assert sentence_voice['status']=='ready',sentence_voice
                assert sentence_voice['provider']=='kokoro'
                assert sentence_voice['timing']['effective_device']=='cuda',sentence_voice
            item={'case':case['id'],'pronunciation_model':model,'audio':audio,
                  'reply_sec':reply_sec,'voice_ready_sec':voice_sec,'full_elapsed_sec':round(time.monotonic()-before,3),
                  'asr_runtime':speech['asr_runtime'],'pronunciation_runtime':runtime,
                  'qwen_runtime':job['metadata']['qwen_runtime'],'voice':voice,
                  'timings':speech['timings'],'transcript':speech['transcript'],'reply':job['text'],
                  'sentence_voice':sentence_voice}
            results['jobs'].append(item)
            (out/f'{index+1}-{case["id"]}.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:item[k] for k in ('case','reply_sec','voice_ready_sec','full_elapsed_sec','pronunciation_runtime')},ensure_ascii=False),flush=True)
    except Exception as error:
        results.update(status='error',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        stop.set();worker.join(timeout=4)
        results['observed_peak_gpu_used_mib']=max((r['used_mib'] for r in results['gpu_samples']),default=None)
        (out/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    # Restore the fast default B resident model after checking CRO's memory gate.
    encoded,_=encode_audio(ROOT/cases[0]['audio'])
    from scripts.benchmark_demo_latency import run_once
    timing,job=run_once(client,encoded,1,.15,100.)
    assert job['speech']['pronunciation_runtime']['device']=='cuda:0'
    (out/'restored-B.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
    results.update(status='completed',default_B_restored=True)
    (out/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Live CUDA Whisper/B/Qwen/Kokoro passed; default B restored.',flush=True)


if __name__=='__main__':main()
