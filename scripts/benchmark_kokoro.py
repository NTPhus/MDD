"""Measure actual local TTS, with a saved AI33 baseline and no provider calls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import shutil
import statistics
import os
import subprocess
import sys
import threading
import time
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.kokoro_tts import KokoroTTS,shutdown,manifest,warmup

def gpu_memory():
    try:
        line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0).splitlines()[0]
        used,total=map(int,line.split(','))
        return {'used_mib':used,'total_mib':total}
    except (OSError,ValueError,IndexError,subprocess.SubprocessError):return None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda','auto'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.run:parser.error('Use --run to perform local synthesis')
    if args.device:os.environ['KOKORO_DEVICE']=args.device
    selected=os.environ.get('KOKORO_DEVICE',manifest().get('device','cpu'))
    folder=(args.output or ROOT/('data/generated/kokoro-gpu-validation-v1' if selected!='cpu' else 'data/generated/kokoro-validation-v1')).resolve()
    if not folder.is_relative_to((ROOT/'data/generated').resolve()):parser.error('--output must stay in data/generated')
    folder.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((ROOT/'data/generated/ai33-opening-latency-v1/summary.json').read_text(encoding='utf-8'))
    text=baseline['text'];client=KokoroTTS();rows=[]
    shutdown()
    memory_before=gpu_memory() if selected!='cpu' else None
    samples=[];stop=threading.Event()
    def sample_memory():
        while not stop.is_set():
            value=gpu_memory()
            if value:samples.append(value['used_mib'])
            stop.wait(.2)
    sampler=threading.Thread(target=sample_memory,daemon=True) if selected!='cpu' else None
    if sampler:sampler.start()
    try:
        for index in range(4):
            started=time.perf_counter();path=client.synthesize(text,voice_id='af_heart',speed=1.,force=True)
            rows.append({'label':'cold_start' if index==0 else 'warm_uncached','elapsed_sec':round(time.perf_counter()-started,6),**client.last_timing})
            print(json.dumps(rows[-1]),flush=True)
        cached=client.synthesize(text,voice_id='af_heart',speed=1.)
        rows.append({'label':'cached_repeat',**client.last_timing})
        ready=warmup()
        if selected=='cuda':assert ready['execution_provider']=='CUDAExecutionProvider',ready
        # Distinct voice/rate requests exercise style IDs and fractional speeds.
        extra=[]
        for voice,speed in [('am_michael',1.),('bf_emma',.85),('af_heart',1.15)]:
            p=client.synthesize('What do you enjoy about learning English?',voice_id=voice,speed=speed,force=True)
            with wave.open(str(p),'rb') as wav:
                assert wav.getframerate()==24000 and wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getnframes()>1000
            extra.append({'voice_id':voice,'speed':speed,'audio':p.relative_to(ROOT).as_posix(),'timing':client.last_timing})
        sample=folder/'kokoro-opening-af-heart.wav';shutil.copy2(cached,sample)
        warm=[r['total_sec'] for r in rows if r['label']=='warm_uncached']
        saved=baseline['runs'][0]['elapsed_sec']
        result={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'engine':'kokoro-onnx-0.6.1',
            'device':ready.get('execution_provider',ready['providers'][0]),'runtime':ready,
            'threads':manifest().get('threads',2),'model_file':manifest()['model_file'],
            'text':text,'voice_id':'af_heart','runs':rows,
            'warm_uncached_mean_sec':round(statistics.mean(warm),6),'warm_uncached_median_sec':round(statistics.median(warm),6),
            'saved_ai33_uncached_sec':saved,'saved_ai33_cached_sec':baseline['runs'][1]['elapsed_sec'],
            'saved_ai33_over_kokoro_warm_ratio':round(saved/statistics.mean(warm),3),
            'saved_ai33_source':'data/generated/ai33-opening-latency-v1/summary.json',
            'sample_audio':sample.relative_to(ROOT).as_posix(),'additional_voice_checks':extra,
            'remote_api_calls':0,'training_started':False,
            'scope':'Same authored opening; AI33 saved uncached n=1 vs Kokoro warm uncached n=3. Different voices, timings not simultaneous. Cold and exact cache reported separately.'}
        cpu_path=ROOT/'data/generated/kokoro-validation-v1/summary.json'
        if selected!='cpu' and cpu_path.is_file():
            cpu=json.loads(cpu_path.read_text(encoding='utf-8'))
            result['saved_kokoro_cpu_warm_sec']=cpu['warm_uncached_mean_sec']
            result['saved_cpu_over_gpu_warm_ratio']=round(cpu['warm_uncached_mean_sec']/statistics.mean(warm),3)
        if memory_before:
            result['gpu_memory']={'before':memory_before,'after':gpu_memory(),
                'observed_peak_used_mib':max(samples,default=memory_before['used_mib']),
                'scope':'Whole GPU including other processes; sampled about every 0.2 seconds, not a hard VRAM cap.'}
        (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ['runs','additional_voice_checks']},ensure_ascii=False,indent=2))
    finally:
        stop.set()
        if sampler:sampler.join(timeout=6)
        shutdown()

if __name__=='__main__':main()
