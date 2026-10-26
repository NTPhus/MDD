"""Paired CPU/CUDA latency probe using local audio, no training or paid APIs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def selected_cases():
    manifest=json.loads((ROOT/'data/generated/asr-backends-20261023/manifest.json').read_text(encoding='utf-8'))
    cases=manifest['cases']
    tlv=[row for row in cases if row['id'].startswith('TLV-')]
    return [tlv[0],tlv[len(tlv)//2],max(tlv,key=lambda r:r['duration_sec']),
            next(r for r in cases if r['group']=='saa_external_same_prompt'),
            next(r for r in cases if r['id']=='user-road'),next(r for r in cases if r['id']=='user-stella')]


def run(output):
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.compare_asr_backends import load_audio, alignment
    from scripts.whisper_asr import WhisperASR, OPTIONS
    from mdd.demo_inference import LocalInference
    output.mkdir(parents=True,exist_ok=True)
    cases=selected_cases()
    samples={row['id']:load_audio(ROOT/row['audio']) for row in cases}
    audio_bytes={}
    for identifier,audio in samples.items():
        buffer=io.BytesIO();sf.write(buffer,audio,16000,format='WAV',subtype='PCM_16')
        audio_bytes[identifier]=buffer.getvalue()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
            'scope':'Same six development clips; warm latency, CPU then CUDA. No claim of independent accuracy or whole-pipeline speedup.',
            'cases':[{'id':r['id'],'duration_sec':len(samples[r['id']])/16000,'audio_sha256':hashlib.sha256((ROOT/r['audio']).read_bytes()).hexdigest()} for r in cases],
            'whisper_options':OPTIONS,'predictions':{},'summaries':{},'gpu_samples':[],
            'training_started':False,'remote_api_calls':0}
    stopped=threading.Event()
    def sample_gpu():
        while not stopped.is_set():
            try:
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                                            text=True,creationflags=flags,timeout=3).strip()
                memory,usage=[float(x.strip()) for x in raw.split(',')]
                result['gpu_samples'].append({'elapsed_sec':round(time.monotonic()-start,3),'used_mib':memory,'utilization_percent':usage})
            except (OSError,ValueError,subprocess.SubprocessError): pass
            stopped.wait(.25)
    start=time.monotonic();monitor=threading.Thread(target=sample_gpu,daemon=True);monitor.start()
    model={'id':'cascade-B','label':'wav2vec2 B','kind':'phoneme','experimental':False,
           'path':ROOT/'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint'}
    try:
        for kind in ('whisper','phoneme-B'):
            for device in ('cpu','cuda'):
                key=f'{kind}-{device}';result['predictions'][key]=[]
                backend=(WhisperASR(ROOT,device=device) if kind=='whisper'
                         else LocalInference(ROOT,device=device))
                # Exclude checksum/load and the first kernel initialization.
                if kind=='whisper': backend.transcribe(samples[cases[0]['id']])
                else: backend.posterior_evidence(model,audio_bytes[cases[0]['id']])
                for case in cases:
                    identifier=case['id']
                    if torch.cuda.is_initialized(): torch.cuda.synchronize()
                    before=time.perf_counter()
                    if kind=='whisper':
                        value=backend.transcribe(samples[identifier])
                        prediction={'text':value['raw_text'],'runtime':{k:value[k] for k in ('device','compute_type','requested_device','fallback_reason')},
                                    'word_errors':alignment(case['reference_text'],value['raw_text'])}
                    else:
                        value=backend.posterior_evidence(model,audio_bytes[identifier])
                        prediction={'phones':[p['phone'] for p in value['phonemes']],'runtime':value['runtime'],
                                    'posterior_finite':bool(np.isfinite(value['log_probs']).all())}
                    if torch.cuda.is_initialized(): torch.cuda.synchronize()
                    prediction.update(id=identifier,elapsed_sec=round(time.perf_counter()-before,4))
                    result['predictions'][key].append(prediction)
                    save(output/'results.json',result)
                    print(json.dumps({'stage':key,'case':identifier,'sec':prediction['elapsed_sec'],'runtime':prediction['runtime']}),flush=True)
                result['summaries'][key]={'cases':len(cases),'total_sec':round(sum(r['elapsed_sec'] for r in result['predictions'][key]),4)}
                if kind=='whisper': backend.close()
                del backend,value
                gc.collect()
                if torch.cuda.is_initialized(): torch.cuda.empty_cache()
        for kind in ('whisper','phoneme-B'):
            result['summaries'][kind+'-speedup']=round(result['summaries'][kind+'-cpu']['total_sec']/result['summaries'][kind+'-cuda']['total_sec'],3)
        result['text_changes_cpu_cuda']=[r['id'] for a,r in zip(result['predictions']['whisper-cpu'],result['predictions']['whisper-cuda']) if a['text']!=r['text']]
        result['phone_changes_cpu_cuda']=[r['id'] for a,r in zip(result['predictions']['phoneme-B-cpu'],result['predictions']['phoneme-B-cuda']) if a['phones']!=r['phones']]
        result['status']='completed'
    finally:
        stopped.set();monitor.join(timeout=4)
        result['observed_peak_gpu_used_mib']=max((r['used_mib'] for r in result['gpu_samples']),default=None)
        result['elapsed_sec']=round(time.monotonic()-start,3)
        save(output/'results.json',result)
    print(json.dumps(result['summaries'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'data/generated/gpu-inference-v1/models')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'data/generated'): parser.error('Output must remain under data/generated')
    run(output)
