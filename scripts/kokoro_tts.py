"""Local Kokoro TTS adapter. Standard-library parent; isolated persistent worker."""
from __future__ import annotations
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import secrets
import subprocess
import sys
import threading
import time

try:
    from scripts.kokoro_worker import WorkerError, runtime_config
except ModuleNotFoundError:
    from kokoro_worker import WorkerError, runtime_config

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models/pretrained/kokoro-82m'
CACHE=ROOT/'data/generated/tts/kokoro'
_LOCK=threading.RLock()
_PROCESS=None
_EVENTS=None
_KEY=None
_READY=None


class KokoroError(RuntimeError):
    def __init__(self,message='Local Kokoro voice is unavailable.',code='kokoro_unavailable'):
        super().__init__(message);self.code=code;self.retryable=False


def manifest():
    try:
        value=json.loads((MODEL_DIR/'setup_manifest.json').read_text(encoding='utf-8'))
        runtime=(ROOT/value['runtime_python']).resolve()
        if not runtime.is_relative_to(ROOT) or not runtime.is_file():raise ValueError()
        for key in ['model_file','voices_file']:
            p=(MODEL_DIR/value[key]).resolve()
            if not p.is_relative_to(MODEL_DIR.resolve()) or not p.is_file():raise ValueError()
        return value
    except (OSError,ValueError,KeyError,TypeError):
        raise KokoroError('Run scripts/setup_kokoro.py before using local TTS.') from None


def _configuration(record):
    try:return runtime_config(record)
    except WorkerError as error:raise KokoroError(str(error),code=error.code) from None


def _worker_key(record, config=None):
    config=_configuration(record) if config is None else config
    return (record['files'][record['model_file']]['sha256'],record['files'][record['voices_file']]['sha256'],
            int(os.environ.get('KOKORO_THREADS',record.get('threads',2))),config['requested_device'],
            config['cuda_dll_directory'],json.dumps(config['cuda_provider_options'],sort_keys=True),
            record.get('runtime_python'),os.environ.get('CUDA_VISIBLE_DEVICES'))


def _execution(record):
    config=_configuration(record)
    if config['requested_device']=='cpu':
        return {'requested_device':'cpu','effective_device':'cpu','execution_provider':'CPUExecutionProvider',
                'provider_options':{'CPUExecutionProvider':{}}}
    ready=_start()
    effective=ready.get('effective_device')
    if effective not in ('cpu','cuda'):raise KokoroError(code='kokoro_protocol_error')
    if config['requested_device']=='cuda' and effective!='cuda':
        raise KokoroError('CUDA was requested but is unavailable.',code='kokoro_cuda_fallback')
    provider='CUDAExecutionProvider' if effective=='cuda' else 'CPUExecutionProvider'
    value={'requested_device':config['requested_device'],'effective_device':effective,'execution_provider':provider,
           'provider_options':{provider:dict(config['cuda_provider_options']) if effective=='cuda' else {}}}
    if ready.get('fallback_reason') in ('cuda_unavailable','cuda_session_fell_back','cuda_load_failed'):
        value['fallback_reason']=ready['fallback_reason']
    return value


def shutdown():
    global _PROCESS,_EVENTS,_KEY,_READY
    with _LOCK:
        process=_PROCESS
        _PROCESS=_EVENTS=_KEY=_READY=None
        if process is None:return
        try:
            if process.poll() is None:
                process.stdin.write('{"type":"shutdown"}\n');process.stdin.flush()
                process.wait(timeout=5)
        except (OSError,subprocess.TimeoutExpired):
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        finally:
            for stream in [process.stdin,process.stdout]:
                if stream:stream.close()


def _read_until(timeout):
    deadline=time.monotonic()+timeout
    while True:
        try:
            value=_EVENTS.get(timeout=min(.25,max(.001,deadline-time.monotonic())))
        except Empty:
            if _PROCESS is None or _PROCESS.poll() is not None:raise KokoroError(code='kokoro_worker_stopped')
            if time.monotonic()>=deadline:raise KokoroError(code='kokoro_timeout')
            continue
        if value is None:raise KokoroError(code='kokoro_worker_stopped')
        return value


def _start():
    global _PROCESS,_EVENTS,_KEY,_READY
    record=manifest()
    key=_worker_key(record)
    if _PROCESS is not None and _PROCESS.poll() is None and _KEY==key:return _READY
    shutdown()
    CACHE.mkdir(parents=True,exist_ok=True)
    # phonemizer copies its eSpeak DLL into a fresh temp directory before
    # synthesizing. The demo may run under a restricted Windows account whose
    # profile %TEMP% allows directory creation but denies writing the DLL.
    # Keep the worker's temporary files under the writable project cache.
    worker_temp=CACHE/('worker-tmp-'+secrets.token_hex(6))
    worker_temp.mkdir(parents=True,exist_ok=True)
    espeak_temp_dirs=[worker_temp/f'espeak-{index}' for index in range(8)]
    for path in espeak_temp_dirs:path.mkdir()
    environment={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
        'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'KOKORO_ESPEAK_TEMP_DIRS':os.pathsep.join(map(str,espeak_temp_dirs))}
    with (CACHE/'worker.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([str(ROOT/record['runtime_python']),'-X','utf8','-u',str(ROOT/'scripts/kokoro_worker.py')],
            cwd=ROOT,env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    events=Queue()
    def receive():
        try:
            for line in process.stdout:
                try:events.put(json.loads(line))
                except ValueError:continue
        finally:events.put(None)
    _PROCESS,_EVENTS,_KEY=process,events,key
    threading.Thread(target=receive,daemon=True,name='kokoro-voice-output').start()
    try:
        ready=_read_until(120.)
        if ready.get('type')!='ready':
            code=ready.get('code')
            allowed={'kokoro_invalid_device','kokoro_invalid_gpu_memory','kokoro_invalid_cuda_directory',
                     'kokoro_cuda_unavailable','kokoro_cuda_load_failed','kokoro_cuda_fallback'}
            if code not in allowed:code='kokoro_load_failed'
            raise KokoroError('Local Kokoro runtime could not start with the requested device.',code=code)
        _READY=ready
        return ready
    except Exception:
        shutdown();raise


def warmup():
    with _LOCK:return dict(_start())


def prewarm_async():
    def warm():
        try:
            warmup()
            KokoroTTS().synthesize('Ready for English practice.',force=True)
        except Exception:pass
    task=threading.Thread(target=warm,daemon=True,name='kokoro-prewarm');task.start();return task


def _request(text,voice,speed,path):
    previous_process=_PROCESS
    _start()
    load_sec=_READY['load_sec'] if _PROCESS is not previous_process else 0.
    identifier=secrets.token_hex(16)
    try:
        _PROCESS.stdin.write(json.dumps({'id':identifier,'type':'synthesize','text':text,
            'voice_id':voice,'speed':speed,'output':str(path)},ensure_ascii=False)+'\n');_PROCESS.stdin.flush()
        value=_read_until(180.)
        if value.get('id')!=identifier:raise KokoroError(code='kokoro_protocol_error')
        if value.get('type')!='done':raise KokoroError(code='kokoro_synthesis_failed')
        value['load_sec']=load_sec
        return value
    except (OSError,KokoroError):
        shutdown();raise


class KokoroTTS:
    def __init__(self,cache_dir=CACHE):
        self.cache_dir=Path(cache_dir).resolve()
        if not self.cache_dir.is_relative_to(CACHE.resolve()):raise KokoroError(code='kokoro_invalid_cache')
        self.record=manifest();self.last_timing={}

    def _options(self,text,locale,voice_id,speed,execution=None):
        if not isinstance(text,str) or not 0<len(text.strip())<=5000:raise KokoroError(code='kokoro_invalid_text')
        if not isinstance(locale,str) or locale not in {'en-US','en-GB'}:raise KokoroError(code='kokoro_language_unsupported')
        if voice_id is not None and (not isinstance(voice_id,str) or not voice_id):raise KokoroError(code='kokoro_invalid_voice')
        voice=self.record.get('default_voice_id','af_heart') if voice_id is None else voice_id
        known=self.record.get('voices') or ['af_heart','af_bella','af_sarah','am_adam','am_michael','bf_emma','bm_george']
        if voice not in known or not voice.startswith(('af_','am_','bf_','bm_')):raise KokoroError(code='kokoro_invalid_voice')
        speed=1. if speed is None else speed
        if type(speed) not in (int,float) or not math.isfinite(speed) or not .5<=speed<=1.5:raise KokoroError(code='kokoro_invalid_speed')
        key={'engine':'kokoro-onnx-0.6.1','model_sha256':self.record['files'][self.record['model_file']]['sha256'],
             'voices_sha256':self.record['files'][self.record['voices_file']]['sha256'],
             'text':text.strip(),'locale':locale,'voice_id':voice,'speed':float(speed)}
        if execution is not None and execution['effective_device']=='cuda':
            key.update({'execution_device':'cuda','execution_provider':'CUDAExecutionProvider',
                        'cuda_provider_options':execution['provider_options']['CUDAExecutionProvider']})
        stem=hashlib.sha256(json.dumps(key,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        return key,self.cache_dir/(stem+'.wav')

    def synthesize(self,text,*,locale='en-US',voice_id=None,speed=None,force=False):
        started=time.perf_counter()
        # Validate user input before starting a GPU worker, even for cached audio.
        self._options(text,locale,voice_id,speed)
        with _LOCK:
            previous_process=_PROCESS
            execution=_execution(self.record)
            preflight_load_sec=0.
            if _PROCESS is not previous_process and isinstance(_READY,dict):
                value=_READY.get('load_sec')
                if type(value) in (int,float) and math.isfinite(value) and value>=0:preflight_load_sec=value
            key,path=self._options(text,locale,voice_id,speed,execution)
            lookup=time.perf_counter()
            meta=path.with_suffix('.json')
            if not force and path.is_file() and meta.is_file():
                try:
                    value=json.loads(meta.read_text(encoding='utf-8'))
                    with path.open('rb') as f:valid=f.read(4)==b'RIFF'
                    if valid and value['request']==key and value['audio_sha256']==hashlib.sha256(path.read_bytes()).hexdigest():
                        self.last_timing={'total_sec':round(time.perf_counter()-started,6),'cache_lookup_sec':round(time.perf_counter()-lookup,6),
                            'cache_hit':True,'synthesis_sec':0.,'load_sec':preflight_load_sec,'new_tasks_submitted':0,
                            'audio_duration_sec':value['audio_duration_sec'],'real_time_factor':0.,**execution}
                        return path
                except (OSError,ValueError,KeyError):pass
            lookup_sec=time.perf_counter()-lookup
            path.parent.mkdir(parents=True,exist_ok=True)
            result=_request(key['text'],key['voice_id'],key['speed'],path)
            result['load_sec']=result['load_sec']+preflight_load_sec
            if Path(result['path']).resolve()!=path or not path.is_file():raise KokoroError(code='kokoro_invalid_output')
            if result.get('effective_device',execution['effective_device'])!=execution['effective_device']:
                raise KokoroError('The synthesis provider changed unexpectedly.',code='kokoro_device_mismatch')
            metadata={'request':key,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'audio_duration_sec':result['audio_duration_sec'],'sample_rate':result['sample_rate'],'execution':execution}
            temp=meta.with_suffix('.json.tmp');temp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(meta)
            self.last_timing={'total_sec':round(time.perf_counter()-started,6),'cache_lookup_sec':round(lookup_sec,6),
                'cache_hit':False,'new_tasks_submitted':0,**{k:result[k] for k in ['synthesis_sec','load_sec','audio_duration_sec','real_time_factor']},
                **{k:result[k] for k in ['phonemize_sec','inference_sec'] if k in result},**execution}
            return path

    def resume(self,text,**options):return self.synthesize(text,**options)
    def can_resume(self,text,**options):return False


atexit.register(shutdown)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text',nargs='?',default='Hello! Kokoro is ready for local English speaking practice.')
    parser.add_argument('--voice',default='af_heart');parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--warmup',action='store_true');args=parser.parse_args()
    if args.warmup:print(json.dumps(warmup(),indent=2))
    else:
        client=KokoroTTS();path=client.synthesize(args.text,voice_id=args.voice,speed=args.speed)
        print(json.dumps({'audio':str(path),'timing':client.last_timing},indent=2))
