"""Persistent, isolated Kokoro worker; stdin/stdout are bounded JSONL IPC."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
_DLL_HANDLES=[]
_ESPEAK_TEMP_DIRS=None


def phonemize(engine,text,language):
    """Use parent-created DLL temp folders for Windows sandbox ACLs."""
    import tempfile
    global _ESPEAK_TEMP_DIRS
    if os.name!='nt' or _ESPEAK_TEMP_DIRS is None:
        return engine.tokenizer.phonemize(text,language)
    original=tempfile.mkdtemp
    def existing_tempdir(*args,**kwargs):
        try:path=next(_ESPEAK_TEMP_DIRS)
        except StopIteration:raise WorkerError('Kokoro exhausted its prepared eSpeak temp folders.','kokoro_temp_unavailable') from None
        if not path.is_dir():raise WorkerError('Kokoro eSpeak temp folder is unavailable.','kokoro_temp_unavailable')
        return str(path)
    tempfile.mkdtemp=existing_tempdir
    try:return engine.tokenizer.phonemize(text,language)
    finally:tempfile.mkdtemp=original


class WorkerError(RuntimeError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code=code


def runtime_config(record, *, environ=None, root=ROOT):
    """Read device settings with no runtime imports; keep DLL paths inside the repo."""
    environment=os.environ if environ is None else environ
    device=environment.get('KOKORO_DEVICE',record.get('device','cpu'))
    if not isinstance(device,str) or device.strip().lower() not in {'cpu','cuda','auto'}:
        raise WorkerError('Kokoro device must be cpu, cuda, or auto.','kokoro_invalid_device')
    device=device.strip().lower()
    memory=environment.get('KOKORO_GPU_MEM_LIMIT_MB',record.get('gpu_mem_limit_mb',768))
    try:
        if isinstance(memory,bool) or str(int(memory))!=str(memory):raise ValueError()
        memory=int(memory)
        if not 64<=memory<=65536:raise ValueError()
    except (TypeError,ValueError,OverflowError):
        raise WorkerError('Kokoro GPU memory limit must be 64–65536 MiB.','kokoro_invalid_gpu_memory') from None
    dll=record.get('cuda_dll_directory') or None
    if dll is not None:
        if not isinstance(dll,str) or Path(dll).is_absolute():
            raise WorkerError('Kokoro CUDA DLL directory must be inside the project.','kokoro_invalid_cuda_directory')
        path=(root/dll).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_dir():
            raise WorkerError('Kokoro CUDA DLL directory is unavailable.','kokoro_invalid_cuda_directory')
        dll=str(path)
    options={'device_id':'0','gpu_mem_limit':str(memory*1024*1024),
             'arena_extend_strategy':'kSameAsRequested','cudnn_conv_algo_search':'HEURISTIC',
             'do_copy_in_default_stream':'1','cudnn_conv_use_max_workspace':'0','use_tf32':'1'}
    return {'requested_device':device,'cuda_dll_directory':dll,'gpu_mem_limit_mb':memory,
            'cuda_provider_options':options}


def preload_cuda(ort, directory):
    """ORT 1.21+ handles DLLs; older Windows wheels use the existing Torch DLLs."""
    loader=getattr(ort,'preload_dlls',None)
    if callable(loader):
        loader(directory=directory)
        return
    if os.name!='nt' or directory is None:return
    import ctypes
    _DLL_HANDLES.append(os.add_dll_directory(directory))
    names=['cudart64_12.dll','cublasLt64_12.dll','cublas64_12.dll','cufft64_11.dll',
           'cudnn64_9.dll']
    folder=Path(directory)
    names.extend(path.name for path in sorted(folder.glob('cudnn*64_9.dll')) if path.name not in names)
    for name in names:
        path=folder/name
        if not path.is_file():raise WorkerError('A required CUDA runtime DLL is missing.','kokoro_cuda_load_failed')
        _DLL_HANDLES.append(ctypes.WinDLL(str(path)))


def session_telemetry(session, config):
    providers=session.get_providers()
    effective='cuda' if 'CUDAExecutionProvider' in providers else 'cpu'
    if config['requested_device']=='cuda' and effective!='cuda':
        raise WorkerError('CUDA was requested but the session uses CPU.','kokoro_cuda_fallback')
    provider='CUDAExecutionProvider' if effective=='cuda' else 'CPUExecutionProvider'
    return {'requested_device':config['requested_device'],'effective_device':effective,
            'execution_provider':provider,'providers':providers,
            'provider_options':{provider:dict(config['cuda_provider_options']) if effective=='cuda' else {}}}


def create_session(ort, model_path, options, config):
    requested=config['requested_device']
    cuda=requested!='cpu' and 'CUDAExecutionProvider' in ort.get_available_providers()
    fallback_reason=None
    if requested=='cuda' and not cuda:
        raise WorkerError('CUDAExecutionProvider is unavailable; install the GPU runtime.','kokoro_cuda_unavailable')
    if requested=='auto' and not cuda:fallback_reason='cuda_unavailable'
    if cuda:
        try:
            preload_cuda(ort,config['cuda_dll_directory'])
            session=ort.InferenceSession(str(model_path),sess_options=options,
                providers=[('CUDAExecutionProvider',config['cuda_provider_options'])],enable_fallback=False)
            telemetry=session_telemetry(session,config)
            if telemetry['effective_device']=='cuda':
                disable=getattr(session,'disable_fallback',None)
                if callable(disable):disable()
                return session,telemetry
            fallback_reason='cuda_session_fell_back'
        except Exception as error:
            if requested=='cuda':
                if isinstance(error,WorkerError):raise
                raise WorkerError('CUDA session initialization failed.','kokoro_cuda_load_failed') from None
            fallback_reason='cuda_load_failed'
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    telemetry=session_telemetry(session,config)
    if fallback_reason:telemetry['fallback_reason']=fallback_reason
    return session,telemetry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models/pretrained/kokoro-82m')
    args=parser.parse_args()
    folder=args.model_dir.resolve()
    if not folder.is_relative_to(ROOT):raise ValueError('Invalid model directory')
    record=json.loads((folder/'setup_manifest.json').read_text(encoding='utf-8'))
    config=runtime_config(record)
    model_path=(folder/record['model_file']).resolve()
    voice_path=(folder/record['voices_file']).resolve()
    for path in [model_path,voice_path]:
        if not path.is_relative_to(folder):raise ValueError('Invalid model file')
        expected=record['files'][path.name]['sha256']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:raise ValueError('Model checksum mismatch')
    load_started=time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
        from kokoro_onnx import Kokoro
        options=ort.SessionOptions()
        threads=int(os.environ.get('KOKORO_THREADS',record.get('threads',2)))
        if not 1<=threads<=8:raise ValueError('Kokoro threads must be 1–8')
        options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session,telemetry=create_session(ort,model_path,options,config)
        engine=Kokoro.from_session(session,str(voice_path))
    global _ESPEAK_TEMP_DIRS
    raw_temp_dirs=os.environ.get('KOKORO_ESPEAK_TEMP_DIRS')
    if os.name=='nt' and raw_temp_dirs:
        allowed=(ROOT/'data/generated/tts/kokoro').resolve()
        paths=[Path(value).resolve() for value in raw_temp_dirs.split(os.pathsep) if value]
        if not paths or any(not path.is_relative_to(allowed) or not path.is_dir() for path in paths):
            raise WorkerError('Prepared eSpeak temp folders are invalid.','kokoro_temp_unavailable')
        _ESPEAK_TEMP_DIRS=iter(paths)
    load_sec=time.perf_counter()-load_started
    voices=sorted(v for v in engine.voices.files if v.startswith(('af_','am_','bf_','bm_')))
    logging.getLogger('kokoro_onnx').setLevel(logging.WARNING)
    print(json.dumps({'type':'ready','voices':voices,'load_sec':round(load_sec,6),**telemetry}),flush=True)
    for line in sys.stdin:
        identifier=None
        try:
            if len(line)>20000:raise ValueError('Oversized request')
            request=json.loads(line);identifier=request.get('id')
            if request.get('type')=='shutdown':break
            if not isinstance(identifier,str) or not 1<=len(identifier)<=64:raise ValueError('Invalid ID')
            if request.get('type')=='ping':
                print(json.dumps({'id':identifier,'type':'done','voices':voices,'load_sec':round(load_sec,6),
                                 **session_telemetry(session,config)}),flush=True);continue
            if request.get('type')!='synthesize':raise ValueError('Invalid operation')
            text=request.get('text');voice=request.get('voice_id');speed=request.get('speed')
            if not isinstance(text,str) or not 0<len(text.strip())<=5000:raise ValueError('Invalid text')
            if voice not in voices:raise ValueError('Invalid voice')
            if type(speed) not in (int,float) or not np.isfinite(speed) or not .5<=speed<=1.5:raise ValueError('Invalid speed')
            path=Path(request['output']).resolve()
            allowed=(ROOT/'data/generated/tts/kokoro').resolve()
            if not path.is_relative_to(allowed) or path.suffix!='.wav':raise ValueError('Invalid audio destination')
            path.parent.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                language='en-gb' if voice.startswith('b') else 'en-us'
                phonemes=phonemize(engine,text.strip(),language)
                phonemize_sec=time.perf_counter()-started
                infer_started=time.perf_counter()
                audio,rate=engine.create(phonemes,voice=voice,speed=float(speed),lang=language,is_phonemes=True)
                inference_sec=time.perf_counter()-infer_started
            elapsed=time.perf_counter()-started
            audio=np.asarray(audio,dtype=np.float32)
            if audio.ndim!=1 or not 0<len(audio)<=24000*600 or not np.isfinite(audio).all():raise ValueError('Invalid audio')
            temporary=path.with_suffix('.'+identifier+'.tmp')
            sf.write(str(temporary),audio,rate,format='WAV',subtype='PCM_16');temporary.replace(path)
            duration=len(audio)/rate
            print(json.dumps({'id':identifier,'type':'done','path':str(path),'sample_rate':int(rate),
                'synthesis_sec':round(elapsed,6),'load_sec':round(load_sec,6),'audio_duration_sec':round(duration,6),
                'phonemize_sec':round(phonemize_sec,6),'inference_sec':round(inference_sec,6),
                'real_time_factor':round(elapsed/duration,6),'voice_id':voice,
                **session_telemetry(session,config)}),flush=True)
        except Exception as error:
            # The caller gets a typed error, never learner text or a raw exception.
            import traceback
            frames=traceback.extract_tb(error.__traceback__)
            location=frames[-1] if frames else None
            print(json.dumps({'kokoro_error':type(error).__name__,
                'source':Path(location.filename).name if location else '',
                'line':location.lineno if location else 0}),file=sys.stderr,flush=True)
            if os.environ.get('KOKORO_DEBUG')=='1':traceback.print_exc(file=sys.stderr)
            print(json.dumps({'id':identifier,'type':'error','code':'kokoro_synthesis_failed','exception':type(error).__name__}),flush=True)

if __name__=='__main__':
    try:main()
    except WorkerError as error:
        print(json.dumps({'type':'error','code':error.code,'message':str(error)}),flush=True)
        sys.exit(1)
