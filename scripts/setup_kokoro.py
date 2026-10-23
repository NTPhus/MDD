"""Install isolated Kokoro with CPU or CUDA and verified official model files."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models/pretrained/kokoro-82m'
TAG='model-files-v1.1'
BASE='https://github.com/thewh1teagle/kokoro-onnx/releases/download/'+TAG+'/'
FILES={
    # SHA256 pinned from the official, fixed model-files-v1.0 optimized asset.
    'kokoro-v1.0.optimized.onnx':'7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5',
    'kokoro-v1.0.onnx':'beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a',
    'kokoro-v1.0.int8.onnx':'ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c',
    'voices-v1.0.bin':'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d',
}

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def install(device='cpu'):
    folder=ROOT/'.venv-kokoro'
    python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():subprocess.run([sys.executable,'-m','venv',str(folder)],check=True)
    # Clean either previous device first, including a GPU -> CPU reinstall.
    subprocess.run([str(python),'-m','pip','--isolated','uninstall','-y',
        'onnxruntime','onnxruntime-gpu'],check=True)
    subprocess.run([str(python),'-m','pip','--isolated','install','--disable-pip-version-check',
        '-r',str(ROOT/'requirements/kokoro.txt')],check=True)
    if device in ('cuda','auto'):
        # Kokoro's metadata names the CPU distribution. Both ORT distributions
        # write the same import package: install GPU last, never leave both.
        subprocess.run([str(python),'-m','pip','--isolated','uninstall','-y',
            'onnxruntime','onnxruntime-gpu'],check=True)
        subprocess.run([str(python),'-m','pip','--isolated','install','--disable-pip-version-check',
            '-r',str(ROOT/'requirements/kokoro-gpu.txt')],check=True)
    return python

def device_settings(device, memory_mb=768, dll_directory=None):
    result={'device':device,'gpu_mem_limit_mb':memory_mb}
    if device in ('cuda','auto') and os.name=='nt':
        path=(dll_directory or ROOT/'.venv/Lib/site-packages/torch/lib').resolve()
        if not path.is_relative_to(ROOT) or not path.is_dir():
            raise ValueError('CUDA DLL directory must exist inside the project (default: .venv Torch/lib)')
        required=['cudart64_12.dll','cublasLt64_12.dll','cublas64_12.dll','cufft64_11.dll','cudnn64_9.dll']
        if any(not (path/name).is_file() for name in required):
            raise ValueError('CUDA 12 / cuDNN 9 DLLs are missing from the selected directory')
        result['cuda_dll_directory']=path.relative_to(ROOT).as_posix()
    return result

def download(precision='optimized', *, settings=None):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    model_name={'optimized':'kokoro-v1.0.optimized.onnx','fp32':'kokoro-v1.0.onnx','int8':'kokoro-v1.0.int8.onnx'}[precision]
    selected={name:FILES[name] for name in [model_name,'voices-v1.0.bin']}
    urls={name:BASE+name for name in selected}
    if precision=='optimized':urls[model_name]='https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx'
    for name,sha in selected.items():
        path=MODEL_DIR/name
        if path.is_file() and digest(path)==sha:
            print('Verified existing',name,flush=True);continue
        temporary=path.with_suffix(path.suffix+'.part')
        req=urllib.request.Request(urls[name],headers={'User-Agent':'MDD-local-Kokoro-setup/1.0'})
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(req,timeout=90) as response,temporary.open('wb') as target:
            while data:=response.read(1024*1024):target.write(data)
        if digest(temporary)!=sha:raise ValueError('Official model checksum mismatch: '+name)
        temporary.replace(path)
        print('Verified',name,path.stat().st_size,'bytes',flush=True)
    record={'status':'downloaded','model_id':'hexgrad/Kokoro-82M','runtime':'kokoro-onnx==0.6.1',
        'provider':'kokoro','device':'cpu','threads':4,'precision':'fp32-optimized' if precision=='optimized' else precision,'model_file':model_name,
        'voices_file':'voices-v1.0.bin','runtime_python':'.venv-kokoro/Scripts/python.exe' if os.name=='nt' else '.venv-kokoro/bin/python',
        'default_voice_id':'af_heart','release':'model-files-v1.0' if precision=='optimized' else TAG,
        'source':'https://github.com/thewh1teagle/kokoro-onnx/releases/tag/'+('model-files-v1.0' if precision=='optimized' else TAG),
        'checksum_source':('Official pinned v1.0 download; SHA256 retained locally' if precision=='optimized' else 'https://github.com/thewh1teagle/kokoro-onnx/releases/expanded_assets/'+TAG),
        'model_license':'Apache-2.0','runtime_license':'MIT',
        'files':{name:{'sha256':sha,'bytes':(MODEL_DIR/name).stat().st_size,'url':urls[name]} for name,sha in selected.items()},
        'downloaded_at_utc':datetime.now(timezone.utc).isoformat()}
    record.update(settings or {'device':'cpu'})
    (MODEL_DIR/'setup_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')

def verify():
    record=json.loads((MODEL_DIR/'setup_manifest.json').read_text(encoding='utf-8'))
    for name in [record['model_file'],record['voices_file']]:
        expected=FILES[name]
        if not (MODEL_DIR/name).is_file() or digest(MODEL_DIR/name)!=expected:raise ValueError('Missing/corrupt '+name)
    print('Kokoro model files verified')

def smoke():
    """Mark ready only after the installed Windows runtime makes real audio."""
    sys.path.insert(0,str(ROOT))
    from scripts.kokoro_tts import KokoroTTS,warmup,shutdown
    record=json.loads((MODEL_DIR/'setup_manifest.json').read_text(encoding='utf-8'))
    client=KokoroTTS()
    try:
        ready=warmup()
        path=client.synthesize('Hello! Kokoro is ready for local English speaking practice.',force=True)
        if path.stat().st_size<1000:raise ValueError('No usable smoke audio')
        runtime=ROOT/record['runtime_python']
        package_code="import json,importlib.metadata as m; names=['kokoro-onnx','onnxruntime','onnxruntime-gpu','numpy','soundfile','espeakng-loader','phonemizer']; installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; print(json.dumps({x:installed[x] for x in names if x in installed}))"
        packages=json.loads(subprocess.check_output([str(runtime),'-X','utf8','-c',package_code],text=True,encoding='utf-8'))
        record.update(status='ready',voices=ready['voices'],packages=packages,
            verified_at_utc=datetime.now(timezone.utc).isoformat(),
            smoke={'audio':path.relative_to(ROOT).as_posix(),'timing':client.last_timing,'providers':ready['providers'],
                   'effective_device':ready.get('effective_device','cpu'),
                   'execution_provider':ready.get('execution_provider',ready['providers'][0])})
        (MODEL_DIR/'setup_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'status':'ready','voices':len(ready['voices']),'smoke':record['smoke']},indent=2))
    finally:shutdown()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--model',choices=['optimized','fp32','int8'],default='optimized')
    parser.add_argument('--device',choices=['cpu','cuda','auto'],help='Preserve the installed device when omitted')
    parser.add_argument('--gpu-mem-limit-mb',type=int,default=768,help='CUDA arena limit; total VRAM can be higher')
    parser.add_argument('--cuda-dll-directory',type=Path,help='Project-local CUDA 12 / cuDNN 9 directory on Windows')
    args=parser.parse_args()
    if not any([args.install,args.download,args.verify,args.smoke]):parser.error('Select --install, --download, --verify and/or --smoke')
    if not 256<=args.gpu_mem_limit_mb<=4096:parser.error('--gpu-mem-limit-mb must be 256–4096')
    previous={}
    if (MODEL_DIR/'setup_manifest.json').is_file():
        previous=json.loads((MODEL_DIR/'setup_manifest.json').read_text(encoding='utf-8'))
    device=args.device or previous.get('device','cpu')
    memory_mb=args.gpu_mem_limit_mb if args.device else previous.get('gpu_mem_limit_mb',args.gpu_mem_limit_mb)
    dll_directory=args.cuda_dll_directory or (ROOT/previous['cuda_dll_directory'] if previous.get('cuda_dll_directory') else None)
    settings=device_settings(device,memory_mb,dll_directory)
    if args.install:install(device)
    if args.download:download(args.model,settings=settings)
    elif args.device and (args.install or args.smoke):
        if not previous:parser.error('Download the model before configuring a device')
        previous.update(settings,status='downloaded')
        (MODEL_DIR/'setup_manifest.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.verify:verify()
    if args.smoke:smoke()
