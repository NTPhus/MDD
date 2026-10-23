"""Exercise installed Kokoro through the existing app; no LLM or paid TTS calls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import io
import json
import time
import urllib.request
import uuid
import wave

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8765')
    parser.add_argument('--expect-device',choices=['cpu','cuda'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.base_url!='http://127.0.0.1:8765':raise ValueError('This validation targets the existing local app only')
    origin=args.base_url
    def request(path,payload=None):
        req=urllib.request.Request(origin+path,data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','Origin':origin})
        with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)
    config=request('/api/chat/config')
    assert config.get('kokoro_available') is True and config['default_voice_provider']=='kokoro'
    voices={v['id'] for v in config['kokoro_voices']}
    assert {'af_heart','am_michael','bf_emma'}<=voices
    topic='daily-life'
    assert any(t['id']==topic for t in config['topics'])
    results=[]
    folder=(args.output or ROOT/('data/generated/kokoro-gpu-validation-v1' if args.expect_device=='cuda' else 'data/generated/kokoro-validation-v1')).resolve()
    if not folder.is_relative_to((ROOT/'data/generated').resolve()):raise ValueError('Output must stay in data/generated')
    folder.mkdir(parents=True,exist_ok=True)
    for index in range(2):
        job=request('/api/chat/topic',{'model_id':'qwen-base','topic_id':topic,'request_id':uuid.uuid4().hex})
        assert job['status']=='ready'
        payload={'job_id':job['id'],'locale':'en-US','provider':'kokoro','voice_id':'am_michael','speed':1.}
        started=time.perf_counter();value=request('/api/chat/voice',payload)
        deadline=time.monotonic()+90
        while value['voice']['status']=='generating' and time.monotonic()<deadline:
            time.sleep(.1);value=request('/api/chat/jobs/'+job['id'])
        voice=value['voice'];assert voice['status']=='ready',voice
        assert voice['provider']=='kokoro'
        if args.expect_device:
            assert voice['timing']['effective_device']==args.expect_device,voice
            expected='CUDAExecutionProvider' if args.expect_device=='cuda' else 'CPUExecutionProvider'
            assert voice['timing']['execution_provider']==expected,voice
        with urllib.request.urlopen(origin+voice['url'],timeout=10) as r:
            content_type=r.headers.get_content_type();audio=r.read()
        assert content_type=='audio/wav' and audio[:4]==b'RIFF'
        with wave.open(io.BytesIO(audio),'rb') as wav:
            assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())==(1,2,24000)
            duration=wav.getnframes()/wav.getframerate();assert duration>1
        again=request('/api/chat/voice',payload)
        assert again['voice']['url']==voice['url']
        results.append({'index':index+1,'voice_ready_sec':round(time.perf_counter()-started,6),
            'provider':voice['provider'],'voice_id':voice['voice_id'],'timing':voice.get('timing'),
            'audio_bytes':len(audio),'audio_duration_sec':duration,'repeat_same_job_idempotent':True})
        if index==0:(folder/'live-opening-michael.wav').write_bytes(audio)
    assert results[1]['timing']['cache_hit'] is True
    result={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'base_url':origin,'default_provider':'kokoro','available_voices':len(voices),'runs':results,
        'remote_tts_calls':0,'llm_calls':0,'training_started':False,
        'scope':'Authored opening through POST topic/voice, job polling and actual WAV retrieval; no learner audio.'}
    (folder/'live-api.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
