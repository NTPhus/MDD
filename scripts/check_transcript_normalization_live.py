"""One real local audio/chat pass; no training, LLM API or TTS calls."""
from pathlib import Path
import base64
import io
import json
import math
import secrets
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transcript_normalization import preservation_reason


def api(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:8765/api/chat/' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def main():
    import scipy.signal
    import soundfile as sf
    output = ROOT / 'data/generated/transcript-normalization-v1'
    config = api('config')
    assert config.get('transcript_normalization') == 'lexical_preserving_v1', 'Restart the updated demo server first'
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    sample_id = protocol['audio_ids'][2]
    samples = [json.loads(line) for line in (ROOT / protocol['source_manifest']).read_text(encoding='utf-8').splitlines() if line.strip()]
    sample = next(row for row in samples if row['utterance_id'] == sample_id)
    waveform, rate = sf.read(ROOT / sample['audio_path'], dtype='float32')
    if waveform.ndim != 1: waveform = waveform.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = scipy.signal.resample_poly(waveform, 16000 // divisor, rate // divisor)
    buffer = io.BytesIO(); sf.write(buffer, waveform, 16000, format='WAV', subtype='PCM_16')
    started = time.monotonic()
    job = api('speech', {'model_id': 'qwen-base', 'request_id': secrets.token_hex(16),
        'audio_base64': base64.b64encode(buffer.getvalue()).decode(), 'history': [], 'topic_id': None})
    stages, identifier = [], job['id']
    while job['status'] not in {'ready', 'error', 'cancelled'}:
        if not stages or stages[-1] != job['status']:
            stages.append(job['status']); print(json.dumps({'stage': job['status'], 'sec': round(time.monotonic() - started, 1)}), flush=True)
        if time.monotonic() - started > 300:
            api('cancel', {'job_id': identifier})
            raise RuntimeError('Only this QA job was cancelled after 300 seconds')
        time.sleep(1)
        job = api('jobs/' + identifier)
    (output / 'live-job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    assert job['status'] == 'ready', job.get('error')
    speech = job['speech']
    assert speech['reference_text'] == speech['transcript'] and speech['reference_source'] == 'asr'
    assert speech['conversation_source'] == 'asr_normalized'
    assert speech['conversation_text'] == speech['normalization']['text']
    assert preservation_reason(speech['transcript'], speech['conversation_text']) is None
    assert speech['normalization']['reference_confirmed'] is False
    assert speech['rewrite']['status'] != 'pending'
    assert not any(key.startswith('_') for key in speech)
    assert job['voice']['status'] == job['sentence_voice']['status'] == 'not_requested'
    report = {'status': 'passed', 'inference': 'Real CPU wav2vec2 ASR + B + base Qwen normalization/writing/dialogue',
        'sample': sample_id, 'transcript': speech['transcript'], 'normalized': speech['conversation_text'],
        'normalization': speech['normalization'], 'reference_source': speech['reference_source'],
        'conversation_source': speech['conversation_source'], 'reply': job['text'],
        'stages_observed': stages, 'elapsed_sec': round(time.monotonic() - started, 3),
        'paid_calls': 0, 'original_audio_and_reference_preserved': True}
    (output / 'live-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
