"""Real local audio: immediate review, changed-word regeneration, unchanged no-op."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import secrets
import sys
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated/optional-reference-qa/live-pipeline.json')
    parser.add_argument('--feedback-first-only', action='store_true', help='Check one recording and feedback/reply order without reference regenerations')
    args = parser.parse_args()
    if not args.url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise ValueError('This check only sends the validation recording to a local server')
    from mdd.demo_inference import LocalInference
    import numpy as np
    base = args.url.rstrip('/')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, payload=None):
        headers = {'Origin': base}
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(base + path,
            data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        with opener.open(request, timeout=25) as response:
            return json.load(response)

    timelines = []
    def wait(job, revision):
        then = time.monotonic(); stages = []
        trace = {'revision': revision, 'feedback_ready_sec': None, 'first_text_sec': None}
        timelines.append(trace)
        while time.monotonic() - then < 300:
            assert job['metadata'].get('reference_revision', 0) == revision
            if job['status'] not in stages:
                stages.append(job['status']); print('Revision', revision, job['status'], flush=True)
            speech = job.get('speech') or {}
            feedback_ready = bool(speech.get('transcript')) and all(
                (speech.get(field) or {}).get('status') in {'ready', 'unavailable', 'not_needed'}
                for field in ('rewrite', 'correction', 'coaching'))
            if feedback_ready and trace['feedback_ready_sec'] is None:
                trace['feedback_ready_sec'] = round(time.monotonic() - then, 3)
                trace['feedback_snapshot'] = job
            if job.get('text') and trace['first_text_sec'] is None:
                assert feedback_ready, 'Bot text appeared before pronunciation/writing feedback completed'
                trace['first_text_sec'] = round(time.monotonic() - then, 3)
                trace['first_text_snapshot'] = job
            if job['status'] in {'ready', 'error', 'cancelled'}:
                if job['status'] != 'ready':
                    raise RuntimeError('Reply did not finish: ' + json.dumps(job.get('error')))
                return job, stages
            time.sleep(.4)
            job = api('/api/chat/jobs/' + job['id'])
        raise RuntimeError('Timed out waiting for local reply')

    config = api('/api/chat/config')
    assert config['reference_regenerates_reply'] is True
    assert config['reference_regenerates_only_on_change'] is True
    assert config['pronunciation_review'] == 'immediate_provisional'
    fixtures = json.loads((ROOT/'data/generated/pronunciation-error-qa/real-audio.json').read_text(encoding='utf-8'))
    case = next(item for item in fixtures['examples'] if item['utterance_id'] == 'LXC/arctic_a0029')
    source = ROOT/case['audio_path']
    waveform, _ = LocalInference(ROOT)._audio(source.read_bytes())
    encoded = io.BytesIO()
    with wave.open(encoded, 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes((np.clip(waveform, -1, 1) * 32767).astype('<i2').tobytes())
    original, stages = wait(api('/api/chat/speech', {'model_id': 'qwen-base', 'request_id': secrets.token_hex(16),
        'audio_base64': base64.b64encode(encoded.getvalue()).decode(), 'history': []}), 0)
    sound = original['speech']['sound_candidates']
    assert sound['status'] == 'available' and sound['provisional'] is True
    assert sound['reference_confirmed'] is False and sound['count'] > 0
    assert all(word['status'] in {'uncertain', 'unavailable'} for word in original['speech']['comparison']['words'])
    record = {'status': 'running', 'scope': 'real local HTTP audio -> immediate provisional review -> unchanged no-op -> changed reply -> punctuation no-op',
              'source_audio': str(source.relative_to(ROOT)), 'training': False, 'paid_api_calls': False,
              'initial': original, 'initial_stages': stages, 'revisions': [], 'unchanged_checks': []}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Save the report inside the workspace')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.feedback_first_only:
        assert timelines[0]['feedback_ready_sec'] is not None and timelines[0]['first_text_sec'] is not None
        record.update(status='passed', feedback_first=timelines[0],
                      scope='One real local recording; terminal pronunciation/writing feedback before first bot text',
                      limits=['Polling gives upper bounds on observed stage timing, not token-level performance.',
                              'No TTS call or claim of grammar accuracy is made by this ordering check.'])
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Feedback completed before first bot text:', timelines[0]['feedback_ready_sec'], timelines[0]['first_text_sec'], flush=True)
        print(output, flush=True)
        return
    def same_words(before, text, phase):
        same = api('/api/chat/reference', {'job_id': before['id'], 'reference_text': text})
        assert same['reference_update'] == {'changed': False, 'regenerated_reply': False}
        assert same['status'] == 'ready' and same['text'] == before['text']
        assert same['metadata']['reference_revision'] == before['metadata']['reference_revision']
        assert same['voice'] == before['voice']
        assert same['speech']['transcript'] == original['speech']['transcript']
        assert same['speech']['phonemes'] == original['speech']['phonemes']
        record['unchanged_checks'].append({'phase': phase, 'text': text, 'response': same})
        print('No generation:', phase, flush=True)
        return same

    same_words(original, original['speech']['transcript'], 'initial_same_words')
    for revision, text in enumerate((case['reference_text'],), 1):
        accepted = api('/api/chat/reference', {'job_id': original['id'], 'reference_text': text})
        assert accepted['status'] == 'generating' and accepted['text'] == ''
        assert accepted['reference_update'] == {'changed': True, 'regenerated_reply': True}
        assert accepted['voice']['status'] == 'not_requested'
        assert accepted['metadata']['reference_revision'] == revision
        assert accepted['speech']['conversation_text'] == text
        final, sequence = wait(accepted, revision)
        assert not set(sequence) & {'transcribing', 'analyzing'}
        assert final['speech']['conversation_text'] == text
        assert final['metadata']['regenerated_from_reference'] is True
        assert final['speech']['transcript'] == original['speech']['transcript']
        assert final['speech']['phonemes'] == original['speech']['phonemes']
        assert final['speech']['audio'] == original['speech']['audio']
        assert not final['speech']['transcript_clarity']['needs_clarification']
        assert any(word['text'].strip('.,!?').casefold() == 'forces' for word in final['speech']['sound_candidates']['words'])
        assert final['text'].startswith('Sorry for the misunderstanding. ')
        assert final['text'].count('Sorry for the misunderstanding.') == 1
        from mdd.demo_chat import spoken_text_key
        visible_words = spoken_text_key(final['text'])
        target_words = spoken_text_key(text)
        assert not any(visible_words[index:index + len(target_words)] == target_words
                       for index in range(len(visible_words) - len(target_words) + 1))
        assert '_acoustic_evidence' not in json.dumps(final)
        assert final['voice']['status'] == 'not_requested'
        record['revisions'].append({'confirmed_text': text, 'accepted': accepted, 'final': final, 'stages': sequence})
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Candidate retained; new reply:', final['text'], flush=True)
    same_words(final, final['speech']['reference_text'].upper().rstrip('.!?') + '!', 'punctuation_and_case_only')
    record['status'] = 'passed'
    record['limits'] = ['Conversation quality is not certified by this integration check.',
                       'No ai33 request or physical speaker/microphone test was performed.']
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
