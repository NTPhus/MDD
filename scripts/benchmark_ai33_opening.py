"""Measure one authored topic opening with ai33, then repeat its cached audio.

Requires --run. At most ONE new paid TTS task may be submitted. Existing cache
and task receipts are preserved. No learner audio, transcript, LLM or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from mdd.speech_topics import TOPICS
from scripts.ai33_tts import AI33Error, AI33TTS, load_env


class MeasuredClient(AI33TTS):
    """Limit submission count even if a benchmark accidentally calls twice."""
    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self.posts = 0
        self.events = []

    def _request_once(self, method, path, payload=None, timeout=30):
        if method == 'POST':
            if path != '/v3/text-to-speech' or self.posts >= 1:
                raise AI33Error('The one-task benchmark budget has been reached.', code='benchmark_task_budget')
            self.posts += 1
            stage = 'submit_task'
        elif method == 'GET' and path.startswith('/v1/task/'):
            stage = 'poll_task'
        else:
            raise AI33Error('Unsupported benchmark request.', code='benchmark_route')
        started = time.perf_counter()
        try:
            return super()._request_once(method, path, payload, timeout)
        finally:
            event = {'stage': stage, 'elapsed_sec': round(time.perf_counter() - started, 4)}
            self.events.append(event)
            print(json.dumps(event), flush=True)


def cache_state(client, text):
    _, receipt_path, _, _ = client._cache_request(text, 'en-US', speed=client.speed)
    if not receipt_path.is_file():
        return 'not_submitted'
    try:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        name = receipt.get('audio_file')
        if isinstance(name, str) and Path(name).name == name:
            audio = client.cache_dir / name
            if audio.is_file() and hashlib.sha256(audio.read_bytes()).hexdigest() == receipt.get('audio_sha256'):
                return 'audio_cached'
        if isinstance(receipt.get('task_id'), str) and receipt['task_id'].strip():
            return 'task_receipt'
    except (OSError, ValueError):
        pass
    raise AI33Error('An existing cache receipt needs inspection; it will not be overwritten.', code='benchmark_cache_invalid')


def run_probe(topic='auto', output=None):
    output = Path(output or ROOT / 'data/generated/ai33-opening-latency-v1')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'summary.json'
    report = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'max_new_tasks': 1, 'runs': [], 'scope': 'One authored opening, then exact cached repeat.',
              'learner_audio_sent': False, 'llm_calls': 0, 'training_started': False}
    client = None
    try:
        load_env(ROOT / '.env')
        client = MeasuredClient(ROOT / 'data/generated/tts/ai33/chat')
        states = {key: cache_state(client, item['opening']) for key, item in TOPICS.items()}
        if topic == 'auto':
            topic = next((key for key, status in states.items() if status == 'not_submitted'), 'food')
        text = TOPICS[topic]['opening']
        _, _, _, voice = client._cache_request(text, 'en-US', speed=client.speed)
        report.update(topic_id=topic, text=text, voice_id=voice, speed=client.speed,
                      provider_poll_interval_sec=client.interval, cache_before=states[topic])
        for index in (1, 2):
            before_posts, before_events = client.posts, len(client.events)
            started = time.perf_counter()
            audio = client.synthesize(text, locale='en-US', speed=client.speed)
            measured = {'index': index, 'label': 'initial call' if index == 1 else 'cached repeat',
                        'elapsed_sec': round(time.perf_counter() - started, 4),
                        'new_tasks_submitted': client.posts - before_posts,
                        'audio_bytes': audio.stat().st_size,
                        'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                        'timing': dict(client.last_timing), 'requests': client.events[before_events:]}
            try:
                import soundfile as sf
                measured['audio_duration_sec'] = round(sf.info(str(audio)).duration, 3)
            except (ImportError, RuntimeError):
                pass
            report['runs'].append(measured)
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(measured, ensure_ascii=False), flush=True)
        assert client.posts <= 1
        assert report['runs'][0]['audio_sha256'] == report['runs'][1]['audio_sha256']
        assert report['runs'][1]['new_tasks_submitted'] == 0
        report['status'] = 'passed'
    except Exception as error:
        code = getattr(error, 'code', None)
        report.update(status='failed', error_code=code if isinstance(code, str) else type(error).__name__)
    finally:
        report['new_task_posts_total'] = client.posts if client else 0
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'summary': str(destination),
                      'new_task_posts_total': report['new_task_posts_total']}), flush=True)
    return 0 if report['status'] == 'passed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Permit at most one new paid ai33 task using local settings')
    parser.add_argument('--topic', choices=['auto', *TOPICS], default='auto',
                        help='auto prefers an authored opening not submitted previously')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/generated/ai33-opening-latency-v1')
    args = parser.parse_args(argv)
    if not args.run:
        parser.error('--run is required for a real ai33 measurement')
    return run_probe(args.topic, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
