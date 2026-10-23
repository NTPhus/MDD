"""Run local Qwen intent-recovery diagnostics on text fixtures, without audio or TTS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from scripts.transcript_normalization import WORDS
from scripts.transcript_recovery import VERSION, recover


def word_key(text):
    return [word.casefold().replace('’', "'") for word in WORDS.findall(text or '')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly run local model inference; never starts training.')
    parser.add_argument('--cases', type=Path, default=ROOT / 'configs/transcript_recovery_eval.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'auto', 'cuda'], default='auto')
    args = parser.parse_args()
    if not args.run:
        parser.error('Pass --run to execute the local model.')
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'results.json'
    if destination.exists():
        parser.error('Choose a new output folder so previous diagnostics remain reviewable.')
    config = json.loads(args.cases.read_text(encoding='utf-8'))
    from scripts.local_chat_worker import QwenChatEngine
    engine = QwenChatEngine(device=args.device)
    cancelled = threading.Event()

    class RecordedQwen:
        def generate(self, request, event, emit):
            self.request = request
            self.chunks = []
            def record(value):
                if value.get('type') == 'chunk': self.chunks.append(value['text'])
                emit(value)
            self.metadata = engine.generate(request, event, record)
            return self.metadata
    qwen = RecordedQwen()
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': config['description'], 'version': VERSION, 'model_id': 'qwen-base',
              'config_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ROOT / 'scripts/transcript_recovery.py').read_bytes()).hexdigest(),
              'training_started': False, 'external_api_calls': 0, 'audio_inputs': 0, 'cases': []}
    for row in config['cases']:
        result = recover(row['transcript'], qwen, cancelled, history=row.get('history', []))
        expected = row['expected_text']
        value = {**row, 'result': result,
                 'text_matches_expected': word_key(result['text']) == word_key(expected) if expected is not None else None,
                 'status_matches_expected': result['status'] == row['expected_status'],
                 'raw_model_output': ''.join(getattr(qwen, 'chunks', [])),
                 'generation': getattr(qwen, 'metadata', {})}
        report['cases'].append(value)
        report['runtime'] = engine.runtime
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'id': row['id'], 'status': result['status'], 'text': result['text'],
                          'reason': result['reason'], 'seconds': result['latency_sec'],
                          'text_matches_expected': value['text_matches_expected']}, ensure_ascii=False), flush=True)
    # A small, exposed development fixture set; these counts do not estimate population accuracy.
    report['summary'] = {'cases': len(report['cases']),
        'selected': sum(row['result']['status'] in {'accepted', 'unchanged'} for row in report['cases']),
        'exact_target_matches': sum(row['text_matches_expected'] is True for row in report['cases']),
        'expected_ambiguous': sum(row['expected_text'] is None for row in report['cases']),
        'status_matches': sum(row['status_matches_expected'] for row in report['cases'])}
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
