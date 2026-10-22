"""Five bounded real-Qwen conversation checks; ASR/audio/completion are fixtures.

Every dialogue attempt, including the one allowed production retry, uses the
actual local base Qwen3-0.6B CPU worker. This is a text-pipeline regression, not
an audio/pronunciation/TTS test, and never substitutes authored dialogue text.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts import local_chat_process
from scripts.check_order_conversation_live import analyzer_fixture, learner_sentence_echo
from scripts.local_speech_engine import LocalSpeechEngine
from scripts.conversation_policy import _empty_contribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated/chat-recovery-qa/live-summary.json')
    parser.add_argument('--focused-recovery', action='store_true', help='Three focused work/context/unclear-ASR cases; at most six real calls.')
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {'started_at': datetime.now(timezone.utc).isoformat(),
        'model_id': 'qwen-base', 'model': 'Qwen3-0.6B', 'device': 'cpu', 'adapter': None,
        'training': False, 'network_api_calls': False, 'audio_inference': False, 'tts': False,
        'completion': 'unchanged-text fixture, not a Qwen generation',
        'max_real_calls_per_case': 2, 'max_real_calls_total': 6 if args.focused_recovery else 10,
        'model_calls': [], 'cases': [], 'checks': [], 'qualitative_review_required': True,
        'limitations': ['This exercises real LocalSpeechEngine.run dialogue/guard/retry with explicitly stubbed acoustic and completion stages.',
            'Initial scenario history is a fixed test fixture; subsequent work turns use exact previous visible model/pipeline replies.',
            'The reported unclear-ASR phrase is exact, but the original recording/topic/history were not supplied to this probe.',
            'One stochastic response per case/attempt cannot establish conversational reliability.']}

    def save():
        summary['elapsed_sec'] = round(time.monotonic() - started, 3)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    class RealDialogue:
        def __init__(self, label, utterance):
            self.label, self.utterance, self.calls = label, utterance, 0

        def generate(self, request, cancelled, emit):
            if request.get('generation_profile') != 'conversation':
                payload = json.loads(request['messages'][-1]['content'])
                if payload.get('transcript') != self.utterance:
                    raise RuntimeError('Unexpected non-dialogue generation; no additional model was called')
                emit({'type': 'chunk', 'text': json.dumps({'corrected': self.utterance,
                    'explanation': 'Test fixture: no change to the supplied utterance.'})})
                return {'finish_reason': 'stop', 'source': 'completion_fixture_not_inference'}
            if request.get('model_id') != 'qwen-base' or self.calls >= 2 or len(summary['model_calls']) >= summary['max_real_calls_total']:
                raise RuntimeError('Base-model or bounded-generation constraint violated')
            self.calls += 1
            record = {'case': self.label, 'attempt': self.calls, 'request': deepcopy(request),
                      'raw_chunks': [], 'raw_output': '', 'metadata': None}
            summary['model_calls'].append(record)
            print(f'{self.label}: real Qwen attempt {self.calls}', flush=True)
            then = time.monotonic()

            def receive(text):
                record['raw_chunks'].append(text)
                record['raw_output'] += text
                emit({'type': 'chunk', 'text': text})

            try:
                metadata = local_chat_process.complete(request['messages'],
                    {'device': 'cpu', 'adapter': None, 'generation_profile': 'conversation'},
                    request['max_tokens'], request['timeout'], on_text=receive, cancelled=cancelled)
                record['metadata'] = deepcopy(metadata)
                return metadata
            finally:
                record['elapsed_sec'] = round(time.monotonic() - then, 3)
                save()
                print(f"  raw={record['raw_output']!r}", flush=True)

    work_thread = [{'role': 'user', 'content': 'I started a new job last month.'},
                   {'role': 'assistant', 'content': 'What do you work on most days?'}]
    cases = [
        {'label': 'cafe_known_order', 'topic_id': 'food', 'history': [],
         'utterance': 'I would like to order one coffee and one milk tea please',
         'review': 'Does it ask a useful missing drink detail instead of greeting only, repeating the order, or inventing cafe rules?'},
        {'label': 'work_specific_utterance', 'topic_id': 'work', 'history': work_thread,
         'utterance': 'I am a software developer, and I spend most of my day fixing bugs.',
         'review': 'Does it respond specifically to debugging work, with a new thought or useful question, without restating the sentence?'},
        {'label': 'work_follow_up_with_actual_previous_reply', 'topic_id': 'work', 'history': None,
         'utterance': "The hardest part is understanding someone else's code.",
         'review': 'Does it connect to the actual recent software/developer exchange rather than start the scenario again?'},
        {'label': 'learner_changes_topic_to_travel', 'topic_id': 'work', 'history': None,
         'utterance': "Actually, I'm visiting Da Nang next week.",
         'review': 'Does it follow the new travel subject without returning to work or inventing a shared meeting?'},
        {'label': 'reported_unclear_asr_phrase', 'topic_id': None, 'history': [],
         'utterance': 'It was the divicote true getesir',
         'missing_words': ['divicote', 'getesir'],
         'review': 'Does it ask what the unclear words mean instead of replying only Hi or silently guessing what the person intended?',
         'reported_job_id': '657247e2aae5d61563530e3deaf070c7'},
    ]
    if args.focused_recovery:
        cases = [case for case in cases if case['label'] in {
            'work_specific_utterance', 'work_follow_up_with_actual_previous_reply', 'reported_unclear_asr_phrase'}]
    try:
        for case in cases:
            history = deepcopy(work_thread if case['history'] is None else case['history'])
            request = {'model_id': 'qwen-base', 'topic_id': case['topic_id'], 'history': history,
                       'rewrite_enabled': False,
                       'audio_base64': base64.b64encode(b'TEXT_FIXTURE_NOT_AUDIO').decode('ascii')}
            engine = object.__new__(LocalSpeechEngine)
            engine.analyze = lambda audio, reference=None, text=case['utterance'], missing=case.get('missing_words', []): {
                **analyzer_fixture(text), 'missing_words': list(missing)}
            qwen, events = RealDialogue(case['label'], case['utterance']), []
            metadata = engine.run(request, threading.Event(), lambda value: events.append(deepcopy(value)), qwen)
            visible = ''.join(value['text'] for value in events if value['type'] == 'chunk').strip()
            record = {'label': case['label'], 'transcript_fixture': case['utterance'], 'history': history,
                      'visible_output': visible, 'events': events, 'pipeline_metadata': deepcopy(metadata),
                      'real_model_calls': qwen.calls,
                      'qualitative_review': {'status': 'pending', 'look_for': case['review']}}
            if case.get('reported_job_id'):
                record['reported_job_id'] = case['reported_job_id']
            summary['cases'].append(record)
            summary['checks'].append({'case': case['label'],
                'not_greeting_only': bool(visible) and not _empty_contribution(visible),
                'no_complete_learner_sentence_echo': not learner_sentence_echo(visible, case['utterance']),
                'bounded_real_model_calls': 1 <= qwen.calls <= 2,
                'real_model_reply_not_authored_fallback': metadata.get('reply_source') != 'conversation_unavailable'})
            if case['label'].startswith('work_'):
                work_thread = [*history, {'role': 'user', 'content': case['utterance']},
                               {'role': 'assistant', 'content': visible}]
            print(f"  visible={visible!r}; source={metadata.get('reply_source', 'qwen')}", flush=True)
            save()
        summary['status'] = 'completed'
        summary['narrow_checks_passed'] = all(all(value for name, value in row.items() if name != 'case')
                                               for row in summary['checks'])
    except Exception as error:
        summary.update(status='error', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        summary['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        local_chat_process.shutdown()
        print(f'Report: {output}', flush=True)
    return 0 if summary['narrow_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
