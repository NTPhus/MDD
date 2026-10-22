"""Measure local ASR -> lexical-preserving normalization; never starts training.

Controls are development cases. Audio is a deterministic sample of an existing
held-out L2-ARCTIC-plus manifest; its published reference is the reading prompt,
not a human verification of every word actually spoken. No API/TTS calls.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.transcript_normalization import VERSION, normalization_request, parse_normalization, preservation_reason


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*|\d+(?:[.:]\d+)*", text.lower().replace('’', "'"))


def distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        updated = [i]
        for j, b in enumerate(right, 1):
            updated.append(min(updated[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = updated
    return row[-1]


def aggregate(rows):
    count = sum(len(words(row['reference'])) for row in rows)
    result = {'sentences': len(rows), 'reference_words': count,
              'statuses': dict(Counter(row['normalization']['status'] for row in rows)),
              'changed': sum(row['transcript'] != row['normalized'] for row in rows),
              'guard_violations': sum(preservation_reason(row['transcript'], row['normalized']) is not None for row in rows)}
    for field, label in [('transcript', 'before'), ('proposal', 'qwen_unguarded'), ('normalized', 'after')]:
        errors = sum(distance(words(row['reference']), words(row[field])) for row in rows)
        char_errors = sum(distance(' '.join(words(row['reference'])), ' '.join(words(row[field]))) for row in rows)
        chars = sum(len(' '.join(words(row['reference']))) for row in rows)
        result['wer_' + label] = errors / max(count, 1)
        result['cer_' + label] = char_errors / max(chars, 1)
    changes = [distance(words(row['reference']), words(row['normalized'])) - distance(words(row['reference']), words(row['transcript'])) for row in rows]
    result.update(improved=sum(value < 0 for value in changes), worsened=sum(value > 0 for value in changes),
                  same_word_errors=sum(value == 0 for value in changes),
                  correct_transcripts_damaged=sum(words(row['transcript']) == words(row['reference'])
                      and words(row['normalized']) != words(row['reference']) for row in rows),
                  mean_normalization_sec=sum(row['normalization']['latency_sec'] for row in rows) / max(len(rows), 1))
    return result


def prepare(limit):
    config_path = ROOT / 'configs/transcript_normalization_eval.json'
    controls = json.loads(config_path.read_text(encoding='utf-8'))['controls']
    manifest = ROOT / 'data/processed/feedback-projector-en-v1/test.jsonl'
    candidates = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    groups = {}
    for row in candidates:
        if row['duration_sec'] <= 20:
            groups.setdefault(row['speaker_id'], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: hashlib.sha256(row['utterance_id'].encode()).hexdigest())
    audio = []
    while len(audio) < limit:
        before = len(audio)
        for speaker in sorted(groups):
            if groups[speaker] and len(audio) < limit:
                row = groups[speaker].pop(0)
                path = ROOT / row['audio_path']
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['audio_sha256']:
                    raise ValueError('Audio missing or checksum mismatch: ' + row['utterance_id'])
                audio.append({**row, 'id': row['utterance_id'], 'reference': row['reference_text']})
        if len(audio) == before: break
    if len(audio) != limit: raise ValueError('Not enough held-out audio')
    protocol = {'version': VERSION, 'controls_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'normalizer_sha256': hashlib.sha256((ROOT / 'scripts/transcript_normalization.py').read_bytes()).hexdigest(),
        'source_manifest': str(manifest.relative_to(ROOT)), 'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'audio_ids': [row['id'] for row in audio], 'audio_limit': limit,
        'selection': 'speaker round robin, ordered by SHA256(utterance_id), duration <=20s; fixed before inference',
        'normalization_policy': 'No free lexical substitution; only orthography, closed apostrophe/compound rules',
        'models': {}, 'target_use': 'The published reference is NEVER provided to the normalizer.',
        'limitations': ['Development controls are authored regression examples, not independent accuracy evidence.',
            'Audio reading prompts can differ from what a learner actually said.',
            'Small exploratory sample, not a calibration set or evidence of conversational ASR reliability.',
            'WER/CER ignore capitalization/punctuation; guard checks are invariants, not semantic human judgments.',
            'No training, threshold fitting, external API, LLM judge, or paid TTS.']}
    for model in ('Qwen3-0.6B', 'wav2vec2-base-960h'):
        path = ROOT / 'models/pretrained' / model / 'model.safetensors'
        protocol['models'][model] = {'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'device': 'cpu', 'adapter': None}
    return controls, audio, protocol


def report(output, protocol, records):
    summary = {group: aggregate([row for row in records if row['group'] == group]) for group in ('controls', 'audio')}
    payload = {'protocol': protocol, 'summary': summary, 'records': records}
    (output / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (group, values) in zip(axes, summary.items()):
        ax.bar(['Raw ASR', 'Qwen draft', 'Guarded'], [values['wer_before'] * 100, values['wer_qwen_unguarded'] * 100,
            values['wer_after'] * 100], color=['#94a3b8', '#f59e0b', '#7c3aed'])
        ax.set_title(f"{group.title()} (n={values['sentences']})"); ax.set_ylabel('WER (%)')
    fig.tight_layout(); fig.savefig(output / 'wer.png', dpi=150); plt.close(fig)
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(row[key])) + '</td>'
        for key in ('group', 'id', 'transcript', 'proposal', 'normalized', 'reference')) + '</tr>' for row in records)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Transcript normalization</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#312342}td,th{padding:10px;border:1px solid #ddd;vertical-align:top}table{border-collapse:collapse}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<h1>Transcript normalization — exploratory measurement</h1><p>No training. Lexical content is preserved; this is not grammar correction.</p>'
        '<img src="wer.png"><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre><p>'
        + html.escape(' '.join(protocol['limitations'])) + '</p><table><tr><th>Group</th><th>ID</th><th>Raw</th><th>Qwen draft</th><th>Accepted</th><th>Reference</th></tr>'
        + table + '</table>', encoding='utf-8')
    import nbformat
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Transcript normalization\nExploratory measurements; inference only. No training or API calls.\n\n' + '\n'.join('- ' + item for item in protocol['limitations'])),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent\nOUT = ROOT / " + repr(str(output.relative_to(ROOT)).replace('\\', '/')) + "\nresults = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))"),
        nbformat.v4.new_code_cell("display(pd.DataFrame(results['summary']).T)"),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT / 'wer.png')))"),
        nbformat.v4.new_code_cell("rows = pd.DataFrame(results['records'])\ndisplay(rows[['group', 'id', 'transcript', 'proposal', 'normalized', 'reference']])"),
        nbformat.v4.new_code_cell("display(results['protocol'])"),
    ])
    nbformat.write(notebook, ROOT / 'notebooks/transcript_normalization_validation.ipynb')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly run CPU inference; never trains')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/generated/transcript-normalization-v1')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100: parser.error('limit must be 1..100')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT): parser.error('output must remain inside the workspace')
    controls, audio, protocol = prepare(args.limit)
    output.mkdir(parents=True, exist_ok=True)
    if not args.run:
        print(json.dumps({'ready': True, 'controls': len(controls), 'audio': len(audio), 'instruction': 'Add --run to measure CPU inference; this never starts training.'})); return
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Existing frozen protocol differs; choose a new output directory')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    from scripts.local_chat_worker import QwenChatEngine
    from scripts.local_speech_engine import LocalSpeechEngine
    import soundfile as sf
    import scipy.signal
    import math
    qwen = QwenChatEngine(); qwen.load(False)
    asr = object.__new__(LocalSpeechEngine); asr.asr = asr.processor = None
    cancelled, records = threading.Event(), []
    cache = output / 'records'; cache.mkdir(exist_ok=True)
    for group, rows in [('controls', controls), ('audio', audio)]:
        for row in rows:
            filename = cache / (group + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:20] + '.json')
            if filename.exists():
                records.append(json.loads(filename.read_text(encoding='utf-8'))); continue
            raw = row.get('transcript')
            if raw is None:
                waveform, rate = sf.read(ROOT / row['audio_path'], dtype='float32')
                if waveform.ndim != 1: waveform = waveform.mean(axis=1)
                if rate != 16000:
                    divisor = math.gcd(rate, 16000); waveform = scipy.signal.resample_poly(waveform, 16000 // divisor, rate // divisor)
                raw = asr.transcribe(waveform)
            chunks, started = [], time.monotonic()
            metadata = qwen.generate(normalization_request(raw, row.get('history', [])), cancelled,
                lambda event: chunks.append(event['text']) if event['type'] == 'chunk' else None)
            response = ''.join(chunks)
            normalized = parse_normalization(response, metadata['finish_reason'] == 'stop', raw)
            normalized['latency_sec'] = round(time.monotonic() - started, 3)
            try: proposal = json.loads(response).get('normalized_text', raw)
            except (ValueError, AttributeError): proposal = raw
            if not isinstance(proposal, str): proposal = raw
            record = {'id': row['id'], 'group': group, 'transcript': raw, 'proposal': proposal,
                'normalized': normalized['text'], 'reference': row['reference'], 'normalization': normalized,
                'raw_output': response, 'finish_reason': metadata['finish_reason'], 'note': row.get('note'),
                'audio_sha256': row.get('audio_sha256')}
            filename.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8'); records.append(record)
            print(json.dumps({'group': group, 'id': row['id'], 'status': normalized['status'], 'reason': normalized['reason'],
                'sec': normalized['latency_sec']}, ensure_ascii=False), flush=True)
    print(json.dumps(report(output, protocol, records), indent=2), flush=True)


if __name__ == '__main__': main()
