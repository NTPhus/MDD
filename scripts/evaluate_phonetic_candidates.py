"""Measure audio -> ASR/phones -> sound-neighbour retrieval, inference only.

Case JSON contains audio, hint and hint_source. Hints never enter recognition or
retrieval; they are used only afterwards for a labelled diagnostic comparison.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/generated/phonetic-candidates-v1')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--reassess-only', action='store_true', help='Reuse saved ASR/retrieval; extract B posteriors for newly confirmed targets')
    parser.add_argument('--retrieve-only', action='store_true', help='Reuse saved ASR/phones/assessments; rerun dictionary retrieval without neural inference')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not 1 <= len(cases) <= 20: parser.error('Provide 1..20 cases')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT): parser.error('Output must remain inside workspace')
    identities = []
    for case in cases:
        path = Path(case['audio']).resolve()
        identities.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'hint': case.get('hint'), 'hint_source': case.get('hint_source', 'unconfirmed')})
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'inputs': identities, 'inference': 'local CPU only', 'models': 'wav2vec2-base-960h and frozen phoneme-B',
        'code_sha256': hashlib.sha256((ROOT / 'src/mdd/phonetic_candidates.py').read_bytes()).hexdigest(),
        'hints_used_in_recognition_or_retrieval': False, 'training': False, 'external_api_calls': 0,
        'limitations': ['ASR word boundaries and heard phones are both model predictions.',
            'Phoneme edit distance is not a calibrated probability or a confirmed pronunciation error.',
            'Single-word retrieval cannot reliably split a merged word or resolve homophones and names.',
            'Confirmed intended text is not a human annotation of the sounds actually produced.',
            'No human phone labels were supplied; pronunciation precision/recall cannot be measured.']}
    if not args.run: print(json.dumps(protocol, indent=2)); return
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    from scripts.local_speech_engine import LocalSpeechEngine
    from mdd.chat_reference import reference_resolution
    from mdd.chat_reference import reassess_reference
    from mdd.acoustic_evidence import pack_evidence
    from mdd.demo_reference import dictionary_reference
    from mdd.demo_inference import canonical_comparison
    from scripts.evaluate_transcript_normalization import distance, words
    engine, records = LocalSpeechEngine(), []
    for index, (case, identity) in enumerate(zip(cases, identities)):
        started = time.monotonic()
        # Crucially, no hint/reference argument is passed to analyze().
        if args.reassess_only or args.retrieve_only:
            saved = json.loads((output / f'sample-{index + 1}.json').read_text(encoding='utf-8'))
            if saved['record']['sha256'] != identity['sha256']: raise ValueError('Audio changed since original retrieval')
            speech = saved['speech']
            if args.retrieve_only:
                from mdd.phonetic_candidates import sound_candidates
                intervals = [{'text': row['asr_word'], 'start_sec': row['start_sec'], 'end_sec': row['end_sec']}
                             for row in speech['phonetic_candidates']['words']]
                speech['phonetic_candidates'] = sound_candidates(ROOT, intervals, speech['phonemes'], speech['transcript'])
            else:
                evidence = engine.phonemes.posterior_evidence({'id': 'cascade-B', 'label': 'wav2vec2 B', 'kind': 'phoneme', 'experimental': False,
                    'path': ROOT / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint'}, Path(case['audio']).read_bytes())
                speech['_acoustic_evidence'] = pack_evidence(evidence)
        else:
            speech = engine.analyze(Path(case['audio']).read_bytes())
        hint = case.get('hint')
        comparison = None
        confirmed = None
        if hint:
            reference, missing = dictionary_reference(ROOT, hint)
            if reference is not None:
                comparison = canonical_comparison(hint, reference, [row['phone'] for row in speech['phonemes']])
            if case.get('hint_source') == 'user_confirmed_in_chat':
                if args.retrieve_only:
                    if saved['record']['hint'] != hint: raise ValueError('Reference changed; use --reassess-only first')
                    confirmed = saved['record'].get('confirmed_assessment')
                else:
                    assessed = reassess_reference(ROOT, speech, hint)
                    confirmed = {key: assessed.get(key) for key in ('reference_text', 'reference_source', 'assessment', 'comparison', 'sound_candidates', 'target_words')}
        speech = {key: value for key, value in speech.items() if not key.startswith('_')}
        spell = reference_resolution(ROOT, speech['transcript'])
        record = {**identity, 'transcript': speech['transcript'], 'spelling_draft': spell['suggested_text'],
            'dictionary_checks': spell['word_checks'], 'spelling_suggestions': spell['suggestions'],
            'phonetic_candidates': speech['phonetic_candidates'], 'phones': speech['phonemes'],
            'hint_phone_comparison': comparison,
            'confirmed_assessment': confirmed,
            'wer_vs_hint': distance(words(hint), words(speech['transcript'])) / max(1, len(words(hint))) if hint else None,
            'seconds': round(time.monotonic() - started, 3)}
        records.append(record)
        (output / f'sample-{index + 1}.json').write_text(json.dumps({'record': record, 'speech': speech}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'file': identity['file'], 'transcript': speech['transcript'],
            'phonetic_status': speech['phonetic_candidates']['status'], 'seconds': record['seconds']}, ensure_ascii=False), flush=True)
    payload = {'protocol': protocol, 'records': records}
    (output / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for record in records:
        for word in record['phonetic_candidates']['words']:
            rows.append({'file': record['file'], 'asr_word': word['asr_word'], 'heard_phones': ' '.join(word['heard_phones']),
                'candidates': ' / '.join(candidate['word'] for candidate in word['candidates']),
                'distances': ' / '.join(str(candidate['edit_distance']) for candidate in word['candidates'])})
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(row[key])) + '</td>' for key in ('file', 'asr_word', 'heard_phones', 'candidates', 'distances')) + '</tr>' for row in rows)
    diagnoses = [{'file': row['file'], 'intended_text': row['hint'],
        'review_candidates': (row['confirmed_assessment'] or {}).get('sound_candidates'),
        'word_assessment': (row['confirmed_assessment'] or {}).get('comparison')}
        for row in records if row.get('confirmed_assessment')]
    summary = [{'file': row['file'], 'hint': row['hint'], 'hint_source': row['hint_source'],
                'transcript': row['transcript'], 'spelling_draft': row['spelling_draft'], 'wer_vs_hint': row['wer_vs_hint']} for row in records]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh([f'Sample {index+1}' for index in range(len(records))], [row['wer_vs_hint'] * 100 if row['wer_vs_hint'] is not None else 0 for row in records], color='#7c3aed')
    ax.set_xlabel('Word error rate vs intended-text hint (%) — not pronunciation accuracy')
    fig.tight_layout(); fig.savefig(output / 'wer-vs-hint.png', dpi=150); plt.close(fig)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Sound-neighbour retrieval</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#30213e}td,th{padding:10px;border:1px solid #ddd;vertical-align:top}table{border-collapse:collapse}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<h1>Sound-neighbour retrieval: three recordings</h1><p>' + html.escape(' '.join(protocol['limitations']))
        + '</p><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre><img src="wer-vs-hint.png">'
        + '<table><tr><th>File</th><th>ASR word</th><th>Predicted phones</th><th>Nearest phone candidates</th><th>Phone edits</th></tr>' + table + '</table>'
        + '<h2>Confirmed intended text: acoustic review (unvalidated)</h2><pre>' + html.escape(json.dumps(diagnoses, ensure_ascii=False, indent=2)) + '</pre>', encoding='utf-8')
    import nbformat
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Sound-neighbour retrieval\nAudio → ASR word timing + phoneme-B → CMUdict sound neighbours. No training or API calls.\n\n' + '\n'.join('- ' + item for item in protocol['limitations'])),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent\nOUT = ROOT / " + repr(output.relative_to(ROOT).as_posix()) + "\nresults = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))"),
        nbformat.v4.new_code_cell("display(pd.DataFrame([{k:r[k] for k in ['file','hint','hint_source','transcript','spelling_draft','wer_vs_hint','seconds']} for r in results['records']]))"),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT / 'wer-vs-hint.png')))"),
        nbformat.v4.new_code_cell("rows = []\nfor record in results['records']:\n    for word in record['phonetic_candidates']['words']:\n        rows.append({'file':record['file'],'ASR':word['asr_word'],'predicted_phones':' '.join(word['heard_phones']),'candidates':[(c['word'],c['edit_distance']) for c in word['candidates']]})\ndisplay(pd.DataFrame(rows))"),
        nbformat.v4.new_code_cell("for record in results['records']:\n    assessment = record.get('confirmed_assessment')\n    if assessment:\n        print(record['file'], '— intended:', record['hint'])\n        display(pd.DataFrame((assessment.get('sound_candidates') or {}).get('words', [])))\n        display(assessment.get('assessment'))"),
        nbformat.v4.new_code_cell("display(results['protocol'])"),
    ])
    nbformat.write(notebook, ROOT / 'notebooks/phonetic_candidates_validation.ipynb')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
