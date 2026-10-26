"""Frozen local editing comparison: Qwen base/LoRA versus official CoEdIT.

No training, API inference, prompt tuning on test rows, or demo configuration.
Invalid Qwen contracts remain unavailable: text metrics report coverage and an
explicit no-change delivery fallback, never call schema validity grammar accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/model-comparison/python'))
OUT = ROOT / 'data/generated/model-comparison-20261024/grammar'
MODEL = ROOT / 'models/pretrained/coedit-large'
MODEL_ID = 'grammarly/coedit-large'
MODEL_REV = '5637bcdf9d8d4419f97c8cfea36f7d35c79232b6'
DATA_REV = 'e9a255c33ef910bc33a9d2b522653fa87521583e'
WEIGHTS_SHA = 'c692f68bca5c6899801bc9fad626fefd3359ccd95e26dccae4dd72186fd98852'
JFLEG_REV = 'ee06ff806a208aba815ac45313f4e750a48330a5'
PROMPTS = {'grammar_corrected': 'Fix grammatical errors in this sentence: ',
           'natural': 'Make the sentence fluent: '}
ARMS = ('qwen-base', 'qwen-grammar', 'coedit-large')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def canonical(text):
    return ' '.join(re.findall(r'[^\W_]+', text.casefold()))


def fetch(url, destination, expected=None):
    destination = Path(destination)
    if destination.exists() and (expected is None or digest(destination) == expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.download')
    headers = {'User-Agent': 'MDD-local-model-comparison/1.0'}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = 'ab' if offset and response.status == 206 else 'wb'
        with partial.open(mode) as stream:
            last = time.perf_counter()
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                stream.write(chunk)
                if time.perf_counter() - last > 15:
                    print(f'Download {destination.name}: {stream.tell() / 2**20:.0f} MiB', flush=True)
                    last = time.perf_counter()
    if expected and digest(partial) != expected:
        raise ValueError(f'Download checksum mismatch: {destination}')
    partial.replace(destination)


def download():
    for filename in ('README.md', 'config.json', 'generation_config.json', 'special_tokens_map.json',
                     'spiece.model', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors'):
        fetch(f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REV}/{filename}', MODEL / filename,
              WEIGHTS_SHA if filename == 'model.safetensors' else None)
    for filename in ('train.jsonl', 'validation.jsonl'):
        fetch(f'https://huggingface.co/datasets/grammarly/coedit/resolve/{DATA_REV}/{filename}',
              OUT / 'source-audit' / ('coedit-' + filename))
    fetch(f'https://raw.githubusercontent.com/keisks/jfleg/{JFLEG_REV}/eval/gleu.py',
          OUT / 'source-audit/official_jfleg_gleu.py')
    save(MODEL / 'setup_manifest.json', {'repo_id': MODEL_ID, 'revision': MODEL_REV,
         'files': {p.name: {'sha256': digest(p), 'size': p.stat().st_size}
                   for p in MODEL.iterdir() if p.name != 'setup_manifest.json' and p.is_file()},
         'source': 'https://huggingface.co/grammarly/coedit-large',
         'paper': 'https://aclanthology.org/2023.findings-emnlp.350/',
         'license': 'cc-by-nc-4.0'})


def prepare():
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    detok = TreebankWordDetokenizer().detokenize
    test_path = ROOT / 'data/processed/grammar-wi-jfleg-v1/test.jsonl'
    test = read_jsonl(test_path)
    qwen_seen = set()
    for split in ('train', 'validation'):
        for item in read_jsonl(test_path.with_name(split + '.jsonl')):
            qwen_seen.update(canonical(value) for value in [item['source'], *item['references']])
    coedit_seen = set()
    counts = {}
    for split in ('train', 'validation'):
        path = OUT / 'source-audit' / f'coedit-{split}.jsonl'
        data = read_jsonl(path)
        counts[split] = {'rows': len(data), 'sha256': digest(path), 'tasks': dict(Counter(r['task'] for r in data))}
        for row in data:
            # CoEdIT released inputs consist of a verbalizer followed by ': '.
            # Also compare full src and target, conservatively excluding matches.
            coedit_seen.add(canonical(row['src']))
            if ': ' in row['src']:
                coedit_seen.add(canonical(row['src'].split(': ', 1)[1]))
            coedit_seen.add(canonical(row['tgt']))
    excluded, rows = Counter(), []
    eligible_counts = {}
    for task in ('grammar_corrected', 'natural'):
        candidates, seen = [], set()
        for row in test:
            if row['task'] != task:
                continue
            key = canonical(row['source'])
            if key in qwen_seen:
                excluded['qwen_train_or_validation_source'] += 1
            elif key in coedit_seen:
                excluded['coedit_train_or_validation_source_or_target'] += 1
            elif key in seen:
                excluded['duplicate_source'] += 1
            else:
                seen.add(key)
                candidates.append(row)
        eligible_counts[task] = len(candidates)
        candidates.sort(key=lambda row: object_sha('model-comparison-20261024:' + row['id']))
        if len(candidates) < 100:
            raise ValueError(f'Insufficient heldout rows for {task}: {len(candidates)}')
        rows.extend({**r, 'cohort': 'published_heldout'} for r in candidates[:100])
    jfleg_dir = ROOT / 'data/external/qwen-grammar/jfleg/test'
    jfleg_refs = [(jfleg_dir / f'test.ref{i}').read_text(encoding='utf-8').splitlines() for i in range(4)]
    for row in rows:
        if row['corpus'] == 'jfleg':
            index = int(row['source_id'].rsplit(':', 1)[1])
            row['references'] = [detok(ref[index].split()) for ref in jfleg_refs]
    examples = [
        ('usually', 'i usually wakes up at seven am then i brush my teeth and go to work by bus'),
        ('yesterday', 'Yesterday I wake up at 7, I no brush teeth and go to work by bus, when arrive office I eating breakfast.'),
        ('profession', 'Currently I am a software engineer.'),
        ('coffee', 'I would like to order a cup of coffee and one iced tea please.'),
        ('garbled', 'I will wi to order a cupf coffee and one ice ta please'),
        ('nonsense', 'Banana software yesterday purple.'),
        ('negation', 'I did not enjoy the coffee, but I might try the tea.'),
        ('numbers', 'My brother Bob ordered three coffees, not thirteen.'),
    ]
    for name, source in examples:
        for task in ('grammar_corrected', 'natural'):
            rows.append({'id': 'development:' + name + ':' + task, 'source': source, 'task': task,
                         'cohort': 'user_development', 'references': [], 'group_id': name})
    qwen = ROOT / 'models/pretrained/Qwen3-0.6B'
    adapter = ROOT / 'runs/grammar-qwen3-06b-wi-jfleg-v1/adapter'
    manifest = {'version': 1, 'date_local': '2026-10-24', 'selection': '100 per task; SHA256-sorted IDs; no performance selection',
       'dataset': {'source': str(test_path.relative_to(ROOT)), 'sha256': digest(test_path),
                   'eligible_counts': eligible_counts, 'excluded': dict(excluded),
                   'coedit_dataset_revision': DATA_REV, 'coedit_splits': counts},
       'models': {'qwen-base': {'path': str(qwen.relative_to(ROOT)), 'weights_sha256': digest(qwen / 'model.safetensors')},
                  'qwen-grammar': {'path': str(adapter.relative_to(ROOT)), 'weights_sha256': digest(adapter / 'adapter_model.safetensors')},
                  'coedit-large': {'path': str(MODEL.relative_to(ROOT)), 'revision': MODEL_REV,
                                    'weights_sha256': WEIGHTS_SHA}},
       'generation': {'do_sample': False, 'max_new_tokens': 256, 'num_beams': 1, 'use_cache': True},
       'precision': 'bfloat16', 'device': 'cuda:0', 'seed': 42,
       'prompts': {'coedit': PROMPTS, 'qwen': 'scripts.grammar_data.messages; original trained JSON contract'},
       'rows': rows,
       'limitations': ['W&I internal heldout comes from published training corpus; unknown private CoEdIT training or near-duplicates remain possible.',
                      'Exact lexical overlap removed against Qwen training/validation and public CoEdIT training/validation source/target.',
                      'JFLEG official test is a stronger heldout benchmark, but base pretraining contamination is unknown.',
                      'No human semantic or naturalness judgments. Development examples excluded from published metric aggregates.'],
       'evaluator_sha256': digest(__file__), 'qwen_prompt_sha256': digest(ROOT / 'scripts/grammar_data.py'),
       'qwen_parser_sha256': digest(ROOT / 'scripts/evaluate_grammar_model.py'),
       'tokenizer_hashes': {name: {p.name: digest(p) for p in directory.iterdir()
                                  if p.is_file() and (p.name.startswith('tokenizer') or p.name in ('spiece.model', 'vocab.json', 'merges.txt', 'config.json'))}
                            for name, directory in (('qwen', qwen), ('coedit', MODEL))},
       'gleu_sha256': digest(OUT / 'source-audit/official_jfleg_gleu.py')}
    manifest['jfleg_original_four_reference_hashes'] = {f'test.ref{i}': digest(jfleg_dir / f'test.ref{i}') for i in range(4)}
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Existing frozen manifest differs; do not overwrite')
    save(path, manifest)
    print('Frozen rows:', len(rows), 'excluded:', dict(excluded), 'eligible:', eligible_counts, flush=True)


def run(arm):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from scripts.grammar_data import messages
    from scripts.evaluate_grammar_model import parse_output
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['evaluator_sha256'] != digest(__file__):
        raise ValueError('Evaluator differs from frozen manifest')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('This frozen run requires CUDA BF16; do not silently change precision')
    torch.set_num_threads(4)
    set_seed(42)
    path = OUT / arm
    path.mkdir(exist_ok=True)
    predictions_path = path / 'predictions.jsonl'
    prior = read_jsonl(predictions_path) if predictions_path.exists() else []
    if [x['id'] for x in prior] != [x['id'] for x in manifest['rows'][:len(prior)]]:
        raise ValueError('Resume rows do not match manifest prefix')
    for original, observed in zip(manifest['rows'], prior):
        if observed['source_row_sha256'] != object_sha(original):
            raise ValueError('Resume source row differs')
    if (path / 'metadata.json').exists():
        old_metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if old_metadata['manifest_sha256'] != digest(OUT / 'manifest.json'):
            raise ValueError('Resume frozen manifest differs')
    if len(prior) == len(manifest['rows']):
        print('Already complete:', arm)
        return
    model_path = ROOT / manifest['models']['qwen-base' if arm.startswith('qwen') else arm]['path']
    weight = model_path / 'model.safetensors'
    if digest(weight) != manifest['models']['qwen-base' if arm.startswith('qwen') else arm]['weights_sha256']:
        raise ValueError('Frozen model weights differ')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                              use_fast=True)
    factory = AutoModelForCausalLM if arm.startswith('qwen') else AutoModelForSeq2SeqLM
    load_start = time.perf_counter()
    model = factory.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                   dtype=torch.bfloat16, device_map={'': 'cuda:0'})
    if arm == 'qwen-grammar':
        from peft import PeftModel
        adapter = ROOT / manifest['models'][arm]['path']
        if digest(adapter / 'adapter_model.safetensors') != manifest['models'][arm]['weights_sha256']:
            raise ValueError('Frozen adapter weights differ')
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True,
                                         device_map={'': 'cuda:0'}, autocast_adapter_dtype=False)
    model.eval().requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats()
    metadata = {'arm': arm, 'manifest_sha256': digest(OUT / 'manifest.json'),
                'status': 'running', 'device': torch.cuda.get_device_name(),
                'dtype': 'bfloat16', 'parameter_count': sum(p.numel() for p in model.parameters()),
                'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'load_seconds': load_seconds, 'packages': {n: version(n) for n in ('torch', 'transformers', 'peft', 'sacrebleu')},
                'started_at_utc': datetime.now(timezone.utc).isoformat()}
    save(path / 'metadata.json', metadata)
    for index, row in enumerate(manifest['rows'][len(prior):], start=len(prior)):
        if arm.startswith('qwen'):
            prompt = tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda:0')
            prefix = inputs['input_ids'].shape[1]
        else:
            prompt = PROMPTS[row['task']] + row['source']
            inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
            prefix = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **manifest['generation'],
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output_ids = generated[0, prefix:]
        raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        finished = bool(len(output_ids) and int(output_ids[-1]) == tokenizer.eos_token_id)
        if arm.startswith('qwen'):
            prediction, error = parse_output(raw, row['task'], finished)
        else:
            prediction = raw if raw and finished else None
            error = None if prediction else 'empty_or_budget_exhausted'
        observed = {**row, 'source_row_sha256': object_sha(row), 'prompt_sha256': object_sha(prompt),
                    'raw': raw, 'prediction': prediction, 'error': error, 'finished': finished,
                    'input_tokens': int(inputs['input_ids'].shape[1]), 'output_tokens': len(output_ids),
                    'seconds': elapsed, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        with predictions_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(observed, ensure_ascii=False) + '\n')
            stream.flush()
        if (index + 1) % 10 == 0 or index + 1 == len(manifest['rows']):
            print(f'{arm} {index+1}/{len(manifest["rows"])} valid={prediction is not None} sec={elapsed:.3f}', flush=True)
    metadata.update(status='completed', completed_rows=len(manifest['rows']),
                    predictions_sha256=digest(predictions_path),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    finished_at_utc=datetime.now(timezone.utc).isoformat())
    save(path / 'metadata.json', metadata)


def gleu(rows, hypotheses, destination):
    """Run pinned, unmodified JFLEG GLEU (2016), 500 random-reference iterations."""
    spec = importlib.util.spec_from_file_location('mdd_official_jfleg_gleu', OUT / 'source-audit/official_jfleg_gleu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / 'source.txt'
    source.write_text(''.join(r.get('source_tokenized', r['source']) + '\n' for r in rows), encoding='utf-8')
    hypothesis = destination / 'hypothesis.txt'
    # Use one fixed Moses tokenization for all predictions and references; never
    # score bare predictions against whitespace-tokenized published references.
    from sacremoses import MosesTokenizer
    tokenizer = MosesTokenizer(lang='en')
    tok = lambda value: tokenizer.tokenize(value, return_str=True, escape=False)
    source.write_text(''.join(tok(r['source']) + '\n' for r in rows), encoding='utf-8')
    hypothesis.write_text(''.join(tok(h) + '\n' for h in hypotheses), encoding='utf-8')
    paths = []
    count = max(len(r['references']) for r in rows)
    for ref in range(count):
        path = destination / f'reference{ref}.txt'
        path.write_text(''.join(tok(r['references'][min(ref, len(r['references']) - 1)]) + '\n' for r in rows), encoding='utf-8')
        paths.append(str(path))
    calculator = module.GLEU(4)
    calculator.load_sources(str(source))
    calculator.load_references(paths)
    # Original code uses locale-default open; use UTF-8 mode when invoking CLI.
    score = list(calculator.run_iterations(num_iterations=500, hypothesis=str(hypothesis), per_sent=False))[0]
    return {'score': float(score[0]) * 100, 'reference_sampling_std': float(score[1]) * 100,
            'version': 'official JFLEG eval/gleu.py; updated2016; order4; iterations500; Moses en; case-sensitive',
            'scope': 'This selected subset only; not the published full JFLEG leaderboard score.'}


def summarize():
    from sacrebleu.metrics import CHRF
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    result = {'manifest_sha256': digest(OUT / 'manifest.json'), 'models': {}, 'metric_scope':
              'Reference similarity, contract availability, copy controls, and token length diagnostics. '
              'Not human grammar accuracy, factual preservation, or naturalness ratings.',
              'limitations': manifest['limitations'], 'copy_baseline': {}}
    published = [r for r in manifest['rows'] if r['cohort'] == 'published_heldout']
    def quality(rows, hypotheses, arm, task, name):
        if not rows:
            return None
        refs = [[r['references'][min(i, len(r['references']) - 1)] for r in rows]
                for i in range(max(len(r['references']) for r in rows))]
        metric = CHRF()
        value = metric.corpus_score(hypotheses, refs)
        normalized = lambda s: ' '.join(s.split())
        return {'rows': len(rows), 'chrf': value.score, 'chrf_signature': str(metric.get_signature()),
                'exact_reference_rate': sum(normalized(h) in {normalized(t) for t in r['references']}
                                            for r, h in zip(rows, hypotheses)) / len(rows),
                'gleu': gleu(rows, hypotheses, OUT / 'metric-inputs' / arm / task / name)}
    for task in ('grammar_corrected', 'natural'):
        rows = [r for r in published if r['task'] == task]
        result['copy_baseline'][task] = quality(rows, [r['source'] for r in rows], 'copy', task, 'all')
    for arm in ARMS:
        path = OUT / arm
        metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        observations = read_jsonl(path / 'predictions.jsonl')
        if metadata['status'] != 'completed' or metadata['predictions_sha256'] != digest(path / 'predictions.jsonl'):
            raise ValueError('Incomplete or changed predictions: ' + arm)
        if metadata['manifest_sha256'] != digest(OUT / 'manifest.json'):
            raise ValueError('Predictions manifest differs')
        if (metadata['completed_rows'] != len(manifest['rows']) or
                [r['id'] for r in observations] != [r['id'] for r in manifest['rows']]):
            raise ValueError('Incomplete or reordered predictions: ' + arm)
        for original, observed in zip(manifest['rows'], observations):
            if observed['source_row_sha256'] != object_sha(original):
                raise ValueError('Prediction source row differs: ' + arm)
        summary = {'metadata': metadata, 'tasks': {}}
        for task in ('grammar_corrected', 'natural'):
            rows = [r for r in observations if r['task'] == task and r['cohort'] == 'published_heldout']
            valid = [r for r in rows if r['prediction'] is not None]
            controls = [r for r in rows if canonical(r['source']) in {canonical(s) for s in r['references']}]
            summary['tasks'][task] = {'rows': len(rows), 'available_rate': len(valid) / len(rows),
                'contract_errors': dict(Counter(r['error'] for r in rows if r['error'])),
                'quality_on_valid_only': quality(valid, [r['prediction'] for r in valid], arm, task, 'valid'),
                'quality_with_explicit_source_fallback': quality(rows, [r['prediction'] if r['prediction'] is not None else r['source'] for r in rows], arm, task, 'delivery'),
                'unchanged_controls': len(controls),
                'unchanged_controls_preserved_exact': sum(r['prediction'] == r['source'] for r in controls),
                'unchanged_controls_preserved_lexical': sum(r['prediction'] is not None and canonical(r['prediction']) == canonical(r['source']) for r in controls),
                'mean_seconds': sum(r['seconds'] for r in rows) / len(rows),
                'p50_seconds': sorted(r['seconds'] for r in rows)[len(rows)//2],
                'changed_lexical_rate_on_valid': sum(canonical(r['prediction']) != canonical(r['source']) for r in valid) / len(valid) if valid else None}
        summary['development'] = [r for r in observations if r['cohort'] == 'user_development']
        result['models'][arm] = summary
    save(OUT / 'summary.json', result)
    print('Summary:', OUT / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('download', 'prepare', 'run', 'summarize'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action == 'download': download()
    elif args.action == 'prepare': prepare()
    elif args.action == 'summarize': summarize()
    elif args.arm: run(args.arm)
    else: parser.error('--arm required for run')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
