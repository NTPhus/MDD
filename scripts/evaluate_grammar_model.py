"""Matched base/adapter rewriting evaluation. No inference without --run.

Exact-reference match and chrF are diagnostics, not grammar accuracy or human
naturalness scores. Raw invalid outputs remain failures; there is no fallback.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

if __package__:
    from .grammar_data import ROOT, load, messages, object_sha, rooted, sha, write_json
else:
    from grammar_data import ROOT, load, messages, object_sha, rooted, sha, write_json


def parse_output(text, task, finished=True):
    if not finished:
        return None, 'generation_budget_exhausted'
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None, 'invalid_json'
    if not isinstance(value, dict) or set(value) != {task} or not isinstance(value[task], str) or not value[task].strip():
        return None, 'invalid_schema'
    return value[task].strip(), None


def selected_rows(rows, limit):
    # Interleave tasks so a development probe does not accidentally omit JFLEG.
    tasks = defaultdict(list)
    for row in sorted(rows, key=lambda r: object_sha(r['id'])):
        tasks[row['task']].append(row)
    result = []
    for index in range(max(map(len, tasks.values()), default=0)):
        for name in sorted(tasks):
            if index < len(tasks[name]):
                result.append(tasks[name][index])
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    return result if limit is None else result[:limit]


def score(observations):
    from sacrebleu.metrics import CHRF
    groups = defaultdict(list)
    for row in observations:
        groups[row['task']].append(row)
    result = {}
    for task, rows in groups.items():
        refs = [[r['references'][min(i, len(r['references']) - 1)] for r in rows]
                for i in range(max(len(r['references']) for r in rows))]
        predictions = [r['prediction'] if r['prediction'] is not None else '' for r in rows]
        controls = [r for r in rows if r['source'] in r['references']]
        norm = lambda s: ' '.join(s.split())
        metric = CHRF()
        chrf = metric.corpus_score(predictions, refs)
        result[task] = {
            'rows': len(rows), 'valid_json_rate': sum(r['prediction'] is not None for r in rows) / len(rows),
            'exact_reference_match': sum(r['prediction'] is not None and norm(r['prediction']) in
                                         {norm(t) for t in r['references']} for r in rows) / len(rows),
            'chrf': chrf.score, 'chrf_signature': str(metric.get_signature()),
            'reference_supported_unchanged_controls': len(controls),
            'unchanged_preservation_rate': (sum(r['prediction'] == r['source'] for r in controls) / len(controls)
                                          if controls else None),
            'mean_generation_seconds': sum(r['seconds'] for r in rows) / len(rows),
        }
    return {'tasks': result, 'metric_scope': 'Text-reference diagnostics only. Not official ERRANT F0.5, '
            'JFLEG GLEU, semantic faithfulness, or human grammar/naturalness accuracy.'}


def evaluate(args):
    import torch
    from filelock import FileLock
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['dataset_kind'] != 'published_grammar_fluency':
        raise ValueError('Expected grammar config')
    parts, manifest = load(rooted(config['dataset_dir']))
    rows = selected_rows(parts[args.split], args.limit)
    path = rooted(config['model_path'])
    if sha(path / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from dataset')
    adapter = args.adapter.resolve() if args.adapter else None
    provenance = None
    if adapter:
        provenance = json.loads((adapter.parent / 'training_provenance.json').read_text(encoding='utf-8'))
        if provenance['status'] != 'completed' or provenance['dataset'] != manifest:
            raise ValueError('Incomplete adapter or different training dataset')
        if provenance['adapter_sha256'] != sha(adapter / 'adapter_model.safetensors'):
            raise ValueError('Adapter checksum mismatch')
    identity = {'dataset_manifest_sha256': object_sha(manifest), 'split': args.split,
                'ordered_ids': [r['id'] for r in rows], 'development_probe': args.limit is not None,
                'base_weight_sha256': sha(path / 'model.safetensors'),
                'tokenizer_sha256': manifest['tokenizer_sha256'], 'prompt_sha256': manifest['prompt_sha256'],
                'adapter_sha256': sha(adapter / 'adapter_model.safetensors') if adapter else None,
                'device': args.device, 'precision': 'nf4-double-bf16-or-fp16' if args.device == 'cuda' else 'float32',
                'generation': {'do_sample': False, 'max_new_tokens': 512}, 'seed': config['seed'],
                'packages': {name: version(name) for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'sacrebleu')},
                'compute_dtype': ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if args.device == 'cuda' else 'float32',
                'hardware': torch.cuda.get_device_name() if args.device == 'cuda' and torch.cuda.is_available() else 'cpu',
                'evaluator_sha256': sha(Path(__file__))}
    if provenance and identity['base_weight_sha256'] != provenance['base_weight_sha256']:
        raise ValueError('Adapter and evaluation base weights differ')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Evaluation output exists; choose a new path or --resume')
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.evaluation.lock'), timeout=1):
        metadata_path, predictions_path = output / 'metadata.json', output / 'predictions.jsonl'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata['identity'] != identity:
                raise ValueError('Resume identity differs')
        else:
            if predictions_path.exists():
                raise ValueError('Predictions without metadata')
            metadata = {'identity': identity, 'status': 'started',
                        'started_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json(metadata_path, metadata)
        observations = ([json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines()]
                        if predictions_path.exists() else [])
        if [r['id'] for r in observations] != identity['ordered_ids'][:len(observations)]:
            raise ValueError('Predictions are not the selected dataset prefix')
        for original, prior in zip(rows, observations):
            if prior['source_row_sha256'] != object_sha(original):
                raise ValueError('Cached prediction row differs')
        if len(observations) < len(rows):
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
            if args.device == 'cuda' and not torch.cuda.is_available():
                raise ValueError('CUDA unavailable; explicitly select --device cpu for a slower float32 comparison')
            torch.set_num_threads(4)
            set_seed(config['seed'])
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if args.device == 'cuda' else torch.float32
            quantization = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if args.device == 'cuda' else None)
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                trust_remote_code=False, dtype=dtype, quantization_config=quantization, device_map={'': args.device})
            if adapter:
                model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True, is_trainable=False,
                                                  torch_device=args.device, device_map={'': args.device})
                if args.device == 'cpu': model.to('cpu')
            model.eval().requires_grad_(False)
            for row in rows[len(observations):]:
                prompt = tokenizer.apply_chat_template(messages(row), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(args.device)
                count = inputs['input_ids'].shape[1]
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, **identity['generation'], use_cache=True,
                        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                token_ids = generated[0, count:]
                raw = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                prediction, error = parse_output(raw, row['task'], bool(len(token_ids) and int(token_ids[-1]) == tokenizer.eos_token_id))
                observed = {'id': row['id'], 'source_row_sha256': object_sha(row), 'source': row['source'],
                            'task': row['task'], 'references': row['references'], 'raw': raw,
                            'prediction': prediction, 'error': error, 'seconds': time.perf_counter() - started}
                with predictions_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(observed, ensure_ascii=False) + '\n')
                    stream.flush()
                observations.append(observed)
                print(f"{len(observations)}/{len(rows)} {row['id']} valid={prediction is not None}", flush=True)
        write_json(output / 'summary.json', score(observations))
        metadata.update(status='completed', predictions_sha256=sha(predictions_path),
                        completed_rows=len(observations), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)


def compare(before, after, output):
    a, b = [json.loads((p / 'metadata.json').read_text(encoding='utf-8')) for p in (before, after)]
    if a['status'] != 'completed' or b['status'] != 'completed':
        raise ValueError('Both evaluations must be completed')
    if a['identity']['adapter_sha256'] is not None or b['identity']['adapter_sha256'] is None:
        raise ValueError('Compare base before against adapter after')
    common = lambda m: {k: v for k, v in m['identity'].items() if k != 'adapter_sha256'}
    if common(a) != common(b):
        raise ValueError('Before/after data, generation, device, precision or model identity differs')
    summaries = []
    for directory, meta in ((before, a), (after, b)):
        if sha(directory / 'predictions.jsonl') != meta['predictions_sha256']:
            raise ValueError('Predictions checksum changed')
        observations = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(observations) != len(meta['identity']['ordered_ids']):
            raise ValueError('Incomplete predictions')
        summaries.append(score(observations))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {'identity': common(a), 'before': summaries[0], 'after': summaries[1],
                        'automatic_promotion': False})
    print('Matched comparison:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grammar_qwen3_06b_qlora.json')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('BEFORE', 'AFTER'))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare, args.output)
    elif args.run:
        evaluate(args)
    else:
        parser.error('Use --run to explicitly request inference, or --compare for saved results')


if __name__ == '__main__':
    main()
