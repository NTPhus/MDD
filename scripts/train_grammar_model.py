"""Validate the dedicated rewriting adapter config; training requires --run."""
import argparse
import json
from pathlib import Path

if __package__:
    from .grammar_data import ROOT, encode, sha
    from .train_feedback_model import data_check, train, from_root
else:
    from grammar_data import ROOT, encode, sha
    from train_feedback_model import data_check, train, from_root


def check(config):
    from transformers import AutoTokenizer
    if config.get('dataset_kind') != 'published_grammar_fluency':
        raise ValueError('This entry point accepts only the separate grammar/fluency dataset')
    parts, manifest = data_check(config)
    model = from_root(config['model_path'])
    if sha(model / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from frozen preflight')
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    for split, rows in parts.items():
        for row in rows:
            if len(encode(tokenizer, row, config['max_length'])['input_ids']) != row['total_tokens']:
                raise ValueError('Token preflight changed: ' + row['id'])
    return parts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grammar_qwen3_06b_qlora.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-data-only', action='store_true')
    mode.add_argument('--run', action='store_true', help='Explicitly start GPU training')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    parts, manifest = check(config)
    print(json.dumps({'rows': {k: len(v) for k, v in parts.items()},
                      'output': config['output_dir'], 'training_requested': args.run}, indent=2), flush=True)
    if args.run:
        train(config, parts, manifest, resume=args.resume)


if __name__ == '__main__':
    main()
