"""Server-owned model selection; incomplete/unvalidated runs fall back to base."""
import json
from pathlib import Path
import re

def _read(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}

def asr_selection(root):
    root=Path(root)
    run=root/'runs/asr-vietnamese-english-v1'
    record=_read(run/'training_provenance.json');results=record.get('results',{})
    candidate=run/'best_checkpoint'
    if (record.get('status')=='completed' and results.get('validation_improved') is True
            and isinstance(results.get('trained_sha256'),str)
            and (candidate/'model.safetensors').is_file()):
        return {'id':'wav2vec2-vi-en-v1','path':candidate,'sha256':results['trained_sha256'],
                'trained':True,'selection':'validation_wer_improved'}
    base=root/'models/pretrained/wav2vec2-base-960h'
    files=_read(base/'download_manifest.json').get('files',[])
    expected=next((row.get('sha256') for row in files if row.get('filename')=='model.safetensors'),None)
    return {'id':'facebook/wav2vec2-base-960h','path':base,'sha256':expected,
            'trained':False,'selection':'pretrained_fallback'}

def grammar_selection(root):
    root=Path(root);run=root/'runs/grammar-qwen3-06b-wi-jfleg-v1'
    record=_read(run/'training_provenance.json');validation=_read(run/'demo_validation.json')
    expected=record.get('adapter_sha256')
    ready=(record.get('status')=='completed' and record.get('effective_config',{}).get('dataset_kind')=='published_grammar_fluency'
           and validation.get('status')=='passed' and validation.get('adapter_sha256')==expected
           and isinstance(expected,str) and (run/'adapter/adapter_model.safetensors').is_file())
    return {'id':'qwen-grammar' if ready else 'qwen-base','ready':bool(ready),
            'adapter':run/'adapter','sha256':expected,'validation':validation.get('scope')}


def recovery_selection(root):
    """Choose the installed masked LM; check full checksums once at model load.

    Config polling is deliberately cheap. Download provenance and file sizes
    establish installation readiness here; the inference loader verifies bytes.
    """
    path = Path(root) / 'models/pretrained/bert-base-uncased'
    manifest = _read(path / 'setup_manifest.json')
    files = manifest.get('files', {})
    required = ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors')
    revision = manifest.get('revision')
    ready = (manifest.get('status') in {'downloaded', 'ready'}
             and manifest.get('model_id') == 'google-bert/bert-base-uncased'
             and isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision) is not None
             and isinstance(files, dict))
    for name in required:
        row = files.get(name, {}) if isinstance(files, dict) else {}
        if not isinstance(row, dict):
            row = {}
        try:
            ready = ready and (path / name).is_file() and (path / name).stat().st_size == row.get('bytes')
        except OSError:
            ready = False
    return {'provider': 'bert' if ready else 'qwen', 'ready': bool(ready), 'path': path,
            'model_id': 'google-bert/bert-base-uncased' if ready else 'Qwen/Qwen3-0.6B',
            'revision': revision if ready else None,
            'version': 'bert_masked_recovery_v1' if ready else 'guarded_intent_recovery_v1'}
