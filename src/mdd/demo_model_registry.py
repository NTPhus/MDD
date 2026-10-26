"""Server-owned model selection; incomplete/unvalidated runs fall back to base."""
import json
import os
from pathlib import Path

def _read(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def speech_device_policy(component, environment=None):
    """Cheap process policy; actual devices are reported after model loading."""
    names = {'asr': 'MDD_CHAT_ASR_DEVICE', 'phoneme': 'MDD_CHAT_PHONEME_DEVICE',
             'crottc': 'MDD_CHAT_CROTTC_DEVICE'}
    if component not in names:
        raise ValueError('Unknown speech device component')
    environment = os.environ if environment is None else environment
    default = 'cpu' if environment.get('MDD_CHAT_DEVICE', '').strip().lower() == 'cpu' else 'auto'
    choice = environment.get(names[component], default).strip().lower()
    if choice not in {'cpu', 'cuda', 'auto'}:
        raise ValueError(f'{names[component]} must be auto, cpu or cuda')
    return choice

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


def speech_asr_selection(root, preference=None):
    """Use Whisper for conversation; legacy CTC requires an explicit override."""
    choice = (os.environ.get('MDD_CHAT_ASR', 'whisper-small-en') if preference is None else preference).strip().lower()
    if choice not in {'auto', 'whisper-small-en', 'wav2vec2'}:
        raise ValueError('MDD_CHAT_ASR must be auto, whisper-small-en or wav2vec2')
    if choice != 'wav2vec2':
        from scripts.whisper_asr import whisper_selection
        whisper = whisper_selection(root)
        policy = speech_device_policy('asr')
        whisper = {**whisper, 'requested_device': policy, 'device': policy,
                   'compute_type': 'int8' if policy == 'cpu' else 'int8_float16',
                   'device_is_policy': True}
        if whisper['ready'] or choice == 'whisper-small-en':
            return {**whisper, 'provider': 'whisper', 'trained': False,
                    'selection': 'local_whisper_small_en'}
    selected = asr_selection(root)
    return {**selected, 'provider': 'wav2vec2', 'ready': (selected['path'] / 'model.safetensors').is_file(),
            'device': 'cpu', 'compute_type': 'float32', 'revision': None}

def grammar_selection(root):
    root=Path(root);run=root/'runs/grammar-qwen3-06b-wi-jfleg-v1'
    record=_read(run/'training_provenance.json');validation=_read(run/'demo_validation.json')
    expected=record.get('adapter_sha256')
    ready=(record.get('status')=='completed' and record.get('effective_config',{}).get('dataset_kind')=='published_grammar_fluency'
           and validation.get('status')=='passed' and validation.get('adapter_sha256')==expected
           and isinstance(expected,str) and (run/'adapter/adapter_model.safetensors').is_file())
    return {'id':'qwen-grammar' if ready else 'qwen-base','ready':bool(ready),
            'adapter':run/'adapter','sha256':expected,'validation':validation.get('scope')}


def pronunciation_selection(root, preference=None):
    """Keep B as default; CROTTC is available for explicit experimental trials."""
    root = Path(root)
    choice = (os.environ.get('MDD_CHAT_PRONUNCIATION', 'wav2vec2-b') if preference is None else preference).strip().lower()
    if choice not in {'auto', 'crottc-if', 'wav2vec2-b'}:
        raise ValueError('MDD_CHAT_PRONUNCIATION must be auto, crottc-if or wav2vec2-b')
    if choice != 'wav2vec2-b':
        from scripts.crottc_pronunciation import crottc_selection
        selected = crottc_selection(root)
        selected = {**selected, 'requested_device': speech_device_policy('crottc'),
                    'device': speech_device_policy('crottc'),
                    'device_is_policy': True}
        if selected['ready'] or choice == 'crottc-if':
            return {**selected, 'id': 'crottc-if-l2-arctic', 'provider': 'crottc-if'}
    checkpoint = root / 'runs/cascade-l2arctic-plus-seed42/B/best_checkpoint'
    return {'id': 'cascade-B', 'provider': 'wav2vec2-b', 'path': checkpoint,
            'ready': (checkpoint / 'model.safetensors').is_file(),
            'requested_device': speech_device_policy('phoneme'), 'device_is_policy': True,
            'device': speech_device_policy('phoneme'),
            'method': 'ctc_variant_posterior_margin_v1'}


def pronunciation_models(root):
    return [{'id': value['id'], 'ready': value['ready'], 'label': label, 'experimental': experimental}
            for choice, label, experimental in [('wav2vec2-b', 'wav2vec2 B', False),
                ('crottc-if', 'CROTTC-IF · Thử nghiệm', True)]
            for value in [pronunciation_selection(root, choice)]]


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
             and revision == '86b5e0934494bd15c9632b12f734a8a67f723594'
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
