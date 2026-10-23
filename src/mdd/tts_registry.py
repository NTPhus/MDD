"""Small local TTS catalog; checking readiness never imports a model runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path


KOKORO_VOICES = (
    ('af_heart', 'Heart', 'en-US', 'female'),
    ('af_bella', 'Bella', 'en-US', 'female'),
    ('af_sarah', 'Sarah', 'en-US', 'female'),
    ('am_adam', 'Adam', 'en-US', 'male'),
    ('am_michael', 'Michael', 'en-US', 'male'),
    ('bf_emma', 'Emma', 'en-GB', 'female'),
    ('bm_george', 'George', 'en-GB', 'male'),
)


def kokoro_status(root):
    """Trust a completed local setup only while its required assets still exist."""
    root = Path(root).resolve()
    model_dir = root / 'models/pretrained/kokoro-82m'
    runtime = root / '.venv-kokoro' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    unavailable = {'ready': False, 'voices': [], 'default_voice_id': ''}
    try:
        manifest = json.loads((model_dir / 'setup_manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return unavailable
    if not isinstance(manifest, dict) or manifest.get('status') != 'ready' or not runtime.is_file():
        return unavailable
    try:
        for key in ('model_file', 'voices_file'):
            filename = manifest.get(key)
            if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
                return unavailable
            path = (model_dir / filename).resolve()
            if not path.is_relative_to(model_dir.resolve()) or not path.is_file() or not path.stat().st_size:
                return unavailable
    except (OSError, ValueError):
        return unavailable
    available_ids = manifest.get('voices')
    if not isinstance(available_ids, list):
        return unavailable
    voices = [
        {'id': identifier, 'name': name, 'locale': locale, 'gender': gender,
         'provider': 'kokoro', 'preview_url': '/api/chat/voice-preview/' + identifier}
        for identifier, name, locale, gender in KOKORO_VOICES if identifier in available_ids
    ]
    if not voices:
        return unavailable
    default = manifest.get('default_voice_id', 'af_heart')
    if not any(voice['id'] == default for voice in voices):
        default = voices[0]['id']
    return {'ready': True, 'voices': voices, 'default_voice_id': default}
