"""Public English voice metadata; API credentials never enter the catalog."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

PREVIEW_HOSTS = {'cdn.ai33.pro', 'storage.googleapis.com'}


def normalize_voice(row, provider):
    identifier = row.get('voice_id') or row.get('id')
    locale = row.get('locale') or row.get('language')
    if not isinstance(identifier, str) or not re.fullmatch(r'[\w-]{1,160}', identifier):
        return None
    if locale not in {'en-US', 'en-GB'}:
        return None
    name = str(row.get('name') or identifier)[:120]
    if provider == 'edge':
        name = name.removeprefix(locale + '-').removesuffix('Neural')
    preview = row.get('preview_url') or ''
    try:
        url = urlsplit(preview) if isinstance(preview, str) else urlsplit('')
        valid = (url.scheme == 'https' and url.hostname in PREVIEW_HOSTS and not url.username
                 and not url.password and not url.query and not url.fragment and url.port in {None, 443})
    except ValueError:
        valid = False
    if not valid:
        preview = ''
    return {'id': identifier, 'name': name, 'locale': locale,
            'gender': str(row.get('gender') or '').lower() if str(row.get('gender') or '').lower() in {'male', 'female'} else '',
            'provider': provider, 'preview_url': preview}


def load_voices(root):
    try:
        rows = json.loads((Path(root) / 'configs/ai33_voices.json').read_text(encoding='utf-8'))['voices']
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    result = {}
    for row in rows[:100]:
        if not isinstance(row, dict) or row.get('provider') not in {'edge', 'elevenlabs'}:
            continue
        voice = normalize_voice({**row, 'voice_id': row.get('id')}, row['provider'])
        if voice:
            result[voice['id']] = voice
    return list(result.values())


def preview_stem(cache_dir, identifier):
    return Path(cache_dir) / 'previews' / hashlib.sha256(identifier.encode()).hexdigest()
