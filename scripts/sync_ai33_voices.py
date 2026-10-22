"""Refresh English ai33 voices and optionally download existing preview samples.

Only uses GET/list/download calls; never submits a text-to-speech task.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.ai33_tts import AI33TTS, load_env
from mdd.chat_voices import normalize_voice, preview_stem


def voice_rows(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('voices', 'items', 'results', 'data'):
            if key in data:
                rows = voice_rows(data[key])
                if rows:
                    return rows
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previews', action='store_true', help='Download provider samples without generating TTS')
    args = parser.parse_args()
    load_env(ROOT / '.env')
    client = AI33TTS(ROOT / 'data/generated/tts/ai33/chat')
    voices = {}
    for provider, search in [('edge', 'en-US'), ('edge', 'en-GB'), ('elevenlabs', '')]:
        data = client.list_voices(provider, search)
        for row in voice_rows(data):
            voice = normalize_voice(row, provider)
            if voice and voice['preview_url']:
                voices[voice['id']] = voice
    if not voices:
        raise RuntimeError('No English voices returned; existing catalog was retained.')
    # Preserve the configured voice at the top, then expose both accents/providers.
    import os
    default = os.environ.get('AI33_VOICE_EN', '')
    rows = sorted(voices.values(), key=lambda v: (v['id'] != default, v['provider'], v['locale'], v['name']))[:60]
    target = ROOT / 'configs/ai33_voices.json'
    snapshot = {'updated_at': datetime.now(timezone.utc).isoformat(), 'voices': rows}
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    downloaded = 0
    if args.previews:
        for voice in rows:
            stem = preview_stem(client.cache_dir, voice['id'])
            stem.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not stem.with_suffix('.mp3').is_file():
                    client.download(voice['preview_url'], stem)
                downloaded += 1
            except Exception:
                print('Preview unavailable:', voice['name'])
    print(f'Saved {len(rows)} English voices; {downloaded} existing previews available. No TTS submitted.')


if __name__ == '__main__':
    main()
