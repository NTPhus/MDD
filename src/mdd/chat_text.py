"""Formatting helpers for chat text and speech output."""
from __future__ import annotations

from html import unescape
import re


def plain_speech_text(value: object) -> str:
    """Decode common entities and remove Markdown syntax before TTS."""
    text = unescape(str(value or '')).replace('\u00a0', ' ')
    text = re.sub(r'\\([*_~`])', r'\1', text)
    text = re.sub(r'\*\*\s*\*\*([^*\n]+?)\*\*\s*\*\*', r'**\1**', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'^\s{0,3}#{1,6}\s+|^\s{0,3}>\s?|^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]{1,100}>', '', text)
    text = re.sub(r'(?:\*\*|__|~~|`)+(?=\S)|(?<=\S)(?:\*\*|__|~~|`)+(?!\w)', '', text)
    text = re.sub(r'(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)', '', text)
    return re.sub(r'\n{3,}', '\n\n', re.sub(r'[ \t]+\n', '\n', text)).strip()
