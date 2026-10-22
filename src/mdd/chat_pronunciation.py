"""Verified dictionary stress for display, separate from acoustic comparison.

The sound model compares stress-free phones.  This helper adds the stress marks
from the pinned CMUdict entry for a word only when its pronunciation matches the
already selected canonical reference; it never changes that reference.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re
import unicodedata

from . import demo_reference


def _word(token):
    # Keep lookup identical to demo_reference.dictionary_reference and the
    # chat reference draft, including punctuation and curly apostrophes.
    key = unicodedata.normalize('NFKC', token).replace('’', "'").lower().strip('.,!?;:()[]{}"“”')
    return key[1:-1] if key.startswith("'") and key.endswith("'") and len(key) > 2 else key


@lru_cache(maxsize=4)
def _load(path, modified, size):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != demo_reference.DICTIONARY_SHA256:
        return {}
    result = {}
    for line in content.decode('utf-8').splitlines():
        fields = line.partition('#')[0].split()
        if len(fields) < 2:
            continue
        word = re.sub(r'\(\d+\)$', '', fields[0])
        phones = fields[1:]
        # The same acceptance rule as the existing stress-free loader.  Keep
        # the first accepted variant, independent of the learner's recording.
        if all(re.fullmatch(r'[A-Z]+', re.sub(r'[012]$', '', phone)) for phone in phones):
            result.setdefault(word, tuple(phones))
    return result


def _dictionary(root):
    try:
        root = Path(root).resolve()
        path = (root / 'data/external/cmudict' / demo_reference.COMMIT / 'cmudict.dict').resolve()
        if not path.is_relative_to(root):
            return {}
        info = path.stat()
        return _load(str(path), info.st_mtime_ns, info.st_size)
    except (OSError, ValueError, UnicodeError):
        return {}


def target_words(root, text, reference):
    """Return existing per-word phones plus optional verified display phones.

    ``pronunciation_phones`` retains CMUdict stress digits.  It is omitted if
    the file is missing, its checksum is invalid, the word is absent, or the
    dictionary's first pronunciation differs from the selected reference.
    ``phones`` always comes from the supplied canonical reference unchanged.
    """
    if not isinstance(reference, dict) or not isinstance(text, str):
        return []
    canonical = reference.get('canonical_phones')
    indices = reference.get('canonical_word_indices')
    if not isinstance(canonical, list) or not isinstance(indices, list):
        return []
    dictionary = _dictionary(root)
    result = []
    for index, word in enumerate(text.split()):
        phones = [phone for phone, position in zip(canonical, indices) if position == index]
        item = {'text': word, 'phones': phones}
        pronunciation = dictionary.get(_word(word))
        if pronunciation and [re.sub(r'[012]$', '', phone) for phone in pronunciation] == phones:
            item['pronunciation_phones'] = list(pronunciation)
        result.append(item)
    return result
