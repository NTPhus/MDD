"""Offline US-English references, independent of audio/model predictions."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re
import unicodedata

COMMIT = "74790861f652b15e4ac49015a90074ad62a27690"
DICTIONARY_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
LICENSE_SHA256 = "bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a"


@lru_cache(maxsize=4)
def _load(path: str, modified: int, size: int) -> dict:
    """Return the primary (first) pronunciation variant for each word."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != DICTIONARY_SHA256:
        return {}
    result = {}
    for line in content.decode("utf-8").splitlines():
        parts = line.partition("#")[0].split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0])
        phones = [re.sub(r"[012]$", "", phone) for phone in parts[1:]]
        if all(re.fullmatch(r"[A-Z]+", phone) for phone in phones):
            # First listed variant is fixed before receiving any audio.
            result.setdefault(word, phones)
    return result


@lru_cache(maxsize=4)
def _load_all_variants(path: str, modified: int, size: int) -> dict:
    """Return all pronunciation variants for each word as a list of phone lists.

    CMUdict lists alternate pronunciations with a numeric suffix on the entry
    name (e.g. THE(2), EITHER(2)).  This loader collects every variant so that
    canonical_comparison can accept any of them instead of rejecting a correct
    pronunciation that happens to differ from the primary entry.
    """
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != DICTIONARY_SHA256:
        return {}
    result: dict[str, list[list[str]]] = {}
    for line in content.decode("utf-8").splitlines():
        parts = line.partition("#")[0].split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0])
        phones = [re.sub(r"[012]$", "", phone) for phone in parts[1:]]
        if all(re.fullmatch(r"[A-Z]+", phone) for phone in phones):
            result.setdefault(word, []).append(phones)
    return result


def dictionary_reference(root: Path, text: str) -> tuple[dict | None, list[str]]:
    path = root / "data/external/cmudict" / COMMIT / "cmudict.dict"
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return None, text.split()
        info = path.stat()
        dictionary = _load(str(path), info.st_mtime_ns, info.st_size)
        all_variants = _load_all_variants(str(path), info.st_mtime_ns, info.st_size)
    except (OSError, ValueError):
        dictionary = {}
        all_variants = {}
    canonical, mapping, missing = [], [], []
    word_all_variants: list[list[list[str]]] = []
    for index, word in enumerate(text.split()):
        key = unicodedata.normalize("NFKC", word).replace("\u2019", "'").lower().strip('.,!?;:()[]{}\u201c\u201d\u201e\u201f')
        if key.startswith("'") and key.endswith("'") and len(key) > 2:
            key = key[1:-1]
        phones = dictionary.get(key)
        if not phones:
            missing.append(word)
            word_all_variants.append([])
        else:
            canonical.extend(phones)
            mapping.extend([index] * len(phones))
            word_all_variants.append(all_variants.get(key, [phones]))
    if missing or not canonical:
        return None, missing
    return {"prompt_text": text.strip(), "canonical_phones": canonical, "canonical_word_indices": mapping,
            "word_all_variants": word_all_variants,
            "reference_source": "cmudict", "reference_commit": COMMIT}, []
