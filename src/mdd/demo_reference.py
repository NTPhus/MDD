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


def dictionary_reference(root: Path, text: str) -> tuple[dict | None, list[str]]:
    path = root / "data/external/cmudict" / COMMIT / "cmudict.dict"
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return None, text.split()
        info = path.stat()
        dictionary = _load(str(path), info.st_mtime_ns, info.st_size)
    except (OSError, ValueError):
        dictionary = {}
    canonical, mapping, missing = [], [], []
    for index, word in enumerate(text.split()):
        key = unicodedata.normalize("NFKC", word).replace("’", "'").lower().strip('.,!?;:()[]{}"“”')
        if key.startswith("'") and key.endswith("'") and len(key) > 2:
            key = key[1:-1]
        phones = dictionary.get(key)
        if not phones:
            missing.append(word)
        else:
            canonical.extend(phones)
            mapping.extend([index] * len(phones))
    if missing or not canonical:
        return None, missing
    return {"prompt_text": text.strip(), "canonical_phones": canonical, "canonical_word_indices": mapping,
            "reference_source": "cmudict", "reference_commit": COMMIT}, []
