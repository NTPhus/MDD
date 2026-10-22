"""Download pinned CMUdict data and its license for offline custom-text practice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdd.demo_reference import COMMIT, DICTIONARY_SHA256, LICENSE_SHA256


def main():
    destination = ROOT / "data/external/cmudict" / COMMIT
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"repository": "https://github.com/cmusphinx/cmudict", "commit": COMMIT, "files": {}}
    for name, expected in (("cmudict.dict", DICTIONARY_SHA256), ("LICENSE", LICENSE_SHA256)):
        path = destination / name
        url = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{COMMIT}/{name}"
        content = path.read_bytes() if path.is_file() else urlopen(url, timeout=60).read(8 * 1024 * 1024)
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"CMUdict hash mismatch: {name}; no existing file was replaced")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        receipt["files"][name] = {"url": url, "sha256": digest, "bytes": len(content)}
    (destination / "source-manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("CMUdict and LICENSE verified; custom-text practice works offline.")


if __name__ == "__main__":
    main()
