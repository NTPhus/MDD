"""Download pinned author Charsiu CTC/frame-classification weights and vocabulary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.setup_phoneme_baseline import file_record

RELATIVE = Path("models/pretrained/charsiu-phoneme")
TOKENIZER_ID = "charsiu/tokenizer_en_cmu"
TOKENIZER_REVISION = "10507401aedf5e0aba164128535b49225ff95260"
SOURCE_REVISION = "13a69f2a22ca0c0962b75cc693399b0ae23a12c9"
MODELS = {
    "charsiu-ctc": {"model_id": "charsiu/en_w2v2_ctc_libris_and_cv",
                    "revision": "70f5061463f2927a27236d7e9d309cf0fd5282b3",
                    "weight_sha256": "fd35d545cedfe652b1cd835ecd4f99da86372103c219cd3ae1ea4ba47537ef40",
                    "config_bytes": 1877, "config_sha256": "2d3868d5baf70c3166ae060c4e846a274c53146d5b2c6cc03fc912a3b86e27ab",
                    "objective": "CTC phone recognition", "stride_samples": 320},
    "charsiu-fc10ms": {"model_id": "charsiu/en_w2v2_fc_10ms",
                       "revision": "e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f",
                       "weight_sha256": "6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1",
                       "config_bytes": 1920, "config_sha256": "7406aa4f917267640865688aa62f2337664a3abb9a49a2f204d932b53aeb6cb7",
                       "objective": "Frame-wise cross-entropy phone classification", "stride_samples": 160},
}
TOKENIZER_PINS = {
    "vocab.json": (403, "781514e251e9615ea4a05bbf3760ffcf8ec2ebe67355f411177ea64b0602d897"),
    "tokenizer_config.json": (180, "56e02b54de01a0ec4cc0c705ad3c303d7b73c83d61063a94b84cbf3a55da76a1"),
    "special_tokens_map.json": (85, "50eb73d51191696209d30d42d6ede50e57e7a542ca1db12df714b2c0aa3da8e2"),
}
PREPROCESSING = {"feature_size": 1, "sampling_rate": 16000, "padding_value": 0.0,
                 "do_normalize": True, "return_attention_mask": False}


def paths_and_pins():
    result = {}
    for key, model in MODELS.items():
        result[key + "/pytorch_model.bin"] = (377706220, model["weight_sha256"])
        result[key + "/config.json"] = (model["config_bytes"], model["config_sha256"])
    result.update({"tokenizer/" + name: pin for name, pin in TOKENIZER_PINS.items()})
    return result


def source_url(name):
    group, filename = name.split("/", 1)
    mid, revision = (TOKENIZER_ID, TOKENIZER_REVISION) if group == "tokenizer" else (MODELS[group]["model_id"], MODELS[group]["revision"])
    return f"https://huggingface.co/{mid}/resolve/{revision}/{filename}"


def verify(root=ROOT, *, require_inspection=True):
    folder = Path(root) / RELATIVE
    actual = {name: file_record(folder / name, pin) for name, pin in paths_and_pins().items()}
    if require_inspection:
        saved = json.loads((folder / "setup_manifest.json").read_text(encoding="utf-8"))
        if saved.get("files") != actual:
            raise ValueError("Charsiu setup manifest file identities differ")
        if any(saved.get("state_inspection", {}).get(model, {}).get("unsafe_globals") != []
               or saved["state_inspection"][model].get("weights_only") is not True for model in MODELS):
            raise ValueError("Charsiu safe state inspection is incomplete")
    return actual


def inspect_states(root=ROOT):
    """Never allow pickle globals; inspect tensor states on CPU before model loading."""
    import torch
    result = {}
    for model in MODELS:
        path = Path(root) / RELATIVE / model / "pytorch_model.bin"
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        if unsafe:
            raise ValueError("Unapproved checkpoint globals: " + str(unsafe))
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError("Expected a tensor-only model state dictionary")
        if tuple(state["lm_head.weight"].shape) != (42, 768):
            raise ValueError("Unexpected Charsiu phone classification head")
        result[model] = {"unsafe_globals": unsafe, "weights_only": True, "map_location": "cpu",
                         "state_keys": len(state), "tensor_elements": sum(v.numel() for v in state.values()),
                         "classification_head_shape": list(state["lm_head.weight"].shape)}
        del state
        gc.collect()
    return result


def download(root=ROOT):
    folder = Path(root) / RELATIVE
    for name, pin in paths_and_pins().items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_record(path, pin)
            continue
        except ValueError:
            pass
        print(json.dumps({"download": name, "bytes": pin[0]}), flush=True)
        temporary = path.with_name(path.name + ".part")
        request = urllib.request.Request(source_url(name), headers={"User-Agent": "MDD-Charsiu-evaluation/1.0"})
        with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        file_record(temporary, pin)
        temporary.replace(path)
    files = verify(root, require_inspection=False)
    states = inspect_states(root)
    record = {"status": "downloaded_verified_inspected", "models": MODELS,
              "tokenizer_id": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
              "files": files, "state_inspection": states, "preprocessing": PREPROCESSING,
              "source_revision": SOURCE_REVISION, "source_repository": "https://github.com/lingjzhu/charsiu",
              "paper": "https://arxiv.org/abs/2110.03876", "training": False,
              "license_note": "Author source repository is MIT; individual HF weight repositories have no separate license/model-card metadata.",
              "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    (folder / "setup_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = download() if args.download else verify()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
