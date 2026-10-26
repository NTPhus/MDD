"""Pinned official CROTTC-IF inference in an isolated CPU or CUDA subprocess.

This is the released audio-only AR/CTC beam decoder, not greedy CTC decoding.
It supplies perceived phone sequences, not word recognition, calibrated scores,
or phone timestamps. Downloaded code is verified before execution and checkpoint
loading is restricted to weights-only deserialization in the worker.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time

MODEL_ID = "Haopeng/CROTTC-IF-l2-arctic"
REVISION = "9149e5107b3332b9aa3f5573f6328a4e312006ad"
MODEL_SHA256 = "3a9a4a03d2af6c90f71db8972a51dd4ce4559d1e9801f078904a65ce4aa113ef"
SSL_SHA256 = "a1ff5b43b55c412e73381e8b257c9af3c2237fa71b76bac5119ca8b31a531ec4"
RELATIVE = Path(".tools/crottc")
RUNTIME_PINS = {"speechbrain": "1.0.3", "HyperPyYAML": "1.2.2", "sentencepiece": "0.2.1",
                "ruamel.yaml": "0.18.16", "ruamel.yaml.clib": "0.2.14", "transformers": "4.57.0",
                "wandb": "0.22.2", "librosa": "0.11.0", "lazy-loader": "0.4", "soxr": "1.0.0",
                "pydantic": "2.12.3", "pydantic-core": "2.41.4", "annotated-types": "0.7.0",
                "typing-inspection": "0.4.2", "gitpython": "3.1.45", "gitdb": "4.0.12",
                "smmap": "5.0.2", "sentry-sdk": "2.42.0", "protobuf": "6.33.0",
                "torchaudio": "2.7.1+cu126"}
# Official Git blobs at REVISION; large checkpoints use published LFS SHA256.
FILE_PINS = {
    "README.md": (1199, "5d239c62982aa6f5dfcd5581f3ee8ded114afe0f"),
    "checkpoint/CKPT.yaml": (187, "be925a9a3dd49283b6e3a3658a2ee42ba4895b43"),
    "checkpoint/model.ckpt": (73894282, MODEL_SHA256),
    "checkpoint/perceived_ssl.ckpt": (1262009130, SSL_SHA256),
    "custom_interface.py": (5745, "b62914ddf97a285e2f31f9984f2fbd270add004a"),
    "hyperparams.yaml": (19737, "3d8777580e95496f6f7fe3c79123b2d64f5591fc"),
    "label_encoder.txt": (583, "91de8ef1bd3cadd3ab05c32d9ee5aa8259653b60"),
    "manifest.json": (545, "a61b22b6a9390dd88eb9b7023fc2cda5033a3c3e"),
    "requirements.txt": (230, "066c6cfc9774f257be2ca6a2f8b36547934be68b"),
    "runtime/models/Trans_IFMDD_ConPCO_ver2.py": (112938, "edd77ecf40c9459fe573dc85241ccce06f0433a5"),
    "runtime/models/__init__.py": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "runtime/models/projector.py": (2999, "c4a289a1e2484119d379866b10c6e4863ef08d40"),
    "runtime/mpd_eval_v4.py": (30799, "7155f59f89c925065f742f2a64f588c48dbb9c49"),
    "runtime/trainer/AutoSSLoader.py": (5393, "105bec22fd2f56d58ddfa81aa508b3c5ad2ff165"),
    "runtime/trainer/ConPCO_TransASR.py": (10599, "e8748e07a76101b094e90128184dffd8a9ab5336"),
    "runtime/trainer/__init__.py": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "runtime/utils/__init__.py": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "runtime/utils/layers/__init__.py": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "runtime/utils/layers/utils.py": (7244, "306be530cd1c824c29a32d691150e53364b70719"),
    "runtime/utils/losses/ConPCO_norm.py": (6027, "2bffd09fcd2416a7b017ff0b2f59c2c700653a60"),
    "runtime/utils/losses/__init__.py": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "runtime/utils/plot/__init__.py": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "runtime/utils/plot/plot_attn.py": (1162, "016f6eb46bd944d888d9a33fce52981541e3f0c6"),
    "runtime/utils/release_bundle.py": (9823, "e4078b76dd06da42ee3ad8ceb1b780718744d2c4"),
    "wavlm/config.json": (2222, "b1d1becf90dd05db908a9114148c204484ebec69"),
    "wavlm/preprocessor_config.json": (214, "73caa151574001d3d495fae897e1d38968249712"),
}
ARPABET = frozenset("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split())


def file_record(path, pin, *, checksum=True):
    path = Path(path)
    size, expected = pin
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Missing or wrong-sized CROTTC file: {path.name}")
    if not checksum:
        return {"bytes": size, "pin": expected}
    sha = hashlib.sha256()
    blob = hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            sha.update(chunk)
            blob.update(chunk)
    digest = sha.hexdigest()
    if (digest if len(expected) == 64 else blob.hexdigest()) != expected:
        raise ValueError(f"Pinned CROTTC checksum mismatch: {path.name}")
    return {"bytes": size, "pin": expected, "sha256": digest}


def verify(root, *, checksum=True, require_ready=True):
    folder = Path(root).resolve() / RELATIVE
    record = json.loads((folder / "setup_manifest.json").read_text(encoding="utf-8"))
    statuses = {"ready"} if require_ready else {"downloaded", "ready"}
    if (not isinstance(record, dict) or record.get("status") not in statuses
            or record.get("model_id") != MODEL_ID or record.get("revision") != REVISION):
        raise ValueError("CROTTC is not prepared and smoke-tested at the pinned revision")
    for name, pin in FILE_PINS.items():
        actual = file_record(folder / "bundle" / name, pin, checksum=checksum)
        saved = record.get("files", {}).get(name, {})
        if any(saved.get(key) != actual[key] for key in actual):
            raise ValueError(f"CROTTC manifest mismatch: {name}")
    if record.get("runtime_packages") != RUNTIME_PINS or runtime_versions(root) != RUNTIME_PINS:
        raise ValueError("CROTTC isolated runtime package pins changed")
    return record


def runtime_versions(root):
    path = Path(root).resolve() / RELATIVE / "python"
    versions = {distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
                for distribution in importlib.metadata.distributions(path=[str(path)])}
    result = {}
    for name, expected in RUNTIME_PINS.items():
        actual = versions.get(name.lower().replace("_", "-"))
        if actual != expected:
            raise ValueError(f"CROTTC optional runtime requires {name}=={expected}")
        result[name] = actual
    return result


def crottc_selection(root):
    folder = Path(root).resolve() / RELATIVE
    result = {"provider": "crottc-if", "id": "crottc-if-l2-arctic", "model_id": MODEL_ID,
              "revision": REVISION, "path": str(folder / "bundle"), "ready": False,
              "checkpoint_sha256": MODEL_SHA256, "ssl_checkpoint_sha256": SSL_SHA256,
              "vocabulary_sha256": None, "device": "cpu", "compute_type": "float32",
              "decoder": "official_speechbrain_ar_ctc_beam10_weight0.99", "phone_timing": "unavailable"}
    try:
        record = verify(root, checksum=False)
        result.update(ready=True, vocabulary_sha256=record["files"]["label_encoder.txt"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["reason"] = str(error)
    return result


def phone_evidence(labels):
    """Preserve unknown labels separately; never invent timing or map err to a phone."""
    phones, unresolved, excluded = [], [], []
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise ValueError("CROTTC decoded label must be text")
        phone = label.upper()
        if phone in ARPABET:
            phones.append({"phone": phone})
        elif label.lower() in {"sil", "<bos>", "<eos>", "<blank>"}:
            excluded.append({"index": index, "label": label})
        else:
            unresolved.append({"index": index, "label": label})
    return phones, unresolved, excluded


class CROTTCPronunciation:
    def __init__(self, root, *, timeout=300., require_ready=True, device="cpu", min_free_cuda_mib=2560):
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("CROTTC device must be cpu, cuda or auto")
        if isinstance(min_free_cuda_mib, bool) or not isinstance(min_free_cuda_mib, int) or min_free_cuda_mib < 0:
            raise ValueError("CROTTC minimum free CUDA memory must be a nonnegative integer MiB")
        self.root = Path(root).resolve()
        self.timeout = timeout
        self.require_ready = require_ready
        self.device = device
        self.min_free_cuda_mib = min_free_cuda_mib
        self.process = None
        self.events = None
        self.runtime = None
        self._lock = threading.RLock()
        self._sequence = 0

    def _start(self):
        folder = self.root / RELATIVE
        folder.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2", "PYTHONIOENCODING": "utf-8"}
        # CPU-only evaluation must remain isolated from CUDA. An explicit GPU
        # request preserves any device visibility already chosen by the caller.
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            self.process = subprocess.Popen([sys.executable, "-X", "utf8", "-u",
                str(self.root / "scripts/crottc_worker.py"), "--root", str(self.root),
                "--device", self.device, "--min-free-cuda-mib", str(self.min_free_cuda_mib),
                *( ["--allow-downloaded"] if not self.require_ready else [])],
                cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                encoding="utf-8", text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        process, events = self.process, Queue()
        self.events = events
        def receive():
            try:
                for line in process.stdout:
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            events.put(value)
                    except ValueError:
                        continue
            finally:
                events.put(None)
        threading.Thread(target=receive, daemon=True, name="crottc-worker-output").start()

    def _request(self, request, cancelled):
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("CROTTC inference cancelled")
        self._sequence += 1
        identifier = self._sequence
        self.process.stdin.write(json.dumps({"id": identifier, **request}, allow_nan=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            if cancelled is not None and cancelled.is_set():
                self.close()
                raise InterruptedError("CROTTC inference cancelled")
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("CROTTC inference timed out")
            try:
                response = self.events.get(timeout=.1)
            except Empty:
                continue
            if response is None:
                self.close()
                raise RuntimeError("CROTTC worker stopped; see .tools/crottc/worker.log")
            if response.get("id") != identifier:
                continue
            if "error" in response:
                raise RuntimeError(response["error"])
            return response["result"]

    def load(self, cancelled=None):
        with self._lock:
            if self.process is not None and self.process.poll() is None and self.runtime:
                return self
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("CROTTC inference cancelled")
            verify(self.root, checksum=False, require_ready=self.require_ready)
            self._start()
            try:
                self.runtime = self._request({"type": "load"}, cancelled)
            except BaseException:
                self.close()
                raise
        return self

    def infer(self, waveform, cancelled=None):
        import numpy as np
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or not 400 <= audio.size <= 16000 * 30 or not np.isfinite(audio).all():
            raise ValueError("CROTTC expects finite mono 16 kHz audio, 0.025 to 30 seconds")
        with self._lock:
            self.load(cancelled)
            result = self._request({"type": "infer", "waveform": audio.tolist()}, cancelled)
            if isinstance(result.get('runtime'), dict):
                self.runtime = dict(result['runtime'])
            return result

    def close(self):
        with self._lock:
            process = self.process
            self.process = self.events = self.runtime = None
            if process is None:
                return
            if process.poll() is None:
                try:
                    process.stdin.close()
                    process.wait(timeout=1.)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=3.)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3.)
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except (OSError, AttributeError):
                    pass
