"""Measure the existing local speech demo, without voice synthesis or training.

Example (the server must already be running):
  python -X utf8 scripts/benchmark_demo_latency.py --run --audio sample.wav
      --output data/generated/demo-latency --runs 3

The first request is not labelled cold: an existing worker may already be warm.
Wall times include HTTP polling and therefore have the chosen polling precision.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"ready", "error", "cancelled"}
JOB_ID = re.compile(r"[0-9a-f]{32}")
ROUTE = re.compile(r"/api/chat/(?:config|speech|cancel|jobs/[0-9a-f]{32})")


def local_url(value):
    """Reject remote hosts, credentials, paths and proxy/redirect escapes."""
    try:
        parts = urllib.parse.urlsplit(value)
        if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parts.username is not None or parts.password is not None
                or parts.path not in {"", "/"} or parts.query or parts.fragment
                or parts.port is None or not 1 <= parts.port <= 65535):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use a loopback HTTP URL with a port, such as http://127.0.0.1:8765") from None
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The local benchmark does not follow HTTP redirects")


class LocalDemoClient:
    def __init__(self, base):
        self.base = local_url(base)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, payload=None):
        if not ROUTE.fullmatch(path):
            raise ValueError("Unsupported benchmark route; voice requests are never permitted")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data,
            headers={"Content-Type": "application/json", "Origin": self.base})
        with self.opener.open(request, timeout=20) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("The local API did not return a JSON object")
        return result


def encode_audio(path):
    """Decode and resample a local file; never initialize a model."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError("Audio must be an existing local file of at most 32 MiB")
    info = sf.info(str(path))
    duration = info.frames / info.samplerate
    if not .25 <= duration <= 30 or info.channels not in {1, 2} or not 8000 <= info.samplerate <= 192000:
        raise ValueError("Audio must be 0.25–30 seconds, mono/stereo, and 8–192 kHz")
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains non-finite samples")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, 16000, format="WAV", subtype="PCM_16")
    data = buffer.getvalue()
    return base64.b64encode(data).decode("ascii"), {
        "filename": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "submitted_wav_sha256": hashlib.sha256(data).hexdigest(),
        "duration_sec": round(len(waveform) / 16000, 6), "sample_rate": 16000,
    }


def summarize_job(job, elapsed, stages, reply):
    """Keep timing/provenance and text, excluding acoustic arrays and audio."""
    speech = job.get("speech") or {}
    normalization = speech.get("normalization") or {}
    timings = {key: value for key, value in speech.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)
               and (key.endswith("_sec") or key.endswith("_seconds"))}
    for key in ("timings", "runtime", "generation_runtime"):
        if isinstance(speech.get(key), dict):
            timings[key] = speech[key]
    return {
        "job_id": job.get("id"), "status": job.get("status"), "full_elapsed_sec": round(elapsed, 3),
        "server_elapsed_sec": job.get("elapsed_sec"), "reply_ready": reply,
        "stage_transitions": stages, "transcript": speech.get("transcript"), "reply": job.get("text"),
        "normalization": {key: normalization[key] for key in
                          ("status", "source", "version", "latency_sec") if key in normalization},
        "speech_timings": timings, "metadata": job.get("metadata") or {},
        "models": {key: speech[key] for key in ("asr_model", "phoneme_model") if key in speech},
        "feedback_status": {key: (speech.get(key) or {}).get("status")
                            for key in ("rewrite", "correction", "coaching")},
        "voice_status": (job.get("voice") or {}).get("status"),
        "error": job.get("error"),
    }


def run_once(client, encoded, index, poll_interval, timeout, *, clock=time.monotonic, sleep=time.sleep):
    started, job, stages, reply = clock(), None, [], None
    identifier = None
    try:
        job = client.request("/api/chat/speech", {"model_id": "qwen-base", "request_id": uuid.uuid4().hex,
            "history": [], "audio_base64": encoded})
        candidate = job.get("id")
        if not isinstance(candidate, str) or not JOB_ID.fullmatch(candidate):
            raise ValueError("Speech submission did not return a valid job ID")
        identifier = candidate
        while True:
            elapsed = clock() - started
            status = job.get("status")
            if not stages or stages[-1]["stage"] != status:
                stages.append({"stage": status, "observed_elapsed_sec": round(elapsed, 3)})
            speech = job.get("speech") or {}
            if speech.get("reply_ready") is True and reply is None:
                reply = {"elapsed_sec": round(elapsed, 3), "job_status": status,
                         "rewrite_status": (speech.get("rewrite") or {}).get("status"),
                         "coaching_status": (speech.get("coaching") or {}).get("status")}
            if (job.get("voice") or {}).get("status") not in {None, "not_requested"}:
                raise ValueError("Unexpected voice activity on a benchmark-owned job")
            if status in TERMINAL:
                break
            if elapsed >= timeout:
                raise TimeoutError("The local speech job exceeded the benchmark timeout")
            sleep(min(poll_interval, timeout - elapsed))
            job = client.request("/api/chat/jobs/" + identifier)
        result = summarize_job(job, elapsed, stages, reply)
        result.update(index=index, label="first request" if index == 1 else "repeat request")
        if status != "ready":
            raise RuntimeError("The local speech job finished with status " + str(status))
        if reply is None:
            raise RuntimeError("The completed job did not publish reply_ready")
        return result, job
    except BaseException:
        if identifier is not None and (job or {}).get("status") not in TERMINAL:
            try:
                client.request("/api/chat/cancel", {"job_id": identifier})
            except Exception:
                pass  # Preserve the initial failure; no unrelated job is cancelled.
        raise


def source_hashes():
    files = ["scripts/benchmark_demo_latency.py", "scripts/local_chat_worker.py", "scripts/local_chat_process.py",
             "scripts/local_speech_engine.py", "scripts/transcript_normalization.py", "src/mdd/demo_chat.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Explicitly permit local audio inference; never calls a voice API")
    parser.add_argument("--audio", type=Path, required=True, help="Local recording; decoded to mono 16 kHz PCM WAV")
    parser.add_argument("--output", type=Path, required=True, help="Directory for summary.json")
    parser.add_argument("--runs", type=int, default=2, help="At least two sequential requests (default: 2)")
    parser.add_argument("--base-url", type=local_url, default="http://127.0.0.1:8765")
    parser.add_argument("--poll-interval", type=float, default=.15, help="Polling interval from 0.1 to 0.2 seconds")
    parser.add_argument("--timeout", type=float, default=420., help="Maximum wall seconds per request")
    parser.add_argument("--save-jobs", action="store_true", help="Save final public jobs separately (includes acoustic arrays)")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("--run is required before starting inference")
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if not math.isfinite(args.poll_interval) or not .1 <= args.poll_interval <= .2:
        parser.error("--poll-interval must be between 0.1 and 0.2")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url, "poll_interval_sec": args.poll_interval, "requested_runs": args.runs,
        "voice_api_calls": 0, "audio_sent_externally": False, "training_started": False, "runs": [],
        "first_request_is_not_necessarily_cold": True,
        "scope": "Local end-to-end latency, empty history. No speech-quality benchmark or voice-generation measurement.",
        "source_sha256_before": source_hashes()}
    destination = args.output / "summary.json"
    try:
        encoded, summary["audio"] = encode_audio(args.audio.resolve())
        client = LocalDemoClient(args.base_url)
        config = client.request("/api/chat/config")
        summary["config"] = {key: config[key] for key in ("device", "threads", "asr_model_id", "grammar_model_id",
            "reply_model_id", "transcript_normalization", "feedback_order", "inference_device",
            "transcript_recovery_provider", "transcript_recovery_model", "transcript_recovery_revision",
            "inference_warmup") if key in config}
        for index in range(1, args.runs + 1):
            result, job = run_once(client, encoded, index, args.poll_interval, args.timeout)
            summary["runs"].append(result)
            if args.save_jobs:
                (args.output / f"job-{index:02d}.json").write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"index": index, "label": result["label"], "reply_ready": result["reply_ready"],
                              "full_elapsed_sec": result["full_elapsed_sec"]}, ensure_ascii=False), flush=True)
        summary["status"] = "passed"
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error={"type": type(error).__name__, "message": str(error)[:500]})
    finally:
        summary["source_sha256_after"] = source_hashes()
        summary["runtime_source_changed"] = summary["source_sha256_before"] != summary["source_sha256_after"]
        destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(destination.resolve()),
                      "completed_runs": len(summary["runs"]), "runtime_source_changed": summary["runtime_source_changed"]}), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
