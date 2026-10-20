"""AI33 TTS client and CLI. Python standard library only; no local TTS model."""

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/generated/tts/ai33"


class AI33Error(RuntimeError):
    pass


def load_env(path):
    """Read AI33_KEY=value settings without executing shell code; environment wins."""
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not separator:
            raise AI33Error(f"Invalid env assignment at line {number}.")
        if not key.startswith("AI33_"):
            continue
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise AI33Error(f"Unclosed env quote at line {number}.")
            value = value[1:-1]
        os.environ.setdefault(key, value)


def objects(payload):
    """Known task envelopes; public discovery docs do not fix a response schema."""
    pending = deque([payload])
    while pending:
        item = pending.popleft()
        if isinstance(item, dict):
            yield item
            pending.extend(item.get(key) for key in ("data", "result", "metadata", "output"))


def field(payload, *names):
    for item in objects(payload):
        for name in names:
            value = item.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def positive_number(name, default):
    try:
        number = float(os.environ.get(name, str(default)))
    except ValueError:
        raise AI33Error(f"{name} must be a number.") from None
    if not math.isfinite(number) or number <= 0:
        raise AI33Error(f"{name} must be finite and positive.")
    return number


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward an authenticated API request to a different host.
        return None


class AI33TTS:
    def __init__(self, cache_dir=CACHE):
        self.key = os.environ.get("AI33_API_KEY", "").strip()
        if not self.key:
            raise AI33Error("Set AI33_API_KEY in the local .env file first (see .env.example).")
        self.base_url = os.environ.get("AI33_BASE_URL", "https://api.ai33.pro").rstrip("/")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path):
            raise AI33Error("AI33_BASE_URL must be an HTTPS origin, e.g. https://api.ai33.pro.")
        self.speed = positive_number("AI33_SPEED", 1)
        if not 0.5 <= self.speed <= 1.5:
            raise AI33Error("AI33_SPEED must be between 0.5 and 1.5.")
        self.timeout = positive_number("AI33_TIMEOUT_SEC", 120)
        self.interval = positive_number("AI33_POLL_INTERVAL_SEC", 2)
        self.cache_dir = Path(cache_dir)
        self.api = build_opener(NoRedirect())
        self.downloads = build_opener()

    def request(self, method, path, payload=None, timeout=30):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(self.base_url + path, data=body, method=method, headers={
            "xi-api-key": self.key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        try:
            with self.api.open(req, timeout=timeout) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            # Do not print response bodies: these can echo credentials or inputs.
            raise AI33Error(f"AI33 HTTP {exc.code} for {method} {path.split('?')[0]}.") from None
        except (URLError, TimeoutError, OSError):
            message = "AI33 connection failed."
            if method == "POST":
                message += " No automatic retry; check task history before submitting again."
            raise AI33Error(message) from None
        except (ValueError, UnicodeError):
            raise AI33Error("AI33 returned invalid JSON; check the account API documentation.") from None
        if isinstance(result, dict) and (result.get("success") is False or result.get("error")):
            raise AI33Error("AI33 rejected the request; check task history and account settings.")
        return result

    def list_voices(self, provider, search="", page=1):
        query = urlencode({"provider": provider, "search": search, "page": page})
        return self.request("GET", "/v3/voices?" + query)

    def wait(self, task_id):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            result = self.request("GET", "/v1/task/" + quote(task_id, safe=""),
                                  timeout=max(0.001, min(30, remaining)))
            status = (field(result, "status", "state") or "").lower()
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise AI33Error(f"AI33 task failed ({status}); cached task ID retained.")
            audio_url = field(result, "audio_url", "output_uri", "output_url")
            if audio_url:
                return audio_url
            if status in {"done", "completed", "success", "succeeded", "finished"}:
                raise AI33Error("Task completed but audio URL was not recognized. "
                                "Check the authenticated response schema; cached task ID retained.")
            if not status:
                raise AI33Error("Task status was not recognized. Check the authenticated response schema.")
            time.sleep(max(0, min(self.interval, deadline - time.monotonic())))
        raise AI33Error("AI33 task timed out; run the same command again to resume polling.")

    def download(self, url, stem):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise AI33Error("AI33 returned an invalid HTTPS audio URL.")
        # CDN download deliberately has no xi-api-key or Authorization header.
        try:
            with self.downloads.open(Request(url), timeout=30) as response:
                content_type = response.headers.get_content_type()
                data = response.read(25 * 1024 * 1024 + 1)
        except (HTTPError, URLError, TimeoutError, OSError):
            raise AI33Error("Audio download failed; rerun to resume the existing task.") from None
        if not data or len(data) > 25 * 1024 * 1024:
            raise AI33Error("Audio is empty or exceeds this demo's 25 MiB limit.")
        suffix = None
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            suffix = ".wav"
        elif data.startswith(b"ID3") or (len(data) > 1 and data[0] == 255 and data[1] & 224 == 224):
            suffix = ".mp3"
        elif data.startswith(b"OggS"):
            suffix = ".ogg"
        elif data.startswith(b"fLaC"):
            suffix = ".flac"
        elif data[4:8] == b"ftyp":
            suffix = ".m4a"
        if suffix is None or (not content_type.startswith("audio/")
                              and content_type not in {"application/octet-stream", "video/mp4"}):
            raise AI33Error("Download is not a supported audio file; check the task response schema.")
        path = stem.with_suffix(suffix)
        path.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()

    def synthesize(self, text, locale="en-US", voice_id=None):
        """Return a cached local audio Path; reruns resume a previously submitted task."""
        if locale not in {"en-US", "en-GB", "vi-VN"}:
            raise AI33Error("Supported locales: en-US, en-GB, vi-VN.")
        if not isinstance(text, str) or not text.strip() or len(text) > 5000:
            raise AI33Error("Provide 1 to 5000 characters for a demo TTS segment.")
        setting = "AI33_VOICE_VI" if locale == "vi-VN" else "AI33_VOICE_EN"
        voice_id = voice_id or os.environ.get(setting, "").strip()
        if not voice_id:
            raise AI33Error(f"Set {setting} to a voice ID from your account for {locale}.")
        payload = {"text": text, "voice_id": voice_id, "speed": self.speed}
        cache_input = {"base_url": self.base_url, "locale": locale, "payload": payload}
        digest = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = self.cache_dir / (digest + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("audio_file"):
                audio = self.cache_dir / receipt["audio_file"]
                if audio.is_file() and hashlib.sha256(audio.read_bytes()).hexdigest() == receipt.get("audio_sha256"):
                    return audio
            task_id = receipt["task_id"]
        else:
            result = self.request("POST", "/v3/text-to-speech", payload)
            task_id = field(result, "task_id")
            if not task_id:
                raise AI33Error("Submission returned no recognized task_id. Check task history "
                                "before submitting again; the response schema may differ.")
            receipt = {
                "provider": "ai33", "base_url": self.base_url, "task_id": task_id,
                "locale": locale, "voice_id": voice_id, "speed": self.speed,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            self.save_receipt(receipt_path, receipt)
        url = self.wait(task_id)
        audio, audio_hash = self.download(url, self.cache_dir / digest)
        receipt.update(audio_file=audio.name, audio_sha256=audio_hash)
        self.save_receipt(receipt_path, receipt)
        return audio

    @staticmethod
    def save_receipt(path, receipt):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text")
    mode.add_argument("--list-voices", action="store_true")
    parser.add_argument("--locale", choices=["en-US", "en-GB", "vi-VN"], default="en-US")
    parser.add_argument("--voice-id", help="Override AI33_VOICE_EN or AI33_VOICE_VI for one call")
    parser.add_argument("--provider", choices=["elevenlabs", "minimax", "edge", "vbee", "fishaudio", "clone"], default="edge")
    parser.add_argument("--search", default="")
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args()
    if args.page < 1:
        parser.error("--page must be at least 1")
    try:
        load_env(args.env_file)
        client = AI33TTS(args.cache_dir)
        if args.list_voices:
            print(json.dumps(client.list_voices(args.provider, args.search, args.page), ensure_ascii=False, indent=2))
        else:
            print(client.synthesize(args.text, args.locale, args.voice_id))
    except (AI33Error, OSError, ValueError, KeyError):
        # Known adapter errors contain no credentials; never dump raw API responses.
        error = sys.exc_info()[1]
        message = str(error) if isinstance(error, AI33Error) else "Local TTS configuration/cache error."
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
