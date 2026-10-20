"""Main LLM gateway: chat and validated pronunciation feedback."""

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from jsonschema import ValidationError

if __package__:
    from .validate_contracts import read_json, validate_pair
else:
    from validate_contracts import read_json, validate_pair


ROOT = Path(__file__).resolve().parents[1]


class LLMError(RuntimeError):
    pass


def load_env(path=ROOT / ".env"):
    """Read only LLM settings; never execute values and preserve environment overrides."""
    path = Path(path)
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if not key.strip().startswith("LLM_"):
            continue
        if not separator:
            raise LLMError(f"Invalid LLM env assignment at line {number}.")
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise LLMError(f"Unclosed LLM env quote at line {number}.")
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the bearer key to a redirect target.


@dataclass
class Completion:
    content: str
    requested_model: str
    response_model: str | None
    request_id: str | None
    usage: dict
    finish_reason: str | None
    latency_sec: float


@dataclass
class FeedbackResult:
    feedback: dict
    source: str
    fallback_reason: str | None
    completions: list[Completion]
    validation_errors: list[str] = field(default_factory=list)

    def metadata(self, assessment):
        return {
            "utterance_id": assessment["utterance_id"],
            "data_kind": assessment["data_kind"],
            "feedback_source": self.source,
            "fallback_reason": self.fallback_reason,
            "contract_valid": True,
            "validation_errors": self.validation_errors,
            "completions": [
                {key: value for key, value in asdict(item).items() if key != "content"}
                for item in self.completions
            ],
        }


def fallback_feedback(assessment):
    if assessment["status"] != "ok":
        summary = "Chưa đủ thông tin để đánh giá phát âm. Bạn hãy thu lại và xác nhận câu cần đọc."
    elif assessment["errors"]:
        summary = "Model đánh dấu một số vị trí cần luyện. Bạn hãy xem các từ được đánh dấu và thử đọc lại."
    else:
        summary = "Hệ thống chưa phát hiện lỗi trong lượt đọc này."
    return {
        "schema_version": "1.0", "utterance_id": assessment["utterance_id"],
        "language": "vi", "summary": summary, "tips": [], "tts_segments": [],
    }


def parse_feedback(content):
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        if lines[0].strip().lower() in {"```", "```json"}:
            content = "\n".join(lines[1:-1])
    return json.loads(content)


class LLMClient:
    def __init__(self):
        self.key = os.environ.get("LLM_API_KEY", "").strip()
        if not self.key:
            raise LLMError("Set LLM_API_KEY in the local .env file first.")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://llm.mrdnd.dev/v1").rstrip("/")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise LLMError("LLM_BASE_URL must be an HTTPS API URL without credentials or query parameters.")
        self.model = os.environ.get("LLM_MODEL", "claude-sonnet-5").strip()
        if not self.model:
            raise LLMError("LLM_MODEL cannot be empty.")
        try:
            self.timeout = float(os.environ.get("LLM_TIMEOUT_SEC", "60"))
            self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1536"))
        except ValueError:
            raise LLMError("Invalid LLM timeout or token limit.") from None
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_tokens < 1:
            raise LLMError("LLM timeout and token limit must be positive and finite.")
        self.api = build_opener(NoRedirect())

    def complete(self, messages):
        if not isinstance(messages, list) or not messages or any(
            not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}
            or not isinstance(item.get("content"), str) or not item["content"].strip()
            for item in messages
        ):
            raise LLMError("Provide a nonempty list of role/content messages.")
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        request = Request(
            self.base_url + "/chat/completions", method="POST",
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        started = time.monotonic()
        try:
            with self.api.open(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            raise LLMError(f"LLM HTTP {exc.code}; check key, model ID and gateway availability.") from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("LLM connection failed; request was not automatically retried.") from None
        except (ValueError, UnicodeError):
            raise LLMError("LLM gateway returned invalid JSON.") from None
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if finish_reason in {"length", "content_filter"}:
                raise LLMError("LLM response was truncated or filtered.")
            if not isinstance(content, str) or not content.strip():
                raise LLMError("LLM gateway returned no text content.")
            usage = result.get("usage")
            return Completion(
                content=content.strip(), requested_model=self.model,
                response_model=result.get("model"), request_id=result.get("id"),
                usage=usage if isinstance(usage, dict) else {}, finish_reason=finish_reason,
                latency_sec=round(time.monotonic() - started, 4),
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("LLM gateway response does not match the chat completions format.") from None

    def generate_feedback(self, assessment):
        fallback = fallback_feedback(assessment)
        # Reject invalid acoustic output before sending any data to the gateway.
        validate_pair(assessment, fallback)
        if assessment["status"] != "ok":
            return FeedbackResult(fallback, "template", "assessment_not_ok", [])
        prompt = (ROOT / "prompts/pronunciation_feedback.txt").read_text(encoding="utf-8")
        schema = read_json(ROOT / "contracts/llm_feedback.schema.json")
        messages = [
            {"role": "system", "content": prompt + "\n\nJSON schema:\n" + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": json.dumps(assessment, ensure_ascii=False)},
        ]
        completions = []
        validation_errors = []
        for attempt in range(2):
            try:
                completion = self.complete(messages)
            except LLMError as exc:
                return FeedbackResult(fallback, "template", str(exc), completions, validation_errors)
            completions.append(completion)
            try:
                feedback = parse_feedback(completion.content)
                validate_pair(assessment, feedback)
            except (ValueError, ValidationError) as exc:
                if isinstance(exc, ValidationError):
                    path = ".".join(str(part) for part in exc.absolute_path) or "root"
                    detail = f"{path}: {exc.message[:300]}"
                elif isinstance(exc, json.JSONDecodeError):
                    detail = f"Invalid JSON at line {exc.lineno}, column {exc.colno}."
                else:
                    detail = str(exc)
                validation_errors.append(detail)
                if attempt == 0:
                    messages.extend([
                        {"role": "assistant", "content": completion.content},
                        {"role": "user", "content": "JSON chưa đúng schema hoặc liên kết với assessment. "
                         "Hãy tạo lại đúng một JSON, giữ utterance_id và câu tham chiếu; "
                         "chỉ dùng error_id đã có. Không thêm Markdown. Lỗi cần sửa: " + detail},
                    ])
                continue
            return FeedbackResult(feedback, "llm", None, completions, validation_errors)
        return FeedbackResult(fallback, "template", "invalid_feedback_after_repair", completions, validation_errors)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--message", help="Send a chat message")
    mode.add_argument("--assessment", type=Path, help="Generate validated feedback from an MDD assessment")
    parser.add_argument("--output", type=Path, help="Save chat content+metadata or feedback JSON with a .meta.json sidecar")
    parser.add_argument("--trace", action="store_true", help="Also save raw feedback completions in a local .trace.json file")
    args = parser.parse_args()
    if args.trace and (args.assessment is None or args.output is None):
        parser.error("--trace requires --assessment and --output")
    try:
        load_env(args.env_file)
        client = LLMClient()
        if args.message is not None:
            result = client.complete([
                {"role": "system", "content": "Bạn là trợ giảng tiếng Anh. Trả lời bằng tiếng Việt. "
                 "Khi chưa có kết quả từ model âm học, không tự kết luận lỗi phát âm."},
                {"role": "user", "content": args.message},
            ])
            print(result.content)
            if args.output:
                write_json(args.output, asdict(result))
        else:
            assessment = read_json(args.assessment)
            result = client.generate_feedback(assessment)
            print(json.dumps(result.feedback, ensure_ascii=False, indent=2))
            print("feedback_source=" + result.source, file=sys.stderr)
            if args.output:
                write_json(args.output, result.feedback)
                write_json(args.output.with_suffix(".meta.json"), result.metadata(assessment))
                if args.trace:
                    write_json(args.output.with_suffix(".trace.json"), {
                        "utterance_id": assessment["utterance_id"],
                        "data_kind": assessment["data_kind"],
                        "completions": [asdict(item) for item in result.completions],
                    })
    except (LLMError, OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        message = str(exc) if isinstance(exc, LLMError) else "Invalid input, configuration or output file."
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
