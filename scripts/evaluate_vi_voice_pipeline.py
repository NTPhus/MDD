"""Evaluate public Vietnamese-accented English clips through the local demo.

Inference only: no model training, voice synthesis, external audio submission,
reference injection, threshold fitting, or automatic semantic quality judge.
Published reading prompts may differ from words actually realized by a speaker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_demo_latency import LocalDemoClient, encode_audio, local_url, run_once, source_hashes
from scripts.evaluate_transcript_normalization import distance, words

EXPECTED_MODELS = {"asr_model_id": "wav2vec2-vi-en-v1", "grammar_model_id": "qwen-grammar", "reply_model_id": "qwen-base"}
SUPPORTED_ASR = {'wav2vec2-vi-en-v1', 'facebook/wav2vec2-base-960h', 'whisper-small-en'}
CONFIG_KEYS = (*EXPECTED_MODELS, "device", "threads", "acoustic_device", "transcript_normalization", "transcript_recovery",
               "transcript_recovery_provider", "transcript_recovery_model", "transcript_recovery_revision",
               "pronunciation_reference_policy", "show_raw_asr", "asr_backend", "asr_revision", "asr_compute_type",
               "feedback_order", "inference_device")
SAFE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}\Z")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_source_hashes(config=None):
    result = source_hashes()
    for relative in ('scripts/transcript_recovery.py', 'scripts/transcript_normalization.py',
                     'scripts/conversation_context.py', 'scripts/grammar_inference.py',
                     'src/mdd/phonetic_candidates.py', 'src/mdd/demo_model_registry.py'):
        result[relative] = digest(ROOT / relative)
    bert_module = ROOT / 'scripts/bert_transcript_recovery.py'
    # An absent optional module remains compatible with older Qwen-only runs.
    # A configured BERT deployment must freeze the actual implementation bytes.
    if bert_module.is_file() or (config or {}).get('transcript_recovery_provider') == 'bert':
        result['scripts/bert_transcript_recovery.py'] = digest(bert_module)
    for relative in ('scripts/whisper_asr.py', 'scripts/sentence_recovery.py'):
        if (ROOT / relative).is_file():
            result[relative] = digest(ROOT / relative)
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rooted(path, root=ROOT):
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Evaluation paths must remain in the workspace")
    return resolved


def load_manifest(path, root=ROOT):
    """Freeze all cases and verify bytes before opening any model/API request."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest needs a nonempty cases list")
    ids = set()
    for case in cases:
        identifier = case.get("id", "")
        if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier) or identifier in ids:
            raise ValueError("Case IDs must be unique safe filenames")
        ids.add(identifier)
        for key in ("corpus", "speaker_id", "audio", "audio_sha256", "reference_text", "reference_kind", "source_url", "split_notes"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Case {identifier} is missing {key}")
        if not words(case["reference_text"]):
            raise ValueError("Reference must contain words")
        if digest(rooted(case["audio"], root)) != case["audio_sha256"]:
            raise ValueError("Audio SHA256 mismatch: " + identifier)
    return manifest, hashlib.sha256(raw).hexdigest()


def frozen_config(config):
    for key, expected in EXPECTED_MODELS.items():
        if key == 'asr_model_id' and config.get(key) in SUPPORTED_ASR:
            continue
        if config.get(key) != expected:
            raise ValueError(f"Expected {key}={expected}; got {config.get(key)!r}")
    if config.get("speech_ready") is not True:
        raise ValueError("The local speech pipeline is not ready")
    if config.get('transcript_recovery_provider') == 'bert':
        if config.get('transcript_recovery_model') != 'google-bert/bert-base-uncased':
            raise ValueError('BERT recovery must identify its configured pretrained model')
        if not re.fullmatch(r'[0-9a-f]{40}', config.get('transcript_recovery_revision') or ''):
            raise ValueError('BERT recovery must freeze an exact Hugging Face revision')
    reference_policy = config.get('pronunciation_reference_policy')
    if reference_policy not in (None, 'asr', 'bert_output'):
        raise ValueError('Unknown pronunciation reference policy')
    if reference_policy == 'bert_output' and (
            config.get('transcript_recovery_provider') != 'bert' or not config.get('transcript_recovery')):
        raise ValueError('The bert_output pronunciation reference policy requires BERT recovery')
    if 'show_raw_asr' in config and not isinstance(config['show_raw_asr'], bool):
        raise ValueError('show_raw_asr must be a boolean')
    return {key: config[key] for key in CONFIG_KEYS if key in config}


def verify_job_models(job, config=None):
    speech, metadata = job.get("speech") or {}, job.get("metadata") or {}
    actual = {"asr_model_id": speech.get("asr_model"),
              "grammar_model_id": (speech.get("rewrite") or {}).get("model_id"),
              "reply_model_id": metadata.get("reply_model_id")}
    expected = {**EXPECTED_MODELS, 'asr_model_id': (config or {}).get('asr_model_id', EXPECTED_MODELS['asr_model_id'])}
    if actual != expected:
        raise ValueError("Completed job did not use the frozen pipeline models: " + repr(actual))


def verify_job_recovery(job, config):
    """Check the completed inference against its frozen, server-owned provider."""
    speech = job.get('speech') or {}
    recovery = speech.get('interpretation') or {}
    if (job.get('sentence_voice') or {}).get('status') != 'not_requested':
        raise ValueError('Unexpected sentence TTS on an evaluation-owned job')
    if not config.get('transcript_recovery'):
        if config.get('pronunciation_reference_policy') == 'bert_output':
            raise ValueError('The bert_output pronunciation reference policy requires BERT recovery')
        if config.get('pronunciation_reference_policy') == 'asr' and (
                speech.get('reference_text') != speech.get('transcript')
                or speech.get('reference_source') != 'asr'
                or speech.get('pronunciation_reference_ready') is not True
                or recovery.get('applied_as_pronunciation_reference') not in (None, False)):
            raise ValueError('The pronunciation reference does not match the selected ASR transcript')
        return
    if (recovery.get('version') != config['transcript_recovery']
            or recovery.get('original') != speech.get('transcript')):
        raise ValueError('The job did not preserve the configured recovery provenance')
    for config_key, recovery_key in (
            ('transcript_recovery_provider', 'provider'),
            ('transcript_recovery_model', 'model_id'),
            ('transcript_recovery_revision', 'revision')):
        if config.get(config_key) is not None and recovery.get(recovery_key) != config[config_key]:
            raise ValueError('Recovery model identity differs from frozen config: ' + config_key)
    if config.get('transcript_recovery_provider') == 'bert' and recovery.get('device') != 'cpu':
        raise ValueError('The configured BERT recovery did not report CPU inference')
    if config.get('transcript_recovery_provider') == 'bert' and recovery.get('reference_confirmed') is not False:
        raise ValueError('BERT recovery cannot claim an inferred transcript was confirmed')
    reference_policy = config.get('pronunciation_reference_policy')
    if reference_policy == 'bert_output':
        if config.get('transcript_recovery_provider') != 'bert':
            raise ValueError('The bert_output pronunciation reference policy requires BERT recovery')
        if recovery.get('status') in {'accepted', 'unchanged'}:
            recovered_text = recovery.get('text')
            if (not isinstance(recovered_text, str) or not recovered_text.strip()
                    or speech.get('reference_text') != recovered_text
                    or speech.get('reference_source') != 'bert_inferred'
                    or speech.get('pronunciation_reference_ready') is not True
                    or recovery.get('applied_as_pronunciation_reference') is not True):
                raise ValueError('BERT pronunciation reference does not match the authorized recovery')
        elif (speech.get('reference_text') != speech.get('transcript')
              or speech.get('reference_source') != 'asr'
              or recovery.get('applied_as_pronunciation_reference') not in (None, False)):
            raise ValueError('An unavailable or ambiguous recovery replaced the acoustic reference')
    elif (reference_policy not in (None, 'asr')
          or speech.get('reference_text') != speech.get('transcript')
          or speech.get('reference_source') != 'asr'
          or recovery.get('applied_as_pronunciation_reference') not in (None, False)):
        raise ValueError('An unconfirmed inference replaced the acoustic reference')


class RecordingClient:
    """Capture the final public job; never cancel another client's work."""
    def __init__(self, client, *, clock=time.monotonic, sleep=time.sleep):
        self.client, self.clock, self.sleep = client, clock, sleep
        self.last_job = None
        self.busy_wait_sec = 0.

    def request(self, route, payload=None):
        started = self.clock()
        while True:
            try:
                result = self.client.request(route, payload)
                if route == "/api/chat/speech":
                    self.busy_wait_sec = self.clock() - started
                if isinstance(result.get("id"), str):
                    self.last_job = result
                return result
            except urllib.error.HTTPError as error:
                # Retry only rejected submissions; no job was accepted in 409.
                if route != "/api/chat/speech" or error.code != 409:
                    raise
                try:
                    response = json.load(error)
                    detail = response.get("error")
                    code = detail.get("code") if isinstance(detail, dict) else detail
                except (ValueError, AttributeError):
                    raise error
                if code != "chat_busy" or self.clock() - started >= 60:
                    raise error
                self.sleep(1.)


def record_case(case, job, timing=None, error=None):
    speech, metadata = job.get("speech") or {}, job.get("metadata") or {}
    normalization = speech.get("normalization") or {}
    transcript = speech.get("transcript") or ""
    normalized = normalization.get("text") or transcript
    conversation = speech.get("conversation_text") or normalized
    assessment = speech.get("assessment") or {}
    comparison = speech.get("comparison") or {}
    statuses = dict(Counter(word.get("status", "unknown") for word in comparison.get("words", [])))
    row = {key: case[key] for key in ("id", "corpus", "speaker_id", "audio", "audio_sha256", "reference_text", "reference_kind", "source_url", "split_notes")}
    row.update(status=job.get("status", "unavailable") if error is None else "error",
        error=error or job.get("error"), job_id=job.get("id"), transcript=transcript,
        normalized=normalized, conversation_text=conversation, normalization=normalization,
        interpretation=speech.get('interpretation') or {},
        pronunciation_reference_text=speech.get('reference_text'),
        pronunciation_reference_ready=speech.get('pronunciation_reference_ready'),
        conversation_source=speech.get('conversation_source') or metadata.get('conversation_source'),
        reply=job.get("text") or "", rewrite=speech.get("rewrite") or {},
        assessment={"status": assessment.get("status"), "counts": comparison.get("counts") or {},
                    "word_status_counts": statuses, "reference_source": speech.get("reference_source")},
        transcript_clarity=speech.get("transcript_clarity") or {},
        latencies=timing or {}, qwen_runtime=metadata.get("qwen_runtime") or metadata.get("generation_runtime") or {},
        models={key: speech.get(key) for key in ("asr_model", "phoneme_model")},
        reply_diagnostics={key: value for key, value in metadata.items()
                           if key == "reply_source" or any(term in key for term in ("fallback", "echo", "attempt", "blocked", "clarification"))},
        human_review={"status": "pending", "transcript_matches_audio": None,
                      "reply_meaningful": None, "reply_relevant_to_actual_audio": None, "notes": ""})
    return row


def aggregate(records):
    ready = [row for row in records if row["status"] == "ready"]
    result = {"completed_cases": len(ready), "total_records": len(records),
              "statuses": dict(Counter(row["status"] for row in records)),
              "human_semantic_reviews_completed": sum(row.get("human_review", {}).get("status") == "reviewed" for row in records)}
    for field in ("transcript", "normalized", "conversation_text"):
        word_count = sum(len(words(row["reference_text"])) for row in ready)
        char_count = sum(len(" ".join(words(row["reference_text"]))) for row in ready)
        word_errors = sum(distance(words(row["reference_text"]), words(row[field])) for row in ready)
        char_errors = sum(distance(" ".join(words(row["reference_text"])), " ".join(words(row[field]))) for row in ready)
        exact = sum(words(row["reference_text"]) == words(row[field]) for row in ready)
        result[field] = {"reference_words": word_count, "word_errors": word_errors,
            "reference_chars": char_count, "char_errors": char_errors,
            "wer": word_errors / word_count if word_count else None,
            "cer": char_errors / char_count if char_count else None,
            "exact_matches": exact, "exact_match_rate": exact / len(ready) if ready else None}
    # These are literal diagnostic counts, not an estimate of semantic quality.
    result["literal_reply_equals_transcript"] = sum(words(row["reply"]) == words(row["transcript"]) and bool(words(row["reply"])) for row in ready)
    result["literal_repeated_reply_count"] = sum(count - 1 for count in Counter(row["reply"].strip() for row in ready if row["reply"].strip()).values())
    result["empty_replies"] = sum(not row["reply"].strip() for row in ready)
    result["reply_sources"] = dict(Counter(row.get("reply_diagnostics", {}).get("reply_source", "qwen") for row in ready))
    result["fallback_replies"] = sum(row.get("reply_diagnostics", {}).get("reply_source") in {"clarification_fallback", "conversation_unavailable"} for row in ready)
    result['interpretation_statuses'] = dict(Counter(row.get('interpretation', {}).get('status', 'not_requested') for row in ready))
    result['interpretation_providers'] = dict(Counter(row.get('interpretation', {}).get('provider', 'not_recorded') for row in ready))
    result['interpretation_models'] = dict(Counter(row.get('interpretation', {}).get('model_id', 'not_recorded') for row in ready))
    result['conversation_wording_changed'] = sum(words(row['transcript']) != words(row['conversation_text']) for row in ready)
    result['inferred_interpretations_requiring_confirmation'] = sum(
        row.get('conversation_source') == 'asr_interpreted' and row.get('interpretation', {}).get('requires_confirmation') is True
        for row in ready)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", type=local_url)
    parser.add_argument("--timeout", type=float, default=420.)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("--run is required to submit local audio inference")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout and limit must be positive")
    args.manifest, args.output = rooted(args.manifest), rooted(args.output)
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest, manifest_sha = load_manifest(args.manifest)
    selected = manifest["cases"][:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    client = LocalDemoClient(args.base_url)
    config = frozen_config(client.request("/api/chat/config"))
    identity = {"manifest_sha256": manifest_sha, "case_ids": [case["id"] for case in selected],
                "config": config, "source_sha256": runtime_source_hashes(config),
                "runner_sha256": digest(Path(__file__)), "base_url": args.base_url}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = args.output / "results.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if not args.resume or previous.get("identity_sha256") != identity_sha:
            raise ValueError("Existing results require --resume and exactly matching manifest, runtime and config")
        records = previous.get("records", [])
    else:
        previous, records = {}, []
    output = {"schema": "vi-voice-local-pipeline-v1", "status": "running", "identity": identity,
        "identity_sha256": identity_sha, "started_at_utc": previous.get("started_at_utc") or datetime.now(timezone.utc).isoformat(),
        "voice_api_calls": 0, "audio_sent_externally": False, "training_started": False,
        "references_submitted_to_models": False, "history": [], "records": records,
        "limitations": ["Published reading prompts are not verified transcripts of every realized word.",
            "Selection is fixed before inference; this is exploratory evaluation, not threshold fitting.",
            "WER/CER ignore capitalization and punctuation; no semantic accuracy score is inferred.",
            "Recovery and Qwen conversation receive recognized text and configured local acoustic candidates, never the published reference.",
            ("Accepted BERT output is used as the pronunciation reference under the explicit bert_output policy; it is inferred and not user-confirmed."
             if config.get('pronunciation_reference_policy') == 'bert_output'
             else "An inferred recovery is not proof of the realized spoken words; raw pronunciation references remain unchanged unless confirmed."),
            "No AI33 or other speech synthesis is measured. Timing includes local polling overhead."]}
    jobs_dir = args.output / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    def save():
        output["aggregate"] = aggregate(records)
        write_json(destination, output)
    save()
    try:
        for index, case in enumerate(selected, 1):
            existing = next((row for row in records if row["id"] == case["id"]), None)
            if existing is not None and existing["status"] == "ready":
                continue
            if digest(args.manifest) != manifest_sha or digest(rooted(case["audio"])) != case["audio_sha256"]:
                raise ValueError("Manifest or audio changed since freezing")
            if frozen_config(client.request("/api/chat/config")) != config:
                raise ValueError("Local pipeline config changed during evaluation")
            recording = RecordingClient(client)
            try:
                encoded, audio = encode_audio(rooted(case["audio"]))
                if audio["source_sha256"] != case["audio_sha256"]:
                    raise ValueError("Audio changed during encoding")
                timing, job = run_once(recording, encoded, index, .15, args.timeout)
                verify_job_models(job, config)
                verify_job_recovery(job, config)
                timing["submission_wait_sec"] = round(recording.busy_wait_sec, 3)
                row = record_case(case, job, timing)
                row["submitted_audio"] = audio
            except Exception as error:
                job = recording.last_job or {}
                row = record_case(case, job, error={"type": type(error).__name__, "message": str(error)[:500]})
            write_json(jobs_dir / (case["id"] + ".json"), job)
            if existing is not None:
                records.remove(existing)
            records.append(row)
            save()
            print(json.dumps({"case": case["id"], "status": row["status"], "transcript": row["transcript"],
                              'conversation_text': row['conversation_text'], 'recovery_status': row['interpretation'].get('status'),
                              "reply": row["reply"]}, ensure_ascii=False), flush=True)
        output["status"] = "completed" if all(row["status"] == "ready" for row in records) else "completed_with_errors"
    except BaseException as error:
        output["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        output["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        output["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        output["source_sha256_after"] = runtime_source_hashes(config)
        output["runtime_source_changed"] = output["source_sha256_after"] != identity["source_sha256"]
        output["manifest_changed"] = digest(args.manifest) != manifest_sha
        save()
    return 0 if output["status"] == "completed" and not output["runtime_source_changed"] and not output["manifest_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
