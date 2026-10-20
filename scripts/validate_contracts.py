"""Validate MDD/LLM JSON shapes and cross-document references; no inference calls."""

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pair(assessment, feedback):
    for name, payload in (("mdd_assessment", assessment), ("llm_feedback", feedback)):
        schema = read_json(ROOT / "contracts" / f"{name}.schema.json")
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)

    indices = [word["index"] for word in assessment["words"]]
    require(indices == list(range(len(indices))), "Word indices must be contiguous and start at zero")
    error_ids = [error["id"] for error in assessment["errors"]]
    require(len(error_ids) == len(set(error_ids)), "Error IDs must be unique")
    require(feedback["utterance_id"] == assessment["utterance_id"], "Utterance IDs must match")
    require(assessment["status"] == "ok" or not assessment["errors"], "Uncertain/invalid audio cannot carry diagnoses")

    for error in assessment["errors"]:
        require(error["word_index"] in indices, "Error references an unknown word")
        start, end = error["start_sec"], error["end_sec"]
        require((start is None) == (end is None), "Both timestamps must be null or present")
        require(start is None or end >= start, "End timestamp precedes start")
        require(assessment["confidence_calibrated"] or error["confidence"] is None,
                "Uncalibrated confidence must be null")
        expected, observed = error["expected_phone"], error["observed_phone"]
        if error["type"] == "substitution":
            require(expected is not None and observed is not None, "Substitution requires both phone labels")
        elif error["type"] == "deletion":
            require(expected is not None and observed is None and error["observed_ipa"] is None,
                    "Deletion requires an expected phone and no observed phone")
        else:
            require(expected is None and error["expected_ipa"] is None and observed is not None,
                    "Addition requires an observed phone and no expected phone")

    tip_ids = [tip["error_id"] for tip in feedback["tips"]]
    require(set(tip_ids) <= set(error_ids), "LLM invented an error ID")
    require(len(tip_ids) == len(set(tip_ids)), "Duplicate feedback for an error ID")
    if assessment["status"] != "ok":
        require(not feedback["tips"] and not feedback["tts_segments"],
                "Invalid audio must produce a retry instruction without diagnosis or TTS")
    for segment in feedback["tts_segments"]:
        if segment["purpose"] == "reference":
            require(segment["text"] == assessment["reference_text"], "LLM changed the reference text")
            require(segment["language"] == assessment["target_locale"], "Reference TTS locale must match")
        else:
            expected_locale = "vi-VN" if feedback["language"] == "vi" else assessment["target_locale"]
            require(segment["language"] == expected_locale, "Feedback TTS locale must match feedback language")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assessment", type=Path)
    parser.add_argument("feedback", type=Path)
    args = parser.parse_args()
    assessment, feedback = read_json(args.assessment), read_json(args.feedback)
    validate_pair(assessment, feedback)
    print(f"Valid contracts for {assessment['utterance_id']}; data_kind={assessment['data_kind']}")
    print("Schema/reference validation does not verify the accuracy of the model or teaching advice.")


if __name__ == "__main__":
    main()
