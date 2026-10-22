"""Convert predicted phones and canonical reference only; never read gold perception."""
from __future__ import annotations

from copy import deepcopy
import json
import sys
import unicodedata

from scripts.l2arctic_plus_data import ROOT, json_digest, rooted
from scripts.prepare_feedback_data import digest

sys.path.insert(0, str(ROOT / "src"))
from mdd.metrics import align_phones


def word_key(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold().replace("’", "'") if c.isalnum() or c == "'")


def predicted_condition(native_row, canonical_reference, predicted_phones):
    """Return the same coaching format from a deployable phone alignment.

    Insertions use the following canonical phone's word, matching demo/scorer.
    Like the published word dictionary, repeated word keys keep the last occurrence.
    No timestamps or confidence are invented.
    """
    result = deepcopy(native_row)
    canonical = canonical_reference["canonical_phones"]
    mapping = canonical_reference["canonical_word_indices"]
    words = canonical_reference["word_intervals"]
    keys = {word_key(key): key for key in native_row["annotation_info"]}
    try:
        if not canonical or len(mapping) != len(canonical) or len(keys) != len(native_row["annotation_info"]):
            raise ValueError("Canonical mapping or word keys are ambiguous")
        indexed = {word["index"]: keys[word_key(word["text"])] for word in words}
        if set(indexed.values()) != set(native_row["annotation_info"]):
            raise ValueError("Published and canonical word keys differ")
        by_index = {index: [] for index in indexed}
        for event in align_phones(canonical, predicted_phones):
            index = mapping[min(event["position"], len(mapping) - 1)]
            if index not in indexed:
                raise ValueError("A predicted phone cannot be assigned to a canonical word")
            expected, observed = event["expected_phone"], event["observed_phone"]
            if event["kind"] == "match":
                label = expected
            else:
                code = {"substitution": "s", "deletion": "d", "addition": "a"}[event["kind"]]
                label = f"{expected or 'sil'}, {observed or 'sil'}, {code}"
            by_index[index].append(label)
        conditioned = {}
        for index in sorted(indexed):
            conditioned[indexed[index]] = by_index[index]
        result["annotation_info"] = conditioned
        result["input_alignment_status"] = "ok"
    except (KeyError, ValueError, TypeError) as error:
        # This is abstention, not a replacement by gold or a teaching template.
        result["annotation_info"] = {}
        result["input_alignment_status"] = "uncertain"
        result["input_alignment_error"] = str(error)
    result["condition_sha256"] = json_digest({"text": result["reference_text"], "phones": list(predicted_phones),
                                             "annotation_info": result["annotation_info"]})
    return result


def load_condition(rows, manifest, directory, *, allow_technical=False):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    predictions_path = directory / "predictions.jsonl"
    if digest(predictions_path) != metrics["prediction_sha256"]:
        raise ValueError("Audio prediction checksum mismatch")
    metadata = metrics["training_metadata"]
    if metrics["purpose"] != "scientific_training" and not allow_technical:
        raise ValueError("Technical audio predictions cannot become scientific cascade evidence")
    if metrics.get("limited_technical_subset") and not allow_technical:
        raise ValueError("Audio evaluation is only a limited technical subset")
    if not metadata.get("completed_training", {}).get("status") == "completed" and not allow_technical:
        raise ValueError("Audio training has not completed")
    expected = manifest.get("cascade_protocol_sha256")
    actual = metadata["provenance"]["files"]["protocol"]["sha256"]
    if not expected or expected != actual:
        raise ValueError("Audio checkpoint used a different split; old A/B checkpoints are excluded")
    config = metadata["effective_config"]
    reference_rows = {row["utterance_id"]: row for row in
        (json.loads(line) for line in rooted(config["annotations"]).read_text(encoding="utf-8").splitlines() if line.strip())}
    predictions = {row["utterance_id"]: row for row in
        (json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    if set(row["utterance_id"] for row in rows) - set(predictions):
        raise ValueError("Audio predictions do not cover the requested coaching rows")
    conditioned = [predicted_condition(row, reference_rows[row["utterance_id"]], predictions[row["utterance_id"]]["predicted_phones"]) for row in rows]
    return conditioned, {"condition": "wav2vec2_predicted_phones", "audio_stage": metadata["stage"],
        "audio_checkpoint_sha256": metrics["checkpoint_sha256"], "audio_predictions_sha256": digest(predictions_path),
        "audio_protocol_sha256": actual, "bridge_sha256": digest(__file__),
        "uncertain_input_rows": sum(row["input_alignment_status"] != "ok" for row in conditioned),
        "conditions_sha256": json_digest([row["condition_sha256"] for row in conditioned])}
