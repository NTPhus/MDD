"""Protect common holdouts, deployment inputs and before/after denominators."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.cascade_bridge import predicted_condition, load_condition
from scripts.l2arctic_plus_data import messages, ROOT
from scripts.prepare_cascade_data import CONFIG, check_common_split
from scripts.prepare_feedback_data import digest
from scripts.published_feedback_metrics import parse_prediction, summarize, paired_bootstrap
from scripts.report_cascade import verify_pair


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.native = {"utterance_id": "fixture/one", "reference_text": "say the",
            "annotation_info": {"say": ["S", "EY"], "the": ["DH, Z, s", "AH"]},
            "feedback": {"the": [{"issue": "DH became Z", "suggestion": "Practice voiced DH"}]}}
        self.reference = {"canonical_phones": ["S", "EY", "DH", "AH"], "canonical_word_indices": [0,0,1,1],
            "word_intervals": [{"index":0,"text":"say"},{"index":1,"text":"the"}],
            "perceived_phones": ["private-gold"], "gold_error_events": [{"secret":True}]}

    def test_deployment_condition_ignores_gold_perception_and_feedback(self):
        first = predicted_condition(self.native, self.reference, ["S","EY","Z","AH"])
        changed_ref, changed_native = deepcopy(self.reference), deepcopy(self.native)
        changed_ref["perceived_phones"] = ["completely-different"]
        changed_ref["gold_error_events"] = []
        changed_native["feedback"] = {"secret-target-not-for-input": []}
        second = predicted_condition(changed_native, changed_ref, ["S","EY","Z","AH"])
        self.assertEqual(messages(first), messages(second))
        self.assertEqual(first["annotation_info"]["the"], ["DH, Z, s", "AH"])
        self.assertNotIn("private-gold", json.dumps(messages(first)))

    def test_unmappable_words_abstain_without_using_native_gold(self):
        reference = deepcopy(self.reference)
        reference["canonical_word_indices"][2] = None
        result = predicted_condition(self.native, reference, ["S","EY","Z","AH"])
        self.assertEqual(result["input_alignment_status"], "uncertain")
        self.assertEqual(result["annotation_info"], {})

    def test_duplicate_json_and_invented_fields_are_invalid(self):
        for value in ('{"say":[],"say":[]}', '{"say":[{"issue":"x","suggestion":"y","invented":1}]}'):
            with self.assertRaises(ValueError): parse_prediction(value)

    def test_invalid_generation_gets_zero_credit_and_stays_in_denominator(self):
        good = {"utterance_id":"a", "speaker_id":"speaker-a", "reference_feedback":self.native["feedback"],
            "prediction":self.native["feedback"], "valid_json":True, "word_order":["say","the"], "generation_sec":1.}
        invalid = {**deepcopy(good), "utterance_id":"b", "speaker_id":"speaker-b", "valid_json":False}
        summary, scores = summarize([good,invalid])
        self.assertEqual(summary["rows"],2)
        self.assertEqual(summary["json_valid_percent"],50)
        self.assertAlmostEqual(summary["suggestion_rougeL_on_reference_error_rows"],50)
        self.assertEqual(scores[1]["suggestion_rougeL"],0)
        interval = paired_bootstrap(scores,scores,repetitions=100)
        self.assertEqual(interval["ci95"],[0.,0.])
        altered=deepcopy(scores); altered[0]["utterance_id"]="wrong"
        with self.assertRaises(ValueError):paired_bootstrap(scores,altered,repetitions=100)

    def test_old_audio_split_is_rejected_before_loading_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"data/generated") as temporary:
            folder=Path(temporary)
            (folder/"predictions.jsonl").write_text('{}\n',encoding="utf-8")
            metrics={"prediction_sha256":digest(folder/"predictions.jsonl"),"purpose":"scientific_training",
                "training_metadata":{"completed_training":{"status":"completed"},
                    "provenance":{"files":{"protocol":{"sha256":"old-split"}}}}}
            (folder/"metrics.json").write_text(json.dumps(metrics),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"different split"):
                load_condition([],{"cascade_protocol_sha256":"new-split"},folder)

    def test_comparison_rejects_different_generation_or_audio_conditions(self):
        keys=("task","split","rows","full_split_rows","dataset_split_sha256","prompt_sha256","tokenizer_sha256",
              "base_weight_sha256","base_revision","protocol_sha256","ordered_rows_sha256","gpu","packages","input_condition")
        identity={key:'fixture-only' for key in keys};identity['adapter_sha256']=None
        before=({'identity':identity},[{'utterance_id':'fixture','reference_feedback':{}}])
        after=deepcopy(before);after[0]['identity']['adapter_sha256']='trained-fixture'
        verify_pair(before,after)
        for key in ('protocol_sha256','input_condition','base_weight_sha256'):
            changed=deepcopy(after);changed[0]['identity'][key]='different'
            with self.assertRaisesRegex(ValueError,'mismatch'):verify_pair(before,changed)

    @unittest.skipUnless((ROOT/"data/processed/feedback-cascade-en-v1/manifest.json").is_file(),"local published corpus integration")
    def test_real_common_split_holds_out_all_900_from_both_models(self):
        config=json.loads(CONFIG.read_text(encoding="utf-8"))
        protocol,split,_=check_common_split(config)
        self.assertEqual(protocol["joint_counts"],{"train":2249,"validation":450,"test":900})
        test=set(split["cascade_test_ids"])
        self.assertFalse(test & set(split["source_train_ids"]))
        self.assertFalse(test & set(split["folds"][0]["train_ids"]))


if __name__ == "__main__":unittest.main()
