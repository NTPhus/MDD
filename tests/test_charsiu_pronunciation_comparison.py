"""Decoder, safety and evidence contracts for the registered Charsiu comparison."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_charsiu_pronunciation as comparison
from scripts import setup_charsiu_phoneme as setup
from scripts.experiment_pronunciation_calibration import save_evidence, load_evidence, _cache_path


VOCAB = {0: "[SIL]", 1: "NG", 2: "F", 13: "AA", 21: "ZH", 30: "AO", 40: "[UNK]", 41: "[PAD]"}


class DecoderTests(unittest.TestCase):
    def test_ctc_blank_separates_repetitions(self):
        labels, phones = comparison.decode_frames([2, 2, 41, 2], VOCAB, ctc=True)
        self.assertEqual(labels, ["F", "F"])
        self.assertEqual(phones, ["F", "F"])

    def test_fc_pad_is_preserved_as_unknown_not_deleted_like_ctc(self):
        labels, phones = comparison.decode_frames([2, 2, 41, 41, 2], VOCAB, ctc=False)
        self.assertEqual(labels, ["F", "[PAD]", "F"])
        self.assertEqual(phones, ["F", "[UNK]", "F"])

    def test_fc_same_phone_on_both_sides_of_silence_remains_two_phones(self):
        labels, phones = comparison.decode_frames([2, 2, 0, 0, 2, 2], VOCAB, ctc=False)
        self.assertEqual(phones, ["F", "F"])

    def test_ctc_same_phone_across_silence_remains_two_phones(self):
        self.assertEqual(comparison.decode_frames([2, 0, 2], VOCAB, ctc=True)[1], ["F", "F"])

    def test_both_full39_contrasts_retained_and_unknown_not_guessed(self):
        for ctc in (True, False):
            self.assertEqual(comparison.decode_frames([13, 30, 21, 40, 99], VOCAB, ctc=ctc)[1],
                             ["AA", "AO", "ZH", "[UNK]", "[UNK]"])

    def test_both_heads_registered_with_correct_objective_and_resolution(self):
        self.assertEqual(set(setup.MODELS), {"charsiu-ctc", "charsiu-fc10ms"})
        self.assertEqual(setup.MODELS["charsiu-ctc"]["stride_samples"], 320)
        self.assertEqual(setup.MODELS["charsiu-fc10ms"]["stride_samples"], 160)
        self.assertEqual(len(setup.MODELS["charsiu-fc10ms"]["weight_sha256"]), 64)
        self.assertEqual(len(setup.MODELS["charsiu-fc10ms"]["config_sha256"]), 64)


class ResourceAndEvidenceTests(unittest.TestCase):
    def test_physical_gpu_pressure_blocks_new_model_despite_wddm_budget(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda _:(3*2**30,4*2**30)))
        with patch.dict("sys.modules", {"torch": fake_torch}), patch.object(comparison, "physical_cuda_memory", return_value={"free_mib": 500, "total_mib": 4096}):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                comparison.require_free_gpu()

    def test_no_cuda_does_not_silently_change_registered_device(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                comparison.require_free_gpu()

    def test_posterior_npz_matches_legacy_reader_without_pickle(self):
        evidence = {"log_probs": np.log(np.array([[.25,.75],[.8,.2]],dtype=np.float32)),
                    "id_to_phone": {0:"[BLANK]",1:"F"}, "blank_id":0,"duration_sec":1.,
                    "source_audio_sha256":"a"*64,"checkpoint_sha256":"b"*64,"vocabulary_sha256":"c"*64,
                    "phonemes":[{"phone":"F"}],"sample_rate":16000,"sample_count":16000,"case_id":"NJS-example"}
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            path=_cache_path(folder,"NJS/example")
            save_evidence(path,evidence)
            loaded=load_evidence(path)
            np.testing.assert_array_equal(loaded["log_probs"],evidence["log_probs"])
            self.assertEqual(loaded["id_to_phone"],evidence["id_to_phone"])
            self.assertEqual(loaded["source_audio_sha256"],"a"*64)
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(data["log_probs"].dtype,np.float32)
                self.assertEqual(data["metadata"].dtype.kind,"U")

    def test_unsafe_pickle_globals_are_rejected_before_state_deserialization(self):
        fake_load = lambda *args,**kwargs: self.fail("Unsafe checkpoint must never be deserialized")
        fake_torch=SimpleNamespace(serialization=SimpleNamespace(get_unsafe_globals_in_checkpoint=lambda _:["malicious.global"]),load=fake_load)
        with patch.dict("sys.modules",{"torch":fake_torch}):
            with self.assertRaisesRegex(ValueError,"Unapproved checkpoint globals"):
                setup.inspect_states(Path("unused"))


if __name__ == "__main__":
    unittest.main()
