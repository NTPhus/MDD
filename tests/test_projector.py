"""Mechanism tests: real tiny Qwen gradients, causal masks, checkpoint and decoding."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model

from scripts.projector_model import AudioProjector, ProjectedQwen
from scripts.projector_data import prompt_parts, target_ids, prepare, read_json, CONFIG, load
from scripts.report_projector import read_evaluation, verify_pair
from scripts.projector_data import save
from unittest.mock import patch
from scripts.prepare_feedback_data import digest


class SimpleTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages) + "ASSISTANT:"

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


def tiny_model(lora=False):
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
                        attention_dropout=0., tie_word_embeddings=True, use_cache=False,
                        eos_token_id=40, pad_token_id=0)
    llm = Qwen3ForCausalLM(config).requires_grad_(False)
    if lora:
        llm = get_peft_model(llm, LoraConfig(r=2, lora_alpha=4, lora_dropout=0., target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    return ProjectedQwen(llm, AudioProjector(8, 16, 12), chunk_size=2)


def example(frames=5):
    return {"features": torch.randn(1, frames, 8), "prefix_ids": torch.tensor([[1, 2]]),
            "suffix_ids": torch.tensor([[3, 4, 5]]), "target_ids": torch.tensor([[6, 7, 8, 40]])}


class ProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_two_linear_gelu_and_parameter_count(self):
        projector = AudioProjector()
        self.assertIsInstance(projector.layers[1], torch.nn.GELU)
        self.assertEqual(sum(p.numel() for p in projector.parameters()), 1837056)
        self.assertEqual(projector(torch.randn(1, 7, 768)).shape, (1, 7, 1024))

    def test_chunked_causal_loss_and_gradients_equal_full_loss(self):
        for frames in (2, 7):
            efficient = tiny_model().eval()
            reference = deepcopy(efficient)
            batch = example(frames)
            inputs, labels, prompt_length = reference.embeddings(**batch)
            self.assertTrue(labels[:, :prompt_length].eq(-100).all())
            self.assertTrue(torch.equal(labels[:, prompt_length:], batch["target_ids"]))
            expected = reference.llm(inputs_embeds=inputs, attention_mask=torch.ones_like(labels), labels=labels, use_cache=False).loss
            actual = efficient(**batch)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            expected.backward()
            actual.backward()
            for left, right in zip(reference.projector.parameters(), efficient.projector.parameters()):
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
            self.assertTrue(all(p.grad is None for p in efficient.llm.parameters()))
            self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in efficient.projector.parameters()))

    def test_alignment_gradients_survive_decoder_checkpointing(self):
        model = tiny_model().train()
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        original = [p.detach().clone() for p in model.projector.parameters()]
        optimizer = torch.optim.AdamW(model.projector.parameters(), lr=1e-3)
        model(**example()).backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(original, model.projector.parameters())))
        self.assertTrue(all(p.grad is None for p in model.llm.parameters()))

    def test_feedback_updates_projector_and_lora_not_backbone(self):
        model = tiny_model(lora=True).train()
        original = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        model(**example()).backward()
        optimizer.step()
        changed = [name for name, p in model.named_parameters() if not torch.equal(original[name], p)]
        self.assertTrue(any(name.startswith("projector.") for name in changed))
        self.assertTrue(any("lora_B" in name for name in changed))
        self.assertTrue(all(name.startswith("projector.") or "lora_" in name for name in changed))

    def test_checkpoint_roundtrip_stores_only_projector_and_adapter(self):
        for lora in (False, True):
            model = tiny_model(lora=lora).eval()
            batch = example()
            model(**batch).backward()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            optimizer.step()
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / "weights"
                model.save_weights(folder)
                self.assertFalse((folder / "model.safetensors").exists())
                restored = tiny_model(lora=lora).eval()
                restored.load_weights(folder)
                torch.testing.assert_close(restored(**batch), model(**batch))
                self.assertEqual((folder / "adapter/adapter_model.safetensors").exists(), lora)

    def test_generation_returns_only_continuation_matching_manual_decode(self):
        for lora in (False, True):
            model = tiny_model(lora=lora).eval()
            batch = example()
            del batch["target_ids"]
            with torch.no_grad():
                embeds, _, _ = model.embeddings(**batch)
                expected = []
                for _ in range(3):
                    token = model.llm(inputs_embeds=embeds, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                    expected.append(token)
                    embeds = torch.cat([embeds, model.llm.get_input_embeddings()(token)], dim=1)
                actual = model.generate_response(**batch, eos_token_id=None, pad_token_id=0,
                                                max_new_tokens=3, do_sample=False, num_beams=1)
            self.assertEqual(actual.shape, (1, 3))
            self.assertTrue(torch.equal(actual, torch.cat(expected, dim=1)))

    def test_prompts_never_include_oracle_annotations_or_target_feedback(self):
        tokenizer = SimpleTokenizer()
        row = {"reference_text": "CANONICAL UNIQUE SENTENCE <audio_latents>", "annotation_info": {"secret_phone": ["GOLD"]},
               "feedback": {"secret_word": [{"issue": "SECRET_TARGET", "suggestion": "SECRET_ADVICE"}]}}
        for stage in ("alignment", "feedback"):
            left, right = prompt_parts(tokenizer, row, stage)
            text = "".join(map(chr, left + right))
            self.assertNotIn("SECRET", text)
            self.assertNotIn("secret_phone", text)
            self.assertEqual("CANONICAL UNIQUE SENTENCE" in text, stage == "feedback")
            self.assertGreater(len(target_ids(tokenizer, row, stage)), 1)

    def test_prepared_roles_are_identical_to_cascade_and_speaker_disjoint(self):
        config = read_json(CONFIG)
        if not (Path(config["cascade_config"])).exists():
            self.skipTest("Local frozen data unavailable")
        parts, manifest = load(config)
        self.assertEqual({role: len(rows) for role, rows in parts.items()}, {"train": 2249, "validation": 450, "test": 900})
        speakers = {role: {r["speaker_id"] for r in rows} for role, rows in parts.items()}
        self.assertFalse(speakers["train"] & speakers["test"])
        self.assertFalse(speakers["validation"] & speakers["test"])
        self.assertTrue(all("annotation_info" not in row for rows in parts.values() for row in rows))

    def test_report_rejects_technical_and_incomplete_scores(self):
        for purpose, count in (("technical_smoke", 900), ("held_out_benchmark", 1)):
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                content = "".join(json.dumps({"utterance_id": str(i)}) + "\n" for i in range(count))
                (folder / "predictions.jsonl").write_text(content)
                meta = {"status": "completed", "identity": {"purpose": purpose, "split": "test", "full_split_rows": 900},
                        "predictions_sha256": digest(folder / "predictions.jsonl")}
                (folder / "metadata.json").write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "full verified"):
                    read_evaluation(folder)

    def test_windows_status_sharing_violation_is_retried(self):
        with patch('scripts.projector_data.atomic_save', side_effect=[PermissionError('busy'), None]) as writer:
            with patch('scripts.projector_data.time.sleep'):
                save(Path('status.json'), {'status':'running'})
            self.assertEqual(writer.call_count, 2)

    def test_pair_rejects_mismatched_audio_encoder(self):
        fields = ('task', 'split', 'rows', 'full_split_rows', 'dataset_split_sha256', 'protocol_sha256', 'ordered_rows_sha256',
                  'base_weight_sha256', 'tokenizer_sha256', 'input_condition', 'prompt_sha256', 'generation',
                  'precision_profile', 'seed', 'gpu', 'packages', 'model_code_sha256', 'evaluation_code_sha256', 'context_length', 'metrics_code_sha256')
        left = {'identity': {key:'same' for key in fields} | {'case':'P0'}}
        right = deepcopy(left)
        right['identity']['case'] = 'P1'
        right['identity']['input_condition'] = 'different_encoder'
        with self.assertRaisesRegex(ValueError, 'input_condition'):
            verify_pair((left, []), (right, []))


if __name__ == "__main__":
    unittest.main()
