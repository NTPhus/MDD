"""Projecting only supervised positions must preserve causal loss and gradients."""
from copy import deepcopy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from scripts.train_feedback_model import collate_pairs, chunked_causal_loss


class AssistantProjectionTests(unittest.TestCase):
    def test_chunked_projection_preserves_loss_gradients_and_accumulation_denominator(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            attention_dropout=0., tie_word_embeddings=True, use_cache=False)
        rows = [{"input_ids": [1,2,3,4,5,6,7], "attention_mask": [1]*7,
                 "labels": [-100,-100,-100,4,5,6,7]},
                {"input_ids": [1,2,3,4,5], "attention_mask": [1]*5, "labels": [-100,-100,3,4,5]}]
        batch = collate_pairs(rows, 0)
        for denominator in (None, torch.tensor(17)):
            original = Qwen3ForCausalLM(config).eval()
            efficient = deepcopy(original)
            extra = {} if denominator is None else {"num_items_in_batch": denominator}
            expected = original(**batch, **extra).loss
            actual = chunked_causal_loss(efficient, batch, 2, denominator)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            expected.backward(); actual.backward()
            for a,b in zip(original.parameters(), efficient.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_padding_mixed_prompt_lengths_and_accumulation_denominator_preserve_loss_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
                            attention_dropout=0., tie_word_embeddings=True, use_cache=False)
        rows = [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], "attention_mask": [1] * 8,
             "labels": [-100, -100, -100, -100, -100, 6, 7, 8]},
            {"input_ids": [1, 2, 3, 4, 5, 6], "attention_mask": [1] * 6,
             "labels": [-100, -100, -100, 4, 5, 6]},
        ]
        for denominator in (None, torch.tensor(17)):
            with self.subTest(denominator=denominator):
                original = Qwen3ForCausalLM(config).eval()
                efficient = deepcopy(original)
                full = collate_pairs(rows, 0)
                reduced = collate_pairs(rows, 0, assistant_only_logits=True)
                self.assertTrue(torch.equal(full["input_ids"], reduced["input_ids"]))
                self.assertTrue(torch.equal(full["attention_mask"], reduced["attention_mask"]))
                self.assertTrue(torch.equal(full["labels"], reduced["labels"]))
                self.assertLess(len(reduced["logits_to_keep"]), full["input_ids"].shape[1])
                extra = {} if denominator is None else {"num_items_in_batch": denominator}
                original_result = original(**full, **extra)
                efficient_result = efficient(**reduced, **extra)
                torch.testing.assert_close(original_result.loss, efficient_result.loss, atol=1e-6, rtol=1e-6)
                original_result.loss.backward()
                efficient_result.loss.backward()
                for (name_a, parameter_a), (name_b, parameter_b) in zip(original.named_parameters(), efficient.named_parameters()):
                    self.assertEqual(name_a, name_b)
                    torch.testing.assert_close(parameter_a.grad, parameter_b.grad, atol=2e-6, rtol=2e-5)

    def test_no_target_batch_is_rejected(self):
        row = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100] * 3}
        with self.assertRaisesRegex(ValueError, "supervised"):
            collate_pairs([row], 0, assistant_only_logits=True)


if __name__ == "__main__":
    unittest.main()
