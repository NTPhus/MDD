"""Conversation context eviction, with a tokenizer double and no model loads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.local_chat_worker import bounded_inputs


class Tokenizer:
    """Count words plus one token per message, recording each attempted prompt."""
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(deepcopy(messages))
        return json.dumps(messages)

    def __call__(self, text, **kwargs):
        messages = json.loads(text)
        count = sum(len(message["content"].split()) + 1 for message in messages)

        class Inputs(dict):
            def to(self, device):
                return self

        return Inputs(input_ids=SimpleNamespace(shape=(1, count)))


def message(role, content):
    return {"role": role, "content": content}


class ChatContextTests(unittest.TestCase):
    def setUp(self):
        self.system = message("system", "Be friendly.")
        self.current = message("user", "Because I enjoy local food.")
        self.old = [message("user", "Japan " * 30), message("assistant", "Why Japan?")]

    def bounded(self, messages, context, max_tokens=4):
        original = deepcopy(messages)
        tokenizer = Tokenizer()
        result = bounded_inputs(tokenizer, messages, context, max_tokens)
        self.assertEqual(messages, original, "Trimming must not mutate caller history")
        for attempt in tokenizer.prompts:
            self.assertEqual(attempt[0], self.system)
            self.assertEqual(attempt[-1], self.current)
        return tokenizer, result

    def test_short_history_is_kept_without_drops(self):
        messages = [self.system, *self.old, self.current]
        tokenizer, (_, dropped) = self.bounded(messages, 100)
        self.assertEqual(dropped, 0)
        self.assertEqual(tokenizer.prompts, [messages])

    def test_free_chat_removes_whole_old_pair_and_preserves_current_user(self):
        tokenizer, (_, dropped) = self.bounded([self.system, *self.old, self.current], 14)
        self.assertEqual(dropped, 2)
        self.assertEqual(tokenizer.prompts[-1], [self.system, self.current])

    def test_topic_opener_is_removed_independently_before_an_old_pair(self):
        opener = message("assistant", "Where would you like to travel?")
        tokenizer, (_, dropped) = self.bounded([self.system, opener, *self.old, self.current], 14)
        self.assertEqual(dropped, 3)
        self.assertEqual(tokenizer.prompts[1], [self.system, *self.old, self.current])
        self.assertEqual(tokenizer.prompts[-1], [self.system, self.current])

    def test_only_the_authored_opener_is_removed_when_that_is_enough(self):
        opener = message("assistant", "A long opening " * 20)
        recent = [message("user", "Japan."), message("assistant", "Why Japan?")]
        tokenizer, (_, dropped) = self.bounded([self.system, opener, *recent, self.current], 19)
        self.assertEqual(dropped, 1)
        self.assertEqual(tokenizer.prompts[-1], [self.system, *recent, self.current])
        self.assertEqual([entry["role"] for entry in tokenizer.prompts[-1]],
                         ["system", "user", "assistant", "user"])

    def test_multiple_old_pairs_are_removed_in_order_without_losing_recent_context(self):
        recent = [message("user", "Japan."), message("assistant", "Why Japan?")]
        opener = message("assistant", "Let's plan a trip.")
        older = [message("user", "old " * 30), message("assistant", "Earlier reply.")]
        tokenizer, (_, dropped) = self.bounded([self.system, opener, *older, *self.old, *recent, self.current], 19)
        self.assertEqual(dropped, 5)
        self.assertEqual(tokenizer.prompts[-1], [self.system, *recent, self.current])

    def test_oversized_current_utterance_raises_without_deleting_it_or_system(self):
        tokenizer = Tokenizer()
        messages = [self.system, message("assistant", "Topic opening."), *self.old, self.current]
        original = deepcopy(messages)
        with self.assertRaisesRegex(ValueError, "chat_context_too_long"):
            bounded_inputs(tokenizer, messages, 4, 4)
        self.assertEqual(messages, original)
        self.assertEqual(tokenizer.prompts[-1], [self.system, self.current])
        self.assertTrue(all(attempt[-1] == self.current for attempt in tokenizer.prompts))


if __name__ == "__main__":
    unittest.main()
