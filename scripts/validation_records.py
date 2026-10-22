"""Strict physical JSONL records and Unicode-preserving transcription scoring."""
import json
from pathlib import Path
import re


def read_json_lines(path):
    # JSON permits U+0085/U+2028/U+2029 in strings; only LF delimits records.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def word_counts(reference, hypothesis):
    # Preserve non-English hallucinated words as words instead of dropping them.
    def normalize(text):
        return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold().replace("’", "'"))
    left, right = normalize(reference), normalize(hypothesis)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return {"word_errors": previous[-1], "reference_words": len(left), "exact_sentence": left == right}
