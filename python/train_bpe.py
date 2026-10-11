#!/usr/bin/env python3
"""Train a deterministic byte-level BPE tokenizer without external libraries."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BASE_VOCAB_SIZE = 256
BPE_FORMAT = "sora-byte-bpe-v1"
PIECE_PATTERN = re.compile(r"\s+|[\w]+|[^\w\s]+", re.UNICODE)


def split_pieces(text: str) -> list[str]:
    """Split into stable word/whitespace/punctuation pieces; never merge across pieces."""
    return PIECE_PATTERN.findall(text)


def train_bpe(text: str, vocab_size: int) -> dict:
    """Learn BPE merges from repeated UTF-8 byte sequences.

    Frequencies are accumulated over repeated pre-tokenized pieces rather than
    every occurrence in the full corpus. Merge application is deterministic and
    uses the same piece boundaries as ByteBPETokenizer.
    """
    if vocab_size < BASE_VOCAB_SIZE:
        raise ValueError("vocab_size must be at least 256")

    frequencies = Counter(tuple(piece.encode("utf-8")) for piece in split_pieces(text))
    frequencies = Counter({tokens: count for tokens, count in frequencies.items() if tokens})
    merges: list[list[int]] = []
    next_id = BASE_VOCAB_SIZE

    for _ in range(vocab_size - BASE_VOCAB_SIZE):
        pair_counts = Counter()
        for tokens, frequency in frequencies.items():
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += frequency
        if not pair_counts:
            break

        pair, count = pair_counts.most_common(1)[0]
        if count < 2:
            break

        left, right = pair
        merged_id = next_id
        next_id += 1
        merges.append([left, right, merged_id])

        updated = Counter()
        for tokens, frequency in frequencies.items():
            result = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
                    result.append(merged_id)
                    index += 2
                else:
                    result.append(tokens[index])
                    index += 1
            updated[tuple(result)] += frequency
        frequencies = updated

    return {
        "format": BPE_FORMAT,
        "base_vocab_size": BASE_VOCAB_SIZE,
        "vocab_size": BASE_VOCAB_SIZE + len(merges),
        "merges": merges,
        "notes": "Byte-level BPE learned on word/whitespace/punctuation pieces; train only on the training split."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="training text only, never validation/test text")
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()
    if not args.corpus.is_file():
        parser.error(f"Corpus not found: {args.corpus}")
    text = args.corpus.read_text(encoding="utf-8")
    if not text.strip():
        parser.error("Training corpus is empty")
    model = train_bpe(text, args.vocab_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"Saved {model['vocab_size']} token IDs and {len(model['merges'])} merges to {args.output}")


if __name__ == "__main__":
    main()
