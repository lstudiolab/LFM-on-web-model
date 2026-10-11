#!/usr/bin/env python3
"""Train a small byte-level BPE merge table using Python's standard library."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def train_bpe(text: str, vocab_size: int) -> dict:
    """Train BPE merges over a concatenated UTF-8 byte stream.

    This is an intentionally simple baseline. It does not preserve word
    boundaries; a later version should train over word frequencies and add
    an explicit boundary marker.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")

    sequence = list(text.encode("utf-8"))
    merges: list[list[int]] = []
    next_id = 256

    for _ in range(vocab_size - 256):
        if len(sequence) < 2:
            break

        pairs = Counter(zip(sequence, sequence[1:]))
        if not pairs:
            break

        pair, count = pairs.most_common(1)[0]
        if count < 2:
            break

        merged_id = next_id
        next_id += 1
        merges.append([pair[0], pair[1], merged_id])

        merged: list[int] = []
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                merged.append(merged_id)
                i += 2
            else:
                merged.append(sequence[i])
                i += 1
        sequence = merged

    return {
        "format": "sora-byte-bpe-v1",
        "base_vocab_size": 256,
        "vocab_size": 256 + len(merges),
        "merges": merges,
        "notes": "Educational baseline over a concatenated UTF-8 byte stream."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    model = train_bpe(text, args.vocab_size)
    args.output.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"Saved {model['vocab_size']} token IDs and {len(model['merges'])} merges to {args.output}")


if __name__ == "__main__":
    main()
