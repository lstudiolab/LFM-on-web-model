#!/usr/bin/env python3
"""Train a deterministic byte-level BPE tokenizer without external libraries."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

BASE_VOCAB_SIZE = 256
BPE_FORMAT = "sora-byte-bpe-v1"
# Newline is its own boundary. Horizontal whitespace is grouped separately so
# streaming line-by-line tokenization is identical to full-text tokenization.
PIECE_PATTERN = re.compile(r"\r\n|\r|\n|[^\S\r\n]+|[\w]+|[^\w\s]+", re.UNICODE)


def split_pieces(text: str) -> list[str]:
    """Split into stable word/whitespace/punctuation pieces without crossing line breaks."""
    return PIECE_PATTERN.findall(text)


def _train_from_frequencies(frequencies: Counter, vocab_size: int) -> dict:
    if vocab_size < BASE_VOCAB_SIZE:
        raise ValueError("vocab_size must be at least 256")
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
        "notes": "Byte-level BPE trained on word/whitespace/punctuation pieces; train only on the training split."
    }


def _piece_frequencies(pieces) -> Counter:
    frequencies = Counter()
    for piece in pieces:
        raw = tuple(piece.encode("utf-8"))
        if raw:
            frequencies[raw] += 1
    return frequencies


def train_bpe(text: str, vocab_size: int) -> dict:
    """Train from an in-memory text string; useful for tests and small inputs."""
    if vocab_size < BASE_VOCAB_SIZE:
        raise ValueError("vocab_size must be at least 256")
    return _train_from_frequencies(_piece_frequencies(split_pieces(text)), vocab_size)


def train_bpe_lines(lines, vocab_size: int, max_lines: int = 200_000, seed: int = 7) -> dict:
    """Train on a deterministic reservoir sample of lines to bound tokenizer cost.

    Set max_lines=0 to use the entire iterator. The sample is selected across the
    input stream, so a long multilingual corpus is not restricted to its first files.
    """
    if vocab_size < BASE_VOCAB_SIZE:
        raise ValueError("vocab_size must be at least 256")
    if max_lines < 0:
        raise ValueError("max_lines cannot be negative")
    if max_lines == 0:
        selected_lines = lines
    else:
        rng = random.Random(seed)
        reservoir = []
        for index, line in enumerate(lines):
            if index < max_lines:
                reservoir.append(line)
            else:
                slot = rng.randrange(index + 1)
                if slot < max_lines:
                    reservoir[slot] = line
        selected_lines = reservoir
    frequencies = Counter()
    for line in selected_lines:
        frequencies.update(_piece_frequencies(split_pieces(line)))
    return _train_from_frequencies(frequencies, vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="training text only, never validation/test text")
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-lines", type=int, default=200_000, help="deterministic reservoir sample size; 0 uses every line")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.corpus.is_file():
        parser.error(f"Corpus not found: {args.corpus}")
    if args.max_lines < 0:
        parser.error("--max-lines cannot be negative")
    try:
        with args.corpus.open("r", encoding="utf-8", newline="") as source:
            model = train_bpe_lines(source, args.vocab_size, args.max_lines, args.seed)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"Saved {model['vocab_size']} token IDs and {len(model['merges'])} merges to {args.output}")


if __name__ == "__main__":
    main()
