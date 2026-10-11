#!/usr/bin/env python3
"""Evaluate a saved causal Transformer on a separate held-out UTF-8 test corpus."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from sora_transformer import SoraTransformer, sha256_file
from tokenize_corpus import tokenize_file
from tokenizer import ByteBPETokenizer
import numpy as np


def evaluate_checkpoint(checkpoint: Path, corpus: Path, tokenizer_path=None):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not corpus.is_file():
        raise FileNotFoundError(f"Test corpus not found: {corpus}")

    model = SoraTransformer.load(checkpoint)
    metadata = model.metadata or {}
    tokenizer = ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else None
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path else None
    if tokenizer_hash != metadata.get("tokenizer_sha256"):
        raise ValueError(
            "Tokenizer fingerprint does not match the checkpoint. "
            "Pass the exact tokenizer used for training; omit --tokenizer only for byte-level checkpoints."
        )
    vocab_size = tokenizer.vocab_size if tokenizer else 256
    if vocab_size != model.vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({vocab_size}) does not match model ({model.vocab_size})")

    cache_dir = checkpoint.parent / "evaluation-cache"
    cache_path = cache_dir / (corpus.name + ".tokens.bin")
    _, cache_manifest = tokenize_file(corpus, cache_path, tokenizer_path)
    token_count = int(cache_manifest["token_count"])
    dtype = np.dtype(cache_manifest["dtype"])
    tokens = np.memmap(cache_path, dtype=dtype, mode="r", shape=(token_count,))
    if token_count < 2:
        raise ValueError("Test corpus must produce at least two tokens")

    # Non-overlapping target spans cover every next-token target once.
    total_nll = 0.0
    total_targets = 0
    window = model.max_seq_len + 1
    for start in range(0, token_count - 1, model.max_seq_len):
        sequence = np.asarray(tokens[start:min(start + window, token_count)], dtype=np.int64)
        if len(sequence) < 2:
            continue
        loss = model.sequence_loss(sequence)
        targets = len(sequence) - 1
        total_nll += loss * targets
        total_targets += targets

    mean_loss = total_nll / total_targets
    result = {
        "format": "lfm-test-evaluation-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_step": model.step,
        "test_corpus": str(corpus),
        "test_corpus_sha256": sha256_file(corpus),
        "tokenizer_sha256": tokenizer_hash,
        "test_tokens": token_count,
        "evaluated_targets": total_targets,
        "sequence_length": model.max_seq_len,
        "test_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "warning": "Keep this test corpus held out: do not train on it or use it to select checkpoints or tune hyperparameters."
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("test_corpus", type=Path, help="held-out test text never used for training or validation")
    parser.add_argument("--tokenizer", type=Path, help="exact BPE tokenizer used for this checkpoint")
    parser.add_argument("--output", type=Path, help="optional path to write the JSON report")
    args = parser.parse_args()
    try:
        report = evaluate_checkpoint(args.checkpoint, args.test_corpus, args.tokenizer)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
