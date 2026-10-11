#!/usr/bin/env python3
"""Held-out next-token evaluation for Sora's PyTorch checkpoints."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from sora_torch_transformer import SoraTorchTransformer, load_checkpoint
from sora_transformer import sha256_file
from tokenize_corpus import tokenize_file
from tokenizer import ByteBPETokenizer
from train_torch_transformer import choose_device, open_token_array


def evaluate_torch_checkpoint(checkpoint: Path, corpus: Path, tokenizer_path=None, device_name="auto"):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not Path(corpus).is_file():
        raise FileNotFoundError(f"Test corpus not found: {corpus}")
    device = choose_device(device_name)
    payload = load_checkpoint(checkpoint, map_location=device)
    config = payload["model_config"]
    model = SoraTorchTransformer(**config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path else None
    metadata = payload.get("metadata", {})
    if tokenizer_hash != metadata.get("tokenizer_sha256"):
        raise ValueError("Tokenizer fingerprint does not match this checkpoint; pass the exact tokenizer used for training.")
    tokenizer = ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else None
    vocab_size = tokenizer.vocab_size if tokenizer else 256
    if vocab_size != model.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model")

    corpus = Path(corpus)
    cache_key = sha256_file(corpus)[:16]
    cache_path = Path(checkpoint).parent / "evaluation-cache" / f"{corpus.stem}-{cache_key}.tokens.bin"
    _, cache_manifest = tokenize_file(corpus, cache_path, tokenizer_path)
    tokens = open_token_array(cache_path, cache_manifest)
    if len(tokens) < 2:
        raise ValueError("Test corpus must produce at least two tokens")

    total_nll, total_targets = 0.0, 0
    stride = model.max_seq_len
    with torch.inference_mode():
        for start in range(0, len(tokens) - 1, stride):
            sequence = np.asarray(tokens[start:min(start + stride + 1, len(tokens))], dtype=np.int64)
            if len(sequence) < 2:
                continue
            inputs = torch.as_tensor(sequence[:-1][None, :], dtype=torch.long, device=device)
            targets = torch.as_tensor(sequence[1:][None, :], dtype=torch.long, device=device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            count = int(targets.numel())
            total_nll += float(loss.item()) * count
            total_targets += count
    mean_loss = total_nll / total_targets
    return {
        "format": "lfm-torch-test-evaluation-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(metadata.get("global_step", 0)),
        "test_corpus": str(corpus),
        "test_corpus_sha256": sha256_file(corpus),
        "tokenizer_sha256": tokenizer_hash,
        "test_tokens": int(len(tokens)),
        "evaluated_targets": total_targets,
        "sequence_length": model.max_seq_len,
        "test_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "warning": "Perplexity is only a next-token metric. Keep the test corpus out of training and tuning; this score alone does not prove fluency or conversational quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("test_corpus", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_torch_checkpoint(args.checkpoint, args.test_corpus, args.tokenizer, args.device)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
