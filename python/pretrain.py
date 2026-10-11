#!/usr/bin/env python3
"""Prepare a corpus, train tokenizer, preflight artifacts, and optionally start pretraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from prepare_corpus import prepare
from sora_model import sha256_file
from tokenize_corpus import tokenize_file
from tokenizer import ByteBPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True,
                        help="UTF-8 corpus files/folders (train corpus must be legally usable)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--min-document-chars", type=int, default=20)
    parser.add_argument("--license-note", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true",
                        help="prepare and verify all artifacts without training")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.steps < 1 or args.hidden_size < 1 or args.seq_len < 2:
        parser.error("steps and hidden size must be positive; seq-len must be at least 2")
    if args.vocab_size < 256 or args.vocab_size > 1_000_000:
        parser.error("vocab-size must be between 256 and 1000000")
    if args.learning_rate <= 0 or args.eval_every < 1:
        parser.error("learning-rate must be positive and eval-every must be positive")

    run_dir = args.run_dir
    corpus_dir = run_dir / "corpus"
    token_dir = run_dir / "tokens"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preparing and deduplicating corpus; splitting on document boundaries.")
    try:
        manifest = prepare(
            args.input, corpus_dir, args.validation_fraction, args.seed,
            args.min_document_chars, args.license_note,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Documents: {manifest['documents_total_unique']:,} unique | "
          f"train={manifest['train_documents']:,} | validation={manifest['validation_documents']:,}")
    print(f"Dataset manifest: {corpus_dir / 'manifest.json'}")

    print("[2/5] Learning BPE only from the training split.")
    train_text = corpus_dir / "train.txt"
    tokenizer_path = run_dir / "tokenizer.json"
    script = Path(__file__).with_name("train_bpe.py")
    result = subprocess.run(
        [sys.executable, str(script), str(train_text), str(tokenizer_path),
         "--vocab-size", str(args.vocab_size)],
        check=False,
    )
    if result.returncode != 0:
        parser.error("BPE tokenizer training failed")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)

    print("[3/5] Tokenizing train and validation data and checking cache manifests.")
    _, train_tokens = tokenize_file(train_text, token_dir / "train.tokens.bin", tokenizer_path)
    _, valid_tokens = tokenize_file(corpus_dir / "validation.txt", token_dir / "validation.tokens.bin", tokenizer_path)
    if train_tokens["tokenizer_sha256"] != valid_tokens["tokenizer_sha256"]:
        parser.error("Train/validation tokenizer fingerprints differ")
    if train_tokens["token_count"] < 4 or valid_tokens["token_count"] < 2:
        parser.error("Prepared corpus produced too few tokens for training and validation")
    if manifest["train_sha256"] == manifest["validation_sha256"]:
        parser.error("Training and validation split hashes are identical")
    print(f"BPE vocabulary: {tokenizer.vocab_size:,} | train tokens: {train_tokens['token_count']:,} | "
          f"validation tokens: {valid_tokens['token_count']:,}")

    print("[4/5] Estimating model size and writing run configuration.")
    # Count parameters algebraically so preflight does not allocate model weights.
    h, v = args.hidden_size, tokenizer.vocab_size
    parameter_count = 2 * v * h + 2 * h * h + h + v
    config = {
        "format": "lfm-pretraining-run-v1",
        "model": "manual-gradient byte/BPE RNN (experimental baseline)",
        "hidden_size": h,
        "vocab_size": v,
        "parameters": parameter_count,
        "approx_parameter_plus_adam_state_bytes_fp32": parameter_count * 12,
        "sequence_length": args.seq_len,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_tokens": train_tokens["token_count"],
        "validation_tokens": valid_tokens["token_count"],
        "training_manifest": str(corpus_dir / "manifest.json"),
        "warning": "This is a small RNN baseline, not a modern or production-scale LLM."
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Parameter count: {parameter_count:,}")
    print(f"Approx. float32 parameters + Adam moments: "
          f"{config['approx_parameter_plus_adam_state_bytes_fp32'] / (1024**2):.2f} MiB")
    print(f"Configuration: {run_dir / 'run_config.json'}")

    if args.prepare_only:
        print("[5/5] Preflight complete. No training was started (--prepare-only).")
        return

    checkpoint = run_dir / "sora-checkpoint.npz"
    train_script = Path(__file__).with_name("train.py")
    command = [
        sys.executable, str(train_script), str(train_text),
        "--validation-corpus", str(corpus_dir / "validation.txt"),
        "--tokenizer", str(tokenizer_path),
        "--checkpoint", str(checkpoint),
        "--token-cache-dir", str(token_dir),
        "--steps", str(args.steps),
        "--hidden-size", str(args.hidden_size),
        "--seq-len", str(args.seq_len),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--eval-every", str(args.eval_every),
    ]
    if args.resume:
        command.append("--resume")
    print("[5/5] Starting training. Interrupt safely and resume with --resume.")
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
