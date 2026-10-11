#!/usr/bin/env python3
"""One-command setup and training runner for Sora's from-scratch RNN."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_bpe import train_bpe_lines

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/samples/train.txt")
    ap.add_argument("--validation-corpus", type=Path, default=ROOT / "data/samples/validation.txt")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/sora-checkpoint.npz")
    ap.add_argument("--tokenizer-path", type=Path, default=ROOT / "artifacts/sora-tokenizer.json")
    ap.add_argument("--token-cache-dir", type=Path, default=ROOT / "artifacts/cache")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="Sora learns")
    ap.add_argument("--generate-tokens", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--raw-bytes", action="store_true", help="Skip BPE and train directly on byte IDs")
    ap.add_argument("--resume", action="store_true", help="Resume checkpoint and preserve its tokenizer")
    args = ap.parse_args()

    for label, path in (("training corpus", args.corpus), ("validation corpus", args.validation_corpus)):
        if not path.is_file():
            ap.error(f"{label} not found: {path}")
    if args.steps < 1 or args.hidden_size < 1 or args.seq_len < 2 or args.vocab_size < 256:
        ap.error("steps/hidden-size must be positive, seq-len at least 2, and vocab-size at least 256")
    if args.eval_every < 1 or args.eval_batches < 1 or args.generate_tokens < 0 or args.learning_rate <= 0:
        ap.error("invalid evaluation interval, generation length, or learning rate")

    command = [sys.executable, str(ROOT / "python/train.py"), str(args.corpus.resolve()),
               "--validation-corpus", str(args.validation_corpus.resolve()),
               "--steps", str(args.steps), "--hidden-size", str(args.hidden_size),
               "--seq-len", str(args.seq_len), "--learning-rate", str(args.learning_rate),
               "--checkpoint", str(args.checkpoint.resolve()), "--eval-every", str(args.eval_every),
               "--eval-batches", str(args.eval_batches), "--seed", str(args.seed),
               "--prompt", args.prompt, "--generate-tokens", str(args.generate_tokens),
               "--top-k", str(args.top_k), "--token-cache-dir", str(args.token_cache_dir.resolve())]

    if args.raw_bytes:
        print("Using raw UTF-8 byte IDs; no BPE vocabulary will be trained.", flush=True)
    else:
        tokenizer_path = args.tokenizer_path.resolve()
        if args.resume:
            if not tokenizer_path.is_file():
                ap.error(f"Cannot resume BPE run: tokenizer not found: {tokenizer_path}")
            print(f"Reusing tokenizer for checkpoint: {tokenizer_path}", flush=True)
        else:
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            with args.corpus.open("r", encoding="utf-8", newline="") as source:
                tokenizer_data = train_bpe_lines(source, args.vocab_size)
            tokenizer_path.write_text(json.dumps(tokenizer_data, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
            print(f"Trained BPE vocabulary ({tokenizer_data['vocab_size']} IDs): {tokenizer_path}", flush=True)
        command.extend(["--tokenizer", str(tokenizer_path)])

    if args.resume:
        command.append("--resume")
    print("Starting model training from random initialization or the specified checkpoint.", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
