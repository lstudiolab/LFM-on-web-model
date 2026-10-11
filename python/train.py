#!/usr/bin/env python3
"""Train Sora's from-scratch recurrent language model on a UTF-8 corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from sora_model import SoraRNN, sha256_file
from tokenize_corpus import tokenize_file
from tokenizer import ByteBPETokenizer


def open_token_array(path: Path, manifest):
    dtype = np.dtype(manifest["dtype"])
    return np.memmap(path, dtype=dtype, mode="r", shape=(int(manifest["token_count"]),))


def evaluate(model, ids, seq_len, batches=4):
    if len(ids) < 2:
        raise ValueError("Validation corpus must contain at least two tokens")
    window_size = min(seq_len + 1, len(ids))
    max_start = len(ids) - window_size
    starts = np.linspace(0, max_start, num=min(batches, max_start + 1), dtype=int)
    values = []
    for start in starts:
        sequence = np.asarray(ids[start:start + window_size], dtype=np.int64)
        values.append((model.sequence_loss(sequence), len(sequence) - 1))
    total_tokens = sum(count for _, count in values)
    return sum(loss * count for loss, count in values) / total_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path, help="UTF-8 training corpus or prepared train.txt")
    ap.add_argument("--validation-corpus", type=Path, help="separate validation file, preferably split by document")
    ap.add_argument("--tokenizer", type=Path, help="trained BPE tokenizer JSON; omit for raw byte IDs")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--checkpoint", type=Path, default=Path("sora-checkpoint.npz"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="Sora")
    ap.add_argument("--generate-tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--log", type=Path, help="JSONL metrics log")
    ap.add_argument("--token-cache-dir", type=Path, help="default: CHECKPOINT_DIR/tokens")
    args = ap.parse_args()

    if args.steps < 1:
        ap.error("--steps must be positive")
    if args.seq_len < 2:
        ap.error("--seq-len must be at least 2")
    if args.learning_rate <= 0:
        ap.error("--learning-rate must be positive")
    if args.eval_every < 1 or args.eval_batches < 1:
        ap.error("--eval-every and --eval-batches must be positive")
    if args.generate_tokens < 0:
        ap.error("--generate-tokens cannot be negative")
    if not args.corpus.is_file():
        ap.error(f"Training corpus not found: {args.corpus}")
    if args.validation_corpus and not args.validation_corpus.is_file():
        ap.error(f"Validation corpus not found: {args.validation_corpus}")
    if args.tokenizer and not args.tokenizer.is_file():
        ap.error(f"Tokenizer not found: {args.tokenizer}")

    tokenizer = ByteBPETokenizer.load(args.tokenizer) if args.tokenizer else None
    vocab_size = tokenizer.vocab_size if tokenizer else 256
    tokenizer_hash = sha256_file(args.tokenizer) if args.tokenizer else None
    training_corpus_hash = sha256_file(args.corpus)
    validation_corpus_hash = sha256_file(args.validation_corpus) if args.validation_corpus else None
    cache_dir = args.token_cache_dir or (args.checkpoint.parent / "tokens")
    train_cache = cache_dir / f"{args.corpus.stem}.tokens.bin"
    _, train_manifest = tokenize_file(args.corpus, train_cache, args.tokenizer)
    train_all = open_token_array(train_cache, train_manifest)

    if args.validation_corpus:
        valid_cache = cache_dir / f"{args.validation_corpus.stem}.tokens.bin"
        _, valid_manifest = tokenize_file(args.validation_corpus, valid_cache, args.tokenizer)
        train = train_all
        valid = open_token_array(valid_cache, valid_manifest)
    else:
        # Fallback for quick experiments only; prepare_corpus.py performs a safer
        # document-level split and should be used for real pretraining.
        if len(train_all) < 4:
            ap.error("Corpus needs at least four tokens when --validation-corpus is omitted")
        split = min(max(2, int(len(train_all) * 0.9)), len(train_all) - 2)
        train = train_all[:split]
        valid = train_all[split:]

    if len(train) < 2 or len(valid) < 2:
        ap.error("Training and validation data must each contain at least two tokens")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log or args.checkpoint.with_suffix(".metrics.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        log_path.write_text("", encoding="utf-8")

    saved_rng_state = None
    if args.resume:
        if not args.checkpoint.is_file():
            ap.error(f"Cannot resume: checkpoint not found: {args.checkpoint}")
        model = SoraRNN.load(args.checkpoint)
        if model.vocab_size != vocab_size:
            ap.error(f"Checkpoint vocabulary ({model.vocab_size}) does not match tokenizer ({vocab_size})")
        if model.hidden_size != args.hidden_size:
            ap.error(f"Checkpoint hidden size ({model.hidden_size}) does not match --hidden-size ({args.hidden_size})")
        previous = getattr(model, "metadata", {})
        if previous.get("tokenizer_sha256") != tokenizer_hash:
            ap.error("Checkpoint tokenizer fingerprint does not match this run. Use the original tokenizer or start a new run.")
        if previous.get("training_corpus_sha256") not in (None, training_corpus_hash):
            ap.error("Training corpus changed since the checkpoint was written; resume with the original corpus or start a new run.")
        if previous.get("validation_corpus_sha256") not in (None, validation_corpus_hash):
            ap.error("Validation corpus changed since the checkpoint was written; resume with the original corpus or start a new run.")
        if previous.get("seq_len") not in (None, args.seq_len):
            ap.error("Sequence length differs from the saved run configuration.")
        if previous.get("learning_rate") not in (None, args.learning_rate):
            ap.error("Learning rate differs from the saved run configuration.")
        if previous.get("seed") not in (None, args.seed):
            ap.error("Seed differs from the saved run configuration.")
        saved_rng_state = previous.get("rng_state")
        best_validation = float(previous.get("best_validation_loss", float("inf")))
    else:
        model = SoraRNN(args.hidden_size, args.seed, vocab_size=vocab_size)
        best_validation = float("inf")

    # New checkpoints preserve the exact sampling position across a resume.
    # Older checkpoints have no RNG state; maintain a deterministic fallback.
    rng = np.random.default_rng(args.seed + model.step)
    if saved_rng_state is not None:
        try:
            rng.bit_generator.state = saved_rng_state
        except (TypeError, ValueError, KeyError) as exc:
            ap.error(f"Checkpoint contains an invalid RNG state: {exc}")
    print(f"Training tokens: {len(train):,} | validation tokens: {len(valid):,}")
    print(f"Model: manual byte/BPE RNN | vocab: {vocab_size:,} | hidden: {model.hidden_size:,}")
    print(f"Parameters: {model.parameter_count:,} | trainable arrays: {len(model.p)}")
    print(f"Token cache: {cache_dir} | metrics: {log_path}")
    if not args.validation_corpus:
        print("WARNING: token-level split fallback is for smoke tests; use prepare_corpus.py for document-level splits.")
    started = time.time()

    for local_step in range(1, args.steps + 1):
        window = min(args.seq_len + 1, len(train))
        max_start = len(train) - window
        start = int(rng.integers(0, max_start + 1)) if max_start else 0
        sequence = np.asarray(train[start:start + window], dtype=np.int64)
        loss = model.train_sequence(sequence, args.learning_rate)
        global_step = model.step
        should_eval = local_step == 1 or local_step % args.eval_every == 0 or local_step == args.steps
        if not should_eval:
            continue

        validation_loss = evaluate(model, valid, args.seq_len, args.eval_batches)
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
        metadata = {
            "training_corpus": args.corpus.name,
            "training_corpus_sha256": training_corpus_hash,
            "validation_corpus": args.validation_corpus.name if args.validation_corpus else None,
            "validation_corpus_sha256": validation_corpus_hash,
            "tokenizer_sha256": tokenizer_hash,
            "best_validation_loss": best_validation,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "learning_rate": args.learning_rate,
            "rng_state": rng.bit_generator.state,
            "run_step": global_step,
        }
        model.save(args.checkpoint, metadata)
        if improved:
            best_path = args.checkpoint.with_name(args.checkpoint.stem + ".best" + args.checkpoint.suffix)
            model.save(best_path, metadata)

        elapsed = time.time() - started
        record = {
            "step": global_step,
            "train_loss": float(loss),
            "validation_loss": float(validation_loss),
            "best_validation_loss": float(best_validation),
            "tokens_per_second_estimate": round(local_step * window / max(elapsed, 1e-9), 3),
            "elapsed_seconds": round(elapsed, 3),
            "checkpoint": str(args.checkpoint),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"step={global_step} train_loss={loss:.4f} validation_loss={validation_loss:.4f} "
              f"best={best_validation:.4f} elapsed={elapsed:.1f}s")

    print(f"Checkpoint saved: {args.checkpoint}")
    print("--- Sample (early samples may be incoherent) ---")
    if tokenizer:
        prompt_ids = tokenizer.encode(args.prompt)
        output_ids = model.generate_ids(prompt_ids, args.generate_tokens, seed=args.seed, top_k=args.top_k)
        print(tokenizer.decode(output_ids))
    else:
        print(model.generate(args.prompt, args.generate_tokens, seed=args.seed, top_k=args.top_k))


if __name__ == "__main__":
    main()
