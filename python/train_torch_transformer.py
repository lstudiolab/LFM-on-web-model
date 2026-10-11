#!/usr/bin/env python3
"""Train Sora's causal Transformer from random initialization with batched PyTorch ops.

This optional backend supports CUDA/MPS when the installed PyTorch build does. It does
not download data or use pretrained model weights. Use reviewed corpora, separate
validation data, and a checkpoint/tokenizer from this same backend for staged tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from sora_torch_transformer import SoraTorchTransformer, load_checkpoint, save_checkpoint
from sora_transformer import sha256_file
from tokenize_corpus import tokenize_file
from tokenizer import ByteBPETokenizer


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available in this PyTorch build")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available on this machine")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def open_token_array(path: Path, manifest):
    return np.memmap(
        path, dtype=np.dtype(manifest["dtype"]), mode="r",
        shape=(int(manifest["token_count"]),),
    )


def evaluation_loss(model, token_ids, seq_len: int, batch_count: int, device) -> float:
    if len(token_ids) < 2:
        raise ValueError("Validation corpus must contain at least two tokens")
    window = min(seq_len + 1, len(token_ids))
    max_start = len(token_ids) - window
    starts = np.linspace(0, max_start, num=min(batch_count, max_start + 1), dtype=int)
    total_loss, total_targets = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in starts:
            sequence = np.asarray(token_ids[start:start + window], dtype=np.int64)
            ids = torch.as_tensor(sequence[:-1][None, :], dtype=torch.long, device=device)
            targets = torch.as_tensor(sequence[1:][None, :], dtype=torch.long, device=device)
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            count = int(targets.numel())
            total_loss += float(loss.item()) * count
            total_targets += count
    model.train()
    return total_loss / total_targets


def learning_rate_for_step(step: int, planned_steps: int, base_lr: float,
                           warmup_steps: int, minimum_ratio: float) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    decay_steps = max(1, planned_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def save_run_checkpoint(path, model, optimizer, metadata, rng):
    state = {
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    save_checkpoint(path, model, optimizer, metadata, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="UTF-8 training corpus")
    parser.add_argument("--validation-corpus", type=Path, required=True,
                        help="separate validation text; keep the final test set separate")
    parser.add_argument("--tokenizer", type=Path, help="BPE tokenizer JSON; omit for raw UTF-8 byte IDs")
    parser.add_argument("--checkpoint", type=Path, default=Path("sora-torch.pt"))
    parser.add_argument("--initialize-from", type=Path, help="previous PyTorch checkpoint; starts a new stage and resets optimizer state")
    parser.add_argument("--resume", action="store_true", help="resume exactly the same run configuration and data")
    parser.add_argument("--steps", type=int, default=10000, help="additional optimizer updates for this run")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--token-cache-dir", type=Path, help="default: CHECKPOINT_DIR/tokens")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.eval_batches < 1:
        parser.error("steps, batch-size, eval-every, and eval-batches must be positive")
    if args.d_model < 8 or args.heads < 1 or args.d_model % args.heads:
        parser.error("d-model must be at least 8 and divisible by a positive --heads")
    if args.layers < 1 or args.seq_len < 2:
        parser.error("layers must be positive and seq-len must be at least 2")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("learning-rate and grad-clip must be positive; weight-decay cannot be negative")
    if args.warmup_steps < 0 or not 0 <= args.min_learning_rate_ratio <= 1:
        parser.error("warmup-steps must be non-negative and min-learning-rate-ratio in [0, 1]")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from cannot be used together")
    if not args.corpus.is_file() or not args.validation_corpus.is_file():
        parser.error("Both training corpus and separate validation corpus must exist")
    if args.tokenizer and not args.tokenizer.is_file():
        parser.error(f"Tokenizer not found: {args.tokenizer}")
    if args.initialize_from and not args.initialize_from.is_file():
        parser.error(f"Initial checkpoint not found: {args.initialize_from}")
    if args.initialize_from and args.initialize_from.resolve() == args.checkpoint.resolve():
        parser.error("Use a different output checkpoint path when initializing from a checkpoint")
    if args.resume and not args.checkpoint.is_file():
        parser.error(f"Cannot resume: checkpoint not found: {args.checkpoint}")

    try:
        device = choose_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(args.seed)

    tokenizer = ByteBPETokenizer.load(args.tokenizer) if args.tokenizer else None
    vocab_size = tokenizer.vocab_size if tokenizer else 256
    tokenizer_hash = sha256_file(args.tokenizer) if args.tokenizer else None
    training_hash = sha256_file(args.corpus)
    validation_hash = sha256_file(args.validation_corpus)
    cache_dir = args.token_cache_dir or (args.checkpoint.parent / "tokens")
    train_path, train_manifest = tokenize_file(args.corpus, cache_dir / "train.tokens.bin", args.tokenizer)
    valid_path, valid_manifest = tokenize_file(args.validation_corpus, cache_dir / "validation.tokens.bin", args.tokenizer)
    train = open_token_array(train_path, train_manifest)
    valid = open_token_array(valid_path, valid_manifest)
    if len(train) < 2 or len(valid) < 2:
        parser.error("Training and validation corpora must each contain at least two tokens")

    config = {
        "vocab_size": vocab_size, "d_model": args.d_model, "n_heads": args.heads,
        "n_layers": args.layers, "max_seq_len": args.seq_len, "dropout": args.dropout,
    }
    initial_checkpoint_hash = None
    best_validation = float("inf")
    global_step = 0
    optimizer = None
    if args.resume:
        payload = load_checkpoint(args.checkpoint, map_location=device)
        if payload["model_config"] != config:
            parser.error("Checkpoint model configuration differs from this run")
        metadata = payload.get("metadata", {})
        expected_hashes = {
            "tokenizer_sha256": tokenizer_hash,
            "training_corpus_sha256": training_hash,
            "validation_corpus_sha256": validation_hash,
        }
        for key, value in expected_hashes.items():
            if metadata.get(key) != value:
                parser.error(f"Checkpoint {key} differs from this run")
        for key, value in {
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
        }.items():
            if metadata.get(key) != value:
                parser.error(f"Checkpoint {key} differs from this run")
        model = SoraTorchTransformer(**config).to(device)
        model.load_state_dict(payload["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        if payload.get("optimizer_state") is None:
            parser.error("Checkpoint has no optimizer state to resume")
        optimizer.load_state_dict(payload["optimizer_state"])
        global_step = int(metadata.get("global_step", 0))
        best_validation = float(metadata.get("best_validation_loss", float("inf")))
        state = payload.get("training_state", {})
        if state.get("numpy_rng_state"):
            rng.bit_generator.state = state["numpy_rng_state"]
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_states") is not None:
            try:
                torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_states"]])
            except (RuntimeError, ValueError):
                print("WARNING: CUDA RNG state could not be restored exactly; continuing with current CUDA RNG.")
    else:
        model = SoraTorchTransformer(**config).to(device)
        if args.initialize_from:
            payload = load_checkpoint(args.initialize_from, map_location=device)
            if payload["model_config"] != config:
                parser.error("Initial checkpoint model configuration differs from this run")
            metadata = payload.get("metadata", {})
            if metadata.get("tokenizer_sha256") != tokenizer_hash:
                parser.error("Initial checkpoint tokenizer fingerprint differs from this run")
            model.load_state_dict(payload["model_state"])
            initial_checkpoint_hash = sha256_file(args.initialize_from)
            print(f"Initialized weights from {args.initialize_from}; optimizer state reset.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)

    planned_steps = global_step + args.steps
    optimizer_steps = 0
    if not args.resume:
        global_step = 0
    best_path = args.checkpoint.with_name(args.checkpoint.stem + ".best" + args.checkpoint.suffix)
    print(f"Device: {device} | train tokens: {len(train):,} | validation tokens: {len(valid):,}")
    print(f"Model parameters: {model.parameter_count:,} | vocab: {vocab_size:,} | batch: {args.batch_size} x {args.seq_len}")
    print(f"Checkpoint: {args.checkpoint} | metrics: {args.checkpoint.with_suffix('.metrics.jsonl')}")
    log_path = args.checkpoint.with_suffix(".metrics.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
    started = time.time()

    model.train()
    for local_step in range(1, args.steps + 1):
        step_for_schedule = global_step + 1
        lr = learning_rate_for_step(step_for_schedule, planned_steps, args.learning_rate,
                                    args.warmup_steps, args.min_learning_rate_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr

        window = min(args.seq_len + 1, len(train))
        max_start = len(train) - window
        starts = rng.integers(0, max_start + 1, size=args.batch_size)
        sequences = np.stack([
            np.asarray(train[int(start):int(start) + window], dtype=np.int64)
            for start in starts
        ])
        batch = torch.as_tensor(sequences, dtype=torch.long, device=device)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1
        optimizer_steps += 1

        if local_step != 1 and local_step % args.eval_every != 0 and local_step != args.steps:
            continue
        validation_loss = evaluation_loss(model, valid, args.seq_len, args.eval_batches, device)
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
        metadata = {
            "global_step": global_step,
            "planned_total_steps": planned_steps,
            "best_validation_loss": best_validation,
            "tokenizer_sha256": tokenizer_hash,
            "training_corpus": args.corpus.name,
            "training_corpus_sha256": training_hash,
            "validation_corpus": args.validation_corpus.name,
            "validation_corpus_sha256": validation_hash,
            "initial_checkpoint_sha256": initial_checkpoint_hash,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "current_learning_rate": lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "device": str(device),
            "training_backend": "pytorch",
        }
        save_run_checkpoint(args.checkpoint, model, optimizer, metadata, rng)
        if improved:
            save_run_checkpoint(best_path, model, optimizer, metadata, rng)

        elapsed = max(time.time() - started, 1e-9)
        tokens_seen = optimizer_steps * args.batch_size * (window - 1)
        record = {
            "step": global_step, "train_loss": float(loss.item()),
            "validation_loss": float(validation_loss), "best_validation_loss": float(best_validation),
            "learning_rate": lr, "tokens_seen_this_stage": tokens_seen,
            "tokens_per_second_estimate": round(tokens_seen / elapsed, 3),
            "elapsed_seconds": round(elapsed, 3), "device": str(device),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"step={global_step} train_loss={loss.item():.4f} validation_loss={validation_loss:.4f} "
              f"best={best_validation:.4f} lr={lr:.2e} elapsed={elapsed:.1f}s")

    print(f"Training complete. Latest checkpoint: {args.checkpoint}")
    print(f"Best validation checkpoint: {best_path}")


if __name__ == "__main__":
    main()
