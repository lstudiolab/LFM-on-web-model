#!/usr/bin/env python3
"""Run Sora's multilingual-first training pipeline from random initialization.

Example on a CUDA notebook:
  python python/run_multilingual_from_scratch.py --steps 20000

The data collector is resumable. This runner deliberately never accepts an
initialize-from checkpoint argument: the model begins with random weights.
"""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, sys
from collections import defaultdict
from pathlib import Path

def run(command):
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)

def load_samples(data_dir, min_chars):
    by_language, seen = defaultdict(list), set()
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = item.get("text")
                language = item.get("language_script") or item.get("source_config") or "unknown"
                if not isinstance(text, str):
                    continue
                text = " ".join(text.split())
                if len(text) < min_chars:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                by_language[language].append(text)
    return by_language

def balanced_split(samples, output_dir, seed, validation_fraction):
    train_queues, valid_queues = [], []
    summary = {}
    for language in sorted(samples):
        docs = sorted(samples[language],
                      key=lambda text: hashlib.sha256((str(seed) + text).encode("utf-8")).hexdigest())
        if len(docs) >= 20:
            count = max(1, round(len(docs) * validation_fraction))
        elif len(docs) >= 2:
            count = 1
        else:
            count = 0
        valid_docs, train_docs = docs[:count], docs[count:]
        random.Random(f"{seed}:{language}:train").shuffle(train_docs)
        random.Random(f"{seed}:{language}:valid").shuffle(valid_docs)
        if train_docs:
            train_queues.append((language, train_docs))
        if valid_docs:
            valid_queues.append((language, valid_docs))
        summary[language] = {"documents": len(docs), "train_documents": len(train_docs),
                             "validation_documents": len(valid_docs)}

    def write_round_robin(queues, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        active = [(language, list(docs)) for language, docs in queues]
        total = 0
        with path.open("w", encoding="utf-8", newline="\n") as sink:
            while active:
                rest = []
                for language, docs in active:
                    sink.write(docs.pop() + "\n")
                    total += 1
                    if docs:
                        rest.append((language, docs))
                active = rest
        return total

    corpus_dir = output_dir / "corpus"
    train_docs = write_round_robin(train_queues, corpus_dir / "train.txt")
    valid_docs = write_round_robin(valid_queues, corpus_dir / "validation.txt")
    if train_docs < 2 or valid_docs < 2:
        raise RuntimeError(f"Not enough sampled documents (train={train_docs}, validation={valid_docs}). Collect more data.")
    manifest = {
        "format": "lfm-balanced-multilingual-split-v1", "seed": seed,
        "validation_fraction_target": validation_fraction,
        "language_script_pairs_with_documents": len(summary),
        "train_documents": train_docs, "validation_documents": valid_docs,
        "total_documents": train_docs + valid_docs, "languages": summary,
        "note": "Language/script pairs are interleaved to avoid first-file dominance. This is not proof of language proficiency."
    }
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Corpus: {train_docs:,} train docs; {valid_docs:,} validation docs; {len(summary):,} language/script pairs.")
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/multilingual/fineweb2"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/sora-multilingual-from-scratch"))
    parser.add_argument("--max-docs-per-config", type=int, default=100)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--english-docs", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-lines", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.vocab_size < 256:
        parser.error("steps and batch-size must be positive; vocab-size must be >= 256")
    if args.heads < 1 or args.d_model < 8 or args.d_model % args.heads or args.layers < 1:
        parser.error("d-model must be >= 8 and divisible by positive heads; layers must be positive")
    if args.seq_len < 2 or args.tokenizer_lines < 0 or not 0 < args.validation_fraction < 0.5 or args.min_chars < 1:
        parser.error("invalid sequence length, tokenizer sample, validation fraction, or minimum document length")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        run([
            sys.executable, str(Path(__file__).with_name("download_multilingual_fineweb2.py")),
            "--output-dir", str(args.data_dir),
            "--max-docs-per-config", str(args.max_docs_per_config),
            "--max-configs", str(args.max_configs),
            "--english-docs", str(args.english_docs),
            "--min-chars", str(args.min_chars),
            "--seed", str(args.seed),
        ])

    train_text = args.run_dir / "corpus" / "train.txt"
    valid_text = args.run_dir / "corpus" / "validation.txt"
    tokenizer_path = args.run_dir / "tokenizer.json"
    checkpoint = args.run_dir / "sora-multilingual.pt"

    if args.resume:
        # A resume must reuse the exact corpus split and tokenizer from the saved run.
        # Rebuilding either one can invalidate the trainer's data/tokenizer fingerprints.
        required = [train_text, valid_text, tokenizer_path, checkpoint]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            parser.error("Cannot resume; missing artifacts: " + ", ".join(missing))
        print("Resuming with the existing train split, validation split, tokenizer, and checkpoint.")
        print("Dataset downloading and tokenizer training are skipped for a resume.")
    else:
        samples = load_samples(args.data_dir, args.min_chars)
        if not samples:
            parser.error(f"No usable JSONL documents found in {args.data_dir}; run with internet enabled and inspect the downloader manifest.")
        split_manifest = balanced_split(samples, args.run_dir, args.seed, args.validation_fraction)
        if split_manifest["language_script_pairs_with_documents"] < 2:
            print("WARNING: this corpus contains fewer than two language/script pairs.", flush=True)
        run([
            sys.executable, str(Path(__file__).with_name("train_bpe.py")), str(train_text), str(tokenizer_path),
            "--vocab-size", str(args.vocab_size), "--max-lines", str(args.tokenizer_lines), "--seed", str(args.seed)
        ])

    if args.prepare_only:
        print("PREPARED ONLY: no model training started.")
        print(f"Training corpus: {train_text}")
        print(f"Validation corpus: {valid_text}")
        print(f"Tokenizer: {tokenizer_path}")
        return 0

    command = [
        sys.executable, str(Path(__file__).with_name("train_torch_transformer.py")), str(train_text),
        "--validation-corpus", str(valid_text), "--tokenizer", str(tokenizer_path),
        "--checkpoint", str(checkpoint), "--steps", str(args.steps), "--batch-size", str(args.batch_size),
        "--d-model", str(args.d_model), "--heads", str(args.heads), "--layers", str(args.layers),
        "--seq-len", str(args.seq_len), "--dropout", "0.1", "--learning-rate", str(args.learning_rate),
        "--warmup-steps", "500", "--weight-decay", "0.1", "--eval-every", str(args.eval_every),
        "--eval-batches", str(args.eval_batches), "--seed", str(args.seed), "--device", args.device
    ]
    if args.resume:
        command.append("--resume")
    # Deliberately no --initialize-from: the weights must start randomly.
    run(command)
    print("Training process completed. Review metrics and per-language generations before claiming language ability.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
