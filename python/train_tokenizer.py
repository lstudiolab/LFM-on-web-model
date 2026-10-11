#!/usr/bin/env python3
"""Train Sora's byte-level BPE vocabulary from a UTF-8 text corpus."""
import argparse
from pathlib import Path
from sora_tokenizer import train_bpe, save_bpe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path, help="UTF-8 text file")
    ap.add_argument("--output", type=Path, default=Path("artifacts/sora.bpe"))
    ap.add_argument("--vocab-size", type=int, default=2048, help="Target size before adding EOS (minimum 256)")
    args = ap.parse_args()
    text = args.corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("Corpus is empty; provide a non-empty UTF-8 text file.")
    rules = train_bpe(text, args.vocab_size)
    save_bpe(args.output, rules)
    print(f"Tokenizer saved to {args.output}; {256 + len(rules)} ordinary token IDs and one EOS token.")


if __name__ == "__main__":
    main()
