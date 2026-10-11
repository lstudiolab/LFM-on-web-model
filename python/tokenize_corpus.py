#!/usr/bin/env python3
"""Tokenize UTF-8 text into a compact binary array with a reproducible cache manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sora_model import sha256_file
from tokenizer import ByteBPETokenizer


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def tokenize_file(source_path, output_path, tokenizer_path=None, force=False):
    source_path, output_path = Path(source_path), Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {source_path}")
    tokenizer = ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else None
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path else None
    source_hash = sha256_file(source_path)
    vocab_size = tokenizer.vocab_size if tokenizer else 256
    dtype = np.dtype(np.uint16 if vocab_size <= 65536 else np.uint32)
    if vocab_size > np.iinfo(np.uint32).max:
        raise ValueError("Tokenizer vocabulary is too large for the supported token cache")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _manifest_path(output_path)
    if not force and output_path.is_file() and meta_path.is_file():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if (existing.get("source_sha256") == source_hash
                    and existing.get("tokenizer_sha256") == tokenizer_hash
                    and existing.get("vocab_size") == vocab_size
                    and existing.get("dtype") == dtype.name
                    and output_path.stat().st_size == existing.get("token_count", -1) * dtype.itemsize):
                return output_path, existing
        except (OSError, ValueError, TypeError):
            pass

    token_count = 0
    with source_path.open("rb") as source, output_path.open("wb") as target:
        if tokenizer is None:
            # Byte IDs are the raw UTF-8 bytes: no Python list or RAM copy required.
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                token_count += len(block)
        else:
            # A newline is a tokenizer boundary, so line-by-line encoding is exact.
            for raw_line in source:
                line = raw_line.decode("utf-8", errors="strict")
                ids = tokenizer.encode(line)
                if ids:
                    np.asarray(ids, dtype=dtype).tofile(target)
                    token_count += len(ids)
    if token_count < 2:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Corpus must produce at least two tokens: {source_path}")

    result = {
        "format": "lfm-token-cache-v1",
        "source": source_path.name,
        "source_sha256": source_hash,
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": vocab_size,
        "dtype": dtype.name,
        "token_count": token_count,
        "cache_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        path, manifest = tokenize_file(args.corpus, args.output, args.tokenizer, args.force)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Tokenized {manifest['token_count']:,} tokens ({manifest['dtype']}, "
          f"vocab={manifest['vocab_size']}) to {path}")


if __name__ == "__main__":
    main()
