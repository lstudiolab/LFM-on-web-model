#!/usr/bin/env python3
"""Prepare a reproducible train/validation split at document boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".rst", ".html", ".xml", ".csv", ".jsonl"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def discover_files(inputs):
    files = []
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
        else:
            raise ValueError(f"Not a regular file or directory: {path}")
    unique = sorted({p.resolve() for p in files}, key=lambda p: str(p).casefold())
    if not unique:
        raise ValueError("No input text files found")
    return unique


def extract_documents(text: str, min_chars: int):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    chunks = re.split(r"\n[ \t]*\n+", text)
    for chunk in chunks:
        document = chunk.strip()
        if len(document) >= min_chars:
            yield document


def prepare(inputs, output_dir: Path, validation_fraction=0.02, seed=7,
            min_document_chars=1, license_note=""):
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")
    if min_document_chars < 1:
        raise ValueError("min_document_chars must be positive")

    files = discover_files(inputs)
    documents = []
    file_records = []
    seen = set()
    duplicate_count = 0
    for path in files:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        file_records.append({
            "name": path.name,
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        })
        for document in extract_documents(text, min_document_chars):
            key = hashlib.sha256(document.encode("utf-8")).hexdigest()
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            documents.append(document)

    if len(documents) < 2:
        raise ValueError(
            f"Need at least two unique documents/paragraphs for a document-level split; found {len(documents)}. "
            "Add more corpus text or use separate train and validation files."
        )

    indexes = list(range(len(documents)))
    random.Random(seed).shuffle(indexes)
    valid_count = max(1, round(len(documents) * validation_fraction))
    valid_count = min(valid_count, len(documents) - 1)
    valid_indexes = set(indexes[:valid_count])
    train_docs = [doc for i, doc in enumerate(documents) if i not in valid_indexes]
    valid_docs = [doc for i, doc in enumerate(documents) if i in valid_indexes]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    valid_path = output_dir / "validation.txt"
    train_text = "\n\n".join(train_docs) + "\n"
    valid_text = "\n\n".join(valid_docs) + "\n"
    train_path.write_text(train_text, encoding="utf-8", newline="\n")
    valid_path.write_text(valid_text, encoding="utf-8", newline="\n")
    manifest = {
        "format": "lfm-corpus-manifest-v1",
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "min_document_chars": min_document_chars,
        "files": file_records,
        "documents_total_unique": len(documents),
        "documents_deduplicated": duplicate_count,
        "train_documents": len(train_docs),
        "validation_documents": len(valid_docs),
        "train_chars": len(train_text),
        "validation_chars": len(valid_text),
        "train_bytes": len(train_text.encode("utf-8")),
        "validation_bytes": len(valid_text.encode("utf-8")),
        "train_sha256": sha256_bytes(train_text.encode("utf-8")),
        "validation_sha256": sha256_bytes(valid_text.encode("utf-8")),
        "license_note": license_note,
        "warning": "Verify permissions, license, privacy, duplication, and quality of every source before training."
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="UTF-8 text files and/or folders of text files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-document-chars", type=int, default=1)
    parser.add_argument("--license-note", default="", help="Human-reviewed license/source note recorded in manifest")
    args = parser.parse_args()
    try:
        report = prepare(args.inputs, args.output_dir, args.validation_fraction,
                         args.seed, args.min_document_chars, args.license_note)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared {report['documents_total_unique']} unique documents "
          f"({report['train_documents']} train, {report['validation_documents']} validation)")
    print(f"Train: {args.output_dir / 'train.txt'} ({report['train_bytes']:,} bytes)")
    print(f"Validation: {args.output_dir / 'validation.txt'} ({report['validation_bytes']:,} bytes)")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
