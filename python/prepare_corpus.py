#!/usr/bin/env python3
"""Prepare a reproducible train/validation split at document boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import unicodedata
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".rst", ".html", ".xml", ".csv", ".jsonl"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def iter_documents(stream, min_chars: int):
    """Yield paragraph documents from a text stream without loading the whole file."""
    lines = []

    def normalize_lines(raw_lines):
        normalized = [unicodedata.normalize("NFC", line.rstrip("\r\n").rstrip())
                      for line in raw_lines]
        return "\n".join(normalized).strip()

    for line in stream:
        if not line.strip():
            document = normalize_lines(lines) if lines else ""
            if len(document) >= min_chars:
                yield document
            lines = []
        else:
            lines.append(line)
    document = normalize_lines(lines) if lines else ""
    if len(document) >= min_chars:
        yield document


def extract_documents(text: str, min_chars: int):
    """In-memory helper retained for tests and small callers."""
    from io import StringIO
    yield from iter_documents(StringIO(text.replace("\r\n", "\n").replace("\r", "\n")),
                              min_chars)


def prepare(inputs, output_dir: Path, validation_fraction=0.02, seed=7,
            min_document_chars=1, license_note=""):
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")
    if min_document_chars < 1:
        raise ValueError("min_document_chars must be positive")

    files = discover_files(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / ".corpus-work.sqlite3"
    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, digest TEXT UNIQUE NOT NULL, text TEXT NOT NULL)"
        )
        file_records = []
        duplicate_count = 0
        document_count = 0

        for path in files:
            file_records.append({
                "name": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
            with path.open("r", encoding="utf-8", newline=None) as source:
                for document in iter_documents(source, min_document_chars):
                    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO documents (digest, text) VALUES (?, ?)",
                        (digest, document),
                    )
                    if cursor.rowcount == 1:
                        document_count += 1
                    else:
                        duplicate_count += 1
            connection.commit()

        if document_count < 2:
            raise ValueError(
                f"Need at least two unique documents/paragraphs for a document-level split; found {document_count}. "
                "Add more corpus text or use separate train and validation files."
            )

        ids = [int(row[0]) for row in connection.execute("SELECT id FROM documents ORDER BY id")]
        random.Random(seed).shuffle(ids)
        validation_count = max(1, round(document_count * validation_fraction))
        validation_count = min(validation_count, document_count - 1)
        validation_ids = set(ids[:validation_count])

        train_path = output_dir / "train.txt"
        validation_path = output_dir / "validation.txt"

        def write_split(path: Path, is_validation: bool):
            digest = hashlib.sha256()
            char_count = 0
            byte_count = 0
            count = 0
            first = True
            with path.open("w", encoding="utf-8", newline="\n") as target:
                for doc_id, document in connection.execute("SELECT id, text FROM documents ORDER BY id"):
                    belongs_to_validation = int(doc_id) in validation_ids
                    if belongs_to_validation != is_validation:
                        continue
                    if not first:
                        target.write("\n\n")
                        digest.update(b"\n\n")
                        byte_count += 2
                    target.write(document)
                    encoded = document.encode("utf-8")
                    digest.update(encoded)
                    byte_count += len(encoded)
                    char_count += len(document)
                    count += 1
                    first = False
                target.write("\n")
                digest.update(b"\n")
                byte_count += 1
            return {
                "documents": count,
                "chars": char_count,
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
            }

        train_stats = write_split(train_path, False)
        validation_stats = write_split(validation_path, True)
        manifest = {
            "format": "lfm-corpus-manifest-v1",
            "seed": seed,
            "validation_fraction_requested": validation_fraction,
            "min_document_chars": min_document_chars,
            "files": file_records,
            "documents_total_unique": document_count,
            "documents_deduplicated": duplicate_count,
            "train_documents": train_stats["documents"],
            "validation_documents": validation_stats["documents"],
            "train_chars": train_stats["chars"],
            "validation_chars": validation_stats["chars"],
            "train_bytes": train_stats["bytes"],
            "validation_bytes": validation_stats["bytes"],
            "train_sha256": train_stats["sha256"],
            "validation_sha256": validation_stats["sha256"],
            "license_note": license_note,
            "warning": "Verify permissions, license, privacy, duplication, and quality of every source before training."
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


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
    except (OSError, UnicodeError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(f"Prepared {report['documents_total_unique']} unique documents "
          f"({report['train_documents']} train, {report['validation_documents']} validation)")
    print(f"Train: {args.output_dir / 'train.txt'} ({report['train_bytes']:,} bytes)")
    print(f"Validation: {args.output_dir / 'validation.txt'} ({report['validation_bytes']:,} bytes)")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
