#!/usr/bin/env python3
"""Validate and format licensed multilingual user/assistant examples for Sora tuning.

Input JSONL fields: language, user, assistant, source, license. Each row is one
exchange. This prepares training text; it does not train or verify language fluency.
"""
from __future__ import annotations
import argparse, json, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from prepare_multilingual_corpus import load_policy, sha256_file


def format_example(language: str, example: dict) -> str:
    return (
        f"<|system|>You are Sora, a helpful conversational assistant. Reply naturally and accurately in language {language}; follow the user's language unless asked otherwise.\n"
        f"<|user|>{example['user']}\n"
        f"<|assistant|>{example['assistant']}\n"
        "<|end|>\n\n"
    )


def prepare_chat_corpus(input_path: Path, policy_path: Path, output_dir: Path,
                        max_examples_per_language: int = 0) -> dict:
    input_path, policy_path, output_dir = map(Path, (input_path, policy_path, output_dir))
    if not input_path.is_file():
        raise FileNotFoundError(f"Dialogue JSONL not found: {input_path}")
    if max_examples_per_language < 0:
        raise ValueError("max_examples_per_language cannot be negative")
    _, languages = load_policy(policy_path)
    allowed = set(languages)
    rows = defaultdict(list)
    licenses = Counter()
    sources = Counter()
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must be a JSON object")
            language, user, assistant = row.get("language"), row.get("user"), row.get("assistant")
            source, license_name = row.get("source"), row.get("license")
            if not all(isinstance(v, str) and v.strip()
                       for v in (language, user, assistant, source, license_name)):
                raise ValueError(
                    f"Line {line_number} requires non-empty language, user, assistant, source, and license fields"
                )
            language = language.strip()
            if language not in allowed:
                raise ValueError(
                    f"Line {line_number} language {language!r} is not present in the country policy"
                )
            user = unicodedata.normalize("NFC", user.strip())
            assistant = unicodedata.normalize("NFC", assistant.strip())
            if not user or not assistant:
                raise ValueError(f"Line {line_number} has an empty user or assistant message")
            rows[language].append({
                "user": user, "assistant": assistant, "source": source.strip(),
                "license": license_name.strip(), "line": line_number
            })
            licenses[license_name.strip()] += 1
            sources[source.strip()] += 1

    if not rows:
        raise ValueError("No valid dialogue examples found")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_languages, selected_by_language = {}, {}
    total_examples = 0
    for language in languages:
        examples = rows.get(language, [])
        if max_examples_per_language:
            examples = examples[:max_examples_per_language]
        if not examples:
            report_languages[language] = {"status": "missing-examples", "examples": 0, "output": None}
            continue
        selected_by_language[language] = examples
        destination = output_dir / f"{language}.txt"
        with destination.open("w", encoding="utf-8", newline="\n") as target:
            for example in examples:
                target.write(format_example(language, example))
        report_languages[language] = {
            "status": "prepared",
            "examples": len(examples),
            "output": destination.name,
            "output_sha256": sha256_file(destination),
            "source_records": sorted({item["source"] for item in examples}),
            "licenses": dict(Counter(item["license"] for item in examples)),
        }
        total_examples += len(examples)

    combined_path = output_dir / "chat-tuning.txt"
    ordered_languages = sorted(selected_by_language)
    max_rows = max((len(selected_by_language[language]) for language in ordered_languages), default=0)
    # Interleave languages by example index instead of placing whole languages consecutively.
    with combined_path.open("w", encoding="utf-8", newline="\n") as target:
        for row_index in range(max_rows):
            for language in ordered_languages:
                examples = selected_by_language[language]
                if row_index < len(examples):
                    target.write(format_example(language, examples[row_index]))
    combined_hash = sha256_file(combined_path)

    counts = [entry["examples"] for entry in report_languages.values() if entry["examples"]]
    manifest = {
        "format": "lfm-chat-tuning-corpus-v1",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "languages": report_languages,
        "combined_training_file": combined_path.name,
        "combined_training_sha256": combined_hash,
        "summary": {
            "examples": total_examples,
            "languages_with_examples": len(counts),
            "languages_missing_examples": [code for code, item in report_languages.items()
                                           if item["status"] != "prepared"],
            "min_examples_per_available_language": min(counts) if counts else 0,
            "max_examples_per_language": max(counts) if counts else 0,
            "licenses_seen": dict(licenses),
            "sources_seen": dict(sources),
        },
        "warning": "Check translations, grammar, cultural relevance, and source licenses manually. A formatted dataset is not evidence of fluency.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="licensed JSONL of multilingual user/assistant examples")
    parser.add_argument("--policy", type=Path, default=Path("configs/country_primary_languages.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/multilingual/chat-selected"))
    parser.add_argument("--max-examples-per-language", type=int, default=0,
                        help="optional cap for each language; 0 keeps all examples")
    args = parser.parse_args()
    try:
        manifest = prepare_chat_corpus(args.input, args.policy, args.output_dir,
                                       args.max_examples_per_language)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
