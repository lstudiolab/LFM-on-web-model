#!/usr/bin/env python3
"""Prepare a corpus limited to explicitly selected country-primary languages.

Expected layout: data/multilingual/raw/<language-code>/*.txt
This script prepares local, reviewed data; it does not download data or train a model.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from pathlib import Path

SUFFIXES = {".txt", ".text", ".md", ".markdown", ".rst"}
LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_policy(path):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("format") != "lfm-country-primary-languages-v1":
        raise ValueError("Unsupported country-language policy format")
    countries = policy.get("countries")
    if not isinstance(countries, list) or not countries:
        raise ValueError("Policy must contain a non-empty countries list")
    seen, languages = set(), set()
    for entry in countries:
        code, name, language = entry.get("country_code"), entry.get("country"), entry.get("primary_language")
        if not all(isinstance(v, str) and v.strip() for v in (code, name, language)):
            raise ValueError("Each entry needs country_code, country, and primary_language")
        code = code.upper()
        if code in seen:
            raise ValueError(f"Duplicate country code: {code}")
        if not LANGUAGE_CODE.fullmatch(language):
            raise ValueError(f"Invalid language code {language!r} for {code}")
        seen.add(code)
        languages.add(language)
    return policy, sorted(languages)

def build_corpus(source_root, policy_path, output_dir, max_chars_per_language=0, seed=7):
    source_root, policy_path, output_dir = map(Path, (source_root, policy_path, output_dir))
    policy, languages = load_policy(policy_path)
    if max_chars_per_language < 0:
        raise ValueError("max_chars_per_language cannot be negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "lfm-multilingual-corpus-v1",
        "policy_file": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "policy_name": policy.get("name", ""),
        "selection_rule": "Only folders matching language codes in the policy are included.",
        "countries": policy["countries"],
        "languages": {},
        "warnings": [
            "This starter policy is not a complete list of countries or languages.",
            "Multilingual countries may have several widely used or official languages; this policy deliberately chooses one.",
            "Folder names are trusted labels, not automatic language detection. Review content and licenses.",
            "A prepared corpus or smoke test is not evidence of language competence."
        ],
    }
    for language in languages:
        root = source_root / language
        files = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES),
                       key=lambda p: str(p).casefold()) if root.is_dir() else []
        chunks, sources = [], []
        for path in files:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            chunks.append(text.strip())
            sources.append({"path": str(path.relative_to(source_root)), "sha256": sha256_file(path),
                            "characters": len(text)})
        text = "\n\n".join(chunks)
        if max_chars_per_language and len(text) > max_chars_per_language:
            # Keep paragraph/article units intact while sampling a reproducible,
            # similarly sized slice from each language.
            documents = [part.strip() for chunk in chunks for part in chunk.split("\n\n") if part.strip()]
            random.Random(seed).shuffle(documents)
            selected, count = [], 0
            for document in documents:
                if count + len(document) <= max_chars_per_language:
                    selected.append(document)
                    count += len(document)
            text = "\n\n".join(selected)
        status = "prepared" if len(text.strip()) >= 2 else ("missing-source-data" if not files else "insufficient-source-data")
        record = {"status": status, "files": len(sources), "characters": len(text), "sources": sources}
        if status == "prepared":
            output = output_dir / f"{language}.txt"
            output.write_text(text + "\n", encoding="utf-8")
            record.update({"output": output.name, "output_sha256": sha256_file(output)})
        else:
            record["output"] = None
        manifest["languages"][language] = record
    manifest["summary"] = {
        "countries_in_policy": len(policy["countries"]),
        "selected_languages": len(languages),
        "languages_with_data": sum(v["status"] == "prepared" for v in manifest["languages"].values()),
        "languages_missing_data": [k for k, v in manifest["languages"].items() if v["status"] != "prepared"],
        "total_characters": sum(v["characters"] for v in manifest["languages"].values()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/multilingual/raw"))
    parser.add_argument("--policy", type=Path, default=Path("configs/country_primary_languages.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/multilingual/selected"))
    parser.add_argument("--max-chars-per-language", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        manifest = build_corpus(args.source_root, args.policy, args.output_dir,
                                args.max_chars_per_language, args.seed)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    if manifest["summary"]["languages_missing_data"]:
        print("Add reviewed, licensed source text for missing languages before training.")

if __name__ == "__main__":
    main()
