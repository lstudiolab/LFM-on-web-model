#!/usr/bin/env python3
"""Evaluate one checkpoint on separate held-out test files, one per language.

Expected test files are <language-code>.txt. Never point this at training data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from evaluate import evaluate_checkpoint
from prepare_multilingual_corpus import load_policy


def evaluate_languages(checkpoint: Path, test_dir: Path, policy: Path, tokenizer_path=None,
                       backend: str = "numpy", device: str = "auto"):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_dir.is_dir():
        raise NotADirectoryError(f"Test directory not found: {test_dir}")
    _, languages = load_policy(policy)
    reports = {}
    for language in languages:
        candidates = [test_dir / f"{language}.txt", test_dir / language / "test.txt"]
        test_file = next((path for path in candidates if path.is_file()), None)
        if test_file is None:
            reports[language] = {"status": "missing-test-corpus"}
            continue
        try:
            if backend == "torch":
                from evaluate_torch import evaluate_torch_checkpoint
                result = evaluate_torch_checkpoint(checkpoint, test_file, tokenizer_path, device)
            else:
                result = evaluate_checkpoint(checkpoint, test_file, tokenizer_path)
            result["status"] = "evaluated"
            result["language"] = language
            reports[language] = result
        except (OSError, UnicodeError, ValueError) as exc:
            reports[language] = {"status": "evaluation-error", "error": str(exc)}
    evaluated = [item for item in reports.values() if item.get("status") == "evaluated"]
    return {
        "format": "lfm-multilingual-evaluation-v1",
        "checkpoint": str(checkpoint),
        "backend": backend,
        "policy": str(policy),
        "test_directory": str(test_dir),
        "languages_in_policy": len(languages),
        "languages_evaluated": len(evaluated),
        "languages_missing_test_data": [language for language, item in reports.items()
                                         if item.get("status") == "missing-test-corpus"],
        "per_language": reports,
        "warning": "Perplexity only measures next-token prediction on each corpus. It does not prove conversational fluency, factual accuracy, or translation quality. Keep all test files outside training and tuning.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--test-dir", type=Path, default=Path("data/multilingual/test"))
    parser.add_argument("--policy", type=Path, default=Path("configs/country_primary_languages.json"))
    parser.add_argument("--tokenizer", type=Path, help="exact BPE tokenizer used by this checkpoint")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", default="auto", help="used only with --backend torch")
    parser.add_argument("--output", type=Path, default=Path("runs/sora-multilingual/language-evaluation.json"))
    args = parser.parse_args()
    try:
        report = evaluate_languages(args.checkpoint, args.test_dir, args.policy, args.tokenizer,
                                    args.backend, args.device)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
