#!/usr/bin/env python3
"""Resumably collect a bounded sample from available FineWeb-2 language/script subsets."""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path

FINEWEB2 = "HuggingFaceFW/fineweb-2"
FINEWEB_ENGLISH = "HuggingFaceFW/fineweb"

def clean_text(value, min_chars):
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) >= min_chars else None

def read_existing(path):
    count, hashes = 0, set()
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    text = json.loads(line).get("text", "")
                    if isinstance(text, str) and text:
                        count += 1
                        hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
                except (json.JSONDecodeError, AttributeError):
                    pass
    return count, hashes

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/multilingual/fineweb2"))
    parser.add_argument("--max-docs-per-config", type=int, default=100, help="0 streams each subset without a document cap")
    parser.add_argument("--max-configs", type=int, default=0, help="0 means all listed non-English configs")
    parser.add_argument("--english-docs", type=int, default=5000, help="English docs from FineWeb sample-10BT; 0 disables")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if min(args.max_docs_per_config, args.max_configs, args.english_docs) < 0:
        parser.error("document and config limits cannot be negative")
    if args.min_chars < 1 or args.shuffle_buffer < 0:
        parser.error("min-chars must be positive and shuffle-buffer cannot be negative")
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        print("Missing dependency. Run: pip install -U datasets huggingface_hub", file=sys.stderr)
        return 2

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        previous = {}
    manifest = {
        "format": "lfm-fineweb2-multilingual-sample-v1",
        "dataset": FINEWEB2,
        "dataset_license": "ODC-By 1.0; verify current dataset card and Common Crawl terms before redistribution",
        "english_dataset": FINEWEB_ENGLISH,
        "english_dataset_license": "ODC-By 1.0; verify current dataset card and Common Crawl terms before redistribution",
        "sampling": {"max_docs_per_config": args.max_docs_per_config, "max_configs": args.max_configs,
                     "english_docs": args.english_docs, "min_chars": args.min_chars, "seed": args.seed},
        "warning": "This is a broad sample, not every language on Earth. Presence of sample text is not proof of fluency.",
        "subsets": previous.get("subsets", {}),
        "started_at_unix": int(time.time()),
    }
    try:
        configs = get_dataset_config_names(FINEWEB2)
    except Exception as exc:
        print(f"Cannot list FineWeb-2 subsets: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    configs = sorted(c for c in configs if not c.endswith("_removed") and not c.startswith("und_")
                     and not c.lower().startswith("eng_"))
    if args.max_configs:
        configs = configs[:args.max_configs]
    print(f"FineWeb-2 language/script configs selected: {len(configs)}", flush=True)

    def collect(dataset_id, config, split, target, limit, label):
        count, hashes = read_existing(target)
        if limit and count >= limit:
            return {"status": "complete", "documents": count, "file": target.name}
        error, status = None, "complete"
        try:
            ds = load_dataset(dataset_id, name=config, split=split, streaming=True)
            if args.shuffle_buffer > 1:
                ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            with target.open("a", encoding="utf-8") as sink:
                for row in ds:
                    text = clean_text(row.get("text"), args.min_chars)
                    if not text:
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if digest in hashes:
                        continue
                    sink.write(json.dumps({
                        "text": text, "language_script": config, "source_dataset": dataset_id,
                        "source_config": config, "split_source": split,
                        "license": "ODC-By 1.0; review current dataset card and Common Crawl terms"
                    }, ensure_ascii=False) + "\n")
                    hashes.add(digest)
                    count += 1
                    if limit and count >= limit:
                        break
            if limit and count < limit:
                status = "exhausted"
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            print(f"[failed] {label}: {error}", file=sys.stderr, flush=True)
        result = {"status": status, "documents": count, "file": target.name}
        if error:
            result["error"] = error
        return result

    for index, config in enumerate(configs, start=1):
        target = out / f"{config}.jsonl"
        prior = manifest["subsets"].get(config, {})
        count, _ = read_existing(target)
        if args.max_docs_per_config and count >= args.max_docs_per_config:
            continue
        if prior.get("status") == "failed" and not args.retry_failed:
            continue
        result = collect(FINEWEB2, config, "train", target, args.max_docs_per_config, config)
        manifest["subsets"][config] = result
        if index == 1 or index % 10 == 0 or index == len(configs):
            atomic_json(manifest_path, manifest)
            print(f"[{index}/{len(configs)}] {config}: {result['status']} ({result['documents']} docs)", flush=True)

    if args.english_docs:
        target = out / "english_fineweb.jsonl"
        count, _ = read_existing(target)
        previous_english = manifest["subsets"].get("english_fineweb", {})
        if count < args.english_docs:
            result = collect(FINEWEB_ENGLISH, "sample-10BT", "train", target, args.english_docs, "english FineWeb")
            manifest["subsets"]["english_fineweb"] = result
            print(f"English FineWeb: {result['status']} ({result['documents']} docs)", flush=True)
    manifest["finished_at_unix"] = int(time.time())
    manifest["summary"] = {
        "configs_recorded": len(manifest["subsets"]),
        "configs_with_documents": sum(v.get("documents", 0) > 0 for v in manifest["subsets"].values()),
        "total_documents": sum(v.get("documents", 0) for v in manifest["subsets"].values()),
        "failed_configs": [k for k, v in manifest["subsets"].items() if v.get("status") == "failed"],
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0 if manifest["summary"]["total_documents"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
