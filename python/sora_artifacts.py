#!/usr/bin/env python3
"""Collect, verify, and publish Sora training artifacts without committing binaries to Git history.

Typical Colab use:
  python python/sora_artifacts.py collect --search-root /content --output /content/sora-state-bundle.tar.gz
  python python/sora_artifacts.py publish --bundle /content/sora-state-bundle.tar.gz --repo lstudiolab/LFM-on-web-model --tag sora-state-2026-10-11

Set GITHUB_TOKEN in the environment before publish. The token must have Contents: read/write
and Releases: read/write access to the target repository. Never hard-code or print the token.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

REPO_DEFAULT = "lstudiolab/LFM-on-web-model"
MANIFEST_PATH = "models/sora-artifacts.json"
KNOWN_DIRS = (
    "from_scratch_checkpoints",
    "pretraining_checkpoints",
    "pretraining_tokenizer",
    "pretraining",
    "data/pretraining",
    "data/reports",
    "runs/sora-multilingual-from-scratch",
    "data/multilingual/fineweb2",
    "checkpoints",
)
STATE_NAMES = {
    "sora-first-training.pt", "sora-latest.pt", "sora-epoch-1.pt",
    "sora-multilingual.pt", "sora-multilingual.best.pt", "sora-tokenizer.json",
    "tokenizer.json", "sora_config.json", "run_config.json",
    "sora_pretraining_run.json", "sora_pretraining_v1_report.json", "sora_baseline_evaluation.json",
    "baseline_evaluation.json", "sora_500m_readiness.json",
    "manifest.json", "sora-multilingual.metrics.jsonl",
}
MODEL_SUFFIXES = {".pt", ".pth", ".npz", ".safetensors"}
CORPUS_SUFFIXES = {".jsonl", ".txt", ".text", ".tokens.bin"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(search_roots: list[Path], include_corpus: bool) -> list[tuple[Path, str]]:
    found: dict[str, Path] = {}
    corpus_prefixes = ("data/pretraining", "data/multilingual", "runs/sora-multilingual-from-scratch",
                       "pretraining_checkpoints", "pretraining_tokenizer")
    for root in search_roots:
        if not root.exists():
            print(f"[skip] Search root not found: {root}")
            continue
        root = root.resolve()
        for rel in KNOWN_DIRS:
            directory = root / rel
            if not directory.is_dir():
                continue
            for p in directory.rglob("*"):
                if not p.is_file() or any(part in SKIP_PARTS for part in p.parts):
                    continue
                name = p.name.lower()
                suffix = p.suffix.lower()
                is_model = suffix in MODEL_SUFFIXES
                is_state = name in {n.lower() for n in STATE_NAMES} or name.endswith(".metrics.jsonl")
                rel_posix = p.relative_to(root).as_posix()
                is_corpus = suffix in CORPUS_SUFFIXES or name.endswith(".tokens.bin")
                if is_corpus and not include_corpus:
                    # Manifests are kept because they identify data provenance and fingerprints.
                    if name != "manifest.json":
                        continue
                elif is_corpus and include_corpus:
                    if not any(rel_posix.startswith(prefix) for prefix in corpus_prefixes):
                        continue
                if is_model or is_state or (include_corpus and is_corpus):
                    # Preserve a root marker to avoid collisions between search roots.
                    label = root.name or "root"
                    arc = f"{label}/{rel_posix}"
                    found.setdefault(arc, p)
        # Catch the previous run's root-level config file.
        for name in ("sora_config.json",):
            p = root / name
            if p.is_file():
                found.setdefault(f"{root.name}/{name}", p)
    return [(path, arc) for arc, path in sorted(found.items())]


def collect(args) -> None:
    roots = [Path(x) for x in args.search_root]
    items = discover(roots, args.include_corpus)
    if not items:
        raise SystemExit("No matching Sora artifacts found. Check --search-root; no files were changed.")
    if not any(path.suffix.lower() in MODEL_SUFFIXES for path, _ in items):
        print("WARNING: no model checkpoint file was discovered; this bundle cannot restore trained weights.")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(output, "w:gz") as archive:
        for path, arcname in items:
            size = path.stat().st_size
            entries.append({"path": arcname, "size_bytes": size, "sha256": sha256(path)})
            archive.add(path, arcname=arcname, recursive=False)
        manifest = {
            "format": "sora-training-artifacts-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "contains_raw_corpus": bool(args.include_corpus),
            "files": entries,
            "file_count": len(entries),
            "total_uncompressed_bytes": sum(e["size_bytes"] for e in entries),
            "checkpoint_files": [e["path"] for e in entries if Path(e["path"]).suffix.lower() in MODEL_SUFFIXES],
            "warning": "This bundle preserves only the files discovered in this runtime. Missing Colab files cannot be recreated from this manifest."
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("SORA_BUNDLE_MANIFEST.json")
        info.size = len(payload)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(payload))
    print(f"Bundle: {output}")
    print(f"Files: {len(items)} | Uncompressed size: {manifest['total_uncompressed_bytes']:,} bytes")
    print(f"Checkpoint files found: {len(manifest['checkpoint_files'])}")
    if not manifest["checkpoint_files"]:
        print("IMPORTANT: This is not sufficient to restore Sora's learned weights.")


def request_json(url: str, token: str, method: str = "GET", payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Sora-Artifact-Publisher",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API returned HTTP {exc.code}: {detail[:1200]}") from exc


def get_existing_manifest(repo: str, token: str):
    url = f"https://api.github.com/repos/{repo}/contents/{MANIFEST_PATH}"
    try:
        return request_json(url, token)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise


def publish(args) -> None:
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"Missing {args.token_env}. Set it from a hidden input or secret store; do not paste it into this script.")
    repo = args.repo.strip("/")
    bundle = Path(args.bundle).resolve()
    if not bundle.is_file():
        raise SystemExit(f"Bundle not found: {bundle}")
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            member = archive.getmember("SORA_BUNDLE_MANIFEST.json")
            bundled_manifest = json.load(archive.extractfile(member))
    except (tarfile.TarError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Not a valid Sora artifact bundle: {exc}") from exc
    if not bundled_manifest.get("checkpoint_files"):
        raise SystemExit("Refusing to publish: this bundle has no model checkpoint. Find the trained .pt/.pth/.npz file first.")
    if bundle.stat().st_size >= 2 * 1024 * 1024 * 1024:
        raise SystemExit("GitHub release assets must be smaller than 2 GiB. Create a state-only bundle or split the corpus into a separate licensed artifact.")
    # Fetch public repo metadata; this also validates repository naming.
    repo_info = request_json(f"https://api.github.com/repos/{repo}", token)
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(args.tag, safe='')}"
    try:
        release = request_json(release_url, token)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        release = request_json(f"https://api.github.com/repos/{repo}/releases", token, "POST", {
            "tag_name": args.tag,
            "name": args.title or f"Sora training artifacts {args.tag}",
            "body": args.notes or "Sora checkpoint/tokenizer/config artifact bundle exported from Google Colab. See the bundled manifest for hashes and included files.",
            "draft": False,
            "prerelease": False,
        })
    upload_url = release.get("upload_url", "").split("{", 1)[0]
    if not upload_url:
        raise RuntimeError("GitHub release response did not include an upload URL.")
    asset_name = bundle.name
    # Replace an existing asset of the same name to make retrying safe.
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            request_json(asset["url"], token, "DELETE")
    upload_endpoint = upload_url + "?" + urllib.parse.urlencode({"name": asset_name})
    req = urllib.request.Request(upload_endpoint, data=bundle.read_bytes(), method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/gzip",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Sora-Artifact-Publisher",
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            asset = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Uploading release asset failed (HTTP {exc.code}): {detail[:1200]}") from exc
    manifest = {
        "format": "sora-release-artifact-pointer-v1",
        "repository": repo,
        "release_tag": args.tag,
        "release_url": release.get("html_url"),
        "asset_name": asset.get("name"),
        "asset_url": asset.get("browser_download_url"),
        "asset_size_bytes": asset.get("size"),
        "asset_sha256": sha256(bundle),
        "checkpoint_files": bundled_manifest.get("checkpoint_files", []),
        "contains_raw_corpus": bundled_manifest.get("contains_raw_corpus", False),
        "bundle_file_count": bundled_manifest.get("file_count"),
        "uploaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "usage": "Download and verify the release asset before loading the checkpoint. The tarball manifest lists individual file hashes.",
        "warning": "This manifest points only to artifacts actually found in the Colab runtime. It does not prove that missing older runs were recovered."
    }
    existing = get_existing_manifest(repo, token)
    body = {
        "message": f"Register Sora artifact release {args.tag}",
        "content": base64.b64encode((json.dumps(manifest, indent=2) + "\n").encode("utf-8")).decode("ascii"),
        "branch": args.branch,
    }
    if existing:
        body["sha"] = existing.get("sha")
    request_json(f"https://api.github.com/repos/{repo}/contents/{MANIFEST_PATH}", token, "PUT", body)
    print("Published release asset and updated repository manifest.")
    print(f"Repository: {repo_info.get('html_url', repo)}")
    print(f"Release: {release.get('html_url')}")
    print(f"Asset: {asset.get('browser_download_url')}")
    print(f"Manifest: https://github.com/{repo}/blob/{args.branch}/{MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("collect", help="find local Colab artifacts and create a hash-manifested tarball")
    scan.add_argument("--search-root", action="append", default=[],
                      help="root to search; repeat for /content and mounted Drive")
    scan.add_argument("--output", required=True)
    scan.add_argument("--include-corpus", action="store_true",
                      help="also bundle corpus/split/token-cache files from known data directories; review licensing before public upload")
    scan.set_defaults(func=collect)
    pub = sub.add_parser("publish", help="upload bundle as a GitHub Release asset and register a pointer in this repository")
    pub.add_argument("--bundle", required=True)
    pub.add_argument("--repo", default=REPO_DEFAULT)
    pub.add_argument("--tag", required=True, help="unique release tag, e.g. sora-state-2026-10-11")
    pub.add_argument("--title", default="")
    pub.add_argument("--notes", default="")
    pub.add_argument("--token-env", default="GITHUB_TOKEN")
    pub.add_argument("--branch", default="main")
    pub.set_defaults(func=publish)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
