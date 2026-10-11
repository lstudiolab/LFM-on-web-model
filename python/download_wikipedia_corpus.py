#!/usr/bin/env python3
"""Download a small licensed Wikipedia-article starter corpus per selected language.

This is a bootstrap data collector, not a complete pretraining dataset. For serious
training, use substantially larger reviewed datasets and keep attribution metadata.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "LlabFoundationModelCorpusBuilder/1.0 (research; contact repo maintainer)"
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def load_policy(path: Path) -> tuple[dict, list[str]]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("format") != "lfm-country-primary-languages-v1":
        raise ValueError("Unsupported country-language policy format")
    countries = policy.get("countries")
    if not isinstance(countries, list) or not countries:
        raise ValueError("Policy has no countries")
    languages = set()
    for entry in countries:
        code = entry.get("primary_language")
        country = entry.get("country")
        if not isinstance(code, str) or not code.strip() or not isinstance(country, str):
            raise ValueError("Invalid country/language entry in policy")
        languages.add(code.strip())
    return policy, sorted(languages)


def request_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_batch(language: str, timeout: float) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": "20",
        "prop": "extracts",
        "explaintext": "1",
        "exintro": "1",
        "exchars": "1200",
        "redirects": "1",
    }
    url = f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    payload = request_json(url, timeout)
    if "error" in payload:
        raise RuntimeError(payload["error"].get("info", "MediaWiki API error"))
    pages = payload.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    results = []
    for page in pages:
        title = str(page.get("title", "")).strip()
        extract = str(page.get("extract", "")).strip()
        page_id = page.get("pageid")
        if not title or not extract or page_id is None or page.get("missing"):
            continue
        results.append({
            "pageid": int(page_id),
            "title": title,
            "extract": extract,
            "url": f"https://{language}.wikipedia.org/?curid={int(page_id)}",
        })
    return results


def download_language(language: str, output_root: Path, pages_per_language: int,
                      delay: float, timeout: float, max_attempts: int) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    text_path = output_root / language / "wikipedia.txt"
    source_path = output_root / language / "wikipedia-sources.json"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    pages_by_id = {}
    if text_path.exists() and source_path.exists():
        try:
            prior = json.loads(source_path.read_text(encoding="utf-8"))
            if prior.get("language") == language:
                for page in prior.get("pages", []):
                    if page.get("pageid") and page.get("extract"):
                        pages_by_id[int(page["pageid"])] = page
        except (OSError, ValueError, TypeError):
            pass

    requests_made = 0
    errors = []
    while len(pages_by_id) < pages_per_language and requests_made < max_attempts:
        requests_made += 1
        try:
            batch = fetch_batch(language, timeout)
            for page in batch:
                pages_by_id.setdefault(page["pageid"], page)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            if len(errors) >= 3:
                break
        if len(pages_by_id) < pages_per_language:
            time.sleep(delay)

    pages = list(pages_by_id.values())[:pages_per_language]
    temp_text = text_path.with_suffix(".txt.tmp")
    with temp_text.open("w", encoding="utf-8", newline="\n") as target:
        for page in pages:
            target.write(page["title"] + "\n")
            target.write(page["extract"] + "\n\n")
    temp_text.replace(text_path)

    sources = {
        "format": "lfm-wikipedia-sources-v1",
        "language": language,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": f"https://{language}.wikipedia.org/",
        "license": WIKIPEDIA_LICENSE,
        "license_url": WIKIPEDIA_LICENSE_URL,
        "attribution_note": "Wikipedia content is collaboratively authored. Retain article URLs and review Wikimedia reuse requirements before redistributing derived datasets or models.",
        "pages": [{key: value for key, value in page.items() if key != "extract"} for page in pages],
        "page_count": len(pages),
        "characters": sum(len(p["title"]) + len(p["extract"]) for p in pages),
        "requests_made": requests_made,
        "errors": errors,
    }
    source_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return {
        "language": language,
        "pages": len(pages),
        "characters": sources["characters"],
        "text_path": str(text_path),
        "sources_path": str(source_path),
        "status": "downloaded" if pages else "unavailable-or-empty",
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/country_primary_languages.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/multilingual/raw"))
    parser.add_argument("--pages-per-language", type=int, default=100,
                        help="unique random article intros per selected language (not full articles)")
    parser.add_argument("--delay", type=float, default=0.25, help="delay between API requests per language")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-attempts-per-language", type=int, default=1000)
    parser.add_argument("--language", action="append", help="only download this language code; repeatable")
    args = parser.parse_args()
    if args.pages_per_language < 1 or args.delay < 0 or args.timeout <= 0 or args.max_attempts_per_language < 1:
        parser.error("pages-per-language and max-attempts must be positive; delay must be non-negative; timeout must be positive")
    try:
        policy, policy_languages = load_policy(args.policy)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    languages = sorted(set(args.language)) if args.language else policy_languages
    unknown = sorted(set(languages) - set(policy_languages))
    if unknown:
        parser.error(f"Language codes are not in the country policy: {', '.join(unknown)}")
    reports = []
    for index, language in enumerate(languages, start=1):
        print(f"[{index}/{len(languages)}] Downloading {language}: up to {args.pages_per_language} unique article intros")
        report = download_language(language, args.output_root, args.pages_per_language,
                                   args.delay, args.timeout, args.max_attempts_per_language)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
    manifest = {
        "format": "lfm-wikipedia-download-manifest-v1",
        "policy": str(args.policy),
        "policy_name": policy.get("name", ""),
        "language_count": len(languages),
        "reports": reports,
        "warning": "Random Wikipedia article introductions are only a starter corpus, not enough to claim fluency or worldwide language coverage.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "download-manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Download manifest: {out}")


if __name__ == "__main__":
    main()
