import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_multilingual_corpus import build_corpus


class MultilingualCorpusTests(unittest.TestCase):
    def test_only_policy_languages_are_prepared_and_missing_languages_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            (source / "fil").mkdir(parents=True)
            (source / "en").mkdir(parents=True)
            (source / "fil" / "one.txt").write_text("Mahal kita.\nKumusta ka?", encoding="utf-8")
            (source / "en" / "one.txt").write_text("Hello there.\nHow are you?", encoding="utf-8")
            policy = root / "policy.json"
            policy.write_text(json.dumps({
                "format": "lfm-country-primary-languages-v1",
                "countries": [
                    {"country_code": "PH", "country": "Philippines", "primary_language": "fil"},
                    {"country_code": "US", "country": "United States", "primary_language": "en"},
                    {"country_code": "JP", "country": "Japan", "primary_language": "ja"}
                ]
            }), encoding="utf-8")
            output = root / "selected"
            manifest = build_corpus(source, policy, output)
            self.assertEqual((output / "fil.txt").read_text(encoding="utf-8").strip(),
                             "Mahal kita.\nKumusta ka?")
            self.assertEqual((output / "en.txt").read_text(encoding="utf-8").strip(),
                             "Hello there.\nHow are you?")
            self.assertEqual(manifest["languages"]["ja"]["status"], "missing-source-data")
            self.assertEqual(manifest["summary"]["languages_with_data"], 2)
            self.assertIn("ja", manifest["summary"]["languages_missing_data"])
            saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["format"], "lfm-multilingual-corpus-v1")

    def test_duplicate_country_codes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy.json"
            policy.write_text(json.dumps({
                "format": "lfm-country-primary-languages-v1",
                "countries": [
                    {"country_code": "PH", "country": "Philippines", "primary_language": "fil"},
                    {"country_code": "PH", "country": "Philippines again", "primary_language": "tl"}
                ]
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate country code"):
                build_corpus(root / "raw", policy, root / "selected")


if __name__ == "__main__":
    unittest.main()
