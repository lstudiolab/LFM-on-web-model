import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

downloader = load_module("download_multilingual_fineweb2", "python/download_multilingual_fineweb2.py")
runner = load_module("run_multilingual_from_scratch", "python/run_multilingual_from_scratch.py")

class MultilingualPipelineTests(unittest.TestCase):
    def test_clean_text_normalizes_whitespace_and_rejects_short_values(self):
        self.assertEqual(downloader.clean_text("  hello\n\tworld  ", 5), "hello world")
        self.assertIsNone(downloader.clean_text(" short ", 20))
        self.assertIsNone(downloader.clean_text(None, 1))

    def test_read_existing_counts_and_hashes_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "subset.jsonl"
            path.write_text(
                json.dumps({"text": "hola mundo"}) + "\n"
                + "{bad json}\n"
                + json.dumps({"text": "bonjour le monde"}) + "\n",
                encoding="utf-8",
            )
            count, hashes = downloader.read_existing(path)
            self.assertEqual(count, 2)
            self.assertEqual(len(hashes), 2)

    def test_load_samples_and_balanced_split_keep_language_buckets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            data_dir.mkdir()
            lines = []
            for language in ("spa_Latn", "tgl_Latn"):
                for i in range(40):
                    lines.append(json.dumps({
                        "text": f"{language} document number {i} with enough characters for this unit test.",
                        "language_script": language,
                    }, ensure_ascii=False))
            (data_dir / "sample.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            samples = runner.load_samples(data_dir, 10)
            self.assertEqual(set(samples), {"spa_Latn", "tgl_Latn"})
            manifest = runner.balanced_split(samples, root / "run", seed=2026, validation_fraction=0.1)
            self.assertEqual(manifest["language_script_pairs_with_documents"], 2)
            self.assertEqual(manifest["total_documents"], 80)
            train = (root / "run" / "corpus" / "train.txt").read_text(encoding="utf-8").splitlines()
            valid = (root / "run" / "corpus" / "validation.txt").read_text(encoding="utf-8").splitlines()
            self.assertTrue(train)
            self.assertTrue(valid)
            self.assertFalse(set(train) & set(valid))
            self.assertEqual(len(train) + len(valid), 80)

if __name__ == "__main__":
    unittest.main()
