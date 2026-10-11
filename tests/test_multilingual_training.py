import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from download_wikipedia_corpus import fetch_batch
from prepare_chat_corpus import prepare_chat_corpus
from sora_transformer import SoraTransformer, sha256_file


class TestMultilingualDataTools(unittest.TestCase):
    def _policy(self, path):
        path.write_text(json.dumps({
            "format": "lfm-country-primary-languages-v1",
            "name": "test-policy",
            "countries": [
                {"country_code": "PH", "country": "Philippines", "primary_language": "tl"},
                {"country_code": "US", "country": "United States", "primary_language": "en"}
            ]
        }), encoding="utf-8")

    def test_chat_corpus_formats_and_interleaves_available_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            self._policy(policy)
            source = root / "dialogue.jsonl"
            rows = [
                {"language": "tl", "user": "Kumusta ka?", "assistant": "Mabuti ako.",
                 "source": "original", "license": "CC0"},
                {"language": "en", "user": "How are you?", "assistant": "I am well.",
                 "source": "original", "license": "CC0"},
                {"language": "tl", "user": "Salamat.", "assistant": "Walang anuman.",
                 "source": "original", "license": "CC0"}
            ]
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                              encoding="utf-8")
            output = root / "prepared"
            manifest = prepare_chat_corpus(source, policy, output)
            combined = (output / "chat-tuning.txt").read_text(encoding="utf-8")
            self.assertIn("<|user|>Kumusta ka?", combined)
            self.assertIn("<|assistant|>I am well.", combined)
            self.assertEqual(manifest["summary"]["examples"], 3)
            self.assertEqual(manifest["languages"]["tl"]["examples"], 2)
            self.assertEqual(manifest["languages"]["en"]["examples"], 1)
            self.assertTrue((output / "manifest.json").is_file())

    def test_chat_corpus_rejects_unselected_language(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            self._policy(policy)
            source = root / "dialogue.jsonl"
            source.write_text(json.dumps({
                "language": "xx", "user": "x", "assistant": "y",
                "source": "example", "license": "CC0"
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not present in the country policy"):
                prepare_chat_corpus(source, policy, root / "out")

    def test_wikipedia_fetch_parses_page_extracts(self):
        payload = {
            "query": {"pages": [
                {"pageid": 12, "title": "Pahina", "extract": "Tekstong Tagalog"},
                {"pageid": 13, "title": "Walang teksto", "extract": ""}
            ]}
        }
        fake_response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch("urllib.request.urlopen", return_value=fake_response):
            result = fetch_batch("tl", timeout=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pageid"], 12)
        self.assertEqual(result[0]["extract"], "Tekstong Tagalog")
        self.assertIn("tl.wikipedia.org", result[0]["url"])

    def test_initialize_from_starts_a_new_training_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.npz"
            model = SoraTransformer(vocab_size=256, d_model=8, n_heads=2,
                                   n_layers=1, max_seq_len=8, seed=3)
            model.save(initial, {"tokenizer_sha256": None})
            training = root / "train.txt"
            validation = root / "validation.txt"
            training.write_text("hello world\nhello again\nSora talks.", encoding="utf-8")
            validation.write_text("different held out validation text.", encoding="utf-8")
            output = root / "fine-tuned.npz"
            import subprocess
            command = [
                sys.executable, str(ROOT / "python" / "train_transformer.py"),
                str(training), "--validation-corpus", str(validation),
                "--initialize-from", str(initial), "--checkpoint", str(output),
                "--steps", "1", "--d-model", "8", "--heads", "2", "--layers", "1",
                "--seq-len", "8", "--learning-rate", "0.005", "--eval-every", "1",
                "--eval-batches", "1", "--generate-tokens", "0"
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            trained = SoraTransformer.load(output)
            self.assertEqual(trained.step, 1)
            self.assertEqual(trained.metadata["initial_checkpoint_sha256"], sha256_file(initial))
            self.assertIsNone(trained.metadata["tokenizer_sha256"])


if __name__ == "__main__":
    unittest.main()
