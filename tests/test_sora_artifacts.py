from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import sora_artifacts  # noqa: E402


class SoraArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_discover_keeps_checkpoint_and_tokenizer_but_excludes_corpus_and_lora(self):
        checkpoint = self.write("pretraining_checkpoints/sora-latest.pt", b"weights")
        tokenizer = self.write("pretraining_tokenizer/tokenizer.json", b'{"vocab":{}}')
        self.write("data/pretraining/sora_corpus.jsonl", b'{"text":"training text"}\n')
        self.write("checkpoints/lora/adapter_model.safetensors", b"not Sora full weights")

        found = sora_artifacts.discover([self.root], include_corpus=False)
        sources = {str(path) for path, _ in found}
        self.assertIn(str(checkpoint), sources)
        self.assertIn(str(tokenizer), sources)
        self.assertNotIn(str(self.root / "data/pretraining/sora_corpus.jsonl"), sources)
        self.assertNotIn(str(self.root / "checkpoints/lora/adapter_model.safetensors"), sources)

    def test_collect_writes_bundle_manifest_and_hashes(self):
        self.write("pretraining_checkpoints/sora-latest.pt", b"weights-for-test")
        self.write("pretraining_tokenizer/tokenizer.json", b'{"vocab":{}}')
        bundle = self.root / "bundle.tar.gz"
        args = type("Args", (), {
            "search_root": [str(self.root)],
            "output": str(bundle),
            "include_corpus": False,
        })()
        sora_artifacts.collect(args)
        self.assertTrue(bundle.is_file())
        with tarfile.open(bundle, "r:gz") as archive:
            manifest = json.load(archive.extractfile("SORA_BUNDLE_MANIFEST.json"))
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(len(manifest["checkpoint_files"]), 1)
            for item in manifest["files"]:
                stream = archive.extractfile(item["path"])
                self.assertIsNotNone(stream)
                data = stream.read()
                import hashlib
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
                self.assertEqual(len(data), item["size_bytes"])

    def test_include_corpus_is_explicit(self):
        corpus = self.write("data/pretraining/sora_corpus.jsonl", b'{"text":"training text"}\n')
        self.write("pretraining_checkpoints/sora-latest.pt", b"weights")
        found_default = sora_artifacts.discover([self.root], include_corpus=False)
        found_include = sora_artifacts.discover([self.root], include_corpus=True)
        self.assertNotIn(str(corpus), {str(path) for path, _ in found_default})
        self.assertIn(str(corpus), {str(path) for path, _ in found_include})


if __name__ == "__main__":
    unittest.main()
