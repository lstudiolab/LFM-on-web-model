import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_corpus import prepare
from sora_model import SoraRNN
from tokenize_corpus import tokenize_file
from tokenizer import ByteBPETokenizer
from train_bpe import train_bpe, train_bpe_lines


class TestPretrainingPipeline(unittest.TestCase):
    def test_bpe_round_trip_unicode_and_whitespace(self):
        text = "Sora mahal kita.\nSora belajar AI! 😊"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            vocab = root / "tokenizer.json"
            source.write_text((text + "\n") * 5, encoding="utf-8")
            vocab.write_text(json.dumps(train_bpe(source.read_text(encoding="utf-8"), 280)),
                             encoding="utf-8")
            tokenizer = ByteBPETokenizer.load(vocab)
            ids = tokenizer.encode(text)
            self.assertEqual(tokenizer.decode(ids), text)
            self.assertGreaterEqual(tokenizer.vocab_size, 256)

    def test_byte_cache_uses_one_byte_per_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            output = root / "tokens.bin"
            content = "Sora 😊".encode("utf-8")
            source.write_bytes(content)
            path, manifest = tokenize_file(source, output)
            self.assertEqual(manifest["dtype"], "uint8")
            self.assertEqual(manifest["token_count"], len(content))
            self.assertEqual(path.stat().st_size, len(content))
            values = np.fromfile(path, dtype=np.uint8)
            self.assertEqual(values.tolist(), list(content))

    def test_bpe_cache_is_compact_and_invalid_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            vocab = root / "tokenizer.json"
            output = root / "tokens.bin"
            source.write_text(("Sora belajar bahasa Indonesia. " * 30), encoding="utf-8")
            vocab.write_text(json.dumps(train_bpe(source.read_text(encoding="utf-8"), 300)),
                             encoding="utf-8")
            _, manifest = tokenize_file(source, output, vocab)
            self.assertEqual(manifest["dtype"], "uint16")
            output.write_bytes(b"corrupted")
            _, rebuilt = tokenize_file(source, output, vocab)
            self.assertEqual(rebuilt["cache_sha256"], __import__("hashlib").sha256(output.read_bytes()).hexdigest())
            self.assertEqual(output.stat().st_size, rebuilt["token_count"] * 2)


    def test_streaming_tokenization_matches_full_text_encoding(self):
        text = "Hello  \n  world!\nSora learns. \n\tIndented line 😊\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            vocab = root / "tokenizer.json"
            output = root / "tokens.bin"
            source.write_text(text, encoding="utf-8", newline="")
            with source.open("r", encoding="utf-8", newline="") as stream:
                trained = train_bpe_lines(stream, 300)
            vocab.write_text(json.dumps(trained), encoding="utf-8")
            tokenizer = ByteBPETokenizer.load(vocab)
            _, manifest = tokenize_file(source, output, vocab)
            actual = np.fromfile(output, dtype=np.dtype(manifest["dtype"])).astype(int).tolist()
            self.assertEqual(actual, tokenizer.encode(text))


    def test_cpp_tokenizer_matches_python_for_unicode_text(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is not installed")
        text = "Sora mahal kita. 😊"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / "tokenizer.json"
            executable = root / "sora-tokenizer"
            vocab.write_text(json.dumps(train_bpe((text + "\n") * 20, 300)),
                             encoding="utf-8")
            compile_result = subprocess.run(
                [compiler, "-std=c++17", "-O2",
                 str(ROOT / "cpp" / "main.cpp"),
                 str(ROOT / "cpp" / "tokenizer.cpp"),
                 "-o", str(executable)],
                capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run = subprocess.run(
                [str(executable), str(vocab), *text.split(" ")],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            match = re.search(r"IDs:(.*?)\nDecoded:", run.stdout, flags=re.DOTALL)
            self.assertIsNotNone(match, run.stdout)
            cpp_ids = [int(token) for token in match.group(1).split()]
            tokenizer = ByteBPETokenizer.load(vocab)
            self.assertEqual(cpp_ids, tokenizer.encode(text))
            self.assertIn(text, run.stdout)

    def test_document_split_is_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            docs = [f"Unique document number {i} with enough content for training." for i in range(12)]
            source.write_text("\n\n".join(docs), encoding="utf-8")
            a = prepare([source], root / "a", 0.25, 7, 1)
            b = prepare([source], root / "b", 0.25, 7, 1)
            train_a = (root / "a" / "train.txt").read_text(encoding="utf-8")
            val_a = (root / "a" / "validation.txt").read_text(encoding="utf-8")
            self.assertEqual(a["train_sha256"], b["train_sha256"])
            self.assertEqual(a["validation_sha256"], b["validation_sha256"])
            train_docs = set(train_a.strip().split("\n\n"))
            val_docs = set(val_a.strip().split("\n\n"))
            self.assertTrue(train_docs)
            self.assertTrue(val_docs)
            self.assertFalse(train_docs & val_docs)

    def test_end_to_end_short_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            valid = root / "validation.txt"
            tokenizer_path = root / "tokenizer.json"
            train.write_text(("Sora learns from training text. " * 8), encoding="utf-8")
            valid.write_text(("Sora is tested on validation text. " * 5), encoding="utf-8")
            tokenizer_path.write_text(json.dumps(train_bpe(train.read_text(encoding="utf-8"), 270)),
                                      encoding="utf-8")
            checkpoint = root / "run.npz"
            command = [
                sys.executable, str(ROOT / "python" / "train.py"), str(train),
                "--validation-corpus", str(valid), "--tokenizer", str(tokenizer_path),
                "--steps", "2", "--hidden-size", "8", "--seq-len", "8",
                "--eval-every", "1", "--generate-tokens", "4", "--top-k", "5",
                "--checkpoint", str(checkpoint), "--seed", "3",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(checkpoint.with_name("run.best.npz").exists())
            loaded = SoraRNN.load(checkpoint)
            self.assertEqual(loaded.step, 2)
            self.assertEqual(loaded.vocab_size, ByteBPETokenizer.load(tokenizer_path).vocab_size)
            self.assertTrue(checkpoint.with_suffix(".metrics.jsonl").exists())

            resume_command = command.copy()
            resume_command[resume_command.index("--steps") + 1] = "1"
            resume_command.append("--resume")
            resumed = subprocess.run(resume_command, capture_output=True, text=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + "\n" + resumed.stderr)
            self.assertEqual(SoraRNN.load(checkpoint).step, 3)

    def test_checked_in_smoke_corpus_can_pretrain(self):
        source = ROOT / "data" / "samples" / "pretrain-source"
        if not source.is_dir():
            self.skipTest("Checked-in smoke corpus is not present")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "ci-smoke"
            command = [
                sys.executable, str(ROOT / "python" / "pretrain.py"),
                "--input", str(source),
                "--run-dir", str(run_dir),
                "--steps", "12", "--hidden-size", "8", "--seq-len", "16",
                "--vocab-size", "320", "--validation-fraction", "0.2",
                "--min-document-chars", "1", "--eval-every", "6",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            self.assertTrue((run_dir / "corpus" / "manifest.json").is_file())
            self.assertTrue((run_dir / "tokenizer.json").is_file())
            self.assertTrue((run_dir / "run_config.json").is_file())
            self.assertTrue((run_dir / "sora-checkpoint.npz").is_file())
            self.assertTrue((run_dir / "sora-checkpoint.metrics.jsonl").is_file())

    def test_one_command_pretraining_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            documents = [
                "Document alpha teaches Sora to predict the next token from text.",
                "Document beta explains why validation examples must stay separate.",
                "Document gamma describes gradients, training loss, and checkpoints.",
                "Document delta has Filipino text: Ang modelo ay natututo sa teksto.",
                "Document epsilon tests punctuation, numbers 123, and Unicode 😊.",
                "Document zeta describes the tokenizer and memory separately.",
                "Document eta adds enough repeated patterns for byte pair merges.",
                "Document theta records that all model weights begin randomly.",
            ]
            (raw / "corpus.txt").write_text("\n\n".join(documents), encoding="utf-8")
            run_dir = root / "run"
            command = [
                sys.executable, str(ROOT / "python" / "pretrain.py"),
                "--input", str(raw), "--run-dir", str(run_dir),
                "--steps", "2", "--hidden-size", "8", "--seq-len", "8",
                "--vocab-size", "270", "--validation-fraction", "0.25",
                "--min-document-chars", "1", "--eval-every", "1",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            self.assertTrue((run_dir / "corpus" / "manifest.json").is_file())
            self.assertTrue((run_dir / "tokenizer.json").is_file())
            self.assertTrue((run_dir / "run_config.json").is_file())
            self.assertTrue((run_dir / "sora-checkpoint.npz").is_file())
            self.assertTrue((run_dir / "sora-checkpoint.best.npz").is_file())
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertGreater(config["parameters"], 0)



if __name__ == "__main__":
    unittest.main()
