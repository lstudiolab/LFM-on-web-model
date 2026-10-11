import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from evaluate_torch import evaluate_torch_checkpoint
from sora_torch_transformer import SoraTorchTransformer, load_checkpoint, save_checkpoint
from sora_transformer import sha256_file


class TestTorchTraining(unittest.TestCase):
    def test_forward_shape_and_parameter_initialization(self):
        torch.manual_seed(11)
        model = SoraTorchTransformer(vocab_size=256, d_model=8, n_heads=2,
                                     n_layers=1, max_seq_len=8, dropout=0.0)
        logits = model(torch.tensor([[65, 66, 67]], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 3, 256))
        self.assertGreater(model.parameter_count, 0)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_train_checkpoint_evaluation_and_initialize_from(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            valid = root / "validation.txt"
            train.write_text("hello world\nhello Sora\nKumusta ka\n", encoding="utf-8")
            valid.write_text("separate validation material\n", encoding="utf-8")
            first = root / "first.pt"
            base = SoraTorchTransformer(vocab_size=256, d_model=8, n_heads=2,
                                        n_layers=1, max_seq_len=8, dropout=0.0)
            save_checkpoint(first, base, None, {"tokenizer_sha256": None})
            latest = root / "trained.pt"
            command = [
                sys.executable, str(ROOT / "python" / "train_torch_transformer.py"),
                str(train), "--validation-corpus", str(valid),
                "--checkpoint", str(latest), "--initialize-from", str(first),
                "--steps", "1", "--batch-size", "2", "--d-model", "8",
                "--heads", "2", "--layers", "1", "--seq-len", "8",
                "--dropout", "0", "--learning-rate", "0.001",
                "--warmup-steps", "0", "--eval-every", "1",
                "--eval-batches", "1", "--device", "cpu"
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            payload = load_checkpoint(latest, map_location="cpu")
            self.assertEqual(payload["metadata"]["global_step"], 1)
            self.assertEqual(payload["metadata"]["initial_checkpoint_sha256"], sha256_file(first))
            self.assertTrue((root / "trained.best.pt").is_file())

            report = evaluate_torch_checkpoint(latest, valid, device_name="cpu")
            self.assertEqual(report["evaluated_targets"], report["test_tokens"] - 1)
            self.assertTrue(report["perplexity"] >= 1.0)

    def test_generation_cli_loads_torch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            model = SoraTorchTransformer(vocab_size=256, d_model=8, n_heads=2,
                                         n_layers=1, max_seq_len=8, dropout=0.0)
            save_checkpoint(checkpoint, model, None, {"tokenizer_sha256": None})
            command = [
                sys.executable, str(ROOT / "python" / "generate_torch.py"),
                str(checkpoint), "--prompt", "Hello", "--tokens", "1",
                "--top-k", "1", "--device", "cpu"
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)


if __name__ == "__main__":
    unittest.main()
