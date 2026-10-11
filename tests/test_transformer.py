import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sora_transformer import SoraTransformer
from evaluate import evaluate_checkpoint
from tokenizer import ByteBPETokenizer
from train_bpe import train_bpe


class TestSoraTransformer(unittest.TestCase):
    def test_attention_gradient_matches_finite_difference(self):
        model = SoraTransformer(
            vocab_size=270, d_model=4, n_heads=2, n_layers=1,
            max_seq_len=4, seed=17,
        )
        ids = [1, 2, 3, 4, 5]
        loss, gradients = model.loss_and_gradients(ids)
        self.assertTrue(np.isfinite(loss))
        analytic_matrix = gradients["layer0.Wq"]
        index = np.unravel_index(np.argmax(np.abs(analytic_matrix)), analytic_matrix.shape)
        analytic = float(analytic_matrix[index])
        original = float(model.p["layer0.Wq"][index])
        epsilon = 0.01
        model.p["layer0.Wq"][index] = original + epsilon
        plus = model.sequence_loss(ids)
        model.p["layer0.Wq"][index] = original - epsilon
        minus = model.sequence_loss(ids)
        model.p["layer0.Wq"][index] = original
        numeric = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(
            analytic, numeric,
            delta=max(0.003, abs(analytic) * 0.2),
            msg=f"manual attention gradient {analytic} versus finite difference {numeric}",
        )

    def test_transformer_train_checkpoint_and_generation(self):
        model = SoraTransformer(
            vocab_size=270, d_model=8, n_heads=2, n_layers=1,
            max_seq_len=8, seed=5,
        )
        ids = [1, 2, 3, 4, 5, 6]
        original = {key: value.copy() for key, value in model.p.items()}
        loss = model.train_sequence(ids, learning_rate=0.005)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(model.step, 1)
        self.assertTrue(any(not np.array_equal(original[key], model.p[key]) for key in model.p))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "transformer.npz"
            model.save(checkpoint, {"training_corpus_sha256": "test"})
            loaded = SoraTransformer.load(checkpoint)
            self.assertEqual(loaded.step, model.step)
            self.assertEqual(loaded.parameter_count, model.parameter_count)
            for key in model.p:
                np.testing.assert_array_equal(model.p[key], loaded.p[key])
                np.testing.assert_array_equal(model.m[key], loaded.m[key])
                np.testing.assert_array_equal(model.v[key], loaded.v[key])
            self.assertEqual(loaded.metadata["training_corpus_sha256"], "test")
            generated = loaded.generate_ids([1, 2], max_new_tokens=3, seed=9, top_k=5)
            self.assertEqual(len(generated), 5)

    def test_held_out_evaluation_uses_checkpoint_tokenizer_and_all_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            valid = root / "validation.txt"
            test = root / "held-out-test.txt"
            tokenizer_file = root / "tokenizer.json"
            checkpoint = root / "model.npz"
            train.write_text(("Sora learns from examples. " * 10), encoding="utf-8")
            valid.write_text(("Validation is not training. " * 8), encoding="utf-8")
            test.write_text(("This test corpus is never used for training. 😊\n" * 7), encoding="utf-8")
            tokenizer_file.write_text(
                json.dumps(train_bpe(train.read_text(encoding="utf-8"), 280)),
                encoding="utf-8",
            )
            command = [
                sys.executable, str(ROOT / "python" / "train_transformer.py"),
                str(train), "--validation-corpus", str(valid),
                "--tokenizer", str(tokenizer_file), "--steps", "2",
                "--d-model", "8", "--heads", "2", "--layers", "1",
                "--seq-len", "8", "--eval-every", "1", "--generate-tokens", "0",
                "--checkpoint", str(checkpoint),
            ]
            trained = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(trained.returncode, 0, trained.stdout + "\n" + trained.stderr)

            report = evaluate_checkpoint(checkpoint, test, tokenizer_file)
            self.assertEqual(report["test_tokens"] - 1, report["evaluated_targets"])
            self.assertTrue(np.isfinite(report["test_loss"]))
            self.assertGreaterEqual(report["perplexity"], 1.0)
            self.assertEqual(
                report["test_corpus_sha256"],
                __import__("hashlib").sha256(test.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "Tokenizer fingerprint"):
                evaluate_checkpoint(checkpoint, test, None)

    def test_transformer_trainer_and_resume_match_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            valid = root / "validation.txt"
            tokenizer_file = root / "tokenizer.json"
            train.write_text(("Sora learns language patterns from examples. " * 8), encoding="utf-8")
            valid.write_text(("Validation text is held out from training. " * 6), encoding="utf-8")
            tokenizer_file.write_text(
                json.dumps(train_bpe(train.read_text(encoding="utf-8"), 270)),
                encoding="utf-8",
            )
            common = [
                sys.executable, str(ROOT / "python" / "train_transformer.py"),
                str(train), "--validation-corpus", str(valid),
                "--tokenizer", str(tokenizer_file), "--d-model", "8",
                "--heads", "2", "--layers", "1", "--seq-len", "8",
                "--learning-rate", "0.001", "--eval-every", "1",
                "--generate-tokens", "0", "--top-k", "5", "--seed", "13",
            ]

            continuous_path = root / "continuous.npz"
            continuous = common + ["--steps", "4", "--checkpoint", str(continuous_path)]
            result = subprocess.run(continuous, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

            resumed_path = root / "resumed.npz"
            split = common + ["--steps", "2", "--checkpoint", str(resumed_path)]
            first_half = subprocess.run(split, capture_output=True, text=True, timeout=60)
            self.assertEqual(first_half.returncode, 0, first_half.stdout + "\n" + first_half.stderr)
            split[split.index("--steps") + 1] = "2"
            split.append("--resume")
            second_half = subprocess.run(split, capture_output=True, text=True, timeout=60)
            self.assertEqual(second_half.returncode, 0, second_half.stdout + "\n" + second_half.stderr)

            expected = SoraTransformer.load(continuous_path)
            actual = SoraTransformer.load(resumed_path)
            self.assertEqual(expected.step, actual.step)
            for key in expected.p:
                np.testing.assert_array_equal(expected.p[key], actual.p[key], err_msg=key)
                np.testing.assert_array_equal(expected.m[key], actual.m[key], err_msg=f"Adam m: {key}")
                np.testing.assert_array_equal(expected.v[key], actual.v[key], err_msg=f"Adam v: {key}")


if __name__ == "__main__":
    unittest.main()
