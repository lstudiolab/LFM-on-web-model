import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sora_model import SoraRNN
from memory import MemoryStore

class TestSoraModel(unittest.TestCase):
    def test_loss_is_finite_and_training_updates(self):
        model = SoraRNN(hidden_size=8, seed=1)
        seq = np.frombuffer(b"hello hello hello", dtype=np.uint8).astype(np.int64)
        before = model.p["Wy"].copy()
        loss = model.train_sequence(seq)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, model.p["Wy"]))

    def test_checkpoint_round_trip(self):
        model = SoraRNN(hidden_size=8, seed=1)
        seq = np.frombuffer(b"hello", dtype=np.uint8).astype(np.int64)
        model.train_sequence(seq)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            model.save(path)
            loaded = SoraRNN.load(path)
            self.assertEqual(loaded.step, model.step)
            for key in model.p: np.testing.assert_allclose(loaded.p[key], model.p[key])

    def test_persistent_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "memory.json"
            store = MemoryStore(path)
            store.add("Sora is learning Filipino", "project")
            loaded = MemoryStore(path)
            self.assertEqual(len(loaded.search("Filipino")), 1)
            loaded.forget(0)
            self.assertEqual(loaded.data["memories"], [])

if __name__ == "__main__": unittest.main()
