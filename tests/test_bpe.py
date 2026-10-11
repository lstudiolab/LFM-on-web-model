import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_bpe import train_bpe


class TestBPETrainer(unittest.TestCase):
    def test_trains_repeated_pair(self):
        model = train_bpe("abababab", 258)
        self.assertEqual(model["format"], "sora-byte-bpe-v1")
        self.assertGreaterEqual(model["vocab_size"], 257)
        self.assertLessEqual(model["vocab_size"], 258)

    def test_minimum_vocab(self):
        with self.assertRaises(ValueError):
            train_bpe("hello", 255)

    def test_unicode_bytes_supported(self):
        model = train_bpe("Sora mahal kita. Sora mahal kita. 😊", 270)
        self.assertEqual(model["base_vocab_size"], 256)
        self.assertLessEqual(model["vocab_size"], 270)


if __name__ == "__main__":
    unittest.main()
