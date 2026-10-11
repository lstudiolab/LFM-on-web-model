import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sora_tokenizer import BPETokenizer, save_bpe, train_bpe


class TokenizerTests(unittest.TestCase):
    def test_unicode_round_trip(self):
        text = "Sora can learn. Kumusta ka? 😊\nNew line!"
        tok = BPETokenizer(train_bpe((text + "\n") * 8, 300))
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_serialization_and_determinism(self):
        text = "alpha beta alpha beta alpha"
        rules = train_bpe((text + "\n") * 5, 280)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.bpe"
            save_bpe(path, rules)
            tok = BPETokenizer.load(path)
            self.assertEqual(tok.encode(text), BPETokenizer.load(path).encode(text))
            self.assertEqual(tok.decode(tok.encode(text)), text)
            self.assertEqual(tok.vocab_size, 257 + len(rules))

    def test_vocab_floor(self):
        with self.assertRaises(ValueError):
            train_bpe("abc", 255)


if __name__ == "__main__":
    unittest.main()
