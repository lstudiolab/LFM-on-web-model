import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_bpe import train_bpe
from tokenizer import ByteBPETokenizer


class ByteBPETokenizerTests(unittest.TestCase):
    def test_unicode_round_trip_after_json_reload(self):
        text = "Sora can learn. Kumusta ka? 😊\nAng modelo ay natututo."
        learned = train_bpe((text + "\n") * 6, 320)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sora-tokenizer.json"
            path.write_text(json.dumps(learned, ensure_ascii=False), encoding="utf-8")
            tokenizer = ByteBPETokenizer.load(path)
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.vocab_size, learned["vocab_size"])


if __name__ == "__main__":
    unittest.main()
