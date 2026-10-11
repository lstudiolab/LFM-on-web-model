"""Deterministic byte-level BPE tokenizer compatible with python/train_bpe.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from train_bpe import BASE_VOCAB_SIZE, BPE_FORMAT, split_pieces


class ByteBPETokenizer:
    def __init__(self, merges, vocab_size=None):
        self.merges = [(int(a), int(b), int(c)) for a, b, c in merges]
        expected = BASE_VOCAB_SIZE
        expansions = {}
        for rank, (left, right, merged) in enumerate(self.merges):
            if merged != expected:
                raise ValueError(
                    f"Non-contiguous BPE ID at merge {rank}: expected {expected}, got {merged}"
                )
            if left >= merged or right >= merged or left < 0 or right < 0:
                raise ValueError(f"Invalid merge operands in merge {rank}")
            expansions[merged] = (left, right)
            expected += 1
        self.vocab_size = expected
        if vocab_size is not None and int(vocab_size) != self.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size mismatch: file says {vocab_size}, merges imply {self.vocab_size}"
            )
        self.expansions = expansions
        self._piece_cache = {}

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != BPE_FORMAT:
            raise ValueError(f"Unsupported tokenizer format: {payload.get('format')!r}")
        if int(payload.get("base_vocab_size", -1)) != BASE_VOCAB_SIZE:
            raise ValueError("Tokenizer base vocabulary must contain all 256 byte values")
        return cls(payload.get("merges", []), payload.get("vocab_size"))

    def to_dict(self):
        return {
            "format": BPE_FORMAT,
            "base_vocab_size": BASE_VOCAB_SIZE,
            "vocab_size": self.vocab_size,
            "merges": [list(item) for item in self.merges],
            "notes": "Byte-level BPE; merges are applied independently to Unicode word/space/punctuation pieces.",
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    def _encode_piece(self, raw: bytes) -> tuple[int, ...]:
        cached = self._piece_cache.get(raw)
        if cached is not None:
            return cached
        ids = list(raw)
        for left, right, merged in self.merges:
            if len(ids) < 2:
                break
            result = []
            index = 0
            while index < len(ids):
                if index + 1 < len(ids) and ids[index] == left and ids[index + 1] == right:
                    result.append(merged)
                    index += 2
                else:
                    result.append(ids[index])
                    index += 1
            ids = result
        encoded = tuple(ids)
        if len(self._piece_cache) < 100_000:
            self._piece_cache[raw] = encoded
        return encoded

    def encode(self, text: str) -> list[int]:
        tokens = []
        for piece in split_pieces(text):
            tokens.extend(self._encode_piece(piece.encode("utf-8")))
        return tokens

    def decode(self, ids, errors="replace") -> str:
        raw = bytearray()
        for token_id in ids:
            stack = [int(token_id)]
            while stack:
                current = stack.pop()
                if 0 <= current < BASE_VOCAB_SIZE:
                    raw.append(current)
                    continue
                pair = self.expansions.get(current)
                if pair is None:
                    raise ValueError(f"Unknown token ID: {current}")
                stack.append(pair[1])
                stack.append(pair[0])
        return raw.decode("utf-8", errors=errors)

    @property
    def fingerprint(self) -> str:
        packed = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(packed).hexdigest()
