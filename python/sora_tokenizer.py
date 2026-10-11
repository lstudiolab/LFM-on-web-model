"""From-scratch byte-level BPE tokenizer for Sora/LFM. Uses only the stdlib."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import re

MAGIC = "SORA_BPE_V1"
_SEGMENT_RE = re.compile(rb"[ \t\n\r\v\f]+|[^ \t\n\r\v\f]+")


def _segments(text: str) -> list[list[int]]:
    # ASCII-whitespace boundaries keep Python and C++ tokenization identical.
    raw = text.encode("utf-8")
    return [list(part) for part in _SEGMENT_RE.findall(raw)]


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int, int]]:
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 to represent every byte")
    segments = _segments(text)
    rules: list[tuple[int, int, int]] = []
    next_id = 256
    for _ in range(vocab_size - 256):
        counts: Counter[tuple[int, int]] = Counter()
        for seq in segments:
            counts.update(zip(seq, seq[1:]))
        if not counts:
            break
        best_count = max(counts.values())
        if best_count < 2:
            break
        pair = min(p for p, count in counts.items() if count == best_count)
        merged_id = next_id
        next_id += 1
        rules.append((pair[0], pair[1], merged_id))
        next_segments = []
        for seq in segments:
            merged: list[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    merged.append(merged_id)
                    i += 2
                else:
                    merged.append(seq[i])
                    i += 1
            next_segments.append(merged)
        segments = next_segments
    return rules


def save_bpe(path: str | Path, rules: list[tuple[int, int, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{MAGIC}\n{256 + len(rules)}\n{len(rules)}\n")
        for left, right, merged in rules:
            f.write(f"{left} {right} {merged}\n")


class BPETokenizer:
    def __init__(self, rules: list[tuple[int, int, int]]):
        self.rules = list(rules)
        self.eos_id = 256 + len(self.rules)
        self.vocab_size = self.eos_id + 1
        self._merge_rank = {(a, b): i for i, (a, b, _) in enumerate(self.rules)}
        self._pair_to_id = {(a, b): merged for a, b, merged in self.rules}
        self._expansion = {merged: (a, b) for a, b, merged in self.rules}
        expected = 256
        for a, b, merged in self.rules:
            if merged != expected or a >= merged or b >= merged or min(a, b) < 0:
                raise ValueError("Invalid BPE rules: IDs must be sequential and reference prior IDs")
            expected += 1

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            header = f.readline().strip()
            if header != MAGIC:
                raise ValueError(f"Unsupported tokenizer format: {header!r}")
            vocab_size = int(f.readline())
            merge_count = int(f.readline())
            rules = []
            for _ in range(merge_count):
                parts = f.readline().split()
                if len(parts) != 3:
                    raise ValueError("Malformed BPE merge row")
                rules.append(tuple(map(int, parts)))
        tok = cls(rules)
        if vocab_size != 256 + merge_count:
            raise ValueError("Vocabulary size does not match merge count")
        return tok

    def encode(self, text: str) -> list[int]:
        output: list[int] = []
        for seq in _segments(text):
            while len(seq) > 1:
                best_i, best_rank = -1, None
                for i, pair in enumerate(zip(seq, seq[1:])):
                    rank = self._merge_rank.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_i, best_rank = i, rank
                if best_i < 0:
                    break
                merged = self._pair_to_id[(seq[best_i], seq[best_i + 1])]
                seq[best_i:best_i + 2] = [merged]
            output.extend(seq)
        return output

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        def expand(token_id: int) -> None:
            if token_id == self.eos_id:
                return
            if 0 <= token_id < 256:
                raw.append(token_id)
                return
            pair = self._expansion.get(token_id)
            if pair is None:
                raise ValueError(f"Unknown token ID: {token_id}")
            expand(pair[0])
            expand(pair[1])
        for token_id in ids:
            expand(int(token_id))
        return raw.decode("utf-8", errors="replace")
