"""A small byte/BPE-level recurrent language model implemented from scratch.

NumPy supplies array arithmetic only. The model, softmax loss, BPTT, clipping,
Adam updates and token sampling are implemented here. This is a research baseline,
not a production-scale foundation model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

BYTE_VOCAB_SIZE = 256
CHECKPOINT_FORMAT = "lfm-sora-rnn-v2"


class SoraRNN:
    def __init__(self, hidden_size: int = 128, seed: int = 7,
                 vocab_size: int = BYTE_VOCAB_SIZE):
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError("vocab_size must be at least 256")
        rng = np.random.default_rng(seed)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        h, v = self.hidden_size, self.vocab_size
        self.p = {
            "E": rng.normal(0, 0.03, (v, h)),
            "Wx": rng.normal(0, 1 / np.sqrt(h), (h, h)),
            "Wh": rng.normal(0, 1 / np.sqrt(h), (h, h)),
            "bh": np.zeros((1, h)),
            "Wy": rng.normal(0, 1 / np.sqrt(h), (h, v)),
            "by": np.zeros((1, v)),
        }
        self.m = {k: np.zeros_like(value) for k, value in self.p.items()}
        self.v = {k: np.zeros_like(value) for k, value in self.p.items()}
        self.step = 0

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.p.values())

    def _validate_ids(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if len(ids) < 2:
            raise ValueError("Need at least two tokens")
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size:
            raise ValueError(f"Token IDs must be in [0, {self.vocab_size - 1}]")
        return ids

    def _step_hidden(self, token_id: int, hidden: np.ndarray) -> np.ndarray:
        return np.tanh(
            self.p["E"][token_id:token_id + 1] @ self.p["Wx"]
            + hidden @ self.p["Wh"] + self.p["bh"]
        )

    def _logits(self, hidden: np.ndarray) -> np.ndarray:
        return hidden @ self.p["Wy"] + self.p["by"]

    @staticmethod
    def _cross_entropy(logits: np.ndarray, target: int) -> float:
        shifted = logits - np.max(logits)
        log_norm = np.log(np.exp(shifted).sum())
        return float(log_norm - shifted[0, target])

    def sequence_loss(self, ids: np.ndarray) -> float:
        """Evaluate next-token cross-entropy without constructing gradients."""
        ids = self._validate_ids(ids)
        hidden = np.zeros((1, self.hidden_size))
        total = 0.0
        for t in range(len(ids) - 1):
            hidden = self._step_hidden(int(ids[t]), hidden)
            total += self._cross_entropy(self._logits(hidden), int(ids[t + 1]))
        return total / (len(ids) - 1)

    def loss_and_gradients(self, ids: np.ndarray):
        ids = self._validate_ids(ids)
        x, y = ids[:-1], ids[1:]
        hs = [np.zeros((1, self.hidden_size))]
        probs = []
        loss = 0.0

        for t, token_id in enumerate(x):
            hidden = self._step_hidden(int(token_id), hs[-1])
            logits = self._logits(hidden)
            logits -= np.max(logits)
            probability = np.exp(logits)
            probability /= probability.sum()
            loss -= np.log(probability[0, int(y[t])] + 1e-12)
            hs.append(hidden)
            probs.append(probability)

        grads = {k: np.zeros_like(value) for k, value in self.p.items()}
        dh_next = np.zeros((1, self.hidden_size))
        for t in reversed(range(len(x))):
            dy = probs[t].copy()
            dy[0, int(y[t])] -= 1.0
            grads["Wy"] += hs[t + 1].T @ dy
            grads["by"] += dy
            dh = dy @ self.p["Wy"].T + dh_next
            da = dh * (1.0 - hs[t + 1] ** 2)
            grads["bh"] += da
            grads["Wx"] += self.p["E"][x[t]:x[t] + 1].T @ da
            grads["Wh"] += hs[t].T @ da
            grads["E"][x[t]] += (da @ self.p["Wx"].T).ravel()
            dh_next = da @ self.p["Wh"].T

        count = len(x)
        for key in grads:
            grads[key] /= count
        return loss / count, grads

    def train_sequence(self, ids, learning_rate: float = 0.001,
                       clip: float = 1.0) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if clip <= 0:
            raise ValueError("clip must be positive")
        loss, grads = self.loss_and_gradients(ids)
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):
            raise FloatingPointError("Non-finite loss or gradient; stopping training")
        if norm > clip:
            scale = clip / (norm + 1e-12)
            for grad in grads.values():
                grad *= scale

        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key in self.p:
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] * grads[key])
            m_hat = self.m[key] / (1 - beta1 ** self.step)
            v_hat = self.v[key] / (1 - beta2 ** self.step)
            self.p[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return float(loss)

    def generate_ids(self, prompt_ids: Iterable[int], max_new_tokens: int = 200,
                     temperature: float = 0.8, seed: int = 1,
                     top_k: int | None = None) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= self.vocab_size:
            raise ValueError("top_k must be between 1 and vocab_size")
        ids = [int(x) for x in prompt_ids]
        if not ids:
            ids = [32]
        if min(ids) < 0 or max(ids) >= self.vocab_size:
            raise ValueError("Prompt token ID outside model vocabulary")

        hidden = np.zeros((1, self.hidden_size))
        for token_id in ids:
            hidden = self._step_hidden(token_id, hidden)

        rng = np.random.default_rng(seed)
        generated = list(ids)
        for index in range(max_new_tokens):
            logits = self._logits(hidden).ravel() / temperature
            logits -= np.max(logits)
            if top_k is not None and top_k < self.vocab_size:
                keep = np.argpartition(logits, -top_k)[-top_k:]
                filtered = np.full_like(logits, -np.inf)
                filtered[keep] = logits[keep]
                logits = filtered
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            next_token = int(rng.choice(self.vocab_size, p=probabilities))
            generated.append(next_token)
            if index + 1 < max_new_tokens:
                hidden = self._step_hidden(next_token, hidden)
        return generated

    def generate(self, prompt: str, max_new_bytes: int = 200,
                 temperature: float = 0.8, seed: int = 1,
                 top_k: int | None = None) -> str:
        prompt_bytes = list(prompt.encode("utf-8"))
        generated = self.generate_ids(prompt_bytes, max_new_bytes, temperature, seed, top_k)
        return bytes(generated).decode("utf-8", errors="replace")

    def save(self, path, metadata=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"param_{k}": value for k, value in self.p.items()}
        payload.update({f"adam_m_{k}": value for k, value in self.m.items()})
        payload.update({f"adam_v_{k}": value for k, value in self.v.items()})
        payload["meta"] = np.asarray(json.dumps({
            "format": CHECKPOINT_FORMAT,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "step": self.step,
            "metadata": metadata or {},
        }))
        np.savez_compressed(destination, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            # Read old byte-only checkpoints as well.
            if meta.get("format") not in {"lfm-sora-rnn-v1", CHECKPOINT_FORMAT}:
                raise ValueError("Unsupported checkpoint format")
            model = cls(
                hidden_size=int(meta["hidden_size"]),
                vocab_size=int(meta.get("vocab_size", BYTE_VOCAB_SIZE)),
            )
            for key in model.p:
                for prefix, target in (("param_", model.p), ("adam_m_", model.m),
                                       ("adam_v_", model.v)):
                    name = f"{prefix}{key}"
                    if name not in data:
                        raise ValueError(f"Incomplete checkpoint: missing {name}")
                    value = data[name].copy()
                    if value.shape != target[key].shape:
                        raise ValueError(f"Invalid shape for checkpoint array {name}")
                    target[key] = value
            model.step = int(meta.get("step", 0))
            if model.step < 0:
                raise ValueError("Invalid checkpoint optimizer step")
            model.metadata = meta.get("metadata", {})
        return model


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
