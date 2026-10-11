"""A compact causal Transformer language model with manually implemented gradients.

Only NumPy tensor/array operations are used. Multi-head causal attention, residual
blocks, layer normalization, ReLU feed-forward layers, cross-entropy, backward
propagation, gradient clipping, and Adam are implemented in this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

BYTE_VOCAB_SIZE = 256
CHECKPOINT_FORMAT = "lfm-sora-transformer-v1"


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


class SoraTransformer:
    def __init__(self, vocab_size: int = BYTE_VOCAB_SIZE, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 128,
                 seed: int = 7):
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError("vocab_size must be at least 256")
        if d_model < 4:
            raise ValueError("d_model must be at least 4")
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model evenly")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = self.d_model // self.n_heads

        rng = np.random.default_rng(seed)
        d, v, t, f = self.d_model, self.vocab_size, self.max_seq_len, 4 * self.d_model

        def weight(shape, scale):
            return rng.normal(0.0, scale, shape).astype(np.float32)

        self.p = {
            # Tied input/output embedding weights.
            "E": weight((v, d), 0.02),
            "P": weight((t, d), 0.02),
            "by": np.zeros((v,), dtype=np.float32),
            "final_g": np.ones((d,), dtype=np.float32),
            "final_b": np.zeros((d,), dtype=np.float32),
        }
        for index in range(self.n_layers):
            prefix = f"layer{index}."
            self.p.update({
                prefix + "Wq": weight((d, d), 1.0 / np.sqrt(d)),
                prefix + "Wk": weight((d, d), 1.0 / np.sqrt(d)),
                prefix + "Wv": weight((d, d), 1.0 / np.sqrt(d)),
                prefix + "Wo": weight((d, d), 1.0 / np.sqrt(d)),
                prefix + "bq": np.zeros((d,), dtype=np.float32),
                prefix + "bk": np.zeros((d,), dtype=np.float32),
                prefix + "bv": np.zeros((d,), dtype=np.float32),
                prefix + "bo": np.zeros((d,), dtype=np.float32),
                prefix + "ln1_g": np.ones((d,), dtype=np.float32),
                prefix + "ln1_b": np.zeros((d,), dtype=np.float32),
                prefix + "ln2_g": np.ones((d,), dtype=np.float32),
                prefix + "ln2_b": np.zeros((d,), dtype=np.float32),
                prefix + "W1": weight((d, f), np.sqrt(2.0 / d)),
                prefix + "b1": np.zeros((f,), dtype=np.float32),
                prefix + "W2": weight((f, d), 1.0 / np.sqrt(f)),
                prefix + "b2": np.zeros((d,), dtype=np.float32),
            })
        self.m = {key: np.zeros_like(value) for key, value in self.p.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.p.items()}
        self.step = 0
        self.metadata = {}

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.p.values())

    def _validate_tokens(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if len(ids) < 1:
            raise ValueError("Need at least one token")
        if len(ids) > self.max_seq_len:
            raise ValueError(f"Context length {len(ids)} exceeds max_seq_len {self.max_seq_len}")
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size:
            raise ValueError(f"Token IDs must be in [0, {self.vocab_size - 1}]")
        return ids

    def _validate_sequence(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if len(ids) < 2:
            raise ValueError("Need at least two tokens for next-token training")
        if len(ids) - 1 > self.max_seq_len:
            raise ValueError(
                f"Training sequence has {len(ids) - 1} input tokens; max_seq_len is {self.max_seq_len}"
            )
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size:
            raise ValueError(f"Token IDs must be in [0, {self.vocab_size - 1}]")
        return ids

    def _layer_norm(self, x, gamma_key: str, beta_key: str):
        mean = np.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(variance + 1e-5)
        normalized = centered * inv_std
        result = normalized * self.p[gamma_key] + self.p[beta_key]
        return result, (normalized, inv_std, gamma_key, beta_key)

    def _layer_norm_backward(self, dout, cache, grads):
        normalized, inv_std, gamma_key, beta_key = cache
        grads[gamma_key] += np.sum(dout * normalized, axis=0)
        grads[beta_key] += np.sum(dout, axis=0)
        dxhat = dout * self.p[gamma_key]
        dimension = dxhat.shape[-1]
        sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * normalized, axis=-1, keepdims=True)
        return (inv_std / dimension) * (
            dimension * dxhat - sum_dxhat - normalized * sum_dxhat_xhat
        )

    def _forward(self, token_ids, keep_cache: bool = False):
        ids = self._validate_tokens(token_ids)
        length = len(ids)
        x = self.p["E"][ids] + self.p["P"][:length]
        layer_caches = [] if keep_cache else None
        causal_mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        scale = 1.0 / np.sqrt(self.head_dim)

        for index in range(self.n_layers):
            prefix = f"layer{index}."
            residual_in = x
            norm1, norm1_cache = self._layer_norm(
                residual_in, prefix + "ln1_g", prefix + "ln1_b"
            )

            q = norm1 @ self.p[prefix + "Wq"] + self.p[prefix + "bq"]
            k = norm1 @ self.p[prefix + "Wk"] + self.p[prefix + "bk"]
            value = norm1 @ self.p[prefix + "Wv"] + self.p[prefix + "bv"]

            q_heads = q.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k_heads = k.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            v_heads = value.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)

            scores = (q_heads @ k_heads.transpose(0, 2, 1)) * scale
            scores = np.where(causal_mask[None, :, :], -1e9, scores)
            attention = _softmax(scores, axis=-1)
            context_heads = attention @ v_heads
            context = context_heads.transpose(1, 0, 2).reshape(length, self.d_model)
            attention_out = context @ self.p[prefix + "Wo"] + self.p[prefix + "bo"]
            residual_mid = residual_in + attention_out

            norm2, norm2_cache = self._layer_norm(
                residual_mid, prefix + "ln2_g", prefix + "ln2_b"
            )
            pre_activation = norm2 @ self.p[prefix + "W1"] + self.p[prefix + "b1"]
            activation = np.maximum(pre_activation, 0.0)
            feedforward = activation @ self.p[prefix + "W2"] + self.p[prefix + "b2"]
            x = residual_mid + feedforward

            if keep_cache:
                layer_caches.append({
                    "prefix": prefix,
                    "residual_in": residual_in,
                    "norm1": norm1,
                    "norm1_cache": norm1_cache,
                    "q_heads": q_heads,
                    "k_heads": k_heads,
                    "v_heads": v_heads,
                    "attention": attention,
                    "context": context,
                    "context_heads": context_heads,
                    "norm2": norm2,
                    "norm2_cache": norm2_cache,
                    "pre_activation": pre_activation,
                    "activation": activation,
                    "residual_mid": residual_mid,
                })

        final_norm, final_norm_cache = self._layer_norm(x, "final_g", "final_b")
        logits = final_norm @ self.p["E"].T + self.p["by"]
        cache = None
        if keep_cache:
            cache = {
                "ids": ids,
                "layer_caches": layer_caches,
                "final_norm": final_norm,
                "final_norm_cache": final_norm_cache,
            }
        return logits, cache

    def sequence_loss(self, token_ids) -> float:
        ids = self._validate_sequence(token_ids)
        inputs, targets = ids[:-1], ids[1:]
        logits, _ = self._forward(inputs, keep_cache=False)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=-1))
        chosen = shifted[np.arange(len(targets)), targets]
        return float(np.mean(log_normalizer - chosen))

    def loss_and_gradients(self, token_ids):
        ids = self._validate_sequence(token_ids)
        inputs, targets = ids[:-1], ids[1:]
        logits, cache = self._forward(inputs, keep_cache=True)
        length = len(targets)
        probs = _softmax(logits, axis=-1)
        loss = -np.log(probs[np.arange(length), targets] + 1e-12).mean()
        dlogits = probs
        dlogits[np.arange(length), targets] -= 1.0
        dlogits /= length

        grads = {key: np.zeros_like(value) for key, value in self.p.items()}
        final_norm = cache["final_norm"]
        grads["by"] += dlogits.sum(axis=0)
        # E is shared between the input embedding and the output projection.
        grads["E"] += dlogits.T @ final_norm
        dx = dlogits @ self.p["E"]
        dx = self._layer_norm_backward(dx, cache["final_norm_cache"], grads)

        scale = 1.0 / np.sqrt(self.head_dim)
        for layer_cache in reversed(cache["layer_caches"]):
            prefix = layer_cache["prefix"]
            # Residual path around the feed-forward sublayer.
            d_residual_mid = dx.copy()
            d_feedforward = dx

            grads[prefix + "W2"] += layer_cache["activation"].T @ d_feedforward
            grads[prefix + "b2"] += d_feedforward.sum(axis=0)
            d_activation = d_feedforward @ self.p[prefix + "W2"].T
            d_pre_activation = d_activation * (layer_cache["pre_activation"] > 0.0)
            grads[prefix + "W1"] += layer_cache["norm2"].T @ d_pre_activation
            grads[prefix + "b1"] += d_pre_activation.sum(axis=0)
            d_norm2 = d_pre_activation @ self.p[prefix + "W1"].T
            d_residual_mid += self._layer_norm_backward(
                d_norm2, layer_cache["norm2_cache"], grads
            )

            # Residual path around causal self-attention.
            d_attention_out = d_residual_mid
            d_input_residual = d_residual_mid.copy()
            grads[prefix + "Wo"] += layer_cache["context"].T @ d_attention_out
            grads[prefix + "bo"] += d_attention_out.sum(axis=0)
            d_context = d_attention_out @ self.p[prefix + "Wo"].T
            d_context_heads = d_context.reshape(
                len(inputs), self.n_heads, self.head_dim
            ).transpose(1, 0, 2)

            attention = layer_cache["attention"]
            q_heads = layer_cache["q_heads"]
            k_heads = layer_cache["k_heads"]
            v_heads = layer_cache["v_heads"]
            d_attention = d_context_heads @ v_heads.transpose(0, 2, 1)
            d_v_heads = attention.transpose(0, 2, 1) @ d_context_heads
            d_scores = attention * (
                d_attention - np.sum(d_attention * attention, axis=-1, keepdims=True)
            )
            d_q_heads = (d_scores @ k_heads) * scale
            d_k_heads = (d_scores.transpose(0, 2, 1) @ q_heads) * scale

            d_q = d_q_heads.transpose(1, 0, 2).reshape(len(inputs), self.d_model)
            d_k = d_k_heads.transpose(1, 0, 2).reshape(len(inputs), self.d_model)
            d_value = d_v_heads.transpose(1, 0, 2).reshape(len(inputs), self.d_model)
            grads[prefix + "Wq"] += layer_cache["norm1"].T @ d_q
            grads[prefix + "bq"] += d_q.sum(axis=0)
            grads[prefix + "Wk"] += layer_cache["norm1"].T @ d_k
            grads[prefix + "bk"] += d_k.sum(axis=0)
            grads[prefix + "Wv"] += layer_cache["norm1"].T @ d_value
            grads[prefix + "bv"] += d_value.sum(axis=0)
            d_norm1 = (
                d_q @ self.p[prefix + "Wq"].T
                + d_k @ self.p[prefix + "Wk"].T
                + d_value @ self.p[prefix + "Wv"].T
            )
            dx = d_input_residual + self._layer_norm_backward(
                d_norm1, layer_cache["norm1_cache"], grads
            )

        # Input embedding and position embedding gradients.
        grads["P"][:len(inputs)] += dx
        for position, token_id in enumerate(inputs):
            grads["E"][int(token_id)] += dx[position]
        return float(loss), grads

    def train_sequence(self, token_ids, learning_rate: float = 0.001,
                       clip: float = 1.0) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if clip <= 0:
            raise ValueError("clip must be positive")
        loss, grads = self.loss_and_gradients(token_ids)
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
        generated = [int(token_id) for token_id in prompt_ids]
        if not generated:
            generated = [32]
        if min(generated) < 0 or max(generated) >= self.vocab_size:
            raise ValueError("Prompt token ID outside model vocabulary")

        rng = np.random.default_rng(seed)
        for _ in range(max_new_tokens):
            context = generated[-self.max_seq_len:]
            logits, _ = self._forward(context, keep_cache=False)
            next_logits = logits[-1].copy() / temperature
            next_logits -= np.max(next_logits)
            if top_k is not None and top_k < self.vocab_size:
                candidates = np.argpartition(next_logits, -top_k)[-top_k:]
                filtered = np.full_like(next_logits, -np.inf)
                filtered[candidates] = next_logits[candidates]
                next_logits = filtered
            probabilities = np.exp(next_logits)
            probabilities /= probabilities.sum()
            generated.append(int(rng.choice(self.vocab_size, p=probabilities)))
        return generated

    def save(self, path, metadata=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp.npz")
        payload = {f"param_{key}": value for key, value in self.p.items()}
        payload.update({f"adam_m_{key}": value for key, value in self.m.items()})
        payload.update({f"adam_v_{key}": value for key, value in self.v.items()})
        payload["meta"] = np.asarray(json.dumps({
            "format": CHECKPOINT_FORMAT,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "step": self.step,
            "metadata": metadata or {},
        }))
        try:
            np.savez_compressed(temporary, **payload)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            if meta.get("format") != CHECKPOINT_FORMAT:
                raise ValueError("Unsupported Transformer checkpoint format")
            model = cls(
                vocab_size=int(meta["vocab_size"]),
                d_model=int(meta["d_model"]),
                n_heads=int(meta["n_heads"]),
                n_layers=int(meta["n_layers"]),
                max_seq_len=int(meta["max_seq_len"]),
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
                    target[key] = value.astype(np.float32, copy=False)
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
