"""A tiny byte-level recurrent language model with manual backpropagation.

NumPy is used for matrix arithmetic only. The RNN, softmax, loss,
backpropagation, Adam updates, sampling, and checkpoint format are implemented here.
This is an educational baseline, not a production-scale LLM.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

VOCAB_SIZE = 256

class SoraRNN:
    def __init__(self, hidden_size: int = 128, seed: int = 7):
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        h = hidden_size
        self.p = {
            "E": rng.normal(0, 0.03, (VOCAB_SIZE, h)),
            "Wx": rng.normal(0, 1 / np.sqrt(h), (h, h)),
            "Wh": rng.normal(0, 1 / np.sqrt(h), (h, h)),
            "bh": np.zeros((1, h)),
            "Wy": rng.normal(0, 1 / np.sqrt(h), (h, VOCAB_SIZE)),
            "by": np.zeros((1, VOCAB_SIZE)),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    def loss_and_gradients(self, ids: np.ndarray):
        if len(ids) < 2:
            raise ValueError("Need at least two bytes to train")
        x, y = ids[:-1], ids[1:]
        xs, hs, ps = [], [np.zeros((1, self.hidden_size))], []
        loss = 0.0
        for t in range(len(x)):
            xt = self.p["E"][x[t]:x[t]+1]
            ht = np.tanh(xt @ self.p["Wx"] + hs[-1] @ self.p["Wh"] + self.p["bh"])
            logits = ht @ self.p["Wy"] + self.p["by"]
            logits -= logits.max()
            prob = np.exp(logits); prob /= prob.sum()
            loss -= np.log(prob[0, int(y[t])] + 1e-12)
            xs.append(int(x[t])); hs.append(ht); ps.append(prob)
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh_next = np.zeros((1, self.hidden_size))
        for t in reversed(range(len(x))):
            dy = ps[t].copy()
            dy[0, int(y[t])] -= 1.0
            grads["Wy"] += hs[t+1].T @ dy
            grads["by"] += dy
            dh = dy @ self.p["Wy"].T + dh_next
            da = dh * (1.0 - hs[t+1] ** 2)
            grads["bh"] += da
            grads["Wx"] += self.p["E"][xs[t]:xs[t]+1].T @ da
            grads["Wh"] += hs[t].T @ da
            grads["E"][xs[t]] += (da @ self.p["Wx"].T).ravel()
            dh_next = da @ self.p["Wh"].T
        n = len(x)
        for k in grads: grads[k] /= n
        return loss / n, grads

    def train_sequence(self, ids, learning_rate=0.001, clip=1.0):
        loss, grads = self.loss_and_gradients(np.asarray(ids, dtype=np.int64))
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        if norm > clip:
            for g in grads.values(): g *= clip / (norm + 1e-12)
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            self.m[k] = beta1*self.m[k] + (1-beta1)*grads[k]
            self.v[k] = beta2*self.v[k] + (1-beta2)*(grads[k]*grads[k])
            mh = self.m[k] / (1-beta1**self.step)
            vh = self.v[k] / (1-beta2**self.step)
            self.p[k] -= learning_rate * mh / (np.sqrt(vh) + eps)
        return float(loss)

    def generate(self, prompt: str, max_new_bytes=200, temperature=0.8, seed=1):
        if temperature <= 0: raise ValueError("temperature must be positive")
        rng = np.random.default_rng(seed)
        data = list(prompt.encode("utf-8")) or [32]
        h = np.zeros((1, self.hidden_size))
        for b in data:
            h = np.tanh(self.p["E"][b:b+1] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["bh"])
        out = bytearray(data)
        current = data[-1]
        for _ in range(max_new_bytes):
            h = np.tanh(self.p["E"][current:current+1] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["bh"])
            logits = (h @ self.p["Wy"] + self.p["by"]).ravel() / temperature
            logits -= logits.max()
            probs = np.exp(logits); probs /= probs.sum()
            current = int(rng.choice(VOCAB_SIZE, p=probs))
            out.append(current)
        return out.decode("utf-8", errors="replace")

    def save(self, path, metadata=None):
        payload = {f"param_{k}": v for k, v in self.p.items()}
        payload.update({f"adam_m_{k}": v for k, v in self.m.items()})
        payload.update({f"adam_v_{k}": v for k, v in self.v.items()})
        payload["meta"] = np.asarray(json.dumps({
            "format": "lfm-sora-rnn-v1", "hidden_size": self.hidden_size,
            "step": self.step, "metadata": metadata or {}
        }))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            if meta.get("format") != "lfm-sora-rnn-v1":
                raise ValueError("Unsupported checkpoint format")
            model = cls(meta["hidden_size"])
            for k in model.p:
                model.p[k] = data[f"param_{k}"].copy()
                model.m[k] = data[f"adam_m_{k}"].copy()
                model.v[k] = data[f"adam_v_{k}"].copy()
            model.step = int(meta["step"])
        return model
