"""PyTorch causal Transformer for scalable training from random initialization."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
        )
        x = x + self.residual_dropout(attended)
        x = x + self.feed_forward(self.norm2(x))
        return x


class SoraTorchTransformer(nn.Module):
    """Decoder-only language model. All parameters are randomly initialized."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if d_model < 8:
            raise ValueError("d_model must be at least 8")
        if n_heads < 1 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.dropout_rate = float(dropout)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, max_seq_len, d_model))
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length < 1 or length > self.max_seq_len:
            raise ValueError(f"Sequence length must be in [1, {self.max_seq_len}]")
        if input_ids.numel() and (
            input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size
        ):
            raise ValueError(f"Token IDs must be in [0, {self.vocab_size - 1}]")
        x = self.token_embedding(input_ids) + self.position_embedding[:, :length]
        x = self.embedding_dropout(x)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.final_norm(x)
        # Weight tying helps keep the output vocabulary connected to token embeddings.
        return F.linear(x, self.token_embedding.weight, self.output_bias)

    def config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_rate,
        }



TORCH_CHECKPOINT_FORMAT = "lfm-sora-torch-v1"


def save_checkpoint(path, model, optimizer, metadata: dict, training_state: dict | None = None):
    """Persist weights, optimizer state, configuration, and provenance metadata."""
    payload = {
        "format": TORCH_CHECKPOINT_FORMAT,
        "model_config": model.config(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata,
        "training_state": training_state or {},
    }
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint written by this project; never loads pretrained weights implicitly."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != TORCH_CHECKPOINT_FORMAT:
        raise ValueError("Unsupported Sora PyTorch checkpoint format")
    if not isinstance(payload.get("model_config"), dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError("Checkpoint is missing model configuration or weights")
    return payload
