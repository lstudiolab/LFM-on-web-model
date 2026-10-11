#!/usr/bin/env python3
"""Generate text from a Sora PyTorch checkpoint trained from scratch."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F

from sora_torch_transformer import SoraTorchTransformer, load_checkpoint
from sora_transformer import sha256_file
from tokenizer import ByteBPETokenizer
from train_torch_transformer import choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="required for BPE checkpoints")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 uses the full vocabulary")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0 or args.top_k < 0:
        parser.error("tokens/top-k must be non-negative and temperature must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.tokenizer and not args.tokenizer.is_file():
        parser.error(f"Tokenizer not found: {args.tokenizer}")
    try:
        device = choose_device(args.device)
        payload = load_checkpoint(args.checkpoint, map_location=device)
        model = SoraTorchTransformer(**payload["model_config"]).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        metadata = payload.get("metadata", {})
        tokenizer_hash = sha256_file(args.tokenizer) if args.tokenizer else None
        if tokenizer_hash != metadata.get("tokenizer_sha256"):
            parser.error("Tokenizer fingerprint does not match this checkpoint")
        tokenizer = ByteBPETokenizer.load(args.tokenizer) if args.tokenizer else None
        vocab_size = tokenizer.vocab_size if tokenizer else 256
        if vocab_size != model.vocab_size:
            parser.error("Tokenizer vocabulary does not match checkpoint")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    ids = tokenizer.encode(args.prompt) if tokenizer else list(args.prompt.encode("utf-8"))
    if not ids:
        parser.error("Prompt encoded to an empty sequence")
    with torch.inference_mode():
        for _ in range(args.tokens):
            context = torch.tensor(ids[-model.max_seq_len:], dtype=torch.long, device=device)[None, :]
            logits = model(context)[0, -1, :] / args.temperature
            if args.top_k and args.top_k < logits.numel():
                values, indices = torch.topk(logits, args.top_k)
                probabilities = F.softmax(values, dim=-1)
                selected_index = int(torch.multinomial(probabilities, 1).item())
                next_token = int(indices[selected_index].item())
            else:
                probabilities = F.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probabilities, 1).item())
            ids.append(next_token)
    if tokenizer:
        print(tokenizer.decode(ids))
    else:
        print(bytes(value for value in ids if 0 <= value <= 255).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
