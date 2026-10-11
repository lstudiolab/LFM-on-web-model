#!/usr/bin/env python3
"""Generate text from a Sora RNN or causal Transformer checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np

from sora_model import SoraRNN, sha256_file
from sora_transformer import CHECKPOINT_FORMAT as TRANSFORMER_FORMAT, SoraTransformer
from tokenizer import ByteBPETokenizer


def checkpoint_format(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return str(json.loads(str(data["meta"].item())).get("format", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="required for BPE checkpoints")
    parser.add_argument("--prompt", default="Sora")
    parser.add_argument("--tokens", "--bytes", dest="tokens", type=int, default=200,
                        help="maximum number of new tokens (legacy --bytes alias retained)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("--tokens cannot be negative")

    try:
        is_transformer = checkpoint_format(args.checkpoint) == TRANSFORMER_FORMAT
        model = SoraTransformer.load(args.checkpoint) if is_transformer else SoraRNN.load(args.checkpoint)
        if model.vocab_size > 256:
            if not args.tokenizer:
                parser.error("This checkpoint uses BPE; pass --tokenizer tokenizer.json")
            tokenizer = ByteBPETokenizer.load(args.tokenizer)
            if tokenizer.vocab_size != model.vocab_size:
                parser.error("Tokenizer vocabulary does not match checkpoint")
            expected_hash = getattr(model, "metadata", {}).get("tokenizer_sha256")
            if expected_hash and expected_hash != sha256_file(args.tokenizer):
                parser.error("Tokenizer file does not match the checkpoint tokenizer fingerprint")
            prompt_ids = tokenizer.encode(args.prompt)
            output_ids = model.generate_ids(
                prompt_ids, args.tokens, args.temperature, args.seed, args.top_k
            )
            print(tokenizer.decode(output_ids))
        elif is_transformer:
            prompt_ids = list(args.prompt.encode("utf-8"))
            output_ids = model.generate_ids(
                prompt_ids, args.tokens, args.temperature, args.seed, args.top_k
            )
            print(bytes(output_ids).decode("utf-8", errors="replace"))
        else:
            print(model.generate(args.prompt, args.tokens, args.temperature, args.seed, args.top_k))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
