#!/usr/bin/env python3
"""Generate text from a Sora checkpoint."""
import argparse
from pathlib import Path

from sora_model import SoraRNN, sha256_file
from tokenizer import ByteBPETokenizer


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
        model = SoraRNN.load(args.checkpoint)
        if model.vocab_size > 256:
            if not args.tokenizer:
                parser.error("This checkpoint uses BPE; pass --tokenizer tokenizer.json")
            tokenizer = ByteBPETokenizer.load(args.tokenizer)
            if tokenizer.vocab_size != model.vocab_size:
                parser.error("Tokenizer vocabulary does not match checkpoint")
            expected_hash = getattr(model, "metadata", {}).get("tokenizer_sha256")
            if expected_hash and expected_hash != sha256_file(args.tokenizer):
                parser.error("Tokenizer file does not match the checkpoint tokenizer fingerprint")
            ids = tokenizer.encode(args.prompt)
            output = model.generate_ids(ids, args.tokens, args.temperature, args.seed, args.top_k)
            print(tokenizer.decode(output))
        else:
            print(model.generate(args.prompt, args.tokens, args.temperature, args.seed, args.top_k))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
