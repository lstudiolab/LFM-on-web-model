#!/usr/bin/env python3
"""Train Sora's from-scratch byte-level RNN on a UTF-8 text file."""
import argparse
from pathlib import Path
import numpy as np
from sora_model import SoraRNN

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path, help="UTF-8 plain-text training corpus")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--checkpoint", type=Path, default=Path("sora-checkpoint.npz"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="Sora")
    ap.add_argument("--generate-bytes", type=int, default=200)
    args = ap.parse_args()
    if args.steps < 1 or args.seq_len < 2: ap.error("--steps must be positive and --seq-len at least 2")
    raw = args.corpus.read_bytes()
    if len(raw) < 4: ap.error("Corpus is too small; provide a larger UTF-8 text file")
    ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    split = max(2, int(len(ids)*0.9))
    train, valid = ids[:split], ids[split:]
    if len(valid) < 2: valid = train[-max(2, len(train)//10):]
    model = SoraRNN.load(args.checkpoint) if args.resume else SoraRNN(args.hidden_size, args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"Corpus bytes: {len(ids)} | train: {len(train)} | validation: {len(valid)}")
    print(f"Model: byte-level RNN | hidden size: {model.hidden_size} | parameters: {sum(v.size for v in model.p.values()):,}")
    for step in range(1, args.steps+1):
        start = int(rng.integers(0, max(1, len(train)-args.seq_len-1)))
        seq = train[start:start+args.seq_len+1]
        loss = model.train_sequence(seq, args.learning_rate)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            eval_seq = valid[:min(len(valid), args.seq_len+1)]
            val_loss, _ = model.loss_and_gradients(eval_seq)
            print(f"step={model.step} train_loss={loss:.4f} validation_loss={val_loss:.4f}")
            model.save(args.checkpoint, {"corpus": args.corpus.name})
    model.save(args.checkpoint, {"corpus": args.corpus.name})
    print(f"Checkpoint saved: {args.checkpoint}")
    print("--- Sample (not necessarily coherent early in training) ---")
    print(model.generate(args.prompt, args.generate_bytes))

if __name__ == "__main__":
    main()
