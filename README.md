# Llab Foundation Model (LFM)

**LFM** means **Llab Foundation Model**. **Sora** is the experimental conversational AI being developed on top of this project.

This repository contains a small language model that starts from random weights. It is meant to make the whole training loop inspectable and repeatable, not to claim that a tiny model is already a capable assistant.

## What is implemented

- **Training from scratch:** a small recurrent neural network (RNN) predicts the next token. It does not load pretrained model weights.
- **Training math:** softmax cross-entropy, backpropagation through time, gradient clipping and Adam are implemented in \`python/sora_model.py\`. NumPy provides array/matrix operations; a prebuilt neural-network training framework is not used.
- **Two input representations:** raw UTF-8 bytes work out of the box. Optional byte-level BPE reduces sequence length for frequent text pieces.
- **Tokenizer:** \`python/train_bpe.py\` learns a JSON vocabulary, and \`python/tokenizer.py\` encodes/decodes with that same vocabulary. The C++17 utility uses the same JSON format for tokenizer checks.
- **Training pipeline:** \`python/train.py\` prepares a token cache, trains, measures validation loss, writes JSONL metrics, saves checkpoints, and can resume with the matching tokenizer.
- **Text generation:** \`python/generate.py\` generates continuations from a trained checkpoint.
- **Explicit memory store:** \`python/memory.py\` and \`python/memory_cli.py\` provide separate JSON memory commands. This symbolic memory is not the same thing as learning model weights.
- **Tests and CI:** unit tests plus a GitHub Actions smoke job exercise BPE round-tripping and a short end-to-end training run.

## Quick start: train Sora

Requirements: Python 3.10+ and NumPy. Run these commands from the repository root.

\`\`\`bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
\`\`\`

The repository includes small synthetic example files only to verify the pipeline. They are not a useful pretraining corpus.

### Option A: train with BPE

First, learn a tokenizer from the training split only:

\`\`\`bash
python python/train_bpe.py data/samples/train.txt artifacts/sora-tokenizer.json --vocab-size 320
\`\`\`

Then train a small model from random initialization:

\`\`\`bash
python python/train.py data/samples/train.txt \\
  --validation-corpus data/samples/validation.txt \\
  --tokenizer artifacts/sora-tokenizer.json \\
  --steps 300 --hidden-size 32 --seq-len 32 \\
  --checkpoint artifacts/sora-checkpoint.npz \\
  --eval-every 25 --eval-batches 4 \\
  --generate-tokens 60 --token-cache-dir artifacts/cache
\`\`\`

Generate a continuation from the resulting checkpoint:

\`\`\`bash
python python/generate.py artifacts/sora-checkpoint.npz \\
  --tokenizer artifacts/sora-tokenizer.json \\
  --prompt "Sora learns" --tokens 60 --temperature 0.8 --top-k 20
\`\`\`

### Option B: train directly on UTF-8 bytes

This needs no tokenizer file:

\`\`\`bash
python python/train.py data/samples/train.txt \\
  --validation-corpus data/samples/validation.txt \\
  --steps 300 --hidden-size 32 --seq-len 64 \\
  --checkpoint artifacts/sora-byte-checkpoint.npz \\
  --eval-every 25 --generate-tokens 60
\`\`\`

### Resume a run

Keep the same model checkpoint, training setup, and tokenizer file. For BPE, the tokenizer fingerprint is checked so token IDs cannot silently change:

\`\`\`bash
python python/train.py data/samples/train.txt \\
  --validation-corpus data/samples/validation.txt \\
  --tokenizer artifacts/sora-tokenizer.json \\
  --steps 300 --hidden-size 32 --seq-len 32 \\
  --checkpoint artifacts/sora-checkpoint.npz --resume
\`\`\`

When resuming, \`--steps\` is the number of additional optimizer updates. Checkpoint files include learned parameters and Adam state; the metrics log is written beside the checkpoint.

## Optional C++ tokenizer utility

The C++ utility checks the BPE JSON emitted by the Python trainer:

\`\`\`bash
c++ -std=c++17 -O2 cpp/main.cpp cpp/tokenizer.cpp -o sora-tokenizer
./sora-tokenizer artifacts/sora-tokenizer.json "Sora mahal kita 😊"
\`\`\`

It prints token IDs and the decoded text. Unicode word-boundary classification in the C++ demonstration is intentionally lightweight; use Python's tokenizer as the reference for training and evaluation.

## Use a real corpus

Replace the synthetic sample with UTF-8 text you have permission to use. Clean duplicates and unwanted personal data. Keep validation documents separate from training documents; do not tune against your final test set. Train the BPE vocabulary on the training split only. For larger corpora, use the corpus preparation/token caching tools in \`python/\` rather than repeatedly tokenizing raw text.

## What the model can and cannot do

This is an educational RNN baseline, not a Transformer or production foundation model. It has limited capacity and a short recurrent context; it will produce weak or incoherent text when trained only on the sample corpus. Falling training loss demonstrates optimization, not reasoning or robust language understanding. Meaningful conversational behavior requires substantially more high-quality data, compute, stronger architectures, thorough evaluation, and later work on dialogue data, safety, retrieval, and application integration.

Training changes the neural network weights. The explicit JSON memory store only saves selected facts and notes; it does not itself train or update those weights.
