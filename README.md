# Llab Foundation Model (LFM)

**LFM (Llab Foundation Model)** is the foundation-model research project behind **Sora**. The model starts from random weights: it does not load an existing pretrained model.

> **Current status:** the repository now includes a reproducible data/tokenizer/training pipeline and tests. The network is still a small, manually implemented RNN baseline—not a modern Transformer or production-ready LLM. No full pretraining run is claimed.

## Current pipeline

1. **Corpus preparation** — reads UTF-8 text, normalizes Unicode to NFC, removes exact duplicate paragraphs/documents, creates a deterministic train/validation split at blank-line document boundaries, and writes hashes/statistics in a manifest.
2. **Tokenizer training** — learns byte-level BPE merges from the training split only. Validation text is not used to learn tokenizer merges.
3. **Tokenization/cache** — applies the same tokenizer in training and generation, stores compact byte/uint16/uint32 token arrays on disk, and verifies cache metadata and hashes.
4. **Neural network** — randomly initialized embedding + recurrent tanh layer + softmax output. The forward pass, cross-entropy, backpropagation through time, gradient clipping, and Adam update are implemented in Python/NumPy. NumPy provides numerical array operations; no pretrained weights or neural-network training framework are used.
5. **Training checks** — periodic train/validation loss, JSONL metric log, last checkpoint, best-validation checkpoint, saved Adam optimizer state, and deterministic random seeds.
6. **Generation** — loads the checkpoint and matching tokenizer and samples new tokens.

The separate python/memory.py module stores explicit notes in JSON. That is not the same as learned weights, and this memory store is **not yet integrated into language-model generation**.

## Requirements and setup

Python 3.10+ is recommended. NumPy is the only third-party Python dependency.

~~~sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
~~~

## Prepare data before pretraining

Supply UTF-8 text files you have permission to use. A folder is searched recursively for .txt, .text, .md, .markdown, .rst, .html, .xml, .csv, and .jsonl files. Blank lines separate documents/paragraphs; the split is made at those boundaries to reduce train/validation leakage. A single unbroken text file with no blank-line boundaries counts as one document, so add separate files or paragraph separators.

Example folder:

~~~text
data/
  raw/
    book_or_article_001.txt
    book_or_article_002.txt
~~~

Do not include private conversations, credentials, sensitive data, or copyrighted material without the required rights. Review source licenses, privacy, quality, repetition, language balance, and harmful or low-quality content yourself. The --license-note option records your human-reviewed note in the manifest; it does not validate legal rights.

## One-command preparation and pretraining

The pretrain.py command prepares the train/validation corpus, learns the tokenizer from train only, tokenizes both splits, writes manifests/configuration, checks artifacts, then starts training.

~~~sh
python python/pretrain.py \
  --input data/raw \
  --run-dir runs/sora-base \
  --steps 10000 \
  --hidden-size 128 \
  --seq-len 128 \
  --vocab-size 1024
~~~

To prepare and verify all files **without starting training**:

~~~sh
python python/pretrain.py --input data/raw --run-dir runs/sora-base --prepare-only
~~~

For a small smoke test with a tiny corpus, lower the options (for example, --min-document-chars 1 --validation-fraction 0.2 --vocab-size 300 --hidden-size 16 --seq-len 32 --steps 4 --eval-every 2). A smoke test verifies code flow; it does not teach meaningful language ability.

The run directory contains:

- corpus/train.txt and corpus/validation.txt — deterministic document-level split.
- corpus/manifest.json — source hashes, split counts, byte/character counts, and requested license note.
- tokenizer.json — the learned BPE vocabulary/merge rules.
- tokens/*.tokens.bin and companion JSON — compact token cache, source/tokenizer hashes, type, and token count.
- run_config.json — architecture, vocabulary, parameter-count and rough float32 memory estimate.
- sora-checkpoint.npz — current/latest parameters and Adam state.
- sora-checkpoint.best.npz — checkpoint with the best measured validation loss.
- sora-checkpoint.metrics.jsonl — machine-readable step/loss/time records.

Training resumes from the latest checkpoint, preserving Adam state:

~~~sh
python python/pretrain.py --input data/raw --run-dir runs/sora-base --steps 10000 --resume
~~~

Keep the corpus and settings the same when resuming. The checkpoint refuses to resume with a different tokenizer fingerprint.

## Run the pieces individually

Prepare train/validation files:

~~~sh
python python/prepare_corpus.py data/raw --output-dir runs/sora-base/corpus --validation-fraction 0.02 --seed 7 --license-note "Reviewed source licenses"
~~~

Train the tokenizer using training text only:

~~~sh
python python/train_bpe.py runs/sora-base/corpus/train.txt runs/sora-base/tokenizer.json --vocab-size 1024
~~~

Train the model:

~~~sh
python python/train.py runs/sora-base/corpus/train.txt \
  --validation-corpus runs/sora-base/corpus/validation.txt \
  --tokenizer runs/sora-base/tokenizer.json \
  --checkpoint runs/sora-base/sora-checkpoint.npz \
  --steps 10000 --hidden-size 128 --seq-len 128
~~~

Generate text using the matching tokenizer:

~~~sh
python python/generate.py runs/sora-base/sora-checkpoint.npz \
  --tokenizer runs/sora-base/tokenizer.json --prompt "Sora" --tokens 200
~~~

Use the model with raw byte IDs (no BPE tokenizer) only for byte-level experiments. When a BPE checkpoint is used, pass its matching tokenizer.json.

## Tests

Run locally:

~~~sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

GitHub Actions runs the same test suite on pushes and pull requests. Tests cover model updates/checkpoints, persistent memory, BPE encode/decode including Unicode, cache correctness, deterministic data splitting, and a short end-to-end training run.

## What this does—and does not—mean

A pretraining run learns statistical next-token prediction from the supplied corpus. Lower validation loss is evidence that prediction improved on that evaluation split; it does not prove reasoning, factual reliability, multilingual ability, or general intelligence. Training requires a substantial representative corpus and compute, and corpus quality strongly affects the result.

Important limitations still remaining:

1. **Architecture:** the current manual RNN has limited context and capacity. A causal Transformer with attention is the next major model-development step before attempting serious foundation-model scale.
2. **Performance:** this is a CPU-oriented educational implementation with no GPU acceleration or distributed training. Long runs and large models will be slow.
3. **Evaluation:** add held-out benchmark suites, data-contamination checks, perplexity reports across languages/domains, and behavioral/safety evaluations.
4. **Memory/retrieval:** the JSON memory store is separate from model weights and is not fed into generation yet.
5. **Scale:** meaningful capability requires substantially more data, parameters, optimization tuning, and engineering than a small smoke test.

No model is pretrained merely by adding this code or downloading the repository. You must supply the corpus, inspect the manifest, run the preparation/training command, and evaluate the resulting checkpoints.
