# Llab Foundation Model (LFM)

**LFM** means **Llab Foundation Model**. **Sora** is the experimental conversational AI being developed on top of this project.

This repository implements a small language model from random initialization. It is intended to make the training loop inspectable and repeatable, not to claim that a tiny model is already a capable assistant.

> **Current status:** the corpus-to-tokenizer-to-training workflow is implemented, and regression tests plus GitHub Actions are configured. The default pretraining path is a compact causal Transformer with manually implemented NumPy forward/backward math; the older RNN remains available as a baseline. This is still experimental, CPU-oriented, and not production-scale. No full pretraining run is claimed.

## What is implemented

- **Training from scratch:** the default is a causal Transformer with learned token/position embeddings, multi-head masked self-attention, residual connections, layer normalization, a feed-forward block, and tied input/output embeddings. All parameters begin from random initialization. `python/train.py` retains the RNN baseline.
- **Training math:** the Transformer manually implements causal attention, softmax cross-entropy, layer-normalization gradients, feed-forward backpropagation, gradient clipping, and Adam in `python/sora_transformer.py`; the RNN's BPTT remains in `python/sora_model.py`. NumPy provides array operations, not an autograd or neural-network training framework.
- **Raw bytes or BPE:** UTF-8 bytes work out of the box. The optional byte-level BPE tokenizer reduces sequence length for frequent text pieces.
- **Tokenizer consistency:** python/train_bpe.py learns the vocabulary; python/tokenizer.py encodes and decodes with it. The training and generation paths require a matching tokenizer fingerprint for BPE checkpoints.
- **Corpus preparation:** python/prepare_corpus.py streams text by paragraphs, normalizes Unicode, deduplicates documents using a temporary SQLite index, then creates a deterministic document-level train/validation split and a hash manifest.
- **Token caches:** python/tokenize_corpus.py writes compact token IDs to disk and checks corpus/tokenizer hashes before reusing a cache. It avoids keeping a complete token list in Python memory.
- **Training pipeline:** python/train.py reports train/validation loss, writes JSONL metrics, saves latest and best-validation checkpoints, preserves Adam state, and checks corpus/tokenizer/configuration fingerprints when resuming.
- **One-command pipeline:** python/pretrain.py prepares the corpus, trains BPE on the training split only, tokenizes both splits, writes a run configuration, performs preflight checks, and optionally starts training.
- **Text generation:** python/generate.py generates continuations from a trained checkpoint.
- **Explicit memory store:** python/memory.py and python/memory_cli.py provide a separate JSON memory store. This symbolic memory is not the same as the model learning its parameters, and memory retrieval is not yet wired into generation.
- **Tests and CI:** tests cover BPE Unicode and whitespace round-trips, C++/Python tokenizer parity, data split repeatability, cache integrity, model checkpoints, manual attention-gradient finite differences, a real sample-corpus pretraining run, and exact uninterrupted-versus-resumed training. GitHub Actions runs the suite on pushes and pull requests.

## Requirements and quick test

Python 3.10+ and NumPy are required. Run these commands from the repository root.

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

The files under data/samples are tiny synthetic examples to verify code paths. They are not a useful pretraining corpus.

### One-command end-to-end smoke training

To exercise corpus preparation, document-level splitting, BPE training, tokenization, validation, checkpoint saving, and sample generation with one command:

~~~bash
python python/pretrain.py \
  --input data/samples/pretrain-source \
  --run-dir runs/sora-smoke \
  --steps 30 --hidden-size 16 --seq-len 32 \
  --vocab-size 320 --heads 4 --layers 2 --validation-fraction 0.2 \
  --min-document-chars 1 --eval-every 10
~~~

This verifies that the pipeline works; the sample corpus is too small to produce useful language ability. The same workflow is run automatically by GitHub Actions.

## Option A: exercise the tokenizer and trainer on the included sample

Learn BPE from the training file only:

~~~bash
python python/train_bpe.py data/samples/train.txt artifacts/sora-tokenizer.json --vocab-size 320
~~~

Train the RNN baseline from random initialization:

~~~bash
python python/train.py data/samples/train.txt \
  --validation-corpus data/samples/validation.txt \
  --tokenizer artifacts/sora-tokenizer.json \
  --steps 300 --hidden-size 32 --seq-len 32 \
  --checkpoint artifacts/sora-checkpoint.npz \
  --eval-every 25 --eval-batches 4 \
  --generate-tokens 60 --token-cache-dir artifacts/cache
~~~

Generate a continuation:

~~~bash
python python/generate.py artifacts/sora-checkpoint.npz \
  --tokenizer artifacts/sora-tokenizer.json \
  --prompt "Sora learns" --tokens 60 --temperature 0.8 --top-k 20
~~~

To train directly on UTF-8 bytes, omit the BPE tokenizer flags:

~~~bash
python python/train.py data/samples/train.txt \
  --validation-corpus data/samples/validation.txt \
  --steps 300 --hidden-size 32 --seq-len 64 \
  --checkpoint artifacts/sora-byte-checkpoint.npz \
  --eval-every 25 --generate-tokens 60
~~~

This default path trains the causal Transformer; `--hidden-size` is its model width, and `--heads` must divide that width. For a real run, do not use the tiny sample files.

## Option B: prepare a real pretraining corpus

Put source files you have the rights to use into a dedicated folder such as data/raw. The preparer searches recursively for UTF-8 text files with extensions .txt, .text, .md, .markdown, .rst, .html, .xml, .csv, and .jsonl. Blank lines separate documents/paragraphs. A single unbroken text file without blank-line boundaries counts as one document; provide multiple files or paragraph separators so the splitter can keep documents apart.

Review every source for licensing and permission, privacy, secrets, quality, language/domain balance, repeated text, and inappropriate or corrupted material. The tool records source hashes and an optional human-written license note in the manifest; it does not determine whether you have legal rights to the data. Do not use private conversations, credentials, or sensitive user data without appropriate authorization.

Prepare, verify, and start a run with one command:

~~~bash
python python/pretrain.py \
  --input data/raw \
  --run-dir runs/sora-base \
  --steps 10000 \
  --hidden-size 128 \
  --seq-len 128 \
  --vocab-size 1024
~~~

To generate the dataset/tokenizer/cache/configuration and run preflight without training, append --prepare-only:

~~~bash
python python/pretrain.py --input data/raw --run-dir runs/sora-base --prepare-only
~~~

For a tiny smoke test, use options such as --min-document-chars 1 --validation-fraction 0.2 --vocab-size 300 --hidden-size 16 --seq-len 32 --steps 4 --eval-every 2. Smoke tests check that the pipeline runs; they do not create useful language ability.

Do not pass data/samples as the input folder to pretrain.py: that folder includes a separate validation file for Option A, and the one-command pipeline would correctly treat every file in the input folder as source training material. To use that prepared train/validation pair, use the explicit python/train.py command shown above.

The run directory contains:

- corpus/train.txt and corpus/validation.txt — deterministic document-level splits.
- corpus/manifest.json — source hashes, document counts, byte/character counts, and the optional license note.
- tokenizer.json — learned BPE merge rules.
- tokens/*.tokens.bin and companion JSON manifests — token IDs, data type, token count, and source/tokenizer hashes.
- run_config.json — model size estimate, vocabulary, training options, and tokenizer fingerprint.
- sora-checkpoint.npz — latest parameters and Adam optimizer state.
- sora-checkpoint.best.npz — checkpoint with the best measured validation loss.
- sora-checkpoint.metrics.jsonl — step metrics for plotting or further analysis.

Resume with the same data and settings:

~~~bash
python python/pretrain.py --input data/raw --run-dir runs/sora-base --steps 10000 --resume
~~~

When resuming, --steps means additional optimizer updates. The checkpoint will reject a changed tokenizer, training corpus, validation corpus, hidden size, sequence length, or learning rate rather than silently mixing incompatible runs.

## Optional C++ tokenizer utility

The C++17 utility checks tokenization using a BPE JSON file emitted by the Python trainer:

~~~bash
c++ -std=c++17 -O2 cpp/main.cpp cpp/tokenizer.cpp -o sora-tokenizer
./sora-tokenizer artifacts/sora-tokenizer.json "Sora mahal kita 😊"
~~~

Unicode word-boundary handling in the demonstration C++ utility is intentionally lightweight; Python's tokenizer is the reference for training and generation.

## What this does—and does not—mean

Pretraining minimizes next-token prediction loss on the supplied text. Lower validation loss means the model improved on that evaluation split; it does not prove reasoning, factual reliability, multilingual competence, or general intelligence. No model is pretrained merely by committing code or downloading the repository. You must provide a corpus, inspect the manifest, execute training, and evaluate the resulting checkpoint.

Important limitations that remain:

1. **Architecture:** the manually implemented RNN has limited capacity and recurrent context. A causal Transformer with attention is the next major model-development step before serious foundation-model scale.
2. **Performance:** the current model is CPU-oriented, has no GPU acceleration or distributed training, and will be slow for long or large runs.
3. **Evaluation:** benchmark suites, data-contamination checks, per-language/domain perplexity reports, and behavioral/safety evaluations are still needed.
4. **Memory/retrieval:** the JSON memory store is separate from learned parameters and is not supplied to the generator yet.
5. **Scale:** useful capability requires significantly more data, parameters, optimization tuning, compute, and engineering than a smoke test.
