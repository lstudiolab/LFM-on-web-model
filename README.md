# Llab Foundation Model (LFM)

**LFM** means **Llab Foundation Model**. **Sora** is the experimental conversational AI being developed on top of this project.

This repository implements a small language model from random initialization. It is intended to make the training loop inspectable and repeatable, not to claim that a tiny model is already a capable assistant.

> **Current status:** the NumPy Transformer remains an inspectable baseline, and an optional PyTorch Transformer trainer now supports batched training on CPU/CUDA/MPS with randomly initialized weights. A country-language policy, Wikimedia starter-data downloader, dialogue-tuning formatter, and per-language held-out evaluation path are also included. No full multilingual pretraining run has been completed; these tools prepare and train a model only when you run them with real data and suitable compute.

## What is implemented

- **Training from scratch:** the default is a causal Transformer with learned token/position embeddings, multi-head masked self-attention, residual connections, layer normalization, a feed-forward block, and tied input/output embeddings. All parameters begin from random initialization. `python/train.py` retains the RNN baseline.
- **Training backends:** `python/train_transformer.py` is the small manual-gradient NumPy trainer. `python/train_torch_transformer.py` is the recommended scalable path: it uses batched PyTorch ops, AdamW, warmup/cosine learning-rate scheduling, gradient clipping, checkpoint resume, and initialization from an earlier Sora checkpoint for a new training stage. Both start model weights randomly when no initialization checkpoint is supplied.
- **Raw bytes or BPE:** UTF-8 bytes work out of the box. The optional byte-level BPE tokenizer reduces sequence length for frequent text pieces.
- **Tokenizer consistency:** python/train_bpe.py learns the vocabulary; python/tokenizer.py encodes and decodes with it. The training and generation paths require a matching tokenizer fingerprint for BPE checkpoints.
- **Corpus preparation:** python/prepare_corpus.py streams text by paragraphs, normalizes Unicode, deduplicates documents using a temporary SQLite index, then creates a deterministic document-level train/validation split and a hash manifest.
- **Token caches:** python/tokenize_corpus.py writes compact token IDs to disk and checks corpus/tokenizer hashes before reusing a cache. It avoids keeping a complete token list in Python memory.
- **Training pipeline:** python/train.py reports train/validation loss, writes JSONL metrics, saves latest and best-validation checkpoints, preserves Adam state, and checks corpus/tokenizer/configuration fingerprints when resuming.
- **One-command pipeline:** python/pretrain.py prepares the corpus, trains BPE on the training split only, tokenizes both splits, writes a run configuration, performs preflight checks, and optionally starts training.
- **Text generation:** python/generate.py generates continuations from a trained checkpoint.
- **Explicit memory store:** python/memory.py and python/memory_cli.py provide a separate JSON memory store. This symbolic memory is not the same as the model learning its parameters, and memory retrieval is not yet wired into generation.
- **Tests and CI:** tests cover BPE Unicode and whitespace round-trips, C++/Python tokenizer parity, data split repeatability, cache integrity, model checkpoints, manual attention-gradient finite differences, a real sample-corpus pretraining run, and exact uninterrupted-versus-resumed training. GitHub Actions runs the suite on pushes and pull requests.

## Preserve and publish actual Colab-trained Sora weights

Training scripts alone do not contain the learned weights from a Colab run. To recover actual checkpoints and matching tokenizers/configs from a Colab filesystem and register them as a repository Release asset, follow [COLAB_RECOVER_SORA.md](COLAB_RECOVER_SORA.md). The resulting release asset is linked from [models/sora-artifacts.json](models/sora-artifacts.json). It remains marked unpublished until that procedure is run in a runtime containing a real checkpoint. Raw corpus text is excluded by default and should not be redistributed without checking source licenses.

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

## Option A: exercise the tokenizer and trainers on the included sample

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

Train the default causal Transformer directly on the explicit train/validation pair:

~~~bash
python python/train_transformer.py data/samples/train.txt --validation-corpus data/samples/validation.txt --tokenizer artifacts/sora-tokenizer.json --steps 300 --d-model 64 --heads 4 --layers 2 --seq-len 64 --checkpoint artifacts/sora-transformer.npz --eval-every 25 --generate-tokens 60 --token-cache-dir artifacts/transformer-cache
~~~

The Transformer trainer is also the model used by python/pretrain.py. The command above is only a test run; the included sample files are too small for meaningful learning.

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

`--hidden-size` is the Transformer model width, and `--heads` must divide that width in the one-command path above. For a real run, do not use the tiny sample files.

## Final evaluation on a held-out test set

Keep three distinct splits: training data updates weights, validation data selects checkpoints/configurations, and test data is used only for final reporting. Do not use the test score to tune settings or choose the best checkpoint.

After training, run the evaluator against a separately curated test file:

~~~bash
python python/evaluate.py \
  runs/sora-base/sora-checkpoint.best.npz \
  data/held-out/test.txt \
  --tokenizer runs/sora-base/tokenizer.json \
  --output runs/sora-base/test-results.json
~~~

For byte-level checkpoints, omit --tokenizer. The evaluator checks the checkpoint/tokenizer match, evaluates every next-token target in the test file once, and writes mean cross-entropy loss and perplexity with hashes identifying the checkpoint inputs. Perplexity is a language-model metric, not proof of reasoning or factual accuracy.

## Option B: prepare a real pretraining corpus

Put source files you have the rights to use into a dedicated folder such as data/raw. The preparer searches recursively for UTF-8 text files with extensions .txt, .text, .md, .markdown, .rst, .html, .xml, .csv, and .jsonl. Blank lines separate documents/paragraphs. A single unbroken text file without blank-line boundaries counts as one document; provide multiple files or paragraph separators so the splitter can keep documents apart.

Review every source for licensing and permission, privacy, secrets, quality, language/domain balance, repeated text, and inappropriate or corrupted material. The tool records source hashes and an optional human-written license note in the manifest; it does not determine whether you have legal rights to the data. Do not use private conversations, credentials, or sensitive user data without appropriate authorization.

Prepare, verify, and start a run with one command:

~~~bash
python python/pretrain.py \
  --input data/raw \
  --run-dir runs/sora-base \
  --steps 10000 \
  --hidden-size 128 --heads 4 --layers 2 \
  --seq-len 128 \
  --vocab-size 1024
~~~

To generate the dataset/tokenizer/cache/configuration and run preflight without training, append --prepare-only:

~~~bash
python python/pretrain.py --input data/raw --run-dir runs/sora-base --prepare-only
~~~

For a tiny smoke test, use options such as --min-document-chars 1 --validation-fraction 0.2 --vocab-size 300 --hidden-size 16 --heads 4 --layers 1 --seq-len 32 --steps 4 --eval-every 2. Smoke tests check that the pipeline runs; they do not create useful language ability.

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

## Multilingual training and conversational tuning

The country-language policy contains 197 country/territory entries and 110 distinct language codes. The Philippines selects **Tagalog (`tl`) only**. The policy deliberately chooses one language per entry, even though many countries are multilingual; some choices need local review, and not all selected languages have a Wikipedia edition. Do not treat it as an authoritative linguistic census.

### 1. Install the scalable training backend

~~~bash
python -m pip install -r requirements-torch.txt
~~~

The PyTorch model is initialized from random weights. PyTorch provides optimized tensor operations; it does not provide a pretrained language model here. Use an appropriate PyTorch build for the hardware you have. The trainer selects CUDA when available, then Apple MPS, then CPU.

### 2. Get starter text and prepare the language corpora

~~~bash
python python/download_wikipedia_corpus.py --pages-per-language 1000
python python/prepare_multilingual_corpus.py --max-chars-per-language 5000000
~~~

The downloader fetches **article introductions**, not complete Wikipedia articles. It saves article URLs and license metadata beside each language's text. The Wikipedia API is documented at [MediaWiki API:Random](https://www.mediawiki.org/wiki/API:Random) and [API:Extracts](https://www.mediawiki.org/wiki/Extension:TextExtracts). Most Wikimedia text is offered under CC BY-SA 4.0; review the [license terms](https://creativecommons.org/licenses/by-sa/4.0/) and Wikimedia's [attribution guidance](https://attribution.wikimedia.org/) before redistributing training material or model artifacts.

This is only a bootstrap dataset, not enough to claim fluency. Some language editions may not exist or may return no usable pages. The scripts report missing languages in `data/multilingual/raw/download-manifest.json` and `data/multilingual/selected/manifest.json`. Add larger, reviewed, appropriately licensed corpora under `data/multilingual/raw/<language-code>/` before serious training. Avoid private messages, credentials, and sensitive data.

### 3. Prepare the tokenizer and language-level split

~~~bash
python python/pretrain.py \
  --input data/multilingual/selected \
  --run-dir runs/sora-multilingual \
  --prepare-only \
  --hidden-size 256 --heads 8 --layers 6 --seq-len 256 \
  --vocab-size 2048 --tokenizer-max-lines 200000
~~~

This prepares training and validation splits, learns the BPE tokenizer from a bounded deterministic sample of training lines, and writes token caches/configuration. It does **not** train the model because `--prepare-only` is set. Inspect `runs/sora-multilingual/corpus/manifest.json`, `tokenizer.json`, and `run_config.json` before continuing.

### 4. Actually pretrain Sora from scratch

~~~bash
python python/train_torch_transformer.py \
  runs/sora-multilingual/corpus/train.txt \
  --validation-corpus runs/sora-multilingual/corpus/validation.txt \
  --tokenizer runs/sora-multilingual/tokenizer.json \
  --checkpoint runs/sora-multilingual/sora-torch.pt \
  --steps 100000 --batch-size 8 \
  --d-model 256 --heads 8 --layers 6 --seq-len 256 \
  --device auto
~~~

The trainer saves a latest checkpoint, a best-validation checkpoint, and a JSONL metrics log. The number of updates is an example starting budget, not a guarantee of quality. Monitor validation loss, resource use, and per-language balance; resume with the same checkpoint and data plus `--resume`. A larger reviewed corpus and more compute are likely required for good multilingual coverage.

### 5. Teach it to respond conversationally

Next-token pretraining on encyclopedia text alone does not teach assistant behavior. Provide a separate JSONL file with **native-speaker-reviewed or otherwise high-quality licensed dialogues**, one record per line:

~~~json
{"language":"tl","user":"Kumusta ka?","assistant":"Mabuti ako, salamat! Ikaw?","source":"your-dataset-name","license":"CC0"}
~~~

Each record must include `language`, `user`, `assistant`, `source`, and `license`. Use data you have rights to train on, and review grammar and naturalness. Keep training, validation, and final test examples separate.

~~~bash
python python/prepare_chat_corpus.py data/multilingual/dialogue-train.jsonl \
  --output-dir data/multilingual/chat-train
python python/prepare_chat_corpus.py data/multilingual/dialogue-validation.jsonl \
  --output-dir data/multilingual/chat-validation

python python/train_torch_transformer.py \
  data/multilingual/chat-train/chat-tuning.txt \
  --validation-corpus data/multilingual/chat-validation/chat-tuning.txt \
  --tokenizer runs/sora-multilingual/tokenizer.json \
  --initialize-from runs/sora-multilingual/sora-torch.best.pt \
  --checkpoint runs/sora-chat/sora-chat.pt \
  --steps 20000 --batch-size 8 \
  --d-model 256 --heads 8 --layers 6 --seq-len 256 \
  --learning-rate 0.0001 --device auto
~~~

The tuning checkpoint must use the same tokenizer and model configuration as the base checkpoint. This stage resets optimizer state and starts a new run from the base **weights**; it does not replace the base checkpoint. Generation should use the conversation markers learned from the dialogue corpus, for example:

~~~bash
python python/generate_torch.py runs/sora-chat/sora-chat.best.pt \
  --tokenizer runs/sora-multilingual/tokenizer.json \
  --prompt "<|system|>You are Sora. Answer naturally in Tagalog.\n<|user|>Kumusta ka?\n<|assistant|>" \
  --tokens 120 --device auto
~~~

### 6. Evaluate each language separately

Place **held-out test text never used for training or tuning** at `data/multilingual/test/<language-code>.txt`, then run:

~~~bash
python python/evaluate_multilingual.py runs/sora-chat/sora-chat.best.pt \
  --backend torch \
  --test-dir data/multilingual/test \
  --tokenizer runs/sora-multilingual/tokenizer.json \
  --output runs/sora-chat/language-evaluation.json
~~~

Missing test files are reported rather than counted as successful languages. Perplexity measures next-token prediction only. To establish that Sora speaks a language well, native speakers must also review conversations for grammar, naturalness, understanding, and appropriate responses.

## What this does—and does not—mean

Pretraining minimizes next-token prediction loss on the supplied text. Lower validation loss means the model improved on that evaluation split; it does not prove reasoning, factual reliability, multilingual competence, or general intelligence. No model is pretrained merely by committing code or downloading the repository. You must provide a corpus, inspect the manifest, execute training, and evaluate the resulting checkpoint.

Important limitations that remain:

1. **Architecture/scale:** both implementations here are compact experiments, much smaller than a modern production LLM. The optional PyTorch path batches sequences and can use CUDA/MPS, but has no distributed training, sharded optimizer, or large-scale data pipeline.
2. **Performance:** the manual NumPy implementation is CPU-oriented and slow. PyTorch speeds up tensor math when suitable hardware is available, but large runs still require substantial compute and memory.
3. **Evaluation:** benchmark suites, data-contamination checks, per-language/domain perplexity reports, and behavioral/safety evaluations are still needed.
4. **Memory/retrieval:** the JSON memory store is separate from learned parameters and is not supplied to the generator yet.
5. **Scale:** useful capability requires significantly more data, parameters, optimization tuning, compute, and engineering than a smoke test.
