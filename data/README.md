# Multilingual training data

The country-language policy lives at `configs/country_primary_languages.json`. It currently contains 197 country/territory entries and 110 distinct language codes. The Philippines selects **Tagalog (`tl`) only**. One language per country is a deliberate simplification: some choices are approximate defaults, many countries are multilingual, and some languages have no Wikipedia edition. Review the mapping before treating it as authoritative.

## Collect bootstrap text

Install the optional PyTorch training dependency:

~~~bash
python -m pip install -r requirements-torch.txt
~~~

Download Wikipedia article introductions where an edition exists. The downloader records article URLs and license information in `wikipedia-sources.json`; source availability varies by language.

~~~bash
python python/download_wikipedia_corpus.py --pages-per-language 1000
python python/prepare_multilingual_corpus.py --max-chars-per-language 5000000
~~~

Check `data/multilingual/raw/download-manifest.json` and `data/multilingual/selected/manifest.json`. Missing-language entries are not filled with invented text. Wikipedia introductions are just a starter corpus and will not make a small model fluent by themselves. Add much larger reviewed datasets with clear reuse rights in `data/multilingual/raw/<language-code>/`. Keep private conversations, credentials, sensitive data, and any text you do not have permission to use out of the training corpus. Review the relevant source licenses, including [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and Wikimedia's [attribution guidance](https://attribution.wikimedia.org/).

## Prepare and pretrain from random initialization

Prepare the language-labelled source files, document-level train/validation split, tokenizer, and caches:

~~~bash
python python/pretrain.py \
  --input data/multilingual/selected \
  --run-dir runs/sora-multilingual \
  --prepare-only \
  --hidden-size 256 --heads 8 --layers 6 --seq-len 256 \
  --vocab-size 2048 --tokenizer-max-lines 200000
~~~

Inspect the manifest, split files, tokenizer, and run configuration. Then run actual training with the optional PyTorch backend:

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

All weights are initialized from scratch if `--initialize-from` is not passed. Use the same training corpus, validation corpus, tokenizer, model dimensions, optimizer settings, and `--resume` to continue an interrupted run. The step count is a starting budget, not a promise of fluent language behavior; watch the validation metrics and available memory/compute.

## Dialogue fine-tuning

Plain-text articles teach next-token patterns, not how to answer a user helpfully. Prepare a separate JSONL file of high-quality dialogue pairs reviewed by fluent speakers or drawn from licensed sources. Every row needs these fields:

~~~json
{"language":"tl","user":"Kumusta ka?","assistant":"Mabuti ako, salamat! Ikaw?","source":"your-dataset-name","license":"CC0"}
~~~

Run this on a training file and a separate validation file:

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

The tuning job initializes its model weights from the pretraining checkpoint but resets the optimizer state. The tokenizer and model dimensions must exactly match the base checkpoint. Keep a separate final test set for each language at `data/multilingual/test/<language-code>.txt`.

## Evaluate by language

~~~bash
python python/evaluate_multilingual.py runs/sora-chat/sora-chat.best.pt \
  --backend torch \
  --test-dir data/multilingual/test \
  --tokenizer runs/sora-multilingual/tokenizer.json \
  --output runs/sora-chat/language-evaluation.json
~~~

The evaluator reports missing test corpora separately from completed evaluations. Perplexity is a next-token metric, not proof that Sora converses naturally. Use independent, fluent-speaker review for grammar, comprehension, responses, and translation quality.
