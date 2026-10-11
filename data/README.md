# Multilingual training data

The multilingual corpus builder uses explicitly labelled language folders:

```text
data/multilingual/raw/
  fil/   # Filipino (Tagalog-based)
  en/    # English
  ja/    # Japanese
  ...
```

Place only reviewed, legally usable text in each folder. The folder's language code is treated as a label; the script does not automatically detect or translate languages. Keep source URLs, license/permission notes, and attribution alongside your dataset records.

The starter policy is `configs/country_primary_languages.json`. It is intentionally incomplete and must not be described as every country's primary language. It selects Filipino (Tagalog-based) for the Philippines, matching this project's requested scope. Some countries are multilingual; selecting one language is a project simplification, not a statement that other languages are unimportant or unused.

Prepare language-specific corpora:

```bash
python python/prepare_multilingual_corpus.py
```

Review `data/multilingual/selected/manifest.json`. Missing language data is reported, not fabricated. Then train using the selected text files as input to the existing pretraining pipeline, after you have completed and reviewed the country-language policy and added enough high-quality data:

```bash
python python/pretrain.py --input data/multilingual/selected --run-dir runs/sora-multilingual --steps 10000 --hidden-size 128 --heads 4 --layers 2 --seq-len 128 --vocab-size 4096
```

This starter script and policy do not themselves train Sora, download datasets, or establish multilingual capability. The repository's current NumPy Transformer is experimental and CPU-oriented; meaningful multilingual pretraining needs a much larger reviewed corpus, appropriate compute, and per-language held-out evaluation.
