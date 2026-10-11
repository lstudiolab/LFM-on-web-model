# Multilingual-first training from scratch

This plan changes the training direction without deleting old files. It creates a new checkpoint and tokenizer initialized from random weights rather than continuing from the earlier story, educational-text, or chat checkpoints. Those old checkpoints remain untouched for rollback and comparison.

## Goal and realistic scope

The first phase aims for broad language and writing-system exposure, then measures validation loss and generated text. FineWeb-2 publishes 1,868 language/script subsets, but only some have substantial data; that dataset is not a complete inventory of every language spoken on Earth. A sample appearing in training does not prove that Sora can actually speak it. Useful multilingual ability requires much more data, a much larger model, substantial accelerator time, and evaluation for each target language.

## One-command run on Kaggle or another CUDA notebook

From the repository root, with internet enabled:

    pip install -U datasets huggingface_hub
    python python/run_multilingual_from_scratch.py --steps 20000

The defaults sample up to 100 documents from each available non-English FineWeb-2 language/script subset plus 5,000 English documents from FineWeb sample-10BT. The downloader stores source metadata, counts, failures, and progress in data/multilingual/fineweb2/manifest.json. The first collection can take a long time because it attempts many subsets. If a session stops, run the same command again to continue collection.

To test the pipeline with only 30 language/script configs before the full broad sweep:

    python python/run_multilingual_from_scratch.py --max-configs 30 --english-docs 500 --steps 500

To prepare the corpus and multilingual tokenizer without training model weights:

    python python/run_multilingual_from_scratch.py --prepare-only

To resume a paused model-training run using the same data and architecture:

    python python/run_multilingual_from_scratch.py --skip-download --resume --steps 20000

Keep the same run directory, data files, tokenizer and architecture when resuming. The trainer verifies the corpus and tokenizer fingerprints; do not change training inputs between resume runs.

## Outputs

- data/multilingual/fineweb2/*.jsonl: bounded samples per language/script subset.
- data/multilingual/fineweb2/manifest.json: subset status, source, document counts and failures.
- runs/sora-multilingual-from-scratch/corpus/train.txt and validation.txt: deterministic split with documents interleaved across available language/script pairs.
- runs/sora-multilingual-from-scratch/tokenizer.json: byte-level BPE learned from the training split only, retaining UTF-8 byte fallback for unseen characters.
- runs/sora-multilingual-from-scratch/sora-multilingual.pt, sora-multilingual.best.pt and sora-multilingual.metrics.jsonl: model checkpoints and training metrics.

## Important constraints

1. The new pipeline starts from random weights. It deliberately does not pass an earlier checkpoint to the trainer.
2. The default is a broad starter sample, not full-data pretraining. 20,000 updates on a small Transformer cannot make it fluent in every language.
3. FineWeb-2 and FineWeb are released under ODC-By 1.0. Check the current dataset cards and Common Crawl terms before redistributing text or checkpoints. The manifest preserves source information for review.
4. The collector excludes subsets labelled as removed or unclassified, and uses FineWeb for English. Some subsets can fail or contain too little data; inspect the manifest.
5. Internet access is needed for dataset collection. A CUDA GPU is strongly recommended; CPU mode is intended for debugging.
6. Committing the scripts is not the same as running training. Only logs, checkpoints and evaluation results from an actual run can confirm that training happened.

Dataset references:
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
