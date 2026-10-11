# Training data

The files in samples/ are tiny synthetic examples for checking the tokenizer and training loop. They are not a meaningful language-model training corpus.

For pretraining, add source text that you are legally allowed to use under raw/ or another dedicated directory. Keep private conversations, credentials, and sensitive information out of the corpus and repository. Review licenses, source quality, duplicate content, language/domain balance, privacy, and formatting before training.

Run the full corpus pipeline from the repository root:

~~~sh
python python/pretrain.py --input data/raw --run-dir runs/sora-base --prepare-only
~~~

Inspect runs/sora-base/corpus/manifest.json, the train/validation split, tokenizer.json, and run_config.json before starting a long training job. When ready, omit --prepare-only. The preparer deduplicates paragraphs/documents and makes a deterministic document-level split. The BPE vocabulary is learned from training text only; validation is kept separate. A single unbroken file without blank-line paragraph boundaries is one document, so provide multiple files or paragraph separators.

See the top-level README for full training, resume, generation, and testing commands.
