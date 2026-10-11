# Sora training artifacts: what is and is not in GitHub

This file is the artifact checklist for `lstudiolab/LFM-on-web-model`.

## Important status

The repository contains training code and dataset-recreation pipelines. It does **not** currently contain the exact outputs from the earlier Google Colab session, including the approximately 105,000-document corpus, its old 8,000-token tokenizer, or the checkpoint trained on the earlier 5,000-document split. Those files exist only if they were saved elsewhere and then uploaded. A GitHub commit cannot recover files from a disconnected Colab runtime.

The earlier run's reported checkpoint was an approximately 8.9-million-parameter model trained for one epoch (793 steps); its reported validation loss was about 5.99. This is a training experiment, not evidence of a capable assistant. Do not treat the earlier tokenizer/checkpoint as compatible with the newer multilingual pipeline.

## What belongs in this repository

- Training and inference source code.
- Dataset source definitions, collection/preparation scripts, license notes, split/tokenizer procedures.
- Model configuration, evaluation scripts, tests, and run manifests.
- Small test fixtures only.

## What must be copied from Colab to preserve the exact learned state

To preserve a run exactly, save and back up the **checkpoint plus its matching tokenizer, model configuration, optimizer/training state (if resuming), data/split manifest, and evaluation metrics**. A checkpoint by itself may be enough for inference only if its matching model architecture and tokenizer are also available; it is not enough to reproduce/resume every training step.

Recommended artifact bundle layout outside ordinary Git:

```text
sora-artifacts/
  pretraining-v1/
    sora-latest.pt
    tokenizer.json
    config.json
    corpus-manifest.json
    train-validation-manifest.json
    metrics.jsonl
```

Use a GitHub Release attached to this repository (the helper in `python/sora_artifacts.py` creates the release and records its URL in `models/sora-artifacts.json`) or a dedicated model/artifact store for binary files. Do not commit hundreds of megabytes of raw corpus text or model checkpoints to ordinary Git history. Keep dataset source/license metadata and the code that reconstructs the corpus in this repository. The release manifest remains marked `not_yet_published` until a Colab runtime containing a real checkpoint uploads it.

## Recover files from a Colab runtime

Follow [COLAB_RECOVER_SORA.md](COLAB_RECOVER_SORA.md). That procedure searches known output paths, checksums discovered files, and publishes a real checkpoint/tokenizer/config bundle as a repository release asset. It also updates [models/sora-artifacts.json](models/sora-artifacts.json). The procedure deliberately excludes LoRA adapters and Qwen artifacts.

## Re-create training data on a fresh Colab

Follow [COLAB_SORA_TRAINING.md](COLAB_SORA_TRAINING.md) and [MULTILINGUAL_TRAINING.md](MULTILINGUAL_TRAINING.md). This rebuilds a dataset from the documented public sources and trains from random initialization. It **does not restore the exact corpus, tokenizer, or learned weights from the earlier Colab run**.

Before resuming an existing checkpoint, keep the exact corpus, tokenizer, configuration, and checkpoint together. If any of these are missing, run a new versioned experiment rather than pretending it is the same training run.
