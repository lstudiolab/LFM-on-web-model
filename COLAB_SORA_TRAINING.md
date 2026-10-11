# Sora training in Google Colab

This repository is the source of truth for Sora's code and reproducible training setup. Large downloaded corpora, token caches, and model checkpoints should be generated in the Colab runtime (or stored in a separate artifact store), not committed to Git: they can be hundreds of megabytes or larger and are reproducible from the scripts below.

## Start a fresh Colab runtime

Choose a GPU runtime, then run these cells in order. These commands clone the repository and use its existing from-scratch pipeline. They do not download or load Qwen, LoRA, or other pretrained model weights.

### Cell 1 — clone the latest repository

```python
%cd /content
!git clone https://github.com/lstudiolab/LFM-on-web-model.git
%cd /content/LFM-on-web-model
!python -m pip install -U pip
!python -m pip install -r requirements.txt
!python -m pip install -U datasets huggingface_hub
```

### Cell 2 — verify the runtime and repository

```python
import torch, pathlib, subprocess
print("Repository:", pathlib.Path.cwd())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
subprocess.run(["git", "log", "-1", "--oneline"], check=True)
```

### Cell 3 — build a multilingual corpus and train Sora from random initialization

This uses the repository's maintained multilingual pipeline, which downloads a bounded sample from available FineWeb-2 language/script subsets plus English FineWeb data, builds a tokenizer and document-level train/validation split, and trains a new model from random weights.

```python
!python python/run_multilingual_from_scratch.py --max-configs 30 --english-docs 5000 --steps 1500
```

This is a practical first Colab run, not a claim that 1,500 updates make a capable assistant. The first run may take time while data is downloaded and prepared. Review the printed output and the manifest for missing or failed language subsets.

### Cell 4 — inspect the actual outputs and parameter count

```python
import json, pathlib, torch
root = pathlib.Path("/content/LFM-on-web-model")
manifest = root / "data/multilingual/fineweb2/manifest.json"
run = root / "runs/sora-multilingual-from-scratch"
print("Manifest exists:", manifest.exists())
if manifest.exists():
    data = json.loads(manifest.read_text())
    print(json.dumps(data, indent=2)[:6000])
for p in sorted(run.glob("*")):
    print(p.name, f"{p.stat().st_size:,} bytes" if p.is_file() else "(directory)")
```

### Continue the same run

If Colab disconnects, clone the repository again, install the same requirements, then restore the generated `data/multilingual` and `runs/sora-multilingual-from-scratch` directories from your own persistent storage before running:

```python
!python python/run_multilingual_from_scratch.py --skip-download --resume --steps 3000
```

**Important:** `--resume` is only valid when the same checkpoint, corpus, tokenizer, and architecture are present. Do not use it with a newly generated tokenizer or changed corpus. Keep a backup of the run directory before continuing.

## Keep training artifacts out of Git

- Keep scripts, configuration, documentation, and small test fixtures in this repository.
- Keep full corpora, token caches, checkpoints, and run logs in Colab storage or a dedicated artifact store.
- Do not overwrite older checkpoints when experimenting with a different model size; use a separate run directory for each version.
- Compare validation loss and held-out evaluations between runs. More parameters alone do not guarantee better language ability; training data, training duration, and evaluation matter too.

For the broader default multilingual sweep and full pipeline options, see [MULTILINGUAL_TRAINING.md](MULTILINGUAL_TRAINING.md).
