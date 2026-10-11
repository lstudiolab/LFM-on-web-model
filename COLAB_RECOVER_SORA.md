# Recover Sora's actual Colab-trained state

This is the practical bridge between an earlier Google Colab session and this GitHub repository. The repository connector cannot read a user's live Colab RAM or files, so run the cells below in the Colab notebook that still has the files, or in a new notebook after restoring the old files from persistent storage.

## What this does

1. Searches known Sora output folders for model checkpoints, matching tokenizers, configs, manifests, and metrics.
2. Packages the files it actually finds into a hash-manifested archive.
3. Publishes the archive as a release asset attached to this repository, then writes a small pointer at [models/sora-artifacts.json](models/sora-artifacts.json).
4. Lets the web backend or another training notebook discover the public artifact from the repository manifest.

It **does not invent or recreate old weights**. The publisher refuses to publish a bundle unless it contains at least one checkpoint. If the old Colab files were lost and never saved to Drive or another host, their exact learned weights cannot be recovered; the source code can train a new model but that is not the same model.

## Cell 1 — clone the current repository and install the small artifact tool

Run in Google Colab. If the repository is already cloned, this cell uses the existing clone.

```python
from pathlib import Path
import subprocess

repo = Path("/content/LFM-on-web-model")
if not repo.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/lstudiolab/LFM-on-web-model.git", str(repo)],
        check=True,
    )
%cd /content/LFM-on-web-model
```

## Cell 2 — optionally mount Google Drive

Skip this cell if no old Sora files are stored in Drive. If they are stored there, mount Drive so the collector can see them.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 3 — collect model state from known Sora folders

This excludes raw text corpora by default, but includes checkpoint files, tokenizer files, model/run configs, reports, data manifests, and metric logs. It looks in the common paths used by the earlier and current Sora runs. It prints a warning if it cannot find a checkpoint.

```python
!python python/sora_artifacts.py collect \
  --search-root /content \
  --search-root /content/LFM-on-web-model \
  --search-root /content/drive/MyDrive/Sora \
  --output /content/sora-state-bundle.tar.gz
```

Inspect the files and size before publishing:

```python
import tarfile, json, os
bundle = "/content/sora-state-bundle.tar.gz"
print("Bundle size (MiB):", round(os.path.getsize(bundle) / (1024**2), 2))
with tarfile.open(bundle, "r:gz") as archive:
    manifest = json.load(archive.extractfile("SORA_BUNDLE_MANIFEST.json"))
print("Checkpoints:", manifest["checkpoint_files"])
print("Files:", manifest["file_count"])
print("Includes raw corpus:", manifest["contains_raw_corpus"])
for item in manifest["files"]:
    print(item["path"], item["size_bytes"], item["sha256"])
```

**Do not continue to publishing if the checkpoint list is empty.** That would mean the searched folders do not include the trained model weights. Search for the old checkpoint file in Colab's Files sidebar or your persistent storage, then add its parent folder as another `--search-root`.

## Cell 4 — publish the checkpoint bundle as a repository release asset

The repository is public, so release assets are publicly downloadable. Only publish weights and files you are allowed to redistribute. Do not include the raw FineWeb/FineWeb-Edu corpus in this public bundle unless you have confirmed that redistribution is permitted. The code can reconstruct fresh public-source samples; that does not reproduce the exact old corpus.

Create a **fine-grained GitHub personal access token** for `lstudiolab/LFM-on-web-model` with repository **Contents: Read and write** access. Keep it private. This cell asks for it without echoing it.

```python
import os, getpass
os.environ["GITHUB_TOKEN"] = getpass.getpass("GitHub token (hidden): ")
```

Then publish. Use a new tag when publishing a different training run.

```python
!python python/sora_artifacts.py publish \
  --bundle /content/sora-state-bundle.tar.gz \
  --repo lstudiolab/LFM-on-web-model \
  --tag sora-colab-state-2026-10-11
```

On success, the script creates a GitHub Release and updates `models/sora-artifacts.json` with the release asset URL, SHA-256, and checkpoint file names. The token is never written to the repository or printed by the script. Remove it from the runtime when finished:

```python
os.environ.pop("GITHUB_TOKEN", None)
```

## About the corpus and resuming training

The model checkpoint and its exact matching tokenizer/config are the learned state needed to load the model for inference. To **resume optimizer training exactly**, you also need the original training/validation data, the compatible training state, and the original run settings. Large raw corpora and token caches are not included by default. Keep them in private/persistent storage unless redistribution rights have been checked.

If you want to archive the corpus too, rerun `collect` with `--include-corpus`, inspect the manifest and license for every source, and publish only if you have permission. Prefer a private/persistent artifact store for the raw data.

## Important compatibility note

The previous 8.9M-parameter experiment and the current multilingual pipeline may use different model classes/tokenizers. Keep each checkpoint with its own tokenizer/config and do not try to load one run's weights using a different run's architecture. This process preserves artifacts; it does not assert that old checkpoints are compatible with the current multilingual inference code.
