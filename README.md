# Llab Foundation Model (LFM)

**LFM (Llab Foundation Model)** is the foundation-model project behind **Sora**, an experimental conversational AI being developed from scratch.

## Included training pipeline

- **Randomly initialized byte-level RNN:** predicts the next UTF-8 byte; no pretrained model.
- **From-scratch training math:** softmax cross-entropy, backpropagation through time, gradient clipping, and Adam are implemented in `python/sora_model.py`. NumPy is used for numerical array operations, not a prebuilt neural-network trainer.
- **Training and validation:** `python/train.py` prints loss metrics and periodically saves checkpoints.
- **Checkpoint/resume:** parameters and Adam optimizer state are saved in `sora-checkpoint.npz`.
- **Generation:** `python/generate.py` generates text from a checkpoint.
- **Persistent memory:** `python/memory.py` stores explicit notes/facts in JSON, and `python/memory_cli.py` lets you add, search, list, and delete them.
- **BPE prototype:** `python/train_bpe.py` remains an experimental vocabulary trainer; the current neural model uses raw byte IDs and does not yet consume the BPE vocabulary.

## Learned parameters vs. memory

Pretraining/training changes the neural network's parameters. The checkpoint stores those learned parameters and optimizer state. The knowledge is encoded statistically in the weights; it is not stored as a human-readable list of facts.

The separate JSON memory is explicit external memory. It persists selected text records but does not itself train the network. The current memory search uses simple word overlap. To use memories during a conversation, a later retrieval layer must find relevant records and add them to the model's context. This small RNN has a short recurrent context and is not yet a capable conversational model.

## Setup

Python 3.10+:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## Train from scratch

Create a UTF-8 text file, for example `data/train.txt`, containing text you have rights to use.

```sh
python python/train.py data/train.txt --steps 1000 --hidden-size 64 --seq-len 128
```

The model starts from random weights. It prints training and validation loss, saves a checkpoint, and produces a sample. A falling loss indicates better next-byte prediction on the evaluated data, not proof of general intelligence. Use a small dataset first to smoke-test the pipeline; meaningful language ability requires substantially more data, a stronger architecture, compute, and evaluation.

Resume:

```sh
python python/train.py data/train.txt --steps 1000 --resume
```

Generate from a saved checkpoint:

```sh
python python/generate.py sora-checkpoint.npz --prompt "Sora" --bytes 200
```

## Persistent memory commands

```sh
python python/memory_cli.py --file sora-memory.json add "Sora is the Llab Foundation Model project" --category project
python python/memory_cli.py --file sora-memory.json search "Llab Foundation Model"
python python/memory_cli.py --file sora-memory.json list
python python/memory_cli.py --file sora-memory.json forget 0
```

Memory files can contain private user data; keep them out of source control.

## Tests

```sh
python -m unittest discover -s tests -v
```

## Known limitations / next work

1. Integrate the BPE vocabulary into training and inference; the current model uses 256 byte IDs.
2. Implement and test a Transformer with attention.
3. Add stronger dataset streaming and train/validation splitting.
4. Add retrieval-augmented context so stored memories can be supplied to generation.
5. Add GPU acceleration and scaling benchmarks.
6. Run tests and end-to-end training on the target environment before claiming production readiness.
