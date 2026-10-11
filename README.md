# Llab Foundation Model (LFM)

LFM is the foundation-model project behind **Sora**, an experimental conversational AI being developed from scratch.

## Project goals

- Build a tokenizer and vocabulary trainer without pretrained tokenizers.
- Implement a small neural network and training algorithm from first principles.
- Train on curated text data, then measure learning with reproducible evaluations.
- Add checkpointing so learned parameters can be saved and training can resume.
- Develop memory and reasoning components only after the core training loop works.

## Technology

- **C++17:** core engine and performance-sensitive components.
- **Python 3:** data preparation, training experiments, and evaluation.
- **No pretrained language model:** initial experiments begin with randomly initialized parameters.

## Current stage: tokenizer prototype

The first component is a byte-level BPE tokenizer:

- `python/train_bpe.py` trains a small merge table from a UTF-8 text corpus.
- `cpp/tokenizer.hpp` and `cpp/tokenizer.cpp` encode and decode token IDs.
- `cpp/main.cpp` provides a tiny demonstration.
- `tests/test_bpe.py` contains basic Python tests.

This is an educational prototype, not a production tokenizer. The current Python trainer operates over a concatenated byte stream and does not yet preserve word boundaries. The C++ demo uses a small illustrative merge table; loading the Python-generated vocabulary into C++ is a follow-up task.

## Next milestones

1. Improve BPE training to operate over corpus word frequencies and preserve boundaries.
2. Define a stable vocabulary file format and load it in C++.
3. Add round-trip, Unicode, determinism, and cross-language tests.
4. Implement tensor operations and a small neural network from scratch.
5. Add next-token prediction, loss calculation, backpropagation, and an optimizer.
6. Add checkpointing and evaluation on held-out data.

## Build the C++ demo

```sh
c++ -std=c++17 -O2 cpp/main.cpp cpp/tokenizer.cpp -o sora-tokenizer
./sora-tokenizer "Sora can learn."
```

## Train an initial vocabulary

```sh
python3 python/train_bpe.py corpus.txt vocab.json --vocab-size 512
```

The corpus must be UTF-8 text. Use data you have permission to use, and keep evaluation examples separate from training data.
