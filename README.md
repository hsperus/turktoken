# TurkToken

A Turkish-optimized Byte Pair Encoding (BPE) tokenizer with GPT-4 style pre-tokenization patterns.

## Installation

```bash
pip install turktoken
```

## Quick Start

```python
from turktoken import TurkishBPETokenizer

tokenizer = TurkishBPETokenizer()
```

## Training

Train a tokenizer on your own corpus:

```python
from turktoken import TurkishBPETokenizer

tokenizer = TurkishBPETokenizer()

# Load your training data
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Train with desired vocabulary size
tokenizer.train(text, vocab_size=8000)

# Save the trained tokenizer
tokenizer.save("./my_tokenizer")
```

## Encoding and Decoding

```python
from turktoken import TurkishBPETokenizer

# Load a trained tokenizer
tokenizer = TurkishBPETokenizer()
tokenizer.load("./my_tokenizer")

# Encode text to token IDs
text = "Merhaba dünya! Bu bir test cümlesidir."
ids = tokenizer.encode(text)
print(ids)  # [256, 312, 445, 78, ...]

# Decode token IDs back to text
decoded_text = tokenizer.decode(ids)
print(decoded_text)  # "Merhaba dünya! Bu bir test cümlesidir."
```

## Special Tokens

Add custom special tokens for your use case:

```python
from turktoken import TurkishBPETokenizer

tokenizer = TurkishBPETokenizer()
tokenizer.train(training_text, vocab_size=4096)

# Add special tokens
tokenizer.add_special_tokens([
    "<|bos|>",      # beginning of sequence
    "<|eos|>",      # end of sequence
    "<|pad|>",      # padding
    "<|unk|>",      # unknown token
    "<|sep|>"       # separator
])

tokenizer.save("./my_tokenizer")
```

Using special tokens during encoding:

```python
tokenizer = TurkishBPETokenizer()
tokenizer.load("./my_tokenizer")

# Special tokens are recognized during encoding
ids = tokenizer.encode("<|bos|>Merhaba dünya!<|eos|>")
text = tokenizer.decode(ids)
```

## API Reference

### TurkishBPETokenizer

| Method | Description |
|--------|-------------|
| `train(text, vocab_size=512)` | Train the tokenizer on the given text corpus |
| `encode(text)` | Convert text to a list of token IDs |
| `decode(ids)` | Convert a list of token IDs back to text |
| `add_special_tokens(tokens)` | Add a list of special tokens to the vocabulary |
| `save(directory)` | Save the tokenizer to a directory |
| `load(directory)` | Load a tokenizer from a directory |

## Features

- Unicode-aware pre-tokenization optimized for Turkish text
- Byte Pair Encoding for efficient subword tokenization
- Support for custom special tokens
- Save and load trained tokenizers
- Minimal dependencies (only `regex`)

## License

MIT
