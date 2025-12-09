---
title: TurkToken
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# TurkToken

**Turkish-optimized Byte Pair Encoding (BPE) Tokenizer**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/omertarikyilmaz/turktoken)
[![PyPI version](https://badge.fury.io/py/turktoken.svg)](https://pypi.org/project/turktoken/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

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

## Features

- Unicode-aware pre-tokenization optimized for Turkish text
- Byte Pair Encoding for efficient subword tokenization
- Support for custom special tokens
- Save and load trained tokenizers
- Minimal dependencies (only `regex`)

## Links

- [PyPI Package](https://pypi.org/project/turktoken/)
- [GitHub Repository](https://github.com/hsperus/turktoken)
- [Try on Hugging Face](https://huggingface.co/spaces/omertarikyilmaz/turktoken)

---

<div align="center">

## Quick Example

</div>

```python
from turktoken import TurkishBPETokenizer

# Initialize and train
tokenizer = TurkishBPETokenizer()
tokenizer.train("Türkiye'nin başkenti Ankara'dır. İstanbul en büyük şehirdir.", vocab_size=512)

# Encode
ids = tokenizer.encode("Merhaba Türkiye!")
print(f"Token IDs: {ids}")

# Decode
text = tokenizer.decode(ids)
print(f"Decoded: {text}")

# Add special tokens
tokenizer.add_special_tokens(["<|bos|>", "<|eos|>"])

# Save & Load
tokenizer.save("./my_tokenizer")
tokenizer.load("./my_tokenizer")
```

---

<p align="center">
  <b>License:</b> MIT
</p>
