"""
TurkToken - Türkçe uyumlu BPE Tokenizer
GPT-4 tarzı pre-tokenization ile Byte Pair Encoding
"""

from .tokenizer import TurkishBPETokenizer

__version__ = "0.1.0"
__all__ = ["TurkishBPETokenizer"]
