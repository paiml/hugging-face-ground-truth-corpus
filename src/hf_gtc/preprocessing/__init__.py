"""Preprocessing recipes for HuggingFace data pipelines.

This module provides utilities for text preprocessing,
tokenization, and data streaming.

Examples:
    >>> from hf_gtc.preprocessing import preprocess_text
    >>> preprocess_text("  Hello World  ")
    'hello world'
"""

from __future__ import annotations

from hf_gtc.preprocessing.tokenization import (
    create_preprocessing_function,
    preprocess_text,
    tokenize_batch,
)

__all__ = [
    "create_preprocessing_function",
    "preprocess_text",
    "tokenize_batch",
]
