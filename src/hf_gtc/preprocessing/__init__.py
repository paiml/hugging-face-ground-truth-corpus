"""Preprocessing recipes for HuggingFace data pipelines.

This module provides utilities for text preprocessing,
tokenization, and data streaming.

Examples:
    >>> from hf_gtc.preprocessing import preprocess_text
    >>> preprocess_text("  Hello World  ")
    'hello world'
"""

from __future__ import annotations

from hf_gtc.preprocessing.datasets import (
    DatasetInfo,
    create_train_test_split,
    create_train_val_test_split,
    filter_by_length,
    get_dataset_info,
    rename_columns,
    sample_dataset,
    select_columns,
)
from hf_gtc.preprocessing.tokenization import (
    create_preprocessing_function,
    preprocess_text,
    tokenize_batch,
)

__all__ = [
    "DatasetInfo",
    "create_preprocessing_function",
    "create_train_test_split",
    "create_train_val_test_split",
    "filter_by_length",
    "get_dataset_info",
    "preprocess_text",
    "rename_columns",
    "sample_dataset",
    "select_columns",
    "tokenize_batch",
]
