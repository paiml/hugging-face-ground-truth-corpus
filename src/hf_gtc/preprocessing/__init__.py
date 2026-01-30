"""Preprocessing recipes for HuggingFace data pipelines.

This module provides utilities for text preprocessing,
tokenization, and data streaming.

Examples:
    >>> from hf_gtc.preprocessing import preprocess_text, StreamConfig
    >>> preprocess_text("  Hello World  ")
    'hello world'
    >>> config = StreamConfig(batch_size=500)
    >>> config.batch_size
    500
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
from hf_gtc.preprocessing.streaming import (
    ShuffleMode,
    StreamConfig,
    StreamProgress,
    StreamStats,
    compute_stream_stats,
    create_stream_iterator,
    filter_stream,
    list_shuffle_modes,
    map_stream,
    skip_stream,
    stream_batches,
    stream_dataset,
    take_stream,
    validate_shuffle_mode,
    validate_stream_config,
)
from hf_gtc.preprocessing.tokenization import (
    create_preprocessing_function,
    preprocess_text,
    tokenize_batch,
)

__all__ = [
    "DatasetInfo",
    "ShuffleMode",
    "StreamConfig",
    "StreamProgress",
    "StreamStats",
    "compute_stream_stats",
    "create_preprocessing_function",
    "create_stream_iterator",
    "create_train_test_split",
    "create_train_val_test_split",
    "filter_by_length",
    "filter_stream",
    "get_dataset_info",
    "list_shuffle_modes",
    "map_stream",
    "preprocess_text",
    "rename_columns",
    "sample_dataset",
    "select_columns",
    "skip_stream",
    "stream_batches",
    "stream_dataset",
    "take_stream",
    "tokenize_batch",
    "validate_shuffle_mode",
    "validate_stream_config",
]
