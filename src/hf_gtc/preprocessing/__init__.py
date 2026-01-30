"""Preprocessing recipes for HuggingFace data pipelines.

This module provides utilities for text preprocessing,
tokenization, data streaming, and data augmentation.

Examples:
    >>> from hf_gtc.preprocessing import preprocess_text, StreamConfig, AugmentConfig
    >>> preprocess_text("  Hello World  ")
    'hello world'
    >>> config = StreamConfig(batch_size=500)
    >>> config.batch_size
    500
    >>> aug_config = AugmentConfig(probability=0.2)
    >>> aug_config.probability
    0.2
"""

from __future__ import annotations

from hf_gtc.preprocessing.augmentation import (
    AugmentationType,
    AugmentConfig,
    AugmentResult,
    augment_batch,
    augment_text,
    chain_augmentations,
    compute_augmentation_stats,
    create_augmenter,
    get_augmentation_type,
    list_augmentation_types,
    random_delete,
    random_insert,
    random_swap,
    synonym_replace,
    validate_augment_config,
    validate_augmentation_type,
)
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
    "AugmentConfig",
    "AugmentResult",
    "AugmentationType",
    "DatasetInfo",
    "ShuffleMode",
    "StreamConfig",
    "StreamProgress",
    "StreamStats",
    "augment_batch",
    "augment_text",
    "chain_augmentations",
    "compute_augmentation_stats",
    "compute_stream_stats",
    "create_augmenter",
    "create_preprocessing_function",
    "create_stream_iterator",
    "create_train_test_split",
    "create_train_val_test_split",
    "filter_by_length",
    "filter_stream",
    "get_augmentation_type",
    "get_dataset_info",
    "list_augmentation_types",
    "list_shuffle_modes",
    "map_stream",
    "preprocess_text",
    "random_delete",
    "random_insert",
    "random_swap",
    "rename_columns",
    "sample_dataset",
    "select_columns",
    "skip_stream",
    "stream_batches",
    "stream_dataset",
    "synonym_replace",
    "take_stream",
    "tokenize_batch",
    "validate_augment_config",
    "validate_augmentation_type",
    "validate_shuffle_mode",
    "validate_stream_config",
]
