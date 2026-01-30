"""Inference recipes for HuggingFace models.

This module provides utilities for creating pipelines, batch inference,
and managing device placement for inference.

Examples:
    >>> from hf_gtc.inference import get_device, BatchConfig
    >>> device = get_device()
    >>> device in ("cuda", "mps", "cpu")
    True
    >>> config = BatchConfig(batch_size=32)
    >>> config.batch_size
    32
"""

from __future__ import annotations

from hf_gtc.inference.batch import (
    BatchConfig,
    BatchResult,
    BatchStats,
    PaddingStrategy,
    compute_batch_stats,
    compute_num_batches,
    create_batches,
    estimate_memory_per_batch,
    get_optimal_batch_size,
    list_padding_strategies,
    validate_batch_config,
)
from hf_gtc.inference.device import (
    clear_gpu_memory,
    get_device,
    get_device_map,
    get_gpu_memory_info,
)
from hf_gtc.inference.pipelines import create_pipeline, list_supported_tasks

__all__ = [
    "BatchConfig",
    "BatchResult",
    "BatchStats",
    "PaddingStrategy",
    "clear_gpu_memory",
    "compute_batch_stats",
    "compute_num_batches",
    "create_batches",
    "create_pipeline",
    "estimate_memory_per_batch",
    "get_device",
    "get_device_map",
    "get_gpu_memory_info",
    "get_optimal_batch_size",
    "list_padding_strategies",
    "list_supported_tasks",
    "validate_batch_config",
]
