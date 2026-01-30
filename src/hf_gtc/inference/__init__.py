"""Inference recipes for HuggingFace models.

This module provides utilities for creating pipelines, batch inference,
device placement, and inference optimizations.

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
from hf_gtc.inference.optimization import (
    VALID_ATTENTION_IMPLS,
    VALID_CACHE_TYPES,
    VALID_KV_QUANT_TYPES,
    AttentionImplementation,
    ContinuousBatchingConfig,
    FlashAttentionConfig,
    KVCacheConfig,
    KVCacheType,
    QuantizedKVConfig,
    SpeculativeDecodingConfig,
    calculate_speculative_speedup,
    create_continuous_batching_config,
    create_flash_attention_config,
    create_kv_cache_config,
    create_quantized_kv_config,
    create_speculative_decoding_config,
    estimate_kv_cache_memory,
    get_recommended_attention,
    list_attention_implementations,
    list_kv_cache_types,
    validate_kv_cache_config,
    validate_speculative_config,
)
from hf_gtc.inference.pipelines import create_pipeline, list_supported_tasks

__all__: list[str] = [
    "VALID_ATTENTION_IMPLS",
    "VALID_CACHE_TYPES",
    "VALID_KV_QUANT_TYPES",
    "AttentionImplementation",
    "BatchConfig",
    "BatchResult",
    "BatchStats",
    "ContinuousBatchingConfig",
    "FlashAttentionConfig",
    "KVCacheConfig",
    "KVCacheType",
    "PaddingStrategy",
    "QuantizedKVConfig",
    "SpeculativeDecodingConfig",
    "calculate_speculative_speedup",
    "clear_gpu_memory",
    "compute_batch_stats",
    "compute_num_batches",
    "create_batches",
    "create_continuous_batching_config",
    "create_flash_attention_config",
    "create_kv_cache_config",
    "create_pipeline",
    "create_quantized_kv_config",
    "create_speculative_decoding_config",
    "estimate_kv_cache_memory",
    "estimate_memory_per_batch",
    "get_device",
    "get_device_map",
    "get_gpu_memory_info",
    "get_optimal_batch_size",
    "get_recommended_attention",
    "list_attention_implementations",
    "list_kv_cache_types",
    "list_padding_strategies",
    "list_supported_tasks",
    "validate_batch_config",
    "validate_kv_cache_config",
    "validate_speculative_config",
]
