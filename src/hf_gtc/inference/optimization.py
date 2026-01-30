"""Inference optimization utilities.

This module provides functions for configuring inference optimizations
including KV cache, flash attention, and speculative decoding.

Examples:
    >>> from hf_gtc.inference.optimization import create_kv_cache_config
    >>> config = create_kv_cache_config(max_batch_size=32)
    >>> config.max_batch_size
    32
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class AttentionImplementation(Enum):
    """Supported attention implementations.

    Attributes:
        EAGER: Standard eager attention.
        SDPA: Scaled dot-product attention (PyTorch 2.0+).
        FLASH_ATTENTION_2: Flash Attention 2.
        FLASH_ATTENTION_3: Flash Attention 3.

    Examples:
        >>> AttentionImplementation.FLASH_ATTENTION_2.value
        'flash_attention_2'
        >>> AttentionImplementation.SDPA.value
        'sdpa'
    """

    EAGER = "eager"
    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"


VALID_ATTENTION_IMPLS = frozenset(a.value for a in AttentionImplementation)


class KVCacheType(Enum):
    """Supported KV cache types.

    Attributes:
        DYNAMIC: Dynamic cache that grows with sequence.
        STATIC: Pre-allocated static cache.
        QUANTIZED: Quantized KV cache for memory savings.
        SLIDING_WINDOW: Sliding window cache for long sequences.

    Examples:
        >>> KVCacheType.DYNAMIC.value
        'dynamic'
        >>> KVCacheType.QUANTIZED.value
        'quantized'
    """

    DYNAMIC = "dynamic"
    STATIC = "static"
    QUANTIZED = "quantized"
    SLIDING_WINDOW = "sliding_window"


VALID_CACHE_TYPES = frozenset(c.value for c in KVCacheType)

# Quantization types for KV cache
KVQuantType = Literal["fp16", "fp8", "int8", "int4"]
VALID_KV_QUANT_TYPES = frozenset({"fp16", "fp8", "int8", "int4"})


@dataclass(frozen=True, slots=True)
class KVCacheConfig:
    """Configuration for KV cache.

    Attributes:
        cache_type: Type of KV cache.
        max_batch_size: Maximum batch size.
        max_seq_length: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.

    Examples:
        >>> config = KVCacheConfig(
        ...     cache_type=KVCacheType.DYNAMIC,
        ...     max_batch_size=32,
        ...     max_seq_length=2048,
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ... )
        >>> config.max_batch_size
        32
    """

    cache_type: KVCacheType
    max_batch_size: int
    max_seq_length: int
    num_layers: int
    num_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class FlashAttentionConfig:
    """Configuration for Flash Attention.

    Attributes:
        implementation: Attention implementation to use.
        sliding_window: Sliding window size (None for full attention).
        softmax_scale: Custom softmax scale (None for default).
        return_softmax: Whether to return softmax weights.

    Examples:
        >>> config = FlashAttentionConfig(
        ...     implementation=AttentionImplementation.FLASH_ATTENTION_2,
        ...     sliding_window=None,
        ...     softmax_scale=None,
        ...     return_softmax=False,
        ... )
        >>> config.implementation
        <AttentionImplementation.FLASH_ATTENTION_2: 'flash_attention_2'>
    """

    implementation: AttentionImplementation
    sliding_window: int | None
    softmax_scale: float | None
    return_softmax: bool


@dataclass(frozen=True, slots=True)
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding.

    Attributes:
        draft_model_id: Model ID for draft model.
        num_speculative_tokens: Number of tokens to speculate.
        acceptance_threshold: Threshold for accepting speculated tokens.
        use_assistant_model: Whether to use assistant model approach.

    Examples:
        >>> config = SpeculativeDecodingConfig(
        ...     draft_model_id="gpt2",
        ...     num_speculative_tokens=5,
        ...     acceptance_threshold=0.9,
        ...     use_assistant_model=True,
        ... )
        >>> config.num_speculative_tokens
        5
    """

    draft_model_id: str
    num_speculative_tokens: int
    acceptance_threshold: float
    use_assistant_model: bool


@dataclass(frozen=True, slots=True)
class QuantizedKVConfig:
    """Configuration for quantized KV cache.

    Attributes:
        quant_type: Quantization type for KV cache.
        residual_length: Length of residual (unquantized) cache.
        compute_dtype: Dtype for attention computation.

    Examples:
        >>> config = QuantizedKVConfig(
        ...     quant_type="fp8",
        ...     residual_length=128,
        ...     compute_dtype="float16",
        ... )
        >>> config.quant_type
        'fp8'
    """

    quant_type: KVQuantType
    residual_length: int
    compute_dtype: str


@dataclass(frozen=True, slots=True)
class ContinuousBatchingConfig:
    """Configuration for continuous batching.

    Attributes:
        max_num_seqs: Maximum number of sequences.
        max_num_batched_tokens: Maximum tokens per batch.
        max_paddings: Maximum padding tokens allowed.
        preemption_mode: Preemption mode for memory management.

    Examples:
        >>> config = ContinuousBatchingConfig(
        ...     max_num_seqs=256,
        ...     max_num_batched_tokens=8192,
        ...     max_paddings=256,
        ...     preemption_mode="swap",
        ... )
        >>> config.max_num_seqs
        256
    """

    max_num_seqs: int
    max_num_batched_tokens: int
    max_paddings: int
    preemption_mode: str


def validate_kv_cache_config(config: KVCacheConfig) -> None:
    """Validate KV cache configuration.

    Args:
        config: KV cache configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = KVCacheConfig(
        ...     KVCacheType.DYNAMIC, 32, 2048, 32, 32, 128
        ... )
        >>> validate_kv_cache_config(config)  # No error

        >>> bad = KVCacheConfig(KVCacheType.DYNAMIC, 0, 2048, 32, 32, 128)
        >>> validate_kv_cache_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_batch_size must be positive
    """
    if config.max_batch_size <= 0:
        msg = f"max_batch_size must be positive, got {config.max_batch_size}"
        raise ValueError(msg)

    if config.max_seq_length <= 0:
        msg = f"max_seq_length must be positive, got {config.max_seq_length}"
        raise ValueError(msg)

    if config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)

    if config.num_heads <= 0:
        msg = f"num_heads must be positive, got {config.num_heads}"
        raise ValueError(msg)

    if config.head_dim <= 0:
        msg = f"head_dim must be positive, got {config.head_dim}"
        raise ValueError(msg)


def validate_speculative_config(config: SpeculativeDecodingConfig) -> None:
    """Validate speculative decoding configuration.

    Args:
        config: Speculative decoding configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SpeculativeDecodingConfig("gpt2", 5, 0.9, True)
        >>> validate_speculative_config(config)  # No error

        >>> bad = SpeculativeDecodingConfig("", 5, 0.9, True)
        >>> validate_speculative_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: draft_model_id cannot be empty
    """
    if not config.draft_model_id:
        msg = "draft_model_id cannot be empty"
        raise ValueError(msg)

    if config.num_speculative_tokens <= 0:
        msg = (
            f"num_speculative_tokens must be positive, "
            f"got {config.num_speculative_tokens}"
        )
        raise ValueError(msg)

    if not 0.0 <= config.acceptance_threshold <= 1.0:
        msg = (
            f"acceptance_threshold must be between 0.0 and 1.0, "
            f"got {config.acceptance_threshold}"
        )
        raise ValueError(msg)


def create_kv_cache_config(
    cache_type: str = "dynamic",
    max_batch_size: int = 32,
    max_seq_length: int = 2048,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
) -> KVCacheConfig:
    """Create a KV cache configuration.

    Args:
        cache_type: Type of cache. Defaults to "dynamic".
        max_batch_size: Maximum batch size. Defaults to 32.
        max_seq_length: Maximum sequence length. Defaults to 2048.
        num_layers: Number of layers. Defaults to 32.
        num_heads: Number of attention heads. Defaults to 32.
        head_dim: Head dimension. Defaults to 128.

    Returns:
        KVCacheConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_kv_cache_config(max_batch_size=64)
        >>> config.max_batch_size
        64

        >>> config = create_kv_cache_config(cache_type="static")
        >>> config.cache_type
        <KVCacheType.STATIC: 'static'>

        >>> create_kv_cache_config(max_batch_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_batch_size must be positive
    """
    if cache_type not in VALID_CACHE_TYPES:
        msg = f"cache_type must be one of {VALID_CACHE_TYPES}, got '{cache_type}'"
        raise ValueError(msg)

    config = KVCacheConfig(
        cache_type=KVCacheType(cache_type),
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    validate_kv_cache_config(config)
    return config


def create_flash_attention_config(
    implementation: str = "flash_attention_2",
    sliding_window: int | None = None,
    softmax_scale: float | None = None,
    return_softmax: bool = False,
) -> FlashAttentionConfig:
    """Create a Flash Attention configuration.

    Args:
        implementation: Attention implementation. Defaults to "flash_attention_2".
        sliding_window: Sliding window size. Defaults to None.
        softmax_scale: Custom softmax scale. Defaults to None.
        return_softmax: Whether to return softmax. Defaults to False.

    Returns:
        FlashAttentionConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_flash_attention_config()
        >>> config.implementation
        <AttentionImplementation.FLASH_ATTENTION_2: 'flash_attention_2'>

        >>> config = create_flash_attention_config(sliding_window=4096)
        >>> config.sliding_window
        4096

        >>> create_flash_attention_config(implementation="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: implementation must be one of
    """
    if implementation not in VALID_ATTENTION_IMPLS:
        msg = (
            f"implementation must be one of {VALID_ATTENTION_IMPLS}, "
            f"got '{implementation}'"
        )
        raise ValueError(msg)

    if sliding_window is not None and sliding_window <= 0:
        msg = f"sliding_window must be positive if set, got {sliding_window}"
        raise ValueError(msg)

    return FlashAttentionConfig(
        implementation=AttentionImplementation(implementation),
        sliding_window=sliding_window,
        softmax_scale=softmax_scale,
        return_softmax=return_softmax,
    )


def create_speculative_decoding_config(
    draft_model_id: str,
    num_speculative_tokens: int = 5,
    acceptance_threshold: float = 0.9,
    use_assistant_model: bool = True,
) -> SpeculativeDecodingConfig:
    """Create a speculative decoding configuration.

    Args:
        draft_model_id: Model ID for draft model.
        num_speculative_tokens: Number of tokens to speculate. Defaults to 5.
        acceptance_threshold: Acceptance threshold. Defaults to 0.9.
        use_assistant_model: Whether to use assistant model. Defaults to True.

    Returns:
        SpeculativeDecodingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_speculative_decoding_config("gpt2")
        >>> config.draft_model_id
        'gpt2'

        >>> config = create_speculative_decoding_config(
        ...     "gpt2", num_speculative_tokens=8
        ... )
        >>> config.num_speculative_tokens
        8

        >>> create_speculative_decoding_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: draft_model_id cannot be empty
    """
    config = SpeculativeDecodingConfig(
        draft_model_id=draft_model_id,
        num_speculative_tokens=num_speculative_tokens,
        acceptance_threshold=acceptance_threshold,
        use_assistant_model=use_assistant_model,
    )
    validate_speculative_config(config)
    return config


def create_quantized_kv_config(
    quant_type: KVQuantType = "fp8",
    residual_length: int = 128,
    compute_dtype: str = "float16",
) -> QuantizedKVConfig:
    """Create a quantized KV cache configuration.

    Args:
        quant_type: Quantization type. Defaults to "fp8".
        residual_length: Residual cache length. Defaults to 128.
        compute_dtype: Compute dtype. Defaults to "float16".

    Returns:
        QuantizedKVConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_quantized_kv_config(quant_type="int8")
        >>> config.quant_type
        'int8'

        >>> create_quantized_kv_config(quant_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quant_type must be one of
    """
    if quant_type not in VALID_KV_QUANT_TYPES:
        msg = f"quant_type must be one of {VALID_KV_QUANT_TYPES}, got '{quant_type}'"
        raise ValueError(msg)

    if residual_length < 0:
        msg = f"residual_length must be non-negative, got {residual_length}"
        raise ValueError(msg)

    return QuantizedKVConfig(
        quant_type=quant_type,
        residual_length=residual_length,
        compute_dtype=compute_dtype,
    )


def create_continuous_batching_config(
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    max_paddings: int = 256,
    preemption_mode: str = "swap",
) -> ContinuousBatchingConfig:
    """Create a continuous batching configuration.

    Args:
        max_num_seqs: Maximum sequences. Defaults to 256.
        max_num_batched_tokens: Maximum batched tokens. Defaults to 8192.
        max_paddings: Maximum paddings. Defaults to 256.
        preemption_mode: Preemption mode. Defaults to "swap".

    Returns:
        ContinuousBatchingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_continuous_batching_config(max_num_seqs=128)
        >>> config.max_num_seqs
        128

        >>> create_continuous_batching_config(max_num_seqs=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_num_seqs must be positive
    """
    if max_num_seqs <= 0:
        msg = f"max_num_seqs must be positive, got {max_num_seqs}"
        raise ValueError(msg)

    if max_num_batched_tokens <= 0:
        msg = f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}"
        raise ValueError(msg)

    if max_paddings < 0:
        msg = f"max_paddings must be non-negative, got {max_paddings}"
        raise ValueError(msg)

    valid_modes = {"swap", "recompute"}
    if preemption_mode not in valid_modes:
        msg = f"preemption_mode must be one of {valid_modes}, got '{preemption_mode}'"
        raise ValueError(msg)

    return ContinuousBatchingConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_paddings=max_paddings,
        preemption_mode=preemption_mode,
    )


def list_attention_implementations() -> list[str]:
    """List supported attention implementations.

    Returns:
        Sorted list of implementation names.

    Examples:
        >>> impls = list_attention_implementations()
        >>> "flash_attention_2" in impls
        True
        >>> "sdpa" in impls
        True
        >>> impls == sorted(impls)
        True
    """
    return sorted(VALID_ATTENTION_IMPLS)


def list_kv_cache_types() -> list[str]:
    """List supported KV cache types.

    Returns:
        Sorted list of cache type names.

    Examples:
        >>> types = list_kv_cache_types()
        >>> "dynamic" in types
        True
        >>> "static" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CACHE_TYPES)


def estimate_kv_cache_memory(
    config: KVCacheConfig,
    dtype_bytes: int = 2,
) -> int:
    """Estimate KV cache memory usage.

    Args:
        config: KV cache configuration.
        dtype_bytes: Bytes per element. Defaults to 2 (FP16).

    Returns:
        Estimated memory in bytes.

    Examples:
        >>> config = create_kv_cache_config(
        ...     max_batch_size=1,
        ...     max_seq_length=2048,
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ... )
        >>> mem = estimate_kv_cache_memory(config)
        >>> mem > 0
        True
    """
    # KV cache: 2 (K + V) * layers * batch * seq * heads * head_dim * dtype
    kv_size = (
        2
        * config.num_layers
        * config.max_batch_size
        * config.max_seq_length
        * config.num_heads
        * config.head_dim
    )
    return kv_size * dtype_bytes


def get_recommended_attention(
    model_size: str,
    has_flash_attention: bool = True,
) -> AttentionImplementation:
    """Get recommended attention implementation.

    Args:
        model_size: Model size category ("small", "medium", "large", "xlarge").
        has_flash_attention: Whether Flash Attention is available.

    Returns:
        Recommended AttentionImplementation.

    Raises:
        ValueError: If model_size is invalid.

    Examples:
        >>> get_recommended_attention("large", has_flash_attention=True)
        <AttentionImplementation.FLASH_ATTENTION_2: 'flash_attention_2'>

        >>> get_recommended_attention("small", has_flash_attention=False)
        <AttentionImplementation.SDPA: 'sdpa'>
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    if has_flash_attention and model_size in ("large", "xlarge"):
        return AttentionImplementation.FLASH_ATTENTION_2

    if model_size in ("medium", "large", "xlarge"):
        return AttentionImplementation.SDPA

    return AttentionImplementation.EAGER


def calculate_speculative_speedup(
    acceptance_rate: float,
    num_speculative_tokens: int,
    draft_latency_ratio: float = 0.1,
) -> float:
    """Calculate theoretical speedup from speculative decoding.

    Args:
        acceptance_rate: Rate at which speculated tokens are accepted.
        num_speculative_tokens: Number of tokens speculated per step.
        draft_latency_ratio: Draft model latency as ratio of main model.

    Returns:
        Theoretical speedup factor.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> speedup = calculate_speculative_speedup(0.8, 5, 0.1)
        >>> speedup > 1.0
        True

        >>> calculate_speculative_speedup(1.5, 5, 0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: acceptance_rate must be between 0.0 and 1.0
    """
    if not 0.0 <= acceptance_rate <= 1.0:
        msg = f"acceptance_rate must be between 0.0 and 1.0, got {acceptance_rate}"
        raise ValueError(msg)

    if num_speculative_tokens <= 0:
        msg = (
            f"num_speculative_tokens must be positive, got {num_speculative_tokens}"
        )
        raise ValueError(msg)

    if not 0.0 < draft_latency_ratio < 1.0:
        msg = (
            f"draft_latency_ratio must be between 0.0 and 1.0 (exclusive), "
            f"got {draft_latency_ratio}"
        )
        raise ValueError(msg)

    # Expected tokens per step with speculation
    expected_tokens = 1 + acceptance_rate * num_speculative_tokens

    # Latency per step (1 main forward + k draft forwards)
    latency_per_step = 1.0 + num_speculative_tokens * draft_latency_ratio

    # Speedup = tokens / latency
    return expected_tokens / latency_per_step
