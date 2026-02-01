"""Attention mechanisms utilities for transformer models.

This module provides utilities for configuring and analyzing attention
mechanisms including Flash Attention, Multi-Query, and Grouped-Query Attention.

Examples:
    >>> from hf_gtc.models.attention import AttentionType, create_attention_config
    >>> config = create_attention_config(attention_type="flash", num_heads=32)
    >>> config.attention_type
    <AttentionType.FLASH: 'flash'>
    >>> config.num_heads
    32
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class AttentionType(Enum):
    """Attention mechanism types.

    Attributes:
        STANDARD: Standard scaled dot-product attention.
        FLASH: Flash Attention v1.
        FLASH_V2: Flash Attention v2 with improved performance.
        MULTI_QUERY: Multi-Query Attention (MQA).
        GROUPED_QUERY: Grouped-Query Attention (GQA).
        LINEAR: Linear attention approximation.
        SPARSE: Sparse attention patterns.

    Examples:
        >>> AttentionType.STANDARD.value
        'standard'
        >>> AttentionType.FLASH.value
        'flash'
        >>> AttentionType.GROUPED_QUERY.value
        'grouped_query'
    """

    STANDARD = "standard"
    FLASH = "flash"
    FLASH_V2 = "flash_v2"
    MULTI_QUERY = "multi_query"
    GROUPED_QUERY = "grouped_query"
    LINEAR = "linear"
    SPARSE = "sparse"


VALID_ATTENTION_TYPES = frozenset(t.value for t in AttentionType)


class AttentionMask(Enum):
    """Attention mask types.

    Attributes:
        CAUSAL: Causal (autoregressive) masking.
        BIDIRECTIONAL: No masking, full attention.
        PREFIX: Prefix LM masking (bidirectional on prefix, causal on rest).
        CUSTOM: Custom user-defined mask.

    Examples:
        >>> AttentionMask.CAUSAL.value
        'causal'
        >>> AttentionMask.BIDIRECTIONAL.value
        'bidirectional'
    """

    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"
    PREFIX = "prefix"
    CUSTOM = "custom"


VALID_ATTENTION_MASKS = frozenset(m.value for m in AttentionMask)


class AttentionImplementation(Enum):
    """Attention implementation backends.

    Attributes:
        EAGER: Eager (manual) implementation.
        SDPA: PyTorch scaled_dot_product_attention.
        FLASH_ATTENTION_2: Flash Attention 2 library.

    Examples:
        >>> AttentionImplementation.EAGER.value
        'eager'
        >>> AttentionImplementation.SDPA.value
        'sdpa'
        >>> AttentionImplementation.FLASH_ATTENTION_2.value
        'flash_attention_2'
    """

    EAGER = "eager"
    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"


VALID_ATTENTION_IMPLEMENTATIONS = frozenset(i.value for i in AttentionImplementation)


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Configuration for attention mechanism.

    Attributes:
        attention_type: Type of attention mechanism.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        dropout: Dropout probability.
        use_bias: Whether to use bias in projections.

    Examples:
        >>> config = AttentionConfig(
        ...     attention_type=AttentionType.STANDARD,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     dropout=0.0,
        ...     use_bias=True,
        ... )
        >>> config.num_heads
        32
        >>> config.head_dim
        128
    """

    attention_type: AttentionType
    num_heads: int
    head_dim: int
    dropout: float
    use_bias: bool


@dataclass(frozen=True, slots=True)
class FlashAttentionConfig:
    """Configuration for Flash Attention.

    Attributes:
        window_size: Sliding window size for local attention (-1 for global).
        causal: Whether to use causal masking.
        softmax_scale: Custom softmax scale (None for default 1/sqrt(head_dim)).

    Examples:
        >>> config = FlashAttentionConfig(
        ...     window_size=-1,
        ...     causal=True,
        ...     softmax_scale=None,
        ... )
        >>> config.causal
        True
        >>> config.window_size
        -1
    """

    window_size: int
    causal: bool
    softmax_scale: float | None


@dataclass(frozen=True, slots=True)
class GroupedQueryConfig:
    """Configuration for Grouped-Query Attention (GQA).

    Attributes:
        num_kv_heads: Number of key-value heads.
        num_query_groups: Number of query groups sharing KV heads.

    Examples:
        >>> config = GroupedQueryConfig(
        ...     num_kv_heads=8,
        ...     num_query_groups=4,
        ... )
        >>> config.num_kv_heads
        8
        >>> config.num_query_groups
        4
    """

    num_kv_heads: int
    num_query_groups: int


@dataclass(frozen=True, slots=True)
class AttentionStats:
    """Statistics for attention computation.

    Attributes:
        memory_usage_mb: Memory usage in megabytes.
        flops: Floating point operations.
        throughput: Throughput in tokens per second.

    Examples:
        >>> stats = AttentionStats(
        ...     memory_usage_mb=1024.0,
        ...     flops=1e12,
        ...     throughput=10000.0,
        ... )
        >>> stats.memory_usage_mb
        1024.0
    """

    memory_usage_mb: float
    flops: float
    throughput: float


def validate_attention_config(config: AttentionConfig) -> None:
    """Validate attention configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_heads is not positive.
        ValueError: If head_dim is not positive.
        ValueError: If dropout is not between 0 and 1.

    Examples:
        >>> config = AttentionConfig(
        ...     AttentionType.STANDARD, 32, 128, 0.0, True
        ... )
        >>> validate_attention_config(config)  # No error

        >>> validate_attention_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AttentionConfig(AttentionType.STANDARD, 0, 128, 0.0, True)
        >>> validate_attention_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_heads must be positive
    """
    validate_not_none(config, "config")

    if config.num_heads <= 0:
        msg = f"num_heads must be positive, got {config.num_heads}"
        raise ValueError(msg)

    if config.head_dim <= 0:
        msg = f"head_dim must be positive, got {config.head_dim}"
        raise ValueError(msg)

    if not 0 <= config.dropout <= 1:
        msg = f"dropout must be between 0 and 1, got {config.dropout}"
        raise ValueError(msg)


def validate_flash_attention_config(config: FlashAttentionConfig) -> None:
    """Validate Flash Attention configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If window_size is invalid.
        ValueError: If softmax_scale is not positive when set.

    Examples:
        >>> config = FlashAttentionConfig(-1, True, None)
        >>> validate_flash_attention_config(config)  # No error

        >>> validate_flash_attention_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = FlashAttentionConfig(-2, True, None)
        >>> validate_flash_attention_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be -1 or positive
    """
    validate_not_none(config, "config")

    if config.window_size < -1 or config.window_size == 0:
        msg = f"window_size must be -1 or positive, got {config.window_size}"
        raise ValueError(msg)

    if config.softmax_scale is not None and config.softmax_scale <= 0:
        msg = f"softmax_scale must be positive, got {config.softmax_scale}"
        raise ValueError(msg)


def validate_grouped_query_config(config: GroupedQueryConfig) -> None:
    """Validate Grouped-Query Attention configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_kv_heads is not positive.
        ValueError: If num_query_groups is not positive.

    Examples:
        >>> config = GroupedQueryConfig(8, 4)
        >>> validate_grouped_query_config(config)  # No error

        >>> validate_grouped_query_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = GroupedQueryConfig(0, 4)
        >>> validate_grouped_query_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_kv_heads must be positive
    """
    validate_not_none(config, "config")

    if config.num_kv_heads <= 0:
        msg = f"num_kv_heads must be positive, got {config.num_kv_heads}"
        raise ValueError(msg)

    if config.num_query_groups <= 0:
        msg = f"num_query_groups must be positive, got {config.num_query_groups}"
        raise ValueError(msg)


def create_attention_config(
    attention_type: AttentionType | str = AttentionType.STANDARD,
    num_heads: int = 32,
    head_dim: int = 128,
    dropout: float = 0.0,
    use_bias: bool = True,
) -> AttentionConfig:
    """Create an attention configuration.

    Args:
        attention_type: Type of attention mechanism. Defaults to STANDARD.
        num_heads: Number of attention heads. Defaults to 32.
        head_dim: Dimension per head. Defaults to 128.
        dropout: Dropout probability. Defaults to 0.0.
        use_bias: Whether to use bias. Defaults to True.

    Returns:
        Validated AttentionConfig instance.

    Raises:
        ValueError: If attention_type is invalid.
        ValueError: If num_heads is not positive.

    Examples:
        >>> config = create_attention_config(attention_type="flash", num_heads=16)
        >>> config.attention_type
        <AttentionType.FLASH: 'flash'>
        >>> config.num_heads
        16

        >>> attn_type = AttentionType.GROUPED_QUERY
        >>> config2 = create_attention_config(attention_type=attn_type)
        >>> config2.attention_type
        <AttentionType.GROUPED_QUERY: 'grouped_query'>
    """
    if isinstance(attention_type, str):
        attention_type = get_attention_type(attention_type)

    config = AttentionConfig(
        attention_type=attention_type,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=dropout,
        use_bias=use_bias,
    )
    validate_attention_config(config)
    return config


def create_flash_attention_config(
    window_size: int = -1,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> FlashAttentionConfig:
    """Create a Flash Attention configuration.

    Args:
        window_size: Sliding window size (-1 for global). Defaults to -1.
        causal: Whether to use causal masking. Defaults to True.
        softmax_scale: Custom softmax scale. Defaults to None.

    Returns:
        Validated FlashAttentionConfig instance.

    Raises:
        ValueError: If window_size is invalid.

    Examples:
        >>> config = create_flash_attention_config(window_size=4096, causal=True)
        >>> config.window_size
        4096
        >>> config.causal
        True

        >>> config2 = create_flash_attention_config()
        >>> config2.window_size
        -1
    """
    config = FlashAttentionConfig(
        window_size=window_size,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    validate_flash_attention_config(config)
    return config


def create_grouped_query_config(
    num_kv_heads: int = 8,
    num_query_groups: int = 4,
) -> GroupedQueryConfig:
    """Create a Grouped-Query Attention configuration.

    Args:
        num_kv_heads: Number of key-value heads. Defaults to 8.
        num_query_groups: Number of query groups. Defaults to 4.

    Returns:
        Validated GroupedQueryConfig instance.

    Raises:
        ValueError: If num_kv_heads is not positive.

    Examples:
        >>> config = create_grouped_query_config(num_kv_heads=4, num_query_groups=8)
        >>> config.num_kv_heads
        4
        >>> config.num_query_groups
        8
    """
    config = GroupedQueryConfig(
        num_kv_heads=num_kv_heads,
        num_query_groups=num_query_groups,
    )
    validate_grouped_query_config(config)
    return config


def list_attention_types() -> list[str]:
    """List available attention types.

    Returns:
        Sorted list of attention type names.

    Examples:
        >>> types = list_attention_types()
        >>> "standard" in types
        True
        >>> "flash" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(t.value for t in AttentionType)


def list_attention_masks() -> list[str]:
    """List available attention mask types.

    Returns:
        Sorted list of mask type names.

    Examples:
        >>> masks = list_attention_masks()
        >>> "causal" in masks
        True
        >>> "bidirectional" in masks
        True
        >>> masks == sorted(masks)
        True
    """
    return sorted(m.value for m in AttentionMask)


def list_attention_implementations() -> list[str]:
    """List available attention implementations.

    Returns:
        Sorted list of implementation names.

    Examples:
        >>> impls = list_attention_implementations()
        >>> "eager" in impls
        True
        >>> "sdpa" in impls
        True
        >>> impls == sorted(impls)
        True
    """
    return sorted(i.value for i in AttentionImplementation)


def get_attention_type(name: str) -> AttentionType:
    """Get attention type enum from string.

    Args:
        name: Type name.

    Returns:
        AttentionType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_attention_type("flash")
        <AttentionType.FLASH: 'flash'>
        >>> get_attention_type("grouped_query")
        <AttentionType.GROUPED_QUERY: 'grouped_query'>

        >>> get_attention_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: attention type must be one of ...
    """
    for t in AttentionType:
        if t.value == name:
            return t
    msg = f"attention type must be one of {VALID_ATTENTION_TYPES}, got {name}"
    raise ValueError(msg)


def get_attention_mask(name: str) -> AttentionMask:
    """Get attention mask enum from string.

    Args:
        name: Mask name.

    Returns:
        AttentionMask enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_attention_mask("causal")
        <AttentionMask.CAUSAL: 'causal'>
        >>> get_attention_mask("bidirectional")
        <AttentionMask.BIDIRECTIONAL: 'bidirectional'>

        >>> get_attention_mask("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: attention mask must be one of ...
    """
    for m in AttentionMask:
        if m.value == name:
            return m
    msg = f"attention mask must be one of {VALID_ATTENTION_MASKS}, got {name}"
    raise ValueError(msg)


def get_attention_implementation(name: str) -> AttentionImplementation:
    """Get attention implementation enum from string.

    Args:
        name: Implementation name.

    Returns:
        AttentionImplementation enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_attention_implementation("sdpa")
        <AttentionImplementation.SDPA: 'sdpa'>
        >>> get_attention_implementation("flash_attention_2")
        <AttentionImplementation.FLASH_ATTENTION_2: 'flash_attention_2'>

        >>> get_attention_implementation("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: attention implementation must be one of ...
    """
    for i in AttentionImplementation:
        if i.value == name:
            return i
    valid = VALID_ATTENTION_IMPLEMENTATIONS
    msg = f"attention implementation must be one of {valid}, got {name}"
    raise ValueError(msg)


def calculate_attention_memory(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> float:
    """Calculate memory usage for attention computation.

    Computes the memory needed for Q, K, V matrices and attention scores.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32). Defaults to 2.

    Returns:
        Memory usage in megabytes.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> mem = calculate_attention_memory(1, 2048, 32, 128)
        >>> mem > 0
        True
        >>> isinstance(mem, float)
        True

        >>> calculate_attention_memory(0, 2048, 32, 128)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive, got 0
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if seq_len <= 0:
        msg = f"seq_len must be positive, got {seq_len}"
        raise ValueError(msg)
    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)
    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)
    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    # Q, K, V each: batch_size * num_heads * seq_len * head_dim
    qkv_size = 3 * batch_size * num_heads * seq_len * head_dim * dtype_bytes

    # Attention scores: batch_size * num_heads * seq_len * seq_len
    attn_scores_size = batch_size * num_heads * seq_len * seq_len * dtype_bytes

    # Output: batch_size * num_heads * seq_len * head_dim
    output_size = batch_size * num_heads * seq_len * head_dim * dtype_bytes

    total_bytes = qkv_size + attn_scores_size + output_size
    return total_bytes / (1024 * 1024)


def estimate_attention_flops(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> float:
    """Estimate FLOPs for attention computation.

    Computes approximate floating point operations for scaled dot-product attention.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Estimated FLOPs.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> flops = estimate_attention_flops(1, 2048, 32, 128)
        >>> flops > 0
        True
        >>> isinstance(flops, float)
        True

        >>> estimate_attention_flops(0, 2048, 32, 128)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive, got 0
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if seq_len <= 0:
        msg = f"seq_len must be positive, got {seq_len}"
        raise ValueError(msg)
    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)
    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)

    # QK^T matmul: 2 * batch * heads * seq * seq * head_dim
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    # Softmax: ~5 ops per element (exp, sum, div, max, sub)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len

    # Attention * V: 2 * batch * heads * seq * head_dim * seq
    av_flops = 2 * batch_size * num_heads * seq_len * head_dim * seq_len

    return float(qk_flops + softmax_flops + av_flops)


def select_attention_implementation(
    seq_len: int,
    head_dim: int,
    has_flash_attention: bool = False,
    has_sdpa: bool = True,
) -> AttentionImplementation:
    """Select optimal attention implementation based on parameters.

    Chooses the best available implementation considering sequence length,
    head dimension, and available backends.

    Args:
        seq_len: Sequence length.
        head_dim: Dimension per head.
        has_flash_attention: Whether Flash Attention 2 is available. Defaults to False.
        has_sdpa: Whether PyTorch SDPA is available. Defaults to True.

    Returns:
        Recommended AttentionImplementation.

    Raises:
        ValueError: If seq_len is not positive.
        ValueError: If head_dim is not positive.

    Examples:
        >>> impl = select_attention_implementation(2048, 128, has_flash_attention=True)
        >>> impl
        <AttentionImplementation.FLASH_ATTENTION_2: 'flash_attention_2'>

        >>> impl2 = select_attention_implementation(512, 64)
        >>> impl2
        <AttentionImplementation.SDPA: 'sdpa'>

        >>> impl3 = select_attention_implementation(256, 64, has_sdpa=False)
        >>> impl3
        <AttentionImplementation.EAGER: 'eager'>
    """
    if seq_len <= 0:
        msg = f"seq_len must be positive, got {seq_len}"
        raise ValueError(msg)
    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)

    # Flash Attention 2 is best for long sequences when available
    # and head_dim is a multiple of 8
    if has_flash_attention and head_dim % 8 == 0:
        return AttentionImplementation.FLASH_ATTENTION_2

    # SDPA is good general-purpose implementation
    if has_sdpa:
        return AttentionImplementation.SDPA

    # Fall back to eager implementation
    return AttentionImplementation.EAGER


def calculate_kv_cache_size(
    batch_size: int,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> float:
    """Calculate KV cache size for inference.

    Computes the memory needed for key-value cache across all layers.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
        dtype_bytes: Bytes per element. Defaults to 2 (fp16).

    Returns:
        KV cache size in megabytes.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> size = calculate_kv_cache_size(1, 4096, 32, 8, 128)
        >>> size > 0
        True
        >>> isinstance(size, float)
        True

        >>> calculate_kv_cache_size(0, 4096, 32, 8, 128)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive, got 0
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if max_seq_len <= 0:
        msg = f"max_seq_len must be positive, got {max_seq_len}"
        raise ValueError(msg)
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)
    if num_kv_heads <= 0:
        msg = f"num_kv_heads must be positive, got {num_kv_heads}"
        raise ValueError(msg)
    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)
    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    # K and V caches: 2 * layers * batch * seq * kv_heads * head_dim
    kv_cache_elements = (
        2 * num_layers * batch_size * max_seq_len * num_kv_heads * head_dim
    )
    kv_cache_bytes = kv_cache_elements * dtype_bytes

    return kv_cache_bytes / (1024 * 1024)


def format_attention_stats(stats: AttentionStats) -> str:
    """Format attention statistics for display.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = AttentionStats(1024.0, 1e12, 10000.0)
        >>> formatted = format_attention_stats(stats)
        >>> "Memory:" in formatted
        True
        >>> "FLOPs:" in formatted
        True
        >>> "Throughput:" in formatted
        True
    """
    validate_not_none(stats, "stats")

    def format_flops(flops: float) -> str:
        """Format FLOPs with appropriate unit."""
        if flops >= 1e15:
            return f"{flops / 1e15:.2f} PFLOPs"
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        if flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        if flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        return f"{flops:.2f} FLOPs"

    lines = [
        f"Memory: {stats.memory_usage_mb:.1f} MB",
        f"FLOPs: {format_flops(stats.flops)}",
        f"Throughput: {stats.throughput:.1f} tokens/sec",
    ]

    return "\n".join(lines)


def get_recommended_attention_config(
    model_size: str,
    use_flash: bool = True,
) -> AttentionConfig:
    """Get recommended attention configuration for model size.

    Args:
        model_size: Model size string (e.g., "7b", "13b", "70b").
        use_flash: Whether to prefer Flash Attention. Defaults to True.

    Returns:
        Recommended AttentionConfig.

    Raises:
        ValueError: If model_size is not recognized.

    Examples:
        >>> config = get_recommended_attention_config("7b")
        >>> config.num_heads
        32
        >>> config.head_dim
        128

        >>> config_70b = get_recommended_attention_config("70b")
        >>> config_70b.num_heads
        64

        >>> get_recommended_attention_config("invalid")
        Traceback (most recent call last):
            ...
        ValueError: unrecognized model size: invalid
    """
    model_size = model_size.lower().strip()

    # Common configurations for different model sizes
    configs = {
        "7b": {"num_heads": 32, "head_dim": 128},
        "13b": {"num_heads": 40, "head_dim": 128},
        "70b": {"num_heads": 64, "head_dim": 128},
    }

    if model_size not in configs:
        msg = f"unrecognized model size: {model_size}"
        raise ValueError(msg)

    cfg = configs[model_size]
    attention_type = AttentionType.FLASH if use_flash else AttentionType.STANDARD

    return create_attention_config(
        attention_type=attention_type,
        num_heads=cfg["num_heads"],
        head_dim=cfg["head_dim"],
        dropout=0.0,
        use_bias=False,
    )


def create_attention_stats(
    memory_usage_mb: float,
    flops: float,
    throughput: float,
) -> AttentionStats:
    """Create attention statistics.

    Args:
        memory_usage_mb: Memory usage in megabytes.
        flops: Floating point operations.
        throughput: Throughput in tokens per second.

    Returns:
        AttentionStats instance.

    Raises:
        ValueError: If memory_usage_mb is negative.
        ValueError: If flops is negative.
        ValueError: If throughput is negative.

    Examples:
        >>> stats = create_attention_stats(512.0, 1e12, 5000.0)
        >>> stats.memory_usage_mb
        512.0
        >>> stats.flops
        1000000000000.0

        >>> create_attention_stats(-1.0, 1e12, 5000.0)
        Traceback (most recent call last):
            ...
        ValueError: memory_usage_mb must be non-negative, got -1.0
    """
    if memory_usage_mb < 0:
        msg = f"memory_usage_mb must be non-negative, got {memory_usage_mb}"
        raise ValueError(msg)
    if flops < 0:
        msg = f"flops must be non-negative, got {flops}"
        raise ValueError(msg)
    if throughput < 0:
        msg = f"throughput must be non-negative, got {throughput}"
        raise ValueError(msg)

    return AttentionStats(
        memory_usage_mb=memory_usage_mb,
        flops=flops,
        throughput=throughput,
    )
