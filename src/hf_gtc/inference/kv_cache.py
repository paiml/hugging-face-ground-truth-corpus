"""KV cache management utilities for efficient inference.

This module provides utilities for managing KV caches in transformer models,
including eviction policies, compression strategies, and paged attention
configurations for memory-efficient inference.

Examples:
    >>> from hf_gtc.inference.kv_cache import create_kv_cache_management_config
    >>> config = create_kv_cache_management_config(max_cache_size_gb=8.0)
    >>> config.max_cache_size_gb
    8.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class EvictionPolicy(Enum):
    """Eviction policy for KV cache entries.

    Attributes:
        LRU: Least recently used eviction.
        FIFO: First-in-first-out eviction.
        LFU: Least frequently used eviction.
        RANDOM: Random eviction.
        ATTENTION_SCORE: Evict based on attention scores.

    Examples:
        >>> EvictionPolicy.LRU.value
        'lru'
        >>> EvictionPolicy.ATTENTION_SCORE.value
        'attention_score'
    """

    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"
    RANDOM = "random"
    ATTENTION_SCORE = "attention_score"


VALID_EVICTION_POLICIES = frozenset(p.value for p in EvictionPolicy)


class CacheCompression(Enum):
    """Compression strategy for KV cache.

    Attributes:
        NONE: No compression.
        QUANTIZE_FP16: Quantize to FP16.
        QUANTIZE_INT8: Quantize to INT8.
        SPARSE: Sparse compression.

    Examples:
        >>> CacheCompression.NONE.value
        'none'
        >>> CacheCompression.QUANTIZE_INT8.value
        'quantize_int8'
    """

    NONE = "none"
    QUANTIZE_FP16 = "quantize_fp16"
    QUANTIZE_INT8 = "quantize_int8"
    SPARSE = "sparse"


VALID_CACHE_COMPRESSIONS = frozenset(c.value for c in CacheCompression)


class PagedAttentionMode(Enum):
    """Paged attention mode for memory management.

    Attributes:
        DISABLED: Paged attention disabled.
        STATIC: Static page allocation.
        DYNAMIC: Dynamic page allocation.

    Examples:
        >>> PagedAttentionMode.DISABLED.value
        'disabled'
        >>> PagedAttentionMode.DYNAMIC.value
        'dynamic'
    """

    DISABLED = "disabled"
    STATIC = "static"
    DYNAMIC = "dynamic"


VALID_PAGED_ATTENTION_MODES = frozenset(m.value for m in PagedAttentionMode)


# Type aliases for string literal types
EvictionPolicyStr = Literal["lru", "fifo", "lfu", "random", "attention_score"]
CacheCompressionStr = Literal["none", "quantize_fp16", "quantize_int8", "sparse"]
PagedAttentionModeStr = Literal["disabled", "static", "dynamic"]
DtypeStr = Literal["float32", "float16", "bfloat16", "int8"]


@dataclass(frozen=True, slots=True)
class KVCacheManagementConfig:
    """Configuration for KV cache management.

    Attributes:
        max_cache_size_gb: Maximum cache size in gigabytes.
        eviction_policy: Policy for evicting cache entries.
        compression: Compression strategy for cache.
        dtype: Data type for cache entries.

    Examples:
        >>> config = KVCacheManagementConfig(
        ...     max_cache_size_gb=8.0,
        ...     eviction_policy=EvictionPolicy.LRU,
        ...     compression=CacheCompression.NONE,
        ...     dtype="float16",
        ... )
        >>> config.max_cache_size_gb
        8.0
        >>> config.eviction_policy
        <EvictionPolicy.LRU: 'lru'>
    """

    max_cache_size_gb: float
    eviction_policy: EvictionPolicy
    compression: CacheCompression
    dtype: DtypeStr


@dataclass(frozen=True, slots=True)
class PagedAttentionConfig:
    """Configuration for paged attention.

    Attributes:
        mode: Paged attention mode.
        page_size: Number of tokens per page.
        max_pages: Maximum number of pages.
        block_size: Block size for memory allocation.

    Examples:
        >>> config = PagedAttentionConfig(
        ...     mode=PagedAttentionMode.DYNAMIC,
        ...     page_size=16,
        ...     max_pages=1024,
        ...     block_size=256,
        ... )
        >>> config.page_size
        16
        >>> config.mode
        <PagedAttentionMode.DYNAMIC: 'dynamic'>
    """

    mode: PagedAttentionMode
    page_size: int
    max_pages: int
    block_size: int


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Statistics for KV cache operations.

    Attributes:
        hit_rate: Cache hit rate (0.0 to 1.0).
        eviction_count: Number of evictions performed.
        memory_usage_gb: Current memory usage in gigabytes.
        compression_ratio: Compression ratio achieved (1.0 = no compression).

    Examples:
        >>> stats = CacheStats(
        ...     hit_rate=0.95,
        ...     eviction_count=100,
        ...     memory_usage_gb=4.5,
        ...     compression_ratio=0.5,
        ... )
        >>> stats.hit_rate
        0.95
        >>> stats.memory_usage_gb
        4.5
    """

    hit_rate: float
    eviction_count: int
    memory_usage_gb: float
    compression_ratio: float


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A single entry in the KV cache.

    Attributes:
        sequence_id: Unique identifier for the sequence.
        num_tokens: Number of tokens cached.
        memory_bytes: Memory used by this entry in bytes.
        last_access_time: Unix timestamp of last access.

    Examples:
        >>> entry = CacheEntry(
        ...     sequence_id="seq_001",
        ...     num_tokens=512,
        ...     memory_bytes=1048576,
        ...     last_access_time=1704067200.0,
        ... )
        >>> entry.num_tokens
        512
        >>> entry.sequence_id
        'seq_001'
    """

    sequence_id: str
    num_tokens: int
    memory_bytes: int
    last_access_time: float


def validate_kv_cache_management_config(config: KVCacheManagementConfig) -> None:
    """Validate KV cache management configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = KVCacheManagementConfig(
        ...     max_cache_size_gb=8.0,
        ...     eviction_policy=EvictionPolicy.LRU,
        ...     compression=CacheCompression.NONE,
        ...     dtype="float16",
        ... )
        >>> validate_kv_cache_management_config(config)  # No error

        >>> bad_config = KVCacheManagementConfig(
        ...     max_cache_size_gb=0.0,
        ...     eviction_policy=EvictionPolicy.LRU,
        ...     compression=CacheCompression.NONE,
        ...     dtype="float16",
        ... )
        >>> validate_kv_cache_management_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_cache_size_gb must be positive
    """
    if config.max_cache_size_gb <= 0:
        msg = f"max_cache_size_gb must be positive, got {config.max_cache_size_gb}"
        raise ValueError(msg)

    valid_dtypes = {"float32", "float16", "bfloat16", "int8"}
    if config.dtype not in valid_dtypes:
        msg = f"dtype must be one of {valid_dtypes}, got '{config.dtype}'"
        raise ValueError(msg)


def validate_paged_attention_config(config: PagedAttentionConfig) -> None:
    """Validate paged attention configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PagedAttentionConfig(
        ...     mode=PagedAttentionMode.DYNAMIC,
        ...     page_size=16,
        ...     max_pages=1024,
        ...     block_size=256,
        ... )
        >>> validate_paged_attention_config(config)  # No error

        >>> bad_config = PagedAttentionConfig(
        ...     mode=PagedAttentionMode.DYNAMIC,
        ...     page_size=0,
        ...     max_pages=1024,
        ...     block_size=256,
        ... )
        >>> validate_paged_attention_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: page_size must be positive
    """
    if config.page_size <= 0:
        msg = f"page_size must be positive, got {config.page_size}"
        raise ValueError(msg)

    if config.max_pages <= 0:
        msg = f"max_pages must be positive, got {config.max_pages}"
        raise ValueError(msg)

    if config.block_size <= 0:
        msg = f"block_size must be positive, got {config.block_size}"
        raise ValueError(msg)


def validate_cache_entry(entry: CacheEntry) -> None:
    """Validate cache entry.

    Args:
        entry: Cache entry to validate.

    Raises:
        ValueError: If entry is invalid.

    Examples:
        >>> entry = CacheEntry("seq_001", 512, 1048576, 1704067200.0)
        >>> validate_cache_entry(entry)  # No error

        >>> bad_entry = CacheEntry("", 512, 1048576, 1704067200.0)
        >>> validate_cache_entry(bad_entry)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sequence_id cannot be empty
    """
    if not entry.sequence_id:
        msg = "sequence_id cannot be empty"
        raise ValueError(msg)

    if entry.num_tokens < 0:
        msg = f"num_tokens cannot be negative, got {entry.num_tokens}"
        raise ValueError(msg)

    if entry.memory_bytes < 0:
        msg = f"memory_bytes cannot be negative, got {entry.memory_bytes}"
        raise ValueError(msg)

    if entry.last_access_time < 0:
        msg = f"last_access_time cannot be negative, got {entry.last_access_time}"
        raise ValueError(msg)


def create_kv_cache_management_config(
    max_cache_size_gb: float = 8.0,
    eviction_policy: EvictionPolicyStr = "lru",
    compression: CacheCompressionStr = "none",
    dtype: DtypeStr = "float16",
) -> KVCacheManagementConfig:
    """Create a KV cache management configuration.

    Args:
        max_cache_size_gb: Maximum cache size in GB. Defaults to 8.0.
        eviction_policy: Eviction policy. Defaults to "lru".
        compression: Compression strategy. Defaults to "none".
        dtype: Data type. Defaults to "float16".

    Returns:
        KVCacheManagementConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_kv_cache_management_config(max_cache_size_gb=16.0)
        >>> config.max_cache_size_gb
        16.0

        >>> config = create_kv_cache_management_config(eviction_policy="lfu")
        >>> config.eviction_policy
        <EvictionPolicy.LFU: 'lfu'>

        >>> create_kv_cache_management_config(max_cache_size_gb=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_cache_size_gb must be positive
    """
    if eviction_policy not in VALID_EVICTION_POLICIES:
        msg = (
            f"eviction_policy must be one of {VALID_EVICTION_POLICIES}, "
            f"got '{eviction_policy}'"
        )
        raise ValueError(msg)

    if compression not in VALID_CACHE_COMPRESSIONS:
        msg = (
            f"compression must be one of {VALID_CACHE_COMPRESSIONS}, "
            f"got '{compression}'"
        )
        raise ValueError(msg)

    config = KVCacheManagementConfig(
        max_cache_size_gb=max_cache_size_gb,
        eviction_policy=EvictionPolicy(eviction_policy),
        compression=CacheCompression(compression),
        dtype=dtype,
    )
    validate_kv_cache_management_config(config)
    return config


def create_paged_attention_config(
    mode: PagedAttentionModeStr = "dynamic",
    page_size: int = 16,
    max_pages: int = 1024,
    block_size: int = 256,
) -> PagedAttentionConfig:
    """Create a paged attention configuration.

    Args:
        mode: Paged attention mode. Defaults to "dynamic".
        page_size: Tokens per page. Defaults to 16.
        max_pages: Maximum pages. Defaults to 1024.
        block_size: Block size. Defaults to 256.

    Returns:
        PagedAttentionConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_paged_attention_config(page_size=32)
        >>> config.page_size
        32

        >>> config = create_paged_attention_config(mode="static")
        >>> config.mode
        <PagedAttentionMode.STATIC: 'static'>

        >>> create_paged_attention_config(page_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: page_size must be positive
    """
    if mode not in VALID_PAGED_ATTENTION_MODES:
        msg = f"mode must be one of {VALID_PAGED_ATTENTION_MODES}, got '{mode}'"
        raise ValueError(msg)

    config = PagedAttentionConfig(
        mode=PagedAttentionMode(mode),
        page_size=page_size,
        max_pages=max_pages,
        block_size=block_size,
    )
    validate_paged_attention_config(config)
    return config


def create_cache_entry(
    sequence_id: str,
    num_tokens: int,
    memory_bytes: int,
    last_access_time: float,
) -> CacheEntry:
    """Create a cache entry.

    Args:
        sequence_id: Unique sequence identifier.
        num_tokens: Number of tokens in the entry.
        memory_bytes: Memory used by the entry.
        last_access_time: Unix timestamp of last access.

    Returns:
        CacheEntry with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> entry = create_cache_entry("seq_001", 512, 1048576, 1704067200.0)
        >>> entry.sequence_id
        'seq_001'

        >>> create_cache_entry("", 512, 1048576, 1704067200.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sequence_id cannot be empty
    """
    entry = CacheEntry(
        sequence_id=sequence_id,
        num_tokens=num_tokens,
        memory_bytes=memory_bytes,
        last_access_time=last_access_time,
    )
    validate_cache_entry(entry)
    return entry


def list_eviction_policies() -> list[str]:
    """List available eviction policies.

    Returns:
        Sorted list of eviction policy names.

    Examples:
        >>> policies = list_eviction_policies()
        >>> "lru" in policies
        True
        >>> "fifo" in policies
        True
        >>> policies == sorted(policies)
        True
    """
    return sorted(VALID_EVICTION_POLICIES)


def list_cache_compressions() -> list[str]:
    """List available cache compression strategies.

    Returns:
        Sorted list of compression strategy names.

    Examples:
        >>> compressions = list_cache_compressions()
        >>> "none" in compressions
        True
        >>> "quantize_int8" in compressions
        True
        >>> compressions == sorted(compressions)
        True
    """
    return sorted(VALID_CACHE_COMPRESSIONS)


def list_paged_attention_modes() -> list[str]:
    """List available paged attention modes.

    Returns:
        Sorted list of paged attention mode names.

    Examples:
        >>> modes = list_paged_attention_modes()
        >>> "dynamic" in modes
        True
        >>> "disabled" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_PAGED_ATTENTION_MODES)


def get_eviction_policy(name: str) -> EvictionPolicy:
    """Get an eviction policy by name.

    Args:
        name: Name of the eviction policy.

    Returns:
        The corresponding EvictionPolicy enum value.

    Raises:
        ValueError: If name is not a valid eviction policy.

    Examples:
        >>> get_eviction_policy("lru")
        <EvictionPolicy.LRU: 'lru'>
        >>> get_eviction_policy("attention_score")
        <EvictionPolicy.ATTENTION_SCORE: 'attention_score'>

        >>> get_eviction_policy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown eviction policy
    """
    if name not in VALID_EVICTION_POLICIES:
        msg = f"Unknown eviction policy: '{name}'. Valid: {VALID_EVICTION_POLICIES}"
        raise ValueError(msg)
    return EvictionPolicy(name)


def get_cache_compression(name: str) -> CacheCompression:
    """Get a cache compression strategy by name.

    Args:
        name: Name of the compression strategy.

    Returns:
        The corresponding CacheCompression enum value.

    Raises:
        ValueError: If name is not a valid compression strategy.

    Examples:
        >>> get_cache_compression("none")
        <CacheCompression.NONE: 'none'>
        >>> get_cache_compression("quantize_fp16")
        <CacheCompression.QUANTIZE_FP16: 'quantize_fp16'>

        >>> get_cache_compression("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown cache compression
    """
    if name not in VALID_CACHE_COMPRESSIONS:
        msg = f"Unknown cache compression: '{name}'. Valid: {VALID_CACHE_COMPRESSIONS}"
        raise ValueError(msg)
    return CacheCompression(name)


def get_paged_attention_mode(name: str) -> PagedAttentionMode:
    """Get a paged attention mode by name.

    Args:
        name: Name of the paged attention mode.

    Returns:
        The corresponding PagedAttentionMode enum value.

    Raises:
        ValueError: If name is not a valid paged attention mode.

    Examples:
        >>> get_paged_attention_mode("dynamic")
        <PagedAttentionMode.DYNAMIC: 'dynamic'>
        >>> get_paged_attention_mode("static")
        <PagedAttentionMode.STATIC: 'static'>

        >>> get_paged_attention_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown paged attention mode
    """
    if name not in VALID_PAGED_ATTENTION_MODES:
        msg = (
            f"Unknown paged attention mode: '{name}'. "
            f"Valid: {VALID_PAGED_ATTENTION_MODES}"
        )
        raise ValueError(msg)
    return PagedAttentionMode(name)


def calculate_cache_memory(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_length: int,
    batch_size: int = 1,
    dtype: DtypeStr = "float16",
) -> float:
    """Calculate KV cache memory usage in gigabytes.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        max_seq_length: Maximum sequence length.
        batch_size: Batch size. Defaults to 1.
        dtype: Data type for cache. Defaults to "float16".

    Returns:
        Memory usage in gigabytes.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> mem = calculate_cache_memory(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     max_seq_length=2048,
        ...     batch_size=1,
        ...     dtype="float16",
        ... )
        >>> mem > 0
        True
        >>> round(mem, 2)
        1.0

        >>> calculate_cache_memory(0, 32, 128, 2048)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_layers must be positive
    """
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)

    if max_seq_length <= 0:
        msg = f"max_seq_length must be positive, got {max_seq_length}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    # Bytes per element based on dtype
    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }
    bytes_per_elem = dtype_bytes.get(dtype, 2)

    # KV cache: 2 (K + V) * layers * batch * seq * heads * head_dim * dtype
    total_bytes = (
        2
        * num_layers
        * batch_size
        * max_seq_length
        * num_heads
        * head_dim
        * bytes_per_elem
    )

    # Convert to gigabytes
    return total_bytes / (1024**3)


def estimate_max_sequences(
    available_memory_gb: float,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    avg_seq_length: int,
    dtype: DtypeStr = "float16",
    memory_fraction: float = 0.8,
) -> int:
    """Estimate maximum number of sequences that fit in memory.

    Args:
        available_memory_gb: Available memory in gigabytes.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        avg_seq_length: Average sequence length.
        dtype: Data type for cache. Defaults to "float16".
        memory_fraction: Fraction of memory to use. Defaults to 0.8.

    Returns:
        Maximum number of sequences.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> max_seqs = estimate_max_sequences(
        ...     available_memory_gb=16.0,
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     avg_seq_length=512,
        ... )
        >>> max_seqs > 0
        True

        >>> estimate_max_sequences(0.0, 32, 32, 128, 512)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: available_memory_gb must be positive
    """
    if available_memory_gb <= 0:
        msg = f"available_memory_gb must be positive, got {available_memory_gb}"
        raise ValueError(msg)

    if not 0 < memory_fraction <= 1:
        msg = f"memory_fraction must be in (0, 1], got {memory_fraction}"
        raise ValueError(msg)

    # Calculate memory for one sequence
    mem_per_seq = calculate_cache_memory(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_length=avg_seq_length,
        batch_size=1,
        dtype=dtype,
    )

    usable_memory = available_memory_gb * memory_fraction
    max_seqs = int(usable_memory / mem_per_seq) if mem_per_seq > 0 else 0

    return max(1, max_seqs)


def calculate_optimal_page_size(
    head_dim: int,
    num_heads: int,
    target_memory_kb: int = 64,
) -> int:
    """Calculate optimal page size for paged attention.

    Args:
        head_dim: Dimension per attention head.
        num_heads: Number of attention heads.
        target_memory_kb: Target memory per page in KB. Defaults to 64.

    Returns:
        Optimal page size (number of tokens per page).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> page_size = calculate_optimal_page_size(
        ...     head_dim=128,
        ...     num_heads=32,
        ...     target_memory_kb=64,
        ... )
        >>> page_size > 0
        True
        >>> page_size == 4
        True

        >>> calculate_optimal_page_size(0, 32)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: head_dim must be positive
    """
    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if target_memory_kb <= 0:
        msg = f"target_memory_kb must be positive, got {target_memory_kb}"
        raise ValueError(msg)

    # Memory per token (K + V, FP16) = 2 * 2 * num_heads * head_dim bytes
    bytes_per_token = 4 * num_heads * head_dim
    target_bytes = target_memory_kb * 1024

    page_size = target_bytes // bytes_per_token

    # Round to power of 2 for memory alignment
    if page_size <= 0:
        return 1

    # Find nearest power of 2
    power = 1
    while power * 2 <= page_size:
        power *= 2

    return power


def calculate_compression_savings(
    original_dtype: DtypeStr,
    compressed_dtype: DtypeStr,
    sparsity_ratio: float = 0.0,
) -> float:
    """Calculate memory savings from compression.

    Args:
        original_dtype: Original data type.
        compressed_dtype: Compressed data type.
        sparsity_ratio: Ratio of zero values (0.0 to 1.0). Defaults to 0.0.

    Returns:
        Compression ratio (0.0 to 1.0, lower is better compression).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> ratio = calculate_compression_savings("float32", "float16")
        >>> ratio
        0.5

        >>> ratio = calculate_compression_savings("float16", "int8")
        >>> ratio
        0.5

        >>> ratio = calculate_compression_savings(
        ...     "float16", "float16", sparsity_ratio=0.5
        ... )
        >>> ratio
        0.5

        >>> calculate_compression_savings("float32", "float16", sparsity_ratio=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sparsity_ratio must be between 0.0 and 1.0
    """
    if not 0.0 <= sparsity_ratio <= 1.0:
        msg = f"sparsity_ratio must be between 0.0 and 1.0, got {sparsity_ratio}"
        raise ValueError(msg)

    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }

    if original_dtype not in dtype_bytes:
        msg = f"Unknown original_dtype: '{original_dtype}'"
        raise ValueError(msg)

    if compressed_dtype not in dtype_bytes:
        msg = f"Unknown compressed_dtype: '{compressed_dtype}'"
        raise ValueError(msg)

    original_size = dtype_bytes[original_dtype]
    compressed_size = dtype_bytes[compressed_dtype]

    # Base compression ratio from dtype change
    dtype_ratio = compressed_size / original_size

    # Apply sparsity savings (sparse values take no space)
    final_ratio = dtype_ratio * (1.0 - sparsity_ratio)

    return final_ratio


def format_cache_stats(stats: CacheStats) -> str:
    """Format cache statistics as a human-readable string.

    Args:
        stats: Cache statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = CacheStats(
        ...     hit_rate=0.95,
        ...     eviction_count=100,
        ...     memory_usage_gb=4.5,
        ...     compression_ratio=0.5,
        ... )
        >>> formatted = format_cache_stats(stats)
        >>> "Hit Rate: 95.00%" in formatted
        True
        >>> "Memory Usage: 4.50 GB" in formatted
        True

        >>> empty_stats = CacheStats(0.0, 0, 0.0, 1.0)
        >>> "Hit Rate: 0.00%" in format_cache_stats(empty_stats)
        True
    """
    lines = [
        f"Hit Rate: {stats.hit_rate * 100:.2f}%",
        f"Eviction Count: {stats.eviction_count}",
        f"Memory Usage: {stats.memory_usage_gb:.2f} GB",
        f"Compression Ratio: {stats.compression_ratio:.2f}",
    ]
    return "\n".join(lines)


def get_recommended_cache_config(
    model_size_gb: float,
    available_memory_gb: float,
) -> KVCacheManagementConfig:
    """Get recommended KV cache configuration based on model and memory.

    Args:
        model_size_gb: Model size in gigabytes.
        available_memory_gb: Available GPU memory in gigabytes.

    Returns:
        Recommended KVCacheManagementConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_cache_config(
        ...     model_size_gb=7.0,
        ...     available_memory_gb=24.0,
        ... )
        >>> config.max_cache_size_gb > 0
        True
        >>> config.eviction_policy
        <EvictionPolicy.LRU: 'lru'>

        >>> config = get_recommended_cache_config(
        ...     model_size_gb=70.0,
        ...     available_memory_gb=80.0,
        ... )
        >>> config.compression
        <CacheCompression.QUANTIZE_INT8: 'quantize_int8'>

        >>> get_recommended_cache_config(0.0, 24.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size_gb must be positive
    """
    if model_size_gb <= 0:
        msg = f"model_size_gb must be positive, got {model_size_gb}"
        raise ValueError(msg)

    if available_memory_gb <= 0:
        msg = f"available_memory_gb must be positive, got {available_memory_gb}"
        raise ValueError(msg)

    # Reserve memory for model and activations
    model_overhead = model_size_gb * 1.2  # 20% overhead for activations
    available_for_cache = max(0.5, available_memory_gb - model_overhead)

    # Determine compression based on memory constraints
    memory_ratio = available_memory_gb / model_size_gb

    if memory_ratio < 1.5:
        # Very tight memory - aggressive compression
        compression = CacheCompression.QUANTIZE_INT8
        dtype: DtypeStr = "int8"
    elif memory_ratio < 2.5:
        # Moderate memory - some compression
        compression = CacheCompression.QUANTIZE_FP16
        dtype = "float16"
    else:
        # Plenty of memory - no compression
        compression = CacheCompression.NONE
        dtype = "float16"

    # Eviction policy: LRU is generally best
    eviction_policy = EvictionPolicy.LRU

    return KVCacheManagementConfig(
        max_cache_size_gb=available_for_cache,
        eviction_policy=eviction_policy,
        compression=compression,
        dtype=dtype,
    )
