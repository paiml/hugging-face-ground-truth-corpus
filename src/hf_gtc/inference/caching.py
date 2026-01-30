"""Prompt caching and prefix sharing utilities for efficient inference.

This module provides utilities for configuring prompt caching, prefix sharing,
and semantic caching to accelerate inference by reusing computation for
repeated or similar prompts.

Examples:
    >>> from hf_gtc.inference.caching import create_cache_config
    >>> config = create_cache_config(max_size_mb=512)
    >>> config.max_size_mb
    512
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class CacheType(Enum):
    """Type of cache for prompt caching.

    Attributes:
        PROMPT: Cache entire prompt embeddings for exact match reuse.
        PREFIX: Cache common prefixes shared across multiple prompts.
        SEMANTIC: Cache based on semantic similarity of prompts.
        KV: Cache key-value pairs from transformer attention layers.

    Examples:
        >>> CacheType.PROMPT.value
        'prompt'
        >>> CacheType.PREFIX.value
        'prefix'
        >>> CacheType.SEMANTIC.value
        'semantic'
        >>> CacheType.KV.value
        'kv'
    """

    PROMPT = "prompt"
    PREFIX = "prefix"
    SEMANTIC = "semantic"
    KV = "kv"


VALID_CACHE_TYPES = frozenset(c.value for c in CacheType)


class EvictionPolicy(Enum):
    """Eviction policy for cache management.

    Attributes:
        LRU: Least recently used eviction.
        LFU: Least frequently used eviction.
        FIFO: First-in-first-out eviction.
        TTL: Time-to-live based eviction.
        ADAPTIVE: Adaptive policy based on access patterns.

    Examples:
        >>> EvictionPolicy.LRU.value
        'lru'
        >>> EvictionPolicy.LFU.value
        'lfu'
        >>> EvictionPolicy.FIFO.value
        'fifo'
        >>> EvictionPolicy.TTL.value
        'ttl'
        >>> EvictionPolicy.ADAPTIVE.value
        'adaptive'
    """

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


VALID_EVICTION_POLICIES = frozenset(p.value for p in EvictionPolicy)


class CacheBackend(Enum):
    """Backend for cache storage.

    Attributes:
        MEMORY: In-memory cache for fastest access.
        REDIS: Redis-based distributed cache.
        DISK: Disk-based persistent cache.
        HYBRID: Combination of memory and disk caching.

    Examples:
        >>> CacheBackend.MEMORY.value
        'memory'
        >>> CacheBackend.REDIS.value
        'redis'
        >>> CacheBackend.DISK.value
        'disk'
        >>> CacheBackend.HYBRID.value
        'hybrid'
    """

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


VALID_CACHE_BACKENDS = frozenset(b.value for b in CacheBackend)


# Type aliases for string literal types
CacheTypeStr = Literal["prompt", "prefix", "semantic", "kv"]
EvictionPolicyStr = Literal["lru", "lfu", "fifo", "ttl", "adaptive"]
CacheBackendStr = Literal["memory", "redis", "disk", "hybrid"]
HashAlgorithmStr = Literal["sha256", "md5", "xxhash", "cityhash"]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Configuration for prompt caching.

    Attributes:
        cache_type: Type of cache to use.
        max_size_mb: Maximum cache size in megabytes.
        eviction_policy: Policy for evicting cache entries.
        ttl_seconds: Time-to-live for cache entries in seconds.

    Examples:
        >>> config = CacheConfig(
        ...     cache_type=CacheType.PROMPT,
        ...     max_size_mb=512,
        ...     eviction_policy=EvictionPolicy.LRU,
        ...     ttl_seconds=3600,
        ... )
        >>> config.max_size_mb
        512
        >>> config.cache_type
        <CacheType.PROMPT: 'prompt'>
    """

    cache_type: CacheType
    max_size_mb: int
    eviction_policy: EvictionPolicy
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class PrefixConfig:
    """Configuration for prefix sharing.

    Attributes:
        shared_prefix_length: Minimum length of prefix to share.
        reuse_threshold: Minimum number of prompts sharing prefix to cache.
        hash_algorithm: Algorithm for prefix hashing.

    Examples:
        >>> config = PrefixConfig(
        ...     shared_prefix_length=64,
        ...     reuse_threshold=3,
        ...     hash_algorithm="sha256",
        ... )
        >>> config.shared_prefix_length
        64
        >>> config.reuse_threshold
        3
    """

    shared_prefix_length: int
    reuse_threshold: int
    hash_algorithm: HashAlgorithmStr


@dataclass(frozen=True, slots=True)
class SemanticCacheConfig:
    """Configuration for semantic caching.

    Attributes:
        similarity_threshold: Minimum similarity score for cache hit (0.0-1.0).
        embedding_model: Model used for computing embeddings.
        max_entries: Maximum number of entries in semantic cache.

    Examples:
        >>> config = SemanticCacheConfig(
        ...     similarity_threshold=0.95,
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     max_entries=10000,
        ... )
        >>> config.similarity_threshold
        0.95
        >>> config.max_entries
        10000
    """

    similarity_threshold: float
    embedding_model: str
    max_entries: int


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Statistics for cache operations.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        hit_rate: Cache hit rate (0.0 to 1.0).
        evictions: Number of cache evictions.
        memory_used_mb: Current memory usage in megabytes.

    Examples:
        >>> stats = CacheStats(
        ...     hits=800,
        ...     misses=200,
        ...     hit_rate=0.8,
        ...     evictions=50,
        ...     memory_used_mb=256.5,
        ... )
        >>> stats.hits
        800
        >>> stats.hit_rate
        0.8
    """

    hits: int
    misses: int
    hit_rate: float
    evictions: int
    memory_used_mb: float


def validate_cache_config(config: CacheConfig) -> None:
    """Validate cache configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = CacheConfig(
        ...     cache_type=CacheType.PROMPT,
        ...     max_size_mb=512,
        ...     eviction_policy=EvictionPolicy.LRU,
        ...     ttl_seconds=3600,
        ... )
        >>> validate_cache_config(config)  # No error

        >>> bad_config = CacheConfig(
        ...     cache_type=CacheType.PROMPT,
        ...     max_size_mb=0,
        ...     eviction_policy=EvictionPolicy.LRU,
        ...     ttl_seconds=3600,
        ... )
        >>> validate_cache_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_size_mb must be positive
    """
    if config.max_size_mb <= 0:
        msg = f"max_size_mb must be positive, got {config.max_size_mb}"
        raise ValueError(msg)

    if config.ttl_seconds < 0:
        msg = f"ttl_seconds cannot be negative, got {config.ttl_seconds}"
        raise ValueError(msg)


def validate_prefix_config(config: PrefixConfig) -> None:
    """Validate prefix configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PrefixConfig(
        ...     shared_prefix_length=64,
        ...     reuse_threshold=3,
        ...     hash_algorithm="sha256",
        ... )
        >>> validate_prefix_config(config)  # No error

        >>> bad_config = PrefixConfig(
        ...     shared_prefix_length=0,
        ...     reuse_threshold=3,
        ...     hash_algorithm="sha256",
        ... )
        >>> validate_prefix_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: shared_prefix_length must be positive
    """
    if config.shared_prefix_length <= 0:
        msg = (
            f"shared_prefix_length must be positive, got {config.shared_prefix_length}"
        )
        raise ValueError(msg)

    if config.reuse_threshold <= 0:
        msg = f"reuse_threshold must be positive, got {config.reuse_threshold}"
        raise ValueError(msg)

    valid_hash_algorithms = {"sha256", "md5", "xxhash", "cityhash"}
    if config.hash_algorithm not in valid_hash_algorithms:
        msg = (
            f"hash_algorithm must be one of {valid_hash_algorithms}, "
            f"got '{config.hash_algorithm}'"
        )
        raise ValueError(msg)


def validate_semantic_cache_config(config: SemanticCacheConfig) -> None:
    """Validate semantic cache configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SemanticCacheConfig(
        ...     similarity_threshold=0.95,
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     max_entries=10000,
        ... )
        >>> validate_semantic_cache_config(config)  # No error

        >>> bad_config = SemanticCacheConfig(
        ...     similarity_threshold=1.5,
        ...     embedding_model="test-model",
        ...     max_entries=10000,
        ... )
        >>> validate_semantic_cache_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: similarity_threshold must be between 0.0 and 1.0
    """
    if not 0.0 <= config.similarity_threshold <= 1.0:
        msg = (
            f"similarity_threshold must be between 0.0 and 1.0, "
            f"got {config.similarity_threshold}"
        )
        raise ValueError(msg)

    if not config.embedding_model:
        msg = "embedding_model cannot be empty"
        raise ValueError(msg)

    if config.max_entries <= 0:
        msg = f"max_entries must be positive, got {config.max_entries}"
        raise ValueError(msg)


def validate_cache_stats(stats: CacheStats) -> None:
    """Validate cache statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = CacheStats(800, 200, 0.8, 50, 256.5)
        >>> validate_cache_stats(stats)  # No error

        >>> bad_stats = CacheStats(-1, 200, 0.8, 50, 256.5)
        >>> validate_cache_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hits cannot be negative
    """
    if stats.hits < 0:
        msg = f"hits cannot be negative, got {stats.hits}"
        raise ValueError(msg)

    if stats.misses < 0:
        msg = f"misses cannot be negative, got {stats.misses}"
        raise ValueError(msg)

    if not 0.0 <= stats.hit_rate <= 1.0:
        msg = f"hit_rate must be between 0.0 and 1.0, got {stats.hit_rate}"
        raise ValueError(msg)

    if stats.evictions < 0:
        msg = f"evictions cannot be negative, got {stats.evictions}"
        raise ValueError(msg)

    if stats.memory_used_mb < 0:
        msg = f"memory_used_mb cannot be negative, got {stats.memory_used_mb}"
        raise ValueError(msg)


def create_cache_config(
    cache_type: CacheTypeStr = "prompt",
    max_size_mb: int = 512,
    eviction_policy: EvictionPolicyStr = "lru",
    ttl_seconds: int = 3600,
) -> CacheConfig:
    """Create a cache configuration.

    Args:
        cache_type: Type of cache. Defaults to "prompt".
        max_size_mb: Maximum cache size in MB. Defaults to 512.
        eviction_policy: Eviction policy. Defaults to "lru".
        ttl_seconds: TTL in seconds. Defaults to 3600.

    Returns:
        CacheConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_cache_config(max_size_mb=1024)
        >>> config.max_size_mb
        1024
        >>> config.cache_type
        <CacheType.PROMPT: 'prompt'>

        >>> config = create_cache_config(cache_type="prefix", eviction_policy="lfu")
        >>> config.cache_type
        <CacheType.PREFIX: 'prefix'>
        >>> config.eviction_policy
        <EvictionPolicy.LFU: 'lfu'>

        >>> create_cache_config(max_size_mb=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_size_mb must be positive

        >>> create_cache_config(cache_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: cache_type must be one of
    """
    if cache_type not in VALID_CACHE_TYPES:
        msg = f"cache_type must be one of {VALID_CACHE_TYPES}, got '{cache_type}'"
        raise ValueError(msg)

    if eviction_policy not in VALID_EVICTION_POLICIES:
        msg = (
            f"eviction_policy must be one of {VALID_EVICTION_POLICIES}, "
            f"got '{eviction_policy}'"
        )
        raise ValueError(msg)

    config = CacheConfig(
        cache_type=CacheType(cache_type),
        max_size_mb=max_size_mb,
        eviction_policy=EvictionPolicy(eviction_policy),
        ttl_seconds=ttl_seconds,
    )
    validate_cache_config(config)
    return config


def create_prefix_config(
    shared_prefix_length: int = 64,
    reuse_threshold: int = 3,
    hash_algorithm: HashAlgorithmStr = "sha256",
) -> PrefixConfig:
    """Create a prefix sharing configuration.

    Args:
        shared_prefix_length: Minimum prefix length to share. Defaults to 64.
        reuse_threshold: Minimum prompts sharing prefix. Defaults to 3.
        hash_algorithm: Hashing algorithm. Defaults to "sha256".

    Returns:
        PrefixConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_prefix_config(shared_prefix_length=128)
        >>> config.shared_prefix_length
        128
        >>> config.hash_algorithm
        'sha256'

        >>> config = create_prefix_config(hash_algorithm="xxhash")
        >>> config.hash_algorithm
        'xxhash'

        >>> create_prefix_config(shared_prefix_length=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: shared_prefix_length must be positive
    """
    config = PrefixConfig(
        shared_prefix_length=shared_prefix_length,
        reuse_threshold=reuse_threshold,
        hash_algorithm=hash_algorithm,
    )
    validate_prefix_config(config)
    return config


def create_semantic_cache_config(
    similarity_threshold: float = 0.95,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_entries: int = 10000,
) -> SemanticCacheConfig:
    """Create a semantic cache configuration.

    Args:
        similarity_threshold: Minimum similarity for cache hit. Defaults to 0.95.
        embedding_model: Embedding model to use. Defaults to all-MiniLM-L6-v2.
        max_entries: Maximum cache entries. Defaults to 10000.

    Returns:
        SemanticCacheConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_semantic_cache_config(similarity_threshold=0.9)
        >>> config.similarity_threshold
        0.9
        >>> config.embedding_model
        'sentence-transformers/all-MiniLM-L6-v2'

        >>> config = create_semantic_cache_config(max_entries=50000)
        >>> config.max_entries
        50000

        >>> create_semantic_cache_config(similarity_threshold=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: similarity_threshold must be between 0.0 and 1.0
    """
    config = SemanticCacheConfig(
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        max_entries=max_entries,
    )
    validate_semantic_cache_config(config)
    return config


def create_cache_stats(
    hits: int = 0,
    misses: int = 0,
    hit_rate: float = 0.0,
    evictions: int = 0,
    memory_used_mb: float = 0.0,
) -> CacheStats:
    """Create cache statistics.

    Args:
        hits: Number of cache hits. Defaults to 0.
        misses: Number of cache misses. Defaults to 0.
        hit_rate: Cache hit rate. Defaults to 0.0.
        evictions: Number of evictions. Defaults to 0.
        memory_used_mb: Memory used in MB. Defaults to 0.0.

    Returns:
        CacheStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_cache_stats(hits=800, misses=200, hit_rate=0.8)
        >>> stats.hits
        800
        >>> stats.hit_rate
        0.8

        >>> create_cache_stats(hits=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hits cannot be negative
    """
    stats = CacheStats(
        hits=hits,
        misses=misses,
        hit_rate=hit_rate,
        evictions=evictions,
        memory_used_mb=memory_used_mb,
    )
    validate_cache_stats(stats)
    return stats


def list_cache_types() -> list[str]:
    """List available cache types.

    Returns:
        Sorted list of cache type names.

    Examples:
        >>> types = list_cache_types()
        >>> "prompt" in types
        True
        >>> "prefix" in types
        True
        >>> "semantic" in types
        True
        >>> "kv" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CACHE_TYPES)


def list_eviction_policies() -> list[str]:
    """List available eviction policies.

    Returns:
        Sorted list of eviction policy names.

    Examples:
        >>> policies = list_eviction_policies()
        >>> "lru" in policies
        True
        >>> "lfu" in policies
        True
        >>> "adaptive" in policies
        True
        >>> policies == sorted(policies)
        True
    """
    return sorted(VALID_EVICTION_POLICIES)


def list_cache_backends() -> list[str]:
    """List available cache backends.

    Returns:
        Sorted list of cache backend names.

    Examples:
        >>> backends = list_cache_backends()
        >>> "memory" in backends
        True
        >>> "redis" in backends
        True
        >>> "disk" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_CACHE_BACKENDS)


def get_cache_type(name: str) -> CacheType:
    """Get a cache type by name.

    Args:
        name: Name of the cache type.

    Returns:
        The corresponding CacheType enum value.

    Raises:
        ValueError: If name is not a valid cache type.

    Examples:
        >>> get_cache_type("prompt")
        <CacheType.PROMPT: 'prompt'>
        >>> get_cache_type("prefix")
        <CacheType.PREFIX: 'prefix'>

        >>> get_cache_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown cache type: 'invalid'
    """
    if name not in VALID_CACHE_TYPES:
        msg = f"Unknown cache type: '{name}'. Valid: {sorted(VALID_CACHE_TYPES)}"
        raise ValueError(msg)
    return CacheType(name)


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
        >>> get_eviction_policy("adaptive")
        <EvictionPolicy.ADAPTIVE: 'adaptive'>

        >>> get_eviction_policy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown eviction policy: 'invalid'
    """
    if name not in VALID_EVICTION_POLICIES:
        msg = (
            f"Unknown eviction policy: '{name}'. "
            f"Valid: {sorted(VALID_EVICTION_POLICIES)}"
        )
        raise ValueError(msg)
    return EvictionPolicy(name)


def get_cache_backend(name: str) -> CacheBackend:
    """Get a cache backend by name.

    Args:
        name: Name of the cache backend.

    Returns:
        The corresponding CacheBackend enum value.

    Raises:
        ValueError: If name is not a valid cache backend.

    Examples:
        >>> get_cache_backend("memory")
        <CacheBackend.MEMORY: 'memory'>
        >>> get_cache_backend("redis")
        <CacheBackend.REDIS: 'redis'>

        >>> get_cache_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown cache backend: 'invalid'
    """
    if name not in VALID_CACHE_BACKENDS:
        msg = f"Unknown cache backend: '{name}'. Valid: {sorted(VALID_CACHE_BACKENDS)}"
        raise ValueError(msg)
    return CacheBackend(name)


def calculate_cache_hit_rate(hits: int, misses: int) -> float:
    """Calculate cache hit rate from hits and misses.

    Args:
        hits: Number of cache hits.
        misses: Number of cache misses.

    Returns:
        Hit rate as a float between 0.0 and 1.0.

    Raises:
        ValueError: If hits or misses are negative.

    Examples:
        >>> calculate_cache_hit_rate(80, 20)
        0.8
        >>> calculate_cache_hit_rate(0, 0)
        0.0
        >>> calculate_cache_hit_rate(100, 0)
        1.0
        >>> calculate_cache_hit_rate(0, 100)
        0.0

        >>> calculate_cache_hit_rate(-1, 20)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hits cannot be negative

        >>> calculate_cache_hit_rate(80, -1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: misses cannot be negative
    """
    if hits < 0:
        msg = f"hits cannot be negative, got {hits}"
        raise ValueError(msg)

    if misses < 0:
        msg = f"misses cannot be negative, got {misses}"
        raise ValueError(msg)

    total = hits + misses
    if total == 0:
        return 0.0

    return hits / total


def estimate_latency_savings(
    hit_rate: float,
    avg_inference_latency_ms: float,
    cache_lookup_latency_ms: float = 1.0,
) -> float:
    """Estimate latency savings from caching.

    Args:
        hit_rate: Cache hit rate (0.0 to 1.0).
        avg_inference_latency_ms: Average inference latency in milliseconds.
        cache_lookup_latency_ms: Cache lookup latency in ms. Defaults to 1.0.

    Returns:
        Estimated latency savings as percentage (0.0 to 100.0).

    Raises:
        ValueError: If hit_rate is not in [0.0, 1.0].
        ValueError: If latencies are not positive.

    Examples:
        >>> savings = estimate_latency_savings(0.8, 100.0, 1.0)
        >>> savings > 0
        True
        >>> round(savings, 1)
        79.2

        >>> estimate_latency_savings(0.0, 100.0)
        0.0

        >>> estimate_latency_savings(1.0, 100.0, 1.0)
        99.0

        >>> estimate_latency_savings(1.5, 100.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hit_rate must be between 0.0 and 1.0

        >>> estimate_latency_savings(0.8, 0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: avg_inference_latency_ms must be positive
    """
    if not 0.0 <= hit_rate <= 1.0:
        msg = f"hit_rate must be between 0.0 and 1.0, got {hit_rate}"
        raise ValueError(msg)

    if avg_inference_latency_ms <= 0:
        msg = (
            f"avg_inference_latency_ms must be positive, got {avg_inference_latency_ms}"
        )
        raise ValueError(msg)

    if cache_lookup_latency_ms < 0:
        msg = (
            f"cache_lookup_latency_ms cannot be negative, got {cache_lookup_latency_ms}"
        )
        raise ValueError(msg)

    # With caching: hit_rate * lookup_latency + (1 - hit_rate) * inference_latency
    # Without caching: inference_latency
    # Savings = (without - with) / without * 100

    latency_with_cache = (
        hit_rate * cache_lookup_latency_ms + (1 - hit_rate) * avg_inference_latency_ms
    )

    savings_ms = avg_inference_latency_ms - latency_with_cache
    savings_pct = (savings_ms / avg_inference_latency_ms) * 100

    return savings_pct


def calculate_memory_overhead(
    num_entries: int,
    avg_entry_size_kb: float,
    overhead_factor: float = 1.2,
) -> float:
    """Calculate memory overhead for cache.

    Args:
        num_entries: Number of cache entries.
        avg_entry_size_kb: Average entry size in kilobytes.
        overhead_factor: Memory overhead factor for data structures. Defaults to 1.2.

    Returns:
        Total memory usage in megabytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> mem = calculate_memory_overhead(1000, 10.0)
        >>> round(mem, 2)
        11.72

        >>> calculate_memory_overhead(0, 10.0)
        0.0

        >>> calculate_memory_overhead(-1, 10.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_entries cannot be negative

        >>> calculate_memory_overhead(1000, -1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: avg_entry_size_kb cannot be negative
    """
    if num_entries < 0:
        msg = f"num_entries cannot be negative, got {num_entries}"
        raise ValueError(msg)

    if avg_entry_size_kb < 0:
        msg = f"avg_entry_size_kb cannot be negative, got {avg_entry_size_kb}"
        raise ValueError(msg)

    if overhead_factor < 1.0:
        msg = f"overhead_factor must be at least 1.0, got {overhead_factor}"
        raise ValueError(msg)

    total_kb = num_entries * avg_entry_size_kb * overhead_factor
    return total_kb / 1024  # Convert to MB


def optimize_cache_size(
    available_memory_mb: float,
    target_hit_rate: float,
    avg_entry_size_kb: float,
    estimated_unique_prompts: int,
) -> int:
    """Optimize cache size based on constraints.

    Args:
        available_memory_mb: Available memory in megabytes.
        target_hit_rate: Target cache hit rate (0.0 to 1.0).
        avg_entry_size_kb: Average entry size in kilobytes.
        estimated_unique_prompts: Estimated number of unique prompts.

    Returns:
        Recommended cache size in megabytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> size = optimize_cache_size(1024.0, 0.9, 10.0, 5000)
        >>> 0 < size <= 1024
        True

        >>> optimize_cache_size(1024.0, 0.9, 10.0, 100)
        1

        >>> optimize_cache_size(0.0, 0.9, 10.0, 5000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: available_memory_mb must be positive

        >>> optimize_cache_size(1024.0, 1.5, 10.0, 5000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_hit_rate must be between 0.0 and 1.0
    """
    if available_memory_mb <= 0:
        msg = f"available_memory_mb must be positive, got {available_memory_mb}"
        raise ValueError(msg)

    if not 0.0 <= target_hit_rate <= 1.0:
        msg = f"target_hit_rate must be between 0.0 and 1.0, got {target_hit_rate}"
        raise ValueError(msg)

    if avg_entry_size_kb <= 0:
        msg = f"avg_entry_size_kb must be positive, got {avg_entry_size_kb}"
        raise ValueError(msg)

    if estimated_unique_prompts <= 0:
        msg = (
            f"estimated_unique_prompts must be positive, got {estimated_unique_prompts}"
        )
        raise ValueError(msg)

    # Calculate entries needed for target hit rate
    # Assuming Zipf distribution, top X% of prompts account for ~X% of requests
    entries_for_hit_rate = int(estimated_unique_prompts * target_hit_rate)

    # Calculate memory needed (with 1.2x overhead)
    memory_needed_kb = entries_for_hit_rate * avg_entry_size_kb * 1.2
    memory_needed_mb = memory_needed_kb / 1024

    # Return the minimum of needed and available
    optimal_size = int(min(memory_needed_mb, available_memory_mb))

    # Ensure at least 1 MB
    return max(1, optimal_size)


def format_cache_stats(stats: CacheStats) -> str:
    """Format cache statistics as a human-readable string.

    Args:
        stats: Cache statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = CacheStats(800, 200, 0.8, 50, 256.5)
        >>> formatted = format_cache_stats(stats)
        >>> "Hits: 800" in formatted
        True
        >>> "Misses: 200" in formatted
        True
        >>> "Hit Rate: 80.00%" in formatted
        True
        >>> "Evictions: 50" in formatted
        True
        >>> "Memory Used: 256.50 MB" in formatted
        True

        >>> empty_stats = CacheStats(0, 0, 0.0, 0, 0.0)
        >>> "Hit Rate: 0.00%" in format_cache_stats(empty_stats)
        True
    """
    lines = [
        "Cache Statistics:",
        f"  Hits: {stats.hits}",
        f"  Misses: {stats.misses}",
        f"  Total: {stats.hits + stats.misses}",
        f"  Hit Rate: {stats.hit_rate * 100:.2f}%",
        f"  Evictions: {stats.evictions}",
        f"  Memory Used: {stats.memory_used_mb:.2f} MB",
    ]
    return "\n".join(lines)


def get_recommended_cache_config(
    use_case: str,
    available_memory_mb: int = 512,
) -> CacheConfig:
    """Get recommended cache configuration for a use case.

    Args:
        use_case: Use case type ("chatbot", "batch", "api", "development").
        available_memory_mb: Available memory in MB. Defaults to 512.

    Returns:
        Recommended CacheConfig for the use case.

    Raises:
        ValueError: If use_case is invalid.

    Examples:
        >>> config = get_recommended_cache_config("chatbot")
        >>> config.cache_type
        <CacheType.SEMANTIC: 'semantic'>
        >>> config.eviction_policy
        <EvictionPolicy.LRU: 'lru'>

        >>> config = get_recommended_cache_config("batch")
        >>> config.cache_type
        <CacheType.PREFIX: 'prefix'>
        >>> config.eviction_policy
        <EvictionPolicy.FIFO: 'fifo'>

        >>> config = get_recommended_cache_config("api", available_memory_mb=2048)
        >>> config.max_size_mb
        2048
        >>> config.cache_type
        <CacheType.KV: 'kv'>

        >>> config = get_recommended_cache_config("development")
        >>> config.ttl_seconds
        300

        >>> get_recommended_cache_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case must be one of
    """
    valid_use_cases = {"chatbot", "batch", "api", "development"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    if available_memory_mb <= 0:
        msg = f"available_memory_mb must be positive, got {available_memory_mb}"
        raise ValueError(msg)

    # Configuration recommendations by use case
    configs = {
        "chatbot": {
            "cache_type": CacheType.SEMANTIC,
            "eviction_policy": EvictionPolicy.LRU,
            "ttl_seconds": 7200,  # 2 hours
        },
        "batch": {
            "cache_type": CacheType.PREFIX,
            "eviction_policy": EvictionPolicy.FIFO,
            "ttl_seconds": 1800,  # 30 minutes
        },
        "api": {
            "cache_type": CacheType.KV,
            "eviction_policy": EvictionPolicy.LFU,
            "ttl_seconds": 3600,  # 1 hour
        },
        "development": {
            "cache_type": CacheType.PROMPT,
            "eviction_policy": EvictionPolicy.TTL,
            "ttl_seconds": 300,  # 5 minutes
        },
    }

    params = configs[use_case]
    return CacheConfig(
        cache_type=params["cache_type"],
        max_size_mb=available_memory_mb,
        eviction_policy=params["eviction_policy"],
        ttl_seconds=params["ttl_seconds"],
    )
