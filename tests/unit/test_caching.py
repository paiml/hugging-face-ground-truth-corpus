"""Tests for inference.caching module."""

from __future__ import annotations

import pytest

from hf_gtc.inference.caching import (
    VALID_CACHE_BACKENDS,
    VALID_CACHE_TYPES,
    VALID_EVICTION_POLICIES,
    CacheBackend,
    CacheConfig,
    CacheStats,
    CacheType,
    EvictionPolicy,
    PrefixConfig,
    SemanticCacheConfig,
    calculate_cache_hit_rate,
    calculate_memory_overhead,
    create_cache_config,
    create_cache_stats,
    create_prefix_config,
    create_semantic_cache_config,
    estimate_latency_savings,
    format_cache_stats,
    get_cache_backend,
    get_cache_type,
    get_eviction_policy,
    get_recommended_cache_config,
    list_cache_backends,
    list_cache_types,
    list_eviction_policies,
    optimize_cache_size,
    validate_cache_config,
    validate_cache_stats,
    validate_prefix_config,
    validate_semantic_cache_config,
)


class TestCacheType:
    """Tests for CacheType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for cache_type in CacheType:
            assert isinstance(cache_type.value, str)

    def test_prompt_value(self) -> None:
        """PROMPT has correct value."""
        assert CacheType.PROMPT.value == "prompt"

    def test_prefix_value(self) -> None:
        """PREFIX has correct value."""
        assert CacheType.PREFIX.value == "prefix"

    def test_semantic_value(self) -> None:
        """SEMANTIC has correct value."""
        assert CacheType.SEMANTIC.value == "semantic"

    def test_kv_value(self) -> None:
        """KV has correct value."""
        assert CacheType.KV.value == "kv"

    def test_valid_types_frozenset(self) -> None:
        """VALID_CACHE_TYPES is a frozenset."""
        assert isinstance(VALID_CACHE_TYPES, frozenset)

    def test_valid_types_contains_all(self) -> None:
        """VALID_CACHE_TYPES contains all enum values."""
        assert len(VALID_CACHE_TYPES) == len(CacheType)


class TestEvictionPolicy:
    """Tests for EvictionPolicy enum."""

    def test_all_policies_have_values(self) -> None:
        """All policies have string values."""
        for policy in EvictionPolicy:
            assert isinstance(policy.value, str)

    def test_lru_value(self) -> None:
        """LRU has correct value."""
        assert EvictionPolicy.LRU.value == "lru"

    def test_lfu_value(self) -> None:
        """LFU has correct value."""
        assert EvictionPolicy.LFU.value == "lfu"

    def test_fifo_value(self) -> None:
        """FIFO has correct value."""
        assert EvictionPolicy.FIFO.value == "fifo"

    def test_ttl_value(self) -> None:
        """TTL has correct value."""
        assert EvictionPolicy.TTL.value == "ttl"

    def test_adaptive_value(self) -> None:
        """ADAPTIVE has correct value."""
        assert EvictionPolicy.ADAPTIVE.value == "adaptive"

    def test_valid_policies_frozenset(self) -> None:
        """VALID_EVICTION_POLICIES is a frozenset."""
        assert isinstance(VALID_EVICTION_POLICIES, frozenset)


class TestCacheBackend:
    """Tests for CacheBackend enum."""

    def test_all_backends_have_values(self) -> None:
        """All backends have string values."""
        for backend in CacheBackend:
            assert isinstance(backend.value, str)

    def test_memory_value(self) -> None:
        """MEMORY has correct value."""
        assert CacheBackend.MEMORY.value == "memory"

    def test_redis_value(self) -> None:
        """REDIS has correct value."""
        assert CacheBackend.REDIS.value == "redis"

    def test_disk_value(self) -> None:
        """DISK has correct value."""
        assert CacheBackend.DISK.value == "disk"

    def test_hybrid_value(self) -> None:
        """HYBRID has correct value."""
        assert CacheBackend.HYBRID.value == "hybrid"

    def test_valid_backends_frozenset(self) -> None:
        """VALID_CACHE_BACKENDS is a frozenset."""
        assert isinstance(VALID_CACHE_BACKENDS, frozenset)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_create_config(self) -> None:
        """Create cache config."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=512,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=3600,
        )
        assert config.max_size_mb == 512
        assert config.cache_type == CacheType.PROMPT

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=512,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=3600,
        )
        with pytest.raises(AttributeError):
            config.max_size_mb = 1024  # type: ignore[misc]


class TestPrefixConfig:
    """Tests for PrefixConfig dataclass."""

    def test_create_config(self) -> None:
        """Create prefix config."""
        config = PrefixConfig(
            shared_prefix_length=64,
            reuse_threshold=3,
            hash_algorithm="sha256",
        )
        assert config.shared_prefix_length == 64
        assert config.reuse_threshold == 3

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PrefixConfig(64, 3, "sha256")
        with pytest.raises(AttributeError):
            config.shared_prefix_length = 128  # type: ignore[misc]


class TestSemanticCacheConfig:
    """Tests for SemanticCacheConfig dataclass."""

    def test_create_config(self) -> None:
        """Create semantic cache config."""
        config = SemanticCacheConfig(
            similarity_threshold=0.95,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            max_entries=10000,
        )
        assert config.similarity_threshold == 0.95
        assert config.max_entries == 10000

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SemanticCacheConfig(0.95, "test-model", 10000)
        with pytest.raises(AttributeError):
            config.similarity_threshold = 0.9  # type: ignore[misc]


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_create_stats(self) -> None:
        """Create cache stats."""
        stats = CacheStats(
            hits=800,
            misses=200,
            hit_rate=0.8,
            evictions=50,
            memory_used_mb=256.5,
        )
        assert stats.hits == 800
        assert stats.hit_rate == 0.8

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = CacheStats(800, 200, 0.8, 50, 256.5)
        with pytest.raises(AttributeError):
            stats.hits = 900  # type: ignore[misc]


class TestValidateCacheConfig:
    """Tests for validate_cache_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=512,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=3600,
        )
        validate_cache_config(config)

    def test_zero_size_raises(self) -> None:
        """Zero size raises ValueError."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=0,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=3600,
        )
        with pytest.raises(ValueError, match="max_size_mb must be positive"):
            validate_cache_config(config)

    def test_negative_size_raises(self) -> None:
        """Negative size raises ValueError."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=-100,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=3600,
        )
        with pytest.raises(ValueError, match="max_size_mb must be positive"):
            validate_cache_config(config)

    def test_negative_ttl_raises(self) -> None:
        """Negative TTL raises ValueError."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=512,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=-1,
        )
        with pytest.raises(ValueError, match="ttl_seconds cannot be negative"):
            validate_cache_config(config)

    def test_zero_ttl_valid(self) -> None:
        """Zero TTL is valid (no expiration)."""
        config = CacheConfig(
            cache_type=CacheType.PROMPT,
            max_size_mb=512,
            eviction_policy=EvictionPolicy.LRU,
            ttl_seconds=0,
        )
        validate_cache_config(config)


class TestValidatePrefixConfig:
    """Tests for validate_prefix_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PrefixConfig(64, 3, "sha256")
        validate_prefix_config(config)

    def test_zero_prefix_length_raises(self) -> None:
        """Zero prefix length raises ValueError."""
        config = PrefixConfig(0, 3, "sha256")
        with pytest.raises(ValueError, match="shared_prefix_length must be positive"):
            validate_prefix_config(config)

    def test_zero_reuse_threshold_raises(self) -> None:
        """Zero reuse threshold raises ValueError."""
        config = PrefixConfig(64, 0, "sha256")
        with pytest.raises(ValueError, match="reuse_threshold must be positive"):
            validate_prefix_config(config)

    def test_invalid_hash_algorithm_raises(self) -> None:
        """Invalid hash algorithm raises ValueError."""
        config = PrefixConfig(64, 3, "invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="hash_algorithm must be one of"):
            validate_prefix_config(config)

    def test_all_valid_hash_algorithms(self) -> None:
        """All valid hash algorithms pass validation."""
        for algo in ["sha256", "md5", "xxhash", "cityhash"]:
            config = PrefixConfig(64, 3, algo)  # type: ignore[arg-type]
            validate_prefix_config(config)


class TestValidateSemanticCacheConfig:
    """Tests for validate_semantic_cache_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SemanticCacheConfig(0.95, "test-model", 10000)
        validate_semantic_cache_config(config)

    def test_threshold_too_high_raises(self) -> None:
        """Threshold > 1.0 raises ValueError."""
        config = SemanticCacheConfig(1.5, "test-model", 10000)
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_semantic_cache_config(config)

    def test_threshold_too_low_raises(self) -> None:
        """Threshold < 0.0 raises ValueError."""
        config = SemanticCacheConfig(-0.1, "test-model", 10000)
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_semantic_cache_config(config)

    def test_empty_model_raises(self) -> None:
        """Empty embedding model raises ValueError."""
        config = SemanticCacheConfig(0.95, "", 10000)
        with pytest.raises(ValueError, match="embedding_model cannot be empty"):
            validate_semantic_cache_config(config)

    def test_zero_entries_raises(self) -> None:
        """Zero max entries raises ValueError."""
        config = SemanticCacheConfig(0.95, "test-model", 0)
        with pytest.raises(ValueError, match="max_entries must be positive"):
            validate_semantic_cache_config(config)

    def test_boundary_thresholds_valid(self) -> None:
        """Boundary thresholds 0.0 and 1.0 are valid."""
        validate_semantic_cache_config(SemanticCacheConfig(0.0, "test", 100))
        validate_semantic_cache_config(SemanticCacheConfig(1.0, "test", 100))


class TestValidateCacheStats:
    """Tests for validate_cache_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats pass validation."""
        stats = CacheStats(800, 200, 0.8, 50, 256.5)
        validate_cache_stats(stats)

    def test_negative_hits_raises(self) -> None:
        """Negative hits raises ValueError."""
        stats = CacheStats(-1, 200, 0.8, 50, 256.5)
        with pytest.raises(ValueError, match="hits cannot be negative"):
            validate_cache_stats(stats)

    def test_negative_misses_raises(self) -> None:
        """Negative misses raises ValueError."""
        stats = CacheStats(800, -1, 0.8, 50, 256.5)
        with pytest.raises(ValueError, match="misses cannot be negative"):
            validate_cache_stats(stats)

    def test_invalid_hit_rate_raises(self) -> None:
        """Invalid hit rate raises ValueError."""
        stats = CacheStats(800, 200, 1.5, 50, 256.5)
        with pytest.raises(ValueError, match="hit_rate must be between"):
            validate_cache_stats(stats)

    def test_negative_evictions_raises(self) -> None:
        """Negative evictions raises ValueError."""
        stats = CacheStats(800, 200, 0.8, -1, 256.5)
        with pytest.raises(ValueError, match="evictions cannot be negative"):
            validate_cache_stats(stats)

    def test_negative_memory_raises(self) -> None:
        """Negative memory raises ValueError."""
        stats = CacheStats(800, 200, 0.8, 50, -1.0)
        with pytest.raises(ValueError, match="memory_used_mb cannot be negative"):
            validate_cache_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        stats = CacheStats(0, 0, 0.0, 0, 0.0)
        validate_cache_stats(stats)


class TestCreateCacheConfig:
    """Tests for create_cache_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_cache_config()
        assert config.cache_type == CacheType.PROMPT
        assert config.max_size_mb == 512
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.ttl_seconds == 3600

    def test_custom_cache_type(self) -> None:
        """Create config with custom cache type."""
        config = create_cache_config(cache_type="semantic")
        assert config.cache_type == CacheType.SEMANTIC

    def test_custom_size(self) -> None:
        """Create config with custom size."""
        config = create_cache_config(max_size_mb=1024)
        assert config.max_size_mb == 1024

    def test_custom_eviction_policy(self) -> None:
        """Create config with custom eviction policy."""
        config = create_cache_config(eviction_policy="lfu")
        assert config.eviction_policy == EvictionPolicy.LFU

    def test_custom_ttl(self) -> None:
        """Create config with custom TTL."""
        config = create_cache_config(ttl_seconds=7200)
        assert config.ttl_seconds == 7200

    def test_invalid_cache_type_raises(self) -> None:
        """Invalid cache type raises ValueError."""
        with pytest.raises(ValueError, match="cache_type must be one of"):
            create_cache_config(cache_type="invalid")  # type: ignore[arg-type]

    def test_invalid_eviction_policy_raises(self) -> None:
        """Invalid eviction policy raises ValueError."""
        with pytest.raises(ValueError, match="eviction_policy must be one of"):
            create_cache_config(eviction_policy="invalid")  # type: ignore[arg-type]

    def test_zero_size_raises(self) -> None:
        """Zero size raises ValueError."""
        with pytest.raises(ValueError, match="max_size_mb must be positive"):
            create_cache_config(max_size_mb=0)


class TestCreatePrefixConfig:
    """Tests for create_prefix_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_prefix_config()
        assert config.shared_prefix_length == 64
        assert config.reuse_threshold == 3
        assert config.hash_algorithm == "sha256"

    def test_custom_prefix_length(self) -> None:
        """Create config with custom prefix length."""
        config = create_prefix_config(shared_prefix_length=128)
        assert config.shared_prefix_length == 128

    def test_custom_reuse_threshold(self) -> None:
        """Create config with custom reuse threshold."""
        config = create_prefix_config(reuse_threshold=5)
        assert config.reuse_threshold == 5

    def test_custom_hash_algorithm(self) -> None:
        """Create config with custom hash algorithm."""
        config = create_prefix_config(hash_algorithm="xxhash")
        assert config.hash_algorithm == "xxhash"

    def test_zero_prefix_length_raises(self) -> None:
        """Zero prefix length raises ValueError."""
        with pytest.raises(ValueError, match="shared_prefix_length must be positive"):
            create_prefix_config(shared_prefix_length=0)


class TestCreateSemanticCacheConfig:
    """Tests for create_semantic_cache_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_semantic_cache_config()
        assert config.similarity_threshold == 0.95
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.max_entries == 10000

    def test_custom_threshold(self) -> None:
        """Create config with custom threshold."""
        config = create_semantic_cache_config(similarity_threshold=0.9)
        assert config.similarity_threshold == 0.9

    def test_custom_model(self) -> None:
        """Create config with custom model."""
        config = create_semantic_cache_config(embedding_model="custom-model")
        assert config.embedding_model == "custom-model"

    def test_custom_max_entries(self) -> None:
        """Create config with custom max entries."""
        config = create_semantic_cache_config(max_entries=50000)
        assert config.max_entries == 50000

    def test_invalid_threshold_raises(self) -> None:
        """Invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            create_semantic_cache_config(similarity_threshold=1.5)


class TestCreateCacheStats:
    """Tests for create_cache_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_cache_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.evictions == 0
        assert stats.memory_used_mb == 0.0

    def test_custom_values(self) -> None:
        """Create stats with custom values."""
        stats = create_cache_stats(
            hits=800,
            misses=200,
            hit_rate=0.8,
            evictions=50,
            memory_used_mb=256.5,
        )
        assert stats.hits == 800
        assert stats.hit_rate == 0.8

    def test_negative_hits_raises(self) -> None:
        """Negative hits raises ValueError."""
        with pytest.raises(ValueError, match="hits cannot be negative"):
            create_cache_stats(hits=-1)


class TestListCacheTypes:
    """Tests for list_cache_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_cache_types()
        assert types == sorted(types)

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_cache_types()
        assert "prompt" in types
        assert "prefix" in types
        assert "semantic" in types
        assert "kv" in types

    def test_correct_count(self) -> None:
        """Has correct number of types."""
        types = list_cache_types()
        assert len(types) == len(VALID_CACHE_TYPES)


class TestListEvictionPolicies:
    """Tests for list_eviction_policies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        policies = list_eviction_policies()
        assert policies == sorted(policies)

    def test_contains_all_policies(self) -> None:
        """Contains all policies."""
        policies = list_eviction_policies()
        assert "lru" in policies
        assert "lfu" in policies
        assert "adaptive" in policies

    def test_correct_count(self) -> None:
        """Has correct number of policies."""
        policies = list_eviction_policies()
        assert len(policies) == len(VALID_EVICTION_POLICIES)


class TestListCacheBackends:
    """Tests for list_cache_backends function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        backends = list_cache_backends()
        assert backends == sorted(backends)

    def test_contains_all_backends(self) -> None:
        """Contains all backends."""
        backends = list_cache_backends()
        assert "memory" in backends
        assert "redis" in backends
        assert "disk" in backends

    def test_correct_count(self) -> None:
        """Has correct number of backends."""
        backends = list_cache_backends()
        assert len(backends) == len(VALID_CACHE_BACKENDS)


class TestGetCacheType:
    """Tests for get_cache_type function."""

    def test_get_prompt(self) -> None:
        """Get prompt type."""
        cache_type = get_cache_type("prompt")
        assert cache_type == CacheType.PROMPT

    def test_get_prefix(self) -> None:
        """Get prefix type."""
        cache_type = get_cache_type("prefix")
        assert cache_type == CacheType.PREFIX

    def test_get_semantic(self) -> None:
        """Get semantic type."""
        cache_type = get_cache_type("semantic")
        assert cache_type == CacheType.SEMANTIC

    def test_get_kv(self) -> None:
        """Get kv type."""
        cache_type = get_cache_type("kv")
        assert cache_type == CacheType.KV

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cache type"):
            get_cache_type("invalid")


class TestGetEvictionPolicy:
    """Tests for get_eviction_policy function."""

    def test_get_lru(self) -> None:
        """Get LRU policy."""
        policy = get_eviction_policy("lru")
        assert policy == EvictionPolicy.LRU

    def test_get_adaptive(self) -> None:
        """Get adaptive policy."""
        policy = get_eviction_policy("adaptive")
        assert policy == EvictionPolicy.ADAPTIVE

    def test_invalid_policy_raises(self) -> None:
        """Invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown eviction policy"):
            get_eviction_policy("invalid")


class TestGetCacheBackend:
    """Tests for get_cache_backend function."""

    def test_get_memory(self) -> None:
        """Get memory backend."""
        backend = get_cache_backend("memory")
        assert backend == CacheBackend.MEMORY

    def test_get_redis(self) -> None:
        """Get redis backend."""
        backend = get_cache_backend("redis")
        assert backend == CacheBackend.REDIS

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cache backend"):
            get_cache_backend("invalid")


class TestCalculateCacheHitRate:
    """Tests for calculate_cache_hit_rate function."""

    def test_basic_calculation(self) -> None:
        """Basic hit rate calculation."""
        rate = calculate_cache_hit_rate(80, 20)
        assert rate == 0.8

    def test_all_hits(self) -> None:
        """All hits gives 1.0."""
        rate = calculate_cache_hit_rate(100, 0)
        assert rate == 1.0

    def test_all_misses(self) -> None:
        """All misses gives 0.0."""
        rate = calculate_cache_hit_rate(0, 100)
        assert rate == 0.0

    def test_no_requests(self) -> None:
        """No requests gives 0.0."""
        rate = calculate_cache_hit_rate(0, 0)
        assert rate == 0.0

    def test_negative_hits_raises(self) -> None:
        """Negative hits raises ValueError."""
        with pytest.raises(ValueError, match="hits cannot be negative"):
            calculate_cache_hit_rate(-1, 20)

    def test_negative_misses_raises(self) -> None:
        """Negative misses raises ValueError."""
        with pytest.raises(ValueError, match="misses cannot be negative"):
            calculate_cache_hit_rate(80, -1)


class TestEstimateLatencySavings:
    """Tests for estimate_latency_savings function."""

    def test_basic_calculation(self) -> None:
        """Basic savings calculation."""
        savings = estimate_latency_savings(0.8, 100.0, 1.0)
        assert savings > 0
        # With 80% hit rate: 0.8 * 1 + 0.2 * 100 = 20.8ms avg
        # Savings = (100 - 20.8) / 100 * 100 = 79.2%
        assert round(savings, 1) == 79.2

    def test_no_hits(self) -> None:
        """No hits gives 0% savings."""
        savings = estimate_latency_savings(0.0, 100.0)
        assert savings == 0.0

    def test_all_hits(self) -> None:
        """All hits gives maximum savings."""
        savings = estimate_latency_savings(1.0, 100.0, 1.0)
        # With 100% hit rate: all lookups from cache
        # Savings = (100 - 1) / 100 * 100 = 99%
        assert savings == 99.0

    def test_invalid_hit_rate_raises(self) -> None:
        """Invalid hit rate raises ValueError."""
        with pytest.raises(ValueError, match="hit_rate must be between"):
            estimate_latency_savings(1.5, 100.0)

    def test_zero_latency_raises(self) -> None:
        """Zero latency raises ValueError."""
        with pytest.raises(ValueError, match="avg_inference_latency_ms must be pos"):
            estimate_latency_savings(0.8, 0.0)

    def test_negative_latency_raises(self) -> None:
        """Negative latency raises ValueError."""
        with pytest.raises(ValueError, match="avg_inference_latency_ms must be pos"):
            estimate_latency_savings(0.8, -100.0)

    def test_negative_lookup_latency_raises(self) -> None:
        """Negative lookup latency raises ValueError."""
        with pytest.raises(ValueError, match=r"cache_lookup_latency_ms cannot be"):
            estimate_latency_savings(0.8, 100.0, -1.0)


class TestCalculateMemoryOverhead:
    """Tests for calculate_memory_overhead function."""

    def test_basic_calculation(self) -> None:
        """Basic memory calculation."""
        mem = calculate_memory_overhead(1000, 10.0)
        # 1000 * 10 * 1.2 / 1024 = 11.71875
        assert round(mem, 2) == 11.72

    def test_zero_entries(self) -> None:
        """Zero entries gives 0 memory."""
        mem = calculate_memory_overhead(0, 10.0)
        assert mem == 0.0

    def test_negative_entries_raises(self) -> None:
        """Negative entries raises ValueError."""
        with pytest.raises(ValueError, match="num_entries cannot be negative"):
            calculate_memory_overhead(-1, 10.0)

    def test_negative_size_raises(self) -> None:
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="avg_entry_size_kb cannot be negative"):
            calculate_memory_overhead(1000, -1.0)

    def test_invalid_overhead_factor_raises(self) -> None:
        """Overhead factor < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"overhead_factor must be at least 1\.0"):
            calculate_memory_overhead(1000, 10.0, overhead_factor=0.5)

    def test_custom_overhead_factor(self) -> None:
        """Custom overhead factor is applied."""
        mem1 = calculate_memory_overhead(1000, 10.0, overhead_factor=1.0)
        mem2 = calculate_memory_overhead(1000, 10.0, overhead_factor=2.0)
        assert mem2 == mem1 * 2


class TestOptimizeCacheSize:
    """Tests for optimize_cache_size function."""

    def test_basic_optimization(self) -> None:
        """Basic size optimization."""
        size = optimize_cache_size(1024.0, 0.9, 10.0, 5000)
        assert 0 < size <= 1024

    def test_small_prompts(self) -> None:
        """Small number of prompts gives small cache."""
        size = optimize_cache_size(1024.0, 0.9, 10.0, 100)
        # 100 * 0.9 = 90 entries * 10KB * 1.2 / 1024 = ~1MB
        assert size >= 1
        assert size <= 10

    def test_memory_constrained(self) -> None:
        """Memory constraint limits cache size."""
        size = optimize_cache_size(100.0, 0.9, 10.0, 100000)
        assert size <= 100

    def test_minimum_size(self) -> None:
        """Returns at least 1 MB."""
        size = optimize_cache_size(1024.0, 0.1, 0.1, 10)
        assert size >= 1

    def test_zero_memory_raises(self) -> None:
        """Zero memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_mb must be positive"):
            optimize_cache_size(0.0, 0.9, 10.0, 5000)

    def test_invalid_hit_rate_raises(self) -> None:
        """Invalid hit rate raises ValueError."""
        with pytest.raises(ValueError, match="target_hit_rate must be between"):
            optimize_cache_size(1024.0, 1.5, 10.0, 5000)

    def test_zero_entry_size_raises(self) -> None:
        """Zero entry size raises ValueError."""
        with pytest.raises(ValueError, match="avg_entry_size_kb must be positive"):
            optimize_cache_size(1024.0, 0.9, 0.0, 5000)

    def test_zero_prompts_raises(self) -> None:
        """Zero prompts raises ValueError."""
        with pytest.raises(ValueError, match="estimated_unique_prompts must be pos"):
            optimize_cache_size(1024.0, 0.9, 10.0, 0)


class TestFormatCacheStats:
    """Tests for format_cache_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = CacheStats(800, 200, 0.8, 50, 256.5)
        formatted = format_cache_stats(stats)
        assert "Hits: 800" in formatted
        assert "Misses: 200" in formatted
        assert "Total: 1000" in formatted
        assert "Hit Rate: 80.00%" in formatted
        assert "Evictions: 50" in formatted
        assert "Memory Used: 256.50 MB" in formatted

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = CacheStats(0, 0, 0.0, 0, 0.0)
        formatted = format_cache_stats(stats)
        assert "Hits: 0" in formatted
        assert "Hit Rate: 0.00%" in formatted

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = CacheStats(800, 200, 0.8, 50, 256.5)
        formatted = format_cache_stats(stats)
        lines = formatted.split("\n")
        assert len(lines) == 7


class TestGetRecommendedCacheConfig:
    """Tests for get_recommended_cache_config function."""

    def test_chatbot_config(self) -> None:
        """Chatbot use case returns semantic cache."""
        config = get_recommended_cache_config("chatbot")
        assert config.cache_type == CacheType.SEMANTIC
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.ttl_seconds == 7200

    def test_batch_config(self) -> None:
        """Batch use case returns prefix cache."""
        config = get_recommended_cache_config("batch")
        assert config.cache_type == CacheType.PREFIX
        assert config.eviction_policy == EvictionPolicy.FIFO
        assert config.ttl_seconds == 1800

    def test_api_config(self) -> None:
        """API use case returns KV cache."""
        config = get_recommended_cache_config("api")
        assert config.cache_type == CacheType.KV
        assert config.eviction_policy == EvictionPolicy.LFU
        assert config.ttl_seconds == 3600

    def test_development_config(self) -> None:
        """Development use case returns prompt cache."""
        config = get_recommended_cache_config("development")
        assert config.cache_type == CacheType.PROMPT
        assert config.eviction_policy == EvictionPolicy.TTL
        assert config.ttl_seconds == 300

    def test_custom_memory(self) -> None:
        """Custom memory is applied."""
        config = get_recommended_cache_config("api", available_memory_mb=2048)
        assert config.max_size_mb == 2048

    def test_invalid_use_case_raises(self) -> None:
        """Invalid use case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_cache_config("invalid")

    def test_zero_memory_raises(self) -> None:
        """Zero memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_mb must be positive"):
            get_recommended_cache_config("chatbot", available_memory_mb=0)


class TestAllCacheTypes:
    """Test all cache types can be created."""

    @pytest.mark.parametrize("cache_type", list(VALID_CACHE_TYPES))
    def test_create_config_with_type(self, cache_type: str) -> None:
        """Config can be created with each type."""
        config = create_cache_config(cache_type=cache_type)  # type: ignore[arg-type]
        assert config.cache_type.value == cache_type


class TestAllEvictionPolicies:
    """Test all eviction policies can be created."""

    @pytest.mark.parametrize("policy", list(VALID_EVICTION_POLICIES))
    def test_create_config_with_policy(self, policy: str) -> None:
        """Config can be created with each policy."""
        config = create_cache_config(eviction_policy=policy)  # type: ignore[arg-type]
        assert config.eviction_policy.value == policy


class TestAllCacheBackends:
    """Test all cache backends can be retrieved."""

    @pytest.mark.parametrize("backend", list(VALID_CACHE_BACKENDS))
    def test_get_backend(self, backend: str) -> None:
        """Backend can be retrieved by name."""
        result = get_cache_backend(backend)
        assert result.value == backend
