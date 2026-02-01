"""Tests for inference.kv_cache module."""

from __future__ import annotations

import pytest

from hf_gtc.inference.kv_cache import (
    VALID_CACHE_COMPRESSIONS,
    VALID_EVICTION_POLICIES,
    VALID_PAGED_ATTENTION_MODES,
    CacheCompression,
    CacheEntry,
    CacheStats,
    EvictionPolicy,
    KVCacheManagementConfig,
    PagedAttentionConfig,
    PagedAttentionMode,
    calculate_cache_memory,
    calculate_compression_savings,
    calculate_optimal_page_size,
    create_cache_entry,
    create_kv_cache_management_config,
    create_paged_attention_config,
    estimate_max_sequences,
    format_cache_stats,
    get_cache_compression,
    get_eviction_policy,
    get_paged_attention_mode,
    get_recommended_cache_config,
    list_cache_compressions,
    list_eviction_policies,
    list_paged_attention_modes,
    validate_cache_entry,
    validate_kv_cache_management_config,
    validate_paged_attention_config,
)


class TestEvictionPolicy:
    """Tests for EvictionPolicy enum."""

    def test_all_policies_have_values(self) -> None:
        """All policies have string values."""
        for policy in EvictionPolicy:
            assert isinstance(policy.value, str)

    def test_lru_value(self) -> None:
        """LRU has correct value."""
        assert EvictionPolicy.LRU.value == "lru"

    def test_fifo_value(self) -> None:
        """FIFO has correct value."""
        assert EvictionPolicy.FIFO.value == "fifo"

    def test_lfu_value(self) -> None:
        """LFU has correct value."""
        assert EvictionPolicy.LFU.value == "lfu"

    def test_random_value(self) -> None:
        """RANDOM has correct value."""
        assert EvictionPolicy.RANDOM.value == "random"

    def test_attention_score_value(self) -> None:
        """ATTENTION_SCORE has correct value."""
        assert EvictionPolicy.ATTENTION_SCORE.value == "attention_score"

    def test_valid_policies_frozenset(self) -> None:
        """VALID_EVICTION_POLICIES is a frozenset."""
        assert isinstance(VALID_EVICTION_POLICIES, frozenset)


class TestCacheCompression:
    """Tests for CacheCompression enum."""

    def test_all_compressions_have_values(self) -> None:
        """All compressions have string values."""
        for comp in CacheCompression:
            assert isinstance(comp.value, str)

    def test_none_value(self) -> None:
        """NONE has correct value."""
        assert CacheCompression.NONE.value == "none"

    def test_quantize_fp16_value(self) -> None:
        """QUANTIZE_FP16 has correct value."""
        assert CacheCompression.QUANTIZE_FP16.value == "quantize_fp16"

    def test_quantize_int8_value(self) -> None:
        """QUANTIZE_INT8 has correct value."""
        assert CacheCompression.QUANTIZE_INT8.value == "quantize_int8"

    def test_sparse_value(self) -> None:
        """SPARSE has correct value."""
        assert CacheCompression.SPARSE.value == "sparse"

    def test_valid_compressions_frozenset(self) -> None:
        """VALID_CACHE_COMPRESSIONS is a frozenset."""
        assert isinstance(VALID_CACHE_COMPRESSIONS, frozenset)


class TestPagedAttentionMode:
    """Tests for PagedAttentionMode enum."""

    def test_all_modes_have_values(self) -> None:
        """All modes have string values."""
        for mode in PagedAttentionMode:
            assert isinstance(mode.value, str)

    def test_disabled_value(self) -> None:
        """DISABLED has correct value."""
        assert PagedAttentionMode.DISABLED.value == "disabled"

    def test_static_value(self) -> None:
        """STATIC has correct value."""
        assert PagedAttentionMode.STATIC.value == "static"

    def test_dynamic_value(self) -> None:
        """DYNAMIC has correct value."""
        assert PagedAttentionMode.DYNAMIC.value == "dynamic"

    def test_valid_modes_frozenset(self) -> None:
        """VALID_PAGED_ATTENTION_MODES is a frozenset."""
        assert isinstance(VALID_PAGED_ATTENTION_MODES, frozenset)


class TestKVCacheManagementConfig:
    """Tests for KVCacheManagementConfig dataclass."""

    def test_create_config(self) -> None:
        """Create KV cache management config."""
        config = KVCacheManagementConfig(
            max_cache_size_gb=8.0,
            eviction_policy=EvictionPolicy.LRU,
            compression=CacheCompression.NONE,
            dtype="float16",
        )
        assert config.max_cache_size_gb == pytest.approx(8.0)
        assert config.eviction_policy == EvictionPolicy.LRU

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = KVCacheManagementConfig(
            max_cache_size_gb=8.0,
            eviction_policy=EvictionPolicy.LRU,
            compression=CacheCompression.NONE,
            dtype="float16",
        )
        with pytest.raises(AttributeError):
            config.max_cache_size_gb = 16.0  # type: ignore[misc]


class TestPagedAttentionConfig:
    """Tests for PagedAttentionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create paged attention config."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=16,
            max_pages=1024,
            block_size=256,
        )
        assert config.page_size == 16
        assert config.mode == PagedAttentionMode.DYNAMIC

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=16,
            max_pages=1024,
            block_size=256,
        )
        with pytest.raises(AttributeError):
            config.page_size = 32  # type: ignore[misc]


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_create_stats(self) -> None:
        """Create cache stats."""
        stats = CacheStats(
            hit_rate=0.95,
            eviction_count=100,
            memory_usage_gb=4.5,
            compression_ratio=0.5,
        )
        assert stats.hit_rate == pytest.approx(0.95)
        assert stats.eviction_count == 100

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = CacheStats(0.95, 100, 4.5, 0.5)
        with pytest.raises(AttributeError):
            stats.hit_rate = 0.99  # type: ignore[misc]


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self) -> None:
        """Create cache entry."""
        entry = CacheEntry(
            sequence_id="seq_001",
            num_tokens=512,
            memory_bytes=1048576,
            last_access_time=1704067200.0,
        )
        assert entry.sequence_id == "seq_001"
        assert entry.num_tokens == 512

    def test_entry_is_frozen(self) -> None:
        """Entry is immutable."""
        entry = CacheEntry("seq_001", 512, 1048576, 1704067200.0)
        with pytest.raises(AttributeError):
            entry.num_tokens = 1024  # type: ignore[misc]


class TestValidateKVCacheManagementConfig:
    """Tests for validate_kv_cache_management_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = KVCacheManagementConfig(
            max_cache_size_gb=8.0,
            eviction_policy=EvictionPolicy.LRU,
            compression=CacheCompression.NONE,
            dtype="float16",
        )
        validate_kv_cache_management_config(config)

    def test_zero_cache_size_raises(self) -> None:
        """Zero cache size raises ValueError."""
        config = KVCacheManagementConfig(
            max_cache_size_gb=0.0,
            eviction_policy=EvictionPolicy.LRU,
            compression=CacheCompression.NONE,
            dtype="float16",
        )
        with pytest.raises(ValueError, match="max_cache_size_gb must be positive"):
            validate_kv_cache_management_config(config)

    def test_negative_cache_size_raises(self) -> None:
        """Negative cache size raises ValueError."""
        config = KVCacheManagementConfig(
            max_cache_size_gb=-1.0,
            eviction_policy=EvictionPolicy.LRU,
            compression=CacheCompression.NONE,
            dtype="float16",
        )
        with pytest.raises(ValueError, match="max_cache_size_gb must be positive"):
            validate_kv_cache_management_config(config)

    def test_invalid_dtype_raises(self) -> None:
        """Invalid dtype raises ValueError."""
        config = KVCacheManagementConfig(
            max_cache_size_gb=8.0,
            eviction_policy=EvictionPolicy.LRU,
            compression=CacheCompression.NONE,
            dtype="invalid",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="dtype must be one of"):
            validate_kv_cache_management_config(config)


class TestValidatePagedAttentionConfig:
    """Tests for validate_paged_attention_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=16,
            max_pages=1024,
            block_size=256,
        )
        validate_paged_attention_config(config)

    def test_zero_page_size_raises(self) -> None:
        """Zero page size raises ValueError."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=0,
            max_pages=1024,
            block_size=256,
        )
        with pytest.raises(ValueError, match="page_size must be positive"):
            validate_paged_attention_config(config)

    def test_zero_max_pages_raises(self) -> None:
        """Zero max pages raises ValueError."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=16,
            max_pages=0,
            block_size=256,
        )
        with pytest.raises(ValueError, match="max_pages must be positive"):
            validate_paged_attention_config(config)

    def test_zero_block_size_raises(self) -> None:
        """Zero block size raises ValueError."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=16,
            max_pages=1024,
            block_size=0,
        )
        with pytest.raises(ValueError, match="block_size must be positive"):
            validate_paged_attention_config(config)

    def test_negative_page_size_raises(self) -> None:
        """Negative page size raises ValueError."""
        config = PagedAttentionConfig(
            mode=PagedAttentionMode.DYNAMIC,
            page_size=-1,
            max_pages=1024,
            block_size=256,
        )
        with pytest.raises(ValueError, match="page_size must be positive"):
            validate_paged_attention_config(config)


class TestValidateCacheEntry:
    """Tests for validate_cache_entry function."""

    def test_valid_entry(self) -> None:
        """Valid entry passes validation."""
        entry = CacheEntry("seq_001", 512, 1048576, 1704067200.0)
        validate_cache_entry(entry)

    def test_empty_sequence_id_raises(self) -> None:
        """Empty sequence ID raises ValueError."""
        entry = CacheEntry("", 512, 1048576, 1704067200.0)
        with pytest.raises(ValueError, match="sequence_id cannot be empty"):
            validate_cache_entry(entry)

    def test_negative_num_tokens_raises(self) -> None:
        """Negative num_tokens raises ValueError."""
        entry = CacheEntry("seq_001", -1, 1048576, 1704067200.0)
        with pytest.raises(ValueError, match="num_tokens cannot be negative"):
            validate_cache_entry(entry)

    def test_negative_memory_bytes_raises(self) -> None:
        """Negative memory_bytes raises ValueError."""
        entry = CacheEntry("seq_001", 512, -1, 1704067200.0)
        with pytest.raises(ValueError, match="memory_bytes cannot be negative"):
            validate_cache_entry(entry)

    def test_negative_last_access_time_raises(self) -> None:
        """Negative last_access_time raises ValueError."""
        entry = CacheEntry("seq_001", 512, 1048576, -1.0)
        with pytest.raises(ValueError, match="last_access_time cannot be negative"):
            validate_cache_entry(entry)

    def test_zero_values_valid(self) -> None:
        """Zero values (except sequence_id) are valid."""
        entry = CacheEntry("seq_001", 0, 0, 0.0)
        validate_cache_entry(entry)


class TestCreateKVCacheManagementConfig:
    """Tests for create_kv_cache_management_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_kv_cache_management_config()
        assert config.max_cache_size_gb == pytest.approx(8.0)
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.compression == CacheCompression.NONE
        assert config.dtype == "float16"

    def test_custom_cache_size(self) -> None:
        """Create config with custom cache size."""
        config = create_kv_cache_management_config(max_cache_size_gb=16.0)
        assert config.max_cache_size_gb == pytest.approx(16.0)

    def test_custom_eviction_policy(self) -> None:
        """Create config with custom eviction policy."""
        config = create_kv_cache_management_config(eviction_policy="lfu")
        assert config.eviction_policy == EvictionPolicy.LFU

    def test_custom_compression(self) -> None:
        """Create config with custom compression."""
        config = create_kv_cache_management_config(compression="quantize_int8")
        assert config.compression == CacheCompression.QUANTIZE_INT8

    def test_custom_dtype(self) -> None:
        """Create config with custom dtype."""
        config = create_kv_cache_management_config(dtype="float32")
        assert config.dtype == "float32"

    def test_invalid_eviction_policy_raises(self) -> None:
        """Invalid eviction policy raises ValueError."""
        with pytest.raises(ValueError, match="eviction_policy must be one of"):
            create_kv_cache_management_config(eviction_policy="invalid")  # type: ignore[arg-type]

    def test_invalid_compression_raises(self) -> None:
        """Invalid compression raises ValueError."""
        with pytest.raises(ValueError, match="compression must be one of"):
            create_kv_cache_management_config(compression="invalid")  # type: ignore[arg-type]

    def test_zero_cache_size_raises(self) -> None:
        """Zero cache size raises ValueError."""
        with pytest.raises(ValueError, match="max_cache_size_gb must be positive"):
            create_kv_cache_management_config(max_cache_size_gb=0.0)


class TestCreatePagedAttentionConfig:
    """Tests for create_paged_attention_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_paged_attention_config()
        assert config.mode == PagedAttentionMode.DYNAMIC
        assert config.page_size == 16
        assert config.max_pages == 1024
        assert config.block_size == 256

    def test_custom_mode(self) -> None:
        """Create config with custom mode."""
        config = create_paged_attention_config(mode="static")
        assert config.mode == PagedAttentionMode.STATIC

    def test_custom_page_size(self) -> None:
        """Create config with custom page size."""
        config = create_paged_attention_config(page_size=32)
        assert config.page_size == 32

    def test_custom_max_pages(self) -> None:
        """Create config with custom max pages."""
        config = create_paged_attention_config(max_pages=2048)
        assert config.max_pages == 2048

    def test_custom_block_size(self) -> None:
        """Create config with custom block size."""
        config = create_paged_attention_config(block_size=512)
        assert config.block_size == 512

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            create_paged_attention_config(mode="invalid")  # type: ignore[arg-type]

    def test_zero_page_size_raises(self) -> None:
        """Zero page size raises ValueError."""
        with pytest.raises(ValueError, match="page_size must be positive"):
            create_paged_attention_config(page_size=0)

    def test_zero_max_pages_raises(self) -> None:
        """Zero max pages raises ValueError."""
        with pytest.raises(ValueError, match="max_pages must be positive"):
            create_paged_attention_config(max_pages=0)

    def test_zero_block_size_raises(self) -> None:
        """Zero block size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            create_paged_attention_config(block_size=0)


class TestCreateCacheEntry:
    """Tests for create_cache_entry function."""

    def test_create_entry(self) -> None:
        """Create cache entry."""
        entry = create_cache_entry("seq_001", 512, 1048576, 1704067200.0)
        assert entry.sequence_id == "seq_001"
        assert entry.num_tokens == 512
        assert entry.memory_bytes == 1048576
        assert entry.last_access_time == pytest.approx(1704067200.0)

    def test_empty_sequence_id_raises(self) -> None:
        """Empty sequence ID raises ValueError."""
        with pytest.raises(ValueError, match="sequence_id cannot be empty"):
            create_cache_entry("", 512, 1048576, 1704067200.0)


class TestListEvictionPolicies:
    """Tests for list_eviction_policies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        policies = list_eviction_policies()
        assert policies == sorted(policies)

    def test_contains_lru(self) -> None:
        """Contains LRU."""
        policies = list_eviction_policies()
        assert "lru" in policies

    def test_contains_all_policies(self) -> None:
        """Contains all policies."""
        policies = list_eviction_policies()
        assert len(policies) == len(VALID_EVICTION_POLICIES)


class TestListCacheCompressions:
    """Tests for list_cache_compressions function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        compressions = list_cache_compressions()
        assert compressions == sorted(compressions)

    def test_contains_none(self) -> None:
        """Contains none."""
        compressions = list_cache_compressions()
        assert "none" in compressions

    def test_contains_all_compressions(self) -> None:
        """Contains all compressions."""
        compressions = list_cache_compressions()
        assert len(compressions) == len(VALID_CACHE_COMPRESSIONS)


class TestListPagedAttentionModes:
    """Tests for list_paged_attention_modes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        modes = list_paged_attention_modes()
        assert modes == sorted(modes)

    def test_contains_dynamic(self) -> None:
        """Contains dynamic."""
        modes = list_paged_attention_modes()
        assert "dynamic" in modes

    def test_contains_all_modes(self) -> None:
        """Contains all modes."""
        modes = list_paged_attention_modes()
        assert len(modes) == len(VALID_PAGED_ATTENTION_MODES)


class TestGetEvictionPolicy:
    """Tests for get_eviction_policy function."""

    def test_get_lru(self) -> None:
        """Get LRU policy."""
        policy = get_eviction_policy("lru")
        assert policy == EvictionPolicy.LRU

    def test_get_attention_score(self) -> None:
        """Get attention score policy."""
        policy = get_eviction_policy("attention_score")
        assert policy == EvictionPolicy.ATTENTION_SCORE

    def test_invalid_policy_raises(self) -> None:
        """Invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown eviction policy"):
            get_eviction_policy("invalid")


class TestGetCacheCompression:
    """Tests for get_cache_compression function."""

    def test_get_none(self) -> None:
        """Get none compression."""
        comp = get_cache_compression("none")
        assert comp == CacheCompression.NONE

    def test_get_quantize_fp16(self) -> None:
        """Get quantize_fp16 compression."""
        comp = get_cache_compression("quantize_fp16")
        assert comp == CacheCompression.QUANTIZE_FP16

    def test_invalid_compression_raises(self) -> None:
        """Invalid compression raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cache compression"):
            get_cache_compression("invalid")


class TestGetPagedAttentionMode:
    """Tests for get_paged_attention_mode function."""

    def test_get_dynamic(self) -> None:
        """Get dynamic mode."""
        mode = get_paged_attention_mode("dynamic")
        assert mode == PagedAttentionMode.DYNAMIC

    def test_get_static(self) -> None:
        """Get static mode."""
        mode = get_paged_attention_mode("static")
        assert mode == PagedAttentionMode.STATIC

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown paged attention mode"):
            get_paged_attention_mode("invalid")


class TestCalculateCacheMemory:
    """Tests for calculate_cache_memory function."""

    def test_basic_calculation(self) -> None:
        """Basic memory calculation."""
        mem = calculate_cache_memory(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_seq_length=2048,
            batch_size=1,
            dtype="float16",
        )
        assert mem > 0

    def test_expected_value(self) -> None:
        """Expected memory value for known configuration."""
        mem = calculate_cache_memory(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_seq_length=2048,
            batch_size=1,
            dtype="float16",
        )
        # 2 * 32 * 1 * 2048 * 32 * 128 * 2 bytes = 1073741824 bytes = 1 GB
        assert round(mem, 2) == pytest.approx(1.0)

    def test_larger_batch_more_memory(self) -> None:
        """Larger batch uses more memory."""
        mem1 = calculate_cache_memory(32, 32, 128, 2048, batch_size=1)
        mem4 = calculate_cache_memory(32, 32, 128, 2048, batch_size=4)
        assert mem4 == mem1 * 4

    def test_float32_double_memory(self) -> None:
        """Float32 uses double the memory of float16."""
        mem16 = calculate_cache_memory(32, 32, 128, 2048, dtype="float16")
        mem32 = calculate_cache_memory(32, 32, 128, 2048, dtype="float32")
        assert mem32 == mem16 * 2

    def test_int8_half_memory(self) -> None:
        """Int8 uses half the memory of float16."""
        mem16 = calculate_cache_memory(32, 32, 128, 2048, dtype="float16")
        mem8 = calculate_cache_memory(32, 32, 128, 2048, dtype="int8")
        assert mem8 == mem16 / 2

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            calculate_cache_memory(0, 32, 128, 2048)

    def test_zero_heads_raises(self) -> None:
        """Zero heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            calculate_cache_memory(32, 0, 128, 2048)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            calculate_cache_memory(32, 32, 0, 2048)

    def test_zero_seq_length_raises(self) -> None:
        """Zero seq length raises ValueError."""
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            calculate_cache_memory(32, 32, 128, 0)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_cache_memory(32, 32, 128, 2048, batch_size=0)


class TestEstimateMaxSequences:
    """Tests for estimate_max_sequences function."""

    def test_basic_estimation(self) -> None:
        """Basic sequence estimation."""
        max_seqs = estimate_max_sequences(
            available_memory_gb=16.0,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            avg_seq_length=512,
        )
        assert max_seqs > 0

    def test_more_memory_more_sequences(self) -> None:
        """More memory allows more sequences."""
        seqs16 = estimate_max_sequences(16.0, 32, 32, 128, 512)
        seqs32 = estimate_max_sequences(32.0, 32, 32, 128, 512)
        assert seqs32 > seqs16

    def test_longer_sequences_fewer_fits(self) -> None:
        """Longer sequences reduce count."""
        seqs512 = estimate_max_sequences(16.0, 32, 32, 128, 512)
        seqs2048 = estimate_max_sequences(16.0, 32, 32, 128, 2048)
        assert seqs512 > seqs2048

    def test_zero_memory_raises(self) -> None:
        """Zero memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_gb must be positive"):
            estimate_max_sequences(0.0, 32, 32, 128, 512)

    def test_invalid_memory_fraction_raises(self) -> None:
        """Invalid memory fraction raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction must be in"):
            estimate_max_sequences(16.0, 32, 32, 128, 512, memory_fraction=1.5)

    def test_minimum_one_sequence(self) -> None:
        """Returns at least one sequence."""
        max_seqs = estimate_max_sequences(
            available_memory_gb=0.001,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            avg_seq_length=512,
        )
        assert max_seqs >= 1


class TestCalculateOptimalPageSize:
    """Tests for calculate_optimal_page_size function."""

    def test_basic_calculation(self) -> None:
        """Basic page size calculation."""
        page_size = calculate_optimal_page_size(
            head_dim=128,
            num_heads=32,
            target_memory_kb=64,
        )
        assert page_size > 0

    def test_expected_value(self) -> None:
        """Expected page size for known configuration."""
        # 4 * 32 * 128 = 16384 bytes per token
        # 64 * 1024 = 65536 bytes target
        # 65536 / 16384 = 4 tokens, nearest power of 2 = 4
        page_size = calculate_optimal_page_size(128, 32, 64)
        assert page_size == 4

    def test_power_of_two(self) -> None:
        """Page size is power of two."""
        page_size = calculate_optimal_page_size(128, 32, 64)
        # Check if power of 2
        assert page_size > 0
        assert (page_size & (page_size - 1)) == 0

    def test_zero_head_dim_raises(self) -> None:
        """Zero head dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            calculate_optimal_page_size(0, 32)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            calculate_optimal_page_size(128, 0)

    def test_zero_target_memory_raises(self) -> None:
        """Zero target memory raises ValueError."""
        with pytest.raises(ValueError, match="target_memory_kb must be positive"):
            calculate_optimal_page_size(128, 32, 0)

    def test_small_target_returns_one(self) -> None:
        """Very small target memory returns 1."""
        # With very small target memory relative to head dimensions
        # page_size calc will be 0, should return 1
        page_size = calculate_optimal_page_size(
            head_dim=4096,
            num_heads=128,
            target_memory_kb=1,  # Only 1KB target
        )
        # 4 * 128 * 4096 = 2097152 bytes per token
        # 1024 bytes target / 2097152 = 0 (floor division)
        # Should return 1
        assert page_size == 1


class TestCalculateCompressionSavings:
    """Tests for calculate_compression_savings function."""

    def test_float32_to_float16(self) -> None:
        """Float32 to float16 gives 0.5 ratio."""
        ratio = calculate_compression_savings("float32", "float16")
        assert ratio == pytest.approx(0.5)

    def test_float16_to_int8(self) -> None:
        """Float16 to int8 gives 0.5 ratio."""
        ratio = calculate_compression_savings("float16", "int8")
        assert ratio == pytest.approx(0.5)

    def test_same_dtype(self) -> None:
        """Same dtype gives 1.0 ratio."""
        ratio = calculate_compression_savings("float16", "float16")
        assert ratio == pytest.approx(1.0)

    def test_with_sparsity(self) -> None:
        """Sparsity reduces ratio further."""
        ratio = calculate_compression_savings("float16", "float16", sparsity_ratio=0.5)
        assert ratio == pytest.approx(0.5)

    def test_full_sparsity(self) -> None:
        """Full sparsity gives 0 ratio."""
        ratio = calculate_compression_savings("float16", "float16", sparsity_ratio=1.0)
        assert ratio == pytest.approx(0.0)

    def test_invalid_sparsity_raises(self) -> None:
        """Invalid sparsity raises ValueError."""
        with pytest.raises(ValueError, match="sparsity_ratio must be between"):
            calculate_compression_savings("float32", "float16", sparsity_ratio=1.5)

    def test_invalid_original_dtype_raises(self) -> None:
        """Invalid original dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unknown original_dtype"):
            calculate_compression_savings("invalid", "float16")

    def test_invalid_compressed_dtype_raises(self) -> None:
        """Invalid compressed dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compressed_dtype"):
            calculate_compression_savings("float32", "invalid")


class TestFormatCacheStats:
    """Tests for format_cache_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = CacheStats(
            hit_rate=0.95,
            eviction_count=100,
            memory_usage_gb=4.5,
            compression_ratio=0.5,
        )
        formatted = format_cache_stats(stats)
        assert "Hit Rate: 95.00%" in formatted
        assert "Memory Usage: 4.50 GB" in formatted
        assert "Eviction Count: 100" in formatted
        assert "Compression Ratio: 0.50" in formatted

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = CacheStats(0.0, 0, 0.0, 1.0)
        formatted = format_cache_stats(stats)
        assert "Hit Rate: 0.00%" in formatted
        assert "Eviction Count: 0" in formatted

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = CacheStats(0.95, 100, 4.5, 0.5)
        formatted = format_cache_stats(stats)
        lines = formatted.split("\n")
        assert len(lines) == 4


class TestGetRecommendedCacheConfig:
    """Tests for get_recommended_cache_config function."""

    def test_plenty_of_memory(self) -> None:
        """Plenty of memory returns no compression."""
        config = get_recommended_cache_config(
            model_size_gb=7.0,
            available_memory_gb=24.0,
        )
        assert config.max_cache_size_gb > 0
        assert config.compression == CacheCompression.NONE
        assert config.eviction_policy == EvictionPolicy.LRU

    def test_tight_memory(self) -> None:
        """Tight memory returns aggressive compression."""
        config = get_recommended_cache_config(
            model_size_gb=70.0,
            available_memory_gb=80.0,
        )
        assert config.compression == CacheCompression.QUANTIZE_INT8

    def test_moderate_memory(self) -> None:
        """Moderate memory returns FP16 compression."""
        config = get_recommended_cache_config(
            model_size_gb=7.0,
            available_memory_gb=14.0,
        )
        assert config.compression == CacheCompression.QUANTIZE_FP16

    def test_zero_model_size_raises(self) -> None:
        """Zero model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            get_recommended_cache_config(0.0, 24.0)

    def test_zero_memory_raises(self) -> None:
        """Zero memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_gb must be positive"):
            get_recommended_cache_config(7.0, 0.0)

    def test_minimum_cache_size(self) -> None:
        """Cache size has minimum value."""
        config = get_recommended_cache_config(
            model_size_gb=100.0,
            available_memory_gb=100.0,
        )
        assert config.max_cache_size_gb >= 0.5


class TestAllEvictionPolicies:
    """Test all eviction policies can be created."""

    @pytest.mark.parametrize("policy", list(VALID_EVICTION_POLICIES))
    def test_create_config_with_policy(self, policy: str) -> None:
        """Config can be created with each policy."""
        config = create_kv_cache_management_config(eviction_policy=policy)  # type: ignore[arg-type]
        assert config.eviction_policy.value == policy


class TestAllCacheCompressions:
    """Test all cache compressions can be created."""

    @pytest.mark.parametrize("compression", list(VALID_CACHE_COMPRESSIONS))
    def test_create_config_with_compression(self, compression: str) -> None:
        """Config can be created with each compression."""
        config = create_kv_cache_management_config(compression=compression)  # type: ignore[arg-type]
        assert config.compression.value == compression


class TestAllPagedAttentionModes:
    """Test all paged attention modes can be created."""

    @pytest.mark.parametrize("mode", list(VALID_PAGED_ATTENTION_MODES))
    def test_create_config_with_mode(self, mode: str) -> None:
        """Config can be created with each mode."""
        config = create_paged_attention_config(mode=mode)  # type: ignore[arg-type]
        assert config.mode.value == mode
