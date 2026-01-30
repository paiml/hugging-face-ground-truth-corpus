"""Tests for models.attention module."""

from __future__ import annotations

import pytest

from hf_gtc.models.attention import (
    VALID_ATTENTION_IMPLEMENTATIONS,
    VALID_ATTENTION_MASKS,
    VALID_ATTENTION_TYPES,
    AttentionConfig,
    AttentionImplementation,
    AttentionMask,
    AttentionStats,
    AttentionType,
    FlashAttentionConfig,
    GroupedQueryConfig,
    calculate_attention_memory,
    calculate_kv_cache_size,
    create_attention_config,
    create_attention_stats,
    create_flash_attention_config,
    create_grouped_query_config,
    estimate_attention_flops,
    format_attention_stats,
    get_attention_implementation,
    get_attention_mask,
    get_attention_type,
    get_recommended_attention_config,
    list_attention_implementations,
    list_attention_masks,
    list_attention_types,
    select_attention_implementation,
    validate_attention_config,
    validate_flash_attention_config,
    validate_grouped_query_config,
)


class TestAttentionType:
    """Tests for AttentionType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for attention_type in AttentionType:
            assert isinstance(attention_type.value, str)

    def test_standard_value(self) -> None:
        """Standard has correct value."""
        assert AttentionType.STANDARD.value == "standard"

    def test_flash_value(self) -> None:
        """Flash has correct value."""
        assert AttentionType.FLASH.value == "flash"

    def test_flash_v2_value(self) -> None:
        """Flash v2 has correct value."""
        assert AttentionType.FLASH_V2.value == "flash_v2"

    def test_multi_query_value(self) -> None:
        """Multi-query has correct value."""
        assert AttentionType.MULTI_QUERY.value == "multi_query"

    def test_grouped_query_value(self) -> None:
        """Grouped-query has correct value."""
        assert AttentionType.GROUPED_QUERY.value == "grouped_query"

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert AttentionType.LINEAR.value == "linear"

    def test_sparse_value(self) -> None:
        """Sparse has correct value."""
        assert AttentionType.SPARSE.value == "sparse"

    def test_valid_attention_types_frozenset(self) -> None:
        """VALID_ATTENTION_TYPES is a frozenset."""
        assert isinstance(VALID_ATTENTION_TYPES, frozenset)

    def test_valid_attention_types_contains_all(self) -> None:
        """VALID_ATTENTION_TYPES contains all enum values."""
        for t in AttentionType:
            assert t.value in VALID_ATTENTION_TYPES


class TestAttentionMask:
    """Tests for AttentionMask enum."""

    def test_all_masks_have_values(self) -> None:
        """All masks have string values."""
        for mask in AttentionMask:
            assert isinstance(mask.value, str)

    def test_causal_value(self) -> None:
        """Causal has correct value."""
        assert AttentionMask.CAUSAL.value == "causal"

    def test_bidirectional_value(self) -> None:
        """Bidirectional has correct value."""
        assert AttentionMask.BIDIRECTIONAL.value == "bidirectional"

    def test_prefix_value(self) -> None:
        """Prefix has correct value."""
        assert AttentionMask.PREFIX.value == "prefix"

    def test_custom_value(self) -> None:
        """Custom has correct value."""
        assert AttentionMask.CUSTOM.value == "custom"

    def test_valid_attention_masks_frozenset(self) -> None:
        """VALID_ATTENTION_MASKS is a frozenset."""
        assert isinstance(VALID_ATTENTION_MASKS, frozenset)

    def test_valid_attention_masks_contains_all(self) -> None:
        """VALID_ATTENTION_MASKS contains all enum values."""
        for m in AttentionMask:
            assert m.value in VALID_ATTENTION_MASKS


class TestAttentionImplementation:
    """Tests for AttentionImplementation enum."""

    def test_all_implementations_have_values(self) -> None:
        """All implementations have string values."""
        for impl in AttentionImplementation:
            assert isinstance(impl.value, str)

    def test_eager_value(self) -> None:
        """Eager has correct value."""
        assert AttentionImplementation.EAGER.value == "eager"

    def test_sdpa_value(self) -> None:
        """SDPA has correct value."""
        assert AttentionImplementation.SDPA.value == "sdpa"

    def test_flash_attention_2_value(self) -> None:
        """Flash Attention 2 has correct value."""
        assert AttentionImplementation.FLASH_ATTENTION_2.value == "flash_attention_2"

    def test_valid_attention_implementations_frozenset(self) -> None:
        """VALID_ATTENTION_IMPLEMENTATIONS is a frozenset."""
        assert isinstance(VALID_ATTENTION_IMPLEMENTATIONS, frozenset)

    def test_valid_attention_implementations_contains_all(self) -> None:
        """VALID_ATTENTION_IMPLEMENTATIONS contains all enum values."""
        for i in AttentionImplementation:
            assert i.value in VALID_ATTENTION_IMPLEMENTATIONS


class TestAttentionConfig:
    """Tests for AttentionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create attention config."""
        config = AttentionConfig(
            attention_type=AttentionType.STANDARD,
            num_heads=32,
            head_dim=128,
            dropout=0.0,
            use_bias=True,
        )
        assert config.attention_type == AttentionType.STANDARD
        assert config.num_heads == 32
        assert config.head_dim == 128

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = AttentionConfig(AttentionType.STANDARD, 32, 128, 0.0, True)
        with pytest.raises(AttributeError):
            config.num_heads = 64  # type: ignore[misc]


class TestFlashAttentionConfig:
    """Tests for FlashAttentionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create flash attention config."""
        config = FlashAttentionConfig(
            window_size=-1,
            causal=True,
            softmax_scale=None,
        )
        assert config.window_size == -1
        assert config.causal is True
        assert config.softmax_scale is None

    def test_config_with_window(self) -> None:
        """Create config with window size."""
        config = FlashAttentionConfig(4096, True, 0.125)
        assert config.window_size == 4096
        assert config.softmax_scale == 0.125

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FlashAttentionConfig(-1, True, None)
        with pytest.raises(AttributeError):
            config.causal = False  # type: ignore[misc]


class TestGroupedQueryConfig:
    """Tests for GroupedQueryConfig dataclass."""

    def test_create_config(self) -> None:
        """Create grouped query config."""
        config = GroupedQueryConfig(
            num_kv_heads=8,
            num_query_groups=4,
        )
        assert config.num_kv_heads == 8
        assert config.num_query_groups == 4

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = GroupedQueryConfig(8, 4)
        with pytest.raises(AttributeError):
            config.num_kv_heads = 16  # type: ignore[misc]


class TestAttentionStats:
    """Tests for AttentionStats dataclass."""

    def test_create_stats(self) -> None:
        """Create attention stats."""
        stats = AttentionStats(
            memory_usage_mb=1024.0,
            flops=1e12,
            throughput=10000.0,
        )
        assert stats.memory_usage_mb == 1024.0
        assert stats.flops == 1e12
        assert stats.throughput == 10000.0

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = AttentionStats(1024.0, 1e12, 10000.0)
        with pytest.raises(AttributeError):
            stats.memory_usage_mb = 2048.0  # type: ignore[misc]


class TestValidateAttentionConfig:
    """Tests for validate_attention_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AttentionConfig(AttentionType.STANDARD, 32, 128, 0.0, True)
        validate_attention_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_attention_config(None)  # type: ignore[arg-type]

    def test_zero_num_heads_raises(self) -> None:
        """Zero num_heads raises ValueError."""
        config = AttentionConfig(AttentionType.STANDARD, 0, 128, 0.0, True)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_attention_config(config)

    def test_negative_num_heads_raises(self) -> None:
        """Negative num_heads raises ValueError."""
        config = AttentionConfig(AttentionType.STANDARD, -1, 128, 0.0, True)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_attention_config(config)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head_dim raises ValueError."""
        config = AttentionConfig(AttentionType.STANDARD, 32, 0, 0.0, True)
        with pytest.raises(ValueError, match="head_dim must be positive"):
            validate_attention_config(config)

    def test_negative_head_dim_raises(self) -> None:
        """Negative head_dim raises ValueError."""
        config = AttentionConfig(AttentionType.STANDARD, 32, -1, 0.0, True)
        with pytest.raises(ValueError, match="head_dim must be positive"):
            validate_attention_config(config)

    def test_negative_dropout_raises(self) -> None:
        """Negative dropout raises ValueError."""
        config = AttentionConfig(AttentionType.STANDARD, 32, 128, -0.1, True)
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_attention_config(config)

    def test_dropout_above_one_raises(self) -> None:
        """Dropout above 1 raises ValueError."""
        config = AttentionConfig(AttentionType.STANDARD, 32, 128, 1.1, True)
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            validate_attention_config(config)


class TestValidateFlashAttentionConfig:
    """Tests for validate_flash_attention_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FlashAttentionConfig(-1, True, None)
        validate_flash_attention_config(config)

    def test_valid_config_with_window(self) -> None:
        """Valid config with window passes validation."""
        config = FlashAttentionConfig(4096, True, 0.125)
        validate_flash_attention_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_flash_attention_config(None)  # type: ignore[arg-type]

    def test_invalid_window_size_raises(self) -> None:
        """Invalid window size raises ValueError."""
        config = FlashAttentionConfig(-2, True, None)
        with pytest.raises(ValueError, match="window_size must be -1 or positive"):
            validate_flash_attention_config(config)

    def test_zero_window_size_raises(self) -> None:
        """Zero window size raises ValueError."""
        config = FlashAttentionConfig(0, True, None)
        with pytest.raises(ValueError, match="window_size must be -1 or positive"):
            validate_flash_attention_config(config)

    def test_zero_softmax_scale_raises(self) -> None:
        """Zero softmax scale raises ValueError."""
        config = FlashAttentionConfig(-1, True, 0.0)
        with pytest.raises(ValueError, match="softmax_scale must be positive"):
            validate_flash_attention_config(config)

    def test_negative_softmax_scale_raises(self) -> None:
        """Negative softmax scale raises ValueError."""
        config = FlashAttentionConfig(-1, True, -0.1)
        with pytest.raises(ValueError, match="softmax_scale must be positive"):
            validate_flash_attention_config(config)


class TestValidateGroupedQueryConfig:
    """Tests for validate_grouped_query_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = GroupedQueryConfig(8, 4)
        validate_grouped_query_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_grouped_query_config(None)  # type: ignore[arg-type]

    def test_zero_num_kv_heads_raises(self) -> None:
        """Zero num_kv_heads raises ValueError."""
        config = GroupedQueryConfig(0, 4)
        with pytest.raises(ValueError, match="num_kv_heads must be positive"):
            validate_grouped_query_config(config)

    def test_negative_num_kv_heads_raises(self) -> None:
        """Negative num_kv_heads raises ValueError."""
        config = GroupedQueryConfig(-1, 4)
        with pytest.raises(ValueError, match="num_kv_heads must be positive"):
            validate_grouped_query_config(config)

    def test_zero_num_query_groups_raises(self) -> None:
        """Zero num_query_groups raises ValueError."""
        config = GroupedQueryConfig(8, 0)
        with pytest.raises(ValueError, match="num_query_groups must be positive"):
            validate_grouped_query_config(config)

    def test_negative_num_query_groups_raises(self) -> None:
        """Negative num_query_groups raises ValueError."""
        config = GroupedQueryConfig(8, -1)
        with pytest.raises(ValueError, match="num_query_groups must be positive"):
            validate_grouped_query_config(config)


class TestCreateAttentionConfig:
    """Tests for create_attention_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_attention_config()
        assert config.attention_type == AttentionType.STANDARD
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.dropout == 0.0
        assert config.use_bias is True

    def test_custom_config_with_string(self) -> None:
        """Create custom config with string type."""
        config = create_attention_config(
            attention_type="flash",
            num_heads=16,
            head_dim=64,
            dropout=0.1,
            use_bias=False,
        )
        assert config.attention_type == AttentionType.FLASH
        assert config.num_heads == 16
        assert config.head_dim == 64
        assert config.dropout == 0.1
        assert config.use_bias is False

    def test_custom_config_with_enum(self) -> None:
        """Create custom config with enum type."""
        config = create_attention_config(
            attention_type=AttentionType.GROUPED_QUERY,
            num_heads=64,
        )
        assert config.attention_type == AttentionType.GROUPED_QUERY
        assert config.num_heads == 64

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="attention type must be one of"):
            create_attention_config(attention_type="invalid")

    def test_zero_num_heads_raises(self) -> None:
        """Zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            create_attention_config(num_heads=0)


class TestCreateFlashAttentionConfig:
    """Tests for create_flash_attention_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_flash_attention_config()
        assert config.window_size == -1
        assert config.causal is True
        assert config.softmax_scale is None

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_flash_attention_config(
            window_size=4096,
            causal=False,
            softmax_scale=0.125,
        )
        assert config.window_size == 4096
        assert config.causal is False
        assert config.softmax_scale == 0.125

    def test_invalid_window_raises(self) -> None:
        """Invalid window raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be -1 or positive"):
            create_flash_attention_config(window_size=-2)


class TestCreateGroupedQueryConfig:
    """Tests for create_grouped_query_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_grouped_query_config()
        assert config.num_kv_heads == 8
        assert config.num_query_groups == 4

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_grouped_query_config(
            num_kv_heads=4,
            num_query_groups=8,
        )
        assert config.num_kv_heads == 4
        assert config.num_query_groups == 8

    def test_zero_kv_heads_raises(self) -> None:
        """Zero kv heads raises ValueError."""
        with pytest.raises(ValueError, match="num_kv_heads must be positive"):
            create_grouped_query_config(num_kv_heads=0)


class TestListAttentionTypes:
    """Tests for list_attention_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_attention_types()
        assert types == sorted(types)

    def test_contains_standard(self) -> None:
        """Contains standard."""
        types = list_attention_types()
        assert "standard" in types

    def test_contains_flash(self) -> None:
        """Contains flash."""
        types = list_attention_types()
        assert "flash" in types

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_attention_types()
        for t in AttentionType:
            assert t.value in types


class TestListAttentionMasks:
    """Tests for list_attention_masks function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        masks = list_attention_masks()
        assert masks == sorted(masks)

    def test_contains_causal(self) -> None:
        """Contains causal."""
        masks = list_attention_masks()
        assert "causal" in masks

    def test_contains_bidirectional(self) -> None:
        """Contains bidirectional."""
        masks = list_attention_masks()
        assert "bidirectional" in masks


class TestListAttentionImplementations:
    """Tests for list_attention_implementations function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        impls = list_attention_implementations()
        assert impls == sorted(impls)

    def test_contains_eager(self) -> None:
        """Contains eager."""
        impls = list_attention_implementations()
        assert "eager" in impls

    def test_contains_sdpa(self) -> None:
        """Contains sdpa."""
        impls = list_attention_implementations()
        assert "sdpa" in impls


class TestGetAttentionType:
    """Tests for get_attention_type function."""

    def test_get_standard(self) -> None:
        """Get standard type."""
        assert get_attention_type("standard") == AttentionType.STANDARD

    def test_get_flash(self) -> None:
        """Get flash type."""
        assert get_attention_type("flash") == AttentionType.FLASH

    def test_get_grouped_query(self) -> None:
        """Get grouped query type."""
        assert get_attention_type("grouped_query") == AttentionType.GROUPED_QUERY

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="attention type must be one of"):
            get_attention_type("invalid")


class TestGetAttentionMask:
    """Tests for get_attention_mask function."""

    def test_get_causal(self) -> None:
        """Get causal mask."""
        assert get_attention_mask("causal") == AttentionMask.CAUSAL

    def test_get_bidirectional(self) -> None:
        """Get bidirectional mask."""
        assert get_attention_mask("bidirectional") == AttentionMask.BIDIRECTIONAL

    def test_invalid_mask_raises(self) -> None:
        """Invalid mask raises ValueError."""
        with pytest.raises(ValueError, match="attention mask must be one of"):
            get_attention_mask("invalid")


class TestGetAttentionImplementation:
    """Tests for get_attention_implementation function."""

    def test_get_eager(self) -> None:
        """Get eager implementation."""
        assert get_attention_implementation("eager") == AttentionImplementation.EAGER

    def test_get_sdpa(self) -> None:
        """Get SDPA implementation."""
        assert get_attention_implementation("sdpa") == AttentionImplementation.SDPA

    def test_get_flash_attention_2(self) -> None:
        """Get Flash Attention 2 implementation."""
        assert (
            get_attention_implementation("flash_attention_2")
            == AttentionImplementation.FLASH_ATTENTION_2
        )

    def test_invalid_implementation_raises(self) -> None:
        """Invalid implementation raises ValueError."""
        with pytest.raises(ValueError, match="attention implementation must be one of"):
            get_attention_implementation("invalid")


class TestCalculateAttentionMemory:
    """Tests for calculate_attention_memory function."""

    def test_basic_calculation(self) -> None:
        """Basic memory calculation."""
        mem = calculate_attention_memory(1, 2048, 32, 128)
        assert mem > 0
        assert isinstance(mem, float)

    def test_larger_batch_more_memory(self) -> None:
        """Larger batch uses more memory."""
        mem1 = calculate_attention_memory(1, 2048, 32, 128)
        mem2 = calculate_attention_memory(4, 2048, 32, 128)
        assert mem2 > mem1

    def test_longer_sequence_more_memory(self) -> None:
        """Longer sequence uses more memory."""
        mem1 = calculate_attention_memory(1, 1024, 32, 128)
        mem2 = calculate_attention_memory(1, 2048, 32, 128)
        assert mem2 > mem1

    def test_fp32_more_memory(self) -> None:
        """FP32 uses more memory than FP16."""
        mem_fp16 = calculate_attention_memory(1, 2048, 32, 128, dtype_bytes=2)
        mem_fp32 = calculate_attention_memory(1, 2048, 32, 128, dtype_bytes=4)
        assert mem_fp32 > mem_fp16

    def test_zero_batch_raises(self) -> None:
        """Zero batch raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_attention_memory(0, 2048, 32, 128)

    def test_zero_seq_len_raises(self) -> None:
        """Zero seq_len raises ValueError."""
        with pytest.raises(ValueError, match="seq_len must be positive"):
            calculate_attention_memory(1, 0, 32, 128)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            calculate_attention_memory(1, 2048, 0, 128)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head_dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            calculate_attention_memory(1, 2048, 32, 0)

    def test_zero_dtype_bytes_raises(self) -> None:
        """Zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            calculate_attention_memory(1, 2048, 32, 128, dtype_bytes=0)


class TestEstimateAttentionFlops:
    """Tests for estimate_attention_flops function."""

    def test_basic_calculation(self) -> None:
        """Basic FLOPs calculation."""
        flops = estimate_attention_flops(1, 2048, 32, 128)
        assert flops > 0
        assert isinstance(flops, float)

    def test_larger_batch_more_flops(self) -> None:
        """Larger batch uses more FLOPs."""
        flops1 = estimate_attention_flops(1, 2048, 32, 128)
        flops2 = estimate_attention_flops(4, 2048, 32, 128)
        assert flops2 > flops1

    def test_longer_sequence_more_flops(self) -> None:
        """Longer sequence uses more FLOPs."""
        flops1 = estimate_attention_flops(1, 1024, 32, 128)
        flops2 = estimate_attention_flops(1, 2048, 32, 128)
        assert flops2 > flops1

    def test_zero_batch_raises(self) -> None:
        """Zero batch raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_attention_flops(0, 2048, 32, 128)

    def test_zero_seq_len_raises(self) -> None:
        """Zero seq_len raises ValueError."""
        with pytest.raises(ValueError, match="seq_len must be positive"):
            estimate_attention_flops(1, 0, 32, 128)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            estimate_attention_flops(1, 2048, 0, 128)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head_dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            estimate_attention_flops(1, 2048, 32, 0)


class TestSelectAttentionImplementation:
    """Tests for select_attention_implementation function."""

    def test_flash_attention_preferred(self) -> None:
        """Flash Attention preferred when available."""
        impl = select_attention_implementation(2048, 128, has_flash_attention=True)
        assert impl == AttentionImplementation.FLASH_ATTENTION_2

    def test_sdpa_fallback(self) -> None:
        """SDPA used as fallback."""
        impl = select_attention_implementation(2048, 128, has_flash_attention=False)
        assert impl == AttentionImplementation.SDPA

    def test_eager_fallback(self) -> None:
        """Eager used as last resort."""
        impl = select_attention_implementation(
            2048, 128, has_flash_attention=False, has_sdpa=False
        )
        assert impl == AttentionImplementation.EAGER

    def test_non_aligned_head_dim_no_flash(self) -> None:
        """Non-aligned head_dim doesn't use Flash Attention."""
        impl = select_attention_implementation(2048, 65, has_flash_attention=True)
        assert impl == AttentionImplementation.SDPA

    def test_zero_seq_len_raises(self) -> None:
        """Zero seq_len raises ValueError."""
        with pytest.raises(ValueError, match="seq_len must be positive"):
            select_attention_implementation(0, 128)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head_dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            select_attention_implementation(2048, 0)


class TestCalculateKvCacheSize:
    """Tests for calculate_kv_cache_size function."""

    def test_basic_calculation(self) -> None:
        """Basic KV cache calculation."""
        size = calculate_kv_cache_size(1, 4096, 32, 8, 128)
        assert size > 0
        assert isinstance(size, float)

    def test_larger_batch_more_cache(self) -> None:
        """Larger batch uses more cache."""
        size1 = calculate_kv_cache_size(1, 4096, 32, 8, 128)
        size2 = calculate_kv_cache_size(4, 4096, 32, 8, 128)
        assert size2 > size1

    def test_more_layers_more_cache(self) -> None:
        """More layers use more cache."""
        size1 = calculate_kv_cache_size(1, 4096, 16, 8, 128)
        size2 = calculate_kv_cache_size(1, 4096, 32, 8, 128)
        assert size2 > size1

    def test_zero_batch_raises(self) -> None:
        """Zero batch raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_kv_cache_size(0, 4096, 32, 8, 128)

    def test_zero_seq_len_raises(self) -> None:
        """Zero max_seq_len raises ValueError."""
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            calculate_kv_cache_size(1, 0, 32, 8, 128)

    def test_zero_layers_raises(self) -> None:
        """Zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            calculate_kv_cache_size(1, 4096, 0, 8, 128)

    def test_zero_kv_heads_raises(self) -> None:
        """Zero num_kv_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_kv_heads must be positive"):
            calculate_kv_cache_size(1, 4096, 32, 0, 128)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head_dim raises ValueError."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            calculate_kv_cache_size(1, 4096, 32, 8, 0)

    def test_zero_dtype_bytes_raises(self) -> None:
        """Zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            calculate_kv_cache_size(1, 4096, 32, 8, 128, dtype_bytes=0)


class TestFormatAttentionStats:
    """Tests for format_attention_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = AttentionStats(1024.0, 1e12, 10000.0)
        formatted = format_attention_stats(stats)
        assert "Memory:" in formatted
        assert "FLOPs:" in formatted
        assert "Throughput:" in formatted

    def test_memory_value(self) -> None:
        """Memory value is formatted."""
        stats = AttentionStats(1024.0, 1e12, 10000.0)
        formatted = format_attention_stats(stats)
        assert "1024.0 MB" in formatted

    def test_tflops_format(self) -> None:
        """TFLOPs are formatted correctly."""
        stats = AttentionStats(1024.0, 1e12, 10000.0)
        formatted = format_attention_stats(stats)
        assert "TFLOPs" in formatted

    def test_pflops_format(self) -> None:
        """PFLOPs are formatted correctly."""
        stats = AttentionStats(1024.0, 1e15, 10000.0)
        formatted = format_attention_stats(stats)
        assert "PFLOPs" in formatted

    def test_gflops_format(self) -> None:
        """GFLOPs are formatted correctly."""
        stats = AttentionStats(1024.0, 1e9, 10000.0)
        formatted = format_attention_stats(stats)
        assert "GFLOPs" in formatted

    def test_mflops_format(self) -> None:
        """MFLOPs are formatted correctly."""
        stats = AttentionStats(1024.0, 1e6, 10000.0)
        formatted = format_attention_stats(stats)
        assert "MFLOPs" in formatted

    def test_small_flops_format(self) -> None:
        """Small FLOPs are formatted correctly."""
        stats = AttentionStats(1024.0, 1000.0, 10000.0)
        formatted = format_attention_stats(stats)
        assert "FLOPs" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_attention_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedAttentionConfig:
    """Tests for get_recommended_attention_config function."""

    def test_7b_config(self) -> None:
        """Get 7B config."""
        config = get_recommended_attention_config("7b")
        assert config.num_heads == 32
        assert config.head_dim == 128

    def test_13b_config(self) -> None:
        """Get 13B config."""
        config = get_recommended_attention_config("13b")
        assert config.num_heads == 40
        assert config.head_dim == 128

    def test_70b_config(self) -> None:
        """Get 70B config."""
        config = get_recommended_attention_config("70b")
        assert config.num_heads == 64
        assert config.head_dim == 128

    def test_flash_attention_default(self) -> None:
        """Flash attention is default."""
        config = get_recommended_attention_config("7b")
        assert config.attention_type == AttentionType.FLASH

    def test_standard_attention(self) -> None:
        """Standard attention when flash disabled."""
        config = get_recommended_attention_config("7b", use_flash=False)
        assert config.attention_type == AttentionType.STANDARD

    def test_case_insensitive(self) -> None:
        """Model size is case insensitive."""
        config = get_recommended_attention_config("7B")
        assert config.num_heads == 32

    def test_invalid_model_size_raises(self) -> None:
        """Invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized model size"):
            get_recommended_attention_config("invalid")


class TestCreateAttentionStats:
    """Tests for create_attention_stats function."""

    def test_basic_creation(self) -> None:
        """Create basic stats."""
        stats = create_attention_stats(512.0, 1e12, 5000.0)
        assert stats.memory_usage_mb == 512.0
        assert stats.flops == 1e12
        assert stats.throughput == 5000.0

    def test_zero_values(self) -> None:
        """Zero values are valid."""
        stats = create_attention_stats(0.0, 0.0, 0.0)
        assert stats.memory_usage_mb == 0.0
        assert stats.flops == 0.0
        assert stats.throughput == 0.0

    def test_negative_memory_raises(self) -> None:
        """Negative memory raises ValueError."""
        with pytest.raises(ValueError, match="memory_usage_mb must be non-negative"):
            create_attention_stats(-1.0, 1e12, 5000.0)

    def test_negative_flops_raises(self) -> None:
        """Negative FLOPs raises ValueError."""
        with pytest.raises(ValueError, match="flops must be non-negative"):
            create_attention_stats(512.0, -1.0, 5000.0)

    def test_negative_throughput_raises(self) -> None:
        """Negative throughput raises ValueError."""
        with pytest.raises(ValueError, match="throughput must be non-negative"):
            create_attention_stats(512.0, 1e12, -1.0)
