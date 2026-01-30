"""Tests for inference.optimization module."""

from __future__ import annotations

import pytest

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


class TestAttentionImplementation:
    """Tests for AttentionImplementation enum."""

    def test_all_impls_have_values(self) -> None:
        """All implementations have string values."""
        for impl in AttentionImplementation:
            assert isinstance(impl.value, str)

    def test_flash_attention_2_value(self) -> None:
        """Flash Attention 2 has correct value."""
        assert AttentionImplementation.FLASH_ATTENTION_2.value == "flash_attention_2"

    def test_sdpa_value(self) -> None:
        """SDPA has correct value."""
        assert AttentionImplementation.SDPA.value == "sdpa"

    def test_valid_impls_frozenset(self) -> None:
        """VALID_ATTENTION_IMPLS is a frozenset."""
        assert isinstance(VALID_ATTENTION_IMPLS, frozenset)


class TestKVCacheType:
    """Tests for KVCacheType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for cache_type in KVCacheType:
            assert isinstance(cache_type.value, str)

    def test_dynamic_value(self) -> None:
        """Dynamic has correct value."""
        assert KVCacheType.DYNAMIC.value == "dynamic"

    def test_valid_types_frozenset(self) -> None:
        """VALID_CACHE_TYPES is a frozenset."""
        assert isinstance(VALID_CACHE_TYPES, frozenset)


class TestKVCacheConfig:
    """Tests for KVCacheConfig dataclass."""

    def test_create_config(self) -> None:
        """Create KV cache config."""
        config = KVCacheConfig(
            cache_type=KVCacheType.DYNAMIC,
            max_batch_size=32,
            max_seq_length=2048,
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        assert config.max_batch_size == 32

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 32, 2048, 32, 32, 128)
        with pytest.raises(AttributeError):
            config.max_batch_size = 64  # type: ignore[misc]


class TestValidateKVCacheConfig:
    """Tests for validate_kv_cache_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 32, 2048, 32, 32, 128)
        validate_kv_cache_config(config)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 0, 2048, 32, 32, 128)
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            validate_kv_cache_config(config)

    def test_zero_seq_length_raises(self) -> None:
        """Zero seq length raises ValueError."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 32, 0, 32, 32, 128)
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_kv_cache_config(config)

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 32, 2048, 0, 32, 128)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_kv_cache_config(config)

    def test_zero_heads_raises(self) -> None:
        """Zero heads raises ValueError."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 32, 2048, 32, 0, 128)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_kv_cache_config(config)

    def test_zero_head_dim_raises(self) -> None:
        """Zero head dim raises ValueError."""
        config = KVCacheConfig(KVCacheType.DYNAMIC, 32, 2048, 32, 32, 0)
        with pytest.raises(ValueError, match="head_dim must be positive"):
            validate_kv_cache_config(config)


class TestSpeculativeDecodingConfig:
    """Tests for SpeculativeDecodingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create speculative decoding config."""
        config = SpeculativeDecodingConfig(
            draft_model_id="gpt2",
            num_speculative_tokens=5,
            acceptance_threshold=0.9,
            use_assistant_model=True,
        )
        assert config.num_speculative_tokens == 5


class TestValidateSpeculativeConfig:
    """Tests for validate_speculative_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SpeculativeDecodingConfig("gpt2", 5, 0.9, True)
        validate_speculative_config(config)

    def test_empty_model_id_raises(self) -> None:
        """Empty model ID raises ValueError."""
        config = SpeculativeDecodingConfig("", 5, 0.9, True)
        with pytest.raises(ValueError, match="draft_model_id cannot be empty"):
            validate_speculative_config(config)

    def test_zero_tokens_raises(self) -> None:
        """Zero tokens raises ValueError."""
        config = SpeculativeDecodingConfig("gpt2", 0, 0.9, True)
        with pytest.raises(ValueError, match="num_speculative_tokens must be positive"):
            validate_speculative_config(config)

    def test_invalid_threshold_raises(self) -> None:
        """Invalid threshold raises ValueError."""
        config = SpeculativeDecodingConfig("gpt2", 5, 1.5, True)
        with pytest.raises(ValueError, match="acceptance_threshold must be between"):
            validate_speculative_config(config)


class TestCreateKVCacheConfig:
    """Tests for create_kv_cache_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_kv_cache_config()
        assert config.cache_type == KVCacheType.DYNAMIC
        assert config.max_batch_size == 32

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_kv_cache_config(
            cache_type="static",
            max_batch_size=64,
        )
        assert config.cache_type == KVCacheType.STATIC
        assert config.max_batch_size == 64

    def test_invalid_cache_type_raises(self) -> None:
        """Invalid cache type raises ValueError."""
        with pytest.raises(ValueError, match="cache_type must be one of"):
            create_kv_cache_config(cache_type="invalid")

    def test_invalid_batch_size_raises(self) -> None:
        """Invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            create_kv_cache_config(max_batch_size=0)


class TestCreateFlashAttentionConfig:
    """Tests for create_flash_attention_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_flash_attention_config()
        assert config.implementation == AttentionImplementation.FLASH_ATTENTION_2

    def test_with_sliding_window(self) -> None:
        """Create config with sliding window."""
        config = create_flash_attention_config(sliding_window=4096)
        assert config.sliding_window == 4096

    def test_invalid_impl_raises(self) -> None:
        """Invalid implementation raises ValueError."""
        with pytest.raises(ValueError, match="implementation must be one of"):
            create_flash_attention_config(implementation="invalid")

    def test_invalid_sliding_window_raises(self) -> None:
        """Invalid sliding window raises ValueError."""
        with pytest.raises(ValueError, match="sliding_window must be positive"):
            create_flash_attention_config(sliding_window=0)


class TestCreateSpeculativeDecodingConfig:
    """Tests for create_speculative_decoding_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_speculative_decoding_config("gpt2")
        assert config.draft_model_id == "gpt2"
        assert config.num_speculative_tokens == 5

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_speculative_decoding_config(
            "gpt2",
            num_speculative_tokens=8,
        )
        assert config.num_speculative_tokens == 8

    def test_empty_model_id_raises(self) -> None:
        """Empty model ID raises ValueError."""
        with pytest.raises(ValueError, match="draft_model_id cannot be empty"):
            create_speculative_decoding_config("")


class TestCreateQuantizedKVConfig:
    """Tests for create_quantized_kv_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_quantized_kv_config()
        assert config.quant_type == "fp8"
        assert config.residual_length == 128

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_quantized_kv_config(quant_type="int8")
        assert config.quant_type == "int8"

    def test_invalid_quant_type_raises(self) -> None:
        """Invalid quant type raises ValueError."""
        with pytest.raises(ValueError, match="quant_type must be one of"):
            create_quantized_kv_config(quant_type="invalid")

    def test_negative_residual_raises(self) -> None:
        """Negative residual length raises ValueError."""
        with pytest.raises(ValueError, match="residual_length must be non-negative"):
            create_quantized_kv_config(residual_length=-1)


class TestCreateContinuousBatchingConfig:
    """Tests for create_continuous_batching_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_continuous_batching_config()
        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens == 8192

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_continuous_batching_config(max_num_seqs=128)
        assert config.max_num_seqs == 128

    def test_zero_seqs_raises(self) -> None:
        """Zero seqs raises ValueError."""
        with pytest.raises(ValueError, match="max_num_seqs must be positive"):
            create_continuous_batching_config(max_num_seqs=0)

    def test_zero_tokens_raises(self) -> None:
        """Zero tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_num_batched_tokens must be positive"):
            create_continuous_batching_config(max_num_batched_tokens=0)

    def test_negative_paddings_raises(self) -> None:
        """Negative paddings raises ValueError."""
        with pytest.raises(ValueError, match="max_paddings must be non-negative"):
            create_continuous_batching_config(max_paddings=-1)

    def test_invalid_mode_raises(self) -> None:
        """Invalid preemption mode raises ValueError."""
        with pytest.raises(ValueError, match="preemption_mode must be one of"):
            create_continuous_batching_config(preemption_mode="invalid")


class TestListAttentionImplementations:
    """Tests for list_attention_implementations function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        impls = list_attention_implementations()
        assert impls == sorted(impls)

    def test_contains_flash_attention(self) -> None:
        """Contains flash attention."""
        impls = list_attention_implementations()
        assert "flash_attention_2" in impls


class TestListKVCacheTypes:
    """Tests for list_kv_cache_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_kv_cache_types()
        assert types == sorted(types)

    def test_contains_dynamic(self) -> None:
        """Contains dynamic."""
        types = list_kv_cache_types()
        assert "dynamic" in types


class TestEstimateKVCacheMemory:
    """Tests for estimate_kv_cache_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic memory estimate."""
        config = create_kv_cache_config(
            max_batch_size=1,
            max_seq_length=2048,
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        mem = estimate_kv_cache_memory(config)
        assert mem > 0

    def test_larger_batch_more_memory(self) -> None:
        """Larger batch uses more memory."""
        config1 = create_kv_cache_config(max_batch_size=1)
        config4 = create_kv_cache_config(max_batch_size=4)
        mem1 = estimate_kv_cache_memory(config1)
        mem4 = estimate_kv_cache_memory(config4)
        assert mem4 == mem1 * 4


class TestGetRecommendedAttention:
    """Tests for get_recommended_attention function."""

    def test_large_with_flash(self) -> None:
        """Large model with flash attention."""
        impl = get_recommended_attention("large", has_flash_attention=True)
        assert impl == AttentionImplementation.FLASH_ATTENTION_2

    def test_small_without_flash(self) -> None:
        """Small model without flash attention."""
        impl = get_recommended_attention("small", has_flash_attention=False)
        assert impl == AttentionImplementation.EAGER

    def test_medium_model(self) -> None:
        """Medium model uses SDPA."""
        impl = get_recommended_attention("medium", has_flash_attention=False)
        assert impl == AttentionImplementation.SDPA

    def test_invalid_size_raises(self) -> None:
        """Invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_attention("invalid")


class TestCalculateSpeculativeSpeedup:
    """Tests for calculate_speculative_speedup function."""

    def test_positive_speedup(self) -> None:
        """Positive acceptance rate gives speedup > 1."""
        speedup = calculate_speculative_speedup(0.8, 5, 0.1)
        assert speedup > 1.0

    def test_zero_acceptance_no_speedup(self) -> None:
        """Zero acceptance rate gives minimal speedup."""
        speedup = calculate_speculative_speedup(0.0, 5, 0.1)
        assert speedup < 1.0

    def test_invalid_acceptance_rate_raises(self) -> None:
        """Invalid acceptance rate raises ValueError."""
        with pytest.raises(ValueError, match="acceptance_rate must be between"):
            calculate_speculative_speedup(1.5, 5, 0.1)

    def test_invalid_tokens_raises(self) -> None:
        """Invalid tokens raises ValueError."""
        with pytest.raises(ValueError, match="num_speculative_tokens must be positive"):
            calculate_speculative_speedup(0.8, 0, 0.1)

    def test_invalid_latency_ratio_raises(self) -> None:
        """Invalid latency ratio raises ValueError."""
        with pytest.raises(ValueError, match="draft_latency_ratio must be between"):
            calculate_speculative_speedup(0.8, 5, 1.0)


class TestFlashAttentionConfig:
    """Tests for FlashAttentionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Flash Attention config."""
        config = FlashAttentionConfig(
            implementation=AttentionImplementation.FLASH_ATTENTION_2,
            sliding_window=None,
            softmax_scale=None,
            return_softmax=False,
        )
        assert config.implementation == AttentionImplementation.FLASH_ATTENTION_2


class TestQuantizedKVConfig:
    """Tests for QuantizedKVConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Quantized KV config."""
        config = QuantizedKVConfig(
            quant_type="fp8",
            residual_length=128,
            compute_dtype="float16",
        )
        assert config.quant_type == "fp8"

    def test_valid_quant_types(self) -> None:
        """VALID_KV_QUANT_TYPES contains expected values."""
        assert "fp8" in VALID_KV_QUANT_TYPES
        assert "int8" in VALID_KV_QUANT_TYPES


class TestContinuousBatchingConfig:
    """Tests for ContinuousBatchingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create continuous batching config."""
        config = ContinuousBatchingConfig(
            max_num_seqs=256,
            max_num_batched_tokens=8192,
            max_paddings=256,
            preemption_mode="swap",
        )
        assert config.max_num_seqs == 256
