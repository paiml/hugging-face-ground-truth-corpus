"""Tests for models.architectures module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.models.architectures import (
    VALID_ARCHITECTURE_TYPES,
    VALID_ATTENTION_PATTERNS,
    VALID_MODEL_FAMILIES,
    ArchitectureStats,
    ArchitectureType,
    AttentionPattern,
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderConfig,
    ModelFamily,
    TransformerConfig,
    calculate_model_params,
    compare_architectures,
    create_decoder_config,
    create_encoder_config,
    create_encoder_decoder_config,
    create_transformer_config,
    estimate_memory_footprint,
    format_architecture_stats,
    get_architecture_type,
    get_attention_pattern,
    get_hidden_states_shape,
    get_model_family,
    get_recommended_architecture_config,
    list_architecture_types,
    list_attention_patterns,
    list_model_families,
    validate_architecture_stats,
    validate_decoder_config,
    validate_encoder_config,
    validate_encoder_decoder_config,
    validate_transformer_config,
)


class TestArchitectureType:
    """Tests for ArchitectureType enum."""

    def test_encoder_only_value(self) -> None:
        """Test ENCODER_ONLY enum value."""
        assert ArchitectureType.ENCODER_ONLY.value == "encoder_only"

    def test_decoder_only_value(self) -> None:
        """Test DECODER_ONLY enum value."""
        assert ArchitectureType.DECODER_ONLY.value == "decoder_only"

    def test_encoder_decoder_value(self) -> None:
        """Test ENCODER_DECODER enum value."""
        assert ArchitectureType.ENCODER_DECODER.value == "encoder_decoder"

    def test_prefix_lm_value(self) -> None:
        """Test PREFIX_LM enum value."""
        assert ArchitectureType.PREFIX_LM.value == "prefix_lm"

    def test_valid_architecture_types_frozenset(self) -> None:
        """Test VALID_ARCHITECTURE_TYPES is a frozenset."""
        assert isinstance(VALID_ARCHITECTURE_TYPES, frozenset)

    def test_valid_architecture_types_contains_all(self) -> None:
        """Test VALID_ARCHITECTURE_TYPES contains all enum values."""
        for t in ArchitectureType:
            assert t.value in VALID_ARCHITECTURE_TYPES


class TestModelFamily:
    """Tests for ModelFamily enum."""

    def test_bert_value(self) -> None:
        """Test BERT enum value."""
        assert ModelFamily.BERT.value == "bert"

    def test_gpt_value(self) -> None:
        """Test GPT enum value."""
        assert ModelFamily.GPT.value == "gpt"

    def test_t5_value(self) -> None:
        """Test T5 enum value."""
        assert ModelFamily.T5.value == "t5"

    def test_llama_value(self) -> None:
        """Test LLAMA enum value."""
        assert ModelFamily.LLAMA.value == "llama"

    def test_mistral_value(self) -> None:
        """Test MISTRAL enum value."""
        assert ModelFamily.MISTRAL.value == "mistral"

    def test_falcon_value(self) -> None:
        """Test FALCON enum value."""
        assert ModelFamily.FALCON.value == "falcon"

    def test_valid_model_families_frozenset(self) -> None:
        """Test VALID_MODEL_FAMILIES is a frozenset."""
        assert isinstance(VALID_MODEL_FAMILIES, frozenset)

    def test_valid_model_families_contains_all(self) -> None:
        """Test VALID_MODEL_FAMILIES contains all enum values."""
        for f in ModelFamily:
            assert f.value in VALID_MODEL_FAMILIES


class TestAttentionPattern:
    """Tests for AttentionPattern enum."""

    def test_dense_value(self) -> None:
        """Test DENSE enum value."""
        assert AttentionPattern.DENSE.value == "dense"

    def test_sparse_value(self) -> None:
        """Test SPARSE enum value."""
        assert AttentionPattern.SPARSE.value == "sparse"

    def test_local_value(self) -> None:
        """Test LOCAL enum value."""
        assert AttentionPattern.LOCAL.value == "local"

    def test_sliding_window_value(self) -> None:
        """Test SLIDING_WINDOW enum value."""
        assert AttentionPattern.SLIDING_WINDOW.value == "sliding_window"

    def test_valid_attention_patterns_frozenset(self) -> None:
        """Test VALID_ATTENTION_PATTERNS is a frozenset."""
        assert isinstance(VALID_ATTENTION_PATTERNS, frozenset)

    def test_valid_attention_patterns_contains_all(self) -> None:
        """Test VALID_ATTENTION_PATTERNS contains all enum values."""
        for p in AttentionPattern:
            assert p.value in VALID_ATTENTION_PATTERNS


class TestTransformerConfig:
    """Tests for TransformerConfig dataclass."""

    def test_create_config(self) -> None:
        """Test basic creation."""
        config = TransformerConfig(
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            vocab_size=30522,
        )
        assert config.num_layers == 12
        assert config.hidden_size == 768
        assert config.num_heads == 12
        assert config.intermediate_size == 3072
        assert config.vocab_size == 30522

    def test_config_is_frozen(self) -> None:
        """Test config is immutable."""
        config = TransformerConfig(12, 768, 12, 3072, 30522)
        with pytest.raises(AttributeError):
            config.num_layers = 24  # type: ignore[misc]


class TestEncoderConfig:
    """Tests for EncoderConfig dataclass."""

    def test_create_config(self) -> None:
        """Test basic creation."""
        base = TransformerConfig(12, 768, 12, 3072, 30522)
        config = EncoderConfig(
            transformer_config=base,
            pooler_type="cls",
            mask_token_id=103,
        )
        assert config.transformer_config == base
        assert config.pooler_type == "cls"
        assert config.mask_token_id == 103

    def test_config_is_frozen(self) -> None:
        """Test config is immutable."""
        base = TransformerConfig(12, 768, 12, 3072, 30522)
        config = EncoderConfig(base, "cls", 103)
        with pytest.raises(AttributeError):
            config.pooler_type = "mean"  # type: ignore[misc]


class TestDecoderConfig:
    """Tests for DecoderConfig dataclass."""

    def test_create_config(self) -> None:
        """Test basic creation."""
        base = TransformerConfig(32, 4096, 32, 11008, 32000)
        config = DecoderConfig(
            transformer_config=base,
            tie_word_embeddings=False,
            use_cache=True,
        )
        assert config.transformer_config == base
        assert config.tie_word_embeddings is False
        assert config.use_cache is True

    def test_config_is_frozen(self) -> None:
        """Test config is immutable."""
        base = TransformerConfig(32, 4096, 32, 11008, 32000)
        config = DecoderConfig(base, False, True)
        with pytest.raises(AttributeError):
            config.use_cache = False  # type: ignore[misc]


class TestEncoderDecoderConfig:
    """Tests for EncoderDecoderConfig dataclass."""

    def test_create_config(self) -> None:
        """Test basic creation."""
        enc_base = TransformerConfig(12, 768, 12, 3072, 32128)
        dec_base = TransformerConfig(12, 768, 12, 3072, 32128)
        enc = EncoderConfig(enc_base, "none", -1)
        dec = DecoderConfig(dec_base, True, True)
        config = EncoderDecoderConfig(
            encoder_config=enc,
            decoder_config=dec,
            cross_attention=True,
        )
        assert config.encoder_config == enc
        assert config.decoder_config == dec
        assert config.cross_attention is True

    def test_config_is_frozen(self) -> None:
        """Test config is immutable."""
        enc_base = TransformerConfig(12, 768, 12, 3072, 32128)
        dec_base = TransformerConfig(12, 768, 12, 3072, 32128)
        enc = EncoderConfig(enc_base, "none", -1)
        dec = DecoderConfig(dec_base, True, True)
        config = EncoderDecoderConfig(enc, dec, True)
        with pytest.raises(AttributeError):
            config.cross_attention = False  # type: ignore[misc]


class TestArchitectureStats:
    """Tests for ArchitectureStats dataclass."""

    def test_create_stats(self) -> None:
        """Test basic creation."""
        stats = ArchitectureStats(
            total_params=110_000_000,
            trainable_params=110_000_000,
            memory_footprint_mb=420.0,
        )
        assert stats.total_params == 110_000_000
        assert stats.trainable_params == 110_000_000
        assert stats.memory_footprint_mb == 420.0

    def test_stats_is_frozen(self) -> None:
        """Test stats is immutable."""
        stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        with pytest.raises(AttributeError):
            stats.total_params = 200_000_000  # type: ignore[misc]


class TestValidateTransformerConfig:
    """Tests for validate_transformer_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes validation."""
        config = TransformerConfig(12, 768, 12, 3072, 30522)
        validate_transformer_config(config)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_transformer_config(None)  # type: ignore[arg-type]

    def test_zero_num_layers_raises(self) -> None:
        """Test zero num_layers raises ValueError."""
        config = TransformerConfig(0, 768, 12, 3072, 30522)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_transformer_config(config)

    def test_negative_num_layers_raises(self) -> None:
        """Test negative num_layers raises ValueError."""
        config = TransformerConfig(-1, 768, 12, 3072, 30522)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_transformer_config(config)

    def test_zero_hidden_size_raises(self) -> None:
        """Test zero hidden_size raises ValueError."""
        config = TransformerConfig(12, 0, 12, 3072, 30522)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_transformer_config(config)

    def test_zero_num_heads_raises(self) -> None:
        """Test zero num_heads raises ValueError."""
        config = TransformerConfig(12, 768, 0, 3072, 30522)
        with pytest.raises(ValueError, match="num_heads must be positive"):
            validate_transformer_config(config)

    def test_hidden_size_not_divisible_raises(self) -> None:
        """Test hidden_size not divisible by num_heads raises ValueError."""
        config = TransformerConfig(12, 100, 12, 3072, 30522)
        with pytest.raises(ValueError, match="must be divisible"):
            validate_transformer_config(config)

    def test_zero_intermediate_size_raises(self) -> None:
        """Test zero intermediate_size raises ValueError."""
        config = TransformerConfig(12, 768, 12, 0, 30522)
        with pytest.raises(ValueError, match="intermediate_size must be positive"):
            validate_transformer_config(config)

    def test_zero_vocab_size_raises(self) -> None:
        """Test zero vocab_size raises ValueError."""
        config = TransformerConfig(12, 768, 12, 3072, 0)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_transformer_config(config)


class TestValidateEncoderConfig:
    """Tests for validate_encoder_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes validation."""
        base = TransformerConfig(12, 768, 12, 3072, 30522)
        config = EncoderConfig(base, "cls", 103)
        validate_encoder_config(config)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_encoder_config(None)  # type: ignore[arg-type]

    def test_invalid_pooler_type_raises(self) -> None:
        """Test invalid pooler_type raises ValueError."""
        base = TransformerConfig(12, 768, 12, 3072, 30522)
        config = EncoderConfig(base, "invalid", 103)
        with pytest.raises(ValueError, match="pooler_type must be one of"):
            validate_encoder_config(config)

    def test_all_valid_pooler_types(self) -> None:
        """Test all valid pooler types pass validation."""
        base = TransformerConfig(12, 768, 12, 3072, 30522)
        for pooler in ["cls", "mean", "max", "none"]:
            config = EncoderConfig(base, pooler, 103)
            validate_encoder_config(config)


class TestValidateDecoderConfig:
    """Tests for validate_decoder_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes validation."""
        base = TransformerConfig(32, 4096, 32, 11008, 32000)
        config = DecoderConfig(base, False, True)
        validate_decoder_config(config)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_decoder_config(None)  # type: ignore[arg-type]


class TestValidateEncoderDecoderConfig:
    """Tests for validate_encoder_decoder_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes validation."""
        enc_base = TransformerConfig(12, 768, 12, 3072, 32128)
        dec_base = TransformerConfig(12, 768, 12, 3072, 32128)
        enc = EncoderConfig(enc_base, "none", -1)
        dec = DecoderConfig(dec_base, True, True)
        config = EncoderDecoderConfig(enc, dec, True)
        validate_encoder_decoder_config(config)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_encoder_decoder_config(None)  # type: ignore[arg-type]


class TestValidateArchitectureStats:
    """Tests for validate_architecture_stats function."""

    def test_valid_stats(self) -> None:
        """Test valid stats pass validation."""
        stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        validate_architecture_stats(stats)

    def test_none_stats_raises(self) -> None:
        """Test None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_architecture_stats(None)  # type: ignore[arg-type]

    def test_negative_total_params_raises(self) -> None:
        """Test negative total_params raises ValueError."""
        stats = ArchitectureStats(-1, 0, 0.0)
        with pytest.raises(ValueError, match="total_params must be non-negative"):
            validate_architecture_stats(stats)

    def test_negative_trainable_params_raises(self) -> None:
        """Test negative trainable_params raises ValueError."""
        stats = ArchitectureStats(100, -1, 0.0)
        with pytest.raises(ValueError, match="trainable_params must be non-negative"):
            validate_architecture_stats(stats)

    def test_trainable_exceeds_total_raises(self) -> None:
        """Test trainable_params > total_params raises ValueError."""
        stats = ArchitectureStats(100, 200, 0.0)
        with pytest.raises(ValueError, match="cannot exceed"):
            validate_architecture_stats(stats)

    def test_negative_memory_raises(self) -> None:
        """Test negative memory_footprint_mb raises ValueError."""
        stats = ArchitectureStats(100, 100, -1.0)
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_architecture_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Test zero values are valid."""
        stats = ArchitectureStats(0, 0, 0.0)
        validate_architecture_stats(stats)


class TestCreateTransformerConfig:
    """Tests for create_transformer_config function."""

    def test_default_config(self) -> None:
        """Test default config creation."""
        config = create_transformer_config()
        assert config.num_layers == 12
        assert config.hidden_size == 768
        assert config.num_heads == 12
        assert config.intermediate_size == 3072
        assert config.vocab_size == 30522

    def test_custom_config(self) -> None:
        """Test custom config creation."""
        config = create_transformer_config(
            num_layers=24,
            hidden_size=1024,
            num_heads=16,
            intermediate_size=4096,
            vocab_size=50000,
        )
        assert config.num_layers == 24
        assert config.hidden_size == 1024
        assert config.num_heads == 16
        assert config.intermediate_size == 4096
        assert config.vocab_size == 50000

    def test_default_intermediate_size(self) -> None:
        """Test default intermediate_size is 4x hidden_size."""
        config = create_transformer_config(hidden_size=512, num_heads=8)
        assert config.intermediate_size == 2048

    def test_invalid_num_layers_raises(self) -> None:
        """Test invalid num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            create_transformer_config(num_layers=0)


class TestCreateEncoderConfig:
    """Tests for create_encoder_config function."""

    def test_default_config(self) -> None:
        """Test default config creation."""
        config = create_encoder_config()
        assert config.pooler_type == "cls"
        assert config.mask_token_id == 103
        assert config.transformer_config.num_layers == 12

    def test_custom_config(self) -> None:
        """Test custom config creation."""
        base = create_transformer_config(num_layers=6)
        config = create_encoder_config(
            transformer_config=base,
            pooler_type="mean",
            mask_token_id=200,
        )
        assert config.pooler_type == "mean"
        assert config.mask_token_id == 200
        assert config.transformer_config.num_layers == 6

    def test_invalid_pooler_raises(self) -> None:
        """Test invalid pooler_type raises ValueError."""
        with pytest.raises(ValueError, match="pooler_type must be one of"):
            create_encoder_config(pooler_type="invalid")


class TestCreateDecoderConfig:
    """Tests for create_decoder_config function."""

    def test_default_config(self) -> None:
        """Test default config creation."""
        config = create_decoder_config()
        assert config.tie_word_embeddings is False
        assert config.use_cache is True
        assert config.transformer_config.num_layers == 32

    def test_custom_config(self) -> None:
        """Test custom config creation."""
        base = create_transformer_config(num_layers=16)
        config = create_decoder_config(
            transformer_config=base,
            tie_word_embeddings=True,
            use_cache=False,
        )
        assert config.tie_word_embeddings is True
        assert config.use_cache is False
        assert config.transformer_config.num_layers == 16


class TestCreateEncoderDecoderConfig:
    """Tests for create_encoder_decoder_config function."""

    def test_default_config(self) -> None:
        """Test default config creation."""
        config = create_encoder_decoder_config()
        assert config.cross_attention is True
        assert config.encoder_config.pooler_type == "none"
        assert config.decoder_config.tie_word_embeddings is True

    def test_custom_config(self) -> None:
        """Test custom config creation."""
        enc = create_encoder_config(pooler_type="mean")
        dec = create_decoder_config(tie_word_embeddings=False)
        config = create_encoder_decoder_config(
            encoder_config=enc,
            decoder_config=dec,
            cross_attention=False,
        )
        assert config.cross_attention is False
        assert config.encoder_config.pooler_type == "mean"
        assert config.decoder_config.tie_word_embeddings is False


class TestListArchitectureTypes:
    """Tests for list_architecture_types function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        types = list_architecture_types()
        assert types == sorted(types)

    def test_contains_encoder_only(self) -> None:
        """Test contains encoder_only."""
        types = list_architecture_types()
        assert "encoder_only" in types

    def test_contains_decoder_only(self) -> None:
        """Test contains decoder_only."""
        types = list_architecture_types()
        assert "decoder_only" in types

    def test_contains_all_types(self) -> None:
        """Test contains all types."""
        types = list_architecture_types()
        for t in ArchitectureType:
            assert t.value in types


class TestListModelFamilies:
    """Tests for list_model_families function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        families = list_model_families()
        assert families == sorted(families)

    def test_contains_bert(self) -> None:
        """Test contains bert."""
        families = list_model_families()
        assert "bert" in families

    def test_contains_llama(self) -> None:
        """Test contains llama."""
        families = list_model_families()
        assert "llama" in families

    def test_contains_all_families(self) -> None:
        """Test contains all families."""
        families = list_model_families()
        for f in ModelFamily:
            assert f.value in families


class TestListAttentionPatterns:
    """Tests for list_attention_patterns function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        patterns = list_attention_patterns()
        assert patterns == sorted(patterns)

    def test_contains_dense(self) -> None:
        """Test contains dense."""
        patterns = list_attention_patterns()
        assert "dense" in patterns

    def test_contains_sliding_window(self) -> None:
        """Test contains sliding_window."""
        patterns = list_attention_patterns()
        assert "sliding_window" in patterns

    def test_contains_all_patterns(self) -> None:
        """Test contains all patterns."""
        patterns = list_attention_patterns()
        for p in AttentionPattern:
            assert p.value in patterns


class TestGetArchitectureType:
    """Tests for get_architecture_type function."""

    def test_get_encoder_only(self) -> None:
        """Test get encoder_only type."""
        assert get_architecture_type("encoder_only") == ArchitectureType.ENCODER_ONLY

    def test_get_decoder_only(self) -> None:
        """Test get decoder_only type."""
        assert get_architecture_type("decoder_only") == ArchitectureType.DECODER_ONLY

    def test_get_encoder_decoder(self) -> None:
        """Test get encoder_decoder type."""
        assert (
            get_architecture_type("encoder_decoder") == ArchitectureType.ENCODER_DECODER
        )

    def test_get_prefix_lm(self) -> None:
        """Test get prefix_lm type."""
        assert get_architecture_type("prefix_lm") == ArchitectureType.PREFIX_LM

    def test_invalid_type_raises(self) -> None:
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="architecture type must be one of"):
            get_architecture_type("invalid")


class TestGetModelFamily:
    """Tests for get_model_family function."""

    def test_get_bert(self) -> None:
        """Test get bert family."""
        assert get_model_family("bert") == ModelFamily.BERT

    def test_get_llama(self) -> None:
        """Test get llama family."""
        assert get_model_family("llama") == ModelFamily.LLAMA

    def test_get_mistral(self) -> None:
        """Test get mistral family."""
        assert get_model_family("mistral") == ModelFamily.MISTRAL

    def test_invalid_family_raises(self) -> None:
        """Test invalid family raises ValueError."""
        with pytest.raises(ValueError, match="model family must be one of"):
            get_model_family("invalid")


class TestGetAttentionPattern:
    """Tests for get_attention_pattern function."""

    def test_get_dense(self) -> None:
        """Test get dense pattern."""
        assert get_attention_pattern("dense") == AttentionPattern.DENSE

    def test_get_sliding_window(self) -> None:
        """Test get sliding_window pattern."""
        result = get_attention_pattern("sliding_window")
        assert result == AttentionPattern.SLIDING_WINDOW

    def test_get_sparse(self) -> None:
        """Test get sparse pattern."""
        assert get_attention_pattern("sparse") == AttentionPattern.SPARSE

    def test_invalid_pattern_raises(self) -> None:
        """Test invalid pattern raises ValueError."""
        with pytest.raises(ValueError, match="attention pattern must be one of"):
            get_attention_pattern("invalid")


class TestCalculateModelParams:
    """Tests for calculate_model_params function."""

    def test_basic_calculation(self) -> None:
        """Test basic parameter calculation."""
        config = create_transformer_config(num_layers=12, hidden_size=768, num_heads=12)
        params = calculate_model_params(config)
        assert params > 100_000_000  # BERT-base ~110M
        assert isinstance(params, int)

    def test_larger_model_more_params(self) -> None:
        """Test larger model has more parameters."""
        small = create_transformer_config(num_layers=6, hidden_size=512, num_heads=8)
        large = create_transformer_config(num_layers=12, hidden_size=768, num_heads=12)
        assert calculate_model_params(large) > calculate_model_params(small)

    def test_more_layers_more_params(self) -> None:
        """Test more layers means more parameters."""
        shallow = create_transformer_config(num_layers=6, hidden_size=768)
        deep = create_transformer_config(num_layers=12, hidden_size=768)
        assert calculate_model_params(deep) > calculate_model_params(shallow)

    def test_larger_vocab_more_params(self) -> None:
        """Test larger vocab means more parameters."""
        small_vocab = create_transformer_config(vocab_size=30000)
        large_vocab = create_transformer_config(vocab_size=50000)
        assert calculate_model_params(large_vocab) > calculate_model_params(small_vocab)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            calculate_model_params(None)  # type: ignore[arg-type]


class TestEstimateMemoryFootprint:
    """Tests for estimate_memory_footprint function."""

    def test_basic_estimation(self) -> None:
        """Test basic memory estimation."""
        config = create_transformer_config()
        mem = estimate_memory_footprint(config)
        assert mem > 0
        assert isinstance(mem, float)

    def test_fp32_more_memory_than_fp16(self) -> None:
        """Test fp32 uses more memory than fp16."""
        config = create_transformer_config()
        mem_fp16 = estimate_memory_footprint(config, dtype_bytes=2)
        mem_fp32 = estimate_memory_footprint(config, dtype_bytes=4)
        assert mem_fp32 > mem_fp16

    def test_no_gradients_less_memory(self) -> None:
        """Test excluding gradients uses less memory."""
        config = create_transformer_config()
        mem_with = estimate_memory_footprint(config, include_gradients=True)
        mem_without = estimate_memory_footprint(config, include_gradients=False)
        assert mem_without < mem_with

    def test_no_optimizer_less_memory(self) -> None:
        """Test excluding optimizer uses less memory."""
        config = create_transformer_config()
        mem_with = estimate_memory_footprint(config, include_optimizer=True)
        mem_without = estimate_memory_footprint(config, include_optimizer=False)
        assert mem_without < mem_with

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            estimate_memory_footprint(None)  # type: ignore[arg-type]

    def test_invalid_dtype_bytes_raises(self) -> None:
        """Test invalid dtype_bytes raises ValueError."""
        config = create_transformer_config()
        with pytest.raises(ValueError, match="dtype_bytes must be"):
            estimate_memory_footprint(config, dtype_bytes=3)


class TestCompareArchitectures:
    """Tests for compare_architectures function."""

    def test_basic_comparison(self) -> None:
        """Test basic architecture comparison."""
        small = create_transformer_config(num_layers=6, hidden_size=512, num_heads=8)
        large = create_transformer_config(num_layers=12, hidden_size=768, num_heads=12)
        results = compare_architectures([("small", small), ("large", large)])
        assert "small" in results
        assert "large" in results
        assert results["large"].total_params > results["small"].total_params

    def test_single_config(self) -> None:
        """Test single config comparison."""
        config = create_transformer_config()
        results = compare_architectures([("model", config)])
        assert "model" in results
        assert results["model"].total_params > 0

    def test_empty_list_raises(self) -> None:
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="configs cannot be empty"):
            compare_architectures([])

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compare_architectures([("model", None)])  # type: ignore[list-item]


class TestGetHiddenStatesShape:
    """Tests for get_hidden_states_shape function."""

    def test_basic_shape(self) -> None:
        """Test basic shape calculation."""
        config = create_transformer_config(hidden_size=768)
        shape = get_hidden_states_shape(config, batch_size=4, seq_length=512)
        assert shape == (4, 512, 768)

    def test_different_dimensions(self) -> None:
        """Test different dimensions."""
        config = create_transformer_config(hidden_size=1024, num_heads=16)
        shape = get_hidden_states_shape(config, batch_size=8, seq_length=256)
        assert shape == (8, 256, 1024)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_hidden_states_shape(None, 1, 512)  # type: ignore[arg-type]

    def test_zero_batch_size_raises(self) -> None:
        """Test zero batch_size raises ValueError."""
        config = create_transformer_config()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            get_hidden_states_shape(config, batch_size=0, seq_length=512)

    def test_negative_batch_size_raises(self) -> None:
        """Test negative batch_size raises ValueError."""
        config = create_transformer_config()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            get_hidden_states_shape(config, batch_size=-1, seq_length=512)

    def test_zero_seq_length_raises(self) -> None:
        """Test zero seq_length raises ValueError."""
        config = create_transformer_config()
        with pytest.raises(ValueError, match="seq_length must be positive"):
            get_hidden_states_shape(config, batch_size=1, seq_length=0)

    def test_negative_seq_length_raises(self) -> None:
        """Test negative seq_length raises ValueError."""
        config = create_transformer_config()
        with pytest.raises(ValueError, match="seq_length must be positive"):
            get_hidden_states_shape(config, batch_size=1, seq_length=-1)


class TestFormatArchitectureStats:
    """Tests for format_architecture_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        formatted = format_architecture_stats(stats)
        assert "Total Parameters:" in formatted
        assert "Trainable Parameters:" in formatted
        assert "Memory Footprint:" in formatted

    def test_millions_format(self) -> None:
        """Test millions formatting."""
        stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        formatted = format_architecture_stats(stats)
        assert "M" in formatted

    def test_billions_format(self) -> None:
        """Test billions formatting."""
        stats = ArchitectureStats(7_000_000_000, 7_000_000_000, 14000.0)
        formatted = format_architecture_stats(stats)
        assert "B" in formatted

    def test_trillions_format(self) -> None:
        """Test trillions formatting."""
        stats = ArchitectureStats(1_000_000_000_000, 1_000_000_000_000, 2000000.0)
        formatted = format_architecture_stats(stats)
        assert "T" in formatted

    def test_gb_format(self) -> None:
        """Test GB formatting."""
        stats = ArchitectureStats(7_000_000_000, 7_000_000_000, 14000.0)
        formatted = format_architecture_stats(stats)
        assert "GB" in formatted

    def test_mb_format(self) -> None:
        """Test MB formatting."""
        stats = ArchitectureStats(110_000_000, 110_000_000, 420.0)
        formatted = format_architecture_stats(stats)
        assert "MB" in formatted

    def test_none_stats_raises(self) -> None:
        """Test None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_architecture_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedArchitectureConfig:
    """Tests for get_recommended_architecture_config function."""

    def test_7b_config(self) -> None:
        """Test 7B config."""
        config = get_recommended_architecture_config("7b")
        assert config.num_layers == 32
        assert config.hidden_size == 4096
        assert config.num_heads == 32

    def test_13b_config(self) -> None:
        """Test 13B config."""
        config = get_recommended_architecture_config("13b")
        assert config.num_layers == 40
        assert config.hidden_size == 5120

    def test_70b_config(self) -> None:
        """Test 70B config."""
        config = get_recommended_architecture_config("70b")
        assert config.num_layers == 80
        assert config.hidden_size == 8192

    def test_case_insensitive(self) -> None:
        """Test model size is case insensitive."""
        config = get_recommended_architecture_config("7B")
        assert config.num_layers == 32

    def test_encoder_only_base(self) -> None:
        """Test encoder_only base config."""
        config = get_recommended_architecture_config("base", "encoder_only")
        assert config.num_layers == 12
        assert config.hidden_size == 768

    def test_encoder_only_large(self) -> None:
        """Test encoder_only large config."""
        config = get_recommended_architecture_config("large", "encoder_only")
        assert config.num_layers == 24
        assert config.hidden_size == 1024

    def test_invalid_model_size_raises(self) -> None:
        """Test invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized model size"):
            get_recommended_architecture_config("invalid")

    def test_invalid_architecture_type_raises(self) -> None:
        """Test invalid architecture type raises ValueError."""
        with pytest.raises(ValueError, match="architecture_type must be one of"):
            get_recommended_architecture_config("7b", "invalid")


class TestPropertyBasedTests:
    """Property-based tests using hypothesis."""

    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=64).map(lambda x: x * 64),
        st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=20)
    def test_params_always_positive(
        self, num_layers: int, hidden_size: int, num_heads: int
    ) -> None:
        """Test parameter count is always positive."""
        if hidden_size % num_heads != 0:
            return  # Skip invalid configs
        config = create_transformer_config(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        params = calculate_model_params(config)
        assert params > 0

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=4096),
        st.integers(min_value=1, max_value=64).map(lambda x: x * 64),
    )
    @settings(max_examples=20)
    def test_hidden_states_shape_valid(
        self, batch_size: int, seq_length: int, hidden_size: int
    ) -> None:
        """Test hidden states shape is always valid."""
        config = create_transformer_config(hidden_size=hidden_size, num_heads=8)
        shape = get_hidden_states_shape(config, batch_size, seq_length)
        assert shape[0] == batch_size
        assert shape[1] == seq_length
        assert shape[2] == hidden_size

    @given(
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=64).map(lambda x: x * 64),
    )
    @settings(max_examples=20)
    def test_memory_footprint_positive(self, num_layers: int, hidden_size: int) -> None:
        """Test memory footprint is always positive."""
        config = create_transformer_config(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=8,
        )
        mem = estimate_memory_footprint(config)
        assert mem > 0

    @given(st.sampled_from(list(VALID_ARCHITECTURE_TYPES)))
    @settings(max_examples=10)
    def test_all_architecture_types_gettable(self, arch_type: str) -> None:
        """Test all valid architecture types can be gotten."""
        result = get_architecture_type(arch_type)
        assert result.value == arch_type

    @given(st.sampled_from(list(VALID_MODEL_FAMILIES)))
    @settings(max_examples=10)
    def test_all_model_families_gettable(self, family: str) -> None:
        """Test all valid model families can be gotten."""
        result = get_model_family(family)
        assert result.value == family

    @given(st.sampled_from(list(VALID_ATTENTION_PATTERNS)))
    @settings(max_examples=10)
    def test_all_attention_patterns_gettable(self, pattern: str) -> None:
        """Test all valid attention patterns can be gotten."""
        result = get_attention_pattern(pattern)
        assert result.value == pattern
