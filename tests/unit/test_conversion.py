"""Tests for model format conversion utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.conversion import (
    VALID_CONVERSION_PRECISIONS,
    VALID_MODEL_FORMATS,
    VALID_SHARDING_STRATEGIES,
    ConversionConfig,
    ConversionPrecision,
    ConversionStats,
    GGUFConversionConfig,
    ModelFormat,
    SafeTensorsConversionConfig,
    ShardingStrategy,
    calculate_precision_loss,
    create_conversion_config,
    create_conversion_stats,
    create_gguf_conversion_config,
    create_safetensors_conversion_config,
    estimate_conversion_time,
    estimate_converted_size,
    format_conversion_stats,
    get_conversion_config_dict,
    get_conversion_precision,
    get_gguf_conversion_config_dict,
    get_model_format,
    get_recommended_conversion_config,
    get_safetensors_conversion_config_dict,
    get_sharding_strategy,
    list_conversion_precisions,
    list_model_formats,
    list_sharding_strategies,
    validate_conversion_config,
    validate_format_compatibility,
    validate_gguf_conversion_config,
    validate_safetensors_conversion_config,
)


class TestModelFormat:
    """Tests for ModelFormat enum."""

    def test_huggingface_value(self) -> None:
        """Test HUGGINGFACE value."""
        assert ModelFormat.HUGGINGFACE.value == "huggingface"

    def test_safetensors_value(self) -> None:
        """Test SAFETENSORS value."""
        assert ModelFormat.SAFETENSORS.value == "safetensors"

    def test_gguf_value(self) -> None:
        """Test GGUF value."""
        assert ModelFormat.GGUF.value == "gguf"

    def test_onnx_value(self) -> None:
        """Test ONNX value."""
        assert ModelFormat.ONNX.value == "onnx"

    def test_torchscript_value(self) -> None:
        """Test TORCHSCRIPT value."""
        assert ModelFormat.TORCHSCRIPT.value == "torchscript"

    def test_tflite_value(self) -> None:
        """Test TFLITE value."""
        assert ModelFormat.TFLITE.value == "tflite"


class TestConversionPrecision:
    """Tests for ConversionPrecision enum."""

    def test_fp32_value(self) -> None:
        """Test FP32 value."""
        assert ConversionPrecision.FP32.value == "fp32"

    def test_fp16_value(self) -> None:
        """Test FP16 value."""
        assert ConversionPrecision.FP16.value == "fp16"

    def test_bf16_value(self) -> None:
        """Test BF16 value."""
        assert ConversionPrecision.BF16.value == "bf16"

    def test_int8_value(self) -> None:
        """Test INT8 value."""
        assert ConversionPrecision.INT8.value == "int8"

    def test_int4_value(self) -> None:
        """Test INT4 value."""
        assert ConversionPrecision.INT4.value == "int4"


class TestShardingStrategy:
    """Tests for ShardingStrategy enum."""

    def test_none_value(self) -> None:
        """Test NONE value."""
        assert ShardingStrategy.NONE.value == "none"

    def test_layer_value(self) -> None:
        """Test LAYER value."""
        assert ShardingStrategy.LAYER.value == "layer"

    def test_tensor_value(self) -> None:
        """Test TENSOR value."""
        assert ShardingStrategy.TENSOR.value == "tensor"

    def test_hybrid_value(self) -> None:
        """Test HYBRID value."""
        assert ShardingStrategy.HYBRID.value == "hybrid"


class TestValidFrozensets:
    """Tests for VALID_* frozensets."""

    def test_valid_model_formats(self) -> None:
        """Test VALID_MODEL_FORMATS."""
        assert "huggingface" in VALID_MODEL_FORMATS
        assert "safetensors" in VALID_MODEL_FORMATS
        assert "gguf" in VALID_MODEL_FORMATS
        assert len(VALID_MODEL_FORMATS) == 6

    def test_valid_conversion_precisions(self) -> None:
        """Test VALID_CONVERSION_PRECISIONS."""
        assert "fp32" in VALID_CONVERSION_PRECISIONS
        assert "fp16" in VALID_CONVERSION_PRECISIONS
        assert "int8" in VALID_CONVERSION_PRECISIONS
        assert len(VALID_CONVERSION_PRECISIONS) == 5

    def test_valid_sharding_strategies(self) -> None:
        """Test VALID_SHARDING_STRATEGIES."""
        assert "none" in VALID_SHARDING_STRATEGIES
        assert "layer" in VALID_SHARDING_STRATEGIES
        assert len(VALID_SHARDING_STRATEGIES) == 4


class TestConversionConfig:
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ConversionConfig(
            source_format=ModelFormat.HUGGINGFACE,
            target_format=ModelFormat.SAFETENSORS,
        )
        assert config.precision == ConversionPrecision.FP16
        assert config.shard_size_gb == pytest.approx(5.0)

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ConversionConfig(
            source_format=ModelFormat.HUGGINGFACE,
            target_format=ModelFormat.GGUF,
            precision=ConversionPrecision.INT4,
            shard_size_gb=4.0,
        )
        assert config.source_format == ModelFormat.HUGGINGFACE
        assert config.target_format == ModelFormat.GGUF
        assert config.precision == ConversionPrecision.INT4
        assert config.shard_size_gb == pytest.approx(4.0)

    def test_frozen(self) -> None:
        """Test that ConversionConfig is immutable."""
        config = ConversionConfig(
            source_format=ModelFormat.HUGGINGFACE,
            target_format=ModelFormat.SAFETENSORS,
        )
        with pytest.raises(AttributeError):
            config.precision = ConversionPrecision.FP32  # type: ignore[misc]


class TestGGUFConversionConfig:
    """Tests for GGUFConversionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = GGUFConversionConfig()
        assert config.quantization_type == "q4_k_m"
        assert config.use_mmap is True
        assert config.vocab_only is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = GGUFConversionConfig(
            quantization_type="q8_0",
            use_mmap=False,
            vocab_only=True,
        )
        assert config.quantization_type == "q8_0"
        assert config.use_mmap is False
        assert config.vocab_only is True

    def test_frozen(self) -> None:
        """Test that GGUFConversionConfig is immutable."""
        config = GGUFConversionConfig()
        with pytest.raises(AttributeError):
            config.use_mmap = False  # type: ignore[misc]


class TestSafeTensorsConversionConfig:
    """Tests for SafeTensorsConversionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = SafeTensorsConversionConfig()
        assert config.metadata is None
        assert config.shard_size == 5_000_000_000
        assert config.strict is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = SafeTensorsConversionConfig(
            metadata={"format": "pt"},
            shard_size=2_000_000_000,
            strict=False,
        )
        assert config.metadata == {"format": "pt"}
        assert config.shard_size == 2_000_000_000
        assert config.strict is False

    def test_frozen(self) -> None:
        """Test that SafeTensorsConversionConfig is immutable."""
        config = SafeTensorsConversionConfig()
        with pytest.raises(AttributeError):
            config.strict = False  # type: ignore[misc]


class TestConversionStats:
    """Tests for ConversionStats dataclass."""

    def test_creation(self) -> None:
        """Test creating stats."""
        stats = ConversionStats(
            original_size_mb=14000.0,
            converted_size_mb=4000.0,
            conversion_time_seconds=120.5,
            precision_loss=0.05,
        )
        assert stats.original_size_mb == pytest.approx(14000.0)
        assert stats.converted_size_mb == pytest.approx(4000.0)
        assert stats.conversion_time_seconds == pytest.approx(120.5)
        assert stats.precision_loss == pytest.approx(0.05)

    def test_frozen(self) -> None:
        """Test that ConversionStats is immutable."""
        stats = ConversionStats(
            original_size_mb=14000.0,
            converted_size_mb=4000.0,
            conversion_time_seconds=120.5,
            precision_loss=0.05,
        )
        with pytest.raises(AttributeError):
            stats.precision_loss = 0.1  # type: ignore[misc]


class TestValidateConversionConfig:
    """Tests for validate_conversion_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = ConversionConfig(
            source_format=ModelFormat.HUGGINGFACE,
            target_format=ModelFormat.SAFETENSORS,
        )
        validate_conversion_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_conversion_config(None)  # type: ignore[arg-type]

    def test_same_formats_raises_error(self) -> None:
        """Test that same source and target formats raises ValueError."""
        config = ConversionConfig(
            source_format=ModelFormat.GGUF,
            target_format=ModelFormat.GGUF,
        )
        with pytest.raises(
            ValueError, match="source and target formats cannot be the same"
        ):
            validate_conversion_config(config)

    def test_zero_shard_size_raises_error(self) -> None:
        """Test that zero shard size raises ValueError."""
        config = ConversionConfig(
            source_format=ModelFormat.HUGGINGFACE,
            target_format=ModelFormat.SAFETENSORS,
            shard_size_gb=0.0,
        )
        with pytest.raises(ValueError, match="shard_size_gb must be positive"):
            validate_conversion_config(config)

    def test_negative_shard_size_raises_error(self) -> None:
        """Test that negative shard size raises ValueError."""
        config = ConversionConfig(
            source_format=ModelFormat.HUGGINGFACE,
            target_format=ModelFormat.SAFETENSORS,
            shard_size_gb=-1.0,
        )
        with pytest.raises(ValueError, match="shard_size_gb must be positive"):
            validate_conversion_config(config)


class TestValidateGGUFConversionConfig:
    """Tests for validate_gguf_conversion_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = GGUFConversionConfig(quantization_type="q4_k_m")
        validate_gguf_conversion_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_gguf_conversion_config(None)  # type: ignore[arg-type]

    def test_empty_quant_type_raises_error(self) -> None:
        """Test that empty quantization type raises ValueError."""
        config = GGUFConversionConfig(quantization_type="")
        with pytest.raises(ValueError, match="quantization_type cannot be empty"):
            validate_gguf_conversion_config(config)


class TestValidateSafeTensorsConversionConfig:
    """Tests for validate_safetensors_conversion_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = SafeTensorsConversionConfig()
        validate_safetensors_conversion_config(config)  # Should not raise

    def test_valid_config_with_metadata(self) -> None:
        """Test validating config with metadata."""
        config = SafeTensorsConversionConfig(metadata={"key": "value"})
        validate_safetensors_conversion_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_safetensors_conversion_config(None)  # type: ignore[arg-type]

    def test_zero_shard_size_raises_error(self) -> None:
        """Test that zero shard size raises ValueError."""
        config = SafeTensorsConversionConfig(shard_size=0)
        with pytest.raises(ValueError, match="shard_size must be positive"):
            validate_safetensors_conversion_config(config)

    def test_negative_shard_size_raises_error(self) -> None:
        """Test that negative shard size raises ValueError."""
        config = SafeTensorsConversionConfig(shard_size=-1)
        with pytest.raises(ValueError, match="shard_size must be positive"):
            validate_safetensors_conversion_config(config)

    def test_non_string_metadata_key_raises_error(self) -> None:
        """Test that non-string metadata key raises ValueError."""
        # Create a config with a dict that has a non-string key
        config = SafeTensorsConversionConfig(metadata={123: "value"})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="metadata keys must be strings"):
            validate_safetensors_conversion_config(config)

    def test_non_string_metadata_value_raises_error(self) -> None:
        """Test that non-string metadata value raises ValueError."""
        # Create a config with a dict that has a non-string value
        config = SafeTensorsConversionConfig(metadata={"key": 123})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="metadata values must be strings"):
            validate_safetensors_conversion_config(config)


class TestCreateConversionConfig:
    """Tests for create_conversion_config function."""

    def test_creates_config_from_strings(self) -> None:
        """Test creating config from string arguments."""
        config = create_conversion_config("huggingface", "safetensors")
        assert config.source_format == ModelFormat.HUGGINGFACE
        assert config.target_format == ModelFormat.SAFETENSORS

    def test_creates_config_from_enums(self) -> None:
        """Test creating config from enum arguments."""
        config = create_conversion_config(
            ModelFormat.HUGGINGFACE,
            ModelFormat.GGUF,
        )
        assert config.source_format == ModelFormat.HUGGINGFACE
        assert config.target_format == ModelFormat.GGUF

    def test_creates_config_with_precision(self) -> None:
        """Test creating config with precision."""
        config = create_conversion_config(
            "huggingface",
            "gguf",
            precision="int4",
        )
        assert config.precision == ConversionPrecision.INT4

    def test_creates_config_with_shard_size(self) -> None:
        """Test creating config with shard size."""
        config = create_conversion_config(
            "huggingface",
            "safetensors",
            shard_size_gb=10.0,
        )
        assert config.shard_size_gb == pytest.approx(10.0)

    def test_same_format_raises_error(self) -> None:
        """Test that same source and target raises error."""
        with pytest.raises(
            ValueError, match="source and target formats cannot be the same"
        ):
            create_conversion_config("gguf", "gguf")


class TestCreateGGUFConversionConfig:
    """Tests for create_gguf_conversion_config function."""

    def test_creates_config(self) -> None:
        """Test creating config."""
        config = create_gguf_conversion_config(quantization_type="q8_0")
        assert config.quantization_type == "q8_0"

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_gguf_conversion_config()
        assert config.quantization_type == "q4_k_m"
        assert config.use_mmap is True
        assert config.vocab_only is False

    def test_empty_quant_type_raises_error(self) -> None:
        """Test that empty quant type raises error."""
        with pytest.raises(ValueError, match="quantization_type cannot be empty"):
            create_gguf_conversion_config(quantization_type="")


class TestCreateSafeTensorsConversionConfig:
    """Tests for create_safetensors_conversion_config function."""

    def test_creates_config(self) -> None:
        """Test creating config."""
        config = create_safetensors_conversion_config()
        assert config.strict is True

    def test_creates_config_with_metadata(self) -> None:
        """Test creating config with metadata."""
        config = create_safetensors_conversion_config(
            metadata={"author": "user"},
        )
        assert config.metadata == {"author": "user"}

    def test_zero_shard_size_raises_error(self) -> None:
        """Test that zero shard size raises error."""
        with pytest.raises(ValueError, match="shard_size must be positive"):
            create_safetensors_conversion_config(shard_size=0)


class TestListModelFormats:
    """Tests for list_model_formats function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        formats = list_model_formats()
        assert isinstance(formats, list)

    def test_contains_expected_formats(self) -> None:
        """Test that list contains expected formats."""
        formats = list_model_formats()
        assert "huggingface" in formats
        assert "safetensors" in formats
        assert "gguf" in formats

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        formats = list_model_formats()
        assert formats == sorted(formats)


class TestListConversionPrecisions:
    """Tests for list_conversion_precisions function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        precisions = list_conversion_precisions()
        assert isinstance(precisions, list)

    def test_contains_expected_precisions(self) -> None:
        """Test that list contains expected precisions."""
        precisions = list_conversion_precisions()
        assert "fp32" in precisions
        assert "fp16" in precisions
        assert "int8" in precisions

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        precisions = list_conversion_precisions()
        assert precisions == sorted(precisions)


class TestListShardingStrategies:
    """Tests for list_sharding_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_sharding_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_sharding_strategies()
        assert "none" in strategies
        assert "layer" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_sharding_strategies()
        assert strategies == sorted(strategies)


class TestGetModelFormat:
    """Tests for get_model_format function."""

    def test_get_huggingface(self) -> None:
        """Test getting HUGGINGFACE format."""
        assert get_model_format("huggingface") == ModelFormat.HUGGINGFACE

    def test_get_safetensors(self) -> None:
        """Test getting SAFETENSORS format."""
        assert get_model_format("safetensors") == ModelFormat.SAFETENSORS

    def test_get_gguf(self) -> None:
        """Test getting GGUF format."""
        assert get_model_format("gguf") == ModelFormat.GGUF

    def test_invalid_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="invalid model format"):
            get_model_format("invalid")


class TestGetConversionPrecision:
    """Tests for get_conversion_precision function."""

    def test_get_fp32(self) -> None:
        """Test getting FP32 precision."""
        assert get_conversion_precision("fp32") == ConversionPrecision.FP32

    def test_get_int8(self) -> None:
        """Test getting INT8 precision."""
        assert get_conversion_precision("int8") == ConversionPrecision.INT8

    def test_invalid_raises_error(self) -> None:
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="invalid conversion precision"):
            get_conversion_precision("invalid")


class TestGetShardingStrategy:
    """Tests for get_sharding_strategy function."""

    def test_get_none(self) -> None:
        """Test getting NONE strategy."""
        assert get_sharding_strategy("none") == ShardingStrategy.NONE

    def test_get_layer(self) -> None:
        """Test getting LAYER strategy."""
        assert get_sharding_strategy("layer") == ShardingStrategy.LAYER

    def test_invalid_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid sharding strategy"):
            get_sharding_strategy("invalid")


class TestEstimateConvertedSize:
    """Tests for estimate_converted_size function."""

    def test_estimates_size(self) -> None:
        """Test estimating size."""
        size = estimate_converted_size(
            7_000_000_000,
            ConversionPrecision.FP32,
            ConversionPrecision.FP16,
        )
        assert size > 0
        assert size < 28000

    def test_int4_smaller_than_fp16(self) -> None:
        """Test that INT4 is smaller than FP16."""
        size_fp16 = estimate_converted_size(
            7_000_000_000,
            ConversionPrecision.FP32,
            ConversionPrecision.FP16,
        )
        size_int4 = estimate_converted_size(
            7_000_000_000,
            ConversionPrecision.FP32,
            ConversionPrecision.INT4,
        )
        assert size_int4 < size_fp16

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_converted_size(
                0,
                ConversionPrecision.FP32,
                ConversionPrecision.FP16,
            )

    def test_negative_params_raises_error(self) -> None:
        """Test that negative params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_converted_size(
                -1,
                ConversionPrecision.FP32,
                ConversionPrecision.FP16,
            )


class TestValidateFormatCompatibility:
    """Tests for validate_format_compatibility function."""

    def test_huggingface_to_safetensors(self) -> None:
        """Test HuggingFace to SafeTensors."""
        assert (
            validate_format_compatibility(
                ModelFormat.HUGGINGFACE,
                ModelFormat.SAFETENSORS,
            )
            is True
        )

    def test_huggingface_to_gguf(self) -> None:
        """Test HuggingFace to GGUF."""
        assert (
            validate_format_compatibility(
                ModelFormat.HUGGINGFACE,
                ModelFormat.GGUF,
            )
            is True
        )

    def test_huggingface_to_onnx(self) -> None:
        """Test HuggingFace to ONNX."""
        assert (
            validate_format_compatibility(
                ModelFormat.HUGGINGFACE,
                ModelFormat.ONNX,
            )
            is True
        )

    def test_tflite_to_gguf_unsupported(self) -> None:
        """Test TFLite to GGUF (unsupported)."""
        assert (
            validate_format_compatibility(
                ModelFormat.TFLITE,
                ModelFormat.GGUF,
            )
            is False
        )

    def test_same_format_unsupported(self) -> None:
        """Test same format (unsupported)."""
        assert (
            validate_format_compatibility(
                ModelFormat.HUGGINGFACE,
                ModelFormat.HUGGINGFACE,
            )
            is False
        )

    def test_safetensors_to_gguf(self) -> None:
        """Test SafeTensors to GGUF."""
        assert (
            validate_format_compatibility(
                ModelFormat.SAFETENSORS,
                ModelFormat.GGUF,
            )
            is True
        )

    def test_onnx_to_tflite(self) -> None:
        """Test ONNX to TFLite."""
        assert (
            validate_format_compatibility(
                ModelFormat.ONNX,
                ModelFormat.TFLITE,
            )
            is True
        )


class TestCalculatePrecisionLoss:
    """Tests for calculate_precision_loss function."""

    def test_same_precision_no_loss(self) -> None:
        """Test same precision has no loss."""
        loss = calculate_precision_loss(
            ConversionPrecision.FP32,
            ConversionPrecision.FP32,
        )
        assert loss == pytest.approx(0.0)

    def test_fp32_to_fp16(self) -> None:
        """Test FP32 to FP16 loss."""
        loss = calculate_precision_loss(
            ConversionPrecision.FP32,
            ConversionPrecision.FP16,
        )
        assert loss == pytest.approx(0.1)

    def test_fp32_to_int8(self) -> None:
        """Test FP32 to INT8 loss."""
        loss = calculate_precision_loss(
            ConversionPrecision.FP32,
            ConversionPrecision.INT8,
        )
        assert loss == pytest.approx(1.0)

    def test_fp32_to_int4(self) -> None:
        """Test FP32 to INT4 loss."""
        loss = calculate_precision_loss(
            ConversionPrecision.FP32,
            ConversionPrecision.INT4,
        )
        assert loss == pytest.approx(5.0)

    def test_fp16_to_fp32_no_loss(self) -> None:
        """Test FP16 to FP32 has no loss (upcast)."""
        loss = calculate_precision_loss(
            ConversionPrecision.FP16,
            ConversionPrecision.FP32,
        )
        assert loss == pytest.approx(0.0)

    def test_int8_to_int4(self) -> None:
        """Test INT8 to INT4 loss."""
        loss = calculate_precision_loss(
            ConversionPrecision.INT8,
            ConversionPrecision.INT4,
        )
        assert loss == pytest.approx(3.0)


class TestEstimateConversionTime:
    """Tests for estimate_conversion_time function."""

    def test_estimates_time(self) -> None:
        """Test estimating time."""
        time = estimate_conversion_time(
            7_000_000_000,
            ModelFormat.HUGGINGFACE,
            ModelFormat.SAFETENSORS,
        )
        assert time > 0
        assert time < 300

    def test_gguf_takes_longer(self) -> None:
        """Test that GGUF conversion takes longer."""
        time_st = estimate_conversion_time(
            7_000_000_000,
            ModelFormat.HUGGINGFACE,
            ModelFormat.SAFETENSORS,
        )
        time_gguf = estimate_conversion_time(
            7_000_000_000,
            ModelFormat.HUGGINGFACE,
            ModelFormat.GGUF,
        )
        assert time_gguf > time_st

    def test_quantization_increases_time(self) -> None:
        """Test that quantization increases time."""
        time_fp16 = estimate_conversion_time(
            7_000_000_000,
            ModelFormat.HUGGINGFACE,
            ModelFormat.GGUF,
            ConversionPrecision.FP16,
        )
        time_int4 = estimate_conversion_time(
            7_000_000_000,
            ModelFormat.HUGGINGFACE,
            ModelFormat.GGUF,
            ConversionPrecision.INT4,
        )
        assert time_int4 > time_fp16

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_conversion_time(
                0,
                ModelFormat.HUGGINGFACE,
                ModelFormat.SAFETENSORS,
            )


class TestFormatConversionStats:
    """Tests for format_conversion_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting stats."""
        stats = ConversionStats(
            original_size_mb=14000.0,
            converted_size_mb=4000.0,
            conversion_time_seconds=120.5,
            precision_loss=0.05,
        )
        formatted = format_conversion_stats(stats)
        assert "Original size:" in formatted
        assert "Converted size:" in formatted
        assert "Compression ratio:" in formatted
        assert "Conversion time:" in formatted
        assert "Precision loss:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_conversion_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedConversionConfig:
    """Tests for get_recommended_conversion_config function."""

    def test_7b_safetensors(self) -> None:
        """Test 7B to SafeTensors."""
        config = get_recommended_conversion_config("7b", "safetensors")
        assert config.precision == ConversionPrecision.FP16

    def test_7b_gguf_size(self) -> None:
        """Test 7B to GGUF with size optimization."""
        config = get_recommended_conversion_config(
            "7b",
            ModelFormat.GGUF,
            optimize_for="size",
        )
        assert config.precision == ConversionPrecision.INT4

    def test_70b_quality(self) -> None:
        """Test 70B with quality optimization."""
        config = get_recommended_conversion_config(
            "70b",
            "safetensors",
            optimize_for="quality",
        )
        assert config.precision == ConversionPrecision.FP32

    def test_invalid_size_raises_error(self) -> None:
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized model size"):
            get_recommended_conversion_config("invalid", "safetensors")

    def test_invalid_optimize_for_raises_error(self) -> None:
        """Test that invalid optimize_for raises ValueError."""
        with pytest.raises(ValueError, match="optimize_for must be one of"):
            get_recommended_conversion_config(
                "7b", "safetensors", optimize_for="invalid"
            )

    def test_speed_optimization(self) -> None:
        """Test speed optimization."""
        config = get_recommended_conversion_config(
            "13b",
            "safetensors",
            optimize_for="speed",
        )
        assert config.precision == ConversionPrecision.FP16

    def test_size_optimization_non_gguf(self) -> None:
        """Test size optimization for non-GGUF format uses INT8."""
        config = get_recommended_conversion_config(
            "7b",
            "safetensors",
            optimize_for="size",
        )
        assert config.precision == ConversionPrecision.INT8


class TestGetConversionConfigDict:
    """Tests for get_conversion_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_conversion_config("huggingface", "safetensors")
        d = get_conversion_config_dict(config)
        assert d["source_format"] == "huggingface"
        assert d["target_format"] == "safetensors"

    def test_includes_precision(self) -> None:
        """Test that precision is included."""
        config = create_conversion_config(
            "huggingface",
            "gguf",
            precision="int4",
        )
        d = get_conversion_config_dict(config)
        assert d["precision"] == "int4"

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_conversion_config_dict(None)  # type: ignore[arg-type]


class TestGetGGUFConversionConfigDict:
    """Tests for get_gguf_conversion_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_gguf_conversion_config(quantization_type="q4_k_m")
        d = get_gguf_conversion_config_dict(config)
        assert d["quantization_type"] == "q4_k_m"
        assert d["use_mmap"] is True

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_gguf_conversion_config_dict(None)  # type: ignore[arg-type]


class TestGetSafeTensorsConversionConfigDict:
    """Tests for get_safetensors_conversion_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_safetensors_conversion_config()
        d = get_safetensors_conversion_config_dict(config)
        assert d["strict"] is True
        assert d["shard_size"] == 5_000_000_000

    def test_with_metadata(self) -> None:
        """Test with metadata."""
        config = create_safetensors_conversion_config(metadata={"key": "value"})
        d = get_safetensors_conversion_config_dict(config)
        assert d["metadata"] == {"key": "value"}

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_safetensors_conversion_config_dict(None)  # type: ignore[arg-type]


class TestCreateConversionStats:
    """Tests for create_conversion_stats function."""

    def test_creates_stats(self) -> None:
        """Test creating stats."""
        stats = create_conversion_stats(14000.0, 4000.0, 120.5, 0.05)
        assert stats.original_size_mb == pytest.approx(14000.0)

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original size raises ValueError."""
        with pytest.raises(ValueError, match="original_size_mb must be positive"):
            create_conversion_stats(0, 4000.0, 120.5, 0.05)

    def test_zero_converted_raises_error(self) -> None:
        """Test that zero converted size raises ValueError."""
        with pytest.raises(ValueError, match="converted_size_mb must be positive"):
            create_conversion_stats(14000.0, 0, 120.5, 0.05)

    def test_negative_time_raises_error(self) -> None:
        """Test that negative time raises ValueError."""
        with pytest.raises(
            ValueError, match="conversion_time_seconds cannot be negative"
        ):
            create_conversion_stats(14000.0, 4000.0, -1.0, 0.05)

    def test_negative_precision_loss_raises_error(self) -> None:
        """Test that negative precision loss raises ValueError."""
        with pytest.raises(ValueError, match="precision_loss cannot be negative"):
            create_conversion_stats(14000.0, 4000.0, 120.5, -0.05)

    def test_zero_time_allowed(self) -> None:
        """Test that zero time is allowed."""
        stats = create_conversion_stats(14000.0, 4000.0, 0.0, 0.05)
        assert stats.conversion_time_seconds == pytest.approx(0.0)

    def test_zero_precision_loss_allowed(self) -> None:
        """Test that zero precision loss is allowed."""
        stats = create_conversion_stats(14000.0, 4000.0, 120.5, 0.0)
        assert stats.precision_loss == pytest.approx(0.0)
