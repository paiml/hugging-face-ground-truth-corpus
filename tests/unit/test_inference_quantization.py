"""Tests for inference quantization functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.inference.quantization import (
    VALID_CALIBRATION_STRATEGIES,
    VALID_QUANTIZATION_GRANULARITIES,
    VALID_QUANTIZATION_METHODS,
    AWQConfig,
    CalibrationStrategy,
    GGUFConfig,
    GPTQConfig,
    QuantizationConfig,
    QuantizationGranularity,
    QuantizationMethod,
    QuantizationStats,
    calculate_bits_per_weight,
    calculate_quantization_error,
    create_awq_config,
    create_gguf_config,
    create_gptq_config,
    create_quantization_config,
    create_quantization_stats,
    estimate_memory_savings,
    format_quantization_stats,
    get_calibration_strategy,
    get_quantization_granularity,
    get_quantization_method,
    get_recommended_quantization_config,
    list_calibration_strategies,
    list_quantization_granularities,
    list_quantization_methods,
    select_calibration_data,
    validate_awq_config,
    validate_gguf_config,
    validate_gptq_config,
    validate_quantization_config,
    validate_quantization_stats,
)


class TestQuantizationMethod:
    """Tests for QuantizationMethod enum."""

    def test_gptq_value(self) -> None:
        """Test GPTQ enum value."""
        assert QuantizationMethod.GPTQ.value == "gptq"

    def test_awq_value(self) -> None:
        """Test AWQ enum value."""
        assert QuantizationMethod.AWQ.value == "awq"

    def test_gguf_value(self) -> None:
        """Test GGUF enum value."""
        assert QuantizationMethod.GGUF.value == "gguf"

    def test_dynamic_value(self) -> None:
        """Test DYNAMIC enum value."""
        assert QuantizationMethod.DYNAMIC.value == "dynamic"

    def test_static_value(self) -> None:
        """Test STATIC enum value."""
        assert QuantizationMethod.STATIC.value == "static"

    def test_int8_value(self) -> None:
        """Test INT8 enum value."""
        assert QuantizationMethod.INT8.value == "int8"

    def test_int4_value(self) -> None:
        """Test INT4 enum value."""
        assert QuantizationMethod.INT4.value == "int4"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_QUANTIZATION_METHODS."""
        for method in QuantizationMethod:
            assert method.value in VALID_QUANTIZATION_METHODS


class TestCalibrationStrategy:
    """Tests for CalibrationStrategy enum."""

    def test_minmax_value(self) -> None:
        """Test MINMAX enum value."""
        assert CalibrationStrategy.MINMAX.value == "minmax"

    def test_percentile_value(self) -> None:
        """Test PERCENTILE enum value."""
        assert CalibrationStrategy.PERCENTILE.value == "percentile"

    def test_mse_value(self) -> None:
        """Test MSE enum value."""
        assert CalibrationStrategy.MSE.value == "mse"

    def test_entropy_value(self) -> None:
        """Test ENTROPY enum value."""
        assert CalibrationStrategy.ENTROPY.value == "entropy"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_CALIBRATION_STRATEGIES."""
        for strategy in CalibrationStrategy:
            assert strategy.value in VALID_CALIBRATION_STRATEGIES


class TestQuantizationGranularity:
    """Tests for QuantizationGranularity enum."""

    def test_per_tensor_value(self) -> None:
        """Test PER_TENSOR enum value."""
        assert QuantizationGranularity.PER_TENSOR.value == "per_tensor"

    def test_per_channel_value(self) -> None:
        """Test PER_CHANNEL enum value."""
        assert QuantizationGranularity.PER_CHANNEL.value == "per_channel"

    def test_per_group_value(self) -> None:
        """Test PER_GROUP enum value."""
        assert QuantizationGranularity.PER_GROUP.value == "per_group"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_QUANTIZATION_GRANULARITIES."""
        for granularity in QuantizationGranularity:
            assert granularity.value in VALID_QUANTIZATION_GRANULARITIES


class TestGPTQConfig:
    """Tests for GPTQConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating GPTQConfig instance."""
        config = GPTQConfig(bits=4, group_size=128, damp_percent=0.01, desc_act=True)
        assert config.bits == 4
        assert config.group_size == 128
        assert config.damp_percent == pytest.approx(0.01)
        assert config.desc_act is True

    def test_frozen(self) -> None:
        """Test that GPTQConfig is immutable."""
        config = GPTQConfig(bits=4, group_size=128, damp_percent=0.01, desc_act=True)
        with pytest.raises(AttributeError):
            config.bits = 8  # type: ignore[misc]


class TestAWQConfig:
    """Tests for AWQConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating AWQConfig instance."""
        config = AWQConfig(bits=4, group_size=128, zero_point=True, version="gemm")
        assert config.bits == 4
        assert config.group_size == 128
        assert config.zero_point is True
        assert config.version == "gemm"

    def test_frozen(self) -> None:
        """Test that AWQConfig is immutable."""
        config = AWQConfig(bits=4, group_size=128, zero_point=True, version="gemm")
        with pytest.raises(AttributeError):
            config.bits = 8  # type: ignore[misc]


class TestGGUFConfig:
    """Tests for GGUFConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating GGUFConfig instance."""
        config = GGUFConfig(quantization_type="q4_0", allow_requantize=False)
        assert config.quantization_type == "q4_0"
        assert config.allow_requantize is False

    def test_frozen(self) -> None:
        """Test that GGUFConfig is immutable."""
        config = GGUFConfig(quantization_type="q4_0", allow_requantize=False)
        with pytest.raises(AttributeError):
            config.quantization_type = "q8_0"  # type: ignore[misc]


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating QuantizationConfig instance."""
        config = QuantizationConfig(
            method=QuantizationMethod.DYNAMIC,
            bits=8,
            calibration_strategy=CalibrationStrategy.MINMAX,
            calibration_samples=128,
            granularity=QuantizationGranularity.PER_TENSOR,
        )
        assert config.method == QuantizationMethod.DYNAMIC
        assert config.bits == 8
        assert config.calibration_strategy == CalibrationStrategy.MINMAX
        assert config.calibration_samples == 128
        assert config.granularity == QuantizationGranularity.PER_TENSOR

    def test_frozen(self) -> None:
        """Test that QuantizationConfig is immutable."""
        config = QuantizationConfig(
            method=QuantizationMethod.DYNAMIC,
            bits=8,
            calibration_strategy=CalibrationStrategy.MINMAX,
            calibration_samples=128,
            granularity=QuantizationGranularity.PER_TENSOR,
        )
        with pytest.raises(AttributeError):
            config.bits = 4  # type: ignore[misc]


class TestQuantizationStats:
    """Tests for QuantizationStats dataclass."""

    def test_creation(self) -> None:
        """Test creating QuantizationStats instance."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        assert stats.original_size_mb == pytest.approx(14000.0)
        assert stats.quantized_size_mb == pytest.approx(4000.0)
        assert stats.compression_ratio == pytest.approx(3.5)
        assert stats.accuracy_drop == pytest.approx(0.5)

    def test_frozen(self) -> None:
        """Test that QuantizationStats is immutable."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        with pytest.raises(AttributeError):
            stats.compression_ratio = 2.0  # type: ignore[misc]


class TestValidateGPTQConfig:
    """Tests for validate_gptq_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = GPTQConfig(bits=4, group_size=128, damp_percent=0.01, desc_act=True)
        validate_gptq_config(config)  # Should not raise

    def test_valid_8bit_config(self) -> None:
        """Test validating valid 8-bit config."""
        config = GPTQConfig(bits=8, group_size=64, damp_percent=0.05, desc_act=False)
        validate_gptq_config(config)  # Should not raise

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        config = GPTQConfig(bits=5, group_size=128, damp_percent=0.01, desc_act=True)
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            validate_gptq_config(config)

    def test_zero_group_size_raises_error(self) -> None:
        """Test that zero group_size raises ValueError."""
        config = GPTQConfig(bits=4, group_size=0, damp_percent=0.01, desc_act=True)
        with pytest.raises(ValueError, match="group_size must be positive"):
            validate_gptq_config(config)

    def test_negative_group_size_raises_error(self) -> None:
        """Test that negative group_size raises ValueError."""
        config = GPTQConfig(bits=4, group_size=-1, damp_percent=0.01, desc_act=True)
        with pytest.raises(ValueError, match="group_size must be positive"):
            validate_gptq_config(config)

    def test_invalid_damp_percent_raises_error(self) -> None:
        """Test that invalid damp_percent raises ValueError."""
        config = GPTQConfig(bits=4, group_size=128, damp_percent=1.5, desc_act=True)
        with pytest.raises(ValueError, match="damp_percent must be between"):
            validate_gptq_config(config)

    def test_negative_damp_percent_raises_error(self) -> None:
        """Test that negative damp_percent raises ValueError."""
        config = GPTQConfig(bits=4, group_size=128, damp_percent=-0.1, desc_act=True)
        with pytest.raises(ValueError, match="damp_percent must be between"):
            validate_gptq_config(config)


class TestValidateAWQConfig:
    """Tests for validate_awq_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = AWQConfig(bits=4, group_size=128, zero_point=True, version="gemm")
        validate_awq_config(config)  # Should not raise

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        config = AWQConfig(bits=8, group_size=128, zero_point=True, version="gemm")
        with pytest.raises(ValueError, match="AWQ only supports 4-bit"):
            validate_awq_config(config)

    def test_zero_group_size_raises_error(self) -> None:
        """Test that zero group_size raises ValueError."""
        config = AWQConfig(bits=4, group_size=0, zero_point=True, version="gemm")
        with pytest.raises(ValueError, match="group_size must be positive"):
            validate_awq_config(config)

    def test_invalid_version_raises_error(self) -> None:
        """Test that invalid version raises ValueError."""
        config = AWQConfig(bits=4, group_size=128, zero_point=True, version="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="version must be one of"):
            validate_awq_config(config)


class TestValidateGGUFConfig:
    """Tests for validate_gguf_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = GGUFConfig(quantization_type="q4_0", allow_requantize=False)
        validate_gguf_config(config)  # Should not raise

    def test_valid_q8_config(self) -> None:
        """Test validating valid q8 config."""
        config = GGUFConfig(quantization_type="q8_0", allow_requantize=True)
        validate_gguf_config(config)  # Should not raise

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid quantization_type raises ValueError."""
        config = GGUFConfig(quantization_type="invalid", allow_requantize=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="quantization_type must be one of"):
            validate_gguf_config(config)


class TestValidateQuantizationConfig:
    """Tests for validate_quantization_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = QuantizationConfig(
            method=QuantizationMethod.DYNAMIC,
            bits=8,
            calibration_strategy=CalibrationStrategy.MINMAX,
            calibration_samples=128,
            granularity=QuantizationGranularity.PER_TENSOR,
        )
        validate_quantization_config(config)  # Should not raise

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        config = QuantizationConfig(
            method=QuantizationMethod.DYNAMIC,
            bits=6,
            calibration_strategy=CalibrationStrategy.MINMAX,
            calibration_samples=128,
            granularity=QuantizationGranularity.PER_TENSOR,
        )
        with pytest.raises(ValueError, match="bits must be one of"):
            validate_quantization_config(config)

    def test_zero_calibration_samples_raises_error(self) -> None:
        """Test that zero calibration_samples raises ValueError."""
        config = QuantizationConfig(
            method=QuantizationMethod.DYNAMIC,
            bits=8,
            calibration_strategy=CalibrationStrategy.MINMAX,
            calibration_samples=0,
            granularity=QuantizationGranularity.PER_TENSOR,
        )
        with pytest.raises(ValueError, match="calibration_samples must be positive"):
            validate_quantization_config(config)


class TestValidateQuantizationStats:
    """Tests for validate_quantization_stats function."""

    def test_valid_stats(self) -> None:
        """Test validating valid stats."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        validate_quantization_stats(stats)  # Should not raise

    def test_zero_original_size_raises_error(self) -> None:
        """Test that zero original_size raises ValueError."""
        stats = QuantizationStats(
            original_size_mb=0.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        with pytest.raises(ValueError, match="original_size_mb must be positive"):
            validate_quantization_stats(stats)

    def test_negative_original_size_raises_error(self) -> None:
        """Test that negative original_size raises ValueError."""
        stats = QuantizationStats(
            original_size_mb=-100.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        with pytest.raises(ValueError, match="original_size_mb must be positive"):
            validate_quantization_stats(stats)

    def test_zero_quantized_size_raises_error(self) -> None:
        """Test that zero quantized_size raises ValueError."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=0.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        with pytest.raises(ValueError, match="quantized_size_mb must be positive"):
            validate_quantization_stats(stats)

    def test_zero_compression_ratio_raises_error(self) -> None:
        """Test that zero compression_ratio raises ValueError."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=4000.0,
            compression_ratio=0.0,
            accuracy_drop=0.5,
        )
        with pytest.raises(ValueError, match="compression_ratio must be positive"):
            validate_quantization_stats(stats)

    def test_negative_accuracy_drop_raises_error(self) -> None:
        """Test that negative accuracy_drop raises ValueError."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=-0.5,
        )
        with pytest.raises(ValueError, match="accuracy_drop cannot be negative"):
            validate_quantization_stats(stats)


class TestCreateGPTQConfig:
    """Tests for create_gptq_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_gptq_config()
        assert config.bits == 4
        assert config.group_size == 128
        assert config.damp_percent == pytest.approx(0.01)
        assert config.desc_act is True

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_gptq_config(bits=8, group_size=64, damp_percent=0.05)
        assert config.bits == 8
        assert config.group_size == 64
        assert config.damp_percent == pytest.approx(0.05)

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            create_gptq_config(bits=5)


class TestCreateAWQConfig:
    """Tests for create_awq_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_awq_config()
        assert config.bits == 4
        assert config.group_size == 128
        assert config.zero_point is True
        assert config.version == "gemm"

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_awq_config(group_size=64, version="marlin")
        assert config.group_size == 64
        assert config.version == "marlin"

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="AWQ only supports 4-bit"):
            create_awq_config(bits=8)

    def test_invalid_version_raises_error(self) -> None:
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="version must be one of"):
            create_awq_config(version="invalid")  # type: ignore[arg-type]


class TestCreateGGUFConfig:
    """Tests for create_gguf_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_gguf_config()
        assert config.quantization_type == "q4_0"
        assert config.allow_requantize is False

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_gguf_config(quantization_type="q8_0", allow_requantize=True)
        assert config.quantization_type == "q8_0"
        assert config.allow_requantize is True

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid quantization_type raises ValueError."""
        with pytest.raises(ValueError, match="quantization_type must be one of"):
            create_gguf_config(quantization_type="invalid")  # type: ignore[arg-type]


class TestCreateQuantizationConfig:
    """Tests for create_quantization_config function."""

    def test_default_values(self) -> None:
        """Test creating config with default values."""
        config = create_quantization_config()
        assert config.method == QuantizationMethod.DYNAMIC
        assert config.bits == 8
        assert config.calibration_strategy == CalibrationStrategy.MINMAX
        assert config.calibration_samples == 128
        assert config.granularity == QuantizationGranularity.PER_TENSOR

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_quantization_config(
            method="gptq",
            bits=4,
            calibration_strategy="entropy",
            calibration_samples=256,
            granularity="per_group",
        )
        assert config.method == QuantizationMethod.GPTQ
        assert config.bits == 4
        assert config.calibration_strategy == CalibrationStrategy.ENTROPY
        assert config.calibration_samples == 256
        assert config.granularity == QuantizationGranularity.PER_GROUP

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_quantization_config(method="invalid")  # type: ignore[arg-type]

    def test_invalid_calibration_strategy_raises_error(self) -> None:
        """Test that invalid calibration_strategy raises ValueError."""
        with pytest.raises(ValueError, match="calibration_strategy must be one of"):
            create_quantization_config(calibration_strategy="invalid")  # type: ignore[arg-type]

    def test_invalid_granularity_raises_error(self) -> None:
        """Test that invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="granularity must be one of"):
            create_quantization_config(granularity="invalid")  # type: ignore[arg-type]

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="bits must be one of"):
            create_quantization_config(bits=6)


class TestCreateQuantizationStats:
    """Tests for create_quantization_stats function."""

    def test_computes_compression_ratio(self) -> None:
        """Test that compression_ratio is computed."""
        stats = create_quantization_stats(14000.0, 4000.0)
        assert stats.compression_ratio == pytest.approx(3.5)

    def test_accepts_custom_compression_ratio(self) -> None:
        """Test accepting custom compression_ratio."""
        stats = create_quantization_stats(14000.0, 4000.0, compression_ratio=4.0)
        assert stats.compression_ratio == pytest.approx(4.0)

    def test_accepts_accuracy_drop(self) -> None:
        """Test accepting accuracy_drop."""
        stats = create_quantization_stats(14000.0, 4000.0, accuracy_drop=0.5)
        assert stats.accuracy_drop == pytest.approx(0.5)

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original_size raises ValueError."""
        with pytest.raises(ValueError, match="original_size_mb must be positive"):
            create_quantization_stats(0.0, 4000.0)

    def test_zero_quantized_raises_error(self) -> None:
        """Test that zero quantized_size raises ValueError."""
        with pytest.raises(ValueError, match="quantized_size_mb must be positive"):
            create_quantization_stats(14000.0, 0.0)


class TestListQuantizationMethods:
    """Tests for list_quantization_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_quantization_methods()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_quantization_methods()
        assert result == sorted(result)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        result = list_quantization_methods()
        assert "gptq" in result
        assert "awq" in result
        assert "dynamic" in result
        assert "int8" in result


class TestListCalibrationStrategies:
    """Tests for list_calibration_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_calibration_strategies()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_calibration_strategies()
        assert result == sorted(result)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        result = list_calibration_strategies()
        assert "minmax" in result
        assert "entropy" in result


class TestListQuantizationGranularities:
    """Tests for list_quantization_granularities function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_quantization_granularities()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_quantization_granularities()
        assert result == sorted(result)

    def test_contains_expected_granularities(self) -> None:
        """Test that list contains expected granularities."""
        result = list_quantization_granularities()
        assert "per_tensor" in result
        assert "per_channel" in result
        assert "per_group" in result


class TestGetQuantizationMethod:
    """Tests for get_quantization_method function."""

    def test_get_gptq(self) -> None:
        """Test getting GPTQ method."""
        assert get_quantization_method("gptq") == QuantizationMethod.GPTQ

    def test_get_dynamic(self) -> None:
        """Test getting DYNAMIC method."""
        assert get_quantization_method("dynamic") == QuantizationMethod.DYNAMIC

    def test_invalid_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization method"):
            get_quantization_method("invalid")


class TestGetCalibrationStrategy:
    """Tests for get_calibration_strategy function."""

    def test_get_minmax(self) -> None:
        """Test getting MINMAX strategy."""
        assert get_calibration_strategy("minmax") == CalibrationStrategy.MINMAX

    def test_get_entropy(self) -> None:
        """Test getting ENTROPY strategy."""
        assert get_calibration_strategy("entropy") == CalibrationStrategy.ENTROPY

    def test_invalid_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown calibration strategy"):
            get_calibration_strategy("invalid")


class TestGetQuantizationGranularity:
    """Tests for get_quantization_granularity function."""

    def test_get_per_tensor(self) -> None:
        """Test getting PER_TENSOR granularity."""
        result = get_quantization_granularity("per_tensor")
        assert result == QuantizationGranularity.PER_TENSOR

    def test_get_per_channel(self) -> None:
        """Test getting PER_CHANNEL granularity."""
        result = get_quantization_granularity("per_channel")
        assert result == QuantizationGranularity.PER_CHANNEL

    def test_invalid_raises_error(self) -> None:
        """Test that invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization granularity"):
            get_quantization_granularity("invalid")


class TestCalculateQuantizationError:
    """Tests for calculate_quantization_error function."""

    def test_mse_error(self) -> None:
        """Test MSE error calculation."""
        original = [1.0, 2.0, 3.0, 4.0]
        quantized = [1.1, 2.1, 3.1, 4.1]
        error = calculate_quantization_error(original, quantized, "mse")
        assert abs(error - 0.01) < 0.001

    def test_mae_error(self) -> None:
        """Test MAE error calculation."""
        original = [1.0, 2.0, 3.0, 4.0]
        quantized = [1.1, 2.1, 3.1, 4.1]
        error = calculate_quantization_error(original, quantized, "mae")
        assert abs(error - 0.1) < 0.001

    def test_max_error(self) -> None:
        """Test max error calculation."""
        original = [1.0, 2.0, 3.0, 4.0]
        quantized = [1.1, 2.1, 3.1, 4.2]
        error = calculate_quantization_error(original, quantized, "max")
        assert abs(error - 0.2) < 0.001

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="original_values cannot be empty"):
            calculate_quantization_error([], [])

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_quantization_error([1.0], [1.0, 2.0])

    def test_zero_error(self) -> None:
        """Test zero error when values are identical."""
        original = [1.0, 2.0, 3.0]
        quantized = [1.0, 2.0, 3.0]
        error = calculate_quantization_error(original, quantized, "mse")
        assert error == pytest.approx(0.0)

    @given(
        st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1)
    )
    @settings(max_examples=50)
    def test_zero_error_identical_values(self, values: list[float]) -> None:
        """Test that identical values have zero error."""
        error = calculate_quantization_error(values, values, "mse")
        assert error == pytest.approx(0.0)


class TestEstimateMemorySavings:
    """Tests for estimate_memory_savings function."""

    def test_4x_compression(self) -> None:
        """Test 4x compression (fp16 to int4)."""
        orig, quant = estimate_memory_savings(16, 4, 7_000_000_000)
        assert orig / quant == pytest.approx(4.0)

    def test_2x_compression(self) -> None:
        """Test 2x compression (fp16 to int8)."""
        orig, quant = estimate_memory_savings(16, 8, 7_000_000_000)
        assert orig / quant == pytest.approx(2.0)

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original_bits raises ValueError."""
        with pytest.raises(ValueError, match="original_bits must be positive"):
            estimate_memory_savings(0, 4, 1000)

    def test_zero_quantized_raises_error(self) -> None:
        """Test that zero quantized_bits raises ValueError."""
        with pytest.raises(ValueError, match="quantized_bits must be positive"):
            estimate_memory_savings(16, 0, 1000)

    def test_zero_params_raises_error(self) -> None:
        """Test that zero model_params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_memory_savings(16, 4, 0)

    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=50)
    def test_savings_always_positive(self, original: int, quantized: int) -> None:
        """Test that savings are always positive."""
        orig, quant = estimate_memory_savings(original, quantized, 1_000_000)
        assert orig > 0
        assert quant > 0


class TestCalculateBitsPerWeight:
    """Tests for calculate_bits_per_weight function."""

    def test_int8(self) -> None:
        """Test INT8 bits per weight."""
        assert calculate_bits_per_weight("int8") == 8.0

    def test_int4(self) -> None:
        """Test INT4 bits per weight."""
        assert calculate_bits_per_weight("int4") == 4.0

    def test_q4_0(self) -> None:
        """Test GGUF q4_0 bits per weight."""
        assert calculate_bits_per_weight("q4_0") == 4.5

    def test_fp16(self) -> None:
        """Test FP16 bits per weight."""
        assert calculate_bits_per_weight("fp16") == 16.0

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization type"):
            calculate_bits_per_weight("invalid")


class TestSelectCalibrationData:
    """Tests for select_calibration_data function."""

    def test_correct_count(self) -> None:
        """Test that correct number of indices is returned."""
        indices = select_calibration_data(1000, 128, seed=42)
        assert len(indices) == 128

    def test_valid_range(self) -> None:
        """Test that all indices are within valid range."""
        indices = select_calibration_data(1000, 128, seed=42)
        assert all(0 <= i < 1000 for i in indices)

    def test_unique_indices(self) -> None:
        """Test that all indices are unique."""
        indices = select_calibration_data(1000, 128, seed=42)
        assert len(set(indices)) == len(indices)

    def test_deterministic_with_seed(self) -> None:
        """Test that results are deterministic with same seed."""
        indices1 = select_calibration_data(1000, 128, seed=42)
        indices2 = select_calibration_data(1000, 128, seed=42)
        assert indices1 == indices2

    def test_different_with_different_seed(self) -> None:
        """Test that results are different with different seed."""
        indices1 = select_calibration_data(1000, 128, seed=42)
        indices2 = select_calibration_data(1000, 128, seed=43)
        assert indices1 != indices2

    def test_zero_dataset_size_raises_error(self) -> None:
        """Test that zero dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            select_calibration_data(0, 128)

    def test_zero_target_samples_raises_error(self) -> None:
        """Test that zero target_samples raises ValueError."""
        with pytest.raises(ValueError, match="target_samples must be positive"):
            select_calibration_data(1000, 0)

    def test_target_exceeds_dataset_raises_error(self) -> None:
        """Test that target exceeding dataset raises ValueError."""
        with pytest.raises(ValueError, match="target_samples cannot exceed"):
            select_calibration_data(100, 200)

    def test_full_dataset_selection(self) -> None:
        """Test selecting all items from dataset."""
        indices = select_calibration_data(100, 100, seed=42)
        assert len(indices) == 100
        assert len(set(indices)) == 100


class TestFormatQuantizationStats:
    """Tests for format_quantization_stats function."""

    def test_formats_all_fields(self) -> None:
        """Test that all fields are formatted."""
        stats = QuantizationStats(
            original_size_mb=14000.0,
            quantized_size_mb=4000.0,
            compression_ratio=3.5,
            accuracy_drop=0.5,
        )
        formatted = format_quantization_stats(stats)
        assert "Original Size: 14000.00 MB" in formatted
        assert "Quantized Size: 4000.00 MB" in formatted
        assert "Compression Ratio: 3.50x" in formatted
        assert "Accuracy Drop: 0.50%" in formatted

    def test_zero_accuracy_drop(self) -> None:
        """Test formatting with zero accuracy drop."""
        stats = QuantizationStats(100.0, 25.0, 4.0, 0.0)
        formatted = format_quantization_stats(stats)
        assert "Accuracy Drop: 0.00%" in formatted


class TestGetRecommendedQuantizationConfig:
    """Tests for get_recommended_quantization_config function."""

    def test_large_gpu(self) -> None:
        """Test recommendation for large model on GPU."""
        config = get_recommended_quantization_config("large", "gpu")
        assert config.method == QuantizationMethod.GPTQ
        assert config.bits == 4

    def test_small_cpu(self) -> None:
        """Test recommendation for small model on CPU."""
        config = get_recommended_quantization_config("small", "cpu")
        assert config.method == QuantizationMethod.DYNAMIC
        assert config.bits == 8

    def test_medium_mobile(self) -> None:
        """Test recommendation for medium model on mobile."""
        config = get_recommended_quantization_config("medium", "mobile")
        assert config.bits == 4

    def test_xlarge_cpu(self) -> None:
        """Test recommendation for xlarge model on CPU."""
        config = get_recommended_quantization_config("xlarge", "cpu")
        assert config.method == QuantizationMethod.INT8

    def test_small_gpu(self) -> None:
        """Test recommendation for small model on GPU."""
        config = get_recommended_quantization_config("small", "gpu")
        assert config.method == QuantizationMethod.DYNAMIC

    def test_medium_gpu(self) -> None:
        """Test recommendation for medium model on GPU."""
        config = get_recommended_quantization_config("medium", "gpu")
        assert config.method == QuantizationMethod.INT8

    def test_invalid_model_size_raises_error(self) -> None:
        """Test that invalid model_size raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model size"):
            get_recommended_quantization_config("invalid", "gpu")  # type: ignore[arg-type]

    def test_invalid_device_raises_error(self) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target device"):
            get_recommended_quantization_config("large", "invalid")  # type: ignore[arg-type]


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        st.floats(min_value=0.001, max_value=1e9),
        st.floats(min_value=0.001, max_value=1e9),
    )
    @settings(max_examples=100)
    def test_compression_ratio_positive(
        self, original: float, quantized: float
    ) -> None:
        """Test that compression ratio is always positive."""
        stats = create_quantization_stats(original, quantized)
        assert stats.compression_ratio > 0

    @given(
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=1, max_value=10_000_000_000),
    )
    @settings(max_examples=50)
    def test_memory_savings_ratio(
        self, original_bits: int, quantized_bits: int, params: int
    ) -> None:
        """Test memory savings ratio matches bits ratio."""
        orig, quant = estimate_memory_savings(original_bits, quantized_bits, params)
        ratio = orig / quant
        expected_ratio = original_bits / quantized_bits
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    @given(
        st.integers(min_value=10, max_value=10000),
        st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_calibration_selection_valid(
        self, dataset_size: int, target_samples: int
    ) -> None:
        """Test calibration selection always produces valid indices."""
        if target_samples > dataset_size:
            target_samples = dataset_size

        indices = select_calibration_data(dataset_size, target_samples, seed=42)
        assert len(indices) == target_samples
        assert all(0 <= i < dataset_size for i in indices)
        assert len(set(indices)) == len(indices)
