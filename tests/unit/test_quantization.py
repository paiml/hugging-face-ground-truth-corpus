"""Tests for model quantization utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.quantization import (
    AWQConfig,
    CalibrationConfig,
    CalibrationMethod,
    GPTQConfig,
    QuantGranularity,
    QuantMethod,
    QuantProfile,
    QuantResult,
    compute_compression_ratio,
    create_awq_config,
    create_calibration_config,
    create_gptq_config,
    create_quant_profile,
    create_quant_result,
    estimate_quantized_size,
    format_quant_result,
    get_awq_dict,
    get_calibration_method,
    get_gptq_dict,
    get_quant_granularity,
    get_quant_method,
    get_recommended_profile,
    list_calibration_methods,
    list_quant_granularities,
    list_quant_methods,
    validate_calibration_config,
    validate_calibration_method,
    validate_quant_granularity,
    validate_quant_method,
    validate_quant_profile,
)


class TestQuantMethod:
    """Tests for QuantMethod enum."""

    def test_gptq_value(self) -> None:
        """Test GPTQ value."""
        assert QuantMethod.GPTQ.value == "gptq"

    def test_awq_value(self) -> None:
        """Test AWQ value."""
        assert QuantMethod.AWQ.value == "awq"

    def test_gguf_value(self) -> None:
        """Test GGUF value."""
        assert QuantMethod.GGUF.value == "gguf"

    def test_bitsandbytes_value(self) -> None:
        """Test BITSANDBYTES value."""
        assert QuantMethod.BITSANDBYTES.value == "bitsandbytes"


class TestCalibrationMethod:
    """Tests for CalibrationMethod enum."""

    def test_minmax_value(self) -> None:
        """Test MINMAX value."""
        assert CalibrationMethod.MINMAX.value == "minmax"

    def test_entropy_value(self) -> None:
        """Test ENTROPY value."""
        assert CalibrationMethod.ENTROPY.value == "entropy"

    def test_percentile_value(self) -> None:
        """Test PERCENTILE value."""
        assert CalibrationMethod.PERCENTILE.value == "percentile"


class TestQuantGranularity:
    """Tests for QuantGranularity enum."""

    def test_per_tensor_value(self) -> None:
        """Test PER_TENSOR value."""
        assert QuantGranularity.PER_TENSOR.value == "per_tensor"

    def test_per_channel_value(self) -> None:
        """Test PER_CHANNEL value."""
        assert QuantGranularity.PER_CHANNEL.value == "per_channel"

    def test_per_group_value(self) -> None:
        """Test PER_GROUP value."""
        assert QuantGranularity.PER_GROUP.value == "per_group"


class TestQuantProfile:
    """Tests for QuantProfile dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        profile = QuantProfile()
        assert profile.method == QuantMethod.GPTQ
        assert profile.bits == 4
        assert profile.group_size == 128

    def test_custom_values(self) -> None:
        """Test custom values."""
        profile = QuantProfile(method=QuantMethod.AWQ, bits=4, group_size=64)
        assert profile.method == QuantMethod.AWQ
        assert profile.group_size == 64

    def test_frozen(self) -> None:
        """Test that QuantProfile is immutable."""
        profile = QuantProfile()
        with pytest.raises(AttributeError):
            profile.bits = 8  # type: ignore[misc]


class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = CalibrationConfig()
        assert config.method == CalibrationMethod.MINMAX
        assert config.num_samples == 128
        assert config.sequence_length == 2048

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = CalibrationConfig(num_samples=256, sequence_length=4096)
        assert config.num_samples == 256
        assert config.sequence_length == 4096


class TestQuantResult:
    """Tests for QuantResult dataclass."""

    def test_creation(self) -> None:
        """Test creating result."""
        result = QuantResult(
            original_size_mb=14000,
            quantized_size_mb=4000,
            compression_ratio=3.5,
        )
        assert result.compression_ratio == 3.5

    def test_with_perplexity(self) -> None:
        """Test with perplexity values."""
        result = QuantResult(
            original_size_mb=14000,
            quantized_size_mb=4000,
            compression_ratio=3.5,
            perplexity_before=5.0,
            perplexity_after=5.5,
        )
        assert result.perplexity_before == 5.0


class TestGPTQConfig:
    """Tests for GPTQConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = GPTQConfig()
        assert config.bits == 4
        assert config.group_size == 128
        assert config.damp_percent == 0.01

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = GPTQConfig(bits=8, group_size=64)
        assert config.bits == 8
        assert config.group_size == 64


class TestAWQConfig:
    """Tests for AWQConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = AWQConfig()
        assert config.bits == 4
        assert config.group_size == 128
        assert config.version == "gemm"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = AWQConfig(group_size=64, version="marlin")
        assert config.group_size == 64
        assert config.version == "marlin"


class TestValidateQuantProfile:
    """Tests for validate_quant_profile function."""

    def test_valid_profile(self) -> None:
        """Test validating valid profile."""
        profile = QuantProfile(method=QuantMethod.GPTQ, bits=4)
        validate_quant_profile(profile)  # Should not raise

    def test_none_profile_raises_error(self) -> None:
        """Test that None profile raises ValueError."""
        with pytest.raises(ValueError, match="profile cannot be None"):
            validate_quant_profile(None)  # type: ignore[arg-type]

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        profile = QuantProfile(bits=5)
        with pytest.raises(ValueError, match="bits must be 2, 3, 4, or 8"):
            validate_quant_profile(profile)

    def test_zero_group_size_raises_error(self) -> None:
        """Test that zero group_size raises ValueError."""
        profile = QuantProfile(group_size=0)
        with pytest.raises(ValueError, match="group_size must be positive"):
            validate_quant_profile(profile)


class TestValidateCalibrationConfig:
    """Tests for validate_calibration_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = CalibrationConfig(num_samples=128)
        validate_calibration_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_calibration_config(None)  # type: ignore[arg-type]

    def test_zero_samples_raises_error(self) -> None:
        """Test that zero num_samples raises ValueError."""
        config = CalibrationConfig(num_samples=0)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            validate_calibration_config(config)

    def test_zero_sequence_length_raises_error(self) -> None:
        """Test that zero sequence_length raises ValueError."""
        config = CalibrationConfig(sequence_length=0)
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            validate_calibration_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = CalibrationConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_calibration_config(config)


class TestCreateQuantProfile:
    """Tests for create_quant_profile function."""

    def test_creates_profile(self) -> None:
        """Test creating profile."""
        profile = create_quant_profile(method="gptq", bits=4)
        assert profile.method == QuantMethod.GPTQ
        assert profile.bits == 4

    def test_enum_method(self) -> None:
        """Test with enum method."""
        profile = create_quant_profile(method=QuantMethod.AWQ)
        assert profile.method == QuantMethod.AWQ


class TestCreateCalibrationConfig:
    """Tests for create_calibration_config function."""

    def test_creates_config(self) -> None:
        """Test creating config."""
        config = create_calibration_config(num_samples=256)
        assert config.num_samples == 256

    def test_string_method(self) -> None:
        """Test with string method."""
        config = create_calibration_config(method="entropy")
        assert config.method == CalibrationMethod.ENTROPY


class TestCreateGPTQConfig:
    """Tests for create_gptq_config function."""

    def test_creates_config(self) -> None:
        """Test creating config."""
        config = create_gptq_config(bits=4)
        assert config.bits == 4

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            create_gptq_config(bits=3)

    def test_zero_group_size_raises_error(self) -> None:
        """Test that zero group_size raises ValueError."""
        with pytest.raises(ValueError, match="group_size must be positive"):
            create_gptq_config(group_size=0)


class TestCreateAWQConfig:
    """Tests for create_awq_config function."""

    def test_creates_config(self) -> None:
        """Test creating config."""
        config = create_awq_config(bits=4)
        assert config.bits == 4

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="AWQ only supports 4-bit"):
            create_awq_config(bits=8)

    def test_invalid_version_raises_error(self) -> None:
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="version must be one of"):
            create_awq_config(version="invalid")

    def test_zero_group_size_raises_error(self) -> None:
        """Test that zero group_size raises ValueError."""
        with pytest.raises(ValueError, match="group_size must be positive"):
            create_awq_config(group_size=0)


class TestEstimateQuantizedSize:
    """Tests for estimate_quantized_size function."""

    def test_estimates_size(self) -> None:
        """Test estimating size."""
        size = estimate_quantized_size(7_000_000_000, bits=4)
        assert size > 0
        assert size < 7000

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_quantized_size(0)

    def test_invalid_bits_raises_error(self) -> None:
        """Test that invalid bits raises ValueError."""
        with pytest.raises(ValueError, match="bits must be one of"):
            estimate_quantized_size(7_000_000_000, bits=5)


class TestComputeCompressionRatio:
    """Tests for compute_compression_ratio function."""

    def test_computes_ratio(self) -> None:
        """Test computing ratio."""
        assert compute_compression_ratio(16, 4) == 4.0
        assert compute_compression_ratio(32, 4) == 8.0

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original raises ValueError."""
        with pytest.raises(ValueError, match="original_bits must be positive"):
            compute_compression_ratio(0, 4)


class TestCreateQuantResult:
    """Tests for create_quant_result function."""

    def test_creates_result(self) -> None:
        """Test creating result."""
        result = create_quant_result(14000, 4000)
        assert result.compression_ratio == 3.5

    def test_with_perplexity(self) -> None:
        """Test with perplexity values."""
        result = create_quant_result(14000, 4000, 5.0, 5.5)
        assert result.perplexity_degradation == 10.0

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original raises ValueError."""
        with pytest.raises(ValueError, match="original_size_mb must be positive"):
            create_quant_result(0, 4000)


class TestGetGPTQDict:
    """Tests for get_gptq_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_gptq_config(bits=4)
        d = get_gptq_dict(config)
        assert d["bits"] == 4
        assert d["group_size"] == 128

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_gptq_dict(None)  # type: ignore[arg-type]


class TestGetAWQDict:
    """Tests for get_awq_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_awq_config(bits=4)
        d = get_awq_dict(config)
        assert d["bits"] == 4
        assert d["zero_point"] is True

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_awq_dict(None)  # type: ignore[arg-type]


class TestFormatQuantResult:
    """Tests for format_quant_result function."""

    def test_formats_result(self) -> None:
        """Test formatting result."""
        result = create_quant_result(14000, 4000)
        formatted = format_quant_result(result)
        assert "Original:" in formatted
        assert "Compression:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            format_quant_result(None)  # type: ignore[arg-type]


class TestListQuantMethods:
    """Tests for list_quant_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_quant_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_quant_methods()
        assert "gptq" in methods
        assert "awq" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_quant_methods()
        assert methods == sorted(methods)


class TestValidateQuantMethod:
    """Tests for validate_quant_method function."""

    def test_valid_gptq(self) -> None:
        """Test validation of gptq method."""
        assert validate_quant_method("gptq") is True

    def test_valid_awq(self) -> None:
        """Test validation of awq method."""
        assert validate_quant_method("awq") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_quant_method("invalid") is False


class TestGetQuantMethod:
    """Tests for get_quant_method function."""

    def test_get_gptq(self) -> None:
        """Test getting GPTQ method."""
        assert get_quant_method("gptq") == QuantMethod.GPTQ

    def test_get_awq(self) -> None:
        """Test getting AWQ method."""
        assert get_quant_method("awq") == QuantMethod.AWQ

    def test_invalid_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid quant method"):
            get_quant_method("invalid")


class TestListCalibrationMethods:
    """Tests for list_calibration_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_calibration_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_calibration_methods()
        assert "minmax" in methods
        assert "entropy" in methods


class TestValidateCalibrationMethod:
    """Tests for validate_calibration_method function."""

    def test_valid_minmax(self) -> None:
        """Test validation of minmax method."""
        assert validate_calibration_method("minmax") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_calibration_method("invalid") is False


class TestGetCalibrationMethod:
    """Tests for get_calibration_method function."""

    def test_get_minmax(self) -> None:
        """Test getting MINMAX method."""
        assert get_calibration_method("minmax") == CalibrationMethod.MINMAX

    def test_invalid_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid calibration method"):
            get_calibration_method("invalid")


class TestListQuantGranularities:
    """Tests for list_quant_granularities function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        granularities = list_quant_granularities()
        assert isinstance(granularities, list)

    def test_contains_expected_values(self) -> None:
        """Test that list contains expected values."""
        granularities = list_quant_granularities()
        assert "per_tensor" in granularities


class TestValidateQuantGranularity:
    """Tests for validate_quant_granularity function."""

    def test_valid_per_tensor(self) -> None:
        """Test validation of per_tensor granularity."""
        assert validate_quant_granularity("per_tensor") is True

    def test_invalid_granularity(self) -> None:
        """Test validation of invalid granularity."""
        assert validate_quant_granularity("invalid") is False


class TestGetQuantGranularity:
    """Tests for get_quant_granularity function."""

    def test_get_per_tensor(self) -> None:
        """Test getting PER_TENSOR granularity."""
        assert get_quant_granularity("per_tensor") == QuantGranularity.PER_TENSOR

    def test_invalid_raises_error(self) -> None:
        """Test that invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="invalid granularity"):
            get_quant_granularity("invalid")


class TestGetRecommendedProfile:
    """Tests for get_recommended_profile function."""

    def test_7b_profile(self) -> None:
        """Test 7B model profile."""
        profile = get_recommended_profile("7b")
        assert profile.method == QuantMethod.GPTQ
        assert profile.bits == 4

    def test_70b_profile(self) -> None:
        """Test 70B model profile."""
        profile = get_recommended_profile("70b")
        assert profile.group_size == 64

    def test_invalid_size_raises_error(self) -> None:
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized model size"):
            get_recommended_profile("invalid")
