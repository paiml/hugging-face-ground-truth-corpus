"""Tests for deployment optimization functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.deployment.optimization import (
    VALID_QUANTIZATION_TYPES,
    OptimizationResult,
    QuantizationConfig,
    QuantizationType,
    calculate_compression_ratio,
    estimate_model_size,
    get_model_loading_kwargs,
    get_optimization_result,
    get_quantization_config,
    list_quantization_types,
)


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_int8_value(self) -> None:
        """Test INT8 enum value."""
        assert QuantizationType.INT8.value == "int8"

    def test_int4_value(self) -> None:
        """Test INT4 enum value."""
        assert QuantizationType.INT4.value == "int4"

    def test_fp16_value(self) -> None:
        """Test FP16 enum value."""
        assert QuantizationType.FP16.value == "fp16"

    def test_bf16_value(self) -> None:
        """Test BF16 enum value."""
        assert QuantizationType.BF16.value == "bf16"

    def test_none_value(self) -> None:
        """Test NONE enum value."""
        assert QuantizationType.NONE.value == "none"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_QUANTIZATION_TYPES."""
        for qtype in QuantizationType:
            assert qtype.value in VALID_QUANTIZATION_TYPES


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating QuantizationConfig instance."""
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            load_in_8bit=True,
            load_in_4bit=False,
            torch_dtype="float16",
            device_map="auto",
        )
        assert config.quantization_type == QuantizationType.INT8
        assert config.load_in_8bit is True
        assert config.load_in_4bit is False
        assert config.torch_dtype == "float16"
        assert config.device_map == "auto"

    def test_frozen(self) -> None:
        """Test that QuantizationConfig is immutable."""
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            load_in_8bit=True,
            load_in_4bit=False,
            torch_dtype="float16",
            device_map="auto",
        )
        with pytest.raises(AttributeError):
            config.load_in_8bit = False  # type: ignore[misc]


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating OptimizationResult instance."""
        result = OptimizationResult(
            original_size_mb=1000.0,
            optimized_size_mb=250.0,
            compression_ratio=4.0,
            quantization_type=QuantizationType.INT8,
        )
        assert result.original_size_mb == 1000.0
        assert result.optimized_size_mb == 250.0
        assert result.compression_ratio == 4.0
        assert result.quantization_type == QuantizationType.INT8

    def test_frozen(self) -> None:
        """Test that OptimizationResult is immutable."""
        result = OptimizationResult(
            original_size_mb=1000.0,
            optimized_size_mb=250.0,
            compression_ratio=4.0,
            quantization_type=QuantizationType.INT8,
        )
        with pytest.raises(AttributeError):
            result.compression_ratio = 2.0  # type: ignore[misc]


class TestGetQuantizationConfig:
    """Tests for get_quantization_config function."""

    def test_int8_config(self) -> None:
        """Test INT8 quantization config."""
        config = get_quantization_config("int8")
        assert config.quantization_type == QuantizationType.INT8
        assert config.load_in_8bit is True
        assert config.load_in_4bit is False
        assert config.torch_dtype == "float16"

    def test_int4_config(self) -> None:
        """Test INT4 quantization config."""
        config = get_quantization_config("int4")
        assert config.quantization_type == QuantizationType.INT4
        assert config.load_in_8bit is False
        assert config.load_in_4bit is True
        assert config.torch_dtype == "float16"

    def test_fp16_config(self) -> None:
        """Test FP16 config."""
        config = get_quantization_config("fp16")
        assert config.quantization_type == QuantizationType.FP16
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.torch_dtype == "float16"

    def test_bf16_config(self) -> None:
        """Test BF16 config."""
        config = get_quantization_config("bf16")
        assert config.quantization_type == QuantizationType.BF16
        assert config.torch_dtype == "bfloat16"

    def test_none_config(self) -> None:
        """Test no quantization config."""
        config = get_quantization_config("none")
        assert config.quantization_type == QuantizationType.NONE
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.torch_dtype == "float32"

    def test_custom_device_map(self) -> None:
        """Test custom device map."""
        config = get_quantization_config("int8", device_map="cuda:0")
        assert config.device_map == "cuda:0"

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid quantization type raises ValueError."""
        with pytest.raises(ValueError, match="quantization_type must be one of"):
            get_quantization_config("invalid")


class TestEstimateModelSize:
    """Tests for estimate_model_size function."""

    def test_float32_size(self) -> None:
        """Test size estimation for float32 (no quantization)."""
        # 1B params * 4 bytes / (1024*1024) = ~3814.7 MB
        size = estimate_model_size(1_000_000_000, "none")
        assert abs(size - 3814.697265625) < 0.001

    def test_fp16_size(self) -> None:
        """Test size estimation for fp16."""
        # 1B params * 2 bytes / (1024*1024) = ~1907.35 MB
        size = estimate_model_size(1_000_000_000, "fp16")
        assert abs(size - 1907.3486328125) < 0.001

    def test_int8_size(self) -> None:
        """Test size estimation for int8."""
        # 1B params * 1 byte / (1024*1024) = ~953.67 MB
        size = estimate_model_size(1_000_000_000, "int8")
        assert abs(size - 953.6743164062) < 0.001

    def test_int4_size(self) -> None:
        """Test size estimation for int4."""
        # 1B params * 0.5 bytes / (1024*1024) = ~476.84 MB
        size = estimate_model_size(1_000_000_000, "int4")
        assert abs(size - 476.8371582031) < 0.001

    def test_bf16_size(self) -> None:
        """Test size estimation for bf16."""
        size = estimate_model_size(1_000_000_000, "bf16")
        assert abs(size - 1907.3486328125) < 0.001

    def test_zero_parameters_raises_error(self) -> None:
        """Test that zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_model_size(0)

    def test_negative_parameters_raises_error(self) -> None:
        """Test that negative parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_model_size(-100)

    def test_invalid_quantization_type_raises_error(self) -> None:
        """Test that invalid quantization type raises ValueError."""
        with pytest.raises(ValueError, match="quantization_type must be one of"):
            estimate_model_size(1_000_000, "invalid")

    @given(st.integers(min_value=1, max_value=100_000_000_000))
    @settings(max_examples=50)
    def test_size_always_positive(self, num_params: int) -> None:
        """Test that estimated size is always positive."""
        size = estimate_model_size(num_params, "none")
        assert size > 0


class TestCalculateCompressionRatio:
    """Tests for calculate_compression_ratio function."""

    def test_4x_compression(self) -> None:
        """Test 4x compression ratio."""
        ratio = calculate_compression_ratio(1000.0, 250.0)
        assert ratio == 4.0

    def test_2x_compression(self) -> None:
        """Test 2x compression ratio."""
        ratio = calculate_compression_ratio(100.0, 50.0)
        assert ratio == 2.0

    def test_no_compression(self) -> None:
        """Test 1x compression (no change)."""
        ratio = calculate_compression_ratio(100.0, 100.0)
        assert ratio == 1.0

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original size raises ValueError."""
        with pytest.raises(ValueError, match="original_size must be positive"):
            calculate_compression_ratio(0, 100.0)

    def test_negative_original_raises_error(self) -> None:
        """Test that negative original size raises ValueError."""
        with pytest.raises(ValueError, match="original_size must be positive"):
            calculate_compression_ratio(-100.0, 100.0)

    def test_zero_optimized_raises_error(self) -> None:
        """Test that zero optimized size raises ValueError."""
        with pytest.raises(ValueError, match="optimized_size must be positive"):
            calculate_compression_ratio(100.0, 0)

    def test_negative_optimized_raises_error(self) -> None:
        """Test that negative optimized size raises ValueError."""
        with pytest.raises(ValueError, match="optimized_size must be positive"):
            calculate_compression_ratio(100.0, -50.0)


class TestGetOptimizationResult:
    """Tests for get_optimization_result function."""

    def test_int8_optimization(self) -> None:
        """Test optimization result for int8."""
        result = get_optimization_result(1_000_000_000, "int8")
        # float32 (4 bytes) to int8 (1 byte) = 4x compression
        assert result.compression_ratio == 4.0
        assert result.quantization_type == QuantizationType.INT8

    def test_int4_optimization(self) -> None:
        """Test optimization result for int4."""
        result = get_optimization_result(1_000_000_000, "int4")
        # float32 (4 bytes) to int4 (0.5 bytes) = 8x compression
        assert result.compression_ratio == 8.0
        assert result.quantization_type == QuantizationType.INT4

    def test_no_optimization(self) -> None:
        """Test optimization result with no quantization."""
        result = get_optimization_result(1_000_000_000, "none")
        assert result.compression_ratio == 1.0


class TestGetModelLoadingKwargs:
    """Tests for get_model_loading_kwargs function."""

    def test_int8_kwargs(self) -> None:
        """Test kwargs for int8 loading."""
        config = get_quantization_config("int8")
        kwargs = get_model_loading_kwargs(config)
        assert kwargs["load_in_8bit"] is True
        assert kwargs["device_map"] == "auto"
        assert "torch_dtype" not in kwargs

    def test_int4_kwargs(self) -> None:
        """Test kwargs for int4 loading."""
        config = get_quantization_config("int4")
        kwargs = get_model_loading_kwargs(config)
        assert kwargs["load_in_4bit"] is True
        assert kwargs["device_map"] == "auto"
        assert "torch_dtype" not in kwargs

    def test_fp16_kwargs(self) -> None:
        """Test kwargs for fp16 loading."""
        config = get_quantization_config("fp16")
        kwargs = get_model_loading_kwargs(config)
        assert "load_in_8bit" not in kwargs
        assert "load_in_4bit" not in kwargs
        assert kwargs["torch_dtype"] == "float16"
        assert kwargs["device_map"] == "auto"

    def test_none_kwargs(self) -> None:
        """Test kwargs for no quantization."""
        config = get_quantization_config("none")
        kwargs = get_model_loading_kwargs(config)
        assert kwargs["torch_dtype"] == "float32"


class TestListQuantizationTypes:
    """Tests for list_quantization_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_quantization_types()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_quantization_types()
        assert result == sorted(result)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        result = list_quantization_types()
        assert "int8" in result
        assert "int4" in result
        assert "fp16" in result
        assert "bf16" in result
        assert "none" in result

    def test_correct_count(self) -> None:
        """Test correct number of types."""
        result = list_quantization_types()
        assert len(result) == 5
