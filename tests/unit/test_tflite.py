"""Tests for TensorFlow Lite conversion utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.tflite import (
    ConversionStats,
    DelegateConfig,
    OptimizationTarget,
    QuantizationConfig,
    TFLiteConvertConfig,
    TFLiteDelegate,
    TFLiteModelInfo,
    TFLiteQuantization,
    calculate_compression_ratio,
    create_conversion_stats,
    create_convert_config,
    create_delegate_config,
    create_model_info,
    create_quantization_config,
    estimate_tflite_size,
    format_conversion_stats,
    format_model_info,
    get_convert_config_dict,
    get_delegate_config_dict,
    get_delegate_type,
    get_optimization_target,
    get_quantization_type,
    get_recommended_config,
    get_recommended_delegate,
    list_delegate_types,
    list_optimization_targets,
    list_quantization_types,
    validate_convert_config,
    validate_delegate_config,
    validate_quantization_config,
)


class TestTFLiteQuantization:
    """Tests for TFLiteQuantization enum."""

    def test_none_value(self) -> None:
        """Test NONE value."""
        assert TFLiteQuantization.NONE.value == "none"

    def test_dynamic_value(self) -> None:
        """Test DYNAMIC value."""
        assert TFLiteQuantization.DYNAMIC.value == "dynamic"

    def test_full_integer_value(self) -> None:
        """Test FULL_INTEGER value."""
        assert TFLiteQuantization.FULL_INTEGER.value == "full_integer"

    def test_float16_value(self) -> None:
        """Test FLOAT16 value."""
        assert TFLiteQuantization.FLOAT16.value == "float16"


class TestTFLiteDelegate:
    """Tests for TFLiteDelegate enum."""

    def test_none_value(self) -> None:
        """Test NONE value."""
        assert TFLiteDelegate.NONE.value == "none"

    def test_gpu_value(self) -> None:
        """Test GPU value."""
        assert TFLiteDelegate.GPU.value == "gpu"

    def test_nnapi_value(self) -> None:
        """Test NNAPI value."""
        assert TFLiteDelegate.NNAPI.value == "nnapi"

    def test_xnnpack_value(self) -> None:
        """Test XNNPACK value."""
        assert TFLiteDelegate.XNNPACK.value == "xnnpack"

    def test_coreml_value(self) -> None:
        """Test COREML value."""
        assert TFLiteDelegate.COREML.value == "coreml"

    def test_hexagon_value(self) -> None:
        """Test HEXAGON value."""
        assert TFLiteDelegate.HEXAGON.value == "hexagon"


class TestOptimizationTarget:
    """Tests for OptimizationTarget enum."""

    def test_default_value(self) -> None:
        """Test DEFAULT value."""
        assert OptimizationTarget.DEFAULT.value == "default"

    def test_size_value(self) -> None:
        """Test SIZE value."""
        assert OptimizationTarget.SIZE.value == "size"

    def test_latency_value(self) -> None:
        """Test LATENCY value."""
        assert OptimizationTarget.LATENCY.value == "latency"


class TestTFLiteConvertConfig:
    """Tests for TFLiteConvertConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = TFLiteConvertConfig()
        assert config.quantization == TFLiteQuantization.NONE
        assert config.optimization_target == OptimizationTarget.DEFAULT
        assert config.supported_ops == frozenset({"TFLITE_BUILTINS"})
        assert config.allow_custom_ops is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = TFLiteConvertConfig(
            quantization=TFLiteQuantization.DYNAMIC,
            optimization_target=OptimizationTarget.SIZE,
            supported_ops=frozenset({"TFLITE_BUILTINS", "SELECT_TF_OPS"}),
            allow_custom_ops=True,
        )
        assert config.quantization == TFLiteQuantization.DYNAMIC
        assert config.optimization_target == OptimizationTarget.SIZE
        assert "SELECT_TF_OPS" in config.supported_ops
        assert config.allow_custom_ops is True

    def test_frozen(self) -> None:
        """Test that TFLiteConvertConfig is immutable."""
        config = TFLiteConvertConfig()
        with pytest.raises(AttributeError):
            config.quantization = TFLiteQuantization.DYNAMIC  # type: ignore[misc]


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = QuantizationConfig()
        assert config.representative_dataset_size == 100
        assert config.num_calibration_steps == 100
        assert config.output_type == "int8"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = QuantizationConfig(
            representative_dataset_size=200,
            num_calibration_steps=50,
            output_type="uint8",
        )
        assert config.representative_dataset_size == 200
        assert config.num_calibration_steps == 50
        assert config.output_type == "uint8"

    def test_frozen(self) -> None:
        """Test that QuantizationConfig is immutable."""
        config = QuantizationConfig()
        with pytest.raises(AttributeError):
            config.output_type = "float16"  # type: ignore[misc]


class TestDelegateConfig:
    """Tests for DelegateConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = DelegateConfig()
        assert config.delegate_type == TFLiteDelegate.NONE
        assert config.num_threads == 4
        assert config.enable_fallback is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = DelegateConfig(
            delegate_type=TFLiteDelegate.GPU,
            num_threads=8,
            enable_fallback=False,
        )
        assert config.delegate_type == TFLiteDelegate.GPU
        assert config.num_threads == 8
        assert config.enable_fallback is False

    def test_frozen(self) -> None:
        """Test that DelegateConfig is immutable."""
        config = DelegateConfig()
        with pytest.raises(AttributeError):
            config.num_threads = 8  # type: ignore[misc]


class TestTFLiteModelInfo:
    """Tests for TFLiteModelInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating model info."""
        info = TFLiteModelInfo(
            model_path="/models/model.tflite",
            input_details=[{"name": "input", "shape": [1, 224, 224, 3]}],
            output_details=[{"name": "output", "shape": [1, 1000]}],
            model_size_bytes=4_000_000,
        )
        assert info.model_path == "/models/model.tflite"
        assert info.input_details == [{"name": "input", "shape": [1, 224, 224, 3]}]
        assert info.output_details == [{"name": "output", "shape": [1, 1000]}]
        assert info.model_size_bytes == 4_000_000

    def test_with_empty_details(self) -> None:
        """Test model info with empty details lists."""
        info = TFLiteModelInfo(
            model_path="/models/bert.tflite",
            input_details=[],
            output_details=[],
            model_size_bytes=100_000_000,
        )
        assert len(info.input_details) == 0
        assert len(info.output_details) == 0

    def test_frozen(self) -> None:
        """Test that TFLiteModelInfo is immutable."""
        info = TFLiteModelInfo(
            model_path="/models/model.tflite",
            input_details=[],
            output_details=[],
            model_size_bytes=1000,
        )
        with pytest.raises(AttributeError):
            info.model_path = "/other/path.tflite"  # type: ignore[misc]


class TestConversionStats:
    """Tests for ConversionStats dataclass."""

    def test_creation(self) -> None:
        """Test creating stats."""
        stats = ConversionStats(
            conversion_time_seconds=10.5,
            original_size=100_000_000,
            converted_size=25_000_000,
            compression_ratio=4.0,
        )
        assert stats.conversion_time_seconds == pytest.approx(10.5)
        assert stats.original_size == 100_000_000
        assert stats.converted_size == 25_000_000
        assert stats.compression_ratio == pytest.approx(4.0)

    def test_with_no_compression(self) -> None:
        """Test stats with 1.0 compression ratio."""
        stats = ConversionStats(
            conversion_time_seconds=5.0,
            original_size=50_000_000,
            converted_size=50_000_000,
            compression_ratio=1.0,
        )
        assert stats.compression_ratio == pytest.approx(1.0)

    def test_frozen(self) -> None:
        """Test that ConversionStats is immutable."""
        stats = ConversionStats(
            conversion_time_seconds=10.5,
            original_size=100_000_000,
            converted_size=25_000_000,
            compression_ratio=4.0,
        )
        with pytest.raises(AttributeError):
            stats.compression_ratio = 5.0  # type: ignore[misc]


class TestValidateConvertConfig:
    """Tests for validate_convert_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = TFLiteConvertConfig()
        validate_convert_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_convert_config(None)  # type: ignore[arg-type]

    def test_empty_supported_ops_raises_error(self) -> None:
        """Test that empty supported_ops raises ValueError."""
        config = TFLiteConvertConfig(supported_ops=frozenset())
        with pytest.raises(ValueError, match="supported_ops cannot be empty"):
            validate_convert_config(config)


class TestValidateQuantizationConfig:
    """Tests for validate_quantization_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = QuantizationConfig()
        validate_quantization_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_quantization_config(None)  # type: ignore[arg-type]

    def test_zero_representative_dataset_size_raises_error(self) -> None:
        """Test that zero representative_dataset_size raises ValueError."""
        config = QuantizationConfig(representative_dataset_size=0)
        with pytest.raises(
            ValueError, match="representative_dataset_size must be positive"
        ):
            validate_quantization_config(config)

    def test_negative_representative_dataset_size_raises_error(self) -> None:
        """Test that negative representative_dataset_size raises ValueError."""
        config = QuantizationConfig(representative_dataset_size=-10)
        with pytest.raises(
            ValueError, match="representative_dataset_size must be positive"
        ):
            validate_quantization_config(config)

    def test_zero_num_calibration_steps_raises_error(self) -> None:
        """Test that zero num_calibration_steps raises ValueError."""
        config = QuantizationConfig(num_calibration_steps=0)
        with pytest.raises(ValueError, match="num_calibration_steps must be positive"):
            validate_quantization_config(config)

    def test_negative_num_calibration_steps_raises_error(self) -> None:
        """Test that negative num_calibration_steps raises ValueError."""
        config = QuantizationConfig(num_calibration_steps=-1)
        with pytest.raises(ValueError, match="num_calibration_steps must be positive"):
            validate_quantization_config(config)

    def test_invalid_output_type_raises_error(self) -> None:
        """Test that invalid output_type raises ValueError."""
        config = QuantizationConfig(output_type="invalid")
        with pytest.raises(ValueError, match="output_type must be one of"):
            validate_quantization_config(config)

    @pytest.mark.parametrize("output_type", ["int8", "uint8", "float16"])
    def test_valid_output_types(self, output_type: str) -> None:
        """Test all valid output types."""
        config = QuantizationConfig(output_type=output_type)
        validate_quantization_config(config)  # Should not raise


class TestValidateDelegateConfig:
    """Tests for validate_delegate_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = DelegateConfig()
        validate_delegate_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_delegate_config(None)  # type: ignore[arg-type]

    def test_zero_num_threads_raises_error(self) -> None:
        """Test that zero num_threads raises ValueError."""
        config = DelegateConfig(num_threads=0)
        with pytest.raises(ValueError, match="num_threads must be positive"):
            validate_delegate_config(config)

    def test_negative_num_threads_raises_error(self) -> None:
        """Test that negative num_threads raises ValueError."""
        config = DelegateConfig(num_threads=-1)
        with pytest.raises(ValueError, match="num_threads must be positive"):
            validate_delegate_config(config)


class TestCreateConvertConfig:
    """Tests for create_convert_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_convert_config()
        assert config.quantization == TFLiteQuantization.NONE
        assert config.optimization_target == OptimizationTarget.DEFAULT

    def test_creates_config_with_string_quantization(self) -> None:
        """Test creating config with string quantization."""
        config = create_convert_config(quantization="dynamic")
        assert config.quantization == TFLiteQuantization.DYNAMIC

    def test_creates_config_with_enum_quantization(self) -> None:
        """Test creating config with enum quantization."""
        config = create_convert_config(quantization=TFLiteQuantization.FULL_INTEGER)
        assert config.quantization == TFLiteQuantization.FULL_INTEGER

    def test_creates_config_with_string_optimization_target(self) -> None:
        """Test creating config with string optimization target."""
        config = create_convert_config(optimization_target="size")
        assert config.optimization_target == OptimizationTarget.SIZE

    def test_creates_config_with_enum_optimization_target(self) -> None:
        """Test creating config with enum optimization target."""
        config = create_convert_config(optimization_target=OptimizationTarget.LATENCY)
        assert config.optimization_target == OptimizationTarget.LATENCY

    def test_creates_config_with_custom_supported_ops(self) -> None:
        """Test creating config with custom supported ops."""
        ops = frozenset({"TFLITE_BUILTINS", "SELECT_TF_OPS"})
        config = create_convert_config(supported_ops=ops)
        assert "SELECT_TF_OPS" in config.supported_ops

    def test_creates_config_with_allow_custom_ops(self) -> None:
        """Test creating config with allow_custom_ops enabled."""
        config = create_convert_config(allow_custom_ops=True)
        assert config.allow_custom_ops is True

    def test_empty_supported_ops_raises_error(self) -> None:
        """Test that empty supported_ops raises ValueError."""
        with pytest.raises(ValueError, match="supported_ops cannot be empty"):
            create_convert_config(supported_ops=frozenset())

    @pytest.mark.parametrize("quant", ["none", "dynamic", "full_integer", "float16"])
    def test_all_valid_string_quantizations(self, quant: str) -> None:
        """Test all valid string quantization types."""
        config = create_convert_config(quantization=quant)
        assert config.quantization.value == quant


class TestCreateQuantizationConfig:
    """Tests for create_quantization_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_quantization_config()
        assert config.representative_dataset_size == 100
        assert config.num_calibration_steps == 100
        assert config.output_type == "int8"

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_quantization_config(
            representative_dataset_size=200,
            num_calibration_steps=50,
            output_type="uint8",
        )
        assert config.representative_dataset_size == 200
        assert config.num_calibration_steps == 50
        assert config.output_type == "uint8"

    def test_zero_representative_dataset_size_raises_error(self) -> None:
        """Test that zero representative_dataset_size raises ValueError."""
        with pytest.raises(
            ValueError, match="representative_dataset_size must be positive"
        ):
            create_quantization_config(representative_dataset_size=0)

    def test_zero_num_calibration_steps_raises_error(self) -> None:
        """Test that zero num_calibration_steps raises ValueError."""
        with pytest.raises(ValueError, match="num_calibration_steps must be positive"):
            create_quantization_config(num_calibration_steps=0)

    def test_invalid_output_type_raises_error(self) -> None:
        """Test that invalid output_type raises ValueError."""
        with pytest.raises(ValueError, match="output_type must be one of"):
            create_quantization_config(output_type="invalid")


class TestCreateDelegateConfig:
    """Tests for create_delegate_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_delegate_config()
        assert config.delegate_type == TFLiteDelegate.NONE
        assert config.num_threads == 4
        assert config.enable_fallback is True

    def test_creates_config_with_string_delegate(self) -> None:
        """Test creating config with string delegate type."""
        config = create_delegate_config(delegate_type="xnnpack")
        assert config.delegate_type == TFLiteDelegate.XNNPACK

    def test_creates_config_with_enum_delegate(self) -> None:
        """Test creating config with enum delegate type."""
        config = create_delegate_config(delegate_type=TFLiteDelegate.GPU)
        assert config.delegate_type == TFLiteDelegate.GPU

    def test_creates_config_with_custom_threads(self) -> None:
        """Test creating config with custom thread count."""
        config = create_delegate_config(num_threads=8)
        assert config.num_threads == 8

    def test_creates_config_with_fallback_disabled(self) -> None:
        """Test creating config with fallback disabled."""
        config = create_delegate_config(enable_fallback=False)
        assert config.enable_fallback is False

    def test_zero_num_threads_raises_error(self) -> None:
        """Test that zero num_threads raises ValueError."""
        with pytest.raises(ValueError, match="num_threads must be positive"):
            create_delegate_config(num_threads=0)

    @pytest.mark.parametrize(
        "delegate", ["none", "gpu", "nnapi", "xnnpack", "coreml", "hexagon"]
    )
    def test_all_valid_string_delegates(self, delegate: str) -> None:
        """Test all valid string delegate types."""
        config = create_delegate_config(delegate_type=delegate)
        assert config.delegate_type.value == delegate


class TestListQuantizationTypes:
    """Tests for list_quantization_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_quantization_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_quantization_types()
        assert "none" in types
        assert "dynamic" in types
        assert "full_integer" in types
        assert "float16" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_quantization_types()
        assert types == sorted(types)

    def test_length_matches_enum(self) -> None:
        """Test that length matches enum members."""
        types = list_quantization_types()
        assert len(types) == len(TFLiteQuantization)


class TestListDelegateTypes:
    """Tests for list_delegate_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_delegate_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_delegate_types()
        assert "none" in types
        assert "gpu" in types
        assert "nnapi" in types
        assert "xnnpack" in types
        assert "coreml" in types
        assert "hexagon" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_delegate_types()
        assert types == sorted(types)

    def test_length_matches_enum(self) -> None:
        """Test that length matches enum members."""
        types = list_delegate_types()
        assert len(types) == len(TFLiteDelegate)


class TestListOptimizationTargets:
    """Tests for list_optimization_targets function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        targets = list_optimization_targets()
        assert isinstance(targets, list)

    def test_contains_expected_targets(self) -> None:
        """Test that list contains expected targets."""
        targets = list_optimization_targets()
        assert "default" in targets
        assert "size" in targets
        assert "latency" in targets

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        targets = list_optimization_targets()
        assert targets == sorted(targets)

    def test_length_matches_enum(self) -> None:
        """Test that length matches enum members."""
        targets = list_optimization_targets()
        assert len(targets) == len(OptimizationTarget)


class TestGetQuantizationType:
    """Tests for get_quantization_type function."""

    def test_get_none(self) -> None:
        """Test getting NONE type."""
        assert get_quantization_type("none") == TFLiteQuantization.NONE

    def test_get_dynamic(self) -> None:
        """Test getting DYNAMIC type."""
        assert get_quantization_type("dynamic") == TFLiteQuantization.DYNAMIC

    def test_get_full_integer(self) -> None:
        """Test getting FULL_INTEGER type."""
        assert get_quantization_type("full_integer") == TFLiteQuantization.FULL_INTEGER

    def test_get_float16(self) -> None:
        """Test getting FLOAT16 type."""
        assert get_quantization_type("float16") == TFLiteQuantization.FLOAT16

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid quantization type: invalid"):
            get_quantization_type("invalid")


class TestGetDelegateType:
    """Tests for get_delegate_type function."""

    def test_get_none(self) -> None:
        """Test getting NONE type."""
        assert get_delegate_type("none") == TFLiteDelegate.NONE

    def test_get_gpu(self) -> None:
        """Test getting GPU type."""
        assert get_delegate_type("gpu") == TFLiteDelegate.GPU

    def test_get_nnapi(self) -> None:
        """Test getting NNAPI type."""
        assert get_delegate_type("nnapi") == TFLiteDelegate.NNAPI

    def test_get_xnnpack(self) -> None:
        """Test getting XNNPACK type."""
        assert get_delegate_type("xnnpack") == TFLiteDelegate.XNNPACK

    def test_get_coreml(self) -> None:
        """Test getting COREML type."""
        assert get_delegate_type("coreml") == TFLiteDelegate.COREML

    def test_get_hexagon(self) -> None:
        """Test getting HEXAGON type."""
        assert get_delegate_type("hexagon") == TFLiteDelegate.HEXAGON

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid delegate type: invalid"):
            get_delegate_type("invalid")


class TestGetOptimizationTarget:
    """Tests for get_optimization_target function."""

    def test_get_default(self) -> None:
        """Test getting DEFAULT target."""
        assert get_optimization_target("default") == OptimizationTarget.DEFAULT

    def test_get_size(self) -> None:
        """Test getting SIZE target."""
        assert get_optimization_target("size") == OptimizationTarget.SIZE

    def test_get_latency(self) -> None:
        """Test getting LATENCY target."""
        assert get_optimization_target("latency") == OptimizationTarget.LATENCY

    def test_invalid_raises_error(self) -> None:
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError, match="invalid optimization target: invalid"):
            get_optimization_target("invalid")


class TestEstimateTfliteSize:
    """Tests for estimate_tflite_size function."""

    def test_estimates_fp32_size(self) -> None:
        """Test estimating FP32 size (no quantization)."""
        size = estimate_tflite_size(1_000_000, TFLiteQuantization.NONE)
        # 1M params * 4 bytes * 1.05 overhead = 4.2M bytes
        assert size > 4_000_000
        assert size < 4_500_000

    def test_estimates_fp16_size(self) -> None:
        """Test estimating FP16 size."""
        size = estimate_tflite_size(1_000_000, TFLiteQuantization.FLOAT16)
        # 1M params * 2 bytes * 1.05 overhead = 2.1M bytes
        assert size > 2_000_000
        assert size < 2_500_000

    def test_estimates_dynamic_size(self) -> None:
        """Test estimating dynamic quantization size."""
        size = estimate_tflite_size(1_000_000, TFLiteQuantization.DYNAMIC)
        # 1M params * 1.5 bytes * 1.05 overhead = 1.575M bytes
        assert size > 1_500_000
        assert size < 2_000_000

    def test_estimates_full_integer_size(self) -> None:
        """Test estimating full integer quantization size."""
        size = estimate_tflite_size(1_000_000, TFLiteQuantization.FULL_INTEGER)
        # 1M params * 1 byte * 1.05 overhead = 1.05M bytes
        assert size > 1_000_000
        assert size < 1_200_000

    def test_quantization_reduces_size(self) -> None:
        """Test that quantization reduces estimated size."""
        size_none = estimate_tflite_size(1_000_000, TFLiteQuantization.NONE)
        size_dynamic = estimate_tflite_size(1_000_000, TFLiteQuantization.DYNAMIC)
        size_full_int = estimate_tflite_size(1_000_000, TFLiteQuantization.FULL_INTEGER)
        assert size_full_int < size_dynamic < size_none

    def test_custom_overhead_factor(self) -> None:
        """Test custom overhead factor."""
        size_default = estimate_tflite_size(1_000_000, overhead_factor=1.05)
        size_higher = estimate_tflite_size(1_000_000, overhead_factor=1.5)
        assert size_higher > size_default

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_tflite_size(0)

    def test_negative_params_raises_error(self) -> None:
        """Test that negative params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_tflite_size(-100)

    def test_overhead_below_one_raises_error(self) -> None:
        """Test that overhead_factor < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"overhead_factor must be >= 1\.0"):
            estimate_tflite_size(1_000_000, overhead_factor=0.9)


class TestCalculateCompressionRatio:
    """Tests for calculate_compression_ratio function."""

    def test_calculates_4x_ratio(self) -> None:
        """Test calculating 4x compression ratio."""
        ratio = calculate_compression_ratio(100_000_000, 25_000_000)
        assert ratio == pytest.approx(4.0)

    def test_calculates_1x_ratio(self) -> None:
        """Test calculating 1x compression ratio (no compression)."""
        ratio = calculate_compression_ratio(50_000_000, 50_000_000)
        assert ratio == pytest.approx(1.0)

    def test_calculates_8x_ratio(self) -> None:
        """Test calculating 8x compression ratio."""
        ratio = calculate_compression_ratio(100_000_000, 12_500_000)
        assert ratio == pytest.approx(8.0)

    def test_zero_original_size_raises_error(self) -> None:
        """Test that zero original_size raises ValueError."""
        with pytest.raises(ValueError, match="original_size must be positive"):
            calculate_compression_ratio(0, 100)

    def test_negative_original_size_raises_error(self) -> None:
        """Test that negative original_size raises ValueError."""
        with pytest.raises(ValueError, match="original_size must be positive"):
            calculate_compression_ratio(-100, 50)

    def test_zero_converted_size_raises_error(self) -> None:
        """Test that zero converted_size raises ValueError."""
        with pytest.raises(ValueError, match="converted_size must be positive"):
            calculate_compression_ratio(100, 0)

    def test_negative_converted_size_raises_error(self) -> None:
        """Test that negative converted_size raises ValueError."""
        with pytest.raises(ValueError, match="converted_size must be positive"):
            calculate_compression_ratio(100, -50)


class TestCreateConversionStats:
    """Tests for create_conversion_stats function."""

    def test_creates_stats(self) -> None:
        """Test creating stats."""
        stats = create_conversion_stats(10.5, 100_000_000, 25_000_000)
        assert stats.conversion_time_seconds == pytest.approx(10.5)
        assert stats.original_size == 100_000_000
        assert stats.converted_size == 25_000_000
        assert stats.compression_ratio == pytest.approx(4.0)

    def test_negative_conversion_time_raises_error(self) -> None:
        """Test that negative conversion time raises ValueError."""
        with pytest.raises(
            ValueError, match="conversion_time_seconds cannot be negative"
        ):
            create_conversion_stats(-1.0, 100, 50)

    def test_zero_conversion_time_allowed(self) -> None:
        """Test that zero conversion time is allowed."""
        stats = create_conversion_stats(0.0, 100, 50)
        assert stats.conversion_time_seconds == pytest.approx(0.0)

    def test_zero_original_size_raises_error(self) -> None:
        """Test that zero original_size raises ValueError."""
        with pytest.raises(ValueError, match="original_size must be positive"):
            create_conversion_stats(1.0, 0, 50)

    def test_zero_converted_size_raises_error(self) -> None:
        """Test that zero converted_size raises ValueError."""
        with pytest.raises(ValueError, match="converted_size must be positive"):
            create_conversion_stats(1.0, 100, 0)


class TestCreateModelInfo:
    """Tests for create_model_info function."""

    def test_creates_model_info(self) -> None:
        """Test creating model info."""
        info = create_model_info(
            "/models/model.tflite",
            [{"name": "input", "shape": [1, 224, 224, 3]}],
            [{"name": "output", "shape": [1, 1000]}],
            4_000_000,
        )
        assert info.model_path == "/models/model.tflite"
        assert len(info.input_details) == 1
        assert len(info.output_details) == 1
        assert info.model_size_bytes == 4_000_000

    def test_creates_model_info_with_empty_details(self) -> None:
        """Test creating model info with empty details."""
        info = create_model_info("/model.tflite", [], [], 1000)
        assert len(info.input_details) == 0
        assert len(info.output_details) == 0

    def test_empty_path_raises_error(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="model_path cannot be empty"):
            create_model_info("", [], [], 100)

    def test_zero_model_size_raises_error(self) -> None:
        """Test that zero model_size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="model_size_bytes must be positive"):
            create_model_info("/model.tflite", [], [], 0)

    def test_negative_model_size_raises_error(self) -> None:
        """Test that negative model_size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="model_size_bytes must be positive"):
            create_model_info("/model.tflite", [], [], -1)


class TestFormatModelInfo:
    """Tests for format_model_info function."""

    def test_formats_info(self) -> None:
        """Test formatting model info."""
        info = TFLiteModelInfo(
            model_path="/models/model.tflite",
            input_details=[{"name": "input"}],
            output_details=[{"name": "output"}],
            model_size_bytes=4_000_000,
        )
        formatted = format_model_info(info)
        assert "Path:" in formatted
        assert "/models/model.tflite" in formatted
        assert "Size:" in formatted
        assert "MB" in formatted
        assert "Inputs:" in formatted
        assert "Outputs:" in formatted

    def test_formats_size_in_megabytes(self) -> None:
        """Test that size is formatted in MB."""
        info = TFLiteModelInfo(
            model_path="/model.tflite",
            input_details=[],
            output_details=[],
            model_size_bytes=1_048_576,  # 1 MB
        )
        formatted = format_model_info(info)
        assert "1.00 MB" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="info cannot be None"):
            format_model_info(None)  # type: ignore[arg-type]


class TestFormatConversionStats:
    """Tests for format_conversion_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting conversion stats."""
        stats = ConversionStats(
            conversion_time_seconds=10.5,
            original_size=100_000_000,
            converted_size=25_000_000,
            compression_ratio=4.0,
        )
        formatted = format_conversion_stats(stats)
        assert "Conversion time:" in formatted
        assert "10.50s" in formatted
        assert "Original size:" in formatted
        assert "Converted size:" in formatted
        assert "Compression:" in formatted
        assert "4.00x" in formatted

    def test_formats_size_in_megabytes(self) -> None:
        """Test that sizes are formatted in MB."""
        stats = ConversionStats(
            conversion_time_seconds=1.0,
            original_size=104_857_600,  # 100 MB
            converted_size=26_214_400,  # 25 MB
            compression_ratio=4.0,
        )
        formatted = format_conversion_stats(stats)
        assert "100.00 MB" in formatted
        assert "25.00 MB" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_conversion_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedConfig:
    """Tests for get_recommended_config function."""

    def test_mobile_size_config(self) -> None:
        """Test recommended config for mobile with size priority."""
        config = get_recommended_config("mobile", "size")
        assert config.quantization == TFLiteQuantization.FULL_INTEGER
        assert config.optimization_target == OptimizationTarget.SIZE

    def test_mobile_latency_config(self) -> None:
        """Test recommended config for mobile with latency priority."""
        config = get_recommended_config("mobile", "latency")
        assert config.quantization == TFLiteQuantization.DYNAMIC
        assert config.optimization_target == OptimizationTarget.LATENCY

    def test_mobile_balanced_config(self) -> None:
        """Test recommended config for mobile with balanced priority."""
        config = get_recommended_config("mobile", "balanced")
        assert config.quantization == TFLiteQuantization.DYNAMIC
        assert config.optimization_target == OptimizationTarget.DEFAULT

    def test_edge_size_config(self) -> None:
        """Test recommended config for edge with size priority."""
        config = get_recommended_config("edge", "size")
        assert config.quantization == TFLiteQuantization.FULL_INTEGER
        assert config.optimization_target == OptimizationTarget.SIZE

    def test_edge_latency_config(self) -> None:
        """Test recommended config for edge with latency priority."""
        config = get_recommended_config("edge", "latency")
        assert config.quantization == TFLiteQuantization.DYNAMIC
        assert config.optimization_target == OptimizationTarget.LATENCY

    def test_edge_balanced_config(self) -> None:
        """Test recommended config for edge with balanced priority."""
        config = get_recommended_config("edge", "balanced")
        assert config.quantization == TFLiteQuantization.FLOAT16
        assert config.optimization_target == OptimizationTarget.DEFAULT

    def test_server_size_config(self) -> None:
        """Test recommended config for server with size priority."""
        config = get_recommended_config("server", "size")
        assert config.quantization == TFLiteQuantization.FLOAT16
        assert config.optimization_target == OptimizationTarget.SIZE

    def test_server_latency_config(self) -> None:
        """Test recommended config for server with latency priority."""
        config = get_recommended_config("server", "latency")
        assert config.quantization == TFLiteQuantization.NONE
        assert config.optimization_target == OptimizationTarget.LATENCY

    def test_server_balanced_config(self) -> None:
        """Test recommended config for server with balanced priority."""
        config = get_recommended_config("server", "balanced")
        assert config.quantization == TFLiteQuantization.FLOAT16
        assert config.optimization_target == OptimizationTarget.DEFAULT

    def test_invalid_device_raises_error(self) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="invalid target device: unknown"):
            get_recommended_config("unknown")

    def test_invalid_optimize_for_raises_error(self) -> None:
        """Test that invalid optimize_for raises ValueError."""
        with pytest.raises(ValueError, match="invalid optimization priority: unknown"):
            get_recommended_config("mobile", "unknown")

    @pytest.mark.parametrize("device", ["mobile", "edge", "server"])
    def test_all_valid_devices(self, device: str) -> None:
        """Test all valid devices."""
        config = get_recommended_config(device)
        assert config is not None

    @pytest.mark.parametrize("priority", ["size", "latency", "balanced"])
    def test_all_valid_priorities(self, priority: str) -> None:
        """Test all valid priorities."""
        config = get_recommended_config("mobile", priority)
        assert config is not None


class TestGetRecommendedDelegate:
    """Tests for get_recommended_delegate function."""

    def test_mobile_config(self) -> None:
        """Test recommended delegate for mobile."""
        config = get_recommended_delegate("mobile")
        assert config.delegate_type == TFLiteDelegate.GPU
        assert config.num_threads == 4
        assert config.enable_fallback is True

    def test_edge_config(self) -> None:
        """Test recommended delegate for edge."""
        config = get_recommended_delegate("edge")
        assert config.delegate_type == TFLiteDelegate.XNNPACK
        assert config.num_threads == 2
        assert config.enable_fallback is True

    def test_server_config(self) -> None:
        """Test recommended delegate for server."""
        config = get_recommended_delegate("server")
        assert config.delegate_type == TFLiteDelegate.XNNPACK
        assert config.num_threads == 8
        assert config.enable_fallback is False

    def test_invalid_device_raises_error(self) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="invalid target device: unknown"):
            get_recommended_delegate("unknown")

    @pytest.mark.parametrize("device", ["mobile", "edge", "server"])
    def test_all_valid_devices(self, device: str) -> None:
        """Test all valid devices."""
        config = get_recommended_delegate(device)
        assert config is not None


class TestGetConvertConfigDict:
    """Tests for get_convert_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config to dict."""
        config = create_convert_config(quantization="dynamic")
        d = get_convert_config_dict(config)
        assert d["quantization"] == "dynamic"
        assert d["optimization_target"] == "default"
        assert d["allow_custom_ops"] is False
        assert isinstance(d["supported_ops"], list)

    def test_supported_ops_converted_to_list(self) -> None:
        """Test that supported_ops frozenset is converted to list."""
        config = create_convert_config(
            supported_ops=frozenset({"TFLITE_BUILTINS", "SELECT_TF_OPS"})
        )
        d = get_convert_config_dict(config)
        assert isinstance(d["supported_ops"], list)
        assert "TFLITE_BUILTINS" in d["supported_ops"]
        assert "SELECT_TF_OPS" in d["supported_ops"]

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_convert_config_dict(None)  # type: ignore[arg-type]


class TestGetDelegateConfigDict:
    """Tests for get_delegate_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config to dict."""
        config = create_delegate_config(delegate_type="xnnpack")
        d = get_delegate_config_dict(config)
        assert d["delegate_type"] == "xnnpack"
        assert d["num_threads"] == 4
        assert d["enable_fallback"] is True

    def test_with_custom_values(self) -> None:
        """Test with custom config values."""
        config = create_delegate_config(
            delegate_type=TFLiteDelegate.GPU,
            num_threads=8,
            enable_fallback=False,
        )
        d = get_delegate_config_dict(config)
        assert d["delegate_type"] == "gpu"
        assert d["num_threads"] == 8
        assert d["enable_fallback"] is False

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_delegate_config_dict(None)  # type: ignore[arg-type]
