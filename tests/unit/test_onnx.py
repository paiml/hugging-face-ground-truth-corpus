"""Tests for ONNX export and runtime utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.onnx import (
    VALID_EXECUTION_PROVIDERS,
    VALID_ONNX_OPSET_VERSIONS,
    VALID_OPSET_VERSIONS,
    VALID_OPTIMIZATION_LEVELS,
    ExecutionProvider,
    ExportStats,
    ONNXExportConfig,
    ONNXModelInfo,
    ONNXOpset,
    ONNXOptimizeConfig,
    OptimizationLevel,
    RuntimeConfig,
    create_export_config,
    create_optimize_config,
    create_runtime_config,
    estimate_onnx_model_size,
    format_export_stats,
    format_model_info,
    get_execution_provider,
    get_export_config_dict,
    get_opset_version,
    get_optimization_level,
    get_recommended_opset,
    get_runtime_session_options,
    list_execution_providers,
    list_opset_versions,
    list_optimization_levels,
    validate_export_config,
    validate_runtime_config,
)


class TestONNXOpset:
    """Tests for ONNXOpset enum."""

    def test_opset_11_value(self) -> None:
        """Test OPSET_11 value."""
        assert ONNXOpset.OPSET_11.value == 11

    def test_opset_12_value(self) -> None:
        """Test OPSET_12 value."""
        assert ONNXOpset.OPSET_12.value == 12

    def test_opset_13_value(self) -> None:
        """Test OPSET_13 value."""
        assert ONNXOpset.OPSET_13.value == 13

    def test_opset_14_value(self) -> None:
        """Test OPSET_14 value."""
        assert ONNXOpset.OPSET_14.value == 14

    def test_opset_15_value(self) -> None:
        """Test OPSET_15 value."""
        assert ONNXOpset.OPSET_15.value == 15

    def test_opset_16_value(self) -> None:
        """Test OPSET_16 value."""
        assert ONNXOpset.OPSET_16.value == 16

    def test_opset_17_value(self) -> None:
        """Test OPSET_17 value."""
        assert ONNXOpset.OPSET_17.value == 17


class TestOptimizationLevel:
    """Tests for OptimizationLevel enum."""

    def test_disable_value(self) -> None:
        """Test DISABLE value."""
        assert OptimizationLevel.DISABLE.value == "disable"

    def test_basic_value(self) -> None:
        """Test BASIC value."""
        assert OptimizationLevel.BASIC.value == "basic"

    def test_extended_value(self) -> None:
        """Test EXTENDED value."""
        assert OptimizationLevel.EXTENDED.value == "extended"

    def test_all_value(self) -> None:
        """Test ALL value."""
        assert OptimizationLevel.ALL.value == "all"


class TestExecutionProvider:
    """Tests for ExecutionProvider enum."""

    def test_cpu_value(self) -> None:
        """Test CPU value."""
        assert ExecutionProvider.CPU.value == "cpu"

    def test_cuda_value(self) -> None:
        """Test CUDA value."""
        assert ExecutionProvider.CUDA.value == "cuda"

    def test_tensorrt_value(self) -> None:
        """Test TENSORRT value."""
        assert ExecutionProvider.TENSORRT.value == "tensorrt"

    def test_coreml_value(self) -> None:
        """Test COREML value."""
        assert ExecutionProvider.COREML.value == "coreml"

    def test_directml_value(self) -> None:
        """Test DIRECTML value."""
        assert ExecutionProvider.DIRECTML.value == "directml"


class TestValidConstants:
    """Tests for module-level valid constants."""

    def test_valid_opset_versions_contains_expected(self) -> None:
        """Test VALID_OPSET_VERSIONS contains expected versions."""
        assert 11 in VALID_OPSET_VERSIONS
        assert 14 in VALID_OPSET_VERSIONS
        assert 17 in VALID_OPSET_VERSIONS

    def test_valid_onnx_opset_versions_alias(self) -> None:
        """Test VALID_ONNX_OPSET_VERSIONS is alias for VALID_OPSET_VERSIONS."""
        assert VALID_ONNX_OPSET_VERSIONS == VALID_OPSET_VERSIONS

    def test_valid_optimization_levels_contains_expected(self) -> None:
        """Test VALID_OPTIMIZATION_LEVELS contains expected levels."""
        assert "disable" in VALID_OPTIMIZATION_LEVELS
        assert "basic" in VALID_OPTIMIZATION_LEVELS
        assert "extended" in VALID_OPTIMIZATION_LEVELS
        assert "all" in VALID_OPTIMIZATION_LEVELS

    def test_valid_execution_providers_contains_expected(self) -> None:
        """Test VALID_EXECUTION_PROVIDERS contains expected providers."""
        assert "cpu" in VALID_EXECUTION_PROVIDERS
        assert "cuda" in VALID_EXECUTION_PROVIDERS
        assert "tensorrt" in VALID_EXECUTION_PROVIDERS


class TestONNXExportConfig:
    """Tests for ONNXExportConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ONNXExportConfig()
        assert config.opset_version == 14
        assert config.do_constant_folding is True
        assert config.dynamic_axes is None
        assert config.input_names == ("input_ids",)
        assert config.output_names == ("logits",)

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ONNXExportConfig(
            opset_version=17,
            do_constant_folding=False,
            dynamic_axes={"input_ids": {0: "batch"}},
            input_names=("input_ids", "attention_mask"),
            output_names=("output",),
        )
        assert config.opset_version == 17
        assert config.do_constant_folding is False
        assert config.dynamic_axes == {"input_ids": {0: "batch"}}
        assert config.input_names == ("input_ids", "attention_mask")
        assert config.output_names == ("output",)

    def test_frozen(self) -> None:
        """Test that ONNXExportConfig is immutable."""
        config = ONNXExportConfig()
        with pytest.raises(AttributeError):
            config.opset_version = 17  # type: ignore[misc]


class TestONNXOptimizeConfig:
    """Tests for ONNXOptimizeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ONNXOptimizeConfig()
        assert config.level == OptimizationLevel.BASIC
        assert config.enable_gelu_fusion is True
        assert config.enable_layer_norm_fusion is True
        assert config.enable_attention_fusion is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ONNXOptimizeConfig(
            level=OptimizationLevel.ALL,
            enable_gelu_fusion=False,
            enable_layer_norm_fusion=False,
            enable_attention_fusion=True,
        )
        assert config.level == OptimizationLevel.ALL
        assert config.enable_gelu_fusion is False
        assert config.enable_layer_norm_fusion is False
        assert config.enable_attention_fusion is True

    def test_frozen(self) -> None:
        """Test that ONNXOptimizeConfig is immutable."""
        config = ONNXOptimizeConfig()
        with pytest.raises(AttributeError):
            config.level = OptimizationLevel.ALL  # type: ignore[misc]


class TestRuntimeConfig:
    """Tests for RuntimeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = RuntimeConfig()
        assert config.provider == ExecutionProvider.CPU
        assert config.num_threads == 0
        assert config.graph_optimization_level == OptimizationLevel.BASIC
        assert config.enable_profiling is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = RuntimeConfig(
            provider=ExecutionProvider.CUDA,
            num_threads=8,
            graph_optimization_level=OptimizationLevel.ALL,
            enable_profiling=True,
        )
        assert config.provider == ExecutionProvider.CUDA
        assert config.num_threads == 8
        assert config.graph_optimization_level == OptimizationLevel.ALL
        assert config.enable_profiling is True

    def test_frozen(self) -> None:
        """Test that RuntimeConfig is immutable."""
        config = RuntimeConfig()
        with pytest.raises(AttributeError):
            config.num_threads = 4  # type: ignore[misc]


class TestONNXModelInfo:
    """Tests for ONNXModelInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating model info."""
        info = ONNXModelInfo(
            model_path="/path/to/model.onnx",
            opset_version=14,
            input_shapes={"input_ids": [1, 512]},
            output_shapes={"logits": [1, 512, 768]},
            num_parameters=110_000_000,
        )
        assert info.model_path == "/path/to/model.onnx"
        assert info.opset_version == 14
        assert info.input_shapes == {"input_ids": [1, 512]}
        assert info.output_shapes == {"logits": [1, 512, 768]}
        assert info.num_parameters == 110_000_000

    def test_frozen(self) -> None:
        """Test that ONNXModelInfo is immutable."""
        info = ONNXModelInfo(
            model_path="/path/to/model.onnx",
            opset_version=14,
            input_shapes={"input_ids": [1, 512]},
            output_shapes={"logits": [1, 512, 768]},
            num_parameters=110_000_000,
        )
        with pytest.raises(AttributeError):
            info.model_path = "/other/path.onnx"  # type: ignore[misc]


class TestExportStats:
    """Tests for ExportStats dataclass."""

    def test_creation(self) -> None:
        """Test creating export stats."""
        stats = ExportStats(
            export_time_seconds=5.5,
            model_size_bytes=440_000_000,
            num_nodes=1500,
            num_initializers=200,
        )
        assert stats.export_time_seconds == pytest.approx(5.5)
        assert stats.model_size_bytes == 440_000_000
        assert stats.num_nodes == 1500
        assert stats.num_initializers == 200

    def test_frozen(self) -> None:
        """Test that ExportStats is immutable."""
        stats = ExportStats(
            export_time_seconds=5.5,
            model_size_bytes=440_000_000,
            num_nodes=1500,
            num_initializers=200,
        )
        with pytest.raises(AttributeError):
            stats.num_nodes = 2000  # type: ignore[misc]


class TestValidateExportConfig:
    """Tests for validate_export_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = ONNXExportConfig(opset_version=14)
        validate_export_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_export_config(None)  # type: ignore[arg-type]

    def test_invalid_opset_version_raises_error(self) -> None:
        """Test that invalid opset version raises ValueError."""
        config = ONNXExportConfig(opset_version=10)
        with pytest.raises(ValueError, match="opset_version must be one of"):
            validate_export_config(config)

    def test_empty_input_names_raises_error(self) -> None:
        """Test that empty input_names raises ValueError."""
        config = ONNXExportConfig(input_names=())
        with pytest.raises(ValueError, match="input_names cannot be empty"):
            validate_export_config(config)

    def test_empty_output_names_raises_error(self) -> None:
        """Test that empty output_names raises ValueError."""
        config = ONNXExportConfig(output_names=())
        with pytest.raises(ValueError, match="output_names cannot be empty"):
            validate_export_config(config)

    @pytest.mark.parametrize("opset_version", [11, 12, 13, 14, 15, 16, 17])
    def test_all_valid_opset_versions(self, opset_version: int) -> None:
        """Test all valid opset versions."""
        config = ONNXExportConfig(opset_version=opset_version)
        validate_export_config(config)  # Should not raise


class TestValidateRuntimeConfig:
    """Tests for validate_runtime_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = RuntimeConfig(num_threads=4)
        validate_runtime_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_runtime_config(None)  # type: ignore[arg-type]

    def test_negative_num_threads_raises_error(self) -> None:
        """Test that negative num_threads raises ValueError."""
        config = RuntimeConfig(num_threads=-1)
        with pytest.raises(ValueError, match="num_threads cannot be negative"):
            validate_runtime_config(config)

    def test_zero_num_threads_valid(self) -> None:
        """Test that zero num_threads is valid (auto)."""
        config = RuntimeConfig(num_threads=0)
        validate_runtime_config(config)  # Should not raise


class TestCreateExportConfig:
    """Tests for create_export_config function."""

    def test_creates_config(self) -> None:
        """Test creating config with defaults."""
        config = create_export_config()
        assert config.opset_version == 14
        assert config.do_constant_folding is True

    def test_with_integer_opset(self) -> None:
        """Test with integer opset version."""
        config = create_export_config(opset_version=17)
        assert config.opset_version == 17

    def test_with_enum_opset(self) -> None:
        """Test with ONNXOpset enum."""
        config = create_export_config(opset_version=ONNXOpset.OPSET_17)
        assert config.opset_version == 17

    def test_with_list_input_names(self) -> None:
        """Test with list input names (converted to tuple)."""
        config = create_export_config(input_names=["input_ids", "attention_mask"])
        assert config.input_names == ("input_ids", "attention_mask")

    def test_with_list_output_names(self) -> None:
        """Test with list output names (converted to tuple)."""
        config = create_export_config(output_names=["logits", "hidden_states"])
        assert config.output_names == ("logits", "hidden_states")

    def test_with_dynamic_axes(self) -> None:
        """Test with dynamic axes."""
        dynamic_axes = {"input_ids": {0: "batch", 1: "seq_len"}}
        config = create_export_config(dynamic_axes=dynamic_axes)
        assert config.dynamic_axes == dynamic_axes

    def test_invalid_opset_raises_error(self) -> None:
        """Test that invalid opset raises ValueError."""
        with pytest.raises(ValueError, match="opset_version must be one of"):
            create_export_config(opset_version=10)

    def test_constant_folding_false(self) -> None:
        """Test with constant folding disabled."""
        config = create_export_config(do_constant_folding=False)
        assert config.do_constant_folding is False


class TestCreateOptimizeConfig:
    """Tests for create_optimize_config function."""

    def test_creates_config(self) -> None:
        """Test creating config with defaults."""
        config = create_optimize_config()
        assert config.level == OptimizationLevel.BASIC

    def test_with_string_level(self) -> None:
        """Test with string level."""
        config = create_optimize_config(level="extended")
        assert config.level == OptimizationLevel.EXTENDED

    def test_with_enum_level(self) -> None:
        """Test with OptimizationLevel enum."""
        config = create_optimize_config(level=OptimizationLevel.ALL)
        assert config.level == OptimizationLevel.ALL

    def test_with_fusion_flags(self) -> None:
        """Test with fusion flags."""
        config = create_optimize_config(
            enable_gelu_fusion=False,
            enable_layer_norm_fusion=False,
            enable_attention_fusion=True,
        )
        assert config.enable_gelu_fusion is False
        assert config.enable_layer_norm_fusion is False
        assert config.enable_attention_fusion is True

    def test_invalid_level_raises_error(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="invalid optimization level"):
            create_optimize_config(level="invalid")

    @pytest.mark.parametrize("level", ["disable", "basic", "extended", "all"])
    def test_all_valid_string_levels(self, level: str) -> None:
        """Test all valid string levels."""
        config = create_optimize_config(level=level)
        assert config.level.value == level


class TestCreateRuntimeConfig:
    """Tests for create_runtime_config function."""

    def test_creates_config(self) -> None:
        """Test creating config with defaults."""
        config = create_runtime_config()
        assert config.provider == ExecutionProvider.CPU
        assert config.num_threads == 0

    def test_with_string_provider(self) -> None:
        """Test with string provider."""
        config = create_runtime_config(provider="cuda")
        assert config.provider == ExecutionProvider.CUDA

    def test_with_enum_provider(self) -> None:
        """Test with ExecutionProvider enum."""
        config = create_runtime_config(provider=ExecutionProvider.TENSORRT)
        assert config.provider == ExecutionProvider.TENSORRT

    def test_with_string_graph_optimization_level(self) -> None:
        """Test with string graph optimization level."""
        config = create_runtime_config(graph_optimization_level="all")
        assert config.graph_optimization_level == OptimizationLevel.ALL

    def test_with_enum_graph_optimization_level(self) -> None:
        """Test with enum graph optimization level."""
        config = create_runtime_config(
            graph_optimization_level=OptimizationLevel.EXTENDED
        )
        assert config.graph_optimization_level == OptimizationLevel.EXTENDED

    def test_with_custom_threads(self) -> None:
        """Test with custom thread count."""
        config = create_runtime_config(num_threads=8)
        assert config.num_threads == 8

    def test_with_profiling_enabled(self) -> None:
        """Test with profiling enabled."""
        config = create_runtime_config(enable_profiling=True)
        assert config.enable_profiling is True

    def test_negative_threads_raises_error(self) -> None:
        """Test that negative threads raises ValueError."""
        with pytest.raises(ValueError, match="num_threads cannot be negative"):
            create_runtime_config(num_threads=-1)

    @pytest.mark.parametrize(
        "provider", ["cpu", "cuda", "tensorrt", "coreml", "directml"]
    )
    def test_all_valid_string_providers(self, provider: str) -> None:
        """Test all valid string providers."""
        config = create_runtime_config(provider=provider)
        assert config.provider.value == provider


class TestListOpsetVersions:
    """Tests for list_opset_versions function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        versions = list_opset_versions()
        assert isinstance(versions, list)

    def test_contains_expected_versions(self) -> None:
        """Test that list contains expected versions."""
        versions = list_opset_versions()
        assert 14 in versions
        assert 17 in versions

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        versions = list_opset_versions()
        assert versions == sorted(versions)

    def test_length_matches_enum(self) -> None:
        """Test that length matches enum members."""
        versions = list_opset_versions()
        assert len(versions) == len(ONNXOpset)


class TestListOptimizationLevels:
    """Tests for list_optimization_levels function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        levels = list_optimization_levels()
        assert isinstance(levels, list)

    def test_contains_expected_levels(self) -> None:
        """Test that list contains expected levels."""
        levels = list_optimization_levels()
        assert "basic" in levels
        assert "all" in levels

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        levels = list_optimization_levels()
        assert levels == sorted(levels)


class TestListExecutionProviders:
    """Tests for list_execution_providers function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        providers = list_execution_providers()
        assert isinstance(providers, list)

    def test_contains_expected_providers(self) -> None:
        """Test that list contains expected providers."""
        providers = list_execution_providers()
        assert "cpu" in providers
        assert "cuda" in providers

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        providers = list_execution_providers()
        assert providers == sorted(providers)


class TestGetOpsetVersion:
    """Tests for get_opset_version function."""

    def test_get_opset_14(self) -> None:
        """Test getting OPSET_14."""
        assert get_opset_version(14) == ONNXOpset.OPSET_14

    def test_get_opset_17(self) -> None:
        """Test getting OPSET_17."""
        assert get_opset_version(17) == ONNXOpset.OPSET_17

    def test_get_from_string(self) -> None:
        """Test getting from string."""
        assert get_opset_version("14") == ONNXOpset.OPSET_14

    def test_invalid_raises_error(self) -> None:
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="invalid opset version: 10"):
            get_opset_version(10)

    @pytest.mark.parametrize("version", [11, 12, 13, 14, 15, 16, 17])
    def test_all_valid_versions(self, version: int) -> None:
        """Test all valid versions."""
        result = get_opset_version(version)
        assert result.value == version


class TestGetOptimizationLevel:
    """Tests for get_optimization_level function."""

    def test_get_basic(self) -> None:
        """Test getting BASIC level."""
        assert get_optimization_level("basic") == OptimizationLevel.BASIC

    def test_get_all(self) -> None:
        """Test getting ALL level."""
        assert get_optimization_level("all") == OptimizationLevel.ALL

    def test_get_disable(self) -> None:
        """Test getting DISABLE level."""
        assert get_optimization_level("disable") == OptimizationLevel.DISABLE

    def test_get_extended(self) -> None:
        """Test getting EXTENDED level."""
        assert get_optimization_level("extended") == OptimizationLevel.EXTENDED

    def test_invalid_raises_error(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="invalid optimization level: invalid"):
            get_optimization_level("invalid")


class TestGetExecutionProvider:
    """Tests for get_execution_provider function."""

    def test_get_cpu(self) -> None:
        """Test getting CPU provider."""
        assert get_execution_provider("cpu") == ExecutionProvider.CPU

    def test_get_cuda(self) -> None:
        """Test getting CUDA provider."""
        assert get_execution_provider("cuda") == ExecutionProvider.CUDA

    def test_get_tensorrt(self) -> None:
        """Test getting TENSORRT provider."""
        assert get_execution_provider("tensorrt") == ExecutionProvider.TENSORRT

    def test_get_coreml(self) -> None:
        """Test getting COREML provider."""
        assert get_execution_provider("coreml") == ExecutionProvider.COREML

    def test_get_directml(self) -> None:
        """Test getting DIRECTML provider."""
        assert get_execution_provider("directml") == ExecutionProvider.DIRECTML

    def test_invalid_raises_error(self) -> None:
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="invalid execution provider: invalid"):
            get_execution_provider("invalid")


class TestEstimateOnnxModelSize:
    """Tests for estimate_onnx_model_size function."""

    def test_estimates_fp32_size(self) -> None:
        """Test estimating FP32 size."""
        size = estimate_onnx_model_size(1_000_000, precision="fp32")
        # 1M params * 4 bytes * 1.1 overhead = 4.4M bytes
        assert size > 4_000_000
        assert size < 5_000_000

    def test_estimates_fp16_size(self) -> None:
        """Test estimating FP16 size."""
        size = estimate_onnx_model_size(1_000_000, precision="fp16")
        # 1M params * 2 bytes * 1.1 overhead = 2.2M bytes
        assert size > 2_000_000
        assert size < 3_000_000

    def test_estimates_int8_size(self) -> None:
        """Test estimating INT8 size."""
        size = estimate_onnx_model_size(1_000_000, precision="int8")
        # 1M params * 1 byte * 1.1 overhead = 1.1M bytes
        assert size > 1_000_000
        assert size < 2_000_000

    def test_fp16_smaller_than_fp32(self) -> None:
        """Test that FP16 is smaller than FP32."""
        size_fp32 = estimate_onnx_model_size(1_000_000, precision="fp32")
        size_fp16 = estimate_onnx_model_size(1_000_000, precision="fp16")
        assert size_fp16 < size_fp32

    def test_int8_smaller_than_fp16(self) -> None:
        """Test that INT8 is smaller than FP16."""
        size_fp16 = estimate_onnx_model_size(1_000_000, precision="fp16")
        size_int8 = estimate_onnx_model_size(1_000_000, precision="int8")
        assert size_int8 < size_fp16

    def test_custom_overhead_factor(self) -> None:
        """Test custom overhead factor."""
        size_default = estimate_onnx_model_size(1_000_000, overhead_factor=1.1)
        size_higher = estimate_onnx_model_size(1_000_000, overhead_factor=1.5)
        assert size_higher > size_default

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_onnx_model_size(0)

    def test_negative_params_raises_error(self) -> None:
        """Test that negative params raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_onnx_model_size(-100)

    def test_invalid_precision_raises_error(self) -> None:
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="precision must be one of"):
            estimate_onnx_model_size(1_000_000, precision="invalid")

    def test_overhead_below_one_raises_error(self) -> None:
        """Test that overhead_factor < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"overhead_factor must be >= 1\.0"):
            estimate_onnx_model_size(1_000_000, overhead_factor=0.9)


class TestGetRecommendedOpset:
    """Tests for get_recommended_opset function."""

    def test_bert_gets_opset_14(self) -> None:
        """Test BERT gets opset 14."""
        assert get_recommended_opset("bert") == ONNXOpset.OPSET_14

    def test_gpt2_gets_opset_14(self) -> None:
        """Test GPT2 gets opset 14."""
        assert get_recommended_opset("gpt2") == ONNXOpset.OPSET_14

    def test_t5_gets_opset_14(self) -> None:
        """Test T5 gets opset 14."""
        assert get_recommended_opset("t5") == ONNXOpset.OPSET_14

    def test_llama_gets_opset_17(self) -> None:
        """Test LLaMA gets opset 17."""
        assert get_recommended_opset("llama") == ONNXOpset.OPSET_17

    def test_mistral_gets_opset_17(self) -> None:
        """Test Mistral gets opset 17."""
        assert get_recommended_opset("mistral") == ONNXOpset.OPSET_17

    def test_falcon_gets_opset_17(self) -> None:
        """Test Falcon gets opset 17."""
        assert get_recommended_opset("falcon") == ONNXOpset.OPSET_17

    def test_unknown_model_gets_opset_14(self) -> None:
        """Test unknown model defaults to opset 14."""
        assert get_recommended_opset("unknown-model") == ONNXOpset.OPSET_14

    def test_case_insensitive(self) -> None:
        """Test that model type is case insensitive."""
        assert get_recommended_opset("BERT") == ONNXOpset.OPSET_14
        assert get_recommended_opset("LLaMA") == ONNXOpset.OPSET_17

    def test_strips_whitespace(self) -> None:
        """Test that model type strips whitespace."""
        assert get_recommended_opset("  bert  ") == ONNXOpset.OPSET_14

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_opset("")

    @pytest.mark.parametrize("model_type", ["roberta", "bart", "mbart", "distilbert"])
    def test_medium_opset_models(self, model_type: str) -> None:
        """Test medium opset models get opset 14."""
        assert get_recommended_opset(model_type) == ONNXOpset.OPSET_14

    @pytest.mark.parametrize("model_type", ["phi", "qwen", "gemma"])
    def test_high_opset_models(self, model_type: str) -> None:
        """Test high opset models get opset 17."""
        assert get_recommended_opset(model_type) == ONNXOpset.OPSET_17


class TestGetExportConfigDict:
    """Tests for get_export_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_export_config(opset_version=14)
        d = get_export_config_dict(config)
        assert d["opset_version"] == 14
        assert d["do_constant_folding"] is True
        assert d["input_names"] == ["input_ids"]
        assert d["output_names"] == ["logits"]

    def test_with_dynamic_axes(self) -> None:
        """Test with dynamic axes."""
        dynamic_axes = {"input_ids": {0: "batch"}}
        config = create_export_config(dynamic_axes=dynamic_axes)
        d = get_export_config_dict(config)
        assert d["dynamic_axes"] == dynamic_axes

    def test_without_dynamic_axes(self) -> None:
        """Test without dynamic axes (None)."""
        config = create_export_config()
        d = get_export_config_dict(config)
        assert "dynamic_axes" not in d

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_export_config_dict(None)  # type: ignore[arg-type]

    def test_input_names_converted_to_list(self) -> None:
        """Test that input_names tuple is converted to list."""
        config = create_export_config(input_names=("a", "b"))
        d = get_export_config_dict(config)
        assert d["input_names"] == ["a", "b"]
        assert isinstance(d["input_names"], list)


class TestGetRuntimeSessionOptions:
    """Tests for get_runtime_session_options function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_runtime_config(num_threads=4, enable_profiling=True)
        opts = get_runtime_session_options(config)
        assert opts["intra_op_num_threads"] == 4
        assert opts["enable_profiling"] is True

    def test_optimization_level_mapping(self) -> None:
        """Test optimization level mapping."""
        # Test DISABLE maps to 0
        config = create_runtime_config(graph_optimization_level="disable")
        opts = get_runtime_session_options(config)
        assert opts["graph_optimization_level"] == 0

        # Test BASIC maps to 1
        config = create_runtime_config(graph_optimization_level="basic")
        opts = get_runtime_session_options(config)
        assert opts["graph_optimization_level"] == 1

        # Test EXTENDED maps to 2
        config = create_runtime_config(graph_optimization_level="extended")
        opts = get_runtime_session_options(config)
        assert opts["graph_optimization_level"] == 2

        # Test ALL maps to 99
        config = create_runtime_config(graph_optimization_level="all")
        opts = get_runtime_session_options(config)
        assert opts["graph_optimization_level"] == 99

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_runtime_session_options(None)  # type: ignore[arg-type]


class TestFormatModelInfo:
    """Tests for format_model_info function."""

    def test_formats_info(self) -> None:
        """Test formatting model info."""
        info = ONNXModelInfo(
            model_path="/path/to/model.onnx",
            opset_version=14,
            input_shapes={"input_ids": [1, 512]},
            output_shapes={"logits": [1, 512, 768]},
            num_parameters=110_000_000,
        )
        formatted = format_model_info(info)
        assert "Model: /path/to/model.onnx" in formatted
        assert "Opset: 14" in formatted
        assert "Parameters: 110,000,000" in formatted
        assert "Inputs:" in formatted
        assert "input_ids: [1, 512]" in formatted
        assert "Outputs:" in formatted
        assert "logits: [1, 512, 768]" in formatted

    def test_multiple_inputs_outputs(self) -> None:
        """Test with multiple inputs and outputs."""
        info = ONNXModelInfo(
            model_path="/model.onnx",
            opset_version=14,
            input_shapes={"input_ids": [1, 512], "attention_mask": [1, 512]},
            output_shapes={"logits": [1, 512, 768], "hidden_states": [1, 512, 768]},
            num_parameters=100,
        )
        formatted = format_model_info(info)
        assert "input_ids:" in formatted
        assert "attention_mask:" in formatted
        assert "logits:" in formatted
        assert "hidden_states:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="info cannot be None"):
            format_model_info(None)  # type: ignore[arg-type]


class TestFormatExportStats:
    """Tests for format_export_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting export stats."""
        stats = ExportStats(
            export_time_seconds=5.5,
            model_size_bytes=440_000_000,
            num_nodes=1500,
            num_initializers=200,
        )
        formatted = format_export_stats(stats)
        assert "Export Time: 5.50s" in formatted
        assert "Model Size:" in formatted
        assert "MB" in formatted
        assert "Nodes: 1,500" in formatted
        assert "Initializers: 200" in formatted

    def test_size_in_megabytes(self) -> None:
        """Test that size is formatted in MB."""
        stats = ExportStats(
            export_time_seconds=1.0,
            model_size_bytes=1_048_576,  # 1 MB
            num_nodes=100,
            num_initializers=10,
        )
        formatted = format_export_stats(stats)
        assert "1.0 MB" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_export_stats(None)  # type: ignore[arg-type]
