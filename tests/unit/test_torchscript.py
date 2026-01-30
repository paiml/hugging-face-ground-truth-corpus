"""Tests for TorchScript model compilation utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.torchscript import (
    CompilationStats,
    FreezeMode,
    MobileConfig,
    OptimizationConfig,
    OptimizeFor,
    ScriptMode,
    TorchScriptConfig,
    TorchScriptInfo,
    TraceConfig,
    check_scriptable,
    create_compilation_stats,
    create_mobile_config,
    create_optimization_config,
    create_torchscript_config,
    create_torchscript_info,
    create_trace_config,
    estimate_script_size,
    format_compilation_stats,
    format_torchscript_info,
    get_freeze_mode,
    get_mobile_config_dict,
    get_optimization_config_dict,
    get_optimize_for,
    get_recommended_config,
    get_script_mode,
    get_torchscript_config_dict,
    get_trace_config_dict,
    list_freeze_modes,
    list_optimize_for_options,
    list_script_modes,
    validate_mobile_config,
    validate_optimization_config,
    validate_torchscript_config,
    validate_trace_config,
)


class TestScriptMode:
    """Tests for ScriptMode enum."""

    def test_trace_value(self) -> None:
        """Test TRACE value."""
        assert ScriptMode.TRACE.value == "trace"

    def test_script_value(self) -> None:
        """Test SCRIPT value."""
        assert ScriptMode.SCRIPT.value == "script"

    def test_hybrid_value(self) -> None:
        """Test HYBRID value."""
        assert ScriptMode.HYBRID.value == "hybrid"


class TestOptimizeFor:
    """Tests for OptimizeFor enum."""

    def test_inference_value(self) -> None:
        """Test INFERENCE value."""
        assert OptimizeFor.INFERENCE.value == "inference"

    def test_mobile_value(self) -> None:
        """Test MOBILE value."""
        assert OptimizeFor.MOBILE.value == "mobile"

    def test_training_value(self) -> None:
        """Test TRAINING value."""
        assert OptimizeFor.TRAINING.value == "training"


class TestFreezeMode:
    """Tests for FreezeMode enum."""

    def test_none_value(self) -> None:
        """Test NONE value."""
        assert FreezeMode.NONE.value == "none"

    def test_preserve_parameters_value(self) -> None:
        """Test PRESERVE_PARAMETERS value."""
        assert FreezeMode.PRESERVE_PARAMETERS.value == "preserve_parameters"

    def test_full_value(self) -> None:
        """Test FULL value."""
        assert FreezeMode.FULL.value == "full"


class TestTorchScriptConfig:
    """Tests for TorchScriptConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = TorchScriptConfig()
        assert config.mode == ScriptMode.TRACE
        assert config.strict is True
        assert config.optimize_for == OptimizeFor.INFERENCE
        assert config.check_trace is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = TorchScriptConfig(
            mode=ScriptMode.SCRIPT,
            strict=False,
            optimize_for=OptimizeFor.MOBILE,
            check_trace=False,
        )
        assert config.mode == ScriptMode.SCRIPT
        assert config.strict is False
        assert config.optimize_for == OptimizeFor.MOBILE
        assert config.check_trace is False

    def test_frozen(self) -> None:
        """Test that TorchScriptConfig is immutable."""
        config = TorchScriptConfig()
        with pytest.raises(AttributeError):
            config.strict = False  # type: ignore[misc]


class TestTraceConfig:
    """Tests for TraceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = TraceConfig()
        assert config.example_inputs == ""
        assert config.check_inputs is True
        assert config.check_tolerance == pytest.approx(1e-5)

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = TraceConfig(
            example_inputs="tensor([1, 2, 3])",
            check_inputs=False,
            check_tolerance=1e-4,
        )
        assert config.example_inputs == "tensor([1, 2, 3])"
        assert config.check_inputs is False
        assert config.check_tolerance == pytest.approx(1e-4)

    def test_frozen(self) -> None:
        """Test that TraceConfig is immutable."""
        config = TraceConfig()
        with pytest.raises(AttributeError):
            config.check_inputs = False  # type: ignore[misc]


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = OptimizationConfig()
        assert config.freeze_mode == FreezeMode.NONE
        assert config.fuse_operations is True
        assert config.inline_functions is True
        assert config.remove_dropout is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = OptimizationConfig(
            freeze_mode=FreezeMode.FULL,
            fuse_operations=False,
            inline_functions=False,
            remove_dropout=False,
        )
        assert config.freeze_mode == FreezeMode.FULL
        assert config.fuse_operations is False
        assert config.inline_functions is False
        assert config.remove_dropout is False

    def test_frozen(self) -> None:
        """Test that OptimizationConfig is immutable."""
        config = OptimizationConfig()
        with pytest.raises(AttributeError):
            config.fuse_operations = False  # type: ignore[misc]


class TestMobileConfig:
    """Tests for MobileConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = MobileConfig()
        assert config.optimize_for_mobile is False
        assert config.backend == "cpu"
        assert config.preserve_dtype is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = MobileConfig(
            optimize_for_mobile=True,
            backend="vulkan",
            preserve_dtype=False,
        )
        assert config.optimize_for_mobile is True
        assert config.backend == "vulkan"
        assert config.preserve_dtype is False

    def test_frozen(self) -> None:
        """Test that MobileConfig is immutable."""
        config = MobileConfig()
        with pytest.raises(AttributeError):
            config.backend = "metal"  # type: ignore[misc]


class TestTorchScriptInfo:
    """Tests for TorchScriptInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating info."""
        info = TorchScriptInfo(
            model_path="/models/model.pt",
            script_mode=ScriptMode.TRACE,
            graph_size=1500,
            num_parameters=7_000_000_000,
        )
        assert info.model_path == "/models/model.pt"
        assert info.script_mode == ScriptMode.TRACE
        assert info.graph_size == 1500
        assert info.num_parameters == 7_000_000_000

    def test_frozen(self) -> None:
        """Test that TorchScriptInfo is immutable."""
        info = TorchScriptInfo(
            model_path="/models/model.pt",
            script_mode=ScriptMode.TRACE,
            graph_size=1500,
            num_parameters=7_000_000_000,
        )
        with pytest.raises(AttributeError):
            info.graph_size = 2000  # type: ignore[misc]


class TestCompilationStats:
    """Tests for CompilationStats dataclass."""

    def test_creation(self) -> None:
        """Test creating stats."""
        stats = CompilationStats(
            compile_time_seconds=15.5,
            graph_nodes=1200,
            fused_ops=350,
            model_size_bytes=4_000_000_000,
        )
        assert stats.compile_time_seconds == pytest.approx(15.5)
        assert stats.graph_nodes == 1200
        assert stats.fused_ops == 350
        assert stats.model_size_bytes == 4_000_000_000

    def test_frozen(self) -> None:
        """Test that CompilationStats is immutable."""
        stats = CompilationStats(
            compile_time_seconds=15.5,
            graph_nodes=1200,
            fused_ops=350,
            model_size_bytes=4_000_000_000,
        )
        with pytest.raises(AttributeError):
            stats.fused_ops = 400  # type: ignore[misc]


class TestValidateTorchScriptConfig:
    """Tests for validate_torchscript_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = TorchScriptConfig(mode=ScriptMode.TRACE)
        validate_torchscript_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_torchscript_config(None)  # type: ignore[arg-type]


class TestValidateTraceConfig:
    """Tests for validate_trace_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = TraceConfig(example_inputs="tensor([1])")
        validate_trace_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_trace_config(None)  # type: ignore[arg-type]

    def test_negative_tolerance_raises_error(self) -> None:
        """Test that negative tolerance raises ValueError."""
        config = TraceConfig(check_tolerance=-1.0)
        with pytest.raises(ValueError, match="check_tolerance must be positive"):
            validate_trace_config(config)

    def test_zero_tolerance_raises_error(self) -> None:
        """Test that zero tolerance raises ValueError."""
        config = TraceConfig(check_tolerance=0.0)
        with pytest.raises(ValueError, match="check_tolerance must be positive"):
            validate_trace_config(config)


class TestValidateOptimizationConfig:
    """Tests for validate_optimization_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = OptimizationConfig(freeze_mode=FreezeMode.FULL)
        validate_optimization_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_optimization_config(None)  # type: ignore[arg-type]


class TestValidateMobileConfig:
    """Tests for validate_mobile_config function."""

    def test_valid_cpu_config(self) -> None:
        """Test validating valid config with cpu backend."""
        config = MobileConfig(backend="cpu")
        validate_mobile_config(config)  # Should not raise

    def test_valid_vulkan_config(self) -> None:
        """Test validating valid config with vulkan backend."""
        config = MobileConfig(backend="vulkan")
        validate_mobile_config(config)  # Should not raise

    def test_valid_metal_config(self) -> None:
        """Test validating valid config with metal backend."""
        config = MobileConfig(backend="metal")
        validate_mobile_config(config)  # Should not raise

    def test_valid_nnapi_config(self) -> None:
        """Test validating valid config with nnapi backend."""
        config = MobileConfig(backend="nnapi")
        validate_mobile_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_mobile_config(None)  # type: ignore[arg-type]

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend raises ValueError."""
        config = MobileConfig(backend="invalid")
        with pytest.raises(ValueError, match="backend must be one of"):
            validate_mobile_config(config)


class TestCreateTorchScriptConfig:
    """Tests for create_torchscript_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_torchscript_config()
        assert config.mode == ScriptMode.TRACE
        assert config.strict is True
        assert config.optimize_for == OptimizeFor.INFERENCE

    def test_creates_config_with_string_mode(self) -> None:
        """Test creating config with string mode."""
        config = create_torchscript_config(mode="trace")
        assert config.mode == ScriptMode.TRACE

    def test_creates_config_with_enum_mode(self) -> None:
        """Test creating config with enum mode."""
        config = create_torchscript_config(mode=ScriptMode.SCRIPT)
        assert config.mode == ScriptMode.SCRIPT

    def test_creates_config_with_string_optimize_for(self) -> None:
        """Test creating config with string optimize_for."""
        config = create_torchscript_config(optimize_for="mobile")
        assert config.optimize_for == OptimizeFor.MOBILE

    def test_creates_config_with_enum_optimize_for(self) -> None:
        """Test creating config with enum optimize_for."""
        config = create_torchscript_config(optimize_for=OptimizeFor.TRAINING)
        assert config.optimize_for == OptimizeFor.TRAINING

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with all custom values."""
        config = create_torchscript_config(
            mode="script",
            strict=False,
            optimize_for="training",
            check_trace=False,
        )
        assert config.mode == ScriptMode.SCRIPT
        assert config.strict is False
        assert config.optimize_for == OptimizeFor.TRAINING
        assert config.check_trace is False


class TestCreateTraceConfig:
    """Tests for create_trace_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_trace_config()
        assert config.example_inputs == ""
        assert config.check_inputs is True
        assert config.check_tolerance == pytest.approx(1e-5)

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_trace_config(
            example_inputs="input tensor",
            check_inputs=False,
            check_tolerance=1e-4,
        )
        assert config.example_inputs == "input tensor"
        assert config.check_inputs is False
        assert config.check_tolerance == pytest.approx(1e-4)

    def test_negative_tolerance_raises_error(self) -> None:
        """Test that negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="check_tolerance must be positive"):
            create_trace_config(check_tolerance=-1.0)

    def test_zero_tolerance_raises_error(self) -> None:
        """Test that zero tolerance raises ValueError."""
        with pytest.raises(ValueError, match="check_tolerance must be positive"):
            create_trace_config(check_tolerance=0.0)


class TestCreateOptimizationConfig:
    """Tests for create_optimization_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_optimization_config()
        assert config.freeze_mode == FreezeMode.NONE
        assert config.fuse_operations is True
        assert config.inline_functions is True
        assert config.remove_dropout is True

    def test_creates_config_with_string_freeze_mode(self) -> None:
        """Test creating config with string freeze mode."""
        config = create_optimization_config(freeze_mode="full")
        assert config.freeze_mode == FreezeMode.FULL

    def test_creates_config_with_enum_freeze_mode(self) -> None:
        """Test creating config with enum freeze mode."""
        config = create_optimization_config(freeze_mode=FreezeMode.PRESERVE_PARAMETERS)
        assert config.freeze_mode == FreezeMode.PRESERVE_PARAMETERS

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with all custom values."""
        config = create_optimization_config(
            freeze_mode="preserve_parameters",
            fuse_operations=False,
            inline_functions=False,
            remove_dropout=False,
        )
        assert config.freeze_mode == FreezeMode.PRESERVE_PARAMETERS
        assert config.fuse_operations is False
        assert config.inline_functions is False
        assert config.remove_dropout is False


class TestCreateMobileConfig:
    """Tests for create_mobile_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_mobile_config()
        assert config.optimize_for_mobile is False
        assert config.backend == "cpu"
        assert config.preserve_dtype is True

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_mobile_config(
            optimize_for_mobile=True,
            backend="vulkan",
            preserve_dtype=False,
        )
        assert config.optimize_for_mobile is True
        assert config.backend == "vulkan"
        assert config.preserve_dtype is False

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            create_mobile_config(backend="invalid")

    @pytest.mark.parametrize("backend", ["cpu", "vulkan", "metal", "nnapi"])
    def test_valid_backends(self, backend: str) -> None:
        """Test all valid backends."""
        config = create_mobile_config(backend=backend)
        assert config.backend == backend


class TestListScriptModes:
    """Tests for list_script_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        modes = list_script_modes()
        assert isinstance(modes, list)

    def test_contains_expected_modes(self) -> None:
        """Test that list contains expected modes."""
        modes = list_script_modes()
        assert "trace" in modes
        assert "script" in modes
        assert "hybrid" in modes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        modes = list_script_modes()
        assert modes == sorted(modes)


class TestListOptimizeForOptions:
    """Tests for list_optimize_for_options function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        options = list_optimize_for_options()
        assert isinstance(options, list)

    def test_contains_expected_options(self) -> None:
        """Test that list contains expected options."""
        options = list_optimize_for_options()
        assert "inference" in options
        assert "mobile" in options
        assert "training" in options

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        options = list_optimize_for_options()
        assert options == sorted(options)


class TestListFreezeModes:
    """Tests for list_freeze_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        modes = list_freeze_modes()
        assert isinstance(modes, list)

    def test_contains_expected_modes(self) -> None:
        """Test that list contains expected modes."""
        modes = list_freeze_modes()
        assert "none" in modes
        assert "preserve_parameters" in modes
        assert "full" in modes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        modes = list_freeze_modes()
        assert modes == sorted(modes)


class TestGetScriptMode:
    """Tests for get_script_mode function."""

    def test_get_trace(self) -> None:
        """Test getting TRACE mode."""
        assert get_script_mode("trace") == ScriptMode.TRACE

    def test_get_script(self) -> None:
        """Test getting SCRIPT mode."""
        assert get_script_mode("script") == ScriptMode.SCRIPT

    def test_get_hybrid(self) -> None:
        """Test getting HYBRID mode."""
        assert get_script_mode("hybrid") == ScriptMode.HYBRID

    def test_invalid_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid script mode"):
            get_script_mode("invalid")


class TestGetOptimizeFor:
    """Tests for get_optimize_for function."""

    def test_get_inference(self) -> None:
        """Test getting INFERENCE target."""
        assert get_optimize_for("inference") == OptimizeFor.INFERENCE

    def test_get_mobile(self) -> None:
        """Test getting MOBILE target."""
        assert get_optimize_for("mobile") == OptimizeFor.MOBILE

    def test_get_training(self) -> None:
        """Test getting TRAINING target."""
        assert get_optimize_for("training") == OptimizeFor.TRAINING

    def test_invalid_raises_error(self) -> None:
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError, match="invalid optimize_for value"):
            get_optimize_for("invalid")


class TestGetFreezeMode:
    """Tests for get_freeze_mode function."""

    def test_get_none(self) -> None:
        """Test getting NONE mode."""
        assert get_freeze_mode("none") == FreezeMode.NONE

    def test_get_preserve_parameters(self) -> None:
        """Test getting PRESERVE_PARAMETERS mode."""
        assert get_freeze_mode("preserve_parameters") == FreezeMode.PRESERVE_PARAMETERS

    def test_get_full(self) -> None:
        """Test getting FULL mode."""
        assert get_freeze_mode("full") == FreezeMode.FULL

    def test_invalid_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid freeze mode"):
            get_freeze_mode("invalid")


class TestEstimateScriptSize:
    """Tests for estimate_script_size function."""

    def test_estimates_size(self) -> None:
        """Test estimating size."""
        size = estimate_script_size(7_000_000_000)
        assert size > 0
        assert size < 30000

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_script_size(0)

    def test_negative_params_raises_error(self) -> None:
        """Test that negative params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_script_size(-1)

    def test_mobile_smaller_than_training(self) -> None:
        """Test mobile optimization produces smaller size than training."""
        size_mobile = estimate_script_size(
            7_000_000_000,
            optimize_for=OptimizeFor.MOBILE,
        )
        size_training = estimate_script_size(
            7_000_000_000,
            optimize_for=OptimizeFor.TRAINING,
        )
        assert size_mobile < size_training

    def test_inference_smaller_than_training(self) -> None:
        """Test inference optimization produces smaller size than training."""
        size_inference = estimate_script_size(
            7_000_000_000,
            optimize_for=OptimizeFor.INFERENCE,
        )
        size_training = estimate_script_size(
            7_000_000_000,
            optimize_for=OptimizeFor.TRAINING,
        )
        assert size_inference < size_training

    def test_full_freeze_reduces_size(self) -> None:
        """Test full freeze mode reduces model size."""
        size_none = estimate_script_size(
            7_000_000_000,
            freeze_mode=FreezeMode.NONE,
        )
        size_full = estimate_script_size(
            7_000_000_000,
            freeze_mode=FreezeMode.FULL,
        )
        assert size_full < size_none

    def test_preserve_parameters_reduces_size(self) -> None:
        """Test preserve parameters freeze mode reduces model size."""
        size_none = estimate_script_size(
            7_000_000_000,
            freeze_mode=FreezeMode.NONE,
        )
        size_preserve = estimate_script_size(
            7_000_000_000,
            freeze_mode=FreezeMode.PRESERVE_PARAMETERS,
        )
        assert size_preserve < size_none


class TestCheckScriptable:
    """Tests for check_scriptable function."""

    def test_transformer_scriptable(self) -> None:
        """Test transformer is scriptable."""
        assert check_scriptable("transformer") is True

    def test_cnn_scriptable(self) -> None:
        """Test CNN is scriptable."""
        assert check_scriptable("cnn") is True

    def test_rnn_scriptable(self) -> None:
        """Test RNN is scriptable."""
        assert check_scriptable("rnn") is True

    def test_lstm_scriptable(self) -> None:
        """Test LSTM is scriptable."""
        assert check_scriptable("lstm") is True

    def test_gru_scriptable(self) -> None:
        """Test GRU is scriptable."""
        assert check_scriptable("gru") is True

    def test_mlp_scriptable(self) -> None:
        """Test MLP is scriptable."""
        assert check_scriptable("mlp") is True

    def test_resnet_scriptable(self) -> None:
        """Test ResNet is scriptable."""
        assert check_scriptable("resnet") is True

    def test_vgg_scriptable(self) -> None:
        """Test VGG is scriptable."""
        assert check_scriptable("vgg") is True

    def test_bert_scriptable(self) -> None:
        """Test BERT is scriptable."""
        assert check_scriptable("bert") is True

    def test_gpt_scriptable(self) -> None:
        """Test GPT is scriptable."""
        assert check_scriptable("gpt") is True

    def test_empty_type_not_scriptable(self) -> None:
        """Test empty type is not scriptable."""
        assert check_scriptable("") is False

    def test_with_control_flow_only_scriptable(self) -> None:
        """Test model with control flow only is scriptable."""
        assert check_scriptable("transformer", has_control_flow=True) is True

    def test_with_dynamic_shapes_only_scriptable(self) -> None:
        """Test model with dynamic shapes only is scriptable."""
        assert check_scriptable("transformer", has_dynamic_shapes=True) is True

    def test_with_both_not_scriptable(self) -> None:
        """Test model with both control flow and dynamic shapes is not scriptable."""
        assert (
            check_scriptable(
                "custom", has_control_flow=True, has_dynamic_shapes=True
            )
            is False
        )

    def test_known_type_with_both_not_scriptable(self) -> None:
        """Test known type with both features is not scriptable."""
        assert (
            check_scriptable(
                "transformer", has_control_flow=True, has_dynamic_shapes=True
            )
            is False
        )

    def test_unknown_type_without_dynamic_features_scriptable(self) -> None:
        """Test unknown type without dynamic features is scriptable."""
        assert check_scriptable("custom") is True

    def test_case_insensitive(self) -> None:
        """Test model type matching is case insensitive."""
        assert check_scriptable("TRANSFORMER") is True
        assert check_scriptable("Transformer") is True
        assert check_scriptable("CNN") is True


class TestGetTorchScriptConfigDict:
    """Tests for get_torchscript_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_torchscript_config(mode="trace")
        d = get_torchscript_config_dict(config)
        assert d["mode"] == "trace"
        assert d["strict"] is True
        assert d["optimize_for"] == "inference"
        assert d["check_trace"] is True

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_torchscript_config_dict(None)  # type: ignore[arg-type]


class TestGetTraceConfigDict:
    """Tests for get_trace_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_trace_config(example_inputs="tensor")
        d = get_trace_config_dict(config)
        assert d["example_inputs"] == "tensor"
        assert d["check_inputs"] is True
        assert d["check_tolerance"] == pytest.approx(1e-5)

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_trace_config_dict(None)  # type: ignore[arg-type]


class TestGetOptimizationConfigDict:
    """Tests for get_optimization_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_optimization_config(freeze_mode="full")
        d = get_optimization_config_dict(config)
        assert d["freeze_mode"] == "full"
        assert d["fuse_operations"] is True
        assert d["inline_functions"] is True
        assert d["remove_dropout"] is True

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_optimization_config_dict(None)  # type: ignore[arg-type]


class TestGetMobileConfigDict:
    """Tests for get_mobile_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_mobile_config(backend="vulkan")
        d = get_mobile_config_dict(config)
        assert d["backend"] == "vulkan"
        assert d["optimize_for_mobile"] is False
        assert d["preserve_dtype"] is True

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_mobile_config_dict(None)  # type: ignore[arg-type]


class TestFormatTorchScriptInfo:
    """Tests for format_torchscript_info function."""

    def test_formats_info(self) -> None:
        """Test formatting info."""
        info = TorchScriptInfo(
            model_path="/models/model.pt",
            script_mode=ScriptMode.TRACE,
            graph_size=1500,
            num_parameters=7_000_000_000,
        )
        formatted = format_torchscript_info(info)
        assert "Path:" in formatted
        assert "/models/model.pt" in formatted
        assert "Mode:" in formatted
        assert "trace" in formatted
        assert "Graph size:" in formatted
        assert "1500" in formatted
        assert "Parameters:" in formatted
        assert "7,000,000,000" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="info cannot be None"):
            format_torchscript_info(None)  # type: ignore[arg-type]


class TestFormatCompilationStats:
    """Tests for format_compilation_stats function."""

    def test_formats_stats(self) -> None:
        """Test formatting stats."""
        stats = CompilationStats(
            compile_time_seconds=15.5,
            graph_nodes=1200,
            fused_ops=350,
            model_size_bytes=4_000_000_000,
        )
        formatted = format_compilation_stats(stats)
        assert "Compile time:" in formatted
        assert "15.50s" in formatted
        assert "Graph nodes:" in formatted
        assert "1200" in formatted
        assert "Fused ops:" in formatted
        assert "350" in formatted
        assert "Model size:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_compilation_stats(None)  # type: ignore[arg-type]


class TestCreateTorchScriptInfo:
    """Tests for create_torchscript_info function."""

    def test_creates_info_with_string_mode(self) -> None:
        """Test creating info with string mode."""
        info = create_torchscript_info(
            "/models/model.pt", "trace", 1500, 7_000_000_000
        )
        assert info.script_mode == ScriptMode.TRACE

    def test_creates_info_with_enum_mode(self) -> None:
        """Test creating info with enum mode."""
        info = create_torchscript_info(
            "/models/model.pt", ScriptMode.SCRIPT, 1500, 7_000_000_000
        )
        assert info.script_mode == ScriptMode.SCRIPT

    def test_empty_path_raises_error(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="model_path cannot be empty"):
            create_torchscript_info("", "trace", 100, 1000)

    def test_negative_graph_size_raises_error(self) -> None:
        """Test that negative graph size raises ValueError."""
        with pytest.raises(ValueError, match="graph_size cannot be negative"):
            create_torchscript_info("/models/model.pt", "trace", -1, 1000)

    def test_zero_graph_size_allowed(self) -> None:
        """Test that zero graph size is allowed."""
        info = create_torchscript_info("/models/model.pt", "trace", 0, 1000)
        assert info.graph_size == 0

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            create_torchscript_info("/models/model.pt", "trace", 100, 0)

    def test_negative_params_raises_error(self) -> None:
        """Test that negative params raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            create_torchscript_info("/models/model.pt", "trace", 100, -1)


class TestCreateCompilationStats:
    """Tests for create_compilation_stats function."""

    def test_creates_stats(self) -> None:
        """Test creating stats."""
        stats = create_compilation_stats(15.5, 1200, 350, 4_000_000_000)
        assert stats.compile_time_seconds == pytest.approx(15.5)
        assert stats.graph_nodes == 1200
        assert stats.fused_ops == 350
        assert stats.model_size_bytes == 4_000_000_000

    def test_negative_compile_time_raises_error(self) -> None:
        """Test that negative compile time raises ValueError."""
        with pytest.raises(ValueError, match="compile_time_seconds cannot be negative"):
            create_compilation_stats(-1.0, 100, 10, 1000)

    def test_zero_compile_time_allowed(self) -> None:
        """Test that zero compile time is allowed."""
        stats = create_compilation_stats(0.0, 100, 10, 1000)
        assert stats.compile_time_seconds == pytest.approx(0.0)

    def test_negative_graph_nodes_raises_error(self) -> None:
        """Test that negative graph nodes raises ValueError."""
        with pytest.raises(ValueError, match="graph_nodes cannot be negative"):
            create_compilation_stats(1.0, -1, 10, 1000)

    def test_negative_fused_ops_raises_error(self) -> None:
        """Test that negative fused ops raises ValueError."""
        with pytest.raises(ValueError, match="fused_ops cannot be negative"):
            create_compilation_stats(1.0, 100, -1, 1000)

    def test_zero_model_size_raises_error(self) -> None:
        """Test that zero model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_bytes must be positive"):
            create_compilation_stats(1.0, 100, 10, 0)

    def test_negative_model_size_raises_error(self) -> None:
        """Test that negative model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_bytes must be positive"):
            create_compilation_stats(1.0, 100, 10, -1)


class TestGetRecommendedConfig:
    """Tests for get_recommended_config function."""

    def test_transformer_server(self) -> None:
        """Test recommended config for transformer on server."""
        config = get_recommended_config("transformer")
        assert config.mode == ScriptMode.TRACE
        assert config.optimize_for == OptimizeFor.INFERENCE

    def test_cnn_server(self) -> None:
        """Test recommended config for CNN on server."""
        config = get_recommended_config("cnn")
        assert config.mode == ScriptMode.TRACE
        assert config.optimize_for == OptimizeFor.INFERENCE

    def test_rnn_server(self) -> None:
        """Test recommended config for RNN on server."""
        config = get_recommended_config("rnn")
        assert config.mode == ScriptMode.SCRIPT
        assert config.optimize_for == OptimizeFor.INFERENCE

    def test_lstm_server(self) -> None:
        """Test recommended config for LSTM on server."""
        config = get_recommended_config("lstm")
        assert config.mode == ScriptMode.SCRIPT

    def test_gru_server(self) -> None:
        """Test recommended config for GRU on server."""
        config = get_recommended_config("gru")
        assert config.mode == ScriptMode.SCRIPT

    def test_transformer_mobile(self) -> None:
        """Test recommended config for transformer on mobile."""
        config = get_recommended_config("transformer", "mobile")
        assert config.mode == ScriptMode.TRACE
        assert config.optimize_for == OptimizeFor.MOBILE

    def test_cnn_mobile(self) -> None:
        """Test recommended config for CNN on mobile."""
        config = get_recommended_config("cnn", "mobile")
        assert config.optimize_for == OptimizeFor.MOBILE

    def test_transformer_edge(self) -> None:
        """Test recommended config for transformer on edge."""
        config = get_recommended_config("transformer", "edge")
        assert config.optimize_for == OptimizeFor.INFERENCE

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_config("")

    def test_invalid_platform_raises_error(self) -> None:
        """Test that invalid platform raises ValueError."""
        with pytest.raises(ValueError, match="target_platform must be one of"):
            get_recommended_config("transformer", "invalid")

    @pytest.mark.parametrize("platform", ["server", "mobile", "edge"])
    def test_all_valid_platforms(self, platform: str) -> None:
        """Test all valid platforms work."""
        config = get_recommended_config("transformer", platform)
        assert config is not None
        assert config.strict is True
        assert config.check_trace is True
