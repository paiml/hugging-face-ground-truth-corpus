"""Tests for training.debugging module."""

from __future__ import annotations

import pytest

from hf_gtc.training.debugging import (
    VALID_ANOMALY_TYPES,
    VALID_DEBUG_LEVELS,
    VALID_VISUALIZATION_TYPES,
    ActivationConfig,
    AnomalyType,
    DebugConfig,
    DebugLevel,
    DebugStats,
    GradientFlowConfig,
    VisualizationType,
    analyze_loss_landscape,
    compute_activation_stats,
    compute_gradient_flow,
    create_activation_config,
    create_debug_config,
    create_debug_stats,
    create_gradient_flow_config,
    detect_anomalies,
    diagnose_training_issues,
    format_debug_stats,
    get_anomaly_type,
    get_debug_level,
    get_recommended_debug_config,
    get_visualization_type,
    list_anomaly_types,
    list_debug_levels,
    list_visualization_types,
    validate_activation_config,
    validate_debug_config,
    validate_debug_stats,
    validate_gradient_flow_config,
)


class TestDebugLevel:
    """Tests for DebugLevel enum."""

    def test_all_levels_have_values(self) -> None:
        """All levels have string values."""
        for level in DebugLevel:
            assert isinstance(level.value, str)

    def test_none_value(self) -> None:
        """None has correct value."""
        assert DebugLevel.NONE.value == "none"

    def test_basic_value(self) -> None:
        """Basic has correct value."""
        assert DebugLevel.BASIC.value == "basic"

    def test_verbose_value(self) -> None:
        """Verbose has correct value."""
        assert DebugLevel.VERBOSE.value == "verbose"

    def test_trace_value(self) -> None:
        """Trace has correct value."""
        assert DebugLevel.TRACE.value == "trace"

    def test_valid_debug_levels_frozenset(self) -> None:
        """VALID_DEBUG_LEVELS is a frozenset."""
        assert isinstance(VALID_DEBUG_LEVELS, frozenset)
        assert len(VALID_DEBUG_LEVELS) == 4


class TestVisualizationType:
    """Tests for VisualizationType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for vtype in VisualizationType:
            assert isinstance(vtype.value, str)

    def test_gradient_flow_value(self) -> None:
        """Gradient flow has correct value."""
        assert VisualizationType.GRADIENT_FLOW.value == "gradient_flow"

    def test_activation_histogram_value(self) -> None:
        """Activation histogram has correct value."""
        assert VisualizationType.ACTIVATION_HISTOGRAM.value == "activation_histogram"

    def test_attention_weights_value(self) -> None:
        """Attention weights has correct value."""
        assert VisualizationType.ATTENTION_WEIGHTS.value == "attention_weights"

    def test_loss_landscape_value(self) -> None:
        """Loss landscape has correct value."""
        assert VisualizationType.LOSS_LANDSCAPE.value == "loss_landscape"

    def test_valid_visualization_types_frozenset(self) -> None:
        """VALID_VISUALIZATION_TYPES is a frozenset."""
        assert isinstance(VALID_VISUALIZATION_TYPES, frozenset)
        assert len(VALID_VISUALIZATION_TYPES) == 4


class TestAnomalyType:
    """Tests for AnomalyType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for atype in AnomalyType:
            assert isinstance(atype.value, str)

    def test_nan_value(self) -> None:
        """NaN has correct value."""
        assert AnomalyType.NAN.value == "nan"

    def test_inf_value(self) -> None:
        """Inf has correct value."""
        assert AnomalyType.INF.value == "inf"

    def test_exploding_gradient_value(self) -> None:
        """Exploding gradient has correct value."""
        assert AnomalyType.EXPLODING_GRADIENT.value == "exploding_gradient"

    def test_vanishing_gradient_value(self) -> None:
        """Vanishing gradient has correct value."""
        assert AnomalyType.VANISHING_GRADIENT.value == "vanishing_gradient"

    def test_dead_neuron_value(self) -> None:
        """Dead neuron has correct value."""
        assert AnomalyType.DEAD_NEURON.value == "dead_neuron"

    def test_valid_anomaly_types_frozenset(self) -> None:
        """VALID_ANOMALY_TYPES is a frozenset."""
        assert isinstance(VALID_ANOMALY_TYPES, frozenset)
        assert len(VALID_ANOMALY_TYPES) == 5


class TestDebugConfig:
    """Tests for DebugConfig dataclass."""

    def test_create_config(self) -> None:
        """Create debug config."""
        config = DebugConfig(
            level=DebugLevel.BASIC,
            log_gradients=True,
            log_activations=False,
            check_anomalies=True,
        )
        assert config.level == DebugLevel.BASIC
        assert config.log_gradients is True
        assert config.log_activations is False
        assert config.check_anomalies is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = DebugConfig(
            level=DebugLevel.BASIC,
            log_gradients=True,
            log_activations=False,
            check_anomalies=True,
        )
        with pytest.raises(AttributeError):
            config.level = DebugLevel.VERBOSE  # type: ignore[misc]


class TestGradientFlowConfig:
    """Tests for GradientFlowConfig dataclass."""

    def test_create_config(self) -> None:
        """Create gradient flow config."""
        config = GradientFlowConfig(
            layers=("encoder", "decoder"),
            reduction="mean",
            log_frequency=100,
        )
        assert config.layers == ("encoder", "decoder")
        assert config.reduction == "mean"
        assert config.log_frequency == 100

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = GradientFlowConfig(
            layers=("layer1",),
            reduction="mean",
            log_frequency=100,
        )
        with pytest.raises(AttributeError):
            config.reduction = "max"  # type: ignore[misc]


class TestActivationConfig:
    """Tests for ActivationConfig dataclass."""

    def test_create_config(self) -> None:
        """Create activation config."""
        config = ActivationConfig(
            layers=("attention", "ffn"),
            num_bins=50,
            track_statistics=True,
        )
        assert config.layers == ("attention", "ffn")
        assert config.num_bins == 50
        assert config.track_statistics is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ActivationConfig(
            layers=("layer1",),
            num_bins=50,
            track_statistics=True,
        )
        with pytest.raises(AttributeError):
            config.num_bins = 100  # type: ignore[misc]


class TestDebugStats:
    """Tests for DebugStats dataclass."""

    def test_create_stats(self) -> None:
        """Create debug stats."""
        stats = DebugStats(
            anomalies_detected={"nan": 0, "inf": 1},
            gradient_norm_history=(0.5, 0.6, 0.55),
            activation_stats={"layer1": {"mean": 0.1, "std": 0.05}},
        )
        assert stats.anomalies_detected == {"nan": 0, "inf": 1}
        assert stats.gradient_norm_history == (0.5, 0.6, 0.55)
        assert stats.activation_stats["layer1"]["mean"] == pytest.approx(0.1)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = DebugStats(
            anomalies_detected={},
            gradient_norm_history=(),
            activation_stats={},
        )
        with pytest.raises(AttributeError):
            stats.anomalies_detected = {"nan": 1}  # type: ignore[misc]


class TestValidateDebugConfig:
    """Tests for validate_debug_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DebugConfig(DebugLevel.BASIC, True, False, True)
        validate_debug_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_debug_config(None)  # type: ignore[arg-type]


class TestValidateGradientFlowConfig:
    """Tests for validate_gradient_flow_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = GradientFlowConfig(("layer1",), "mean", 100)
        validate_gradient_flow_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_gradient_flow_config(None)  # type: ignore[arg-type]

    def test_invalid_reduction_raises(self) -> None:
        """Invalid reduction raises ValueError."""
        config = GradientFlowConfig(("layer1",), "invalid", 100)
        with pytest.raises(ValueError, match="reduction must be one of"):
            validate_gradient_flow_config(config)

    def test_zero_log_frequency_raises(self) -> None:
        """Zero log_frequency raises ValueError."""
        config = GradientFlowConfig(("layer1",), "mean", 0)
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            validate_gradient_flow_config(config)

    def test_negative_log_frequency_raises(self) -> None:
        """Negative log_frequency raises ValueError."""
        config = GradientFlowConfig(("layer1",), "mean", -1)
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            validate_gradient_flow_config(config)


class TestValidateActivationConfig:
    """Tests for validate_activation_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ActivationConfig(("layer1",), 50, True)
        validate_activation_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_activation_config(None)  # type: ignore[arg-type]

    def test_zero_num_bins_raises(self) -> None:
        """Zero num_bins raises ValueError."""
        config = ActivationConfig(("layer1",), 0, True)
        with pytest.raises(ValueError, match="num_bins must be positive"):
            validate_activation_config(config)

    def test_negative_num_bins_raises(self) -> None:
        """Negative num_bins raises ValueError."""
        config = ActivationConfig(("layer1",), -1, True)
        with pytest.raises(ValueError, match="num_bins must be positive"):
            validate_activation_config(config)


class TestValidateDebugStats:
    """Tests for validate_debug_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = DebugStats({"nan": 0}, (0.5, 0.6), {"l1": {"mean": 0.1}})
        validate_debug_stats(stats)

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_debug_stats(None)  # type: ignore[arg-type]

    def test_negative_anomaly_count_raises(self) -> None:
        """Negative anomaly count raises ValueError."""
        stats = DebugStats({"nan": -1}, (0.5,), {})
        with pytest.raises(ValueError, match="anomaly count cannot be negative"):
            validate_debug_stats(stats)

    def test_negative_gradient_norm_raises(self) -> None:
        """Negative gradient norm raises ValueError."""
        stats = DebugStats({}, (-0.5,), {})
        with pytest.raises(ValueError, match="gradient norm cannot be negative"):
            validate_debug_stats(stats)


class TestCreateDebugConfig:
    """Tests for create_debug_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_debug_config()
        assert config.level == DebugLevel.BASIC
        assert config.log_gradients is True
        assert config.log_activations is False
        assert config.check_anomalies is True

    def test_verbose_config(self) -> None:
        """Create verbose config."""
        config = create_debug_config(level="verbose")
        assert config.level == DebugLevel.VERBOSE

    def test_trace_config(self) -> None:
        """Create trace config."""
        config = create_debug_config(level="trace")
        assert config.level == DebugLevel.TRACE

    def test_none_level(self) -> None:
        """Create config with none level."""
        config = create_debug_config(level="none")
        assert config.level == DebugLevel.NONE

    def test_all_flags(self) -> None:
        """Create config with all flags."""
        config = create_debug_config(
            level="verbose",
            log_gradients=True,
            log_activations=True,
            check_anomalies=False,
        )
        assert config.log_activations is True
        assert config.check_anomalies is False

    def test_invalid_level_raises(self) -> None:
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="level must be one of"):
            create_debug_config(level="invalid")


class TestCreateGradientFlowConfig:
    """Tests for create_gradient_flow_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_gradient_flow_config()
        assert config.layers == ()
        assert config.reduction == "mean"
        assert config.log_frequency == 100

    def test_with_layers(self) -> None:
        """Create config with layers."""
        config = create_gradient_flow_config(layers=("encoder", "decoder"))
        assert config.layers == ("encoder", "decoder")

    def test_max_reduction(self) -> None:
        """Create config with max reduction."""
        config = create_gradient_flow_config(reduction="max")
        assert config.reduction == "max"

    def test_min_reduction(self) -> None:
        """Create config with min reduction."""
        config = create_gradient_flow_config(reduction="min")
        assert config.reduction == "min"

    def test_custom_log_frequency(self) -> None:
        """Create config with custom log frequency."""
        config = create_gradient_flow_config(log_frequency=50)
        assert config.log_frequency == 50

    def test_invalid_reduction_raises(self) -> None:
        """Invalid reduction raises ValueError."""
        with pytest.raises(ValueError, match="reduction must be one of"):
            create_gradient_flow_config(reduction="invalid")

    def test_zero_log_frequency_raises(self) -> None:
        """Zero log_frequency raises ValueError."""
        with pytest.raises(ValueError, match="log_frequency must be positive"):
            create_gradient_flow_config(log_frequency=0)


class TestCreateActivationConfig:
    """Tests for create_activation_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_activation_config()
        assert config.layers == ()
        assert config.num_bins == 50
        assert config.track_statistics is True

    def test_with_layers(self) -> None:
        """Create config with layers."""
        config = create_activation_config(layers=("attention",))
        assert config.layers == ("attention",)

    def test_custom_bins(self) -> None:
        """Create config with custom bins."""
        config = create_activation_config(num_bins=100)
        assert config.num_bins == 100

    def test_no_track_statistics(self) -> None:
        """Create config without tracking statistics."""
        config = create_activation_config(track_statistics=False)
        assert config.track_statistics is False

    def test_zero_bins_raises(self) -> None:
        """Zero bins raises ValueError."""
        with pytest.raises(ValueError, match="num_bins must be positive"):
            create_activation_config(num_bins=0)


class TestCreateDebugStats:
    """Tests for create_debug_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_debug_stats()
        assert stats.anomalies_detected == {}
        assert stats.gradient_norm_history == ()
        assert stats.activation_stats == {}

    def test_with_anomalies(self) -> None:
        """Create stats with anomalies."""
        stats = create_debug_stats(anomalies_detected={"nan": 1, "inf": 2})
        assert stats.anomalies_detected == {"nan": 1, "inf": 2}

    def test_with_gradient_history(self) -> None:
        """Create stats with gradient history."""
        stats = create_debug_stats(gradient_norm_history=(0.5, 0.6, 0.7))
        assert stats.gradient_norm_history == (0.5, 0.6, 0.7)

    def test_with_activation_stats(self) -> None:
        """Create stats with activation stats."""
        act_stats = {"layer1": {"mean": 0.1, "std": 0.05}}
        stats = create_debug_stats(activation_stats=act_stats)
        assert stats.activation_stats == act_stats


class TestDetectAnomalies:
    """Tests for detect_anomalies function."""

    def test_no_anomalies(self) -> None:
        """No anomalies in normal values."""
        result = detect_anomalies([1.0, 2.0, 3.0])
        assert result["nan"] == 0
        assert result["inf"] == 0
        assert result["exploding_gradient"] == 0
        assert result["vanishing_gradient"] == 0
        assert result["dead_neuron"] == 0

    def test_detect_nan(self) -> None:
        """Detect NaN values."""
        result = detect_anomalies([float("nan"), 1.0])
        assert result["nan"] == 1

    def test_detect_multiple_nan(self) -> None:
        """Detect multiple NaN values."""
        result = detect_anomalies([float("nan"), float("nan"), 1.0])
        assert result["nan"] == 2

    def test_detect_inf(self) -> None:
        """Detect inf values."""
        result = detect_anomalies([float("inf"), 1.0])
        assert result["inf"] == 1

    def test_detect_negative_inf(self) -> None:
        """Detect negative inf values."""
        result = detect_anomalies([float("-inf"), 1.0])
        assert result["inf"] == 1

    def test_detect_exploding_gradient(self) -> None:
        """Detect exploding gradients."""
        result = detect_anomalies([2000.0, 1.0])
        assert result["exploding_gradient"] == 1

    def test_detect_vanishing_gradient(self) -> None:
        """Detect vanishing gradients."""
        result = detect_anomalies([1e-10, 1.0])
        assert result["vanishing_gradient"] == 1

    def test_detect_dead_neuron(self) -> None:
        """Detect dead neurons (all zeros)."""
        result = detect_anomalies([0.0, 0.0, 0.0])
        assert result["dead_neuron"] == 1

    def test_not_dead_neuron_with_nonzero(self) -> None:
        """Not dead neuron if any non-zero."""
        result = detect_anomalies([0.0, 0.0, 0.1])
        assert result["dead_neuron"] == 0

    def test_custom_exploding_threshold(self) -> None:
        """Use custom exploding threshold."""
        result = detect_anomalies([200.0, 1.0], exploding_threshold=100.0)
        assert result["exploding_gradient"] == 1

    def test_custom_vanishing_threshold(self) -> None:
        """Use custom vanishing threshold."""
        result = detect_anomalies([0.001, 1.0], vanishing_threshold=0.01)
        assert result["vanishing_gradient"] == 1

    def test_empty_values_raises(self) -> None:
        """Empty values raises ValueError."""
        with pytest.raises(ValueError, match="values cannot be empty"):
            detect_anomalies([])

    def test_zero_exploding_threshold_raises(self) -> None:
        """Zero exploding threshold raises ValueError."""
        with pytest.raises(ValueError, match="exploding_threshold must be positive"):
            detect_anomalies([1.0], exploding_threshold=0.0)

    def test_negative_exploding_threshold_raises(self) -> None:
        """Negative exploding threshold raises ValueError."""
        with pytest.raises(ValueError, match="exploding_threshold must be positive"):
            detect_anomalies([1.0], exploding_threshold=-1.0)

    def test_negative_vanishing_threshold_raises(self) -> None:
        """Negative vanishing threshold raises ValueError."""
        with pytest.raises(ValueError, match="vanishing_threshold cannot be negative"):
            detect_anomalies([1.0], vanishing_threshold=-1.0)


class TestComputeGradientFlow:
    """Tests for compute_gradient_flow function."""

    def test_mean_reduction(self) -> None:
        """Compute with mean reduction."""
        config = GradientFlowConfig(("layer1",), "mean", 100)
        grads = {"layer1": [1.0, 2.0, 3.0]}
        result = compute_gradient_flow(grads, config)
        assert result["layer1"] == pytest.approx(2.0)

    def test_max_reduction(self) -> None:
        """Compute with max reduction."""
        config = GradientFlowConfig(("layer1",), "max", 100)
        grads = {"layer1": [1.0, 2.0, 3.0]}
        result = compute_gradient_flow(grads, config)
        assert result["layer1"] == pytest.approx(3.0)

    def test_min_reduction(self) -> None:
        """Compute with min reduction."""
        config = GradientFlowConfig(("layer1",), "min", 100)
        grads = {"layer1": [1.0, 2.0, 3.0]}
        result = compute_gradient_flow(grads, config)
        assert result["layer1"] == pytest.approx(1.0)

    def test_multiple_layers(self) -> None:
        """Compute for multiple layers."""
        config = GradientFlowConfig(("layer1", "layer2"), "mean", 100)
        grads = {"layer1": [1.0, 2.0], "layer2": [3.0, 4.0]}
        result = compute_gradient_flow(grads, config)
        assert result["layer1"] == pytest.approx(1.5)
        assert result["layer2"] == pytest.approx(3.5)

    def test_negative_gradients(self) -> None:
        """Compute with negative gradients uses absolute values."""
        config = GradientFlowConfig(("layer1",), "mean", 100)
        grads = {"layer1": [-1.0, -2.0, -3.0]}
        result = compute_gradient_flow(grads, config)
        assert result["layer1"] == pytest.approx(2.0)

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        config = GradientFlowConfig(("layer1",), "mean", 100)
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            compute_gradient_flow({}, config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            compute_gradient_flow({"layer1": [1.0]}, None)  # type: ignore[arg-type]

    def test_empty_layer_gradients_raises(self) -> None:
        """Empty layer gradients raises ValueError."""
        config = GradientFlowConfig(("layer1",), "mean", 100)
        with pytest.raises(ValueError, match=r"gradients for layer.*cannot be empty"):
            compute_gradient_flow({"layer1": []}, config)


class TestComputeActivationStats:
    """Tests for compute_activation_stats function."""

    def test_basic_stats(self) -> None:
        """Compute basic statistics."""
        config = ActivationConfig(("layer1",), 50, True)
        acts = {"layer1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        result = compute_activation_stats(acts, config)
        assert result["layer1"]["mean"] == pytest.approx(3.0)
        assert result["layer1"]["min"] == pytest.approx(1.0)
        assert result["layer1"]["max"] == pytest.approx(5.0)

    def test_std_calculation(self) -> None:
        """Compute standard deviation correctly."""
        config = ActivationConfig(("layer1",), 50, True)
        acts = {"layer1": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}
        result = compute_activation_stats(acts, config)
        assert result["layer1"]["mean"] == pytest.approx(5.0)
        assert abs(result["layer1"]["std"] - 2.0) < 0.01

    def test_sparsity_calculation(self) -> None:
        """Compute sparsity correctly."""
        config = ActivationConfig(("layer1",), 50, True)
        acts = {"layer1": [0.0, 0.0, 1.0, 2.0, 0.0]}
        result = compute_activation_stats(acts, config)
        assert result["layer1"]["sparsity"] == pytest.approx(0.6)

    def test_multiple_layers(self) -> None:
        """Compute for multiple layers."""
        config = ActivationConfig(("layer1", "layer2"), 50, True)
        acts = {"layer1": [1.0, 2.0], "layer2": [3.0, 4.0]}
        result = compute_activation_stats(acts, config)
        assert result["layer1"]["mean"] == pytest.approx(1.5)
        assert result["layer2"]["mean"] == pytest.approx(3.5)

    def test_empty_activations_raises(self) -> None:
        """Empty activations raises ValueError."""
        config = ActivationConfig(("layer1",), 50, True)
        with pytest.raises(ValueError, match="activations cannot be empty"):
            compute_activation_stats({}, config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            compute_activation_stats({"layer1": [1.0]}, None)  # type: ignore[arg-type]

    def test_empty_layer_activations_raises(self) -> None:
        """Empty layer activations raises ValueError."""
        config = ActivationConfig(("layer1",), 50, True)
        with pytest.raises(ValueError, match=r"activations for layer.*cannot be empty"):
            compute_activation_stats({"layer1": []}, config)


class TestAnalyzeLossLandscape:
    """Tests for analyze_loss_landscape function."""

    def test_decreasing_loss(self) -> None:
        """Analyze decreasing loss trend."""
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        result = analyze_loss_landscape(losses)
        assert result["trend"] == -0.1
        assert result["min_loss"] == pytest.approx(0.6)
        assert result["max_loss"] == pytest.approx(1.0)
        assert result["final_loss"] == pytest.approx(0.6)

    def test_increasing_loss(self) -> None:
        """Analyze increasing loss trend."""
        losses = [0.6, 0.7, 0.8, 0.9, 1.0]
        result = analyze_loss_landscape(losses)
        assert result["trend"] == pytest.approx(0.1)

    def test_volatile_loss(self) -> None:
        """Analyze volatile loss."""
        losses = [1.0, 0.5, 1.2, 0.3, 0.8]
        result = analyze_loss_landscape(losses)
        assert result["volatility"] > 0

    def test_smooth_loss(self) -> None:
        """Analyze smooth loss landscape."""
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        result = analyze_loss_landscape(losses)
        assert result["smoothness"] > 0
        assert result["smoothness"] <= 1.0

    def test_custom_step_size(self) -> None:
        """Use custom step size."""
        losses = [1.0, 0.99, 0.98]
        result = analyze_loss_landscape(losses, step_size=0.001)
        assert "trend" in result

    def test_minimum_two_elements(self) -> None:
        """Requires at least 2 elements."""
        result = analyze_loss_landscape([1.0, 0.5])
        assert "trend" in result

    def test_single_element_raises(self) -> None:
        """Single element raises ValueError."""
        with pytest.raises(ValueError, match="loss_values must have at least 2"):
            analyze_loss_landscape([1.0])

    def test_empty_list_raises(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="loss_values must have at least 2"):
            analyze_loss_landscape([])

    def test_zero_step_size_raises(self) -> None:
        """Zero step_size raises ValueError."""
        with pytest.raises(ValueError, match="step_size must be positive"):
            analyze_loss_landscape([1.0, 0.5], step_size=0.0)

    def test_negative_step_size_raises(self) -> None:
        """Negative step_size raises ValueError."""
        with pytest.raises(ValueError, match="step_size must be positive"):
            analyze_loss_landscape([1.0, 0.5], step_size=-0.01)


class TestDiagnoseTrainingIssues:
    """Tests for diagnose_training_issues function."""

    def test_no_issues(self) -> None:
        """No issues with stable training."""
        norms = [1.0, 1.1, 1.0, 0.9, 1.0]
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        issues = diagnose_training_issues(norms, losses)
        assert len(issues) == 0

    def test_exploding_gradients(self) -> None:
        """Detect exploding gradients."""
        norms = [1.0, 10.0, 100.0, 1000.0]
        losses = [1.0, 2.0, 5.0, 10.0]
        issues = diagnose_training_issues(norms, losses)
        assert any("exploding" in issue.lower() for issue in issues)

    def test_vanishing_gradients(self) -> None:
        """Detect vanishing gradients."""
        norms = [1.0, 0.1, 0.01, 0.001]
        losses = [1.0, 1.0, 1.0, 1.0]
        issues = diagnose_training_issues(norms, losses)
        assert any("vanishing" in issue.lower() for issue in issues)

    def test_loss_not_decreasing(self) -> None:
        """Detect loss not decreasing."""
        norms = [1.0, 1.0, 1.0, 1.0, 1.0]
        losses = [1.0, 1.0, 1.0, 1.0, 1.0]
        issues = diagnose_training_issues(norms, losses)
        assert any("not decreasing" in issue.lower() for issue in issues)

    def test_high_volatility(self) -> None:
        """Detect high loss volatility."""
        norms = [1.0] * 10
        # Need many stable changes followed by one large change to trigger
        # volatility detection (std > mean_change * 2)
        losses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        issues = diagnose_training_issues(norms, losses)
        assert any("volatility" in issue.lower() for issue in issues)

    def test_lr_too_high(self) -> None:
        """Detect learning rate too high."""
        norms = [100.0, 100.0, 100.0]
        losses = [1.0, 0.9, 0.8]
        issues = diagnose_training_issues(norms, losses, learning_rate=0.1)
        assert any("too high" in issue.lower() for issue in issues)

    def test_lr_too_low(self) -> None:
        """Detect learning rate too low."""
        norms = [1e-6, 1e-6, 1e-6]
        losses = [1.0, 0.99, 0.98]
        issues = diagnose_training_issues(norms, losses, learning_rate=1e-8)
        assert any("too low" in issue.lower() for issue in issues)

    def test_empty_gradient_norms_raises(self) -> None:
        """Empty gradient_norms raises ValueError."""
        with pytest.raises(ValueError, match="gradient_norms cannot be empty"):
            diagnose_training_issues([], [1.0])

    def test_empty_loss_values_raises(self) -> None:
        """Empty loss_values raises ValueError."""
        with pytest.raises(ValueError, match="loss_values cannot be empty"):
            diagnose_training_issues([1.0], [])

    def test_zero_lr_raises(self) -> None:
        """Zero learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            diagnose_training_issues([1.0], [1.0], learning_rate=0.0)

    def test_negative_lr_raises(self) -> None:
        """Negative learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            diagnose_training_issues([1.0], [1.0], learning_rate=-0.01)


class TestFormatDebugStats:
    """Tests for format_debug_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = DebugStats(
            {"nan": 0, "inf": 1},
            (0.5, 0.6, 0.55),
            {"layer1": {"mean": 0.1}},
        )
        formatted = format_debug_stats(stats)
        assert "Anomalies Detected:" in formatted
        assert "nan: 0" in formatted
        assert "inf: 1" in formatted
        assert "Gradient Norm History:" in formatted
        assert "Activation Statistics:" in formatted
        assert "layer1:" in formatted

    def test_empty_stats(self) -> None:
        """Format empty stats."""
        stats = DebugStats({}, (), {})
        formatted = format_debug_stats(stats)
        assert "None" in formatted
        assert "No history" in formatted
        assert "No statistics" in formatted

    def test_gradient_history_mean(self) -> None:
        """Format shows gradient history mean."""
        stats = DebugStats({}, (1.0, 2.0, 3.0), {})
        formatted = format_debug_stats(stats)
        assert "Mean: 2.0000" in formatted

    def test_recent_gradient_history(self) -> None:
        """Format shows recent gradient history."""
        stats = DebugStats({}, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), {})
        formatted = format_debug_stats(stats)
        assert "Recent:" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_debug_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedDebugConfig:
    """Tests for get_recommended_debug_config function."""

    def test_fine_tuning_default(self) -> None:
        """Get config for fine tuning."""
        config = get_recommended_debug_config()
        assert config.level == DebugLevel.BASIC
        assert config.log_gradients is True

    def test_pretraining(self) -> None:
        """Get config for pretraining."""
        config = get_recommended_debug_config("pretraining")
        assert config.check_anomalies is True

    def test_rlhf(self) -> None:
        """Get config for RLHF."""
        config = get_recommended_debug_config("rlhf")
        assert config.check_anomalies is True

    def test_verbose_flag(self) -> None:
        """Get verbose config."""
        config = get_recommended_debug_config(verbose=True)
        assert config.level == DebugLevel.VERBOSE

    def test_large_model(self) -> None:
        """Get config for large model."""
        config = get_recommended_debug_config(model_size="70b")
        assert config.log_activations is True

    def test_175b_model(self) -> None:
        """Get config for 175B model."""
        config = get_recommended_debug_config(model_size="175b")
        assert config.log_activations is True

    def test_invalid_training_phase_raises(self) -> None:
        """Invalid training phase raises ValueError."""
        with pytest.raises(ValueError, match="training_phase must be one of"):
            get_recommended_debug_config("invalid")


class TestListDebugLevels:
    """Tests for list_debug_levels function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        levels = list_debug_levels()
        assert levels == sorted(levels)

    def test_contains_basic(self) -> None:
        """Contains basic."""
        levels = list_debug_levels()
        assert "basic" in levels

    def test_contains_verbose(self) -> None:
        """Contains verbose."""
        levels = list_debug_levels()
        assert "verbose" in levels

    def test_contains_trace(self) -> None:
        """Contains trace."""
        levels = list_debug_levels()
        assert "trace" in levels

    def test_contains_none(self) -> None:
        """Contains none."""
        levels = list_debug_levels()
        assert "none" in levels


class TestListVisualizationTypes:
    """Tests for list_visualization_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_visualization_types()
        assert types == sorted(types)

    def test_contains_gradient_flow(self) -> None:
        """Contains gradient_flow."""
        types = list_visualization_types()
        assert "gradient_flow" in types

    def test_contains_activation_histogram(self) -> None:
        """Contains activation_histogram."""
        types = list_visualization_types()
        assert "activation_histogram" in types

    def test_contains_attention_weights(self) -> None:
        """Contains attention_weights."""
        types = list_visualization_types()
        assert "attention_weights" in types

    def test_contains_loss_landscape(self) -> None:
        """Contains loss_landscape."""
        types = list_visualization_types()
        assert "loss_landscape" in types


class TestListAnomalyTypes:
    """Tests for list_anomaly_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_anomaly_types()
        assert types == sorted(types)

    def test_contains_nan(self) -> None:
        """Contains nan."""
        types = list_anomaly_types()
        assert "nan" in types

    def test_contains_inf(self) -> None:
        """Contains inf."""
        types = list_anomaly_types()
        assert "inf" in types

    def test_contains_exploding_gradient(self) -> None:
        """Contains exploding_gradient."""
        types = list_anomaly_types()
        assert "exploding_gradient" in types

    def test_contains_vanishing_gradient(self) -> None:
        """Contains vanishing_gradient."""
        types = list_anomaly_types()
        assert "vanishing_gradient" in types

    def test_contains_dead_neuron(self) -> None:
        """Contains dead_neuron."""
        types = list_anomaly_types()
        assert "dead_neuron" in types


class TestGetDebugLevel:
    """Tests for get_debug_level function."""

    def test_get_none(self) -> None:
        """Get none."""
        assert get_debug_level("none") == DebugLevel.NONE

    def test_get_basic(self) -> None:
        """Get basic."""
        assert get_debug_level("basic") == DebugLevel.BASIC

    def test_get_verbose(self) -> None:
        """Get verbose."""
        assert get_debug_level("verbose") == DebugLevel.VERBOSE

    def test_get_trace(self) -> None:
        """Get trace."""
        assert get_debug_level("trace") == DebugLevel.TRACE

    def test_invalid_level_raises(self) -> None:
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="debug level must be one of"):
            get_debug_level("invalid")


class TestGetVisualizationType:
    """Tests for get_visualization_type function."""

    def test_get_gradient_flow(self) -> None:
        """Get gradient_flow."""
        result = get_visualization_type("gradient_flow")
        assert result == VisualizationType.GRADIENT_FLOW

    def test_get_activation_histogram(self) -> None:
        """Get activation_histogram."""
        result = get_visualization_type("activation_histogram")
        assert result == VisualizationType.ACTIVATION_HISTOGRAM

    def test_get_attention_weights(self) -> None:
        """Get attention_weights."""
        result = get_visualization_type("attention_weights")
        assert result == VisualizationType.ATTENTION_WEIGHTS

    def test_get_loss_landscape(self) -> None:
        """Get loss_landscape."""
        result = get_visualization_type("loss_landscape")
        assert result == VisualizationType.LOSS_LANDSCAPE

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="visualization type must be one of"):
            get_visualization_type("invalid")


class TestGetAnomalyType:
    """Tests for get_anomaly_type function."""

    def test_get_nan(self) -> None:
        """Get nan."""
        assert get_anomaly_type("nan") == AnomalyType.NAN

    def test_get_inf(self) -> None:
        """Get inf."""
        assert get_anomaly_type("inf") == AnomalyType.INF

    def test_get_exploding_gradient(self) -> None:
        """Get exploding_gradient."""
        result = get_anomaly_type("exploding_gradient")
        assert result == AnomalyType.EXPLODING_GRADIENT

    def test_get_vanishing_gradient(self) -> None:
        """Get vanishing_gradient."""
        result = get_anomaly_type("vanishing_gradient")
        assert result == AnomalyType.VANISHING_GRADIENT

    def test_get_dead_neuron(self) -> None:
        """Get dead_neuron."""
        assert get_anomaly_type("dead_neuron") == AnomalyType.DEAD_NEURON

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="anomaly type must be one of"):
            get_anomaly_type("invalid")
