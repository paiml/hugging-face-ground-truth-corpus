"""Tests for training.dynamics module."""

from __future__ import annotations

import pytest

from hf_gtc.training.dynamics import (
    VALID_ANALYSIS_WINDOWS,
    VALID_DYNAMICS_METRICS,
    VALID_TREND_TYPES,
    AnalysisWindow,
    DynamicsMetric,
    DynamicsStats,
    GradientDynamics,
    LossCurve,
    TrainingSnapshot,
    TrendType,
    analyze_loss_curve,
    compute_gradient_statistics,
    create_dynamics_stats,
    create_gradient_dynamics,
    create_loss_curve,
    create_training_snapshot,
    detect_convergence,
    format_dynamics_report,
    get_analysis_window,
    get_dynamics_metric,
    get_recommended_dynamics_config,
    get_trend_type,
    identify_training_issues,
    list_analysis_windows,
    list_dynamics_metrics,
    list_trend_types,
    smooth_curve,
    validate_dynamics_stats,
    validate_gradient_dynamics,
    validate_loss_curve,
    validate_training_snapshot,
)


class TestDynamicsMetric:
    """Tests for DynamicsMetric enum."""

    def test_all_metrics_have_values(self) -> None:
        """All metrics have string values."""
        for metric in DynamicsMetric:
            assert isinstance(metric.value, str)

    def test_loss_value(self) -> None:
        """Loss has correct value."""
        assert DynamicsMetric.LOSS.value == "loss"

    def test_gradient_norm_value(self) -> None:
        """Gradient norm has correct value."""
        assert DynamicsMetric.GRADIENT_NORM.value == "gradient_norm"

    def test_learning_rate_value(self) -> None:
        """Learning rate has correct value."""
        assert DynamicsMetric.LEARNING_RATE.value == "learning_rate"

    def test_weight_norm_value(self) -> None:
        """Weight norm has correct value."""
        assert DynamicsMetric.WEIGHT_NORM.value == "weight_norm"

    def test_activation_norm_value(self) -> None:
        """Activation norm has correct value."""
        assert DynamicsMetric.ACTIVATION_NORM.value == "activation_norm"

    def test_valid_dynamics_metrics_frozenset(self) -> None:
        """VALID_DYNAMICS_METRICS is a frozenset."""
        assert isinstance(VALID_DYNAMICS_METRICS, frozenset)
        assert len(VALID_DYNAMICS_METRICS) == 5


class TestTrendType:
    """Tests for TrendType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for trend in TrendType:
            assert isinstance(trend.value, str)

    def test_decreasing_value(self) -> None:
        """Decreasing has correct value."""
        assert TrendType.DECREASING.value == "decreasing"

    def test_increasing_value(self) -> None:
        """Increasing has correct value."""
        assert TrendType.INCREASING.value == "increasing"

    def test_plateau_value(self) -> None:
        """Plateau has correct value."""
        assert TrendType.PLATEAU.value == "plateau"

    def test_oscillating_value(self) -> None:
        """Oscillating has correct value."""
        assert TrendType.OSCILLATING.value == "oscillating"

    def test_diverging_value(self) -> None:
        """Diverging has correct value."""
        assert TrendType.DIVERGING.value == "diverging"

    def test_valid_trend_types_frozenset(self) -> None:
        """VALID_TREND_TYPES is a frozenset."""
        assert isinstance(VALID_TREND_TYPES, frozenset)
        assert len(VALID_TREND_TYPES) == 5


class TestAnalysisWindow:
    """Tests for AnalysisWindow enum."""

    def test_all_windows_have_values(self) -> None:
        """All windows have string values."""
        for window in AnalysisWindow:
            assert isinstance(window.value, str)

    def test_global_value(self) -> None:
        """Global has correct value."""
        assert AnalysisWindow.GLOBAL.value == "global"

    def test_recent_value(self) -> None:
        """Recent has correct value."""
        assert AnalysisWindow.RECENT.value == "recent"

    def test_epoch_value(self) -> None:
        """Epoch has correct value."""
        assert AnalysisWindow.EPOCH.value == "epoch"

    def test_step_value(self) -> None:
        """Step has correct value."""
        assert AnalysisWindow.STEP.value == "step"

    def test_valid_analysis_windows_frozenset(self) -> None:
        """VALID_ANALYSIS_WINDOWS is a frozenset."""
        assert isinstance(VALID_ANALYSIS_WINDOWS, frozenset)
        assert len(VALID_ANALYSIS_WINDOWS) == 4


class TestLossCurve:
    """Tests for LossCurve dataclass."""

    def test_create_loss_curve(self) -> None:
        """Create loss curve."""
        curve = LossCurve(
            steps=(0, 1, 2),
            values=(1.0, 0.5, 0.3),
            smoothed_values=(1.0, 0.75, 0.525),
            trend=TrendType.DECREASING,
        )
        assert curve.steps == (0, 1, 2)
        assert curve.values == (1.0, 0.5, 0.3)
        assert curve.trend == TrendType.DECREASING

    def test_curve_is_frozen(self) -> None:
        """Curve is immutable."""
        curve = LossCurve(
            steps=(0, 1, 2),
            values=(1.0, 0.5, 0.3),
            smoothed_values=(1.0, 0.75, 0.525),
            trend=TrendType.DECREASING,
        )
        with pytest.raises(AttributeError):
            curve.trend = TrendType.INCREASING  # type: ignore[misc]


class TestGradientDynamics:
    """Tests for GradientDynamics dataclass."""

    def test_create_dynamics(self) -> None:
        """Create gradient dynamics."""
        dynamics = GradientDynamics(
            norms=(0.5, 0.4, 0.3),
            layer_norms={"layer1": (0.5, 0.4, 0.3)},
            max_norm=0.5,
            vanishing_layers=(),
            exploding_layers=(),
        )
        assert dynamics.norms == (0.5, 0.4, 0.3)
        assert dynamics.max_norm == pytest.approx(0.5)

    def test_dynamics_is_frozen(self) -> None:
        """Dynamics is immutable."""
        dynamics = GradientDynamics(
            norms=(0.5, 0.4, 0.3),
            layer_norms={},
            max_norm=0.5,
            vanishing_layers=(),
            exploding_layers=(),
        )
        with pytest.raises(AttributeError):
            dynamics.max_norm = 1.0  # type: ignore[misc]


class TestTrainingSnapshot:
    """Tests for TrainingSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        """Create training snapshot."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            gradient_norm=0.1,
            learning_rate=1e-4,
            weight_norms={"layer1": 1.0},
        )
        assert snapshot.step == 100
        assert snapshot.loss == pytest.approx(0.5)
        assert snapshot.gradient_norm == pytest.approx(0.1)

    def test_snapshot_is_frozen(self) -> None:
        """Snapshot is immutable."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            gradient_norm=0.1,
            learning_rate=1e-4,
            weight_norms={},
        )
        with pytest.raises(AttributeError):
            snapshot.step = 200  # type: ignore[misc]


class TestDynamicsStats:
    """Tests for DynamicsStats dataclass."""

    def test_create_stats(self) -> None:
        """Create dynamics stats."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=0.9,
            oscillation_frequency=0.1,
            plateau_steps=0,
        )
        assert stats.convergence_rate == pytest.approx(0.01)
        assert stats.stability_score == pytest.approx(0.9)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=0.9,
            oscillation_frequency=0.1,
            plateau_steps=0,
        )
        with pytest.raises(AttributeError):
            stats.convergence_rate = 0.02  # type: ignore[misc]


class TestValidateLossCurve:
    """Tests for validate_loss_curve function."""

    def test_valid_curve(self) -> None:
        """Valid curve passes validation."""
        curve = LossCurve(
            steps=(0, 1, 2),
            values=(1.0, 0.5, 0.3),
            smoothed_values=(1.0, 0.75, 0.525),
            trend=TrendType.DECREASING,
        )
        validate_loss_curve(curve)

    def test_none_curve_raises(self) -> None:
        """None curve raises ValueError."""
        with pytest.raises(ValueError, match="curve cannot be None"):
            validate_loss_curve(None)  # type: ignore[arg-type]

    def test_mismatched_steps_values_raises(self) -> None:
        """Mismatched steps and values raises ValueError."""
        curve = LossCurve(
            steps=(0, 1, 2),
            values=(1.0, 0.5),
            smoothed_values=(1.0, 0.5),
            trend=TrendType.DECREASING,
        )
        with pytest.raises(ValueError, match="steps and values must have same length"):
            validate_loss_curve(curve)

    def test_mismatched_steps_smoothed_raises(self) -> None:
        """Mismatched steps and smoothed_values raises ValueError."""
        curve = LossCurve(
            steps=(0, 1, 2),
            values=(1.0, 0.5, 0.3),
            smoothed_values=(1.0, 0.75),
            trend=TrendType.DECREASING,
        )
        with pytest.raises(
            ValueError, match="steps and smoothed_values must have same length"
        ):
            validate_loss_curve(curve)

    def test_empty_steps_raises(self) -> None:
        """Empty steps raises ValueError."""
        curve = LossCurve(
            steps=(),
            values=(),
            smoothed_values=(),
            trend=TrendType.DECREASING,
        )
        with pytest.raises(ValueError, match="steps cannot be empty"):
            validate_loss_curve(curve)


class TestValidateGradientDynamics:
    """Tests for validate_gradient_dynamics function."""

    def test_valid_dynamics(self) -> None:
        """Valid dynamics passes validation."""
        dynamics = GradientDynamics(
            norms=(0.5, 0.4, 0.3),
            layer_norms={},
            max_norm=0.5,
            vanishing_layers=(),
            exploding_layers=(),
        )
        validate_gradient_dynamics(dynamics)

    def test_none_dynamics_raises(self) -> None:
        """None dynamics raises ValueError."""
        with pytest.raises(ValueError, match="dynamics cannot be None"):
            validate_gradient_dynamics(None)  # type: ignore[arg-type]

    def test_negative_max_norm_raises(self) -> None:
        """Negative max_norm raises ValueError."""
        dynamics = GradientDynamics(
            norms=(0.5,),
            layer_norms={},
            max_norm=-0.5,
            vanishing_layers=(),
            exploding_layers=(),
        )
        with pytest.raises(ValueError, match="max_norm cannot be negative"):
            validate_gradient_dynamics(dynamics)


class TestValidateTrainingSnapshot:
    """Tests for validate_training_snapshot function."""

    def test_valid_snapshot(self) -> None:
        """Valid snapshot passes validation."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            gradient_norm=0.1,
            learning_rate=1e-4,
            weight_norms={},
        )
        validate_training_snapshot(snapshot)

    def test_none_snapshot_raises(self) -> None:
        """None snapshot raises ValueError."""
        with pytest.raises(ValueError, match="snapshot cannot be None"):
            validate_training_snapshot(None)  # type: ignore[arg-type]

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        snapshot = TrainingSnapshot(
            step=-1,
            loss=0.5,
            gradient_norm=0.1,
            learning_rate=1e-4,
            weight_norms={},
        )
        with pytest.raises(ValueError, match="step cannot be negative"):
            validate_training_snapshot(snapshot)

    def test_negative_gradient_norm_raises(self) -> None:
        """Negative gradient_norm raises ValueError."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            gradient_norm=-0.1,
            learning_rate=1e-4,
            weight_norms={},
        )
        with pytest.raises(ValueError, match="gradient_norm cannot be negative"):
            validate_training_snapshot(snapshot)

    def test_zero_learning_rate_raises(self) -> None:
        """Zero learning_rate raises ValueError."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            gradient_norm=0.1,
            learning_rate=0.0,
            weight_norms={},
        )
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_training_snapshot(snapshot)

    def test_negative_learning_rate_raises(self) -> None:
        """Negative learning_rate raises ValueError."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            gradient_norm=0.1,
            learning_rate=-1e-4,
            weight_norms={},
        )
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_training_snapshot(snapshot)


class TestValidateDynamicsStats:
    """Tests for validate_dynamics_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=0.9,
            oscillation_frequency=0.1,
            plateau_steps=0,
        )
        validate_dynamics_stats(stats)

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_dynamics_stats(None)  # type: ignore[arg-type]

    def test_stability_score_below_zero_raises(self) -> None:
        """Stability score below 0 raises ValueError."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=-0.1,
            oscillation_frequency=0.1,
            plateau_steps=0,
        )
        with pytest.raises(ValueError, match="stability_score must be in"):
            validate_dynamics_stats(stats)

    def test_stability_score_above_one_raises(self) -> None:
        """Stability score above 1 raises ValueError."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=1.5,
            oscillation_frequency=0.1,
            plateau_steps=0,
        )
        with pytest.raises(ValueError, match="stability_score must be in"):
            validate_dynamics_stats(stats)

    def test_negative_oscillation_frequency_raises(self) -> None:
        """Negative oscillation_frequency raises ValueError."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=0.9,
            oscillation_frequency=-0.1,
            plateau_steps=0,
        )
        with pytest.raises(
            ValueError, match="oscillation_frequency cannot be negative"
        ):
            validate_dynamics_stats(stats)

    def test_negative_plateau_steps_raises(self) -> None:
        """Negative plateau_steps raises ValueError."""
        stats = DynamicsStats(
            convergence_rate=0.01,
            stability_score=0.9,
            oscillation_frequency=0.1,
            plateau_steps=-1,
        )
        with pytest.raises(ValueError, match="plateau_steps cannot be negative"):
            validate_dynamics_stats(stats)


class TestCreateLossCurve:
    """Tests for create_loss_curve function."""

    def test_basic_decreasing_curve(self) -> None:
        """Create basic decreasing curve."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        assert curve.steps == (0, 1, 2)
        assert curve.values == (1.0, 0.5, 0.3)
        assert curve.trend == TrendType.DECREASING

    def test_increasing_curve(self) -> None:
        """Create increasing curve."""
        curve = create_loss_curve([0, 1, 2], [0.5, 0.52, 0.54])
        assert curve.trend == TrendType.INCREASING

    def test_custom_smoothing_factor(self) -> None:
        """Create curve with custom smoothing."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3], smoothing_factor=0.5)
        assert len(curve.smoothed_values) == 3

    def test_empty_steps_raises(self) -> None:
        """Empty steps raises ValueError."""
        with pytest.raises(ValueError, match="steps cannot be empty"):
            create_loss_curve([], [])

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="steps and values must have same length"):
            create_loss_curve([0, 1, 2], [1.0, 0.5])

    def test_invalid_smoothing_factor_zero_raises(self) -> None:
        """Zero smoothing factor raises ValueError."""
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3], smoothing_factor=0.0)

    def test_invalid_smoothing_factor_one_raises(self) -> None:
        """One smoothing factor raises ValueError."""
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3], smoothing_factor=1.0)


class TestCreateGradientDynamics:
    """Tests for create_gradient_dynamics function."""

    def test_basic_dynamics(self) -> None:
        """Create basic gradient dynamics."""
        dynamics = create_gradient_dynamics([0.5, 0.4, 0.3])
        assert dynamics.norms == (0.5, 0.4, 0.3)
        assert dynamics.max_norm == pytest.approx(0.5)

    def test_with_layer_norms(self) -> None:
        """Create dynamics with layer norms."""
        dynamics = create_gradient_dynamics(
            [0.5, 0.4, 0.3],
            {"layer1": [0.5, 0.4, 0.3]},
        )
        assert "layer1" in dynamics.layer_norms

    def test_detect_vanishing_gradients(self) -> None:
        """Detect vanishing gradients."""
        dynamics = create_gradient_dynamics(
            [0.5, 0.4, 0.3],
            {"layer1": [1e-8, 1e-8, 1e-8]},
        )
        assert "layer1" in dynamics.vanishing_layers

    def test_detect_exploding_gradients(self) -> None:
        """Detect exploding gradients."""
        dynamics = create_gradient_dynamics(
            [0.5, 0.4, 0.3],
            {"layer1": [200.0, 200.0, 200.0]},
        )
        assert "layer1" in dynamics.exploding_layers

    def test_custom_thresholds(self) -> None:
        """Create dynamics with custom thresholds."""
        dynamics = create_gradient_dynamics(
            [0.5, 0.4, 0.3],
            {"layer1": [0.001, 0.001, 0.001]},
            vanishing_threshold=0.01,
        )
        assert "layer1" in dynamics.vanishing_layers

    def test_empty_norms_raises(self) -> None:
        """Empty norms raises ValueError."""
        with pytest.raises(ValueError, match="norms cannot be empty"):
            create_gradient_dynamics([])

    def test_invalid_vanishing_threshold_raises(self) -> None:
        """Invalid vanishing threshold raises ValueError."""
        with pytest.raises(ValueError, match="vanishing_threshold must be positive"):
            create_gradient_dynamics([0.5], vanishing_threshold=0)

    def test_invalid_exploding_threshold_raises(self) -> None:
        """Invalid exploding threshold raises ValueError."""
        with pytest.raises(ValueError, match="exploding_threshold must be positive"):
            create_gradient_dynamics([0.5], exploding_threshold=0)


class TestCreateTrainingSnapshot:
    """Tests for create_training_snapshot function."""

    def test_basic_snapshot(self) -> None:
        """Create basic snapshot."""
        snapshot = create_training_snapshot(100, 0.5)
        assert snapshot.step == 100
        assert snapshot.loss == pytest.approx(0.5)
        assert snapshot.gradient_norm == pytest.approx(0.0)
        assert snapshot.learning_rate == 1e-4

    def test_full_snapshot(self) -> None:
        """Create snapshot with all parameters."""
        snapshot = create_training_snapshot(
            100,
            0.5,
            gradient_norm=0.1,
            learning_rate=1e-5,
            weight_norms={"layer1": 1.0},
        )
        assert snapshot.gradient_norm == pytest.approx(0.1)
        assert snapshot.learning_rate == 1e-5
        assert snapshot.weight_norms == {"layer1": 1.0}

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        with pytest.raises(ValueError, match="step cannot be negative"):
            create_training_snapshot(-1, 0.5)

    def test_negative_gradient_norm_raises(self) -> None:
        """Negative gradient_norm raises ValueError."""
        with pytest.raises(ValueError, match="gradient_norm cannot be negative"):
            create_training_snapshot(100, 0.5, gradient_norm=-0.1)

    def test_zero_learning_rate_raises(self) -> None:
        """Zero learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            create_training_snapshot(100, 0.5, learning_rate=0.0)


class TestCreateDynamicsStats:
    """Tests for create_dynamics_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_dynamics_stats()
        assert stats.convergence_rate == pytest.approx(0.0)
        assert stats.stability_score == pytest.approx(1.0)
        assert stats.oscillation_frequency == pytest.approx(0.0)
        assert stats.plateau_steps == 0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_dynamics_stats(
            convergence_rate=0.01,
            stability_score=0.9,
            oscillation_frequency=0.1,
            plateau_steps=5,
        )
        assert stats.convergence_rate == pytest.approx(0.01)
        assert stats.stability_score == pytest.approx(0.9)

    def test_invalid_stability_score_raises(self) -> None:
        """Invalid stability_score raises ValueError."""
        with pytest.raises(ValueError, match="stability_score must be in"):
            create_dynamics_stats(stability_score=1.5)

    def test_negative_oscillation_frequency_raises(self) -> None:
        """Negative oscillation_frequency raises ValueError."""
        with pytest.raises(
            ValueError, match="oscillation_frequency cannot be negative"
        ):
            create_dynamics_stats(oscillation_frequency=-0.1)

    def test_negative_plateau_steps_raises(self) -> None:
        """Negative plateau_steps raises ValueError."""
        with pytest.raises(ValueError, match="plateau_steps cannot be negative"):
            create_dynamics_stats(plateau_steps=-1)


class TestSmoothCurve:
    """Tests for smooth_curve function."""

    def test_basic_smoothing(self) -> None:
        """Apply basic smoothing."""
        result = smooth_curve([1.0, 0.5, 0.3], 0.9)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.95)
        assert result[2] == pytest.approx(0.885)

    def test_single_value(self) -> None:
        """Smooth single value."""
        result = smooth_curve([1.0], 0.9)
        assert result == [1.0]

    def test_empty_values_raises(self) -> None:
        """Empty values raises ValueError."""
        with pytest.raises(ValueError, match="values cannot be empty"):
            smooth_curve([], 0.9)

    def test_invalid_smoothing_factor_zero_raises(self) -> None:
        """Zero smoothing factor raises ValueError."""
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            smooth_curve([1.0, 0.5], 0.0)

    def test_invalid_smoothing_factor_one_raises(self) -> None:
        """One smoothing factor raises ValueError."""
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            smooth_curve([1.0, 0.5], 1.0)


class TestAnalyzeLossCurve:
    """Tests for analyze_loss_curve function."""

    def test_basic_analysis(self) -> None:
        """Analyze basic curve."""
        curve = create_loss_curve([0, 1, 2, 3, 4], [1.0, 0.8, 0.6, 0.4, 0.2])
        stats = analyze_loss_curve(curve)
        assert stats.convergence_rate > 0

    def test_recent_window(self) -> None:
        """Analyze with recent window."""
        curve = create_loss_curve([0, 1, 2, 3, 4], [1.0, 0.8, 0.6, 0.4, 0.2])
        stats = analyze_loss_curve(curve, AnalysisWindow.RECENT)
        assert stats.stability_score >= 0

    def test_epoch_window(self) -> None:
        """Analyze with epoch window."""
        curve = create_loss_curve([0, 1, 2, 3, 4], [1.0, 0.8, 0.6, 0.4, 0.2])
        stats = analyze_loss_curve(curve, AnalysisWindow.EPOCH)
        assert stats.stability_score >= 0

    def test_none_curve_raises(self) -> None:
        """None curve raises ValueError."""
        with pytest.raises(ValueError, match="curve cannot be None"):
            analyze_loss_curve(None)  # type: ignore[arg-type]

    def test_single_value_curve(self) -> None:
        """Analyze single value curve."""
        curve = create_loss_curve([0], [1.0])
        stats = analyze_loss_curve(curve)
        assert stats.convergence_rate == pytest.approx(0.0)


class TestDetectConvergence:
    """Tests for detect_convergence function."""

    def test_converged_curve(self) -> None:
        """Detect converged curve."""
        curve = create_loss_curve(
            [0, 1, 2, 3, 4, 5],
            [1.0, 0.5, 0.3, 0.3, 0.3, 0.3],
        )
        converged, step = detect_convergence(curve, patience=3)
        assert converged is True
        assert step is not None

    def test_not_converged_curve(self) -> None:
        """Detect non-converged curve."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        converged, step = detect_convergence(curve)
        assert converged is False
        assert step is None

    def test_custom_threshold(self) -> None:
        """Use custom threshold."""
        curve = create_loss_curve(
            [0, 1, 2, 3, 4, 5, 6],
            [1.0, 0.99, 0.985, 0.983, 0.982, 0.981, 0.980],
        )
        converged, _ = detect_convergence(curve, threshold=0.02, patience=3)
        assert converged is True

    def test_none_curve_raises(self) -> None:
        """None curve raises ValueError."""
        with pytest.raises(ValueError, match="curve cannot be None"):
            detect_convergence(None)  # type: ignore[arg-type]

    def test_invalid_threshold_raises(self) -> None:
        """Invalid threshold raises ValueError."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        with pytest.raises(ValueError, match="threshold must be positive"):
            detect_convergence(curve, threshold=0)

    def test_invalid_patience_raises(self) -> None:
        """Invalid patience raises ValueError."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        with pytest.raises(ValueError, match="patience must be positive"):
            detect_convergence(curve, patience=0)


class TestIdentifyTrainingIssues:
    """Tests for identify_training_issues function."""

    def test_diverging_loss(self) -> None:
        """Identify diverging loss."""
        curve = create_loss_curve([0, 1, 2], [1.0, 1.5, 2.0])
        issues = identify_training_issues(curve)
        assert any(
            "diverging" in issue.lower() or "increasing" in issue.lower()
            for issue in issues
        )

    def test_no_issues(self) -> None:
        """Identify no issues for healthy curve."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        issues = identify_training_issues(curve)
        assert len(issues) == 0

    def test_nan_values(self) -> None:
        """Identify NaN values."""
        curve = LossCurve(
            steps=(0, 1, 2),
            values=(1.0, float("nan"), 0.5),
            smoothed_values=(1.0, 1.0, 1.0),
            trend=TrendType.OSCILLATING,
        )
        issues = identify_training_issues(curve)
        assert any("nan" in issue.lower() for issue in issues)

    def test_with_gradient_dynamics(self) -> None:
        """Identify gradient issues."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        dynamics = create_gradient_dynamics(
            [0.5, 0.4, 0.3],
            {"layer1": [1e-8, 1e-8, 1e-8]},
        )
        issues = identify_training_issues(curve, dynamics)
        assert any("vanishing" in issue.lower() for issue in issues)

    def test_exploding_gradients(self) -> None:
        """Identify exploding gradients."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        dynamics = create_gradient_dynamics(
            [0.5, 0.4, 0.3],
            {"layer1": [200.0, 200.0, 200.0]},
        )
        issues = identify_training_issues(curve, dynamics)
        assert any("exploding" in issue.lower() for issue in issues)

    def test_gradient_spikes(self) -> None:
        """Identify gradient spikes."""
        n = 30
        steps = list(range(n))
        values = [1.0 - 0.02 * i for i in range(n)]
        norms = [1.0] * (n - 1) + [500.0]
        curve = create_loss_curve(steps, values)
        dynamics = create_gradient_dynamics(norms)
        issues = identify_training_issues(curve, dynamics)
        assert any("spike" in issue.lower() for issue in issues)

    def test_none_curve_raises(self) -> None:
        """None curve raises ValueError."""
        with pytest.raises(ValueError, match="curve cannot be None"):
            identify_training_issues(None)  # type: ignore[arg-type]


class TestComputeGradientStatistics:
    """Tests for compute_gradient_statistics function."""

    def test_basic_statistics(self) -> None:
        """Compute basic statistics."""
        stats = compute_gradient_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["median"] == pytest.approx(3.0)

    def test_single_value(self) -> None:
        """Compute statistics for single value."""
        stats = compute_gradient_statistics([5.0])
        assert stats["mean"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["median"] == pytest.approx(5.0)

    def test_even_count_median(self) -> None:
        """Compute median for even count."""
        stats = compute_gradient_statistics([1.0, 2.0, 3.0, 4.0])
        assert stats["median"] == pytest.approx(2.5)

    def test_empty_norms_raises(self) -> None:
        """Empty norms raises ValueError."""
        with pytest.raises(ValueError, match="norms cannot be empty"):
            compute_gradient_statistics([])


class TestFormatDynamicsReport:
    """Tests for format_dynamics_report function."""

    def test_basic_report(self) -> None:
        """Format basic report."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        stats = create_dynamics_stats(convergence_rate=0.01, stability_score=0.9)
        report = format_dynamics_report(curve, stats)
        assert "Training Dynamics Report" in report
        assert "Convergence Rate:" in report
        assert "Stability Score:" in report

    def test_report_with_dynamics(self) -> None:
        """Format report with gradient dynamics."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        stats = create_dynamics_stats()
        dynamics = create_gradient_dynamics([0.5, 0.4, 0.3])
        report = format_dynamics_report(curve, stats, dynamics)
        assert "Gradient Statistics" in report
        assert "Max Norm:" in report

    def test_report_with_issues(self) -> None:
        """Format report with identified issues."""
        curve = create_loss_curve([0, 1, 2], [1.0, 1.5, 2.0])
        stats = create_dynamics_stats()
        report = format_dynamics_report(curve, stats)
        assert "Identified Issues" in report

    def test_none_curve_raises(self) -> None:
        """None curve raises ValueError."""
        stats = create_dynamics_stats()
        with pytest.raises(ValueError, match="curve cannot be None"):
            format_dynamics_report(None, stats)  # type: ignore[arg-type]

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_dynamics_report(curve, None)  # type: ignore[arg-type]


class TestGetRecommendedDynamicsConfig:
    """Tests for get_recommended_dynamics_config function."""

    def test_fine_tuning_7b(self) -> None:
        """Get config for 7B fine tuning."""
        config = get_recommended_dynamics_config("7b", "fine_tuning")
        assert config["smoothing_factor"] == pytest.approx(0.9)
        assert config["convergence_threshold"] == pytest.approx(0.001)

    def test_pretraining_70b(self) -> None:
        """Get config for 70B pretraining."""
        config = get_recommended_dynamics_config("70b", "pretraining")
        assert config["gradient_check_frequency"] == 100
        assert config["analysis_window"] == "recent"

    def test_rlhf(self) -> None:
        """Get config for RLHF."""
        config = get_recommended_dynamics_config("7b", "rlhf")
        assert config["smoothing_factor"] == pytest.approx(0.95)
        assert config["convergence_threshold"] == pytest.approx(0.0001)

    def test_invalid_training_type_raises(self) -> None:
        """Invalid training type raises ValueError."""
        with pytest.raises(ValueError, match="training_type must be one of"):
            get_recommended_dynamics_config("7b", "invalid")


class TestListDynamicsMetrics:
    """Tests for list_dynamics_metrics function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        metrics = list_dynamics_metrics()
        assert metrics == sorted(metrics)

    def test_contains_loss(self) -> None:
        """Contains loss."""
        metrics = list_dynamics_metrics()
        assert "loss" in metrics

    def test_contains_gradient_norm(self) -> None:
        """Contains gradient_norm."""
        metrics = list_dynamics_metrics()
        assert "gradient_norm" in metrics


class TestListTrendTypes:
    """Tests for list_trend_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_trend_types()
        assert types == sorted(types)

    def test_contains_decreasing(self) -> None:
        """Contains decreasing."""
        types = list_trend_types()
        assert "decreasing" in types

    def test_contains_increasing(self) -> None:
        """Contains increasing."""
        types = list_trend_types()
        assert "increasing" in types


class TestListAnalysisWindows:
    """Tests for list_analysis_windows function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        windows = list_analysis_windows()
        assert windows == sorted(windows)

    def test_contains_global(self) -> None:
        """Contains global."""
        windows = list_analysis_windows()
        assert "global" in windows

    def test_contains_recent(self) -> None:
        """Contains recent."""
        windows = list_analysis_windows()
        assert "recent" in windows


class TestGetDynamicsMetric:
    """Tests for get_dynamics_metric function."""

    def test_get_loss(self) -> None:
        """Get loss."""
        assert get_dynamics_metric("loss") == DynamicsMetric.LOSS

    def test_get_gradient_norm(self) -> None:
        """Get gradient_norm."""
        assert get_dynamics_metric("gradient_norm") == DynamicsMetric.GRADIENT_NORM

    def test_get_learning_rate(self) -> None:
        """Get learning_rate."""
        assert get_dynamics_metric("learning_rate") == DynamicsMetric.LEARNING_RATE

    def test_get_weight_norm(self) -> None:
        """Get weight_norm."""
        assert get_dynamics_metric("weight_norm") == DynamicsMetric.WEIGHT_NORM

    def test_get_activation_norm(self) -> None:
        """Get activation_norm."""
        assert get_dynamics_metric("activation_norm") == DynamicsMetric.ACTIVATION_NORM

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="dynamics metric must be one of"):
            get_dynamics_metric("invalid")


class TestGetTrendType:
    """Tests for get_trend_type function."""

    def test_get_decreasing(self) -> None:
        """Get decreasing."""
        assert get_trend_type("decreasing") == TrendType.DECREASING

    def test_get_increasing(self) -> None:
        """Get increasing."""
        assert get_trend_type("increasing") == TrendType.INCREASING

    def test_get_plateau(self) -> None:
        """Get plateau."""
        assert get_trend_type("plateau") == TrendType.PLATEAU

    def test_get_oscillating(self) -> None:
        """Get oscillating."""
        assert get_trend_type("oscillating") == TrendType.OSCILLATING

    def test_get_diverging(self) -> None:
        """Get diverging."""
        assert get_trend_type("diverging") == TrendType.DIVERGING

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="trend type must be one of"):
            get_trend_type("invalid")


class TestGetAnalysisWindow:
    """Tests for get_analysis_window function."""

    def test_get_global(self) -> None:
        """Get global."""
        assert get_analysis_window("global") == AnalysisWindow.GLOBAL

    def test_get_recent(self) -> None:
        """Get recent."""
        assert get_analysis_window("recent") == AnalysisWindow.RECENT

    def test_get_epoch(self) -> None:
        """Get epoch."""
        assert get_analysis_window("epoch") == AnalysisWindow.EPOCH

    def test_get_step(self) -> None:
        """Get step."""
        assert get_analysis_window("step") == AnalysisWindow.STEP

    def test_invalid_window_raises(self) -> None:
        """Invalid window raises ValueError."""
        with pytest.raises(ValueError, match="analysis window must be one of"):
            get_analysis_window("invalid")
