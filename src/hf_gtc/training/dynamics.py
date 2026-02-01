"""Training dynamics and loss curve analysis utilities.

This module provides utilities for analyzing training dynamics including
loss curve analysis, gradient statistics, convergence detection, and
identification of common training issues.

Examples:
    >>> from hf_gtc.training.dynamics import create_loss_curve
    >>> curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
    >>> curve.trend
    <TrendType.DECREASING: 'decreasing'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class DynamicsMetric(Enum):
    """Metrics for analyzing training dynamics.

    Attributes:
        LOSS: Training or validation loss.
        GRADIENT_NORM: L2 norm of gradients.
        LEARNING_RATE: Current learning rate value.
        WEIGHT_NORM: L2 norm of model weights.
        ACTIVATION_NORM: L2 norm of layer activations.

    Examples:
        >>> DynamicsMetric.LOSS.value
        'loss'
        >>> DynamicsMetric.GRADIENT_NORM.value
        'gradient_norm'
        >>> DynamicsMetric.LEARNING_RATE.value
        'learning_rate'
        >>> DynamicsMetric.WEIGHT_NORM.value
        'weight_norm'
        >>> DynamicsMetric.ACTIVATION_NORM.value
        'activation_norm'
    """

    LOSS = "loss"
    GRADIENT_NORM = "gradient_norm"
    LEARNING_RATE = "learning_rate"
    WEIGHT_NORM = "weight_norm"
    ACTIVATION_NORM = "activation_norm"


VALID_DYNAMICS_METRICS = frozenset(m.value for m in DynamicsMetric)


class TrendType(Enum):
    """Types of trends in training metrics.

    Attributes:
        DECREASING: Consistently decreasing values.
        INCREASING: Consistently increasing values.
        PLATEAU: Values have stabilized with minimal change.
        OSCILLATING: Values fluctuating around a mean.
        DIVERGING: Values increasing without bound (unstable).

    Examples:
        >>> TrendType.DECREASING.value
        'decreasing'
        >>> TrendType.INCREASING.value
        'increasing'
        >>> TrendType.PLATEAU.value
        'plateau'
        >>> TrendType.OSCILLATING.value
        'oscillating'
        >>> TrendType.DIVERGING.value
        'diverging'
    """

    DECREASING = "decreasing"
    INCREASING = "increasing"
    PLATEAU = "plateau"
    OSCILLATING = "oscillating"
    DIVERGING = "diverging"


VALID_TREND_TYPES = frozenset(t.value for t in TrendType)


class AnalysisWindow(Enum):
    """Window types for training dynamics analysis.

    Attributes:
        GLOBAL: Analyze across entire training history.
        RECENT: Analyze only recent steps/epochs.
        EPOCH: Analyze within single epochs.
        STEP: Analyze at step granularity.

    Examples:
        >>> AnalysisWindow.GLOBAL.value
        'global'
        >>> AnalysisWindow.RECENT.value
        'recent'
        >>> AnalysisWindow.EPOCH.value
        'epoch'
        >>> AnalysisWindow.STEP.value
        'step'
    """

    GLOBAL = "global"
    RECENT = "recent"
    EPOCH = "epoch"
    STEP = "step"


VALID_ANALYSIS_WINDOWS = frozenset(w.value for w in AnalysisWindow)


@dataclass(frozen=True, slots=True)
class LossCurve:
    """Represents a loss curve with analysis.

    Attributes:
        steps: Training steps where loss was recorded.
        values: Raw loss values at each step.
        smoothed_values: Smoothed loss values using exponential moving average.
        trend: Detected trend in the loss curve.

    Examples:
        >>> curve = LossCurve(
        ...     steps=(0, 1, 2),
        ...     values=(1.0, 0.5, 0.3),
        ...     smoothed_values=(1.0, 0.75, 0.525),
        ...     trend=TrendType.DECREASING,
        ... )
        >>> curve.steps
        (0, 1, 2)
        >>> curve.trend
        <TrendType.DECREASING: 'decreasing'>
    """

    steps: tuple[int, ...]
    values: tuple[float, ...]
    smoothed_values: tuple[float, ...]
    trend: TrendType


@dataclass(frozen=True, slots=True)
class GradientDynamics:
    """Statistics about gradient behavior during training.

    Attributes:
        norms: Gradient norms at each recorded step.
        layer_norms: Mapping of layer names to their gradient norms.
        max_norm: Maximum gradient norm observed.
        vanishing_layers: Layers with vanishing gradients.
        exploding_layers: Layers with exploding gradients.

    Examples:
        >>> dynamics = GradientDynamics(
        ...     norms=(0.5, 0.4, 0.3),
        ...     layer_norms={"layer1": (0.5, 0.4, 0.3)},
        ...     max_norm=0.5,
        ...     vanishing_layers=(),
        ...     exploding_layers=(),
        ... )
        >>> dynamics.max_norm
        0.5
        >>> len(dynamics.vanishing_layers)
        0
    """

    norms: tuple[float, ...]
    layer_norms: dict[str, tuple[float, ...]]
    max_norm: float
    vanishing_layers: tuple[str, ...]
    exploding_layers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """A snapshot of training state at a specific step.

    Attributes:
        step: Current training step.
        loss: Loss value at this step.
        gradient_norm: Gradient norm at this step.
        learning_rate: Learning rate at this step.
        weight_norms: Weight norms per layer.

    Examples:
        >>> snapshot = TrainingSnapshot(
        ...     step=100,
        ...     loss=0.5,
        ...     gradient_norm=0.1,
        ...     learning_rate=1e-4,
        ...     weight_norms={"layer1": 1.0},
        ... )
        >>> snapshot.step
        100
        >>> snapshot.loss
        0.5
    """

    step: int
    loss: float
    gradient_norm: float
    learning_rate: float
    weight_norms: dict[str, float]


@dataclass(frozen=True, slots=True)
class DynamicsStats:
    """Statistics summarizing training dynamics.

    Attributes:
        convergence_rate: Rate of convergence (loss decrease per step).
        stability_score: Score indicating training stability (0-1).
        oscillation_frequency: Frequency of loss oscillations.
        plateau_steps: Number of steps in plateau regions.

    Examples:
        >>> stats = DynamicsStats(
        ...     convergence_rate=0.01,
        ...     stability_score=0.9,
        ...     oscillation_frequency=0.1,
        ...     plateau_steps=0,
        ... )
        >>> stats.convergence_rate
        0.01
        >>> stats.stability_score
        0.9
    """

    convergence_rate: float
    stability_score: float
    oscillation_frequency: float
    plateau_steps: int


def validate_loss_curve(curve: LossCurve) -> None:
    """Validate a loss curve.

    Args:
        curve: LossCurve to validate.

    Raises:
        ValueError: If curve is None.
        ValueError: If steps and values have different lengths.
        ValueError: If steps and smoothed_values have different lengths.
        ValueError: If steps is empty.

    Examples:
        >>> curve = LossCurve(
        ...     steps=(0, 1, 2),
        ...     values=(1.0, 0.5, 0.3),
        ...     smoothed_values=(1.0, 0.75, 0.525),
        ...     trend=TrendType.DECREASING,
        ... )
        >>> validate_loss_curve(curve)  # No error

        >>> validate_loss_curve(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: curve cannot be None
    """
    if curve is None:
        msg = "curve cannot be None"
        raise ValueError(msg)

    if len(curve.steps) != len(curve.values):
        msg = (
            f"steps and values must have same length, "
            f"got {len(curve.steps)} and {len(curve.values)}"
        )
        raise ValueError(msg)

    if len(curve.steps) != len(curve.smoothed_values):
        msg = (
            f"steps and smoothed_values must have same length, "
            f"got {len(curve.steps)} and {len(curve.smoothed_values)}"
        )
        raise ValueError(msg)

    if len(curve.steps) == 0:
        msg = "steps cannot be empty"
        raise ValueError(msg)


def validate_gradient_dynamics(dynamics: GradientDynamics) -> None:
    """Validate gradient dynamics.

    Args:
        dynamics: GradientDynamics to validate.

    Raises:
        ValueError: If dynamics is None.
        ValueError: If max_norm is negative.

    Examples:
        >>> dynamics = GradientDynamics(
        ...     norms=(0.5, 0.4, 0.3),
        ...     layer_norms={},
        ...     max_norm=0.5,
        ...     vanishing_layers=(),
        ...     exploding_layers=(),
        ... )
        >>> validate_gradient_dynamics(dynamics)  # No error

        >>> validate_gradient_dynamics(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dynamics cannot be None
    """
    if dynamics is None:
        msg = "dynamics cannot be None"
        raise ValueError(msg)

    if dynamics.max_norm < 0:
        msg = f"max_norm cannot be negative, got {dynamics.max_norm}"
        raise ValueError(msg)


def validate_training_snapshot(snapshot: TrainingSnapshot) -> None:
    """Validate a training snapshot.

    Args:
        snapshot: TrainingSnapshot to validate.

    Raises:
        ValueError: If snapshot is None.
        ValueError: If step is negative.
        ValueError: If gradient_norm is negative.
        ValueError: If learning_rate is not positive.

    Examples:
        >>> snapshot = TrainingSnapshot(
        ...     step=100,
        ...     loss=0.5,
        ...     gradient_norm=0.1,
        ...     learning_rate=1e-4,
        ...     weight_norms={},
        ... )
        >>> validate_training_snapshot(snapshot)  # No error

        >>> validate_training_snapshot(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: snapshot cannot be None
    """
    if snapshot is None:
        msg = "snapshot cannot be None"
        raise ValueError(msg)

    if snapshot.step < 0:
        msg = f"step cannot be negative, got {snapshot.step}"
        raise ValueError(msg)

    if snapshot.gradient_norm < 0:
        msg = f"gradient_norm cannot be negative, got {snapshot.gradient_norm}"
        raise ValueError(msg)

    if snapshot.learning_rate <= 0:
        msg = f"learning_rate must be positive, got {snapshot.learning_rate}"
        raise ValueError(msg)


def validate_dynamics_stats(stats: DynamicsStats) -> None:
    """Validate dynamics statistics.

    Args:
        stats: DynamicsStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If stability_score is not in [0, 1].
        ValueError: If oscillation_frequency is negative.
        ValueError: If plateau_steps is negative.

    Examples:
        >>> stats = DynamicsStats(
        ...     convergence_rate=0.01,
        ...     stability_score=0.9,
        ...     oscillation_frequency=0.1,
        ...     plateau_steps=0,
        ... )
        >>> validate_dynamics_stats(stats)  # No error

        >>> validate_dynamics_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    if not 0 <= stats.stability_score <= 1:
        msg = f"stability_score must be in [0, 1], got {stats.stability_score}"
        raise ValueError(msg)

    if stats.oscillation_frequency < 0:
        msg = (
            "oscillation_frequency cannot be negative,"
            f" got {stats.oscillation_frequency}"
        )
        raise ValueError(msg)

    if stats.plateau_steps < 0:
        msg = f"plateau_steps cannot be negative, got {stats.plateau_steps}"
        raise ValueError(msg)


def create_loss_curve(
    steps: list[int],
    values: list[float],
    smoothing_factor: float = 0.9,
) -> LossCurve:
    """Create a loss curve from steps and values.

    Args:
        steps: Training steps where loss was recorded.
        values: Raw loss values at each step.
        smoothing_factor: EMA smoothing factor. Defaults to 0.9.

    Returns:
        Validated LossCurve instance.

    Raises:
        ValueError: If steps is empty.
        ValueError: If steps and values have different lengths.
        ValueError: If smoothing_factor is not in (0, 1).

    Examples:
        >>> curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        >>> curve.steps
        (0, 1, 2)
        >>> curve.trend
        <TrendType.DECREASING: 'decreasing'>

        >>> curve = create_loss_curve([0, 1, 2], [0.3, 0.5, 1.0])
        >>> curve.trend
        <TrendType.INCREASING: 'increasing'>

        >>> create_loss_curve([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: steps cannot be empty
    """
    if not steps:
        msg = "steps cannot be empty"
        raise ValueError(msg)

    if len(steps) != len(values):
        msg = (
            "steps and values must have same length,"
            f" got {len(steps)} and {len(values)}"
        )
        raise ValueError(msg)

    if not 0 < smoothing_factor < 1:
        msg = f"smoothing_factor must be in (0, 1), got {smoothing_factor}"
        raise ValueError(msg)

    smoothed = smooth_curve(values, smoothing_factor)
    trend = _detect_trend(values)

    curve = LossCurve(
        steps=tuple(steps),
        values=tuple(values),
        smoothed_values=tuple(smoothed),
        trend=trend,
    )
    validate_loss_curve(curve)
    return curve


def create_gradient_dynamics(
    norms: list[float],
    layer_norms: dict[str, list[float]] | None = None,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 100.0,
) -> GradientDynamics:
    """Create gradient dynamics from gradient norms.

    Args:
        norms: Gradient norms at each recorded step.
        layer_norms: Optional mapping of layer names to their gradient norms.
        vanishing_threshold: Threshold below which gradients are vanishing.
        exploding_threshold: Threshold above which gradients are exploding.

    Returns:
        Validated GradientDynamics instance.

    Raises:
        ValueError: If norms is empty.
        ValueError: If thresholds are invalid.

    Examples:
        >>> dynamics = create_gradient_dynamics([0.5, 0.4, 0.3])
        >>> dynamics.max_norm
        0.5
        >>> len(dynamics.vanishing_layers)
        0

        >>> dynamics = create_gradient_dynamics(
        ...     [0.5, 0.4, 0.3],
        ...     {"layer1": [1e-8, 1e-8, 1e-8]},
        ... )
        >>> "layer1" in dynamics.vanishing_layers
        True

        >>> create_gradient_dynamics([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: norms cannot be empty
    """
    if not norms:
        msg = "norms cannot be empty"
        raise ValueError(msg)

    if vanishing_threshold <= 0:
        msg = f"vanishing_threshold must be positive, got {vanishing_threshold}"
        raise ValueError(msg)

    if exploding_threshold <= 0:
        msg = f"exploding_threshold must be positive, got {exploding_threshold}"
        raise ValueError(msg)

    if layer_norms is None:
        layer_norms = {}

    max_norm = max(norms)
    vanishing_layers: list[str] = []
    exploding_layers: list[str] = []

    # Detect vanishing/exploding gradients per layer
    for layer_name, layer_norm_values in layer_norms.items():
        if not layer_norm_values:
            continue
        avg_norm = sum(layer_norm_values) / len(layer_norm_values)
        if avg_norm < vanishing_threshold:
            vanishing_layers.append(layer_name)
        elif avg_norm > exploding_threshold:
            exploding_layers.append(layer_name)

    # Convert layer_norms lists to tuples
    layer_norms_tuples = {k: tuple(v) for k, v in layer_norms.items()}

    dynamics = GradientDynamics(
        norms=tuple(norms),
        layer_norms=layer_norms_tuples,
        max_norm=max_norm,
        vanishing_layers=tuple(vanishing_layers),
        exploding_layers=tuple(exploding_layers),
    )
    validate_gradient_dynamics(dynamics)
    return dynamics


def create_training_snapshot(
    step: int,
    loss: float,
    gradient_norm: float = 0.0,
    learning_rate: float = 1e-4,
    weight_norms: dict[str, float] | None = None,
) -> TrainingSnapshot:
    """Create a training snapshot.

    Args:
        step: Current training step.
        loss: Loss value at this step.
        gradient_norm: Gradient norm at this step. Defaults to 0.0.
        learning_rate: Learning rate at this step. Defaults to 1e-4.
        weight_norms: Optional weight norms per layer.

    Returns:
        Validated TrainingSnapshot instance.

    Raises:
        ValueError: If step is negative.
        ValueError: If gradient_norm is negative.
        ValueError: If learning_rate is not positive.

    Examples:
        >>> snapshot = create_training_snapshot(100, 0.5)
        >>> snapshot.step
        100
        >>> snapshot.loss
        0.5

        >>> snapshot = create_training_snapshot(
        ...     100, 0.5, gradient_norm=0.1, learning_rate=1e-5
        ... )
        >>> snapshot.gradient_norm
        0.1

        >>> create_training_snapshot(-1, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: step cannot be negative
    """
    if weight_norms is None:
        weight_norms = {}

    snapshot = TrainingSnapshot(
        step=step,
        loss=loss,
        gradient_norm=gradient_norm,
        learning_rate=learning_rate,
        weight_norms=weight_norms,
    )
    validate_training_snapshot(snapshot)
    return snapshot


def create_dynamics_stats(
    convergence_rate: float = 0.0,
    stability_score: float = 1.0,
    oscillation_frequency: float = 0.0,
    plateau_steps: int = 0,
) -> DynamicsStats:
    """Create dynamics statistics.

    Args:
        convergence_rate: Rate of convergence. Defaults to 0.0.
        stability_score: Training stability score (0-1). Defaults to 1.0.
        oscillation_frequency: Frequency of oscillations. Defaults to 0.0.
        plateau_steps: Number of plateau steps. Defaults to 0.

    Returns:
        Validated DynamicsStats instance.

    Raises:
        ValueError: If stability_score is not in [0, 1].
        ValueError: If oscillation_frequency is negative.
        ValueError: If plateau_steps is negative.

    Examples:
        >>> stats = create_dynamics_stats()
        >>> stats.stability_score
        1.0
        >>> stats.oscillation_frequency
        0.0

        >>> stats = create_dynamics_stats(convergence_rate=0.01, stability_score=0.9)
        >>> stats.convergence_rate
        0.01

        >>> create_dynamics_stats(stability_score=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stability_score must be in [0, 1]
    """
    stats = DynamicsStats(
        convergence_rate=convergence_rate,
        stability_score=stability_score,
        oscillation_frequency=oscillation_frequency,
        plateau_steps=plateau_steps,
    )
    validate_dynamics_stats(stats)
    return stats


def smooth_curve(
    values: list[float],
    smoothing_factor: float = 0.9,
) -> list[float]:
    """Apply exponential moving average smoothing to values.

    Args:
        values: Raw values to smooth.
        smoothing_factor: EMA smoothing factor (higher = more smoothing).

    Returns:
        Smoothed values.

    Raises:
        ValueError: If values is empty.
        ValueError: If smoothing_factor is not in (0, 1).

    Examples:
        >>> smooth_curve([1.0, 0.5, 0.3], 0.9)
        [1.0, 0.95, 0.885]

        >>> smooth_curve([1.0], 0.9)
        [1.0]

        >>> smooth_curve([], 0.9)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: values cannot be empty
    """
    if not values:
        msg = "values cannot be empty"
        raise ValueError(msg)

    if not 0 < smoothing_factor < 1:
        msg = f"smoothing_factor must be in (0, 1), got {smoothing_factor}"
        raise ValueError(msg)

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_val = (
            smoothing_factor * smoothed[-1] + (1 - smoothing_factor) * values[i]
        )
        smoothed.append(smoothed_val)

    return smoothed


def _detect_trend(values: list[float], window_size: int = 10) -> TrendType:
    """Detect the trend in a sequence of values.

    Args:
        values: Sequence of values to analyze.
        window_size: Size of the window for trend analysis.

    Returns:
        Detected TrendType.
    """
    if len(values) < 2:
        return TrendType.PLATEAU

    recent = values[-min(window_size, len(values)) :]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]

    if not diffs:
        return TrendType.PLATEAU

    stats = _compute_trend_stats(recent, diffs)
    return _classify_trend(recent, stats)


def _compute_trend_stats(recent: list[float], diffs: list[float]) -> dict[str, float]:
    """Compute statistics used for trend classification."""
    decreasing = sum(1 for d in diffs if d < 0)
    increasing = sum(1 for d in diffs if d > 0)
    total = len(diffs)
    mean_val = sum(recent) / len(recent)
    variance = sum((v - mean_val) ** 2 for v in recent) / len(recent)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    avg_increase = sum(d for d in diffs if d > 0) / max(increasing, 1)
    return {
        "decreasing": decreasing,
        "increasing": increasing,
        "total": total,
        "mean_val": mean_val,
        "std_dev": std_dev,
        "avg_increase": avg_increase,
    }


def _classify_trend(recent: list[float], s: dict[str, float]) -> TrendType:
    """Classify a trend based on precomputed statistics."""
    total = s["total"]
    inc = s["increasing"]
    dec = s["decreasing"]
    mean_val = s["mean_val"]
    std_dev = s["std_dev"]

    # Check for divergence
    if len(recent) >= 3 and inc > 0.8 * total and s["avg_increase"] > mean_val * 0.1:
        return TrendType.DIVERGING

    # Check for oscillation
    if std_dev > mean_val * 0.3 and abs(dec - inc) < total * 0.3:
        return TrendType.OSCILLATING

    if dec > 0.7 * total:
        return TrendType.DECREASING
    if inc > 0.7 * total:
        return TrendType.INCREASING

    # Check for plateau
    total_change = abs(recent[-1] - recent[0])
    if mean_val != 0 and total_change / abs(mean_val) < 0.05:
        return TrendType.PLATEAU

    return TrendType.OSCILLATING


def analyze_loss_curve(
    curve: LossCurve,
    window: AnalysisWindow = AnalysisWindow.GLOBAL,
) -> DynamicsStats:
    """Analyze a loss curve to compute training dynamics statistics.

    Args:
        curve: Loss curve to analyze.
        window: Analysis window type. Defaults to GLOBAL.

    Returns:
        DynamicsStats summarizing the training dynamics.

    Raises:
        ValueError: If curve is None.

    Examples:
        >>> curve = create_loss_curve([0, 1, 2, 3, 4], [1.0, 0.8, 0.6, 0.4, 0.2])
        >>> stats = analyze_loss_curve(curve)
        >>> stats.convergence_rate > 0
        True
        >>> stats.stability_score > 0
        True

        >>> analyze_loss_curve(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: curve cannot be None
    """
    if curve is None:
        msg = "curve cannot be None"
        raise ValueError(msg)

    values = list(curve.values)

    # Select analysis window
    if window == AnalysisWindow.RECENT:
        window_size = min(100, len(values))
        values = values[-window_size:]
    elif window == AnalysisWindow.EPOCH:
        # Assume epoch size of 100 steps
        epoch_size = min(100, len(values))
        values = values[-epoch_size:]

    if len(values) < 2:
        return create_dynamics_stats()

    # Calculate convergence rate (average loss decrease per step)
    total_decrease = values[0] - values[-1]
    convergence_rate = total_decrease / len(values) if len(values) > 0 else 0.0

    # Calculate stability score based on variance
    mean_val = sum(values) / len(values)
    if mean_val != 0:
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        cv = math.sqrt(variance) / abs(mean_val)  # Coefficient of variation
        stability_score = max(0.0, min(1.0, 1.0 - cv))
    else:
        stability_score = 1.0

    # Calculate oscillation frequency
    sign_changes = 0
    for i in range(1, len(values) - 1):
        if (values[i] - values[i - 1]) * (values[i + 1] - values[i]) < 0:
            sign_changes += 1
    oscillation_frequency = sign_changes / len(values) if len(values) > 2 else 0.0

    # Count plateau steps (minimal change)
    plateau_steps = 0
    threshold = 0.001 * abs(mean_val) if mean_val != 0 else 0.001
    for i in range(1, len(values)):
        if abs(values[i] - values[i - 1]) < threshold:
            plateau_steps += 1

    return create_dynamics_stats(
        convergence_rate=convergence_rate,
        stability_score=stability_score,
        oscillation_frequency=oscillation_frequency,
        plateau_steps=plateau_steps,
    )


def detect_convergence(
    curve: LossCurve,
    threshold: float = 0.001,
    patience: int = 10,
) -> tuple[bool, int | None]:
    """Detect if training has converged.

    Args:
        curve: Loss curve to analyze.
        threshold: Minimum change to not be considered converged.
        patience: Number of steps of no improvement before declaring convergence.

    Returns:
        Tuple of (has_converged, convergence_step).

    Raises:
        ValueError: If curve is None.
        ValueError: If threshold is not positive.
        ValueError: If patience is not positive.

    Examples:
        >>> curve = create_loss_curve(
        ...     [0, 1, 2, 3, 4, 5],
        ...     [1.0, 0.5, 0.3, 0.3, 0.3, 0.3],
        ... )
        >>> converged, step = detect_convergence(curve, patience=3)
        >>> converged
        True

        >>> curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        >>> converged, step = detect_convergence(curve)
        >>> converged
        False

        >>> detect_convergence(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: curve cannot be None
    """
    if curve is None:
        msg = "curve cannot be None"
        raise ValueError(msg)

    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise ValueError(msg)

    if patience <= 0:
        msg = f"patience must be positive, got {patience}"
        raise ValueError(msg)

    values = curve.values
    if len(values) < 2:
        return False, None

    # Track consecutive steps without improvement
    no_improvement_count = 0
    best_value = values[0]

    for i in range(1, len(values)):
        if best_value - values[i] > threshold:
            best_value = values[i]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            return True, curve.steps[i - patience + 1]

    return False, None


def identify_training_issues(
    curve: LossCurve,
    dynamics: GradientDynamics | None = None,
) -> list[str]:
    """Identify potential training issues from loss curve and gradients.

    Args:
        curve: Loss curve to analyze.
        dynamics: Optional gradient dynamics for gradient-based issues.

    Returns:
        List of identified issue descriptions.

    Raises:
        ValueError: If curve is None.

    Examples:
        >>> curve = create_loss_curve([0, 1, 2], [1.0, 1.5, 2.0])
        >>> issues = identify_training_issues(curve)
        >>> any(
        ...     "diverging" in issue.lower()
        ...     or "increasing" in issue.lower()
        ...     for issue in issues
        ... )
        True

        >>> curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        >>> issues = identify_training_issues(curve)
        >>> len(issues)
        0

        >>> identify_training_issues(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: curve cannot be None
    """
    if curve is None:
        msg = "curve cannot be None"
        raise ValueError(msg)

    issues: list[str] = []

    # Check trend-based issues
    if curve.trend == TrendType.DIVERGING:
        issues.append("Loss is diverging - consider reducing learning rate")
    elif curve.trend == TrendType.INCREASING:
        issues.append("Loss is increasing - training may be unstable")
    elif curve.trend == TrendType.OSCILLATING:
        stats = analyze_loss_curve(curve)
        if stats.oscillation_frequency > 0.3:
            issues.append(
                "High loss oscillation - consider reducing learning rate or batch size"
            )

    # Check for NaN/Inf values
    if any(math.isnan(v) or math.isinf(v) for v in curve.values):
        issues.append(
            "NaN or Inf values detected in loss - training has become unstable"
        )

    # Check gradient-based issues
    if dynamics is not None:
        if dynamics.vanishing_layers:
            layers_str = ", ".join(dynamics.vanishing_layers[:3])
            if len(dynamics.vanishing_layers) > 3:
                layers_str += f" and {len(dynamics.vanishing_layers) - 3} more"
            issues.append(f"Vanishing gradients detected in: {layers_str}")

        if dynamics.exploding_layers:
            layers_str = ", ".join(dynamics.exploding_layers[:3])
            if len(dynamics.exploding_layers) > 3:
                layers_str += f" and {len(dynamics.exploding_layers) - 3} more"
            issues.append(f"Exploding gradients detected in: {layers_str}")

        # Check for gradient spikes
        if len(dynamics.norms) > 1:
            mean_norm = sum(dynamics.norms) / len(dynamics.norms)
            if dynamics.max_norm > 10 * mean_norm:
                issues.append("Gradient spikes detected - consider gradient clipping")

    return issues


def compute_gradient_statistics(
    norms: list[float],
) -> dict[str, float]:
    """Compute statistics from gradient norms.

    Args:
        norms: List of gradient norms.

    Returns:
        Dictionary with statistics (mean, std, min, max, median).

    Raises:
        ValueError: If norms is empty.

    Examples:
        >>> stats = compute_gradient_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> stats["mean"]
        3.0
        >>> stats["min"]
        1.0
        >>> stats["max"]
        5.0

        >>> compute_gradient_statistics([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: norms cannot be empty
    """
    if not norms:
        msg = "norms cannot be empty"
        raise ValueError(msg)

    n = len(norms)
    mean_val = sum(norms) / n
    variance = sum((x - mean_val) ** 2 for x in norms) / n
    std_val = math.sqrt(variance)

    sorted_norms = sorted(norms)
    if n % 2 == 0:
        median_val = (sorted_norms[n // 2 - 1] + sorted_norms[n // 2]) / 2
    else:
        median_val = sorted_norms[n // 2]

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min(norms),
        "max": max(norms),
        "median": median_val,
    }


def format_dynamics_report(
    curve: LossCurve,
    stats: DynamicsStats,
    dynamics: GradientDynamics | None = None,
) -> str:
    """Format a comprehensive training dynamics report.

    Args:
        curve: Loss curve data.
        stats: Dynamics statistics.
        dynamics: Optional gradient dynamics.

    Returns:
        Formatted report string.

    Raises:
        ValueError: If curve is None.
        ValueError: If stats is None.

    Examples:
        >>> curve = create_loss_curve([0, 1, 2], [1.0, 0.5, 0.3])
        >>> stats = create_dynamics_stats(convergence_rate=0.01, stability_score=0.9)
        >>> report = format_dynamics_report(curve, stats)
        >>> "Training Dynamics Report" in report
        True
        >>> "Convergence Rate:" in report
        True

        >>> format_dynamics_report(None, stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: curve cannot be None
    """
    if curve is None:
        msg = "curve cannot be None"
        raise ValueError(msg)

    validate_not_none(stats, "stats")

    lines = [
        "=" * 50,
        "Training Dynamics Report",
        "=" * 50,
        "",
        "Loss Curve Analysis",
        "-" * 20,
        f"  Steps: {curve.steps[0]} to {curve.steps[-1]}",
        f"  Initial Loss: {curve.values[0]:.6f}",
        f"  Final Loss: {curve.values[-1]:.6f}",
        f"  Trend: {curve.trend.value}",
        "",
        "Training Statistics",
        "-" * 20,
        f"  Convergence Rate: {stats.convergence_rate:.6f}",
        f"  Stability Score: {stats.stability_score:.2%}",
        f"  Oscillation Frequency: {stats.oscillation_frequency:.2%}",
        f"  Plateau Steps: {stats.plateau_steps}",
    ]

    if dynamics is not None:
        lines.extend(
            [
                "",
                "Gradient Statistics",
                "-" * 20,
                f"  Max Norm: {dynamics.max_norm:.6f}",
                f"  Vanishing Layers: {len(dynamics.vanishing_layers)}",
                f"  Exploding Layers: {len(dynamics.exploding_layers)}",
            ]
        )

        if dynamics.vanishing_layers:
            lines.append(f"    - {', '.join(dynamics.vanishing_layers[:5])}")
        if dynamics.exploding_layers:
            lines.append(f"    - {', '.join(dynamics.exploding_layers[:5])}")

    # Add issues
    issues = identify_training_issues(curve, dynamics)
    if issues:
        lines.extend(
            [
                "",
                "Identified Issues",
                "-" * 20,
            ]
        )
        for issue in issues:
            lines.append(f"  ! {issue}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def get_recommended_dynamics_config(
    model_size: str = "7b",
    training_type: str = "fine_tuning",
) -> dict[str, float | int | str]:
    """Get recommended configuration for training dynamics monitoring.

    Args:
        model_size: Model size string (e.g., "7b", "70b"). Defaults to "7b".
        training_type: Type of training. Defaults to "fine_tuning".

    Returns:
        Dictionary with recommended configuration values.

    Raises:
        ValueError: If training_type is invalid.

    Examples:
        >>> config = get_recommended_dynamics_config("7b", "fine_tuning")
        >>> config["smoothing_factor"]
        0.9
        >>> config["convergence_threshold"]
        0.001

        >>> config = get_recommended_dynamics_config("70b", "pretraining")
        >>> config["gradient_check_frequency"]
        100

        >>> get_recommended_dynamics_config("7b", "invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: training_type must be one of
    """
    valid_training_types = {"fine_tuning", "pretraining", "rlhf"}
    if training_type not in valid_training_types:
        msg = (
            f"training_type must be one of "
            f"{valid_training_types}, got '{training_type}'"
        )
        raise ValueError(msg)

    model_size = model_size.lower().strip()
    is_large = model_size in ("70b", "70", "175b", "175")

    # Base configuration
    config: dict[str, float | int | str] = {
        "smoothing_factor": 0.9,
        "convergence_threshold": 0.001,
        "convergence_patience": 10,
        "analysis_window": "global",
        "log_frequency": 10,
    }

    # Adjust for model size
    if is_large:
        config["gradient_check_frequency"] = 100
        config["vanishing_threshold"] = 1e-8
        config["exploding_threshold"] = 1000.0
    else:
        config["gradient_check_frequency"] = 50
        config["vanishing_threshold"] = 1e-7
        config["exploding_threshold"] = 100.0

    # Adjust for training type
    if training_type == "pretraining":
        config["convergence_patience"] = 20
        config["log_frequency"] = 100
        config["analysis_window"] = "recent"
    elif training_type == "rlhf":
        config["smoothing_factor"] = 0.95
        config["convergence_threshold"] = 0.0001
        config["convergence_patience"] = 5

    return config


def list_dynamics_metrics() -> list[str]:
    """List all available dynamics metrics.

    Returns:
        Sorted list of dynamics metric names.

    Examples:
        >>> metrics = list_dynamics_metrics()
        >>> "loss" in metrics
        True
        >>> "gradient_norm" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_DYNAMICS_METRICS)


def list_trend_types() -> list[str]:
    """List all available trend types.

    Returns:
        Sorted list of trend type names.

    Examples:
        >>> types = list_trend_types()
        >>> "decreasing" in types
        True
        >>> "increasing" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_TREND_TYPES)


def list_analysis_windows() -> list[str]:
    """List all available analysis windows.

    Returns:
        Sorted list of analysis window names.

    Examples:
        >>> windows = list_analysis_windows()
        >>> "global" in windows
        True
        >>> "recent" in windows
        True
        >>> windows == sorted(windows)
        True
    """
    return sorted(VALID_ANALYSIS_WINDOWS)


def get_dynamics_metric(name: str) -> DynamicsMetric:
    """Get dynamics metric from name.

    Args:
        name: Dynamics metric name.

    Returns:
        DynamicsMetric enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_dynamics_metric("loss")
        <DynamicsMetric.LOSS: 'loss'>

        >>> get_dynamics_metric("gradient_norm")
        <DynamicsMetric.GRADIENT_NORM: 'gradient_norm'>

        >>> get_dynamics_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dynamics metric must be one of
    """
    if name not in VALID_DYNAMICS_METRICS:
        msg = f"dynamics metric must be one of {VALID_DYNAMICS_METRICS}, got '{name}'"
        raise ValueError(msg)
    return DynamicsMetric(name)


def get_trend_type(name: str) -> TrendType:
    """Get trend type from name.

    Args:
        name: Trend type name.

    Returns:
        TrendType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_trend_type("decreasing")
        <TrendType.DECREASING: 'decreasing'>

        >>> get_trend_type("increasing")
        <TrendType.INCREASING: 'increasing'>

        >>> get_trend_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: trend type must be one of
    """
    if name not in VALID_TREND_TYPES:
        msg = f"trend type must be one of {VALID_TREND_TYPES}, got '{name}'"
        raise ValueError(msg)
    return TrendType(name)


def get_analysis_window(name: str) -> AnalysisWindow:
    """Get analysis window from name.

    Args:
        name: Analysis window name.

    Returns:
        AnalysisWindow enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_analysis_window("global")
        <AnalysisWindow.GLOBAL: 'global'>

        >>> get_analysis_window("recent")
        <AnalysisWindow.RECENT: 'recent'>

        >>> get_analysis_window("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: analysis window must be one of
    """
    if name not in VALID_ANALYSIS_WINDOWS:
        msg = f"analysis window must be one of {VALID_ANALYSIS_WINDOWS}, got '{name}'"
        raise ValueError(msg)
    return AnalysisWindow(name)
