"""Training debugging and visualization utilities.

This module provides utilities for debugging and visualizing model training,
including gradient flow analysis, activation histograms, anomaly detection,
and loss landscape analysis.

Examples:
    >>> from hf_gtc.training.debugging import create_debug_config
    >>> config = create_debug_config()
    >>> config.level
    <DebugLevel.BASIC: 'basic'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DebugLevel(Enum):
    """Debug verbosity levels.

    Attributes:
        NONE: No debugging output.
        BASIC: Basic metrics only.
        VERBOSE: Detailed debugging information.
        TRACE: Full trace with all internals.

    Examples:
        >>> DebugLevel.NONE.value
        'none'
        >>> DebugLevel.BASIC.value
        'basic'
        >>> DebugLevel.VERBOSE.value
        'verbose'
        >>> DebugLevel.TRACE.value
        'trace'
    """

    NONE = "none"
    BASIC = "basic"
    VERBOSE = "verbose"
    TRACE = "trace"


VALID_DEBUG_LEVELS = frozenset(level.value for level in DebugLevel)


class VisualizationType(Enum):
    """Types of training visualizations.

    Attributes:
        GRADIENT_FLOW: Gradient flow through layers.
        ACTIVATION_HISTOGRAM: Distribution of activations.
        ATTENTION_WEIGHTS: Attention weight patterns.
        LOSS_LANDSCAPE: Loss surface visualization.

    Examples:
        >>> VisualizationType.GRADIENT_FLOW.value
        'gradient_flow'
        >>> VisualizationType.ACTIVATION_HISTOGRAM.value
        'activation_histogram'
        >>> VisualizationType.ATTENTION_WEIGHTS.value
        'attention_weights'
        >>> VisualizationType.LOSS_LANDSCAPE.value
        'loss_landscape'
    """

    GRADIENT_FLOW = "gradient_flow"
    ACTIVATION_HISTOGRAM = "activation_histogram"
    ATTENTION_WEIGHTS = "attention_weights"
    LOSS_LANDSCAPE = "loss_landscape"


VALID_VISUALIZATION_TYPES = frozenset(vt.value for vt in VisualizationType)


class AnomalyType(Enum):
    """Types of training anomalies.

    Attributes:
        NAN: Not a Number values detected.
        INF: Infinite values detected.
        EXPLODING_GRADIENT: Gradient magnitude too large.
        VANISHING_GRADIENT: Gradient magnitude too small.
        DEAD_NEURON: Neuron output always zero.

    Examples:
        >>> AnomalyType.NAN.value
        'nan'
        >>> AnomalyType.INF.value
        'inf'
        >>> AnomalyType.EXPLODING_GRADIENT.value
        'exploding_gradient'
        >>> AnomalyType.VANISHING_GRADIENT.value
        'vanishing_gradient'
        >>> AnomalyType.DEAD_NEURON.value
        'dead_neuron'
    """

    NAN = "nan"
    INF = "inf"
    EXPLODING_GRADIENT = "exploding_gradient"
    VANISHING_GRADIENT = "vanishing_gradient"
    DEAD_NEURON = "dead_neuron"


VALID_ANOMALY_TYPES = frozenset(at.value for at in AnomalyType)


@dataclass(frozen=True, slots=True)
class DebugConfig:
    """Configuration for training debugging.

    Attributes:
        level: Debug verbosity level.
        log_gradients: Whether to log gradient statistics.
        log_activations: Whether to log activation statistics.
        check_anomalies: Whether to check for anomalies.

    Examples:
        >>> config = DebugConfig(
        ...     level=DebugLevel.BASIC,
        ...     log_gradients=True,
        ...     log_activations=False,
        ...     check_anomalies=True,
        ... )
        >>> config.level
        <DebugLevel.BASIC: 'basic'>
        >>> config.log_gradients
        True
    """

    level: DebugLevel
    log_gradients: bool
    log_activations: bool
    check_anomalies: bool


@dataclass(frozen=True, slots=True)
class GradientFlowConfig:
    """Configuration for gradient flow analysis.

    Attributes:
        layers: List of layer names to monitor.
        reduction: Reduction method ('mean', 'max', 'min').
        log_frequency: How often to log (in steps).

    Examples:
        >>> config = GradientFlowConfig(
        ...     layers=("encoder", "decoder"),
        ...     reduction="mean",
        ...     log_frequency=100,
        ... )
        >>> config.reduction
        'mean'
        >>> config.log_frequency
        100
    """

    layers: tuple[str, ...]
    reduction: str
    log_frequency: int


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """Configuration for activation analysis.

    Attributes:
        layers: List of layer names to monitor.
        num_bins: Number of histogram bins.
        track_statistics: Whether to track running statistics.

    Examples:
        >>> config = ActivationConfig(
        ...     layers=("attention", "ffn"),
        ...     num_bins=50,
        ...     track_statistics=True,
        ... )
        >>> config.num_bins
        50
        >>> config.track_statistics
        True
    """

    layers: tuple[str, ...]
    num_bins: int
    track_statistics: bool


@dataclass(frozen=True, slots=True)
class DebugStats:
    """Statistics from debugging session.

    Attributes:
        anomalies_detected: Dictionary of anomaly type to count.
        gradient_norm_history: History of gradient norms.
        activation_stats: Dictionary of layer to statistics.

    Examples:
        >>> stats = DebugStats(
        ...     anomalies_detected={"nan": 0, "inf": 1},
        ...     gradient_norm_history=(0.5, 0.6, 0.55),
        ...     activation_stats={"layer1": {"mean": 0.1, "std": 0.05}},
        ... )
        >>> stats.anomalies_detected["inf"]
        1
        >>> stats.gradient_norm_history[0]
        0.5
    """

    anomalies_detected: dict[str, int]
    gradient_norm_history: tuple[float, ...]
    activation_stats: dict[str, dict[str, float]]


def validate_debug_config(config: DebugConfig) -> None:
    """Validate debug configuration.

    Args:
        config: DebugConfig to validate.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = DebugConfig(DebugLevel.BASIC, True, False, True)
        >>> validate_debug_config(config)  # No error

        >>> validate_debug_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)


def validate_gradient_flow_config(config: GradientFlowConfig) -> None:
    """Validate gradient flow configuration.

    Args:
        config: GradientFlowConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If reduction is not valid.
        ValueError: If log_frequency is not positive.

    Examples:
        >>> config = GradientFlowConfig(("layer1",), "mean", 100)
        >>> validate_gradient_flow_config(config)  # No error

        >>> validate_gradient_flow_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = GradientFlowConfig(("layer1",), "invalid", 100)
        >>> validate_gradient_flow_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: reduction must be one of
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    valid_reductions = {"mean", "max", "min"}
    if config.reduction not in valid_reductions:
        msg = f"reduction must be one of {valid_reductions}, got '{config.reduction}'"
        raise ValueError(msg)

    if config.log_frequency <= 0:
        msg = f"log_frequency must be positive, got {config.log_frequency}"
        raise ValueError(msg)


def validate_activation_config(config: ActivationConfig) -> None:
    """Validate activation configuration.

    Args:
        config: ActivationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_bins is not positive.

    Examples:
        >>> config = ActivationConfig(("layer1",), 50, True)
        >>> validate_activation_config(config)  # No error

        >>> validate_activation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ActivationConfig(("layer1",), 0, True)
        >>> validate_activation_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_bins must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_bins <= 0:
        msg = f"num_bins must be positive, got {config.num_bins}"
        raise ValueError(msg)


def validate_debug_stats(stats: DebugStats) -> None:
    """Validate debug statistics.

    Args:
        stats: DebugStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If any anomaly count is negative.
        ValueError: If any gradient norm is negative.

    Examples:
        >>> stats = DebugStats({"nan": 0}, (0.5, 0.6), {"l1": {"mean": 0.1}})
        >>> validate_debug_stats(stats)  # No error

        >>> validate_debug_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = DebugStats({"nan": -1}, (0.5,), {})
        >>> validate_debug_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: anomaly count cannot be negative
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    for anomaly_type, count in stats.anomalies_detected.items():
        if count < 0:
            msg = f"anomaly count cannot be negative for {anomaly_type}, got {count}"
            raise ValueError(msg)

    for norm in stats.gradient_norm_history:
        if norm < 0:
            msg = f"gradient norm cannot be negative, got {norm}"
            raise ValueError(msg)


def create_debug_config(
    level: str = "basic",
    log_gradients: bool = True,
    log_activations: bool = False,
    check_anomalies: bool = True,
) -> DebugConfig:
    """Create a debug configuration.

    Args:
        level: Debug level. Defaults to "basic".
        log_gradients: Whether to log gradients. Defaults to True.
        log_activations: Whether to log activations. Defaults to False.
        check_anomalies: Whether to check anomalies. Defaults to True.

    Returns:
        Validated DebugConfig instance.

    Raises:
        ValueError: If level is invalid.

    Examples:
        >>> config = create_debug_config()
        >>> config.level
        <DebugLevel.BASIC: 'basic'>
        >>> config.log_gradients
        True

        >>> config = create_debug_config(level="verbose", log_activations=True)
        >>> config.level
        <DebugLevel.VERBOSE: 'verbose'>
        >>> config.log_activations
        True

        >>> create_debug_config(level="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: level must be one of
    """
    if level not in VALID_DEBUG_LEVELS:
        msg = f"level must be one of {VALID_DEBUG_LEVELS}, got '{level}'"
        raise ValueError(msg)

    config = DebugConfig(
        level=DebugLevel(level),
        log_gradients=log_gradients,
        log_activations=log_activations,
        check_anomalies=check_anomalies,
    )
    validate_debug_config(config)
    return config


def create_gradient_flow_config(
    layers: tuple[str, ...] | None = None,
    reduction: str = "mean",
    log_frequency: int = 100,
) -> GradientFlowConfig:
    """Create a gradient flow configuration.

    Args:
        layers: Layer names to monitor. Defaults to empty tuple.
        reduction: Reduction method. Defaults to "mean".
        log_frequency: Log frequency in steps. Defaults to 100.

    Returns:
        Validated GradientFlowConfig instance.

    Raises:
        ValueError: If reduction is invalid.
        ValueError: If log_frequency is not positive.

    Examples:
        >>> config = create_gradient_flow_config()
        >>> config.reduction
        'mean'
        >>> config.log_frequency
        100

        >>> config = create_gradient_flow_config(
        ...     layers=("encoder", "decoder"),
        ...     reduction="max",
        ...     log_frequency=50,
        ... )
        >>> config.layers
        ('encoder', 'decoder')

        >>> create_gradient_flow_config(reduction="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: reduction must be one of
    """
    if layers is None:
        layers = ()

    config = GradientFlowConfig(
        layers=layers,
        reduction=reduction,
        log_frequency=log_frequency,
    )
    validate_gradient_flow_config(config)
    return config


def create_activation_config(
    layers: tuple[str, ...] | None = None,
    num_bins: int = 50,
    track_statistics: bool = True,
) -> ActivationConfig:
    """Create an activation configuration.

    Args:
        layers: Layer names to monitor. Defaults to empty tuple.
        num_bins: Number of histogram bins. Defaults to 50.
        track_statistics: Whether to track stats. Defaults to True.

    Returns:
        Validated ActivationConfig instance.

    Raises:
        ValueError: If num_bins is not positive.

    Examples:
        >>> config = create_activation_config()
        >>> config.num_bins
        50
        >>> config.track_statistics
        True

        >>> config = create_activation_config(
        ...     layers=("attention",),
        ...     num_bins=100,
        ... )
        >>> config.layers
        ('attention',)

        >>> create_activation_config(num_bins=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_bins must be positive
    """
    if layers is None:
        layers = ()

    config = ActivationConfig(
        layers=layers,
        num_bins=num_bins,
        track_statistics=track_statistics,
    )
    validate_activation_config(config)
    return config


def create_debug_stats(
    anomalies_detected: dict[str, int] | None = None,
    gradient_norm_history: tuple[float, ...] | None = None,
    activation_stats: dict[str, dict[str, float]] | None = None,
) -> DebugStats:
    """Create debug statistics.

    Args:
        anomalies_detected: Anomaly counts. Defaults to empty dict.
        gradient_norm_history: Gradient norm history. Defaults to empty tuple.
        activation_stats: Activation stats. Defaults to empty dict.

    Returns:
        Validated DebugStats instance.

    Raises:
        ValueError: If any anomaly count is negative.
        ValueError: If any gradient norm is negative.

    Examples:
        >>> stats = create_debug_stats()
        >>> stats.anomalies_detected
        {}
        >>> stats.gradient_norm_history
        ()

        >>> stats = create_debug_stats(
        ...     anomalies_detected={"nan": 1},
        ...     gradient_norm_history=(0.5, 0.6),
        ... )
        >>> stats.anomalies_detected["nan"]
        1
    """
    if anomalies_detected is None:
        anomalies_detected = {}
    if gradient_norm_history is None:
        gradient_norm_history = ()
    if activation_stats is None:
        activation_stats = {}

    stats = DebugStats(
        anomalies_detected=anomalies_detected,
        gradient_norm_history=gradient_norm_history,
        activation_stats=activation_stats,
    )
    validate_debug_stats(stats)
    return stats


def detect_anomalies(
    values: list[float],
    exploding_threshold: float = 1000.0,
    vanishing_threshold: float = 1e-8,
) -> dict[str, int]:
    """Detect anomalies in values.

    Args:
        values: List of values to check.
        exploding_threshold: Threshold for exploding detection. Defaults to 1000.0.
        vanishing_threshold: Threshold for vanishing detection. Defaults to 1e-8.

    Returns:
        Dictionary of anomaly type to count.

    Raises:
        ValueError: If values is empty.
        ValueError: If exploding_threshold is not positive.
        ValueError: If vanishing_threshold is negative.

    Examples:
        >>> result = detect_anomalies([1.0, 2.0, 3.0])
        >>> result["nan"]
        0
        >>> result["inf"]
        0

        >>> result = detect_anomalies([float("nan"), 1.0])
        >>> result["nan"]
        1

        >>> result = detect_anomalies([float("inf"), 1.0])
        >>> result["inf"]
        1

        >>> result = detect_anomalies([2000.0, 1.0])
        >>> result["exploding_gradient"]
        1

        >>> result = detect_anomalies([1e-10, 1.0])
        >>> result["vanishing_gradient"]
        1

        >>> detect_anomalies([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: values cannot be empty
    """
    if not values:
        msg = "values cannot be empty"
        raise ValueError(msg)

    if exploding_threshold <= 0:
        msg = f"exploding_threshold must be positive, got {exploding_threshold}"
        raise ValueError(msg)

    if vanishing_threshold < 0:
        msg = f"vanishing_threshold cannot be negative, got {vanishing_threshold}"
        raise ValueError(msg)

    anomalies: dict[str, int] = {
        "nan": 0,
        "inf": 0,
        "exploding_gradient": 0,
        "vanishing_gradient": 0,
        "dead_neuron": 0,
    }

    zero_count = 0
    for v in values:
        if math.isnan(v):
            anomalies["nan"] += 1
        elif math.isinf(v):
            anomalies["inf"] += 1
        elif abs(v) > exploding_threshold:
            anomalies["exploding_gradient"] += 1
        elif 0 < abs(v) < vanishing_threshold:
            anomalies["vanishing_gradient"] += 1
        elif v == 0:
            zero_count += 1

    # If all values are zero, it's a dead neuron
    if zero_count == len(values):
        anomalies["dead_neuron"] = 1

    return anomalies


def compute_gradient_flow(
    gradients: dict[str, list[float]],
    config: GradientFlowConfig,
) -> dict[str, float]:
    """Compute gradient flow statistics for layers.

    Args:
        gradients: Dictionary of layer name to gradient values.
        config: Gradient flow configuration.

    Returns:
        Dictionary of layer name to computed statistic.

    Raises:
        ValueError: If gradients is empty.
        ValueError: If config is None.
        ValueError: If any layer has empty gradients.

    Examples:
        >>> config = GradientFlowConfig(("layer1",), "mean", 100)
        >>> grads = {"layer1": [1.0, 2.0, 3.0]}
        >>> result = compute_gradient_flow(grads, config)
        >>> result["layer1"]
        2.0

        >>> config = GradientFlowConfig(("layer1",), "max", 100)
        >>> result = compute_gradient_flow(grads, config)
        >>> result["layer1"]
        3.0

        >>> config = GradientFlowConfig(("layer1",), "min", 100)
        >>> result = compute_gradient_flow(grads, config)
        >>> result["layer1"]
        1.0

        >>> compute_gradient_flow({}, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradients cannot be empty
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    result: dict[str, float] = {}
    for layer_name, layer_grads in gradients.items():
        if not layer_grads:
            msg = f"gradients for layer '{layer_name}' cannot be empty"
            raise ValueError(msg)

        abs_grads = [abs(g) for g in layer_grads]

        if config.reduction == "mean":
            result[layer_name] = sum(abs_grads) / len(abs_grads)
        elif config.reduction == "max":
            result[layer_name] = max(abs_grads)
        elif config.reduction == "min":
            result[layer_name] = min(abs_grads)

    return result


def compute_activation_stats(
    activations: dict[str, list[float]],
    config: ActivationConfig,
) -> dict[str, dict[str, float]]:
    """Compute activation statistics for layers.

    Args:
        activations: Dictionary of layer name to activation values.
        config: Activation configuration.

    Returns:
        Dictionary of layer name to statistics dictionary.

    Raises:
        ValueError: If activations is empty.
        ValueError: If config is None.
        ValueError: If any layer has empty activations.

    Examples:
        >>> config = ActivationConfig(("layer1",), 50, True)
        >>> acts = {"layer1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        >>> result = compute_activation_stats(acts, config)
        >>> result["layer1"]["mean"]
        3.0
        >>> result["layer1"]["min"]
        1.0
        >>> result["layer1"]["max"]
        5.0

        >>> compute_activation_stats({}, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: activations cannot be empty
    """
    if not activations:
        msg = "activations cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    result: dict[str, dict[str, float]] = {}
    for layer_name, layer_acts in activations.items():
        if not layer_acts:
            msg = f"activations for layer '{layer_name}' cannot be empty"
            raise ValueError(msg)

        n = len(layer_acts)
        mean = sum(layer_acts) / n
        variance = sum((x - mean) ** 2 for x in layer_acts) / n
        std = math.sqrt(variance)

        result[layer_name] = {
            "mean": mean,
            "std": std,
            "min": min(layer_acts),
            "max": max(layer_acts),
            "sparsity": sum(1 for x in layer_acts if x == 0) / n,
        }

    return result


def analyze_loss_landscape(
    loss_values: list[float],
    step_size: float = 0.01,
) -> dict[str, float]:
    """Analyze loss landscape characteristics.

    Args:
        loss_values: List of loss values along a trajectory.
        step_size: Step size used for trajectory. Defaults to 0.01.

    Returns:
        Dictionary of landscape characteristics.

    Raises:
        ValueError: If loss_values has fewer than 2 elements.
        ValueError: If step_size is not positive.

    Examples:
        >>> losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        >>> result = analyze_loss_landscape(losses)
        >>> result["trend"]  # Decreasing trend
        -0.1
        >>> result["smoothness"] >= 0
        True

        >>> losses = [1.0, 0.5, 1.2, 0.3, 0.8]
        >>> result = analyze_loss_landscape(losses)
        >>> result["volatility"] > 0
        True

        >>> analyze_loss_landscape([1.0])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: loss_values must have at least 2 elements
    """
    if len(loss_values) < 2:
        msg = f"loss_values must have at least 2 elements, got {len(loss_values)}"
        raise ValueError(msg)

    if step_size <= 0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)

    # Calculate gradients (first derivatives)
    gradients = []
    for i in range(len(loss_values) - 1):
        grad = (loss_values[i + 1] - loss_values[i]) / step_size
        gradients.append(grad)

    # Calculate second derivatives (curvature)
    curvatures = []
    for i in range(len(gradients) - 1):
        curv = (gradients[i + 1] - gradients[i]) / step_size
        curvatures.append(curv)

    # Calculate statistics
    mean_gradient = sum(gradients) / len(gradients)
    volatility = math.sqrt(
        sum((g - mean_gradient) ** 2 for g in gradients) / len(gradients)
    )

    # Smoothness: inverse of mean absolute curvature
    if curvatures:
        mean_abs_curvature = sum(abs(c) for c in curvatures) / len(curvatures)
        smoothness = 1.0 / (1.0 + mean_abs_curvature)
    else:
        smoothness = 1.0

    # Trend: average gradient (negative = improving)
    trend = round(mean_gradient * step_size, 10)

    return {
        "trend": trend,
        "volatility": volatility,
        "smoothness": smoothness,
        "min_loss": min(loss_values),
        "max_loss": max(loss_values),
        "final_loss": loss_values[-1],
    }


def diagnose_training_issues(
    gradient_norms: list[float],
    loss_values: list[float],
    learning_rate: float = 1e-4,
) -> list[str]:
    """Diagnose potential training issues.

    Args:
        gradient_norms: History of gradient norms.
        loss_values: History of loss values.
        learning_rate: Current learning rate. Defaults to 1e-4.

    Returns:
        List of diagnostic messages.

    Raises:
        ValueError: If gradient_norms is empty.
        ValueError: If loss_values is empty.
        ValueError: If learning_rate is not positive.

    Examples:
        >>> norms = [1.0, 1.1, 1.0, 0.9, 1.0]
        >>> losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        >>> issues = diagnose_training_issues(norms, losses)
        >>> len(issues)
        0

        >>> norms = [1.0, 10.0, 100.0, 1000.0]
        >>> losses = [1.0, 2.0, 5.0, 10.0]
        >>> issues = diagnose_training_issues(norms, losses)
        >>> any("exploding" in issue.lower() for issue in issues)
        True

        >>> norms = [1.0, 0.1, 0.01, 0.001]
        >>> losses = [1.0, 1.0, 1.0, 1.0]
        >>> issues = diagnose_training_issues(norms, losses)
        >>> any("vanishing" in issue.lower() for issue in issues)
        True

        >>> diagnose_training_issues([], [1.0])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradient_norms cannot be empty
    """
    if not gradient_norms:
        msg = "gradient_norms cannot be empty"
        raise ValueError(msg)

    if not loss_values:
        msg = "loss_values cannot be empty"
        raise ValueError(msg)

    if learning_rate <= 0:
        msg = f"learning_rate must be positive, got {learning_rate}"
        raise ValueError(msg)

    issues: list[str] = []

    # Check for exploding gradients
    if len(gradient_norms) >= 2:
        growth_rate = gradient_norms[-1] / (gradient_norms[0] + 1e-10)
        if growth_rate > 100:
            issues.append(
                f"Exploding gradients detected: gradient norm grew {growth_rate:.1f}x"
            )

    # Check for vanishing gradients
    if len(gradient_norms) >= 2:
        decay_rate = gradient_norms[-1] / (gradient_norms[0] + 1e-10)
        if decay_rate < 0.01:
            issues.append(
                f"Vanishing gradients detected: gradient norm decayed to "
                f"{decay_rate:.4f}x"
            )

    # Check for loss not decreasing
    if len(loss_values) >= 5:
        first_half = sum(loss_values[: len(loss_values) // 2]) / (len(loss_values) // 2)
        second_half = sum(loss_values[len(loss_values) // 2 :]) / (
            len(loss_values) - len(loss_values) // 2
        )
        if second_half >= first_half * 0.99:
            issues.append(
                "Loss not decreasing: consider adjusting learning rate or architecture"
            )

    # Check for loss instability
    if len(loss_values) >= 3:
        loss_changes = [
            abs(loss_values[i + 1] - loss_values[i])
            for i in range(len(loss_values) - 1)
        ]
        mean_change = sum(loss_changes) / len(loss_changes)
        variance = sum((c - mean_change) ** 2 for c in loss_changes) / len(loss_changes)
        if math.sqrt(variance) > mean_change * 2:
            issues.append(
                "High loss volatility: consider reducing learning rate or "
                "using gradient clipping"
            )

    # Check for potential learning rate issues
    if gradient_norms:
        mean_norm = sum(gradient_norms) / len(gradient_norms)
        if mean_norm * learning_rate > 1.0:
            issues.append(
                f"Learning rate may be too high: gradient_norm * lr = "
                f"{mean_norm * learning_rate:.4f}"
            )
        elif mean_norm * learning_rate < 1e-8:
            issues.append(
                f"Learning rate may be too low: gradient_norm * lr = "
                f"{mean_norm * learning_rate:.10f}"
            )

    return issues


def format_debug_stats(stats: DebugStats) -> str:
    """Format debug statistics for display.

    Args:
        stats: Debug statistics to format.

    Returns:
        Formatted string with statistics.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = DebugStats(
        ...     {"nan": 0, "inf": 1},
        ...     (0.5, 0.6, 0.55),
        ...     {"layer1": {"mean": 0.1}},
        ... )
        >>> formatted = format_debug_stats(stats)
        >>> "Anomalies Detected:" in formatted
        True
        >>> "Gradient Norm History:" in formatted
        True

        >>> format_debug_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = []

    # Anomalies section
    lines.append("Anomalies Detected:")
    if stats.anomalies_detected:
        for anomaly_type, count in sorted(stats.anomalies_detected.items()):
            lines.append(f"  {anomaly_type}: {count}")
    else:
        lines.append("  None")

    # Gradient norm history
    lines.append("\nGradient Norm History:")
    if stats.gradient_norm_history:
        recent = stats.gradient_norm_history[-5:]
        lines.append(f"  Recent: {[round(n, 4) for n in recent]}")
        mean_norm = sum(stats.gradient_norm_history) / len(stats.gradient_norm_history)
        lines.append(f"  Mean: {mean_norm:.4f}")
    else:
        lines.append("  No history")

    # Activation stats
    lines.append("\nActivation Statistics:")
    if stats.activation_stats:
        for layer_name, layer_stats in sorted(stats.activation_stats.items()):
            lines.append(f"  {layer_name}:")
            for stat_name, value in sorted(layer_stats.items()):
                lines.append(f"    {stat_name}: {value:.4f}")
    else:
        lines.append("  No statistics")

    return "\n".join(lines)


def get_recommended_debug_config(
    training_phase: str = "fine_tuning",
    model_size: str = "7b",
    *,
    verbose: bool = False,
) -> DebugConfig:
    """Get recommended debug configuration.

    Args:
        training_phase: Training phase. Defaults to "fine_tuning".
        model_size: Model size string. Defaults to "7b".
        verbose: Whether to use verbose output. Defaults to False.

    Returns:
        Recommended DebugConfig.

    Raises:
        ValueError: If training_phase is invalid.

    Examples:
        >>> config = get_recommended_debug_config()
        >>> config.level
        <DebugLevel.BASIC: 'basic'>

        >>> config = get_recommended_debug_config("pretraining", verbose=True)
        >>> config.level
        <DebugLevel.VERBOSE: 'verbose'>

        >>> config = get_recommended_debug_config("rlhf")
        >>> config.check_anomalies
        True

        >>> get_recommended_debug_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: training_phase must be one of
    """
    valid_phases = {"fine_tuning", "pretraining", "rlhf"}
    if training_phase not in valid_phases:
        msg = f"training_phase must be one of {valid_phases}, got '{training_phase}'"
        raise ValueError(msg)

    # Determine debug level
    if verbose:
        level = "verbose"
    elif training_phase == "pretraining":
        level = "basic"
    else:
        level = "basic"

    # RLHF typically needs more monitoring
    check_anomalies = training_phase in ("rlhf", "pretraining")
    log_gradients = training_phase != "none"

    # Larger models need activation monitoring
    model_size_lower = model_size.lower().strip()
    log_activations = model_size_lower in ("70b", "70", "175b", "175")

    return create_debug_config(
        level=level,
        log_gradients=log_gradients,
        log_activations=log_activations,
        check_anomalies=check_anomalies,
    )


def list_debug_levels() -> list[str]:
    """List supported debug levels.

    Returns:
        Sorted list of debug level names.

    Examples:
        >>> levels = list_debug_levels()
        >>> "basic" in levels
        True
        >>> "verbose" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_DEBUG_LEVELS)


def list_visualization_types() -> list[str]:
    """List supported visualization types.

    Returns:
        Sorted list of visualization type names.

    Examples:
        >>> types = list_visualization_types()
        >>> "gradient_flow" in types
        True
        >>> "activation_histogram" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_VISUALIZATION_TYPES)


def list_anomaly_types() -> list[str]:
    """List supported anomaly types.

    Returns:
        Sorted list of anomaly type names.

    Examples:
        >>> types = list_anomaly_types()
        >>> "nan" in types
        True
        >>> "exploding_gradient" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ANOMALY_TYPES)


def get_debug_level(name: str) -> DebugLevel:
    """Get debug level from name.

    Args:
        name: Debug level name.

    Returns:
        DebugLevel enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_debug_level("basic")
        <DebugLevel.BASIC: 'basic'>

        >>> get_debug_level("verbose")
        <DebugLevel.VERBOSE: 'verbose'>

        >>> get_debug_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: debug level must be one of
    """
    if name not in VALID_DEBUG_LEVELS:
        msg = f"debug level must be one of {VALID_DEBUG_LEVELS}, got '{name}'"
        raise ValueError(msg)
    return DebugLevel(name)


def get_visualization_type(name: str) -> VisualizationType:
    """Get visualization type from name.

    Args:
        name: Visualization type name.

    Returns:
        VisualizationType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_visualization_type("gradient_flow")
        <VisualizationType.GRADIENT_FLOW: 'gradient_flow'>

        >>> get_visualization_type("loss_landscape")
        <VisualizationType.LOSS_LANDSCAPE: 'loss_landscape'>

        >>> get_visualization_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: visualization type must be one of
    """
    if name not in VALID_VISUALIZATION_TYPES:
        msg = (
            f"visualization type must be one of {VALID_VISUALIZATION_TYPES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return VisualizationType(name)


def get_anomaly_type(name: str) -> AnomalyType:
    """Get anomaly type from name.

    Args:
        name: Anomaly type name.

    Returns:
        AnomalyType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_anomaly_type("nan")
        <AnomalyType.NAN: 'nan'>

        >>> get_anomaly_type("exploding_gradient")
        <AnomalyType.EXPLODING_GRADIENT: 'exploding_gradient'>

        >>> get_anomaly_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: anomaly type must be one of
    """
    if name not in VALID_ANOMALY_TYPES:
        msg = f"anomaly type must be one of {VALID_ANOMALY_TYPES}, got '{name}'"
        raise ValueError(msg)
    return AnomalyType(name)
