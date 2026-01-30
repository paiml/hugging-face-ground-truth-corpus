"""Model calibration and uncertainty estimation utilities.

This module provides utilities for model calibration including
temperature scaling, Platt scaling, isotonic regression, and
various uncertainty estimation methods.

Examples:
    >>> from hf_gtc.evaluation.calibration import CalibrationMethod, UncertaintyType
    >>> CalibrationMethod.TEMPERATURE.value
    'temperature'
    >>> UncertaintyType.ALEATORIC.value
    'aleatoric'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class CalibrationMethod(Enum):
    """Calibration methods for model probability adjustment.

    Attributes:
        TEMPERATURE: Temperature scaling (single parameter).
        PLATT: Platt scaling (logistic regression).
        ISOTONIC: Isotonic regression calibration.
        HISTOGRAM: Histogram binning calibration.
        FOCAL: Focal loss-based calibration.

    Examples:
        >>> CalibrationMethod.TEMPERATURE.value
        'temperature'
        >>> CalibrationMethod.PLATT.value
        'platt'
    """

    TEMPERATURE = "temperature"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    HISTOGRAM = "histogram"
    FOCAL = "focal"


VALID_CALIBRATION_METHODS = frozenset(m.value for m in CalibrationMethod)


class UncertaintyType(Enum):
    """Types of uncertainty in model predictions.

    Attributes:
        ALEATORIC: Data-inherent uncertainty (irreducible).
        EPISTEMIC: Model uncertainty (reducible with more data).
        PREDICTIVE: Total uncertainty (aleatoric + epistemic).

    Examples:
        >>> UncertaintyType.ALEATORIC.value
        'aleatoric'
        >>> UncertaintyType.EPISTEMIC.value
        'epistemic'
    """

    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    PREDICTIVE = "predictive"


VALID_UNCERTAINTY_TYPES = frozenset(u.value for u in UncertaintyType)


class CalibrationMetric(Enum):
    """Metrics for evaluating calibration quality.

    Attributes:
        ECE: Expected Calibration Error.
        MCE: Maximum Calibration Error.
        BRIER: Brier score (mean squared error of probabilities).
        RELIABILITY: Reliability diagram data.

    Examples:
        >>> CalibrationMetric.ECE.value
        'ece'
        >>> CalibrationMetric.BRIER.value
        'brier'
    """

    ECE = "ece"
    MCE = "mce"
    BRIER = "brier"
    RELIABILITY = "reliability"


VALID_CALIBRATION_METRICS = frozenset(m.value for m in CalibrationMetric)


@dataclass(frozen=True, slots=True)
class TemperatureConfig:
    """Configuration for temperature scaling.

    Attributes:
        initial_temp: Initial temperature value. Defaults to 1.0.
        optimize: Whether to optimize temperature. Defaults to True.
        lr: Learning rate for optimization. Defaults to 0.01.

    Examples:
        >>> config = TemperatureConfig()
        >>> config.initial_temp
        1.0
        >>> config.optimize
        True
    """

    initial_temp: float = 1.0
    optimize: bool = True
    lr: float = 0.01


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Configuration for model calibration.

    Attributes:
        method: Calibration method to use.
        temperature_config: Temperature scaling configuration (optional).
        n_bins: Number of bins for histogram-based methods. Defaults to 15.
        validate_before: Whether to validate calibration before applying.

    Examples:
        >>> config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        >>> config.method
        <CalibrationMethod.TEMPERATURE: 'temperature'>
        >>> config.n_bins
        15
    """

    method: CalibrationMethod
    temperature_config: TemperatureConfig | None = None
    n_bins: int = 15
    validate_before: bool = True


@dataclass(frozen=True, slots=True)
class UncertaintyResult:
    """Result of uncertainty estimation.

    Attributes:
        mean: Mean prediction value.
        variance: Prediction variance.
        confidence_interval: Tuple of (lower, upper) bounds.
        uncertainty_type: Type of uncertainty estimated.

    Examples:
        >>> result = UncertaintyResult(
        ...     mean=0.7,
        ...     variance=0.05,
        ...     confidence_interval=(0.5, 0.9),
        ...     uncertainty_type=UncertaintyType.PREDICTIVE,
        ... )
        >>> result.mean
        0.7
        >>> result.variance
        0.05
    """

    mean: float
    variance: float
    confidence_interval: tuple[float, float]
    uncertainty_type: UncertaintyType


@dataclass(frozen=True, slots=True)
class ReliabilityDiagram:
    """Data for a reliability diagram.

    Attributes:
        bin_confidences: Average confidence in each bin.
        bin_accuracies: Average accuracy in each bin.
        bin_counts: Number of samples in each bin.
        n_bins: Number of bins used.

    Examples:
        >>> diagram = ReliabilityDiagram(
        ...     bin_confidences=(0.1, 0.5, 0.9),
        ...     bin_accuracies=(0.15, 0.45, 0.85),
        ...     bin_counts=(100, 200, 100),
        ...     n_bins=3,
        ... )
        >>> diagram.n_bins
        3
    """

    bin_confidences: tuple[float, ...]
    bin_accuracies: tuple[float, ...]
    bin_counts: tuple[int, ...]
    n_bins: int


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    """Statistics from calibration evaluation.

    Attributes:
        ece: Expected Calibration Error.
        mce: Maximum Calibration Error.
        brier_score: Brier score.
        reliability_diagram: Reliability diagram data (optional).
        optimal_temperature: Optimal temperature found (optional).

    Examples:
        >>> stats = CalibrationStats(
        ...     ece=0.05,
        ...     mce=0.15,
        ...     brier_score=0.12,
        ...     reliability_diagram=None,
        ...     optimal_temperature=1.5,
        ... )
        >>> stats.ece
        0.05
    """

    ece: float
    mce: float
    brier_score: float
    reliability_diagram: ReliabilityDiagram | None
    optimal_temperature: float | None


def validate_temperature_config(config: TemperatureConfig) -> None:
    """Validate temperature configuration.

    Args:
        config: TemperatureConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If initial_temp is not positive.
        ValueError: If lr is not positive.

    Examples:
        >>> config = TemperatureConfig()
        >>> validate_temperature_config(config)  # No error

        >>> validate_temperature_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = TemperatureConfig(initial_temp=-1.0)
        >>> validate_temperature_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: initial_temp must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.initial_temp <= 0.0:
        msg = f"initial_temp must be positive, got {config.initial_temp}"
        raise ValueError(msg)

    if config.lr <= 0.0:
        msg = f"lr must be positive, got {config.lr}"
        raise ValueError(msg)


def validate_calibration_config(config: CalibrationConfig) -> None:
    """Validate calibration configuration.

    Args:
        config: CalibrationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If n_bins is not positive.
        ValueError: If temperature_config is invalid.

    Examples:
        >>> config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        >>> validate_calibration_config(config)  # No error

        >>> validate_calibration_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = CalibrationConfig(method=CalibrationMethod.HISTOGRAM, n_bins=0)
        >>> validate_calibration_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: n_bins must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.n_bins <= 0:
        msg = f"n_bins must be positive, got {config.n_bins}"
        raise ValueError(msg)

    if config.temperature_config is not None:
        validate_temperature_config(config.temperature_config)


def validate_uncertainty_result(result: UncertaintyResult) -> None:
    """Validate uncertainty result.

    Args:
        result: UncertaintyResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If variance is negative.
        ValueError: If confidence interval is invalid.

    Examples:
        >>> result = UncertaintyResult(
        ...     mean=0.5, variance=0.1,
        ...     confidence_interval=(0.3, 0.7),
        ...     uncertainty_type=UncertaintyType.PREDICTIVE,
        ... )
        >>> validate_uncertainty_result(result)  # No error

        >>> validate_uncertainty_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = UncertaintyResult(0.5, -0.1, (0.3, 0.7), UncertaintyType.PREDICTIVE)
        >>> validate_uncertainty_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: variance cannot be negative
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    if result.variance < 0.0:
        msg = f"variance cannot be negative, got {result.variance}"
        raise ValueError(msg)

    lower, upper = result.confidence_interval
    if lower > upper:
        msg = (
            f"confidence interval lower bound ({lower}) "
            f"cannot be greater than upper bound ({upper})"
        )
        raise ValueError(msg)


def validate_calibration_stats(stats: CalibrationStats) -> None:
    """Validate calibration statistics.

    Args:
        stats: CalibrationStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If ECE is negative.
        ValueError: If MCE is negative.
        ValueError: If Brier score is out of range.
        ValueError: If optimal_temperature is not positive.

    Examples:
        >>> stats = CalibrationStats(0.05, 0.15, 0.12, None, 1.5)
        >>> validate_calibration_stats(stats)  # No error

        >>> validate_calibration_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = CalibrationStats(-0.1, 0.15, 0.12, None, 1.5)
        >>> validate_calibration_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ece cannot be negative
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.ece < 0.0:
        msg = f"ece cannot be negative, got {stats.ece}"
        raise ValueError(msg)

    if stats.mce < 0.0:
        msg = f"mce cannot be negative, got {stats.mce}"
        raise ValueError(msg)

    if not 0.0 <= stats.brier_score <= 1.0:
        msg = f"brier_score must be between 0 and 1, got {stats.brier_score}"
        raise ValueError(msg)

    if stats.optimal_temperature is not None and stats.optimal_temperature <= 0.0:
        msg = f"optimal_temperature must be positive, got {stats.optimal_temperature}"
        raise ValueError(msg)


def create_temperature_config(
    *,
    initial_temp: float = 1.0,
    optimize: bool = True,
    lr: float = 0.01,
) -> TemperatureConfig:
    """Create and validate a temperature configuration.

    Args:
        initial_temp: Initial temperature value. Defaults to 1.0.
        optimize: Whether to optimize temperature. Defaults to True.
        lr: Learning rate for optimization. Defaults to 0.01.

    Returns:
        Validated TemperatureConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_temperature_config()
        >>> config.initial_temp
        1.0

        >>> config = create_temperature_config(initial_temp=2.0, lr=0.001)
        >>> config.initial_temp
        2.0
        >>> config.lr
        0.001
    """
    config = TemperatureConfig(
        initial_temp=initial_temp,
        optimize=optimize,
        lr=lr,
    )
    validate_temperature_config(config)
    return config


def create_calibration_config(
    method: CalibrationMethod,
    *,
    temperature_config: TemperatureConfig | None = None,
    n_bins: int = 15,
    validate_before: bool = True,
) -> CalibrationConfig:
    """Create and validate a calibration configuration.

    Args:
        method: Calibration method to use.
        temperature_config: Optional temperature config. Defaults to None.
        n_bins: Number of bins for histogram methods. Defaults to 15.
        validate_before: Whether to validate before calibration. Defaults to True.

    Returns:
        Validated CalibrationConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_calibration_config(CalibrationMethod.TEMPERATURE)
        >>> config.method
        <CalibrationMethod.TEMPERATURE: 'temperature'>

        >>> temp_cfg = create_temperature_config(initial_temp=1.5)
        >>> config = create_calibration_config(
        ...     CalibrationMethod.TEMPERATURE,
        ...     temperature_config=temp_cfg,
        ... )
        >>> config.temperature_config.initial_temp
        1.5
    """
    config = CalibrationConfig(
        method=method,
        temperature_config=temperature_config,
        n_bins=n_bins,
        validate_before=validate_before,
    )
    validate_calibration_config(config)
    return config


def create_uncertainty_result(
    mean: float,
    variance: float,
    confidence_interval: tuple[float, float],
    uncertainty_type: UncertaintyType,
) -> UncertaintyResult:
    """Create and validate an uncertainty result.

    Args:
        mean: Mean prediction value.
        variance: Prediction variance.
        confidence_interval: Tuple of (lower, upper) bounds.
        uncertainty_type: Type of uncertainty.

    Returns:
        Validated UncertaintyResult instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_uncertainty_result(
        ...     0.7, 0.05, (0.5, 0.9), UncertaintyType.PREDICTIVE
        ... )
        >>> result.mean
        0.7

        >>> result = create_uncertainty_result(
        ...     0.5, 0.0, (0.5, 0.5), UncertaintyType.ALEATORIC
        ... )
        >>> result.variance
        0.0
    """
    result = UncertaintyResult(
        mean=mean,
        variance=variance,
        confidence_interval=confidence_interval,
        uncertainty_type=uncertainty_type,
    )
    validate_uncertainty_result(result)
    return result


def list_calibration_methods() -> list[str]:
    """List all available calibration methods.

    Returns:
        Sorted list of calibration method names.

    Examples:
        >>> methods = list_calibration_methods()
        >>> "temperature" in methods
        True
        >>> "platt" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_CALIBRATION_METHODS)


def get_calibration_method(name: str) -> CalibrationMethod:
    """Get CalibrationMethod enum from string name.

    Args:
        name: Name of the calibration method.

    Returns:
        Corresponding CalibrationMethod enum value.

    Raises:
        ValueError: If name is not a valid calibration method.

    Examples:
        >>> get_calibration_method("temperature")
        <CalibrationMethod.TEMPERATURE: 'temperature'>

        >>> get_calibration_method("platt")
        <CalibrationMethod.PLATT: 'platt'>

        >>> get_calibration_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid calibration method: invalid
    """
    if name not in VALID_CALIBRATION_METHODS:
        msg = f"invalid calibration method: {name}"
        raise ValueError(msg)

    return CalibrationMethod(name)


def validate_calibration_method(name: str) -> bool:
    """Check if a calibration method name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_calibration_method("temperature")
        True
        >>> validate_calibration_method("invalid")
        False
        >>> validate_calibration_method("")
        False
    """
    return name in VALID_CALIBRATION_METHODS


def list_uncertainty_types() -> list[str]:
    """List all available uncertainty types.

    Returns:
        Sorted list of uncertainty type names.

    Examples:
        >>> types = list_uncertainty_types()
        >>> "aleatoric" in types
        True
        >>> "epistemic" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_UNCERTAINTY_TYPES)


def get_uncertainty_type(name: str) -> UncertaintyType:
    """Get UncertaintyType enum from string name.

    Args:
        name: Name of the uncertainty type.

    Returns:
        Corresponding UncertaintyType enum value.

    Raises:
        ValueError: If name is not a valid uncertainty type.

    Examples:
        >>> get_uncertainty_type("aleatoric")
        <UncertaintyType.ALEATORIC: 'aleatoric'>

        >>> get_uncertainty_type("epistemic")
        <UncertaintyType.EPISTEMIC: 'epistemic'>

        >>> get_uncertainty_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid uncertainty type: invalid
    """
    if name not in VALID_UNCERTAINTY_TYPES:
        msg = f"invalid uncertainty type: {name}"
        raise ValueError(msg)

    return UncertaintyType(name)


def validate_uncertainty_type(name: str) -> bool:
    """Check if an uncertainty type name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_uncertainty_type("aleatoric")
        True
        >>> validate_uncertainty_type("invalid")
        False
        >>> validate_uncertainty_type("")
        False
    """
    return name in VALID_UNCERTAINTY_TYPES


def list_calibration_metrics() -> list[str]:
    """List all available calibration metrics.

    Returns:
        Sorted list of calibration metric names.

    Examples:
        >>> metrics = list_calibration_metrics()
        >>> "ece" in metrics
        True
        >>> "brier" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_CALIBRATION_METRICS)


def get_calibration_metric(name: str) -> CalibrationMetric:
    """Get CalibrationMetric enum from string name.

    Args:
        name: Name of the calibration metric.

    Returns:
        Corresponding CalibrationMetric enum value.

    Raises:
        ValueError: If name is not a valid calibration metric.

    Examples:
        >>> get_calibration_metric("ece")
        <CalibrationMetric.ECE: 'ece'>

        >>> get_calibration_metric("brier")
        <CalibrationMetric.BRIER: 'brier'>

        >>> get_calibration_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid calibration metric: invalid
    """
    if name not in VALID_CALIBRATION_METRICS:
        msg = f"invalid calibration metric: {name}"
        raise ValueError(msg)

    return CalibrationMetric(name)


def validate_calibration_metric(name: str) -> bool:
    """Check if a calibration metric name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_calibration_metric("ece")
        True
        >>> validate_calibration_metric("invalid")
        False
        >>> validate_calibration_metric("")
        False
    """
    return name in VALID_CALIBRATION_METRICS


def calculate_ece(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 15,
) -> float:
    """Calculate Expected Calibration Error.

    ECE measures the difference between predicted confidence and actual accuracy
    across different confidence bins, weighted by the number of samples in each bin.

    Args:
        confidences: Model confidence scores for each prediction.
        accuracies: Actual accuracy (0 or 1) for each prediction.
        n_bins: Number of bins to use. Defaults to 15.

    Returns:
        Expected Calibration Error (lower is better).

    Raises:
        ValueError: If confidences is None or empty.
        ValueError: If accuracies is None or empty.
        ValueError: If lengths don't match.
        ValueError: If n_bins is not positive.
        ValueError: If confidences are not in [0, 1].

    Examples:
        >>> # Perfect calibration
        >>> calculate_ece([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], n_bins=2)
        0.0

        >>> # Well-calibrated (80% confidence, 80% accuracy)
        >>> ece = calculate_ece([0.8, 0.8, 0.8, 0.8, 0.8], [1, 1, 1, 1, 0], n_bins=2)
        >>> ece == 0.0
        True

        >>> calculate_ece(None, [1, 0])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: confidences cannot be None

        >>> calculate_ece([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: confidences cannot be empty
    """
    if confidences is None:
        msg = "confidences cannot be None"
        raise ValueError(msg)

    if accuracies is None:
        msg = "accuracies cannot be None"
        raise ValueError(msg)

    if len(confidences) == 0:
        msg = "confidences cannot be empty"
        raise ValueError(msg)

    if len(accuracies) == 0:
        msg = "accuracies cannot be empty"
        raise ValueError(msg)

    if len(confidences) != len(accuracies):
        msg = (
            f"confidences and accuracies must have same length, "
            f"got {len(confidences)} and {len(accuracies)}"
        )
        raise ValueError(msg)

    if n_bins <= 0:
        msg = f"n_bins must be positive, got {n_bins}"
        raise ValueError(msg)

    # Validate confidence range
    for conf in confidences:
        if not 0.0 <= conf <= 1.0:
            msg = f"confidences must be in [0, 1], got {conf}"
            raise ValueError(msg)

    n_samples = len(confidences)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    ece = 0.0
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Get samples in this bin
        bin_mask = [lower <= conf < upper for conf in confidences]
        # Handle edge case for last bin (include upper bound)
        if i == n_bins - 1:
            bin_mask = [lower <= conf <= upper for conf in confidences]

        bin_count = sum(bin_mask)
        if bin_count == 0:
            continue

        # Calculate average confidence and accuracy in bin
        bin_confidences = [c for c, m in zip(confidences, bin_mask, strict=True) if m]
        bin_accuracies = [a for a, m in zip(accuracies, bin_mask, strict=True) if m]

        avg_confidence = sum(bin_confidences) / bin_count
        avg_accuracy = sum(bin_accuracies) / bin_count

        # Add weighted absolute difference
        ece += (bin_count / n_samples) * abs(avg_accuracy - avg_confidence)

    return ece


def calculate_brier_score(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> float:
    """Calculate Brier score for probabilistic predictions.

    The Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower is better.

    Args:
        probabilities: Predicted probabilities for the positive class.
        labels: Actual labels (0 or 1).

    Returns:
        Brier score between 0 (perfect) and 1 (worst).

    Raises:
        ValueError: If probabilities is None or empty.
        ValueError: If labels is None or empty.
        ValueError: If lengths don't match.
        ValueError: If probabilities are not in [0, 1].
        ValueError: If labels are not 0 or 1.

    Examples:
        >>> # Perfect predictions
        >>> calculate_brier_score([1.0, 0.0, 1.0], [1, 0, 1])
        0.0

        >>> # Worst predictions
        >>> calculate_brier_score([0.0, 1.0], [1, 0])
        1.0

        >>> # Typical case
        >>> bs = calculate_brier_score([0.7, 0.3, 0.8], [1, 0, 1])
        >>> 0.0 < bs < 0.2
        True

        >>> calculate_brier_score(None, [1])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probabilities cannot be None
    """
    if probabilities is None:
        msg = "probabilities cannot be None"
        raise ValueError(msg)

    if labels is None:
        msg = "labels cannot be None"
        raise ValueError(msg)

    if len(probabilities) == 0:
        msg = "probabilities cannot be empty"
        raise ValueError(msg)

    if len(labels) == 0:
        msg = "labels cannot be empty"
        raise ValueError(msg)

    if len(probabilities) != len(labels):
        msg = (
            f"probabilities and labels must have same length, "
            f"got {len(probabilities)} and {len(labels)}"
        )
        raise ValueError(msg)

    # Validate inputs
    for prob in probabilities:
        if not 0.0 <= prob <= 1.0:
            msg = f"probabilities must be in [0, 1], got {prob}"
            raise ValueError(msg)

    for label in labels:
        if label not in (0, 1):
            msg = f"labels must be 0 or 1, got {label}"
            raise ValueError(msg)

    # Brier score = mean((probability - label)^2)
    squared_errors = [
        (p - lbl) ** 2 for p, lbl in zip(probabilities, labels, strict=True)
    ]
    return sum(squared_errors) / len(squared_errors)


def compute_reliability_diagram(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 15,
) -> ReliabilityDiagram:
    """Compute data for a reliability diagram.

    A reliability diagram plots predicted confidence vs actual accuracy
    across confidence bins. For a well-calibrated model, points should
    lie on the diagonal.

    Args:
        confidences: Model confidence scores for each prediction.
        accuracies: Actual accuracy (0 or 1) for each prediction.
        n_bins: Number of bins to use. Defaults to 15.

    Returns:
        ReliabilityDiagram containing bin data.

    Raises:
        ValueError: If confidences is None or empty.
        ValueError: If accuracies is None or empty.
        ValueError: If lengths don't match.
        ValueError: If n_bins is not positive.

    Examples:
        >>> diagram = compute_reliability_diagram(
        ...     [0.2, 0.3, 0.7, 0.8],
        ...     [0, 0, 1, 1],
        ...     n_bins=2,
        ... )
        >>> diagram.n_bins
        2
        >>> len(diagram.bin_confidences)
        2

        >>> compute_reliability_diagram(
        ...     None, [1]
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: confidences cannot be None
    """
    if confidences is None:
        msg = "confidences cannot be None"
        raise ValueError(msg)

    if accuracies is None:
        msg = "accuracies cannot be None"
        raise ValueError(msg)

    if len(confidences) == 0:
        msg = "confidences cannot be empty"
        raise ValueError(msg)

    if len(accuracies) == 0:
        msg = "accuracies cannot be empty"
        raise ValueError(msg)

    if len(confidences) != len(accuracies):
        msg = (
            f"confidences and accuracies must have same length, "
            f"got {len(confidences)} and {len(accuracies)}"
        )
        raise ValueError(msg)

    if n_bins <= 0:
        msg = f"n_bins must be positive, got {n_bins}"
        raise ValueError(msg)

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Get samples in this bin
        bin_mask = [lower <= conf < upper for conf in confidences]
        # Handle edge case for last bin
        if i == n_bins - 1:
            bin_mask = [lower <= conf <= upper for conf in confidences]

        bin_count = sum(bin_mask)
        bin_counts.append(bin_count)

        if bin_count == 0:
            # Use bin center as confidence, 0 as accuracy for empty bins
            bin_confidences.append((lower + upper) / 2)
            bin_accuracies.append(0.0)
        else:
            bin_conf = [c for c, m in zip(confidences, bin_mask, strict=True) if m]
            bin_acc = [a for a, m in zip(accuracies, bin_mask, strict=True) if m]
            bin_confidences.append(sum(bin_conf) / bin_count)
            bin_accuracies.append(sum(bin_acc) / bin_count)

    return ReliabilityDiagram(
        bin_confidences=tuple(bin_confidences),
        bin_accuracies=tuple(bin_accuracies),
        bin_counts=tuple(bin_counts),
        n_bins=n_bins,
    )


def optimize_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    config: TemperatureConfig | None = None,
) -> float:
    """Find optimal temperature for calibration.

    Temperature scaling divides logits by temperature before softmax.
    Higher temperature = softer probabilities (more uncertain).
    Lower temperature = sharper probabilities (more confident).

    Args:
        logits: Model logits (pre-softmax values), shape [n_samples, n_classes].
        labels: Ground truth labels.
        config: Optional temperature configuration.

    Returns:
        Optimal temperature value.

    Raises:
        ValueError: If logits is None or empty.
        ValueError: If labels is None or empty.
        ValueError: If lengths don't match.

    Examples:
        >>> # Simple case with 2 classes
        >>> logits = [[2.0, 0.5], [0.3, 2.5], [2.1, 0.2]]
        >>> labels = [0, 1, 0]
        >>> temp = optimize_temperature(logits, labels)
        >>> 0.1 <= temp <= 10.0
        True

        >>> optimize_temperature(None, [0])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: logits cannot be None
    """
    if logits is None:
        msg = "logits cannot be None"
        raise ValueError(msg)

    if labels is None:
        msg = "labels cannot be None"
        raise ValueError(msg)

    if len(logits) == 0:
        msg = "logits cannot be empty"
        raise ValueError(msg)

    if len(labels) == 0:
        msg = "labels cannot be empty"
        raise ValueError(msg)

    if len(logits) != len(labels):
        msg = (
            f"logits and labels must have same length, "
            f"got {len(logits)} and {len(labels)}"
        )
        raise ValueError(msg)

    if config is None:
        config = TemperatureConfig()

    if not config.optimize:
        return config.initial_temp

    # Simple grid search for optimal temperature
    # In production, use scipy.optimize or torch optimization
    best_temp = config.initial_temp
    best_nll = float("inf")

    for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        nll = _compute_nll_with_temperature(logits, labels, temp)
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    return best_temp


def _compute_nll_with_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
) -> float:
    """Compute negative log-likelihood with temperature scaling.

    Args:
        logits: Model logits.
        labels: Ground truth labels.
        temperature: Temperature value.

    Returns:
        Negative log-likelihood.
    """
    total_nll = 0.0

    for sample_logits, label in zip(logits, labels, strict=True):
        # Apply temperature
        scaled_logits = [logit / temperature for logit in sample_logits]

        # Softmax (numerically stable)
        max_logit = max(scaled_logits)
        exp_logits = [math.exp(logit - max_logit) for logit in scaled_logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # NLL for this sample
        prob_correct = probs[label]
        if prob_correct > 0:
            total_nll -= math.log(prob_correct)
        else:
            total_nll += 1e10  # Large penalty for zero probability

    return total_nll / len(logits)


def estimate_uncertainty(
    predictions: Sequence[float],
    *,
    uncertainty_type: UncertaintyType = UncertaintyType.PREDICTIVE,
    confidence_level: float = 0.95,
) -> UncertaintyResult:
    """Estimate prediction uncertainty.

    Args:
        predictions: Multiple predictions from model (e.g., MC Dropout samples).
        uncertainty_type: Type of uncertainty to estimate. Defaults to PREDICTIVE.
        confidence_level: Confidence level for interval. Defaults to 0.95.

    Returns:
        UncertaintyResult with mean, variance, and confidence interval.

    Raises:
        ValueError: If predictions is None or empty.
        ValueError: If confidence_level is not in (0, 1).

    Examples:
        >>> # Estimate from multiple MC Dropout predictions
        >>> preds = [0.68, 0.72, 0.65, 0.70, 0.71]
        >>> result = estimate_uncertainty(preds)
        >>> 0.65 <= result.mean <= 0.72
        True
        >>> result.variance >= 0.0
        True

        >>> estimate_uncertainty(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be None

        >>> estimate_uncertainty([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if predictions is None:
        msg = "predictions cannot be None"
        raise ValueError(msg)

    if len(predictions) == 0:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(msg)

    # Compute statistics
    n = len(predictions)
    mean = sum(predictions) / n

    variance = 0.0 if n == 1 else sum((p - mean) ** 2 for p in predictions) / (n - 1)

    # Compute confidence interval using t-distribution approximation
    # For simplicity, use normal approximation (z-score)
    # z = 1.96 for 95% CI, 2.576 for 99% CI
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)

    std_error = math.sqrt(variance / n) if n > 1 else 0.0

    margin = z * std_error
    lower = mean - margin
    upper = mean + margin

    return UncertaintyResult(
        mean=mean,
        variance=variance,
        confidence_interval=(lower, upper),
        uncertainty_type=uncertainty_type,
    )


def format_calibration_stats(stats: CalibrationStats) -> str:
    """Format calibration statistics as a human-readable string.

    Args:
        stats: CalibrationStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = CalibrationStats(0.05, 0.15, 0.12, None, 1.5)
        >>> formatted = format_calibration_stats(stats)
        >>> "ECE" in formatted
        True
        >>> "0.05" in formatted
        True

        >>> format_calibration_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        "Model Calibration Statistics",
        "=" * 30,
        f"ECE (Expected Calibration Error): {stats.ece:.4f}",
        f"MCE (Maximum Calibration Error): {stats.mce:.4f}",
        f"Brier Score: {stats.brier_score:.4f}",
    ]

    if stats.optimal_temperature is not None:
        lines.append(f"Optimal Temperature: {stats.optimal_temperature:.3f}")

    # Add interpretation
    lines.append("")
    lines.append("Interpretation:")

    if stats.ece < 0.05:
        lines.append("  - Excellent calibration (ECE < 0.05)")
    elif stats.ece < 0.10:
        lines.append("  - Good calibration (ECE < 0.10)")
    elif stats.ece < 0.15:
        lines.append("  - Moderate calibration (ECE < 0.15)")
    else:
        lines.append("  - Poor calibration (ECE >= 0.15)")

    if stats.brier_score < 0.1:
        lines.append("  - Excellent Brier score (< 0.1)")
    elif stats.brier_score < 0.2:
        lines.append("  - Good Brier score (< 0.2)")
    else:
        lines.append("  - High Brier score (>= 0.2)")

    if stats.reliability_diagram is not None:
        lines.append("")
        lines.append("Reliability Diagram:")
        for i, (conf, acc, count) in enumerate(
            zip(
                stats.reliability_diagram.bin_confidences,
                stats.reliability_diagram.bin_accuracies,
                stats.reliability_diagram.bin_counts,
                strict=True,
            )
        ):
            if count > 0:
                lines.append(f"  Bin {i}: conf={conf:.3f}, acc={acc:.3f}, n={count}")

    return "\n".join(lines)


def get_recommended_calibration_config(
    model_type: str,
    *,
    dataset_size: int | None = None,
) -> dict[str, Any]:
    """Get recommended calibration configuration for a model type.

    Args:
        model_type: Type of model (e.g., "classification", "regression", "llm").
        dataset_size: Size of calibration dataset (optional).

    Returns:
        Dictionary with recommended configuration.

    Raises:
        ValueError: If model_type is None or empty.

    Examples:
        >>> config = get_recommended_calibration_config("classification")
        >>> "method" in config
        True
        >>> config["method"] == CalibrationMethod.TEMPERATURE
        True

        >>> config = get_recommended_calibration_config("llm")
        >>> config["method"] == CalibrationMethod.TEMPERATURE
        True

        >>> get_recommended_calibration_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty

        >>> config = get_recommended_calibration_config(None)  # doctest: +SKIP
    """
    if model_type is None:
        msg = "model_type cannot be None"
        raise ValueError(msg)

    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    # Base configuration
    base_config: dict[str, Any] = {
        "method": CalibrationMethod.TEMPERATURE,
        "n_bins": 15,
        "validate_before": True,
        "temperature_config": {
            "initial_temp": 1.0,
            "optimize": True,
            "lr": 0.01,
        },
    }

    # Model-specific configurations
    model_configs: dict[str, dict[str, Any]] = {
        "classification": {
            "method": CalibrationMethod.TEMPERATURE,
            "n_bins": 15,
            "validate_before": True,
            "temperature_config": {
                "initial_temp": 1.0,
                "optimize": True,
                "lr": 0.01,
            },
            "metrics": [CalibrationMetric.ECE, CalibrationMetric.BRIER],
        },
        "multi_class": {
            "method": CalibrationMethod.TEMPERATURE,
            "n_bins": 20,
            "validate_before": True,
            "temperature_config": {
                "initial_temp": 1.5,
                "optimize": True,
                "lr": 0.005,
            },
            "metrics": [CalibrationMetric.ECE, CalibrationMetric.MCE],
        },
        "regression": {
            "method": CalibrationMethod.ISOTONIC,
            "n_bins": 10,
            "validate_before": True,
            "metrics": [CalibrationMetric.BRIER],
        },
        "llm": {
            "method": CalibrationMethod.TEMPERATURE,
            "n_bins": 10,
            "validate_before": True,
            "temperature_config": {
                "initial_temp": 1.0,
                "optimize": True,
                "lr": 0.01,
            },
            "metrics": [CalibrationMetric.ECE],
            "uncertainty_type": UncertaintyType.PREDICTIVE,
        },
        "medical": {
            "method": CalibrationMethod.PLATT,
            "n_bins": 20,
            "validate_before": True,
            "metrics": [
                CalibrationMetric.ECE,
                CalibrationMetric.MCE,
                CalibrationMetric.BRIER,
                CalibrationMetric.RELIABILITY,
            ],
        },
    }

    # Get model-specific config or base
    config = model_configs.get(model_type.lower(), base_config)

    # Adjust for dataset size
    if dataset_size is not None:
        if dataset_size < 1000:
            config["n_bins"] = min(config.get("n_bins", 15), 10)
        elif dataset_size > 10000:
            config["n_bins"] = max(config.get("n_bins", 15), 20)

    return config
