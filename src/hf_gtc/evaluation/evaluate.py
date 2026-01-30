"""HuggingFace Evaluate library utilities.

This module provides functions for loading, configuring, and using
metrics from the HuggingFace evaluate library, as well as creating
custom metrics and metric combinations.

Examples:
    >>> from hf_gtc.evaluation.evaluate import create_metric_config
    >>> config = create_metric_config("accuracy")
    >>> config.metric_name
    'accuracy'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


class MetricType(Enum):
    """Types of evaluation metrics.

    Attributes:
        CLASSIFICATION: Classification metrics (accuracy, f1, etc.).
        REGRESSION: Regression metrics (mse, mae, r2, etc.).
        GENERATION: Text generation metrics (bleu, rouge, etc.).
        SIMILARITY: Similarity metrics (bertscore, etc.).
        CUSTOM: User-defined custom metrics.

    Examples:
        >>> MetricType.CLASSIFICATION.value
        'classification'
        >>> MetricType.GENERATION.value
        'generation'
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    SIMILARITY = "similarity"
    CUSTOM = "custom"


VALID_METRIC_TYPES = frozenset(m.value for m in MetricType)

# Common metric names by type
CLASSIFICATION_METRICS = frozenset(
    {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "matthews_correlation",
        "roc_auc",
    }
)

REGRESSION_METRICS = frozenset(
    {
        "mse",
        "mae",
        "rmse",
        "r_squared",
        "mape",
        "smape",
    }
)

GENERATION_METRICS = frozenset(
    {
        "bleu",
        "rouge",
        "meteor",
        "bertscore",
        "sacrebleu",
        "chrf",
    }
)

# Aggregation methods for combining metrics
AggregationMethod = Literal["mean", "weighted", "min", "max"]
VALID_AGGREGATION_METHODS = frozenset({"mean", "weighted", "min", "max"})


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Configuration for a single metric.

    Attributes:
        metric_name: Name of the metric.
        metric_type: Type of the metric.
        average: Averaging method for multi-class (micro, macro, weighted).
        num_labels: Number of labels for classification.
        kwargs: Additional keyword arguments for the metric.

    Examples:
        >>> config = MetricConfig(
        ...     metric_name="accuracy",
        ...     metric_type=MetricType.CLASSIFICATION,
        ...     average=None,
        ...     num_labels=None,
        ...     kwargs=None,
        ... )
        >>> config.metric_name
        'accuracy'
    """

    metric_name: str
    metric_type: MetricType
    average: str | None
    num_labels: int | None
    kwargs: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Result from computing a metric.

    Attributes:
        metric_name: Name of the computed metric.
        value: Primary metric value.
        details: Additional metric details (e.g., per-class scores).

    Examples:
        >>> result = MetricResult(
        ...     metric_name="accuracy",
        ...     value=0.95,
        ...     details=None,
        ... )
        >>> result.value
        0.95
    """

    metric_name: str
    value: float
    details: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class CombinedMetricConfig:
    """Configuration for combining multiple metrics.

    Attributes:
        metrics: Tuple of metric configurations.
        aggregation: Method for combining metric values.
        weights: Optional weights for weighted aggregation.

    Examples:
        >>> m = MetricConfig("accuracy", MetricType.CLASSIFICATION, None, None, None)
        >>> config = CombinedMetricConfig(
        ...     metrics=(m,),
        ...     aggregation="mean",
        ...     weights=None,
        ... )
        >>> config.aggregation
        'mean'
    """

    metrics: tuple[MetricConfig, ...]
    aggregation: AggregationMethod
    weights: tuple[float, ...] | None


def validate_metric_config(config: MetricConfig) -> None:
    """Validate metric configuration parameters.

    Args:
        config: Metric configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = MetricConfig(
        ...     metric_name="accuracy",
        ...     metric_type=MetricType.CLASSIFICATION,
        ...     average=None,
        ...     num_labels=None,
        ...     kwargs=None,
        ... )
        >>> validate_metric_config(config)  # No error

        >>> bad_config = MetricConfig(
        ...     metric_name="",
        ...     metric_type=MetricType.CLASSIFICATION,
        ...     average=None,
        ...     num_labels=None,
        ...     kwargs=None,
        ... )
        >>> validate_metric_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric_name cannot be empty
    """
    if not config.metric_name:
        msg = "metric_name cannot be empty"
        raise ValueError(msg)

    if config.num_labels is not None and config.num_labels <= 0:
        msg = f"num_labels must be positive, got {config.num_labels}"
        raise ValueError(msg)

    valid_averages = {"micro", "macro", "weighted", "samples", "binary", None}
    if config.average is not None and config.average not in valid_averages:
        msg = f"average must be one of {valid_averages}, got '{config.average}'"
        raise ValueError(msg)


def create_metric_config(
    metric_name: str,
    metric_type: str = "classification",
    average: str | None = None,
    num_labels: int | None = None,
    **kwargs: Any,
) -> MetricConfig:
    """Create a metric configuration.

    Args:
        metric_name: Name of the metric (e.g., "accuracy", "f1").
        metric_type: Type of metric. Defaults to "classification".
        average: Averaging method for multi-class. Defaults to None.
        num_labels: Number of labels. Defaults to None.
        **kwargs: Additional metric arguments.

    Returns:
        MetricConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_metric_config("accuracy")
        >>> config.metric_name
        'accuracy'

        >>> config = create_metric_config("f1", average="macro")
        >>> config.average
        'macro'

        >>> create_metric_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric_name cannot be empty
    """
    if metric_type not in VALID_METRIC_TYPES:
        msg = f"metric_type must be one of {VALID_METRIC_TYPES}, got '{metric_type}'"
        raise ValueError(msg)

    config = MetricConfig(
        metric_name=metric_name,
        metric_type=MetricType(metric_type),
        average=average,
        num_labels=num_labels,
        kwargs=kwargs if kwargs else None,
    )
    validate_metric_config(config)
    return config


def create_combined_config(
    metrics: tuple[MetricConfig, ...],
    aggregation: AggregationMethod = "mean",
    weights: tuple[float, ...] | None = None,
) -> CombinedMetricConfig:
    """Create a combined metric configuration.

    Args:
        metrics: Tuple of metric configurations.
        aggregation: Aggregation method. Defaults to "mean".
        weights: Optional weights for weighted aggregation.

    Returns:
        CombinedMetricConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> m1 = create_metric_config("accuracy")
        >>> m2 = create_metric_config("f1")
        >>> config = create_combined_config((m1, m2))
        >>> config.aggregation
        'mean'

        >>> create_combined_config(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metrics cannot be empty
    """
    if not metrics:
        msg = "metrics cannot be empty"
        raise ValueError(msg)

    if aggregation not in VALID_AGGREGATION_METHODS:
        msg = (
            f"aggregation must be one of {VALID_AGGREGATION_METHODS}, "
            f"got '{aggregation}'"
        )
        raise ValueError(msg)

    if weights is not None:
        if len(weights) != len(metrics):
            msg = (
                f"weights length ({len(weights)}) must match "
                f"metrics length ({len(metrics)})"
            )
            raise ValueError(msg)

        if any(w < 0 for w in weights):
            msg = "weights cannot be negative"
            raise ValueError(msg)

    return CombinedMetricConfig(
        metrics=metrics,
        aggregation=aggregation,
        weights=weights,
    )


def list_classification_metrics() -> list[str]:
    """List available classification metrics.

    Returns:
        Sorted list of classification metric names.

    Examples:
        >>> metrics = list_classification_metrics()
        >>> "accuracy" in metrics
        True
        >>> "f1" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(CLASSIFICATION_METRICS)


def list_regression_metrics() -> list[str]:
    """List available regression metrics.

    Returns:
        Sorted list of regression metric names.

    Examples:
        >>> metrics = list_regression_metrics()
        >>> "mse" in metrics
        True
        >>> "mae" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(REGRESSION_METRICS)


def list_generation_metrics() -> list[str]:
    """List available text generation metrics.

    Returns:
        Sorted list of generation metric names.

    Examples:
        >>> metrics = list_generation_metrics()
        >>> "bleu" in metrics
        True
        >>> "rouge" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(GENERATION_METRICS)


def get_metric_type(metric_name: str) -> MetricType:
    """Infer metric type from metric name.

    Args:
        metric_name: Name of the metric.

    Returns:
        Inferred MetricType.

    Examples:
        >>> get_metric_type("accuracy")
        <MetricType.CLASSIFICATION: 'classification'>
        >>> get_metric_type("mse")
        <MetricType.REGRESSION: 'regression'>
        >>> get_metric_type("bleu")
        <MetricType.GENERATION: 'generation'>
        >>> get_metric_type("unknown_metric")
        <MetricType.CUSTOM: 'custom'>
    """
    if metric_name in CLASSIFICATION_METRICS:
        return MetricType.CLASSIFICATION
    if metric_name in REGRESSION_METRICS:
        return MetricType.REGRESSION
    if metric_name in GENERATION_METRICS:
        return MetricType.GENERATION
    return MetricType.CUSTOM


def aggregate_results(
    results: tuple[MetricResult, ...],
    method: AggregationMethod = "mean",
    weights: tuple[float, ...] | None = None,
) -> float:
    """Aggregate multiple metric results into a single value.

    Args:
        results: Tuple of metric results.
        method: Aggregation method. Defaults to "mean".
        weights: Optional weights for weighted aggregation.

    Returns:
        Aggregated metric value.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> r1 = MetricResult("accuracy", 0.9, None)
        >>> r2 = MetricResult("f1", 0.8, None)
        >>> round(aggregate_results((r1, r2), "mean"), 2)
        0.85

        >>> round(aggregate_results((r1, r2), "min"), 2)
        0.8

        >>> round(aggregate_results((r1, r2), "max"), 2)
        0.9

        >>> aggregate_results(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be empty
    """
    if not results:
        msg = "results cannot be empty"
        raise ValueError(msg)

    if method not in VALID_AGGREGATION_METHODS:
        msg = f"method must be one of {VALID_AGGREGATION_METHODS}, got '{method}'"
        raise ValueError(msg)

    values = tuple(r.value for r in results)

    if method == "mean":
        return sum(values) / len(values)

    if method == "min":
        return min(values)

    if method == "max":
        return max(values)

    # Weighted aggregation
    if weights is None:
        msg = "weights required for weighted aggregation"
        raise ValueError(msg)

    if len(weights) != len(values):
        msg = (
            f"weights length ({len(weights)}) must match results length ({len(values)})"
        )
        raise ValueError(msg)

    total_weight = sum(weights)
    if total_weight == 0:
        msg = "total weight cannot be zero"
        raise ValueError(msg)

    return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight


def compute_accuracy(
    predictions: tuple[int, ...],
    references: tuple[int, ...],
) -> float:
    """Compute accuracy score.

    Args:
        predictions: Predicted labels.
        references: True labels.

    Returns:
        Accuracy score in range [0, 1].

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> compute_accuracy((1, 0, 1, 1), (1, 0, 0, 1))
        0.75

        >>> compute_accuracy((1, 1, 1), (1, 1, 1))
        1.0

        >>> compute_accuracy((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if not predictions:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(references):
        msg = (
            f"predictions length ({len(predictions)}) must match "
            f"references length ({len(references)})"
        )
        raise ValueError(msg)

    correct = sum(1 for p, r in zip(predictions, references, strict=True) if p == r)
    return correct / len(predictions)


def compute_precision(
    predictions: tuple[int, ...],
    references: tuple[int, ...],
    positive_label: int = 1,
) -> float:
    """Compute precision score.

    Args:
        predictions: Predicted labels.
        references: True labels.
        positive_label: Label considered positive. Defaults to 1.

    Returns:
        Precision score in range [0, 1].

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> compute_precision((1, 1, 0, 1), (1, 0, 0, 1))
        0.6666666666666666

        >>> compute_precision((0, 0, 0), (1, 1, 1))
        0.0

        >>> compute_precision((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if not predictions:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(references):
        msg = (
            f"predictions length ({len(predictions)}) must match "
            f"references length ({len(references)})"
        )
        raise ValueError(msg)

    true_positives = sum(
        1
        for p, r in zip(predictions, references, strict=True)
        if p == positive_label and r == positive_label
    )
    predicted_positives = sum(1 for p in predictions if p == positive_label)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def compute_recall(
    predictions: tuple[int, ...],
    references: tuple[int, ...],
    positive_label: int = 1,
) -> float:
    """Compute recall score.

    Args:
        predictions: Predicted labels.
        references: True labels.
        positive_label: Label considered positive. Defaults to 1.

    Returns:
        Recall score in range [0, 1].

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> compute_recall((1, 1, 0, 1), (1, 0, 0, 1))
        1.0

        >>> compute_recall((0, 0, 0), (1, 1, 1))
        0.0

        >>> compute_recall((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if not predictions:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(references):
        msg = (
            f"predictions length ({len(predictions)}) must match "
            f"references length ({len(references)})"
        )
        raise ValueError(msg)

    true_positives = sum(
        1
        for p, r in zip(predictions, references, strict=True)
        if p == positive_label and r == positive_label
    )
    actual_positives = sum(1 for r in references if r == positive_label)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def compute_f1(
    predictions: tuple[int, ...],
    references: tuple[int, ...],
    positive_label: int = 1,
) -> float:
    """Compute F1 score.

    Args:
        predictions: Predicted labels.
        references: True labels.
        positive_label: Label considered positive. Defaults to 1.

    Returns:
        F1 score in range [0, 1].

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> round(compute_f1((1, 1, 0, 1), (1, 0, 0, 1)), 4)
        0.8

        >>> compute_f1((0, 0, 0), (1, 1, 1))
        0.0

        >>> compute_f1((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    precision = compute_precision(predictions, references, positive_label)
    recall = compute_recall(predictions, references, positive_label)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_mse(
    predictions: tuple[float, ...],
    references: tuple[float, ...],
) -> float:
    """Compute Mean Squared Error.

    Args:
        predictions: Predicted values.
        references: True values.

    Returns:
        MSE value (always non-negative).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> compute_mse((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        0.0

        >>> compute_mse((1.0, 2.0, 3.0), (2.0, 3.0, 4.0))
        1.0

        >>> compute_mse((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if not predictions:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(references):
        msg = (
            f"predictions length ({len(predictions)}) must match "
            f"references length ({len(references)})"
        )
        raise ValueError(msg)

    squared_errors = tuple(
        (p - r) ** 2 for p, r in zip(predictions, references, strict=True)
    )
    return sum(squared_errors) / len(squared_errors)


def compute_mae(
    predictions: tuple[float, ...],
    references: tuple[float, ...],
) -> float:
    """Compute Mean Absolute Error.

    Args:
        predictions: Predicted values.
        references: True values.

    Returns:
        MAE value (always non-negative).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> compute_mae((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        0.0

        >>> compute_mae((1.0, 2.0, 3.0), (2.0, 3.0, 4.0))
        1.0

        >>> compute_mae((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if not predictions:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(references):
        msg = (
            f"predictions length ({len(predictions)}) must match "
            f"references length ({len(references)})"
        )
        raise ValueError(msg)

    absolute_errors = tuple(
        abs(p - r) for p, r in zip(predictions, references, strict=True)
    )
    return sum(absolute_errors) / len(absolute_errors)


def get_recommended_metrics(task: str) -> tuple[str, ...]:
    """Get recommended metrics for a task type.

    Args:
        task: Task type (classification, regression, generation, qa).

    Returns:
        Tuple of recommended metric names.

    Raises:
        ValueError: If task is not recognized.

    Examples:
        >>> metrics = get_recommended_metrics("classification")
        >>> "accuracy" in metrics
        True
        >>> "f1" in metrics
        True

        >>> metrics = get_recommended_metrics("regression")
        >>> "mse" in metrics
        True
    """
    valid_tasks = {"classification", "regression", "generation", "qa"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    recommendations = {
        "classification": ("accuracy", "f1", "precision", "recall"),
        "regression": ("mse", "mae", "r_squared"),
        "generation": ("bleu", "rouge", "bertscore"),
        "qa": ("f1", "exact_match"),
    }
    return recommendations[task]
