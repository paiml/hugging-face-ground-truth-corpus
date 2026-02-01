"""Evaluation harness and task runners for model evaluation.

This module provides utilities for configuring and running evaluation harnesses,
including task configuration, result aggregation, and output formatting.

Examples:
    >>> from hf_gtc.evaluation.harness import TaskType, OutputFormat
    >>> TaskType.MULTIPLE_CHOICE.value
    'multiple_choice'
    >>> OutputFormat.JSON.value
    'json'
    >>> from hf_gtc.evaluation.harness import AggregationMethod
    >>> AggregationMethod.MEAN.value
    'mean'
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from hf_gtc._validation import validate_not_none


class TaskType(Enum):
    """Types of evaluation tasks.

    Attributes:
        MULTIPLE_CHOICE: Multiple choice question answering tasks.
        GENERATION: Free-form text generation tasks.
        CLASSIFICATION: Text classification tasks.
        EXTRACTION: Information extraction tasks.
        RANKING: Ranking and preference tasks.

    Examples:
        >>> TaskType.MULTIPLE_CHOICE.value
        'multiple_choice'
        >>> TaskType.GENERATION.value
        'generation'
        >>> TaskType.CLASSIFICATION.value
        'classification'
        >>> TaskType.EXTRACTION.value
        'extraction'
        >>> TaskType.RANKING.value
        'ranking'
    """

    MULTIPLE_CHOICE = "multiple_choice"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    RANKING = "ranking"


VALID_TASK_TYPES = frozenset(t.value for t in TaskType)


class OutputFormat(Enum):
    """Output formats for evaluation results.

    Attributes:
        JSON: JSON format output.
        CSV: CSV format output.
        MARKDOWN: Markdown table format.
        LATEX: LaTeX table format.

    Examples:
        >>> OutputFormat.JSON.value
        'json'
        >>> OutputFormat.CSV.value
        'csv'
        >>> OutputFormat.MARKDOWN.value
        'markdown'
        >>> OutputFormat.LATEX.value
        'latex'
    """

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    LATEX = "latex"


VALID_OUTPUT_FORMATS = frozenset(f.value for f in OutputFormat)


class AggregationMethod(Enum):
    """Methods for aggregating evaluation results.

    Attributes:
        MEAN: Simple arithmetic mean of scores.
        WEIGHTED_MEAN: Weighted mean based on sample counts.
        MEDIAN: Median of scores.
        MAX: Maximum score.

    Examples:
        >>> AggregationMethod.MEAN.value
        'mean'
        >>> AggregationMethod.WEIGHTED_MEAN.value
        'weighted_mean'
        >>> AggregationMethod.MEDIAN.value
        'median'
        >>> AggregationMethod.MAX.value
        'max'
    """

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"
    MAX = "max"


VALID_AGGREGATION_METHODS = frozenset(m.value for m in AggregationMethod)


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Configuration for an evaluation task.

    Attributes:
        task_type: Type of evaluation task.
        num_fewshot: Number of few-shot examples. Defaults to 0.
        batch_size: Batch size for evaluation. Defaults to 32.
        limit: Maximum number of samples to evaluate. Defaults to None (all).

    Examples:
        >>> config = TaskConfig(task_type=TaskType.MULTIPLE_CHOICE)
        >>> config.task_type
        <TaskType.MULTIPLE_CHOICE: 'multiple_choice'>
        >>> config.num_fewshot
        0
        >>> config.batch_size
        32

        >>> config = TaskConfig(
        ...     task_type=TaskType.GENERATION,
        ...     num_fewshot=5,
        ...     batch_size=16,
        ...     limit=100,
        ... )
        >>> config.num_fewshot
        5
        >>> config.limit
        100
    """

    task_type: TaskType
    num_fewshot: int = 0
    batch_size: int = 32
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Configuration for an evaluation harness.

    Attributes:
        tasks: Tuple of task names to evaluate.
        output_format: Output format for results. Defaults to JSON.
        aggregation: Result aggregation method. Defaults to MEAN.
        log_samples: Whether to log individual samples. Defaults to False.
        cache_requests: Whether to cache model requests. Defaults to True.

    Examples:
        >>> config = HarnessConfig(tasks=("mmlu", "hellaswag"))
        >>> config.tasks
        ('mmlu', 'hellaswag')
        >>> config.output_format
        <OutputFormat.JSON: 'json'>
        >>> config.aggregation
        <AggregationMethod.MEAN: 'mean'>

        >>> config = HarnessConfig(
        ...     tasks=("arc", "winogrande"),
        ...     output_format=OutputFormat.MARKDOWN,
        ...     aggregation=AggregationMethod.WEIGHTED_MEAN,
        ...     log_samples=True,
        ...     cache_requests=False,
        ... )
        >>> config.log_samples
        True
        >>> config.cache_requests
        False
    """

    tasks: tuple[str, ...]
    output_format: OutputFormat = OutputFormat.JSON
    aggregation: AggregationMethod = AggregationMethod.MEAN
    log_samples: bool = False
    cache_requests: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Result from evaluating a single task.

    Attributes:
        task_name: Name of the evaluated task.
        metrics: Dictionary of metric names to values.
        num_samples: Number of samples evaluated.
        duration_seconds: Time taken for evaluation in seconds.

    Examples:
        >>> result = EvaluationResult(
        ...     task_name="mmlu",
        ...     metrics={"accuracy": 0.65, "f1": 0.63},
        ...     num_samples=1000,
        ...     duration_seconds=120.5,
        ... )
        >>> result.task_name
        'mmlu'
        >>> result.metrics["accuracy"]
        0.65
        >>> result.num_samples
        1000

        >>> result = EvaluationResult(
        ...     task_name="hellaswag",
        ...     metrics={"accuracy": 0.79},
        ...     num_samples=500,
        ...     duration_seconds=60.0,
        ... )
        >>> result.duration_seconds
        60.0
    """

    task_name: str
    metrics: dict[str, float]
    num_samples: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class HarnessStats:
    """Statistics from a harness evaluation run.

    Attributes:
        total_tasks: Total number of tasks configured.
        completed_tasks: Number of tasks successfully completed.
        failed_tasks: Number of tasks that failed.
        total_samples: Total number of samples evaluated.
        avg_score: Average score across all tasks.

    Examples:
        >>> stats = HarnessStats(
        ...     total_tasks=5,
        ...     completed_tasks=4,
        ...     failed_tasks=1,
        ...     total_samples=5000,
        ...     avg_score=0.72,
        ... )
        >>> stats.total_tasks
        5
        >>> stats.completed_tasks
        4
        >>> stats.avg_score
        0.72

        >>> stats = HarnessStats(
        ...     total_tasks=3,
        ...     completed_tasks=3,
        ...     failed_tasks=0,
        ...     total_samples=3000,
        ...     avg_score=0.85,
        ... )
        >>> stats.failed_tasks
        0
    """

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_samples: int
    avg_score: float


# Validation functions


def validate_task_config(config: TaskConfig) -> None:
    """Validate task configuration.

    Args:
        config: TaskConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_fewshot is negative.
        ValueError: If batch_size is not positive.
        ValueError: If limit is specified and not positive.

    Examples:
        >>> config = TaskConfig(task_type=TaskType.MULTIPLE_CHOICE)
        >>> validate_task_config(config)  # No error

        >>> validate_task_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = TaskConfig(
        ...     task_type=TaskType.GENERATION,
        ...     num_fewshot=-1,
        ... )
        >>> validate_task_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_fewshot cannot be negative

        >>> bad = TaskConfig(
        ...     task_type=TaskType.CLASSIFICATION,
        ...     batch_size=0,
        ... )
        >>> validate_task_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    validate_not_none(config, "config")

    if config.num_fewshot < 0:
        msg = f"num_fewshot cannot be negative, got {config.num_fewshot}"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.limit is not None and config.limit <= 0:
        msg = f"limit must be positive when specified, got {config.limit}"
        raise ValueError(msg)


def validate_harness_config(config: HarnessConfig) -> None:
    """Validate harness configuration.

    Args:
        config: HarnessConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If tasks is empty.
        ValueError: If tasks contains empty strings.

    Examples:
        >>> config = HarnessConfig(tasks=("mmlu",))
        >>> validate_harness_config(config)  # No error

        >>> validate_harness_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = HarnessConfig(tasks=())
        >>> validate_harness_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tasks cannot be empty

        >>> bad = HarnessConfig(tasks=("mmlu", ""))
        >>> validate_harness_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tasks cannot contain empty strings
    """
    validate_not_none(config, "config")

    if len(config.tasks) == 0:
        msg = "tasks cannot be empty"
        raise ValueError(msg)

    for task in config.tasks:
        if not task:
            msg = "tasks cannot contain empty strings"
            raise ValueError(msg)


def validate_evaluation_result(result: EvaluationResult) -> None:
    """Validate evaluation result.

    Args:
        result: EvaluationResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If task_name is empty.
        ValueError: If num_samples is not positive.
        ValueError: If duration_seconds is negative.

    Examples:
        >>> result = EvaluationResult(
        ...     task_name="mmlu",
        ...     metrics={"accuracy": 0.65},
        ...     num_samples=100,
        ...     duration_seconds=10.0,
        ... )
        >>> validate_evaluation_result(result)  # No error

        >>> validate_evaluation_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = EvaluationResult(
        ...     task_name="",
        ...     metrics={},
        ...     num_samples=100,
        ...     duration_seconds=10.0,
        ... )
        >>> validate_evaluation_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_name cannot be empty
    """
    validate_not_none(result, "result")

    if not result.task_name:
        msg = "task_name cannot be empty"
        raise ValueError(msg)

    if result.num_samples <= 0:
        msg = f"num_samples must be positive, got {result.num_samples}"
        raise ValueError(msg)

    if result.duration_seconds < 0:
        msg = f"duration_seconds cannot be negative, got {result.duration_seconds}"
        raise ValueError(msg)


def validate_harness_stats(stats: HarnessStats) -> None:
    """Validate harness statistics.

    Args:
        stats: HarnessStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If total_tasks is not positive.
        ValueError: If completed_tasks + failed_tasks > total_tasks.
        ValueError: If any count is negative.

    Examples:
        >>> stats = HarnessStats(
        ...     total_tasks=5,
        ...     completed_tasks=4,
        ...     failed_tasks=1,
        ...     total_samples=5000,
        ...     avg_score=0.72,
        ... )
        >>> validate_harness_stats(stats)  # No error

        >>> validate_harness_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = HarnessStats(
        ...     total_tasks=0,
        ...     completed_tasks=0,
        ...     failed_tasks=0,
        ...     total_samples=0,
        ...     avg_score=0.0,
        ... )
        >>> validate_harness_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_tasks must be positive
    """
    validate_not_none(stats, "stats")

    if stats.total_tasks <= 0:
        msg = f"total_tasks must be positive, got {stats.total_tasks}"
        raise ValueError(msg)

    if stats.completed_tasks < 0:
        msg = f"completed_tasks cannot be negative, got {stats.completed_tasks}"
        raise ValueError(msg)

    if stats.failed_tasks < 0:
        msg = f"failed_tasks cannot be negative, got {stats.failed_tasks}"
        raise ValueError(msg)

    if stats.completed_tasks + stats.failed_tasks > stats.total_tasks:
        msg = (
            f"completed_tasks ({stats.completed_tasks}) + failed_tasks "
            f"({stats.failed_tasks}) cannot exceed total_tasks ({stats.total_tasks})"
        )
        raise ValueError(msg)

    if stats.total_samples < 0:
        msg = f"total_samples cannot be negative, got {stats.total_samples}"
        raise ValueError(msg)


# Factory functions


def create_task_config(
    task_type: TaskType,
    *,
    num_fewshot: int = 0,
    batch_size: int = 32,
    limit: int | None = None,
) -> TaskConfig:
    """Create and validate a task configuration.

    Args:
        task_type: Type of evaluation task.
        num_fewshot: Number of few-shot examples. Defaults to 0.
        batch_size: Batch size for evaluation. Defaults to 32.
        limit: Maximum number of samples. Defaults to None.

    Returns:
        Validated TaskConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_task_config(TaskType.MULTIPLE_CHOICE)
        >>> config.task_type
        <TaskType.MULTIPLE_CHOICE: 'multiple_choice'>

        >>> config = create_task_config(
        ...     TaskType.GENERATION,
        ...     num_fewshot=5,
        ...     batch_size=16,
        ... )
        >>> config.num_fewshot
        5
        >>> config.batch_size
        16
    """
    config = TaskConfig(
        task_type=task_type,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )
    validate_task_config(config)
    return config


def create_harness_config(
    tasks: tuple[str, ...],
    *,
    output_format: OutputFormat = OutputFormat.JSON,
    aggregation: AggregationMethod = AggregationMethod.MEAN,
    log_samples: bool = False,
    cache_requests: bool = True,
) -> HarnessConfig:
    """Create and validate a harness configuration.

    Args:
        tasks: Tuple of task names to evaluate.
        output_format: Output format for results. Defaults to JSON.
        aggregation: Result aggregation method. Defaults to MEAN.
        log_samples: Whether to log individual samples. Defaults to False.
        cache_requests: Whether to cache model requests. Defaults to True.

    Returns:
        Validated HarnessConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_harness_config(("mmlu", "hellaswag"))
        >>> config.tasks
        ('mmlu', 'hellaswag')
        >>> config.output_format
        <OutputFormat.JSON: 'json'>

        >>> config = create_harness_config(
        ...     ("arc",),
        ...     output_format=OutputFormat.MARKDOWN,
        ...     aggregation=AggregationMethod.WEIGHTED_MEAN,
        ... )
        >>> config.aggregation
        <AggregationMethod.WEIGHTED_MEAN: 'weighted_mean'>
    """
    config = HarnessConfig(
        tasks=tasks,
        output_format=output_format,
        aggregation=aggregation,
        log_samples=log_samples,
        cache_requests=cache_requests,
    )
    validate_harness_config(config)
    return config


def create_evaluation_result(
    task_name: str,
    metrics: dict[str, float],
    num_samples: int,
    duration_seconds: float,
) -> EvaluationResult:
    """Create and validate an evaluation result.

    Args:
        task_name: Name of the evaluated task.
        metrics: Dictionary of metric names to values.
        num_samples: Number of samples evaluated.
        duration_seconds: Time taken for evaluation in seconds.

    Returns:
        Validated EvaluationResult instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_evaluation_result(
        ...     "mmlu",
        ...     {"accuracy": 0.65},
        ...     1000,
        ...     120.5,
        ... )
        >>> result.task_name
        'mmlu'
        >>> result.metrics["accuracy"]
        0.65

        >>> result = create_evaluation_result(
        ...     "hellaswag",
        ...     {"accuracy": 0.79, "f1": 0.78},
        ...     500,
        ...     60.0,
        ... )
        >>> result.num_samples
        500
    """
    result = EvaluationResult(
        task_name=task_name,
        metrics=dict(metrics),  # Make a copy
        num_samples=num_samples,
        duration_seconds=duration_seconds,
    )
    validate_evaluation_result(result)
    return result


# List/get functions for enums


def list_task_types() -> list[str]:
    """List all available task types.

    Returns:
        Sorted list of task type names.

    Examples:
        >>> types = list_task_types()
        >>> "multiple_choice" in types
        True
        >>> "generation" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_TASK_TYPES)


def get_task_type(name: str) -> TaskType:
    """Get TaskType enum from string name.

    Args:
        name: Name of the task type.

    Returns:
        Corresponding TaskType enum value.

    Raises:
        ValueError: If name is not a valid task type.

    Examples:
        >>> get_task_type("multiple_choice")
        <TaskType.MULTIPLE_CHOICE: 'multiple_choice'>

        >>> get_task_type("generation")
        <TaskType.GENERATION: 'generation'>

        >>> get_task_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid task type: invalid
    """
    if name not in VALID_TASK_TYPES:
        msg = f"invalid task type: {name}"
        raise ValueError(msg)

    return TaskType(name)


def validate_task_type(task_type: str) -> bool:
    """Validate if a string is a valid task type.

    Args:
        task_type: The task type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_task_type("multiple_choice")
        True
        >>> validate_task_type("generation")
        True
        >>> validate_task_type("invalid")
        False
        >>> validate_task_type("")
        False
    """
    return task_type in VALID_TASK_TYPES


def list_output_formats() -> list[str]:
    """List all available output formats.

    Returns:
        Sorted list of output format names.

    Examples:
        >>> formats = list_output_formats()
        >>> "json" in formats
        True
        >>> "markdown" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_OUTPUT_FORMATS)


def get_output_format(name: str) -> OutputFormat:
    """Get OutputFormat enum from string name.

    Args:
        name: Name of the output format.

    Returns:
        Corresponding OutputFormat enum value.

    Raises:
        ValueError: If name is not a valid output format.

    Examples:
        >>> get_output_format("json")
        <OutputFormat.JSON: 'json'>

        >>> get_output_format("markdown")
        <OutputFormat.MARKDOWN: 'markdown'>

        >>> get_output_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid output format: invalid
    """
    if name not in VALID_OUTPUT_FORMATS:
        msg = f"invalid output format: {name}"
        raise ValueError(msg)

    return OutputFormat(name)


def validate_output_format(output_format: str) -> bool:
    """Validate if a string is a valid output format.

    Args:
        output_format: The output format string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_output_format("json")
        True
        >>> validate_output_format("csv")
        True
        >>> validate_output_format("invalid")
        False
        >>> validate_output_format("")
        False
    """
    return output_format in VALID_OUTPUT_FORMATS


def list_aggregation_methods() -> list[str]:
    """List all available aggregation methods.

    Returns:
        Sorted list of aggregation method names.

    Examples:
        >>> methods = list_aggregation_methods()
        >>> "mean" in methods
        True
        >>> "weighted_mean" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_AGGREGATION_METHODS)


def get_aggregation_method(name: str) -> AggregationMethod:
    """Get AggregationMethod enum from string name.

    Args:
        name: Name of the aggregation method.

    Returns:
        Corresponding AggregationMethod enum value.

    Raises:
        ValueError: If name is not a valid aggregation method.

    Examples:
        >>> get_aggregation_method("mean")
        <AggregationMethod.MEAN: 'mean'>

        >>> get_aggregation_method("weighted_mean")
        <AggregationMethod.WEIGHTED_MEAN: 'weighted_mean'>

        >>> get_aggregation_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid aggregation method: invalid
    """
    if name not in VALID_AGGREGATION_METHODS:
        msg = f"invalid aggregation method: {name}"
        raise ValueError(msg)

    return AggregationMethod(name)


def validate_aggregation_method(method: str) -> bool:
    """Validate if a string is a valid aggregation method.

    Args:
        method: The aggregation method string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_aggregation_method("mean")
        True
        >>> validate_aggregation_method("median")
        True
        >>> validate_aggregation_method("invalid")
        False
        >>> validate_aggregation_method("")
        False
    """
    return method in VALID_AGGREGATION_METHODS


# Core functions


def run_evaluation_task(
    task_name: str,
    config: TaskConfig,
    inference_fn: Callable[[Sequence[Any]], Sequence[Any]],
    data: Sequence[Any],
    metrics_fn: (
        Callable[[Sequence[Any], Sequence[Any]], dict[str, float]] | None
    ) = None,
) -> EvaluationResult:
    """Run evaluation on a single task.

    Args:
        task_name: Name of the task being evaluated.
        config: Task configuration.
        inference_fn: Function that performs inference on inputs.
        data: Data samples to evaluate.
        metrics_fn: Optional function to compute metrics from predictions and labels.

    Returns:
        EvaluationResult with metrics and statistics.

    Raises:
        ValueError: If task_name is empty.
        ValueError: If config is None.
        ValueError: If inference_fn is None.
        ValueError: If data is None.

    Examples:
        >>> config = create_task_config(TaskType.CLASSIFICATION)
        >>> data = [(1, 1), (2, 0), (3, 1)]  # (input, label) pairs
        >>> def inference(inputs):
        ...     return [1 for _ in inputs]  # Always predict 1
        >>> def metrics(preds, labels):
        ...     correct = sum(p == l for p, l in zip(preds, labels))
        ...     return {"accuracy": correct / len(preds)}
        >>> result = run_evaluation_task(
        ...     "test_task",
        ...     config,
        ...     inference,
        ...     data,
        ...     metrics,
        ... )
        >>> result.task_name
        'test_task'
        >>> 0.66 <= result.metrics["accuracy"] <= 0.67
        True
    """
    if not task_name:
        msg = "task_name cannot be empty"
        raise ValueError(msg)

    validate_not_none(config, "config")

    if inference_fn is None:
        msg = "inference_fn cannot be None"
        raise ValueError(msg)

    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    validate_task_config(config)

    # Apply limit if specified
    samples = data[: config.limit] if config.limit is not None else data
    num_samples = len(samples)

    start_time = time.perf_counter()

    try:
        # Extract inputs and labels
        inputs = [
            sample[0] if isinstance(sample, tuple) else sample for sample in samples
        ]
        labels = [
            sample[1] if isinstance(sample, tuple) else None for sample in samples
        ]

        # Run inference in batches
        predictions: list[Any] = []
        for i in range(0, len(inputs), config.batch_size):
            batch = inputs[i : i + config.batch_size]
            batch_preds = inference_fn(batch)
            predictions.extend(batch_preds)

        # Compute metrics
        metrics: dict[str, float] = {}
        if metrics_fn is not None and labels[0] is not None:
            metrics = metrics_fn(predictions, labels)
        else:
            # Default: compute accuracy if we have labels
            if labels[0] is not None:
                correct = sum(
                    pred == lbl for pred, lbl in zip(predictions, labels, strict=True)
                )
                metrics = {"accuracy": correct / len(predictions)}

        duration = time.perf_counter() - start_time

        return create_evaluation_result(
            task_name=task_name,
            metrics=metrics,
            num_samples=num_samples,
            duration_seconds=duration,
        )

    except Exception:
        duration = time.perf_counter() - start_time
        return EvaluationResult(
            task_name=task_name,
            metrics={"error": 0.0},
            num_samples=num_samples,
            duration_seconds=duration,
        )


def aggregate_results(
    results: Sequence[EvaluationResult],
    method: AggregationMethod = AggregationMethod.MEAN,
) -> dict[str, float]:
    """Aggregate results from multiple evaluation tasks.

    Args:
        results: Sequence of EvaluationResult objects.
        method: Aggregation method to use. Defaults to MEAN.

    Returns:
        Dictionary with aggregated metrics.

    Raises:
        ValueError: If results is None or empty.

    Examples:
        >>> r1 = create_evaluation_result("task1", {"accuracy": 0.8}, 100, 10.0)
        >>> r2 = create_evaluation_result("task2", {"accuracy": 0.9}, 200, 20.0)
        >>> agg = aggregate_results([r1, r2])
        >>> 0.84 < agg["accuracy"] < 0.86
        True

        >>> agg = aggregate_results([r1, r2], AggregationMethod.WEIGHTED_MEAN)
        >>> 0.86 < agg["accuracy"] < 0.88
        True

        >>> agg = aggregate_results([r1, r2], AggregationMethod.MAX)
        >>> agg["accuracy"]
        0.9

        >>> aggregate_results([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be empty
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if len(results) == 0:
        msg = "results cannot be empty"
        raise ValueError(msg)

    # Collect all metrics
    all_metrics: dict[str, list[tuple[float, int]]] = {}
    for result in results:
        for metric_name, value in result.metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append((value, result.num_samples))

    # Aggregate based on method
    aggregated: dict[str, float] = {}

    for metric_name, values_with_samples in all_metrics.items():
        values = [v for v, _ in values_with_samples]
        samples = [s for _, s in values_with_samples]

        if method == AggregationMethod.MEAN:
            aggregated[metric_name] = sum(values) / len(values)
        elif method == AggregationMethod.WEIGHTED_MEAN:
            total_samples = sum(samples)
            if total_samples > 0:
                weighted_sum = sum(v * s for v, s in values_with_samples)
                aggregated[metric_name] = weighted_sum / total_samples
            else:
                aggregated[metric_name] = sum(values) / len(values)
        elif method == AggregationMethod.MEDIAN:
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                aggregated[metric_name] = (
                    sorted_values[n // 2 - 1] + sorted_values[n // 2]
                ) / 2
            else:
                aggregated[metric_name] = sorted_values[n // 2]
        elif method == AggregationMethod.MAX:
            aggregated[metric_name] = max(values)

    return aggregated


def format_results_table(
    results: Sequence[EvaluationResult],
    output_format: OutputFormat = OutputFormat.MARKDOWN,
) -> str:
    """Format evaluation results as a table.

    Args:
        results: Sequence of EvaluationResult objects.
        output_format: Output format to use. Defaults to MARKDOWN.

    Returns:
        Formatted string representation of results.

    Raises:
        ValueError: If results is None or empty.

    Examples:
        >>> r1 = create_evaluation_result("mmlu", {"accuracy": 0.65}, 1000, 120.0)
        >>> r2 = create_evaluation_result("hellaswag", {"accuracy": 0.79}, 500, 60.0)
        >>> table = format_results_table([r1, r2])
        >>> "mmlu" in table
        True
        >>> "hellaswag" in table
        True

        >>> table = format_results_table([r1], OutputFormat.JSON)
        >>> "mmlu" in table
        True

        >>> table = format_results_table([r1], OutputFormat.CSV)
        >>> "mmlu" in table
        True

        >>> format_results_table([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be empty
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if len(results) == 0:
        msg = "results cannot be empty"
        raise ValueError(msg)

    if output_format == OutputFormat.JSON:
        data = []
        for result in results:
            entry = {
                "task_name": result.task_name,
                "num_samples": result.num_samples,
                "duration_seconds": result.duration_seconds,
                **result.metrics,
            }
            data.append(entry)
        return json.dumps(data, indent=2)

    elif output_format == OutputFormat.CSV:
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        metric_names = sorted(all_metrics)

        header_parts = ["task_name", "num_samples", "duration_seconds", *metric_names]
        lines = [",".join(header_parts)]
        for result in results:
            row = [
                result.task_name,
                str(result.num_samples),
                f"{result.duration_seconds:.2f}",
            ]
            for metric in metric_names:
                value = result.metrics.get(metric, 0.0)
                row.append(f"{value:.4f}")
            lines.append(",".join(row))
        return "\n".join(lines)

    elif output_format == OutputFormat.LATEX:
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        metric_names = sorted(all_metrics)

        header_cols = ["Task", "Samples", "Duration (s)"] + [
            m.replace("_", " ").title() for m in metric_names
        ]
        lines = [
            "\\begin{tabular}{l" + "r" * (len(header_cols) - 1) + "}",
            "\\toprule",
            " & ".join(header_cols) + " \\\\",
            "\\midrule",
        ]

        for result in results:
            row = [
                result.task_name,
                str(result.num_samples),
                f"{result.duration_seconds:.2f}",
            ]
            for metric in metric_names:
                value = result.metrics.get(metric, 0.0)
                row.append(f"{value:.4f}")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines)

    else:  # MARKDOWN
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        metric_names = sorted(all_metrics)

        header = "| Task | Samples | Duration (s) |"
        separator = "|------|---------|--------------|"
        for metric in metric_names:
            header += f" {metric.replace('_', ' ').title()} |"
            separator += "--------|"

        lines = [header, separator]

        for result in results:
            duration_str = f"{result.duration_seconds:.2f}"
            row = f"| {result.task_name} | {result.num_samples} | {duration_str} |"
            for metric in metric_names:
                value = result.metrics.get(metric, 0.0)
                row += f" {value:.4f} |"
            lines.append(row)

        return "\n".join(lines)


def estimate_evaluation_time(
    num_samples: int,
    batch_size: int,
    avg_inference_time: float,
) -> float:
    """Estimate total evaluation time.

    Args:
        num_samples: Total number of samples to evaluate.
        batch_size: Batch size for evaluation.
        avg_inference_time: Average inference time per batch in seconds.

    Returns:
        Estimated total time in seconds.

    Raises:
        ValueError: If num_samples is not positive.
        ValueError: If batch_size is not positive.
        ValueError: If avg_inference_time is not positive.

    Examples:
        >>> estimate_evaluation_time(1000, 32, 0.5)
        16.0

        >>> estimate_evaluation_time(100, 10, 1.0)
        10.0

        >>> estimate_evaluation_time(0, 32, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples must be positive
    """
    if num_samples <= 0:
        msg = f"num_samples must be positive, got {num_samples}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if avg_inference_time <= 0:
        msg = f"avg_inference_time must be positive, got {avg_inference_time}"
        raise ValueError(msg)

    num_batches = (num_samples + batch_size - 1) // batch_size
    return num_batches * avg_inference_time


def format_harness_stats(stats: HarnessStats) -> str:
    """Format harness statistics as a human-readable string.

    Args:
        stats: HarnessStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = HarnessStats(
        ...     total_tasks=5,
        ...     completed_tasks=4,
        ...     failed_tasks=1,
        ...     total_samples=5000,
        ...     avg_score=0.72,
        ... )
        >>> formatted = format_harness_stats(stats)
        >>> "5" in formatted
        True
        >>> "4" in formatted
        True
        >>> "0.72" in formatted
        True

        >>> format_harness_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        "Evaluation Harness Statistics",
        "=" * 40,
        f"Total Tasks: {stats.total_tasks}",
        f"Completed Tasks: {stats.completed_tasks}",
        f"Failed Tasks: {stats.failed_tasks}",
        f"Total Samples: {stats.total_samples}",
        f"Average Score: {stats.avg_score:.4f}",
    ]

    success_rate = (
        stats.completed_tasks / stats.total_tasks * 100 if stats.total_tasks > 0 else 0
    )
    lines.append(f"Success Rate: {success_rate:.1f}%")

    return "\n".join(lines)


def get_recommended_harness_config(
    model_type: str,
    *,
    num_tasks: int | None = None,
) -> dict[str, Any]:
    """Get recommended harness configuration for a model type.

    Args:
        model_type: Type of model (e.g., "base", "instruction", "code").
        num_tasks: Optional specific number of tasks.

    Returns:
        Dictionary with recommended configuration settings.

    Raises:
        ValueError: If model_type is empty.

    Examples:
        >>> config = get_recommended_harness_config("base")
        >>> "tasks" in config
        True
        >>> len(config["tasks"]) > 0
        True

        >>> config = get_recommended_harness_config("code")
        >>> "humaneval" in config["tasks"] or "mbpp" in config["tasks"]
        True

        >>> config = get_recommended_harness_config("instruction")
        >>> config["num_fewshot"]
        0

        >>> get_recommended_harness_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty
    """
    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    model_lower = model_type.lower()

    # Base configurations for different model types
    configs: dict[str, dict[str, Any]] = {
        "base": {
            "tasks": ("mmlu", "hellaswag", "arc_easy", "arc_challenge", "winogrande"),
            "num_fewshot": 5,
            "batch_size": 32,
            "output_format": OutputFormat.MARKDOWN,
            "aggregation": AggregationMethod.WEIGHTED_MEAN,
        },
        "instruction": {
            "tasks": ("mmlu", "truthfulqa", "alpaca_eval"),
            "num_fewshot": 0,
            "batch_size": 16,
            "output_format": OutputFormat.JSON,
            "aggregation": AggregationMethod.MEAN,
        },
        "code": {
            "tasks": ("humaneval", "mbpp"),
            "num_fewshot": 0,
            "batch_size": 8,
            "output_format": OutputFormat.JSON,
            "aggregation": AggregationMethod.MEAN,
        },
        "chat": {
            "tasks": ("mt_bench", "alpaca_eval", "truthfulqa"),
            "num_fewshot": 0,
            "batch_size": 1,
            "output_format": OutputFormat.MARKDOWN,
            "aggregation": AggregationMethod.MEAN,
        },
        "reasoning": {
            "tasks": ("gsm8k", "math", "arc_challenge"),
            "num_fewshot": 8,
            "batch_size": 16,
            "output_format": OutputFormat.JSON,
            "aggregation": AggregationMethod.MEAN,
        },
    }

    config = configs.get(model_lower, configs["base"])

    # Apply num_tasks limit if specified
    if num_tasks is not None and num_tasks > 0:
        config["tasks"] = config["tasks"][:num_tasks]

    return config
