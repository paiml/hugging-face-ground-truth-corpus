"""Benchmark runners for model evaluation.

This module provides utilities for running benchmarks, measuring
performance metrics, and comparing model capabilities across
standard datasets and tasks.

Examples:
    >>> from hf_gtc.evaluation.benchmarks import BenchmarkConfig
    >>> config = BenchmarkConfig(name="test", num_samples=100)
    >>> config.name
    'test'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class BenchmarkTask(Enum):
    """Standard benchmark tasks.

    Attributes:
        TEXT_CLASSIFICATION: Text classification task.
        QUESTION_ANSWERING: Question answering task.
        SUMMARIZATION: Text summarization task.
        TRANSLATION: Machine translation task.
        NER: Named entity recognition task.
        SENTIMENT: Sentiment analysis task.
        CUSTOM: Custom user-defined task.

    Examples:
        >>> BenchmarkTask.TEXT_CLASSIFICATION.value
        'text_classification'
        >>> BenchmarkTask.QUESTION_ANSWERING.value
        'question_answering'
    """

    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    NER = "ner"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


VALID_BENCHMARK_TASKS = frozenset(t.value for t in BenchmarkTask)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        name: Name of the benchmark.
        task: Benchmark task type. Defaults to CUSTOM.
        num_samples: Number of samples to evaluate. Defaults to None (all).
        batch_size: Batch size for evaluation. Defaults to 32.
        warmup_runs: Number of warmup runs. Defaults to 1.
        num_runs: Number of timed runs. Defaults to 3.

    Examples:
        >>> config = BenchmarkConfig(name="test", num_samples=100)
        >>> config.name
        'test'
        >>> config.num_samples
        100
    """

    name: str
    task: BenchmarkTask = BenchmarkTask.CUSTOM
    num_samples: int | None = None
    batch_size: int = 32
    warmup_runs: int = 1
    num_runs: int = 3


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Timing results from a benchmark run.

    Attributes:
        total_time: Total elapsed time in seconds.
        samples_per_second: Throughput in samples per second.
        latency_p50: 50th percentile latency in milliseconds.
        latency_p90: 90th percentile latency in milliseconds.
        latency_p99: 99th percentile latency in milliseconds.

    Examples:
        >>> result = TimingResult(
        ...     total_time=1.5,
        ...     samples_per_second=1000.0,
        ...     latency_p50=0.8,
        ...     latency_p90=1.2,
        ...     latency_p99=1.8,
        ... )
        >>> result.total_time
        1.5
    """

    total_time: float
    samples_per_second: float
    latency_p50: float
    latency_p90: float
    latency_p99: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete results from a benchmark run.

    Attributes:
        config: Benchmark configuration used.
        timing: Timing metrics.
        metrics: Task-specific metrics dictionary.
        samples_evaluated: Number of samples evaluated.
        success: Whether the benchmark completed successfully.
        error_message: Error message if benchmark failed.

    Examples:
        >>> config = BenchmarkConfig(name="test")
        >>> timing = TimingResult(1.0, 100.0, 0.5, 0.8, 1.0)
        >>> result = BenchmarkResult(
        ...     config=config,
        ...     timing=timing,
        ...     metrics={"accuracy": 0.95},
        ...     samples_evaluated=100,
        ...     success=True,
        ...     error_message=None,
        ... )
        >>> result.success
        True
    """

    config: BenchmarkConfig
    timing: TimingResult
    metrics: dict[str, float]
    samples_evaluated: int
    success: bool
    error_message: str | None


@dataclass
class BenchmarkRunner:
    """Runner for executing benchmarks.

    Attributes:
        config: Benchmark configuration.
        latencies: List of latencies from runs.

    Examples:
        >>> config = BenchmarkConfig(name="test")
        >>> runner = BenchmarkRunner(config)
        >>> runner.config.name
        'test'
    """

    config: BenchmarkConfig
    latencies: list[float] = field(default_factory=list)


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    """Validate benchmark configuration.

    Args:
        config: BenchmarkConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If name is empty.
        ValueError: If batch_size is not positive.
        ValueError: If warmup_runs is negative.
        ValueError: If num_runs is not positive.

    Examples:
        >>> config = BenchmarkConfig(name="test", batch_size=16)
        >>> validate_benchmark_config(config)  # No error

        >>> validate_benchmark_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BenchmarkConfig(name="")
        >>> validate_benchmark_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.warmup_runs < 0:
        msg = f"warmup_runs cannot be negative, got {config.warmup_runs}"
        raise ValueError(msg)

    if config.num_runs <= 0:
        msg = f"num_runs must be positive, got {config.num_runs}"
        raise ValueError(msg)


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    """Compute a percentile from a sequence of values.

    Args:
        values: Sequence of numeric values.
        percentile: Percentile to compute (0-100).

    Returns:
        The computed percentile value.

    Raises:
        ValueError: If values is None or empty.
        ValueError: If percentile is not in [0, 100].

    Examples:
        >>> compute_percentile([1, 2, 3, 4, 5], 50)
        3.0

        >>> compute_percentile([1, 2, 3, 4, 5], 0)
        1.0

        >>> compute_percentile([1, 2, 3, 4, 5], 100)
        5.0

        >>> compute_percentile([], 50)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: values cannot be empty

        >>> compute_percentile([1, 2], 150)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: percentile must be between 0 and 100
    """
    if values is None:
        msg = "values cannot be None"
        raise ValueError(msg)

    if len(values) == 0:
        msg = "values cannot be empty"
        raise ValueError(msg)

    if not 0 <= percentile <= 100:
        msg = f"percentile must be between 0 and 100, got {percentile}"
        raise ValueError(msg)

    sorted_values = sorted(values)
    n = len(sorted_values)

    if percentile == 0:
        return float(sorted_values[0])
    if percentile == 100:
        return float(sorted_values[-1])

    idx = (percentile / 100) * (n - 1)
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = idx - lower_idx

    return float(
        sorted_values[lower_idx]
        + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])
    )


def compute_timing_stats(
    latencies: Sequence[float],
    total_samples: int,
) -> TimingResult:
    """Compute timing statistics from latency measurements.

    Args:
        latencies: Sequence of latency measurements in seconds.
        total_samples: Total number of samples processed.

    Returns:
        TimingResult with computed statistics.

    Raises:
        ValueError: If latencies is None or empty.
        ValueError: If total_samples is not positive.

    Examples:
        >>> latencies = [0.1, 0.15, 0.12, 0.11, 0.13]
        >>> result = compute_timing_stats(latencies, 500)
        >>> result.total_time > 0
        True

        >>> compute_timing_stats([], 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: latencies cannot be empty
    """
    if latencies is None:
        msg = "latencies cannot be None"
        raise ValueError(msg)

    if len(latencies) == 0:
        msg = "latencies cannot be empty"
        raise ValueError(msg)

    if total_samples <= 0:
        msg = f"total_samples must be positive, got {total_samples}"
        raise ValueError(msg)

    total_time = sum(latencies)
    samples_per_sec = total_samples / total_time if total_time > 0 else 0.0

    # Convert to milliseconds for percentiles
    latencies_ms = [lat * 1000 for lat in latencies]

    return TimingResult(
        total_time=total_time,
        samples_per_second=samples_per_sec,
        latency_p50=compute_percentile(latencies_ms, 50),
        latency_p90=compute_percentile(latencies_ms, 90),
        latency_p99=compute_percentile(latencies_ms, 99),
    )


def create_benchmark_runner(config: BenchmarkConfig) -> BenchmarkRunner:
    """Create a benchmark runner with the given configuration.

    Args:
        config: Benchmark configuration.

    Returns:
        Configured BenchmarkRunner instance.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = BenchmarkConfig(name="test")
        >>> runner = create_benchmark_runner(config)
        >>> runner.config.name
        'test'

        >>> create_benchmark_runner(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_benchmark_config(config)
    return BenchmarkRunner(config=config)


def run_benchmark(
    runner: BenchmarkRunner,
    data: Sequence[Any],
    inference_fn: Callable[[Sequence[Any]], Any],
    metrics_fn: Callable[[Any, Sequence[Any]], dict[str, float]] | None = None,
) -> BenchmarkResult:
    """Execute a benchmark run.

    Args:
        runner: Configured benchmark runner.
        data: Data samples to evaluate.
        inference_fn: Function that performs inference on a batch.
        metrics_fn: Optional function to compute metrics from predictions.

    Returns:
        BenchmarkResult with timing and metrics.

    Raises:
        ValueError: If runner is None.
        ValueError: If data is None.
        ValueError: If inference_fn is None.

    Examples:
        >>> config = BenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
        >>> runner = create_benchmark_runner(config)
        >>> data = [1, 2, 3, 4, 5]
        >>> result = run_benchmark(runner, data, lambda x: x)
        >>> result.success
        True

        >>> run_benchmark(None, [], lambda x: x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: runner cannot be None
    """
    if runner is None:
        msg = "runner cannot be None"
        raise ValueError(msg)

    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    if inference_fn is None:
        msg = "inference_fn cannot be None"
        raise ValueError(msg)

    config = runner.config
    num_samples = config.num_samples or len(data)
    samples = data[:num_samples] if num_samples < len(data) else data

    try:
        # Warmup runs
        for _ in range(config.warmup_runs):
            _ = inference_fn(samples[: config.batch_size])

        # Timed runs
        latencies = []
        predictions = None
        for _ in range(config.num_runs):
            start = time.perf_counter()
            predictions = inference_fn(samples)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        runner.latencies.extend(latencies)

        # Compute timing stats
        timing = compute_timing_stats(latencies, len(samples) * config.num_runs)

        # Compute metrics if function provided
        metrics: dict[str, float] = {}
        if metrics_fn is not None and predictions is not None:
            metrics = metrics_fn(predictions, samples)

        return BenchmarkResult(
            config=config,
            timing=timing,
            metrics=metrics,
            samples_evaluated=len(samples),
            success=True,
            error_message=None,
        )

    except Exception as e:
        empty_timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        return BenchmarkResult(
            config=config,
            timing=empty_timing,
            metrics={},
            samples_evaluated=0,
            success=False,
            error_message=str(e),
        )


def compare_benchmark_results(
    results: Sequence[BenchmarkResult],
) -> dict[str, Any]:
    """Compare multiple benchmark results.

    Args:
        results: Sequence of benchmark results to compare.

    Returns:
        Dictionary with comparison statistics.

    Raises:
        ValueError: If results is None or empty.

    Examples:
        >>> config1 = BenchmarkConfig(name="model1")
        >>> config2 = BenchmarkConfig(name="model2")
        >>> timing1 = TimingResult(1.0, 100.0, 0.5, 0.8, 1.0)
        >>> timing2 = TimingResult(0.8, 125.0, 0.4, 0.6, 0.8)
        >>> r1 = BenchmarkResult(config1, timing1, {"accuracy": 0.9}, 100, True, None)
        >>> r2 = BenchmarkResult(config2, timing2, {"accuracy": 0.95}, 100, True, None)
        >>> comparison = compare_benchmark_results([r1, r2])
        >>> comparison["fastest"]
        'model2'

        >>> compare_benchmark_results([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be empty
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if len(results) == 0:
        msg = "results cannot be empty"
        raise ValueError(msg)

    successful = [r for r in results if r.success]
    if not successful:
        return {
            "total_benchmarks": len(results),
            "successful": 0,
            "failed": len(results),
            "fastest": None,
            "slowest": None,
        }

    fastest = min(successful, key=lambda r: r.timing.total_time)
    slowest = max(successful, key=lambda r: r.timing.total_time)

    # Collect all metrics
    all_metrics: dict[str, list[tuple[str, float]]] = {}
    for result in successful:
        for metric_name, value in result.metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append((result.config.name, value))

    # Find best for each metric
    best_metrics: dict[str, str] = {}
    for metric_name, values in all_metrics.items():
        best_name, _ = max(values, key=lambda x: x[1])
        best_metrics[metric_name] = best_name

    return {
        "total_benchmarks": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "fastest": fastest.config.name,
        "slowest": slowest.config.name,
        "throughput_range": {
            "min": min(r.timing.samples_per_second for r in successful),
            "max": max(r.timing.samples_per_second for r in successful),
        },
        "best_metrics": best_metrics,
    }


def list_benchmark_tasks() -> list[str]:
    """List all available benchmark tasks.

    Returns:
        Sorted list of benchmark task names.

    Examples:
        >>> tasks = list_benchmark_tasks()
        >>> "text_classification" in tasks
        True
        >>> "question_answering" in tasks
        True
        >>> tasks == sorted(tasks)
        True
    """
    return sorted(VALID_BENCHMARK_TASKS)


def validate_benchmark_task(task: str) -> bool:
    """Validate if a string is a valid benchmark task.

    Args:
        task: The task string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_benchmark_task("text_classification")
        True
        >>> validate_benchmark_task("question_answering")
        True
        >>> validate_benchmark_task("invalid")
        False
        >>> validate_benchmark_task("")
        False
    """
    return task in VALID_BENCHMARK_TASKS


def get_benchmark_task(name: str) -> BenchmarkTask:
    """Get BenchmarkTask enum from string name.

    Args:
        name: Name of the benchmark task.

    Returns:
        Corresponding BenchmarkTask enum value.

    Raises:
        ValueError: If name is not a valid benchmark task.

    Examples:
        >>> get_benchmark_task("text_classification")
        <BenchmarkTask.TEXT_CLASSIFICATION: 'text_classification'>

        >>> get_benchmark_task("custom")
        <BenchmarkTask.CUSTOM: 'custom'>

        >>> get_benchmark_task("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid benchmark task: invalid
    """
    if not validate_benchmark_task(name):
        msg = f"invalid benchmark task: {name}"
        raise ValueError(msg)

    return BenchmarkTask(name)


def format_benchmark_result(result: BenchmarkResult) -> str:
    """Format a benchmark result as a human-readable string.

    Args:
        result: Benchmark result to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> config = BenchmarkConfig(name="test")
        >>> timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        >>> result = BenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)
        >>> "test" in format_benchmark_result(result)
        True

        >>> format_benchmark_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    lines = [
        f"Benchmark: {result.config.name}",
        f"Task: {result.config.task.value}",
        f"Success: {result.success}",
    ]

    if result.success:
        lines.extend(
            [
                f"Samples evaluated: {result.samples_evaluated}",
                f"Total time: {result.timing.total_time:.3f}s",
                f"Throughput: {result.timing.samples_per_second:.1f} samples/s",
                f"Latency P50: {result.timing.latency_p50:.2f}ms",
                f"Latency P90: {result.timing.latency_p90:.2f}ms",
                f"Latency P99: {result.timing.latency_p99:.2f}ms",
            ]
        )

        if result.metrics:
            lines.append("Metrics:")
            for name, value in sorted(result.metrics.items()):
                lines.append(f"  {name}: {value:.4f}")
    else:
        lines.append(f"Error: {result.error_message}")

    return "\n".join(lines)


def aggregate_benchmark_results(
    results: Sequence[BenchmarkResult],
) -> dict[str, float]:
    """Aggregate metrics from multiple benchmark results.

    Args:
        results: Sequence of benchmark results.

    Returns:
        Dictionary with averaged metrics.

    Raises:
        ValueError: If results is None.

    Examples:
        >>> config = BenchmarkConfig(name="test")
        >>> timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        >>> r1 = BenchmarkResult(config, timing, {"acc": 0.9}, 100, True, None)
        >>> r2 = BenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)
        >>> agg = aggregate_benchmark_results([r1, r2])
        >>> 0.92 < agg["acc_mean"] < 0.93
        True

        >>> aggregate_benchmark_results(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be None
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    successful = [r for r in results if r.success]
    if not successful:
        return {}

    # Collect all metrics
    all_metrics: dict[str, list[float]] = {}
    for result in successful:
        for name, value in result.metrics.items():
            if name not in all_metrics:
                all_metrics[name] = []
            all_metrics[name].append(value)

    # Compute mean, min, max for each metric
    aggregated: dict[str, float] = {}
    for name, values in all_metrics.items():
        aggregated[f"{name}_mean"] = sum(values) / len(values)
        aggregated[f"{name}_min"] = min(values)
        aggregated[f"{name}_max"] = max(values)

    # Add timing aggregates
    total_times = [r.timing.total_time for r in successful]
    throughputs = [r.timing.samples_per_second for r in successful]

    aggregated["total_time_mean"] = sum(total_times) / len(total_times)
    aggregated["throughput_mean"] = sum(throughputs) / len(throughputs)

    return aggregated
