"""Evaluation recipes for HuggingFace models.

This module provides utilities for computing metrics,
running benchmarks, and evaluating model performance.

Examples:
    >>> from hf_gtc.evaluation import BenchmarkConfig, compute_accuracy
    >>> config = BenchmarkConfig(name="test")
    >>> config.name
    'test'
"""

from __future__ import annotations

from hf_gtc.evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTask,
    TimingResult,
    aggregate_benchmark_results,
    compare_benchmark_results,
    compute_percentile,
    compute_timing_stats,
    create_benchmark_runner,
    format_benchmark_result,
    get_benchmark_task,
    list_benchmark_tasks,
    run_benchmark,
    validate_benchmark_config,
    validate_benchmark_task,
)
from hf_gtc.evaluation.metrics import (
    ClassificationMetrics,
    compute_accuracy,
    compute_classification_metrics,
    compute_f1,
    compute_mean_loss,
    compute_perplexity,
    compute_precision,
    compute_recall,
    create_compute_metrics_fn,
)

__all__: list[str] = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "ClassificationMetrics",
    "TimingResult",
    "aggregate_benchmark_results",
    "compare_benchmark_results",
    "compute_accuracy",
    "compute_classification_metrics",
    "compute_f1",
    "compute_mean_loss",
    "compute_percentile",
    "compute_perplexity",
    "compute_precision",
    "compute_recall",
    "compute_timing_stats",
    "create_benchmark_runner",
    "create_compute_metrics_fn",
    "format_benchmark_result",
    "get_benchmark_task",
    "list_benchmark_tasks",
    "run_benchmark",
    "validate_benchmark_config",
    "validate_benchmark_task",
]
