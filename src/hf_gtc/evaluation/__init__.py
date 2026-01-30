"""Evaluation recipes for HuggingFace models.

This module provides utilities for computing metrics,
running benchmarks, leaderboard integration, and evaluating model performance.

Examples:
    >>> from hf_gtc.evaluation import BenchmarkConfig, LeaderboardConfig
    >>> config = BenchmarkConfig(name="test")
    >>> config.name
    'test'
    >>> lb_config = LeaderboardConfig(name="my-board")
    >>> lb_config.name
    'my-board'
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
from hf_gtc.evaluation.leaderboards import (
    Leaderboard,
    LeaderboardCategory,
    LeaderboardConfig,
    LeaderboardEntry,
    ModelScore,
    SubmissionResult,
    SubmissionStatus,
    add_entry,
    compare_entries,
    compute_average_score,
    compute_leaderboard_stats,
    create_leaderboard,
    create_submission,
    filter_entries_by_size,
    find_entry_by_model,
    format_leaderboard,
    get_category,
    get_score_by_metric,
    get_top_entries,
    list_categories,
    list_submission_statuses,
    parse_submission_result,
    validate_category,
    validate_leaderboard_config,
    validate_submission_status,
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
    "Leaderboard",
    "LeaderboardCategory",
    "LeaderboardConfig",
    "LeaderboardEntry",
    "ModelScore",
    "SubmissionResult",
    "SubmissionStatus",
    "TimingResult",
    "add_entry",
    "aggregate_benchmark_results",
    "compare_benchmark_results",
    "compare_entries",
    "compute_accuracy",
    "compute_average_score",
    "compute_classification_metrics",
    "compute_f1",
    "compute_leaderboard_stats",
    "compute_mean_loss",
    "compute_percentile",
    "compute_perplexity",
    "compute_precision",
    "compute_recall",
    "compute_timing_stats",
    "create_benchmark_runner",
    "create_compute_metrics_fn",
    "create_leaderboard",
    "create_submission",
    "filter_entries_by_size",
    "find_entry_by_model",
    "format_benchmark_result",
    "format_leaderboard",
    "get_benchmark_task",
    "get_category",
    "get_score_by_metric",
    "get_top_entries",
    "list_benchmark_tasks",
    "list_categories",
    "list_submission_statuses",
    "parse_submission_result",
    "run_benchmark",
    "validate_benchmark_config",
    "validate_benchmark_task",
    "validate_category",
    "validate_leaderboard_config",
    "validate_submission_status",
]
