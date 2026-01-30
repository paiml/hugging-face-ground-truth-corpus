"""Benchmark configurations for model evaluation.

This module provides utilities for configuring and running standard
LLM benchmarks including MMLU, HellaSwag, TruthfulQA, ARC, WinoGrande,
GSM8K, HumanEval, and MBPP.

Examples:
    >>> from hf_gtc.evaluation.benchmarks import BenchmarkType, EvaluationMode
    >>> BenchmarkType.MMLU.value
    'mmlu'
    >>> EvaluationMode.ZERO_SHOT.value
    'zero_shot'
    >>> from hf_gtc.evaluation.benchmarks import ScoringMethod
    >>> ScoringMethod.EXACT_MATCH.value
    'exact_match'
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class BenchmarkType(Enum):
    """Standard LLM benchmark types.

    Attributes:
        MMLU: Massive Multitask Language Understanding benchmark.
        HELLASWAG: HellaSwag commonsense reasoning benchmark.
        TRUTHFULQA: TruthfulQA benchmark for truthfulness.
        ARC: AI2 Reasoning Challenge benchmark.
        WINOGRANDE: WinoGrande commonsense reasoning benchmark.
        GSM8K: Grade School Math 8K benchmark.
        HUMANEVAL: HumanEval code generation benchmark.
        MBPP: Mostly Basic Programming Problems benchmark.

    Examples:
        >>> BenchmarkType.MMLU.value
        'mmlu'
        >>> BenchmarkType.HELLASWAG.value
        'hellaswag'
        >>> BenchmarkType.TRUTHFULQA.value
        'truthfulqa'
    """

    MMLU = "mmlu"
    HELLASWAG = "hellaswag"
    TRUTHFULQA = "truthfulqa"
    ARC = "arc"
    WINOGRANDE = "winogrande"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


VALID_BENCHMARK_TYPES = frozenset(t.value for t in BenchmarkType)


class EvaluationMode(Enum):
    """Evaluation modes for benchmarks.

    Attributes:
        ZERO_SHOT: Direct prompting without examples.
        FEW_SHOT: Prompting with few-shot examples.
        CHAIN_OF_THOUGHT: Chain-of-thought prompting.

    Examples:
        >>> EvaluationMode.ZERO_SHOT.value
        'zero_shot'
        >>> EvaluationMode.FEW_SHOT.value
        'few_shot'
        >>> EvaluationMode.CHAIN_OF_THOUGHT.value
        'chain_of_thought'
    """

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


VALID_EVALUATION_MODES = frozenset(m.value for m in EvaluationMode)


class ScoringMethod(Enum):
    """Scoring methods for benchmark evaluation.

    Attributes:
        EXACT_MATCH: Exact string match scoring.
        F1: F1 score for token overlap.
        ACCURACY: Classification accuracy.
        PASS_AT_K: Pass@k for code generation.

    Examples:
        >>> ScoringMethod.EXACT_MATCH.value
        'exact_match'
        >>> ScoringMethod.F1.value
        'f1'
        >>> ScoringMethod.ACCURACY.value
        'accuracy'
        >>> ScoringMethod.PASS_AT_K.value
        'pass_at_k'
    """

    EXACT_MATCH = "exact_match"
    F1 = "f1"
    ACCURACY = "accuracy"
    PASS_AT_K = "pass_at_k"


VALID_SCORING_METHODS = frozenset(m.value for m in ScoringMethod)


class BenchmarkTask(Enum):
    """Standard benchmark tasks (legacy).

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
        benchmark_type: Type of benchmark to run.
        num_few_shot: Number of few-shot examples. Defaults to 0.
        evaluation_mode: Evaluation mode. Defaults to ZERO_SHOT.
        subset: Optional benchmark subset (e.g., MMLU subject).

    Examples:
        >>> config = BenchmarkConfig(benchmark_type=BenchmarkType.MMLU)
        >>> config.benchmark_type
        <BenchmarkType.MMLU: 'mmlu'>
        >>> config.num_few_shot
        0

        >>> config = BenchmarkConfig(
        ...     benchmark_type=BenchmarkType.MMLU,
        ...     num_few_shot=5,
        ...     evaluation_mode=EvaluationMode.FEW_SHOT,
        ...     subset="abstract_algebra",
        ... )
        >>> config.num_few_shot
        5
        >>> config.subset
        'abstract_algebra'
    """

    benchmark_type: BenchmarkType
    num_few_shot: int = 0
    evaluation_mode: EvaluationMode = EvaluationMode.ZERO_SHOT
    subset: str | None = None


@dataclass(frozen=True, slots=True)
class MMLUConfig:
    """Configuration for MMLU benchmark.

    Attributes:
        subjects: Tuple of subjects to evaluate. Defaults to None (all).
        num_few_shot: Number of few-shot examples. Defaults to 5.

    Examples:
        >>> config = MMLUConfig()
        >>> config.num_few_shot
        5
        >>> config.subjects is None
        True

        >>> config = MMLUConfig(
        ...     subjects=("abstract_algebra", "anatomy"),
        ...     num_few_shot=3,
        ... )
        >>> config.subjects
        ('abstract_algebra', 'anatomy')
    """

    subjects: tuple[str, ...] | None = None
    num_few_shot: int = 5


@dataclass(frozen=True, slots=True)
class HumanEvalConfig:
    """Configuration for HumanEval benchmark.

    Attributes:
        k_values: Tuple of k values for pass@k. Defaults to (1, 10, 100).
        timeout_seconds: Timeout for code execution. Defaults to 10.0.

    Examples:
        >>> config = HumanEvalConfig()
        >>> config.k_values
        (1, 10, 100)
        >>> config.timeout_seconds
        10.0

        >>> config = HumanEvalConfig(k_values=(1, 5), timeout_seconds=5.0)
        >>> config.k_values
        (1, 5)
    """

    k_values: tuple[int, ...] = (1, 10, 100)
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Result from a benchmark evaluation.

    Attributes:
        benchmark: The benchmark type evaluated.
        score: Primary score (0-1 scale).
        num_samples: Number of samples evaluated.
        breakdown: Per-task or per-subject breakdown of scores.

    Examples:
        >>> result = BenchmarkResult(
        ...     benchmark=BenchmarkType.MMLU,
        ...     score=0.65,
        ...     num_samples=1000,
        ...     breakdown={"abstract_algebra": 0.7, "anatomy": 0.6},
        ... )
        >>> result.score
        0.65
        >>> result.num_samples
        1000
    """

    benchmark: BenchmarkType
    score: float
    num_samples: int
    breakdown: dict[str, float]


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    """Statistics from benchmark evaluation.

    Attributes:
        overall_score: Overall benchmark score (0-1 scale).
        per_task_scores: Dictionary mapping tasks to scores.
        confidence_interval: Tuple of (lower, upper) 95% CI bounds.

    Examples:
        >>> stats = BenchmarkStats(
        ...     overall_score=0.72,
        ...     per_task_scores={"mmlu": 0.65, "hellaswag": 0.79},
        ...     confidence_interval=(0.70, 0.74),
        ... )
        >>> stats.overall_score
        0.72
        >>> stats.per_task_scores["mmlu"]
        0.65
    """

    overall_score: float
    per_task_scores: dict[str, float]
    confidence_interval: tuple[float, float]


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
class LegacyBenchmarkConfig:
    """Legacy configuration for benchmark runs.

    Attributes:
        name: Name of the benchmark.
        task: Benchmark task type. Defaults to CUSTOM.
        num_samples: Number of samples to evaluate. Defaults to None (all).
        batch_size: Batch size for evaluation. Defaults to 32.
        warmup_runs: Number of warmup runs. Defaults to 1.
        num_runs: Number of timed runs. Defaults to 3.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test", num_samples=100)
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
class LegacyBenchmarkResult:
    """Complete results from a legacy benchmark run.

    Attributes:
        config: Benchmark configuration used.
        timing: Timing metrics.
        metrics: Task-specific metrics dictionary.
        samples_evaluated: Number of samples evaluated.
        success: Whether the benchmark completed successfully.
        error_message: Error message if benchmark failed.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test")
        >>> timing = TimingResult(1.0, 100.0, 0.5, 0.8, 1.0)
        >>> result = LegacyBenchmarkResult(
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

    config: LegacyBenchmarkConfig
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
        >>> config = LegacyBenchmarkConfig(name="test")
        >>> runner = BenchmarkRunner(config)
        >>> runner.config.name
        'test'
    """

    config: LegacyBenchmarkConfig
    latencies: list[float] = field(default_factory=list)


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    """Validate benchmark configuration.

    Args:
        config: BenchmarkConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_few_shot is negative.
        ValueError: If few_shot mode but num_few_shot is 0.

    Examples:
        >>> config = BenchmarkConfig(benchmark_type=BenchmarkType.MMLU)
        >>> validate_benchmark_config(config)  # No error

        >>> validate_benchmark_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BenchmarkConfig(
        ...     benchmark_type=BenchmarkType.MMLU,
        ...     num_few_shot=-1,
        ... )
        >>> validate_benchmark_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_few_shot cannot be negative
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_few_shot < 0:
        msg = f"num_few_shot cannot be negative, got {config.num_few_shot}"
        raise ValueError(msg)

    if (
        config.evaluation_mode == EvaluationMode.FEW_SHOT
        and config.num_few_shot == 0
    ):
        msg = "num_few_shot must be > 0 when using FEW_SHOT evaluation mode"
        raise ValueError(msg)


def validate_mmlu_config(config: MMLUConfig) -> None:
    """Validate MMLU configuration.

    Args:
        config: MMLUConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_few_shot is negative.
        ValueError: If subjects contains empty strings.

    Examples:
        >>> config = MMLUConfig()
        >>> validate_mmlu_config(config)  # No error

        >>> validate_mmlu_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = MMLUConfig(num_few_shot=-1)
        >>> validate_mmlu_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_few_shot cannot be negative
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_few_shot < 0:
        msg = f"num_few_shot cannot be negative, got {config.num_few_shot}"
        raise ValueError(msg)

    if config.subjects is not None:
        for subject in config.subjects:
            if not subject:
                msg = "subjects cannot contain empty strings"
                raise ValueError(msg)


def validate_humaneval_config(config: HumanEvalConfig) -> None:
    """Validate HumanEval configuration.

    Args:
        config: HumanEvalConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If k_values is empty.
        ValueError: If any k value is not positive.
        ValueError: If timeout_seconds is not positive.

    Examples:
        >>> config = HumanEvalConfig()
        >>> validate_humaneval_config(config)  # No error

        >>> validate_humaneval_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = HumanEvalConfig(k_values=())
        >>> validate_humaneval_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: k_values cannot be empty

        >>> bad = HumanEvalConfig(timeout_seconds=-1.0)
        >>> validate_humaneval_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: timeout_seconds must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if len(config.k_values) == 0:
        msg = "k_values cannot be empty"
        raise ValueError(msg)

    for k in config.k_values:
        if k <= 0:
            msg = f"k values must be positive, got {k}"
            raise ValueError(msg)

    if config.timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {config.timeout_seconds}"
        raise ValueError(msg)


def validate_benchmark_result(result: BenchmarkResult) -> None:
    """Validate benchmark result.

    Args:
        result: BenchmarkResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If score is not in [0, 1].
        ValueError: If num_samples is not positive.

    Examples:
        >>> result = BenchmarkResult(
        ...     benchmark=BenchmarkType.MMLU,
        ...     score=0.65,
        ...     num_samples=100,
        ...     breakdown={},
        ... )
        >>> validate_benchmark_result(result)  # No error

        >>> validate_benchmark_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = BenchmarkResult(
        ...     benchmark=BenchmarkType.MMLU,
        ...     score=1.5,
        ...     num_samples=100,
        ...     breakdown={},
        ... )
        >>> validate_benchmark_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: score must be between 0 and 1
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    if not 0.0 <= result.score <= 1.0:
        msg = f"score must be between 0 and 1, got {result.score}"
        raise ValueError(msg)

    if result.num_samples <= 0:
        msg = f"num_samples must be positive, got {result.num_samples}"
        raise ValueError(msg)


def validate_benchmark_stats(stats: BenchmarkStats) -> None:
    """Validate benchmark statistics.

    Args:
        stats: BenchmarkStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If overall_score is not in [0, 1].
        ValueError: If confidence interval is invalid.

    Examples:
        >>> stats = BenchmarkStats(
        ...     overall_score=0.72,
        ...     per_task_scores={},
        ...     confidence_interval=(0.70, 0.74),
        ... )
        >>> validate_benchmark_stats(stats)  # No error

        >>> validate_benchmark_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = BenchmarkStats(
        ...     overall_score=1.5,
        ...     per_task_scores={},
        ...     confidence_interval=(0.70, 0.74),
        ... )
        >>> validate_benchmark_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overall_score must be between 0 and 1
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if not 0.0 <= stats.overall_score <= 1.0:
        msg = f"overall_score must be between 0 and 1, got {stats.overall_score}"
        raise ValueError(msg)

    lower, upper = stats.confidence_interval
    if lower > upper:
        msg = (
            f"confidence interval lower bound ({lower}) "
            f"cannot be greater than upper bound ({upper})"
        )
        raise ValueError(msg)


def validate_legacy_benchmark_config(config: LegacyBenchmarkConfig) -> None:
    """Validate legacy benchmark configuration.

    Args:
        config: LegacyBenchmarkConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If name is empty.
        ValueError: If batch_size is not positive.
        ValueError: If warmup_runs is negative.
        ValueError: If num_runs is not positive.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test", batch_size=16)
        >>> validate_legacy_benchmark_config(config)  # No error

        >>> validate_legacy_benchmark_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LegacyBenchmarkConfig(name="")
        >>> validate_legacy_benchmark_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
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


def create_benchmark_config(
    benchmark_type: BenchmarkType,
    *,
    num_few_shot: int = 0,
    evaluation_mode: EvaluationMode = EvaluationMode.ZERO_SHOT,
    subset: str | None = None,
) -> BenchmarkConfig:
    """Create and validate a benchmark configuration.

    Args:
        benchmark_type: Type of benchmark to run.
        num_few_shot: Number of few-shot examples. Defaults to 0.
        evaluation_mode: Evaluation mode. Defaults to ZERO_SHOT.
        subset: Optional benchmark subset.

    Returns:
        Validated BenchmarkConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_benchmark_config(BenchmarkType.MMLU)
        >>> config.benchmark_type
        <BenchmarkType.MMLU: 'mmlu'>

        >>> config = create_benchmark_config(
        ...     BenchmarkType.HELLASWAG,
        ...     num_few_shot=10,
        ...     evaluation_mode=EvaluationMode.FEW_SHOT,
        ... )
        >>> config.num_few_shot
        10
    """
    config = BenchmarkConfig(
        benchmark_type=benchmark_type,
        num_few_shot=num_few_shot,
        evaluation_mode=evaluation_mode,
        subset=subset,
    )
    validate_benchmark_config(config)
    return config


def create_mmlu_config(
    *,
    subjects: tuple[str, ...] | None = None,
    num_few_shot: int = 5,
) -> MMLUConfig:
    """Create and validate an MMLU configuration.

    Args:
        subjects: Tuple of subjects to evaluate. Defaults to None (all).
        num_few_shot: Number of few-shot examples. Defaults to 5.

    Returns:
        Validated MMLUConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_mmlu_config()
        >>> config.num_few_shot
        5

        >>> config = create_mmlu_config(
        ...     subjects=("abstract_algebra", "anatomy"),
        ...     num_few_shot=3,
        ... )
        >>> config.subjects
        ('abstract_algebra', 'anatomy')
    """
    config = MMLUConfig(
        subjects=subjects,
        num_few_shot=num_few_shot,
    )
    validate_mmlu_config(config)
    return config


def create_humaneval_config(
    *,
    k_values: tuple[int, ...] = (1, 10, 100),
    timeout_seconds: float = 10.0,
) -> HumanEvalConfig:
    """Create and validate a HumanEval configuration.

    Args:
        k_values: Tuple of k values for pass@k. Defaults to (1, 10, 100).
        timeout_seconds: Timeout for code execution. Defaults to 10.0.

    Returns:
        Validated HumanEvalConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_humaneval_config()
        >>> config.k_values
        (1, 10, 100)

        >>> config = create_humaneval_config(k_values=(1, 5), timeout_seconds=5.0)
        >>> config.k_values
        (1, 5)
    """
    config = HumanEvalConfig(
        k_values=k_values,
        timeout_seconds=timeout_seconds,
    )
    validate_humaneval_config(config)
    return config


def create_benchmark_result(
    benchmark: BenchmarkType,
    score: float,
    num_samples: int,
    breakdown: dict[str, float] | None = None,
) -> BenchmarkResult:
    """Create and validate a benchmark result.

    Args:
        benchmark: The benchmark type evaluated.
        score: Primary score (0-1 scale).
        num_samples: Number of samples evaluated.
        breakdown: Per-task breakdown of scores. Defaults to empty dict.

    Returns:
        Validated BenchmarkResult instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_benchmark_result(
        ...     BenchmarkType.MMLU,
        ...     0.65,
        ...     1000,
        ... )
        >>> result.score
        0.65

        >>> result = create_benchmark_result(
        ...     BenchmarkType.HELLASWAG,
        ...     0.79,
        ...     500,
        ...     {"easy": 0.85, "hard": 0.70},
        ... )
        >>> result.breakdown["easy"]
        0.85
    """
    result = BenchmarkResult(
        benchmark=benchmark,
        score=score,
        num_samples=num_samples,
        breakdown=breakdown if breakdown is not None else {},
    )
    validate_benchmark_result(result)
    return result


def list_benchmark_types() -> list[str]:
    """List all available benchmark types.

    Returns:
        Sorted list of benchmark type names.

    Examples:
        >>> types = list_benchmark_types()
        >>> "mmlu" in types
        True
        >>> "hellaswag" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_BENCHMARK_TYPES)


def get_benchmark_type(name: str) -> BenchmarkType:
    """Get BenchmarkType enum from string name.

    Args:
        name: Name of the benchmark type.

    Returns:
        Corresponding BenchmarkType enum value.

    Raises:
        ValueError: If name is not a valid benchmark type.

    Examples:
        >>> get_benchmark_type("mmlu")
        <BenchmarkType.MMLU: 'mmlu'>

        >>> get_benchmark_type("hellaswag")
        <BenchmarkType.HELLASWAG: 'hellaswag'>

        >>> get_benchmark_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid benchmark type: invalid
    """
    if name not in VALID_BENCHMARK_TYPES:
        msg = f"invalid benchmark type: {name}"
        raise ValueError(msg)

    return BenchmarkType(name)


def validate_benchmark_type(name: str) -> bool:
    """Check if a benchmark type name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_benchmark_type("mmlu")
        True
        >>> validate_benchmark_type("invalid")
        False
        >>> validate_benchmark_type("")
        False
    """
    return name in VALID_BENCHMARK_TYPES


def list_evaluation_modes() -> list[str]:
    """List all available evaluation modes.

    Returns:
        Sorted list of evaluation mode names.

    Examples:
        >>> modes = list_evaluation_modes()
        >>> "zero_shot" in modes
        True
        >>> "few_shot" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_EVALUATION_MODES)


def get_evaluation_mode(name: str) -> EvaluationMode:
    """Get EvaluationMode enum from string name.

    Args:
        name: Name of the evaluation mode.

    Returns:
        Corresponding EvaluationMode enum value.

    Raises:
        ValueError: If name is not a valid evaluation mode.

    Examples:
        >>> get_evaluation_mode("zero_shot")
        <EvaluationMode.ZERO_SHOT: 'zero_shot'>

        >>> get_evaluation_mode("few_shot")
        <EvaluationMode.FEW_SHOT: 'few_shot'>

        >>> get_evaluation_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid evaluation mode: invalid
    """
    if name not in VALID_EVALUATION_MODES:
        msg = f"invalid evaluation mode: {name}"
        raise ValueError(msg)

    return EvaluationMode(name)


def validate_evaluation_mode(name: str) -> bool:
    """Check if an evaluation mode name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_evaluation_mode("zero_shot")
        True
        >>> validate_evaluation_mode("invalid")
        False
        >>> validate_evaluation_mode("")
        False
    """
    return name in VALID_EVALUATION_MODES


def list_scoring_methods() -> list[str]:
    """List all available scoring methods.

    Returns:
        Sorted list of scoring method names.

    Examples:
        >>> methods = list_scoring_methods()
        >>> "exact_match" in methods
        True
        >>> "accuracy" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_SCORING_METHODS)


def get_scoring_method(name: str) -> ScoringMethod:
    """Get ScoringMethod enum from string name.

    Args:
        name: Name of the scoring method.

    Returns:
        Corresponding ScoringMethod enum value.

    Raises:
        ValueError: If name is not a valid scoring method.

    Examples:
        >>> get_scoring_method("exact_match")
        <ScoringMethod.EXACT_MATCH: 'exact_match'>

        >>> get_scoring_method("accuracy")
        <ScoringMethod.ACCURACY: 'accuracy'>

        >>> get_scoring_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid scoring method: invalid
    """
    if name not in VALID_SCORING_METHODS:
        msg = f"invalid scoring method: {name}"
        raise ValueError(msg)

    return ScoringMethod(name)


def validate_scoring_method(name: str) -> bool:
    """Check if a scoring method name is valid.

    Args:
        name: Name to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_scoring_method("exact_match")
        True
        >>> validate_scoring_method("invalid")
        False
        >>> validate_scoring_method("")
        False
    """
    return name in VALID_SCORING_METHODS


def list_benchmark_tasks() -> list[str]:
    """List all available benchmark tasks (legacy).

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


def calculate_benchmark_score(
    correct: int,
    total: int,
) -> float:
    """Calculate benchmark score as accuracy.

    Args:
        correct: Number of correct predictions.
        total: Total number of samples.

    Returns:
        Score as a float between 0 and 1.

    Raises:
        ValueError: If correct is negative.
        ValueError: If total is not positive.
        ValueError: If correct > total.

    Examples:
        >>> calculate_benchmark_score(80, 100)
        0.8

        >>> calculate_benchmark_score(0, 100)
        0.0

        >>> calculate_benchmark_score(100, 100)
        1.0

        >>> calculate_benchmark_score(-1, 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: correct cannot be negative

        >>> calculate_benchmark_score(50, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total must be positive
    """
    if correct < 0:
        msg = f"correct cannot be negative, got {correct}"
        raise ValueError(msg)

    if total <= 0:
        msg = f"total must be positive, got {total}"
        raise ValueError(msg)

    if correct > total:
        msg = f"correct ({correct}) cannot be greater than total ({total})"
        raise ValueError(msg)

    return correct / total


def aggregate_results(
    results: Sequence[BenchmarkResult],
) -> BenchmarkStats:
    """Aggregate multiple benchmark results into statistics.

    Args:
        results: Sequence of benchmark results to aggregate.

    Returns:
        BenchmarkStats with aggregated scores.

    Raises:
        ValueError: If results is None or empty.

    Examples:
        >>> r1 = BenchmarkResult(BenchmarkType.MMLU, 0.65, 100, {})
        >>> r2 = BenchmarkResult(BenchmarkType.HELLASWAG, 0.79, 100, {})
        >>> stats = aggregate_results([r1, r2])
        >>> 0.70 <= stats.overall_score <= 0.75
        True

        >>> aggregate_results(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be None

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

    # Calculate weighted average (by num_samples)
    total_samples = sum(r.num_samples for r in results)
    weighted_sum = sum(r.score * r.num_samples for r in results)
    overall_score = weighted_sum / total_samples

    # Collect per-task scores
    per_task_scores = {r.benchmark.value: r.score for r in results}

    # Calculate confidence interval
    lower, upper = calculate_confidence_interval(
        [r.score for r in results],
        confidence_level=0.95,
    )

    return BenchmarkStats(
        overall_score=overall_score,
        per_task_scores=per_task_scores,
        confidence_interval=(lower, upper),
    )


def calculate_confidence_interval(
    scores: Sequence[float],
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate confidence interval for a sequence of scores.

    Args:
        scores: Sequence of score values.
        confidence_level: Confidence level (0-1). Defaults to 0.95.

    Returns:
        Tuple of (lower, upper) confidence interval bounds.

    Raises:
        ValueError: If scores is None or empty.
        ValueError: If confidence_level is not in (0, 1).

    Examples:
        >>> lower, upper = calculate_confidence_interval([0.7, 0.8, 0.75])
        >>> 0.65 <= lower <= 0.75
        True
        >>> 0.75 <= upper <= 0.85
        True

        >>> calculate_confidence_interval(
        ...     None
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores cannot be None

        >>> calculate_confidence_interval([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores cannot be empty
    """
    if scores is None:
        msg = "scores cannot be None"
        raise ValueError(msg)

    if len(scores) == 0:
        msg = "scores cannot be empty"
        raise ValueError(msg)

    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(msg)

    n = len(scores)
    mean = sum(scores) / n

    if n == 1:
        return (mean, mean)

    # Calculate standard error
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std_error = math.sqrt(variance / n)

    # Z-scores for common confidence levels
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)

    margin = z * std_error
    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)

    return (lower, upper)


def compare_benchmarks(
    results_a: Sequence[BenchmarkResult],
    results_b: Sequence[BenchmarkResult],
) -> dict[str, Any]:
    """Compare two sets of benchmark results.

    Args:
        results_a: First set of benchmark results.
        results_b: Second set of benchmark results.

    Returns:
        Dictionary with comparison statistics.

    Raises:
        ValueError: If either results sequence is None or empty.

    Examples:
        >>> r1 = [BenchmarkResult(BenchmarkType.MMLU, 0.65, 100, {})]
        >>> r2 = [BenchmarkResult(BenchmarkType.MMLU, 0.70, 100, {})]
        >>> comparison = compare_benchmarks(r1, r2)
        >>> comparison["better"] == "b"
        True

        >>> compare_benchmarks(
        ...     None, []
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results_a cannot be None
    """
    if results_a is None:
        msg = "results_a cannot be None"
        raise ValueError(msg)

    if results_b is None:
        msg = "results_b cannot be None"
        raise ValueError(msg)

    if len(results_a) == 0:
        msg = "results_a cannot be empty"
        raise ValueError(msg)

    if len(results_b) == 0:
        msg = "results_b cannot be empty"
        raise ValueError(msg)

    stats_a = aggregate_results(results_a)
    stats_b = aggregate_results(results_b)

    # Determine which is better
    if stats_a.overall_score > stats_b.overall_score:
        better = "a"
    elif stats_b.overall_score > stats_a.overall_score:
        better = "b"
    else:
        better = "tie"

    # Per-benchmark comparison
    per_benchmark: dict[str, dict[str, float | str]] = {}

    # Create maps by benchmark type
    scores_a = {r.benchmark.value: r.score for r in results_a}
    scores_b = {r.benchmark.value: r.score for r in results_b}

    all_benchmarks = set(scores_a.keys()) | set(scores_b.keys())
    for benchmark in all_benchmarks:
        score_a = scores_a.get(benchmark)
        score_b = scores_b.get(benchmark)

        if score_a is not None and score_b is not None:
            diff = score_b - score_a
            per_benchmark[benchmark] = {
                "score_a": score_a,
                "score_b": score_b,
                "difference": diff,
                "better": "b" if diff > 0 else "a" if diff < 0 else "tie",
            }

    return {
        "overall_a": stats_a.overall_score,
        "overall_b": stats_b.overall_score,
        "difference": stats_b.overall_score - stats_a.overall_score,
        "better": better,
        "per_benchmark": per_benchmark,
        "ci_a": stats_a.confidence_interval,
        "ci_b": stats_b.confidence_interval,
    }


def format_benchmark_stats(stats: BenchmarkStats) -> str:
    """Format benchmark statistics as a human-readable string.

    Args:
        stats: BenchmarkStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = BenchmarkStats(
        ...     overall_score=0.72,
        ...     per_task_scores={"mmlu": 0.65, "hellaswag": 0.79},
        ...     confidence_interval=(0.70, 0.74),
        ... )
        >>> formatted = format_benchmark_stats(stats)
        >>> "0.72" in formatted
        True
        >>> "mmlu" in formatted
        True

        >>> format_benchmark_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        "Benchmark Statistics",
        "=" * 40,
        f"Overall Score: {stats.overall_score:.4f}",
        f"95% Confidence Interval: [{stats.confidence_interval[0]:.4f}, "
        f"{stats.confidence_interval[1]:.4f}]",
        "",
        "Per-Task Scores:",
    ]

    for task, score in sorted(stats.per_task_scores.items()):
        lines.append(f"  {task}: {score:.4f}")

    return "\n".join(lines)


def get_recommended_benchmark_config(
    model_type: str,
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Get recommended benchmark configuration for a model type.

    Args:
        model_type: Type of model (e.g., "llm", "code", "instruction").
        task_type: Specific task type (optional).

    Returns:
        Dictionary with recommended configuration.

    Raises:
        ValueError: If model_type is None or empty.

    Examples:
        >>> config = get_recommended_benchmark_config("llm")
        >>> "benchmarks" in config
        True
        >>> BenchmarkType.MMLU in config["benchmarks"]
        True

        >>> config = get_recommended_benchmark_config("code")
        >>> BenchmarkType.HUMANEVAL in config["benchmarks"]
        True

        >>> get_recommended_benchmark_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty
    """
    if model_type is None:
        msg = "model_type cannot be None"
        raise ValueError(msg)

    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    # Base configuration for general LLMs
    base_config: dict[str, Any] = {
        "benchmarks": [
            BenchmarkType.MMLU,
            BenchmarkType.HELLASWAG,
            BenchmarkType.TRUTHFULQA,
            BenchmarkType.ARC,
            BenchmarkType.WINOGRANDE,
        ],
        "evaluation_mode": EvaluationMode.FEW_SHOT,
        "num_few_shot": 5,
        "scoring_method": ScoringMethod.ACCURACY,
    }

    # Model-specific configurations
    model_configs: dict[str, dict[str, Any]] = {
        "llm": {
            "benchmarks": [
                BenchmarkType.MMLU,
                BenchmarkType.HELLASWAG,
                BenchmarkType.TRUTHFULQA,
                BenchmarkType.ARC,
                BenchmarkType.WINOGRANDE,
            ],
            "evaluation_mode": EvaluationMode.FEW_SHOT,
            "num_few_shot": 5,
            "scoring_method": ScoringMethod.ACCURACY,
        },
        "code": {
            "benchmarks": [
                BenchmarkType.HUMANEVAL,
                BenchmarkType.MBPP,
            ],
            "evaluation_mode": EvaluationMode.ZERO_SHOT,
            "num_few_shot": 0,
            "scoring_method": ScoringMethod.PASS_AT_K,
            "humaneval_config": {
                "k_values": (1, 10, 100),
                "timeout_seconds": 10.0,
            },
        },
        "instruction": {
            "benchmarks": [
                BenchmarkType.MMLU,
                BenchmarkType.TRUTHFULQA,
            ],
            "evaluation_mode": EvaluationMode.ZERO_SHOT,
            "num_few_shot": 0,
            "scoring_method": ScoringMethod.ACCURACY,
        },
        "math": {
            "benchmarks": [
                BenchmarkType.GSM8K,
            ],
            "evaluation_mode": EvaluationMode.CHAIN_OF_THOUGHT,
            "num_few_shot": 8,
            "scoring_method": ScoringMethod.EXACT_MATCH,
        },
        "reasoning": {
            "benchmarks": [
                BenchmarkType.HELLASWAG,
                BenchmarkType.WINOGRANDE,
                BenchmarkType.ARC,
            ],
            "evaluation_mode": EvaluationMode.FEW_SHOT,
            "num_few_shot": 25,
            "scoring_method": ScoringMethod.ACCURACY,
        },
    }

    config = model_configs.get(model_type.lower(), base_config)

    # Apply task-specific adjustments
    if task_type:
        task_lower = task_type.lower()
        if task_lower == "few_shot":
            config["evaluation_mode"] = EvaluationMode.FEW_SHOT
            config["num_few_shot"] = 5
        elif task_lower == "zero_shot":
            config["evaluation_mode"] = EvaluationMode.ZERO_SHOT
            config["num_few_shot"] = 0
        elif task_lower == "cot":
            config["evaluation_mode"] = EvaluationMode.CHAIN_OF_THOUGHT
            config["num_few_shot"] = 8

    return config


# Legacy compatibility functions


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


def create_benchmark_runner(config: LegacyBenchmarkConfig) -> BenchmarkRunner:
    """Create a benchmark runner with the given configuration.

    Args:
        config: Benchmark configuration.

    Returns:
        Configured BenchmarkRunner instance.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test")
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

    validate_legacy_benchmark_config(config)
    return BenchmarkRunner(config=config)


def run_benchmark(
    runner: BenchmarkRunner,
    data: Sequence[Any],
    inference_fn: Callable[[Sequence[Any]], Any],
    metrics_fn: Callable[[Any, Sequence[Any]], dict[str, float]] | None = None,
) -> LegacyBenchmarkResult:
    """Execute a benchmark run.

    Args:
        runner: Configured benchmark runner.
        data: Data samples to evaluate.
        inference_fn: Function that performs inference on a batch.
        metrics_fn: Optional function to compute metrics from predictions.

    Returns:
        LegacyBenchmarkResult with timing and metrics.

    Raises:
        ValueError: If runner is None.
        ValueError: If data is None.
        ValueError: If inference_fn is None.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
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

        return LegacyBenchmarkResult(
            config=config,
            timing=timing,
            metrics=metrics,
            samples_evaluated=len(samples),
            success=True,
            error_message=None,
        )

    except Exception as e:
        empty_timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        return LegacyBenchmarkResult(
            config=config,
            timing=empty_timing,
            metrics={},
            samples_evaluated=0,
            success=False,
            error_message=str(e),
        )


def compare_benchmark_results(
    results: Sequence[LegacyBenchmarkResult],
) -> dict[str, Any]:
    """Compare multiple benchmark results.

    Args:
        results: Sequence of benchmark results to compare.

    Returns:
        Dictionary with comparison statistics.

    Raises:
        ValueError: If results is None or empty.

    Examples:
        >>> config1 = LegacyBenchmarkConfig(name="model1")
        >>> config2 = LegacyBenchmarkConfig(name="model2")
        >>> timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        >>> timing2 = TimingResult(0.8, 125.0, 4.0, 6.0, 8.0)
        >>> r1 = LegacyBenchmarkResult(
        ...     config1, timing1, {"accuracy": 0.9}, 100, True, None
        ... )
        >>> r2 = LegacyBenchmarkResult(
        ...     config2, timing2, {"accuracy": 0.95}, 100, True, None
        ... )
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


def format_benchmark_result(result: LegacyBenchmarkResult) -> str:
    """Format a benchmark result as a human-readable string.

    Args:
        result: Benchmark result to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test")
        >>> timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        >>> result = LegacyBenchmarkResult(
        ...     config, timing, {"acc": 0.95}, 100, True, None
        ... )
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
    results: Sequence[LegacyBenchmarkResult],
) -> dict[str, float]:
    """Aggregate metrics from multiple benchmark results.

    Args:
        results: Sequence of benchmark results.

    Returns:
        Dictionary with averaged metrics.

    Raises:
        ValueError: If results is None.

    Examples:
        >>> config = LegacyBenchmarkConfig(name="test")
        >>> timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        >>> r1 = LegacyBenchmarkResult(config, timing, {"acc": 0.9}, 100, True, None)
        >>> r2 = LegacyBenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)
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
