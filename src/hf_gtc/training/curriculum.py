"""Curriculum Learning utilities for training.

This module provides functions for configuring and implementing curriculum
learning strategies, enabling progressive training from easy to hard samples
based on various difficulty metrics and pacing functions.

Examples:
    >>> from hf_gtc.training.curriculum import (
    ...     create_curriculum_config,
    ...     DifficultyMetric,
    ... )
    >>> config = create_curriculum_config()
    >>> config.difficulty_config.metric
    <DifficultyMetric.LENGTH: 'length'>
    >>> config.sampling_strategy
    <SamplingStrategy.THRESHOLD: 'threshold'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DifficultyMetric(Enum):
    """Metrics for measuring sample difficulty.

    Attributes:
        LENGTH: Difficulty based on sequence/sample length.
        PERPLEXITY: Difficulty based on model perplexity on sample.
        LOSS: Difficulty based on training loss on sample.
        CONFIDENCE: Difficulty based on model confidence (inverse).
        MANUAL: Manually assigned difficulty scores.

    Examples:
        >>> DifficultyMetric.LENGTH.value
        'length'
        >>> DifficultyMetric.PERPLEXITY.value
        'perplexity'
        >>> DifficultyMetric.LOSS.value
        'loss'
    """

    LENGTH = "length"
    PERPLEXITY = "perplexity"
    LOSS = "loss"
    CONFIDENCE = "confidence"
    MANUAL = "manual"


class PacingFunction(Enum):
    """Pacing functions for curriculum progression.

    Attributes:
        LINEAR: Linear increase in difficulty over time.
        EXPONENTIAL: Exponential increase in difficulty.
        STEP: Step-wise increase at fixed intervals.
        SELF_PACED: Adaptive pacing based on model performance.
        COMPETENCE: Competence-based curriculum learning.

    Examples:
        >>> PacingFunction.LINEAR.value
        'linear'
        >>> PacingFunction.EXPONENTIAL.value
        'exponential'
        >>> PacingFunction.SELF_PACED.value
        'self_paced'
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    SELF_PACED = "self_paced"
    COMPETENCE = "competence"


class SamplingStrategy(Enum):
    """Strategies for sampling curriculum samples.

    Attributes:
        THRESHOLD: Include samples below current difficulty threshold.
        WEIGHTED: Weight samples by difficulty-based probability.
        PROBABILISTIC: Probabilistically include samples based on difficulty.

    Examples:
        >>> SamplingStrategy.THRESHOLD.value
        'threshold'
        >>> SamplingStrategy.WEIGHTED.value
        'weighted'
        >>> SamplingStrategy.PROBABILISTIC.value
        'probabilistic'
    """

    THRESHOLD = "threshold"
    WEIGHTED = "weighted"
    PROBABILISTIC = "probabilistic"


VALID_DIFFICULTY_METRICS = frozenset(m.value for m in DifficultyMetric)
VALID_PACING_FUNCTIONS = frozenset(p.value for p in PacingFunction)
VALID_SAMPLING_STRATEGIES = frozenset(s.value for s in SamplingStrategy)


@dataclass(frozen=True, slots=True)
class DifficultyConfig:
    """Configuration for difficulty scoring.

    Attributes:
        metric: Metric to use for measuring difficulty.
        normalize: Whether to normalize difficulty scores to [0, 1].
        buckets: Number of difficulty buckets for discretization.
        ascending: Whether higher values mean more difficult.

    Examples:
        >>> config = DifficultyConfig(
        ...     metric=DifficultyMetric.LENGTH,
        ...     normalize=True,
        ...     buckets=10,
        ...     ascending=True,
        ... )
        >>> config.metric
        <DifficultyMetric.LENGTH: 'length'>
        >>> config.normalize
        True
        >>> config.buckets
        10
    """

    metric: DifficultyMetric
    normalize: bool
    buckets: int
    ascending: bool


@dataclass(frozen=True, slots=True)
class PacingConfig:
    """Configuration for curriculum pacing.

    Attributes:
        function: Pacing function to use.
        initial_difficulty: Starting difficulty level (0 to 1).
        target_difficulty: Final difficulty level (0 to 1).
        warmup_steps: Number of warmup steps before curriculum begins.

    Examples:
        >>> config = PacingConfig(
        ...     function=PacingFunction.LINEAR,
        ...     initial_difficulty=0.0,
        ...     target_difficulty=1.0,
        ...     warmup_steps=1000,
        ... )
        >>> config.function
        <PacingFunction.LINEAR: 'linear'>
        >>> config.initial_difficulty
        0.0
        >>> config.target_difficulty
        1.0
    """

    function: PacingFunction
    initial_difficulty: float
    target_difficulty: float
    warmup_steps: int


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Main configuration for curriculum learning.

    Attributes:
        difficulty_config: Configuration for difficulty scoring.
        pacing_config: Configuration for curriculum pacing.
        sampling_strategy: Strategy for sampling curriculum samples.

    Examples:
        >>> diff_config = DifficultyConfig(
        ...     metric=DifficultyMetric.LENGTH,
        ...     normalize=True,
        ...     buckets=10,
        ...     ascending=True,
        ... )
        >>> pacing_config = PacingConfig(
        ...     function=PacingFunction.LINEAR,
        ...     initial_difficulty=0.0,
        ...     target_difficulty=1.0,
        ...     warmup_steps=1000,
        ... )
        >>> config = CurriculumConfig(
        ...     difficulty_config=diff_config,
        ...     pacing_config=pacing_config,
        ...     sampling_strategy=SamplingStrategy.THRESHOLD,
        ... )
        >>> config.sampling_strategy
        <SamplingStrategy.THRESHOLD: 'threshold'>
    """

    difficulty_config: DifficultyConfig
    pacing_config: PacingConfig
    sampling_strategy: SamplingStrategy


@dataclass(frozen=True, slots=True)
class CurriculumStats:
    """Statistics from curriculum learning training.

    Attributes:
        current_difficulty: Current difficulty threshold.
        samples_seen: Total number of samples seen.
        competence_score: Current model competence (0 to 1).
        curriculum_progress: Progress through curriculum (0 to 1).

    Examples:
        >>> stats = CurriculumStats(
        ...     current_difficulty=0.5,
        ...     samples_seen=10000,
        ...     competence_score=0.8,
        ...     curriculum_progress=0.5,
        ... )
        >>> stats.current_difficulty
        0.5
        >>> stats.samples_seen
        10000
        >>> stats.competence_score
        0.8
    """

    current_difficulty: float
    samples_seen: int
    competence_score: float
    curriculum_progress: float


def validate_difficulty_config(config: DifficultyConfig) -> None:
    """Validate difficulty configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = DifficultyConfig(
        ...     metric=DifficultyMetric.LENGTH,
        ...     normalize=True,
        ...     buckets=10,
        ...     ascending=True,
        ... )
        >>> validate_difficulty_config(config)

        >>> bad_config = DifficultyConfig(
        ...     metric=DifficultyMetric.LENGTH,
        ...     normalize=True,
        ...     buckets=0,
        ...     ascending=True,
        ... )
        >>> validate_difficulty_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: buckets must be positive, got 0
    """
    if config.buckets <= 0:
        msg = f"buckets must be positive, got {config.buckets}"
        raise ValueError(msg)


def validate_pacing_config(config: PacingConfig) -> None:
    """Validate pacing configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = PacingConfig(
        ...     function=PacingFunction.LINEAR,
        ...     initial_difficulty=0.0,
        ...     target_difficulty=1.0,
        ...     warmup_steps=1000,
        ... )
        >>> validate_pacing_config(config)

        >>> bad_config = PacingConfig(
        ...     function=PacingFunction.LINEAR,
        ...     initial_difficulty=-0.1,
        ...     target_difficulty=1.0,
        ...     warmup_steps=1000,
        ... )
        >>> validate_pacing_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: initial_difficulty must be between 0 and 1, got -0.1
    """
    if not 0 <= config.initial_difficulty <= 1:
        msg = (
            f"initial_difficulty must be between 0 and 1, "
            f"got {config.initial_difficulty}"
        )
        raise ValueError(msg)
    if not 0 <= config.target_difficulty <= 1:
        msg = (
            f"target_difficulty must be between 0 and 1, got {config.target_difficulty}"
        )
        raise ValueError(msg)
    if config.warmup_steps < 0:
        msg = f"warmup_steps must be non-negative, got {config.warmup_steps}"
        raise ValueError(msg)


def validate_curriculum_config(config: CurriculumConfig) -> None:
    """Validate curriculum configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> diff_config = DifficultyConfig(
        ...     metric=DifficultyMetric.LENGTH,
        ...     normalize=True,
        ...     buckets=10,
        ...     ascending=True,
        ... )
        >>> pacing_config = PacingConfig(
        ...     function=PacingFunction.LINEAR,
        ...     initial_difficulty=0.0,
        ...     target_difficulty=1.0,
        ...     warmup_steps=1000,
        ... )
        >>> config = CurriculumConfig(
        ...     difficulty_config=diff_config,
        ...     pacing_config=pacing_config,
        ...     sampling_strategy=SamplingStrategy.THRESHOLD,
        ... )
        >>> validate_curriculum_config(config)

        >>> bad_diff_config = DifficultyConfig(
        ...     metric=DifficultyMetric.LENGTH,
        ...     normalize=True,
        ...     buckets=-1,
        ...     ascending=True,
        ... )
        >>> bad_config = CurriculumConfig(
        ...     difficulty_config=bad_diff_config,
        ...     pacing_config=pacing_config,
        ...     sampling_strategy=SamplingStrategy.THRESHOLD,
        ... )
        >>> validate_curriculum_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: buckets must be positive, got -1
    """
    validate_difficulty_config(config.difficulty_config)
    validate_pacing_config(config.pacing_config)


def validate_curriculum_stats(stats: CurriculumStats) -> None:
    """Validate curriculum statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any statistic value is invalid.

    Examples:
        >>> stats = CurriculumStats(
        ...     current_difficulty=0.5,
        ...     samples_seen=10000,
        ...     competence_score=0.8,
        ...     curriculum_progress=0.5,
        ... )
        >>> validate_curriculum_stats(stats)

        >>> bad_stats = CurriculumStats(
        ...     current_difficulty=1.5,
        ...     samples_seen=10000,
        ...     competence_score=0.8,
        ...     curriculum_progress=0.5,
        ... )
        >>> validate_curriculum_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: current_difficulty must be between 0 and 1, got 1.5
    """
    if not 0 <= stats.current_difficulty <= 1:
        msg = (
            f"current_difficulty must be between 0 and 1, "
            f"got {stats.current_difficulty}"
        )
        raise ValueError(msg)
    if stats.samples_seen < 0:
        msg = f"samples_seen must be non-negative, got {stats.samples_seen}"
        raise ValueError(msg)
    if not 0 <= stats.competence_score <= 1:
        msg = f"competence_score must be between 0 and 1, got {stats.competence_score}"
        raise ValueError(msg)
    if not 0 <= stats.curriculum_progress <= 1:
        msg = (
            f"curriculum_progress must be between 0 and 1, "
            f"got {stats.curriculum_progress}"
        )
        raise ValueError(msg)


def create_difficulty_config(
    metric: str | DifficultyMetric = DifficultyMetric.LENGTH,
    normalize: bool = True,
    buckets: int = 10,
    ascending: bool = True,
) -> DifficultyConfig:
    """Create a difficulty configuration with validation.

    Args:
        metric: Metric to use for measuring difficulty.
        normalize: Whether to normalize difficulty scores to [0, 1].
        buckets: Number of difficulty buckets for discretization.
        ascending: Whether higher values mean more difficult.

    Returns:
        Validated DifficultyConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_difficulty_config()
        >>> config.metric
        <DifficultyMetric.LENGTH: 'length'>
        >>> config.normalize
        True

        >>> config = create_difficulty_config(metric="perplexity", buckets=5)
        >>> config.metric
        <DifficultyMetric.PERPLEXITY: 'perplexity'>
        >>> config.buckets
        5

        >>> create_difficulty_config(buckets=0)
        Traceback (most recent call last):
            ...
        ValueError: buckets must be positive, got 0
    """
    if isinstance(metric, str):
        metric = get_difficulty_metric(metric)

    config = DifficultyConfig(
        metric=metric,
        normalize=normalize,
        buckets=buckets,
        ascending=ascending,
    )
    validate_difficulty_config(config)
    return config


def create_pacing_config(
    function: str | PacingFunction = PacingFunction.LINEAR,
    initial_difficulty: float = 0.0,
    target_difficulty: float = 1.0,
    warmup_steps: int = 0,
) -> PacingConfig:
    """Create a pacing configuration with validation.

    Args:
        function: Pacing function to use.
        initial_difficulty: Starting difficulty level (0 to 1).
        target_difficulty: Final difficulty level (0 to 1).
        warmup_steps: Number of warmup steps before curriculum begins.

    Returns:
        Validated PacingConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_pacing_config()
        >>> config.function
        <PacingFunction.LINEAR: 'linear'>
        >>> config.initial_difficulty
        0.0

        >>> config = create_pacing_config(
        ...     function="exponential",
        ...     initial_difficulty=0.2,
        ...     warmup_steps=500,
        ... )
        >>> config.function
        <PacingFunction.EXPONENTIAL: 'exponential'>
        >>> config.warmup_steps
        500

        >>> create_pacing_config(initial_difficulty=1.5)
        Traceback (most recent call last):
            ...
        ValueError: initial_difficulty must be between 0 and 1, got 1.5
    """
    if isinstance(function, str):
        function = get_pacing_function(function)

    config = PacingConfig(
        function=function,
        initial_difficulty=initial_difficulty,
        target_difficulty=target_difficulty,
        warmup_steps=warmup_steps,
    )
    validate_pacing_config(config)
    return config


def create_curriculum_config(
    metric: str | DifficultyMetric = DifficultyMetric.LENGTH,
    normalize: bool = True,
    buckets: int = 10,
    ascending: bool = True,
    function: str | PacingFunction = PacingFunction.LINEAR,
    initial_difficulty: float = 0.0,
    target_difficulty: float = 1.0,
    warmup_steps: int = 0,
    sampling_strategy: str | SamplingStrategy = SamplingStrategy.THRESHOLD,
) -> CurriculumConfig:
    """Create a curriculum configuration with validation.

    Args:
        metric: Metric to use for measuring difficulty.
        normalize: Whether to normalize difficulty scores to [0, 1].
        buckets: Number of difficulty buckets for discretization.
        ascending: Whether higher values mean more difficult.
        function: Pacing function to use.
        initial_difficulty: Starting difficulty level (0 to 1).
        target_difficulty: Final difficulty level (0 to 1).
        warmup_steps: Number of warmup steps before curriculum begins.
        sampling_strategy: Strategy for sampling curriculum samples.

    Returns:
        Validated CurriculumConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_curriculum_config()
        >>> config.difficulty_config.metric
        <DifficultyMetric.LENGTH: 'length'>
        >>> config.pacing_config.function
        <PacingFunction.LINEAR: 'linear'>
        >>> config.sampling_strategy
        <SamplingStrategy.THRESHOLD: 'threshold'>

        >>> config = create_curriculum_config(
        ...     metric="perplexity",
        ...     function="exponential",
        ...     sampling_strategy="weighted",
        ... )
        >>> config.difficulty_config.metric
        <DifficultyMetric.PERPLEXITY: 'perplexity'>
        >>> config.sampling_strategy
        <SamplingStrategy.WEIGHTED: 'weighted'>

        >>> create_curriculum_config(buckets=-1)
        Traceback (most recent call last):
            ...
        ValueError: buckets must be positive, got -1
    """
    difficulty_config = create_difficulty_config(
        metric=metric,
        normalize=normalize,
        buckets=buckets,
        ascending=ascending,
    )
    pacing_config = create_pacing_config(
        function=function,
        initial_difficulty=initial_difficulty,
        target_difficulty=target_difficulty,
        warmup_steps=warmup_steps,
    )

    if isinstance(sampling_strategy, str):
        sampling_strategy = get_sampling_strategy(sampling_strategy)

    config = CurriculumConfig(
        difficulty_config=difficulty_config,
        pacing_config=pacing_config,
        sampling_strategy=sampling_strategy,
    )
    validate_curriculum_config(config)
    return config


def create_curriculum_stats(
    current_difficulty: float = 0.0,
    samples_seen: int = 0,
    competence_score: float = 0.0,
    curriculum_progress: float = 0.0,
) -> CurriculumStats:
    """Create curriculum statistics with validation.

    Args:
        current_difficulty: Current difficulty threshold.
        samples_seen: Total number of samples seen.
        competence_score: Current model competence (0 to 1).
        curriculum_progress: Progress through curriculum (0 to 1).

    Returns:
        Validated CurriculumStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_curriculum_stats()
        >>> stats.current_difficulty
        0.0
        >>> stats.samples_seen
        0

        >>> stats = create_curriculum_stats(
        ...     current_difficulty=0.5,
        ...     samples_seen=10000,
        ...     competence_score=0.8,
        ... )
        >>> stats.competence_score
        0.8

        >>> create_curriculum_stats(current_difficulty=1.5)
        Traceback (most recent call last):
            ...
        ValueError: current_difficulty must be between 0 and 1, got 1.5
    """
    stats = CurriculumStats(
        current_difficulty=current_difficulty,
        samples_seen=samples_seen,
        competence_score=competence_score,
        curriculum_progress=curriculum_progress,
    )
    validate_curriculum_stats(stats)
    return stats


def list_difficulty_metrics() -> list[str]:
    """List all available difficulty metrics.

    Returns:
        Sorted list of difficulty metric names.

    Examples:
        >>> metrics = list_difficulty_metrics()
        >>> "length" in metrics
        True
        >>> "perplexity" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_DIFFICULTY_METRICS)


def list_pacing_functions() -> list[str]:
    """List all available pacing functions.

    Returns:
        Sorted list of pacing function names.

    Examples:
        >>> functions = list_pacing_functions()
        >>> "linear" in functions
        True
        >>> "exponential" in functions
        True
        >>> functions == sorted(functions)
        True
    """
    return sorted(VALID_PACING_FUNCTIONS)


def list_sampling_strategies() -> list[str]:
    """List all available sampling strategies.

    Returns:
        Sorted list of sampling strategy names.

    Examples:
        >>> strategies = list_sampling_strategies()
        >>> "threshold" in strategies
        True
        >>> "weighted" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SAMPLING_STRATEGIES)


def get_difficulty_metric(name: str) -> DifficultyMetric:
    """Get difficulty metric enum from string name.

    Args:
        name: Name of the difficulty metric.

    Returns:
        Corresponding DifficultyMetric enum.

    Raises:
        ValueError: If metric name is invalid.

    Examples:
        >>> get_difficulty_metric("length")
        <DifficultyMetric.LENGTH: 'length'>
        >>> get_difficulty_metric("perplexity")
        <DifficultyMetric.PERPLEXITY: 'perplexity'>

        >>> get_difficulty_metric("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: metric must be one of ...
    """
    if name not in VALID_DIFFICULTY_METRICS:
        msg = f"metric must be one of {VALID_DIFFICULTY_METRICS}, got '{name}'"
        raise ValueError(msg)
    return DifficultyMetric(name)


def get_pacing_function(name: str) -> PacingFunction:
    """Get pacing function enum from string name.

    Args:
        name: Name of the pacing function.

    Returns:
        Corresponding PacingFunction enum.

    Raises:
        ValueError: If function name is invalid.

    Examples:
        >>> get_pacing_function("linear")
        <PacingFunction.LINEAR: 'linear'>
        >>> get_pacing_function("exponential")
        <PacingFunction.EXPONENTIAL: 'exponential'>

        >>> get_pacing_function("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: pacing_function must be one of ...
    """
    if name not in VALID_PACING_FUNCTIONS:
        msg = f"pacing_function must be one of {VALID_PACING_FUNCTIONS}, got '{name}'"
        raise ValueError(msg)
    return PacingFunction(name)


def get_sampling_strategy(name: str) -> SamplingStrategy:
    """Get sampling strategy enum from string name.

    Args:
        name: Name of the sampling strategy.

    Returns:
        Corresponding SamplingStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_sampling_strategy("threshold")
        <SamplingStrategy.THRESHOLD: 'threshold'>
        >>> get_sampling_strategy("weighted")
        <SamplingStrategy.WEIGHTED: 'weighted'>

        >>> get_sampling_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: sampling_strategy must be one of ...
    """
    if name not in VALID_SAMPLING_STRATEGIES:
        msg = (
            f"sampling_strategy must be one of {VALID_SAMPLING_STRATEGIES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return SamplingStrategy(name)


def calculate_sample_difficulty(
    value: float,
    min_value: float,
    max_value: float,
    ascending: bool = True,
) -> float:
    """Calculate normalized difficulty score for a sample.

    Args:
        value: Raw difficulty value for the sample.
        min_value: Minimum value in the dataset.
        max_value: Maximum value in the dataset.
        ascending: Whether higher values mean more difficult.

    Returns:
        Normalized difficulty score between 0 and 1.

    Raises:
        ValueError: If value is outside valid range or min >= max.

    Examples:
        >>> calculate_sample_difficulty(50.0, 0.0, 100.0)
        0.5
        >>> calculate_sample_difficulty(0.0, 0.0, 100.0)
        0.0
        >>> calculate_sample_difficulty(100.0, 0.0, 100.0)
        1.0

        >>> calculate_sample_difficulty(50.0, 0.0, 100.0, ascending=False)
        0.5

        >>> calculate_sample_difficulty(50.0, 100.0, 100.0)
        Traceback (most recent call last):
            ...
        ValueError: min_value must be less than max_value
    """
    if min_value >= max_value:
        msg = "min_value must be less than max_value"
        raise ValueError(msg)

    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0.0, min(1.0, normalized))

    if not ascending:
        normalized = 1.0 - normalized

    return normalized


def calculate_competence_score(
    recent_losses: tuple[float, ...],
    target_loss: float = 0.1,
    smoothing: float = 0.9,
) -> float:
    """Calculate model competence based on recent losses.

    Competence is a measure of how well the model is performing,
    used for self-paced and competence-based curriculum learning.

    Args:
        recent_losses: Tuple of recent loss values.
        target_loss: Target loss for full competence.
        smoothing: Exponential smoothing factor.

    Returns:
        Competence score between 0 and 1.

    Raises:
        ValueError: If recent_losses is empty or parameters are invalid.

    Examples:
        >>> round(calculate_competence_score((0.5, 0.4, 0.3)), 4)
        0.3503
        >>> calculate_competence_score((0.1,), target_loss=0.1)
        1.0
        >>> round(calculate_competence_score((1.0, 0.8, 0.6, 0.4)), 4)
        0.2025

        >>> calculate_competence_score(())
        Traceback (most recent call last):
            ...
        ValueError: recent_losses cannot be empty
    """
    if not recent_losses:
        msg = "recent_losses cannot be empty"
        raise ValueError(msg)
    if target_loss <= 0:
        msg = f"target_loss must be positive, got {target_loss}"
        raise ValueError(msg)
    if not 0 < smoothing <= 1:
        msg = f"smoothing must be in (0, 1], got {smoothing}"
        raise ValueError(msg)

    smoothed_loss = recent_losses[0]
    for loss in recent_losses[1:]:
        smoothed_loss = smoothing * smoothed_loss + (1 - smoothing) * loss

    competence = target_loss / (target_loss + smoothed_loss)
    return min(1.0, competence * 2)


def get_difficulty_at_step(
    config: PacingConfig,
    current_step: int,
    total_steps: int,
    competence_score: float = 0.0,
) -> float:
    """Calculate difficulty threshold at a given training step.

    Args:
        config: Pacing configuration.
        current_step: Current training step.
        total_steps: Total number of training steps.
        competence_score: Current model competence (for self-paced/competence).

    Returns:
        Difficulty threshold at the current step.

    Raises:
        ValueError: If step parameters are invalid.

    Examples:
        >>> config = create_pacing_config(
        ...     function="linear",
        ...     initial_difficulty=0.0,
        ...     target_difficulty=1.0,
        ... )
        >>> get_difficulty_at_step(config, 500, 1000)
        0.5
        >>> get_difficulty_at_step(config, 0, 1000)
        0.0
        >>> get_difficulty_at_step(config, 1000, 1000)
        1.0

        >>> config = create_pacing_config(function="exponential")
        >>> 0.5 < get_difficulty_at_step(config, 500, 1000) < 1.0
        True

        >>> get_difficulty_at_step(config, -1, 1000)
        Traceback (most recent call last):
            ...
        ValueError: current_step must be non-negative, got -1
    """
    if current_step < 0:
        msg = f"current_step must be non-negative, got {current_step}"
        raise ValueError(msg)
    if total_steps <= 0:
        msg = f"total_steps must be positive, got {total_steps}"
        raise ValueError(msg)
    if current_step > total_steps:
        msg = f"current_step ({current_step}) cannot exceed total_steps ({total_steps})"
        raise ValueError(msg)

    warmup = config.warmup_steps
    initial = config.initial_difficulty
    target = config.target_difficulty
    func = config.function

    if current_step < warmup:
        return initial

    effective_step = current_step - warmup
    effective_total = total_steps - warmup

    if effective_total <= 0:
        return target

    progress = effective_step / effective_total

    if func == PacingFunction.LINEAR:
        return initial + (target - initial) * progress
    elif func == PacingFunction.EXPONENTIAL:
        return initial + (target - initial) * (1 - math.exp(-3 * progress))
    elif func == PacingFunction.STEP:
        num_steps = 5
        step_idx = min(int(progress * num_steps), num_steps - 1)
        return initial + (target - initial) * (step_idx + 1) / num_steps
    elif func == PacingFunction.SELF_PACED:
        return min(target, initial + (target - initial) * competence_score)
    elif func == PacingFunction.COMPETENCE:
        sqrt_progress = math.sqrt(progress)
        competence_factor = min(1.0, sqrt_progress * (1 + competence_score) / 2)
        return initial + (target - initial) * competence_factor

    return initial


def calculate_sample_weights(
    difficulties: tuple[float, ...],
    current_difficulty: float,
    strategy: SamplingStrategy,
    temperature: float = 1.0,
) -> tuple[float, ...]:
    """Calculate sampling weights for samples based on difficulty.

    Args:
        difficulties: Tuple of difficulty scores for each sample.
        current_difficulty: Current difficulty threshold.
        strategy: Sampling strategy to use.
        temperature: Temperature for probabilistic sampling.

    Returns:
        Tuple of sampling weights for each sample.

    Raises:
        ValueError: If difficulties is empty or parameters are invalid.

    Examples:
        >>> difficulties = (0.1, 0.3, 0.5, 0.7, 0.9)
        >>> weights = calculate_sample_weights(
        ...     difficulties, 0.5, SamplingStrategy.THRESHOLD
        ... )
        >>> weights
        (1.0, 1.0, 1.0, 0.0, 0.0)

        >>> weights = calculate_sample_weights(
        ...     difficulties, 0.5, SamplingStrategy.WEIGHTED
        ... )
        >>> all(w > 0 for w in weights)
        True
        >>> weights[0] > weights[-1]
        True

        >>> calculate_sample_weights((), 0.5, SamplingStrategy.THRESHOLD)
        Traceback (most recent call last):
            ...
        ValueError: difficulties cannot be empty
    """
    if not difficulties:
        msg = "difficulties cannot be empty"
        raise ValueError(msg)
    if not 0 <= current_difficulty <= 1:
        msg = f"current_difficulty must be between 0 and 1, got {current_difficulty}"
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    if strategy == SamplingStrategy.THRESHOLD:
        return tuple(1.0 if d <= current_difficulty else 0.0 for d in difficulties)

    elif strategy == SamplingStrategy.WEIGHTED:
        weights = []
        for d in difficulties:
            if d <= current_difficulty:
                weight = 1.0
            else:
                weight = max(0.0, 1.0 - (d - current_difficulty) / temperature)
            weights.append(weight)
        return tuple(weights)

    else:  # PROBABILISTIC
        weights = []
        for d in difficulties:
            prob = 1.0 / (1.0 + math.exp((d - current_difficulty) / temperature))
            weights.append(prob)
        return tuple(weights)


def format_curriculum_stats(stats: CurriculumStats) -> str:
    """Format curriculum statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_curriculum_stats(
        ...     current_difficulty=0.5,
        ...     samples_seen=10000,
        ...     competence_score=0.8,
        ...     curriculum_progress=0.5,
        ... )
        >>> formatted = format_curriculum_stats(stats)
        >>> "Difficulty: 50.0%" in formatted
        True
        >>> "Samples: 10000" in formatted
        True
        >>> "Competence: 80.0%" in formatted
        True
        >>> "Progress: 50.0%" in formatted
        True
    """
    return (
        f"Curriculum Stats:\n"
        f"  Difficulty: {stats.current_difficulty * 100:.1f}%\n"
        f"  Samples: {stats.samples_seen}\n"
        f"  Competence: {stats.competence_score * 100:.1f}%\n"
        f"  Progress: {stats.curriculum_progress * 100:.1f}%"
    )


def get_recommended_curriculum_config(task_type: str) -> CurriculumConfig:
    """Get recommended curriculum configuration for a task type.

    Args:
        task_type: Type of task (classification, generation, translation,
            summarization, qa).

    Returns:
        Recommended CurriculumConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_curriculum_config("classification")
        >>> config.difficulty_config.metric
        <DifficultyMetric.CONFIDENCE: 'confidence'>
        >>> config.pacing_config.function
        <PacingFunction.LINEAR: 'linear'>

        >>> config = get_recommended_curriculum_config("generation")
        >>> config.difficulty_config.metric
        <DifficultyMetric.LENGTH: 'length'>
        >>> config.pacing_config.function
        <PacingFunction.EXPONENTIAL: 'exponential'>

        >>> get_recommended_curriculum_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_type must be one of ...
    """
    valid_tasks = frozenset(
        {
            "classification",
            "generation",
            "translation",
            "summarization",
            "qa",
        }
    )

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    if task_type == "classification":
        return create_curriculum_config(
            metric=DifficultyMetric.CONFIDENCE,
            function=PacingFunction.LINEAR,
            initial_difficulty=0.2,
            target_difficulty=1.0,
            sampling_strategy=SamplingStrategy.THRESHOLD,
        )
    elif task_type == "generation":
        return create_curriculum_config(
            metric=DifficultyMetric.LENGTH,
            function=PacingFunction.EXPONENTIAL,
            initial_difficulty=0.1,
            target_difficulty=1.0,
            sampling_strategy=SamplingStrategy.WEIGHTED,
        )
    elif task_type == "translation":
        return create_curriculum_config(
            metric=DifficultyMetric.LENGTH,
            function=PacingFunction.COMPETENCE,
            initial_difficulty=0.1,
            target_difficulty=1.0,
            sampling_strategy=SamplingStrategy.PROBABILISTIC,
        )
    elif task_type == "summarization":
        return create_curriculum_config(
            metric=DifficultyMetric.PERPLEXITY,
            function=PacingFunction.SELF_PACED,
            initial_difficulty=0.2,
            target_difficulty=1.0,
            sampling_strategy=SamplingStrategy.WEIGHTED,
        )
    else:  # qa
        return create_curriculum_config(
            metric=DifficultyMetric.LOSS,
            function=PacingFunction.STEP,
            initial_difficulty=0.0,
            target_difficulty=1.0,
            sampling_strategy=SamplingStrategy.THRESHOLD,
        )
