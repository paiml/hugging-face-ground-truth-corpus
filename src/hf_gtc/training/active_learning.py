"""Active Learning utilities for training.

This module provides functions for configuring and implementing active
learning strategies, enabling efficient sample selection for labeling
based on uncertainty and diversity measures.

Examples:
    >>> from hf_gtc.training.active_learning import (
    ...     create_query_config,
    ...     QueryStrategy,
    ... )
    >>> config = create_query_config()
    >>> config.strategy
    <QueryStrategy.UNCERTAINTY: 'uncertainty'>
    >>> config.batch_size
    32
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class QueryStrategy(Enum):
    """Strategies for selecting samples to query for labels.

    Attributes:
        UNCERTAINTY: Select samples with highest model uncertainty.
        MARGIN: Select samples with smallest margin between top predictions.
        ENTROPY: Select samples with highest prediction entropy.
        BALD: Bayesian Active Learning by Disagreement.
        CORESET: Select samples that maximize coverage of feature space.
        BADGE: Batch Active learning by Diverse Gradient Embeddings.

    Examples:
        >>> QueryStrategy.UNCERTAINTY.value
        'uncertainty'
        >>> QueryStrategy.MARGIN.value
        'margin'
        >>> QueryStrategy.BALD.value
        'bald'
    """

    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    ENTROPY = "entropy"
    BALD = "bald"
    CORESET = "coreset"
    BADGE = "badge"


class UncertaintyMeasure(Enum):
    """Measures for quantifying model uncertainty.

    Attributes:
        PREDICTIVE_ENTROPY: Entropy of the predictive distribution.
        MUTUAL_INFORMATION: Mutual information between predictions and parameters.
        VARIATION_RATIO: Ratio of non-modal predictions.
        LEAST_CONFIDENCE: One minus the maximum predicted probability.

    Examples:
        >>> UncertaintyMeasure.PREDICTIVE_ENTROPY.value
        'predictive_entropy'
        >>> UncertaintyMeasure.MUTUAL_INFORMATION.value
        'mutual_information'
        >>> UncertaintyMeasure.LEAST_CONFIDENCE.value
        'least_confidence'
    """

    PREDICTIVE_ENTROPY = "predictive_entropy"
    MUTUAL_INFORMATION = "mutual_information"
    VARIATION_RATIO = "variation_ratio"
    LEAST_CONFIDENCE = "least_confidence"


class PoolingMethod(Enum):
    """Methods for pooling samples from the unlabeled pool.

    Attributes:
        RANDOM: Random sampling from the pool.
        CLUSTER: Cluster-based sampling for diversity.
        DIVERSITY: Diversity-based sampling using distances.

    Examples:
        >>> PoolingMethod.RANDOM.value
        'random'
        >>> PoolingMethod.CLUSTER.value
        'cluster'
        >>> PoolingMethod.DIVERSITY.value
        'diversity'
    """

    RANDOM = "random"
    CLUSTER = "cluster"
    DIVERSITY = "diversity"


VALID_QUERY_STRATEGIES = frozenset(s.value for s in QueryStrategy)
VALID_UNCERTAINTY_MEASURES = frozenset(m.value for m in UncertaintyMeasure)
VALID_POOLING_METHODS = frozenset(p.value for p in PoolingMethod)


@dataclass(frozen=True, slots=True)
class QueryConfig:
    """Configuration for sample query selection.

    Attributes:
        strategy: Strategy for selecting samples to query.
        batch_size: Number of samples to select per query round.
        uncertainty_measure: Measure for quantifying model uncertainty.
        diversity_weight: Weight for diversity in hybrid strategies (0 to 1).

    Examples:
        >>> config = QueryConfig(
        ...     strategy=QueryStrategy.UNCERTAINTY,
        ...     batch_size=32,
        ...     uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
        ...     diversity_weight=0.5,
        ... )
        >>> config.strategy
        <QueryStrategy.UNCERTAINTY: 'uncertainty'>
        >>> config.batch_size
        32
        >>> config.diversity_weight
        0.5
    """

    strategy: QueryStrategy
    batch_size: int
    uncertainty_measure: UncertaintyMeasure
    diversity_weight: float


@dataclass(frozen=True, slots=True)
class ActiveLearningConfig:
    """Main configuration for active learning.

    Attributes:
        query_config: Configuration for sample query selection.
        initial_pool_size: Size of initial labeled pool.
        labeling_budget: Total labeling budget (max samples to label).
        stopping_criterion: Minimum accuracy improvement to continue.

    Examples:
        >>> query_config = QueryConfig(
        ...     strategy=QueryStrategy.UNCERTAINTY,
        ...     batch_size=32,
        ...     uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
        ...     diversity_weight=0.5,
        ... )
        >>> config = ActiveLearningConfig(
        ...     query_config=query_config,
        ...     initial_pool_size=100,
        ...     labeling_budget=1000,
        ...     stopping_criterion=0.001,
        ... )
        >>> config.initial_pool_size
        100
        >>> config.labeling_budget
        1000
    """

    query_config: QueryConfig
    initial_pool_size: int
    labeling_budget: int
    stopping_criterion: float


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Result of a sample query operation.

    Attributes:
        indices: Tuple of indices of selected samples.
        scores: Tuple of selection scores for each sample.
        uncertainty_values: Tuple of uncertainty values for each sample.

    Examples:
        >>> result = QueryResult(
        ...     indices=(5, 12, 23, 45),
        ...     scores=(0.95, 0.92, 0.88, 0.85),
        ...     uncertainty_values=(0.9, 0.85, 0.8, 0.75),
        ... )
        >>> result.indices
        (5, 12, 23, 45)
        >>> result.scores
        (0.95, 0.92, 0.88, 0.85)
    """

    indices: tuple[int, ...]
    scores: tuple[float, ...]
    uncertainty_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ActiveLearningStats:
    """Statistics from active learning training.

    Attributes:
        total_labeled: Total number of samples labeled.
        accuracy_history: Tuple of accuracy values over rounds.
        query_efficiency: Efficiency of queries (accuracy gain per sample).
        labeling_cost: Estimated total labeling cost.

    Examples:
        >>> stats = ActiveLearningStats(
        ...     total_labeled=500,
        ...     accuracy_history=(0.65, 0.72, 0.78, 0.82),
        ...     query_efficiency=0.015,
        ...     labeling_cost=250.0,
        ... )
        >>> stats.total_labeled
        500
        >>> stats.query_efficiency
        0.015
    """

    total_labeled: int
    accuracy_history: tuple[float, ...]
    query_efficiency: float
    labeling_cost: float


def validate_query_config(config: QueryConfig) -> None:
    """Validate query configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = QueryConfig(
        ...     strategy=QueryStrategy.UNCERTAINTY,
        ...     batch_size=32,
        ...     uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
        ...     diversity_weight=0.5,
        ... )
        >>> validate_query_config(config)

        >>> bad_config = QueryConfig(
        ...     strategy=QueryStrategy.UNCERTAINTY,
        ...     batch_size=0,
        ...     uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
        ...     diversity_weight=0.5,
        ... )
        >>> validate_query_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive, got 0
    """
    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)
    if not 0 <= config.diversity_weight <= 1:
        msg = f"diversity_weight must be between 0 and 1, got {config.diversity_weight}"
        raise ValueError(msg)


def validate_active_learning_config(config: ActiveLearningConfig) -> None:
    """Validate active learning configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> query_config = QueryConfig(
        ...     strategy=QueryStrategy.UNCERTAINTY,
        ...     batch_size=32,
        ...     uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
        ...     diversity_weight=0.5,
        ... )
        >>> config = ActiveLearningConfig(
        ...     query_config=query_config,
        ...     initial_pool_size=100,
        ...     labeling_budget=1000,
        ...     stopping_criterion=0.001,
        ... )
        >>> validate_active_learning_config(config)

        >>> bad_config = ActiveLearningConfig(
        ...     query_config=query_config,
        ...     initial_pool_size=-1,
        ...     labeling_budget=1000,
        ...     stopping_criterion=0.001,
        ... )
        >>> validate_active_learning_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: initial_pool_size must be non-negative, got -1
    """
    validate_query_config(config.query_config)
    if config.initial_pool_size < 0:
        msg = f"initial_pool_size must be non-negative, got {config.initial_pool_size}"
        raise ValueError(msg)
    if config.labeling_budget <= 0:
        msg = f"labeling_budget must be positive, got {config.labeling_budget}"
        raise ValueError(msg)
    if config.stopping_criterion < 0:
        msg = (
            f"stopping_criterion must be non-negative, got {config.stopping_criterion}"
        )
        raise ValueError(msg)


def validate_query_result(result: QueryResult) -> None:
    """Validate query result.

    Args:
        result: Result to validate.

    Raises:
        ValueError: If any result value is invalid.

    Examples:
        >>> result = QueryResult(
        ...     indices=(5, 12, 23),
        ...     scores=(0.95, 0.92, 0.88),
        ...     uncertainty_values=(0.9, 0.85, 0.8),
        ... )
        >>> validate_query_result(result)

        >>> bad_result = QueryResult(
        ...     indices=(5, 12),
        ...     scores=(0.95, 0.92, 0.88),
        ...     uncertainty_values=(0.9, 0.85, 0.8),
        ... )
        >>> validate_query_result(bad_result)
        Traceback (most recent call last):
            ...
        ValueError: indices, scores, and uncertainty_values must have same length
    """
    indices_len = len(result.indices)
    scores_len = len(result.scores)
    uncertainty_len = len(result.uncertainty_values)
    if not (indices_len == scores_len == uncertainty_len):
        msg = "indices, scores, and uncertainty_values must have same length"
        raise ValueError(msg)
    if any(i < 0 for i in result.indices):
        msg = "all indices must be non-negative"
        raise ValueError(msg)


def validate_active_learning_stats(stats: ActiveLearningStats) -> None:
    """Validate active learning statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any statistic value is invalid.

    Examples:
        >>> stats = ActiveLearningStats(
        ...     total_labeled=500,
        ...     accuracy_history=(0.65, 0.72, 0.78),
        ...     query_efficiency=0.015,
        ...     labeling_cost=250.0,
        ... )
        >>> validate_active_learning_stats(stats)

        >>> bad_stats = ActiveLearningStats(
        ...     total_labeled=-1,
        ...     accuracy_history=(0.65, 0.72),
        ...     query_efficiency=0.015,
        ...     labeling_cost=250.0,
        ... )
        >>> validate_active_learning_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: total_labeled must be non-negative, got -1
    """
    if stats.total_labeled < 0:
        msg = f"total_labeled must be non-negative, got {stats.total_labeled}"
        raise ValueError(msg)
    if stats.labeling_cost < 0:
        msg = f"labeling_cost must be non-negative, got {stats.labeling_cost}"
        raise ValueError(msg)
    for acc in stats.accuracy_history:
        if not 0 <= acc <= 1:
            msg = f"all accuracy values must be between 0 and 1, got {acc}"
            raise ValueError(msg)


def create_query_config(
    strategy: str | QueryStrategy = QueryStrategy.UNCERTAINTY,
    batch_size: int = 32,
    uncertainty_measure: str | UncertaintyMeasure = (
        UncertaintyMeasure.PREDICTIVE_ENTROPY
    ),
    diversity_weight: float = 0.5,
) -> QueryConfig:
    """Create a query configuration with validation.

    Args:
        strategy: Strategy for selecting samples to query.
        batch_size: Number of samples to select per query round.
        uncertainty_measure: Measure for quantifying model uncertainty.
        diversity_weight: Weight for diversity in hybrid strategies (0 to 1).

    Returns:
        Validated QueryConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_query_config()
        >>> config.strategy
        <QueryStrategy.UNCERTAINTY: 'uncertainty'>
        >>> config.batch_size
        32

        >>> config = create_query_config(strategy="margin", batch_size=64)
        >>> config.strategy
        <QueryStrategy.MARGIN: 'margin'>
        >>> config.batch_size
        64

        >>> create_query_config(batch_size=0)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive, got 0
    """
    if isinstance(strategy, str):
        strategy = get_query_strategy(strategy)
    if isinstance(uncertainty_measure, str):
        uncertainty_measure = get_uncertainty_measure(uncertainty_measure)

    config = QueryConfig(
        strategy=strategy,
        batch_size=batch_size,
        uncertainty_measure=uncertainty_measure,
        diversity_weight=diversity_weight,
    )
    validate_query_config(config)
    return config


def create_active_learning_config(
    strategy: str | QueryStrategy = QueryStrategy.UNCERTAINTY,
    batch_size: int = 32,
    uncertainty_measure: str | UncertaintyMeasure = (
        UncertaintyMeasure.PREDICTIVE_ENTROPY
    ),
    diversity_weight: float = 0.5,
    initial_pool_size: int = 100,
    labeling_budget: int = 1000,
    stopping_criterion: float = 0.001,
) -> ActiveLearningConfig:
    """Create an active learning configuration with validation.

    Args:
        strategy: Strategy for selecting samples to query.
        batch_size: Number of samples to select per query round.
        uncertainty_measure: Measure for quantifying model uncertainty.
        diversity_weight: Weight for diversity in hybrid strategies (0 to 1).
        initial_pool_size: Size of initial labeled pool.
        labeling_budget: Total labeling budget (max samples to label).
        stopping_criterion: Minimum accuracy improvement to continue.

    Returns:
        Validated ActiveLearningConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_active_learning_config()
        >>> config.query_config.strategy
        <QueryStrategy.UNCERTAINTY: 'uncertainty'>
        >>> config.initial_pool_size
        100
        >>> config.labeling_budget
        1000

        >>> config = create_active_learning_config(
        ...     strategy="bald",
        ...     initial_pool_size=50,
        ...     labeling_budget=500,
        ... )
        >>> config.query_config.strategy
        <QueryStrategy.BALD: 'bald'>
        >>> config.initial_pool_size
        50

        >>> create_active_learning_config(labeling_budget=0)
        Traceback (most recent call last):
            ...
        ValueError: labeling_budget must be positive, got 0
    """
    query_config = create_query_config(
        strategy=strategy,
        batch_size=batch_size,
        uncertainty_measure=uncertainty_measure,
        diversity_weight=diversity_weight,
    )

    config = ActiveLearningConfig(
        query_config=query_config,
        initial_pool_size=initial_pool_size,
        labeling_budget=labeling_budget,
        stopping_criterion=stopping_criterion,
    )
    validate_active_learning_config(config)
    return config


def create_query_result(
    indices: tuple[int, ...] = (),
    scores: tuple[float, ...] = (),
    uncertainty_values: tuple[float, ...] = (),
) -> QueryResult:
    """Create a query result with validation.

    Args:
        indices: Tuple of indices of selected samples.
        scores: Tuple of selection scores for each sample.
        uncertainty_values: Tuple of uncertainty values for each sample.

    Returns:
        Validated QueryResult.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> result = create_query_result()
        >>> result.indices
        ()

        >>> result = create_query_result(
        ...     indices=(5, 12, 23),
        ...     scores=(0.95, 0.92, 0.88),
        ...     uncertainty_values=(0.9, 0.85, 0.8),
        ... )
        >>> result.indices
        (5, 12, 23)

        >>> create_query_result(
        ...     indices=(5, 12),
        ...     scores=(0.95,),
        ...     uncertainty_values=(0.9, 0.85),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: indices, scores, and uncertainty_values must have same length
    """
    result = QueryResult(
        indices=indices,
        scores=scores,
        uncertainty_values=uncertainty_values,
    )
    validate_query_result(result)
    return result


def create_active_learning_stats(
    total_labeled: int = 0,
    accuracy_history: tuple[float, ...] = (),
    query_efficiency: float = 0.0,
    labeling_cost: float = 0.0,
) -> ActiveLearningStats:
    """Create active learning statistics with validation.

    Args:
        total_labeled: Total number of samples labeled.
        accuracy_history: Tuple of accuracy values over rounds.
        query_efficiency: Efficiency of queries (accuracy gain per sample).
        labeling_cost: Estimated total labeling cost.

    Returns:
        Validated ActiveLearningStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_active_learning_stats()
        >>> stats.total_labeled
        0

        >>> stats = create_active_learning_stats(
        ...     total_labeled=500,
        ...     accuracy_history=(0.65, 0.72, 0.78),
        ...     query_efficiency=0.015,
        ...     labeling_cost=250.0,
        ... )
        >>> stats.total_labeled
        500

        >>> create_active_learning_stats(total_labeled=-1)
        Traceback (most recent call last):
            ...
        ValueError: total_labeled must be non-negative, got -1
    """
    stats = ActiveLearningStats(
        total_labeled=total_labeled,
        accuracy_history=accuracy_history,
        query_efficiency=query_efficiency,
        labeling_cost=labeling_cost,
    )
    validate_active_learning_stats(stats)
    return stats


def list_query_strategies() -> list[str]:
    """List all available query strategies.

    Returns:
        Sorted list of query strategy names.

    Examples:
        >>> strategies = list_query_strategies()
        >>> "uncertainty" in strategies
        True
        >>> "bald" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_QUERY_STRATEGIES)


def list_uncertainty_measures() -> list[str]:
    """List all available uncertainty measures.

    Returns:
        Sorted list of uncertainty measure names.

    Examples:
        >>> measures = list_uncertainty_measures()
        >>> "predictive_entropy" in measures
        True
        >>> "mutual_information" in measures
        True
        >>> measures == sorted(measures)
        True
    """
    return sorted(VALID_UNCERTAINTY_MEASURES)


def list_pooling_methods() -> list[str]:
    """List all available pooling methods.

    Returns:
        Sorted list of pooling method names.

    Examples:
        >>> methods = list_pooling_methods()
        >>> "random" in methods
        True
        >>> "cluster" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_POOLING_METHODS)


def get_query_strategy(name: str) -> QueryStrategy:
    """Get query strategy enum from string name.

    Args:
        name: Name of the query strategy.

    Returns:
        Corresponding QueryStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_query_strategy("uncertainty")
        <QueryStrategy.UNCERTAINTY: 'uncertainty'>
        >>> get_query_strategy("bald")
        <QueryStrategy.BALD: 'bald'>

        >>> get_query_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: strategy must be one of ...
    """
    if name not in VALID_QUERY_STRATEGIES:
        msg = f"strategy must be one of {VALID_QUERY_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return QueryStrategy(name)


def get_uncertainty_measure(name: str) -> UncertaintyMeasure:
    """Get uncertainty measure enum from string name.

    Args:
        name: Name of the uncertainty measure.

    Returns:
        Corresponding UncertaintyMeasure enum.

    Raises:
        ValueError: If measure name is invalid.

    Examples:
        >>> get_uncertainty_measure("predictive_entropy")
        <UncertaintyMeasure.PREDICTIVE_ENTROPY: 'predictive_entropy'>
        >>> get_uncertainty_measure("mutual_information")
        <UncertaintyMeasure.MUTUAL_INFORMATION: 'mutual_information'>

        >>> get_uncertainty_measure("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: uncertainty_measure must be one of ...
    """
    if name not in VALID_UNCERTAINTY_MEASURES:
        msg = (
            f"uncertainty_measure must be one of {VALID_UNCERTAINTY_MEASURES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return UncertaintyMeasure(name)


def get_pooling_method(name: str) -> PoolingMethod:
    """Get pooling method enum from string name.

    Args:
        name: Name of the pooling method.

    Returns:
        Corresponding PoolingMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_pooling_method("random")
        <PoolingMethod.RANDOM: 'random'>
        >>> get_pooling_method("cluster")
        <PoolingMethod.CLUSTER: 'cluster'>

        >>> get_pooling_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: pooling_method must be one of ...
    """
    if name not in VALID_POOLING_METHODS:
        msg = f"pooling_method must be one of {VALID_POOLING_METHODS}, got '{name}'"
        raise ValueError(msg)
    return PoolingMethod(name)


def calculate_uncertainty(
    probabilities: tuple[float, ...],
    measure: UncertaintyMeasure = UncertaintyMeasure.PREDICTIVE_ENTROPY,
    mc_samples: tuple[tuple[float, ...], ...] | None = None,
) -> float:
    """Calculate uncertainty for a single sample.

    Args:
        probabilities: Tuple of predicted class probabilities.
        measure: Uncertainty measure to use.
        mc_samples: Monte Carlo samples for Bayesian measures (optional).

    Returns:
        Uncertainty value (higher means more uncertain).

    Raises:
        ValueError: If probabilities is empty or invalid.

    Examples:
        >>> probs = (0.5, 0.5)
        >>> round(calculate_uncertainty(probs), 4)
        0.6931

        >>> probs = (0.9, 0.1)
        >>> round(calculate_uncertainty(probs), 4)
        0.3251

        >>> probs = (1.0, 0.0)
        >>> calculate_uncertainty(probs)
        0.0

        >>> probs = (0.5, 0.5)
        >>> calculate_uncertainty(probs, UncertaintyMeasure.LEAST_CONFIDENCE)
        0.5

        >>> calculate_uncertainty(())
        Traceback (most recent call last):
            ...
        ValueError: probabilities cannot be empty
    """
    if not probabilities:
        msg = "probabilities cannot be empty"
        raise ValueError(msg)

    # Normalize probabilities
    total = sum(probabilities)
    if total <= 0:
        msg = "sum of probabilities must be positive"
        raise ValueError(msg)
    probs = tuple(p / total for p in probabilities)

    if measure == UncertaintyMeasure.PREDICTIVE_ENTROPY:
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    elif measure in (
        UncertaintyMeasure.LEAST_CONFIDENCE,
        UncertaintyMeasure.VARIATION_RATIO,
    ):
        return 1.0 - max(probs)

    elif measure == UncertaintyMeasure.MUTUAL_INFORMATION:
        # For MI, we need MC samples; without them, fall back to entropy
        if mc_samples is None:
            entropy = 0.0
            for p in probs:
                if p > 0:
                    entropy -= p * math.log(p)
            return entropy

        # Calculate entropy of mean predictions
        mean_probs = [0.0] * len(probs)
        for sample in mc_samples:
            for i, p in enumerate(sample):
                mean_probs[i] += p
        mean_probs = [p / len(mc_samples) for p in mean_probs]

        mean_entropy = 0.0
        for p in mean_probs:
            if p > 0:
                mean_entropy -= p * math.log(p)

        # Calculate mean of entropies
        sample_entropies = []
        for sample in mc_samples:
            sample_entropy = 0.0
            for p in sample:
                if p > 0:
                    sample_entropy -= p * math.log(p)
            sample_entropies.append(sample_entropy)
        mean_sample_entropy = sum(sample_entropies) / len(mc_samples)

        return mean_entropy - mean_sample_entropy

    return 0.0


def select_samples(
    uncertainties: tuple[float, ...],
    batch_size: int,
    strategy: QueryStrategy = QueryStrategy.UNCERTAINTY,
    diversity_scores: tuple[float, ...] | None = None,
    diversity_weight: float = 0.5,
) -> tuple[int, ...]:
    """Select samples for labeling based on uncertainty and diversity.

    Args:
        uncertainties: Tuple of uncertainty values for each sample.
        batch_size: Number of samples to select.
        strategy: Query strategy to use.
        diversity_scores: Optional diversity scores for each sample.
        diversity_weight: Weight for diversity in combined score.

    Returns:
        Tuple of indices of selected samples.

    Raises:
        ValueError: If uncertainties is empty or batch_size is invalid.

    Examples:
        >>> uncertainties = (0.9, 0.5, 0.8, 0.3, 0.7)
        >>> select_samples(uncertainties, 3)
        (0, 2, 4)

        >>> select_samples(uncertainties, 2, QueryStrategy.MARGIN)
        (0, 2)

        >>> select_samples((), 3)
        Traceback (most recent call last):
            ...
        ValueError: uncertainties cannot be empty

        >>> select_samples((0.9, 0.5), 0)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive, got 0
    """
    if not uncertainties:
        msg = "uncertainties cannot be empty"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    batch_size = min(batch_size, len(uncertainties))

    # Compute combined scores
    if diversity_scores is not None and len(diversity_scores) == len(uncertainties):
        scores = tuple(
            (1 - diversity_weight) * u + diversity_weight * d
            for u, d in zip(uncertainties, diversity_scores, strict=False)
        )
    else:
        scores = uncertainties

    # Select top-k by score
    indexed_scores = [(i, s) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    return tuple(idx for idx, _ in indexed_scores[:batch_size])


def estimate_labeling_cost(
    num_samples: int,
    cost_per_sample: float = 0.5,
    complexity_factor: float = 1.0,
) -> float:
    """Estimate the cost of labeling a number of samples.

    Args:
        num_samples: Number of samples to label.
        cost_per_sample: Base cost per sample in dollars.
        complexity_factor: Multiplier for task complexity (1.0 = average).

    Returns:
        Estimated total cost in dollars.

    Raises:
        ValueError: If num_samples is negative or costs are invalid.

    Examples:
        >>> estimate_labeling_cost(100)
        50.0
        >>> estimate_labeling_cost(100, cost_per_sample=1.0)
        100.0
        >>> estimate_labeling_cost(100, cost_per_sample=0.5, complexity_factor=2.0)
        100.0

        >>> estimate_labeling_cost(-1)
        Traceback (most recent call last):
            ...
        ValueError: num_samples must be non-negative, got -1
    """
    if num_samples < 0:
        msg = f"num_samples must be non-negative, got {num_samples}"
        raise ValueError(msg)
    if cost_per_sample < 0:
        msg = f"cost_per_sample must be non-negative, got {cost_per_sample}"
        raise ValueError(msg)
    if complexity_factor < 0:
        msg = f"complexity_factor must be non-negative, got {complexity_factor}"
        raise ValueError(msg)

    return num_samples * cost_per_sample * complexity_factor


def calculate_query_efficiency(
    accuracy_history: tuple[float, ...],
    samples_per_round: tuple[int, ...],
) -> float:
    """Calculate query efficiency as accuracy gain per labeled sample.

    Args:
        accuracy_history: Tuple of accuracy values over rounds.
        samples_per_round: Tuple of samples labeled per round.

    Returns:
        Query efficiency (accuracy gain per sample).

    Raises:
        ValueError: If inputs are invalid or mismatched.

    Examples:
        >>> accuracy_history = (0.5, 0.6, 0.7, 0.75)
        >>> samples_per_round = (100, 50, 50, 50)
        >>> round(calculate_query_efficiency(accuracy_history, samples_per_round), 4)
        0.0017

        >>> calculate_query_efficiency((), ())
        Traceback (most recent call last):
            ...
        ValueError: accuracy_history cannot be empty

        >>> calculate_query_efficiency((0.5, 0.6), (100,))
        Traceback (most recent call last):
            ...
        ValueError: accuracy_history and samples_per_round must have same length
    """
    if not accuracy_history:
        msg = "accuracy_history cannot be empty"
        raise ValueError(msg)
    if len(accuracy_history) != len(samples_per_round):
        msg = "accuracy_history and samples_per_round must have same length"
        raise ValueError(msg)

    if len(accuracy_history) < 2:
        return 0.0

    total_accuracy_gain = accuracy_history[-1] - accuracy_history[0]
    total_samples = sum(samples_per_round[1:])  # Exclude initial pool

    if total_samples <= 0:
        return 0.0

    return total_accuracy_gain / total_samples


def format_active_learning_stats(stats: ActiveLearningStats) -> str:
    """Format active learning statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_active_learning_stats(
        ...     total_labeled=500,
        ...     accuracy_history=(0.65, 0.72, 0.78),
        ...     query_efficiency=0.015,
        ...     labeling_cost=250.0,
        ... )
        >>> formatted = format_active_learning_stats(stats)
        >>> "Total Labeled: 500" in formatted
        True
        >>> "Efficiency: 0.0150" in formatted
        True
        >>> "Cost: $250.00" in formatted
        True
    """
    acc_str = ", ".join(f"{a:.2%}" for a in stats.accuracy_history)
    if not acc_str:
        acc_str = "N/A"

    return (
        f"Active Learning Stats:\n"
        f"  Total Labeled: {stats.total_labeled}\n"
        f"  Accuracy History: [{acc_str}]\n"
        f"  Efficiency: {stats.query_efficiency:.4f}\n"
        f"  Cost: ${stats.labeling_cost:.2f}"
    )


def get_recommended_active_learning_config(
    task_type: str,
    data_size: int = 10000,
) -> ActiveLearningConfig:
    """Get recommended active learning configuration for a task type.

    Args:
        task_type: Type of task (classification, ner, sentiment, qa, generation).
        data_size: Size of the unlabeled data pool.

    Returns:
        Recommended ActiveLearningConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_active_learning_config("classification")
        >>> config.query_config.strategy
        <QueryStrategy.UNCERTAINTY: 'uncertainty'>
        >>> config.query_config.uncertainty_measure
        <UncertaintyMeasure.PREDICTIVE_ENTROPY: 'predictive_entropy'>

        >>> config = get_recommended_active_learning_config("ner")
        >>> config.query_config.strategy
        <QueryStrategy.BALD: 'bald'>

        >>> get_recommended_active_learning_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_type must be one of ...
    """
    valid_tasks = frozenset(
        {
            "classification",
            "ner",
            "sentiment",
            "qa",
            "generation",
        }
    )

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    # Scale batch size and budget based on data size
    batch_size = max(16, min(64, data_size // 100))
    labeling_budget = max(100, data_size // 10)
    initial_pool = max(50, data_size // 100)

    if task_type == "classification":
        return create_active_learning_config(
            strategy=QueryStrategy.UNCERTAINTY,
            batch_size=batch_size,
            uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
            diversity_weight=0.3,
            initial_pool_size=initial_pool,
            labeling_budget=labeling_budget,
            stopping_criterion=0.001,
        )
    elif task_type == "ner":
        return create_active_learning_config(
            strategy=QueryStrategy.BALD,
            batch_size=batch_size,
            uncertainty_measure=UncertaintyMeasure.MUTUAL_INFORMATION,
            diversity_weight=0.5,
            initial_pool_size=initial_pool,
            labeling_budget=labeling_budget,
            stopping_criterion=0.002,
        )
    elif task_type == "sentiment":
        return create_active_learning_config(
            strategy=QueryStrategy.MARGIN,
            batch_size=batch_size,
            uncertainty_measure=UncertaintyMeasure.LEAST_CONFIDENCE,
            diversity_weight=0.4,
            initial_pool_size=initial_pool,
            labeling_budget=labeling_budget,
            stopping_criterion=0.001,
        )
    elif task_type == "qa":
        return create_active_learning_config(
            strategy=QueryStrategy.BADGE,
            batch_size=batch_size,
            uncertainty_measure=UncertaintyMeasure.VARIATION_RATIO,
            diversity_weight=0.6,
            initial_pool_size=initial_pool,
            labeling_budget=labeling_budget,
            stopping_criterion=0.002,
        )
    else:  # generation
        return create_active_learning_config(
            strategy=QueryStrategy.CORESET,
            batch_size=batch_size,
            uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
            diversity_weight=0.7,
            initial_pool_size=initial_pool,
            labeling_budget=labeling_budget,
            stopping_criterion=0.001,
        )
