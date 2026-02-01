"""Neural Architecture Search utilities for training.

This module provides functions for configuring and implementing neural
architecture search (NAS) strategies, enabling automated discovery of optimal
model architectures through various search methods including random, grid,
evolutionary, reinforcement learning, and differentiable approaches.

Examples:
    >>> from hf_gtc.training.nas import (
    ...     create_search_config,
    ...     SearchStrategy,
    ... )
    >>> config = create_search_config()
    >>> config.strategy
    <SearchStrategy.RANDOM: 'random'>
    >>> config.num_trials
    100
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SearchStrategy(Enum):
    """Search strategies for neural architecture search.

    Attributes:
        RANDOM: Random sampling from search space.
        GRID: Exhaustive grid search over search space.
        EVOLUTIONARY: Evolutionary algorithms (mutation, crossover).
        REINFORCEMENT: Reinforcement learning-based search.
        DIFFERENTIABLE: Differentiable NAS (DARTS-style).

    Examples:
        >>> SearchStrategy.RANDOM.value
        'random'
        >>> SearchStrategy.EVOLUTIONARY.value
        'evolutionary'
        >>> SearchStrategy.DIFFERENTIABLE.value
        'differentiable'
    """

    RANDOM = "random"
    GRID = "grid"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    DIFFERENTIABLE = "differentiable"


class SearchSpace(Enum):
    """Types of search spaces for architecture search.

    Attributes:
        MICRO: Micro search space (cell-level operations).
        MACRO: Macro search space (network-level topology).
        HIERARCHICAL: Hierarchical search (both micro and macro).

    Examples:
        >>> SearchSpace.MICRO.value
        'micro'
        >>> SearchSpace.MACRO.value
        'macro'
        >>> SearchSpace.HIERARCHICAL.value
        'hierarchical'
    """

    MICRO = "micro"
    MACRO = "macro"
    HIERARCHICAL = "hierarchical"


class PerformancePredictor(Enum):
    """Performance prediction methods for architecture evaluation.

    Attributes:
        SURROGATE: Surrogate model-based prediction (Gaussian process, etc.).
        ZERO_COST: Zero-cost proxy metrics (gradients, synflow, etc.).
        WEIGHT_SHARING: Weight sharing across architectures (supernet).

    Examples:
        >>> PerformancePredictor.SURROGATE.value
        'surrogate'
        >>> PerformancePredictor.ZERO_COST.value
        'zero_cost'
        >>> PerformancePredictor.WEIGHT_SHARING.value
        'weight_sharing'
    """

    SURROGATE = "surrogate"
    ZERO_COST = "zero_cost"
    WEIGHT_SHARING = "weight_sharing"


VALID_SEARCH_STRATEGIES = frozenset(s.value for s in SearchStrategy)
VALID_SEARCH_SPACES = frozenset(s.value for s in SearchSpace)
VALID_PERFORMANCE_PREDICTORS = frozenset(p.value for p in PerformancePredictor)


@dataclass(frozen=True, slots=True)
class SearchSpaceConfig:
    """Configuration for the architecture search space.

    Attributes:
        space_type: Type of search space (micro, macro, hierarchical).
        num_layers_range: Tuple of (min, max) number of layers to search.
        hidden_dims_range: Tuple of (min, max) hidden dimensions to search.
        num_heads_range: Tuple of (min, max) attention heads to search.

    Examples:
        >>> config = SearchSpaceConfig(
        ...     space_type=SearchSpace.MICRO,
        ...     num_layers_range=(4, 12),
        ...     hidden_dims_range=(256, 1024),
        ...     num_heads_range=(4, 16),
        ... )
        >>> config.space_type
        <SearchSpace.MICRO: 'micro'>
        >>> config.num_layers_range
        (4, 12)
        >>> config.hidden_dims_range
        (256, 1024)
    """

    space_type: SearchSpace
    num_layers_range: tuple[int, int]
    hidden_dims_range: tuple[int, int]
    num_heads_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Configuration for neural architecture search.

    Attributes:
        strategy: Search strategy to use.
        search_space: Configuration of the architecture search space.
        num_trials: Number of architecture trials to evaluate.
        early_stopping_patience: Patience for early stopping (0 to disable).

    Examples:
        >>> space_config = SearchSpaceConfig(
        ...     space_type=SearchSpace.MICRO,
        ...     num_layers_range=(4, 12),
        ...     hidden_dims_range=(256, 1024),
        ...     num_heads_range=(4, 16),
        ... )
        >>> config = SearchConfig(
        ...     strategy=SearchStrategy.RANDOM,
        ...     search_space=space_config,
        ...     num_trials=100,
        ...     early_stopping_patience=10,
        ... )
        >>> config.strategy
        <SearchStrategy.RANDOM: 'random'>
        >>> config.num_trials
        100
    """

    strategy: SearchStrategy
    search_space: SearchSpaceConfig
    num_trials: int
    early_stopping_patience: int


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    """A candidate architecture with evaluation results.

    Attributes:
        config: Dictionary of architecture hyperparameters.
        performance: Performance metric value (e.g., accuracy, F1).
        cost: Computational cost (e.g., FLOPs, parameters, latency).
        rank: Rank among all evaluated candidates (1 = best).

    Examples:
        >>> candidate = ArchitectureCandidate(
        ...     config={"num_layers": 6, "hidden_dim": 512, "num_heads": 8},
        ...     performance=0.92,
        ...     cost=1.5e9,
        ...     rank=1,
        ... )
        >>> candidate.performance
        0.92
        >>> candidate.rank
        1
        >>> candidate.config["num_layers"]
        6
    """

    config: dict[str, int | float | str]
    performance: float
    cost: float
    rank: int


@dataclass(frozen=True, slots=True)
class NASStats:
    """Statistics from neural architecture search.

    Attributes:
        total_trials: Total number of architectures evaluated.
        best_performance: Best performance achieved.
        search_time_hours: Total search time in hours.
        pareto_front_size: Number of architectures on the Pareto front.

    Examples:
        >>> stats = NASStats(
        ...     total_trials=100,
        ...     best_performance=0.95,
        ...     search_time_hours=24.5,
        ...     pareto_front_size=5,
        ... )
        >>> stats.total_trials
        100
        >>> stats.best_performance
        0.95
        >>> stats.pareto_front_size
        5
    """

    total_trials: int
    best_performance: float
    search_time_hours: float
    pareto_front_size: int


def _validate_search_range(name: str, lo: int, hi: int) -> None:
    """Validate a search space range: min <= max and min > 0."""
    if lo > hi:
        msg = f"{name} min ({lo}) must be <= max ({hi})"
        raise ValueError(msg)
    if lo <= 0:
        msg = f"{name} min must be positive, got {lo}"
        raise ValueError(msg)


def validate_search_space_config(config: SearchSpaceConfig) -> None:
    """Validate search space configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = SearchSpaceConfig(
        ...     space_type=SearchSpace.MICRO,
        ...     num_layers_range=(4, 12),
        ...     hidden_dims_range=(256, 1024),
        ...     num_heads_range=(4, 16),
        ... )
        >>> validate_search_space_config(config)

        >>> bad_config = SearchSpaceConfig(
        ...     space_type=SearchSpace.MICRO,
        ...     num_layers_range=(12, 4),
        ...     hidden_dims_range=(256, 1024),
        ...     num_heads_range=(4, 16),
        ... )
        >>> validate_search_space_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_layers_range min (12) must be <= max (4)
    """
    ranges = (
        ("num_layers_range", config.num_layers_range),
        ("hidden_dims_range", config.hidden_dims_range),
        ("num_heads_range", config.num_heads_range),
    )
    for name, (lo, hi) in ranges:
        _validate_search_range(name, lo, hi)


def validate_search_config(config: SearchConfig) -> None:
    """Validate search configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> space_config = SearchSpaceConfig(
        ...     space_type=SearchSpace.MICRO,
        ...     num_layers_range=(4, 12),
        ...     hidden_dims_range=(256, 1024),
        ...     num_heads_range=(4, 16),
        ... )
        >>> config = SearchConfig(
        ...     strategy=SearchStrategy.RANDOM,
        ...     search_space=space_config,
        ...     num_trials=100,
        ...     early_stopping_patience=10,
        ... )
        >>> validate_search_config(config)

        >>> bad_config = SearchConfig(
        ...     strategy=SearchStrategy.RANDOM,
        ...     search_space=space_config,
        ...     num_trials=0,
        ...     early_stopping_patience=10,
        ... )
        >>> validate_search_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_trials must be positive, got 0
    """
    validate_search_space_config(config.search_space)

    if config.num_trials <= 0:
        msg = f"num_trials must be positive, got {config.num_trials}"
        raise ValueError(msg)

    if config.early_stopping_patience < 0:
        msg = (
            f"early_stopping_patience must be non-negative, "
            f"got {config.early_stopping_patience}"
        )
        raise ValueError(msg)


def validate_architecture_candidate(candidate: ArchitectureCandidate) -> None:
    """Validate architecture candidate parameters.

    Args:
        candidate: Candidate to validate.

    Raises:
        ValueError: If any value is invalid.

    Examples:
        >>> candidate = ArchitectureCandidate(
        ...     config={"num_layers": 6},
        ...     performance=0.92,
        ...     cost=1.5e9,
        ...     rank=1,
        ... )
        >>> validate_architecture_candidate(candidate)

        >>> bad_candidate = ArchitectureCandidate(
        ...     config={"num_layers": 6},
        ...     performance=-0.1,
        ...     cost=1.5e9,
        ...     rank=1,
        ... )
        >>> validate_architecture_candidate(bad_candidate)
        Traceback (most recent call last):
            ...
        ValueError: performance must be non-negative, got -0.1
    """
    if candidate.performance < 0:
        msg = f"performance must be non-negative, got {candidate.performance}"
        raise ValueError(msg)

    if candidate.cost < 0:
        msg = f"cost must be non-negative, got {candidate.cost}"
        raise ValueError(msg)

    if candidate.rank < 1:
        msg = f"rank must be positive, got {candidate.rank}"
        raise ValueError(msg)


def validate_nas_stats(stats: NASStats) -> None:
    """Validate NAS statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any statistic value is invalid.

    Examples:
        >>> stats = NASStats(
        ...     total_trials=100,
        ...     best_performance=0.95,
        ...     search_time_hours=24.5,
        ...     pareto_front_size=5,
        ... )
        >>> validate_nas_stats(stats)

        >>> bad_stats = NASStats(
        ...     total_trials=-1,
        ...     best_performance=0.95,
        ...     search_time_hours=24.5,
        ...     pareto_front_size=5,
        ... )
        >>> validate_nas_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: total_trials must be non-negative, got -1
    """
    if stats.total_trials < 0:
        msg = f"total_trials must be non-negative, got {stats.total_trials}"
        raise ValueError(msg)

    if stats.best_performance < 0:
        msg = f"best_performance must be non-negative, got {stats.best_performance}"
        raise ValueError(msg)

    if stats.search_time_hours < 0:
        msg = f"search_time_hours must be non-negative, got {stats.search_time_hours}"
        raise ValueError(msg)

    if stats.pareto_front_size < 0:
        msg = f"pareto_front_size must be non-negative, got {stats.pareto_front_size}"
        raise ValueError(msg)


def create_search_space_config(
    space_type: str | SearchSpace = SearchSpace.MICRO,
    num_layers_range: tuple[int, int] = (4, 12),
    hidden_dims_range: tuple[int, int] = (256, 1024),
    num_heads_range: tuple[int, int] = (4, 16),
) -> SearchSpaceConfig:
    """Create a search space configuration with validation.

    Args:
        space_type: Type of search space (micro, macro, hierarchical).
        num_layers_range: Tuple of (min, max) number of layers to search.
        hidden_dims_range: Tuple of (min, max) hidden dimensions to search.
        num_heads_range: Tuple of (min, max) attention heads to search.

    Returns:
        Validated SearchSpaceConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_search_space_config()
        >>> config.space_type
        <SearchSpace.MICRO: 'micro'>
        >>> config.num_layers_range
        (4, 12)

        >>> config = create_search_space_config(
        ...     space_type="macro",
        ...     num_layers_range=(6, 24),
        ... )
        >>> config.space_type
        <SearchSpace.MACRO: 'macro'>
        >>> config.num_layers_range
        (6, 24)

        >>> create_search_space_config(num_layers_range=(12, 4))
        Traceback (most recent call last):
            ...
        ValueError: num_layers_range min (12) must be <= max (4)
    """
    if isinstance(space_type, str):
        space_type = get_search_space(space_type)

    config = SearchSpaceConfig(
        space_type=space_type,
        num_layers_range=num_layers_range,
        hidden_dims_range=hidden_dims_range,
        num_heads_range=num_heads_range,
    )
    validate_search_space_config(config)
    return config


def create_search_config(
    strategy: str | SearchStrategy = SearchStrategy.RANDOM,
    space_type: str | SearchSpace = SearchSpace.MICRO,
    num_layers_range: tuple[int, int] = (4, 12),
    hidden_dims_range: tuple[int, int] = (256, 1024),
    num_heads_range: tuple[int, int] = (4, 16),
    num_trials: int = 100,
    early_stopping_patience: int = 10,
) -> SearchConfig:
    """Create a search configuration with validation.

    Args:
        strategy: Search strategy to use.
        space_type: Type of search space (micro, macro, hierarchical).
        num_layers_range: Tuple of (min, max) number of layers to search.
        hidden_dims_range: Tuple of (min, max) hidden dimensions to search.
        num_heads_range: Tuple of (min, max) attention heads to search.
        num_trials: Number of architecture trials to evaluate.
        early_stopping_patience: Patience for early stopping (0 to disable).

    Returns:
        Validated SearchConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_search_config()
        >>> config.strategy
        <SearchStrategy.RANDOM: 'random'>
        >>> config.num_trials
        100

        >>> config = create_search_config(
        ...     strategy="evolutionary",
        ...     num_trials=200,
        ...     early_stopping_patience=20,
        ... )
        >>> config.strategy
        <SearchStrategy.EVOLUTIONARY: 'evolutionary'>
        >>> config.num_trials
        200

        >>> create_search_config(num_trials=0)
        Traceback (most recent call last):
            ...
        ValueError: num_trials must be positive, got 0
    """
    if isinstance(strategy, str):
        strategy = get_search_strategy(strategy)

    search_space = create_search_space_config(
        space_type=space_type,
        num_layers_range=num_layers_range,
        hidden_dims_range=hidden_dims_range,
        num_heads_range=num_heads_range,
    )

    config = SearchConfig(
        strategy=strategy,
        search_space=search_space,
        num_trials=num_trials,
        early_stopping_patience=early_stopping_patience,
    )
    validate_search_config(config)
    return config


def create_architecture_candidate(
    config: dict[str, int | float | str] | None = None,
    performance: float = 0.0,
    cost: float = 0.0,
    rank: int = 1,
) -> ArchitectureCandidate:
    """Create an architecture candidate with validation.

    Args:
        config: Dictionary of architecture hyperparameters.
        performance: Performance metric value.
        cost: Computational cost.
        rank: Rank among all evaluated candidates.

    Returns:
        Validated ArchitectureCandidate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> candidate = create_architecture_candidate()
        >>> candidate.performance
        0.0
        >>> candidate.rank
        1

        >>> candidate = create_architecture_candidate(
        ...     config={"num_layers": 6, "hidden_dim": 512},
        ...     performance=0.92,
        ...     cost=1.5e9,
        ...     rank=1,
        ... )
        >>> candidate.performance
        0.92

        >>> create_architecture_candidate(performance=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: performance must be non-negative, got -0.1
    """
    if config is None:
        config = {}

    candidate = ArchitectureCandidate(
        config=config,
        performance=performance,
        cost=cost,
        rank=rank,
    )
    validate_architecture_candidate(candidate)
    return candidate


def create_nas_stats(
    total_trials: int = 0,
    best_performance: float = 0.0,
    search_time_hours: float = 0.0,
    pareto_front_size: int = 0,
) -> NASStats:
    """Create NAS statistics with validation.

    Args:
        total_trials: Total number of architectures evaluated.
        best_performance: Best performance achieved.
        search_time_hours: Total search time in hours.
        pareto_front_size: Number of architectures on the Pareto front.

    Returns:
        Validated NASStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_nas_stats()
        >>> stats.total_trials
        0
        >>> stats.best_performance
        0.0

        >>> stats = create_nas_stats(
        ...     total_trials=100,
        ...     best_performance=0.95,
        ...     search_time_hours=24.5,
        ...     pareto_front_size=5,
        ... )
        >>> stats.total_trials
        100

        >>> create_nas_stats(total_trials=-1)
        Traceback (most recent call last):
            ...
        ValueError: total_trials must be non-negative, got -1
    """
    stats = NASStats(
        total_trials=total_trials,
        best_performance=best_performance,
        search_time_hours=search_time_hours,
        pareto_front_size=pareto_front_size,
    )
    validate_nas_stats(stats)
    return stats


def list_search_strategies() -> list[str]:
    """List all available search strategies.

    Returns:
        Sorted list of search strategy names.

    Examples:
        >>> strategies = list_search_strategies()
        >>> "random" in strategies
        True
        >>> "evolutionary" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SEARCH_STRATEGIES)


def list_search_spaces() -> list[str]:
    """List all available search space types.

    Returns:
        Sorted list of search space names.

    Examples:
        >>> spaces = list_search_spaces()
        >>> "micro" in spaces
        True
        >>> "macro" in spaces
        True
        >>> spaces == sorted(spaces)
        True
    """
    return sorted(VALID_SEARCH_SPACES)


def list_performance_predictors() -> list[str]:
    """List all available performance predictors.

    Returns:
        Sorted list of performance predictor names.

    Examples:
        >>> predictors = list_performance_predictors()
        >>> "surrogate" in predictors
        True
        >>> "zero_cost" in predictors
        True
        >>> predictors == sorted(predictors)
        True
    """
    return sorted(VALID_PERFORMANCE_PREDICTORS)


def get_search_strategy(name: str) -> SearchStrategy:
    """Get search strategy enum from string name.

    Args:
        name: Name of the search strategy.

    Returns:
        Corresponding SearchStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_search_strategy("random")
        <SearchStrategy.RANDOM: 'random'>
        >>> get_search_strategy("evolutionary")
        <SearchStrategy.EVOLUTIONARY: 'evolutionary'>

        >>> get_search_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: strategy must be one of ...
    """
    if name not in VALID_SEARCH_STRATEGIES:
        msg = f"strategy must be one of {VALID_SEARCH_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return SearchStrategy(name)


def get_search_space(name: str) -> SearchSpace:
    """Get search space enum from string name.

    Args:
        name: Name of the search space.

    Returns:
        Corresponding SearchSpace enum.

    Raises:
        ValueError: If space name is invalid.

    Examples:
        >>> get_search_space("micro")
        <SearchSpace.MICRO: 'micro'>
        >>> get_search_space("macro")
        <SearchSpace.MACRO: 'macro'>

        >>> get_search_space("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: search_space must be one of ...
    """
    if name not in VALID_SEARCH_SPACES:
        msg = f"search_space must be one of {VALID_SEARCH_SPACES}, got '{name}'"
        raise ValueError(msg)
    return SearchSpace(name)


def get_performance_predictor(name: str) -> PerformancePredictor:
    """Get performance predictor enum from string name.

    Args:
        name: Name of the performance predictor.

    Returns:
        Corresponding PerformancePredictor enum.

    Raises:
        ValueError: If predictor name is invalid.

    Examples:
        >>> get_performance_predictor("surrogate")
        <PerformancePredictor.SURROGATE: 'surrogate'>
        >>> get_performance_predictor("zero_cost")
        <PerformancePredictor.ZERO_COST: 'zero_cost'>

        >>> get_performance_predictor("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: performance_predictor must be one of ...
    """
    if name not in VALID_PERFORMANCE_PREDICTORS:
        msg = (
            f"performance_predictor must be one of {VALID_PERFORMANCE_PREDICTORS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return PerformancePredictor(name)


def calculate_search_space_size(config: SearchSpaceConfig) -> int:
    """Calculate the size of the search space.

    Calculates the total number of possible architectures based on the
    search space configuration ranges.

    Args:
        config: Search space configuration.

    Returns:
        Total number of possible architectures.

    Examples:
        >>> config = create_search_space_config(
        ...     num_layers_range=(4, 6),
        ...     hidden_dims_range=(256, 258),
        ...     num_heads_range=(4, 8),
        ... )
        >>> calculate_search_space_size(config)
        45

        >>> config = create_search_space_config(
        ...     num_layers_range=(6, 6),
        ...     hidden_dims_range=(512, 512),
        ...     num_heads_range=(8, 8),
        ... )
        >>> calculate_search_space_size(config)
        1
    """
    min_layers, max_layers = config.num_layers_range
    min_hidden, max_hidden = config.hidden_dims_range
    min_heads, max_heads = config.num_heads_range

    num_layers_options = max_layers - min_layers + 1
    num_hidden_options = max_hidden - min_hidden + 1
    num_heads_options = max_heads - min_heads + 1

    return num_layers_options * num_hidden_options * num_heads_options


def estimate_search_cost(
    config: SearchConfig,
    hours_per_trial: float = 0.5,
    gpu_cost_per_hour: float = 1.0,
) -> float:
    """Estimate the total cost of the architecture search.

    Args:
        config: Search configuration.
        hours_per_trial: Average hours per architecture evaluation.
        gpu_cost_per_hour: Cost per GPU hour in dollars.

    Returns:
        Estimated total cost in dollars.

    Raises:
        ValueError: If hours_per_trial or gpu_cost_per_hour is invalid.

    Examples:
        >>> config = create_search_config(num_trials=100)
        >>> estimate_search_cost(config, hours_per_trial=0.5, gpu_cost_per_hour=1.0)
        50.0

        >>> config = create_search_config(num_trials=50)
        >>> estimate_search_cost(config, hours_per_trial=1.0, gpu_cost_per_hour=2.0)
        100.0

        >>> estimate_search_cost(config, hours_per_trial=-1.0)
        Traceback (most recent call last):
            ...
        ValueError: hours_per_trial must be positive, got -1.0
    """
    if hours_per_trial <= 0:
        msg = f"hours_per_trial must be positive, got {hours_per_trial}"
        raise ValueError(msg)
    if gpu_cost_per_hour < 0:
        msg = f"gpu_cost_per_hour must be non-negative, got {gpu_cost_per_hour}"
        raise ValueError(msg)

    total_hours = config.num_trials * hours_per_trial
    return total_hours * gpu_cost_per_hour


def evaluate_architecture(
    candidate_config: dict[str, int | float | str],
    performance_metric: float,
    cost_metric: float,
) -> ArchitectureCandidate:
    """Evaluate an architecture and create a candidate.

    Args:
        candidate_config: Dictionary of architecture hyperparameters.
        performance_metric: Performance metric value (e.g., accuracy).
        cost_metric: Computational cost metric (e.g., FLOPs).

    Returns:
        Architecture candidate with evaluation results.

    Raises:
        ValueError: If metrics are invalid.

    Examples:
        >>> result = evaluate_architecture(
        ...     {"num_layers": 6, "hidden_dim": 512},
        ...     performance_metric=0.92,
        ...     cost_metric=1.5e9,
        ... )
        >>> result.performance
        0.92
        >>> result.cost
        1500000000.0
        >>> result.rank
        1

        >>> evaluate_architecture({}, performance_metric=-0.1, cost_metric=1.0)
        Traceback (most recent call last):
            ...
        ValueError: performance_metric must be non-negative, got -0.1
    """
    if performance_metric < 0:
        msg = f"performance_metric must be non-negative, got {performance_metric}"
        raise ValueError(msg)
    if cost_metric < 0:
        msg = f"cost_metric must be non-negative, got {cost_metric}"
        raise ValueError(msg)

    return ArchitectureCandidate(
        config=candidate_config,
        performance=performance_metric,
        cost=cost_metric,
        rank=1,  # Will be updated when comparing with other candidates
    )


def select_pareto_optimal(
    candidates: tuple[ArchitectureCandidate, ...],
) -> tuple[ArchitectureCandidate, ...]:
    """Select Pareto optimal candidates from a set.

    Returns candidates that are not dominated by any other candidate.
    A candidate is dominated if another candidate has both higher performance
    and lower cost.

    Args:
        candidates: Tuple of architecture candidates.

    Returns:
        Tuple of Pareto optimal candidates with updated ranks.

    Raises:
        ValueError: If candidates is empty.

    Examples:
        >>> c1 = create_architecture_candidate(
        ...     config={"id": 1}, performance=0.9, cost=100, rank=1
        ... )
        >>> c2 = create_architecture_candidate(
        ...     config={"id": 2}, performance=0.85, cost=80, rank=2
        ... )
        >>> c3 = create_architecture_candidate(
        ...     config={"id": 3}, performance=0.8, cost=120, rank=3
        ... )
        >>> pareto = select_pareto_optimal((c1, c2, c3))
        >>> len(pareto)
        2
        >>> sorted([c.config["id"] for c in pareto])
        [1, 2]

        >>> select_pareto_optimal(())
        Traceback (most recent call last):
            ...
        ValueError: candidates cannot be empty
    """
    if not candidates:
        msg = "candidates cannot be empty"
        raise ValueError(msg)

    pareto_front = []
    for candidate in candidates:
        is_dominated = False
        for other in candidates:
            if other is candidate:
                continue
            # Check if 'other' dominates 'candidate'
            if (
                other.performance >= candidate.performance
                and other.cost <= candidate.cost
                and (
                    other.performance > candidate.performance
                    or other.cost < candidate.cost
                )
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(candidate)

    # Sort by performance (descending) and assign ranks
    pareto_front.sort(key=lambda c: c.performance, reverse=True)
    ranked = []
    for i, candidate in enumerate(pareto_front, 1):
        ranked.append(
            ArchitectureCandidate(
                config=candidate.config,
                performance=candidate.performance,
                cost=candidate.cost,
                rank=i,
            )
        )

    return tuple(ranked)


def format_nas_stats(stats: NASStats) -> str:
    """Format NAS statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_nas_stats(
        ...     total_trials=100,
        ...     best_performance=0.95,
        ...     search_time_hours=24.5,
        ...     pareto_front_size=5,
        ... )
        >>> formatted = format_nas_stats(stats)
        >>> "Total Trials: 100" in formatted
        True
        >>> "Best Performance: 95.0%" in formatted
        True
        >>> "Search Time: 24.5 hours" in formatted
        True
        >>> "Pareto Front Size: 5" in formatted
        True
    """
    return (
        f"NAS Statistics:\n"
        f"  Total Trials: {stats.total_trials}\n"
        f"  Best Performance: {stats.best_performance * 100:.1f}%\n"
        f"  Search Time: {stats.search_time_hours} hours\n"
        f"  Pareto Front Size: {stats.pareto_front_size}"
    )


def get_recommended_nas_config(task_type: str) -> SearchConfig:
    """Get recommended NAS configuration for a task type.

    Args:
        task_type: Type of task (classification, generation, translation,
            summarization, qa).

    Returns:
        Recommended SearchConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_nas_config("classification")
        >>> config.strategy
        <SearchStrategy.EVOLUTIONARY: 'evolutionary'>
        >>> config.search_space.space_type
        <SearchSpace.MICRO: 'micro'>

        >>> config = get_recommended_nas_config("generation")
        >>> config.strategy
        <SearchStrategy.DIFFERENTIABLE: 'differentiable'>
        >>> config.search_space.space_type
        <SearchSpace.HIERARCHICAL: 'hierarchical'>

        >>> get_recommended_nas_config("unknown")  # doctest: +ELLIPSIS
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
        return create_search_config(
            strategy=SearchStrategy.EVOLUTIONARY,
            space_type=SearchSpace.MICRO,
            num_layers_range=(4, 12),
            hidden_dims_range=(256, 768),
            num_heads_range=(4, 12),
            num_trials=100,
            early_stopping_patience=15,
        )
    elif task_type == "generation":
        return create_search_config(
            strategy=SearchStrategy.DIFFERENTIABLE,
            space_type=SearchSpace.HIERARCHICAL,
            num_layers_range=(6, 24),
            hidden_dims_range=(512, 2048),
            num_heads_range=(8, 32),
            num_trials=50,
            early_stopping_patience=10,
        )
    elif task_type == "translation":
        return create_search_config(
            strategy=SearchStrategy.REINFORCEMENT,
            space_type=SearchSpace.MACRO,
            num_layers_range=(6, 18),
            hidden_dims_range=(512, 1024),
            num_heads_range=(8, 16),
            num_trials=80,
            early_stopping_patience=12,
        )
    elif task_type == "summarization":
        return create_search_config(
            strategy=SearchStrategy.EVOLUTIONARY,
            space_type=SearchSpace.HIERARCHICAL,
            num_layers_range=(6, 12),
            hidden_dims_range=(512, 1024),
            num_heads_range=(8, 16),
            num_trials=60,
            early_stopping_patience=10,
        )
    else:  # qa
        return create_search_config(
            strategy=SearchStrategy.RANDOM,
            space_type=SearchSpace.MICRO,
            num_layers_range=(4, 12),
            hidden_dims_range=(256, 768),
            num_heads_range=(4, 12),
            num_trials=100,
            early_stopping_patience=20,
        )


def estimate_architecture_cost(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    sequence_length: int = 512,
) -> float:
    """Estimate computational cost (FLOPs) for an architecture.

    Estimates the FLOPs for a single forward pass through a transformer
    architecture with the given parameters.

    Args:
        num_layers: Number of transformer layers.
        hidden_dim: Hidden dimension size.
        num_heads: Number of attention heads.
        sequence_length: Input sequence length.

    Returns:
        Estimated FLOPs for a forward pass.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> cost = estimate_architecture_cost(6, 512, 8)
        >>> cost > 0
        True

        >>> cost_small = estimate_architecture_cost(6, 256, 4)
        >>> cost_large = estimate_architecture_cost(12, 512, 8)
        >>> cost_small < cost_large
        True

        >>> estimate_architecture_cost(0, 512, 8)
        Traceback (most recent call last):
            ...
        ValueError: num_layers must be positive, got 0
    """
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)
    if hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {hidden_dim}"
        raise ValueError(msg)
    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)
    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    # Attention FLOPs: 4 * seq_len^2 * hidden_dim per layer
    attention_flops = 4 * sequence_length * sequence_length * hidden_dim

    # FFN FLOPs: 8 * seq_len * hidden_dim^2 per layer (assuming 4x expansion)
    ffn_flops = 8 * sequence_length * hidden_dim * hidden_dim

    # Total per layer
    flops_per_layer = attention_flops + ffn_flops

    # Total for all layers
    return float(num_layers * flops_per_layer)


def calculate_efficiency_score(
    performance: float,
    cost: float,
    performance_weight: float = 0.7,
) -> float:
    """Calculate an efficiency score balancing performance and cost.

    Args:
        performance: Performance metric (0 to 1).
        cost: Computational cost (FLOPs or similar).
        performance_weight: Weight for performance vs cost (0 to 1).

    Returns:
        Efficiency score (higher is better).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> round(calculate_efficiency_score(0.9, 1e9), 4)
        0.855

        >>> score_efficient = calculate_efficiency_score(0.85, 1e8)
        >>> score_costly = calculate_efficiency_score(0.9, 1e10)
        >>> score_efficient > score_costly
        True

        >>> calculate_efficiency_score(1.5, 1e9)
        Traceback (most recent call last):
            ...
        ValueError: performance must be between 0 and 1, got 1.5
    """
    if not 0 <= performance <= 1:
        msg = f"performance must be between 0 and 1, got {performance}"
        raise ValueError(msg)
    if cost <= 0:
        msg = f"cost must be positive, got {cost}"
        raise ValueError(msg)
    if not 0 <= performance_weight <= 1:
        msg = f"performance_weight must be between 0 and 1, got {performance_weight}"
        raise ValueError(msg)

    # Normalize cost using log scale (1e8 to 1e12 typical range)
    log_cost = math.log10(cost)
    normalized_cost = max(0, min(1, (log_cost - 8) / 4))  # Maps 1e8-1e12 to 0-1

    # Efficiency favors low cost
    cost_score = 1 - normalized_cost

    # Weighted combination
    return performance_weight * performance + (1 - performance_weight) * cost_score
