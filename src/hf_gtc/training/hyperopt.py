"""Hyperparameter optimization utilities for training.

This module provides functions for configuring and implementing hyperparameter
optimization strategies, enabling efficient search over hyperparameter spaces
using various algorithms like grid search, random search, Bayesian optimization,
TPE, CMA-ES, and Hyperband.

Examples:
    >>> from hf_gtc.training.hyperopt import (
    ...     create_hyperopt_config,
    ...     SearchAlgorithm,
    ... )
    >>> config = create_hyperopt_config()
    >>> config.algorithm
    <SearchAlgorithm.RANDOM: 'random'>
    >>> config.n_trials
    100
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SearchAlgorithm(Enum):
    """Algorithms for hyperparameter search.

    Attributes:
        GRID: Exhaustive grid search over parameter combinations.
        RANDOM: Random sampling from parameter distributions.
        BAYESIAN: Bayesian optimization with Gaussian processes.
        TPE: Tree-structured Parzen Estimator.
        CMAES: Covariance Matrix Adaptation Evolution Strategy.
        HYPERBAND: Adaptive resource allocation with early stopping.

    Examples:
        >>> SearchAlgorithm.GRID.value
        'grid'
        >>> SearchAlgorithm.RANDOM.value
        'random'
        >>> SearchAlgorithm.BAYESIAN.value
        'bayesian'
    """

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    TPE = "tpe"
    CMAES = "cmaes"
    HYPERBAND = "hyperband"


class ParameterType(Enum):
    """Types of hyperparameters for optimization.

    Attributes:
        CONTINUOUS: Continuous real-valued parameters.
        DISCRETE: Discrete integer-valued parameters.
        CATEGORICAL: Categorical parameters with fixed choices.
        LOG_UNIFORM: Log-uniformly distributed parameters.

    Examples:
        >>> ParameterType.CONTINUOUS.value
        'continuous'
        >>> ParameterType.DISCRETE.value
        'discrete'
        >>> ParameterType.CATEGORICAL.value
        'categorical'
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"


class ObjectiveDirection(Enum):
    """Direction of optimization objective.

    Attributes:
        MINIMIZE: Minimize the objective value (e.g., loss).
        MAXIMIZE: Maximize the objective value (e.g., accuracy).

    Examples:
        >>> ObjectiveDirection.MINIMIZE.value
        'minimize'
        >>> ObjectiveDirection.MAXIMIZE.value
        'maximize'
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


VALID_SEARCH_ALGORITHMS = frozenset(a.value for a in SearchAlgorithm)
VALID_PARAMETER_TYPES = frozenset(p.value for p in ParameterType)
VALID_OBJECTIVE_DIRECTIONS = frozenset(d.value for d in ObjectiveDirection)


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """Definition of a single hyperparameter space.

    Attributes:
        name: Name of the hyperparameter.
        param_type: Type of the parameter.
        low: Lower bound for continuous/discrete/log_uniform types.
        high: Upper bound for continuous/discrete/log_uniform types.
        choices: Possible values for categorical type.

    Examples:
        >>> space = ParameterSpace(
        ...     name="learning_rate",
        ...     param_type=ParameterType.LOG_UNIFORM,
        ...     low=1e-5,
        ...     high=1e-1,
        ...     choices=None,
        ... )
        >>> space.name
        'learning_rate'
        >>> space.param_type
        <ParameterType.LOG_UNIFORM: 'log_uniform'>
        >>> space.low
        1e-05
    """

    name: str
    param_type: ParameterType
    low: float | None
    high: float | None
    choices: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class HyperoptConfig:
    """Main configuration for hyperparameter optimization.

    Attributes:
        algorithm: Search algorithm to use.
        parameter_spaces: Tuple of parameter space definitions.
        n_trials: Number of trials to run.
        timeout_seconds: Maximum time for optimization in seconds.
        direction: Direction of optimization (minimize or maximize).

    Examples:
        >>> space = ParameterSpace(
        ...     name="lr",
        ...     param_type=ParameterType.LOG_UNIFORM,
        ...     low=1e-5,
        ...     high=1e-1,
        ...     choices=None,
        ... )
        >>> config = HyperoptConfig(
        ...     algorithm=SearchAlgorithm.RANDOM,
        ...     parameter_spaces=(space,),
        ...     n_trials=100,
        ...     timeout_seconds=3600,
        ...     direction=ObjectiveDirection.MINIMIZE,
        ... )
        >>> config.algorithm
        <SearchAlgorithm.RANDOM: 'random'>
        >>> config.n_trials
        100
    """

    algorithm: SearchAlgorithm
    parameter_spaces: tuple[ParameterSpace, ...]
    n_trials: int
    timeout_seconds: int
    direction: ObjectiveDirection


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Result of a single hyperparameter trial.

    Attributes:
        params: Dictionary of parameter name to value.
        objective_value: The objective metric value achieved.
        trial_number: The trial number (1-indexed).
        duration_seconds: Time taken for this trial in seconds.

    Examples:
        >>> result = TrialResult(
        ...     params={"lr": 0.001, "batch_size": 32},
        ...     objective_value=0.05,
        ...     trial_number=1,
        ...     duration_seconds=120.5,
        ... )
        >>> result.objective_value
        0.05
        >>> result.trial_number
        1
    """

    params: dict[str, float | int | str]
    objective_value: float
    trial_number: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class HyperoptStats:
    """Statistics from hyperparameter optimization.

    Attributes:
        total_trials: Total number of trials completed.
        best_value: Best objective value achieved.
        best_params: Parameters that achieved the best value.
        convergence_curve: Tuple of best values at each trial.

    Examples:
        >>> stats = HyperoptStats(
        ...     total_trials=50,
        ...     best_value=0.02,
        ...     best_params={"lr": 0.0005, "batch_size": 64},
        ...     convergence_curve=(0.1, 0.08, 0.05, 0.03, 0.02),
        ... )
        >>> stats.total_trials
        50
        >>> stats.best_value
        0.02
    """

    total_trials: int
    best_value: float
    best_params: dict[str, float | int | str]
    convergence_curve: tuple[float, ...]


def validate_parameter_space(space: ParameterSpace) -> None:
    """Validate parameter space configuration.

    Args:
        space: Parameter space to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> space = ParameterSpace(
        ...     name="learning_rate",
        ...     param_type=ParameterType.CONTINUOUS,
        ...     low=0.0,
        ...     high=1.0,
        ...     choices=None,
        ... )
        >>> validate_parameter_space(space)

        >>> bad_space = ParameterSpace(
        ...     name="",
        ...     param_type=ParameterType.CONTINUOUS,
        ...     low=0.0,
        ...     high=1.0,
        ...     choices=None,
        ... )
        >>> validate_parameter_space(bad_space)
        Traceback (most recent call last):
            ...
        ValueError: name cannot be empty
    """
    if not space.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if space.param_type == ParameterType.CATEGORICAL:
        _validate_categorical_space(space)
    else:
        _validate_numeric_space(space)


def _validate_categorical_space(space: ParameterSpace) -> None:
    """Validate a categorical parameter space."""
    if space.choices is None or len(space.choices) == 0:
        msg = "categorical parameter must have non-empty choices"
        raise ValueError(msg)


def _validate_numeric_space(space: ParameterSpace) -> None:
    """Validate a numeric (continuous/discrete/log_uniform) parameter space."""
    if space.low is None or space.high is None:
        msg = f"{space.param_type.value} parameter must have low and high bounds"
        raise ValueError(msg)
    if space.low >= space.high:
        msg = f"low ({space.low}) must be less than high ({space.high})"
        raise ValueError(msg)
    if space.param_type == ParameterType.LOG_UNIFORM and space.low <= 0:
        msg = f"log_uniform parameter low must be positive, got {space.low}"
        raise ValueError(msg)


def validate_hyperopt_config(config: HyperoptConfig) -> None:
    """Validate hyperparameter optimization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> space = ParameterSpace(
        ...     name="lr",
        ...     param_type=ParameterType.CONTINUOUS,
        ...     low=0.0,
        ...     high=1.0,
        ...     choices=None,
        ... )
        >>> config = HyperoptConfig(
        ...     algorithm=SearchAlgorithm.RANDOM,
        ...     parameter_spaces=(space,),
        ...     n_trials=100,
        ...     timeout_seconds=3600,
        ...     direction=ObjectiveDirection.MINIMIZE,
        ... )
        >>> validate_hyperopt_config(config)

        >>> bad_config = HyperoptConfig(
        ...     algorithm=SearchAlgorithm.RANDOM,
        ...     parameter_spaces=(),
        ...     n_trials=100,
        ...     timeout_seconds=3600,
        ...     direction=ObjectiveDirection.MINIMIZE,
        ... )
        >>> validate_hyperopt_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: parameter_spaces cannot be empty
    """
    if not config.parameter_spaces:
        msg = "parameter_spaces cannot be empty"
        raise ValueError(msg)

    for space in config.parameter_spaces:
        validate_parameter_space(space)

    if config.n_trials <= 0:
        msg = f"n_trials must be positive, got {config.n_trials}"
        raise ValueError(msg)

    if config.timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {config.timeout_seconds}"
        raise ValueError(msg)


def validate_trial_result(result: TrialResult) -> None:
    """Validate trial result.

    Args:
        result: Trial result to validate.

    Raises:
        ValueError: If any value is invalid.

    Examples:
        >>> result = TrialResult(
        ...     params={"lr": 0.001},
        ...     objective_value=0.05,
        ...     trial_number=1,
        ...     duration_seconds=120.5,
        ... )
        >>> validate_trial_result(result)

        >>> bad_result = TrialResult(
        ...     params={},
        ...     objective_value=0.05,
        ...     trial_number=1,
        ...     duration_seconds=120.5,
        ... )
        >>> validate_trial_result(bad_result)
        Traceback (most recent call last):
            ...
        ValueError: params cannot be empty
    """
    if not result.params:
        msg = "params cannot be empty"
        raise ValueError(msg)

    if result.trial_number <= 0:
        msg = f"trial_number must be positive, got {result.trial_number}"
        raise ValueError(msg)

    if result.duration_seconds < 0:
        msg = f"duration_seconds must be non-negative, got {result.duration_seconds}"
        raise ValueError(msg)


def validate_hyperopt_stats(stats: HyperoptStats) -> None:
    """Validate hyperparameter optimization statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any value is invalid.

    Examples:
        >>> stats = HyperoptStats(
        ...     total_trials=50,
        ...     best_value=0.02,
        ...     best_params={"lr": 0.0005},
        ...     convergence_curve=(0.1, 0.05, 0.02),
        ... )
        >>> validate_hyperopt_stats(stats)

        >>> bad_stats = HyperoptStats(
        ...     total_trials=-1,
        ...     best_value=0.02,
        ...     best_params={"lr": 0.0005},
        ...     convergence_curve=(0.1, 0.05, 0.02),
        ... )
        >>> validate_hyperopt_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: total_trials must be non-negative, got -1
    """
    if stats.total_trials < 0:
        msg = f"total_trials must be non-negative, got {stats.total_trials}"
        raise ValueError(msg)

    if not stats.best_params:
        msg = "best_params cannot be empty"
        raise ValueError(msg)


def create_parameter_space(
    name: str,
    param_type: str | ParameterType = ParameterType.CONTINUOUS,
    low: float | None = None,
    high: float | None = None,
    choices: tuple[str, ...] | None = None,
) -> ParameterSpace:
    """Create a parameter space with validation.

    Args:
        name: Name of the hyperparameter.
        param_type: Type of the parameter.
        low: Lower bound for continuous/discrete/log_uniform types.
        high: Upper bound for continuous/discrete/log_uniform types.
        choices: Possible values for categorical type.

    Returns:
        Validated ParameterSpace.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> space = create_parameter_space("learning_rate", "log_uniform", 1e-5, 1e-1)
        >>> space.name
        'learning_rate'
        >>> space.param_type
        <ParameterType.LOG_UNIFORM: 'log_uniform'>

        >>> space = create_parameter_space(
        ...     "optimizer",
        ...     param_type="categorical",
        ...     choices=("adam", "sgd", "adamw"),
        ... )
        >>> space.choices
        ('adam', 'sgd', 'adamw')

        >>> create_parameter_space("")
        Traceback (most recent call last):
            ...
        ValueError: name cannot be empty
    """
    if isinstance(param_type, str):
        param_type = get_parameter_type(param_type)

    space = ParameterSpace(
        name=name,
        param_type=param_type,
        low=low,
        high=high,
        choices=choices,
    )
    validate_parameter_space(space)
    return space


def create_hyperopt_config(
    algorithm: str | SearchAlgorithm = SearchAlgorithm.RANDOM,
    parameter_spaces: tuple[ParameterSpace, ...] | None = None,
    n_trials: int = 100,
    timeout_seconds: int = 3600,
    direction: str | ObjectiveDirection = ObjectiveDirection.MINIMIZE,
) -> HyperoptConfig:
    """Create a hyperparameter optimization configuration with validation.

    Args:
        algorithm: Search algorithm to use.
        parameter_spaces: Tuple of parameter space definitions.
        n_trials: Number of trials to run.
        timeout_seconds: Maximum time for optimization in seconds.
        direction: Direction of optimization (minimize or maximize).

    Returns:
        Validated HyperoptConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_hyperopt_config()
        >>> config.algorithm
        <SearchAlgorithm.RANDOM: 'random'>
        >>> config.n_trials
        100

        >>> lr_space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        >>> config = create_hyperopt_config(
        ...     algorithm="bayesian",
        ...     parameter_spaces=(lr_space,),
        ...     n_trials=50,
        ... )
        >>> config.algorithm
        <SearchAlgorithm.BAYESIAN: 'bayesian'>

        >>> create_hyperopt_config(n_trials=0)
        Traceback (most recent call last):
            ...
        ValueError: n_trials must be positive, got 0
    """
    if isinstance(algorithm, str):
        algorithm = get_search_algorithm(algorithm)

    if isinstance(direction, str):
        direction = get_objective_direction(direction)

    if parameter_spaces is None:
        # Default parameter space for demonstration
        parameter_spaces = (
            create_parameter_space(
                "learning_rate", ParameterType.LOG_UNIFORM, 1e-5, 1e-1
            ),
        )

    config = HyperoptConfig(
        algorithm=algorithm,
        parameter_spaces=parameter_spaces,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        direction=direction,
    )
    validate_hyperopt_config(config)
    return config


def create_trial_result(
    params: dict[str, float | int | str],
    objective_value: float,
    trial_number: int,
    duration_seconds: float = 0.0,
) -> TrialResult:
    """Create a trial result with validation.

    Args:
        params: Dictionary of parameter name to value.
        objective_value: The objective metric value achieved.
        trial_number: The trial number (1-indexed).
        duration_seconds: Time taken for this trial in seconds.

    Returns:
        Validated TrialResult.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> result = create_trial_result(
        ...     params={"lr": 0.001, "batch_size": 32},
        ...     objective_value=0.05,
        ...     trial_number=1,
        ... )
        >>> result.objective_value
        0.05

        >>> create_trial_result({}, 0.05, 1)
        Traceback (most recent call last):
            ...
        ValueError: params cannot be empty
    """
    result = TrialResult(
        params=params,
        objective_value=objective_value,
        trial_number=trial_number,
        duration_seconds=duration_seconds,
    )
    validate_trial_result(result)
    return result


def create_hyperopt_stats(
    total_trials: int = 0,
    best_value: float = float("inf"),
    best_params: dict[str, float | int | str] | None = None,
    convergence_curve: tuple[float, ...] = (),
) -> HyperoptStats:
    """Create hyperparameter optimization statistics with validation.

    Args:
        total_trials: Total number of trials completed.
        best_value: Best objective value achieved.
        best_params: Parameters that achieved the best value.
        convergence_curve: Tuple of best values at each trial.

    Returns:
        Validated HyperoptStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_hyperopt_stats(
        ...     total_trials=50,
        ...     best_value=0.02,
        ...     best_params={"lr": 0.0005},
        ... )
        >>> stats.total_trials
        50

        >>> create_hyperopt_stats(total_trials=-1, best_params={"lr": 0.001})
        Traceback (most recent call last):
            ...
        ValueError: total_trials must be non-negative, got -1
    """
    if best_params is None:
        best_params = {"placeholder": 0.0}

    stats = HyperoptStats(
        total_trials=total_trials,
        best_value=best_value,
        best_params=best_params,
        convergence_curve=convergence_curve,
    )
    validate_hyperopt_stats(stats)
    return stats


def list_search_algorithms() -> list[str]:
    """List all available search algorithms.

    Returns:
        Sorted list of search algorithm names.

    Examples:
        >>> algorithms = list_search_algorithms()
        >>> "random" in algorithms
        True
        >>> "bayesian" in algorithms
        True
        >>> algorithms == sorted(algorithms)
        True
    """
    return sorted(VALID_SEARCH_ALGORITHMS)


def list_parameter_types() -> list[str]:
    """List all available parameter types.

    Returns:
        Sorted list of parameter type names.

    Examples:
        >>> types = list_parameter_types()
        >>> "continuous" in types
        True
        >>> "categorical" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_PARAMETER_TYPES)


def list_objective_directions() -> list[str]:
    """List all available objective directions.

    Returns:
        Sorted list of objective direction names.

    Examples:
        >>> directions = list_objective_directions()
        >>> "minimize" in directions
        True
        >>> "maximize" in directions
        True
        >>> directions == sorted(directions)
        True
    """
    return sorted(VALID_OBJECTIVE_DIRECTIONS)


def get_search_algorithm(name: str) -> SearchAlgorithm:
    """Get search algorithm enum from string name.

    Args:
        name: Name of the search algorithm.

    Returns:
        Corresponding SearchAlgorithm enum.

    Raises:
        ValueError: If algorithm name is invalid.

    Examples:
        >>> get_search_algorithm("random")
        <SearchAlgorithm.RANDOM: 'random'>
        >>> get_search_algorithm("bayesian")
        <SearchAlgorithm.BAYESIAN: 'bayesian'>

        >>> get_search_algorithm("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: algorithm must be one of ...
    """
    if name not in VALID_SEARCH_ALGORITHMS:
        msg = f"algorithm must be one of {VALID_SEARCH_ALGORITHMS}, got '{name}'"
        raise ValueError(msg)
    return SearchAlgorithm(name)


def get_parameter_type(name: str) -> ParameterType:
    """Get parameter type enum from string name.

    Args:
        name: Name of the parameter type.

    Returns:
        Corresponding ParameterType enum.

    Raises:
        ValueError: If type name is invalid.

    Examples:
        >>> get_parameter_type("continuous")
        <ParameterType.CONTINUOUS: 'continuous'>
        >>> get_parameter_type("categorical")
        <ParameterType.CATEGORICAL: 'categorical'>

        >>> get_parameter_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: parameter_type must be one of ...
    """
    if name not in VALID_PARAMETER_TYPES:
        msg = f"parameter_type must be one of {VALID_PARAMETER_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ParameterType(name)


def get_objective_direction(name: str) -> ObjectiveDirection:
    """Get objective direction enum from string name.

    Args:
        name: Name of the objective direction.

    Returns:
        Corresponding ObjectiveDirection enum.

    Raises:
        ValueError: If direction name is invalid.

    Examples:
        >>> get_objective_direction("minimize")
        <ObjectiveDirection.MINIMIZE: 'minimize'>
        >>> get_objective_direction("maximize")
        <ObjectiveDirection.MAXIMIZE: 'maximize'>

        >>> get_objective_direction("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: direction must be one of ...
    """
    if name not in VALID_OBJECTIVE_DIRECTIONS:
        msg = f"direction must be one of {VALID_OBJECTIVE_DIRECTIONS}, got '{name}'"
        raise ValueError(msg)
    return ObjectiveDirection(name)


def sample_parameters(
    spaces: tuple[ParameterSpace, ...],
    seed: int | None = None,
) -> dict[str, float | int | str]:
    """Sample a set of parameters from the parameter spaces.

    Args:
        spaces: Tuple of parameter space definitions.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of parameter name to sampled value.

    Raises:
        ValueError: If spaces is empty.

    Examples:
        >>> lr_space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        >>> batch_space = create_parameter_space("batch_size", "discrete", 8, 128)
        >>> params = sample_parameters((lr_space, batch_space), seed=42)
        >>> "lr" in params
        True
        >>> "batch_size" in params
        True
        >>> 0.0 <= params["lr"] <= 1.0
        True

        >>> sample_parameters(())
        Traceback (most recent call last):
            ...
        ValueError: spaces cannot be empty
    """
    if not spaces:
        msg = "spaces cannot be empty"
        raise ValueError(msg)

    if seed is not None:
        random.seed(seed)

    params: dict[str, float | int | str] = {}

    for space in spaces:
        if space.param_type == ParameterType.CONTINUOUS:
            assert space.low is not None and space.high is not None
            params[space.name] = random.uniform(space.low, space.high)
        elif space.param_type == ParameterType.DISCRETE:
            assert space.low is not None and space.high is not None
            params[space.name] = random.randint(int(space.low), int(space.high))
        elif space.param_type == ParameterType.CATEGORICAL:
            assert space.choices is not None
            params[space.name] = random.choice(space.choices)
        elif space.param_type == ParameterType.LOG_UNIFORM:
            assert space.low is not None and space.high is not None
            log_low = math.log(space.low)
            log_high = math.log(space.high)
            params[space.name] = math.exp(random.uniform(log_low, log_high))

    return params


def calculate_improvement(
    current_value: float,
    best_value: float,
    direction: ObjectiveDirection,
) -> float:
    """Calculate improvement of current value over best value.

    Args:
        current_value: The current objective value.
        best_value: The previous best objective value.
        direction: Direction of optimization.

    Returns:
        Improvement ratio (positive if improved, negative if worse).

    Examples:
        >>> calculate_improvement(0.05, 0.1, ObjectiveDirection.MINIMIZE)
        0.5
        >>> calculate_improvement(0.15, 0.1, ObjectiveDirection.MINIMIZE)
        -0.5
        >>> calculate_improvement(0.9, 0.8, ObjectiveDirection.MAXIMIZE)
        0.125
        >>> calculate_improvement(0.7, 0.8, ObjectiveDirection.MAXIMIZE)
        -0.125

        >>> calculate_improvement(0.5, 0.0, ObjectiveDirection.MINIMIZE)
        0.0
    """
    if best_value == 0:
        return 0.0

    if direction == ObjectiveDirection.MINIMIZE:
        return (best_value - current_value) / abs(best_value)
    else:
        return (current_value - best_value) / abs(best_value)


def estimate_remaining_trials(
    elapsed_seconds: float,
    completed_trials: int,
    total_trials: int,
    timeout_seconds: int,
) -> int:
    """Estimate the number of remaining trials before timeout.

    Args:
        elapsed_seconds: Time elapsed so far.
        completed_trials: Number of trials completed.
        total_trials: Total number of planned trials.
        timeout_seconds: Maximum time for optimization.

    Returns:
        Estimated number of remaining trials.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> estimate_remaining_trials(100.0, 10, 100, 3600)
        90
        >>> estimate_remaining_trials(1800.0, 50, 100, 3600)
        50
        >>> estimate_remaining_trials(3500.0, 90, 100, 3600)
        10

        >>> estimate_remaining_trials(-1.0, 10, 100, 3600)
        Traceback (most recent call last):
            ...
        ValueError: elapsed_seconds must be non-negative, got -1.0
    """
    if elapsed_seconds < 0:
        msg = f"elapsed_seconds must be non-negative, got {elapsed_seconds}"
        raise ValueError(msg)
    if completed_trials < 0:
        msg = f"completed_trials must be non-negative, got {completed_trials}"
        raise ValueError(msg)
    if total_trials <= 0:
        msg = f"total_trials must be positive, got {total_trials}"
        raise ValueError(msg)
    if timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {timeout_seconds}"
        raise ValueError(msg)

    remaining_by_count = total_trials - completed_trials

    if completed_trials == 0:
        return remaining_by_count

    avg_time_per_trial = elapsed_seconds / completed_trials
    remaining_time = timeout_seconds - elapsed_seconds

    if remaining_time <= 0:
        return 0

    remaining_by_time = int(remaining_time / avg_time_per_trial)

    return min(remaining_by_count, remaining_by_time)


def suggest_next_params(
    config: HyperoptConfig,
    history: tuple[TrialResult, ...],
    seed: int | None = None,
) -> dict[str, float | int | str]:
    """Suggest the next set of parameters to try.

    This function implements a simplified version of each algorithm.
    For production use, consider using Optuna or similar libraries.

    Args:
        config: Hyperparameter optimization configuration.
        history: Tuple of previous trial results.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of suggested parameter values.

    Examples:
        >>> lr_space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        >>> config = create_hyperopt_config(
        ...     algorithm="random",
        ...     parameter_spaces=(lr_space,),
        ... )
        >>> params = suggest_next_params(config, (), seed=42)
        >>> "lr" in params
        True
        >>> 1e-5 <= params["lr"] <= 1e-1
        True
    """
    if seed is not None:
        random.seed(seed)

    algorithm = config.algorithm
    spaces = config.parameter_spaces

    if algorithm == SearchAlgorithm.GRID:
        # For grid search, use structured sampling
        return _grid_sample(spaces, len(history))
    elif algorithm == SearchAlgorithm.RANDOM:
        return sample_parameters(spaces, seed)
    elif algorithm in (
        SearchAlgorithm.BAYESIAN,
        SearchAlgorithm.TPE,
        SearchAlgorithm.CMAES,
    ):
        # For advanced algorithms, fall back to exploration + exploitation
        return _bayesian_sample(spaces, history, config.direction, seed)
    else:  # HYPERBAND
        return sample_parameters(spaces, seed)


def _grid_sample(
    spaces: tuple[ParameterSpace, ...],
    trial_idx: int,
) -> dict[str, float | int | str]:
    """Sample parameters using grid search.

    Args:
        spaces: Parameter spaces to sample from.
        trial_idx: Current trial index.

    Returns:
        Dictionary of parameter values.
    """
    params: dict[str, float | int | str] = {}
    n_points = 10  # Number of grid points per dimension

    for i, space in enumerate(spaces):
        # Calculate index for this parameter
        idx = (trial_idx // (n_points**i)) % n_points

        if space.param_type == ParameterType.CONTINUOUS:
            assert space.low is not None and space.high is not None
            step = (space.high - space.low) / (n_points - 1)
            params[space.name] = space.low + idx * step
        elif space.param_type == ParameterType.DISCRETE:
            assert space.low is not None and space.high is not None
            step = (int(space.high) - int(space.low)) // (n_points - 1)
            params[space.name] = int(space.low) + idx * step
        elif space.param_type == ParameterType.CATEGORICAL:
            assert space.choices is not None
            params[space.name] = space.choices[idx % len(space.choices)]
        elif space.param_type == ParameterType.LOG_UNIFORM:
            assert space.low is not None and space.high is not None
            log_low = math.log(space.low)
            log_high = math.log(space.high)
            step = (log_high - log_low) / (n_points - 1)
            params[space.name] = math.exp(log_low + idx * step)

    return params


def _bayesian_sample(
    spaces: tuple[ParameterSpace, ...],
    history: tuple[TrialResult, ...],
    direction: ObjectiveDirection,
    seed: int | None = None,
) -> dict[str, float | int | str]:
    """Sample parameters using simplified Bayesian-inspired approach.

    This is a simplified implementation that:
    1. With 20% probability, explores randomly
    2. Otherwise, samples near the best known parameters

    Args:
        spaces: Parameter spaces to sample from.
        history: Previous trial results.
        direction: Optimization direction.
        seed: Random seed.

    Returns:
        Dictionary of parameter values.
    """
    if seed is not None:
        random.seed(seed)

    # If no history, sample randomly
    if not history:
        return sample_parameters(spaces, seed)

    # Find best trial
    if direction == ObjectiveDirection.MINIMIZE:
        best_trial = min(history, key=lambda t: t.objective_value)
    else:
        best_trial = max(history, key=lambda t: t.objective_value)

    # 20% exploration, 80% exploitation
    if random.random() < 0.2:
        return sample_parameters(spaces, seed)

    # Sample near best parameters
    params: dict[str, float | int | str] = {}
    noise_scale = 0.1  # Scale of perturbation

    for space in spaces:
        best_val = best_trial.params.get(space.name)

        if best_val is None or space.param_type == ParameterType.CATEGORICAL:
            # Fall back to random sampling
            if space.param_type == ParameterType.CATEGORICAL:
                assert space.choices is not None
                params[space.name] = random.choice(space.choices)
            else:
                sample = sample_parameters((space,), seed)
                params[space.name] = sample[space.name]
        else:
            assert space.low is not None and space.high is not None
            val = float(best_val)

            if space.param_type == ParameterType.LOG_UNIFORM:
                log_val = math.log(val)
                log_range = math.log(space.high) - math.log(space.low)
                perturbation = random.gauss(0, noise_scale * log_range)
                new_log_val = log_val + perturbation
                new_val = math.exp(
                    max(math.log(space.low), min(math.log(space.high), new_log_val))
                )
                params[space.name] = new_val
            elif space.param_type == ParameterType.DISCRETE:
                val_range = space.high - space.low
                perturbation = random.gauss(0, noise_scale * val_range)
                new_val = int(val + perturbation)
                params[space.name] = max(int(space.low), min(int(space.high), new_val))
            else:  # CONTINUOUS
                val_range = space.high - space.low
                perturbation = random.gauss(0, noise_scale * val_range)
                new_val = val + perturbation
                params[space.name] = max(space.low, min(space.high, new_val))

    return params


def format_hyperopt_stats(stats: HyperoptStats) -> str:
    """Format hyperparameter optimization statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_hyperopt_stats(
        ...     total_trials=50,
        ...     best_value=0.02,
        ...     best_params={"lr": 0.0005, "batch_size": 64},
        ...     convergence_curve=(0.1, 0.05, 0.02),
        ... )
        >>> formatted = format_hyperopt_stats(stats)
        >>> "Trials: 50" in formatted
        True
        >>> "Best Value: 0.0200" in formatted
        True
        >>> "lr" in formatted
        True
    """
    params_str = ", ".join(
        f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(stats.best_params.items())
    )

    curve_str = ""
    if stats.convergence_curve:
        curve_values = [f"{v:.4f}" for v in stats.convergence_curve[-5:]]
        curve_str = f"\n  Convergence (last 5): [{', '.join(curve_values)}]"

    return (
        f"Hyperopt Stats:\n"
        f"  Trials: {stats.total_trials}\n"
        f"  Best Value: {stats.best_value:.4f}\n"
        f"  Best Params: {{{params_str}}}{curve_str}"
    )


def get_recommended_hyperopt_config(task_type: str) -> HyperoptConfig:
    """Get recommended hyperparameter optimization configuration for a task type.

    Args:
        task_type: Type of task (classification, generation, fine_tuning,
            pretraining, rl).

    Returns:
        Recommended HyperoptConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_hyperopt_config("classification")
        >>> config.algorithm
        <SearchAlgorithm.TPE: 'tpe'>
        >>> len(config.parameter_spaces) > 0
        True

        >>> config = get_recommended_hyperopt_config("fine_tuning")
        >>> config.algorithm
        <SearchAlgorithm.BAYESIAN: 'bayesian'>

        >>> get_recommended_hyperopt_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_type must be one of ...
    """
    valid_tasks = frozenset(
        {
            "classification",
            "generation",
            "fine_tuning",
            "pretraining",
            "rl",
        }
    )

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    if task_type == "classification":
        return create_hyperopt_config(
            algorithm=SearchAlgorithm.TPE,
            parameter_spaces=(
                create_parameter_space(
                    "learning_rate", ParameterType.LOG_UNIFORM, 1e-5, 1e-2
                ),
                create_parameter_space("batch_size", ParameterType.DISCRETE, 16, 128),
                create_parameter_space(
                    "weight_decay", ParameterType.LOG_UNIFORM, 1e-6, 1e-2
                ),
                create_parameter_space(
                    "warmup_ratio", ParameterType.CONTINUOUS, 0.0, 0.2
                ),
            ),
            n_trials=50,
            timeout_seconds=7200,
            direction=ObjectiveDirection.MAXIMIZE,
        )
    elif task_type == "generation":
        return create_hyperopt_config(
            algorithm=SearchAlgorithm.TPE,
            parameter_spaces=(
                create_parameter_space(
                    "learning_rate", ParameterType.LOG_UNIFORM, 1e-6, 1e-3
                ),
                create_parameter_space("batch_size", ParameterType.DISCRETE, 4, 64),
                create_parameter_space(
                    "gradient_accumulation", ParameterType.DISCRETE, 1, 16
                ),
                create_parameter_space(
                    "max_grad_norm", ParameterType.CONTINUOUS, 0.5, 2.0
                ),
            ),
            n_trials=100,
            timeout_seconds=14400,
            direction=ObjectiveDirection.MINIMIZE,
        )
    elif task_type == "fine_tuning":
        return create_hyperopt_config(
            algorithm=SearchAlgorithm.BAYESIAN,
            parameter_spaces=(
                create_parameter_space(
                    "learning_rate", ParameterType.LOG_UNIFORM, 1e-6, 5e-4
                ),
                create_parameter_space("lora_r", ParameterType.DISCRETE, 4, 64),
                create_parameter_space("lora_alpha", ParameterType.DISCRETE, 8, 128),
                create_parameter_space(
                    "lora_dropout", ParameterType.CONTINUOUS, 0.0, 0.3
                ),
            ),
            n_trials=30,
            timeout_seconds=3600,
            direction=ObjectiveDirection.MINIMIZE,
        )
    elif task_type == "pretraining":
        return create_hyperopt_config(
            algorithm=SearchAlgorithm.HYPERBAND,
            parameter_spaces=(
                create_parameter_space(
                    "learning_rate", ParameterType.LOG_UNIFORM, 1e-5, 1e-3
                ),
                create_parameter_space("batch_size", ParameterType.DISCRETE, 32, 512),
                create_parameter_space(
                    "warmup_steps", ParameterType.DISCRETE, 100, 2000
                ),
                create_parameter_space(
                    "weight_decay", ParameterType.LOG_UNIFORM, 1e-5, 1e-1
                ),
            ),
            n_trials=20,
            timeout_seconds=28800,
            direction=ObjectiveDirection.MINIMIZE,
        )
    else:  # rl
        return create_hyperopt_config(
            algorithm=SearchAlgorithm.CMAES,
            parameter_spaces=(
                create_parameter_space(
                    "learning_rate", ParameterType.LOG_UNIFORM, 1e-6, 1e-3
                ),
                create_parameter_space("ppo_epochs", ParameterType.DISCRETE, 1, 8),
                create_parameter_space("gamma", ParameterType.CONTINUOUS, 0.9, 0.999),
                create_parameter_space(
                    "clip_range", ParameterType.CONTINUOUS, 0.1, 0.4
                ),
            ),
            n_trials=40,
            timeout_seconds=7200,
            direction=ObjectiveDirection.MAXIMIZE,
        )
