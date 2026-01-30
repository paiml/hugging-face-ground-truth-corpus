"""Multi-Task Learning utilities for training.

This module provides functions for configuring and implementing multi-task
learning strategies, enabling training models on multiple tasks simultaneously
with gradient balancing and conflict resolution techniques.

Examples:
    >>> from hf_gtc.training.multitask import (
    ...     create_multitask_config,
    ...     TaskBalancing,
    ... )
    >>> config = create_multitask_config()
    >>> config.balancing_method
    <TaskBalancing.UNIFORM: 'uniform'>
    >>> len(config.tasks)
    0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TaskBalancing(Enum):
    """Methods for balancing task weights during training.

    Attributes:
        UNIFORM: Equal weights for all tasks.
        UNCERTAINTY: Weight by task uncertainty (homoscedastic).
        GRADNORM: Adaptive weighting via GradNorm algorithm.
        PCGRAD: Project Conflicting Gradients.
        CAGRAD: Conflict-Averse Gradient descent.

    Examples:
        >>> TaskBalancing.UNIFORM.value
        'uniform'
        >>> TaskBalancing.UNCERTAINTY.value
        'uncertainty'
        >>> TaskBalancing.GRADNORM.value
        'gradnorm'
    """

    UNIFORM = "uniform"
    UNCERTAINTY = "uncertainty"
    GRADNORM = "gradnorm"
    PCGRAD = "pcgrad"
    CAGRAD = "cagrad"


class GradientStrategy(Enum):
    """Strategies for combining gradients from multiple tasks.

    Attributes:
        SUM: Simple sum of task gradients.
        PCGRAD: Project conflicting gradients.
        CAGRAD: Conflict-averse gradient descent.
        MGDA: Multiple Gradient Descent Algorithm.
        NASH: Nash equilibrium-based gradient combination.

    Examples:
        >>> GradientStrategy.SUM.value
        'sum'
        >>> GradientStrategy.PCGRAD.value
        'pcgrad'
        >>> GradientStrategy.MGDA.value
        'mgda'
    """

    SUM = "sum"
    PCGRAD = "pcgrad"
    CAGRAD = "cagrad"
    MGDA = "mgda"
    NASH = "nash"


class TaskRelation(Enum):
    """Types of relationships between tasks.

    Attributes:
        INDEPENDENT: Tasks are trained independently.
        AUXILIARY: Some tasks help others (auxiliary learning).
        HIERARCHICAL: Tasks are organized hierarchically.

    Examples:
        >>> TaskRelation.INDEPENDENT.value
        'independent'
        >>> TaskRelation.AUXILIARY.value
        'auxiliary'
        >>> TaskRelation.HIERARCHICAL.value
        'hierarchical'
    """

    INDEPENDENT = "independent"
    AUXILIARY = "auxiliary"
    HIERARCHICAL = "hierarchical"


VALID_TASK_BALANCING = frozenset(b.value for b in TaskBalancing)
VALID_GRADIENT_STRATEGIES = frozenset(s.value for s in GradientStrategy)
VALID_TASK_RELATIONS = frozenset(r.value for r in TaskRelation)


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Configuration for a single task in multi-task learning.

    Attributes:
        name: Unique identifier for the task.
        weight: Initial weight for this task (0 to infinity).
        loss_type: Type of loss function for this task.
        relation: Relationship to other tasks.

    Examples:
        >>> config = TaskConfig(
        ...     name="classification",
        ...     weight=1.0,
        ...     loss_type="cross_entropy",
        ...     relation=TaskRelation.INDEPENDENT,
        ... )
        >>> config.name
        'classification'
        >>> config.weight
        1.0
    """

    name: str
    weight: float
    loss_type: str
    relation: TaskRelation


@dataclass(frozen=True, slots=True)
class GradientConfig:
    """Configuration for gradient handling in multi-task learning.

    Attributes:
        strategy: Strategy for combining task gradients.
        conflict_threshold: Threshold for detecting gradient conflicts.
        normalize: Whether to normalize gradients before combining.

    Examples:
        >>> config = GradientConfig(
        ...     strategy=GradientStrategy.SUM,
        ...     conflict_threshold=0.0,
        ...     normalize=False,
        ... )
        >>> config.strategy
        <GradientStrategy.SUM: 'sum'>
        >>> config.conflict_threshold
        0.0
    """

    strategy: GradientStrategy
    conflict_threshold: float
    normalize: bool


@dataclass(frozen=True, slots=True)
class MultiTaskConfig:
    """Main configuration for multi-task learning.

    Attributes:
        tasks: Tuple of task configurations.
        gradient_config: Configuration for gradient handling.
        balancing_method: Method for balancing task weights.
        shared_layers: Tuple of layer names that are shared across tasks.

    Examples:
        >>> gradient_config = GradientConfig(
        ...     strategy=GradientStrategy.SUM,
        ...     conflict_threshold=0.0,
        ...     normalize=False,
        ... )
        >>> config = MultiTaskConfig(
        ...     tasks=(),
        ...     gradient_config=gradient_config,
        ...     balancing_method=TaskBalancing.UNIFORM,
        ...     shared_layers=("encoder",),
        ... )
        >>> config.balancing_method
        <TaskBalancing.UNIFORM: 'uniform'>
    """

    tasks: tuple[TaskConfig, ...]
    gradient_config: GradientConfig
    balancing_method: TaskBalancing
    shared_layers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultiTaskStats:
    """Statistics from multi-task learning training.

    Attributes:
        task_losses: Mapping of task names to their losses.
        task_accuracies: Mapping of task names to their accuracies.
        gradient_conflicts: Number of gradient conflicts detected.
        effective_weights: Current effective weights per task.

    Examples:
        >>> stats = MultiTaskStats(
        ...     task_losses={"task1": 0.5, "task2": 0.3},
        ...     task_accuracies={"task1": 0.9, "task2": 0.85},
        ...     gradient_conflicts=5,
        ...     effective_weights={"task1": 1.0, "task2": 1.2},
        ... )
        >>> stats.task_losses["task1"]
        0.5
        >>> stats.gradient_conflicts
        5
    """

    task_losses: dict[str, float]
    task_accuracies: dict[str, float]
    gradient_conflicts: int
    effective_weights: dict[str, float]


def validate_task_config(config: TaskConfig) -> None:
    """Validate task configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = TaskConfig(
        ...     name="classification",
        ...     weight=1.0,
        ...     loss_type="cross_entropy",
        ...     relation=TaskRelation.INDEPENDENT,
        ... )
        >>> validate_task_config(config)

        >>> bad_config = TaskConfig(
        ...     name="",
        ...     weight=1.0,
        ...     loss_type="cross_entropy",
        ...     relation=TaskRelation.INDEPENDENT,
        ... )
        >>> validate_task_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: task name cannot be empty
    """
    if not config.name or not config.name.strip():
        msg = "task name cannot be empty"
        raise ValueError(msg)
    if config.weight < 0:
        msg = f"weight must be non-negative, got {config.weight}"
        raise ValueError(msg)
    if not config.loss_type or not config.loss_type.strip():
        msg = "loss_type cannot be empty"
        raise ValueError(msg)


def validate_gradient_config(config: GradientConfig) -> None:
    """Validate gradient configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = GradientConfig(
        ...     strategy=GradientStrategy.SUM,
        ...     conflict_threshold=0.0,
        ...     normalize=False,
        ... )
        >>> validate_gradient_config(config)

        >>> bad_config = GradientConfig(
        ...     strategy=GradientStrategy.PCGRAD,
        ...     conflict_threshold=-0.5,
        ...     normalize=False,
        ... )
        >>> validate_gradient_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: conflict_threshold must be non-negative, got -0.5
    """
    if config.conflict_threshold < 0:
        msg = (
            f"conflict_threshold must be non-negative, "
            f"got {config.conflict_threshold}"
        )
        raise ValueError(msg)


def validate_multitask_config(config: MultiTaskConfig) -> None:
    """Validate multi-task configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = create_multitask_config()
        >>> validate_multitask_config(config)

        >>> task = TaskConfig(
        ...     name="",
        ...     weight=1.0,
        ...     loss_type="ce",
        ...     relation=TaskRelation.INDEPENDENT,
        ... )
        >>> gradient_config = GradientConfig(
        ...     strategy=GradientStrategy.SUM,
        ...     conflict_threshold=0.0,
        ...     normalize=False,
        ... )
        >>> bad_config = MultiTaskConfig(
        ...     tasks=(task,),
        ...     gradient_config=gradient_config,
        ...     balancing_method=TaskBalancing.UNIFORM,
        ...     shared_layers=(),
        ... )
        >>> validate_multitask_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: task name cannot be empty
    """
    for task in config.tasks:
        validate_task_config(task)
    validate_gradient_config(config.gradient_config)

    # Check for duplicate task names
    task_names = [task.name for task in config.tasks]
    if len(task_names) != len(set(task_names)):
        msg = "duplicate task names found"
        raise ValueError(msg)


def validate_multitask_stats(stats: MultiTaskStats) -> None:
    """Validate multi-task statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any statistic value is invalid.

    Examples:
        >>> stats = MultiTaskStats(
        ...     task_losses={"task1": 0.5},
        ...     task_accuracies={"task1": 0.9},
        ...     gradient_conflicts=5,
        ...     effective_weights={"task1": 1.0},
        ... )
        >>> validate_multitask_stats(stats)

        >>> bad_stats = MultiTaskStats(
        ...     task_losses={"task1": -0.5},
        ...     task_accuracies={"task1": 0.9},
        ...     gradient_conflicts=5,
        ...     effective_weights={"task1": 1.0},
        ... )
        >>> validate_multitask_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: task losses must be non-negative
    """
    for loss in stats.task_losses.values():
        if loss < 0:
            msg = "task losses must be non-negative"
            raise ValueError(msg)
    for acc in stats.task_accuracies.values():
        if not 0 <= acc <= 1:
            msg = "task accuracies must be between 0 and 1"
            raise ValueError(msg)
    if stats.gradient_conflicts < 0:
        msg = f"gradient_conflicts must be non-negative, got {stats.gradient_conflicts}"
        raise ValueError(msg)
    for weight in stats.effective_weights.values():
        if weight < 0:
            msg = "effective weights must be non-negative"
            raise ValueError(msg)


def create_task_config(
    name: str,
    weight: float = 1.0,
    loss_type: str = "cross_entropy",
    relation: str | TaskRelation = TaskRelation.INDEPENDENT,
) -> TaskConfig:
    """Create a task configuration with validation.

    Args:
        name: Unique identifier for the task.
        weight: Initial weight for this task.
        loss_type: Type of loss function for this task.
        relation: Relationship to other tasks.

    Returns:
        Validated TaskConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_task_config("classification")
        >>> config.name
        'classification'
        >>> config.weight
        1.0

        >>> config = create_task_config(
        ...     "auxiliary_task",
        ...     weight=0.5,
        ...     relation="auxiliary",
        ... )
        >>> config.relation
        <TaskRelation.AUXILIARY: 'auxiliary'>

        >>> create_task_config("")
        Traceback (most recent call last):
            ...
        ValueError: task name cannot be empty
    """
    if isinstance(relation, str):
        relation = get_task_relation(relation)

    config = TaskConfig(
        name=name,
        weight=weight,
        loss_type=loss_type,
        relation=relation,
    )
    validate_task_config(config)
    return config


def create_gradient_config(
    strategy: str | GradientStrategy = GradientStrategy.SUM,
    conflict_threshold: float = 0.0,
    normalize: bool = False,
) -> GradientConfig:
    """Create a gradient configuration with validation.

    Args:
        strategy: Strategy for combining task gradients.
        conflict_threshold: Threshold for detecting gradient conflicts.
        normalize: Whether to normalize gradients before combining.

    Returns:
        Validated GradientConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_gradient_config()
        >>> config.strategy
        <GradientStrategy.SUM: 'sum'>
        >>> config.conflict_threshold
        0.0

        >>> config = create_gradient_config(strategy="pcgrad", normalize=True)
        >>> config.strategy
        <GradientStrategy.PCGRAD: 'pcgrad'>
        >>> config.normalize
        True

        >>> create_gradient_config(conflict_threshold=-1.0)
        Traceback (most recent call last):
            ...
        ValueError: conflict_threshold must be non-negative, got -1.0
    """
    if isinstance(strategy, str):
        strategy = get_gradient_strategy(strategy)

    config = GradientConfig(
        strategy=strategy,
        conflict_threshold=conflict_threshold,
        normalize=normalize,
    )
    validate_gradient_config(config)
    return config


def create_multitask_config(
    tasks: tuple[TaskConfig, ...] | None = None,
    strategy: str | GradientStrategy = GradientStrategy.SUM,
    conflict_threshold: float = 0.0,
    normalize: bool = False,
    balancing_method: str | TaskBalancing = TaskBalancing.UNIFORM,
    shared_layers: tuple[str, ...] = (),
) -> MultiTaskConfig:
    """Create a multi-task configuration with validation.

    Args:
        tasks: Tuple of task configurations.
        strategy: Strategy for combining task gradients.
        conflict_threshold: Threshold for detecting gradient conflicts.
        normalize: Whether to normalize gradients before combining.
        balancing_method: Method for balancing task weights.
        shared_layers: Tuple of layer names that are shared across tasks.

    Returns:
        Validated MultiTaskConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_multitask_config()
        >>> config.balancing_method
        <TaskBalancing.UNIFORM: 'uniform'>
        >>> len(config.tasks)
        0

        >>> task = create_task_config("classification")
        >>> config = create_multitask_config(
        ...     tasks=(task,),
        ...     strategy="pcgrad",
        ...     balancing_method="uncertainty",
        ... )
        >>> config.gradient_config.strategy
        <GradientStrategy.PCGRAD: 'pcgrad'>
        >>> config.balancing_method
        <TaskBalancing.UNCERTAINTY: 'uncertainty'>

        >>> create_multitask_config(conflict_threshold=-0.5)
        Traceback (most recent call last):
            ...
        ValueError: conflict_threshold must be non-negative, got -0.5
    """
    if tasks is None:
        tasks = ()

    gradient_config = create_gradient_config(
        strategy=strategy,
        conflict_threshold=conflict_threshold,
        normalize=normalize,
    )

    if isinstance(balancing_method, str):
        balancing_method = get_task_balancing(balancing_method)

    config = MultiTaskConfig(
        tasks=tasks,
        gradient_config=gradient_config,
        balancing_method=balancing_method,
        shared_layers=shared_layers,
    )
    validate_multitask_config(config)
    return config


def create_multitask_stats(
    task_losses: dict[str, float] | None = None,
    task_accuracies: dict[str, float] | None = None,
    gradient_conflicts: int = 0,
    effective_weights: dict[str, float] | None = None,
) -> MultiTaskStats:
    """Create multi-task statistics with validation.

    Args:
        task_losses: Mapping of task names to their losses.
        task_accuracies: Mapping of task names to their accuracies.
        gradient_conflicts: Number of gradient conflicts detected.
        effective_weights: Current effective weights per task.

    Returns:
        Validated MultiTaskStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_multitask_stats()
        >>> stats.gradient_conflicts
        0
        >>> len(stats.task_losses)
        0

        >>> stats = create_multitask_stats(
        ...     task_losses={"task1": 0.5},
        ...     task_accuracies={"task1": 0.9},
        ...     effective_weights={"task1": 1.0},
        ... )
        >>> stats.task_losses["task1"]
        0.5

        >>> create_multitask_stats(gradient_conflicts=-1)
        Traceback (most recent call last):
            ...
        ValueError: gradient_conflicts must be non-negative, got -1
    """
    if task_losses is None:
        task_losses = {}
    if task_accuracies is None:
        task_accuracies = {}
    if effective_weights is None:
        effective_weights = {}

    stats = MultiTaskStats(
        task_losses=task_losses,
        task_accuracies=task_accuracies,
        gradient_conflicts=gradient_conflicts,
        effective_weights=effective_weights,
    )
    validate_multitask_stats(stats)
    return stats


def list_task_balancing_methods() -> list[str]:
    """List all available task balancing methods.

    Returns:
        Sorted list of task balancing method names.

    Examples:
        >>> methods = list_task_balancing_methods()
        >>> "uniform" in methods
        True
        >>> "uncertainty" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_TASK_BALANCING)


def list_gradient_strategies() -> list[str]:
    """List all available gradient strategies.

    Returns:
        Sorted list of gradient strategy names.

    Examples:
        >>> strategies = list_gradient_strategies()
        >>> "sum" in strategies
        True
        >>> "pcgrad" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_GRADIENT_STRATEGIES)


def list_task_relations() -> list[str]:
    """List all available task relations.

    Returns:
        Sorted list of task relation names.

    Examples:
        >>> relations = list_task_relations()
        >>> "independent" in relations
        True
        >>> "auxiliary" in relations
        True
        >>> relations == sorted(relations)
        True
    """
    return sorted(VALID_TASK_RELATIONS)


def get_task_balancing(name: str) -> TaskBalancing:
    """Get task balancing enum from string name.

    Args:
        name: Name of the task balancing method.

    Returns:
        Corresponding TaskBalancing enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_task_balancing("uniform")
        <TaskBalancing.UNIFORM: 'uniform'>
        >>> get_task_balancing("uncertainty")
        <TaskBalancing.UNCERTAINTY: 'uncertainty'>

        >>> get_task_balancing("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: balancing_method must be one of ...
    """
    if name not in VALID_TASK_BALANCING:
        msg = f"balancing_method must be one of {VALID_TASK_BALANCING}, got '{name}'"
        raise ValueError(msg)
    return TaskBalancing(name)


def get_gradient_strategy(name: str) -> GradientStrategy:
    """Get gradient strategy enum from string name.

    Args:
        name: Name of the gradient strategy.

    Returns:
        Corresponding GradientStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_gradient_strategy("sum")
        <GradientStrategy.SUM: 'sum'>
        >>> get_gradient_strategy("pcgrad")
        <GradientStrategy.PCGRAD: 'pcgrad'>

        >>> get_gradient_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: gradient_strategy must be one of ...
    """
    if name not in VALID_GRADIENT_STRATEGIES:
        msg = (
            f"gradient_strategy must be one of {VALID_GRADIENT_STRATEGIES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return GradientStrategy(name)


def get_task_relation(name: str) -> TaskRelation:
    """Get task relation enum from string name.

    Args:
        name: Name of the task relation.

    Returns:
        Corresponding TaskRelation enum.

    Raises:
        ValueError: If relation name is invalid.

    Examples:
        >>> get_task_relation("independent")
        <TaskRelation.INDEPENDENT: 'independent'>
        >>> get_task_relation("auxiliary")
        <TaskRelation.AUXILIARY: 'auxiliary'>

        >>> get_task_relation("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_relation must be one of ...
    """
    if name not in VALID_TASK_RELATIONS:
        msg = f"task_relation must be one of {VALID_TASK_RELATIONS}, got '{name}'"
        raise ValueError(msg)
    return TaskRelation(name)


def calculate_task_weights(
    losses: tuple[float, ...],
    method: TaskBalancing,
    uncertainties: tuple[float, ...] | None = None,
    initial_losses: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Calculate task weights based on balancing method.

    Args:
        losses: Current losses for each task.
        method: Balancing method to use.
        uncertainties: Task uncertainties (for uncertainty weighting).
        initial_losses: Initial losses (for GradNorm).

    Returns:
        Tuple of weights for each task.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> calculate_task_weights((0.5, 0.3), TaskBalancing.UNIFORM)
        (1.0, 1.0)

        >>> weights = calculate_task_weights(
        ...     (0.5, 0.3),
        ...     TaskBalancing.UNCERTAINTY,
        ...     uncertainties=(0.1, 0.2),
        ... )
        >>> weights[0] > weights[1]
        True

        >>> calculate_task_weights((), TaskBalancing.UNIFORM)
        Traceback (most recent call last):
            ...
        ValueError: losses cannot be empty
    """
    if not losses:
        msg = "losses cannot be empty"
        raise ValueError(msg)

    for loss in losses:
        if loss < 0:
            msg = "losses must be non-negative"
            raise ValueError(msg)

    n_tasks = len(losses)

    if method == TaskBalancing.UNIFORM:
        return tuple(1.0 for _ in range(n_tasks))

    elif method == TaskBalancing.UNCERTAINTY:
        if uncertainties is None:
            msg = "uncertainties required for uncertainty weighting"
            raise ValueError(msg)
        if len(uncertainties) != n_tasks:
            msg = "uncertainties must match number of tasks"
            raise ValueError(msg)
        for u in uncertainties:
            if u <= 0:
                msg = "uncertainties must be positive"
                raise ValueError(msg)
        # Weight inversely by variance (1/sigma^2)
        return tuple(1.0 / (u * u) for u in uncertainties)

    elif method == TaskBalancing.GRADNORM:
        if initial_losses is None:
            msg = "initial_losses required for gradnorm weighting"
            raise ValueError(msg)
        if len(initial_losses) != n_tasks:
            msg = "initial_losses must match number of tasks"
            raise ValueError(msg)
        # Compute relative loss changes
        ratios = []
        for loss, init_loss in zip(losses, initial_losses, strict=True):
            if init_loss <= 0:
                ratios.append(1.0)
            else:
                ratios.append(loss / init_loss)
        # Normalize so weights sum to n_tasks
        mean_ratio = sum(ratios) / n_tasks
        if mean_ratio > 0:
            return tuple(r / mean_ratio for r in ratios)
        return tuple(1.0 for _ in range(n_tasks))

    elif method in (TaskBalancing.PCGRAD, TaskBalancing.CAGRAD):
        # PCGrad and CAGrad don't modify weights, they modify gradients
        return tuple(1.0 for _ in range(n_tasks))

    return tuple(1.0 for _ in range(n_tasks))


def detect_gradient_conflicts(
    gradients: tuple[tuple[float, ...], ...],
    threshold: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    """Detect pairs of tasks with conflicting gradients.

    Two gradients conflict if their cosine similarity is below the threshold.

    Args:
        gradients: Tuple of gradient vectors for each task.
        threshold: Cosine similarity threshold for conflict.

    Returns:
        Tuple of (task_i, task_j) pairs with conflicting gradients.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> # Opposite gradients
        >>> conflicts = detect_gradient_conflicts(
        ...     ((1.0, 0.0), (-1.0, 0.0)),
        ...     threshold=0.0,
        ... )
        >>> (0, 1) in conflicts
        True

        >>> # Same direction gradients
        >>> conflicts = detect_gradient_conflicts(
        ...     ((1.0, 0.0), (1.0, 0.0)),
        ...     threshold=0.0,
        ... )
        >>> len(conflicts)
        0

        >>> detect_gradient_conflicts((), threshold=0.0)
        Traceback (most recent call last):
            ...
        ValueError: gradients cannot be empty
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if threshold < -1 or threshold > 1:
        msg = f"threshold must be between -1 and 1, got {threshold}"
        raise ValueError(msg)

    n_tasks = len(gradients)
    if n_tasks < 2:
        return ()

    grad_dim = len(gradients[0])
    for g in gradients:
        if len(g) != grad_dim:
            msg = "all gradients must have the same dimension"
            raise ValueError(msg)

    def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    conflicts = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            sim = cosine_similarity(gradients[i], gradients[j])
            if sim < threshold:
                conflicts.append((i, j))

    return tuple(conflicts)


def project_conflicting_gradients(
    gradients: tuple[tuple[float, ...], ...],
    strategy: GradientStrategy = GradientStrategy.PCGRAD,
) -> tuple[float, ...]:
    """Project gradients to resolve conflicts between tasks.

    Args:
        gradients: Tuple of gradient vectors for each task.
        strategy: Gradient projection strategy.

    Returns:
        Combined gradient vector after projection.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> # Opposite gradients should cancel
        >>> result = project_conflicting_gradients(
        ...     ((1.0, 0.0), (-1.0, 0.0)),
        ...     strategy=GradientStrategy.SUM,
        ... )
        >>> abs(result[0]) < 1e-10
        True

        >>> # Same direction gradients should add
        >>> result = project_conflicting_gradients(
        ...     ((1.0, 0.0), (2.0, 0.0)),
        ...     strategy=GradientStrategy.SUM,
        ... )
        >>> result[0]
        3.0

        >>> project_conflicting_gradients(())
        Traceback (most recent call last):
            ...
        ValueError: gradients cannot be empty
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if not gradients[0]:
        msg = "gradients cannot have zero dimension"
        raise ValueError(msg)

    grad_dim = len(gradients[0])
    for g in gradients:
        if len(g) != grad_dim:
            msg = "all gradients must have the same dimension"
            raise ValueError(msg)

    if strategy == GradientStrategy.SUM:
        # Simple sum
        result = [0.0] * grad_dim
        for g in gradients:
            for i, val in enumerate(g):
                result[i] += val
        return tuple(result)

    elif strategy == GradientStrategy.PCGRAD:
        # Project Conflicting Gradients
        projected = [list(g) for g in gradients]

        for i in range(len(gradients)):
            for j in range(len(gradients)):
                if i == j:
                    continue
                # Compute dot product
                dot = sum(
                    projected[i][k] * gradients[j][k] for k in range(grad_dim)
                )
                norm_sq = sum(gradients[j][k] ** 2 for k in range(grad_dim))
                if dot < 0 and norm_sq > 0:
                    # Project out conflicting component
                    for k in range(grad_dim):
                        projected[i][k] -= (dot / norm_sq) * gradients[j][k]

        # Sum projected gradients
        result = [0.0] * grad_dim
        for g in projected:
            for i, val in enumerate(g):
                result[i] += val
        return tuple(result)

    elif strategy == GradientStrategy.CAGRAD:
        # Conflict-Averse Gradient
        # First compute average gradient
        avg = [0.0] * grad_dim
        for g in gradients:
            for i, val in enumerate(g):
                avg[i] += val / len(gradients)

        # Find direction that improves all tasks
        c = 0.5  # conflict aversion coefficient
        result = list(avg)
        for g in gradients:
            dot = sum(result[k] * g[k] for k in range(grad_dim))
            if dot < 0:
                norm_sq = sum(g[k] ** 2 for k in range(grad_dim))
                if norm_sq > 0:
                    for k in range(grad_dim):
                        result[k] += c * (-dot / norm_sq) * g[k]

        return tuple(result)

    elif strategy == GradientStrategy.MGDA:
        # Multiple Gradient Descent Algorithm (simplified)
        # Find min-norm point in convex hull
        # For simplicity, use equal weights as approximation
        result = [0.0] * grad_dim
        for g in gradients:
            for i, val in enumerate(g):
                result[i] += val / len(gradients)
        return tuple(result)

    elif strategy == GradientStrategy.NASH:
        # Nash equilibrium (simplified)
        # Similar to MGDA but with different weighting
        result = [0.0] * grad_dim
        for g in gradients:
            for i, val in enumerate(g):
                result[i] += val / len(gradients)
        return tuple(result)

    # Default to sum
    result = [0.0] * grad_dim
    for g in gradients:
        for i, val in enumerate(g):
            result[i] += val
    return tuple(result)


def estimate_task_difficulty(
    losses: tuple[float, ...],
    accuracies: tuple[float, ...],
) -> tuple[float, ...]:
    """Estimate relative difficulty of each task.

    Difficulty is estimated as a combination of loss and inverse accuracy.

    Args:
        losses: Current losses for each task.
        accuracies: Current accuracies for each task.

    Returns:
        Tuple of difficulty scores for each task.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> difficulties = estimate_task_difficulty(
        ...     (0.5, 0.1),
        ...     (0.8, 0.95),
        ... )
        >>> difficulties[0] > difficulties[1]
        True

        >>> estimate_task_difficulty((), ())
        Traceback (most recent call last):
            ...
        ValueError: losses and accuracies cannot be empty

        >>> estimate_task_difficulty((0.5,), (0.8, 0.9))
        Traceback (most recent call last):
            ...
        ValueError: losses and accuracies must have same length
    """
    if not losses or not accuracies:
        msg = "losses and accuracies cannot be empty"
        raise ValueError(msg)

    if len(losses) != len(accuracies):
        msg = "losses and accuracies must have same length"
        raise ValueError(msg)

    for loss in losses:
        if loss < 0:
            msg = "losses must be non-negative"
            raise ValueError(msg)

    for acc in accuracies:
        if not 0 <= acc <= 1:
            msg = "accuracies must be between 0 and 1"
            raise ValueError(msg)

    difficulties = []
    for loss, acc in zip(losses, accuracies, strict=True):
        # Combine loss and inverse accuracy
        inv_acc = 1.0 - acc if acc < 1.0 else 0.01
        difficulty = loss * inv_acc
        difficulties.append(difficulty)

    # Normalize to [0, 1]
    max_diff = max(difficulties) if difficulties else 1.0
    if max_diff > 0:
        difficulties = [d / max_diff for d in difficulties]

    return tuple(difficulties)


def format_multitask_stats(stats: MultiTaskStats) -> str:
    """Format multi-task statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_multitask_stats(
        ...     task_losses={"task1": 0.5, "task2": 0.3},
        ...     task_accuracies={"task1": 0.9, "task2": 0.85},
        ...     gradient_conflicts=5,
        ...     effective_weights={"task1": 1.0, "task2": 1.2},
        ... )
        >>> formatted = format_multitask_stats(stats)
        >>> "Gradient Conflicts: 5" in formatted
        True
        >>> "task1" in formatted
        True
    """
    lines = ["Multi-Task Stats:"]

    if stats.task_losses:
        lines.append("  Task Losses:")
        for name, loss in sorted(stats.task_losses.items()):
            lines.append(f"    {name}: {loss:.4f}")

    if stats.task_accuracies:
        lines.append("  Task Accuracies:")
        for name, acc in sorted(stats.task_accuracies.items()):
            lines.append(f"    {name}: {acc * 100:.1f}%")

    lines.append(f"  Gradient Conflicts: {stats.gradient_conflicts}")

    if stats.effective_weights:
        lines.append("  Effective Weights:")
        for name, weight in sorted(stats.effective_weights.items()):
            lines.append(f"    {name}: {weight:.4f}")

    return "\n".join(lines)


def get_recommended_multitask_config(task_type: str) -> MultiTaskConfig:
    """Get recommended multi-task configuration for a task type.

    Args:
        task_type: Type of multi-task scenario (classification, generation,
            mixed, hierarchical).

    Returns:
        Recommended MultiTaskConfig for the task type.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_multitask_config("classification")
        >>> config.balancing_method
        <TaskBalancing.UNIFORM: 'uniform'>
        >>> config.gradient_config.strategy
        <GradientStrategy.SUM: 'sum'>

        >>> config = get_recommended_multitask_config("mixed")
        >>> config.balancing_method
        <TaskBalancing.UNCERTAINTY: 'uncertainty'>
        >>> config.gradient_config.strategy
        <GradientStrategy.PCGRAD: 'pcgrad'>

        >>> get_recommended_multitask_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_type must be one of ...
    """
    valid_tasks = frozenset({
        "classification",
        "generation",
        "mixed",
        "hierarchical",
    })

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    if task_type == "classification":
        return create_multitask_config(
            strategy=GradientStrategy.SUM,
            balancing_method=TaskBalancing.UNIFORM,
        )
    elif task_type == "generation":
        return create_multitask_config(
            strategy=GradientStrategy.CAGRAD,
            balancing_method=TaskBalancing.GRADNORM,
        )
    elif task_type == "mixed":
        return create_multitask_config(
            strategy=GradientStrategy.PCGRAD,
            balancing_method=TaskBalancing.UNCERTAINTY,
            normalize=True,
        )
    else:  # hierarchical
        return create_multitask_config(
            strategy=GradientStrategy.MGDA,
            balancing_method=TaskBalancing.UNCERTAINTY,
        )
