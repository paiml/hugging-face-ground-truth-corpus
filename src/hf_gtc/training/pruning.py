"""Model pruning utilities for neural network compression.

This module provides functions for configuring and applying various
pruning techniques to reduce model size and improve inference efficiency,
including magnitude pruning, movement pruning, and lottery ticket methods.

Examples:
    >>> from hf_gtc.training.pruning import (
    ...     create_pruning_config,
    ...     PruningMethod,
    ... )
    >>> config = create_pruning_config(target_sparsity=0.5)
    >>> config.target_sparsity
    0.5
    >>> config.method
    <PruningMethod.MAGNITUDE: 'magnitude'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PruningMethod(Enum):
    """Supported pruning methods for model compression.

    Attributes:
        MAGNITUDE: Remove weights with smallest absolute values.
        MOVEMENT: Prune based on weight movement during training.
        LOTTERY_TICKET: Find sparse subnetworks via iterative pruning.
        STRUCTURED: Remove entire structures (heads, neurons).
        UNSTRUCTURED: Remove individual weights independently.
        GRADUAL: Gradually increase sparsity during training.

    Examples:
        >>> PruningMethod.MAGNITUDE.value
        'magnitude'
        >>> PruningMethod.LOTTERY_TICKET.value
        'lottery_ticket'
        >>> PruningMethod.STRUCTURED.value
        'structured'
    """

    MAGNITUDE = "magnitude"
    MOVEMENT = "movement"
    LOTTERY_TICKET = "lottery_ticket"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    GRADUAL = "gradual"


class PruningSchedule(Enum):
    """Scheduling strategies for applying pruning.

    Attributes:
        ONE_SHOT: Apply all pruning at once.
        ITERATIVE: Apply pruning in multiple iterations.
        CUBIC: Use cubic sparsity schedule.
        LINEAR: Linearly increase sparsity over time.

    Examples:
        >>> PruningSchedule.ONE_SHOT.value
        'one_shot'
        >>> PruningSchedule.ITERATIVE.value
        'iterative'
        >>> PruningSchedule.CUBIC.value
        'cubic'
    """

    ONE_SHOT = "one_shot"
    ITERATIVE = "iterative"
    CUBIC = "cubic"
    LINEAR = "linear"


class PruningScope(Enum):
    """Scope of pruning application.

    Attributes:
        GLOBAL_UNSTRUCTURED: Prune globally across all layers.
        LOCAL_UNSTRUCTURED: Prune within each layer independently.
        STRUCTURED_HEADS: Prune entire attention heads.
        STRUCTURED_NEURONS: Prune entire FFN neurons.

    Examples:
        >>> PruningScope.GLOBAL_UNSTRUCTURED.value
        'global_unstructured'
        >>> PruningScope.STRUCTURED_HEADS.value
        'structured_heads'
        >>> PruningScope.STRUCTURED_NEURONS.value
        'structured_neurons'
    """

    GLOBAL_UNSTRUCTURED = "global_unstructured"
    LOCAL_UNSTRUCTURED = "local_unstructured"
    STRUCTURED_HEADS = "structured_heads"
    STRUCTURED_NEURONS = "structured_neurons"


VALID_PRUNING_METHODS = frozenset(m.value for m in PruningMethod)
VALID_PRUNING_SCHEDULES = frozenset(s.value for s in PruningSchedule)
VALID_PRUNING_SCOPES = frozenset(s.value for s in PruningScope)


@dataclass(frozen=True, slots=True)
class PruningConfig:
    """Configuration for model pruning.

    Attributes:
        method: Pruning method to use.
        target_sparsity: Target fraction of weights to prune (0 to 1).
        schedule: How to apply pruning over time.
        scope: Scope of pruning application.

    Examples:
        >>> config = PruningConfig(
        ...     method=PruningMethod.MAGNITUDE,
        ...     target_sparsity=0.5,
        ...     schedule=PruningSchedule.ONE_SHOT,
        ...     scope=PruningScope.GLOBAL_UNSTRUCTURED,
        ... )
        >>> config.target_sparsity
        0.5
        >>> config.method
        <PruningMethod.MAGNITUDE: 'magnitude'>
    """

    method: PruningMethod
    target_sparsity: float
    schedule: PruningSchedule
    scope: PruningScope


@dataclass(frozen=True, slots=True)
class IterativePruningConfig:
    """Configuration for iterative pruning.

    Attributes:
        initial_sparsity: Starting sparsity level.
        final_sparsity: Target final sparsity level.
        pruning_steps: Number of pruning iterations.
        rewind_epoch: Epoch to rewind weights to after pruning.

    Examples:
        >>> config = IterativePruningConfig(
        ...     initial_sparsity=0.0,
        ...     final_sparsity=0.9,
        ...     pruning_steps=10,
        ...     rewind_epoch=0,
        ... )
        >>> config.final_sparsity
        0.9
        >>> config.pruning_steps
        10
    """

    initial_sparsity: float
    final_sparsity: float
    pruning_steps: int
    rewind_epoch: int


@dataclass(frozen=True, slots=True)
class LotteryTicketConfig:
    """Configuration for lottery ticket hypothesis experiments.

    Attributes:
        rewind_epoch: Epoch to rewind weights to for winning ticket.
        num_iterations: Number of iterative pruning rounds.
        target_sparsity: Target sparsity for final ticket.

    Examples:
        >>> config = LotteryTicketConfig(
        ...     rewind_epoch=0,
        ...     num_iterations=15,
        ...     target_sparsity=0.9,
        ... )
        >>> config.num_iterations
        15
        >>> config.target_sparsity
        0.9
    """

    rewind_epoch: int
    num_iterations: int
    target_sparsity: float


@dataclass(frozen=True, slots=True)
class PruningStats:
    """Statistics from model pruning.

    Attributes:
        original_params: Number of parameters before pruning.
        pruned_params: Number of parameters after pruning (non-zero).
        sparsity: Achieved sparsity level (fraction of zeros).
        speedup_factor: Estimated inference speedup.

    Examples:
        >>> stats = PruningStats(
        ...     original_params=110_000_000,
        ...     pruned_params=55_000_000,
        ...     sparsity=0.5,
        ...     speedup_factor=1.8,
        ... )
        >>> stats.sparsity
        0.5
        >>> stats.speedup_factor
        1.8
    """

    original_params: int
    pruned_params: int
    sparsity: float
    speedup_factor: float


def validate_pruning_config(config: PruningConfig) -> None:
    """Validate pruning configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = PruningConfig(
        ...     method=PruningMethod.MAGNITUDE,
        ...     target_sparsity=0.5,
        ...     schedule=PruningSchedule.ONE_SHOT,
        ...     scope=PruningScope.GLOBAL_UNSTRUCTURED,
        ... )
        >>> validate_pruning_config(config)

        >>> bad_config = PruningConfig(
        ...     method=PruningMethod.MAGNITUDE,
        ...     target_sparsity=1.5,
        ...     schedule=PruningSchedule.ONE_SHOT,
        ...     scope=PruningScope.GLOBAL_UNSTRUCTURED,
        ... )
        >>> validate_pruning_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: target_sparsity must be between 0 and 1, got 1.5
    """
    if not 0 <= config.target_sparsity <= 1:
        msg = f"target_sparsity must be between 0 and 1, got {config.target_sparsity}"
        raise ValueError(msg)


def validate_iterative_pruning_config(config: IterativePruningConfig) -> None:
    """Validate iterative pruning configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = IterativePruningConfig(
        ...     initial_sparsity=0.0,
        ...     final_sparsity=0.9,
        ...     pruning_steps=10,
        ...     rewind_epoch=0,
        ... )
        >>> validate_iterative_pruning_config(config)

        >>> bad_config = IterativePruningConfig(
        ...     initial_sparsity=0.5,
        ...     final_sparsity=0.3,
        ...     pruning_steps=10,
        ...     rewind_epoch=0,
        ... )
        >>> validate_iterative_pruning_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: final_sparsity (0.3) must be >= initial_sparsity (0.5)
    """
    if not 0 <= config.initial_sparsity <= 1:
        msg = f"initial_sparsity must be between 0 and 1, got {config.initial_sparsity}"
        raise ValueError(msg)
    if not 0 <= config.final_sparsity <= 1:
        msg = f"final_sparsity must be between 0 and 1, got {config.final_sparsity}"
        raise ValueError(msg)
    if config.final_sparsity < config.initial_sparsity:
        msg = (
            f"final_sparsity ({config.final_sparsity}) must be >= "
            f"initial_sparsity ({config.initial_sparsity})"
        )
        raise ValueError(msg)
    if config.pruning_steps <= 0:
        msg = f"pruning_steps must be positive, got {config.pruning_steps}"
        raise ValueError(msg)
    if config.rewind_epoch < 0:
        msg = f"rewind_epoch must be non-negative, got {config.rewind_epoch}"
        raise ValueError(msg)


def validate_lottery_ticket_config(config: LotteryTicketConfig) -> None:
    """Validate lottery ticket configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = LotteryTicketConfig(
        ...     rewind_epoch=0,
        ...     num_iterations=15,
        ...     target_sparsity=0.9,
        ... )
        >>> validate_lottery_ticket_config(config)

        >>> bad_config = LotteryTicketConfig(
        ...     rewind_epoch=-1,
        ...     num_iterations=15,
        ...     target_sparsity=0.9,
        ... )
        >>> validate_lottery_ticket_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: rewind_epoch must be non-negative, got -1
    """
    if config.rewind_epoch < 0:
        msg = f"rewind_epoch must be non-negative, got {config.rewind_epoch}"
        raise ValueError(msg)
    if config.num_iterations <= 0:
        msg = f"num_iterations must be positive, got {config.num_iterations}"
        raise ValueError(msg)
    if not 0 <= config.target_sparsity <= 1:
        msg = f"target_sparsity must be between 0 and 1, got {config.target_sparsity}"
        raise ValueError(msg)


def validate_pruning_stats(stats: PruningStats) -> None:
    """Validate pruning statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If any statistics value is invalid.

    Examples:
        >>> stats = PruningStats(
        ...     original_params=110_000_000,
        ...     pruned_params=55_000_000,
        ...     sparsity=0.5,
        ...     speedup_factor=1.8,
        ... )
        >>> validate_pruning_stats(stats)

        >>> bad_stats = PruningStats(
        ...     original_params=0,
        ...     pruned_params=55_000_000,
        ...     sparsity=0.5,
        ...     speedup_factor=1.8,
        ... )
        >>> validate_pruning_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: original_params must be positive, got 0
    """
    if stats.original_params <= 0:
        msg = f"original_params must be positive, got {stats.original_params}"
        raise ValueError(msg)
    if stats.pruned_params < 0:
        msg = f"pruned_params must be non-negative, got {stats.pruned_params}"
        raise ValueError(msg)
    if stats.pruned_params > stats.original_params:
        msg = (
            f"pruned_params ({stats.pruned_params}) cannot exceed "
            f"original_params ({stats.original_params})"
        )
        raise ValueError(msg)
    if not 0 <= stats.sparsity <= 1:
        msg = f"sparsity must be between 0 and 1, got {stats.sparsity}"
        raise ValueError(msg)
    if stats.speedup_factor < 1:
        msg = f"speedup_factor must be >= 1, got {stats.speedup_factor}"
        raise ValueError(msg)


def create_pruning_config(
    method: str | PruningMethod = PruningMethod.MAGNITUDE,
    target_sparsity: float = 0.5,
    schedule: str | PruningSchedule = PruningSchedule.ONE_SHOT,
    scope: str | PruningScope = PruningScope.GLOBAL_UNSTRUCTURED,
) -> PruningConfig:
    """Create a pruning configuration with validation.

    Args:
        method: Pruning method to use.
        target_sparsity: Target fraction of weights to prune.
        schedule: How to apply pruning over time.
        scope: Scope of pruning application.

    Returns:
        Validated PruningConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_pruning_config()
        >>> config.target_sparsity
        0.5
        >>> config.method
        <PruningMethod.MAGNITUDE: 'magnitude'>

        >>> config = create_pruning_config(method="movement", target_sparsity=0.7)
        >>> config.method
        <PruningMethod.MOVEMENT: 'movement'>
        >>> config.target_sparsity
        0.7

        >>> create_pruning_config(target_sparsity=1.5)
        Traceback (most recent call last):
            ...
        ValueError: target_sparsity must be between 0 and 1, got 1.5
    """
    if isinstance(method, str):
        method = get_pruning_method(method)
    if isinstance(schedule, str):
        schedule = get_pruning_schedule(schedule)
    if isinstance(scope, str):
        scope = get_pruning_scope(scope)

    config = PruningConfig(
        method=method,
        target_sparsity=target_sparsity,
        schedule=schedule,
        scope=scope,
    )
    validate_pruning_config(config)
    return config


def create_iterative_pruning_config(
    initial_sparsity: float = 0.0,
    final_sparsity: float = 0.9,
    pruning_steps: int = 10,
    rewind_epoch: int = 0,
) -> IterativePruningConfig:
    """Create an iterative pruning configuration with validation.

    Args:
        initial_sparsity: Starting sparsity level.
        final_sparsity: Target final sparsity level.
        pruning_steps: Number of pruning iterations.
        rewind_epoch: Epoch to rewind weights to.

    Returns:
        Validated IterativePruningConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_iterative_pruning_config()
        >>> config.initial_sparsity
        0.0
        >>> config.final_sparsity
        0.9
        >>> config.pruning_steps
        10

        >>> config = create_iterative_pruning_config(
        ...     initial_sparsity=0.3,
        ...     final_sparsity=0.95,
        ...     pruning_steps=20,
        ... )
        >>> config.final_sparsity
        0.95

        >>> create_iterative_pruning_config(pruning_steps=0)
        Traceback (most recent call last):
            ...
        ValueError: pruning_steps must be positive, got 0
    """
    config = IterativePruningConfig(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        pruning_steps=pruning_steps,
        rewind_epoch=rewind_epoch,
    )
    validate_iterative_pruning_config(config)
    return config


def create_lottery_ticket_config(
    rewind_epoch: int = 0,
    num_iterations: int = 15,
    target_sparsity: float = 0.9,
) -> LotteryTicketConfig:
    """Create a lottery ticket configuration with validation.

    Args:
        rewind_epoch: Epoch to rewind weights to for winning ticket.
        num_iterations: Number of iterative pruning rounds.
        target_sparsity: Target sparsity for final ticket.

    Returns:
        Validated LotteryTicketConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_lottery_ticket_config()
        >>> config.rewind_epoch
        0
        >>> config.num_iterations
        15
        >>> config.target_sparsity
        0.9

        >>> config = create_lottery_ticket_config(
        ...     rewind_epoch=1,
        ...     num_iterations=20,
        ...     target_sparsity=0.95,
        ... )
        >>> config.num_iterations
        20

        >>> create_lottery_ticket_config(num_iterations=0)
        Traceback (most recent call last):
            ...
        ValueError: num_iterations must be positive, got 0
    """
    config = LotteryTicketConfig(
        rewind_epoch=rewind_epoch,
        num_iterations=num_iterations,
        target_sparsity=target_sparsity,
    )
    validate_lottery_ticket_config(config)
    return config


def create_pruning_stats(
    original_params: int,
    pruned_params: int,
    sparsity: float | None = None,
    speedup_factor: float | None = None,
) -> PruningStats:
    """Create pruning statistics with optional auto-calculation.

    Args:
        original_params: Number of parameters before pruning.
        pruned_params: Number of parameters after pruning (non-zero).
        sparsity: Achieved sparsity level (auto-calculated if None).
        speedup_factor: Estimated inference speedup (auto-calculated if None).

    Returns:
        Validated PruningStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_pruning_stats(110_000_000, 55_000_000)
        >>> stats.sparsity
        0.5
        >>> stats.speedup_factor > 1.0
        True

        >>> stats = create_pruning_stats(
        ...     original_params=100_000,
        ...     pruned_params=10_000,
        ...     sparsity=0.9,
        ...     speedup_factor=2.5,
        ... )
        >>> stats.sparsity
        0.9

        >>> create_pruning_stats(0, 50_000)
        Traceback (most recent call last):
            ...
        ValueError: original_params must be positive, got 0
    """
    if original_params <= 0:
        msg = f"original_params must be positive, got {original_params}"
        raise ValueError(msg)

    if sparsity is None:
        sparsity = calculate_sparsity(original_params, pruned_params)
    if speedup_factor is None:
        speedup_factor = estimate_speedup(sparsity)

    stats = PruningStats(
        original_params=original_params,
        pruned_params=pruned_params,
        sparsity=sparsity,
        speedup_factor=speedup_factor,
    )
    validate_pruning_stats(stats)
    return stats


def list_pruning_methods() -> list[str]:
    """List all available pruning methods.

    Returns:
        Sorted list of pruning method names.

    Examples:
        >>> methods = list_pruning_methods()
        >>> "magnitude" in methods
        True
        >>> "lottery_ticket" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_PRUNING_METHODS)


def list_pruning_schedules() -> list[str]:
    """List all available pruning schedules.

    Returns:
        Sorted list of pruning schedule names.

    Examples:
        >>> schedules = list_pruning_schedules()
        >>> "one_shot" in schedules
        True
        >>> "iterative" in schedules
        True
        >>> schedules == sorted(schedules)
        True
    """
    return sorted(VALID_PRUNING_SCHEDULES)


def list_pruning_scopes() -> list[str]:
    """List all available pruning scopes.

    Returns:
        Sorted list of pruning scope names.

    Examples:
        >>> scopes = list_pruning_scopes()
        >>> "global_unstructured" in scopes
        True
        >>> "structured_heads" in scopes
        True
        >>> scopes == sorted(scopes)
        True
    """
    return sorted(VALID_PRUNING_SCOPES)


def get_pruning_method(name: str) -> PruningMethod:
    """Get pruning method enum from string name.

    Args:
        name: Name of the pruning method.

    Returns:
        Corresponding PruningMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_pruning_method("magnitude")
        <PruningMethod.MAGNITUDE: 'magnitude'>
        >>> get_pruning_method("lottery_ticket")
        <PruningMethod.LOTTERY_TICKET: 'lottery_ticket'>

        >>> get_pruning_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_PRUNING_METHODS:
        msg = f"method must be one of {VALID_PRUNING_METHODS}, got '{name}'"
        raise ValueError(msg)
    return PruningMethod(name)


def get_pruning_schedule(name: str) -> PruningSchedule:
    """Get pruning schedule enum from string name.

    Args:
        name: Name of the pruning schedule.

    Returns:
        Corresponding PruningSchedule enum.

    Raises:
        ValueError: If schedule name is invalid.

    Examples:
        >>> get_pruning_schedule("one_shot")
        <PruningSchedule.ONE_SHOT: 'one_shot'>
        >>> get_pruning_schedule("iterative")
        <PruningSchedule.ITERATIVE: 'iterative'>

        >>> get_pruning_schedule("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: schedule must be one of ...
    """
    if name not in VALID_PRUNING_SCHEDULES:
        msg = f"schedule must be one of {VALID_PRUNING_SCHEDULES}, got '{name}'"
        raise ValueError(msg)
    return PruningSchedule(name)


def get_pruning_scope(name: str) -> PruningScope:
    """Get pruning scope enum from string name.

    Args:
        name: Name of the pruning scope.

    Returns:
        Corresponding PruningScope enum.

    Raises:
        ValueError: If scope name is invalid.

    Examples:
        >>> get_pruning_scope("global_unstructured")
        <PruningScope.GLOBAL_UNSTRUCTURED: 'global_unstructured'>
        >>> get_pruning_scope("structured_heads")
        <PruningScope.STRUCTURED_HEADS: 'structured_heads'>

        >>> get_pruning_scope("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: scope must be one of ...
    """
    if name not in VALID_PRUNING_SCOPES:
        msg = f"scope must be one of {VALID_PRUNING_SCOPES}, got '{name}'"
        raise ValueError(msg)
    return PruningScope(name)


def calculate_sparsity(original_params: int, remaining_params: int) -> float:
    """Calculate sparsity from parameter counts.

    Args:
        original_params: Number of parameters before pruning.
        remaining_params: Number of non-zero parameters after pruning.

    Returns:
        Sparsity level (fraction of zeros).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_sparsity(100, 50)
        0.5
        >>> calculate_sparsity(1000, 100)
        0.9
        >>> calculate_sparsity(100, 100)
        0.0

        >>> calculate_sparsity(0, 50)
        Traceback (most recent call last):
            ...
        ValueError: original_params must be positive, got 0
    """
    if original_params <= 0:
        msg = f"original_params must be positive, got {original_params}"
        raise ValueError(msg)
    if remaining_params < 0:
        msg = f"remaining_params must be non-negative, got {remaining_params}"
        raise ValueError(msg)
    if remaining_params > original_params:
        msg = (
            f"remaining_params ({remaining_params}) cannot exceed "
            f"original_params ({original_params})"
        )
        raise ValueError(msg)

    return 1.0 - (remaining_params / original_params)


def estimate_speedup(sparsity: float, efficiency_factor: float = 0.7) -> float:
    """Estimate inference speedup from sparsity level.

    The speedup is estimated based on typical hardware efficiency for
    sparse operations. Perfect linear speedup is rarely achieved.

    Args:
        sparsity: Fraction of weights that are zero (0 to 1).
        efficiency_factor: Hardware efficiency for sparse ops (0 to 1).

    Returns:
        Estimated speedup factor (>= 1.0).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate_speedup(0.0)
        1.0
        >>> speedup = estimate_speedup(0.5)
        >>> 1.0 < speedup < 2.0
        True
        >>> estimate_speedup(0.9) > estimate_speedup(0.5)
        True

        >>> estimate_speedup(1.5)
        Traceback (most recent call last):
            ...
        ValueError: sparsity must be between 0 and 1, got 1.5
    """
    if not 0 <= sparsity <= 1:
        msg = f"sparsity must be between 0 and 1, got {sparsity}"
        raise ValueError(msg)
    if not 0 < efficiency_factor <= 1:
        msg = f"efficiency_factor must be between 0 and 1, got {efficiency_factor}"
        raise ValueError(msg)

    if sparsity == 0:
        return 1.0

    # Theoretical max speedup = 1 / (1 - sparsity)
    # Actual speedup is reduced by efficiency factor
    theoretical_speedup = 1.0 / (1.0 - sparsity) if sparsity < 1 else 100.0
    actual_speedup = 1.0 + (theoretical_speedup - 1.0) * efficiency_factor

    return actual_speedup


def calculate_pruning_mask(
    weights: tuple[float, ...],
    sparsity: float,
    method: PruningMethod = PruningMethod.MAGNITUDE,
) -> tuple[bool, ...]:
    """Calculate a pruning mask for given weights.

    Args:
        weights: Weight values to evaluate.
        sparsity: Target sparsity (fraction to prune).
        method: Pruning method to use.

    Returns:
        Boolean mask where True means keep the weight.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> weights = (0.1, 0.5, 0.2, 0.8, 0.3)
        >>> mask = calculate_pruning_mask(weights, 0.4)
        >>> sum(mask)  # 60% kept
        3
        >>> mask
        (False, True, False, True, True)

        >>> calculate_pruning_mask((), 0.5)
        ()

        >>> calculate_pruning_mask(weights, 1.5)
        Traceback (most recent call last):
            ...
        ValueError: sparsity must be between 0 and 1, got 1.5
    """
    if not 0 <= sparsity <= 1:
        msg = f"sparsity must be between 0 and 1, got {sparsity}"
        raise ValueError(msg)

    if not weights:
        return ()

    n = len(weights)
    num_to_prune = int(n * sparsity)

    if method == PruningMethod.MAGNITUDE:
        # Sort by absolute value, keep largest
        indexed = sorted(enumerate(weights), key=lambda x: abs(x[1]))
        prune_indices = {idx for idx, _ in indexed[:num_to_prune]}
        return tuple(i not in prune_indices for i in range(n))
    else:
        # For other methods, use magnitude as fallback
        indexed = sorted(enumerate(weights), key=lambda x: abs(x[1]))
        prune_indices = {idx for idx, _ in indexed[:num_to_prune]}
        return tuple(i not in prune_indices for i in range(n))


def schedule_sparsity(
    current_step: int,
    total_steps: int,
    initial_sparsity: float,
    final_sparsity: float,
    schedule: PruningSchedule = PruningSchedule.CUBIC,
) -> float:
    """Calculate sparsity at a given training step based on schedule.

    Args:
        current_step: Current training step.
        total_steps: Total number of training steps.
        initial_sparsity: Starting sparsity level.
        final_sparsity: Target final sparsity level.
        schedule: Scheduling strategy to use.

    Returns:
        Sparsity level at the current step.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> schedule_sparsity(0, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        0.0
        >>> schedule_sparsity(500, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        0.45
        >>> schedule_sparsity(1000, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
        0.9

        >>> schedule_sparsity(-1, 1000, 0.0, 0.9, PruningSchedule.LINEAR)
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
    if not 0 <= initial_sparsity <= 1:
        msg = f"initial_sparsity must be between 0 and 1, got {initial_sparsity}"
        raise ValueError(msg)
    if not 0 <= final_sparsity <= 1:
        msg = f"final_sparsity must be between 0 and 1, got {final_sparsity}"
        raise ValueError(msg)

    progress = current_step / total_steps

    if schedule == PruningSchedule.ONE_SHOT:
        return final_sparsity if current_step == total_steps else initial_sparsity

    elif schedule == PruningSchedule.LINEAR:
        return initial_sparsity + (final_sparsity - initial_sparsity) * progress

    elif schedule == PruningSchedule.CUBIC:
        # Cubic schedule: slower start, faster end
        cubic_progress = progress**3
        return initial_sparsity + (final_sparsity - initial_sparsity) * cubic_progress

    elif schedule == PruningSchedule.ITERATIVE:
        # Step function (for external iteration control)
        return initial_sparsity + (final_sparsity - initial_sparsity) * progress

    return initial_sparsity


def format_pruning_stats(stats: PruningStats) -> str:
    """Format pruning statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_pruning_stats(110_000_000, 55_000_000)
        >>> formatted = format_pruning_stats(stats)
        >>> "Sparsity: 50.0%" in formatted
        True
        >>> "Original: 110.00M" in formatted
        True
    """

    def format_params(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        else:
            return str(n)

    return (
        f"Pruning Stats:\n"
        f"  Original: {format_params(stats.original_params)}\n"
        f"  Remaining: {format_params(stats.pruned_params)}\n"
        f"  Sparsity: {stats.sparsity * 100:.1f}%\n"
        f"  Speedup: {stats.speedup_factor:.2f}x"
    )


def get_recommended_pruning_config(
    model_size: str = "base",
    task: str = "classification",
) -> PruningConfig:
    """Get recommended pruning configuration for a model and task.

    Args:
        model_size: Model size category (small, base, large, xl).
        task: Task type (classification, generation, embedding).

    Returns:
        Recommended PruningConfig for the combination.

    Raises:
        ValueError: If model_size or task is unknown.

    Examples:
        >>> config = get_recommended_pruning_config()
        >>> config.target_sparsity
        0.5
        >>> config.method
        <PruningMethod.MAGNITUDE: 'magnitude'>

        >>> config = get_recommended_pruning_config("large", "generation")
        >>> config.target_sparsity
        0.7

        >>> get_recommended_pruning_config("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: model_size must be one of ...
    """
    valid_sizes = frozenset({"small", "base", "large", "xl"})
    valid_tasks = frozenset({"classification", "generation", "embedding"})

    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    # Base sparsity by model size
    size_sparsity = {
        "small": 0.3,
        "base": 0.5,
        "large": 0.7,
        "xl": 0.8,
    }

    # Task-based method selection
    if task == "classification":
        method = PruningMethod.MAGNITUDE
        scope = PruningScope.GLOBAL_UNSTRUCTURED
    elif task == "generation":
        method = PruningMethod.GRADUAL
        scope = PruningScope.LOCAL_UNSTRUCTURED
    else:  # embedding
        method = PruningMethod.STRUCTURED
        scope = PruningScope.STRUCTURED_NEURONS

    # Larger models benefit from iterative pruning
    schedule = (
        PruningSchedule.ITERATIVE
        if model_size in ("large", "xl")
        else PruningSchedule.ONE_SHOT
    )

    return create_pruning_config(
        method=method,
        target_sparsity=size_sparsity[model_size],
        schedule=schedule,
        scope=scope,
    )
