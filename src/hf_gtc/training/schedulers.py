"""Learning rate scheduler utilities for training.

This module provides functions for configuring and computing learning rate schedules,
including linear, cosine, polynomial, and warmup strategies. It enables precise
control over how learning rates evolve during training.

Examples:
    >>> from hf_gtc.training.schedulers import (
    ...     create_lr_scheduler_config,
    ...     LRSchedulerType,
    ... )
    >>> config = create_lr_scheduler_config()
    >>> config.scheduler_type
    <LRSchedulerType.COSINE: 'cosine'>
    >>> config.total_steps
    1000
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class LRSchedulerType(Enum):
    """Types of learning rate schedulers.

    Attributes:
        CONSTANT: Constant learning rate (no decay).
        LINEAR: Linear decay from initial to final LR.
        COSINE: Cosine annealing schedule.
        COSINE_RESTARTS: Cosine annealing with warm restarts.
        POLYNOMIAL: Polynomial decay schedule.
        INVERSE_SQRT: Inverse square root decay.
        ONE_CYCLE: One-cycle policy with warmup and cooldown.

    Examples:
        >>> LRSchedulerType.CONSTANT.value
        'constant'
        >>> LRSchedulerType.LINEAR.value
        'linear'
        >>> LRSchedulerType.COSINE.value
        'cosine'
    """

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_RESTARTS = "cosine_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"
    ONE_CYCLE = "one_cycle"


class WarmupType(Enum):
    """Types of warmup strategies.

    Attributes:
        LINEAR: Linear warmup from initial to base LR.
        EXPONENTIAL: Exponential warmup curve.
        CONSTANT: Constant warmup at a fraction of base LR.

    Examples:
        >>> WarmupType.LINEAR.value
        'linear'
        >>> WarmupType.EXPONENTIAL.value
        'exponential'
        >>> WarmupType.CONSTANT.value
        'constant'
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"


class DecayType(Enum):
    """Types of learning rate decay.

    Attributes:
        LINEAR: Linear decay to min LR.
        EXPONENTIAL: Exponential decay to min LR.
        COSINE: Cosine annealing decay.

    Examples:
        >>> DecayType.LINEAR.value
        'linear'
        >>> DecayType.EXPONENTIAL.value
        'exponential'
        >>> DecayType.COSINE.value
        'cosine'
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"


VALID_SCHEDULER_TYPES = frozenset(s.value for s in LRSchedulerType)
VALID_WARMUP_TYPES = frozenset(w.value for w in WarmupType)
VALID_DECAY_TYPES = frozenset(d.value for d in DecayType)


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    """Configuration for learning rate warmup.

    Attributes:
        warmup_steps: Number of warmup steps (mutually exclusive with warmup_ratio).
        warmup_ratio: Ratio of total steps for warmup (0.0-1.0).
        warmup_type: Type of warmup schedule.

    Examples:
        >>> config = WarmupConfig(
        ...     warmup_steps=100,
        ...     warmup_ratio=0.0,
        ...     warmup_type=WarmupType.LINEAR,
        ... )
        >>> config.warmup_steps
        100
        >>> config.warmup_type
        <WarmupType.LINEAR: 'linear'>

        >>> config2 = WarmupConfig(
        ...     warmup_steps=0,
        ...     warmup_ratio=0.1,
        ...     warmup_type=WarmupType.EXPONENTIAL,
        ... )
        >>> config2.warmup_ratio
        0.1
    """

    warmup_steps: int
    warmup_ratio: float
    warmup_type: WarmupType


@dataclass(frozen=True, slots=True)
class CosineConfig:
    """Configuration for cosine scheduler.

    Attributes:
        num_cycles: Number of cosine cycles (for cosine_restarts).
        min_lr_ratio: Minimum LR as ratio of base LR (0.0-1.0).
        eta_min: Absolute minimum learning rate.

    Examples:
        >>> config = CosineConfig(
        ...     num_cycles=1.0,
        ...     min_lr_ratio=0.0,
        ...     eta_min=0.0,
        ... )
        >>> config.num_cycles
        1.0
        >>> config.min_lr_ratio
        0.0

        >>> config2 = CosineConfig(
        ...     num_cycles=3.0,
        ...     min_lr_ratio=0.1,
        ...     eta_min=1e-6,
        ... )
        >>> config2.num_cycles
        3.0
    """

    num_cycles: float
    min_lr_ratio: float
    eta_min: float


@dataclass(frozen=True, slots=True)
class PolynomialConfig:
    """Configuration for polynomial scheduler.

    Attributes:
        power: Power of the polynomial decay (1.0 = linear, 2.0 = quadratic).
        lr_end: Final learning rate at end of training.

    Examples:
        >>> config = PolynomialConfig(power=1.0, lr_end=0.0)
        >>> config.power
        1.0
        >>> config.lr_end
        0.0

        >>> config2 = PolynomialConfig(power=2.0, lr_end=1e-6)
        >>> config2.power
        2.0
    """

    power: float
    lr_end: float


@dataclass(frozen=True, slots=True)
class LRSchedulerConfig:
    """Main configuration for learning rate scheduling.

    Attributes:
        scheduler_type: Type of scheduler to use.
        warmup_config: Warmup configuration (optional).
        cosine_config: Cosine-specific configuration (optional).
        polynomial_config: Polynomial-specific configuration (optional).
        total_steps: Total training steps.
        num_epochs: Total number of training epochs.
        base_lr: Base learning rate.

    Examples:
        >>> warmup = WarmupConfig(100, 0.0, WarmupType.LINEAR)
        >>> cosine = CosineConfig(1.0, 0.0, 0.0)
        >>> config = LRSchedulerConfig(
        ...     scheduler_type=LRSchedulerType.COSINE,
        ...     warmup_config=warmup,
        ...     cosine_config=cosine,
        ...     polynomial_config=None,
        ...     total_steps=1000,
        ...     num_epochs=3,
        ...     base_lr=1e-4,
        ... )
        >>> config.scheduler_type
        <LRSchedulerType.COSINE: 'cosine'>
        >>> config.total_steps
        1000
    """

    scheduler_type: LRSchedulerType
    warmup_config: WarmupConfig | None
    cosine_config: CosineConfig | None
    polynomial_config: PolynomialConfig | None
    total_steps: int
    num_epochs: int
    base_lr: float


@dataclass(frozen=True, slots=True)
class SchedulerStats:
    """Statistics from learning rate scheduling.

    Attributes:
        current_lr: Current learning rate value.
        step: Current training step.
        warmup_complete: Whether warmup phase is complete.
        decay_progress: Progress through decay phase (0.0-1.0).

    Examples:
        >>> stats = SchedulerStats(
        ...     current_lr=5e-5,
        ...     step=500,
        ...     warmup_complete=True,
        ...     decay_progress=0.5,
        ... )
        >>> stats.current_lr
        5e-05
        >>> stats.warmup_complete
        True

        >>> stats2 = SchedulerStats(
        ...     current_lr=1e-5,
        ...     step=50,
        ...     warmup_complete=False,
        ...     decay_progress=0.0,
        ... )
        >>> stats2.warmup_complete
        False
    """

    current_lr: float
    step: int
    warmup_complete: bool
    decay_progress: float


def validate_warmup_config(config: WarmupConfig) -> None:
    """Validate warmup configuration.

    Args:
        config: Warmup configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = WarmupConfig(100, 0.0, WarmupType.LINEAR)
        >>> validate_warmup_config(config)

        >>> bad_config = WarmupConfig(-1, 0.0, WarmupType.LINEAR)
        >>> validate_warmup_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: warmup_steps cannot be negative

        >>> bad_config2 = WarmupConfig(0, -0.1, WarmupType.LINEAR)
        >>> validate_warmup_config(bad_config2)
        Traceback (most recent call last):
            ...
        ValueError: warmup_ratio must be between 0.0 and 1.0
    """
    if config.warmup_steps < 0:
        msg = "warmup_steps cannot be negative"
        raise ValueError(msg)

    if not 0.0 <= config.warmup_ratio <= 1.0:
        msg = "warmup_ratio must be between 0.0 and 1.0"
        raise ValueError(msg)


def validate_cosine_config(config: CosineConfig) -> None:
    """Validate cosine scheduler configuration.

    Args:
        config: Cosine configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = CosineConfig(1.0, 0.0, 0.0)
        >>> validate_cosine_config(config)

        >>> bad_config = CosineConfig(0.0, 0.0, 0.0)
        >>> validate_cosine_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_cycles must be positive

        >>> bad_config2 = CosineConfig(1.0, -0.1, 0.0)
        >>> validate_cosine_config(bad_config2)
        Traceback (most recent call last):
            ...
        ValueError: min_lr_ratio must be between 0.0 and 1.0
    """
    if config.num_cycles <= 0:
        msg = "num_cycles must be positive"
        raise ValueError(msg)

    if not 0.0 <= config.min_lr_ratio <= 1.0:
        msg = "min_lr_ratio must be between 0.0 and 1.0"
        raise ValueError(msg)

    if config.eta_min < 0:
        msg = "eta_min cannot be negative"
        raise ValueError(msg)


def validate_polynomial_config(config: PolynomialConfig) -> None:
    """Validate polynomial scheduler configuration.

    Args:
        config: Polynomial configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = PolynomialConfig(1.0, 0.0)
        >>> validate_polynomial_config(config)

        >>> bad_config = PolynomialConfig(0.0, 0.0)
        >>> validate_polynomial_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: power must be positive

        >>> bad_config2 = PolynomialConfig(1.0, -1e-6)
        >>> validate_polynomial_config(bad_config2)
        Traceback (most recent call last):
            ...
        ValueError: lr_end cannot be negative
    """
    if config.power <= 0:
        msg = "power must be positive"
        raise ValueError(msg)

    if config.lr_end < 0:
        msg = "lr_end cannot be negative"
        raise ValueError(msg)


def validate_lr_scheduler_config(config: LRSchedulerConfig) -> None:
    """Validate learning rate scheduler configuration.

    Args:
        config: Scheduler configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> warmup = WarmupConfig(100, 0.0, WarmupType.LINEAR)
        >>> cosine = CosineConfig(1.0, 0.0, 0.0)
        >>> config = LRSchedulerConfig(
        ...     scheduler_type=LRSchedulerType.COSINE,
        ...     warmup_config=warmup,
        ...     cosine_config=cosine,
        ...     polynomial_config=None,
        ...     total_steps=1000,
        ...     num_epochs=3,
        ...     base_lr=1e-4,
        ... )
        >>> validate_lr_scheduler_config(config)

        >>> bad_config = LRSchedulerConfig(
        ...     scheduler_type=LRSchedulerType.COSINE,
        ...     warmup_config=warmup,
        ...     cosine_config=cosine,
        ...     polynomial_config=None,
        ...     total_steps=0,
        ...     num_epochs=3,
        ...     base_lr=1e-4,
        ... )
        >>> validate_lr_scheduler_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: total_steps must be positive
    """
    if config.total_steps <= 0:
        msg = "total_steps must be positive"
        raise ValueError(msg)

    if config.num_epochs <= 0:
        msg = "num_epochs must be positive"
        raise ValueError(msg)

    if config.base_lr <= 0:
        msg = "base_lr must be positive"
        raise ValueError(msg)

    if config.warmup_config is not None:
        validate_warmup_config(config.warmup_config)

    if config.cosine_config is not None:
        validate_cosine_config(config.cosine_config)

    if config.polynomial_config is not None:
        validate_polynomial_config(config.polynomial_config)


def validate_scheduler_stats(stats: SchedulerStats) -> None:
    """Validate scheduler statistics.

    Args:
        stats: Scheduler statistics to validate.

    Raises:
        ValueError: If any value is invalid.

    Examples:
        >>> stats = SchedulerStats(1e-4, 100, True, 0.5)
        >>> validate_scheduler_stats(stats)

        >>> bad_stats = SchedulerStats(-1e-4, 100, True, 0.5)
        >>> validate_scheduler_stats(bad_stats)
        Traceback (most recent call last):
            ...
        ValueError: current_lr cannot be negative

        >>> bad_stats2 = SchedulerStats(1e-4, -1, True, 0.5)
        >>> validate_scheduler_stats(bad_stats2)
        Traceback (most recent call last):
            ...
        ValueError: step cannot be negative
    """
    if stats.current_lr < 0:
        msg = "current_lr cannot be negative"
        raise ValueError(msg)

    if stats.step < 0:
        msg = "step cannot be negative"
        raise ValueError(msg)

    if not 0.0 <= stats.decay_progress <= 1.0:
        msg = "decay_progress must be between 0.0 and 1.0"
        raise ValueError(msg)


def create_warmup_config(
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    warmup_type: str | WarmupType = WarmupType.LINEAR,
) -> WarmupConfig:
    """Create a warmup configuration with validation.

    Args:
        warmup_steps: Number of warmup steps.
        warmup_ratio: Ratio of total steps for warmup.
        warmup_type: Type of warmup schedule.

    Returns:
        Validated WarmupConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_warmup_config(warmup_steps=100)
        >>> config.warmup_steps
        100
        >>> config.warmup_type
        <WarmupType.LINEAR: 'linear'>

        >>> config2 = create_warmup_config(warmup_ratio=0.1, warmup_type="exponential")
        >>> config2.warmup_ratio
        0.1
        >>> config2.warmup_type
        <WarmupType.EXPONENTIAL: 'exponential'>

        >>> create_warmup_config(warmup_steps=-1)
        Traceback (most recent call last):
            ...
        ValueError: warmup_steps cannot be negative
    """
    if isinstance(warmup_type, str):
        warmup_type = get_warmup_type(warmup_type)

    config = WarmupConfig(
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        warmup_type=warmup_type,
    )
    validate_warmup_config(config)
    return config


def create_cosine_config(
    num_cycles: float = 1.0,
    min_lr_ratio: float = 0.0,
    eta_min: float = 0.0,
) -> CosineConfig:
    """Create a cosine scheduler configuration with validation.

    Args:
        num_cycles: Number of cosine cycles.
        min_lr_ratio: Minimum LR as ratio of base LR.
        eta_min: Absolute minimum learning rate.

    Returns:
        Validated CosineConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_cosine_config()
        >>> config.num_cycles
        1.0
        >>> config.min_lr_ratio
        0.0

        >>> config2 = create_cosine_config(num_cycles=3.0, min_lr_ratio=0.1)
        >>> config2.num_cycles
        3.0

        >>> create_cosine_config(num_cycles=0.0)
        Traceback (most recent call last):
            ...
        ValueError: num_cycles must be positive
    """
    config = CosineConfig(
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
        eta_min=eta_min,
    )
    validate_cosine_config(config)
    return config


def create_polynomial_config(
    power: float = 1.0,
    lr_end: float = 0.0,
) -> PolynomialConfig:
    """Create a polynomial scheduler configuration with validation.

    Args:
        power: Power of the polynomial decay.
        lr_end: Final learning rate.

    Returns:
        Validated PolynomialConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_polynomial_config()
        >>> config.power
        1.0
        >>> config.lr_end
        0.0

        >>> config2 = create_polynomial_config(power=2.0, lr_end=1e-6)
        >>> config2.power
        2.0

        >>> create_polynomial_config(power=0.0)
        Traceback (most recent call last):
            ...
        ValueError: power must be positive
    """
    config = PolynomialConfig(power=power, lr_end=lr_end)
    validate_polynomial_config(config)
    return config


def create_lr_scheduler_config(
    scheduler_type: str | LRSchedulerType = LRSchedulerType.COSINE,
    warmup_config: WarmupConfig | None = None,
    cosine_config: CosineConfig | None = None,
    polynomial_config: PolynomialConfig | None = None,
    total_steps: int = 1000,
    num_epochs: int = 3,
    base_lr: float = 1e-4,
) -> LRSchedulerConfig:
    """Create a learning rate scheduler configuration with validation.

    Args:
        scheduler_type: Type of scheduler to use.
        warmup_config: Warmup configuration.
        cosine_config: Cosine-specific configuration.
        polynomial_config: Polynomial-specific configuration.
        total_steps: Total training steps.
        num_epochs: Total number of training epochs.
        base_lr: Base learning rate.

    Returns:
        Validated LRSchedulerConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_lr_scheduler_config()
        >>> config.scheduler_type
        <LRSchedulerType.COSINE: 'cosine'>
        >>> config.total_steps
        1000
        >>> config.base_lr
        0.0001

        >>> warmup = create_warmup_config(warmup_steps=100)
        >>> config2 = create_lr_scheduler_config(
        ...     scheduler_type="linear",
        ...     warmup_config=warmup,
        ...     total_steps=2000,
        ... )
        >>> config2.scheduler_type
        <LRSchedulerType.LINEAR: 'linear'>

        >>> create_lr_scheduler_config(total_steps=0)
        Traceback (most recent call last):
            ...
        ValueError: total_steps must be positive
    """
    if isinstance(scheduler_type, str):
        scheduler_type = get_scheduler_type(scheduler_type)

    config = LRSchedulerConfig(
        scheduler_type=scheduler_type,
        warmup_config=warmup_config,
        cosine_config=cosine_config,
        polynomial_config=polynomial_config,
        total_steps=total_steps,
        num_epochs=num_epochs,
        base_lr=base_lr,
    )
    validate_lr_scheduler_config(config)
    return config


def create_scheduler_stats(
    current_lr: float = 0.0,
    step: int = 0,
    warmup_complete: bool = False,
    decay_progress: float = 0.0,
) -> SchedulerStats:
    """Create scheduler statistics with validation.

    Args:
        current_lr: Current learning rate value.
        step: Current training step.
        warmup_complete: Whether warmup phase is complete.
        decay_progress: Progress through decay phase (0.0-1.0).

    Returns:
        Validated SchedulerStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_scheduler_stats(current_lr=1e-4, step=100)
        >>> stats.current_lr
        0.0001
        >>> stats.step
        100

        >>> create_scheduler_stats(current_lr=-1e-4)
        Traceback (most recent call last):
            ...
        ValueError: current_lr cannot be negative
    """
    stats = SchedulerStats(
        current_lr=current_lr,
        step=step,
        warmup_complete=warmup_complete,
        decay_progress=decay_progress,
    )
    validate_scheduler_stats(stats)
    return stats


def list_scheduler_types() -> list[str]:
    """List all available scheduler types.

    Returns:
        Sorted list of scheduler type names.

    Examples:
        >>> types = list_scheduler_types()
        >>> "cosine" in types
        True
        >>> "linear" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_SCHEDULER_TYPES)


def list_warmup_types() -> list[str]:
    """List all available warmup types.

    Returns:
        Sorted list of warmup type names.

    Examples:
        >>> types = list_warmup_types()
        >>> "linear" in types
        True
        >>> "exponential" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_WARMUP_TYPES)


def list_decay_types() -> list[str]:
    """List all available decay types.

    Returns:
        Sorted list of decay type names.

    Examples:
        >>> types = list_decay_types()
        >>> "linear" in types
        True
        >>> "cosine" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_DECAY_TYPES)


def get_scheduler_type(name: str) -> LRSchedulerType:
    """Get scheduler type enum from string name.

    Args:
        name: Name of the scheduler type.

    Returns:
        Corresponding LRSchedulerType enum.

    Raises:
        ValueError: If scheduler type name is invalid.

    Examples:
        >>> get_scheduler_type("cosine")
        <LRSchedulerType.COSINE: 'cosine'>
        >>> get_scheduler_type("linear")
        <LRSchedulerType.LINEAR: 'linear'>

        >>> get_scheduler_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: scheduler_type must be one of ...
    """
    if name not in VALID_SCHEDULER_TYPES:
        msg = f"scheduler_type must be one of {VALID_SCHEDULER_TYPES}, got '{name}'"
        raise ValueError(msg)
    return LRSchedulerType(name)


def get_warmup_type(name: str) -> WarmupType:
    """Get warmup type enum from string name.

    Args:
        name: Name of the warmup type.

    Returns:
        Corresponding WarmupType enum.

    Raises:
        ValueError: If warmup type name is invalid.

    Examples:
        >>> get_warmup_type("linear")
        <WarmupType.LINEAR: 'linear'>
        >>> get_warmup_type("exponential")
        <WarmupType.EXPONENTIAL: 'exponential'>

        >>> get_warmup_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: warmup_type must be one of ...
    """
    if name not in VALID_WARMUP_TYPES:
        msg = f"warmup_type must be one of {VALID_WARMUP_TYPES}, got '{name}'"
        raise ValueError(msg)
    return WarmupType(name)


def get_decay_type(name: str) -> DecayType:
    """Get decay type enum from string name.

    Args:
        name: Name of the decay type.

    Returns:
        Corresponding DecayType enum.

    Raises:
        ValueError: If decay type name is invalid.

    Examples:
        >>> get_decay_type("linear")
        <DecayType.LINEAR: 'linear'>
        >>> get_decay_type("cosine")
        <DecayType.COSINE: 'cosine'>

        >>> get_decay_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: decay_type must be one of ...
    """
    if name not in VALID_DECAY_TYPES:
        msg = f"decay_type must be one of {VALID_DECAY_TYPES}, got '{name}'"
        raise ValueError(msg)
    return DecayType(name)


def calculate_warmup_lr(
    step: int,
    warmup_steps: int,
    base_lr: float,
    warmup_type: WarmupType = WarmupType.LINEAR,
) -> float:
    """Calculate learning rate during warmup phase.

    Args:
        step: Current training step.
        warmup_steps: Total warmup steps.
        base_lr: Target base learning rate after warmup.
        warmup_type: Type of warmup schedule.

    Returns:
        Learning rate at the given step.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> calculate_warmup_lr(0, 100, 1e-4)
        0.0
        >>> calculate_warmup_lr(50, 100, 1e-4)
        5e-05
        >>> calculate_warmup_lr(100, 100, 1e-4)
        0.0001

        >>> calculate_warmup_lr(50, 100, 1e-4, WarmupType.CONSTANT)
        0.0001

        >>> calculate_warmup_lr(-1, 100, 1e-4)
        Traceback (most recent call last):
            ...
        ValueError: step cannot be negative
    """
    if step < 0:
        msg = "step cannot be negative"
        raise ValueError(msg)

    if warmup_steps <= 0:
        msg = "warmup_steps must be positive"
        raise ValueError(msg)

    if base_lr <= 0:
        msg = "base_lr must be positive"
        raise ValueError(msg)

    if step >= warmup_steps:
        return base_lr

    progress = step / warmup_steps

    if warmup_type == WarmupType.LINEAR:
        return base_lr * progress
    elif warmup_type == WarmupType.EXPONENTIAL:
        # Exponential warmup: lr = base_lr * (exp_factor ^ (1 - progress))
        # Start very small, grow exponentially
        min_factor = 0.01
        return base_lr * (min_factor + (1 - min_factor) * (progress**2))
    else:  # CONSTANT
        return base_lr


def calculate_decay_lr(
    step: int,
    decay_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
    decay_type: DecayType = DecayType.COSINE,
) -> float:
    """Calculate learning rate during decay phase.

    Args:
        step: Current step in decay phase (0 = start of decay).
        decay_steps: Total steps in decay phase.
        base_lr: Starting learning rate for decay.
        min_lr: Minimum learning rate.
        decay_type: Type of decay schedule.

    Returns:
        Learning rate at the given step.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> calculate_decay_lr(0, 100, 1e-4)
        0.0001
        >>> round(calculate_decay_lr(50, 100, 1e-4), 10)
        5e-05
        >>> round(calculate_decay_lr(100, 100, 1e-4), 10)
        0.0

        >>> calculate_decay_lr(50, 100, 1e-4, decay_type=DecayType.LINEAR)
        5e-05

        >>> calculate_decay_lr(-1, 100, 1e-4)
        Traceback (most recent call last):
            ...
        ValueError: step cannot be negative
    """
    if step < 0:
        msg = "step cannot be negative"
        raise ValueError(msg)

    if decay_steps <= 0:
        msg = "decay_steps must be positive"
        raise ValueError(msg)

    if base_lr < 0:
        msg = "base_lr cannot be negative"
        raise ValueError(msg)

    if min_lr < 0:
        msg = "min_lr cannot be negative"
        raise ValueError(msg)

    if step >= decay_steps:
        return min_lr

    progress = step / decay_steps
    lr_range = base_lr - min_lr

    if decay_type == DecayType.LINEAR:
        return base_lr - lr_range * progress
    elif decay_type == DecayType.EXPONENTIAL:
        # Exponential decay: lr = base_lr * decay_rate^progress
        # We want it to reach min_lr at progress=1
        if min_lr > 0:
            decay_rate = min_lr / base_lr
            return base_lr * (decay_rate**progress)
        else:
            # Approximate with quadratic decay when min_lr is 0
            return base_lr * ((1 - progress) ** 2)
    else:  # COSINE
        # Cosine annealing: smooth decay following cosine curve
        return min_lr + lr_range * 0.5 * (1 + math.cos(math.pi * progress))


def calculate_lr_at_step(
    step: int,
    config: LRSchedulerConfig,
) -> float:
    """Calculate learning rate at a specific training step.

    This is the main function for computing LR given a scheduler configuration.
    It handles warmup, decay, and various scheduler types.

    Args:
        step: Current training step (0-indexed).
        config: Scheduler configuration.

    Returns:
        Learning rate at the given step.

    Raises:
        ValueError: If step is negative or config is invalid.

    Examples:
        >>> config = create_lr_scheduler_config(
        ...     scheduler_type="cosine",
        ...     total_steps=1000,
        ...     base_lr=1e-4,
        ... )
        >>> calculate_lr_at_step(0, config)
        0.0001
        >>> lr_mid = calculate_lr_at_step(500, config)
        >>> 0 < lr_mid < 1e-4
        True
        >>> round(calculate_lr_at_step(1000, config), 10)
        0.0

        >>> warmup = create_warmup_config(warmup_steps=100)
        >>> config2 = create_lr_scheduler_config(
        ...     scheduler_type="linear",
        ...     warmup_config=warmup,
        ...     total_steps=1000,
        ...     base_lr=1e-4,
        ... )
        >>> calculate_lr_at_step(50, config2)
        5e-05
        >>> calculate_lr_at_step(100, config2)
        0.0001

        >>> calculate_lr_at_step(-1, config)
        Traceback (most recent call last):
            ...
        ValueError: step cannot be negative
    """
    if step < 0:
        msg = "step cannot be negative"
        raise ValueError(msg)

    base_lr = config.base_lr
    total_steps = config.total_steps

    # Determine warmup parameters
    warmup_steps = 0
    warmup_type = WarmupType.LINEAR
    if config.warmup_config is not None:
        warmup_steps = config.warmup_config.warmup_steps
        if warmup_steps == 0 and config.warmup_config.warmup_ratio > 0:
            warmup_steps = int(total_steps * config.warmup_config.warmup_ratio)
        warmup_type = config.warmup_config.warmup_type

    # Warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        return calculate_warmup_lr(step, warmup_steps, base_lr, warmup_type)

    # Post-warmup step (adjusted for warmup)
    decay_step = step - warmup_steps
    decay_steps = total_steps - warmup_steps

    if decay_steps <= 0:
        return base_lr

    scheduler_type = config.scheduler_type

    # Determine min LR
    min_lr = 0.0
    if config.cosine_config is not None:
        min_lr = max(
            config.cosine_config.eta_min,
            base_lr * config.cosine_config.min_lr_ratio,
        )

    if config.polynomial_config is not None:
        min_lr = config.polynomial_config.lr_end

    if scheduler_type == LRSchedulerType.CONSTANT:
        return base_lr

    elif scheduler_type == LRSchedulerType.LINEAR:
        return calculate_decay_lr(
            decay_step, decay_steps, base_lr, min_lr, DecayType.LINEAR
        )

    elif scheduler_type == LRSchedulerType.COSINE:
        return calculate_decay_lr(
            decay_step, decay_steps, base_lr, min_lr, DecayType.COSINE
        )

    elif scheduler_type == LRSchedulerType.COSINE_RESTARTS:
        num_cycles = 1.0
        if config.cosine_config is not None:
            num_cycles = config.cosine_config.num_cycles

        # Calculate position within current cycle
        cycle_length = decay_steps / num_cycles
        cycle_step = decay_step % cycle_length
        progress = cycle_step / cycle_length

        lr_range = base_lr - min_lr
        return min_lr + lr_range * 0.5 * (1 + math.cos(math.pi * progress))

    elif scheduler_type == LRSchedulerType.POLYNOMIAL:
        power = 1.0
        if config.polynomial_config is not None:
            power = config.polynomial_config.power

        if decay_step >= decay_steps:
            return min_lr

        progress = decay_step / decay_steps
        lr_range = base_lr - min_lr
        return min_lr + lr_range * ((1 - progress) ** power)

    elif scheduler_type == LRSchedulerType.INVERSE_SQRT:
        # lr = base_lr / sqrt(step)
        # With offset to avoid division by zero
        offset = warmup_steps if warmup_steps > 0 else 1
        return base_lr * math.sqrt(offset / (step + 1))

    elif scheduler_type == LRSchedulerType.ONE_CYCLE:
        # One-cycle: warmup to peak, then decay to min
        # First half: warmup (already handled above if warmup_config provided)
        # Second half: decay
        mid_point = decay_steps // 2
        if decay_step < mid_point:
            # Rise phase (if no warmup) or sustain
            progress = decay_step / mid_point
            return min_lr + (base_lr - min_lr) * progress
        else:
            # Decay phase
            progress = (decay_step - mid_point) / (decay_steps - mid_point)
            lr_range = base_lr - min_lr
            return min_lr + lr_range * 0.5 * (1 + math.cos(math.pi * progress))

    return base_lr


def plot_lr_schedule(
    config: LRSchedulerConfig,
    num_points: int = 100,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Generate data points for plotting a learning rate schedule.

    Args:
        config: Scheduler configuration.
        num_points: Number of points to generate.

    Returns:
        Tuple of (steps, learning_rates) for plotting.

    Raises:
        ValueError: If num_points is not positive.

    Examples:
        >>> config = create_lr_scheduler_config(total_steps=100, base_lr=1e-3)
        >>> steps, lrs = plot_lr_schedule(config, num_points=10)
        >>> len(steps)
        10
        >>> len(lrs)
        10
        >>> steps[0]
        0
        >>> lrs[0]
        0.001

        >>> plot_lr_schedule(config, num_points=0)
        Traceback (most recent call last):
            ...
        ValueError: num_points must be positive
    """
    if num_points <= 0:
        msg = "num_points must be positive"
        raise ValueError(msg)

    total_steps = config.total_steps
    step_size = max(1, total_steps // num_points)

    steps = []
    lrs = []

    for i in range(num_points):
        step = min(i * step_size, total_steps)
        lr = calculate_lr_at_step(step, config)
        steps.append(step)
        lrs.append(lr)

    return tuple(steps), tuple(lrs)


def format_scheduler_stats(stats: SchedulerStats) -> str:
    """Format scheduler statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_scheduler_stats(
        ...     current_lr=5e-5,
        ...     step=500,
        ...     warmup_complete=True,
        ...     decay_progress=0.5,
        ... )
        >>> formatted = format_scheduler_stats(stats)
        >>> "LR: 5.00e-05" in formatted
        True
        >>> "Step: 500" in formatted
        True
        >>> "Warmup: Complete" in formatted
        True
        >>> "Decay: 50.0%" in formatted
        True
    """
    warmup_status = "Complete" if stats.warmup_complete else "In Progress"
    decay_pct = stats.decay_progress * 100

    return (
        f"Scheduler Stats:\n"
        f"  LR: {stats.current_lr:.2e}\n"
        f"  Step: {stats.step}\n"
        f"  Warmup: {warmup_status}\n"
        f"  Decay: {decay_pct:.1f}%"
    )


def get_recommended_scheduler_config(task_type: str) -> LRSchedulerConfig:
    """Get recommended scheduler configuration for a task type.

    Args:
        task_type: Type of task (classification, generation, fine_tuning,
            pretraining, rlhf).

    Returns:
        Recommended LRSchedulerConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_scheduler_config("classification")
        >>> config.scheduler_type
        <LRSchedulerType.COSINE: 'cosine'>
        >>> config.warmup_config is not None
        True

        >>> config2 = get_recommended_scheduler_config("fine_tuning")
        >>> config2.scheduler_type
        <LRSchedulerType.COSINE: 'cosine'>

        >>> get_recommended_scheduler_config("unknown")  # doctest: +ELLIPSIS
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
            "rlhf",
        }
    )

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    if task_type == "classification":
        return create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE,
            warmup_config=create_warmup_config(warmup_ratio=0.1),
            cosine_config=create_cosine_config(min_lr_ratio=0.01),
            total_steps=10000,
            num_epochs=10,
            base_lr=2e-5,
        )
    elif task_type == "generation":
        return create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE,
            warmup_config=create_warmup_config(warmup_ratio=0.05),
            cosine_config=create_cosine_config(min_lr_ratio=0.1, eta_min=1e-6),
            total_steps=50000,
            num_epochs=3,
            base_lr=5e-5,
        )
    elif task_type == "fine_tuning":
        return create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE,
            warmup_config=create_warmup_config(warmup_ratio=0.03),
            cosine_config=create_cosine_config(min_lr_ratio=0.0),
            total_steps=5000,
            num_epochs=3,
            base_lr=2e-5,
        )
    elif task_type == "pretraining":
        return create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE,
            warmup_config=create_warmup_config(warmup_steps=2000),
            cosine_config=create_cosine_config(min_lr_ratio=0.1),
            total_steps=100000,
            num_epochs=1,
            base_lr=1e-4,
        )
    else:  # rlhf
        return create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.LINEAR,
            warmup_config=create_warmup_config(warmup_ratio=0.1),
            polynomial_config=create_polynomial_config(power=1.0, lr_end=0.0),
            total_steps=20000,
            num_epochs=1,
            base_lr=1e-5,
        )
