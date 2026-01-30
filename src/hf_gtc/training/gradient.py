"""Gradient utilities for training.

This module provides utilities for gradient manipulation during training,
including gradient clipping, scaling, and accumulation strategies.

Examples:
    >>> from hf_gtc.training.gradient import create_gradient_config
    >>> config = create_gradient_config()
    >>> config.clipping_config.method
    <ClippingMethod.NORM: 'norm'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ClippingMethod(Enum):
    """Methods for gradient clipping.

    Attributes:
        NORM: Clip by global L2 norm.
        VALUE: Clip by absolute value.
        ADAPTIVE: Adaptive clipping based on gradient history.
        NONE: No clipping applied.

    Examples:
        >>> ClippingMethod.NORM.value
        'norm'
        >>> ClippingMethod.VALUE.value
        'value'
        >>> ClippingMethod.ADAPTIVE.value
        'adaptive'
        >>> ClippingMethod.NONE.value
        'none'
    """

    NORM = "norm"
    VALUE = "value"
    ADAPTIVE = "adaptive"
    NONE = "none"


VALID_CLIPPING_METHODS = frozenset(m.value for m in ClippingMethod)


class ScalingMethod(Enum):
    """Methods for gradient scaling.

    Attributes:
        STATIC: Fixed scaling factor.
        DYNAMIC: Automatically adjusted scale.
        LOSS_SCALE: Scale based on loss value.

    Examples:
        >>> ScalingMethod.STATIC.value
        'static'
        >>> ScalingMethod.DYNAMIC.value
        'dynamic'
        >>> ScalingMethod.LOSS_SCALE.value
        'loss_scale'
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    LOSS_SCALE = "loss_scale"


VALID_SCALING_METHODS = frozenset(m.value for m in ScalingMethod)


class AccumulationStrategy(Enum):
    """Strategies for gradient accumulation.

    Attributes:
        MEAN: Average gradients across accumulation steps.
        SUM: Sum gradients across accumulation steps.
        WEIGHTED: Weight gradients by step importance.

    Examples:
        >>> AccumulationStrategy.MEAN.value
        'mean'
        >>> AccumulationStrategy.SUM.value
        'sum'
        >>> AccumulationStrategy.WEIGHTED.value
        'weighted'
    """

    MEAN = "mean"
    SUM = "sum"
    WEIGHTED = "weighted"


VALID_ACCUMULATION_STRATEGIES = frozenset(s.value for s in AccumulationStrategy)


@dataclass(frozen=True, slots=True)
class ClippingConfig:
    """Configuration for gradient clipping.

    Attributes:
        method: Clipping method to use.
        max_norm: Maximum gradient norm for norm clipping.
        max_value: Maximum absolute value for value clipping.
        norm_type: Norm type (1, 2, or inf).

    Examples:
        >>> config = ClippingConfig(
        ...     method=ClippingMethod.NORM,
        ...     max_norm=1.0,
        ...     max_value=1.0,
        ...     norm_type=2.0,
        ... )
        >>> config.method
        <ClippingMethod.NORM: 'norm'>
        >>> config.max_norm
        1.0
    """

    method: ClippingMethod
    max_norm: float
    max_value: float
    norm_type: float


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    """Configuration for gradient scaling.

    Attributes:
        method: Scaling method to use.
        initial_scale: Initial scaling factor.
        growth_factor: Factor to increase scale by.
        backoff_factor: Factor to decrease scale by.

    Examples:
        >>> config = ScalingConfig(
        ...     method=ScalingMethod.DYNAMIC,
        ...     initial_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ... )
        >>> config.method
        <ScalingMethod.DYNAMIC: 'dynamic'>
        >>> config.initial_scale
        65536.0
    """

    method: ScalingMethod
    initial_scale: float
    growth_factor: float
    backoff_factor: float


@dataclass(frozen=True, slots=True)
class AccumulationConfig:
    """Configuration for gradient accumulation.

    Attributes:
        steps: Number of accumulation steps.
        strategy: Accumulation strategy to use.
        sync_grads: Whether to synchronize gradients.

    Examples:
        >>> config = AccumulationConfig(
        ...     steps=4,
        ...     strategy=AccumulationStrategy.MEAN,
        ...     sync_grads=True,
        ... )
        >>> config.steps
        4
        >>> config.strategy
        <AccumulationStrategy.MEAN: 'mean'>
    """

    steps: int
    strategy: AccumulationStrategy
    sync_grads: bool


@dataclass(frozen=True, slots=True)
class GradientConfig:
    """Combined gradient configuration.

    Attributes:
        clipping_config: Configuration for gradient clipping.
        scaling_config: Configuration for gradient scaling.
        accumulation_config: Configuration for gradient accumulation.

    Examples:
        >>> clipping = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        >>> scaling = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        >>> accumulation = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        >>> config = GradientConfig(clipping, scaling, accumulation)
        >>> config.clipping_config.method
        <ClippingMethod.NORM: 'norm'>
    """

    clipping_config: ClippingConfig
    scaling_config: ScalingConfig
    accumulation_config: AccumulationConfig


@dataclass(frozen=True, slots=True)
class GradientStats:
    """Statistics for gradient processing.

    Attributes:
        grad_norm: Current gradient norm.
        clipped_ratio: Ratio of gradients that were clipped.
        overflow_count: Count of gradient overflow events.
        effective_batch_size: Effective batch size with accumulation.

    Examples:
        >>> stats = GradientStats(
        ...     grad_norm=0.5,
        ...     clipped_ratio=0.1,
        ...     overflow_count=2,
        ...     effective_batch_size=32,
        ... )
        >>> stats.grad_norm
        0.5
        >>> stats.effective_batch_size
        32
    """

    grad_norm: float
    clipped_ratio: float
    overflow_count: int
    effective_batch_size: int


def validate_clipping_config(config: ClippingConfig) -> None:
    """Validate clipping configuration.

    Args:
        config: Clipping configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_norm is not positive for norm clipping.
        ValueError: If max_value is not positive for value clipping.
        ValueError: If norm_type is not positive.

    Examples:
        >>> config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        >>> validate_clipping_config(config)  # No error

        >>> validate_clipping_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ClippingConfig(ClippingMethod.NORM, 0.0, 1.0, 2.0)
        >>> validate_clipping_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_norm must be positive for norm clipping
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.method == ClippingMethod.NORM and config.max_norm <= 0:
        msg = f"max_norm must be positive for norm clipping, got {config.max_norm}"
        raise ValueError(msg)

    if config.method == ClippingMethod.VALUE and config.max_value <= 0:
        msg = f"max_value must be positive for value clipping, got {config.max_value}"
        raise ValueError(msg)

    if config.norm_type <= 0:
        msg = f"norm_type must be positive, got {config.norm_type}"
        raise ValueError(msg)


def validate_scaling_config(config: ScalingConfig) -> None:
    """Validate scaling configuration.

    Args:
        config: Scaling configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If initial_scale is not positive.
        ValueError: If growth_factor is not greater than 1.
        ValueError: If backoff_factor is not between 0 and 1.

    Examples:
        >>> config = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        >>> validate_scaling_config(config)  # No error

        >>> validate_scaling_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ScalingConfig(ScalingMethod.DYNAMIC, 0.0, 2.0, 0.5)
        >>> validate_scaling_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: initial_scale must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.initial_scale <= 0:
        msg = f"initial_scale must be positive, got {config.initial_scale}"
        raise ValueError(msg)

    if config.method != ScalingMethod.STATIC and config.growth_factor <= 1:
        msg = f"growth_factor must be greater than 1, got {config.growth_factor}"
        raise ValueError(msg)

    if config.method != ScalingMethod.STATIC and not 0 < config.backoff_factor < 1:
        msg = f"backoff_factor must be between 0 and 1, got {config.backoff_factor}"
        raise ValueError(msg)


def validate_accumulation_config(config: AccumulationConfig) -> None:
    """Validate accumulation configuration.

    Args:
        config: Accumulation configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If steps is not positive.

    Examples:
        >>> config = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        >>> validate_accumulation_config(config)  # No error

        >>> validate_accumulation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AccumulationConfig(0, AccumulationStrategy.MEAN, True)
        >>> validate_accumulation_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: steps must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.steps <= 0:
        msg = f"steps must be positive, got {config.steps}"
        raise ValueError(msg)


def validate_gradient_config(config: GradientConfig) -> None:
    """Validate gradient configuration.

    Args:
        config: Gradient configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If sub-configs are invalid.

    Examples:
        >>> clipping = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        >>> scaling = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        >>> accumulation = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        >>> config = GradientConfig(clipping, scaling, accumulation)
        >>> validate_gradient_config(config)  # No error

        >>> validate_gradient_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_clipping_config(config.clipping_config)
    validate_scaling_config(config.scaling_config)
    validate_accumulation_config(config.accumulation_config)


def validate_gradient_stats(stats: GradientStats) -> None:
    """Validate gradient statistics.

    Args:
        stats: Gradient statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If grad_norm is negative.
        ValueError: If clipped_ratio is not in [0, 1].
        ValueError: If overflow_count is negative.
        ValueError: If effective_batch_size is not positive.

    Examples:
        >>> stats = GradientStats(0.5, 0.1, 2, 32)
        >>> validate_gradient_stats(stats)  # No error

        >>> validate_gradient_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = GradientStats(-0.5, 0.1, 2, 32)
        >>> validate_gradient_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: grad_norm cannot be negative
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.grad_norm < 0:
        msg = f"grad_norm cannot be negative, got {stats.grad_norm}"
        raise ValueError(msg)

    if not 0 <= stats.clipped_ratio <= 1:
        msg = f"clipped_ratio must be in [0, 1], got {stats.clipped_ratio}"
        raise ValueError(msg)

    if stats.overflow_count < 0:
        msg = f"overflow_count cannot be negative, got {stats.overflow_count}"
        raise ValueError(msg)

    if stats.effective_batch_size <= 0:
        msg = f"effective_batch_size must be positive, got {stats.effective_batch_size}"
        raise ValueError(msg)


def create_clipping_config(
    method: str = "norm",
    max_norm: float = 1.0,
    max_value: float = 1.0,
    norm_type: float = 2.0,
) -> ClippingConfig:
    """Create a clipping configuration.

    Args:
        method: Clipping method. Defaults to "norm".
        max_norm: Maximum gradient norm. Defaults to 1.0.
        max_value: Maximum gradient value. Defaults to 1.0.
        norm_type: Norm type (1, 2, or inf). Defaults to 2.0.

    Returns:
        Validated ClippingConfig instance.

    Raises:
        ValueError: If method is invalid.
        ValueError: If parameters are invalid for the method.

    Examples:
        >>> config = create_clipping_config()
        >>> config.method
        <ClippingMethod.NORM: 'norm'>
        >>> config.max_norm
        1.0

        >>> config = create_clipping_config(method="value", max_value=0.5)
        >>> config.method
        <ClippingMethod.VALUE: 'value'>
        >>> config.max_value
        0.5

        >>> create_clipping_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if method not in VALID_CLIPPING_METHODS:
        msg = f"method must be one of {VALID_CLIPPING_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = ClippingConfig(
        method=ClippingMethod(method),
        max_norm=max_norm,
        max_value=max_value,
        norm_type=norm_type,
    )
    validate_clipping_config(config)
    return config


def create_scaling_config(
    method: str = "dynamic",
    initial_scale: float = 65536.0,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
) -> ScalingConfig:
    """Create a scaling configuration.

    Args:
        method: Scaling method. Defaults to "dynamic".
        initial_scale: Initial scale value. Defaults to 65536.0.
        growth_factor: Scale growth factor. Defaults to 2.0.
        backoff_factor: Scale backoff factor. Defaults to 0.5.

    Returns:
        Validated ScalingConfig instance.

    Raises:
        ValueError: If method is invalid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_scaling_config()
        >>> config.method
        <ScalingMethod.DYNAMIC: 'dynamic'>
        >>> config.initial_scale
        65536.0

        >>> config = create_scaling_config(method="static", initial_scale=1.0)
        >>> config.method
        <ScalingMethod.STATIC: 'static'>

        >>> create_scaling_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if method not in VALID_SCALING_METHODS:
        msg = f"method must be one of {VALID_SCALING_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = ScalingConfig(
        method=ScalingMethod(method),
        initial_scale=initial_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
    )
    validate_scaling_config(config)
    return config


def create_accumulation_config(
    steps: int = 1,
    strategy: str = "mean",
    sync_grads: bool = True,
) -> AccumulationConfig:
    """Create an accumulation configuration.

    Args:
        steps: Number of accumulation steps. Defaults to 1.
        strategy: Accumulation strategy. Defaults to "mean".
        sync_grads: Whether to sync gradients. Defaults to True.

    Returns:
        Validated AccumulationConfig instance.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If steps is not positive.

    Examples:
        >>> config = create_accumulation_config()
        >>> config.steps
        1
        >>> config.strategy
        <AccumulationStrategy.MEAN: 'mean'>

        >>> config = create_accumulation_config(steps=4, strategy="sum")
        >>> config.steps
        4
        >>> config.strategy
        <AccumulationStrategy.SUM: 'sum'>

        >>> create_accumulation_config(strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if strategy not in VALID_ACCUMULATION_STRATEGIES:
        msg = (
            f"strategy must be one of {VALID_ACCUMULATION_STRATEGIES}, got '{strategy}'"
        )
        raise ValueError(msg)

    config = AccumulationConfig(
        steps=steps,
        strategy=AccumulationStrategy(strategy),
        sync_grads=sync_grads,
    )
    validate_accumulation_config(config)
    return config


def create_gradient_config(
    clipping_config: ClippingConfig | None = None,
    scaling_config: ScalingConfig | None = None,
    accumulation_config: AccumulationConfig | None = None,
) -> GradientConfig:
    """Create a gradient configuration.

    Args:
        clipping_config: Clipping settings. Defaults to norm clipping.
        scaling_config: Scaling settings. Defaults to dynamic scaling.
        accumulation_config: Accumulation settings. Defaults to no accumulation.

    Returns:
        Validated GradientConfig instance.

    Examples:
        >>> config = create_gradient_config()
        >>> config.clipping_config.method
        <ClippingMethod.NORM: 'norm'>
        >>> config.scaling_config.method
        <ScalingMethod.DYNAMIC: 'dynamic'>

        >>> custom_clipping = create_clipping_config(method="value", max_value=0.5)
        >>> config = create_gradient_config(clipping_config=custom_clipping)
        >>> config.clipping_config.method
        <ClippingMethod.VALUE: 'value'>
    """
    if clipping_config is None:
        clipping_config = create_clipping_config()

    if scaling_config is None:
        scaling_config = create_scaling_config()

    if accumulation_config is None:
        accumulation_config = create_accumulation_config()

    config = GradientConfig(
        clipping_config=clipping_config,
        scaling_config=scaling_config,
        accumulation_config=accumulation_config,
    )
    validate_gradient_config(config)
    return config


def create_gradient_stats(
    grad_norm: float = 0.0,
    clipped_ratio: float = 0.0,
    overflow_count: int = 0,
    effective_batch_size: int = 1,
) -> GradientStats:
    """Create gradient statistics.

    Args:
        grad_norm: Current gradient norm. Defaults to 0.0.
        clipped_ratio: Ratio of clipped gradients. Defaults to 0.0.
        overflow_count: Number of overflow events. Defaults to 0.
        effective_batch_size: Effective batch size. Defaults to 1.

    Returns:
        Validated GradientStats instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_gradient_stats(grad_norm=0.5, effective_batch_size=32)
        >>> stats.grad_norm
        0.5
        >>> stats.effective_batch_size
        32

        >>> stats = create_gradient_stats()
        >>> stats.clipped_ratio
        0.0
    """
    stats = GradientStats(
        grad_norm=grad_norm,
        clipped_ratio=clipped_ratio,
        overflow_count=overflow_count,
        effective_batch_size=effective_batch_size,
    )
    validate_gradient_stats(stats)
    return stats


def calculate_grad_norm(
    gradients: list[float],
    norm_type: float = 2.0,
) -> float:
    """Calculate gradient norm.

    Args:
        gradients: List of gradient magnitudes.
        norm_type: Type of norm (1, 2, or inf). Defaults to 2.0.

    Returns:
        Calculated gradient norm.

    Raises:
        ValueError: If gradients is empty.
        ValueError: If norm_type is not positive.

    Examples:
        >>> calculate_grad_norm([3.0, 4.0], 2.0)
        5.0

        >>> calculate_grad_norm([1.0, 2.0, 3.0], 1.0)
        6.0

        >>> calculate_grad_norm([1.0, 5.0, 2.0], float('inf'))
        5.0

        >>> calculate_grad_norm([], 2.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradients cannot be empty
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if norm_type <= 0:
        msg = f"norm_type must be positive, got {norm_type}"
        raise ValueError(msg)

    if norm_type == float("inf"):
        return max(abs(g) for g in gradients)

    total = sum(abs(g) ** norm_type for g in gradients)
    return round(total ** (1.0 / norm_type), 10)


def clip_gradients(
    gradients: list[float],
    config: ClippingConfig,
) -> tuple[list[float], float]:
    """Clip gradients according to configuration.

    Args:
        gradients: List of gradient values.
        config: Clipping configuration.

    Returns:
        Tuple of (clipped_gradients, clip_ratio).

    Raises:
        ValueError: If gradients is empty.
        ValueError: If config is None.

    Examples:
        >>> config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        >>> grads = [0.6, 0.8]
        >>> clipped, ratio = clip_gradients(grads, config)
        >>> all(g <= 1.0 for g in clipped)
        True

        >>> config = ClippingConfig(ClippingMethod.VALUE, 1.0, 0.5, 2.0)
        >>> grads = [1.0, -2.0, 0.3]
        >>> clipped, ratio = clip_gradients(grads, config)
        >>> all(-0.5 <= g <= 0.5 for g in clipped)
        True

        >>> config = ClippingConfig(ClippingMethod.NONE, 1.0, 1.0, 2.0)
        >>> grads = [1.0, 2.0]
        >>> clipped, ratio = clip_gradients(grads, config)
        >>> clipped
        [1.0, 2.0]
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.method == ClippingMethod.NONE:
        return list(gradients), 0.0

    if config.method == ClippingMethod.VALUE:
        clipped = []
        clipped_count = 0
        for g in gradients:
            if abs(g) > config.max_value:
                clipped.append(config.max_value if g > 0 else -config.max_value)
                clipped_count += 1
            else:
                clipped.append(g)
        return clipped, clipped_count / len(gradients)

    if config.method in (ClippingMethod.NORM, ClippingMethod.ADAPTIVE):
        current_norm = calculate_grad_norm(gradients, config.norm_type)

        max_norm = config.max_norm
        if config.method == ClippingMethod.ADAPTIVE:
            # Adaptive clipping uses a smaller threshold
            max_norm = config.max_norm * 0.9

        if current_norm > max_norm:
            scale = max_norm / current_norm
            clipped = [g * scale for g in gradients]
            return clipped, 1.0
        return list(gradients), 0.0

    return list(gradients), 0.0


def scale_gradients(
    gradients: list[float],
    scale: float,
) -> list[float]:
    """Scale gradients by a factor.

    Args:
        gradients: List of gradient values.
        scale: Scaling factor.

    Returns:
        Scaled gradients.

    Raises:
        ValueError: If gradients is empty.
        ValueError: If scale is not positive.

    Examples:
        >>> scale_gradients([1.0, 2.0, 3.0], 2.0)
        [2.0, 4.0, 6.0]

        >>> scale_gradients([1.0, 2.0], 0.5)
        [0.5, 1.0]

        >>> scale_gradients([], 1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradients cannot be empty
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if scale <= 0:
        msg = f"scale must be positive, got {scale}"
        raise ValueError(msg)

    return [g * scale for g in gradients]


def accumulate_gradients(
    gradient_batches: list[list[float]],
    strategy: AccumulationStrategy,
    weights: list[float] | None = None,
) -> list[float]:
    """Accumulate gradients across batches.

    Args:
        gradient_batches: List of gradient batches to accumulate.
        strategy: Accumulation strategy to use.
        weights: Optional weights for weighted strategy.

    Returns:
        Accumulated gradients.

    Raises:
        ValueError: If gradient_batches is empty.
        ValueError: If gradient batches have different lengths.
        ValueError: If weights length doesn't match batches for weighted strategy.

    Examples:
        >>> batches = [[1.0, 2.0], [3.0, 4.0]]
        >>> accumulate_gradients(batches, AccumulationStrategy.MEAN)
        [2.0, 3.0]

        >>> accumulate_gradients(batches, AccumulationStrategy.SUM)
        [4.0, 6.0]

        >>> weights = [0.25, 0.75]
        >>> accumulate_gradients(batches, AccumulationStrategy.WEIGHTED, weights)
        [2.5, 3.5]

        >>> accumulate_gradients([], AccumulationStrategy.MEAN)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradient_batches cannot be empty
    """
    if not gradient_batches:
        msg = "gradient_batches cannot be empty"
        raise ValueError(msg)

    batch_length = len(gradient_batches[0])
    if any(len(batch) != batch_length for batch in gradient_batches):
        msg = "all gradient batches must have the same length"
        raise ValueError(msg)

    if strategy == AccumulationStrategy.WEIGHTED:
        if weights is None:
            msg = "weights must be provided for weighted strategy"
            raise ValueError(msg)
        if len(weights) != len(gradient_batches):
            msg = (
                f"weights length ({len(weights)}) must match "
                f"gradient_batches length ({len(gradient_batches)})"
            )
            raise ValueError(msg)

    num_batches = len(gradient_batches)
    accumulated = [0.0] * batch_length

    if strategy == AccumulationStrategy.SUM:
        for batch in gradient_batches:
            for i, g in enumerate(batch):
                accumulated[i] += g
    elif strategy == AccumulationStrategy.MEAN:
        for batch in gradient_batches:
            for i, g in enumerate(batch):
                accumulated[i] += g
        accumulated = [g / num_batches for g in accumulated]
    elif strategy == AccumulationStrategy.WEIGHTED:
        assert weights is not None  # Already validated above
        for batch_idx, batch in enumerate(gradient_batches):
            for i, g in enumerate(batch):
                accumulated[i] += g * weights[batch_idx]

    return accumulated


def detect_overflow(
    gradients: list[float],
    threshold: float = 65504.0,
) -> tuple[bool, int]:
    """Detect gradient overflow.

    Args:
        gradients: List of gradient values.
        threshold: Overflow threshold. Defaults to FP16 max (65504.0).

    Returns:
        Tuple of (overflow_detected, overflow_count).

    Raises:
        ValueError: If gradients is empty.
        ValueError: If threshold is not positive.

    Examples:
        >>> detect_overflow([1.0, 2.0, 3.0])
        (False, 0)

        >>> detect_overflow([1.0, 70000.0, 3.0])
        (True, 1)

        >>> detect_overflow([float('inf'), float('nan'), 1.0])
        (True, 2)

        >>> detect_overflow([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradients cannot be empty
    """
    if not gradients:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise ValueError(msg)

    import math

    overflow_count = 0
    for g in gradients:
        if math.isnan(g) or math.isinf(g) or abs(g) > threshold:
            overflow_count += 1

    return overflow_count > 0, overflow_count


def format_gradient_stats(stats: GradientStats) -> str:
    """Format gradient statistics for display.

    Args:
        stats: Gradient statistics to format.

    Returns:
        Formatted string with statistics.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = GradientStats(0.5, 0.1, 2, 32)
        >>> formatted = format_gradient_stats(stats)
        >>> "Gradient Norm:" in formatted
        True
        >>> "Clipped Ratio:" in formatted
        True

        >>> format_gradient_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        f"Gradient Norm: {stats.grad_norm:.6f}",
        f"Clipped Ratio: {stats.clipped_ratio:.2%}",
        f"Overflow Count: {stats.overflow_count}",
        f"Effective Batch Size: {stats.effective_batch_size}",
    ]
    return "\n".join(lines)


def get_recommended_gradient_config(
    model_size: str = "7b",
    training_type: str = "fine_tuning",
    accumulation_steps: int = 1,
) -> GradientConfig:
    """Get recommended gradient configuration.

    Args:
        model_size: Model size string (e.g., "7b", "70b"). Defaults to "7b".
        training_type: Type of training. Defaults to "fine_tuning".
        accumulation_steps: Number of accumulation steps. Defaults to 1.

    Returns:
        Recommended GradientConfig.

    Raises:
        ValueError: If training_type is invalid.
        ValueError: If accumulation_steps is not positive.

    Examples:
        >>> config = get_recommended_gradient_config("7b", "fine_tuning")
        >>> config.clipping_config.method
        <ClippingMethod.NORM: 'norm'>

        >>> config = get_recommended_gradient_config("70b", "pretraining", 4)
        >>> config.accumulation_config.steps
        4

        >>> get_recommended_gradient_config("7b", "invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: training_type must be one of
    """
    valid_training_types = {"fine_tuning", "pretraining", "rlhf"}
    if training_type not in valid_training_types:
        msg = (
            f"training_type must be one of {valid_training_types}, "
            f"got '{training_type}'"
        )
        raise ValueError(msg)

    if accumulation_steps <= 0:
        msg = f"accumulation_steps must be positive, got {accumulation_steps}"
        raise ValueError(msg)

    model_size = model_size.lower().strip()

    # Determine clipping settings based on model size and training type
    if training_type == "rlhf":
        # RLHF often needs stricter clipping
        max_norm = 0.5
        clipping_method = "norm"
    elif model_size in ("70b", "70", "175b", "175"):
        # Large models benefit from looser clipping
        max_norm = 2.0
        clipping_method = "norm"
    else:
        max_norm = 1.0
        clipping_method = "norm"

    clipping_config = create_clipping_config(
        method=clipping_method,
        max_norm=max_norm,
    )

    # Scaling config - use dynamic for most cases
    initial_scale = 2**16 if training_type == "pretraining" else 2**15

    scaling_config = create_scaling_config(
        method="dynamic",
        initial_scale=float(initial_scale),
    )

    # Accumulation config
    accumulation_config = create_accumulation_config(
        steps=accumulation_steps,
        strategy="mean",
        sync_grads=True,
    )

    return create_gradient_config(
        clipping_config=clipping_config,
        scaling_config=scaling_config,
        accumulation_config=accumulation_config,
    )


def list_clipping_methods() -> list[str]:
    """List supported clipping methods.

    Returns:
        Sorted list of clipping method names.

    Examples:
        >>> methods = list_clipping_methods()
        >>> "norm" in methods
        True
        >>> "value" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_CLIPPING_METHODS)


def list_scaling_methods() -> list[str]:
    """List supported scaling methods.

    Returns:
        Sorted list of scaling method names.

    Examples:
        >>> methods = list_scaling_methods()
        >>> "dynamic" in methods
        True
        >>> "static" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_SCALING_METHODS)


def list_accumulation_strategies() -> list[str]:
    """List supported accumulation strategies.

    Returns:
        Sorted list of accumulation strategy names.

    Examples:
        >>> strategies = list_accumulation_strategies()
        >>> "mean" in strategies
        True
        >>> "sum" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_ACCUMULATION_STRATEGIES)


def get_clipping_method(name: str) -> ClippingMethod:
    """Get clipping method from name.

    Args:
        name: Clipping method name.

    Returns:
        ClippingMethod enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_clipping_method("norm")
        <ClippingMethod.NORM: 'norm'>

        >>> get_clipping_method("value")
        <ClippingMethod.VALUE: 'value'>

        >>> get_clipping_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: clipping method must be one of
    """
    if name not in VALID_CLIPPING_METHODS:
        msg = f"clipping method must be one of {VALID_CLIPPING_METHODS}, got '{name}'"
        raise ValueError(msg)
    return ClippingMethod(name)


def get_scaling_method(name: str) -> ScalingMethod:
    """Get scaling method from name.

    Args:
        name: Scaling method name.

    Returns:
        ScalingMethod enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_scaling_method("dynamic")
        <ScalingMethod.DYNAMIC: 'dynamic'>

        >>> get_scaling_method("static")
        <ScalingMethod.STATIC: 'static'>

        >>> get_scaling_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scaling method must be one of
    """
    if name not in VALID_SCALING_METHODS:
        msg = f"scaling method must be one of {VALID_SCALING_METHODS}, got '{name}'"
        raise ValueError(msg)
    return ScalingMethod(name)


def get_accumulation_strategy(name: str) -> AccumulationStrategy:
    """Get accumulation strategy from name.

    Args:
        name: Accumulation strategy name.

    Returns:
        AccumulationStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_accumulation_strategy("mean")
        <AccumulationStrategy.MEAN: 'mean'>

        >>> get_accumulation_strategy("sum")
        <AccumulationStrategy.SUM: 'sum'>

        >>> get_accumulation_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: accumulation strategy must be one of
    """
    if name not in VALID_ACCUMULATION_STRATEGIES:
        msg = (
            f"accumulation strategy must be one of {VALID_ACCUMULATION_STRATEGIES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return AccumulationStrategy(name)
