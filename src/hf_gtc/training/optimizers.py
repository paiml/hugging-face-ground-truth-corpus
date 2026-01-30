"""Optimizer configuration utilities.

This module provides functions for configuring and comparing various optimizers
used in training ML models, including AdamW, Lion, Sophia, Adafactor, and 8-bit Adam.

Examples:
    >>> from hf_gtc.training.optimizers import create_adamw_config
    >>> config = create_adamw_config(lr=1e-4)
    >>> config.lr
    0.0001
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class OptimizerType(Enum):
    """Supported optimizer types.

    Attributes:
        ADAMW: AdamW optimizer with decoupled weight decay.
        ADAM: Standard Adam optimizer.
        SGD: Stochastic Gradient Descent.
        LION: Lion optimizer (EvoLved Sign Momentum).
        SOPHIA: Sophia optimizer with Hessian-based preconditioning.
        ADAFACTOR: Memory-efficient Adafactor optimizer.
        ADAM_8BIT: 8-bit Adam for memory efficiency.
        PAGED_ADAMW: Paged AdamW for large models.

    Examples:
        >>> OptimizerType.ADAMW.value
        'adamw'
        >>> OptimizerType.LION.value
        'lion'
        >>> OptimizerType.SOPHIA.value
        'sophia'
    """

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    LION = "lion"
    SOPHIA = "sophia"
    ADAFACTOR = "adafactor"
    ADAM_8BIT = "adam_8bit"
    PAGED_ADAMW = "paged_adamw"


VALID_OPTIMIZER_TYPES = frozenset(o.value for o in OptimizerType)


class WeightDecayType(Enum):
    """Weight decay implementation types.

    Attributes:
        DECOUPLED: Decoupled weight decay (AdamW style).
        L2: L2 regularization.
        NONE: No weight decay.

    Examples:
        >>> WeightDecayType.DECOUPLED.value
        'decoupled'
        >>> WeightDecayType.L2.value
        'l2'
        >>> WeightDecayType.NONE.value
        'none'
    """

    DECOUPLED = "decoupled"
    L2 = "l2"
    NONE = "none"


VALID_WEIGHT_DECAY_TYPES = frozenset(w.value for w in WeightDecayType)


class MomentumType(Enum):
    """Momentum implementation types.

    Attributes:
        STANDARD: Standard momentum.
        NESTEROV: Nesterov accelerated gradient.
        HEAVY_BALL: Heavy ball momentum.

    Examples:
        >>> MomentumType.STANDARD.value
        'standard'
        >>> MomentumType.NESTEROV.value
        'nesterov'
        >>> MomentumType.HEAVY_BALL.value
        'heavy_ball'
    """

    STANDARD = "standard"
    NESTEROV = "nesterov"
    HEAVY_BALL = "heavy_ball"


VALID_MOMENTUM_TYPES = frozenset(m.value for m in MomentumType)


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    """Configuration for AdamW optimizer.

    Attributes:
        lr: Learning rate.
        betas: Coefficients for running averages (beta1, beta2).
        eps: Term added for numerical stability.
        weight_decay: Weight decay coefficient.
        amsgrad: Whether to use AMSGrad variant.

    Examples:
        >>> config = AdamWConfig(
        ...     lr=1e-4,
        ...     betas=(0.9, 0.999),
        ...     eps=1e-8,
        ...     weight_decay=0.01,
        ...     amsgrad=False,
        ... )
        >>> config.lr
        0.0001
        >>> config.betas
        (0.9, 0.999)
    """

    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool


@dataclass(frozen=True, slots=True)
class LionConfig:
    """Configuration for Lion optimizer.

    Attributes:
        lr: Learning rate.
        betas: Coefficients for running averages (beta1, beta2).
        weight_decay: Weight decay coefficient.

    Examples:
        >>> config = LionConfig(
        ...     lr=1e-4,
        ...     betas=(0.9, 0.99),
        ...     weight_decay=0.01,
        ... )
        >>> config.lr
        0.0001
        >>> config.betas
        (0.9, 0.99)
    """

    lr: float
    betas: tuple[float, float]
    weight_decay: float


@dataclass(frozen=True, slots=True)
class SophiaConfig:
    """Configuration for Sophia optimizer.

    Attributes:
        lr: Learning rate.
        betas: Coefficients for running averages (beta1, beta2).
        rho: Hessian estimation coefficient.
        weight_decay: Weight decay coefficient.

    Examples:
        >>> config = SophiaConfig(
        ...     lr=1e-4,
        ...     betas=(0.965, 0.99),
        ...     rho=0.04,
        ...     weight_decay=0.1,
        ... )
        >>> config.lr
        0.0001
        >>> config.rho
        0.04
    """

    lr: float
    betas: tuple[float, float]
    rho: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class AdafactorConfig:
    """Configuration for Adafactor optimizer.

    Attributes:
        lr: Learning rate (None for relative step).
        eps: Regularization constants (eps1, eps2).
        clip_threshold: Gradient clipping threshold.
        scale_parameter: Whether to scale parameter learning rate.

    Examples:
        >>> config = AdafactorConfig(
        ...     lr=None,
        ...     eps=(1e-30, 1e-3),
        ...     clip_threshold=1.0,
        ...     scale_parameter=True,
        ... )
        >>> config.lr is None
        True
        >>> config.scale_parameter
        True
    """

    lr: float | None
    eps: tuple[float, float]
    clip_threshold: float
    scale_parameter: bool


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Combined optimizer configuration.

    Attributes:
        optimizer_type: Type of optimizer to use.
        adamw_config: AdamW configuration (if applicable).
        lion_config: Lion configuration (if applicable).
        sophia_config: Sophia configuration (if applicable).
        adafactor_config: Adafactor configuration (if applicable).

    Examples:
        >>> adamw = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        >>> config = OptimizerConfig(
        ...     optimizer_type=OptimizerType.ADAMW,
        ...     adamw_config=adamw,
        ...     lion_config=None,
        ...     sophia_config=None,
        ...     adafactor_config=None,
        ... )
        >>> config.optimizer_type
        <OptimizerType.ADAMW: 'adamw'>
    """

    optimizer_type: OptimizerType
    adamw_config: AdamWConfig | None
    lion_config: LionConfig | None
    sophia_config: SophiaConfig | None
    adafactor_config: AdafactorConfig | None


@dataclass(frozen=True, slots=True)
class OptimizerStats:
    """Statistics for optimizer comparison.

    Attributes:
        memory_mb: Memory usage in megabytes.
        convergence_speed: Relative convergence speed.
        stability_score: Training stability score (0-1).
        recommended_for: List of recommended use cases.

    Examples:
        >>> stats = OptimizerStats(
        ...     memory_mb=1000.0,
        ...     convergence_speed=1.0,
        ...     stability_score=0.9,
        ...     recommended_for=("large_models", "fine_tuning"),
        ... )
        >>> stats.memory_mb
        1000.0
    """

    memory_mb: float
    convergence_speed: float
    stability_score: float
    recommended_for: tuple[str, ...]


def validate_adamw_config(config: AdamWConfig) -> None:
    """Validate AdamW configuration.

    Args:
        config: AdamW configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If lr is not positive.
        ValueError: If betas are not in (0, 1).
        ValueError: If eps is not positive.
        ValueError: If weight_decay is negative.

    Examples:
        >>> config = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        >>> validate_adamw_config(config)  # No error

        >>> validate_adamw_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AdamWConfig(0.0, (0.9, 0.999), 1e-8, 0.01, False)
        >>> validate_adamw_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: lr must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.lr <= 0:
        msg = f"lr must be positive, got {config.lr}"
        raise ValueError(msg)

    beta1, beta2 = config.betas
    if not 0 < beta1 < 1:
        msg = f"beta1 must be in (0, 1), got {beta1}"
        raise ValueError(msg)

    if not 0 < beta2 < 1:
        msg = f"beta2 must be in (0, 1), got {beta2}"
        raise ValueError(msg)

    if config.eps <= 0:
        msg = f"eps must be positive, got {config.eps}"
        raise ValueError(msg)

    if config.weight_decay < 0:
        msg = f"weight_decay cannot be negative, got {config.weight_decay}"
        raise ValueError(msg)


def validate_lion_config(config: LionConfig) -> None:
    """Validate Lion configuration.

    Args:
        config: Lion configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If lr is not positive.
        ValueError: If betas are not in (0, 1).
        ValueError: If weight_decay is negative.

    Examples:
        >>> config = LionConfig(1e-4, (0.9, 0.99), 0.01)
        >>> validate_lion_config(config)  # No error

        >>> validate_lion_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LionConfig(-0.001, (0.9, 0.99), 0.01)
        >>> validate_lion_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: lr must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.lr <= 0:
        msg = f"lr must be positive, got {config.lr}"
        raise ValueError(msg)

    beta1, beta2 = config.betas
    if not 0 < beta1 < 1:
        msg = f"beta1 must be in (0, 1), got {beta1}"
        raise ValueError(msg)

    if not 0 < beta2 < 1:
        msg = f"beta2 must be in (0, 1), got {beta2}"
        raise ValueError(msg)

    if config.weight_decay < 0:
        msg = f"weight_decay cannot be negative, got {config.weight_decay}"
        raise ValueError(msg)


def validate_sophia_config(config: SophiaConfig) -> None:
    """Validate Sophia configuration.

    Args:
        config: Sophia configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If lr is not positive.
        ValueError: If betas are not in (0, 1).
        ValueError: If rho is not positive.
        ValueError: If weight_decay is negative.

    Examples:
        >>> config = SophiaConfig(1e-4, (0.965, 0.99), 0.04, 0.1)
        >>> validate_sophia_config(config)  # No error

        >>> validate_sophia_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SophiaConfig(1e-4, (0.965, 0.99), 0.0, 0.1)
        >>> validate_sophia_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rho must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.lr <= 0:
        msg = f"lr must be positive, got {config.lr}"
        raise ValueError(msg)

    beta1, beta2 = config.betas
    if not 0 < beta1 < 1:
        msg = f"beta1 must be in (0, 1), got {beta1}"
        raise ValueError(msg)

    if not 0 < beta2 < 1:
        msg = f"beta2 must be in (0, 1), got {beta2}"
        raise ValueError(msg)

    if config.rho <= 0:
        msg = f"rho must be positive, got {config.rho}"
        raise ValueError(msg)

    if config.weight_decay < 0:
        msg = f"weight_decay cannot be negative, got {config.weight_decay}"
        raise ValueError(msg)


def validate_adafactor_config(config: AdafactorConfig) -> None:
    """Validate Adafactor configuration.

    Args:
        config: Adafactor configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If lr is not positive (when not None).
        ValueError: If eps values are not positive.
        ValueError: If clip_threshold is not positive.

    Examples:
        >>> config = AdafactorConfig(None, (1e-30, 1e-3), 1.0, True)
        >>> validate_adafactor_config(config)  # No error

        >>> validate_adafactor_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AdafactorConfig(1e-4, (1e-30, 1e-3), 0.0, True)
        >>> validate_adafactor_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: clip_threshold must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.lr is not None and config.lr <= 0:
        msg = f"lr must be positive when specified, got {config.lr}"
        raise ValueError(msg)

    eps1, eps2 = config.eps
    if eps1 <= 0:
        msg = f"eps1 must be positive, got {eps1}"
        raise ValueError(msg)

    if eps2 <= 0:
        msg = f"eps2 must be positive, got {eps2}"
        raise ValueError(msg)

    if config.clip_threshold <= 0:
        msg = f"clip_threshold must be positive, got {config.clip_threshold}"
        raise ValueError(msg)


def validate_optimizer_config(config: OptimizerConfig) -> None:
    """Validate optimizer configuration.

    Args:
        config: Optimizer configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If required sub-config is missing.

    Examples:
        >>> adamw = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        >>> config = OptimizerConfig(
        ...     OptimizerType.ADAMW, adamw, None, None, None
        ... )
        >>> validate_optimizer_config(config)  # No error

        >>> validate_optimizer_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    adamw_types = {OptimizerType.ADAMW, OptimizerType.ADAM, OptimizerType.PAGED_ADAMW}
    if config.optimizer_type in adamw_types:
        if config.adamw_config is None:
            msg = f"adamw_config required for {config.optimizer_type.value}"
            raise ValueError(msg)
        validate_adamw_config(config.adamw_config)

    if config.optimizer_type == OptimizerType.LION:
        if config.lion_config is None:
            msg = "lion_config required for lion optimizer"
            raise ValueError(msg)
        validate_lion_config(config.lion_config)

    if config.optimizer_type == OptimizerType.SOPHIA:
        if config.sophia_config is None:
            msg = "sophia_config required for sophia optimizer"
            raise ValueError(msg)
        validate_sophia_config(config.sophia_config)

    if config.optimizer_type == OptimizerType.ADAFACTOR:
        if config.adafactor_config is None:
            msg = "adafactor_config required for adafactor optimizer"
            raise ValueError(msg)
        validate_adafactor_config(config.adafactor_config)


def create_adamw_config(
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    amsgrad: bool = False,
) -> AdamWConfig:
    """Create AdamW configuration.

    Args:
        lr: Learning rate. Defaults to 1e-4.
        betas: Beta coefficients. Defaults to (0.9, 0.999).
        eps: Numerical stability term. Defaults to 1e-8.
        weight_decay: Weight decay coefficient. Defaults to 0.01.
        amsgrad: Use AMSGrad variant. Defaults to False.

    Returns:
        Validated AdamWConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_adamw_config()
        >>> config.lr
        0.0001
        >>> config.betas
        (0.9, 0.999)

        >>> config = create_adamw_config(lr=3e-4, weight_decay=0.1)
        >>> config.lr
        0.0003
        >>> config.weight_decay
        0.1

        >>> create_adamw_config(lr=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: lr must be positive
    """
    config = AdamWConfig(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    validate_adamw_config(config)
    return config


def create_lion_config(
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    weight_decay: float = 0.01,
) -> LionConfig:
    """Create Lion configuration.

    Args:
        lr: Learning rate. Defaults to 1e-4.
        betas: Beta coefficients. Defaults to (0.9, 0.99).
        weight_decay: Weight decay coefficient. Defaults to 0.01.

    Returns:
        Validated LionConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_lion_config()
        >>> config.lr
        0.0001
        >>> config.betas
        (0.9, 0.99)

        >>> config = create_lion_config(lr=3e-5, weight_decay=0.0)
        >>> config.lr
        3e-05

        >>> create_lion_config(lr=-0.001)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: lr must be positive
    """
    config = LionConfig(
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    validate_lion_config(config)
    return config


def create_sophia_config(
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.965, 0.99),
    rho: float = 0.04,
    weight_decay: float = 0.1,
) -> SophiaConfig:
    """Create Sophia configuration.

    Args:
        lr: Learning rate. Defaults to 1e-4.
        betas: Beta coefficients. Defaults to (0.965, 0.99).
        rho: Hessian estimation coefficient. Defaults to 0.04.
        weight_decay: Weight decay coefficient. Defaults to 0.1.

    Returns:
        Validated SophiaConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_sophia_config()
        >>> config.lr
        0.0001
        >>> config.rho
        0.04

        >>> config = create_sophia_config(lr=2e-4, rho=0.05)
        >>> config.lr
        0.0002

        >>> create_sophia_config(rho=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rho must be positive
    """
    config = SophiaConfig(
        lr=lr,
        betas=betas,
        rho=rho,
        weight_decay=weight_decay,
    )
    validate_sophia_config(config)
    return config


def create_adafactor_config(
    lr: float | None = None,
    eps: tuple[float, float] = (1e-30, 1e-3),
    clip_threshold: float = 1.0,
    scale_parameter: bool = True,
) -> AdafactorConfig:
    """Create Adafactor configuration.

    Args:
        lr: Learning rate (None for relative step). Defaults to None.
        eps: Regularization constants. Defaults to (1e-30, 1e-3).
        clip_threshold: Gradient clipping threshold. Defaults to 1.0.
        scale_parameter: Scale parameter learning rate. Defaults to True.

    Returns:
        Validated AdafactorConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_adafactor_config()
        >>> config.lr is None
        True
        >>> config.scale_parameter
        True

        >>> config = create_adafactor_config(lr=1e-3)
        >>> config.lr
        0.001

        >>> create_adafactor_config(clip_threshold=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: clip_threshold must be positive
    """
    config = AdafactorConfig(
        lr=lr,
        eps=eps,
        clip_threshold=clip_threshold,
        scale_parameter=scale_parameter,
    )
    validate_adafactor_config(config)
    return config


def create_optimizer_config(
    optimizer_type: str = "adamw",
    adamw_config: AdamWConfig | None = None,
    lion_config: LionConfig | None = None,
    sophia_config: SophiaConfig | None = None,
    adafactor_config: AdafactorConfig | None = None,
) -> OptimizerConfig:
    """Create optimizer configuration.

    Args:
        optimizer_type: Type of optimizer. Defaults to "adamw".
        adamw_config: AdamW configuration. Defaults to None.
        lion_config: Lion configuration. Defaults to None.
        sophia_config: Sophia configuration. Defaults to None.
        adafactor_config: Adafactor configuration. Defaults to None.

    Returns:
        Validated OptimizerConfig instance.

    Raises:
        ValueError: If optimizer_type is invalid.
        ValueError: If required sub-config is missing.

    Examples:
        >>> config = create_optimizer_config()
        >>> config.optimizer_type
        <OptimizerType.ADAMW: 'adamw'>

        >>> lion = create_lion_config()
        >>> config = create_optimizer_config("lion", lion_config=lion)
        >>> config.optimizer_type
        <OptimizerType.LION: 'lion'>

        >>> create_optimizer_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: optimizer_type must be one of
    """
    if optimizer_type not in VALID_OPTIMIZER_TYPES:
        msg = (
            f"optimizer_type must be one of {VALID_OPTIMIZER_TYPES}, "
            f"got '{optimizer_type}'"
        )
        raise ValueError(msg)

    opt_type = OptimizerType(optimizer_type)

    # Create default configs if needed
    adamw_types = {OptimizerType.ADAMW, OptimizerType.ADAM, OptimizerType.PAGED_ADAMW}
    if opt_type in adamw_types and adamw_config is None:
        adamw_config = create_adamw_config()

    if opt_type == OptimizerType.LION and lion_config is None:
        lion_config = create_lion_config()

    if opt_type == OptimizerType.SOPHIA and sophia_config is None:
        sophia_config = create_sophia_config()

    if opt_type == OptimizerType.ADAFACTOR and adafactor_config is None:
        adafactor_config = create_adafactor_config()

    if opt_type == OptimizerType.ADAM_8BIT and adamw_config is None:
        adamw_config = create_adamw_config()

    if opt_type == OptimizerType.SGD and adamw_config is None:
        # SGD uses similar config structure (betas represent momentum)
        # Use small valid beta2 value since SGD doesn't use second moment
        adamw_config = create_adamw_config(
            betas=(0.9, 0.001), eps=1e-8, weight_decay=0.0
        )

    config = OptimizerConfig(
        optimizer_type=opt_type,
        adamw_config=adamw_config,
        lion_config=lion_config,
        sophia_config=sophia_config,
        adafactor_config=adafactor_config,
    )
    validate_optimizer_config(config)
    return config


def calculate_optimizer_memory(
    model_params: int,
    optimizer_type: str = "adamw",
    precision: str = "fp32",
) -> float:
    """Calculate optimizer state memory usage.

    Args:
        model_params: Number of model parameters.
        optimizer_type: Type of optimizer. Defaults to "adamw".
        precision: Precision type ("fp32", "fp16", "8bit"). Defaults to "fp32".

    Returns:
        Memory usage in megabytes.

    Raises:
        ValueError: If model_params is not positive.
        ValueError: If optimizer_type is invalid.

    Examples:
        >>> mem = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        >>> mem > 0
        True

        >>> mem_8bit = calculate_optimizer_memory(1_000_000, "adam_8bit", "8bit")
        >>> mem_full = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        >>> mem_8bit < mem_full
        True

        >>> calculate_optimizer_memory(0, "adamw")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = f"model_params must be positive, got {model_params}"
        raise ValueError(msg)

    if optimizer_type not in VALID_OPTIMIZER_TYPES:
        msg = (
            f"optimizer_type must be one of {VALID_OPTIMIZER_TYPES}, "
            f"got '{optimizer_type}'"
        )
        raise ValueError(msg)

    # Bytes per element
    bytes_per_param = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "8bit": 1.0,
    }
    param_bytes = bytes_per_param.get(precision, 4.0)

    # State multiplier based on optimizer
    # AdamW/Adam stores m (first moment) and v (second moment)
    state_multipliers = {
        "adamw": 2.0,  # m + v
        "adam": 2.0,  # m + v
        "sgd": 1.0,  # momentum only
        "lion": 1.0,  # single momentum buffer
        "sophia": 2.0,  # m + h (Hessian)
        "adafactor": 0.5,  # row + col factors instead of full matrices
        "adam_8bit": 2.0,  # same as adam but 8-bit
        "paged_adamw": 2.0,  # same as adamw
    }
    multiplier = state_multipliers.get(optimizer_type, 2.0)

    # Calculate memory in bytes, convert to MB
    memory_bytes = model_params * param_bytes * multiplier
    memory_mb = memory_bytes / (1024 * 1024)

    return round(memory_mb, 2)


def estimate_convergence_speed(
    optimizer_type: str,
    model_size: str = "7b",
    task_type: str = "fine_tuning",
) -> float:
    """Estimate relative convergence speed for an optimizer.

    Args:
        optimizer_type: Type of optimizer.
        model_size: Model size ("small", "7b", "13b", "70b").
            Defaults to "7b".
        task_type: Type of task ("pretraining", "fine_tuning").
            Defaults to "fine_tuning".

    Returns:
        Relative convergence speed (1.0 = baseline AdamW).

    Raises:
        ValueError: If optimizer_type is invalid.

    Examples:
        >>> speed = estimate_convergence_speed("adamw", "7b", "fine_tuning")
        >>> speed >= 1.0
        True

        >>> speed = estimate_convergence_speed("sophia", "7b", "pretraining")
        >>> speed > 1.0
        True

        >>> estimate_convergence_speed("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: optimizer_type must be one of
    """
    if optimizer_type not in VALID_OPTIMIZER_TYPES:
        msg = (
            f"optimizer_type must be one of {VALID_OPTIMIZER_TYPES}, "
            f"got '{optimizer_type}'"
        )
        raise ValueError(msg)

    # Base convergence speeds (relative to AdamW = 1.0)
    base_speeds = {
        "adamw": 1.0,
        "adam": 1.0,
        "sgd": 0.7,  # Often slower convergence
        "lion": 1.1,  # Slightly faster in some cases
        "sophia": 1.5,  # Significantly faster with Hessian info
        "adafactor": 0.9,  # Slightly slower but memory efficient
        "adam_8bit": 0.95,  # Slightly slower due to quantization
        "paged_adamw": 1.0,  # Same as adamw
    }
    base_speed = base_speeds[optimizer_type]

    # Model size adjustments
    size_adjustments = {
        "small": 1.1,
        "7b": 1.0,
        "13b": 0.95,
        "70b": 0.9,
    }
    size_factor = size_adjustments.get(model_size.lower(), 1.0)

    # Task type adjustments
    task_adjustments = {
        "pretraining": 1.0,
        "fine_tuning": 1.1,  # Generally faster convergence
        "rlhf": 0.8,  # More complex optimization
    }
    task_factor = task_adjustments.get(task_type.lower(), 1.0)

    # Sophia benefits more from pretraining
    if optimizer_type == "sophia" and task_type == "pretraining":
        task_factor *= 1.2

    speed = base_speed * size_factor * task_factor
    return round(speed, 2)


def compare_optimizers(
    optimizer_types: tuple[str, ...],
    model_params: int = 7_000_000_000,
) -> dict[str, OptimizerStats]:
    """Compare multiple optimizers.

    Args:
        optimizer_types: Tuple of optimizer types to compare.
        model_params: Number of model parameters. Defaults to 7B.

    Returns:
        Dictionary mapping optimizer type to stats.

    Raises:
        ValueError: If any optimizer_type is invalid.
        ValueError: If model_params is not positive.

    Examples:
        >>> stats = compare_optimizers(("adamw", "lion"))
        >>> "adamw" in stats
        True
        >>> "lion" in stats
        True
        >>> stats["adamw"].memory_mb > 0
        True

        >>> compare_optimizers(("invalid",))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: optimizer_type must be one of
    """
    if model_params <= 0:
        msg = f"model_params must be positive, got {model_params}"
        raise ValueError(msg)

    results = {}

    # Recommended use cases for each optimizer
    recommended_cases = {
        "adamw": ("general_purpose", "fine_tuning", "classification"),
        "adam": ("general_purpose", "small_models"),
        "sgd": ("computer_vision", "contrastive_learning"),
        "lion": ("large_models", "language_modeling"),
        "sophia": ("pretraining", "large_models"),
        "adafactor": ("memory_constrained", "large_models"),
        "adam_8bit": ("memory_constrained", "fine_tuning"),
        "paged_adamw": ("very_large_models", "limited_memory"),
    }

    # Stability scores (empirical estimates)
    stability_scores = {
        "adamw": 0.95,
        "adam": 0.9,
        "sgd": 0.85,
        "lion": 0.88,
        "sophia": 0.82,  # Can be less stable
        "adafactor": 0.87,
        "adam_8bit": 0.85,
        "paged_adamw": 0.93,
    }

    for opt_type in optimizer_types:
        if opt_type not in VALID_OPTIMIZER_TYPES:
            msg = (
                f"optimizer_type must be one of {VALID_OPTIMIZER_TYPES}, "
                f"got '{opt_type}'"
            )
            raise ValueError(msg)

        precision = "8bit" if opt_type == "adam_8bit" else "fp32"
        memory = calculate_optimizer_memory(model_params, opt_type, precision)
        speed = estimate_convergence_speed(opt_type)
        stability = stability_scores.get(opt_type, 0.9)
        recommended = recommended_cases.get(opt_type, ("general_purpose",))

        results[opt_type] = OptimizerStats(
            memory_mb=memory,
            convergence_speed=speed,
            stability_score=stability,
            recommended_for=recommended,
        )

    return results


def get_param_groups(
    model_params: int,
    optimizer_type: str = "adamw",
    weight_decay: float = 0.01,
    no_decay_keywords: tuple[str, ...] = ("bias", "LayerNorm", "layer_norm"),
) -> list[dict[str, float | tuple[str, ...]]]:
    """Get parameter groups for optimizer.

    Args:
        model_params: Number of model parameters.
        optimizer_type: Type of optimizer. Defaults to "adamw".
        weight_decay: Weight decay for decayed params. Defaults to 0.01.
        no_decay_keywords: Keywords for params without decay.
            Defaults to ("bias", "LayerNorm", "layer_norm").

    Returns:
        List of parameter group dictionaries.

    Raises:
        ValueError: If model_params is not positive.
        ValueError: If weight_decay is negative.

    Examples:
        >>> groups = get_param_groups(1_000_000, "adamw", 0.01)
        >>> len(groups)
        2
        >>> groups[0]["weight_decay"]
        0.01
        >>> groups[1]["weight_decay"]
        0.0

        >>> get_param_groups(0, "adamw")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = f"model_params must be positive, got {model_params}"
        raise ValueError(msg)

    if weight_decay < 0:
        msg = f"weight_decay cannot be negative, got {weight_decay}"
        raise ValueError(msg)

    # Return standard parameter group structure
    return [
        {
            "weight_decay": weight_decay,
            "no_decay_keywords": (),
        },
        {
            "weight_decay": 0.0,
            "no_decay_keywords": no_decay_keywords,
        },
    ]


def format_optimizer_stats(stats: OptimizerStats) -> str:
    """Format optimizer statistics for display.

    Args:
        stats: Optimizer statistics to format.

    Returns:
        Formatted string with statistics.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = OptimizerStats(1000.0, 1.0, 0.9, ("fine_tuning",))
        >>> formatted = format_optimizer_stats(stats)
        >>> "Memory:" in formatted
        True
        >>> "Convergence Speed:" in formatted
        True

        >>> format_optimizer_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    recommended = ", ".join(stats.recommended_for)
    lines = [
        f"Memory: {stats.memory_mb:,.2f} MB",
        f"Convergence Speed: {stats.convergence_speed:.2f}x",
        f"Stability Score: {stats.stability_score:.2f}",
        f"Recommended For: {recommended}",
    ]
    return "\n".join(lines)


def get_recommended_optimizer_config(
    model_size: str = "7b",
    task_type: str = "fine_tuning",
    memory_constrained: bool = False,
) -> OptimizerConfig:
    """Get recommended optimizer configuration.

    Args:
        model_size: Model size ("small", "7b", "13b", "70b").
            Defaults to "7b".
        task_type: Type of task ("pretraining", "fine_tuning").
            Defaults to "fine_tuning".
        memory_constrained: Whether memory is limited. Defaults to False.

    Returns:
        Recommended OptimizerConfig.

    Raises:
        ValueError: If model_size is invalid.

    Examples:
        >>> config = get_recommended_optimizer_config("7b", "fine_tuning")
        >>> config.optimizer_type in (OptimizerType.ADAMW, OptimizerType.LION)
        True

        >>> config = get_recommended_optimizer_config("70b", memory_constrained=True)
        >>> config.optimizer_type in (
        ...     OptimizerType.ADAFACTOR,
        ...     OptimizerType.ADAM_8BIT,
        ...     OptimizerType.PAGED_ADAMW,
        ... )
        True

        >>> get_recommended_optimizer_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size must be one of
    """
    valid_sizes = {"small", "7b", "13b", "70b"}
    model_size_lower = model_size.lower()
    if model_size_lower not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    task_type_lower = task_type.lower()

    # Selection logic
    if memory_constrained:
        if model_size_lower in ("70b", "13b"):
            # Very large models with memory constraints
            optimizer_type = "paged_adamw"
            adamw_config = create_adamw_config(lr=1e-5, weight_decay=0.1)
        else:
            # Smaller models with memory constraints
            optimizer_type = "adam_8bit"
            adamw_config = create_adamw_config(lr=2e-5, weight_decay=0.01)

        return create_optimizer_config(optimizer_type, adamw_config=adamw_config)

    if task_type_lower == "pretraining":
        if model_size_lower in ("70b", "13b"):
            # Large model pretraining benefits from Sophia
            optimizer_type = "sophia"
            sophia_config = create_sophia_config(lr=1e-4, rho=0.03)
            return create_optimizer_config(optimizer_type, sophia_config=sophia_config)
        else:
            # Standard AdamW for smaller pretraining
            optimizer_type = "adamw"
            adamw_config = create_adamw_config(lr=1e-4, weight_decay=0.01)
            return create_optimizer_config(optimizer_type, adamw_config=adamw_config)

    # Fine-tuning defaults
    if model_size_lower in ("70b", "13b"):
        # Lion works well for large model fine-tuning
        optimizer_type = "lion"
        lion_config = create_lion_config(lr=3e-5, weight_decay=0.01)
        return create_optimizer_config(optimizer_type, lion_config=lion_config)

    # Default: AdamW for smaller fine-tuning
    optimizer_type = "adamw"
    adamw_config = create_adamw_config(lr=2e-5, weight_decay=0.01)
    return create_optimizer_config(optimizer_type, adamw_config=adamw_config)


def list_optimizer_types() -> list[str]:
    """List all supported optimizer types.

    Returns:
        Sorted list of optimizer type names.

    Examples:
        >>> types = list_optimizer_types()
        >>> "adamw" in types
        True
        >>> "lion" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_OPTIMIZER_TYPES)


def list_weight_decay_types() -> list[str]:
    """List all supported weight decay types.

    Returns:
        Sorted list of weight decay type names.

    Examples:
        >>> types = list_weight_decay_types()
        >>> "decoupled" in types
        True
        >>> "l2" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_WEIGHT_DECAY_TYPES)


def list_momentum_types() -> list[str]:
    """List all supported momentum types.

    Returns:
        Sorted list of momentum type names.

    Examples:
        >>> types = list_momentum_types()
        >>> "standard" in types
        True
        >>> "nesterov" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_MOMENTUM_TYPES)


def get_optimizer_type(name: str) -> OptimizerType:
    """Get optimizer type from name.

    Args:
        name: Optimizer type name.

    Returns:
        OptimizerType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_optimizer_type("adamw")
        <OptimizerType.ADAMW: 'adamw'>

        >>> get_optimizer_type("lion")
        <OptimizerType.LION: 'lion'>

        >>> get_optimizer_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: optimizer type must be one of
    """
    if name not in VALID_OPTIMIZER_TYPES:
        msg = f"optimizer type must be one of {VALID_OPTIMIZER_TYPES}, got '{name}'"
        raise ValueError(msg)
    return OptimizerType(name)


def get_weight_decay_type(name: str) -> WeightDecayType:
    """Get weight decay type from name.

    Args:
        name: Weight decay type name.

    Returns:
        WeightDecayType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_weight_decay_type("decoupled")
        <WeightDecayType.DECOUPLED: 'decoupled'>

        >>> get_weight_decay_type("l2")
        <WeightDecayType.L2: 'l2'>

        >>> get_weight_decay_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: weight decay type must be one of
    """
    if name not in VALID_WEIGHT_DECAY_TYPES:
        msg = (
            f"weight decay type must be one of {VALID_WEIGHT_DECAY_TYPES}, got '{name}'"
        )
        raise ValueError(msg)
    return WeightDecayType(name)


def get_momentum_type(name: str) -> MomentumType:
    """Get momentum type from name.

    Args:
        name: Momentum type name.

    Returns:
        MomentumType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_momentum_type("standard")
        <MomentumType.STANDARD: 'standard'>

        >>> get_momentum_type("nesterov")
        <MomentumType.NESTEROV: 'nesterov'>

        >>> get_momentum_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: momentum type must be one of
    """
    if name not in VALID_MOMENTUM_TYPES:
        msg = f"momentum type must be one of {VALID_MOMENTUM_TYPES}, got '{name}'"
        raise ValueError(msg)
    return MomentumType(name)
