"""Custom loss functions for training ML models.

This module provides utilities for computing various loss functions including
focal loss for imbalanced classification, contrastive learning losses, and
label smoothing for regularization.

Examples:
    >>> from hf_gtc.training.losses import (
    ...     create_focal_loss_config,
    ...     LossType,
    ... )
    >>> config = create_focal_loss_config()
    >>> config.gamma
    2.0
    >>> LossType.FOCAL.value
    'focal'
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class LossType(Enum):
    """Types of loss functions.

    Attributes:
        CROSS_ENTROPY: Standard cross-entropy loss.
        FOCAL: Focal loss for imbalanced classification.
        CONTRASTIVE: Contrastive learning loss.
        TRIPLET: Triplet margin loss.
        COSINE: Cosine embedding loss.
        KL_DIVERGENCE: Kullback-Leibler divergence.
        MSE: Mean squared error loss.
        MAE: Mean absolute error loss.

    Examples:
        >>> LossType.CROSS_ENTROPY.value
        'cross_entropy'
        >>> LossType.FOCAL.value
        'focal'
        >>> LossType.CONTRASTIVE.value
        'contrastive'
    """

    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    COSINE = "cosine"
    KL_DIVERGENCE = "kl_divergence"
    MSE = "mse"
    MAE = "mae"


class ReductionType(Enum):
    """Types of loss reduction.

    Attributes:
        MEAN: Compute mean of losses.
        SUM: Compute sum of losses.
        NONE: No reduction, return per-sample losses.
        BATCHMEAN: Sum divided by batch size (for KL divergence).

    Examples:
        >>> ReductionType.MEAN.value
        'mean'
        >>> ReductionType.SUM.value
        'sum'
        >>> ReductionType.NONE.value
        'none'
        >>> ReductionType.BATCHMEAN.value
        'batchmean'
    """

    MEAN = "mean"
    SUM = "sum"
    NONE = "none"
    BATCHMEAN = "batchmean"


class ContrastiveType(Enum):
    """Types of contrastive loss.

    Attributes:
        SIMCLR: SimCLR contrastive loss.
        INFONCE: InfoNCE loss.
        TRIPLET: Triplet margin loss.
        NTXENT: Normalized temperature-scaled cross entropy.

    Examples:
        >>> ContrastiveType.SIMCLR.value
        'simclr'
        >>> ContrastiveType.INFONCE.value
        'infonce'
        >>> ContrastiveType.TRIPLET.value
        'triplet'
        >>> ContrastiveType.NTXENT.value
        'ntxent'
    """

    SIMCLR = "simclr"
    INFONCE = "infonce"
    TRIPLET = "triplet"
    NTXENT = "ntxent"


VALID_LOSS_TYPES = frozenset(lt.value for lt in LossType)
VALID_REDUCTION_TYPES = frozenset(rt.value for rt in ReductionType)
VALID_CONTRASTIVE_TYPES = frozenset(ct.value for ct in ContrastiveType)


@dataclass(frozen=True, slots=True)
class FocalLossConfig:
    """Configuration for focal loss.

    Attributes:
        alpha: Weighting factor for positive class (0.0-1.0).
        gamma: Focusing parameter (0.0 = cross-entropy, 2.0 typical).
        reduction: How to reduce the loss.

    Examples:
        >>> config = FocalLossConfig(
        ...     alpha=0.25,
        ...     gamma=2.0,
        ...     reduction=ReductionType.MEAN,
        ... )
        >>> config.alpha
        0.25
        >>> config.gamma
        2.0
        >>> config.reduction
        <ReductionType.MEAN: 'mean'>

        >>> config2 = FocalLossConfig(
        ...     alpha=0.5,
        ...     gamma=1.0,
        ...     reduction=ReductionType.SUM,
        ... )
        >>> config2.gamma
        1.0
    """

    alpha: float
    gamma: float
    reduction: ReductionType


@dataclass(frozen=True, slots=True)
class ContrastiveLossConfig:
    """Configuration for contrastive loss.

    Attributes:
        temperature: Temperature scaling parameter.
        margin: Margin for triplet loss.
        contrastive_type: Type of contrastive loss.

    Examples:
        >>> config = ContrastiveLossConfig(
        ...     temperature=0.07,
        ...     margin=1.0,
        ...     contrastive_type=ContrastiveType.SIMCLR,
        ... )
        >>> config.temperature
        0.07
        >>> config.margin
        1.0
        >>> config.contrastive_type
        <ContrastiveType.SIMCLR: 'simclr'>

        >>> config2 = ContrastiveLossConfig(
        ...     temperature=0.1,
        ...     margin=0.5,
        ...     contrastive_type=ContrastiveType.TRIPLET,
        ... )
        >>> config2.margin
        0.5
    """

    temperature: float
    margin: float
    contrastive_type: ContrastiveType


@dataclass(frozen=True, slots=True)
class LabelSmoothingConfig:
    """Configuration for label smoothing.

    Attributes:
        smoothing: Smoothing factor (0.0-1.0, typically 0.1).
        reduction: How to reduce the loss.

    Examples:
        >>> config = LabelSmoothingConfig(
        ...     smoothing=0.1,
        ...     reduction=ReductionType.MEAN,
        ... )
        >>> config.smoothing
        0.1
        >>> config.reduction
        <ReductionType.MEAN: 'mean'>

        >>> config2 = LabelSmoothingConfig(
        ...     smoothing=0.2,
        ...     reduction=ReductionType.SUM,
        ... )
        >>> config2.smoothing
        0.2
    """

    smoothing: float
    reduction: ReductionType


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Combined loss configuration.

    Attributes:
        loss_type: Type of loss function.
        focal_config: Focal loss configuration (if applicable).
        contrastive_config: Contrastive loss configuration (if applicable).
        label_smoothing_config: Label smoothing configuration (if applicable).
        weight: Class weights for weighted loss.

    Examples:
        >>> focal = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        >>> config = LossConfig(
        ...     loss_type=LossType.FOCAL,
        ...     focal_config=focal,
        ...     contrastive_config=None,
        ...     label_smoothing_config=None,
        ...     weight=None,
        ... )
        >>> config.loss_type
        <LossType.FOCAL: 'focal'>
        >>> config.focal_config is not None
        True

        >>> config2 = LossConfig(
        ...     loss_type=LossType.CROSS_ENTROPY,
        ...     focal_config=None,
        ...     contrastive_config=None,
        ...     label_smoothing_config=None,
        ...     weight=(1.0, 2.0, 1.5),
        ... )
        >>> config2.weight
        (1.0, 2.0, 1.5)
    """

    loss_type: LossType
    focal_config: FocalLossConfig | None
    contrastive_config: ContrastiveLossConfig | None
    label_smoothing_config: LabelSmoothingConfig | None
    weight: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class LossStats:
    """Statistics from loss computation.

    Attributes:
        loss_value: Computed loss value.
        num_samples: Number of samples in batch.
        per_class_loss: Per-class loss values (if applicable).

    Examples:
        >>> stats = LossStats(
        ...     loss_value=0.5,
        ...     num_samples=32,
        ...     per_class_loss=(0.3, 0.5, 0.7),
        ... )
        >>> stats.loss_value
        0.5
        >>> stats.num_samples
        32
        >>> stats.per_class_loss
        (0.3, 0.5, 0.7)

        >>> stats2 = LossStats(
        ...     loss_value=0.25,
        ...     num_samples=64,
        ...     per_class_loss=None,
        ... )
        >>> stats2.num_samples
        64
    """

    loss_value: float
    num_samples: int
    per_class_loss: tuple[float, ...] | None


def validate_focal_loss_config(config: FocalLossConfig) -> None:
    """Validate focal loss configuration.

    Args:
        config: Focal loss configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If alpha is not in [0, 1].
        ValueError: If gamma is negative.

    Examples:
        >>> config = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        >>> validate_focal_loss_config(config)

        >>> validate_focal_loss_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = FocalLossConfig(-0.1, 2.0, ReductionType.MEAN)
        >>> validate_focal_loss_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: alpha must be between 0.0 and 1.0
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.alpha <= 1.0:
        msg = f"alpha must be between 0.0 and 1.0, got {config.alpha}"
        raise ValueError(msg)

    if config.gamma < 0:
        msg = f"gamma cannot be negative, got {config.gamma}"
        raise ValueError(msg)


def validate_contrastive_loss_config(config: ContrastiveLossConfig) -> None:
    """Validate contrastive loss configuration.

    Args:
        config: Contrastive loss configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If temperature is not positive.
        ValueError: If margin is negative.

    Examples:
        >>> config = ContrastiveLossConfig(0.07, 1.0, ContrastiveType.SIMCLR)
        >>> validate_contrastive_loss_config(config)

        >>> validate_contrastive_loss_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ContrastiveLossConfig(0.0, 1.0, ContrastiveType.SIMCLR)
        >>> validate_contrastive_loss_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive
    """
    validate_not_none(config, "config")

    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)

    if config.margin < 0:
        msg = f"margin cannot be negative, got {config.margin}"
        raise ValueError(msg)


def validate_label_smoothing_config(config: LabelSmoothingConfig) -> None:
    """Validate label smoothing configuration.

    Args:
        config: Label smoothing configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If smoothing is not in [0, 1].

    Examples:
        >>> config = LabelSmoothingConfig(0.1, ReductionType.MEAN)
        >>> validate_label_smoothing_config(config)

        >>> validate_label_smoothing_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LabelSmoothingConfig(1.5, ReductionType.MEAN)
        >>> validate_label_smoothing_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: smoothing must be between 0.0 and 1.0
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.smoothing <= 1.0:
        msg = f"smoothing must be between 0.0 and 1.0, got {config.smoothing}"
        raise ValueError(msg)


def _validate_loss_type_config(config: LossConfig) -> None:
    """Validate that the correct sub-config is present for the loss type."""
    loss_type_validators: dict[
        LossType,
        tuple[str, str, Callable[[Any], None]],
    ] = {
        LossType.FOCAL: ("focal_config", "focal loss", validate_focal_loss_config),
        LossType.CONTRASTIVE: (
            "contrastive_config",
            "contrastive loss",
            validate_contrastive_loss_config,
        ),
        LossType.TRIPLET: (
            "contrastive_config",
            "triplet loss",
            validate_contrastive_loss_config,
        ),
    }
    entry = loss_type_validators.get(config.loss_type)
    if entry is None:
        return
    attr_name, label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} required for {label}"
        raise ValueError(msg)
    validator(sub_config)


def validate_loss_config(config: LossConfig) -> None:
    """Validate loss configuration.

    Args:
        config: Loss configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If required sub-config is missing.
        ValueError: If weight contains negative values.

    Examples:
        >>> focal = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        >>> config = LossConfig(LossType.FOCAL, focal, None, None, None)
        >>> validate_loss_config(config)

        >>> validate_loss_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LossConfig(LossType.FOCAL, None, None, None, None)
        >>> validate_loss_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: focal_config required for focal loss
    """
    validate_not_none(config, "config")

    _validate_loss_type_config(config)

    if config.weight is not None:
        for w in config.weight:
            if w < 0:
                msg = f"weight cannot contain negative values, got {w}"
                raise ValueError(msg)


def validate_loss_stats(stats: LossStats) -> None:
    """Validate loss statistics.

    Args:
        stats: Loss statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If num_samples is not positive.

    Examples:
        >>> stats = LossStats(0.5, 32, None)
        >>> validate_loss_stats(stats)

        >>> validate_loss_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = LossStats(0.5, 0, None)
        >>> validate_loss_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples must be positive
    """
    validate_not_none(stats, "stats")

    if stats.num_samples <= 0:
        msg = f"num_samples must be positive, got {stats.num_samples}"
        raise ValueError(msg)


def create_focal_loss_config(
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str | ReductionType = ReductionType.MEAN,
) -> FocalLossConfig:
    """Create focal loss configuration.

    Args:
        alpha: Weighting factor for positive class. Defaults to 0.25.
        gamma: Focusing parameter. Defaults to 2.0.
        reduction: How to reduce the loss. Defaults to "mean".

    Returns:
        Validated FocalLossConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_focal_loss_config()
        >>> config.alpha
        0.25
        >>> config.gamma
        2.0
        >>> config.reduction
        <ReductionType.MEAN: 'mean'>

        >>> config2 = create_focal_loss_config(alpha=0.5, gamma=1.0)
        >>> config2.alpha
        0.5

        >>> create_focal_loss_config(alpha=-0.1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: alpha must be between 0.0 and 1.0
    """
    if isinstance(reduction, str):
        reduction = get_reduction_type(reduction)

    config = FocalLossConfig(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
    )
    validate_focal_loss_config(config)
    return config


def create_contrastive_loss_config(
    temperature: float = 0.07,
    margin: float = 1.0,
    contrastive_type: str | ContrastiveType = ContrastiveType.SIMCLR,
) -> ContrastiveLossConfig:
    """Create contrastive loss configuration.

    Args:
        temperature: Temperature scaling parameter. Defaults to 0.07.
        margin: Margin for triplet loss. Defaults to 1.0.
        contrastive_type: Type of contrastive loss. Defaults to "simclr".

    Returns:
        Validated ContrastiveLossConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_contrastive_loss_config()
        >>> config.temperature
        0.07
        >>> config.margin
        1.0
        >>> config.contrastive_type
        <ContrastiveType.SIMCLR: 'simclr'>

        >>> config2 = create_contrastive_loss_config(
        ...     temperature=0.1,
        ...     contrastive_type="ntxent"
        ... )
        >>> config2.temperature
        0.1
        >>> config2.contrastive_type
        <ContrastiveType.NTXENT: 'ntxent'>

        >>> create_contrastive_loss_config(temperature=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive
    """
    if isinstance(contrastive_type, str):
        contrastive_type = get_contrastive_type(contrastive_type)

    config = ContrastiveLossConfig(
        temperature=temperature,
        margin=margin,
        contrastive_type=contrastive_type,
    )
    validate_contrastive_loss_config(config)
    return config


def create_label_smoothing_config(
    smoothing: float = 0.1,
    reduction: str | ReductionType = ReductionType.MEAN,
) -> LabelSmoothingConfig:
    """Create label smoothing configuration.

    Args:
        smoothing: Smoothing factor. Defaults to 0.1.
        reduction: How to reduce the loss. Defaults to "mean".

    Returns:
        Validated LabelSmoothingConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_label_smoothing_config()
        >>> config.smoothing
        0.1
        >>> config.reduction
        <ReductionType.MEAN: 'mean'>

        >>> config2 = create_label_smoothing_config(smoothing=0.2)
        >>> config2.smoothing
        0.2

        >>> create_label_smoothing_config(smoothing=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: smoothing must be between 0.0 and 1.0
    """
    if isinstance(reduction, str):
        reduction = get_reduction_type(reduction)

    config = LabelSmoothingConfig(
        smoothing=smoothing,
        reduction=reduction,
    )
    validate_label_smoothing_config(config)
    return config


def create_loss_config(
    loss_type: str | LossType = LossType.CROSS_ENTROPY,
    focal_config: FocalLossConfig | None = None,
    contrastive_config: ContrastiveLossConfig | None = None,
    label_smoothing_config: LabelSmoothingConfig | None = None,
    weight: tuple[float, ...] | None = None,
) -> LossConfig:
    """Create loss configuration.

    Args:
        loss_type: Type of loss function. Defaults to "cross_entropy".
        focal_config: Focal loss configuration. Defaults to None.
        contrastive_config: Contrastive loss configuration. Defaults to None.
        label_smoothing_config: Label smoothing configuration. Defaults to None.
        weight: Class weights for weighted loss. Defaults to None.

    Returns:
        Validated LossConfig instance.

    Raises:
        ValueError: If loss_type is invalid.
        ValueError: If required sub-config is missing.

    Examples:
        >>> config = create_loss_config()
        >>> config.loss_type
        <LossType.CROSS_ENTROPY: 'cross_entropy'>

        >>> focal = create_focal_loss_config()
        >>> config2 = create_loss_config("focal", focal_config=focal)
        >>> config2.loss_type
        <LossType.FOCAL: 'focal'>

        >>> create_loss_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: loss_type must be one of
    """
    if isinstance(loss_type, str):
        loss_type = get_loss_type(loss_type)

    # Create default configs if needed
    if loss_type == LossType.FOCAL and focal_config is None:
        focal_config = create_focal_loss_config()

    needs_contrastive = loss_type in (LossType.CONTRASTIVE, LossType.TRIPLET)
    if needs_contrastive and contrastive_config is None:
        contrastive_type = (
            ContrastiveType.TRIPLET
            if loss_type == LossType.TRIPLET
            else ContrastiveType.SIMCLR
        )
        contrastive_config = create_contrastive_loss_config(
            contrastive_type=contrastive_type
        )

    config = LossConfig(
        loss_type=loss_type,
        focal_config=focal_config,
        contrastive_config=contrastive_config,
        label_smoothing_config=label_smoothing_config,
        weight=weight,
    )
    validate_loss_config(config)
    return config


def create_loss_stats(
    loss_value: float = 0.0,
    num_samples: int = 1,
    per_class_loss: tuple[float, ...] | None = None,
) -> LossStats:
    """Create loss statistics.

    Args:
        loss_value: Computed loss value. Defaults to 0.0.
        num_samples: Number of samples in batch. Defaults to 1.
        per_class_loss: Per-class loss values. Defaults to None.

    Returns:
        Validated LossStats instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_loss_stats(loss_value=0.5, num_samples=32)
        >>> stats.loss_value
        0.5
        >>> stats.num_samples
        32

        >>> stats2 = create_loss_stats(
        ...     loss_value=0.3,
        ...     num_samples=16,
        ...     per_class_loss=(0.2, 0.3, 0.4)
        ... )
        >>> stats2.per_class_loss
        (0.2, 0.3, 0.4)

        >>> create_loss_stats(num_samples=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples must be positive
    """
    stats = LossStats(
        loss_value=loss_value,
        num_samples=num_samples,
        per_class_loss=per_class_loss,
    )
    validate_loss_stats(stats)
    return stats


def list_loss_types() -> list[str]:
    """List all available loss types.

    Returns:
        Sorted list of loss type names.

    Examples:
        >>> types = list_loss_types()
        >>> "cross_entropy" in types
        True
        >>> "focal" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_LOSS_TYPES)


def list_reduction_types() -> list[str]:
    """List all available reduction types.

    Returns:
        Sorted list of reduction type names.

    Examples:
        >>> types = list_reduction_types()
        >>> "mean" in types
        True
        >>> "sum" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_REDUCTION_TYPES)


def list_contrastive_types() -> list[str]:
    """List all available contrastive loss types.

    Returns:
        Sorted list of contrastive type names.

    Examples:
        >>> types = list_contrastive_types()
        >>> "simclr" in types
        True
        >>> "ntxent" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CONTRASTIVE_TYPES)


def get_loss_type(name: str) -> LossType:
    """Get loss type enum from string name.

    Args:
        name: Name of the loss type.

    Returns:
        Corresponding LossType enum.

    Raises:
        ValueError: If loss type name is invalid.

    Examples:
        >>> get_loss_type("focal")
        <LossType.FOCAL: 'focal'>
        >>> get_loss_type("cross_entropy")
        <LossType.CROSS_ENTROPY: 'cross_entropy'>

        >>> get_loss_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: loss_type must be one of
    """
    if name not in VALID_LOSS_TYPES:
        msg = f"loss_type must be one of {VALID_LOSS_TYPES}, got '{name}'"
        raise ValueError(msg)
    return LossType(name)


def get_reduction_type(name: str) -> ReductionType:
    """Get reduction type enum from string name.

    Args:
        name: Name of the reduction type.

    Returns:
        Corresponding ReductionType enum.

    Raises:
        ValueError: If reduction type name is invalid.

    Examples:
        >>> get_reduction_type("mean")
        <ReductionType.MEAN: 'mean'>
        >>> get_reduction_type("sum")
        <ReductionType.SUM: 'sum'>

        >>> get_reduction_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: reduction_type must be one of
    """
    if name not in VALID_REDUCTION_TYPES:
        msg = f"reduction_type must be one of {VALID_REDUCTION_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ReductionType(name)


def get_contrastive_type(name: str) -> ContrastiveType:
    """Get contrastive type enum from string name.

    Args:
        name: Name of the contrastive type.

    Returns:
        Corresponding ContrastiveType enum.

    Raises:
        ValueError: If contrastive type name is invalid.

    Examples:
        >>> get_contrastive_type("simclr")
        <ContrastiveType.SIMCLR: 'simclr'>
        >>> get_contrastive_type("ntxent")
        <ContrastiveType.NTXENT: 'ntxent'>

        >>> get_contrastive_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: contrastive_type must be one of
    """
    if name not in VALID_CONTRASTIVE_TYPES:
        msg = f"contrastive_type must be one of {VALID_CONTRASTIVE_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ContrastiveType(name)


def calculate_focal_loss(
    predictions: tuple[float, ...],
    targets: tuple[int, ...],
    config: FocalLossConfig,
) -> float:
    """Calculate focal loss for imbalanced classification.

    Focal loss down-weights well-classified examples and focuses on hard,
    misclassified examples. This is useful for imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        predictions: Predicted probabilities for each sample.
        targets: Ground truth labels (0 or 1 for binary).
        config: Focal loss configuration.

    Returns:
        Computed focal loss value.

    Raises:
        ValueError: If predictions and targets have different lengths.
        ValueError: If predictions contains values outside [0, 1].

    Examples:
        >>> config = create_focal_loss_config(alpha=0.25, gamma=2.0)
        >>> loss = calculate_focal_loss((0.9, 0.8, 0.7), (1, 1, 1), config)
        >>> 0 < loss < 1
        True

        >>> loss2 = calculate_focal_loss((0.1, 0.2), (0, 0), config)
        >>> 0 < loss2 < 1
        True

        >>> calculate_focal_loss((0.5,), (1, 0), config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions and targets must have the same length

        >>> calculate_focal_loss((), (), config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if len(predictions) != len(targets):
        msg = "predictions and targets must have the same length"
        raise ValueError(msg)

    if len(predictions) == 0:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    for p in predictions:
        if not 0.0 <= p <= 1.0:
            msg = f"predictions must be in [0, 1], got {p}"
            raise ValueError(msg)

    eps = 1e-7
    losses = []

    for pred, target in zip(predictions, targets, strict=True):
        # p_t = p if target == 1 else 1 - p
        p_t = pred if target == 1 else 1.0 - pred
        p_t = max(p_t, eps)  # Clip to avoid log(0)

        # alpha_t = alpha if target == 1 else 1 - alpha
        alpha_t = config.alpha if target == 1 else 1.0 - config.alpha

        # focal loss
        focal_weight = (1.0 - p_t) ** config.gamma
        loss = -alpha_t * focal_weight * math.log(p_t)
        losses.append(loss)

    if config.reduction == ReductionType.MEAN:
        return sum(losses) / len(losses)
    elif config.reduction == ReductionType.SUM:
        return sum(losses)
    else:
        # Return mean for NONE and BATCHMEAN
        return sum(losses) / len(losses)


def calculate_contrastive_loss(
    anchor_embeddings: tuple[tuple[float, ...], ...],
    positive_embeddings: tuple[tuple[float, ...], ...],
    negative_embeddings: tuple[tuple[float, ...], ...] | None,
    config: ContrastiveLossConfig,
) -> float:
    """Calculate contrastive loss for representation learning.

    Args:
        anchor_embeddings: Anchor embeddings (batch of vectors).
        positive_embeddings: Positive pair embeddings.
        negative_embeddings: Negative pair embeddings (for triplet loss).
        config: Contrastive loss configuration.

    Returns:
        Computed contrastive loss value.

    Raises:
        ValueError: If embeddings have mismatched shapes.
        ValueError: If embeddings are empty.

    Examples:
        >>> config = create_contrastive_loss_config(temperature=0.5)
        >>> anchor = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        >>> positive = ((0.9, 0.1, 0.0), (0.1, 0.9, 0.0))
        >>> loss = calculate_contrastive_loss(anchor, positive, None, config)
        >>> loss >= 0
        True

        >>> triplet_config = create_contrastive_loss_config(
        ...     contrastive_type="triplet", margin=1.0
        ... )
        >>> negative = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        >>> loss2 = calculate_contrastive_loss(
        ...     anchor, positive, negative, triplet_config
        ... )
        >>> loss2 >= 0
        True

        >>> calculate_contrastive_loss((), (), None, config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: embeddings cannot be empty
    """
    if len(anchor_embeddings) == 0:
        msg = "embeddings cannot be empty"
        raise ValueError(msg)

    if len(anchor_embeddings) != len(positive_embeddings):
        msg = "anchor and positive embeddings must have the same batch size"
        raise ValueError(msg)

    if config.contrastive_type == ContrastiveType.TRIPLET:
        if negative_embeddings is None:
            msg = "negative_embeddings required for triplet loss"
            raise ValueError(msg)
        if len(anchor_embeddings) != len(negative_embeddings):
            msg = "anchor and negative embeddings must have the same batch size"
            raise ValueError(msg)

    def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def euclidean_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        """Compute Euclidean distance between two vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))

    if config.contrastive_type == ContrastiveType.TRIPLET:
        # Triplet loss: max(0, d(a, p) - d(a, n) + margin)
        losses = []
        if negative_embeddings is None:
            msg = "negative_embeddings required for triplet loss"
            raise ValueError(msg)
        for anchor, positive, negative in zip(
            anchor_embeddings, positive_embeddings, negative_embeddings, strict=True
        ):
            d_pos = euclidean_distance(anchor, positive)
            d_neg = euclidean_distance(anchor, negative)
            loss = max(0.0, d_pos - d_neg + config.margin)
            losses.append(loss)
        return sum(losses) / len(losses)

    # SimCLR/NT-Xent/InfoNCE style loss
    # For simplicity, compute average over positive pairs with temperature scaling
    losses = []
    for anchor, positive in zip(anchor_embeddings, positive_embeddings, strict=True):
        sim = cosine_similarity(anchor, positive)
        # Scaled negative log probability
        scaled_sim = sim / config.temperature
        # Use simplified version: -log(exp(sim/T) / sum(exp(all_sims/T)))
        # For a single positive pair, this simplifies to -sim/T + log(sum_exp)
        # Here we use a simplified approximation
        loss = -scaled_sim + math.log(math.exp(scaled_sim) + 1)
        losses.append(loss)

    return sum(losses) / len(losses)


def apply_label_smoothing(
    targets: tuple[int, ...],
    num_classes: int,
    config: LabelSmoothingConfig,
) -> tuple[tuple[float, ...], ...]:
    """Apply label smoothing to target labels.

    Label smoothing converts hard labels to soft labels by distributing
    some probability mass to non-target classes.

    Args:
        targets: Hard target labels (class indices).
        num_classes: Total number of classes.
        config: Label smoothing configuration.

    Returns:
        Soft labels as tuples of probabilities.

    Raises:
        ValueError: If targets is empty.
        ValueError: If num_classes is not positive.
        ValueError: If any target is out of range.

    Examples:
        >>> config = create_label_smoothing_config(smoothing=0.1)
        >>> soft_labels = apply_label_smoothing((0, 1, 2), 3, config)
        >>> len(soft_labels)
        3
        >>> len(soft_labels[0])
        3
        >>> soft_labels[0][0] > soft_labels[0][1]  # Target class has higher prob
        True
        >>> abs(sum(soft_labels[0]) - 1.0) < 1e-6  # Sums to 1
        True

        >>> apply_label_smoothing((), 3, config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: targets cannot be empty

        >>> apply_label_smoothing((0,), 0, config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_classes must be positive
    """
    if len(targets) == 0:
        msg = "targets cannot be empty"
        raise ValueError(msg)

    if num_classes <= 0:
        msg = f"num_classes must be positive, got {num_classes}"
        raise ValueError(msg)

    for t in targets:
        if not 0 <= t < num_classes:
            msg = f"target {t} is out of range [0, {num_classes})"
            raise ValueError(msg)

    smoothing = config.smoothing
    confidence = 1.0 - smoothing
    smooth_value = smoothing / num_classes

    result = []
    for target in targets:
        label = [smooth_value] * num_classes
        label[target] = confidence + smooth_value
        result.append(tuple(label))

    return tuple(result)


def compute_class_weights(
    class_counts: tuple[int, ...],
    method: str = "inverse",
) -> tuple[float, ...]:
    """Compute class weights for handling imbalanced datasets.

    Args:
        class_counts: Number of samples per class.
        method: Weighting method ("inverse", "inverse_sqrt", "effective").
            Defaults to "inverse".

    Returns:
        Normalized class weights.

    Raises:
        ValueError: If class_counts is empty.
        ValueError: If any count is not positive.
        ValueError: If method is invalid.

    Examples:
        >>> weights = compute_class_weights((100, 50, 25), "inverse")
        >>> len(weights)
        3
        >>> weights[2] > weights[0]  # Minority class has higher weight
        True

        >>> weights2 = compute_class_weights((100, 100, 100), "inverse")
        >>> all(abs(w - weights2[0]) < 1e-6 for w in weights2)  # Equal weights
        True

        >>> compute_class_weights((), "inverse")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: class_counts cannot be empty

        >>> compute_class_weights((100, 0), "inverse")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: class counts must be positive
    """
    if len(class_counts) == 0:
        msg = "class_counts cannot be empty"
        raise ValueError(msg)

    for count in class_counts:
        if count <= 0:
            msg = f"class counts must be positive, got {count}"
            raise ValueError(msg)

    valid_methods = frozenset({"inverse", "inverse_sqrt", "effective"})
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    total = sum(class_counts)
    num_classes = len(class_counts)

    if method == "inverse":
        # Weight = total / (num_classes * count)
        weights = tuple(total / (num_classes * count) for count in class_counts)
    elif method == "inverse_sqrt":
        # Weight = sqrt(total / (num_classes * count))
        weights = tuple(
            math.sqrt(total / (num_classes * count)) for count in class_counts
        )
    else:  # effective
        # Effective number weighting: (1 - beta^n) / (1 - beta)
        # where beta is typically 0.9999
        beta = 0.9999
        effective_nums = tuple(
            (1.0 - beta**count) / (1.0 - beta) for count in class_counts
        )
        weights = tuple(1.0 / en for en in effective_nums)

    # Normalize weights to sum to num_classes
    weight_sum = sum(weights)
    normalized = tuple(w * num_classes / weight_sum for w in weights)

    return normalized


def format_loss_stats(stats: LossStats) -> str:
    """Format loss statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = create_loss_stats(
        ...     loss_value=0.5,
        ...     num_samples=32,
        ...     per_class_loss=(0.3, 0.5, 0.7),
        ... )
        >>> formatted = format_loss_stats(stats)
        >>> "Loss: 0.5000" in formatted
        True
        >>> "Samples: 32" in formatted
        True
        >>> "Per-class:" in formatted
        True

        >>> format_loss_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        f"Loss: {stats.loss_value:.4f}",
        f"Samples: {stats.num_samples}",
    ]

    if stats.per_class_loss is not None:
        per_class_str = ", ".join(f"{loss:.4f}" for loss in stats.per_class_loss)
        lines.append(f"Per-class: [{per_class_str}]")

    return "\n".join(lines)


def get_recommended_loss_config(task_type: str) -> LossConfig:
    """Get recommended loss configuration for a task type.

    Args:
        task_type: Type of task (classification, imbalanced_classification,
            contrastive_learning, regression, generation).

    Returns:
        Recommended LossConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_loss_config("classification")
        >>> config.loss_type
        <LossType.CROSS_ENTROPY: 'cross_entropy'>

        >>> config2 = get_recommended_loss_config("imbalanced_classification")
        >>> config2.loss_type
        <LossType.FOCAL: 'focal'>
        >>> config2.focal_config is not None
        True

        >>> config3 = get_recommended_loss_config("contrastive_learning")
        >>> config3.loss_type
        <LossType.CONTRASTIVE: 'contrastive'>

        >>> get_recommended_loss_config("unknown")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_type must be one of
    """
    valid_tasks = frozenset(
        {
            "classification",
            "imbalanced_classification",
            "contrastive_learning",
            "regression",
            "generation",
        }
    )

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    if task_type == "classification":
        label_smoothing = create_label_smoothing_config(smoothing=0.1)
        return create_loss_config(
            loss_type=LossType.CROSS_ENTROPY,
            label_smoothing_config=label_smoothing,
        )

    elif task_type == "imbalanced_classification":
        focal = create_focal_loss_config(alpha=0.25, gamma=2.0)
        return create_loss_config(
            loss_type=LossType.FOCAL,
            focal_config=focal,
        )

    elif task_type == "contrastive_learning":
        contrastive = create_contrastive_loss_config(
            temperature=0.07,
            contrastive_type=ContrastiveType.NTXENT,
        )
        return create_loss_config(
            loss_type=LossType.CONTRASTIVE,
            contrastive_config=contrastive,
        )

    elif task_type == "regression":
        return create_loss_config(loss_type=LossType.MSE)

    else:  # generation
        label_smoothing = create_label_smoothing_config(smoothing=0.1)
        return create_loss_config(
            loss_type=LossType.CROSS_ENTROPY,
            label_smoothing_config=label_smoothing,
        )
