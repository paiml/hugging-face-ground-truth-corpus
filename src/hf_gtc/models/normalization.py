"""Normalization layer utilities for transformer models.

This module provides utilities for configuring and analyzing normalization
layers commonly used in transformer architectures, including LayerNorm,
RMSNorm, BatchNorm, GroupNorm, and InstanceNorm.

Examples:
    >>> from hf_gtc.models.normalization import create_layer_norm_config
    >>> config = create_layer_norm_config(normalized_shape=768)
    >>> config.normalized_shape
    (768,)
    >>> config.eps
    1e-05
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class NormType(Enum):
    """Normalization layer types.

    Attributes:
        LAYER_NORM: Standard Layer Normalization.
        RMS_NORM: Root Mean Square Layer Normalization.
        BATCH_NORM: Batch Normalization.
        GROUP_NORM: Group Normalization.
        INSTANCE_NORM: Instance Normalization.
        NONE: No normalization.

    Examples:
        >>> NormType.LAYER_NORM.value
        'layer_norm'
        >>> NormType.RMS_NORM.value
        'rms_norm'
        >>> NormType.BATCH_NORM.value
        'batch_norm'
    """

    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"
    NONE = "none"


VALID_NORM_TYPES = frozenset(n.value for n in NormType)


class NormPosition(Enum):
    """Position of normalization in transformer blocks.

    Attributes:
        PRE: Pre-normalization (before attention/FFN).
        POST: Post-normalization (after attention/FFN).
        SANDWICH: Both pre and post normalization.

    Examples:
        >>> NormPosition.PRE.value
        'pre'
        >>> NormPosition.POST.value
        'post'
        >>> NormPosition.SANDWICH.value
        'sandwich'
    """

    PRE = "pre"
    POST = "post"
    SANDWICH = "sandwich"


VALID_NORM_POSITIONS = frozenset(p.value for p in NormPosition)


class EpsType(Enum):
    """Epsilon value types for numerical stability.

    Attributes:
        STANDARD: Standard epsilon (1e-5).
        FP16_SAFE: Safe epsilon for FP16 training (1e-5).
        BF16_SAFE: Safe epsilon for BF16 training (1e-6).

    Examples:
        >>> EpsType.STANDARD.value
        'standard'
        >>> EpsType.FP16_SAFE.value
        'fp16_safe'
        >>> EpsType.BF16_SAFE.value
        'bf16_safe'
    """

    STANDARD = "standard"
    FP16_SAFE = "fp16_safe"
    BF16_SAFE = "bf16_safe"


VALID_EPS_TYPES = frozenset(e.value for e in EpsType)


@dataclass(frozen=True, slots=True)
class LayerNormConfig:
    """Configuration for Layer Normalization.

    Attributes:
        normalized_shape: Shape of the normalized dimension(s).
        eps: Epsilon value for numerical stability.
        elementwise_affine: Whether to use learnable affine parameters.

    Examples:
        >>> config = LayerNormConfig(
        ...     normalized_shape=(768,),
        ...     eps=1e-5,
        ...     elementwise_affine=True,
        ... )
        >>> config.normalized_shape
        (768,)
        >>> config.eps
        1e-05
        >>> config.elementwise_affine
        True
    """

    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool


@dataclass(frozen=True, slots=True)
class RMSNormConfig:
    """Configuration for RMS Layer Normalization.

    Attributes:
        hidden_size: Size of the hidden dimension.
        eps: Epsilon value for numerical stability.
        add_unit_offset: Whether to add 1 to the weight (LLaMA style).

    Examples:
        >>> config = RMSNormConfig(
        ...     hidden_size=4096,
        ...     eps=1e-6,
        ...     add_unit_offset=False,
        ... )
        >>> config.hidden_size
        4096
        >>> config.eps
        1e-06
        >>> config.add_unit_offset
        False
    """

    hidden_size: int
    eps: float
    add_unit_offset: bool


@dataclass(frozen=True, slots=True)
class GroupNormConfig:
    """Configuration for Group Normalization.

    Attributes:
        num_groups: Number of groups to separate the channels into.
        num_channels: Number of channels.
        eps: Epsilon value for numerical stability.
        affine: Whether to use learnable affine parameters.

    Examples:
        >>> config = GroupNormConfig(
        ...     num_groups=32,
        ...     num_channels=256,
        ...     eps=1e-5,
        ...     affine=True,
        ... )
        >>> config.num_groups
        32
        >>> config.num_channels
        256
    """

    num_groups: int
    num_channels: int
    eps: float
    affine: bool = True


@dataclass(frozen=True, slots=True)
class BatchNormConfig:
    """Configuration for Batch Normalization.

    Attributes:
        num_features: Number of features/channels.
        eps: Epsilon value for numerical stability.
        momentum: Momentum for running mean and variance.
        affine: Whether to use learnable affine parameters.
        track_running_stats: Whether to track running statistics.

    Examples:
        >>> config = BatchNormConfig(
        ...     num_features=256,
        ...     eps=1e-5,
        ...     momentum=0.1,
        ...     affine=True,
        ...     track_running_stats=True,
        ... )
        >>> config.num_features
        256
        >>> config.momentum
        0.1
    """

    num_features: int
    eps: float
    momentum: float
    affine: bool = True
    track_running_stats: bool = True


@dataclass(frozen=True, slots=True)
class NormConfig:
    """Combined normalization configuration.

    Attributes:
        norm_type: Type of normalization to use.
        layer_norm_config: Layer norm configuration (if applicable).
        rms_norm_config: RMS norm configuration (if applicable).
        group_norm_config: Group norm configuration (if applicable).
        batch_norm_config: Batch norm configuration (if applicable).
        position: Position of normalization in transformer blocks.

    Examples:
        >>> layer_config = LayerNormConfig((768,), 1e-5, True)
        >>> config = NormConfig(
        ...     norm_type=NormType.LAYER_NORM,
        ...     layer_norm_config=layer_config,
        ...     rms_norm_config=None,
        ...     group_norm_config=None,
        ...     batch_norm_config=None,
        ...     position=NormPosition.PRE,
        ... )
        >>> config.norm_type
        <NormType.LAYER_NORM: 'layer_norm'>
        >>> config.position
        <NormPosition.PRE: 'pre'>
    """

    norm_type: NormType
    layer_norm_config: LayerNormConfig | None
    rms_norm_config: RMSNormConfig | None
    group_norm_config: GroupNormConfig | None
    batch_norm_config: BatchNormConfig | None
    position: NormPosition


@dataclass(frozen=True, slots=True)
class NormStats:
    """Statistics for normalization analysis.

    Attributes:
        mean_activation: Mean activation value.
        std_activation: Standard deviation of activations.
        num_parameters: Number of learnable parameters.
        memory_bytes: Memory usage in bytes.
        flops_per_token: FLOPs per token for normalization.

    Examples:
        >>> stats = NormStats(
        ...     mean_activation=0.0,
        ...     std_activation=1.0,
        ...     num_parameters=1536,
        ...     memory_bytes=6144,
        ...     flops_per_token=4608,
        ... )
        >>> stats.num_parameters
        1536
    """

    mean_activation: float
    std_activation: float
    num_parameters: int
    memory_bytes: int
    flops_per_token: int


def validate_layer_norm_config(config: LayerNormConfig) -> None:
    """Validate Layer Normalization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If normalized_shape is empty.
        ValueError: If normalized_shape contains non-positive values.
        ValueError: If eps is not positive.

    Examples:
        >>> config = LayerNormConfig((768,), 1e-5, True)
        >>> validate_layer_norm_config(config)  # No error

        >>> validate_layer_norm_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LayerNormConfig((), 1e-5, True)
        >>> validate_layer_norm_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: normalized_shape cannot be empty
    """
    validate_not_none(config, "config")

    if not config.normalized_shape:
        msg = "normalized_shape cannot be empty"
        raise ValueError(msg)

    for dim in config.normalized_shape:
        if dim <= 0:
            msg = f"normalized_shape dimensions must be positive, got {dim}"
            raise ValueError(msg)

    if config.eps <= 0:
        msg = f"eps must be positive, got {config.eps}"
        raise ValueError(msg)


def validate_rms_norm_config(config: RMSNormConfig) -> None:
    """Validate RMS Normalization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If hidden_size is not positive.
        ValueError: If eps is not positive.

    Examples:
        >>> config = RMSNormConfig(4096, 1e-6, False)
        >>> validate_rms_norm_config(config)  # No error

        >>> validate_rms_norm_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = RMSNormConfig(0, 1e-6, False)
        >>> validate_rms_norm_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_size must be positive
    """
    validate_not_none(config, "config")

    if config.hidden_size <= 0:
        msg = f"hidden_size must be positive, got {config.hidden_size}"
        raise ValueError(msg)

    if config.eps <= 0:
        msg = f"eps must be positive, got {config.eps}"
        raise ValueError(msg)


def validate_group_norm_config(config: GroupNormConfig) -> None:
    """Validate Group Normalization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_groups is not positive.
        ValueError: If num_channels is not positive.
        ValueError: If num_channels is not divisible by num_groups.
        ValueError: If eps is not positive.

    Examples:
        >>> config = GroupNormConfig(32, 256, 1e-5)
        >>> validate_group_norm_config(config)  # No error

        >>> validate_group_norm_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = GroupNormConfig(32, 100, 1e-5)
        >>> validate_group_norm_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_channels must be divisible by num_groups
    """
    validate_not_none(config, "config")

    if config.num_groups <= 0:
        msg = f"num_groups must be positive, got {config.num_groups}"
        raise ValueError(msg)

    if config.num_channels <= 0:
        msg = f"num_channels must be positive, got {config.num_channels}"
        raise ValueError(msg)

    if config.num_channels % config.num_groups != 0:
        msg = (
            f"num_channels must be divisible by num_groups, "
            f"got {config.num_channels} % {config.num_groups} != 0"
        )
        raise ValueError(msg)

    if config.eps <= 0:
        msg = f"eps must be positive, got {config.eps}"
        raise ValueError(msg)


def validate_batch_norm_config(config: BatchNormConfig) -> None:
    """Validate Batch Normalization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_features is not positive.
        ValueError: If eps is not positive.
        ValueError: If momentum is not between 0 and 1.

    Examples:
        >>> config = BatchNormConfig(256, 1e-5, 0.1)
        >>> validate_batch_norm_config(config)  # No error

        >>> validate_batch_norm_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BatchNormConfig(256, 1e-5, 1.5)
        >>> validate_batch_norm_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: momentum must be between 0 and 1
    """
    validate_not_none(config, "config")

    if config.num_features <= 0:
        msg = f"num_features must be positive, got {config.num_features}"
        raise ValueError(msg)

    if config.eps <= 0:
        msg = f"eps must be positive, got {config.eps}"
        raise ValueError(msg)

    if not 0 <= config.momentum <= 1:
        msg = f"momentum must be between 0 and 1, got {config.momentum}"
        raise ValueError(msg)


_NORM_CONFIG_VALIDATORS: dict[NormType, tuple[str, Any]] = {
    NormType.LAYER_NORM: ("layer_norm_config", validate_layer_norm_config),
    NormType.RMS_NORM: ("rms_norm_config", validate_rms_norm_config),
    NormType.GROUP_NORM: ("group_norm_config", validate_group_norm_config),
    NormType.BATCH_NORM: ("batch_norm_config", validate_batch_norm_config),
}


def _validate_norm_subconfig(config: NormConfig) -> None:
    """Validate the sub-configuration for a specific normalization type."""
    entry = _NORM_CONFIG_VALIDATORS.get(config.norm_type)
    if entry is None:
        return
    attr_name, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} required for {config.norm_type.value} type"
        raise ValueError(msg)
    validator(sub_config)


def validate_norm_config(config: NormConfig) -> None:
    """Validate combined normalization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If required sub-config is missing.

    Examples:
        >>> layer_config = LayerNormConfig((768,), 1e-5, True)
        >>> config = NormConfig(
        ...     NormType.LAYER_NORM, layer_config, None, None, None, NormPosition.PRE
        ... )
        >>> validate_norm_config(config)  # No error

        >>> validate_norm_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = NormConfig(
        ...     NormType.LAYER_NORM, None, None, None, None, NormPosition.PRE
        ... )
        >>> validate_norm_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_norm_config required for LAYER_NORM type
    """
    validate_not_none(config, "config")

    if config.norm_type == NormType.NONE:
        return

    _validate_norm_subconfig(config)


def create_layer_norm_config(
    normalized_shape: int | tuple[int, ...] = 768,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
) -> LayerNormConfig:
    """Create a Layer Normalization configuration.

    Args:
        normalized_shape: Shape of normalized dimension(s). Defaults to 768.
        eps: Epsilon value. Defaults to 1e-5.
        elementwise_affine: Use learnable parameters. Defaults to True.

    Returns:
        Validated LayerNormConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_layer_norm_config(768)
        >>> config.normalized_shape
        (768,)

        >>> config = create_layer_norm_config((512, 768))
        >>> config.normalized_shape
        (512, 768)

        >>> create_layer_norm_config(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: normalized_shape dimensions must be positive
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    config = LayerNormConfig(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )
    validate_layer_norm_config(config)
    return config


def create_rms_norm_config(
    hidden_size: int = 4096,
    eps: float = 1e-6,
    add_unit_offset: bool = False,
) -> RMSNormConfig:
    """Create an RMS Normalization configuration.

    Args:
        hidden_size: Hidden dimension size. Defaults to 4096.
        eps: Epsilon value. Defaults to 1e-6.
        add_unit_offset: Add 1 to weight (LLaMA style). Defaults to False.

    Returns:
        Validated RMSNormConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_rms_norm_config(4096)
        >>> config.hidden_size
        4096

        >>> config = create_rms_norm_config(2048, eps=1e-5)
        >>> config.eps
        1e-05

        >>> create_rms_norm_config(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_size must be positive
    """
    config = RMSNormConfig(
        hidden_size=hidden_size,
        eps=eps,
        add_unit_offset=add_unit_offset,
    )
    validate_rms_norm_config(config)
    return config


def create_group_norm_config(
    num_groups: int = 32,
    num_channels: int = 256,
    eps: float = 1e-5,
    affine: bool = True,
) -> GroupNormConfig:
    """Create a Group Normalization configuration.

    Args:
        num_groups: Number of groups. Defaults to 32.
        num_channels: Number of channels. Defaults to 256.
        eps: Epsilon value. Defaults to 1e-5.
        affine: Use learnable parameters. Defaults to True.

    Returns:
        Validated GroupNormConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_group_norm_config(32, 256)
        >>> config.num_groups
        32
        >>> config.num_channels
        256

        >>> create_group_norm_config(32, 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_channels must be divisible by num_groups
    """
    config = GroupNormConfig(
        num_groups=num_groups,
        num_channels=num_channels,
        eps=eps,
        affine=affine,
    )
    validate_group_norm_config(config)
    return config


def create_batch_norm_config(
    num_features: int = 256,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
    track_running_stats: bool = True,
) -> BatchNormConfig:
    """Create a Batch Normalization configuration.

    Args:
        num_features: Number of features/channels. Defaults to 256.
        eps: Epsilon value. Defaults to 1e-5.
        momentum: Momentum for running stats. Defaults to 0.1.
        affine: Use learnable parameters. Defaults to True.
        track_running_stats: Track running statistics. Defaults to True.

    Returns:
        Validated BatchNormConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_batch_norm_config(256)
        >>> config.num_features
        256
        >>> config.momentum
        0.1

        >>> create_batch_norm_config(256, momentum=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: momentum must be between 0 and 1
    """
    config = BatchNormConfig(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
    )
    validate_batch_norm_config(config)
    return config


def create_norm_config(
    norm_type: str | NormType = "layer_norm",
    hidden_size: int = 768,
    eps: float | None = None,
    position: str | NormPosition = "pre",
    elementwise_affine: bool = True,
    add_unit_offset: bool = False,
    num_groups: int = 32,
) -> NormConfig:
    """Create a combined normalization configuration.

    Args:
        norm_type: Type of normalization. Defaults to "layer_norm".
        hidden_size: Hidden dimension size. Defaults to 768.
        eps: Epsilon value. Defaults to type-specific default.
        position: Position of normalization. Defaults to "pre".
        elementwise_affine: Use learnable parameters. Defaults to True.
        add_unit_offset: Add 1 to weight (RMSNorm only). Defaults to False.
        num_groups: Number of groups (GroupNorm only). Defaults to 32.

    Returns:
        Validated NormConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_norm_config("layer_norm", 768)
        >>> config.norm_type
        <NormType.LAYER_NORM: 'layer_norm'>

        >>> config = create_norm_config("rms_norm", 4096)
        >>> config.rms_norm_config.hidden_size
        4096

        >>> config = create_norm_config("none")
        >>> config.norm_type
        <NormType.NONE: 'none'>

        >>> create_norm_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: norm_type must be one of
    """
    if isinstance(norm_type, str):
        norm_type = get_norm_type(norm_type)

    if isinstance(position, str):
        position = get_norm_position(position)

    layer_norm_config = None
    rms_norm_config = None
    group_norm_config = None
    batch_norm_config = None

    if norm_type == NormType.LAYER_NORM:
        layer_eps = eps if eps is not None else 1e-5
        layer_norm_config = create_layer_norm_config(
            normalized_shape=hidden_size,
            eps=layer_eps,
            elementwise_affine=elementwise_affine,
        )

    elif norm_type == NormType.RMS_NORM:
        rms_eps = eps if eps is not None else 1e-6
        rms_norm_config = create_rms_norm_config(
            hidden_size=hidden_size,
            eps=rms_eps,
            add_unit_offset=add_unit_offset,
        )

    elif norm_type == NormType.GROUP_NORM:
        group_eps = eps if eps is not None else 1e-5
        group_norm_config = create_group_norm_config(
            num_groups=num_groups,
            num_channels=hidden_size,
            eps=group_eps,
            affine=elementwise_affine,
        )

    elif norm_type == NormType.BATCH_NORM:
        batch_eps = eps if eps is not None else 1e-5
        batch_norm_config = create_batch_norm_config(
            num_features=hidden_size,
            eps=batch_eps,
            affine=elementwise_affine,
        )

    config = NormConfig(
        norm_type=norm_type,
        layer_norm_config=layer_norm_config,
        rms_norm_config=rms_norm_config,
        group_norm_config=group_norm_config,
        batch_norm_config=batch_norm_config,
        position=position,
    )
    validate_norm_config(config)
    return config


def _layer_norm_params(config: NormConfig) -> int:
    """Calculate learnable parameters for layer normalization."""
    if config.layer_norm_config is None:
        return 0
    if not config.layer_norm_config.elementwise_affine:
        return 0
    total_dim = 1
    for dim in config.layer_norm_config.normalized_shape:
        total_dim *= dim
    return total_dim * 2


def _rms_norm_params(config: NormConfig) -> int:
    """Calculate learnable parameters for RMS normalization."""
    if config.rms_norm_config is None:
        return 0
    return config.rms_norm_config.hidden_size


def _group_norm_params(config: NormConfig) -> int:
    """Calculate learnable parameters for group normalization."""
    if config.group_norm_config is None:
        return 0
    if not config.group_norm_config.affine:
        return 0
    return config.group_norm_config.num_channels * 2


def _batch_norm_params(config: NormConfig) -> int:
    """Calculate learnable parameters for batch normalization."""
    if config.batch_norm_config is None:
        return 0
    if not config.batch_norm_config.affine:
        return 0
    return config.batch_norm_config.num_features * 2


_NORM_PARAM_CALCULATORS: dict[NormType, Any] = {
    NormType.LAYER_NORM: _layer_norm_params,
    NormType.RMS_NORM: _rms_norm_params,
    NormType.GROUP_NORM: _group_norm_params,
    NormType.BATCH_NORM: _batch_norm_params,
}


def calculate_norm_params(config: NormConfig) -> int:
    """Calculate the number of learnable parameters for normalization.

    Args:
        config: Normalization configuration.

    Returns:
        Number of learnable parameters.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_norm_config("layer_norm", 768)
        >>> calculate_norm_params(config)
        1536

        >>> config = create_norm_config("rms_norm", 4096)
        >>> calculate_norm_params(config)
        4096

        >>> config = create_norm_config("none")
        >>> calculate_norm_params(config)
        0
    """
    validate_not_none(config, "config")

    calculator = _NORM_PARAM_CALCULATORS.get(config.norm_type)
    if calculator is not None:
        return calculator(config)
    return 0


def estimate_norm_memory(
    config: NormConfig,
    batch_size: int = 1,
    sequence_length: int = 512,
    dtype_bytes: int = 4,
) -> int:
    """Estimate memory usage for normalization in bytes.

    Args:
        config: Normalization configuration.
        batch_size: Batch size. Defaults to 1.
        sequence_length: Sequence length. Defaults to 512.
        dtype_bytes: Bytes per element. Defaults to 4 (FP32).

    Returns:
        Estimated memory usage in bytes.

    Raises:
        ValueError: If config is None.
        ValueError: If batch_size is not positive.
        ValueError: If sequence_length is not positive.

    Examples:
        >>> config = create_norm_config("layer_norm", 768)
        >>> mem = estimate_norm_memory(config, batch_size=1, sequence_length=512)
        >>> mem > 0
        True

        >>> config = create_norm_config("none")
        >>> estimate_norm_memory(config)
        0

        >>> estimate_norm_memory(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    if config.norm_type == NormType.NONE:
        return 0

    # Parameter memory
    params = calculate_norm_params(config)
    param_memory = params * dtype_bytes

    # Activation memory (depends on norm type)
    hidden_size = _get_hidden_size_from_config(config)
    activation_memory = batch_size * sequence_length * hidden_size * dtype_bytes

    # Statistics memory (for mean/variance computation)
    stats_memory = hidden_size * dtype_bytes * 2  # mean + variance

    return param_memory + activation_memory + stats_memory


def _get_hidden_size_from_config(config: NormConfig) -> int:
    """Get hidden size from normalization config."""
    if config.norm_type == NormType.LAYER_NORM and config.layer_norm_config:
        return config.layer_norm_config.normalized_shape[-1]
    if config.norm_type == NormType.RMS_NORM and config.rms_norm_config:
        return config.rms_norm_config.hidden_size
    if config.norm_type == NormType.GROUP_NORM and config.group_norm_config:
        return config.group_norm_config.num_channels
    if config.norm_type == NormType.BATCH_NORM and config.batch_norm_config:
        return config.batch_norm_config.num_features
    return 0


def compare_norm_stability(
    norm_type_a: str | NormType,
    norm_type_b: str | NormType,
    hidden_size: int = 768,
) -> dict[str, str]:
    """Compare numerical stability characteristics of two normalization types.

    Args:
        norm_type_a: First normalization type.
        norm_type_b: Second normalization type.
        hidden_size: Hidden dimension size. Defaults to 768.

    Returns:
        Dictionary with stability comparison results.

    Raises:
        ValueError: If norm types are invalid.

    Examples:
        >>> result = compare_norm_stability("layer_norm", "rms_norm")
        >>> "recommended" in result
        True
        >>> "reason" in result
        True

        >>> result = compare_norm_stability("layer_norm", "batch_norm")
        >>> result["recommended"] in ("layer_norm", "batch_norm")
        True
    """
    if isinstance(norm_type_a, str):
        norm_type_a = get_norm_type(norm_type_a)
    if isinstance(norm_type_b, str):
        norm_type_b = get_norm_type(norm_type_b)

    # Stability rankings (higher = more stable for transformers)
    stability_scores = {
        NormType.RMS_NORM: 5,  # Best for LLMs
        NormType.LAYER_NORM: 4,  # Standard, well-tested
        NormType.GROUP_NORM: 3,  # Good for small batches
        NormType.BATCH_NORM: 2,  # Batch-dependent
        NormType.INSTANCE_NORM: 2,  # Instance-dependent
        NormType.NONE: 0,  # No normalization
    }

    score_a = stability_scores.get(norm_type_a, 1)
    score_b = stability_scores.get(norm_type_b, 1)

    # Determine recommendation
    if norm_type_a == NormType.NONE:
        recommended = norm_type_b.value
        reason = f"{norm_type_b.value} provides normalization, 'none' does not"
    elif norm_type_b == NormType.NONE:
        recommended = norm_type_a.value
        reason = f"{norm_type_a.value} provides normalization, 'none' does not"
    elif score_a > score_b:
        recommended = norm_type_a.value
        reason = f"{norm_type_a.value} has better stability for transformer training"
    elif score_b > score_a:
        recommended = norm_type_b.value
        reason = f"{norm_type_b.value} has better stability for transformer training"
    else:
        recommended = norm_type_a.value
        reason = "Both have similar stability characteristics"

    # Memory efficiency comparison
    config_a = create_norm_config(norm_type_a, hidden_size)
    config_b = create_norm_config(norm_type_b, hidden_size)
    params_a = calculate_norm_params(config_a)
    params_b = calculate_norm_params(config_b)

    more_efficient = norm_type_a.value if params_a <= params_b else norm_type_b.value

    return {
        "recommended": recommended,
        "reason": reason,
        "more_efficient": more_efficient,
        f"{norm_type_a.value}_params": str(params_a),
        f"{norm_type_b.value}_params": str(params_b),
    }


def select_eps_for_dtype(dtype: str = "fp32") -> float:
    """Select appropriate epsilon value for a given data type.

    Args:
        dtype: Data type string ("fp32", "fp16", "bf16"). Defaults to "fp32".

    Returns:
        Recommended epsilon value.

    Raises:
        ValueError: If dtype is invalid.

    Examples:
        >>> select_eps_for_dtype("fp32")
        1e-05

        >>> select_eps_for_dtype("fp16")
        1e-05

        >>> select_eps_for_dtype("bf16")
        1e-06

        >>> select_eps_for_dtype("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dtype must be one of
    """
    valid_dtypes = {"fp32", "fp16", "bf16", "fp8"}
    if dtype not in valid_dtypes:
        msg = f"dtype must be one of {valid_dtypes}, got '{dtype}'"
        raise ValueError(msg)

    eps_values = {
        "fp32": 1e-5,
        "fp16": 1e-5,  # Same as FP32, safe for FP16
        "bf16": 1e-6,  # Smaller due to larger range
        "fp8": 1e-4,  # Larger due to limited range
    }
    return eps_values[dtype]


def format_norm_stats(stats: NormStats) -> str:
    """Format normalization statistics for display.

    Args:
        stats: Normalization statistics to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = NormStats(0.0, 1.0, 1536, 6144, 4608)
        >>> formatted = format_norm_stats(stats)
        >>> "Parameters:" in formatted
        True
        >>> "Memory:" in formatted
        True

        >>> format_norm_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    memory_kb = stats.memory_bytes / 1024

    lines = [
        f"Mean Activation: {stats.mean_activation:.4f}",
        f"Std Activation: {stats.std_activation:.4f}",
        f"Parameters: {stats.num_parameters:,}",
        f"Memory: {memory_kb:.2f} KB",
        f"FLOPs/token: {stats.flops_per_token:,}",
    ]
    return "\n".join(lines)


def get_recommended_norm_config(
    model_type: str = "transformer",
    hidden_size: int = 768,
    use_case: str = "training",
    dtype: str = "fp32",
) -> NormConfig:
    """Get recommended normalization configuration.

    Args:
        model_type: Model type ("transformer", "cnn", "rnn"). Defaults to "transformer".
        hidden_size: Hidden dimension size. Defaults to 768.
        use_case: Use case ("training", "inference"). Defaults to "training".
        dtype: Data type. Defaults to "fp32".

    Returns:
        Recommended NormConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_norm_config("transformer", 768)
        >>> config.norm_type in (NormType.LAYER_NORM, NormType.RMS_NORM)
        True

        >>> config = get_recommended_norm_config("cnn", 256)
        >>> config.norm_type == NormType.BATCH_NORM
        True

        >>> get_recommended_norm_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of
    """
    valid_model_types = {"transformer", "cnn", "rnn", "vit", "llm"}
    if model_type not in valid_model_types:
        msg = f"model_type must be one of {valid_model_types}, got '{model_type}'"
        raise ValueError(msg)

    valid_use_cases = {"training", "inference"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    eps = select_eps_for_dtype(dtype)

    # Determine norm type based on model type
    if model_type in ("transformer", "vit"):
        norm_type = "layer_norm"
        position = "pre"  # Pre-norm is more stable
    elif model_type == "llm":
        norm_type = "rms_norm"  # RMSNorm is standard for LLMs
        position = "pre"
    elif model_type == "cnn":
        norm_type = "batch_norm"
        position = "post"
    elif model_type == "rnn":
        norm_type = "layer_norm"
        position = "post"
    else:
        norm_type = "layer_norm"
        position = "pre"

    return create_norm_config(
        norm_type=norm_type,
        hidden_size=hidden_size,
        eps=eps,
        position=position,
    )


def list_norm_types() -> list[str]:
    """List available normalization types.

    Returns:
        Sorted list of normalization type names.

    Examples:
        >>> types = list_norm_types()
        >>> "layer_norm" in types
        True
        >>> "rms_norm" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_NORM_TYPES)


def list_norm_positions() -> list[str]:
    """List available normalization positions.

    Returns:
        Sorted list of position names.

    Examples:
        >>> positions = list_norm_positions()
        >>> "pre" in positions
        True
        >>> "post" in positions
        True
        >>> positions == sorted(positions)
        True
    """
    return sorted(VALID_NORM_POSITIONS)


def list_eps_types() -> list[str]:
    """List available epsilon types.

    Returns:
        Sorted list of epsilon type names.

    Examples:
        >>> types = list_eps_types()
        >>> "standard" in types
        True
        >>> "fp16_safe" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_EPS_TYPES)


def get_norm_type(name: str) -> NormType:
    """Get normalization type enum from string.

    Args:
        name: Normalization type name.

    Returns:
        NormType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_norm_type("layer_norm")
        <NormType.LAYER_NORM: 'layer_norm'>

        >>> get_norm_type("rms_norm")
        <NormType.RMS_NORM: 'rms_norm'>

        >>> get_norm_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: norm_type must be one of
    """
    if name not in VALID_NORM_TYPES:
        msg = f"norm_type must be one of {VALID_NORM_TYPES}, got '{name}'"
        raise ValueError(msg)
    return NormType(name)


def get_norm_position(name: str) -> NormPosition:
    """Get normalization position enum from string.

    Args:
        name: Position name.

    Returns:
        NormPosition enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_norm_position("pre")
        <NormPosition.PRE: 'pre'>

        >>> get_norm_position("post")
        <NormPosition.POST: 'post'>

        >>> get_norm_position("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: norm_position must be one of
    """
    if name not in VALID_NORM_POSITIONS:
        msg = f"norm_position must be one of {VALID_NORM_POSITIONS}, got '{name}'"
        raise ValueError(msg)
    return NormPosition(name)


def get_eps_type(name: str) -> EpsType:
    """Get epsilon type enum from string.

    Args:
        name: Epsilon type name.

    Returns:
        EpsType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_eps_type("standard")
        <EpsType.STANDARD: 'standard'>

        >>> get_eps_type("fp16_safe")
        <EpsType.FP16_SAFE: 'fp16_safe'>

        >>> get_eps_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: eps_type must be one of
    """
    if name not in VALID_EPS_TYPES:
        msg = f"eps_type must be one of {VALID_EPS_TYPES}, got '{name}'"
        raise ValueError(msg)
    return EpsType(name)


def get_eps_value(eps_type: str | EpsType) -> float:
    """Get epsilon value for a given epsilon type.

    Args:
        eps_type: Epsilon type string or enum.

    Returns:
        Epsilon value.

    Raises:
        ValueError: If eps_type is invalid.

    Examples:
        >>> get_eps_value("standard")
        1e-05

        >>> get_eps_value("fp16_safe")
        1e-05

        >>> get_eps_value("bf16_safe")
        1e-06

        >>> get_eps_value(EpsType.STANDARD)
        1e-05
    """
    if isinstance(eps_type, str):
        eps_type = get_eps_type(eps_type)

    eps_values = {
        EpsType.STANDARD: 1e-5,
        EpsType.FP16_SAFE: 1e-5,
        EpsType.BF16_SAFE: 1e-6,
    }
    return eps_values[eps_type]
