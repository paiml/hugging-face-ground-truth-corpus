"""Positional encoding utilities for transformer models.

This module provides utilities for configuring and calculating positional
encodings used in transformer architectures, including sinusoidal, learned,
RoPE (Rotary Position Embedding), ALiBi (Attention with Linear Biases),
and relative position encodings.

Examples:
    >>> from hf_gtc.models.positional import create_rope_config
    >>> config = create_rope_config(dim=64, max_position=4096)
    >>> config.dim
    64
    >>> config.max_position
    4096
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class PositionalType(Enum):
    """Type of positional encoding.

    Attributes:
        SINUSOIDAL: Fixed sinusoidal position embeddings (Vaswani et al.).
        LEARNED: Learned position embeddings.
        ROTARY: Rotary Position Embedding (RoPE).
        ALIBI: Attention with Linear Biases.
        RELATIVE: Relative position encoding.
        NONE: No positional encoding.

    Examples:
        >>> PositionalType.SINUSOIDAL.value
        'sinusoidal'
        >>> PositionalType.ROTARY.value
        'rotary'
    """

    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROTARY = "rotary"
    ALIBI = "alibi"
    RELATIVE = "relative"
    NONE = "none"


VALID_POSITIONAL_TYPES = frozenset(p.value for p in PositionalType)


class RoPEScaling(Enum):
    """Scaling method for RoPE position embeddings.

    Attributes:
        LINEAR: Linear interpolation scaling.
        DYNAMIC: Dynamic NTK scaling.
        YARN: Yet Another RoPE extensioN.
        NTK_AWARE: NTK-aware interpolation.

    Examples:
        >>> RoPEScaling.LINEAR.value
        'linear'
        >>> RoPEScaling.YARN.value
        'yarn'
    """

    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    NTK_AWARE = "ntk_aware"


VALID_ROPE_SCALINGS = frozenset(s.value for s in RoPEScaling)


class InterpolationType(Enum):
    """Interpolation method for position extension.

    Attributes:
        LINEAR: Linear interpolation.
        DYNAMIC: Dynamic interpolation based on sequence length.
        YARN: YaRN interpolation with attention scaling.

    Examples:
        >>> InterpolationType.LINEAR.value
        'linear'
        >>> InterpolationType.DYNAMIC.value
        'dynamic'
    """

    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"


VALID_INTERPOLATION_TYPES = frozenset(i.value for i in InterpolationType)


# Type aliases
PositionalTypeStr = Literal[
    "sinusoidal", "learned", "rotary", "alibi", "relative", "none"
]
RoPEScalingStr = Literal["linear", "dynamic", "yarn", "ntk_aware"]
InterpolationTypeStr = Literal["linear", "dynamic", "yarn"]


@dataclass(frozen=True, slots=True)
class SinusoidalConfig:
    """Configuration for sinusoidal positional embeddings.

    Attributes:
        max_length: Maximum sequence length.
        embed_dim: Embedding dimension.
        base: Base for frequency computation. Defaults to 10000.

    Examples:
        >>> config = SinusoidalConfig(max_length=512, embed_dim=768, base=10000)
        >>> config.max_length
        512
        >>> config.embed_dim
        768
    """

    max_length: int
    embed_dim: int
    base: float = 10000.0


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    """Configuration for Rotary Position Embedding (RoPE).

    Attributes:
        dim: Dimension of the rotary embedding (usually head_dim).
        max_position: Maximum position for precomputation.
        base: Base frequency for rotation. Defaults to 10000.
        scaling_type: Type of scaling for extended context. Defaults to None.
        scaling_factor: Factor for position scaling. Defaults to 1.0.

    Examples:
        >>> config = RoPEConfig(
        ...     dim=64,
        ...     max_position=4096,
        ...     base=10000.0,
        ...     scaling_type=None,
        ...     scaling_factor=1.0,
        ... )
        >>> config.dim
        64
        >>> config.base
        10000.0
    """

    dim: int
    max_position: int
    base: float = 10000.0
    scaling_type: RoPEScaling | None = None
    scaling_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class ALiBiConfig:
    """Configuration for Attention with Linear Biases (ALiBi).

    Attributes:
        num_heads: Number of attention heads.
        slopes_power: Power for computing slopes. Defaults to 8 (2^(-8/n)).

    Examples:
        >>> config = ALiBiConfig(num_heads=8, slopes_power=8)
        >>> config.num_heads
        8
        >>> config.slopes_power
        8
    """

    num_heads: int
    slopes_power: int = 8


@dataclass(frozen=True, slots=True)
class PositionalConfig:
    """Unified configuration for positional encodings.

    Attributes:
        pos_type: Type of positional encoding.
        rope_config: RoPE configuration (if pos_type is ROTARY).
        alibi_config: ALiBi configuration (if pos_type is ALIBI).
        max_length: Maximum sequence length for sinusoidal/learned.

    Examples:
        >>> config = PositionalConfig(
        ...     pos_type=PositionalType.ROTARY,
        ...     rope_config=RoPEConfig(dim=64, max_position=4096),
        ...     alibi_config=None,
        ...     max_length=4096,
        ... )
        >>> config.pos_type
        <PositionalType.ROTARY: 'rotary'>
    """

    pos_type: PositionalType
    rope_config: RoPEConfig | None
    alibi_config: ALiBiConfig | None
    max_length: int


def validate_sinusoidal_config(config: SinusoidalConfig) -> None:
    """Validate sinusoidal configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SinusoidalConfig(max_length=512, embed_dim=768)
        >>> validate_sinusoidal_config(config)  # No error

        >>> bad_config = SinusoidalConfig(max_length=0, embed_dim=768)
        >>> validate_sinusoidal_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.embed_dim <= 0:
        msg = f"embed_dim must be positive, got {config.embed_dim}"
        raise ValueError(msg)

    if config.embed_dim % 2 != 0:
        msg = f"embed_dim must be even for sinusoidal, got {config.embed_dim}"
        raise ValueError(msg)

    if config.base <= 0:
        msg = f"base must be positive, got {config.base}"
        raise ValueError(msg)


def validate_rope_config(config: RoPEConfig) -> None:
    """Validate RoPE configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = RoPEConfig(dim=64, max_position=4096)
        >>> validate_rope_config(config)  # No error

        >>> bad_config = RoPEConfig(dim=0, max_position=4096)
        >>> validate_rope_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dim must be positive
    """
    if config.dim <= 0:
        msg = f"dim must be positive, got {config.dim}"
        raise ValueError(msg)

    if config.dim % 2 != 0:
        msg = f"dim must be even for RoPE, got {config.dim}"
        raise ValueError(msg)

    if config.max_position <= 0:
        msg = f"max_position must be positive, got {config.max_position}"
        raise ValueError(msg)

    if config.base <= 0:
        msg = f"base must be positive, got {config.base}"
        raise ValueError(msg)

    if config.scaling_factor <= 0:
        msg = f"scaling_factor must be positive, got {config.scaling_factor}"
        raise ValueError(msg)


def validate_alibi_config(config: ALiBiConfig) -> None:
    """Validate ALiBi configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ALiBiConfig(num_heads=8)
        >>> validate_alibi_config(config)  # No error

        >>> bad_config = ALiBiConfig(num_heads=0)
        >>> validate_alibi_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_heads must be positive
    """
    if config.num_heads <= 0:
        msg = f"num_heads must be positive, got {config.num_heads}"
        raise ValueError(msg)

    if config.slopes_power <= 0:
        msg = f"slopes_power must be positive, got {config.slopes_power}"
        raise ValueError(msg)


def _validate_positional_type_config(config: PositionalConfig) -> None:
    """Validate positional sub-config based on type."""
    pos_validators: dict[PositionalType, tuple[str, str, object]] = {
        PositionalType.ROTARY: (
            "rope_config",
            "ROTARY positional type",
            validate_rope_config,
        ),
        PositionalType.ALIBI: (
            "alibi_config",
            "ALIBI positional type",
            validate_alibi_config,
        ),
    }
    entry = pos_validators.get(config.pos_type)
    if entry is None:
        return
    attr_name, label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} required for {label}"
        raise ValueError(msg)
    validator(sub_config)


def validate_positional_config(config: PositionalConfig) -> None:
    """Validate positional configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PositionalConfig(
        ...     pos_type=PositionalType.ROTARY,
        ...     rope_config=RoPEConfig(dim=64, max_position=4096),
        ...     alibi_config=None,
        ...     max_length=4096,
        ... )
        >>> validate_positional_config(config)  # No error

        >>> bad_config = PositionalConfig(
        ...     pos_type=PositionalType.ROTARY,
        ...     rope_config=None,
        ...     alibi_config=None,
        ...     max_length=4096,
        ... )
        >>> validate_positional_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rope_config required for ROTARY positional type
    """
    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    _validate_positional_type_config(config)


def create_sinusoidal_config(
    max_length: int = 512,
    embed_dim: int = 768,
    base: float = 10000.0,
) -> SinusoidalConfig:
    """Create a sinusoidal configuration.

    Args:
        max_length: Maximum sequence length. Defaults to 512.
        embed_dim: Embedding dimension. Defaults to 768.
        base: Base for frequency computation. Defaults to 10000.0.

    Returns:
        SinusoidalConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_sinusoidal_config(max_length=1024)
        >>> config.max_length
        1024

        >>> config = create_sinusoidal_config(embed_dim=512)
        >>> config.embed_dim
        512

        >>> create_sinusoidal_config(max_length=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    config = SinusoidalConfig(
        max_length=max_length,
        embed_dim=embed_dim,
        base=base,
    )
    validate_sinusoidal_config(config)
    return config


def create_rope_config(
    dim: int = 64,
    max_position: int = 4096,
    base: float = 10000.0,
    scaling_type: RoPEScalingStr | None = None,
    scaling_factor: float = 1.0,
) -> RoPEConfig:
    """Create a RoPE configuration.

    Args:
        dim: Dimension of rotary embedding. Defaults to 64.
        max_position: Maximum position. Defaults to 4096.
        base: Base frequency. Defaults to 10000.0.
        scaling_type: Type of scaling. Defaults to None.
        scaling_factor: Scaling factor. Defaults to 1.0.

    Returns:
        RoPEConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_rope_config(dim=128, max_position=8192)
        >>> config.dim
        128

        >>> config = create_rope_config(scaling_type="linear", scaling_factor=2.0)
        >>> config.scaling_type
        <RoPEScaling.LINEAR: 'linear'>

        >>> create_rope_config(dim=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dim must be positive
    """
    if scaling_type is not None and scaling_type not in VALID_ROPE_SCALINGS:
        msg = f"scaling_type must be one of {VALID_ROPE_SCALINGS}, got '{scaling_type}'"
        raise ValueError(msg)

    scaling_enum = RoPEScaling(scaling_type) if scaling_type else None

    config = RoPEConfig(
        dim=dim,
        max_position=max_position,
        base=base,
        scaling_type=scaling_enum,
        scaling_factor=scaling_factor,
    )
    validate_rope_config(config)
    return config


def create_alibi_config(
    num_heads: int = 8,
    slopes_power: int = 8,
) -> ALiBiConfig:
    """Create an ALiBi configuration.

    Args:
        num_heads: Number of attention heads. Defaults to 8.
        slopes_power: Power for slope computation. Defaults to 8.

    Returns:
        ALiBiConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_alibi_config(num_heads=12)
        >>> config.num_heads
        12

        >>> create_alibi_config(num_heads=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_heads must be positive
    """
    config = ALiBiConfig(
        num_heads=num_heads,
        slopes_power=slopes_power,
    )
    validate_alibi_config(config)
    return config


def create_positional_config(
    pos_type: PositionalTypeStr = "sinusoidal",
    rope_config: RoPEConfig | None = None,
    alibi_config: ALiBiConfig | None = None,
    max_length: int = 4096,
) -> PositionalConfig:
    """Create a positional configuration.

    Args:
        pos_type: Type of positional encoding. Defaults to "sinusoidal".
        rope_config: RoPE config (required for rotary). Defaults to None.
        alibi_config: ALiBi config (required for alibi). Defaults to None.
        max_length: Maximum sequence length. Defaults to 4096.

    Returns:
        PositionalConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_positional_config(pos_type="sinusoidal")
        >>> config.pos_type
        <PositionalType.SINUSOIDAL: 'sinusoidal'>

        >>> rope = create_rope_config(dim=64)
        >>> config = create_positional_config(pos_type="rotary", rope_config=rope)
        >>> config.pos_type
        <PositionalType.ROTARY: 'rotary'>

        >>> create_positional_config(pos_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pos_type must be one of
    """
    if pos_type not in VALID_POSITIONAL_TYPES:
        msg = f"pos_type must be one of {VALID_POSITIONAL_TYPES}, got '{pos_type}'"
        raise ValueError(msg)

    config = PositionalConfig(
        pos_type=PositionalType(pos_type),
        rope_config=rope_config,
        alibi_config=alibi_config,
        max_length=max_length,
    )
    validate_positional_config(config)
    return config


def list_positional_types() -> list[str]:
    """List available positional encoding types.

    Returns:
        Sorted list of positional type names.

    Examples:
        >>> types = list_positional_types()
        >>> "sinusoidal" in types
        True
        >>> "rotary" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_POSITIONAL_TYPES)


def list_rope_scalings() -> list[str]:
    """List available RoPE scaling methods.

    Returns:
        Sorted list of RoPE scaling method names.

    Examples:
        >>> scalings = list_rope_scalings()
        >>> "linear" in scalings
        True
        >>> "yarn" in scalings
        True
        >>> scalings == sorted(scalings)
        True
    """
    return sorted(VALID_ROPE_SCALINGS)


def list_interpolation_types() -> list[str]:
    """List available interpolation types.

    Returns:
        Sorted list of interpolation type names.

    Examples:
        >>> types = list_interpolation_types()
        >>> "linear" in types
        True
        >>> "dynamic" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_INTERPOLATION_TYPES)


def get_positional_type(name: str) -> PositionalType:
    """Get a positional type by name.

    Args:
        name: Name of the positional type.

    Returns:
        The corresponding PositionalType enum value.

    Raises:
        ValueError: If name is not a valid positional type.

    Examples:
        >>> get_positional_type("sinusoidal")
        <PositionalType.SINUSOIDAL: 'sinusoidal'>
        >>> get_positional_type("rotary")
        <PositionalType.ROTARY: 'rotary'>

        >>> get_positional_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown positional type
    """
    if name not in VALID_POSITIONAL_TYPES:
        msg = f"Unknown positional type: '{name}'. Valid: {VALID_POSITIONAL_TYPES}"
        raise ValueError(msg)
    return PositionalType(name)


def get_rope_scaling(name: str) -> RoPEScaling:
    """Get a RoPE scaling method by name.

    Args:
        name: Name of the scaling method.

    Returns:
        The corresponding RoPEScaling enum value.

    Raises:
        ValueError: If name is not a valid scaling method.

    Examples:
        >>> get_rope_scaling("linear")
        <RoPEScaling.LINEAR: 'linear'>
        >>> get_rope_scaling("yarn")
        <RoPEScaling.YARN: 'yarn'>

        >>> get_rope_scaling("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown RoPE scaling
    """
    if name not in VALID_ROPE_SCALINGS:
        msg = f"Unknown RoPE scaling: '{name}'. Valid: {VALID_ROPE_SCALINGS}"
        raise ValueError(msg)
    return RoPEScaling(name)


def get_interpolation_type(name: str) -> InterpolationType:
    """Get an interpolation type by name.

    Args:
        name: Name of the interpolation type.

    Returns:
        The corresponding InterpolationType enum value.

    Raises:
        ValueError: If name is not a valid interpolation type.

    Examples:
        >>> get_interpolation_type("linear")
        <InterpolationType.LINEAR: 'linear'>
        >>> get_interpolation_type("dynamic")
        <InterpolationType.DYNAMIC: 'dynamic'>

        >>> get_interpolation_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown interpolation type
    """
    if name not in VALID_INTERPOLATION_TYPES:
        valid = VALID_INTERPOLATION_TYPES
        msg = f"Unknown interpolation type: '{name}'. Valid: {valid}"
        raise ValueError(msg)
    return InterpolationType(name)


def calculate_sinusoidal_embeddings(
    max_length: int,
    embed_dim: int,
    base: float = 10000.0,
) -> list[list[float]]:
    """Calculate sinusoidal positional embeddings.

    Implements the sinusoidal positional encoding from "Attention Is All You Need".
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_length: Maximum sequence length.
        embed_dim: Embedding dimension (must be even).
        base: Base for frequency computation. Defaults to 10000.0.

    Returns:
        List of embeddings, shape [max_length, embed_dim].

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> embeddings = calculate_sinusoidal_embeddings(4, 8)
        >>> len(embeddings)
        4
        >>> len(embeddings[0])
        8

        >>> # Position 0 should have sin(0)=0 for even indices
        >>> embeddings[0][0]
        0.0

        >>> calculate_sinusoidal_embeddings(0, 8)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if embed_dim <= 0:
        msg = f"embed_dim must be positive, got {embed_dim}"
        raise ValueError(msg)

    if embed_dim % 2 != 0:
        msg = f"embed_dim must be even, got {embed_dim}"
        raise ValueError(msg)

    if base <= 0:
        msg = f"base must be positive, got {base}"
        raise ValueError(msg)

    embeddings: list[list[float]] = []

    for pos in range(max_length):
        pos_encoding: list[float] = []
        for i in range(embed_dim // 2):
            exponent = 2.0 * i / embed_dim
            frequency = pos / (base**exponent)
            pos_encoding.append(math.sin(frequency))
            pos_encoding.append(math.cos(frequency))
        embeddings.append(pos_encoding)

    return embeddings


def calculate_rope_frequencies(
    dim: int,
    max_position: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
) -> list[list[float]]:
    """Calculate RoPE frequency matrix.

    Computes the rotation frequencies for Rotary Position Embedding.
    freq_i = 1 / (base^(2i/d)) for i in [0, d/2)

    Args:
        dim: Embedding dimension (must be even).
        max_position: Maximum position for precomputation.
        base: Base frequency. Defaults to 10000.0.
        scaling_factor: Position scaling factor. Defaults to 1.0.

    Returns:
        Frequency matrix of shape [max_position, dim].

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> freqs = calculate_rope_frequencies(8, 4)
        >>> len(freqs)
        4
        >>> len(freqs[0])
        8

        >>> calculate_rope_frequencies(0, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dim must be positive

        >>> calculate_rope_frequencies(7, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dim must be even
    """
    if dim <= 0:
        msg = f"dim must be positive, got {dim}"
        raise ValueError(msg)

    if dim % 2 != 0:
        msg = f"dim must be even, got {dim}"
        raise ValueError(msg)

    if max_position <= 0:
        msg = f"max_position must be positive, got {max_position}"
        raise ValueError(msg)

    if base <= 0:
        msg = f"base must be positive, got {base}"
        raise ValueError(msg)

    if scaling_factor <= 0:
        msg = f"scaling_factor must be positive, got {scaling_factor}"
        raise ValueError(msg)

    # Compute inverse frequencies
    inv_freq: list[float] = []
    for i in range(dim // 2):
        exponent = 2.0 * i / dim
        inv_freq.append(1.0 / (base**exponent))

    # Compute frequencies for each position
    frequencies: list[list[float]] = []
    for pos in range(max_position):
        scaled_pos = pos / scaling_factor
        pos_freq: list[float] = []
        for freq in inv_freq:
            angle = scaled_pos * freq
            # Store both cos and sin for rotation
            pos_freq.append(math.cos(angle))
            pos_freq.append(math.sin(angle))
        frequencies.append(pos_freq)

    return frequencies


def calculate_alibi_slopes(
    num_heads: int,
    slopes_power: int = 8,
) -> list[float]:
    """Calculate ALiBi attention bias slopes.

    Computes slopes for ALiBi: slope_i = 2^(-8 * i / num_heads)
    where i ranges from 1 to num_heads.

    Args:
        num_heads: Number of attention heads.
        slopes_power: Power base (typically 8). Defaults to 8.

    Returns:
        List of slopes for each head.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> slopes = calculate_alibi_slopes(8)
        >>> len(slopes)
        8
        >>> slopes[0] > slopes[1]  # Slopes decrease
        True

        >>> slopes = calculate_alibi_slopes(4)
        >>> len(slopes)
        4

        >>> calculate_alibi_slopes(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_heads must be positive
    """
    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if slopes_power <= 0:
        msg = f"slopes_power must be positive, got {slopes_power}"
        raise ValueError(msg)

    slopes: list[float] = []
    for i in range(1, num_heads + 1):
        # slope = 2^(-slopes_power * i / num_heads)
        exponent = -slopes_power * i / num_heads
        slopes.append(2.0**exponent)

    return slopes


def estimate_position_memory(
    pos_type: PositionalTypeStr,
    max_length: int,
    embed_dim: int = 768,
    num_heads: int = 12,
    head_dim: int = 64,
    dtype_bytes: int = 4,
) -> float:
    """Estimate memory usage for positional encodings in MB.

    Args:
        pos_type: Type of positional encoding.
        max_length: Maximum sequence length.
        embed_dim: Embedding dimension (for sinusoidal/learned).
        num_heads: Number of attention heads (for ALiBi).
        head_dim: Head dimension (for RoPE).
        dtype_bytes: Bytes per element. Defaults to 4 (float32).

    Returns:
        Estimated memory usage in megabytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> mem = estimate_position_memory("sinusoidal", 512, embed_dim=768)
        >>> mem > 0
        True

        >>> mem = estimate_position_memory("alibi", 512, num_heads=12)
        >>> mem > 0
        True

        >>> estimate_position_memory("sinusoidal", 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    if pos_type not in VALID_POSITIONAL_TYPES:
        msg = f"pos_type must be one of {VALID_POSITIONAL_TYPES}, got '{pos_type}'"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    bytes_to_mb = 1.0 / (1024 * 1024)
    total_bytes: float = 0.0

    if pos_type == "sinusoidal" or pos_type == "learned":
        # [max_length, embed_dim] embeddings
        total_bytes = max_length * embed_dim * dtype_bytes

    elif pos_type == "rotary":
        # [max_length, head_dim] frequencies (cos and sin stored)
        total_bytes = max_length * head_dim * 2 * dtype_bytes

    elif pos_type == "alibi":
        # [num_heads] slopes + potential bias cache
        # The bias matrix is [num_heads, max_length, max_length] but computed on-the-fly
        # Only slopes are stored permanently
        total_bytes = num_heads * dtype_bytes

    elif pos_type == "relative":
        # [2 * max_length - 1, embed_dim] relative position embeddings
        total_bytes = (2 * max_length - 1) * embed_dim * dtype_bytes

    elif pos_type == "none":
        total_bytes = 0.0

    return total_bytes * bytes_to_mb


def format_positional_stats(
    pos_type: PositionalTypeStr,
    max_length: int,
    memory_mb: float,
    embed_dim: int | None = None,
    num_heads: int | None = None,
) -> str:
    """Format positional encoding statistics as a human-readable string.

    Args:
        pos_type: Type of positional encoding.
        max_length: Maximum sequence length.
        memory_mb: Memory usage in megabytes.
        embed_dim: Embedding dimension. Defaults to None.
        num_heads: Number of attention heads. Defaults to None.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = format_positional_stats("sinusoidal", 512, 1.5, embed_dim=768)
        >>> "Type: sinusoidal" in stats
        True
        >>> "Max Length: 512" in stats
        True

        >>> stats = format_positional_stats("alibi", 4096, 0.001, num_heads=32)
        >>> "Num Heads: 32" in stats
        True
    """
    lines = [
        f"Type: {pos_type}",
        f"Max Length: {max_length}",
        f"Memory: {memory_mb:.3f} MB",
    ]

    if embed_dim is not None:
        lines.append(f"Embed Dim: {embed_dim}")

    if num_heads is not None:
        lines.append(f"Num Heads: {num_heads}")

    return "\n".join(lines)


def get_recommended_positional_config(
    model_type: str = "decoder",
    context_length: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
) -> PositionalConfig:
    """Get recommended positional encoding configuration.

    Provides sensible defaults based on modern transformer practices:
    - Decoder-only models: RoPE (most common for LLMs)
    - Encoder models: Sinusoidal or Learned
    - Long-context models: RoPE with scaling

    Args:
        model_type: Type of model ("decoder", "encoder", "encoder-decoder").
        context_length: Maximum context length.
        num_heads: Number of attention heads. Defaults to 32.
        head_dim: Dimension per head. Defaults to 128.

    Returns:
        Recommended PositionalConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_positional_config(model_type="decoder")
        >>> config.pos_type
        <PositionalType.ROTARY: 'rotary'>

        >>> config = get_recommended_positional_config(model_type="encoder")
        >>> config.pos_type
        <PositionalType.SINUSOIDAL: 'sinusoidal'>

        >>> config = get_recommended_positional_config(context_length=32000)
        >>> config.rope_config.scaling_type
        <RoPEScaling.DYNAMIC: 'dynamic'>
    """
    valid_model_types = {"decoder", "encoder", "encoder-decoder"}
    if model_type not in valid_model_types:
        msg = f"model_type must be one of {valid_model_types}, got '{model_type}'"
        raise ValueError(msg)

    if context_length <= 0:
        msg = f"context_length must be positive, got {context_length}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if head_dim <= 0:
        msg = f"head_dim must be positive, got {head_dim}"
        raise ValueError(msg)

    if model_type in ("encoder", "encoder-decoder"):
        # Encoder models typically use sinusoidal or learned positions
        return PositionalConfig(
            pos_type=PositionalType.SINUSOIDAL,
            rope_config=None,
            alibi_config=None,
            max_length=context_length,
        )

    # Decoder-only models use RoPE
    # For very long contexts, use dynamic scaling
    scaling_type = None
    scaling_factor = 1.0

    if context_length > 8192:
        scaling_type = RoPEScaling.DYNAMIC
        scaling_factor = context_length / 8192.0

    rope_config = RoPEConfig(
        dim=head_dim,
        max_position=context_length,
        base=10000.0,
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
    )

    return PositionalConfig(
        pos_type=PositionalType.ROTARY,
        rope_config=rope_config,
        alibi_config=None,
        max_length=context_length,
    )
