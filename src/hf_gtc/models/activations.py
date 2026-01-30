"""Activation functions for transformer models (GELU, SwiGLU, GeGLU, Mish).

This module provides configuration and utilities for various activation functions
commonly used in transformer architectures, including GELU variants, gated linear
units (GLU), and modern alternatives like Mish.

Examples:
    >>> from hf_gtc.models.activations import create_activation_config, ActivationType
    >>> config = create_activation_config(activation_type="gelu")
    >>> config.activation_type
    <ActivationType.GELU: 'gelu'>
    >>> config.gelu_config.approximate
    <GELUApproximation.NONE: 'none'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ActivationType(Enum):
    """Supported activation function types.

    Attributes:
        RELU: Rectified Linear Unit.
        GELU: Gaussian Error Linear Unit.
        GELU_NEW: New GELU approximation used in some models.
        SILU: Sigmoid Linear Unit (also known as Swish).
        SWIGLU: SwiGLU gated linear unit.
        GEGLU: GeGLU gated linear unit.
        MISH: Mish activation function.
        TANH: Hyperbolic tangent.
        SIGMOID: Sigmoid activation.

    Examples:
        >>> ActivationType.RELU.value
        'relu'
        >>> ActivationType.GELU.value
        'gelu'
        >>> ActivationType.GELU_NEW.value
        'gelu_new'
        >>> ActivationType.SILU.value
        'silu'
        >>> ActivationType.SWIGLU.value
        'swiglu'
        >>> ActivationType.GEGLU.value
        'geglu'
        >>> ActivationType.MISH.value
        'mish'
        >>> ActivationType.TANH.value
        'tanh'
        >>> ActivationType.SIGMOID.value
        'sigmoid'
    """

    RELU = "relu"
    GELU = "gelu"
    GELU_NEW = "gelu_new"
    SILU = "silu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    MISH = "mish"
    TANH = "tanh"
    SIGMOID = "sigmoid"


VALID_ACTIVATION_TYPES = frozenset(t.value for t in ActivationType)


class GELUApproximation(Enum):
    """GELU approximation methods.

    Attributes:
        NONE: Exact GELU computation.
        TANH: Tanh-based approximation (faster but less accurate).
        SIGMOID: Sigmoid-based approximation.

    Examples:
        >>> GELUApproximation.NONE.value
        'none'
        >>> GELUApproximation.TANH.value
        'tanh'
        >>> GELUApproximation.SIGMOID.value
        'sigmoid'
    """

    NONE = "none"
    TANH = "tanh"
    SIGMOID = "sigmoid"


VALID_GELU_APPROXIMATIONS = frozenset(a.value for a in GELUApproximation)


class GLUVariant(Enum):
    """Gated Linear Unit variants.

    Attributes:
        SWIGLU: SwiGLU using SiLU (Swish) as the gating function.
        GEGLU: GeGLU using GELU as the gating function.
        REGLU: ReGLU using ReLU as the gating function.
        BILINEAR: Standard bilinear GLU.

    Examples:
        >>> GLUVariant.SWIGLU.value
        'swiglu'
        >>> GLUVariant.GEGLU.value
        'geglu'
        >>> GLUVariant.REGLU.value
        'reglu'
        >>> GLUVariant.BILINEAR.value
        'bilinear'
    """

    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"
    BILINEAR = "bilinear"


VALID_GLU_VARIANTS = frozenset(v.value for v in GLUVariant)


@dataclass(frozen=True, slots=True)
class GELUConfig:
    """Configuration for GELU activation.

    Attributes:
        approximate: Approximation method to use.

    Examples:
        >>> config = GELUConfig(approximate=GELUApproximation.NONE)
        >>> config.approximate
        <GELUApproximation.NONE: 'none'>

        >>> config = GELUConfig(approximate=GELUApproximation.TANH)
        >>> config.approximate.value
        'tanh'
    """

    approximate: GELUApproximation


@dataclass(frozen=True, slots=True)
class SwiGLUConfig:
    """Configuration for SwiGLU/GLU-variant activations.

    Attributes:
        hidden_dim: Hidden dimension of the FFN.
        bias: Whether to use bias in linear layers.
        gate_dim: Dimension of the gate projection (defaults to hidden_dim).

    Examples:
        >>> config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=4096)
        >>> config.hidden_dim
        4096
        >>> config.bias
        False

        >>> config = SwiGLUConfig(hidden_dim=2048, bias=True, gate_dim=2048)
        >>> config.gate_dim
        2048
    """

    hidden_dim: int
    bias: bool
    gate_dim: int


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """Unified activation configuration.

    Attributes:
        activation_type: Type of activation function.
        gelu_config: GELU-specific configuration.
        swiglu_config: SwiGLU-specific configuration.
        inplace: Whether to perform operation in-place.

    Examples:
        >>> from hf_gtc.models.activations import create_activation_config
        >>> config = create_activation_config(activation_type="gelu")
        >>> config.activation_type
        <ActivationType.GELU: 'gelu'>
        >>> config.gelu_config is not None
        True
    """

    activation_type: ActivationType
    gelu_config: GELUConfig | None
    swiglu_config: SwiGLUConfig | None
    inplace: bool


@dataclass(frozen=True, slots=True)
class ActivationStats:
    """Statistics for activation function properties.

    Attributes:
        memory_overhead: Memory overhead factor relative to input size.
        compute_cost: Relative computational cost (1.0 = ReLU baseline).
        gradient_stability: Gradient stability score (higher = more stable).

    Examples:
        >>> stats = ActivationStats(
        ...     memory_overhead=1.0,
        ...     compute_cost=1.5,
        ...     gradient_stability=0.9,
        ... )
        >>> stats.memory_overhead
        1.0
        >>> stats.compute_cost
        1.5
        >>> stats.gradient_stability
        0.9
    """

    memory_overhead: float
    compute_cost: float
    gradient_stability: float


def validate_gelu_config(config: GELUConfig) -> None:
    """Validate GELU configuration.

    Args:
        config: GELU configuration to validate.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = GELUConfig(approximate=GELUApproximation.NONE)
        >>> validate_gelu_config(config)  # No error

        >>> validate_gelu_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)


def validate_swiglu_config(config: SwiGLUConfig) -> None:
    """Validate SwiGLU configuration.

    Args:
        config: SwiGLU configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If hidden_dim is not positive.
        ValueError: If gate_dim is not positive.

    Examples:
        >>> config = SwiGLUConfig(hidden_dim=4096, bias=False, gate_dim=4096)
        >>> validate_swiglu_config(config)  # No error

        >>> validate_swiglu_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = SwiGLUConfig(hidden_dim=0, bias=False, gate_dim=4096)
        >>> validate_swiglu_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {config.hidden_dim}"
        raise ValueError(msg)

    if config.gate_dim <= 0:
        msg = f"gate_dim must be positive, got {config.gate_dim}"
        raise ValueError(msg)


def validate_activation_config(config: ActivationConfig) -> None:
    """Validate activation configuration.

    Args:
        config: Activation configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If GELU-type activation lacks gelu_config.
        ValueError: If GLU-type activation lacks swiglu_config.

    Examples:
        >>> from hf_gtc.models.activations import create_activation_config
        >>> config = create_activation_config(activation_type="gelu")
        >>> validate_activation_config(config)  # No error

        >>> validate_activation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    gelu_types = (ActivationType.GELU, ActivationType.GELU_NEW)
    if config.activation_type in gelu_types and config.gelu_config is None:
        msg = f"gelu_config required for {config.activation_type.value}"
        raise ValueError(msg)

    glu_types = (ActivationType.SWIGLU, ActivationType.GEGLU)
    if config.activation_type in glu_types and config.swiglu_config is None:
        msg = f"swiglu_config required for {config.activation_type.value}"
        raise ValueError(msg)

    if config.gelu_config is not None:
        validate_gelu_config(config.gelu_config)

    if config.swiglu_config is not None:
        validate_swiglu_config(config.swiglu_config)


def validate_activation_stats(stats: ActivationStats) -> None:
    """Validate activation statistics.

    Args:
        stats: Activation statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If memory_overhead is not positive.
        ValueError: If compute_cost is not positive.
        ValueError: If gradient_stability is not in [0, 1].

    Examples:
        >>> stats = ActivationStats(
        ...     memory_overhead=1.0,
        ...     compute_cost=1.5,
        ...     gradient_stability=0.9,
        ... )
        >>> validate_activation_stats(stats)  # No error

        >>> validate_activation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad_stats = ActivationStats(
        ...     memory_overhead=0.0,
        ...     compute_cost=1.0,
        ...     gradient_stability=0.5,
        ... )
        >>> validate_activation_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory_overhead must be positive
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.memory_overhead <= 0:
        msg = f"memory_overhead must be positive, got {stats.memory_overhead}"
        raise ValueError(msg)

    if stats.compute_cost <= 0:
        msg = f"compute_cost must be positive, got {stats.compute_cost}"
        raise ValueError(msg)

    if not 0.0 <= stats.gradient_stability <= 1.0:
        msg = (
            f"gradient_stability must be in [0, 1], got {stats.gradient_stability}"
        )
        raise ValueError(msg)


def create_gelu_config(
    approximate: str = "none",
) -> GELUConfig:
    """Create a GELU configuration.

    Args:
        approximate: Approximation method ("none", "tanh", "sigmoid").
            Defaults to "none".

    Returns:
        Validated GELUConfig instance.

    Raises:
        ValueError: If approximate is not valid.

    Examples:
        >>> config = create_gelu_config()
        >>> config.approximate
        <GELUApproximation.NONE: 'none'>

        >>> config = create_gelu_config(approximate="tanh")
        >>> config.approximate
        <GELUApproximation.TANH: 'tanh'>

        >>> create_gelu_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     approximate="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: approximate must be one of...
    """
    if approximate not in VALID_GELU_APPROXIMATIONS:
        msg = (
            f"approximate must be one of {VALID_GELU_APPROXIMATIONS}, "
            f"got '{approximate}'"
        )
        raise ValueError(msg)

    config = GELUConfig(approximate=GELUApproximation(approximate))
    validate_gelu_config(config)
    return config


def create_swiglu_config(
    hidden_dim: int = 4096,
    bias: bool = False,
    gate_dim: int | None = None,
) -> SwiGLUConfig:
    """Create a SwiGLU configuration.

    Args:
        hidden_dim: Hidden dimension of the FFN. Defaults to 4096.
        bias: Whether to use bias. Defaults to False.
        gate_dim: Gate projection dimension. Defaults to hidden_dim.

    Returns:
        Validated SwiGLUConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_swiglu_config(hidden_dim=4096)
        >>> config.hidden_dim
        4096
        >>> config.gate_dim
        4096

        >>> config = create_swiglu_config(hidden_dim=2048, gate_dim=1024)
        >>> config.gate_dim
        1024

        >>> create_swiglu_config(hidden_dim=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    if gate_dim is None:
        gate_dim = hidden_dim

    config = SwiGLUConfig(
        hidden_dim=hidden_dim,
        bias=bias,
        gate_dim=gate_dim,
    )
    validate_swiglu_config(config)
    return config


def create_activation_config(
    activation_type: str = "gelu",
    approximate: str = "none",
    hidden_dim: int = 4096,
    bias: bool = False,
    gate_dim: int | None = None,
    inplace: bool = False,
) -> ActivationConfig:
    """Create a unified activation configuration.

    Args:
        activation_type: Type of activation function. Defaults to "gelu".
        approximate: GELU approximation method. Defaults to "none".
        hidden_dim: Hidden dimension for GLU variants. Defaults to 4096.
        bias: Whether to use bias in GLU layers. Defaults to False.
        gate_dim: Gate dimension for GLU variants. Defaults to hidden_dim.
        inplace: Whether to perform in-place. Defaults to False.

    Returns:
        Validated ActivationConfig instance.

    Raises:
        ValueError: If activation_type is not valid.

    Examples:
        >>> config = create_activation_config(activation_type="gelu")
        >>> config.activation_type
        <ActivationType.GELU: 'gelu'>
        >>> config.gelu_config.approximate
        <GELUApproximation.NONE: 'none'>

        >>> config = create_activation_config(activation_type="swiglu", hidden_dim=2048)
        >>> config.activation_type
        <ActivationType.SWIGLU: 'swiglu'>
        >>> config.swiglu_config.hidden_dim
        2048

        >>> config = create_activation_config(activation_type="relu")
        >>> config.activation_type
        <ActivationType.RELU: 'relu'>

        >>> create_activation_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     activation_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: activation_type must be one of...
    """
    if activation_type not in VALID_ACTIVATION_TYPES:
        msg = (
            f"activation_type must be one of {VALID_ACTIVATION_TYPES}, "
            f"got '{activation_type}'"
        )
        raise ValueError(msg)

    activation_type_enum = ActivationType(activation_type)

    gelu_config: GELUConfig | None = None
    swiglu_config: SwiGLUConfig | None = None

    gelu_types = (ActivationType.GELU, ActivationType.GELU_NEW)
    if activation_type_enum in gelu_types:
        gelu_config = create_gelu_config(approximate=approximate)

    glu_types = (ActivationType.SWIGLU, ActivationType.GEGLU)
    if activation_type_enum in glu_types:
        swiglu_config = create_swiglu_config(
            hidden_dim=hidden_dim,
            bias=bias,
            gate_dim=gate_dim,
        )

    config = ActivationConfig(
        activation_type=activation_type_enum,
        gelu_config=gelu_config,
        swiglu_config=swiglu_config,
        inplace=inplace,
    )
    validate_activation_config(config)
    return config


def create_activation_stats(
    activation_type: str,
) -> ActivationStats:
    """Create activation statistics for a given activation type.

    Args:
        activation_type: Type of activation function.

    Returns:
        ActivationStats with properties for the activation type.

    Raises:
        ValueError: If activation_type is not valid.

    Examples:
        >>> stats = create_activation_stats("relu")
        >>> stats.compute_cost
        1.0
        >>> stats.gradient_stability
        0.7

        >>> stats = create_activation_stats("gelu")
        >>> stats.compute_cost > 1.0
        True

        >>> create_activation_stats("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: activation_type must be one of...
    """
    if activation_type not in VALID_ACTIVATION_TYPES:
        msg = (
            f"activation_type must be one of {VALID_ACTIVATION_TYPES}, "
            f"got '{activation_type}'"
        )
        raise ValueError(msg)

    # Activation-specific properties (relative to ReLU baseline)
    properties = {
        "relu": (1.0, 1.0, 0.7),  # (memory, compute, stability)
        "gelu": (1.0, 2.0, 0.9),
        "gelu_new": (1.0, 1.5, 0.9),
        "silu": (1.0, 1.8, 0.85),
        "swiglu": (2.0, 2.5, 0.95),  # GLU has 2x params
        "geglu": (2.0, 3.0, 0.95),
        "mish": (1.0, 2.2, 0.88),
        "tanh": (1.0, 1.2, 0.8),
        "sigmoid": (1.0, 1.1, 0.75),
    }

    memory, compute, stability = properties[activation_type]
    return ActivationStats(
        memory_overhead=memory,
        compute_cost=compute,
        gradient_stability=stability,
    )


def calculate_activation_memory(
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    activation_type: str,
    dtype_bytes: int = 2,
) -> float:
    """Calculate memory usage for activation tensors.

    Args:
        batch_size: Batch size.
        seq_length: Sequence length.
        hidden_dim: Hidden dimension.
        activation_type: Type of activation function.
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32). Defaults to 2.

    Returns:
        Memory usage in megabytes.

    Raises:
        ValueError: If any dimension is not positive.
        ValueError: If activation_type is not valid.
        ValueError: If dtype_bytes is not valid.

    Examples:
        >>> mem = calculate_activation_memory(
        ...     batch_size=1, seq_length=512, hidden_dim=4096, activation_type="relu"
        ... )
        >>> mem > 0
        True

        >>> mem_glu = calculate_activation_memory(
        ...     batch_size=1, seq_length=512, hidden_dim=4096, activation_type="swiglu"
        ... )
        >>> mem_glu > mem
        True

        >>> calculate_activation_memory(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     batch_size=0, seq_length=512, hidden_dim=4096, activation_type="relu"
        ... )
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if seq_length <= 0:
        msg = f"seq_length must be positive, got {seq_length}"
        raise ValueError(msg)

    if hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {hidden_dim}"
        raise ValueError(msg)

    if activation_type not in VALID_ACTIVATION_TYPES:
        msg = (
            f"activation_type must be one of {VALID_ACTIVATION_TYPES}, "
            f"got '{activation_type}'"
        )
        raise ValueError(msg)

    if dtype_bytes not in (2, 4):
        msg = f"dtype_bytes must be 2 (fp16) or 4 (fp32), got {dtype_bytes}"
        raise ValueError(msg)

    stats = create_activation_stats(activation_type)
    base_elements = batch_size * seq_length * hidden_dim
    total_elements = base_elements * stats.memory_overhead
    memory_bytes = total_elements * dtype_bytes
    return memory_bytes / (1024 * 1024)


def estimate_glu_expansion(
    hidden_dim: int,
    intermediate_ratio: float = 4.0,
) -> tuple[int, int]:
    """Estimate GLU layer dimensions for a given hidden dimension.

    GLU variants require intermediate dimension expansion. This function
    calculates the required gate and up projection dimensions.

    Args:
        hidden_dim: Model hidden dimension.
        intermediate_ratio: Ratio of intermediate to hidden dim. Defaults to 4.0.

    Returns:
        Tuple of (gate_dim, up_dim) for the GLU layer.

    Raises:
        ValueError: If hidden_dim is not positive.
        ValueError: If intermediate_ratio is not positive.

    Examples:
        >>> gate, up = estimate_glu_expansion(4096)
        >>> gate == up
        True
        >>> gate > 4096
        True

        >>> gate, up = estimate_glu_expansion(4096, intermediate_ratio=8.0 / 3.0)
        >>> gate == int(4096 * 8.0 / 3.0)
        True

        >>> estimate_glu_expansion(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    if hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {hidden_dim}"
        raise ValueError(msg)

    if intermediate_ratio <= 0:
        msg = f"intermediate_ratio must be positive, got {intermediate_ratio}"
        raise ValueError(msg)

    # For GLU variants, the intermediate dimension is typically:
    # intermediate_dim = hidden_dim * intermediate_ratio
    # But since GLU splits into gate and value, each is intermediate_dim
    intermediate_dim = int(hidden_dim * intermediate_ratio)
    return intermediate_dim, intermediate_dim


def compare_activation_properties(
    activation_types: list[str],
) -> dict[str, ActivationStats]:
    """Compare properties of multiple activation functions.

    Args:
        activation_types: List of activation type names to compare.

    Returns:
        Dictionary mapping activation type to its stats.

    Raises:
        ValueError: If activation_types is empty.
        ValueError: If any activation_type is not valid.

    Examples:
        >>> results = compare_activation_properties(["relu", "gelu", "swiglu"])
        >>> "relu" in results
        True
        >>> results["relu"].compute_cost < results["swiglu"].compute_cost
        True

        >>> results = compare_activation_properties(["gelu", "gelu_new"])
        >>> results["gelu_new"].compute_cost < results["gelu"].compute_cost
        True

        >>> compare_activation_properties([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: activation_types cannot be empty

        >>> compare_activation_properties(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     ["invalid"]
        ... )
        Traceback (most recent call last):
        ValueError: activation_type must be one of...
    """
    if not activation_types:
        msg = "activation_types cannot be empty"
        raise ValueError(msg)

    return {
        act_type: create_activation_stats(act_type) for act_type in activation_types
    }


def calculate_gradient_magnitude(
    activation_type: str,
    input_value: float,
) -> float:
    """Calculate approximate gradient magnitude for an activation at a point.

    This provides a rough estimate of the gradient for analysis purposes.

    Args:
        activation_type: Type of activation function.
        input_value: Input value to compute gradient at.

    Returns:
        Approximate gradient magnitude.

    Raises:
        ValueError: If activation_type is not valid.

    Examples:
        >>> grad = calculate_gradient_magnitude("relu", 1.0)
        >>> grad == 1.0
        True

        >>> grad = calculate_gradient_magnitude("relu", -1.0)
        >>> grad == 0.0
        True

        >>> grad = calculate_gradient_magnitude("sigmoid", 0.0)
        >>> 0.24 < grad < 0.26
        True

        >>> grad = calculate_gradient_magnitude("tanh", 0.0)
        >>> grad == 1.0
        True

        >>> calculate_gradient_magnitude(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid", 0.0
        ... )
        Traceback (most recent call last):
        ValueError: activation_type must be one of...
    """
    if activation_type not in VALID_ACTIVATION_TYPES:
        msg = (
            f"activation_type must be one of {VALID_ACTIVATION_TYPES}, "
            f"got '{activation_type}'"
        )
        raise ValueError(msg)

    x = input_value

    if activation_type == "relu":
        return 1.0 if x > 0 else 0.0

    if activation_type == "sigmoid":
        sig = 1.0 / (1.0 + math.exp(-x))
        return sig * (1.0 - sig)

    if activation_type == "tanh":
        tanh_val = math.tanh(x)
        return 1.0 - tanh_val * tanh_val

    if activation_type in ("gelu", "gelu_new"):
        # Approximate GELU gradient
        # GELU'(x) ~ Phi(x) + x * phi(x)
        # where Phi is CDF and phi is PDF of standard normal
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return cdf + x * pdf

    if activation_type == "silu":
        # SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sig = 1.0 / (1.0 + math.exp(-x))
        return sig + x * sig * (1.0 - sig)

    if activation_type == "mish":
        # Mish'(x) is complex, approximate
        sp = math.log(1.0 + math.exp(x))  # softplus
        tanh_sp = math.tanh(sp)
        sig = 1.0 / (1.0 + math.exp(-x))
        return tanh_sp + x * sig * (1.0 - tanh_sp * tanh_sp)

    if activation_type in ("swiglu", "geglu"):
        # GLU gradients depend on both gate and value
        # Return a representative value for the gating function
        if activation_type == "swiglu":
            sig = 1.0 / (1.0 + math.exp(-x))
            return sig + x * sig * (1.0 - sig)
        else:  # geglu
            cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
            return cdf + x * pdf

    # Should not reach here
    return 0.0


def format_activation_stats(stats: ActivationStats) -> str:
    """Format activation statistics for display.

    Args:
        stats: Activation statistics to format.

    Returns:
        Formatted string with stats breakdown.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = ActivationStats(
        ...     memory_overhead=1.0,
        ...     compute_cost=2.0,
        ...     gradient_stability=0.9,
        ... )
        >>> formatted = format_activation_stats(stats)
        >>> "Memory Overhead: 1.00x" in formatted
        True
        >>> "Compute Cost: 2.00x" in formatted
        True
        >>> "Gradient Stability: 0.90" in formatted
        True

        >>> format_activation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        f"Memory Overhead: {stats.memory_overhead:.2f}x",
        f"Compute Cost: {stats.compute_cost:.2f}x",
        f"Gradient Stability: {stats.gradient_stability:.2f}",
    ]
    return "\n".join(lines)


def get_recommended_activation_config(
    model_type: str = "llm",
    efficiency_priority: bool = False,
) -> ActivationConfig:
    """Get recommended activation configuration for a model type.

    Args:
        model_type: Model type ("llm", "vision", "encoder"). Defaults to "llm".
        efficiency_priority: Whether to prioritize efficiency over quality.
            Defaults to False.

    Returns:
        Recommended ActivationConfig.

    Raises:
        ValueError: If model_type is not recognized.

    Examples:
        >>> config = get_recommended_activation_config("llm")
        >>> config.activation_type
        <ActivationType.SWIGLU: 'swiglu'>

        >>> config = get_recommended_activation_config("llm", efficiency_priority=True)
        >>> config.activation_type
        <ActivationType.GELU_NEW: 'gelu_new'>

        >>> config = get_recommended_activation_config("vision")
        >>> config.activation_type
        <ActivationType.GELU: 'gelu'>

        >>> config = get_recommended_activation_config("encoder")
        >>> config.activation_type
        <ActivationType.GELU: 'gelu'>

        >>> get_recommended_activation_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: model_type must be one of...
    """
    valid_types = {"llm", "vision", "encoder"}
    if model_type not in valid_types:
        msg = f"model_type must be one of {valid_types}, got '{model_type}'"
        raise ValueError(msg)

    if model_type == "llm":
        if efficiency_priority:
            return create_activation_config(
                activation_type="gelu_new", approximate="tanh"
            )
        return create_activation_config(
            activation_type="swiglu", hidden_dim=4096
        )

    # Vision and encoder models typically use GELU
    if efficiency_priority:
        return create_activation_config(activation_type="gelu_new", approximate="tanh")
    return create_activation_config(activation_type="gelu", approximate="none")


def list_activation_types() -> list[str]:
    """List all supported activation types.

    Returns:
        Sorted list of activation type names.

    Examples:
        >>> types = list_activation_types()
        >>> "relu" in types
        True
        >>> "gelu" in types
        True
        >>> "swiglu" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ACTIVATION_TYPES)


def list_gelu_approximations() -> list[str]:
    """List all supported GELU approximation methods.

    Returns:
        Sorted list of approximation method names.

    Examples:
        >>> methods = list_gelu_approximations()
        >>> "none" in methods
        True
        >>> "tanh" in methods
        True
        >>> "sigmoid" in methods
        True
    """
    return sorted(VALID_GELU_APPROXIMATIONS)


def list_glu_variants() -> list[str]:
    """List all supported GLU variants.

    Returns:
        Sorted list of GLU variant names.

    Examples:
        >>> variants = list_glu_variants()
        >>> "swiglu" in variants
        True
        >>> "geglu" in variants
        True
        >>> "reglu" in variants
        True
        >>> "bilinear" in variants
        True
    """
    return sorted(VALID_GLU_VARIANTS)


def get_activation_type(name: str) -> ActivationType:
    """Get activation type enum from string.

    Args:
        name: Activation type name.

    Returns:
        ActivationType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_activation_type("relu")
        <ActivationType.RELU: 'relu'>
        >>> get_activation_type("gelu")
        <ActivationType.GELU: 'gelu'>
        >>> get_activation_type("swiglu")
        <ActivationType.SWIGLU: 'swiglu'>

        >>> get_activation_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid activation type: invalid
    """
    for at in ActivationType:
        if at.value == name:
            return at
    msg = f"invalid activation type: {name}"
    raise ValueError(msg)


def get_gelu_approximation(name: str) -> GELUApproximation:
    """Get GELU approximation enum from string.

    Args:
        name: Approximation method name.

    Returns:
        GELUApproximation enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_gelu_approximation("none")
        <GELUApproximation.NONE: 'none'>
        >>> get_gelu_approximation("tanh")
        <GELUApproximation.TANH: 'tanh'>

        >>> get_gelu_approximation("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid GELU approximation: invalid
    """
    for ga in GELUApproximation:
        if ga.value == name:
            return ga
    msg = f"invalid GELU approximation: {name}"
    raise ValueError(msg)


def get_glu_variant(name: str) -> GLUVariant:
    """Get GLU variant enum from string.

    Args:
        name: GLU variant name.

    Returns:
        GLUVariant enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_glu_variant("swiglu")
        <GLUVariant.SWIGLU: 'swiglu'>
        >>> get_glu_variant("geglu")
        <GLUVariant.GEGLU: 'geglu'>

        >>> get_glu_variant("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid GLU variant: invalid
    """
    for gv in GLUVariant:
        if gv.value == name:
            return gv
    msg = f"invalid GLU variant: {name}"
    raise ValueError(msg)
