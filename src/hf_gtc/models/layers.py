"""Neural network layer configurations for transformer models.

This module provides configuration and utilities for various neural network layers
commonly used in transformer architectures, including MLPs, gated MLPs, and
cross-attention layers.

Examples:
    >>> from hf_gtc.models.layers import create_mlp_config, LayerType
    >>> config = create_mlp_config(hidden_dim=768, intermediate_dim=3072)
    >>> config.hidden_dim
    768
    >>> config.intermediate_dim
    3072

    >>> from hf_gtc.models.layers import create_gated_mlp_config, GatingType
    >>> config = create_gated_mlp_config(hidden_dim=4096, gating_type="swiglu")
    >>> config.gating_type
    <GatingType.SWIGLU: 'swiglu'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class LayerType(Enum):
    """Supported neural network layer types.

    Attributes:
        MLP: Standard Multi-Layer Perceptron.
        FFN: Feed-Forward Network (alias for MLP).
        GATED_MLP: Gated MLP with activation gating.
        CROSS_ATTENTION: Cross-attention layer.
        SELF_ATTENTION: Self-attention layer.
        CONV1D: 1D convolutional layer.

    Examples:
        >>> LayerType.MLP.value
        'mlp'
        >>> LayerType.FFN.value
        'ffn'
        >>> LayerType.GATED_MLP.value
        'gated_mlp'
        >>> LayerType.CROSS_ATTENTION.value
        'cross_attention'
        >>> LayerType.SELF_ATTENTION.value
        'self_attention'
        >>> LayerType.CONV1D.value
        'conv1d'
    """

    MLP = "mlp"
    FFN = "ffn"
    GATED_MLP = "gated_mlp"
    CROSS_ATTENTION = "cross_attention"
    SELF_ATTENTION = "self_attention"
    CONV1D = "conv1d"


VALID_LAYER_TYPES = frozenset(t.value for t in LayerType)


class GatingType(Enum):
    """Gating mechanism types for gated MLPs.

    Attributes:
        NONE: No gating (standard MLP).
        SWIGLU: SwiGLU gating (SiLU activation on gate).
        GEGLU: GeGLU gating (GELU activation on gate).
        REGLU: ReGLU gating (ReLU activation on gate).

    Examples:
        >>> GatingType.NONE.value
        'none'
        >>> GatingType.SWIGLU.value
        'swiglu'
        >>> GatingType.GEGLU.value
        'geglu'
        >>> GatingType.REGLU.value
        'reglu'
    """

    NONE = "none"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"


VALID_GATING_TYPES = frozenset(t.value for t in GatingType)


class ProjectionType(Enum):
    """Projection layer types.

    Attributes:
        LINEAR: Standard linear projection.
        LOW_RANK: Low-rank factorized projection.
        SPARSE: Sparse projection.

    Examples:
        >>> ProjectionType.LINEAR.value
        'linear'
        >>> ProjectionType.LOW_RANK.value
        'low_rank'
        >>> ProjectionType.SPARSE.value
        'sparse'
    """

    LINEAR = "linear"
    LOW_RANK = "low_rank"
    SPARSE = "sparse"


VALID_PROJECTION_TYPES = frozenset(t.value for t in ProjectionType)


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Configuration for standard MLP layer.

    Attributes:
        hidden_dim: Input/output hidden dimension.
        intermediate_dim: Intermediate (expanded) dimension.
        activation: Activation function name.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.

    Examples:
        >>> config = MLPConfig(
        ...     hidden_dim=768,
        ...     intermediate_dim=3072,
        ...     activation="gelu",
        ...     dropout=0.1,
        ...     bias=True,
        ... )
        >>> config.hidden_dim
        768
        >>> config.intermediate_dim
        3072
        >>> config.activation
        'gelu'
    """

    hidden_dim: int
    intermediate_dim: int
    activation: str
    dropout: float
    bias: bool


@dataclass(frozen=True, slots=True)
class GatedMLPConfig:
    """Configuration for gated MLP layer.

    Attributes:
        mlp_config: Base MLP configuration.
        gating_type: Type of gating mechanism.
        gate_dim: Dimension of the gate projection.

    Examples:
        >>> mlp = MLPConfig(
        ...     hidden_dim=4096,
        ...     intermediate_dim=11008,
        ...     activation="silu",
        ...     dropout=0.0,
        ...     bias=False,
        ... )
        >>> config = GatedMLPConfig(
        ...     mlp_config=mlp,
        ...     gating_type=GatingType.SWIGLU,
        ...     gate_dim=11008,
        ... )
        >>> config.gating_type
        <GatingType.SWIGLU: 'swiglu'>
        >>> config.gate_dim
        11008
    """

    mlp_config: MLPConfig
    gating_type: GatingType
    gate_dim: int


@dataclass(frozen=True, slots=True)
class CrossAttentionConfig:
    """Configuration for cross-attention layer.

    Attributes:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        kv_dim: Key-value dimension (from encoder).
        dropout: Dropout probability.

    Examples:
        >>> config = CrossAttentionConfig(
        ...     hidden_dim=768,
        ...     num_heads=12,
        ...     kv_dim=512,
        ...     dropout=0.1,
        ... )
        >>> config.hidden_dim
        768
        >>> config.num_heads
        12
        >>> config.kv_dim
        512
    """

    hidden_dim: int
    num_heads: int
    kv_dim: int
    dropout: float


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Unified layer configuration.

    Attributes:
        layer_type: Type of layer.
        mlp_config: MLP configuration (for MLP/FFN types).
        gated_mlp_config: Gated MLP configuration (for GATED_MLP type).
        cross_attention_config: Cross-attention config (for CROSS_ATTENTION type).

    Examples:
        >>> from hf_gtc.models.layers import create_layer_config
        >>> config = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> config.layer_type
        <LayerType.MLP: 'mlp'>
        >>> config.mlp_config is not None
        True
    """

    layer_type: LayerType
    mlp_config: MLPConfig | None
    gated_mlp_config: GatedMLPConfig | None
    cross_attention_config: CrossAttentionConfig | None


@dataclass(frozen=True, slots=True)
class LayerStats:
    """Statistics for layer computation.

    Attributes:
        params: Number of parameters.
        flops: Floating point operations per forward pass.
        memory_mb: Memory usage in megabytes.

    Examples:
        >>> stats = LayerStats(params=3145728, flops=6291456, memory_mb=12.0)
        >>> stats.params
        3145728
        >>> stats.flops
        6291456
        >>> stats.memory_mb
        12.0
    """

    params: int
    flops: int
    memory_mb: float


def validate_mlp_config(config: MLPConfig) -> None:
    """Validate MLP configuration.

    Args:
        config: MLP configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If hidden_dim is not positive.
        ValueError: If intermediate_dim is not positive.
        ValueError: If dropout is not between 0 and 1.

    Examples:
        >>> config = MLPConfig(
        ...     hidden_dim=768,
        ...     intermediate_dim=3072,
        ...     activation="gelu",
        ...     dropout=0.1,
        ...     bias=True,
        ... )
        >>> validate_mlp_config(config)  # No error

        >>> validate_mlp_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = MLPConfig(
        ...     hidden_dim=0,
        ...     intermediate_dim=3072,
        ...     activation="gelu",
        ...     dropout=0.1,
        ...     bias=True,
        ... )
        >>> validate_mlp_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    validate_not_none(config, "config")

    if config.hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {config.hidden_dim}"
        raise ValueError(msg)

    if config.intermediate_dim <= 0:
        msg = f"intermediate_dim must be positive, got {config.intermediate_dim}"
        raise ValueError(msg)

    if not 0 <= config.dropout <= 1:
        msg = f"dropout must be between 0 and 1, got {config.dropout}"
        raise ValueError(msg)


def validate_gated_mlp_config(config: GatedMLPConfig) -> None:
    """Validate gated MLP configuration.

    Args:
        config: Gated MLP configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If mlp_config is invalid.
        ValueError: If gate_dim is not positive.

    Examples:
        >>> mlp = MLPConfig(
        ...     hidden_dim=4096,
        ...     intermediate_dim=11008,
        ...     activation="silu",
        ...     dropout=0.0,
        ...     bias=False,
        ... )
        >>> config = GatedMLPConfig(
        ...     mlp_config=mlp,
        ...     gating_type=GatingType.SWIGLU,
        ...     gate_dim=11008,
        ... )
        >>> validate_gated_mlp_config(config)  # No error

        >>> validate_gated_mlp_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_mlp_config(config.mlp_config)

    if config.gate_dim <= 0:
        msg = f"gate_dim must be positive, got {config.gate_dim}"
        raise ValueError(msg)


def validate_cross_attention_config(config: CrossAttentionConfig) -> None:
    """Validate cross-attention configuration.

    Args:
        config: Cross-attention configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If hidden_dim is not positive.
        ValueError: If num_heads is not positive.
        ValueError: If kv_dim is not positive.
        ValueError: If dropout is not between 0 and 1.

    Examples:
        >>> config = CrossAttentionConfig(
        ...     hidden_dim=768,
        ...     num_heads=12,
        ...     kv_dim=512,
        ...     dropout=0.1,
        ... )
        >>> validate_cross_attention_config(config)  # No error

        >>> validate_cross_attention_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = CrossAttentionConfig(
        ...     hidden_dim=0,
        ...     num_heads=12,
        ...     kv_dim=512,
        ...     dropout=0.1,
        ... )
        >>> validate_cross_attention_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    validate_not_none(config, "config")

    if config.hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {config.hidden_dim}"
        raise ValueError(msg)

    if config.num_heads <= 0:
        msg = f"num_heads must be positive, got {config.num_heads}"
        raise ValueError(msg)

    if config.kv_dim <= 0:
        msg = f"kv_dim must be positive, got {config.kv_dim}"
        raise ValueError(msg)

    if not 0 <= config.dropout <= 1:
        msg = f"dropout must be between 0 and 1, got {config.dropout}"
        raise ValueError(msg)


def _validate_layer_type_config(config: LayerConfig) -> None:
    """Validate layer sub-config based on layer type."""
    mlp_types = (LayerType.MLP, LayerType.FFN)
    if config.layer_type in mlp_types:
        if config.mlp_config is None:
            msg = f"mlp_config required for {config.layer_type.value}"
            raise ValueError(msg)
        validate_mlp_config(config.mlp_config)
        return

    layer_validators: dict[LayerType, tuple[str, str, object]] = {
        LayerType.GATED_MLP: (
            "gated_mlp_config",
            "gated_mlp",
            validate_gated_mlp_config,
        ),
        LayerType.CROSS_ATTENTION: (
            "cross_attention_config",
            "cross_attention",
            validate_cross_attention_config,
        ),
    }
    entry = layer_validators.get(config.layer_type)
    if entry is None:
        return
    attr_name, label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} required for {label}"
        raise ValueError(msg)
    validator(sub_config)


def validate_layer_config(config: LayerConfig) -> None:
    """Validate layer configuration.

    Args:
        config: Layer configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If required sub-config is missing for layer type.

    Examples:
        >>> from hf_gtc.models.layers import create_layer_config
        >>> config = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> validate_layer_config(config)  # No error

        >>> validate_layer_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    _validate_layer_type_config(config)


def validate_layer_stats(stats: LayerStats) -> None:
    """Validate layer statistics.

    Args:
        stats: Layer statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If params is negative.
        ValueError: If flops is negative.
        ValueError: If memory_mb is negative.

    Examples:
        >>> stats = LayerStats(params=1000000, flops=2000000, memory_mb=4.0)
        >>> validate_layer_stats(stats)  # No error

        >>> validate_layer_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad_stats = LayerStats(params=-1, flops=2000000, memory_mb=4.0)
        >>> validate_layer_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: params must be non-negative
    """
    validate_not_none(stats, "stats")

    if stats.params < 0:
        msg = f"params must be non-negative, got {stats.params}"
        raise ValueError(msg)

    if stats.flops < 0:
        msg = f"flops must be non-negative, got {stats.flops}"
        raise ValueError(msg)

    if stats.memory_mb < 0:
        msg = f"memory_mb must be non-negative, got {stats.memory_mb}"
        raise ValueError(msg)


def create_mlp_config(
    hidden_dim: int = 768,
    intermediate_dim: int | None = None,
    activation: str = "gelu",
    dropout: float = 0.0,
    bias: bool = True,
) -> MLPConfig:
    """Create an MLP configuration.

    Args:
        hidden_dim: Input/output hidden dimension. Defaults to 768.
        intermediate_dim: Intermediate dimension. Defaults to 4 * hidden_dim.
        activation: Activation function name. Defaults to "gelu".
        dropout: Dropout probability. Defaults to 0.0.
        bias: Whether to use bias. Defaults to True.

    Returns:
        Validated MLPConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_mlp_config(hidden_dim=768)
        >>> config.hidden_dim
        768
        >>> config.intermediate_dim
        3072

        >>> config = create_mlp_config(hidden_dim=4096, intermediate_dim=11008)
        >>> config.intermediate_dim
        11008

        >>> create_mlp_config(hidden_dim=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    if intermediate_dim is None:
        intermediate_dim = 4 * hidden_dim

    config = MLPConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        activation=activation,
        dropout=dropout,
        bias=bias,
    )
    validate_mlp_config(config)
    return config


def create_gated_mlp_config(
    hidden_dim: int = 4096,
    intermediate_dim: int | None = None,
    gating_type: str = "swiglu",
    activation: str = "silu",
    dropout: float = 0.0,
    bias: bool = False,
    gate_dim: int | None = None,
) -> GatedMLPConfig:
    """Create a gated MLP configuration.

    Args:
        hidden_dim: Input/output hidden dimension. Defaults to 4096.
        intermediate_dim: Intermediate dimension. Defaults to 8/3 * hidden_dim.
        gating_type: Type of gating mechanism. Defaults to "swiglu".
        activation: Activation function name. Defaults to "silu".
        dropout: Dropout probability. Defaults to 0.0.
        bias: Whether to use bias. Defaults to False.
        gate_dim: Gate projection dimension. Defaults to intermediate_dim.

    Returns:
        Validated GatedMLPConfig instance.

    Raises:
        ValueError: If gating_type is not valid.
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_gated_mlp_config(hidden_dim=4096)
        >>> config.gating_type
        <GatingType.SWIGLU: 'swiglu'>
        >>> config.mlp_config.hidden_dim
        4096

        >>> config = create_gated_mlp_config(gating_type="geglu")
        >>> config.gating_type
        <GatingType.GEGLU: 'geglu'>

        >>> create_gated_mlp_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     gating_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: gating_type must be one of...
    """
    if gating_type not in VALID_GATING_TYPES:
        msg = f"gating_type must be one of {VALID_GATING_TYPES}, got '{gating_type}'"
        raise ValueError(msg)

    # For gated MLPs, common intermediate ratio is 8/3 to maintain param count
    if intermediate_dim is None:
        intermediate_dim = int(hidden_dim * 8 / 3)

    if gate_dim is None:
        gate_dim = intermediate_dim

    mlp_config = MLPConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        activation=activation,
        dropout=dropout,
        bias=bias,
    )

    config = GatedMLPConfig(
        mlp_config=mlp_config,
        gating_type=GatingType(gating_type),
        gate_dim=gate_dim,
    )
    validate_gated_mlp_config(config)
    return config


def create_cross_attention_config(
    hidden_dim: int = 768,
    num_heads: int = 12,
    kv_dim: int | None = None,
    dropout: float = 0.0,
) -> CrossAttentionConfig:
    """Create a cross-attention configuration.

    Args:
        hidden_dim: Hidden dimension. Defaults to 768.
        num_heads: Number of attention heads. Defaults to 12.
        kv_dim: Key-value dimension. Defaults to hidden_dim.
        dropout: Dropout probability. Defaults to 0.0.

    Returns:
        Validated CrossAttentionConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_cross_attention_config(hidden_dim=768, num_heads=12)
        >>> config.hidden_dim
        768
        >>> config.num_heads
        12
        >>> config.kv_dim
        768

        >>> config = create_cross_attention_config(hidden_dim=768, kv_dim=512)
        >>> config.kv_dim
        512

        >>> create_cross_attention_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     hidden_dim=0
        ... )
        Traceback (most recent call last):
        ValueError: hidden_dim must be positive
    """
    if kv_dim is None:
        kv_dim = hidden_dim

    config = CrossAttentionConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        kv_dim=kv_dim,
        dropout=dropout,
    )
    validate_cross_attention_config(config)
    return config


def create_layer_config(
    layer_type: str = "mlp",
    hidden_dim: int = 768,
    intermediate_dim: int | None = None,
    activation: str = "gelu",
    dropout: float = 0.0,
    bias: bool = True,
    gating_type: str = "swiglu",
    num_heads: int = 12,
    kv_dim: int | None = None,
) -> LayerConfig:
    """Create a unified layer configuration.

    Args:
        layer_type: Type of layer. Defaults to "mlp".
        hidden_dim: Hidden dimension. Defaults to 768.
        intermediate_dim: Intermediate dimension. Defaults based on layer type.
        activation: Activation function name. Defaults to "gelu".
        dropout: Dropout probability. Defaults to 0.0.
        bias: Whether to use bias. Defaults to True.
        gating_type: Gating type for gated MLP. Defaults to "swiglu".
        num_heads: Number of attention heads. Defaults to 12.
        kv_dim: Key-value dimension for cross-attention. Defaults to hidden_dim.

    Returns:
        Validated LayerConfig instance.

    Raises:
        ValueError: If layer_type is not valid.
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> config.layer_type
        <LayerType.MLP: 'mlp'>
        >>> config.mlp_config.hidden_dim
        768

        >>> config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        >>> config.layer_type
        <LayerType.GATED_MLP: 'gated_mlp'>
        >>> config.gated_mlp_config is not None
        True

        >>> config = create_layer_config(layer_type="cross_attention")
        >>> config.layer_type
        <LayerType.CROSS_ATTENTION: 'cross_attention'>

        >>> create_layer_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     layer_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: layer_type must be one of...
    """
    if layer_type not in VALID_LAYER_TYPES:
        msg = f"layer_type must be one of {VALID_LAYER_TYPES}, got '{layer_type}'"
        raise ValueError(msg)

    layer_type_enum = LayerType(layer_type)

    mlp_config: MLPConfig | None = None
    gated_mlp_config: GatedMLPConfig | None = None
    cross_attention_config: CrossAttentionConfig | None = None

    if layer_type_enum in (LayerType.MLP, LayerType.FFN):
        mlp_config = create_mlp_config(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

    if layer_type_enum == LayerType.GATED_MLP:
        gated_mlp_config = create_gated_mlp_config(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            gating_type=gating_type,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

    if layer_type_enum == LayerType.CROSS_ATTENTION:
        cross_attention_config = create_cross_attention_config(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            dropout=dropout,
        )

    config = LayerConfig(
        layer_type=layer_type_enum,
        mlp_config=mlp_config,
        gated_mlp_config=gated_mlp_config,
        cross_attention_config=cross_attention_config,
    )
    validate_layer_config(config)
    return config


def list_layer_types() -> list[str]:
    """List all supported layer types.

    Returns:
        Sorted list of layer type names.

    Examples:
        >>> types = list_layer_types()
        >>> "mlp" in types
        True
        >>> "gated_mlp" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_LAYER_TYPES)


def list_gating_types() -> list[str]:
    """List all supported gating types.

    Returns:
        Sorted list of gating type names.

    Examples:
        >>> types = list_gating_types()
        >>> "swiglu" in types
        True
        >>> "geglu" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_GATING_TYPES)


def list_projection_types() -> list[str]:
    """List all supported projection types.

    Returns:
        Sorted list of projection type names.

    Examples:
        >>> types = list_projection_types()
        >>> "linear" in types
        True
        >>> "low_rank" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_PROJECTION_TYPES)


def get_layer_type(name: str) -> LayerType:
    """Get layer type enum from string.

    Args:
        name: Layer type name.

    Returns:
        LayerType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_layer_type("mlp")
        <LayerType.MLP: 'mlp'>
        >>> get_layer_type("gated_mlp")
        <LayerType.GATED_MLP: 'gated_mlp'>

        >>> get_layer_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid layer type: invalid
    """
    for lt in LayerType:
        if lt.value == name:
            return lt
    msg = f"invalid layer type: {name}"
    raise ValueError(msg)


def get_gating_type(name: str) -> GatingType:
    """Get gating type enum from string.

    Args:
        name: Gating type name.

    Returns:
        GatingType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_gating_type("swiglu")
        <GatingType.SWIGLU: 'swiglu'>
        >>> get_gating_type("geglu")
        <GatingType.GEGLU: 'geglu'>

        >>> get_gating_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid gating type: invalid
    """
    for gt in GatingType:
        if gt.value == name:
            return gt
    msg = f"invalid gating type: {name}"
    raise ValueError(msg)


def get_projection_type(name: str) -> ProjectionType:
    """Get projection type enum from string.

    Args:
        name: Projection type name.

    Returns:
        ProjectionType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_projection_type("linear")
        <ProjectionType.LINEAR: 'linear'>
        >>> get_projection_type("low_rank")
        <ProjectionType.LOW_RANK: 'low_rank'>

        >>> get_projection_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid projection type: invalid
    """
    for pt in ProjectionType:
        if pt.value == name:
            return pt
    msg = f"invalid projection type: {name}"
    raise ValueError(msg)


def _mlp_params(config: LayerConfig) -> int:
    """Calculate parameters for MLP/FFN layer types."""
    mlp = config.mlp_config
    if mlp is None:
        msg = "mlp_config required for MLP/FFN layer type"
        raise ValueError(msg)
    up_params = mlp.hidden_dim * mlp.intermediate_dim
    down_params = mlp.intermediate_dim * mlp.hidden_dim
    if mlp.bias:
        up_params += mlp.intermediate_dim
        down_params += mlp.hidden_dim
    return up_params + down_params


def _gated_mlp_params(config: LayerConfig) -> int:
    """Calculate parameters for gated MLP layer type."""
    gated = config.gated_mlp_config
    if gated is None:
        msg = "gated_mlp_config required for GATED_MLP layer type"
        raise ValueError(msg)
    mlp = gated.mlp_config
    gate_params = mlp.hidden_dim * gated.gate_dim
    up_params = mlp.hidden_dim * mlp.intermediate_dim
    down_params = mlp.intermediate_dim * mlp.hidden_dim
    if mlp.bias:
        gate_params += gated.gate_dim
        up_params += mlp.intermediate_dim
        down_params += mlp.hidden_dim
    return gate_params + up_params + down_params


def _cross_attention_params(config: LayerConfig) -> int:
    """Calculate parameters for cross-attention layer type."""
    cross = config.cross_attention_config
    if cross is None:
        msg = "cross_attention_config required for CROSS_ATTENTION layer type"
        raise ValueError(msg)
    q_params = cross.hidden_dim * cross.hidden_dim
    k_params = cross.kv_dim * cross.hidden_dim
    v_params = cross.kv_dim * cross.hidden_dim
    o_params = cross.hidden_dim * cross.hidden_dim
    return q_params + k_params + v_params + o_params


def calculate_layer_params(config: LayerConfig) -> int:
    """Calculate number of parameters for a layer configuration.

    Args:
        config: Layer configuration.

    Returns:
        Number of parameters.

    Raises:
        ValueError: If config is None.
        ValueError: If layer type is not supported for param calculation.

    Examples:
        >>> config = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> params = calculate_layer_params(config)
        >>> params > 0
        True

        >>> config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        >>> params = calculate_layer_params(config)
        >>> params > 0
        True

        >>> calculate_layer_params(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    _layer_param_calculators = {
        LayerType.MLP: _mlp_params,
        LayerType.FFN: _mlp_params,
        LayerType.GATED_MLP: _gated_mlp_params,
        LayerType.CROSS_ATTENTION: _cross_attention_params,
    }

    calculator = _layer_param_calculators.get(config.layer_type)
    if calculator is not None:
        return calculator(config)
    return 0


def estimate_layer_flops(
    config: LayerConfig,
    batch_size: int = 1,
    seq_length: int = 512,
) -> int:
    """Estimate FLOPs for a layer forward pass.

    Args:
        config: Layer configuration.
        batch_size: Batch size. Defaults to 1.
        seq_length: Sequence length. Defaults to 512.

    Returns:
        Estimated FLOPs.

    Raises:
        ValueError: If config is None.
        ValueError: If batch_size is not positive.
        ValueError: If seq_length is not positive.

    Examples:
        >>> config = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> flops = estimate_layer_flops(config)
        >>> flops > 0
        True

        >>> config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        >>> flops = estimate_layer_flops(config, batch_size=2, seq_length=1024)
        >>> flops > 0
        True

        >>> estimate_layer_flops(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if seq_length <= 0:
        msg = f"seq_length must be positive, got {seq_length}"
        raise ValueError(msg)

    tokens = batch_size * seq_length

    if config.layer_type in (LayerType.MLP, LayerType.FFN):
        mlp = config.mlp_config
        if mlp is None:
            return 0
        # 2 * for multiply-add
        up_flops = 2 * tokens * mlp.hidden_dim * mlp.intermediate_dim
        down_flops = 2 * tokens * mlp.intermediate_dim * mlp.hidden_dim
        return up_flops + down_flops

    if config.layer_type == LayerType.GATED_MLP:
        gated = config.gated_mlp_config
        if gated is None:
            return 0
        mlp = gated.mlp_config
        gate_flops = 2 * tokens * mlp.hidden_dim * gated.gate_dim
        up_flops = 2 * tokens * mlp.hidden_dim * mlp.intermediate_dim
        down_flops = 2 * tokens * mlp.intermediate_dim * mlp.hidden_dim
        # Element-wise multiply for gating
        elem_flops = tokens * mlp.intermediate_dim
        return gate_flops + up_flops + down_flops + elem_flops

    if config.layer_type == LayerType.CROSS_ATTENTION:
        cross = config.cross_attention_config
        if cross is None:
            return 0
        # Q, K, V projections
        qkv_flops = 2 * tokens * cross.hidden_dim * cross.hidden_dim * 3
        # Attention scores and output
        attn_flops = 2 * tokens * seq_length * cross.hidden_dim
        # Output projection
        out_flops = 2 * tokens * cross.hidden_dim * cross.hidden_dim
        return qkv_flops + attn_flops + out_flops

    return 0


def calculate_layer_memory(
    config: LayerConfig,
    batch_size: int = 1,
    seq_length: int = 512,
    dtype_bytes: int = 2,
) -> float:
    """Calculate memory usage for a layer.

    Args:
        config: Layer configuration.
        batch_size: Batch size. Defaults to 1.
        seq_length: Sequence length. Defaults to 512.
        dtype_bytes: Bytes per element. Defaults to 2 (fp16).

    Returns:
        Memory usage in megabytes.

    Raises:
        ValueError: If config is None.
        ValueError: If batch_size is not positive.
        ValueError: If seq_length is not positive.
        ValueError: If dtype_bytes is not valid.

    Examples:
        >>> config = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> mem = calculate_layer_memory(config)
        >>> mem > 0
        True

        >>> config = create_layer_config(layer_type="gated_mlp", hidden_dim=4096)
        >>> mem = calculate_layer_memory(config, batch_size=2, seq_length=1024)
        >>> mem > 0
        True

        >>> calculate_layer_memory(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if seq_length <= 0:
        msg = f"seq_length must be positive, got {seq_length}"
        raise ValueError(msg)

    if dtype_bytes not in (2, 4):
        msg = f"dtype_bytes must be 2 (fp16) or 4 (fp32), got {dtype_bytes}"
        raise ValueError(msg)

    tokens = batch_size * seq_length

    # Calculate parameter memory
    params = calculate_layer_params(config)
    param_memory = params * dtype_bytes

    # Calculate activation memory
    if config.layer_type in (LayerType.MLP, LayerType.FFN):
        mlp = config.mlp_config
        if mlp is None:
            return param_memory / (1024 * 1024)
        # Input, intermediate, output activations
        act_memory = tokens * (mlp.hidden_dim + mlp.intermediate_dim + mlp.hidden_dim)
        act_memory *= dtype_bytes

    elif config.layer_type == LayerType.GATED_MLP:
        gated = config.gated_mlp_config
        if gated is None:
            return param_memory / (1024 * 1024)
        mlp = gated.mlp_config
        # Input, gate, up, combined, output activations
        act_memory = tokens * (
            mlp.hidden_dim + gated.gate_dim + mlp.intermediate_dim * 2 + mlp.hidden_dim
        )
        act_memory *= dtype_bytes

    elif config.layer_type == LayerType.CROSS_ATTENTION:
        cross = config.cross_attention_config
        if cross is None:
            return param_memory / (1024 * 1024)
        # Q, K, V, attention scores, output
        act_memory = tokens * cross.hidden_dim * 4
        act_memory += tokens * seq_length * cross.num_heads
        act_memory *= dtype_bytes

    else:
        act_memory = 0

    total_memory = param_memory + act_memory
    return total_memory / (1024 * 1024)


def compare_layer_configs(
    configs: list[LayerConfig],
) -> dict[str, LayerStats]:
    """Compare multiple layer configurations.

    Args:
        configs: List of layer configurations to compare.

    Returns:
        Dictionary mapping layer type to its stats.

    Raises:
        ValueError: If configs is empty.
        ValueError: If any config is None.

    Examples:
        >>> mlp = create_layer_config(layer_type="mlp", hidden_dim=768)
        >>> gated = create_layer_config(layer_type="gated_mlp", hidden_dim=768)
        >>> results = compare_layer_configs([mlp, gated])
        >>> "mlp" in results
        True
        >>> "gated_mlp" in results
        True

        >>> compare_layer_configs([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: configs cannot be empty
    """
    if not configs:
        msg = "configs cannot be empty"
        raise ValueError(msg)

    results: dict[str, LayerStats] = {}

    for config in configs:
        if config is None:
            msg = "config cannot be None"
            raise ValueError(msg)

        params = calculate_layer_params(config)
        flops = estimate_layer_flops(config)
        memory = calculate_layer_memory(config)

        stats = LayerStats(params=params, flops=flops, memory_mb=memory)
        results[config.layer_type.value] = stats

    return results


def format_layer_stats(stats: LayerStats) -> str:
    """Format layer statistics for display.

    Args:
        stats: Layer statistics to format.

    Returns:
        Formatted string with stats breakdown.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = LayerStats(params=3145728, flops=6291456, memory_mb=12.0)
        >>> formatted = format_layer_stats(stats)
        >>> "Parameters:" in formatted
        True
        >>> "FLOPs:" in formatted
        True
        >>> "Memory:" in formatted
        True

        >>> format_layer_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    def format_number(n: int) -> str:
        """Format large numbers with units."""
        if n >= 1e12:
            return f"{n / 1e12:.2f}T"
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        if n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)

    lines = [
        f"Parameters: {format_number(stats.params)}",
        f"FLOPs: {format_number(stats.flops)}",
        f"Memory: {stats.memory_mb:.2f} MB",
    ]

    return "\n".join(lines)


def get_recommended_layer_config(
    model_type: str = "llm",
    hidden_dim: int = 4096,
    efficiency_priority: bool = False,
) -> LayerConfig:
    """Get recommended layer configuration for a model type.

    Args:
        model_type: Model type ("llm", "encoder", "decoder"). Defaults to "llm".
        hidden_dim: Hidden dimension. Defaults to 4096.
        efficiency_priority: Whether to prioritize efficiency. Defaults to False.

    Returns:
        Recommended LayerConfig.

    Raises:
        ValueError: If model_type is not recognized.

    Examples:
        >>> config = get_recommended_layer_config("llm")
        >>> config.layer_type
        <LayerType.GATED_MLP: 'gated_mlp'>

        >>> config = get_recommended_layer_config("llm", efficiency_priority=True)
        >>> config.layer_type
        <LayerType.MLP: 'mlp'>

        >>> config = get_recommended_layer_config("encoder")
        >>> config.layer_type
        <LayerType.MLP: 'mlp'>

        >>> get_recommended_layer_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of...
    """
    valid_types = {"llm", "encoder", "decoder"}
    if model_type not in valid_types:
        msg = f"model_type must be one of {valid_types}, got '{model_type}'"
        raise ValueError(msg)

    if model_type == "llm":
        if efficiency_priority:
            return create_layer_config(
                layer_type="mlp",
                hidden_dim=hidden_dim,
                activation="gelu",
            )
        return create_layer_config(
            layer_type="gated_mlp",
            hidden_dim=hidden_dim,
            gating_type="swiglu",
            activation="silu",
            bias=False,
        )

    if model_type == "encoder":
        return create_layer_config(
            layer_type="mlp",
            hidden_dim=hidden_dim,
            activation="gelu",
            bias=True,
        )

    # decoder
    if efficiency_priority:
        return create_layer_config(
            layer_type="mlp",
            hidden_dim=hidden_dim,
            activation="gelu",
        )
    return create_layer_config(
        layer_type="gated_mlp",
        hidden_dim=hidden_dim,
        gating_type="swiglu",
        activation="silu",
        bias=False,
    )
