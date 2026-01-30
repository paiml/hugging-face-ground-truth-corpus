"""Mixture of Experts (MoE) training utilities.

This module provides functions for configuring and managing Mixture of
Experts models, including router configuration, load balancing, and
expert utilization tracking.

Examples:
    >>> from hf_gtc.training.moe import (
    ...     create_moe_config,
    ...     RouterType,
    ... )
    >>> config = create_moe_config()
    >>> config.router_config.num_experts
    8
    >>> config.router_config.router_type
    <RouterType.TOP_K: 'top_k'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RouterType(Enum):
    """Supported routing strategies for expert selection.

    Attributes:
        TOP_K: Select top-k experts per token based on router scores.
        EXPERT_CHOICE: Experts choose which tokens to process.
        SOFT: Soft routing using weighted combination of experts.
        HASH: Deterministic hash-based routing.

    Examples:
        >>> RouterType.TOP_K.value
        'top_k'
        >>> RouterType.EXPERT_CHOICE.value
        'expert_choice'
        >>> RouterType.SOFT.value
        'soft'
        >>> RouterType.HASH.value
        'hash'
    """

    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    SOFT = "soft"
    HASH = "hash"


class LoadBalancingLoss(Enum):
    """Load balancing loss functions for MoE training.

    Attributes:
        AUXILIARY: Auxiliary loss to encourage balanced routing.
        Z_LOSS: Router z-loss for numerical stability.
        SWITCH: Switch transformer style load balancing loss.
        NONE: No load balancing loss.

    Examples:
        >>> LoadBalancingLoss.AUXILIARY.value
        'auxiliary'
        >>> LoadBalancingLoss.Z_LOSS.value
        'z_loss'
        >>> LoadBalancingLoss.SWITCH.value
        'switch'
        >>> LoadBalancingLoss.NONE.value
        'none'
    """

    AUXILIARY = "auxiliary"
    Z_LOSS = "z_loss"
    SWITCH = "switch"
    NONE = "none"


class ExpertActivation(Enum):
    """Activation functions for expert feed-forward networks.

    Attributes:
        RELU: Rectified Linear Unit activation.
        GELU: Gaussian Error Linear Unit activation.
        SWIGLU: SwiGLU activation (GLU variant with Swish).
        GEGLU: GeGLU activation (GLU variant with GELU).

    Examples:
        >>> ExpertActivation.RELU.value
        'relu'
        >>> ExpertActivation.GELU.value
        'gelu'
        >>> ExpertActivation.SWIGLU.value
        'swiglu'
        >>> ExpertActivation.GEGLU.value
        'geglu'
    """

    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"


VALID_ROUTER_TYPES = frozenset(rt.value for rt in RouterType)
VALID_LOAD_BALANCING_LOSSES = frozenset(lb.value for lb in LoadBalancingLoss)
VALID_EXPERT_ACTIVATIONS = frozenset(ea.value for ea in ExpertActivation)


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Configuration for MoE router.

    Attributes:
        router_type: Type of routing strategy.
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        jitter_noise: Noise to add during training for exploration.
        temperature: Temperature for softmax routing scores.

    Examples:
        >>> config = RouterConfig(
        ...     router_type=RouterType.TOP_K,
        ...     num_experts=8,
        ...     top_k=2,
        ...     jitter_noise=0.1,
        ...     temperature=1.0,
        ... )
        >>> config.num_experts
        8
        >>> config.top_k
        2
        >>> config.router_type
        <RouterType.TOP_K: 'top_k'>
    """

    router_type: RouterType
    num_experts: int
    top_k: int
    jitter_noise: float
    temperature: float


@dataclass(frozen=True, slots=True)
class LoadBalanceConfig:
    """Configuration for load balancing in MoE.

    Attributes:
        loss_type: Type of load balancing loss.
        loss_weight: Weight for the load balancing loss term.
        capacity_factor: Capacity multiplier for expert buffers.
        drop_tokens: Whether to drop tokens exceeding capacity.

    Examples:
        >>> config = LoadBalanceConfig(
        ...     loss_type=LoadBalancingLoss.AUXILIARY,
        ...     loss_weight=0.01,
        ...     capacity_factor=1.25,
        ...     drop_tokens=False,
        ... )
        >>> config.loss_weight
        0.01
        >>> config.capacity_factor
        1.25
    """

    loss_type: LoadBalancingLoss
    loss_weight: float
    capacity_factor: float
    drop_tokens: bool


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """Configuration for individual experts in MoE.

    Attributes:
        hidden_dim: Hidden dimension of expert FFN.
        activation: Activation function for expert FFN.
        dropout: Dropout probability in expert layers.
        shared_expert: Whether to include a shared expert.

    Examples:
        >>> config = ExpertConfig(
        ...     hidden_dim=4096,
        ...     activation=ExpertActivation.GELU,
        ...     dropout=0.1,
        ...     shared_expert=False,
        ... )
        >>> config.hidden_dim
        4096
        >>> config.activation
        <ExpertActivation.GELU: 'gelu'>
    """

    hidden_dim: int
    activation: ExpertActivation
    dropout: float
    shared_expert: bool


@dataclass(frozen=True, slots=True)
class MoEConfig:
    """Main configuration for Mixture of Experts models.

    Attributes:
        router_config: Router configuration.
        expert_config: Expert network configuration.
        balance_config: Load balancing configuration.
        num_layers: Number of MoE layers in the model.

    Examples:
        >>> router = RouterConfig(
        ...     router_type=RouterType.TOP_K,
        ...     num_experts=8,
        ...     top_k=2,
        ...     jitter_noise=0.1,
        ...     temperature=1.0,
        ... )
        >>> expert = ExpertConfig(
        ...     hidden_dim=4096,
        ...     activation=ExpertActivation.GELU,
        ...     dropout=0.1,
        ...     shared_expert=False,
        ... )
        >>> balance = LoadBalanceConfig(
        ...     loss_type=LoadBalancingLoss.AUXILIARY,
        ...     loss_weight=0.01,
        ...     capacity_factor=1.25,
        ...     drop_tokens=False,
        ... )
        >>> config = MoEConfig(
        ...     router_config=router,
        ...     expert_config=expert,
        ...     balance_config=balance,
        ...     num_layers=12,
        ... )
        >>> config.num_layers
        12
    """

    router_config: RouterConfig
    expert_config: ExpertConfig
    balance_config: LoadBalanceConfig
    num_layers: int


@dataclass(frozen=True, slots=True)
class MoEStats:
    """Statistics from MoE training and inference.

    Attributes:
        router_entropy: Entropy of router probability distribution.
        load_balance_loss: Current load balancing loss value.
        expert_utilization: Fraction of experts used per batch.
        dropped_tokens: Number of tokens dropped due to capacity.

    Examples:
        >>> stats = MoEStats(
        ...     router_entropy=2.5,
        ...     load_balance_loss=0.05,
        ...     expert_utilization=0.75,
        ...     dropped_tokens=128,
        ... )
        >>> stats.router_entropy
        2.5
        >>> stats.expert_utilization
        0.75
    """

    router_entropy: float
    load_balance_loss: float
    expert_utilization: float
    dropped_tokens: int


def validate_router_config(config: RouterConfig) -> None:
    """Validate router configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = RouterConfig(
        ...     router_type=RouterType.TOP_K,
        ...     num_experts=8,
        ...     top_k=2,
        ...     jitter_noise=0.1,
        ...     temperature=1.0,
        ... )
        >>> validate_router_config(config)

        >>> bad_config = RouterConfig(
        ...     router_type=RouterType.TOP_K,
        ...     num_experts=0,
        ...     top_k=2,
        ...     jitter_noise=0.1,
        ...     temperature=1.0,
        ... )
        >>> validate_router_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_experts must be positive, got 0
    """
    if config.num_experts <= 0:
        msg = f"num_experts must be positive, got {config.num_experts}"
        raise ValueError(msg)
    if config.top_k <= 0:
        msg = f"top_k must be positive, got {config.top_k}"
        raise ValueError(msg)
    if config.top_k > config.num_experts:
        msg = (
            f"top_k ({config.top_k}) cannot exceed "
            f"num_experts ({config.num_experts})"
        )
        raise ValueError(msg)
    if config.jitter_noise < 0:
        msg = f"jitter_noise must be non-negative, got {config.jitter_noise}"
        raise ValueError(msg)
    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)


def validate_load_balance_config(config: LoadBalanceConfig) -> None:
    """Validate load balance configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = LoadBalanceConfig(
        ...     loss_type=LoadBalancingLoss.AUXILIARY,
        ...     loss_weight=0.01,
        ...     capacity_factor=1.25,
        ...     drop_tokens=False,
        ... )
        >>> validate_load_balance_config(config)

        >>> bad_config = LoadBalanceConfig(
        ...     loss_type=LoadBalancingLoss.AUXILIARY,
        ...     loss_weight=-0.1,
        ...     capacity_factor=1.25,
        ...     drop_tokens=False,
        ... )
        >>> validate_load_balance_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: loss_weight must be non-negative, got -0.1
    """
    if config.loss_weight < 0:
        msg = f"loss_weight must be non-negative, got {config.loss_weight}"
        raise ValueError(msg)
    if config.capacity_factor <= 0:
        msg = f"capacity_factor must be positive, got {config.capacity_factor}"
        raise ValueError(msg)


def validate_expert_config(config: ExpertConfig) -> None:
    """Validate expert configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = ExpertConfig(
        ...     hidden_dim=4096,
        ...     activation=ExpertActivation.GELU,
        ...     dropout=0.1,
        ...     shared_expert=False,
        ... )
        >>> validate_expert_config(config)

        >>> bad_config = ExpertConfig(
        ...     hidden_dim=0,
        ...     activation=ExpertActivation.GELU,
        ...     dropout=0.1,
        ...     shared_expert=False,
        ... )
        >>> validate_expert_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: hidden_dim must be positive, got 0
    """
    if config.hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {config.hidden_dim}"
        raise ValueError(msg)
    if not 0 <= config.dropout <= 1:
        msg = f"dropout must be between 0 and 1, got {config.dropout}"
        raise ValueError(msg)


def validate_moe_config(config: MoEConfig) -> None:
    """Validate MoE configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> router = RouterConfig(
        ...     router_type=RouterType.TOP_K,
        ...     num_experts=8,
        ...     top_k=2,
        ...     jitter_noise=0.1,
        ...     temperature=1.0,
        ... )
        >>> expert = ExpertConfig(
        ...     hidden_dim=4096,
        ...     activation=ExpertActivation.GELU,
        ...     dropout=0.1,
        ...     shared_expert=False,
        ... )
        >>> balance = LoadBalanceConfig(
        ...     loss_type=LoadBalancingLoss.AUXILIARY,
        ...     loss_weight=0.01,
        ...     capacity_factor=1.25,
        ...     drop_tokens=False,
        ... )
        >>> config = MoEConfig(
        ...     router_config=router,
        ...     expert_config=expert,
        ...     balance_config=balance,
        ...     num_layers=12,
        ... )
        >>> validate_moe_config(config)

        >>> bad_config = MoEConfig(
        ...     router_config=router,
        ...     expert_config=expert,
        ...     balance_config=balance,
        ...     num_layers=0,
        ... )
        >>> validate_moe_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_layers must be positive, got 0
    """
    validate_router_config(config.router_config)
    validate_expert_config(config.expert_config)
    validate_load_balance_config(config.balance_config)
    if config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)


def create_router_config(
    router_type: str | RouterType = RouterType.TOP_K,
    num_experts: int = 8,
    top_k: int = 2,
    jitter_noise: float = 0.1,
    temperature: float = 1.0,
) -> RouterConfig:
    """Create a router configuration with validation.

    Args:
        router_type: Type of routing strategy.
        num_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        jitter_noise: Noise to add during training.
        temperature: Temperature for softmax routing.

    Returns:
        Validated RouterConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_router_config()
        >>> config.num_experts
        8
        >>> config.top_k
        2

        >>> config = create_router_config(router_type="expert_choice", num_experts=16)
        >>> config.router_type
        <RouterType.EXPERT_CHOICE: 'expert_choice'>
        >>> config.num_experts
        16

        >>> create_router_config(num_experts=0)
        Traceback (most recent call last):
            ...
        ValueError: num_experts must be positive, got 0
    """
    if isinstance(router_type, str):
        router_type = get_router_type(router_type)

    config = RouterConfig(
        router_type=router_type,
        num_experts=num_experts,
        top_k=top_k,
        jitter_noise=jitter_noise,
        temperature=temperature,
    )
    validate_router_config(config)
    return config


def create_load_balance_config(
    loss_type: str | LoadBalancingLoss = LoadBalancingLoss.AUXILIARY,
    loss_weight: float = 0.01,
    capacity_factor: float = 1.25,
    drop_tokens: bool = False,
) -> LoadBalanceConfig:
    """Create a load balance configuration with validation.

    Args:
        loss_type: Type of load balancing loss.
        loss_weight: Weight for the loss term.
        capacity_factor: Capacity multiplier for buffers.
        drop_tokens: Whether to drop overflow tokens.

    Returns:
        Validated LoadBalanceConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_load_balance_config()
        >>> config.loss_weight
        0.01
        >>> config.capacity_factor
        1.25

        >>> config = create_load_balance_config(loss_type="z_loss", loss_weight=0.001)
        >>> config.loss_type
        <LoadBalancingLoss.Z_LOSS: 'z_loss'>

        >>> create_load_balance_config(loss_weight=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: loss_weight must be non-negative, got -0.1
    """
    if isinstance(loss_type, str):
        loss_type = get_load_balancing_loss(loss_type)

    config = LoadBalanceConfig(
        loss_type=loss_type,
        loss_weight=loss_weight,
        capacity_factor=capacity_factor,
        drop_tokens=drop_tokens,
    )
    validate_load_balance_config(config)
    return config


def create_expert_config(
    hidden_dim: int = 4096,
    activation: str | ExpertActivation = ExpertActivation.GELU,
    dropout: float = 0.1,
    shared_expert: bool = False,
) -> ExpertConfig:
    """Create an expert configuration with validation.

    Args:
        hidden_dim: Hidden dimension of expert FFN.
        activation: Activation function.
        dropout: Dropout probability.
        shared_expert: Whether to include shared expert.

    Returns:
        Validated ExpertConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_expert_config()
        >>> config.hidden_dim
        4096
        >>> config.activation
        <ExpertActivation.GELU: 'gelu'>

        >>> config = create_expert_config(activation="swiglu", hidden_dim=8192)
        >>> config.activation
        <ExpertActivation.SWIGLU: 'swiglu'>

        >>> create_expert_config(hidden_dim=0)
        Traceback (most recent call last):
            ...
        ValueError: hidden_dim must be positive, got 0
    """
    if isinstance(activation, str):
        activation = get_expert_activation(activation)

    config = ExpertConfig(
        hidden_dim=hidden_dim,
        activation=activation,
        dropout=dropout,
        shared_expert=shared_expert,
    )
    validate_expert_config(config)
    return config


def create_moe_config(
    router_config: RouterConfig | None = None,
    expert_config: ExpertConfig | None = None,
    balance_config: LoadBalanceConfig | None = None,
    num_layers: int = 12,
) -> MoEConfig:
    """Create a MoE configuration with validation.

    Args:
        router_config: Router configuration.
        expert_config: Expert configuration.
        balance_config: Load balance configuration.
        num_layers: Number of MoE layers.

    Returns:
        Validated MoEConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_moe_config()
        >>> config.router_config.num_experts
        8
        >>> config.num_layers
        12

        >>> custom_router = create_router_config(num_experts=16, top_k=4)
        >>> config = create_moe_config(router_config=custom_router, num_layers=24)
        >>> config.router_config.num_experts
        16

        >>> create_moe_config(num_layers=0)
        Traceback (most recent call last):
            ...
        ValueError: num_layers must be positive, got 0
    """
    if router_config is None:
        router_config = create_router_config()
    if expert_config is None:
        expert_config = create_expert_config()
    if balance_config is None:
        balance_config = create_load_balance_config()

    config = MoEConfig(
        router_config=router_config,
        expert_config=expert_config,
        balance_config=balance_config,
        num_layers=num_layers,
    )
    validate_moe_config(config)
    return config


def create_moe_stats(
    router_entropy: float = 0.0,
    load_balance_loss: float = 0.0,
    expert_utilization: float = 0.0,
    dropped_tokens: int = 0,
) -> MoEStats:
    """Create MoE statistics.

    Args:
        router_entropy: Entropy of router distribution.
        load_balance_loss: Load balancing loss value.
        expert_utilization: Fraction of experts used.
        dropped_tokens: Number of dropped tokens.

    Returns:
        MoEStats instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_moe_stats()
        >>> stats.router_entropy
        0.0
        >>> stats.dropped_tokens
        0

        >>> stats = create_moe_stats(
        ...     router_entropy=2.5,
        ...     expert_utilization=0.75,
        ... )
        >>> stats.router_entropy
        2.5

        >>> create_moe_stats(dropped_tokens=-1)
        Traceback (most recent call last):
            ...
        ValueError: dropped_tokens must be non-negative, got -1
    """
    if router_entropy < 0:
        msg = f"router_entropy must be non-negative, got {router_entropy}"
        raise ValueError(msg)
    if load_balance_loss < 0:
        msg = f"load_balance_loss must be non-negative, got {load_balance_loss}"
        raise ValueError(msg)
    if not 0 <= expert_utilization <= 1:
        msg = f"expert_utilization must be between 0 and 1, got {expert_utilization}"
        raise ValueError(msg)
    if dropped_tokens < 0:
        msg = f"dropped_tokens must be non-negative, got {dropped_tokens}"
        raise ValueError(msg)

    return MoEStats(
        router_entropy=router_entropy,
        load_balance_loss=load_balance_loss,
        expert_utilization=expert_utilization,
        dropped_tokens=dropped_tokens,
    )


def list_router_types() -> list[str]:
    """List all available router types.

    Returns:
        Sorted list of router type names.

    Examples:
        >>> types = list_router_types()
        >>> "top_k" in types
        True
        >>> "expert_choice" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ROUTER_TYPES)


def list_load_balancing_losses() -> list[str]:
    """List all available load balancing loss types.

    Returns:
        Sorted list of load balancing loss names.

    Examples:
        >>> losses = list_load_balancing_losses()
        >>> "auxiliary" in losses
        True
        >>> "z_loss" in losses
        True
        >>> losses == sorted(losses)
        True
    """
    return sorted(VALID_LOAD_BALANCING_LOSSES)


def list_expert_activations() -> list[str]:
    """List all available expert activation functions.

    Returns:
        Sorted list of activation names.

    Examples:
        >>> activations = list_expert_activations()
        >>> "gelu" in activations
        True
        >>> "swiglu" in activations
        True
        >>> activations == sorted(activations)
        True
    """
    return sorted(VALID_EXPERT_ACTIVATIONS)


def get_router_type(name: str) -> RouterType:
    """Get router type enum from string name.

    Args:
        name: Name of the router type.

    Returns:
        Corresponding RouterType enum.

    Raises:
        ValueError: If router type name is invalid.

    Examples:
        >>> get_router_type("top_k")
        <RouterType.TOP_K: 'top_k'>
        >>> get_router_type("expert_choice")
        <RouterType.EXPERT_CHOICE: 'expert_choice'>

        >>> get_router_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: router_type must be one of ...
    """
    if name not in VALID_ROUTER_TYPES:
        msg = f"router_type must be one of {VALID_ROUTER_TYPES}, got '{name}'"
        raise ValueError(msg)
    return RouterType(name)


def get_load_balancing_loss(name: str) -> LoadBalancingLoss:
    """Get load balancing loss enum from string name.

    Args:
        name: Name of the load balancing loss.

    Returns:
        Corresponding LoadBalancingLoss enum.

    Raises:
        ValueError: If loss name is invalid.

    Examples:
        >>> get_load_balancing_loss("auxiliary")
        <LoadBalancingLoss.AUXILIARY: 'auxiliary'>
        >>> get_load_balancing_loss("z_loss")
        <LoadBalancingLoss.Z_LOSS: 'z_loss'>

        >>> get_load_balancing_loss("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: load_balancing_loss must be one of ...
    """
    if name not in VALID_LOAD_BALANCING_LOSSES:
        msg = (
            f"load_balancing_loss must be one of {VALID_LOAD_BALANCING_LOSSES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return LoadBalancingLoss(name)


def get_expert_activation(name: str) -> ExpertActivation:
    """Get expert activation enum from string name.

    Args:
        name: Name of the activation function.

    Returns:
        Corresponding ExpertActivation enum.

    Raises:
        ValueError: If activation name is invalid.

    Examples:
        >>> get_expert_activation("gelu")
        <ExpertActivation.GELU: 'gelu'>
        >>> get_expert_activation("swiglu")
        <ExpertActivation.SWIGLU: 'swiglu'>

        >>> get_expert_activation("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: expert_activation must be one of ...
    """
    if name not in VALID_EXPERT_ACTIVATIONS:
        msg = (
            f"expert_activation must be one of {VALID_EXPERT_ACTIVATIONS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return ExpertActivation(name)


def calculate_capacity_factor(
    num_tokens: int,
    num_experts: int,
    top_k: int = 1,
    buffer_ratio: float = 0.25,
) -> float:
    """Calculate optimal capacity factor for MoE routing.

    The capacity factor determines the buffer size for each expert.
    A factor of 1.0 means each expert can process exactly its fair share
    of tokens. Higher factors allow for imbalanced loads.

    Args:
        num_tokens: Number of tokens in the batch.
        num_experts: Number of experts.
        top_k: Number of experts per token.
        buffer_ratio: Additional buffer ratio for overflow.

    Returns:
        Recommended capacity factor.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> factor = calculate_capacity_factor(1024, 8, top_k=2)
        >>> 1.0 < factor < 2.0
        True

        >>> factor = calculate_capacity_factor(2048, 16)
        >>> factor > 1.0
        True

        >>> calculate_capacity_factor(0, 8)
        Traceback (most recent call last):
            ...
        ValueError: num_tokens must be positive, got 0
    """
    if num_tokens <= 0:
        msg = f"num_tokens must be positive, got {num_tokens}"
        raise ValueError(msg)
    if num_experts <= 0:
        msg = f"num_experts must be positive, got {num_experts}"
        raise ValueError(msg)
    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)
    if top_k > num_experts:
        msg = f"top_k ({top_k}) cannot exceed num_experts ({num_experts})"
        raise ValueError(msg)
    if buffer_ratio < 0:
        msg = f"buffer_ratio must be non-negative, got {buffer_ratio}"
        raise ValueError(msg)

    # Base capacity: tokens per expert if evenly distributed
    # Multiply by top_k since each token goes to top_k experts
    base_capacity = (num_tokens * top_k) / num_experts

    # Add buffer for imbalanced routing
    capacity_factor = 1.0 + buffer_ratio

    # Adjust for very small batches to prevent numerical issues
    if base_capacity < 1.0:
        capacity_factor = max(capacity_factor, 2.0)

    return capacity_factor


def estimate_expert_utilization(
    router_probs: tuple[float, ...],
    threshold: float = 0.01,
) -> float:
    """Estimate expert utilization from router probabilities.

    Utilization measures what fraction of experts receive meaningful
    routing weights, indicating diversity in expert selection.

    Args:
        router_probs: Tuple of routing probabilities for each expert.
        threshold: Minimum probability to consider an expert "used".

    Returns:
        Fraction of experts with probability above threshold.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> probs = (0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0)
        >>> estimate_expert_utilization(probs)
        0.5

        >>> probs = (0.125,) * 8
        >>> estimate_expert_utilization(probs)
        1.0

        >>> estimate_expert_utilization(())
        Traceback (most recent call last):
            ...
        ValueError: router_probs cannot be empty
    """
    if not router_probs:
        msg = "router_probs cannot be empty"
        raise ValueError(msg)
    if threshold < 0:
        msg = f"threshold must be non-negative, got {threshold}"
        raise ValueError(msg)

    used_experts = sum(1 for p in router_probs if p > threshold)
    return used_experts / len(router_probs)


def calculate_router_entropy(router_probs: tuple[float, ...]) -> float:
    """Calculate entropy of router probability distribution.

    Higher entropy indicates more uniform routing across experts,
    which generally promotes better load balancing.

    Args:
        router_probs: Tuple of routing probabilities for each expert.

    Returns:
        Shannon entropy of the distribution (in nats).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> probs = (0.5, 0.5)
        >>> entropy = calculate_router_entropy(probs)
        >>> abs(entropy - 0.693) < 0.01  # ln(2) approx 0.693
        True

        >>> probs = (0.25, 0.25, 0.25, 0.25)
        >>> entropy = calculate_router_entropy(probs)
        >>> abs(entropy - 1.386) < 0.01  # ln(4) approx 1.386
        True

        >>> calculate_router_entropy(())
        Traceback (most recent call last):
            ...
        ValueError: router_probs cannot be empty
    """
    if not router_probs:
        msg = "router_probs cannot be empty"
        raise ValueError(msg)

    # Validate probabilities sum close to 1
    prob_sum = sum(router_probs)
    if not 0.99 <= prob_sum <= 1.01:
        msg = f"router_probs must sum to 1, got {prob_sum}"
        raise ValueError(msg)

    # Calculate Shannon entropy: H = -sum(p * log(p))
    entropy = 0.0
    for p in router_probs:
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


def calculate_load_balance_loss(
    router_probs: tuple[float, ...],
    expert_counts: tuple[int, ...],
    loss_type: LoadBalancingLoss = LoadBalancingLoss.AUXILIARY,
) -> float:
    """Calculate load balancing loss for MoE training.

    Args:
        router_probs: Average routing probability per expert.
        expert_counts: Token count assigned to each expert.
        loss_type: Type of load balancing loss to compute.

    Returns:
        Load balancing loss value.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> probs = (0.25, 0.25, 0.25, 0.25)
        >>> counts = (100, 100, 100, 100)
        >>> loss = calculate_load_balance_loss(probs, counts)
        >>> loss == 1.0  # Balanced case: 4 * (0.25 * 0.25 * 4) = 1.0
        True

        >>> probs = (0.5, 0.5, 0.0, 0.0)
        >>> counts = (200, 200, 0, 0)
        >>> loss = calculate_load_balance_loss(probs, counts)
        >>> loss > 1.0  # Imbalanced case has higher loss
        True

        >>> calculate_load_balance_loss((), ())
        Traceback (most recent call last):
            ...
        ValueError: router_probs cannot be empty
    """
    if not router_probs:
        msg = "router_probs cannot be empty"
        raise ValueError(msg)
    if not expert_counts:
        msg = "expert_counts cannot be empty"
        raise ValueError(msg)
    if len(router_probs) != len(expert_counts):
        msg = "router_probs and expert_counts must have same length"
        raise ValueError(msg)

    if loss_type == LoadBalancingLoss.NONE:
        return 0.0

    num_experts = len(router_probs)
    total_tokens = sum(expert_counts)

    if total_tokens == 0:
        return 0.0

    # Normalize counts to fractions
    count_fracs = tuple(c / total_tokens for c in expert_counts)

    if loss_type == LoadBalancingLoss.AUXILIARY:
        # Auxiliary loss: num_experts * sum(prob_i * count_frac_i)
        # This encourages balanced probability * usage
        loss = num_experts * sum(
            p * f for p, f in zip(router_probs, count_fracs, strict=True)
        )
        return loss

    elif loss_type == LoadBalancingLoss.Z_LOSS:
        # Z-loss: penalize large router logits for numerical stability
        # Simplified version using prob entropy proxy
        mean_prob = sum(router_probs) / num_experts
        variance = sum((p - mean_prob) ** 2 for p in router_probs) / num_experts
        return variance

    elif loss_type == LoadBalancingLoss.SWITCH:
        # Switch transformer loss: f_i * P_i summed over experts
        # Where f_i is fraction of tokens and P_i is routing probability
        loss = num_experts * sum(
            p * f for p, f in zip(router_probs, count_fracs, strict=True)
        )
        return loss

    return 0.0


def format_moe_stats(stats: MoEStats) -> str:
    """Format MoE statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_moe_stats(
        ...     router_entropy=2.5,
        ...     load_balance_loss=0.05,
        ...     expert_utilization=0.75,
        ...     dropped_tokens=128,
        ... )
        >>> formatted = format_moe_stats(stats)
        >>> "Router Entropy: 2.50" in formatted
        True
        >>> "Expert Utilization: 75.0%" in formatted
        True
        >>> "Dropped Tokens: 128" in formatted
        True
    """
    return (
        f"MoE Stats:\n"
        f"  Router Entropy: {stats.router_entropy:.2f}\n"
        f"  Load Balance Loss: {stats.load_balance_loss:.4f}\n"
        f"  Expert Utilization: {stats.expert_utilization * 100:.1f}%\n"
        f"  Dropped Tokens: {stats.dropped_tokens}"
    )


def get_recommended_moe_config(model_params: int) -> MoEConfig:
    """Get recommended MoE configuration based on model size.

    Provides sensible defaults for different model scales.

    Args:
        model_params: Approximate number of model parameters.

    Returns:
        Recommended MoEConfig for the model size.

    Raises:
        ValueError: If model_params is not positive.

    Examples:
        >>> config = get_recommended_moe_config(7_000_000_000)
        >>> config.router_config.num_experts
        8
        >>> config.router_config.top_k
        2

        >>> config = get_recommended_moe_config(70_000_000_000)
        >>> config.router_config.num_experts >= 8
        True

        >>> get_recommended_moe_config(0)
        Traceback (most recent call last):
            ...
        ValueError: model_params must be positive, got 0
    """
    if model_params <= 0:
        msg = f"model_params must be positive, got {model_params}"
        raise ValueError(msg)

    # Scale experts with model size
    if model_params < 1_000_000_000:  # < 1B
        num_experts = 4
        top_k = 1
        hidden_dim = 2048
        num_layers = 6
    elif model_params < 10_000_000_000:  # 1B - 10B
        num_experts = 8
        top_k = 2
        hidden_dim = 4096
        num_layers = 12
    elif model_params < 100_000_000_000:  # 10B - 100B
        num_experts = 16
        top_k = 2
        hidden_dim = 8192
        num_layers = 24
    else:  # 100B+
        num_experts = 64
        top_k = 4
        hidden_dim = 16384
        num_layers = 32

    router_config = create_router_config(
        router_type=RouterType.TOP_K,
        num_experts=num_experts,
        top_k=top_k,
        jitter_noise=0.1,
        temperature=1.0,
    )

    expert_config = create_expert_config(
        hidden_dim=hidden_dim,
        activation=ExpertActivation.SWIGLU,
        dropout=0.1,
        shared_expert=model_params >= 10_000_000_000,
    )

    balance_config = create_load_balance_config(
        loss_type=LoadBalancingLoss.AUXILIARY,
        loss_weight=0.01,
        capacity_factor=1.25,
        drop_tokens=False,
    )

    return create_moe_config(
        router_config=router_config,
        expert_config=expert_config,
        balance_config=balance_config,
        num_layers=num_layers,
    )
