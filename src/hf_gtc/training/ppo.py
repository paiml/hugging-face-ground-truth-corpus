"""Proximal Policy Optimization (PPO) training utilities.

This module provides functions for configuring and running PPO-based
reinforcement learning from human feedback (RLHF) with the HuggingFace
TRL library.

Examples:
    >>> from hf_gtc.training.ppo import create_ppo_config, PPOVariant
    >>> config = create_ppo_config(variant="ppo_clip")
    >>> config.variant
    <PPOVariant.PPO_CLIP: 'ppo_clip'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class PPOVariant(Enum):
    """PPO algorithm variants.

    Attributes:
        STANDARD: Standard PPO with both clipping and KL penalty.
        REINFORCE: Basic REINFORCE algorithm (no clipping).
        A2C: Advantage Actor-Critic.
        PPO_CLIP: PPO with clipped objective only.
        PPO_PENALTY: PPO with adaptive KL penalty only.

    Examples:
        >>> PPOVariant.PPO_CLIP.value
        'ppo_clip'
        >>> PPOVariant.STANDARD.value
        'standard'
    """

    STANDARD = "standard"
    REINFORCE = "reinforce"
    A2C = "a2c"
    PPO_CLIP = "ppo_clip"
    PPO_PENALTY = "ppo_penalty"


VALID_PPO_VARIANTS = frozenset(v.value for v in PPOVariant)


class RewardModelType(Enum):
    """Types of reward models for RLHF.

    Attributes:
        LEARNED: Learned reward model from human preferences.
        RULE_BASED: Rule-based reward (e.g., length, format).
        HYBRID: Combination of learned and rule-based.
        CONSTITUTIONAL: Constitutional AI-style self-critique.

    Examples:
        >>> RewardModelType.LEARNED.value
        'learned'
        >>> RewardModelType.CONSTITUTIONAL.value
        'constitutional'
    """

    LEARNED = "learned"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    CONSTITUTIONAL = "constitutional"


VALID_REWARD_MODEL_TYPES = frozenset(t.value for t in RewardModelType)


class ValueHeadType(Enum):
    """Types of value heads for PPO.

    Attributes:
        LINEAR: Single linear layer.
        MLP: Multi-layer perceptron.
        SHARED: Shared layers with policy head.

    Examples:
        >>> ValueHeadType.LINEAR.value
        'linear'
        >>> ValueHeadType.MLP.value
        'mlp'
    """

    LINEAR = "linear"
    MLP = "mlp"
    SHARED = "shared"


VALID_VALUE_HEAD_TYPES = frozenset(t.value for t in ValueHeadType)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Configuration for PPO algorithm.

    Attributes:
        variant: PPO algorithm variant.
        clip_range: Clipping parameter for PPO objective.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        target_kl: Target KL divergence for early stopping.

    Examples:
        >>> config = PPOConfig(
        ...     variant=PPOVariant.PPO_CLIP,
        ...     clip_range=0.2,
        ...     vf_coef=0.5,
        ...     ent_coef=0.01,
        ...     max_grad_norm=0.5,
        ...     target_kl=0.02,
        ... )
        >>> config.clip_range
        0.2
    """

    variant: PPOVariant
    clip_range: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float
    target_kl: float


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Configuration for reward model.

    Attributes:
        model_type: Type of reward model.
        model_id: HuggingFace model ID for learned reward.
        normalize_rewards: Whether to normalize rewards.
        clip_rewards: Maximum absolute reward value (None for no clipping).

    Examples:
        >>> config = RewardConfig(
        ...     model_type=RewardModelType.LEARNED,
        ...     model_id="OpenAssistant/reward-model-deberta-v3-large-v2",
        ...     normalize_rewards=True,
        ...     clip_rewards=10.0,
        ... )
        >>> config.normalize_rewards
        True
    """

    model_type: RewardModelType
    model_id: str | None
    normalize_rewards: bool
    clip_rewards: float | None


@dataclass(frozen=True, slots=True)
class ValueConfig:
    """Configuration for value head.

    Attributes:
        head_type: Type of value head architecture.
        hidden_size: Hidden layer size for MLP head.
        num_layers: Number of layers for MLP head.
        dropout: Dropout probability.

    Examples:
        >>> config = ValueConfig(
        ...     head_type=ValueHeadType.MLP,
        ...     hidden_size=256,
        ...     num_layers=2,
        ...     dropout=0.1,
        ... )
        >>> config.hidden_size
        256
    """

    head_type: ValueHeadType
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    """Configuration for PPO training loop.

    Attributes:
        learning_rate: Learning rate for optimizer.
        batch_size: Number of experiences per batch.
        mini_batch_size: Mini-batch size for PPO updates.
        ppo_epochs: Number of PPO epochs per batch.
        rollout_steps: Number of rollout steps to collect.

    Examples:
        >>> config = PPOTrainingConfig(
        ...     learning_rate=1e-5,
        ...     batch_size=128,
        ...     mini_batch_size=32,
        ...     ppo_epochs=4,
        ...     rollout_steps=128,
        ... )
        >>> config.ppo_epochs
        4
    """

    learning_rate: float
    batch_size: int
    mini_batch_size: int
    ppo_epochs: int
    rollout_steps: int


@dataclass(frozen=True, slots=True)
class PPOStats:
    """Statistics from PPO training step.

    Attributes:
        policy_loss: Policy gradient loss.
        value_loss: Value function loss.
        entropy: Policy entropy.
        kl_divergence: KL divergence from reference policy.
        clip_fraction: Fraction of samples that were clipped.
        explained_variance: Explained variance of value predictions.

    Examples:
        >>> stats = PPOStats(
        ...     policy_loss=0.1,
        ...     value_loss=0.5,
        ...     entropy=0.8,
        ...     kl_divergence=0.01,
        ...     clip_fraction=0.15,
        ...     explained_variance=0.85,
        ... )
        >>> stats.explained_variance
        0.85
    """

    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    clip_fraction: float
    explained_variance: float


def validate_ppo_config(config: PPOConfig) -> None:
    """Validate PPO configuration parameters.

    Args:
        config: PPO configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = PPOConfig(
        ...     variant=PPOVariant.PPO_CLIP,
        ...     clip_range=0.2,
        ...     vf_coef=0.5,
        ...     ent_coef=0.01,
        ...     max_grad_norm=0.5,
        ...     target_kl=0.02,
        ... )
        >>> validate_ppo_config(config)  # No error

        >>> bad_config = PPOConfig(
        ...     variant=PPOVariant.PPO_CLIP,
        ...     clip_range=-0.1,
        ...     vf_coef=0.5,
        ...     ent_coef=0.01,
        ...     max_grad_norm=0.5,
        ...     target_kl=0.02,
        ... )
        >>> validate_ppo_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: clip_range must be positive
    """
    if config.clip_range <= 0:
        msg = f"clip_range must be positive, got {config.clip_range}"
        raise ValueError(msg)

    if config.vf_coef < 0:
        msg = f"vf_coef must be non-negative, got {config.vf_coef}"
        raise ValueError(msg)

    if config.ent_coef < 0:
        msg = f"ent_coef must be non-negative, got {config.ent_coef}"
        raise ValueError(msg)

    if config.max_grad_norm <= 0:
        msg = f"max_grad_norm must be positive, got {config.max_grad_norm}"
        raise ValueError(msg)

    if config.target_kl <= 0:
        msg = f"target_kl must be positive, got {config.target_kl}"
        raise ValueError(msg)


def validate_reward_config(config: RewardConfig) -> None:
    """Validate reward configuration parameters.

    Args:
        config: Reward configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = RewardConfig(
        ...     model_type=RewardModelType.LEARNED,
        ...     model_id="OpenAssistant/reward-model-deberta-v3-large-v2",
        ...     normalize_rewards=True,
        ...     clip_rewards=10.0,
        ... )
        >>> validate_reward_config(config)  # No error

        >>> bad_config = RewardConfig(
        ...     model_type=RewardModelType.LEARNED,
        ...     model_id=None,
        ...     normalize_rewards=True,
        ...     clip_rewards=10.0,
        ... )
        >>> validate_reward_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id is required for learned reward models
    """
    if config.model_type == RewardModelType.LEARNED and not config.model_id:
        msg = "model_id is required for learned reward models"
        raise ValueError(msg)

    if config.clip_rewards is not None and config.clip_rewards <= 0:
        msg = f"clip_rewards must be positive if set, got {config.clip_rewards}"
        raise ValueError(msg)


def validate_value_config(config: ValueConfig) -> None:
    """Validate value head configuration parameters.

    Args:
        config: Value configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = ValueConfig(
        ...     head_type=ValueHeadType.MLP,
        ...     hidden_size=256,
        ...     num_layers=2,
        ...     dropout=0.1,
        ... )
        >>> validate_value_config(config)  # No error

        >>> bad_config = ValueConfig(
        ...     head_type=ValueHeadType.MLP,
        ...     hidden_size=0,
        ...     num_layers=2,
        ...     dropout=0.1,
        ... )
        >>> validate_value_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hidden_size must be positive
    """
    if config.hidden_size <= 0:
        msg = f"hidden_size must be positive, got {config.hidden_size}"
        raise ValueError(msg)

    if config.num_layers <= 0:
        msg = f"num_layers must be positive, got {config.num_layers}"
        raise ValueError(msg)

    if not 0.0 <= config.dropout < 1.0:
        msg = f"dropout must be in [0.0, 1.0), got {config.dropout}"
        raise ValueError(msg)


def validate_ppo_training_config(config: PPOTrainingConfig) -> None:
    """Validate PPO training configuration parameters.

    Args:
        config: Training configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = PPOTrainingConfig(
        ...     learning_rate=1e-5,
        ...     batch_size=128,
        ...     mini_batch_size=32,
        ...     ppo_epochs=4,
        ...     rollout_steps=128,
        ... )
        >>> validate_ppo_training_config(config)  # No error

        >>> bad_config = PPOTrainingConfig(
        ...     learning_rate=1e-5,
        ...     batch_size=32,
        ...     mini_batch_size=64,
        ...     ppo_epochs=4,
        ...     rollout_steps=128,
        ... )
        >>> validate_ppo_training_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: mini_batch_size cannot exceed batch_size
    """
    if config.learning_rate <= 0:
        msg = f"learning_rate must be positive, got {config.learning_rate}"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.mini_batch_size <= 0:
        msg = f"mini_batch_size must be positive, got {config.mini_batch_size}"
        raise ValueError(msg)

    if config.mini_batch_size > config.batch_size:
        msg = (
            f"mini_batch_size ({config.mini_batch_size}) cannot exceed "
            f"batch_size ({config.batch_size})"
        )
        raise ValueError(msg)

    if config.ppo_epochs <= 0:
        msg = f"ppo_epochs must be positive, got {config.ppo_epochs}"
        raise ValueError(msg)

    if config.rollout_steps <= 0:
        msg = f"rollout_steps must be positive, got {config.rollout_steps}"
        raise ValueError(msg)


def create_ppo_config(
    variant: str = "ppo_clip",
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.02,
) -> PPOConfig:
    """Create a PPO configuration.

    Args:
        variant: PPO algorithm variant. Defaults to "ppo_clip".
        clip_range: Clipping parameter. Defaults to 0.2.
        vf_coef: Value function coefficient. Defaults to 0.5.
        ent_coef: Entropy coefficient. Defaults to 0.01.
        max_grad_norm: Maximum gradient norm. Defaults to 0.5.
        target_kl: Target KL divergence. Defaults to 0.02.

    Returns:
        PPOConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_ppo_config(variant="ppo_clip")
        >>> config.variant
        <PPOVariant.PPO_CLIP: 'ppo_clip'>

        >>> config = create_ppo_config(clip_range=0.1, ent_coef=0.02)
        >>> config.clip_range
        0.1

        >>> create_ppo_config(variant="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: variant must be one of
    """
    if variant not in VALID_PPO_VARIANTS:
        msg = f"variant must be one of {VALID_PPO_VARIANTS}, got '{variant}'"
        raise ValueError(msg)

    config = PPOConfig(
        variant=PPOVariant(variant),
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
    )
    validate_ppo_config(config)
    return config


def create_reward_config(
    model_type: str = "learned",
    model_id: str | None = None,
    normalize_rewards: bool = True,
    clip_rewards: float | None = 10.0,
) -> RewardConfig:
    """Create a reward configuration.

    Args:
        model_type: Type of reward model. Defaults to "learned".
        model_id: HuggingFace model ID. Defaults to None.
        normalize_rewards: Whether to normalize. Defaults to True.
        clip_rewards: Maximum absolute reward. Defaults to 10.0.

    Returns:
        RewardConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_reward_config(
        ...     model_type="learned",
        ...     model_id="OpenAssistant/reward-model-deberta-v3-large-v2",
        ... )
        >>> config.model_type
        <RewardModelType.LEARNED: 'learned'>

        >>> config = create_reward_config(model_type="rule_based")
        >>> config.model_type
        <RewardModelType.RULE_BASED: 'rule_based'>

        >>> create_reward_config(model_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of
    """
    if model_type not in VALID_REWARD_MODEL_TYPES:
        msg = (
            f"model_type must be one of {VALID_REWARD_MODEL_TYPES}, got '{model_type}'"
        )
        raise ValueError(msg)

    config = RewardConfig(
        model_type=RewardModelType(model_type),
        model_id=model_id,
        normalize_rewards=normalize_rewards,
        clip_rewards=clip_rewards,
    )
    validate_reward_config(config)
    return config


def create_value_config(
    head_type: str = "mlp",
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> ValueConfig:
    """Create a value head configuration.

    Args:
        head_type: Type of value head. Defaults to "mlp".
        hidden_size: Hidden layer size. Defaults to 256.
        num_layers: Number of layers. Defaults to 2.
        dropout: Dropout probability. Defaults to 0.1.

    Returns:
        ValueConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_value_config(head_type="mlp")
        >>> config.head_type
        <ValueHeadType.MLP: 'mlp'>

        >>> config = create_value_config(hidden_size=512, num_layers=3)
        >>> config.hidden_size
        512

        >>> create_value_config(head_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: head_type must be one of
    """
    if head_type not in VALID_VALUE_HEAD_TYPES:
        msg = f"head_type must be one of {VALID_VALUE_HEAD_TYPES}, got '{head_type}'"
        raise ValueError(msg)

    config = ValueConfig(
        head_type=ValueHeadType(head_type),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    validate_value_config(config)
    return config


def create_ppo_training_config(
    learning_rate: float = 1e-5,
    batch_size: int = 128,
    mini_batch_size: int = 32,
    ppo_epochs: int = 4,
    rollout_steps: int = 128,
) -> PPOTrainingConfig:
    """Create a PPO training configuration.

    Args:
        learning_rate: Learning rate. Defaults to 1e-5.
        batch_size: Batch size. Defaults to 128.
        mini_batch_size: Mini-batch size. Defaults to 32.
        ppo_epochs: PPO epochs per batch. Defaults to 4.
        rollout_steps: Rollout steps. Defaults to 128.

    Returns:
        PPOTrainingConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_ppo_training_config(learning_rate=2e-5)
        >>> config.learning_rate
        2e-05

        >>> config = create_ppo_training_config(ppo_epochs=8)
        >>> config.ppo_epochs
        8

        >>> create_ppo_training_config(learning_rate=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: learning_rate must be positive
    """
    config = PPOTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
        rollout_steps=rollout_steps,
    )
    validate_ppo_training_config(config)
    return config


def list_ppo_variants() -> list[str]:
    """List all supported PPO variants.

    Returns:
        Sorted list of PPO variant names.

    Examples:
        >>> variants = list_ppo_variants()
        >>> "ppo_clip" in variants
        True
        >>> "standard" in variants
        True
        >>> variants == sorted(variants)
        True
    """
    return sorted(VALID_PPO_VARIANTS)


def list_reward_model_types() -> list[str]:
    """List all supported reward model types.

    Returns:
        Sorted list of reward model type names.

    Examples:
        >>> types = list_reward_model_types()
        >>> "learned" in types
        True
        >>> "constitutional" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_REWARD_MODEL_TYPES)


def list_value_head_types() -> list[str]:
    """List all supported value head types.

    Returns:
        Sorted list of value head type names.

    Examples:
        >>> types = list_value_head_types()
        >>> "mlp" in types
        True
        >>> "linear" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_VALUE_HEAD_TYPES)


def get_ppo_variant(name: str) -> PPOVariant:
    """Get PPO variant from name.

    Args:
        name: Variant name.

    Returns:
        PPOVariant enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_ppo_variant("ppo_clip")
        <PPOVariant.PPO_CLIP: 'ppo_clip'>

        >>> get_ppo_variant("standard")
        <PPOVariant.STANDARD: 'standard'>

        >>> get_ppo_variant("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: variant must be one of
    """
    if name not in VALID_PPO_VARIANTS:
        msg = f"variant must be one of {VALID_PPO_VARIANTS}, got '{name}'"
        raise ValueError(msg)
    return PPOVariant(name)


def get_reward_model_type(name: str) -> RewardModelType:
    """Get reward model type from name.

    Args:
        name: Type name.

    Returns:
        RewardModelType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_reward_model_type("learned")
        <RewardModelType.LEARNED: 'learned'>

        >>> get_reward_model_type("constitutional")
        <RewardModelType.CONSTITUTIONAL: 'constitutional'>

        >>> get_reward_model_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of
    """
    if name not in VALID_REWARD_MODEL_TYPES:
        msg = f"model_type must be one of {VALID_REWARD_MODEL_TYPES}, got '{name}'"
        raise ValueError(msg)
    return RewardModelType(name)


def get_value_head_type(name: str) -> ValueHeadType:
    """Get value head type from name.

    Args:
        name: Type name.

    Returns:
        ValueHeadType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_value_head_type("mlp")
        <ValueHeadType.MLP: 'mlp'>

        >>> get_value_head_type("linear")
        <ValueHeadType.LINEAR: 'linear'>

        >>> get_value_head_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: head_type must be one of
    """
    if name not in VALID_VALUE_HEAD_TYPES:
        msg = f"head_type must be one of {VALID_VALUE_HEAD_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ValueHeadType(name)


def calculate_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """Calculate Generalized Advantage Estimation (GAE).

    Computes advantages using the GAE formula:
    A_t = delta_t + (gamma * lam) * delta_{t+1} + ...

    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        rewards: Sequence of rewards from rollout.
        values: Sequence of value estimates.
        next_value: Value estimate of final next state.
        gamma: Discount factor. Defaults to 0.99.
        lam: GAE lambda parameter. Defaults to 0.95.

    Returns:
        List of advantage estimates for each timestep.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> rewards = [1.0, 1.0, 1.0]
        >>> values = [0.5, 0.6, 0.7]
        >>> advantages = calculate_gae(rewards, values, next_value=0.8)
        >>> len(advantages)
        3
        >>> all(isinstance(a, float) for a in advantages)
        True

        >>> advantages = calculate_gae([1.0], [0.0], 0.0, gamma=1.0, lam=1.0)
        >>> round(advantages[0], 4)
        1.0

        >>> calculate_gae([], [], 0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rewards sequence cannot be empty

        >>> calculate_gae([1.0], [0.5, 0.6], 0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rewards and values must have same length
    """
    if len(rewards) == 0:
        msg = "rewards sequence cannot be empty"
        raise ValueError(msg)

    if len(rewards) != len(values):
        msg = (
            f"rewards and values must have same length "
            f"({len(rewards)} != {len(values)})"
        )
        raise ValueError(msg)

    if not 0.0 <= gamma <= 1.0:
        msg = f"gamma must be in [0.0, 1.0], got {gamma}"
        raise ValueError(msg)

    if not 0.0 <= lam <= 1.0:
        msg = f"lam must be in [0.0, 1.0], got {lam}"
        raise ValueError(msg)

    n = len(rewards)
    advantages: list[float] = [0.0] * n
    gae = 0.0

    # Iterate backwards through timesteps
    for t in range(n - 1, -1, -1):
        next_val = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages


def calculate_ppo_loss(
    log_probs: Sequence[float],
    old_log_probs: Sequence[float],
    advantages: Sequence[float],
    clip_range: float = 0.2,
) -> tuple[float, float]:
    """Calculate PPO clipped surrogate loss.

    Computes the clipped objective:
    L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    where ratio = exp(log_prob - old_log_prob)

    Args:
        log_probs: Current policy log probabilities.
        old_log_probs: Old policy log probabilities.
        advantages: Advantage estimates.
        clip_range: Clipping parameter epsilon. Defaults to 0.2.

    Returns:
        Tuple of (policy_loss, clip_fraction) where:
        - policy_loss: Mean clipped surrogate loss (negated for gradient ascent).
        - clip_fraction: Fraction of samples that were clipped.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> log_probs = [-1.0, -1.5, -2.0]
        >>> old_log_probs = [-1.0, -1.5, -2.0]
        >>> advantages = [1.0, 0.5, -0.5]
        >>> loss, clip_frac = calculate_ppo_loss(log_probs, old_log_probs, advantages)
        >>> round(loss, 4)
        -0.3333
        >>> clip_frac
        0.0

        >>> calculate_ppo_loss([], [], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: input sequences cannot be empty

        >>> calculate_ppo_loss([1.0], [1.0], [1.0], clip_range=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: clip_range must be positive
    """
    if len(log_probs) == 0:
        msg = "input sequences cannot be empty"
        raise ValueError(msg)

    if len(log_probs) != len(old_log_probs) or len(log_probs) != len(advantages):
        msg = "log_probs, old_log_probs, and advantages must have same length"
        raise ValueError(msg)

    if clip_range <= 0:
        msg = f"clip_range must be positive, got {clip_range}"
        raise ValueError(msg)

    n = len(log_probs)
    total_loss = 0.0
    clip_count = 0

    for i in range(n):
        # Compute probability ratio
        ratio = _exp_safe(log_probs[i] - old_log_probs[i])

        # Clipped ratio
        clipped_ratio = max(1.0 - clip_range, min(1.0 + clip_range, ratio))

        # Surrogate objectives
        surr1 = ratio * advantages[i]
        surr2 = clipped_ratio * advantages[i]

        # Take minimum (pessimistic bound)
        loss = min(surr1, surr2) if advantages[i] >= 0 else max(surr1, surr2)

        total_loss += loss

        # Check if clipped
        if abs(ratio - clipped_ratio) > 1e-8:
            clip_count += 1

    mean_loss = total_loss / n
    clip_fraction = clip_count / n

    # Negate because we maximize the objective (gradient ascent)
    return -mean_loss, clip_fraction


def _exp_safe(x: float) -> float:
    """Safe exponential to avoid overflow.

    Args:
        x: Input value.

    Returns:
        exp(x) with clamping to avoid overflow.

    Examples:
        >>> _exp_safe(0.0)
        1.0
        >>> _exp_safe(100.0) < float('inf')
        True
    """
    import math

    # Clamp to avoid overflow
    x_clamped = max(-20.0, min(20.0, x))
    return math.exp(x_clamped)
