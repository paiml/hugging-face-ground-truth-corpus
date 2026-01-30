"""TRL (Transformer Reinforcement Learning) utilities.

This module provides functions for configuring and running RLHF,
DPO (Direct Preference Optimization), and PPO training with the
HuggingFace TRL library.

Examples:
    >>> from hf_gtc.training.trl import create_dpo_config
    >>> config = create_dpo_config(beta=0.1)
    >>> config.beta
    0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class TrainingMethod(Enum):
    """Supported TRL training methods.

    Attributes:
        DPO: Direct Preference Optimization.
        PPO: Proximal Policy Optimization.
        SFT: Supervised Fine-Tuning.
        ORPO: Odds Ratio Preference Optimization.
        KTO: Kahneman-Tversky Optimization.

    Examples:
        >>> TrainingMethod.DPO.value
        'dpo'
        >>> TrainingMethod.PPO.value
        'ppo'
    """

    DPO = "dpo"
    PPO = "ppo"
    SFT = "sft"
    ORPO = "orpo"
    KTO = "kto"


VALID_TRAINING_METHODS = frozenset(m.value for m in TrainingMethod)

# Loss types for DPO
DPOLossType = Literal["sigmoid", "hinge", "ipo", "kto_pair"]
VALID_DPO_LOSS_TYPES = frozenset({"sigmoid", "hinge", "ipo", "kto_pair"})


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Configuration for DPO training.

    Attributes:
        beta: Temperature parameter for DPO loss.
        loss_type: Type of DPO loss function.
        max_length: Maximum sequence length.
        max_prompt_length: Maximum prompt length.
        label_smoothing: Label smoothing factor.
        generate_during_eval: Whether to generate during evaluation.

    Examples:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     loss_type="sigmoid",
        ...     max_length=512,
        ...     max_prompt_length=256,
        ...     label_smoothing=0.0,
        ...     generate_during_eval=False,
        ... )
        >>> config.beta
        0.1
    """

    beta: float
    loss_type: DPOLossType
    max_length: int
    max_prompt_length: int
    label_smoothing: float
    generate_during_eval: bool


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Configuration for PPO training.

    Attributes:
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.
        mini_batch_size: Mini-batch size for PPO updates.
        ppo_epochs: Number of PPO epochs per batch.
        gamma: Discount factor for rewards.
        lam: Lambda for GAE (Generalized Advantage Estimation).
        cliprange: Clip range for PPO.
        cliprange_value: Clip range for value function.
        vf_coef: Value function coefficient.
        max_grad_norm: Maximum gradient norm.

    Examples:
        >>> config = PPOConfig(
        ...     learning_rate=1e-5,
        ...     batch_size=128,
        ...     mini_batch_size=32,
        ...     ppo_epochs=4,
        ...     gamma=1.0,
        ...     lam=0.95,
        ...     cliprange=0.2,
        ...     cliprange_value=0.2,
        ...     vf_coef=0.1,
        ...     max_grad_norm=1.0,
        ... )
        >>> config.ppo_epochs
        4
    """

    learning_rate: float
    batch_size: int
    mini_batch_size: int
    ppo_epochs: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Configuration for Supervised Fine-Tuning.

    Attributes:
        max_seq_length: Maximum sequence length.
        packing: Whether to use sequence packing.
        dataset_text_field: Name of the text field in dataset.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate.

    Examples:
        >>> config = SFTConfig(
        ...     max_seq_length=2048,
        ...     packing=True,
        ...     dataset_text_field="text",
        ...     num_train_epochs=3,
        ...     learning_rate=2e-5,
        ... )
        >>> config.packing
        True
    """

    max_seq_length: int
    packing: bool
    dataset_text_field: str
    num_train_epochs: int
    learning_rate: float


def validate_dpo_config(config: DPOConfig) -> None:
    """Validate DPO configuration parameters.

    Args:
        config: DPO configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     loss_type="sigmoid",
        ...     max_length=512,
        ...     max_prompt_length=256,
        ...     label_smoothing=0.0,
        ...     generate_during_eval=False,
        ... )
        >>> validate_dpo_config(config)  # No error

        >>> bad_config = DPOConfig(
        ...     beta=0.0,
        ...     loss_type="sigmoid",
        ...     max_length=512,
        ...     max_prompt_length=256,
        ...     label_smoothing=0.0,
        ...     generate_during_eval=False,
        ... )
        >>> validate_dpo_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beta must be positive
    """
    if config.beta <= 0:
        msg = f"beta must be positive, got {config.beta}"
        raise ValueError(msg)

    if config.loss_type not in VALID_DPO_LOSS_TYPES:
        msg = (
            f"loss_type must be one of {VALID_DPO_LOSS_TYPES}, got '{config.loss_type}'"
        )
        raise ValueError(msg)

    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.max_prompt_length <= 0:
        msg = f"max_prompt_length must be positive, got {config.max_prompt_length}"
        raise ValueError(msg)

    if config.max_prompt_length >= config.max_length:
        msg = (
            f"max_prompt_length ({config.max_prompt_length}) must be less than "
            f"max_length ({config.max_length})"
        )
        raise ValueError(msg)

    if not 0.0 <= config.label_smoothing < 1.0:
        msg = f"label_smoothing must be in [0.0, 1.0), got {config.label_smoothing}"
        raise ValueError(msg)


def validate_ppo_config(config: PPOConfig) -> None:
    """Validate PPO configuration parameters.

    Args:
        config: PPO configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = PPOConfig(
        ...     learning_rate=1e-5,
        ...     batch_size=128,
        ...     mini_batch_size=32,
        ...     ppo_epochs=4,
        ...     gamma=1.0,
        ...     lam=0.95,
        ...     cliprange=0.2,
        ...     cliprange_value=0.2,
        ...     vf_coef=0.1,
        ...     max_grad_norm=1.0,
        ... )
        >>> validate_ppo_config(config)  # No error

        >>> bad_config = PPOConfig(
        ...     learning_rate=0.0,
        ...     batch_size=128,
        ...     mini_batch_size=32,
        ...     ppo_epochs=4,
        ...     gamma=1.0,
        ...     lam=0.95,
        ...     cliprange=0.2,
        ...     cliprange_value=0.2,
        ...     vf_coef=0.1,
        ...     max_grad_norm=1.0,
        ... )
        >>> validate_ppo_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: learning_rate must be positive
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

    if not 0.0 <= config.gamma <= 1.0:
        msg = f"gamma must be in [0.0, 1.0], got {config.gamma}"
        raise ValueError(msg)

    if not 0.0 <= config.lam <= 1.0:
        msg = f"lam must be in [0.0, 1.0], got {config.lam}"
        raise ValueError(msg)

    if config.cliprange <= 0:
        msg = f"cliprange must be positive, got {config.cliprange}"
        raise ValueError(msg)

    if config.max_grad_norm <= 0:
        msg = f"max_grad_norm must be positive, got {config.max_grad_norm}"
        raise ValueError(msg)


def validate_sft_config(config: SFTConfig) -> None:
    """Validate SFT configuration parameters.

    Args:
        config: SFT configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = SFTConfig(
        ...     max_seq_length=2048,
        ...     packing=True,
        ...     dataset_text_field="text",
        ...     num_train_epochs=3,
        ...     learning_rate=2e-5,
        ... )
        >>> validate_sft_config(config)  # No error

        >>> bad_config = SFTConfig(
        ...     max_seq_length=0,
        ...     packing=True,
        ...     dataset_text_field="text",
        ...     num_train_epochs=3,
        ...     learning_rate=2e-5,
        ... )
        >>> validate_sft_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_seq_length must be positive
    """
    if config.max_seq_length <= 0:
        msg = f"max_seq_length must be positive, got {config.max_seq_length}"
        raise ValueError(msg)

    if not config.dataset_text_field:
        msg = "dataset_text_field cannot be empty"
        raise ValueError(msg)

    if config.num_train_epochs <= 0:
        msg = f"num_train_epochs must be positive, got {config.num_train_epochs}"
        raise ValueError(msg)

    if config.learning_rate <= 0:
        msg = f"learning_rate must be positive, got {config.learning_rate}"
        raise ValueError(msg)


def create_dpo_config(
    beta: float = 0.1,
    loss_type: DPOLossType = "sigmoid",
    max_length: int = 512,
    max_prompt_length: int = 256,
    label_smoothing: float = 0.0,
    generate_during_eval: bool = False,
) -> DPOConfig:
    """Create a DPO configuration.

    Args:
        beta: Temperature parameter. Defaults to 0.1.
        loss_type: Loss function type. Defaults to "sigmoid".
        max_length: Maximum sequence length. Defaults to 512.
        max_prompt_length: Maximum prompt length. Defaults to 256.
        label_smoothing: Label smoothing factor. Defaults to 0.0.
        generate_during_eval: Generate during eval. Defaults to False.

    Returns:
        DPOConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_dpo_config(beta=0.1)
        >>> config.beta
        0.1

        >>> config = create_dpo_config(loss_type="hinge")
        >>> config.loss_type
        'hinge'

        >>> create_dpo_config(beta=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beta must be positive
    """
    config = DPOConfig(
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        label_smoothing=label_smoothing,
        generate_during_eval=generate_during_eval,
    )
    validate_dpo_config(config)
    return config


def create_ppo_config(
    learning_rate: float = 1e-5,
    batch_size: int = 128,
    mini_batch_size: int = 32,
    ppo_epochs: int = 4,
    gamma: float = 1.0,
    lam: float = 0.95,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    vf_coef: float = 0.1,
    max_grad_norm: float = 1.0,
) -> PPOConfig:
    """Create a PPO configuration.

    Args:
        learning_rate: Learning rate. Defaults to 1e-5.
        batch_size: Batch size. Defaults to 128.
        mini_batch_size: Mini-batch size. Defaults to 32.
        ppo_epochs: PPO epochs per batch. Defaults to 4.
        gamma: Discount factor. Defaults to 1.0.
        lam: GAE lambda. Defaults to 0.95.
        cliprange: PPO clip range. Defaults to 0.2.
        cliprange_value: Value function clip range. Defaults to 0.2.
        vf_coef: Value function coefficient. Defaults to 0.1.
        max_grad_norm: Maximum gradient norm. Defaults to 1.0.

    Returns:
        PPOConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_ppo_config(learning_rate=1e-5)
        >>> config.learning_rate
        1e-05

        >>> config = create_ppo_config(ppo_epochs=8)
        >>> config.ppo_epochs
        8

        >>> create_ppo_config(learning_rate=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: learning_rate must be positive
    """
    config = PPOConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
        gamma=gamma,
        lam=lam,
        cliprange=cliprange,
        cliprange_value=cliprange_value,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
    )
    validate_ppo_config(config)
    return config


def create_sft_config(
    max_seq_length: int = 2048,
    packing: bool = True,
    dataset_text_field: str = "text",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
) -> SFTConfig:
    """Create an SFT configuration.

    Args:
        max_seq_length: Maximum sequence length. Defaults to 2048.
        packing: Whether to use packing. Defaults to True.
        dataset_text_field: Text field name. Defaults to "text".
        num_train_epochs: Training epochs. Defaults to 3.
        learning_rate: Learning rate. Defaults to 2e-5.

    Returns:
        SFTConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_sft_config(max_seq_length=4096)
        >>> config.max_seq_length
        4096

        >>> config = create_sft_config(packing=False)
        >>> config.packing
        False

        >>> create_sft_config(max_seq_length=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_seq_length must be positive
    """
    config = SFTConfig(
        max_seq_length=max_seq_length,
        packing=packing,
        dataset_text_field=dataset_text_field,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
    )
    validate_sft_config(config)
    return config


def list_training_methods() -> list[str]:
    """List all supported TRL training methods.

    Returns:
        Sorted list of training method names.

    Examples:
        >>> methods = list_training_methods()
        >>> "dpo" in methods
        True
        >>> "ppo" in methods
        True
        >>> "sft" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_TRAINING_METHODS)


def get_recommended_beta(model_size: str) -> float:
    """Get recommended DPO beta for a model size.

    Args:
        model_size: Model size category ("small", "medium", "large").

    Returns:
        Recommended beta value.

    Raises:
        ValueError: If model_size is invalid.

    Examples:
        >>> get_recommended_beta("small")
        0.5
        >>> get_recommended_beta("medium")
        0.1
        >>> get_recommended_beta("large")
        0.05
    """
    valid_sizes = {"small", "medium", "large"}
    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    beta_map = {
        "small": 0.5,
        "medium": 0.1,
        "large": 0.05,
    }
    return beta_map[model_size]


def estimate_reward_model_params(
    base_model_params: int,
    value_head_hidden_size: int = 768,
) -> int:
    """Estimate reward model parameter count.

    Args:
        base_model_params: Parameters in base model.
        value_head_hidden_size: Hidden size of value head. Defaults to 768.

    Returns:
        Estimated total parameters including value head.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> estimate_reward_model_params(7_000_000_000)
        7000000769

        >>> estimate_reward_model_params(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: base_model_params must be positive
    """
    if base_model_params <= 0:
        msg = f"base_model_params must be positive, got {base_model_params}"
        raise ValueError(msg)

    if value_head_hidden_size <= 0:
        msg = f"value_head_hidden_size must be positive, got {value_head_hidden_size}"
        raise ValueError(msg)

    # Value head: hidden_size -> 1 (scalar reward)
    value_head_params = value_head_hidden_size + 1  # weights + bias

    return base_model_params + value_head_params


def calculate_kl_penalty(
    log_prob_policy: float,
    log_prob_reference: float,
    kl_coef: float = 0.1,
) -> float:
    """Calculate KL divergence penalty for PPO.

    Args:
        log_prob_policy: Log probability from policy model.
        log_prob_reference: Log probability from reference model.
        kl_coef: KL penalty coefficient. Defaults to 0.1.

    Returns:
        KL penalty value.

    Raises:
        ValueError: If kl_coef is not positive.

    Examples:
        >>> penalty = calculate_kl_penalty(-2.0, -2.5, kl_coef=0.1)
        >>> round(penalty, 4)
        0.05

        >>> calculate_kl_penalty(-2.0, -2.5, kl_coef=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: kl_coef must be positive
    """
    if kl_coef <= 0:
        msg = f"kl_coef must be positive, got {kl_coef}"
        raise ValueError(msg)

    # KL divergence approximation: log(p/q) = log(p) - log(q)
    kl_div = log_prob_policy - log_prob_reference

    return kl_coef * kl_div


def get_default_reward_prompt() -> str:
    """Get a default system prompt for reward modeling.

    Returns:
        Default system prompt for reward model training.

    Examples:
        >>> prompt = get_default_reward_prompt()
        >>> "helpful" in prompt.lower()
        True
        >>> "harmless" in prompt.lower()
        True
    """
    return (
        "You are a helpful, harmless, and honest AI assistant. "
        "Evaluate the response based on helpfulness, accuracy, and safety."
    )
