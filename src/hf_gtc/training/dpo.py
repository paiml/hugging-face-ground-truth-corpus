"""Direct Preference Optimization (DPO) training utilities.

This module provides functions for configuring and running DPO training
and its variants (RSPO, IPO, KTO, ORPO) for preference-based fine-tuning
of language models.

Examples:
    >>> from hf_gtc.training.dpo import create_dpo_config, DPOVariant
    >>> config = create_dpo_config(variant=DPOVariant.STANDARD, beta=0.1)
    >>> config.beta
    0.1
    >>> config.variant
    <DPOVariant.STANDARD: 'standard'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DPOVariant(Enum):
    """Supported DPO algorithm variants.

    Attributes:
        STANDARD: Standard Direct Preference Optimization.
        RSPO: Robust Self-Play Preference Optimization.
        IPO: Identity Preference Optimization.
        KTO: Kahneman-Tversky Optimization.
        ORPO: Odds Ratio Preference Optimization.

    Examples:
        >>> DPOVariant.STANDARD.value
        'standard'
        >>> DPOVariant.IPO.value
        'ipo'
        >>> DPOVariant.KTO.value
        'kto'
    """

    STANDARD = "standard"
    RSPO = "rspo"
    IPO = "ipo"
    KTO = "kto"
    ORPO = "orpo"


class LossType(Enum):
    """Loss function types for DPO training.

    Attributes:
        SIGMOID: Standard sigmoid cross-entropy loss.
        HINGE: Hinge loss for margin-based optimization.
        IPO_LOSS: Identity preference optimization loss.

    Examples:
        >>> LossType.SIGMOID.value
        'sigmoid'
        >>> LossType.HINGE.value
        'hinge'
        >>> LossType.IPO_LOSS.value
        'ipo_loss'
    """

    SIGMOID = "sigmoid"
    HINGE = "hinge"
    IPO_LOSS = "ipo_loss"


class ReferencePolicy(Enum):
    """Reference policy types for DPO.

    Attributes:
        FROZEN: Frozen reference model (standard DPO).
        ONLINE: Online reference model with periodic updates.
        SELF: Self-reference (reference-free DPO).

    Examples:
        >>> ReferencePolicy.FROZEN.value
        'frozen'
        >>> ReferencePolicy.ONLINE.value
        'online'
        >>> ReferencePolicy.SELF.value
        'self'
    """

    FROZEN = "frozen"
    ONLINE = "online"
    SELF = "self"


VALID_DPO_VARIANTS = frozenset(v.value for v in DPOVariant)
VALID_LOSS_TYPES = frozenset(lt.value for lt in LossType)
VALID_REFERENCE_POLICIES = frozenset(r.value for r in ReferencePolicy)


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Configuration for DPO training.

    Attributes:
        variant: DPO algorithm variant to use.
        beta: Temperature parameter controlling deviation from reference.
        loss_type: Type of loss function to use.
        reference_free: Whether to use reference-free DPO.
        label_smoothing: Label smoothing factor for regularization.

    Examples:
        >>> config = DPOConfig(
        ...     variant=DPOVariant.STANDARD,
        ...     beta=0.1,
        ...     loss_type=LossType.SIGMOID,
        ...     reference_free=False,
        ...     label_smoothing=0.0,
        ... )
        >>> config.beta
        0.1
        >>> config.variant
        <DPOVariant.STANDARD: 'standard'>

        >>> config2 = DPOConfig(
        ...     variant=DPOVariant.IPO,
        ...     beta=0.5,
        ...     loss_type=LossType.IPO_LOSS,
        ...     reference_free=False,
        ...     label_smoothing=0.1,
        ... )
        >>> config2.label_smoothing
        0.1
    """

    variant: DPOVariant
    beta: float
    loss_type: LossType
    reference_free: bool
    label_smoothing: float


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """A preference pair for DPO training.

    Attributes:
        prompt: The input prompt text.
        chosen: The preferred/chosen response.
        rejected: The rejected/dispreferred response.
        chosen_score: Optional score for the chosen response.
        rejected_score: Optional score for the rejected response.

    Examples:
        >>> pair = PreferencePair(
        ...     prompt="What is 2+2?",
        ...     chosen="4",
        ...     rejected="5",
        ...     chosen_score=0.9,
        ...     rejected_score=0.1,
        ... )
        >>> pair.prompt
        'What is 2+2?'
        >>> pair.chosen
        '4'

        >>> pair2 = PreferencePair(
        ...     prompt="Hello",
        ...     chosen="Hi there!",
        ...     rejected="Go away",
        ...     chosen_score=None,
        ...     rejected_score=None,
        ... )
        >>> pair2.chosen_score is None
        True
    """

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float | None
    rejected_score: float | None


@dataclass(frozen=True, slots=True)
class DPOTrainingConfig:
    """Training configuration for DPO.

    Attributes:
        learning_rate: Learning rate for the optimizer.
        batch_size: Number of samples per training batch.
        max_length: Maximum total sequence length (prompt + response).
        max_prompt_length: Maximum length for the prompt portion.

    Examples:
        >>> config = DPOTrainingConfig(
        ...     learning_rate=5e-7,
        ...     batch_size=4,
        ...     max_length=512,
        ...     max_prompt_length=256,
        ... )
        >>> config.learning_rate
        5e-07
        >>> config.batch_size
        4

        >>> config2 = DPOTrainingConfig(
        ...     learning_rate=1e-6,
        ...     batch_size=8,
        ...     max_length=1024,
        ...     max_prompt_length=512,
        ... )
        >>> config2.max_length
        1024
    """

    learning_rate: float
    batch_size: int
    max_length: int
    max_prompt_length: int


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Configuration for the reference policy in DPO.

    Attributes:
        policy_type: Type of reference policy.
        update_frequency: Steps between reference model updates (for ONLINE).
        ema_decay: Exponential moving average decay for online updates.

    Examples:
        >>> config = ReferenceConfig(
        ...     policy_type=ReferencePolicy.FROZEN,
        ...     update_frequency=0,
        ...     ema_decay=0.0,
        ... )
        >>> config.policy_type
        <ReferencePolicy.FROZEN: 'frozen'>

        >>> config2 = ReferenceConfig(
        ...     policy_type=ReferencePolicy.ONLINE,
        ...     update_frequency=100,
        ...     ema_decay=0.99,
        ... )
        >>> config2.update_frequency
        100
    """

    policy_type: ReferencePolicy
    update_frequency: int
    ema_decay: float


@dataclass(frozen=True, slots=True)
class DPOStats:
    """Statistics from DPO training.

    Attributes:
        chosen_rewards_mean: Mean reward for chosen responses.
        rejected_rewards_mean: Mean reward for rejected responses.
        reward_margin: Difference between chosen and rejected rewards.
        accuracy: Accuracy of preference prediction.

    Examples:
        >>> stats = DPOStats(
        ...     chosen_rewards_mean=2.5,
        ...     rejected_rewards_mean=-1.5,
        ...     reward_margin=4.0,
        ...     accuracy=0.85,
        ... )
        >>> stats.chosen_rewards_mean
        2.5
        >>> stats.accuracy
        0.85

        >>> stats2 = DPOStats(
        ...     chosen_rewards_mean=1.0,
        ...     rejected_rewards_mean=0.5,
        ...     reward_margin=0.5,
        ...     accuracy=0.65,
        ... )
        >>> stats2.reward_margin
        0.5
    """

    chosen_rewards_mean: float
    rejected_rewards_mean: float
    reward_margin: float
    accuracy: float


def validate_dpo_config(config: DPOConfig) -> None:
    """Validate DPO configuration parameters.

    Args:
        config: DPO configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If beta is not positive.
        ValueError: If label_smoothing is not in [0.0, 1.0).

    Examples:
        >>> config = DPOConfig(
        ...     variant=DPOVariant.STANDARD,
        ...     beta=0.1,
        ...     loss_type=LossType.SIGMOID,
        ...     reference_free=False,
        ...     label_smoothing=0.0,
        ... )
        >>> validate_dpo_config(config)  # No error

        >>> validate_dpo_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = DPOConfig(
        ...     variant=DPOVariant.STANDARD,
        ...     beta=0.0,
        ...     loss_type=LossType.SIGMOID,
        ...     reference_free=False,
        ...     label_smoothing=0.0,
        ... )
        >>> validate_dpo_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beta must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.beta <= 0:
        msg = f"beta must be positive, got {config.beta}"
        raise ValueError(msg)

    if not 0.0 <= config.label_smoothing < 1.0:
        msg = f"label_smoothing must be in [0.0, 1.0), got {config.label_smoothing}"
        raise ValueError(msg)


def validate_preference_pair(pair: PreferencePair) -> None:
    """Validate a preference pair.

    Args:
        pair: Preference pair to validate.

    Raises:
        ValueError: If pair is None.
        ValueError: If prompt is empty.
        ValueError: If chosen is empty.
        ValueError: If rejected is empty.
        ValueError: If chosen and rejected are identical.

    Examples:
        >>> pair = PreferencePair(
        ...     prompt="Question?",
        ...     chosen="Good answer",
        ...     rejected="Bad answer",
        ...     chosen_score=None,
        ...     rejected_score=None,
        ... )
        >>> validate_preference_pair(pair)  # No error

        >>> validate_preference_pair(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pair cannot be None

        >>> bad_pair = PreferencePair(
        ...     prompt="",
        ...     chosen="Good",
        ...     rejected="Bad",
        ...     chosen_score=None,
        ...     rejected_score=None,
        ... )
        >>> validate_preference_pair(bad_pair)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: prompt cannot be empty

        >>> same_pair = PreferencePair(
        ...     prompt="Q?",
        ...     chosen="Same",
        ...     rejected="Same",
        ...     chosen_score=None,
        ...     rejected_score=None,
        ... )
        >>> validate_preference_pair(same_pair)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chosen and rejected cannot be identical
    """
    if pair is None:
        msg = "pair cannot be None"
        raise ValueError(msg)

    if not pair.prompt:
        msg = "prompt cannot be empty"
        raise ValueError(msg)

    if not pair.chosen:
        msg = "chosen cannot be empty"
        raise ValueError(msg)

    if not pair.rejected:
        msg = "rejected cannot be empty"
        raise ValueError(msg)

    if pair.chosen == pair.rejected:
        msg = "chosen and rejected cannot be identical"
        raise ValueError(msg)


def create_dpo_config(
    variant: DPOVariant | str = DPOVariant.STANDARD,
    beta: float = 0.1,
    loss_type: LossType | str = LossType.SIGMOID,
    reference_free: bool = False,
    label_smoothing: float = 0.0,
) -> DPOConfig:
    """Create a DPO configuration.

    Args:
        variant: DPO algorithm variant. Defaults to STANDARD.
        beta: Temperature parameter. Defaults to 0.1.
        loss_type: Loss function type. Defaults to SIGMOID.
        reference_free: Whether to use reference-free DPO. Defaults to False.
        label_smoothing: Label smoothing factor. Defaults to 0.0.

    Returns:
        Validated DPOConfig instance.

    Raises:
        ValueError: If beta is not positive.
        ValueError: If label_smoothing is not in [0.0, 1.0).
        ValueError: If variant is invalid.
        ValueError: If loss_type is invalid.

    Examples:
        >>> config = create_dpo_config(beta=0.1)
        >>> config.beta
        0.1
        >>> config.variant
        <DPOVariant.STANDARD: 'standard'>

        >>> config2 = create_dpo_config(variant="ipo", loss_type="ipo_loss")
        >>> config2.variant
        <DPOVariant.IPO: 'ipo'>

        >>> create_dpo_config(beta=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beta must be positive

        >>> create_dpo_config(variant="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid DPO variant: invalid
    """
    if isinstance(variant, str):
        variant = get_dpo_variant(variant)
    if isinstance(loss_type, str):
        loss_type = get_loss_type(loss_type)

    config = DPOConfig(
        variant=variant,
        beta=beta,
        loss_type=loss_type,
        reference_free=reference_free,
        label_smoothing=label_smoothing,
    )
    validate_dpo_config(config)
    return config


def create_preference_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    chosen_score: float | None = None,
    rejected_score: float | None = None,
) -> PreferencePair:
    """Create a preference pair.

    Args:
        prompt: The input prompt text.
        chosen: The preferred/chosen response.
        rejected: The rejected/dispreferred response.
        chosen_score: Optional score for chosen. Defaults to None.
        rejected_score: Optional score for rejected. Defaults to None.

    Returns:
        Validated PreferencePair instance.

    Raises:
        ValueError: If prompt is empty.
        ValueError: If chosen is empty.
        ValueError: If rejected is empty.
        ValueError: If chosen and rejected are identical.

    Examples:
        >>> pair = create_preference_pair(
        ...     prompt="What is Python?",
        ...     chosen="A programming language",
        ...     rejected="A snake",
        ... )
        >>> pair.prompt
        'What is Python?'

        >>> pair2 = create_preference_pair(
        ...     prompt="Hello",
        ...     chosen="Hi!",
        ...     rejected="...",
        ...     chosen_score=0.9,
        ...     rejected_score=0.2,
        ... )
        >>> pair2.chosen_score
        0.9

        >>> create_preference_pair("", "a", "b")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: prompt cannot be empty
    """
    pair = PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_score=chosen_score,
        rejected_score=rejected_score,
    )
    validate_preference_pair(pair)
    return pair


def create_dpo_training_config(
    learning_rate: float = 5e-7,
    batch_size: int = 4,
    max_length: int = 512,
    max_prompt_length: int = 256,
) -> DPOTrainingConfig:
    """Create a DPO training configuration.

    Args:
        learning_rate: Learning rate. Defaults to 5e-7.
        batch_size: Batch size. Defaults to 4.
        max_length: Maximum sequence length. Defaults to 512.
        max_prompt_length: Maximum prompt length. Defaults to 256.

    Returns:
        Validated DPOTrainingConfig instance.

    Raises:
        ValueError: If learning_rate is not positive.
        ValueError: If batch_size is not positive.
        ValueError: If max_length is not positive.
        ValueError: If max_prompt_length is not positive.
        ValueError: If max_prompt_length >= max_length.

    Examples:
        >>> config = create_dpo_training_config(learning_rate=1e-6)
        >>> config.learning_rate
        1e-06

        >>> config2 = create_dpo_training_config(batch_size=8, max_length=1024)
        >>> config2.batch_size
        8

        >>> create_dpo_training_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     learning_rate=0.0
        ... )
        Traceback (most recent call last):
        ValueError: learning_rate must be positive

        >>> create_dpo_training_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     max_length=256, max_prompt_length=256
        ... )
        Traceback (most recent call last):
        ValueError: max_prompt_length must be less than max_length
    """
    if learning_rate <= 0:
        msg = f"learning_rate must be positive, got {learning_rate}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if max_prompt_length <= 0:
        msg = f"max_prompt_length must be positive, got {max_prompt_length}"
        raise ValueError(msg)

    if max_prompt_length >= max_length:
        msg = (
            f"max_prompt_length ({max_prompt_length}) must be less than "
            f"max_length ({max_length})"
        )
        raise ValueError(msg)

    return DPOTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )


def create_reference_config(
    policy_type: ReferencePolicy | str = ReferencePolicy.FROZEN,
    update_frequency: int = 0,
    ema_decay: float = 0.0,
) -> ReferenceConfig:
    """Create a reference policy configuration.

    Args:
        policy_type: Type of reference policy. Defaults to FROZEN.
        update_frequency: Steps between updates. Defaults to 0.
        ema_decay: EMA decay factor. Defaults to 0.0.

    Returns:
        Validated ReferenceConfig instance.

    Raises:
        ValueError: If policy_type is invalid.
        ValueError: If update_frequency is negative.
        ValueError: If ema_decay is not in [0.0, 1.0].
        ValueError: If ONLINE policy has zero update_frequency.

    Examples:
        >>> config = create_reference_config()
        >>> config.policy_type
        <ReferencePolicy.FROZEN: 'frozen'>

        >>> config2 = create_reference_config(
        ...     policy_type="online",
        ...     update_frequency=100,
        ...     ema_decay=0.99,
        ... )
        >>> config2.update_frequency
        100

        >>> create_reference_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     policy_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: invalid reference policy: invalid

        >>> create_reference_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     policy_type="online", update_frequency=0
        ... )
        Traceback (most recent call last):
        ValueError: ONLINE policy requires positive update_frequency
    """
    if isinstance(policy_type, str):
        policy_type = get_reference_policy(policy_type)

    if update_frequency < 0:
        msg = f"update_frequency cannot be negative, got {update_frequency}"
        raise ValueError(msg)

    if not 0.0 <= ema_decay <= 1.0:
        msg = f"ema_decay must be in [0.0, 1.0], got {ema_decay}"
        raise ValueError(msg)

    if policy_type == ReferencePolicy.ONLINE and update_frequency == 0:
        msg = "ONLINE policy requires positive update_frequency"
        raise ValueError(msg)

    return ReferenceConfig(
        policy_type=policy_type,
        update_frequency=update_frequency,
        ema_decay=ema_decay,
    )


def list_dpo_variants() -> list[str]:
    """List all supported DPO variants.

    Returns:
        Sorted list of DPO variant names.

    Examples:
        >>> variants = list_dpo_variants()
        >>> "standard" in variants
        True
        >>> "ipo" in variants
        True
        >>> "kto" in variants
        True
        >>> variants == sorted(variants)
        True
    """
    return sorted(VALID_DPO_VARIANTS)


def list_loss_types() -> list[str]:
    """List all supported loss types.

    Returns:
        Sorted list of loss type names.

    Examples:
        >>> types = list_loss_types()
        >>> "sigmoid" in types
        True
        >>> "hinge" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_LOSS_TYPES)


def list_reference_policies() -> list[str]:
    """List all supported reference policy types.

    Returns:
        Sorted list of reference policy names.

    Examples:
        >>> policies = list_reference_policies()
        >>> "frozen" in policies
        True
        >>> "online" in policies
        True
        >>> "self" in policies
        True
        >>> policies == sorted(policies)
        True
    """
    return sorted(VALID_REFERENCE_POLICIES)


def get_dpo_variant(name: str) -> DPOVariant:
    """Get DPO variant enum from string.

    Args:
        name: Variant name.

    Returns:
        DPOVariant enum value.

    Raises:
        ValueError: If name is not a valid variant.

    Examples:
        >>> get_dpo_variant("standard")
        <DPOVariant.STANDARD: 'standard'>
        >>> get_dpo_variant("ipo")
        <DPOVariant.IPO: 'ipo'>
        >>> get_dpo_variant("kto")
        <DPOVariant.KTO: 'kto'>

        >>> get_dpo_variant("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid DPO variant: invalid
    """
    for v in DPOVariant:
        if v.value == name:
            return v
    msg = f"invalid DPO variant: {name}"
    raise ValueError(msg)


def get_loss_type(name: str) -> LossType:
    """Get loss type enum from string.

    Args:
        name: Loss type name.

    Returns:
        LossType enum value.

    Raises:
        ValueError: If name is not a valid loss type.

    Examples:
        >>> get_loss_type("sigmoid")
        <LossType.SIGMOID: 'sigmoid'>
        >>> get_loss_type("hinge")
        <LossType.HINGE: 'hinge'>
        >>> get_loss_type("ipo_loss")
        <LossType.IPO_LOSS: 'ipo_loss'>

        >>> get_loss_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid loss type: invalid
    """
    for lt in LossType:
        if lt.value == name:
            return lt
    msg = f"invalid loss type: {name}"
    raise ValueError(msg)


def get_reference_policy(name: str) -> ReferencePolicy:
    """Get reference policy enum from string.

    Args:
        name: Reference policy name.

    Returns:
        ReferencePolicy enum value.

    Raises:
        ValueError: If name is not a valid reference policy.

    Examples:
        >>> get_reference_policy("frozen")
        <ReferencePolicy.FROZEN: 'frozen'>
        >>> get_reference_policy("online")
        <ReferencePolicy.ONLINE: 'online'>
        >>> get_reference_policy("self")
        <ReferencePolicy.SELF: 'self'>

        >>> get_reference_policy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid reference policy: invalid
    """
    for r in ReferencePolicy:
        if r.value == name:
            return r
    msg = f"invalid reference policy: {name}"
    raise ValueError(msg)


def calculate_dpo_loss(
    chosen_reward: float,
    rejected_reward: float,
    beta: float = 0.1,
    loss_type: LossType = LossType.SIGMOID,
    label_smoothing: float = 0.0,
) -> float:
    """Compute DPO loss given reward values.

    Calculates the DPO loss based on the chosen and rejected reward
    values using the specified loss function.

    Args:
        chosen_reward: Reward/log-probability for chosen response.
        rejected_reward: Reward/log-probability for rejected response.
        beta: Temperature parameter. Defaults to 0.1.
        loss_type: Type of loss function. Defaults to SIGMOID.
        label_smoothing: Label smoothing factor. Defaults to 0.0.

    Returns:
        Computed DPO loss value.

    Raises:
        ValueError: If beta is not positive.
        ValueError: If label_smoothing is not in [0.0, 1.0).

    Examples:
        >>> loss = calculate_dpo_loss(2.0, -1.0, beta=0.1)
        >>> loss < 1.0
        True

        >>> loss_hinge = calculate_dpo_loss(
        ...     2.0, -1.0, beta=0.1, loss_type=LossType.HINGE
        ... )
        >>> loss_hinge >= 0.0
        True

        >>> loss_ipo = calculate_dpo_loss(
        ...     0.5, 0.3, beta=0.1, loss_type=LossType.IPO_LOSS
        ... )
        >>> loss_ipo >= 0.0
        True

        >>> calculate_dpo_loss(1.0, 0.0, beta=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beta must be positive
    """
    if beta <= 0:
        msg = f"beta must be positive, got {beta}"
        raise ValueError(msg)

    if not 0.0 <= label_smoothing < 1.0:
        msg = f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}"
        raise ValueError(msg)

    # Compute reward margin scaled by beta
    margin = beta * (chosen_reward - rejected_reward)

    if loss_type == LossType.SIGMOID:
        # Sigmoid cross-entropy loss: -log(sigmoid(margin))
        # With numerical stability
        if margin >= 0:
            loss = math.log(1 + math.exp(-margin))
        else:
            loss = -margin + math.log(1 + math.exp(margin))

        # Apply label smoothing
        if label_smoothing > 0:
            loss = (1 - label_smoothing) * loss + label_smoothing * 0.5

    elif loss_type == LossType.HINGE:
        # Hinge loss: max(0, 1 - margin)
        loss = max(0.0, 1.0 - margin)

    else:  # IPO_LOSS
        # IPO loss: (margin - 1)^2 / 2
        loss = ((margin - 1) ** 2) / 2

    return loss


def estimate_training_steps(
    dataset_size: int,
    batch_size: int = 4,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Estimate total training steps for a dataset.

    Args:
        dataset_size: Number of samples in the dataset.
        batch_size: Training batch size. Defaults to 4.
        num_epochs: Number of training epochs. Defaults to 1.
        gradient_accumulation_steps: Gradient accumulation. Defaults to 1.

    Returns:
        Estimated total number of training steps.

    Raises:
        ValueError: If dataset_size is not positive.
        ValueError: If batch_size is not positive.
        ValueError: If num_epochs is not positive.
        ValueError: If gradient_accumulation_steps is not positive.

    Examples:
        >>> estimate_training_steps(1000, batch_size=4, num_epochs=3)
        750

        >>> estimate_training_steps(500, batch_size=8, num_epochs=2)
        124

        >>> estimate_training_steps(1000, batch_size=4, gradient_accumulation_steps=2)
        125

        >>> estimate_training_steps(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     0, batch_size=4
        ... )
        Traceback (most recent call last):
        ValueError: dataset_size must be positive

        >>> estimate_training_steps(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     100, batch_size=0
        ... )
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if dataset_size <= 0:
        msg = f"dataset_size must be positive, got {dataset_size}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if num_epochs <= 0:
        msg = f"num_epochs must be positive, got {num_epochs}"
        raise ValueError(msg)

    if gradient_accumulation_steps <= 0:
        msg = (
            f"gradient_accumulation_steps must be positive, "
            f"got {gradient_accumulation_steps}"
        )
        raise ValueError(msg)

    # Effective batch size accounts for gradient accumulation
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Steps per epoch (floor division)
    steps_per_epoch = dataset_size // effective_batch_size

    # Total steps across all epochs
    return steps_per_epoch * num_epochs
