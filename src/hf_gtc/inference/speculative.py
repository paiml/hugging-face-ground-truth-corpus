"""Speculative decoding utilities for accelerated inference.

This module provides utilities for configuring speculative decoding with
HuggingFace models, including draft model configuration, verification
strategies, and performance statistics.

Examples:
    >>> from hf_gtc.inference.speculative import (
    ...     SpeculativeConfig,
    ...     DraftModelType,
    ...     create_speculative_config,
    ... )
    >>> config = create_speculative_config(model_name="gpt2")
    >>> config.draft_config.model_name
    'gpt2'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class VerificationStrategy(Enum):
    """Strategy for verifying speculated tokens.

    Attributes:
        GREEDY: Accept tokens that match greedy decoding from target model.
        SAMPLING: Use rejection sampling based on target distribution.
        NUCLEUS: Use nucleus (top-p) sampling for verification.

    Examples:
        >>> VerificationStrategy.GREEDY.value
        'greedy'
        >>> VerificationStrategy.SAMPLING.value
        'sampling'
        >>> VerificationStrategy.NUCLEUS.value
        'nucleus'
    """

    GREEDY = "greedy"
    SAMPLING = "sampling"
    NUCLEUS = "nucleus"


VALID_VERIFICATION_STRATEGIES = frozenset(v.value for v in VerificationStrategy)


class DraftModelType(Enum):
    """Type of draft model used for speculation.

    Attributes:
        SMALLER_SAME_FAMILY: Smaller model from same family (e.g., Llama-7B).
        DISTILLED: Knowledge-distilled version of target model.
        NGRAM: N-gram based draft model for simple pattern matching.
        MEDUSA: Multi-head Medusa architecture for parallel speculation.

    Examples:
        >>> DraftModelType.SMALLER_SAME_FAMILY.value
        'smaller_same_family'
        >>> DraftModelType.DISTILLED.value
        'distilled'
        >>> DraftModelType.NGRAM.value
        'ngram'
        >>> DraftModelType.MEDUSA.value
        'medusa'
    """

    SMALLER_SAME_FAMILY = "smaller_same_family"
    DISTILLED = "distilled"
    NGRAM = "ngram"
    MEDUSA = "medusa"


VALID_DRAFT_MODEL_TYPES = frozenset(d.value for d in DraftModelType)


class AcceptanceCriteria(Enum):
    """Criteria for accepting speculated tokens.

    Attributes:
        THRESHOLD: Accept if probability exceeds fixed threshold.
        TOP_K: Accept if token is in top-k of target distribution.
        ADAPTIVE: Dynamically adjust acceptance based on running statistics.

    Examples:
        >>> AcceptanceCriteria.THRESHOLD.value
        'threshold'
        >>> AcceptanceCriteria.TOP_K.value
        'top_k'
        >>> AcceptanceCriteria.ADAPTIVE.value
        'adaptive'
    """

    THRESHOLD = "threshold"
    TOP_K = "top_k"
    ADAPTIVE = "adaptive"


VALID_ACCEPTANCE_CRITERIA = frozenset(a.value for a in AcceptanceCriteria)


@dataclass(frozen=True, slots=True)
class DraftModelConfig:
    """Configuration for the draft model in speculative decoding.

    Attributes:
        model_type: Type of draft model. Defaults to SMALLER_SAME_FAMILY.
        model_name: HuggingFace model ID or path. Defaults to empty string.
        gamma_tokens: Number of tokens to speculate per step. Defaults to 5.
        temperature: Sampling temperature for draft model. Defaults to 1.0.

    Examples:
        >>> config = DraftModelConfig(
        ...     model_type=DraftModelType.SMALLER_SAME_FAMILY,
        ...     model_name="gpt2",
        ...     gamma_tokens=5,
        ...     temperature=0.8,
        ... )
        >>> config.model_name
        'gpt2'
        >>> config.gamma_tokens
        5

        >>> config = DraftModelConfig()
        >>> config.gamma_tokens
        5
        >>> config.temperature
        1.0
    """

    model_type: DraftModelType = DraftModelType.SMALLER_SAME_FAMILY
    model_name: str = ""
    gamma_tokens: int = 5
    temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """Configuration for verification of speculated tokens.

    Attributes:
        strategy: Verification strategy. Defaults to GREEDY.
        acceptance_criteria: Criteria for token acceptance. Defaults to THRESHOLD.
        threshold: Probability threshold for acceptance. Defaults to 0.9.
        fallback_to_target: Fall back to target model on rejection. Defaults to True.

    Examples:
        >>> config = VerificationConfig(
        ...     strategy=VerificationStrategy.GREEDY,
        ...     acceptance_criteria=AcceptanceCriteria.THRESHOLD,
        ...     threshold=0.95,
        ...     fallback_to_target=True,
        ... )
        >>> config.strategy
        <VerificationStrategy.GREEDY: 'greedy'>
        >>> config.threshold
        0.95

        >>> config = VerificationConfig()
        >>> config.threshold
        0.9
        >>> config.fallback_to_target
        True
    """

    strategy: VerificationStrategy = VerificationStrategy.GREEDY
    acceptance_criteria: AcceptanceCriteria = AcceptanceCriteria.THRESHOLD
    threshold: float = 0.9
    fallback_to_target: bool = True


@dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    """Complete configuration for speculative decoding.

    Attributes:
        draft_config: Configuration for the draft model.
        verification_config: Configuration for token verification.
        max_speculation_length: Maximum length of speculated sequence. Defaults to 128.

    Examples:
        >>> draft = DraftModelConfig(model_name="gpt2", gamma_tokens=5)
        >>> verification = VerificationConfig(threshold=0.9)
        >>> config = SpeculativeConfig(
        ...     draft_config=draft,
        ...     verification_config=verification,
        ...     max_speculation_length=256,
        ... )
        >>> config.draft_config.model_name
        'gpt2'
        >>> config.max_speculation_length
        256

        >>> config = SpeculativeConfig(
        ...     draft_config=DraftModelConfig(),
        ...     verification_config=VerificationConfig(),
        ... )
        >>> config.max_speculation_length
        128
    """

    draft_config: DraftModelConfig
    verification_config: VerificationConfig
    max_speculation_length: int = 128


@dataclass(frozen=True, slots=True)
class SpeculativeStats:
    """Statistics from speculative decoding inference.

    Attributes:
        accepted_tokens: Number of tokens accepted from draft model.
        rejected_tokens: Number of tokens rejected by verification.
        speedup_factor: Measured speedup compared to standard decoding.
        acceptance_rate: Rate of token acceptance (0.0 to 1.0).

    Examples:
        >>> stats = SpeculativeStats(
        ...     accepted_tokens=80,
        ...     rejected_tokens=20,
        ...     speedup_factor=2.5,
        ...     acceptance_rate=0.8,
        ... )
        >>> stats.accepted_tokens
        80
        >>> stats.speedup_factor
        2.5
        >>> stats.acceptance_rate
        0.8
    """

    accepted_tokens: int
    rejected_tokens: int
    speedup_factor: float
    acceptance_rate: float


def validate_draft_model_config(config: DraftModelConfig) -> None:
    """Validate draft model configuration.

    Args:
        config: DraftModelConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If gamma_tokens is not positive.
        ValueError: If temperature is not positive.

    Examples:
        >>> config = DraftModelConfig(model_name="gpt2", gamma_tokens=5)
        >>> validate_draft_model_config(config)  # No error

        >>> validate_draft_model_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = DraftModelConfig(gamma_tokens=0)
        >>> validate_draft_model_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma_tokens must be positive

        >>> bad_config = DraftModelConfig(temperature=-0.5)
        >>> validate_draft_model_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive
    """
    validate_not_none(config, "config")

    if config.gamma_tokens <= 0:
        msg = f"gamma_tokens must be positive, got {config.gamma_tokens}"
        raise ValueError(msg)

    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)


def validate_verification_config(config: VerificationConfig) -> None:
    """Validate verification configuration.

    Args:
        config: VerificationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If threshold is not in [0.0, 1.0].

    Examples:
        >>> config = VerificationConfig(threshold=0.9)
        >>> validate_verification_config(config)  # No error

        >>> validate_verification_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = VerificationConfig(threshold=-0.1)
        >>> validate_verification_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0.0 and 1.0

        >>> bad_config = VerificationConfig(threshold=1.5)
        >>> validate_verification_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0.0 and 1.0
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.threshold <= 1.0:
        msg = f"threshold must be between 0.0 and 1.0, got {config.threshold}"
        raise ValueError(msg)


def validate_speculative_config(config: SpeculativeConfig) -> None:
    """Validate complete speculative decoding configuration.

    Args:
        config: SpeculativeConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_speculation_length is not positive.
        ValueError: If draft_config is invalid.
        ValueError: If verification_config is invalid.

    Examples:
        >>> draft = DraftModelConfig(model_name="gpt2", gamma_tokens=5)
        >>> verification = VerificationConfig(threshold=0.9)
        >>> config = SpeculativeConfig(draft, verification, 256)
        >>> validate_speculative_config(config)  # No error

        >>> validate_speculative_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = SpeculativeConfig(
        ...     DraftModelConfig(), VerificationConfig(), max_speculation_length=0
        ... )
        >>> validate_speculative_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_speculation_length must be positive
    """
    validate_not_none(config, "config")

    if config.max_speculation_length <= 0:
        msg = (
            f"max_speculation_length must be positive, "
            f"got {config.max_speculation_length}"
        )
        raise ValueError(msg)

    validate_draft_model_config(config.draft_config)
    validate_verification_config(config.verification_config)


def create_draft_model_config(
    model_type: str = "smaller_same_family",
    model_name: str = "",
    gamma_tokens: int = 5,
    temperature: float = 1.0,
) -> DraftModelConfig:
    """Create a draft model configuration.

    Args:
        model_type: Type of draft model. Defaults to "smaller_same_family".
        model_name: HuggingFace model ID or path. Defaults to empty string.
        gamma_tokens: Number of tokens to speculate. Defaults to 5.
        temperature: Sampling temperature. Defaults to 1.0.

    Returns:
        DraftModelConfig with the specified settings.

    Raises:
        ValueError: If model_type is invalid.
        ValueError: If gamma_tokens is not positive.
        ValueError: If temperature is not positive.

    Examples:
        >>> config = create_draft_model_config(model_name="gpt2")
        >>> config.model_name
        'gpt2'
        >>> config.model_type
        <DraftModelType.SMALLER_SAME_FAMILY: 'smaller_same_family'>

        >>> config = create_draft_model_config(
        ...     model_type="distilled",
        ...     model_name="distilgpt2",
        ...     gamma_tokens=8,
        ... )
        >>> config.model_type
        <DraftModelType.DISTILLED: 'distilled'>
        >>> config.gamma_tokens
        8

        >>> create_draft_model_config(model_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of

        >>> create_draft_model_config(gamma_tokens=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma_tokens must be positive
    """
    if model_type not in VALID_DRAFT_MODEL_TYPES:
        msg = f"model_type must be one of {VALID_DRAFT_MODEL_TYPES}, got '{model_type}'"
        raise ValueError(msg)

    config = DraftModelConfig(
        model_type=DraftModelType(model_type),
        model_name=model_name,
        gamma_tokens=gamma_tokens,
        temperature=temperature,
    )
    validate_draft_model_config(config)
    return config


def create_verification_config(
    strategy: str = "greedy",
    acceptance_criteria: str = "threshold",
    threshold: float = 0.9,
    fallback_to_target: bool = True,
) -> VerificationConfig:
    """Create a verification configuration.

    Args:
        strategy: Verification strategy. Defaults to "greedy".
        acceptance_criteria: Acceptance criteria. Defaults to "threshold".
        threshold: Probability threshold. Defaults to 0.9.
        fallback_to_target: Whether to fall back to target. Defaults to True.

    Returns:
        VerificationConfig with the specified settings.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If acceptance_criteria is invalid.
        ValueError: If threshold is not in [0.0, 1.0].

    Examples:
        >>> config = create_verification_config(threshold=0.95)
        >>> config.threshold
        0.95
        >>> config.strategy
        <VerificationStrategy.GREEDY: 'greedy'>

        >>> config = create_verification_config(
        ...     strategy="sampling",
        ...     acceptance_criteria="adaptive",
        ... )
        >>> config.strategy
        <VerificationStrategy.SAMPLING: 'sampling'>
        >>> config.acceptance_criteria
        <AcceptanceCriteria.ADAPTIVE: 'adaptive'>

        >>> create_verification_config(strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of

        >>> create_verification_config(acceptance_criteria="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: acceptance_criteria must be one of

        >>> create_verification_config(threshold=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0.0 and 1.0
    """
    if strategy not in VALID_VERIFICATION_STRATEGIES:
        msg = (
            f"strategy must be one of {VALID_VERIFICATION_STRATEGIES}, got '{strategy}'"
        )
        raise ValueError(msg)

    if acceptance_criteria not in VALID_ACCEPTANCE_CRITERIA:
        msg = (
            f"acceptance_criteria must be one of {VALID_ACCEPTANCE_CRITERIA}, "
            f"got '{acceptance_criteria}'"
        )
        raise ValueError(msg)

    config = VerificationConfig(
        strategy=VerificationStrategy(strategy),
        acceptance_criteria=AcceptanceCriteria(acceptance_criteria),
        threshold=threshold,
        fallback_to_target=fallback_to_target,
    )
    validate_verification_config(config)
    return config


def create_speculative_config(
    model_name: str = "",
    model_type: str = "smaller_same_family",
    gamma_tokens: int = 5,
    temperature: float = 1.0,
    strategy: str = "greedy",
    acceptance_criteria: str = "threshold",
    threshold: float = 0.9,
    fallback_to_target: bool = True,
    max_speculation_length: int = 128,
) -> SpeculativeConfig:
    """Create a complete speculative decoding configuration.

    Args:
        model_name: HuggingFace model ID or path. Defaults to empty string.
        model_type: Type of draft model. Defaults to "smaller_same_family".
        gamma_tokens: Number of tokens to speculate. Defaults to 5.
        temperature: Sampling temperature. Defaults to 1.0.
        strategy: Verification strategy. Defaults to "greedy".
        acceptance_criteria: Acceptance criteria. Defaults to "threshold".
        threshold: Probability threshold. Defaults to 0.9.
        fallback_to_target: Whether to fall back to target. Defaults to True.
        max_speculation_length: Maximum speculation length. Defaults to 128.

    Returns:
        SpeculativeConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_speculative_config(model_name="gpt2")
        >>> config.draft_config.model_name
        'gpt2'
        >>> config.verification_config.threshold
        0.9

        >>> config = create_speculative_config(
        ...     model_name="distilgpt2",
        ...     model_type="distilled",
        ...     gamma_tokens=8,
        ...     strategy="sampling",
        ...     max_speculation_length=256,
        ... )
        >>> config.draft_config.gamma_tokens
        8
        >>> config.max_speculation_length
        256

        >>> create_speculative_config(model_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of

        >>> create_speculative_config(max_speculation_length=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_speculation_length must be positive
    """
    draft_config = create_draft_model_config(
        model_type=model_type,
        model_name=model_name,
        gamma_tokens=gamma_tokens,
        temperature=temperature,
    )
    verification_config = create_verification_config(
        strategy=strategy,
        acceptance_criteria=acceptance_criteria,
        threshold=threshold,
        fallback_to_target=fallback_to_target,
    )
    config = SpeculativeConfig(
        draft_config=draft_config,
        verification_config=verification_config,
        max_speculation_length=max_speculation_length,
    )
    validate_speculative_config(config)
    return config


def list_verification_strategies() -> list[str]:
    """List all available verification strategies.

    Returns:
        Sorted list of verification strategy names.

    Examples:
        >>> strategies = list_verification_strategies()
        >>> "greedy" in strategies
        True
        >>> "sampling" in strategies
        True
        >>> "nucleus" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_VERIFICATION_STRATEGIES)


def list_draft_model_types() -> list[str]:
    """List all available draft model types.

    Returns:
        Sorted list of draft model type names.

    Examples:
        >>> types = list_draft_model_types()
        >>> "smaller_same_family" in types
        True
        >>> "distilled" in types
        True
        >>> "ngram" in types
        True
        >>> "medusa" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_DRAFT_MODEL_TYPES)


def list_acceptance_criteria() -> list[str]:
    """List all available acceptance criteria.

    Returns:
        Sorted list of acceptance criteria names.

    Examples:
        >>> criteria = list_acceptance_criteria()
        >>> "threshold" in criteria
        True
        >>> "top_k" in criteria
        True
        >>> "adaptive" in criteria
        True
        >>> criteria == sorted(criteria)
        True
    """
    return sorted(VALID_ACCEPTANCE_CRITERIA)


def get_verification_strategy(name: str) -> VerificationStrategy:
    """Get a verification strategy by name.

    Args:
        name: Name of the verification strategy.

    Returns:
        The corresponding VerificationStrategy enum value.

    Raises:
        ValueError: If name is not a valid verification strategy.

    Examples:
        >>> get_verification_strategy("greedy")
        <VerificationStrategy.GREEDY: 'greedy'>
        >>> get_verification_strategy("sampling")
        <VerificationStrategy.SAMPLING: 'sampling'>
        >>> get_verification_strategy("nucleus")
        <VerificationStrategy.NUCLEUS: 'nucleus'>

        >>> get_verification_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown verification strategy: 'invalid'

        >>> get_verification_strategy("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown verification strategy: ''
    """
    if name not in VALID_VERIFICATION_STRATEGIES:
        valid = sorted(VALID_VERIFICATION_STRATEGIES)
        msg = f"Unknown verification strategy: '{name}'. Valid strategies: {valid}"
        raise ValueError(msg)

    return VerificationStrategy(name)


def get_draft_model_type(name: str) -> DraftModelType:
    """Get a draft model type by name.

    Args:
        name: Name of the draft model type.

    Returns:
        The corresponding DraftModelType enum value.

    Raises:
        ValueError: If name is not a valid draft model type.

    Examples:
        >>> get_draft_model_type("smaller_same_family")
        <DraftModelType.SMALLER_SAME_FAMILY: 'smaller_same_family'>
        >>> get_draft_model_type("distilled")
        <DraftModelType.DISTILLED: 'distilled'>
        >>> get_draft_model_type("ngram")
        <DraftModelType.NGRAM: 'ngram'>
        >>> get_draft_model_type("medusa")
        <DraftModelType.MEDUSA: 'medusa'>

        >>> get_draft_model_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown draft model type: 'invalid'

        >>> get_draft_model_type("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown draft model type: ''
    """
    if name not in VALID_DRAFT_MODEL_TYPES:
        valid = sorted(VALID_DRAFT_MODEL_TYPES)
        msg = f"Unknown draft model type: '{name}'. Valid types: {valid}"
        raise ValueError(msg)

    return DraftModelType(name)


def get_acceptance_criteria(name: str) -> AcceptanceCriteria:
    """Get an acceptance criteria by name.

    Args:
        name: Name of the acceptance criteria.

    Returns:
        The corresponding AcceptanceCriteria enum value.

    Raises:
        ValueError: If name is not a valid acceptance criteria.

    Examples:
        >>> get_acceptance_criteria("threshold")
        <AcceptanceCriteria.THRESHOLD: 'threshold'>
        >>> get_acceptance_criteria("top_k")
        <AcceptanceCriteria.TOP_K: 'top_k'>
        >>> get_acceptance_criteria("adaptive")
        <AcceptanceCriteria.ADAPTIVE: 'adaptive'>

        >>> get_acceptance_criteria("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown acceptance criteria: 'invalid'

        >>> get_acceptance_criteria("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown acceptance criteria: ''
    """
    if name not in VALID_ACCEPTANCE_CRITERIA:
        valid = sorted(VALID_ACCEPTANCE_CRITERIA)
        msg = f"Unknown acceptance criteria: '{name}'. Valid criteria: {valid}"
        raise ValueError(msg)

    return AcceptanceCriteria(name)


def calculate_expected_speedup(
    acceptance_rate: float,
    gamma_tokens: int,
    draft_latency_ratio: float = 0.1,
) -> float:
    """Calculate expected speedup from speculative decoding.

    The speedup is calculated based on the expected number of tokens accepted
    per verification step and the relative latency of the draft model.

    Args:
        acceptance_rate: Expected rate of token acceptance (0.0 to 1.0).
        gamma_tokens: Number of tokens speculated per step.
        draft_latency_ratio: Draft model latency as ratio of target model.
            Defaults to 0.1 (draft is 10x faster).

    Returns:
        Expected speedup factor (>1.0 means faster).

    Raises:
        ValueError: If acceptance_rate is not in [0.0, 1.0].
        ValueError: If gamma_tokens is not positive.
        ValueError: If draft_latency_ratio is not in (0.0, 1.0).

    Examples:
        >>> speedup = calculate_expected_speedup(0.8, 5, 0.1)
        >>> speedup > 1.0
        True
        >>> round(speedup, 2)
        3.33

        >>> calculate_expected_speedup(0.0, 5, 0.1)
        0.67

        >>> calculate_expected_speedup(1.0, 5, 0.1)
        4.0

        >>> calculate_expected_speedup(-0.1, 5, 0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: acceptance_rate must be between 0.0 and 1.0

        >>> calculate_expected_speedup(0.8, 0, 0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma_tokens must be positive

        >>> calculate_expected_speedup(0.8, 5, 1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: draft_latency_ratio must be between 0.0 and 1.0 (exclusive)
    """
    if not 0.0 <= acceptance_rate <= 1.0:
        msg = f"acceptance_rate must be between 0.0 and 1.0, got {acceptance_rate}"
        raise ValueError(msg)

    if gamma_tokens <= 0:
        msg = f"gamma_tokens must be positive, got {gamma_tokens}"
        raise ValueError(msg)

    if not 0.0 < draft_latency_ratio < 1.0:
        msg = (
            f"draft_latency_ratio must be between 0.0 and 1.0 (exclusive), "
            f"got {draft_latency_ratio}"
        )
        raise ValueError(msg)

    # Expected tokens per verification step:
    # 1 (from target) + sum_{i=0}^{gamma-1} acceptance_rate^(i+1)
    # Simplified: 1 + acceptance_rate * (1 - acceptance_rate^gamma) / (1 - ar)
    # For acceptance_rate = 1.0: 1 + gamma
    if acceptance_rate == 1.0:
        expected_tokens = 1.0 + gamma_tokens
    else:
        expected_tokens = 1.0 + acceptance_rate * gamma_tokens

    # Latency per step: 1 target forward + gamma draft forwards
    latency_per_step = 1.0 + gamma_tokens * draft_latency_ratio

    speedup = expected_tokens / latency_per_step
    return round(speedup, 2)


def estimate_acceptance_rate(
    draft_perplexity: float,
    target_perplexity: float,
    temperature: float = 1.0,
) -> float:
    """Estimate acceptance rate based on model perplexities.

    Higher perplexity gap between draft and target models typically
    leads to lower acceptance rates.

    Args:
        draft_perplexity: Perplexity of draft model on test data.
        target_perplexity: Perplexity of target model on test data.
        temperature: Sampling temperature. Defaults to 1.0.

    Returns:
        Estimated acceptance rate (0.0 to 1.0).

    Raises:
        ValueError: If draft_perplexity is not positive.
        ValueError: If target_perplexity is not positive.
        ValueError: If temperature is not positive.

    Examples:
        >>> rate = estimate_acceptance_rate(20.0, 15.0)
        >>> 0.0 <= rate <= 1.0
        True
        >>> round(rate, 2)
        0.75

        >>> estimate_acceptance_rate(15.0, 15.0)
        1.0

        >>> estimate_acceptance_rate(30.0, 10.0)
        0.33

        >>> estimate_acceptance_rate(0, 15.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: draft_perplexity must be positive

        >>> estimate_acceptance_rate(20.0, 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_perplexity must be positive

        >>> estimate_acceptance_rate(20.0, 15.0, temperature=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive
    """
    if draft_perplexity <= 0:
        msg = f"draft_perplexity must be positive, got {draft_perplexity}"
        raise ValueError(msg)

    if target_perplexity <= 0:
        msg = f"target_perplexity must be positive, got {target_perplexity}"
        raise ValueError(msg)

    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    # Acceptance rate approximation based on perplexity ratio
    # Lower draft perplexity (closer to target) = higher acceptance
    ratio = target_perplexity / draft_perplexity

    # Apply temperature scaling
    scaled_ratio = ratio ** (1.0 / temperature)

    # Clamp to valid range
    acceptance = min(1.0, scaled_ratio)
    return round(acceptance, 2)


def calculate_speculation_efficiency(
    accepted_tokens: int,
    rejected_tokens: int,
    gamma_tokens: int,
) -> float:
    """Calculate speculation efficiency metric.

    Efficiency measures how well the draft model predicts target model tokens,
    normalized by the speculation length.

    Args:
        accepted_tokens: Number of tokens accepted.
        rejected_tokens: Number of tokens rejected.
        gamma_tokens: Number of tokens speculated per step.

    Returns:
        Efficiency metric (0.0 to 1.0).

    Raises:
        ValueError: If accepted_tokens is negative.
        ValueError: If rejected_tokens is negative.
        ValueError: If gamma_tokens is not positive.

    Examples:
        >>> calculate_speculation_efficiency(80, 20, 5)
        0.8

        >>> calculate_speculation_efficiency(0, 100, 5)
        0.0

        >>> calculate_speculation_efficiency(100, 0, 5)
        1.0

        >>> calculate_speculation_efficiency(-1, 20, 5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: accepted_tokens cannot be negative

        >>> calculate_speculation_efficiency(80, -1, 5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rejected_tokens cannot be negative

        >>> calculate_speculation_efficiency(80, 20, 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma_tokens must be positive
    """
    if accepted_tokens < 0:
        msg = f"accepted_tokens cannot be negative, got {accepted_tokens}"
        raise ValueError(msg)

    if rejected_tokens < 0:
        msg = f"rejected_tokens cannot be negative, got {rejected_tokens}"
        raise ValueError(msg)

    if gamma_tokens <= 0:
        msg = f"gamma_tokens must be positive, got {gamma_tokens}"
        raise ValueError(msg)

    total = accepted_tokens + rejected_tokens
    if total == 0:
        return 0.0

    return accepted_tokens / total


def format_speculative_stats(stats: SpeculativeStats) -> str:
    """Format speculative decoding statistics for display.

    Args:
        stats: SpeculativeStats to format.

    Returns:
        Formatted string representation of statistics.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = SpeculativeStats(
        ...     accepted_tokens=80,
        ...     rejected_tokens=20,
        ...     speedup_factor=2.5,
        ...     acceptance_rate=0.8,
        ... )
        >>> output = format_speculative_stats(stats)
        >>> "Accepted: 80" in output
        True
        >>> "Rejected: 20" in output
        True
        >>> "Speedup: 2.50x" in output
        True
        >>> "Acceptance Rate: 80.0%" in output
        True

        >>> format_speculative_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    total = stats.accepted_tokens + stats.rejected_tokens
    lines = [
        "Speculative Decoding Statistics:",
        f"  Accepted: {stats.accepted_tokens}",
        f"  Rejected: {stats.rejected_tokens}",
        f"  Total: {total}",
        f"  Speedup: {stats.speedup_factor:.2f}x",
        f"  Acceptance Rate: {stats.acceptance_rate * 100:.1f}%",
    ]
    return "\n".join(lines)


def get_recommended_speculative_config(model_size: str) -> SpeculativeConfig:
    """Get recommended speculative decoding configuration for model size.

    Provides sensible defaults based on the target model size category.
    Larger models benefit more from speculative decoding with more
    aggressive speculation lengths.

    Args:
        model_size: Model size category ("small", "medium", "large", "xlarge").

    Returns:
        Recommended SpeculativeConfig for the model size.

    Raises:
        ValueError: If model_size is invalid.

    Examples:
        >>> config = get_recommended_speculative_config("small")
        >>> config.draft_config.gamma_tokens
        3
        >>> config.verification_config.threshold
        0.95

        >>> config = get_recommended_speculative_config("medium")
        >>> config.draft_config.gamma_tokens
        4
        >>> config.max_speculation_length
        128

        >>> config = get_recommended_speculative_config("large")
        >>> config.draft_config.gamma_tokens
        5
        >>> config.verification_config.threshold
        0.85

        >>> config = get_recommended_speculative_config("xlarge")
        >>> config.draft_config.gamma_tokens
        8
        >>> config.max_speculation_length
        512

        >>> get_recommended_speculative_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size must be one of

        >>> get_recommended_speculative_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size must be one of
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    # Configuration recommendations by model size
    configs = {
        "small": {
            "gamma_tokens": 3,
            "threshold": 0.95,
            "max_speculation_length": 64,
        },
        "medium": {
            "gamma_tokens": 4,
            "threshold": 0.9,
            "max_speculation_length": 128,
        },
        "large": {
            "gamma_tokens": 5,
            "threshold": 0.85,
            "max_speculation_length": 256,
        },
        "xlarge": {
            "gamma_tokens": 8,
            "threshold": 0.8,
            "max_speculation_length": 512,
        },
    }

    params = configs[model_size]
    return create_speculative_config(
        gamma_tokens=int(params["gamma_tokens"]),
        threshold=float(params["threshold"]),
        max_speculation_length=int(params["max_speculation_length"]),
    )
