"""Decoding strategies for text generation with HuggingFace models.

This module provides utilities for configuring decoding strategies including
beam search, nucleus sampling, contrastive decoding, and various stopping
criteria for controlled text generation.

Examples:
    >>> from hf_gtc.inference.decoding import (
    ...     DecodingMethod,
    ...     create_decoding_config,
    ...     create_beam_config,
    ... )
    >>> beam = create_beam_config(num_beams=5)
    >>> beam.num_beams
    5
    >>> config = create_decoding_config(method="beam", beam_config=beam)
    >>> config.method
    <DecodingMethod.BEAM: 'beam'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class DecodingMethod(Enum):
    """Decoding method for text generation.

    Attributes:
        GREEDY: Select most likely token at each step.
        BEAM: Beam search for exploring multiple hypotheses.
        SAMPLING: Random sampling from distribution.
        NUCLEUS: Top-p (nucleus) sampling.
        TOP_K: Top-k sampling.
        CONTRASTIVE: Contrastive decoding with amateur model.
        TYPICAL: Typical decoding based on entropy.

    Examples:
        >>> DecodingMethod.GREEDY.value
        'greedy'
        >>> DecodingMethod.BEAM.value
        'beam'
        >>> DecodingMethod.NUCLEUS.value
        'nucleus'
        >>> DecodingMethod.CONTRASTIVE.value
        'contrastive'
    """

    GREEDY = "greedy"
    BEAM = "beam"
    SAMPLING = "sampling"
    NUCLEUS = "nucleus"
    TOP_K = "top_k"
    CONTRASTIVE = "contrastive"
    TYPICAL = "typical"


VALID_DECODING_METHODS = frozenset(m.value for m in DecodingMethod)


class StoppingCriteria(Enum):
    """Stopping criteria for generation.

    Attributes:
        MAX_LENGTH: Stop at maximum length.
        EOS_TOKEN: Stop at end-of-sequence token.
        MAX_TIME: Stop at maximum time limit.

    Examples:
        >>> StoppingCriteria.MAX_LENGTH.value
        'max_length'
        >>> StoppingCriteria.EOS_TOKEN.value
        'eos_token'
        >>> StoppingCriteria.MAX_TIME.value
        'max_time'
    """

    MAX_LENGTH = "max_length"
    EOS_TOKEN = "eos_token"
    MAX_TIME = "max_time"


VALID_STOPPING_CRITERIA = frozenset(c.value for c in StoppingCriteria)


class RepetitionPenaltyType(Enum):
    """Type of repetition penalty to apply.

    Attributes:
        MULTIPLICATIVE: Divide logits by penalty factor.
        ADDITIVE: Subtract penalty from logits.
        PRESENCE: Apply constant penalty for presence.

    Examples:
        >>> RepetitionPenaltyType.MULTIPLICATIVE.value
        'multiplicative'
        >>> RepetitionPenaltyType.ADDITIVE.value
        'additive'
        >>> RepetitionPenaltyType.PRESENCE.value
        'presence'
    """

    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    PRESENCE = "presence"


VALID_REPETITION_PENALTY_TYPES = frozenset(t.value for t in RepetitionPenaltyType)


# Type aliases for string literal types
DecodingMethodStr = Literal[
    "greedy", "beam", "sampling", "nucleus", "top_k", "contrastive", "typical"
]
StoppingCriteriaStr = Literal["max_length", "eos_token", "max_time"]
RepetitionPenaltyTypeStr = Literal["multiplicative", "additive", "presence"]


@dataclass(frozen=True, slots=True)
class BeamConfig:
    """Configuration for beam search decoding.

    Attributes:
        num_beams: Number of beams for beam search. Defaults to 4.
        length_penalty: Exponential penalty for sequence length. Defaults to 1.0.
        early_stopping: Whether to stop when all beams finish. Defaults to False.
        num_return_sequences: Number of sequences to return. Defaults to 1.

    Examples:
        >>> config = BeamConfig(
        ...     num_beams=5,
        ...     length_penalty=0.8,
        ...     early_stopping=True,
        ...     num_return_sequences=3,
        ... )
        >>> config.num_beams
        5
        >>> config.length_penalty
        0.8

        >>> config = BeamConfig()
        >>> config.num_beams
        4
        >>> config.early_stopping
        False
    """

    num_beams: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = False
    num_return_sequences: int = 1


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Configuration for sampling-based decoding.

    Attributes:
        temperature: Sampling temperature (higher = more random). Defaults to 1.0.
        top_k: Number of highest probability tokens to keep. Defaults to 50.
        top_p: Cumulative probability for nucleus sampling. Defaults to 1.0.
        typical_p: Typical probability mass for typical decoding. Defaults to 1.0.

    Examples:
        >>> config = SamplingConfig(
        ...     temperature=0.7,
        ...     top_k=40,
        ...     top_p=0.9,
        ...     typical_p=0.95,
        ... )
        >>> config.temperature
        0.7
        >>> config.top_p
        0.9

        >>> config = SamplingConfig()
        >>> config.temperature
        1.0
        >>> config.top_k
        50
    """

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    typical_p: float = 1.0


@dataclass(frozen=True, slots=True)
class ContrastiveConfig:
    """Configuration for contrastive decoding.

    Attributes:
        penalty_alpha: Weight for contrastive penalty. Defaults to 0.6.
        top_k: Number of candidates to consider. Defaults to 4.
        amateur_model: Model ID for amateur/weaker model. Defaults to empty string.

    Examples:
        >>> config = ContrastiveConfig(
        ...     penalty_alpha=0.5,
        ...     top_k=6,
        ...     amateur_model="gpt2-small",
        ... )
        >>> config.penalty_alpha
        0.5
        >>> config.amateur_model
        'gpt2-small'

        >>> config = ContrastiveConfig()
        >>> config.penalty_alpha
        0.6
        >>> config.top_k
        4
    """

    penalty_alpha: float = 0.6
    top_k: int = 4
    amateur_model: str = ""


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    """Complete configuration for decoding.

    Attributes:
        method: Decoding method to use. Defaults to GREEDY.
        beam_config: Configuration for beam search. Defaults to None.
        sampling_config: Configuration for sampling. Defaults to None.
        contrastive_config: Configuration for contrastive decoding. Defaults to None.
        max_length: Maximum generation length. Defaults to 128.
        repetition_penalty: Penalty factor for repetition. Defaults to 1.0.
        repetition_penalty_type: Type of repetition penalty. Defaults to MULTIPLICATIVE.

    Examples:
        >>> config = DecodingConfig(
        ...     method=DecodingMethod.BEAM,
        ...     beam_config=BeamConfig(num_beams=5),
        ...     max_length=256,
        ...     repetition_penalty=1.2,
        ... )
        >>> config.method
        <DecodingMethod.BEAM: 'beam'>
        >>> config.max_length
        256

        >>> config = DecodingConfig()
        >>> config.method
        <DecodingMethod.GREEDY: 'greedy'>
        >>> config.max_length
        128
    """

    method: DecodingMethod = DecodingMethod.GREEDY
    beam_config: BeamConfig | None = None
    sampling_config: SamplingConfig | None = None
    contrastive_config: ContrastiveConfig | None = None
    max_length: int = 128
    repetition_penalty: float = 1.0
    repetition_penalty_type: RepetitionPenaltyType = (
        RepetitionPenaltyType.MULTIPLICATIVE
    )


@dataclass(frozen=True, slots=True)
class DecodingStats:
    """Statistics from decoding inference.

    Attributes:
        tokens_generated: Number of tokens generated.
        time_per_token_ms: Average time per token in milliseconds.
        tokens_per_second: Throughput in tokens per second.

    Examples:
        >>> stats = DecodingStats(
        ...     tokens_generated=100,
        ...     time_per_token_ms=25.5,
        ...     tokens_per_second=39.2,
        ... )
        >>> stats.tokens_generated
        100
        >>> stats.time_per_token_ms
        25.5
        >>> stats.tokens_per_second
        39.2
    """

    tokens_generated: int
    time_per_token_ms: float
    tokens_per_second: float


def validate_beam_config(config: BeamConfig) -> None:
    """Validate beam search configuration.

    Args:
        config: BeamConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_beams is not positive.
        ValueError: If length_penalty is negative.
        ValueError: If num_return_sequences exceeds num_beams.

    Examples:
        >>> config = BeamConfig(num_beams=5, num_return_sequences=3)
        >>> validate_beam_config(config)  # No error

        >>> validate_beam_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = BeamConfig(num_beams=0)
        >>> validate_beam_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_beams must be positive

        >>> bad_config = BeamConfig(num_beams=3, num_return_sequences=5)
        >>> validate_beam_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_return_sequences cannot exceed num_beams
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_beams <= 0:
        msg = f"num_beams must be positive, got {config.num_beams}"
        raise ValueError(msg)

    if config.length_penalty < 0:
        msg = f"length_penalty cannot be negative, got {config.length_penalty}"
        raise ValueError(msg)

    if config.num_return_sequences <= 0:
        msg = (
            f"num_return_sequences must be positive, got {config.num_return_sequences}"
        )
        raise ValueError(msg)

    if config.num_return_sequences > config.num_beams:
        msg = (
            f"num_return_sequences cannot exceed num_beams, "
            f"got {config.num_return_sequences} > {config.num_beams}"
        )
        raise ValueError(msg)


def validate_sampling_config(config: SamplingConfig) -> None:
    """Validate sampling configuration.

    Args:
        config: SamplingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If temperature is not positive.
        ValueError: If top_k is negative.
        ValueError: If top_p is not in (0.0, 1.0].
        ValueError: If typical_p is not in (0.0, 1.0].

    Examples:
        >>> config = SamplingConfig(temperature=0.7, top_p=0.9)
        >>> validate_sampling_config(config)  # No error

        >>> validate_sampling_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = SamplingConfig(temperature=0)
        >>> validate_sampling_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive

        >>> bad_config = SamplingConfig(top_p=0)
        >>> validate_sampling_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_p must be in (0.0, 1.0]
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)

    if config.top_k < 0:
        msg = f"top_k cannot be negative, got {config.top_k}"
        raise ValueError(msg)

    if not (0.0 < config.top_p <= 1.0):
        msg = f"top_p must be in (0.0, 1.0], got {config.top_p}"
        raise ValueError(msg)

    if not (0.0 < config.typical_p <= 1.0):
        msg = f"typical_p must be in (0.0, 1.0], got {config.typical_p}"
        raise ValueError(msg)


def validate_contrastive_config(config: ContrastiveConfig) -> None:
    """Validate contrastive decoding configuration.

    Args:
        config: ContrastiveConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If penalty_alpha is not in [0.0, 1.0].
        ValueError: If top_k is not positive.

    Examples:
        >>> config = ContrastiveConfig(penalty_alpha=0.5, top_k=6)
        >>> validate_contrastive_config(config)  # No error

        >>> validate_contrastive_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = ContrastiveConfig(penalty_alpha=-0.1)
        >>> validate_contrastive_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: penalty_alpha must be in [0.0, 1.0]

        >>> bad_config = ContrastiveConfig(top_k=0)
        >>> validate_contrastive_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not (0.0 <= config.penalty_alpha <= 1.0):
        msg = f"penalty_alpha must be in [0.0, 1.0], got {config.penalty_alpha}"
        raise ValueError(msg)

    if config.top_k <= 0:
        msg = f"top_k must be positive, got {config.top_k}"
        raise ValueError(msg)


def validate_decoding_config(config: DecodingConfig) -> None:
    """Validate complete decoding configuration.

    Args:
        config: DecodingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_length is not positive.
        ValueError: If repetition_penalty is not positive.
        ValueError: If beam decoding is used without beam_config.
        ValueError: If sampling decoding is used without sampling_config.
        ValueError: If contrastive decoding is used without contrastive_config.

    Examples:
        >>> config = DecodingConfig(max_length=256)
        >>> validate_decoding_config(config)  # No error

        >>> validate_decoding_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = DecodingConfig(max_length=0)
        >>> validate_decoding_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive

        >>> bad_config = DecodingConfig(method=DecodingMethod.BEAM)
        >>> validate_decoding_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beam_config required for beam decoding
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.repetition_penalty <= 0:
        msg = f"repetition_penalty must be positive, got {config.repetition_penalty}"
        raise ValueError(msg)

    # Validate method-specific configs
    if config.method == DecodingMethod.BEAM:
        if config.beam_config is None:
            msg = "beam_config required for beam decoding"
            raise ValueError(msg)
        validate_beam_config(config.beam_config)

    if config.method in (
        DecodingMethod.SAMPLING,
        DecodingMethod.NUCLEUS,
        DecodingMethod.TOP_K,
        DecodingMethod.TYPICAL,
    ):
        if config.sampling_config is None:
            msg = "sampling_config required for sampling-based decoding"
            raise ValueError(msg)
        validate_sampling_config(config.sampling_config)

    if config.method == DecodingMethod.CONTRASTIVE:
        if config.contrastive_config is None:
            msg = "contrastive_config required for contrastive decoding"
            raise ValueError(msg)
        validate_contrastive_config(config.contrastive_config)


def validate_decoding_stats(stats: DecodingStats) -> None:
    """Validate decoding statistics.

    Args:
        stats: DecodingStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If tokens_generated is negative.
        ValueError: If time_per_token_ms is negative.
        ValueError: If tokens_per_second is negative.

    Examples:
        >>> stats = DecodingStats(100, 25.5, 39.2)
        >>> validate_decoding_stats(stats)  # No error

        >>> validate_decoding_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad_stats = DecodingStats(-1, 25.5, 39.2)
        >>> validate_decoding_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens_generated cannot be negative
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.tokens_generated < 0:
        msg = f"tokens_generated cannot be negative, got {stats.tokens_generated}"
        raise ValueError(msg)

    if stats.time_per_token_ms < 0:
        msg = f"time_per_token_ms cannot be negative, got {stats.time_per_token_ms}"
        raise ValueError(msg)

    if stats.tokens_per_second < 0:
        msg = f"tokens_per_second cannot be negative, got {stats.tokens_per_second}"
        raise ValueError(msg)


def create_beam_config(
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    num_return_sequences: int = 1,
) -> BeamConfig:
    """Create a beam search configuration.

    Args:
        num_beams: Number of beams. Defaults to 4.
        length_penalty: Length penalty exponent. Defaults to 1.0.
        early_stopping: Stop when all beams finish. Defaults to False.
        num_return_sequences: Number of sequences to return. Defaults to 1.

    Returns:
        BeamConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_beam_config(num_beams=5)
        >>> config.num_beams
        5

        >>> config = create_beam_config(
        ...     num_beams=8,
        ...     length_penalty=0.8,
        ...     early_stopping=True,
        ...     num_return_sequences=4,
        ... )
        >>> config.length_penalty
        0.8
        >>> config.num_return_sequences
        4

        >>> create_beam_config(num_beams=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_beams must be positive

        >>> create_beam_config(num_beams=3, num_return_sequences=5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_return_sequences cannot exceed num_beams
    """
    config = BeamConfig(
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        num_return_sequences=num_return_sequences,
    )
    validate_beam_config(config)
    return config


def create_sampling_config(
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    typical_p: float = 1.0,
) -> SamplingConfig:
    """Create a sampling configuration.

    Args:
        temperature: Sampling temperature. Defaults to 1.0.
        top_k: Number of top tokens to consider. Defaults to 50.
        top_p: Cumulative probability threshold. Defaults to 1.0.
        typical_p: Typical probability mass. Defaults to 1.0.

    Returns:
        SamplingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_sampling_config(temperature=0.7)
        >>> config.temperature
        0.7

        >>> config = create_sampling_config(
        ...     temperature=0.8,
        ...     top_k=40,
        ...     top_p=0.9,
        ...     typical_p=0.95,
        ... )
        >>> config.top_p
        0.9

        >>> create_sampling_config(temperature=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive

        >>> create_sampling_config(top_p=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_p must be in (0.0, 1.0]
    """
    config = SamplingConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        typical_p=typical_p,
    )
    validate_sampling_config(config)
    return config


def create_contrastive_config(
    penalty_alpha: float = 0.6,
    top_k: int = 4,
    amateur_model: str = "",
) -> ContrastiveConfig:
    """Create a contrastive decoding configuration.

    Args:
        penalty_alpha: Weight for contrastive penalty. Defaults to 0.6.
        top_k: Number of candidates. Defaults to 4.
        amateur_model: Model ID for amateur model. Defaults to empty string.

    Returns:
        ContrastiveConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_contrastive_config(penalty_alpha=0.5)
        >>> config.penalty_alpha
        0.5

        >>> config = create_contrastive_config(
        ...     penalty_alpha=0.7,
        ...     top_k=6,
        ...     amateur_model="gpt2-small",
        ... )
        >>> config.amateur_model
        'gpt2-small'

        >>> create_contrastive_config(penalty_alpha=-0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: penalty_alpha must be in [0.0, 1.0]

        >>> create_contrastive_config(top_k=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    config = ContrastiveConfig(
        penalty_alpha=penalty_alpha,
        top_k=top_k,
        amateur_model=amateur_model,
    )
    validate_contrastive_config(config)
    return config


def create_decoding_config(
    method: DecodingMethodStr = "greedy",
    beam_config: BeamConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    contrastive_config: ContrastiveConfig | None = None,
    max_length: int = 128,
    repetition_penalty: float = 1.0,
    repetition_penalty_type: RepetitionPenaltyTypeStr = "multiplicative",
) -> DecodingConfig:
    """Create a complete decoding configuration.

    Args:
        method: Decoding method. Defaults to "greedy".
        beam_config: Beam search config. Defaults to None.
        sampling_config: Sampling config. Defaults to None.
        contrastive_config: Contrastive decoding config. Defaults to None.
        max_length: Maximum generation length. Defaults to 128.
        repetition_penalty: Repetition penalty factor. Defaults to 1.0.
        repetition_penalty_type: Type of penalty. Defaults to "multiplicative".

    Returns:
        DecodingConfig with the specified settings.

    Raises:
        ValueError: If method is invalid.
        ValueError: If repetition_penalty_type is invalid.
        ValueError: If method-specific config is missing.

    Examples:
        >>> config = create_decoding_config(method="greedy")
        >>> config.method
        <DecodingMethod.GREEDY: 'greedy'>

        >>> beam = create_beam_config(num_beams=5)
        >>> config = create_decoding_config(method="beam", beam_config=beam)
        >>> config.beam_config.num_beams
        5

        >>> create_decoding_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of

        >>> create_decoding_config(method="beam")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: beam_config required for beam decoding
    """
    if method not in VALID_DECODING_METHODS:
        msg = f"method must be one of {VALID_DECODING_METHODS}, got '{method}'"
        raise ValueError(msg)

    if repetition_penalty_type not in VALID_REPETITION_PENALTY_TYPES:
        msg = (
            f"repetition_penalty_type must be one of {VALID_REPETITION_PENALTY_TYPES}, "
            f"got '{repetition_penalty_type}'"
        )
        raise ValueError(msg)

    config = DecodingConfig(
        method=DecodingMethod(method),
        beam_config=beam_config,
        sampling_config=sampling_config,
        contrastive_config=contrastive_config,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        repetition_penalty_type=RepetitionPenaltyType(repetition_penalty_type),
    )
    validate_decoding_config(config)
    return config


def list_decoding_methods() -> list[str]:
    """List all available decoding methods.

    Returns:
        Sorted list of decoding method names.

    Examples:
        >>> methods = list_decoding_methods()
        >>> "greedy" in methods
        True
        >>> "beam" in methods
        True
        >>> "nucleus" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_DECODING_METHODS)


def list_stopping_criteria() -> list[str]:
    """List all available stopping criteria.

    Returns:
        Sorted list of stopping criteria names.

    Examples:
        >>> criteria = list_stopping_criteria()
        >>> "max_length" in criteria
        True
        >>> "eos_token" in criteria
        True
        >>> "max_time" in criteria
        True
        >>> criteria == sorted(criteria)
        True
    """
    return sorted(VALID_STOPPING_CRITERIA)


def list_repetition_penalty_types() -> list[str]:
    """List all available repetition penalty types.

    Returns:
        Sorted list of repetition penalty type names.

    Examples:
        >>> types = list_repetition_penalty_types()
        >>> "multiplicative" in types
        True
        >>> "additive" in types
        True
        >>> "presence" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_REPETITION_PENALTY_TYPES)


def get_decoding_method(name: str) -> DecodingMethod:
    """Get a decoding method by name.

    Args:
        name: Name of the decoding method.

    Returns:
        The corresponding DecodingMethod enum value.

    Raises:
        ValueError: If name is not a valid decoding method.

    Examples:
        >>> get_decoding_method("greedy")
        <DecodingMethod.GREEDY: 'greedy'>
        >>> get_decoding_method("beam")
        <DecodingMethod.BEAM: 'beam'>
        >>> get_decoding_method("nucleus")
        <DecodingMethod.NUCLEUS: 'nucleus'>

        >>> get_decoding_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown decoding method: 'invalid'

        >>> get_decoding_method("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown decoding method: ''
    """
    if name not in VALID_DECODING_METHODS:
        valid = sorted(VALID_DECODING_METHODS)
        msg = f"Unknown decoding method: '{name}'. Valid methods: {valid}"
        raise ValueError(msg)

    return DecodingMethod(name)


def get_stopping_criteria(name: str) -> StoppingCriteria:
    """Get a stopping criteria by name.

    Args:
        name: Name of the stopping criteria.

    Returns:
        The corresponding StoppingCriteria enum value.

    Raises:
        ValueError: If name is not a valid stopping criteria.

    Examples:
        >>> get_stopping_criteria("max_length")
        <StoppingCriteria.MAX_LENGTH: 'max_length'>
        >>> get_stopping_criteria("eos_token")
        <StoppingCriteria.EOS_TOKEN: 'eos_token'>
        >>> get_stopping_criteria("max_time")
        <StoppingCriteria.MAX_TIME: 'max_time'>

        >>> get_stopping_criteria("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown stopping criteria: 'invalid'

        >>> get_stopping_criteria("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown stopping criteria: ''
    """
    if name not in VALID_STOPPING_CRITERIA:
        valid = sorted(VALID_STOPPING_CRITERIA)
        msg = f"Unknown stopping criteria: '{name}'. Valid criteria: {valid}"
        raise ValueError(msg)

    return StoppingCriteria(name)


def get_repetition_penalty_type(name: str) -> RepetitionPenaltyType:
    """Get a repetition penalty type by name.

    Args:
        name: Name of the repetition penalty type.

    Returns:
        The corresponding RepetitionPenaltyType enum value.

    Raises:
        ValueError: If name is not a valid repetition penalty type.

    Examples:
        >>> get_repetition_penalty_type("multiplicative")
        <RepetitionPenaltyType.MULTIPLICATIVE: 'multiplicative'>
        >>> get_repetition_penalty_type("additive")
        <RepetitionPenaltyType.ADDITIVE: 'additive'>
        >>> get_repetition_penalty_type("presence")
        <RepetitionPenaltyType.PRESENCE: 'presence'>

        >>> get_repetition_penalty_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown repetition penalty type: 'invalid'

        >>> get_repetition_penalty_type("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown repetition penalty type: ''
    """
    if name not in VALID_REPETITION_PENALTY_TYPES:
        valid = sorted(VALID_REPETITION_PENALTY_TYPES)
        msg = f"Unknown repetition penalty type: '{name}'. Valid types: {valid}"
        raise ValueError(msg)

    return RepetitionPenaltyType(name)


def calculate_beam_memory(
    num_beams: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dtype_bytes: int = 2,
) -> float:
    """Calculate memory required for beam search in gigabytes.

    Beam search requires maintaining multiple hypotheses which increases
    memory proportionally to the number of beams.

    Args:
        num_beams: Number of beams.
        batch_size: Batch size.
        sequence_length: Maximum sequence length.
        hidden_size: Model hidden dimension.
        num_layers: Number of transformer layers.
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32). Defaults to 2.

    Returns:
        Memory requirement in gigabytes.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> mem = calculate_beam_memory(
        ...     num_beams=4,
        ...     batch_size=1,
        ...     sequence_length=512,
        ...     hidden_size=4096,
        ...     num_layers=32,
        ...     dtype_bytes=2,
        ... )
        >>> mem > 0
        True
        >>> round(mem, 2)
        1.0

        >>> calculate_beam_memory(0, 1, 512, 4096, 32)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_beams must be positive

        >>> calculate_beam_memory(4, 0, 512, 4096, 32)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if num_beams <= 0:
        msg = f"num_beams must be positive, got {num_beams}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    # KV cache for all beams: 2 * layers * batch * beams * seq * hidden * dtype
    total_bytes = (
        2
        * num_layers
        * batch_size
        * num_beams
        * sequence_length
        * hidden_size
        * dtype_bytes
    )

    # Convert to gigabytes
    return total_bytes / (1024**3)


def estimate_generation_time(
    num_tokens: int,
    time_per_token_ms: float,
    num_beams: int = 1,
) -> float:
    """Estimate generation time in seconds.

    Args:
        num_tokens: Number of tokens to generate.
        time_per_token_ms: Time per token in milliseconds.
        num_beams: Number of beams (affects latency). Defaults to 1.

    Returns:
        Estimated generation time in seconds.

    Raises:
        ValueError: If num_tokens is not positive.
        ValueError: If time_per_token_ms is not positive.
        ValueError: If num_beams is not positive.

    Examples:
        >>> time = estimate_generation_time(100, 25.0)
        >>> time
        2.5

        >>> time = estimate_generation_time(100, 25.0, num_beams=4)
        >>> round(time, 2)
        6.25

        >>> estimate_generation_time(0, 25.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_tokens must be positive

        >>> estimate_generation_time(100, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: time_per_token_ms must be positive
    """
    if num_tokens <= 0:
        msg = f"num_tokens must be positive, got {num_tokens}"
        raise ValueError(msg)

    if time_per_token_ms <= 0:
        msg = f"time_per_token_ms must be positive, got {time_per_token_ms}"
        raise ValueError(msg)

    if num_beams <= 0:
        msg = f"num_beams must be positive, got {num_beams}"
        raise ValueError(msg)

    # Beam search overhead approximately doubles time
    beam_overhead = 1.0 + (num_beams - 1) * 0.5

    total_time_ms = num_tokens * time_per_token_ms * beam_overhead
    return total_time_ms / 1000.0


def apply_repetition_penalty(
    logits: list[float],
    generated_token_ids: list[int],
    penalty: float = 1.2,
    penalty_type: RepetitionPenaltyTypeStr = "multiplicative",
) -> list[float]:
    """Apply repetition penalty to logits.

    Args:
        logits: Token logits as a list of floats.
        generated_token_ids: Previously generated token IDs.
        penalty: Penalty factor. Defaults to 1.2.
        penalty_type: Type of penalty to apply. Defaults to "multiplicative".

    Returns:
        Penalized logits.

    Raises:
        ValueError: If penalty is not positive for multiplicative.
        ValueError: If penalty_type is invalid.

    Examples:
        >>> logits = [1.0, 2.0, 3.0, 4.0]
        >>> generated = [2]  # Token 2 was generated
        >>> result = apply_repetition_penalty(logits, generated, 1.2)
        >>> round(result[2], 2)
        2.5
        >>> result[0]
        1.0

        >>> result = apply_repetition_penalty(logits, generated, 0.5, "additive")
        >>> round(result[2], 2)
        2.5
        >>> result[0]
        1.0

        >>> apply_repetition_penalty(logits, generated, 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: penalty must be positive for multiplicative penalty

        >>> apply_repetition_penalty(logits, generated, 1.2, "invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: penalty_type must be one of
    """
    if penalty_type not in VALID_REPETITION_PENALTY_TYPES:
        msg = (
            f"penalty_type must be one of {VALID_REPETITION_PENALTY_TYPES}, "
            f"got '{penalty_type}'"
        )
        raise ValueError(msg)

    if penalty_type == "multiplicative" and penalty <= 0:
        msg = f"penalty must be positive for multiplicative penalty, got {penalty}"
        raise ValueError(msg)

    # Copy logits to avoid modifying input
    result = list(logits)
    generated_set = set(generated_token_ids)

    for token_id in generated_set:
        if 0 <= token_id < len(result):
            if penalty_type == "multiplicative":
                # Divide positive logits, multiply negative logits
                if result[token_id] > 0:
                    result[token_id] /= penalty
                else:
                    result[token_id] *= penalty
            elif penalty_type == "additive":
                result[token_id] -= penalty
            elif penalty_type == "presence":
                # Constant penalty regardless of frequency
                result[token_id] -= penalty

    return result


def calculate_entropy(probabilities: list[float]) -> float:
    """Calculate entropy of a probability distribution.

    Args:
        probabilities: List of probabilities (must sum to 1.0).

    Returns:
        Entropy in nats (natural units).

    Raises:
        ValueError: If probabilities is empty.
        ValueError: If any probability is negative.

    Examples:
        >>> import math
        >>> probs = [0.5, 0.5]
        >>> entropy = calculate_entropy(probs)
        >>> round(entropy, 4)
        0.6931

        >>> probs = [1.0]
        >>> calculate_entropy(probs)
        0.0

        >>> probs = [0.25, 0.25, 0.25, 0.25]
        >>> entropy = calculate_entropy(probs)
        >>> round(entropy, 4)
        1.3863

        >>> calculate_entropy([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probabilities cannot be empty

        >>> calculate_entropy([-0.5, 1.5])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probabilities cannot be negative
    """
    if not probabilities:
        msg = "probabilities cannot be empty"
        raise ValueError(msg)

    if any(p < 0 for p in probabilities):
        msg = "probabilities cannot be negative"
        raise ValueError(msg)

    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


def format_decoding_stats(stats: DecodingStats) -> str:
    """Format decoding statistics for display.

    Args:
        stats: DecodingStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = DecodingStats(
        ...     tokens_generated=100,
        ...     time_per_token_ms=25.5,
        ...     tokens_per_second=39.2,
        ... )
        >>> output = format_decoding_stats(stats)
        >>> "Tokens Generated: 100" in output
        True
        >>> "Time per Token: 25.50 ms" in output
        True
        >>> "Throughput: 39.20 tokens/sec" in output
        True

        >>> format_decoding_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        "Decoding Statistics:",
        f"  Tokens Generated: {stats.tokens_generated}",
        f"  Time per Token: {stats.time_per_token_ms:.2f} ms",
        f"  Throughput: {stats.tokens_per_second:.2f} tokens/sec",
    ]
    return "\n".join(lines)


def get_recommended_decoding_config(
    task: str,
    quality_priority: bool = True,
) -> DecodingConfig:
    """Get recommended decoding configuration for a task.

    Provides sensible defaults based on the generation task type.

    Args:
        task: Task type ("creative", "translation", "qa", "code", "chat").
        quality_priority: Prioritize quality over speed. Defaults to True.

    Returns:
        Recommended DecodingConfig for the task.

    Raises:
        ValueError: If task is invalid.

    Examples:
        >>> config = get_recommended_decoding_config("creative")
        >>> config.method
        <DecodingMethod.NUCLEUS: 'nucleus'>
        >>> config.sampling_config.temperature
        0.9

        >>> config = get_recommended_decoding_config("translation")
        >>> config.method
        <DecodingMethod.BEAM: 'beam'>
        >>> config.beam_config.num_beams
        5

        >>> config = get_recommended_decoding_config("qa", quality_priority=False)
        >>> config.method
        <DecodingMethod.GREEDY: 'greedy'>

        >>> config = get_recommended_decoding_config("code")
        >>> config.method
        <DecodingMethod.SAMPLING: 'sampling'>
        >>> config.sampling_config.temperature
        0.2

        >>> config = get_recommended_decoding_config("chat")
        >>> config.method
        <DecodingMethod.NUCLEUS: 'nucleus'>

        >>> get_recommended_decoding_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of

        >>> get_recommended_decoding_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of
    """
    valid_tasks = {"creative", "translation", "qa", "code", "chat"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    if task == "creative":
        # Creative writing: high temperature nucleus sampling
        return create_decoding_config(
            method="nucleus",
            sampling_config=create_sampling_config(
                temperature=0.9,
                top_p=0.95,
                top_k=0,
            ),
            max_length=512,
            repetition_penalty=1.1,
        )

    if task == "translation":
        if quality_priority:
            # Translation: beam search for quality
            return create_decoding_config(
                method="beam",
                beam_config=create_beam_config(
                    num_beams=5,
                    length_penalty=0.6,
                    early_stopping=True,
                ),
                max_length=256,
                repetition_penalty=1.0,
            )
        # Fast translation: greedy
        return create_decoding_config(
            method="greedy",
            max_length=256,
            repetition_penalty=1.0,
        )

    if task == "qa":
        if quality_priority:
            # QA: beam search with length penalty
            return create_decoding_config(
                method="beam",
                beam_config=create_beam_config(
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                ),
                max_length=128,
                repetition_penalty=1.0,
            )
        # Fast QA: greedy
        return create_decoding_config(
            method="greedy",
            max_length=128,
            repetition_penalty=1.0,
        )

    if task == "code":
        # Code generation: low temperature sampling
        return create_decoding_config(
            method="sampling",
            sampling_config=create_sampling_config(
                temperature=0.2,
                top_p=0.95,
                top_k=0,
            ),
            max_length=1024,
            repetition_penalty=1.05,
        )

    # chat task
    return create_decoding_config(
        method="nucleus",
        sampling_config=create_sampling_config(
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        ),
        max_length=256,
        repetition_penalty=1.1,
    )
