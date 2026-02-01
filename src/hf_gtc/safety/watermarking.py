"""Watermarking utilities for LLM text generation.

This module provides configuration and utilities for embedding and detecting
watermarks in LLM-generated text, enabling provenance tracking and
AI-generated content identification.

Examples:
    >>> from hf_gtc.safety.watermarking import WatermarkType, WatermarkStrength
    >>> wtype = WatermarkType.SOFT
    >>> wtype.value
    'soft'
    >>> strength = WatermarkStrength.MEDIUM
    >>> strength.value
    'medium'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class WatermarkType(Enum):
    """Types of watermarking strategies.

    Attributes:
        SOFT: Soft watermarking with probabilistic bias.
        HARD: Hard watermarking with strict token constraints.
        SEMANTIC: Semantic-level watermarking preserving meaning.
        STATISTICAL: Statistical watermarking based on token distributions.

    Examples:
        >>> WatermarkType.SOFT.value
        'soft'
        >>> WatermarkType.HARD.value
        'hard'
        >>> WatermarkType.SEMANTIC.value
        'semantic'
        >>> WatermarkType.STATISTICAL.value
        'statistical'
    """

    SOFT = "soft"
    HARD = "hard"
    SEMANTIC = "semantic"
    STATISTICAL = "statistical"


class WatermarkStrength(Enum):
    """Watermark strength levels.

    Higher strength increases detectability but may affect text quality.

    Attributes:
        LOW: Low strength, minimal impact on text quality.
        MEDIUM: Medium strength, balanced trade-off.
        HIGH: High strength, maximum detectability.

    Examples:
        >>> WatermarkStrength.LOW.value
        'low'
        >>> WatermarkStrength.MEDIUM.value
        'medium'
        >>> WatermarkStrength.HIGH.value
        'high'
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DetectionMethod(Enum):
    """Methods for detecting watermarks in text.

    Attributes:
        Z_SCORE: Z-score based detection using green list statistics.
        LOG_LIKELIHOOD: Log-likelihood ratio test detection.
        PERPLEXITY: Perplexity-based anomaly detection.
        ENTROPY: Entropy analysis detection.

    Examples:
        >>> DetectionMethod.Z_SCORE.value
        'z_score'
        >>> DetectionMethod.LOG_LIKELIHOOD.value
        'log_likelihood'
        >>> DetectionMethod.PERPLEXITY.value
        'perplexity'
        >>> DetectionMethod.ENTROPY.value
        'entropy'
    """

    Z_SCORE = "z_score"
    LOG_LIKELIHOOD = "log_likelihood"
    PERPLEXITY = "perplexity"
    ENTROPY = "entropy"


@dataclass(frozen=True, slots=True)
class WatermarkConfig:
    """Configuration for watermark embedding.

    Attributes:
        watermark_type: Type of watermarking strategy.
        strength: Watermark strength level.
        gamma: Fraction of vocabulary in the green list.
        delta: Bias added to green list token logits.
        seeding_scheme: Scheme for seeding the hash function.

    Examples:
        >>> config = WatermarkConfig(
        ...     watermark_type=WatermarkType.SOFT,
        ...     strength=WatermarkStrength.MEDIUM,
        ...     gamma=0.25,
        ...     delta=2.0,
        ... )
        >>> config.gamma
        0.25
        >>> config.delta
        2.0
        >>> config.watermark_type
        <WatermarkType.SOFT: 'soft'>

        >>> config2 = WatermarkConfig()
        >>> config2.seeding_scheme
        'selfhash'
    """

    watermark_type: WatermarkType = WatermarkType.SOFT
    strength: WatermarkStrength = WatermarkStrength.MEDIUM
    gamma: float = 0.25
    delta: float = 2.0
    seeding_scheme: str = "selfhash"


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Configuration for watermark detection.

    Attributes:
        method: Detection method to use.
        threshold: Z-score threshold for positive detection.
        window_size: Size of sliding window for detection.
        ignore_repeated: Whether to ignore repeated tokens.

    Examples:
        >>> config = DetectionConfig(
        ...     method=DetectionMethod.Z_SCORE,
        ...     threshold=4.0,
        ...     window_size=256,
        ... )
        >>> config.threshold
        4.0
        >>> config.window_size
        256

        >>> config2 = DetectionConfig()
        >>> config2.ignore_repeated
        True
    """

    method: DetectionMethod = DetectionMethod.Z_SCORE
    threshold: float = 4.0
    window_size: int = 256
    ignore_repeated: bool = True


@dataclass(frozen=True, slots=True)
class WatermarkResult:
    """Result of watermark detection.

    Attributes:
        is_watermarked: Whether watermark was detected.
        confidence: Confidence score between 0 and 1.
        z_score: Computed z-score for detection.
        p_value: Statistical p-value for the detection.

    Examples:
        >>> result = WatermarkResult(
        ...     is_watermarked=True,
        ...     confidence=0.95,
        ...     z_score=5.2,
        ...     p_value=0.00001,
        ... )
        >>> result.is_watermarked
        True
        >>> result.confidence
        0.95

        >>> result2 = WatermarkResult(
        ...     is_watermarked=False,
        ...     confidence=0.1,
        ...     z_score=1.2,
        ...     p_value=0.23,
        ... )
        >>> result2.is_watermarked
        False
    """

    is_watermarked: bool
    confidence: float
    z_score: float
    p_value: float


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for watermark embedding parameters.

    Attributes:
        vocab_fraction: Fraction of vocabulary used for green list.
        hash_key: Secret key for hash function.
        context_width: Number of previous tokens for context hashing.

    Examples:
        >>> config = EmbeddingConfig(
        ...     vocab_fraction=0.5,
        ...     hash_key=42,
        ...     context_width=1,
        ... )
        >>> config.vocab_fraction
        0.5
        >>> config.hash_key
        42

        >>> config2 = EmbeddingConfig()
        >>> config2.context_width
        1
    """

    vocab_fraction: float = 0.5
    hash_key: int = 15485863
    context_width: int = 1


@dataclass(frozen=True, slots=True)
class WatermarkStats:
    """Statistics from watermark processing.

    Attributes:
        tokens_processed: Total number of tokens processed.
        tokens_watermarked: Number of tokens with watermark applied.
        detection_rate: Proportion of tokens detected as watermarked.

    Examples:
        >>> stats = WatermarkStats(
        ...     tokens_processed=1000,
        ...     tokens_watermarked=750,
        ...     detection_rate=0.75,
        ... )
        >>> stats.tokens_processed
        1000
        >>> stats.tokens_watermarked
        750
        >>> stats.detection_rate
        0.75
    """

    tokens_processed: int
    tokens_watermarked: int
    detection_rate: float


def validate_watermark_config(config: WatermarkConfig) -> None:
    """Validate watermark configuration parameters.

    Args:
        config: WatermarkConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If gamma is not in (0, 1).
        ValueError: If delta is not positive.
        ValueError: If seeding_scheme is empty.

    Examples:
        >>> config = WatermarkConfig(gamma=0.25, delta=2.0)
        >>> validate_watermark_config(config)  # No error

        >>> validate_watermark_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = WatermarkConfig(gamma=1.5)
        >>> validate_watermark_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma must be between 0 and 1

        >>> bad_delta = WatermarkConfig(delta=-1.0)
        >>> validate_watermark_config(bad_delta)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: delta must be positive
    """
    validate_not_none(config, "config")

    if not 0 < config.gamma < 1:
        msg = f"gamma must be between 0 and 1, got {config.gamma}"
        raise ValueError(msg)

    if config.delta <= 0:
        msg = f"delta must be positive, got {config.delta}"
        raise ValueError(msg)

    if not config.seeding_scheme or not config.seeding_scheme.strip():
        msg = "seeding_scheme cannot be empty"
        raise ValueError(msg)


def validate_detection_config(config: DetectionConfig) -> None:
    """Validate detection configuration parameters.

    Args:
        config: DetectionConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If threshold is not positive.
        ValueError: If window_size is not positive.

    Examples:
        >>> config = DetectionConfig(threshold=4.0, window_size=256)
        >>> validate_detection_config(config)  # No error

        >>> validate_detection_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = DetectionConfig(threshold=-1.0)
        >>> validate_detection_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be positive

        >>> bad_window = DetectionConfig(window_size=0)
        >>> validate_detection_config(bad_window)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be positive
    """
    validate_not_none(config, "config")

    if config.threshold <= 0:
        msg = f"threshold must be positive, got {config.threshold}"
        raise ValueError(msg)

    if config.window_size <= 0:
        msg = f"window_size must be positive, got {config.window_size}"
        raise ValueError(msg)


def create_watermark_config(
    watermark_type: WatermarkType | str = WatermarkType.SOFT,
    strength: WatermarkStrength | str = WatermarkStrength.MEDIUM,
    gamma: float = 0.25,
    delta: float = 2.0,
    seeding_scheme: str = "selfhash",
) -> WatermarkConfig:
    """Create a validated watermark configuration.

    Args:
        watermark_type: Type of watermarking strategy.
        strength: Watermark strength level.
        gamma: Fraction of vocabulary in green list.
        delta: Bias added to green list logits.
        seeding_scheme: Scheme for seeding the hash function.

    Returns:
        Validated WatermarkConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_watermark_config(
        ...     watermark_type="soft",
        ...     strength="medium",
        ...     gamma=0.25,
        ... )
        >>> config.watermark_type
        <WatermarkType.SOFT: 'soft'>
        >>> config.strength
        <WatermarkStrength.MEDIUM: 'medium'>

        >>> config2 = create_watermark_config(
        ...     watermark_type=WatermarkType.HARD,
        ...     delta=3.0,
        ... )
        >>> config2.delta
        3.0

        >>> create_watermark_config(gamma=2.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma must be between 0 and 1
    """
    if isinstance(watermark_type, str):
        watermark_type = get_watermark_type(watermark_type)

    if isinstance(strength, str):
        strength = get_watermark_strength(strength)

    config = WatermarkConfig(
        watermark_type=watermark_type,
        strength=strength,
        gamma=gamma,
        delta=delta,
        seeding_scheme=seeding_scheme,
    )
    validate_watermark_config(config)
    return config


def create_detection_config(
    method: DetectionMethod | str = DetectionMethod.Z_SCORE,
    threshold: float = 4.0,
    window_size: int = 256,
    ignore_repeated: bool = True,
) -> DetectionConfig:
    """Create a validated detection configuration.

    Args:
        method: Detection method to use.
        threshold: Z-score threshold for detection.
        window_size: Size of sliding window.
        ignore_repeated: Whether to ignore repeated tokens.

    Returns:
        Validated DetectionConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_detection_config(
        ...     method="z_score",
        ...     threshold=4.0,
        ... )
        >>> config.method
        <DetectionMethod.Z_SCORE: 'z_score'>

        >>> config2 = create_detection_config(
        ...     method=DetectionMethod.LOG_LIKELIHOOD,
        ...     window_size=512,
        ... )
        >>> config2.window_size
        512

        >>> create_detection_config(threshold=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be positive
    """
    if isinstance(method, str):
        method = get_detection_method(method)

    config = DetectionConfig(
        method=method,
        threshold=threshold,
        window_size=window_size,
        ignore_repeated=ignore_repeated,
    )
    validate_detection_config(config)
    return config


def create_embedding_config(
    vocab_fraction: float = 0.5,
    hash_key: int = 15485863,
    context_width: int = 1,
) -> EmbeddingConfig:
    """Create a validated embedding configuration.

    Args:
        vocab_fraction: Fraction of vocabulary for green list.
        hash_key: Secret key for hash function.
        context_width: Number of previous tokens for context.

    Returns:
        Validated EmbeddingConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_embedding_config(vocab_fraction=0.5, hash_key=42)
        >>> config.vocab_fraction
        0.5
        >>> config.hash_key
        42

        >>> config2 = create_embedding_config(context_width=2)
        >>> config2.context_width
        2

        >>> create_embedding_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     vocab_fraction=1.5
        ... )
        Traceback (most recent call last):
        ValueError: vocab_fraction must be between 0 and 1

        >>> create_embedding_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     context_width=0
        ... )
        Traceback (most recent call last):
        ValueError: context_width must be positive
    """
    if not 0 < vocab_fraction < 1:
        msg = f"vocab_fraction must be between 0 and 1, got {vocab_fraction}"
        raise ValueError(msg)

    if context_width <= 0:
        msg = f"context_width must be positive, got {context_width}"
        raise ValueError(msg)

    return EmbeddingConfig(
        vocab_fraction=vocab_fraction,
        hash_key=hash_key,
        context_width=context_width,
    )


def list_watermark_types() -> list[str]:
    """List all available watermark types.

    Returns:
        Sorted list of watermark type names.

    Examples:
        >>> types = list_watermark_types()
        >>> "soft" in types
        True
        >>> "hard" in types
        True
        >>> "semantic" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(t.value for t in WatermarkType)


def list_watermark_strengths() -> list[str]:
    """List all available watermark strength levels.

    Returns:
        Sorted list of strength level names.

    Examples:
        >>> strengths = list_watermark_strengths()
        >>> "low" in strengths
        True
        >>> "medium" in strengths
        True
        >>> "high" in strengths
        True
        >>> strengths == sorted(strengths)
        True
    """
    return sorted(s.value for s in WatermarkStrength)


def list_detection_methods() -> list[str]:
    """List all available detection methods.

    Returns:
        Sorted list of detection method names.

    Examples:
        >>> methods = list_detection_methods()
        >>> "z_score" in methods
        True
        >>> "log_likelihood" in methods
        True
        >>> "perplexity" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(m.value for m in DetectionMethod)


def get_watermark_type(name: str) -> WatermarkType:
    """Get watermark type enum from string.

    Args:
        name: Watermark type name.

    Returns:
        WatermarkType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_watermark_type("soft")
        <WatermarkType.SOFT: 'soft'>
        >>> get_watermark_type("hard")
        <WatermarkType.HARD: 'hard'>
        >>> get_watermark_type("semantic")
        <WatermarkType.SEMANTIC: 'semantic'>

        >>> get_watermark_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid watermark type: invalid
    """
    for wtype in WatermarkType:
        if wtype.value == name:
            return wtype
    msg = f"invalid watermark type: {name}"
    raise ValueError(msg)


def get_watermark_strength(name: str) -> WatermarkStrength:
    """Get watermark strength enum from string.

    Args:
        name: Strength level name.

    Returns:
        WatermarkStrength enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_watermark_strength("low")
        <WatermarkStrength.LOW: 'low'>
        >>> get_watermark_strength("medium")
        <WatermarkStrength.MEDIUM: 'medium'>
        >>> get_watermark_strength("high")
        <WatermarkStrength.HIGH: 'high'>

        >>> get_watermark_strength("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid watermark strength: invalid
    """
    for strength in WatermarkStrength:
        if strength.value == name:
            return strength
    msg = f"invalid watermark strength: {name}"
    raise ValueError(msg)


def get_detection_method(name: str) -> DetectionMethod:
    """Get detection method enum from string.

    Args:
        name: Detection method name.

    Returns:
        DetectionMethod enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_detection_method("z_score")
        <DetectionMethod.Z_SCORE: 'z_score'>
        >>> get_detection_method("log_likelihood")
        <DetectionMethod.LOG_LIKELIHOOD: 'log_likelihood'>
        >>> get_detection_method("perplexity")
        <DetectionMethod.PERPLEXITY: 'perplexity'>

        >>> get_detection_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid detection method: invalid
    """
    for method in DetectionMethod:
        if method.value == name:
            return method
    msg = f"invalid detection method: {name}"
    raise ValueError(msg)


def calculate_z_score(
    green_token_count: int,
    total_tokens: int,
    gamma: float,
) -> float:
    """Calculate z-score for watermark detection.

    The z-score measures how many standard deviations the observed
    green token count is from the expected count under no watermark.

    z = (observed - expected) / std_dev
    where expected = total_tokens * gamma
    and std_dev = sqrt(total_tokens * gamma * (1 - gamma))

    Args:
        green_token_count: Number of tokens from the green list.
        total_tokens: Total number of tokens analyzed.
        gamma: Fraction of vocabulary in the green list.

    Returns:
        Z-score for watermark detection.

    Raises:
        ValueError: If total_tokens is not positive.
        ValueError: If gamma is not in (0, 1).
        ValueError: If green_token_count is negative.
        ValueError: If green_token_count exceeds total_tokens.

    Examples:
        >>> z = calculate_z_score(75, 100, 0.5)
        >>> round(z, 4)
        5.0

        >>> z2 = calculate_z_score(50, 100, 0.5)
        >>> round(z2, 4)
        0.0

        >>> z3 = calculate_z_score(60, 100, 0.5)
        >>> round(z3, 4)
        2.0

        >>> calculate_z_score(10, 0, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_tokens must be positive

        >>> calculate_z_score(10, 100, 1.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma must be between 0 and 1

        >>> calculate_z_score(-1, 100, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: green_token_count cannot be negative

        >>> calculate_z_score(150, 100, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: green_token_count cannot exceed total_tokens
    """
    if total_tokens <= 0:
        msg = f"total_tokens must be positive, got {total_tokens}"
        raise ValueError(msg)

    if not 0 < gamma < 1:
        msg = f"gamma must be between 0 and 1, got {gamma}"
        raise ValueError(msg)

    if green_token_count < 0:
        msg = f"green_token_count cannot be negative, got {green_token_count}"
        raise ValueError(msg)

    if green_token_count > total_tokens:
        msg = (
            f"green_token_count cannot exceed total_tokens, "
            f"got {green_token_count} > {total_tokens}"
        )
        raise ValueError(msg)

    expected = total_tokens * gamma
    variance = total_tokens * gamma * (1 - gamma)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    return (green_token_count - expected) / std_dev


def estimate_detectability(
    num_tokens: int,
    gamma: float,
    delta: float,
    threshold: float = 4.0,
) -> float:
    """Estimate detectability of watermark given parameters.

    Estimates the probability that a watermark will be successfully
    detected given the number of tokens and watermark parameters.

    Args:
        num_tokens: Number of tokens in the text.
        gamma: Fraction of vocabulary in the green list.
        delta: Bias added to green list token logits.
        threshold: Z-score threshold for detection.

    Returns:
        Estimated probability of successful detection (0 to 1).

    Raises:
        ValueError: If num_tokens is not positive.
        ValueError: If gamma is not in (0, 1).
        ValueError: If delta is not positive.
        ValueError: If threshold is not positive.

    Examples:
        >>> prob = estimate_detectability(100, 0.5, 2.0)
        >>> 0 <= prob <= 1
        True

        >>> # More tokens increases detectability
        >>> prob_short = estimate_detectability(50, 0.5, 2.0)
        >>> prob_long = estimate_detectability(500, 0.5, 2.0)
        >>> prob_long >= prob_short
        True

        >>> # Higher delta increases detectability
        >>> prob_low_delta = estimate_detectability(100, 0.5, 1.0)
        >>> prob_high_delta = estimate_detectability(100, 0.5, 4.0)
        >>> prob_high_delta >= prob_low_delta
        True

        >>> estimate_detectability(0, 0.5, 2.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_tokens must be positive

        >>> estimate_detectability(100, 1.5, 2.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gamma must be between 0 and 1
    """
    if num_tokens <= 0:
        msg = f"num_tokens must be positive, got {num_tokens}"
        raise ValueError(msg)

    if not 0 < gamma < 1:
        msg = f"gamma must be between 0 and 1, got {gamma}"
        raise ValueError(msg)

    if delta <= 0:
        msg = f"delta must be positive, got {delta}"
        raise ValueError(msg)

    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise ValueError(msg)

    # Estimate expected green token ratio with watermark
    # Using sigmoid approximation for softmax effect
    expected_green_ratio = gamma + (1 - gamma) * (1 - math.exp(-delta / 2))

    # Expected z-score with watermark
    variance = num_tokens * gamma * (1 - gamma)
    if variance <= 0:
        return 0.0

    std_dev = math.sqrt(variance)
    expected_excess = num_tokens * (expected_green_ratio - gamma)
    expected_z_score = expected_excess / std_dev

    # Probability of exceeding threshold (using normal CDF approximation)
    # P(Z > threshold) where Z ~ N(expected_z_score, 1)
    z_diff = expected_z_score - threshold

    # Simple sigmoid approximation of normal CDF
    prob = 1 / (1 + math.exp(-1.7 * z_diff))

    return min(max(prob, 0.0), 1.0)
