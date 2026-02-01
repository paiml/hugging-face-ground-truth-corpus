"""Data collators and padding strategies for training.

This module provides utilities for batch collation during training, including
padding strategies, truncation handling, and memory-efficient batch processing.
It supports various collator types for different training tasks.

Examples:
    >>> from hf_gtc.training.collators import (
    ...     create_padding_config,
    ...     create_collator_config,
    ...     PaddingStrategy,
    ... )
    >>> padding_config = create_padding_config()
    >>> padding_config.strategy
    <PaddingStrategy.LONGEST: 'longest'>
    >>> collator_config = create_collator_config()
    >>> collator_config.collator_type
    <CollatorType.DEFAULT: 'default'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class PaddingStrategy(Enum):
    """Strategies for padding sequences in a batch.

    Attributes:
        LONGEST: Pad to the longest sequence in the batch.
        MAX_LENGTH: Pad to a fixed maximum length.
        DO_NOT_PAD: Do not apply padding.

    Examples:
        >>> PaddingStrategy.LONGEST.value
        'longest'
        >>> PaddingStrategy.MAX_LENGTH.value
        'max_length'
        >>> PaddingStrategy.DO_NOT_PAD.value
        'do_not_pad'
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(Enum):
    """Strategies for truncating sequences.

    Attributes:
        LONGEST_FIRST: Truncate the longest sequence first.
        ONLY_FIRST: Truncate only the first sequence.
        ONLY_SECOND: Truncate only the second sequence.
        DO_NOT_TRUNCATE: Do not apply truncation.

    Examples:
        >>> TruncationStrategy.LONGEST_FIRST.value
        'longest_first'
        >>> TruncationStrategy.ONLY_FIRST.value
        'only_first'
        >>> TruncationStrategy.ONLY_SECOND.value
        'only_second'
        >>> TruncationStrategy.DO_NOT_TRUNCATE.value
        'do_not_truncate'
    """

    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    DO_NOT_TRUNCATE = "do_not_truncate"


class CollatorType(Enum):
    """Types of data collators for different training tasks.

    Attributes:
        DEFAULT: Default collator for general classification/regression.
        LANGUAGE_MODELING: Collator for causal/masked language modeling.
        SEQ2SEQ: Collator for sequence-to-sequence tasks.
        COMPLETION_ONLY: Collator for completion-only fine-tuning.

    Examples:
        >>> CollatorType.DEFAULT.value
        'default'
        >>> CollatorType.LANGUAGE_MODELING.value
        'language_modeling'
        >>> CollatorType.SEQ2SEQ.value
        'seq2seq'
        >>> CollatorType.COMPLETION_ONLY.value
        'completion_only'
    """

    DEFAULT = "default"
    LANGUAGE_MODELING = "language_modeling"
    SEQ2SEQ = "seq2seq"
    COMPLETION_ONLY = "completion_only"


class PaddingSide(Enum):
    """Side on which to apply padding.

    Attributes:
        LEFT: Pad on the left side (beginning of sequence).
        RIGHT: Pad on the right side (end of sequence).

    Examples:
        >>> PaddingSide.LEFT.value
        'left'
        >>> PaddingSide.RIGHT.value
        'right'
    """

    LEFT = "left"
    RIGHT = "right"


VALID_PADDING_STRATEGIES = frozenset(s.value for s in PaddingStrategy)
VALID_TRUNCATION_STRATEGIES = frozenset(s.value for s in TruncationStrategy)
VALID_COLLATOR_TYPES = frozenset(c.value for c in CollatorType)
VALID_PADDING_SIDES = frozenset(s.value for s in PaddingSide)


@dataclass(frozen=True, slots=True)
class PaddingConfig:
    """Configuration for padding behavior.

    Attributes:
        strategy: The padding strategy to use.
        max_length: Maximum length when using MAX_LENGTH strategy.
        pad_to_multiple_of: Pad to a multiple of this value.
        padding_side: Which side to pad on (left or right).

    Examples:
        >>> config = PaddingConfig(
        ...     strategy=PaddingStrategy.LONGEST,
        ...     max_length=512,
        ...     pad_to_multiple_of=8,
        ...     padding_side=PaddingSide.RIGHT,
        ... )
        >>> config.strategy
        <PaddingStrategy.LONGEST: 'longest'>
        >>> config.max_length
        512
        >>> config.pad_to_multiple_of
        8
        >>> config.padding_side
        <PaddingSide.RIGHT: 'right'>

        >>> config2 = PaddingConfig(
        ...     strategy=PaddingStrategy.MAX_LENGTH,
        ...     max_length=1024,
        ...     pad_to_multiple_of=None,
        ...     padding_side=PaddingSide.LEFT,
        ... )
        >>> config2.strategy
        <PaddingStrategy.MAX_LENGTH: 'max_length'>
    """

    strategy: PaddingStrategy
    max_length: int | None
    pad_to_multiple_of: int | None
    padding_side: PaddingSide


@dataclass(frozen=True, slots=True)
class TruncationConfig:
    """Configuration for truncation behavior.

    Attributes:
        strategy: The truncation strategy to use.
        max_length: Maximum length after truncation.
        stride: Stride for overlapping truncation (0 for no overlap).

    Examples:
        >>> config = TruncationConfig(
        ...     strategy=TruncationStrategy.LONGEST_FIRST,
        ...     max_length=512,
        ...     stride=0,
        ... )
        >>> config.strategy
        <TruncationStrategy.LONGEST_FIRST: 'longest_first'>
        >>> config.max_length
        512
        >>> config.stride
        0

        >>> config2 = TruncationConfig(
        ...     strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        ...     max_length=None,
        ...     stride=0,
        ... )
        >>> config2.strategy
        <TruncationStrategy.DO_NOT_TRUNCATE: 'do_not_truncate'>
    """

    strategy: TruncationStrategy
    max_length: int | None
    stride: int


@dataclass(frozen=True, slots=True)
class CollatorConfig:
    """Main configuration for data collation.

    Attributes:
        collator_type: Type of collator to use.
        padding_config: Configuration for padding.
        truncation_config: Configuration for truncation.
        mlm_probability: Probability for masked language modeling (0.0-1.0).
        return_tensors: Format for returned tensors (pt, tf, np, or None).

    Examples:
        >>> padding = PaddingConfig(
        ...     strategy=PaddingStrategy.LONGEST,
        ...     max_length=512,
        ...     pad_to_multiple_of=8,
        ...     padding_side=PaddingSide.RIGHT,
        ... )
        >>> truncation = TruncationConfig(
        ...     strategy=TruncationStrategy.LONGEST_FIRST,
        ...     max_length=512,
        ...     stride=0,
        ... )
        >>> config = CollatorConfig(
        ...     collator_type=CollatorType.DEFAULT,
        ...     padding_config=padding,
        ...     truncation_config=truncation,
        ...     mlm_probability=0.0,
        ...     return_tensors="pt",
        ... )
        >>> config.collator_type
        <CollatorType.DEFAULT: 'default'>
        >>> config.return_tensors
        'pt'

        >>> lm_config = CollatorConfig(
        ...     collator_type=CollatorType.LANGUAGE_MODELING,
        ...     padding_config=padding,
        ...     truncation_config=truncation,
        ...     mlm_probability=0.15,
        ...     return_tensors="pt",
        ... )
        >>> lm_config.mlm_probability
        0.15
    """

    collator_type: CollatorType
    padding_config: PaddingConfig
    truncation_config: TruncationConfig
    mlm_probability: float
    return_tensors: str | None


@dataclass(frozen=True, slots=True)
class CollatorStats:
    """Statistics from batch collation.

    Attributes:
        avg_length: Average sequence length in the batch.
        max_length: Maximum sequence length in the batch.
        padding_ratio: Ratio of padding tokens to total tokens.
        truncation_ratio: Ratio of samples that were truncated.

    Examples:
        >>> stats = CollatorStats(
        ...     avg_length=256.5,
        ...     max_length=512,
        ...     padding_ratio=0.25,
        ...     truncation_ratio=0.1,
        ... )
        >>> stats.avg_length
        256.5
        >>> stats.max_length
        512
        >>> stats.padding_ratio
        0.25
        >>> stats.truncation_ratio
        0.1

        >>> stats2 = CollatorStats(
        ...     avg_length=128.0,
        ...     max_length=128,
        ...     padding_ratio=0.0,
        ...     truncation_ratio=0.0,
        ... )
        >>> stats2.padding_ratio
        0.0
    """

    avg_length: float
    max_length: int
    padding_ratio: float
    truncation_ratio: float


def validate_padding_config(config: PaddingConfig) -> None:
    """Validate padding configuration.

    Args:
        config: Padding configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_length is not positive when required.
        ValueError: If pad_to_multiple_of is not positive when set.

    Examples:
        >>> config = PaddingConfig(
        ...     strategy=PaddingStrategy.LONGEST,
        ...     max_length=512,
        ...     pad_to_multiple_of=8,
        ...     padding_side=PaddingSide.RIGHT,
        ... )
        >>> validate_padding_config(config)  # No error

        >>> validate_padding_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = PaddingConfig(
        ...     strategy=PaddingStrategy.MAX_LENGTH,
        ...     max_length=None,
        ...     pad_to_multiple_of=None,
        ...     padding_side=PaddingSide.RIGHT,
        ... )
        >>> validate_padding_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length is required when strategy is MAX_LENGTH
    """
    validate_not_none(config, "config")

    if config.strategy == PaddingStrategy.MAX_LENGTH and config.max_length is None:
        msg = "max_length is required when strategy is MAX_LENGTH"
        raise ValueError(msg)

    if config.max_length is not None and config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.pad_to_multiple_of is not None and config.pad_to_multiple_of <= 0:
        msg = f"pad_to_multiple_of must be positive, got {config.pad_to_multiple_of}"
        raise ValueError(msg)


def validate_truncation_config(config: TruncationConfig) -> None:
    """Validate truncation configuration.

    Args:
        config: Truncation configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_length is not positive when required.
        ValueError: If stride is negative.

    Examples:
        >>> config = TruncationConfig(
        ...     strategy=TruncationStrategy.LONGEST_FIRST,
        ...     max_length=512,
        ...     stride=0,
        ... )
        >>> validate_truncation_config(config)  # No error

        >>> validate_truncation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = TruncationConfig(
        ...     strategy=TruncationStrategy.LONGEST_FIRST,
        ...     max_length=-1,
        ...     stride=0,
        ... )
        >>> validate_truncation_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    validate_not_none(config, "config")

    if (
        config.strategy != TruncationStrategy.DO_NOT_TRUNCATE
        and config.max_length is not None
        and config.max_length <= 0
    ):
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.stride < 0:
        msg = f"stride cannot be negative, got {config.stride}"
        raise ValueError(msg)


def validate_collator_config(config: CollatorConfig) -> None:
    """Validate collator configuration.

    Args:
        config: Collator configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If mlm_probability is out of range.
        ValueError: If return_tensors is invalid.

    Examples:
        >>> padding = PaddingConfig(
        ...     strategy=PaddingStrategy.LONGEST,
        ...     max_length=512,
        ...     pad_to_multiple_of=8,
        ...     padding_side=PaddingSide.RIGHT,
        ... )
        >>> truncation = TruncationConfig(
        ...     strategy=TruncationStrategy.LONGEST_FIRST,
        ...     max_length=512,
        ...     stride=0,
        ... )
        >>> config = CollatorConfig(
        ...     collator_type=CollatorType.DEFAULT,
        ...     padding_config=padding,
        ...     truncation_config=truncation,
        ...     mlm_probability=0.0,
        ...     return_tensors="pt",
        ... )
        >>> validate_collator_config(config)  # No error

        >>> validate_collator_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = CollatorConfig(
        ...     collator_type=CollatorType.LANGUAGE_MODELING,
        ...     padding_config=padding,
        ...     truncation_config=truncation,
        ...     mlm_probability=1.5,
        ...     return_tensors="pt",
        ... )
        >>> validate_collator_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: mlm_probability must be between 0.0 and 1.0
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.mlm_probability <= 1.0:
        msg = (
            f"mlm_probability must be between 0.0 and 1.0, got {config.mlm_probability}"
        )
        raise ValueError(msg)

    valid_return_tensors = frozenset({"pt", "tf", "np", None})
    if config.return_tensors not in valid_return_tensors:
        msg = (
            f"return_tensors must be one of {valid_return_tensors},"
            f" got '{config.return_tensors}'"
        )
        raise ValueError(msg)

    validate_padding_config(config.padding_config)
    validate_truncation_config(config.truncation_config)


def validate_collator_stats(stats: CollatorStats) -> None:
    """Validate collator statistics.

    Args:
        stats: Collator statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If any value is out of range.

    Examples:
        >>> stats = CollatorStats(
        ...     avg_length=256.5,
        ...     max_length=512,
        ...     padding_ratio=0.25,
        ...     truncation_ratio=0.1,
        ... )
        >>> validate_collator_stats(stats)  # No error

        >>> validate_collator_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad_stats = CollatorStats(
        ...     avg_length=-10.0,
        ...     max_length=512,
        ...     padding_ratio=0.25,
        ...     truncation_ratio=0.1,
        ... )
        >>> validate_collator_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: avg_length cannot be negative
    """
    validate_not_none(stats, "stats")

    if stats.avg_length < 0:
        msg = f"avg_length cannot be negative, got {stats.avg_length}"
        raise ValueError(msg)

    if stats.max_length < 0:
        msg = f"max_length cannot be negative, got {stats.max_length}"
        raise ValueError(msg)

    if not 0.0 <= stats.padding_ratio <= 1.0:
        msg = f"padding_ratio must be between 0.0 and 1.0, got {stats.padding_ratio}"
        raise ValueError(msg)

    if not 0.0 <= stats.truncation_ratio <= 1.0:
        msg = (
            "truncation_ratio must be between 0.0 and 1.0,"
            f" got {stats.truncation_ratio}"
        )
        raise ValueError(msg)


def create_padding_config(
    strategy: str | PaddingStrategy = PaddingStrategy.LONGEST,
    max_length: int | None = 512,
    pad_to_multiple_of: int | None = 8,
    padding_side: str | PaddingSide = PaddingSide.RIGHT,
) -> PaddingConfig:
    """Create a padding configuration with validation.

    Args:
        strategy: Padding strategy to use.
        max_length: Maximum length for padding.
        pad_to_multiple_of: Pad to a multiple of this value.
        padding_side: Which side to pad on.

    Returns:
        Validated PaddingConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_padding_config()
        >>> config.strategy
        <PaddingStrategy.LONGEST: 'longest'>
        >>> config.max_length
        512
        >>> config.pad_to_multiple_of
        8
        >>> config.padding_side
        <PaddingSide.RIGHT: 'right'>

        >>> config2 = create_padding_config(
        ...     strategy="max_length",
        ...     max_length=1024,
        ...     padding_side="left",
        ... )
        >>> config2.strategy
        <PaddingStrategy.MAX_LENGTH: 'max_length'>
        >>> config2.max_length
        1024
        >>> config2.padding_side
        <PaddingSide.LEFT: 'left'>

        >>> create_padding_config(max_length=-1)
        Traceback (most recent call last):
            ...
        ValueError: max_length must be positive, got -1
    """
    if isinstance(strategy, str):
        strategy = get_padding_strategy(strategy)

    if isinstance(padding_side, str):
        padding_side = get_padding_side(padding_side)

    config = PaddingConfig(
        strategy=strategy,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        padding_side=padding_side,
    )
    validate_padding_config(config)
    return config


def create_truncation_config(
    strategy: str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
    max_length: int | None = 512,
    stride: int = 0,
) -> TruncationConfig:
    """Create a truncation configuration with validation.

    Args:
        strategy: Truncation strategy to use.
        max_length: Maximum length after truncation.
        stride: Stride for overlapping truncation.

    Returns:
        Validated TruncationConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_truncation_config()
        >>> config.strategy
        <TruncationStrategy.LONGEST_FIRST: 'longest_first'>
        >>> config.max_length
        512
        >>> config.stride
        0

        >>> config2 = create_truncation_config(
        ...     strategy="only_first",
        ...     max_length=256,
        ...     stride=128,
        ... )
        >>> config2.strategy
        <TruncationStrategy.ONLY_FIRST: 'only_first'>
        >>> config2.stride
        128

        >>> create_truncation_config(stride=-1)
        Traceback (most recent call last):
            ...
        ValueError: stride cannot be negative, got -1
    """
    if isinstance(strategy, str):
        strategy = get_truncation_strategy(strategy)

    config = TruncationConfig(
        strategy=strategy,
        max_length=max_length,
        stride=stride,
    )
    validate_truncation_config(config)
    return config


def create_collator_config(
    collator_type: str | CollatorType = CollatorType.DEFAULT,
    padding_config: PaddingConfig | None = None,
    truncation_config: TruncationConfig | None = None,
    mlm_probability: float = 0.0,
    return_tensors: str | None = "pt",
) -> CollatorConfig:
    """Create a collator configuration with validation.

    Args:
        collator_type: Type of collator to use.
        padding_config: Padding configuration (uses defaults if None).
        truncation_config: Truncation configuration (uses defaults if None).
        mlm_probability: Probability for masked language modeling.
        return_tensors: Format for returned tensors.

    Returns:
        Validated CollatorConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_collator_config()
        >>> config.collator_type
        <CollatorType.DEFAULT: 'default'>
        >>> config.mlm_probability
        0.0
        >>> config.return_tensors
        'pt'

        >>> config2 = create_collator_config(
        ...     collator_type="language_modeling",
        ...     mlm_probability=0.15,
        ... )
        >>> config2.collator_type
        <CollatorType.LANGUAGE_MODELING: 'language_modeling'>
        >>> config2.mlm_probability
        0.15

        >>> create_collator_config(mlm_probability=1.5)
        Traceback (most recent call last):
            ...
        ValueError: mlm_probability must be between 0.0 and 1.0, got 1.5
    """
    if isinstance(collator_type, str):
        collator_type = get_collator_type(collator_type)

    if padding_config is None:
        padding_config = create_padding_config()

    if truncation_config is None:
        truncation_config = create_truncation_config()

    config = CollatorConfig(
        collator_type=collator_type,
        padding_config=padding_config,
        truncation_config=truncation_config,
        mlm_probability=mlm_probability,
        return_tensors=return_tensors,
    )
    validate_collator_config(config)
    return config


def create_collator_stats(
    avg_length: float = 0.0,
    max_length: int = 0,
    padding_ratio: float = 0.0,
    truncation_ratio: float = 0.0,
) -> CollatorStats:
    """Create collator statistics with validation.

    Args:
        avg_length: Average sequence length.
        max_length: Maximum sequence length.
        padding_ratio: Ratio of padding tokens.
        truncation_ratio: Ratio of truncated samples.

    Returns:
        Validated CollatorStats.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_collator_stats(
        ...     avg_length=256.5,
        ...     max_length=512,
        ...     padding_ratio=0.25,
        ...     truncation_ratio=0.1,
        ... )
        >>> stats.avg_length
        256.5
        >>> stats.max_length
        512

        >>> create_collator_stats(avg_length=-10.0)
        Traceback (most recent call last):
            ...
        ValueError: avg_length cannot be negative, got -10.0
    """
    stats = CollatorStats(
        avg_length=avg_length,
        max_length=max_length,
        padding_ratio=padding_ratio,
        truncation_ratio=truncation_ratio,
    )
    validate_collator_stats(stats)
    return stats


def list_padding_strategies() -> list[str]:
    """List all available padding strategies.

    Returns:
        Sorted list of padding strategy names.

    Examples:
        >>> strategies = list_padding_strategies()
        >>> "longest" in strategies
        True
        >>> "max_length" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_PADDING_STRATEGIES)


def list_truncation_strategies() -> list[str]:
    """List all available truncation strategies.

    Returns:
        Sorted list of truncation strategy names.

    Examples:
        >>> strategies = list_truncation_strategies()
        >>> "longest_first" in strategies
        True
        >>> "do_not_truncate" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_TRUNCATION_STRATEGIES)


def list_collator_types() -> list[str]:
    """List all available collator types.

    Returns:
        Sorted list of collator type names.

    Examples:
        >>> types = list_collator_types()
        >>> "default" in types
        True
        >>> "language_modeling" in types
        True
        >>> "seq2seq" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_COLLATOR_TYPES)


def list_padding_sides() -> list[str]:
    """List all available padding sides.

    Returns:
        Sorted list of padding side names.

    Examples:
        >>> sides = list_padding_sides()
        >>> "left" in sides
        True
        >>> "right" in sides
        True
        >>> sides == sorted(sides)
        True
    """
    return sorted(VALID_PADDING_SIDES)


def get_padding_strategy(name: str) -> PaddingStrategy:
    """Get padding strategy enum from string name.

    Args:
        name: Name of the padding strategy.

    Returns:
        Corresponding PaddingStrategy enum.

    Raises:
        ValueError: If padding strategy name is invalid.

    Examples:
        >>> get_padding_strategy("longest")
        <PaddingStrategy.LONGEST: 'longest'>
        >>> get_padding_strategy("max_length")
        <PaddingStrategy.MAX_LENGTH: 'max_length'>

        >>> get_padding_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: padding_strategy must be one of ...
    """
    if name not in VALID_PADDING_STRATEGIES:
        msg = (
            f"padding_strategy must be one of {VALID_PADDING_STRATEGIES}, got '{name}'"
        )
        raise ValueError(msg)
    return PaddingStrategy(name)


def get_truncation_strategy(name: str) -> TruncationStrategy:
    """Get truncation strategy enum from string name.

    Args:
        name: Name of the truncation strategy.

    Returns:
        Corresponding TruncationStrategy enum.

    Raises:
        ValueError: If truncation strategy name is invalid.

    Examples:
        >>> get_truncation_strategy("longest_first")
        <TruncationStrategy.LONGEST_FIRST: 'longest_first'>
        >>> get_truncation_strategy("only_first")
        <TruncationStrategy.ONLY_FIRST: 'only_first'>

        >>> get_truncation_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: truncation_strategy must be one of ...
    """
    if name not in VALID_TRUNCATION_STRATEGIES:
        msg = (
            f"truncation_strategy must be one of "
            f"{VALID_TRUNCATION_STRATEGIES}, got '{name}'"
        )
        raise ValueError(msg)
    return TruncationStrategy(name)


def get_collator_type(name: str) -> CollatorType:
    """Get collator type enum from string name.

    Args:
        name: Name of the collator type.

    Returns:
        Corresponding CollatorType enum.

    Raises:
        ValueError: If collator type name is invalid.

    Examples:
        >>> get_collator_type("default")
        <CollatorType.DEFAULT: 'default'>
        >>> get_collator_type("language_modeling")
        <CollatorType.LANGUAGE_MODELING: 'language_modeling'>

        >>> get_collator_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: collator_type must be one of ...
    """
    if name not in VALID_COLLATOR_TYPES:
        msg = f"collator_type must be one of {VALID_COLLATOR_TYPES}, got '{name}'"
        raise ValueError(msg)
    return CollatorType(name)


def get_padding_side(name: str) -> PaddingSide:
    """Get padding side enum from string name.

    Args:
        name: Name of the padding side.

    Returns:
        Corresponding PaddingSide enum.

    Raises:
        ValueError: If padding side name is invalid.

    Examples:
        >>> get_padding_side("left")
        <PaddingSide.LEFT: 'left'>
        >>> get_padding_side("right")
        <PaddingSide.RIGHT: 'right'>

        >>> get_padding_side("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: padding_side must be one of ...
    """
    if name not in VALID_PADDING_SIDES:
        msg = f"padding_side must be one of {VALID_PADDING_SIDES}, got '{name}'"
        raise ValueError(msg)
    return PaddingSide(name)


def calculate_padding_stats(
    sequence_lengths: list[int],
    padded_length: int,
    original_max_length: int | None = None,
) -> CollatorStats:
    """Calculate padding statistics for a batch.

    Computes average length, padding ratio, and truncation ratio
    based on the input sequence lengths and target padded length.

    Args:
        sequence_lengths: List of original sequence lengths.
        padded_length: Final padded length for all sequences.
        original_max_length: Maximum length before any truncation (optional).

    Returns:
        CollatorStats with computed statistics.

    Raises:
        ValueError: If sequence_lengths is empty.
        ValueError: If padded_length is not positive.

    Examples:
        >>> stats = calculate_padding_stats([100, 200, 300], 300)
        >>> stats.avg_length
        200.0
        >>> stats.max_length
        300
        >>> round(stats.padding_ratio, 4)
        0.3333

        >>> stats2 = calculate_padding_stats([50, 50, 50], 50)
        >>> stats2.padding_ratio
        0.0

        >>> stats3 = calculate_padding_stats([100, 200], 150, original_max_length=250)
        >>> stats3.truncation_ratio
        0.5

        >>> calculate_padding_stats([], 100)
        Traceback (most recent call last):
            ...
        ValueError: sequence_lengths cannot be empty

        >>> calculate_padding_stats([100, 200], 0)
        Traceback (most recent call last):
            ...
        ValueError: padded_length must be positive
    """
    if not sequence_lengths:
        msg = "sequence_lengths cannot be empty"
        raise ValueError(msg)

    if padded_length <= 0:
        msg = f"padded_length must be positive, got {padded_length}"
        raise ValueError(msg)

    n_samples = len(sequence_lengths)
    avg_length = sum(sequence_lengths) / n_samples
    max_length = max(sequence_lengths)

    # Calculate padding ratio
    total_tokens = padded_length * n_samples
    actual_tokens = sum(min(length, padded_length) for length in sequence_lengths)
    padding_tokens = total_tokens - actual_tokens
    padding_ratio = padding_tokens / total_tokens if total_tokens > 0 else 0.0

    # Calculate truncation ratio
    if original_max_length is not None:
        truncated_samples = sum(
            1 for length in sequence_lengths if length > padded_length
        )
        truncation_ratio = truncated_samples / n_samples
    else:
        truncation_ratio = 0.0

    return CollatorStats(
        avg_length=avg_length,
        max_length=max_length,
        padding_ratio=padding_ratio,
        truncation_ratio=truncation_ratio,
    )


def estimate_memory_per_batch(
    batch_size: int,
    sequence_length: int,
    hidden_size: int = 768,
    dtype_bytes: int = 4,
    include_activations: bool = True,
) -> int:
    """Estimate memory usage per batch in bytes.

    Calculates the approximate memory footprint for a batch of
    sequences based on model dimensions and data types.

    Args:
        batch_size: Number of samples in the batch.
        sequence_length: Length of each sequence.
        hidden_size: Hidden dimension of the model.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).
        include_activations: Whether to include activation memory estimate.

    Returns:
        Estimated memory in bytes.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> mem = estimate_memory_per_batch(32, 512, 768, 4)
        >>> mem > 0
        True

        >>> mem_fp16 = estimate_memory_per_batch(32, 512, 768, 2)
        >>> mem_fp16 < mem
        True

        >>> mem_no_act = estimate_memory_per_batch(32, 512, 768, 4, False)
        >>> mem_no_act < mem
        True

        >>> estimate_memory_per_batch(0, 512, 768, 4)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be positive

        >>> estimate_memory_per_batch(32, 0, 768, 4)
        Traceback (most recent call last):
            ...
        ValueError: sequence_length must be positive
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if sequence_length <= 0:
        msg = "sequence_length must be positive"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = "hidden_size must be positive"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = "dtype_bytes must be positive"
        raise ValueError(msg)

    # Base memory: embeddings
    embedding_memory = batch_size * sequence_length * hidden_size * dtype_bytes

    # Attention memory: QKV + attention scores + output
    attention_memory = 0
    if include_activations:
        # Q, K, V projections
        qkv_memory = 3 * batch_size * sequence_length * hidden_size * dtype_bytes
        # Attention scores (batch, heads, seq, seq) - approximation
        attention_scores_memory = (
            batch_size * sequence_length * sequence_length * dtype_bytes
        )
        attention_memory = qkv_memory + attention_scores_memory

    return embedding_memory + attention_memory


def optimize_batch_padding(
    sequence_lengths: list[int],
    max_allowed_length: int,
    pad_to_multiple_of: int | None = 8,
) -> tuple[int, float]:
    """Calculate optimal padding length to minimize waste.

    Finds a padding length that balances between accommodating
    most sequences and minimizing padding overhead.

    Args:
        sequence_lengths: List of original sequence lengths.
        max_allowed_length: Maximum allowed sequence length.
        pad_to_multiple_of: Pad to a multiple of this value (for efficiency).

    Returns:
        Tuple of (optimal_padding_length, waste_ratio).

    Raises:
        ValueError: If sequence_lengths is empty.
        ValueError: If max_allowed_length is not positive.

    Examples:
        >>> length, waste = optimize_batch_padding([100, 200, 300, 500], 512)
        >>> length <= 512
        True
        >>> 0.0 <= waste <= 1.0
        True

        >>> length2, waste2 = optimize_batch_padding([128, 128, 128], 512)
        >>> length2
        128
        >>> waste2
        0.0

        >>> optimize_batch_padding([], 512)
        Traceback (most recent call last):
            ...
        ValueError: sequence_lengths cannot be empty

        >>> optimize_batch_padding([100, 200], 0)
        Traceback (most recent call last):
            ...
        ValueError: max_allowed_length must be positive
    """
    if not sequence_lengths:
        msg = "sequence_lengths cannot be empty"
        raise ValueError(msg)

    if max_allowed_length <= 0:
        msg = f"max_allowed_length must be positive, got {max_allowed_length}"
        raise ValueError(msg)

    # Cap all lengths at max_allowed_length
    capped_lengths = [min(length, max_allowed_length) for length in sequence_lengths]

    # Find the maximum length in the capped list
    optimal_length = max(capped_lengths)

    # Round up to multiple if specified
    if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
        remainder = optimal_length % pad_to_multiple_of
        if remainder > 0:
            optimal_length = optimal_length + (pad_to_multiple_of - remainder)

    # Ensure we don't exceed max_allowed_length
    optimal_length = min(optimal_length, max_allowed_length)

    # Calculate waste ratio
    total_capacity = optimal_length * len(capped_lengths)
    actual_content = sum(capped_lengths)
    waste = total_capacity - actual_content
    waste_ratio = waste / total_capacity if total_capacity > 0 else 0.0

    return optimal_length, waste_ratio


def validate_collator_output(
    batch: dict[str, Any],
    expected_keys: list[str] | None = None,
    expected_batch_size: int | None = None,
    expected_sequence_length: int | None = None,
) -> bool:
    """Validate the output of a data collator.

    Checks that the batch contains expected keys and dimensions.

    Args:
        batch: The collated batch dictionary.
        expected_keys: List of keys that must be present.
        expected_batch_size: Expected batch size (optional).
        expected_sequence_length: Expected sequence length (optional).

    Returns:
        True if validation passes.

    Raises:
        ValueError: If batch is None or empty.
        ValueError: If expected keys are missing.
        ValueError: If dimensions don't match expectations.

    Examples:
        >>> import numpy as np
        >>> batch = {
        ...     "input_ids": np.zeros((4, 128)),
        ...     "attention_mask": np.ones((4, 128)),
        ... }
        >>> validate_collator_output(batch)
        True

        >>> validate_collator_output(
        ...     batch,
        ...     expected_keys=["input_ids", "attention_mask"],
        ... )
        True

        >>> validate_collator_output(batch, expected_batch_size=4)
        True

        >>> validate_collator_output(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch cannot be None

        >>> validate_collator_output({})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch cannot be empty

        >>> validate_collator_output(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     batch,
        ...     expected_keys=["labels"],
        ... )
        Traceback (most recent call last):
        ValueError: Missing expected keys: ['labels']
    """
    if batch is None:
        msg = "batch cannot be None"
        raise ValueError(msg)

    if not batch:
        msg = "batch cannot be empty"
        raise ValueError(msg)

    # Check expected keys
    if expected_keys is not None:
        missing_keys = [key for key in expected_keys if key not in batch]
        if missing_keys:
            msg = f"Missing expected keys: {missing_keys}"
            raise ValueError(msg)

    # Check dimensions
    if expected_batch_size is not None or expected_sequence_length is not None:
        # Get first tensor-like value to check dimensions
        for _key, value in batch.items():
            if hasattr(value, "shape"):
                shape = value.shape
                if (
                    expected_batch_size is not None
                    and len(shape) > 0
                    and shape[0] != expected_batch_size
                ):
                    msg = f"Expected batch size {expected_batch_size}, got {shape[0]}"
                    raise ValueError(msg)
                if (
                    expected_sequence_length is not None
                    and len(shape) > 1
                    and shape[1] != expected_sequence_length
                ):
                    msg = (
                        f"Expected sequence length "
                        f"{expected_sequence_length}, "
                        f"got {shape[1]}"
                    )
                    raise ValueError(msg)
                break

    return True


def format_collator_stats(stats: CollatorStats) -> str:
    """Format collator statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_collator_stats(
        ...     avg_length=256.5,
        ...     max_length=512,
        ...     padding_ratio=0.25,
        ...     truncation_ratio=0.1,
        ... )
        >>> formatted = format_collator_stats(stats)
        >>> "Avg Length: 256.50" in formatted
        True
        >>> "Max Length: 512" in formatted
        True
        >>> "Padding: 25.0%" in formatted
        True
        >>> "Truncation: 10.0%" in formatted
        True
    """
    padding_pct = stats.padding_ratio * 100
    truncation_pct = stats.truncation_ratio * 100

    return (
        f"Collator Stats:\n"
        f"  Avg Length: {stats.avg_length:.2f}\n"
        f"  Max Length: {stats.max_length}\n"
        f"  Padding: {padding_pct:.1f}%\n"
        f"  Truncation: {truncation_pct:.1f}%"
    )


def _collator_classification() -> CollatorConfig:
    """Create collator config for classification tasks."""
    return create_collator_config(
        collator_type=CollatorType.DEFAULT,
        padding_config=create_padding_config(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
            pad_to_multiple_of=8,
        ),
        truncation_config=create_truncation_config(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
        ),
        mlm_probability=0.0,
    )


def _collator_generation() -> CollatorConfig:
    """Create collator config for generation tasks."""
    return create_collator_config(
        collator_type=CollatorType.LANGUAGE_MODELING,
        padding_config=create_padding_config(
            strategy=PaddingStrategy.LONGEST,
            max_length=2048,
            padding_side=PaddingSide.LEFT,
        ),
        truncation_config=create_truncation_config(
            strategy=TruncationStrategy.ONLY_FIRST,
            max_length=2048,
        ),
        mlm_probability=0.0,
    )


def _collator_seq2seq() -> CollatorConfig:
    """Create collator config for seq2seq tasks."""
    return create_collator_config(
        collator_type=CollatorType.SEQ2SEQ,
        padding_config=create_padding_config(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
        ),
        truncation_config=create_truncation_config(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
        ),
        mlm_probability=0.0,
    )


def _collator_masked_lm() -> CollatorConfig:
    """Create collator config for masked LM tasks."""
    return create_collator_config(
        collator_type=CollatorType.LANGUAGE_MODELING,
        padding_config=create_padding_config(
            strategy=PaddingStrategy.MAX_LENGTH,
            max_length=512,
        ),
        truncation_config=create_truncation_config(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
        ),
        mlm_probability=0.15,
    )


def _collator_causal_lm() -> CollatorConfig:
    """Create collator config for causal LM tasks."""
    return create_collator_config(
        collator_type=CollatorType.LANGUAGE_MODELING,
        padding_config=create_padding_config(
            strategy=PaddingStrategy.LONGEST,
            max_length=2048,
            padding_side=PaddingSide.LEFT,
        ),
        truncation_config=create_truncation_config(
            strategy=TruncationStrategy.ONLY_FIRST,
            max_length=2048,
        ),
        mlm_probability=0.0,
    )


def _collator_completion() -> CollatorConfig:
    """Create collator config for completion tasks."""
    return create_collator_config(
        collator_type=CollatorType.COMPLETION_ONLY,
        padding_config=create_padding_config(
            strategy=PaddingStrategy.LONGEST,
            max_length=4096,
            padding_side=PaddingSide.LEFT,
        ),
        truncation_config=create_truncation_config(
            strategy=TruncationStrategy.ONLY_FIRST,
            max_length=4096,
        ),
        mlm_probability=0.0,
    )


def get_recommended_collator_config(task_type: str) -> CollatorConfig:
    """Get recommended collator configuration for a task type.

    Args:
        task_type: Type of task (classification, generation, seq2seq,
            masked_lm, causal_lm, completion).

    Returns:
        Recommended CollatorConfig for the task.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_collator_config("classification")
        >>> config.collator_type
        <CollatorType.DEFAULT: 'default'>
        >>> config.padding_config.strategy
        <PaddingStrategy.LONGEST: 'longest'>

        >>> config2 = get_recommended_collator_config("masked_lm")
        >>> config2.collator_type
        <CollatorType.LANGUAGE_MODELING: 'language_modeling'>
        >>> config2.mlm_probability
        0.15

        >>> config3 = get_recommended_collator_config("seq2seq")
        >>> config3.collator_type
        <CollatorType.SEQ2SEQ: 'seq2seq'>

        >>> get_recommended_collator_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task_type must be one of ...
    """
    valid_tasks = frozenset(
        {
            "classification",
            "generation",
            "seq2seq",
            "masked_lm",
            "causal_lm",
            "completion",
        }
    )

    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    builders: dict[str, object] = {
        "classification": _collator_classification,
        "generation": _collator_generation,
        "seq2seq": _collator_seq2seq,
        "masked_lm": _collator_masked_lm,
        "causal_lm": _collator_causal_lm,
        "completion": _collator_completion,
    }
    return builders[task_type]()
