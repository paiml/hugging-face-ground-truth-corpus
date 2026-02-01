"""Data augmentation utilities for text preprocessing.

This module provides text augmentation techniques for training data
enhancement, including synonym replacement, random operations,
back-translation patterns, and noise injection.

Examples:
    >>> from hf_gtc.preprocessing.augmentation import AugmentConfig
    >>> config = AugmentConfig(probability=0.2)
    >>> config.probability
    0.2
    >>> from hf_gtc.preprocessing.augmentation import NoiseType, AugmentationLevel
    >>> NoiseType.CHARACTER.value
    'character'
    >>> AugmentationLevel.MEDIUM.value
    'medium'
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from hf_gtc._validation import validate_not_none


class AugmentationType(Enum):
    """Types of text augmentation operations.

    Attributes:
        SYNONYM_REPLACE: Replace words with synonyms.
        RANDOM_INSERT: Insert random words.
        RANDOM_SWAP: Swap adjacent words.
        RANDOM_DELETE: Delete random words.
        BACK_TRANSLATE: Back-translation augmentation.
        NOISE: Add noise to text (character/word level).
        NONE: No augmentation.

    Examples:
        >>> AugmentationType.SYNONYM_REPLACE.value
        'synonym_replace'
        >>> AugmentationType.NOISE.value
        'noise'
        >>> AugmentationType.NONE.value
        'none'
    """

    SYNONYM_REPLACE = "synonym_replace"
    RANDOM_INSERT = "random_insert"
    RANDOM_SWAP = "random_swap"
    RANDOM_DELETE = "random_delete"
    BACK_TRANSLATE = "back_translate"
    NOISE = "noise"
    NONE = "none"


VALID_AUGMENTATION_TYPES = frozenset(t.value for t in AugmentationType)


class NoiseType(Enum):
    """Types of noise injection for text augmentation.

    Attributes:
        CHARACTER: Character-level noise (typos, substitutions).
        WORD: Word-level noise (word drops, shuffles).
        KEYBOARD: Keyboard-based typos (nearby key substitution).
        OCR: OCR-like errors (similar-looking character substitution).

    Examples:
        >>> NoiseType.CHARACTER.value
        'character'
        >>> NoiseType.KEYBOARD.value
        'keyboard'
        >>> NoiseType.OCR.value
        'ocr'
    """

    CHARACTER = "character"
    WORD = "word"
    KEYBOARD = "keyboard"
    OCR = "ocr"


VALID_NOISE_TYPES = frozenset(t.value for t in NoiseType)


class AugmentationLevel(Enum):
    """Intensity levels for augmentation operations.

    Attributes:
        LIGHT: Minimal changes, preserves most of original text.
        MEDIUM: Moderate changes, balanced augmentation.
        HEAVY: Significant changes, aggressive augmentation.

    Examples:
        >>> AugmentationLevel.LIGHT.value
        'light'
        >>> AugmentationLevel.MEDIUM.value
        'medium'
        >>> AugmentationLevel.HEAVY.value
        'heavy'
    """

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


VALID_AUGMENTATION_LEVELS = frozenset(level.value for level in AugmentationLevel)


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Configuration for text augmentation.

    Attributes:
        probability: Probability of applying augmentation per word. Defaults to 0.1.
        num_augmentations: Number of augmented versions to generate. Defaults to 1.
        augmentation_type: Type of augmentation. Defaults to SYNONYM_REPLACE.
        min_length: Minimum text length to augment. Defaults to 3.
        preserve_case: Whether to preserve original case. Defaults to True.

    Examples:
        >>> config = AugmentConfig(probability=0.2, num_augmentations=2)
        >>> config.probability
        0.2
        >>> config.num_augmentations
        2
    """

    probability: float = 0.1
    num_augmentations: int = 1
    augmentation_type: AugmentationType = AugmentationType.SYNONYM_REPLACE
    min_length: int = 3
    preserve_case: bool = True


@dataclass(frozen=True, slots=True)
class SynonymConfig:
    """Configuration for synonym-based augmentation.

    Attributes:
        method: Method for finding synonyms (wordnet, embedding, thesaurus).
        max_replacements: Maximum number of words to replace per text.
        similarity_threshold: Minimum similarity score for valid synonyms.

    Examples:
        >>> config = SynonymConfig(
        ...     method="wordnet",
        ...     max_replacements=5,
        ...     similarity_threshold=0.8,
        ... )
        >>> config.method
        'wordnet'
        >>> config.max_replacements
        5
        >>> config.similarity_threshold
        0.8
    """

    method: str
    max_replacements: int
    similarity_threshold: float


@dataclass(frozen=True, slots=True)
class BacktranslationConfig:
    """Configuration for back-translation augmentation.

    Attributes:
        pivot_languages: Tuple of languages to use as pivots (e.g., 'de', 'fr').
        num_translations: Number of back-translation variants to generate.

    Examples:
        >>> config = BacktranslationConfig(
        ...     pivot_languages=("de", "fr", "es"),
        ...     num_translations=2,
        ... )
        >>> len(config.pivot_languages)
        3
        >>> config.num_translations
        2
    """

    pivot_languages: tuple[str, ...]
    num_translations: int


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Configuration for noise injection augmentation.

    Attributes:
        noise_type: Type of noise to inject.
        probability: Probability of injecting noise per character/word.
        preserve_semantics: Whether to try preserving semantic meaning.

    Examples:
        >>> config = NoiseConfig(
        ...     noise_type=NoiseType.CHARACTER,
        ...     probability=0.1,
        ...     preserve_semantics=True,
        ... )
        >>> config.noise_type
        <NoiseType.CHARACTER: 'character'>
        >>> config.probability
        0.1
    """

    noise_type: NoiseType
    probability: float
    preserve_semantics: bool


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """General configuration for augmentation pipelines.

    Attributes:
        aug_type: Type of augmentation to apply.
        probability: Overall probability of applying augmentation.
        preserve_labels: Whether to preserve labels during augmentation.

    Examples:
        >>> config = AugmentationConfig(
        ...     aug_type=AugmentationType.SYNONYM_REPLACE,
        ...     probability=0.3,
        ...     preserve_labels=True,
        ... )
        >>> config.aug_type
        <AugmentationType.SYNONYM_REPLACE: 'synonym_replace'>
        >>> config.probability
        0.3
    """

    aug_type: AugmentationType
    probability: float
    preserve_labels: bool


@dataclass(frozen=True, slots=True)
class AugmentResult:
    """Result of an augmentation operation.

    Attributes:
        original: Original text.
        augmented: List of augmented texts.
        operations_applied: Count of operations per augmented text.

    Examples:
        >>> result = AugmentResult(
        ...     original="hello world",
        ...     augmented=["hi world", "hello earth"],
        ...     operations_applied=[1, 1],
        ... )
        >>> result.original
        'hello world'
        >>> len(result.augmented)
        2
    """

    original: str
    augmented: list[str]
    operations_applied: list[int]


@dataclass(frozen=True, slots=True)
class AugmentationStats:
    """Statistics from augmentation operations.

    Attributes:
        total_texts: Total number of texts processed.
        total_augmented: Total number of augmented versions created.
        total_operations: Total number of operations applied.
        avg_operations_per_text: Average operations per input text.
        diversity_score: Measure of diversity in augmented outputs.

    Examples:
        >>> stats = AugmentationStats(
        ...     total_texts=100,
        ...     total_augmented=300,
        ...     total_operations=450,
        ...     avg_operations_per_text=4.5,
        ...     diversity_score=0.75,
        ... )
        >>> stats.total_texts
        100
        >>> stats.diversity_score
        0.75
    """

    total_texts: int
    total_augmented: int
    total_operations: int
    avg_operations_per_text: float
    diversity_score: float


def validate_augment_config(config: AugmentConfig) -> None:
    """Validate augmentation configuration.

    Args:
        config: AugmentConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If probability is not in [0, 1].
        ValueError: If num_augmentations is not positive.
        ValueError: If min_length is negative.

    Examples:
        >>> config = AugmentConfig(probability=0.2)
        >>> validate_augment_config(config)  # No error

        >>> validate_augment_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AugmentConfig(probability=1.5)
        >>> validate_augment_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probability must be between 0 and 1
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.probability <= 1.0:
        msg = f"probability must be between 0 and 1, got {config.probability}"
        raise ValueError(msg)

    if config.num_augmentations <= 0:
        msg = f"num_augmentations must be positive, got {config.num_augmentations}"
        raise ValueError(msg)

    if config.min_length < 0:
        msg = f"min_length cannot be negative, got {config.min_length}"
        raise ValueError(msg)


def validate_synonym_config(config: SynonymConfig) -> None:
    """Validate synonym configuration.

    Args:
        config: SynonymConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If method is empty.
        ValueError: If max_replacements is not positive.
        ValueError: If similarity_threshold is not in [0, 1].

    Examples:
        >>> config = SynonymConfig(
        ...     method="wordnet",
        ...     max_replacements=5,
        ...     similarity_threshold=0.8,
        ... )
        >>> validate_synonym_config(config)  # No error

        >>> validate_synonym_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SynonymConfig(method="", max_replacements=5, similarity_threshold=0.8)
        >>> validate_synonym_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method cannot be empty
    """
    validate_not_none(config, "config")

    if not config.method:
        msg = "method cannot be empty"
        raise ValueError(msg)

    if config.max_replacements <= 0:
        msg = f"max_replacements must be positive, got {config.max_replacements}"
        raise ValueError(msg)

    if not 0.0 <= config.similarity_threshold <= 1.0:
        msg = (
            f"similarity_threshold must be between 0 and 1, "
            f"got {config.similarity_threshold}"
        )
        raise ValueError(msg)


def validate_backtranslation_config(config: BacktranslationConfig) -> None:
    """Validate back-translation configuration.

    Args:
        config: BacktranslationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If pivot_languages is empty.
        ValueError: If num_translations is not positive.

    Examples:
        >>> config = BacktranslationConfig(
        ...     pivot_languages=("de", "fr"),
        ...     num_translations=2,
        ... )
        >>> validate_backtranslation_config(config)  # No error

        >>> validate_backtranslation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BacktranslationConfig(pivot_languages=(), num_translations=2)
        >>> validate_backtranslation_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pivot_languages cannot be empty
    """
    validate_not_none(config, "config")

    if not config.pivot_languages:
        msg = "pivot_languages cannot be empty"
        raise ValueError(msg)

    if config.num_translations <= 0:
        msg = f"num_translations must be positive, got {config.num_translations}"
        raise ValueError(msg)


def validate_noise_config(config: NoiseConfig) -> None:
    """Validate noise configuration.

    Args:
        config: NoiseConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If probability is not in [0, 1].

    Examples:
        >>> config = NoiseConfig(
        ...     noise_type=NoiseType.CHARACTER,
        ...     probability=0.1,
        ...     preserve_semantics=True,
        ... )
        >>> validate_noise_config(config)  # No error

        >>> validate_noise_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = NoiseConfig(
        ...     noise_type=NoiseType.CHARACTER,
        ...     probability=1.5,
        ...     preserve_semantics=True,
        ... )
        >>> validate_noise_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probability must be between 0 and 1
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.probability <= 1.0:
        msg = f"probability must be between 0 and 1, got {config.probability}"
        raise ValueError(msg)


def validate_augmentation_config(config: AugmentationConfig) -> None:
    """Validate general augmentation configuration.

    Args:
        config: AugmentationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If probability is not in [0, 1].

    Examples:
        >>> config = AugmentationConfig(
        ...     aug_type=AugmentationType.SYNONYM_REPLACE,
        ...     probability=0.3,
        ...     preserve_labels=True,
        ... )
        >>> validate_augmentation_config(config)  # No error

        >>> validate_augmentation_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AugmentationConfig(
        ...     aug_type=AugmentationType.NONE,
        ...     probability=-0.1,
        ...     preserve_labels=True,
        ... )
        >>> validate_augmentation_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probability must be between 0 and 1
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.probability <= 1.0:
        msg = f"probability must be between 0 and 1, got {config.probability}"
        raise ValueError(msg)


def create_synonym_config(
    method: str = "wordnet",
    max_replacements: int = 5,
    similarity_threshold: float = 0.8,
) -> SynonymConfig:
    """Create a synonym configuration.

    Args:
        method: Method for finding synonyms. Defaults to "wordnet".
        max_replacements: Maximum replacements per text. Defaults to 5.
        similarity_threshold: Minimum similarity score. Defaults to 0.8.

    Returns:
        SynonymConfig with the specified settings.

    Raises:
        ValueError: If method is empty.
        ValueError: If max_replacements is not positive.
        ValueError: If similarity_threshold is not in [0, 1].

    Examples:
        >>> config = create_synonym_config(method="embedding")
        >>> config.method
        'embedding'

        >>> create_synonym_config(method="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method cannot be empty

        >>> create_synonym_config(max_replacements=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_replacements must be positive
    """
    config = SynonymConfig(
        method=method,
        max_replacements=max_replacements,
        similarity_threshold=similarity_threshold,
    )
    validate_synonym_config(config)
    return config


def create_backtranslation_config(
    pivot_languages: tuple[str, ...] = ("de", "fr"),
    num_translations: int = 1,
) -> BacktranslationConfig:
    """Create a back-translation configuration.

    Args:
        pivot_languages: Languages to use as pivots. Defaults to ("de", "fr").
        num_translations: Number of translations to generate. Defaults to 1.

    Returns:
        BacktranslationConfig with the specified settings.

    Raises:
        ValueError: If pivot_languages is empty.
        ValueError: If num_translations is not positive.

    Examples:
        >>> config = create_backtranslation_config(pivot_languages=("es", "it"))
        >>> len(config.pivot_languages)
        2

        >>> create_backtranslation_config(pivot_languages=())
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pivot_languages cannot be empty

        >>> create_backtranslation_config(num_translations=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_translations must be positive
    """
    config = BacktranslationConfig(
        pivot_languages=pivot_languages,
        num_translations=num_translations,
    )
    validate_backtranslation_config(config)
    return config


def create_noise_config(
    noise_type: str = "character",
    probability: float = 0.1,
    preserve_semantics: bool = True,
) -> NoiseConfig:
    """Create a noise configuration.

    Args:
        noise_type: Type of noise to inject. Defaults to "character".
        probability: Probability of noise injection. Defaults to 0.1.
        preserve_semantics: Whether to preserve meaning. Defaults to True.

    Returns:
        NoiseConfig with the specified settings.

    Raises:
        ValueError: If noise_type is not valid.
        ValueError: If probability is not in [0, 1].

    Examples:
        >>> config = create_noise_config(noise_type="keyboard")
        >>> config.noise_type
        <NoiseType.KEYBOARD: 'keyboard'>

        >>> create_noise_config(noise_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: noise_type must be one of

        >>> create_noise_config(probability=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probability must be between 0 and 1
    """
    if noise_type not in VALID_NOISE_TYPES:
        msg = f"noise_type must be one of {VALID_NOISE_TYPES}, got '{noise_type}'"
        raise ValueError(msg)

    config = NoiseConfig(
        noise_type=NoiseType(noise_type),
        probability=probability,
        preserve_semantics=preserve_semantics,
    )
    validate_noise_config(config)
    return config


def create_augmentation_config(
    aug_type: str = "synonym_replace",
    probability: float = 0.2,
    preserve_labels: bool = True,
) -> AugmentationConfig:
    """Create a general augmentation configuration.

    Args:
        aug_type: Type of augmentation. Defaults to "synonym_replace".
        probability: Probability of augmentation. Defaults to 0.2.
        preserve_labels: Whether to preserve labels. Defaults to True.

    Returns:
        AugmentationConfig with the specified settings.

    Raises:
        ValueError: If aug_type is not valid.
        ValueError: If probability is not in [0, 1].

    Examples:
        >>> config = create_augmentation_config(aug_type="random_delete")
        >>> config.aug_type
        <AugmentationType.RANDOM_DELETE: 'random_delete'>

        >>> create_augmentation_config(aug_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: aug_type must be one of

        >>> create_augmentation_config(probability=-0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: probability must be between 0 and 1
    """
    if aug_type not in VALID_AUGMENTATION_TYPES:
        msg = f"aug_type must be one of {VALID_AUGMENTATION_TYPES}, got '{aug_type}'"
        raise ValueError(msg)

    config = AugmentationConfig(
        aug_type=AugmentationType(aug_type),
        probability=probability,
        preserve_labels=preserve_labels,
    )
    validate_augmentation_config(config)
    return config


def list_augmentation_types() -> list[str]:
    """List all available augmentation types.

    Returns:
        Sorted list of augmentation type names.

    Examples:
        >>> types = list_augmentation_types()
        >>> "synonym_replace" in types
        True
        >>> "none" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_AUGMENTATION_TYPES)


def list_noise_types() -> list[str]:
    """List all available noise types.

    Returns:
        Sorted list of noise type names.

    Examples:
        >>> types = list_noise_types()
        >>> "character" in types
        True
        >>> "keyboard" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_NOISE_TYPES)


def list_augmentation_levels() -> list[str]:
    """List all available augmentation levels.

    Returns:
        Sorted list of augmentation level names.

    Examples:
        >>> levels = list_augmentation_levels()
        >>> "light" in levels
        True
        >>> "heavy" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_AUGMENTATION_LEVELS)


def validate_augmentation_type(aug_type: str) -> bool:
    """Validate if a string is a valid augmentation type.

    Args:
        aug_type: The type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_augmentation_type("synonym_replace")
        True
        >>> validate_augmentation_type("random_delete")
        True
        >>> validate_augmentation_type("invalid")
        False
        >>> validate_augmentation_type("")
        False
    """
    return aug_type in VALID_AUGMENTATION_TYPES


def validate_noise_type(noise_type: str) -> bool:
    """Validate if a string is a valid noise type.

    Args:
        noise_type: The type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_noise_type("character")
        True
        >>> validate_noise_type("keyboard")
        True
        >>> validate_noise_type("invalid")
        False
        >>> validate_noise_type("")
        False
    """
    return noise_type in VALID_NOISE_TYPES


def validate_augmentation_level(level: str) -> bool:
    """Validate if a string is a valid augmentation level.

    Args:
        level: The level string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_augmentation_level("light")
        True
        >>> validate_augmentation_level("heavy")
        True
        >>> validate_augmentation_level("invalid")
        False
        >>> validate_augmentation_level("")
        False
    """
    return level in VALID_AUGMENTATION_LEVELS


def get_augmentation_type(name: str) -> AugmentationType:
    """Get AugmentationType enum from string name.

    Args:
        name: Name of the augmentation type.

    Returns:
        Corresponding AugmentationType enum value.

    Raises:
        ValueError: If name is not a valid augmentation type.

    Examples:
        >>> get_augmentation_type("synonym_replace")
        <AugmentationType.SYNONYM_REPLACE: 'synonym_replace'>

        >>> get_augmentation_type("none")
        <AugmentationType.NONE: 'none'>

        >>> get_augmentation_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid augmentation type: invalid
    """
    if not validate_augmentation_type(name):
        msg = f"invalid augmentation type: {name}"
        raise ValueError(msg)

    return AugmentationType(name)


def get_noise_type(name: str) -> NoiseType:
    """Get NoiseType enum from string name.

    Args:
        name: Name of the noise type.

    Returns:
        Corresponding NoiseType enum value.

    Raises:
        ValueError: If name is not a valid noise type.

    Examples:
        >>> get_noise_type("character")
        <NoiseType.CHARACTER: 'character'>

        >>> get_noise_type("keyboard")
        <NoiseType.KEYBOARD: 'keyboard'>

        >>> get_noise_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid noise type: invalid
    """
    if not validate_noise_type(name):
        msg = f"invalid noise type: {name}"
        raise ValueError(msg)

    return NoiseType(name)


def get_augmentation_level(name: str) -> AugmentationLevel:
    """Get AugmentationLevel enum from string name.

    Args:
        name: Name of the augmentation level.

    Returns:
        Corresponding AugmentationLevel enum value.

    Raises:
        ValueError: If name is not a valid augmentation level.

    Examples:
        >>> get_augmentation_level("light")
        <AugmentationLevel.LIGHT: 'light'>

        >>> get_augmentation_level("heavy")
        <AugmentationLevel.HEAVY: 'heavy'>

        >>> get_augmentation_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid augmentation level: invalid
    """
    if not validate_augmentation_level(name):
        msg = f"invalid augmentation level: {name}"
        raise ValueError(msg)

    return AugmentationLevel(name)


def random_delete(
    words: list[str],
    probability: float,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Randomly delete words from a list.

    Args:
        words: List of words to process.
        probability: Probability of deleting each word.
        rng: Random number generator. Defaults to None (uses global random).

    Returns:
        List with some words randomly removed.

    Raises:
        ValueError: If words is None.
        ValueError: If probability is not in [0, 1].

    Examples:
        >>> rng = random.Random(42)
        >>> random_delete(["hello", "world", "test"], 0.0, rng=rng)
        ['hello', 'world', 'test']

        >>> random_delete(["hello", "world"], 1.0, rng=rng)
        []

        >>> random_delete(None, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: words cannot be None
    """
    if words is None:
        msg = "words cannot be None"
        raise ValueError(msg)

    if not 0.0 <= probability <= 1.0:
        msg = f"probability must be between 0 and 1, got {probability}"
        raise ValueError(msg)

    if not words or probability == 0.0:
        return list(words)

    if probability == 1.0:
        return []

    rng = rng or random.Random()
    return [w for w in words if rng.random() > probability]


def random_swap(
    words: list[str],
    num_swaps: int,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Randomly swap adjacent words.

    Args:
        words: List of words to process.
        num_swaps: Number of swap operations to perform.
        rng: Random number generator. Defaults to None.

    Returns:
        List with some adjacent words swapped.

    Raises:
        ValueError: If words is None.
        ValueError: If num_swaps is negative.

    Examples:
        >>> random_swap(["a", "b", "c"], 0)
        ['a', 'b', 'c']

        >>> random_swap(["a"], 5)
        ['a']

        >>> random_swap(None, 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: words cannot be None

        >>> random_swap(["a", "b"], -1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_swaps cannot be negative
    """
    if words is None:
        msg = "words cannot be None"
        raise ValueError(msg)

    if num_swaps < 0:
        msg = f"num_swaps cannot be negative, got {num_swaps}"
        raise ValueError(msg)

    if len(words) < 2 or num_swaps == 0:
        return list(words)

    rng = rng or random.Random()
    result = list(words)

    for _ in range(num_swaps):
        idx = rng.randint(0, len(result) - 2)
        result[idx], result[idx + 1] = result[idx + 1], result[idx]

    return result


def random_insert(
    words: list[str],
    word_pool: list[str],
    num_inserts: int,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Randomly insert words from a pool.

    Args:
        words: List of words to modify.
        word_pool: Pool of words to insert from.
        num_inserts: Number of insertions to perform.
        rng: Random number generator. Defaults to None.

    Returns:
        List with additional words inserted.

    Raises:
        ValueError: If words is None.
        ValueError: If word_pool is None.
        ValueError: If num_inserts is negative.

    Examples:
        >>> rng = random.Random(42)
        >>> random_insert(["hello", "world"], ["new"], 1, rng=rng)
        ['new', 'hello', 'world']

        >>> random_insert(["a"], [], 5, rng=rng)
        ['a']

        >>> random_insert(None, ["x"], 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: words cannot be None

        >>> random_insert(["a"], None, 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: word_pool cannot be None
    """
    if words is None:
        msg = "words cannot be None"
        raise ValueError(msg)

    if word_pool is None:
        msg = "word_pool cannot be None"
        raise ValueError(msg)

    if num_inserts < 0:
        msg = f"num_inserts cannot be negative, got {num_inserts}"
        raise ValueError(msg)

    if not word_pool or num_inserts == 0:
        return list(words)

    rng = rng or random.Random()
    result = list(words)

    for _ in range(num_inserts):
        word = rng.choice(word_pool)
        pos = rng.randint(0, len(result))
        result.insert(pos, word)

    return result


def synonym_replace(
    words: list[str],
    synonyms: dict[str, list[str]],
    probability: float,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Replace words with synonyms.

    Args:
        words: List of words to process.
        synonyms: Mapping of words to their synonyms.
        probability: Probability of replacing each word.
        rng: Random number generator. Defaults to None.

    Returns:
        List with some words replaced by synonyms.

    Raises:
        ValueError: If words is None.
        ValueError: If synonyms is None.
        ValueError: If probability is not in [0, 1].

    Examples:
        >>> synonyms = {"hello": ["hi", "hey"], "world": ["earth"]}
        >>> rng = random.Random(42)
        >>> result = synonym_replace(["hello", "world"], synonyms, 1.0, rng=rng)
        >>> result[0] in ["hi", "hey", "hello"]
        True

        >>> synonym_replace(["a", "b"], {}, 1.0)
        ['a', 'b']

        >>> synonym_replace(None, {}, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: words cannot be None

        >>> synonym_replace(["a"], None, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: synonyms cannot be None
    """
    if words is None:
        msg = "words cannot be None"
        raise ValueError(msg)

    if synonyms is None:
        msg = "synonyms cannot be None"
        raise ValueError(msg)

    if not 0.0 <= probability <= 1.0:
        msg = f"probability must be between 0 and 1, got {probability}"
        raise ValueError(msg)

    if not words or not synonyms or probability == 0.0:
        return list(words)

    rng = rng or random.Random()
    result = []

    for word in words:
        lower_word = word.lower()
        if lower_word in synonyms and rng.random() < probability:
            replacement = rng.choice(synonyms[lower_word])
            # Preserve case if original was capitalized
            if word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(word)

    return result


def inject_noise(
    text: str,
    config: NoiseConfig,
    *,
    rng: random.Random | None = None,
) -> str:
    """Inject noise into text based on configuration.

    Args:
        text: Text to add noise to.
        config: Noise configuration.
        rng: Random number generator. Defaults to None.

    Returns:
        Text with noise injected.

    Raises:
        ValueError: If text is None.
        ValueError: If config is None.

    Examples:
        >>> config = NoiseConfig(
        ...     noise_type=NoiseType.CHARACTER,
        ...     probability=0.0,
        ...     preserve_semantics=True,
        ... )
        >>> inject_noise("hello world", config)
        'hello world'

        >>> inject_noise(None, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> inject_noise("hello", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    validate_not_none(config, "config")

    if config.probability == 0.0 or not text:
        return text

    rng = rng or random.Random()

    if config.noise_type == NoiseType.CHARACTER:
        return _inject_character_noise(text, config.probability, rng)
    elif config.noise_type == NoiseType.WORD:
        return _inject_word_noise(text, config.probability, rng)
    elif config.noise_type == NoiseType.KEYBOARD:
        return _inject_keyboard_noise(text, config.probability, rng)
    elif config.noise_type == NoiseType.OCR:
        return _inject_ocr_noise(text, config.probability, rng)

    return text


def _inject_character_noise(text: str, probability: float, rng: random.Random) -> str:
    """Inject character-level noise (typos, substitutions)."""
    chars = list(text)
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for i, char in enumerate(chars):
        if char.isalpha() and rng.random() < probability:
            is_upper = char.isupper()
            replacement = rng.choice(alphabet)
            chars[i] = replacement.upper() if is_upper else replacement

    return "".join(chars)


def _inject_word_noise(text: str, probability: float, rng: random.Random) -> str:
    """Inject word-level noise (drops, shuffles)."""
    words = text.split()
    if not words:
        return text

    result = []
    for word in words:
        word = _apply_word_noise(word, probability, rng, result)
    return " ".join(result) if result else text


def _apply_word_noise(
    word: str,
    probability: float,
    rng: random.Random,
    result: list[str],
) -> str:
    """Apply noise to a single word, appending to result."""
    if rng.random() < probability:
        if rng.random() < 0.5:
            return word  # Drop word (don't append)
        word = _shuffle_middle(word, rng)
    result.append(word)
    return word


def _shuffle_middle(word: str, rng: random.Random) -> str:
    """Shuffle middle characters of a word if long enough."""
    if len(word) > 3:
        middle = list(word[1:-1])
        rng.shuffle(middle)
        return word[0] + "".join(middle) + word[-1]
    return word


# Keyboard layout for typo simulation
KEYBOARD_NEIGHBORS: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrd",
    "r": "etf",
    "t": "ryg",
    "y": "tuh",
    "u": "yij",
    "i": "uok",
    "o": "ipl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _inject_keyboard_noise(text: str, probability: float, rng: random.Random) -> str:
    """Inject keyboard-based typos (nearby key substitution)."""
    chars = list(text)

    for i, char in enumerate(chars):
        lower_char = char.lower()
        if lower_char in KEYBOARD_NEIGHBORS and rng.random() < probability:
            neighbors = KEYBOARD_NEIGHBORS[lower_char]
            replacement = rng.choice(neighbors)
            chars[i] = replacement.upper() if char.isupper() else replacement

    return "".join(chars)


# OCR confusion pairs (characters that look similar)
OCR_CONFUSIONS: dict[str, str] = {
    "0": "oO",
    "o": "0O",
    "O": "0o",
    "1": "lI",
    "l": "1I",
    "I": "1l",
    "5": "S",
    "S": "5",
    "8": "B",
    "B": "8",
    "g": "9q",
    "9": "g",
    "q": "g",
    "c": "e",
    "e": "c",
    "m": "rn",
    "n": "r",
    "r": "n",
}


def _inject_ocr_noise(text: str, probability: float, rng: random.Random) -> str:
    """Inject OCR-like errors (similar-looking character substitution)."""
    chars = list(text)

    for i, char in enumerate(chars):
        if char in OCR_CONFUSIONS and rng.random() < probability:
            replacement = rng.choice(OCR_CONFUSIONS[char])
            chars[i] = replacement

    return "".join(chars)


def augment_text(
    text: str,
    config: AugmentConfig | None = None,
    *,
    synonyms: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> AugmentResult:
    """Augment text using configured augmentation strategy.

    Args:
        text: Text to augment.
        config: Augmentation configuration. Defaults to None.
        synonyms: Optional synonym dictionary for synonym replacement.
        rng: Random number generator. Defaults to None.

    Returns:
        AugmentResult containing original and augmented texts.

    Raises:
        ValueError: If text is None.

    Examples:
        >>> cfg = AugmentConfig(augmentation_type=AugmentationType.NONE)
        >>> result = augment_text("hello world", cfg)
        >>> result.original
        'hello world'
        >>> result.augmented
        ['hello world']

        >>> augment_text(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    # Apply NFC normalization for consistent Unicode handling
    text = unicodedata.normalize("NFC", text)

    effective_config = config or AugmentConfig()

    if effective_config.augmentation_type == AugmentationType.NONE:
        return AugmentResult(
            original=text,
            augmented=[text] * effective_config.num_augmentations,
            operations_applied=[0] * effective_config.num_augmentations,
        )

    if len(text.split()) < effective_config.min_length:
        return AugmentResult(
            original=text,
            augmented=[text] * effective_config.num_augmentations,
            operations_applied=[0] * effective_config.num_augmentations,
        )

    rng = rng or random.Random()
    augmented = []
    operations = []

    for _ in range(effective_config.num_augmentations):
        words = text.split()
        ops_count = 0

        if effective_config.augmentation_type == AugmentationType.RANDOM_DELETE:
            original_len = len(words)
            words = random_delete(words, effective_config.probability, rng=rng)
            ops_count = original_len - len(words)

        elif effective_config.augmentation_type == AugmentationType.RANDOM_SWAP:
            num_swaps = max(1, int(len(words) * effective_config.probability))
            words = random_swap(words, num_swaps, rng=rng)
            ops_count = num_swaps

        elif effective_config.augmentation_type == AugmentationType.SYNONYM_REPLACE:
            if synonyms:
                original_words = list(words)
                words = synonym_replace(
                    words, synonyms, effective_config.probability, rng=rng
                )
                ops_count = sum(
                    1 for o, n in zip(original_words, words, strict=True) if o != n
                )

        elif effective_config.augmentation_type == AugmentationType.RANDOM_INSERT:
            num_inserts = max(1, int(len(words) * effective_config.probability))
            words = random_insert(words, words, num_inserts, rng=rng)
            ops_count = num_inserts

        elif effective_config.augmentation_type == AugmentationType.NOISE:
            noise_config = NoiseConfig(
                noise_type=NoiseType.CHARACTER,
                probability=effective_config.probability,
                preserve_semantics=True,
            )
            augmented_text = inject_noise(" ".join(words), noise_config, rng=rng)
            words = augmented_text.split()
            ops_count = 1

        augmented.append(" ".join(words))
        operations.append(ops_count)

    return AugmentResult(
        original=text,
        augmented=augmented,
        operations_applied=operations,
    )


def apply_augmentation(
    text: str,
    config: AugmentationConfig,
    *,
    synonyms: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Apply augmentation to text based on configuration.

    Args:
        text: Text to augment.
        config: Augmentation configuration.
        synonyms: Optional synonym dictionary.
        rng: Random number generator. Defaults to None.

    Returns:
        Augmented text.

    Raises:
        ValueError: If text is None.
        ValueError: If config is None.

    Examples:
        >>> config = AugmentationConfig(
        ...     aug_type=AugmentationType.NONE,
        ...     probability=0.2,
        ...     preserve_labels=True,
        ... )
        >>> apply_augmentation("hello world", config)
        'hello world'

        >>> apply_augmentation(None, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> apply_augmentation("hello", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    validate_not_none(config, "config")

    rng = rng or random.Random()

    # Check if augmentation should be applied based on probability
    if rng.random() > config.probability:
        return text

    if config.aug_type == AugmentationType.NONE:
        return text

    # Convert to AugmentConfig for reuse
    aug_config = AugmentConfig(
        probability=config.probability,
        num_augmentations=1,
        augmentation_type=config.aug_type,
        min_length=1,
        preserve_case=True,
    )

    result = augment_text(text, aug_config, synonyms=synonyms, rng=rng)
    return result.augmented[0]


def create_augmenter(
    config: AugmentConfig,
    *,
    synonyms: dict[str, list[str]] | None = None,
) -> Callable[[str], AugmentResult]:
    """Create a reusable augmenter function.

    Args:
        config: Augmentation configuration.
        synonyms: Optional synonym dictionary.

    Returns:
        Function that augments text with the given configuration.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        >>> augmenter = create_augmenter(config)
        >>> result = augmenter("hello world")
        >>> result.original
        'hello world'

        >>> create_augmenter(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_augment_config(config)

    def augmenter(text: str) -> AugmentResult:
        """Apply the configured augmentation pipeline to input text."""
        return augment_text(text, config, synonyms=synonyms)

    return augmenter


def augment_batch(
    texts: Sequence[str],
    config: AugmentConfig | None = None,
    *,
    synonyms: dict[str, list[str]] | None = None,
) -> list[AugmentResult]:
    """Augment a batch of texts.

    Args:
        texts: Sequence of texts to augment.
        config: Augmentation configuration. Defaults to None.
        synonyms: Optional synonym dictionary.

    Returns:
        List of AugmentResult for each input text.

    Raises:
        ValueError: If texts is None.

    Examples:
        >>> config = AugmentConfig(augmentation_type=AugmentationType.NONE)
        >>> results = augment_batch(["hello", "world"], config)
        >>> len(results)
        2

        >>> augment_batch(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: texts cannot be None
    """
    if texts is None:
        msg = "texts cannot be None"
        raise ValueError(msg)

    return [augment_text(text, config, synonyms=synonyms) for text in texts]


def chain_augmentations(
    text: str,
    augmentation_types: list[AugmentationType],
    probability: float = 0.1,
    *,
    synonyms: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Apply multiple augmentation types in sequence.

    Args:
        text: Text to augment.
        augmentation_types: List of augmentation types to apply.
        probability: Probability for each augmentation. Defaults to 0.1.
        synonyms: Optional synonym dictionary.
        rng: Random number generator. Defaults to None.

    Returns:
        Augmented text after all operations.

    Raises:
        ValueError: If text is None.
        ValueError: If augmentation_types is None.

    Examples:
        >>> result = chain_augmentations("hello world test", [AugmentationType.NONE])
        >>> result
        'hello world test'

        >>> chain_augmentations(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> chain_augmentations("test", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: augmentation_types cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if augmentation_types is None:
        msg = "augmentation_types cannot be None"
        raise ValueError(msg)

    result = text

    for aug_type in augmentation_types:
        config = AugmentConfig(
            probability=probability,
            num_augmentations=1,
            augmentation_type=aug_type,
        )
        aug_result = augment_text(result, config, synonyms=synonyms, rng=rng)
        result = aug_result.augmented[0]

    return result


def compute_augmentation_stats(
    results: list[AugmentResult],
) -> dict[str, Any]:
    """Compute statistics from augmentation results.

    Args:
        results: List of augmentation results.

    Returns:
        Dictionary with augmentation statistics.

    Raises:
        ValueError: If results is None.

    Examples:
        >>> results = [
        ...     AugmentResult("a", ["b"], [1]),
        ...     AugmentResult("c", ["d"], [2]),
        ... ]
        >>> stats = compute_augmentation_stats(results)
        >>> stats["total_texts"]
        2
        >>> stats["total_operations"]
        3

        >>> compute_augmentation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be None
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if not results:
        return {
            "total_texts": 0,
            "total_augmented": 0,
            "total_operations": 0,
            "avg_operations_per_text": 0.0,
        }

    total_ops = sum(sum(r.operations_applied) for r in results)
    total_augmented = sum(len(r.augmented) for r in results)

    return {
        "total_texts": len(results),
        "total_augmented": total_augmented,
        "total_operations": total_ops,
        "avg_operations_per_text": total_ops / len(results) if results else 0.0,
    }


def calculate_augmentation_factor(
    num_augmentations: int,
    num_techniques: int,
    chain_probability: float = 1.0,
) -> float:
    """Calculate the data multiplication factor from augmentation.

    Args:
        num_augmentations: Number of augmented versions per sample.
        num_techniques: Number of augmentation techniques applied.
        chain_probability: Probability of applying chained augmentations.

    Returns:
        Factor by which dataset size increases.

    Raises:
        ValueError: If num_augmentations is not positive.
        ValueError: If num_techniques is negative.
        ValueError: If chain_probability is not in [0, 1].

    Examples:
        >>> calculate_augmentation_factor(3, 2)
        7.0
        >>> calculate_augmentation_factor(1, 0)
        1.0
        >>> calculate_augmentation_factor(2, 1, chain_probability=0.5)
        2.0

        >>> calculate_augmentation_factor(0, 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_augmentations must be positive

        >>> calculate_augmentation_factor(1, -1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_techniques cannot be negative
    """
    if num_augmentations <= 0:
        msg = f"num_augmentations must be positive, got {num_augmentations}"
        raise ValueError(msg)

    if num_techniques < 0:
        msg = f"num_techniques cannot be negative, got {num_techniques}"
        raise ValueError(msg)

    if not 0.0 <= chain_probability <= 1.0:
        msg = f"chain_probability must be between 0 and 1, got {chain_probability}"
        raise ValueError(msg)

    if num_techniques == 0:
        return 1.0

    # Factor = 1 (original) + num_augmentations * num_techniques * chain_probability
    augmented_factor = num_augmentations * num_techniques * chain_probability
    return 1.0 + augmented_factor


def estimate_diversity_gain(
    results: list[AugmentResult],
) -> float:
    """Estimate the diversity gain from augmentation.

    Calculates a score between 0 and 1 representing how much
    diversity the augmentation added to the dataset.

    Args:
        results: List of augmentation results.

    Returns:
        Diversity score between 0 (no diversity) and 1 (maximum diversity).

    Raises:
        ValueError: If results is None.

    Examples:
        >>> results = [
        ...     AugmentResult("hello", ["hi", "hey"], [1, 1]),
        ...     AugmentResult("world", ["earth", "globe"], [1, 1]),
        ... ]
        >>> score = estimate_diversity_gain(results)
        >>> 0.0 <= score <= 1.0
        True

        >>> results_same = [
        ...     AugmentResult("hello", ["hello", "hello"], [0, 0]),
        ... ]
        >>> estimate_diversity_gain(results_same)
        0.0

        >>> estimate_diversity_gain(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be None
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if not results:
        return 0.0

    total_pairs = 0
    different_pairs = 0

    for result in results:
        # Compare original with each augmented version
        for augmented in result.augmented:
            total_pairs += 1
            if augmented != result.original:
                different_pairs += 1

    return different_pairs / total_pairs if total_pairs > 0 else 0.0


def format_augmentation_stats(stats: AugmentationStats) -> str:
    """Format augmentation statistics as a human-readable string.

    Args:
        stats: AugmentationStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = AugmentationStats(
        ...     total_texts=100,
        ...     total_augmented=300,
        ...     total_operations=450,
        ...     avg_operations_per_text=4.5,
        ...     diversity_score=0.75,
        ... )
        >>> formatted = format_augmentation_stats(stats)
        >>> "100" in formatted
        True
        >>> "300" in formatted
        True

        >>> format_augmentation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    augmentation_factor = (
        stats.total_augmented / stats.total_texts if stats.total_texts > 0 else 0.0
    )

    lines = [
        "Augmentation Statistics",
        "=" * 40,
        f"Total texts:          {stats.total_texts:,}",
        f"Total augmented:      {stats.total_augmented:,}",
        f"Total operations:     {stats.total_operations:,}",
        f"Avg ops per text:     {stats.avg_operations_per_text:.2f}",
        f"Augmentation factor:  {augmentation_factor:.2f}x",
        f"Diversity score:      {stats.diversity_score:.2%}",
    ]

    return "\n".join(lines)


def get_recommended_augmentation_config(
    task_type: str,
    dataset_size: int,
) -> AugmentationConfig:
    """Get recommended augmentation configuration based on task and data.

    Args:
        task_type: Type of NLP task (classification, ner, qa, generation).
        dataset_size: Number of samples in the dataset.

    Returns:
        Recommended AugmentationConfig for the task.

    Raises:
        ValueError: If task_type is empty.
        ValueError: If dataset_size is not positive.

    Examples:
        >>> config = get_recommended_augmentation_config("classification", 1000)
        >>> config.preserve_labels
        True

        >>> config = get_recommended_augmentation_config("generation", 100000)
        >>> 0.0 <= config.probability <= 1.0
        True

        >>> get_recommended_augmentation_config("", 1000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_type cannot be empty

        >>> get_recommended_augmentation_config("classification", 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_size must be positive
    """
    if not task_type:
        msg = "task_type cannot be empty"
        raise ValueError(msg)

    if dataset_size <= 0:
        msg = f"dataset_size must be positive, got {dataset_size}"
        raise ValueError(msg)

    task_type_lower = task_type.lower()

    # Small datasets benefit more from augmentation
    if dataset_size < 1000:
        probability = 0.3
    elif dataset_size < 10000:
        probability = 0.2
    else:
        probability = 0.1

    # Task-specific recommendations
    if task_type_lower in ("classification", "sentiment"):
        # Synonym replacement works well for classification
        return AugmentationConfig(
            aug_type=AugmentationType.SYNONYM_REPLACE,
            probability=probability,
            preserve_labels=True,
        )
    elif task_type_lower in ("ner", "pos", "tagging"):
        # Be careful with NER - prefer conservative augmentation
        return AugmentationConfig(
            aug_type=AugmentationType.RANDOM_SWAP,
            probability=probability * 0.5,  # More conservative
            preserve_labels=True,
        )
    elif task_type_lower in ("qa", "question_answering"):
        # Back-translation is good for QA
        return AugmentationConfig(
            aug_type=AugmentationType.BACK_TRANSLATE,
            probability=probability,
            preserve_labels=True,
        )
    elif task_type_lower in ("generation", "lm", "language_model"):
        # Various augmentations for generation
        return AugmentationConfig(
            aug_type=AugmentationType.RANDOM_INSERT,
            probability=probability,
            preserve_labels=False,
        )
    else:
        # Default: synonym replacement with label preservation
        return AugmentationConfig(
            aug_type=AugmentationType.SYNONYM_REPLACE,
            probability=probability,
            preserve_labels=True,
        )
