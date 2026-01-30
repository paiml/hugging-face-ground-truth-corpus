"""Data augmentation utilities for text preprocessing.

This module provides text augmentation techniques for training data
enhancement, including synonym replacement, random operations, and
back-translation patterns.

Examples:
    >>> from hf_gtc.preprocessing.augmentation import AugmentConfig
    >>> config = AugmentConfig(probability=0.2)
    >>> config.probability
    0.2
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class AugmentationType(Enum):
    """Types of text augmentation operations.

    Attributes:
        SYNONYM_REPLACE: Replace words with synonyms.
        RANDOM_INSERT: Insert random words.
        RANDOM_SWAP: Swap adjacent words.
        RANDOM_DELETE: Delete random words.
        BACK_TRANSLATE: Back-translation augmentation.
        NONE: No augmentation.

    Examples:
        >>> AugmentationType.SYNONYM_REPLACE.value
        'synonym_replace'
        >>> AugmentationType.NONE.value
        'none'
    """

    SYNONYM_REPLACE = "synonym_replace"
    RANDOM_INSERT = "random_insert"
    RANDOM_SWAP = "random_swap"
    RANDOM_DELETE = "random_delete"
    BACK_TRANSLATE = "back_translate"
    NONE = "none"


VALID_AUGMENTATION_TYPES = frozenset(t.value for t in AugmentationType)


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
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 0.0 <= config.probability <= 1.0:
        msg = f"probability must be between 0 and 1, got {config.probability}"
        raise ValueError(msg)

    if config.num_augmentations <= 0:
        msg = f"num_augmentations must be positive, got {config.num_augmentations}"
        raise ValueError(msg)

    if config.min_length < 0:
        msg = f"min_length cannot be negative, got {config.min_length}"
        raise ValueError(msg)


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

        augmented.append(" ".join(words))
        operations.append(ops_count)

    return AugmentResult(
        original=text,
        augmented=augmented,
        operations_applied=operations,
    )


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
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_augment_config(config)

    def augmenter(text: str) -> AugmentResult:
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
