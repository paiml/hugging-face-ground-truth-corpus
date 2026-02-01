"""Data quality utilities for preprocessing pipelines.

This module provides utilities for data quality assessment, deduplication,
contamination detection, and quality filtering for ML datasets.

Examples:
    >>> from hf_gtc.preprocessing.quality import DeduplicationConfig
    >>> config = DeduplicationConfig(
    ...     method=DeduplicationMethod.EXACT_HASH,
    ...     similarity_threshold=0.9,
    ...     ngram_size=5,
    ...     num_hashes=128,
    ... )
    >>> config.similarity_threshold
    0.9
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from hf_gtc._validation import validate_not_none


class DeduplicationMethod(Enum):
    """Methods for detecting and removing duplicates.

    Attributes:
        EXACT_HASH: Exact hash-based deduplication.
        MINHASH: MinHash LSH for near-duplicate detection.
        SIMHASH: SimHash fingerprinting.
        SEMANTIC: Semantic similarity-based deduplication.
        NGRAM: N-gram overlap deduplication.

    Examples:
        >>> DeduplicationMethod.EXACT_HASH.value
        'exact_hash'
        >>> DeduplicationMethod.MINHASH.value
        'minhash'
    """

    EXACT_HASH = "exact_hash"
    MINHASH = "minhash"
    SIMHASH = "simhash"
    SEMANTIC = "semantic"
    NGRAM = "ngram"


VALID_DEDUPLICATION_METHODS = frozenset(m.value for m in DeduplicationMethod)


class QualityMetric(Enum):
    """Metrics for assessing text quality.

    Attributes:
        PERPLEXITY: Language model perplexity score.
        LENGTH: Text length metrics.
        REPETITION: Repetition detection score.
        TOXICITY: Toxicity detection score.
        LANGUAGE_SCORE: Language identification confidence.

    Examples:
        >>> QualityMetric.PERPLEXITY.value
        'perplexity'
        >>> QualityMetric.TOXICITY.value
        'toxicity'
    """

    PERPLEXITY = "perplexity"
    LENGTH = "length"
    REPETITION = "repetition"
    TOXICITY = "toxicity"
    LANGUAGE_SCORE = "language_score"


VALID_QUALITY_METRICS = frozenset(m.value for m in QualityMetric)


class FilterStrategy(Enum):
    """Strategies for filtering samples based on quality scores.

    Attributes:
        THRESHOLD: Filter using fixed threshold values.
        PERCENTILE: Filter using percentile cutoffs.
        ZSCORE: Filter using z-score outlier detection.

    Examples:
        >>> FilterStrategy.THRESHOLD.value
        'threshold'
        >>> FilterStrategy.PERCENTILE.value
        'percentile'
    """

    THRESHOLD = "threshold"
    PERCENTILE = "percentile"
    ZSCORE = "zscore"


VALID_FILTER_STRATEGIES = frozenset(s.value for s in FilterStrategy)


@dataclass(frozen=True, slots=True)
class DeduplicationConfig:
    """Configuration for deduplication operations.

    Attributes:
        method: Deduplication method to use.
        similarity_threshold: Threshold for near-duplicate detection (0.0-1.0).
        ngram_size: Size of n-grams for n-gram based methods.
        num_hashes: Number of hash functions for MinHash.

    Examples:
        >>> config = DeduplicationConfig(
        ...     method=DeduplicationMethod.MINHASH,
        ...     similarity_threshold=0.8,
        ...     ngram_size=3,
        ...     num_hashes=64,
        ... )
        >>> config.method
        <DeduplicationMethod.MINHASH: 'minhash'>
        >>> config.num_hashes
        64
    """

    method: DeduplicationMethod
    similarity_threshold: float
    ngram_size: int
    num_hashes: int


@dataclass(frozen=True, slots=True)
class QualityFilterConfig:
    """Configuration for quality-based filtering.

    Attributes:
        metrics: Tuple of quality metrics to evaluate.
        thresholds: Mapping of metric names to threshold values.
        filter_strategy: Strategy for applying filters.
        remove_outliers: Whether to remove statistical outliers.

    Examples:
        >>> config = QualityFilterConfig(
        ...     metrics=(QualityMetric.PERPLEXITY, QualityMetric.LENGTH),
        ...     thresholds={"perplexity": 100.0, "length": 10},
        ...     filter_strategy=FilterStrategy.THRESHOLD,
        ...     remove_outliers=True,
        ... )
        >>> config.filter_strategy
        <FilterStrategy.THRESHOLD: 'threshold'>
        >>> config.remove_outliers
        True
    """

    metrics: tuple[QualityMetric, ...]
    thresholds: dict[str, float]
    filter_strategy: FilterStrategy
    remove_outliers: bool


@dataclass(frozen=True, slots=True)
class ContaminationConfig:
    """Configuration for test set contamination detection.

    Attributes:
        test_datasets: Tuple of test dataset identifiers.
        ngram_overlap_threshold: N-gram overlap threshold for contamination.
        exact_match: Whether to check for exact matches.

    Examples:
        >>> config = ContaminationConfig(
        ...     test_datasets=("squad", "trivia_qa"),
        ...     ngram_overlap_threshold=0.8,
        ...     exact_match=True,
        ... )
        >>> config.exact_match
        True
        >>> len(config.test_datasets)
        2
    """

    test_datasets: tuple[str, ...]
    ngram_overlap_threshold: float
    exact_match: bool


@dataclass(frozen=True, slots=True)
class QualityStats:
    """Statistics from quality assessment operations.

    Attributes:
        total_samples: Total number of samples processed.
        filtered_count: Number of samples filtered out.
        duplicate_count: Number of duplicates detected.
        quality_distribution: Distribution of quality scores.

    Examples:
        >>> stats = QualityStats(
        ...     total_samples=10000,
        ...     filtered_count=500,
        ...     duplicate_count=200,
        ...     quality_distribution={"low": 100, "medium": 400, "high": 9500},
        ... )
        >>> stats.total_samples
        10000
        >>> stats.filtered_count
        500
    """

    total_samples: int
    filtered_count: int
    duplicate_count: int
    quality_distribution: dict[str, int]


def validate_deduplication_config(config: DeduplicationConfig) -> None:
    """Validate deduplication configuration.

    Args:
        config: DeduplicationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If similarity_threshold is not in [0, 1].
        ValueError: If ngram_size is not positive.
        ValueError: If num_hashes is not positive.

    Examples:
        >>> config = DeduplicationConfig(
        ...     method=DeduplicationMethod.MINHASH,
        ...     similarity_threshold=0.8,
        ...     ngram_size=3,
        ...     num_hashes=64,
        ... )
        >>> validate_deduplication_config(config)  # No error

        >>> validate_deduplication_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = DeduplicationConfig(
        ...     method=DeduplicationMethod.MINHASH,
        ...     similarity_threshold=1.5,
        ...     ngram_size=3,
        ...     num_hashes=64,
        ... )
        >>> validate_deduplication_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: similarity_threshold must be between 0 and 1
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.similarity_threshold <= 1.0:
        msg = (
            f"similarity_threshold must be between 0 and 1, "
            f"got {config.similarity_threshold}"
        )
        raise ValueError(msg)

    if config.ngram_size <= 0:
        msg = f"ngram_size must be positive, got {config.ngram_size}"
        raise ValueError(msg)

    if config.num_hashes <= 0:
        msg = f"num_hashes must be positive, got {config.num_hashes}"
        raise ValueError(msg)


def validate_quality_filter_config(config: QualityFilterConfig) -> None:
    """Validate quality filter configuration.

    Args:
        config: QualityFilterConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If metrics is empty.
        ValueError: If thresholds is None.

    Examples:
        >>> config = QualityFilterConfig(
        ...     metrics=(QualityMetric.PERPLEXITY,),
        ...     thresholds={"perplexity": 100.0},
        ...     filter_strategy=FilterStrategy.THRESHOLD,
        ...     remove_outliers=False,
        ... )
        >>> validate_quality_filter_config(config)  # No error

        >>> validate_quality_filter_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = QualityFilterConfig(
        ...     metrics=(),
        ...     thresholds={},
        ...     filter_strategy=FilterStrategy.THRESHOLD,
        ...     remove_outliers=False,
        ... )
        >>> validate_quality_filter_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metrics cannot be empty
    """
    validate_not_none(config, "config")

    if not config.metrics:
        msg = "metrics cannot be empty"
        raise ValueError(msg)

    if config.thresholds is None:
        msg = "thresholds cannot be None"
        raise ValueError(msg)


def validate_contamination_config(config: ContaminationConfig) -> None:
    """Validate contamination detection configuration.

    Args:
        config: ContaminationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If test_datasets is empty.
        ValueError: If ngram_overlap_threshold is not in [0, 1].

    Examples:
        >>> config = ContaminationConfig(
        ...     test_datasets=("squad",),
        ...     ngram_overlap_threshold=0.8,
        ...     exact_match=True,
        ... )
        >>> validate_contamination_config(config)  # No error

        >>> validate_contamination_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ContaminationConfig(
        ...     test_datasets=(),
        ...     ngram_overlap_threshold=0.8,
        ...     exact_match=True,
        ... )
        >>> validate_contamination_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: test_datasets cannot be empty
    """
    validate_not_none(config, "config")

    if not config.test_datasets:
        msg = "test_datasets cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.ngram_overlap_threshold <= 1.0:
        msg = (
            f"ngram_overlap_threshold must be between 0 and 1, "
            f"got {config.ngram_overlap_threshold}"
        )
        raise ValueError(msg)


def create_deduplication_config(
    method: str = "exact_hash",
    similarity_threshold: float = 0.9,
    ngram_size: int = 5,
    num_hashes: int = 128,
) -> DeduplicationConfig:
    """Create a deduplication configuration.

    Args:
        method: Deduplication method name. Defaults to "exact_hash".
        similarity_threshold: Similarity threshold. Defaults to 0.9.
        ngram_size: N-gram size. Defaults to 5.
        num_hashes: Number of hash functions. Defaults to 128.

    Returns:
        DeduplicationConfig with the specified settings.

    Raises:
        ValueError: If method is not valid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_deduplication_config(method="minhash")
        >>> config.method
        <DeduplicationMethod.MINHASH: 'minhash'>

        >>> create_deduplication_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of

        >>> create_deduplication_config(ngram_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ngram_size must be positive
    """
    if method not in VALID_DEDUPLICATION_METHODS:
        msg = f"method must be one of {VALID_DEDUPLICATION_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = DeduplicationConfig(
        method=DeduplicationMethod(method),
        similarity_threshold=similarity_threshold,
        ngram_size=ngram_size,
        num_hashes=num_hashes,
    )
    validate_deduplication_config(config)
    return config


def create_quality_filter_config(
    metrics: tuple[str, ...] = ("perplexity", "length"),
    thresholds: dict[str, float] | None = None,
    filter_strategy: str = "threshold",
    remove_outliers: bool = True,
) -> QualityFilterConfig:
    """Create a quality filter configuration.

    Args:
        metrics: Tuple of metric names. Defaults to ("perplexity", "length").
        thresholds: Metric thresholds. Defaults to None (uses defaults).
        filter_strategy: Filter strategy name. Defaults to "threshold".
        remove_outliers: Whether to remove outliers. Defaults to True.

    Returns:
        QualityFilterConfig with the specified settings.

    Raises:
        ValueError: If any metric is invalid.
        ValueError: If filter_strategy is invalid.

    Examples:
        >>> config = create_quality_filter_config(metrics=("perplexity",))
        >>> config.metrics[0]
        <QualityMetric.PERPLEXITY: 'perplexity'>

        >>> create_quality_filter_config(metrics=("invalid",))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric must be one of

        >>> create_quality_filter_config(filter_strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: filter_strategy must be one of
    """
    for metric in metrics:
        if metric not in VALID_QUALITY_METRICS:
            msg = f"metric must be one of {VALID_QUALITY_METRICS}, got '{metric}'"
            raise ValueError(msg)

    if filter_strategy not in VALID_FILTER_STRATEGIES:
        msg = (
            f"filter_strategy must be one of {VALID_FILTER_STRATEGIES}, "
            f"got '{filter_strategy}'"
        )
        raise ValueError(msg)

    # Default thresholds
    effective_thresholds = thresholds or {
        "perplexity": 1000.0,
        "length": 10,
        "repetition": 0.5,
        "toxicity": 0.5,
        "language_score": 0.8,
    }

    metric_enums = tuple(QualityMetric(m) for m in metrics)

    config = QualityFilterConfig(
        metrics=metric_enums,
        thresholds=effective_thresholds,
        filter_strategy=FilterStrategy(filter_strategy),
        remove_outliers=remove_outliers,
    )
    validate_quality_filter_config(config)
    return config


def create_contamination_config(
    test_datasets: tuple[str, ...],
    ngram_overlap_threshold: float = 0.8,
    exact_match: bool = True,
) -> ContaminationConfig:
    """Create a contamination detection configuration.

    Args:
        test_datasets: Tuple of test dataset identifiers.
        ngram_overlap_threshold: N-gram overlap threshold. Defaults to 0.8.
        exact_match: Whether to check exact matches. Defaults to True.

    Returns:
        ContaminationConfig with the specified settings.

    Raises:
        ValueError: If test_datasets is empty.
        ValueError: If ngram_overlap_threshold is invalid.

    Examples:
        >>> config = create_contamination_config(test_datasets=("squad", "nq"))
        >>> len(config.test_datasets)
        2

        >>> create_contamination_config(test_datasets=())
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: test_datasets cannot be empty

        >>> create_contamination_config(
        ...     test_datasets=("a",), ngram_overlap_threshold=1.5
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ngram_overlap_threshold must be between 0 and 1
    """
    config = ContaminationConfig(
        test_datasets=test_datasets,
        ngram_overlap_threshold=ngram_overlap_threshold,
        exact_match=exact_match,
    )
    validate_contamination_config(config)
    return config


def list_deduplication_methods() -> list[str]:
    """List all available deduplication methods.

    Returns:
        Sorted list of deduplication method names.

    Examples:
        >>> methods = list_deduplication_methods()
        >>> "exact_hash" in methods
        True
        >>> "minhash" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_DEDUPLICATION_METHODS)


def get_deduplication_method(name: str) -> DeduplicationMethod:
    """Get DeduplicationMethod enum from string name.

    Args:
        name: Name of the deduplication method.

    Returns:
        Corresponding DeduplicationMethod enum value.

    Raises:
        ValueError: If name is not a valid deduplication method.

    Examples:
        >>> get_deduplication_method("exact_hash")
        <DeduplicationMethod.EXACT_HASH: 'exact_hash'>

        >>> get_deduplication_method("minhash")
        <DeduplicationMethod.MINHASH: 'minhash'>

        >>> get_deduplication_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid deduplication method: invalid
    """
    if name not in VALID_DEDUPLICATION_METHODS:
        msg = f"invalid deduplication method: {name}"
        raise ValueError(msg)

    return DeduplicationMethod(name)


def list_quality_metrics() -> list[str]:
    """List all available quality metrics.

    Returns:
        Sorted list of quality metric names.

    Examples:
        >>> metrics = list_quality_metrics()
        >>> "perplexity" in metrics
        True
        >>> "toxicity" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_QUALITY_METRICS)


def get_quality_metric(name: str) -> QualityMetric:
    """Get QualityMetric enum from string name.

    Args:
        name: Name of the quality metric.

    Returns:
        Corresponding QualityMetric enum value.

    Raises:
        ValueError: If name is not a valid quality metric.

    Examples:
        >>> get_quality_metric("perplexity")
        <QualityMetric.PERPLEXITY: 'perplexity'>

        >>> get_quality_metric("toxicity")
        <QualityMetric.TOXICITY: 'toxicity'>

        >>> get_quality_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid quality metric: invalid
    """
    if name not in VALID_QUALITY_METRICS:
        msg = f"invalid quality metric: {name}"
        raise ValueError(msg)

    return QualityMetric(name)


def list_filter_strategies() -> list[str]:
    """List all available filter strategies.

    Returns:
        Sorted list of filter strategy names.

    Examples:
        >>> strategies = list_filter_strategies()
        >>> "threshold" in strategies
        True
        >>> "percentile" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_FILTER_STRATEGIES)


def get_filter_strategy(name: str) -> FilterStrategy:
    """Get FilterStrategy enum from string name.

    Args:
        name: Name of the filter strategy.

    Returns:
        Corresponding FilterStrategy enum value.

    Raises:
        ValueError: If name is not a valid filter strategy.

    Examples:
        >>> get_filter_strategy("threshold")
        <FilterStrategy.THRESHOLD: 'threshold'>

        >>> get_filter_strategy("zscore")
        <FilterStrategy.ZSCORE: 'zscore'>

        >>> get_filter_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid filter strategy: invalid
    """
    if name not in VALID_FILTER_STRATEGIES:
        msg = f"invalid filter strategy: {name}"
        raise ValueError(msg)

    return FilterStrategy(name)


def calculate_text_quality_score(
    text: str,
    *,
    metrics: Sequence[QualityMetric] | None = None,
) -> dict[str, float]:
    """Calculate quality scores for a text sample.

    Args:
        text: Text to evaluate.
        metrics: Metrics to calculate. Defaults to all metrics.

    Returns:
        Dictionary mapping metric names to scores.

    Raises:
        ValueError: If text is None.

    Examples:
        >>> scores = calculate_text_quality_score("Hello world, this is a test.")
        >>> "length" in scores
        True
        >>> "repetition" in scores
        True
        >>> scores["length"] > 0
        True

        >>> scores = calculate_text_quality_score("")
        >>> scores["length"]
        0.0
        >>> scores["repetition"]
        0.0

        >>> calculate_text_quality_score(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    # Apply NFC normalization for consistent Unicode handling
    text = unicodedata.normalize("NFC", text)

    effective_metrics = metrics or list(QualityMetric)

    scores: dict[str, float] = {}

    for metric in effective_metrics:
        if metric == QualityMetric.LENGTH:
            scores["length"] = float(len(text))
        elif metric == QualityMetric.REPETITION:
            scores["repetition"] = _calculate_repetition_score(text)
        elif metric == QualityMetric.LANGUAGE_SCORE:
            # Simplified: check for ASCII ratio as language confidence proxy
            scores["language_score"] = _calculate_language_score(text)
        elif metric == QualityMetric.PERPLEXITY:
            # Simplified perplexity estimate based on character entropy
            scores["perplexity"] = _estimate_perplexity(text)
        elif metric == QualityMetric.TOXICITY:
            # Placeholder: real implementation would use a toxicity model
            scores["toxicity"] = 0.0

    return scores


def _calculate_repetition_score(text: str) -> float:
    """Calculate repetition score for text.

    Returns value between 0 (no repetition) and 1 (high repetition).
    """
    if not text:
        return 0.0

    words = text.lower().split()
    if not words:
        return 0.0

    unique_words = set(words)
    return 1.0 - (len(unique_words) / len(words))


def _calculate_language_score(text: str) -> float:
    """Calculate language confidence score.

    Uses ASCII ratio as a simple proxy for English text.
    """
    if not text:
        return 0.0

    ascii_count = sum(1 for c in text if c.isascii())
    return ascii_count / len(text)


def _estimate_perplexity(text: str) -> float:
    """Estimate perplexity using character-level entropy.

    This is a simplified approximation - real perplexity requires a language model.
    """
    if not text:
        return 0.0

    # Character frequency distribution
    char_counts: dict[str, int] = {}
    for char in text.lower():
        char_counts[char] = char_counts.get(char, 0) + 1

    total = len(text)
    entropy = 0.0

    for count in char_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)

    # Convert entropy to pseudo-perplexity
    return 2**entropy


def detect_duplicates(
    texts: Sequence[str],
    config: DeduplicationConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Detect duplicate pairs in a collection of texts.

    Args:
        texts: Sequence of texts to check.
        config: Deduplication configuration. Defaults to exact hash.

    Returns:
        List of (index1, index2, similarity) tuples for detected duplicates.

    Raises:
        ValueError: If texts is None.

    Examples:
        >>> texts = ["hello world", "hello world", "different text"]
        >>> duplicates = detect_duplicates(texts)
        >>> len(duplicates) >= 1
        True
        >>> duplicates[0][2]  # Similarity of first duplicate pair
        1.0

        >>> detect_duplicates([])
        []

        >>> detect_duplicates(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: texts cannot be None
    """
    if texts is None:
        msg = "texts cannot be None"
        raise ValueError(msg)

    if len(texts) < 2:
        return []

    effective_config = config or create_deduplication_config()
    validate_deduplication_config(effective_config)

    duplicates: list[tuple[int, int, float]] = []

    if effective_config.method == DeduplicationMethod.EXACT_HASH:
        duplicates = _detect_exact_duplicates(texts)
    elif effective_config.method == DeduplicationMethod.NGRAM:
        duplicates = _detect_ngram_duplicates(
            texts,
            effective_config.ngram_size,
            effective_config.similarity_threshold,
        )
    else:
        # For other methods, fall back to exact hash
        duplicates = _detect_exact_duplicates(texts)

    return duplicates


def _detect_exact_duplicates(texts: Sequence[str]) -> list[tuple[int, int, float]]:
    """Detect exact duplicates using hash comparison."""
    hash_to_indices: dict[str, list[int]] = {}

    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        if text_hash not in hash_to_indices:
            hash_to_indices[text_hash] = []
        hash_to_indices[text_hash].append(i)

    duplicates: list[tuple[int, int, float]] = []

    for indices in hash_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    duplicates.append((indices[i], indices[j], 1.0))

    return duplicates


def _detect_ngram_duplicates(
    texts: Sequence[str],
    ngram_size: int,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Detect near-duplicates using n-gram overlap."""
    duplicates: list[tuple[int, int, float]] = []

    # Compute n-grams for all texts
    text_ngrams: list[set[tuple[str, ...]]] = []
    for text in texts:
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - ngram_size + 1):
            ngrams.add(tuple(words[i : i + ngram_size]))
        text_ngrams.append(ngrams)

    # Compare all pairs
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if not text_ngrams[i] or not text_ngrams[j]:
                continue

            intersection = len(text_ngrams[i] & text_ngrams[j])
            union = len(text_ngrams[i] | text_ngrams[j])
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= threshold:
                duplicates.append((i, j, similarity))

    return duplicates


def check_contamination(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    config: ContaminationConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Check for test set contamination in training data.

    Args:
        train_texts: Sequence of training texts.
        test_texts: Sequence of test texts.
        config: Contamination detection configuration.

    Returns:
        List of (train_index, test_index, overlap_score) tuples.

    Raises:
        ValueError: If train_texts is None.
        ValueError: If test_texts is None.

    Examples:
        >>> train = ["The quick brown fox", "Hello world"]
        >>> test = ["The quick brown fox", "Different text"]
        >>> contaminated = check_contamination(train, test)
        >>> len(contaminated) >= 1
        True

        >>> check_contamination([], [])
        []

        >>> check_contamination(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: train_texts cannot be None

        >>> check_contamination([], None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: test_texts cannot be None
    """
    if train_texts is None:
        msg = "train_texts cannot be None"
        raise ValueError(msg)

    if test_texts is None:
        msg = "test_texts cannot be None"
        raise ValueError(msg)

    if not train_texts or not test_texts:
        return []

    effective_config = config or create_contamination_config(
        test_datasets=("default",),
        ngram_overlap_threshold=0.8,
        exact_match=True,
    )

    contaminated: list[tuple[int, int, float]] = []

    # Check for exact matches first
    if effective_config.exact_match:
        train_hashes: dict[str, int] = {}
        for i, text in enumerate(train_texts):
            text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
            train_hashes[text_hash] = i

        for j, test_text in enumerate(test_texts):
            test_hash = hashlib.md5(
                test_text.encode(), usedforsecurity=False
            ).hexdigest()
            if test_hash in train_hashes:
                contaminated.append((train_hashes[test_hash], j, 1.0))

    # Check for n-gram overlap
    threshold = effective_config.ngram_overlap_threshold

    for i, train_text in enumerate(train_texts):
        train_words = set(train_text.lower().split())
        if not train_words:
            continue

        for j, test_text in enumerate(test_texts):
            test_words = set(test_text.lower().split())
            if not test_words:
                continue

            intersection = len(train_words & test_words)
            union = len(train_words | test_words)
            overlap = intersection / union if union > 0 else 0.0

            if overlap >= threshold and (i, j, 1.0) not in contaminated:
                contaminated.append((i, j, overlap))

    return contaminated


def calculate_perplexity_score(
    text: str,
    *,
    vocabulary_size: int = 50000,
) -> float:
    """Calculate a perplexity-like score for text.

    This is a simplified approximation based on character entropy.
    Real perplexity calculation requires a trained language model.

    Args:
        text: Text to evaluate.
        vocabulary_size: Assumed vocabulary size for scaling. Defaults to 50000.

    Returns:
        Perplexity-like score (lower is better quality).

    Raises:
        ValueError: If text is None.
        ValueError: If vocabulary_size is not positive.

    Examples:
        >>> score = calculate_perplexity_score("Hello world, this is a test.")
        >>> score > 0
        True

        >>> calculate_perplexity_score("")
        0.0

        >>> calculate_perplexity_score(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> calculate_perplexity_score("test", vocabulary_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocabulary_size must be positive
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if vocabulary_size <= 0:
        msg = f"vocabulary_size must be positive, got {vocabulary_size}"
        raise ValueError(msg)

    if not text:
        return 0.0

    return _estimate_perplexity(text)


def format_quality_stats(stats: QualityStats) -> str:
    """Format quality statistics as a human-readable string.

    Args:
        stats: QualityStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = QualityStats(
        ...     total_samples=10000,
        ...     filtered_count=500,
        ...     duplicate_count=200,
        ...     quality_distribution={"high": 9000, "low": 1000},
        ... )
        >>> formatted = format_quality_stats(stats)
        >>> "10,000" in formatted
        True
        >>> "500" in formatted
        True

        >>> format_quality_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    retention_rate = (
        (stats.total_samples - stats.filtered_count) / stats.total_samples * 100
        if stats.total_samples > 0
        else 0.0
    )

    duplicate_rate = (
        stats.duplicate_count / stats.total_samples * 100
        if stats.total_samples > 0
        else 0.0
    )

    lines = [
        "Quality Statistics",
        "=" * 40,
        f"Total samples:    {stats.total_samples:,}",
        f"Filtered count:   {stats.filtered_count:,}",
        f"Duplicate count:  {stats.duplicate_count:,}",
        f"Retention rate:   {retention_rate:.1f}%",
        f"Duplicate rate:   {duplicate_rate:.1f}%",
        "",
        "Quality Distribution:",
    ]

    for level, count in sorted(stats.quality_distribution.items()):
        percentage = count / stats.total_samples * 100 if stats.total_samples > 0 else 0
        lines.append(f"  {level}: {count:,} ({percentage:.1f}%)")

    return "\n".join(lines)


def get_recommended_quality_config(dataset_size: int) -> QualityFilterConfig:
    """Get recommended quality configuration based on dataset size.

    Args:
        dataset_size: Number of samples in the dataset.

    Returns:
        Recommended QualityFilterConfig for the dataset size.

    Raises:
        ValueError: If dataset_size is not positive.

    Examples:
        >>> config = get_recommended_quality_config(1000)
        >>> config.remove_outliers
        True

        >>> config = get_recommended_quality_config(1000000)
        >>> FilterStrategy.PERCENTILE in [config.filter_strategy]
        True

        >>> get_recommended_quality_config(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_size must be positive

        >>> get_recommended_quality_config(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_size must be positive
    """
    if dataset_size <= 0:
        msg = f"dataset_size must be positive, got {dataset_size}"
        raise ValueError(msg)

    # Small datasets: be conservative with filtering
    if dataset_size < 10000:
        return create_quality_filter_config(
            metrics=("length", "repetition"),
            thresholds={"length": 5, "repetition": 0.8},
            filter_strategy="threshold",
            remove_outliers=True,
        )

    # Medium datasets: standard filtering
    if dataset_size < 100000:
        return create_quality_filter_config(
            metrics=("perplexity", "length", "repetition"),
            thresholds={"perplexity": 500.0, "length": 10, "repetition": 0.6},
            filter_strategy="threshold",
            remove_outliers=True,
        )

    # Large datasets: use percentile-based filtering for robustness
    return create_quality_filter_config(
        metrics=("perplexity", "length", "repetition", "language_score"),
        thresholds={"perplexity": 95.0, "length": 5.0, "repetition": 95.0},
        filter_strategy="percentile",
        remove_outliers=True,
    )


def compute_quality_stats(
    texts: Sequence[str],
    config: QualityFilterConfig | None = None,
) -> QualityStats:
    """Compute quality statistics for a collection of texts.

    Args:
        texts: Sequence of texts to analyze.
        config: Quality filter configuration.

    Returns:
        QualityStats with computed metrics.

    Raises:
        ValueError: If texts is None.

    Examples:
        >>> texts = ["Hello world", "Test text here", "Short"]
        >>> stats = compute_quality_stats(texts)
        >>> stats.total_samples
        3

        >>> compute_quality_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: texts cannot be None
    """
    if texts is None:
        msg = "texts cannot be None"
        raise ValueError(msg)

    if not texts:
        return QualityStats(
            total_samples=0,
            filtered_count=0,
            duplicate_count=0,
            quality_distribution={},
        )

    effective_config = config or create_quality_filter_config()

    # Calculate quality scores for all texts
    quality_scores: list[dict[str, float]] = []
    for text in texts:
        scores = calculate_text_quality_score(text, metrics=effective_config.metrics)
        quality_scores.append(scores)

    # Categorize quality
    high_quality = 0
    medium_quality = 0
    low_quality = 0
    filtered_count = 0

    for scores in quality_scores:
        # Simple categorization based on length and repetition
        length = scores.get("length", 0)
        repetition = scores.get("repetition", 0)

        if length < 10 or repetition > 0.7:
            low_quality += 1
            filtered_count += 1
        elif length < 50 or repetition > 0.4:
            medium_quality += 1
        else:
            high_quality += 1

    # Detect duplicates
    duplicates = detect_duplicates(list(texts))
    duplicate_count = len(duplicates)

    return QualityStats(
        total_samples=len(texts),
        filtered_count=filtered_count,
        duplicate_count=duplicate_count,
        quality_distribution={
            "high": high_quality,
            "medium": medium_quality,
            "low": low_quality,
        },
    )
