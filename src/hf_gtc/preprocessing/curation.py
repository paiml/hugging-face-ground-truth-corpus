"""Dataset curation and cleaning pipelines for ML preprocessing.

This module provides utilities for dataset curation including deduplication,
filtering, cleaning, normalization, validation, and decontamination detection
for high-quality ML training datasets.

Examples:
    >>> from hf_gtc.preprocessing.curation import CurationStep, CleaningMethod
    >>> CurationStep.DEDUP.value
    'dedup'
    >>> CleaningMethod.WHITESPACE.value
    'whitespace'
    >>> from hf_gtc.preprocessing.curation import DecontaminationType
    >>> DecontaminationType.EXACT_MATCH.value
    'exact_match'
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class CurationStep(Enum):
    """Steps in a dataset curation pipeline.

    Attributes:
        DEDUP: Remove duplicate samples from the dataset.
        FILTER: Filter samples based on quality criteria.
        CLEAN: Apply text cleaning transformations.
        NORMALIZE: Normalize text format and encoding.
        VALIDATE: Validate samples against schema or rules.

    Examples:
        >>> CurationStep.DEDUP.value
        'dedup'
        >>> CurationStep.FILTER.value
        'filter'
        >>> CurationStep.CLEAN.value
        'clean'
    """

    DEDUP = "dedup"
    FILTER = "filter"
    CLEAN = "clean"
    NORMALIZE = "normalize"
    VALIDATE = "validate"


VALID_CURATION_STEPS = frozenset(s.value for s in CurationStep)


class CleaningMethod(Enum):
    """Methods for text cleaning operations.

    Attributes:
        WHITESPACE: Normalize whitespace (collapse multiple spaces, trim).
        UNICODE: Normalize unicode characters to NFC form.
        HTML: Remove or decode HTML entities and tags.
        MARKDOWN: Remove markdown formatting.

    Examples:
        >>> CleaningMethod.WHITESPACE.value
        'whitespace'
        >>> CleaningMethod.UNICODE.value
        'unicode'
        >>> CleaningMethod.HTML.value
        'html'
    """

    WHITESPACE = "whitespace"
    UNICODE = "unicode"
    HTML = "html"
    MARKDOWN = "markdown"


VALID_CLEANING_METHODS = frozenset(m.value for m in CleaningMethod)


class DecontaminationType(Enum):
    """Types of test set decontamination detection.

    Attributes:
        EXACT_MATCH: Detect exact string matches between train and test.
        NGRAM_OVERLAP: Detect significant n-gram overlap.
        EMBEDDING_SIMILARITY: Detect semantic similarity via embeddings.

    Examples:
        >>> DecontaminationType.EXACT_MATCH.value
        'exact_match'
        >>> DecontaminationType.NGRAM_OVERLAP.value
        'ngram_overlap'
        >>> DecontaminationType.EMBEDDING_SIMILARITY.value
        'embedding_similarity'
    """

    EXACT_MATCH = "exact_match"
    NGRAM_OVERLAP = "ngram_overlap"
    EMBEDDING_SIMILARITY = "embedding_similarity"


VALID_DECONTAMINATION_TYPES = frozenset(t.value for t in DecontaminationType)


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Configuration for text cleaning operations.

    Attributes:
        methods: Tuple of cleaning methods to apply in order.
        lowercase: Whether to convert text to lowercase.
        strip_accents: Whether to strip accent marks from characters.
        min_length: Minimum text length after cleaning (chars).

    Examples:
        >>> config = CleaningConfig(
        ...     methods=(CleaningMethod.WHITESPACE, CleaningMethod.UNICODE),
        ...     lowercase=True,
        ...     strip_accents=False,
        ...     min_length=10,
        ... )
        >>> config.lowercase
        True
        >>> len(config.methods)
        2
    """

    methods: tuple[CleaningMethod, ...]
    lowercase: bool
    strip_accents: bool
    min_length: int


@dataclass(frozen=True, slots=True)
class DecontaminationConfig:
    """Configuration for test set decontamination detection.

    Attributes:
        decontam_type: Type of decontamination detection to use.
        test_datasets: Tuple of test dataset identifiers to check against.
        threshold: Similarity threshold for contamination detection (0.0-1.0).
        ngram_size: N-gram size for overlap detection.

    Examples:
        >>> config = DecontaminationConfig(
        ...     decontam_type=DecontaminationType.NGRAM_OVERLAP,
        ...     test_datasets=("squad", "trivia_qa"),
        ...     threshold=0.8,
        ...     ngram_size=5,
        ... )
        >>> config.threshold
        0.8
        >>> len(config.test_datasets)
        2
    """

    decontam_type: DecontaminationType
    test_datasets: tuple[str, ...]
    threshold: float
    ngram_size: int


@dataclass(frozen=True, slots=True)
class CurationPipeline:
    """Configuration for a complete dataset curation pipeline.

    Attributes:
        steps: Tuple of curation steps to execute in order.
        cleaning_config: Configuration for cleaning step.
        decontam_config: Configuration for decontamination.
        validate_output: Whether to validate output at each step.

    Examples:
        >>> cleaning = CleaningConfig(
        ...     methods=(CleaningMethod.WHITESPACE,),
        ...     lowercase=False,
        ...     strip_accents=False,
        ...     min_length=5,
        ... )
        >>> pipeline = CurationPipeline(
        ...     steps=(CurationStep.CLEAN, CurationStep.DEDUP),
        ...     cleaning_config=cleaning,
        ...     decontam_config=None,
        ...     validate_output=True,
        ... )
        >>> len(pipeline.steps)
        2
        >>> pipeline.validate_output
        True
    """

    steps: tuple[CurationStep, ...]
    cleaning_config: CleaningConfig | None
    decontam_config: DecontaminationConfig | None
    validate_output: bool


@dataclass(frozen=True, slots=True)
class CurationStats:
    """Statistics from a curation pipeline run.

    Attributes:
        original_size: Original number of samples before curation.
        final_size: Final number of samples after curation.
        removed_duplicates: Number of duplicate samples removed.
        removed_contamination: Number of contaminated samples removed.

    Examples:
        >>> stats = CurationStats(
        ...     original_size=10000,
        ...     final_size=8500,
        ...     removed_duplicates=1000,
        ...     removed_contamination=500,
        ... )
        >>> stats.original_size
        10000
        >>> stats.final_size
        8500
    """

    original_size: int
    final_size: int
    removed_duplicates: int
    removed_contamination: int


def validate_cleaning_config(config: CleaningConfig) -> None:
    """Validate cleaning configuration.

    Args:
        config: CleaningConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If methods is empty.
        ValueError: If min_length is negative.

    Examples:
        >>> config = CleaningConfig(
        ...     methods=(CleaningMethod.WHITESPACE,),
        ...     lowercase=False,
        ...     strip_accents=False,
        ...     min_length=5,
        ... )
        >>> validate_cleaning_config(config)  # No error

        >>> validate_cleaning_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = CleaningConfig(
        ...     methods=(),
        ...     lowercase=False,
        ...     strip_accents=False,
        ...     min_length=5,
        ... )
        >>> validate_cleaning_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: methods cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.methods:
        msg = "methods cannot be empty"
        raise ValueError(msg)

    if config.min_length < 0:
        msg = f"min_length must be non-negative, got {config.min_length}"
        raise ValueError(msg)


def validate_decontamination_config(config: DecontaminationConfig) -> None:
    """Validate decontamination configuration.

    Args:
        config: DecontaminationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If test_datasets is empty.
        ValueError: If threshold is not in [0, 1].
        ValueError: If ngram_size is not positive.

    Examples:
        >>> config = DecontaminationConfig(
        ...     decontam_type=DecontaminationType.EXACT_MATCH,
        ...     test_datasets=("squad",),
        ...     threshold=0.8,
        ...     ngram_size=5,
        ... )
        >>> validate_decontamination_config(config)  # No error

        >>> validate_decontamination_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = DecontaminationConfig(
        ...     decontam_type=DecontaminationType.EXACT_MATCH,
        ...     test_datasets=(),
        ...     threshold=0.8,
        ...     ngram_size=5,
        ... )
        >>> validate_decontamination_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: test_datasets cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.test_datasets:
        msg = "test_datasets cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.threshold <= 1.0:
        msg = f"threshold must be between 0 and 1, got {config.threshold}"
        raise ValueError(msg)

    if config.ngram_size <= 0:
        msg = f"ngram_size must be positive, got {config.ngram_size}"
        raise ValueError(msg)


def validate_curation_pipeline(config: CurationPipeline) -> None:
    """Validate curation pipeline configuration.

    Args:
        config: CurationPipeline to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If steps is empty.
        ValueError: If CLEAN step requires cleaning_config.

    Examples:
        >>> cleaning = CleaningConfig(
        ...     methods=(CleaningMethod.WHITESPACE,),
        ...     lowercase=False,
        ...     strip_accents=False,
        ...     min_length=5,
        ... )
        >>> pipeline = CurationPipeline(
        ...     steps=(CurationStep.CLEAN,),
        ...     cleaning_config=cleaning,
        ...     decontam_config=None,
        ...     validate_output=True,
        ... )
        >>> validate_curation_pipeline(pipeline)  # No error

        >>> validate_curation_pipeline(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = CurationPipeline(
        ...     steps=(),
        ...     cleaning_config=None,
        ...     decontam_config=None,
        ...     validate_output=True,
        ... )
        >>> validate_curation_pipeline(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: steps cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.steps:
        msg = "steps cannot be empty"
        raise ValueError(msg)

    if CurationStep.CLEAN in config.steps:
        if config.cleaning_config is None:
            msg = "cleaning_config required when CLEAN step is included"
            raise ValueError(msg)
        validate_cleaning_config(config.cleaning_config)

    if config.decontam_config is not None:
        validate_decontamination_config(config.decontam_config)


def validate_curation_stats(stats: CurationStats) -> None:
    """Validate curation statistics.

    Args:
        stats: CurationStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If original_size is negative.
        ValueError: If final_size is greater than original_size.

    Examples:
        >>> stats = CurationStats(
        ...     original_size=1000,
        ...     final_size=800,
        ...     removed_duplicates=100,
        ...     removed_contamination=100,
        ... )
        >>> validate_curation_stats(stats)  # No error

        >>> validate_curation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = CurationStats(
        ...     original_size=100,
        ...     final_size=200,
        ...     removed_duplicates=0,
        ...     removed_contamination=0,
        ... )
        >>> validate_curation_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: final_size cannot be greater than original_size
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.original_size < 0:
        msg = f"original_size must be non-negative, got {stats.original_size}"
        raise ValueError(msg)

    if stats.final_size > stats.original_size:
        msg = "final_size cannot be greater than original_size"
        raise ValueError(msg)


def create_cleaning_config(
    methods: tuple[str, ...] | None = None,
    lowercase: bool = False,
    strip_accents: bool = False,
    min_length: int = 0,
) -> CleaningConfig:
    """Create a cleaning configuration.

    Args:
        methods: Cleaning method names. Defaults to ("whitespace", "unicode").
        lowercase: Convert to lowercase. Defaults to False.
        strip_accents: Remove accents. Defaults to False.
        min_length: Minimum text length. Defaults to 0.

    Returns:
        CleaningConfig with the specified settings.

    Raises:
        ValueError: If any method is invalid.
        ValueError: If min_length is negative.

    Examples:
        >>> config = create_cleaning_config(methods=("whitespace",))
        >>> config.methods[0]
        <CleaningMethod.WHITESPACE: 'whitespace'>

        >>> config = create_cleaning_config(lowercase=True)
        >>> config.lowercase
        True

        >>> create_cleaning_config(methods=("invalid",))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if methods is None:
        method_enums = (CleaningMethod.WHITESPACE, CleaningMethod.UNICODE)
    else:
        for method in methods:
            if method not in VALID_CLEANING_METHODS:
                msg = f"method must be one of {VALID_CLEANING_METHODS}, got '{method}'"
                raise ValueError(msg)
        method_enums = tuple(CleaningMethod(m) for m in methods)

    config = CleaningConfig(
        methods=method_enums,
        lowercase=lowercase,
        strip_accents=strip_accents,
        min_length=min_length,
    )
    validate_cleaning_config(config)
    return config


def create_decontamination_config(
    decontam_type: str = "ngram_overlap",
    test_datasets: tuple[str, ...] = ("default",),
    threshold: float = 0.8,
    ngram_size: int = 5,
) -> DecontaminationConfig:
    """Create a decontamination configuration.

    Args:
        decontam_type: Type of decontamination. Defaults to "ngram_overlap".
        test_datasets: Test dataset identifiers. Defaults to ("default",).
        threshold: Similarity threshold. Defaults to 0.8.
        ngram_size: N-gram size. Defaults to 5.

    Returns:
        DecontaminationConfig with the specified settings.

    Raises:
        ValueError: If decontam_type is invalid.
        ValueError: If test_datasets is empty.
        ValueError: If threshold is not in [0, 1].
        ValueError: If ngram_size is not positive.

    Examples:
        >>> config = create_decontamination_config(
        ...     test_datasets=("squad", "nq")
        ... )
        >>> len(config.test_datasets)
        2

        >>> config = create_decontamination_config(threshold=0.9)
        >>> config.threshold
        0.9

        >>> create_decontamination_config(decontam_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: decontam_type must be one of
    """
    if decontam_type not in VALID_DECONTAMINATION_TYPES:
        msg = (
            f"decontam_type must be one of {VALID_DECONTAMINATION_TYPES}, "
            f"got '{decontam_type}'"
        )
        raise ValueError(msg)

    config = DecontaminationConfig(
        decontam_type=DecontaminationType(decontam_type),
        test_datasets=test_datasets,
        threshold=threshold,
        ngram_size=ngram_size,
    )
    validate_decontamination_config(config)
    return config


def create_curation_pipeline(
    steps: tuple[str, ...] | None = None,
    cleaning_config: CleaningConfig | None = None,
    decontam_config: DecontaminationConfig | None = None,
    validate_output: bool = True,
) -> CurationPipeline:
    """Create a curation pipeline configuration.

    Args:
        steps: Curation step names. Defaults to ("dedup", "clean", "filter").
        cleaning_config: Cleaning configuration. Created if CLEAN step included.
        decontam_config: Decontamination configuration.
        validate_output: Whether to validate output. Defaults to True.

    Returns:
        CurationPipeline with the specified settings.

    Raises:
        ValueError: If any step is invalid.
        ValueError: If cleaning_config required but not provided.

    Examples:
        >>> pipeline = create_curation_pipeline(steps=("dedup",))
        >>> pipeline.steps[0]
        <CurationStep.DEDUP: 'dedup'>

        >>> create_curation_pipeline(steps=("invalid",))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: step must be one of
    """
    if steps is None:
        step_enums = (CurationStep.DEDUP, CurationStep.CLEAN, CurationStep.FILTER)
    else:
        for step in steps:
            if step not in VALID_CURATION_STEPS:
                msg = f"step must be one of {VALID_CURATION_STEPS}, got '{step}'"
                raise ValueError(msg)
        step_enums = tuple(CurationStep(s) for s in steps)

    # Auto-create cleaning config if CLEAN step is included
    effective_cleaning = cleaning_config
    if CurationStep.CLEAN in step_enums and effective_cleaning is None:
        effective_cleaning = create_cleaning_config()

    config = CurationPipeline(
        steps=step_enums,
        cleaning_config=effective_cleaning,
        decontam_config=decontam_config,
        validate_output=validate_output,
    )
    validate_curation_pipeline(config)
    return config


def list_curation_steps() -> list[str]:
    """List all available curation steps.

    Returns:
        Sorted list of curation step names.

    Examples:
        >>> steps = list_curation_steps()
        >>> "dedup" in steps
        True
        >>> "clean" in steps
        True
        >>> steps == sorted(steps)
        True
    """
    return sorted(VALID_CURATION_STEPS)


def get_curation_step(name: str) -> CurationStep:
    """Get CurationStep enum from string name.

    Args:
        name: Name of the curation step.

    Returns:
        Corresponding CurationStep enum value.

    Raises:
        ValueError: If name is not a valid curation step.

    Examples:
        >>> get_curation_step("dedup")
        <CurationStep.DEDUP: 'dedup'>

        >>> get_curation_step("clean")
        <CurationStep.CLEAN: 'clean'>

        >>> get_curation_step("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid curation step: invalid
    """
    if name not in VALID_CURATION_STEPS:
        msg = f"invalid curation step: {name}"
        raise ValueError(msg)

    return CurationStep(name)


def list_cleaning_methods() -> list[str]:
    """List all available cleaning methods.

    Returns:
        Sorted list of cleaning method names.

    Examples:
        >>> methods = list_cleaning_methods()
        >>> "whitespace" in methods
        True
        >>> "html" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_CLEANING_METHODS)


def get_cleaning_method(name: str) -> CleaningMethod:
    """Get CleaningMethod enum from string name.

    Args:
        name: Name of the cleaning method.

    Returns:
        Corresponding CleaningMethod enum value.

    Raises:
        ValueError: If name is not a valid cleaning method.

    Examples:
        >>> get_cleaning_method("whitespace")
        <CleaningMethod.WHITESPACE: 'whitespace'>

        >>> get_cleaning_method("html")
        <CleaningMethod.HTML: 'html'>

        >>> get_cleaning_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid cleaning method: invalid
    """
    if name not in VALID_CLEANING_METHODS:
        msg = f"invalid cleaning method: {name}"
        raise ValueError(msg)

    return CleaningMethod(name)


def list_decontamination_types() -> list[str]:
    """List all available decontamination types.

    Returns:
        Sorted list of decontamination type names.

    Examples:
        >>> types = list_decontamination_types()
        >>> "exact_match" in types
        True
        >>> "ngram_overlap" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_DECONTAMINATION_TYPES)


def get_decontamination_type(name: str) -> DecontaminationType:
    """Get DecontaminationType enum from string name.

    Args:
        name: Name of the decontamination type.

    Returns:
        Corresponding DecontaminationType enum value.

    Raises:
        ValueError: If name is not a valid decontamination type.

    Examples:
        >>> get_decontamination_type("exact_match")
        <DecontaminationType.EXACT_MATCH: 'exact_match'>

        >>> get_decontamination_type("ngram_overlap")
        <DecontaminationType.NGRAM_OVERLAP: 'ngram_overlap'>

        >>> get_decontamination_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid decontamination type: invalid
    """
    if name not in VALID_DECONTAMINATION_TYPES:
        msg = f"invalid decontamination type: {name}"
        raise ValueError(msg)

    return DecontaminationType(name)


# Regex patterns for cleaning
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
_MARKDOWN_ITALIC_PATTERN = re.compile(r"\*([^*]+)\*")
_MARKDOWN_CODE_PATTERN = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_HEADER_PATTERN = re.compile(r"^#+\s*", re.MULTILINE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def apply_cleaning(
    text: str,
    config: CleaningConfig | None = None,
) -> str:
    """Apply cleaning transformations to text.

    Args:
        text: Text to clean.
        config: Cleaning configuration. Defaults to whitespace and unicode.

    Returns:
        Cleaned text.

    Raises:
        ValueError: If text is None.

    Examples:
        >>> apply_cleaning("  Hello   World  ")
        'Hello World'

        >>> apply_cleaning("<p>Hello</p>", create_cleaning_config(methods=("html",)))
        'Hello'

        >>> apply_cleaning("")
        ''

        >>> apply_cleaning(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if not text:
        return text

    effective_config = config or create_cleaning_config()
    result = text

    for method in effective_config.methods:
        if method == CleaningMethod.WHITESPACE:
            result = _clean_whitespace(result)
        elif method == CleaningMethod.UNICODE:
            result = _clean_unicode(result)
        elif method == CleaningMethod.HTML:
            result = _clean_html(result)
        elif method == CleaningMethod.MARKDOWN:
            result = _clean_markdown(result)

    if effective_config.lowercase:
        result = result.lower()

    if effective_config.strip_accents:
        result = _strip_accents(result)

    return result


def _clean_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _clean_unicode(text: str) -> str:
    """Normalize unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove HTML tags
    result = _HTML_TAG_PATTERN.sub("", text)
    # Decode HTML entities
    result = html.unescape(result)
    return result


def _clean_markdown(text: str) -> str:
    """Remove markdown formatting."""
    result = text
    # Remove bold
    result = _MARKDOWN_BOLD_PATTERN.sub(r"\1", result)
    # Remove italic
    result = _MARKDOWN_ITALIC_PATTERN.sub(r"\1", result)
    # Remove inline code
    result = _MARKDOWN_CODE_PATTERN.sub(r"\1", result)
    # Remove links, keep text
    result = _MARKDOWN_LINK_PATTERN.sub(r"\1", result)
    # Remove headers
    result = _MARKDOWN_HEADER_PATTERN.sub("", result)
    return result


def _strip_accents(text: str) -> str:
    """Strip accent marks from characters."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def detect_contamination(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    config: DecontaminationConfig | None = None,
) -> list[tuple[int, int, float]]:
    """Detect test set contamination in training data.

    Args:
        train_texts: Sequence of training texts.
        test_texts: Sequence of test texts.
        config: Decontamination configuration.

    Returns:
        List of (train_index, test_index, similarity) tuples for contaminated pairs.

    Raises:
        ValueError: If train_texts is None.
        ValueError: If test_texts is None.

    Examples:
        >>> train = ["The quick brown fox jumps over the lazy dog"]
        >>> test = ["The quick brown fox jumps over the lazy dog"]
        >>> cfg = DecontaminationConfig(
        ...     decontam_type=DecontaminationType.EXACT_MATCH,
        ...     test_datasets=("default",),
        ...     threshold=0.8,
        ...     ngram_size=5,
        ... )
        >>> contaminated = detect_contamination(train, test, cfg)
        >>> len(contaminated) >= 1
        True

        >>> detect_contamination([], [])
        []

        >>> detect_contamination(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: train_texts cannot be None

        >>> detect_contamination([], None)  # doctest: +IGNORE_EXCEPTION_DETAIL
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

    effective_config = config or create_decontamination_config()

    contaminated: list[tuple[int, int, float]] = []

    if effective_config.decontam_type == DecontaminationType.EXACT_MATCH:
        contaminated = _detect_exact_contamination(train_texts, test_texts)
    elif effective_config.decontam_type == DecontaminationType.NGRAM_OVERLAP:
        contaminated = _detect_ngram_contamination(
            train_texts,
            test_texts,
            effective_config.ngram_size,
            effective_config.threshold,
        )
    elif effective_config.decontam_type == DecontaminationType.EMBEDDING_SIMILARITY:
        # Simplified: fall back to ngram overlap for embedding similarity
        contaminated = _detect_ngram_contamination(
            train_texts,
            test_texts,
            effective_config.ngram_size,
            effective_config.threshold,
        )

    return contaminated


def _detect_exact_contamination(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
) -> list[tuple[int, int, float]]:
    """Detect exact string matches between train and test."""
    train_hashes: dict[str, int] = {}
    for i, text in enumerate(train_texts):
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        train_hashes[text_hash] = i

    contaminated: list[tuple[int, int, float]] = []

    for j, test_text in enumerate(test_texts):
        test_hash = hashlib.md5(test_text.encode(), usedforsecurity=False).hexdigest()
        if test_hash in train_hashes:
            contaminated.append((train_hashes[test_hash], j, 1.0))

    return contaminated


def _detect_ngram_contamination(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    ngram_size: int,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Detect n-gram overlap contamination."""
    contaminated: list[tuple[int, int, float]] = []

    # Compute n-grams for all texts
    train_ngrams: list[set[tuple[str, ...]]] = []
    for text in train_texts:
        words = text.lower().split()
        ngrams: set[tuple[str, ...]] = set()
        for i in range(len(words) - ngram_size + 1):
            ngrams.add(tuple(words[i : i + ngram_size]))
        train_ngrams.append(ngrams)

    test_ngrams: list[set[tuple[str, ...]]] = []
    for text in test_texts:
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - ngram_size + 1):
            ngrams.add(tuple(words[i : i + ngram_size]))
        test_ngrams.append(ngrams)

    # Compare all pairs
    for i, train_ng in enumerate(train_ngrams):
        for j, test_ng in enumerate(test_ngrams):
            if not train_ng or not test_ng:
                continue

            intersection = len(train_ng & test_ng)
            union = len(train_ng | test_ng)
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= threshold:
                contaminated.append((i, j, similarity))

    return contaminated


def run_curation_pipeline(
    texts: Sequence[str],
    config: CurationPipeline | None = None,
) -> tuple[list[str], CurationStats]:
    """Run a curation pipeline on a collection of texts.

    Args:
        texts: Sequence of texts to curate.
        config: Pipeline configuration.

    Returns:
        Tuple of (curated_texts, statistics).

    Raises:
        ValueError: If texts is None.

    Examples:
        >>> texts = ["  Hello  ", "  Hello  ", "World"]
        >>> pipeline = create_curation_pipeline(steps=("clean", "dedup"))
        >>> curated, stats = run_curation_pipeline(texts, pipeline)
        >>> len(curated) < len(texts)
        True
        >>> stats.original_size
        3

        >>> run_curation_pipeline(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: texts cannot be None
    """
    if texts is None:
        msg = "texts cannot be None"
        raise ValueError(msg)

    if not texts:
        return [], CurationStats(
            original_size=0,
            final_size=0,
            removed_duplicates=0,
            removed_contamination=0,
        )

    effective_config = config or create_curation_pipeline()
    validate_curation_pipeline(effective_config)

    result = list(texts)
    original_size = len(result)
    removed_duplicates = 0
    removed_contamination = 0

    for step in effective_config.steps:
        if step == CurationStep.CLEAN and effective_config.cleaning_config:
            cleaning_cfg = effective_config.cleaning_config
            result = [apply_cleaning(text, cleaning_cfg) for text in result]
            # Filter by min_length
            min_len = effective_config.cleaning_config.min_length
            result = [text for text in result if len(text) >= min_len]

        elif step == CurationStep.DEDUP:
            before_dedup = len(result)
            result = _deduplicate_texts(result)
            removed_duplicates = before_dedup - len(result)

        elif step == CurationStep.FILTER:
            # Basic quality filter: non-empty text
            result = [text for text in result if text.strip()]

        elif step == CurationStep.NORMALIZE:
            result = [_clean_whitespace(_clean_unicode(text)) for text in result]

        elif step == CurationStep.VALIDATE:
            # Basic validation: non-empty after strip
            result = [text for text in result if text.strip()]

    stats = CurationStats(
        original_size=original_size,
        final_size=len(result),
        removed_duplicates=removed_duplicates,
        removed_contamination=removed_contamination,
    )

    return result, stats


def _deduplicate_texts(texts: list[str]) -> list[str]:
    """Remove duplicate texts using hash-based deduplication."""
    seen: set[str] = set()
    result: list[str] = []

    for text in texts:
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        if text_hash not in seen:
            seen.add(text_hash)
            result.append(text)

    return result


def estimate_curation_time(
    num_samples: int,
    config: CurationPipeline | None = None,
) -> float:
    """Estimate time to run curation pipeline.

    Args:
        num_samples: Number of samples to process.
        config: Pipeline configuration.

    Returns:
        Estimated time in seconds.

    Raises:
        ValueError: If num_samples is negative.

    Examples:
        >>> time = estimate_curation_time(10000)
        >>> time > 0
        True

        >>> estimate_curation_time(0)
        0.0

        >>> estimate_curation_time(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples must be non-negative
    """
    if num_samples < 0:
        msg = f"num_samples must be non-negative, got {num_samples}"
        raise ValueError(msg)

    if num_samples == 0:
        return 0.0

    effective_config = config or create_curation_pipeline()

    # Base time estimates per sample (in microseconds)
    step_times = {
        CurationStep.DEDUP: 10.0,  # Hash comparison
        CurationStep.FILTER: 5.0,  # Simple filtering
        CurationStep.CLEAN: 50.0,  # Text processing
        CurationStep.NORMALIZE: 20.0,  # Unicode normalization
        CurationStep.VALIDATE: 5.0,  # Validation checks
    }

    total_us = 0.0
    for step in effective_config.steps:
        total_us += step_times.get(step, 10.0) * num_samples

    # Convert to seconds
    return total_us / 1_000_000


def format_curation_stats(stats: CurationStats) -> str:
    """Format curation statistics as a human-readable string.

    Args:
        stats: CurationStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = CurationStats(
        ...     original_size=10000,
        ...     final_size=8500,
        ...     removed_duplicates=1000,
        ...     removed_contamination=500,
        ... )
        >>> formatted = format_curation_stats(stats)
        >>> "10,000" in formatted
        True
        >>> "8,500" in formatted
        True

        >>> format_curation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    retention_rate = (
        stats.final_size / stats.original_size * 100 if stats.original_size > 0 else 0.0
    )

    duplicate_rate = (
        stats.removed_duplicates / stats.original_size * 100
        if stats.original_size > 0
        else 0.0
    )

    contamination_rate = (
        stats.removed_contamination / stats.original_size * 100
        if stats.original_size > 0
        else 0.0
    )

    lines = [
        "Curation Statistics",
        "=" * 40,
        f"Original size:         {stats.original_size:,}",
        f"Final size:            {stats.final_size:,}",
        f"Retention rate:        {retention_rate:.1f}%",
        "",
        "Removed samples:",
        f"  Duplicates:          {stats.removed_duplicates:,} ({duplicate_rate:.1f}%)",
        f"  Contamination:       {stats.removed_contamination:,} "
        f"({contamination_rate:.1f}%)",
    ]

    return "\n".join(lines)


def get_recommended_curation_config(
    use_case: str,
) -> CurationPipeline:
    """Get recommended curation configuration for a use case.

    Args:
        use_case: Type of use case (e.g., "training", "benchmark", "production").

    Returns:
        Recommended CurationPipeline for the use case.

    Raises:
        ValueError: If use_case is empty.

    Examples:
        >>> pipeline = get_recommended_curation_config("training")
        >>> CurationStep.DEDUP in pipeline.steps
        True

        >>> pipeline = get_recommended_curation_config("benchmark")
        >>> pipeline.decontam_config is not None
        True

        >>> get_recommended_curation_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case cannot be empty
    """
    if not use_case:
        msg = "use_case cannot be empty"
        raise ValueError(msg)

    use_case_lower = use_case.lower()

    if use_case_lower in ("training", "fine-tuning", "pretraining"):
        # Training needs thorough cleaning and deduplication
        return create_curation_pipeline(
            steps=("clean", "dedup", "filter", "validate"),
            cleaning_config=create_cleaning_config(
                methods=("whitespace", "unicode", "html"),
                lowercase=False,
                strip_accents=False,
                min_length=10,
            ),
            validate_output=True,
        )

    elif use_case_lower in ("benchmark", "evaluation", "test"):
        # Benchmarks need decontamination checking
        return create_curation_pipeline(
            steps=("clean", "dedup", "validate"),
            cleaning_config=create_cleaning_config(
                methods=("whitespace", "unicode"),
                min_length=5,
            ),
            decontam_config=create_decontamination_config(
                decontam_type="ngram_overlap",
                test_datasets=("default",),
                threshold=0.8,
                ngram_size=5,
            ),
            validate_output=True,
        )

    elif use_case_lower in ("production", "deployment", "api"):
        # Production needs strict cleaning
        return create_curation_pipeline(
            steps=("clean", "filter", "normalize", "validate"),
            cleaning_config=create_cleaning_config(
                methods=("whitespace", "unicode", "html", "markdown"),
                lowercase=False,
                strip_accents=False,
                min_length=1,
            ),
            validate_output=True,
        )

    else:
        # Default: basic cleaning and deduplication
        return create_curation_pipeline(
            steps=("clean", "dedup"),
            cleaning_config=create_cleaning_config(),
            validate_output=True,
        )
