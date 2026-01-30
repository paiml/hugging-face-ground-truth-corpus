"""Data filtering utilities for toxicity removal and PII detection.

This module provides utilities for filtering data based on toxicity,
PII detection and redaction, language filtering, and quality-based filtering.

Examples:
    >>> from hf_gtc.preprocessing.filtering import FilterType, PIIType
    >>> FilterType.TOXICITY.value
    'toxicity'
    >>> PIIType.EMAIL.value
    'email'
    >>> from hf_gtc.preprocessing.filtering import ToxicityCategory
    >>> ToxicityCategory.HATE.value
    'hate'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class FilterType(Enum):
    """Types of data filtering operations.

    Attributes:
        TOXICITY: Filter based on toxicity scores.
        PII: Filter based on PII detection.
        LANGUAGE: Filter based on language detection.
        QUALITY: Filter based on quality metrics.
        LENGTH: Filter based on text length.
        DUPLICATE: Filter based on duplicate detection.

    Examples:
        >>> FilterType.TOXICITY.value
        'toxicity'
        >>> FilterType.PII.value
        'pii'
        >>> FilterType.LANGUAGE.value
        'language'
    """

    TOXICITY = "toxicity"
    PII = "pii"
    LANGUAGE = "language"
    QUALITY = "quality"
    LENGTH = "length"
    DUPLICATE = "duplicate"


VALID_FILTER_TYPES = frozenset(t.value for t in FilterType)


class PIIType(Enum):
    """Types of Personally Identifiable Information (PII).

    Attributes:
        EMAIL: Email addresses.
        PHONE: Phone numbers.
        SSN: Social Security Numbers.
        CREDIT_CARD: Credit card numbers.
        IP_ADDRESS: IP addresses.
        NAME: Personal names.

    Examples:
        >>> PIIType.EMAIL.value
        'email'
        >>> PIIType.PHONE.value
        'phone'
        >>> PIIType.SSN.value
        'ssn'
    """

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"


VALID_PII_TYPES = frozenset(t.value for t in PIIType)


class ToxicityCategory(Enum):
    """Categories of toxic content.

    Attributes:
        HATE: Hate speech content.
        HARASSMENT: Harassment content.
        VIOLENCE: Violent content.
        SEXUAL: Sexual content.
        SELF_HARM: Self-harm related content.

    Examples:
        >>> ToxicityCategory.HATE.value
        'hate'
        >>> ToxicityCategory.HARASSMENT.value
        'harassment'
        >>> ToxicityCategory.VIOLENCE.value
        'violence'
    """

    HATE = "hate"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"


VALID_TOXICITY_CATEGORIES = frozenset(c.value for c in ToxicityCategory)


@dataclass(frozen=True, slots=True)
class ToxicityConfig:
    """Configuration for toxicity detection.

    Attributes:
        threshold: Threshold score for toxicity (0.0-1.0).
        categories: Tuple of toxicity categories to detect.
        model_name: Name of the toxicity detection model.

    Examples:
        >>> config = ToxicityConfig(
        ...     threshold=0.5,
        ...     categories=(ToxicityCategory.HATE, ToxicityCategory.HARASSMENT),
        ...     model_name="detoxify",
        ... )
        >>> config.threshold
        0.5
        >>> len(config.categories)
        2
    """

    threshold: float
    categories: tuple[ToxicityCategory, ...]
    model_name: str


@dataclass(frozen=True, slots=True)
class PIIConfig:
    """Configuration for PII detection and redaction.

    Attributes:
        pii_types: Tuple of PII types to detect.
        redaction_char: Character used for redaction.
        detect_only: If True, detect but don't redact.

    Examples:
        >>> config = PIIConfig(
        ...     pii_types=(PIIType.EMAIL, PIIType.PHONE),
        ...     redaction_char="*",
        ...     detect_only=False,
        ... )
        >>> config.redaction_char
        '*'
        >>> len(config.pii_types)
        2
    """

    pii_types: tuple[PIIType, ...]
    redaction_char: str
    detect_only: bool


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    """Configuration for language detection and filtering.

    Attributes:
        allowed_languages: Tuple of allowed language codes (e.g., 'en', 'es').
        confidence_threshold: Minimum confidence for language detection.

    Examples:
        >>> config = LanguageConfig(
        ...     allowed_languages=("en", "es", "fr"),
        ...     confidence_threshold=0.8,
        ... )
        >>> "en" in config.allowed_languages
        True
        >>> config.confidence_threshold
        0.8
    """

    allowed_languages: tuple[str, ...]
    confidence_threshold: float


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Combined configuration for data filtering.

    Attributes:
        filter_type: Type of filter to apply.
        toxicity_config: Configuration for toxicity filtering.
        pii_config: Configuration for PII filtering.
        language_config: Configuration for language filtering.

    Examples:
        >>> tox_config = ToxicityConfig(
        ...     threshold=0.5,
        ...     categories=(ToxicityCategory.HATE,),
        ...     model_name="detoxify",
        ... )
        >>> config = FilterConfig(
        ...     filter_type=FilterType.TOXICITY,
        ...     toxicity_config=tox_config,
        ...     pii_config=None,
        ...     language_config=None,
        ... )
        >>> config.filter_type
        <FilterType.TOXICITY: 'toxicity'>
    """

    filter_type: FilterType
    toxicity_config: ToxicityConfig | None
    pii_config: PIIConfig | None
    language_config: LanguageConfig | None


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Result of a filtering operation on a single text.

    Attributes:
        passed: Whether the text passed the filter.
        failed_reason: Reason for filter failure, if any.
        score: Numeric score from the filter (e.g., toxicity score).
        redacted_text: Text with redactions applied, if applicable.

    Examples:
        >>> result = FilterResult(
        ...     passed=True,
        ...     failed_reason=None,
        ...     score=0.1,
        ...     redacted_text=None,
        ... )
        >>> result.passed
        True
        >>> result.score
        0.1
    """

    passed: bool
    failed_reason: str | None
    score: float | None
    redacted_text: str | None


@dataclass(frozen=True, slots=True)
class FilterStats:
    """Statistics from filtering operations.

    Attributes:
        total_processed: Total number of texts processed.
        passed_count: Number of texts that passed filtering.
        filtered_counts: Mapping of filter type to count filtered.
        pii_detected: Mapping of PII type to detection count.

    Examples:
        >>> stats = FilterStats(
        ...     total_processed=1000,
        ...     passed_count=900,
        ...     filtered_counts={"toxicity": 50, "pii": 50},
        ...     pii_detected={"email": 30, "phone": 20},
        ... )
        >>> stats.total_processed
        1000
        >>> stats.passed_count
        900
    """

    total_processed: int
    passed_count: int
    filtered_counts: dict[str, int]
    pii_detected: dict[str, int]


def validate_toxicity_config(config: ToxicityConfig) -> None:
    """Validate toxicity configuration.

    Args:
        config: ToxicityConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If threshold is not in [0, 1].
        ValueError: If categories is empty.
        ValueError: If model_name is empty.

    Examples:
        >>> config = ToxicityConfig(
        ...     threshold=0.5,
        ...     categories=(ToxicityCategory.HATE,),
        ...     model_name="detoxify",
        ... )
        >>> validate_toxicity_config(config)  # No error

        >>> validate_toxicity_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ToxicityConfig(threshold=1.5, categories=(), model_name="test")
        >>> validate_toxicity_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0 and 1
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 0.0 <= config.threshold <= 1.0:
        msg = f"threshold must be between 0 and 1, got {config.threshold}"
        raise ValueError(msg)

    if not config.categories:
        msg = "categories cannot be empty"
        raise ValueError(msg)

    if not config.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)


def validate_pii_config(config: PIIConfig) -> None:
    """Validate PII configuration.

    Args:
        config: PIIConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If pii_types is empty.
        ValueError: If redaction_char is empty.

    Examples:
        >>> config = PIIConfig(
        ...     pii_types=(PIIType.EMAIL,),
        ...     redaction_char="*",
        ...     detect_only=False,
        ... )
        >>> validate_pii_config(config)  # No error

        >>> validate_pii_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = PIIConfig(pii_types=(), redaction_char="*", detect_only=False)
        >>> validate_pii_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pii_types cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.pii_types:
        msg = "pii_types cannot be empty"
        raise ValueError(msg)

    if not config.redaction_char:
        msg = "redaction_char cannot be empty"
        raise ValueError(msg)


def validate_language_config(config: LanguageConfig) -> None:
    """Validate language configuration.

    Args:
        config: LanguageConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If allowed_languages is empty.
        ValueError: If confidence_threshold is not in [0, 1].

    Examples:
        >>> config = LanguageConfig(
        ...     allowed_languages=("en",),
        ...     confidence_threshold=0.8,
        ... )
        >>> validate_language_config(config)  # No error

        >>> validate_language_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = LanguageConfig(allowed_languages=(), confidence_threshold=0.8)
        >>> validate_language_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: allowed_languages cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.allowed_languages:
        msg = "allowed_languages cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.confidence_threshold <= 1.0:
        msg = (
            f"confidence_threshold must be between 0 and 1, "
            f"got {config.confidence_threshold}"
        )
        raise ValueError(msg)


def validate_filter_config(config: FilterConfig) -> None:
    """Validate filter configuration.

    Args:
        config: FilterConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If filter_type is TOXICITY but toxicity_config is None.
        ValueError: If filter_type is PII but pii_config is None.
        ValueError: If filter_type is LANGUAGE but language_config is None.

    Examples:
        >>> tox_config = ToxicityConfig(
        ...     threshold=0.5,
        ...     categories=(ToxicityCategory.HATE,),
        ...     model_name="detoxify",
        ... )
        >>> config = FilterConfig(
        ...     filter_type=FilterType.TOXICITY,
        ...     toxicity_config=tox_config,
        ...     pii_config=None,
        ...     language_config=None,
        ... )
        >>> validate_filter_config(config)  # No error

        >>> validate_filter_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = FilterConfig(
        ...     filter_type=FilterType.TOXICITY,
        ...     toxicity_config=None,
        ...     pii_config=None,
        ...     language_config=None,
        ... )
        >>> validate_filter_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: toxicity_config required for TOXICITY filter
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.filter_type == FilterType.TOXICITY:
        if config.toxicity_config is None:
            msg = "toxicity_config required for TOXICITY filter"
            raise ValueError(msg)
        validate_toxicity_config(config.toxicity_config)

    elif config.filter_type == FilterType.PII:
        if config.pii_config is None:
            msg = "pii_config required for PII filter"
            raise ValueError(msg)
        validate_pii_config(config.pii_config)

    elif config.filter_type == FilterType.LANGUAGE:
        if config.language_config is None:
            msg = "language_config required for LANGUAGE filter"
            raise ValueError(msg)
        validate_language_config(config.language_config)


def create_toxicity_config(
    threshold: float = 0.5,
    categories: tuple[str, ...] | None = None,
    model_name: str = "detoxify",
) -> ToxicityConfig:
    """Create a toxicity configuration.

    Args:
        threshold: Toxicity threshold. Defaults to 0.5.
        categories: Toxicity categories to detect. Defaults to all categories.
        model_name: Model name for detection. Defaults to "detoxify".

    Returns:
        ToxicityConfig with the specified settings.

    Raises:
        ValueError: If threshold is not in [0, 1].
        ValueError: If any category is invalid.
        ValueError: If model_name is empty.

    Examples:
        >>> config = create_toxicity_config(threshold=0.7)
        >>> config.threshold
        0.7

        >>> config = create_toxicity_config(categories=("hate", "violence"))
        >>> len(config.categories)
        2

        >>> create_toxicity_config(threshold=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0 and 1
    """
    if categories is None:
        category_enums = tuple(ToxicityCategory)
    else:
        for cat in categories:
            if cat not in VALID_TOXICITY_CATEGORIES:
                msg = (
                    f"category must be one of {VALID_TOXICITY_CATEGORIES}, got '{cat}'"
                )
                raise ValueError(msg)
        category_enums = tuple(ToxicityCategory(c) for c in categories)

    config = ToxicityConfig(
        threshold=threshold,
        categories=category_enums,
        model_name=model_name,
    )
    validate_toxicity_config(config)
    return config


def create_pii_config(
    pii_types: tuple[str, ...] | None = None,
    redaction_char: str = "*",
    detect_only: bool = False,
) -> PIIConfig:
    """Create a PII configuration.

    Args:
        pii_types: PII types to detect. Defaults to all types.
        redaction_char: Character for redaction. Defaults to "*".
        detect_only: Whether to only detect without redacting. Defaults to False.

    Returns:
        PIIConfig with the specified settings.

    Raises:
        ValueError: If any pii_type is invalid.
        ValueError: If redaction_char is empty.

    Examples:
        >>> config = create_pii_config(pii_types=("email", "phone"))
        >>> len(config.pii_types)
        2

        >>> config = create_pii_config(redaction_char="X")
        >>> config.redaction_char
        'X'

        >>> create_pii_config(pii_types=("invalid",))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pii_type must be one of
    """
    if pii_types is None:
        pii_type_enums = tuple(PIIType)
    else:
        for pii_type in pii_types:
            if pii_type not in VALID_PII_TYPES:
                msg = f"pii_type must be one of {VALID_PII_TYPES}, got '{pii_type}'"
                raise ValueError(msg)
        pii_type_enums = tuple(PIIType(t) for t in pii_types)

    config = PIIConfig(
        pii_types=pii_type_enums,
        redaction_char=redaction_char,
        detect_only=detect_only,
    )
    validate_pii_config(config)
    return config


def create_language_config(
    allowed_languages: tuple[str, ...] = ("en",),
    confidence_threshold: float = 0.8,
) -> LanguageConfig:
    """Create a language configuration.

    Args:
        allowed_languages: Allowed language codes. Defaults to ("en",).
        confidence_threshold: Minimum confidence. Defaults to 0.8.

    Returns:
        LanguageConfig with the specified settings.

    Raises:
        ValueError: If allowed_languages is empty.
        ValueError: If confidence_threshold is not in [0, 1].

    Examples:
        >>> config = create_language_config(allowed_languages=("en", "es"))
        >>> len(config.allowed_languages)
        2

        >>> create_language_config(allowed_languages=())
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: allowed_languages cannot be empty

        >>> create_language_config(confidence_threshold=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: confidence_threshold must be between 0 and 1
    """
    config = LanguageConfig(
        allowed_languages=allowed_languages,
        confidence_threshold=confidence_threshold,
    )
    validate_language_config(config)
    return config


def create_filter_config(
    filter_type: str = "toxicity",
    toxicity_config: ToxicityConfig | None = None,
    pii_config: PIIConfig | None = None,
    language_config: LanguageConfig | None = None,
) -> FilterConfig:
    """Create a filter configuration.

    Args:
        filter_type: Type of filter. Defaults to "toxicity".
        toxicity_config: Toxicity configuration if filter_type is "toxicity".
        pii_config: PII configuration if filter_type is "pii".
        language_config: Language configuration if filter_type is "language".

    Returns:
        FilterConfig with the specified settings.

    Raises:
        ValueError: If filter_type is invalid.
        ValueError: If required config is missing for filter_type.

    Examples:
        >>> tox_config = create_toxicity_config()
        >>> config = create_filter_config(
        ...     filter_type="toxicity",
        ...     toxicity_config=tox_config,
        ... )
        >>> config.filter_type
        <FilterType.TOXICITY: 'toxicity'>

        >>> create_filter_config(filter_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: filter_type must be one of
    """
    if filter_type not in VALID_FILTER_TYPES:
        msg = f"filter_type must be one of {VALID_FILTER_TYPES}, got '{filter_type}'"
        raise ValueError(msg)

    config = FilterConfig(
        filter_type=FilterType(filter_type),
        toxicity_config=toxicity_config,
        pii_config=pii_config,
        language_config=language_config,
    )
    validate_filter_config(config)
    return config


def list_filter_types() -> list[str]:
    """List all available filter types.

    Returns:
        Sorted list of filter type names.

    Examples:
        >>> types = list_filter_types()
        >>> "toxicity" in types
        True
        >>> "pii" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_FILTER_TYPES)


def get_filter_type(name: str) -> FilterType:
    """Get FilterType enum from string name.

    Args:
        name: Name of the filter type.

    Returns:
        Corresponding FilterType enum value.

    Raises:
        ValueError: If name is not a valid filter type.

    Examples:
        >>> get_filter_type("toxicity")
        <FilterType.TOXICITY: 'toxicity'>

        >>> get_filter_type("pii")
        <FilterType.PII: 'pii'>

        >>> get_filter_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid filter type: invalid
    """
    if name not in VALID_FILTER_TYPES:
        msg = f"invalid filter type: {name}"
        raise ValueError(msg)

    return FilterType(name)


def list_pii_types() -> list[str]:
    """List all available PII types.

    Returns:
        Sorted list of PII type names.

    Examples:
        >>> types = list_pii_types()
        >>> "email" in types
        True
        >>> "phone" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_PII_TYPES)


def get_pii_type(name: str) -> PIIType:
    """Get PIIType enum from string name.

    Args:
        name: Name of the PII type.

    Returns:
        Corresponding PIIType enum value.

    Raises:
        ValueError: If name is not a valid PII type.

    Examples:
        >>> get_pii_type("email")
        <PIIType.EMAIL: 'email'>

        >>> get_pii_type("ssn")
        <PIIType.SSN: 'ssn'>

        >>> get_pii_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid PII type: invalid
    """
    if name not in VALID_PII_TYPES:
        msg = f"invalid PII type: {name}"
        raise ValueError(msg)

    return PIIType(name)


def list_toxicity_categories() -> list[str]:
    """List all available toxicity categories.

    Returns:
        Sorted list of toxicity category names.

    Examples:
        >>> categories = list_toxicity_categories()
        >>> "hate" in categories
        True
        >>> "harassment" in categories
        True
        >>> categories == sorted(categories)
        True
    """
    return sorted(VALID_TOXICITY_CATEGORIES)


def get_toxicity_category(name: str) -> ToxicityCategory:
    """Get ToxicityCategory enum from string name.

    Args:
        name: Name of the toxicity category.

    Returns:
        Corresponding ToxicityCategory enum value.

    Raises:
        ValueError: If name is not a valid toxicity category.

    Examples:
        >>> get_toxicity_category("hate")
        <ToxicityCategory.HATE: 'hate'>

        >>> get_toxicity_category("violence")
        <ToxicityCategory.VIOLENCE: 'violence'>

        >>> get_toxicity_category("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid toxicity category: invalid
    """
    if name not in VALID_TOXICITY_CATEGORIES:
        msg = f"invalid toxicity category: {name}"
        raise ValueError(msg)

    return ToxicityCategory(name)


# PII Detection Patterns
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
)
_SSN_PATTERN = re.compile(r"\b[0-9]{3}[-.\s]?[0-9]{2}[-.\s]?[0-9]{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:[0-9]{4}[-.\s]?){3}[0-9]{4}\b")
_IP_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")


def detect_toxicity(
    text: str,
    config: ToxicityConfig | None = None,
) -> dict[str, float]:
    """Detect toxicity in text.

    Args:
        text: Text to analyze.
        config: Toxicity configuration. Defaults to None (uses defaults).

    Returns:
        Dictionary mapping toxicity categories to scores (0.0-1.0).

    Raises:
        ValueError: If text is None.

    Examples:
        >>> scores = detect_toxicity("Hello, how are you?")
        >>> all(0.0 <= v <= 1.0 for v in scores.values())
        True

        >>> scores = detect_toxicity("")
        >>> all(v == 0.0 for v in scores.values())
        True

        >>> detect_toxicity(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    effective_config = config or create_toxicity_config()

    # Return zero scores for empty text
    if not text.strip():
        return {cat.value: 0.0 for cat in effective_config.categories}

    # Simplified toxicity detection using keyword matching
    # Real implementation would use a toxicity model like detoxify
    scores: dict[str, float] = {}
    text_lower = text.lower()

    for category in effective_config.categories:
        score = _estimate_toxicity_score(text_lower, category)
        scores[category.value] = score

    return scores


def _estimate_toxicity_score(text: str, category: ToxicityCategory) -> float:
    """Estimate toxicity score for a category using keyword matching.

    This is a simplified implementation. Real toxicity detection would
    use a trained classifier like detoxify or Perspective API.
    """
    # Simple keyword-based scoring (placeholder for real model)
    # In production, this would use a proper toxicity classifier
    keywords: dict[ToxicityCategory, list[str]] = {
        ToxicityCategory.HATE: ["hate", "racist", "sexist", "bigot"],
        ToxicityCategory.HARASSMENT: ["stupid", "idiot", "loser", "harass"],
        ToxicityCategory.VIOLENCE: ["kill", "murder", "attack", "hurt", "weapon"],
        ToxicityCategory.SEXUAL: ["explicit", "nude", "sexual"],
        ToxicityCategory.SELF_HARM: ["suicide", "self-harm", "cut myself"],
    }

    category_keywords = keywords.get(category, [])
    if not category_keywords:
        return 0.0

    matches = sum(1 for kw in category_keywords if kw in text)
    # Normalize to 0-1 range
    return min(matches / max(len(category_keywords), 1), 1.0)


def detect_pii(
    text: str,
    config: PIIConfig | None = None,
) -> dict[str, list[str]]:
    """Detect PII in text.

    Args:
        text: Text to analyze.
        config: PII configuration. Defaults to None (uses defaults).

    Returns:
        Dictionary mapping PII types to lists of detected instances.

    Raises:
        ValueError: If text is None.

    Examples:
        >>> result = detect_pii("Contact me at test@example.com")
        >>> "email" in result
        True
        >>> len(result["email"]) > 0
        True

        >>> result = detect_pii("")
        >>> all(len(v) == 0 for v in result.values())
        True

        >>> detect_pii(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    effective_config = config or create_pii_config()

    detected: dict[str, list[str]] = {
        pii.value: [] for pii in effective_config.pii_types
    }

    if not text.strip():
        return detected

    for pii_type in effective_config.pii_types:
        matches = _detect_pii_type(text, pii_type)
        detected[pii_type.value] = matches

    return detected


def _detect_pii_type(text: str, pii_type: PIIType) -> list[str]:
    """Detect specific PII type in text."""
    patterns: dict[PIIType, re.Pattern[str]] = {
        PIIType.EMAIL: _EMAIL_PATTERN,
        PIIType.PHONE: _PHONE_PATTERN,
        PIIType.SSN: _SSN_PATTERN,
        PIIType.CREDIT_CARD: _CREDIT_CARD_PATTERN,
        PIIType.IP_ADDRESS: _IP_PATTERN,
    }

    if pii_type == PIIType.NAME:
        # Name detection is complex - use simple capitalized word detection
        # Real implementation would use NER
        return _detect_names(text)

    pattern = patterns.get(pii_type)
    if pattern is None:
        return []

    return pattern.findall(text)


def _detect_names(text: str) -> list[str]:
    """Detect potential names in text using simple heuristics.

    This is a simplified implementation. Real name detection would use NER.
    """
    # Simple heuristic: capitalized words that aren't at sentence start
    words = text.split()
    names = []

    for i, word in enumerate(words):
        if not word:
            continue
        # Check if word is capitalized and not at sentence start
        clean_word = word.strip(".,!?;:'\"")
        is_capitalized = (
            clean_word and clean_word[0].isupper() and clean_word[1:].islower()
        )
        is_not_sentence_start = i > 0 and not words[i - 1].endswith((".", "!", "?"))
        if is_capitalized and is_not_sentence_start:
            names.append(clean_word)

    return names


def redact_pii(
    text: str,
    config: PIIConfig | None = None,
) -> str:
    """Redact PII from text.

    Args:
        text: Text to redact.
        config: PII configuration. Defaults to None (uses defaults).

    Returns:
        Text with PII redacted.

    Raises:
        ValueError: If text is None.

    Examples:
        >>> redact_pii("Email me at test@example.com")
        'Email me at ****************'

        >>> redact_pii("Call 555-123-4567 today")
        'Call ************ today'

        >>> redact_pii("")
        ''

        >>> redact_pii(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    effective_config = config or create_pii_config()

    if effective_config.detect_only:
        return text

    if not text.strip():
        return text

    result = text
    redact_char = effective_config.redaction_char

    for pii_type in effective_config.pii_types:
        result = _redact_pii_type(result, pii_type, redact_char)

    return result


def _redact_pii_type(text: str, pii_type: PIIType, redact_char: str) -> str:
    """Redact specific PII type from text."""
    patterns: dict[PIIType, re.Pattern[str]] = {
        PIIType.EMAIL: _EMAIL_PATTERN,
        PIIType.PHONE: _PHONE_PATTERN,
        PIIType.SSN: _SSN_PATTERN,
        PIIType.CREDIT_CARD: _CREDIT_CARD_PATTERN,
        PIIType.IP_ADDRESS: _IP_PATTERN,
    }

    if pii_type == PIIType.NAME:
        return _redact_names(text, redact_char)

    pattern = patterns.get(pii_type)
    if pattern is None:
        return text

    def replace_match(match: re.Match[str]) -> str:
        return redact_char * len(match.group(0))

    return pattern.sub(replace_match, text)


def _redact_names(text: str, redact_char: str) -> str:
    """Redact names from text using simple heuristics."""
    names = _detect_names(text)
    result = text
    for name in names:
        result = result.replace(name, redact_char * len(name))
    return result


def detect_language(
    text: str,
    config: LanguageConfig | None = None,
) -> tuple[str, float]:
    """Detect the language of text.

    Args:
        text: Text to analyze.
        config: Language configuration. Defaults to None (uses defaults).

    Returns:
        Tuple of (detected_language_code, confidence_score).

    Raises:
        ValueError: If text is None.

    Examples:
        >>> lang, conf = detect_language("Hello, how are you today?")
        >>> lang
        'en'
        >>> 0.0 <= conf <= 1.0
        True

        >>> detect_language("")
        ('unknown', 0.0)

        >>> detect_language(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if not text.strip():
        return ("unknown", 0.0)

    # Simplified language detection using character analysis
    # Real implementation would use langdetect or fasttext
    ascii_ratio = sum(1 for c in text if c.isascii()) / max(len(text), 1)

    # Simplified: assume high ASCII ratio indicates English
    # This is a placeholder for proper language detection
    if ascii_ratio > 0.9:
        return ("en", ascii_ratio)
    elif ascii_ratio > 0.5:
        return ("en", ascii_ratio * 0.8)
    else:
        return ("unknown", 0.5)


def apply_filters(
    text: str,
    configs: Sequence[FilterConfig],
) -> FilterResult:
    """Apply multiple filters to text.

    Args:
        text: Text to filter.
        configs: Sequence of filter configurations to apply.

    Returns:
        FilterResult with combined filtering outcome.

    Raises:
        ValueError: If text is None.
        ValueError: If configs is None.

    Examples:
        >>> tox_config = create_toxicity_config(threshold=0.5)
        >>> filter_config = create_filter_config(
        ...     filter_type="toxicity",
        ...     toxicity_config=tox_config,
        ... )
        >>> result = apply_filters("Hello world", [filter_config])
        >>> result.passed
        True

        >>> apply_filters(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> apply_filters("test", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: configs cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if configs is None:
        msg = "configs cannot be None"
        raise ValueError(msg)

    if not configs:
        return FilterResult(
            passed=True,
            failed_reason=None,
            score=None,
            redacted_text=None,
        )

    redacted_text = text
    max_score: float | None = None

    for config in configs:
        validate_filter_config(config)

        if config.filter_type == FilterType.TOXICITY:
            scores = detect_toxicity(text, config.toxicity_config)
            max_tox_score = max(scores.values()) if scores else 0.0
            if max_score is None:
                max_score = max_tox_score
            else:
                max_score = max(max_score, max_tox_score)

            tox_conf = config.toxicity_config
            threshold = tox_conf.threshold if tox_conf else 0.5
            if max_tox_score >= threshold:
                reason = f"toxicity score {max_tox_score:.2f} exceeds threshold"
                return FilterResult(
                    passed=False,
                    failed_reason=reason,
                    score=max_tox_score,
                    redacted_text=None,
                )

        elif config.filter_type == FilterType.PII:
            detected = detect_pii(text, config.pii_config)
            has_pii = any(matches for matches in detected.values())

            if has_pii:
                if config.pii_config and not config.pii_config.detect_only:
                    redacted_text = redact_pii(text, config.pii_config)
                return FilterResult(
                    passed=False,
                    failed_reason="PII detected",
                    score=None,
                    redacted_text=redacted_text if redacted_text != text else None,
                )

        elif config.filter_type == FilterType.LANGUAGE:
            lang, conf = detect_language(text, config.language_config)

            if config.language_config:
                if lang not in config.language_config.allowed_languages:
                    return FilterResult(
                        passed=False,
                        failed_reason=f"language '{lang}' not in allowed languages",
                        score=conf,
                        redacted_text=None,
                    )
                if conf < config.language_config.confidence_threshold:
                    return FilterResult(
                        passed=False,
                        failed_reason=f"language confidence {conf:.2f} below threshold",
                        score=conf,
                        redacted_text=None,
                    )

        elif config.filter_type == FilterType.LENGTH:
            # Simple length filter - text too short
            if len(text.strip()) < 10:
                return FilterResult(
                    passed=False,
                    failed_reason="text too short",
                    score=float(len(text)),
                    redacted_text=None,
                )

    return FilterResult(
        passed=True,
        failed_reason=None,
        score=max_score,
        redacted_text=redacted_text if redacted_text != text else None,
    )


def format_filter_stats(stats: FilterStats) -> str:
    """Format filter statistics as a human-readable string.

    Args:
        stats: FilterStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = FilterStats(
        ...     total_processed=1000,
        ...     passed_count=900,
        ...     filtered_counts={"toxicity": 50, "pii": 50},
        ...     pii_detected={"email": 30, "phone": 20},
        ... )
        >>> formatted = format_filter_stats(stats)
        >>> "1,000" in formatted or "1000" in formatted
        True
        >>> "900" in formatted
        True

        >>> format_filter_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    pass_rate = (
        stats.passed_count / stats.total_processed * 100
        if stats.total_processed > 0
        else 0.0
    )

    lines = [
        "Filter Statistics",
        "=" * 40,
        f"Total processed:  {stats.total_processed:,}",
        f"Passed count:     {stats.passed_count:,}",
        f"Pass rate:        {pass_rate:.1f}%",
        "",
        "Filtered by type:",
    ]

    for filter_type, count in sorted(stats.filtered_counts.items()):
        percentage = (
            count / stats.total_processed * 100 if stats.total_processed > 0 else 0.0
        )
        lines.append(f"  {filter_type}: {count:,} ({percentage:.1f}%)")

    if stats.pii_detected:
        lines.append("")
        lines.append("PII detected:")
        for pii_type, count in sorted(stats.pii_detected.items()):
            lines.append(f"  {pii_type}: {count:,}")

    return "\n".join(lines)


def get_recommended_filter_config(
    use_case: str,
) -> list[FilterConfig]:
    """Get recommended filter configurations for a use case.

    Args:
        use_case: Type of use case (e.g., "chat", "training", "production").

    Returns:
        List of recommended FilterConfig instances.

    Raises:
        ValueError: If use_case is empty.

    Examples:
        >>> configs = get_recommended_filter_config("chat")
        >>> len(configs) >= 1
        True
        >>> all(isinstance(c, FilterConfig) for c in configs)
        True

        >>> get_recommended_filter_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case cannot be empty
    """
    if not use_case:
        msg = "use_case cannot be empty"
        raise ValueError(msg)

    use_case_lower = use_case.lower()

    if use_case_lower in ("chat", "chatbot", "conversation"):
        # Chatbots need strong toxicity and PII filtering
        return [
            create_filter_config(
                filter_type="toxicity",
                toxicity_config=create_toxicity_config(
                    threshold=0.3,  # Strict threshold
                    categories=("hate", "harassment", "violence", "self_harm"),
                ),
            ),
            create_filter_config(
                filter_type="pii",
                pii_config=create_pii_config(
                    pii_types=("email", "phone", "ssn", "credit_card"),
                    redaction_char="*",
                    detect_only=False,
                ),
            ),
        ]

    elif use_case_lower in ("training", "fine-tuning", "dataset"):
        # Training data needs broad filtering
        return [
            create_filter_config(
                filter_type="toxicity",
                toxicity_config=create_toxicity_config(
                    threshold=0.5,
                ),
            ),
            create_filter_config(
                filter_type="pii",
                pii_config=create_pii_config(detect_only=False),
            ),
            create_filter_config(
                filter_type="language",
                language_config=create_language_config(
                    allowed_languages=("en",),
                    confidence_threshold=0.8,
                ),
            ),
        ]

    elif use_case_lower in ("production", "api", "enterprise"):
        # Production systems need comprehensive filtering
        return [
            create_filter_config(
                filter_type="toxicity",
                toxicity_config=create_toxicity_config(
                    threshold=0.2,  # Very strict
                ),
            ),
            create_filter_config(
                filter_type="pii",
                pii_config=create_pii_config(detect_only=False),
            ),
        ]

    else:
        # Default: basic toxicity and PII filtering
        return [
            create_filter_config(
                filter_type="toxicity",
                toxicity_config=create_toxicity_config(threshold=0.5),
            ),
            create_filter_config(
                filter_type="pii",
                pii_config=create_pii_config(detect_only=True),
            ),
        ]
