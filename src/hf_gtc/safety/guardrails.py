"""Content safety guardrails for LLM applications.

This module provides utilities for implementing content safety guardrails,
including input/output filtering, PII detection, and toxicity classification.

Examples:
    >>> from hf_gtc.safety.guardrails import GuardrailType, ActionType
    >>> GuardrailType.INPUT_FILTER.value
    'input_filter'
    >>> ActionType.BLOCK.value
    'block'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class GuardrailType(Enum):
    """Type of guardrail to apply.

    Attributes:
        INPUT_FILTER: Filter applied to input content before processing.
        OUTPUT_FILTER: Filter applied to model output.
        TOPIC_FILTER: Filter for specific topics or subjects.
        PII_FILTER: Filter for personally identifiable information.
        TOXICITY_FILTER: Filter for toxic or harmful content.

    Examples:
        >>> GuardrailType.INPUT_FILTER.value
        'input_filter'
        >>> GuardrailType.PII_FILTER.value
        'pii_filter'
        >>> GuardrailType.TOXICITY_FILTER.value
        'toxicity_filter'
    """

    INPUT_FILTER = "input_filter"
    OUTPUT_FILTER = "output_filter"
    TOPIC_FILTER = "topic_filter"
    PII_FILTER = "pii_filter"
    TOXICITY_FILTER = "toxicity_filter"


class ContentCategory(Enum):
    """Category of content for safety classification.

    Attributes:
        SAFE: Content is safe and appropriate.
        UNSAFE: Content is potentially unsafe (general).
        HATE: Content contains hate speech or discrimination.
        VIOLENCE: Content contains violent or threatening language.
        SEXUAL: Content contains sexual or explicit material.
        SELF_HARM: Content promotes or describes self-harm.
        DANGEROUS: Content promotes dangerous activities.

    Examples:
        >>> ContentCategory.SAFE.value
        'safe'
        >>> ContentCategory.HATE.value
        'hate'
        >>> ContentCategory.VIOLENCE.value
        'violence'
    """

    SAFE = "safe"
    UNSAFE = "unsafe"
    HATE = "hate"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    DANGEROUS = "dangerous"


class ActionType(Enum):
    """Action to take when a guardrail is triggered.

    Attributes:
        ALLOW: Allow the content to pass through.
        BLOCK: Block the content entirely.
        WARN: Allow but include a warning.
        REDACT: Remove or mask the problematic content.
        REPHRASE: Attempt to rephrase the content safely.

    Examples:
        >>> ActionType.ALLOW.value
        'allow'
        >>> ActionType.BLOCK.value
        'block'
        >>> ActionType.REDACT.value
        'redact'
    """

    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    REPHRASE = "rephrase"


VALID_GUARDRAIL_TYPES: frozenset[str] = frozenset(g.value for g in GuardrailType)
VALID_CONTENT_CATEGORIES: frozenset[str] = frozenset(c.value for c in ContentCategory)
VALID_ACTION_TYPES: frozenset[str] = frozenset(a.value for a in ActionType)


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Configuration for a content guardrail.

    Attributes:
        guardrail_type: Type of guardrail to apply.
        action_on_violation: Action to take when guardrail is triggered.
        threshold: Confidence threshold for triggering (0.0 to 1.0).
        enabled: Whether the guardrail is active.

    Examples:
        >>> config = GuardrailConfig(
        ...     guardrail_type=GuardrailType.TOXICITY_FILTER,
        ...     action_on_violation=ActionType.BLOCK,
        ...     threshold=0.8,
        ... )
        >>> config.threshold
        0.8
        >>> config.enabled
        True

        >>> config2 = GuardrailConfig(
        ...     guardrail_type=GuardrailType.PII_FILTER,
        ...     action_on_violation=ActionType.REDACT,
        ...     enabled=False,
        ... )
        >>> config2.enabled
        False
    """

    guardrail_type: GuardrailType
    action_on_violation: ActionType
    threshold: float = 0.5
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ContentPolicyConfig:
    """Configuration for content policy rules.

    Attributes:
        blocked_categories: Set of content categories to block.
        allowed_topics: Set of topics that are explicitly allowed.
        custom_rules: List of custom regex patterns to block.

    Examples:
        >>> cats = frozenset({ContentCategory.HATE, ContentCategory.VIOLENCE})
        >>> policy = ContentPolicyConfig(
        ...     blocked_categories=cats,
        ...     allowed_topics=frozenset({"science", "technology"}),
        ... )
        >>> ContentCategory.HATE in policy.blocked_categories
        True
        >>> "science" in policy.allowed_topics
        True

        >>> policy_empty = ContentPolicyConfig()
        >>> len(policy_empty.blocked_categories)
        0
    """

    blocked_categories: frozenset[ContentCategory] = field(default_factory=frozenset)
    allowed_topics: frozenset[str] = field(default_factory=frozenset)
    custom_rules: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PIIConfig:
    """Configuration for PII detection and handling.

    Attributes:
        detect_email: Whether to detect email addresses.
        detect_phone: Whether to detect phone numbers.
        detect_ssn: Whether to detect social security numbers.
        detect_credit_card: Whether to detect credit card numbers.
        redaction_char: Character to use for redaction.

    Examples:
        >>> config = PIIConfig(detect_email=True, detect_phone=True)
        >>> config.detect_email
        True
        >>> config.redaction_char
        '*'

        >>> config2 = PIIConfig(redaction_char='X')
        >>> config2.redaction_char
        'X'
    """

    detect_email: bool = True
    detect_phone: bool = True
    detect_ssn: bool = True
    detect_credit_card: bool = True
    redaction_char: str = "*"


@dataclass(frozen=True, slots=True)
class ToxicityConfig:
    """Configuration for toxicity detection.

    Attributes:
        model_id: HuggingFace model ID for toxicity classification.
        threshold: Confidence threshold for toxicity detection.
        categories: Tuple of toxicity categories to detect.

    Examples:
        >>> config = ToxicityConfig(
        ...     model_id="unitary/toxic-bert",
        ...     threshold=0.7,
        ... )
        >>> config.threshold
        0.7
        >>> "toxic" in config.categories
        True

        >>> config2 = ToxicityConfig(categories=("toxic", "insult"))
        >>> len(config2.categories)
        2
    """

    model_id: str = "unitary/toxic-bert"
    threshold: float = 0.5
    categories: tuple[str, ...] = field(
        default_factory=lambda: (
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        )
    )


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Result of applying a guardrail check.

    Attributes:
        is_safe: Whether the content passed the safety check.
        violations: List of detected violations.
        action_taken: Action that was taken.
        modified_content: Modified content (if redacted/rephrased), or None.

    Examples:
        >>> result = GuardrailResult(
        ...     is_safe=True,
        ...     violations=[],
        ...     action_taken=ActionType.ALLOW,
        ... )
        >>> result.is_safe
        True
        >>> result.modified_content is None
        True

        >>> result2 = GuardrailResult(
        ...     is_safe=False,
        ...     violations=["PII detected: email"],
        ...     action_taken=ActionType.REDACT,
        ...     modified_content="Contact: ****@****.***",
        ... )
        >>> result2.is_safe
        False
        >>> len(result2.violations)
        1
    """

    is_safe: bool
    violations: list[str] = field(default_factory=list)
    action_taken: ActionType = ActionType.ALLOW
    modified_content: str | None = None


def validate_guardrail_config(config: GuardrailConfig) -> None:
    """Validate a guardrail configuration.

    Args:
        config: GuardrailConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If threshold is not in [0.0, 1.0].

    Examples:
        >>> config = GuardrailConfig(
        ...     guardrail_type=GuardrailType.TOXICITY_FILTER,
        ...     action_on_violation=ActionType.BLOCK,
        ...     threshold=0.8,
        ... )
        >>> validate_guardrail_config(config)  # No error

        >>> validate_guardrail_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = GuardrailConfig(
        ...     guardrail_type=GuardrailType.TOXICITY_FILTER,
        ...     action_on_violation=ActionType.BLOCK,
        ...     threshold=1.5,
        ... )
        >>> validate_guardrail_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0.0 and 1.0
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 0.0 <= config.threshold <= 1.0:
        msg = f"threshold must be between 0.0 and 1.0, got {config.threshold}"
        raise ValueError(msg)


def validate_content_policy_config(config: ContentPolicyConfig) -> None:
    """Validate a content policy configuration.

    Args:
        config: ContentPolicyConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If custom_rules contains invalid regex patterns.

    Examples:
        >>> config = ContentPolicyConfig(
        ...     blocked_categories=frozenset({ContentCategory.HATE}),
        ...     custom_rules=("bad_word",),
        ... )
        >>> validate_content_policy_config(config)  # No error

        >>> validate_content_policy_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = ContentPolicyConfig(custom_rules=("[invalid",))
        >>> validate_content_policy_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid regex pattern in custom_rules...
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    for pattern in config.custom_rules:
        try:
            re.compile(pattern)
        except re.error as e:
            msg = f"invalid regex pattern in custom_rules: {pattern!r} ({e})"
            raise ValueError(msg) from e


def create_guardrail_config(
    guardrail_type: GuardrailType | str,
    action_on_violation: ActionType | str = ActionType.BLOCK,
    threshold: float = 0.5,
    enabled: bool = True,
) -> GuardrailConfig:
    """Create a validated guardrail configuration.

    Args:
        guardrail_type: Type of guardrail or string name.
        action_on_violation: Action to take on violation.
        threshold: Confidence threshold (0.0 to 1.0).
        enabled: Whether the guardrail is active.

    Returns:
        Validated GuardrailConfig instance.

    Raises:
        ValueError: If threshold is not in [0.0, 1.0].
        ValueError: If guardrail_type string is invalid.

    Examples:
        >>> config = create_guardrail_config(
        ...     guardrail_type=GuardrailType.TOXICITY_FILTER,
        ...     action_on_violation=ActionType.BLOCK,
        ...     threshold=0.8,
        ... )
        >>> config.guardrail_type
        <GuardrailType.TOXICITY_FILTER: 'toxicity_filter'>

        >>> config2 = create_guardrail_config("pii_filter", "redact")
        >>> config2.guardrail_type
        <GuardrailType.PII_FILTER: 'pii_filter'>
        >>> config2.action_on_violation
        <ActionType.REDACT: 'redact'>

        >>> create_guardrail_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid guardrail type: invalid
    """
    if isinstance(guardrail_type, str):
        guardrail_type = get_guardrail_type(guardrail_type)

    if isinstance(action_on_violation, str):
        action_on_violation = get_action_type(action_on_violation)

    config = GuardrailConfig(
        guardrail_type=guardrail_type,
        action_on_violation=action_on_violation,
        threshold=threshold,
        enabled=enabled,
    )
    validate_guardrail_config(config)
    return config


def create_content_policy_config(
    blocked_categories: frozenset[ContentCategory] | None = None,
    allowed_topics: frozenset[str] | None = None,
    custom_rules: tuple[str, ...] | list[str] | None = None,
) -> ContentPolicyConfig:
    """Create a validated content policy configuration.

    Args:
        blocked_categories: Set of content categories to block.
        allowed_topics: Set of topics explicitly allowed.
        custom_rules: List of custom regex patterns to block.

    Returns:
        Validated ContentPolicyConfig instance.

    Raises:
        ValueError: If custom_rules contains invalid regex.

    Examples:
        >>> config = create_content_policy_config(
        ...     blocked_categories=frozenset({ContentCategory.HATE}),
        ...     allowed_topics=frozenset({"science"}),
        ... )
        >>> ContentCategory.HATE in config.blocked_categories
        True

        >>> config2 = create_content_policy_config()
        >>> len(config2.blocked_categories)
        0

        >>> create_content_policy_config(custom_rules=["[invalid"])
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid regex pattern in custom_rules...
    """
    if blocked_categories is None:
        blocked_categories = frozenset()
    if allowed_topics is None:
        allowed_topics = frozenset()
    if custom_rules is None:
        custom_rules = ()
    elif isinstance(custom_rules, list):
        custom_rules = tuple(custom_rules)

    config = ContentPolicyConfig(
        blocked_categories=blocked_categories,
        allowed_topics=allowed_topics,
        custom_rules=custom_rules,
    )
    validate_content_policy_config(config)
    return config


def create_pii_config(
    detect_email: bool = True,
    detect_phone: bool = True,
    detect_ssn: bool = True,
    detect_credit_card: bool = True,
    redaction_char: str = "*",
) -> PIIConfig:
    """Create a PII detection configuration.

    Args:
        detect_email: Whether to detect email addresses.
        detect_phone: Whether to detect phone numbers.
        detect_ssn: Whether to detect social security numbers.
        detect_credit_card: Whether to detect credit card numbers.
        redaction_char: Character to use for redaction.

    Returns:
        PIIConfig instance.

    Raises:
        ValueError: If redaction_char is empty or more than one character.

    Examples:
        >>> config = create_pii_config(detect_email=True, detect_phone=False)
        >>> config.detect_email
        True
        >>> config.detect_phone
        False

        >>> config2 = create_pii_config(redaction_char='X')
        >>> config2.redaction_char
        'X'

        >>> create_pii_config(redaction_char='')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: redaction_char must be a single character
    """
    if len(redaction_char) != 1:
        msg = f"redaction_char must be a single character, got {redaction_char!r}"
        raise ValueError(msg)

    return PIIConfig(
        detect_email=detect_email,
        detect_phone=detect_phone,
        detect_ssn=detect_ssn,
        detect_credit_card=detect_credit_card,
        redaction_char=redaction_char,
    )


def create_toxicity_config(
    model_id: str = "unitary/toxic-bert",
    threshold: float = 0.5,
    categories: tuple[str, ...] | list[str] | None = None,
) -> ToxicityConfig:
    """Create a toxicity detection configuration.

    Args:
        model_id: HuggingFace model ID for toxicity classification.
        threshold: Confidence threshold for detection.
        categories: Tuple of toxicity categories to detect.

    Returns:
        ToxicityConfig instance.

    Raises:
        ValueError: If model_id is empty.
        ValueError: If threshold is not in [0.0, 1.0].
        ValueError: If categories is empty.

    Examples:
        >>> config = create_toxicity_config(threshold=0.7)
        >>> config.threshold
        0.7

        >>> config2 = create_toxicity_config(categories=["toxic", "insult"])
        >>> len(config2.categories)
        2

        >>> create_toxicity_config(model_id="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id cannot be empty

        >>> create_toxicity_config(threshold=1.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: threshold must be between 0.0 and 1.0
    """
    if not model_id:
        msg = "model_id cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be between 0.0 and 1.0, got {threshold}"
        raise ValueError(msg)

    if categories is None:
        categories = (
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        )
    elif isinstance(categories, list):
        categories = tuple(categories)

    if len(categories) == 0:
        msg = "categories cannot be empty"
        raise ValueError(msg)

    return ToxicityConfig(
        model_id=model_id,
        threshold=threshold,
        categories=categories,
    )


def list_guardrail_types() -> list[str]:
    """List all available guardrail types.

    Returns:
        Sorted list of guardrail type names.

    Examples:
        >>> types = list_guardrail_types()
        >>> "input_filter" in types
        True
        >>> "pii_filter" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_GUARDRAIL_TYPES)


def list_content_categories() -> list[str]:
    """List all available content categories.

    Returns:
        Sorted list of content category names.

    Examples:
        >>> categories = list_content_categories()
        >>> "safe" in categories
        True
        >>> "hate" in categories
        True
        >>> categories == sorted(categories)
        True
    """
    return sorted(VALID_CONTENT_CATEGORIES)


def list_action_types() -> list[str]:
    """List all available action types.

    Returns:
        Sorted list of action type names.

    Examples:
        >>> actions = list_action_types()
        >>> "allow" in actions
        True
        >>> "block" in actions
        True
        >>> actions == sorted(actions)
        True
    """
    return sorted(VALID_ACTION_TYPES)


def get_guardrail_type(name: str) -> GuardrailType:
    """Get guardrail type enum from string name.

    Args:
        name: Guardrail type name.

    Returns:
        GuardrailType enum value.

    Raises:
        ValueError: If name is not a valid guardrail type.

    Examples:
        >>> get_guardrail_type("input_filter")
        <GuardrailType.INPUT_FILTER: 'input_filter'>
        >>> get_guardrail_type("pii_filter")
        <GuardrailType.PII_FILTER: 'pii_filter'>

        >>> get_guardrail_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid guardrail type: invalid
    """
    for guardrail_type in GuardrailType:
        if guardrail_type.value == name:
            return guardrail_type
    msg = f"invalid guardrail type: {name}"
    raise ValueError(msg)


def get_content_category(name: str) -> ContentCategory:
    """Get content category enum from string name.

    Args:
        name: Content category name.

    Returns:
        ContentCategory enum value.

    Raises:
        ValueError: If name is not a valid content category.

    Examples:
        >>> get_content_category("safe")
        <ContentCategory.SAFE: 'safe'>
        >>> get_content_category("hate")
        <ContentCategory.HATE: 'hate'>

        >>> get_content_category("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid content category: invalid
    """
    for category in ContentCategory:
        if category.value == name:
            return category
    msg = f"invalid content category: {name}"
    raise ValueError(msg)


def get_action_type(name: str) -> ActionType:
    """Get action type enum from string name.

    Args:
        name: Action type name.

    Returns:
        ActionType enum value.

    Raises:
        ValueError: If name is not a valid action type.

    Examples:
        >>> get_action_type("allow")
        <ActionType.ALLOW: 'allow'>
        >>> get_action_type("block")
        <ActionType.BLOCK: 'block'>

        >>> get_action_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid action type: invalid
    """
    for action in ActionType:
        if action.value == name:
            return action
    msg = f"invalid action type: {name}"
    raise ValueError(msg)


# PII detection patterns
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
)
_SSN_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b|\b\d{15,16}\b")


def _detect_pii(content: str, config: PIIConfig) -> list[str]:
    """Detect PII in content based on configuration.

    Args:
        content: Text content to scan.
        config: PII configuration.

    Returns:
        List of detected PII types.
    """
    violations = []

    if config.detect_email and _EMAIL_PATTERN.search(content):
        violations.append("PII detected: email")

    if config.detect_phone and _PHONE_PATTERN.search(content):
        violations.append("PII detected: phone")

    if config.detect_ssn and _SSN_PATTERN.search(content):
        violations.append("PII detected: ssn")

    if config.detect_credit_card and _CREDIT_CARD_PATTERN.search(content):
        violations.append("PII detected: credit_card")

    return violations


def _redact_pii(content: str, config: PIIConfig) -> str:
    """Redact PII from content based on configuration.

    Args:
        content: Text content to redact.
        config: PII configuration.

    Returns:
        Content with PII redacted.
    """
    redacted = content
    char = config.redaction_char

    if config.detect_email:
        redacted = _EMAIL_PATTERN.sub(lambda m: char * len(m.group()), redacted)

    if config.detect_phone:
        redacted = _PHONE_PATTERN.sub(lambda m: char * len(m.group()), redacted)

    if config.detect_ssn:
        redacted = _SSN_PATTERN.sub(lambda m: char * len(m.group()), redacted)

    if config.detect_credit_card:
        redacted = _CREDIT_CARD_PATTERN.sub(lambda m: char * len(m.group()), redacted)

    return redacted


def check_content_safety(
    content: str,
    guardrail_config: GuardrailConfig | None = None,
    pii_config: PIIConfig | None = None,
    policy_config: ContentPolicyConfig | None = None,
) -> GuardrailResult:
    """Check content for safety violations.

    Applies guardrail checks based on provided configurations. If no
    configurations are provided, returns a safe result by default.

    Args:
        content: Text content to check.
        guardrail_config: General guardrail configuration.
        pii_config: PII detection configuration.
        policy_config: Content policy configuration.

    Returns:
        GuardrailResult indicating safety status and any actions taken.

    Raises:
        ValueError: If content is None.

    Examples:
        >>> result = check_content_safety("Hello, world!")
        >>> result.is_safe
        True
        >>> result.action_taken
        <ActionType.ALLOW: 'allow'>

        >>> pii_cfg = create_pii_config()
        >>> guardrail_cfg = create_guardrail_config(
        ...     guardrail_type=GuardrailType.PII_FILTER,
        ...     action_on_violation=ActionType.REDACT,
        ... )
        >>> result = check_content_safety(
        ...     "Email: test@example.com",
        ...     guardrail_config=guardrail_cfg,
        ...     pii_config=pii_cfg,
        ... )
        >>> result.is_safe
        False
        >>> "email" in result.violations[0]
        True
        >>> "***" in result.modified_content
        True

        >>> check_content_safety(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: content cannot be None

        >>> # Empty content is allowed
        >>> result = check_content_safety("")
        >>> result.is_safe
        True
    """
    if content is None:
        msg = "content cannot be None"
        raise ValueError(msg)

    violations: list[str] = []
    modified_content: str | None = None
    action_taken = ActionType.ALLOW

    # Check if guardrail is enabled
    if guardrail_config is not None and not guardrail_config.enabled:
        return GuardrailResult(
            is_safe=True,
            violations=[],
            action_taken=ActionType.ALLOW,
            modified_content=None,
        )

    # PII detection
    if pii_config is not None:
        pii_violations = _detect_pii(content, pii_config)
        violations.extend(pii_violations)

        if pii_violations and guardrail_config is not None:
            action_taken = guardrail_config.action_on_violation
            if action_taken == ActionType.REDACT:
                modified_content = _redact_pii(content, pii_config)

    # Custom rule checking
    if policy_config is not None:
        for pattern in policy_config.custom_rules:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(f"Custom rule violation: {pattern}")

    # Determine final action if violations found
    if violations:
        if guardrail_config is not None:
            action_taken = guardrail_config.action_on_violation
        else:
            action_taken = ActionType.WARN

    is_safe = len(violations) == 0

    return GuardrailResult(
        is_safe=is_safe,
        violations=violations,
        action_taken=action_taken,
        modified_content=modified_content,
    )


def calculate_safety_score(
    content: str,
    pii_config: PIIConfig | None = None,
    policy_config: ContentPolicyConfig | None = None,
) -> float:
    """Calculate a safety score for content.

    Returns a score between 0.0 (unsafe) and 1.0 (safe). The score is
    computed based on the number and severity of detected violations.

    Args:
        content: Text content to score.
        pii_config: PII detection configuration.
        policy_config: Content policy configuration.

    Returns:
        Safety score between 0.0 and 1.0.

    Raises:
        ValueError: If content is None.

    Examples:
        >>> score = calculate_safety_score("Hello, world!")
        >>> score
        1.0

        >>> pii_cfg = create_pii_config()
        >>> score = calculate_safety_score(
        ...     "Email: test@example.com", pii_config=pii_cfg
        ... )
        >>> 0.0 <= score < 1.0
        True

        >>> score = calculate_safety_score(
        ...     "SSN: 123-45-6789, Phone: 555-123-4567",
        ...     pii_config=pii_cfg,
        ... )
        >>> score < 0.8
        True

        >>> calculate_safety_score(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: content cannot be None

        >>> # Empty content is safe
        >>> calculate_safety_score("")
        1.0
    """
    if content is None:
        msg = "content cannot be None"
        raise ValueError(msg)

    if not content:
        return 1.0

    violation_count = 0

    # Count PII violations
    if pii_config is not None:
        pii_violations = _detect_pii(content, pii_config)
        violation_count += len(pii_violations)

    # Count policy violations
    if policy_config is not None:
        for pattern in policy_config.custom_rules:
            matches = re.findall(pattern, content, re.IGNORECASE)
            violation_count += len(matches)

    # Calculate score: each violation reduces score by 0.2, minimum 0.0
    score = max(0.0, 1.0 - (violation_count * 0.2))
    return score
