r"""Model card utilities for HuggingFace Hub.

This module provides functions for parsing, validating, and working with
model cards from the HuggingFace Hub.

Examples:
    >>> from hf_gtc.hub.cards import ModelCard, parse_model_card
    >>> card = parse_model_card("# Model Card\n\nThis is a test model.")
    >>> card.content is not None
    True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hf_gtc._validation import validate_not_none

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class ModelCardSection:
    """A section within a model card.

    Attributes:
        title: Section title (e.g., "Model Description").
        level: Heading level (1-6).
        content: Section content text.

    Examples:
        >>> section = ModelCardSection(
        ...     title="Model Description",
        ...     level=2,
        ...     content="This model does X.",
        ... )
        >>> section.title
        'Model Description'
        >>> section.level
        2
    """

    title: str
    level: int
    content: str


@dataclass(frozen=True, slots=True)
class ModelCardMetadata:
    """Metadata extracted from model card YAML frontmatter.

    Attributes:
        license: Model license (e.g., "mit", "apache-2.0").
        language: Model language(s).
        tags: List of tags.
        datasets: List of training datasets.
        metrics: List of evaluation metrics.
        library_name: Library name (e.g., "transformers").
        pipeline_tag: Pipeline task tag.
        base_model: Base model identifier if fine-tuned.
        extra: Additional metadata fields.

    Examples:
        >>> metadata = ModelCardMetadata(
        ...     license="mit",
        ...     language=["en"],
        ...     tags=["text-classification"],
        ... )
        >>> metadata.license
        'mit'
        >>> "en" in metadata.language
        True
    """

    license: str | None = None
    language: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    library_name: str | None = None
    pipeline_tag: str | None = None
    base_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelCard:
    r"""A parsed model card from HuggingFace Hub.

    Attributes:
        model_id: The model identifier.
        content: Raw markdown content.
        metadata: Parsed YAML frontmatter metadata.
        sections: Parsed sections from the markdown.

    Examples:
        >>> card = ModelCard(
        ...     model_id="test/model",
        ...     content="# Test Model\n\nDescription here.",
        ...     metadata=ModelCardMetadata(),
        ...     sections=[],
        ... )
        >>> card.model_id
        'test/model'
    """

    model_id: str
    content: str
    metadata: ModelCardMetadata
    sections: list[ModelCardSection]


# Required sections for a complete model card
REQUIRED_SECTIONS = frozenset(
    {
        "model description",
        "intended uses",
        "limitations",
    }
)

# Recommended sections
RECOMMENDED_SECTIONS = frozenset(
    {
        "training data",
        "evaluation",
        "how to use",
        "bias",
        "environmental impact",
    }
)


def _parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter from markdown content.

    Args:
        content: Raw markdown content.

    Returns:
        Tuple of (metadata dict, remaining content).

    Examples:
        >>> content = '''---
        ... license: mit
        ... tags:
        ...   - text-classification
        ... ---
        ... # Model Card'''
        >>> meta, body = _parse_yaml_frontmatter(content)
        >>> meta.get("license")
        'mit'
        >>> "# Model Card" in body
        True

        >>> meta, body = _parse_yaml_frontmatter("# No frontmatter")
        >>> meta
        {}
        >>> "# No frontmatter" in body
        True
    """
    # Check for YAML frontmatter (starts with ---)
    if not content.startswith("---"):
        return {}, content

    # Find closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if end_match is None:
        return {}, content

    yaml_content = content[3 : end_match.start() + 3]
    remaining = content[end_match.end() + 3 :]

    metadata = _parse_simple_yaml(yaml_content)
    return metadata, remaining


def _parse_yaml_value(value: str) -> str | list[str]:
    """Parse a single YAML scalar or inline list value."""
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [item.strip().strip("'\"") for item in items]
    return value.strip("'\"")


def _parse_simple_yaml(yaml_content: str) -> dict[str, Any]:
    """Parse simple YAML key-value pairs without external dependency.

    Supports scalar values, inline lists, and indented list items.
    """
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for raw_line in yaml_content.split("\n"):
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith("  - ") and current_key is not None:
            if current_list is None:
                current_list = []
                metadata[current_key] = current_list
            current_list.append(line[4:].strip())
            continue

        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            current_list = None
            if value:
                metadata[key] = _parse_yaml_value(value)

    return metadata


def _parse_sections(content: str) -> list[ModelCardSection]:
    """Parse markdown content into sections.

    Args:
        content: Markdown content without frontmatter.

    Returns:
        List of ModelCardSection objects.

    Examples:
        >>> content = '''# Title
        ... Some intro.
        ...
        ... ## Section 1
        ... Content 1.
        ...
        ... ## Section 2
        ... Content 2.'''
        >>> sections = _parse_sections(content)
        >>> len(sections)
        3
        >>> sections[0].title
        'Title'
        >>> sections[1].title
        'Section 1'
    """
    sections: list[ModelCardSection] = []
    current_title: str | None = None
    current_level: int = 0
    current_content: list[str] = []

    for line in content.split("\n"):
        # Check for heading
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_title is not None:
                sections.append(
                    ModelCardSection(
                        title=current_title,
                        level=current_level,
                        content="\n".join(current_content).strip(),
                    )
                )

            current_title = heading_match.group(2).strip()
            current_level = len(heading_match.group(1))
            current_content = []
        else:
            current_content.append(line)

    # Save final section
    if current_title is not None:
        sections.append(
            ModelCardSection(
                title=current_title,
                level=current_level,
                content="\n".join(current_content).strip(),
            )
        )

    return sections


def parse_model_card(
    content: str,
    model_id: str = "unknown",
) -> ModelCard:
    """Parse a model card from markdown content.

    Args:
        content: Raw markdown content of the model card.
        model_id: Model identifier. Defaults to "unknown".

    Returns:
        Parsed ModelCard object.

    Raises:
        ValueError: If content is empty.

    Examples:
        >>> content = '''---
        ... license: mit
        ... ---
        ... # My Model
        ... A great model.'''
        >>> card = parse_model_card(content, "test/model")
        >>> card.model_id
        'test/model'
        >>> card.metadata.license
        'mit'

        >>> parse_model_card("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: content cannot be empty
    """
    if not content or not content.strip():
        msg = "content cannot be empty"
        raise ValueError(msg)

    # Parse frontmatter
    yaml_data, body = _parse_yaml_frontmatter(content)

    # Create metadata
    metadata = ModelCardMetadata(
        license=yaml_data.get("license"),
        language=yaml_data.get("language", [])
        if isinstance(yaml_data.get("language"), list)
        else [yaml_data["language"]]
        if yaml_data.get("language")
        else [],
        tags=yaml_data.get("tags", []),
        datasets=yaml_data.get("datasets", []),
        metrics=yaml_data.get("metrics", []),
        library_name=yaml_data.get("library_name"),
        pipeline_tag=yaml_data.get("pipeline_tag"),
        base_model=yaml_data.get("base_model"),
        extra={
            k: v
            for k, v in yaml_data.items()
            if k
            not in {
                "license",
                "language",
                "tags",
                "datasets",
                "metrics",
                "library_name",
                "pipeline_tag",
                "base_model",
            }
        },
    )

    # Parse sections
    sections = _parse_sections(body)

    return ModelCard(
        model_id=model_id,
        content=content,
        metadata=metadata,
        sections=sections,
    )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of model card validation.

    Attributes:
        is_valid: Whether the model card passes all required checks.
        missing_sections: List of missing required sections.
        missing_metadata: List of missing recommended metadata fields.
        warnings: List of warning messages.

    Examples:
        >>> result = ValidationResult(
        ...     is_valid=True,
        ...     missing_sections=[],
        ...     missing_metadata=[],
        ...     warnings=[],
        ... )
        >>> result.is_valid
        True
    """

    is_valid: bool
    missing_sections: list[str]
    missing_metadata: list[str]
    warnings: list[str]


def validate_model_card(card: ModelCard) -> ValidationResult:
    """Validate a model card against best practices.

    Checks for required sections and recommended metadata fields.

    Args:
        card: ModelCard to validate.

    Returns:
        ValidationResult with validation details.

    Raises:
        ValueError: If card is None.

    Examples:
        >>> card = parse_model_card('''---
        ... license: mit
        ... ---
        ... # Model
        ... ## Model Description
        ... A model.
        ... ## Intended Uses
        ... For testing.
        ... ## Limitations
        ... None.''')
        >>> result = validate_model_card(card)
        >>> result.is_valid
        True

        >>> validate_model_card(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: card cannot be None
    """
    validate_not_none(card, "card")

    missing_sections: list[str] = []
    missing_metadata: list[str] = []
    warnings: list[str] = []

    # Check required sections
    section_titles = {s.title.lower() for s in card.sections}
    for required in REQUIRED_SECTIONS:
        if required not in section_titles:
            missing_sections.append(required)

    # Check recommended metadata
    if card.metadata.license is None:
        missing_metadata.append("license")

    if not card.metadata.language:
        missing_metadata.append("language")

    # Add warnings for recommended sections
    for recommended in RECOMMENDED_SECTIONS:
        if recommended not in section_titles:
            warnings.append(f"Recommended section missing: {recommended}")

    is_valid = len(missing_sections) == 0

    return ValidationResult(
        is_valid=is_valid,
        missing_sections=missing_sections,
        missing_metadata=missing_metadata,
        warnings=warnings,
    )


def _build_frontmatter(card_data: Any) -> str:
    """Build YAML frontmatter string from HuggingFace card data."""
    if card_data is None:
        return ""
    parts = ["---"]
    if card_data.license:
        parts.append(f"license: {card_data.license}")
    if card_data.language:
        _append_language_field(parts, card_data.language)
    if card_data.tags:
        parts.append("tags:")
        parts.extend(f"  - {tag}" for tag in card_data.tags)
    parts.append("---\n")
    return "\n".join(parts)


def _append_language_field(parts: list[str], language: Any) -> None:
    """Append language field to frontmatter parts list."""
    if isinstance(language, list):
        parts.append("language:")
        parts.extend(f"  - {lang}" for lang in language)
    else:
        parts.append(f"language: {language}")


def _fetch_readme_content(api: Any, model_id: str, frontmatter: str) -> str:
    """Fetch README content from the Hub, falling back to frontmatter."""
    try:
        readme = api.hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            repo_type="model",
        )
        with open(readme, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return frontmatter if frontmatter else f"# {model_id}\n\nNo README."


def get_model_card(model_id: str) -> ModelCard:
    """Fetch and parse a model card from HuggingFace Hub.

    Args:
        model_id: The model identifier (e.g., "bert-base-uncased").

    Returns:
        Parsed ModelCard object.

    Raises:
        ValueError: If model_id is empty.
        RuntimeError: If the model card cannot be fetched.

    Examples:
        >>> get_model_card("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id cannot be empty
    """
    if not model_id or not model_id.strip():
        msg = "model_id cannot be empty"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()

    try:
        model_info = api.model_info(model_id)
        frontmatter = _build_frontmatter(model_info.card_data)
        content = _fetch_readme_content(api, model_id, frontmatter)
        return parse_model_card(content, model_id)
    except Exception as e:
        msg = f"Failed to fetch model card for {model_id}: {e}"
        raise RuntimeError(msg) from e


def extract_model_description(card: ModelCard) -> str | None:
    """Extract the model description from a model card.

    Looks for a "Model Description" section or falls back to
    the first paragraph of content.

    Args:
        card: ModelCard to extract description from.

    Returns:
        Model description text, or None if not found.

    Raises:
        ValueError: If card is None.

    Examples:
        >>> card = parse_model_card('''# Model
        ... ## Model Description
        ... This is the description.''')
        >>> extract_model_description(card)
        'This is the description.'

        >>> extract_model_description(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: card cannot be None
    """
    validate_not_none(card, "card")

    # Look for Model Description section
    for section in card.sections:
        if section.title.lower() in {"model description", "description"}:
            return section.content if section.content else None

    # Fall back to first section content
    if card.sections and card.sections[0].content:
        # Return first paragraph
        content = card.sections[0].content
        paragraphs = content.split("\n\n")
        return paragraphs[0].strip() if paragraphs else None

    return None


def list_model_card_sections(card: ModelCard) -> list[str]:
    """List all section titles in a model card.

    Args:
        card: ModelCard to list sections from.

    Returns:
        List of section titles.

    Raises:
        ValueError: If card is None.

    Examples:
        >>> card = parse_model_card('''# Title
        ... ## Section A
        ... Content A.
        ... ## Section B
        ... Content B.''')
        >>> sections = list_model_card_sections(card)
        >>> "Section A" in sections
        True
        >>> "Section B" in sections
        True

        >>> list_model_card_sections(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: card cannot be None
    """
    validate_not_none(card, "card")

    return [section.title for section in card.sections]
