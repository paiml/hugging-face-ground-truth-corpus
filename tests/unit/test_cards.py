"""Tests for model card utilities functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.hub.cards import (
    ModelCard,
    ModelCardMetadata,
    ModelCardSection,
    ValidationResult,
    extract_model_description,
    get_model_card,
    list_model_card_sections,
    parse_model_card,
    validate_model_card,
)


class TestModelCardSection:
    """Tests for ModelCardSection dataclass."""

    def test_creation(self) -> None:
        """Test creating ModelCardSection instance."""
        section = ModelCardSection(
            title="Test Section",
            level=2,
            content="Test content.",
        )
        assert section.title == "Test Section"
        assert section.level == 2
        assert section.content == "Test content."

    def test_frozen(self) -> None:
        """Test that ModelCardSection is immutable."""
        section = ModelCardSection(title="Test", level=1, content="Content")
        with pytest.raises(AttributeError):
            section.title = "New Title"  # type: ignore[misc]


class TestModelCardMetadata:
    """Tests for ModelCardMetadata dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ModelCardMetadata with default values."""
        metadata = ModelCardMetadata()
        assert metadata.license is None
        assert metadata.language == []
        assert metadata.tags == []
        assert metadata.datasets == []
        assert metadata.metrics == []
        assert metadata.library_name is None
        assert metadata.pipeline_tag is None
        assert metadata.base_model is None
        assert metadata.extra == {}

    def test_creation_with_values(self) -> None:
        """Test creating ModelCardMetadata with values."""
        metadata = ModelCardMetadata(
            license="mit",
            language=["en", "de"],
            tags=["text-classification"],
            datasets=["imdb"],
            library_name="transformers",
        )
        assert metadata.license == "mit"
        assert metadata.language == ["en", "de"]
        assert metadata.tags == ["text-classification"]
        assert metadata.datasets == ["imdb"]
        assert metadata.library_name == "transformers"

    def test_frozen(self) -> None:
        """Test that ModelCardMetadata is immutable."""
        metadata = ModelCardMetadata(license="mit")
        with pytest.raises(AttributeError):
            metadata.license = "apache-2.0"  # type: ignore[misc]


class TestModelCard:
    """Tests for ModelCard dataclass."""

    def test_creation(self) -> None:
        """Test creating ModelCard instance."""
        card = ModelCard(
            model_id="test/model",
            content="# Test",
            metadata=ModelCardMetadata(),
            sections=[],
        )
        assert card.model_id == "test/model"
        assert card.content == "# Test"
        assert card.metadata is not None
        assert card.sections == []

    def test_frozen(self) -> None:
        """Test that ModelCard is immutable."""
        card = ModelCard(
            model_id="test/model",
            content="# Test",
            metadata=ModelCardMetadata(),
            sections=[],
        )
        with pytest.raises(AttributeError):
            card.model_id = "other/model"  # type: ignore[misc]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test a valid validation result."""
        result = ValidationResult(
            is_valid=True,
            missing_sections=[],
            missing_metadata=[],
            warnings=[],
        )
        assert result.is_valid is True
        assert result.missing_sections == []

    def test_invalid_result(self) -> None:
        """Test an invalid validation result."""
        result = ValidationResult(
            is_valid=False,
            missing_sections=["model description"],
            missing_metadata=["license"],
            warnings=["Missing recommended section"],
        )
        assert result.is_valid is False
        assert "model description" in result.missing_sections
        assert "license" in result.missing_metadata


class TestParseModelCard:
    """Tests for parse_model_card function."""

    def test_simple_card(self) -> None:
        """Test parsing a simple model card."""
        content = "# Test Model\n\nThis is a test."
        card = parse_model_card(content, "test/model")
        assert card.model_id == "test/model"
        assert len(card.sections) == 1
        assert card.sections[0].title == "Test Model"

    def test_card_with_frontmatter(self) -> None:
        """Test parsing a card with YAML frontmatter."""
        content = """---
license: mit
language:
  - en
tags:
  - text-classification
---
# My Model

Description here."""
        card = parse_model_card(content, "test/model")
        assert card.metadata.license == "mit"
        assert "en" in card.metadata.language
        assert "text-classification" in card.metadata.tags

    def test_card_with_multiple_sections(self) -> None:
        """Test parsing a card with multiple sections."""
        content = """# Model Name

Intro.

## Model Description

A great model.

## Intended Uses

For testing.

### Subsection

More details."""
        card = parse_model_card(content)
        assert len(card.sections) == 4
        assert card.sections[0].title == "Model Name"
        assert card.sections[1].title == "Model Description"
        assert card.sections[2].title == "Intended Uses"
        assert card.sections[3].title == "Subsection"
        assert card.sections[3].level == 3

    def test_empty_content_raises_error(self) -> None:
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_model_card("")

    def test_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only content raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_model_card("   \n\n   ")

    def test_default_model_id(self) -> None:
        """Test that default model_id is 'unknown'."""
        card = parse_model_card("# Test")
        assert card.model_id == "unknown"

    def test_frontmatter_with_inline_list(self) -> None:
        """Test parsing frontmatter with inline list syntax."""
        content = """---
language: [en, de, fr]
---
# Model"""
        card = parse_model_card(content)
        assert card.metadata.language == ["en", "de", "fr"]

    def test_frontmatter_with_single_language(self) -> None:
        """Test parsing frontmatter with single language value."""
        content = """---
language: en
---
# Model"""
        card = parse_model_card(content)
        assert card.metadata.language == ["en"]

    def test_extra_metadata_captured(self) -> None:
        """Test that extra metadata fields are captured."""
        content = """---
license: mit
custom_field: custom_value
---
# Model"""
        card = parse_model_card(content)
        assert card.metadata.extra.get("custom_field") == "custom_value"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_model_id_preserved(self, model_id: str) -> None:
        """Test that model_id is preserved in parsed card."""
        # Skip strings that are only whitespace
        if not model_id.strip():
            return
        card = parse_model_card("# Test", model_id)
        assert card.model_id == model_id


class TestValidateModelCard:
    """Tests for validate_model_card function."""

    def test_valid_card(self) -> None:
        """Test validation of a complete model card."""
        content = """---
license: mit
---
# Model

## Model Description
A model that does things.

## Intended Uses
For testing purposes.

## Limitations
None known."""
        card = parse_model_card(content)
        result = validate_model_card(card)
        assert result.is_valid is True
        assert result.missing_sections == []

    def test_missing_required_sections(self) -> None:
        """Test validation with missing required sections."""
        content = "# Model\n\nJust a title."
        card = parse_model_card(content)
        result = validate_model_card(card)
        assert result.is_valid is False
        assert "model description" in result.missing_sections
        assert "intended uses" in result.missing_sections
        assert "limitations" in result.missing_sections

    def test_missing_license_metadata(self) -> None:
        """Test validation with missing license."""
        content = """# Model
## Model Description
Desc.
## Intended Uses
Use.
## Limitations
Limit."""
        card = parse_model_card(content)
        result = validate_model_card(card)
        assert "license" in result.missing_metadata

    def test_none_card_raises_error(self) -> None:
        """Test that None card raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_model_card(None)  # type: ignore[arg-type]

    def test_warnings_for_recommended_sections(self) -> None:
        """Test that warnings are generated for missing recommended sections."""
        content = """# Model
## Model Description
Desc.
## Intended Uses
Use.
## Limitations
Limit."""
        card = parse_model_card(content)
        result = validate_model_card(card)
        # Should have warnings for missing recommended sections
        assert len(result.warnings) > 0
        warning_text = " ".join(result.warnings).lower()
        assert "recommended" in warning_text


class TestGetModelCard:
    """Tests for get_model_card function."""

    def test_empty_model_id_raises_error(self) -> None:
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_model_card("")

    def test_whitespace_model_id_raises_error(self) -> None:
        """Test that whitespace model_id raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_model_card("   ")

    @patch("huggingface_hub.HfApi")
    def test_successful_fetch(self, mock_api_class: MagicMock) -> None:
        """Test successful model card fetch."""
        # Setup mock
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        mock_model_info = MagicMock()
        mock_model_info.card_data = None
        mock_api.model_info.return_value = mock_model_info

        # Mock the download to raise so we use fallback
        mock_api.hf_hub_download.side_effect = Exception("Not found")

        card = get_model_card("test/model")
        assert card.model_id == "test/model"
        mock_api.model_info.assert_called_once_with("test/model")

    @patch("huggingface_hub.HfApi")
    def test_fetch_failure_raises_runtime_error(
        self, mock_api_class: MagicMock
    ) -> None:
        """Test that fetch failure raises RuntimeError."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.model_info.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            get_model_card("test/model")

    @patch("huggingface_hub.HfApi")
    def test_fetch_with_card_data(self, mock_api_class: MagicMock) -> None:
        """Test fetch when model has card_data."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        # Create mock card_data
        mock_card_data = MagicMock()
        mock_card_data.license = "apache-2.0"
        mock_card_data.language = ["en", "de"]
        mock_card_data.tags = ["text-generation", "llm"]

        mock_model_info = MagicMock()
        mock_model_info.card_data = mock_card_data
        mock_api.model_info.return_value = mock_model_info

        # Mock download to raise so we use generated content
        mock_api.hf_hub_download.side_effect = Exception("Not found")

        card = get_model_card("test/model")
        assert card.model_id == "test/model"
        # The frontmatter should have been generated from card_data
        assert "apache-2.0" in card.content or card.metadata.license == "apache-2.0"

    @patch("huggingface_hub.HfApi")
    def test_fetch_with_single_language(self, mock_api_class: MagicMock) -> None:
        """Test fetch when card_data has single language string."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        mock_card_data = MagicMock()
        mock_card_data.license = "mit"
        mock_card_data.language = "en"  # Single string, not list
        mock_card_data.tags = []

        mock_model_info = MagicMock()
        mock_model_info.card_data = mock_card_data
        mock_api.model_info.return_value = mock_model_info

        mock_api.hf_hub_download.side_effect = Exception("Not found")

        card = get_model_card("test/model")
        assert card.model_id == "test/model"


class TestExtractModelDescription:
    """Tests for extract_model_description function."""

    def test_description_section(self) -> None:
        """Test extraction from Model Description section."""
        content = """# Model
## Model Description
This is the model description."""
        card = parse_model_card(content)
        desc = extract_model_description(card)
        assert desc == "This is the model description."

    def test_alternative_description_section(self) -> None:
        """Test extraction from Description section (without 'Model')."""
        content = """# Model
## Description
Alternative description."""
        card = parse_model_card(content)
        desc = extract_model_description(card)
        assert desc == "Alternative description."

    def test_fallback_to_first_section(self) -> None:
        """Test fallback to first section content."""
        content = """# Model Name
This is the intro paragraph.

More content here."""
        card = parse_model_card(content)
        desc = extract_model_description(card)
        assert desc == "This is the intro paragraph."

    def test_no_content_returns_none(self) -> None:
        """Test that empty content returns None."""
        content = "# Model"
        card = parse_model_card(content)
        desc = extract_model_description(card)
        assert desc is None

    def test_none_card_raises_error(self) -> None:
        """Test that None card raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            extract_model_description(None)  # type: ignore[arg-type]


class TestListModelCardSections:
    """Tests for list_model_card_sections function."""

    def test_multiple_sections(self) -> None:
        """Test listing multiple sections."""
        content = """# Title
## Section A
Content A.
## Section B
Content B.
### Subsection
More content."""
        card = parse_model_card(content)
        sections = list_model_card_sections(card)
        assert "Title" in sections
        assert "Section A" in sections
        assert "Section B" in sections
        assert "Subsection" in sections
        assert len(sections) == 4

    def test_empty_sections(self) -> None:
        """Test listing when no sections (edge case)."""
        # Card with no headings - unusual but possible
        card = ModelCard(
            model_id="test",
            content="No headings here.",
            metadata=ModelCardMetadata(),
            sections=[],
        )
        sections = list_model_card_sections(card)
        assert sections == []

    def test_none_card_raises_error(self) -> None:
        """Test that None card raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            list_model_card_sections(None)  # type: ignore[arg-type]
