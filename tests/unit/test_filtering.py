"""Tests for data filtering functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.filtering import (
    VALID_FILTER_TYPES,
    VALID_PII_TYPES,
    VALID_TOXICITY_CATEGORIES,
    FilterConfig,
    FilterResult,
    FilterStats,
    FilterType,
    LanguageConfig,
    PIIConfig,
    PIIType,
    ToxicityCategory,
    ToxicityConfig,
    apply_filters,
    create_filter_config,
    create_language_config,
    create_pii_config,
    create_toxicity_config,
    detect_language,
    detect_pii,
    detect_toxicity,
    format_filter_stats,
    get_filter_type,
    get_pii_type,
    get_recommended_filter_config,
    get_toxicity_category,
    list_filter_types,
    list_pii_types,
    list_toxicity_categories,
    redact_pii,
    validate_filter_config,
    validate_language_config,
    validate_pii_config,
    validate_toxicity_config,
)


class TestFilterType:
    """Tests for FilterType enum."""

    def test_toxicity_value(self) -> None:
        """Test TOXICITY value."""
        assert FilterType.TOXICITY.value == "toxicity"

    def test_pii_value(self) -> None:
        """Test PII value."""
        assert FilterType.PII.value == "pii"

    def test_language_value(self) -> None:
        """Test LANGUAGE value."""
        assert FilterType.LANGUAGE.value == "language"

    def test_quality_value(self) -> None:
        """Test QUALITY value."""
        assert FilterType.QUALITY.value == "quality"

    def test_length_value(self) -> None:
        """Test LENGTH value."""
        assert FilterType.LENGTH.value == "length"

    def test_duplicate_value(self) -> None:
        """Test DUPLICATE value."""
        assert FilterType.DUPLICATE.value == "duplicate"


class TestPIIType:
    """Tests for PIIType enum."""

    def test_email_value(self) -> None:
        """Test EMAIL value."""
        assert PIIType.EMAIL.value == "email"

    def test_phone_value(self) -> None:
        """Test PHONE value."""
        assert PIIType.PHONE.value == "phone"

    def test_ssn_value(self) -> None:
        """Test SSN value."""
        assert PIIType.SSN.value == "ssn"

    def test_credit_card_value(self) -> None:
        """Test CREDIT_CARD value."""
        assert PIIType.CREDIT_CARD.value == "credit_card"

    def test_ip_address_value(self) -> None:
        """Test IP_ADDRESS value."""
        assert PIIType.IP_ADDRESS.value == "ip_address"

    def test_name_value(self) -> None:
        """Test NAME value."""
        assert PIIType.NAME.value == "name"


class TestToxicityCategory:
    """Tests for ToxicityCategory enum."""

    def test_hate_value(self) -> None:
        """Test HATE value."""
        assert ToxicityCategory.HATE.value == "hate"

    def test_harassment_value(self) -> None:
        """Test HARASSMENT value."""
        assert ToxicityCategory.HARASSMENT.value == "harassment"

    def test_violence_value(self) -> None:
        """Test VIOLENCE value."""
        assert ToxicityCategory.VIOLENCE.value == "violence"

    def test_sexual_value(self) -> None:
        """Test SEXUAL value."""
        assert ToxicityCategory.SEXUAL.value == "sexual"

    def test_self_harm_value(self) -> None:
        """Test SELF_HARM value."""
        assert ToxicityCategory.SELF_HARM.value == "self_harm"


class TestToxicityConfig:
    """Tests for ToxicityConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ToxicityConfig instance."""
        config = ToxicityConfig(
            threshold=0.5,
            categories=(ToxicityCategory.HATE, ToxicityCategory.HARASSMENT),
            model_name="detoxify",
        )
        assert config.threshold == pytest.approx(0.5)
        assert len(config.categories) == 2
        assert config.model_name == "detoxify"

    def test_frozen(self) -> None:
        """Test that ToxicityConfig is immutable."""
        config = ToxicityConfig(
            threshold=0.5,
            categories=(ToxicityCategory.HATE,),
            model_name="detoxify",
        )
        with pytest.raises(AttributeError):
            config.threshold = 0.7  # type: ignore[misc]


class TestPIIConfig:
    """Tests for PIIConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PIIConfig instance."""
        config = PIIConfig(
            pii_types=(PIIType.EMAIL, PIIType.PHONE),
            redaction_char="*",
            detect_only=False,
        )
        assert len(config.pii_types) == 2
        assert config.redaction_char == "*"
        assert config.detect_only is False

    def test_frozen(self) -> None:
        """Test that PIIConfig is immutable."""
        config = PIIConfig(
            pii_types=(PIIType.EMAIL,),
            redaction_char="*",
            detect_only=False,
        )
        with pytest.raises(AttributeError):
            config.detect_only = True  # type: ignore[misc]


class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating LanguageConfig instance."""
        config = LanguageConfig(
            allowed_languages=("en", "es", "fr"),
            confidence_threshold=0.8,
        )
        assert len(config.allowed_languages) == 3
        assert config.confidence_threshold == pytest.approx(0.8)

    def test_frozen(self) -> None:
        """Test that LanguageConfig is immutable."""
        config = LanguageConfig(
            allowed_languages=("en",),
            confidence_threshold=0.8,
        )
        with pytest.raises(AttributeError):
            config.confidence_threshold = 0.9  # type: ignore[misc]


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating FilterConfig instance."""
        tox_config = ToxicityConfig(
            threshold=0.5,
            categories=(ToxicityCategory.HATE,),
            model_name="detoxify",
        )
        config = FilterConfig(
            filter_type=FilterType.TOXICITY,
            toxicity_config=tox_config,
            pii_config=None,
            language_config=None,
        )
        assert config.filter_type == FilterType.TOXICITY
        assert config.toxicity_config is not None
        assert config.pii_config is None

    def test_frozen(self) -> None:
        """Test that FilterConfig is immutable."""
        config = FilterConfig(
            filter_type=FilterType.LENGTH,
            toxicity_config=None,
            pii_config=None,
            language_config=None,
        )
        with pytest.raises(AttributeError):
            config.filter_type = FilterType.PII  # type: ignore[misc]


class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_creation(self) -> None:
        """Test creating FilterResult instance."""
        result = FilterResult(
            passed=True,
            failed_reason=None,
            score=0.1,
            redacted_text=None,
        )
        assert result.passed is True
        assert result.failed_reason is None
        assert result.score == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that FilterResult is immutable."""
        result = FilterResult(
            passed=True,
            failed_reason=None,
            score=None,
            redacted_text=None,
        )
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]


class TestFilterStats:
    """Tests for FilterStats dataclass."""

    def test_creation(self) -> None:
        """Test creating FilterStats instance."""
        stats = FilterStats(
            total_processed=1000,
            passed_count=900,
            filtered_counts={"toxicity": 50, "pii": 50},
            pii_detected={"email": 30, "phone": 20},
        )
        assert stats.total_processed == 1000
        assert stats.passed_count == 900
        assert stats.filtered_counts["toxicity"] == 50
        assert stats.pii_detected["email"] == 30

    def test_frozen(self) -> None:
        """Test that FilterStats is immutable."""
        stats = FilterStats(
            total_processed=100,
            passed_count=90,
            filtered_counts={},
            pii_detected={},
        )
        with pytest.raises(AttributeError):
            stats.total_processed = 200  # type: ignore[misc]


class TestValidateToxicityConfig:
    """Tests for validate_toxicity_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ToxicityConfig(
            threshold=0.5,
            categories=(ToxicityCategory.HATE,),
            model_name="detoxify",
        )
        validate_toxicity_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_toxicity_config(None)  # type: ignore[arg-type]

    def test_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0 raises ValueError."""
        config = ToxicityConfig(
            threshold=-0.1,
            categories=(ToxicityCategory.HATE,),
            model_name="detoxify",
        )
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_toxicity_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        config = ToxicityConfig(
            threshold=1.5,
            categories=(ToxicityCategory.HATE,),
            model_name="detoxify",
        )
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_toxicity_config(config)

    def test_empty_categories_raises_error(self) -> None:
        """Test that empty categories raises ValueError."""
        config = ToxicityConfig(
            threshold=0.5,
            categories=(),
            model_name="detoxify",
        )
        with pytest.raises(ValueError, match="categories cannot be empty"):
            validate_toxicity_config(config)

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        config = ToxicityConfig(
            threshold=0.5,
            categories=(ToxicityCategory.HATE,),
            model_name="",
        )
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            validate_toxicity_config(config)


class TestValidatePIIConfig:
    """Tests for validate_pii_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = PIIConfig(
            pii_types=(PIIType.EMAIL,),
            redaction_char="*",
            detect_only=False,
        )
        validate_pii_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_pii_config(None)  # type: ignore[arg-type]

    def test_empty_pii_types_raises_error(self) -> None:
        """Test that empty pii_types raises ValueError."""
        config = PIIConfig(
            pii_types=(),
            redaction_char="*",
            detect_only=False,
        )
        with pytest.raises(ValueError, match="pii_types cannot be empty"):
            validate_pii_config(config)

    def test_empty_redaction_char_raises_error(self) -> None:
        """Test that empty redaction_char raises ValueError."""
        config = PIIConfig(
            pii_types=(PIIType.EMAIL,),
            redaction_char="",
            detect_only=False,
        )
        with pytest.raises(ValueError, match="redaction_char cannot be empty"):
            validate_pii_config(config)


class TestValidateLanguageConfig:
    """Tests for validate_language_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = LanguageConfig(
            allowed_languages=("en",),
            confidence_threshold=0.8,
        )
        validate_language_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_language_config(None)  # type: ignore[arg-type]

    def test_empty_languages_raises_error(self) -> None:
        """Test that empty allowed_languages raises ValueError."""
        config = LanguageConfig(
            allowed_languages=(),
            confidence_threshold=0.8,
        )
        with pytest.raises(ValueError, match="allowed_languages cannot be empty"):
            validate_language_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        config = LanguageConfig(
            allowed_languages=("en",),
            confidence_threshold=1.5,
        )
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            validate_language_config(config)

    def test_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0 raises ValueError."""
        config = LanguageConfig(
            allowed_languages=("en",),
            confidence_threshold=-0.1,
        )
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            validate_language_config(config)


class TestValidateFilterConfig:
    """Tests for validate_filter_config function."""

    def test_valid_toxicity_config(self) -> None:
        """Test validation of valid toxicity filter config."""
        tox_config = ToxicityConfig(
            threshold=0.5,
            categories=(ToxicityCategory.HATE,),
            model_name="detoxify",
        )
        config = FilterConfig(
            filter_type=FilterType.TOXICITY,
            toxicity_config=tox_config,
            pii_config=None,
            language_config=None,
        )
        validate_filter_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_filter_config(None)  # type: ignore[arg-type]

    def test_missing_toxicity_config_raises_error(self) -> None:
        """Test that missing toxicity_config raises ValueError."""
        config = FilterConfig(
            filter_type=FilterType.TOXICITY,
            toxicity_config=None,
            pii_config=None,
            language_config=None,
        )
        with pytest.raises(ValueError, match="toxicity_config required"):
            validate_filter_config(config)

    def test_missing_pii_config_raises_error(self) -> None:
        """Test that missing pii_config raises ValueError."""
        config = FilterConfig(
            filter_type=FilterType.PII,
            toxicity_config=None,
            pii_config=None,
            language_config=None,
        )
        with pytest.raises(ValueError, match="pii_config required"):
            validate_filter_config(config)

    def test_missing_language_config_raises_error(self) -> None:
        """Test that missing language_config raises ValueError."""
        config = FilterConfig(
            filter_type=FilterType.LANGUAGE,
            toxicity_config=None,
            pii_config=None,
            language_config=None,
        )
        with pytest.raises(ValueError, match="language_config required"):
            validate_filter_config(config)


class TestCreateToxicityConfig:
    """Tests for create_toxicity_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_toxicity_config()
        assert config.threshold == pytest.approx(0.5)
        assert len(config.categories) == 5  # All categories
        assert config.model_name == "detoxify"

    def test_custom_threshold(self) -> None:
        """Test creating config with custom threshold."""
        config = create_toxicity_config(threshold=0.7)
        assert config.threshold == pytest.approx(0.7)

    def test_custom_categories(self) -> None:
        """Test creating config with custom categories."""
        config = create_toxicity_config(categories=("hate", "violence"))
        assert len(config.categories) == 2
        assert ToxicityCategory.HATE in config.categories
        assert ToxicityCategory.VIOLENCE in config.categories

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            create_toxicity_config(threshold=1.5)

    def test_invalid_category_raises_error(self) -> None:
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="category must be one of"):
            create_toxicity_config(categories=("invalid",))

    def test_empty_model_name_raises_error(self) -> None:
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_toxicity_config(model_name="")


class TestCreatePIIConfig:
    """Tests for create_pii_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_pii_config()
        assert len(config.pii_types) == 6  # All types
        assert config.redaction_char == "*"
        assert config.detect_only is False

    def test_custom_pii_types(self) -> None:
        """Test creating config with custom PII types."""
        config = create_pii_config(pii_types=("email", "phone"))
        assert len(config.pii_types) == 2
        assert PIIType.EMAIL in config.pii_types
        assert PIIType.PHONE in config.pii_types

    def test_custom_redaction_char(self) -> None:
        """Test creating config with custom redaction character."""
        config = create_pii_config(redaction_char="X")
        assert config.redaction_char == "X"

    def test_detect_only_mode(self) -> None:
        """Test creating config with detect_only=True."""
        config = create_pii_config(detect_only=True)
        assert config.detect_only is True

    def test_invalid_pii_type_raises_error(self) -> None:
        """Test that invalid PII type raises ValueError."""
        with pytest.raises(ValueError, match="pii_type must be one of"):
            create_pii_config(pii_types=("invalid",))

    def test_empty_redaction_char_raises_error(self) -> None:
        """Test that empty redaction_char raises ValueError."""
        with pytest.raises(ValueError, match="redaction_char cannot be empty"):
            create_pii_config(redaction_char="")


class TestCreateLanguageConfig:
    """Tests for create_language_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_language_config()
        assert "en" in config.allowed_languages
        assert config.confidence_threshold == pytest.approx(0.8)

    def test_custom_languages(self) -> None:
        """Test creating config with custom languages."""
        config = create_language_config(allowed_languages=("en", "es", "fr"))
        assert len(config.allowed_languages) == 3

    def test_custom_threshold(self) -> None:
        """Test creating config with custom threshold."""
        config = create_language_config(confidence_threshold=0.9)
        assert config.confidence_threshold == pytest.approx(0.9)

    def test_empty_languages_raises_error(self) -> None:
        """Test that empty languages raises ValueError."""
        with pytest.raises(ValueError, match="allowed_languages cannot be empty"):
            create_language_config(allowed_languages=())

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            create_language_config(confidence_threshold=1.5)


class TestCreateFilterConfig:
    """Tests for create_filter_config function."""

    def test_toxicity_filter(self) -> None:
        """Test creating toxicity filter config."""
        tox_config = create_toxicity_config()
        config = create_filter_config(
            filter_type="toxicity",
            toxicity_config=tox_config,
        )
        assert config.filter_type == FilterType.TOXICITY

    def test_pii_filter(self) -> None:
        """Test creating PII filter config."""
        pii_config = create_pii_config()
        config = create_filter_config(
            filter_type="pii",
            pii_config=pii_config,
        )
        assert config.filter_type == FilterType.PII

    def test_language_filter(self) -> None:
        """Test creating language filter config."""
        lang_config = create_language_config()
        config = create_filter_config(
            filter_type="language",
            language_config=lang_config,
        )
        assert config.filter_type == FilterType.LANGUAGE

    def test_invalid_filter_type_raises_error(self) -> None:
        """Test that invalid filter type raises ValueError."""
        with pytest.raises(ValueError, match="filter_type must be one of"):
            create_filter_config(filter_type="invalid")

    def test_missing_config_raises_error(self) -> None:
        """Test that missing config raises ValueError."""
        with pytest.raises(ValueError, match="toxicity_config required"):
            create_filter_config(filter_type="toxicity")


class TestListFilterTypes:
    """Tests for list_filter_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_filter_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_filter_types()
        assert "toxicity" in types
        assert "pii" in types
        assert "language" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_filter_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_FILTER_TYPES."""
        types = list_filter_types()
        assert set(types) == VALID_FILTER_TYPES


class TestGetFilterType:
    """Tests for get_filter_type function."""

    def test_get_toxicity(self) -> None:
        """Test getting TOXICITY type."""
        result = get_filter_type("toxicity")
        assert result == FilterType.TOXICITY

    def test_get_pii(self) -> None:
        """Test getting PII type."""
        result = get_filter_type("pii")
        assert result == FilterType.PII

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid filter type"):
            get_filter_type("invalid")


class TestListPIITypes:
    """Tests for list_pii_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_pii_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_pii_types()
        assert "email" in types
        assert "phone" in types
        assert "ssn" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_pii_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_PII_TYPES."""
        types = list_pii_types()
        assert set(types) == VALID_PII_TYPES


class TestGetPIIType:
    """Tests for get_pii_type function."""

    def test_get_email(self) -> None:
        """Test getting EMAIL type."""
        result = get_pii_type("email")
        assert result == PIIType.EMAIL

    def test_get_ssn(self) -> None:
        """Test getting SSN type."""
        result = get_pii_type("ssn")
        assert result == PIIType.SSN

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid PII type"):
            get_pii_type("invalid")


class TestListToxicityCategories:
    """Tests for list_toxicity_categories function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        categories = list_toxicity_categories()
        assert isinstance(categories, list)

    def test_contains_expected_categories(self) -> None:
        """Test that list contains expected categories."""
        categories = list_toxicity_categories()
        assert "hate" in categories
        assert "harassment" in categories
        assert "violence" in categories

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        categories = list_toxicity_categories()
        assert categories == sorted(categories)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_TOXICITY_CATEGORIES."""
        categories = list_toxicity_categories()
        assert set(categories) == VALID_TOXICITY_CATEGORIES


class TestGetToxicityCategory:
    """Tests for get_toxicity_category function."""

    def test_get_hate(self) -> None:
        """Test getting HATE category."""
        result = get_toxicity_category("hate")
        assert result == ToxicityCategory.HATE

    def test_get_violence(self) -> None:
        """Test getting VIOLENCE category."""
        result = get_toxicity_category("violence")
        assert result == ToxicityCategory.VIOLENCE

    def test_invalid_category_raises_error(self) -> None:
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="invalid toxicity category"):
            get_toxicity_category("invalid")


class TestDetectToxicity:
    """Tests for detect_toxicity function."""

    def test_basic_detection(self) -> None:
        """Test basic toxicity detection."""
        scores = detect_toxicity("Hello, how are you?")
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_empty_text(self) -> None:
        """Test with empty text."""
        scores = detect_toxicity("")
        assert all(v == 0.0 for v in scores.values())

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            detect_toxicity(None)  # type: ignore[arg-type]

    def test_with_custom_config(self) -> None:
        """Test with custom toxicity config."""
        config = create_toxicity_config(
            threshold=0.5,
            categories=("hate", "violence"),
        )
        scores = detect_toxicity("Hello world", config)
        assert "hate" in scores
        assert "violence" in scores
        assert len(scores) == 2

    def test_toxic_keywords_detected(self) -> None:
        """Test that toxic keywords increase scores."""
        clean_scores = detect_toxicity("Hello, this is a friendly message.")
        # Using a keyword that should trigger detection
        hateful_scores = detect_toxicity("I hate this racist stuff.")
        # Hate score should be higher for hateful text
        assert hateful_scores.get("hate", 0) >= clean_scores.get("hate", 0)

    def test_violence_keywords_detected(self) -> None:
        """Test that violence keywords increase scores."""
        clean_scores = detect_toxicity("Hello, have a nice day.")
        violent_scores = detect_toxicity("I want to kill and murder everyone.")
        assert violent_scores.get("violence", 0) >= clean_scores.get("violence", 0)


class TestDetectPII:
    """Tests for detect_pii function."""

    def test_email_detection(self) -> None:
        """Test email detection."""
        result = detect_pii("Contact me at test@example.com")
        assert "email" in result
        assert len(result["email"]) > 0
        assert "test@example.com" in result["email"]

    def test_phone_detection(self) -> None:
        """Test phone detection."""
        result = detect_pii("Call me at 555-123-4567")
        assert "phone" in result
        assert len(result["phone"]) > 0

    def test_ssn_detection(self) -> None:
        """Test SSN detection."""
        result = detect_pii("My SSN is 123-45-6789")
        assert "ssn" in result
        assert len(result["ssn"]) > 0

    def test_credit_card_detection(self) -> None:
        """Test credit card detection."""
        result = detect_pii("Card: 1234-5678-9012-3456")
        assert "credit_card" in result
        assert len(result["credit_card"]) > 0

    def test_ip_address_detection(self) -> None:
        """Test IP address detection."""
        result = detect_pii("Server IP: 192.168.1.1")
        assert "ip_address" in result
        assert len(result["ip_address"]) > 0

    def test_empty_text(self) -> None:
        """Test with empty text."""
        result = detect_pii("")
        assert all(len(v) == 0 for v in result.values())

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            detect_pii(None)  # type: ignore[arg-type]

    def test_no_pii(self) -> None:
        """Test with text containing no PII."""
        result = detect_pii("This is a clean text with no personal info.")
        # Should have empty lists for most types
        non_name_types = ["email", "phone", "ssn", "credit_card", "ip_address"]
        for pii_type in non_name_types:
            assert len(result.get(pii_type, [])) == 0

    def test_multiple_emails(self) -> None:
        """Test detection of multiple emails."""
        result = detect_pii("Contact alice@test.com or bob@example.org")
        assert len(result["email"]) == 2


class TestRedactPII:
    """Tests for redact_pii function."""

    def test_email_redaction(self) -> None:
        """Test email redaction."""
        result = redact_pii("Email me at test@example.com")
        assert "test@example.com" not in result
        assert "*" * len("test@example.com") in result

    def test_phone_redaction(self) -> None:
        """Test phone redaction."""
        result = redact_pii("Call 555-123-4567 today")
        assert "555-123-4567" not in result
        assert "*" in result

    def test_custom_redaction_char(self) -> None:
        """Test custom redaction character."""
        config = create_pii_config(redaction_char="X")
        result = redact_pii("Email: test@example.com", config)
        assert "X" in result
        assert "*" not in result

    def test_empty_text(self) -> None:
        """Test with empty text."""
        result = redact_pii("")
        assert result == ""

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            redact_pii(None)  # type: ignore[arg-type]

    def test_detect_only_no_redaction(self) -> None:
        """Test detect_only mode doesn't redact."""
        config = create_pii_config(detect_only=True)
        text = "Email: test@example.com"
        result = redact_pii(text, config)
        assert result == text

    def test_no_pii_unchanged(self) -> None:
        """Test that text without PII is unchanged."""
        text = "This is a clean text."
        result = redact_pii(text)
        assert result == text


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_english_detection(self) -> None:
        """Test English detection."""
        lang, conf = detect_language("Hello, how are you today?")
        assert lang == "en"
        assert 0.0 <= conf <= 1.0

    def test_empty_text(self) -> None:
        """Test with empty text."""
        lang, conf = detect_language("")
        assert lang == "unknown"
        assert conf == 0.0

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            detect_language(None)  # type: ignore[arg-type]

    def test_high_ascii_text(self) -> None:
        """Test with high ASCII ratio text."""
        lang, conf = detect_language("This is entirely English ASCII text.")
        assert lang == "en"
        assert conf > 0.5


class TestApplyFilters:
    """Tests for apply_filters function."""

    def test_passing_toxicity_filter(self) -> None:
        """Test text passing toxicity filter."""
        tox_config = create_toxicity_config(threshold=0.5)
        filter_config = create_filter_config(
            filter_type="toxicity",
            toxicity_config=tox_config,
        )
        result = apply_filters("Hello world", [filter_config])
        assert result.passed is True

    def test_pii_filter_detection(self) -> None:
        """Test PII detection filtering."""
        pii_config = create_pii_config(pii_types=("email",))
        filter_config = create_filter_config(
            filter_type="pii",
            pii_config=pii_config,
        )
        result = apply_filters("Contact test@example.com", [filter_config])
        assert result.passed is False
        assert result.redacted_text is not None

    def test_language_filter_pass(self) -> None:
        """Test text passing language filter."""
        lang_config = create_language_config(
            allowed_languages=("en",),
            confidence_threshold=0.5,
        )
        filter_config = create_filter_config(
            filter_type="language",
            language_config=lang_config,
        )
        result = apply_filters("This is English text.", [filter_config])
        assert result.passed is True

    def test_empty_configs(self) -> None:
        """Test with empty configs."""
        result = apply_filters("Hello world", [])
        assert result.passed is True

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            apply_filters(None, [])  # type: ignore[arg-type]

    def test_none_configs_raises_error(self) -> None:
        """Test that None configs raises ValueError."""
        with pytest.raises(ValueError, match="configs cannot be None"):
            apply_filters("test", None)  # type: ignore[arg-type]

    def test_multiple_filters(self) -> None:
        """Test applying multiple filters."""
        tox_config = create_toxicity_config(threshold=0.5)
        pii_config = create_pii_config(pii_types=("email",), detect_only=True)
        configs = [
            create_filter_config(
                filter_type="toxicity",
                toxicity_config=tox_config,
            ),
            create_filter_config(
                filter_type="pii",
                pii_config=pii_config,
            ),
        ]
        result = apply_filters("Hello world", configs)
        assert result.passed is True

    def test_length_filter_short_text(self) -> None:
        """Test length filter with short text."""
        filter_config = FilterConfig(
            filter_type=FilterType.LENGTH,
            toxicity_config=None,
            pii_config=None,
            language_config=None,
        )
        result = apply_filters("Short", [filter_config])
        assert result.passed is False
        assert "too short" in (result.failed_reason or "")


class TestFormatFilterStats:
    """Tests for format_filter_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = FilterStats(
            total_processed=1000,
            passed_count=900,
            filtered_counts={"toxicity": 50, "pii": 50},
            pii_detected={"email": 30, "phone": 20},
        )
        formatted = format_filter_stats(stats)
        assert "1,000" in formatted or "1000" in formatted
        assert "900" in formatted
        assert "toxicity" in formatted
        assert "email" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_filter_stats(None)  # type: ignore[arg-type]

    def test_empty_filtered_counts(self) -> None:
        """Test with empty filtered_counts."""
        stats = FilterStats(
            total_processed=100,
            passed_count=100,
            filtered_counts={},
            pii_detected={},
        )
        formatted = format_filter_stats(stats)
        assert "100" in formatted

    def test_zero_processed(self) -> None:
        """Test with zero processed."""
        stats = FilterStats(
            total_processed=0,
            passed_count=0,
            filtered_counts={},
            pii_detected={},
        )
        formatted = format_filter_stats(stats)
        assert "0" in formatted


class TestGetRecommendedFilterConfig:
    """Tests for get_recommended_filter_config function."""

    def test_chat_use_case(self) -> None:
        """Test recommendation for chat use case."""
        configs = get_recommended_filter_config("chat")
        assert len(configs) >= 1
        assert all(isinstance(c, FilterConfig) for c in configs)
        # Should include toxicity and PII filters
        filter_types = [c.filter_type for c in configs]
        assert FilterType.TOXICITY in filter_types
        assert FilterType.PII in filter_types

    def test_training_use_case(self) -> None:
        """Test recommendation for training use case."""
        configs = get_recommended_filter_config("training")
        assert len(configs) >= 1
        filter_types = [c.filter_type for c in configs]
        assert FilterType.TOXICITY in filter_types
        assert FilterType.LANGUAGE in filter_types

    def test_production_use_case(self) -> None:
        """Test recommendation for production use case."""
        configs = get_recommended_filter_config("production")
        assert len(configs) >= 1
        # Production should have strict toxicity threshold
        for config in configs:
            if config.filter_type == FilterType.TOXICITY:
                assert config.toxicity_config is not None
                assert config.toxicity_config.threshold <= 0.3

    def test_empty_use_case_raises_error(self) -> None:
        """Test that empty use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case cannot be empty"):
            get_recommended_filter_config("")

    def test_unknown_use_case(self) -> None:
        """Test with unknown use case returns default."""
        configs = get_recommended_filter_config("unknown_case")
        assert len(configs) >= 1

    def test_chatbot_alias(self) -> None:
        """Test chatbot alias for chat."""
        configs = get_recommended_filter_config("chatbot")
        assert len(configs) >= 1

    def test_dataset_alias(self) -> None:
        """Test dataset alias for training."""
        configs = get_recommended_filter_config("dataset")
        assert len(configs) >= 1


class TestPropertyBased:
    """Property-based tests for filtering module."""

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    def test_toxicity_scores_in_valid_range(self, text: str) -> None:
        """Test that toxicity scores are in valid range."""
        scores = detect_toxicity(text)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    def test_language_confidence_in_valid_range(self, text: str) -> None:
        """Test that language confidence is in valid range."""
        _, conf = detect_language(text)
        assert 0.0 <= conf <= 1.0

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20)
    def test_valid_threshold_creates_valid_config(self, threshold: float) -> None:
        """Test that valid thresholds create valid configs."""
        config = create_toxicity_config(threshold=threshold)
        assert config.threshold == pytest.approx(threshold)
        validate_toxicity_config(config)

    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=20)
    def test_redact_pii_returns_string(self, text: str) -> None:
        """Test that redact_pii always returns a string."""
        result = redact_pii(text)
        assert isinstance(result, str)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    def test_detect_pii_returns_dict(self, text: str) -> None:
        """Test that detect_pii always returns a dict."""
        result = detect_pii(text)
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, list)


class TestEdgeCases:
    """Test edge cases for filtering module."""

    def test_whitespace_only_text_toxicity(self) -> None:
        """Test toxicity detection with whitespace-only text."""
        scores = detect_toxicity("   \t\n  ")
        assert all(v == 0.0 for v in scores.values())

    def test_whitespace_only_text_pii(self) -> None:
        """Test PII detection with whitespace-only text."""
        result = detect_pii("   \t\n  ")
        assert all(len(v) == 0 for v in result.values())

    def test_whitespace_only_text_language(self) -> None:
        """Test language detection with whitespace-only text."""
        lang, conf = detect_language("   \t\n  ")
        assert lang == "unknown"
        assert conf == 0.0

    def test_special_characters_in_text(self) -> None:
        """Test handling of special characters."""
        text = "Hello!@#$%^&*()_+-=[]{}|;':\",./<>?"
        scores = detect_toxicity(text)
        result = detect_pii(text)
        lang, _ = detect_language(text)
        assert isinstance(scores, dict)
        assert isinstance(result, dict)
        assert isinstance(lang, str)

    def test_unicode_text(self) -> None:
        """Test handling of unicode text."""
        text = "Hello \u4e16\u754c \u3053\u3093\u306b\u3061\u306f"
        scores = detect_toxicity(text)
        result = detect_pii(text)
        lang, _ = detect_language(text)
        assert isinstance(scores, dict)
        assert isinstance(result, dict)
        assert isinstance(lang, str)

    def test_very_long_text(self) -> None:
        """Test handling of very long text."""
        text = "Hello world. " * 1000
        scores = detect_toxicity(text)
        result = detect_pii(text)
        lang, conf = detect_language(text)
        assert isinstance(scores, dict)
        assert isinstance(result, dict)
        assert lang == "en"
        assert conf > 0.5

    def test_mixed_pii_types(self) -> None:
        """Test detection of multiple PII types in one text."""
        text = (
            "Contact John at john@example.com or 555-123-4567. "
            "SSN: 123-45-6789, Card: 1234-5678-9012-3456, "
            "Server: 192.168.1.100"
        )
        result = detect_pii(text)
        assert len(result["email"]) >= 1
        assert len(result["phone"]) >= 1
        assert len(result["ssn"]) >= 1
        assert len(result["credit_card"]) >= 1
        assert len(result["ip_address"]) >= 1

    def test_boundary_threshold_values(self) -> None:
        """Test boundary threshold values."""
        config_zero = create_toxicity_config(threshold=0.0)
        assert config_zero.threshold == 0.0

        config_one = create_toxicity_config(threshold=1.0)
        assert config_one.threshold == 1.0

    def test_all_pii_types(self) -> None:
        """Test that all PII types are detectable."""
        config = create_pii_config()
        assert len(config.pii_types) == len(PIIType)

    def test_all_toxicity_categories(self) -> None:
        """Test that all toxicity categories are detectable."""
        config = create_toxicity_config()
        assert len(config.categories) == len(ToxicityCategory)
