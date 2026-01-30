"""Tests for content safety guardrails functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.safety.guardrails import (
    VALID_ACTION_TYPES,
    VALID_CONTENT_CATEGORIES,
    VALID_GUARDRAIL_TYPES,
    ActionType,
    ContentCategory,
    ContentPolicyConfig,
    GuardrailConfig,
    GuardrailResult,
    GuardrailType,
    PIIConfig,
    ToxicityConfig,
    calculate_safety_score,
    check_content_safety,
    create_content_policy_config,
    create_guardrail_config,
    create_pii_config,
    create_toxicity_config,
    get_action_type,
    get_content_category,
    get_guardrail_type,
    list_action_types,
    list_content_categories,
    list_guardrail_types,
    validate_content_policy_config,
    validate_guardrail_config,
)


class TestGuardrailType:
    """Tests for GuardrailType enum."""

    def test_input_filter_value(self) -> None:
        """Test INPUT_FILTER value."""
        assert GuardrailType.INPUT_FILTER.value == "input_filter"

    def test_output_filter_value(self) -> None:
        """Test OUTPUT_FILTER value."""
        assert GuardrailType.OUTPUT_FILTER.value == "output_filter"

    def test_topic_filter_value(self) -> None:
        """Test TOPIC_FILTER value."""
        assert GuardrailType.TOPIC_FILTER.value == "topic_filter"

    def test_pii_filter_value(self) -> None:
        """Test PII_FILTER value."""
        assert GuardrailType.PII_FILTER.value == "pii_filter"

    def test_toxicity_filter_value(self) -> None:
        """Test TOXICITY_FILTER value."""
        assert GuardrailType.TOXICITY_FILTER.value == "toxicity_filter"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [g.value for g in GuardrailType]
        assert len(values) == len(set(values))


class TestContentCategory:
    """Tests for ContentCategory enum."""

    def test_safe_value(self) -> None:
        """Test SAFE value."""
        assert ContentCategory.SAFE.value == "safe"

    def test_unsafe_value(self) -> None:
        """Test UNSAFE value."""
        assert ContentCategory.UNSAFE.value == "unsafe"

    def test_hate_value(self) -> None:
        """Test HATE value."""
        assert ContentCategory.HATE.value == "hate"

    def test_violence_value(self) -> None:
        """Test VIOLENCE value."""
        assert ContentCategory.VIOLENCE.value == "violence"

    def test_sexual_value(self) -> None:
        """Test SEXUAL value."""
        assert ContentCategory.SEXUAL.value == "sexual"

    def test_self_harm_value(self) -> None:
        """Test SELF_HARM value."""
        assert ContentCategory.SELF_HARM.value == "self_harm"

    def test_dangerous_value(self) -> None:
        """Test DANGEROUS value."""
        assert ContentCategory.DANGEROUS.value == "dangerous"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [c.value for c in ContentCategory]
        assert len(values) == len(set(values))


class TestActionType:
    """Tests for ActionType enum."""

    def test_allow_value(self) -> None:
        """Test ALLOW value."""
        assert ActionType.ALLOW.value == "allow"

    def test_block_value(self) -> None:
        """Test BLOCK value."""
        assert ActionType.BLOCK.value == "block"

    def test_warn_value(self) -> None:
        """Test WARN value."""
        assert ActionType.WARN.value == "warn"

    def test_redact_value(self) -> None:
        """Test REDACT value."""
        assert ActionType.REDACT.value == "redact"

    def test_rephrase_value(self) -> None:
        """Test REPHRASE value."""
        assert ActionType.REPHRASE.value == "rephrase"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [a.value for a in ActionType]
        assert len(values) == len(set(values))


class TestValidFrozensets:
    """Tests for VALID_* frozenset constants."""

    def test_valid_guardrail_types_contains_all_enums(self) -> None:
        """Test VALID_GUARDRAIL_TYPES contains all enum values."""
        for g in GuardrailType:
            assert g.value in VALID_GUARDRAIL_TYPES

    def test_valid_content_categories_contains_all_enums(self) -> None:
        """Test VALID_CONTENT_CATEGORIES contains all enum values."""
        for c in ContentCategory:
            assert c.value in VALID_CONTENT_CATEGORIES

    def test_valid_action_types_contains_all_enums(self) -> None:
        """Test VALID_ACTION_TYPES contains all enum values."""
        for a in ActionType:
            assert a.value in VALID_ACTION_TYPES

    def test_frozensets_are_immutable(self) -> None:
        """Test that frozensets cannot be modified."""
        with pytest.raises(AttributeError):
            VALID_GUARDRAIL_TYPES.add("new")  # type: ignore[attr-defined]


class TestGuardrailConfig:
    """Tests for GuardrailConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating GuardrailConfig instance."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=0.8,
            enabled=True,
        )
        assert config.guardrail_type == GuardrailType.TOXICITY_FILTER
        assert config.action_on_violation == ActionType.BLOCK
        assert config.threshold == 0.8
        assert config.enabled is True

    def test_default_values(self) -> None:
        """Test default values for GuardrailConfig."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.PII_FILTER,
            action_on_violation=ActionType.REDACT,
        )
        assert config.threshold == 0.5
        assert config.enabled is True

    def test_frozen(self) -> None:
        """Test that GuardrailConfig is immutable."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.INPUT_FILTER,
            action_on_violation=ActionType.ALLOW,
        )
        with pytest.raises(AttributeError):
            config.threshold = 0.9  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that GuardrailConfig uses slots."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.INPUT_FILTER,
            action_on_violation=ActionType.ALLOW,
        )
        # Frozen dataclasses with slots don't have __dict__
        assert not hasattr(config, "__dict__")


class TestContentPolicyConfig:
    """Tests for ContentPolicyConfig dataclass."""

    def test_creation_with_values(self) -> None:
        """Test creating ContentPolicyConfig with values."""
        cats = frozenset({ContentCategory.HATE, ContentCategory.VIOLENCE})
        config = ContentPolicyConfig(
            blocked_categories=cats,
            allowed_topics=frozenset({"science", "technology"}),
            custom_rules=("pattern1", "pattern2"),
        )
        assert ContentCategory.HATE in config.blocked_categories
        assert "science" in config.allowed_topics
        assert "pattern1" in config.custom_rules

    def test_default_values(self) -> None:
        """Test default values for ContentPolicyConfig."""
        config = ContentPolicyConfig()
        assert config.blocked_categories == frozenset()
        assert config.allowed_topics == frozenset()
        assert config.custom_rules == ()

    def test_frozen(self) -> None:
        """Test that ContentPolicyConfig is immutable."""
        config = ContentPolicyConfig()
        with pytest.raises(AttributeError):
            config.blocked_categories = frozenset({ContentCategory.HATE})  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that ContentPolicyConfig uses slots."""
        config = ContentPolicyConfig()
        # Frozen dataclasses with slots don't have __dict__
        assert not hasattr(config, "__dict__")


class TestPIIConfig:
    """Tests for PIIConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating PIIConfig instance."""
        config = PIIConfig(
            detect_email=True,
            detect_phone=False,
            detect_ssn=True,
            detect_credit_card=False,
            redaction_char="X",
        )
        assert config.detect_email is True
        assert config.detect_phone is False
        assert config.detect_ssn is True
        assert config.detect_credit_card is False
        assert config.redaction_char == "X"

    def test_default_values(self) -> None:
        """Test default values for PIIConfig."""
        config = PIIConfig()
        assert config.detect_email is True
        assert config.detect_phone is True
        assert config.detect_ssn is True
        assert config.detect_credit_card is True
        assert config.redaction_char == "*"

    def test_frozen(self) -> None:
        """Test that PIIConfig is immutable."""
        config = PIIConfig()
        with pytest.raises(AttributeError):
            config.detect_email = False  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that PIIConfig uses slots."""
        config = PIIConfig()
        # Frozen dataclasses with slots don't have __dict__
        assert not hasattr(config, "__dict__")


class TestToxicityConfig:
    """Tests for ToxicityConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ToxicityConfig instance."""
        config = ToxicityConfig(
            model_id="custom/model",
            threshold=0.7,
            categories=("toxic", "insult"),
        )
        assert config.model_id == "custom/model"
        assert config.threshold == 0.7
        assert config.categories == ("toxic", "insult")

    def test_default_values(self) -> None:
        """Test default values for ToxicityConfig."""
        config = ToxicityConfig()
        assert config.model_id == "unitary/toxic-bert"
        assert config.threshold == 0.5
        assert "toxic" in config.categories
        assert "severe_toxic" in config.categories

    def test_frozen(self) -> None:
        """Test that ToxicityConfig is immutable."""
        config = ToxicityConfig()
        with pytest.raises(AttributeError):
            config.threshold = 0.9  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that ToxicityConfig uses slots."""
        config = ToxicityConfig()
        # Frozen dataclasses with slots don't have __dict__
        assert not hasattr(config, "__dict__")


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""

    def test_creation_safe(self) -> None:
        """Test creating safe GuardrailResult."""
        result = GuardrailResult(
            is_safe=True,
            violations=[],
            action_taken=ActionType.ALLOW,
        )
        assert result.is_safe is True
        assert result.violations == []
        assert result.action_taken == ActionType.ALLOW
        assert result.modified_content is None

    def test_creation_unsafe(self) -> None:
        """Test creating unsafe GuardrailResult."""
        result = GuardrailResult(
            is_safe=False,
            violations=["PII detected: email"],
            action_taken=ActionType.REDACT,
            modified_content="Contact: ****@****.***",
        )
        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.action_taken == ActionType.REDACT
        assert result.modified_content is not None

    def test_default_values(self) -> None:
        """Test default values for GuardrailResult."""
        result = GuardrailResult(is_safe=True)
        assert result.violations == []
        assert result.action_taken == ActionType.ALLOW
        assert result.modified_content is None

    def test_frozen(self) -> None:
        """Test that GuardrailResult is immutable."""
        result = GuardrailResult(is_safe=True)
        with pytest.raises(AttributeError):
            result.is_safe = False  # type: ignore[misc]


class TestValidateGuardrailConfig:
    """Tests for validate_guardrail_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=0.8,
        )
        validate_guardrail_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_guardrail_config(None)  # type: ignore[arg-type]

    def test_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0 raises ValueError."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=-0.1,
        )
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_guardrail_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=1.1,
        )
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_guardrail_config(config)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_boundary_thresholds_valid(self, threshold: float) -> None:
        """Test that boundary thresholds are valid."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=threshold,
        )
        validate_guardrail_config(config)  # Should not raise


class TestValidateContentPolicyConfig:
    """Tests for validate_content_policy_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ContentPolicyConfig(
            blocked_categories=frozenset({ContentCategory.HATE}),
            custom_rules=("bad_word",),
        )
        validate_content_policy_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_content_policy_config(None)  # type: ignore[arg-type]

    def test_invalid_regex_raises_error(self) -> None:
        """Test that invalid regex raises ValueError."""
        config = ContentPolicyConfig(custom_rules=("[invalid",))
        with pytest.raises(ValueError, match="invalid regex pattern"):
            validate_content_policy_config(config)

    def test_multiple_invalid_regex_first_caught(self) -> None:
        """Test that first invalid regex is caught."""
        config = ContentPolicyConfig(custom_rules=("[first", "[second"))
        with pytest.raises(ValueError, match="\\[first"):
            validate_content_policy_config(config)

    def test_valid_regex_patterns(self) -> None:
        """Test that valid regex patterns pass validation."""
        config = ContentPolicyConfig(
            custom_rules=(r"\bword\b", r"pattern.*test", r"^start")
        )
        validate_content_policy_config(config)  # Should not raise


class TestCreateGuardrailConfig:
    """Tests for create_guardrail_config function."""

    def test_create_with_enum_types(self) -> None:
        """Test creating config with enum types."""
        config = create_guardrail_config(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=0.8,
        )
        assert config.guardrail_type == GuardrailType.TOXICITY_FILTER
        assert config.action_on_violation == ActionType.BLOCK
        assert config.threshold == 0.8

    def test_create_with_string_types(self) -> None:
        """Test creating config with string types."""
        config = create_guardrail_config(
            guardrail_type="pii_filter",
            action_on_violation="redact",
        )
        assert config.guardrail_type == GuardrailType.PII_FILTER
        assert config.action_on_violation == ActionType.REDACT

    def test_invalid_guardrail_type_string(self) -> None:
        """Test that invalid guardrail type string raises ValueError."""
        with pytest.raises(ValueError, match="invalid guardrail type"):
            create_guardrail_config("invalid")

    def test_invalid_action_type_string(self) -> None:
        """Test that invalid action type string raises ValueError."""
        with pytest.raises(ValueError, match="invalid action type"):
            create_guardrail_config(
                guardrail_type=GuardrailType.PII_FILTER,
                action_on_violation="invalid",
            )

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            create_guardrail_config(
                guardrail_type=GuardrailType.PII_FILTER,
                threshold=1.5,
            )

    def test_default_values(self) -> None:
        """Test default values in create_guardrail_config."""
        config = create_guardrail_config(GuardrailType.INPUT_FILTER)
        assert config.action_on_violation == ActionType.BLOCK
        assert config.threshold == 0.5
        assert config.enabled is True


class TestCreateContentPolicyConfig:
    """Tests for create_content_policy_config function."""

    def test_create_with_values(self) -> None:
        """Test creating config with values."""
        config = create_content_policy_config(
            blocked_categories=frozenset({ContentCategory.HATE}),
            allowed_topics=frozenset({"science"}),
            custom_rules=("pattern",),
        )
        assert ContentCategory.HATE in config.blocked_categories
        assert "science" in config.allowed_topics
        assert "pattern" in config.custom_rules

    def test_create_with_none_values(self) -> None:
        """Test creating config with None values."""
        config = create_content_policy_config()
        assert config.blocked_categories == frozenset()
        assert config.allowed_topics == frozenset()
        assert config.custom_rules == ()

    def test_list_custom_rules_converted_to_tuple(self) -> None:
        """Test that list custom_rules are converted to tuple."""
        config = create_content_policy_config(custom_rules=["a", "b"])
        assert config.custom_rules == ("a", "b")

    def test_invalid_regex_raises_error(self) -> None:
        """Test that invalid regex raises ValueError."""
        with pytest.raises(ValueError, match="invalid regex pattern"):
            create_content_policy_config(custom_rules=["[invalid"])


class TestCreatePIIConfig:
    """Tests for create_pii_config function."""

    def test_create_with_values(self) -> None:
        """Test creating config with values."""
        config = create_pii_config(
            detect_email=True,
            detect_phone=False,
            redaction_char="X",
        )
        assert config.detect_email is True
        assert config.detect_phone is False
        assert config.redaction_char == "X"

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_pii_config()
        assert config.detect_email is True
        assert config.detect_phone is True
        assert config.detect_ssn is True
        assert config.detect_credit_card is True
        assert config.redaction_char == "*"

    def test_empty_redaction_char_raises_error(self) -> None:
        """Test that empty redaction char raises ValueError."""
        with pytest.raises(ValueError, match="redaction_char must be a single"):
            create_pii_config(redaction_char="")

    def test_multiple_char_redaction_raises_error(self) -> None:
        """Test that multi-char redaction raises ValueError."""
        with pytest.raises(ValueError, match="redaction_char must be a single"):
            create_pii_config(redaction_char="XX")


class TestCreateToxicityConfig:
    """Tests for create_toxicity_config function."""

    def test_create_with_values(self) -> None:
        """Test creating config with values."""
        config = create_toxicity_config(
            model_id="custom/model",
            threshold=0.7,
            categories=("toxic", "insult"),
        )
        assert config.model_id == "custom/model"
        assert config.threshold == 0.7
        assert config.categories == ("toxic", "insult")

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_toxicity_config()
        assert config.model_id == "unitary/toxic-bert"
        assert config.threshold == 0.5
        assert "toxic" in config.categories

    def test_empty_model_id_raises_error(self) -> None:
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            create_toxicity_config(model_id="")

    def test_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            create_toxicity_config(threshold=-0.1)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            create_toxicity_config(threshold=1.1)

    def test_empty_categories_raises_error(self) -> None:
        """Test that empty categories raises ValueError."""
        with pytest.raises(ValueError, match="categories cannot be empty"):
            create_toxicity_config(categories=())

    def test_list_categories_converted_to_tuple(self) -> None:
        """Test that list categories are converted to tuple."""
        config = create_toxicity_config(categories=["toxic", "insult"])
        assert config.categories == ("toxic", "insult")


class TestListGuardrailTypes:
    """Tests for list_guardrail_types function."""

    def test_returns_list(self) -> None:
        """Test that list_guardrail_types returns a list."""
        result = list_guardrail_types()
        assert isinstance(result, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        result = list_guardrail_types()
        assert "input_filter" in result
        assert "output_filter" in result
        assert "pii_filter" in result
        assert "toxicity_filter" in result

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_guardrail_types()
        assert result == sorted(result)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        result = list_guardrail_types()
        assert all(isinstance(t, str) for t in result)


class TestListContentCategories:
    """Tests for list_content_categories function."""

    def test_returns_list(self) -> None:
        """Test that list_content_categories returns a list."""
        result = list_content_categories()
        assert isinstance(result, list)

    def test_contains_expected_categories(self) -> None:
        """Test that list contains expected categories."""
        result = list_content_categories()
        assert "safe" in result
        assert "hate" in result
        assert "violence" in result

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_content_categories()
        assert result == sorted(result)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        result = list_content_categories()
        assert all(isinstance(c, str) for c in result)


class TestListActionTypes:
    """Tests for list_action_types function."""

    def test_returns_list(self) -> None:
        """Test that list_action_types returns a list."""
        result = list_action_types()
        assert isinstance(result, list)

    def test_contains_expected_actions(self) -> None:
        """Test that list contains expected actions."""
        result = list_action_types()
        assert "allow" in result
        assert "block" in result
        assert "redact" in result

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_action_types()
        assert result == sorted(result)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        result = list_action_types()
        assert all(isinstance(a, str) for a in result)


class TestGetGuardrailType:
    """Tests for get_guardrail_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("input_filter", GuardrailType.INPUT_FILTER),
            ("output_filter", GuardrailType.OUTPUT_FILTER),
            ("topic_filter", GuardrailType.TOPIC_FILTER),
            ("pii_filter", GuardrailType.PII_FILTER),
            ("toxicity_filter", GuardrailType.TOXICITY_FILTER),
        ],
    )
    def test_valid_names(self, name: str, expected: GuardrailType) -> None:
        """Test getting guardrail type by valid name."""
        assert get_guardrail_type(name) == expected

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="invalid guardrail type"):
            get_guardrail_type("invalid")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="invalid guardrail type"):
            get_guardrail_type("")


class TestGetContentCategory:
    """Tests for get_content_category function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("safe", ContentCategory.SAFE),
            ("unsafe", ContentCategory.UNSAFE),
            ("hate", ContentCategory.HATE),
            ("violence", ContentCategory.VIOLENCE),
            ("sexual", ContentCategory.SEXUAL),
            ("self_harm", ContentCategory.SELF_HARM),
            ("dangerous", ContentCategory.DANGEROUS),
        ],
    )
    def test_valid_names(self, name: str, expected: ContentCategory) -> None:
        """Test getting content category by valid name."""
        assert get_content_category(name) == expected

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="invalid content category"):
            get_content_category("invalid")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="invalid content category"):
            get_content_category("")


class TestGetActionType:
    """Tests for get_action_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("allow", ActionType.ALLOW),
            ("block", ActionType.BLOCK),
            ("warn", ActionType.WARN),
            ("redact", ActionType.REDACT),
            ("rephrase", ActionType.REPHRASE),
        ],
    )
    def test_valid_names(self, name: str, expected: ActionType) -> None:
        """Test getting action type by valid name."""
        assert get_action_type(name) == expected

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="invalid action type"):
            get_action_type("invalid")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="invalid action type"):
            get_action_type("")


class TestCheckContentSafety:
    """Tests for check_content_safety function."""

    def test_safe_content_no_config(self) -> None:
        """Test safe content without any config."""
        result = check_content_safety("Hello, world!")
        assert result.is_safe is True
        assert result.action_taken == ActionType.ALLOW

    def test_none_content_raises_error(self) -> None:
        """Test that None content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be None"):
            check_content_safety(None)  # type: ignore[arg-type]

    def test_empty_content_is_safe(self) -> None:
        """Test that empty content is considered safe."""
        result = check_content_safety("")
        assert result.is_safe is True

    def test_disabled_guardrail_allows_content(self) -> None:
        """Test that disabled guardrail allows all content."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.PII_FILTER,
            action_on_violation=ActionType.BLOCK,
            enabled=False,
        )
        pii_config = create_pii_config()
        result = check_content_safety(
            "Email: test@example.com",
            guardrail_config=config,
            pii_config=pii_config,
        )
        assert result.is_safe is True
        assert result.action_taken == ActionType.ALLOW

    def test_email_detection(self) -> None:
        """Test email PII detection."""
        pii_config = create_pii_config()
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.WARN,
        )
        result = check_content_safety(
            "Contact: test@example.com",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert "email" in result.violations[0].lower()

    def test_phone_detection(self) -> None:
        """Test phone number PII detection."""
        pii_config = create_pii_config()
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.WARN,
        )
        result = check_content_safety(
            "Call: 555-123-4567",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert "phone" in result.violations[0].lower()

    def test_ssn_detection(self) -> None:
        """Test SSN PII detection."""
        pii_config = create_pii_config()
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.WARN,
        )
        result = check_content_safety(
            "SSN: 123-45-6789",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert "ssn" in result.violations[0].lower()

    def test_credit_card_detection(self) -> None:
        """Test credit card PII detection."""
        pii_config = create_pii_config()
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.WARN,
        )
        result = check_content_safety(
            "Card: 4111-1111-1111-1111",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert "credit_card" in result.violations[0].lower()

    def test_redaction_action(self) -> None:
        """Test redaction of PII."""
        pii_config = create_pii_config()
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "Email: test@example.com",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert result.action_taken == ActionType.REDACT
        assert result.modified_content is not None
        assert "*" in result.modified_content
        assert "test@example.com" not in result.modified_content

    def test_custom_redaction_char(self) -> None:
        """Test custom redaction character."""
        pii_config = create_pii_config(redaction_char="X")
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "Email: test@example.com",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.modified_content is not None
        assert "X" in result.modified_content

    def test_custom_rule_violation(self) -> None:
        """Test custom rule violation detection."""
        policy_config = create_content_policy_config(
            custom_rules=(r"\bforbidden\b",),
        )
        result = check_content_safety(
            "This is forbidden content",
            policy_config=policy_config,
        )
        assert result.is_safe is False
        assert "Custom rule violation" in result.violations[0]

    def test_multiple_violations(self) -> None:
        """Test detection of multiple violations."""
        pii_config = create_pii_config()
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.WARN,
        )
        result = check_content_safety(
            "Email: test@example.com, Phone: 555-123-4567",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert len(result.violations) == 2

    def test_no_guardrail_config_uses_warn(self) -> None:
        """Test that no guardrail config uses WARN action."""
        pii_config = create_pii_config()
        result = check_content_safety(
            "Email: test@example.com",
            pii_config=pii_config,
        )
        assert result.is_safe is False
        assert result.action_taken == ActionType.WARN

    def test_disabled_pii_detection(self) -> None:
        """Test disabling specific PII detection."""
        pii_config = create_pii_config(detect_email=False)
        result = check_content_safety(
            "Email: test@example.com",
            pii_config=pii_config,
        )
        assert result.is_safe is True

    def test_custom_rule_no_match(self) -> None:
        """Test custom rule that doesn't match content."""
        policy_config = create_content_policy_config(
            custom_rules=(r"\bforbidden\b",),
        )
        result = check_content_safety(
            "This is allowed content",
            policy_config=policy_config,
        )
        assert result.is_safe is True
        assert len(result.violations) == 0

    def test_multiple_custom_rules_partial_match(self) -> None:
        """Test multiple custom rules with only some matching."""
        policy_config = create_content_policy_config(
            custom_rules=(r"\bforbidden\b", r"\ballowed\b"),
        )
        result = check_content_safety(
            "This content is forbidden",
            policy_config=policy_config,
        )
        assert result.is_safe is False
        assert len(result.violations) == 1  # Only "forbidden" matches


class TestCalculateSafetyScore:
    """Tests for calculate_safety_score function."""

    def test_safe_content_score_1(self) -> None:
        """Test safe content returns score of 1.0."""
        score = calculate_safety_score("Hello, world!")
        assert score == 1.0

    def test_empty_content_score_1(self) -> None:
        """Test empty content returns score of 1.0."""
        score = calculate_safety_score("")
        assert score == 1.0

    def test_none_content_raises_error(self) -> None:
        """Test that None content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be None"):
            calculate_safety_score(None)  # type: ignore[arg-type]

    def test_single_violation_reduces_score(self) -> None:
        """Test that single violation reduces score."""
        pii_config = create_pii_config()
        score = calculate_safety_score(
            "Email: test@example.com",
            pii_config=pii_config,
        )
        assert 0.0 <= score < 1.0
        assert score == 0.8  # 1.0 - 0.2

    def test_multiple_violations_reduce_score_more(self) -> None:
        """Test that multiple violations reduce score more."""
        pii_config = create_pii_config()
        score = calculate_safety_score(
            "Email: test@example.com, Phone: 555-123-4567",
            pii_config=pii_config,
        )
        assert score == 0.6  # 1.0 - 0.4

    def test_many_violations_min_score_zero(self) -> None:
        """Test that many violations result in minimum score of 0.0."""
        pii_config = create_pii_config()
        score = calculate_safety_score(
            "Email: a@b.com b@c.com c@d.com d@e.com e@f.com f@g.com",
            pii_config=pii_config,
        )
        # Score should not go below 0.0
        assert score >= 0.0

    def test_policy_violations_counted(self) -> None:
        """Test that policy violations are counted."""
        policy_config = create_content_policy_config(
            custom_rules=(r"bad",),
        )
        score = calculate_safety_score(
            "bad bad bad",  # Multiple matches
            policy_config=policy_config,
        )
        assert score < 1.0

    def test_no_config_returns_1(self) -> None:
        """Test that no config returns score of 1.0."""
        score = calculate_safety_score("test@example.com")
        assert score == 1.0  # Without PII config, no detection


class TestPIIPatterns:
    """Tests for PII detection patterns."""

    @pytest.mark.parametrize(
        "email",
        [
            "test@example.com",
            "user.name@domain.org",
            "first+last@company.co.uk",
            "user123@test.io",
        ],
    )
    def test_email_patterns(self, email: str) -> None:
        """Test various email patterns are detected."""
        pii_config = create_pii_config()
        result = check_content_safety(
            f"Contact: {email}",
            pii_config=pii_config,
        )
        assert "email" in result.violations[0].lower()

    @pytest.mark.parametrize(
        "phone",
        [
            "555-123-4567",
            "555.123.4567",
            "(555) 123-4567",
            "+1 555-123-4567",
        ],
    )
    def test_phone_patterns(self, phone: str) -> None:
        """Test various phone patterns are detected."""
        pii_config = create_pii_config()
        result = check_content_safety(
            f"Call: {phone}",
            pii_config=pii_config,
        )
        assert "phone" in result.violations[0].lower()

    @pytest.mark.parametrize(
        "ssn",
        [
            "123-45-6789",
            "123.45.6789",
            "123 45 6789",
        ],
    )
    def test_ssn_patterns(self, ssn: str) -> None:
        """Test various SSN patterns are detected."""
        pii_config = create_pii_config()
        result = check_content_safety(
            f"SSN: {ssn}",
            pii_config=pii_config,
        )
        assert "ssn" in result.violations[0].lower()

    @pytest.mark.parametrize(
        "cc",
        [
            "4111-1111-1111-1111",
            "5500 0000 0000 0004",
        ],
    )
    def test_credit_card_patterns(self, cc: str) -> None:
        """Test various credit card patterns are detected."""
        pii_config = create_pii_config()
        result = check_content_safety(
            f"Card: {cc}",
            pii_config=pii_config,
        )
        assert "credit_card" in result.violations[0].lower()

    def test_credit_card_16_digit_no_separators(self) -> None:
        """Test 16-digit credit card without separators."""
        # Note: 16-digit numbers without separators may also match phone patterns
        # because the phone pattern is flexible. We just ensure credit_card is detected.
        pii_config = create_pii_config(detect_phone=False)
        result = check_content_safety(
            "Card: 4111111111111111",
            pii_config=pii_config,
        )
        assert any("credit_card" in v.lower() for v in result.violations)


class TestRedactionWithDisabledDetection:
    """Tests for PII redaction with specific detection types disabled."""

    def test_redaction_with_email_disabled(self) -> None:
        """Test redaction when email detection is disabled."""
        pii_config = create_pii_config(detect_email=False)
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "Email: test@example.com, Phone: 555-123-4567",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        # Email should still be visible, phone should be redacted
        assert "test@example.com" in (result.modified_content or "")
        assert "555-123-4567" not in (result.modified_content or "")

    def test_redaction_with_phone_disabled(self) -> None:
        """Test redaction when phone detection is disabled."""
        pii_config = create_pii_config(detect_phone=False)
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "Email: test@example.com, Phone: 555-123-4567",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        # Phone should still be visible, email should be redacted
        assert "test@example.com" not in (result.modified_content or "")
        assert "555-123-4567" in (result.modified_content or "")

    def test_redaction_with_ssn_disabled(self) -> None:
        """Test redaction when SSN detection is disabled."""
        pii_config = create_pii_config(detect_ssn=False)
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "SSN: 123-45-6789, Email: test@example.com",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        # SSN should still be visible, email should be redacted
        assert "123-45-6789" in (result.modified_content or "")
        assert "test@example.com" not in (result.modified_content or "")

    def test_redaction_with_credit_card_disabled(self) -> None:
        """Test redaction when credit card detection is disabled."""
        pii_config = create_pii_config(detect_credit_card=False)
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "Card: 4111-1111-1111-1111, Email: test@example.com",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        # Credit card should still be visible, email should be redacted
        assert "4111-1111-1111-1111" in (result.modified_content or "")
        assert "test@example.com" not in (result.modified_content or "")

    def test_redaction_all_disabled_except_email(self) -> None:
        """Test redaction with only email detection enabled."""
        pii_config = create_pii_config(
            detect_email=True,
            detect_phone=False,
            detect_ssn=False,
            detect_credit_card=False,
        )
        guardrail_config = create_guardrail_config(
            GuardrailType.PII_FILTER,
            ActionType.REDACT,
        )
        result = check_content_safety(
            "Email: test@example.com, Phone: 555-123-4567",
            guardrail_config=guardrail_config,
            pii_config=pii_config,
        )
        # Only email should be redacted
        assert "test@example.com" not in (result.modified_content or "")
        assert "555-123-4567" in (result.modified_content or "")


class TestHypothesis:
    """Property-based tests using Hypothesis."""

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_safety_check_never_raises_on_string(self, content: str) -> None:
        """Test that safety check never raises on any string input."""
        result = check_content_safety(content)
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.is_safe, bool)

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20)
    def test_valid_threshold_accepted(self, threshold: float) -> None:
        """Test that valid thresholds are accepted."""
        config = GuardrailConfig(
            guardrail_type=GuardrailType.TOXICITY_FILTER,
            action_on_violation=ActionType.BLOCK,
            threshold=threshold,
        )
        validate_guardrail_config(config)  # Should not raise

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_safety_score_in_range(self, content: str) -> None:
        """Test that safety score is always in valid range."""
        score = calculate_safety_score(content)
        assert 0.0 <= score <= 1.0

    @given(st.sampled_from(list(GuardrailType)))
    def test_all_guardrail_types_have_string_value(
        self, guardrail_type: GuardrailType
    ) -> None:
        """Test that all guardrail types have string values."""
        result = get_guardrail_type(guardrail_type.value)
        assert result == guardrail_type

    @given(st.sampled_from(list(ContentCategory)))
    def test_all_content_categories_have_string_value(
        self, category: ContentCategory
    ) -> None:
        """Test that all content categories have string values."""
        result = get_content_category(category.value)
        assert result == category

    @given(st.sampled_from(list(ActionType)))
    def test_all_action_types_have_string_value(self, action: ActionType) -> None:
        """Test that all action types have string values."""
        result = get_action_type(action.value)
        assert result == action
