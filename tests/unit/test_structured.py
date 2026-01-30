"""Tests for generation.structured module."""

from __future__ import annotations

import pytest

from hf_gtc.generation.structured import (
    VALID_OUTPUT_FORMATS,
    VALID_SCHEMA_TYPES,
    ChoicesConfig,
    GrammarConfig,
    JSONSchemaConfig,
    OutputFormat,
    SchemaType,
    StructuredConfig,
    ValidationResult,
    create_choices_config,
    create_grammar_config,
    create_json_schema_config,
    create_structured_config,
    create_validation_result,
    estimate_structured_tokens,
    format_validation_result,
    get_output_format,
    get_recommended_structured_config,
    get_schema_type,
    list_output_formats,
    list_schema_types,
    validate_choices_config,
    validate_grammar_config,
    validate_json_output,
    validate_json_schema_config,
    validate_structured_config,
)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_have_values(self) -> None:
        """All formats have string values."""
        for fmt in OutputFormat:
            assert isinstance(fmt.value, str)

    def test_valid_output_formats_frozenset(self) -> None:
        """VALID_OUTPUT_FORMATS is a frozenset of all format values."""
        assert isinstance(VALID_OUTPUT_FORMATS, frozenset)
        assert len(VALID_OUTPUT_FORMATS) == len(OutputFormat)

    def test_json_value(self) -> None:
        """JSON format has correct value."""
        assert OutputFormat.JSON.value == "json"

    def test_json_schema_value(self) -> None:
        """JSON_SCHEMA format has correct value."""
        assert OutputFormat.JSON_SCHEMA.value == "json_schema"

    def test_regex_value(self) -> None:
        """REGEX format has correct value."""
        assert OutputFormat.REGEX.value == "regex"

    def test_grammar_value(self) -> None:
        """GRAMMAR format has correct value."""
        assert OutputFormat.GRAMMAR.value == "grammar"

    def test_choices_value(self) -> None:
        """CHOICES format has correct value."""
        assert OutputFormat.CHOICES.value == "choices"


class TestSchemaType:
    """Tests for SchemaType enum."""

    def test_all_types_have_values(self) -> None:
        """All schema types have string values."""
        for schema_type in SchemaType:
            assert isinstance(schema_type.value, str)

    def test_valid_schema_types_frozenset(self) -> None:
        """VALID_SCHEMA_TYPES is a frozenset of all type values."""
        assert isinstance(VALID_SCHEMA_TYPES, frozenset)
        assert len(VALID_SCHEMA_TYPES) == len(SchemaType)

    def test_object_value(self) -> None:
        """OBJECT type has correct value."""
        assert SchemaType.OBJECT.value == "object"

    def test_array_value(self) -> None:
        """ARRAY type has correct value."""
        assert SchemaType.ARRAY.value == "array"

    def test_string_value(self) -> None:
        """STRING type has correct value."""
        assert SchemaType.STRING.value == "string"

    def test_number_value(self) -> None:
        """NUMBER type has correct value."""
        assert SchemaType.NUMBER.value == "number"

    def test_boolean_value(self) -> None:
        """BOOLEAN type has correct value."""
        assert SchemaType.BOOLEAN.value == "boolean"

    def test_null_value(self) -> None:
        """NULL type has correct value."""
        assert SchemaType.NULL.value == "null"


class TestStructuredConfig:
    """Tests for StructuredConfig dataclass."""

    def test_create_config(self) -> None:
        """Create structured config."""
        config = StructuredConfig(
            output_format=OutputFormat.JSON,
            schema=None,
            strict_mode=True,
            max_retries=3,
        )
        assert config.output_format == OutputFormat.JSON
        assert config.schema is None
        assert config.strict_mode is True
        assert config.max_retries == 3

    def test_config_with_schema(self) -> None:
        """Create config with schema."""
        schema = {"type": "object"}
        config = StructuredConfig(
            output_format=OutputFormat.JSON_SCHEMA,
            schema=schema,
            strict_mode=True,
            max_retries=3,
        )
        assert config.schema == schema

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = StructuredConfig(OutputFormat.JSON, None, True, 3)
        with pytest.raises(AttributeError):
            config.strict_mode = False  # type: ignore[misc]


class TestJSONSchemaConfig:
    """Tests for JSONSchemaConfig dataclass."""

    def test_create_config(self) -> None:
        """Create JSON schema config."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        config = JSONSchemaConfig(
            schema_dict=schema,
            required_fields=("name",),
            additional_properties=False,
        )
        assert config.schema_dict == schema
        assert config.required_fields == ("name",)
        assert config.additional_properties is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = JSONSchemaConfig({"type": "object"}, (), True)
        with pytest.raises(AttributeError):
            config.additional_properties = False  # type: ignore[misc]


class TestGrammarConfig:
    """Tests for GrammarConfig dataclass."""

    def test_create_config(self) -> None:
        """Create grammar config."""
        config = GrammarConfig(
            grammar_string="root ::= 'hello' | 'world'",
            start_symbol="root",
            max_tokens=100,
        )
        assert config.grammar_string == "root ::= 'hello' | 'world'"
        assert config.start_symbol == "root"
        assert config.max_tokens == 100

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = GrammarConfig("root ::= 'test'", "root", 100)
        with pytest.raises(AttributeError):
            config.max_tokens = 200  # type: ignore[misc]


class TestChoicesConfig:
    """Tests for ChoicesConfig dataclass."""

    def test_create_config(self) -> None:
        """Create choices config."""
        config = ChoicesConfig(
            choices=("yes", "no", "maybe"),
            allow_multiple=False,
            separator=", ",
        )
        assert config.choices == ("yes", "no", "maybe")
        assert config.allow_multiple is False
        assert config.separator == ", "

    def test_config_with_multiple(self) -> None:
        """Create config with allow_multiple=True."""
        config = ChoicesConfig(
            choices=("apple", "banana"),
            allow_multiple=True,
            separator="; ",
        )
        assert config.allow_multiple is True
        assert config.separator == "; "

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ChoicesConfig(("yes", "no"), False, ", ")
        with pytest.raises(AttributeError):
            config.allow_multiple = True  # type: ignore[misc]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_valid_result(self) -> None:
        """Create valid result."""
        result = ValidationResult(
            is_valid=True,
            errors=(),
            parsed_output={"name": "test"},
        )
        assert result.is_valid is True
        assert result.errors == ()
        assert result.parsed_output == {"name": "test"}

    def test_create_invalid_result(self) -> None:
        """Create invalid result."""
        result = ValidationResult(
            is_valid=False,
            errors=("Missing field", "Invalid type"),
            parsed_output=None,
        )
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert result.parsed_output is None

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = ValidationResult(True, (), {"a": 1})
        with pytest.raises(AttributeError):
            result.is_valid = False  # type: ignore[misc]


class TestValidateStructuredConfig:
    """Tests for validate_structured_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = StructuredConfig(OutputFormat.JSON, None, True, 3)
        validate_structured_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_structured_config(None)  # type: ignore[arg-type]

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_retries raises ValueError."""
        config = StructuredConfig(OutputFormat.JSON, None, True, -1)
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            validate_structured_config(config)

    def test_json_schema_without_schema_raises(self) -> None:
        """JSON_SCHEMA format without schema raises ValueError."""
        config = StructuredConfig(OutputFormat.JSON_SCHEMA, None, True, 3)
        with pytest.raises(ValueError, match="schema is required for JSON_SCHEMA"):
            validate_structured_config(config)

    def test_json_schema_with_schema_passes(self) -> None:
        """JSON_SCHEMA format with schema passes."""
        config = StructuredConfig(
            OutputFormat.JSON_SCHEMA, {"type": "object"}, True, 3
        )
        validate_structured_config(config)


class TestValidateJSONSchemaConfig:
    """Tests for validate_json_schema_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = JSONSchemaConfig({"type": "object"}, ("name",), False)
        validate_json_schema_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_json_schema_config(None)  # type: ignore[arg-type]

    def test_empty_schema_raises(self) -> None:
        """Empty schema_dict raises ValueError."""
        config = JSONSchemaConfig({}, (), True)
        with pytest.raises(ValueError, match="schema_dict cannot be empty"):
            validate_json_schema_config(config)

    def test_non_string_required_field_raises(self) -> None:
        """Non-string in required_fields raises ValueError."""
        # Create config with a non-string field (type ignore for testing)
        config = JSONSchemaConfig({"type": "object"}, (123,), False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="required_fields must contain strings"):
            validate_json_schema_config(config)


class TestValidateGrammarConfig:
    """Tests for validate_grammar_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = GrammarConfig("root ::= 'test'", "root", 100)
        validate_grammar_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_grammar_config(None)  # type: ignore[arg-type]

    def test_empty_grammar_string_raises(self) -> None:
        """Empty grammar_string raises ValueError."""
        config = GrammarConfig("", "root", 100)
        with pytest.raises(ValueError, match="grammar_string cannot be empty"):
            validate_grammar_config(config)

    def test_empty_start_symbol_raises(self) -> None:
        """Empty start_symbol raises ValueError."""
        config = GrammarConfig("root ::= 'test'", "", 100)
        with pytest.raises(ValueError, match="start_symbol cannot be empty"):
            validate_grammar_config(config)

    def test_zero_max_tokens_raises(self) -> None:
        """Zero max_tokens raises ValueError."""
        config = GrammarConfig("root ::= 'test'", "root", 0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_grammar_config(config)

    def test_negative_max_tokens_raises(self) -> None:
        """Negative max_tokens raises ValueError."""
        config = GrammarConfig("root ::= 'test'", "root", -1)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_grammar_config(config)


class TestValidateChoicesConfig:
    """Tests for validate_choices_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ChoicesConfig(("yes", "no"), False, ", ")
        validate_choices_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_choices_config(None)  # type: ignore[arg-type]

    def test_empty_choices_raises(self) -> None:
        """Empty choices raises ValueError."""
        config = ChoicesConfig((), False, ", ")
        with pytest.raises(ValueError, match="choices cannot be empty"):
            validate_choices_config(config)

    def test_empty_separator_with_multiple_raises(self) -> None:
        """Empty separator with allow_multiple raises ValueError."""
        config = ChoicesConfig(("yes", "no"), True, "")
        with pytest.raises(ValueError, match="separator cannot be empty"):
            validate_choices_config(config)

    def test_empty_separator_without_multiple_passes(self) -> None:
        """Empty separator without allow_multiple passes."""
        config = ChoicesConfig(("yes", "no"), False, "")
        validate_choices_config(config)


class TestCreateStructuredConfig:
    """Tests for create_structured_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_structured_config()
        assert config.output_format == OutputFormat.JSON
        assert config.strict_mode is True
        assert config.max_retries == 3

    def test_with_string_format(self) -> None:
        """Create config with string format."""
        config = create_structured_config(
            output_format="json_schema", schema={"type": "object"}
        )
        assert config.output_format == OutputFormat.JSON_SCHEMA

    def test_with_enum_format(self) -> None:
        """Create config with enum format."""
        config = create_structured_config(output_format=OutputFormat.GRAMMAR)
        assert config.output_format == OutputFormat.GRAMMAR

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_structured_config(
            output_format="json",
            strict_mode=False,
            max_retries=5,
        )
        assert config.strict_mode is False
        assert config.max_retries == 5

    def test_invalid_max_retries_raises(self) -> None:
        """Invalid max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            create_structured_config(max_retries=-1)

    def test_json_schema_without_schema_raises(self) -> None:
        """JSON_SCHEMA without schema raises ValueError."""
        with pytest.raises(ValueError, match="schema is required"):
            create_structured_config(output_format="json_schema")


class TestCreateJSONSchemaConfig:
    """Tests for create_json_schema_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        schema = {"type": "object"}
        config = create_json_schema_config(schema_dict=schema)
        assert config.schema_dict == schema
        assert config.required_fields == ()
        assert config.additional_properties is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        config = create_json_schema_config(
            schema_dict=schema,
            required_fields=("name",),
            additional_properties=True,
        )
        assert config.required_fields == ("name",)
        assert config.additional_properties is True

    def test_empty_schema_raises(self) -> None:
        """Empty schema raises ValueError."""
        with pytest.raises(ValueError, match="schema_dict cannot be empty"):
            create_json_schema_config(schema_dict={})


class TestCreateGrammarConfig:
    """Tests for create_grammar_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_grammar_config(grammar_string="root ::= 'hello'")
        assert config.grammar_string == "root ::= 'hello'"
        assert config.start_symbol == "root"
        assert config.max_tokens == 1024

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_grammar_config(
            grammar_string="expr ::= digit",
            start_symbol="expr",
            max_tokens=200,
        )
        assert config.start_symbol == "expr"
        assert config.max_tokens == 200

    def test_empty_grammar_raises(self) -> None:
        """Empty grammar raises ValueError."""
        with pytest.raises(ValueError, match="grammar_string cannot be empty"):
            create_grammar_config(grammar_string="")


class TestCreateChoicesConfig:
    """Tests for create_choices_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_choices_config(choices=("yes", "no"))
        assert config.choices == ("yes", "no")
        assert config.allow_multiple is False
        assert config.separator == ", "

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_choices_config(
            choices=("a", "b", "c"),
            allow_multiple=True,
            separator="; ",
        )
        assert config.allow_multiple is True
        assert config.separator == "; "

    def test_empty_choices_raises(self) -> None:
        """Empty choices raises ValueError."""
        with pytest.raises(ValueError, match="choices cannot be empty"):
            create_choices_config(choices=())


class TestCreateValidationResult:
    """Tests for create_validation_result function."""

    def test_valid_result(self) -> None:
        """Create valid result."""
        result = create_validation_result(is_valid=True, parsed_output={"key": "value"})
        assert result.is_valid is True
        assert result.errors == ()
        assert result.parsed_output == {"key": "value"}

    def test_invalid_result(self) -> None:
        """Create invalid result."""
        result = create_validation_result(
            is_valid=False,
            errors=("error1", "error2"),
        )
        assert result.is_valid is False
        assert result.errors == ("error1", "error2")
        assert result.parsed_output is None


class TestListOutputFormats:
    """Tests for list_output_formats function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        formats = list_output_formats()
        assert formats == sorted(formats)

    def test_contains_all_formats(self) -> None:
        """Contains all format values."""
        formats = list_output_formats()
        assert "json" in formats
        assert "json_schema" in formats
        assert "regex" in formats
        assert "grammar" in formats
        assert "choices" in formats

    def test_correct_count(self) -> None:
        """Has correct number of formats."""
        formats = list_output_formats()
        assert len(formats) == len(OutputFormat)


class TestListSchemaTypes:
    """Tests for list_schema_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_schema_types()
        assert types == sorted(types)

    def test_contains_all_types(self) -> None:
        """Contains all type values."""
        types = list_schema_types()
        assert "object" in types
        assert "array" in types
        assert "string" in types
        assert "number" in types
        assert "boolean" in types
        assert "null" in types

    def test_correct_count(self) -> None:
        """Has correct number of types."""
        types = list_schema_types()
        assert len(types) == len(SchemaType)


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("json", OutputFormat.JSON),
            ("json_schema", OutputFormat.JSON_SCHEMA),
            ("regex", OutputFormat.REGEX),
            ("grammar", OutputFormat.GRAMMAR),
            ("choices", OutputFormat.CHOICES),
        ],
    )
    def test_get_valid_format(self, name: str, expected: OutputFormat) -> None:
        """Get valid formats."""
        assert get_output_format(name) == expected

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="invalid output format"):
            get_output_format("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid output format"):
            get_output_format("")


class TestGetSchemaType:
    """Tests for get_schema_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("object", SchemaType.OBJECT),
            ("array", SchemaType.ARRAY),
            ("string", SchemaType.STRING),
            ("number", SchemaType.NUMBER),
            ("boolean", SchemaType.BOOLEAN),
            ("null", SchemaType.NULL),
        ],
    )
    def test_get_valid_type(self, name: str, expected: SchemaType) -> None:
        """Get valid types."""
        assert get_schema_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid schema type"):
            get_schema_type("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid schema type"):
            get_schema_type("")


class TestValidateJsonOutput:
    """Tests for validate_json_output function."""

    def test_valid_json(self) -> None:
        """Validate valid JSON."""
        result = validate_json_output('{"name": "test"}')
        assert result.is_valid is True
        assert result.parsed_output == {"name": "test"}

    def test_invalid_json(self) -> None:
        """Validate invalid JSON."""
        result = validate_json_output("not json")
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Invalid JSON" in result.errors[0]

    def test_empty_string(self) -> None:
        """Empty string is invalid."""
        result = validate_json_output("")
        assert result.is_valid is False
        assert "Empty output" in result.errors[0]

    def test_whitespace_only(self) -> None:
        """Whitespace-only string is invalid."""
        result = validate_json_output("   \n\t  ")
        assert result.is_valid is False
        assert "Empty output" in result.errors[0]

    def test_with_schema_valid(self) -> None:
        """Validate JSON with schema - valid."""
        schema = create_json_schema_config(
            schema_dict={"type": "object", "properties": {"name": {"type": "string"}}},
            required_fields=("name",),
        )
        result = validate_json_output('{"name": "test"}', schema)
        assert result.is_valid is True

    def test_with_schema_missing_required(self) -> None:
        """Validate JSON with schema - missing required field."""
        schema = create_json_schema_config(
            schema_dict={"type": "object", "properties": {"name": {"type": "string"}}},
            required_fields=("name",),
        )
        result = validate_json_output('{"other": "value"}', schema)
        assert result.is_valid is False
        assert "Missing required field: name" in result.errors

    def test_with_schema_additional_properties_not_allowed(self) -> None:
        """Validate JSON with schema - additional properties not allowed."""
        schema = create_json_schema_config(
            schema_dict={"type": "object", "properties": {"name": {"type": "string"}}},
            required_fields=(),
            additional_properties=False,
        )
        result = validate_json_output('{"name": "test", "extra": "field"}', schema)
        assert result.is_valid is False
        assert "Additional property not allowed: extra" in result.errors

    def test_with_schema_additional_properties_allowed(self) -> None:
        """Validate JSON with schema - additional properties allowed."""
        schema = create_json_schema_config(
            schema_dict={"type": "object", "properties": {"name": {"type": "string"}}},
            required_fields=(),
            additional_properties=True,
        )
        result = validate_json_output('{"name": "test", "extra": "field"}', schema)
        assert result.is_valid is True

    def test_json_array(self) -> None:
        """Validate JSON array."""
        result = validate_json_output('[1, 2, 3]')
        assert result.is_valid is True
        assert result.parsed_output == [1, 2, 3]

    def test_json_number(self) -> None:
        """Validate JSON number."""
        result = validate_json_output("42")
        assert result.is_valid is True
        assert result.parsed_output == 42

    def test_json_string(self) -> None:
        """Validate JSON string."""
        result = validate_json_output('"hello"')
        assert result.is_valid is True
        assert result.parsed_output == "hello"

    def test_json_boolean(self) -> None:
        """Validate JSON boolean."""
        result = validate_json_output("true")
        assert result.is_valid is True
        assert result.parsed_output is True

    def test_json_null(self) -> None:
        """Validate JSON null."""
        result = validate_json_output("null")
        assert result.is_valid is True
        assert result.parsed_output is None

    def test_non_dict_with_schema(self) -> None:
        """Validate non-dict JSON with schema config (skips dict validation)."""
        schema = create_json_schema_config(
            schema_dict={"type": "array", "items": {"type": "number"}},
            required_fields=(),
            additional_properties=False,
        )
        # Array with schema - dict validation is skipped
        result = validate_json_output("[1, 2, 3]", schema)
        assert result.is_valid is True
        assert result.parsed_output == [1, 2, 3]


class TestEstimateStructuredTokens:
    """Tests for estimate_structured_tokens function."""

    def test_default_estimate(self) -> None:
        """Default estimate returns base_tokens."""
        assert estimate_structured_tokens() == 50

    def test_custom_base_tokens(self) -> None:
        """Custom base_tokens returns correctly."""
        assert estimate_structured_tokens(base_tokens=100) == 100

    def test_grammar_config_uses_max_tokens(self) -> None:
        """Grammar config uses its max_tokens."""
        grammar = create_grammar_config(
            grammar_string="root ::= 'test'", max_tokens=200
        )
        assert estimate_structured_tokens(grammar_config=grammar) == 200

    def test_choices_config_single(self) -> None:
        """Choices config (single) returns small estimate."""
        choices = create_choices_config(choices=("yes", "no"))
        result = estimate_structured_tokens(choices_config=choices)
        assert result >= 5

    def test_choices_config_multiple(self) -> None:
        """Choices config (multiple) returns larger estimate."""
        choices = create_choices_config(
            choices=("apple", "banana", "cherry"),
            allow_multiple=True,
        )
        result = estimate_structured_tokens(choices_config=choices)
        # Should be larger than single choice
        assert result > 10

    def test_schema_config_adds_tokens(self) -> None:
        """Schema config adds tokens for properties."""
        schema = create_json_schema_config(
            schema_dict={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                },
            },
            required_fields=("name", "age"),
        )
        result = estimate_structured_tokens(schema_config=schema)
        assert result > 50  # More than base

    def test_zero_base_tokens_raises(self) -> None:
        """Zero base_tokens raises ValueError."""
        with pytest.raises(ValueError, match="base_tokens must be positive"):
            estimate_structured_tokens(base_tokens=0)

    def test_negative_base_tokens_raises(self) -> None:
        """Negative base_tokens raises ValueError."""
        with pytest.raises(ValueError, match="base_tokens must be positive"):
            estimate_structured_tokens(base_tokens=-1)


class TestGetRecommendedStructuredConfig:
    """Tests for get_recommended_structured_config function."""

    def test_extraction_task(self) -> None:
        """Get config for extraction task."""
        config = get_recommended_structured_config("extraction")
        assert config.output_format == OutputFormat.JSON_SCHEMA
        assert config.strict_mode is True

    def test_classification_task(self) -> None:
        """Get config for classification task."""
        config = get_recommended_structured_config("classification")
        assert config.output_format == OutputFormat.CHOICES

    def test_qa_task(self) -> None:
        """Get config for QA task."""
        config = get_recommended_structured_config("qa")
        assert config.output_format == OutputFormat.JSON
        assert config.strict_mode is False

    def test_generation_task(self) -> None:
        """Get config for generation task."""
        config = get_recommended_structured_config("generation")
        assert config.output_format == OutputFormat.GRAMMAR

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_structured_config("invalid")

    @pytest.mark.parametrize(
        "task", ["extraction", "classification", "qa", "generation"]
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All valid tasks return valid configs."""
        config = get_recommended_structured_config(task)
        assert isinstance(config, StructuredConfig)
        assert config.max_retries >= 0


class TestFormatValidationResult:
    """Tests for format_validation_result function."""

    def test_format_valid_result(self) -> None:
        """Format valid result."""
        result = create_validation_result(
            is_valid=True,
            parsed_output={"name": "test"},
        )
        formatted = format_validation_result(result)
        assert "Valid: True" in formatted
        assert "Parsed:" in formatted

    def test_format_invalid_result(self) -> None:
        """Format invalid result."""
        result = create_validation_result(
            is_valid=False,
            errors=("Missing field",),
        )
        formatted = format_validation_result(result)
        assert "Valid: False" in formatted
        assert "Errors:" in formatted
        assert "Missing field" in formatted

    def test_format_multiple_errors(self) -> None:
        """Format result with multiple errors."""
        result = create_validation_result(
            is_valid=False,
            errors=("Error 1", "Error 2", "Error 3"),
        )
        formatted = format_validation_result(result)
        assert "Error 1" in formatted
        assert "Error 2" in formatted
        assert "Error 3" in formatted

    def test_format_none_raises(self) -> None:
        """None result raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            format_validation_result(None)  # type: ignore[arg-type]

    def test_format_no_parsed_output(self) -> None:
        """Format result without parsed output."""
        result = create_validation_result(
            is_valid=False,
            errors=("Invalid JSON",),
        )
        formatted = format_validation_result(result)
        assert "Parsed:" not in formatted

    def test_format_valid_with_no_errors(self) -> None:
        """Format valid result has no errors section."""
        result = create_validation_result(
            is_valid=True,
            parsed_output=42,
        )
        formatted = format_validation_result(result)
        assert "Errors:" not in formatted


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_schema_with_empty_properties(self) -> None:
        """Schema with no properties defined."""
        schema = create_json_schema_config(
            schema_dict={"type": "object"},
            required_fields=(),
            additional_properties=True,
        )
        result = validate_json_output('{"any": "value"}', schema)
        assert result.is_valid is True

    def test_choices_with_single_choice(self) -> None:
        """Choices with only one option."""
        config = create_choices_config(choices=("only_option",))
        assert len(config.choices) == 1
        validate_choices_config(config)

    def test_grammar_with_whitespace_start_symbol(self) -> None:
        """Grammar with whitespace in start_symbol."""
        config = create_grammar_config(
            grammar_string="start ::= 'test'",
            start_symbol="start",
        )
        assert config.start_symbol == "start"

    def test_max_retries_zero(self) -> None:
        """Max retries of zero is valid."""
        config = create_structured_config(max_retries=0)
        assert config.max_retries == 0

    def test_json_with_nested_objects(self) -> None:
        """Validate nested JSON objects."""
        json_str = '{"outer": {"inner": {"value": 123}}}'
        result = validate_json_output(json_str)
        assert result.is_valid is True
        assert result.parsed_output["outer"]["inner"]["value"] == 123

    def test_json_with_unicode(self) -> None:
        """Validate JSON with unicode characters."""
        json_str = '{"message": "Hello \\u4e16\\u754c"}'
        result = validate_json_output(json_str)
        assert result.is_valid is True

    def test_json_with_special_characters(self) -> None:
        """Validate JSON with special characters."""
        json_str = '{"text": "line1\\nline2\\ttab"}'
        result = validate_json_output(json_str)
        assert result.is_valid is True

    def test_large_schema_tokens_estimation(self) -> None:
        """Estimate tokens for large schema."""
        schema = create_json_schema_config(
            schema_dict={
                "type": "object",
                "properties": {field: {"type": "string"} for field in "abcdefghij"},
            },
            required_fields=tuple("abcdefghij"),
        )
        tokens = estimate_structured_tokens(schema_config=schema)
        # 50 base + 10 properties * 15 + 10 required * 10
        expected = 50 + 10 * 15 + 10 * 10
        assert tokens == expected
