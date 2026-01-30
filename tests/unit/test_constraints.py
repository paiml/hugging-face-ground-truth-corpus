"""Tests for generation.constraints module."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.generation.constraints import (
    VALID_CONSTRAINT_TYPES,
    VALID_ENFORCEMENT_MODES,
    VALID_GRAMMAR_FORMATS,
    CompiledConstraint,
    ConstraintConfig,
    ConstraintStats,
    ConstraintType,
    EnforcementMode,
    GrammarConstraint,
    GrammarFormat,
    JSONSchemaConstraint,
    RegexConstraint,
    combine_constraints,
    compile_constraint,
    create_constraint_config,
    create_constraint_stats,
    create_grammar_constraint,
    create_json_schema_constraint,
    create_regex_constraint,
    estimate_constraint_overhead,
    format_constraint_stats,
    get_constraint_type,
    get_enforcement_mode,
    get_grammar_format,
    get_recommended_constraint_config,
    list_constraint_types,
    list_enforcement_modes,
    list_grammar_formats,
    validate_constraint_config,
    validate_grammar_constraint,
    validate_json_schema_constraint,
    validate_output,
    validate_regex_constraint,
)


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for ct in ConstraintType:
            assert isinstance(ct.value, str)

    def test_valid_constraint_types_frozenset(self) -> None:
        """VALID_CONSTRAINT_TYPES is a frozenset of all type values."""
        assert isinstance(VALID_CONSTRAINT_TYPES, frozenset)
        assert len(VALID_CONSTRAINT_TYPES) == len(ConstraintType)

    def test_json_schema_value(self) -> None:
        """JSON_SCHEMA type has correct value."""
        assert ConstraintType.JSON_SCHEMA.value == "json_schema"

    def test_regex_value(self) -> None:
        """REGEX type has correct value."""
        assert ConstraintType.REGEX.value == "regex"

    def test_grammar_value(self) -> None:
        """GRAMMAR type has correct value."""
        assert ConstraintType.GRAMMAR.value == "grammar"

    def test_choices_value(self) -> None:
        """CHOICES type has correct value."""
        assert ConstraintType.CHOICES.value == "choices"

    def test_length_value(self) -> None:
        """LENGTH type has correct value."""
        assert ConstraintType.LENGTH.value == "length"


class TestEnforcementMode:
    """Tests for EnforcementMode enum."""

    def test_all_modes_have_values(self) -> None:
        """All modes have string values."""
        for em in EnforcementMode:
            assert isinstance(em.value, str)

    def test_valid_enforcement_modes_frozenset(self) -> None:
        """VALID_ENFORCEMENT_MODES is a frozenset of all mode values."""
        assert isinstance(VALID_ENFORCEMENT_MODES, frozenset)
        assert len(VALID_ENFORCEMENT_MODES) == len(EnforcementMode)

    def test_strict_value(self) -> None:
        """STRICT mode has correct value."""
        assert EnforcementMode.STRICT.value == "strict"

    def test_soft_value(self) -> None:
        """SOFT mode has correct value."""
        assert EnforcementMode.SOFT.value == "soft"

    def test_sample_and_filter_value(self) -> None:
        """SAMPLE_AND_FILTER mode has correct value."""
        assert EnforcementMode.SAMPLE_AND_FILTER.value == "sample_and_filter"


class TestGrammarFormat:
    """Tests for GrammarFormat enum."""

    def test_all_formats_have_values(self) -> None:
        """All formats have string values."""
        for gf in GrammarFormat:
            assert isinstance(gf.value, str)

    def test_valid_grammar_formats_frozenset(self) -> None:
        """VALID_GRAMMAR_FORMATS is a frozenset of all format values."""
        assert isinstance(VALID_GRAMMAR_FORMATS, frozenset)
        assert len(VALID_GRAMMAR_FORMATS) == len(GrammarFormat)

    def test_ebnf_value(self) -> None:
        """EBNF format has correct value."""
        assert GrammarFormat.EBNF.value == "ebnf"

    def test_lark_value(self) -> None:
        """LARK format has correct value."""
        assert GrammarFormat.LARK.value == "lark"

    def test_antlr_value(self) -> None:
        """ANTLR format has correct value."""
        assert GrammarFormat.ANTLR.value == "antlr"

    def test_regex_value(self) -> None:
        """REGEX format has correct value."""
        assert GrammarFormat.REGEX.value == "regex"


class TestJSONSchemaConstraint:
    """Tests for JSONSchemaConstraint dataclass."""

    def test_create_constraint(self) -> None:
        """Create JSON schema constraint."""
        constraint = JSONSchemaConstraint(
            schema={"type": "object"},
            strict=True,
            allow_extra=False,
        )
        assert constraint.schema == {"type": "object"}
        assert constraint.strict is True
        assert constraint.allow_extra is False

    def test_constraint_is_frozen(self) -> None:
        """Constraint is immutable."""
        constraint = JSONSchemaConstraint({"type": "object"}, True, False)
        with pytest.raises(AttributeError):
            constraint.strict = False  # type: ignore[misc]

    def test_complex_schema(self) -> None:
        """Constraint with complex schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name"],
        }
        constraint = JSONSchemaConstraint(schema, True, False)
        assert "properties" in constraint.schema


class TestRegexConstraint:
    """Tests for RegexConstraint dataclass."""

    def test_create_constraint(self) -> None:
        """Create regex constraint."""
        constraint = RegexConstraint(
            pattern=r"[A-Z][a-z]+",
            full_match=True,
            max_length=100,
        )
        assert constraint.pattern == r"[A-Z][a-z]+"
        assert constraint.full_match is True
        assert constraint.max_length == 100

    def test_constraint_is_frozen(self) -> None:
        """Constraint is immutable."""
        constraint = RegexConstraint(r"[a-z]+", True, 100)
        with pytest.raises(AttributeError):
            constraint.pattern = r"[A-Z]+"  # type: ignore[misc]

    def test_partial_match(self) -> None:
        """Constraint with partial match."""
        constraint = RegexConstraint(r"\d+", False, 50)
        assert constraint.full_match is False


class TestGrammarConstraint:
    """Tests for GrammarConstraint dataclass."""

    def test_create_constraint(self) -> None:
        """Create grammar constraint."""
        constraint = GrammarConstraint(
            grammar="root ::= 'hello' | 'world'",
            format=GrammarFormat.EBNF,
            start_symbol="root",
        )
        assert constraint.grammar == "root ::= 'hello' | 'world'"
        assert constraint.format == GrammarFormat.EBNF
        assert constraint.start_symbol == "root"

    def test_constraint_is_frozen(self) -> None:
        """Constraint is immutable."""
        constraint = GrammarConstraint("root ::= 'x'", GrammarFormat.EBNF, "root")
        with pytest.raises(AttributeError):
            constraint.grammar = "new"  # type: ignore[misc]

    def test_lark_format(self) -> None:
        """Constraint with Lark format."""
        constraint = GrammarConstraint("start: 'hello'", GrammarFormat.LARK, "start")
        assert constraint.format == GrammarFormat.LARK


class TestConstraintConfig:
    """Tests for ConstraintConfig dataclass."""

    def test_create_config(self) -> None:
        """Create constraint config."""
        json_c = JSONSchemaConstraint({"type": "object"}, True, False)
        config = ConstraintConfig(
            constraint_type=ConstraintType.JSON_SCHEMA,
            json_constraint=json_c,
            regex_constraint=None,
            grammar_constraint=None,
            enforcement=EnforcementMode.STRICT,
        )
        assert config.constraint_type == ConstraintType.JSON_SCHEMA
        assert config.enforcement == EnforcementMode.STRICT
        assert config.json_constraint is not None
        assert config.regex_constraint is None

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        json_c = JSONSchemaConstraint({"type": "object"}, True, False)
        config = ConstraintConfig(
            ConstraintType.JSON_SCHEMA, json_c, None, None, EnforcementMode.STRICT
        )
        with pytest.raises(AttributeError):
            config.enforcement = EnforcementMode.SOFT  # type: ignore[misc]

    def test_regex_config(self) -> None:
        """Config with regex constraint."""
        regex_c = RegexConstraint(r"[a-z]+", True, 100)
        config = ConstraintConfig(
            ConstraintType.REGEX, None, regex_c, None, EnforcementMode.SOFT
        )
        assert config.constraint_type == ConstraintType.REGEX
        assert config.regex_constraint is not None


class TestConstraintStats:
    """Tests for ConstraintStats dataclass."""

    def test_create_stats(self) -> None:
        """Create constraint stats."""
        stats = ConstraintStats(
            valid_outputs=100,
            invalid_outputs=5,
            retries=10,
            avg_generation_time_ms=150.5,
        )
        assert stats.valid_outputs == 100
        assert stats.invalid_outputs == 5
        assert stats.retries == 10
        assert stats.avg_generation_time_ms == 150.5

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = ConstraintStats(100, 5, 10, 150.5)
        with pytest.raises(AttributeError):
            stats.valid_outputs = 200  # type: ignore[misc]


class TestValidateJSONSchemaConstraint:
    """Tests for validate_json_schema_constraint function."""

    def test_valid_constraint(self) -> None:
        """Valid constraint passes."""
        constraint = JSONSchemaConstraint({"type": "object"}, True, False)
        validate_json_schema_constraint(constraint)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="constraint cannot be None"):
            validate_json_schema_constraint(None)  # type: ignore[arg-type]

    def test_empty_schema_raises(self) -> None:
        """Empty schema raises ValueError."""
        constraint = JSONSchemaConstraint({}, True, False)
        with pytest.raises(ValueError, match="schema cannot be empty"):
            validate_json_schema_constraint(constraint)


class TestValidateRegexConstraint:
    """Tests for validate_regex_constraint function."""

    def test_valid_constraint(self) -> None:
        """Valid constraint passes."""
        constraint = RegexConstraint(r"[A-Z]+", True, 100)
        validate_regex_constraint(constraint)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="constraint cannot be None"):
            validate_regex_constraint(None)  # type: ignore[arg-type]

    def test_empty_pattern_raises(self) -> None:
        """Empty pattern raises ValueError."""
        constraint = RegexConstraint("", True, 100)
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            validate_regex_constraint(constraint)

    def test_invalid_pattern_raises(self) -> None:
        """Invalid regex pattern raises ValueError."""
        constraint = RegexConstraint(r"[invalid", True, 100)
        with pytest.raises(ValueError, match="invalid regex pattern"):
            validate_regex_constraint(constraint)

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        constraint = RegexConstraint(r"[A-Z]+", True, 0)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_regex_constraint(constraint)

    def test_negative_max_length_raises(self) -> None:
        """Negative max_length raises ValueError."""
        constraint = RegexConstraint(r"[A-Z]+", True, -1)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_regex_constraint(constraint)


class TestValidateGrammarConstraint:
    """Tests for validate_grammar_constraint function."""

    def test_valid_constraint(self) -> None:
        """Valid constraint passes."""
        constraint = GrammarConstraint("root ::= 'x'", GrammarFormat.EBNF, "root")
        validate_grammar_constraint(constraint)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="constraint cannot be None"):
            validate_grammar_constraint(None)  # type: ignore[arg-type]

    def test_empty_grammar_raises(self) -> None:
        """Empty grammar raises ValueError."""
        constraint = GrammarConstraint("", GrammarFormat.EBNF, "root")
        with pytest.raises(ValueError, match="grammar cannot be empty"):
            validate_grammar_constraint(constraint)

    def test_empty_start_symbol_raises(self) -> None:
        """Empty start_symbol raises ValueError."""
        constraint = GrammarConstraint("root ::= 'x'", GrammarFormat.EBNF, "")
        with pytest.raises(ValueError, match="start_symbol cannot be empty"):
            validate_grammar_constraint(constraint)


class TestValidateConstraintConfig:
    """Tests for validate_constraint_config function."""

    def test_valid_json_schema_config(self) -> None:
        """Valid JSON schema config passes."""
        json_c = JSONSchemaConstraint({"type": "object"}, True, False)
        config = ConstraintConfig(
            ConstraintType.JSON_SCHEMA, json_c, None, None, EnforcementMode.STRICT
        )
        validate_constraint_config(config)

    def test_valid_regex_config(self) -> None:
        """Valid regex config passes."""
        regex_c = RegexConstraint(r"[a-z]+", True, 100)
        config = ConstraintConfig(
            ConstraintType.REGEX, None, regex_c, None, EnforcementMode.STRICT
        )
        validate_constraint_config(config)

    def test_valid_grammar_config(self) -> None:
        """Valid grammar config passes."""
        grammar_c = GrammarConstraint("root ::= 'x'", GrammarFormat.EBNF, "root")
        config = ConstraintConfig(
            ConstraintType.GRAMMAR, None, None, grammar_c, EnforcementMode.STRICT
        )
        validate_constraint_config(config)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_constraint_config(None)  # type: ignore[arg-type]

    def test_missing_json_constraint_raises(self) -> None:
        """Missing JSON constraint raises ValueError."""
        config = ConstraintConfig(
            ConstraintType.JSON_SCHEMA, None, None, None, EnforcementMode.STRICT
        )
        with pytest.raises(ValueError, match="json_constraint is required"):
            validate_constraint_config(config)

    def test_missing_regex_constraint_raises(self) -> None:
        """Missing regex constraint raises ValueError."""
        config = ConstraintConfig(
            ConstraintType.REGEX, None, None, None, EnforcementMode.STRICT
        )
        with pytest.raises(ValueError, match="regex_constraint is required"):
            validate_constraint_config(config)

    def test_missing_grammar_constraint_raises(self) -> None:
        """Missing grammar constraint raises ValueError."""
        config = ConstraintConfig(
            ConstraintType.GRAMMAR, None, None, None, EnforcementMode.STRICT
        )
        with pytest.raises(ValueError, match="grammar_constraint is required"):
            validate_constraint_config(config)

    def test_invalid_json_constraint_raises(self) -> None:
        """Invalid JSON constraint raises ValueError."""
        json_c = JSONSchemaConstraint({}, True, False)
        config = ConstraintConfig(
            ConstraintType.JSON_SCHEMA, json_c, None, None, EnforcementMode.STRICT
        )
        with pytest.raises(ValueError, match="schema cannot be empty"):
            validate_constraint_config(config)


class TestCreateJSONSchemaConstraint:
    """Tests for create_json_schema_constraint function."""

    def test_basic_creation(self) -> None:
        """Create basic constraint."""
        constraint = create_json_schema_constraint(schema={"type": "object"})
        assert constraint.schema == {"type": "object"}
        assert constraint.strict is True
        assert constraint.allow_extra is False

    def test_with_options(self) -> None:
        """Create with options."""
        constraint = create_json_schema_constraint(
            schema={"type": "array"},
            strict=False,
            allow_extra=True,
        )
        assert constraint.strict is False
        assert constraint.allow_extra is True

    def test_empty_schema_raises(self) -> None:
        """Empty schema raises ValueError."""
        with pytest.raises(ValueError, match="schema cannot be empty"):
            create_json_schema_constraint(schema={})


class TestCreateRegexConstraint:
    """Tests for create_regex_constraint function."""

    def test_basic_creation(self) -> None:
        """Create basic constraint."""
        constraint = create_regex_constraint(pattern=r"[A-Z]+")
        assert constraint.pattern == r"[A-Z]+"
        assert constraint.full_match is True
        assert constraint.max_length == 1024

    def test_with_options(self) -> None:
        """Create with options."""
        constraint = create_regex_constraint(
            pattern=r"\d+",
            full_match=False,
            max_length=50,
        )
        assert constraint.full_match is False
        assert constraint.max_length == 50

    def test_empty_pattern_raises(self) -> None:
        """Empty pattern raises ValueError."""
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            create_regex_constraint(pattern="")

    def test_invalid_pattern_raises(self) -> None:
        """Invalid pattern raises ValueError."""
        with pytest.raises(ValueError, match="invalid regex pattern"):
            create_regex_constraint(pattern=r"[invalid")


class TestCreateGrammarConstraint:
    """Tests for create_grammar_constraint function."""

    def test_basic_creation(self) -> None:
        """Create basic constraint."""
        constraint = create_grammar_constraint(grammar="root ::= 'hello'")
        assert constraint.grammar == "root ::= 'hello'"
        assert constraint.format == GrammarFormat.EBNF
        assert constraint.start_symbol == "root"

    def test_with_string_format(self) -> None:
        """Create with string format."""
        constraint = create_grammar_constraint(grammar="start: 'x'", format="lark")
        assert constraint.format == GrammarFormat.LARK

    def test_with_enum_format(self) -> None:
        """Create with enum format."""
        constraint = create_grammar_constraint(
            grammar="grammar G;", format=GrammarFormat.ANTLR
        )
        assert constraint.format == GrammarFormat.ANTLR

    def test_with_custom_start_symbol(self) -> None:
        """Create with custom start symbol."""
        constraint = create_grammar_constraint(
            grammar="expr ::= term", start_symbol="expr"
        )
        assert constraint.start_symbol == "expr"

    def test_empty_grammar_raises(self) -> None:
        """Empty grammar raises ValueError."""
        with pytest.raises(ValueError, match="grammar cannot be empty"):
            create_grammar_constraint(grammar="")


class TestCreateConstraintConfig:
    """Tests for create_constraint_config function."""

    def test_json_schema_config(self) -> None:
        """Create JSON schema config."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema",
            json_constraint=json_c,
        )
        assert config.constraint_type == ConstraintType.JSON_SCHEMA
        assert config.enforcement == EnforcementMode.STRICT

    def test_regex_config(self) -> None:
        """Create regex config."""
        regex_c = create_regex_constraint(pattern=r"[a-z]+")
        config = create_constraint_config(
            constraint_type="regex",
            regex_constraint=regex_c,
        )
        assert config.constraint_type == ConstraintType.REGEX

    def test_grammar_config(self) -> None:
        """Create grammar config."""
        grammar_c = create_grammar_constraint(grammar="root ::= 'x'")
        config = create_constraint_config(
            constraint_type="grammar",
            grammar_constraint=grammar_c,
        )
        assert config.constraint_type == ConstraintType.GRAMMAR

    def test_with_enum_type(self) -> None:
        """Create with enum type."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type=ConstraintType.JSON_SCHEMA,
            json_constraint=json_c,
        )
        assert config.constraint_type == ConstraintType.JSON_SCHEMA

    def test_with_string_enforcement(self) -> None:
        """Create with string enforcement mode."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema",
            json_constraint=json_c,
            enforcement="soft",
        )
        assert config.enforcement == EnforcementMode.SOFT

    def test_missing_constraint_raises(self) -> None:
        """Missing required constraint raises ValueError."""
        with pytest.raises(ValueError, match="json_constraint is required"):
            create_constraint_config(constraint_type="json_schema")


class TestCreateConstraintStats:
    """Tests for create_constraint_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_constraint_stats()
        assert stats.valid_outputs == 0
        assert stats.invalid_outputs == 0
        assert stats.retries == 0
        assert stats.avg_generation_time_ms == 0.0

    def test_with_values(self) -> None:
        """Create with values."""
        stats = create_constraint_stats(
            valid_outputs=100,
            invalid_outputs=5,
            retries=10,
            avg_generation_time_ms=150.5,
        )
        assert stats.valid_outputs == 100
        assert stats.invalid_outputs == 5

    def test_negative_valid_outputs_raises(self) -> None:
        """Negative valid_outputs raises ValueError."""
        with pytest.raises(ValueError, match="valid_outputs must be non-negative"):
            create_constraint_stats(valid_outputs=-1)

    def test_negative_invalid_outputs_raises(self) -> None:
        """Negative invalid_outputs raises ValueError."""
        with pytest.raises(ValueError, match="invalid_outputs must be non-negative"):
            create_constraint_stats(invalid_outputs=-1)

    def test_negative_retries_raises(self) -> None:
        """Negative retries raises ValueError."""
        with pytest.raises(ValueError, match="retries must be non-negative"):
            create_constraint_stats(retries=-1)

    def test_negative_avg_time_raises(self) -> None:
        """Negative avg_generation_time_ms raises ValueError."""
        with pytest.raises(
            ValueError, match="avg_generation_time_ms must be non-negative"
        ):
            create_constraint_stats(avg_generation_time_ms=-1.0)


class TestListConstraintTypes:
    """Tests for list_constraint_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_constraint_types()
        assert types == sorted(types)

    def test_contains_all_types(self) -> None:
        """Contains all types."""
        types = list_constraint_types()
        assert "json_schema" in types
        assert "regex" in types
        assert "grammar" in types
        assert "choices" in types
        assert "length" in types

    def test_correct_count(self) -> None:
        """Has correct count."""
        types = list_constraint_types()
        assert len(types) == len(ConstraintType)


class TestListEnforcementModes:
    """Tests for list_enforcement_modes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        modes = list_enforcement_modes()
        assert modes == sorted(modes)

    def test_contains_all_modes(self) -> None:
        """Contains all modes."""
        modes = list_enforcement_modes()
        assert "strict" in modes
        assert "soft" in modes
        assert "sample_and_filter" in modes

    def test_correct_count(self) -> None:
        """Has correct count."""
        modes = list_enforcement_modes()
        assert len(modes) == len(EnforcementMode)


class TestListGrammarFormats:
    """Tests for list_grammar_formats function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        formats = list_grammar_formats()
        assert formats == sorted(formats)

    def test_contains_all_formats(self) -> None:
        """Contains all formats."""
        formats = list_grammar_formats()
        assert "ebnf" in formats
        assert "lark" in formats
        assert "antlr" in formats
        assert "regex" in formats

    def test_correct_count(self) -> None:
        """Has correct count."""
        formats = list_grammar_formats()
        assert len(formats) == len(GrammarFormat)


class TestGetConstraintType:
    """Tests for get_constraint_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("json_schema", ConstraintType.JSON_SCHEMA),
            ("regex", ConstraintType.REGEX),
            ("grammar", ConstraintType.GRAMMAR),
            ("choices", ConstraintType.CHOICES),
            ("length", ConstraintType.LENGTH),
        ],
    )
    def test_get_valid_type(self, name: str, expected: ConstraintType) -> None:
        """Get valid types."""
        assert get_constraint_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid constraint type"):
            get_constraint_type("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid constraint type"):
            get_constraint_type("")


class TestGetEnforcementMode:
    """Tests for get_enforcement_mode function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("strict", EnforcementMode.STRICT),
            ("soft", EnforcementMode.SOFT),
            ("sample_and_filter", EnforcementMode.SAMPLE_AND_FILTER),
        ],
    )
    def test_get_valid_mode(self, name: str, expected: EnforcementMode) -> None:
        """Get valid modes."""
        assert get_enforcement_mode(name) == expected

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid enforcement mode"):
            get_enforcement_mode("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid enforcement mode"):
            get_enforcement_mode("")


class TestGetGrammarFormat:
    """Tests for get_grammar_format function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ebnf", GrammarFormat.EBNF),
            ("lark", GrammarFormat.LARK),
            ("antlr", GrammarFormat.ANTLR),
            ("regex", GrammarFormat.REGEX),
        ],
    )
    def test_get_valid_format(self, name: str, expected: GrammarFormat) -> None:
        """Get valid formats."""
        assert get_grammar_format(name) == expected

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="invalid grammar format"):
            get_grammar_format("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid grammar format"):
            get_grammar_format("")


class TestValidateOutput:
    """Tests for validate_output function."""

    def test_valid_json_output(self) -> None:
        """Valid JSON output passes."""
        json_c = create_json_schema_constraint(
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            allow_extra=True,
        )
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, errors = validate_output('{"name": "test"}', config)
        assert is_valid is True
        assert errors == ()

    def test_invalid_json_output(self) -> None:
        """Invalid JSON output fails."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, errors = validate_output("not json", config)
        assert is_valid is False
        assert len(errors) > 0

    def test_valid_regex_output(self) -> None:
        """Valid regex output passes."""
        regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        is_valid, _errors = validate_output("Hello", config)
        assert is_valid is True

    def test_invalid_regex_output(self) -> None:
        """Invalid regex output fails."""
        regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        is_valid, _errors = validate_output("hello", config)
        assert is_valid is False

    def test_regex_partial_match(self) -> None:
        """Regex partial match works."""
        regex_c = create_regex_constraint(pattern=r"[A-Z]+", full_match=False)
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        is_valid, _errors = validate_output("hello ABC world", config)
        assert is_valid is True

    def test_regex_length_exceeded(self) -> None:
        """Regex max length exceeded fails."""
        regex_c = create_regex_constraint(pattern=r".*", max_length=5)
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        is_valid, errors = validate_output("hello world", config)
        assert is_valid is False
        assert any("exceeds max" in e for e in errors)

    def test_grammar_empty_output_fails(self) -> None:
        """Empty output for grammar constraint fails."""
        grammar_c = create_grammar_constraint(grammar="root ::= 'x'")
        config = create_constraint_config(
            constraint_type="grammar", grammar_constraint=grammar_c
        )
        is_valid, _errors = validate_output("   ", config)
        assert is_valid is False

    def test_grammar_nonempty_output_passes(self) -> None:
        """Non-empty output for grammar constraint passes."""
        grammar_c = create_grammar_constraint(grammar="root ::= 'x'")
        config = create_constraint_config(
            constraint_type="grammar", grammar_constraint=grammar_c
        )
        is_valid, _errors = validate_output("hello", config)
        assert is_valid is True

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_output("test", None)  # type: ignore[arg-type]

    def test_choices_type_passes(self) -> None:
        """CHOICES type passes by default."""
        config = ConstraintConfig(
            ConstraintType.CHOICES, None, None, None, EnforcementMode.STRICT
        )
        is_valid, _errors = validate_output("test", config)
        assert is_valid is True

    def test_length_type_passes(self) -> None:
        """LENGTH type passes by default."""
        config = ConstraintConfig(
            ConstraintType.LENGTH, None, None, None, EnforcementMode.STRICT
        )
        is_valid, _errors = validate_output("test", config)
        assert is_valid is True


class TestJSONSchemaValidation:
    """Tests for JSON schema validation details."""

    def test_required_field_missing(self) -> None:
        """Missing required field is detected."""
        schema = {"type": "object", "required": ["name"], "properties": {"name": {}}}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, errors = validate_output('{"other": "value"}', config)
        assert is_valid is False
        assert any("name" in e for e in errors)

    def test_extra_property_not_allowed(self) -> None:
        """Extra property is detected when not allowed."""
        schema = {"type": "object", "properties": {"name": {}}}
        json_c = create_json_schema_constraint(schema=schema, allow_extra=False)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, errors = validate_output('{"name": "test", "extra": "value"}', config)
        assert is_valid is False
        assert any("Additional property" in e for e in errors)

    def test_extra_property_allowed(self) -> None:
        """Extra property is allowed when configured."""
        schema = {"type": "object", "properties": {"name": {}}}
        json_c = create_json_schema_constraint(schema=schema, allow_extra=True)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _errors = validate_output(
            '{"name": "test", "extra": "value"}', config
        )
        assert is_valid is True

    def test_wrong_type(self) -> None:
        """Wrong JSON type is detected."""
        schema = {"type": "object"}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, errors = validate_output("[1, 2, 3]", config)
        assert is_valid is False
        assert any("Expected type" in e for e in errors)

    def test_array_type_valid(self) -> None:
        """Array type validated correctly."""
        schema = {"type": "array"}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _errors = validate_output("[1, 2, 3]", config)
        assert is_valid is True

    def test_string_type_valid(self) -> None:
        """String type validated correctly."""
        schema = {"type": "string"}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _errors = validate_output('"hello"', config)
        assert is_valid is True

    def test_number_type_valid(self) -> None:
        """Number type validated correctly."""
        schema = {"type": "number"}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _errors = validate_output("42.5", config)
        assert is_valid is True

    def test_boolean_type_valid(self) -> None:
        """Boolean type validated correctly."""
        schema = {"type": "boolean"}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _errors = validate_output("true", config)
        assert is_valid is True

    def test_null_type_valid(self) -> None:
        """Null type validated correctly."""
        schema = {"type": "null"}
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _errors = validate_output("null", config)
        assert is_valid is True


class TestCompileConstraint:
    """Tests for compile_constraint function."""

    def test_compile_regex_constraint(self) -> None:
        """Compile regex constraint."""
        regex_c = create_regex_constraint(pattern=r"[A-Z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        compiled = compile_constraint(config)
        assert isinstance(compiled, CompiledConstraint)
        assert compiled.compiled_pattern is not None

    def test_compile_json_schema_constraint(self) -> None:
        """Compile JSON schema constraint (no pattern)."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        compiled = compile_constraint(config)
        assert compiled.compiled_pattern is None

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            compile_constraint(None)  # type: ignore[arg-type]

    def test_compiled_constraint_validate(self) -> None:
        """Compiled constraint validate method works."""
        regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        compiled = compile_constraint(config)
        is_valid, _errors = compiled.validate("Hello")
        assert is_valid is True

    def test_compiled_constraint_validate_failure(self) -> None:
        """Compiled constraint validate detects failures."""
        regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        compiled = compile_constraint(config)
        is_valid, _errors = compiled.validate("hello")
        assert is_valid is False

    def test_compiled_constraint_length_check(self) -> None:
        """Compiled constraint checks length."""
        regex_c = create_regex_constraint(pattern=r".*", max_length=5)
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        compiled = compile_constraint(config)
        is_valid, _errors = compiled.validate("hello world")
        assert is_valid is False

    def test_compiled_json_falls_back(self) -> None:
        """Compiled JSON constraint falls back to validate_output."""
        json_c = create_json_schema_constraint(
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            allow_extra=True,
        )
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        compiled = compile_constraint(config)
        is_valid, _errors = compiled.validate('{"name": "test"}')
        assert is_valid is True

    def test_compiled_regex_none_constraint(self) -> None:
        """Compiled regex with None constraint passes."""
        config = ConstraintConfig(
            ConstraintType.REGEX, None, None, None, EnforcementMode.STRICT
        )
        # This would fail validation, but we're testing the compiled case
        compiled = CompiledConstraint(config, None)
        is_valid, _errors = compiled.validate("test")
        assert is_valid is True


class TestEstimateConstraintOverhead:
    """Tests for estimate_constraint_overhead function."""

    def test_json_schema_overhead(self) -> None:
        """Estimate JSON schema overhead."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        overhead = estimate_constraint_overhead(config)
        assert overhead > 0

    def test_regex_overhead(self) -> None:
        """Estimate regex overhead."""
        regex_c = create_regex_constraint(pattern=r"[a-z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        overhead = estimate_constraint_overhead(config)
        assert overhead > 0

    def test_grammar_overhead(self) -> None:
        """Estimate grammar overhead."""
        grammar_c = create_grammar_constraint(grammar="root ::= 'x'")
        config = create_constraint_config(
            constraint_type="grammar", grammar_constraint=grammar_c
        )
        overhead = estimate_constraint_overhead(config)
        assert overhead > 0

    def test_soft_mode_lower_overhead(self) -> None:
        """Soft mode has lower overhead."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config_strict = create_constraint_config(
            constraint_type="json_schema",
            json_constraint=json_c,
            enforcement="strict",
        )
        config_soft = create_constraint_config(
            constraint_type="json_schema",
            json_constraint=json_c,
            enforcement="soft",
        )
        overhead_strict = estimate_constraint_overhead(config_strict)
        overhead_soft = estimate_constraint_overhead(config_soft)
        assert overhead_soft < overhead_strict

    def test_sample_and_filter_higher_overhead(self) -> None:
        """Sample and filter has higher overhead."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config_strict = create_constraint_config(
            constraint_type="json_schema",
            json_constraint=json_c,
            enforcement="strict",
        )
        config_saf = create_constraint_config(
            constraint_type="json_schema",
            json_constraint=json_c,
            enforcement="sample_and_filter",
        )
        overhead_strict = estimate_constraint_overhead(config_strict)
        overhead_saf = estimate_constraint_overhead(config_saf)
        assert overhead_saf > overhead_strict

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            estimate_constraint_overhead(None)  # type: ignore[arg-type]


class TestCombineConstraints:
    """Tests for combine_constraints function."""

    def test_combine_single_constraint(self) -> None:
        """Combine single constraint."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        combined = combine_constraints((config,))
        assert combined.constraint_type == ConstraintType.JSON_SCHEMA

    def test_combine_multiple_constraints(self) -> None:
        """Combine multiple constraints."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config1 = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        regex_c = create_regex_constraint(pattern=r".+")
        config2 = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        combined = combine_constraints((config1, config2), mode="all")
        assert combined is not None

    def test_empty_constraints_raises(self) -> None:
        """Empty constraints raises ValueError."""
        with pytest.raises(ValueError, match="constraints cannot be empty"):
            combine_constraints(())

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        with pytest.raises(ValueError, match="mode must be 'all' or 'any'"):
            combine_constraints((config,), mode="invalid")

    def test_any_mode(self) -> None:
        """'any' mode works."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        combined = combine_constraints((config,), mode="any")
        assert combined is not None


class TestFormatConstraintStats:
    """Tests for format_constraint_stats function."""

    def test_format_basic_stats(self) -> None:
        """Format basic stats."""
        stats = create_constraint_stats(
            valid_outputs=100, invalid_outputs=5, retries=10
        )
        result = format_constraint_stats(stats)
        assert "Valid outputs: 100" in result
        assert "Invalid outputs: 5" in result
        assert "Retries: 10" in result

    def test_format_with_time(self) -> None:
        """Format stats with time."""
        stats = create_constraint_stats(avg_generation_time_ms=150.5)
        result = format_constraint_stats(stats)
        assert "150.5ms" in result

    def test_format_success_rate(self) -> None:
        """Format includes success rate."""
        stats = create_constraint_stats(valid_outputs=90, invalid_outputs=10)
        result = format_constraint_stats(stats)
        assert "Success rate: 90.0%" in result

    def test_format_zero_total(self) -> None:
        """Format with zero total outputs."""
        stats = create_constraint_stats()
        result = format_constraint_stats(stats)
        assert "Success rate: 0.0%" in result

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_constraint_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedConstraintConfig:
    """Tests for get_recommended_constraint_config function."""

    def test_extraction_task(self) -> None:
        """Get config for extraction task."""
        config = get_recommended_constraint_config("extraction")
        assert config.constraint_type == ConstraintType.JSON_SCHEMA
        assert config.enforcement == EnforcementMode.STRICT

    def test_classification_task(self) -> None:
        """Get config for classification task."""
        config = get_recommended_constraint_config("classification")
        assert config.constraint_type == ConstraintType.CHOICES

    def test_generation_task(self) -> None:
        """Get config for generation task."""
        config = get_recommended_constraint_config("generation")
        assert config.constraint_type == ConstraintType.REGEX
        assert config.enforcement == EnforcementMode.SOFT

    def test_code_task(self) -> None:
        """Get config for code task."""
        config = get_recommended_constraint_config("code")
        assert config.constraint_type == ConstraintType.GRAMMAR
        assert config.enforcement == EnforcementMode.SAMPLE_AND_FILTER

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_constraint_config("invalid")

    @pytest.mark.parametrize(
        "task", ["extraction", "classification", "generation", "code"]
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All valid tasks return valid configs."""
        config = get_recommended_constraint_config(task)
        assert isinstance(config, ConstraintConfig)


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_regex_constraint_creation_valid_pattern(self, text: str) -> None:
        """Valid regex pattern creates constraint."""
        # Use simple patterns that are always valid
        pattern = re.escape(text)
        constraint = create_regex_constraint(pattern=pattern)
        assert constraint.pattern == pattern

    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=50)
    def test_regex_constraint_max_length_positive(self, max_length: int) -> None:
        """Positive max_length creates valid constraint."""
        constraint = create_regex_constraint(pattern=r".*", max_length=max_length)
        assert constraint.max_length == max_length

    @given(
        valid=st.integers(min_value=0, max_value=1000),
        invalid=st.integers(min_value=0, max_value=1000),
        retries=st.integers(min_value=0, max_value=1000),
        time_ms=st.floats(min_value=0.0, max_value=10000.0),
    )
    @settings(max_examples=50)
    def test_constraint_stats_non_negative(
        self, valid: int, invalid: int, retries: int, time_ms: float
    ) -> None:
        """Non-negative values create valid stats."""
        stats = create_constraint_stats(
            valid_outputs=valid,
            invalid_outputs=invalid,
            retries=retries,
            avg_generation_time_ms=time_ms,
        )
        assert stats.valid_outputs == valid
        assert stats.invalid_outputs == invalid
        assert stats.retries == retries
        assert stats.avg_generation_time_ms == time_ms

    @given(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(min_codepoint=65, max_codepoint=90),
        )
    )
    @settings(max_examples=50)
    def test_validate_output_uppercase_regex(self, text: str) -> None:
        """Uppercase text matches uppercase pattern."""
        regex_c = create_regex_constraint(pattern=r"[A-Z]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        is_valid, _ = validate_output(text, config)
        assert is_valid is True


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_complex_json_schema(self) -> None:
        """Complex JSON schema works."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "addresses": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["name"],
        }
        json_c = create_json_schema_constraint(schema=schema)
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _ = validate_output('{"name": "Test", "age": 30}', config)
        assert is_valid is True

    def test_unicode_regex(self) -> None:
        """Unicode in regex works."""
        regex_c = create_regex_constraint(pattern=r"[\u4e00-\u9fff]+")
        config = create_constraint_config(
            constraint_type="regex", regex_constraint=regex_c
        )
        # This should match Chinese characters
        is_valid, _ = validate_output("\u4e2d\u6587", config)
        assert is_valid is True

    def test_empty_json_array_valid(self) -> None:
        """Empty JSON array is valid for array type."""
        json_c = create_json_schema_constraint(schema={"type": "array"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _ = validate_output("[]", config)
        assert is_valid is True

    def test_empty_json_object_valid(self) -> None:
        """Empty JSON object is valid for object type."""
        json_c = create_json_schema_constraint(schema={"type": "object"})
        config = create_constraint_config(
            constraint_type="json_schema", json_constraint=json_c
        )
        is_valid, _ = validate_output("{}", config)
        assert is_valid is True

    def test_very_long_regex_pattern(self) -> None:
        """Very long regex pattern works."""
        pattern = r"[a-z]" * 100
        regex_c = create_regex_constraint(pattern=pattern)
        assert len(regex_c.pattern) == 500

    def test_special_characters_in_grammar(self) -> None:
        """Special characters in grammar work."""
        grammar_c = create_grammar_constraint(
            grammar="root ::= 'hello\\nworld' | '\\t'"
        )
        assert "\\n" in grammar_c.grammar

    def test_constraint_stats_all_valid(self) -> None:
        """Stats with all valid outputs."""
        stats = create_constraint_stats(valid_outputs=1000, invalid_outputs=0)
        result = format_constraint_stats(stats)
        assert "100.0%" in result

    def test_constraint_stats_all_invalid(self) -> None:
        """Stats with all invalid outputs."""
        stats = create_constraint_stats(valid_outputs=0, invalid_outputs=1000)
        result = format_constraint_stats(stats)
        assert "0.0%" in result
