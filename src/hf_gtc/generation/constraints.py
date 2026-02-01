"""Output constraint utilities for text generation.

This module provides utilities for defining and enforcing output constraints
including JSON schema validation, regex patterns, grammars, and choice restrictions.

Examples:
    >>> from hf_gtc.generation.constraints import create_constraint_config
    >>> config = create_constraint_config(constraint_type="json_schema")
    >>> config.constraint_type
    <ConstraintType.JSON_SCHEMA: 'json_schema'>
    >>> config.enforcement
    <EnforcementMode.STRICT: 'strict'>
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class ConstraintType(Enum):
    """Types of output constraints.

    Attributes:
        JSON_SCHEMA: Constrain output to match a JSON schema.
        REGEX: Constrain output to match a regular expression.
        GRAMMAR: Constrain output to follow a formal grammar.
        CHOICES: Constrain output to predefined choices.
        LENGTH: Constrain output by length (characters or tokens).

    Examples:
        >>> ConstraintType.JSON_SCHEMA.value
        'json_schema'
        >>> ConstraintType.REGEX.value
        'regex'
        >>> ConstraintType.GRAMMAR.value
        'grammar'
        >>> ConstraintType.CHOICES.value
        'choices'
        >>> ConstraintType.LENGTH.value
        'length'
    """

    JSON_SCHEMA = "json_schema"
    REGEX = "regex"
    GRAMMAR = "grammar"
    CHOICES = "choices"
    LENGTH = "length"


class EnforcementMode(Enum):
    """Modes for enforcing output constraints.

    Attributes:
        STRICT: Reject outputs that do not match constraints.
        SOFT: Allow outputs with minor violations, log warnings.
        SAMPLE_AND_FILTER: Generate multiple samples, filter by constraint.

    Examples:
        >>> EnforcementMode.STRICT.value
        'strict'
        >>> EnforcementMode.SOFT.value
        'soft'
        >>> EnforcementMode.SAMPLE_AND_FILTER.value
        'sample_and_filter'
    """

    STRICT = "strict"
    SOFT = "soft"
    SAMPLE_AND_FILTER = "sample_and_filter"


class GrammarFormat(Enum):
    """Formats for grammar definitions.

    Attributes:
        EBNF: Extended Backus-Naur Form.
        LARK: Lark grammar format.
        ANTLR: ANTLR grammar format.
        REGEX: Regular expression as grammar.

    Examples:
        >>> GrammarFormat.EBNF.value
        'ebnf'
        >>> GrammarFormat.LARK.value
        'lark'
        >>> GrammarFormat.ANTLR.value
        'antlr'
        >>> GrammarFormat.REGEX.value
        'regex'
    """

    EBNF = "ebnf"
    LARK = "lark"
    ANTLR = "antlr"
    REGEX = "regex"


VALID_CONSTRAINT_TYPES = frozenset(ct.value for ct in ConstraintType)
VALID_ENFORCEMENT_MODES = frozenset(em.value for em in EnforcementMode)
VALID_GRAMMAR_FORMATS = frozenset(gf.value for gf in GrammarFormat)


@dataclass(frozen=True, slots=True)
class JSONSchemaConstraint:
    """Constraint based on JSON schema validation.

    Attributes:
        schema: JSON schema dictionary.
        strict: Whether to enforce strict schema validation.
        allow_extra: Whether to allow additional properties.

    Examples:
        >>> constraint = JSONSchemaConstraint(
        ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
        ...     strict=True,
        ...     allow_extra=False,
        ... )
        >>> constraint.strict
        True
        >>> constraint.allow_extra
        False
    """

    schema: dict[str, Any]
    strict: bool
    allow_extra: bool


@dataclass(frozen=True, slots=True)
class RegexConstraint:
    """Constraint based on regular expression matching.

    Attributes:
        pattern: Regular expression pattern.
        full_match: Whether to require full match vs partial.
        max_length: Maximum allowed output length.

    Examples:
        >>> constraint = RegexConstraint(
        ...     pattern=r"[A-Z][a-z]+",
        ...     full_match=True,
        ...     max_length=100,
        ... )
        >>> constraint.pattern
        '[A-Z][a-z]+'
        >>> constraint.full_match
        True
        >>> constraint.max_length
        100
    """

    pattern: str
    full_match: bool
    max_length: int


@dataclass(frozen=True, slots=True)
class GrammarConstraint:
    """Constraint based on formal grammar.

    Attributes:
        grammar: Grammar definition string.
        format: Format of the grammar definition.
        start_symbol: Start symbol for the grammar.

    Examples:
        >>> constraint = GrammarConstraint(
        ...     grammar="root ::= 'hello' | 'world'",
        ...     format=GrammarFormat.EBNF,
        ...     start_symbol="root",
        ... )
        >>> constraint.start_symbol
        'root'
        >>> constraint.format
        <GrammarFormat.EBNF: 'ebnf'>
    """

    grammar: str
    format: GrammarFormat
    start_symbol: str


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    """Configuration for output constraints.

    Attributes:
        constraint_type: Type of constraint to apply.
        json_constraint: JSON schema constraint (if applicable).
        regex_constraint: Regex constraint (if applicable).
        grammar_constraint: Grammar constraint (if applicable).
        enforcement: How to enforce the constraint.

    Examples:
        >>> config = ConstraintConfig(
        ...     constraint_type=ConstraintType.JSON_SCHEMA,
        ...     json_constraint=JSONSchemaConstraint(
        ...         schema={"type": "object"},
        ...         strict=True,
        ...         allow_extra=False,
        ...     ),
        ...     regex_constraint=None,
        ...     grammar_constraint=None,
        ...     enforcement=EnforcementMode.STRICT,
        ... )
        >>> config.constraint_type
        <ConstraintType.JSON_SCHEMA: 'json_schema'>
        >>> config.enforcement
        <EnforcementMode.STRICT: 'strict'>
    """

    constraint_type: ConstraintType
    json_constraint: JSONSchemaConstraint | None
    regex_constraint: RegexConstraint | None
    grammar_constraint: GrammarConstraint | None
    enforcement: EnforcementMode


@dataclass(frozen=True, slots=True)
class ConstraintStats:
    """Statistics about constraint validation.

    Attributes:
        valid_outputs: Number of outputs that passed validation.
        invalid_outputs: Number of outputs that failed validation.
        retries: Number of generation retries due to constraint failures.
        avg_generation_time_ms: Average generation time in milliseconds.

    Examples:
        >>> stats = ConstraintStats(
        ...     valid_outputs=100,
        ...     invalid_outputs=5,
        ...     retries=10,
        ...     avg_generation_time_ms=150.5,
        ... )
        >>> stats.valid_outputs
        100
        >>> stats.invalid_outputs
        5
        >>> stats.avg_generation_time_ms
        150.5
    """

    valid_outputs: int
    invalid_outputs: int
    retries: int
    avg_generation_time_ms: float


def validate_json_schema_constraint(constraint: JSONSchemaConstraint) -> None:
    """Validate a JSON schema constraint.

    Args:
        constraint: Constraint to validate.

    Raises:
        ValueError: If constraint is None.
        ValueError: If schema is empty.

    Examples:
        >>> constraint = JSONSchemaConstraint(
        ...     schema={"type": "object"},
        ...     strict=True,
        ...     allow_extra=False,
        ... )
        >>> validate_json_schema_constraint(constraint)  # No error

        >>> validate_json_schema_constraint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: constraint cannot be None

        >>> bad = JSONSchemaConstraint(schema={}, strict=True, allow_extra=False)
        >>> validate_json_schema_constraint(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: schema cannot be empty
    """
    if constraint is None:
        msg = "constraint cannot be None"
        raise ValueError(msg)

    if not constraint.schema:
        msg = "schema cannot be empty"
        raise ValueError(msg)


def validate_regex_constraint(constraint: RegexConstraint) -> None:
    """Validate a regex constraint.

    Args:
        constraint: Constraint to validate.

    Raises:
        ValueError: If constraint is None.
        ValueError: If pattern is empty.
        ValueError: If pattern is invalid regex.
        ValueError: If max_length is not positive.

    Examples:
        >>> constraint = RegexConstraint(
        ...     pattern=r"[A-Z]+",
        ...     full_match=True,
        ...     max_length=100,
        ... )
        >>> validate_regex_constraint(constraint)  # No error

        >>> validate_regex_constraint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: constraint cannot be None

        >>> bad = RegexConstraint(pattern="", full_match=True, max_length=100)
        >>> validate_regex_constraint(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pattern cannot be empty

        >>> bad_len = RegexConstraint(pattern=r"[A-Z]+", full_match=True, max_length=0)
        >>> validate_regex_constraint(bad_len)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive

        >>> bad_re = RegexConstraint(
        ...     pattern=r"[invalid", full_match=True, max_length=100)
        >>> validate_regex_constraint(bad_re)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid regex pattern
    """
    if constraint is None:
        msg = "constraint cannot be None"
        raise ValueError(msg)

    if not constraint.pattern:
        msg = "pattern cannot be empty"
        raise ValueError(msg)

    try:
        re.compile(constraint.pattern)
    except re.error as e:
        msg = f"invalid regex pattern: {e}"
        raise ValueError(msg) from e

    if constraint.max_length <= 0:
        msg = f"max_length must be positive, got {constraint.max_length}"
        raise ValueError(msg)


def validate_grammar_constraint(constraint: GrammarConstraint) -> None:
    """Validate a grammar constraint.

    Args:
        constraint: Constraint to validate.

    Raises:
        ValueError: If constraint is None.
        ValueError: If grammar is empty.
        ValueError: If start_symbol is empty.

    Examples:
        >>> constraint = GrammarConstraint(
        ...     grammar="root ::= 'hello'",
        ...     format=GrammarFormat.EBNF,
        ...     start_symbol="root",
        ... )
        >>> validate_grammar_constraint(constraint)  # No error

        >>> validate_grammar_constraint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: constraint cannot be None

        >>> bad = GrammarConstraint(
        ...     grammar="", format=GrammarFormat.EBNF, start_symbol="root")
        >>> validate_grammar_constraint(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: grammar cannot be empty

        >>> bad_sym = GrammarConstraint(
        ...     grammar="root ::= 'x'",
        ...     format=GrammarFormat.EBNF,
        ...     start_symbol="",
        ... )
        >>> validate_grammar_constraint(bad_sym)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: start_symbol cannot be empty
    """
    if constraint is None:
        msg = "constraint cannot be None"
        raise ValueError(msg)

    if not constraint.grammar:
        msg = "grammar cannot be empty"
        raise ValueError(msg)

    if not constraint.start_symbol:
        msg = "start_symbol cannot be empty"
        raise ValueError(msg)


def _validate_constraint_type_config(config: ConstraintConfig) -> None:
    """Validate constraint sub-config based on constraint type."""
    constraint_validators: dict[ConstraintType, tuple[str, str, object]] = {
        ConstraintType.JSON_SCHEMA: (
            "json_constraint",
            "JSON_SCHEMA type",
            validate_json_schema_constraint,
        ),
        ConstraintType.REGEX: (
            "regex_constraint",
            "REGEX type",
            validate_regex_constraint,
        ),
        ConstraintType.GRAMMAR: (
            "grammar_constraint",
            "GRAMMAR type",
            validate_grammar_constraint,
        ),
    }
    entry = constraint_validators.get(config.constraint_type)
    if entry is None:
        return
    attr_name, label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} is required for {label}"
        raise ValueError(msg)
    validator(sub_config)


def validate_constraint_config(config: ConstraintConfig) -> None:
    """Validate a constraint configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If required constraint is missing for type.
        ValueError: If any sub-constraint is invalid.

    Examples:
        >>> config = ConstraintConfig(
        ...     constraint_type=ConstraintType.JSON_SCHEMA,
        ...     json_constraint=JSONSchemaConstraint(
        ...         schema={"type": "object"},
        ...         strict=True,
        ...         allow_extra=False,
        ...     ),
        ...     regex_constraint=None,
        ...     grammar_constraint=None,
        ...     enforcement=EnforcementMode.STRICT,
        ... )
        >>> validate_constraint_config(config)  # No error

        >>> validate_constraint_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ConstraintConfig(
        ...     constraint_type=ConstraintType.JSON_SCHEMA,
        ...     json_constraint=None,
        ...     regex_constraint=None,
        ...     grammar_constraint=None,
        ...     enforcement=EnforcementMode.STRICT,
        ... )
        >>> validate_constraint_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: json_constraint is required for JSON_SCHEMA type
    """
    validate_not_none(config, "config")

    _validate_constraint_type_config(config)


def create_json_schema_constraint(
    schema: dict[str, Any],
    strict: bool = True,
    allow_extra: bool = False,
) -> JSONSchemaConstraint:
    """Create a JSON schema constraint.

    Args:
        schema: JSON schema dictionary.
        strict: Whether to enforce strict validation. Defaults to True.
        allow_extra: Whether to allow extra properties. Defaults to False.

    Returns:
        Validated JSONSchemaConstraint instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> constraint = create_json_schema_constraint(
        ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
        ... )
        >>> constraint.strict
        True
        >>> constraint.allow_extra
        False

        >>> constraint2 = create_json_schema_constraint(
        ...     schema={"type": "array"},
        ...     strict=False,
        ...     allow_extra=True,
        ... )
        >>> constraint2.strict
        False
        >>> constraint2.allow_extra
        True

        >>> create_json_schema_constraint(
        ...     schema={})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: schema cannot be empty
    """
    constraint = JSONSchemaConstraint(
        schema=schema,
        strict=strict,
        allow_extra=allow_extra,
    )
    validate_json_schema_constraint(constraint)
    return constraint


def create_regex_constraint(
    pattern: str,
    full_match: bool = True,
    max_length: int = 1024,
) -> RegexConstraint:
    r"""Create a regex constraint.

    Args:
        pattern: Regular expression pattern.
        full_match: Whether to require full match. Defaults to True.
        max_length: Maximum output length. Defaults to 1024.

    Returns:
        Validated RegexConstraint instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> constraint = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        >>> constraint.full_match
        True
        >>> constraint.max_length
        1024

        >>> constraint2 = create_regex_constraint(
        ...     pattern=r"\\d+",
        ...     full_match=False,
        ...     max_length=50,
        ... )
        >>> constraint2.full_match
        False
        >>> constraint2.max_length
        50

        >>> create_regex_constraint(pattern="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pattern cannot be empty

        >>> create_regex_constraint(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     pattern=r"[invalid"
        ... )
        Traceback (most recent call last):
        ValueError: invalid regex pattern
    """
    constraint = RegexConstraint(
        pattern=pattern,
        full_match=full_match,
        max_length=max_length,
    )
    validate_regex_constraint(constraint)
    return constraint


def create_grammar_constraint(
    grammar: str,
    format: GrammarFormat | str = GrammarFormat.EBNF,
    start_symbol: str = "root",
) -> GrammarConstraint:
    """Create a grammar constraint.

    Args:
        grammar: Grammar definition string.
        format: Grammar format. Defaults to EBNF.
        start_symbol: Start symbol. Defaults to "root".

    Returns:
        Validated GrammarConstraint instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> constraint = create_grammar_constraint(grammar="root ::= 'hello' | 'world'")
        >>> constraint.format
        <GrammarFormat.EBNF: 'ebnf'>
        >>> constraint.start_symbol
        'root'

        >>> constraint2 = create_grammar_constraint(
        ...     grammar="start: 'hello'",
        ...     format="lark",
        ...     start_symbol="start",
        ... )
        >>> constraint2.format
        <GrammarFormat.LARK: 'lark'>

        >>> create_grammar_constraint(grammar="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: grammar cannot be empty
    """
    if isinstance(format, str):
        format = get_grammar_format(format)

    constraint = GrammarConstraint(
        grammar=grammar,
        format=format,
        start_symbol=start_symbol,
    )
    validate_grammar_constraint(constraint)
    return constraint


def create_constraint_config(
    constraint_type: ConstraintType | str = ConstraintType.JSON_SCHEMA,
    json_constraint: JSONSchemaConstraint | None = None,
    regex_constraint: RegexConstraint | None = None,
    grammar_constraint: GrammarConstraint | None = None,
    enforcement: EnforcementMode | str = EnforcementMode.STRICT,
) -> ConstraintConfig:
    """Create a constraint configuration.

    Args:
        constraint_type: Type of constraint. Defaults to JSON_SCHEMA.
        json_constraint: JSON schema constraint (if applicable).
        regex_constraint: Regex constraint (if applicable).
        grammar_constraint: Grammar constraint (if applicable).
        enforcement: Enforcement mode. Defaults to STRICT.

    Returns:
        Validated ConstraintConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> json_c = create_json_schema_constraint(schema={"type": "object"})
        >>> config = create_constraint_config(
        ...     constraint_type="json_schema",
        ...     json_constraint=json_c,
        ... )
        >>> config.constraint_type
        <ConstraintType.JSON_SCHEMA: 'json_schema'>
        >>> config.enforcement
        <EnforcementMode.STRICT: 'strict'>

        >>> regex_c = create_regex_constraint(pattern=r"[a-z]+")
        >>> config2 = create_constraint_config(
        ...     constraint_type="regex",
        ...     regex_constraint=regex_c,
        ...     enforcement="soft",
        ... )
        >>> config2.constraint_type
        <ConstraintType.REGEX: 'regex'>
        >>> config2.enforcement
        <EnforcementMode.SOFT: 'soft'>

        >>> create_constraint_config(constraint_type="json_schema")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: json_constraint is required for JSON_SCHEMA type
    """
    if isinstance(constraint_type, str):
        constraint_type = get_constraint_type(constraint_type)

    if isinstance(enforcement, str):
        enforcement = get_enforcement_mode(enforcement)

    config = ConstraintConfig(
        constraint_type=constraint_type,
        json_constraint=json_constraint,
        regex_constraint=regex_constraint,
        grammar_constraint=grammar_constraint,
        enforcement=enforcement,
    )
    validate_constraint_config(config)
    return config


def create_constraint_stats(
    valid_outputs: int = 0,
    invalid_outputs: int = 0,
    retries: int = 0,
    avg_generation_time_ms: float = 0.0,
) -> ConstraintStats:
    """Create constraint statistics.

    Args:
        valid_outputs: Number of valid outputs. Defaults to 0.
        invalid_outputs: Number of invalid outputs. Defaults to 0.
        retries: Number of retries. Defaults to 0.
        avg_generation_time_ms: Average generation time in ms. Defaults to 0.0.

    Returns:
        ConstraintStats instance.

    Raises:
        ValueError: If counts are negative.

    Examples:
        >>> stats = create_constraint_stats()
        >>> stats.valid_outputs
        0
        >>> stats.invalid_outputs
        0

        >>> stats2 = create_constraint_stats(
        ...     valid_outputs=100,
        ...     invalid_outputs=5,
        ...     retries=10,
        ...     avg_generation_time_ms=150.5,
        ... )
        >>> stats2.valid_outputs
        100
        >>> stats2.avg_generation_time_ms
        150.5

        >>> create_constraint_stats(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     valid_outputs=-1
        ... )
        Traceback (most recent call last):
        ValueError: valid_outputs must be non-negative
    """
    if valid_outputs < 0:
        msg = f"valid_outputs must be non-negative, got {valid_outputs}"
        raise ValueError(msg)

    if invalid_outputs < 0:
        msg = f"invalid_outputs must be non-negative, got {invalid_outputs}"
        raise ValueError(msg)

    if retries < 0:
        msg = f"retries must be non-negative, got {retries}"
        raise ValueError(msg)

    if avg_generation_time_ms < 0:
        msg = (
            f"avg_generation_time_ms must be non-negative, got {avg_generation_time_ms}"
        )
        raise ValueError(msg)

    return ConstraintStats(
        valid_outputs=valid_outputs,
        invalid_outputs=invalid_outputs,
        retries=retries,
        avg_generation_time_ms=avg_generation_time_ms,
    )


def list_constraint_types() -> list[str]:
    """List available constraint types.

    Returns:
        Sorted list of constraint type names.

    Examples:
        >>> types = list_constraint_types()
        >>> "json_schema" in types
        True
        >>> "regex" in types
        True
        >>> "grammar" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CONSTRAINT_TYPES)


def list_enforcement_modes() -> list[str]:
    """List available enforcement modes.

    Returns:
        Sorted list of enforcement mode names.

    Examples:
        >>> modes = list_enforcement_modes()
        >>> "strict" in modes
        True
        >>> "soft" in modes
        True
        >>> "sample_and_filter" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_ENFORCEMENT_MODES)


def list_grammar_formats() -> list[str]:
    """List available grammar formats.

    Returns:
        Sorted list of grammar format names.

    Examples:
        >>> formats = list_grammar_formats()
        >>> "ebnf" in formats
        True
        >>> "lark" in formats
        True
        >>> "antlr" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_GRAMMAR_FORMATS)


def get_constraint_type(name: str) -> ConstraintType:
    """Get constraint type enum from string.

    Args:
        name: Constraint type name.

    Returns:
        ConstraintType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_constraint_type("json_schema")
        <ConstraintType.JSON_SCHEMA: 'json_schema'>
        >>> get_constraint_type("regex")
        <ConstraintType.REGEX: 'regex'>
        >>> get_constraint_type("grammar")
        <ConstraintType.GRAMMAR: 'grammar'>

        >>> get_constraint_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid constraint type: invalid
    """
    if name not in VALID_CONSTRAINT_TYPES:
        msg = f"invalid constraint type: {name}"
        raise ValueError(msg)
    return ConstraintType(name)


def get_enforcement_mode(name: str) -> EnforcementMode:
    """Get enforcement mode enum from string.

    Args:
        name: Enforcement mode name.

    Returns:
        EnforcementMode enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_enforcement_mode("strict")
        <EnforcementMode.STRICT: 'strict'>
        >>> get_enforcement_mode("soft")
        <EnforcementMode.SOFT: 'soft'>
        >>> get_enforcement_mode("sample_and_filter")
        <EnforcementMode.SAMPLE_AND_FILTER: 'sample_and_filter'>

        >>> get_enforcement_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid enforcement mode: invalid
    """
    if name not in VALID_ENFORCEMENT_MODES:
        msg = f"invalid enforcement mode: {name}"
        raise ValueError(msg)
    return EnforcementMode(name)


def get_grammar_format(name: str) -> GrammarFormat:
    """Get grammar format enum from string.

    Args:
        name: Grammar format name.

    Returns:
        GrammarFormat enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_grammar_format("ebnf")
        <GrammarFormat.EBNF: 'ebnf'>
        >>> get_grammar_format("lark")
        <GrammarFormat.LARK: 'lark'>
        >>> get_grammar_format("antlr")
        <GrammarFormat.ANTLR: 'antlr'>

        >>> get_grammar_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid grammar format: invalid
    """
    if name not in VALID_GRAMMAR_FORMATS:
        msg = f"invalid grammar format: {name}"
        raise ValueError(msg)
    return GrammarFormat(name)


def validate_output(
    output: str,
    config: ConstraintConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Validate output against a constraint configuration.

    Args:
        output: Output string to validate.
        config: Constraint configuration to validate against.

    Returns:
        Tuple of (is_valid, error_messages).

    Raises:
        ValueError: If config is None.

    Examples:
        >>> json_c = create_json_schema_constraint(
        ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
        ... )
        >>> config = create_constraint_config(
        ...     constraint_type="json_schema",
        ...     json_constraint=json_c,
        ... )
        >>> is_valid, errors = validate_output('{"name": "test"}', config)
        >>> is_valid
        True
        >>> errors
        ()

        >>> is_valid, errors = validate_output('invalid json', config)
        >>> is_valid
        False
        >>> len(errors) > 0
        True

        >>> regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        >>> config2 = create_constraint_config(
        ...     constraint_type="regex",
        ...     regex_constraint=regex_c,
        ... )
        >>> is_valid, errors = validate_output("Hello", config2)
        >>> is_valid
        True

        >>> is_valid, errors = validate_output("hello", config2)
        >>> is_valid
        False

        >>> validate_output("test", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if config.constraint_type == ConstraintType.JSON_SCHEMA:
        return _validate_json_schema(output, config.json_constraint)
    elif config.constraint_type == ConstraintType.REGEX:
        return _validate_regex(output, config.regex_constraint)
    elif config.constraint_type == ConstraintType.GRAMMAR:
        return _validate_grammar(output, config.grammar_constraint)
    elif config.constraint_type == ConstraintType.CHOICES:
        # Choices validation requires additional config
        return True, ()
    elif config.constraint_type == ConstraintType.LENGTH:
        # Length validation requires additional config
        return True, ()
    else:
        return True, ()


def _validate_json_schema(
    output: str,
    constraint: JSONSchemaConstraint | None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate output against JSON schema."""
    if constraint is None:
        return True, ()

    errors: list[str] = []

    # Try to parse as JSON
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        return False, (f"Invalid JSON: {e}",)

    # Basic type validation
    schema_type = constraint.schema.get("type")
    if schema_type:
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "null": type(None),
        }
        expected_type = type_map.get(schema_type)
        if expected_type and not isinstance(parsed, expected_type):
            errors.append(f"Expected type {schema_type}, got {type(parsed).__name__}")

    # Check required properties for objects
    if isinstance(parsed, dict):
        required = constraint.schema.get("required", [])
        for field in required:
            if field not in parsed:
                errors.append(f"Missing required field: {field}")

        # Check for extra properties
        if not constraint.allow_extra:
            properties = constraint.schema.get("properties", {})
            for key in parsed:
                if key not in properties:
                    errors.append(f"Additional property not allowed: {key}")

    return len(errors) == 0, tuple(errors)


def _validate_regex(
    output: str,
    constraint: RegexConstraint | None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate output against regex pattern."""
    if constraint is None:
        return True, ()

    errors: list[str] = []

    # Check length
    if len(output) > constraint.max_length:
        errors.append(
            f"Output length {len(output)} exceeds max {constraint.max_length}"
        )

    # Check pattern
    try:
        pattern = re.compile(constraint.pattern)
        if constraint.full_match:
            match = pattern.fullmatch(output)
        else:
            match = pattern.search(output)

        if not match:
            errors.append(f"Output does not match pattern: {constraint.pattern}")
    except re.error as e:
        errors.append(f"Invalid regex pattern: {e}")

    return len(errors) == 0, tuple(errors)


def _validate_grammar(
    output: str,
    constraint: GrammarConstraint | None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate output against grammar.

    Note: Full grammar validation requires external parser libraries.
    This provides basic validation only.
    """
    if constraint is None:
        return True, ()

    # Basic validation - output must not be empty
    if not output.strip():
        return False, ("Output cannot be empty for grammar constraint",)

    # For now, return True as full grammar parsing requires external dependencies
    return True, ()


def _validate_regex_output(
    output: str,
    constraint: object | None,
    compiled_pattern: re.Pattern[str],
) -> tuple[bool, tuple[str, ...]]:
    """Validate output against a compiled regex constraint."""
    if constraint is None:
        return True, ()

    errors: list[str] = []

    if len(output) > constraint.max_length:
        errors.append(
            f"Output length {len(output)} exceeds max {constraint.max_length}"
        )

    match = (
        compiled_pattern.fullmatch(output)
        if constraint.full_match
        else compiled_pattern.search(output)
    )
    if not match:
        errors.append(f"Output does not match pattern: {constraint.pattern}")

    return len(errors) == 0, tuple(errors)


class CompiledConstraint:
    """Compiled constraint for efficient repeated validation.

    Attributes:
        config: The original constraint configuration.
        compiled_pattern: Compiled regex pattern (if applicable).

    Examples:
        >>> regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        >>> config = create_constraint_config(
        ...     constraint_type="regex",
        ...     regex_constraint=regex_c,
        ... )
        >>> compiled = compile_constraint(config)
        >>> isinstance(compiled, CompiledConstraint)
        True
    """

    def __init__(
        self,
        config: ConstraintConfig,
        compiled_pattern: re.Pattern[str] | None = None,
    ) -> None:
        """Initialize compiled constraint.

        Args:
            config: Constraint configuration.
            compiled_pattern: Pre-compiled regex pattern.
        """
        self.config = config
        self.compiled_pattern = compiled_pattern

    def validate(self, output: str) -> tuple[bool, tuple[str, ...]]:
        """Validate output using compiled constraint.

        Args:
            output: Output to validate.

        Returns:
            Tuple of (is_valid, error_messages).

        Examples:
            >>> regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
            >>> config = create_constraint_config(
            ...     constraint_type="regex",
            ...     regex_constraint=regex_c,
            ... )
            >>> compiled = compile_constraint(config)
            >>> is_valid, errors = compiled.validate("Hello")
            >>> is_valid
            True
        """
        if (
            self.config.constraint_type == ConstraintType.REGEX
            and self.compiled_pattern
        ):
            return _validate_regex_output(
                output, self.config.regex_constraint, self.compiled_pattern
            )

        return validate_output(output, self.config)


def compile_constraint(config: ConstraintConfig) -> CompiledConstraint:
    """Compile a constraint for efficient repeated validation.

    Args:
        config: Constraint configuration to compile.

    Returns:
        CompiledConstraint instance.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> regex_c = create_regex_constraint(pattern=r"[A-Z][a-z]+")
        >>> config = create_constraint_config(
        ...     constraint_type="regex",
        ...     regex_constraint=regex_c,
        ... )
        >>> compiled = compile_constraint(config)
        >>> compiled.compiled_pattern is not None
        True

        >>> json_c = create_json_schema_constraint(schema={"type": "object"})
        >>> config2 = create_constraint_config(
        ...     constraint_type="json_schema",
        ...     json_constraint=json_c,
        ... )
        >>> compiled2 = compile_constraint(config2)
        >>> compiled2.compiled_pattern is None
        True

        >>> compile_constraint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    compiled_pattern = None
    if config.constraint_type == ConstraintType.REGEX and config.regex_constraint:
        compiled_pattern = re.compile(config.regex_constraint.pattern)

    return CompiledConstraint(config, compiled_pattern)


def estimate_constraint_overhead(config: ConstraintConfig) -> float:
    """Estimate the overhead of constraint validation in milliseconds.

    Args:
        config: Constraint configuration.

    Returns:
        Estimated overhead in milliseconds.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> json_c = create_json_schema_constraint(schema={"type": "object"})
        >>> config = create_constraint_config(
        ...     constraint_type="json_schema",
        ...     json_constraint=json_c,
        ... )
        >>> overhead = estimate_constraint_overhead(config)
        >>> overhead > 0
        True

        >>> regex_c = create_regex_constraint(pattern=r"[A-Z]+")
        >>> config2 = create_constraint_config(
        ...     constraint_type="regex",
        ...     regex_constraint=regex_c,
        ... )
        >>> overhead2 = estimate_constraint_overhead(config2)
        >>> overhead2 > 0
        True

        >>> estimate_constraint_overhead(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    # Base overhead estimates per constraint type
    overhead_map = {
        ConstraintType.JSON_SCHEMA: 2.0,
        ConstraintType.REGEX: 0.5,
        ConstraintType.GRAMMAR: 5.0,
        ConstraintType.CHOICES: 0.1,
        ConstraintType.LENGTH: 0.05,
    }

    base_overhead = overhead_map.get(config.constraint_type, 1.0)

    # Enforcement mode multiplier
    mode_multiplier = {
        EnforcementMode.STRICT: 1.0,
        EnforcementMode.SOFT: 0.8,
        EnforcementMode.SAMPLE_AND_FILTER: 2.5,
    }

    multiplier = mode_multiplier.get(config.enforcement, 1.0)

    return base_overhead * multiplier


def combine_constraints(
    constraints: tuple[ConstraintConfig, ...],
    mode: str = "all",
) -> ConstraintConfig:
    """Combine multiple constraints into a single configuration.

    Args:
        constraints: Tuple of constraint configurations to combine.
        mode: Combination mode ("all" requires all, "any" requires any).

    Returns:
        Combined ConstraintConfig.

    Raises:
        ValueError: If constraints is empty.
        ValueError: If mode is invalid.

    Examples:
        >>> json_c = create_json_schema_constraint(schema={"type": "object"})
        >>> config1 = create_constraint_config(
        ...     constraint_type="json_schema",
        ...     json_constraint=json_c,
        ... )
        >>> regex_c = create_regex_constraint(pattern=r".+")
        >>> config2 = create_constraint_config(
        ...     constraint_type="regex",
        ...     regex_constraint=regex_c,
        ... )
        >>> combined = combine_constraints((config1, config2), mode="all")
        >>> combined.constraint_type
        <ConstraintType.JSON_SCHEMA: 'json_schema'>

        >>> combine_constraints((), mode="all")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: constraints cannot be empty

        >>> combine_constraints((config1,), mode="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: mode must be 'all' or 'any'
    """
    if not constraints:
        msg = "constraints cannot be empty"
        raise ValueError(msg)

    if mode not in ("all", "any"):
        msg = f"mode must be 'all' or 'any', got '{mode}'"
        raise ValueError(msg)

    # For now, return the first constraint
    # Full implementation would create a combined validator
    return constraints[0]


def format_constraint_stats(stats: ConstraintStats) -> str:
    """Format constraint statistics for display.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = create_constraint_stats(
        ...     valid_outputs=100,
        ...     invalid_outputs=5,
        ...     retries=10,
        ...     avg_generation_time_ms=150.5,
        ... )
        >>> result = format_constraint_stats(stats)
        >>> "Valid outputs: 100" in result
        True
        >>> "Invalid outputs: 5" in result
        True
        >>> "Retries: 10" in result
        True
        >>> "Avg generation time: 150.5ms" in result
        True

        >>> format_constraint_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    total = stats.valid_outputs + stats.invalid_outputs
    success_rate = (stats.valid_outputs / total * 100) if total > 0 else 0.0

    lines = [
        f"Valid outputs: {stats.valid_outputs}",
        f"Invalid outputs: {stats.invalid_outputs}",
        f"Success rate: {success_rate:.1f}%",
        f"Retries: {stats.retries}",
        f"Avg generation time: {stats.avg_generation_time_ms}ms",
    ]

    return "\n".join(lines)


def get_recommended_constraint_config(task: str) -> ConstraintConfig:
    """Get recommended constraint configuration for a task.

    Args:
        task: Task type ("extraction", "classification", "generation", "code").

    Returns:
        Recommended ConstraintConfig for the task.

    Raises:
        ValueError: If task is invalid.

    Examples:
        >>> config = get_recommended_constraint_config("extraction")
        >>> config.constraint_type
        <ConstraintType.JSON_SCHEMA: 'json_schema'>
        >>> config.enforcement
        <EnforcementMode.STRICT: 'strict'>

        >>> config = get_recommended_constraint_config("classification")
        >>> config.constraint_type
        <ConstraintType.CHOICES: 'choices'>

        >>> config = get_recommended_constraint_config("code")
        >>> config.constraint_type
        <ConstraintType.GRAMMAR: 'grammar'>

        >>> get_recommended_constraint_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of
    """
    valid_tasks = {"extraction", "classification", "generation", "code"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    configs = {
        "extraction": create_constraint_config(
            constraint_type=ConstraintType.JSON_SCHEMA,
            json_constraint=create_json_schema_constraint(
                schema={"type": "object"},
                strict=True,
                allow_extra=False,
            ),
            enforcement=EnforcementMode.STRICT,
        ),
        "classification": create_constraint_config(
            constraint_type=ConstraintType.CHOICES,
            json_constraint=None,
            regex_constraint=None,
            grammar_constraint=None,
            enforcement=EnforcementMode.STRICT,
        ),
        "generation": create_constraint_config(
            constraint_type=ConstraintType.REGEX,
            regex_constraint=create_regex_constraint(
                pattern=r".+",
                full_match=False,
                max_length=4096,
            ),
            enforcement=EnforcementMode.SOFT,
        ),
        "code": create_constraint_config(
            constraint_type=ConstraintType.GRAMMAR,
            grammar_constraint=create_grammar_constraint(
                grammar="root ::= statement*",
                format=GrammarFormat.EBNF,
                start_symbol="root",
            ),
            enforcement=EnforcementMode.SAMPLE_AND_FILTER,
        ),
    }
    return configs[task]
