"""Structured output generation utilities.

This module provides utilities for generating structured output from LLMs,
including JSON schemas, grammars, and constrained choices.

Examples:
    >>> from hf_gtc.generation.structured import create_structured_config
    >>> config = create_structured_config(output_format="json", strict_mode=True)
    >>> config.output_format
    <OutputFormat.JSON: 'json'>
    >>> config.strict_mode
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class OutputFormat(Enum):
    """Supported output formats for structured generation.

    Attributes:
        JSON: Free-form JSON output.
        JSON_SCHEMA: JSON constrained by a schema.
        REGEX: Output matching a regular expression.
        GRAMMAR: Output following a formal grammar.
        CHOICES: Output from a predefined set of choices.

    Examples:
        >>> OutputFormat.JSON.value
        'json'
        >>> OutputFormat.JSON_SCHEMA.value
        'json_schema'
        >>> OutputFormat.GRAMMAR.value
        'grammar'
    """

    JSON = "json"
    JSON_SCHEMA = "json_schema"
    REGEX = "regex"
    GRAMMAR = "grammar"
    CHOICES = "choices"


class SchemaType(Enum):
    """JSON Schema types.

    Attributes:
        OBJECT: Object/dict type.
        ARRAY: Array/list type.
        STRING: String type.
        NUMBER: Numeric type (integer or float).
        BOOLEAN: Boolean type.
        NULL: Null type.

    Examples:
        >>> SchemaType.OBJECT.value
        'object'
        >>> SchemaType.STRING.value
        'string'
        >>> SchemaType.BOOLEAN.value
        'boolean'
    """

    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"


VALID_OUTPUT_FORMATS = frozenset(f.value for f in OutputFormat)
VALID_SCHEMA_TYPES = frozenset(t.value for t in SchemaType)


@dataclass(frozen=True, slots=True)
class StructuredConfig:
    """Configuration for structured output generation.

    Attributes:
        output_format: Format of the structured output.
        schema: Optional schema dict for JSON_SCHEMA format.
        strict_mode: Whether to enforce strict schema compliance.
        max_retries: Maximum retries for failed validation.

    Examples:
        >>> config = StructuredConfig(
        ...     output_format=OutputFormat.JSON,
        ...     schema=None,
        ...     strict_mode=True,
        ...     max_retries=3,
        ... )
        >>> config.output_format
        <OutputFormat.JSON: 'json'>
        >>> config.strict_mode
        True
    """

    output_format: OutputFormat
    schema: dict[str, Any] | None
    strict_mode: bool
    max_retries: int


@dataclass(frozen=True, slots=True)
class JSONSchemaConfig:
    """Configuration for JSON schema validation.

    Attributes:
        schema_dict: The JSON schema dictionary.
        required_fields: Tuple of required field names.
        additional_properties: Whether to allow additional properties.

    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> config = JSONSchemaConfig(
        ...     schema_dict=schema,
        ...     required_fields=("name",),
        ...     additional_properties=False,
        ... )
        >>> config.required_fields
        ('name',)
        >>> config.additional_properties
        False
    """

    schema_dict: dict[str, Any]
    required_fields: tuple[str, ...]
    additional_properties: bool


@dataclass(frozen=True, slots=True)
class GrammarConfig:
    """Configuration for grammar-based generation.

    Attributes:
        grammar_string: The grammar definition string (e.g., EBNF).
        start_symbol: The start symbol for the grammar.
        max_tokens: Maximum tokens to generate.

    Examples:
        >>> config = GrammarConfig(
        ...     grammar_string="root ::= 'hello' | 'world'",
        ...     start_symbol="root",
        ...     max_tokens=100,
        ... )
        >>> config.start_symbol
        'root'
        >>> config.max_tokens
        100
    """

    grammar_string: str
    start_symbol: str
    max_tokens: int


@dataclass(frozen=True, slots=True)
class ChoicesConfig:
    """Configuration for choice-based generation.

    Attributes:
        choices: Tuple of allowed choices.
        allow_multiple: Whether multiple selections are allowed.
        separator: Separator for multiple selections.

    Examples:
        >>> config = ChoicesConfig(
        ...     choices=("yes", "no", "maybe"),
        ...     allow_multiple=False,
        ...     separator=", ",
        ... )
        >>> config.choices
        ('yes', 'no', 'maybe')
        >>> config.allow_multiple
        False
    """

    choices: tuple[str, ...]
    allow_multiple: bool
    separator: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating structured output.

    Attributes:
        is_valid: Whether the output is valid.
        errors: Tuple of validation error messages.
        parsed_output: The parsed output if valid, None otherwise.

    Examples:
        >>> result = ValidationResult(
        ...     is_valid=True,
        ...     errors=(),
        ...     parsed_output={"name": "test"},
        ... )
        >>> result.is_valid
        True
        >>> result.errors
        ()

        >>> result_error = ValidationResult(
        ...     is_valid=False,
        ...     errors=("Missing required field: name",),
        ...     parsed_output=None,
        ... )
        >>> result_error.is_valid
        False
        >>> len(result_error.errors)
        1
    """

    is_valid: bool
    errors: tuple[str, ...]
    parsed_output: Any | None


def validate_structured_config(config: StructuredConfig) -> None:
    """Validate structured configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_retries is negative.
        ValueError: If schema is required but not provided.

    Examples:
        >>> config = StructuredConfig(
        ...     output_format=OutputFormat.JSON,
        ...     schema=None,
        ...     strict_mode=True,
        ...     max_retries=3,
        ... )
        >>> validate_structured_config(config)  # No error

        >>> validate_structured_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = StructuredConfig(
        ...     output_format=OutputFormat.JSON,
        ...     schema=None,
        ...     strict_mode=True,
        ...     max_retries=-1,
        ... )
        >>> validate_structured_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_retries must be non-negative

        >>> schema_required = StructuredConfig(
        ...     output_format=OutputFormat.JSON_SCHEMA,
        ...     schema=None,
        ...     strict_mode=True,
        ...     max_retries=3,
        ... )
        >>> validate_structured_config(schema_required)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: schema is required for JSON_SCHEMA format
    """
    validate_not_none(config, "config")

    if config.max_retries < 0:
        msg = f"max_retries must be non-negative, got {config.max_retries}"
        raise ValueError(msg)

    if config.output_format == OutputFormat.JSON_SCHEMA and config.schema is None:
        msg = "schema is required for JSON_SCHEMA format"
        raise ValueError(msg)


def validate_json_schema_config(config: JSONSchemaConfig) -> None:
    """Validate JSON schema configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If schema_dict is empty.
        ValueError: If required_fields contains non-string values.

    Examples:
        >>> config = JSONSchemaConfig(
        ...     schema_dict={"type": "object"},
        ...     required_fields=("name",),
        ...     additional_properties=False,
        ... )
        >>> validate_json_schema_config(config)  # No error

        >>> validate_json_schema_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> empty_schema = JSONSchemaConfig(
        ...     schema_dict={},
        ...     required_fields=(),
        ...     additional_properties=True,
        ... )
        >>> validate_json_schema_config(empty_schema)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: schema_dict cannot be empty
    """
    validate_not_none(config, "config")

    if not config.schema_dict:
        msg = "schema_dict cannot be empty"
        raise ValueError(msg)

    for field in config.required_fields:
        if not isinstance(field, str):
            msg = f"required_fields must contain strings, got {type(field).__name__}"
            raise ValueError(msg)


def validate_grammar_config(config: GrammarConfig) -> None:
    """Validate grammar configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If grammar_string is empty.
        ValueError: If start_symbol is empty.
        ValueError: If max_tokens is not positive.

    Examples:
        >>> config = GrammarConfig(
        ...     grammar_string="root ::= 'test'",
        ...     start_symbol="root",
        ...     max_tokens=100,
        ... )
        >>> validate_grammar_config(config)  # No error

        >>> validate_grammar_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> empty_grammar = GrammarConfig(
        ...     grammar_string="",
        ...     start_symbol="root",
        ...     max_tokens=100,
        ... )
        >>> validate_grammar_config(empty_grammar)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: grammar_string cannot be empty
    """
    validate_not_none(config, "config")

    if not config.grammar_string:
        msg = "grammar_string cannot be empty"
        raise ValueError(msg)

    if not config.start_symbol:
        msg = "start_symbol cannot be empty"
        raise ValueError(msg)

    if config.max_tokens <= 0:
        msg = f"max_tokens must be positive, got {config.max_tokens}"
        raise ValueError(msg)


def validate_choices_config(config: ChoicesConfig) -> None:
    """Validate choices configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If choices is empty.
        ValueError: If separator is empty when allow_multiple is True.

    Examples:
        >>> config = ChoicesConfig(
        ...     choices=("yes", "no"),
        ...     allow_multiple=False,
        ...     separator=", ",
        ... )
        >>> validate_choices_config(config)  # No error

        >>> validate_choices_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> empty_choices = ChoicesConfig(
        ...     choices=(),
        ...     allow_multiple=False,
        ...     separator=", ",
        ... )
        >>> validate_choices_config(empty_choices)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: choices cannot be empty
    """
    validate_not_none(config, "config")

    if not config.choices:
        msg = "choices cannot be empty"
        raise ValueError(msg)

    if config.allow_multiple and not config.separator:
        msg = "separator cannot be empty when allow_multiple is True"
        raise ValueError(msg)


def create_structured_config(
    output_format: OutputFormat | str = OutputFormat.JSON,
    schema: dict[str, Any] | None = None,
    strict_mode: bool = True,
    max_retries: int = 3,
) -> StructuredConfig:
    """Create a structured output configuration.

    Args:
        output_format: Format of the structured output. Defaults to JSON.
        schema: Optional schema dict for JSON_SCHEMA format.
        strict_mode: Whether to enforce strict schema compliance. Defaults to True.
        max_retries: Maximum retries for failed validation. Defaults to 3.

    Returns:
        Validated StructuredConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_structured_config(output_format="json")
        >>> config.output_format
        <OutputFormat.JSON: 'json'>

        >>> config2 = create_structured_config(
        ...     output_format="json_schema",
        ...     schema={"type": "object"},
        ... )
        >>> config2.output_format
        <OutputFormat.JSON_SCHEMA: 'json_schema'>

        >>> create_structured_config(max_retries=-1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_retries must be non-negative
    """
    if isinstance(output_format, str):
        output_format = get_output_format(output_format)

    config = StructuredConfig(
        output_format=output_format,
        schema=schema,
        strict_mode=strict_mode,
        max_retries=max_retries,
    )
    validate_structured_config(config)
    return config


def create_json_schema_config(
    schema_dict: dict[str, Any],
    required_fields: tuple[str, ...] = (),
    additional_properties: bool = False,
) -> JSONSchemaConfig:
    """Create a JSON schema configuration.

    Args:
        schema_dict: The JSON schema dictionary.
        required_fields: Tuple of required field names. Defaults to empty.
        additional_properties: Allow additional properties. Defaults to False.

    Returns:
        Validated JSONSchemaConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> config = create_json_schema_config(
        ...     schema_dict=schema,
        ...     required_fields=("name",),
        ... )
        >>> config.required_fields
        ('name',)

        >>> create_json_schema_config(schema_dict={})
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: schema_dict cannot be empty
    """
    config = JSONSchemaConfig(
        schema_dict=schema_dict,
        required_fields=required_fields,
        additional_properties=additional_properties,
    )
    validate_json_schema_config(config)
    return config


def create_grammar_config(
    grammar_string: str,
    start_symbol: str = "root",
    max_tokens: int = 1024,
) -> GrammarConfig:
    """Create a grammar configuration.

    Args:
        grammar_string: The grammar definition string (e.g., EBNF).
        start_symbol: The start symbol for the grammar. Defaults to "root".
        max_tokens: Maximum tokens to generate. Defaults to 1024.

    Returns:
        Validated GrammarConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_grammar_config(
        ...     grammar_string="root ::= 'hello' | 'world'",
        ... )
        >>> config.start_symbol
        'root'
        >>> config.max_tokens
        1024

        >>> create_grammar_config(grammar_string="")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: grammar_string cannot be empty
    """
    config = GrammarConfig(
        grammar_string=grammar_string,
        start_symbol=start_symbol,
        max_tokens=max_tokens,
    )
    validate_grammar_config(config)
    return config


def create_choices_config(
    choices: tuple[str, ...],
    allow_multiple: bool = False,
    separator: str = ", ",
) -> ChoicesConfig:
    """Create a choices configuration.

    Args:
        choices: Tuple of allowed choices.
        allow_multiple: Whether multiple selections are allowed. Defaults to False.
        separator: Separator for multiple selections. Defaults to ", ".

    Returns:
        Validated ChoicesConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_choices_config(
        ...     choices=("yes", "no", "maybe"),
        ... )
        >>> config.choices
        ('yes', 'no', 'maybe')
        >>> config.allow_multiple
        False

        >>> config2 = create_choices_config(
        ...     choices=("apple", "banana", "cherry"),
        ...     allow_multiple=True,
        ...     separator="; ",
        ... )
        >>> config2.separator
        '; '

        >>> create_choices_config(choices=())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: choices cannot be empty
    """
    config = ChoicesConfig(
        choices=choices,
        allow_multiple=allow_multiple,
        separator=separator,
    )
    validate_choices_config(config)
    return config


def create_validation_result(
    is_valid: bool,
    errors: tuple[str, ...] = (),
    parsed_output: Any | None = None,
) -> ValidationResult:
    """Create a validation result.

    Args:
        is_valid: Whether the output is valid.
        errors: Tuple of validation error messages. Defaults to ().
        parsed_output: The parsed output if valid. Defaults to None.

    Returns:
        ValidationResult instance.

    Examples:
        >>> result = create_validation_result(
        ...     is_valid=True,
        ...     parsed_output={"name": "test"},
        ... )
        >>> result.is_valid
        True

        >>> result_error = create_validation_result(
        ...     is_valid=False,
        ...     errors=("Invalid JSON",),
        ... )
        >>> result_error.errors
        ('Invalid JSON',)
    """
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        parsed_output=parsed_output,
    )


def list_output_formats() -> list[str]:
    """List available output formats.

    Returns:
        Sorted list of output format names.

    Examples:
        >>> formats = list_output_formats()
        >>> "json" in formats
        True
        >>> "json_schema" in formats
        True
        >>> "grammar" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_OUTPUT_FORMATS)


def list_schema_types() -> list[str]:
    """List available JSON schema types.

    Returns:
        Sorted list of schema type names.

    Examples:
        >>> types = list_schema_types()
        >>> "object" in types
        True
        >>> "string" in types
        True
        >>> "number" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_SCHEMA_TYPES)


def get_output_format(name: str) -> OutputFormat:
    """Get output format enum from string.

    Args:
        name: Format name.

    Returns:
        OutputFormat enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_output_format("json")
        <OutputFormat.JSON: 'json'>
        >>> get_output_format("json_schema")
        <OutputFormat.JSON_SCHEMA: 'json_schema'>
        >>> get_output_format("grammar")
        <OutputFormat.GRAMMAR: 'grammar'>

        >>> get_output_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid output format: invalid
    """
    for fmt in OutputFormat:
        if fmt.value == name:
            return fmt
    msg = f"invalid output format: {name}"
    raise ValueError(msg)


def get_schema_type(name: str) -> SchemaType:
    """Get schema type enum from string.

    Args:
        name: Schema type name.

    Returns:
        SchemaType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_schema_type("object")
        <SchemaType.OBJECT: 'object'>
        >>> get_schema_type("string")
        <SchemaType.STRING: 'string'>
        >>> get_schema_type("number")
        <SchemaType.NUMBER: 'number'>

        >>> get_schema_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid schema type: invalid
    """
    for schema_type in SchemaType:
        if schema_type.value == name:
            return schema_type
    msg = f"invalid schema type: {name}"
    raise ValueError(msg)


def validate_json_output(
    output: str,
    schema_config: JSONSchemaConfig | None = None,
) -> ValidationResult:
    """Validate JSON output against an optional schema.

    Args:
        output: The JSON string to validate.
        schema_config: Optional schema configuration for validation.

    Returns:
        ValidationResult with validation status and parsed output.

    Examples:
        >>> result = validate_json_output('{"name": "test"}')
        >>> result.is_valid
        True
        >>> result.parsed_output
        {'name': 'test'}

        >>> result_invalid = validate_json_output('not json')
        >>> result_invalid.is_valid
        False
        >>> len(result_invalid.errors) > 0
        True

        >>> schema_d = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> schema = create_json_schema_config(
        ...     schema_dict=schema_d,
        ...     required_fields=("name",),
        ... )
        >>> result_with_schema = validate_json_output('{"name": "test"}', schema)
        >>> result_with_schema.is_valid
        True

        >>> result_missing = validate_json_output('{"other": "value"}', schema)
        >>> result_missing.is_valid
        False
        >>> "name" in result_missing.errors[0]
        True

        >>> result_empty = validate_json_output('')
        >>> result_empty.is_valid
        False
    """
    import json

    # Handle empty input
    if not output or not output.strip():
        return create_validation_result(
            is_valid=False,
            errors=("Empty output",),
        )

    # Try to parse JSON
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        return create_validation_result(
            is_valid=False,
            errors=(f"Invalid JSON: {e}",),
        )

    # If no schema, just return parsed result
    if schema_config is None:
        return create_validation_result(
            is_valid=True,
            parsed_output=parsed,
        )

    # Validate against schema
    errors: list[str] = []

    # Check required fields
    if isinstance(parsed, dict):
        for field in schema_config.required_fields:
            if field not in parsed:
                errors.append(f"Missing required field: {field}")

        # Check additional properties
        if not schema_config.additional_properties:
            schema_properties = schema_config.schema_dict.get("properties", {})
            for key in parsed:
                if key not in schema_properties:
                    errors.append(f"Additional property not allowed: {key}")

    if errors:
        return create_validation_result(
            is_valid=False,
            errors=tuple(errors),
            parsed_output=parsed,
        )

    return create_validation_result(
        is_valid=True,
        parsed_output=parsed,
    )


def estimate_structured_tokens(
    schema_config: JSONSchemaConfig | None = None,
    grammar_config: GrammarConfig | None = None,
    choices_config: ChoicesConfig | None = None,
    base_tokens: int = 50,
) -> int:
    """Estimate tokens needed for structured output.

    This function estimates the minimum number of tokens needed
    to generate a valid structured output based on the configuration.

    Args:
        schema_config: Optional JSON schema configuration.
        grammar_config: Optional grammar configuration.
        choices_config: Optional choices configuration.
        base_tokens: Base token estimate. Defaults to 50.

    Returns:
        Estimated token count.

    Raises:
        ValueError: If base_tokens is not positive.

    Examples:
        >>> estimate_structured_tokens()
        50

        >>> schema = create_json_schema_config(
        ...     schema_dict={
        ...         "type": "object",
        ...         "properties": {
        ...             "name": {"type": "string"},
        ...             "age": {"type": "number"},
        ...         },
        ...     },
        ...     required_fields=("name", "age"),
        ... )
        >>> tokens = estimate_structured_tokens(schema_config=schema)
        >>> tokens > 50
        True

        >>> grammar = create_grammar_config(
        ...     grammar_string="root ::= 'hello'",
        ...     max_tokens=200,
        ... )
        >>> estimate_structured_tokens(grammar_config=grammar)
        200

        >>> choices = create_choices_config(choices=("yes", "no"))
        >>> estimate_structured_tokens(choices_config=choices)
        10

        >>> estimate_structured_tokens(base_tokens=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: base_tokens must be positive
    """
    if base_tokens <= 0:
        msg = f"base_tokens must be positive, got {base_tokens}"
        raise ValueError(msg)

    # Grammar config has explicit max_tokens
    if grammar_config is not None:
        return grammar_config.max_tokens

    # Choices are typically short
    if choices_config is not None:
        max_choice_len = max(len(c) for c in choices_config.choices)
        if choices_config.allow_multiple:
            # Estimate for multiple choices
            return max_choice_len * len(choices_config.choices) + 10
        return max(10, max_choice_len // 4 + 5)

    # Schema-based estimation
    if schema_config is not None:
        tokens = base_tokens
        properties = schema_config.schema_dict.get("properties", {})

        # Add tokens per property
        tokens += len(properties) * 15  # ~15 tokens per field

        # Add tokens for required fields (more content expected)
        tokens += len(schema_config.required_fields) * 10

        return tokens

    return base_tokens


def get_recommended_structured_config(task: str) -> StructuredConfig:
    """Get recommended structured config for a task.

    Args:
        task: Task type ("extraction", "classification", "qa", "generation").

    Returns:
        Recommended StructuredConfig for the task.

    Raises:
        ValueError: If task is invalid.

    Examples:
        >>> config = get_recommended_structured_config("extraction")
        >>> config.output_format
        <OutputFormat.JSON_SCHEMA: 'json_schema'>
        >>> config.strict_mode
        True

        >>> config = get_recommended_structured_config("classification")
        >>> config.output_format
        <OutputFormat.CHOICES: 'choices'>

        >>> config = get_recommended_structured_config("qa")
        >>> config.output_format
        <OutputFormat.JSON: 'json'>

        >>> get_recommended_structured_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of
    """
    valid_tasks = {"extraction", "classification", "qa", "generation"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    configs = {
        "extraction": create_structured_config(
            output_format=OutputFormat.JSON_SCHEMA,
            schema={"type": "object"},
            strict_mode=True,
            max_retries=3,
        ),
        "classification": create_structured_config(
            output_format=OutputFormat.CHOICES,
            strict_mode=True,
            max_retries=2,
        ),
        "qa": create_structured_config(
            output_format=OutputFormat.JSON,
            strict_mode=False,
            max_retries=1,
        ),
        "generation": create_structured_config(
            output_format=OutputFormat.GRAMMAR,
            strict_mode=False,
            max_retries=2,
        ),
    }
    return configs[task]


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result for display.

    Args:
        result: Result to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> result = create_validation_result(
        ...     is_valid=True,
        ...     parsed_output={"name": "test"},
        ... )
        >>> formatted = format_validation_result(result)
        >>> "Valid: True" in formatted
        True

        >>> result_error = create_validation_result(
        ...     is_valid=False,
        ...     errors=("Missing field",),
        ... )
        >>> formatted = format_validation_result(result_error)
        >>> "Valid: False" in formatted
        True
        >>> "Missing field" in formatted
        True

        >>> format_validation_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    validate_not_none(result, "result")

    lines = [f"Valid: {result.is_valid}"]

    if result.errors:
        lines.append("Errors:")
        for error in result.errors:
            lines.append(f"  - {error}")

    if result.parsed_output is not None:
        lines.append(f"Parsed: {result.parsed_output}")

    return "\n".join(lines)
