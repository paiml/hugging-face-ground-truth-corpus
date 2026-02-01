"""Function calling and tool use utilities for text generation.

This module provides utilities for configuring and working with
function calling and tool use in LLM text generation, supporting
multiple provider formats (OpenAI, Anthropic, etc.).

Examples:
    >>> from hf_gtc.generation.tools import create_tool_definition
    >>> tool = create_tool_definition(
    ...     name="get_weather",
    ...     description="Get current weather for a location",
    ...     parameters={"location": "string", "units": "string"},
    ...     required=("location",),
    ... )
    >>> tool.name
    'get_weather'
    >>> len(tool.required)
    1
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class ToolFormat(Enum):
    """Supported tool definition formats.

    Attributes:
        OPENAI: OpenAI function calling format.
        ANTHROPIC: Anthropic tool use format.
        JSON_SCHEMA: Generic JSON schema format.
        XML: XML-based tool format.

    Examples:
        >>> ToolFormat.OPENAI.value
        'openai'
        >>> ToolFormat.ANTHROPIC.value
        'anthropic'
        >>> ToolFormat.JSON_SCHEMA.value
        'json_schema'
        >>> ToolFormat.XML.value
        'xml'
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    JSON_SCHEMA = "json_schema"
    XML = "xml"


class ParsingStrategy(Enum):
    """Tool call parsing strategies.

    Attributes:
        JSON: Parse tool calls as JSON.
        REGEX: Parse tool calls using regex patterns.
        STRUCTURED: Use structured output parsing.
        LLM: Use LLM to parse tool calls.

    Examples:
        >>> ParsingStrategy.JSON.value
        'json'
        >>> ParsingStrategy.REGEX.value
        'regex'
        >>> ParsingStrategy.STRUCTURED.value
        'structured'
        >>> ParsingStrategy.LLM.value
        'llm'
    """

    JSON = "json"
    REGEX = "regex"
    STRUCTURED = "structured"
    LLM = "llm"


class ExecutionMode(Enum):
    """Tool execution modes.

    Attributes:
        SEQUENTIAL: Execute tools one at a time in order.
        PARALLEL: Execute tools concurrently.
        AUTO: Automatically determine execution mode based on dependencies.

    Examples:
        >>> ExecutionMode.SEQUENTIAL.value
        'sequential'
        >>> ExecutionMode.PARALLEL.value
        'parallel'
        >>> ExecutionMode.AUTO.value
        'auto'
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    AUTO = "auto"


VALID_TOOL_FORMATS = frozenset(f.value for f in ToolFormat)
VALID_PARSING_STRATEGIES = frozenset(s.value for s in ParsingStrategy)
VALID_EXECUTION_MODES = frozenset(m.value for m in ExecutionMode)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Definition of a callable tool.

    Attributes:
        name: Name of the tool (alphanumeric and underscores).
        description: Human-readable description of what the tool does.
        parameters: Dictionary mapping parameter names to their types/schemas.
        required: Tuple of required parameter names.

    Examples:
        >>> tool = ToolDefinition(
        ...     name="search",
        ...     description="Search the web",
        ...     parameters={"query": "string", "max_results": "integer"},
        ...     required=("query",),
        ... )
        >>> tool.name
        'search'
        >>> tool.description
        'Search the web'
        >>> "query" in tool.parameters
        True
        >>> "query" in tool.required
        True
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Represents a tool call from model output.

    Attributes:
        name: Name of the tool being called.
        arguments: Dictionary of arguments for the tool.
        id: Unique identifier for this tool call.

    Examples:
        >>> call = ToolCall(
        ...     name="get_weather",
        ...     arguments={"location": "New York", "units": "celsius"},
        ...     id="call_001",
        ... )
        >>> call.name
        'get_weather'
        >>> call.arguments["location"]
        'New York'
        >>> call.id
        'call_001'
    """

    name: str
    arguments: dict[str, Any]
    id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result from executing a tool.

    Attributes:
        call_id: ID of the tool call this result corresponds to.
        output: Output from the tool execution.
        error: Error message if execution failed, None otherwise.

    Examples:
        >>> result = ToolResult(
        ...     call_id="call_001",
        ...     output="Temperature: 72F, Sunny",
        ...     error=None,
        ... )
        >>> result.call_id
        'call_001'
        >>> result.output
        'Temperature: 72F, Sunny'
        >>> result.error is None
        True

        >>> error_result = ToolResult(
        ...     call_id="call_002",
        ...     output="",
        ...     error="Location not found",
        ... )
        >>> error_result.error
        'Location not found'
    """

    call_id: str
    output: str
    error: str | None


@dataclass(frozen=True, slots=True)
class ToolConfig:
    """Configuration for tool use in generation.

    Attributes:
        format: Format for tool definitions.
        definitions: Tuple of tool definitions.
        parsing_strategy: Strategy for parsing tool calls.
        execution_mode: Mode for executing multiple tools.
        max_calls: Maximum tool calls per generation.

    Examples:
        >>> config = ToolConfig(
        ...     format=ToolFormat.OPENAI,
        ...     definitions=(),
        ...     parsing_strategy=ParsingStrategy.JSON,
        ...     execution_mode=ExecutionMode.SEQUENTIAL,
        ...     max_calls=10,
        ... )
        >>> config.format
        <ToolFormat.OPENAI: 'openai'>
        >>> config.max_calls
        10
    """

    format: ToolFormat
    definitions: tuple[ToolDefinition, ...]
    parsing_strategy: ParsingStrategy
    execution_mode: ExecutionMode
    max_calls: int


@dataclass(frozen=True, slots=True)
class ToolStats:
    """Statistics about tool execution.

    Attributes:
        total_calls: Total number of tool calls.
        successful_calls: Number of successful calls.
        failed_calls: Number of failed calls.
        total_time_ms: Total execution time in milliseconds.
        calls_by_tool: Dictionary mapping tool names to call counts.

    Examples:
        >>> stats = ToolStats(
        ...     total_calls=5,
        ...     successful_calls=4,
        ...     failed_calls=1,
        ...     total_time_ms=150.5,
        ...     calls_by_tool={"search": 3, "calculate": 2},
        ... )
        >>> stats.total_calls
        5
        >>> stats.successful_calls
        4
    """

    total_calls: int
    successful_calls: int
    failed_calls: int
    total_time_ms: float
    calls_by_tool: dict[str, int]


def validate_tool_definition(tool: ToolDefinition) -> None:
    """Validate a tool definition.

    Args:
        tool: Tool definition to validate.

    Raises:
        ValueError: If tool definition is None.
        ValueError: If name is empty or invalid format.
        ValueError: If description is empty.
        ValueError: If required parameters not in parameters dict.

    Examples:
        >>> tool = ToolDefinition("search", "Search web", {"q": "string"}, ("q",))
        >>> validate_tool_definition(tool)  # No error

        >>> validate_tool_definition(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tool cannot be None

        >>> bad = ToolDefinition("", "desc", {}, ())
        >>> validate_tool_definition(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty

        >>> bad_name = ToolDefinition("bad-name", "desc", {}, ())
        >>> validate_tool_definition(bad_name)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name must be alphanumeric with underscores

        >>> bad_req = ToolDefinition("search", "desc", {}, ("missing",))
        >>> validate_tool_definition(bad_req)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: required parameter 'missing' not in parameters
    """
    if tool is None:
        msg = "tool cannot be None"
        raise ValueError(msg)

    if not tool.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if not tool.name.replace("_", "").isalnum():
        msg = f"name must be alphanumeric with underscores, got '{tool.name}'"
        raise ValueError(msg)

    if not tool.description:
        msg = "description cannot be empty"
        raise ValueError(msg)

    for req in tool.required:
        if req not in tool.parameters:
            msg = f"required parameter '{req}' not in parameters"
            raise ValueError(msg)


def validate_tool_call(call: ToolCall) -> None:
    """Validate a tool call.

    Args:
        call: Tool call to validate.

    Raises:
        ValueError: If call is None.
        ValueError: If name is empty.
        ValueError: If id is empty.

    Examples:
        >>> call = ToolCall("search", {"q": "test"}, "call_001")
        >>> validate_tool_call(call)  # No error

        >>> validate_tool_call(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: call cannot be None

        >>> bad = ToolCall("", {"q": "test"}, "call_001")
        >>> validate_tool_call(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if call is None:
        msg = "call cannot be None"
        raise ValueError(msg)

    if not call.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if not call.id:
        msg = "id cannot be empty"
        raise ValueError(msg)


def validate_tool_result(result: ToolResult) -> None:
    """Validate a tool result.

    Args:
        result: Tool result to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If call_id is empty.

    Examples:
        >>> result = ToolResult("call_001", "output", None)
        >>> validate_tool_result(result)  # No error

        >>> validate_tool_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = ToolResult("", "output", None)
        >>> validate_tool_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: call_id cannot be empty
    """
    validate_not_none(result, "result")

    if not result.call_id:
        msg = "call_id cannot be empty"
        raise ValueError(msg)


def validate_tool_config(config: ToolConfig) -> None:
    """Validate a tool configuration.

    Args:
        config: Tool configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_calls is not positive.
        ValueError: If any tool definition is invalid.

    Examples:
        >>> config = ToolConfig(
        ...     ToolFormat.OPENAI, (), ParsingStrategy.JSON,
        ...     ExecutionMode.SEQUENTIAL, 10
        ... )
        >>> validate_tool_config(config)  # No error

        >>> validate_tool_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ToolConfig(
        ...     ToolFormat.OPENAI, (), ParsingStrategy.JSON,
        ...     ExecutionMode.SEQUENTIAL, 0
        ... )
        >>> validate_tool_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_calls must be positive
    """
    validate_not_none(config, "config")

    if config.max_calls <= 0:
        msg = f"max_calls must be positive, got {config.max_calls}"
        raise ValueError(msg)

    for tool in config.definitions:
        validate_tool_definition(tool)


def create_tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    required: tuple[str, ...] = (),
) -> ToolDefinition:
    """Create a tool definition.

    Args:
        name: Name of the tool.
        description: Description of what the tool does.
        parameters: Parameter schema dictionary. Defaults to empty dict.
        required: Tuple of required parameter names. Defaults to ().

    Returns:
        Validated ToolDefinition instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> tool = create_tool_definition(
        ...     name="get_weather",
        ...     description="Get weather for a location",
        ...     parameters={"location": "string"},
        ...     required=("location",),
        ... )
        >>> tool.name
        'get_weather'
        >>> tool.required
        ('location',)

        >>> tool = create_tool_definition("search", "Search the web")
        >>> tool.parameters
        {}

        >>> create_tool_definition("", "desc")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    tool = ToolDefinition(
        name=name,
        description=description,
        parameters=parameters if parameters is not None else {},
        required=required,
    )
    validate_tool_definition(tool)
    return tool


def create_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    id: str = "",
) -> ToolCall:
    """Create a tool call.

    Args:
        name: Name of the tool to call.
        arguments: Arguments for the tool. Defaults to empty dict.
        id: Unique identifier. Auto-generated if empty.

    Returns:
        Validated ToolCall instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> call = create_tool_call(
        ...     name="search",
        ...     arguments={"query": "python"},
        ...     id="call_001",
        ... )
        >>> call.name
        'search'
        >>> call.arguments["query"]
        'python'

        >>> call = create_tool_call("search")
        >>> call.id.startswith("call_")
        True

        >>> create_tool_call("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    call_id = id if id else f"call_{abs(hash(name)) % 100000:05d}"
    call = ToolCall(
        name=name,
        arguments=arguments if arguments is not None else {},
        id=call_id,
    )
    validate_tool_call(call)
    return call


def create_tool_result(
    call_id: str,
    output: str = "",
    error: str | None = None,
) -> ToolResult:
    """Create a tool result.

    Args:
        call_id: ID of the tool call.
        output: Output from execution. Defaults to "".
        error: Error message if failed. Defaults to None.

    Returns:
        Validated ToolResult instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_tool_result(
        ...     call_id="call_001",
        ...     output="Success!",
        ... )
        >>> result.output
        'Success!'
        >>> result.error is None
        True

        >>> result = create_tool_result(
        ...     call_id="call_002",
        ...     error="Not found",
        ... )
        >>> result.error
        'Not found'

        >>> create_tool_result("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: call_id cannot be empty
    """
    result = ToolResult(
        call_id=call_id,
        output=output,
        error=error,
    )
    validate_tool_result(result)
    return result


def create_tool_config(
    format: ToolFormat | str = ToolFormat.OPENAI,
    definitions: tuple[ToolDefinition, ...] = (),
    parsing_strategy: ParsingStrategy | str = ParsingStrategy.JSON,
    execution_mode: ExecutionMode | str = ExecutionMode.SEQUENTIAL,
    max_calls: int = 10,
) -> ToolConfig:
    """Create a tool configuration.

    Args:
        format: Tool definition format. Defaults to OPENAI.
        definitions: Tuple of tool definitions. Defaults to ().
        parsing_strategy: Parsing strategy. Defaults to JSON.
        execution_mode: Execution mode. Defaults to SEQUENTIAL.
        max_calls: Maximum tool calls. Defaults to 10.

    Returns:
        Validated ToolConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_tool_config()
        >>> config.format
        <ToolFormat.OPENAI: 'openai'>
        >>> config.max_calls
        10

        >>> config = create_tool_config(format="anthropic", max_calls=5)
        >>> config.format
        <ToolFormat.ANTHROPIC: 'anthropic'>
        >>> config.max_calls
        5

        >>> create_tool_config(max_calls=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_calls must be positive
    """
    if isinstance(format, str):
        format = get_tool_format(format)
    if isinstance(parsing_strategy, str):
        parsing_strategy = get_parsing_strategy(parsing_strategy)
    if isinstance(execution_mode, str):
        execution_mode = get_execution_mode(execution_mode)

    config = ToolConfig(
        format=format,
        definitions=definitions,
        parsing_strategy=parsing_strategy,
        execution_mode=execution_mode,
        max_calls=max_calls,
    )
    validate_tool_config(config)
    return config


def create_tool_stats(
    total_calls: int = 0,
    successful_calls: int = 0,
    failed_calls: int = 0,
    total_time_ms: float = 0.0,
    calls_by_tool: dict[str, int] | None = None,
) -> ToolStats:
    """Create tool statistics.

    Args:
        total_calls: Total calls made. Defaults to 0.
        successful_calls: Successful calls. Defaults to 0.
        failed_calls: Failed calls. Defaults to 0.
        total_time_ms: Total time in ms. Defaults to 0.0.
        calls_by_tool: Calls per tool. Defaults to empty dict.

    Returns:
        ToolStats instance.

    Raises:
        ValueError: If counts are negative.

    Examples:
        >>> stats = create_tool_stats(total_calls=5, successful_calls=4, failed_calls=1)
        >>> stats.total_calls
        5
        >>> stats.successful_calls
        4

        >>> create_tool_stats(total_calls=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_calls must be non-negative
    """
    if total_calls < 0:
        msg = f"total_calls must be non-negative, got {total_calls}"
        raise ValueError(msg)

    if successful_calls < 0:
        msg = f"successful_calls must be non-negative, got {successful_calls}"
        raise ValueError(msg)

    if failed_calls < 0:
        msg = f"failed_calls must be non-negative, got {failed_calls}"
        raise ValueError(msg)

    if total_time_ms < 0:
        msg = f"total_time_ms must be non-negative, got {total_time_ms}"
        raise ValueError(msg)

    return ToolStats(
        total_calls=total_calls,
        successful_calls=successful_calls,
        failed_calls=failed_calls,
        total_time_ms=total_time_ms,
        calls_by_tool=calls_by_tool if calls_by_tool is not None else {},
    )


def list_tool_formats() -> list[str]:
    """List available tool definition formats.

    Returns:
        Sorted list of format names.

    Examples:
        >>> formats = list_tool_formats()
        >>> "openai" in formats
        True
        >>> "anthropic" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_TOOL_FORMATS)


def list_parsing_strategies() -> list[str]:
    """List available parsing strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_parsing_strategies()
        >>> "json" in strategies
        True
        >>> "regex" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_PARSING_STRATEGIES)


def list_execution_modes() -> list[str]:
    """List available execution modes.

    Returns:
        Sorted list of mode names.

    Examples:
        >>> modes = list_execution_modes()
        >>> "sequential" in modes
        True
        >>> "parallel" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_EXECUTION_MODES)


def get_tool_format(name: str) -> ToolFormat:
    """Get tool format enum from name.

    Args:
        name: Format name.

    Returns:
        ToolFormat enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_tool_format("openai")
        <ToolFormat.OPENAI: 'openai'>
        >>> get_tool_format("anthropic")
        <ToolFormat.ANTHROPIC: 'anthropic'>

        >>> get_tool_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid tool format: invalid
    """
    if name not in VALID_TOOL_FORMATS:
        msg = f"invalid tool format: {name}"
        raise ValueError(msg)
    return ToolFormat(name)


def get_parsing_strategy(name: str) -> ParsingStrategy:
    """Get parsing strategy enum from name.

    Args:
        name: Strategy name.

    Returns:
        ParsingStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_parsing_strategy("json")
        <ParsingStrategy.JSON: 'json'>
        >>> get_parsing_strategy("regex")
        <ParsingStrategy.REGEX: 'regex'>

        >>> get_parsing_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid parsing strategy: invalid
    """
    if name not in VALID_PARSING_STRATEGIES:
        msg = f"invalid parsing strategy: {name}"
        raise ValueError(msg)
    return ParsingStrategy(name)


def get_execution_mode(name: str) -> ExecutionMode:
    """Get execution mode enum from name.

    Args:
        name: Mode name.

    Returns:
        ExecutionMode enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_execution_mode("sequential")
        <ExecutionMode.SEQUENTIAL: 'sequential'>
        >>> get_execution_mode("parallel")
        <ExecutionMode.PARALLEL: 'parallel'>

        >>> get_execution_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid execution mode: invalid
    """
    if name not in VALID_EXECUTION_MODES:
        msg = f"invalid execution mode: {name}"
        raise ValueError(msg)
    return ExecutionMode(name)


def format_tools_for_prompt(
    tools: tuple[ToolDefinition, ...],
    format: ToolFormat = ToolFormat.OPENAI,
) -> str:
    """Format tool definitions for inclusion in a prompt.

    Args:
        tools: Tuple of tool definitions.
        format: Output format to use.

    Returns:
        Formatted string representation of tools.

    Raises:
        ValueError: If tools is None.

    Examples:
        >>> tool = create_tool_definition(
        ...     name="search",
        ...     description="Search the web",
        ...     parameters={"query": "string"},
        ...     required=("query",),
        ... )
        >>> result = format_tools_for_prompt((tool,), ToolFormat.OPENAI)
        >>> "search" in result
        True
        >>> "function" in result
        True

        >>> result = format_tools_for_prompt((tool,), ToolFormat.XML)
        >>> "<tool" in result
        True

        >>> format_tools_for_prompt(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tools cannot be None
    """
    if tools is None:
        msg = "tools cannot be None"
        raise ValueError(msg)

    if not tools:
        return ""

    if format == ToolFormat.OPENAI:
        return _format_tools_openai(tools)
    elif format == ToolFormat.ANTHROPIC:
        return _format_tools_anthropic(tools)
    elif format == ToolFormat.JSON_SCHEMA:
        return _format_tools_json_schema(tools)
    elif format == ToolFormat.XML:
        return _format_tools_xml(tools)
    else:
        return _format_tools_openai(tools)


def _format_tools_openai(tools: tuple[ToolDefinition, ...]) -> str:
    """Format tools in OpenAI function calling format."""
    formatted = []
    for tool in tools:
        tool_obj = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": list(tool.required),
                },
            },
        }
        formatted.append(tool_obj)
    return json.dumps(formatted, indent=2)


def _format_tools_anthropic(tools: tuple[ToolDefinition, ...]) -> str:
    """Format tools in Anthropic tool use format."""
    formatted = []
    for tool in tools:
        tool_obj = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": tool.parameters,
                "required": list(tool.required),
            },
        }
        formatted.append(tool_obj)
    return json.dumps(formatted, indent=2)


def _format_tools_json_schema(tools: tuple[ToolDefinition, ...]) -> str:
    """Format tools as generic JSON schema."""
    formatted = []
    for tool in tools:
        schema = {
            "name": tool.name,
            "description": tool.description,
            "schema": {
                "type": "object",
                "properties": tool.parameters,
                "required": list(tool.required),
            },
        }
        formatted.append(schema)
    return json.dumps(formatted, indent=2)


def _format_tools_xml(tools: tuple[ToolDefinition, ...]) -> str:
    """Format tools as XML."""
    lines = ["<tools>"]
    for tool in tools:
        lines.append(f'  <tool name="{tool.name}">')
        lines.append(f"    <description>{tool.description}</description>")
        lines.append("    <parameters>")
        for param, ptype in tool.parameters.items():
            req_str = "true" if param in tool.required else "false"
            lines.append(
                f'      <param name="{param}" type="{ptype}" required="{req_str}"/>'
            )
        lines.append("    </parameters>")
        lines.append("  </tool>")
    lines.append("</tools>")
    return "\n".join(lines)


def parse_tool_calls(
    text: str,
    available_tools: tuple[str, ...],
    strategy: ParsingStrategy = ParsingStrategy.JSON,
) -> tuple[ToolCall, ...]:
    """Parse tool calls from model output text.

    Args:
        text: Model output text potentially containing tool calls.
        available_tools: Tuple of valid tool names.
        strategy: Parsing strategy to use.

    Returns:
        Tuple of parsed ToolCall objects.

    Examples:
        >>> text = '{"name": "search", "arguments": {"query": "python"}}'
        >>> calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        >>> len(calls)
        1
        >>> calls[0].name
        'search'

        >>> text = "search(query='test')"
        >>> calls = parse_tool_calls(text, ("search",), ParsingStrategy.REGEX)
        >>> len(calls)
        1
        >>> calls[0].name
        'search'

        >>> parse_tool_calls("no tools here", ("search",))
        ()
    """
    if strategy == ParsingStrategy.JSON:
        return _parse_tool_calls_json(text, available_tools)
    elif strategy == ParsingStrategy.REGEX:
        return _parse_tool_calls_regex(text, available_tools)
    elif strategy == ParsingStrategy.STRUCTURED:
        return _parse_tool_calls_structured(text, available_tools)
    else:
        return _parse_tool_calls_json(text, available_tools)


def _extract_json_objects(text: str) -> list[str]:
    """Extract potential JSON objects from text using bracket matching."""
    objects = []
    i = 0

    while i < len(text):
        if text[i] == "{":
            obj_str, i = _match_json_object(text, i)
            if obj_str is not None and '"name"' in obj_str:
                objects.append(obj_str)
        else:
            i += 1

    return objects


def _match_json_object(text: str, start: int) -> tuple[str | None, int]:
    """Match a JSON object starting at the given position.

    Returns:
        Tuple of (matched_string_or_None, next_position).
    """
    depth = 1
    i = start + 1
    in_string = False
    escape_next = False

    while i < len(text) and depth > 0:
        char = text[i]
        if escape_next:
            escape_next = False
        elif char == "\\":
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
        i += 1

    if depth == 0:
        return text[start:i], i
    return None, i


def _parse_tool_calls_json(
    text: str, available_tools: tuple[str, ...]
) -> tuple[ToolCall, ...]:
    """Parse tool calls using JSON parsing."""
    calls = []

    # Try to find JSON objects in the text using bracket matching
    potential_jsons = _extract_json_objects(text)

    for potential_json in potential_jsons:
        try:
            obj = json.loads(potential_json)
            if isinstance(obj, dict) and "name" in obj:
                name = obj.get("name", "")
                if name in available_tools:
                    arguments = obj.get("arguments", {})
                    call_id = obj.get("id", f"call_{len(calls):05d}")
                    calls.append(create_tool_call(name, arguments, call_id))
        except json.JSONDecodeError:
            continue

    # Also try to parse the entire text as JSON
    if not calls:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "name" in data:
                name = data.get("name", "")
                if name in available_tools:
                    arguments = data.get("arguments", {})
                    call_id = data.get("id", "call_00000")
                    calls.append(create_tool_call(name, arguments, call_id))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        name = item.get("name", "")
                        if name in available_tools:
                            arguments = item.get("arguments", {})
                            call_id = item.get("id", f"call_{len(calls):05d}")
                            calls.append(create_tool_call(name, arguments, call_id))
        except json.JSONDecodeError:
            pass

    return tuple(calls)


def _parse_tool_calls_regex(
    text: str, available_tools: tuple[str, ...]
) -> tuple[ToolCall, ...]:
    """Parse tool calls using regex patterns."""
    calls = []

    for tool_name in available_tools:
        # Pattern: tool_name(arg1='value1', arg2='value2')
        pattern = rf"{tool_name}\s*\(([^)]*)\)"
        matches = re.findall(pattern, text)

        for match in matches:
            arguments: dict[str, Any] = {}
            # Parse key='value' or key="value" pairs
            arg_pattern = r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]"
            arg_matches = re.findall(arg_pattern, match)
            for key, value in arg_matches:
                arguments[key] = value

            # Also try key=number pattern
            num_pattern = r"(\w+)\s*=\s*(\d+(?:\.\d+)?)"
            num_matches = re.findall(num_pattern, match)
            for key, value in num_matches:
                if key not in arguments:
                    try:
                        arguments[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        arguments[key] = value

            call_id = f"call_{len(calls):05d}"
            calls.append(create_tool_call(tool_name, arguments, call_id))

    return tuple(calls)


def _parse_tool_calls_structured(
    text: str, available_tools: tuple[str, ...]
) -> tuple[ToolCall, ...]:
    """Parse tool calls from structured output."""
    # First try JSON parsing
    calls = _parse_tool_calls_json(text, available_tools)
    if calls:
        return calls

    # Fall back to regex parsing
    return _parse_tool_calls_regex(text, available_tools)


def validate_tool_arguments(
    call: ToolCall,
    tool: ToolDefinition,
) -> tuple[bool, tuple[str, ...]]:
    """Validate tool call arguments against tool definition.

    Args:
        call: Tool call to validate.
        tool: Tool definition to validate against.

    Returns:
        Tuple of (is_valid, error_messages).

    Raises:
        ValueError: If call or tool is None.
        ValueError: If tool name doesn't match call name.

    Examples:
        >>> tool = create_tool_definition(
        ...     name="search",
        ...     description="Search",
        ...     parameters={"query": "string"},
        ...     required=("query",),
        ... )
        >>> call = create_tool_call("search", {"query": "test"}, "call_001")
        >>> is_valid, errors = validate_tool_arguments(call, tool)
        >>> is_valid
        True
        >>> errors
        ()

        >>> bad_call = create_tool_call("search", {}, "call_002")
        >>> is_valid, errors = validate_tool_arguments(bad_call, tool)
        >>> is_valid
        False
        >>> "query" in errors[0]
        True

        >>> validate_tool_arguments(None, tool)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: call cannot be None
    """
    if call is None:
        msg = "call cannot be None"
        raise ValueError(msg)

    if tool is None:
        msg = "tool cannot be None"
        raise ValueError(msg)

    if call.name != tool.name:
        msg = f"tool name mismatch: call has '{call.name}', tool has '{tool.name}'"
        raise ValueError(msg)

    errors: list[str] = []

    # Check required parameters
    for req in tool.required:
        if req not in call.arguments:
            errors.append(f"missing required parameter: {req}")

    # Check for unknown parameters
    for param in call.arguments:
        if param not in tool.parameters:
            errors.append(f"unknown parameter: {param}")

    return len(errors) == 0, tuple(errors)


def execute_tool_calls(
    calls: tuple[ToolCall, ...],
    tool_handlers: dict[str, Callable[..., str]],
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
) -> tuple[ToolResult, ...]:
    """Execute tool calls using provided handlers.

    Args:
        calls: Tuple of tool calls to execute.
        tool_handlers: Dictionary mapping tool names to handler functions.
        mode: Execution mode. PARALLEL mode falls back to SEQUENTIAL.

    Returns:
        Tuple of ToolResult objects.

    Examples:
        >>> def search_handler(query: str) -> str:
        ...     return f"Results for: {query}"
        >>> handlers = {"search": search_handler}
        >>> call = create_tool_call("search", {"query": "python"}, "call_001")
        >>> results = execute_tool_calls((call,), handlers)
        >>> len(results)
        1
        >>> "Results for: python" in results[0].output
        True

        >>> call = create_tool_call("unknown", {}, "call_002")
        >>> results = execute_tool_calls((call,), handlers)
        >>> results[0].error is not None
        True
    """
    results = []

    for call in calls:
        if call.name not in tool_handlers:
            results.append(
                create_tool_result(
                    call_id=call.id,
                    output="",
                    error=f"No handler for tool: {call.name}",
                )
            )
            continue

        handler = tool_handlers[call.name]
        try:
            output = handler(**call.arguments)
            results.append(
                create_tool_result(
                    call_id=call.id,
                    output=str(output),
                    error=None,
                )
            )
        except Exception as e:
            results.append(
                create_tool_result(
                    call_id=call.id,
                    output="",
                    error=str(e),
                )
            )

    return tuple(results)


def format_tool_stats(stats: ToolStats) -> str:
    """Format tool statistics for display.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = create_tool_stats(
        ...     total_calls=5,
        ...     successful_calls=4,
        ...     failed_calls=1,
        ...     total_time_ms=150.5,
        ...     calls_by_tool={"search": 3, "calc": 2},
        ... )
        >>> result = format_tool_stats(stats)
        >>> "Total calls: 5" in result
        True
        >>> "Successful: 4" in result
        True
        >>> "Failed: 1" in result
        True

        >>> format_tool_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        f"Total calls: {stats.total_calls}",
        f"Successful: {stats.successful_calls}",
        f"Failed: {stats.failed_calls}",
        f"Total time: {stats.total_time_ms:.1f}ms",
    ]

    if stats.calls_by_tool:
        lines.append("Calls by tool:")
        for tool, count in sorted(stats.calls_by_tool.items()):
            lines.append(f"  {tool}: {count}")

    return "\n".join(lines)


def get_recommended_tool_config(task: str) -> ToolConfig:
    """Get recommended tool configuration for a task.

    Args:
        task: Task type ("agent", "assistant", "extraction", "code").

    Returns:
        Recommended ToolConfig for the task.

    Raises:
        ValueError: If task is invalid.

    Examples:
        >>> config = get_recommended_tool_config("agent")
        >>> config.execution_mode
        <ExecutionMode.AUTO: 'auto'>
        >>> config.max_calls
        20

        >>> config = get_recommended_tool_config("assistant")
        >>> config.execution_mode
        <ExecutionMode.SEQUENTIAL: 'sequential'>

        >>> config = get_recommended_tool_config("code")
        >>> config.format
        <ToolFormat.JSON_SCHEMA: 'json_schema'>

        >>> get_recommended_tool_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of
    """
    valid_tasks = {"agent", "assistant", "extraction", "code"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    configs = {
        "agent": create_tool_config(
            format=ToolFormat.OPENAI,
            parsing_strategy=ParsingStrategy.JSON,
            execution_mode=ExecutionMode.AUTO,
            max_calls=20,
        ),
        "assistant": create_tool_config(
            format=ToolFormat.OPENAI,
            parsing_strategy=ParsingStrategy.JSON,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_calls=10,
        ),
        "extraction": create_tool_config(
            format=ToolFormat.JSON_SCHEMA,
            parsing_strategy=ParsingStrategy.STRUCTURED,
            execution_mode=ExecutionMode.PARALLEL,
            max_calls=5,
        ),
        "code": create_tool_config(
            format=ToolFormat.JSON_SCHEMA,
            parsing_strategy=ParsingStrategy.JSON,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_calls=15,
        ),
    }
    return configs[task]
