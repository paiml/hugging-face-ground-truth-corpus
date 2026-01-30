"""Agent tool use utilities.

This module provides functions for defining tools, parsing tool calls,
and building agent configurations.

Examples:
    >>> from hf_gtc.agents.tools import create_tool_definition
    >>> tool = create_tool_definition("search", "Search the web")
    >>> tool.name
    'search'
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ToolType(Enum):
    """Supported tool types.

    Attributes:
        FUNCTION: Function-calling tool.
        CODE_INTERPRETER: Code execution tool.
        RETRIEVAL: Document retrieval tool.
        WEB_SEARCH: Web search tool.
        API: External API tool.

    Examples:
        >>> ToolType.FUNCTION.value
        'function'
        >>> ToolType.CODE_INTERPRETER.value
        'code_interpreter'
    """

    FUNCTION = "function"
    CODE_INTERPRETER = "code_interpreter"
    RETRIEVAL = "retrieval"
    WEB_SEARCH = "web_search"
    API = "api"


VALID_TOOL_TYPES = frozenset(t.value for t in ToolType)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Definition of a tool.

    Attributes:
        name: Tool name.
        description: Tool description.
        tool_type: Type of tool.
        parameters: Parameter schema (JSON Schema style).

    Examples:
        >>> tool = ToolDefinition(
        ...     name="calculator",
        ...     description="Perform calculations",
        ...     tool_type=ToolType.FUNCTION,
        ...     parameters={"expression": "string"},
        ... )
        >>> tool.name
        'calculator'
    """

    name: str
    description: str
    tool_type: ToolType
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        tool_name: Name of the tool to call.
        arguments: Arguments for the tool.
        call_id: Unique identifier for this call.

    Examples:
        >>> call = ToolCall(
        ...     tool_name="search",
        ...     arguments={"query": "weather"},
        ...     call_id="call_123",
        ... )
        >>> call.tool_name
        'search'
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result from a tool execution.

    Attributes:
        call_id: ID of the tool call.
        output: Tool output.
        error: Error message if failed.
        success: Whether execution succeeded.

    Examples:
        >>> result = ToolResult(
        ...     call_id="call_123",
        ...     output="The weather is sunny",
        ...     error=None,
        ...     success=True,
        ... )
        >>> result.success
        True
    """

    call_id: str
    output: str
    error: str | None
    success: bool


@dataclass(frozen=True, slots=True)
class AgentStep:
    """Represents one step in agent execution.

    Attributes:
        thought: Agent's reasoning.
        action: Action taken.
        observation: Result of action.
        step_number: Step number.

    Examples:
        >>> step = AgentStep(
        ...     thought="I need to search for info",
        ...     action="search(query='weather')",
        ...     observation="Weather is sunny",
        ...     step_number=1,
        ... )
        >>> step.step_number
        1
    """

    thought: str
    action: str
    observation: str
    step_number: int


@dataclass(frozen=True, slots=True)
class AgentState:
    """Current state of an agent.

    Attributes:
        steps: Tuple of completed steps.
        current_step: Current step number.
        is_complete: Whether agent is done.
        final_answer: Final answer if complete.

    Examples:
        >>> state = AgentState(
        ...     steps=(),
        ...     current_step=0,
        ...     is_complete=False,
        ...     final_answer=None,
        ... )
        >>> state.is_complete
        False
    """

    steps: tuple[AgentStep, ...]
    current_step: int
    is_complete: bool
    final_answer: str | None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configuration for an agent.

    Attributes:
        max_steps: Maximum steps before stopping.
        tools: Tuple of available tools.
        system_prompt: System prompt for the agent.
        temperature: Sampling temperature.

    Examples:
        >>> config = AgentConfig(
        ...     max_steps=10,
        ...     tools=(),
        ...     system_prompt="You are a helpful assistant.",
        ...     temperature=0.7,
        ... )
        >>> config.max_steps
        10
    """

    max_steps: int
    tools: tuple[ToolDefinition, ...]
    system_prompt: str
    temperature: float


@dataclass(frozen=True, slots=True)
class ReActConfig:
    """Configuration for ReAct agent pattern.

    Attributes:
        thought_prefix: Prefix for thought section.
        action_prefix: Prefix for action section.
        observation_prefix: Prefix for observation section.
        final_answer_prefix: Prefix for final answer.

    Examples:
        >>> config = ReActConfig(
        ...     thought_prefix="Thought:",
        ...     action_prefix="Action:",
        ...     observation_prefix="Observation:",
        ...     final_answer_prefix="Final Answer:",
        ... )
        >>> config.thought_prefix
        'Thought:'
    """

    thought_prefix: str
    action_prefix: str
    observation_prefix: str
    final_answer_prefix: str


def validate_tool_definition(tool: ToolDefinition) -> None:
    """Validate tool definition.

    Args:
        tool: Tool definition to validate.

    Raises:
        ValueError: If definition is invalid.

    Examples:
        >>> tool = ToolDefinition("calc", "Calculate", ToolType.FUNCTION, {})
        >>> validate_tool_definition(tool)  # No error

        >>> bad = ToolDefinition("", "Calculate", ToolType.FUNCTION, {})
        >>> validate_tool_definition(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not tool.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if not tool.description:
        msg = "description cannot be empty"
        raise ValueError(msg)

    # Validate name format (alphanumeric and underscores)
    if not tool.name.replace("_", "").isalnum():
        msg = f"name must be alphanumeric with underscores, got '{tool.name}'"
        raise ValueError(msg)


def validate_agent_config(config: AgentConfig) -> None:
    """Validate agent configuration.

    Args:
        config: Agent configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = AgentConfig(10, (), "System prompt", 0.7)
        >>> validate_agent_config(config)  # No error

        >>> bad = AgentConfig(0, (), "System prompt", 0.7)
        >>> validate_agent_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_steps must be positive
    """
    if config.max_steps <= 0:
        msg = f"max_steps must be positive, got {config.max_steps}"
        raise ValueError(msg)

    if config.temperature < 0:
        msg = f"temperature must be non-negative, got {config.temperature}"
        raise ValueError(msg)


def create_tool_definition(
    name: str,
    description: str,
    tool_type: str = "function",
    parameters: dict[str, Any] | None = None,
) -> ToolDefinition:
    """Create a tool definition.

    Args:
        name: Tool name.
        description: Tool description.
        tool_type: Type of tool. Defaults to "function".
        parameters: Parameter schema. Defaults to None.

    Returns:
        ToolDefinition with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> tool = create_tool_definition("search", "Search the web")
        >>> tool.name
        'search'

        >>> tool = create_tool_definition(
        ...     "calc",
        ...     "Calculate",
        ...     parameters={"expr": "string"},
        ... )
        >>> "expr" in tool.parameters
        True

        >>> create_tool_definition("", "desc")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if tool_type not in VALID_TOOL_TYPES:
        msg = f"tool_type must be one of {VALID_TOOL_TYPES}, got '{tool_type}'"
        raise ValueError(msg)

    tool = ToolDefinition(
        name=name,
        description=description,
        tool_type=ToolType(tool_type),
        parameters=parameters if parameters is not None else {},
    )
    validate_tool_definition(tool)
    return tool


def create_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str = "",
) -> ToolCall:
    """Create a tool call.

    Args:
        tool_name: Name of tool to call.
        arguments: Arguments for the call.
        call_id: Unique call ID. Defaults to "".

    Returns:
        ToolCall with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> call = create_tool_call("search", {"query": "weather"})
        >>> call.tool_name
        'search'

        >>> create_tool_call("", {"query": "test"})
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tool_name cannot be empty
    """
    if not tool_name:
        msg = "tool_name cannot be empty"
        raise ValueError(msg)

    return ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        call_id=call_id if call_id else f"call_{hash(tool_name) % 10000:04d}",
    )


def create_tool_result(
    call_id: str,
    output: str = "",
    error: str | None = None,
    success: bool = True,
) -> ToolResult:
    """Create a tool result.

    Args:
        call_id: ID of the tool call.
        output: Tool output. Defaults to "".
        error: Error message. Defaults to None.
        success: Whether successful. Defaults to True.

    Returns:
        ToolResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_tool_result("call_123", output="42")
        >>> result.output
        '42'

        >>> result = create_tool_result("call_123", error="Not found", success=False)
        >>> result.success
        False

        >>> create_tool_result("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: call_id cannot be empty
    """
    if not call_id:
        msg = "call_id cannot be empty"
        raise ValueError(msg)

    return ToolResult(
        call_id=call_id,
        output=output,
        error=error,
        success=success,
    )


def create_agent_config(
    max_steps: int = 10,
    tools: tuple[ToolDefinition, ...] = (),
    system_prompt: str = "You are a helpful assistant with access to tools.",
    temperature: float = 0.7,
) -> AgentConfig:
    """Create an agent configuration.

    Args:
        max_steps: Maximum steps. Defaults to 10.
        tools: Available tools. Defaults to ().
        system_prompt: System prompt. Defaults to helpful assistant.
        temperature: Sampling temperature. Defaults to 0.7.

    Returns:
        AgentConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_agent_config(max_steps=20)
        >>> config.max_steps
        20

        >>> create_agent_config(max_steps=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_steps must be positive
    """
    config = AgentConfig(
        max_steps=max_steps,
        tools=tools,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    validate_agent_config(config)
    return config


def create_react_config(
    thought_prefix: str = "Thought:",
    action_prefix: str = "Action:",
    observation_prefix: str = "Observation:",
    final_answer_prefix: str = "Final Answer:",
) -> ReActConfig:
    """Create a ReAct configuration.

    Args:
        thought_prefix: Thought prefix. Defaults to "Thought:".
        action_prefix: Action prefix. Defaults to "Action:".
        observation_prefix: Observation prefix. Defaults to "Observation:".
        final_answer_prefix: Final answer prefix. Defaults to "Final Answer:".

    Returns:
        ReActConfig with the specified settings.

    Examples:
        >>> config = create_react_config()
        >>> config.thought_prefix
        'Thought:'

        >>> config = create_react_config(thought_prefix="Think:")
        >>> config.thought_prefix
        'Think:'
    """
    return ReActConfig(
        thought_prefix=thought_prefix,
        action_prefix=action_prefix,
        observation_prefix=observation_prefix,
        final_answer_prefix=final_answer_prefix,
    )


def list_tool_types() -> list[str]:
    """List supported tool types.

    Returns:
        Sorted list of tool type names.

    Examples:
        >>> types = list_tool_types()
        >>> "function" in types
        True
        >>> "retrieval" in types
        True
    """
    return sorted(VALID_TOOL_TYPES)


def get_tool_type(name: str) -> ToolType:
    """Get tool type from name.

    Args:
        name: Tool type name.

    Returns:
        ToolType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_tool_type("function")
        <ToolType.FUNCTION: 'function'>

        >>> get_tool_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tool type must be one of
    """
    if name not in VALID_TOOL_TYPES:
        msg = f"tool type must be one of {VALID_TOOL_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ToolType(name)


def format_tool_for_prompt(tool: ToolDefinition) -> str:
    """Format tool definition for inclusion in prompt.

    Args:
        tool: Tool definition to format.

    Returns:
        Formatted string for prompt.

    Examples:
        >>> tool = create_tool_definition("search", "Search the web")
        >>> formatted = format_tool_for_prompt(tool)
        >>> "search" in formatted
        True
        >>> "Search the web" in formatted
        True
    """
    params_str = json.dumps(tool.parameters) if tool.parameters else "{}"
    return f"- {tool.name}: {tool.description}\n  Parameters: {params_str}"


def parse_tool_call(text: str, available_tools: tuple[str, ...]) -> ToolCall | None:
    """Parse a tool call from model output.

    Args:
        text: Model output text.
        available_tools: Tuple of available tool names.

    Returns:
        Parsed ToolCall or None if no valid call found.

    Examples:
        >>> call = parse_tool_call("search(query='weather')", ("search",))
        >>> call is not None
        True
        >>> call.tool_name
        'search'

        >>> call = parse_tool_call("invalid text", ("search",))
        >>> call is None
        True
    """
    # Simple parsing: look for tool_name(args) pattern
    for tool_name in available_tools:
        if f"{tool_name}(" in text:
            # Extract arguments between parentheses
            start = text.find(f"{tool_name}(") + len(tool_name) + 1
            end = text.find(")", start)
            if end > start:
                args_str = text[start:end]
                # Parse simple key='value' patterns
                arguments: dict[str, Any] = {}
                for part in args_str.split(","):
                    part = part.strip()
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        arguments[key] = value
                return create_tool_call(tool_name, arguments)
    return None
