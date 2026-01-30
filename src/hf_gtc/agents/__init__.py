"""Agent and tool use recipes for HuggingFace models.

This module provides utilities for building agents with tool use
capabilities, including ReAct patterns and function calling.

Examples:
    >>> from hf_gtc.agents import ToolDefinition, ToolType
    >>> tool = ToolDefinition(
    ...     name="calculator",
    ...     description="Perform math calculations",
    ...     tool_type=ToolType.FUNCTION,
    ...     parameters={"expression": "string"},
    ... )
    >>> tool.name
    'calculator'
"""

from __future__ import annotations

from hf_gtc.agents.tools import (
    VALID_TOOL_TYPES,
    AgentConfig,
    AgentState,
    AgentStep,
    ReActConfig,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolType,
    create_agent_config,
    create_react_config,
    create_tool_call,
    create_tool_definition,
    create_tool_result,
    format_tool_for_prompt,
    get_tool_type,
    list_tool_types,
    parse_tool_call,
    validate_agent_config,
    validate_tool_definition,
)

__all__: list[str] = [
    "VALID_TOOL_TYPES",
    "AgentConfig",
    "AgentState",
    "AgentStep",
    "ReActConfig",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ToolType",
    "create_agent_config",
    "create_react_config",
    "create_tool_call",
    "create_tool_definition",
    "create_tool_result",
    "format_tool_for_prompt",
    "get_tool_type",
    "list_tool_types",
    "parse_tool_call",
    "validate_agent_config",
    "validate_tool_definition",
]
