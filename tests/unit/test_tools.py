"""Tests for agents.tools module."""

from __future__ import annotations

import pytest

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


class TestToolType:
    """Tests for ToolType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for tool_type in ToolType:
            assert isinstance(tool_type.value, str)

    def test_function_value(self) -> None:
        """Function has correct value."""
        assert ToolType.FUNCTION.value == "function"

    def test_code_interpreter_value(self) -> None:
        """Code interpreter has correct value."""
        assert ToolType.CODE_INTERPRETER.value == "code_interpreter"

    def test_valid_tool_types_frozenset(self) -> None:
        """VALID_TOOL_TYPES is a frozenset."""
        assert isinstance(VALID_TOOL_TYPES, frozenset)


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_create_definition(self) -> None:
        """Create tool definition."""
        definition = ToolDefinition(
            name="calculator",
            description="Perform calculations",
            tool_type=ToolType.FUNCTION,
            parameters={"expression": "string"},
        )
        assert definition.name == "calculator"

    def test_definition_is_frozen(self) -> None:
        """Definition is immutable."""
        definition = ToolDefinition(
            "calc", "Calculate", ToolType.FUNCTION, {}
        )
        with pytest.raises(AttributeError):
            definition.name = "new_name"  # type: ignore[misc]


class TestValidateToolDefinition:
    """Tests for validate_tool_definition function."""

    def test_valid_definition(self) -> None:
        """Valid definition passes validation."""
        definition = ToolDefinition("calc", "Calculate", ToolType.FUNCTION, {})
        validate_tool_definition(definition)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        definition = ToolDefinition("", "Calculate", ToolType.FUNCTION, {})
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_tool_definition(definition)

    def test_empty_description_raises(self) -> None:
        """Empty description raises ValueError."""
        definition = ToolDefinition("calc", "", ToolType.FUNCTION, {})
        with pytest.raises(ValueError, match="description cannot be empty"):
            validate_tool_definition(definition)

    def test_invalid_name_format_raises(self) -> None:
        """Invalid name format raises ValueError."""
        definition = ToolDefinition("calc-tool", "Calculate", ToolType.FUNCTION, {})
        with pytest.raises(ValueError, match="name must be alphanumeric"):
            validate_tool_definition(definition)


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_call(self) -> None:
        """Create tool call."""
        call = ToolCall(
            tool_name="search",
            arguments={"query": "weather"},
            call_id="call_123",
        )
        assert call.tool_name == "search"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_create_result(self) -> None:
        """Create tool result."""
        result = ToolResult(
            call_id="call_123",
            output="The weather is sunny",
            error=None,
            success=True,
        )
        assert result.success is True


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_create_config(self) -> None:
        """Create agent config."""
        config = AgentConfig(
            max_steps=10,
            tools=(),
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
        )
        assert config.max_steps == 10


class TestValidateAgentConfig:
    """Tests for validate_agent_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AgentConfig(10, (), "System prompt", 0.7)
        validate_agent_config(config)

    def test_zero_steps_raises(self) -> None:
        """Zero steps raises ValueError."""
        config = AgentConfig(0, (), "System prompt", 0.7)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            validate_agent_config(config)

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        config = AgentConfig(10, (), "System prompt", -0.1)
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            validate_agent_config(config)


class TestReActConfig:
    """Tests for ReActConfig dataclass."""

    def test_create_config(self) -> None:
        """Create ReAct config."""
        config = ReActConfig(
            thought_prefix="Thought:",
            action_prefix="Action:",
            observation_prefix="Observation:",
            final_answer_prefix="Final Answer:",
        )
        assert config.thought_prefix == "Thought:"


class TestCreateToolDefinition:
    """Tests for create_tool_definition function."""

    def test_default_definition(self) -> None:
        """Create default definition."""
        definition = create_tool_definition("search", "Search the web")
        assert definition.name == "search"
        assert definition.tool_type == ToolType.FUNCTION

    def test_custom_definition(self) -> None:
        """Create custom definition."""
        definition = create_tool_definition(
            "search",
            "Search the web",
            tool_type="retrieval",
            parameters={"query": "string"},
        )
        assert definition.tool_type == ToolType.RETRIEVAL
        assert "query" in definition.parameters

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_tool_definition("", "Description")

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="tool_type must be one of"):
            create_tool_definition("search", "Search", tool_type="invalid")


class TestCreateToolCall:
    """Tests for create_tool_call function."""

    def test_default_call(self) -> None:
        """Create default call."""
        call = create_tool_call("search", {"query": "weather"})
        assert call.tool_name == "search"
        assert call.arguments["query"] == "weather"

    def test_with_call_id(self) -> None:
        """Create call with ID."""
        call = create_tool_call("search", {"query": "weather"}, call_id="custom_id")
        assert call.call_id == "custom_id"

    def test_auto_call_id(self) -> None:
        """Auto-generated call ID."""
        call = create_tool_call("search", {})
        assert call.call_id.startswith("call_")

    def test_empty_tool_name_raises(self) -> None:
        """Empty tool name raises ValueError."""
        with pytest.raises(ValueError, match="tool_name cannot be empty"):
            create_tool_call("", {})


class TestCreateToolResult:
    """Tests for create_tool_result function."""

    def test_default_result(self) -> None:
        """Create default result."""
        result = create_tool_result("call_123", output="42")
        assert result.output == "42"
        assert result.success is True

    def test_error_result(self) -> None:
        """Create error result."""
        result = create_tool_result("call_123", error="Not found", success=False)
        assert result.error == "Not found"
        assert result.success is False

    def test_empty_call_id_raises(self) -> None:
        """Empty call_id raises ValueError."""
        with pytest.raises(ValueError, match="call_id cannot be empty"):
            create_tool_result("")


class TestCreateAgentConfig:
    """Tests for create_agent_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_agent_config()
        assert config.max_steps == 10
        assert config.temperature == 0.7

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_agent_config(max_steps=20, temperature=0.5)
        assert config.max_steps == 20
        assert config.temperature == 0.5

    def test_zero_steps_raises(self) -> None:
        """Zero steps raises ValueError."""
        with pytest.raises(ValueError, match="max_steps must be positive"):
            create_agent_config(max_steps=0)


class TestCreateReActConfig:
    """Tests for create_react_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_react_config()
        assert config.thought_prefix == "Thought:"
        assert config.action_prefix == "Action:"

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_react_config(thought_prefix="Think:")
        assert config.thought_prefix == "Think:"


class TestListToolTypes:
    """Tests for list_tool_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_tool_types()
        assert types == sorted(types)

    def test_contains_function(self) -> None:
        """Contains function."""
        types = list_tool_types()
        assert "function" in types


class TestGetToolType:
    """Tests for get_tool_type function."""

    def test_get_function(self) -> None:
        """Get function type."""
        assert get_tool_type("function") == ToolType.FUNCTION

    def test_get_retrieval(self) -> None:
        """Get retrieval type."""
        assert get_tool_type("retrieval") == ToolType.RETRIEVAL

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="tool type must be one of"):
            get_tool_type("invalid")


class TestFormatToolForPrompt:
    """Tests for format_tool_for_prompt function."""

    def test_basic_format(self) -> None:
        """Format basic tool."""
        tool = create_tool_definition("search", "Search the web")
        formatted = format_tool_for_prompt(tool)
        assert "search" in formatted
        assert "Search the web" in formatted

    def test_with_parameters(self) -> None:
        """Format tool with parameters."""
        tool = create_tool_definition(
            "search",
            "Search the web",
            parameters={"query": "string"},
        )
        formatted = format_tool_for_prompt(tool)
        assert "query" in formatted


class TestParseToolCall:
    """Tests for parse_tool_call function."""

    def test_parse_valid_call(self) -> None:
        """Parse valid tool call."""
        call = parse_tool_call("search(query='weather')", ("search",))
        assert call is not None
        assert call.tool_name == "search"
        assert call.arguments["query"] == "weather"

    def test_no_match_returns_none(self) -> None:
        """No match returns None."""
        call = parse_tool_call("invalid text", ("search",))
        assert call is None

    def test_wrong_tool_returns_none(self) -> None:
        """Wrong tool returns None."""
        call = parse_tool_call("calculate(expr='1+1')", ("search",))
        assert call is None

    def test_multiple_arguments(self) -> None:
        """Parse multiple arguments."""
        call = parse_tool_call(
            "search(query='weather', location='NYC')",
            ("search",),
        )
        assert call is not None
        assert call.arguments["query"] == "weather"
        assert call.arguments["location"] == "NYC"


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_create_step(self) -> None:
        """Create agent step."""
        step = AgentStep(
            thought="I need to search for info",
            action="search(query='weather')",
            observation="Weather is sunny",
            step_number=1,
        )
        assert step.step_number == 1


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_create_state(self) -> None:
        """Create agent state."""
        state = AgentState(
            steps=(),
            current_step=0,
            is_complete=False,
            final_answer=None,
        )
        assert state.is_complete is False

    def test_complete_state(self) -> None:
        """Create complete state."""
        state = AgentState(
            steps=(),
            current_step=5,
            is_complete=True,
            final_answer="The answer is 42",
        )
        assert state.is_complete is True
        assert state.final_answer == "The answer is 42"
