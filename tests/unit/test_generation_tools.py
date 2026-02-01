"""Tests for generation.tools module."""

from __future__ import annotations

import pytest

from hf_gtc.generation.tools import (
    VALID_EXECUTION_MODES,
    VALID_PARSING_STRATEGIES,
    VALID_TOOL_FORMATS,
    ExecutionMode,
    ParsingStrategy,
    ToolCall,
    ToolConfig,
    ToolDefinition,
    ToolFormat,
    ToolResult,
    ToolStats,
    create_tool_call,
    create_tool_config,
    create_tool_definition,
    create_tool_result,
    create_tool_stats,
    execute_tool_calls,
    format_tool_stats,
    format_tools_for_prompt,
    get_execution_mode,
    get_parsing_strategy,
    get_recommended_tool_config,
    get_tool_format,
    list_execution_modes,
    list_parsing_strategies,
    list_tool_formats,
    parse_tool_calls,
    validate_tool_arguments,
    validate_tool_call,
    validate_tool_config,
    validate_tool_definition,
    validate_tool_result,
)


class TestToolFormat:
    """Tests for ToolFormat enum."""

    def test_all_formats_have_values(self) -> None:
        """All formats have string values."""
        for fmt in ToolFormat:
            assert isinstance(fmt.value, str)

    def test_valid_tool_formats_frozenset(self) -> None:
        """VALID_TOOL_FORMATS is a frozenset of all format values."""
        assert isinstance(VALID_TOOL_FORMATS, frozenset)
        assert len(VALID_TOOL_FORMATS) == len(ToolFormat)

    def test_openai_value(self) -> None:
        """OPENAI format has correct value."""
        assert ToolFormat.OPENAI.value == "openai"

    def test_anthropic_value(self) -> None:
        """ANTHROPIC format has correct value."""
        assert ToolFormat.ANTHROPIC.value == "anthropic"

    def test_json_schema_value(self) -> None:
        """JSON_SCHEMA format has correct value."""
        assert ToolFormat.JSON_SCHEMA.value == "json_schema"

    def test_xml_value(self) -> None:
        """XML format has correct value."""
        assert ToolFormat.XML.value == "xml"


class TestParsingStrategy:
    """Tests for ParsingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ParsingStrategy:
            assert isinstance(strategy.value, str)

    def test_valid_parsing_strategies_frozenset(self) -> None:
        """VALID_PARSING_STRATEGIES is a frozenset of all strategy values."""
        assert isinstance(VALID_PARSING_STRATEGIES, frozenset)
        assert len(VALID_PARSING_STRATEGIES) == len(ParsingStrategy)

    def test_json_value(self) -> None:
        """JSON strategy has correct value."""
        assert ParsingStrategy.JSON.value == "json"

    def test_regex_value(self) -> None:
        """REGEX strategy has correct value."""
        assert ParsingStrategy.REGEX.value == "regex"

    def test_structured_value(self) -> None:
        """STRUCTURED strategy has correct value."""
        assert ParsingStrategy.STRUCTURED.value == "structured"

    def test_llm_value(self) -> None:
        """LLM strategy has correct value."""
        assert ParsingStrategy.LLM.value == "llm"


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_all_modes_have_values(self) -> None:
        """All modes have string values."""
        for mode in ExecutionMode:
            assert isinstance(mode.value, str)

    def test_valid_execution_modes_frozenset(self) -> None:
        """VALID_EXECUTION_MODES is a frozenset of all mode values."""
        assert isinstance(VALID_EXECUTION_MODES, frozenset)
        assert len(VALID_EXECUTION_MODES) == len(ExecutionMode)

    def test_sequential_value(self) -> None:
        """SEQUENTIAL mode has correct value."""
        assert ExecutionMode.SEQUENTIAL.value == "sequential"

    def test_parallel_value(self) -> None:
        """PARALLEL mode has correct value."""
        assert ExecutionMode.PARALLEL.value == "parallel"

    def test_auto_value(self) -> None:
        """AUTO mode has correct value."""
        assert ExecutionMode.AUTO.value == "auto"


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_create_definition(self) -> None:
        """Create tool definition."""
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"query": "string"},
            required=("query",),
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "query" in tool.parameters
        assert "query" in tool.required

    def test_definition_is_frozen(self) -> None:
        """Definition is immutable."""
        tool = ToolDefinition("search", "Search", {}, ())
        with pytest.raises(AttributeError):
            tool.name = "other"  # type: ignore[misc]

    def test_empty_parameters(self) -> None:
        """Definition with empty parameters."""
        tool = ToolDefinition("tool", "Description", {}, ())
        assert tool.parameters == {}
        assert tool.required == ()


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_call(self) -> None:
        """Create tool call."""
        call = ToolCall(
            name="search",
            arguments={"query": "python"},
            id="call_001",
        )
        assert call.name == "search"
        assert call.arguments["query"] == "python"
        assert call.id == "call_001"

    def test_call_is_frozen(self) -> None:
        """Call is immutable."""
        call = ToolCall("search", {"q": "test"}, "call_001")
        with pytest.raises(AttributeError):
            call.name = "other"  # type: ignore[misc]

    def test_empty_arguments(self) -> None:
        """Call with empty arguments."""
        call = ToolCall("tool", {}, "call_001")
        assert call.arguments == {}


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_create_success_result(self) -> None:
        """Create successful result."""
        result = ToolResult(
            call_id="call_001",
            output="Success!",
            error=None,
        )
        assert result.call_id == "call_001"
        assert result.output == "Success!"
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Create error result."""
        result = ToolResult(
            call_id="call_001",
            output="",
            error="Tool not found",
        )
        assert result.error == "Tool not found"

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = ToolResult("call_001", "output", None)
        with pytest.raises(AttributeError):
            result.output = "new"  # type: ignore[misc]


class TestToolConfig:
    """Tests for ToolConfig dataclass."""

    def test_create_config(self) -> None:
        """Create tool config."""
        config = ToolConfig(
            format=ToolFormat.OPENAI,
            definitions=(),
            parsing_strategy=ParsingStrategy.JSON,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_calls=10,
        )
        assert config.format == ToolFormat.OPENAI
        assert config.definitions == ()
        assert config.parsing_strategy == ParsingStrategy.JSON
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.max_calls == 10

    def test_config_with_definitions(self) -> None:
        """Create config with tool definitions."""
        tool = ToolDefinition("search", "Search", {}, ())
        config = ToolConfig(
            format=ToolFormat.ANTHROPIC,
            definitions=(tool,),
            parsing_strategy=ParsingStrategy.REGEX,
            execution_mode=ExecutionMode.PARALLEL,
            max_calls=5,
        )
        assert len(config.definitions) == 1
        assert config.definitions[0].name == "search"

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ToolConfig(
            ToolFormat.OPENAI, (), ParsingStrategy.JSON, ExecutionMode.SEQUENTIAL, 10
        )
        with pytest.raises(AttributeError):
            config.max_calls = 20  # type: ignore[misc]


class TestToolStats:
    """Tests for ToolStats dataclass."""

    def test_create_stats(self) -> None:
        """Create tool stats."""
        stats = ToolStats(
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            total_time_ms=500.0,
            calls_by_tool={"search": 5, "calc": 5},
        )
        assert stats.total_calls == 10
        assert stats.successful_calls == 8
        assert stats.failed_calls == 2
        assert stats.total_time_ms == pytest.approx(500.0)
        assert stats.calls_by_tool["search"] == 5

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = ToolStats(5, 4, 1, 100.0, {})
        with pytest.raises(AttributeError):
            stats.total_calls = 10  # type: ignore[misc]


class TestValidateToolDefinition:
    """Tests for validate_tool_definition function."""

    def test_valid_definition(self) -> None:
        """Valid definition passes."""
        tool = ToolDefinition("search", "Search web", {"q": "string"}, ("q",))
        validate_tool_definition(tool)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="tool cannot be None"):
            validate_tool_definition(None)  # type: ignore[arg-type]

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        tool = ToolDefinition("", "Description", {}, ())
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_tool_definition(tool)

    def test_invalid_name_format_raises(self) -> None:
        """Invalid name format raises ValueError."""
        tool = ToolDefinition("bad-name", "Description", {}, ())
        with pytest.raises(ValueError, match="name must be alphanumeric"):
            validate_tool_definition(tool)

    def test_empty_description_raises(self) -> None:
        """Empty description raises ValueError."""
        tool = ToolDefinition("search", "", {}, ())
        with pytest.raises(ValueError, match="description cannot be empty"):
            validate_tool_definition(tool)

    def test_missing_required_param_raises(self) -> None:
        """Required param not in parameters raises ValueError."""
        tool = ToolDefinition("search", "Search", {}, ("missing",))
        with pytest.raises(ValueError, match="required parameter 'missing' not in"):
            validate_tool_definition(tool)

    def test_valid_with_underscores(self) -> None:
        """Name with underscores is valid."""
        tool = ToolDefinition("my_search_tool", "Search", {}, ())
        validate_tool_definition(tool)


class TestValidateToolCall:
    """Tests for validate_tool_call function."""

    def test_valid_call(self) -> None:
        """Valid call passes."""
        call = ToolCall("search", {"q": "test"}, "call_001")
        validate_tool_call(call)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="call cannot be None"):
            validate_tool_call(None)  # type: ignore[arg-type]

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        call = ToolCall("", {}, "call_001")
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_tool_call(call)

    def test_empty_id_raises(self) -> None:
        """Empty id raises ValueError."""
        call = ToolCall("search", {}, "")
        with pytest.raises(ValueError, match="id cannot be empty"):
            validate_tool_call(call)


class TestValidateToolResult:
    """Tests for validate_tool_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes."""
        result = ToolResult("call_001", "output", None)
        validate_tool_result(result)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            validate_tool_result(None)  # type: ignore[arg-type]

    def test_empty_call_id_raises(self) -> None:
        """Empty call_id raises ValueError."""
        result = ToolResult("", "output", None)
        with pytest.raises(ValueError, match="call_id cannot be empty"):
            validate_tool_result(result)


class TestValidateToolConfig:
    """Tests for validate_tool_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes."""
        config = ToolConfig(
            ToolFormat.OPENAI, (), ParsingStrategy.JSON, ExecutionMode.SEQUENTIAL, 10
        )
        validate_tool_config(config)

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_tool_config(None)  # type: ignore[arg-type]

    def test_zero_max_calls_raises(self) -> None:
        """Zero max_calls raises ValueError."""
        config = ToolConfig(
            ToolFormat.OPENAI, (), ParsingStrategy.JSON, ExecutionMode.SEQUENTIAL, 0
        )
        with pytest.raises(ValueError, match="max_calls must be positive"):
            validate_tool_config(config)

    def test_negative_max_calls_raises(self) -> None:
        """Negative max_calls raises ValueError."""
        config = ToolConfig(
            ToolFormat.OPENAI, (), ParsingStrategy.JSON, ExecutionMode.SEQUENTIAL, -1
        )
        with pytest.raises(ValueError, match="max_calls must be positive"):
            validate_tool_config(config)

    def test_invalid_tool_definition_raises(self) -> None:
        """Invalid tool definition raises ValueError."""
        bad_tool = ToolDefinition("", "desc", {}, ())
        config = ToolConfig(
            ToolFormat.OPENAI,
            (bad_tool,),
            ParsingStrategy.JSON,
            ExecutionMode.SEQUENTIAL,
            10,
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_tool_config(config)


class TestCreateToolDefinition:
    """Tests for create_tool_definition function."""

    def test_basic_creation(self) -> None:
        """Create basic tool definition."""
        tool = create_tool_definition("search", "Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.parameters == {}
        assert tool.required == ()

    def test_with_parameters(self) -> None:
        """Create with parameters."""
        tool = create_tool_definition(
            name="search",
            description="Search",
            parameters={"query": "string", "limit": "integer"},
            required=("query",),
        )
        assert "query" in tool.parameters
        assert "limit" in tool.parameters
        assert "query" in tool.required

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_tool_definition("", "description")

    def test_empty_description_raises(self) -> None:
        """Empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            create_tool_definition("search", "")


class TestCreateToolCall:
    """Tests for create_tool_call function."""

    def test_basic_creation(self) -> None:
        """Create basic tool call."""
        call = create_tool_call("search", {"query": "python"}, "call_001")
        assert call.name == "search"
        assert call.arguments["query"] == "python"
        assert call.id == "call_001"

    def test_auto_generated_id(self) -> None:
        """Auto-generate ID when not provided."""
        call = create_tool_call("search")
        assert call.id.startswith("call_")

    def test_empty_arguments_default(self) -> None:
        """Default to empty arguments."""
        call = create_tool_call("search", id="call_001")
        assert call.arguments == {}

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_tool_call("")


class TestCreateToolResult:
    """Tests for create_tool_result function."""

    def test_success_result(self) -> None:
        """Create success result."""
        result = create_tool_result("call_001", output="Done!")
        assert result.call_id == "call_001"
        assert result.output == "Done!"
        assert result.error is None

    def test_error_result(self) -> None:
        """Create error result."""
        result = create_tool_result("call_001", error="Failed")
        assert result.error == "Failed"

    def test_empty_call_id_raises(self) -> None:
        """Empty call_id raises ValueError."""
        with pytest.raises(ValueError, match="call_id cannot be empty"):
            create_tool_result("")


class TestCreateToolConfig:
    """Tests for create_tool_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_tool_config()
        assert config.format == ToolFormat.OPENAI
        assert config.definitions == ()
        assert config.parsing_strategy == ParsingStrategy.JSON
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.max_calls == 10

    def test_with_string_format(self) -> None:
        """Create with string format."""
        config = create_tool_config(format="anthropic")
        assert config.format == ToolFormat.ANTHROPIC

    def test_with_enum_format(self) -> None:
        """Create with enum format."""
        config = create_tool_config(format=ToolFormat.XML)
        assert config.format == ToolFormat.XML

    def test_with_string_strategy(self) -> None:
        """Create with string parsing strategy."""
        config = create_tool_config(parsing_strategy="regex")
        assert config.parsing_strategy == ParsingStrategy.REGEX

    def test_with_string_mode(self) -> None:
        """Create with string execution mode."""
        config = create_tool_config(execution_mode="parallel")
        assert config.execution_mode == ExecutionMode.PARALLEL

    def test_with_definitions(self) -> None:
        """Create with tool definitions."""
        tool = create_tool_definition("search", "Search")
        config = create_tool_config(definitions=(tool,))
        assert len(config.definitions) == 1

    def test_custom_max_calls(self) -> None:
        """Create with custom max_calls."""
        config = create_tool_config(max_calls=20)
        assert config.max_calls == 20

    def test_zero_max_calls_raises(self) -> None:
        """Zero max_calls raises ValueError."""
        with pytest.raises(ValueError, match="max_calls must be positive"):
            create_tool_config(max_calls=0)


class TestCreateToolStats:
    """Tests for create_tool_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_tool_stats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.total_time_ms == pytest.approx(0.0)
        assert stats.calls_by_tool == {}

    def test_with_values(self) -> None:
        """Create with values."""
        stats = create_tool_stats(
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            total_time_ms=100.5,
            calls_by_tool={"search": 5},
        )
        assert stats.total_calls == 10
        assert stats.calls_by_tool["search"] == 5

    def test_negative_total_calls_raises(self) -> None:
        """Negative total_calls raises ValueError."""
        with pytest.raises(ValueError, match="total_calls must be non-negative"):
            create_tool_stats(total_calls=-1)

    def test_negative_successful_calls_raises(self) -> None:
        """Negative successful_calls raises ValueError."""
        with pytest.raises(ValueError, match="successful_calls must be non-negative"):
            create_tool_stats(successful_calls=-1)

    def test_negative_failed_calls_raises(self) -> None:
        """Negative failed_calls raises ValueError."""
        with pytest.raises(ValueError, match="failed_calls must be non-negative"):
            create_tool_stats(failed_calls=-1)

    def test_negative_total_time_raises(self) -> None:
        """Negative total_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="total_time_ms must be non-negative"):
            create_tool_stats(total_time_ms=-1.0)


class TestListToolFormats:
    """Tests for list_tool_formats function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        formats = list_tool_formats()
        assert formats == sorted(formats)

    def test_contains_all_formats(self) -> None:
        """Contains all formats."""
        formats = list_tool_formats()
        assert "openai" in formats
        assert "anthropic" in formats
        assert "json_schema" in formats
        assert "xml" in formats

    def test_correct_count(self) -> None:
        """Has correct count."""
        formats = list_tool_formats()
        assert len(formats) == len(ToolFormat)


class TestListParsingStrategies:
    """Tests for list_parsing_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_parsing_strategies()
        assert strategies == sorted(strategies)

    def test_contains_all_strategies(self) -> None:
        """Contains all strategies."""
        strategies = list_parsing_strategies()
        assert "json" in strategies
        assert "regex" in strategies
        assert "structured" in strategies
        assert "llm" in strategies

    def test_correct_count(self) -> None:
        """Has correct count."""
        strategies = list_parsing_strategies()
        assert len(strategies) == len(ParsingStrategy)


class TestListExecutionModes:
    """Tests for list_execution_modes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        modes = list_execution_modes()
        assert modes == sorted(modes)

    def test_contains_all_modes(self) -> None:
        """Contains all modes."""
        modes = list_execution_modes()
        assert "sequential" in modes
        assert "parallel" in modes
        assert "auto" in modes

    def test_correct_count(self) -> None:
        """Has correct count."""
        modes = list_execution_modes()
        assert len(modes) == len(ExecutionMode)


class TestGetToolFormat:
    """Tests for get_tool_format function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("openai", ToolFormat.OPENAI),
            ("anthropic", ToolFormat.ANTHROPIC),
            ("json_schema", ToolFormat.JSON_SCHEMA),
            ("xml", ToolFormat.XML),
        ],
    )
    def test_get_valid_format(self, name: str, expected: ToolFormat) -> None:
        """Get valid formats."""
        assert get_tool_format(name) == expected

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="invalid tool format"):
            get_tool_format("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid tool format"):
            get_tool_format("")


class TestGetParsingStrategy:
    """Tests for get_parsing_strategy function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("json", ParsingStrategy.JSON),
            ("regex", ParsingStrategy.REGEX),
            ("structured", ParsingStrategy.STRUCTURED),
            ("llm", ParsingStrategy.LLM),
        ],
    )
    def test_get_valid_strategy(self, name: str, expected: ParsingStrategy) -> None:
        """Get valid strategies."""
        assert get_parsing_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid parsing strategy"):
            get_parsing_strategy("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid parsing strategy"):
            get_parsing_strategy("")


class TestGetExecutionMode:
    """Tests for get_execution_mode function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sequential", ExecutionMode.SEQUENTIAL),
            ("parallel", ExecutionMode.PARALLEL),
            ("auto", ExecutionMode.AUTO),
        ],
    )
    def test_get_valid_mode(self, name: str, expected: ExecutionMode) -> None:
        """Get valid modes."""
        assert get_execution_mode(name) == expected

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid execution mode"):
            get_execution_mode("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="invalid execution mode"):
            get_execution_mode("")


class TestFormatToolsForPrompt:
    """Tests for format_tools_for_prompt function."""

    def test_none_tools_raises(self) -> None:
        """None tools raises ValueError."""
        with pytest.raises(ValueError, match="tools cannot be None"):
            format_tools_for_prompt(None)  # type: ignore[arg-type]

    def test_empty_tools_returns_empty(self) -> None:
        """Empty tools returns empty string."""
        result = format_tools_for_prompt(())
        assert result == ""

    def test_openai_format(self) -> None:
        """Format in OpenAI format."""
        tool = create_tool_definition(
            name="search",
            description="Search the web",
            parameters={"query": "string"},
            required=("query",),
        )
        result = format_tools_for_prompt((tool,), ToolFormat.OPENAI)
        assert "function" in result
        assert "search" in result
        assert "Search the web" in result

    def test_anthropic_format(self) -> None:
        """Format in Anthropic format."""
        tool = create_tool_definition(
            name="search",
            description="Search the web",
            parameters={"query": "string"},
            required=("query",),
        )
        result = format_tools_for_prompt((tool,), ToolFormat.ANTHROPIC)
        assert "input_schema" in result
        assert "search" in result

    def test_json_schema_format(self) -> None:
        """Format in JSON schema format."""
        tool = create_tool_definition(
            name="search",
            description="Search the web",
            parameters={"query": "string"},
            required=("query",),
        )
        result = format_tools_for_prompt((tool,), ToolFormat.JSON_SCHEMA)
        assert "schema" in result
        assert "search" in result

    def test_xml_format(self) -> None:
        """Format in XML format."""
        tool = create_tool_definition(
            name="search",
            description="Search the web",
            parameters={"query": "string"},
            required=("query",),
        )
        result = format_tools_for_prompt((tool,), ToolFormat.XML)
        assert "<tool" in result
        assert "<tools>" in result
        assert "search" in result
        assert "<description>" in result

    def test_multiple_tools(self) -> None:
        """Format multiple tools."""
        tool1 = create_tool_definition("search", "Search web")
        tool2 = create_tool_definition("calculate", "Calculate math")
        result = format_tools_for_prompt((tool1, tool2), ToolFormat.OPENAI)
        assert "search" in result
        assert "calculate" in result


class TestParseToolCalls:
    """Tests for parse_tool_calls function."""

    def test_parse_json_single_call(self) -> None:
        """Parse single JSON tool call."""
        text = '{"name": "search", "arguments": {"query": "python"}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["query"] == "python"

    def test_parse_json_with_id(self) -> None:
        """Parse JSON tool call with ID."""
        text = '{"name": "search", "arguments": {"query": "test"}, "id": "call_abc"}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1
        assert calls[0].id == "call_abc"

    def test_parse_json_multiple_calls(self) -> None:
        """Parse multiple JSON tool calls."""
        text = """[
            {"name": "search", "arguments": {"query": "a"}},
            {"name": "search", "arguments": {"query": "b"}}
        ]"""
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 2

    def test_parse_json_no_match(self) -> None:
        """Parse returns empty when no match."""
        text = "no tool calls here"
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert calls == ()

    def test_parse_json_unknown_tool(self) -> None:
        """Parse ignores unknown tools."""
        text = '{"name": "unknown", "arguments": {}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert calls == ()

    def test_parse_regex_single_call(self) -> None:
        """Parse single regex tool call."""
        text = "search(query='python')"
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["query"] == "python"

    def test_parse_regex_multiple_args(self) -> None:
        """Parse regex tool call with multiple args."""
        text = "search(query='test', limit=10)"
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        assert calls[0].arguments["query"] == "test"

    def test_parse_regex_double_quotes(self) -> None:
        """Parse regex tool call with double quotes."""
        text = 'search(query="python")'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        assert calls[0].arguments["query"] == "python"

    def test_parse_regex_numeric_arg(self) -> None:
        """Parse regex tool call with numeric arg."""
        text = "calculate(value=42)"
        calls = parse_tool_calls(text, ("calculate",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        assert calls[0].arguments["value"] == 42

    def test_parse_regex_float_arg(self) -> None:
        """Parse regex tool call with float arg."""
        text = "calculate(value=3.14)"
        calls = parse_tool_calls(text, ("calculate",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        assert calls[0].arguments["value"] == pytest.approx(3.14)

    def test_parse_structured_json(self) -> None:
        """Parse structured falls back to JSON."""
        text = '{"name": "search", "arguments": {"query": "test"}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.STRUCTURED)
        assert len(calls) == 1

    def test_parse_structured_regex(self) -> None:
        """Parse structured falls back to regex when JSON fails."""
        text = "search(query='test')"
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.STRUCTURED)
        assert len(calls) == 1

    def test_parse_embedded_json(self) -> None:
        """Parse JSON embedded in text."""
        text = 'Some text {"name": "search", "arguments": {"q": "test"}} more text'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1


class TestValidateToolArguments:
    """Tests for validate_tool_arguments function."""

    def test_valid_arguments(self) -> None:
        """Valid arguments pass."""
        tool = create_tool_definition(
            name="search",
            description="Search",
            parameters={"query": "string"},
            required=("query",),
        )
        call = create_tool_call("search", {"query": "test"}, "call_001")
        is_valid, errors = validate_tool_arguments(call, tool)
        assert is_valid is True
        assert errors == ()

    def test_missing_required(self) -> None:
        """Missing required parameter fails."""
        tool = create_tool_definition(
            name="search",
            description="Search",
            parameters={"query": "string"},
            required=("query",),
        )
        call = create_tool_call("search", {}, "call_001")
        is_valid, errors = validate_tool_arguments(call, tool)
        assert is_valid is False
        assert "missing required parameter: query" in errors

    def test_unknown_parameter(self) -> None:
        """Unknown parameter fails."""
        tool = create_tool_definition(
            name="search",
            description="Search",
            parameters={"query": "string"},
            required=(),
        )
        call = create_tool_call("search", {"unknown": "value"}, "call_001")
        is_valid, errors = validate_tool_arguments(call, tool)
        assert is_valid is False
        assert "unknown parameter: unknown" in errors

    def test_none_call_raises(self) -> None:
        """None call raises ValueError."""
        tool = create_tool_definition("search", "Search")
        with pytest.raises(ValueError, match="call cannot be None"):
            validate_tool_arguments(None, tool)  # type: ignore[arg-type]

    def test_none_tool_raises(self) -> None:
        """None tool raises ValueError."""
        call = create_tool_call("search", {}, "call_001")
        with pytest.raises(ValueError, match="tool cannot be None"):
            validate_tool_arguments(call, None)  # type: ignore[arg-type]

    def test_name_mismatch_raises(self) -> None:
        """Name mismatch raises ValueError."""
        tool = create_tool_definition("search", "Search")
        call = create_tool_call("other", {}, "call_001")
        with pytest.raises(ValueError, match="tool name mismatch"):
            validate_tool_arguments(call, tool)


class TestExecuteToolCalls:
    """Tests for execute_tool_calls function."""

    def test_execute_single_call(self) -> None:
        """Execute single tool call."""

        def search_handler(query: str) -> str:
            return f"Results for: {query}"

        handlers = {"search": search_handler}
        call = create_tool_call("search", {"query": "python"}, "call_001")
        results = execute_tool_calls((call,), handlers)
        assert len(results) == 1
        assert results[0].output == "Results for: python"
        assert results[0].error is None

    def test_execute_multiple_calls(self) -> None:
        """Execute multiple tool calls."""

        def search_handler(query: str) -> str:
            return f"Results for: {query}"

        handlers = {"search": search_handler}
        calls = (
            create_tool_call("search", {"query": "a"}, "call_001"),
            create_tool_call("search", {"query": "b"}, "call_002"),
        )
        results = execute_tool_calls(calls, handlers)
        assert len(results) == 2
        assert "a" in results[0].output
        assert "b" in results[1].output

    def test_execute_unknown_tool(self) -> None:
        """Execute unknown tool returns error."""
        handlers: dict = {}
        call = create_tool_call("unknown", {}, "call_001")
        results = execute_tool_calls((call,), handlers)
        assert len(results) == 1
        assert results[0].error is not None
        assert "No handler for tool" in results[0].error

    def test_execute_handler_exception(self) -> None:
        """Handler exception is captured."""

        def failing_handler() -> str:
            raise RuntimeError("Something went wrong")

        handlers = {"fail": failing_handler}
        call = create_tool_call("fail", {}, "call_001")
        results = execute_tool_calls((call,), handlers)
        assert len(results) == 1
        assert results[0].error is not None
        assert "Something went wrong" in results[0].error

    def test_execute_parallel_mode(self) -> None:
        """Execute in parallel mode (falls back to sequential)."""

        def handler(x: int) -> str:
            return str(x * 2)

        handlers = {"calc": handler}
        call = create_tool_call("calc", {"x": 5}, "call_001")
        results = execute_tool_calls((call,), handlers, ExecutionMode.PARALLEL)
        assert len(results) == 1
        assert results[0].output == "10"


class TestFormatToolStats:
    """Tests for format_tool_stats function."""

    def test_format_basic_stats(self) -> None:
        """Format basic stats."""
        stats = create_tool_stats(total_calls=5, successful_calls=4, failed_calls=1)
        result = format_tool_stats(stats)
        assert "Total calls: 5" in result
        assert "Successful: 4" in result
        assert "Failed: 1" in result

    def test_format_with_time(self) -> None:
        """Format stats with time."""
        stats = create_tool_stats(total_time_ms=150.5)
        result = format_tool_stats(stats)
        assert "150.5ms" in result

    def test_format_with_calls_by_tool(self) -> None:
        """Format stats with calls by tool."""
        stats = create_tool_stats(calls_by_tool={"search": 3, "calc": 2})
        result = format_tool_stats(stats)
        assert "Calls by tool:" in result
        assert "search: 3" in result
        assert "calc: 2" in result

    def test_format_none_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_tool_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedToolConfig:
    """Tests for get_recommended_tool_config function."""

    def test_agent_task(self) -> None:
        """Get config for agent task."""
        config = get_recommended_tool_config("agent")
        assert config.execution_mode == ExecutionMode.AUTO
        assert config.max_calls == 20

    def test_assistant_task(self) -> None:
        """Get config for assistant task."""
        config = get_recommended_tool_config("assistant")
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.max_calls == 10

    def test_extraction_task(self) -> None:
        """Get config for extraction task."""
        config = get_recommended_tool_config("extraction")
        assert config.format == ToolFormat.JSON_SCHEMA
        assert config.parsing_strategy == ParsingStrategy.STRUCTURED
        assert config.execution_mode == ExecutionMode.PARALLEL

    def test_code_task(self) -> None:
        """Get config for code task."""
        config = get_recommended_tool_config("code")
        assert config.format == ToolFormat.JSON_SCHEMA
        assert config.execution_mode == ExecutionMode.SEQUENTIAL

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_tool_config("invalid")

    @pytest.mark.parametrize("task", ["agent", "assistant", "extraction", "code"])
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All valid tasks return valid configs."""
        config = get_recommended_tool_config(task)
        assert isinstance(config, ToolConfig)
        validate_tool_config(config)


class TestAdditionalParsing:
    """Additional tests for parsing edge cases."""

    def test_parse_llm_strategy_fallback(self) -> None:
        """LLM strategy falls back to JSON parsing."""
        text = '{"name": "search", "arguments": {"query": "test"}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.LLM)
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_parse_json_entire_text_single_object(self) -> None:
        """Parse entire text as single JSON object."""
        # This text can be parsed directly as JSON
        text = '{"name": "search", "arguments": {"query": "test"}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1

    def test_parse_json_entire_text_array(self) -> None:
        """Parse entire text as JSON array."""
        text = '[{"name": "search", "arguments": {"q": "a"}}]'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1

    def test_extract_json_with_escaped_quotes(self) -> None:
        """JSON extraction handles escaped quotes."""
        text = r'{"name": "search", "arguments": {"query": "hello \"world\""}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1

    def test_extract_json_unbalanced_braces(self) -> None:
        """JSON extraction handles unbalanced braces gracefully."""
        text = '{"name": "search" incomplete'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 0

    def test_parse_regex_with_conflict(self) -> None:
        """Regex parsing handles numeric conflicts with string args."""
        text = "calc(value=42, label='test')"
        calls = parse_tool_calls(text, ("calc",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        # String arg should be found, numeric should be int
        assert calls[0].arguments["label"] == "test"
        assert calls[0].arguments["value"] == 42

    def test_parse_regex_only_numeric_args(self) -> None:
        """Regex parsing with only numeric arguments."""
        text = "calc(x=100, y=200)"
        calls = parse_tool_calls(text, ("calc",), ParsingStrategy.REGEX)
        assert len(calls) == 1
        assert calls[0].arguments["x"] == 100
        assert calls[0].arguments["y"] == 200

    def test_parse_json_fallback_to_entire_text(self) -> None:
        """JSON parsing when extracted objects fail, try entire text."""
        # JSON that doesn't have nested objects but parses as whole
        text = '{"name": "search", "arguments": {}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1

    def test_parse_json_array_multiple_valid_tools(self) -> None:
        """Parse JSON array with multiple tool calls."""
        text = """[
            {"name": "search", "arguments": {"q": "a"}},
            {"name": "calc", "arguments": {"x": 1}},
            {"name": "unknown", "arguments": {}}
        ]"""
        calls = parse_tool_calls(text, ("search", "calc"), ParsingStrategy.JSON)
        assert len(calls) == 2

    def test_parse_json_with_nested_deep_braces(self) -> None:
        """Parse JSON with deeply nested braces."""
        text = '{"name": "search", "arguments": {"data": {"nested": "value"}}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1
        assert "data" in calls[0].arguments


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_tool_with_many_parameters(self) -> None:
        """Tool with many parameters."""
        params = {f"param{i}": "string" for i in range(20)}
        required = tuple(f"param{i}" for i in range(5))
        tool = create_tool_definition(
            name="complex_tool",
            description="A complex tool",
            parameters=params,
            required=required,
        )
        assert len(tool.parameters) == 20
        assert len(tool.required) == 5

    def test_tool_name_with_numbers(self) -> None:
        """Tool name with numbers."""
        tool = create_tool_definition("tool123", "Tool 123")
        assert tool.name == "tool123"

    def test_parse_malformed_json(self) -> None:
        """Parse handles malformed JSON gracefully."""
        text = '{"name": "search", "arguments": {invalid}}'
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert calls == ()

    def test_execute_empty_calls(self) -> None:
        """Execute empty calls tuple."""
        results = execute_tool_calls((), {})
        assert results == ()

    def test_format_empty_tools_xml(self) -> None:
        """Format empty tools in XML."""
        result = format_tools_for_prompt((), ToolFormat.XML)
        assert result == ""

    def test_stats_all_zeros(self) -> None:
        """Stats with all zeros."""
        stats = create_tool_stats()
        result = format_tool_stats(stats)
        assert "Total calls: 0" in result

    def test_parse_json_in_markdown(self) -> None:
        """Parse JSON inside markdown code block."""
        text = """```json
{"name": "search", "arguments": {"query": "test"}}
```"""
        calls = parse_tool_calls(text, ("search",), ParsingStrategy.JSON)
        assert len(calls) == 1

    def test_handler_returns_non_string(self) -> None:
        """Handler returning non-string is converted."""

        def handler() -> int:
            return 42

        handlers = {"calc": handler}
        call = create_tool_call("calc", {}, "call_001")
        results = execute_tool_calls((call,), handlers)
        assert results[0].output == "42"

    def test_xml_format_required_parameter(self) -> None:
        """XML format shows required parameter correctly."""
        tool = create_tool_definition(
            name="search",
            description="Search",
            parameters={"query": "string", "limit": "number"},
            required=("query",),
        )
        result = format_tools_for_prompt((tool,), ToolFormat.XML)
        assert 'required="true"' in result
        assert 'required="false"' in result
