"""Tests for generation.prompts module."""

from __future__ import annotations

import pytest

from hf_gtc.generation.prompts import (
    VALID_OUTPUT_FORMATS,
    VALID_PROMPT_ROLES,
    VALID_PROMPT_STYLES,
    VALID_SELECTION_STRATEGIES,
    ChainOfThoughtConfig,
    ExampleSelectionStrategy,
    FewShotConfig,
    FewShotExample,
    OutputFormat,
    PromptConfig,
    PromptMessage,
    PromptRole,
    PromptStats,
    PromptStyle,
    PromptTemplate,
    SystemPromptConfig,
    build_conversation_prompt,
    create_cot_config,
    create_few_shot_config,
    create_few_shot_example,
    create_prompt_config,
    create_prompt_message,
    create_prompt_stats,
    create_prompt_template,
    create_system_prompt_config,
    estimate_prompt_tokens,
    format_cot_prompt,
    format_few_shot_prompt,
    format_messages,
    format_prompt,
    format_prompt_stats,
    format_system_prompt,
    get_output_format,
    get_prompt_role,
    get_prompt_style,
    get_recommended_prompt_config,
    get_selection_strategy,
    list_output_formats,
    list_prompt_roles,
    list_prompt_styles,
    list_selection_strategies,
    truncate_prompt,
    validate_cot_config,
    validate_few_shot_config,
    validate_prompt_template,
    validate_system_prompt_config,
)


class TestPromptStyle:
    """Tests for PromptStyle enum."""

    def test_all_styles_have_values(self) -> None:
        """All styles have string values."""
        for style in PromptStyle:
            assert isinstance(style.value, str)

    def test_zero_shot_value(self) -> None:
        """Zero shot has correct value."""
        assert PromptStyle.ZERO_SHOT.value == "zero_shot"

    def test_few_shot_value(self) -> None:
        """Few shot has correct value."""
        assert PromptStyle.FEW_SHOT.value == "few_shot"

    def test_chain_of_thought_value(self) -> None:
        """Chain of thought has correct value."""
        assert PromptStyle.CHAIN_OF_THOUGHT.value == "chain_of_thought"

    def test_tree_of_thought_value(self) -> None:
        """Tree of thought has correct value."""
        assert PromptStyle.TREE_OF_THOUGHT.value == "tree_of_thought"

    def test_self_consistency_value(self) -> None:
        """Self consistency has correct value."""
        assert PromptStyle.SELF_CONSISTENCY.value == "self_consistency"

    def test_valid_styles_frozenset(self) -> None:
        """VALID_PROMPT_STYLES is a frozenset."""
        assert isinstance(VALID_PROMPT_STYLES, frozenset)
        assert len(VALID_PROMPT_STYLES) == 5


class TestPromptRole:
    """Tests for PromptRole enum."""

    def test_all_roles_have_values(self) -> None:
        """All roles have string values."""
        for role in PromptRole:
            assert isinstance(role.value, str)

    def test_system_value(self) -> None:
        """System has correct value."""
        assert PromptRole.SYSTEM.value == "system"

    def test_user_value(self) -> None:
        """User has correct value."""
        assert PromptRole.USER.value == "user"

    def test_assistant_value(self) -> None:
        """Assistant has correct value."""
        assert PromptRole.ASSISTANT.value == "assistant"

    def test_valid_roles_frozenset(self) -> None:
        """VALID_PROMPT_ROLES is a frozenset."""
        assert isinstance(VALID_PROMPT_ROLES, frozenset)
        assert len(VALID_PROMPT_ROLES) == 3


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_have_values(self) -> None:
        """All formats have string values."""
        for fmt in OutputFormat:
            assert isinstance(fmt.value, str)

    def test_text_value(self) -> None:
        """Text has correct value."""
        assert OutputFormat.TEXT.value == "text"

    def test_json_value(self) -> None:
        """JSON has correct value."""
        assert OutputFormat.JSON.value == "json"

    def test_markdown_value(self) -> None:
        """Markdown has correct value."""
        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_code_value(self) -> None:
        """Code has correct value."""
        assert OutputFormat.CODE.value == "code"

    def test_list_value(self) -> None:
        """List has correct value."""
        assert OutputFormat.LIST.value == "list"

    def test_valid_formats_frozenset(self) -> None:
        """VALID_OUTPUT_FORMATS is a frozenset."""
        assert isinstance(VALID_OUTPUT_FORMATS, frozenset)
        assert len(VALID_OUTPUT_FORMATS) == 5


class TestExampleSelectionStrategy:
    """Tests for ExampleSelectionStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ExampleSelectionStrategy:
            assert isinstance(strategy.value, str)

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert ExampleSelectionStrategy.RANDOM.value == "random"

    def test_similarity_value(self) -> None:
        """Similarity has correct value."""
        assert ExampleSelectionStrategy.SIMILARITY.value == "similarity"

    def test_diversity_value(self) -> None:
        """Diversity has correct value."""
        assert ExampleSelectionStrategy.DIVERSITY.value == "diversity"

    def test_fixed_value(self) -> None:
        """Fixed has correct value."""
        assert ExampleSelectionStrategy.FIXED.value == "fixed"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_SELECTION_STRATEGIES is a frozenset."""
        assert isinstance(VALID_SELECTION_STRATEGIES, frozenset)
        assert len(VALID_SELECTION_STRATEGIES) == 4


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_create_template(self) -> None:
        """Create prompt template."""
        template = PromptTemplate(
            template="Hello, {name}!",
            variables=("name",),
            description="Greeting",
            default_values={},
        )
        assert template.template == "Hello, {name}!"
        assert template.variables == ("name",)

    def test_template_is_frozen(self) -> None:
        """Template is immutable."""
        template = PromptTemplate("Hello", (), "", {})
        with pytest.raises(AttributeError):
            template.template = "Goodbye"  # type: ignore[misc]


class TestFewShotExample:
    """Tests for FewShotExample dataclass."""

    def test_create_example(self) -> None:
        """Create few-shot example."""
        example = FewShotExample(
            input_text="2+2",
            output_text="4",
            explanation="Basic math",
        )
        assert example.input_text == "2+2"
        assert example.output_text == "4"

    def test_example_is_frozen(self) -> None:
        """Example is immutable."""
        example = FewShotExample("a", "b", "")
        with pytest.raises(AttributeError):
            example.input_text = "c"  # type: ignore[misc]


class TestFewShotConfig:
    """Tests for FewShotConfig dataclass."""

    def test_create_config(self) -> None:
        """Create few-shot config."""
        config = FewShotConfig(
            examples=(FewShotExample("a", "b", ""),),
            num_examples=1,
            selection_strategy=ExampleSelectionStrategy.FIXED,
            example_separator="\n",
            input_prefix="Q: ",
            output_prefix="A: ",
        )
        assert config.num_examples == 1
        assert len(config.examples) == 1

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FewShotConfig(
            (FewShotExample("a", "b", ""),),
            1,
            ExampleSelectionStrategy.FIXED,
            "\n",
            "Q:",
            "A:",
        )
        with pytest.raises(AttributeError):
            config.num_examples = 2  # type: ignore[misc]


class TestChainOfThoughtConfig:
    """Tests for ChainOfThoughtConfig dataclass."""

    def test_create_config(self) -> None:
        """Create chain-of-thought config."""
        config = ChainOfThoughtConfig(
            enable_reasoning=True,
            reasoning_prefix="Think:",
            answer_prefix="Answer:",
            step_marker="Step {n}:",
            max_steps=5,
        )
        assert config.enable_reasoning is True
        assert config.max_steps == 5

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ChainOfThoughtConfig(True, "Think:", "Answer:", "Step:", 5)
        with pytest.raises(AttributeError):
            config.max_steps = 10  # type: ignore[misc]


class TestSystemPromptConfig:
    """Tests for SystemPromptConfig dataclass."""

    def test_create_config(self) -> None:
        """Create system prompt config."""
        config = SystemPromptConfig(
            persona="You are helpful.",
            constraints=("Be concise",),
            output_format=OutputFormat.TEXT,
            format_instructions="",
            examples_in_system=False,
        )
        assert config.persona == "You are helpful."
        assert config.output_format == OutputFormat.TEXT

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SystemPromptConfig("Helpful", (), OutputFormat.TEXT, "", False)
        with pytest.raises(AttributeError):
            config.persona = "Different"  # type: ignore[misc]


class TestPromptMessage:
    """Tests for PromptMessage dataclass."""

    def test_create_message(self) -> None:
        """Create prompt message."""
        msg = PromptMessage(
            role=PromptRole.USER,
            content="Hello!",
        )
        assert msg.role == PromptRole.USER
        assert msg.content == "Hello!"

    def test_message_is_frozen(self) -> None:
        """Message is immutable."""
        msg = PromptMessage(PromptRole.USER, "Hello")
        with pytest.raises(AttributeError):
            msg.content = "Goodbye"  # type: ignore[misc]


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_create_config(self) -> None:
        """Create prompt config."""
        config = PromptConfig(
            style=PromptStyle.ZERO_SHOT,
            template=None,
            system_config=None,
            few_shot_config=None,
            cot_config=None,
            max_tokens=2048,
        )
        assert config.style == PromptStyle.ZERO_SHOT
        assert config.max_tokens == 2048

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PromptConfig(PromptStyle.ZERO_SHOT, None, None, None, None, 2048)
        with pytest.raises(AttributeError):
            config.max_tokens = 4096  # type: ignore[misc]


class TestPromptStats:
    """Tests for PromptStats dataclass."""

    def test_create_stats(self) -> None:
        """Create prompt stats."""
        stats = PromptStats(
            total_chars=100,
            estimated_tokens=25,
            num_examples=3,
            has_system_prompt=True,
            has_cot=False,
        )
        assert stats.total_chars == 100
        assert stats.estimated_tokens == 25

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = PromptStats(100, 25, 0, False, False)
        with pytest.raises(AttributeError):
            stats.total_chars = 200  # type: ignore[misc]


class TestValidatePromptTemplate:
    """Tests for validate_prompt_template function."""

    def test_valid_template(self) -> None:
        """Valid template passes validation."""
        template = PromptTemplate("Hello, {name}!", ("name",), "", {})
        validate_prompt_template(template)

    def test_empty_template_raises(self) -> None:
        """Empty template raises ValueError."""
        template = PromptTemplate("", (), "", {})
        with pytest.raises(ValueError, match="template cannot be empty"):
            validate_prompt_template(template)

    def test_whitespace_template_raises(self) -> None:
        """Whitespace template raises ValueError."""
        template = PromptTemplate("   ", (), "", {})
        with pytest.raises(ValueError, match="template cannot be empty"):
            validate_prompt_template(template)

    def test_missing_variable_declaration_raises(self) -> None:
        """Missing variable declaration raises ValueError."""
        template = PromptTemplate("Hello, {name}!", (), "", {})
        with pytest.raises(ValueError, match="variables missing"):
            validate_prompt_template(template)

    def test_extra_variable_declaration_raises(self) -> None:
        """Extra variable declaration raises ValueError."""
        template = PromptTemplate("Hello!", ("name",), "", {})
        with pytest.raises(ValueError, match="declared variables not in template"):
            validate_prompt_template(template)


class TestValidateFewShotConfig:
    """Tests for validate_few_shot_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FewShotConfig(
            (FewShotExample("a", "b", ""),),
            1,
            ExampleSelectionStrategy.FIXED,
            "\n",
            "Q:",
            "A:",
        )
        validate_few_shot_config(config)

    def test_empty_examples_raises(self) -> None:
        """Empty examples raises ValueError."""
        config = FewShotConfig(
            (),
            1,
            ExampleSelectionStrategy.FIXED,
            "\n",
            "Q:",
            "A:",
        )
        with pytest.raises(ValueError, match="examples cannot be empty"):
            validate_few_shot_config(config)

    def test_zero_num_examples_raises(self) -> None:
        """Zero num_examples raises ValueError."""
        config = FewShotConfig(
            (FewShotExample("a", "b", ""),),
            0,
            ExampleSelectionStrategy.FIXED,
            "\n",
            "Q:",
            "A:",
        )
        with pytest.raises(ValueError, match="num_examples must be positive"):
            validate_few_shot_config(config)

    def test_num_examples_exceeds_available_raises(self) -> None:
        """num_examples > available raises ValueError."""
        config = FewShotConfig(
            (FewShotExample("a", "b", ""),),
            5,
            ExampleSelectionStrategy.FIXED,
            "\n",
            "Q:",
            "A:",
        )
        with pytest.raises(ValueError, match="cannot exceed available"):
            validate_few_shot_config(config)


class TestValidateCotConfig:
    """Tests for validate_cot_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ChainOfThoughtConfig(True, "Think:", "Answer:", "Step:", 5)
        validate_cot_config(config)

    def test_empty_reasoning_prefix_raises(self) -> None:
        """Empty reasoning_prefix raises ValueError."""
        config = ChainOfThoughtConfig(True, "", "Answer:", "Step:", 5)
        with pytest.raises(ValueError, match="reasoning_prefix cannot be empty"):
            validate_cot_config(config)

    def test_empty_answer_prefix_raises(self) -> None:
        """Empty answer_prefix raises ValueError."""
        config = ChainOfThoughtConfig(True, "Think:", "", "Step:", 5)
        with pytest.raises(ValueError, match="answer_prefix cannot be empty"):
            validate_cot_config(config)

    def test_zero_max_steps_raises(self) -> None:
        """Zero max_steps raises ValueError."""
        config = ChainOfThoughtConfig(True, "Think:", "Answer:", "Step:", 0)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            validate_cot_config(config)

    def test_disabled_reasoning_allows_empty_prefix(self) -> None:
        """Disabled reasoning allows empty prefix."""
        config = ChainOfThoughtConfig(False, "", "", "Step:", 5)
        validate_cot_config(config)


class TestValidateSystemPromptConfig:
    """Tests for validate_system_prompt_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SystemPromptConfig(
            "You are helpful.", (), OutputFormat.TEXT, "", False
        )
        validate_system_prompt_config(config)

    def test_empty_persona_raises(self) -> None:
        """Empty persona raises ValueError."""
        config = SystemPromptConfig("", (), OutputFormat.TEXT, "", False)
        with pytest.raises(ValueError, match="persona cannot be empty"):
            validate_system_prompt_config(config)

    def test_whitespace_persona_raises(self) -> None:
        """Whitespace persona raises ValueError."""
        config = SystemPromptConfig("   ", (), OutputFormat.TEXT, "", False)
        with pytest.raises(ValueError, match="persona cannot be empty"):
            validate_system_prompt_config(config)


class TestCreatePromptTemplate:
    """Tests for create_prompt_template function."""

    def test_simple_template(self) -> None:
        """Create simple template."""
        template = create_prompt_template("Hello, {name}!")
        assert template.variables == ("name",)
        assert template.template == "Hello, {name}!"

    def test_multiple_variables(self) -> None:
        """Create template with multiple variables."""
        template = create_prompt_template("Hi {name}, your {item} is ready.")
        assert "item" in template.variables
        assert "name" in template.variables

    def test_with_description(self) -> None:
        """Create template with description."""
        template = create_prompt_template("Hello!", description="Greeting")
        assert template.description == "Greeting"

    def test_with_defaults(self) -> None:
        """Create template with default values."""
        template = create_prompt_template(
            "Hello, {name}!",
            default_values={"name": "World"},
        )
        assert template.default_values == {"name": "World"}

    def test_empty_template_raises(self) -> None:
        """Empty template raises ValueError."""
        with pytest.raises(ValueError, match="template cannot be empty"):
            create_prompt_template("")


class TestCreateFewShotExample:
    """Tests for create_few_shot_example function."""

    def test_simple_example(self) -> None:
        """Create simple example."""
        example = create_few_shot_example("2+2", "4")
        assert example.input_text == "2+2"
        assert example.output_text == "4"
        assert example.explanation == ""

    def test_with_explanation(self) -> None:
        """Create example with explanation."""
        example = create_few_shot_example("2+2", "4", "Addition")
        assert example.explanation == "Addition"

    def test_empty_input_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="input_text cannot be empty"):
            create_few_shot_example("", "output")

    def test_empty_output_raises(self) -> None:
        """Empty output raises ValueError."""
        with pytest.raises(ValueError, match="output_text cannot be empty"):
            create_few_shot_example("input", "")


class TestCreateFewShotConfig:
    """Tests for create_few_shot_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        examples = (create_few_shot_example("a", "b"),)
        config = create_few_shot_config(examples)
        assert config.num_examples == 1
        assert config.selection_strategy == ExampleSelectionStrategy.FIXED

    def test_custom_num_examples(self) -> None:
        """Create config with custom num_examples."""
        examples = (
            create_few_shot_example("a", "b"),
            create_few_shot_example("c", "d"),
        )
        config = create_few_shot_config(examples, num_examples=1)
        assert config.num_examples == 1

    def test_string_strategy(self) -> None:
        """Create config with string strategy."""
        examples = (create_few_shot_example("a", "b"),)
        config = create_few_shot_config(examples, selection_strategy="random")
        assert config.selection_strategy == ExampleSelectionStrategy.RANDOM

    def test_empty_examples_raises(self) -> None:
        """Empty examples raises ValueError."""
        with pytest.raises(ValueError, match="examples cannot be empty"):
            create_few_shot_config(())


class TestCreateCotConfig:
    """Tests for create_cot_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_cot_config()
        assert config.enable_reasoning is True
        assert config.max_steps == 10

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_cot_config(max_steps=5, reasoning_prefix="Think:")
        assert config.max_steps == 5
        assert config.reasoning_prefix == "Think:"

    def test_zero_max_steps_raises(self) -> None:
        """Zero max_steps raises ValueError."""
        with pytest.raises(ValueError, match="max_steps must be positive"):
            create_cot_config(max_steps=0)


class TestCreateSystemPromptConfig:
    """Tests for create_system_prompt_config function."""

    def test_simple_config(self) -> None:
        """Create simple config."""
        config = create_system_prompt_config("You are helpful.")
        assert config.persona == "You are helpful."
        assert config.output_format == OutputFormat.TEXT

    def test_with_constraints(self) -> None:
        """Create config with constraints."""
        config = create_system_prompt_config(
            "You are a coder.",
            constraints=("Use Python", "Be concise"),
        )
        assert len(config.constraints) == 2

    def test_string_output_format(self) -> None:
        """Create config with string output format."""
        config = create_system_prompt_config("Coder", output_format="code")
        assert config.output_format == OutputFormat.CODE

    def test_empty_persona_raises(self) -> None:
        """Empty persona raises ValueError."""
        with pytest.raises(ValueError, match="persona cannot be empty"):
            create_system_prompt_config("")


class TestCreatePromptConfig:
    """Tests for create_prompt_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_prompt_config()
        assert config.style == PromptStyle.ZERO_SHOT
        assert config.max_tokens == 4096

    def test_string_style(self) -> None:
        """Create config with string style."""
        config = create_prompt_config(style="few_shot")
        assert config.style == PromptStyle.FEW_SHOT

    def test_zero_max_tokens_raises(self) -> None:
        """Zero max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            create_prompt_config(max_tokens=0)


class TestCreatePromptMessage:
    """Tests for create_prompt_message function."""

    def test_simple_message(self) -> None:
        """Create simple message."""
        msg = create_prompt_message("user", "Hello!")
        assert msg.role == PromptRole.USER
        assert msg.content == "Hello!"

    def test_string_role(self) -> None:
        """Create message with string role."""
        msg = create_prompt_message("system", "You are helpful.")
        assert msg.role == PromptRole.SYSTEM

    def test_empty_content_raises(self) -> None:
        """Empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            create_prompt_message("user", "")


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_prompt_styles_sorted(self) -> None:
        """Returns sorted list."""
        styles = list_prompt_styles()
        assert styles == sorted(styles)
        assert "zero_shot" in styles

    def test_list_prompt_roles_sorted(self) -> None:
        """Returns sorted list."""
        roles = list_prompt_roles()
        assert roles == sorted(roles)
        assert "user" in roles

    def test_list_output_formats_sorted(self) -> None:
        """Returns sorted list."""
        formats = list_output_formats()
        assert formats == sorted(formats)
        assert "json" in formats

    def test_list_selection_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_selection_strategies()
        assert strategies == sorted(strategies)
        assert "random" in strategies


class TestGetPromptStyle:
    """Tests for get_prompt_style function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("zero_shot", PromptStyle.ZERO_SHOT),
            ("few_shot", PromptStyle.FEW_SHOT),
            ("chain_of_thought", PromptStyle.CHAIN_OF_THOUGHT),
            ("tree_of_thought", PromptStyle.TREE_OF_THOUGHT),
            ("self_consistency", PromptStyle.SELF_CONSISTENCY),
        ],
    )
    def test_all_styles(self, name: str, expected: PromptStyle) -> None:
        """Test all valid styles."""
        assert get_prompt_style(name) == expected

    def test_invalid_style_raises(self) -> None:
        """Invalid style raises ValueError."""
        with pytest.raises(ValueError, match="style must be one of"):
            get_prompt_style("invalid")


class TestGetPromptRole:
    """Tests for get_prompt_role function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("system", PromptRole.SYSTEM),
            ("user", PromptRole.USER),
            ("assistant", PromptRole.ASSISTANT),
        ],
    )
    def test_all_roles(self, name: str, expected: PromptRole) -> None:
        """Test all valid roles."""
        assert get_prompt_role(name) == expected

    def test_invalid_role_raises(self) -> None:
        """Invalid role raises ValueError."""
        with pytest.raises(ValueError, match="role must be one of"):
            get_prompt_role("invalid")


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("text", OutputFormat.TEXT),
            ("json", OutputFormat.JSON),
            ("markdown", OutputFormat.MARKDOWN),
            ("code", OutputFormat.CODE),
            ("list", OutputFormat.LIST),
        ],
    )
    def test_all_formats(self, name: str, expected: OutputFormat) -> None:
        """Test all valid formats."""
        assert get_output_format(name) == expected

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be one of"):
            get_output_format("invalid")


class TestGetSelectionStrategy:
    """Tests for get_selection_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("random", ExampleSelectionStrategy.RANDOM),
            ("similarity", ExampleSelectionStrategy.SIMILARITY),
            ("diversity", ExampleSelectionStrategy.DIVERSITY),
            ("fixed", ExampleSelectionStrategy.FIXED),
        ],
    )
    def test_all_strategies(
        self, name: str, expected: ExampleSelectionStrategy
    ) -> None:
        """Test all valid strategies."""
        assert get_selection_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_selection_strategy("invalid")


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_simple_format(self) -> None:
        """Format simple template."""
        template = create_prompt_template("Hello, {name}!")
        result = format_prompt(template, name="World")
        assert result == "Hello, World!"

    def test_multiple_variables(self) -> None:
        """Format template with multiple variables."""
        template = create_prompt_template("Hi {name}, your {item} is ready.")
        result = format_prompt(template, name="Alice", item="order")
        assert "Alice" in result
        assert "order" in result

    def test_uses_default_values(self) -> None:
        """Format uses default values."""
        template = create_prompt_template(
            "Hello, {name}!",
            default_values={"name": "World"},
        )
        result = format_prompt(template)
        assert result == "Hello, World!"

    def test_override_default_values(self) -> None:
        """Format can override default values."""
        template = create_prompt_template(
            "Hello, {name}!",
            default_values={"name": "World"},
        )
        result = format_prompt(template, name="Alice")
        assert result == "Hello, Alice!"

    def test_missing_variable_raises(self) -> None:
        """Missing variable raises ValueError."""
        template = create_prompt_template("Hello, {name}!")
        with pytest.raises(ValueError, match="missing required variable"):
            format_prompt(template)


class TestFormatFewShotPrompt:
    """Tests for format_few_shot_prompt function."""

    def test_basic_format(self) -> None:
        """Format basic few-shot prompt."""
        examples = (
            create_few_shot_example("2+2", "4"),
            create_few_shot_example("3+3", "6"),
        )
        config = create_few_shot_config(examples)
        result = format_few_shot_prompt(config, "5+5")
        assert "Input: 2+2" in result
        assert "Output: 4" in result
        assert "Input: 5+5" in result

    def test_without_query(self) -> None:
        """Format without including query."""
        examples = (create_few_shot_example("a", "b"),)
        config = create_few_shot_config(examples)
        result = format_few_shot_prompt(config, "query", include_query=False)
        assert "query" not in result

    def test_respects_num_examples(self) -> None:
        """Format respects num_examples limit."""
        examples = (
            create_few_shot_example("1", "a"),
            create_few_shot_example("2", "b"),
            create_few_shot_example("3", "c"),
        )
        config = create_few_shot_config(examples, num_examples=1)
        result = format_few_shot_prompt(config, "q")
        assert "Input: 1" in result
        assert "Input: 2" not in result


class TestFormatSystemPrompt:
    """Tests for format_system_prompt function."""

    def test_basic_format(self) -> None:
        """Format basic system prompt."""
        config = create_system_prompt_config("You are helpful.")
        result = format_system_prompt(config)
        assert "helpful" in result

    def test_with_constraints(self) -> None:
        """Format with constraints."""
        config = create_system_prompt_config(
            "You are helpful.",
            constraints=("Be concise", "Be accurate"),
        )
        result = format_system_prompt(config)
        assert "Be concise" in result
        assert "Be accurate" in result

    def test_with_format_instructions(self) -> None:
        """Format with format instructions."""
        config = create_system_prompt_config(
            "You are helpful.",
            format_instructions="Respond in JSON.",
        )
        result = format_system_prompt(config)
        assert "JSON" in result


class TestFormatCotPrompt:
    """Tests for format_cot_prompt function."""

    def test_basic_format(self) -> None:
        """Format basic chain-of-thought prompt."""
        config = create_cot_config()
        result = format_cot_prompt(config, "What is 2+2?")
        assert "What is 2+2?" in result
        assert "step by step" in result

    def test_disabled_reasoning(self) -> None:
        """Format with disabled reasoning."""
        config = create_cot_config(enable_reasoning=False)
        result = format_cot_prompt(config, "Question")
        assert "Question" in result
        assert "step" not in result


class TestFormatMessages:
    """Tests for format_messages function."""

    def test_basic_format(self) -> None:
        """Format basic messages."""
        messages = (
            create_prompt_message("system", "You are helpful."),
            create_prompt_message("user", "Hello!"),
        )
        result = format_messages(messages)
        assert "[system]" in result
        assert "helpful" in result
        assert "[user]" in result
        assert "Hello!" in result

    def test_empty_messages(self) -> None:
        """Format empty messages."""
        result = format_messages(())
        assert result == ""


class TestEstimatePromptTokens:
    """Tests for estimate_prompt_tokens function."""

    def test_basic_estimate(self) -> None:
        """Estimate tokens for basic text."""
        result = estimate_prompt_tokens("Hello, world!")
        assert result == 3

    def test_custom_chars_per_token(self) -> None:
        """Estimate with custom chars_per_token."""
        result = estimate_prompt_tokens("12345678", chars_per_token=2.0)
        assert result == 4

    def test_empty_string(self) -> None:
        """Empty string returns 0."""
        result = estimate_prompt_tokens("")
        assert result == 0

    def test_zero_chars_per_token_raises(self) -> None:
        """Zero chars_per_token raises ValueError."""
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            estimate_prompt_tokens("test", chars_per_token=0)


class TestCreatePromptStats:
    """Tests for create_prompt_stats function."""

    def test_basic_stats(self) -> None:
        """Create basic stats."""
        stats = create_prompt_stats("Hello!")
        assert stats.total_chars == 6
        assert stats.estimated_tokens == 1

    def test_with_examples(self) -> None:
        """Create stats with examples."""
        stats = create_prompt_stats("Hello!", num_examples=3)
        assert stats.num_examples == 3

    def test_with_system_prompt(self) -> None:
        """Create stats with system prompt."""
        stats = create_prompt_stats("Hello!", has_system_prompt=True)
        assert stats.has_system_prompt is True

    def test_with_cot(self) -> None:
        """Create stats with chain-of-thought."""
        stats = create_prompt_stats("Hello!", has_cot=True)
        assert stats.has_cot is True


class TestGetRecommendedPromptConfig:
    """Tests for get_recommended_prompt_config function."""

    def test_qa_config(self) -> None:
        """Get config for QA task."""
        config = get_recommended_prompt_config("qa")
        assert config.style == PromptStyle.FEW_SHOT

    def test_summarization_config(self) -> None:
        """Get config for summarization task."""
        config = get_recommended_prompt_config("summarization")
        assert config.style == PromptStyle.ZERO_SHOT

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_prompt_config("classification")
        assert config.style == PromptStyle.FEW_SHOT

    def test_code_config(self) -> None:
        """Get config for code task."""
        config = get_recommended_prompt_config("code")
        assert config.style == PromptStyle.ZERO_SHOT

    def test_chat_config(self) -> None:
        """Get config for chat task."""
        config = get_recommended_prompt_config("chat")
        assert config.style == PromptStyle.ZERO_SHOT

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_prompt_config("invalid")


class TestFormatPromptStats:
    """Tests for format_prompt_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_prompt_stats("Hello!", num_examples=2)
        result = format_prompt_stats(stats)
        assert "Characters: 6" in result
        assert "Examples: 2" in result

    def test_contains_all_fields(self) -> None:
        """Formatted contains all fields."""
        stats = create_prompt_stats("Test")
        result = format_prompt_stats(stats)
        assert "Characters:" in result
        assert "Est. Tokens:" in result
        assert "Examples:" in result
        assert "System Prompt:" in result
        assert "Chain-of-Thought:" in result


class TestBuildConversationPrompt:
    """Tests for build_conversation_prompt function."""

    def test_with_system_prompt(self) -> None:
        """Build with system prompt."""
        system = create_system_prompt_config("You are helpful.")
        messages = (create_prompt_message("user", "Hello!"),)
        result = build_conversation_prompt(system, messages)
        assert len(result) == 2
        assert result[0].role == PromptRole.SYSTEM

    def test_without_system_prompt(self) -> None:
        """Build without system prompt."""
        messages = (create_prompt_message("user", "Hello!"),)
        result = build_conversation_prompt(None, messages)
        assert len(result) == 1
        assert result[0].role == PromptRole.USER

    def test_preserves_message_order(self) -> None:
        """Preserves message order."""
        messages = (
            create_prompt_message("user", "First"),
            create_prompt_message("assistant", "Second"),
            create_prompt_message("user", "Third"),
        )
        result = build_conversation_prompt(None, messages)
        assert result[0].content == "First"
        assert result[1].content == "Second"
        assert result[2].content == "Third"


class TestTruncatePrompt:
    """Tests for truncate_prompt function."""

    def test_no_truncation_needed(self) -> None:
        """No truncation when under limit."""
        result = truncate_prompt("Short", max_tokens=100)
        assert result == "Short"

    def test_truncates_long_text(self) -> None:
        """Truncates text over limit."""
        result = truncate_prompt("Hello, world!", max_tokens=2)
        assert len(result) < len("Hello, world!")
        assert result.endswith("...")

    def test_custom_marker(self) -> None:
        """Uses custom truncation marker."""
        result = truncate_prompt(
            "Hello, world!", max_tokens=2, truncation_marker="[...]"
        )
        assert result.endswith("[...]")

    def test_zero_max_tokens_raises(self) -> None:
        """Zero max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            truncate_prompt("test", max_tokens=0)

    def test_zero_chars_per_token_raises(self) -> None:
        """Zero chars_per_token raises ValueError."""
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            truncate_prompt("test", max_tokens=10, chars_per_token=0)
