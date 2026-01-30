"""Tests for generation.prompting module."""

from __future__ import annotations

import pytest

from hf_gtc.generation.prompting import (
    VALID_FEW_SHOT_STRATEGIES,
    VALID_PROMPT_FORMATS,
    VALID_REASONING_TYPES,
    CoTConfig,
    FewShotConfig,
    FewShotExample,
    FewShotStrategy,
    PromptConfig,
    PromptFormat,
    PromptStats,
    PromptTemplate,
    ReasoningType,
    add_cot_reasoning,
    build_complete_prompt,
    create_cot_config,
    create_few_shot_config,
    create_few_shot_example,
    create_prompt_config,
    create_prompt_stats,
    create_prompt_template,
    estimate_prompt_tokens,
    format_few_shot_examples,
    format_prompt,
    format_prompt_stats,
    get_few_shot_strategy,
    get_prompt_format,
    get_reasoning_type,
    get_recommended_prompt_config,
    list_few_shot_strategies,
    list_prompt_formats,
    list_reasoning_types,
    truncate_prompt,
    validate_cot_config,
    validate_few_shot_config,
    validate_prompt_config,
    validate_prompt_template,
)


class TestPromptFormat:
    """Tests for PromptFormat enum."""

    def test_all_formats_have_values(self) -> None:
        """All formats have string values."""
        for fmt in PromptFormat:
            assert isinstance(fmt.value, str)

    def test_plain_value(self) -> None:
        """Plain format has correct value."""
        assert PromptFormat.PLAIN.value == "plain"

    def test_chat_value(self) -> None:
        """Chat format has correct value."""
        assert PromptFormat.CHAT.value == "chat"

    def test_instruct_value(self) -> None:
        """Instruct format has correct value."""
        assert PromptFormat.INSTRUCT.value == "instruct"

    def test_chatml_value(self) -> None:
        """ChatML format has correct value."""
        assert PromptFormat.CHATML.value == "chatml"

    def test_alpaca_value(self) -> None:
        """Alpaca format has correct value."""
        assert PromptFormat.ALPACA.value == "alpaca"

    def test_llama2_value(self) -> None:
        """Llama2 format has correct value."""
        assert PromptFormat.LLAMA2.value == "llama2"

    def test_valid_formats_frozenset(self) -> None:
        """VALID_PROMPT_FORMATS is a frozenset of all values."""
        assert isinstance(VALID_PROMPT_FORMATS, frozenset)
        assert len(VALID_PROMPT_FORMATS) == 6


class TestFewShotStrategy:
    """Tests for FewShotStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in FewShotStrategy:
            assert isinstance(strategy.value, str)

    def test_random_value(self) -> None:
        """Random strategy has correct value."""
        assert FewShotStrategy.RANDOM.value == "random"

    def test_similar_value(self) -> None:
        """Similar strategy has correct value."""
        assert FewShotStrategy.SIMILAR.value == "similar"

    def test_diverse_value(self) -> None:
        """Diverse strategy has correct value."""
        assert FewShotStrategy.DIVERSE.value == "diverse"

    def test_fixed_value(self) -> None:
        """Fixed strategy has correct value."""
        assert FewShotStrategy.FIXED.value == "fixed"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_FEW_SHOT_STRATEGIES is a frozenset."""
        assert isinstance(VALID_FEW_SHOT_STRATEGIES, frozenset)
        assert len(VALID_FEW_SHOT_STRATEGIES) == 4


class TestReasoningType:
    """Tests for ReasoningType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for rtype in ReasoningType:
            assert isinstance(rtype.value, str)

    def test_none_value(self) -> None:
        """None type has correct value."""
        assert ReasoningType.NONE.value == "none"

    def test_chain_of_thought_value(self) -> None:
        """Chain of thought has correct value."""
        assert ReasoningType.CHAIN_OF_THOUGHT.value == "chain_of_thought"

    def test_tree_of_thought_value(self) -> None:
        """Tree of thought has correct value."""
        assert ReasoningType.TREE_OF_THOUGHT.value == "tree_of_thought"

    def test_step_by_step_value(self) -> None:
        """Step by step has correct value."""
        assert ReasoningType.STEP_BY_STEP.value == "step_by_step"

    def test_valid_types_frozenset(self) -> None:
        """VALID_REASONING_TYPES is a frozenset."""
        assert isinstance(VALID_REASONING_TYPES, frozenset)
        assert len(VALID_REASONING_TYPES) == 4


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_create_template(self) -> None:
        """Create prompt template."""
        template = PromptTemplate(
            format=PromptFormat.PLAIN,
            system_prompt="Be helpful.",
            user_prefix="Q: ",
            assistant_prefix="A: ",
        )
        assert template.format == PromptFormat.PLAIN
        assert template.system_prompt == "Be helpful."

    def test_template_is_frozen(self) -> None:
        """Template is immutable."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "Q:", "A:")
        with pytest.raises(AttributeError):
            template.format = PromptFormat.CHAT  # type: ignore[misc]


class TestFewShotConfig:
    """Tests for FewShotConfig dataclass."""

    def test_create_config(self) -> None:
        """Create few-shot config."""
        config = FewShotConfig(
            strategy=FewShotStrategy.FIXED,
            num_examples=3,
            separator="\n\n",
            include_labels=True,
        )
        assert config.strategy == FewShotStrategy.FIXED
        assert config.num_examples == 3

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FewShotConfig(FewShotStrategy.FIXED, 3, "\n", False)
        with pytest.raises(AttributeError):
            config.num_examples = 5  # type: ignore[misc]


class TestCoTConfig:
    """Tests for CoTConfig dataclass."""

    def test_create_config(self) -> None:
        """Create chain-of-thought config."""
        config = CoTConfig(
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            step_marker="Step {n}: ",
            conclusion_marker="Therefore: ",
        )
        assert config.reasoning_type == ReasoningType.CHAIN_OF_THOUGHT
        assert config.step_marker == "Step {n}: "

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = CoTConfig(ReasoningType.NONE, "", "")
        with pytest.raises(AttributeError):
            config.reasoning_type = ReasoningType.STEP_BY_STEP  # type: ignore[misc]


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_create_config(self) -> None:
        """Create prompt config."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "Q:", "A:")
        config = PromptConfig(
            template=template,
            few_shot_config=None,
            cot_config=None,
            max_length=4096,
        )
        assert config.template == template
        assert config.max_length == 4096

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "Q:", "A:")
        config = PromptConfig(template, None, None, 4096)
        with pytest.raises(AttributeError):
            config.max_length = 2048  # type: ignore[misc]


class TestPromptStats:
    """Tests for PromptStats dataclass."""

    def test_create_stats(self) -> None:
        """Create prompt stats."""
        stats = PromptStats(
            total_chars=500,
            estimated_tokens=125,
            num_examples=3,
            has_reasoning=True,
            format_type="chatml",
        )
        assert stats.total_chars == 500
        assert stats.estimated_tokens == 125

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = PromptStats(100, 25, 0, False, "plain")
        with pytest.raises(AttributeError):
            stats.total_chars = 200  # type: ignore[misc]


class TestFewShotExample:
    """Tests for FewShotExample dataclass."""

    def test_create_example(self) -> None:
        """Create few-shot example."""
        example = FewShotExample(
            input_text="2+2",
            output_text="4",
            label="math",
        )
        assert example.input_text == "2+2"
        assert example.output_text == "4"

    def test_example_is_frozen(self) -> None:
        """Example is immutable."""
        example = FewShotExample("a", "b", "")
        with pytest.raises(AttributeError):
            example.input_text = "c"  # type: ignore[misc]


class TestValidatePromptTemplate:
    """Tests for validate_prompt_template function."""

    def test_valid_template(self) -> None:
        """Valid template passes validation."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "Q:", "A:")
        validate_prompt_template(template)

    def test_empty_prefixes_raises(self) -> None:
        """Both empty prefixes raises ValueError."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "", "")
        with pytest.raises(ValueError, match="cannot both be empty"):
            validate_prompt_template(template)

    def test_user_prefix_only_valid(self) -> None:
        """User prefix only is valid."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "Q:", "")
        validate_prompt_template(template)

    def test_assistant_prefix_only_valid(self) -> None:
        """Assistant prefix only is valid."""
        template = PromptTemplate(PromptFormat.PLAIN, "", "", "A:")
        validate_prompt_template(template)


class TestValidateFewShotConfig:
    """Tests for validate_few_shot_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FewShotConfig(FewShotStrategy.FIXED, 3, "\n", False)
        validate_few_shot_config(config)

    def test_zero_examples_raises(self) -> None:
        """Zero examples raises ValueError."""
        config = FewShotConfig(FewShotStrategy.FIXED, 0, "\n", False)
        with pytest.raises(ValueError, match="num_examples must be positive"):
            validate_few_shot_config(config)

    def test_negative_examples_raises(self) -> None:
        """Negative examples raises ValueError."""
        config = FewShotConfig(FewShotStrategy.FIXED, -1, "\n", False)
        with pytest.raises(ValueError, match="num_examples must be positive"):
            validate_few_shot_config(config)


class TestValidateCoTConfig:
    """Tests for validate_cot_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = CoTConfig(ReasoningType.CHAIN_OF_THOUGHT, "Step:", "Therefore:")
        validate_cot_config(config)

    def test_none_type_allows_empty_marker(self) -> None:
        """NONE type allows empty step_marker."""
        config = CoTConfig(ReasoningType.NONE, "", "")
        validate_cot_config(config)

    def test_empty_step_marker_raises(self) -> None:
        """Empty step_marker with reasoning raises ValueError."""
        config = CoTConfig(ReasoningType.CHAIN_OF_THOUGHT, "", "Therefore:")
        with pytest.raises(ValueError, match="step_marker cannot be empty"):
            validate_cot_config(config)

    def test_tree_of_thought_needs_marker(self) -> None:
        """Tree of thought needs step_marker."""
        config = CoTConfig(ReasoningType.TREE_OF_THOUGHT, "", "")
        with pytest.raises(ValueError, match="step_marker cannot be empty"):
            validate_cot_config(config)


class TestValidatePromptConfig:
    """Tests for validate_prompt_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        template = create_prompt_template()
        config = PromptConfig(template, None, None, 4096)
        validate_prompt_config(config)

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        template = create_prompt_template()
        config = PromptConfig(template, None, None, 0)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_prompt_config(config)

    def test_validates_nested_configs(self) -> None:
        """Validates nested configurations."""
        template = create_prompt_template()
        bad_few_shot = FewShotConfig(FewShotStrategy.FIXED, 0, "\n", False)
        config = PromptConfig(template, bad_few_shot, None, 4096)
        with pytest.raises(ValueError, match="num_examples must be positive"):
            validate_prompt_config(config)


class TestListPromptFormats:
    """Tests for list_prompt_formats function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        formats = list_prompt_formats()
        assert formats == sorted(formats)

    def test_contains_all_formats(self) -> None:
        """Contains all format values."""
        formats = list_prompt_formats()
        assert "plain" in formats
        assert "chatml" in formats
        assert "llama2" in formats


class TestListFewShotStrategies:
    """Tests for list_few_shot_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_few_shot_strategies()
        assert strategies == sorted(strategies)

    def test_contains_all_strategies(self) -> None:
        """Contains all strategy values."""
        strategies = list_few_shot_strategies()
        assert "random" in strategies
        assert "similar" in strategies


class TestListReasoningTypes:
    """Tests for list_reasoning_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_reasoning_types()
        assert types == sorted(types)

    def test_contains_all_types(self) -> None:
        """Contains all type values."""
        types = list_reasoning_types()
        assert "none" in types
        assert "chain_of_thought" in types


class TestGetPromptFormat:
    """Tests for get_prompt_format function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("plain", PromptFormat.PLAIN),
            ("chat", PromptFormat.CHAT),
            ("instruct", PromptFormat.INSTRUCT),
            ("chatml", PromptFormat.CHATML),
            ("alpaca", PromptFormat.ALPACA),
            ("llama2", PromptFormat.LLAMA2),
        ],
    )
    def test_all_formats(self, name: str, expected: PromptFormat) -> None:
        """Test all valid formats."""
        assert get_prompt_format(name) == expected

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be one of"):
            get_prompt_format("invalid")


class TestGetFewShotStrategy:
    """Tests for get_few_shot_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("random", FewShotStrategy.RANDOM),
            ("similar", FewShotStrategy.SIMILAR),
            ("diverse", FewShotStrategy.DIVERSE),
            ("fixed", FewShotStrategy.FIXED),
        ],
    )
    def test_all_strategies(self, name: str, expected: FewShotStrategy) -> None:
        """Test all valid strategies."""
        assert get_few_shot_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_few_shot_strategy("invalid")


class TestGetReasoningType:
    """Tests for get_reasoning_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("none", ReasoningType.NONE),
            ("chain_of_thought", ReasoningType.CHAIN_OF_THOUGHT),
            ("tree_of_thought", ReasoningType.TREE_OF_THOUGHT),
            ("step_by_step", ReasoningType.STEP_BY_STEP),
        ],
    )
    def test_all_types(self, name: str, expected: ReasoningType) -> None:
        """Test all valid types."""
        assert get_reasoning_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="reasoning_type must be one of"):
            get_reasoning_type("invalid")


class TestCreatePromptTemplate:
    """Tests for create_prompt_template function."""

    def test_default_template(self) -> None:
        """Create default template."""
        template = create_prompt_template()
        assert template.format == PromptFormat.PLAIN
        assert template.user_prefix == "User: "
        assert template.assistant_prefix == "Assistant: "

    def test_custom_template(self) -> None:
        """Create custom template."""
        template = create_prompt_template(
            format="chatml",
            system_prompt="Be helpful.",
            user_prefix="Q: ",
            assistant_prefix="A: ",
        )
        assert template.format == PromptFormat.CHATML
        assert template.system_prompt == "Be helpful."

    def test_string_format(self) -> None:
        """Create template with string format."""
        template = create_prompt_template(format="llama2")
        assert template.format == PromptFormat.LLAMA2

    def test_empty_prefixes_raises(self) -> None:
        """Empty prefixes raises ValueError."""
        with pytest.raises(ValueError, match="cannot both be empty"):
            create_prompt_template(user_prefix="", assistant_prefix="")


class TestCreateFewShotConfig:
    """Tests for create_few_shot_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_few_shot_config()
        assert config.strategy == FewShotStrategy.FIXED
        assert config.num_examples == 3

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_few_shot_config(
            strategy="random",
            num_examples=5,
            include_labels=True,
        )
        assert config.strategy == FewShotStrategy.RANDOM
        assert config.num_examples == 5
        assert config.include_labels is True

    def test_zero_examples_raises(self) -> None:
        """Zero examples raises ValueError."""
        with pytest.raises(ValueError, match="num_examples must be positive"):
            create_few_shot_config(num_examples=0)


class TestCreateCoTConfig:
    """Tests for create_cot_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_cot_config()
        assert config.reasoning_type == ReasoningType.CHAIN_OF_THOUGHT
        assert config.step_marker == "Step {n}: "

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_cot_config(
            reasoning_type="step_by_step",
            step_marker="#{n}. ",
            conclusion_marker="Answer: ",
        )
        assert config.reasoning_type == ReasoningType.STEP_BY_STEP
        assert config.step_marker == "#{n}. "

    def test_empty_step_marker_raises(self) -> None:
        """Empty step_marker raises ValueError."""
        with pytest.raises(ValueError, match="step_marker cannot be empty"):
            create_cot_config(step_marker="")


class TestCreatePromptConfig:
    """Tests for create_prompt_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_prompt_config()
        assert config.max_length == 4096
        assert config.template.format == PromptFormat.PLAIN

    def test_custom_config(self) -> None:
        """Create custom config."""
        template = create_prompt_template(format="chatml")
        few_shot = create_few_shot_config()
        cot = create_cot_config()
        config = create_prompt_config(
            template=template,
            few_shot_config=few_shot,
            cot_config=cot,
            max_length=2048,
        )
        assert config.max_length == 2048
        assert config.few_shot_config is not None
        assert config.cot_config is not None

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_prompt_config(max_length=0)


class TestCreateFewShotExample:
    """Tests for create_few_shot_example function."""

    def test_simple_example(self) -> None:
        """Create simple example."""
        example = create_few_shot_example("2+2", "4")
        assert example.input_text == "2+2"
        assert example.output_text == "4"
        assert example.label == ""

    def test_with_label(self) -> None:
        """Create example with label."""
        example = create_few_shot_example("text", "positive", label="sentiment")
        assert example.label == "sentiment"

    def test_empty_input_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="input_text cannot be empty"):
            create_few_shot_example("", "output")

    def test_empty_output_raises(self) -> None:
        """Empty output raises ValueError."""
        with pytest.raises(ValueError, match="output_text cannot be empty"):
            create_few_shot_example("input", "")

    def test_whitespace_input_raises(self) -> None:
        """Whitespace input raises ValueError."""
        with pytest.raises(ValueError, match="input_text cannot be empty"):
            create_few_shot_example("   ", "output")

    def test_whitespace_output_raises(self) -> None:
        """Whitespace output raises ValueError."""
        with pytest.raises(ValueError, match="output_text cannot be empty"):
            create_few_shot_example("input", "   ")


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_plain_format(self) -> None:
        """Format plain prompt."""
        template = create_prompt_template(
            format="plain",
            system_prompt="Be helpful.",
            user_prefix="Q: ",
            assistant_prefix="A: ",
        )
        result = format_prompt(template, "What is 2+2?")
        assert "Be helpful." in result
        assert "Q: What is 2+2?" in result
        assert "A: " in result

    def test_chatml_format(self) -> None:
        """Format ChatML prompt."""
        template = create_prompt_template(
            format="chatml",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Hello")
        assert "<|im_start|>system" in result
        assert "Be helpful." in result
        assert "<|im_start|>user" in result
        assert "Hello" in result

    def test_chatml_with_response(self) -> None:
        """Format ChatML prompt with assistant response."""
        template = create_prompt_template(
            format="chatml",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Hello", assistant_response="Hi there!")
        assert "<|im_start|>assistant" in result
        assert "Hi there!" in result
        assert "<|im_end|>" in result

    def test_chatml_without_system(self) -> None:
        """Format ChatML prompt without system prompt."""
        template = create_prompt_template(format="chatml", system_prompt="")
        result = format_prompt(template, "Hello")
        assert "<|im_start|>system" not in result
        assert "<|im_start|>user" in result

    def test_llama2_format(self) -> None:
        """Format Llama2 prompt."""
        template = create_prompt_template(
            format="llama2",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Hello")
        assert "[INST]" in result
        assert "<<SYS>>" in result
        assert "<</SYS>>" in result
        assert "[/INST]" in result

    def test_llama2_without_system(self) -> None:
        """Format Llama2 prompt without system."""
        template = create_prompt_template(format="llama2")
        result = format_prompt(template, "Hello")
        assert "[INST] Hello [/INST]" in result
        assert "<<SYS>>" not in result

    def test_llama2_with_response(self) -> None:
        """Format Llama2 prompt with assistant response."""
        template = create_prompt_template(
            format="llama2",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Hello", assistant_response="Hi!")
        assert "Hi!" in result

    def test_alpaca_format(self) -> None:
        """Format Alpaca prompt."""
        template = create_prompt_template(
            format="alpaca",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Question")
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result

    def test_alpaca_without_system(self) -> None:
        """Format Alpaca prompt without system prompt."""
        template = create_prompt_template(format="alpaca", system_prompt="")
        result = format_prompt(template, "Question")
        assert "### Instruction:" in result
        assert "### Input:" in result

    def test_alpaca_with_response(self) -> None:
        """Format Alpaca prompt with response."""
        template = create_prompt_template(
            format="alpaca",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Question", assistant_response="Answer")
        assert "Answer" in result

    def test_instruct_format(self) -> None:
        """Format Instruct prompt."""
        template = create_prompt_template(
            format="instruct",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Question")
        assert "System: Be helpful." in result
        assert "User: Question" in result
        assert "Assistant:" in result

    def test_instruct_without_system(self) -> None:
        """Format Instruct prompt without system prompt."""
        template = create_prompt_template(format="instruct", system_prompt="")
        result = format_prompt(template, "Question")
        assert "System:" not in result
        assert "User: Question" in result

    def test_instruct_with_response(self) -> None:
        """Format Instruct prompt with response."""
        template = create_prompt_template(
            format="instruct",
            system_prompt="Be helpful.",
        )
        result = format_prompt(template, "Question", assistant_response="Answer")
        assert "Answer" in result

    def test_with_response(self) -> None:
        """Format with assistant response."""
        template = create_prompt_template()
        result = format_prompt(template, "Hello", assistant_response="Hi there!")
        assert "Hi there!" in result

    def test_empty_message_raises(self) -> None:
        """Empty user message raises ValueError."""
        template = create_prompt_template()
        with pytest.raises(ValueError, match="user_message cannot be empty"):
            format_prompt(template, "")

    def test_whitespace_message_raises(self) -> None:
        """Whitespace user message raises ValueError."""
        template = create_prompt_template()
        with pytest.raises(ValueError, match="user_message cannot be empty"):
            format_prompt(template, "   ")


class TestFormatFewShotExamples:
    """Tests for format_few_shot_examples function."""

    def test_basic_format(self) -> None:
        """Format basic examples."""
        examples = (
            create_few_shot_example("2+2", "4"),
            create_few_shot_example("3+3", "6"),
        )
        config = create_few_shot_config(num_examples=2)
        template = create_prompt_template()
        result = format_few_shot_examples(examples, config, template)
        assert "2+2" in result
        assert "4" in result
        assert "3+3" in result
        assert "6" in result

    def test_respects_num_examples(self) -> None:
        """Respects num_examples limit."""
        examples = (
            create_few_shot_example("a", "1"),
            create_few_shot_example("b", "2"),
            create_few_shot_example("c", "3"),
        )
        config = create_few_shot_config(num_examples=1)
        template = create_prompt_template()
        result = format_few_shot_examples(examples, config, template)
        assert "a" in result
        assert "b" not in result

    def test_with_labels(self) -> None:
        """Format with labels included."""
        examples = (create_few_shot_example("text", "pos", label="sentiment"),)
        config = create_few_shot_config(num_examples=1, include_labels=True)
        template = create_prompt_template()
        result = format_few_shot_examples(examples, config, template)
        assert "[sentiment]" in result

    def test_empty_examples_raises(self) -> None:
        """Empty examples raises ValueError."""
        config = create_few_shot_config()
        template = create_prompt_template()
        with pytest.raises(ValueError, match="examples cannot be empty"):
            format_few_shot_examples((), config, template)


class TestAddCoTReasoning:
    """Tests for add_cot_reasoning function."""

    def test_none_reasoning(self) -> None:
        """None reasoning returns original prompt."""
        config = create_cot_config(reasoning_type="none")
        result = add_cot_reasoning("Question", config)
        assert result == "Question"

    def test_chain_of_thought(self) -> None:
        """Chain of thought adds instructions."""
        config = create_cot_config(reasoning_type="chain_of_thought")
        result = add_cot_reasoning("What is 2+2?", config)
        assert "What is 2+2?" in result
        assert "step by step" in result.lower()

    def test_tree_of_thought(self) -> None:
        """Tree of thought adds paths."""
        config = create_cot_config(reasoning_type="tree_of_thought")
        result = add_cot_reasoning("Problem", config)
        assert "Problem" in result
        assert "Path" in result

    def test_step_by_step(self) -> None:
        """Step by step adds step marker."""
        config = create_cot_config(
            reasoning_type="step_by_step",
            step_marker="Step {n}: ",
        )
        result = add_cot_reasoning("Question", config)
        assert "Question" in result
        assert "Step 1:" in result

    def test_includes_conclusion_marker(self) -> None:
        """Includes conclusion marker."""
        config = create_cot_config(conclusion_marker="Answer: ")
        result = add_cot_reasoning("Question", config)
        assert "Answer: " in result

    def test_without_conclusion_marker(self) -> None:
        """Test without conclusion marker."""
        config = create_cot_config(conclusion_marker="")
        result = add_cot_reasoning("Question", config)
        assert "Question" in result
        assert "step by step" in result.lower()


class TestEstimatePromptTokens:
    """Tests for estimate_prompt_tokens function."""

    def test_basic_estimate(self) -> None:
        """Basic token estimation."""
        result = estimate_prompt_tokens("Hello, world!")
        assert result == 3

    def test_custom_ratio(self) -> None:
        """Custom chars_per_token ratio."""
        result = estimate_prompt_tokens("12345678", chars_per_token=2.0)
        assert result == 4

    def test_empty_string(self) -> None:
        """Empty string returns 0."""
        result = estimate_prompt_tokens("")
        assert result == 0

    def test_zero_ratio_raises(self) -> None:
        """Zero chars_per_token raises ValueError."""
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            estimate_prompt_tokens("test", chars_per_token=0)

    def test_negative_ratio_raises(self) -> None:
        """Negative chars_per_token raises ValueError."""
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            estimate_prompt_tokens("test", chars_per_token=-1)

    def test_minimum_one_token(self) -> None:
        """Minimum 1 token for non-empty text."""
        result = estimate_prompt_tokens("a", chars_per_token=10.0)
        assert result == 1


class TestFormatPromptStats:
    """Tests for format_prompt_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = PromptStats(500, 125, 3, True, "chatml")
        result = format_prompt_stats(stats)
        assert "Characters: 500" in result
        assert "Tokens: 125" in result
        assert "Examples: 3" in result
        assert "Reasoning: True" in result
        assert "Format: chatml" in result

    def test_contains_all_fields(self) -> None:
        """Contains all required fields."""
        stats = PromptStats(100, 25, 0, False, "plain")
        result = format_prompt_stats(stats)
        assert "Characters:" in result
        assert "Tokens:" in result
        assert "Examples:" in result
        assert "Reasoning:" in result
        assert "Format:" in result


class TestCreatePromptStats:
    """Tests for create_prompt_stats function."""

    def test_basic_stats(self) -> None:
        """Create basic stats."""
        stats = create_prompt_stats("Hello, world!")
        assert stats.total_chars == 13
        assert stats.estimated_tokens == 3

    def test_with_options(self) -> None:
        """Create stats with options."""
        stats = create_prompt_stats(
            "Test",
            num_examples=5,
            has_reasoning=True,
            format_type="chatml",
        )
        assert stats.num_examples == 5
        assert stats.has_reasoning is True
        assert stats.format_type == "chatml"


class TestGetRecommendedPromptConfig:
    """Tests for get_recommended_prompt_config function."""

    def test_qa_config(self) -> None:
        """Get config for QA task."""
        config = get_recommended_prompt_config("qa")
        assert config.few_shot_config is not None
        assert config.template.format == PromptFormat.PLAIN

    def test_summarization_config(self) -> None:
        """Get config for summarization task."""
        config = get_recommended_prompt_config("summarization")
        assert config.few_shot_config is None

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_prompt_config("classification")
        assert config.few_shot_config is not None
        assert config.few_shot_config.include_labels is True

    def test_code_config(self) -> None:
        """Get config for code task."""
        config = get_recommended_prompt_config("code")
        assert config.max_length == 4096

    def test_reasoning_config(self) -> None:
        """Get config for reasoning task."""
        config = get_recommended_prompt_config("reasoning")
        assert config.cot_config is not None
        assert config.cot_config.reasoning_type == ReasoningType.CHAIN_OF_THOUGHT

    def test_chat_model_type(self) -> None:
        """Get config with chat model type."""
        config = get_recommended_prompt_config("qa", model_type="chat")
        assert config.template.format == PromptFormat.CHAT

    def test_instruct_model_type(self) -> None:
        """Get config with instruct model type."""
        config = get_recommended_prompt_config("qa", model_type="instruct")
        assert config.template.format == PromptFormat.INSTRUCT

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_prompt_config("invalid")

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_prompt_config("qa", model_type="invalid")


class TestBuildCompletePrompt:
    """Tests for build_complete_prompt function."""

    def test_simple_prompt(self) -> None:
        """Build simple prompt."""
        config = create_prompt_config()
        result = build_complete_prompt(config, "What is 2+2?")
        assert "What is 2+2?" in result

    def test_with_few_shot(self) -> None:
        """Build prompt with few-shot examples."""
        few_shot = create_few_shot_config(num_examples=2)
        config = create_prompt_config(few_shot_config=few_shot)
        examples = (
            create_few_shot_example("1+1", "2"),
            create_few_shot_example("2+2", "4"),
        )
        result = build_complete_prompt(config, "3+3", examples=examples)
        assert "1+1" in result
        assert "2+2" in result
        assert "3+3" in result

    def test_with_cot(self) -> None:
        """Build prompt with chain-of-thought."""
        cot = create_cot_config()
        config = create_prompt_config(cot_config=cot)
        result = build_complete_prompt(config, "Complex question")
        assert "step by step" in result.lower()

    def test_empty_message_raises(self) -> None:
        """Empty message raises ValueError."""
        config = create_prompt_config()
        with pytest.raises(ValueError, match="user_message cannot be empty"):
            build_complete_prompt(config, "")


class TestTruncatePrompt:
    """Tests for truncate_prompt function."""

    def test_no_truncation_needed(self) -> None:
        """No truncation when under limit."""
        result = truncate_prompt("Short", max_length=100)
        assert result == "Short"

    def test_truncates_long_text(self) -> None:
        """Truncates text over limit."""
        result = truncate_prompt("Hello, world!", max_length=8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_custom_marker(self) -> None:
        """Uses custom truncation marker."""
        result = truncate_prompt(
            "Hello, world!", max_length=10, truncation_marker="[...]"
        )
        assert result.endswith("[...]")

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            truncate_prompt("test", max_length=0)

    def test_negative_max_length_raises(self) -> None:
        """Negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            truncate_prompt("test", max_length=-1)

    def test_very_short_max_length(self) -> None:
        """Very short max_length returns truncated marker."""
        result = truncate_prompt("Hello, world!", max_length=2)
        assert len(result) == 2

    def test_exact_length(self) -> None:
        """Exact length returns original."""
        text = "Hello"
        result = truncate_prompt(text, max_length=5)
        assert result == text
