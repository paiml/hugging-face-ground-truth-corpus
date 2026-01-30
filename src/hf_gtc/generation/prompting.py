"""Prompt engineering utilities for model-specific formatting.

This module provides functions for creating, formatting, and optimizing
prompts with model-specific formats, few-shot examples, and chain-of-thought
reasoning patterns.

Examples:
    >>> from hf_gtc.generation.prompting import (
    ...     create_prompt_template,
    ...     format_prompt,
    ... )
    >>> template = create_prompt_template(
    ...     format=PromptFormat.PLAIN,
    ...     system_prompt="You are a helpful assistant.",
    ...     user_prefix="User: ",
    ...     assistant_prefix="Assistant: ",
    ... )
    >>> template.format
    <PromptFormat.PLAIN: 'plain'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PromptFormat(Enum):
    """Model-specific prompt formats.

    Attributes:
        PLAIN: Plain text without special formatting.
        CHAT: Generic chat format with role markers.
        INSTRUCT: Instruction-following format.
        CHATML: ChatML format used by OpenAI.
        ALPACA: Stanford Alpaca instruction format.
        LLAMA2: Llama 2 chat format with special tokens.

    Examples:
        >>> PromptFormat.PLAIN.value
        'plain'
        >>> PromptFormat.CHATML.value
        'chatml'
        >>> PromptFormat.LLAMA2.value
        'llama2'
    """

    PLAIN = "plain"
    CHAT = "chat"
    INSTRUCT = "instruct"
    CHATML = "chatml"
    ALPACA = "alpaca"
    LLAMA2 = "llama2"


class FewShotStrategy(Enum):
    """Strategies for selecting few-shot examples.

    Attributes:
        RANDOM: Random selection from example pool.
        SIMILAR: Select examples most similar to input.
        DIVERSE: Select diverse examples for coverage.
        FIXED: Use fixed predetermined examples.

    Examples:
        >>> FewShotStrategy.RANDOM.value
        'random'
        >>> FewShotStrategy.SIMILAR.value
        'similar'
        >>> FewShotStrategy.DIVERSE.value
        'diverse'
    """

    RANDOM = "random"
    SIMILAR = "similar"
    DIVERSE = "diverse"
    FIXED = "fixed"


class ReasoningType(Enum):
    """Types of reasoning patterns for prompts.

    Attributes:
        NONE: No explicit reasoning pattern.
        CHAIN_OF_THOUGHT: Linear step-by-step reasoning.
        TREE_OF_THOUGHT: Branching exploration of solutions.
        STEP_BY_STEP: Explicit numbered step reasoning.

    Examples:
        >>> ReasoningType.NONE.value
        'none'
        >>> ReasoningType.CHAIN_OF_THOUGHT.value
        'chain_of_thought'
        >>> ReasoningType.STEP_BY_STEP.value
        'step_by_step'
    """

    NONE = "none"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    STEP_BY_STEP = "step_by_step"


VALID_PROMPT_FORMATS = frozenset(f.value for f in PromptFormat)
VALID_FEW_SHOT_STRATEGIES = frozenset(s.value for s in FewShotStrategy)
VALID_REASONING_TYPES = frozenset(t.value for t in ReasoningType)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A model-specific prompt template configuration.

    Attributes:
        format: The prompt format type.
        system_prompt: System-level instructions.
        user_prefix: Prefix before user messages.
        assistant_prefix: Prefix before assistant messages.

    Examples:
        >>> template = PromptTemplate(
        ...     format=PromptFormat.PLAIN,
        ...     system_prompt="You are helpful.",
        ...     user_prefix="User: ",
        ...     assistant_prefix="Assistant: ",
        ... )
        >>> template.format
        <PromptFormat.PLAIN: 'plain'>
        >>> template.user_prefix
        'User: '
    """

    format: PromptFormat
    system_prompt: str
    user_prefix: str
    assistant_prefix: str


@dataclass(frozen=True, slots=True)
class FewShotConfig:
    r"""Configuration for few-shot prompting.

    Attributes:
        strategy: How to select examples.
        num_examples: Number of examples to include.
        separator: Separator between examples.
        include_labels: Whether to include labels in examples.

    Examples:
        >>> config = FewShotConfig(
        ...     strategy=FewShotStrategy.FIXED,
        ...     num_examples=3,
        ...     separator="\\n\\n",
        ...     include_labels=True,
        ... )
        >>> config.num_examples
        3
        >>> config.include_labels
        True
    """

    strategy: FewShotStrategy
    num_examples: int
    separator: str
    include_labels: bool


@dataclass(frozen=True, slots=True)
class CoTConfig:
    """Configuration for chain-of-thought reasoning.

    Attributes:
        reasoning_type: Type of reasoning pattern.
        step_marker: Marker for each reasoning step.
        conclusion_marker: Marker for final conclusion.

    Examples:
        >>> config = CoTConfig(
        ...     reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
        ...     step_marker="Step {n}: ",
        ...     conclusion_marker="Therefore: ",
        ... )
        >>> config.reasoning_type
        <ReasoningType.CHAIN_OF_THOUGHT: 'chain_of_thought'>
        >>> config.step_marker
        'Step {n}: '
    """

    reasoning_type: ReasoningType
    step_marker: str
    conclusion_marker: str


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Main configuration for prompt construction.

    Attributes:
        template: The prompt template.
        few_shot_config: Optional few-shot configuration.
        cot_config: Optional chain-of-thought configuration.
        max_length: Maximum prompt length in characters.

    Examples:
        >>> template = PromptTemplate(
        ...     format=PromptFormat.PLAIN,
        ...     system_prompt="Be helpful.",
        ...     user_prefix="Q: ",
        ...     assistant_prefix="A: ",
        ... )
        >>> config = PromptConfig(
        ...     template=template,
        ...     few_shot_config=None,
        ...     cot_config=None,
        ...     max_length=4096,
        ... )
        >>> config.max_length
        4096
    """

    template: PromptTemplate
    few_shot_config: FewShotConfig | None
    cot_config: CoTConfig | None
    max_length: int


@dataclass(frozen=True, slots=True)
class PromptStats:
    """Statistics about a constructed prompt.

    Attributes:
        total_chars: Total character count.
        estimated_tokens: Estimated token count.
        num_examples: Number of few-shot examples.
        has_reasoning: Whether reasoning is enabled.
        format_type: The prompt format used.

    Examples:
        >>> stats = PromptStats(
        ...     total_chars=500,
        ...     estimated_tokens=125,
        ...     num_examples=3,
        ...     has_reasoning=True,
        ...     format_type="chatml",
        ... )
        >>> stats.estimated_tokens
        125
        >>> stats.has_reasoning
        True
    """

    total_chars: int
    estimated_tokens: int
    num_examples: int
    has_reasoning: bool
    format_type: str


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """A single few-shot example.

    Attributes:
        input_text: The input or query.
        output_text: The expected output.
        label: Optional label for classification.

    Examples:
        >>> example = FewShotExample(
        ...     input_text="What is 2+2?",
        ...     output_text="4",
        ...     label="math",
        ... )
        >>> example.input_text
        'What is 2+2?'
    """

    input_text: str
    output_text: str
    label: str


def validate_prompt_template(template: PromptTemplate) -> None:
    """Validate a prompt template.

    Args:
        template: Template to validate.

    Raises:
        ValueError: If template is invalid.

    Examples:
        >>> template = PromptTemplate(
        ...     format=PromptFormat.PLAIN,
        ...     system_prompt="Be helpful.",
        ...     user_prefix="User: ",
        ...     assistant_prefix="Assistant: ",
        ... )
        >>> validate_prompt_template(template)

        >>> bad_template = PromptTemplate(
        ...     format=PromptFormat.PLAIN,
        ...     system_prompt="",
        ...     user_prefix="",
        ...     assistant_prefix="",
        ... )
        >>> validate_prompt_template(bad_template)
        Traceback (most recent call last):
            ...
        ValueError: user_prefix and assistant_prefix cannot both be empty
    """
    if not template.user_prefix and not template.assistant_prefix:
        msg = "user_prefix and assistant_prefix cannot both be empty"
        raise ValueError(msg)


def validate_few_shot_config(config: FewShotConfig) -> None:
    r"""Validate few-shot configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = FewShotConfig(
        ...     strategy=FewShotStrategy.FIXED,
        ...     num_examples=3,
        ...     separator="\\n",
        ...     include_labels=True,
        ... )
        >>> validate_few_shot_config(config)

        >>> bad_config = FewShotConfig(
        ...     strategy=FewShotStrategy.FIXED,
        ...     num_examples=0,
        ...     separator="\\n",
        ...     include_labels=True,
        ... )
        >>> validate_few_shot_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_examples must be positive, got 0
    """
    if config.num_examples <= 0:
        msg = f"num_examples must be positive, got {config.num_examples}"
        raise ValueError(msg)


def validate_cot_config(config: CoTConfig) -> None:
    """Validate chain-of-thought configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = CoTConfig(
        ...     reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
        ...     step_marker="Step {n}: ",
        ...     conclusion_marker="Therefore: ",
        ... )
        >>> validate_cot_config(config)

        >>> bad_config = CoTConfig(
        ...     reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
        ...     step_marker="",
        ...     conclusion_marker="",
        ... )
        >>> validate_cot_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: step_marker cannot be empty for chain_of_thought reasoning
    """
    if config.reasoning_type != ReasoningType.NONE and not config.step_marker:
        msg = f"step_marker cannot be empty for {config.reasoning_type.value} reasoning"
        raise ValueError(msg)


def validate_prompt_config(config: PromptConfig) -> None:
    """Validate prompt configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> template = create_prompt_template()
        >>> config = PromptConfig(
        ...     template=template,
        ...     few_shot_config=None,
        ...     cot_config=None,
        ...     max_length=4096,
        ... )
        >>> validate_prompt_config(config)

        >>> bad_config = PromptConfig(
        ...     template=template,
        ...     few_shot_config=None,
        ...     cot_config=None,
        ...     max_length=0,
        ... )
        >>> validate_prompt_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: max_length must be positive, got 0
    """
    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)
    validate_prompt_template(config.template)
    if config.few_shot_config:
        validate_few_shot_config(config.few_shot_config)
    if config.cot_config:
        validate_cot_config(config.cot_config)


def list_prompt_formats() -> list[str]:
    """List all available prompt formats.

    Returns:
        Sorted list of prompt format names.

    Examples:
        >>> formats = list_prompt_formats()
        >>> "plain" in formats
        True
        >>> "chatml" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_PROMPT_FORMATS)


def list_few_shot_strategies() -> list[str]:
    """List all available few-shot strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_few_shot_strategies()
        >>> "random" in strategies
        True
        >>> "similar" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_FEW_SHOT_STRATEGIES)


def list_reasoning_types() -> list[str]:
    """List all available reasoning types.

    Returns:
        Sorted list of reasoning type names.

    Examples:
        >>> types = list_reasoning_types()
        >>> "chain_of_thought" in types
        True
        >>> "none" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_REASONING_TYPES)


def get_prompt_format(name: str) -> PromptFormat:
    """Get prompt format enum from string name.

    Args:
        name: Name of the prompt format.

    Returns:
        Corresponding PromptFormat enum.

    Raises:
        ValueError: If format name is invalid.

    Examples:
        >>> get_prompt_format("plain")
        <PromptFormat.PLAIN: 'plain'>
        >>> get_prompt_format("chatml")
        <PromptFormat.CHATML: 'chatml'>

        >>> get_prompt_format("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: format must be one of ...
    """
    if name not in VALID_PROMPT_FORMATS:
        msg = f"format must be one of {VALID_PROMPT_FORMATS}, got '{name}'"
        raise ValueError(msg)
    return PromptFormat(name)


def get_few_shot_strategy(name: str) -> FewShotStrategy:
    """Get few-shot strategy enum from string name.

    Args:
        name: Name of the strategy.

    Returns:
        Corresponding FewShotStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_few_shot_strategy("random")
        <FewShotStrategy.RANDOM: 'random'>
        >>> get_few_shot_strategy("similar")
        <FewShotStrategy.SIMILAR: 'similar'>

        >>> get_few_shot_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: strategy must be one of ...
    """
    if name not in VALID_FEW_SHOT_STRATEGIES:
        msg = f"strategy must be one of {VALID_FEW_SHOT_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return FewShotStrategy(name)


def get_reasoning_type(name: str) -> ReasoningType:
    """Get reasoning type enum from string name.

    Args:
        name: Name of the reasoning type.

    Returns:
        Corresponding ReasoningType enum.

    Raises:
        ValueError: If type name is invalid.

    Examples:
        >>> get_reasoning_type("none")
        <ReasoningType.NONE: 'none'>
        >>> get_reasoning_type("chain_of_thought")
        <ReasoningType.CHAIN_OF_THOUGHT: 'chain_of_thought'>

        >>> get_reasoning_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: reasoning_type must be one of ...
    """
    if name not in VALID_REASONING_TYPES:
        msg = f"reasoning_type must be one of {VALID_REASONING_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ReasoningType(name)


def create_prompt_template(
    format: str | PromptFormat = PromptFormat.PLAIN,
    system_prompt: str = "",
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
) -> PromptTemplate:
    """Create a prompt template.

    Args:
        format: The prompt format type.
        system_prompt: System-level instructions.
        user_prefix: Prefix before user messages.
        assistant_prefix: Prefix before assistant messages.

    Returns:
        Validated PromptTemplate.

    Raises:
        ValueError: If template is invalid.

    Examples:
        >>> template = create_prompt_template()
        >>> template.format
        <PromptFormat.PLAIN: 'plain'>
        >>> template.user_prefix
        'User: '

        >>> template = create_prompt_template(format="chatml")
        >>> template.format
        <PromptFormat.CHATML: 'chatml'>

        >>> create_prompt_template(user_prefix="", assistant_prefix="")
        Traceback (most recent call last):
            ...
        ValueError: user_prefix and assistant_prefix cannot both be empty
    """
    if isinstance(format, str):
        format = get_prompt_format(format)

    template = PromptTemplate(
        format=format,
        system_prompt=system_prompt,
        user_prefix=user_prefix,
        assistant_prefix=assistant_prefix,
    )
    validate_prompt_template(template)
    return template


def create_few_shot_config(
    strategy: str | FewShotStrategy = FewShotStrategy.FIXED,
    num_examples: int = 3,
    separator: str = "\n\n",
    include_labels: bool = False,
) -> FewShotConfig:
    r"""Create a few-shot configuration.

    Args:
        strategy: How to select examples.
        num_examples: Number of examples to include.
        separator: Separator between examples.
        include_labels: Whether to include labels.

    Returns:
        Validated FewShotConfig.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_few_shot_config()
        >>> config.strategy
        <FewShotStrategy.FIXED: 'fixed'>
        >>> config.num_examples
        3

        >>> config = create_few_shot_config(strategy="random", num_examples=5)
        >>> config.strategy
        <FewShotStrategy.RANDOM: 'random'>

        >>> create_few_shot_config(num_examples=0)
        Traceback (most recent call last):
            ...
        ValueError: num_examples must be positive, got 0
    """
    if isinstance(strategy, str):
        strategy = get_few_shot_strategy(strategy)

    config = FewShotConfig(
        strategy=strategy,
        num_examples=num_examples,
        separator=separator,
        include_labels=include_labels,
    )
    validate_few_shot_config(config)
    return config


def create_cot_config(
    reasoning_type: str | ReasoningType = ReasoningType.CHAIN_OF_THOUGHT,
    step_marker: str = "Step {n}: ",
    conclusion_marker: str = "Therefore: ",
) -> CoTConfig:
    """Create a chain-of-thought configuration.

    Args:
        reasoning_type: Type of reasoning pattern.
        step_marker: Marker for each step.
        conclusion_marker: Marker for conclusion.

    Returns:
        Validated CoTConfig.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_cot_config()
        >>> config.reasoning_type
        <ReasoningType.CHAIN_OF_THOUGHT: 'chain_of_thought'>
        >>> config.step_marker
        'Step {n}: '

        >>> config = create_cot_config(reasoning_type="step_by_step")
        >>> config.reasoning_type
        <ReasoningType.STEP_BY_STEP: 'step_by_step'>

        >>> create_cot_config(step_marker="")
        Traceback (most recent call last):
            ...
        ValueError: step_marker cannot be empty for chain_of_thought reasoning
    """
    if isinstance(reasoning_type, str):
        reasoning_type = get_reasoning_type(reasoning_type)

    config = CoTConfig(
        reasoning_type=reasoning_type,
        step_marker=step_marker,
        conclusion_marker=conclusion_marker,
    )
    validate_cot_config(config)
    return config


def create_prompt_config(
    template: PromptTemplate | None = None,
    few_shot_config: FewShotConfig | None = None,
    cot_config: CoTConfig | None = None,
    max_length: int = 4096,
) -> PromptConfig:
    """Create a main prompt configuration.

    Args:
        template: The prompt template (uses default if None).
        few_shot_config: Optional few-shot configuration.
        cot_config: Optional chain-of-thought configuration.
        max_length: Maximum prompt length in characters.

    Returns:
        Validated PromptConfig.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_prompt_config()
        >>> config.max_length
        4096

        >>> config = create_prompt_config(max_length=2048)
        >>> config.max_length
        2048

        >>> create_prompt_config(max_length=0)
        Traceback (most recent call last):
            ...
        ValueError: max_length must be positive, got 0
    """
    if template is None:
        template = create_prompt_template()

    config = PromptConfig(
        template=template,
        few_shot_config=few_shot_config,
        cot_config=cot_config,
        max_length=max_length,
    )
    validate_prompt_config(config)
    return config


def create_few_shot_example(
    input_text: str,
    output_text: str,
    label: str = "",
) -> FewShotExample:
    """Create a few-shot example.

    Args:
        input_text: The input or query.
        output_text: The expected output.
        label: Optional label for classification.

    Returns:
        FewShotExample instance.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> example = create_few_shot_example("2+2", "4")
        >>> example.input_text
        '2+2'
        >>> example.output_text
        '4'

        >>> create_few_shot_example("", "output")
        Traceback (most recent call last):
            ...
        ValueError: input_text cannot be empty

        >>> create_few_shot_example("input", "")
        Traceback (most recent call last):
            ...
        ValueError: output_text cannot be empty
    """
    if not input_text or not input_text.strip():
        msg = "input_text cannot be empty"
        raise ValueError(msg)
    if not output_text or not output_text.strip():
        msg = "output_text cannot be empty"
        raise ValueError(msg)

    return FewShotExample(
        input_text=input_text,
        output_text=output_text,
        label=label,
    )


def format_prompt(
    template: PromptTemplate,
    user_message: str,
    assistant_response: str = "",
) -> str:
    """Format a prompt using the template.

    Args:
        template: The prompt template.
        user_message: The user's message.
        assistant_response: Optional assistant response.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If user_message is empty.

    Examples:
        >>> template = create_prompt_template(
        ...     format=PromptFormat.PLAIN,
        ...     system_prompt="Be helpful.",
        ...     user_prefix="Q: ",
        ...     assistant_prefix="A: ",
        ... )
        >>> result = format_prompt(template, "What is 2+2?")
        >>> "Be helpful." in result
        True
        >>> "Q: What is 2+2?" in result
        True

        >>> format_prompt(template, "")
        Traceback (most recent call last):
            ...
        ValueError: user_message cannot be empty
    """
    if not user_message or not user_message.strip():
        msg = "user_message cannot be empty"
        raise ValueError(msg)

    parts = []

    if template.format == PromptFormat.CHATML:
        if template.system_prompt:
            parts.append(f"<|im_start|>system\n{template.system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
        if assistant_response:
            parts.append(f"<|im_start|>assistant\n{assistant_response}<|im_end|>")
        else:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    elif template.format == PromptFormat.LLAMA2:
        if template.system_prompt:
            parts.append(f"[INST] <<SYS>>\n{template.system_prompt}\n<</SYS>>\n\n")
            parts.append(f"{user_message} [/INST]")
        else:
            parts.append(f"[INST] {user_message} [/INST]")
        if assistant_response:
            parts.append(f" {assistant_response}")
        return "".join(parts)

    elif template.format == PromptFormat.ALPACA:
        parts.append("### Instruction:")
        if template.system_prompt:
            parts.append(template.system_prompt)
        parts.append("")
        parts.append("### Input:")
        parts.append(user_message)
        parts.append("")
        parts.append("### Response:")
        if assistant_response:
            parts.append(assistant_response)
        return "\n".join(parts)

    elif template.format == PromptFormat.INSTRUCT:
        if template.system_prompt:
            parts.append(f"System: {template.system_prompt}")
            parts.append("")
        parts.append(f"User: {user_message}")
        parts.append("")
        parts.append("Assistant:")
        if assistant_response:
            parts.append(f" {assistant_response}")
        return "\n".join(parts)

    else:  # PLAIN or CHAT
        if template.system_prompt:
            parts.append(template.system_prompt)
            parts.append("")
        parts.append(f"{template.user_prefix}{user_message}")
        if assistant_response:
            parts.append(f"{template.assistant_prefix}{assistant_response}")
        else:
            parts.append(template.assistant_prefix)
        return "\n".join(parts)


def format_few_shot_examples(
    examples: tuple[FewShotExample, ...],
    config: FewShotConfig,
    template: PromptTemplate,
) -> str:
    """Format few-shot examples for inclusion in prompt.

    Args:
        examples: Tuple of few-shot examples.
        config: Few-shot configuration.
        template: Prompt template for formatting.

    Returns:
        Formatted examples string.

    Raises:
        ValueError: If examples is empty or config is invalid.

    Examples:
        >>> examples = (
        ...     create_few_shot_example("2+2", "4"),
        ...     create_few_shot_example("3+3", "6"),
        ... )
        >>> config = create_few_shot_config(num_examples=2)
        >>> template = create_prompt_template()
        >>> result = format_few_shot_examples(examples, config, template)
        >>> "2+2" in result
        True
        >>> "4" in result
        True

        >>> format_few_shot_examples((), config, template)
        Traceback (most recent call last):
            ...
        ValueError: examples cannot be empty
    """
    if not examples:
        msg = "examples cannot be empty"
        raise ValueError(msg)

    selected = examples[: config.num_examples]
    formatted_parts = []

    for example in selected:
        if config.include_labels and example.label:
            example_text = (
                f"{template.user_prefix}{example.input_text}\n"
                f"[{example.label}]\n"
                f"{template.assistant_prefix}{example.output_text}"
            )
        else:
            example_text = (
                f"{template.user_prefix}{example.input_text}\n"
                f"{template.assistant_prefix}{example.output_text}"
            )
        formatted_parts.append(example_text)

    return config.separator.join(formatted_parts)


def add_cot_reasoning(
    prompt: str,
    config: CoTConfig,
) -> str:
    """Add chain-of-thought reasoning to a prompt.

    Args:
        prompt: The base prompt.
        config: Chain-of-thought configuration.

    Returns:
        Prompt with reasoning instructions added.

    Examples:
        >>> config = create_cot_config()
        >>> result = add_cot_reasoning("What is 2+2?", config)
        >>> "What is 2+2?" in result
        True
        >>> "step by step" in result.lower() or "Step" in result
        True

        >>> config = create_cot_config(reasoning_type="none")
        >>> result = add_cot_reasoning("Simple question", config)
        >>> result
        'Simple question'
    """
    if config.reasoning_type == ReasoningType.NONE:
        return prompt

    reasoning_instructions = {
        ReasoningType.CHAIN_OF_THOUGHT: ("\n\nLet's think through this step by step:"),
        ReasoningType.TREE_OF_THOUGHT: (
            "\n\nLet's explore multiple solution paths:\nPath 1: \nPath 2: \nPath 3: "
        ),
        ReasoningType.STEP_BY_STEP: (f"\n\n{config.step_marker.format(n=1)}"),
    }

    instruction = reasoning_instructions.get(config.reasoning_type, "")
    if config.conclusion_marker:
        return f"{prompt}{instruction}\n\n{config.conclusion_marker}"
    return f"{prompt}{instruction}"


def estimate_prompt_tokens(
    text: str,
    chars_per_token: float = 4.0,
) -> int:
    """Estimate the number of tokens in a prompt.

    Uses a simple character-based estimation. For accurate
    counts, use a tokenizer directly.

    Args:
        text: The prompt text.
        chars_per_token: Average characters per token.

    Returns:
        Estimated token count.

    Raises:
        ValueError: If chars_per_token is invalid.

    Examples:
        >>> estimate_prompt_tokens("Hello, world!")
        3
        >>> estimate_prompt_tokens("Longer text here.", chars_per_token=5.0)
        3

        >>> estimate_prompt_tokens("")
        0

        >>> estimate_prompt_tokens("test", chars_per_token=0)
        Traceback (most recent call last):
            ...
        ValueError: chars_per_token must be positive, got 0
    """
    if chars_per_token <= 0:
        msg = f"chars_per_token must be positive, got {chars_per_token}"
        raise ValueError(msg)

    if not text:
        return 0

    return max(1, int(len(text) / chars_per_token))


def format_prompt_stats(stats: PromptStats) -> str:
    """Format prompt statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = PromptStats(
        ...     total_chars=500,
        ...     estimated_tokens=125,
        ...     num_examples=3,
        ...     has_reasoning=True,
        ...     format_type="chatml",
        ... )
        >>> formatted = format_prompt_stats(stats)
        >>> "Characters: 500" in formatted
        True
        >>> "Tokens: 125" in formatted
        True
        >>> "Examples: 3" in formatted
        True
    """
    return (
        f"Prompt Stats:\n"
        f"  Characters: {stats.total_chars}\n"
        f"  Tokens: {stats.estimated_tokens}\n"
        f"  Examples: {stats.num_examples}\n"
        f"  Reasoning: {stats.has_reasoning}\n"
        f"  Format: {stats.format_type}"
    )


def create_prompt_stats(
    text: str,
    num_examples: int = 0,
    has_reasoning: bool = False,
    format_type: str = "plain",
) -> PromptStats:
    """Create statistics for a prompt.

    Args:
        text: The prompt text.
        num_examples: Number of few-shot examples.
        has_reasoning: Whether reasoning is enabled.
        format_type: The prompt format type.

    Returns:
        PromptStats instance.

    Examples:
        >>> stats = create_prompt_stats("Hello, world!")
        >>> stats.total_chars
        13
        >>> stats.estimated_tokens
        3

        >>> stats = create_prompt_stats("Test", num_examples=5, has_reasoning=True)
        >>> stats.num_examples
        5
        >>> stats.has_reasoning
        True
    """
    return PromptStats(
        total_chars=len(text),
        estimated_tokens=estimate_prompt_tokens(text),
        num_examples=num_examples,
        has_reasoning=has_reasoning,
        format_type=format_type,
    )


def get_recommended_prompt_config(
    task: str,
    model_type: str = "general",
) -> PromptConfig:
    """Get recommended prompt configuration for a task.

    Args:
        task: Task type (qa, summarization, classification, code, reasoning).
        model_type: Model type (general, instruct, chat).

    Returns:
        Recommended PromptConfig for the task.

    Raises:
        ValueError: If task or model_type is unknown.

    Examples:
        >>> config = get_recommended_prompt_config("qa")
        >>> config.template.format
        <PromptFormat.PLAIN: 'plain'>

        >>> config = get_recommended_prompt_config("reasoning")
        >>> config.cot_config is not None
        True

        >>> config = get_recommended_prompt_config("qa", model_type="chat")
        >>> config.template.format
        <PromptFormat.CHAT: 'chat'>

        >>> get_recommended_prompt_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task must be one of ...
    """
    valid_tasks = frozenset(
        {"qa", "summarization", "classification", "code", "reasoning"}
    )
    valid_model_types = frozenset({"general", "instruct", "chat"})

    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    if model_type not in valid_model_types:
        msg = f"model_type must be one of {valid_model_types}, got '{model_type}'"
        raise ValueError(msg)

    # Select format based on model type
    format_map = {
        "general": PromptFormat.PLAIN,
        "instruct": PromptFormat.INSTRUCT,
        "chat": PromptFormat.CHAT,
    }
    prompt_format = format_map[model_type]

    # Task-specific configurations
    if task == "qa":
        template = create_prompt_template(
            format=prompt_format,
            system_prompt="Answer the question based on the given context.",
            user_prefix="Question: ",
            assistant_prefix="Answer: ",
        )
        few_shot = create_few_shot_config(num_examples=3)
        return create_prompt_config(
            template=template,
            few_shot_config=few_shot,
            max_length=2048,
        )

    elif task == "summarization":
        template = create_prompt_template(
            format=prompt_format,
            system_prompt="Summarize the following text concisely.",
            user_prefix="Text: ",
            assistant_prefix="Summary: ",
        )
        return create_prompt_config(
            template=template,
            max_length=4096,
        )

    elif task == "classification":
        template = create_prompt_template(
            format=prompt_format,
            system_prompt="Classify the input into one of the categories.",
            user_prefix="Input: ",
            assistant_prefix="Category: ",
        )
        few_shot = create_few_shot_config(
            num_examples=5,
            include_labels=True,
        )
        return create_prompt_config(
            template=template,
            few_shot_config=few_shot,
            max_length=1024,
        )

    elif task == "code":
        template = create_prompt_template(
            format=prompt_format,
            system_prompt="Generate code to solve the problem.",
            user_prefix="Problem: ",
            assistant_prefix="Solution:\n```python\n",
        )
        return create_prompt_config(
            template=template,
            max_length=4096,
        )

    else:  # reasoning
        template = create_prompt_template(
            format=prompt_format,
            system_prompt="Solve the problem using step-by-step reasoning.",
            user_prefix="Problem: ",
            assistant_prefix="Solution: ",
        )
        cot = create_cot_config(
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
        )
        return create_prompt_config(
            template=template,
            cot_config=cot,
            max_length=4096,
        )


def build_complete_prompt(
    config: PromptConfig,
    user_message: str,
    examples: tuple[FewShotExample, ...] | None = None,
) -> str:
    """Build a complete prompt from configuration.

    Args:
        config: The prompt configuration.
        user_message: The user's message.
        examples: Optional few-shot examples.

    Returns:
        Complete formatted prompt string.

    Raises:
        ValueError: If user_message is empty or configuration is invalid.

    Examples:
        >>> config = create_prompt_config()
        >>> result = build_complete_prompt(config, "What is 2+2?")
        >>> "What is 2+2?" in result
        True

        >>> build_complete_prompt(config, "")
        Traceback (most recent call last):
            ...
        ValueError: user_message cannot be empty
    """
    if not user_message or not user_message.strip():
        msg = "user_message cannot be empty"
        raise ValueError(msg)

    parts = []

    # Add few-shot examples if configured
    if config.few_shot_config and examples:
        examples_text = format_few_shot_examples(
            examples,
            config.few_shot_config,
            config.template,
        )
        parts.append(examples_text)
        parts.append(config.few_shot_config.separator)

    # Format the main prompt
    main_prompt = format_prompt(config.template, user_message)

    # Add chain-of-thought if configured
    if config.cot_config:
        main_prompt = add_cot_reasoning(main_prompt, config.cot_config)

    parts.append(main_prompt)

    return "".join(parts)


def truncate_prompt(
    prompt: str,
    max_length: int,
    truncation_marker: str = "...",
) -> str:
    """Truncate a prompt to fit within length limit.

    Args:
        prompt: The prompt text.
        max_length: Maximum length in characters.
        truncation_marker: Marker to indicate truncation.

    Returns:
        Truncated prompt string.

    Raises:
        ValueError: If max_length is invalid.

    Examples:
        >>> truncate_prompt("Hello, world!", max_length=8)
        'Hello...'

        >>> truncate_prompt("Short", max_length=100)
        'Short'

        >>> truncate_prompt("Test", max_length=0)
        Traceback (most recent call last):
            ...
        ValueError: max_length must be positive, got 0
    """
    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if len(prompt) <= max_length:
        return prompt

    truncate_at = max_length - len(truncation_marker)
    if truncate_at <= 0:
        return truncation_marker[:max_length]

    return prompt[:truncate_at] + truncation_marker
