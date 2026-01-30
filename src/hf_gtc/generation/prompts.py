"""Prompt engineering utilities for language models.

This module provides functions for creating, formatting, and optimizing
prompts for large language models, including templates, few-shot examples,
chain-of-thought reasoning, and system prompts.

Examples:
    >>> from hf_gtc.generation.prompts import (
    ...     create_prompt_template,
    ...     format_prompt,
    ... )
    >>> template = create_prompt_template("Hello, {name}!")
    >>> format_prompt(template, name="World")
    'Hello, World!'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PromptStyle(Enum):
    """Prompting strategies for language models.

    Attributes:
        ZERO_SHOT: Direct instruction without examples.
        FEW_SHOT: Learning from provided examples.
        CHAIN_OF_THOUGHT: Step-by-step reasoning.
        TREE_OF_THOUGHT: Branching exploration of solutions.
        SELF_CONSISTENCY: Multiple reasoning paths with voting.

    Examples:
        >>> PromptStyle.ZERO_SHOT.value
        'zero_shot'
        >>> PromptStyle.CHAIN_OF_THOUGHT.value
        'chain_of_thought'
        >>> PromptStyle.FEW_SHOT.value
        'few_shot'
    """

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_CONSISTENCY = "self_consistency"


class PromptRole(Enum):
    """Message roles in conversational prompts.

    Attributes:
        SYSTEM: System-level instructions and constraints.
        USER: User input or query.
        ASSISTANT: Model response or completion.

    Examples:
        >>> PromptRole.SYSTEM.value
        'system'
        >>> PromptRole.USER.value
        'user'
        >>> PromptRole.ASSISTANT.value
        'assistant'
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class OutputFormat(Enum):
    """Expected output formats for prompts.

    Attributes:
        TEXT: Free-form text output.
        JSON: Structured JSON output.
        MARKDOWN: Markdown-formatted output.
        CODE: Source code output.
        LIST: Bulleted or numbered list.

    Examples:
        >>> OutputFormat.JSON.value
        'json'
        >>> OutputFormat.CODE.value
        'code'
        >>> OutputFormat.MARKDOWN.value
        'markdown'
    """

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    LIST = "list"


class ExampleSelectionStrategy(Enum):
    """Strategies for selecting few-shot examples.

    Attributes:
        RANDOM: Random selection from pool.
        SIMILARITY: Select most similar to input.
        DIVERSITY: Select diverse examples.
        FIXED: Use fixed predetermined examples.

    Examples:
        >>> ExampleSelectionStrategy.SIMILARITY.value
        'similarity'
        >>> ExampleSelectionStrategy.DIVERSITY.value
        'diversity'
    """

    RANDOM = "random"
    SIMILARITY = "similarity"
    DIVERSITY = "diversity"
    FIXED = "fixed"


VALID_PROMPT_STYLES = frozenset(s.value for s in PromptStyle)
VALID_PROMPT_ROLES = frozenset(r.value for r in PromptRole)
VALID_OUTPUT_FORMATS = frozenset(f.value for f in OutputFormat)
VALID_SELECTION_STRATEGIES = frozenset(s.value for s in ExampleSelectionStrategy)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A reusable prompt template with variable placeholders.

    Attributes:
        template: Template string with {variable} placeholders.
        variables: Tuple of variable names in the template.
        description: Optional description of the template purpose.
        default_values: Default values for variables.

    Examples:
        >>> template = PromptTemplate(
        ...     template="Translate '{text}' to {language}.",
        ...     variables=("text", "language"),
        ...     description="Translation prompt",
        ...     default_values={"language": "French"},
        ... )
        >>> template.variables
        ('text', 'language')
    """

    template: str
    variables: tuple[str, ...]
    description: str
    default_values: dict[str, str]


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """A single example for few-shot prompting.

    Attributes:
        input_text: The input or query for this example.
        output_text: The expected output or response.
        explanation: Optional explanation of the reasoning.

    Examples:
        >>> example = FewShotExample(
        ...     input_text="What is 2 + 2?",
        ...     output_text="4",
        ...     explanation="Basic addition",
        ... )
        >>> example.output_text
        '4'
    """

    input_text: str
    output_text: str
    explanation: str


@dataclass(frozen=True, slots=True)
class FewShotConfig:
    r"""Configuration for few-shot prompting.

    Attributes:
        examples: Tuple of few-shot examples.
        num_examples: Number of examples to include.
        selection_strategy: How to select examples.
        example_separator: Separator between examples.
        input_prefix: Prefix before each input.
        output_prefix: Prefix before each output.

    Examples:
        >>> config = FewShotConfig(
        ...     examples=(
        ...         FewShotExample("2+2", "4", ""),
        ...         FewShotExample("3+3", "6", ""),
        ...     ),
        ...     num_examples=2,
        ...     selection_strategy=ExampleSelectionStrategy.FIXED,
        ...     example_separator="\\n\\n",
        ...     input_prefix="Input: ",
        ...     output_prefix="Output: ",
        ... )
        >>> config.num_examples
        2
    """

    examples: tuple[FewShotExample, ...]
    num_examples: int
    selection_strategy: ExampleSelectionStrategy
    example_separator: str
    input_prefix: str
    output_prefix: str


@dataclass(frozen=True, slots=True)
class ChainOfThoughtConfig:
    """Configuration for chain-of-thought prompting.

    Attributes:
        enable_reasoning: Whether to request step-by-step reasoning.
        reasoning_prefix: Prefix for reasoning section.
        answer_prefix: Prefix for final answer.
        step_marker: Marker for each reasoning step.
        max_steps: Maximum reasoning steps to generate.

    Examples:
        >>> config = ChainOfThoughtConfig(
        ...     enable_reasoning=True,
        ...     reasoning_prefix="Let's think step by step:",
        ...     answer_prefix="Therefore, the answer is:",
        ...     step_marker="Step {n}:",
        ...     max_steps=5,
        ... )
        >>> config.enable_reasoning
        True
    """

    enable_reasoning: bool
    reasoning_prefix: str
    answer_prefix: str
    step_marker: str
    max_steps: int


@dataclass(frozen=True, slots=True)
class SystemPromptConfig:
    """Configuration for system prompts.

    Attributes:
        persona: Description of the assistant's role/persona.
        constraints: Tuple of behavioral constraints.
        output_format: Expected output format.
        format_instructions: Specific formatting instructions.
        examples_in_system: Whether to include examples in system prompt.

    Examples:
        >>> config = SystemPromptConfig(
        ...     persona="You are a helpful coding assistant.",
        ...     constraints=("Be concise", "Use Python"),
        ...     output_format=OutputFormat.CODE,
        ...     format_instructions="Return only valid Python code.",
        ...     examples_in_system=False,
        ... )
        >>> config.output_format
        <OutputFormat.CODE: 'code'>
    """

    persona: str
    constraints: tuple[str, ...]
    output_format: OutputFormat
    format_instructions: str
    examples_in_system: bool


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """A single message in a conversational prompt.

    Attributes:
        role: The role of the message sender.
        content: The message content.

    Examples:
        >>> msg = PromptMessage(
        ...     role=PromptRole.USER,
        ...     content="Hello!",
        ... )
        >>> msg.role
        <PromptRole.USER: 'user'>
    """

    role: PromptRole
    content: str


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Main configuration for prompt construction.

    Attributes:
        style: The prompting style to use.
        template: Optional prompt template.
        system_config: Optional system prompt configuration.
        few_shot_config: Optional few-shot configuration.
        cot_config: Optional chain-of-thought configuration.
        max_tokens: Maximum tokens for the prompt.

    Examples:
        >>> config = PromptConfig(
        ...     style=PromptStyle.ZERO_SHOT,
        ...     template=None,
        ...     system_config=None,
        ...     few_shot_config=None,
        ...     cot_config=None,
        ...     max_tokens=2048,
        ... )
        >>> config.style
        <PromptStyle.ZERO_SHOT: 'zero_shot'>
    """

    style: PromptStyle
    template: PromptTemplate | None
    system_config: SystemPromptConfig | None
    few_shot_config: FewShotConfig | None
    cot_config: ChainOfThoughtConfig | None
    max_tokens: int


@dataclass(frozen=True, slots=True)
class PromptStats:
    """Statistics about a constructed prompt.

    Attributes:
        total_chars: Total character count.
        estimated_tokens: Estimated token count.
        num_examples: Number of few-shot examples included.
        has_system_prompt: Whether system prompt is present.
        has_cot: Whether chain-of-thought is enabled.

    Examples:
        >>> stats = PromptStats(
        ...     total_chars=500,
        ...     estimated_tokens=125,
        ...     num_examples=3,
        ...     has_system_prompt=True,
        ...     has_cot=False,
        ... )
        >>> stats.estimated_tokens
        125
    """

    total_chars: int
    estimated_tokens: int
    num_examples: int
    has_system_prompt: bool
    has_cot: bool


def validate_prompt_template(template: PromptTemplate) -> None:
    """Validate a prompt template.

    Args:
        template: Template to validate.

    Raises:
        ValueError: If template is invalid.

    Examples:
        >>> template = PromptTemplate(
        ...     template="Hello, {name}!",
        ...     variables=("name",),
        ...     description="Greeting",
        ...     default_values={},
        ... )
        >>> validate_prompt_template(template)

        >>> bad_template = PromptTemplate(
        ...     template="",
        ...     variables=(),
        ...     description="Empty",
        ...     default_values={},
        ... )
        >>> validate_prompt_template(bad_template)
        Traceback (most recent call last):
            ...
        ValueError: template cannot be empty
    """
    if not template.template or not template.template.strip():
        msg = "template cannot be empty"
        raise ValueError(msg)

    placeholders = set(re.findall(r"\{(\w+)\}", template.template))
    declared = set(template.variables)

    if placeholders != declared:
        missing = placeholders - declared
        extra = declared - placeholders
        if missing:
            msg = f"variables missing from declaration: {missing}"
            raise ValueError(msg)
        if extra:
            msg = f"declared variables not in template: {extra}"
            raise ValueError(msg)


def validate_few_shot_config(config: FewShotConfig) -> None:
    r"""Validate few-shot configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = FewShotConfig(
        ...     examples=(FewShotExample("a", "b", ""),),
        ...     num_examples=1,
        ...     selection_strategy=ExampleSelectionStrategy.FIXED,
        ...     example_separator="\\n",
        ...     input_prefix="Q: ",
        ...     output_prefix="A: ",
        ... )
        >>> validate_few_shot_config(config)

        >>> bad_config = FewShotConfig(
        ...     examples=(),
        ...     num_examples=1,
        ...     selection_strategy=ExampleSelectionStrategy.FIXED,
        ...     example_separator="\\n",
        ...     input_prefix="Q: ",
        ...     output_prefix="A: ",
        ... )
        >>> validate_few_shot_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: examples cannot be empty
    """
    if not config.examples:
        msg = "examples cannot be empty"
        raise ValueError(msg)
    if config.num_examples <= 0:
        msg = f"num_examples must be positive, got {config.num_examples}"
        raise ValueError(msg)
    if config.num_examples > len(config.examples):
        msg = (
            f"num_examples ({config.num_examples}) cannot exceed "
            f"available examples ({len(config.examples)})"
        )
        raise ValueError(msg)


def validate_cot_config(config: ChainOfThoughtConfig) -> None:
    """Validate chain-of-thought configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ChainOfThoughtConfig(
        ...     enable_reasoning=True,
        ...     reasoning_prefix="Let's think:",
        ...     answer_prefix="Answer:",
        ...     step_marker="Step {n}:",
        ...     max_steps=5,
        ... )
        >>> validate_cot_config(config)

        >>> bad_config = ChainOfThoughtConfig(
        ...     enable_reasoning=True,
        ...     reasoning_prefix="",
        ...     answer_prefix="Answer:",
        ...     step_marker="Step:",
        ...     max_steps=5,
        ... )
        >>> validate_cot_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: reasoning_prefix cannot be empty when reasoning is enabled
    """
    if config.enable_reasoning:
        if not config.reasoning_prefix or not config.reasoning_prefix.strip():
            msg = "reasoning_prefix cannot be empty when reasoning is enabled"
            raise ValueError(msg)
        if not config.answer_prefix or not config.answer_prefix.strip():
            msg = "answer_prefix cannot be empty when reasoning is enabled"
            raise ValueError(msg)
    if config.max_steps <= 0:
        msg = f"max_steps must be positive, got {config.max_steps}"
        raise ValueError(msg)


def validate_system_prompt_config(config: SystemPromptConfig) -> None:
    """Validate system prompt configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SystemPromptConfig(
        ...     persona="You are helpful.",
        ...     constraints=(),
        ...     output_format=OutputFormat.TEXT,
        ...     format_instructions="",
        ...     examples_in_system=False,
        ... )
        >>> validate_system_prompt_config(config)

        >>> bad_config = SystemPromptConfig(
        ...     persona="",
        ...     constraints=(),
        ...     output_format=OutputFormat.TEXT,
        ...     format_instructions="",
        ...     examples_in_system=False,
        ... )
        >>> validate_system_prompt_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: persona cannot be empty
    """
    if not config.persona or not config.persona.strip():
        msg = "persona cannot be empty"
        raise ValueError(msg)


def create_prompt_template(
    template: str,
    description: str = "",
    default_values: dict[str, str] | None = None,
) -> PromptTemplate:
    """Create a prompt template with automatic variable detection.

    Args:
        template: Template string with {variable} placeholders.
        description: Optional description of the template.
        default_values: Default values for variables.

    Returns:
        Validated PromptTemplate.

    Raises:
        ValueError: If template is invalid.

    Examples:
        >>> template = create_prompt_template("Hello, {name}!")
        >>> template.variables
        ('name',)
        >>> template.template
        'Hello, {name}!'

        >>> template = create_prompt_template(
        ...     "Translate '{text}' to {lang}.",
        ...     default_values={"lang": "French"},
        ... )
        >>> template.default_values
        {'lang': 'French'}

        >>> create_prompt_template("")
        Traceback (most recent call last):
            ...
        ValueError: template cannot be empty
    """
    if default_values is None:
        default_values = {}

    variables = tuple(sorted(set(re.findall(r"\{(\w+)\}", template))))

    prompt_template = PromptTemplate(
        template=template,
        variables=variables,
        description=description,
        default_values=default_values,
    )
    validate_prompt_template(prompt_template)
    return prompt_template


def create_few_shot_example(
    input_text: str,
    output_text: str,
    explanation: str = "",
) -> FewShotExample:
    """Create a few-shot example.

    Args:
        input_text: The input or query.
        output_text: The expected output.
        explanation: Optional explanation.

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
        explanation=explanation,
    )


def create_few_shot_config(
    examples: tuple[FewShotExample, ...],
    num_examples: int | None = None,
    selection_strategy: str | ExampleSelectionStrategy = ExampleSelectionStrategy.FIXED,
    example_separator: str = "\n\n",
    input_prefix: str = "Input: ",
    output_prefix: str = "Output: ",
) -> FewShotConfig:
    """Create a few-shot configuration.

    Args:
        examples: Tuple of few-shot examples.
        num_examples: Number of examples to use (defaults to all).
        selection_strategy: How to select examples.
        example_separator: Separator between examples.
        input_prefix: Prefix for inputs.
        output_prefix: Prefix for outputs.

    Returns:
        Validated FewShotConfig.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> examples = (
        ...     create_few_shot_example("a", "b"),
        ...     create_few_shot_example("c", "d"),
        ... )
        >>> config = create_few_shot_config(examples)
        >>> config.num_examples
        2

        >>> config = create_few_shot_config(examples, num_examples=1)
        >>> config.num_examples
        1

        >>> create_few_shot_config(())
        Traceback (most recent call last):
            ...
        ValueError: examples cannot be empty
    """
    if isinstance(selection_strategy, str):
        selection_strategy = get_selection_strategy(selection_strategy)

    if num_examples is None:
        num_examples = len(examples)

    config = FewShotConfig(
        examples=examples,
        num_examples=num_examples,
        selection_strategy=selection_strategy,
        example_separator=example_separator,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
    )
    validate_few_shot_config(config)
    return config


def create_cot_config(
    enable_reasoning: bool = True,
    reasoning_prefix: str = "Let's think step by step:",
    answer_prefix: str = "Therefore, the answer is:",
    step_marker: str = "Step {n}:",
    max_steps: int = 10,
) -> ChainOfThoughtConfig:
    """Create a chain-of-thought configuration.

    Args:
        enable_reasoning: Whether to enable reasoning.
        reasoning_prefix: Prefix for reasoning section.
        answer_prefix: Prefix for final answer.
        step_marker: Marker for each step.
        max_steps: Maximum reasoning steps.

    Returns:
        Validated ChainOfThoughtConfig.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_cot_config()
        >>> config.enable_reasoning
        True
        >>> config.max_steps
        10

        >>> config = create_cot_config(max_steps=5)
        >>> config.max_steps
        5

        >>> create_cot_config(max_steps=0)
        Traceback (most recent call last):
            ...
        ValueError: max_steps must be positive, got 0
    """
    config = ChainOfThoughtConfig(
        enable_reasoning=enable_reasoning,
        reasoning_prefix=reasoning_prefix,
        answer_prefix=answer_prefix,
        step_marker=step_marker,
        max_steps=max_steps,
    )
    validate_cot_config(config)
    return config


def create_system_prompt_config(
    persona: str,
    constraints: tuple[str, ...] = (),
    output_format: str | OutputFormat = OutputFormat.TEXT,
    format_instructions: str = "",
    examples_in_system: bool = False,
) -> SystemPromptConfig:
    """Create a system prompt configuration.

    Args:
        persona: Description of assistant's role.
        constraints: Behavioral constraints.
        output_format: Expected output format.
        format_instructions: Specific format instructions.
        examples_in_system: Whether to include examples.

    Returns:
        Validated SystemPromptConfig.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_system_prompt_config("You are helpful.")
        >>> config.persona
        'You are helpful.'

        >>> config = create_system_prompt_config(
        ...     "You are a coder.",
        ...     output_format="code",
        ...     constraints=("Use Python",),
        ... )
        >>> config.output_format
        <OutputFormat.CODE: 'code'>

        >>> create_system_prompt_config("")
        Traceback (most recent call last):
            ...
        ValueError: persona cannot be empty
    """
    if isinstance(output_format, str):
        output_format = get_output_format(output_format)

    config = SystemPromptConfig(
        persona=persona,
        constraints=constraints,
        output_format=output_format,
        format_instructions=format_instructions,
        examples_in_system=examples_in_system,
    )
    validate_system_prompt_config(config)
    return config


def create_prompt_config(
    style: str | PromptStyle = PromptStyle.ZERO_SHOT,
    template: PromptTemplate | None = None,
    system_config: SystemPromptConfig | None = None,
    few_shot_config: FewShotConfig | None = None,
    cot_config: ChainOfThoughtConfig | None = None,
    max_tokens: int = 4096,
) -> PromptConfig:
    """Create a main prompt configuration.

    Args:
        style: Prompting style.
        template: Optional prompt template.
        system_config: Optional system prompt config.
        few_shot_config: Optional few-shot config.
        cot_config: Optional chain-of-thought config.
        max_tokens: Maximum prompt tokens.

    Returns:
        PromptConfig instance.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_prompt_config()
        >>> config.style
        <PromptStyle.ZERO_SHOT: 'zero_shot'>
        >>> config.max_tokens
        4096

        >>> config = create_prompt_config(style="few_shot", max_tokens=2048)
        >>> config.style
        <PromptStyle.FEW_SHOT: 'few_shot'>

        >>> create_prompt_config(max_tokens=0)
        Traceback (most recent call last):
            ...
        ValueError: max_tokens must be positive, got 0
    """
    if isinstance(style, str):
        style = get_prompt_style(style)

    if max_tokens <= 0:
        msg = f"max_tokens must be positive, got {max_tokens}"
        raise ValueError(msg)

    return PromptConfig(
        style=style,
        template=template,
        system_config=system_config,
        few_shot_config=few_shot_config,
        cot_config=cot_config,
        max_tokens=max_tokens,
    )


def create_prompt_message(
    role: str | PromptRole,
    content: str,
) -> PromptMessage:
    """Create a prompt message.

    Args:
        role: Message role.
        content: Message content.

    Returns:
        PromptMessage instance.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> msg = create_prompt_message("user", "Hello!")
        >>> msg.role
        <PromptRole.USER: 'user'>
        >>> msg.content
        'Hello!'

        >>> create_prompt_message("user", "")
        Traceback (most recent call last):
            ...
        ValueError: content cannot be empty
    """
    if isinstance(role, str):
        role = get_prompt_role(role)

    if not content or not content.strip():
        msg = "content cannot be empty"
        raise ValueError(msg)

    return PromptMessage(role=role, content=content)


def list_prompt_styles() -> list[str]:
    """List all available prompt styles.

    Returns:
        Sorted list of prompt style names.

    Examples:
        >>> styles = list_prompt_styles()
        >>> "zero_shot" in styles
        True
        >>> "few_shot" in styles
        True
        >>> styles == sorted(styles)
        True
    """
    return sorted(VALID_PROMPT_STYLES)


def list_prompt_roles() -> list[str]:
    """List all available prompt roles.

    Returns:
        Sorted list of prompt role names.

    Examples:
        >>> roles = list_prompt_roles()
        >>> "user" in roles
        True
        >>> "system" in roles
        True
        >>> roles == sorted(roles)
        True
    """
    return sorted(VALID_PROMPT_ROLES)


def list_output_formats() -> list[str]:
    """List all available output formats.

    Returns:
        Sorted list of output format names.

    Examples:
        >>> formats = list_output_formats()
        >>> "json" in formats
        True
        >>> "code" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_OUTPUT_FORMATS)


def list_selection_strategies() -> list[str]:
    """List all available example selection strategies.

    Returns:
        Sorted list of selection strategy names.

    Examples:
        >>> strategies = list_selection_strategies()
        >>> "similarity" in strategies
        True
        >>> "random" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SELECTION_STRATEGIES)


def get_prompt_style(name: str) -> PromptStyle:
    """Get prompt style enum from string name.

    Args:
        name: Name of the prompt style.

    Returns:
        Corresponding PromptStyle enum.

    Raises:
        ValueError: If style name is invalid.

    Examples:
        >>> get_prompt_style("zero_shot")
        <PromptStyle.ZERO_SHOT: 'zero_shot'>
        >>> get_prompt_style("few_shot")
        <PromptStyle.FEW_SHOT: 'few_shot'>

        >>> get_prompt_style("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: style must be one of ...
    """
    if name not in VALID_PROMPT_STYLES:
        msg = f"style must be one of {VALID_PROMPT_STYLES}, got '{name}'"
        raise ValueError(msg)
    return PromptStyle(name)


def get_prompt_role(name: str) -> PromptRole:
    """Get prompt role enum from string name.

    Args:
        name: Name of the prompt role.

    Returns:
        Corresponding PromptRole enum.

    Raises:
        ValueError: If role name is invalid.

    Examples:
        >>> get_prompt_role("user")
        <PromptRole.USER: 'user'>
        >>> get_prompt_role("system")
        <PromptRole.SYSTEM: 'system'>

        >>> get_prompt_role("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: role must be one of ...
    """
    if name not in VALID_PROMPT_ROLES:
        msg = f"role must be one of {VALID_PROMPT_ROLES}, got '{name}'"
        raise ValueError(msg)
    return PromptRole(name)


def get_output_format(name: str) -> OutputFormat:
    """Get output format enum from string name.

    Args:
        name: Name of the output format.

    Returns:
        Corresponding OutputFormat enum.

    Raises:
        ValueError: If format name is invalid.

    Examples:
        >>> get_output_format("json")
        <OutputFormat.JSON: 'json'>
        >>> get_output_format("code")
        <OutputFormat.CODE: 'code'>

        >>> get_output_format("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: format must be one of ...
    """
    if name not in VALID_OUTPUT_FORMATS:
        msg = f"format must be one of {VALID_OUTPUT_FORMATS}, got '{name}'"
        raise ValueError(msg)
    return OutputFormat(name)


def get_selection_strategy(name: str) -> ExampleSelectionStrategy:
    """Get selection strategy enum from string name.

    Args:
        name: Name of the selection strategy.

    Returns:
        Corresponding ExampleSelectionStrategy enum.

    Raises:
        ValueError: If strategy name is invalid.

    Examples:
        >>> get_selection_strategy("similarity")
        <ExampleSelectionStrategy.SIMILARITY: 'similarity'>
        >>> get_selection_strategy("random")
        <ExampleSelectionStrategy.RANDOM: 'random'>

        >>> get_selection_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: strategy must be one of ...
    """
    if name not in VALID_SELECTION_STRATEGIES:
        msg = f"strategy must be one of {VALID_SELECTION_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return ExampleSelectionStrategy(name)


def format_prompt(
    template: PromptTemplate,
    **kwargs: str,
) -> str:
    """Format a prompt template with provided variables.

    Args:
        template: The prompt template.
        **kwargs: Variable values to substitute.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If required variables are missing.

    Examples:
        >>> template = create_prompt_template("Hello, {name}!")
        >>> format_prompt(template, name="World")
        'Hello, World!'

        >>> template = create_prompt_template(
        ...     "Translate '{text}' to {lang}.",
        ...     default_values={"lang": "French"},
        ... )
        >>> format_prompt(template, text="Hello")
        "Translate 'Hello' to French."

        >>> template = create_prompt_template("Hi, {name}!")
        >>> format_prompt(template)
        Traceback (most recent call last):
            ...
        ValueError: missing required variable: name
    """
    values = dict(template.default_values)
    values.update(kwargs)

    for var in template.variables:
        if var not in values:
            msg = f"missing required variable: {var}"
            raise ValueError(msg)

    return template.template.format(**values)


def format_few_shot_prompt(
    config: FewShotConfig,
    query: str,
    include_query: bool = True,
) -> str:
    """Format a few-shot prompt with examples and query.

    Args:
        config: Few-shot configuration.
        query: The input query to answer.
        include_query: Whether to include the query at the end.

    Returns:
        Formatted few-shot prompt string.

    Examples:
        >>> examples = (
        ...     create_few_shot_example("2+2", "4"),
        ...     create_few_shot_example("3+3", "6"),
        ... )
        >>> config = create_few_shot_config(examples)
        >>> prompt = format_few_shot_prompt(config, "5+5")
        >>> "Input: 2+2" in prompt
        True
        >>> "Output: 4" in prompt
        True
        >>> "Input: 5+5" in prompt
        True
    """
    parts = []

    selected_examples = config.examples[: config.num_examples]

    for example in selected_examples:
        example_text = (
            f"{config.input_prefix}{example.input_text}\n"
            f"{config.output_prefix}{example.output_text}"
        )
        parts.append(example_text)

    if include_query:
        parts.append(f"{config.input_prefix}{query}\n{config.output_prefix}")

    return config.example_separator.join(parts)


def format_system_prompt(config: SystemPromptConfig) -> str:
    """Format a system prompt from configuration.

    Args:
        config: System prompt configuration.

    Returns:
        Formatted system prompt string.

    Examples:
        >>> config = create_system_prompt_config(
        ...     "You are a helpful assistant.",
        ...     constraints=("Be concise", "Be accurate"),
        ...     format_instructions="Respond in JSON.",
        ... )
        >>> prompt = format_system_prompt(config)
        >>> "helpful assistant" in prompt
        True
        >>> "Be concise" in prompt
        True
        >>> "JSON" in prompt
        True
    """
    parts = [config.persona]

    if config.constraints:
        constraints_text = "\n".join(f"- {c}" for c in config.constraints)
        parts.append(f"\nConstraints:\n{constraints_text}")

    if config.format_instructions:
        parts.append(f"\n{config.format_instructions}")

    return "\n".join(parts)


def format_cot_prompt(
    config: ChainOfThoughtConfig,
    question: str,
) -> str:
    """Format a chain-of-thought prompt.

    Args:
        config: Chain-of-thought configuration.
        question: The question to reason about.

    Returns:
        Formatted chain-of-thought prompt string.

    Examples:
        >>> config = create_cot_config()
        >>> prompt = format_cot_prompt(config, "What is 2+2?")
        >>> "What is 2+2?" in prompt
        True
        >>> "step by step" in prompt
        True
    """
    parts = [question]

    if config.enable_reasoning:
        parts.append(f"\n\n{config.reasoning_prefix}")

    return "".join(parts)


def format_messages(messages: tuple[PromptMessage, ...]) -> str:
    """Format a sequence of messages into a prompt string.

    Args:
        messages: Tuple of prompt messages.

    Returns:
        Formatted conversation string.

    Examples:
        >>> messages = (
        ...     create_prompt_message("system", "You are helpful."),
        ...     create_prompt_message("user", "Hello!"),
        ... )
        >>> formatted = format_messages(messages)
        >>> "[system]" in formatted
        True
        >>> "You are helpful." in formatted
        True
    """
    parts = []
    for msg in messages:
        parts.append(f"[{msg.role.value}]\n{msg.content}")
    return "\n\n".join(parts)


def estimate_prompt_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in a prompt.

    Uses a simple character-based estimation. For more accurate
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
        >>> estimate_prompt_tokens("This is a longer prompt.", chars_per_token=5.0)
        4

        >>> estimate_prompt_tokens("test", chars_per_token=0)
        Traceback (most recent call last):
            ...
        ValueError: chars_per_token must be positive, got 0
    """
    if chars_per_token <= 0:
        msg = f"chars_per_token must be positive, got {chars_per_token}"
        raise ValueError(msg)

    return max(1, int(len(text) / chars_per_token)) if text else 0


def create_prompt_stats(
    text: str,
    num_examples: int = 0,
    has_system_prompt: bool = False,
    has_cot: bool = False,
) -> PromptStats:
    """Create statistics for a constructed prompt.

    Args:
        text: The prompt text.
        num_examples: Number of few-shot examples.
        has_system_prompt: Whether system prompt is present.
        has_cot: Whether chain-of-thought is enabled.

    Returns:
        PromptStats instance.

    Examples:
        >>> stats = create_prompt_stats("Hello, world!")
        >>> stats.total_chars
        13
        >>> stats.estimated_tokens
        3

        >>> stats = create_prompt_stats("Test", num_examples=3)
        >>> stats.num_examples
        3
    """
    return PromptStats(
        total_chars=len(text),
        estimated_tokens=estimate_prompt_tokens(text),
        num_examples=num_examples,
        has_system_prompt=has_system_prompt,
        has_cot=has_cot,
    )


def get_recommended_prompt_config(task: str) -> PromptConfig:
    """Get recommended prompt configuration for a task.

    Args:
        task: Task type (qa, summarization, classification, code, chat).

    Returns:
        Recommended PromptConfig for the task.

    Raises:
        ValueError: If task type is unknown.

    Examples:
        >>> config = get_recommended_prompt_config("qa")
        >>> config.style
        <PromptStyle.FEW_SHOT: 'few_shot'>

        >>> config = get_recommended_prompt_config("code")
        >>> config.style
        <PromptStyle.ZERO_SHOT: 'zero_shot'>

        >>> get_recommended_prompt_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task must be one of ...
    """
    valid_tasks = frozenset({"qa", "summarization", "classification", "code", "chat"})

    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    if task == "qa":
        return create_prompt_config(
            style=PromptStyle.FEW_SHOT,
            max_tokens=2048,
        )
    elif task == "summarization":
        return create_prompt_config(
            style=PromptStyle.ZERO_SHOT,
            max_tokens=4096,
        )
    elif task == "classification":
        return create_prompt_config(
            style=PromptStyle.FEW_SHOT,
            max_tokens=1024,
        )
    elif task == "code":
        return create_prompt_config(
            style=PromptStyle.ZERO_SHOT,
            max_tokens=4096,
        )
    else:  # chat
        return create_prompt_config(
            style=PromptStyle.ZERO_SHOT,
            max_tokens=4096,
        )


def format_prompt_stats(stats: PromptStats) -> str:
    """Format prompt statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_prompt_stats("Hello!", num_examples=2)
        >>> formatted = format_prompt_stats(stats)
        >>> "Characters: 6" in formatted
        True
        >>> "Examples: 2" in formatted
        True
    """
    return (
        f"Prompt Stats:\n"
        f"  Characters: {stats.total_chars}\n"
        f"  Est. Tokens: {stats.estimated_tokens}\n"
        f"  Examples: {stats.num_examples}\n"
        f"  System Prompt: {stats.has_system_prompt}\n"
        f"  Chain-of-Thought: {stats.has_cot}"
    )


def build_conversation_prompt(
    system_config: SystemPromptConfig | None,
    messages: tuple[PromptMessage, ...],
    few_shot_config: FewShotConfig | None = None,
) -> tuple[PromptMessage, ...]:
    """Build a complete conversation prompt.

    Combines system prompt, optional few-shot examples, and messages
    into a properly ordered conversation.

    Args:
        system_config: Optional system prompt configuration.
        messages: User/assistant message history.
        few_shot_config: Optional few-shot configuration.

    Returns:
        Complete tuple of PromptMessages.

    Examples:
        >>> system = create_system_prompt_config("You are helpful.")
        >>> messages = (create_prompt_message("user", "Hello!"),)
        >>> conversation = build_conversation_prompt(system, messages)
        >>> len(conversation)
        2
        >>> conversation[0].role
        <PromptRole.SYSTEM: 'system'>
    """
    result: list[PromptMessage] = []

    if system_config:
        system_text = format_system_prompt(system_config)
        result.append(PromptMessage(role=PromptRole.SYSTEM, content=system_text))

    if few_shot_config and system_config and system_config.examples_in_system:
        examples_text = format_few_shot_prompt(few_shot_config, "", include_query=False)
        if result:
            current = result[0]
            updated_content = f"{current.content}\n\nExamples:\n{examples_text}"
            result[0] = PromptMessage(role=current.role, content=updated_content)

    result.extend(messages)

    return tuple(result)


def truncate_prompt(
    text: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    truncation_marker: str = "...",
) -> str:
    """Truncate a prompt to fit within token limit.

    Args:
        text: The prompt text.
        max_tokens: Maximum allowed tokens.
        chars_per_token: Average characters per token.
        truncation_marker: Marker to indicate truncation.

    Returns:
        Truncated prompt string.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> truncate_prompt("Hello, world!", max_tokens=2)
        'Hello...'

        >>> truncate_prompt("Short", max_tokens=100)
        'Short'

        >>> truncate_prompt("Test", max_tokens=0)
        Traceback (most recent call last):
            ...
        ValueError: max_tokens must be positive, got 0
    """
    if max_tokens <= 0:
        msg = f"max_tokens must be positive, got {max_tokens}"
        raise ValueError(msg)
    if chars_per_token <= 0:
        msg = f"chars_per_token must be positive, got {chars_per_token}"
        raise ValueError(msg)

    current_tokens = estimate_prompt_tokens(text, chars_per_token)

    if current_tokens <= max_tokens:
        return text

    max_chars = int(max_tokens * chars_per_token) - len(truncation_marker)
    if max_chars <= 0:
        return truncation_marker

    return text[:max_chars] + truncation_marker
