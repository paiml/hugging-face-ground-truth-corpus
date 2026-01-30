"""Chat utilities for conversational AI.

This module provides functions for managing chat conversations,
including message formatting, template configuration, and token counting.

Examples:
    >>> from hf_gtc.generation.chat import create_chat_message, MessageRole
    >>> msg = create_chat_message(role="user", content="Hello!")
    >>> msg.role
    <MessageRole.USER: 'user'>
    >>> msg.content
    'Hello!'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class MessageRole(Enum):
    """Role of a chat message participant.

    Attributes:
        SYSTEM: System-level instructions.
        USER: User/human input.
        ASSISTANT: AI assistant response.
        FUNCTION: Function call result (legacy).
        TOOL: Tool call result.

    Examples:
        >>> MessageRole.USER.value
        'user'
        >>> MessageRole.ASSISTANT.value
        'assistant'
        >>> MessageRole.SYSTEM.value
        'system'
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatTemplateFormat(Enum):
    """Chat template formats for different model families.

    Attributes:
        CHATML: OpenAI ChatML format (<|im_start|>role).
        LLAMA: Llama 2/3 instruction format.
        ALPACA: Stanford Alpaca format.
        VICUNA: Vicuna conversation format.
        MISTRAL: Mistral instruction format.
        ZEPHYR: HuggingFace Zephyr format.

    Examples:
        >>> ChatTemplateFormat.CHATML.value
        'chatml'
        >>> ChatTemplateFormat.LLAMA.value
        'llama'
        >>> ChatTemplateFormat.MISTRAL.value
        'mistral'
    """

    CHATML = "chatml"
    LLAMA = "llama"
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    MISTRAL = "mistral"
    ZEPHYR = "zephyr"


VALID_MESSAGE_ROLES = frozenset(r.value for r in MessageRole)
VALID_TEMPLATE_FORMATS = frozenset(f.value for f in ChatTemplateFormat)

# Truncation strategy types
TruncationStrategyType = Literal["oldest_first", "newest_first", "middle_out"]
VALID_TRUNCATION_STRATEGIES = frozenset({"oldest_first", "newest_first", "middle_out"})


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        name: Optional name for function/tool messages.
        tool_calls: Optional tool call information (JSON string).
        tool_call_id: Optional ID linking tool result to tool call.

    Examples:
        >>> msg = ChatMessage(
        ...     role=MessageRole.USER,
        ...     content="What is the weather?",
        ...     name=None,
        ...     tool_calls=None,
        ...     tool_call_id=None,
        ... )
        >>> msg.role
        <MessageRole.USER: 'user'>
        >>> msg.content
        'What is the weather?'

        >>> tool_msg = ChatMessage(
        ...     role=MessageRole.TOOL,
        ...     content='{"temp": 72}',
        ...     name="get_weather",
        ...     tool_calls=None,
        ...     tool_call_id="call_123",
        ... )
        >>> tool_msg.tool_call_id
        'call_123'
    """

    role: MessageRole
    content: str
    name: str | None
    tool_calls: str | None
    tool_call_id: str | None


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """Configuration for chat formatting.

    Attributes:
        template_format: The template format to use.
        system_prompt: Default system prompt.
        max_turns: Maximum conversation turns to include.
        include_system: Whether to include system message.

    Examples:
        >>> config = ChatConfig(
        ...     template_format=ChatTemplateFormat.CHATML,
        ...     system_prompt="You are a helpful assistant.",
        ...     max_turns=10,
        ...     include_system=True,
        ... )
        >>> config.template_format
        <ChatTemplateFormat.CHATML: 'chatml'>
        >>> config.max_turns
        10
    """

    template_format: ChatTemplateFormat
    system_prompt: str | None
    max_turns: int
    include_system: bool


@dataclass(frozen=True, slots=True)
class TemplateConfig:
    """Token configuration for chat templates.

    Attributes:
        bos_token: Beginning of sequence token.
        eos_token: End of sequence token.
        user_prefix: Prefix before user messages.
        assistant_prefix: Prefix before assistant messages.
        system_prefix: Prefix before system messages.

    Examples:
        >>> config = TemplateConfig(
        ...     bos_token="<s>",
        ...     eos_token="</s>",
        ...     user_prefix="[INST]",
        ...     assistant_prefix="[/INST]",
        ...     system_prefix="<<SYS>>",
        ... )
        >>> config.bos_token
        '<s>'
        >>> config.eos_token
        '</s>'
    """

    bos_token: str
    eos_token: str
    user_prefix: str
    assistant_prefix: str
    system_prefix: str


@dataclass(frozen=True, slots=True)
class ConversationConfig:
    """Configuration for conversation management.

    Attributes:
        max_history: Maximum messages to keep in history.
        truncation_strategy: How to truncate when over limit.
        preserve_system: Whether to always keep system message.

    Examples:
        >>> config = ConversationConfig(
        ...     max_history=50,
        ...     truncation_strategy="oldest_first",
        ...     preserve_system=True,
        ... )
        >>> config.max_history
        50
        >>> config.preserve_system
        True
    """

    max_history: int
    truncation_strategy: TruncationStrategyType
    preserve_system: bool


@dataclass(frozen=True, slots=True)
class ChatStats:
    """Statistics about a chat conversation.

    Attributes:
        total_turns: Total number of conversation turns.
        user_tokens: Estimated tokens in user messages.
        assistant_tokens: Estimated tokens in assistant messages.
        system_tokens: Estimated tokens in system messages.

    Examples:
        >>> stats = ChatStats(
        ...     total_turns=5,
        ...     user_tokens=150,
        ...     assistant_tokens=300,
        ...     system_tokens=50,
        ... )
        >>> stats.total_turns
        5
        >>> stats.user_tokens + stats.assistant_tokens + stats.system_tokens
        500
    """

    total_turns: int
    user_tokens: int
    assistant_tokens: int
    system_tokens: int


def validate_chat_message(message: ChatMessage) -> None:
    """Validate a chat message.

    Args:
        message: ChatMessage to validate.

    Raises:
        ValueError: If message is None.
        ValueError: If content is None.
        ValueError: If tool role lacks tool_call_id.

    Examples:
        >>> msg = ChatMessage(
        ...     role=MessageRole.USER,
        ...     content="Hello",
        ...     name=None,
        ...     tool_calls=None,
        ...     tool_call_id=None,
        ... )
        >>> validate_chat_message(msg)  # No error

        >>> validate_chat_message(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: message cannot be None

        >>> bad_msg = ChatMessage(
        ...     role=MessageRole.TOOL,
        ...     content="result",
        ...     name=None,
        ...     tool_calls=None,
        ...     tool_call_id=None,
        ... )
        >>> validate_chat_message(bad_msg)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tool messages must have tool_call_id
    """
    if message is None:
        msg = "message cannot be None"
        raise ValueError(msg)

    if message.content is None:
        msg = "message content cannot be None"
        raise ValueError(msg)

    if message.role == MessageRole.TOOL and not message.tool_call_id:
        msg = "tool messages must have tool_call_id"
        raise ValueError(msg)


def validate_chat_config(config: ChatConfig) -> None:
    """Validate chat configuration.

    Args:
        config: ChatConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_turns is not positive.

    Examples:
        >>> config = ChatConfig(
        ...     template_format=ChatTemplateFormat.CHATML,
        ...     system_prompt="Be helpful.",
        ...     max_turns=10,
        ...     include_system=True,
        ... )
        >>> validate_chat_config(config)  # No error

        >>> validate_chat_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = ChatConfig(
        ...     template_format=ChatTemplateFormat.CHATML,
        ...     system_prompt=None,
        ...     max_turns=0,
        ...     include_system=True,
        ... )
        >>> validate_chat_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_turns must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.max_turns <= 0:
        msg = f"max_turns must be positive, got {config.max_turns}"
        raise ValueError(msg)


def create_chat_message(
    role: str,
    content: str,
    *,
    name: str | None = None,
    tool_calls: str | None = None,
    tool_call_id: str | None = None,
) -> ChatMessage:
    """Create a chat message.

    Args:
        role: Message role (system, user, assistant, function, tool).
        content: Message content text.
        name: Optional name for function/tool messages. Defaults to None.
        tool_calls: Optional tool calls JSON. Defaults to None.
        tool_call_id: Optional tool call ID. Defaults to None.

    Returns:
        ChatMessage with the specified settings.

    Raises:
        ValueError: If role is invalid.
        ValueError: If content is empty.

    Examples:
        >>> msg = create_chat_message("user", "Hello!")
        >>> msg.role
        <MessageRole.USER: 'user'>
        >>> msg.content
        'Hello!'

        >>> tool_msg = create_chat_message(
        ...     "tool",
        ...     '{"result": 42}',
        ...     name="calculator",
        ...     tool_call_id="call_abc",
        ... )
        >>> tool_msg.name
        'calculator'

        >>> create_chat_message("invalid", "test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: role must be one of ...

        >>> create_chat_message("user", "")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: content cannot be empty
    """
    if role not in VALID_MESSAGE_ROLES:
        msg = f"role must be one of {VALID_MESSAGE_ROLES}, got {role!r}"
        raise ValueError(msg)

    if not content:
        msg = "content cannot be empty"
        raise ValueError(msg)

    message = ChatMessage(
        role=MessageRole(role),
        content=content,
        name=name,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    validate_chat_message(message)
    return message


def create_chat_config(
    template_format: str = "chatml",
    system_prompt: str | None = None,
    max_turns: int = 10,
    include_system: bool = True,
) -> ChatConfig:
    """Create a chat configuration.

    Args:
        template_format: Template format to use. Defaults to "chatml".
        system_prompt: System prompt. Defaults to None.
        max_turns: Maximum turns to include. Defaults to 10.
        include_system: Whether to include system. Defaults to True.

    Returns:
        ChatConfig with the specified settings.

    Raises:
        ValueError: If template_format is invalid.
        ValueError: If max_turns is not positive.

    Examples:
        >>> config = create_chat_config(template_format="llama", max_turns=20)
        >>> config.template_format
        <ChatTemplateFormat.LLAMA: 'llama'>
        >>> config.max_turns
        20

        >>> config = create_chat_config(system_prompt="Be concise.")
        >>> config.system_prompt
        'Be concise.'

        >>> create_chat_config(template_format="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: template_format must be one of ...

        >>> create_chat_config(max_turns=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_turns must be positive
    """
    if template_format not in VALID_TEMPLATE_FORMATS:
        msg = (
            f"template_format must be one of {VALID_TEMPLATE_FORMATS}, "
            f"got {template_format!r}"
        )
        raise ValueError(msg)

    config = ChatConfig(
        template_format=ChatTemplateFormat(template_format),
        system_prompt=system_prompt,
        max_turns=max_turns,
        include_system=include_system,
    )
    validate_chat_config(config)
    return config


def create_template_config(
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
    system_prefix: str = "System: ",
) -> TemplateConfig:
    """Create a template token configuration.

    Args:
        bos_token: Beginning of sequence token. Defaults to "<s>".
        eos_token: End of sequence token. Defaults to "</s>".
        user_prefix: User message prefix. Defaults to "User: ".
        assistant_prefix: Assistant message prefix. Defaults to "Assistant: ".
        system_prefix: System message prefix. Defaults to "System: ".

    Returns:
        TemplateConfig with the specified settings.

    Examples:
        >>> config = create_template_config(bos_token="[BOS]", eos_token="[EOS]")
        >>> config.bos_token
        '[BOS]'
        >>> config.eos_token
        '[EOS]'

        >>> config = create_template_config()
        >>> config.user_prefix
        'User: '
    """
    return TemplateConfig(
        bos_token=bos_token,
        eos_token=eos_token,
        user_prefix=user_prefix,
        assistant_prefix=assistant_prefix,
        system_prefix=system_prefix,
    )


def create_conversation_config(
    max_history: int = 50,
    truncation_strategy: TruncationStrategyType = "oldest_first",
    preserve_system: bool = True,
) -> ConversationConfig:
    """Create a conversation management configuration.

    Args:
        max_history: Maximum messages in history. Defaults to 50.
        truncation_strategy: Truncation method. Defaults to "oldest_first".
        preserve_system: Keep system message. Defaults to True.

    Returns:
        ConversationConfig with the specified settings.

    Raises:
        ValueError: If max_history is not positive.
        ValueError: If truncation_strategy is invalid.

    Examples:
        >>> config = create_conversation_config(max_history=100)
        >>> config.max_history
        100

        >>> config = create_conversation_config(truncation_strategy="newest_first")
        >>> config.truncation_strategy
        'newest_first'

        >>> create_conversation_config(max_history=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_history must be positive

        >>> create_conversation_config(truncation_strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: truncation_strategy must be one of ...
    """
    if max_history <= 0:
        msg = f"max_history must be positive, got {max_history}"
        raise ValueError(msg)

    if truncation_strategy not in VALID_TRUNCATION_STRATEGIES:
        msg = (
            f"truncation_strategy must be one of {VALID_TRUNCATION_STRATEGIES}, "
            f"got {truncation_strategy!r}"
        )
        raise ValueError(msg)

    return ConversationConfig(
        max_history=max_history,
        truncation_strategy=truncation_strategy,
        preserve_system=preserve_system,
    )


def list_message_roles() -> list[str]:
    """List all available message roles.

    Returns:
        Sorted list of message role names.

    Examples:
        >>> roles = list_message_roles()
        >>> "user" in roles
        True
        >>> "assistant" in roles
        True
        >>> "system" in roles
        True
        >>> roles == sorted(roles)
        True
    """
    return sorted(VALID_MESSAGE_ROLES)


def list_template_formats() -> list[str]:
    """List all available chat template formats.

    Returns:
        Sorted list of template format names.

    Examples:
        >>> formats = list_template_formats()
        >>> "chatml" in formats
        True
        >>> "llama" in formats
        True
        >>> "mistral" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_TEMPLATE_FORMATS)


def get_message_role(name: str) -> MessageRole:
    """Get message role from name.

    Args:
        name: Role name.

    Returns:
        MessageRole enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_message_role("user")
        <MessageRole.USER: 'user'>

        >>> get_message_role("assistant")
        <MessageRole.ASSISTANT: 'assistant'>

        >>> get_message_role("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: role must be one of ...
    """
    if name not in VALID_MESSAGE_ROLES:
        msg = f"role must be one of {VALID_MESSAGE_ROLES}, got {name!r}"
        raise ValueError(msg)
    return MessageRole(name)


def get_template_format(name: str) -> ChatTemplateFormat:
    """Get template format from name.

    Args:
        name: Template format name.

    Returns:
        ChatTemplateFormat enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_template_format("chatml")
        <ChatTemplateFormat.CHATML: 'chatml'>

        >>> get_template_format("llama")
        <ChatTemplateFormat.LLAMA: 'llama'>

        >>> get_template_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: template_format must be one of ...
    """
    if name not in VALID_TEMPLATE_FORMATS:
        msg = f"template_format must be one of {VALID_TEMPLATE_FORMATS}, got {name!r}"
        raise ValueError(msg)
    return ChatTemplateFormat(name)


def _get_template_tokens(template_format: ChatTemplateFormat) -> TemplateConfig:
    r"""Get template tokens for a specific format.

    Args:
        template_format: The template format.

    Returns:
        TemplateConfig with format-specific tokens.

    Examples:
        >>> config = _get_template_tokens(ChatTemplateFormat.CHATML)
        >>> config.user_prefix
        '<|im_start|>user\n'

        >>> config = _get_template_tokens(ChatTemplateFormat.LLAMA)
        >>> config.bos_token
        '<s>'
    """
    templates = {
        ChatTemplateFormat.CHATML: TemplateConfig(
            bos_token="",
            eos_token="<|im_end|>\n",
            user_prefix="<|im_start|>user\n",
            assistant_prefix="<|im_start|>assistant\n",
            system_prefix="<|im_start|>system\n",
        ),
        ChatTemplateFormat.LLAMA: TemplateConfig(
            bos_token="<s>",
            eos_token="</s>",
            user_prefix="[INST] ",
            assistant_prefix=" [/INST] ",
            system_prefix="<<SYS>>\n",
        ),
        ChatTemplateFormat.ALPACA: TemplateConfig(
            bos_token="",
            eos_token="\n\n",
            user_prefix="### Instruction:\n",
            assistant_prefix="### Response:\n",
            system_prefix="### System:\n",
        ),
        ChatTemplateFormat.VICUNA: TemplateConfig(
            bos_token="",
            eos_token="</s>",
            user_prefix="USER: ",
            assistant_prefix="ASSISTANT: ",
            system_prefix="SYSTEM: ",
        ),
        ChatTemplateFormat.MISTRAL: TemplateConfig(
            bos_token="<s>",
            eos_token="</s>",
            user_prefix="[INST] ",
            assistant_prefix=" [/INST]",
            system_prefix="",
        ),
        ChatTemplateFormat.ZEPHYR: TemplateConfig(
            bos_token="<|system|>",
            eos_token="</s>",
            user_prefix="<|user|>\n",
            assistant_prefix="<|assistant|>\n",
            system_prefix="<|system|>\n",
        ),
    }
    return templates[template_format]


def format_chat_prompt(
    messages: list[ChatMessage],
    config: ChatConfig,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages into a prompt string.

    Args:
        messages: List of chat messages.
        config: Chat configuration.
        add_generation_prompt: Add assistant prefix at end. Defaults to True.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If messages is empty.
        ValueError: If config is None.

    Examples:
        >>> messages = [
        ...     create_chat_message("user", "Hello!"),
        ... ]
        >>> config = create_chat_config(template_format="chatml")
        >>> prompt = format_chat_prompt(messages, config)
        >>> "<|im_start|>user" in prompt
        True
        >>> "Hello!" in prompt
        True

        >>> format_chat_prompt([], config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: messages cannot be empty

        >>> format_chat_prompt(messages, None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if not messages:
        msg = "messages cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_chat_config(config)

    template = _get_template_tokens(config.template_format)
    parts: list[str] = []

    # Handle system prompt
    if config.include_system and config.system_prompt:
        parts.append(template.system_prefix)
        parts.append(config.system_prompt)
        parts.append(template.eos_token)

    # Limit to max_turns (each turn = user + assistant pair)
    turns_count = 0
    filtered_messages: list[ChatMessage] = []
    for message in messages:
        if message.role == MessageRole.USER:
            turns_count += 1
            if turns_count > config.max_turns:
                break
        filtered_messages.append(message)

    # Format each message
    for message in filtered_messages:
        if message.role == MessageRole.SYSTEM:
            if not config.include_system:
                continue
            parts.append(template.system_prefix)
            parts.append(message.content)
            parts.append(template.eos_token)
        elif message.role == MessageRole.USER:
            parts.append(template.user_prefix)
            parts.append(message.content)
            parts.append(template.eos_token)
        elif message.role == MessageRole.ASSISTANT:
            parts.append(template.assistant_prefix)
            parts.append(message.content)
            parts.append(template.eos_token)
        elif message.role in (MessageRole.TOOL, MessageRole.FUNCTION):
            # Tool/function results are typically formatted as user messages
            parts.append(template.user_prefix)
            parts.append(f"[{message.name or 'tool'}]: {message.content}")
            parts.append(template.eos_token)

    # Add generation prompt for assistant
    if add_generation_prompt:
        parts.append(template.assistant_prefix)

    return "".join(parts)


def count_conversation_tokens(
    messages: list[ChatMessage],
    *,
    chars_per_token: float = 4.0,
) -> ChatStats:
    """Estimate token counts in a conversation.

    Uses a simple character-based estimation. For accurate counts,
    use a tokenizer specific to your model.

    Args:
        messages: List of chat messages.
        chars_per_token: Average characters per token. Defaults to 4.0.

    Returns:
        ChatStats with estimated token counts.

    Raises:
        ValueError: If messages is None.
        ValueError: If chars_per_token is not positive.

    Examples:
        >>> messages = [
        ...     create_chat_message("system", "Be helpful."),
        ...     create_chat_message("user", "Hello there!"),
        ...     create_chat_message("assistant", "Hi! How can I help?"),
        ... ]
        >>> stats = count_conversation_tokens(messages)
        >>> stats.total_turns
        1
        >>> stats.system_tokens > 0
        True
        >>> stats.user_tokens > 0
        True
        >>> stats.assistant_tokens > 0
        True

        >>> count_conversation_tokens(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: messages cannot be None

        >>> count_conversation_tokens([], chars_per_token=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chars_per_token must be positive
    """
    if messages is None:
        msg = "messages cannot be None"
        raise ValueError(msg)

    if chars_per_token <= 0:
        msg = f"chars_per_token must be positive, got {chars_per_token}"
        raise ValueError(msg)

    user_tokens = 0
    assistant_tokens = 0
    system_tokens = 0
    user_turns = 0

    for message in messages:
        token_estimate = int(len(message.content) / chars_per_token)

        if message.role == MessageRole.SYSTEM:
            system_tokens += token_estimate
        elif message.role == MessageRole.USER:
            user_tokens += token_estimate
            user_turns += 1
        elif message.role == MessageRole.ASSISTANT:
            assistant_tokens += token_estimate
        else:
            # Tool/function messages count as user tokens
            user_tokens += token_estimate

    return ChatStats(
        total_turns=user_turns,
        user_tokens=user_tokens,
        assistant_tokens=assistant_tokens,
        system_tokens=system_tokens,
    )
