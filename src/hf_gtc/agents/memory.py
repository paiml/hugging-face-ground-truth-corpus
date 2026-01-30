"""Agent memory utilities.

This module provides functions for managing agent memory, including
buffer memory, window memory, summary memory, and entity memory types.

Examples:
    >>> from hf_gtc.agents.memory import create_buffer_config
    >>> config = create_buffer_config(max_messages=100, max_tokens=4096)
    >>> config.max_messages
    100
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class MemoryType(Enum):
    """Supported memory types.

    Attributes:
        BUFFER: Simple buffer memory storing all messages.
        SUMMARY: Summarized memory with condensed history.
        WINDOW: Sliding window memory with recent messages.
        ENTITY: Entity-based memory tracking named entities.
        CONVERSATION: Full conversation memory with metadata.

    Examples:
        >>> MemoryType.BUFFER.value
        'buffer'
        >>> MemoryType.SUMMARY.value
        'summary'
        >>> MemoryType.WINDOW.value
        'window'
    """

    BUFFER = "buffer"
    SUMMARY = "summary"
    WINDOW = "window"
    ENTITY = "entity"
    CONVERSATION = "conversation"


VALID_MEMORY_TYPES = frozenset(t.value for t in MemoryType)


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Configuration for buffer memory.

    Attributes:
        max_messages: Maximum number of messages to store.
        max_tokens: Maximum total tokens to store.
        return_messages: Whether to return messages as list.

    Examples:
        >>> config = BufferConfig(
        ...     max_messages=100,
        ...     max_tokens=4096,
        ...     return_messages=True,
        ... )
        >>> config.max_messages
        100
    """

    max_messages: int
    max_tokens: int
    return_messages: bool


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Configuration for window memory.

    Attributes:
        window_size: Number of messages in the window.
        overlap: Number of overlapping messages between windows.
        include_system: Whether to include system messages.

    Examples:
        >>> config = WindowConfig(
        ...     window_size=10,
        ...     overlap=2,
        ...     include_system=True,
        ... )
        >>> config.window_size
        10
    """

    window_size: int
    overlap: int
    include_system: bool


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    """Configuration for summary memory.

    Attributes:
        max_summary_tokens: Maximum tokens in the summary.
        summarizer_model: Model to use for summarization.
        update_frequency: How often to update summary (in messages).

    Examples:
        >>> config = SummaryConfig(
        ...     max_summary_tokens=500,
        ...     summarizer_model="gpt-3.5-turbo",
        ...     update_frequency=5,
        ... )
        >>> config.max_summary_tokens
        500
    """

    max_summary_tokens: int
    summarizer_model: str
    update_frequency: int


@dataclass(frozen=True, slots=True)
class EntityConfig:
    """Configuration for entity memory.

    Attributes:
        entity_extraction_model: Model to use for entity extraction.
        max_entities: Maximum number of entities to track.
        decay_rate: Rate at which entity relevance decays (0.0 to 1.0).

    Examples:
        >>> config = EntityConfig(
        ...     entity_extraction_model="gpt-3.5-turbo",
        ...     max_entities=50,
        ...     decay_rate=0.1,
        ... )
        >>> config.max_entities
        50
    """

    entity_extraction_model: str
    max_entities: int
    decay_rate: float


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Main configuration for agent memory.

    Attributes:
        memory_type: Type of memory to use.
        human_prefix: Prefix for human messages.
        ai_prefix: Prefix for AI messages.
        input_key: Key for input in memory.
        output_key: Key for output in memory.

    Examples:
        >>> config = MemoryConfig(
        ...     memory_type=MemoryType.BUFFER,
        ...     human_prefix="Human",
        ...     ai_prefix="AI",
        ...     input_key="input",
        ...     output_key="output",
        ... )
        >>> config.memory_type
        <MemoryType.BUFFER: 'buffer'>
    """

    memory_type: MemoryType
    human_prefix: str
    ai_prefix: str
    input_key: str
    output_key: str


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """Represents a message in a conversation.

    Attributes:
        role: Role of the message sender (e.g., "human", "ai", "system").
        content: Message content.
        timestamp: Message timestamp.
        metadata: Additional metadata.

    Examples:
        >>> from datetime import datetime
        >>> msg = ConversationMessage(
        ...     role="human",
        ...     content="Hello!",
        ...     timestamp=datetime(2024, 1, 1, 12, 0, 0),
        ...     metadata={},
        ... )
        >>> msg.role
        'human'
    """

    role: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Statistics for memory usage.

    Attributes:
        total_messages: Total number of messages stored.
        total_tokens: Total number of tokens stored.
        memory_size_bytes: Memory size in bytes.

    Examples:
        >>> stats = MemoryStats(
        ...     total_messages=50,
        ...     total_tokens=2048,
        ...     memory_size_bytes=8192,
        ... )
        >>> stats.total_messages
        50
    """

    total_messages: int
    total_tokens: int
    memory_size_bytes: int


def validate_buffer_config(config: BufferConfig) -> None:
    """Validate buffer configuration.

    Args:
        config: Buffer configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = BufferConfig(100, 4096, True)
        >>> validate_buffer_config(config)  # No error

        >>> bad = BufferConfig(0, 4096, True)
        >>> validate_buffer_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_messages must be positive

        >>> bad = BufferConfig(100, -1, True)
        >>> validate_buffer_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_tokens must be non-negative
    """
    if config.max_messages <= 0:
        msg = f"max_messages must be positive, got {config.max_messages}"
        raise ValueError(msg)

    if config.max_tokens < 0:
        msg = f"max_tokens must be non-negative, got {config.max_tokens}"
        raise ValueError(msg)


def validate_window_config(config: WindowConfig) -> None:
    """Validate window configuration.

    Args:
        config: Window configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = WindowConfig(10, 2, True)
        >>> validate_window_config(config)  # No error

        >>> bad = WindowConfig(0, 2, True)
        >>> validate_window_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be positive

        >>> bad = WindowConfig(10, -1, True)
        >>> validate_window_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overlap must be non-negative

        >>> bad = WindowConfig(10, 15, True)
        >>> validate_window_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overlap (15) must be less than window_size (10)
    """
    if config.window_size <= 0:
        msg = f"window_size must be positive, got {config.window_size}"
        raise ValueError(msg)

    if config.overlap < 0:
        msg = f"overlap must be non-negative, got {config.overlap}"
        raise ValueError(msg)

    if config.overlap >= config.window_size:
        msg = (
            f"overlap ({config.overlap}) must be less than "
            f"window_size ({config.window_size})"
        )
        raise ValueError(msg)


def validate_summary_config(config: SummaryConfig) -> None:
    """Validate summary configuration.

    Args:
        config: Summary configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SummaryConfig(500, "gpt-3.5-turbo", 5)
        >>> validate_summary_config(config)  # No error

        >>> bad = SummaryConfig(0, "gpt-3.5-turbo", 5)
        >>> validate_summary_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_summary_tokens must be positive

        >>> bad = SummaryConfig(500, "", 5)
        >>> validate_summary_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: summarizer_model cannot be empty

        >>> bad = SummaryConfig(500, "gpt-3.5-turbo", 0)
        >>> validate_summary_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: update_frequency must be positive
    """
    if config.max_summary_tokens <= 0:
        msg = f"max_summary_tokens must be positive, got {config.max_summary_tokens}"
        raise ValueError(msg)

    if not config.summarizer_model:
        msg = "summarizer_model cannot be empty"
        raise ValueError(msg)

    if config.update_frequency <= 0:
        msg = f"update_frequency must be positive, got {config.update_frequency}"
        raise ValueError(msg)


def validate_entity_config(config: EntityConfig) -> None:
    """Validate entity configuration.

    Args:
        config: Entity configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = EntityConfig("gpt-3.5-turbo", 50, 0.1)
        >>> validate_entity_config(config)  # No error

        >>> bad = EntityConfig("", 50, 0.1)
        >>> validate_entity_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entity_extraction_model cannot be empty

        >>> bad = EntityConfig("gpt-3.5-turbo", 0, 0.1)
        >>> validate_entity_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_entities must be positive

        >>> bad = EntityConfig("gpt-3.5-turbo", 50, 1.5)
        >>> validate_entity_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: decay_rate must be between 0.0 and 1.0
    """
    if not config.entity_extraction_model:
        msg = "entity_extraction_model cannot be empty"
        raise ValueError(msg)

    if config.max_entities <= 0:
        msg = f"max_entities must be positive, got {config.max_entities}"
        raise ValueError(msg)

    if not 0.0 <= config.decay_rate <= 1.0:
        msg = f"decay_rate must be between 0.0 and 1.0, got {config.decay_rate}"
        raise ValueError(msg)


def validate_memory_config(config: MemoryConfig) -> None:
    """Validate memory configuration.

    Args:
        config: Memory configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "input", "output")
        >>> validate_memory_config(config)  # No error

        >>> bad = MemoryConfig(MemoryType.BUFFER, "", "AI", "input", "output")
        >>> validate_memory_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: human_prefix cannot be empty

        >>> bad = MemoryConfig(MemoryType.BUFFER, "Human", "", "input", "output")
        >>> validate_memory_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ai_prefix cannot be empty

        >>> bad = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "", "output")
        >>> validate_memory_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: input_key cannot be empty
    """
    if not config.human_prefix:
        msg = "human_prefix cannot be empty"
        raise ValueError(msg)

    if not config.ai_prefix:
        msg = "ai_prefix cannot be empty"
        raise ValueError(msg)

    if not config.input_key:
        msg = "input_key cannot be empty"
        raise ValueError(msg)

    if not config.output_key:
        msg = "output_key cannot be empty"
        raise ValueError(msg)


def create_buffer_config(
    max_messages: int = 100,
    max_tokens: int = 4096,
    return_messages: bool = True,
) -> BufferConfig:
    """Create a buffer memory configuration.

    Args:
        max_messages: Maximum messages to store. Defaults to 100.
        max_tokens: Maximum tokens to store. Defaults to 4096.
        return_messages: Whether to return messages as list. Defaults to True.

    Returns:
        BufferConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_buffer_config(max_messages=50)
        >>> config.max_messages
        50

        >>> config = create_buffer_config(max_tokens=8192)
        >>> config.max_tokens
        8192

        >>> create_buffer_config(max_messages=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_messages must be positive
    """
    config = BufferConfig(
        max_messages=max_messages,
        max_tokens=max_tokens,
        return_messages=return_messages,
    )
    validate_buffer_config(config)
    return config


def create_window_config(
    window_size: int = 10,
    overlap: int = 2,
    include_system: bool = True,
) -> WindowConfig:
    """Create a window memory configuration.

    Args:
        window_size: Number of messages in window. Defaults to 10.
        overlap: Number of overlapping messages. Defaults to 2.
        include_system: Include system messages. Defaults to True.

    Returns:
        WindowConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_window_config(window_size=20)
        >>> config.window_size
        20

        >>> config = create_window_config(overlap=5)
        >>> config.overlap
        5

        >>> create_window_config(window_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be positive

        >>> create_window_config(window_size=10, overlap=15)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overlap (15) must be less than window_size (10)
    """
    config = WindowConfig(
        window_size=window_size,
        overlap=overlap,
        include_system=include_system,
    )
    validate_window_config(config)
    return config


def create_summary_config(
    max_summary_tokens: int = 500,
    summarizer_model: str = "gpt-3.5-turbo",
    update_frequency: int = 5,
) -> SummaryConfig:
    """Create a summary memory configuration.

    Args:
        max_summary_tokens: Maximum summary tokens. Defaults to 500.
        summarizer_model: Model for summarization. Defaults to "gpt-3.5-turbo".
        update_frequency: How often to update summary. Defaults to 5.

    Returns:
        SummaryConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_summary_config(max_summary_tokens=1000)
        >>> config.max_summary_tokens
        1000

        >>> config = create_summary_config(summarizer_model="gpt-4")
        >>> config.summarizer_model
        'gpt-4'

        >>> create_summary_config(max_summary_tokens=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_summary_tokens must be positive

        >>> create_summary_config(summarizer_model="")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: summarizer_model cannot be empty
    """
    config = SummaryConfig(
        max_summary_tokens=max_summary_tokens,
        summarizer_model=summarizer_model,
        update_frequency=update_frequency,
    )
    validate_summary_config(config)
    return config


def create_entity_config(
    entity_extraction_model: str = "gpt-3.5-turbo",
    max_entities: int = 50,
    decay_rate: float = 0.1,
) -> EntityConfig:
    """Create an entity memory configuration.

    Args:
        entity_extraction_model: Model for entity extraction.
            Defaults to "gpt-3.5-turbo".
        max_entities: Maximum entities to track. Defaults to 50.
        decay_rate: Entity relevance decay rate. Defaults to 0.1.

    Returns:
        EntityConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_entity_config(max_entities=100)
        >>> config.max_entities
        100

        >>> config = create_entity_config(decay_rate=0.05)
        >>> config.decay_rate
        0.05

        >>> create_entity_config(entity_extraction_model="")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: entity_extraction_model cannot be empty

        >>> create_entity_config(decay_rate=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: decay_rate must be between 0.0 and 1.0
    """
    config = EntityConfig(
        entity_extraction_model=entity_extraction_model,
        max_entities=max_entities,
        decay_rate=decay_rate,
    )
    validate_entity_config(config)
    return config


def create_memory_config(
    memory_type: str = "buffer",
    human_prefix: str = "Human",
    ai_prefix: str = "AI",
    input_key: str = "input",
    output_key: str = "output",
) -> MemoryConfig:
    """Create a memory configuration.

    Args:
        memory_type: Type of memory. Defaults to "buffer".
        human_prefix: Prefix for human messages. Defaults to "Human".
        ai_prefix: Prefix for AI messages. Defaults to "AI".
        input_key: Key for input. Defaults to "input".
        output_key: Key for output. Defaults to "output".

    Returns:
        MemoryConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_memory_config(memory_type="window")
        >>> config.memory_type
        <MemoryType.WINDOW: 'window'>

        >>> config = create_memory_config(human_prefix="User")
        >>> config.human_prefix
        'User'

        >>> create_memory_config(memory_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory_type must be one of

        >>> create_memory_config(human_prefix="")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: human_prefix cannot be empty
    """
    if memory_type not in VALID_MEMORY_TYPES:
        msg = f"memory_type must be one of {VALID_MEMORY_TYPES}, got '{memory_type}'"
        raise ValueError(msg)

    config = MemoryConfig(
        memory_type=MemoryType(memory_type),
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
        input_key=input_key,
        output_key=output_key,
    )
    validate_memory_config(config)
    return config


def create_conversation_message(
    role: str,
    content: str,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConversationMessage:
    """Create a conversation message.

    Args:
        role: Role of the message sender.
        content: Message content.
        timestamp: Message timestamp. Defaults to None (uses current time).
        metadata: Additional metadata. Defaults to None.

    Returns:
        ConversationMessage with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> msg = create_conversation_message("human", "Hello!")
        >>> msg.role
        'human'

        >>> from datetime import datetime
        >>> ts = datetime(2024, 1, 1, 12, 0)
        >>> msg = create_conversation_message("ai", "Hi!", timestamp=ts)
        >>> msg.timestamp
        datetime.datetime(2024, 1, 1, 12, 0)

        >>> create_conversation_message("", "Hello!")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: role cannot be empty

        >>> create_conversation_message("human", "")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: content cannot be empty
    """
    if not role:
        msg = "role cannot be empty"
        raise ValueError(msg)

    if not content:
        msg = "content cannot be empty"
        raise ValueError(msg)

    return ConversationMessage(
        role=role,
        content=content,
        timestamp=timestamp if timestamp is not None else datetime.now(),
        metadata=metadata if metadata is not None else {},
    )


def list_memory_types() -> list[str]:
    """List supported memory types.

    Returns:
        Sorted list of memory type names.

    Examples:
        >>> types = list_memory_types()
        >>> "buffer" in types
        True
        >>> "summary" in types
        True
        >>> "window" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_MEMORY_TYPES)


def get_memory_type(name: str) -> MemoryType:
    """Get memory type from name.

    Args:
        name: Memory type name.

    Returns:
        MemoryType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_memory_type("buffer")
        <MemoryType.BUFFER: 'buffer'>

        >>> get_memory_type("window")
        <MemoryType.WINDOW: 'window'>

        >>> get_memory_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory type must be one of
    """
    if name not in VALID_MEMORY_TYPES:
        msg = f"memory type must be one of {VALID_MEMORY_TYPES}, got '{name}'"
        raise ValueError(msg)
    return MemoryType(name)


def estimate_memory_tokens(
    messages: tuple[ConversationMessage, ...],
    chars_per_token: float = 4.0,
) -> int:
    """Estimate token count for a sequence of messages.

    Uses a simple character-to-token ratio for estimation.
    For more accurate counts, use a proper tokenizer.

    Args:
        messages: Tuple of conversation messages.
        chars_per_token: Average characters per token. Defaults to 4.0.

    Returns:
        Estimated total token count.

    Raises:
        ValueError: If chars_per_token is invalid.

    Examples:
        >>> from datetime import datetime
        >>> msgs = (
        ...     ConversationMessage("human", "Hello!", datetime.now(), {}),
        ...     ConversationMessage("ai", "Hi there!", datetime.now(), {}),
        ... )
        >>> tokens = estimate_memory_tokens(msgs)
        >>> tokens > 0
        True

        >>> estimate_memory_tokens(())
        0

        >>> estimate_memory_tokens(msgs, chars_per_token=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chars_per_token must be positive
    """
    if chars_per_token <= 0:
        msg = f"chars_per_token must be positive, got {chars_per_token}"
        raise ValueError(msg)

    if not messages:
        return 0

    total_chars = sum(len(m.content) + len(m.role) for m in messages)
    return int(total_chars / chars_per_token)


def calculate_window_messages(
    total_messages: int,
    window_size: int,
    include_system: bool = True,
    system_messages: int = 0,
) -> int:
    """Calculate number of messages that will be in the window.

    Args:
        total_messages: Total number of messages available.
        window_size: Size of the window.
        include_system: Whether system messages are included. Defaults to True.
        system_messages: Number of system messages. Defaults to 0.

    Returns:
        Number of messages that will be in the window.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_window_messages(50, 10)
        10

        >>> calculate_window_messages(5, 10)
        5

        >>> calculate_window_messages(50, 10, include_system=True, system_messages=2)
        10

        >>> calculate_window_messages(50, 10, include_system=False, system_messages=2)
        10

        >>> calculate_window_messages(50, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be positive

        >>> calculate_window_messages(-1, 10)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_messages must be non-negative
    """
    if total_messages < 0:
        msg = f"total_messages must be non-negative, got {total_messages}"
        raise ValueError(msg)

    if window_size <= 0:
        msg = f"window_size must be positive, got {window_size}"
        raise ValueError(msg)

    if system_messages < 0:
        msg = f"system_messages must be non-negative, got {system_messages}"
        raise ValueError(msg)

    # Determine available messages
    if include_system:
        available = total_messages
    else:
        available = max(0, total_messages - system_messages)

    return min(available, window_size)


def calculate_memory_size_bytes(
    messages: tuple[ConversationMessage, ...],
    include_metadata: bool = True,
) -> int:
    """Calculate approximate memory size in bytes.

    Args:
        messages: Tuple of conversation messages.
        include_metadata: Whether to include metadata size. Defaults to True.

    Returns:
        Approximate memory size in bytes.

    Examples:
        >>> from datetime import datetime
        >>> msgs = (
        ...     ConversationMessage("human", "Hello!", datetime.now(), {}),
        ... )
        >>> size = calculate_memory_size_bytes(msgs)
        >>> size > 0
        True

        >>> calculate_memory_size_bytes(())
        0
    """
    if not messages:
        return 0

    total_size = 0
    for msg in messages:
        # Content and role strings
        total_size += len(msg.content.encode("utf-8"))
        total_size += len(msg.role.encode("utf-8"))

        # Timestamp (datetime object overhead)
        total_size += 48  # Approximate datetime object size

        # Metadata
        if include_metadata and msg.metadata:
            total_size += len(str(msg.metadata).encode("utf-8"))

    return total_size


def create_memory_stats(
    messages: tuple[ConversationMessage, ...],
    chars_per_token: float = 4.0,
) -> MemoryStats:
    """Create memory statistics from messages.

    Args:
        messages: Tuple of conversation messages.
        chars_per_token: Average characters per token. Defaults to 4.0.

    Returns:
        MemoryStats with computed statistics.

    Examples:
        >>> from datetime import datetime
        >>> msgs = (
        ...     ConversationMessage("human", "Hello!", datetime.now(), {}),
        ...     ConversationMessage("ai", "Hi there!", datetime.now(), {}),
        ... )
        >>> stats = create_memory_stats(msgs)
        >>> stats.total_messages
        2
        >>> stats.total_tokens > 0
        True

        >>> stats = create_memory_stats(())
        >>> stats.total_messages
        0
    """
    return MemoryStats(
        total_messages=len(messages),
        total_tokens=estimate_memory_tokens(messages, chars_per_token),
        memory_size_bytes=calculate_memory_size_bytes(messages),
    )


def format_memory_stats(stats: MemoryStats) -> str:
    """Format memory statistics for display.

    Args:
        stats: Memory statistics.

    Returns:
        Formatted string.

    Examples:
        >>> stats = MemoryStats(50, 2048, 8192)
        >>> formatted = format_memory_stats(stats)
        >>> "50 messages" in formatted
        True
        >>> "2048 tokens" in formatted
        True
    """
    size_kb = stats.memory_size_bytes / 1024
    return (
        f"{stats.total_messages} messages | "
        f"{stats.total_tokens} tokens | "
        f"{size_kb:.1f} KB"
    )


def get_recommended_buffer_config(
    use_case: str = "chat",
    model_context_size: int = 4096,
) -> BufferConfig:
    """Get recommended buffer configuration for a use case.

    Args:
        use_case: Use case type ("chat", "agent", "qa"). Defaults to "chat".
        model_context_size: Model's context window size. Defaults to 4096.

    Returns:
        Recommended BufferConfig.

    Examples:
        >>> config = get_recommended_buffer_config()
        >>> config.max_messages > 0
        True

        >>> config = get_recommended_buffer_config(use_case="agent")
        >>> config.max_messages > 0
        True

        >>> config = get_recommended_buffer_config(model_context_size=8192)
        >>> config.max_tokens <= 8192
        True
    """
    use_case_configs = {
        "chat": {"messages_ratio": 0.1, "tokens_ratio": 0.75},
        "agent": {"messages_ratio": 0.05, "tokens_ratio": 0.5},
        "qa": {"messages_ratio": 0.02, "tokens_ratio": 0.25},
    }

    config_params = use_case_configs.get(
        use_case, {"messages_ratio": 0.1, "tokens_ratio": 0.75}
    )

    max_messages = max(10, int(model_context_size * config_params["messages_ratio"]))
    max_tokens = int(model_context_size * config_params["tokens_ratio"])

    return BufferConfig(
        max_messages=max_messages,
        max_tokens=max_tokens,
        return_messages=True,
    )
