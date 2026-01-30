"""Streaming inference utilities for real-time model output.

This module provides utilities for streaming inference with HuggingFace
models, including streaming modes, callbacks, buffering, and statistics.

Examples:
    >>> from hf_gtc.inference.streaming import StreamConfig, StreamingMode
    >>> config = StreamConfig(mode=StreamingMode.TOKEN)
    >>> config.mode
    <StreamingMode.TOKEN: 'token'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import Any


class StreamingMode(Enum):
    """Streaming output granularity modes.

    Attributes:
        TOKEN: Stream individual tokens as they are generated.
        CHUNK: Stream fixed-size chunks of tokens.
        SENTENCE: Stream complete sentences.
        PARAGRAPH: Stream complete paragraphs.

    Examples:
        >>> StreamingMode.TOKEN.value
        'token'
        >>> StreamingMode.CHUNK.value
        'chunk'
        >>> StreamingMode.SENTENCE.value
        'sentence'
        >>> StreamingMode.PARAGRAPH.value
        'paragraph'
    """

    TOKEN = "token"
    CHUNK = "chunk"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


VALID_STREAMING_MODES = frozenset(m.value for m in StreamingMode)


class CallbackType(Enum):
    """Types of streaming callbacks.

    Attributes:
        ON_TOKEN: Called when a new token is generated.
        ON_CHUNK: Called when a chunk is ready to send.
        ON_COMPLETE: Called when generation is complete.
        ON_ERROR: Called when an error occurs.

    Examples:
        >>> CallbackType.ON_TOKEN.value
        'on_token'
        >>> CallbackType.ON_CHUNK.value
        'on_chunk'
        >>> CallbackType.ON_COMPLETE.value
        'on_complete'
        >>> CallbackType.ON_ERROR.value
        'on_error'
    """

    ON_TOKEN = "on_token"
    ON_CHUNK = "on_chunk"
    ON_COMPLETE = "on_complete"
    ON_ERROR = "on_error"


VALID_CALLBACK_TYPES = frozenset(c.value for c in CallbackType)

# Overflow strategies for buffer management
OverflowStrategy = Literal["drop_oldest", "drop_newest", "block"]
VALID_OVERFLOW_STRATEGIES = frozenset({"drop_oldest", "drop_newest", "block"})


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Configuration for streaming inference.

    Attributes:
        mode: Streaming mode determining output granularity. Defaults to TOKEN.
        chunk_size: Number of tokens per chunk (for CHUNK mode). Defaults to 10.
        timeout_seconds: Timeout for streaming operations. Defaults to 30.0.
        buffer_size: Maximum buffer size in tokens. Defaults to 1024.

    Examples:
        >>> config = StreamConfig(mode=StreamingMode.TOKEN, chunk_size=5)
        >>> config.mode
        <StreamingMode.TOKEN: 'token'>
        >>> config.chunk_size
        5

        >>> config = StreamConfig()
        >>> config.timeout_seconds
        30.0
        >>> config.buffer_size
        1024
    """

    mode: StreamingMode = StreamingMode.TOKEN
    chunk_size: int = 10
    timeout_seconds: float = 30.0
    buffer_size: int = 1024


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Configuration for chunking strategy.

    Attributes:
        min_chunk_tokens: Minimum tokens before sending a chunk. Defaults to 1.
        max_chunk_tokens: Maximum tokens per chunk. Defaults to 100.
        delimiter: Token or string delimiter for chunk boundaries. Defaults to None.

    Examples:
        >>> config = ChunkConfig(min_chunk_tokens=5, max_chunk_tokens=50)
        >>> config.min_chunk_tokens
        5
        >>> config.max_chunk_tokens
        50

        >>> config = ChunkConfig(delimiter=".")
        >>> config.delimiter
        '.'
    """

    min_chunk_tokens: int = 1
    max_chunk_tokens: int = 100
    delimiter: str | None = None


@dataclass(frozen=True, slots=True)
class StreamingStats:
    """Statistics from a streaming inference session.

    Attributes:
        tokens_generated: Total number of tokens generated.
        chunks_sent: Total number of chunks sent to client.
        elapsed_seconds: Total elapsed time in seconds.
        tokens_per_second: Average token generation speed.

    Examples:
        >>> stats = StreamingStats(
        ...     tokens_generated=100,
        ...     chunks_sent=10,
        ...     elapsed_seconds=5.0,
        ...     tokens_per_second=20.0,
        ... )
        >>> stats.tokens_generated
        100
        >>> stats.tokens_per_second
        20.0
    """

    tokens_generated: int
    chunks_sent: int
    elapsed_seconds: float
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A streaming event emitted during inference.

    Attributes:
        event_type: Type of the event (token, chunk, complete, error).
        data: Event payload data.
        timestamp: Unix timestamp when the event occurred.
        sequence_number: Sequential event number for ordering.

    Examples:
        >>> event = StreamEvent(
        ...     event_type="token",
        ...     data="Hello",
        ...     timestamp=1704067200.0,
        ...     sequence_number=1,
        ... )
        >>> event.event_type
        'token'
        >>> event.data
        'Hello'
        >>> event.sequence_number
        1
    """

    event_type: str
    data: Any
    timestamp: float
    sequence_number: int


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Configuration for streaming buffer management.

    Attributes:
        max_buffer_tokens: Maximum tokens to buffer. Defaults to 1024.
        flush_interval: Seconds between automatic flushes. Defaults to 0.1.
        overflow_strategy: How to handle buffer overflow. Defaults to "drop_oldest".

    Examples:
        >>> config = BufferConfig(max_buffer_tokens=512, flush_interval=0.05)
        >>> config.max_buffer_tokens
        512
        >>> config.flush_interval
        0.05

        >>> config = BufferConfig(overflow_strategy="block")
        >>> config.overflow_strategy
        'block'
    """

    max_buffer_tokens: int = 1024
    flush_interval: float = 0.1
    overflow_strategy: OverflowStrategy = "drop_oldest"


def validate_stream_config(config: StreamConfig) -> None:
    """Validate streaming configuration.

    Args:
        config: StreamConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If chunk_size is not positive.
        ValueError: If timeout_seconds is not positive.
        ValueError: If buffer_size is not positive.

    Examples:
        >>> config = StreamConfig(mode=StreamingMode.TOKEN, chunk_size=10)
        >>> validate_stream_config(config)  # No error

        >>> validate_stream_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = StreamConfig(chunk_size=0)
        >>> validate_stream_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive

        >>> bad_config = StreamConfig(timeout_seconds=-1.0)
        >>> validate_stream_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: timeout_seconds must be positive

        >>> bad_config = StreamConfig(buffer_size=0)
        >>> validate_stream_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: buffer_size must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.chunk_size <= 0:
        msg = f"chunk_size must be positive, got {config.chunk_size}"
        raise ValueError(msg)

    if config.timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {config.timeout_seconds}"
        raise ValueError(msg)

    if config.buffer_size <= 0:
        msg = f"buffer_size must be positive, got {config.buffer_size}"
        raise ValueError(msg)


def validate_chunk_config(config: ChunkConfig) -> None:
    """Validate chunk configuration.

    Args:
        config: ChunkConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If min_chunk_tokens is not positive.
        ValueError: If max_chunk_tokens is not positive.
        ValueError: If min_chunk_tokens exceeds max_chunk_tokens.

    Examples:
        >>> config = ChunkConfig(min_chunk_tokens=5, max_chunk_tokens=50)
        >>> validate_chunk_config(config)  # No error

        >>> validate_chunk_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = ChunkConfig(min_chunk_tokens=0)
        >>> validate_chunk_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: min_chunk_tokens must be positive

        >>> bad_config = ChunkConfig(max_chunk_tokens=0)
        >>> validate_chunk_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_chunk_tokens must be positive

        >>> bad_config = ChunkConfig(min_chunk_tokens=100, max_chunk_tokens=10)
        >>> validate_chunk_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: min_chunk_tokens cannot exceed max_chunk_tokens
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.min_chunk_tokens <= 0:
        msg = f"min_chunk_tokens must be positive, got {config.min_chunk_tokens}"
        raise ValueError(msg)

    if config.max_chunk_tokens <= 0:
        msg = f"max_chunk_tokens must be positive, got {config.max_chunk_tokens}"
        raise ValueError(msg)

    if config.min_chunk_tokens > config.max_chunk_tokens:
        msg = (
            f"min_chunk_tokens cannot exceed max_chunk_tokens, "
            f"got {config.min_chunk_tokens} > {config.max_chunk_tokens}"
        )
        raise ValueError(msg)


def create_stream_config(
    mode: str = "token",
    chunk_size: int = 10,
    timeout_seconds: float = 30.0,
    buffer_size: int = 1024,
) -> StreamConfig:
    """Create a streaming configuration.

    Args:
        mode: Streaming mode. Defaults to "token".
        chunk_size: Tokens per chunk. Defaults to 10.
        timeout_seconds: Timeout in seconds. Defaults to 30.0.
        buffer_size: Buffer size in tokens. Defaults to 1024.

    Returns:
        StreamConfig with the specified settings.

    Raises:
        ValueError: If mode is invalid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_stream_config(mode="token")
        >>> config.mode
        <StreamingMode.TOKEN: 'token'>

        >>> config = create_stream_config(chunk_size=20, timeout_seconds=60.0)
        >>> config.chunk_size
        20
        >>> config.timeout_seconds
        60.0

        >>> create_stream_config(mode="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: mode must be one of

        >>> create_stream_config(chunk_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if mode not in VALID_STREAMING_MODES:
        msg = f"mode must be one of {VALID_STREAMING_MODES}, got '{mode}'"
        raise ValueError(msg)

    config = StreamConfig(
        mode=StreamingMode(mode),
        chunk_size=chunk_size,
        timeout_seconds=timeout_seconds,
        buffer_size=buffer_size,
    )
    validate_stream_config(config)
    return config


def create_chunk_config(
    min_chunk_tokens: int = 1,
    max_chunk_tokens: int = 100,
    delimiter: str | None = None,
) -> ChunkConfig:
    r"""Create a chunk configuration.

    Args:
        min_chunk_tokens: Minimum tokens per chunk. Defaults to 1.
        max_chunk_tokens: Maximum tokens per chunk. Defaults to 100.
        delimiter: Optional delimiter for chunk boundaries. Defaults to None.

    Returns:
        ChunkConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_chunk_config(min_chunk_tokens=5, max_chunk_tokens=50)
        >>> config.min_chunk_tokens
        5
        >>> config.max_chunk_tokens
        50

        >>> config = create_chunk_config(delimiter="\n")
        >>> config.delimiter
        '\n'

        >>> create_chunk_config(min_chunk_tokens=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: min_chunk_tokens must be positive

        >>> create_chunk_config(min_chunk_tokens=100, max_chunk_tokens=10)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: min_chunk_tokens cannot exceed max_chunk_tokens
    """
    config = ChunkConfig(
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        delimiter=delimiter,
    )
    validate_chunk_config(config)
    return config


def create_buffer_config(
    max_buffer_tokens: int = 1024,
    flush_interval: float = 0.1,
    overflow_strategy: OverflowStrategy = "drop_oldest",
) -> BufferConfig:
    """Create a buffer configuration.

    Args:
        max_buffer_tokens: Maximum buffer size. Defaults to 1024.
        flush_interval: Flush interval in seconds. Defaults to 0.1.
        overflow_strategy: Buffer overflow strategy. Defaults to "drop_oldest".

    Returns:
        BufferConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_buffer_config(max_buffer_tokens=512)
        >>> config.max_buffer_tokens
        512

        >>> config = create_buffer_config(overflow_strategy="block")
        >>> config.overflow_strategy
        'block'

        >>> create_buffer_config(max_buffer_tokens=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_buffer_tokens must be positive

        >>> create_buffer_config(flush_interval=-0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: flush_interval must be positive

        >>> create_buffer_config(overflow_strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overflow_strategy must be one of
    """
    if max_buffer_tokens <= 0:
        msg = f"max_buffer_tokens must be positive, got {max_buffer_tokens}"
        raise ValueError(msg)

    if flush_interval <= 0:
        msg = f"flush_interval must be positive, got {flush_interval}"
        raise ValueError(msg)

    if overflow_strategy not in VALID_OVERFLOW_STRATEGIES:
        msg = (
            f"overflow_strategy must be one of {VALID_OVERFLOW_STRATEGIES}, "
            f"got '{overflow_strategy}'"
        )
        raise ValueError(msg)

    return BufferConfig(
        max_buffer_tokens=max_buffer_tokens,
        flush_interval=flush_interval,
        overflow_strategy=overflow_strategy,
    )


def list_streaming_modes() -> list[str]:
    """List all available streaming modes.

    Returns:
        Sorted list of streaming mode names.

    Examples:
        >>> modes = list_streaming_modes()
        >>> "token" in modes
        True
        >>> "chunk" in modes
        True
        >>> "sentence" in modes
        True
        >>> "paragraph" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_STREAMING_MODES)


def list_callback_types() -> list[str]:
    """List all available callback types.

    Returns:
        Sorted list of callback type names.

    Examples:
        >>> types = list_callback_types()
        >>> "on_token" in types
        True
        >>> "on_chunk" in types
        True
        >>> "on_complete" in types
        True
        >>> "on_error" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CALLBACK_TYPES)


def get_streaming_mode(name: str) -> StreamingMode:
    """Get a streaming mode by name.

    Args:
        name: Name of the streaming mode.

    Returns:
        The corresponding StreamingMode enum value.

    Raises:
        ValueError: If name is not a valid streaming mode.

    Examples:
        >>> get_streaming_mode("token")
        <StreamingMode.TOKEN: 'token'>
        >>> get_streaming_mode("chunk")
        <StreamingMode.CHUNK: 'chunk'>
        >>> get_streaming_mode("sentence")
        <StreamingMode.SENTENCE: 'sentence'>
        >>> get_streaming_mode("paragraph")
        <StreamingMode.PARAGRAPH: 'paragraph'>

        >>> get_streaming_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown streaming mode: 'invalid'

        >>> get_streaming_mode("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown streaming mode: ''
    """
    if name not in VALID_STREAMING_MODES:
        valid = sorted(VALID_STREAMING_MODES)
        msg = f"Unknown streaming mode: '{name}'. Valid modes: {valid}"
        raise ValueError(msg)

    return StreamingMode(name)


def get_callback_type(name: str) -> CallbackType:
    """Get a callback type by name.

    Args:
        name: Name of the callback type.

    Returns:
        The corresponding CallbackType enum value.

    Raises:
        ValueError: If name is not a valid callback type.

    Examples:
        >>> get_callback_type("on_token")
        <CallbackType.ON_TOKEN: 'on_token'>
        >>> get_callback_type("on_chunk")
        <CallbackType.ON_CHUNK: 'on_chunk'>
        >>> get_callback_type("on_complete")
        <CallbackType.ON_COMPLETE: 'on_complete'>
        >>> get_callback_type("on_error")
        <CallbackType.ON_ERROR: 'on_error'>

        >>> get_callback_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown callback type: 'invalid'

        >>> get_callback_type("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown callback type: ''
    """
    if name not in VALID_CALLBACK_TYPES:
        valid = sorted(VALID_CALLBACK_TYPES)
        msg = f"Unknown callback type: '{name}'. Valid types: {valid}"
        raise ValueError(msg)

    return CallbackType(name)


def calculate_tokens_per_second(
    tokens_generated: int,
    elapsed_seconds: float,
) -> float:
    """Calculate token generation speed.

    Args:
        tokens_generated: Number of tokens generated.
        elapsed_seconds: Time elapsed in seconds.

    Returns:
        Tokens per second rate.

    Raises:
        ValueError: If tokens_generated is negative.
        ValueError: If elapsed_seconds is negative.

    Examples:
        >>> calculate_tokens_per_second(100, 5.0)
        20.0
        >>> calculate_tokens_per_second(0, 1.0)
        0.0
        >>> calculate_tokens_per_second(50, 0.0)
        0.0

        >>> calculate_tokens_per_second(-1, 5.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens_generated cannot be negative

        >>> calculate_tokens_per_second(100, -1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: elapsed_seconds cannot be negative
    """
    if tokens_generated < 0:
        msg = f"tokens_generated cannot be negative, got {tokens_generated}"
        raise ValueError(msg)

    if elapsed_seconds < 0:
        msg = f"elapsed_seconds cannot be negative, got {elapsed_seconds}"
        raise ValueError(msg)

    if elapsed_seconds == 0.0:
        return 0.0

    return tokens_generated / elapsed_seconds


def estimate_stream_duration(
    expected_tokens: int,
    tokens_per_second: float,
) -> float:
    """Estimate streaming duration based on expected tokens and generation speed.

    Args:
        expected_tokens: Number of tokens expected to generate.
        tokens_per_second: Expected token generation rate.

    Returns:
        Estimated duration in seconds.

    Raises:
        ValueError: If expected_tokens is negative.
        ValueError: If tokens_per_second is negative.

    Examples:
        >>> estimate_stream_duration(100, 20.0)
        5.0
        >>> estimate_stream_duration(0, 20.0)
        0.0
        >>> estimate_stream_duration(100, 0.0)
        0.0

        >>> estimate_stream_duration(-1, 20.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: expected_tokens cannot be negative

        >>> estimate_stream_duration(100, -1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens_per_second cannot be negative
    """
    if expected_tokens < 0:
        msg = f"expected_tokens cannot be negative, got {expected_tokens}"
        raise ValueError(msg)

    if tokens_per_second < 0:
        msg = f"tokens_per_second cannot be negative, got {tokens_per_second}"
        raise ValueError(msg)

    if tokens_per_second == 0.0:
        return 0.0

    return expected_tokens / tokens_per_second
