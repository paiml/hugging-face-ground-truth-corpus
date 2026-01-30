"""Tests for inference.streaming module."""

from __future__ import annotations

import pytest

from hf_gtc.inference.streaming import (
    VALID_CALLBACK_TYPES,
    VALID_OVERFLOW_STRATEGIES,
    VALID_STREAMING_MODES,
    BufferConfig,
    CallbackType,
    ChunkConfig,
    StreamConfig,
    StreamEvent,
    StreamingMode,
    StreamingStats,
    calculate_tokens_per_second,
    create_buffer_config,
    create_chunk_config,
    create_stream_config,
    estimate_stream_duration,
    get_callback_type,
    get_streaming_mode,
    list_callback_types,
    list_streaming_modes,
    validate_chunk_config,
    validate_stream_config,
)


class TestStreamingMode:
    """Tests for StreamingMode enum."""

    def test_all_modes_have_values(self) -> None:
        """All modes have string values."""
        for mode in StreamingMode:
            assert isinstance(mode.value, str)

    def test_token_value(self) -> None:
        """TOKEN has correct value."""
        assert StreamingMode.TOKEN.value == "token"

    def test_chunk_value(self) -> None:
        """CHUNK has correct value."""
        assert StreamingMode.CHUNK.value == "chunk"

    def test_sentence_value(self) -> None:
        """SENTENCE has correct value."""
        assert StreamingMode.SENTENCE.value == "sentence"

    def test_paragraph_value(self) -> None:
        """PARAGRAPH has correct value."""
        assert StreamingMode.PARAGRAPH.value == "paragraph"

    def test_valid_modes_frozenset(self) -> None:
        """VALID_STREAMING_MODES is a frozenset."""
        assert isinstance(VALID_STREAMING_MODES, frozenset)

    def test_valid_modes_contains_all(self) -> None:
        """VALID_STREAMING_MODES contains all enum values."""
        for mode in StreamingMode:
            assert mode.value in VALID_STREAMING_MODES


class TestCallbackType:
    """Tests for CallbackType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for callback_type in CallbackType:
            assert isinstance(callback_type.value, str)

    def test_on_token_value(self) -> None:
        """ON_TOKEN has correct value."""
        assert CallbackType.ON_TOKEN.value == "on_token"

    def test_on_chunk_value(self) -> None:
        """ON_CHUNK has correct value."""
        assert CallbackType.ON_CHUNK.value == "on_chunk"

    def test_on_complete_value(self) -> None:
        """ON_COMPLETE has correct value."""
        assert CallbackType.ON_COMPLETE.value == "on_complete"

    def test_on_error_value(self) -> None:
        """ON_ERROR has correct value."""
        assert CallbackType.ON_ERROR.value == "on_error"

    def test_valid_types_frozenset(self) -> None:
        """VALID_CALLBACK_TYPES is a frozenset."""
        assert isinstance(VALID_CALLBACK_TYPES, frozenset)

    def test_valid_types_contains_all(self) -> None:
        """VALID_CALLBACK_TYPES contains all enum values."""
        for callback_type in CallbackType:
            assert callback_type.value in VALID_CALLBACK_TYPES


class TestValidOverflowStrategies:
    """Tests for VALID_OVERFLOW_STRATEGIES."""

    def test_is_frozenset(self) -> None:
        """VALID_OVERFLOW_STRATEGIES is a frozenset."""
        assert isinstance(VALID_OVERFLOW_STRATEGIES, frozenset)

    def test_contains_drop_oldest(self) -> None:
        """Contains drop_oldest strategy."""
        assert "drop_oldest" in VALID_OVERFLOW_STRATEGIES

    def test_contains_drop_newest(self) -> None:
        """Contains drop_newest strategy."""
        assert "drop_newest" in VALID_OVERFLOW_STRATEGIES

    def test_contains_block(self) -> None:
        """Contains block strategy."""
        assert "block" in VALID_OVERFLOW_STRATEGIES


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_create_config(self) -> None:
        """Create stream config."""
        config = StreamConfig(
            mode=StreamingMode.TOKEN,
            chunk_size=10,
            timeout_seconds=30.0,
            buffer_size=1024,
        )
        assert config.mode == StreamingMode.TOKEN
        assert config.chunk_size == 10
        assert config.timeout_seconds == 30.0
        assert config.buffer_size == 1024

    def test_default_values(self) -> None:
        """Default values are correct."""
        config = StreamConfig()
        assert config.mode == StreamingMode.TOKEN
        assert config.chunk_size == 10
        assert config.timeout_seconds == 30.0
        assert config.buffer_size == 1024

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = StreamConfig()
        with pytest.raises(AttributeError):
            config.chunk_size = 20  # type: ignore[misc]

    def test_config_has_slots(self) -> None:
        """Config uses slots for memory efficiency."""
        config = StreamConfig()
        assert hasattr(config, "__slots__") or not hasattr(config, "__dict__")


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_create_config(self) -> None:
        """Create chunk config."""
        config = ChunkConfig(
            min_chunk_tokens=5,
            max_chunk_tokens=50,
            delimiter=".",
        )
        assert config.min_chunk_tokens == 5
        assert config.max_chunk_tokens == 50
        assert config.delimiter == "."

    def test_default_values(self) -> None:
        """Default values are correct."""
        config = ChunkConfig()
        assert config.min_chunk_tokens == 1
        assert config.max_chunk_tokens == 100
        assert config.delimiter is None

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ChunkConfig()
        with pytest.raises(AttributeError):
            config.min_chunk_tokens = 10  # type: ignore[misc]


class TestStreamingStats:
    """Tests for StreamingStats dataclass."""

    def test_create_stats(self) -> None:
        """Create streaming stats."""
        stats = StreamingStats(
            tokens_generated=100,
            chunks_sent=10,
            elapsed_seconds=5.0,
            tokens_per_second=20.0,
        )
        assert stats.tokens_generated == 100
        assert stats.chunks_sent == 10
        assert stats.elapsed_seconds == 5.0
        assert stats.tokens_per_second == 20.0

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = StreamingStats(100, 10, 5.0, 20.0)
        with pytest.raises(AttributeError):
            stats.tokens_generated = 200  # type: ignore[misc]


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_create_event(self) -> None:
        """Create stream event."""
        event = StreamEvent(
            event_type="token",
            data="Hello",
            timestamp=1704067200.0,
            sequence_number=1,
        )
        assert event.event_type == "token"
        assert event.data == "Hello"
        assert event.timestamp == 1704067200.0
        assert event.sequence_number == 1

    def test_event_with_dict_data(self) -> None:
        """Event can contain dict data."""
        data = {"tokens": ["Hello", "world"]}
        event = StreamEvent("chunk", data, 1704067200.0, 1)
        assert event.data == data

    def test_event_is_frozen(self) -> None:
        """Event is immutable."""
        event = StreamEvent("token", "Hello", 1704067200.0, 1)
        with pytest.raises(AttributeError):
            event.data = "World"  # type: ignore[misc]


class TestBufferConfig:
    """Tests for BufferConfig dataclass."""

    def test_create_config(self) -> None:
        """Create buffer config."""
        config = BufferConfig(
            max_buffer_tokens=512,
            flush_interval=0.05,
            overflow_strategy="block",
        )
        assert config.max_buffer_tokens == 512
        assert config.flush_interval == 0.05
        assert config.overflow_strategy == "block"

    def test_default_values(self) -> None:
        """Default values are correct."""
        config = BufferConfig()
        assert config.max_buffer_tokens == 1024
        assert config.flush_interval == 0.1
        assert config.overflow_strategy == "drop_oldest"

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = BufferConfig()
        with pytest.raises(AttributeError):
            config.max_buffer_tokens = 2048  # type: ignore[misc]


class TestValidateStreamConfig:
    """Tests for validate_stream_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = StreamConfig(mode=StreamingMode.TOKEN, chunk_size=10)
        validate_stream_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_stream_config(None)  # type: ignore[arg-type]

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk size raises ValueError."""
        config = StreamConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_stream_config(config)

    def test_negative_chunk_size_raises(self) -> None:
        """Negative chunk size raises ValueError."""
        config = StreamConfig(chunk_size=-1)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_stream_config(config)

    def test_zero_timeout_raises(self) -> None:
        """Zero timeout raises ValueError."""
        config = StreamConfig(timeout_seconds=0.0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_stream_config(config)

    def test_negative_timeout_raises(self) -> None:
        """Negative timeout raises ValueError."""
        config = StreamConfig(timeout_seconds=-1.0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_stream_config(config)

    def test_zero_buffer_size_raises(self) -> None:
        """Zero buffer size raises ValueError."""
        config = StreamConfig(buffer_size=0)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            validate_stream_config(config)

    def test_negative_buffer_size_raises(self) -> None:
        """Negative buffer size raises ValueError."""
        config = StreamConfig(buffer_size=-1)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            validate_stream_config(config)


class TestValidateChunkConfig:
    """Tests for validate_chunk_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ChunkConfig(min_chunk_tokens=5, max_chunk_tokens=50)
        validate_chunk_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_chunk_config(None)  # type: ignore[arg-type]

    def test_zero_min_chunk_tokens_raises(self) -> None:
        """Zero min chunk tokens raises ValueError."""
        config = ChunkConfig(min_chunk_tokens=0)
        with pytest.raises(ValueError, match="min_chunk_tokens must be positive"):
            validate_chunk_config(config)

    def test_negative_min_chunk_tokens_raises(self) -> None:
        """Negative min chunk tokens raises ValueError."""
        config = ChunkConfig(min_chunk_tokens=-1)
        with pytest.raises(ValueError, match="min_chunk_tokens must be positive"):
            validate_chunk_config(config)

    def test_zero_max_chunk_tokens_raises(self) -> None:
        """Zero max chunk tokens raises ValueError."""
        config = ChunkConfig(max_chunk_tokens=0)
        with pytest.raises(ValueError, match="max_chunk_tokens must be positive"):
            validate_chunk_config(config)

    def test_negative_max_chunk_tokens_raises(self) -> None:
        """Negative max chunk tokens raises ValueError."""
        config = ChunkConfig(max_chunk_tokens=-1)
        with pytest.raises(ValueError, match="max_chunk_tokens must be positive"):
            validate_chunk_config(config)

    def test_min_exceeds_max_raises(self) -> None:
        """Min exceeding max raises ValueError."""
        config = ChunkConfig(min_chunk_tokens=100, max_chunk_tokens=10)
        with pytest.raises(
            ValueError, match="min_chunk_tokens cannot exceed max_chunk_tokens"
        ):
            validate_chunk_config(config)

    def test_min_equals_max_valid(self) -> None:
        """Min equals max is valid."""
        config = ChunkConfig(min_chunk_tokens=50, max_chunk_tokens=50)
        validate_chunk_config(config)


class TestCreateStreamConfig:
    """Tests for create_stream_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_stream_config()
        assert config.mode == StreamingMode.TOKEN
        assert config.chunk_size == 10
        assert config.timeout_seconds == 30.0
        assert config.buffer_size == 1024

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("token", StreamingMode.TOKEN),
            ("chunk", StreamingMode.CHUNK),
            ("sentence", StreamingMode.SENTENCE),
            ("paragraph", StreamingMode.PARAGRAPH),
        ],
    )
    def test_all_modes(self, mode: str, expected: StreamingMode) -> None:
        """Test all streaming modes."""
        config = create_stream_config(mode=mode)
        assert config.mode == expected

    def test_custom_chunk_size(self) -> None:
        """Create config with custom chunk size."""
        config = create_stream_config(chunk_size=20)
        assert config.chunk_size == 20

    def test_custom_timeout(self) -> None:
        """Create config with custom timeout."""
        config = create_stream_config(timeout_seconds=60.0)
        assert config.timeout_seconds == 60.0

    def test_custom_buffer_size(self) -> None:
        """Create config with custom buffer size."""
        config = create_stream_config(buffer_size=2048)
        assert config.buffer_size == 2048

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            create_stream_config(mode="invalid")

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_stream_config(chunk_size=0)

    def test_negative_timeout_raises(self) -> None:
        """Negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            create_stream_config(timeout_seconds=-1.0)

    def test_zero_buffer_size_raises(self) -> None:
        """Zero buffer size raises ValueError."""
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            create_stream_config(buffer_size=0)


class TestCreateChunkConfig:
    """Tests for create_chunk_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_chunk_config()
        assert config.min_chunk_tokens == 1
        assert config.max_chunk_tokens == 100
        assert config.delimiter is None

    def test_custom_min_tokens(self) -> None:
        """Create config with custom min tokens."""
        config = create_chunk_config(min_chunk_tokens=5)
        assert config.min_chunk_tokens == 5

    def test_custom_max_tokens(self) -> None:
        """Create config with custom max tokens."""
        config = create_chunk_config(max_chunk_tokens=50)
        assert config.max_chunk_tokens == 50

    def test_custom_delimiter(self) -> None:
        """Create config with custom delimiter."""
        config = create_chunk_config(delimiter="\n")
        assert config.delimiter == "\n"

    def test_delimiter_dot(self) -> None:
        """Create config with dot delimiter."""
        config = create_chunk_config(delimiter=".")
        assert config.delimiter == "."

    def test_zero_min_tokens_raises(self) -> None:
        """Zero min tokens raises ValueError."""
        with pytest.raises(ValueError, match="min_chunk_tokens must be positive"):
            create_chunk_config(min_chunk_tokens=0)

    def test_zero_max_tokens_raises(self) -> None:
        """Zero max tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_chunk_tokens must be positive"):
            create_chunk_config(max_chunk_tokens=0)

    def test_min_exceeds_max_raises(self) -> None:
        """Min exceeding max raises ValueError."""
        with pytest.raises(
            ValueError, match="min_chunk_tokens cannot exceed max_chunk_tokens"
        ):
            create_chunk_config(min_chunk_tokens=100, max_chunk_tokens=10)


class TestCreateBufferConfig:
    """Tests for create_buffer_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_buffer_config()
        assert config.max_buffer_tokens == 1024
        assert config.flush_interval == 0.1
        assert config.overflow_strategy == "drop_oldest"

    def test_custom_max_tokens(self) -> None:
        """Create config with custom max tokens."""
        config = create_buffer_config(max_buffer_tokens=512)
        assert config.max_buffer_tokens == 512

    def test_custom_flush_interval(self) -> None:
        """Create config with custom flush interval."""
        config = create_buffer_config(flush_interval=0.05)
        assert config.flush_interval == 0.05

    @pytest.mark.parametrize(
        "strategy",
        ["drop_oldest", "drop_newest", "block"],
    )
    def test_all_overflow_strategies(self, strategy: str) -> None:
        """Test all overflow strategies."""
        config = create_buffer_config(overflow_strategy=strategy)  # type: ignore[arg-type]
        assert config.overflow_strategy == strategy

    def test_zero_max_tokens_raises(self) -> None:
        """Zero max tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_buffer_tokens must be positive"):
            create_buffer_config(max_buffer_tokens=0)

    def test_negative_max_tokens_raises(self) -> None:
        """Negative max tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_buffer_tokens must be positive"):
            create_buffer_config(max_buffer_tokens=-1)

    def test_zero_flush_interval_raises(self) -> None:
        """Zero flush interval raises ValueError."""
        with pytest.raises(ValueError, match="flush_interval must be positive"):
            create_buffer_config(flush_interval=0.0)

    def test_negative_flush_interval_raises(self) -> None:
        """Negative flush interval raises ValueError."""
        with pytest.raises(ValueError, match="flush_interval must be positive"):
            create_buffer_config(flush_interval=-0.1)

    def test_invalid_strategy_raises(self) -> None:
        """Invalid overflow strategy raises ValueError."""
        with pytest.raises(ValueError, match="overflow_strategy must be one of"):
            create_buffer_config(overflow_strategy="invalid")  # type: ignore[arg-type]


class TestListStreamingModes:
    """Tests for list_streaming_modes function."""

    def test_returns_list(self) -> None:
        """Returns a list."""
        modes = list_streaming_modes()
        assert isinstance(modes, list)

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        modes = list_streaming_modes()
        assert modes == sorted(modes)

    def test_contains_token(self) -> None:
        """Contains token mode."""
        modes = list_streaming_modes()
        assert "token" in modes

    def test_contains_chunk(self) -> None:
        """Contains chunk mode."""
        modes = list_streaming_modes()
        assert "chunk" in modes

    def test_contains_sentence(self) -> None:
        """Contains sentence mode."""
        modes = list_streaming_modes()
        assert "sentence" in modes

    def test_contains_paragraph(self) -> None:
        """Contains paragraph mode."""
        modes = list_streaming_modes()
        assert "paragraph" in modes

    def test_contains_all_valid_modes(self) -> None:
        """Contains all valid streaming modes."""
        modes = list_streaming_modes()
        for mode in VALID_STREAMING_MODES:
            assert mode in modes


class TestListCallbackTypes:
    """Tests for list_callback_types function."""

    def test_returns_list(self) -> None:
        """Returns a list."""
        types = list_callback_types()
        assert isinstance(types, list)

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_callback_types()
        assert types == sorted(types)

    def test_contains_on_token(self) -> None:
        """Contains on_token type."""
        types = list_callback_types()
        assert "on_token" in types

    def test_contains_on_chunk(self) -> None:
        """Contains on_chunk type."""
        types = list_callback_types()
        assert "on_chunk" in types

    def test_contains_on_complete(self) -> None:
        """Contains on_complete type."""
        types = list_callback_types()
        assert "on_complete" in types

    def test_contains_on_error(self) -> None:
        """Contains on_error type."""
        types = list_callback_types()
        assert "on_error" in types

    def test_contains_all_valid_types(self) -> None:
        """Contains all valid callback types."""
        types = list_callback_types()
        for callback_type in VALID_CALLBACK_TYPES:
            assert callback_type in types


class TestGetStreamingMode:
    """Tests for get_streaming_mode function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("token", StreamingMode.TOKEN),
            ("chunk", StreamingMode.CHUNK),
            ("sentence", StreamingMode.SENTENCE),
            ("paragraph", StreamingMode.PARAGRAPH),
        ],
    )
    def test_all_modes(self, name: str, expected: StreamingMode) -> None:
        """Test all streaming modes."""
        result = get_streaming_mode(name)
        assert result == expected

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown streaming mode: 'invalid'"):
            get_streaming_mode("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown streaming mode: ''"):
            get_streaming_mode("")

    def test_case_sensitive(self) -> None:
        """Mode name is case sensitive."""
        with pytest.raises(ValueError, match="Unknown streaming mode: 'TOKEN'"):
            get_streaming_mode("TOKEN")


class TestGetCallbackType:
    """Tests for get_callback_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("on_token", CallbackType.ON_TOKEN),
            ("on_chunk", CallbackType.ON_CHUNK),
            ("on_complete", CallbackType.ON_COMPLETE),
            ("on_error", CallbackType.ON_ERROR),
        ],
    )
    def test_all_types(self, name: str, expected: CallbackType) -> None:
        """Test all callback types."""
        result = get_callback_type(name)
        assert result == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown callback type: 'invalid'"):
            get_callback_type("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown callback type: ''"):
            get_callback_type("")

    def test_case_sensitive(self) -> None:
        """Type name is case sensitive."""
        with pytest.raises(ValueError, match="Unknown callback type: 'ON_TOKEN'"):
            get_callback_type("ON_TOKEN")


class TestCalculateTokensPerSecond:
    """Tests for calculate_tokens_per_second function."""

    def test_basic_calculation(self) -> None:
        """Basic calculation."""
        result = calculate_tokens_per_second(100, 5.0)
        assert result == 20.0

    def test_zero_tokens(self) -> None:
        """Zero tokens returns zero."""
        result = calculate_tokens_per_second(0, 1.0)
        assert result == 0.0

    def test_zero_elapsed_time(self) -> None:
        """Zero elapsed time returns zero."""
        result = calculate_tokens_per_second(50, 0.0)
        assert result == 0.0

    def test_fractional_result(self) -> None:
        """Fractional result."""
        result = calculate_tokens_per_second(1, 3.0)
        assert abs(result - 1 / 3) < 1e-10

    def test_large_values(self) -> None:
        """Large values."""
        result = calculate_tokens_per_second(1000000, 100.0)
        assert result == 10000.0

    def test_small_elapsed_time(self) -> None:
        """Small elapsed time."""
        result = calculate_tokens_per_second(10, 0.001)
        assert result == 10000.0

    def test_negative_tokens_raises(self) -> None:
        """Negative tokens raises ValueError."""
        with pytest.raises(ValueError, match="tokens_generated cannot be negative"):
            calculate_tokens_per_second(-1, 5.0)

    def test_negative_elapsed_raises(self) -> None:
        """Negative elapsed time raises ValueError."""
        with pytest.raises(ValueError, match="elapsed_seconds cannot be negative"):
            calculate_tokens_per_second(100, -1.0)


class TestEstimateStreamDuration:
    """Tests for estimate_stream_duration function."""

    def test_basic_estimate(self) -> None:
        """Basic estimate."""
        result = estimate_stream_duration(100, 20.0)
        assert result == 5.0

    def test_zero_tokens(self) -> None:
        """Zero tokens returns zero."""
        result = estimate_stream_duration(0, 20.0)
        assert result == 0.0

    def test_zero_rate(self) -> None:
        """Zero rate returns zero."""
        result = estimate_stream_duration(100, 0.0)
        assert result == 0.0

    def test_fractional_result(self) -> None:
        """Fractional result."""
        result = estimate_stream_duration(10, 3.0)
        assert abs(result - 10 / 3) < 1e-10

    def test_large_values(self) -> None:
        """Large values."""
        result = estimate_stream_duration(10000, 100.0)
        assert result == 100.0

    def test_small_rate(self) -> None:
        """Small rate."""
        result = estimate_stream_duration(1, 0.001)
        assert result == 1000.0

    def test_negative_tokens_raises(self) -> None:
        """Negative tokens raises ValueError."""
        with pytest.raises(ValueError, match="expected_tokens cannot be negative"):
            estimate_stream_duration(-1, 20.0)

    def test_negative_rate_raises(self) -> None:
        """Negative rate raises ValueError."""
        with pytest.raises(ValueError, match="tokens_per_second cannot be negative"):
            estimate_stream_duration(100, -1.0)


class TestIntegration:
    """Integration tests for streaming module."""

    def test_create_stream_config_and_validate(self) -> None:
        """Create and validate stream config."""
        config = create_stream_config(mode="chunk", chunk_size=20)
        validate_stream_config(config)
        assert config.mode == StreamingMode.CHUNK
        assert config.chunk_size == 20

    def test_create_chunk_config_and_validate(self) -> None:
        """Create and validate chunk config."""
        config = create_chunk_config(min_chunk_tokens=5, max_chunk_tokens=50)
        validate_chunk_config(config)
        assert config.min_chunk_tokens == 5
        assert config.max_chunk_tokens == 50

    def test_streaming_stats_calculation(self) -> None:
        """Calculate and verify streaming stats."""
        tokens = 100
        elapsed = 5.0
        rate = calculate_tokens_per_second(tokens, elapsed)
        stats = StreamingStats(
            tokens_generated=tokens,
            chunks_sent=10,
            elapsed_seconds=elapsed,
            tokens_per_second=rate,
        )
        assert stats.tokens_per_second == 20.0

    def test_duration_estimation(self) -> None:
        """Estimate duration based on rate."""
        rate = calculate_tokens_per_second(100, 5.0)
        duration = estimate_stream_duration(200, rate)
        assert duration == 10.0

    def test_stream_event_sequence(self) -> None:
        """Create sequence of stream events."""
        events = [
            StreamEvent("token", "Hello", 1704067200.0, 0),
            StreamEvent("token", " ", 1704067200.1, 1),
            StreamEvent("token", "world", 1704067200.2, 2),
            StreamEvent("complete", None, 1704067200.3, 3),
        ]
        assert len(events) == 4
        assert events[0].sequence_number == 0
        assert events[-1].event_type == "complete"
        # Verify ordering
        for i in range(len(events) - 1):
            assert events[i].sequence_number < events[i + 1].sequence_number
            assert events[i].timestamp <= events[i + 1].timestamp
