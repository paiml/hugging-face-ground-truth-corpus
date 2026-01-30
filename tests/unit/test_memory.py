"""Tests for agents.memory module."""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.agents.memory import (
    VALID_MEMORY_TYPES,
    BufferConfig,
    ConversationMessage,
    EntityConfig,
    MemoryConfig,
    MemoryStats,
    MemoryType,
    SummaryConfig,
    WindowConfig,
    calculate_memory_size_bytes,
    calculate_window_messages,
    create_buffer_config,
    create_conversation_message,
    create_entity_config,
    create_memory_config,
    create_memory_stats,
    create_summary_config,
    create_window_config,
    estimate_memory_tokens,
    format_memory_stats,
    get_memory_type,
    get_recommended_buffer_config,
    list_memory_types,
    validate_buffer_config,
    validate_entity_config,
    validate_memory_config,
    validate_summary_config,
    validate_window_config,
)


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_buffer_value(self) -> None:
        """Test BUFFER value."""
        assert MemoryType.BUFFER.value == "buffer"

    def test_summary_value(self) -> None:
        """Test SUMMARY value."""
        assert MemoryType.SUMMARY.value == "summary"

    def test_window_value(self) -> None:
        """Test WINDOW value."""
        assert MemoryType.WINDOW.value == "window"

    def test_entity_value(self) -> None:
        """Test ENTITY value."""
        assert MemoryType.ENTITY.value == "entity"

    def test_conversation_value(self) -> None:
        """Test CONVERSATION value."""
        assert MemoryType.CONVERSATION.value == "conversation"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [m.value for m in MemoryType]
        assert len(values) == len(set(values))

    def test_all_types_have_string_values(self) -> None:
        """All types have string values."""
        for memory_type in MemoryType:
            assert isinstance(memory_type.value, str)


class TestValidMemoryTypes:
    """Tests for VALID_MEMORY_TYPES frozenset."""

    def test_is_frozenset(self) -> None:
        """Test VALID_MEMORY_TYPES is a frozenset."""
        assert isinstance(VALID_MEMORY_TYPES, frozenset)

    def test_contains_all_enums(self) -> None:
        """Test VALID_MEMORY_TYPES contains all enum values."""
        for memory_type in MemoryType:
            assert memory_type.value in VALID_MEMORY_TYPES

    def test_is_immutable(self) -> None:
        """Test that frozenset is immutable."""
        with pytest.raises(AttributeError):
            VALID_MEMORY_TYPES.add("new")  # type: ignore[attr-defined]


class TestBufferConfig:
    """Tests for BufferConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating BufferConfig instance."""
        config = BufferConfig(
            max_messages=100,
            max_tokens=4096,
            return_messages=True,
        )
        assert config.max_messages == 100
        assert config.max_tokens == 4096
        assert config.return_messages is True

    def test_frozen(self) -> None:
        """Test that BufferConfig is immutable."""
        config = BufferConfig(100, 4096, True)
        with pytest.raises(AttributeError):
            config.max_messages = 200  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that BufferConfig uses slots."""
        config = BufferConfig(100, 4096, True)
        assert not hasattr(config, "__dict__")


class TestWindowConfig:
    """Tests for WindowConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating WindowConfig instance."""
        config = WindowConfig(
            window_size=10,
            overlap=2,
            include_system=True,
        )
        assert config.window_size == 10
        assert config.overlap == 2
        assert config.include_system is True

    def test_frozen(self) -> None:
        """Test that WindowConfig is immutable."""
        config = WindowConfig(10, 2, True)
        with pytest.raises(AttributeError):
            config.window_size = 20  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that WindowConfig uses slots."""
        config = WindowConfig(10, 2, True)
        assert not hasattr(config, "__dict__")


class TestSummaryConfig:
    """Tests for SummaryConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SummaryConfig instance."""
        config = SummaryConfig(
            max_summary_tokens=500,
            summarizer_model="gpt-3.5-turbo",
            update_frequency=5,
        )
        assert config.max_summary_tokens == 500
        assert config.summarizer_model == "gpt-3.5-turbo"
        assert config.update_frequency == 5

    def test_frozen(self) -> None:
        """Test that SummaryConfig is immutable."""
        config = SummaryConfig(500, "gpt-3.5-turbo", 5)
        with pytest.raises(AttributeError):
            config.max_summary_tokens = 1000  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that SummaryConfig uses slots."""
        config = SummaryConfig(500, "gpt-3.5-turbo", 5)
        assert not hasattr(config, "__dict__")


class TestEntityConfig:
    """Tests for EntityConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating EntityConfig instance."""
        config = EntityConfig(
            entity_extraction_model="gpt-3.5-turbo",
            max_entities=50,
            decay_rate=0.1,
        )
        assert config.entity_extraction_model == "gpt-3.5-turbo"
        assert config.max_entities == 50
        assert config.decay_rate == 0.1

    def test_frozen(self) -> None:
        """Test that EntityConfig is immutable."""
        config = EntityConfig("gpt-3.5-turbo", 50, 0.1)
        with pytest.raises(AttributeError):
            config.max_entities = 100  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that EntityConfig uses slots."""
        config = EntityConfig("gpt-3.5-turbo", 50, 0.1)
        assert not hasattr(config, "__dict__")


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryConfig instance."""
        config = MemoryConfig(
            memory_type=MemoryType.BUFFER,
            human_prefix="Human",
            ai_prefix="AI",
            input_key="input",
            output_key="output",
        )
        assert config.memory_type == MemoryType.BUFFER
        assert config.human_prefix == "Human"
        assert config.ai_prefix == "AI"
        assert config.input_key == "input"
        assert config.output_key == "output"

    def test_frozen(self) -> None:
        """Test that MemoryConfig is immutable."""
        config = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "input", "output")
        with pytest.raises(AttributeError):
            config.human_prefix = "User"  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that MemoryConfig uses slots."""
        config = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "input", "output")
        assert not hasattr(config, "__dict__")


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_creation(self) -> None:
        """Test creating ConversationMessage instance."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        msg = ConversationMessage(
            role="human",
            content="Hello!",
            timestamp=ts,
            metadata={"key": "value"},
        )
        assert msg.role == "human"
        assert msg.content == "Hello!"
        assert msg.timestamp == ts
        assert msg.metadata == {"key": "value"}

    def test_frozen(self) -> None:
        """Test that ConversationMessage is immutable."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        with pytest.raises(AttributeError):
            msg.content = "Hi!"  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that ConversationMessage uses slots."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        assert not hasattr(msg, "__dict__")


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryStats instance."""
        stats = MemoryStats(
            total_messages=50,
            total_tokens=2048,
            memory_size_bytes=8192,
        )
        assert stats.total_messages == 50
        assert stats.total_tokens == 2048
        assert stats.memory_size_bytes == 8192

    def test_frozen(self) -> None:
        """Test that MemoryStats is immutable."""
        stats = MemoryStats(50, 2048, 8192)
        with pytest.raises(AttributeError):
            stats.total_messages = 100  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that MemoryStats uses slots."""
        stats = MemoryStats(50, 2048, 8192)
        assert not hasattr(stats, "__dict__")


class TestValidateBufferConfig:
    """Tests for validate_buffer_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = BufferConfig(100, 4096, True)
        validate_buffer_config(config)

    def test_zero_max_messages_raises(self) -> None:
        """Zero max_messages raises ValueError."""
        config = BufferConfig(0, 4096, True)
        with pytest.raises(ValueError, match="max_messages must be positive"):
            validate_buffer_config(config)

    def test_negative_max_messages_raises(self) -> None:
        """Negative max_messages raises ValueError."""
        config = BufferConfig(-1, 4096, True)
        with pytest.raises(ValueError, match="max_messages must be positive"):
            validate_buffer_config(config)

    def test_negative_max_tokens_raises(self) -> None:
        """Negative max_tokens raises ValueError."""
        config = BufferConfig(100, -1, True)
        with pytest.raises(ValueError, match="max_tokens must be non-negative"):
            validate_buffer_config(config)

    def test_zero_max_tokens_valid(self) -> None:
        """Zero max_tokens is valid."""
        config = BufferConfig(100, 0, True)
        validate_buffer_config(config)


class TestValidateWindowConfig:
    """Tests for validate_window_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = WindowConfig(10, 2, True)
        validate_window_config(config)

    def test_zero_window_size_raises(self) -> None:
        """Zero window_size raises ValueError."""
        config = WindowConfig(0, 2, True)
        with pytest.raises(ValueError, match="window_size must be positive"):
            validate_window_config(config)

    def test_negative_window_size_raises(self) -> None:
        """Negative window_size raises ValueError."""
        config = WindowConfig(-1, 2, True)
        with pytest.raises(ValueError, match="window_size must be positive"):
            validate_window_config(config)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        config = WindowConfig(10, -1, True)
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            validate_window_config(config)

    def test_overlap_equals_window_size_raises(self) -> None:
        """Overlap equals window_size raises ValueError."""
        config = WindowConfig(10, 10, True)
        msg = r"overlap .* must be less than window_size"
        with pytest.raises(ValueError, match=msg):
            validate_window_config(config)

    def test_overlap_exceeds_window_size_raises(self) -> None:
        """Overlap exceeds window_size raises ValueError."""
        config = WindowConfig(10, 15, True)
        msg = r"overlap .* must be less than window_size"
        with pytest.raises(ValueError, match=msg):
            validate_window_config(config)

    def test_zero_overlap_valid(self) -> None:
        """Zero overlap is valid."""
        config = WindowConfig(10, 0, True)
        validate_window_config(config)


class TestValidateSummaryConfig:
    """Tests for validate_summary_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SummaryConfig(500, "gpt-3.5-turbo", 5)
        validate_summary_config(config)

    def test_zero_max_summary_tokens_raises(self) -> None:
        """Zero max_summary_tokens raises ValueError."""
        config = SummaryConfig(0, "gpt-3.5-turbo", 5)
        with pytest.raises(ValueError, match="max_summary_tokens must be positive"):
            validate_summary_config(config)

    def test_negative_max_summary_tokens_raises(self) -> None:
        """Negative max_summary_tokens raises ValueError."""
        config = SummaryConfig(-1, "gpt-3.5-turbo", 5)
        with pytest.raises(ValueError, match="max_summary_tokens must be positive"):
            validate_summary_config(config)

    def test_empty_summarizer_model_raises(self) -> None:
        """Empty summarizer_model raises ValueError."""
        config = SummaryConfig(500, "", 5)
        with pytest.raises(ValueError, match="summarizer_model cannot be empty"):
            validate_summary_config(config)

    def test_zero_update_frequency_raises(self) -> None:
        """Zero update_frequency raises ValueError."""
        config = SummaryConfig(500, "gpt-3.5-turbo", 0)
        with pytest.raises(ValueError, match="update_frequency must be positive"):
            validate_summary_config(config)

    def test_negative_update_frequency_raises(self) -> None:
        """Negative update_frequency raises ValueError."""
        config = SummaryConfig(500, "gpt-3.5-turbo", -1)
        with pytest.raises(ValueError, match="update_frequency must be positive"):
            validate_summary_config(config)


class TestValidateEntityConfig:
    """Tests for validate_entity_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = EntityConfig("gpt-3.5-turbo", 50, 0.1)
        validate_entity_config(config)

    def test_empty_entity_extraction_model_raises(self) -> None:
        """Empty entity_extraction_model raises ValueError."""
        config = EntityConfig("", 50, 0.1)
        with pytest.raises(ValueError, match="entity_extraction_model cannot be empty"):
            validate_entity_config(config)

    def test_zero_max_entities_raises(self) -> None:
        """Zero max_entities raises ValueError."""
        config = EntityConfig("gpt-3.5-turbo", 0, 0.1)
        with pytest.raises(ValueError, match="max_entities must be positive"):
            validate_entity_config(config)

    def test_negative_max_entities_raises(self) -> None:
        """Negative max_entities raises ValueError."""
        config = EntityConfig("gpt-3.5-turbo", -1, 0.1)
        with pytest.raises(ValueError, match="max_entities must be positive"):
            validate_entity_config(config)

    def test_decay_rate_below_zero_raises(self) -> None:
        """Decay rate below 0 raises ValueError."""
        config = EntityConfig("gpt-3.5-turbo", 50, -0.1)
        msg = r"decay_rate must be between 0\.0 and 1\.0"
        with pytest.raises(ValueError, match=msg):
            validate_entity_config(config)

    def test_decay_rate_above_one_raises(self) -> None:
        """Decay rate above 1 raises ValueError."""
        config = EntityConfig("gpt-3.5-turbo", 50, 1.1)
        msg = r"decay_rate must be between 0\.0 and 1\.0"
        with pytest.raises(ValueError, match=msg):
            validate_entity_config(config)

    @pytest.mark.parametrize("decay_rate", [0.0, 0.5, 1.0])
    def test_boundary_decay_rates_valid(self, decay_rate: float) -> None:
        """Boundary decay rates are valid."""
        config = EntityConfig("gpt-3.5-turbo", 50, decay_rate)
        validate_entity_config(config)


class TestValidateMemoryConfig:
    """Tests for validate_memory_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "input", "output")
        validate_memory_config(config)

    def test_empty_human_prefix_raises(self) -> None:
        """Empty human_prefix raises ValueError."""
        config = MemoryConfig(MemoryType.BUFFER, "", "AI", "input", "output")
        with pytest.raises(ValueError, match="human_prefix cannot be empty"):
            validate_memory_config(config)

    def test_empty_ai_prefix_raises(self) -> None:
        """Empty ai_prefix raises ValueError."""
        config = MemoryConfig(MemoryType.BUFFER, "Human", "", "input", "output")
        with pytest.raises(ValueError, match="ai_prefix cannot be empty"):
            validate_memory_config(config)

    def test_empty_input_key_raises(self) -> None:
        """Empty input_key raises ValueError."""
        config = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "", "output")
        with pytest.raises(ValueError, match="input_key cannot be empty"):
            validate_memory_config(config)

    def test_empty_output_key_raises(self) -> None:
        """Empty output_key raises ValueError."""
        config = MemoryConfig(MemoryType.BUFFER, "Human", "AI", "input", "")
        with pytest.raises(ValueError, match="output_key cannot be empty"):
            validate_memory_config(config)


class TestCreateBufferConfig:
    """Tests for create_buffer_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_buffer_config()
        assert config.max_messages == 100
        assert config.max_tokens == 4096
        assert config.return_messages is True

    def test_custom_max_messages(self) -> None:
        """Create config with custom max_messages."""
        config = create_buffer_config(max_messages=50)
        assert config.max_messages == 50

    def test_custom_max_tokens(self) -> None:
        """Create config with custom max_tokens."""
        config = create_buffer_config(max_tokens=8192)
        assert config.max_tokens == 8192

    def test_custom_return_messages(self) -> None:
        """Create config with custom return_messages."""
        config = create_buffer_config(return_messages=False)
        assert config.return_messages is False

    def test_zero_max_messages_raises(self) -> None:
        """Zero max_messages raises ValueError."""
        with pytest.raises(ValueError, match="max_messages must be positive"):
            create_buffer_config(max_messages=0)

    def test_negative_max_tokens_raises(self) -> None:
        """Negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be non-negative"):
            create_buffer_config(max_tokens=-1)


class TestCreateWindowConfig:
    """Tests for create_window_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_window_config()
        assert config.window_size == 10
        assert config.overlap == 2
        assert config.include_system is True

    def test_custom_window_size(self) -> None:
        """Create config with custom window_size."""
        config = create_window_config(window_size=20)
        assert config.window_size == 20

    def test_custom_overlap(self) -> None:
        """Create config with custom overlap."""
        config = create_window_config(overlap=5)
        assert config.overlap == 5

    def test_custom_include_system(self) -> None:
        """Create config with custom include_system."""
        config = create_window_config(include_system=False)
        assert config.include_system is False

    def test_zero_window_size_raises(self) -> None:
        """Zero window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            create_window_config(window_size=0)

    def test_overlap_exceeds_window_size_raises(self) -> None:
        """Overlap exceeds window_size raises ValueError."""
        msg = r"overlap .* must be less than window_size"
        with pytest.raises(ValueError, match=msg):
            create_window_config(window_size=10, overlap=15)


class TestCreateSummaryConfig:
    """Tests for create_summary_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_summary_config()
        assert config.max_summary_tokens == 500
        assert config.summarizer_model == "gpt-3.5-turbo"
        assert config.update_frequency == 5

    def test_custom_max_summary_tokens(self) -> None:
        """Create config with custom max_summary_tokens."""
        config = create_summary_config(max_summary_tokens=1000)
        assert config.max_summary_tokens == 1000

    def test_custom_summarizer_model(self) -> None:
        """Create config with custom summarizer_model."""
        config = create_summary_config(summarizer_model="gpt-4")
        assert config.summarizer_model == "gpt-4"

    def test_custom_update_frequency(self) -> None:
        """Create config with custom update_frequency."""
        config = create_summary_config(update_frequency=10)
        assert config.update_frequency == 10

    def test_zero_max_summary_tokens_raises(self) -> None:
        """Zero max_summary_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_summary_tokens must be positive"):
            create_summary_config(max_summary_tokens=0)

    def test_empty_summarizer_model_raises(self) -> None:
        """Empty summarizer_model raises ValueError."""
        with pytest.raises(ValueError, match="summarizer_model cannot be empty"):
            create_summary_config(summarizer_model="")


class TestCreateEntityConfig:
    """Tests for create_entity_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_entity_config()
        assert config.entity_extraction_model == "gpt-3.5-turbo"
        assert config.max_entities == 50
        assert config.decay_rate == 0.1

    def test_custom_entity_extraction_model(self) -> None:
        """Create config with custom entity_extraction_model."""
        config = create_entity_config(entity_extraction_model="gpt-4")
        assert config.entity_extraction_model == "gpt-4"

    def test_custom_max_entities(self) -> None:
        """Create config with custom max_entities."""
        config = create_entity_config(max_entities=100)
        assert config.max_entities == 100

    def test_custom_decay_rate(self) -> None:
        """Create config with custom decay_rate."""
        config = create_entity_config(decay_rate=0.05)
        assert config.decay_rate == 0.05

    def test_empty_entity_extraction_model_raises(self) -> None:
        """Empty entity_extraction_model raises ValueError."""
        with pytest.raises(ValueError, match="entity_extraction_model cannot be empty"):
            create_entity_config(entity_extraction_model="")

    def test_decay_rate_above_one_raises(self) -> None:
        """Decay rate above 1 raises ValueError."""
        msg = r"decay_rate must be between 0\.0 and 1\.0"
        with pytest.raises(ValueError, match=msg):
            create_entity_config(decay_rate=1.5)


class TestCreateMemoryConfig:
    """Tests for create_memory_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_memory_config()
        assert config.memory_type == MemoryType.BUFFER
        assert config.human_prefix == "Human"
        assert config.ai_prefix == "AI"
        assert config.input_key == "input"
        assert config.output_key == "output"

    @pytest.mark.parametrize(
        "memory_type,expected",
        [
            ("buffer", MemoryType.BUFFER),
            ("summary", MemoryType.SUMMARY),
            ("window", MemoryType.WINDOW),
            ("entity", MemoryType.ENTITY),
            ("conversation", MemoryType.CONVERSATION),
        ],
    )
    def test_all_memory_types(self, memory_type: str, expected: MemoryType) -> None:
        """Create config with all memory types."""
        config = create_memory_config(memory_type=memory_type)
        assert config.memory_type == expected

    def test_custom_human_prefix(self) -> None:
        """Create config with custom human_prefix."""
        config = create_memory_config(human_prefix="User")
        assert config.human_prefix == "User"

    def test_custom_ai_prefix(self) -> None:
        """Create config with custom ai_prefix."""
        config = create_memory_config(ai_prefix="Assistant")
        assert config.ai_prefix == "Assistant"

    def test_custom_input_key(self) -> None:
        """Create config with custom input_key."""
        config = create_memory_config(input_key="query")
        assert config.input_key == "query"

    def test_custom_output_key(self) -> None:
        """Create config with custom output_key."""
        config = create_memory_config(output_key="response")
        assert config.output_key == "response"

    def test_invalid_memory_type_raises(self) -> None:
        """Invalid memory_type raises ValueError."""
        with pytest.raises(ValueError, match="memory_type must be one of"):
            create_memory_config(memory_type="invalid")

    def test_empty_human_prefix_raises(self) -> None:
        """Empty human_prefix raises ValueError."""
        with pytest.raises(ValueError, match="human_prefix cannot be empty"):
            create_memory_config(human_prefix="")


class TestCreateConversationMessage:
    """Tests for create_conversation_message function."""

    def test_default_message(self) -> None:
        """Create default message."""
        msg = create_conversation_message("human", "Hello!")
        assert msg.role == "human"
        assert msg.content == "Hello!"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_with_timestamp(self) -> None:
        """Create message with timestamp."""
        ts = datetime(2024, 1, 1, 12, 0)
        msg = create_conversation_message("ai", "Hi!", timestamp=ts)
        assert msg.timestamp == ts

    def test_with_metadata(self) -> None:
        """Create message with metadata."""
        meta = {"source": "api", "version": 1}
        msg = create_conversation_message("human", "Hello!", metadata=meta)
        assert msg.metadata == meta

    def test_empty_role_raises(self) -> None:
        """Empty role raises ValueError."""
        with pytest.raises(ValueError, match="role cannot be empty"):
            create_conversation_message("", "Hello!")

    def test_empty_content_raises(self) -> None:
        """Empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            create_conversation_message("human", "")


class TestListMemoryTypes:
    """Tests for list_memory_types function."""

    def test_returns_list(self) -> None:
        """Returns a list."""
        types = list_memory_types()
        assert isinstance(types, list)

    def test_contains_buffer(self) -> None:
        """Contains buffer."""
        types = list_memory_types()
        assert "buffer" in types

    def test_contains_summary(self) -> None:
        """Contains summary."""
        types = list_memory_types()
        assert "summary" in types

    def test_contains_window(self) -> None:
        """Contains window."""
        types = list_memory_types()
        assert "window" in types

    def test_is_sorted(self) -> None:
        """List is sorted."""
        types = list_memory_types()
        assert types == sorted(types)

    def test_all_strings(self) -> None:
        """All items are strings."""
        types = list_memory_types()
        assert all(isinstance(t, str) for t in types)


class TestGetMemoryType:
    """Tests for get_memory_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("buffer", MemoryType.BUFFER),
            ("summary", MemoryType.SUMMARY),
            ("window", MemoryType.WINDOW),
            ("entity", MemoryType.ENTITY),
            ("conversation", MemoryType.CONVERSATION),
        ],
    )
    def test_valid_names(self, name: str, expected: MemoryType) -> None:
        """Get memory type by valid name."""
        assert get_memory_type(name) == expected

    def test_invalid_name_raises(self) -> None:
        """Invalid name raises ValueError."""
        with pytest.raises(ValueError, match="memory type must be one of"):
            get_memory_type("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="memory type must be one of"):
            get_memory_type("")


class TestEstimateMemoryTokens:
    """Tests for estimate_memory_tokens function."""

    def test_empty_messages(self) -> None:
        """Empty messages returns 0."""
        assert estimate_memory_tokens(()) == 0

    def test_single_message(self) -> None:
        """Single message returns positive count."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        tokens = estimate_memory_tokens((msg,))
        assert tokens > 0

    def test_multiple_messages(self) -> None:
        """Multiple messages accumulate tokens."""
        msgs = (
            ConversationMessage("human", "Hello!", datetime.now(), {}),
            ConversationMessage("ai", "Hi there!", datetime.now(), {}),
        )
        tokens = estimate_memory_tokens(msgs)
        assert tokens > 0

    def test_custom_chars_per_token(self) -> None:
        """Custom chars_per_token affects result."""
        msg = ConversationMessage("human", "Hello world", datetime.now(), {})
        tokens_4 = estimate_memory_tokens((msg,), chars_per_token=4.0)
        tokens_2 = estimate_memory_tokens((msg,), chars_per_token=2.0)
        assert tokens_2 > tokens_4

    def test_zero_chars_per_token_raises(self) -> None:
        """Zero chars_per_token raises ValueError."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            estimate_memory_tokens((msg,), chars_per_token=0)

    def test_negative_chars_per_token_raises(self) -> None:
        """Negative chars_per_token raises ValueError."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            estimate_memory_tokens((msg,), chars_per_token=-1.0)


class TestCalculateWindowMessages:
    """Tests for calculate_window_messages function."""

    def test_messages_less_than_window(self) -> None:
        """Messages less than window returns all messages."""
        assert calculate_window_messages(5, 10) == 5

    def test_messages_equals_window(self) -> None:
        """Messages equals window returns window size."""
        assert calculate_window_messages(10, 10) == 10

    def test_messages_greater_than_window(self) -> None:
        """Messages greater than window returns window size."""
        assert calculate_window_messages(50, 10) == 10

    def test_with_system_messages_included(self) -> None:
        """System messages included in count."""
        result = calculate_window_messages(
            50, 10, include_system=True, system_messages=2
        )
        assert result == 10

    def test_with_system_messages_excluded(self) -> None:
        """System messages excluded from count."""
        result = calculate_window_messages(
            50, 10, include_system=False, system_messages=2
        )
        assert result == 10  # min(48, 10)

    def test_zero_total_messages(self) -> None:
        """Zero total messages returns 0."""
        assert calculate_window_messages(0, 10) == 0

    def test_negative_total_messages_raises(self) -> None:
        """Negative total_messages raises ValueError."""
        with pytest.raises(ValueError, match="total_messages must be non-negative"):
            calculate_window_messages(-1, 10)

    def test_zero_window_size_raises(self) -> None:
        """Zero window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            calculate_window_messages(50, 0)

    def test_negative_window_size_raises(self) -> None:
        """Negative window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            calculate_window_messages(50, -1)

    def test_negative_system_messages_raises(self) -> None:
        """Negative system_messages raises ValueError."""
        with pytest.raises(ValueError, match="system_messages must be non-negative"):
            calculate_window_messages(50, 10, system_messages=-1)

    def test_system_messages_exceeds_total(self) -> None:
        """System messages exceeds total returns 0."""
        result = calculate_window_messages(
            5, 10, include_system=False, system_messages=10
        )
        assert result == 0


class TestCalculateMemorySizeBytes:
    """Tests for calculate_memory_size_bytes function."""

    def test_empty_messages(self) -> None:
        """Empty messages returns 0."""
        assert calculate_memory_size_bytes(()) == 0

    def test_single_message(self) -> None:
        """Single message returns positive size."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        size = calculate_memory_size_bytes((msg,))
        assert size > 0

    def test_multiple_messages(self) -> None:
        """Multiple messages accumulate size."""
        msgs = (
            ConversationMessage("human", "Hello!", datetime.now(), {}),
            ConversationMessage("ai", "Hi there!", datetime.now(), {}),
        )
        size = calculate_memory_size_bytes(msgs)
        single_size = calculate_memory_size_bytes((msgs[0],))
        assert size > single_size

    def test_with_metadata(self) -> None:
        """Metadata included in size."""
        msg_no_meta = ConversationMessage("human", "Hello!", datetime.now(), {})
        msg_with_meta = ConversationMessage(
            "human", "Hello!", datetime.now(), {"key": "value"}
        )
        size_no_meta = calculate_memory_size_bytes((msg_no_meta,))
        size_with_meta = calculate_memory_size_bytes((msg_with_meta,))
        assert size_with_meta > size_no_meta

    def test_without_metadata(self) -> None:
        """Metadata excluded when include_metadata=False."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {"key": "value"})
        size_with = calculate_memory_size_bytes((msg,), include_metadata=True)
        size_without = calculate_memory_size_bytes((msg,), include_metadata=False)
        assert size_without <= size_with


class TestCreateMemoryStats:
    """Tests for create_memory_stats function."""

    def test_empty_messages(self) -> None:
        """Empty messages returns zero stats."""
        stats = create_memory_stats(())
        assert stats.total_messages == 0
        assert stats.total_tokens == 0
        assert stats.memory_size_bytes == 0

    def test_single_message(self) -> None:
        """Single message returns positive stats."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        stats = create_memory_stats((msg,))
        assert stats.total_messages == 1
        assert stats.total_tokens > 0
        assert stats.memory_size_bytes > 0

    def test_multiple_messages(self) -> None:
        """Multiple messages returns accumulated stats."""
        msgs = (
            ConversationMessage("human", "Hello!", datetime.now(), {}),
            ConversationMessage("ai", "Hi there!", datetime.now(), {}),
        )
        stats = create_memory_stats(msgs)
        assert stats.total_messages == 2

    def test_custom_chars_per_token(self) -> None:
        """Custom chars_per_token affects token count."""
        msg = ConversationMessage("human", "Hello world", datetime.now(), {})
        stats_4 = create_memory_stats((msg,), chars_per_token=4.0)
        stats_2 = create_memory_stats((msg,), chars_per_token=2.0)
        assert stats_2.total_tokens > stats_4.total_tokens


class TestFormatMemoryStats:
    """Tests for format_memory_stats function."""

    def test_format_stats(self) -> None:
        """Format stats produces string."""
        stats = MemoryStats(50, 2048, 8192)
        formatted = format_memory_stats(stats)
        assert isinstance(formatted, str)

    def test_contains_messages(self) -> None:
        """Formatted string contains messages count."""
        stats = MemoryStats(50, 2048, 8192)
        formatted = format_memory_stats(stats)
        assert "50 messages" in formatted

    def test_contains_tokens(self) -> None:
        """Formatted string contains tokens count."""
        stats = MemoryStats(50, 2048, 8192)
        formatted = format_memory_stats(stats)
        assert "2048 tokens" in formatted

    def test_contains_kb_size(self) -> None:
        """Formatted string contains KB size."""
        stats = MemoryStats(50, 2048, 8192)
        formatted = format_memory_stats(stats)
        assert "KB" in formatted


class TestGetRecommendedBufferConfig:
    """Tests for get_recommended_buffer_config function."""

    def test_default_config(self) -> None:
        """Default config for chat use case."""
        config = get_recommended_buffer_config()
        assert config.max_messages > 0
        assert config.max_tokens > 0
        assert config.return_messages is True

    @pytest.mark.parametrize("use_case", ["chat", "agent", "qa"])
    def test_all_use_cases(self, use_case: str) -> None:
        """All use cases return valid config."""
        config = get_recommended_buffer_config(use_case=use_case)
        assert config.max_messages > 0
        assert config.max_tokens > 0

    def test_unknown_use_case_uses_default(self) -> None:
        """Unknown use case uses default ratios."""
        config = get_recommended_buffer_config(use_case="unknown")
        assert config.max_messages > 0
        assert config.max_tokens > 0

    def test_custom_context_size(self) -> None:
        """Custom context size affects config."""
        config_small = get_recommended_buffer_config(model_context_size=4096)
        config_large = get_recommended_buffer_config(model_context_size=8192)
        assert config_large.max_tokens >= config_small.max_tokens

    def test_max_tokens_within_context(self) -> None:
        """Max tokens stays within context size."""
        config = get_recommended_buffer_config(model_context_size=8192)
        assert config.max_tokens <= 8192

    def test_agent_has_fewer_messages_than_chat(self) -> None:
        """Agent use case has fewer messages than chat."""
        chat_config = get_recommended_buffer_config(use_case="chat")
        agent_config = get_recommended_buffer_config(use_case="agent")
        assert agent_config.max_messages <= chat_config.max_messages

    def test_qa_has_fewer_tokens_than_chat(self) -> None:
        """QA use case has fewer tokens than chat."""
        chat_config = get_recommended_buffer_config(use_case="chat")
        qa_config = get_recommended_buffer_config(use_case="qa")
        assert qa_config.max_tokens <= chat_config.max_tokens


class TestHypothesis:
    """Property-based tests using Hypothesis."""

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_valid_max_messages_accepted(self, max_messages: int) -> None:
        """Valid max_messages are accepted."""
        config = create_buffer_config(max_messages=max_messages)
        assert config.max_messages == max_messages

    @given(st.integers(min_value=0, max_value=100000))
    @settings(max_examples=50)
    def test_valid_max_tokens_accepted(self, max_tokens: int) -> None:
        """Valid max_tokens are accepted."""
        config = create_buffer_config(max_tokens=max_tokens)
        assert config.max_tokens == max_tokens

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_window_size_larger_than_overlap(self, window_size: int) -> None:
        """Window size larger than overlap is valid."""
        overlap = window_size // 2
        config = create_window_config(window_size=window_size, overlap=overlap)
        assert config.window_size > config.overlap

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=30)
    def test_valid_decay_rates_accepted(self, decay_rate: float) -> None:
        """Valid decay rates are accepted."""
        config = create_entity_config(decay_rate=decay_rate)
        assert config.decay_rate == decay_rate

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=30)
    def test_non_empty_prefixes_accepted(self, prefix: str) -> None:
        """Non-empty prefixes are accepted."""
        config = create_memory_config(human_prefix=prefix)
        assert config.human_prefix == prefix

    @given(st.sampled_from(list(MemoryType)))
    def test_all_memory_types_have_string_value(self, memory_type: MemoryType) -> None:
        """All memory types have string values."""
        result = get_memory_type(memory_type.value)
        assert result == memory_type

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_window_messages_never_exceeds_total_or_window(
        self, total_messages: int, window_size: int
    ) -> None:
        """Window messages never exceeds total or window size."""
        result = calculate_window_messages(total_messages, window_size)
        assert result <= total_messages
        assert result <= window_size

    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=30)
    def test_positive_chars_per_token_accepted(self, chars_per_token: float) -> None:
        """Positive chars_per_token is accepted."""
        msg = ConversationMessage("human", "Hello!", datetime.now(), {})
        tokens = estimate_memory_tokens((msg,), chars_per_token=chars_per_token)
        assert tokens >= 0
