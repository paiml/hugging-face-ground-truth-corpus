"""Tests for chat utilities functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.generation.chat import (
    VALID_MESSAGE_ROLES,
    VALID_TEMPLATE_FORMATS,
    VALID_TRUNCATION_STRATEGIES,
    ChatConfig,
    ChatMessage,
    ChatStats,
    ChatTemplateFormat,
    ConversationConfig,
    MessageRole,
    TemplateConfig,
    _get_template_tokens,
    count_conversation_tokens,
    create_chat_config,
    create_chat_message,
    create_conversation_config,
    create_template_config,
    format_chat_prompt,
    get_message_role,
    get_template_format,
    list_message_roles,
    list_template_formats,
    validate_chat_config,
    validate_chat_message,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_system_value(self) -> None:
        """Test SYSTEM role value."""
        assert MessageRole.SYSTEM.value == "system"

    def test_user_value(self) -> None:
        """Test USER role value."""
        assert MessageRole.USER.value == "user"

    def test_assistant_value(self) -> None:
        """Test ASSISTANT role value."""
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_function_value(self) -> None:
        """Test FUNCTION role value."""
        assert MessageRole.FUNCTION.value == "function"

    def test_tool_value(self) -> None:
        """Test TOOL role value."""
        assert MessageRole.TOOL.value == "tool"

    def test_all_roles_covered(self) -> None:
        """Test that all roles are covered."""
        expected_roles = {"system", "user", "assistant", "function", "tool"}
        actual_roles = {role.value for role in MessageRole}
        assert actual_roles == expected_roles


class TestChatTemplateFormat:
    """Tests for ChatTemplateFormat enum."""

    def test_chatml_value(self) -> None:
        """Test CHATML format value."""
        assert ChatTemplateFormat.CHATML.value == "chatml"

    def test_llama_value(self) -> None:
        """Test LLAMA format value."""
        assert ChatTemplateFormat.LLAMA.value == "llama"

    def test_alpaca_value(self) -> None:
        """Test ALPACA format value."""
        assert ChatTemplateFormat.ALPACA.value == "alpaca"

    def test_vicuna_value(self) -> None:
        """Test VICUNA format value."""
        assert ChatTemplateFormat.VICUNA.value == "vicuna"

    def test_mistral_value(self) -> None:
        """Test MISTRAL format value."""
        assert ChatTemplateFormat.MISTRAL.value == "mistral"

    def test_zephyr_value(self) -> None:
        """Test ZEPHYR format value."""
        assert ChatTemplateFormat.ZEPHYR.value == "zephyr"

    def test_all_formats_covered(self) -> None:
        """Test that all formats are covered."""
        expected_formats = {"chatml", "llama", "alpaca", "vicuna", "mistral", "zephyr"}
        actual_formats = {fmt.value for fmt in ChatTemplateFormat}
        assert actual_formats == expected_formats


class TestValidSets:
    """Tests for VALID_* module-level frozensets."""

    def test_valid_message_roles(self) -> None:
        """Test VALID_MESSAGE_ROLES contains all role values."""
        expected = frozenset({"system", "user", "assistant", "function", "tool"})
        assert expected == VALID_MESSAGE_ROLES

    def test_valid_template_formats(self) -> None:
        """Test VALID_TEMPLATE_FORMATS contains all format values."""
        expected = frozenset(
            {"chatml", "llama", "alpaca", "vicuna", "mistral", "zephyr"}
        )
        assert expected == VALID_TEMPLATE_FORMATS

    def test_valid_truncation_strategies(self) -> None:
        """Test VALID_TRUNCATION_STRATEGIES contains all strategies."""
        expected = frozenset({"oldest_first", "newest_first", "middle_out"})
        assert expected == VALID_TRUNCATION_STRATEGIES


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Hello!",
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_create_tool_message(self) -> None:
        """Test creating a tool message with all fields."""
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content='{"result": 42}',
            name="calculator",
            tool_calls=None,
            tool_call_id="call_123",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.content == '{"result": 42}'
        assert msg.name == "calculator"
        assert msg.tool_call_id == "call_123"

    def test_create_assistant_with_tool_calls(self) -> None:
        """Test creating assistant message with tool calls."""
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Let me calculate that.",
            name=None,
            tool_calls='[{"id": "call_123", "function": "calc"}]',
            tool_call_id=None,
        )
        assert msg.role == MessageRole.ASSISTANT
        assert msg.tool_calls is not None

    def test_frozen(self) -> None:
        """Test that ChatMessage is immutable."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Hello!",
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        with pytest.raises(AttributeError):
            msg.content = "Modified"  # type: ignore[misc]


class TestChatConfig:
    """Tests for ChatConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a chat config."""
        config = ChatConfig(
            template_format=ChatTemplateFormat.CHATML,
            system_prompt="Be helpful.",
            max_turns=10,
            include_system=True,
        )
        assert config.template_format == ChatTemplateFormat.CHATML
        assert config.system_prompt == "Be helpful."
        assert config.max_turns == 10
        assert config.include_system is True

    def test_config_with_none_system_prompt(self) -> None:
        """Test config with None system prompt."""
        config = ChatConfig(
            template_format=ChatTemplateFormat.LLAMA,
            system_prompt=None,
            max_turns=5,
            include_system=False,
        )
        assert config.system_prompt is None

    def test_frozen(self) -> None:
        """Test that ChatConfig is immutable."""
        config = ChatConfig(
            template_format=ChatTemplateFormat.CHATML,
            system_prompt="Test",
            max_turns=10,
            include_system=True,
        )
        with pytest.raises(AttributeError):
            config.max_turns = 20  # type: ignore[misc]


class TestTemplateConfig:
    """Tests for TemplateConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a template config."""
        config = TemplateConfig(
            bos_token="<s>",
            eos_token="</s>",
            user_prefix="User: ",
            assistant_prefix="Assistant: ",
            system_prefix="System: ",
        )
        assert config.bos_token == "<s>"
        assert config.eos_token == "</s>"
        assert config.user_prefix == "User: "
        assert config.assistant_prefix == "Assistant: "
        assert config.system_prefix == "System: "

    def test_frozen(self) -> None:
        """Test that TemplateConfig is immutable."""
        config = TemplateConfig(
            bos_token="<s>",
            eos_token="</s>",
            user_prefix="User: ",
            assistant_prefix="Assistant: ",
            system_prefix="System: ",
        )
        with pytest.raises(AttributeError):
            config.bos_token = "<begin>"  # type: ignore[misc]


class TestConversationConfig:
    """Tests for ConversationConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a conversation config."""
        config = ConversationConfig(
            max_history=50,
            truncation_strategy="oldest_first",
            preserve_system=True,
        )
        assert config.max_history == 50
        assert config.truncation_strategy == "oldest_first"
        assert config.preserve_system is True

    def test_different_truncation_strategies(self) -> None:
        """Test different truncation strategy values."""
        for strategy in ["oldest_first", "newest_first", "middle_out"]:
            config = ConversationConfig(
                max_history=10,
                truncation_strategy=strategy,  # type: ignore[arg-type]
                preserve_system=False,
            )
            assert config.truncation_strategy == strategy

    def test_frozen(self) -> None:
        """Test that ConversationConfig is immutable."""
        config = ConversationConfig(
            max_history=50,
            truncation_strategy="oldest_first",
            preserve_system=True,
        )
        with pytest.raises(AttributeError):
            config.max_history = 100  # type: ignore[misc]


class TestChatStats:
    """Tests for ChatStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating chat stats."""
        stats = ChatStats(
            total_turns=5,
            user_tokens=150,
            assistant_tokens=300,
            system_tokens=50,
        )
        assert stats.total_turns == 5
        assert stats.user_tokens == 150
        assert stats.assistant_tokens == 300
        assert stats.system_tokens == 50

    def test_total_tokens_calculation(self) -> None:
        """Test that token counts can be summed."""
        stats = ChatStats(
            total_turns=3,
            user_tokens=100,
            assistant_tokens=200,
            system_tokens=50,
        )
        total = stats.user_tokens + stats.assistant_tokens + stats.system_tokens
        assert total == 350

    def test_frozen(self) -> None:
        """Test that ChatStats is immutable."""
        stats = ChatStats(
            total_turns=5,
            user_tokens=150,
            assistant_tokens=300,
            system_tokens=50,
        )
        with pytest.raises(AttributeError):
            stats.total_turns = 10  # type: ignore[misc]


class TestValidateChatMessage:
    """Tests for validate_chat_message function."""

    def test_valid_user_message(self) -> None:
        """Test validation of valid user message."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Hello",
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        validate_chat_message(msg)  # Should not raise

    def test_valid_tool_message(self) -> None:
        """Test validation of valid tool message."""
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content="Result",
            name="function",
            tool_calls=None,
            tool_call_id="call_123",
        )
        validate_chat_message(msg)  # Should not raise

    def test_none_message_raises_error(self) -> None:
        """Test that None message raises ValueError."""
        with pytest.raises(ValueError, match="message cannot be None"):
            validate_chat_message(None)  # type: ignore[arg-type]

    def test_none_content_raises_error(self) -> None:
        """Test that None content raises ValueError."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content=None,  # type: ignore[arg-type]
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        with pytest.raises(ValueError, match="message content cannot be None"):
            validate_chat_message(msg)

    def test_tool_message_without_id_raises_error(self) -> None:
        """Test that tool message without tool_call_id raises ValueError."""
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content="Result",
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        with pytest.raises(ValueError, match="tool messages must have tool_call_id"):
            validate_chat_message(msg)

    def test_tool_message_with_empty_id_raises_error(self) -> None:
        """Test that tool message with empty tool_call_id raises ValueError."""
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content="Result",
            name=None,
            tool_calls=None,
            tool_call_id="",
        )
        with pytest.raises(ValueError, match="tool messages must have tool_call_id"):
            validate_chat_message(msg)


class TestValidateChatConfig:
    """Tests for validate_chat_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ChatConfig(
            template_format=ChatTemplateFormat.CHATML,
            system_prompt="Be helpful.",
            max_turns=10,
            include_system=True,
        )
        validate_chat_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_chat_config(None)  # type: ignore[arg-type]

    def test_zero_max_turns_raises_error(self) -> None:
        """Test that zero max_turns raises ValueError."""
        config = ChatConfig(
            template_format=ChatTemplateFormat.CHATML,
            system_prompt=None,
            max_turns=0,
            include_system=True,
        )
        with pytest.raises(ValueError, match="max_turns must be positive"):
            validate_chat_config(config)

    def test_negative_max_turns_raises_error(self) -> None:
        """Test that negative max_turns raises ValueError."""
        config = ChatConfig(
            template_format=ChatTemplateFormat.CHATML,
            system_prompt=None,
            max_turns=-5,
            include_system=True,
        )
        with pytest.raises(ValueError, match="max_turns must be positive"):
            validate_chat_config(config)


class TestCreateChatMessage:
    """Tests for create_chat_message function."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = create_chat_message("user", "Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = create_chat_message("assistant", "How can I help?")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "How can I help?"

    def test_create_system_message(self) -> None:
        """Test creating a system message."""
        msg = create_chat_message("system", "You are helpful.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful."

    def test_create_tool_message(self) -> None:
        """Test creating a tool message with all fields."""
        msg = create_chat_message(
            "tool",
            '{"result": 42}',
            name="calculator",
            tool_call_id="call_abc",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.content == '{"result": 42}'
        assert msg.name == "calculator"
        assert msg.tool_call_id == "call_abc"

    def test_create_function_message(self) -> None:
        """Test creating a function message (legacy)."""
        msg = create_chat_message(
            "function",
            "result",
            name="my_function",
            tool_call_id="call_xyz",
        )
        assert msg.role == MessageRole.FUNCTION
        assert msg.name == "my_function"

    def test_create_message_with_tool_calls(self) -> None:
        """Test creating message with tool_calls."""
        msg = create_chat_message(
            "assistant",
            "Let me call a tool.",
            tool_calls='[{"id": "call_1"}]',
        )
        assert msg.tool_calls == '[{"id": "call_1"}]'

    def test_invalid_role_raises_error(self) -> None:
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="role must be one of"):
            create_chat_message("invalid", "Hello")

    def test_empty_content_raises_error(self) -> None:
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            create_chat_message("user", "")

    @pytest.mark.parametrize("role", list(VALID_MESSAGE_ROLES))
    def test_all_valid_roles(self, role: str) -> None:
        """Test that all valid roles work."""
        # Tool role needs tool_call_id
        kwargs = {}
        if role == "tool":
            kwargs["tool_call_id"] = "call_test"
        msg = create_chat_message(role, "Content", **kwargs)
        assert msg.role.value == role


class TestCreateChatConfig:
    """Tests for create_chat_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_chat_config()
        assert config.template_format == ChatTemplateFormat.CHATML
        assert config.system_prompt is None
        assert config.max_turns == 10
        assert config.include_system is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = create_chat_config(
            template_format="llama",
            system_prompt="Be concise.",
            max_turns=20,
            include_system=False,
        )
        assert config.template_format == ChatTemplateFormat.LLAMA
        assert config.system_prompt == "Be concise."
        assert config.max_turns == 20
        assert config.include_system is False

    def test_invalid_template_format_raises_error(self) -> None:
        """Test that invalid template_format raises ValueError."""
        with pytest.raises(ValueError, match="template_format must be one of"):
            create_chat_config(template_format="invalid")

    def test_zero_max_turns_raises_error(self) -> None:
        """Test that zero max_turns raises ValueError."""
        with pytest.raises(ValueError, match="max_turns must be positive"):
            create_chat_config(max_turns=0)

    def test_negative_max_turns_raises_error(self) -> None:
        """Test that negative max_turns raises ValueError."""
        with pytest.raises(ValueError, match="max_turns must be positive"):
            create_chat_config(max_turns=-1)

    @pytest.mark.parametrize("template_format", list(VALID_TEMPLATE_FORMATS))
    def test_all_valid_template_formats(self, template_format: str) -> None:
        """Test that all valid template formats work."""
        config = create_chat_config(template_format=template_format)
        assert config.template_format.value == template_format


class TestCreateTemplateConfig:
    """Tests for create_template_config function."""

    def test_default_values(self) -> None:
        """Test default template configuration values."""
        config = create_template_config()
        assert config.bos_token == "<s>"
        assert config.eos_token == "</s>"
        assert config.user_prefix == "User: "
        assert config.assistant_prefix == "Assistant: "
        assert config.system_prefix == "System: "

    def test_custom_values(self) -> None:
        """Test custom template configuration values."""
        config = create_template_config(
            bos_token="[BOS]",
            eos_token="[EOS]",
            user_prefix="Human: ",
            assistant_prefix="AI: ",
            system_prefix="Context: ",
        )
        assert config.bos_token == "[BOS]"
        assert config.eos_token == "[EOS]"
        assert config.user_prefix == "Human: "
        assert config.assistant_prefix == "AI: "
        assert config.system_prefix == "Context: "

    def test_empty_tokens_allowed(self) -> None:
        """Test that empty tokens are allowed."""
        config = create_template_config(
            bos_token="",
            eos_token="",
            user_prefix="",
            assistant_prefix="",
            system_prefix="",
        )
        assert config.bos_token == ""
        assert config.eos_token == ""


class TestCreateConversationConfig:
    """Tests for create_conversation_config function."""

    def test_default_values(self) -> None:
        """Test default conversation configuration values."""
        config = create_conversation_config()
        assert config.max_history == 50
        assert config.truncation_strategy == "oldest_first"
        assert config.preserve_system is True

    def test_custom_values(self) -> None:
        """Test custom conversation configuration values."""
        config = create_conversation_config(
            max_history=100,
            truncation_strategy="newest_first",
            preserve_system=False,
        )
        assert config.max_history == 100
        assert config.truncation_strategy == "newest_first"
        assert config.preserve_system is False

    def test_zero_max_history_raises_error(self) -> None:
        """Test that zero max_history raises ValueError."""
        with pytest.raises(ValueError, match="max_history must be positive"):
            create_conversation_config(max_history=0)

    def test_negative_max_history_raises_error(self) -> None:
        """Test that negative max_history raises ValueError."""
        with pytest.raises(ValueError, match="max_history must be positive"):
            create_conversation_config(max_history=-10)

    def test_invalid_truncation_strategy_raises_error(self) -> None:
        """Test that invalid truncation_strategy raises ValueError."""
        with pytest.raises(ValueError, match="truncation_strategy must be one of"):
            create_conversation_config(
                truncation_strategy="invalid"  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("strategy", ["oldest_first", "newest_first", "middle_out"])
    def test_all_valid_truncation_strategies(self, strategy: str) -> None:
        """Test that all valid truncation strategies work."""
        config = create_conversation_config(
            truncation_strategy=strategy  # type: ignore[arg-type]
        )
        assert config.truncation_strategy == strategy


class TestListMessageRoles:
    """Tests for list_message_roles function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        roles = list_message_roles()
        assert isinstance(roles, list)

    def test_contains_expected_roles(self) -> None:
        """Test that list contains expected roles."""
        roles = list_message_roles()
        assert "user" in roles
        assert "assistant" in roles
        assert "system" in roles
        assert "function" in roles
        assert "tool" in roles

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        roles = list_message_roles()
        assert roles == sorted(roles)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        roles = list_message_roles()
        assert all(isinstance(r, str) for r in roles)


class TestListTemplateFormats:
    """Tests for list_template_formats function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        formats = list_template_formats()
        assert isinstance(formats, list)

    def test_contains_expected_formats(self) -> None:
        """Test that list contains expected formats."""
        formats = list_template_formats()
        assert "chatml" in formats
        assert "llama" in formats
        assert "mistral" in formats
        assert "alpaca" in formats
        assert "vicuna" in formats
        assert "zephyr" in formats

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        formats = list_template_formats()
        assert formats == sorted(formats)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        formats = list_template_formats()
        assert all(isinstance(f, str) for f in formats)


class TestGetMessageRole:
    """Tests for get_message_role function."""

    def test_get_user_role(self) -> None:
        """Test getting USER role."""
        role = get_message_role("user")
        assert role == MessageRole.USER

    def test_get_assistant_role(self) -> None:
        """Test getting ASSISTANT role."""
        role = get_message_role("assistant")
        assert role == MessageRole.ASSISTANT

    def test_get_system_role(self) -> None:
        """Test getting SYSTEM role."""
        role = get_message_role("system")
        assert role == MessageRole.SYSTEM

    def test_get_function_role(self) -> None:
        """Test getting FUNCTION role."""
        role = get_message_role("function")
        assert role == MessageRole.FUNCTION

    def test_get_tool_role(self) -> None:
        """Test getting TOOL role."""
        role = get_message_role("tool")
        assert role == MessageRole.TOOL

    def test_invalid_role_raises_error(self) -> None:
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="role must be one of"):
            get_message_role("invalid")

    @pytest.mark.parametrize("role_name", list(VALID_MESSAGE_ROLES))
    def test_all_valid_roles(self, role_name: str) -> None:
        """Test that all valid role names work."""
        role = get_message_role(role_name)
        assert role.value == role_name


class TestGetTemplateFormat:
    """Tests for get_template_format function."""

    def test_get_chatml_format(self) -> None:
        """Test getting CHATML format."""
        fmt = get_template_format("chatml")
        assert fmt == ChatTemplateFormat.CHATML

    def test_get_llama_format(self) -> None:
        """Test getting LLAMA format."""
        fmt = get_template_format("llama")
        assert fmt == ChatTemplateFormat.LLAMA

    def test_get_mistral_format(self) -> None:
        """Test getting MISTRAL format."""
        fmt = get_template_format("mistral")
        assert fmt == ChatTemplateFormat.MISTRAL

    def test_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="template_format must be one of"):
            get_template_format("invalid")

    @pytest.mark.parametrize("format_name", list(VALID_TEMPLATE_FORMATS))
    def test_all_valid_formats(self, format_name: str) -> None:
        """Test that all valid format names work."""
        fmt = get_template_format(format_name)
        assert fmt.value == format_name


class TestGetTemplateTokens:
    """Tests for _get_template_tokens function."""

    def test_chatml_tokens(self) -> None:
        """Test CHATML template tokens."""
        config = _get_template_tokens(ChatTemplateFormat.CHATML)
        assert config.bos_token == ""
        assert config.eos_token == "<|im_end|>\n"
        assert config.user_prefix == "<|im_start|>user\n"
        assert config.assistant_prefix == "<|im_start|>assistant\n"
        assert config.system_prefix == "<|im_start|>system\n"

    def test_llama_tokens(self) -> None:
        """Test LLAMA template tokens."""
        config = _get_template_tokens(ChatTemplateFormat.LLAMA)
        assert config.bos_token == "<s>"
        assert config.eos_token == "</s>"
        assert config.user_prefix == "[INST] "
        assert config.assistant_prefix == " [/INST] "
        assert config.system_prefix == "<<SYS>>\n"

    def test_alpaca_tokens(self) -> None:
        """Test ALPACA template tokens."""
        config = _get_template_tokens(ChatTemplateFormat.ALPACA)
        assert config.bos_token == ""
        assert config.eos_token == "\n\n"
        assert config.user_prefix == "### Instruction:\n"
        assert config.assistant_prefix == "### Response:\n"
        assert config.system_prefix == "### System:\n"

    def test_vicuna_tokens(self) -> None:
        """Test VICUNA template tokens."""
        config = _get_template_tokens(ChatTemplateFormat.VICUNA)
        assert config.bos_token == ""
        assert config.eos_token == "</s>"
        assert config.user_prefix == "USER: "
        assert config.assistant_prefix == "ASSISTANT: "
        assert config.system_prefix == "SYSTEM: "

    def test_mistral_tokens(self) -> None:
        """Test MISTRAL template tokens."""
        config = _get_template_tokens(ChatTemplateFormat.MISTRAL)
        assert config.bos_token == "<s>"
        assert config.eos_token == "</s>"
        assert config.user_prefix == "[INST] "
        assert config.assistant_prefix == " [/INST]"
        assert config.system_prefix == ""

    def test_zephyr_tokens(self) -> None:
        """Test ZEPHYR template tokens."""
        config = _get_template_tokens(ChatTemplateFormat.ZEPHYR)
        assert config.bos_token == "<|system|>"
        assert config.eos_token == "</s>"
        assert config.user_prefix == "<|user|>\n"
        assert config.assistant_prefix == "<|assistant|>\n"
        assert config.system_prefix == "<|system|>\n"

    @pytest.mark.parametrize("fmt", list(ChatTemplateFormat))
    def test_all_formats_return_template_config(self, fmt: ChatTemplateFormat) -> None:
        """Test that all formats return TemplateConfig."""
        config = _get_template_tokens(fmt)
        assert isinstance(config, TemplateConfig)


class TestFormatChatPrompt:
    """Tests for format_chat_prompt function."""

    def test_single_user_message(self) -> None:
        """Test formatting single user message."""
        messages = [create_chat_message("user", "Hello!")]
        config = create_chat_config(template_format="chatml")
        prompt = format_chat_prompt(messages, config)

        assert "<|im_start|>user\n" in prompt
        assert "Hello!" in prompt
        assert "<|im_start|>assistant\n" in prompt

    def test_user_assistant_exchange(self) -> None:
        """Test formatting user-assistant exchange."""
        messages = [
            create_chat_message("user", "What is 2+2?"),
            create_chat_message("assistant", "4"),
        ]
        config = create_chat_config(template_format="chatml")
        prompt = format_chat_prompt(messages, config)

        assert "What is 2+2?" in prompt
        assert "4" in prompt

    def test_with_system_prompt(self) -> None:
        """Test formatting with system prompt from config."""
        messages = [create_chat_message("user", "Hello")]
        config = create_chat_config(
            template_format="chatml",
            system_prompt="You are helpful.",
        )
        prompt = format_chat_prompt(messages, config)

        assert "You are helpful." in prompt
        assert "<|im_start|>system\n" in prompt

    def test_system_message_in_messages(self) -> None:
        """Test system message within messages list."""
        messages = [
            create_chat_message("system", "Be concise."),
            create_chat_message("user", "Hello"),
        ]
        config = create_chat_config(template_format="chatml", include_system=True)
        prompt = format_chat_prompt(messages, config)

        assert "Be concise." in prompt

    def test_exclude_system(self) -> None:
        """Test excluding system messages."""
        messages = [
            create_chat_message("system", "Be concise."),
            create_chat_message("user", "Hello"),
        ]
        config = create_chat_config(
            template_format="chatml",
            include_system=False,
        )
        prompt = format_chat_prompt(messages, config)

        assert "Be concise." not in prompt

    def test_no_generation_prompt(self) -> None:
        """Test without generation prompt."""
        messages = [create_chat_message("user", "Hello")]
        config = create_chat_config(template_format="chatml")
        prompt = format_chat_prompt(messages, config, add_generation_prompt=False)

        # Should not end with assistant prefix
        assert not prompt.endswith("<|im_start|>assistant\n")

    def test_max_turns_limit(self) -> None:
        """Test that max_turns limits conversation."""
        messages = [
            create_chat_message("user", "First"),
            create_chat_message("assistant", "Response 1"),
            create_chat_message("user", "Second"),
            create_chat_message("assistant", "Response 2"),
            create_chat_message("user", "Third"),
        ]
        config = create_chat_config(template_format="chatml", max_turns=2)
        prompt = format_chat_prompt(messages, config)

        assert "First" in prompt
        assert "Second" in prompt
        assert "Third" not in prompt

    def test_tool_message_formatting(self) -> None:
        """Test formatting tool messages."""
        messages = [
            create_chat_message("user", "Calculate 5+5"),
            create_chat_message(
                "tool",
                '{"result": 10}',
                name="calculator",
                tool_call_id="call_1",
            ),
        ]
        config = create_chat_config(template_format="chatml")
        prompt = format_chat_prompt(messages, config)

        assert "[calculator]:" in prompt
        assert '{"result": 10}' in prompt

    def test_function_message_formatting(self) -> None:
        """Test formatting function messages."""
        messages = [
            create_chat_message("user", "Get data"),
            create_chat_message(
                "function",
                "data here",
                name="get_data",
                tool_call_id="func_1",
            ),
        ]
        config = create_chat_config(template_format="chatml")
        prompt = format_chat_prompt(messages, config)

        assert "[get_data]:" in prompt

    def test_tool_message_without_name(self) -> None:
        """Test tool message defaults to 'tool' when name is None."""
        messages = [
            create_chat_message(
                "tool",
                "result",
                tool_call_id="call_1",
            ),
        ]
        config = create_chat_config(template_format="chatml")
        prompt = format_chat_prompt(messages, config)

        assert "[tool]:" in prompt

    def test_empty_messages_raises_error(self) -> None:
        """Test that empty messages raises ValueError."""
        config = create_chat_config()
        with pytest.raises(ValueError, match="messages cannot be empty"):
            format_chat_prompt([], config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        messages = [create_chat_message("user", "Hello")]
        with pytest.raises(ValueError, match="config cannot be None"):
            format_chat_prompt(messages, None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("template_format", list(VALID_TEMPLATE_FORMATS))
    def test_all_template_formats(self, template_format: str) -> None:
        """Test formatting works with all template formats."""
        messages = [
            create_chat_message("user", "Hello"),
            create_chat_message("assistant", "Hi there"),
        ]
        config = create_chat_config(template_format=template_format)
        prompt = format_chat_prompt(messages, config)

        # Basic sanity check - messages should appear
        assert "Hello" in prompt
        assert "Hi there" in prompt


class TestCountConversationTokens:
    """Tests for count_conversation_tokens function."""

    def test_basic_counting(self) -> None:
        """Test basic token counting."""
        messages = [
            create_chat_message("user", "Hello!"),  # 6 chars = 1.5 -> 1 token
        ]
        stats = count_conversation_tokens(messages)
        assert stats.total_turns == 1
        assert stats.user_tokens > 0
        assert stats.assistant_tokens == 0
        assert stats.system_tokens == 0

    def test_all_message_types(self) -> None:
        """Test counting with all message types."""
        messages = [
            create_chat_message("system", "Be helpful."),  # 11 chars
            create_chat_message("user", "Hello there!"),  # 12 chars
            create_chat_message("assistant", "Hi! How can I help?"),  # 19 chars
        ]
        stats = count_conversation_tokens(messages)
        assert stats.total_turns == 1
        assert stats.system_tokens > 0
        assert stats.user_tokens > 0
        assert stats.assistant_tokens > 0

    def test_multiple_turns(self) -> None:
        """Test counting with multiple conversation turns."""
        messages = [
            create_chat_message("user", "First"),
            create_chat_message("assistant", "Reply 1"),
            create_chat_message("user", "Second"),
            create_chat_message("assistant", "Reply 2"),
        ]
        stats = count_conversation_tokens(messages)
        assert stats.total_turns == 2

    def test_tool_messages_count_as_user(self) -> None:
        """Test that tool messages count toward user tokens."""
        tool_msg = create_chat_message(
            "tool",
            "result data",
            tool_call_id="call_1",
        )
        stats = count_conversation_tokens([tool_msg])
        assert stats.user_tokens > 0
        assert stats.total_turns == 0  # Tool messages don't count as turns

    def test_function_messages_count_as_user(self) -> None:
        """Test that function messages count toward user tokens."""
        func_msg = create_chat_message(
            "function",
            "function result",
            name="my_func",
            tool_call_id="func_1",
        )
        stats = count_conversation_tokens([func_msg])
        assert stats.user_tokens > 0

    def test_custom_chars_per_token(self) -> None:
        """Test with custom chars_per_token."""
        messages = [
            create_chat_message("user", "Hello there!"),  # 12 chars
        ]
        # With 2 chars per token, 12 chars = 6 tokens
        stats = count_conversation_tokens(messages, chars_per_token=2.0)
        assert stats.user_tokens == 6

        # With 6 chars per token, 12 chars = 2 tokens
        stats = count_conversation_tokens(messages, chars_per_token=6.0)
        assert stats.user_tokens == 2

    def test_empty_messages_list(self) -> None:
        """Test with empty messages list."""
        stats = count_conversation_tokens([])
        assert stats.total_turns == 0
        assert stats.user_tokens == 0
        assert stats.assistant_tokens == 0
        assert stats.system_tokens == 0

    def test_none_messages_raises_error(self) -> None:
        """Test that None messages raises ValueError."""
        with pytest.raises(ValueError, match="messages cannot be None"):
            count_conversation_tokens(None)  # type: ignore[arg-type]

    def test_zero_chars_per_token_raises_error(self) -> None:
        """Test that zero chars_per_token raises ValueError."""
        messages = [create_chat_message("user", "Hello")]
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            count_conversation_tokens(messages, chars_per_token=0)

    def test_negative_chars_per_token_raises_error(self) -> None:
        """Test that negative chars_per_token raises ValueError."""
        messages = [create_chat_message("user", "Hello")]
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            count_conversation_tokens(messages, chars_per_token=-1.0)

    @given(
        content_length=st.integers(min_value=1, max_value=100),
        chars_per_token=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_token_count_calculation(
        self, content_length: int, chars_per_token: float
    ) -> None:
        """Test that token count equals len(content) / chars_per_token."""
        content = "x" * content_length
        messages = [create_chat_message("user", content)]
        stats = count_conversation_tokens(messages, chars_per_token=chars_per_token)
        expected_tokens = int(content_length / chars_per_token)
        assert stats.user_tokens == expected_tokens


class TestHypothesisProperties:
    """Property-based tests using Hypothesis."""

    @given(
        role=st.sampled_from(["user", "assistant", "system"]),
        content=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=20)
    def test_create_message_roundtrip(self, role: str, content: str) -> None:
        """Test that created messages have correct role and content."""
        msg = create_chat_message(role, content)
        assert msg.role.value == role
        assert msg.content == content

    @given(
        max_turns=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_chat_config_max_turns(self, max_turns: int) -> None:
        """Test that chat config preserves max_turns."""
        config = create_chat_config(max_turns=max_turns)
        assert config.max_turns == max_turns

    @given(
        max_history=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=20)
    def test_conversation_config_max_history(self, max_history: int) -> None:
        """Test that conversation config preserves max_history."""
        config = create_conversation_config(max_history=max_history)
        assert config.max_history == max_history

    @given(
        num_messages=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10)
    def test_format_prompt_non_empty(self, num_messages: int) -> None:
        """Test that format_chat_prompt always produces non-empty output."""
        messages = [
            create_chat_message("user" if i % 2 == 0 else "assistant", f"Message {i}")
            for i in range(num_messages)
        ]
        config = create_chat_config()
        prompt = format_chat_prompt(messages, config)
        assert len(prompt) > 0
