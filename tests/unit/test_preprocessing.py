"""Tests for preprocessing functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.tokenization import (
    create_preprocessing_function,
    preprocess_text,
    tokenize_batch,
)


class TestPreprocessText:
    """Tests for preprocess_text function."""

    def test_basic_preprocessing(self) -> None:
        """Test basic text preprocessing."""
        result = preprocess_text("  Hello World  ")
        assert result == "hello world"

    def test_lowercase(self) -> None:
        """Test lowercase conversion."""
        result = preprocess_text("HELLO WORLD")
        assert result == "hello world"

    def test_no_lowercase(self) -> None:
        """Test with lowercase disabled."""
        result = preprocess_text("HELLO", lowercase=False)
        assert result == "HELLO"

    def test_strip_whitespace(self) -> None:
        """Test whitespace stripping."""
        result = preprocess_text("  hello  world  ")
        assert result == "hello world"

    def test_no_strip_whitespace(self) -> None:
        """Test with whitespace stripping disabled."""
        result = preprocess_text("  hello  ", strip_whitespace=False)
        assert result == "  hello  "

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        result = preprocess_text("")
        assert result == ""

    def test_only_whitespace(self) -> None:
        """Test string with only whitespace."""
        result = preprocess_text("   ")
        assert result == ""

    def test_unicode_handling(self) -> None:
        """Test Unicode character handling."""
        result = preprocess_text("  Héllo Wörld  ")
        assert result == "héllo wörld"

    def test_emoji_handling(self) -> None:
        """Test emoji preservation."""
        # Emojis should be preserved (not lowercased obviously)
        result = preprocess_text("Hello 👋 World", lowercase=False)
        assert "👋" in result

    @given(st.text(max_size=1000))
    @settings(max_examples=100)
    def test_idempotency(self, text: str) -> None:
        """Test that preprocessing is idempotent."""
        result1 = preprocess_text(text)
        result2 = preprocess_text(result1)
        assert result1 == result2

    @given(st.text(max_size=100))
    def test_returns_string(self, text: str) -> None:
        """Test that result is always a string."""
        result = preprocess_text(text)
        assert isinstance(result, str)


class TestTokenizeBatch:
    """Tests for tokenize_batch function."""

    def test_empty_texts_raises_error(self) -> None:
        """Test that empty texts list raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            tokenize_batch([], MagicMock())

    def test_invalid_max_length_zero(self) -> None:
        """Test that max_length=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            tokenize_batch(["hello"], MagicMock(), max_length=0)

    def test_invalid_max_length_negative(self) -> None:
        """Test that negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            tokenize_batch(["hello"], MagicMock(), max_length=-1)

    def test_calls_tokenizer(self, mock_tokenizer: MagicMock) -> None:
        """Test that tokenizer is called correctly."""
        result = tokenize_batch(["hello", "world"], mock_tokenizer)

        mock_tokenizer.assert_called_once_with(
            ["hello", "world"],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        assert result == mock_tokenizer.return_value

    def test_custom_max_length(self, mock_tokenizer: MagicMock) -> None:
        """Test custom max_length parameter."""
        tokenize_batch(["hello"], mock_tokenizer, max_length=256)

        call_kwargs = mock_tokenizer.call_args.kwargs
        assert call_kwargs["max_length"] == 256

    def test_custom_padding(self, mock_tokenizer: MagicMock) -> None:
        """Test custom padding parameter."""
        tokenize_batch(["hello"], mock_tokenizer, padding="longest")

        call_kwargs = mock_tokenizer.call_args.kwargs
        assert call_kwargs["padding"] == "longest"

    def test_custom_return_tensors(self, mock_tokenizer: MagicMock) -> None:
        """Test custom return_tensors parameter."""
        tokenize_batch(["hello"], mock_tokenizer, return_tensors="np")

        call_kwargs = mock_tokenizer.call_args.kwargs
        assert call_kwargs["return_tensors"] == "np"


class TestCreatePreprocessingFunction:
    """Tests for create_preprocessing_function."""

    def test_empty_text_column_raises_error(self) -> None:
        """Test that empty text_column raises ValueError."""
        with pytest.raises(ValueError, match="text_column cannot be empty"):
            create_preprocessing_function(MagicMock(), text_column="")

    def test_invalid_max_length_zero(self) -> None:
        """Test that max_length=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_preprocessing_function(MagicMock(), max_length=0)

    def test_invalid_max_length_negative(self) -> None:
        """Test that negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_preprocessing_function(MagicMock(), max_length=-10)

    def test_returns_callable(self) -> None:
        """Test that function returns a callable."""
        fn = create_preprocessing_function(MagicMock())
        assert callable(fn)

    def test_preprocessing_function_tokenizes(self) -> None:
        """Test that returned function tokenizes correctly."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        fn = create_preprocessing_function(mock_tokenizer)
        result = fn({"text": ["hello world"]})

        mock_tokenizer.assert_called_once()
        assert "input_ids" in result

    def test_preprocessing_function_includes_labels(self) -> None:
        """Test that labels are included when present."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        fn = create_preprocessing_function(mock_tokenizer, label_column="label")
        result = fn({"text": ["hello"], "label": [1]})

        assert "labels" in result
        assert result["labels"] == [1]

    def test_preprocessing_function_handles_single_text(self) -> None:
        """Test that single text string is handled."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        fn = create_preprocessing_function(mock_tokenizer)
        fn({"text": "single text"})

        # Should wrap in list
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == ["single text"]

    def test_preprocessing_function_no_labels(self) -> None:
        """Test preprocessing without label column."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        fn = create_preprocessing_function(mock_tokenizer, label_column=None)
        result = fn({"text": ["hello"]})

        assert "labels" not in result

    def test_preprocessing_function_custom_text_column(self) -> None:
        """Test custom text column name."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        fn = create_preprocessing_function(mock_tokenizer, text_column="content")
        fn({"content": ["hello"]})

        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == ["hello"]
