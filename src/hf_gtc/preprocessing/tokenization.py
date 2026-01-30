"""Text preprocessing and tokenization utilities.

This module provides functions for text normalization and
batch tokenization compatible with HuggingFace models.

Examples:
    >>> from hf_gtc.preprocessing.tokenization import preprocess_text
    >>> preprocess_text("  HELLO  ")
    'hello'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """Preprocess text for model input.

    Applies normalization steps to prepare text for tokenization.
    This function is idempotent: f(f(x)) == f(x).

    Args:
        text: Input text to preprocess.
        lowercase: Whether to convert to lowercase. Defaults to True.
        strip_whitespace: Whether to strip leading/trailing whitespace
            and normalize internal whitespace. Defaults to True.

    Returns:
        Preprocessed text string.

    Examples:
        >>> preprocess_text("  Hello   World  ")
        'hello world'

        >>> preprocess_text("UPPERCASE", lowercase=False)
        'UPPERCASE'

        >>> preprocess_text("  spaces  ", strip_whitespace=False)
        '  spaces  '

        >>> # Empty string handling
        >>> preprocess_text("")
        ''

        >>> # Idempotency
        >>> text = "  HELLO   WORLD  "
        >>> preprocess_text(preprocess_text(text)) == preprocess_text(text)
        True

        >>> # Unicode handling
        >>> preprocess_text("  Héllo Wörld  ")
        'héllo wörld'
    """
    result = text

    if strip_whitespace:
        # Normalize internal whitespace and strip
        result = " ".join(result.split())

    if lowercase:
        result = result.lower()

    return result


def tokenize_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 512,
    padding: bool | str = True,
    truncation: bool = True,
    return_tensors: str = "pt",
) -> BatchEncoding:
    """Tokenize a batch of texts.

    Wraps the tokenizer with sensible defaults for batch processing.

    Args:
        texts: List of text strings to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum sequence length. Defaults to 512.
        padding: Padding strategy. True for max length, "longest" for
            batch max. Defaults to True.
        truncation: Whether to truncate to max_length. Defaults to True.
        return_tensors: Return type ("pt" for PyTorch, "np" for NumPy).
            Defaults to "pt".

    Returns:
        BatchEncoding with input_ids, attention_mask, etc.

    Raises:
        ValueError: If texts is empty.
        ValueError: If max_length is not positive.

    Examples:
        >>> tokenize_batch([], None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: texts cannot be empty

        >>> from unittest.mock import MagicMock
        >>> mock_tokenizer = MagicMock()
        >>> mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        >>> result = tokenize_batch(["hello"], mock_tokenizer)
        >>> mock_tokenizer.call_count
        1
    """
    if not texts:
        msg = "texts cannot be empty"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )


def create_preprocessing_function(
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    label_column: str | None = "label",
    *,
    max_length: int = 512,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a preprocessing function for dataset.map().

    Returns a function that can be used with HuggingFace datasets
    to preprocess and tokenize text data.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        text_column: Name of the text column. Defaults to "text".
        label_column: Name of the label column, or None to skip.
            Defaults to "label".
        max_length: Maximum sequence length. Defaults to 512.

    Returns:
        Preprocessing function for use with dataset.map().

    Raises:
        ValueError: If text_column is empty.
        ValueError: If max_length is not positive.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> mock_tok = MagicMock()
        >>> mock_tok.return_value = {"input_ids": [1, 2]}
        >>> fn = create_preprocessing_function(mock_tok)
        >>> callable(fn)
        True

        >>> create_preprocessing_function(None, text_column="")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: text_column cannot be empty

        >>> create_preprocessing_function(None, max_length=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: max_length must be positive...
    """
    if not text_column:
        msg = "text_column cannot be empty"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    def preprocess_fn(examples: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a batch of examples."""
        texts = examples[text_column]

        # Handle both single example and batched
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        result = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        # Include labels if present
        if label_column and label_column in examples:
            result["labels"] = examples[label_column]

        return result

    return preprocess_fn
