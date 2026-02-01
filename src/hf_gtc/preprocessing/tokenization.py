"""Tokenizer utilities and vocabulary analysis.

This module provides utilities for tokenizer configuration, vocabulary
analysis, fertility calculation, and tokenizer comparison for ML pipelines.

Examples:
    >>> from hf_gtc.preprocessing.tokenization import TokenizerType, SpecialTokenType
    >>> TokenizerType.BPE.value
    'bpe'
    >>> SpecialTokenType.PAD.value
    'pad'
    >>> from hf_gtc.preprocessing.tokenization import create_tokenizer_config
    >>> config = create_tokenizer_config(tokenizer_type="bpe", vocab_size=32000)
    >>> config.vocab_size
    32000
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase

from hf_gtc._validation import validate_not_none


class TokenizerType(Enum):
    """Types of tokenization algorithms.

    Attributes:
        BPE: Byte Pair Encoding tokenizer.
        WORDPIECE: WordPiece tokenizer (BERT-style).
        UNIGRAM: Unigram language model tokenizer.
        SENTENCEPIECE: SentencePiece tokenizer.
        TIKTOKEN: tiktoken tokenizer (GPT-style).

    Examples:
        >>> TokenizerType.BPE.value
        'bpe'
        >>> TokenizerType.WORDPIECE.value
        'wordpiece'
        >>> TokenizerType.UNIGRAM.value
        'unigram'
    """

    BPE = "bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"
    SENTENCEPIECE = "sentencepiece"
    TIKTOKEN = "tiktoken"


VALID_TOKENIZER_TYPES = frozenset(t.value for t in TokenizerType)


class SpecialTokenType(Enum):
    """Types of special tokens used in tokenizers.

    Attributes:
        PAD: Padding token for batching.
        UNK: Unknown token for out-of-vocabulary words.
        BOS: Beginning of sequence token.
        EOS: End of sequence token.
        SEP: Separator token for sentence pairs.
        CLS: Classification token.
        MASK: Mask token for masked language modeling.

    Examples:
        >>> SpecialTokenType.PAD.value
        'pad'
        >>> SpecialTokenType.UNK.value
        'unk'
        >>> SpecialTokenType.MASK.value
        'mask'
    """

    PAD = "pad"
    UNK = "unk"
    BOS = "bos"
    EOS = "eos"
    SEP = "sep"
    CLS = "cls"
    MASK = "mask"


VALID_SPECIAL_TOKEN_TYPES = frozenset(t.value for t in SpecialTokenType)


class VocabAnalysisMetric(Enum):
    """Metrics for vocabulary analysis.

    Attributes:
        COVERAGE: Vocabulary coverage rate on corpus.
        FERTILITY: Average number of tokens per word.
        UNKNOWN_RATE: Rate of unknown tokens.

    Examples:
        >>> VocabAnalysisMetric.COVERAGE.value
        'coverage'
        >>> VocabAnalysisMetric.FERTILITY.value
        'fertility'
        >>> VocabAnalysisMetric.UNKNOWN_RATE.value
        'unknown_rate'
    """

    COVERAGE = "coverage"
    FERTILITY = "fertility"
    UNKNOWN_RATE = "unknown_rate"


VALID_VOCAB_ANALYSIS_METRICS = frozenset(m.value for m in VocabAnalysisMetric)


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Configuration for tokenizer setup.

    Attributes:
        tokenizer_type: Type of tokenization algorithm.
        vocab_size: Size of the vocabulary.
        special_tokens: Mapping of special token types to token strings.

    Examples:
        >>> special = {SpecialTokenType.PAD: "[PAD]", SpecialTokenType.UNK: "[UNK]"}
        >>> config = TokenizerConfig(
        ...     tokenizer_type=TokenizerType.BPE,
        ...     vocab_size=32000,
        ...     special_tokens=special,
        ... )
        >>> config.vocab_size
        32000
        >>> config.tokenizer_type
        <TokenizerType.BPE: 'bpe'>
    """

    tokenizer_type: TokenizerType
    vocab_size: int
    special_tokens: Mapping[SpecialTokenType, str]


@dataclass(frozen=True, slots=True)
class VocabStats:
    """Statistics from vocabulary analysis.

    Attributes:
        vocab_size: Total vocabulary size.
        coverage: Vocabulary coverage rate (0.0-1.0).
        avg_token_length: Average length of tokens in characters.
        unknown_rate: Rate of unknown tokens (0.0-1.0).

    Examples:
        >>> stats = VocabStats(
        ...     vocab_size=32000,
        ...     coverage=0.95,
        ...     avg_token_length=4.5,
        ...     unknown_rate=0.02,
        ... )
        >>> stats.coverage
        0.95
        >>> stats.unknown_rate
        0.02
    """

    vocab_size: int
    coverage: float
    avg_token_length: float
    unknown_rate: float


@dataclass(frozen=True, slots=True)
class TokenizationResult:
    """Result from tokenization operation.

    Attributes:
        tokens: List of token strings.
        token_ids: List of token integer IDs.
        offsets: List of (start, end) character offsets for each token.
        special_mask: List of booleans indicating special tokens.

    Examples:
        >>> result = TokenizationResult(
        ...     tokens=["[CLS]", "hello", "world", "[SEP]"],
        ...     token_ids=[101, 7592, 2088, 102],
        ...     offsets=[(0, 0), (0, 5), (6, 11), (0, 0)],
        ...     special_mask=[True, False, False, True],
        ... )
        >>> result.tokens
        ['[CLS]', 'hello', 'world', '[SEP]']
        >>> result.special_mask[0]
        True
    """

    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    special_mask: tuple[bool, ...]


def validate_tokenizer_config(config: TokenizerConfig) -> None:
    """Validate tokenizer configuration.

    Args:
        config: TokenizerConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If vocab_size is not positive.
        ValueError: If special_tokens is None.

    Examples:
        >>> config = TokenizerConfig(
        ...     tokenizer_type=TokenizerType.BPE,
        ...     vocab_size=32000,
        ...     special_tokens={SpecialTokenType.PAD: "[PAD]"},
        ... )
        >>> validate_tokenizer_config(config)  # No error

        >>> validate_tokenizer_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = TokenizerConfig(
        ...     tokenizer_type=TokenizerType.BPE,
        ...     vocab_size=0,
        ...     special_tokens={},
        ... )
        >>> validate_tokenizer_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    validate_not_none(config, "config")

    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)

    if config.special_tokens is None:
        msg = "special_tokens cannot be None"
        raise ValueError(msg)


def validate_vocab_stats(stats: VocabStats) -> None:
    """Validate vocabulary statistics.

    Args:
        stats: VocabStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If vocab_size is not positive.
        ValueError: If coverage is not in [0, 1].
        ValueError: If avg_token_length is negative.
        ValueError: If unknown_rate is not in [0, 1].

    Examples:
        >>> stats = VocabStats(
        ...     vocab_size=32000,
        ...     coverage=0.95,
        ...     avg_token_length=4.5,
        ...     unknown_rate=0.02,
        ... )
        >>> validate_vocab_stats(stats)  # No error

        >>> validate_vocab_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = VocabStats(
        ...     vocab_size=-1, coverage=0.5, avg_token_length=4.0, unknown_rate=0.1
        ... )
        >>> validate_vocab_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    validate_not_none(stats, "stats")

    if stats.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {stats.vocab_size}"
        raise ValueError(msg)

    if not 0.0 <= stats.coverage <= 1.0:
        msg = f"coverage must be between 0 and 1, got {stats.coverage}"
        raise ValueError(msg)

    if stats.avg_token_length < 0:
        msg = f"avg_token_length cannot be negative, got {stats.avg_token_length}"
        raise ValueError(msg)

    if not 0.0 <= stats.unknown_rate <= 1.0:
        msg = f"unknown_rate must be between 0 and 1, got {stats.unknown_rate}"
        raise ValueError(msg)


def validate_tokenization_result(result: TokenizationResult) -> None:
    """Validate tokenization result.

    Args:
        result: TokenizationResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If lengths of tokens, token_ids, offsets, special_mask don't match.

    Examples:
        >>> result = TokenizationResult(
        ...     tokens=("hello", "world"),
        ...     token_ids=(1, 2),
        ...     offsets=((0, 5), (6, 11)),
        ...     special_mask=(False, False),
        ... )
        >>> validate_tokenization_result(result)  # No error

        >>> validate_tokenization_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = TokenizationResult(
        ...     tokens=("hello",),
        ...     token_ids=(1, 2),
        ...     offsets=((0, 5),),
        ...     special_mask=(False,),
        ... )
        >>> validate_tokenization_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens, token_ids, offsets, and special_mask must have same length
    """
    validate_not_none(result, "result")

    lengths = {
        len(result.tokens),
        len(result.token_ids),
        len(result.offsets),
        len(result.special_mask),
    }
    if len(lengths) > 1:
        msg = "tokens, token_ids, offsets, and special_mask must have same length"
        raise ValueError(msg)


def create_tokenizer_config(
    tokenizer_type: str = "bpe",
    vocab_size: int = 32000,
    special_tokens: dict[str, str] | None = None,
) -> TokenizerConfig:
    """Create a tokenizer configuration.

    Args:
        tokenizer_type: Tokenizer type name. Defaults to "bpe".
        vocab_size: Vocabulary size. Defaults to 32000.
        special_tokens: Mapping of special token type names to token strings.
            Defaults to common special tokens.

    Returns:
        TokenizerConfig with the specified settings.

    Raises:
        ValueError: If tokenizer_type is not valid.
        ValueError: If vocab_size is not positive.
        ValueError: If any special token type is invalid.

    Examples:
        >>> config = create_tokenizer_config(
        ...     tokenizer_type="wordpiece", vocab_size=30522
        ... )
        >>> config.tokenizer_type
        <TokenizerType.WORDPIECE: 'wordpiece'>
        >>> config.vocab_size
        30522

        >>> create_tokenizer_config(tokenizer_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokenizer_type must be one of

        >>> create_tokenizer_config(vocab_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    if tokenizer_type not in VALID_TOKENIZER_TYPES:
        msg = (
            f"tokenizer_type must be one of {VALID_TOKENIZER_TYPES}, "
            f"got '{tokenizer_type}'"
        )
        raise ValueError(msg)

    # Default special tokens
    default_special_tokens = {
        "pad": "[PAD]",
        "unk": "[UNK]",
        "bos": "[BOS]",
        "eos": "[EOS]",
        "sep": "[SEP]",
        "cls": "[CLS]",
        "mask": "[MASK]",
    }

    effective_tokens = special_tokens or default_special_tokens

    # Validate and convert special tokens
    token_mapping: dict[SpecialTokenType, str] = {}
    for key, value in effective_tokens.items():
        if key not in VALID_SPECIAL_TOKEN_TYPES:
            msg = (
                f"special token type must be one of {VALID_SPECIAL_TOKEN_TYPES}, "
                f"got '{key}'"
            )
            raise ValueError(msg)
        token_mapping[SpecialTokenType(key)] = value

    config = TokenizerConfig(
        tokenizer_type=TokenizerType(tokenizer_type),
        vocab_size=vocab_size,
        special_tokens=token_mapping,
    )
    validate_tokenizer_config(config)
    return config


def create_vocab_stats(
    vocab_size: int,
    coverage: float = 1.0,
    avg_token_length: float = 4.0,
    unknown_rate: float = 0.0,
) -> VocabStats:
    """Create vocabulary statistics.

    Args:
        vocab_size: Total vocabulary size.
        coverage: Vocabulary coverage rate. Defaults to 1.0.
        avg_token_length: Average token length. Defaults to 4.0.
        unknown_rate: Unknown token rate. Defaults to 0.0.

    Returns:
        VocabStats with the specified values.

    Raises:
        ValueError: If vocab_size is not positive.
        ValueError: If coverage is not in [0, 1].
        ValueError: If avg_token_length is negative.
        ValueError: If unknown_rate is not in [0, 1].

    Examples:
        >>> stats = create_vocab_stats(vocab_size=32000, coverage=0.95)
        >>> stats.coverage
        0.95

        >>> create_vocab_stats(vocab_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive

        >>> create_vocab_stats(vocab_size=1000, coverage=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: coverage must be between 0 and 1
    """
    stats = VocabStats(
        vocab_size=vocab_size,
        coverage=coverage,
        avg_token_length=avg_token_length,
        unknown_rate=unknown_rate,
    )
    validate_vocab_stats(stats)
    return stats


def list_tokenizer_types() -> list[str]:
    """List all available tokenizer types.

    Returns:
        Sorted list of tokenizer type names.

    Examples:
        >>> types = list_tokenizer_types()
        >>> "bpe" in types
        True
        >>> "wordpiece" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_TOKENIZER_TYPES)


def get_tokenizer_type(name: str) -> TokenizerType:
    """Get TokenizerType enum from string name.

    Args:
        name: Name of the tokenizer type.

    Returns:
        Corresponding TokenizerType enum value.

    Raises:
        ValueError: If name is not a valid tokenizer type.

    Examples:
        >>> get_tokenizer_type("bpe")
        <TokenizerType.BPE: 'bpe'>

        >>> get_tokenizer_type("wordpiece")
        <TokenizerType.WORDPIECE: 'wordpiece'>

        >>> get_tokenizer_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid tokenizer type: invalid
    """
    if name not in VALID_TOKENIZER_TYPES:
        msg = f"invalid tokenizer type: {name}"
        raise ValueError(msg)

    return TokenizerType(name)


def list_special_token_types() -> list[str]:
    """List all available special token types.

    Returns:
        Sorted list of special token type names.

    Examples:
        >>> types = list_special_token_types()
        >>> "pad" in types
        True
        >>> "unk" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_SPECIAL_TOKEN_TYPES)


def get_special_token_type(name: str) -> SpecialTokenType:
    """Get SpecialTokenType enum from string name.

    Args:
        name: Name of the special token type.

    Returns:
        Corresponding SpecialTokenType enum value.

    Raises:
        ValueError: If name is not a valid special token type.

    Examples:
        >>> get_special_token_type("pad")
        <SpecialTokenType.PAD: 'pad'>

        >>> get_special_token_type("mask")
        <SpecialTokenType.MASK: 'mask'>

        >>> get_special_token_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid special token type: invalid
    """
    if name not in VALID_SPECIAL_TOKEN_TYPES:
        msg = f"invalid special token type: {name}"
        raise ValueError(msg)

    return SpecialTokenType(name)


def list_vocab_analysis_metrics() -> list[str]:
    """List all available vocabulary analysis metrics.

    Returns:
        Sorted list of vocabulary analysis metric names.

    Examples:
        >>> metrics = list_vocab_analysis_metrics()
        >>> "coverage" in metrics
        True
        >>> "fertility" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_VOCAB_ANALYSIS_METRICS)


def get_vocab_analysis_metric(name: str) -> VocabAnalysisMetric:
    """Get VocabAnalysisMetric enum from string name.

    Args:
        name: Name of the vocabulary analysis metric.

    Returns:
        Corresponding VocabAnalysisMetric enum value.

    Raises:
        ValueError: If name is not a valid vocabulary analysis metric.

    Examples:
        >>> get_vocab_analysis_metric("coverage")
        <VocabAnalysisMetric.COVERAGE: 'coverage'>

        >>> get_vocab_analysis_metric("fertility")
        <VocabAnalysisMetric.FERTILITY: 'fertility'>

        >>> get_vocab_analysis_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid vocab analysis metric: invalid
    """
    if name not in VALID_VOCAB_ANALYSIS_METRICS:
        msg = f"invalid vocab analysis metric: {name}"
        raise ValueError(msg)

    return VocabAnalysisMetric(name)


def analyze_vocabulary(
    tokens: Sequence[str],
    vocabulary: Sequence[str],
    *,
    unk_token: str = "[UNK]",
) -> VocabStats:
    """Analyze vocabulary coverage and statistics.

    Computes coverage, average token length, and unknown rate for
    a set of tokens against a vocabulary.

    Args:
        tokens: Sequence of tokens to analyze.
        vocabulary: Vocabulary to check against.
        unk_token: Token used for unknowns. Defaults to "[UNK]".

    Returns:
        VocabStats with computed metrics.

    Raises:
        ValueError: If tokens is None.
        ValueError: If vocabulary is None.

    Examples:
        >>> tokens = ["hello", "world", "test", "[UNK]"]
        >>> vocab = ["hello", "world", "test", "foo", "[UNK]"]
        >>> stats = analyze_vocabulary(tokens, vocab)
        >>> stats.vocab_size
        5
        >>> stats.coverage
        1.0

        >>> tokens = ["hello", "[UNK]", "[UNK]"]
        >>> vocab = ["hello", "[UNK]"]
        >>> stats = analyze_vocabulary(tokens, vocab)
        >>> round(stats.unknown_rate, 2)
        0.67

        >>> analyze_vocabulary(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens cannot be None

        >>> analyze_vocabulary([], None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocabulary cannot be None
    """
    if tokens is None:
        msg = "tokens cannot be None"
        raise ValueError(msg)

    if vocabulary is None:
        msg = "vocabulary cannot be None"
        raise ValueError(msg)

    vocab_set = set(vocabulary)
    vocab_size = len(vocab_set)

    if not tokens:
        return VocabStats(
            vocab_size=max(vocab_size, 1),
            coverage=1.0,
            avg_token_length=0.0,
            unknown_rate=0.0,
        )

    # Calculate coverage
    unique_tokens = set(tokens)
    covered = sum(1 for t in unique_tokens if t in vocab_set)
    coverage = covered / len(unique_tokens) if unique_tokens else 1.0

    # Calculate average token length
    total_length = sum(len(t) for t in tokens)
    avg_length = total_length / len(tokens) if tokens else 0.0

    # Calculate unknown rate
    unknown_count = sum(1 for t in tokens if t == unk_token)
    unknown_rate = unknown_count / len(tokens) if tokens else 0.0

    return VocabStats(
        vocab_size=max(vocab_size, 1),
        coverage=coverage,
        avg_token_length=avg_length,
        unknown_rate=unknown_rate,
    )


def calculate_fertility(
    text: str,
    tokens: Sequence[str],
    *,
    word_pattern: str = r"\b\w+\b",
) -> float:
    """Calculate tokenization fertility (tokens per word).

    Fertility measures how many tokens are produced per word on average.
    Lower fertility indicates more efficient tokenization.

    Args:
        text: Original text before tokenization.
        tokens: Tokens produced by tokenizer.
        word_pattern: Regex pattern for word extraction. Defaults to word boundaries.

    Returns:
        Fertility ratio (tokens per word).

    Raises:
        ValueError: If text is None.
        ValueError: If tokens is None.

    Examples:
        >>> text = "hello world"
        >>> tokens = ["hello", "world"]
        >>> calculate_fertility(text, tokens)
        1.0

        >>> text = "unbelievable"
        >>> tokens = ["un", "##believ", "##able"]
        >>> calculate_fertility(text, tokens)
        3.0

        >>> calculate_fertility("", [])
        0.0

        >>> calculate_fertility(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> calculate_fertility("test", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if tokens is None:
        msg = "tokens cannot be None"
        raise ValueError(msg)

    if not text or not tokens:
        return 0.0

    words = re.findall(word_pattern, text)
    if not words:
        return 0.0

    return len(tokens) / len(words)


def estimate_sequence_length(
    text: str,
    *,
    avg_chars_per_token: float = 4.0,
    include_special_tokens: int = 2,
) -> int:
    """Estimate tokenized sequence length from text.

    Provides a rough estimate of the number of tokens that will be
    produced from a given text. Useful for filtering or batching.

    Args:
        text: Text to estimate length for.
        avg_chars_per_token: Average characters per token. Defaults to 4.0.
        include_special_tokens: Number of special tokens to add. Defaults to 2.

    Returns:
        Estimated number of tokens.

    Raises:
        ValueError: If text is None.
        ValueError: If avg_chars_per_token is not positive.
        ValueError: If include_special_tokens is negative.

    Examples:
        >>> estimate_sequence_length("hello world")
        5

        >>> estimate_sequence_length("")
        2

        >>> estimate_sequence_length("test", avg_chars_per_token=2.0)
        4

        >>> estimate_sequence_length(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> estimate_sequence_length("test", avg_chars_per_token=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: avg_chars_per_token must be positive
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if avg_chars_per_token <= 0:
        msg = f"avg_chars_per_token must be positive, got {avg_chars_per_token}"
        raise ValueError(msg)

    if include_special_tokens < 0:
        msg = f"include_special_tokens cannot be negative, got {include_special_tokens}"
        raise ValueError(msg)

    if not text:
        return include_special_tokens

    estimated_tokens = math.ceil(len(text) / avg_chars_per_token)
    return estimated_tokens + include_special_tokens


def compare_tokenizers(
    text: str,
    tokenizations: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Compare multiple tokenizer outputs on the same text.

    Computes metrics for each tokenizer to enable comparison.

    Args:
        text: Original text that was tokenized.
        tokenizations: Mapping of tokenizer names to token sequences.

    Returns:
        Dictionary mapping tokenizer names to metric dictionaries.

    Raises:
        ValueError: If text is None.
        ValueError: If tokenizations is None.
        ValueError: If tokenizations is empty.

    Examples:
        >>> text = "hello world"
        >>> tokenizations = {
        ...     "bpe": ["hello", "world"],
        ...     "wordpiece": ["hello", "##world"],
        ... }
        >>> result = compare_tokenizers(text, tokenizations)
        >>> result["bpe"]["token_count"]
        2.0
        >>> result["wordpiece"]["token_count"]
        2.0

        >>> compare_tokenizers(None, {})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> compare_tokenizers("test", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokenizations cannot be None

        >>> compare_tokenizers("test", {})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokenizations cannot be empty
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if tokenizations is None:
        msg = "tokenizations cannot be None"
        raise ValueError(msg)

    if not tokenizations:
        msg = "tokenizations cannot be empty"
        raise ValueError(msg)

    results: dict[str, dict[str, float]] = {}

    for name, tokens in tokenizations.items():
        token_list = list(tokens)
        fertility = calculate_fertility(text, token_list)
        avg_length = (
            sum(len(t) for t in token_list) / len(token_list) if token_list else 0.0
        )

        results[name] = {
            "token_count": float(len(token_list)),
            "fertility": fertility,
            "avg_token_length": avg_length,
            "compression_ratio": len(text) / len(token_list) if token_list else 0.0,
        }

    return results


def detect_special_tokens(
    tokens: Sequence[str],
    special_patterns: Sequence[str] | None = None,
) -> list[bool]:
    """Detect which tokens are special tokens.

    Args:
        tokens: Sequence of tokens to analyze.
        special_patterns: Patterns for special tokens. Defaults to common patterns.

    Returns:
        List of booleans, True for special tokens.

    Raises:
        ValueError: If tokens is None.

    Examples:
        >>> tokens = ["[CLS]", "hello", "world", "[SEP]"]
        >>> detect_special_tokens(tokens)
        [True, False, False, True]

        >>> tokens = ["<s>", "hello", "</s>"]
        >>> detect_special_tokens(tokens)
        [True, False, True]

        >>> detect_special_tokens([])
        []

        >>> detect_special_tokens(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens cannot be None
    """
    if tokens is None:
        msg = "tokens cannot be None"
        raise ValueError(msg)

    if not tokens:
        return []

    default_patterns = [
        r"^\[.*\]$",  # [CLS], [SEP], [PAD], etc.
        r"^<.*>$",  # <s>, </s>, <pad>, etc.
        r"^<\|.*\|>$",  # <|endoftext|>, etc.
    ]

    patterns = special_patterns or default_patterns
    compiled = [re.compile(p) for p in patterns]

    result: list[bool] = []
    for token in tokens:
        is_special = any(pattern.match(token) for pattern in compiled)
        result.append(is_special)

    return result


def format_vocab_stats(stats: VocabStats) -> str:
    """Format vocabulary statistics as a human-readable string.

    Args:
        stats: VocabStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = VocabStats(
        ...     vocab_size=32000,
        ...     coverage=0.95,
        ...     avg_token_length=4.5,
        ...     unknown_rate=0.02,
        ... )
        >>> formatted = format_vocab_stats(stats)
        >>> "32,000" in formatted
        True
        >>> "95.0%" in formatted
        True

        >>> format_vocab_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        "Vocabulary Statistics",
        "=" * 40,
        f"Vocabulary size:     {stats.vocab_size:,}",
        f"Coverage:            {stats.coverage * 100:.1f}%",
        f"Avg token length:    {stats.avg_token_length:.2f}",
        f"Unknown rate:        {stats.unknown_rate * 100:.2f}%",
    ]

    return "\n".join(lines)


def get_recommended_tokenizer_config(
    task: str = "general",
    model_family: str = "transformer",
) -> TokenizerConfig:
    """Get recommended tokenizer configuration for a task.

    Args:
        task: Task type ("general", "translation", "code", "multilingual").
            Defaults to "general".
        model_family: Model family ("transformer", "bert", "gpt").
            Defaults to "transformer".

    Returns:
        Recommended TokenizerConfig for the task.

    Raises:
        ValueError: If task is not valid.
        ValueError: If model_family is not valid.

    Examples:
        >>> config = get_recommended_tokenizer_config(task="general")
        >>> config.vocab_size >= 30000
        True

        >>> config = get_recommended_tokenizer_config(task="code")
        >>> config.tokenizer_type
        <TokenizerType.BPE: 'bpe'>

        >>> config = get_recommended_tokenizer_config(model_family="bert")
        >>> config.tokenizer_type
        <TokenizerType.WORDPIECE: 'wordpiece'>

        >>> get_recommended_tokenizer_config(task="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of

        >>> get_recommended_tokenizer_config(model_family="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_family must be one of
    """
    valid_tasks = frozenset({"general", "translation", "code", "multilingual"})
    valid_families = frozenset({"transformer", "bert", "gpt"})

    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    if model_family not in valid_families:
        msg = f"model_family must be one of {valid_families}, got '{model_family}'"
        raise ValueError(msg)

    # BERT family uses WordPiece
    if model_family == "bert":
        return create_tokenizer_config(
            tokenizer_type="wordpiece",
            vocab_size=30522,
            special_tokens={
                "pad": "[PAD]",
                "unk": "[UNK]",
                "cls": "[CLS]",
                "sep": "[SEP]",
                "mask": "[MASK]",
            },
        )

    # GPT family uses BPE
    if model_family == "gpt":
        return create_tokenizer_config(
            tokenizer_type="bpe",
            vocab_size=50257,
            special_tokens={
                "pad": "<|endoftext|>",
                "unk": "<|endoftext|>",
                "bos": "<|endoftext|>",
                "eos": "<|endoftext|>",
            },
        )

    # Task-specific recommendations
    if task == "code":
        return create_tokenizer_config(
            tokenizer_type="bpe",
            vocab_size=50000,
            special_tokens={
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<s>",
                "eos": "</s>",
            },
        )

    if task == "multilingual":
        return create_tokenizer_config(
            tokenizer_type="sentencepiece",
            vocab_size=128000,
            special_tokens={
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<s>",
                "eos": "</s>",
            },
        )

    if task == "translation":
        return create_tokenizer_config(
            tokenizer_type="sentencepiece",
            vocab_size=64000,
            special_tokens={
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<s>",
                "eos": "</s>",
            },
        )

    # Default general purpose
    return create_tokenizer_config(
        tokenizer_type="bpe",
        vocab_size=32000,
        special_tokens={
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<s>",
            "eos": "</s>",
            "sep": "</s>",
            "cls": "<s>",
            "mask": "<mask>",
        },
    )


# ============================================================================
# Legacy functions preserved for backward compatibility
# ============================================================================


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    unicode_normalize: bool = True,
) -> str:
    """Preprocess text for model input.

    Applies normalization steps to prepare text for tokenization.
    This function is idempotent: f(f(x)) == f(x).

    Args:
        text: Input text to preprocess.
        lowercase: Whether to convert to lowercase. Defaults to True.
        strip_whitespace: Whether to strip leading/trailing whitespace
            and normalize internal whitespace. Defaults to True.
        unicode_normalize: Whether to apply Unicode NFC normalization.
            This ensures visually identical characters (e.g., precomposed
            vs decomposed accents) produce identical byte sequences.
            Defaults to True.

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
        >>> preprocess_text("  Hella Wourld  ")
        'hella wourld'

        >>> # NFC normalization: precomposed and decomposed produce same output
        >>> import unicodedata
        >>> nfc = "cafe"  # precomposed e-acute
        >>> nfd = "cafe"  # e + combining acute
        >>> preprocess_text(nfc) == preprocess_text(nfd)
        True

        >>> # Disable unicode normalization
        >>> nfc_raw = preprocess_text(nfc, unicode_normalize=False)
        >>> nfd_raw = preprocess_text(nfd, unicode_normalize=False)
        >>> nfc_raw == nfd_raw
        True
    """
    result = text

    # Apply Unicode NFC normalization first to ensure consistent byte representation
    # This prevents tokenizer divergence from visually identical but byte-different
    # strings (e.g., precomposed vs decomposed accents)
    if unicode_normalize:
        result = unicodedata.normalize("NFC", result)

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
