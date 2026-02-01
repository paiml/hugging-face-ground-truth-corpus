"""Vocabulary and BPE training utilities.

This module provides utilities for vocabulary management, BPE training,
and vocabulary statistics for tokenizer development.

Examples:
    >>> from hf_gtc.preprocessing.vocabulary import VocabTrainingMethod
    >>> VocabTrainingMethod.BPE.value
    'bpe'
    >>> from hf_gtc.preprocessing.vocabulary import create_vocab_training_config
    >>> config = create_vocab_training_config(vocab_size=32000)
    >>> config.vocab_size
    32000
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from hf_gtc._validation import validate_not_none


class VocabTrainingMethod(Enum):
    """Vocabulary training methods.

    Attributes:
        BPE: Byte-Pair Encoding.
        WORDPIECE: WordPiece algorithm.
        UNIGRAM: Unigram Language Model.
        SENTENCEPIECE: SentencePiece training.

    Examples:
        >>> VocabTrainingMethod.BPE.value
        'bpe'
        >>> VocabTrainingMethod.WORDPIECE.value
        'wordpiece'
        >>> VocabTrainingMethod.UNIGRAM.value
        'unigram'
        >>> VocabTrainingMethod.SENTENCEPIECE.value
        'sentencepiece'
    """

    BPE = "bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"
    SENTENCEPIECE = "sentencepiece"


VALID_VOCAB_TRAINING_METHODS = frozenset(m.value for m in VocabTrainingMethod)


class MergeStrategy(Enum):
    """Merge strategies for vocabulary training.

    Attributes:
        FREQUENCY: Merge based on frequency.
        PMI: Merge based on pointwise mutual information.
        BPE_DROPOUT: BPE with dropout regularization.

    Examples:
        >>> MergeStrategy.FREQUENCY.value
        'frequency'
        >>> MergeStrategy.PMI.value
        'pmi'
        >>> MergeStrategy.BPE_DROPOUT.value
        'bpe_dropout'
    """

    FREQUENCY = "frequency"
    PMI = "pmi"
    BPE_DROPOUT = "bpe_dropout"


VALID_MERGE_STRATEGIES = frozenset(s.value for s in MergeStrategy)


class SpecialTokenPosition(Enum):
    """Position for special tokens in vocabulary.

    Attributes:
        START: Add special tokens at the start.
        END: Add special tokens at the end.
        BOTH: Add special tokens at both start and end.

    Examples:
        >>> SpecialTokenPosition.START.value
        'start'
        >>> SpecialTokenPosition.END.value
        'end'
        >>> SpecialTokenPosition.BOTH.value
        'both'
    """

    START = "start"
    END = "end"
    BOTH = "both"


VALID_SPECIAL_TOKEN_POSITIONS = frozenset(p.value for p in SpecialTokenPosition)


@dataclass(frozen=True, slots=True)
class VocabTrainingConfig:
    """Configuration for vocabulary training.

    Attributes:
        method: Vocabulary training method.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for tokens.
        special_tokens: Tuple of special tokens.

    Examples:
        >>> config = VocabTrainingConfig(
        ...     method=VocabTrainingMethod.BPE,
        ...     vocab_size=32000,
        ...     min_frequency=2,
        ...     special_tokens=("<pad>", "<unk>", "<s>", "</s>"),
        ... )
        >>> config.vocab_size
        32000
        >>> config.method
        <VocabTrainingMethod.BPE: 'bpe'>
    """

    method: VocabTrainingMethod
    vocab_size: int
    min_frequency: int
    special_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Configuration for merge operations.

    Attributes:
        strategy: Merge strategy to use.
        num_merges: Number of merges to perform.
        dropout_rate: Dropout rate for BPE dropout (0.0-1.0).

    Examples:
        >>> config = MergeConfig(
        ...     strategy=MergeStrategy.FREQUENCY,
        ...     num_merges=10000,
        ...     dropout_rate=0.0,
        ... )
        >>> config.num_merges
        10000
        >>> config.strategy
        <MergeStrategy.FREQUENCY: 'frequency'>
    """

    strategy: MergeStrategy
    num_merges: int
    dropout_rate: float


@dataclass(frozen=True, slots=True)
class SpecialTokensConfig:
    """Configuration for special tokens.

    Attributes:
        pad: Padding token.
        unk: Unknown token.
        bos: Beginning of sequence token.
        eos: End of sequence token.
        mask: Mask token (for MLM).
        additional: Tuple of additional special tokens.

    Examples:
        >>> config = SpecialTokensConfig(
        ...     pad="<pad>",
        ...     unk="<unk>",
        ...     bos="<s>",
        ...     eos="</s>",
        ...     mask="<mask>",
        ...     additional=("<sep>", "<cls>"),
        ... )
        >>> config.pad
        '<pad>'
        >>> config.unk
        '<unk>'
        >>> len(config.additional)
        2
    """

    pad: str
    unk: str
    bos: str
    eos: str
    mask: str
    additional: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VocabTrainingStats:
    """Statistics from vocabulary training.

    Attributes:
        vocab_size: Final vocabulary size.
        num_merges: Number of merges performed.
        coverage: Vocabulary coverage percentage (0-100).
        avg_token_length: Average token length in characters.

    Examples:
        >>> stats = VocabTrainingStats(
        ...     vocab_size=32000,
        ...     num_merges=31744,
        ...     coverage=98.5,
        ...     avg_token_length=4.2,
        ... )
        >>> stats.vocab_size
        32000
        >>> stats.coverage
        98.5
    """

    vocab_size: int
    num_merges: int
    coverage: float
    avg_token_length: float


def validate_vocab_training_config(config: VocabTrainingConfig) -> None:
    """Validate vocabulary training configuration.

    Args:
        config: VocabTrainingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If vocab_size is not positive.
        ValueError: If min_frequency is less than 1.

    Examples:
        >>> config = VocabTrainingConfig(
        ...     method=VocabTrainingMethod.BPE,
        ...     vocab_size=32000,
        ...     min_frequency=2,
        ...     special_tokens=("<unk>",),
        ... )
        >>> validate_vocab_training_config(config)  # No error

        >>> validate_vocab_training_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = VocabTrainingConfig(
        ...     method=VocabTrainingMethod.BPE,
        ...     vocab_size=0,
        ...     min_frequency=2,
        ...     special_tokens=(),
        ... )
        >>> validate_vocab_training_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    validate_not_none(config, "config")

    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)

    if config.min_frequency < 1:
        msg = f"min_frequency must be >= 1, got {config.min_frequency}"
        raise ValueError(msg)


def validate_merge_config(config: MergeConfig) -> None:
    """Validate merge configuration.

    Args:
        config: MergeConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_merges is not positive.
        ValueError: If dropout_rate is not in [0, 1].

    Examples:
        >>> config = MergeConfig(
        ...     strategy=MergeStrategy.FREQUENCY,
        ...     num_merges=10000,
        ...     dropout_rate=0.1,
        ... )
        >>> validate_merge_config(config)  # No error

        >>> validate_merge_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = MergeConfig(
        ...     strategy=MergeStrategy.FREQUENCY,
        ...     num_merges=0,
        ...     dropout_rate=0.0,
        ... )
        >>> validate_merge_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_merges must be positive
    """
    validate_not_none(config, "config")

    if config.num_merges <= 0:
        msg = f"num_merges must be positive, got {config.num_merges}"
        raise ValueError(msg)

    if not 0.0 <= config.dropout_rate <= 1.0:
        msg = f"dropout_rate must be between 0 and 1, got {config.dropout_rate}"
        raise ValueError(msg)


def validate_special_tokens_config(config: SpecialTokensConfig) -> None:
    """Validate special tokens configuration.

    Args:
        config: SpecialTokensConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If any required token is empty.

    Examples:
        >>> config = SpecialTokensConfig(
        ...     pad="<pad>",
        ...     unk="<unk>",
        ...     bos="<s>",
        ...     eos="</s>",
        ...     mask="<mask>",
        ...     additional=(),
        ... )
        >>> validate_special_tokens_config(config)  # No error

        >>> validate_special_tokens_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SpecialTokensConfig(
        ...     pad="",
        ...     unk="<unk>",
        ...     bos="<s>",
        ...     eos="</s>",
        ...     mask="<mask>",
        ...     additional=(),
        ... )
        >>> validate_special_tokens_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pad token cannot be empty
    """
    validate_not_none(config, "config")

    if not config.pad:
        msg = "pad token cannot be empty"
        raise ValueError(msg)

    if not config.unk:
        msg = "unk token cannot be empty"
        raise ValueError(msg)

    if not config.bos:
        msg = "bos token cannot be empty"
        raise ValueError(msg)

    if not config.eos:
        msg = "eos token cannot be empty"
        raise ValueError(msg)

    if not config.mask:
        msg = "mask token cannot be empty"
        raise ValueError(msg)


def validate_vocab_training_stats(stats: VocabTrainingStats) -> None:
    """Validate vocabulary training statistics.

    Args:
        stats: VocabTrainingStats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If vocab_size is not positive.
        ValueError: If coverage is not in [0, 100].

    Examples:
        >>> stats = VocabTrainingStats(
        ...     vocab_size=32000,
        ...     num_merges=31744,
        ...     coverage=98.5,
        ...     avg_token_length=4.2,
        ... )
        >>> validate_vocab_training_stats(stats)  # No error

        >>> validate_vocab_training_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = VocabTrainingStats(
        ...     vocab_size=32000,
        ...     num_merges=31744,
        ...     coverage=150.0,
        ...     avg_token_length=4.2,
        ... )
        >>> validate_vocab_training_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: coverage must be between 0 and 100
    """
    validate_not_none(stats, "stats")

    if stats.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {stats.vocab_size}"
        raise ValueError(msg)

    if stats.num_merges < 0:
        msg = f"num_merges cannot be negative, got {stats.num_merges}"
        raise ValueError(msg)

    if not 0.0 <= stats.coverage <= 100.0:
        msg = f"coverage must be between 0 and 100, got {stats.coverage}"
        raise ValueError(msg)

    if stats.avg_token_length <= 0:
        msg = f"avg_token_length must be positive, got {stats.avg_token_length}"
        raise ValueError(msg)


def create_vocab_training_config(
    method: str = "bpe",
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>"),
) -> VocabTrainingConfig:
    """Create a vocabulary training configuration.

    Args:
        method: Training method name. Defaults to "bpe".
        vocab_size: Target vocabulary size. Defaults to 32000.
        min_frequency: Minimum token frequency. Defaults to 2.
        special_tokens: Special tokens to include. Defaults to common set.

    Returns:
        VocabTrainingConfig with the specified settings.

    Raises:
        ValueError: If method is not valid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_vocab_training_config(vocab_size=50000)
        >>> config.vocab_size
        50000

        >>> config = create_vocab_training_config(method="wordpiece")
        >>> config.method
        <VocabTrainingMethod.WORDPIECE: 'wordpiece'>

        >>> create_vocab_training_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of

        >>> create_vocab_training_config(vocab_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    if method not in VALID_VOCAB_TRAINING_METHODS:
        msg = f"method must be one of {VALID_VOCAB_TRAINING_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = VocabTrainingConfig(
        method=VocabTrainingMethod(method),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    validate_vocab_training_config(config)
    return config


def create_merge_config(
    strategy: str = "frequency",
    num_merges: int = 10000,
    dropout_rate: float = 0.0,
) -> MergeConfig:
    """Create a merge configuration.

    Args:
        strategy: Merge strategy name. Defaults to "frequency".
        num_merges: Number of merges. Defaults to 10000.
        dropout_rate: Dropout rate for regularization. Defaults to 0.0.

    Returns:
        MergeConfig with the specified settings.

    Raises:
        ValueError: If strategy is not valid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_merge_config(num_merges=20000)
        >>> config.num_merges
        20000

        >>> config = create_merge_config(strategy="pmi")
        >>> config.strategy
        <MergeStrategy.PMI: 'pmi'>

        >>> create_merge_config(strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of

        >>> create_merge_config(num_merges=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_merges must be positive
    """
    if strategy not in VALID_MERGE_STRATEGIES:
        msg = f"strategy must be one of {VALID_MERGE_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = MergeConfig(
        strategy=MergeStrategy(strategy),
        num_merges=num_merges,
        dropout_rate=dropout_rate,
    )
    validate_merge_config(config)
    return config


def create_special_tokens_config(
    pad: str = "<pad>",
    unk: str = "<unk>",
    bos: str = "<s>",
    eos: str = "</s>",
    mask: str = "<mask>",
    additional: tuple[str, ...] = (),
) -> SpecialTokensConfig:
    """Create a special tokens configuration.

    Args:
        pad: Padding token. Defaults to "<pad>".
        unk: Unknown token. Defaults to "<unk>".
        bos: Beginning of sequence token. Defaults to "<s>".
        eos: End of sequence token. Defaults to "</s>".
        mask: Mask token. Defaults to "<mask>".
        additional: Additional special tokens. Defaults to ().

    Returns:
        SpecialTokensConfig with the specified settings.

    Raises:
        ValueError: If any required token is empty.

    Examples:
        >>> config = create_special_tokens_config()
        >>> config.pad
        '<pad>'
        >>> config.unk
        '<unk>'

        >>> config = create_special_tokens_config(
        ...     pad="[PAD]",
        ...     unk="[UNK]",
        ...     bos="[CLS]",
        ...     eos="[SEP]",
        ...     mask="[MASK]",
        ... )
        >>> config.pad
        '[PAD]'

        >>> create_special_tokens_config(pad="")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pad token cannot be empty
    """
    config = SpecialTokensConfig(
        pad=pad,
        unk=unk,
        bos=bos,
        eos=eos,
        mask=mask,
        additional=additional,
    )
    validate_special_tokens_config(config)
    return config


def list_vocab_training_methods() -> list[str]:
    """List all available vocabulary training methods.

    Returns:
        Sorted list of training method names.

    Examples:
        >>> methods = list_vocab_training_methods()
        >>> "bpe" in methods
        True
        >>> "wordpiece" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_VOCAB_TRAINING_METHODS)


def get_vocab_training_method(name: str) -> VocabTrainingMethod:
    """Get VocabTrainingMethod enum from string name.

    Args:
        name: Name of the training method.

    Returns:
        Corresponding VocabTrainingMethod enum value.

    Raises:
        ValueError: If name is not a valid training method.

    Examples:
        >>> get_vocab_training_method("bpe")
        <VocabTrainingMethod.BPE: 'bpe'>

        >>> get_vocab_training_method("wordpiece")
        <VocabTrainingMethod.WORDPIECE: 'wordpiece'>

        >>> get_vocab_training_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid vocab training method: invalid
    """
    if name not in VALID_VOCAB_TRAINING_METHODS:
        msg = f"invalid vocab training method: {name}"
        raise ValueError(msg)

    return VocabTrainingMethod(name)


def list_merge_strategies() -> list[str]:
    """List all available merge strategies.

    Returns:
        Sorted list of merge strategy names.

    Examples:
        >>> strategies = list_merge_strategies()
        >>> "frequency" in strategies
        True
        >>> "pmi" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_MERGE_STRATEGIES)


def get_merge_strategy(name: str) -> MergeStrategy:
    """Get MergeStrategy enum from string name.

    Args:
        name: Name of the merge strategy.

    Returns:
        Corresponding MergeStrategy enum value.

    Raises:
        ValueError: If name is not a valid merge strategy.

    Examples:
        >>> get_merge_strategy("frequency")
        <MergeStrategy.FREQUENCY: 'frequency'>

        >>> get_merge_strategy("pmi")
        <MergeStrategy.PMI: 'pmi'>

        >>> get_merge_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid merge strategy: invalid
    """
    if name not in VALID_MERGE_STRATEGIES:
        msg = f"invalid merge strategy: {name}"
        raise ValueError(msg)

    return MergeStrategy(name)


def list_special_token_positions() -> list[str]:
    """List all available special token positions.

    Returns:
        Sorted list of special token position names.

    Examples:
        >>> positions = list_special_token_positions()
        >>> "start" in positions
        True
        >>> "end" in positions
        True
        >>> positions == sorted(positions)
        True
    """
    return sorted(VALID_SPECIAL_TOKEN_POSITIONS)


def get_special_token_position(name: str) -> SpecialTokenPosition:
    """Get SpecialTokenPosition enum from string name.

    Args:
        name: Name of the special token position.

    Returns:
        Corresponding SpecialTokenPosition enum value.

    Raises:
        ValueError: If name is not a valid position.

    Examples:
        >>> get_special_token_position("start")
        <SpecialTokenPosition.START: 'start'>

        >>> get_special_token_position("end")
        <SpecialTokenPosition.END: 'end'>

        >>> get_special_token_position("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid special token position: invalid
    """
    if name not in VALID_SPECIAL_TOKEN_POSITIONS:
        msg = f"invalid special token position: {name}"
        raise ValueError(msg)

    return SpecialTokenPosition(name)


def calculate_vocab_coverage(
    vocab: Sequence[str],
    corpus: Sequence[str],
) -> float:
    """Calculate vocabulary coverage over a corpus.

    Args:
        vocab: Sequence of vocabulary tokens.
        corpus: Sequence of texts to check coverage.

    Returns:
        Coverage percentage (0-100).

    Raises:
        ValueError: If vocab is None.
        ValueError: If corpus is None.

    Examples:
        >>> vocab = ["hello", "world", "the", "a"]
        >>> corpus = ["hello world", "the world"]
        >>> coverage = calculate_vocab_coverage(vocab, corpus)
        >>> coverage > 0
        True

        >>> calculate_vocab_coverage(["a", "b"], [])
        0.0

        >>> calculate_vocab_coverage(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab cannot be None

        >>> calculate_vocab_coverage([], None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: corpus cannot be None
    """
    if vocab is None:
        msg = "vocab cannot be None"
        raise ValueError(msg)

    if corpus is None:
        msg = "corpus cannot be None"
        raise ValueError(msg)

    if not corpus:
        return 0.0

    if not vocab:
        return 0.0

    vocab_set = set(vocab)

    # Tokenize corpus by whitespace
    total_tokens = 0
    covered_tokens = 0

    for text in corpus:
        words = text.lower().split()
        total_tokens += len(words)
        covered_tokens += sum(1 for word in words if word in vocab_set)

    if total_tokens == 0:
        return 0.0

    return (covered_tokens / total_tokens) * 100.0


def estimate_compression_ratio(
    vocab_size: int,
    avg_word_length: float = 5.0,
) -> float:
    """Estimate compression ratio for a given vocabulary size.

    Args:
        vocab_size: Target vocabulary size.
        avg_word_length: Average word length in characters. Defaults to 5.0.

    Returns:
        Estimated compression ratio (characters per token).

    Raises:
        ValueError: If vocab_size is not positive.
        ValueError: If avg_word_length is not positive.

    Examples:
        >>> ratio = estimate_compression_ratio(32000)
        >>> ratio > 0
        True

        >>> ratio_small = estimate_compression_ratio(1000)
        >>> ratio_large = estimate_compression_ratio(100000)
        >>> ratio_small < ratio_large
        True

        >>> estimate_compression_ratio(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive

        >>> estimate_compression_ratio(32000, avg_word_length=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: avg_word_length must be positive
    """
    if vocab_size <= 0:
        msg = f"vocab_size must be positive, got {vocab_size}"
        raise ValueError(msg)

    if avg_word_length <= 0:
        msg = f"avg_word_length must be positive, got {avg_word_length}"
        raise ValueError(msg)

    # Estimate based on vocabulary size scaling
    # Larger vocabularies yield higher compression ratios
    # Use log10(vocab_size + 1) to ensure positive results even for vocab_size=1
    base_ratio = avg_word_length * 0.6
    scale_factor = math.log10(vocab_size + 1) / 4.0  # log10(10001) ~= 4

    # Ensure a minimum ratio of 1.0 (at least 1 char per token)
    return max(1.0, base_ratio * scale_factor)


def validate_special_tokens(
    tokens: Sequence[str],
    vocab: Sequence[str],
) -> tuple[bool, list[str]]:
    """Validate that special tokens are in vocabulary.

    Args:
        tokens: Sequence of special tokens to validate.
        vocab: Vocabulary to check against.

    Returns:
        Tuple of (all_valid, missing_tokens).

    Raises:
        ValueError: If tokens is None.
        ValueError: If vocab is None.

    Examples:
        >>> vocab = ["<pad>", "<unk>", "hello", "world"]
        >>> tokens = ["<pad>", "<unk>"]
        >>> valid, missing = validate_special_tokens(tokens, vocab)
        >>> valid
        True
        >>> missing
        []

        >>> tokens = ["<pad>", "<mask>"]
        >>> valid, missing = validate_special_tokens(tokens, vocab)
        >>> valid
        False
        >>> "<mask>" in missing
        True

        >>> validate_special_tokens(None, [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens cannot be None

        >>> validate_special_tokens([], None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab cannot be None
    """
    if tokens is None:
        msg = "tokens cannot be None"
        raise ValueError(msg)

    if vocab is None:
        msg = "vocab cannot be None"
        raise ValueError(msg)

    vocab_set = set(vocab)
    missing = [token for token in tokens if token not in vocab_set]

    return len(missing) == 0, missing


def calculate_merge_score(
    pair_frequency: int,
    left_frequency: int,
    right_frequency: int,
    total_pairs: int,
    strategy: MergeStrategy = MergeStrategy.FREQUENCY,
) -> float:
    """Calculate merge score for a token pair.

    Args:
        pair_frequency: Frequency of the pair.
        left_frequency: Frequency of the left token.
        right_frequency: Frequency of the right token.
        total_pairs: Total number of pairs.
        strategy: Merge strategy to use. Defaults to FREQUENCY.

    Returns:
        Merge score (higher is better).

    Raises:
        ValueError: If any frequency is negative.
        ValueError: If total_pairs is not positive.

    Examples:
        >>> score = calculate_merge_score(100, 500, 400, 10000)
        >>> score > 0
        True

        >>> freq_score = calculate_merge_score(
        ...     100, 500, 400, 10000, MergeStrategy.FREQUENCY
        ... )
        >>> pmi_score = calculate_merge_score(
        ...     100, 500, 400, 10000, MergeStrategy.PMI
        ... )
        >>> freq_score != pmi_score
        True

        >>> calculate_merge_score(-1, 500, 400, 10000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pair_frequency cannot be negative

        >>> calculate_merge_score(100, 500, 400, 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_pairs must be positive
    """
    if pair_frequency < 0:
        msg = f"pair_frequency cannot be negative, got {pair_frequency}"
        raise ValueError(msg)

    if left_frequency < 0:
        msg = f"left_frequency cannot be negative, got {left_frequency}"
        raise ValueError(msg)

    if right_frequency < 0:
        msg = f"right_frequency cannot be negative, got {right_frequency}"
        raise ValueError(msg)

    if total_pairs <= 0:
        msg = f"total_pairs must be positive, got {total_pairs}"
        raise ValueError(msg)

    if strategy == MergeStrategy.FREQUENCY:
        return float(pair_frequency)

    if strategy == MergeStrategy.PMI:
        # Pointwise Mutual Information
        if pair_frequency == 0 or left_frequency == 0 or right_frequency == 0:
            return 0.0

        p_pair = pair_frequency / total_pairs
        p_left = left_frequency / total_pairs
        p_right = right_frequency / total_pairs

        return math.log2(p_pair / (p_left * p_right))

    if strategy == MergeStrategy.BPE_DROPOUT:
        # For BPE dropout, use frequency with some noise
        # In actual implementation, dropout is applied during training
        return float(pair_frequency)

    return float(pair_frequency)


def format_vocab_training_stats(stats: VocabTrainingStats) -> str:
    """Format vocabulary training statistics as a human-readable string.

    Args:
        stats: VocabTrainingStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = VocabTrainingStats(
        ...     vocab_size=32000,
        ...     num_merges=31744,
        ...     coverage=98.5,
        ...     avg_token_length=4.2,
        ... )
        >>> formatted = format_vocab_training_stats(stats)
        >>> "32,000" in formatted or "32000" in formatted
        True
        >>> "98.5" in formatted
        True

        >>> format_vocab_training_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        "Vocabulary Statistics",
        "=" * 40,
        f"Vocabulary size:      {stats.vocab_size:,}",
        f"Number of merges:     {stats.num_merges:,}",
        f"Coverage:             {stats.coverage:.1f}%",
        f"Avg token length:     {stats.avg_token_length:.2f} chars",
    ]

    return "\n".join(lines)


def get_recommended_vocab_config(
    model_type: str,
    corpus_size: int,
) -> VocabTrainingConfig:
    """Get recommended vocabulary configuration for a model type.

    Args:
        model_type: Type of model ("gpt", "bert", "t5", "llama").
        corpus_size: Size of the training corpus.

    Returns:
        Recommended VocabTrainingConfig.

    Raises:
        ValueError: If model_type is invalid.
        ValueError: If corpus_size is not positive.

    Examples:
        >>> config = get_recommended_vocab_config("gpt", 1000000)
        >>> config.method
        <VocabTrainingMethod.BPE: 'bpe'>

        >>> config = get_recommended_vocab_config("bert", 1000000)
        >>> config.method
        <VocabTrainingMethod.WORDPIECE: 'wordpiece'>

        >>> config = get_recommended_vocab_config("t5", 1000000)
        >>> config.method
        <VocabTrainingMethod.UNIGRAM: 'unigram'>

        >>> get_recommended_vocab_config("invalid", 1000000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of

        >>> get_recommended_vocab_config("gpt", 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: corpus_size must be positive
    """
    valid_types = {"gpt", "bert", "t5", "llama"}
    if model_type not in valid_types:
        msg = f"model_type must be one of {valid_types}, got '{model_type}'"
        raise ValueError(msg)

    if corpus_size <= 0:
        msg = f"corpus_size must be positive, got {corpus_size}"
        raise ValueError(msg)

    # Method recommendations
    method_map = {
        "gpt": VocabTrainingMethod.BPE,
        "bert": VocabTrainingMethod.WORDPIECE,
        "t5": VocabTrainingMethod.UNIGRAM,
        "llama": VocabTrainingMethod.BPE,
    }

    # Special token recommendations
    special_tokens_map = {
        "gpt": ("<|endoftext|>",),
        "bert": ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"),
        "t5": ("<pad>", "</s>", "<unk>"),
        "llama": ("<s>", "</s>", "<unk>"),
    }

    # Vocab size recommendations based on corpus size
    if corpus_size < 100000:
        vocab_size = 8000
    elif corpus_size < 1000000:
        vocab_size = 16000
    elif corpus_size < 10000000:
        vocab_size = 32000
    else:
        vocab_size = 50000

    return VocabTrainingConfig(
        method=method_map[model_type],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens_map[model_type],
    )
