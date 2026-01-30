"""Tokenizer training utilities.

This module provides functions for training custom tokenizers
using BPE, WordPiece, and Unigram algorithms.

Examples:
    >>> from hf_gtc.preprocessing.tokenizer_training import create_bpe_config
    >>> config = create_bpe_config(vocab_size=32000)
    >>> config.vocab_size
    32000
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class TokenizerAlgorithm(Enum):
    """Supported tokenizer training algorithms.

    Attributes:
        BPE: Byte-Pair Encoding.
        WORDPIECE: WordPiece algorithm.
        UNIGRAM: Unigram Language Model.
        CHAR: Character-level tokenization.

    Examples:
        >>> TokenizerAlgorithm.BPE.value
        'bpe'
        >>> TokenizerAlgorithm.WORDPIECE.value
        'wordpiece'
    """

    BPE = "bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"
    CHAR = "char"


VALID_ALGORITHMS = frozenset(a.value for a in TokenizerAlgorithm)


class PreTokenizerType(Enum):
    """Supported pre-tokenizer types.

    Attributes:
        WHITESPACE: Split on whitespace.
        BYTE_LEVEL: Byte-level pre-tokenization.
        METASPACE: Metaspace pre-tokenization.
        PUNCTUATION: Split on punctuation.
        DIGITS: Split digits.
        BERT: BERT-style pre-tokenization.

    Examples:
        >>> PreTokenizerType.BYTE_LEVEL.value
        'byte_level'
        >>> PreTokenizerType.BERT.value
        'bert'
    """

    WHITESPACE = "whitespace"
    BYTE_LEVEL = "byte_level"
    METASPACE = "metaspace"
    PUNCTUATION = "punctuation"
    DIGITS = "digits"
    BERT = "bert"


VALID_PRE_TOKENIZERS = frozenset(p.value for p in PreTokenizerType)

# Normalization types
NormalizerType = Literal["nfc", "nfd", "nfkc", "nfkd", "lowercase", "strip"]
VALID_NORMALIZERS = frozenset({"nfc", "nfd", "nfkc", "nfkd", "lowercase", "strip"})


@dataclass(frozen=True, slots=True)
class BPEConfig:
    """Configuration for BPE tokenizer training.

    Attributes:
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for merges.
        show_progress: Whether to show progress bar.
        special_tokens: Tuple of special tokens.
        initial_alphabet: Initial character alphabet.

    Examples:
        >>> config = BPEConfig(
        ...     vocab_size=32000,
        ...     min_frequency=2,
        ...     show_progress=True,
        ...     special_tokens=("<pad>", "<unk>", "<s>", "</s>"),
        ...     initial_alphabet=(),
        ... )
        >>> config.vocab_size
        32000
    """

    vocab_size: int
    min_frequency: int
    show_progress: bool
    special_tokens: tuple[str, ...]
    initial_alphabet: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WordPieceConfig:
    """Configuration for WordPiece tokenizer training.

    Attributes:
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for tokens.
        show_progress: Whether to show progress bar.
        special_tokens: Tuple of special tokens.
        continuing_subword_prefix: Prefix for continuing subwords.

    Examples:
        >>> config = WordPieceConfig(
        ...     vocab_size=30522,
        ...     min_frequency=2,
        ...     show_progress=True,
        ...     special_tokens=("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"),
        ...     continuing_subword_prefix="##",
        ... )
        >>> config.continuing_subword_prefix
        '##'
    """

    vocab_size: int
    min_frequency: int
    show_progress: bool
    special_tokens: tuple[str, ...]
    continuing_subword_prefix: str


@dataclass(frozen=True, slots=True)
class UnigramConfig:
    """Configuration for Unigram tokenizer training.

    Attributes:
        vocab_size: Target vocabulary size.
        shrinking_factor: Factor for vocabulary shrinking.
        special_tokens: Tuple of special tokens.
        max_piece_length: Maximum piece length.
        n_sub_iterations: Number of sub-iterations.

    Examples:
        >>> config = UnigramConfig(
        ...     vocab_size=32000,
        ...     shrinking_factor=0.75,
        ...     special_tokens=("<pad>", "<unk>", "<s>", "</s>"),
        ...     max_piece_length=16,
        ...     n_sub_iterations=2,
        ... )
        >>> config.shrinking_factor
        0.75
    """

    vocab_size: int
    shrinking_factor: float
    special_tokens: tuple[str, ...]
    max_piece_length: int
    n_sub_iterations: int


@dataclass(frozen=True, slots=True)
class PreTokenizerConfig:
    """Configuration for pre-tokenization.

    Attributes:
        pre_tokenizer_type: Type of pre-tokenizer.
        add_prefix_space: Whether to add prefix space.
        trim_offsets: Whether to trim offsets.
        use_regex: Whether to use regex splitting.

    Examples:
        >>> config = PreTokenizerConfig(
        ...     pre_tokenizer_type=PreTokenizerType.BYTE_LEVEL,
        ...     add_prefix_space=True,
        ...     trim_offsets=True,
        ...     use_regex=False,
        ... )
        >>> config.add_prefix_space
        True
    """

    pre_tokenizer_type: PreTokenizerType
    add_prefix_space: bool
    trim_offsets: bool
    use_regex: bool


@dataclass(frozen=True, slots=True)
class TrainingCorpusConfig:
    """Configuration for training corpus.

    Attributes:
        files: Tuple of file paths.
        batch_size: Batch size for training.
        length: Total corpus length (optional).
        encoding: File encoding.

    Examples:
        >>> config = TrainingCorpusConfig(
        ...     files=("train.txt", "valid.txt"),
        ...     batch_size=1000,
        ...     length=None,
        ...     encoding="utf-8",
        ... )
        >>> config.batch_size
        1000
    """

    files: tuple[str, ...]
    batch_size: int
    length: int | None
    encoding: str


def validate_bpe_config(config: BPEConfig) -> None:
    """Validate BPE configuration.

    Args:
        config: BPE configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = BPEConfig(32000, 2, True, ("<unk>",), ())
        >>> validate_bpe_config(config)  # No error

        >>> bad = BPEConfig(0, 2, True, ("<unk>",), ())
        >>> validate_bpe_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)

    if config.min_frequency < 1:
        msg = f"min_frequency must be >= 1, got {config.min_frequency}"
        raise ValueError(msg)


def validate_wordpiece_config(config: WordPieceConfig) -> None:
    """Validate WordPiece configuration.

    Args:
        config: WordPiece configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = WordPieceConfig(30522, 2, True, ("[UNK]",), "##")
        >>> validate_wordpiece_config(config)  # No error

        >>> bad = WordPieceConfig(30522, 2, True, ("[UNK]",), "")
        >>> validate_wordpiece_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: continuing_subword_prefix cannot be empty
    """
    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)

    if config.min_frequency < 1:
        msg = f"min_frequency must be >= 1, got {config.min_frequency}"
        raise ValueError(msg)

    if not config.continuing_subword_prefix:
        msg = "continuing_subword_prefix cannot be empty"
        raise ValueError(msg)


def validate_unigram_config(config: UnigramConfig) -> None:
    """Validate Unigram configuration.

    Args:
        config: Unigram configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = UnigramConfig(32000, 0.75, ("<unk>",), 16, 2)
        >>> validate_unigram_config(config)  # No error

        >>> bad = UnigramConfig(32000, 1.5, ("<unk>",), 16, 2)
        >>> validate_unigram_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: shrinking_factor must be between 0.0 and 1.0
    """
    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)

    if not 0.0 < config.shrinking_factor < 1.0:
        msg = (
            f"shrinking_factor must be between 0.0 and 1.0 (exclusive), "
            f"got {config.shrinking_factor}"
        )
        raise ValueError(msg)

    if config.max_piece_length <= 0:
        msg = f"max_piece_length must be positive, got {config.max_piece_length}"
        raise ValueError(msg)

    if config.n_sub_iterations <= 0:
        msg = f"n_sub_iterations must be positive, got {config.n_sub_iterations}"
        raise ValueError(msg)


def create_bpe_config(
    vocab_size: int = 32000,
    min_frequency: int = 2,
    show_progress: bool = True,
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>"),
    initial_alphabet: tuple[str, ...] = (),
) -> BPEConfig:
    """Create a BPE tokenizer configuration.

    Args:
        vocab_size: Target vocabulary size. Defaults to 32000.
        min_frequency: Minimum merge frequency. Defaults to 2.
        show_progress: Show progress bar. Defaults to True.
        special_tokens: Special tokens. Defaults to common set.
        initial_alphabet: Initial alphabet. Defaults to ().

    Returns:
        BPEConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_bpe_config(vocab_size=50000)
        >>> config.vocab_size
        50000

        >>> create_bpe_config(vocab_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    config = BPEConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
    )
    validate_bpe_config(config)
    return config


def create_wordpiece_config(
    vocab_size: int = 30522,
    min_frequency: int = 2,
    show_progress: bool = True,
    special_tokens: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"),
    continuing_subword_prefix: str = "##",
) -> WordPieceConfig:
    """Create a WordPiece tokenizer configuration.

    Args:
        vocab_size: Target vocabulary size. Defaults to 30522.
        min_frequency: Minimum token frequency. Defaults to 2.
        show_progress: Show progress bar. Defaults to True.
        special_tokens: Special tokens. Defaults to BERT tokens.
        continuing_subword_prefix: Subword prefix. Defaults to "##".

    Returns:
        WordPieceConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_wordpiece_config(vocab_size=50000)
        >>> config.vocab_size
        50000

        >>> create_wordpiece_config(continuing_subword_prefix="")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: continuing_subword_prefix cannot be empty
    """
    config = WordPieceConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=special_tokens,
        continuing_subword_prefix=continuing_subword_prefix,
    )
    validate_wordpiece_config(config)
    return config


def create_unigram_config(
    vocab_size: int = 32000,
    shrinking_factor: float = 0.75,
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>"),
    max_piece_length: int = 16,
    n_sub_iterations: int = 2,
) -> UnigramConfig:
    """Create a Unigram tokenizer configuration.

    Args:
        vocab_size: Target vocabulary size. Defaults to 32000.
        shrinking_factor: Vocabulary shrinking factor. Defaults to 0.75.
        special_tokens: Special tokens. Defaults to common set.
        max_piece_length: Maximum piece length. Defaults to 16.
        n_sub_iterations: Number of sub-iterations. Defaults to 2.

    Returns:
        UnigramConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_unigram_config(vocab_size=50000)
        >>> config.vocab_size
        50000

        >>> create_unigram_config(shrinking_factor=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: shrinking_factor must be between 0.0 and 1.0
    """
    config = UnigramConfig(
        vocab_size=vocab_size,
        shrinking_factor=shrinking_factor,
        special_tokens=special_tokens,
        max_piece_length=max_piece_length,
        n_sub_iterations=n_sub_iterations,
    )
    validate_unigram_config(config)
    return config


def create_pre_tokenizer_config(
    pre_tokenizer_type: str = "byte_level",
    add_prefix_space: bool = True,
    trim_offsets: bool = True,
    use_regex: bool = False,
) -> PreTokenizerConfig:
    """Create a pre-tokenizer configuration.

    Args:
        pre_tokenizer_type: Type of pre-tokenizer. Defaults to "byte_level".
        add_prefix_space: Add prefix space. Defaults to True.
        trim_offsets: Trim offsets. Defaults to True.
        use_regex: Use regex splitting. Defaults to False.

    Returns:
        PreTokenizerConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_pre_tokenizer_config(pre_tokenizer_type="whitespace")
        >>> config.pre_tokenizer_type
        <PreTokenizerType.WHITESPACE: 'whitespace'>

        >>> create_pre_tokenizer_config(pre_tokenizer_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pre_tokenizer_type must be one of
    """
    if pre_tokenizer_type not in VALID_PRE_TOKENIZERS:
        msg = (
            f"pre_tokenizer_type must be one of {VALID_PRE_TOKENIZERS}, "
            f"got '{pre_tokenizer_type}'"
        )
        raise ValueError(msg)

    return PreTokenizerConfig(
        pre_tokenizer_type=PreTokenizerType(pre_tokenizer_type),
        add_prefix_space=add_prefix_space,
        trim_offsets=trim_offsets,
        use_regex=use_regex,
    )


def create_training_corpus_config(
    files: tuple[str, ...],
    batch_size: int = 1000,
    length: int | None = None,
    encoding: str = "utf-8",
) -> TrainingCorpusConfig:
    """Create a training corpus configuration.

    Args:
        files: Tuple of file paths.
        batch_size: Batch size. Defaults to 1000.
        length: Corpus length. Defaults to None.
        encoding: File encoding. Defaults to "utf-8".

    Returns:
        TrainingCorpusConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_training_corpus_config(("train.txt",))
        >>> config.batch_size
        1000

        >>> create_training_corpus_config(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: files cannot be empty
    """
    if not files:
        msg = "files cannot be empty"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if length is not None and length <= 0:
        msg = f"length must be positive if set, got {length}"
        raise ValueError(msg)

    return TrainingCorpusConfig(
        files=files,
        batch_size=batch_size,
        length=length,
        encoding=encoding,
    )


def list_tokenizer_algorithms() -> list[str]:
    """List supported tokenizer algorithms.

    Returns:
        Sorted list of algorithm names.

    Examples:
        >>> algos = list_tokenizer_algorithms()
        >>> "bpe" in algos
        True
        >>> "wordpiece" in algos
        True
        >>> algos == sorted(algos)
        True
    """
    return sorted(VALID_ALGORITHMS)


def list_pre_tokenizers() -> list[str]:
    """List supported pre-tokenizers.

    Returns:
        Sorted list of pre-tokenizer names.

    Examples:
        >>> pretoks = list_pre_tokenizers()
        >>> "byte_level" in pretoks
        True
        >>> "whitespace" in pretoks
        True
        >>> pretoks == sorted(pretoks)
        True
    """
    return sorted(VALID_PRE_TOKENIZERS)


def get_tokenizer_algorithm(name: str) -> TokenizerAlgorithm:
    """Get tokenizer algorithm from name.

    Args:
        name: Algorithm name.

    Returns:
        TokenizerAlgorithm enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_tokenizer_algorithm("bpe")
        <TokenizerAlgorithm.BPE: 'bpe'>

        >>> get_tokenizer_algorithm("wordpiece")
        <TokenizerAlgorithm.WORDPIECE: 'wordpiece'>

        >>> get_tokenizer_algorithm("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: algorithm must be one of
    """
    if name not in VALID_ALGORITHMS:
        msg = f"algorithm must be one of {VALID_ALGORITHMS}, got '{name}'"
        raise ValueError(msg)
    return TokenizerAlgorithm(name)


def get_recommended_algorithm(model_type: str) -> TokenizerAlgorithm:
    """Get recommended tokenizer algorithm for a model type.

    Args:
        model_type: Model type ("gpt", "bert", "t5", "llama").

    Returns:
        Recommended TokenizerAlgorithm.

    Raises:
        ValueError: If model_type is invalid.

    Examples:
        >>> get_recommended_algorithm("gpt")
        <TokenizerAlgorithm.BPE: 'bpe'>

        >>> get_recommended_algorithm("bert")
        <TokenizerAlgorithm.WORDPIECE: 'wordpiece'>

        >>> get_recommended_algorithm("t5")
        <TokenizerAlgorithm.UNIGRAM: 'unigram'>
    """
    valid_types = {"gpt", "bert", "t5", "llama"}
    if model_type not in valid_types:
        msg = f"model_type must be one of {valid_types}, got '{model_type}'"
        raise ValueError(msg)

    recommendations = {
        "gpt": TokenizerAlgorithm.BPE,
        "bert": TokenizerAlgorithm.WORDPIECE,
        "t5": TokenizerAlgorithm.UNIGRAM,
        "llama": TokenizerAlgorithm.BPE,
    }
    return recommendations[model_type]


def estimate_vocab_coverage(
    vocab_size: int,
    corpus_size: int,
    algorithm: TokenizerAlgorithm,
) -> float:
    """Estimate vocabulary coverage percentage.

    Args:
        vocab_size: Vocabulary size.
        corpus_size: Corpus size (number of tokens).
        algorithm: Tokenizer algorithm.

    Returns:
        Estimated coverage as a percentage (0-100).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> coverage = estimate_vocab_coverage(32000, 1000000, TokenizerAlgorithm.BPE)
        >>> 0 <= coverage <= 100
        True

        >>> estimate_vocab_coverage(0, 1000000, TokenizerAlgorithm.BPE)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    if vocab_size <= 0:
        msg = f"vocab_size must be positive, got {vocab_size}"
        raise ValueError(msg)

    if corpus_size <= 0:
        msg = f"corpus_size must be positive, got {corpus_size}"
        raise ValueError(msg)

    # Rough coverage estimation based on Zipf's law
    # Higher vocab sizes yield diminishing returns
    base_coverage = min(95.0, 50.0 + 45.0 * (1 - 1 / (1 + vocab_size / 10000)))

    # Algorithm adjustments
    algo_factors = {
        TokenizerAlgorithm.BPE: 1.0,
        TokenizerAlgorithm.WORDPIECE: 0.98,
        TokenizerAlgorithm.UNIGRAM: 1.02,
        TokenizerAlgorithm.CHAR: 0.90,
    }

    return min(99.9, base_coverage * algo_factors[algorithm])


def calculate_compression_ratio(
    original_length: int,
    tokenized_length: int,
) -> float:
    """Calculate tokenization compression ratio.

    Args:
        original_length: Original text length (characters).
        tokenized_length: Number of tokens.

    Returns:
        Compression ratio (characters per token).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_compression_ratio(1000, 250)
        4.0

        >>> calculate_compression_ratio(500, 250)
        2.0

        >>> calculate_compression_ratio(1000, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokenized_length must be positive
    """
    if original_length <= 0:
        msg = f"original_length must be positive, got {original_length}"
        raise ValueError(msg)

    if tokenized_length <= 0:
        msg = f"tokenized_length must be positive, got {tokenized_length}"
        raise ValueError(msg)

    return original_length / tokenized_length
