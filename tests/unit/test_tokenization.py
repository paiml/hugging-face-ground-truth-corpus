"""Tests for tokenization module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.tokenization import (
    VALID_SPECIAL_TOKEN_TYPES,
    VALID_TOKENIZER_TYPES,
    VALID_VOCAB_ANALYSIS_METRICS,
    SpecialTokenType,
    TokenizationResult,
    TokenizerConfig,
    TokenizerType,
    VocabAnalysisMetric,
    VocabStats,
    analyze_vocabulary,
    calculate_fertility,
    compare_tokenizers,
    create_preprocessing_function,
    create_tokenizer_config,
    create_vocab_stats,
    detect_special_tokens,
    estimate_sequence_length,
    format_vocab_stats,
    get_recommended_tokenizer_config,
    get_special_token_type,
    get_tokenizer_type,
    get_vocab_analysis_metric,
    list_special_token_types,
    list_tokenizer_types,
    list_vocab_analysis_metrics,
    preprocess_text,
    tokenize_batch,
    validate_tokenization_result,
    validate_tokenizer_config,
    validate_vocab_stats,
)


class TestTokenizerType:
    """Tests for TokenizerType enum."""

    def test_bpe_value(self) -> None:
        """Test BPE value."""
        assert TokenizerType.BPE.value == "bpe"

    def test_wordpiece_value(self) -> None:
        """Test WORDPIECE value."""
        assert TokenizerType.WORDPIECE.value == "wordpiece"

    def test_unigram_value(self) -> None:
        """Test UNIGRAM value."""
        assert TokenizerType.UNIGRAM.value == "unigram"

    def test_sentencepiece_value(self) -> None:
        """Test SENTENCEPIECE value."""
        assert TokenizerType.SENTENCEPIECE.value == "sentencepiece"

    def test_tiktoken_value(self) -> None:
        """Test TIKTOKEN value."""
        assert TokenizerType.TIKTOKEN.value == "tiktoken"


class TestSpecialTokenType:
    """Tests for SpecialTokenType enum."""

    def test_pad_value(self) -> None:
        """Test PAD value."""
        assert SpecialTokenType.PAD.value == "pad"

    def test_unk_value(self) -> None:
        """Test UNK value."""
        assert SpecialTokenType.UNK.value == "unk"

    def test_bos_value(self) -> None:
        """Test BOS value."""
        assert SpecialTokenType.BOS.value == "bos"

    def test_eos_value(self) -> None:
        """Test EOS value."""
        assert SpecialTokenType.EOS.value == "eos"

    def test_sep_value(self) -> None:
        """Test SEP value."""
        assert SpecialTokenType.SEP.value == "sep"

    def test_cls_value(self) -> None:
        """Test CLS value."""
        assert SpecialTokenType.CLS.value == "cls"

    def test_mask_value(self) -> None:
        """Test MASK value."""
        assert SpecialTokenType.MASK.value == "mask"


class TestVocabAnalysisMetric:
    """Tests for VocabAnalysisMetric enum."""

    def test_coverage_value(self) -> None:
        """Test COVERAGE value."""
        assert VocabAnalysisMetric.COVERAGE.value == "coverage"

    def test_fertility_value(self) -> None:
        """Test FERTILITY value."""
        assert VocabAnalysisMetric.FERTILITY.value == "fertility"

    def test_unknown_rate_value(self) -> None:
        """Test UNKNOWN_RATE value."""
        assert VocabAnalysisMetric.UNKNOWN_RATE.value == "unknown_rate"


class TestTokenizerConfig:
    """Tests for TokenizerConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating TokenizerConfig instance."""
        config = TokenizerConfig(
            tokenizer_type=TokenizerType.BPE,
            vocab_size=32000,
            special_tokens={SpecialTokenType.PAD: "[PAD]"},
        )
        assert config.tokenizer_type == TokenizerType.BPE
        assert config.vocab_size == 32000
        assert SpecialTokenType.PAD in config.special_tokens

    def test_frozen(self) -> None:
        """Test that TokenizerConfig is immutable."""
        config = TokenizerConfig(
            tokenizer_type=TokenizerType.BPE,
            vocab_size=32000,
            special_tokens={},
        )
        with pytest.raises(AttributeError):
            config.vocab_size = 50000  # type: ignore[misc]


class TestVocabStats:
    """Tests for VocabStats dataclass."""

    def test_creation(self) -> None:
        """Test creating VocabStats instance."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=0.95,
            avg_token_length=4.5,
            unknown_rate=0.02,
        )
        assert stats.vocab_size == 32000
        assert stats.coverage == pytest.approx(0.95)
        assert stats.avg_token_length == pytest.approx(4.5)
        assert stats.unknown_rate == pytest.approx(0.02)

    def test_frozen(self) -> None:
        """Test that VocabStats is immutable."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=0.95,
            avg_token_length=4.5,
            unknown_rate=0.02,
        )
        with pytest.raises(AttributeError):
            stats.vocab_size = 50000  # type: ignore[misc]


class TestTokenizationResult:
    """Tests for TokenizationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating TokenizationResult instance."""
        result = TokenizationResult(
            tokens=("[CLS]", "hello", "world", "[SEP]"),
            token_ids=(101, 7592, 2088, 102),
            offsets=((0, 0), (0, 5), (6, 11), (0, 0)),
            special_mask=(True, False, False, True),
        )
        assert result.tokens == ("[CLS]", "hello", "world", "[SEP]")
        assert len(result.token_ids) == 4
        assert result.special_mask[0] is True

    def test_frozen(self) -> None:
        """Test that TokenizationResult is immutable."""
        result = TokenizationResult(
            tokens=("hello",),
            token_ids=(1,),
            offsets=((0, 5),),
            special_mask=(False,),
        )
        with pytest.raises(AttributeError):
            result.tokens = ("world",)  # type: ignore[misc]


class TestValidateTokenizerConfig:
    """Tests for validate_tokenizer_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = TokenizerConfig(
            tokenizer_type=TokenizerType.BPE,
            vocab_size=32000,
            special_tokens={SpecialTokenType.PAD: "[PAD]"},
        )
        validate_tokenizer_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_tokenizer_config(None)  # type: ignore[arg-type]

    def test_zero_vocab_size_raises_error(self) -> None:
        """Test that zero vocab_size raises ValueError."""
        config = TokenizerConfig(
            tokenizer_type=TokenizerType.BPE,
            vocab_size=0,
            special_tokens={},
        )
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_tokenizer_config(config)

    def test_negative_vocab_size_raises_error(self) -> None:
        """Test that negative vocab_size raises ValueError."""
        config = TokenizerConfig(
            tokenizer_type=TokenizerType.BPE,
            vocab_size=-100,
            special_tokens={},
        )
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_tokenizer_config(config)


class TestValidateVocabStats:
    """Tests for validate_vocab_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=0.95,
            avg_token_length=4.5,
            unknown_rate=0.02,
        )
        validate_vocab_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_vocab_stats(None)  # type: ignore[arg-type]

    def test_zero_vocab_size_raises_error(self) -> None:
        """Test that zero vocab_size raises ValueError."""
        stats = VocabStats(
            vocab_size=0,
            coverage=0.5,
            avg_token_length=4.0,
            unknown_rate=0.1,
        )
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_vocab_stats(stats)

    def test_coverage_above_one_raises_error(self) -> None:
        """Test that coverage above 1 raises ValueError."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=1.5,
            avg_token_length=4.0,
            unknown_rate=0.1,
        )
        with pytest.raises(ValueError, match="coverage must be between"):
            validate_vocab_stats(stats)

    def test_coverage_below_zero_raises_error(self) -> None:
        """Test that coverage below 0 raises ValueError."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=-0.1,
            avg_token_length=4.0,
            unknown_rate=0.1,
        )
        with pytest.raises(ValueError, match="coverage must be between"):
            validate_vocab_stats(stats)

    def test_negative_avg_token_length_raises_error(self) -> None:
        """Test that negative avg_token_length raises ValueError."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=0.5,
            avg_token_length=-1.0,
            unknown_rate=0.1,
        )
        with pytest.raises(ValueError, match="avg_token_length cannot be negative"):
            validate_vocab_stats(stats)

    def test_unknown_rate_above_one_raises_error(self) -> None:
        """Test that unknown_rate above 1 raises ValueError."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=0.5,
            avg_token_length=4.0,
            unknown_rate=1.5,
        )
        with pytest.raises(ValueError, match="unknown_rate must be between"):
            validate_vocab_stats(stats)


class TestValidateTokenizationResult:
    """Tests for validate_tokenization_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = TokenizationResult(
            tokens=("hello", "world"),
            token_ids=(1, 2),
            offsets=((0, 5), (6, 11)),
            special_mask=(False, False),
        )
        validate_tokenization_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_tokenization_result(None)  # type: ignore[arg-type]

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        result = TokenizationResult(
            tokens=("hello",),
            token_ids=(1, 2),
            offsets=((0, 5),),
            special_mask=(False,),
        )
        with pytest.raises(ValueError, match="must have same length"):
            validate_tokenization_result(result)

    def test_empty_result_is_valid(self) -> None:
        """Test that empty result is valid."""
        result = TokenizationResult(
            tokens=(),
            token_ids=(),
            offsets=(),
            special_mask=(),
        )
        validate_tokenization_result(result)  # Should not raise


class TestCreateTokenizerConfig:
    """Tests for create_tokenizer_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_tokenizer_config()
        assert config.tokenizer_type == TokenizerType.BPE
        assert config.vocab_size == 32000
        assert len(config.special_tokens) > 0

    def test_custom_tokenizer_type(self) -> None:
        """Test creating config with custom tokenizer type."""
        config = create_tokenizer_config(tokenizer_type="wordpiece")
        assert config.tokenizer_type == TokenizerType.WORDPIECE

    def test_custom_vocab_size(self) -> None:
        """Test creating config with custom vocab size."""
        config = create_tokenizer_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_custom_special_tokens(self) -> None:
        """Test creating config with custom special tokens."""
        config = create_tokenizer_config(
            special_tokens={"pad": "<pad>", "unk": "<unk>"}
        )
        assert config.special_tokens[SpecialTokenType.PAD] == "<pad>"
        assert config.special_tokens[SpecialTokenType.UNK] == "<unk>"

    def test_invalid_tokenizer_type_raises_error(self) -> None:
        """Test that invalid tokenizer type raises ValueError."""
        with pytest.raises(ValueError, match="tokenizer_type must be one of"):
            create_tokenizer_config(tokenizer_type="invalid")

    def test_invalid_vocab_size_raises_error(self) -> None:
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_tokenizer_config(vocab_size=0)

    def test_invalid_special_token_type_raises_error(self) -> None:
        """Test that invalid special token type raises ValueError."""
        with pytest.raises(ValueError, match="special token type must be one of"):
            create_tokenizer_config(special_tokens={"invalid": "[INVALID]"})


class TestCreateVocabStats:
    """Tests for create_vocab_stats function."""

    def test_default_values(self) -> None:
        """Test default statistics values."""
        stats = create_vocab_stats(vocab_size=32000)
        assert stats.vocab_size == 32000
        assert stats.coverage == pytest.approx(1.0)
        assert stats.avg_token_length == pytest.approx(4.0)
        assert stats.unknown_rate == pytest.approx(0.0)

    def test_custom_values(self) -> None:
        """Test creating stats with custom values."""
        stats = create_vocab_stats(
            vocab_size=50000,
            coverage=0.95,
            avg_token_length=5.5,
            unknown_rate=0.05,
        )
        assert stats.vocab_size == 50000
        assert stats.coverage == pytest.approx(0.95)

    def test_invalid_vocab_size_raises_error(self) -> None:
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_vocab_stats(vocab_size=0)

    def test_invalid_coverage_raises_error(self) -> None:
        """Test that invalid coverage raises ValueError."""
        with pytest.raises(ValueError, match="coverage must be between"):
            create_vocab_stats(vocab_size=1000, coverage=1.5)


class TestListTokenizerTypes:
    """Tests for list_tokenizer_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_tokenizer_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_tokenizer_types()
        assert "bpe" in types
        assert "wordpiece" in types
        assert "sentencepiece" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_tokenizer_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_TOKENIZER_TYPES."""
        types = list_tokenizer_types()
        assert set(types) == VALID_TOKENIZER_TYPES


class TestGetTokenizerType:
    """Tests for get_tokenizer_type function."""

    def test_get_bpe(self) -> None:
        """Test getting BPE type."""
        result = get_tokenizer_type("bpe")
        assert result == TokenizerType.BPE

    def test_get_wordpiece(self) -> None:
        """Test getting WORDPIECE type."""
        result = get_tokenizer_type("wordpiece")
        assert result == TokenizerType.WORDPIECE

    def test_get_sentencepiece(self) -> None:
        """Test getting SENTENCEPIECE type."""
        result = get_tokenizer_type("sentencepiece")
        assert result == TokenizerType.SENTENCEPIECE

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid tokenizer type"):
            get_tokenizer_type("invalid")


class TestListSpecialTokenTypes:
    """Tests for list_special_token_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_special_token_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_special_token_types()
        assert "pad" in types
        assert "unk" in types
        assert "mask" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_special_token_types()
        assert types == sorted(types)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_SPECIAL_TOKEN_TYPES."""
        types = list_special_token_types()
        assert set(types) == VALID_SPECIAL_TOKEN_TYPES


class TestGetSpecialTokenType:
    """Tests for get_special_token_type function."""

    def test_get_pad(self) -> None:
        """Test getting PAD type."""
        result = get_special_token_type("pad")
        assert result == SpecialTokenType.PAD

    def test_get_mask(self) -> None:
        """Test getting MASK type."""
        result = get_special_token_type("mask")
        assert result == SpecialTokenType.MASK

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid special token type"):
            get_special_token_type("invalid")


class TestListVocabAnalysisMetrics:
    """Tests for list_vocab_analysis_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        metrics = list_vocab_analysis_metrics()
        assert isinstance(metrics, list)

    def test_contains_expected_metrics(self) -> None:
        """Test that list contains expected metrics."""
        metrics = list_vocab_analysis_metrics()
        assert "coverage" in metrics
        assert "fertility" in metrics
        assert "unknown_rate" in metrics

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        metrics = list_vocab_analysis_metrics()
        assert metrics == sorted(metrics)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_VOCAB_ANALYSIS_METRICS."""
        metrics = list_vocab_analysis_metrics()
        assert set(metrics) == VALID_VOCAB_ANALYSIS_METRICS


class TestGetVocabAnalysisMetric:
    """Tests for get_vocab_analysis_metric function."""

    def test_get_coverage(self) -> None:
        """Test getting COVERAGE metric."""
        result = get_vocab_analysis_metric("coverage")
        assert result == VocabAnalysisMetric.COVERAGE

    def test_get_fertility(self) -> None:
        """Test getting FERTILITY metric."""
        result = get_vocab_analysis_metric("fertility")
        assert result == VocabAnalysisMetric.FERTILITY

    def test_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="invalid vocab analysis metric"):
            get_vocab_analysis_metric("invalid")


class TestAnalyzeVocabulary:
    """Tests for analyze_vocabulary function."""

    def test_full_coverage(self) -> None:
        """Test with full vocabulary coverage."""
        tokens = ["hello", "world", "test"]
        vocab = ["hello", "world", "test", "foo"]
        stats = analyze_vocabulary(tokens, vocab)
        assert stats.coverage == pytest.approx(1.0)

    def test_partial_coverage(self) -> None:
        """Test with partial vocabulary coverage."""
        tokens = ["hello", "world", "unknown"]
        vocab = ["hello", "world"]
        stats = analyze_vocabulary(tokens, vocab)
        assert stats.coverage < 1.0

    def test_unknown_rate_calculation(self) -> None:
        """Test unknown rate calculation."""
        tokens = ["hello", "[UNK]", "[UNK]"]
        vocab = ["hello", "[UNK]"]
        stats = analyze_vocabulary(tokens, vocab)
        assert stats.unknown_rate == pytest.approx(2 / 3)

    def test_custom_unk_token(self) -> None:
        """Test with custom unk token."""
        tokens = ["hello", "<unk>", "<unk>"]
        vocab = ["hello", "<unk>"]
        stats = analyze_vocabulary(tokens, vocab, unk_token="<unk>")
        assert stats.unknown_rate == pytest.approx(2 / 3)

    def test_avg_token_length(self) -> None:
        """Test average token length calculation."""
        tokens = ["ab", "cdef"]  # lengths 2 and 4
        vocab = ["ab", "cdef"]
        stats = analyze_vocabulary(tokens, vocab)
        assert stats.avg_token_length == pytest.approx(3.0)

    def test_empty_tokens(self) -> None:
        """Test with empty tokens list."""
        stats = analyze_vocabulary([], ["hello"])
        assert stats.coverage == pytest.approx(1.0)
        assert stats.unknown_rate == pytest.approx(0.0)
        assert stats.avg_token_length == pytest.approx(0.0)

    def test_empty_vocabulary(self) -> None:
        """Test with empty vocabulary."""
        stats = analyze_vocabulary(["hello"], [])
        assert stats.vocab_size == 1  # Minimum vocab size
        assert stats.coverage == pytest.approx(0.0)

    def test_none_tokens_raises_error(self) -> None:
        """Test that None tokens raises ValueError."""
        with pytest.raises(ValueError, match="tokens cannot be None"):
            analyze_vocabulary(None, [])  # type: ignore[arg-type]

    def test_none_vocabulary_raises_error(self) -> None:
        """Test that None vocabulary raises ValueError."""
        with pytest.raises(ValueError, match="vocabulary cannot be None"):
            analyze_vocabulary([], None)  # type: ignore[arg-type]


class TestCalculateFertility:
    """Tests for calculate_fertility function."""

    def test_one_token_per_word(self) -> None:
        """Test fertility of 1.0 (one token per word)."""
        text = "hello world"
        tokens = ["hello", "world"]
        fertility = calculate_fertility(text, tokens)
        assert fertility == pytest.approx(1.0)

    def test_multiple_tokens_per_word(self) -> None:
        """Test fertility > 1.0 (multiple tokens per word)."""
        text = "unbelievable"
        tokens = ["un", "##believ", "##able"]
        fertility = calculate_fertility(text, tokens)
        assert fertility == pytest.approx(3.0)

    def test_empty_text(self) -> None:
        """Test with empty text."""
        fertility = calculate_fertility("", [])
        assert fertility == pytest.approx(0.0)

    def test_empty_tokens(self) -> None:
        """Test with empty tokens."""
        fertility = calculate_fertility("hello world", [])
        assert fertility == pytest.approx(0.0)

    def test_text_with_no_words(self) -> None:
        """Test with text that has no words (only punctuation)."""
        fertility = calculate_fertility("...", [".", ".", "."])
        assert fertility == pytest.approx(0.0)

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            calculate_fertility(None, [])  # type: ignore[arg-type]

    def test_none_tokens_raises_error(self) -> None:
        """Test that None tokens raises ValueError."""
        with pytest.raises(ValueError, match="tokens cannot be None"):
            calculate_fertility("test", None)  # type: ignore[arg-type]


class TestEstimateSequenceLength:
    """Tests for estimate_sequence_length function."""

    def test_basic_estimate(self) -> None:
        """Test basic length estimation."""
        # "hello world" = 11 chars, 4 chars/token = 3 tokens + 2 special = 5
        length = estimate_sequence_length("hello world")
        assert length == 5

    def test_empty_text(self) -> None:
        """Test with empty text."""
        length = estimate_sequence_length("")
        assert length == 2  # Just special tokens

    def test_custom_chars_per_token(self) -> None:
        """Test with custom chars per token."""
        # "test" = 4 chars, 2 chars/token = 2 tokens + 2 special = 4
        length = estimate_sequence_length("test", avg_chars_per_token=2.0)
        assert length == 4

    def test_custom_special_tokens(self) -> None:
        """Test with custom special token count."""
        # "test" = 4 chars, 4 chars/token = 1 token + 0 special = 1
        length = estimate_sequence_length("test", include_special_tokens=0)
        assert length == 1

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            estimate_sequence_length(None)  # type: ignore[arg-type]

    def test_zero_chars_per_token_raises_error(self) -> None:
        """Test that zero avg_chars_per_token raises ValueError."""
        with pytest.raises(ValueError, match="avg_chars_per_token must be positive"):
            estimate_sequence_length("test", avg_chars_per_token=0)

    def test_negative_chars_per_token_raises_error(self) -> None:
        """Test that negative avg_chars_per_token raises ValueError."""
        with pytest.raises(ValueError, match="avg_chars_per_token must be positive"):
            estimate_sequence_length("test", avg_chars_per_token=-1.0)

    def test_negative_special_tokens_raises_error(self) -> None:
        """Test that negative include_special_tokens raises ValueError."""
        with pytest.raises(
            ValueError, match="include_special_tokens cannot be negative"
        ):
            estimate_sequence_length("test", include_special_tokens=-1)


class TestCompareTokenizers:
    """Tests for compare_tokenizers function."""

    def test_basic_comparison(self) -> None:
        """Test basic tokenizer comparison."""
        text = "hello world"
        tokenizations = {
            "bpe": ["hello", "world"],
            "wordpiece": ["hello", "##world"],
        }
        result = compare_tokenizers(text, tokenizations)
        assert "bpe" in result
        assert "wordpiece" in result
        assert result["bpe"]["token_count"] == pytest.approx(2.0)

    def test_fertility_calculation(self) -> None:
        """Test that fertility is calculated correctly."""
        text = "hello"
        tokenizations = {
            "test": ["hel", "lo"],
        }
        result = compare_tokenizers(text, tokenizations)
        assert result["test"]["fertility"] == pytest.approx(2.0)

    def test_compression_ratio(self) -> None:
        """Test compression ratio calculation."""
        text = "hello"  # 5 chars
        tokenizations = {
            "test": ["hello"],  # 1 token
        }
        result = compare_tokenizers(text, tokenizations)
        assert result["test"]["compression_ratio"] == pytest.approx(5.0)

    def test_empty_tokens_compression_ratio(self) -> None:
        """Test compression ratio with empty tokens."""
        text = "hello"
        tokenizations = {
            "test": [],
        }
        result = compare_tokenizers(text, tokenizations)
        assert result["test"]["compression_ratio"] == pytest.approx(0.0)

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            compare_tokenizers(None, {})  # type: ignore[arg-type]

    def test_none_tokenizations_raises_error(self) -> None:
        """Test that None tokenizations raises ValueError."""
        with pytest.raises(ValueError, match="tokenizations cannot be None"):
            compare_tokenizers("test", None)  # type: ignore[arg-type]

    def test_empty_tokenizations_raises_error(self) -> None:
        """Test that empty tokenizations raises ValueError."""
        with pytest.raises(ValueError, match="tokenizations cannot be empty"):
            compare_tokenizers("test", {})


class TestDetectSpecialTokens:
    """Tests for detect_special_tokens function."""

    def test_bracket_special_tokens(self) -> None:
        """Test detection of bracket-style special tokens."""
        tokens = ["[CLS]", "hello", "world", "[SEP]"]
        result = detect_special_tokens(tokens)
        assert result == [True, False, False, True]

    def test_angle_bracket_special_tokens(self) -> None:
        """Test detection of angle bracket special tokens."""
        tokens = ["<s>", "hello", "</s>"]
        result = detect_special_tokens(tokens)
        assert result == [True, False, True]

    def test_pipe_special_tokens(self) -> None:
        """Test detection of pipe-style special tokens."""
        tokens = ["<|endoftext|>", "hello"]
        result = detect_special_tokens(tokens)
        assert result == [True, False]

    def test_empty_tokens(self) -> None:
        """Test with empty tokens list."""
        result = detect_special_tokens([])
        assert result == []

    def test_no_special_tokens(self) -> None:
        """Test with no special tokens."""
        tokens = ["hello", "world"]
        result = detect_special_tokens(tokens)
        assert result == [False, False]

    def test_custom_patterns(self) -> None:
        """Test with custom patterns."""
        tokens = ["@@START@@", "hello", "@@END@@"]
        patterns = [r"^@@.*@@$"]
        result = detect_special_tokens(tokens, special_patterns=patterns)
        assert result == [True, False, True]

    def test_none_tokens_raises_error(self) -> None:
        """Test that None tokens raises ValueError."""
        with pytest.raises(ValueError, match="tokens cannot be None"):
            detect_special_tokens(None)  # type: ignore[arg-type]


class TestFormatVocabStats:
    """Tests for format_vocab_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = VocabStats(
            vocab_size=32000,
            coverage=0.95,
            avg_token_length=4.5,
            unknown_rate=0.02,
        )
        formatted = format_vocab_stats(stats)
        assert "32,000" in formatted
        assert "95.0%" in formatted
        assert "4.50" in formatted
        assert "2.00%" in formatted

    def test_contains_header(self) -> None:
        """Test that formatted output contains header."""
        stats = VocabStats(
            vocab_size=1000,
            coverage=1.0,
            avg_token_length=3.0,
            unknown_rate=0.0,
        )
        formatted = format_vocab_stats(stats)
        assert "Vocabulary Statistics" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_vocab_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedTokenizerConfig:
    """Tests for get_recommended_tokenizer_config function."""

    def test_general_task(self) -> None:
        """Test recommendation for general task."""
        config = get_recommended_tokenizer_config(task="general")
        assert config.vocab_size >= 30000

    def test_code_task(self) -> None:
        """Test recommendation for code task."""
        config = get_recommended_tokenizer_config(task="code")
        assert config.tokenizer_type == TokenizerType.BPE

    def test_multilingual_task(self) -> None:
        """Test recommendation for multilingual task."""
        config = get_recommended_tokenizer_config(task="multilingual")
        assert config.tokenizer_type == TokenizerType.SENTENCEPIECE
        assert config.vocab_size >= 100000

    def test_translation_task(self) -> None:
        """Test recommendation for translation task."""
        config = get_recommended_tokenizer_config(task="translation")
        assert config.tokenizer_type == TokenizerType.SENTENCEPIECE

    def test_bert_family(self) -> None:
        """Test recommendation for BERT family."""
        config = get_recommended_tokenizer_config(model_family="bert")
        assert config.tokenizer_type == TokenizerType.WORDPIECE

    def test_gpt_family(self) -> None:
        """Test recommendation for GPT family."""
        config = get_recommended_tokenizer_config(model_family="gpt")
        assert config.tokenizer_type == TokenizerType.BPE

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_tokenizer_config(task="invalid")

    def test_invalid_model_family_raises_error(self) -> None:
        """Test that invalid model_family raises ValueError."""
        with pytest.raises(ValueError, match="model_family must be one of"):
            get_recommended_tokenizer_config(model_family="invalid")


class TestPreprocessText:
    """Tests for preprocess_text function (legacy)."""

    def test_lowercase(self) -> None:
        """Test lowercase conversion."""
        result = preprocess_text("HELLO WORLD")
        assert result == "hello world"

    def test_strip_whitespace(self) -> None:
        """Test whitespace stripping."""
        result = preprocess_text("  hello   world  ")
        assert result == "hello world"

    def test_disable_lowercase(self) -> None:
        """Test disabling lowercase."""
        result = preprocess_text("HELLO", lowercase=False)
        assert result == "HELLO"

    def test_disable_strip_whitespace(self) -> None:
        """Test disabling whitespace stripping."""
        result = preprocess_text("  hello  ", strip_whitespace=False)
        assert result == "  hello  "

    def test_empty_string(self) -> None:
        """Test with empty string."""
        result = preprocess_text("")
        assert result == ""

    def test_idempotency(self) -> None:
        """Test that preprocessing is idempotent."""
        text = "  HELLO   WORLD  "
        once = preprocess_text(text)
        twice = preprocess_text(once)
        assert once == twice


class TestTokenizeBatch:
    """Tests for tokenize_batch function (legacy)."""

    def test_empty_texts_raises_error(self) -> None:
        """Test that empty texts raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            tokenize_batch([], None)

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            tokenize_batch(["hello"], MagicMock(), max_length=0)

    def test_negative_max_length_raises_error(self) -> None:
        """Test that negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            tokenize_batch(["hello"], MagicMock(), max_length=-1)

    def test_calls_tokenizer(self) -> None:
        """Test that tokenizer is called correctly."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        tokenize_batch(["hello"], mock_tokenizer)
        assert mock_tokenizer.call_count == 1


class TestCreatePreprocessingFunction:
    """Tests for create_preprocessing_function function (legacy)."""

    def test_empty_text_column_raises_error(self) -> None:
        """Test that empty text_column raises ValueError."""
        with pytest.raises(ValueError, match="text_column cannot be empty"):
            create_preprocessing_function(MagicMock(), text_column="")

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_preprocessing_function(MagicMock(), max_length=0)

    def test_returns_callable(self) -> None:
        """Test that function returns a callable."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2]}
        fn = create_preprocessing_function(mock_tokenizer)
        assert callable(fn)

    def test_preprocessing_function_calls_tokenizer(self) -> None:
        """Test that preprocessing function calls tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2]}
        fn = create_preprocessing_function(mock_tokenizer)
        fn({"text": "hello", "label": 0})
        assert mock_tokenizer.call_count == 1

    def test_preprocessing_function_handles_string(self) -> None:
        """Test that preprocessing function handles single string."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2]}
        fn = create_preprocessing_function(mock_tokenizer)
        fn({"text": "hello"})
        assert mock_tokenizer.call_count == 1

    def test_preprocessing_function_includes_labels(self) -> None:
        """Test that preprocessing function includes labels."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2]}
        fn = create_preprocessing_function(mock_tokenizer, label_column="label")
        result = fn({"text": "hello", "label": 1})
        assert result["labels"] == 1

    def test_preprocessing_function_no_label_column(self) -> None:
        """Test preprocessing function without label column."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2]}
        fn = create_preprocessing_function(mock_tokenizer, label_column=None)
        result = fn({"text": "hello"})
        assert "labels" not in result


class TestPropertyBased:
    """Property-based tests for tokenization module."""

    @given(st.integers(min_value=1, max_value=1000000))
    @settings(max_examples=20)
    def test_create_vocab_stats_always_valid(self, vocab_size: int) -> None:
        """Test that create_vocab_stats produces valid stats."""
        stats = create_vocab_stats(vocab_size=vocab_size)
        validate_vocab_stats(stats)
        assert stats.vocab_size == vocab_size

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_estimate_sequence_length_always_positive(self, text: str) -> None:
        """Test that estimate_sequence_length is always positive."""
        length = estimate_sequence_length(text)
        assert length >= 2  # At least special tokens

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20))
    @settings(max_examples=20)
    def test_detect_special_tokens_returns_correct_length(
        self, tokens: list[str]
    ) -> None:
        """Test that detect_special_tokens returns list of correct length."""
        result = detect_special_tokens(tokens)
        assert len(result) == len(tokens)
        assert all(isinstance(x, bool) for x in result)

    @given(
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
    )
    @settings(max_examples=20)
    def test_analyze_vocabulary_returns_valid_stats(
        self, tokens: list[str], vocab: list[str]
    ) -> None:
        """Test that analyze_vocabulary returns valid stats."""
        stats = analyze_vocabulary(tokens, vocab)
        assert stats.vocab_size >= 1
        assert 0.0 <= stats.coverage <= 1.0
        assert stats.avg_token_length >= 0.0
        assert 0.0 <= stats.unknown_rate <= 1.0

    @given(st.text(min_size=0, max_size=50))
    @settings(max_examples=20)
    def test_preprocess_text_idempotent(self, text: str) -> None:
        """Test that preprocess_text is idempotent."""
        once = preprocess_text(text)
        twice = preprocess_text(once)
        assert once == twice


class TestFrozensets:
    """Tests for frozen sets."""

    def test_valid_tokenizer_types_frozen(self) -> None:
        """Test that VALID_TOKENIZER_TYPES is a frozenset."""
        assert isinstance(VALID_TOKENIZER_TYPES, frozenset)
        assert len(VALID_TOKENIZER_TYPES) == len(TokenizerType)

    def test_valid_special_token_types_frozen(self) -> None:
        """Test that VALID_SPECIAL_TOKEN_TYPES is a frozenset."""
        assert isinstance(VALID_SPECIAL_TOKEN_TYPES, frozenset)
        assert len(VALID_SPECIAL_TOKEN_TYPES) == len(SpecialTokenType)

    def test_valid_vocab_analysis_metrics_frozen(self) -> None:
        """Test that VALID_VOCAB_ANALYSIS_METRICS is a frozenset."""
        assert isinstance(VALID_VOCAB_ANALYSIS_METRICS, frozenset)
        assert len(VALID_VOCAB_ANALYSIS_METRICS) == len(VocabAnalysisMetric)
