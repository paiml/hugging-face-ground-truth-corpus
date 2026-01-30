"""Tests for vocabulary and BPE training utilities."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.vocabulary import (
    VALID_MERGE_STRATEGIES,
    VALID_SPECIAL_TOKEN_POSITIONS,
    VALID_VOCAB_TRAINING_METHODS,
    MergeConfig,
    MergeStrategy,
    SpecialTokenPosition,
    SpecialTokensConfig,
    VocabTrainingConfig,
    VocabTrainingMethod,
    VocabTrainingStats,
    calculate_merge_score,
    calculate_vocab_coverage,
    create_merge_config,
    create_special_tokens_config,
    create_vocab_training_config,
    estimate_compression_ratio,
    format_vocab_training_stats,
    get_merge_strategy,
    get_recommended_vocab_config,
    get_special_token_position,
    get_vocab_training_method,
    list_merge_strategies,
    list_special_token_positions,
    list_vocab_training_methods,
    validate_merge_config,
    validate_special_tokens,
    validate_special_tokens_config,
    validate_vocab_training_config,
    validate_vocab_training_stats,
)


class TestVocabTrainingMethod:
    """Tests for VocabTrainingMethod enum."""

    def test_bpe_value(self) -> None:
        """Test BPE value."""
        assert VocabTrainingMethod.BPE.value == "bpe"

    def test_wordpiece_value(self) -> None:
        """Test WORDPIECE value."""
        assert VocabTrainingMethod.WORDPIECE.value == "wordpiece"

    def test_unigram_value(self) -> None:
        """Test UNIGRAM value."""
        assert VocabTrainingMethod.UNIGRAM.value == "unigram"

    def test_sentencepiece_value(self) -> None:
        """Test SENTENCEPIECE value."""
        assert VocabTrainingMethod.SENTENCEPIECE.value == "sentencepiece"


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_frequency_value(self) -> None:
        """Test FREQUENCY value."""
        assert MergeStrategy.FREQUENCY.value == "frequency"

    def test_pmi_value(self) -> None:
        """Test PMI value."""
        assert MergeStrategy.PMI.value == "pmi"

    def test_bpe_dropout_value(self) -> None:
        """Test BPE_DROPOUT value."""
        assert MergeStrategy.BPE_DROPOUT.value == "bpe_dropout"


class TestSpecialTokenPosition:
    """Tests for SpecialTokenPosition enum."""

    def test_start_value(self) -> None:
        """Test START value."""
        assert SpecialTokenPosition.START.value == "start"

    def test_end_value(self) -> None:
        """Test END value."""
        assert SpecialTokenPosition.END.value == "end"

    def test_both_value(self) -> None:
        """Test BOTH value."""
        assert SpecialTokenPosition.BOTH.value == "both"


class TestValidFrozensets:
    """Tests for VALID_* frozensets."""

    def test_valid_vocab_training_methods(self) -> None:
        """Test VALID_VOCAB_TRAINING_METHODS contents."""
        assert "bpe" in VALID_VOCAB_TRAINING_METHODS
        assert "wordpiece" in VALID_VOCAB_TRAINING_METHODS
        assert "unigram" in VALID_VOCAB_TRAINING_METHODS
        assert "sentencepiece" in VALID_VOCAB_TRAINING_METHODS

    def test_valid_merge_strategies(self) -> None:
        """Test VALID_MERGE_STRATEGIES contents."""
        assert "frequency" in VALID_MERGE_STRATEGIES
        assert "pmi" in VALID_MERGE_STRATEGIES
        assert "bpe_dropout" in VALID_MERGE_STRATEGIES

    def test_valid_special_token_positions(self) -> None:
        """Test VALID_SPECIAL_TOKEN_POSITIONS contents."""
        assert "start" in VALID_SPECIAL_TOKEN_POSITIONS
        assert "end" in VALID_SPECIAL_TOKEN_POSITIONS
        assert "both" in VALID_SPECIAL_TOKEN_POSITIONS


class TestVocabTrainingConfig:
    """Tests for VocabTrainingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating VocabTrainingConfig instance."""
        config = VocabTrainingConfig(
            method=VocabTrainingMethod.BPE,
            vocab_size=32000,
            min_frequency=2,
            special_tokens=("<pad>", "<unk>"),
        )
        assert config.method == VocabTrainingMethod.BPE
        assert config.vocab_size == 32000
        assert config.min_frequency == 2
        assert len(config.special_tokens) == 2

    def test_frozen(self) -> None:
        """Test that VocabTrainingConfig is immutable."""
        config = VocabTrainingConfig(
            method=VocabTrainingMethod.BPE,
            vocab_size=32000,
            min_frequency=2,
            special_tokens=(),
        )
        with pytest.raises(AttributeError):
            config.vocab_size = 64000  # type: ignore[misc]


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating MergeConfig instance."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=10000,
            dropout_rate=0.1,
        )
        assert config.strategy == MergeStrategy.FREQUENCY
        assert config.num_merges == 10000
        assert config.dropout_rate == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that MergeConfig is immutable."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=10000,
            dropout_rate=0.0,
        )
        with pytest.raises(AttributeError):
            config.num_merges = 20000  # type: ignore[misc]


class TestSpecialTokensConfig:
    """Tests for SpecialTokensConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SpecialTokensConfig instance."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="<unk>",
            bos="<s>",
            eos="</s>",
            mask="<mask>",
            additional=("<sep>",),
        )
        assert config.pad == "<pad>"
        assert config.unk == "<unk>"
        assert config.bos == "<s>"
        assert config.eos == "</s>"
        assert config.mask == "<mask>"
        assert len(config.additional) == 1

    def test_frozen(self) -> None:
        """Test that SpecialTokensConfig is immutable."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="<unk>",
            bos="<s>",
            eos="</s>",
            mask="<mask>",
            additional=(),
        )
        with pytest.raises(AttributeError):
            config.pad = "[PAD]"  # type: ignore[misc]


class TestVocabTrainingStats:
    """Tests for VocabTrainingStats dataclass."""

    def test_creation(self) -> None:
        """Test creating VocabTrainingStats instance."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=98.5,
            avg_token_length=4.2,
        )
        assert stats.vocab_size == 32000
        assert stats.num_merges == 31744
        assert stats.coverage == pytest.approx(98.5)
        assert stats.avg_token_length == pytest.approx(4.2)

    def test_frozen(self) -> None:
        """Test that VocabTrainingStats is immutable."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=98.5,
            avg_token_length=4.2,
        )
        with pytest.raises(AttributeError):
            stats.vocab_size = 64000  # type: ignore[misc]


class TestValidateVocabTrainingConfig:
    """Tests for validate_vocab_training_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = VocabTrainingConfig(
            method=VocabTrainingMethod.BPE,
            vocab_size=32000,
            min_frequency=2,
            special_tokens=("<unk>",),
        )
        validate_vocab_training_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_vocab_training_config(None)  # type: ignore[arg-type]

    def test_zero_vocab_size_raises_error(self) -> None:
        """Test that zero vocab_size raises ValueError."""
        config = VocabTrainingConfig(
            method=VocabTrainingMethod.BPE,
            vocab_size=0,
            min_frequency=2,
            special_tokens=(),
        )
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_vocab_training_config(config)

    def test_negative_vocab_size_raises_error(self) -> None:
        """Test that negative vocab_size raises ValueError."""
        config = VocabTrainingConfig(
            method=VocabTrainingMethod.BPE,
            vocab_size=-1,
            min_frequency=2,
            special_tokens=(),
        )
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_vocab_training_config(config)

    def test_zero_min_frequency_raises_error(self) -> None:
        """Test that zero min_frequency raises ValueError."""
        config = VocabTrainingConfig(
            method=VocabTrainingMethod.BPE,
            vocab_size=32000,
            min_frequency=0,
            special_tokens=(),
        )
        with pytest.raises(ValueError, match="min_frequency must be >= 1"):
            validate_vocab_training_config(config)


class TestValidateMergeConfig:
    """Tests for validate_merge_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=10000,
            dropout_rate=0.1,
        )
        validate_merge_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_merge_config(None)  # type: ignore[arg-type]

    def test_zero_num_merges_raises_error(self) -> None:
        """Test that zero num_merges raises ValueError."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=0,
            dropout_rate=0.0,
        )
        with pytest.raises(ValueError, match="num_merges must be positive"):
            validate_merge_config(config)

    def test_negative_num_merges_raises_error(self) -> None:
        """Test that negative num_merges raises ValueError."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=-1,
            dropout_rate=0.0,
        )
        with pytest.raises(ValueError, match="num_merges must be positive"):
            validate_merge_config(config)

    def test_dropout_below_zero_raises_error(self) -> None:
        """Test that dropout_rate below 0 raises ValueError."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=10000,
            dropout_rate=-0.1,
        )
        with pytest.raises(ValueError, match="dropout_rate must be between"):
            validate_merge_config(config)

    def test_dropout_above_one_raises_error(self) -> None:
        """Test that dropout_rate above 1 raises ValueError."""
        config = MergeConfig(
            strategy=MergeStrategy.FREQUENCY,
            num_merges=10000,
            dropout_rate=1.5,
        )
        with pytest.raises(ValueError, match="dropout_rate must be between"):
            validate_merge_config(config)


class TestValidateSpecialTokensConfig:
    """Tests for validate_special_tokens_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="<unk>",
            bos="<s>",
            eos="</s>",
            mask="<mask>",
            additional=(),
        )
        validate_special_tokens_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_special_tokens_config(None)  # type: ignore[arg-type]

    def test_empty_pad_raises_error(self) -> None:
        """Test that empty pad token raises ValueError."""
        config = SpecialTokensConfig(
            pad="",
            unk="<unk>",
            bos="<s>",
            eos="</s>",
            mask="<mask>",
            additional=(),
        )
        with pytest.raises(ValueError, match="pad token cannot be empty"):
            validate_special_tokens_config(config)

    def test_empty_unk_raises_error(self) -> None:
        """Test that empty unk token raises ValueError."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="",
            bos="<s>",
            eos="</s>",
            mask="<mask>",
            additional=(),
        )
        with pytest.raises(ValueError, match="unk token cannot be empty"):
            validate_special_tokens_config(config)

    def test_empty_bos_raises_error(self) -> None:
        """Test that empty bos token raises ValueError."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="<unk>",
            bos="",
            eos="</s>",
            mask="<mask>",
            additional=(),
        )
        with pytest.raises(ValueError, match="bos token cannot be empty"):
            validate_special_tokens_config(config)

    def test_empty_eos_raises_error(self) -> None:
        """Test that empty eos token raises ValueError."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="<unk>",
            bos="<s>",
            eos="",
            mask="<mask>",
            additional=(),
        )
        with pytest.raises(ValueError, match="eos token cannot be empty"):
            validate_special_tokens_config(config)

    def test_empty_mask_raises_error(self) -> None:
        """Test that empty mask token raises ValueError."""
        config = SpecialTokensConfig(
            pad="<pad>",
            unk="<unk>",
            bos="<s>",
            eos="</s>",
            mask="",
            additional=(),
        )
        with pytest.raises(ValueError, match="mask token cannot be empty"):
            validate_special_tokens_config(config)


class TestValidateVocabTrainingStats:
    """Tests for validate_vocab_training_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=98.5,
            avg_token_length=4.2,
        )
        validate_vocab_training_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_vocab_training_stats(None)  # type: ignore[arg-type]

    def test_zero_vocab_size_raises_error(self) -> None:
        """Test that zero vocab_size raises ValueError."""
        stats = VocabTrainingStats(
            vocab_size=0,
            num_merges=0,
            coverage=0.0,
            avg_token_length=1.0,
        )
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_vocab_training_stats(stats)

    def test_negative_num_merges_raises_error(self) -> None:
        """Test that negative num_merges raises ValueError."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=-1,
            coverage=98.5,
            avg_token_length=4.2,
        )
        with pytest.raises(ValueError, match="num_merges cannot be negative"):
            validate_vocab_training_stats(stats)

    def test_coverage_above_100_raises_error(self) -> None:
        """Test that coverage above 100 raises ValueError."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=150.0,
            avg_token_length=4.2,
        )
        with pytest.raises(ValueError, match="coverage must be between"):
            validate_vocab_training_stats(stats)

    def test_coverage_below_zero_raises_error(self) -> None:
        """Test that coverage below 0 raises ValueError."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=-10.0,
            avg_token_length=4.2,
        )
        with pytest.raises(ValueError, match="coverage must be between"):
            validate_vocab_training_stats(stats)

    def test_zero_avg_token_length_raises_error(self) -> None:
        """Test that zero avg_token_length raises ValueError."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=98.5,
            avg_token_length=0.0,
        )
        with pytest.raises(ValueError, match="avg_token_length must be positive"):
            validate_vocab_training_stats(stats)


class TestCreateVocabTrainingConfig:
    """Tests for create_vocab_training_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_vocab_training_config()
        assert config.method == VocabTrainingMethod.BPE
        assert config.vocab_size == 32000
        assert config.min_frequency == 2
        assert len(config.special_tokens) == 4

    def test_custom_method(self) -> None:
        """Test creating config with custom method."""
        config = create_vocab_training_config(method="wordpiece")
        assert config.method == VocabTrainingMethod.WORDPIECE

    def test_custom_vocab_size(self) -> None:
        """Test creating config with custom vocab_size."""
        config = create_vocab_training_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_vocab_training_config(method="invalid")

    def test_invalid_vocab_size_raises_error(self) -> None:
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_vocab_training_config(vocab_size=0)


class TestCreateMergeConfig:
    """Tests for create_merge_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_merge_config()
        assert config.strategy == MergeStrategy.FREQUENCY
        assert config.num_merges == 10000
        assert config.dropout_rate == pytest.approx(0.0)

    def test_custom_strategy(self) -> None:
        """Test creating config with custom strategy."""
        config = create_merge_config(strategy="pmi")
        assert config.strategy == MergeStrategy.PMI

    def test_custom_num_merges(self) -> None:
        """Test creating config with custom num_merges."""
        config = create_merge_config(num_merges=20000)
        assert config.num_merges == 20000

    def test_custom_dropout_rate(self) -> None:
        """Test creating config with custom dropout_rate."""
        config = create_merge_config(dropout_rate=0.1)
        assert config.dropout_rate == pytest.approx(0.1)

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_merge_config(strategy="invalid")

    def test_invalid_num_merges_raises_error(self) -> None:
        """Test that invalid num_merges raises ValueError."""
        with pytest.raises(ValueError, match="num_merges must be positive"):
            create_merge_config(num_merges=0)


class TestCreateSpecialTokensConfig:
    """Tests for create_special_tokens_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_special_tokens_config()
        assert config.pad == "<pad>"
        assert config.unk == "<unk>"
        assert config.bos == "<s>"
        assert config.eos == "</s>"
        assert config.mask == "<mask>"
        assert config.additional == ()

    def test_custom_tokens(self) -> None:
        """Test creating config with custom tokens."""
        config = create_special_tokens_config(
            pad="[PAD]",
            unk="[UNK]",
            bos="[CLS]",
            eos="[SEP]",
            mask="[MASK]",
        )
        assert config.pad == "[PAD]"
        assert config.unk == "[UNK]"

    def test_with_additional_tokens(self) -> None:
        """Test creating config with additional tokens."""
        config = create_special_tokens_config(additional=("<extra>", "<custom>"))
        assert len(config.additional) == 2

    def test_empty_pad_raises_error(self) -> None:
        """Test that empty pad raises ValueError."""
        with pytest.raises(ValueError, match="pad token cannot be empty"):
            create_special_tokens_config(pad="")


class TestListVocabTrainingMethods:
    """Tests for list_vocab_training_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_vocab_training_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_vocab_training_methods()
        assert "bpe" in methods
        assert "wordpiece" in methods
        assert "unigram" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_vocab_training_methods()
        assert methods == sorted(methods)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_VOCAB_TRAINING_METHODS."""
        methods = list_vocab_training_methods()
        assert set(methods) == VALID_VOCAB_TRAINING_METHODS


class TestGetVocabTrainingMethod:
    """Tests for get_vocab_training_method function."""

    def test_get_bpe(self) -> None:
        """Test getting BPE method."""
        result = get_vocab_training_method("bpe")
        assert result == VocabTrainingMethod.BPE

    def test_get_wordpiece(self) -> None:
        """Test getting WORDPIECE method."""
        result = get_vocab_training_method("wordpiece")
        assert result == VocabTrainingMethod.WORDPIECE

    def test_get_unigram(self) -> None:
        """Test getting UNIGRAM method."""
        result = get_vocab_training_method("unigram")
        assert result == VocabTrainingMethod.UNIGRAM

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid vocab training method"):
            get_vocab_training_method("invalid")


class TestListMergeStrategies:
    """Tests for list_merge_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_merge_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_merge_strategies()
        assert "frequency" in strategies
        assert "pmi" in strategies
        assert "bpe_dropout" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_merge_strategies()
        assert strategies == sorted(strategies)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_MERGE_STRATEGIES."""
        strategies = list_merge_strategies()
        assert set(strategies) == VALID_MERGE_STRATEGIES


class TestGetMergeStrategy:
    """Tests for get_merge_strategy function."""

    def test_get_frequency(self) -> None:
        """Test getting FREQUENCY strategy."""
        result = get_merge_strategy("frequency")
        assert result == MergeStrategy.FREQUENCY

    def test_get_pmi(self) -> None:
        """Test getting PMI strategy."""
        result = get_merge_strategy("pmi")
        assert result == MergeStrategy.PMI

    def test_get_bpe_dropout(self) -> None:
        """Test getting BPE_DROPOUT strategy."""
        result = get_merge_strategy("bpe_dropout")
        assert result == MergeStrategy.BPE_DROPOUT

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid merge strategy"):
            get_merge_strategy("invalid")


class TestListSpecialTokenPositions:
    """Tests for list_special_token_positions function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        positions = list_special_token_positions()
        assert isinstance(positions, list)

    def test_contains_expected_positions(self) -> None:
        """Test that list contains expected positions."""
        positions = list_special_token_positions()
        assert "start" in positions
        assert "end" in positions
        assert "both" in positions

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        positions = list_special_token_positions()
        assert positions == sorted(positions)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_SPECIAL_TOKEN_POSITIONS."""
        positions = list_special_token_positions()
        assert set(positions) == VALID_SPECIAL_TOKEN_POSITIONS


class TestGetSpecialTokenPosition:
    """Tests for get_special_token_position function."""

    def test_get_start(self) -> None:
        """Test getting START position."""
        result = get_special_token_position("start")
        assert result == SpecialTokenPosition.START

    def test_get_end(self) -> None:
        """Test getting END position."""
        result = get_special_token_position("end")
        assert result == SpecialTokenPosition.END

    def test_get_both(self) -> None:
        """Test getting BOTH position."""
        result = get_special_token_position("both")
        assert result == SpecialTokenPosition.BOTH

    def test_invalid_position_raises_error(self) -> None:
        """Test that invalid position raises ValueError."""
        with pytest.raises(ValueError, match="invalid special token position"):
            get_special_token_position("invalid")


class TestCalculateVocabCoverage:
    """Tests for calculate_vocab_coverage function."""

    def test_full_coverage(self) -> None:
        """Test with full coverage."""
        vocab = ["hello", "world"]
        corpus = ["hello world"]
        coverage = calculate_vocab_coverage(vocab, corpus)
        assert coverage == pytest.approx(100.0)

    def test_partial_coverage(self) -> None:
        """Test with partial coverage."""
        vocab = ["hello"]
        corpus = ["hello world"]
        coverage = calculate_vocab_coverage(vocab, corpus)
        assert coverage == pytest.approx(50.0)

    def test_no_coverage(self) -> None:
        """Test with no coverage."""
        vocab = ["foo", "bar"]
        corpus = ["hello world"]
        coverage = calculate_vocab_coverage(vocab, corpus)
        assert coverage == pytest.approx(0.0)

    def test_empty_corpus(self) -> None:
        """Test with empty corpus."""
        coverage = calculate_vocab_coverage(["a", "b"], [])
        assert coverage == pytest.approx(0.0)

    def test_empty_vocab(self) -> None:
        """Test with empty vocab."""
        coverage = calculate_vocab_coverage([], ["hello world"])
        assert coverage == pytest.approx(0.0)

    def test_empty_texts_in_corpus(self) -> None:
        """Test with corpus containing only empty strings (zero tokens)."""
        coverage = calculate_vocab_coverage(["hello"], ["", "   "])
        assert coverage == pytest.approx(0.0)

    def test_none_vocab_raises_error(self) -> None:
        """Test that None vocab raises ValueError."""
        with pytest.raises(ValueError, match="vocab cannot be None"):
            calculate_vocab_coverage(None, [])  # type: ignore[arg-type]

    def test_none_corpus_raises_error(self) -> None:
        """Test that None corpus raises ValueError."""
        with pytest.raises(ValueError, match="corpus cannot be None"):
            calculate_vocab_coverage([], None)  # type: ignore[arg-type]


class TestEstimateCompressionRatio:
    """Tests for estimate_compression_ratio function."""

    def test_basic_ratio(self) -> None:
        """Test basic compression ratio calculation."""
        ratio = estimate_compression_ratio(32000)
        assert ratio > 0

    def test_larger_vocab_higher_ratio(self) -> None:
        """Test that larger vocab yields higher ratio."""
        ratio_small = estimate_compression_ratio(1000)
        ratio_large = estimate_compression_ratio(100000)
        assert ratio_small < ratio_large

    def test_custom_avg_word_length(self) -> None:
        """Test with custom avg_word_length."""
        ratio = estimate_compression_ratio(32000, avg_word_length=8.0)
        assert ratio > 0

    def test_zero_vocab_size_raises_error(self) -> None:
        """Test that zero vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            estimate_compression_ratio(0)

    def test_negative_vocab_size_raises_error(self) -> None:
        """Test that negative vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            estimate_compression_ratio(-1)

    def test_zero_avg_word_length_raises_error(self) -> None:
        """Test that zero avg_word_length raises ValueError."""
        with pytest.raises(ValueError, match="avg_word_length must be positive"):
            estimate_compression_ratio(32000, avg_word_length=0)


class TestValidateSpecialTokens:
    """Tests for validate_special_tokens function."""

    def test_all_valid(self) -> None:
        """Test with all tokens valid."""
        vocab = ["<pad>", "<unk>", "hello", "world"]
        tokens = ["<pad>", "<unk>"]
        valid, missing = validate_special_tokens(tokens, vocab)
        assert valid is True
        assert missing == []

    def test_some_missing(self) -> None:
        """Test with some tokens missing."""
        vocab = ["<pad>", "hello", "world"]
        tokens = ["<pad>", "<mask>"]
        valid, missing = validate_special_tokens(tokens, vocab)
        assert valid is False
        assert "<mask>" in missing

    def test_empty_tokens(self) -> None:
        """Test with empty tokens list."""
        vocab = ["<pad>", "hello"]
        tokens: list[str] = []
        valid, missing = validate_special_tokens(tokens, vocab)
        assert valid is True
        assert missing == []

    def test_none_tokens_raises_error(self) -> None:
        """Test that None tokens raises ValueError."""
        with pytest.raises(ValueError, match="tokens cannot be None"):
            validate_special_tokens(None, [])  # type: ignore[arg-type]

    def test_none_vocab_raises_error(self) -> None:
        """Test that None vocab raises ValueError."""
        with pytest.raises(ValueError, match="vocab cannot be None"):
            validate_special_tokens([], None)  # type: ignore[arg-type]


class TestCalculateMergeScore:
    """Tests for calculate_merge_score function."""

    def test_frequency_strategy(self) -> None:
        """Test with FREQUENCY strategy."""
        score = calculate_merge_score(100, 500, 400, 10000, MergeStrategy.FREQUENCY)
        assert score == pytest.approx(100.0)

    def test_pmi_strategy(self) -> None:
        """Test with PMI strategy."""
        score = calculate_merge_score(100, 500, 400, 10000, MergeStrategy.PMI)
        assert score != 100.0  # PMI gives different score

    def test_bpe_dropout_strategy(self) -> None:
        """Test with BPE_DROPOUT strategy."""
        score = calculate_merge_score(100, 500, 400, 10000, MergeStrategy.BPE_DROPOUT)
        assert score == pytest.approx(100.0)

    def test_zero_pair_frequency_pmi(self) -> None:
        """Test PMI with zero pair frequency."""
        score = calculate_merge_score(0, 500, 400, 10000, MergeStrategy.PMI)
        assert score == pytest.approx(0.0)

    def test_negative_pair_frequency_raises_error(self) -> None:
        """Test that negative pair_frequency raises ValueError."""
        with pytest.raises(ValueError, match="pair_frequency cannot be negative"):
            calculate_merge_score(-1, 500, 400, 10000)

    def test_negative_left_frequency_raises_error(self) -> None:
        """Test that negative left_frequency raises ValueError."""
        with pytest.raises(ValueError, match="left_frequency cannot be negative"):
            calculate_merge_score(100, -1, 400, 10000)

    def test_negative_right_frequency_raises_error(self) -> None:
        """Test that negative right_frequency raises ValueError."""
        with pytest.raises(ValueError, match="right_frequency cannot be negative"):
            calculate_merge_score(100, 500, -1, 10000)

    def test_zero_total_pairs_raises_error(self) -> None:
        """Test that zero total_pairs raises ValueError."""
        with pytest.raises(ValueError, match="total_pairs must be positive"):
            calculate_merge_score(100, 500, 400, 0)


class TestFormatVocabTrainingStats:
    """Tests for format_vocab_training_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=98.5,
            avg_token_length=4.2,
        )
        formatted = format_vocab_training_stats(stats)
        assert "32,000" in formatted or "32000" in formatted
        assert "98.5" in formatted
        assert "4.2" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_vocab_training_stats(None)  # type: ignore[arg-type]

    def test_formatting_contains_headers(self) -> None:
        """Test that formatting contains headers."""
        stats = VocabTrainingStats(
            vocab_size=32000,
            num_merges=31744,
            coverage=98.5,
            avg_token_length=4.2,
        )
        formatted = format_vocab_training_stats(stats)
        assert "Vocabulary Statistics" in formatted


class TestGetRecommendedVocabConfig:
    """Tests for get_recommended_vocab_config function."""

    def test_gpt_config(self) -> None:
        """Test recommendation for GPT."""
        config = get_recommended_vocab_config("gpt", 1000000)
        assert config.method == VocabTrainingMethod.BPE

    def test_bert_config(self) -> None:
        """Test recommendation for BERT."""
        config = get_recommended_vocab_config("bert", 1000000)
        assert config.method == VocabTrainingMethod.WORDPIECE

    def test_t5_config(self) -> None:
        """Test recommendation for T5."""
        config = get_recommended_vocab_config("t5", 1000000)
        assert config.method == VocabTrainingMethod.UNIGRAM

    def test_llama_config(self) -> None:
        """Test recommendation for LLaMA."""
        config = get_recommended_vocab_config("llama", 1000000)
        assert config.method == VocabTrainingMethod.BPE

    def test_small_corpus_vocab_size(self) -> None:
        """Test vocab size for small corpus."""
        config = get_recommended_vocab_config("gpt", 50000)
        assert config.vocab_size == 8000

    def test_medium_corpus_vocab_size(self) -> None:
        """Test vocab size for medium corpus."""
        config = get_recommended_vocab_config("gpt", 500000)
        assert config.vocab_size == 16000

    def test_large_corpus_vocab_size(self) -> None:
        """Test vocab size for large corpus."""
        config = get_recommended_vocab_config("gpt", 5000000)
        assert config.vocab_size == 32000

    def test_very_large_corpus_vocab_size(self) -> None:
        """Test vocab size for very large corpus."""
        config = get_recommended_vocab_config("gpt", 50000000)
        assert config.vocab_size == 50000

    def test_invalid_model_type_raises_error(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_vocab_config("invalid", 1000000)

    def test_zero_corpus_size_raises_error(self) -> None:
        """Test that zero corpus_size raises ValueError."""
        with pytest.raises(ValueError, match="corpus_size must be positive"):
            get_recommended_vocab_config("gpt", 0)

    def test_negative_corpus_size_raises_error(self) -> None:
        """Test that negative corpus_size raises ValueError."""
        with pytest.raises(ValueError, match="corpus_size must be positive"):
            get_recommended_vocab_config("gpt", -1)


class TestPropertyBased:
    """Property-based tests for vocabulary module."""

    @given(st.integers(min_value=1, max_value=100000))
    @settings(max_examples=20)
    def test_estimate_compression_ratio_always_positive(
        self, vocab_size: int
    ) -> None:
        """Test that compression ratio is always positive."""
        ratio = estimate_compression_ratio(vocab_size)
        assert ratio > 0

    @given(
        st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10),
        st.lists(st.text(min_size=0, max_size=50), min_size=0, max_size=5),
    )
    @settings(max_examples=20)
    def test_vocab_coverage_in_valid_range(
        self, vocab: list[str], corpus: list[str]
    ) -> None:
        """Test that coverage is always in valid range."""
        coverage = calculate_vocab_coverage(vocab, corpus)
        assert 0.0 <= coverage <= 100.0

    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=20)
    def test_merge_score_is_finite(
        self,
        pair_freq: int,
        left_freq: int,
        right_freq: int,
        total: int,
    ) -> None:
        """Test that merge score is always finite."""
        import math

        score = calculate_merge_score(pair_freq, left_freq, right_freq, total)
        assert not math.isnan(score)  # Check not NaN

    @given(st.sampled_from(["gpt", "bert", "t5", "llama"]))
    @settings(max_examples=10)
    def test_recommended_config_for_all_model_types(self, model_type: str) -> None:
        """Test that recommendations work for all model types."""
        config = get_recommended_vocab_config(model_type, 1000000)
        assert config.vocab_size > 0
        assert config.min_frequency >= 1
        validate_vocab_training_config(config)
