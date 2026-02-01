"""Tests for preprocessing.tokenizer_training module."""

from __future__ import annotations

import pytest

from hf_gtc.preprocessing.tokenizer_training import (
    VALID_ALGORITHMS,
    VALID_NORMALIZERS,
    VALID_PRE_TOKENIZERS,
    BPEConfig,
    PreTokenizerConfig,
    PreTokenizerType,
    TokenizerAlgorithm,
    TrainingCorpusConfig,
    UnigramConfig,
    WordPieceConfig,
    calculate_compression_ratio,
    create_bpe_config,
    create_pre_tokenizer_config,
    create_training_corpus_config,
    create_unigram_config,
    create_wordpiece_config,
    estimate_vocab_coverage,
    get_recommended_algorithm,
    get_tokenizer_algorithm,
    list_pre_tokenizers,
    list_tokenizer_algorithms,
    validate_bpe_config,
    validate_unigram_config,
    validate_wordpiece_config,
)


class TestTokenizerAlgorithm:
    """Tests for TokenizerAlgorithm enum."""

    def test_all_algorithms_have_values(self) -> None:
        """All algorithms have string values."""
        for algo in TokenizerAlgorithm:
            assert isinstance(algo.value, str)

    def test_bpe_value(self) -> None:
        """BPE has correct value."""
        assert TokenizerAlgorithm.BPE.value == "bpe"

    def test_wordpiece_value(self) -> None:
        """WordPiece has correct value."""
        assert TokenizerAlgorithm.WORDPIECE.value == "wordpiece"

    def test_valid_algorithms_frozenset(self) -> None:
        """VALID_ALGORITHMS is a frozenset."""
        assert isinstance(VALID_ALGORITHMS, frozenset)


class TestPreTokenizerType:
    """Tests for PreTokenizerType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for pre_tok in PreTokenizerType:
            assert isinstance(pre_tok.value, str)

    def test_byte_level_value(self) -> None:
        """Byte level has correct value."""
        assert PreTokenizerType.BYTE_LEVEL.value == "byte_level"

    def test_valid_pre_tokenizers_frozenset(self) -> None:
        """VALID_PRE_TOKENIZERS is a frozenset."""
        assert isinstance(VALID_PRE_TOKENIZERS, frozenset)


class TestBPEConfig:
    """Tests for BPEConfig dataclass."""

    def test_create_config(self) -> None:
        """Create BPE config."""
        config = BPEConfig(
            vocab_size=32000,
            min_frequency=2,
            show_progress=True,
            special_tokens=("<unk>",),
            initial_alphabet=(),
        )
        assert config.vocab_size == 32000

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = BPEConfig(32000, 2, True, ("<unk>",), ())
        with pytest.raises(AttributeError):
            config.vocab_size = 50000  # type: ignore[misc]


class TestValidateBPEConfig:
    """Tests for validate_bpe_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = BPEConfig(32000, 2, True, ("<unk>",), ())
        validate_bpe_config(config)

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab size raises ValueError."""
        config = BPEConfig(0, 2, True, ("<unk>",), ())
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_bpe_config(config)

    def test_zero_min_frequency_raises(self) -> None:
        """Zero min frequency raises ValueError."""
        config = BPEConfig(32000, 0, True, ("<unk>",), ())
        with pytest.raises(ValueError, match="min_frequency must be"):
            validate_bpe_config(config)


class TestWordPieceConfig:
    """Tests for WordPieceConfig dataclass."""

    def test_create_config(self) -> None:
        """Create WordPiece config."""
        config = WordPieceConfig(
            vocab_size=30522,
            min_frequency=2,
            show_progress=True,
            special_tokens=("[UNK]",),
            continuing_subword_prefix="##",
        )
        assert config.continuing_subword_prefix == "##"


class TestValidateWordPieceConfig:
    """Tests for validate_wordpiece_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = WordPieceConfig(30522, 2, True, ("[UNK]",), "##")
        validate_wordpiece_config(config)

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab size raises ValueError."""
        config = WordPieceConfig(0, 2, True, ("[UNK]",), "##")
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_wordpiece_config(config)

    def test_empty_prefix_raises(self) -> None:
        """Empty prefix raises ValueError."""
        config = WordPieceConfig(30522, 2, True, ("[UNK]",), "")
        with pytest.raises(
            ValueError, match="continuing_subword_prefix cannot be empty"
        ):
            validate_wordpiece_config(config)


class TestUnigramConfig:
    """Tests for UnigramConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Unigram config."""
        config = UnigramConfig(
            vocab_size=32000,
            shrinking_factor=0.75,
            special_tokens=("<unk>",),
            max_piece_length=16,
            n_sub_iterations=2,
        )
        assert config.shrinking_factor == pytest.approx(0.75)


class TestValidateUnigramConfig:
    """Tests for validate_unigram_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = UnigramConfig(32000, 0.75, ("<unk>",), 16, 2)
        validate_unigram_config(config)

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab size raises ValueError."""
        config = UnigramConfig(0, 0.75, ("<unk>",), 16, 2)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_unigram_config(config)

    def test_invalid_shrinking_factor_raises(self) -> None:
        """Invalid shrinking factor raises ValueError."""
        config = UnigramConfig(32000, 1.5, ("<unk>",), 16, 2)
        with pytest.raises(ValueError, match="shrinking_factor must be between"):
            validate_unigram_config(config)

    def test_zero_max_piece_length_raises(self) -> None:
        """Zero max piece length raises ValueError."""
        config = UnigramConfig(32000, 0.75, ("<unk>",), 0, 2)
        with pytest.raises(ValueError, match="max_piece_length must be positive"):
            validate_unigram_config(config)

    def test_zero_sub_iterations_raises(self) -> None:
        """Zero sub iterations raises ValueError."""
        config = UnigramConfig(32000, 0.75, ("<unk>",), 16, 0)
        with pytest.raises(ValueError, match="n_sub_iterations must be positive"):
            validate_unigram_config(config)


class TestCreateBPEConfig:
    """Tests for create_bpe_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_bpe_config()
        assert config.vocab_size == 32000
        assert config.min_frequency == 2

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_bpe_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_bpe_config(vocab_size=0)


class TestCreateWordPieceConfig:
    """Tests for create_wordpiece_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_wordpiece_config()
        assert config.vocab_size == 30522
        assert config.continuing_subword_prefix == "##"

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_wordpiece_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_empty_prefix_raises(self) -> None:
        """Empty prefix raises ValueError."""
        with pytest.raises(
            ValueError, match="continuing_subword_prefix cannot be empty"
        ):
            create_wordpiece_config(continuing_subword_prefix="")


class TestCreateUnigramConfig:
    """Tests for create_unigram_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_unigram_config()
        assert config.vocab_size == 32000
        assert config.shrinking_factor == pytest.approx(0.75)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_unigram_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_invalid_shrinking_factor_raises(self) -> None:
        """Invalid shrinking factor raises ValueError."""
        with pytest.raises(ValueError, match="shrinking_factor must be between"):
            create_unigram_config(shrinking_factor=1.5)


class TestCreatePreTokenizerConfig:
    """Tests for create_pre_tokenizer_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_pre_tokenizer_config()
        assert config.pre_tokenizer_type == PreTokenizerType.BYTE_LEVEL
        assert config.add_prefix_space is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_pre_tokenizer_config(pre_tokenizer_type="whitespace")
        assert config.pre_tokenizer_type == PreTokenizerType.WHITESPACE

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="pre_tokenizer_type must be one of"):
            create_pre_tokenizer_config(pre_tokenizer_type="invalid")


class TestCreateTrainingCorpusConfig:
    """Tests for create_training_corpus_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_training_corpus_config(("train.txt",))
        assert config.batch_size == 1000
        assert config.encoding == "utf-8"

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_training_corpus_config(
            ("train.txt", "valid.txt"),
            batch_size=2000,
        )
        assert len(config.files) == 2
        assert config.batch_size == 2000

    def test_empty_files_raises(self) -> None:
        """Empty files raises ValueError."""
        with pytest.raises(ValueError, match="files cannot be empty"):
            create_training_corpus_config(())

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_training_corpus_config(("train.txt",), batch_size=0)

    def test_zero_length_raises(self) -> None:
        """Zero length raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            create_training_corpus_config(("train.txt",), length=0)


class TestListTokenizerAlgorithms:
    """Tests for list_tokenizer_algorithms function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        algos = list_tokenizer_algorithms()
        assert algos == sorted(algos)

    def test_contains_bpe(self) -> None:
        """Contains bpe."""
        algos = list_tokenizer_algorithms()
        assert "bpe" in algos

    def test_contains_wordpiece(self) -> None:
        """Contains wordpiece."""
        algos = list_tokenizer_algorithms()
        assert "wordpiece" in algos


class TestListPreTokenizers:
    """Tests for list_pre_tokenizers function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        pretoks = list_pre_tokenizers()
        assert pretoks == sorted(pretoks)

    def test_contains_byte_level(self) -> None:
        """Contains byte_level."""
        pretoks = list_pre_tokenizers()
        assert "byte_level" in pretoks


class TestGetTokenizerAlgorithm:
    """Tests for get_tokenizer_algorithm function."""

    def test_get_bpe(self) -> None:
        """Get BPE algorithm."""
        assert get_tokenizer_algorithm("bpe") == TokenizerAlgorithm.BPE

    def test_get_wordpiece(self) -> None:
        """Get WordPiece algorithm."""
        assert get_tokenizer_algorithm("wordpiece") == TokenizerAlgorithm.WORDPIECE

    def test_invalid_algorithm_raises(self) -> None:
        """Invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            get_tokenizer_algorithm("invalid")


class TestGetRecommendedAlgorithm:
    """Tests for get_recommended_algorithm function."""

    def test_gpt_uses_bpe(self) -> None:
        """GPT uses BPE."""
        assert get_recommended_algorithm("gpt") == TokenizerAlgorithm.BPE

    def test_bert_uses_wordpiece(self) -> None:
        """BERT uses WordPiece."""
        assert get_recommended_algorithm("bert") == TokenizerAlgorithm.WORDPIECE

    def test_t5_uses_unigram(self) -> None:
        """T5 uses Unigram."""
        assert get_recommended_algorithm("t5") == TokenizerAlgorithm.UNIGRAM

    def test_llama_uses_bpe(self) -> None:
        """LLaMA uses BPE."""
        assert get_recommended_algorithm("llama") == TokenizerAlgorithm.BPE

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            get_recommended_algorithm("invalid")


class TestEstimateVocabCoverage:
    """Tests for estimate_vocab_coverage function."""

    def test_basic_estimate(self) -> None:
        """Basic coverage estimate."""
        coverage = estimate_vocab_coverage(32000, 1000000, TokenizerAlgorithm.BPE)
        assert 0 <= coverage <= 100

    def test_larger_vocab_higher_coverage(self) -> None:
        """Larger vocab has higher coverage."""
        cov1 = estimate_vocab_coverage(10000, 1000000, TokenizerAlgorithm.BPE)
        cov2 = estimate_vocab_coverage(50000, 1000000, TokenizerAlgorithm.BPE)
        assert cov2 >= cov1

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            estimate_vocab_coverage(0, 1000000, TokenizerAlgorithm.BPE)

    def test_zero_corpus_size_raises(self) -> None:
        """Zero corpus size raises ValueError."""
        with pytest.raises(ValueError, match="corpus_size must be positive"):
            estimate_vocab_coverage(32000, 0, TokenizerAlgorithm.BPE)


class TestCalculateCompressionRatio:
    """Tests for calculate_compression_ratio function."""

    def test_basic_ratio(self) -> None:
        """Basic compression ratio."""
        ratio = calculate_compression_ratio(1000, 250)
        assert ratio == pytest.approx(4.0)

    def test_lower_ratio(self) -> None:
        """Lower compression ratio."""
        ratio = calculate_compression_ratio(500, 250)
        assert ratio == pytest.approx(2.0)

    def test_zero_original_raises(self) -> None:
        """Zero original length raises ValueError."""
        with pytest.raises(ValueError, match="original_length must be positive"):
            calculate_compression_ratio(0, 250)

    def test_zero_tokenized_raises(self) -> None:
        """Zero tokenized length raises ValueError."""
        with pytest.raises(ValueError, match="tokenized_length must be positive"):
            calculate_compression_ratio(1000, 0)


class TestPreTokenizerConfig:
    """Tests for PreTokenizerConfig dataclass."""

    def test_create_config(self) -> None:
        """Create pre-tokenizer config."""
        config = PreTokenizerConfig(
            pre_tokenizer_type=PreTokenizerType.BYTE_LEVEL,
            add_prefix_space=True,
            trim_offsets=True,
            use_regex=False,
        )
        assert config.add_prefix_space is True


class TestTrainingCorpusConfig:
    """Tests for TrainingCorpusConfig dataclass."""

    def test_create_config(self) -> None:
        """Create training corpus config."""
        config = TrainingCorpusConfig(
            files=("train.txt",),
            batch_size=1000,
            length=None,
            encoding="utf-8",
        )
        assert config.batch_size == 1000


class TestValidNormalizers:
    """Tests for VALID_NORMALIZERS constant."""

    def test_valid_normalizers_frozenset(self) -> None:
        """VALID_NORMALIZERS is a frozenset."""
        assert isinstance(VALID_NORMALIZERS, frozenset)

    def test_contains_expected_values(self) -> None:
        """Contains expected values."""
        assert "nfc" in VALID_NORMALIZERS
        assert "lowercase" in VALID_NORMALIZERS
