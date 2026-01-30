"""Tests for data quality functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.quality import (
    VALID_DEDUPLICATION_METHODS,
    VALID_FILTER_STRATEGIES,
    VALID_QUALITY_METRICS,
    ContaminationConfig,
    DeduplicationConfig,
    DeduplicationMethod,
    FilterStrategy,
    QualityFilterConfig,
    QualityMetric,
    QualityStats,
    calculate_perplexity_score,
    calculate_text_quality_score,
    check_contamination,
    compute_quality_stats,
    create_contamination_config,
    create_deduplication_config,
    create_quality_filter_config,
    detect_duplicates,
    format_quality_stats,
    get_deduplication_method,
    get_filter_strategy,
    get_quality_metric,
    get_recommended_quality_config,
    list_deduplication_methods,
    list_filter_strategies,
    list_quality_metrics,
    validate_contamination_config,
    validate_deduplication_config,
    validate_quality_filter_config,
)


class TestDeduplicationMethod:
    """Tests for DeduplicationMethod enum."""

    def test_exact_hash_value(self) -> None:
        """Test EXACT_HASH value."""
        assert DeduplicationMethod.EXACT_HASH.value == "exact_hash"

    def test_minhash_value(self) -> None:
        """Test MINHASH value."""
        assert DeduplicationMethod.MINHASH.value == "minhash"

    def test_simhash_value(self) -> None:
        """Test SIMHASH value."""
        assert DeduplicationMethod.SIMHASH.value == "simhash"

    def test_semantic_value(self) -> None:
        """Test SEMANTIC value."""
        assert DeduplicationMethod.SEMANTIC.value == "semantic"

    def test_ngram_value(self) -> None:
        """Test NGRAM value."""
        assert DeduplicationMethod.NGRAM.value == "ngram"


class TestQualityMetric:
    """Tests for QualityMetric enum."""

    def test_perplexity_value(self) -> None:
        """Test PERPLEXITY value."""
        assert QualityMetric.PERPLEXITY.value == "perplexity"

    def test_length_value(self) -> None:
        """Test LENGTH value."""
        assert QualityMetric.LENGTH.value == "length"

    def test_repetition_value(self) -> None:
        """Test REPETITION value."""
        assert QualityMetric.REPETITION.value == "repetition"

    def test_toxicity_value(self) -> None:
        """Test TOXICITY value."""
        assert QualityMetric.TOXICITY.value == "toxicity"

    def test_language_score_value(self) -> None:
        """Test LANGUAGE_SCORE value."""
        assert QualityMetric.LANGUAGE_SCORE.value == "language_score"


class TestFilterStrategy:
    """Tests for FilterStrategy enum."""

    def test_threshold_value(self) -> None:
        """Test THRESHOLD value."""
        assert FilterStrategy.THRESHOLD.value == "threshold"

    def test_percentile_value(self) -> None:
        """Test PERCENTILE value."""
        assert FilterStrategy.PERCENTILE.value == "percentile"

    def test_zscore_value(self) -> None:
        """Test ZSCORE value."""
        assert FilterStrategy.ZSCORE.value == "zscore"


class TestDeduplicationConfig:
    """Tests for DeduplicationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating DeduplicationConfig instance."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.MINHASH,
            similarity_threshold=0.8,
            ngram_size=3,
            num_hashes=64,
        )
        assert config.method == DeduplicationMethod.MINHASH
        assert config.similarity_threshold == pytest.approx(0.8)
        assert config.ngram_size == 3
        assert config.num_hashes == 64

    def test_frozen(self) -> None:
        """Test that DeduplicationConfig is immutable."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.EXACT_HASH,
            similarity_threshold=0.9,
            ngram_size=5,
            num_hashes=128,
        )
        with pytest.raises(AttributeError):
            config.ngram_size = 10  # type: ignore[misc]


class TestQualityFilterConfig:
    """Tests for QualityFilterConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating QualityFilterConfig instance."""
        config = QualityFilterConfig(
            metrics=(QualityMetric.PERPLEXITY, QualityMetric.LENGTH),
            thresholds={"perplexity": 100.0, "length": 10},
            filter_strategy=FilterStrategy.THRESHOLD,
            remove_outliers=True,
        )
        assert len(config.metrics) == 2
        assert config.filter_strategy == FilterStrategy.THRESHOLD
        assert config.remove_outliers is True

    def test_frozen(self) -> None:
        """Test that QualityFilterConfig is immutable."""
        config = QualityFilterConfig(
            metrics=(QualityMetric.LENGTH,),
            thresholds={"length": 10},
            filter_strategy=FilterStrategy.THRESHOLD,
            remove_outliers=False,
        )
        with pytest.raises(AttributeError):
            config.remove_outliers = True  # type: ignore[misc]


class TestContaminationConfig:
    """Tests for ContaminationConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ContaminationConfig instance."""
        config = ContaminationConfig(
            test_datasets=("squad", "trivia_qa"),
            ngram_overlap_threshold=0.8,
            exact_match=True,
        )
        assert len(config.test_datasets) == 2
        assert config.ngram_overlap_threshold == pytest.approx(0.8)
        assert config.exact_match is True

    def test_frozen(self) -> None:
        """Test that ContaminationConfig is immutable."""
        config = ContaminationConfig(
            test_datasets=("test",),
            ngram_overlap_threshold=0.5,
            exact_match=False,
        )
        with pytest.raises(AttributeError):
            config.exact_match = True  # type: ignore[misc]


class TestQualityStats:
    """Tests for QualityStats dataclass."""

    def test_creation(self) -> None:
        """Test creating QualityStats instance."""
        stats = QualityStats(
            total_samples=10000,
            filtered_count=500,
            duplicate_count=200,
            quality_distribution={"high": 9000, "low": 1000},
        )
        assert stats.total_samples == 10000
        assert stats.filtered_count == 500
        assert stats.duplicate_count == 200
        assert stats.quality_distribution["high"] == 9000

    def test_frozen(self) -> None:
        """Test that QualityStats is immutable."""
        stats = QualityStats(
            total_samples=100,
            filtered_count=10,
            duplicate_count=5,
            quality_distribution={},
        )
        with pytest.raises(AttributeError):
            stats.total_samples = 200  # type: ignore[misc]


class TestValidateDeduplicationConfig:
    """Tests for validate_deduplication_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.MINHASH,
            similarity_threshold=0.8,
            ngram_size=3,
            num_hashes=64,
        )
        validate_deduplication_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_deduplication_config(None)  # type: ignore[arg-type]

    def test_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0 raises ValueError."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.MINHASH,
            similarity_threshold=-0.1,
            ngram_size=3,
            num_hashes=64,
        )
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_deduplication_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.MINHASH,
            similarity_threshold=1.5,
            ngram_size=3,
            num_hashes=64,
        )
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_deduplication_config(config)

    def test_zero_ngram_size_raises_error(self) -> None:
        """Test that zero ngram_size raises ValueError."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.MINHASH,
            similarity_threshold=0.8,
            ngram_size=0,
            num_hashes=64,
        )
        with pytest.raises(ValueError, match="ngram_size must be positive"):
            validate_deduplication_config(config)

    def test_zero_num_hashes_raises_error(self) -> None:
        """Test that zero num_hashes raises ValueError."""
        config = DeduplicationConfig(
            method=DeduplicationMethod.MINHASH,
            similarity_threshold=0.8,
            ngram_size=3,
            num_hashes=0,
        )
        with pytest.raises(ValueError, match="num_hashes must be positive"):
            validate_deduplication_config(config)


class TestValidateQualityFilterConfig:
    """Tests for validate_quality_filter_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = QualityFilterConfig(
            metrics=(QualityMetric.PERPLEXITY,),
            thresholds={"perplexity": 100.0},
            filter_strategy=FilterStrategy.THRESHOLD,
            remove_outliers=False,
        )
        validate_quality_filter_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_quality_filter_config(None)  # type: ignore[arg-type]

    def test_empty_metrics_raises_error(self) -> None:
        """Test that empty metrics raises ValueError."""
        config = QualityFilterConfig(
            metrics=(),
            thresholds={},
            filter_strategy=FilterStrategy.THRESHOLD,
            remove_outliers=False,
        )
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            validate_quality_filter_config(config)


class TestValidateContaminationConfig:
    """Tests for validate_contamination_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ContaminationConfig(
            test_datasets=("squad",),
            ngram_overlap_threshold=0.8,
            exact_match=True,
        )
        validate_contamination_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_contamination_config(None)  # type: ignore[arg-type]

    def test_empty_datasets_raises_error(self) -> None:
        """Test that empty test_datasets raises ValueError."""
        config = ContaminationConfig(
            test_datasets=(),
            ngram_overlap_threshold=0.8,
            exact_match=True,
        )
        with pytest.raises(ValueError, match="test_datasets cannot be empty"):
            validate_contamination_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1 raises ValueError."""
        config = ContaminationConfig(
            test_datasets=("test",),
            ngram_overlap_threshold=1.5,
            exact_match=True,
        )
        with pytest.raises(ValueError, match="ngram_overlap_threshold must be between"):
            validate_contamination_config(config)


class TestCreateDeduplicationConfig:
    """Tests for create_deduplication_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_deduplication_config()
        assert config.method == DeduplicationMethod.EXACT_HASH
        assert config.similarity_threshold == pytest.approx(0.9)
        assert config.ngram_size == 5
        assert config.num_hashes == 128

    def test_custom_method(self) -> None:
        """Test creating config with custom method."""
        config = create_deduplication_config(method="minhash")
        assert config.method == DeduplicationMethod.MINHASH

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_deduplication_config(method="invalid")

    def test_invalid_ngram_size_raises_error(self) -> None:
        """Test that invalid ngram_size raises ValueError."""
        with pytest.raises(ValueError, match="ngram_size must be positive"):
            create_deduplication_config(ngram_size=0)

    def test_invalid_num_hashes_raises_error(self) -> None:
        """Test that invalid num_hashes raises ValueError."""
        with pytest.raises(ValueError, match="num_hashes must be positive"):
            create_deduplication_config(num_hashes=-1)


class TestCreateQualityFilterConfig:
    """Tests for create_quality_filter_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_quality_filter_config()
        assert len(config.metrics) == 2
        assert config.filter_strategy == FilterStrategy.THRESHOLD
        assert config.remove_outliers is True

    def test_custom_metrics(self) -> None:
        """Test creating config with custom metrics."""
        config = create_quality_filter_config(metrics=("perplexity", "toxicity"))
        assert QualityMetric.PERPLEXITY in config.metrics
        assert QualityMetric.TOXICITY in config.metrics

    def test_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            create_quality_filter_config(metrics=("invalid",))

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid filter_strategy raises ValueError."""
        with pytest.raises(ValueError, match="filter_strategy must be one of"):
            create_quality_filter_config(filter_strategy="invalid")

    def test_custom_thresholds(self) -> None:
        """Test creating config with custom thresholds."""
        config = create_quality_filter_config(
            metrics=("length",),
            thresholds={"length": 50},
        )
        assert config.thresholds["length"] == 50


class TestCreateContaminationConfig:
    """Tests for create_contamination_config function."""

    def test_basic_creation(self) -> None:
        """Test basic config creation."""
        config = create_contamination_config(test_datasets=("squad", "nq"))
        assert len(config.test_datasets) == 2
        assert config.ngram_overlap_threshold == pytest.approx(0.8)
        assert config.exact_match is True

    def test_empty_datasets_raises_error(self) -> None:
        """Test that empty test_datasets raises ValueError."""
        with pytest.raises(ValueError, match="test_datasets cannot be empty"):
            create_contamination_config(test_datasets=())

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(
            ValueError, match="ngram_overlap_threshold must be between"
        ):
            create_contamination_config(
                test_datasets=("test",),
                ngram_overlap_threshold=1.5,
            )


class TestListDeduplicationMethods:
    """Tests for list_deduplication_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_deduplication_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_deduplication_methods()
        assert "exact_hash" in methods
        assert "minhash" in methods
        assert "ngram" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_deduplication_methods()
        assert methods == sorted(methods)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_DEDUPLICATION_METHODS."""
        methods = list_deduplication_methods()
        assert set(methods) == VALID_DEDUPLICATION_METHODS


class TestGetDeduplicationMethod:
    """Tests for get_deduplication_method function."""

    def test_get_exact_hash(self) -> None:
        """Test getting EXACT_HASH method."""
        result = get_deduplication_method("exact_hash")
        assert result == DeduplicationMethod.EXACT_HASH

    def test_get_minhash(self) -> None:
        """Test getting MINHASH method."""
        result = get_deduplication_method("minhash")
        assert result == DeduplicationMethod.MINHASH

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid deduplication method"):
            get_deduplication_method("invalid")


class TestListQualityMetrics:
    """Tests for list_quality_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        metrics = list_quality_metrics()
        assert isinstance(metrics, list)

    def test_contains_expected_metrics(self) -> None:
        """Test that list contains expected metrics."""
        metrics = list_quality_metrics()
        assert "perplexity" in metrics
        assert "length" in metrics
        assert "toxicity" in metrics

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        metrics = list_quality_metrics()
        assert metrics == sorted(metrics)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_QUALITY_METRICS."""
        metrics = list_quality_metrics()
        assert set(metrics) == VALID_QUALITY_METRICS


class TestGetQualityMetric:
    """Tests for get_quality_metric function."""

    def test_get_perplexity(self) -> None:
        """Test getting PERPLEXITY metric."""
        result = get_quality_metric("perplexity")
        assert result == QualityMetric.PERPLEXITY

    def test_get_toxicity(self) -> None:
        """Test getting TOXICITY metric."""
        result = get_quality_metric("toxicity")
        assert result == QualityMetric.TOXICITY

    def test_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="invalid quality metric"):
            get_quality_metric("invalid")


class TestListFilterStrategies:
    """Tests for list_filter_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_filter_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_filter_strategies()
        assert "threshold" in strategies
        assert "percentile" in strategies
        assert "zscore" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_filter_strategies()
        assert strategies == sorted(strategies)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_FILTER_STRATEGIES."""
        strategies = list_filter_strategies()
        assert set(strategies) == VALID_FILTER_STRATEGIES


class TestGetFilterStrategy:
    """Tests for get_filter_strategy function."""

    def test_get_threshold(self) -> None:
        """Test getting THRESHOLD strategy."""
        result = get_filter_strategy("threshold")
        assert result == FilterStrategy.THRESHOLD

    def test_get_zscore(self) -> None:
        """Test getting ZSCORE strategy."""
        result = get_filter_strategy("zscore")
        assert result == FilterStrategy.ZSCORE

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid filter strategy"):
            get_filter_strategy("invalid")


class TestCalculateTextQualityScore:
    """Tests for calculate_text_quality_score function."""

    def test_basic_score(self) -> None:
        """Test basic quality score calculation."""
        scores = calculate_text_quality_score("Hello world, this is a test.")
        assert "length" in scores
        assert "repetition" in scores
        assert scores["length"] > 0

    def test_empty_text(self) -> None:
        """Test with empty text."""
        scores = calculate_text_quality_score("")
        assert scores["length"] == 0.0
        assert scores["repetition"] == 0.0

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            calculate_text_quality_score(None)  # type: ignore[arg-type]

    def test_specific_metrics(self) -> None:
        """Test calculating specific metrics only."""
        scores = calculate_text_quality_score(
            "Hello world",
            metrics=[QualityMetric.LENGTH, QualityMetric.REPETITION],
        )
        assert "length" in scores
        assert "repetition" in scores

    def test_repetition_detection(self) -> None:
        """Test repetition score calculation."""
        # High repetition text
        repetitive = "word word word word word"
        scores = calculate_text_quality_score(repetitive)
        assert scores["repetition"] > 0.5

        # Low repetition text
        diverse = "one two three four five"
        scores = calculate_text_quality_score(diverse)
        assert scores["repetition"] == 0.0

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_length_matches_text_length(self, text: str) -> None:
        """Test that length score matches actual text length."""
        scores = calculate_text_quality_score(text)
        assert scores["length"] == len(text)


class TestDetectDuplicates:
    """Tests for detect_duplicates function."""

    def test_exact_duplicates(self) -> None:
        """Test detection of exact duplicates."""
        texts = ["hello world", "hello world", "different text"]
        duplicates = detect_duplicates(texts)
        assert len(duplicates) >= 1
        assert duplicates[0][2] == pytest.approx(1.0)

    def test_no_duplicates(self) -> None:
        """Test with no duplicates."""
        texts = ["hello world", "different text", "another one"]
        duplicates = detect_duplicates(texts)
        assert len(duplicates) == 0

    def test_empty_list(self) -> None:
        """Test with empty list."""
        duplicates = detect_duplicates([])
        assert duplicates == []

    def test_single_item(self) -> None:
        """Test with single item."""
        duplicates = detect_duplicates(["hello"])
        assert duplicates == []

    def test_none_texts_raises_error(self) -> None:
        """Test that None texts raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be None"):
            detect_duplicates(None)  # type: ignore[arg-type]

    def test_with_ngram_method(self) -> None:
        """Test with n-gram deduplication method."""
        config = create_deduplication_config(
            method="ngram",
            similarity_threshold=0.5,
            ngram_size=2,
        )
        texts = [
            "the quick brown fox jumps",
            "the quick brown fox leaps",
            "completely different text here",
        ]
        duplicates = detect_duplicates(texts, config)
        # First two texts should have high n-gram overlap
        assert any(d[0] == 0 and d[1] == 1 for d in duplicates)

    def test_multiple_duplicates(self) -> None:
        """Test with multiple duplicate pairs."""
        texts = ["a", "a", "a", "b", "b"]
        duplicates = detect_duplicates(texts)
        # Should find duplicates among "a"s and among "b"s
        assert len(duplicates) >= 2


class TestCheckContamination:
    """Tests for check_contamination function."""

    def test_exact_contamination(self) -> None:
        """Test detection of exact contamination."""
        train = ["The quick brown fox", "Hello world"]
        test = ["The quick brown fox", "Different text"]
        contaminated = check_contamination(train, test)
        assert len(contaminated) >= 1

    def test_no_contamination(self) -> None:
        """Test with no contamination."""
        train = ["Hello world", "Test text"]
        test = ["Completely different", "Another one"]
        contaminated = check_contamination(train, test)
        assert len(contaminated) == 0

    def test_empty_train(self) -> None:
        """Test with empty training set."""
        contaminated = check_contamination([], ["test"])
        assert contaminated == []

    def test_empty_test(self) -> None:
        """Test with empty test set."""
        contaminated = check_contamination(["train"], [])
        assert contaminated == []

    def test_none_train_raises_error(self) -> None:
        """Test that None train_texts raises ValueError."""
        with pytest.raises(ValueError, match="train_texts cannot be None"):
            check_contamination(None, [])  # type: ignore[arg-type]

    def test_none_test_raises_error(self) -> None:
        """Test that None test_texts raises ValueError."""
        with pytest.raises(ValueError, match="test_texts cannot be None"):
            check_contamination([], None)  # type: ignore[arg-type]

    def test_with_custom_config(self) -> None:
        """Test with custom contamination config."""
        config = create_contamination_config(
            test_datasets=("test",),
            ngram_overlap_threshold=0.3,
            exact_match=False,
        )
        train = ["the quick brown fox jumps over the lazy dog"]
        test = ["the quick brown fox"]
        contaminated = check_contamination(train, test, config)
        # Should detect high word overlap
        assert len(contaminated) >= 1


class TestCalculatePerplexityScore:
    """Tests for calculate_perplexity_score function."""

    def test_basic_score(self) -> None:
        """Test basic perplexity calculation."""
        score = calculate_perplexity_score("Hello world, this is a test.")
        assert score > 0

    def test_empty_text(self) -> None:
        """Test with empty text."""
        score = calculate_perplexity_score("")
        assert score == 0.0

    def test_none_text_raises_error(self) -> None:
        """Test that None text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be None"):
            calculate_perplexity_score(None)  # type: ignore[arg-type]

    def test_invalid_vocabulary_size_raises_error(self) -> None:
        """Test that invalid vocabulary_size raises ValueError."""
        with pytest.raises(ValueError, match="vocabulary_size must be positive"):
            calculate_perplexity_score("test", vocabulary_size=0)

    def test_negative_vocabulary_size_raises_error(self) -> None:
        """Test that negative vocabulary_size raises ValueError."""
        with pytest.raises(ValueError, match="vocabulary_size must be positive"):
            calculate_perplexity_score("test", vocabulary_size=-1)

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_always_non_negative(self, text: str) -> None:
        """Test that perplexity is always non-negative."""
        score = calculate_perplexity_score(text)
        assert score >= 0


class TestFormatQualityStats:
    """Tests for format_quality_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = QualityStats(
            total_samples=10000,
            filtered_count=500,
            duplicate_count=200,
            quality_distribution={"high": 9000, "low": 1000},
        )
        formatted = format_quality_stats(stats)
        assert "10,000" in formatted or "10000" in formatted
        assert "500" in formatted
        assert "high" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_quality_stats(None)  # type: ignore[arg-type]

    def test_empty_distribution(self) -> None:
        """Test with empty quality distribution."""
        stats = QualityStats(
            total_samples=100,
            filtered_count=10,
            duplicate_count=5,
            quality_distribution={},
        )
        formatted = format_quality_stats(stats)
        assert "100" in formatted

    def test_zero_samples(self) -> None:
        """Test with zero samples."""
        stats = QualityStats(
            total_samples=0,
            filtered_count=0,
            duplicate_count=0,
            quality_distribution={},
        )
        formatted = format_quality_stats(stats)
        assert "0" in formatted


class TestGetRecommendedQualityConfig:
    """Tests for get_recommended_quality_config function."""

    def test_small_dataset(self) -> None:
        """Test recommendation for small dataset."""
        config = get_recommended_quality_config(1000)
        assert config.remove_outliers is True
        assert config.filter_strategy == FilterStrategy.THRESHOLD

    def test_medium_dataset(self) -> None:
        """Test recommendation for medium dataset."""
        config = get_recommended_quality_config(50000)
        assert len(config.metrics) >= 2

    def test_large_dataset(self) -> None:
        """Test recommendation for large dataset."""
        config = get_recommended_quality_config(1000000)
        assert config.filter_strategy == FilterStrategy.PERCENTILE

    def test_zero_size_raises_error(self) -> None:
        """Test that zero dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            get_recommended_quality_config(0)

    def test_negative_size_raises_error(self) -> None:
        """Test that negative dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            get_recommended_quality_config(-1)


class TestComputeQualityStats:
    """Tests for compute_quality_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic quality stats computation."""
        texts = ["Hello world, this is a longer text", "Test text here", "Short"]
        stats = compute_quality_stats(texts)
        assert stats.total_samples == 3

    def test_empty_texts(self) -> None:
        """Test with empty texts list."""
        stats = compute_quality_stats([])
        assert stats.total_samples == 0
        assert stats.filtered_count == 0

    def test_none_texts_raises_error(self) -> None:
        """Test that None texts raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be None"):
            compute_quality_stats(None)  # type: ignore[arg-type]

    def test_with_duplicates(self) -> None:
        """Test with duplicate texts."""
        texts = ["hello world", "hello world", "different"]
        stats = compute_quality_stats(texts)
        assert stats.duplicate_count >= 1

    def test_quality_distribution(self) -> None:
        """Test that quality distribution is computed."""
        texts = [
            "This is a longer text with many words to analyze",
            "Short",
            "word word word word word word word word",  # repetitive
        ]
        stats = compute_quality_stats(texts)
        dist = stats.quality_distribution
        assert "high" in dist or "medium" in dist or "low" in dist

    def test_with_custom_config(self) -> None:
        """Test with custom quality config."""
        config = create_quality_filter_config(
            metrics=("length",),
            thresholds={"length": 100},
        )
        texts = ["Hello world"]
        stats = compute_quality_stats(texts, config)
        assert stats.total_samples == 1


class TestPropertyBased:
    """Property-based tests for quality module."""

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=20))
    @settings(max_examples=20)
    def test_detect_duplicates_returns_valid_indices(self, texts: list[str]) -> None:
        """Test that detect_duplicates returns valid indices."""
        duplicates = detect_duplicates(texts)
        for idx1, idx2, sim in duplicates:
            assert 0 <= idx1 < len(texts)
            assert 0 <= idx2 < len(texts)
            assert idx1 < idx2
            assert 0.0 <= sim <= 1.0

    @given(st.integers(min_value=1, max_value=10000000))
    @settings(max_examples=20)
    def test_get_recommended_config_returns_valid_config(
        self, dataset_size: int
    ) -> None:
        """Test that get_recommended_quality_config returns valid config."""
        config = get_recommended_quality_config(dataset_size)
        assert len(config.metrics) > 0
        assert config.filter_strategy in FilterStrategy
        validate_quality_filter_config(config)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    def test_quality_scores_in_valid_range(self, text: str) -> None:
        """Test that quality scores are in valid ranges."""
        scores = calculate_text_quality_score(text)
        assert scores["length"] >= 0
        assert 0.0 <= scores["repetition"] <= 1.0
        assert 0.0 <= scores["language_score"] <= 1.0
        assert scores["perplexity"] >= 0
        assert scores["toxicity"] >= 0
