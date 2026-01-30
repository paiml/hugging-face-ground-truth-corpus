"""Tests for evaluation metrics functionality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.metrics import (
    VALID_AGGREGATION_METHODS,
    VALID_METRIC_TYPES,
    VALID_ROUGE_VARIANTS,
    AggregationMethod,
    BERTScoreConfig,
    BLEUConfig,
    ClassificationMetrics,
    MetricConfig,
    MetricResult,
    MetricType,
    ROUGEConfig,
    RougeVariant,
    aggregate_metrics,
    calculate_bertscore,
    calculate_bleu,
    calculate_perplexity,
    calculate_rouge,
    compute_accuracy,
    compute_classification_metrics,
    compute_f1,
    compute_mean_loss,
    compute_perplexity,
    compute_precision,
    compute_recall,
    create_bertscore_config,
    create_bleu_config,
    create_compute_metrics_fn,
    create_metric_config,
    create_rouge_config,
    format_metric_stats,
    get_aggregation_method,
    get_metric_type,
    get_recommended_metric_config,
    get_rouge_variant,
    list_aggregation_methods,
    list_metric_types,
    list_rouge_variants,
    validate_aggregation_method,
    validate_bertscore_config,
    validate_bleu_config,
    validate_metric_config,
    validate_metric_result,
    validate_metric_type,
    validate_rouge_config,
    validate_rouge_variant,
)


class TestMetricTypeEnum:
    """Tests for MetricType enum."""

    def test_values(self) -> None:
        """Test MetricType has expected values."""
        assert MetricType.BLEU.value == "bleu"
        assert MetricType.ROUGE.value == "rouge"
        assert MetricType.BERTSCORE.value == "bertscore"
        assert MetricType.METEOR.value == "meteor"
        assert MetricType.PERPLEXITY.value == "perplexity"
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.F1.value == "f1"
        assert MetricType.EXACT_MATCH.value == "exact_match"

    def test_valid_metric_types_frozenset(self) -> None:
        """Test VALID_METRIC_TYPES contains all enum values."""
        for mt in MetricType:
            assert mt.value in VALID_METRIC_TYPES


class TestRougeVariantEnum:
    """Tests for RougeVariant enum."""

    def test_values(self) -> None:
        """Test RougeVariant has expected values."""
        assert RougeVariant.ROUGE1.value == "rouge1"
        assert RougeVariant.ROUGE2.value == "rouge2"
        assert RougeVariant.ROUGEL.value == "rougeL"
        assert RougeVariant.ROUGELSUM.value == "rougeLsum"

    def test_valid_rouge_variants_frozenset(self) -> None:
        """Test VALID_ROUGE_VARIANTS contains all enum values."""
        for rv in RougeVariant:
            assert rv.value in VALID_ROUGE_VARIANTS


class TestAggregationMethodEnum:
    """Tests for AggregationMethod enum."""

    def test_values(self) -> None:
        """Test AggregationMethod has expected values."""
        assert AggregationMethod.MICRO.value == "micro"
        assert AggregationMethod.MACRO.value == "macro"
        assert AggregationMethod.WEIGHTED.value == "weighted"

    def test_valid_aggregation_methods_frozenset(self) -> None:
        """Test VALID_AGGREGATION_METHODS contains all enum values."""
        for am in AggregationMethod:
            assert am.value in VALID_AGGREGATION_METHODS


class TestBLEUConfig:
    """Tests for BLEUConfig dataclass."""

    def test_default_values(self) -> None:
        """Test BLEUConfig default values."""
        config = BLEUConfig()
        assert config.max_ngram == 4
        assert config.smoothing == "add_k"
        assert config.tokenizer == "word"

    def test_custom_values(self) -> None:
        """Test BLEUConfig with custom values."""
        config = BLEUConfig(max_ngram=2, smoothing="floor", tokenizer="char")
        assert config.max_ngram == 2
        assert config.smoothing == "floor"
        assert config.tokenizer == "char"

    def test_frozen(self) -> None:
        """Test BLEUConfig is immutable."""
        config = BLEUConfig()
        with pytest.raises(AttributeError):
            config.max_ngram = 2  # type: ignore[misc]


class TestROUGEConfig:
    """Tests for ROUGEConfig dataclass."""

    def test_default_values(self) -> None:
        """Test ROUGEConfig default values."""
        config = ROUGEConfig()
        assert RougeVariant.ROUGE1 in config.variants
        assert RougeVariant.ROUGE2 in config.variants
        assert config.use_stemmer is False
        assert config.split_summaries is True

    def test_custom_values(self) -> None:
        """Test ROUGEConfig with custom values."""
        config = ROUGEConfig(
            variants=(RougeVariant.ROUGEL,),
            use_stemmer=True,
            split_summaries=False,
        )
        assert len(config.variants) == 1
        assert config.use_stemmer is True
        assert config.split_summaries is False

    def test_frozen(self) -> None:
        """Test ROUGEConfig is immutable."""
        config = ROUGEConfig()
        with pytest.raises(AttributeError):
            config.use_stemmer = True  # type: ignore[misc]


class TestBERTScoreConfig:
    """Tests for BERTScoreConfig dataclass."""

    def test_default_values(self) -> None:
        """Test BERTScoreConfig default values."""
        config = BERTScoreConfig()
        assert config.model_name == "microsoft/deberta-xlarge-mnli"
        assert config.num_layers is None
        assert config.rescale_with_baseline is True

    def test_custom_values(self) -> None:
        """Test BERTScoreConfig with custom values."""
        config = BERTScoreConfig(
            model_name="bert-base-uncased",
            num_layers=8,
            rescale_with_baseline=False,
        )
        assert config.model_name == "bert-base-uncased"
        assert config.num_layers == 8
        assert config.rescale_with_baseline is False

    def test_frozen(self) -> None:
        """Test BERTScoreConfig is immutable."""
        config = BERTScoreConfig()
        with pytest.raises(AttributeError):
            config.model_name = "other"  # type: ignore[misc]


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""

    def test_basic_creation(self) -> None:
        """Test MetricConfig creation."""
        config = MetricConfig(metric_type=MetricType.BLEU)
        assert config.metric_type == MetricType.BLEU

    def test_with_sub_configs(self) -> None:
        """Test MetricConfig with sub-configurations."""
        bleu_cfg = BLEUConfig(max_ngram=2)
        config = MetricConfig(metric_type=MetricType.BLEU, bleu_config=bleu_cfg)
        assert config.bleu_config is not None
        assert config.bleu_config.max_ngram == 2

    def test_frozen(self) -> None:
        """Test MetricConfig is immutable."""
        config = MetricConfig(metric_type=MetricType.BLEU)
        with pytest.raises(AttributeError):
            config.metric_type = MetricType.ROUGE  # type: ignore[misc]


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test MetricResult with just score."""
        result = MetricResult(score=0.85)
        assert result.score == pytest.approx(0.85)
        assert result.precision is None
        assert result.recall is None
        assert result.f1 is None
        assert result.confidence is None

    def test_full_creation(self) -> None:
        """Test MetricResult with all fields."""
        result = MetricResult(
            score=0.85,
            precision=0.9,
            recall=0.8,
            f1=0.85,
            confidence=(0.8, 0.9),
        )
        assert result.score == pytest.approx(0.85)
        assert result.precision == pytest.approx(0.9)
        assert result.recall == pytest.approx(0.8)
        assert result.f1 == pytest.approx(0.85)
        assert result.confidence == (0.8, 0.9)

    def test_frozen(self) -> None:
        """Test MetricResult is immutable."""
        result = MetricResult(score=0.85)
        with pytest.raises(AttributeError):
            result.score = 0.9  # type: ignore[misc]


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating ClassificationMetrics instance."""
        metrics = ClassificationMetrics(
            accuracy=0.9, precision=0.85, recall=0.88, f1=0.865
        )
        assert metrics.accuracy == pytest.approx(0.9)
        assert metrics.precision == pytest.approx(0.85)
        assert metrics.recall == pytest.approx(0.88)
        assert metrics.f1 == pytest.approx(0.865)

    def test_frozen(self) -> None:
        """Test that ClassificationMetrics is immutable."""
        metrics = ClassificationMetrics(
            accuracy=0.9, precision=0.85, recall=0.88, f1=0.865
        )
        with pytest.raises(AttributeError):
            metrics.accuracy = 0.5  # type: ignore[misc]


class TestCreateBleuConfig:
    """Tests for create_bleu_config factory function."""

    def test_defaults(self) -> None:
        """Test default configuration."""
        config = create_bleu_config()
        assert config.max_ngram == 4
        assert config.smoothing == "add_k"
        assert config.tokenizer == "word"

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = create_bleu_config(max_ngram=2, smoothing="floor", tokenizer="char")
        assert config.max_ngram == 2
        assert config.smoothing == "floor"
        assert config.tokenizer == "char"


class TestCreateRougeConfig:
    """Tests for create_rouge_config factory function."""

    def test_defaults(self) -> None:
        """Test default configuration."""
        config = create_rouge_config()
        assert RougeVariant.ROUGE1 in config.variants
        assert config.use_stemmer is False

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = create_rouge_config(
            variants=(RougeVariant.ROUGEL,),
            use_stemmer=True,
        )
        assert RougeVariant.ROUGEL in config.variants
        assert config.use_stemmer is True


class TestCreateBertscoreConfig:
    """Tests for create_bertscore_config factory function."""

    def test_defaults(self) -> None:
        """Test default configuration."""
        config = create_bertscore_config()
        assert config.model_name == "microsoft/deberta-xlarge-mnli"
        assert config.num_layers is None

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = create_bertscore_config(model_name="bert-base-uncased", num_layers=8)
        assert config.model_name == "bert-base-uncased"
        assert config.num_layers == 8


class TestCreateMetricConfig:
    """Tests for create_metric_config factory function."""

    def test_bleu_auto_config(self) -> None:
        """Test BLEU auto-creates sub-config."""
        config = create_metric_config(MetricType.BLEU)
        assert config.metric_type == MetricType.BLEU
        assert config.bleu_config is not None

    def test_rouge_auto_config(self) -> None:
        """Test ROUGE auto-creates sub-config."""
        config = create_metric_config(MetricType.ROUGE)
        assert config.metric_type == MetricType.ROUGE
        assert config.rouge_config is not None

    def test_bertscore_auto_config(self) -> None:
        """Test BERTScore auto-creates sub-config."""
        config = create_metric_config(MetricType.BERTSCORE)
        assert config.metric_type == MetricType.BERTSCORE
        assert config.bertscore_config is not None

    def test_custom_sub_config(self) -> None:
        """Test custom sub-config."""
        bleu_cfg = BLEUConfig(max_ngram=2)
        config = create_metric_config(MetricType.BLEU, bleu_config=bleu_cfg)
        assert config.bleu_config.max_ngram == 2


class TestValidateBleuConfig:
    """Tests for validate_bleu_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes."""
        config = create_bleu_config()
        validate_bleu_config(config)  # Should not raise

    def test_none_config_raises(self) -> None:
        """Test None config raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_bleu_config(None)  # type: ignore[arg-type]

    def test_invalid_max_ngram_raises(self) -> None:
        """Test invalid max_ngram raises."""
        config = BLEUConfig(max_ngram=0)
        with pytest.raises(ValueError, match="must be positive"):
            validate_bleu_config(config)

    def test_invalid_smoothing_raises(self) -> None:
        """Test invalid smoothing raises."""
        config = BLEUConfig(smoothing="invalid")
        with pytest.raises(ValueError, match="must be one of"):
            validate_bleu_config(config)

    def test_empty_tokenizer_raises(self) -> None:
        """Test empty tokenizer raises."""
        config = BLEUConfig(tokenizer="")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_bleu_config(config)


class TestValidateRougeConfig:
    """Tests for validate_rouge_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes."""
        config = create_rouge_config()
        validate_rouge_config(config)  # Should not raise

    def test_none_config_raises(self) -> None:
        """Test None config raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_rouge_config(None)  # type: ignore[arg-type]

    def test_empty_variants_raises(self) -> None:
        """Test empty variants raises."""
        config = ROUGEConfig(variants=())
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_rouge_config(config)


class TestValidateBertscoreConfig:
    """Tests for validate_bertscore_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes."""
        config = create_bertscore_config()
        validate_bertscore_config(config)  # Should not raise

    def test_none_config_raises(self) -> None:
        """Test None config raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_bertscore_config(None)  # type: ignore[arg-type]

    def test_empty_model_name_raises(self) -> None:
        """Test empty model_name raises."""
        config = BERTScoreConfig(model_name="")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_bertscore_config(config)

    def test_invalid_num_layers_raises(self) -> None:
        """Test invalid num_layers raises."""
        config = BERTScoreConfig(num_layers=0)
        with pytest.raises(ValueError, match="must be positive"):
            validate_bertscore_config(config)


class TestValidateMetricConfig:
    """Tests for validate_metric_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes."""
        config = create_metric_config(MetricType.BLEU)
        validate_metric_config(config)  # Should not raise

    def test_none_config_raises(self) -> None:
        """Test None config raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_metric_config(None)  # type: ignore[arg-type]

    def test_validates_sub_configs(self) -> None:
        """Test that sub-configs are validated."""
        # Valid config with ROUGE sub-config
        config_rouge = create_metric_config(MetricType.ROUGE)
        validate_metric_config(config_rouge)  # Should not raise

        # Valid config with BERTScore sub-config
        config_bert = create_metric_config(MetricType.BERTSCORE)
        validate_metric_config(config_bert)  # Should not raise


class TestValidateMetricResult:
    """Tests for validate_metric_result function."""

    def test_valid_result(self) -> None:
        """Test valid result passes."""
        result = MetricResult(score=0.85)
        validate_metric_result(result)  # Should not raise

    def test_none_result_raises(self) -> None:
        """Test None result raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_metric_result(None)  # type: ignore[arg-type]

    def test_nan_score_raises(self) -> None:
        """Test NaN score raises."""
        result = MetricResult(score=float("nan"))
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_metric_result(result)


class TestListMetricTypes:
    """Tests for list_metric_types function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        types = list_metric_types()
        assert types == sorted(types)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_metric_types()
        assert "bleu" in types
        assert "rouge" in types
        assert "bertscore" in types


class TestValidateMetricType:
    """Tests for validate_metric_type function."""

    def test_valid_types(self) -> None:
        """Test valid types return True."""
        assert validate_metric_type("bleu") is True
        assert validate_metric_type("rouge") is True

    def test_invalid_types(self) -> None:
        """Test invalid types return False."""
        assert validate_metric_type("invalid") is False
        assert validate_metric_type("") is False


class TestGetMetricType:
    """Tests for get_metric_type function."""

    def test_valid_types(self) -> None:
        """Test valid types return enum."""
        assert get_metric_type("bleu") == MetricType.BLEU
        assert get_metric_type("rouge") == MetricType.ROUGE

    def test_invalid_type_raises(self) -> None:
        """Test invalid type raises."""
        with pytest.raises(ValueError, match="invalid metric type"):
            get_metric_type("invalid")


class TestListRougeVariants:
    """Tests for list_rouge_variants function."""

    def test_returns_list(self) -> None:
        """Test returns list."""
        variants = list_rouge_variants()
        assert isinstance(variants, list)
        assert "rouge1" in variants
        assert "rougeL" in variants


class TestValidateRougeVariant:
    """Tests for validate_rouge_variant function."""

    def test_valid_variants(self) -> None:
        """Test valid variants return True."""
        assert validate_rouge_variant("rouge1") is True
        assert validate_rouge_variant("rougeL") is True

    def test_invalid_variants(self) -> None:
        """Test invalid variants return False."""
        assert validate_rouge_variant("invalid") is False
        assert validate_rouge_variant("") is False


class TestGetRougeVariant:
    """Tests for get_rouge_variant function."""

    def test_valid_variants(self) -> None:
        """Test valid variants return enum."""
        assert get_rouge_variant("rouge1") == RougeVariant.ROUGE1
        assert get_rouge_variant("rougeL") == RougeVariant.ROUGEL

    def test_invalid_variant_raises(self) -> None:
        """Test invalid variant raises."""
        with pytest.raises(ValueError, match="invalid ROUGE variant"):
            get_rouge_variant("invalid")


class TestListAggregationMethods:
    """Tests for list_aggregation_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        methods = list_aggregation_methods()
        assert methods == sorted(methods)

    def test_contains_expected_methods(self) -> None:
        """Test contains expected methods."""
        methods = list_aggregation_methods()
        assert "micro" in methods
        assert "macro" in methods
        assert "weighted" in methods


class TestValidateAggregationMethod:
    """Tests for validate_aggregation_method function."""

    def test_valid_methods(self) -> None:
        """Test valid methods return True."""
        assert validate_aggregation_method("micro") is True
        assert validate_aggregation_method("macro") is True

    def test_invalid_methods(self) -> None:
        """Test invalid methods return False."""
        assert validate_aggregation_method("invalid") is False
        assert validate_aggregation_method("") is False


class TestGetAggregationMethod:
    """Tests for get_aggregation_method function."""

    def test_valid_methods(self) -> None:
        """Test valid methods return enum."""
        assert get_aggregation_method("micro") == AggregationMethod.MICRO
        assert get_aggregation_method("macro") == AggregationMethod.MACRO

    def test_invalid_method_raises(self) -> None:
        """Test invalid method raises."""
        with pytest.raises(ValueError, match="invalid aggregation method"):
            get_aggregation_method("invalid")


class TestCalculateBleu:
    """Tests for calculate_bleu function."""

    def test_perfect_match(self) -> None:
        """Test perfect match gives high score."""
        cands = ["the cat sat on the mat"]
        refs = [["the cat sat on the mat"]]
        result = calculate_bleu(cands, refs)
        assert result.score > 0.9

    def test_no_match(self) -> None:
        """Test no match gives low score."""
        cands = ["the dog jumped over the fence"]
        refs = [["a quick brown fox"]]
        result = calculate_bleu(cands, refs)
        assert result.score < 0.3

    def test_partial_match(self) -> None:
        """Test partial match gives moderate score."""
        cands = ["the cat sat on the mat"]
        refs = [["the cat was on the mat"]]
        result = calculate_bleu(cands, refs)
        assert 0.3 < result.score < 0.9

    def test_multiple_references(self) -> None:
        """Test with multiple references."""
        cands = ["the cat sat on the mat"]
        refs = [["a cat sat on a mat", "the cat was on the mat"]]
        result = calculate_bleu(cands, refs)
        assert result.score > 0.0

    def test_empty_candidates_raises(self) -> None:
        """Test empty candidates raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_bleu([], [[]])

    def test_none_candidates_raises(self) -> None:
        """Test None candidates raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_bleu(None, [[]])  # type: ignore[arg-type]

    def test_none_references_raises(self) -> None:
        """Test None references raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_bleu(["test"], None)  # type: ignore[arg-type]

    def test_empty_references_raises(self) -> None:
        """Test empty references raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_bleu(["test"], [])

    def test_length_mismatch_raises(self) -> None:
        """Test length mismatch raises."""
        with pytest.raises(ValueError, match="same length"):
            calculate_bleu(["a", "b"], [["a"]])

    def test_custom_config(self) -> None:
        """Test with custom config."""
        cands = ["the cat sat on the mat"]
        refs = [["the cat sat on the mat"]]
        config = BLEUConfig(max_ngram=2, smoothing="floor")
        result = calculate_bleu(cands, refs, config)
        assert result.score > 0.0

    def test_no_smoothing(self) -> None:
        """Test with no smoothing."""
        cands = ["the cat sat"]
        refs = [["a dog ran"]]  # No overlap at higher n-grams
        config = BLEUConfig(max_ngram=4, smoothing="none")
        result = calculate_bleu(cands, refs, config)
        # With no smoothing, zero precision for some n-grams gives 0.0
        assert result.score >= 0.0

    def test_floor_smoothing_short_text(self) -> None:
        """Test floor smoothing with short text."""
        cands = ["a b"]
        refs = [["c d e"]]  # Different text
        config = BLEUConfig(max_ngram=4, smoothing="floor")
        result = calculate_bleu(cands, refs, config)
        assert result.score >= 0.0

    def test_longer_candidate_than_reference(self) -> None:
        """Test brevity penalty when candidate is longer."""
        cands = ["this is a much longer sentence than the reference"]
        refs = [["short ref"]]
        result = calculate_bleu(cands, refs)
        assert result.score >= 0.0


class TestCalculateRouge:
    """Tests for calculate_rouge function."""

    def test_perfect_match(self) -> None:
        """Test perfect match gives high scores."""
        cands = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        results = calculate_rouge(cands, refs)
        assert "rouge1" in results
        assert results["rouge1"].score > 0.9

    def test_partial_match(self) -> None:
        """Test partial match gives moderate scores."""
        cands = ["the cat sat on the mat"]
        refs = ["the cat was on the mat"]
        results = calculate_rouge(cands, refs)
        assert results["rouge1"].score > 0.5

    def test_empty_candidates_raises(self) -> None:
        """Test empty candidates raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_rouge([], [])

    def test_none_candidates_raises(self) -> None:
        """Test None candidates raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_rouge(None, [])  # type: ignore[arg-type]

    def test_none_references_raises(self) -> None:
        """Test None references raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_rouge(["test"], None)  # type: ignore[arg-type]

    def test_empty_references_raises(self) -> None:
        """Test empty references raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_rouge(["test"], [])

    def test_length_mismatch_raises(self) -> None:
        """Test length mismatch raises."""
        with pytest.raises(ValueError, match="same length"):
            calculate_rouge(["a", "b"], ["a"])

    def test_multiple_variants(self) -> None:
        """Test with multiple variants."""
        cands = ["the quick brown fox"]
        refs = ["the quick brown dog"]
        config = ROUGEConfig(
            variants=(RougeVariant.ROUGE1, RougeVariant.ROUGE2, RougeVariant.ROUGEL)
        )
        results = calculate_rouge(cands, refs, config)
        assert "rouge1" in results
        assert "rouge2" in results
        assert "rougeL" in results

    def test_rougelsum_variant(self) -> None:
        """Test with rougeLsum variant."""
        cands = ["the quick brown fox"]
        refs = ["the quick brown dog"]
        config = ROUGEConfig(variants=(RougeVariant.ROUGELSUM,))
        results = calculate_rouge(cands, refs, config)
        assert "rougeLsum" in results

    def test_result_has_precision_recall_f1(self) -> None:
        """Test result includes precision, recall, f1."""
        cands = ["the cat sat"]
        refs = ["the cat"]
        results = calculate_rouge(cands, refs)
        result = results["rouge1"]
        assert result.precision is not None
        assert result.recall is not None
        assert result.f1 is not None

    def test_no_overlap(self) -> None:
        """Test with no word overlap."""
        cands = ["apple banana cherry"]
        refs = ["dog elephant fox"]
        results = calculate_rouge(cands, refs)
        assert results["rouge1"].score == 0.0


class TestCalculateBertscore:
    """Tests for calculate_bertscore function."""

    def test_perfect_match(self) -> None:
        """Test perfect match gives high score."""
        cands = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = calculate_bertscore(cands, refs)
        assert result.score > 0.5

    def test_partial_match(self) -> None:
        """Test partial match gives moderate score."""
        cands = ["the cat sat on the mat"]
        refs = ["the dog was sitting on the rug"]
        result = calculate_bertscore(cands, refs)
        # Should have some overlap
        assert result.score >= 0.0

    def test_empty_candidates_raises(self) -> None:
        """Test empty candidates raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_bertscore([], [])

    def test_none_candidates_raises(self) -> None:
        """Test None candidates raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_bertscore(None, [])  # type: ignore[arg-type]

    def test_none_references_raises(self) -> None:
        """Test None references raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_bertscore(["test"], None)  # type: ignore[arg-type]

    def test_empty_references_raises(self) -> None:
        """Test empty references raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_bertscore(["test"], [])

    def test_length_mismatch_raises(self) -> None:
        """Test length mismatch raises."""
        with pytest.raises(ValueError, match="same length"):
            calculate_bertscore(["a", "b"], ["a"])

    def test_result_has_precision_recall_f1(self) -> None:
        """Test result includes precision, recall, f1."""
        cands = ["the cat sat"]
        refs = ["the cat"]
        result = calculate_bertscore(cands, refs)
        assert result.precision is not None
        assert result.recall is not None
        assert result.f1 is not None

    def test_without_rescaling(self) -> None:
        """Test without baseline rescaling."""
        cands = ["the cat sat"]
        refs = ["the cat sat"]
        config = BERTScoreConfig(rescale_with_baseline=False)
        result = calculate_bertscore(cands, refs, config)
        assert result.score > 0.0

    def test_no_overlap(self) -> None:
        """Test with no word overlap."""
        cands = ["apple banana cherry"]
        refs = ["dog elephant fox"]
        result = calculate_bertscore(cands, refs)
        # No overlap means 0 score
        assert result.score == 0.0


class TestCalculatePerplexity:
    """Tests for calculate_perplexity function."""

    def test_from_loss(self) -> None:
        """Test perplexity from loss."""
        result = calculate_perplexity(loss=2.0)
        assert 7.38 < result.score < 7.40

    def test_from_log_probs(self) -> None:
        """Test perplexity from log probabilities."""
        result = calculate_perplexity(log_probs=[-1.0, -2.0, -1.5])
        assert result.score > 1.0

    def test_zero_loss(self) -> None:
        """Test zero loss gives perplexity of 1."""
        result = calculate_perplexity(loss=0.0)
        assert result.score == pytest.approx(1.0)

    def test_no_input_raises(self) -> None:
        """Test no input raises."""
        with pytest.raises(ValueError, match="must be provided"):
            calculate_perplexity()

    def test_negative_loss_raises(self) -> None:
        """Test negative loss raises."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_perplexity(loss=-1.0)

    def test_overflow_loss_raises(self) -> None:
        """Test overflow loss raises."""
        with pytest.raises(ValueError, match="too large"):
            calculate_perplexity(loss=1000.0)

    def test_empty_log_probs_raises(self) -> None:
        """Test empty log_probs raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_perplexity(log_probs=[])

    def test_overflow_log_probs_raises(self) -> None:
        """Test overflow from log_probs raises."""
        with pytest.raises(ValueError, match="too large"):
            calculate_perplexity(log_probs=[-800.0, -800.0, -800.0])


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_macro_aggregation(self) -> None:
        """Test macro aggregation."""
        r1 = MetricResult(score=0.8)
        r2 = MetricResult(score=0.9)
        agg = aggregate_metrics([r1, r2], method=AggregationMethod.MACRO)
        assert agg.score == pytest.approx(0.85)

    def test_weighted_aggregation(self) -> None:
        """Test weighted aggregation."""
        r1 = MetricResult(score=0.8)
        r2 = MetricResult(score=0.9)
        agg = aggregate_metrics(
            [r1, r2],
            method=AggregationMethod.WEIGHTED,
            weights=[0.3, 0.7],
        )
        expected = 0.8 * 0.3 + 0.9 * 0.7
        assert agg.score == pytest.approx(expected)

    def test_micro_aggregation(self) -> None:
        """Test micro aggregation (falls back to macro for simplicity)."""
        r1 = MetricResult(score=0.8)
        r2 = MetricResult(score=0.9)
        agg = aggregate_metrics([r1, r2], method=AggregationMethod.MICRO)
        # Micro falls back to macro
        assert agg.score == pytest.approx(0.85)

    def test_weighted_without_weights_uses_uniform(self) -> None:
        """Test weighted aggregation without weights uses uniform."""
        r1 = MetricResult(score=0.8)
        r2 = MetricResult(score=0.9)
        agg = aggregate_metrics([r1, r2], method=AggregationMethod.WEIGHTED)
        # Uniform weights = macro
        assert agg.score == pytest.approx(0.85)

    def test_aggregates_precision_recall_f1(self) -> None:
        """Test aggregates precision, recall, f1."""
        r1 = MetricResult(score=0.8, precision=0.9, recall=0.7, f1=0.8)
        r2 = MetricResult(score=0.9, precision=0.85, recall=0.95, f1=0.9)
        agg = aggregate_metrics([r1, r2])
        assert agg.precision is not None
        assert agg.precision == pytest.approx(0.875)

    def test_partial_precision_recall(self) -> None:
        """Test when only some results have precision/recall."""
        r1 = MetricResult(score=0.8, precision=0.9, recall=0.7, f1=0.8)
        r2 = MetricResult(score=0.9)  # No precision/recall/f1
        agg = aggregate_metrics([r1, r2])
        # Should be None since not all have precision
        assert agg.precision is None

    def test_empty_results_raises(self) -> None:
        """Test empty results raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_metrics([])

    def test_none_results_raises(self) -> None:
        """Test None results raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            aggregate_metrics(None)  # type: ignore[arg-type]

    def test_weights_length_mismatch_raises(self) -> None:
        """Test weights length mismatch raises."""
        r1 = MetricResult(score=0.8)
        r2 = MetricResult(score=0.9)
        with pytest.raises(ValueError, match="same length"):
            aggregate_metrics([r1, r2], weights=[0.5])


class TestFormatMetricStats:
    """Tests for format_metric_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic formatting."""
        results = {
            "bleu": MetricResult(score=0.45),
        }
        formatted = format_metric_stats(results)
        assert "BLEU" in formatted
        assert "0.45" in formatted

    def test_with_precision_recall_f1(self) -> None:
        """Test formatting with precision, recall, f1."""
        results = {
            "rouge1": MetricResult(score=0.65, precision=0.7, recall=0.6, f1=0.65),
        }
        formatted = format_metric_stats(results)
        assert "Precision" in formatted
        assert "Recall" in formatted
        assert "F1" in formatted

    def test_with_confidence_interval(self) -> None:
        """Test formatting with confidence interval."""
        results = {
            "metric": MetricResult(score=0.65, confidence=(0.6, 0.7)),
        }
        formatted = format_metric_stats(results)
        assert "Confidence" in formatted
        assert "0.6" in formatted
        assert "0.7" in formatted

    def test_none_results_raises(self) -> None:
        """Test None results raises."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_metric_stats(None)  # type: ignore[arg-type]

    def test_empty_results(self) -> None:
        """Test empty results."""
        formatted = format_metric_stats({})
        assert "Metric Results" in formatted


class TestGetRecommendedMetricConfig:
    """Tests for get_recommended_metric_config function."""

    def test_translation_task(self) -> None:
        """Test translation task returns BLEU."""
        config = get_recommended_metric_config("translation")
        assert config.metric_type == MetricType.BLEU

    def test_summarization_task(self) -> None:
        """Test summarization task returns ROUGE."""
        config = get_recommended_metric_config("summarization")
        assert config.metric_type == MetricType.ROUGE

    def test_qa_task(self) -> None:
        """Test QA task returns exact match."""
        config = get_recommended_metric_config("qa")
        assert config.metric_type == MetricType.EXACT_MATCH

    def test_generation_task(self) -> None:
        """Test generation task returns perplexity."""
        config = get_recommended_metric_config("generation")
        assert config.metric_type == MetricType.PERPLEXITY

    def test_similarity_task(self) -> None:
        """Test similarity task returns BERTScore."""
        config = get_recommended_metric_config("similarity")
        assert config.metric_type == MetricType.BERTSCORE

    def test_empty_task_raises(self) -> None:
        """Test empty task raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_recommended_metric_config("")

    def test_unknown_task_defaults_to_bleu(self) -> None:
        """Test unknown task defaults to BLEU."""
        config = get_recommended_metric_config("unknown")
        assert config.metric_type == MetricType.BLEU


# Original classification metrics tests


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        """Test 100% accuracy."""
        assert compute_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)

    def test_zero_accuracy(self) -> None:
        """Test 0% accuracy."""
        assert compute_accuracy([1, 1, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.0)

    def test_partial_accuracy(self) -> None:
        """Test partial accuracy."""
        assert compute_accuracy([1, 0, 1, 1], [1, 0, 1, 0]) == pytest.approx(0.75)

    def test_empty_predictions_raises_error(self) -> None:
        """Test that empty predictions raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_accuracy([], [1])

    def test_empty_labels_raises_error(self) -> None:
        """Test that empty labels raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_accuracy([1], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_accuracy([1, 0], [1, 0, 1])

    @given(st.lists(st.integers(0, 1), min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_accuracy_range(self, labels: list[int]) -> None:
        """Test that accuracy is always between 0 and 1."""
        predictions = labels  # Perfect predictions
        acc = compute_accuracy(predictions, labels)
        assert 0.0 <= acc <= 1.0


class TestComputePrecision:
    """Tests for compute_precision function."""

    def test_perfect_precision(self) -> None:
        """Test 100% precision."""
        assert compute_precision([1, 1], [1, 1]) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        """Test 0% precision (all predictions wrong)."""
        assert compute_precision([1, 1], [0, 0]) == pytest.approx(0.0)

    def test_no_positive_predictions(self) -> None:
        """Test precision when no positive predictions made."""
        assert compute_precision([0, 0, 0], [1, 1, 1]) == pytest.approx(0.0)

    def test_partial_precision(self) -> None:
        """Test partial precision."""
        # 2 true positives, 1 false positive = 2/3
        result = compute_precision([1, 1, 1, 0], [1, 1, 0, 0])
        assert abs(result - 2 / 3) < 1e-10

    def test_empty_raises_error(self) -> None:
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_precision([], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_precision([1], [1, 0])


class TestComputeRecall:
    """Tests for compute_recall function."""

    def test_perfect_recall(self) -> None:
        """Test 100% recall."""
        assert compute_recall([1, 1], [1, 1]) == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        """Test 0% recall (missed all positives)."""
        assert compute_recall([0, 0], [1, 1]) == pytest.approx(0.0)

    def test_no_actual_positives(self) -> None:
        """Test recall when no actual positives exist."""
        assert compute_recall([1, 1], [0, 0]) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """Test partial recall."""
        # 1 true positive out of 2 actual positives = 0.5
        assert compute_recall([1, 0], [1, 1]) == pytest.approx(0.5)

    def test_empty_raises_error(self) -> None:
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_recall([], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_recall([1, 0], [1, 0, 1])


class TestComputeF1:
    """Tests for compute_f1 function."""

    def test_perfect_f1(self) -> None:
        """Test perfect F1 score."""
        assert compute_f1([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)

    def test_zero_f1(self) -> None:
        """Test F1 of 0 when precision and recall are both 0."""
        assert compute_f1([0, 0, 0], [1, 1, 1]) == pytest.approx(0.0)

    def test_f1_calculation(self) -> None:
        """Test F1 calculation: 2 * p * r / (p + r)."""
        # precision = 2/3, recall = 1.0
        # f1 = 2 * (2/3) * 1 / ((2/3) + 1) = 4/3 / 5/3 = 4/5 = 0.8
        result = compute_f1([1, 1, 0, 1], [1, 0, 0, 1])
        assert result == pytest.approx(0.8)

    def test_empty_raises_error(self) -> None:
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_f1([], [])


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics function."""

    def test_returns_all_metrics(self) -> None:
        """Test that all metrics are returned."""
        metrics = compute_classification_metrics([1, 1, 0, 1], [1, 0, 0, 1])
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == pytest.approx(0.75)
        assert abs(metrics.precision - 2 / 3) < 1e-10
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(0.8)

    def test_perfect_metrics(self) -> None:
        """Test perfect classification."""
        metrics = compute_classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)

    def test_custom_positive_label(self) -> None:
        """Test with custom positive label."""
        # Using 0 as positive label
        metrics = compute_classification_metrics([0, 0, 1], [0, 1, 1], positive_label=0)
        assert metrics.accuracy == pytest.approx(2 / 3)
        assert metrics.precision == pytest.approx(0.5)  # 1 TP, 1 FP
        assert metrics.recall == pytest.approx(1.0)  # 1 TP, 0 FN


class TestComputePerplexitySimple:
    """Tests for compute_perplexity function (simple version)."""

    def test_zero_loss(self) -> None:
        """Test perplexity with zero loss."""
        assert compute_perplexity(0.0) == pytest.approx(1.0)

    def test_loss_one(self) -> None:
        """Test perplexity with loss of 1."""
        result = compute_perplexity(1.0)
        assert abs(result - 2.718281828) < 0.001

    def test_loss_two(self) -> None:
        """Test perplexity with loss of 2."""
        result = compute_perplexity(2.0)
        assert abs(result - 7.389056099) < 0.001

    def test_negative_loss_raises_error(self) -> None:
        """Test that negative loss raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            compute_perplexity(-1.0)

    def test_overflow_loss_raises_error(self) -> None:
        """Test that very large loss raises ValueError."""
        with pytest.raises(ValueError, match="too large"):
            compute_perplexity(1000.0)

    @given(st.floats(min_value=0.0, max_value=100.0))
    @settings(max_examples=50)
    def test_perplexity_always_positive(self, loss: float) -> None:
        """Test that perplexity is always positive."""
        result = compute_perplexity(loss)
        assert result > 0


class TestComputeMeanLoss:
    """Tests for compute_mean_loss function."""

    def test_single_value(self) -> None:
        """Test mean of single value."""
        assert compute_mean_loss([5.0]) == pytest.approx(5.0)

    def test_multiple_values(self) -> None:
        """Test mean of multiple values."""
        assert compute_mean_loss([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_mean_loss([])

    @given(st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_mean_in_range(self, losses: list[float]) -> None:
        """Test that mean is within the range of inputs."""
        result = compute_mean_loss(losses)
        # Use small epsilon for floating point comparison
        eps = 1e-10
        assert min(losses) - eps <= result <= max(losses) + eps


class TestCreateComputeMetricsFn:
    """Tests for create_compute_metrics_fn function."""

    def test_returns_callable(self) -> None:
        """Test that function returns a callable."""
        fn = create_compute_metrics_fn()
        assert callable(fn)

    def test_computes_metrics_from_arrays(self) -> None:
        """Test computing metrics from numpy arrays."""
        fn = create_compute_metrics_fn()
        predictions = np.array([1, 0, 1, 1])
        labels = np.array([1, 0, 1, 0])

        result = fn((predictions, labels))

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert result["accuracy"] == pytest.approx(0.75)

    def test_handles_logits(self) -> None:
        """Test that function handles logits (takes argmax)."""
        fn = create_compute_metrics_fn()
        # Logits: [[0.1, 0.9], [0.8, 0.2]] -> predictions [1, 0]
        logits = np.array([[0.1, 0.9], [0.8, 0.2]])
        labels = np.array([1, 0])

        result = fn((logits, labels))

        assert result["accuracy"] == pytest.approx(1.0)

    def test_custom_positive_label(self) -> None:
        """Test with custom positive label."""
        fn = create_compute_metrics_fn(positive_label=0)
        predictions = np.array([0, 0, 1])
        labels = np.array([0, 1, 1])

        result = fn((predictions, labels))

        # With positive_label=0: TP=1, FP=1, FN=0
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(1.0)
