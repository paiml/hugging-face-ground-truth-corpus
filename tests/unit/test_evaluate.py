"""Tests for HuggingFace evaluate utilities."""

from __future__ import annotations

import pytest

from hf_gtc.evaluation.evaluate import (
    CLASSIFICATION_METRICS,
    GENERATION_METRICS,
    REGRESSION_METRICS,
    VALID_AGGREGATION_METHODS,
    VALID_METRIC_TYPES,
    CombinedMetricConfig,
    MetricConfig,
    MetricResult,
    MetricType,
    aggregate_results,
    compute_accuracy,
    compute_f1,
    compute_mae,
    compute_mse,
    compute_precision,
    compute_recall,
    create_combined_config,
    create_metric_config,
    get_metric_type,
    get_recommended_metrics,
    list_classification_metrics,
    list_generation_metrics,
    list_regression_metrics,
    validate_metric_config,
)


class TestMetricType:
    """Tests for MetricType enum."""

    def test_classification_value(self) -> None:
        """Test CLASSIFICATION type value."""
        assert MetricType.CLASSIFICATION.value == "classification"

    def test_regression_value(self) -> None:
        """Test REGRESSION type value."""
        assert MetricType.REGRESSION.value == "regression"

    def test_generation_value(self) -> None:
        """Test GENERATION type value."""
        assert MetricType.GENERATION.value == "generation"

    def test_all_types_in_valid_set(self) -> None:
        """Test all enum values are in VALID_METRIC_TYPES."""
        for metric_type in MetricType:
            assert metric_type.value in VALID_METRIC_TYPES


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = MetricConfig(
            metric_name="accuracy",
            metric_type=MetricType.CLASSIFICATION,
            average=None,
            num_labels=None,
            kwargs=None,
        )
        assert config.metric_name == "accuracy"
        assert config.metric_type == MetricType.CLASSIFICATION
        assert config.average is None

    def test_with_average(self) -> None:
        """Test config with average setting."""
        config = MetricConfig(
            metric_name="f1",
            metric_type=MetricType.CLASSIFICATION,
            average="macro",
            num_labels=3,
            kwargs=None,
        )
        assert config.average == "macro"
        assert config.num_labels == 3

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = MetricConfig(
            metric_name="accuracy",
            metric_type=MetricType.CLASSIFICATION,
            average=None,
            num_labels=None,
            kwargs=None,
        )
        with pytest.raises(AttributeError):
            config.metric_name = "f1"  # type: ignore[misc]


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = MetricResult(
            metric_name="accuracy",
            value=0.95,
            details=None,
        )
        assert result.metric_name == "accuracy"
        assert result.value == pytest.approx(0.95)
        assert result.details is None

    def test_with_details(self) -> None:
        """Test result with details."""
        result = MetricResult(
            metric_name="f1",
            value=0.85,
            details={"per_class": [0.9, 0.8, 0.85]},
        )
        assert result.details is not None
        assert "per_class" in result.details

    def test_frozen(self) -> None:
        """Test that result is immutable."""
        result = MetricResult(
            metric_name="accuracy",
            value=0.95,
            details=None,
        )
        with pytest.raises(AttributeError):
            result.value = 0.9  # type: ignore[misc]


class TestCombinedMetricConfig:
    """Tests for CombinedMetricConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic combined config creation."""
        m1 = MetricConfig("accuracy", MetricType.CLASSIFICATION, None, None, None)
        config = CombinedMetricConfig(
            metrics=(m1,),
            aggregation="mean",
            weights=None,
        )
        assert len(config.metrics) == 1
        assert config.aggregation == "mean"

    def test_with_weights(self) -> None:
        """Test combined config with weights."""
        m1 = MetricConfig("accuracy", MetricType.CLASSIFICATION, None, None, None)
        m2 = MetricConfig("f1", MetricType.CLASSIFICATION, None, None, None)
        config = CombinedMetricConfig(
            metrics=(m1, m2),
            aggregation="weighted",
            weights=(0.6, 0.4),
        )
        assert config.weights == (0.6, 0.4)


class TestValidateMetricConfig:
    """Tests for validate_metric_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = MetricConfig(
            metric_name="accuracy",
            metric_type=MetricType.CLASSIFICATION,
            average=None,
            num_labels=None,
            kwargs=None,
        )
        validate_metric_config(config)  # Should not raise

    def test_empty_metric_name_raises(self) -> None:
        """Test that empty metric_name raises error."""
        config = MetricConfig(
            metric_name="",
            metric_type=MetricType.CLASSIFICATION,
            average=None,
            num_labels=None,
            kwargs=None,
        )
        with pytest.raises(ValueError, match="metric_name cannot be empty"):
            validate_metric_config(config)

    def test_zero_num_labels_raises(self) -> None:
        """Test that zero num_labels raises error."""
        config = MetricConfig(
            metric_name="f1",
            metric_type=MetricType.CLASSIFICATION,
            average="macro",
            num_labels=0,
            kwargs=None,
        )
        with pytest.raises(ValueError, match="num_labels must be positive"):
            validate_metric_config(config)

    def test_invalid_average_raises(self) -> None:
        """Test that invalid average raises error."""
        config = MetricConfig(
            metric_name="f1",
            metric_type=MetricType.CLASSIFICATION,
            average="invalid",
            num_labels=None,
            kwargs=None,
        )
        with pytest.raises(ValueError, match="average must be one of"):
            validate_metric_config(config)


class TestCreateMetricConfig:
    """Tests for create_metric_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_metric_config("accuracy")
        assert config.metric_name == "accuracy"
        assert config.metric_type == MetricType.CLASSIFICATION
        assert config.average is None

    def test_custom_metric_type(self) -> None:
        """Test custom metric type."""
        config = create_metric_config("mse", metric_type="regression")
        assert config.metric_type == MetricType.REGRESSION

    def test_with_average(self) -> None:
        """Test with average parameter."""
        config = create_metric_config("f1", average="macro")
        assert config.average == "macro"

    def test_empty_name_raises(self) -> None:
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="metric_name cannot be empty"):
            create_metric_config("")

    def test_invalid_metric_type_raises(self) -> None:
        """Test that invalid metric type raises error."""
        with pytest.raises(ValueError, match="metric_type must be one of"):
            create_metric_config("accuracy", metric_type="invalid")


class TestCreateCombinedConfig:
    """Tests for create_combined_config function."""

    def test_basic_creation(self) -> None:
        """Test basic combined config creation."""
        m1 = create_metric_config("accuracy")
        m2 = create_metric_config("f1")
        config = create_combined_config((m1, m2))
        assert config.aggregation == "mean"
        assert config.weights is None

    def test_with_aggregation(self) -> None:
        """Test with custom aggregation."""
        m1 = create_metric_config("accuracy")
        config = create_combined_config((m1,), aggregation="max")
        assert config.aggregation == "max"

    def test_with_weights(self) -> None:
        """Test with weights."""
        m1 = create_metric_config("accuracy")
        m2 = create_metric_config("f1")
        config = create_combined_config(
            (m1, m2), aggregation="weighted", weights=(0.7, 0.3)
        )
        assert config.weights == (0.7, 0.3)

    def test_empty_metrics_raises(self) -> None:
        """Test that empty metrics raises error."""
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            create_combined_config(())

    def test_invalid_aggregation_raises(self) -> None:
        """Test that invalid aggregation raises error."""
        m1 = create_metric_config("accuracy")
        with pytest.raises(ValueError, match="aggregation must be one of"):
            create_combined_config((m1,), aggregation="invalid")  # type: ignore[arg-type]

    def test_weights_length_mismatch_raises(self) -> None:
        """Test that weights length mismatch raises error."""
        m1 = create_metric_config("accuracy")
        m2 = create_metric_config("f1")
        with pytest.raises(ValueError, match=r"weights length.*must match"):
            create_combined_config((m1, m2), weights=(0.5,))

    def test_negative_weights_raises(self) -> None:
        """Test that negative weights raises error."""
        m1 = create_metric_config("accuracy")
        with pytest.raises(ValueError, match="weights cannot be negative"):
            create_combined_config((m1,), weights=(-0.5,))


class TestListMetrics:
    """Tests for list_*_metrics functions."""

    def test_list_classification_metrics(self) -> None:
        """Test listing classification metrics."""
        metrics = list_classification_metrics()
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics == sorted(metrics)

    def test_list_regression_metrics(self) -> None:
        """Test listing regression metrics."""
        metrics = list_regression_metrics()
        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics == sorted(metrics)

    def test_list_generation_metrics(self) -> None:
        """Test listing generation metrics."""
        metrics = list_generation_metrics()
        assert "bleu" in metrics
        assert "rouge" in metrics
        assert metrics == sorted(metrics)


class TestGetMetricType:
    """Tests for get_metric_type function."""

    def test_classification_metric(self) -> None:
        """Test inferring classification metric type."""
        assert get_metric_type("accuracy") == MetricType.CLASSIFICATION
        assert get_metric_type("f1") == MetricType.CLASSIFICATION

    def test_regression_metric(self) -> None:
        """Test inferring regression metric type."""
        assert get_metric_type("mse") == MetricType.REGRESSION
        assert get_metric_type("mae") == MetricType.REGRESSION

    def test_generation_metric(self) -> None:
        """Test inferring generation metric type."""
        assert get_metric_type("bleu") == MetricType.GENERATION
        assert get_metric_type("rouge") == MetricType.GENERATION

    def test_unknown_metric(self) -> None:
        """Test unknown metric returns CUSTOM."""
        assert get_metric_type("unknown_metric") == MetricType.CUSTOM


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_mean_aggregation(self) -> None:
        """Test mean aggregation."""
        r1 = MetricResult("accuracy", 0.9, None)
        r2 = MetricResult("f1", 0.8, None)
        result = aggregate_results((r1, r2), "mean")
        assert result == pytest.approx(0.85)

    def test_min_aggregation(self) -> None:
        """Test min aggregation."""
        r1 = MetricResult("accuracy", 0.9, None)
        r2 = MetricResult("f1", 0.8, None)
        result = aggregate_results((r1, r2), "min")
        assert result == pytest.approx(0.8)

    def test_max_aggregation(self) -> None:
        """Test max aggregation."""
        r1 = MetricResult("accuracy", 0.9, None)
        r2 = MetricResult("f1", 0.8, None)
        result = aggregate_results((r1, r2), "max")
        assert result == pytest.approx(0.9)

    def test_weighted_aggregation(self) -> None:
        """Test weighted aggregation."""
        r1 = MetricResult("accuracy", 0.9, None)
        r2 = MetricResult("f1", 0.8, None)
        result = aggregate_results((r1, r2), "weighted", (0.6, 0.4))
        # 0.9 * 0.6 + 0.8 * 0.4 = 0.54 + 0.32 = 0.86
        assert result == pytest.approx(0.86)

    def test_empty_results_raises(self) -> None:
        """Test that empty results raises error."""
        with pytest.raises(ValueError, match="results cannot be empty"):
            aggregate_results(())

    def test_weighted_without_weights_raises(self) -> None:
        """Test that weighted without weights raises error."""
        r1 = MetricResult("accuracy", 0.9, None)
        with pytest.raises(ValueError, match="weights required"):
            aggregate_results((r1,), "weighted")

    def test_zero_total_weight_raises(self) -> None:
        """Test that zero total weight raises error."""
        r1 = MetricResult("accuracy", 0.9, None)
        with pytest.raises(ValueError, match="total weight cannot be zero"):
            aggregate_results((r1,), "weighted", (0.0,))


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        """Test perfect accuracy score."""
        preds = (1, 0, 1, 0)
        refs = (1, 0, 1, 0)
        assert compute_accuracy(preds, refs) == pytest.approx(1.0)

    def test_partial_accuracy(self) -> None:
        """Test partial accuracy score."""
        preds = (1, 0, 1, 1)
        refs = (1, 0, 0, 1)
        assert compute_accuracy(preds, refs) == pytest.approx(0.75)

    def test_zero_accuracy(self) -> None:
        """Test zero accuracy score."""
        preds = (1, 1, 1)
        refs = (0, 0, 0)
        assert compute_accuracy(preds, refs) == pytest.approx(0.0)

    def test_empty_predictions_raises(self) -> None:
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_accuracy((), ())

    def test_length_mismatch_raises(self) -> None:
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="must match"):
            compute_accuracy((1, 0), (1,))


class TestComputePrecision:
    """Tests for compute_precision function."""

    def test_perfect_precision(self) -> None:
        """Test perfect precision score."""
        preds = (1, 0, 1)
        refs = (1, 0, 1)
        assert compute_precision(preds, refs) == pytest.approx(1.0)

    def test_partial_precision(self) -> None:
        """Test partial precision score."""
        preds = (1, 1, 0, 1)  # 3 predicted positive
        refs = (1, 0, 0, 1)  # 2 true positive
        assert compute_precision(preds, refs) == pytest.approx(2 / 3)

    def test_zero_predicted_positives(self) -> None:
        """Test precision when no positives predicted."""
        preds = (0, 0, 0)
        refs = (1, 1, 1)
        assert compute_precision(preds, refs) == pytest.approx(0.0)

    def test_empty_predictions_raises(self) -> None:
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_precision((), ())


class TestComputeRecall:
    """Tests for compute_recall function."""

    def test_perfect_recall(self) -> None:
        """Test perfect recall score."""
        preds = (1, 1, 1)
        refs = (1, 1, 1)
        assert compute_recall(preds, refs) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        """Test partial recall score."""
        preds = (1, 0, 1, 0)  # 2 predicted positive
        refs = (1, 1, 1, 0)  # 3 actual positive, 2 found
        assert compute_recall(preds, refs) == pytest.approx(2 / 3)

    def test_zero_actual_positives(self) -> None:
        """Test recall when no actual positives."""
        preds = (1, 1, 1)
        refs = (0, 0, 0)
        assert compute_recall(preds, refs) == pytest.approx(0.0)

    def test_empty_predictions_raises(self) -> None:
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_recall((), ())


class TestComputeF1:
    """Tests for compute_f1 function."""

    def test_perfect_f1(self) -> None:
        """Test perfect F1 score."""
        preds = (1, 0, 1, 0)
        refs = (1, 0, 1, 0)
        assert compute_f1(preds, refs) == pytest.approx(1.0)

    def test_partial_f1(self) -> None:
        """Test partial F1 score."""
        preds = (1, 1, 0, 1)  # precision = 2/3, recall = 1.0
        refs = (1, 0, 0, 1)
        # F1 = 2 * (2/3 * 1) / (2/3 + 1) = 2 * 2/3 / 5/3 = 4/3 * 3/5 = 4/5 = 0.8
        assert compute_f1(preds, refs) == pytest.approx(0.8)

    def test_zero_f1(self) -> None:
        """Test zero F1 score."""
        preds = (0, 0, 0)
        refs = (1, 1, 1)
        assert compute_f1(preds, refs) == pytest.approx(0.0)

    def test_empty_predictions_raises(self) -> None:
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_f1((), ())


class TestComputeMSE:
    """Tests for compute_mse function."""

    def test_perfect_mse(self) -> None:
        """Test perfect MSE (zero)."""
        preds = (1.0, 2.0, 3.0)
        refs = (1.0, 2.0, 3.0)
        assert compute_mse(preds, refs) == pytest.approx(0.0)

    def test_nonzero_mse(self) -> None:
        """Test non-zero MSE."""
        preds = (1.0, 2.0, 3.0)
        refs = (2.0, 3.0, 4.0)
        # (1)^2 + (1)^2 + (1)^2 / 3 = 1.0
        assert compute_mse(preds, refs) == pytest.approx(1.0)

    def test_empty_predictions_raises(self) -> None:
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_mse((), ())

    def test_length_mismatch_raises(self) -> None:
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="must match"):
            compute_mse((1.0, 2.0), (1.0,))


class TestComputeMAE:
    """Tests for compute_mae function."""

    def test_perfect_mae(self) -> None:
        """Test perfect MAE (zero)."""
        preds = (1.0, 2.0, 3.0)
        refs = (1.0, 2.0, 3.0)
        assert compute_mae(preds, refs) == pytest.approx(0.0)

    def test_nonzero_mae(self) -> None:
        """Test non-zero MAE."""
        preds = (1.0, 2.0, 3.0)
        refs = (2.0, 3.0, 4.0)
        # (1 + 1 + 1) / 3 = 1.0
        assert compute_mae(preds, refs) == pytest.approx(1.0)

    def test_empty_predictions_raises(self) -> None:
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_mae((), ())


class TestGetRecommendedMetrics:
    """Tests for get_recommended_metrics function."""

    def test_classification_recommendations(self) -> None:
        """Test recommendations for classification."""
        metrics = get_recommended_metrics("classification")
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_regression_recommendations(self) -> None:
        """Test recommendations for regression."""
        metrics = get_recommended_metrics("regression")
        assert "mse" in metrics
        assert "mae" in metrics

    def test_generation_recommendations(self) -> None:
        """Test recommendations for generation."""
        metrics = get_recommended_metrics("generation")
        assert "bleu" in metrics
        assert "rouge" in metrics

    def test_qa_recommendations(self) -> None:
        """Test recommendations for QA."""
        metrics = get_recommended_metrics("qa")
        assert "f1" in metrics

    def test_invalid_task_raises(self) -> None:
        """Test that invalid task raises error."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_metrics("invalid")


class TestConstants:
    """Tests for module constants."""

    def test_classification_metrics_not_empty(self) -> None:
        """Test CLASSIFICATION_METRICS is not empty."""
        assert len(CLASSIFICATION_METRICS) > 0

    def test_regression_metrics_not_empty(self) -> None:
        """Test REGRESSION_METRICS is not empty."""
        assert len(REGRESSION_METRICS) > 0

    def test_generation_metrics_not_empty(self) -> None:
        """Test GENERATION_METRICS is not empty."""
        assert len(GENERATION_METRICS) > 0

    def test_valid_aggregation_methods_contents(self) -> None:
        """Test VALID_AGGREGATION_METHODS contains expected values."""
        assert "mean" in VALID_AGGREGATION_METHODS
        assert "weighted" in VALID_AGGREGATION_METHODS
        assert "min" in VALID_AGGREGATION_METHODS
        assert "max" in VALID_AGGREGATION_METHODS
