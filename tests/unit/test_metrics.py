"""Tests for evaluation metrics functionality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.metrics import (
    ClassificationMetrics,
    compute_accuracy,
    compute_classification_metrics,
    compute_f1,
    compute_mean_loss,
    compute_perplexity,
    compute_precision,
    compute_recall,
    create_compute_metrics_fn,
)


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating ClassificationMetrics instance."""
        metrics = ClassificationMetrics(
            accuracy=0.9, precision=0.85, recall=0.88, f1=0.865
        )
        assert metrics.accuracy == 0.9
        assert metrics.precision == 0.85
        assert metrics.recall == 0.88
        assert metrics.f1 == 0.865

    def test_frozen(self) -> None:
        """Test that ClassificationMetrics is immutable."""
        metrics = ClassificationMetrics(
            accuracy=0.9, precision=0.85, recall=0.88, f1=0.865
        )
        with pytest.raises(AttributeError):
            metrics.accuracy = 0.5  # type: ignore[misc]


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        """Test 100% accuracy."""
        assert compute_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_zero_accuracy(self) -> None:
        """Test 0% accuracy."""
        assert compute_accuracy([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0

    def test_partial_accuracy(self) -> None:
        """Test partial accuracy."""
        assert compute_accuracy([1, 0, 1, 1], [1, 0, 1, 0]) == 0.75

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
        assert compute_precision([1, 1], [1, 1]) == 1.0

    def test_zero_precision(self) -> None:
        """Test 0% precision (all predictions wrong)."""
        assert compute_precision([1, 1], [0, 0]) == 0.0

    def test_no_positive_predictions(self) -> None:
        """Test precision when no positive predictions made."""
        assert compute_precision([0, 0, 0], [1, 1, 1]) == 0.0

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
        assert compute_recall([1, 1], [1, 1]) == 1.0

    def test_zero_recall(self) -> None:
        """Test 0% recall (missed all positives)."""
        assert compute_recall([0, 0], [1, 1]) == 0.0

    def test_no_actual_positives(self) -> None:
        """Test recall when no actual positives exist."""
        assert compute_recall([1, 1], [0, 0]) == 0.0

    def test_partial_recall(self) -> None:
        """Test partial recall."""
        # 1 true positive out of 2 actual positives = 0.5
        assert compute_recall([1, 0], [1, 1]) == 0.5

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
        assert compute_f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_zero_f1(self) -> None:
        """Test F1 of 0 when precision and recall are both 0."""
        assert compute_f1([0, 0, 0], [1, 1, 1]) == 0.0

    def test_f1_calculation(self) -> None:
        """Test F1 calculation: 2 * p * r / (p + r)."""
        # precision = 2/3, recall = 1.0
        # f1 = 2 * (2/3) * 1 / ((2/3) + 1) = 4/3 / 5/3 = 4/5 = 0.8
        result = compute_f1([1, 1, 0, 1], [1, 0, 0, 1])
        assert result == 0.8

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
        assert metrics.accuracy == 0.75
        assert abs(metrics.precision - 2 / 3) < 1e-10
        assert metrics.recall == 1.0
        assert metrics.f1 == 0.8

    def test_perfect_metrics(self) -> None:
        """Test perfect classification."""
        metrics = compute_classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_custom_positive_label(self) -> None:
        """Test with custom positive label."""
        # Using 0 as positive label
        metrics = compute_classification_metrics([0, 0, 1], [0, 1, 1], positive_label=0)
        assert metrics.accuracy == pytest.approx(2 / 3)
        assert metrics.precision == 0.5  # 1 TP, 1 FP
        assert metrics.recall == 1.0  # 1 TP, 0 FN


class TestComputePerplexity:
    """Tests for compute_perplexity function."""

    def test_zero_loss(self) -> None:
        """Test perplexity with zero loss."""
        assert compute_perplexity(0.0) == 1.0

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
        assert compute_mean_loss([5.0]) == 5.0

    def test_multiple_values(self) -> None:
        """Test mean of multiple values."""
        assert compute_mean_loss([1.0, 2.0, 3.0]) == 2.0

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
        assert result["accuracy"] == 0.75

    def test_handles_logits(self) -> None:
        """Test that function handles logits (takes argmax)."""
        fn = create_compute_metrics_fn()
        # Logits: [[0.1, 0.9], [0.8, 0.2]] -> predictions [1, 0]
        logits = np.array([[0.1, 0.9], [0.8, 0.2]])
        labels = np.array([1, 0])

        result = fn((logits, labels))

        assert result["accuracy"] == 1.0

    def test_custom_positive_label(self) -> None:
        """Test with custom positive label."""
        fn = create_compute_metrics_fn(positive_label=0)
        predictions = np.array([0, 0, 1])
        labels = np.array([0, 1, 1])

        result = fn((predictions, labels))

        # With positive_label=0: TP=1, FP=1, FN=0
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0
