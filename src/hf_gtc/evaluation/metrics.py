"""Evaluation metrics for ML models.

This module provides functions for computing common evaluation metrics
used in machine learning tasks.

Examples:
    >>> from hf_gtc.evaluation.metrics import compute_accuracy
    >>> compute_accuracy([1, 0, 1], [1, 0, 0])
    0.6666666666666666
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Container for classification metrics.

    Attributes:
        accuracy: Overall accuracy (correct / total).
        precision: Precision score (true positives / predicted positives).
        recall: Recall score (true positives / actual positives).
        f1: F1 score (harmonic mean of precision and recall).

    Examples:
        >>> metrics = ClassificationMetrics(
        ...     accuracy=0.9, precision=0.85, recall=0.88, f1=0.865
        ... )
        >>> metrics.accuracy
        0.9
    """

    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_accuracy(
    predictions: Sequence[int],
    labels: Sequence[int],
) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.

    Returns:
        Accuracy as a float between 0 and 1.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_accuracy([1, 0, 1, 1], [1, 0, 1, 0])
        0.75
        >>> compute_accuracy([1, 1, 1], [1, 1, 1])
        1.0
        >>> compute_accuracy([0, 0], [1, 1])
        0.0

        >>> compute_accuracy([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions and labels cannot be empty

        >>> compute_accuracy([1], [1, 2])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions and labels must have the same length
    """
    if len(predictions) == 0 or len(labels) == 0:
        msg = "predictions and labels cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(labels):
        msg = (
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
        raise ValueError(msg)

    correct = sum(pred == lbl for pred, lbl in zip(predictions, labels, strict=True))
    return correct / len(predictions)


def compute_precision(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    """Compute precision for binary classification.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        Precision as a float between 0 and 1.
        Returns 0.0 if no positive predictions were made.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_precision([1, 1, 0, 1], [1, 0, 0, 1])
        0.6666666666666666
        >>> compute_precision([0, 0, 0], [1, 1, 1])
        0.0
        >>> compute_precision([1, 1], [1, 1])
        1.0
    """
    if len(predictions) == 0 or len(labels) == 0:
        msg = "predictions and labels cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(labels):
        msg = (
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
        raise ValueError(msg)

    true_positives = sum(
        pred == positive_label and lbl == positive_label
        for pred, lbl in zip(predictions, labels, strict=True)
    )
    predicted_positives = sum(p == positive_label for p in predictions)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def compute_recall(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    """Compute recall for binary classification.

    Recall = True Positives / (True Positives + False Negatives)

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        Recall as a float between 0 and 1.
        Returns 0.0 if no actual positives exist.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_recall([1, 1, 0, 1], [1, 0, 0, 1])
        1.0
        >>> compute_recall([0, 0, 0], [1, 1, 1])
        0.0
        >>> compute_recall([1, 0], [1, 1])
        0.5
    """
    if len(predictions) == 0 or len(labels) == 0:
        msg = "predictions and labels cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(labels):
        msg = (
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
        raise ValueError(msg)

    true_positives = sum(
        pred == positive_label and lbl == positive_label
        for pred, lbl in zip(predictions, labels, strict=True)
    )
    actual_positives = sum(lbl == positive_label for lbl in labels)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def compute_f1(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    """Compute F1 score for binary classification.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        F1 score as a float between 0 and 1.
        Returns 0.0 if precision + recall is 0.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> compute_f1([1, 1, 0, 1], [1, 0, 0, 1])
        0.8
        >>> compute_f1([0, 0, 0], [1, 1, 1])
        0.0
        >>> compute_f1([1, 1], [1, 1])
        1.0
    """
    precision = compute_precision(predictions, labels, positive_label)
    recall = compute_recall(predictions, labels, positive_label)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_classification_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    positive_label: int = 1,
) -> ClassificationMetrics:
    """Compute all classification metrics at once.

    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        ClassificationMetrics containing accuracy, precision, recall, and F1.

    Raises:
        ValueError: If predictions and labels have different lengths.
        ValueError: If inputs are empty.

    Examples:
        >>> metrics = compute_classification_metrics([1, 1, 0, 1], [1, 0, 0, 1])
        >>> metrics.accuracy
        0.75
        >>> round(metrics.precision, 4)
        0.6667
        >>> metrics.recall
        1.0
        >>> metrics.f1
        0.8
    """
    return ClassificationMetrics(
        accuracy=compute_accuracy(predictions, labels),
        precision=compute_precision(predictions, labels, positive_label),
        recall=compute_recall(predictions, labels, positive_label),
        f1=compute_f1(predictions, labels, positive_label),
    )


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity score.

    Raises:
        ValueError: If loss is negative.
        ValueError: If loss would cause overflow.

    Examples:
        >>> round(compute_perplexity(2.0), 4)
        7.3891
        >>> round(compute_perplexity(0.0), 4)
        1.0
        >>> round(compute_perplexity(1.0), 4)
        2.7183

        >>> compute_perplexity(-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: loss cannot be negative
    """
    if loss < 0:
        msg = f"loss cannot be negative, got {loss}"
        raise ValueError(msg)

    # Prevent overflow for very large losses
    if loss > 700:
        msg = f"loss too large, would cause overflow: {loss}"
        raise ValueError(msg)

    return math.exp(loss)


def compute_mean_loss(losses: Sequence[float]) -> float:
    """Compute mean loss from a sequence of losses.

    Args:
        losses: Sequence of loss values.

    Returns:
        Mean loss.

    Raises:
        ValueError: If losses is empty.

    Examples:
        >>> compute_mean_loss([1.0, 2.0, 3.0])
        2.0
        >>> compute_mean_loss([0.5])
        0.5

        >>> compute_mean_loss([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: losses cannot be empty
    """
    if len(losses) == 0:
        msg = "losses cannot be empty"
        raise ValueError(msg)

    return sum(losses) / len(losses)


def create_compute_metrics_fn(
    positive_label: int = 1,
) -> Callable[[tuple], dict[str, float]]:
    """Create a compute_metrics function for HuggingFace Trainer.

    Args:
        positive_label: The label considered positive. Defaults to 1.

    Returns:
        A function compatible with Trainer.compute_metrics.

    Examples:
        >>> fn = create_compute_metrics_fn()
        >>> callable(fn)
        True
    """

    def compute_metrics(eval_pred: tuple) -> dict[str, float]:
        """Compute metrics from evaluation predictions.

        Args:
            eval_pred: Tuple of (predictions, labels) from Trainer.

        Returns:
            Dictionary of metric names to values.
        """
        predictions, labels = eval_pred

        # Handle logits (take argmax)
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=-1)

        preds_list = predictions.tolist()
        labels_list = labels.tolist()

        metrics = compute_classification_metrics(
            preds_list, labels_list, positive_label
        )

        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        }

    return compute_metrics
