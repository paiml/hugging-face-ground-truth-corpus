"""Evaluation recipes for HuggingFace models.

This module provides utilities for computing metrics
and evaluating model performance.
"""

from __future__ import annotations

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

__all__: list[str] = [
    "ClassificationMetrics",
    "compute_accuracy",
    "compute_classification_metrics",
    "compute_f1",
    "compute_mean_loss",
    "compute_perplexity",
    "compute_precision",
    "compute_recall",
    "create_compute_metrics_fn",
]
