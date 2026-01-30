"""RAG evaluation and retrieval metrics.

This module provides utilities for evaluating RAG (Retrieval-Augmented Generation)
systems, including retrieval metrics (precision, recall, MRR, NDCG) and
generation quality metrics (faithfulness, relevance, groundedness).

Examples:
    >>> from hf_gtc.rag.evaluation import create_rag_eval_config
    >>> config = create_rag_eval_config(retrieval_k=10)
    >>> config.retrieval_k
    10
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RAGMetric(Enum):
    """Supported RAG evaluation metrics.

    Attributes:
        RETRIEVAL_PRECISION: Precision of retrieved documents.
        RETRIEVAL_RECALL: Recall of retrieved documents.
        CONTEXT_RELEVANCE: Relevance of context to the query.
        FAITHFULNESS: Faithfulness of answer to retrieved context.
        ANSWER_RELEVANCE: Relevance of answer to the query.

    Examples:
        >>> RAGMetric.RETRIEVAL_PRECISION.value
        'retrieval_precision'
        >>> RAGMetric.FAITHFULNESS.value
        'faithfulness'
    """

    RETRIEVAL_PRECISION = "retrieval_precision"
    RETRIEVAL_RECALL = "retrieval_recall"
    CONTEXT_RELEVANCE = "context_relevance"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"


VALID_RAG_METRICS = frozenset(m.value for m in RAGMetric)


class GroundingType(Enum):
    """Types of grounding for RAG evaluation.

    Attributes:
        CITATION: Citation-based grounding with explicit references.
        ATTRIBUTION: Attribution to source documents.
        FACTUAL: Factual consistency with retrieved content.

    Examples:
        >>> GroundingType.CITATION.value
        'citation'
        >>> GroundingType.FACTUAL.value
        'factual'
    """

    CITATION = "citation"
    ATTRIBUTION = "attribution"
    FACTUAL = "factual"


VALID_GROUNDING_TYPES = frozenset(t.value for t in GroundingType)


class EvaluationLevel(Enum):
    """Granularity level for evaluation.

    Attributes:
        DOCUMENT: Evaluation at document level.
        PASSAGE: Evaluation at passage level.
        SENTENCE: Evaluation at sentence level.

    Examples:
        >>> EvaluationLevel.DOCUMENT.value
        'document'
        >>> EvaluationLevel.SENTENCE.value
        'sentence'
    """

    DOCUMENT = "document"
    PASSAGE = "passage"
    SENTENCE = "sentence"


VALID_EVALUATION_LEVELS = frozenset(level.value for level in EvaluationLevel)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Metrics for retrieval quality evaluation.

    Attributes:
        precision_at_k: Precision at k documents.
        recall_at_k: Recall at k documents.
        mrr: Mean Reciprocal Rank.
        ndcg: Normalized Discounted Cumulative Gain.
        hit_rate: Hit rate (at least one relevant doc retrieved).

    Examples:
        >>> metrics = RetrievalMetrics(
        ...     precision_at_k=0.8,
        ...     recall_at_k=0.6,
        ...     mrr=0.75,
        ...     ndcg=0.82,
        ...     hit_rate=1.0,
        ... )
        >>> metrics.precision_at_k
        0.8
    """

    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    hit_rate: float


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Metrics for generation quality evaluation.

    Attributes:
        faithfulness: How faithful the answer is to the context.
        relevance: How relevant the answer is to the query.
        coherence: How coherent and well-structured the answer is.
        groundedness: How well-grounded the answer is in sources.

    Examples:
        >>> metrics = GenerationMetrics(
        ...     faithfulness=0.9,
        ...     relevance=0.85,
        ...     coherence=0.88,
        ...     groundedness=0.92,
        ... )
        >>> metrics.faithfulness
        0.9
    """

    faithfulness: float
    relevance: float
    coherence: float
    groundedness: float


@dataclass(frozen=True, slots=True)
class RAGEvalConfig:
    """Configuration for RAG evaluation.

    Attributes:
        retrieval_k: Number of documents to evaluate for retrieval.
        metrics: Tuple of metrics to compute.
        grounding_type: Type of grounding to evaluate.
        evaluation_level: Granularity level for evaluation.

    Examples:
        >>> config = RAGEvalConfig(
        ...     retrieval_k=10,
        ...     metrics=(RAGMetric.RETRIEVAL_PRECISION, RAGMetric.FAITHFULNESS),
        ...     grounding_type=GroundingType.CITATION,
        ...     evaluation_level=EvaluationLevel.PASSAGE,
        ... )
        >>> config.retrieval_k
        10
    """

    retrieval_k: int
    metrics: tuple[RAGMetric, ...]
    grounding_type: GroundingType
    evaluation_level: EvaluationLevel


@dataclass(frozen=True, slots=True)
class RAGEvalResult:
    """Result of RAG evaluation.

    Attributes:
        retrieval_metrics: Retrieval quality metrics.
        generation_metrics: Generation quality metrics.
        overall_score: Combined overall score.
        per_query_scores: Tuple of scores per query.

    Examples:
        >>> retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        >>> generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        >>> result = RAGEvalResult(
        ...     retrieval_metrics=retrieval,
        ...     generation_metrics=generation,
        ...     overall_score=0.85,
        ...     per_query_scores=(0.82, 0.88, 0.85),
        ... )
        >>> result.overall_score
        0.85
    """

    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    overall_score: float
    per_query_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RAGEvalStats:
    """Statistics for RAG evaluation results.

    Attributes:
        avg_retrieval_score: Average retrieval score.
        avg_generation_score: Average generation score.
        num_queries: Total number of queries evaluated.
        failed_queries: Number of queries that failed.

    Examples:
        >>> stats = RAGEvalStats(
        ...     avg_retrieval_score=0.78,
        ...     avg_generation_score=0.85,
        ...     num_queries=100,
        ...     failed_queries=2,
        ... )
        >>> stats.num_queries
        100
    """

    avg_retrieval_score: float
    avg_generation_score: float
    num_queries: int
    failed_queries: int


def validate_retrieval_metrics(metrics: RetrievalMetrics) -> None:
    """Validate retrieval metrics.

    Args:
        metrics: RetrievalMetrics to validate.

    Raises:
        ValueError: If metrics are invalid.

    Examples:
        >>> metrics = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        >>> validate_retrieval_metrics(metrics)  # No error

        >>> bad = RetrievalMetrics(-0.1, 0.6, 0.75, 0.82, 1.0)
        >>> validate_retrieval_metrics(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: precision_at_k must be between 0.0 and 1.0
    """
    if not 0.0 <= metrics.precision_at_k <= 1.0:
        msg = (
            f"precision_at_k must be between 0.0 and 1.0, got {metrics.precision_at_k}"
        )
        raise ValueError(msg)

    if not 0.0 <= metrics.recall_at_k <= 1.0:
        msg = f"recall_at_k must be between 0.0 and 1.0, got {metrics.recall_at_k}"
        raise ValueError(msg)

    if not 0.0 <= metrics.mrr <= 1.0:
        msg = f"mrr must be between 0.0 and 1.0, got {metrics.mrr}"
        raise ValueError(msg)

    if not 0.0 <= metrics.ndcg <= 1.0:
        msg = f"ndcg must be between 0.0 and 1.0, got {metrics.ndcg}"
        raise ValueError(msg)

    if not 0.0 <= metrics.hit_rate <= 1.0:
        msg = f"hit_rate must be between 0.0 and 1.0, got {metrics.hit_rate}"
        raise ValueError(msg)


def validate_generation_metrics(metrics: GenerationMetrics) -> None:
    """Validate generation metrics.

    Args:
        metrics: GenerationMetrics to validate.

    Raises:
        ValueError: If metrics are invalid.

    Examples:
        >>> metrics = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        >>> validate_generation_metrics(metrics)  # No error

        >>> bad = GenerationMetrics(1.5, 0.85, 0.88, 0.92)
        >>> validate_generation_metrics(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: faithfulness must be between 0.0 and 1.0
    """
    if not 0.0 <= metrics.faithfulness <= 1.0:
        msg = f"faithfulness must be between 0.0 and 1.0, got {metrics.faithfulness}"
        raise ValueError(msg)

    if not 0.0 <= metrics.relevance <= 1.0:
        msg = f"relevance must be between 0.0 and 1.0, got {metrics.relevance}"
        raise ValueError(msg)

    if not 0.0 <= metrics.coherence <= 1.0:
        msg = f"coherence must be between 0.0 and 1.0, got {metrics.coherence}"
        raise ValueError(msg)

    if not 0.0 <= metrics.groundedness <= 1.0:
        msg = f"groundedness must be between 0.0 and 1.0, got {metrics.groundedness}"
        raise ValueError(msg)


def validate_rag_eval_config(config: RAGEvalConfig) -> None:
    """Validate RAG evaluation configuration.

    Args:
        config: RAGEvalConfig to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = RAGEvalConfig(
        ...     retrieval_k=10,
        ...     metrics=(RAGMetric.RETRIEVAL_PRECISION,),
        ...     grounding_type=GroundingType.CITATION,
        ...     evaluation_level=EvaluationLevel.DOCUMENT,
        ... )
        >>> validate_rag_eval_config(config)  # No error

        >>> bad = RAGEvalConfig(
        ...     retrieval_k=0,
        ...     metrics=(RAGMetric.RETRIEVAL_PRECISION,),
        ...     grounding_type=GroundingType.CITATION,
        ...     evaluation_level=EvaluationLevel.DOCUMENT,
        ... )
        >>> validate_rag_eval_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: retrieval_k must be positive
    """
    if config.retrieval_k <= 0:
        msg = f"retrieval_k must be positive, got {config.retrieval_k}"
        raise ValueError(msg)

    if not config.metrics:
        msg = "metrics cannot be empty"
        raise ValueError(msg)


def validate_rag_eval_result(result: RAGEvalResult) -> None:
    """Validate RAG evaluation result.

    Args:
        result: RAGEvalResult to validate.

    Raises:
        ValueError: If result is invalid.

    Examples:
        >>> retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        >>> generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        >>> result = RAGEvalResult(retrieval, generation, 0.85, (0.8,))
        >>> validate_rag_eval_result(result)  # No error

        >>> bad_result = RAGEvalResult(retrieval, generation, 1.5, (0.8,))
        >>> validate_rag_eval_result(bad_result)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overall_score must be between 0.0 and 1.0
    """
    validate_retrieval_metrics(result.retrieval_metrics)
    validate_generation_metrics(result.generation_metrics)

    if not 0.0 <= result.overall_score <= 1.0:
        msg = f"overall_score must be between 0.0 and 1.0, got {result.overall_score}"
        raise ValueError(msg)

    for i, score in enumerate(result.per_query_scores):
        if not 0.0 <= score <= 1.0:
            msg = f"per_query_scores[{i}] must be between 0.0 and 1.0, got {score}"
            raise ValueError(msg)


def validate_rag_eval_stats(stats: RAGEvalStats) -> None:
    """Validate RAG evaluation statistics.

    Args:
        stats: RAGEvalStats to validate.

    Raises:
        ValueError: If stats are invalid.

    Examples:
        >>> stats = RAGEvalStats(0.78, 0.85, 100, 2)
        >>> validate_rag_eval_stats(stats)  # No error

        >>> bad = RAGEvalStats(0.78, 0.85, -1, 2)
        >>> validate_rag_eval_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_queries must be non-negative
    """
    if not 0.0 <= stats.avg_retrieval_score <= 1.0:
        msg = (
            f"avg_retrieval_score must be between 0.0 and 1.0, "
            f"got {stats.avg_retrieval_score}"
        )
        raise ValueError(msg)

    if not 0.0 <= stats.avg_generation_score <= 1.0:
        msg = (
            f"avg_generation_score must be between 0.0 and 1.0, "
            f"got {stats.avg_generation_score}"
        )
        raise ValueError(msg)

    if stats.num_queries < 0:
        msg = f"num_queries must be non-negative, got {stats.num_queries}"
        raise ValueError(msg)

    if stats.failed_queries < 0:
        msg = f"failed_queries must be non-negative, got {stats.failed_queries}"
        raise ValueError(msg)

    if stats.failed_queries > stats.num_queries:
        msg = (
            f"failed_queries ({stats.failed_queries}) cannot exceed "
            f"num_queries ({stats.num_queries})"
        )
        raise ValueError(msg)


def create_retrieval_metrics(
    precision_at_k: float = 0.0,
    recall_at_k: float = 0.0,
    mrr: float = 0.0,
    ndcg: float = 0.0,
    hit_rate: float = 0.0,
) -> RetrievalMetrics:
    """Create retrieval metrics.

    Args:
        precision_at_k: Precision at k. Defaults to 0.0.
        recall_at_k: Recall at k. Defaults to 0.0.
        mrr: Mean Reciprocal Rank. Defaults to 0.0.
        ndcg: Normalized DCG. Defaults to 0.0.
        hit_rate: Hit rate. Defaults to 0.0.

    Returns:
        RetrievalMetrics with the specified values.

    Raises:
        ValueError: If metrics are invalid.

    Examples:
        >>> metrics = create_retrieval_metrics(precision_at_k=0.8, recall_at_k=0.6)
        >>> metrics.precision_at_k
        0.8

        >>> metrics = create_retrieval_metrics(mrr=0.75, ndcg=0.82)
        >>> metrics.mrr
        0.75

        >>> create_retrieval_metrics(precision_at_k=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: precision_at_k must be between 0.0 and 1.0
    """
    metrics = RetrievalMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg=ndcg,
        hit_rate=hit_rate,
    )
    validate_retrieval_metrics(metrics)
    return metrics


def create_generation_metrics(
    faithfulness: float = 0.0,
    relevance: float = 0.0,
    coherence: float = 0.0,
    groundedness: float = 0.0,
) -> GenerationMetrics:
    """Create generation metrics.

    Args:
        faithfulness: Faithfulness score. Defaults to 0.0.
        relevance: Relevance score. Defaults to 0.0.
        coherence: Coherence score. Defaults to 0.0.
        groundedness: Groundedness score. Defaults to 0.0.

    Returns:
        GenerationMetrics with the specified values.

    Raises:
        ValueError: If metrics are invalid.

    Examples:
        >>> metrics = create_generation_metrics(faithfulness=0.9, relevance=0.85)
        >>> metrics.faithfulness
        0.9

        >>> metrics = create_generation_metrics(coherence=0.88, groundedness=0.92)
        >>> metrics.coherence
        0.88

        >>> create_generation_metrics(faithfulness=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: faithfulness must be between 0.0 and 1.0
    """
    metrics = GenerationMetrics(
        faithfulness=faithfulness,
        relevance=relevance,
        coherence=coherence,
        groundedness=groundedness,
    )
    validate_generation_metrics(metrics)
    return metrics


def create_rag_eval_config(
    retrieval_k: int = 10,
    metrics: tuple[str, ...] | None = None,
    grounding_type: str = "citation",
    evaluation_level: str = "document",
) -> RAGEvalConfig:
    """Create RAG evaluation configuration.

    Args:
        retrieval_k: Number of documents for retrieval eval. Defaults to 10.
        metrics: Tuple of metric names. Defaults to all metrics.
        grounding_type: Type of grounding. Defaults to "citation".
        evaluation_level: Evaluation granularity. Defaults to "document".

    Returns:
        RAGEvalConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_rag_eval_config(retrieval_k=5)
        >>> config.retrieval_k
        5

        >>> config = create_rag_eval_config(grounding_type="factual")
        >>> config.grounding_type
        <GroundingType.FACTUAL: 'factual'>

        >>> create_rag_eval_config(retrieval_k=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: retrieval_k must be positive
    """
    if grounding_type not in VALID_GROUNDING_TYPES:
        msg = (
            f"grounding_type must be one of {VALID_GROUNDING_TYPES}, "
            f"got '{grounding_type}'"
        )
        raise ValueError(msg)

    if evaluation_level not in VALID_EVALUATION_LEVELS:
        msg = (
            f"evaluation_level must be one of {VALID_EVALUATION_LEVELS}, "
            f"got '{evaluation_level}'"
        )
        raise ValueError(msg)

    if metrics is None:
        metric_enums = tuple(RAGMetric)
    else:
        metric_enums = []
        for metric in metrics:
            if metric not in VALID_RAG_METRICS:
                msg = f"metric must be one of {VALID_RAG_METRICS}, got '{metric}'"
                raise ValueError(msg)
            metric_enums.append(RAGMetric(metric))
        metric_enums = tuple(metric_enums)

    config = RAGEvalConfig(
        retrieval_k=retrieval_k,
        metrics=metric_enums,
        grounding_type=GroundingType(grounding_type),
        evaluation_level=EvaluationLevel(evaluation_level),
    )
    validate_rag_eval_config(config)
    return config


def create_rag_eval_result(
    retrieval_metrics: RetrievalMetrics,
    generation_metrics: GenerationMetrics,
    overall_score: float,
    per_query_scores: tuple[float, ...],
) -> RAGEvalResult:
    """Create RAG evaluation result.

    Args:
        retrieval_metrics: Retrieval quality metrics.
        generation_metrics: Generation quality metrics.
        overall_score: Combined overall score.
        per_query_scores: Tuple of scores per query.

    Returns:
        RAGEvalResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        >>> generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        >>> result = create_rag_eval_result(retrieval, generation, 0.85, (0.82, 0.88))
        >>> result.overall_score
        0.85

        >>> create_rag_eval_result(retrieval, generation, 1.5, (0.8,))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overall_score must be between 0.0 and 1.0
    """
    result = RAGEvalResult(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        overall_score=overall_score,
        per_query_scores=per_query_scores,
    )
    validate_rag_eval_result(result)
    return result


def create_rag_eval_stats(
    avg_retrieval_score: float,
    avg_generation_score: float,
    num_queries: int,
    failed_queries: int = 0,
) -> RAGEvalStats:
    """Create RAG evaluation statistics.

    Args:
        avg_retrieval_score: Average retrieval score.
        avg_generation_score: Average generation score.
        num_queries: Total number of queries.
        failed_queries: Number of failed queries. Defaults to 0.

    Returns:
        RAGEvalStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_rag_eval_stats(0.78, 0.85, 100, 2)
        >>> stats.num_queries
        100

        >>> stats = create_rag_eval_stats(0.8, 0.9, 50)
        >>> stats.failed_queries
        0

        >>> create_rag_eval_stats(0.78, 0.85, -1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_queries must be non-negative
    """
    stats = RAGEvalStats(
        avg_retrieval_score=avg_retrieval_score,
        avg_generation_score=avg_generation_score,
        num_queries=num_queries,
        failed_queries=failed_queries,
    )
    validate_rag_eval_stats(stats)
    return stats


def list_rag_metrics() -> list[str]:
    """List supported RAG metrics.

    Returns:
        Sorted list of metric names.

    Examples:
        >>> metrics = list_rag_metrics()
        >>> "faithfulness" in metrics
        True
        >>> "retrieval_precision" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_RAG_METRICS)


def list_grounding_types() -> list[str]:
    """List supported grounding types.

    Returns:
        Sorted list of grounding type names.

    Examples:
        >>> types = list_grounding_types()
        >>> "citation" in types
        True
        >>> "factual" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_GROUNDING_TYPES)


def list_evaluation_levels() -> list[str]:
    """List supported evaluation levels.

    Returns:
        Sorted list of evaluation level names.

    Examples:
        >>> levels = list_evaluation_levels()
        >>> "document" in levels
        True
        >>> "sentence" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_EVALUATION_LEVELS)


def get_rag_metric(name: str) -> RAGMetric:
    """Get RAG metric from name.

    Args:
        name: Metric name.

    Returns:
        RAGMetric enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_rag_metric("faithfulness")
        <RAGMetric.FAITHFULNESS: 'faithfulness'>

        >>> get_rag_metric("retrieval_precision")
        <RAGMetric.RETRIEVAL_PRECISION: 'retrieval_precision'>

        >>> get_rag_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric must be one of
    """
    if name not in VALID_RAG_METRICS:
        msg = f"metric must be one of {VALID_RAG_METRICS}, got '{name}'"
        raise ValueError(msg)
    return RAGMetric(name)


def get_grounding_type(name: str) -> GroundingType:
    """Get grounding type from name.

    Args:
        name: Grounding type name.

    Returns:
        GroundingType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_grounding_type("citation")
        <GroundingType.CITATION: 'citation'>

        >>> get_grounding_type("factual")
        <GroundingType.FACTUAL: 'factual'>

        >>> get_grounding_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: grounding_type must be one of
    """
    if name not in VALID_GROUNDING_TYPES:
        msg = f"grounding_type must be one of {VALID_GROUNDING_TYPES}, got '{name}'"
        raise ValueError(msg)
    return GroundingType(name)


def get_evaluation_level(name: str) -> EvaluationLevel:
    """Get evaluation level from name.

    Args:
        name: Evaluation level name.

    Returns:
        EvaluationLevel enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_evaluation_level("document")
        <EvaluationLevel.DOCUMENT: 'document'>

        >>> get_evaluation_level("sentence")
        <EvaluationLevel.SENTENCE: 'sentence'>

        >>> get_evaluation_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: evaluation_level must be one of
    """
    if name not in VALID_EVALUATION_LEVELS:
        msg = f"evaluation_level must be one of {VALID_EVALUATION_LEVELS}, got '{name}'"
        raise ValueError(msg)
    return EvaluationLevel(name)


def evaluate_retrieval(
    retrieved_ids: tuple[str, ...],
    relevant_ids: tuple[str, ...],
    k: int | None = None,
) -> RetrievalMetrics:
    """Evaluate retrieval quality.

    Computes precision@k, recall@k, MRR, NDCG, and hit rate.

    Args:
        retrieved_ids: Tuple of retrieved document IDs (in order).
        relevant_ids: Tuple of ground truth relevant document IDs.
        k: Number of documents to consider. Defaults to len(retrieved_ids).

    Returns:
        RetrievalMetrics with computed values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> metrics = evaluate_retrieval(
        ...     retrieved_ids=("d1", "d2", "d3", "d4", "d5"),
        ...     relevant_ids=("d1", "d3", "d6"),
        ... )
        >>> metrics.precision_at_k
        0.4
        >>> metrics.hit_rate
        1.0

        >>> metrics = evaluate_retrieval(
        ...     retrieved_ids=("d1", "d2"),
        ...     relevant_ids=("d1", "d3"),
        ...     k=2,
        ... )
        >>> metrics.recall_at_k
        0.5

        >>> evaluate_retrieval((), ("d1",))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: retrieved_ids cannot be empty
    """
    if not retrieved_ids:
        msg = "retrieved_ids cannot be empty"
        raise ValueError(msg)

    if k is None:
        k = len(retrieved_ids)

    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)

    # Limit to k documents
    top_k = retrieved_ids[:k]
    relevant_set = frozenset(relevant_ids)

    # Handle empty relevant set
    if not relevant_set:
        return RetrievalMetrics(
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg=0.0,
            hit_rate=0.0,
        )

    # Precision@k: fraction of retrieved docs that are relevant
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_set)
    precision_at_k = relevant_in_top_k / len(top_k)

    # Recall@k: fraction of relevant docs that were retrieved
    recall_at_k = relevant_in_top_k / len(relevant_set)

    # MRR: reciprocal of the rank of the first relevant document
    mrr = 0.0
    for i, doc in enumerate(top_k, start=1):
        if doc in relevant_set:
            mrr = 1.0 / i
            break

    # NDCG: Normalized Discounted Cumulative Gain
    dcg = 0.0
    for i, doc in enumerate(top_k, start=1):
        if doc in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant docs at top
    ideal_relevant = min(len(relevant_set), len(top_k))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Hit rate: 1 if at least one relevant doc was retrieved
    hit_rate = 1.0 if relevant_in_top_k > 0 else 0.0

    return RetrievalMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg=ndcg,
        hit_rate=hit_rate,
    )


def evaluate_generation(
    answer: str,
    context: str,
    query: str,
) -> GenerationMetrics:
    """Evaluate generation quality.

    Computes faithfulness, relevance, coherence, and groundedness scores
    based on text overlap and structural analysis.

    Args:
        answer: Generated answer text.
        context: Retrieved context text.
        query: Original query text.

    Returns:
        GenerationMetrics with computed values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> metrics = evaluate_generation(
        ...     answer="The capital of France is Paris.",
        ...     context="Paris is the capital and most populous city of France.",
        ...     query="What is the capital of France?",
        ... )
        >>> metrics.faithfulness > 0
        True
        >>> metrics.relevance > 0
        True

        >>> evaluate_generation("", "context", "query")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: answer cannot be empty
    """
    if not answer:
        msg = "answer cannot be empty"
        raise ValueError(msg)

    if not context:
        msg = "context cannot be empty"
        raise ValueError(msg)

    if not query:
        msg = "query cannot be empty"
        raise ValueError(msg)

    # Normalize texts
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    query_words = set(query.lower().split())

    # Faithfulness: how much of the answer is grounded in context
    if answer_words:
        overlap_with_context = len(answer_words & context_words)
        faithfulness = overlap_with_context / len(answer_words)
    else:
        faithfulness = 0.0

    # Relevance: how much the answer addresses the query
    if answer_words:
        overlap_with_query = len(answer_words & query_words)
        # Normalize by query length to reward covering query terms
        relevance = min(1.0, overlap_with_query / max(len(query_words), 1))
    else:
        relevance = 0.0

    # Coherence: based on answer length and structure (simple heuristic)
    # Longer, complete sentences are more coherent
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if sentences:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        # Optimal sentence length is around 10-20 words
        coherence = min(1.0, avg_sentence_length / 15.0)
    else:
        coherence = 0.0

    # Groundedness: combination of faithfulness and context coverage
    groundedness = (faithfulness + min(1.0, len(answer_words) / 10)) / 2

    return GenerationMetrics(
        faithfulness=min(1.0, faithfulness),
        relevance=min(1.0, relevance),
        coherence=min(1.0, coherence),
        groundedness=min(1.0, groundedness),
    )


def evaluate_faithfulness(
    answer: str,
    context: str,
) -> float:
    """Evaluate faithfulness of answer to context.

    Faithfulness measures how well the answer is supported by the context,
    penalizing hallucinated or unsupported information.

    Args:
        answer: Generated answer text.
        context: Retrieved context text.

    Returns:
        Faithfulness score between 0.0 and 1.0.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> score = evaluate_faithfulness(
        ...     answer="Paris is the capital of France.",
        ...     context="Paris is the capital and largest city of France.",
        ... )
        >>> score > 0.5
        True

        >>> evaluate_faithfulness("", "context")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: answer cannot be empty
    """
    if not answer:
        msg = "answer cannot be empty"
        raise ValueError(msg)

    if not context:
        msg = "context cannot be empty"
        raise ValueError(msg)

    # Tokenize and normalize
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return 0.0

    # Calculate word overlap
    overlap = len(answer_words & context_words)
    faithfulness = overlap / len(answer_words)

    return min(1.0, faithfulness)


def calculate_groundedness(
    answer: str,
    sources: tuple[str, ...],
    grounding_type: GroundingType = GroundingType.FACTUAL,
) -> float:
    """Calculate groundedness score for an answer.

    Groundedness measures how well the answer is supported by source documents.

    Args:
        answer: Generated answer text.
        sources: Tuple of source document texts.
        grounding_type: Type of grounding to evaluate.

    Returns:
        Groundedness score between 0.0 and 1.0.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> score = calculate_groundedness(
        ...     answer="The sky is blue.",
        ...     sources=("The sky appears blue due to scattering.",),
        ... )
        >>> score > 0
        True

        >>> score = calculate_groundedness(
        ...     answer="test",
        ...     sources=("source1", "source2"),
        ...     grounding_type=GroundingType.CITATION,
        ... )
        >>> 0 <= score <= 1
        True

        >>> calculate_groundedness("", ("source",))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: answer cannot be empty
    """
    if not answer:
        msg = "answer cannot be empty"
        raise ValueError(msg)

    if not sources:
        msg = "sources cannot be empty"
        raise ValueError(msg)

    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0

    # Combine all source words
    source_words: set[str] = set()
    for source in sources:
        source_words.update(source.lower().split())

    # Base groundedness from word overlap
    overlap = len(answer_words & source_words)
    base_score = overlap / len(answer_words)

    # Apply grounding type modifier
    if grounding_type == GroundingType.CITATION:
        # Citation requires explicit references - stricter
        modifier = 0.9
    elif grounding_type == GroundingType.ATTRIBUTION:
        # Attribution is moderately strict
        modifier = 0.95
    else:  # FACTUAL
        # Factual is most lenient
        modifier = 1.0

    return min(1.0, base_score * modifier)


def compute_rag_score(
    retrieval_metrics: RetrievalMetrics,
    generation_metrics: GenerationMetrics,
    retrieval_weight: float = 0.4,
    generation_weight: float = 0.6,
) -> float:
    """Compute overall RAG score from component metrics.

    Args:
        retrieval_metrics: Retrieval quality metrics.
        generation_metrics: Generation quality metrics.
        retrieval_weight: Weight for retrieval score. Defaults to 0.4.
        generation_weight: Weight for generation score. Defaults to 0.6.

    Returns:
        Combined RAG score between 0.0 and 1.0.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        >>> generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        >>> score = compute_rag_score(retrieval, generation)
        >>> 0 <= score <= 1
        True

        >>> compute_rag_score(retrieval, generation, -0.1, 0.6)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: retrieval_weight must be non-negative
    """
    if retrieval_weight < 0:
        msg = f"retrieval_weight must be non-negative, got {retrieval_weight}"
        raise ValueError(msg)

    if generation_weight < 0:
        msg = f"generation_weight must be non-negative, got {generation_weight}"
        raise ValueError(msg)

    total_weight = retrieval_weight + generation_weight
    if total_weight <= 0:
        msg = "retrieval_weight + generation_weight must be positive"
        raise ValueError(msg)

    # Compute average retrieval score
    retrieval_score = (
        retrieval_metrics.precision_at_k
        + retrieval_metrics.recall_at_k
        + retrieval_metrics.mrr
        + retrieval_metrics.ndcg
        + retrieval_metrics.hit_rate
    ) / 5.0

    # Compute average generation score
    generation_score = (
        generation_metrics.faithfulness
        + generation_metrics.relevance
        + generation_metrics.coherence
        + generation_metrics.groundedness
    ) / 4.0

    # Weighted combination
    combined = (
        retrieval_score * retrieval_weight + generation_score * generation_weight
    ) / total_weight

    return min(1.0, max(0.0, combined))


def format_rag_eval_stats(stats: RAGEvalStats) -> str:
    """Format RAG evaluation stats for display.

    Args:
        stats: RAG evaluation statistics.

    Returns:
        Formatted string.

    Examples:
        >>> stats = create_rag_eval_stats(0.78, 0.85, 100, 2)
        >>> formatted = format_rag_eval_stats(stats)
        >>> "100 queries" in formatted
        True
        >>> "retrieval=0.78" in formatted
        True
    """
    parts = [
        f"RAG Evaluation: {stats.num_queries} queries",
        f"retrieval={stats.avg_retrieval_score:.2f}",
        f"generation={stats.avg_generation_score:.2f}",
    ]

    if stats.failed_queries > 0:
        parts.append(f"failed={stats.failed_queries}")

    return " | ".join(parts)


def get_recommended_rag_eval_config(
    use_case: str = "qa",
    strict: bool = False,
) -> RAGEvalConfig:
    """Get recommended RAG evaluation configuration for a use case.

    Args:
        use_case: Use case type ("qa", "summarization", "chat").
        strict: Whether to use strict evaluation settings.

    Returns:
        Recommended RAGEvalConfig.

    Examples:
        >>> config = get_recommended_rag_eval_config("qa")
        >>> config.retrieval_k
        10

        >>> config = get_recommended_rag_eval_config("summarization")
        >>> config.retrieval_k
        5

        >>> config = get_recommended_rag_eval_config("chat", strict=True)
        >>> config.grounding_type
        <GroundingType.CITATION: 'citation'>
    """
    use_case_configs = {
        "qa": {
            "retrieval_k": 10,
            "grounding_type": "factual",
            "evaluation_level": "passage",
        },
        "summarization": {
            "retrieval_k": 5,
            "grounding_type": "attribution",
            "evaluation_level": "document",
        },
        "chat": {
            "retrieval_k": 3,
            "grounding_type": "factual",
            "evaluation_level": "sentence",
        },
    }

    config_params = use_case_configs.get(
        use_case,
        use_case_configs["qa"],
    )

    if strict:
        config_params["grounding_type"] = "citation"
        config_params["evaluation_level"] = "sentence"

    return create_rag_eval_config(
        retrieval_k=config_params["retrieval_k"],
        grounding_type=config_params["grounding_type"],
        evaluation_level=config_params["evaluation_level"],
    )
