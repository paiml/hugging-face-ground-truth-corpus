"""Tests for rag.evaluation module."""

from __future__ import annotations

import pytest

from hf_gtc.rag.evaluation import (
    VALID_EVALUATION_LEVELS,
    VALID_GROUNDING_TYPES,
    VALID_RAG_METRICS,
    EvaluationLevel,
    GenerationMetrics,
    GroundingType,
    RAGEvalConfig,
    RAGEvalResult,
    RAGEvalStats,
    RAGMetric,
    RetrievalMetrics,
    calculate_groundedness,
    compute_rag_score,
    create_generation_metrics,
    create_rag_eval_config,
    create_rag_eval_result,
    create_rag_eval_stats,
    create_retrieval_metrics,
    evaluate_faithfulness,
    evaluate_generation,
    evaluate_retrieval,
    format_rag_eval_stats,
    get_evaluation_level,
    get_grounding_type,
    get_rag_metric,
    get_recommended_rag_eval_config,
    list_evaluation_levels,
    list_grounding_types,
    list_rag_metrics,
    validate_generation_metrics,
    validate_rag_eval_config,
    validate_rag_eval_result,
    validate_rag_eval_stats,
    validate_retrieval_metrics,
)


class TestRAGMetric:
    """Tests for RAGMetric enum."""

    def test_all_metrics_have_values(self) -> None:
        """All RAG metrics have string values."""
        for metric in RAGMetric:
            assert isinstance(metric.value, str)

    def test_retrieval_precision_value(self) -> None:
        """Retrieval precision has correct value."""
        assert RAGMetric.RETRIEVAL_PRECISION.value == "retrieval_precision"

    def test_retrieval_recall_value(self) -> None:
        """Retrieval recall has correct value."""
        assert RAGMetric.RETRIEVAL_RECALL.value == "retrieval_recall"

    def test_context_relevance_value(self) -> None:
        """Context relevance has correct value."""
        assert RAGMetric.CONTEXT_RELEVANCE.value == "context_relevance"

    def test_faithfulness_value(self) -> None:
        """Faithfulness has correct value."""
        assert RAGMetric.FAITHFULNESS.value == "faithfulness"

    def test_answer_relevance_value(self) -> None:
        """Answer relevance has correct value."""
        assert RAGMetric.ANSWER_RELEVANCE.value == "answer_relevance"

    def test_valid_rag_metrics_frozenset(self) -> None:
        """VALID_RAG_METRICS is a frozenset."""
        assert isinstance(VALID_RAG_METRICS, frozenset)

    def test_valid_rag_metrics_contains_all(self) -> None:
        """VALID_RAG_METRICS contains all enum values."""
        for metric in RAGMetric:
            assert metric.value in VALID_RAG_METRICS


class TestGroundingType:
    """Tests for GroundingType enum."""

    def test_all_types_have_values(self) -> None:
        """All grounding types have string values."""
        for grounding_type in GroundingType:
            assert isinstance(grounding_type.value, str)

    def test_citation_value(self) -> None:
        """Citation has correct value."""
        assert GroundingType.CITATION.value == "citation"

    def test_attribution_value(self) -> None:
        """Attribution has correct value."""
        assert GroundingType.ATTRIBUTION.value == "attribution"

    def test_factual_value(self) -> None:
        """Factual has correct value."""
        assert GroundingType.FACTUAL.value == "factual"

    def test_valid_grounding_types_frozenset(self) -> None:
        """VALID_GROUNDING_TYPES is a frozenset."""
        assert isinstance(VALID_GROUNDING_TYPES, frozenset)

    def test_valid_grounding_types_contains_all(self) -> None:
        """VALID_GROUNDING_TYPES contains all enum values."""
        for grounding_type in GroundingType:
            assert grounding_type.value in VALID_GROUNDING_TYPES


class TestEvaluationLevel:
    """Tests for EvaluationLevel enum."""

    def test_all_levels_have_values(self) -> None:
        """All evaluation levels have string values."""
        for level in EvaluationLevel:
            assert isinstance(level.value, str)

    def test_document_value(self) -> None:
        """Document has correct value."""
        assert EvaluationLevel.DOCUMENT.value == "document"

    def test_passage_value(self) -> None:
        """Passage has correct value."""
        assert EvaluationLevel.PASSAGE.value == "passage"

    def test_sentence_value(self) -> None:
        """Sentence has correct value."""
        assert EvaluationLevel.SENTENCE.value == "sentence"

    def test_valid_evaluation_levels_frozenset(self) -> None:
        """VALID_EVALUATION_LEVELS is a frozenset."""
        assert isinstance(VALID_EVALUATION_LEVELS, frozenset)

    def test_valid_evaluation_levels_contains_all(self) -> None:
        """VALID_EVALUATION_LEVELS contains all enum values."""
        for level in EvaluationLevel:
            assert level.value in VALID_EVALUATION_LEVELS


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Create retrieval metrics."""
        metrics = RetrievalMetrics(
            precision_at_k=0.8,
            recall_at_k=0.6,
            mrr=0.75,
            ndcg=0.82,
            hit_rate=1.0,
        )
        assert metrics.precision_at_k == pytest.approx(0.8)
        assert metrics.recall_at_k == pytest.approx(0.6)
        assert metrics.mrr == pytest.approx(0.75)
        assert metrics.ndcg == pytest.approx(0.82)
        assert metrics.hit_rate == pytest.approx(1.0)

    def test_metrics_is_frozen(self) -> None:
        """Metrics is immutable."""
        metrics = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        with pytest.raises(AttributeError):
            metrics.precision_at_k = 0.9  # type: ignore[misc]


class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Create generation metrics."""
        metrics = GenerationMetrics(
            faithfulness=0.9,
            relevance=0.85,
            coherence=0.88,
            groundedness=0.92,
        )
        assert metrics.faithfulness == pytest.approx(0.9)
        assert metrics.relevance == pytest.approx(0.85)
        assert metrics.coherence == pytest.approx(0.88)
        assert metrics.groundedness == pytest.approx(0.92)

    def test_metrics_is_frozen(self) -> None:
        """Metrics is immutable."""
        metrics = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        with pytest.raises(AttributeError):
            metrics.faithfulness = 0.95  # type: ignore[misc]


class TestRAGEvalConfig:
    """Tests for RAGEvalConfig dataclass."""

    def test_create_config(self) -> None:
        """Create RAG eval config."""
        config = RAGEvalConfig(
            retrieval_k=10,
            metrics=(RAGMetric.RETRIEVAL_PRECISION, RAGMetric.FAITHFULNESS),
            grounding_type=GroundingType.CITATION,
            evaluation_level=EvaluationLevel.PASSAGE,
        )
        assert config.retrieval_k == 10
        assert len(config.metrics) == 2
        assert config.grounding_type == GroundingType.CITATION
        assert config.evaluation_level == EvaluationLevel.PASSAGE

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RAGEvalConfig(
            retrieval_k=10,
            metrics=(RAGMetric.FAITHFULNESS,),
            grounding_type=GroundingType.CITATION,
            evaluation_level=EvaluationLevel.DOCUMENT,
        )
        with pytest.raises(AttributeError):
            config.retrieval_k = 20  # type: ignore[misc]


class TestRAGEvalResult:
    """Tests for RAGEvalResult dataclass."""

    def test_create_result(self) -> None:
        """Create RAG eval result."""
        retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(
            retrieval_metrics=retrieval,
            generation_metrics=generation,
            overall_score=0.85,
            per_query_scores=(0.82, 0.88, 0.85),
        )
        assert result.overall_score == pytest.approx(0.85)
        assert len(result.per_query_scores) == 3

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(retrieval, generation, 0.85, (0.8,))
        with pytest.raises(AttributeError):
            result.overall_score = 0.9  # type: ignore[misc]


class TestRAGEvalStats:
    """Tests for RAGEvalStats dataclass."""

    def test_create_stats(self) -> None:
        """Create RAG eval stats."""
        stats = RAGEvalStats(
            avg_retrieval_score=0.78,
            avg_generation_score=0.85,
            num_queries=100,
            failed_queries=2,
        )
        assert stats.avg_retrieval_score == pytest.approx(0.78)
        assert stats.avg_generation_score == pytest.approx(0.85)
        assert stats.num_queries == 100
        assert stats.failed_queries == 2

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = RAGEvalStats(0.78, 0.85, 100, 2)
        with pytest.raises(AttributeError):
            stats.num_queries = 200  # type: ignore[misc]


class TestValidateRetrievalMetrics:
    """Tests for validate_retrieval_metrics function."""

    def test_valid_metrics(self) -> None:
        """Valid metrics pass validation."""
        metrics = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        validate_retrieval_metrics(metrics)  # Should not raise

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        metrics = RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        validate_retrieval_metrics(metrics)  # Should not raise

    def test_one_values_valid(self) -> None:
        """One values are valid."""
        metrics = RetrievalMetrics(1.0, 1.0, 1.0, 1.0, 1.0)
        validate_retrieval_metrics(metrics)  # Should not raise

    def test_negative_precision_raises(self) -> None:
        """Negative precision raises ValueError."""
        metrics = RetrievalMetrics(-0.1, 0.6, 0.75, 0.82, 1.0)
        with pytest.raises(ValueError, match="precision_at_k must be between"):
            validate_retrieval_metrics(metrics)

    def test_precision_over_one_raises(self) -> None:
        """Precision over 1.0 raises ValueError."""
        metrics = RetrievalMetrics(1.1, 0.6, 0.75, 0.82, 1.0)
        with pytest.raises(ValueError, match="precision_at_k must be between"):
            validate_retrieval_metrics(metrics)

    def test_negative_recall_raises(self) -> None:
        """Negative recall raises ValueError."""
        metrics = RetrievalMetrics(0.8, -0.1, 0.75, 0.82, 1.0)
        with pytest.raises(ValueError, match="recall_at_k must be between"):
            validate_retrieval_metrics(metrics)

    def test_recall_over_one_raises(self) -> None:
        """Recall over 1.0 raises ValueError."""
        metrics = RetrievalMetrics(0.8, 1.1, 0.75, 0.82, 1.0)
        with pytest.raises(ValueError, match="recall_at_k must be between"):
            validate_retrieval_metrics(metrics)

    def test_negative_mrr_raises(self) -> None:
        """Negative MRR raises ValueError."""
        metrics = RetrievalMetrics(0.8, 0.6, -0.1, 0.82, 1.0)
        with pytest.raises(ValueError, match="mrr must be between"):
            validate_retrieval_metrics(metrics)

    def test_mrr_over_one_raises(self) -> None:
        """MRR over 1.0 raises ValueError."""
        metrics = RetrievalMetrics(0.8, 0.6, 1.1, 0.82, 1.0)
        with pytest.raises(ValueError, match="mrr must be between"):
            validate_retrieval_metrics(metrics)

    def test_negative_ndcg_raises(self) -> None:
        """Negative NDCG raises ValueError."""
        metrics = RetrievalMetrics(0.8, 0.6, 0.75, -0.1, 1.0)
        with pytest.raises(ValueError, match="ndcg must be between"):
            validate_retrieval_metrics(metrics)

    def test_ndcg_over_one_raises(self) -> None:
        """NDCG over 1.0 raises ValueError."""
        metrics = RetrievalMetrics(0.8, 0.6, 0.75, 1.1, 1.0)
        with pytest.raises(ValueError, match="ndcg must be between"):
            validate_retrieval_metrics(metrics)

    def test_negative_hit_rate_raises(self) -> None:
        """Negative hit_rate raises ValueError."""
        metrics = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, -0.1)
        with pytest.raises(ValueError, match="hit_rate must be between"):
            validate_retrieval_metrics(metrics)

    def test_hit_rate_over_one_raises(self) -> None:
        """hit_rate over 1.0 raises ValueError."""
        metrics = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.1)
        with pytest.raises(ValueError, match="hit_rate must be between"):
            validate_retrieval_metrics(metrics)


class TestValidateGenerationMetrics:
    """Tests for validate_generation_metrics function."""

    def test_valid_metrics(self) -> None:
        """Valid metrics pass validation."""
        metrics = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        validate_generation_metrics(metrics)  # Should not raise

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        metrics = GenerationMetrics(0.0, 0.0, 0.0, 0.0)
        validate_generation_metrics(metrics)  # Should not raise

    def test_one_values_valid(self) -> None:
        """One values are valid."""
        metrics = GenerationMetrics(1.0, 1.0, 1.0, 1.0)
        validate_generation_metrics(metrics)  # Should not raise

    def test_negative_faithfulness_raises(self) -> None:
        """Negative faithfulness raises ValueError."""
        metrics = GenerationMetrics(-0.1, 0.85, 0.88, 0.92)
        with pytest.raises(ValueError, match="faithfulness must be between"):
            validate_generation_metrics(metrics)

    def test_faithfulness_over_one_raises(self) -> None:
        """Faithfulness over 1.0 raises ValueError."""
        metrics = GenerationMetrics(1.1, 0.85, 0.88, 0.92)
        with pytest.raises(ValueError, match="faithfulness must be between"):
            validate_generation_metrics(metrics)

    def test_negative_relevance_raises(self) -> None:
        """Negative relevance raises ValueError."""
        metrics = GenerationMetrics(0.9, -0.1, 0.88, 0.92)
        with pytest.raises(ValueError, match="relevance must be between"):
            validate_generation_metrics(metrics)

    def test_relevance_over_one_raises(self) -> None:
        """Relevance over 1.0 raises ValueError."""
        metrics = GenerationMetrics(0.9, 1.1, 0.88, 0.92)
        with pytest.raises(ValueError, match="relevance must be between"):
            validate_generation_metrics(metrics)

    def test_negative_coherence_raises(self) -> None:
        """Negative coherence raises ValueError."""
        metrics = GenerationMetrics(0.9, 0.85, -0.1, 0.92)
        with pytest.raises(ValueError, match="coherence must be between"):
            validate_generation_metrics(metrics)

    def test_coherence_over_one_raises(self) -> None:
        """Coherence over 1.0 raises ValueError."""
        metrics = GenerationMetrics(0.9, 0.85, 1.1, 0.92)
        with pytest.raises(ValueError, match="coherence must be between"):
            validate_generation_metrics(metrics)

    def test_negative_groundedness_raises(self) -> None:
        """Negative groundedness raises ValueError."""
        metrics = GenerationMetrics(0.9, 0.85, 0.88, -0.1)
        with pytest.raises(ValueError, match="groundedness must be between"):
            validate_generation_metrics(metrics)

    def test_groundedness_over_one_raises(self) -> None:
        """Groundedness over 1.0 raises ValueError."""
        metrics = GenerationMetrics(0.9, 0.85, 0.88, 1.1)
        with pytest.raises(ValueError, match="groundedness must be between"):
            validate_generation_metrics(metrics)


class TestValidateRAGEvalConfig:
    """Tests for validate_rag_eval_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RAGEvalConfig(
            retrieval_k=10,
            metrics=(RAGMetric.RETRIEVAL_PRECISION,),
            grounding_type=GroundingType.CITATION,
            evaluation_level=EvaluationLevel.DOCUMENT,
        )
        validate_rag_eval_config(config)  # Should not raise

    def test_zero_retrieval_k_raises(self) -> None:
        """Zero retrieval_k raises ValueError."""
        config = RAGEvalConfig(
            retrieval_k=0,
            metrics=(RAGMetric.RETRIEVAL_PRECISION,),
            grounding_type=GroundingType.CITATION,
            evaluation_level=EvaluationLevel.DOCUMENT,
        )
        with pytest.raises(ValueError, match="retrieval_k must be positive"):
            validate_rag_eval_config(config)

    def test_negative_retrieval_k_raises(self) -> None:
        """Negative retrieval_k raises ValueError."""
        config = RAGEvalConfig(
            retrieval_k=-1,
            metrics=(RAGMetric.RETRIEVAL_PRECISION,),
            grounding_type=GroundingType.CITATION,
            evaluation_level=EvaluationLevel.DOCUMENT,
        )
        with pytest.raises(ValueError, match="retrieval_k must be positive"):
            validate_rag_eval_config(config)

    def test_empty_metrics_raises(self) -> None:
        """Empty metrics raises ValueError."""
        config = RAGEvalConfig(
            retrieval_k=10,
            metrics=(),
            grounding_type=GroundingType.CITATION,
            evaluation_level=EvaluationLevel.DOCUMENT,
        )
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            validate_rag_eval_config(config)


class TestValidateRAGEvalResult:
    """Tests for validate_rag_eval_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes validation."""
        retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(retrieval, generation, 0.85, (0.8, 0.9))
        validate_rag_eval_result(result)  # Should not raise

    def test_invalid_overall_score_raises(self) -> None:
        """Invalid overall_score raises ValueError."""
        retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(retrieval, generation, 1.5, (0.8,))
        with pytest.raises(ValueError, match="overall_score must be between"):
            validate_rag_eval_result(result)

    def test_negative_overall_score_raises(self) -> None:
        """Negative overall_score raises ValueError."""
        retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(retrieval, generation, -0.1, (0.8,))
        with pytest.raises(ValueError, match="overall_score must be between"):
            validate_rag_eval_result(result)

    def test_invalid_per_query_score_raises(self) -> None:
        """Invalid per_query_scores raises ValueError."""
        retrieval = RetrievalMetrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(retrieval, generation, 0.85, (0.8, 1.5))
        with pytest.raises(ValueError, match=r"per_query_scores\[1\] must be between"):
            validate_rag_eval_result(result)

    def test_invalid_retrieval_metrics_raises(self) -> None:
        """Invalid retrieval metrics raises ValueError."""
        retrieval = RetrievalMetrics(1.5, 0.6, 0.75, 0.82, 1.0)
        generation = GenerationMetrics(0.9, 0.85, 0.88, 0.92)
        result = RAGEvalResult(retrieval, generation, 0.85, (0.8,))
        with pytest.raises(ValueError, match="precision_at_k must be between"):
            validate_rag_eval_result(result)


class TestValidateRAGEvalStats:
    """Tests for validate_rag_eval_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats pass validation."""
        stats = RAGEvalStats(0.78, 0.85, 100, 2)
        validate_rag_eval_stats(stats)  # Should not raise

    def test_zero_failed_queries_valid(self) -> None:
        """Zero failed queries is valid."""
        stats = RAGEvalStats(0.78, 0.85, 100, 0)
        validate_rag_eval_stats(stats)  # Should not raise

    def test_negative_avg_retrieval_score_raises(self) -> None:
        """Negative avg_retrieval_score raises ValueError."""
        stats = RAGEvalStats(-0.1, 0.85, 100, 2)
        with pytest.raises(ValueError, match="avg_retrieval_score must be between"):
            validate_rag_eval_stats(stats)

    def test_avg_retrieval_score_over_one_raises(self) -> None:
        """avg_retrieval_score over 1.0 raises ValueError."""
        stats = RAGEvalStats(1.1, 0.85, 100, 2)
        with pytest.raises(ValueError, match="avg_retrieval_score must be between"):
            validate_rag_eval_stats(stats)

    def test_negative_avg_generation_score_raises(self) -> None:
        """Negative avg_generation_score raises ValueError."""
        stats = RAGEvalStats(0.78, -0.1, 100, 2)
        with pytest.raises(ValueError, match="avg_generation_score must be between"):
            validate_rag_eval_stats(stats)

    def test_avg_generation_score_over_one_raises(self) -> None:
        """avg_generation_score over 1.0 raises ValueError."""
        stats = RAGEvalStats(0.78, 1.1, 100, 2)
        with pytest.raises(ValueError, match="avg_generation_score must be between"):
            validate_rag_eval_stats(stats)

    def test_negative_num_queries_raises(self) -> None:
        """Negative num_queries raises ValueError."""
        stats = RAGEvalStats(0.78, 0.85, -1, 2)
        with pytest.raises(ValueError, match="num_queries must be non-negative"):
            validate_rag_eval_stats(stats)

    def test_negative_failed_queries_raises(self) -> None:
        """Negative failed_queries raises ValueError."""
        stats = RAGEvalStats(0.78, 0.85, 100, -1)
        with pytest.raises(ValueError, match="failed_queries must be non-negative"):
            validate_rag_eval_stats(stats)

    def test_failed_exceeds_num_queries_raises(self) -> None:
        """failed_queries exceeding num_queries raises ValueError."""
        stats = RAGEvalStats(0.78, 0.85, 10, 20)
        with pytest.raises(ValueError, match=r"failed_queries .* cannot exceed"):
            validate_rag_eval_stats(stats)


class TestCreateRetrievalMetrics:
    """Tests for create_retrieval_metrics function."""

    def test_default_metrics(self) -> None:
        """Create default metrics."""
        metrics = create_retrieval_metrics()
        assert metrics.precision_at_k == pytest.approx(0.0)
        assert metrics.recall_at_k == pytest.approx(0.0)
        assert metrics.mrr == pytest.approx(0.0)
        assert metrics.ndcg == pytest.approx(0.0)
        assert metrics.hit_rate == pytest.approx(0.0)

    def test_custom_precision(self) -> None:
        """Create metrics with custom precision."""
        metrics = create_retrieval_metrics(precision_at_k=0.8)
        assert metrics.precision_at_k == pytest.approx(0.8)

    def test_custom_recall(self) -> None:
        """Create metrics with custom recall."""
        metrics = create_retrieval_metrics(recall_at_k=0.6)
        assert metrics.recall_at_k == pytest.approx(0.6)

    def test_custom_mrr(self) -> None:
        """Create metrics with custom MRR."""
        metrics = create_retrieval_metrics(mrr=0.75)
        assert metrics.mrr == pytest.approx(0.75)

    def test_custom_ndcg(self) -> None:
        """Create metrics with custom NDCG."""
        metrics = create_retrieval_metrics(ndcg=0.82)
        assert metrics.ndcg == pytest.approx(0.82)

    def test_custom_hit_rate(self) -> None:
        """Create metrics with custom hit_rate."""
        metrics = create_retrieval_metrics(hit_rate=1.0)
        assert metrics.hit_rate == pytest.approx(1.0)

    def test_invalid_precision_raises(self) -> None:
        """Invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="precision_at_k must be between"):
            create_retrieval_metrics(precision_at_k=1.5)


class TestCreateGenerationMetrics:
    """Tests for create_generation_metrics function."""

    def test_default_metrics(self) -> None:
        """Create default metrics."""
        metrics = create_generation_metrics()
        assert metrics.faithfulness == pytest.approx(0.0)
        assert metrics.relevance == pytest.approx(0.0)
        assert metrics.coherence == pytest.approx(0.0)
        assert metrics.groundedness == pytest.approx(0.0)

    def test_custom_faithfulness(self) -> None:
        """Create metrics with custom faithfulness."""
        metrics = create_generation_metrics(faithfulness=0.9)
        assert metrics.faithfulness == pytest.approx(0.9)

    def test_custom_relevance(self) -> None:
        """Create metrics with custom relevance."""
        metrics = create_generation_metrics(relevance=0.85)
        assert metrics.relevance == pytest.approx(0.85)

    def test_custom_coherence(self) -> None:
        """Create metrics with custom coherence."""
        metrics = create_generation_metrics(coherence=0.88)
        assert metrics.coherence == pytest.approx(0.88)

    def test_custom_groundedness(self) -> None:
        """Create metrics with custom groundedness."""
        metrics = create_generation_metrics(groundedness=0.92)
        assert metrics.groundedness == pytest.approx(0.92)

    def test_invalid_faithfulness_raises(self) -> None:
        """Invalid faithfulness raises ValueError."""
        with pytest.raises(ValueError, match="faithfulness must be between"):
            create_generation_metrics(faithfulness=1.5)


class TestCreateRAGEvalConfig:
    """Tests for create_rag_eval_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_rag_eval_config()
        assert config.retrieval_k == 10
        assert len(config.metrics) == len(RAGMetric)
        assert config.grounding_type == GroundingType.CITATION
        assert config.evaluation_level == EvaluationLevel.DOCUMENT

    def test_custom_retrieval_k(self) -> None:
        """Create config with custom retrieval_k."""
        config = create_rag_eval_config(retrieval_k=5)
        assert config.retrieval_k == 5

    def test_custom_metrics(self) -> None:
        """Create config with custom metrics."""
        config = create_rag_eval_config(metrics=("faithfulness", "retrieval_precision"))
        assert len(config.metrics) == 2
        assert RAGMetric.FAITHFULNESS in config.metrics
        assert RAGMetric.RETRIEVAL_PRECISION in config.metrics

    @pytest.mark.parametrize(
        ("grounding_type", "expected"),
        [
            ("citation", GroundingType.CITATION),
            ("attribution", GroundingType.ATTRIBUTION),
            ("factual", GroundingType.FACTUAL),
        ],
    )
    def test_all_grounding_types(
        self, grounding_type: str, expected: GroundingType
    ) -> None:
        """Create config with all grounding types."""
        config = create_rag_eval_config(grounding_type=grounding_type)
        assert config.grounding_type == expected

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            ("document", EvaluationLevel.DOCUMENT),
            ("passage", EvaluationLevel.PASSAGE),
            ("sentence", EvaluationLevel.SENTENCE),
        ],
    )
    def test_all_evaluation_levels(self, level: str, expected: EvaluationLevel) -> None:
        """Create config with all evaluation levels."""
        config = create_rag_eval_config(evaluation_level=level)
        assert config.evaluation_level == expected

    def test_invalid_grounding_type_raises(self) -> None:
        """Invalid grounding_type raises ValueError."""
        with pytest.raises(ValueError, match="grounding_type must be one of"):
            create_rag_eval_config(grounding_type="invalid")

    def test_invalid_evaluation_level_raises(self) -> None:
        """Invalid evaluation_level raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_level must be one of"):
            create_rag_eval_config(evaluation_level="invalid")

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            create_rag_eval_config(metrics=("invalid",))

    def test_zero_retrieval_k_raises(self) -> None:
        """Zero retrieval_k raises ValueError."""
        with pytest.raises(ValueError, match="retrieval_k must be positive"):
            create_rag_eval_config(retrieval_k=0)


class TestCreateRAGEvalResult:
    """Tests for create_rag_eval_result function."""

    def test_create_result(self) -> None:
        """Create RAG eval result."""
        retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        result = create_rag_eval_result(retrieval, generation, 0.85, (0.82, 0.88))
        assert result.overall_score == pytest.approx(0.85)
        assert len(result.per_query_scores) == 2

    def test_empty_per_query_scores(self) -> None:
        """Create result with empty per_query_scores."""
        retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        result = create_rag_eval_result(retrieval, generation, 0.85, ())
        assert result.per_query_scores == ()

    def test_invalid_overall_score_raises(self) -> None:
        """Invalid overall_score raises ValueError."""
        retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        with pytest.raises(ValueError, match="overall_score must be between"):
            create_rag_eval_result(retrieval, generation, 1.5, (0.8,))


class TestCreateRAGEvalStats:
    """Tests for create_rag_eval_stats function."""

    def test_create_stats(self) -> None:
        """Create RAG eval stats."""
        stats = create_rag_eval_stats(0.78, 0.85, 100, 2)
        assert stats.avg_retrieval_score == pytest.approx(0.78)
        assert stats.avg_generation_score == pytest.approx(0.85)
        assert stats.num_queries == 100
        assert stats.failed_queries == 2

    def test_default_failed_queries(self) -> None:
        """Create stats with default failed_queries."""
        stats = create_rag_eval_stats(0.78, 0.85, 100)
        assert stats.failed_queries == 0

    def test_invalid_num_queries_raises(self) -> None:
        """Invalid num_queries raises ValueError."""
        with pytest.raises(ValueError, match="num_queries must be non-negative"):
            create_rag_eval_stats(0.78, 0.85, -1)


class TestListRAGMetrics:
    """Tests for list_rag_metrics function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        metrics = list_rag_metrics()
        assert metrics == sorted(metrics)

    def test_contains_faithfulness(self) -> None:
        """Contains faithfulness."""
        metrics = list_rag_metrics()
        assert "faithfulness" in metrics

    def test_contains_retrieval_precision(self) -> None:
        """Contains retrieval_precision."""
        metrics = list_rag_metrics()
        assert "retrieval_precision" in metrics

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        metrics = list_rag_metrics()
        for metric in RAGMetric:
            assert metric.value in metrics


class TestListGroundingTypes:
    """Tests for list_grounding_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_grounding_types()
        assert types == sorted(types)

    def test_contains_citation(self) -> None:
        """Contains citation."""
        types = list_grounding_types()
        assert "citation" in types

    def test_contains_factual(self) -> None:
        """Contains factual."""
        types = list_grounding_types()
        assert "factual" in types

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        types = list_grounding_types()
        for grounding_type in GroundingType:
            assert grounding_type.value in types


class TestListEvaluationLevels:
    """Tests for list_evaluation_levels function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        levels = list_evaluation_levels()
        assert levels == sorted(levels)

    def test_contains_document(self) -> None:
        """Contains document."""
        levels = list_evaluation_levels()
        assert "document" in levels

    def test_contains_sentence(self) -> None:
        """Contains sentence."""
        levels = list_evaluation_levels()
        assert "sentence" in levels

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        levels = list_evaluation_levels()
        for level in EvaluationLevel:
            assert level.value in levels


class TestGetRAGMetric:
    """Tests for get_rag_metric function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("retrieval_precision", RAGMetric.RETRIEVAL_PRECISION),
            ("retrieval_recall", RAGMetric.RETRIEVAL_RECALL),
            ("context_relevance", RAGMetric.CONTEXT_RELEVANCE),
            ("faithfulness", RAGMetric.FAITHFULNESS),
            ("answer_relevance", RAGMetric.ANSWER_RELEVANCE),
        ],
    )
    def test_get_all_metrics(self, name: str, expected: RAGMetric) -> None:
        """Get all RAG metrics."""
        assert get_rag_metric(name) == expected

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            get_rag_metric("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            get_rag_metric("")


class TestGetGroundingType:
    """Tests for get_grounding_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("citation", GroundingType.CITATION),
            ("attribution", GroundingType.ATTRIBUTION),
            ("factual", GroundingType.FACTUAL),
        ],
    )
    def test_get_all_types(self, name: str, expected: GroundingType) -> None:
        """Get all grounding types."""
        assert get_grounding_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="grounding_type must be one of"):
            get_grounding_type("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="grounding_type must be one of"):
            get_grounding_type("")


class TestGetEvaluationLevel:
    """Tests for get_evaluation_level function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("document", EvaluationLevel.DOCUMENT),
            ("passage", EvaluationLevel.PASSAGE),
            ("sentence", EvaluationLevel.SENTENCE),
        ],
    )
    def test_get_all_levels(self, name: str, expected: EvaluationLevel) -> None:
        """Get all evaluation levels."""
        assert get_evaluation_level(name) == expected

    def test_invalid_level_raises(self) -> None:
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_level must be one of"):
            get_evaluation_level("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_level must be one of"):
            get_evaluation_level("")


class TestEvaluateRetrieval:
    """Tests for evaluate_retrieval function."""

    def test_perfect_retrieval(self) -> None:
        """Perfect retrieval scores all 1.0."""
        metrics = evaluate_retrieval(
            retrieved_ids=("d1", "d2", "d3"),
            relevant_ids=("d1", "d2", "d3"),
        )
        assert metrics.precision_at_k == pytest.approx(1.0)
        assert metrics.recall_at_k == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.ndcg == pytest.approx(1.0)
        assert metrics.hit_rate == pytest.approx(1.0)

    def test_partial_retrieval(self) -> None:
        """Partial retrieval with some relevant documents."""
        metrics = evaluate_retrieval(
            retrieved_ids=("d1", "d2", "d3", "d4", "d5"),
            relevant_ids=("d1", "d3", "d6"),
        )
        assert metrics.precision_at_k == pytest.approx(0.4)  # 2/5
        assert metrics.recall_at_k == pytest.approx(2 / 3)  # 2/3
        assert metrics.mrr == pytest.approx(1.0)  # First relevant at position 1
        assert metrics.hit_rate == pytest.approx(1.0)

    def test_no_relevant_retrieved(self) -> None:
        """No relevant documents retrieved."""
        metrics = evaluate_retrieval(
            retrieved_ids=("d1", "d2", "d3"),
            relevant_ids=("d4", "d5", "d6"),
        )
        assert metrics.precision_at_k == pytest.approx(0.0)
        assert metrics.recall_at_k == pytest.approx(0.0)
        assert metrics.mrr == pytest.approx(0.0)
        assert metrics.ndcg == pytest.approx(0.0)
        assert metrics.hit_rate == pytest.approx(0.0)

    def test_empty_relevant_set(self) -> None:
        """Empty relevant set returns zeros."""
        metrics = evaluate_retrieval(
            retrieved_ids=("d1", "d2", "d3"),
            relevant_ids=(),
        )
        assert metrics.precision_at_k == pytest.approx(0.0)
        assert metrics.recall_at_k == pytest.approx(0.0)
        assert metrics.mrr == pytest.approx(0.0)
        assert metrics.hit_rate == pytest.approx(0.0)

    def test_custom_k(self) -> None:
        """Custom k value limits evaluation."""
        metrics = evaluate_retrieval(
            retrieved_ids=("d1", "d2", "d3", "d4", "d5"),
            relevant_ids=("d1", "d3"),
            k=2,
        )
        assert metrics.precision_at_k == pytest.approx(0.5)  # 1/2
        assert metrics.recall_at_k == pytest.approx(0.5)  # 1/2

    def test_mrr_calculation(self) -> None:
        """MRR calculated correctly for different first positions."""
        # First relevant at position 1
        metrics = evaluate_retrieval(("d1", "d2"), ("d1",))
        assert metrics.mrr == pytest.approx(1.0)

        # First relevant at position 2
        metrics = evaluate_retrieval(("d2", "d1"), ("d1",))
        assert metrics.mrr == pytest.approx(0.5)

        # First relevant at position 3
        metrics = evaluate_retrieval(("d2", "d3", "d1"), ("d1",))
        assert metrics.mrr == pytest.approx(1 / 3)

    def test_empty_retrieved_raises(self) -> None:
        """Empty retrieved_ids raises ValueError."""
        with pytest.raises(ValueError, match="retrieved_ids cannot be empty"):
            evaluate_retrieval((), ("d1",))

    def test_zero_k_raises(self) -> None:
        """Zero k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            evaluate_retrieval(("d1",), ("d1",), k=0)

    def test_negative_k_raises(self) -> None:
        """Negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            evaluate_retrieval(("d1",), ("d1",), k=-1)


class TestEvaluateGeneration:
    """Tests for evaluate_generation function."""

    def test_high_overlap_generation(self) -> None:
        """High overlap between answer and context."""
        metrics = evaluate_generation(
            answer="The capital of France is Paris.",
            context="Paris is the capital and most populous city of France.",
            query="What is the capital of France?",
        )
        assert metrics.faithfulness > 0
        assert metrics.relevance > 0
        assert metrics.coherence > 0
        assert metrics.groundedness > 0

    def test_low_overlap_generation(self) -> None:
        """Low overlap between answer and context."""
        metrics = evaluate_generation(
            answer="The answer is completely unrelated text.",
            context="Paris is the capital of France.",
            query="What is the capital?",
        )
        assert metrics.faithfulness < 0.5
        assert 0 <= metrics.relevance <= 1

    def test_empty_answer_raises(self) -> None:
        """Empty answer raises ValueError."""
        with pytest.raises(ValueError, match="answer cannot be empty"):
            evaluate_generation("", "context", "query")

    def test_empty_context_raises(self) -> None:
        """Empty context raises ValueError."""
        with pytest.raises(ValueError, match="context cannot be empty"):
            evaluate_generation("answer", "", "query")

    def test_empty_query_raises(self) -> None:
        """Empty query raises ValueError."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            evaluate_generation("answer", "context", "")

    def test_metrics_in_valid_range(self) -> None:
        """All metrics are in valid range [0, 1]."""
        metrics = evaluate_generation(
            answer="Some answer text here.",
            context="Some context text here.",
            query="What is the question?",
        )
        assert 0 <= metrics.faithfulness <= 1
        assert 0 <= metrics.relevance <= 1
        assert 0 <= metrics.coherence <= 1
        assert 0 <= metrics.groundedness <= 1


class TestEvaluateFaithfulness:
    """Tests for evaluate_faithfulness function."""

    def test_high_faithfulness(self) -> None:
        """High faithfulness with good overlap."""
        score = evaluate_faithfulness(
            answer="Paris is the capital of France.",
            context="Paris is the capital and largest city of France.",
        )
        assert score > 0.5

    def test_low_faithfulness(self) -> None:
        """Low faithfulness with poor overlap."""
        score = evaluate_faithfulness(
            answer="Tokyo is in Japan.",
            context="Paris is the capital of France.",
        )
        assert score < 0.5

    def test_perfect_faithfulness(self) -> None:
        """Perfect faithfulness when answer is subset of context."""
        score = evaluate_faithfulness(
            answer="Paris France capital",
            context="Paris is the capital of France located in Europe.",
        )
        assert score == pytest.approx(1.0)

    def test_empty_answer_raises(self) -> None:
        """Empty answer raises ValueError."""
        with pytest.raises(ValueError, match="answer cannot be empty"):
            evaluate_faithfulness("", "context")

    def test_empty_context_raises(self) -> None:
        """Empty context raises ValueError."""
        with pytest.raises(ValueError, match="context cannot be empty"):
            evaluate_faithfulness("answer", "")

    def test_score_in_valid_range(self) -> None:
        """Score is in valid range [0, 1]."""
        score = evaluate_faithfulness(
            answer="Some answer text.",
            context="Some context text.",
        )
        assert 0 <= score <= 1


class TestCalculateGroundedness:
    """Tests for calculate_groundedness function."""

    def test_high_groundedness(self) -> None:
        """High groundedness with good source coverage."""
        score = calculate_groundedness(
            answer="The sky is blue.",
            sources=("The sky appears blue due to light scattering.",),
        )
        assert score > 0

    def test_multiple_sources(self) -> None:
        """Groundedness with multiple sources."""
        score = calculate_groundedness(
            answer="Paris is the capital of France.",
            sources=("Paris is a city.", "France is a country.", "Paris is capital."),
        )
        assert 0 < score <= 1

    def test_citation_grounding_type(self) -> None:
        """Citation grounding type is stricter."""
        score_factual = calculate_groundedness(
            answer="test answer",
            sources=("test source",),
            grounding_type=GroundingType.FACTUAL,
        )
        score_citation = calculate_groundedness(
            answer="test answer",
            sources=("test source",),
            grounding_type=GroundingType.CITATION,
        )
        assert score_citation <= score_factual

    def test_attribution_grounding_type(self) -> None:
        """Attribution grounding type is moderately strict."""
        score_factual = calculate_groundedness(
            answer="test answer",
            sources=("test source",),
            grounding_type=GroundingType.FACTUAL,
        )
        score_attribution = calculate_groundedness(
            answer="test answer",
            sources=("test source",),
            grounding_type=GroundingType.ATTRIBUTION,
        )
        assert score_attribution <= score_factual

    def test_empty_answer_raises(self) -> None:
        """Empty answer raises ValueError."""
        with pytest.raises(ValueError, match="answer cannot be empty"):
            calculate_groundedness("", ("source",))

    def test_empty_sources_raises(self) -> None:
        """Empty sources raises ValueError."""
        with pytest.raises(ValueError, match="sources cannot be empty"):
            calculate_groundedness("answer", ())

    def test_score_in_valid_range(self) -> None:
        """Score is in valid range [0, 1]."""
        score = calculate_groundedness(
            answer="Some answer.",
            sources=("Some source.",),
        )
        assert 0 <= score <= 1


class TestComputeRAGScore:
    """Tests for compute_rag_score function."""

    def test_default_weights(self) -> None:
        """Compute with default weights."""
        retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        score = compute_rag_score(retrieval, generation)
        assert 0 <= score <= 1

    def test_custom_weights(self) -> None:
        """Compute with custom weights."""
        retrieval = create_retrieval_metrics(0.8, 0.6, 0.75, 0.82, 1.0)
        generation = create_generation_metrics(0.9, 0.85, 0.88, 0.92)
        score = compute_rag_score(
            retrieval, generation, retrieval_weight=0.5, generation_weight=0.5
        )
        assert 0 <= score <= 1

    def test_retrieval_only_weight(self) -> None:
        """Compute with only retrieval weight."""
        retrieval = create_retrieval_metrics(1.0, 1.0, 1.0, 1.0, 1.0)
        generation = create_generation_metrics(0.0, 0.0, 0.0, 0.0)
        score = compute_rag_score(
            retrieval, generation, retrieval_weight=1.0, generation_weight=0.0
        )
        assert score == pytest.approx(1.0)

    def test_generation_only_weight(self) -> None:
        """Compute with only generation weight."""
        retrieval = create_retrieval_metrics(0.0, 0.0, 0.0, 0.0, 0.0)
        generation = create_generation_metrics(1.0, 1.0, 1.0, 1.0)
        score = compute_rag_score(
            retrieval, generation, retrieval_weight=0.0, generation_weight=1.0
        )
        assert score == pytest.approx(1.0)

    def test_negative_retrieval_weight_raises(self) -> None:
        """Negative retrieval_weight raises ValueError."""
        retrieval = create_retrieval_metrics()
        generation = create_generation_metrics()
        with pytest.raises(ValueError, match="retrieval_weight must be non-negative"):
            compute_rag_score(retrieval, generation, retrieval_weight=-0.1)

    def test_negative_generation_weight_raises(self) -> None:
        """Negative generation_weight raises ValueError."""
        retrieval = create_retrieval_metrics()
        generation = create_generation_metrics()
        with pytest.raises(ValueError, match="generation_weight must be non-negative"):
            compute_rag_score(retrieval, generation, generation_weight=-0.1)

    def test_zero_total_weight_raises(self) -> None:
        """Zero total weight raises ValueError."""
        retrieval = create_retrieval_metrics()
        generation = create_generation_metrics()
        expected_match = r"retrieval_weight \+ generation_weight must be positive"
        with pytest.raises(ValueError, match=expected_match):
            compute_rag_score(
                retrieval, generation, retrieval_weight=0.0, generation_weight=0.0
            )


class TestFormatRAGEvalStats:
    """Tests for format_rag_eval_stats function."""

    def test_format_basic_stats(self) -> None:
        """Format basic stats."""
        stats = create_rag_eval_stats(0.78, 0.85, 100, 0)
        formatted = format_rag_eval_stats(stats)
        assert "100 queries" in formatted
        assert "retrieval=0.78" in formatted
        assert "generation=0.85" in formatted

    def test_format_with_failed_queries(self) -> None:
        """Format stats with failed queries."""
        stats = create_rag_eval_stats(0.78, 0.85, 100, 5)
        formatted = format_rag_eval_stats(stats)
        assert "failed=5" in formatted

    def test_format_without_failed_queries(self) -> None:
        """Format stats without failed queries."""
        stats = create_rag_eval_stats(0.78, 0.85, 100, 0)
        formatted = format_rag_eval_stats(stats)
        assert "failed" not in formatted


class TestGetRecommendedRAGEvalConfig:
    """Tests for get_recommended_rag_eval_config function."""

    def test_qa_use_case(self) -> None:
        """Get config for QA use case."""
        config = get_recommended_rag_eval_config("qa")
        assert config.retrieval_k == 10
        assert config.grounding_type == GroundingType.FACTUAL
        assert config.evaluation_level == EvaluationLevel.PASSAGE

    def test_summarization_use_case(self) -> None:
        """Get config for summarization use case."""
        config = get_recommended_rag_eval_config("summarization")
        assert config.retrieval_k == 5
        assert config.grounding_type == GroundingType.ATTRIBUTION
        assert config.evaluation_level == EvaluationLevel.DOCUMENT

    def test_chat_use_case(self) -> None:
        """Get config for chat use case."""
        config = get_recommended_rag_eval_config("chat")
        assert config.retrieval_k == 3
        assert config.grounding_type == GroundingType.FACTUAL
        assert config.evaluation_level == EvaluationLevel.SENTENCE

    def test_unknown_use_case_defaults_to_qa(self) -> None:
        """Unknown use case defaults to QA settings."""
        config = get_recommended_rag_eval_config("unknown")
        assert config.retrieval_k == 10

    def test_strict_mode(self) -> None:
        """Strict mode uses citation grounding and sentence level."""
        config = get_recommended_rag_eval_config("qa", strict=True)
        assert config.grounding_type == GroundingType.CITATION
        assert config.evaluation_level == EvaluationLevel.SENTENCE

    def test_strict_mode_for_chat(self) -> None:
        """Strict mode applies to chat use case."""
        config = get_recommended_rag_eval_config("chat", strict=True)
        assert config.grounding_type == GroundingType.CITATION
        assert config.evaluation_level == EvaluationLevel.SENTENCE

    def test_default_use_case(self) -> None:
        """Default use case is QA."""
        config = get_recommended_rag_eval_config()
        assert config.retrieval_k == 10
