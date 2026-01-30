"""Tests for rag.retrieval module."""

from __future__ import annotations

import pytest

from hf_gtc.rag.retrieval import (
    VALID_CHUNKING_STRATEGIES,
    VALID_DISTANCE_METRICS,
    ChunkingConfig,
    ChunkingStrategy,
    ChunkResult,
    DistanceMetric,
    DocumentChunk,
    RAGConfig,
    RetrievalMethod,
    RetrievalResult,
    RetrievalStats,
    calculate_chunk_count,
    calculate_overlap_ratio,
    create_chunking_config,
    create_document_chunk,
    create_rag_config,
    create_retrieval_result,
    estimate_retrieval_latency,
    format_retrieval_stats,
    get_chunking_strategy,
    get_distance_metric,
    get_recommended_chunk_size,
    get_retrieval_method,
    list_chunking_strategies,
    list_distance_metrics,
    list_retrieval_methods,
    validate_chunking_config,
    validate_rag_config,
)


class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ChunkingStrategy:
            assert isinstance(strategy.value, str)

    def test_recursive_value(self) -> None:
        """Recursive has correct value."""
        assert ChunkingStrategy.RECURSIVE.value == "recursive"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_CHUNKING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_CHUNKING_STRATEGIES, frozenset)


class TestDistanceMetric:
    """Tests for DistanceMetric enum."""

    def test_all_metrics_have_values(self) -> None:
        """All metrics have string values."""
        for metric in DistanceMetric:
            assert isinstance(metric.value, str)

    def test_cosine_value(self) -> None:
        """Cosine has correct value."""
        assert DistanceMetric.COSINE.value == "cosine"

    def test_valid_metrics_frozenset(self) -> None:
        """VALID_DISTANCE_METRICS is a frozenset."""
        assert isinstance(VALID_DISTANCE_METRICS, frozenset)


class TestRetrievalMethod:
    """Tests for RetrievalMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in RetrievalMethod:
            assert isinstance(method.value, str)

    def test_dense_value(self) -> None:
        """Dense has correct value."""
        assert RetrievalMethod.DENSE.value == "dense"


class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create chunking config."""
        config = ChunkingConfig(
            chunk_size=512,
            overlap=50,
            strategy=ChunkingStrategy.RECURSIVE,
            separator="\n\n",
        )
        assert config.chunk_size == 512


class TestValidateChunkingConfig:
    """Tests for validate_chunking_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ChunkingConfig(512, 50, ChunkingStrategy.FIXED, "\n")
        validate_chunking_config(config)

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk size raises ValueError."""
        config = ChunkingConfig(0, 50, ChunkingStrategy.FIXED, "\n")
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_chunking_config(config)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        config = ChunkingConfig(512, -1, ChunkingStrategy.FIXED, "\n")
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            validate_chunking_config(config)

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        """Overlap >= chunk_size raises ValueError."""
        config = ChunkingConfig(512, 512, ChunkingStrategy.FIXED, "\n")
        with pytest.raises(ValueError, match=r"overlap.*must be less than chunk_size"):
            validate_chunking_config(config)


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_create_config(self) -> None:
        """Create RAG config."""
        config = RAGConfig(
            retrieval_method=RetrievalMethod.HYBRID,
            distance_metric=DistanceMetric.COSINE,
            top_k=10,
            score_threshold=0.5,
            rerank_top_k=5,
        )
        assert config.top_k == 10


class TestValidateRAGConfig:
    """Tests for validate_rag_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RAGConfig(RetrievalMethod.DENSE, DistanceMetric.COSINE, 10, 0.5, 5)
        validate_rag_config(config)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        config = RAGConfig(RetrievalMethod.DENSE, DistanceMetric.COSINE, 0, 0.5, 5)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_rag_config(config)

    def test_zero_rerank_top_k_raises(self) -> None:
        """Zero rerank_top_k raises ValueError."""
        config = RAGConfig(RetrievalMethod.DENSE, DistanceMetric.COSINE, 10, 0.5, 0)
        with pytest.raises(ValueError, match="rerank_top_k must be positive"):
            validate_rag_config(config)

    def test_rerank_exceeds_top_k_raises(self) -> None:
        """rerank_top_k > top_k raises ValueError."""
        config = RAGConfig(RetrievalMethod.DENSE, DistanceMetric.COSINE, 5, 0.5, 10)
        with pytest.raises(ValueError, match=r"rerank_top_k.*cannot exceed top_k"):
            validate_rag_config(config)

    def test_invalid_score_threshold_raises(self) -> None:
        """Invalid score threshold raises ValueError."""
        config = RAGConfig(RetrievalMethod.DENSE, DistanceMetric.COSINE, 10, 1.5, 5)
        with pytest.raises(ValueError, match="score_threshold must be between"):
            validate_rag_config(config)


class TestCreateChunkingConfig:
    """Tests for create_chunking_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_chunking_config()
        assert config.chunk_size == 512
        assert config.strategy == ChunkingStrategy.RECURSIVE

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_chunking_config(chunk_size=256, strategy="sentence")
        assert config.chunk_size == 256
        assert config.strategy == ChunkingStrategy.SENTENCE

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_chunking_config(strategy="invalid")

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_chunking_config(chunk_size=0)


class TestCreateDocumentChunk:
    """Tests for create_document_chunk function."""

    def test_default_chunk(self) -> None:
        """Create default chunk."""
        chunk = create_document_chunk("Hello world", "c0", "doc1")
        assert chunk.content == "Hello world"
        assert chunk.start_char == 0

    def test_custom_chunk(self) -> None:
        """Create custom chunk."""
        chunk = create_document_chunk(
            "Hello", "c1", "doc1", start_char=10, metadata={"key": "value"}
        )
        assert chunk.start_char == 10
        assert chunk.metadata["key"] == "value"

    def test_empty_content_raises(self) -> None:
        """Empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            create_document_chunk("", "c0", "doc1")

    def test_empty_chunk_id_raises(self) -> None:
        """Empty chunk_id raises ValueError."""
        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            create_document_chunk("Hello", "", "doc1")

    def test_empty_document_id_raises(self) -> None:
        """Empty document_id raises ValueError."""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            create_document_chunk("Hello", "c0", "")

    def test_negative_start_char_raises(self) -> None:
        """Negative start_char raises ValueError."""
        with pytest.raises(ValueError, match="start_char must be non-negative"):
            create_document_chunk("Hello", "c0", "doc1", start_char=-1)


class TestCreateRAGConfig:
    """Tests for create_rag_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_rag_config()
        assert config.retrieval_method == RetrievalMethod.DENSE
        assert config.top_k == 10

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_rag_config(retrieval_method="hybrid", top_k=20)
        assert config.retrieval_method == RetrievalMethod.HYBRID
        assert config.top_k == 20

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="retrieval_method must be one of"):
            create_rag_config(retrieval_method="invalid")

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="distance_metric must be one of"):
            create_rag_config(distance_metric="invalid")


class TestCreateRetrievalResult:
    """Tests for create_retrieval_result function."""

    def test_default_result(self) -> None:
        """Create default result."""
        chunk = create_document_chunk("text", "c0", "d1")
        result = create_retrieval_result(chunk, score=0.9, rank=1)
        assert result.score == 0.9
        assert result.rank == 1

    def test_invalid_score_raises(self) -> None:
        """Invalid score raises ValueError."""
        chunk = create_document_chunk("text", "c0", "d1")
        with pytest.raises(ValueError, match="score must be between"):
            create_retrieval_result(chunk, score=1.5, rank=1)

    def test_invalid_rank_raises(self) -> None:
        """Invalid rank raises ValueError."""
        chunk = create_document_chunk("text", "c0", "d1")
        with pytest.raises(ValueError, match="rank must be positive"):
            create_retrieval_result(chunk, score=0.9, rank=0)


class TestCalculateChunkCount:
    """Tests for calculate_chunk_count function."""

    def test_basic_calculation(self) -> None:
        """Basic chunk count calculation."""
        count = calculate_chunk_count(1000, 200, 50)
        # With step=150: 0-200, 150-350, 300-500, 450-650, 600-800, 750-950, 900+
        assert count == 7

    def test_exact_fit(self) -> None:
        """Document that fits in one chunk."""
        count = calculate_chunk_count(100, 100, 0)
        assert count == 1

    def test_empty_document(self) -> None:
        """Empty document returns 0."""
        count = calculate_chunk_count(0, 200, 50)
        assert count == 0

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            calculate_chunk_count(1000, 0, 50)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            calculate_chunk_count(1000, 200, -1)

    def test_overlap_exceeds_chunk_raises(self) -> None:
        """Overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match=r"overlap.*must be less than"):
            calculate_chunk_count(1000, 200, 200)


class TestCalculateOverlapRatio:
    """Tests for calculate_overlap_ratio function."""

    def test_basic_ratio(self) -> None:
        """Basic overlap ratio."""
        ratio = calculate_overlap_ratio(50, 500)
        assert ratio == 0.1

    def test_half_overlap(self) -> None:
        """Half overlap ratio."""
        ratio = calculate_overlap_ratio(100, 200)
        assert ratio == 0.5

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            calculate_overlap_ratio(50, 0)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            calculate_overlap_ratio(-1, 500)


class TestGetRecommendedChunkSize:
    """Tests for get_recommended_chunk_size function."""

    def test_qa_use_case(self) -> None:
        """QA use case returns expected size."""
        size = get_recommended_chunk_size()
        assert size == 512

    def test_summarization_use_case(self) -> None:
        """Summarization returns larger size."""
        size = get_recommended_chunk_size(use_case="summarization")
        assert size == 1024

    def test_search_use_case(self) -> None:
        """Search returns smaller size."""
        size = get_recommended_chunk_size(use_case="search")
        assert size == 256


class TestEstimateRetrievalLatency:
    """Tests for estimate_retrieval_latency function."""

    def test_basic_estimate(self) -> None:
        """Basic latency estimate."""
        latency = estimate_retrieval_latency(10000, 10)
        assert latency > 0

    def test_rerank_adds_latency(self) -> None:
        """Reranking adds latency."""
        latency_no = estimate_retrieval_latency(10000, 10, use_rerank=False)
        latency_yes = estimate_retrieval_latency(10000, 10, use_rerank=True)
        assert latency_yes > latency_no


class TestFormatRetrievalStats:
    """Tests for format_retrieval_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = RetrievalStats(10, 0.85, 50.0, True)
        formatted = format_retrieval_stats(stats)
        assert "10 documents" in formatted
        assert "reranked" in formatted


class TestListChunkingStrategies:
    """Tests for list_chunking_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_chunking_strategies()
        assert strategies == sorted(strategies)

    def test_contains_recursive(self) -> None:
        """Contains recursive."""
        strategies = list_chunking_strategies()
        assert "recursive" in strategies


class TestListDistanceMetrics:
    """Tests for list_distance_metrics function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        metrics = list_distance_metrics()
        assert metrics == sorted(metrics)

    def test_contains_cosine(self) -> None:
        """Contains cosine."""
        metrics = list_distance_metrics()
        assert "cosine" in metrics


class TestListRetrievalMethods:
    """Tests for list_retrieval_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_retrieval_methods()
        assert methods == sorted(methods)

    def test_contains_dense(self) -> None:
        """Contains dense."""
        methods = list_retrieval_methods()
        assert "dense" in methods


class TestGetChunkingStrategy:
    """Tests for get_chunking_strategy function."""

    def test_get_recursive(self) -> None:
        """Get recursive strategy."""
        assert get_chunking_strategy("recursive") == ChunkingStrategy.RECURSIVE

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_chunking_strategy("invalid")


class TestGetDistanceMetric:
    """Tests for get_distance_metric function."""

    def test_get_cosine(self) -> None:
        """Get cosine metric."""
        assert get_distance_metric("cosine") == DistanceMetric.COSINE

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            get_distance_metric("invalid")


class TestGetRetrievalMethod:
    """Tests for get_retrieval_method function."""

    def test_get_dense(self) -> None:
        """Get dense method."""
        assert get_retrieval_method("dense") == RetrievalMethod.DENSE

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_retrieval_method("invalid")


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_chunk(self) -> None:
        """Create document chunk."""
        chunk = DocumentChunk(
            content="Some text",
            chunk_id="chunk_0",
            document_id="doc_1",
            start_char=0,
            end_char=9,
            metadata={},
        )
        assert chunk.content == "Some text"


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_create_result(self) -> None:
        """Create chunk result."""
        result = ChunkResult(
            chunks=(),
            total_chunks=0,
            avg_chunk_size=0.0,
            document_id="doc_1",
        )
        assert result.total_chunks == 0


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_result(self) -> None:
        """Create retrieval result."""
        chunk = DocumentChunk("text", "c0", "d1", 0, 4, {})
        result = RetrievalResult(chunk=chunk, score=0.95, rank=1)
        assert result.score == 0.95


class TestRetrievalStats:
    """Tests for RetrievalStats dataclass."""

    def test_create_stats(self) -> None:
        """Create retrieval stats."""
        stats = RetrievalStats(
            total_retrieved=10,
            avg_score=0.85,
            latency_ms=50.0,
            reranked=True,
        )
        assert stats.total_retrieved == 10
