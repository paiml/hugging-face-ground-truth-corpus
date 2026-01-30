"""Tests for rag.vectorstore module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.rag.vectorstore import (
    VALID_DISTANCE_METRICS,
    VALID_INDEX_TYPES,
    VALID_STORE_TYPES,
    DistanceMetric,
    IndexConfig,
    IndexType,
    SearchResult,
    VectorStoreConfig,
    VectorStoreType,
    calculate_index_size,
    calculate_recall_at_k,
    create_index_config,
    create_search_result,
    create_vectorstore_config,
    estimate_search_latency,
    format_search_stats,
    get_distance_metric,
    get_index_type,
    get_recommended_vectorstore_config,
    get_store_type,
    list_distance_metrics,
    list_index_types,
    list_store_types,
    optimize_index_params,
    validate_index_config,
    validate_search_result,
    validate_vectorstore_config,
)


class TestVectorStoreType:
    """Tests for VectorStoreType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for store_type in VectorStoreType:
            assert isinstance(store_type.value, str)

    def test_faiss_value(self) -> None:
        """FAISS has correct value."""
        assert VectorStoreType.FAISS.value == "faiss"

    def test_chromadb_value(self) -> None:
        """ChromaDB has correct value."""
        assert VectorStoreType.CHROMADB.value == "chromadb"

    def test_pinecone_value(self) -> None:
        """Pinecone has correct value."""
        assert VectorStoreType.PINECONE.value == "pinecone"

    def test_weaviate_value(self) -> None:
        """Weaviate has correct value."""
        assert VectorStoreType.WEAVIATE.value == "weaviate"

    def test_qdrant_value(self) -> None:
        """Qdrant has correct value."""
        assert VectorStoreType.QDRANT.value == "qdrant"

    def test_milvus_value(self) -> None:
        """Milvus has correct value."""
        assert VectorStoreType.MILVUS.value == "milvus"

    def test_valid_store_types_frozenset(self) -> None:
        """VALID_STORE_TYPES is a frozenset."""
        assert isinstance(VALID_STORE_TYPES, frozenset)
        assert len(VALID_STORE_TYPES) == 6


class TestIndexType:
    """Tests for IndexType enum."""

    def test_all_types_have_values(self) -> None:
        """All index types have string values."""
        for index_type in IndexType:
            assert isinstance(index_type.value, str)

    def test_flat_value(self) -> None:
        """FLAT has correct value."""
        assert IndexType.FLAT.value == "flat"

    def test_ivf_value(self) -> None:
        """IVF has correct value."""
        assert IndexType.IVF.value == "ivf"

    def test_hnsw_value(self) -> None:
        """HNSW has correct value."""
        assert IndexType.HNSW.value == "hnsw"

    def test_pq_value(self) -> None:
        """PQ has correct value."""
        assert IndexType.PQ.value == "pq"

    def test_scann_value(self) -> None:
        """SCANN has correct value."""
        assert IndexType.SCANN.value == "scann"

    def test_valid_index_types_frozenset(self) -> None:
        """VALID_INDEX_TYPES is a frozenset."""
        assert isinstance(VALID_INDEX_TYPES, frozenset)
        assert len(VALID_INDEX_TYPES) == 5


class TestDistanceMetric:
    """Tests for DistanceMetric enum."""

    def test_all_metrics_have_values(self) -> None:
        """All metrics have string values."""
        for metric in DistanceMetric:
            assert isinstance(metric.value, str)

    def test_cosine_value(self) -> None:
        """COSINE has correct value."""
        assert DistanceMetric.COSINE.value == "cosine"

    def test_euclidean_value(self) -> None:
        """EUCLIDEAN has correct value."""
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"

    def test_dot_product_value(self) -> None:
        """DOT_PRODUCT has correct value."""
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"

    def test_manhattan_value(self) -> None:
        """MANHATTAN has correct value."""
        assert DistanceMetric.MANHATTAN.value == "manhattan"

    def test_valid_distance_metrics_frozenset(self) -> None:
        """VALID_DISTANCE_METRICS is a frozenset."""
        assert isinstance(VALID_DISTANCE_METRICS, frozenset)
        assert len(VALID_DISTANCE_METRICS) == 4


class TestIndexConfig:
    """Tests for IndexConfig dataclass."""

    def test_create_config(self) -> None:
        """Create index config."""
        config = IndexConfig(
            index_type=IndexType.HNSW,
            nlist=100,
            nprobe=10,
            ef_search=128,
        )
        assert config.index_type == IndexType.HNSW
        assert config.nlist == 100

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = IndexConfig(IndexType.FLAT, 100, 10, 128)
        with pytest.raises(AttributeError):
            config.nlist = 200  # type: ignore[misc]

    def test_config_has_slots(self) -> None:
        """Config uses slots."""
        config = IndexConfig(IndexType.FLAT, 100, 10, 128)
        assert not hasattr(config, "__dict__")


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig dataclass."""

    def test_create_config(self) -> None:
        """Create vector store config."""
        idx_config = IndexConfig(IndexType.FLAT, 100, 10, 128)
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            index_config=idx_config,
            distance_metric=DistanceMetric.COSINE,
            dimension=768,
        )
        assert config.dimension == 768
        assert config.store_type == VectorStoreType.FAISS

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        idx_config = IndexConfig(IndexType.FLAT, 100, 10, 128)
        config = VectorStoreConfig(
            VectorStoreType.FAISS, idx_config, DistanceMetric.COSINE, 768
        )
        with pytest.raises(AttributeError):
            config.dimension = 384  # type: ignore[misc]


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_result(self) -> None:
        """Create search result."""
        result = SearchResult(
            ids=("id1", "id2"),
            scores=(0.95, 0.87),
            vectors=(),
            metadata=({"source": "doc1"}, {"source": "doc2"}),
        )
        assert result.ids == ("id1", "id2")
        assert result.scores == (0.95, 0.87)

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = SearchResult(("id1",), (0.9,), (), ({},))
        with pytest.raises(AttributeError):
            result.ids = ("new_id",)  # type: ignore[misc]


class TestValidateIndexConfig:
    """Tests for validate_index_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = IndexConfig(IndexType.IVF, 100, 10, 128)
        validate_index_config(config)

    def test_zero_nlist_raises(self) -> None:
        """Zero nlist raises ValueError."""
        config = IndexConfig(IndexType.IVF, 0, 10, 128)
        with pytest.raises(ValueError, match="nlist must be positive"):
            validate_index_config(config)

    def test_negative_nlist_raises(self) -> None:
        """Negative nlist raises ValueError."""
        config = IndexConfig(IndexType.IVF, -1, 10, 128)
        with pytest.raises(ValueError, match="nlist must be positive"):
            validate_index_config(config)

    def test_zero_nprobe_raises(self) -> None:
        """Zero nprobe raises ValueError."""
        config = IndexConfig(IndexType.IVF, 100, 0, 128)
        with pytest.raises(ValueError, match="nprobe must be positive"):
            validate_index_config(config)

    def test_nprobe_exceeds_nlist_raises(self) -> None:
        """Nprobe > nlist raises ValueError."""
        config = IndexConfig(IndexType.IVF, 10, 100, 128)
        with pytest.raises(ValueError, match=r"nprobe.*cannot exceed nlist"):
            validate_index_config(config)

    def test_zero_ef_search_raises(self) -> None:
        """Zero ef_search raises ValueError."""
        config = IndexConfig(IndexType.HNSW, 100, 10, 0)
        with pytest.raises(ValueError, match="ef_search must be positive"):
            validate_index_config(config)


class TestValidateVectorStoreConfig:
    """Tests for validate_vectorstore_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        idx = IndexConfig(IndexType.FLAT, 100, 10, 128)
        config = VectorStoreConfig(
            VectorStoreType.FAISS, idx, DistanceMetric.COSINE, 768
        )
        validate_vectorstore_config(config)

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        idx = IndexConfig(IndexType.FLAT, 100, 10, 128)
        config = VectorStoreConfig(
            VectorStoreType.FAISS, idx, DistanceMetric.COSINE, 0
        )
        with pytest.raises(ValueError, match="dimension must be positive"):
            validate_vectorstore_config(config)

    def test_negative_dimension_raises(self) -> None:
        """Negative dimension raises ValueError."""
        idx = IndexConfig(IndexType.FLAT, 100, 10, 128)
        config = VectorStoreConfig(
            VectorStoreType.FAISS, idx, DistanceMetric.COSINE, -768
        )
        with pytest.raises(ValueError, match="dimension must be positive"):
            validate_vectorstore_config(config)


class TestValidateSearchResult:
    """Tests for validate_search_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes validation."""
        result = SearchResult(("id1",), (0.9,), (), ({},))
        validate_search_result(result)

    def test_mismatched_ids_scores_raises(self) -> None:
        """Mismatched ids and scores raises ValueError."""
        result = SearchResult(("id1",), (0.9, 0.8), (), ({},))
        with pytest.raises(ValueError, match="ids and scores must have the same"):
            validate_search_result(result)

    def test_mismatched_ids_metadata_raises(self) -> None:
        """Mismatched ids and metadata raises ValueError."""
        result = SearchResult(("id1", "id2"), (0.9, 0.8), (), ({},))
        with pytest.raises(ValueError, match="ids and metadata must have the same"):
            validate_search_result(result)

    def test_mismatched_vectors_raises(self) -> None:
        """Mismatched vectors raises ValueError."""
        result = SearchResult(
            ("id1", "id2"),
            (0.9, 0.8),
            ((0.1, 0.2),),  # Only one vector for two ids
            ({}, {}),
        )
        with pytest.raises(ValueError, match="vectors must be empty or have same"):
            validate_search_result(result)

    def test_score_too_high_raises(self) -> None:
        """Score > 1.0 raises ValueError."""
        result = SearchResult(("id1",), (1.5,), (), ({},))
        with pytest.raises(ValueError, match=r"scores must be between 0\.0 and 1\.0"):
            validate_search_result(result)

    def test_score_too_low_raises(self) -> None:
        """Score < 0.0 raises ValueError."""
        result = SearchResult(("id1",), (-0.1,), (), ({},))
        with pytest.raises(ValueError, match=r"scores must be between 0\.0 and 1\.0"):
            validate_search_result(result)


class TestCreateIndexConfig:
    """Tests for create_index_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_index_config()
        assert config.index_type == IndexType.FLAT
        assert config.nlist == 100
        assert config.nprobe == 10
        assert config.ef_search == 128

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_index_config(index_type="hnsw", ef_search=256)
        assert config.index_type == IndexType.HNSW
        assert config.ef_search == 256

    def test_invalid_index_type_raises(self) -> None:
        """Invalid index_type raises ValueError."""
        with pytest.raises(ValueError, match="index_type must be one of"):
            create_index_config(index_type="invalid")

    def test_zero_nlist_raises(self) -> None:
        """Zero nlist raises ValueError."""
        with pytest.raises(ValueError, match="nlist must be positive"):
            create_index_config(nlist=0)

    @pytest.mark.parametrize("index_type", list(VALID_INDEX_TYPES))
    def test_all_valid_index_types(self, index_type: str) -> None:
        """All valid index types can be created."""
        config = create_index_config(index_type=index_type)
        assert config.index_type.value == index_type


class TestCreateVectorStoreConfig:
    """Tests for create_vectorstore_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_vectorstore_config()
        assert config.store_type == VectorStoreType.FAISS
        assert config.dimension == 768
        assert config.distance_metric == DistanceMetric.COSINE

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_vectorstore_config(
            store_type="chromadb", dimension=384, distance_metric="euclidean"
        )
        assert config.store_type == VectorStoreType.CHROMADB
        assert config.dimension == 384
        assert config.distance_metric == DistanceMetric.EUCLIDEAN

    def test_invalid_store_type_raises(self) -> None:
        """Invalid store_type raises ValueError."""
        with pytest.raises(ValueError, match="store_type must be one of"):
            create_vectorstore_config(store_type="invalid")

    def test_invalid_distance_metric_raises(self) -> None:
        """Invalid distance_metric raises ValueError."""
        with pytest.raises(ValueError, match="distance_metric must be one of"):
            create_vectorstore_config(distance_metric="invalid")

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            create_vectorstore_config(dimension=0)

    @pytest.mark.parametrize("store_type", list(VALID_STORE_TYPES))
    def test_all_valid_store_types(self, store_type: str) -> None:
        """All valid store types can be created."""
        config = create_vectorstore_config(store_type=store_type)
        assert config.store_type.value == store_type

    @pytest.mark.parametrize("metric", list(VALID_DISTANCE_METRICS))
    def test_all_valid_distance_metrics(self, metric: str) -> None:
        """All valid distance metrics can be created."""
        config = create_vectorstore_config(distance_metric=metric)
        assert config.distance_metric.value == metric


class TestCreateSearchResult:
    """Tests for create_search_result function."""

    def test_basic_result(self) -> None:
        """Create basic search result."""
        result = create_search_result(
            ids=("id1", "id2"),
            scores=(0.95, 0.87),
        )
        assert result.ids == ("id1", "id2")
        assert result.scores == (0.95, 0.87)
        assert result.vectors == ()
        assert len(result.metadata) == 2

    def test_result_with_vectors(self) -> None:
        """Create result with vectors."""
        result = create_search_result(
            ids=("id1",),
            scores=(0.9,),
            vectors=((0.1, 0.2, 0.3),),
        )
        assert result.vectors == ((0.1, 0.2, 0.3),)

    def test_result_with_metadata(self) -> None:
        """Create result with metadata."""
        result = create_search_result(
            ids=("id1",),
            scores=(0.9,),
            metadata=({"key": "value"},),
        )
        assert result.metadata == ({"key": "value"},)

    def test_invalid_score_raises(self) -> None:
        """Invalid score raises ValueError."""
        with pytest.raises(ValueError, match="scores must be between"):
            create_search_result(ids=("id1",), scores=(1.5,))


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_store_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_store_types()
        assert types == sorted(types)
        assert "faiss" in types
        assert "chromadb" in types

    def test_list_index_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_index_types()
        assert types == sorted(types)
        assert "flat" in types
        assert "hnsw" in types

    def test_list_distance_metrics_sorted(self) -> None:
        """Returns sorted list."""
        metrics = list_distance_metrics()
        assert metrics == sorted(metrics)
        assert "cosine" in metrics
        assert "euclidean" in metrics


class TestGetFunctions:
    """Tests for get_* functions."""

    def test_get_store_type(self) -> None:
        """Get store type."""
        assert get_store_type("faiss") == VectorStoreType.FAISS

    def test_get_store_type_invalid(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="store_type must be one of"):
            get_store_type("invalid")

    def test_get_index_type(self) -> None:
        """Get index type."""
        assert get_index_type("hnsw") == IndexType.HNSW

    def test_get_index_type_invalid(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="index_type must be one of"):
            get_index_type("invalid")

    def test_get_distance_metric(self) -> None:
        """Get distance metric."""
        assert get_distance_metric("cosine") == DistanceMetric.COSINE

    def test_get_distance_metric_invalid(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="distance_metric must be one of"):
            get_distance_metric("invalid")

    @pytest.mark.parametrize("store_type", list(VALID_STORE_TYPES))
    def test_get_all_store_types(self, store_type: str) -> None:
        """Get all valid store types."""
        result = get_store_type(store_type)
        assert result.value == store_type

    @pytest.mark.parametrize("index_type", list(VALID_INDEX_TYPES))
    def test_get_all_index_types(self, index_type: str) -> None:
        """Get all valid index types."""
        result = get_index_type(index_type)
        assert result.value == index_type

    @pytest.mark.parametrize("metric", list(VALID_DISTANCE_METRICS))
    def test_get_all_distance_metrics(self, metric: str) -> None:
        """Get all valid distance metrics."""
        result = get_distance_metric(metric)
        assert result.value == metric


class TestCalculateIndexSize:
    """Tests for calculate_index_size function."""

    def test_basic_calculation(self) -> None:
        """Basic size calculation."""
        size = calculate_index_size(10000, 768, IndexType.FLAT)
        assert size > 0
        # 10000 * 768 * 4 = 30,720,000
        assert size == 30720000

    def test_pq_smaller_than_flat(self) -> None:
        """PQ index is smaller than flat."""
        flat_size = calculate_index_size(10000, 768, IndexType.FLAT)
        pq_size = calculate_index_size(10000, 768, IndexType.PQ)
        assert pq_size < flat_size

    def test_hnsw_larger_than_flat(self) -> None:
        """HNSW index is larger than flat."""
        flat_size = calculate_index_size(10000, 768, IndexType.FLAT)
        hnsw_size = calculate_index_size(10000, 768, IndexType.HNSW)
        assert hnsw_size > flat_size

    def test_zero_vectors_raises(self) -> None:
        """Zero vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            calculate_index_size(0, 768, IndexType.FLAT)

    def test_negative_vectors_raises(self) -> None:
        """Negative vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            calculate_index_size(-1, 768, IndexType.FLAT)

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            calculate_index_size(10000, 0, IndexType.FLAT)

    def test_zero_bytes_per_float_raises(self) -> None:
        """Zero bytes_per_float raises ValueError."""
        with pytest.raises(ValueError, match="bytes_per_float must be positive"):
            calculate_index_size(10000, 768, IndexType.FLAT, bytes_per_float=0)

    @pytest.mark.parametrize("index_type", list(IndexType))
    def test_all_index_types_calculate(self, index_type: IndexType) -> None:
        """All index types can calculate size."""
        size = calculate_index_size(10000, 768, index_type)
        assert size > 0


class TestEstimateSearchLatency:
    """Tests for estimate_search_latency function."""

    def test_basic_estimation(self) -> None:
        """Basic latency estimation."""
        latency = estimate_search_latency(100000, IndexType.FLAT)
        assert latency > 0

    def test_hnsw_faster_than_flat(self) -> None:
        """HNSW is faster than flat for large datasets."""
        flat_lat = estimate_search_latency(100000, IndexType.FLAT)
        hnsw_lat = estimate_search_latency(100000, IndexType.HNSW)
        assert hnsw_lat < flat_lat

    def test_larger_dataset_slower(self) -> None:
        """Larger dataset has higher latency."""
        small_lat = estimate_search_latency(1000, IndexType.FLAT)
        large_lat = estimate_search_latency(1000000, IndexType.FLAT)
        assert large_lat > small_lat

    def test_zero_vectors_raises(self) -> None:
        """Zero vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            estimate_search_latency(0, IndexType.FLAT)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            estimate_search_latency(10000, IndexType.FLAT, top_k=0)

    @pytest.mark.parametrize("index_type", list(IndexType))
    def test_all_index_types_estimate(self, index_type: IndexType) -> None:
        """All index types can estimate latency."""
        latency = estimate_search_latency(10000, index_type)
        assert latency >= 0.1


class TestCalculateRecallAtK:
    """Tests for calculate_recall_at_k function."""

    def test_basic_recall(self) -> None:
        """Basic recall calculation."""
        recall = calculate_recall_at_k(8, 10, 10)
        assert recall == 0.8

    def test_perfect_recall(self) -> None:
        """Perfect recall is 1.0."""
        recall = calculate_recall_at_k(5, 5, 10)
        assert recall == 1.0

    def test_zero_recall(self) -> None:
        """Zero recall when no relevant items found."""
        recall = calculate_recall_at_k(0, 10, 5)
        assert recall == 0.0

    def test_negative_relevant_raises(self) -> None:
        """Negative relevant_retrieved raises ValueError."""
        with pytest.raises(ValueError, match="relevant_retrieved must be non-negative"):
            calculate_recall_at_k(-1, 10, 5)

    def test_zero_total_relevant_raises(self) -> None:
        """Zero total_relevant raises ValueError."""
        with pytest.raises(ValueError, match="total_relevant must be positive"):
            calculate_recall_at_k(5, 0, 10)

    def test_zero_k_raises(self) -> None:
        """Zero k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            calculate_recall_at_k(5, 10, 0)

    def test_relevant_exceeds_total_raises(self) -> None:
        """Relevant exceeding total raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed total_relevant"):
            calculate_recall_at_k(15, 10, 20)

    def test_relevant_exceeds_k_raises(self) -> None:
        """Relevant exceeding k raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed k"):
            calculate_recall_at_k(15, 20, 10)


class TestOptimizeIndexParams:
    """Tests for optimize_index_params function."""

    def test_small_dataset_uses_flat(self) -> None:
        """Small dataset recommends flat index."""
        params = optimize_index_params(1000, 768)
        assert params["index_type"] == "flat"

    def test_large_dataset_uses_approx(self) -> None:
        """Large dataset recommends approximate index."""
        params = optimize_index_params(1000000, 768)
        assert params["index_type"] in ("hnsw", "ivf")

    def test_returns_expected_keys(self) -> None:
        """Returns expected parameter keys."""
        params = optimize_index_params(100000, 768)
        assert "index_type" in params
        assert "nlist" in params
        assert "nprobe" in params
        assert "ef_search" in params
        assert "estimated_recall" in params
        assert "estimated_latency_ms" in params

    def test_zero_vectors_raises(self) -> None:
        """Zero vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            optimize_index_params(0, 768)

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            optimize_index_params(10000, 0)

    def test_invalid_recall_target_raises(self) -> None:
        """Invalid recall_target raises ValueError."""
        with pytest.raises(ValueError, match="recall_target must be in"):
            optimize_index_params(10000, 768, recall_target=0.0)

    def test_invalid_latency_budget_raises(self) -> None:
        """Invalid latency_budget raises ValueError."""
        with pytest.raises(ValueError, match="latency_budget_ms must be positive"):
            optimize_index_params(10000, 768, latency_budget_ms=0)

    def test_high_recall_target(self) -> None:
        """High recall target affects params."""
        low_recall = optimize_index_params(100000, 768, recall_target=0.8)
        high_recall = optimize_index_params(100000, 768, recall_target=0.99)
        # Higher recall typically means more nprobe
        assert high_recall["nprobe"] >= low_recall["nprobe"]

    def test_tight_latency_budget(self) -> None:
        """Tight latency budget affects index choice."""
        params = optimize_index_params(1000000, 768, latency_budget_ms=5.0)
        assert params["index_type"] == "hnsw"

    def test_very_large_dataset_tight_latency(self) -> None:
        """Very large dataset with tight latency uses hnsw."""
        params = optimize_index_params(2000000, 768, latency_budget_ms=8.0)
        assert params["index_type"] == "hnsw"

    def test_very_large_dataset_moderate_latency(self) -> None:
        """Very large dataset with moderate latency uses ivf."""
        params = optimize_index_params(2000000, 768, latency_budget_ms=30.0)
        assert params["index_type"] == "ivf"

    def test_very_large_dataset_relaxed_latency(self) -> None:
        """Very large dataset with relaxed latency uses hnsw."""
        params = optimize_index_params(2000000, 768, latency_budget_ms=100.0)
        assert params["index_type"] == "hnsw"


class TestFormatSearchStats:
    """Tests for format_search_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        result = create_search_result(("id1", "id2"), (0.95, 0.87))
        stats = format_search_stats(result, 5.2, IndexType.HNSW)
        assert "2 results" in stats
        assert "5.2ms" in stats
        assert "hnsw" in stats.lower()

    def test_includes_avg_score(self) -> None:
        """Includes average score."""
        result = create_search_result(("id1",), (0.9,))
        stats = format_search_stats(result, 1.0, IndexType.FLAT)
        assert "0.900" in stats

    def test_includes_index_type(self) -> None:
        """Includes index type."""
        result = create_search_result(("id1",), (0.9,))
        stats = format_search_stats(result, 1.0, IndexType.IVF)
        assert "ivf" in stats.lower()


class TestGetRecommendedVectorStoreConfig:
    """Tests for get_recommended_vectorstore_config function."""

    def test_default_config(self) -> None:
        """Default configuration."""
        config = get_recommended_vectorstore_config()
        assert config.store_type == VectorStoreType.FAISS
        assert config.dimension == 768

    def test_similarity_search_use_case(self) -> None:
        """Similarity search use case."""
        config = get_recommended_vectorstore_config(use_case="similarity_search")
        assert config.store_type == VectorStoreType.FAISS

    def test_production_use_case(self) -> None:
        """Production use case."""
        config = get_recommended_vectorstore_config(use_case="production")
        assert config.store_type == VectorStoreType.QDRANT

    def test_development_use_case(self) -> None:
        """Development use case."""
        config = get_recommended_vectorstore_config(use_case="development")
        assert config.store_type == VectorStoreType.CHROMADB

    def test_small_dataset_uses_flat(self) -> None:
        """Small dataset recommends flat index."""
        config = get_recommended_vectorstore_config(num_vectors=1000)
        assert config.index_config.index_type == IndexType.FLAT

    def test_large_dataset_uses_approx(self) -> None:
        """Large dataset recommends approximate index."""
        config = get_recommended_vectorstore_config(
            use_case="production", num_vectors=10000000
        )
        assert config.index_config.index_type in (IndexType.HNSW, IndexType.IVF)

    def test_zero_vectors_raises(self) -> None:
        """Zero vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            get_recommended_vectorstore_config(num_vectors=0)

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            get_recommended_vectorstore_config(dimension=0)

    def test_unknown_use_case_uses_default(self) -> None:
        """Unknown use case uses default settings."""
        config = get_recommended_vectorstore_config(use_case="unknown")
        assert config.store_type == VectorStoreType.FAISS


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        num_vectors=st.integers(min_value=1, max_value=1000000),
        dimension=st.integers(min_value=1, max_value=4096),
    )
    @settings(max_examples=50)
    def test_index_size_always_positive(
        self, num_vectors: int, dimension: int
    ) -> None:
        """Index size is always positive for valid inputs."""
        size = calculate_index_size(num_vectors, dimension, IndexType.FLAT)
        assert size > 0

    @given(
        num_vectors=st.integers(min_value=1, max_value=1000000),
    )
    @settings(max_examples=50)
    def test_search_latency_always_positive(self, num_vectors: int) -> None:
        """Search latency is always positive for valid inputs."""
        latency = estimate_search_latency(num_vectors, IndexType.FLAT)
        assert latency >= 0.1

    @given(
        relevant=st.integers(min_value=0, max_value=100),
        total=st.integers(min_value=1, max_value=100),
        k=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_recall_in_valid_range(
        self, relevant: int, total: int, k: int
    ) -> None:
        """Recall is always between 0 and 1."""
        if relevant <= total and relevant <= k:
            recall = calculate_recall_at_k(relevant, total, k)
            assert 0.0 <= recall <= 1.0

    @given(
        dimension=st.integers(min_value=1, max_value=4096),
    )
    @settings(max_examples=20)
    def test_vectorstore_config_always_valid(self, dimension: int) -> None:
        """Created config is always valid."""
        config = create_vectorstore_config(dimension=dimension)
        validate_vectorstore_config(config)

    @given(
        nlist=st.integers(min_value=1, max_value=1000),
        nprobe=st.integers(min_value=1, max_value=1000),
        ef_search=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_index_config_valid_when_constraints_met(
        self, nlist: int, nprobe: int, ef_search: int
    ) -> None:
        """Index config is valid when nprobe <= nlist."""
        if nprobe <= nlist:
            config = IndexConfig(IndexType.IVF, nlist, nprobe, ef_search)
            validate_index_config(config)
