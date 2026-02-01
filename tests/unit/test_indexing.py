"""Tests for rag.indexing module."""

from __future__ import annotations

import pytest

from hf_gtc.rag.indexing import (
    VALID_BACKENDS,
    VALID_DISTANCE_FUNCTIONS,
    VALID_FAISS_INDEX_TYPES,
    ChromaConfig,
    DistanceFunction,
    FAISSConfig,
    FAISSIndexType,
    IndexConfig,
    IndexStats,
    SearchConfig,
    VectorStoreBackend,
    calculate_optimal_nlist,
    calculate_optimal_nprobe,
    create_chroma_config,
    create_faiss_config,
    create_index_config,
    create_index_stats,
    create_search_config,
    estimate_index_size,
    get_backend,
    get_distance_function,
    get_faiss_index_type,
    list_backends,
    list_distance_functions,
    list_faiss_index_types,
    validate_faiss_config,
    validate_search_config,
)


class TestFAISSIndexType:
    """Tests for FAISSIndexType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for index_type in FAISSIndexType:
            assert isinstance(index_type.value, str)

    def test_flat_value(self) -> None:
        """Flat has correct value."""
        assert FAISSIndexType.FLAT.value == "flat"

    def test_hnsw_value(self) -> None:
        """HNSW has correct value."""
        assert FAISSIndexType.HNSW.value == "hnsw"

    def test_valid_types_frozenset(self) -> None:
        """VALID_FAISS_INDEX_TYPES is a frozenset."""
        assert isinstance(VALID_FAISS_INDEX_TYPES, frozenset)


class TestVectorStoreBackend:
    """Tests for VectorStoreBackend enum."""

    def test_all_backends_have_values(self) -> None:
        """All backends have string values."""
        for backend in VectorStoreBackend:
            assert isinstance(backend.value, str)

    def test_faiss_value(self) -> None:
        """FAISS has correct value."""
        assert VectorStoreBackend.FAISS.value == "faiss"

    def test_chroma_value(self) -> None:
        """Chroma has correct value."""
        assert VectorStoreBackend.CHROMA.value == "chroma"


class TestDistanceFunction:
    """Tests for DistanceFunction enum."""

    def test_cosine_value(self) -> None:
        """Cosine has correct value."""
        assert DistanceFunction.COSINE.value == "cosine"

    def test_l2_value(self) -> None:
        """L2 has correct value."""
        assert DistanceFunction.L2.value == "l2"


class TestFAISSConfig:
    """Tests for FAISSConfig dataclass."""

    def test_create_config(self) -> None:
        """Create FAISS config."""
        config = FAISSConfig(
            index_type=FAISSIndexType.FLAT,
            dimension=768,
            nlist=100,
            nprobe=10,
            m=8,
            nbits=8,
            ef_construction=200,
            ef_search=128,
        )
        assert config.dimension == 768

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FAISSConfig(FAISSIndexType.FLAT, 768, 100, 10, 8, 8, 200, 128)
        with pytest.raises(AttributeError):
            config.dimension = 384  # type: ignore[misc]


class TestValidateFAISSConfig:
    """Tests for validate_faiss_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FAISSConfig(FAISSIndexType.FLAT, 768, 100, 10, 8, 8, 200, 128)
        validate_faiss_config(config)

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        config = FAISSConfig(FAISSIndexType.FLAT, 0, 100, 10, 8, 8, 200, 128)
        with pytest.raises(ValueError, match="dimension must be positive"):
            validate_faiss_config(config)

    def test_nprobe_exceeds_nlist_raises(self) -> None:
        """Nprobe > nlist raises ValueError."""
        config = FAISSConfig(FAISSIndexType.FLAT, 768, 10, 100, 8, 8, 200, 128)
        with pytest.raises(ValueError, match=r"nprobe.*cannot exceed nlist"):
            validate_faiss_config(config)


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_create_config(self) -> None:
        """Create search config."""
        config = SearchConfig(
            top_k=10,
            score_threshold=0.7,
            include_metadata=True,
            include_embeddings=False,
        )
        assert config.top_k == 10


class TestValidateSearchConfig:
    """Tests for validate_search_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SearchConfig(10, 0.7, True, False)
        validate_search_config(config)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        config = SearchConfig(0, 0.7, True, False)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_search_config(config)

    def test_invalid_threshold_raises(self) -> None:
        """Invalid score_threshold raises ValueError."""
        config = SearchConfig(10, 1.5, True, False)
        with pytest.raises(ValueError, match="score_threshold must be between"):
            validate_search_config(config)


class TestCreateFAISSConfig:
    """Tests for create_faiss_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_faiss_config()
        assert config.index_type == FAISSIndexType.FLAT
        assert config.dimension == 768

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_faiss_config(index_type="ivf_flat", nlist=200)
        assert config.index_type == FAISSIndexType.IVF_FLAT
        assert config.nlist == 200

    def test_invalid_type_raises(self) -> None:
        """Invalid index_type raises ValueError."""
        with pytest.raises(ValueError, match="index_type must be one of"):
            create_faiss_config(index_type="invalid")


class TestCreateChromaConfig:
    """Tests for create_chroma_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_chroma_config("my_collection")
        assert config.collection_name == "my_collection"
        assert config.distance_function == DistanceFunction.COSINE

    def test_empty_name_raises(self) -> None:
        """Empty collection_name raises ValueError."""
        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            create_chroma_config("")


class TestCreateIndexConfig:
    """Tests for create_index_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_index_config()
        assert config.backend == VectorStoreBackend.FAISS
        assert config.dimension == 768

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_index_config(backend="chroma", dimension=384)
        assert config.backend == VectorStoreBackend.CHROMA
        assert config.dimension == 384

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            create_index_config(backend="invalid")

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            create_index_config(dimension=0)


class TestCreateSearchConfig:
    """Tests for create_search_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_search_config()
        assert config.top_k == 10
        assert config.include_metadata is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_search_config(top_k=5, score_threshold=0.8)
        assert config.top_k == 5
        assert config.score_threshold == pytest.approx(0.8)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            create_search_config(top_k=0)


class TestCreateIndexStats:
    """Tests for create_index_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_index_stats()
        assert stats.total_vectors == 0
        assert stats.is_trained is False

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_index_stats(total_vectors=1000, is_trained=True)
        assert stats.total_vectors == 1000
        assert stats.is_trained is True

    def test_negative_vectors_raises(self) -> None:
        """Negative vectors raises ValueError."""
        with pytest.raises(ValueError, match="total_vectors must be non-negative"):
            create_index_stats(total_vectors=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_faiss_index_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_faiss_index_types()
        assert types == sorted(types)

    def test_list_backends_sorted(self) -> None:
        """Returns sorted list."""
        backends = list_backends()
        assert backends == sorted(backends)

    def test_list_distance_functions_sorted(self) -> None:
        """Returns sorted list."""
        funcs = list_distance_functions()
        assert funcs == sorted(funcs)


class TestGetFunctions:
    """Tests for get_* functions."""

    def test_get_faiss_index_type(self) -> None:
        """Get FAISS index type."""
        assert get_faiss_index_type("flat") == FAISSIndexType.FLAT

    def test_get_faiss_index_type_invalid(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="index_type must be one of"):
            get_faiss_index_type("invalid")

    def test_get_backend(self) -> None:
        """Get backend."""
        assert get_backend("faiss") == VectorStoreBackend.FAISS

    def test_get_backend_invalid(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            get_backend("invalid")

    def test_get_distance_function(self) -> None:
        """Get distance function."""
        assert get_distance_function("cosine") == DistanceFunction.COSINE

    def test_get_distance_function_invalid(self) -> None:
        """Invalid function raises ValueError."""
        with pytest.raises(ValueError, match="distance_function must be one of"):
            get_distance_function("invalid")


class TestEstimateIndexSize:
    """Tests for estimate_index_size function."""

    def test_basic_estimate(self) -> None:
        """Basic size estimate."""
        size = estimate_index_size(10000, 768, FAISSIndexType.FLAT)
        assert size > 0

    def test_pq_smaller_than_flat(self) -> None:
        """PQ index is smaller than flat."""
        flat_size = estimate_index_size(10000, 768, FAISSIndexType.FLAT)
        pq_size = estimate_index_size(10000, 768, FAISSIndexType.IVF_PQ)
        assert pq_size < flat_size

    def test_zero_vectors_raises(self) -> None:
        """Zero vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            estimate_index_size(0, 768, FAISSIndexType.FLAT)


class TestCalculateOptimalNlist:
    """Tests for calculate_optimal_nlist function."""

    def test_basic_calculation(self) -> None:
        """Basic nlist calculation."""
        nlist = calculate_optimal_nlist(100000)
        assert 100 <= nlist <= 1000

    def test_zero_vectors_raises(self) -> None:
        """Zero vectors raises ValueError."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            calculate_optimal_nlist(0)


class TestCalculateOptimalNprobe:
    """Tests for calculate_optimal_nprobe function."""

    def test_basic_calculation(self) -> None:
        """Basic nprobe calculation."""
        nprobe = calculate_optimal_nprobe(100)
        assert 1 <= nprobe <= 100

    def test_zero_nlist_raises(self) -> None:
        """Zero nlist raises ValueError."""
        with pytest.raises(ValueError, match="nlist must be positive"):
            calculate_optimal_nprobe(0)

    def test_invalid_recall_raises(self) -> None:
        """Invalid recall_target raises ValueError."""
        with pytest.raises(ValueError, match="recall_target must be in"):
            calculate_optimal_nprobe(100, recall_target=0.0)


class TestChromaConfig:
    """Tests for ChromaConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Chroma config."""
        config = ChromaConfig(
            collection_name="docs",
            distance_function=DistanceFunction.COSINE,
            persist_directory="./db",
            anonymized_telemetry=False,
        )
        assert config.collection_name == "docs"


class TestIndexConfig:
    """Tests for IndexConfig dataclass."""

    def test_create_config(self) -> None:
        """Create index config."""
        config = IndexConfig(
            backend=VectorStoreBackend.FAISS,
            dimension=768,
            distance_function=DistanceFunction.COSINE,
            batch_size=1000,
            normalize_embeddings=True,
        )
        assert config.normalize_embeddings is True


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_create_stats(self) -> None:
        """Create index stats."""
        stats = IndexStats(
            total_vectors=10000,
            dimension=768,
            index_size_bytes=15360000,
            is_trained=True,
        )
        assert stats.total_vectors == 10000


class TestValidConstants:
    """Tests for validation constants."""

    def test_valid_backends_frozenset(self) -> None:
        """VALID_BACKENDS is a frozenset."""
        assert isinstance(VALID_BACKENDS, frozenset)

    def test_valid_distance_functions_frozenset(self) -> None:
        """VALID_DISTANCE_FUNCTIONS is a frozenset."""
        assert isinstance(VALID_DISTANCE_FUNCTIONS, frozenset)
