"""Tests for sentence embeddings functionality."""

from __future__ import annotations

import pytest

from hf_gtc.inference.embeddings import (
    VALID_DISTANCE_METRICS,
    VALID_POOLING_MODES,
    EmbeddingConfig,
    PoolingMode,
    SimilarityResult,
    chunk_text,
    cosine_similarity,
    create_embedding_config,
    dot_product_similarity,
    estimate_embedding_memory,
    euclidean_distance,
    get_recommended_model,
    list_distance_metrics,
    list_pooling_modes,
    normalize_vector,
    validate_embedding_config,
)


class TestPoolingMode:
    """Tests for PoolingMode enum."""

    def test_mean_value(self) -> None:
        """Test MEAN pooling value."""
        assert PoolingMode.MEAN.value == "mean"

    def test_cls_value(self) -> None:
        """Test CLS pooling value."""
        assert PoolingMode.CLS.value == "cls"

    def test_max_value(self) -> None:
        """Test MAX pooling value."""
        assert PoolingMode.MAX.value == "max"

    def test_all_modes_in_valid_set(self) -> None:
        """Test all enum values are in VALID_POOLING_MODES."""
        for mode in PoolingMode:
            assert mode.value in VALID_POOLING_MODES


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            max_seq_length=256,
            normalize=True,
            pooling_mode=PoolingMode.MEAN,
            device="cpu",
            batch_size=32,
        )
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.max_seq_length == 256
        assert config.normalize is True
        assert config.pooling_mode == PoolingMode.MEAN
        assert config.device == "cpu"
        assert config.batch_size == 32

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            max_seq_length=256,
            normalize=True,
            pooling_mode=PoolingMode.MEAN,
            device="cpu",
            batch_size=32,
        )
        with pytest.raises(AttributeError):
            config.normalize = False  # type: ignore[misc]


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = SimilarityResult(
            indices=(0, 5, 2),
            scores=(0.95, 0.87, 0.82),
            texts=("text1", "text2", "text3"),
        )
        assert len(result.indices) == 3
        assert result.scores[0] == pytest.approx(0.95)
        assert result.texts is not None

    def test_without_texts(self) -> None:
        """Test result without texts."""
        result = SimilarityResult(
            indices=(0,),
            scores=(0.9,),
            texts=None,
        )
        assert result.texts is None

    def test_frozen(self) -> None:
        """Test that result is immutable."""
        result = SimilarityResult(
            indices=(0,),
            scores=(0.9,),
            texts=None,
        )
        with pytest.raises(AttributeError):
            result.scores = (1.0,)  # type: ignore[misc]


class TestValidateEmbeddingConfig:
    """Tests for validate_embedding_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            max_seq_length=256,
            normalize=True,
            pooling_mode=PoolingMode.MEAN,
            device="cpu",
            batch_size=32,
        )
        validate_embedding_config(config)  # Should not raise

    def test_empty_model_name_raises(self) -> None:
        """Test that empty model_name raises error."""
        config = EmbeddingConfig(
            model_name="",
            max_seq_length=256,
            normalize=True,
            pooling_mode=PoolingMode.MEAN,
            device="cpu",
            batch_size=32,
        )
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            validate_embedding_config(config)

    def test_zero_max_seq_length_raises(self) -> None:
        """Test that zero max_seq_length raises error."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            max_seq_length=0,
            normalize=True,
            pooling_mode=PoolingMode.MEAN,
            device="cpu",
            batch_size=32,
        )
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_embedding_config(config)

    def test_zero_batch_size_raises(self) -> None:
        """Test that zero batch_size raises error."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            max_seq_length=256,
            normalize=True,
            pooling_mode=PoolingMode.MEAN,
            device="cpu",
            batch_size=0,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_embedding_config(config)


class TestCreateEmbeddingConfig:
    """Tests for create_embedding_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_embedding_config()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.max_seq_length == 256
        assert config.normalize is True
        assert config.pooling_mode == PoolingMode.MEAN
        assert config.device == "cpu"
        assert config.batch_size == 32

    def test_custom_model(self) -> None:
        """Test custom model name."""
        config = create_embedding_config(model_name="all-mpnet-base-v2")
        assert config.model_name == "all-mpnet-base-v2"

    def test_custom_normalize(self) -> None:
        """Test custom normalize setting."""
        config = create_embedding_config(normalize=False)
        assert config.normalize is False

    def test_custom_pooling_mode(self) -> None:
        """Test custom pooling mode."""
        config = create_embedding_config(pooling_mode="cls")
        assert config.pooling_mode == PoolingMode.CLS

    def test_invalid_pooling_mode_raises(self) -> None:
        """Test that invalid pooling mode raises error."""
        with pytest.raises(ValueError, match="pooling_mode must be one of"):
            create_embedding_config(pooling_mode="invalid")

    def test_empty_model_raises(self) -> None:
        """Test that empty model name raises error."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_embedding_config(model_name="")


class TestListPoolingModes:
    """Tests for list_pooling_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_pooling_modes()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_pooling_modes()
        assert result == sorted(result)

    def test_contains_mean(self) -> None:
        """Test that mean is in the list."""
        result = list_pooling_modes()
        assert "mean" in result

    def test_contains_cls(self) -> None:
        """Test that cls is in the list."""
        result = list_pooling_modes()
        assert "cls" in result


class TestListDistanceMetrics:
    """Tests for list_distance_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_distance_metrics()
        assert isinstance(result, list)

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_distance_metrics()
        assert result == sorted(result)

    def test_contains_cosine(self) -> None:
        """Test that cosine is in the list."""
        result = list_distance_metrics()
        assert "cosine" in result

    def test_contains_euclidean(self) -> None:
        """Test that euclidean is in the list."""
        result = list_distance_metrics()
        assert "euclidean" in result


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        a = (1.0, 0.0, 0.0)
        b = (1.0, 0.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors."""
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors_raises(self) -> None:
        """Test that empty vectors raise error."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            cosine_similarity((), ())

    def test_different_lengths_raises(self) -> None:
        """Test that different length vectors raise error."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            cosine_similarity((1.0,), (1.0, 2.0))

    def test_zero_vector_returns_zero(self) -> None:
        """Test that zero vector returns 0."""
        a = (0.0, 0.0)
        b = (1.0, 1.0)
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_3_4_5_triangle(self) -> None:
        """Test classic 3-4-5 triangle."""
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_same_point(self) -> None:
        """Test distance to same point is 0."""
        a = (1.0, 2.0, 3.0)
        b = (1.0, 2.0, 3.0)
        assert euclidean_distance(a, b) == pytest.approx(0.0)

    def test_empty_vectors_raises(self) -> None:
        """Test that empty vectors raise error."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            euclidean_distance((), ())

    def test_different_lengths_raises(self) -> None:
        """Test that different length vectors raise error."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            euclidean_distance((1.0,), (1.0, 2.0))


class TestDotProductSimilarity:
    """Tests for dot_product_similarity function."""

    def test_basic_dot_product(self) -> None:
        """Test basic dot product calculation."""
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert dot_product_similarity(a, b) == pytest.approx(32.0)

    def test_orthogonal_vectors(self) -> None:
        """Test dot product of orthogonal vectors."""
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert dot_product_similarity(a, b) == pytest.approx(0.0)

    def test_empty_vectors_raises(self) -> None:
        """Test that empty vectors raise error."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            dot_product_similarity((), ())

    def test_different_lengths_raises(self) -> None:
        """Test that different length vectors raise error."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            dot_product_similarity((1.0,), (1.0, 2.0))


class TestNormalizeVector:
    """Tests for normalize_vector function."""

    def test_3_4_vector(self) -> None:
        """Test normalization of 3-4 vector."""
        v = (3.0, 4.0)
        norm = normalize_vector(v)
        assert norm[0] == pytest.approx(0.6)
        assert norm[1] == pytest.approx(0.8)

    def test_unit_length(self) -> None:
        """Test that normalized vector has unit length."""
        v = (1.0, 2.0, 3.0)
        norm = normalize_vector(v)
        length = sum(x * x for x in norm) ** 0.5
        assert length == pytest.approx(1.0)

    def test_already_normalized(self) -> None:
        """Test normalizing already normalized vector."""
        v = (1.0, 0.0, 0.0)
        norm = normalize_vector(v)
        assert norm[0] == pytest.approx(1.0)
        assert norm[1] == pytest.approx(0.0)
        assert norm[2] == pytest.approx(0.0)

    def test_empty_vector_raises(self) -> None:
        """Test that empty vector raises error."""
        with pytest.raises(ValueError, match="vector cannot be empty"):
            normalize_vector(())

    def test_zero_vector_raises(self) -> None:
        """Test that zero vector raises error."""
        with pytest.raises(ValueError, match="cannot normalize zero vector"):
            normalize_vector((0.0, 0.0))


class TestGetRecommendedModel:
    """Tests for get_recommended_model function."""

    def test_general_task(self) -> None:
        """Test recommended model for general task."""
        assert get_recommended_model("general") == "all-MiniLM-L6-v2"

    def test_qa_task(self) -> None:
        """Test recommended model for QA task."""
        assert get_recommended_model("qa") == "multi-qa-MiniLM-L6-cos-v1"

    def test_clustering_task(self) -> None:
        """Test recommended model for clustering task."""
        assert get_recommended_model("clustering") == "all-mpnet-base-v2"

    def test_classification_task(self) -> None:
        """Test recommended model for classification task."""
        assert get_recommended_model("classification") == "all-MiniLM-L12-v2"

    def test_invalid_task_raises(self) -> None:
        """Test that invalid task raises error."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_model("invalid")


class TestEstimateEmbeddingMemory:
    """Tests for estimate_embedding_memory function."""

    def test_basic_estimate(self) -> None:
        """Test basic memory estimation."""
        mem = estimate_embedding_memory(1000, embedding_dim=384)
        assert mem > 0

    def test_fp16_uses_less_memory(self) -> None:
        """Test that fp16 uses less memory than fp32."""
        mem_fp16 = estimate_embedding_memory(1000, precision="fp16")
        mem_fp32 = estimate_embedding_memory(1000, precision="fp32")
        assert mem_fp16 < mem_fp32

    def test_more_vectors_more_memory(self) -> None:
        """Test that more vectors use more memory."""
        mem_1000 = estimate_embedding_memory(1000)
        mem_10000 = estimate_embedding_memory(10000)
        assert mem_10000 > mem_1000

    def test_higher_dim_more_memory(self) -> None:
        """Test that higher dimensions use more memory."""
        mem_384 = estimate_embedding_memory(1000, embedding_dim=384)
        mem_768 = estimate_embedding_memory(1000, embedding_dim=768)
        assert mem_768 > mem_384

    def test_zero_vectors_raises(self) -> None:
        """Test that zero vectors raises error."""
        with pytest.raises(ValueError, match="num_vectors must be positive"):
            estimate_embedding_memory(0)

    def test_zero_dim_raises(self) -> None:
        """Test that zero embedding_dim raises error."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            estimate_embedding_memory(1000, embedding_dim=0)

    def test_invalid_precision_raises(self) -> None:
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="precision must be one of"):
            estimate_embedding_memory(1000, precision="int8")  # type: ignore[arg-type]


class TestChunkText:
    """Tests for chunk_text function."""

    def test_short_text_single_chunk(self) -> None:
        """Test that short text returns single chunk."""
        chunks = chunk_text("short text", chunk_size=100)
        assert chunks == ["short text"]

    def test_empty_text_returns_empty(self) -> None:
        """Test that empty text returns empty list."""
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []

    def test_long_text_multiple_chunks(self) -> None:
        """Test that long text returns multiple chunks."""
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1

    def test_chunks_contain_all_words(self) -> None:
        """Test that all words appear in at least one chunk."""
        text = "a b c d e f g h i j"
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        all_words = set()
        for chunk in chunks:
            all_words.update(chunk.split())
        original_words = set(text.split())
        assert all_words == original_words

    def test_zero_chunk_size_raises(self) -> None:
        """Test that zero chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("text", chunk_size=0)

    def test_negative_overlap_raises(self) -> None:
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            chunk_text("text", chunk_size=10, overlap=-1)

    def test_overlap_equals_chunk_size_raises(self) -> None:
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match=r"overlap.*must be less than chunk_size"):
            chunk_text("text", chunk_size=10, overlap=10)


class TestConstants:
    """Tests for module constants."""

    def test_valid_pooling_modes_not_empty(self) -> None:
        """Test VALID_POOLING_MODES is not empty."""
        assert len(VALID_POOLING_MODES) > 0

    def test_valid_distance_metrics_contents(self) -> None:
        """Test VALID_DISTANCE_METRICS contains expected values."""
        assert "cosine" in VALID_DISTANCE_METRICS
        assert "euclidean" in VALID_DISTANCE_METRICS
        assert "dot_product" in VALID_DISTANCE_METRICS
