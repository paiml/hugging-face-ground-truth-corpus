"""Tests for embedding models and pooling strategies functionality."""

from __future__ import annotations

import pytest

from hf_gtc.inference.embeddings import (
    VALID_DISTANCE_METRICS,
    VALID_EMBEDDING_NORMALIZATIONS,
    VALID_POOLING_MODES,
    VALID_POOLING_STRATEGIES,
    VALID_SIMILARITY_METRICS,
    EmbeddingConfig,
    EmbeddingNormalization,
    EmbeddingResult,
    EmbeddingStats,
    PoolingConfig,
    PoolingMode,
    PoolingStrategy,
    SimilarityMetric,
    SimilarityResult,
    calculate_similarity,
    chunk_text,
    compute_pooled_embedding,
    cosine_similarity,
    create_embedding_config,
    create_embedding_result,
    create_embedding_stats,
    create_pooling_config,
    dot_product_similarity,
    estimate_embedding_memory,
    estimate_embedding_quality,
    euclidean_distance,
    format_embedding_stats,
    get_embedding_normalization,
    get_pooling_strategy,
    get_recommended_embedding_config,
    get_recommended_model,
    get_similarity_metric,
    list_distance_metrics,
    list_embedding_normalizations,
    list_pooling_modes,
    list_pooling_strategies,
    list_similarity_metrics,
    normalize_embeddings,
    normalize_vector,
    validate_embedding_config,
    validate_embedding_result,
    validate_embedding_stats,
    validate_pooling_config,
)


class TestPoolingStrategy:
    """Tests for PoolingStrategy enum."""

    def test_cls_value(self) -> None:
        """Test CLS pooling value."""
        assert PoolingStrategy.CLS.value == "cls"

    def test_mean_value(self) -> None:
        """Test MEAN pooling value."""
        assert PoolingStrategy.MEAN.value == "mean"

    def test_max_value(self) -> None:
        """Test MAX pooling value."""
        assert PoolingStrategy.MAX.value == "max"

    def test_weighted_mean_value(self) -> None:
        """Test WEIGHTED_MEAN pooling value."""
        assert PoolingStrategy.WEIGHTED_MEAN.value == "weighted_mean"

    def test_last_token_value(self) -> None:
        """Test LAST_TOKEN pooling value."""
        assert PoolingStrategy.LAST_TOKEN.value == "last_token"

    def test_all_strategies_in_valid_set(self) -> None:
        """Test all enum values are in VALID_POOLING_STRATEGIES."""
        for strategy in PoolingStrategy:
            assert strategy.value in VALID_POOLING_STRATEGIES


class TestEmbeddingNormalization:
    """Tests for EmbeddingNormalization enum."""

    def test_l2_value(self) -> None:
        """Test L2 normalization value."""
        assert EmbeddingNormalization.L2.value == "l2"

    def test_unit_value(self) -> None:
        """Test UNIT normalization value."""
        assert EmbeddingNormalization.UNIT.value == "unit"

    def test_none_value(self) -> None:
        """Test NONE normalization value."""
        assert EmbeddingNormalization.NONE.value == "none"

    def test_all_normalizations_in_valid_set(self) -> None:
        """Test all enum values are in VALID_EMBEDDING_NORMALIZATIONS."""
        for norm in EmbeddingNormalization:
            assert norm.value in VALID_EMBEDDING_NORMALIZATIONS


class TestSimilarityMetric:
    """Tests for SimilarityMetric enum."""

    def test_cosine_value(self) -> None:
        """Test COSINE metric value."""
        assert SimilarityMetric.COSINE.value == "cosine"

    def test_dot_product_value(self) -> None:
        """Test DOT_PRODUCT metric value."""
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"

    def test_euclidean_value(self) -> None:
        """Test EUCLIDEAN metric value."""
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"

    def test_manhattan_value(self) -> None:
        """Test MANHATTAN metric value."""
        assert SimilarityMetric.MANHATTAN.value == "manhattan"

    def test_all_metrics_in_valid_set(self) -> None:
        """Test all enum values are in VALID_SIMILARITY_METRICS."""
        for metric in SimilarityMetric:
            assert metric.value in VALID_SIMILARITY_METRICS


class TestPoolingConfig:
    """Tests for PoolingConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        config = PoolingConfig(
            strategy=PoolingStrategy.MEAN,
            layer_weights=None,
            attention_mask=True,
        )
        assert config.strategy == PoolingStrategy.MEAN
        assert config.layer_weights is None
        assert config.attention_mask is True

    def test_with_layer_weights(self) -> None:
        """Test config with layer weights."""
        config = PoolingConfig(
            strategy=PoolingStrategy.MEAN,
            layer_weights=(0.5, 0.5),
            attention_mask=True,
        )
        assert config.layer_weights == (0.5, 0.5)

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = PoolingConfig(
            strategy=PoolingStrategy.MEAN,
            layer_weights=None,
            attention_mask=True,
        )
        with pytest.raises(AttributeError):
            config.attention_mask = False  # type: ignore[misc]


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_creation(self) -> None:
        """Test basic config creation."""
        pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            pooling_config=pooling,
            normalization=EmbeddingNormalization.L2,
            dimension=384,
            max_length=512,
        )
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.normalization == EmbeddingNormalization.L2

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            pooling_config=pooling,
            normalization=EmbeddingNormalization.L2,
            dimension=384,
            max_length=512,
        )
        with pytest.raises(AttributeError):
            config.dimension = 768  # type: ignore[misc]


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = EmbeddingResult(
            embeddings=((0.1, 0.2), (0.3, 0.4)),
            tokens=(101, 2003, 102),
            attention_mask=(1, 1, 1),
            pooled=(0.2, 0.3),
        )
        assert len(result.embeddings) == 2
        assert result.pooled == (0.2, 0.3)

    def test_without_pooled(self) -> None:
        """Test result without pooled embedding."""
        result = EmbeddingResult(
            embeddings=((0.1, 0.2),),
            tokens=(101, 102),
            attention_mask=(1, 1),
            pooled=None,
        )
        assert result.pooled is None

    def test_frozen(self) -> None:
        """Test that result is immutable."""
        result = EmbeddingResult(
            embeddings=((0.1, 0.2),),
            tokens=(101, 102),
            attention_mask=(1, 1),
            pooled=None,
        )
        with pytest.raises(AttributeError):
            result.pooled = (0.1,)  # type: ignore[misc]


class TestEmbeddingStats:
    """Tests for EmbeddingStats dataclass."""

    def test_creation(self) -> None:
        """Test basic stats creation."""
        stats = EmbeddingStats(
            dimension=384,
            vocab_coverage=0.95,
            avg_magnitude=1.0,
            isotropy_score=0.85,
        )
        assert stats.dimension == 384
        assert stats.vocab_coverage == 0.95
        assert stats.isotropy_score == 0.85

    def test_frozen(self) -> None:
        """Test that stats is immutable."""
        stats = EmbeddingStats(384, 0.95, 1.0, 0.85)
        with pytest.raises(AttributeError):
            stats.dimension = 768  # type: ignore[misc]


class TestValidatePoolingConfig:
    """Tests for validate_pooling_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = PoolingConfig(PoolingStrategy.MEAN, None, True)
        validate_pooling_config(config)  # Should not raise

    def test_valid_with_weights(self) -> None:
        """Test validation passes with valid weights."""
        config = PoolingConfig(PoolingStrategy.MEAN, (0.5, 0.5), True)
        validate_pooling_config(config)  # Should not raise

    def test_empty_weights_raises(self) -> None:
        """Test that empty weights raises error."""
        config = PoolingConfig(PoolingStrategy.MEAN, (), True)
        with pytest.raises(ValueError, match="layer_weights cannot be empty"):
            validate_pooling_config(config)

    def test_weights_not_sum_to_one_raises(self) -> None:
        """Test that weights not summing to 1.0 raises error."""
        config = PoolingConfig(PoolingStrategy.MEAN, (0.3, 0.3), True)
        with pytest.raises(ValueError, match=r"layer_weights must sum to 1\.0"):
            validate_pooling_config(config)

    def test_negative_weights_raises(self) -> None:
        """Test that negative weights raises error."""
        config = PoolingConfig(PoolingStrategy.MEAN, (-0.5, 1.5), True)
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_pooling_config(config)


class TestValidateEmbeddingConfig:
    """Tests for validate_embedding_config function."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            pooling_config=pooling,
            normalization=EmbeddingNormalization.L2,
            dimension=384,
            max_length=512,
        )
        validate_embedding_config(config)  # Should not raise

    def test_empty_model_name_raises(self) -> None:
        """Test that empty model_name raises error."""
        pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        config = EmbeddingConfig(
            model_name="",
            pooling_config=pooling,
            normalization=EmbeddingNormalization.L2,
            dimension=384,
            max_length=512,
        )
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            validate_embedding_config(config)

    def test_zero_dimension_raises(self) -> None:
        """Test that zero dimension raises error."""
        pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            pooling_config=pooling,
            normalization=EmbeddingNormalization.L2,
            dimension=0,
            max_length=512,
        )
        with pytest.raises(ValueError, match="dimension must be positive"):
            validate_embedding_config(config)

    def test_zero_max_length_raises(self) -> None:
        """Test that zero max_length raises error."""
        pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            pooling_config=pooling,
            normalization=EmbeddingNormalization.L2,
            dimension=384,
            max_length=0,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_embedding_config(config)


class TestValidateEmbeddingResult:
    """Tests for validate_embedding_result function."""

    def test_valid_result(self) -> None:
        """Test validation passes for valid result."""
        result = EmbeddingResult(
            embeddings=((0.1, 0.2),),
            tokens=(101, 102),
            attention_mask=(1, 1),
            pooled=None,
        )
        validate_embedding_result(result)  # Should not raise

    def test_empty_embeddings_raises(self) -> None:
        """Test that empty embeddings raises error."""
        result = EmbeddingResult(
            embeddings=(),
            tokens=(101,),
            attention_mask=(1,),
            pooled=None,
        )
        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            validate_embedding_result(result)

    def test_empty_tokens_raises(self) -> None:
        """Test that empty tokens raises error."""
        result = EmbeddingResult(
            embeddings=((0.1, 0.2),),
            tokens=(),
            attention_mask=(),
            pooled=None,
        )
        with pytest.raises(ValueError, match="tokens cannot be empty"):
            validate_embedding_result(result)

    def test_mask_length_mismatch_raises(self) -> None:
        """Test that mask length mismatch raises error."""
        result = EmbeddingResult(
            embeddings=((0.1, 0.2),),
            tokens=(101, 102),
            attention_mask=(1,),
            pooled=None,
        )
        with pytest.raises(ValueError, match="attention_mask length"):
            validate_embedding_result(result)


class TestValidateEmbeddingStats:
    """Tests for validate_embedding_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation passes for valid stats."""
        stats = EmbeddingStats(384, 0.95, 1.0, 0.85)
        validate_embedding_stats(stats)  # Should not raise

    def test_zero_dimension_raises(self) -> None:
        """Test that zero dimension raises error."""
        stats = EmbeddingStats(0, 0.95, 1.0, 0.85)
        with pytest.raises(ValueError, match="dimension must be positive"):
            validate_embedding_stats(stats)

    def test_invalid_vocab_coverage_raises(self) -> None:
        """Test that invalid vocab_coverage raises error."""
        stats = EmbeddingStats(384, 1.5, 1.0, 0.85)
        with pytest.raises(ValueError, match="vocab_coverage must be in"):
            validate_embedding_stats(stats)

    def test_negative_magnitude_raises(self) -> None:
        """Test that negative magnitude raises error."""
        stats = EmbeddingStats(384, 0.95, -1.0, 0.85)
        with pytest.raises(ValueError, match="avg_magnitude cannot be negative"):
            validate_embedding_stats(stats)

    def test_invalid_isotropy_raises(self) -> None:
        """Test that invalid isotropy raises error."""
        stats = EmbeddingStats(384, 0.95, 1.0, 1.5)
        with pytest.raises(ValueError, match="isotropy_score must be in"):
            validate_embedding_stats(stats)


class TestCreatePoolingConfig:
    """Tests for create_pooling_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_pooling_config()
        assert config.strategy == PoolingStrategy.MEAN
        assert config.layer_weights is None
        assert config.attention_mask is True

    def test_custom_strategy(self) -> None:
        """Test custom strategy."""
        config = create_pooling_config(strategy="cls")
        assert config.strategy == PoolingStrategy.CLS

    def test_custom_attention_mask(self) -> None:
        """Test custom attention mask setting."""
        config = create_pooling_config(attention_mask=False)
        assert config.attention_mask is False

    def test_invalid_strategy_raises(self) -> None:
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_pooling_config(strategy="invalid")  # type: ignore[arg-type]


class TestCreateEmbeddingConfig:
    """Tests for create_embedding_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_embedding_config()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.max_length == 512
        assert config.normalization == EmbeddingNormalization.L2

    def test_custom_model(self) -> None:
        """Test custom model name."""
        config = create_embedding_config(model_name="custom-model")
        assert config.model_name == "custom-model"

    def test_custom_normalization(self) -> None:
        """Test custom normalization."""
        config = create_embedding_config(normalization="none")
        assert config.normalization == EmbeddingNormalization.NONE

    def test_custom_pooling_strategy(self) -> None:
        """Test custom pooling strategy."""
        config = create_embedding_config(pooling_strategy="cls")
        assert config.pooling_config.strategy == PoolingStrategy.CLS

    def test_invalid_normalization_raises(self) -> None:
        """Test that invalid normalization raises error."""
        with pytest.raises(ValueError, match="normalization must be one of"):
            create_embedding_config(normalization="invalid")  # type: ignore[arg-type]

    def test_empty_model_raises(self) -> None:
        """Test that empty model name raises error."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_embedding_config(model_name="")


class TestCreateEmbeddingResult:
    """Tests for create_embedding_result function."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = create_embedding_result(
            embeddings=((0.1, 0.2),),
            tokens=(101, 102),
            attention_mask=(1, 1),
        )
        assert len(result.embeddings) == 1
        assert result.pooled is None

    def test_with_pooled(self) -> None:
        """Test result with pooled embedding."""
        result = create_embedding_result(
            embeddings=((0.1, 0.2),),
            tokens=(101, 102),
            attention_mask=(1, 1),
            pooled=(0.15, 0.25),
        )
        assert result.pooled == (0.15, 0.25)

    def test_empty_embeddings_raises(self) -> None:
        """Test that empty embeddings raises error."""
        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            create_embedding_result(
                embeddings=(),
                tokens=(101,),
                attention_mask=(1,),
            )


class TestCreateEmbeddingStats:
    """Tests for create_embedding_stats function."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = create_embedding_stats()
        assert stats.dimension == 384
        assert stats.vocab_coverage == 1.0
        assert stats.avg_magnitude == 1.0
        assert stats.isotropy_score == 0.5

    def test_custom_dimension(self) -> None:
        """Test custom dimension."""
        stats = create_embedding_stats(dimension=768)
        assert stats.dimension == 768

    def test_custom_isotropy(self) -> None:
        """Test custom isotropy score."""
        stats = create_embedding_stats(isotropy_score=0.9)
        assert stats.isotropy_score == 0.9

    def test_zero_dimension_raises(self) -> None:
        """Test that zero dimension raises error."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            create_embedding_stats(dimension=0)


class TestListFunctions:
    """Tests for list functions."""

    def test_list_pooling_strategies(self) -> None:
        """Test list_pooling_strategies returns sorted list."""
        strategies = list_pooling_strategies()
        assert isinstance(strategies, list)
        assert "mean" in strategies
        assert "cls" in strategies
        assert "weighted_mean" in strategies
        assert strategies == sorted(strategies)

    def test_list_embedding_normalizations(self) -> None:
        """Test list_embedding_normalizations returns sorted list."""
        norms = list_embedding_normalizations()
        assert isinstance(norms, list)
        assert "l2" in norms
        assert "none" in norms
        assert norms == sorted(norms)

    def test_list_similarity_metrics(self) -> None:
        """Test list_similarity_metrics returns sorted list."""
        metrics = list_similarity_metrics()
        assert isinstance(metrics, list)
        assert "cosine" in metrics
        assert "euclidean" in metrics
        assert metrics == sorted(metrics)

    def test_list_pooling_modes(self) -> None:
        """Test list_pooling_modes returns sorted list (backward compat)."""
        modes = list_pooling_modes()
        assert isinstance(modes, list)
        assert "mean" in modes
        assert modes == sorted(modes)

    def test_list_distance_metrics(self) -> None:
        """Test list_distance_metrics returns sorted list (backward compat)."""
        metrics = list_distance_metrics()
        assert isinstance(metrics, list)
        assert "cosine" in metrics
        assert metrics == sorted(metrics)


class TestGetFunctions:
    """Tests for get functions."""

    def test_get_pooling_strategy_valid(self) -> None:
        """Test get_pooling_strategy with valid name."""
        assert get_pooling_strategy("mean") == PoolingStrategy.MEAN
        assert get_pooling_strategy("cls") == PoolingStrategy.CLS

    def test_get_pooling_strategy_invalid(self) -> None:
        """Test get_pooling_strategy with invalid name."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            get_pooling_strategy("invalid")

    def test_get_embedding_normalization_valid(self) -> None:
        """Test get_embedding_normalization with valid name."""
        assert get_embedding_normalization("l2") == EmbeddingNormalization.L2
        assert get_embedding_normalization("none") == EmbeddingNormalization.NONE

    def test_get_embedding_normalization_invalid(self) -> None:
        """Test get_embedding_normalization with invalid name."""
        with pytest.raises(ValueError, match="Unknown embedding normalization"):
            get_embedding_normalization("invalid")

    def test_get_similarity_metric_valid(self) -> None:
        """Test get_similarity_metric with valid name."""
        assert get_similarity_metric("cosine") == SimilarityMetric.COSINE
        assert get_similarity_metric("dot_product") == SimilarityMetric.DOT_PRODUCT

    def test_get_similarity_metric_invalid(self) -> None:
        """Test get_similarity_metric with invalid name."""
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            get_similarity_metric("invalid")


class TestCalculateSimilarity:
    """Tests for calculate_similarity function."""

    def test_cosine_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors."""
        a = (1.0, 0.0, 0.0)
        b = (1.0, 0.0, 0.0)
        assert calculate_similarity(a, b, "cosine") == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self) -> None:
        """Test cosine similarity of orthogonal vectors."""
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert calculate_similarity(a, b, "cosine") == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors."""
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert calculate_similarity(a, b, "cosine") == pytest.approx(-1.0)

    def test_dot_product(self) -> None:
        """Test dot product calculation."""
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        # 1*4 + 2*5 + 3*6 = 32
        assert calculate_similarity(a, b, "dot_product") == pytest.approx(32.0)

    def test_euclidean_3_4_5(self) -> None:
        """Test Euclidean distance with 3-4-5 triangle."""
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        assert calculate_similarity(a, b, "euclidean") == pytest.approx(5.0)

    def test_manhattan(self) -> None:
        """Test Manhattan distance."""
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        assert calculate_similarity(a, b, "manhattan") == pytest.approx(7.0)

    def test_empty_vectors_raises(self) -> None:
        """Test that empty vectors raise error."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            calculate_similarity((), (), "cosine")

    def test_different_lengths_raises(self) -> None:
        """Test that different length vectors raise error."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            calculate_similarity((1.0,), (1.0, 2.0), "cosine")

    def test_invalid_metric_raises(self) -> None:
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="metric must be one of"):
            calculate_similarity((1.0,), (1.0,), "invalid")  # type: ignore[arg-type]

    def test_zero_vector_cosine(self) -> None:
        """Test cosine with zero vector returns 0."""
        a = (0.0, 0.0)
        b = (1.0, 1.0)
        assert calculate_similarity(a, b, "cosine") == pytest.approx(0.0)


class TestComputePooledEmbedding:
    """Tests for compute_pooled_embedding function."""

    def test_mean_pooling(self) -> None:
        """Test mean pooling."""
        emb = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        pooled = compute_pooled_embedding(emb, strategy="mean")
        assert pooled[0] == pytest.approx(3.0)
        assert pooled[1] == pytest.approx(4.0)

    def test_cls_pooling(self) -> None:
        """Test CLS pooling returns first embedding."""
        emb = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        pooled = compute_pooled_embedding(emb, strategy="cls")
        assert pooled == (1.0, 2.0)

    def test_max_pooling(self) -> None:
        """Test max pooling."""
        emb = ((1.0, 6.0), (3.0, 4.0), (5.0, 2.0))
        pooled = compute_pooled_embedding(emb, strategy="max")
        assert pooled == (5.0, 6.0)

    def test_last_token_pooling(self) -> None:
        """Test last token pooling returns last embedding."""
        emb = ((1.0, 2.0), (3.0, 4.0))
        pooled = compute_pooled_embedding(emb, strategy="last_token")
        assert pooled == (3.0, 4.0)

    def test_mean_with_attention_mask(self) -> None:
        """Test mean pooling with attention mask."""
        emb = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        mask = (1, 1, 0)  # Only first two are valid
        pooled = compute_pooled_embedding(emb, attention_mask=mask, strategy="mean")
        assert pooled[0] == pytest.approx(2.0)
        assert pooled[1] == pytest.approx(3.0)

    def test_mean_all_masked(self) -> None:
        """Test mean pooling with all masked returns zeros."""
        emb = ((1.0, 2.0), (3.0, 4.0))
        mask = (0, 0)
        pooled = compute_pooled_embedding(emb, attention_mask=mask, strategy="mean")
        assert pooled == (0.0, 0.0)

    def test_weighted_mean_pooling(self) -> None:
        """Test weighted mean pooling."""
        emb = ((1.0, 0.0), (0.0, 1.0))
        weights = (0.75, 0.25)
        pooled = compute_pooled_embedding(
            emb, strategy="weighted_mean", weights=weights
        )
        assert pooled[0] == pytest.approx(0.75)
        assert pooled[1] == pytest.approx(0.25)

    def test_weighted_mean_zero_weights(self) -> None:
        """Test weighted mean with zero total weight."""
        emb = ((1.0, 2.0), (3.0, 4.0))
        weights = (0.0, 0.0)
        pooled = compute_pooled_embedding(
            emb, strategy="weighted_mean", weights=weights
        )
        assert pooled == (0.0, 0.0)

    def test_weighted_mean_without_weights_raises(self) -> None:
        """Test weighted mean without weights raises error."""
        emb = ((1.0, 2.0),)
        with pytest.raises(ValueError, match="weights required"):
            compute_pooled_embedding(emb, strategy="weighted_mean")

    def test_weighted_mean_length_mismatch_raises(self) -> None:
        """Test weighted mean with wrong weights length raises error."""
        emb = ((1.0, 2.0), (3.0, 4.0))
        weights = (1.0,)
        with pytest.raises(ValueError, match="weights length"):
            compute_pooled_embedding(emb, strategy="weighted_mean", weights=weights)

    def test_empty_embeddings_raises(self) -> None:
        """Test that empty embeddings raises error."""
        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            compute_pooled_embedding((), strategy="mean")

    def test_invalid_strategy_raises(self) -> None:
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            compute_pooled_embedding(((1.0,),), strategy="invalid")  # type: ignore[arg-type]


class TestNormalizeEmbeddings:
    """Tests for normalize_embeddings function."""

    def test_l2_normalization(self) -> None:
        """Test L2 normalization."""
        emb = ((3.0, 4.0),)
        norm = normalize_embeddings(emb, "l2")
        assert norm[0][0] == pytest.approx(0.6)
        assert norm[0][1] == pytest.approx(0.8)

    def test_unit_normalization(self) -> None:
        """Test unit normalization (same as L2)."""
        emb = ((3.0, 4.0),)
        norm = normalize_embeddings(emb, "unit")
        assert norm[0][0] == pytest.approx(0.6)
        assert norm[0][1] == pytest.approx(0.8)

    def test_none_normalization(self) -> None:
        """Test no normalization."""
        emb = ((1.0, 2.0),)
        norm = normalize_embeddings(emb, "none")
        assert norm == emb

    def test_zero_vector_unchanged(self) -> None:
        """Test that zero vector is unchanged."""
        emb = ((0.0, 0.0),)
        norm = normalize_embeddings(emb, "l2")
        assert norm == emb

    def test_multiple_embeddings(self) -> None:
        """Test normalizing multiple embeddings."""
        emb = ((3.0, 4.0), (0.0, 5.0))
        norm = normalize_embeddings(emb, "l2")
        assert norm[0][0] == pytest.approx(0.6)
        assert norm[1][1] == pytest.approx(1.0)

    def test_empty_embeddings_raises(self) -> None:
        """Test that empty embeddings raises error."""
        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            normalize_embeddings((), "l2")

    def test_invalid_normalization_raises(self) -> None:
        """Test that invalid normalization raises error."""
        with pytest.raises(ValueError, match="normalization must be one of"):
            normalize_embeddings(((1.0,),), "invalid")  # type: ignore[arg-type]


class TestEstimateEmbeddingQuality:
    """Tests for estimate_embedding_quality function."""

    def test_basic_quality(self) -> None:
        """Test basic quality estimation."""
        emb = ((1.0, 0.0), (0.0, 1.0))
        stats = estimate_embedding_quality(emb)
        assert stats.dimension == 2
        assert stats.avg_magnitude == pytest.approx(1.0)

    def test_uniform_distribution_high_isotropy(self) -> None:
        """Test that uniform distribution has high isotropy."""
        # Embeddings pointing in opposite directions (centroid at origin)
        emb = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        stats = estimate_embedding_quality(emb)
        # Should have high isotropy
        assert stats.isotropy_score >= 0.5

    def test_single_embedding(self) -> None:
        """Test quality estimation for single embedding."""
        emb = ((1.0, 0.0),)
        stats = estimate_embedding_quality(emb)
        assert stats.dimension == 2
        assert stats.isotropy_score == 0.5  # Default for single embedding

    def test_empty_embeddings_raises(self) -> None:
        """Test that empty embeddings raises error."""
        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            estimate_embedding_quality(())


class TestFormatEmbeddingStats:
    """Tests for format_embedding_stats function."""

    def test_formatting(self) -> None:
        """Test basic formatting."""
        stats = EmbeddingStats(384, 0.95, 1.0, 0.85)
        formatted = format_embedding_stats(stats)
        assert "Dimension: 384" in formatted
        assert "Vocab Coverage: 95.00%" in formatted
        assert "Avg Magnitude: 1.0000" in formatted
        assert "Isotropy Score: 0.85" in formatted

    def test_different_values(self) -> None:
        """Test formatting with different values."""
        stats = EmbeddingStats(768, 1.0, 0.98, 0.72)
        formatted = format_embedding_stats(stats)
        assert "Dimension: 768" in formatted
        assert "Isotropy Score: 0.72" in formatted


class TestGetRecommendedEmbeddingConfig:
    """Tests for get_recommended_embedding_config function."""

    def test_general_task(self) -> None:
        """Test recommended config for general task."""
        config = get_recommended_embedding_config("general")
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimension == 384

    def test_qa_task(self) -> None:
        """Test recommended config for QA task."""
        config = get_recommended_embedding_config("qa")
        assert "multi-qa" in config.model_name
        assert config.max_length == 512

    def test_clustering_task(self) -> None:
        """Test recommended config for clustering task."""
        config = get_recommended_embedding_config("clustering")
        assert config.dimension == 768

    def test_classification_task(self) -> None:
        """Test recommended config for classification task."""
        config = get_recommended_embedding_config("classification")
        assert config.pooling_config.strategy == PoolingStrategy.CLS

    def test_retrieval_task(self) -> None:
        """Test recommended config for retrieval task."""
        config = get_recommended_embedding_config("retrieval")
        assert config.dimension == 768
        assert config.max_length == 512

    def test_invalid_task_raises(self) -> None:
        """Test that invalid task raises error."""
        with pytest.raises(ValueError, match="Unknown task"):
            get_recommended_embedding_config("invalid")  # type: ignore[arg-type]


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_pooling_mode_alias(self) -> None:
        """Test PoolingMode is alias for PoolingStrategy."""
        assert PoolingMode is PoolingStrategy

    def test_valid_pooling_modes_alias(self) -> None:
        """Test VALID_POOLING_MODES is alias."""
        assert VALID_POOLING_MODES == VALID_POOLING_STRATEGIES

    def test_cosine_similarity_function(self) -> None:
        """Test cosine_similarity backward compat function."""
        a = (1.0, 0.0)
        b = (1.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_euclidean_distance_function(self) -> None:
        """Test euclidean_distance backward compat function."""
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_dot_product_similarity_function(self) -> None:
        """Test dot_product_similarity backward compat function."""
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        assert dot_product_similarity(a, b) == pytest.approx(32.0)


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


class TestConstants:
    """Tests for module constants."""

    def test_valid_pooling_strategies_not_empty(self) -> None:
        """Test VALID_POOLING_STRATEGIES is not empty."""
        assert len(VALID_POOLING_STRATEGIES) > 0

    def test_valid_embedding_normalizations_not_empty(self) -> None:
        """Test VALID_EMBEDDING_NORMALIZATIONS is not empty."""
        assert len(VALID_EMBEDDING_NORMALIZATIONS) > 0

    def test_valid_similarity_metrics_not_empty(self) -> None:
        """Test VALID_SIMILARITY_METRICS is not empty."""
        assert len(VALID_SIMILARITY_METRICS) > 0

    def test_valid_distance_metrics_contents(self) -> None:
        """Test VALID_DISTANCE_METRICS contains expected values."""
        assert "cosine" in VALID_DISTANCE_METRICS
        assert "euclidean" in VALID_DISTANCE_METRICS
        assert "dot_product" in VALID_DISTANCE_METRICS
