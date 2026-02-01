"""Tests for rag.reranking module."""

from __future__ import annotations

import pytest

from hf_gtc.rag.reranking import (
    VALID_FUSION_METHODS,
    VALID_RERANKER_TYPES,
    FusionConfig,
    FusionMethod,
    HybridSearchConfig,
    RerankerConfig,
    RerankerResult,
    RerankerType,
    calculate_linear_fusion,
    calculate_rrf_score,
    create_fusion_config,
    create_hybrid_search_config,
    create_reranker_config,
    create_reranker_result,
    get_fusion_method,
    get_reranker_type,
    list_fusion_methods,
    list_reranker_types,
    validate_fusion_config,
    validate_hybrid_search_config,
    validate_reranker_config,
)


class TestRerankerType:
    """Tests for RerankerType enum."""

    def test_all_types_have_values(self) -> None:
        """All reranker types have string values."""
        for reranker_type in RerankerType:
            assert isinstance(reranker_type.value, str)

    def test_cross_encoder_value(self) -> None:
        """Cross encoder has correct value."""
        assert RerankerType.CROSS_ENCODER.value == "cross_encoder"

    def test_bi_encoder_value(self) -> None:
        """Bi encoder has correct value."""
        assert RerankerType.BI_ENCODER.value == "bi_encoder"

    def test_colbert_value(self) -> None:
        """ColBERT has correct value."""
        assert RerankerType.COLBERT.value == "colbert"

    def test_bm25_value(self) -> None:
        """BM25 has correct value."""
        assert RerankerType.BM25.value == "bm25"

    def test_valid_reranker_types_frozenset(self) -> None:
        """VALID_RERANKER_TYPES is a frozenset."""
        assert isinstance(VALID_RERANKER_TYPES, frozenset)

    def test_valid_reranker_types_contains_all(self) -> None:
        """VALID_RERANKER_TYPES contains all enum values."""
        for reranker_type in RerankerType:
            assert reranker_type.value in VALID_RERANKER_TYPES


class TestFusionMethod:
    """Tests for FusionMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All fusion methods have string values."""
        for method in FusionMethod:
            assert isinstance(method.value, str)

    def test_rrf_value(self) -> None:
        """RRF has correct value."""
        assert FusionMethod.RRF.value == "rrf"

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert FusionMethod.LINEAR.value == "linear"

    def test_weighted_value(self) -> None:
        """Weighted has correct value."""
        assert FusionMethod.WEIGHTED.value == "weighted"

    def test_valid_fusion_methods_frozenset(self) -> None:
        """VALID_FUSION_METHODS is a frozenset."""
        assert isinstance(VALID_FUSION_METHODS, frozenset)

    def test_valid_fusion_methods_contains_all(self) -> None:
        """VALID_FUSION_METHODS contains all enum values."""
        for method in FusionMethod:
            assert method.value in VALID_FUSION_METHODS


class TestRerankerConfig:
    """Tests for RerankerConfig dataclass."""

    def test_create_config(self) -> None:
        """Create reranker config."""
        config = RerankerConfig(
            model_id="BAAI/bge-reranker-base",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=512,
            batch_size=32,
        )
        assert config.model_id == "BAAI/bge-reranker-base"
        assert config.reranker_type == RerankerType.CROSS_ENCODER
        assert config.max_length == 512
        assert config.batch_size == 32

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RerankerConfig(
            model_id="test",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=512,
            batch_size=32,
        )
        with pytest.raises(AttributeError):
            config.model_id = "new"  # type: ignore[misc]


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create fusion config."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(0.5, 0.5),
            k=60,
        )
        assert config.method == FusionMethod.RRF
        assert config.weights == (0.5, 0.5)
        assert config.k == 60

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(0.5, 0.5),
            k=60,
        )
        with pytest.raises(AttributeError):
            config.k = 100  # type: ignore[misc]


class TestRerankerResult:
    """Tests for RerankerResult dataclass."""

    def test_create_result(self) -> None:
        """Create reranker result."""
        result = RerankerResult(
            document_id="doc_1",
            original_score=0.75,
            reranked_score=0.92,
            rank=1,
        )
        assert result.document_id == "doc_1"
        assert result.original_score == pytest.approx(0.75)
        assert result.reranked_score == pytest.approx(0.92)
        assert result.rank == 1

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = RerankerResult(
            document_id="doc_1",
            original_score=0.75,
            reranked_score=0.92,
            rank=1,
        )
        with pytest.raises(AttributeError):
            result.rank = 2  # type: ignore[misc]


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig dataclass."""

    def test_create_config(self) -> None:
        """Create hybrid search config."""
        config = HybridSearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            fusion_method=FusionMethod.WEIGHTED,
        )
        assert config.dense_weight == pytest.approx(0.7)
        assert config.sparse_weight == pytest.approx(0.3)
        assert config.fusion_method == FusionMethod.WEIGHTED

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = HybridSearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            fusion_method=FusionMethod.WEIGHTED,
        )
        with pytest.raises(AttributeError):
            config.dense_weight = 0.8  # type: ignore[misc]


class TestValidateRerankerConfig:
    """Tests for validate_reranker_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RerankerConfig(
            model_id="BAAI/bge-reranker-base",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=512,
            batch_size=32,
        )
        validate_reranker_config(config)  # Should not raise

    def test_empty_model_id_raises(self) -> None:
        """Empty model_id raises ValueError."""
        config = RerankerConfig(
            model_id="",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=512,
            batch_size=32,
        )
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            validate_reranker_config(config)

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        config = RerankerConfig(
            model_id="test",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=0,
            batch_size=32,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_reranker_config(config)

    def test_negative_max_length_raises(self) -> None:
        """Negative max_length raises ValueError."""
        config = RerankerConfig(
            model_id="test",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=-1,
            batch_size=32,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_reranker_config(config)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        config = RerankerConfig(
            model_id="test",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=512,
            batch_size=0,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_reranker_config(config)

    def test_negative_batch_size_raises(self) -> None:
        """Negative batch_size raises ValueError."""
        config = RerankerConfig(
            model_id="test",
            reranker_type=RerankerType.CROSS_ENCODER,
            max_length=512,
            batch_size=-1,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_reranker_config(config)


class TestValidateFusionConfig:
    """Tests for validate_fusion_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(0.5, 0.5),
            k=60,
        )
        validate_fusion_config(config)  # Should not raise

    def test_empty_weights_raises(self) -> None:
        """Empty weights raises ValueError."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(),
            k=60,
        )
        with pytest.raises(ValueError, match="weights cannot be empty"):
            validate_fusion_config(config)

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(0.5, -0.1),
            k=60,
        )
        with pytest.raises(ValueError, match=r"weight at index 1 must be non-negative"):
            validate_fusion_config(config)

    def test_first_negative_weight_raises(self) -> None:
        """First negative weight raises ValueError."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(-0.5, 0.5),
            k=60,
        )
        with pytest.raises(ValueError, match=r"weight at index 0 must be non-negative"):
            validate_fusion_config(config)

    def test_zero_k_raises(self) -> None:
        """Zero k raises ValueError."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(0.5, 0.5),
            k=0,
        )
        with pytest.raises(ValueError, match="k must be positive"):
            validate_fusion_config(config)

    def test_negative_k_raises(self) -> None:
        """Negative k raises ValueError."""
        config = FusionConfig(
            method=FusionMethod.RRF,
            weights=(0.5, 0.5),
            k=-1,
        )
        with pytest.raises(ValueError, match="k must be positive"):
            validate_fusion_config(config)


class TestValidateHybridSearchConfig:
    """Tests for validate_hybrid_search_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = HybridSearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            fusion_method=FusionMethod.WEIGHTED,
        )
        validate_hybrid_search_config(config)  # Should not raise

    def test_negative_dense_weight_raises(self) -> None:
        """Negative dense_weight raises ValueError."""
        config = HybridSearchConfig(
            dense_weight=-0.1,
            sparse_weight=0.3,
            fusion_method=FusionMethod.WEIGHTED,
        )
        with pytest.raises(ValueError, match="dense_weight must be non-negative"):
            validate_hybrid_search_config(config)

    def test_negative_sparse_weight_raises(self) -> None:
        """Negative sparse_weight raises ValueError."""
        config = HybridSearchConfig(
            dense_weight=0.7,
            sparse_weight=-0.1,
            fusion_method=FusionMethod.WEIGHTED,
        )
        with pytest.raises(ValueError, match="sparse_weight must be non-negative"):
            validate_hybrid_search_config(config)

    def test_zero_total_weight_raises(self) -> None:
        """Zero total weight raises ValueError."""
        config = HybridSearchConfig(
            dense_weight=0.0,
            sparse_weight=0.0,
            fusion_method=FusionMethod.WEIGHTED,
        )
        expected_match = r"dense_weight \+ sparse_weight must be positive"
        with pytest.raises(ValueError, match=expected_match):
            validate_hybrid_search_config(config)


class TestCreateRerankerConfig:
    """Tests for create_reranker_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_reranker_config()
        assert config.model_id == "BAAI/bge-reranker-base"
        assert config.reranker_type == RerankerType.CROSS_ENCODER
        assert config.max_length == 512
        assert config.batch_size == 32

    def test_custom_model_id(self) -> None:
        """Create config with custom model_id."""
        config = create_reranker_config(model_id="ms-marco-MiniLM-L-6")
        assert config.model_id == "ms-marco-MiniLM-L-6"

    @pytest.mark.parametrize(
        ("reranker_type", "expected"),
        [
            ("cross_encoder", RerankerType.CROSS_ENCODER),
            ("bi_encoder", RerankerType.BI_ENCODER),
            ("colbert", RerankerType.COLBERT),
            ("bm25", RerankerType.BM25),
        ],
    )
    def test_all_reranker_types(
        self, reranker_type: str, expected: RerankerType
    ) -> None:
        """Create config with all reranker types."""
        config = create_reranker_config(reranker_type=reranker_type)
        assert config.reranker_type == expected

    def test_custom_max_length(self) -> None:
        """Create config with custom max_length."""
        config = create_reranker_config(max_length=256)
        assert config.max_length == 256

    def test_custom_batch_size(self) -> None:
        """Create config with custom batch_size."""
        config = create_reranker_config(batch_size=64)
        assert config.batch_size == 64

    def test_invalid_reranker_type_raises(self) -> None:
        """Invalid reranker_type raises ValueError."""
        with pytest.raises(ValueError, match="reranker_type must be one of"):
            create_reranker_config(reranker_type="invalid")

    def test_empty_model_id_raises(self) -> None:
        """Empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            create_reranker_config(model_id="")

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_reranker_config(max_length=0)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_reranker_config(batch_size=0)


class TestCreateFusionConfig:
    """Tests for create_fusion_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_fusion_config()
        assert config.method == FusionMethod.RRF
        assert config.weights == (0.5, 0.5)
        assert config.k == 60

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("rrf", FusionMethod.RRF),
            ("linear", FusionMethod.LINEAR),
            ("weighted", FusionMethod.WEIGHTED),
        ],
    )
    def test_all_fusion_methods(self, method: str, expected: FusionMethod) -> None:
        """Create config with all fusion methods."""
        config = create_fusion_config(method=method)
        assert config.method == expected

    def test_custom_weights(self) -> None:
        """Create config with custom weights."""
        config = create_fusion_config(weights=(0.6, 0.4))
        assert config.weights == (0.6, 0.4)

    def test_custom_k(self) -> None:
        """Create config with custom k."""
        config = create_fusion_config(k=100)
        assert config.k == 100

    def test_three_weights(self) -> None:
        """Create config with three weights."""
        config = create_fusion_config(weights=(0.5, 0.3, 0.2))
        assert config.weights == (0.5, 0.3, 0.2)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_fusion_config(method="invalid")

    def test_empty_weights_raises(self) -> None:
        """Empty weights raises ValueError."""
        with pytest.raises(ValueError, match="weights cannot be empty"):
            create_fusion_config(weights=())

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="weight at index"):
            create_fusion_config(weights=(0.5, -0.1))

    def test_zero_k_raises(self) -> None:
        """Zero k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            create_fusion_config(k=0)


class TestCreateHybridSearchConfig:
    """Tests for create_hybrid_search_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_hybrid_search_config()
        assert config.dense_weight == pytest.approx(0.7)
        assert config.sparse_weight == pytest.approx(0.3)
        assert config.fusion_method == FusionMethod.WEIGHTED

    def test_custom_dense_weight(self) -> None:
        """Create config with custom dense_weight."""
        config = create_hybrid_search_config(dense_weight=0.8)
        assert config.dense_weight == pytest.approx(0.8)

    def test_custom_sparse_weight(self) -> None:
        """Create config with custom sparse_weight."""
        config = create_hybrid_search_config(sparse_weight=0.2)
        assert config.sparse_weight == pytest.approx(0.2)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("rrf", FusionMethod.RRF),
            ("linear", FusionMethod.LINEAR),
            ("weighted", FusionMethod.WEIGHTED),
        ],
    )
    def test_all_fusion_methods(self, method: str, expected: FusionMethod) -> None:
        """Create config with all fusion methods."""
        config = create_hybrid_search_config(fusion_method=method)
        assert config.fusion_method == expected

    def test_invalid_fusion_method_raises(self) -> None:
        """Invalid fusion_method raises ValueError."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            create_hybrid_search_config(fusion_method="invalid")

    def test_negative_dense_weight_raises(self) -> None:
        """Negative dense_weight raises ValueError."""
        with pytest.raises(ValueError, match="dense_weight must be non-negative"):
            create_hybrid_search_config(dense_weight=-0.1)

    def test_negative_sparse_weight_raises(self) -> None:
        """Negative sparse_weight raises ValueError."""
        with pytest.raises(ValueError, match="sparse_weight must be non-negative"):
            create_hybrid_search_config(sparse_weight=-0.1)

    def test_zero_total_weight_raises(self) -> None:
        """Zero total weight raises ValueError."""
        expected_match = r"dense_weight \+ sparse_weight must be positive"
        with pytest.raises(ValueError, match=expected_match):
            create_hybrid_search_config(dense_weight=0.0, sparse_weight=0.0)


class TestCreateRerankerResult:
    """Tests for create_reranker_result function."""

    def test_create_result(self) -> None:
        """Create reranker result."""
        result = create_reranker_result("doc_1", 0.75, 0.92, 1)
        assert result.document_id == "doc_1"
        assert result.original_score == pytest.approx(0.75)
        assert result.reranked_score == pytest.approx(0.92)
        assert result.rank == 1

    def test_empty_document_id_raises(self) -> None:
        """Empty document_id raises ValueError."""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            create_reranker_result("", 0.75, 0.92, 1)

    def test_zero_rank_raises(self) -> None:
        """Zero rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            create_reranker_result("doc_1", 0.75, 0.92, 0)

    def test_negative_rank_raises(self) -> None:
        """Negative rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            create_reranker_result("doc_1", 0.75, 0.92, -1)

    def test_high_rank(self) -> None:
        """Create result with high rank."""
        result = create_reranker_result("doc_100", 0.1, 0.2, 100)
        assert result.rank == 100

    def test_negative_scores_allowed(self) -> None:
        """Negative scores are allowed."""
        result = create_reranker_result("doc_1", -0.5, -0.3, 1)
        assert result.original_score == -0.5
        assert result.reranked_score == -0.3


class TestListRerankerTypes:
    """Tests for list_reranker_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_reranker_types()
        assert types == sorted(types)

    def test_contains_cross_encoder(self) -> None:
        """Contains cross_encoder."""
        types = list_reranker_types()
        assert "cross_encoder" in types

    def test_contains_colbert(self) -> None:
        """Contains colbert."""
        types = list_reranker_types()
        assert "colbert" in types

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        types = list_reranker_types()
        for reranker_type in RerankerType:
            assert reranker_type.value in types


class TestListFusionMethods:
    """Tests for list_fusion_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_fusion_methods()
        assert methods == sorted(methods)

    def test_contains_rrf(self) -> None:
        """Contains rrf."""
        methods = list_fusion_methods()
        assert "rrf" in methods

    def test_contains_linear(self) -> None:
        """Contains linear."""
        methods = list_fusion_methods()
        assert "linear" in methods

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        methods = list_fusion_methods()
        for method in FusionMethod:
            assert method.value in methods


class TestGetRerankerType:
    """Tests for get_reranker_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("cross_encoder", RerankerType.CROSS_ENCODER),
            ("bi_encoder", RerankerType.BI_ENCODER),
            ("colbert", RerankerType.COLBERT),
            ("bm25", RerankerType.BM25),
        ],
    )
    def test_get_all_types(self, name: str, expected: RerankerType) -> None:
        """Get all reranker types."""
        assert get_reranker_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="reranker_type must be one of"):
            get_reranker_type("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="reranker_type must be one of"):
            get_reranker_type("")


class TestGetFusionMethod:
    """Tests for get_fusion_method function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("rrf", FusionMethod.RRF),
            ("linear", FusionMethod.LINEAR),
            ("weighted", FusionMethod.WEIGHTED),
        ],
    )
    def test_get_all_methods(self, name: str, expected: FusionMethod) -> None:
        """Get all fusion methods."""
        assert get_fusion_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            get_fusion_method("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            get_fusion_method("")


class TestCalculateRRFScore:
    """Tests for calculate_rrf_score function."""

    def test_single_rank(self) -> None:
        """Single rank calculation."""
        score = calculate_rrf_score((1,), k=60)
        expected = 1.0 / (60 + 1)
        assert score == pytest.approx(expected)

    def test_two_ranks(self) -> None:
        """Two ranks calculation."""
        score = calculate_rrf_score((1, 2))
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
        assert score == pytest.approx(expected)

    def test_three_ranks(self) -> None:
        """Three ranks calculation."""
        score = calculate_rrf_score((1, 1, 1), k=60)
        expected = 3.0 / (60 + 1)
        assert score == pytest.approx(expected)

    def test_custom_k(self) -> None:
        """Custom k value."""
        score = calculate_rrf_score((1,), k=100)
        expected = 1.0 / (100 + 1)
        assert score == pytest.approx(expected)

    def test_higher_rank_lower_score(self) -> None:
        """Higher rank gives lower score."""
        score1 = calculate_rrf_score((1,), k=60)
        score10 = calculate_rrf_score((10,), k=60)
        assert score1 > score10

    def test_empty_ranks_raises(self) -> None:
        """Empty ranks raises ValueError."""
        with pytest.raises(ValueError, match="ranks cannot be empty"):
            calculate_rrf_score(())

    def test_zero_rank_raises(self) -> None:
        """Zero rank raises ValueError."""
        with pytest.raises(ValueError, match="all ranks must be positive"):
            calculate_rrf_score((0, 1))

    def test_negative_rank_raises(self) -> None:
        """Negative rank raises ValueError."""
        with pytest.raises(ValueError, match="all ranks must be positive"):
            calculate_rrf_score((-1, 1))

    def test_zero_k_raises(self) -> None:
        """Zero k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            calculate_rrf_score((1, 2), k=0)

    def test_negative_k_raises(self) -> None:
        """Negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            calculate_rrf_score((1, 2), k=-1)

    @pytest.mark.parametrize(
        ("ranks", "k", "expected"),
        [
            ((1,), 60, 0.01639344262295082),
            ((1, 2), 60, 0.03252247488101534),
            ((5, 10), 60, 0.02967032967032967),
        ],
    )
    def test_known_values(
        self, ranks: tuple[int, ...], k: int, expected: float
    ) -> None:
        """Test against known expected values."""
        assert calculate_rrf_score(ranks, k=k) == pytest.approx(expected)


class TestCalculateLinearFusion:
    """Tests for calculate_linear_fusion function."""

    def test_equal_weights_default(self) -> None:
        """Equal weights when not specified."""
        score = calculate_linear_fusion((0.8, 0.6))
        assert score == pytest.approx(0.7)

    def test_custom_weights(self) -> None:
        """Custom weights."""
        score = calculate_linear_fusion((0.8, 0.6), weights=(0.7, 0.3))
        assert score == pytest.approx(0.74)

    def test_three_scores(self) -> None:
        """Three scores with weights."""
        score = calculate_linear_fusion((1.0, 0.5, 0.3), weights=(0.5, 0.3, 0.2))
        assert score == pytest.approx(0.71)

    def test_single_score(self) -> None:
        """Single score."""
        score = calculate_linear_fusion((0.9,))
        assert score == pytest.approx(0.9)

    def test_single_score_with_weight(self) -> None:
        """Single score with weight."""
        score = calculate_linear_fusion((0.9,), weights=(2.0,))
        assert score == pytest.approx(0.9)

    def test_weight_normalization(self) -> None:
        """Weights are normalized."""
        score1 = calculate_linear_fusion((0.8, 0.6), weights=(1.0, 1.0))
        score2 = calculate_linear_fusion((0.8, 0.6), weights=(0.5, 0.5))
        assert score1 == pytest.approx(score2)

    def test_empty_scores_raises(self) -> None:
        """Empty scores raises ValueError."""
        with pytest.raises(ValueError, match="scores cannot be empty"):
            calculate_linear_fusion(())

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        expected_match = "scores and weights must have the same length"
        with pytest.raises(ValueError, match=expected_match):
            calculate_linear_fusion((0.8, 0.6), weights=(0.7,))

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="all weights must be non-negative"):
            calculate_linear_fusion((0.8, 0.6), weights=(0.7, -0.1))

    def test_zero_total_weight_raises(self) -> None:
        """Zero total weight raises ValueError."""
        with pytest.raises(ValueError, match="sum of weights must be positive"):
            calculate_linear_fusion((0.8, 0.6), weights=(0.0, 0.0))

    def test_negative_scores_allowed(self) -> None:
        """Negative scores are allowed."""
        score = calculate_linear_fusion((-0.5, 0.5))
        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("scores", "weights", "expected"),
        [
            ((0.8, 0.6), None, 0.7),
            ((0.8, 0.6), (0.7, 0.3), 0.74),
            ((1.0, 0.0), (0.5, 0.5), 0.5),
            ((0.0, 1.0), (0.5, 0.5), 0.5),
        ],
    )
    def test_known_values(
        self,
        scores: tuple[float, ...],
        weights: tuple[float, ...] | None,
        expected: float,
    ) -> None:
        """Test against known expected values."""
        assert calculate_linear_fusion(scores, weights) == pytest.approx(expected)
