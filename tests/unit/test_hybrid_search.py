"""Tests for rag.hybrid_search module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.rag.hybrid_search import (
    VALID_FUSION_METHODS,
    VALID_RETRIEVAL_MODES,
    VALID_SPARSE_METHODS,
    BM25Config,
    FusionMethod,
    HybridConfig,
    HybridSearchResult,
    HybridStats,
    RetrievalMode,
    SparseConfig,
    SparseMethod,
    calculate_bm25_score,
    calculate_rrf_score,
    create_bm25_config,
    create_hybrid_config,
    create_hybrid_search_result,
    create_sparse_config,
    format_hybrid_stats,
    fuse_rankings,
    get_fusion_method,
    get_recommended_hybrid_config,
    get_retrieval_mode,
    get_sparse_method,
    list_fusion_methods,
    list_retrieval_modes,
    list_sparse_methods,
    optimize_fusion_weights,
    validate_bm25_config,
    validate_hybrid_config,
    validate_hybrid_search_result,
    validate_sparse_config,
)


class TestSparseMethod:
    """Tests for SparseMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All sparse methods have string values."""
        for method in SparseMethod:
            assert isinstance(method.value, str)

    def test_bm25_value(self) -> None:
        """BM25 has correct value."""
        assert SparseMethod.BM25.value == "bm25"

    def test_tfidf_value(self) -> None:
        """TFIDF has correct value."""
        assert SparseMethod.TFIDF.value == "tfidf"

    def test_splade_value(self) -> None:
        """SPLADE has correct value."""
        assert SparseMethod.SPLADE.value == "splade"

    def test_valid_sparse_methods_frozenset(self) -> None:
        """VALID_SPARSE_METHODS is a frozenset."""
        assert isinstance(VALID_SPARSE_METHODS, frozenset)

    def test_valid_sparse_methods_contains_all(self) -> None:
        """VALID_SPARSE_METHODS contains all enum values."""
        for method in SparseMethod:
            assert method.value in VALID_SPARSE_METHODS


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
        """LINEAR has correct value."""
        assert FusionMethod.LINEAR.value == "linear"

    def test_convex_value(self) -> None:
        """CONVEX has correct value."""
        assert FusionMethod.CONVEX.value == "convex"

    def test_learned_value(self) -> None:
        """LEARNED has correct value."""
        assert FusionMethod.LEARNED.value == "learned"

    def test_valid_fusion_methods_frozenset(self) -> None:
        """VALID_FUSION_METHODS is a frozenset."""
        assert isinstance(VALID_FUSION_METHODS, frozenset)

    def test_valid_fusion_methods_contains_all(self) -> None:
        """VALID_FUSION_METHODS contains all enum values."""
        for method in FusionMethod:
            assert method.value in VALID_FUSION_METHODS


class TestRetrievalMode:
    """Tests for RetrievalMode enum."""

    def test_all_modes_have_values(self) -> None:
        """All retrieval modes have string values."""
        for mode in RetrievalMode:
            assert isinstance(mode.value, str)

    def test_dense_only_value(self) -> None:
        """DENSE_ONLY has correct value."""
        assert RetrievalMode.DENSE_ONLY.value == "dense_only"

    def test_sparse_only_value(self) -> None:
        """SPARSE_ONLY has correct value."""
        assert RetrievalMode.SPARSE_ONLY.value == "sparse_only"

    def test_hybrid_value(self) -> None:
        """HYBRID has correct value."""
        assert RetrievalMode.HYBRID.value == "hybrid"

    def test_valid_retrieval_modes_frozenset(self) -> None:
        """VALID_RETRIEVAL_MODES is a frozenset."""
        assert isinstance(VALID_RETRIEVAL_MODES, frozenset)

    def test_valid_retrieval_modes_contains_all(self) -> None:
        """VALID_RETRIEVAL_MODES contains all enum values."""
        for mode in RetrievalMode:
            assert mode.value in VALID_RETRIEVAL_MODES


class TestBM25Config:
    """Tests for BM25Config dataclass."""

    def test_create_config(self) -> None:
        """Create BM25 config."""
        config = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        assert config.k1 == 1.5
        assert config.b == 0.75
        assert config.epsilon == 0.25

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        with pytest.raises(AttributeError):
            config.k1 = 2.0  # type: ignore[misc]

    def test_config_has_slots(self) -> None:
        """Config uses slots."""
        config = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        assert not hasattr(config, "__dict__")


class TestSparseConfig:
    """Tests for SparseConfig dataclass."""

    def test_create_config(self) -> None:
        """Create sparse config."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        config = SparseConfig(
            method=SparseMethod.BM25,
            bm25_config=bm25,
            vocab_size=30000,
        )
        assert config.method == SparseMethod.BM25
        assert config.vocab_size == 30000

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        config = SparseConfig(SparseMethod.BM25, bm25, 30000)
        with pytest.raises(AttributeError):
            config.vocab_size = 50000  # type: ignore[misc]


class TestHybridConfig:
    """Tests for HybridConfig dataclass."""

    def test_create_config(self) -> None:
        """Create hybrid config."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(
            sparse_config=sparse,
            dense_weight=0.7,
            sparse_weight=0.3,
            fusion_method=FusionMethod.RRF,
            top_k=10,
        )
        assert config.dense_weight == 0.7
        assert config.sparse_weight == 0.3
        assert config.fusion_method == FusionMethod.RRF
        assert config.top_k == 10

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, 0.7, 0.3, FusionMethod.RRF, 10)
        with pytest.raises(AttributeError):
            config.top_k = 20  # type: ignore[misc]


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""

    def test_create_result(self) -> None:
        """Create hybrid search result."""
        result = HybridSearchResult(
            doc_ids=("doc1", "doc2"),
            scores=(0.92, 0.85),
            dense_scores=(0.90, 0.80),
            sparse_scores=(0.95, 0.88),
        )
        assert result.doc_ids == ("doc1", "doc2")
        assert result.scores == (0.92, 0.85)
        assert result.dense_scores == (0.90, 0.80)
        assert result.sparse_scores == (0.95, 0.88)

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = HybridSearchResult(("doc1",), (0.9,), (0.85,), (0.95,))
        with pytest.raises(AttributeError):
            result.scores = (0.8,)  # type: ignore[misc]


class TestHybridStats:
    """Tests for HybridStats dataclass."""

    def test_create_stats(self) -> None:
        """Create hybrid stats."""
        stats = HybridStats(
            dense_recall=0.75,
            sparse_recall=0.70,
            hybrid_recall=0.85,
            fusion_improvement=0.10,
        )
        assert stats.dense_recall == 0.75
        assert stats.sparse_recall == 0.70
        assert stats.hybrid_recall == 0.85
        assert stats.fusion_improvement == 0.10

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = HybridStats(0.75, 0.70, 0.85, 0.10)
        with pytest.raises(AttributeError):
            stats.hybrid_recall = 0.90  # type: ignore[misc]


class TestValidateBM25Config:
    """Tests for validate_bm25_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        validate_bm25_config(config)  # Should not raise

    def test_negative_k1_raises(self) -> None:
        """Negative k1 raises ValueError."""
        config = BM25Config(k1=-1.0, b=0.75, epsilon=0.25)
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            validate_bm25_config(config)

    def test_b_too_high_raises(self) -> None:
        """B > 1.0 raises ValueError."""
        config = BM25Config(k1=1.5, b=1.5, epsilon=0.25)
        with pytest.raises(ValueError, match=r"b must be between 0\.0 and 1\.0"):
            validate_bm25_config(config)

    def test_b_negative_raises(self) -> None:
        """Negative b raises ValueError."""
        config = BM25Config(k1=1.5, b=-0.1, epsilon=0.25)
        with pytest.raises(ValueError, match=r"b must be between 0\.0 and 1\.0"):
            validate_bm25_config(config)

    def test_negative_epsilon_raises(self) -> None:
        """Negative epsilon raises ValueError."""
        config = BM25Config(k1=1.5, b=0.75, epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            validate_bm25_config(config)

    def test_zero_k1_valid(self) -> None:
        """Zero k1 is valid."""
        config = BM25Config(k1=0.0, b=0.75, epsilon=0.25)
        validate_bm25_config(config)

    def test_zero_b_valid(self) -> None:
        """Zero b is valid."""
        config = BM25Config(k1=1.5, b=0.0, epsilon=0.25)
        validate_bm25_config(config)

    def test_one_b_valid(self) -> None:
        """B = 1.0 is valid."""
        config = BM25Config(k1=1.5, b=1.0, epsilon=0.25)
        validate_bm25_config(config)


class TestValidateSparseConfig:
    """Tests for validate_sparse_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        config = SparseConfig(SparseMethod.BM25, bm25, 30000)
        validate_sparse_config(config)  # Should not raise

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab_size raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        config = SparseConfig(SparseMethod.BM25, bm25, 0)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_sparse_config(config)

    def test_negative_vocab_size_raises(self) -> None:
        """Negative vocab_size raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        config = SparseConfig(SparseMethod.BM25, bm25, -1)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_sparse_config(config)

    def test_invalid_bm25_config_raises(self) -> None:
        """Invalid BM25 config raises ValueError."""
        bm25 = BM25Config(k1=-1.0, b=0.75, epsilon=0.25)
        config = SparseConfig(SparseMethod.BM25, bm25, 30000)
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            validate_sparse_config(config)


class TestValidateHybridConfig:
    """Tests for validate_hybrid_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, 0.7, 0.3, FusionMethod.RRF, 10)
        validate_hybrid_config(config)  # Should not raise

    def test_negative_dense_weight_raises(self) -> None:
        """Negative dense_weight raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, -0.1, 0.3, FusionMethod.RRF, 10)
        with pytest.raises(ValueError, match="dense_weight must be non-negative"):
            validate_hybrid_config(config)

    def test_negative_sparse_weight_raises(self) -> None:
        """Negative sparse_weight raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, 0.7, -0.1, FusionMethod.RRF, 10)
        with pytest.raises(ValueError, match="sparse_weight must be non-negative"):
            validate_hybrid_config(config)

    def test_zero_total_weight_raises(self) -> None:
        """Zero total weight raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, 0.0, 0.0, FusionMethod.RRF, 10)
        expected_match = r"dense_weight \+ sparse_weight must be positive"
        with pytest.raises(ValueError, match=expected_match):
            validate_hybrid_config(config)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, 0.7, 0.3, FusionMethod.RRF, 0)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_hybrid_config(config)

    def test_negative_top_k_raises(self) -> None:
        """Negative top_k raises ValueError."""
        bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        config = HybridConfig(sparse, 0.7, 0.3, FusionMethod.RRF, -1)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_hybrid_config(config)


class TestValidateHybridSearchResult:
    """Tests for validate_hybrid_search_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes validation."""
        result = HybridSearchResult(("doc1",), (0.9,), (0.85,), (0.95,))
        validate_hybrid_search_result(result)  # Should not raise

    def test_mismatched_scores_raises(self) -> None:
        """Mismatched doc_ids and scores raises ValueError."""
        result = HybridSearchResult(("doc1", "doc2"), (0.9,), (0.85, 0.8), (0.95, 0.9))
        expected_match = "doc_ids and scores must have the same length"
        with pytest.raises(ValueError, match=expected_match):
            validate_hybrid_search_result(result)

    def test_mismatched_dense_scores_raises(self) -> None:
        """Mismatched dense_scores raises ValueError."""
        result = HybridSearchResult(("doc1", "doc2"), (0.9, 0.8), (0.85,), (0.95, 0.9))
        expected_match = "doc_ids and dense_scores must have the same length"
        with pytest.raises(ValueError, match=expected_match):
            validate_hybrid_search_result(result)

    def test_mismatched_sparse_scores_raises(self) -> None:
        """Mismatched sparse_scores raises ValueError."""
        result = HybridSearchResult(("doc1", "doc2"), (0.9, 0.8), (0.85, 0.8), (0.95,))
        expected_match = "doc_ids and sparse_scores must have the same length"
        with pytest.raises(ValueError, match=expected_match):
            validate_hybrid_search_result(result)

    def test_empty_result_valid(self) -> None:
        """Empty result is valid."""
        result = HybridSearchResult((), (), (), ())
        validate_hybrid_search_result(result)


class TestCreateBM25Config:
    """Tests for create_bm25_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_bm25_config()
        assert config.k1 == 1.5
        assert config.b == 0.75
        assert config.epsilon == 0.25

    def test_custom_k1(self) -> None:
        """Create config with custom k1."""
        config = create_bm25_config(k1=2.0)
        assert config.k1 == 2.0

    def test_custom_b(self) -> None:
        """Create config with custom b."""
        config = create_bm25_config(b=0.5)
        assert config.b == 0.5

    def test_custom_epsilon(self) -> None:
        """Create config with custom epsilon."""
        config = create_bm25_config(epsilon=0.1)
        assert config.epsilon == 0.1

    def test_negative_k1_raises(self) -> None:
        """Negative k1 raises ValueError."""
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            create_bm25_config(k1=-1.0)

    def test_invalid_b_raises(self) -> None:
        """Invalid b raises ValueError."""
        with pytest.raises(ValueError, match=r"b must be between 0\.0 and 1\.0"):
            create_bm25_config(b=1.5)


class TestCreateSparseConfig:
    """Tests for create_sparse_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_sparse_config()
        assert config.method == SparseMethod.BM25
        assert config.vocab_size == 30000

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("bm25", SparseMethod.BM25),
            ("tfidf", SparseMethod.TFIDF),
            ("splade", SparseMethod.SPLADE),
        ],
    )
    def test_all_sparse_methods(self, method: str, expected: SparseMethod) -> None:
        """Create config with all sparse methods."""
        config = create_sparse_config(method=method)
        assert config.method == expected

    def test_custom_vocab_size(self) -> None:
        """Create config with custom vocab_size."""
        config = create_sparse_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_custom_bm25_config(self) -> None:
        """Create config with custom BM25 config."""
        bm25 = create_bm25_config(k1=2.0)
        config = create_sparse_config(bm25_config=bm25)
        assert config.bm25_config.k1 == 2.0

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_sparse_config(method="invalid")

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_sparse_config(vocab_size=0)


class TestCreateHybridConfig:
    """Tests for create_hybrid_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_hybrid_config()
        assert config.dense_weight == 0.7
        assert config.sparse_weight == 0.3
        assert config.fusion_method == FusionMethod.RRF
        assert config.top_k == 10

    def test_custom_weights(self) -> None:
        """Create config with custom weights."""
        config = create_hybrid_config(dense_weight=0.6, sparse_weight=0.4)
        assert config.dense_weight == 0.6
        assert config.sparse_weight == 0.4

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("rrf", FusionMethod.RRF),
            ("linear", FusionMethod.LINEAR),
            ("convex", FusionMethod.CONVEX),
            ("learned", FusionMethod.LEARNED),
        ],
    )
    def test_all_fusion_methods(self, method: str, expected: FusionMethod) -> None:
        """Create config with all fusion methods."""
        config = create_hybrid_config(fusion_method=method)
        assert config.fusion_method == expected

    def test_custom_top_k(self) -> None:
        """Create config with custom top_k."""
        config = create_hybrid_config(top_k=20)
        assert config.top_k == 20

    def test_custom_sparse_config(self) -> None:
        """Create config with custom sparse config."""
        sparse = create_sparse_config(method="tfidf")
        config = create_hybrid_config(sparse_config=sparse)
        assert config.sparse_config.method == SparseMethod.TFIDF

    def test_invalid_fusion_method_raises(self) -> None:
        """Invalid fusion_method raises ValueError."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            create_hybrid_config(fusion_method="invalid")

    def test_negative_dense_weight_raises(self) -> None:
        """Negative dense_weight raises ValueError."""
        with pytest.raises(ValueError, match="dense_weight must be non-negative"):
            create_hybrid_config(dense_weight=-0.1)

    def test_zero_top_k_raises(self) -> None:
        """Zero top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            create_hybrid_config(top_k=0)


class TestCreateHybridSearchResult:
    """Tests for create_hybrid_search_result function."""

    def test_basic_result(self) -> None:
        """Create basic result."""
        result = create_hybrid_search_result(
            doc_ids=("doc1", "doc2"),
            scores=(0.92, 0.85),
        )
        assert result.doc_ids == ("doc1", "doc2")
        assert result.scores == (0.92, 0.85)
        assert result.dense_scores == (0.92, 0.85)
        assert result.sparse_scores == (0.92, 0.85)

    def test_result_with_all_scores(self) -> None:
        """Create result with all scores specified."""
        result = create_hybrid_search_result(
            doc_ids=("doc1",),
            scores=(0.9,),
            dense_scores=(0.85,),
            sparse_scores=(0.95,),
        )
        assert result.dense_scores == (0.85,)
        assert result.sparse_scores == (0.95,)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        expected_match = "doc_ids and scores must have the same length"
        with pytest.raises(ValueError, match=expected_match):
            create_hybrid_search_result(
                doc_ids=("doc1", "doc2"),
                scores=(0.9,),
            )

    def test_empty_result(self) -> None:
        """Create empty result."""
        result = create_hybrid_search_result(doc_ids=(), scores=())
        assert result.doc_ids == ()


class TestListSparseMethods:
    """Tests for list_sparse_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_sparse_methods()
        assert methods == sorted(methods)

    def test_contains_bm25(self) -> None:
        """Contains bm25."""
        methods = list_sparse_methods()
        assert "bm25" in methods

    def test_contains_splade(self) -> None:
        """Contains splade."""
        methods = list_sparse_methods()
        assert "splade" in methods

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        methods = list_sparse_methods()
        for method in SparseMethod:
            assert method.value in methods


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

    def test_contains_convex(self) -> None:
        """Contains convex."""
        methods = list_fusion_methods()
        assert "convex" in methods

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        methods = list_fusion_methods()
        for method in FusionMethod:
            assert method.value in methods


class TestListRetrievalModes:
    """Tests for list_retrieval_modes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        modes = list_retrieval_modes()
        assert modes == sorted(modes)

    def test_contains_hybrid(self) -> None:
        """Contains hybrid."""
        modes = list_retrieval_modes()
        assert "hybrid" in modes

    def test_contains_dense_only(self) -> None:
        """Contains dense_only."""
        modes = list_retrieval_modes()
        assert "dense_only" in modes

    def test_contains_all_enum_values(self) -> None:
        """Contains all enum values."""
        modes = list_retrieval_modes()
        for mode in RetrievalMode:
            assert mode.value in modes


class TestGetSparseMethod:
    """Tests for get_sparse_method function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("bm25", SparseMethod.BM25),
            ("tfidf", SparseMethod.TFIDF),
            ("splade", SparseMethod.SPLADE),
        ],
    )
    def test_get_all_methods(self, name: str, expected: SparseMethod) -> None:
        """Get all sparse methods."""
        assert get_sparse_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="sparse_method must be one of"):
            get_sparse_method("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="sparse_method must be one of"):
            get_sparse_method("")


class TestGetFusionMethod:
    """Tests for get_fusion_method function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("rrf", FusionMethod.RRF),
            ("linear", FusionMethod.LINEAR),
            ("convex", FusionMethod.CONVEX),
            ("learned", FusionMethod.LEARNED),
        ],
    )
    def test_get_all_methods(self, name: str, expected: FusionMethod) -> None:
        """Get all fusion methods."""
        assert get_fusion_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            get_fusion_method("invalid")


class TestGetRetrievalMode:
    """Tests for get_retrieval_mode function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("dense_only", RetrievalMode.DENSE_ONLY),
            ("sparse_only", RetrievalMode.SPARSE_ONLY),
            ("hybrid", RetrievalMode.HYBRID),
        ],
    )
    def test_get_all_modes(self, name: str, expected: RetrievalMode) -> None:
        """Get all retrieval modes."""
        assert get_retrieval_mode(name) == expected

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="retrieval_mode must be one of"):
            get_retrieval_mode("invalid")


class TestCalculateBM25Score:
    """Tests for calculate_bm25_score function."""

    def test_basic_calculation(self) -> None:
        """Basic BM25 score calculation."""
        score = calculate_bm25_score(
            tf=3, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
        )
        assert score > 0

    def test_zero_tf_returns_zero(self) -> None:
        """Zero term frequency returns zero score."""
        score = calculate_bm25_score(
            tf=0, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
        )
        assert score == 0.0

    def test_higher_tf_higher_score(self) -> None:
        """Higher term frequency gives higher score."""
        score_low = calculate_bm25_score(
            tf=1, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
        )
        score_high = calculate_bm25_score(
            tf=5, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
        )
        assert score_high > score_low

    def test_higher_df_lower_score(self) -> None:
        """Higher document frequency gives lower score (less discriminative)."""
        score_low_df = calculate_bm25_score(
            tf=3, df=10, doc_len=500, avg_doc_len=400.0, num_docs=10000
        )
        score_high_df = calculate_bm25_score(
            tf=3, df=1000, doc_len=500, avg_doc_len=400.0, num_docs=10000
        )
        assert score_low_df > score_high_df

    def test_custom_config(self) -> None:
        """Custom BM25 config affects score."""
        config = create_bm25_config(k1=2.0, b=0.5)
        score = calculate_bm25_score(
            tf=3, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000, config=config
        )
        assert score > 0

    def test_negative_tf_raises(self) -> None:
        """Negative tf raises ValueError."""
        with pytest.raises(ValueError, match="tf must be non-negative"):
            calculate_bm25_score(
                tf=-1, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
            )

    def test_zero_df_with_positive_tf_raises(self) -> None:
        """Zero df with positive tf raises ValueError."""
        expected_match = "df must be positive when tf is positive"
        with pytest.raises(ValueError, match=expected_match):
            calculate_bm25_score(
                tf=3, df=0, doc_len=500, avg_doc_len=400.0, num_docs=10000
            )

    def test_zero_doc_len_raises(self) -> None:
        """Zero doc_len raises ValueError."""
        with pytest.raises(ValueError, match="doc_len must be positive"):
            calculate_bm25_score(
                tf=3, df=100, doc_len=0, avg_doc_len=400.0, num_docs=10000
            )

    def test_zero_avg_doc_len_raises(self) -> None:
        """Zero avg_doc_len raises ValueError."""
        with pytest.raises(ValueError, match="avg_doc_len must be positive"):
            calculate_bm25_score(
                tf=3, df=100, doc_len=500, avg_doc_len=0.0, num_docs=10000
            )

    def test_zero_num_docs_raises(self) -> None:
        """Zero num_docs raises ValueError."""
        with pytest.raises(ValueError, match="num_docs must be positive"):
            calculate_bm25_score(
                tf=3, df=100, doc_len=500, avg_doc_len=400.0, num_docs=0
            )

    def test_df_exceeds_num_docs_raises(self) -> None:
        """Document frequency exceeding num_docs raises ValueError."""
        expected_match = r"df \(\d+\) cannot exceed num_docs \(\d+\)"
        with pytest.raises(ValueError, match=expected_match):
            calculate_bm25_score(
                tf=3, df=20000, doc_len=500, avg_doc_len=400.0, num_docs=10000
            )


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

    def test_three_same_ranks(self) -> None:
        """Three same ranks calculation."""
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


class TestFuseRankings:
    """Tests for fuse_rankings function."""

    def test_basic_rrf_fusion(self) -> None:
        """Basic RRF fusion."""
        dense = {"doc1": 1, "doc2": 2}
        sparse = {"doc1": 2, "doc2": 1}
        results = fuse_rankings(dense, sparse)
        assert len(results) == 2
        assert results[0][0] in ("doc1", "doc2")
        assert results[0][1] > 0

    def test_documents_only_in_dense(self) -> None:
        """Documents only in dense rankings."""
        dense = {"doc1": 1, "doc2": 2}
        sparse = {"doc3": 1}
        results = fuse_rankings(dense, sparse)
        doc_ids = [r[0] for r in results]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids

    def test_empty_rankings(self) -> None:
        """Empty rankings return empty results."""
        results = fuse_rankings({}, {})
        assert results == []

    def test_linear_fusion(self) -> None:
        """Linear fusion method."""
        dense = {"doc1": 1, "doc2": 2}
        sparse = {"doc1": 2, "doc2": 1}
        results = fuse_rankings(dense, sparse, fusion_method=FusionMethod.LINEAR)
        assert len(results) == 2

    def test_convex_fusion(self) -> None:
        """Convex fusion method."""
        dense = {"doc1": 1}
        sparse = {"doc1": 1}
        results = fuse_rankings(dense, sparse, fusion_method=FusionMethod.CONVEX)
        assert len(results) == 1

    def test_learned_fusion(self) -> None:
        """Learned fusion method (fallback to equal weights)."""
        dense = {"doc1": 1}
        sparse = {"doc1": 1}
        results = fuse_rankings(dense, sparse, fusion_method=FusionMethod.LEARNED)
        assert len(results) == 1

    def test_custom_weights(self) -> None:
        """Custom weights affect fusion."""
        dense = {"doc1": 1, "doc2": 10}
        sparse = {"doc1": 10, "doc2": 1}
        # Heavy dense weight should favor doc1
        results_dense = fuse_rankings(
            dense, sparse, dense_weight=0.9, sparse_weight=0.1
        )
        # Heavy sparse weight should favor doc2
        results_sparse = fuse_rankings(
            dense, sparse, dense_weight=0.1, sparse_weight=0.9
        )
        assert results_dense[0][0] == "doc1"
        assert results_sparse[0][0] == "doc2"

    def test_negative_dense_weight_raises(self) -> None:
        """Negative dense_weight raises ValueError."""
        with pytest.raises(ValueError, match="dense_weight must be non-negative"):
            fuse_rankings({"doc1": 1}, {"doc1": 1}, dense_weight=-0.1)

    def test_negative_sparse_weight_raises(self) -> None:
        """Negative sparse_weight raises ValueError."""
        with pytest.raises(ValueError, match="sparse_weight must be non-negative"):
            fuse_rankings({"doc1": 1}, {"doc1": 1}, sparse_weight=-0.1)

    def test_zero_k_raises(self) -> None:
        """Zero k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            fuse_rankings({"doc1": 1}, {"doc1": 1}, k=0)


class TestOptimizeFusionWeights:
    """Tests for optimize_fusion_weights function."""

    def test_basic_optimization(self) -> None:
        """Basic weight optimization."""
        dense = (0.7, 0.8, 0.75)
        sparse = (0.65, 0.72, 0.80)
        hybrid = (
            (0.7, 0.3, 0.82),
            (0.6, 0.4, 0.85),
            (0.5, 0.5, 0.83),
        )
        weights = optimize_fusion_weights(dense, sparse, hybrid)
        # Weights should sum to 1
        assert weights[0] + weights[1] == pytest.approx(1.0)

    def test_returns_best_weights(self) -> None:
        """Returns weights with best recall."""
        dense = (0.7,)
        sparse = (0.6,)
        hybrid = (
            (0.8, 0.2, 0.75),
            (0.6, 0.4, 0.90),  # Best recall
            (0.4, 0.6, 0.80),
        )
        weights = optimize_fusion_weights(dense, sparse, hybrid)
        # Should return (0.6, 0.4) normalized
        assert weights[0] == pytest.approx(0.6)
        assert weights[1] == pytest.approx(0.4)

    def test_empty_hybrid_recalls_raises(self) -> None:
        """Empty hybrid_recalls raises ValueError."""
        with pytest.raises(ValueError, match="hybrid_recalls cannot be empty"):
            optimize_fusion_weights((), (), ())

    def test_zero_total_weight_returns_default(self) -> None:
        """Zero total weight returns default (0.5, 0.5)."""
        weights = optimize_fusion_weights((0.7,), (0.6,), ((0.0, 0.0, 0.5),))
        assert weights == (0.5, 0.5)


class TestFormatHybridStats:
    """Tests for format_hybrid_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = HybridStats(0.75, 0.70, 0.85, 0.10)
        formatted = format_hybrid_stats(stats)
        assert "dense=0.750" in formatted
        assert "sparse=0.700" in formatted
        assert "hybrid=0.850" in formatted
        assert "+10.0%" in formatted

    def test_negative_improvement(self) -> None:
        """Negative improvement formatting."""
        stats = HybridStats(0.80, 0.75, 0.78, -0.02)
        formatted = format_hybrid_stats(stats)
        assert "-2.0%" in formatted

    def test_zero_improvement(self) -> None:
        """Zero improvement formatting."""
        stats = HybridStats(0.80, 0.75, 0.80, 0.0)
        formatted = format_hybrid_stats(stats)
        assert "+0.0%" in formatted


class TestGetRecommendedHybridConfig:
    """Tests for get_recommended_hybrid_config function."""

    def test_default_config(self) -> None:
        """Default config."""
        config = get_recommended_hybrid_config()
        assert config.fusion_method == FusionMethod.RRF
        assert config.dense_weight > 0
        assert config.sparse_weight > 0

    def test_semantic_use_case(self) -> None:
        """Semantic use case has higher dense weight."""
        config = get_recommended_hybrid_config(use_case="semantic")
        assert config.dense_weight > config.sparse_weight

    def test_lexical_use_case(self) -> None:
        """Lexical use case has higher sparse weight."""
        config = get_recommended_hybrid_config(use_case="lexical")
        assert config.sparse_weight > config.dense_weight

    def test_qa_use_case(self) -> None:
        """QA use case configuration."""
        config = get_recommended_hybrid_config(use_case="qa")
        assert config.dense_weight >= 0.7

    def test_short_docs_bm25_params(self) -> None:
        """Short documents affect BM25 params."""
        config = get_recommended_hybrid_config(avg_doc_length=100)
        assert config.sparse_config.bm25_config.b < 0.75

    def test_long_docs_bm25_params(self) -> None:
        """Long documents affect BM25 params."""
        config = get_recommended_hybrid_config(avg_doc_length=2000)
        assert config.sparse_config.bm25_config.b > 0.75

    def test_large_corpus_vocab_size(self) -> None:
        """Large corpus has larger vocab_size."""
        config_small = get_recommended_hybrid_config(corpus_size=10000)
        config_large = get_recommended_hybrid_config(corpus_size=500000)
        small_vocab = config_small.sparse_config.vocab_size
        large_vocab = config_large.sparse_config.vocab_size
        assert large_vocab >= small_vocab

    def test_zero_corpus_size_raises(self) -> None:
        """Zero corpus_size raises ValueError."""
        with pytest.raises(ValueError, match="corpus_size must be positive"):
            get_recommended_hybrid_config(corpus_size=0)

    def test_zero_avg_doc_length_raises(self) -> None:
        """Zero avg_doc_length raises ValueError."""
        with pytest.raises(ValueError, match="avg_doc_length must be positive"):
            get_recommended_hybrid_config(avg_doc_length=0)

    def test_unknown_use_case_uses_general(self) -> None:
        """Unknown use case uses general defaults."""
        config_unknown = get_recommended_hybrid_config(use_case="unknown")
        config_general = get_recommended_hybrid_config(use_case="general")
        assert config_unknown.dense_weight == config_general.dense_weight

    def test_small_corpus_uses_linear_fusion(self) -> None:
        """Small corpus may use linear fusion."""
        config = get_recommended_hybrid_config(corpus_size=1000)
        assert config.fusion_method == FusionMethod.LINEAR

    def test_large_corpus_sparse_weight_increases(self) -> None:
        """Large corpus increases sparse weight."""
        config = get_recommended_hybrid_config(corpus_size=1000000)
        assert config.sparse_weight >= 0.4


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        k1=st.floats(min_value=0.0, max_value=10.0),
        b=st.floats(min_value=0.0, max_value=1.0),
        epsilon=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_bm25_config_always_validates(
        self, k1: float, b: float, epsilon: float
    ) -> None:
        """Valid BM25 config always passes validation."""
        config = BM25Config(k1=k1, b=b, epsilon=epsilon)
        validate_bm25_config(config)

    @given(
        tf=st.integers(min_value=1, max_value=100),
        df=st.integers(min_value=1, max_value=1000),
        doc_len=st.integers(min_value=1, max_value=10000),
        avg_doc_len=st.floats(min_value=1.0, max_value=10000.0),
    )
    @settings(max_examples=50)
    def test_bm25_score_always_positive(
        self, tf: int, df: int, doc_len: int, avg_doc_len: float
    ) -> None:
        """BM25 score is always positive for valid inputs."""
        num_docs = max(df + 1, 10000)
        score = calculate_bm25_score(
            tf=tf, df=df, doc_len=doc_len, avg_doc_len=avg_doc_len, num_docs=num_docs
        )
        assert score > 0

    @given(
        ranks=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=5),
        k=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=50)
    def test_rrf_score_always_positive(self, ranks: list[int], k: int) -> None:
        """RRF score is always positive for valid inputs."""
        score = calculate_rrf_score(tuple(ranks), k=k)
        assert score > 0

    @given(
        dense_weight=st.floats(min_value=0.0, max_value=1.0),
        sparse_weight=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_hybrid_config_valid_when_weights_valid(
        self, dense_weight: float, sparse_weight: float
    ) -> None:
        """Hybrid config is valid when weights are valid and sum > 0."""
        if dense_weight + sparse_weight > 0:
            config = create_hybrid_config(
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            validate_hybrid_config(config)

    @given(
        corpus_size=st.integers(min_value=1, max_value=10000000),
        avg_doc_length=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=30)
    def test_recommended_config_always_valid(
        self, corpus_size: int, avg_doc_length: int
    ) -> None:
        """Recommended config is always valid."""
        config = get_recommended_hybrid_config(
            corpus_size=corpus_size,
            avg_doc_length=avg_doc_length,
        )
        validate_hybrid_config(config)
