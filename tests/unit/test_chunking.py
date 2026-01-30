"""Tests for rag.chunking module."""

from __future__ import annotations

import pytest

from hf_gtc.rag.chunking import (
    VALID_BOUNDARY_DETECTIONS,
    VALID_CHUNKING_STRATEGIES,
    VALID_OVERLAP_TYPES,
    BoundaryDetection,
    ChunkConfig,
    ChunkingStrategy,
    ChunkResult,
    OverlapType,
    SemanticChunkConfig,
    calculate_chunk_count,
    create_chunk_config,
    create_chunk_result,
    create_semantic_chunk_config,
    estimate_retrieval_quality,
    format_chunk_stats,
    get_boundary_detection,
    get_chunking_strategy,
    get_overlap_type,
    get_recommended_chunking_config,
    list_boundary_detections,
    list_chunking_strategies,
    list_overlap_types,
    optimize_chunk_size,
    validate_chunk_boundaries,
    validate_chunk_config,
    validate_chunk_result,
    validate_semantic_chunk_config,
)


class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ChunkingStrategy:
            assert isinstance(strategy.value, str)

    def test_fixed_value(self) -> None:
        """Fixed has correct value."""
        assert ChunkingStrategy.FIXED.value == "fixed"

    def test_sentence_value(self) -> None:
        """Sentence has correct value."""
        assert ChunkingStrategy.SENTENCE.value == "sentence"

    def test_paragraph_value(self) -> None:
        """Paragraph has correct value."""
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"

    def test_semantic_value(self) -> None:
        """Semantic has correct value."""
        assert ChunkingStrategy.SEMANTIC.value == "semantic"

    def test_recursive_value(self) -> None:
        """Recursive has correct value."""
        assert ChunkingStrategy.RECURSIVE.value == "recursive"

    def test_markdown_value(self) -> None:
        """Markdown has correct value."""
        assert ChunkingStrategy.MARKDOWN.value == "markdown"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_CHUNKING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_CHUNKING_STRATEGIES, frozenset)

    def test_valid_strategies_contains_all(self) -> None:
        """VALID_CHUNKING_STRATEGIES contains all enum values."""
        for strategy in ChunkingStrategy:
            assert strategy.value in VALID_CHUNKING_STRATEGIES


class TestOverlapType:
    """Tests for OverlapType enum."""

    def test_all_types_have_values(self) -> None:
        """All overlap types have string values."""
        for overlap_type in OverlapType:
            assert isinstance(overlap_type.value, str)

    def test_token_value(self) -> None:
        """Token has correct value."""
        assert OverlapType.TOKEN.value == "token"

    def test_character_value(self) -> None:
        """Character has correct value."""
        assert OverlapType.CHARACTER.value == "character"

    def test_sentence_value(self) -> None:
        """Sentence has correct value."""
        assert OverlapType.SENTENCE.value == "sentence"

    def test_none_value(self) -> None:
        """None has correct value."""
        assert OverlapType.NONE.value == "none"

    def test_valid_types_frozenset(self) -> None:
        """VALID_OVERLAP_TYPES is a frozenset."""
        assert isinstance(VALID_OVERLAP_TYPES, frozenset)

    def test_valid_types_contains_all(self) -> None:
        """VALID_OVERLAP_TYPES contains all enum values."""
        for overlap_type in OverlapType:
            assert overlap_type.value in VALID_OVERLAP_TYPES


class TestBoundaryDetection:
    """Tests for BoundaryDetection enum."""

    def test_all_methods_have_values(self) -> None:
        """All boundary detection methods have string values."""
        for method in BoundaryDetection:
            assert isinstance(method.value, str)

    def test_regex_value(self) -> None:
        """Regex has correct value."""
        assert BoundaryDetection.REGEX.value == "regex"

    def test_spacy_value(self) -> None:
        """SpaCy has correct value."""
        assert BoundaryDetection.SPACY.value == "spacy"

    def test_nltk_value(self) -> None:
        """NLTK has correct value."""
        assert BoundaryDetection.NLTK.value == "nltk"

    def test_simple_value(self) -> None:
        """Simple has correct value."""
        assert BoundaryDetection.SIMPLE.value == "simple"

    def test_valid_detections_frozenset(self) -> None:
        """VALID_BOUNDARY_DETECTIONS is a frozenset."""
        assert isinstance(VALID_BOUNDARY_DETECTIONS, frozenset)

    def test_valid_detections_contains_all(self) -> None:
        """VALID_BOUNDARY_DETECTIONS contains all enum values."""
        for method in BoundaryDetection:
            assert method.value in VALID_BOUNDARY_DETECTIONS


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_create_config(self) -> None:
        """Create chunk config."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=512,
            overlap_size=50,
            overlap_type=OverlapType.CHARACTER,
        )
        assert config.chunk_size == 512

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 512, 50, OverlapType.CHARACTER
        )
        with pytest.raises(AttributeError):
            config.chunk_size = 256  # type: ignore[misc]

    def test_config_has_slots(self) -> None:
        """Config uses slots for memory efficiency."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 512, 50, OverlapType.CHARACTER
        )
        assert hasattr(config, "__slots__") or not hasattr(config, "__dict__")


class TestSemanticChunkConfig:
    """Tests for SemanticChunkConfig dataclass."""

    def test_create_config(self) -> None:
        """Create semantic chunk config."""
        config = SemanticChunkConfig(
            similarity_threshold=0.8,
            min_chunk_size=100,
            embedding_model="all-MiniLM-L6-v2",
        )
        assert config.similarity_threshold == 0.8

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SemanticChunkConfig(0.8, 100, "all-MiniLM-L6-v2")
        with pytest.raises(AttributeError):
            config.similarity_threshold = 0.9  # type: ignore[misc]


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_create_result(self) -> None:
        """Create chunk result."""
        result = ChunkResult(
            chunks=("chunk1", "chunk2"),
            offsets=((0, 10), (8, 20)),
            metadata={"strategy": "recursive"},
            chunk_count=2,
        )
        assert result.chunk_count == 2

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = ChunkResult(
            chunks=("a",),
            offsets=((0, 1),),
            metadata={},
            chunk_count=1,
        )
        with pytest.raises(AttributeError):
            result.chunk_count = 2  # type: ignore[misc]


class TestValidateChunkConfig:
    """Tests for validate_chunk_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 512, 50, OverlapType.CHARACTER
        )
        validate_chunk_config(config)  # Should not raise

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk_size raises ValueError."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 0, 50, OverlapType.CHARACTER
        )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_chunk_config(config)

    def test_negative_chunk_size_raises(self) -> None:
        """Negative chunk_size raises ValueError."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, -100, 50, OverlapType.CHARACTER
        )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_chunk_config(config)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap_size raises ValueError."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 512, -10, OverlapType.CHARACTER
        )
        with pytest.raises(ValueError, match="overlap_size must be non-negative"):
            validate_chunk_config(config)

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        """Overlap >= chunk_size raises ValueError."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 100, 150, OverlapType.CHARACTER
        )
        with pytest.raises(ValueError, match=r"overlap_size.*must be less than"):
            validate_chunk_config(config)

    def test_overlap_equal_chunk_size_raises(self) -> None:
        """Overlap equal to chunk_size raises ValueError."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 100, 100, OverlapType.CHARACTER
        )
        with pytest.raises(ValueError, match=r"overlap_size.*must be less than"):
            validate_chunk_config(config)

    def test_no_overlap_type_none_valid(self) -> None:
        """Overlap type NONE with any overlap_size is valid."""
        config = ChunkConfig(
            ChunkingStrategy.FIXED, 100, 200, OverlapType.NONE
        )
        validate_chunk_config(config)  # Should not raise


class TestValidateSemanticChunkConfig:
    """Tests for validate_semantic_chunk_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SemanticChunkConfig(0.8, 100, "all-MiniLM-L6-v2")
        validate_semantic_chunk_config(config)  # Should not raise

    def test_threshold_below_zero_raises(self) -> None:
        """Threshold below 0 raises ValueError."""
        config = SemanticChunkConfig(-0.1, 100, "all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_semantic_chunk_config(config)

    def test_threshold_above_one_raises(self) -> None:
        """Threshold above 1 raises ValueError."""
        config = SemanticChunkConfig(1.5, 100, "all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            validate_semantic_chunk_config(config)

    def test_zero_min_chunk_size_raises(self) -> None:
        """Zero min_chunk_size raises ValueError."""
        config = SemanticChunkConfig(0.8, 0, "all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            validate_semantic_chunk_config(config)

    def test_empty_embedding_model_raises(self) -> None:
        """Empty embedding_model raises ValueError."""
        config = SemanticChunkConfig(0.8, 100, "")
        with pytest.raises(ValueError, match="embedding_model cannot be empty"):
            validate_semantic_chunk_config(config)


class TestValidateChunkResult:
    """Tests for validate_chunk_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes validation."""
        result = ChunkResult(("a", "b"), ((0, 1), (1, 2)), {}, 2)
        validate_chunk_result(result)  # Should not raise

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched chunks and offsets raises ValueError."""
        result = ChunkResult(("a", "b"), ((0, 1),), {}, 2)
        with pytest.raises(ValueError, match="chunks and offsets must have"):
            validate_chunk_result(result)

    def test_wrong_chunk_count_raises(self) -> None:
        """Wrong chunk_count raises ValueError."""
        result = ChunkResult(("a", "b"), ((0, 1), (1, 2)), {}, 3)
        with pytest.raises(ValueError, match=r"chunk_count.*must match"):
            validate_chunk_result(result)

    def test_negative_offset_start_raises(self) -> None:
        """Negative offset start raises ValueError."""
        result = ChunkResult(("a",), ((-1, 1),), {}, 1)
        with pytest.raises(ValueError, match=r"offset start.*must be non-negative"):
            validate_chunk_result(result)

    def test_end_less_than_start_raises(self) -> None:
        """End < start raises ValueError."""
        result = ChunkResult(("a",), ((5, 2),), {}, 1)
        with pytest.raises(ValueError, match=r"offset end.*must be >= start"):
            validate_chunk_result(result)


class TestCreateChunkConfig:
    """Tests for create_chunk_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_chunk_config()
        assert config.strategy == ChunkingStrategy.RECURSIVE
        assert config.chunk_size == 512
        assert config.overlap_size == 50
        assert config.overlap_type == OverlapType.CHARACTER

    def test_custom_chunk_size(self) -> None:
        """Create config with custom chunk_size."""
        config = create_chunk_config(chunk_size=256)
        assert config.chunk_size == 256

    def test_custom_overlap_size(self) -> None:
        """Create config with custom overlap_size."""
        config = create_chunk_config(overlap_size=25)
        assert config.overlap_size == 25

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("fixed", ChunkingStrategy.FIXED),
            ("sentence", ChunkingStrategy.SENTENCE),
            ("paragraph", ChunkingStrategy.PARAGRAPH),
            ("semantic", ChunkingStrategy.SEMANTIC),
            ("recursive", ChunkingStrategy.RECURSIVE),
            ("markdown", ChunkingStrategy.MARKDOWN),
        ],
    )
    def test_all_strategies(self, strategy: str, expected: ChunkingStrategy) -> None:
        """Create config with all strategies."""
        config = create_chunk_config(strategy=strategy)
        assert config.strategy == expected

    @pytest.mark.parametrize(
        ("overlap_type", "expected"),
        [
            ("token", OverlapType.TOKEN),
            ("character", OverlapType.CHARACTER),
            ("sentence", OverlapType.SENTENCE),
            ("none", OverlapType.NONE),
        ],
    )
    def test_all_overlap_types(self, overlap_type: str, expected: OverlapType) -> None:
        """Create config with all overlap types."""
        config = create_chunk_config(overlap_type=overlap_type)
        assert config.overlap_type == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_chunk_config(strategy="invalid")

    def test_invalid_overlap_type_raises(self) -> None:
        """Invalid overlap_type raises ValueError."""
        with pytest.raises(ValueError, match="overlap_type must be one of"):
            create_chunk_config(overlap_type="invalid")

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_chunk_config(chunk_size=0)


class TestCreateSemanticChunkConfig:
    """Tests for create_semantic_chunk_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_semantic_chunk_config()
        assert config.similarity_threshold == 0.8
        assert config.min_chunk_size == 100
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_custom_threshold(self) -> None:
        """Create config with custom threshold."""
        config = create_semantic_chunk_config(similarity_threshold=0.9)
        assert config.similarity_threshold == 0.9

    def test_custom_min_chunk_size(self) -> None:
        """Create config with custom min_chunk_size."""
        config = create_semantic_chunk_config(min_chunk_size=200)
        assert config.min_chunk_size == 200

    def test_custom_embedding_model(self) -> None:
        """Create config with custom embedding_model."""
        config = create_semantic_chunk_config(embedding_model="custom-model")
        assert config.embedding_model == "custom-model"

    def test_invalid_threshold_raises(self) -> None:
        """Invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            create_semantic_chunk_config(similarity_threshold=1.5)

    def test_zero_min_chunk_size_raises(self) -> None:
        """Zero min_chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            create_semantic_chunk_config(min_chunk_size=0)

    def test_empty_embedding_model_raises(self) -> None:
        """Empty embedding_model raises ValueError."""
        with pytest.raises(ValueError, match="embedding_model cannot be empty"):
            create_semantic_chunk_config(embedding_model="")


class TestCreateChunkResult:
    """Tests for create_chunk_result function."""

    def test_create_result(self) -> None:
        """Create chunk result."""
        result = create_chunk_result(
            chunks=("hello", "world"),
            offsets=((0, 5), (6, 11)),
        )
        assert result.chunk_count == 2

    def test_with_metadata(self) -> None:
        """Create result with metadata."""
        result = create_chunk_result(
            chunks=("a",),
            offsets=((0, 1),),
            metadata={"key": "value"},
        )
        assert result.metadata == {"key": "value"}

    def test_default_metadata(self) -> None:
        """Default metadata is empty dict."""
        result = create_chunk_result(
            chunks=("a",),
            offsets=((0, 1),),
        )
        assert result.metadata == {}

    def test_empty_chunks(self) -> None:
        """Create result with empty chunks."""
        result = create_chunk_result(chunks=(), offsets=())
        assert result.chunk_count == 0

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched chunks and offsets raises ValueError."""
        with pytest.raises(ValueError, match="chunks and offsets must have"):
            create_chunk_result(("a", "b"), ((0, 1),))


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_chunking_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_chunking_strategies()
        assert strategies == sorted(strategies)

    def test_list_chunking_strategies_contains_all(self) -> None:
        """Contains all enum values."""
        strategies = list_chunking_strategies()
        for strategy in ChunkingStrategy:
            assert strategy.value in strategies

    def test_list_overlap_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_overlap_types()
        assert types == sorted(types)

    def test_list_overlap_types_contains_all(self) -> None:
        """Contains all enum values."""
        types = list_overlap_types()
        for overlap_type in OverlapType:
            assert overlap_type.value in types

    def test_list_boundary_detections_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_boundary_detections()
        assert methods == sorted(methods)

    def test_list_boundary_detections_contains_all(self) -> None:
        """Contains all enum values."""
        methods = list_boundary_detections()
        for method in BoundaryDetection:
            assert method.value in methods


class TestGetChunkingStrategy:
    """Tests for get_chunking_strategy function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("fixed", ChunkingStrategy.FIXED),
            ("sentence", ChunkingStrategy.SENTENCE),
            ("paragraph", ChunkingStrategy.PARAGRAPH),
            ("semantic", ChunkingStrategy.SEMANTIC),
            ("recursive", ChunkingStrategy.RECURSIVE),
            ("markdown", ChunkingStrategy.MARKDOWN),
        ],
    )
    def test_get_all_strategies(self, name: str, expected: ChunkingStrategy) -> None:
        """Get all chunking strategies."""
        assert get_chunking_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_chunking_strategy("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_chunking_strategy("")


class TestGetOverlapType:
    """Tests for get_overlap_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("token", OverlapType.TOKEN),
            ("character", OverlapType.CHARACTER),
            ("sentence", OverlapType.SENTENCE),
            ("none", OverlapType.NONE),
        ],
    )
    def test_get_all_overlap_types(self, name: str, expected: OverlapType) -> None:
        """Get all overlap types."""
        assert get_overlap_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="overlap_type must be one of"):
            get_overlap_type("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="overlap_type must be one of"):
            get_overlap_type("")


class TestGetBoundaryDetection:
    """Tests for get_boundary_detection function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("regex", BoundaryDetection.REGEX),
            ("spacy", BoundaryDetection.SPACY),
            ("nltk", BoundaryDetection.NLTK),
            ("simple", BoundaryDetection.SIMPLE),
        ],
    )
    def test_get_all_boundary_detections(
        self, name: str, expected: BoundaryDetection
    ) -> None:
        """Get all boundary detection methods."""
        assert get_boundary_detection(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="boundary_detection must be one of"):
            get_boundary_detection("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="boundary_detection must be one of"):
            get_boundary_detection("")


class TestCalculateChunkCount:
    """Tests for calculate_chunk_count function."""

    def test_basic_calculation(self) -> None:
        """Basic chunk count calculation."""
        # 1000 chars, 200 chunk size, 50 overlap -> step=150
        # Chunks start at: 0, 150, 300, 450, 600, 750, 900 -> 7 chunks
        count = calculate_chunk_count(1000, 200, 50)
        assert count == 7

    def test_exact_fit(self) -> None:
        """Document fits exactly in one chunk."""
        count = calculate_chunk_count(100, 100, 0)
        assert count == 1

    def test_empty_document(self) -> None:
        """Empty document returns 0."""
        count = calculate_chunk_count(0, 200, 50)
        assert count == 0

    def test_negative_document_length(self) -> None:
        """Negative document length returns 0."""
        count = calculate_chunk_count(-100, 200, 50)
        assert count == 0

    def test_no_overlap(self) -> None:
        """Calculation with no overlap."""
        count = calculate_chunk_count(500, 100, 0)
        assert count == 5

    def test_small_document(self) -> None:
        """Document smaller than chunk size."""
        count = calculate_chunk_count(50, 100, 10)
        assert count == 1

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            calculate_chunk_count(1000, 0, 50)

    def test_negative_chunk_size_raises(self) -> None:
        """Negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            calculate_chunk_count(1000, -100, 50)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap_size must be non-negative"):
            calculate_chunk_count(1000, 200, -10)

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        """Overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match=r"overlap_size.*must be less than"):
            calculate_chunk_count(1000, 100, 150)


class TestEstimateRetrievalQuality:
    """Tests for estimate_retrieval_quality function."""

    def test_returns_valid_range(self) -> None:
        """Quality is between 0 and 1."""
        quality = estimate_retrieval_quality(512, 50)
        assert 0.0 <= quality <= 1.0

    def test_medium_complexity_default(self) -> None:
        """Default complexity is medium."""
        quality = estimate_retrieval_quality(512, 50)
        quality_medium = estimate_retrieval_quality(512, 50, "medium")
        assert quality == quality_medium

    def test_simple_higher_than_complex(self) -> None:
        """Simple queries have higher quality than complex."""
        quality_simple = estimate_retrieval_quality(512, 50, "simple")
        quality_complex = estimate_retrieval_quality(512, 50, "complex")
        assert quality_simple >= quality_complex

    def test_overlap_improves_quality(self) -> None:
        """Overlap improves retrieval quality."""
        quality_no_overlap = estimate_retrieval_quality(512, 0)
        quality_with_overlap = estimate_retrieval_quality(512, 50)
        assert quality_with_overlap >= quality_no_overlap

    def test_zero_chunk_size_raises(self) -> None:
        """Zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            estimate_retrieval_quality(0, 50)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap_size must be non-negative"):
            estimate_retrieval_quality(512, -10)

    def test_invalid_complexity_raises(self) -> None:
        """Invalid complexity raises ValueError."""
        with pytest.raises(ValueError, match="query_complexity must be one of"):
            estimate_retrieval_quality(512, 50, "invalid")

    @pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
    def test_all_complexities_valid(self, complexity: str) -> None:
        """All complexity values work."""
        quality = estimate_retrieval_quality(512, 50, complexity)
        assert 0.0 <= quality <= 1.0

    def test_very_small_chunk_size(self) -> None:
        """Very small chunk size gives lower quality."""
        quality = estimate_retrieval_quality(64, 10)
        assert 0.0 <= quality <= 1.0
        # Small chunks should have lower base quality
        assert quality < estimate_retrieval_quality(512, 50)

    def test_small_chunk_size(self) -> None:
        """Small chunk size (128-256) path."""
        quality = estimate_retrieval_quality(200, 20)
        assert 0.0 <= quality <= 1.0

    def test_large_chunk_size(self) -> None:
        """Large chunk size (512-1024) path."""
        quality = estimate_retrieval_quality(800, 80)
        assert 0.0 <= quality <= 1.0

    def test_very_large_chunk_size(self) -> None:
        """Very large chunk size (>1024) path."""
        quality = estimate_retrieval_quality(2000, 100)
        assert 0.0 <= quality <= 1.0
        # Very large chunks should have lower quality
        assert quality < estimate_retrieval_quality(512, 50)


class TestOptimizeChunkSize:
    """Tests for optimize_chunk_size function."""

    def test_returns_valid_range(self) -> None:
        """Chunk size is in valid range."""
        size = optimize_chunk_size(50)
        assert 128 <= size <= 2048

    def test_longer_queries_larger_chunks(self) -> None:
        """Longer queries suggest larger chunks."""
        size_short = optimize_chunk_size(20)
        size_long = optimize_chunk_size(100)
        assert size_long >= size_short

    def test_code_larger_than_general(self) -> None:
        """Code documents need larger chunks."""
        size_general = optimize_chunk_size(50, document_type="general")
        size_code = optimize_chunk_size(50, document_type="code")
        assert size_code > size_general

    def test_zero_query_length_raises(self) -> None:
        """Zero avg_query_length raises ValueError."""
        with pytest.raises(ValueError, match="avg_query_length must be positive"):
            optimize_chunk_size(0)

    def test_negative_query_length_raises(self) -> None:
        """Negative avg_query_length raises ValueError."""
        with pytest.raises(ValueError, match="avg_query_length must be positive"):
            optimize_chunk_size(-10)

    def test_invalid_document_type_raises(self) -> None:
        """Invalid document_type raises ValueError."""
        with pytest.raises(ValueError, match="document_type must be one of"):
            optimize_chunk_size(50, document_type="invalid")

    def test_zero_embedding_dimension_raises(self) -> None:
        """Zero embedding_dimension raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            optimize_chunk_size(50, embedding_dimension=0)

    @pytest.mark.parametrize("doc_type", ["general", "code", "legal", "scientific"])
    def test_all_document_types_valid(self, doc_type: str) -> None:
        """All document types work."""
        size = optimize_chunk_size(50, document_type=doc_type)
        assert 128 <= size <= 2048


class TestValidateChunkBoundaries:
    """Tests for validate_chunk_boundaries function."""

    def test_valid_boundaries(self) -> None:
        """Valid chunks and offsets pass."""
        text = "Hello world, this is a test."
        chunks = ("Hello", "world")
        offsets = ((0, 5), (6, 11))
        assert validate_chunk_boundaries(chunks, offsets, text) is True

    def test_invalid_offsets(self) -> None:
        """Invalid offsets return False."""
        text = "Hello world"
        chunks = ("Hello", "world")
        offsets = ((0, 5), (7, 12))  # "world" starts at 6, not 7
        assert validate_chunk_boundaries(chunks, offsets, text) is False

    def test_mismatched_content(self) -> None:
        """Mismatched chunk content returns False."""
        text = "Hello world"
        chunks = ("Hello", "WORLD")  # "WORLD" doesn't match "world"
        offsets = ((0, 5), (6, 11))
        assert validate_chunk_boundaries(chunks, offsets, text) is False

    def test_offset_out_of_range(self) -> None:
        """Offset beyond text length returns False."""
        text = "Hello"
        chunks = ("Hello",)
        offsets = ((0, 10),)  # End beyond text length
        assert validate_chunk_boundaries(chunks, offsets, text) is False

    def test_negative_offset(self) -> None:
        """Negative offset returns False."""
        text = "Hello"
        chunks = ("Hello",)
        offsets = ((-1, 5),)
        assert validate_chunk_boundaries(chunks, offsets, text) is False

    def test_empty_chunks_valid(self) -> None:
        """Empty chunks are valid."""
        assert validate_chunk_boundaries((), (), "Hello") is True

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="chunks and offsets must have"):
            validate_chunk_boundaries(("a",), ((0, 1), (1, 2)), "ab")


class TestFormatChunkStats:
    """Tests for format_chunk_stats function."""

    def test_basic_format(self) -> None:
        """Basic formatting."""
        stats = format_chunk_stats(10, 256.5, 500, "recursive")
        assert "10 chunks" in stats
        assert "256.5" in stats
        assert "500" in stats
        assert "recursive" in stats

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = format_chunk_stats(0, 0.0, 0, "fixed")
        assert "0 chunks" in stats
        assert "fixed" in stats

    def test_large_values(self) -> None:
        """Large values format correctly."""
        stats = format_chunk_stats(10000, 512.25, 50000, "semantic")
        assert "10000 chunks" in stats
        assert "semantic" in stats


class TestGetRecommendedChunkingConfig:
    """Tests for get_recommended_chunking_config function."""

    def test_default_config(self) -> None:
        """Default config is for QA."""
        config = get_recommended_chunking_config()
        assert config.strategy == ChunkingStrategy.RECURSIVE

    def test_qa_config(self) -> None:
        """QA config."""
        config = get_recommended_chunking_config(use_case="qa")
        assert config.chunk_size == 512
        assert config.strategy == ChunkingStrategy.RECURSIVE

    def test_search_config(self) -> None:
        """Search config has smaller chunks."""
        config = get_recommended_chunking_config(use_case="search")
        assert config.chunk_size < 512

    def test_summarization_config(self) -> None:
        """Summarization config has larger chunks."""
        config = get_recommended_chunking_config(use_case="summarization")
        assert config.chunk_size > 512

    def test_code_document_type(self) -> None:
        """Code documents have larger chunks."""
        config_general = get_recommended_chunking_config(document_type="general")
        config_code = get_recommended_chunking_config(document_type="code")
        assert config_code.chunk_size > config_general.chunk_size

    def test_invalid_use_case_raises(self) -> None:
        """Invalid use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_chunking_config(use_case="invalid")

    def test_invalid_document_type_raises(self) -> None:
        """Invalid document_type raises ValueError."""
        with pytest.raises(ValueError, match="document_type must be one of"):
            get_recommended_chunking_config(document_type="invalid")

    @pytest.mark.parametrize("use_case", ["qa", "summarization", "search", "chat"])
    def test_all_use_cases_valid(self, use_case: str) -> None:
        """All use cases work."""
        config = get_recommended_chunking_config(use_case=use_case)
        assert config.chunk_size > 0

    @pytest.mark.parametrize("doc_type", ["general", "code", "legal", "scientific"])
    def test_all_document_types_valid(self, doc_type: str) -> None:
        """All document types work."""
        config = get_recommended_chunking_config(document_type=doc_type)
        assert config.chunk_size > 0
