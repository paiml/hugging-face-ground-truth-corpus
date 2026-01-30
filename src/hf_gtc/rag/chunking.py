"""Document chunking strategies for RAG systems.

This module provides comprehensive document chunking utilities for
RAG (Retrieval-Augmented Generation) pipelines, including multiple
chunking strategies, overlap handling, and quality estimation.

Examples:
    >>> from hf_gtc.rag.chunking import create_chunk_config
    >>> config = create_chunk_config(chunk_size=512, overlap_size=50)
    >>> config.chunk_size
    512
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ChunkingStrategy(Enum):
    """Supported document chunking strategies.

    Attributes:
        FIXED: Fixed-size character chunks.
        SENTENCE: Sentence-based chunking.
        PARAGRAPH: Paragraph-based chunking.
        SEMANTIC: Semantic similarity-based chunking.
        RECURSIVE: Recursive character text splitter.
        MARKDOWN: Markdown-aware chunking.

    Examples:
        >>> ChunkingStrategy.FIXED.value
        'fixed'
        >>> ChunkingStrategy.SEMANTIC.value
        'semantic'
        >>> ChunkingStrategy.MARKDOWN.value
        'markdown'
    """

    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"


VALID_CHUNKING_STRATEGIES = frozenset(s.value for s in ChunkingStrategy)


class OverlapType(Enum):
    """Types of overlap between chunks.

    Attributes:
        TOKEN: Token-based overlap.
        CHARACTER: Character-based overlap.
        SENTENCE: Sentence-based overlap.
        NONE: No overlap between chunks.

    Examples:
        >>> OverlapType.TOKEN.value
        'token'
        >>> OverlapType.CHARACTER.value
        'character'
        >>> OverlapType.NONE.value
        'none'
    """

    TOKEN = "token"
    CHARACTER = "character"
    SENTENCE = "sentence"
    NONE = "none"


VALID_OVERLAP_TYPES = frozenset(t.value for t in OverlapType)


class BoundaryDetection(Enum):
    """Methods for detecting chunk boundaries.

    Attributes:
        REGEX: Regular expression-based boundary detection.
        SPACY: spaCy NLP-based boundary detection.
        NLTK: NLTK-based boundary detection.
        SIMPLE: Simple rule-based boundary detection.

    Examples:
        >>> BoundaryDetection.REGEX.value
        'regex'
        >>> BoundaryDetection.SPACY.value
        'spacy'
        >>> BoundaryDetection.SIMPLE.value
        'simple'
    """

    REGEX = "regex"
    SPACY = "spacy"
    NLTK = "nltk"
    SIMPLE = "simple"


VALID_BOUNDARY_DETECTIONS = frozenset(b.value for b in BoundaryDetection)


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Configuration for document chunking.

    Attributes:
        strategy: Chunking strategy to use.
        chunk_size: Maximum chunk size in characters.
        overlap_size: Overlap between chunks in characters.
        overlap_type: Type of overlap between chunks.

    Examples:
        >>> config = ChunkConfig(
        ...     strategy=ChunkingStrategy.RECURSIVE,
        ...     chunk_size=512,
        ...     overlap_size=50,
        ...     overlap_type=OverlapType.CHARACTER,
        ... )
        >>> config.chunk_size
        512
    """

    strategy: ChunkingStrategy
    chunk_size: int
    overlap_size: int
    overlap_type: OverlapType


@dataclass(frozen=True, slots=True)
class SemanticChunkConfig:
    """Configuration for semantic chunking.

    Attributes:
        similarity_threshold: Minimum similarity to keep chunks together.
        min_chunk_size: Minimum chunk size in characters.
        embedding_model: Name of the embedding model to use.

    Examples:
        >>> config = SemanticChunkConfig(
        ...     similarity_threshold=0.8,
        ...     min_chunk_size=100,
        ...     embedding_model="all-MiniLM-L6-v2",
        ... )
        >>> config.similarity_threshold
        0.8
    """

    similarity_threshold: float
    min_chunk_size: int
    embedding_model: str


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Result of chunking a document.

    Attributes:
        chunks: Tuple of chunk text content.
        offsets: Tuple of (start, end) character positions.
        metadata: Additional metadata about chunking.
        chunk_count: Total number of chunks.

    Examples:
        >>> result = ChunkResult(
        ...     chunks=("chunk1", "chunk2"),
        ...     offsets=((0, 10), (8, 20)),
        ...     metadata={"strategy": "recursive"},
        ...     chunk_count=2,
        ... )
        >>> result.chunk_count
        2
    """

    chunks: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]
    metadata: dict[str, Any]
    chunk_count: int


def validate_chunk_config(config: ChunkConfig) -> None:
    """Validate chunk configuration.

    Args:
        config: Chunk configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ChunkConfig(
        ...     ChunkingStrategy.FIXED, 512, 50, OverlapType.CHARACTER
        ... )
        >>> validate_chunk_config(config)  # No error

        >>> bad = ChunkConfig(ChunkingStrategy.FIXED, 0, 50, OverlapType.CHARACTER)
        >>> validate_chunk_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if config.chunk_size <= 0:
        msg = f"chunk_size must be positive, got {config.chunk_size}"
        raise ValueError(msg)

    if config.overlap_size < 0:
        msg = f"overlap_size must be non-negative, got {config.overlap_size}"
        raise ValueError(msg)

    overlap_exceeds = config.overlap_size >= config.chunk_size
    if config.overlap_type != OverlapType.NONE and overlap_exceeds:
        msg = (
            f"overlap_size ({config.overlap_size}) must be less than "
            f"chunk_size ({config.chunk_size})"
        )
        raise ValueError(msg)


def validate_semantic_chunk_config(config: SemanticChunkConfig) -> None:
    """Validate semantic chunk configuration.

    Args:
        config: Semantic chunk configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SemanticChunkConfig(0.8, 100, "all-MiniLM-L6-v2")
        >>> validate_semantic_chunk_config(config)  # No error

        >>> bad = SemanticChunkConfig(1.5, 100, "all-MiniLM-L6-v2")
        >>> validate_semantic_chunk_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: similarity_threshold must be between 0.0 and 1.0
    """
    if not 0.0 <= config.similarity_threshold <= 1.0:
        msg = (
            f"similarity_threshold must be between 0.0 and 1.0, "
            f"got {config.similarity_threshold}"
        )
        raise ValueError(msg)

    if config.min_chunk_size <= 0:
        msg = f"min_chunk_size must be positive, got {config.min_chunk_size}"
        raise ValueError(msg)

    if not config.embedding_model:
        msg = "embedding_model cannot be empty"
        raise ValueError(msg)


def validate_chunk_result(result: ChunkResult) -> None:
    """Validate chunk result.

    Args:
        result: Chunk result to validate.

    Raises:
        ValueError: If result is invalid.

    Examples:
        >>> result = ChunkResult(("a", "b"), ((0, 1), (1, 2)), {}, 2)
        >>> validate_chunk_result(result)  # No error

        >>> bad = ChunkResult(("a", "b"), ((0, 1),), {}, 2)
        >>> validate_chunk_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunks and offsets must have the same length
    """
    if len(result.chunks) != len(result.offsets):
        msg = (
            f"chunks and offsets must have the same length, "
            f"got {len(result.chunks)} and {len(result.offsets)}"
        )
        raise ValueError(msg)

    if result.chunk_count != len(result.chunks):
        msg = (
            f"chunk_count ({result.chunk_count}) must match "
            f"number of chunks ({len(result.chunks)})"
        )
        raise ValueError(msg)

    for i, (start, end) in enumerate(result.offsets):
        if start < 0:
            msg = f"offset start at index {i} must be non-negative, got {start}"
            raise ValueError(msg)
        if end < start:
            msg = f"offset end ({end}) must be >= start ({start}) at index {i}"
            raise ValueError(msg)


def create_chunk_config(
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap_size: int = 50,
    overlap_type: str = "character",
) -> ChunkConfig:
    """Create a chunk configuration.

    Args:
        strategy: Chunking strategy. Defaults to "recursive".
        chunk_size: Maximum chunk size. Defaults to 512.
        overlap_size: Overlap between chunks. Defaults to 50.
        overlap_type: Type of overlap. Defaults to "character".

    Returns:
        ChunkConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_chunk_config(chunk_size=256, overlap_size=25)
        >>> config.chunk_size
        256

        >>> config = create_chunk_config(strategy="sentence")
        >>> config.strategy
        <ChunkingStrategy.SENTENCE: 'sentence'>

        >>> create_chunk_config(chunk_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if strategy not in VALID_CHUNKING_STRATEGIES:
        msg = f"strategy must be one of {VALID_CHUNKING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    if overlap_type not in VALID_OVERLAP_TYPES:
        msg = f"overlap_type must be one of {VALID_OVERLAP_TYPES}, got '{overlap_type}'"
        raise ValueError(msg)

    config = ChunkConfig(
        strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        overlap_type=OverlapType(overlap_type),
    )
    validate_chunk_config(config)
    return config


def create_semantic_chunk_config(
    similarity_threshold: float = 0.8,
    min_chunk_size: int = 100,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> SemanticChunkConfig:
    """Create a semantic chunk configuration.

    Args:
        similarity_threshold: Minimum similarity. Defaults to 0.8.
        min_chunk_size: Minimum chunk size. Defaults to 100.
        embedding_model: Embedding model name. Defaults to "all-MiniLM-L6-v2".

    Returns:
        SemanticChunkConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_semantic_chunk_config(similarity_threshold=0.9)
        >>> config.similarity_threshold
        0.9

        >>> config = create_semantic_chunk_config(min_chunk_size=200)
        >>> config.min_chunk_size
        200

        >>> create_semantic_chunk_config(similarity_threshold=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: similarity_threshold must be between 0.0 and 1.0
    """
    config = SemanticChunkConfig(
        similarity_threshold=similarity_threshold,
        min_chunk_size=min_chunk_size,
        embedding_model=embedding_model,
    )
    validate_semantic_chunk_config(config)
    return config


def create_chunk_result(
    chunks: tuple[str, ...],
    offsets: tuple[tuple[int, int], ...],
    metadata: dict[str, Any] | None = None,
) -> ChunkResult:
    """Create a chunk result.

    Args:
        chunks: Tuple of chunk text content.
        offsets: Tuple of (start, end) character positions.
        metadata: Additional metadata. Defaults to None.

    Returns:
        ChunkResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_chunk_result(
        ...     chunks=("hello", "world"),
        ...     offsets=((0, 5), (6, 11)),
        ... )
        >>> result.chunk_count
        2

        >>> create_chunk_result(("a", "b"), ((0, 1),))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunks and offsets must have the same length
    """
    result = ChunkResult(
        chunks=chunks,
        offsets=offsets,
        metadata=metadata if metadata is not None else {},
        chunk_count=len(chunks),
    )
    validate_chunk_result(result)
    return result


def list_chunking_strategies() -> list[str]:
    """List supported chunking strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_chunking_strategies()
        >>> "recursive" in strategies
        True
        >>> "semantic" in strategies
        True
        >>> "markdown" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_CHUNKING_STRATEGIES)


def list_overlap_types() -> list[str]:
    """List supported overlap types.

    Returns:
        Sorted list of overlap type names.

    Examples:
        >>> types = list_overlap_types()
        >>> "token" in types
        True
        >>> "character" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_OVERLAP_TYPES)


def list_boundary_detections() -> list[str]:
    """List supported boundary detection methods.

    Returns:
        Sorted list of boundary detection names.

    Examples:
        >>> methods = list_boundary_detections()
        >>> "regex" in methods
        True
        >>> "spacy" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_BOUNDARY_DETECTIONS)


def get_chunking_strategy(name: str) -> ChunkingStrategy:
    """Get chunking strategy from name.

    Args:
        name: Strategy name.

    Returns:
        ChunkingStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_chunking_strategy("recursive")
        <ChunkingStrategy.RECURSIVE: 'recursive'>

        >>> get_chunking_strategy("markdown")
        <ChunkingStrategy.MARKDOWN: 'markdown'>

        >>> get_chunking_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if name not in VALID_CHUNKING_STRATEGIES:
        msg = f"strategy must be one of {VALID_CHUNKING_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return ChunkingStrategy(name)


def get_overlap_type(name: str) -> OverlapType:
    """Get overlap type from name.

    Args:
        name: Overlap type name.

    Returns:
        OverlapType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_overlap_type("token")
        <OverlapType.TOKEN: 'token'>

        >>> get_overlap_type("character")
        <OverlapType.CHARACTER: 'character'>

        >>> get_overlap_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overlap_type must be one of
    """
    if name not in VALID_OVERLAP_TYPES:
        msg = f"overlap_type must be one of {VALID_OVERLAP_TYPES}, got '{name}'"
        raise ValueError(msg)
    return OverlapType(name)


def get_boundary_detection(name: str) -> BoundaryDetection:
    """Get boundary detection method from name.

    Args:
        name: Boundary detection name.

    Returns:
        BoundaryDetection enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_boundary_detection("regex")
        <BoundaryDetection.REGEX: 'regex'>

        >>> get_boundary_detection("spacy")
        <BoundaryDetection.SPACY: 'spacy'>

        >>> get_boundary_detection("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: boundary_detection must be one of
    """
    if name not in VALID_BOUNDARY_DETECTIONS:
        msg = (
            f"boundary_detection must be one of {VALID_BOUNDARY_DETECTIONS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return BoundaryDetection(name)


def calculate_chunk_count(
    document_length: int,
    chunk_size: int,
    overlap_size: int,
) -> int:
    """Calculate number of chunks for a document.

    Args:
        document_length: Total document length in characters.
        chunk_size: Size of each chunk.
        overlap_size: Overlap between chunks.

    Returns:
        Estimated number of chunks.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_chunk_count(1000, 200, 50)
        7

        >>> calculate_chunk_count(100, 100, 0)
        1

        >>> calculate_chunk_count(0, 200, 50)
        0

        >>> calculate_chunk_count(500, 100, 0)
        5

        >>> calculate_chunk_count(1000, 0, 50)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    if overlap_size < 0:
        msg = f"overlap_size must be non-negative, got {overlap_size}"
        raise ValueError(msg)

    if overlap_size >= chunk_size:
        msg = (
            f"overlap_size ({overlap_size}) must be less than "
            f"chunk_size ({chunk_size})"
        )
        raise ValueError(msg)

    if document_length <= 0:
        return 0

    if document_length <= chunk_size:
        return 1

    step = chunk_size - overlap_size
    return (document_length - overlap_size + step - 1) // step


def estimate_retrieval_quality(
    chunk_size: int,
    overlap_size: int,
    query_complexity: str = "medium",
) -> float:
    """Estimate retrieval quality for chunk configuration.

    Args:
        chunk_size: Size of each chunk.
        overlap_size: Overlap between chunks.
        query_complexity: Query complexity ("simple", "medium", "complex").

    Returns:
        Estimated quality score between 0.0 and 1.0.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> quality = estimate_retrieval_quality(512, 50)
        >>> 0.0 <= quality <= 1.0
        True

        >>> quality_simple = estimate_retrieval_quality(512, 50, "simple")
        >>> quality_complex = estimate_retrieval_quality(512, 50, "complex")
        >>> quality_simple >= quality_complex
        True

        >>> estimate_retrieval_quality(0, 50)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    if overlap_size < 0:
        msg = f"overlap_size must be non-negative, got {overlap_size}"
        raise ValueError(msg)

    valid_complexities = {"simple", "medium", "complex"}
    if query_complexity not in valid_complexities:
        msg = (
            f"query_complexity must be one of {valid_complexities}, "
            f"got '{query_complexity}'"
        )
        raise ValueError(msg)

    # Base quality from chunk size (optimal around 256-512)
    if chunk_size < 128:
        size_factor = 0.6
    elif chunk_size < 256:
        size_factor = 0.8
    elif chunk_size <= 512:
        size_factor = 1.0
    elif chunk_size <= 1024:
        size_factor = 0.9
    else:
        size_factor = 0.7

    # Overlap improves quality (diminishing returns)
    overlap_ratio = min(overlap_size / chunk_size, 0.5) if chunk_size > 0 else 0
    overlap_factor = 1.0 + overlap_ratio * 0.2

    # Complexity adjustment
    complexity_factors = {
        "simple": 1.0,
        "medium": 0.9,
        "complex": 0.8,
    }
    complexity_factor = complexity_factors[query_complexity]

    quality = size_factor * overlap_factor * complexity_factor
    return min(1.0, max(0.0, quality))


def optimize_chunk_size(
    avg_query_length: int,
    document_type: str = "general",
    embedding_dimension: int = 768,
) -> int:
    """Optimize chunk size for use case.

    Args:
        avg_query_length: Average query length in characters.
        document_type: Document type ("general", "code", "legal", "scientific").
        embedding_dimension: Embedding dimension. Defaults to 768.

    Returns:
        Recommended chunk size in characters.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> size = optimize_chunk_size(50)
        >>> 128 <= size <= 1024
        True

        >>> size_code = optimize_chunk_size(50, document_type="code")
        >>> size_general = optimize_chunk_size(50, document_type="general")
        >>> size_code != size_general  # Different sizes for different types
        True

        >>> optimize_chunk_size(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: avg_query_length must be positive
    """
    if avg_query_length <= 0:
        msg = f"avg_query_length must be positive, got {avg_query_length}"
        raise ValueError(msg)

    valid_types = {"general", "code", "legal", "scientific"}
    if document_type not in valid_types:
        msg = f"document_type must be one of {valid_types}, got '{document_type}'"
        raise ValueError(msg)

    if embedding_dimension <= 0:
        msg = f"embedding_dimension must be positive, got {embedding_dimension}"
        raise ValueError(msg)

    # Base size on query length (chunks should contain enough context)
    base_size = avg_query_length * 8

    # Adjust for document type
    type_multipliers = {
        "general": 1.0,
        "code": 1.5,  # Code needs more context
        "legal": 1.3,  # Legal documents have long sentences
        "scientific": 1.2,  # Scientific papers have complex content
    }
    multiplier = type_multipliers[document_type]

    # Adjust for embedding dimension (higher dim can handle more content)
    dim_factor = embedding_dimension / 768.0

    optimal_size = int(base_size * multiplier * dim_factor)

    # Clamp to reasonable range
    return max(128, min(optimal_size, 2048))


def validate_chunk_boundaries(
    chunks: tuple[str, ...],
    offsets: tuple[tuple[int, int], ...],
    original_text: str,
) -> bool:
    """Validate that chunks match original text at offsets.

    Args:
        chunks: Tuple of chunk text content.
        offsets: Tuple of (start, end) character positions.
        original_text: Original document text.

    Returns:
        True if all chunks match, False otherwise.

    Raises:
        ValueError: If chunks and offsets have different lengths.

    Examples:
        >>> text = "Hello world, this is a test."
        >>> chunks = ("Hello", "world")
        >>> offsets = ((0, 5), (6, 11))
        >>> validate_chunk_boundaries(chunks, offsets, text)
        True

        >>> bad_offsets = ((0, 5), (7, 12))
        >>> validate_chunk_boundaries(chunks, bad_offsets, text)
        False

        >>> validate_chunk_boundaries(("a",), ((0, 1), (1, 2)), "ab")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunks and offsets must have the same length
    """
    if len(chunks) != len(offsets):
        msg = (
            f"chunks and offsets must have the same length, "
            f"got {len(chunks)} and {len(offsets)}"
        )
        raise ValueError(msg)

    for chunk, (start, end) in zip(chunks, offsets, strict=True):
        if start < 0 or end > len(original_text):
            return False
        if original_text[start:end] != chunk:
            return False

    return True


def format_chunk_stats(
    chunk_count: int,
    avg_chunk_size: float,
    total_overlap: int,
    strategy: str,
) -> str:
    """Format chunk statistics for display.

    Args:
        chunk_count: Number of chunks.
        avg_chunk_size: Average chunk size.
        total_overlap: Total overlap characters.
        strategy: Chunking strategy used.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = format_chunk_stats(10, 256.5, 500, "recursive")
        >>> "10 chunks" in stats
        True
        >>> "256.5" in stats
        True
        >>> "recursive" in stats
        True
    """
    return (
        f"Chunking Stats: {chunk_count} chunks | "
        f"avg_size={avg_chunk_size:.1f} chars | "
        f"total_overlap={total_overlap} chars | "
        f"strategy={strategy}"
    )


def get_recommended_chunking_config(
    use_case: str = "qa",
    document_type: str = "general",
) -> ChunkConfig:
    """Get recommended chunking configuration for use case.

    Args:
        use_case: Use case ("qa", "summarization", "search", "chat").
        document_type: Document type ("general", "code", "legal", "scientific").

    Returns:
        Recommended ChunkConfig for the use case.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_chunking_config()
        >>> config.strategy == ChunkingStrategy.RECURSIVE
        True

        >>> config = get_recommended_chunking_config(use_case="search")
        >>> config.chunk_size < 512  # Smaller chunks for search
        True

        >>> config = get_recommended_chunking_config(document_type="code")
        >>> config.strategy
        <ChunkingStrategy.RECURSIVE: 'recursive'>

        >>> get_recommended_chunking_config(use_case="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case must be one of
    """
    valid_use_cases = {"qa", "summarization", "search", "chat"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    valid_types = {"general", "code", "legal", "scientific"}
    if document_type not in valid_types:
        msg = f"document_type must be one of {valid_types}, got '{document_type}'"
        raise ValueError(msg)

    # Base configurations by use case
    use_case_configs: dict[str, dict[str, int | str]] = {
        "qa": {"chunk_size": 512, "overlap_size": 50, "strategy": "recursive"},
        "summarization": {
            "chunk_size": 1024,
            "overlap_size": 100,
            "strategy": "paragraph",
        },
        "search": {"chunk_size": 256, "overlap_size": 25, "strategy": "sentence"},
        "chat": {"chunk_size": 384, "overlap_size": 40, "strategy": "recursive"},
    }

    base = use_case_configs[use_case]

    # Adjust for document type
    type_adjustments = {
        "general": {"size_mult": 1.0, "strategy": base["strategy"]},
        "code": {"size_mult": 1.5, "strategy": "recursive"},
        "legal": {"size_mult": 1.3, "strategy": "paragraph"},
        "scientific": {"size_mult": 1.2, "strategy": "paragraph"},
    }

    adj = type_adjustments[document_type]
    chunk_size = int(base["chunk_size"] * adj["size_mult"])
    overlap_size = int(base["overlap_size"] * adj["size_mult"])
    strategy = adj["strategy"]

    return ChunkConfig(
        strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        overlap_type=OverlapType.CHARACTER,
    )
