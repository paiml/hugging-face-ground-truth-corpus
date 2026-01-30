"""RAG retrieval utilities.

This module provides functions for document chunking, embedding retrieval,
and building RAG (Retrieval-Augmented Generation) pipelines.

Examples:
    >>> from hf_gtc.rag.retrieval import create_chunking_config
    >>> config = create_chunking_config(chunk_size=512, overlap=50)
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
        TOKEN: Token-based chunking.

    Examples:
        >>> ChunkingStrategy.FIXED.value
        'fixed'
        >>> ChunkingStrategy.SEMANTIC.value
        'semantic'
    """

    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    TOKEN = "token"


VALID_CHUNKING_STRATEGIES = frozenset(s.value for s in ChunkingStrategy)


class DistanceMetric(Enum):
    """Supported distance metrics for similarity search.

    Attributes:
        COSINE: Cosine similarity.
        EUCLIDEAN: Euclidean distance (L2).
        DOT_PRODUCT: Dot product similarity.
        MANHATTAN: Manhattan distance (L1).

    Examples:
        >>> DistanceMetric.COSINE.value
        'cosine'
        >>> DistanceMetric.EUCLIDEAN.value
        'euclidean'
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


VALID_DISTANCE_METRICS = frozenset(m.value for m in DistanceMetric)


class RetrievalMethod(Enum):
    """Supported retrieval methods.

    Attributes:
        DENSE: Dense embedding retrieval.
        SPARSE: Sparse retrieval (BM25, TF-IDF).
        HYBRID: Hybrid dense + sparse retrieval.
        RERANK: Two-stage retrieval with reranking.

    Examples:
        >>> RetrievalMethod.DENSE.value
        'dense'
        >>> RetrievalMethod.HYBRID.value
        'hybrid'
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANK = "rerank"


VALID_RETRIEVAL_METHODS = frozenset(m.value for m in RetrievalMethod)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    r"""Configuration for document chunking.

    Attributes:
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap between chunks in characters.
        strategy: Chunking strategy.
        separator: Text separator for splitting.

    Examples:
        >>> config = ChunkingConfig(
        ...     chunk_size=512,
        ...     overlap=50,
        ...     strategy=ChunkingStrategy.RECURSIVE,
        ...     separator="\\n\\n",
        ... )
        >>> config.chunk_size
        512
    """

    chunk_size: int
    overlap: int
    strategy: ChunkingStrategy
    separator: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Represents a chunk of a document.

    Attributes:
        content: Chunk text content.
        chunk_id: Unique chunk identifier.
        document_id: Source document identifier.
        start_char: Start character position.
        end_char: End character position.
        metadata: Additional metadata.

    Examples:
        >>> chunk = DocumentChunk(
        ...     content="Some text",
        ...     chunk_id="chunk_0",
        ...     document_id="doc_1",
        ...     start_char=0,
        ...     end_char=9,
        ...     metadata={},
        ... )
        >>> chunk.content
        'Some text'
    """

    content: str
    chunk_id: str
    document_id: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Result of chunking a document.

    Attributes:
        chunks: Tuple of document chunks.
        total_chunks: Total number of chunks.
        avg_chunk_size: Average chunk size.
        document_id: Source document ID.

    Examples:
        >>> result = ChunkResult(
        ...     chunks=(),
        ...     total_chunks=0,
        ...     avg_chunk_size=0.0,
        ...     document_id="doc_1",
        ... )
        >>> result.total_chunks
        0
    """

    chunks: tuple[DocumentChunk, ...]
    total_chunks: int
    avg_chunk_size: float
    document_id: str


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Configuration for RAG pipeline.

    Attributes:
        retrieval_method: Retrieval method to use.
        distance_metric: Distance metric for similarity.
        top_k: Number of documents to retrieve.
        score_threshold: Minimum similarity score.
        rerank_top_k: Documents to keep after reranking.

    Examples:
        >>> config = RAGConfig(
        ...     retrieval_method=RetrievalMethod.HYBRID,
        ...     distance_metric=DistanceMetric.COSINE,
        ...     top_k=10,
        ...     score_threshold=0.5,
        ...     rerank_top_k=5,
        ... )
        >>> config.top_k
        10
    """

    retrieval_method: RetrievalMethod
    distance_metric: DistanceMetric
    top_k: int
    score_threshold: float
    rerank_top_k: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Result from retrieval.

    Attributes:
        chunk: Retrieved document chunk.
        score: Similarity score.
        rank: Rank in results.

    Examples:
        >>> chunk = DocumentChunk("text", "c0", "d1", 0, 4, {})
        >>> result = RetrievalResult(chunk=chunk, score=0.95, rank=1)
        >>> result.score
        0.95
    """

    chunk: DocumentChunk
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RetrievalStats:
    """Statistics for retrieval operation.

    Attributes:
        total_retrieved: Number of documents retrieved.
        avg_score: Average similarity score.
        latency_ms: Retrieval latency in milliseconds.
        reranked: Whether results were reranked.

    Examples:
        >>> stats = RetrievalStats(
        ...     total_retrieved=10,
        ...     avg_score=0.85,
        ...     latency_ms=50.0,
        ...     reranked=True,
        ... )
        >>> stats.total_retrieved
        10
    """

    total_retrieved: int
    avg_score: float
    latency_ms: float
    reranked: bool


def validate_chunking_config(config: ChunkingConfig) -> None:
    r"""Validate chunking configuration.

    Args:
        config: Chunking configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ChunkingConfig(512, 50, ChunkingStrategy.FIXED, "\\n")
        >>> validate_chunking_config(config)  # No error

        >>> bad = ChunkingConfig(0, 50, ChunkingStrategy.FIXED, "\\n")
        >>> validate_chunking_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if config.chunk_size <= 0:
        msg = f"chunk_size must be positive, got {config.chunk_size}"
        raise ValueError(msg)

    if config.overlap < 0:
        msg = f"overlap must be non-negative, got {config.overlap}"
        raise ValueError(msg)

    if config.overlap >= config.chunk_size:
        msg = (
            f"overlap ({config.overlap}) must be less than "
            f"chunk_size ({config.chunk_size})"
        )
        raise ValueError(msg)


def validate_rag_config(config: RAGConfig) -> None:
    """Validate RAG configuration.

    Args:
        config: RAG configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = RAGConfig(
        ...     RetrievalMethod.DENSE, DistanceMetric.COSINE, 10, 0.5, 5
        ... )
        >>> validate_rag_config(config)  # No error

        >>> bad = RAGConfig(
        ...     RetrievalMethod.DENSE, DistanceMetric.COSINE, 0, 0.5, 5
        ... )
        >>> validate_rag_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    if config.top_k <= 0:
        msg = f"top_k must be positive, got {config.top_k}"
        raise ValueError(msg)

    if config.rerank_top_k <= 0:
        msg = f"rerank_top_k must be positive, got {config.rerank_top_k}"
        raise ValueError(msg)

    if config.rerank_top_k > config.top_k:
        msg = (
            f"rerank_top_k ({config.rerank_top_k}) cannot exceed top_k ({config.top_k})"
        )
        raise ValueError(msg)

    if not 0.0 <= config.score_threshold <= 1.0:
        msg = (
            f"score_threshold must be between 0.0 and 1.0, got {config.score_threshold}"
        )
        raise ValueError(msg)


def create_chunking_config(
    chunk_size: int = 512,
    overlap: int = 50,
    strategy: str = "recursive",
    separator: str = "\n\n",
) -> ChunkingConfig:
    r"""Create a chunking configuration.

    Args:
        chunk_size: Maximum chunk size. Defaults to 512.
        overlap: Chunk overlap. Defaults to 50.
        strategy: Chunking strategy. Defaults to "recursive".
        separator: Text separator. Defaults to "\\n\\n".

    Returns:
        ChunkingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_chunking_config(chunk_size=256, overlap=25)
        >>> config.chunk_size
        256

        >>> config = create_chunking_config(strategy="sentence")
        >>> config.strategy
        <ChunkingStrategy.SENTENCE: 'sentence'>

        >>> create_chunking_config(chunk_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if strategy not in VALID_CHUNKING_STRATEGIES:
        msg = f"strategy must be one of {VALID_CHUNKING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = ChunkingConfig(
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=ChunkingStrategy(strategy),
        separator=separator,
    )
    validate_chunking_config(config)
    return config


def create_document_chunk(
    content: str,
    chunk_id: str,
    document_id: str,
    start_char: int = 0,
    end_char: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    """Create a document chunk.

    Args:
        content: Chunk text content.
        chunk_id: Unique chunk identifier.
        document_id: Source document identifier.
        start_char: Start character position. Defaults to 0.
        end_char: End character position. Defaults to None (length of content).
        metadata: Additional metadata. Defaults to None.

    Returns:
        DocumentChunk with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> chunk = create_document_chunk("Hello world", "c0", "doc1")
        >>> chunk.content
        'Hello world'

        >>> chunk = create_document_chunk("Test", "c1", "doc1", start_char=10)
        >>> chunk.start_char
        10

        >>> create_document_chunk("", "c0", "doc1")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: content cannot be empty
    """
    if not content:
        msg = "content cannot be empty"
        raise ValueError(msg)

    if not chunk_id:
        msg = "chunk_id cannot be empty"
        raise ValueError(msg)

    if not document_id:
        msg = "document_id cannot be empty"
        raise ValueError(msg)

    if start_char < 0:
        msg = f"start_char must be non-negative, got {start_char}"
        raise ValueError(msg)

    actual_end = end_char if end_char is not None else start_char + len(content)

    return DocumentChunk(
        content=content,
        chunk_id=chunk_id,
        document_id=document_id,
        start_char=start_char,
        end_char=actual_end,
        metadata=metadata if metadata is not None else {},
    )


def create_rag_config(
    retrieval_method: str = "dense",
    distance_metric: str = "cosine",
    top_k: int = 10,
    score_threshold: float = 0.0,
    rerank_top_k: int = 5,
) -> RAGConfig:
    """Create a RAG configuration.

    Args:
        retrieval_method: Retrieval method. Defaults to "dense".
        distance_metric: Distance metric. Defaults to "cosine".
        top_k: Documents to retrieve. Defaults to 10.
        score_threshold: Minimum score. Defaults to 0.0.
        rerank_top_k: Documents after reranking. Defaults to 5.

    Returns:
        RAGConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_rag_config(top_k=20)
        >>> config.top_k
        20

        >>> config = create_rag_config(retrieval_method="hybrid")
        >>> config.retrieval_method
        <RetrievalMethod.HYBRID: 'hybrid'>

        >>> create_rag_config(top_k=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    if retrieval_method not in VALID_RETRIEVAL_METHODS:
        msg = (
            f"retrieval_method must be one of {VALID_RETRIEVAL_METHODS}, "
            f"got '{retrieval_method}'"
        )
        raise ValueError(msg)

    if distance_metric not in VALID_DISTANCE_METRICS:
        msg = (
            f"distance_metric must be one of {VALID_DISTANCE_METRICS}, "
            f"got '{distance_metric}'"
        )
        raise ValueError(msg)

    config = RAGConfig(
        retrieval_method=RetrievalMethod(retrieval_method),
        distance_metric=DistanceMetric(distance_metric),
        top_k=top_k,
        score_threshold=score_threshold,
        rerank_top_k=rerank_top_k,
    )
    validate_rag_config(config)
    return config


def create_retrieval_result(
    chunk: DocumentChunk,
    score: float,
    rank: int,
) -> RetrievalResult:
    """Create a retrieval result.

    Args:
        chunk: Retrieved document chunk.
        score: Similarity score.
        rank: Rank in results.

    Returns:
        RetrievalResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> chunk = create_document_chunk("text", "c0", "d1")
        >>> result = create_retrieval_result(chunk, score=0.9, rank=1)
        >>> result.score
        0.9

        >>> create_retrieval_result(chunk, score=1.5, rank=1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: score must be between 0.0 and 1.0
    """
    if not 0.0 <= score <= 1.0:
        msg = f"score must be between 0.0 and 1.0, got {score}"
        raise ValueError(msg)

    if rank < 1:
        msg = f"rank must be positive, got {rank}"
        raise ValueError(msg)

    return RetrievalResult(chunk=chunk, score=score, rank=rank)


def calculate_chunk_count(
    document_length: int,
    chunk_size: int,
    overlap: int,
) -> int:
    """Calculate number of chunks for a document.

    Args:
        document_length: Total document length in characters.
        chunk_size: Size of each chunk.
        overlap: Overlap between chunks.

    Returns:
        Estimated number of chunks.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_chunk_count(1000, 200, 50)
        6

        >>> calculate_chunk_count(100, 100, 0)
        1

        >>> calculate_chunk_count(0, 200, 50)
        0

        >>> calculate_chunk_count(1000, 0, 50)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    if overlap < 0:
        msg = f"overlap must be non-negative, got {overlap}"
        raise ValueError(msg)

    if overlap >= chunk_size:
        msg = f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        raise ValueError(msg)

    if document_length <= 0:
        return 0

    if document_length <= chunk_size:
        return 1

    step = chunk_size - overlap
    return (document_length - overlap + step - 1) // step


def calculate_overlap_ratio(overlap: int, chunk_size: int) -> float:
    """Calculate overlap ratio.

    Args:
        overlap: Overlap in characters.
        chunk_size: Chunk size in characters.

    Returns:
        Overlap as a ratio (0.0 to 1.0).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_overlap_ratio(50, 500)
        0.1

        >>> calculate_overlap_ratio(100, 200)
        0.5

        >>> calculate_overlap_ratio(50, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: chunk_size must be positive
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    if overlap < 0:
        msg = f"overlap must be non-negative, got {overlap}"
        raise ValueError(msg)

    return overlap / chunk_size


def get_recommended_chunk_size(
    embedding_model: str = "all-MiniLM-L6-v2",
    use_case: str = "qa",
) -> int:
    """Get recommended chunk size for use case.

    Args:
        embedding_model: Name of embedding model.
        use_case: Use case type ("qa", "summarization", "search").

    Returns:
        Recommended chunk size in characters.

    Examples:
        >>> get_recommended_chunk_size()
        512

        >>> get_recommended_chunk_size(use_case="summarization")
        1024

        >>> get_recommended_chunk_size(use_case="search")
        256
    """
    # Base sizes by use case
    use_case_sizes = {
        "qa": 512,
        "summarization": 1024,
        "search": 256,
        "chat": 384,
    }

    base_size = use_case_sizes.get(use_case, 512)

    # Adjust for embedding model context
    if "large" in embedding_model.lower():
        base_size = int(base_size * 1.5)
    elif "small" in embedding_model.lower():
        base_size = int(base_size * 0.75)

    return base_size


def estimate_retrieval_latency(
    num_documents: int,
    top_k: int,
    use_rerank: bool = False,
) -> float:
    """Estimate retrieval latency in milliseconds.

    Args:
        num_documents: Number of documents in index.
        top_k: Number of documents to retrieve.
        use_rerank: Whether reranking is used.

    Returns:
        Estimated latency in milliseconds.

    Examples:
        >>> latency = estimate_retrieval_latency(10000, 10)
        >>> latency > 0
        True

        >>> with_rerank = estimate_retrieval_latency(10000, 10, use_rerank=True)
        >>> with_rerank > latency
        True
    """
    # Base latency scales logarithmically with document count
    import math

    base_latency = 5.0 + math.log10(max(1, num_documents)) * 2.0

    # Add latency for top_k
    base_latency += top_k * 0.1

    # Reranking adds significant latency
    if use_rerank:
        base_latency += top_k * 5.0

    return round(base_latency, 2)


def format_retrieval_stats(stats: RetrievalStats) -> str:
    """Format retrieval stats for display.

    Args:
        stats: Retrieval statistics.

    Returns:
        Formatted string.

    Examples:
        >>> stats = RetrievalStats(10, 0.85, 50.0, True)
        >>> formatted = format_retrieval_stats(stats)
        >>> "10 documents" in formatted
        True
        >>> "reranked" in formatted
        True
    """
    parts = [
        f"Retrieved {stats.total_retrieved} documents",
        f"avg_score={stats.avg_score:.2f}",
        f"latency={stats.latency_ms:.1f}ms",
    ]

    if stats.reranked:
        parts.append("(reranked)")

    return " | ".join(parts)


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
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_CHUNKING_STRATEGIES)


def list_distance_metrics() -> list[str]:
    """List supported distance metrics.

    Returns:
        Sorted list of metric names.

    Examples:
        >>> metrics = list_distance_metrics()
        >>> "cosine" in metrics
        True
        >>> "euclidean" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_DISTANCE_METRICS)


def list_retrieval_methods() -> list[str]:
    """List supported retrieval methods.

    Returns:
        Sorted list of method names.

    Examples:
        >>> methods = list_retrieval_methods()
        >>> "dense" in methods
        True
        >>> "hybrid" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_RETRIEVAL_METHODS)


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

        >>> get_chunking_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if name not in VALID_CHUNKING_STRATEGIES:
        msg = f"strategy must be one of {VALID_CHUNKING_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return ChunkingStrategy(name)


def get_distance_metric(name: str) -> DistanceMetric:
    """Get distance metric from name.

    Args:
        name: Metric name.

    Returns:
        DistanceMetric enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_distance_metric("cosine")
        <DistanceMetric.COSINE: 'cosine'>

        >>> get_distance_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric must be one of
    """
    if name not in VALID_DISTANCE_METRICS:
        msg = f"metric must be one of {VALID_DISTANCE_METRICS}, got '{name}'"
        raise ValueError(msg)
    return DistanceMetric(name)


def get_retrieval_method(name: str) -> RetrievalMethod:
    """Get retrieval method from name.

    Args:
        name: Method name.

    Returns:
        RetrievalMethod enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_retrieval_method("dense")
        <RetrievalMethod.DENSE: 'dense'>

        >>> get_retrieval_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if name not in VALID_RETRIEVAL_METHODS:
        msg = f"method must be one of {VALID_RETRIEVAL_METHODS}, got '{name}'"
        raise ValueError(msg)
    return RetrievalMethod(name)
