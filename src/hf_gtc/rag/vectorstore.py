"""Vector store integration and similarity search utilities.

This module provides configuration and utilities for integrating with various
vector store backends and performing similarity search operations.

Examples:
    >>> from hf_gtc.rag.vectorstore import create_vectorstore_config
    >>> config = create_vectorstore_config(store_type="faiss", dimension=768)
    >>> config.dimension
    768
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class VectorStoreType(Enum):
    """Supported vector store backends.

    Attributes:
        FAISS: Facebook AI Similarity Search (local).
        CHROMADB: ChromaDB vector store (local/cloud).
        PINECONE: Pinecone managed vector database.
        WEAVIATE: Weaviate vector search engine.
        QDRANT: Qdrant vector search engine.
        MILVUS: Milvus vector database.

    Examples:
        >>> VectorStoreType.FAISS.value
        'faiss'
        >>> VectorStoreType.CHROMADB.value
        'chromadb'
    """

    FAISS = "faiss"
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"


VALID_STORE_TYPES = frozenset(s.value for s in VectorStoreType)


class IndexType(Enum):
    """Supported index types for vector search.

    Attributes:
        FLAT: Brute-force exact search.
        IVF: Inverted file index.
        HNSW: Hierarchical Navigable Small World graph.
        PQ: Product quantization.
        SCANN: Scalable Nearest Neighbors.

    Examples:
        >>> IndexType.FLAT.value
        'flat'
        >>> IndexType.HNSW.value
        'hnsw'
    """

    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    PQ = "pq"
    SCANN = "scann"


VALID_INDEX_TYPES = frozenset(i.value for i in IndexType)


class DistanceMetric(Enum):
    """Supported distance metrics for similarity search.

    Attributes:
        COSINE: Cosine similarity (1 - cosine distance).
        EUCLIDEAN: Euclidean (L2) distance.
        DOT_PRODUCT: Dot product (inner product).
        MANHATTAN: Manhattan (L1) distance.

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


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Configuration for vector index parameters.

    Attributes:
        index_type: Type of index algorithm.
        nlist: Number of clusters for IVF indices.
        nprobe: Number of clusters to search.
        ef_search: HNSW search parameter.

    Examples:
        >>> config = IndexConfig(
        ...     index_type=IndexType.HNSW,
        ...     nlist=100,
        ...     nprobe=10,
        ...     ef_search=128,
        ... )
        >>> config.index_type
        <IndexType.HNSW: 'hnsw'>
    """

    index_type: IndexType
    nlist: int
    nprobe: int
    ef_search: int


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Configuration for vector store.

    Attributes:
        store_type: Type of vector store backend.
        index_config: Index configuration.
        distance_metric: Distance metric for similarity.
        dimension: Vector dimension.

    Examples:
        >>> idx_config = IndexConfig(IndexType.FLAT, 100, 10, 128)
        >>> config = VectorStoreConfig(
        ...     store_type=VectorStoreType.FAISS,
        ...     index_config=idx_config,
        ...     distance_metric=DistanceMetric.COSINE,
        ...     dimension=768,
        ... )
        >>> config.dimension
        768
    """

    store_type: VectorStoreType
    index_config: IndexConfig
    distance_metric: DistanceMetric
    dimension: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result from similarity search.

    Attributes:
        ids: Tuple of result IDs.
        scores: Tuple of similarity scores.
        vectors: Tuple of vector tuples (optional).
        metadata: Tuple of metadata dicts.

    Examples:
        >>> result = SearchResult(
        ...     ids=("id1", "id2"),
        ...     scores=(0.95, 0.87),
        ...     vectors=(),
        ...     metadata=({"source": "doc1"}, {"source": "doc2"}),
        ... )
        >>> result.ids
        ('id1', 'id2')
    """

    ids: tuple[str, ...]
    scores: tuple[float, ...]
    vectors: tuple[tuple[float, ...], ...]
    metadata: tuple[dict[str, Any], ...]


def validate_index_config(config: IndexConfig) -> None:
    """Validate index configuration.

    Args:
        config: Index configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = IndexConfig(IndexType.IVF, 100, 10, 128)
        >>> validate_index_config(config)  # No error

        >>> bad = IndexConfig(IndexType.IVF, 0, 10, 128)
        >>> validate_index_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: nlist must be positive
    """
    if config.nlist <= 0:
        msg = f"nlist must be positive, got {config.nlist}"
        raise ValueError(msg)

    if config.nprobe <= 0:
        msg = f"nprobe must be positive, got {config.nprobe}"
        raise ValueError(msg)

    if config.nprobe > config.nlist:
        msg = f"nprobe ({config.nprobe}) cannot exceed nlist ({config.nlist})"
        raise ValueError(msg)

    if config.ef_search <= 0:
        msg = f"ef_search must be positive, got {config.ef_search}"
        raise ValueError(msg)


def validate_vectorstore_config(config: VectorStoreConfig) -> None:
    """Validate vector store configuration.

    Args:
        config: Vector store configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> idx = IndexConfig(IndexType.FLAT, 100, 10, 128)
        >>> config = VectorStoreConfig(
        ...     VectorStoreType.FAISS, idx, DistanceMetric.COSINE, 768
        ... )
        >>> validate_vectorstore_config(config)  # No error

        >>> bad_config = VectorStoreConfig(
        ...     VectorStoreType.FAISS, idx, DistanceMetric.COSINE, 0
        ... )
        >>> validate_vectorstore_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dimension must be positive
    """
    if config.dimension <= 0:
        msg = f"dimension must be positive, got {config.dimension}"
        raise ValueError(msg)

    validate_index_config(config.index_config)


def validate_search_result(result: SearchResult) -> None:
    """Validate search result.

    Args:
        result: Search result to validate.

    Raises:
        ValueError: If result is invalid.

    Examples:
        >>> result = SearchResult(("id1",), (0.9,), (), ({},))
        >>> validate_search_result(result)  # No error

        >>> bad = SearchResult(("id1",), (0.9, 0.8), (), ({},))
        >>> validate_search_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ids and scores must have the same length
    """
    if len(result.ids) != len(result.scores):
        msg = (
            f"ids and scores must have the same length, "
            f"got {len(result.ids)} and {len(result.scores)}"
        )
        raise ValueError(msg)

    if len(result.ids) != len(result.metadata):
        msg = (
            f"ids and metadata must have the same length, "
            f"got {len(result.ids)} and {len(result.metadata)}"
        )
        raise ValueError(msg)

    if result.vectors and len(result.vectors) != len(result.ids):
        msg = (
            f"vectors must be empty or have same length as ids, "
            f"got {len(result.vectors)} and {len(result.ids)}"
        )
        raise ValueError(msg)

    for score in result.scores:
        if not 0.0 <= score <= 1.0:
            msg = f"scores must be between 0.0 and 1.0, got {score}"
            raise ValueError(msg)


def create_index_config(
    index_type: str = "flat",
    nlist: int = 100,
    nprobe: int = 10,
    ef_search: int = 128,
) -> IndexConfig:
    """Create an index configuration.

    Args:
        index_type: Type of index. Defaults to "flat".
        nlist: Number of clusters for IVF. Defaults to 100.
        nprobe: Clusters to search. Defaults to 10.
        ef_search: HNSW search parameter. Defaults to 128.

    Returns:
        IndexConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_index_config(index_type="hnsw", ef_search=256)
        >>> config.index_type
        <IndexType.HNSW: 'hnsw'>

        >>> create_index_config(index_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: index_type must be one of
    """
    if index_type not in VALID_INDEX_TYPES:
        msg = f"index_type must be one of {VALID_INDEX_TYPES}, got '{index_type}'"
        raise ValueError(msg)

    config = IndexConfig(
        index_type=IndexType(index_type),
        nlist=nlist,
        nprobe=nprobe,
        ef_search=ef_search,
    )
    validate_index_config(config)
    return config


def create_vectorstore_config(
    store_type: str = "faiss",
    index_type: str = "flat",
    distance_metric: str = "cosine",
    dimension: int = 768,
    nlist: int = 100,
    nprobe: int = 10,
    ef_search: int = 128,
) -> VectorStoreConfig:
    """Create a vector store configuration.

    Args:
        store_type: Type of vector store. Defaults to "faiss".
        index_type: Type of index. Defaults to "flat".
        distance_metric: Distance metric. Defaults to "cosine".
        dimension: Vector dimension. Defaults to 768.
        nlist: Number of clusters for IVF. Defaults to 100.
        nprobe: Clusters to search. Defaults to 10.
        ef_search: HNSW search parameter. Defaults to 128.

    Returns:
        VectorStoreConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_vectorstore_config(store_type="chromadb", dimension=384)
        >>> config.store_type
        <VectorStoreType.CHROMADB: 'chromadb'>

        >>> create_vectorstore_config(store_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: store_type must be one of
    """
    if store_type not in VALID_STORE_TYPES:
        msg = f"store_type must be one of {VALID_STORE_TYPES}, got '{store_type}'"
        raise ValueError(msg)

    if distance_metric not in VALID_DISTANCE_METRICS:
        msg = (
            f"distance_metric must be one of {VALID_DISTANCE_METRICS}, "
            f"got '{distance_metric}'"
        )
        raise ValueError(msg)

    index_config = create_index_config(
        index_type=index_type,
        nlist=nlist,
        nprobe=nprobe,
        ef_search=ef_search,
    )

    config = VectorStoreConfig(
        store_type=VectorStoreType(store_type),
        index_config=index_config,
        distance_metric=DistanceMetric(distance_metric),
        dimension=dimension,
    )
    validate_vectorstore_config(config)
    return config


def create_search_result(
    ids: tuple[str, ...],
    scores: tuple[float, ...],
    vectors: tuple[tuple[float, ...], ...] | None = None,
    metadata: tuple[dict[str, Any], ...] | None = None,
) -> SearchResult:
    """Create a search result.

    Args:
        ids: Result IDs.
        scores: Similarity scores.
        vectors: Result vectors (optional).
        metadata: Result metadata (optional).

    Returns:
        SearchResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_search_result(
        ...     ids=("id1", "id2"),
        ...     scores=(0.95, 0.87),
        ... )
        >>> result.ids
        ('id1', 'id2')

        >>> create_search_result(ids=("id1",), scores=(1.5,))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores must be between 0.0 and 1.0
    """
    if vectors is None:
        vectors = ()

    if metadata is None:
        metadata = tuple({} for _ in ids)

    result = SearchResult(
        ids=ids,
        scores=scores,
        vectors=vectors,
        metadata=metadata,
    )
    validate_search_result(result)
    return result


def list_store_types() -> list[str]:
    """List supported vector store types.

    Returns:
        Sorted list of store type names.

    Examples:
        >>> types = list_store_types()
        >>> "faiss" in types
        True
        >>> "chromadb" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_STORE_TYPES)


def list_index_types() -> list[str]:
    """List supported index types.

    Returns:
        Sorted list of index type names.

    Examples:
        >>> types = list_index_types()
        >>> "flat" in types
        True
        >>> "hnsw" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_INDEX_TYPES)


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


def get_store_type(name: str) -> VectorStoreType:
    """Get vector store type from name.

    Args:
        name: Store type name.

    Returns:
        VectorStoreType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_store_type("faiss")
        <VectorStoreType.FAISS: 'faiss'>

        >>> get_store_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: store_type must be one of
    """
    if name not in VALID_STORE_TYPES:
        msg = f"store_type must be one of {VALID_STORE_TYPES}, got '{name}'"
        raise ValueError(msg)
    return VectorStoreType(name)


def get_index_type(name: str) -> IndexType:
    """Get index type from name.

    Args:
        name: Index type name.

    Returns:
        IndexType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_index_type("hnsw")
        <IndexType.HNSW: 'hnsw'>

        >>> get_index_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: index_type must be one of
    """
    if name not in VALID_INDEX_TYPES:
        msg = f"index_type must be one of {VALID_INDEX_TYPES}, got '{name}'"
        raise ValueError(msg)
    return IndexType(name)


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
        ValueError: distance_metric must be one of
    """
    if name not in VALID_DISTANCE_METRICS:
        msg = f"distance_metric must be one of {VALID_DISTANCE_METRICS}, got '{name}'"
        raise ValueError(msg)
    return DistanceMetric(name)


def calculate_index_size(
    num_vectors: int,
    dimension: int,
    index_type: IndexType,
    bytes_per_float: int = 4,
) -> int:
    """Calculate estimated index size in bytes.

    Args:
        num_vectors: Number of vectors to index.
        dimension: Vector dimension.
        index_type: Type of index.
        bytes_per_float: Bytes per float value. Defaults to 4.

    Returns:
        Estimated size in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> size = calculate_index_size(10000, 768, IndexType.FLAT)
        >>> size > 0
        True

        >>> flat = calculate_index_size(10000, 768, IndexType.FLAT)
        >>> pq = calculate_index_size(10000, 768, IndexType.PQ)
        >>> pq < flat
        True

        >>> calculate_index_size(0, 768, IndexType.FLAT)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_vectors must be positive
    """
    if num_vectors <= 0:
        msg = f"num_vectors must be positive, got {num_vectors}"
        raise ValueError(msg)

    if dimension <= 0:
        msg = f"dimension must be positive, got {dimension}"
        raise ValueError(msg)

    if bytes_per_float <= 0:
        msg = f"bytes_per_float must be positive, got {bytes_per_float}"
        raise ValueError(msg)

    base_size = num_vectors * dimension * bytes_per_float

    # Overhead factors by index type
    overhead_factors = {
        IndexType.FLAT: 1.0,
        IndexType.IVF: 1.1,
        IndexType.HNSW: 1.5,
        IndexType.PQ: 0.25,
        IndexType.SCANN: 0.4,
    }

    factor = overhead_factors.get(index_type, 1.0)
    return int(base_size * factor)


def estimate_search_latency(
    num_vectors: int,
    index_type: IndexType,
    top_k: int = 10,
) -> float:
    """Estimate search latency in milliseconds.

    Args:
        num_vectors: Number of vectors in index.
        index_type: Type of index.
        top_k: Number of results to return. Defaults to 10.

    Returns:
        Estimated latency in milliseconds.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> latency = estimate_search_latency(100000, IndexType.FLAT)
        >>> latency > 0
        True

        >>> flat_lat = estimate_search_latency(100000, IndexType.FLAT)
        >>> hnsw_lat = estimate_search_latency(100000, IndexType.HNSW)
        >>> hnsw_lat < flat_lat
        True

        >>> estimate_search_latency(0, IndexType.FLAT)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_vectors must be positive
    """
    if num_vectors <= 0:
        msg = f"num_vectors must be positive, got {num_vectors}"
        raise ValueError(msg)

    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)

    # Base latency scaling by index type
    latency_factors = {
        IndexType.FLAT: lambda n: n * 0.001,  # Linear scan
        IndexType.IVF: lambda n: math.sqrt(n) * 0.1,  # Sublinear
        IndexType.HNSW: lambda n: math.log10(max(1, n)) * 2.0,  # Logarithmic
        IndexType.PQ: lambda n: math.sqrt(n) * 0.05,  # Sublinear, fast decode
        IndexType.SCANN: lambda n: math.log10(max(1, n)) * 1.5,  # Logarithmic
    }

    latency_fn = latency_factors.get(index_type, lambda n: n * 0.001)
    base_latency = latency_fn(num_vectors)

    # Add overhead for top_k
    base_latency += top_k * 0.01

    return round(max(0.1, base_latency), 2)


def calculate_recall_at_k(
    relevant_retrieved: int,
    total_relevant: int,
    k: int,
) -> float:
    """Calculate recall@k metric.

    Args:
        relevant_retrieved: Number of relevant items in top-k.
        total_relevant: Total relevant items in dataset.
        k: Number of items retrieved.

    Returns:
        Recall@k score (0.0 to 1.0).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_recall_at_k(8, 10, 10)
        0.8

        >>> calculate_recall_at_k(5, 5, 10)
        1.0

        >>> calculate_recall_at_k(0, 10, 5)
        0.0

        >>> calculate_recall_at_k(-1, 10, 5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: relevant_retrieved must be non-negative
    """
    if relevant_retrieved < 0:
        msg = f"relevant_retrieved must be non-negative, got {relevant_retrieved}"
        raise ValueError(msg)

    if total_relevant <= 0:
        msg = f"total_relevant must be positive, got {total_relevant}"
        raise ValueError(msg)

    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)

    if relevant_retrieved > total_relevant:
        msg = (
            f"relevant_retrieved ({relevant_retrieved}) cannot exceed "
            f"total_relevant ({total_relevant})"
        )
        raise ValueError(msg)

    if relevant_retrieved > k:
        msg = f"relevant_retrieved ({relevant_retrieved}) cannot exceed k ({k})"
        raise ValueError(msg)

    return relevant_retrieved / total_relevant


def optimize_index_params(
    num_vectors: int,
    dimension: int,
    recall_target: float = 0.95,
    latency_budget_ms: float = 100.0,
) -> dict[str, Any]:
    """Optimize index parameters for given constraints.

    Args:
        num_vectors: Expected number of vectors.
        dimension: Vector dimension.
        recall_target: Target recall (0-1). Defaults to 0.95.
        latency_budget_ms: Max latency in ms. Defaults to 100.0.

    Returns:
        Dict with optimized parameters.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> params = optimize_index_params(100000, 768)
        >>> "index_type" in params
        True
        >>> "nlist" in params
        True

        >>> params = optimize_index_params(1000, 128)
        >>> params["index_type"]
        'flat'

        >>> optimize_index_params(0, 768)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_vectors must be positive
    """
    if num_vectors <= 0:
        msg = f"num_vectors must be positive, got {num_vectors}"
        raise ValueError(msg)

    if dimension <= 0:
        msg = f"dimension must be positive, got {dimension}"
        raise ValueError(msg)

    if not 0.0 < recall_target <= 1.0:
        msg = f"recall_target must be in (0, 1], got {recall_target}"
        raise ValueError(msg)

    if latency_budget_ms <= 0:
        msg = f"latency_budget_ms must be positive, got {latency_budget_ms}"
        raise ValueError(msg)

    # For small datasets, use flat index
    if num_vectors < 10000:
        return {
            "index_type": "flat",
            "nlist": 1,
            "nprobe": 1,
            "ef_search": 64,
            "estimated_recall": 1.0,
            "estimated_latency_ms": num_vectors * 0.001,
        }

    # Calculate optimal nlist
    nlist = max(16, min(int(math.sqrt(num_vectors)), 4096))

    # Calculate nprobe based on recall target
    nprobe = max(1, min(int(nlist * recall_target * 0.3), nlist))

    # HNSW ef_search based on recall
    ef_search = max(32, min(int(256 * recall_target), 512))

    # Choose index type based on latency budget and dataset size
    if num_vectors > 1000000:
        if latency_budget_ms < 10:
            index_type = "hnsw"
        elif latency_budget_ms < 50:
            index_type = "ivf"
        else:
            index_type = "hnsw"
    elif num_vectors > 100000:
        index_type = "hnsw" if latency_budget_ms < 20 else "ivf"
    else:
        index_type = "ivf" if recall_target < 0.99 else "flat"

    # Estimate actual latency
    idx_type = IndexType(index_type)
    estimated_latency = estimate_search_latency(num_vectors, idx_type)

    return {
        "index_type": index_type,
        "nlist": nlist,
        "nprobe": nprobe,
        "ef_search": ef_search,
        "estimated_recall": min(recall_target, 0.99),
        "estimated_latency_ms": estimated_latency,
    }


def format_search_stats(
    result: SearchResult,
    latency_ms: float,
    index_type: IndexType,
) -> str:
    """Format search statistics for display.

    Args:
        result: Search result.
        latency_ms: Search latency in milliseconds.
        index_type: Type of index used.

    Returns:
        Formatted statistics string.

    Examples:
        >>> result = create_search_result(("id1", "id2"), (0.95, 0.87))
        >>> stats = format_search_stats(result, 5.2, IndexType.HNSW)
        >>> "2 results" in stats
        True
        >>> "5.2ms" in stats
        True
        >>> "hnsw" in stats.lower()
        True
    """
    num_results = len(result.ids)
    avg_score = sum(result.scores) / len(result.scores) if result.scores else 0.0

    parts = [
        f"Found {num_results} results",
        f"avg_score={avg_score:.3f}",
        f"latency={latency_ms:.1f}ms",
        f"index={index_type.value}",
    ]

    return " | ".join(parts)


def get_recommended_vectorstore_config(
    use_case: str = "similarity_search",
    num_vectors: int = 100000,
    dimension: int = 768,
) -> VectorStoreConfig:
    """Get recommended vector store configuration for use case.

    Args:
        use_case: Use case type. Defaults to "similarity_search".
        num_vectors: Expected number of vectors. Defaults to 100000.
        dimension: Vector dimension. Defaults to 768.

    Returns:
        Recommended VectorStoreConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_vectorstore_config()
        >>> config.store_type == VectorStoreType.FAISS
        True

        >>> config = get_recommended_vectorstore_config(
        ...     use_case="production", num_vectors=10000000
        ... )
        >>> config.index_config.index_type in (IndexType.HNSW, IndexType.IVF)
        True

        >>> config = get_recommended_vectorstore_config(num_vectors=1000)
        >>> config.index_config.index_type
        <IndexType.FLAT: 'flat'>
    """
    if num_vectors <= 0:
        msg = f"num_vectors must be positive, got {num_vectors}"
        raise ValueError(msg)

    if dimension <= 0:
        msg = f"dimension must be positive, got {dimension}"
        raise ValueError(msg)

    # Use case to configuration mapping
    use_case_configs = {
        "similarity_search": {
            "store_type": "faiss",
            "distance_metric": "cosine",
            "recall_target": 0.95,
            "latency_budget_ms": 50.0,
        },
        "semantic_search": {
            "store_type": "faiss",
            "distance_metric": "cosine",
            "recall_target": 0.98,
            "latency_budget_ms": 100.0,
        },
        "production": {
            "store_type": "qdrant",
            "distance_metric": "cosine",
            "recall_target": 0.95,
            "latency_budget_ms": 20.0,
        },
        "development": {
            "store_type": "chromadb",
            "distance_metric": "cosine",
            "recall_target": 0.90,
            "latency_budget_ms": 200.0,
        },
    }

    config = use_case_configs.get(
        use_case,
        use_case_configs["similarity_search"],
    )

    # Select index parameters based on recall and latency constraints
    params = optimize_index_params(
        num_vectors=num_vectors,
        dimension=dimension,
        recall_target=float(config["recall_target"]),
        latency_budget_ms=float(config["latency_budget_ms"]),
    )

    return create_vectorstore_config(
        store_type=str(config["store_type"]),
        index_type=str(params["index_type"]),
        distance_metric=str(config["distance_metric"]),
        dimension=dimension,
        nlist=int(params["nlist"]),
        nprobe=int(params["nprobe"]),
        ef_search=int(params["ef_search"]),
    )
