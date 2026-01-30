"""Vector store indexing utilities.

This module provides configuration and utilities for vector store
indexing patterns including FAISS, ChromaDB, and other backends.

Examples:
    >>> from hf_gtc.rag.indexing import create_faiss_config
    >>> config = create_faiss_config(index_type="ivf_flat", nlist=100)
    >>> config.index_type
    <FAISSIndexType.IVF_FLAT: 'ivf_flat'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class FAISSIndexType(Enum):
    """FAISS index types.

    Attributes:
        FLAT: Brute-force L2 search (exact).
        IVF_FLAT: Inverted file with flat quantizer.
        IVF_PQ: Inverted file with product quantization.
        HNSW: Hierarchical Navigable Small World graph.
        LSH: Locality-Sensitive Hashing.

    Examples:
        >>> FAISSIndexType.FLAT.value
        'flat'
        >>> FAISSIndexType.HNSW.value
        'hnsw'
    """

    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    LSH = "lsh"


VALID_FAISS_INDEX_TYPES = frozenset(t.value for t in FAISSIndexType)


class VectorStoreBackend(Enum):
    """Vector store backends.

    Attributes:
        FAISS: Facebook AI Similarity Search.
        CHROMA: ChromaDB vector store.
        PINECONE: Pinecone managed vector database.
        WEAVIATE: Weaviate vector search engine.
        MILVUS: Milvus vector database.
        QDRANT: Qdrant vector search engine.

    Examples:
        >>> VectorStoreBackend.FAISS.value
        'faiss'
        >>> VectorStoreBackend.CHROMA.value
        'chroma'
    """

    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    QDRANT = "qdrant"


VALID_BACKENDS = frozenset(b.value for b in VectorStoreBackend)


class DistanceFunction(Enum):
    """Distance functions for similarity search.

    Attributes:
        L2: Euclidean distance (L2 norm).
        IP: Inner product (dot product).
        COSINE: Cosine similarity.

    Examples:
        >>> DistanceFunction.COSINE.value
        'cosine'
    """

    L2 = "l2"
    IP = "ip"
    COSINE = "cosine"


VALID_DISTANCE_FUNCTIONS = frozenset(d.value for d in DistanceFunction)


IndexTypeStr = Literal["flat", "ivf_flat", "ivf_pq", "hnsw", "lsh"]
BackendStr = Literal["faiss", "chroma", "pinecone", "weaviate", "milvus", "qdrant"]
DistanceFunctionStr = Literal["l2", "ip", "cosine"]


@dataclass(frozen=True, slots=True)
class FAISSConfig:
    """Configuration for FAISS index.

    Attributes:
        index_type: Type of FAISS index.
        dimension: Vector dimension.
        nlist: Number of clusters for IVF indices.
        nprobe: Number of clusters to search.
        m: Number of subquantizers for PQ.
        nbits: Bits per subquantizer for PQ.
        ef_construction: HNSW construction parameter.
        ef_search: HNSW search parameter.

    Examples:
        >>> config = FAISSConfig(
        ...     index_type=FAISSIndexType.IVF_FLAT,
        ...     dimension=768,
        ...     nlist=100,
        ...     nprobe=10,
        ...     m=8,
        ...     nbits=8,
        ...     ef_construction=200,
        ...     ef_search=128,
        ... )
        >>> config.dimension
        768
    """

    index_type: FAISSIndexType
    dimension: int
    nlist: int
    nprobe: int
    m: int
    nbits: int
    ef_construction: int
    ef_search: int


@dataclass(frozen=True, slots=True)
class ChromaConfig:
    """Configuration for ChromaDB.

    Attributes:
        collection_name: Name of the collection.
        distance_function: Distance metric to use.
        persist_directory: Directory to persist data.
        anonymized_telemetry: Whether to enable telemetry.

    Examples:
        >>> config = ChromaConfig(
        ...     collection_name="documents",
        ...     distance_function=DistanceFunction.COSINE,
        ...     persist_directory="./chroma_db",
        ...     anonymized_telemetry=False,
        ... )
        >>> config.collection_name
        'documents'
    """

    collection_name: str
    distance_function: DistanceFunction
    persist_directory: str
    anonymized_telemetry: bool


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """General vector index configuration.

    Attributes:
        backend: Vector store backend.
        dimension: Vector dimension.
        distance_function: Distance metric.
        batch_size: Batch size for indexing.
        normalize_embeddings: Whether to normalize vectors.

    Examples:
        >>> config = IndexConfig(
        ...     backend=VectorStoreBackend.FAISS,
        ...     dimension=768,
        ...     distance_function=DistanceFunction.COSINE,
        ...     batch_size=1000,
        ...     normalize_embeddings=True,
        ... )
        >>> config.normalize_embeddings
        True
    """

    backend: VectorStoreBackend
    dimension: int
    distance_function: DistanceFunction
    batch_size: int
    normalize_embeddings: bool


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Configuration for vector search.

    Attributes:
        top_k: Number of results to return.
        score_threshold: Minimum similarity score.
        include_metadata: Whether to include metadata.
        include_embeddings: Whether to return embeddings.

    Examples:
        >>> config = SearchConfig(
        ...     top_k=10,
        ...     score_threshold=0.7,
        ...     include_metadata=True,
        ...     include_embeddings=False,
        ... )
        >>> config.top_k
        10
    """

    top_k: int
    score_threshold: float
    include_metadata: bool
    include_embeddings: bool


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Statistics for a vector index.

    Attributes:
        total_vectors: Total number of indexed vectors.
        dimension: Vector dimension.
        index_size_bytes: Size of index in bytes.
        is_trained: Whether the index is trained.

    Examples:
        >>> stats = IndexStats(
        ...     total_vectors=10000,
        ...     dimension=768,
        ...     index_size_bytes=15360000,
        ...     is_trained=True,
        ... )
        >>> stats.total_vectors
        10000
    """

    total_vectors: int
    dimension: int
    index_size_bytes: int
    is_trained: bool


def validate_faiss_config(config: FAISSConfig) -> None:
    """Validate FAISS configuration.

    Args:
        config: FAISS configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = FAISSConfig(
        ...     FAISSIndexType.FLAT, 768, 100, 10, 8, 8, 200, 128
        ... )
        >>> validate_faiss_config(config)  # No error

        >>> bad = FAISSConfig(FAISSIndexType.FLAT, 0, 100, 10, 8, 8, 200, 128)
        >>> validate_faiss_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dimension must be positive
    """
    if config.dimension <= 0:
        msg = f"dimension must be positive, got {config.dimension}"
        raise ValueError(msg)

    if config.nlist <= 0:
        msg = f"nlist must be positive, got {config.nlist}"
        raise ValueError(msg)

    if config.nprobe <= 0:
        msg = f"nprobe must be positive, got {config.nprobe}"
        raise ValueError(msg)

    if config.nprobe > config.nlist:
        msg = f"nprobe ({config.nprobe}) cannot exceed nlist ({config.nlist})"
        raise ValueError(msg)


def validate_search_config(config: SearchConfig) -> None:
    """Validate search configuration.

    Args:
        config: Search configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SearchConfig(10, 0.7, True, False)
        >>> validate_search_config(config)  # No error

        >>> bad = SearchConfig(0, 0.7, True, False)
        >>> validate_search_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    if config.top_k <= 0:
        msg = f"top_k must be positive, got {config.top_k}"
        raise ValueError(msg)

    if not 0.0 <= config.score_threshold <= 1.0:
        msg = (
            f"score_threshold must be between 0.0 and 1.0, "
            f"got {config.score_threshold}"
        )
        raise ValueError(msg)


def create_faiss_config(
    index_type: str = "flat",
    dimension: int = 768,
    nlist: int = 100,
    nprobe: int = 10,
    m: int = 8,
    nbits: int = 8,
    ef_construction: int = 200,
    ef_search: int = 128,
) -> FAISSConfig:
    """Create a FAISS configuration.

    Args:
        index_type: Type of FAISS index. Defaults to "flat".
        dimension: Vector dimension. Defaults to 768.
        nlist: Number of clusters for IVF. Defaults to 100.
        nprobe: Clusters to search. Defaults to 10.
        m: Subquantizers for PQ. Defaults to 8.
        nbits: Bits per subquantizer. Defaults to 8.
        ef_construction: HNSW construction param. Defaults to 200.
        ef_search: HNSW search param. Defaults to 128.

    Returns:
        FAISSConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_faiss_config(index_type="ivf_flat", nlist=200)
        >>> config.index_type
        <FAISSIndexType.IVF_FLAT: 'ivf_flat'>

        >>> create_faiss_config(index_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: index_type must be one of
    """
    if index_type not in VALID_FAISS_INDEX_TYPES:
        msg = f"index_type must be one of {VALID_FAISS_INDEX_TYPES}, got '{index_type}'"
        raise ValueError(msg)

    config = FAISSConfig(
        index_type=FAISSIndexType(index_type),
        dimension=dimension,
        nlist=nlist,
        nprobe=nprobe,
        m=m,
        nbits=nbits,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    validate_faiss_config(config)
    return config


def create_chroma_config(
    collection_name: str,
    distance_function: str = "cosine",
    persist_directory: str = "./chroma_db",
    anonymized_telemetry: bool = False,
) -> ChromaConfig:
    """Create a ChromaDB configuration.

    Args:
        collection_name: Name of the collection.
        distance_function: Distance metric. Defaults to "cosine".
        persist_directory: Persist directory. Defaults to "./chroma_db".
        anonymized_telemetry: Enable telemetry. Defaults to False.

    Returns:
        ChromaConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_chroma_config("my_docs")
        >>> config.collection_name
        'my_docs'

        >>> create_chroma_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: collection_name cannot be empty
    """
    if not collection_name:
        msg = "collection_name cannot be empty"
        raise ValueError(msg)

    if distance_function not in VALID_DISTANCE_FUNCTIONS:
        msg = (
            f"distance_function must be one of {VALID_DISTANCE_FUNCTIONS}, "
            f"got '{distance_function}'"
        )
        raise ValueError(msg)

    return ChromaConfig(
        collection_name=collection_name,
        distance_function=DistanceFunction(distance_function),
        persist_directory=persist_directory,
        anonymized_telemetry=anonymized_telemetry,
    )


def create_index_config(
    backend: str = "faiss",
    dimension: int = 768,
    distance_function: str = "cosine",
    batch_size: int = 1000,
    normalize_embeddings: bool = True,
) -> IndexConfig:
    """Create a general index configuration.

    Args:
        backend: Vector store backend. Defaults to "faiss".
        dimension: Vector dimension. Defaults to 768.
        distance_function: Distance metric. Defaults to "cosine".
        batch_size: Indexing batch size. Defaults to 1000.
        normalize_embeddings: Normalize vectors. Defaults to True.

    Returns:
        IndexConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_index_config(backend="chroma", dimension=384)
        >>> config.backend
        <VectorStoreBackend.CHROMA: 'chroma'>

        >>> create_index_config(dimension=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dimension must be positive
    """
    if backend not in VALID_BACKENDS:
        msg = f"backend must be one of {VALID_BACKENDS}, got '{backend}'"
        raise ValueError(msg)

    if dimension <= 0:
        msg = f"dimension must be positive, got {dimension}"
        raise ValueError(msg)

    if distance_function not in VALID_DISTANCE_FUNCTIONS:
        msg = (
            f"distance_function must be one of {VALID_DISTANCE_FUNCTIONS}, "
            f"got '{distance_function}'"
        )
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    return IndexConfig(
        backend=VectorStoreBackend(backend),
        dimension=dimension,
        distance_function=DistanceFunction(distance_function),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )


def create_search_config(
    top_k: int = 10,
    score_threshold: float = 0.0,
    include_metadata: bool = True,
    include_embeddings: bool = False,
) -> SearchConfig:
    """Create a search configuration.

    Args:
        top_k: Number of results. Defaults to 10.
        score_threshold: Minimum score. Defaults to 0.0.
        include_metadata: Include metadata. Defaults to True.
        include_embeddings: Return embeddings. Defaults to False.

    Returns:
        SearchConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_search_config(top_k=5, score_threshold=0.8)
        >>> config.top_k
        5

        >>> create_search_config(top_k=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    config = SearchConfig(
        top_k=top_k,
        score_threshold=score_threshold,
        include_metadata=include_metadata,
        include_embeddings=include_embeddings,
    )
    validate_search_config(config)
    return config


def create_index_stats(
    total_vectors: int = 0,
    dimension: int = 768,
    index_size_bytes: int = 0,
    is_trained: bool = False,
) -> IndexStats:
    """Create index statistics.

    Args:
        total_vectors: Total indexed vectors. Defaults to 0.
        dimension: Vector dimension. Defaults to 768.
        index_size_bytes: Index size in bytes. Defaults to 0.
        is_trained: Whether index is trained. Defaults to False.

    Returns:
        IndexStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_index_stats(total_vectors=1000, is_trained=True)
        >>> stats.total_vectors
        1000

        >>> create_index_stats(total_vectors=-1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_vectors must be non-negative
    """
    if total_vectors < 0:
        msg = f"total_vectors must be non-negative, got {total_vectors}"
        raise ValueError(msg)

    if dimension <= 0:
        msg = f"dimension must be positive, got {dimension}"
        raise ValueError(msg)

    if index_size_bytes < 0:
        msg = f"index_size_bytes must be non-negative, got {index_size_bytes}"
        raise ValueError(msg)

    return IndexStats(
        total_vectors=total_vectors,
        dimension=dimension,
        index_size_bytes=index_size_bytes,
        is_trained=is_trained,
    )


def list_faiss_index_types() -> list[str]:
    """List supported FAISS index types.

    Returns:
        Sorted list of index type names.

    Examples:
        >>> types = list_faiss_index_types()
        >>> "flat" in types
        True
        >>> "hnsw" in types
        True
    """
    return sorted(VALID_FAISS_INDEX_TYPES)


def list_backends() -> list[str]:
    """List supported vector store backends.

    Returns:
        Sorted list of backend names.

    Examples:
        >>> backends = list_backends()
        >>> "faiss" in backends
        True
        >>> "chroma" in backends
        True
    """
    return sorted(VALID_BACKENDS)


def list_distance_functions() -> list[str]:
    """List supported distance functions.

    Returns:
        Sorted list of distance function names.

    Examples:
        >>> funcs = list_distance_functions()
        >>> "cosine" in funcs
        True
    """
    return sorted(VALID_DISTANCE_FUNCTIONS)


def get_faiss_index_type(name: str) -> FAISSIndexType:
    """Get FAISS index type from name.

    Args:
        name: Index type name.

    Returns:
        FAISSIndexType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_faiss_index_type("flat")
        <FAISSIndexType.FLAT: 'flat'>

        >>> get_faiss_index_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: index_type must be one of
    """
    if name not in VALID_FAISS_INDEX_TYPES:
        msg = f"index_type must be one of {VALID_FAISS_INDEX_TYPES}, got '{name}'"
        raise ValueError(msg)
    return FAISSIndexType(name)


def get_backend(name: str) -> VectorStoreBackend:
    """Get vector store backend from name.

    Args:
        name: Backend name.

    Returns:
        VectorStoreBackend enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_backend("faiss")
        <VectorStoreBackend.FAISS: 'faiss'>

        >>> get_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: backend must be one of
    """
    if name not in VALID_BACKENDS:
        msg = f"backend must be one of {VALID_BACKENDS}, got '{name}'"
        raise ValueError(msg)
    return VectorStoreBackend(name)


def get_distance_function(name: str) -> DistanceFunction:
    """Get distance function from name.

    Args:
        name: Distance function name.

    Returns:
        DistanceFunction enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_distance_function("cosine")
        <DistanceFunction.COSINE: 'cosine'>

        >>> get_distance_function("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: distance_function must be one of
    """
    if name not in VALID_DISTANCE_FUNCTIONS:
        msg = (
            f"distance_function must be one of {VALID_DISTANCE_FUNCTIONS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return DistanceFunction(name)


def estimate_index_size(
    num_vectors: int,
    dimension: int,
    index_type: FAISSIndexType,
    bytes_per_float: int = 4,
) -> int:
    """Estimate index size in bytes.

    Args:
        num_vectors: Number of vectors to index.
        dimension: Vector dimension.
        index_type: Type of FAISS index.
        bytes_per_float: Bytes per float value. Defaults to 4.

    Returns:
        Estimated size in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> size = estimate_index_size(10000, 768, FAISSIndexType.FLAT)
        >>> size > 0
        True

        >>> estimate_index_size(0, 768, FAISSIndexType.FLAT)
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

    base_size = num_vectors * dimension * bytes_per_float

    # Add overhead based on index type
    overhead_factors = {
        FAISSIndexType.FLAT: 1.0,
        FAISSIndexType.IVF_FLAT: 1.1,
        FAISSIndexType.IVF_PQ: 0.25,  # PQ compresses significantly
        FAISSIndexType.HNSW: 1.5,  # Graph structure adds overhead
        FAISSIndexType.LSH: 0.5,  # Hash-based compression
    }

    factor = overhead_factors.get(index_type, 1.0)
    return int(base_size * factor)


def calculate_optimal_nlist(num_vectors: int) -> int:
    """Calculate optimal nlist for IVF index.

    Args:
        num_vectors: Expected number of vectors.

    Returns:
        Recommended nlist value.

    Raises:
        ValueError: If num_vectors is invalid.

    Examples:
        >>> nlist = calculate_optimal_nlist(100000)
        >>> 100 <= nlist <= 1000
        True

        >>> calculate_optimal_nlist(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_vectors must be positive
    """
    if num_vectors <= 0:
        msg = f"num_vectors must be positive, got {num_vectors}"
        raise ValueError(msg)

    # Rule of thumb: sqrt(n) for nlist
    import math

    nlist = int(math.sqrt(num_vectors))

    # Clamp to reasonable range
    return max(16, min(nlist, 4096))


def calculate_optimal_nprobe(nlist: int, recall_target: float = 0.9) -> int:
    """Calculate optimal nprobe for IVF index.

    Args:
        nlist: Number of clusters.
        recall_target: Target recall (0-1). Defaults to 0.9.

    Returns:
        Recommended nprobe value.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> nprobe = calculate_optimal_nprobe(100)
        >>> 1 <= nprobe <= 100
        True

        >>> calculate_optimal_nprobe(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: nlist must be positive
    """
    if nlist <= 0:
        msg = f"nlist must be positive, got {nlist}"
        raise ValueError(msg)

    if not 0.0 < recall_target <= 1.0:
        msg = f"recall_target must be in (0, 1], got {recall_target}"
        raise ValueError(msg)

    # Higher recall needs more probes
    # Approximate: nprobe = nlist * (1 - (1 - recall)^(1/k)) where k ~ 3
    base_ratio = recall_target**2
    nprobe = max(1, int(nlist * base_ratio * 0.5))

    return min(nprobe, nlist)
