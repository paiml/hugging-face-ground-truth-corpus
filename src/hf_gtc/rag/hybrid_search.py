"""Hybrid search utilities combining dense and sparse retrieval.

This module provides functions for implementing hybrid search systems
that combine dense embedding-based retrieval with sparse methods like
BM25, TF-IDF, and SPLADE for improved retrieval performance.

Examples:
    >>> from hf_gtc.rag.hybrid_search import create_hybrid_config
    >>> config = create_hybrid_config(dense_weight=0.7, sparse_weight=0.3)
    >>> config.dense_weight
    0.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SparseMethod(Enum):
    """Supported sparse retrieval methods.

    Attributes:
        BM25: BM25 ranking algorithm (Okapi BM25).
        TFIDF: Term Frequency-Inverse Document Frequency.
        SPLADE: Sparse Lexical and Expansion model.

    Examples:
        >>> SparseMethod.BM25.value
        'bm25'
        >>> SparseMethod.SPLADE.value
        'splade'
    """

    BM25 = "bm25"
    TFIDF = "tfidf"
    SPLADE = "splade"


VALID_SPARSE_METHODS = frozenset(m.value for m in SparseMethod)


class FusionMethod(Enum):
    """Score fusion methods for combining dense and sparse rankings.

    Attributes:
        RRF: Reciprocal Rank Fusion - combines ranks from multiple rankers.
        LINEAR: Linear combination of normalized scores.
        CONVEX: Convex combination with constraints (weights sum to 1).
        LEARNED: Learned fusion weights via cross-validation.

    Examples:
        >>> FusionMethod.RRF.value
        'rrf'
        >>> FusionMethod.CONVEX.value
        'convex'
    """

    RRF = "rrf"
    LINEAR = "linear"
    CONVEX = "convex"
    LEARNED = "learned"


VALID_FUSION_METHODS = frozenset(m.value for m in FusionMethod)


class RetrievalMode(Enum):
    """Retrieval modes for hybrid search.

    Attributes:
        DENSE_ONLY: Use only dense embeddings for retrieval.
        SPARSE_ONLY: Use only sparse methods for retrieval.
        HYBRID: Combine dense and sparse retrieval.

    Examples:
        >>> RetrievalMode.HYBRID.value
        'hybrid'
        >>> RetrievalMode.DENSE_ONLY.value
        'dense_only'
    """

    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID = "hybrid"


VALID_RETRIEVAL_MODES = frozenset(m.value for m in RetrievalMode)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Configuration for BM25 algorithm.

    Attributes:
        k1: Term frequency saturation parameter. Higher values give more
            weight to term frequency. Typical range: 1.2-2.0.
        b: Length normalization parameter. 0 means no normalization,
            1 means full normalization. Typical value: 0.75.
        epsilon: Floor value for IDF to avoid negative scores.

    Examples:
        >>> config = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        >>> config.k1
        1.5
        >>> config.b
        0.75
    """

    k1: float
    b: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class SparseConfig:
    """Configuration for sparse retrieval.

    Attributes:
        method: Sparse retrieval method to use.
        bm25_config: BM25 configuration (used when method is BM25).
        vocab_size: Vocabulary size for term indexing.

    Examples:
        >>> bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        >>> config = SparseConfig(
        ...     method=SparseMethod.BM25,
        ...     bm25_config=bm25,
        ...     vocab_size=30000,
        ... )
        >>> config.method
        <SparseMethod.BM25: 'bm25'>
    """

    method: SparseMethod
    bm25_config: BM25Config
    vocab_size: int


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Configuration for hybrid search.

    Attributes:
        sparse_config: Configuration for sparse retrieval.
        dense_weight: Weight for dense retrieval scores (0.0-1.0).
        sparse_weight: Weight for sparse retrieval scores (0.0-1.0).
        fusion_method: Method for combining scores.
        top_k: Number of results to return.

    Examples:
        >>> bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        >>> sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        >>> config = HybridConfig(
        ...     sparse_config=sparse,
        ...     dense_weight=0.7,
        ...     sparse_weight=0.3,
        ...     fusion_method=FusionMethod.RRF,
        ...     top_k=10,
        ... )
        >>> config.dense_weight
        0.7
    """

    sparse_config: SparseConfig
    dense_weight: float
    sparse_weight: float
    fusion_method: FusionMethod
    top_k: int


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """Result from hybrid search.

    Attributes:
        doc_ids: Tuple of retrieved document identifiers.
        scores: Tuple of final fused scores.
        dense_scores: Tuple of dense retrieval scores.
        sparse_scores: Tuple of sparse retrieval scores.

    Examples:
        >>> result = HybridSearchResult(
        ...     doc_ids=("doc1", "doc2"),
        ...     scores=(0.92, 0.85),
        ...     dense_scores=(0.90, 0.80),
        ...     sparse_scores=(0.95, 0.88),
        ... )
        >>> result.doc_ids
        ('doc1', 'doc2')
    """

    doc_ids: tuple[str, ...]
    scores: tuple[float, ...]
    dense_scores: tuple[float, ...]
    sparse_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HybridStats:
    """Statistics for hybrid search performance.

    Attributes:
        dense_recall: Recall@k for dense retrieval alone.
        sparse_recall: Recall@k for sparse retrieval alone.
        hybrid_recall: Recall@k for hybrid retrieval.
        fusion_improvement: Relative improvement from fusion.

    Examples:
        >>> stats = HybridStats(
        ...     dense_recall=0.75,
        ...     sparse_recall=0.70,
        ...     hybrid_recall=0.85,
        ...     fusion_improvement=0.10,
        ... )
        >>> stats.hybrid_recall
        0.85
    """

    dense_recall: float
    sparse_recall: float
    hybrid_recall: float
    fusion_improvement: float


def validate_bm25_config(config: BM25Config) -> None:
    """Validate BM25 configuration.

    Args:
        config: BM25 configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        >>> validate_bm25_config(config)  # No error

        >>> bad = BM25Config(k1=-1.0, b=0.75, epsilon=0.25)
        >>> validate_bm25_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: k1 must be non-negative

        >>> bad = BM25Config(k1=1.5, b=1.5, epsilon=0.25)
        >>> validate_bm25_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: b must be between 0.0 and 1.0
    """
    if config.k1 < 0.0:
        msg = f"k1 must be non-negative, got {config.k1}"
        raise ValueError(msg)

    if not 0.0 <= config.b <= 1.0:
        msg = f"b must be between 0.0 and 1.0, got {config.b}"
        raise ValueError(msg)

    if config.epsilon < 0.0:
        msg = f"epsilon must be non-negative, got {config.epsilon}"
        raise ValueError(msg)


def validate_sparse_config(config: SparseConfig) -> None:
    """Validate sparse configuration.

    Args:
        config: Sparse configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        >>> config = SparseConfig(SparseMethod.BM25, bm25, 30000)
        >>> validate_sparse_config(config)  # No error

        >>> bad = SparseConfig(SparseMethod.BM25, bm25, 0)
        >>> validate_sparse_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vocab_size must be positive
    """
    validate_bm25_config(config.bm25_config)

    if config.vocab_size <= 0:
        msg = f"vocab_size must be positive, got {config.vocab_size}"
        raise ValueError(msg)


def validate_hybrid_config(config: HybridConfig) -> None:
    """Validate hybrid search configuration.

    Args:
        config: Hybrid search configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> bm25 = BM25Config(k1=1.5, b=0.75, epsilon=0.25)
        >>> sparse = SparseConfig(SparseMethod.BM25, bm25, 30000)
        >>> config = HybridConfig(sparse, 0.7, 0.3, FusionMethod.RRF, 10)
        >>> validate_hybrid_config(config)  # No error

        >>> bad = HybridConfig(sparse, -0.1, 0.3, FusionMethod.RRF, 10)
        >>> validate_hybrid_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dense_weight must be non-negative

        >>> bad = HybridConfig(sparse, 0.7, 0.3, FusionMethod.RRF, 0)
        >>> validate_hybrid_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: top_k must be positive
    """
    validate_sparse_config(config.sparse_config)

    if config.dense_weight < 0.0:
        msg = f"dense_weight must be non-negative, got {config.dense_weight}"
        raise ValueError(msg)

    if config.sparse_weight < 0.0:
        msg = f"sparse_weight must be non-negative, got {config.sparse_weight}"
        raise ValueError(msg)

    total_weight = config.dense_weight + config.sparse_weight
    if total_weight <= 0.0:
        msg = "dense_weight + sparse_weight must be positive"
        raise ValueError(msg)

    if config.top_k <= 0:
        msg = f"top_k must be positive, got {config.top_k}"
        raise ValueError(msg)


def validate_hybrid_search_result(result: HybridSearchResult) -> None:
    """Validate hybrid search result.

    Args:
        result: Hybrid search result to validate.

    Raises:
        ValueError: If result is invalid.

    Examples:
        >>> result = HybridSearchResult(
        ...     ("doc1",), (0.9,), (0.85,), (0.95,)
        ... )
        >>> validate_hybrid_search_result(result)  # No error

        >>> bad = HybridSearchResult(
        ...     ("doc1", "doc2"), (0.9,), (0.85,), (0.95,)
        ... )
        >>> validate_hybrid_search_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: doc_ids and scores must have the same length
    """
    n = len(result.doc_ids)

    if len(result.scores) != n:
        msg = (
            f"doc_ids and scores must have the same length, "
            f"got {n} doc_ids and {len(result.scores)} scores"
        )
        raise ValueError(msg)

    if len(result.dense_scores) != n:
        msg = (
            f"doc_ids and dense_scores must have the same length, "
            f"got {n} doc_ids and {len(result.dense_scores)} dense_scores"
        )
        raise ValueError(msg)

    if len(result.sparse_scores) != n:
        msg = (
            f"doc_ids and sparse_scores must have the same length, "
            f"got {n} doc_ids and {len(result.sparse_scores)} sparse_scores"
        )
        raise ValueError(msg)


def create_bm25_config(
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> BM25Config:
    """Create a BM25 configuration.

    Args:
        k1: Term frequency saturation. Defaults to 1.5.
        b: Length normalization. Defaults to 0.75.
        epsilon: IDF floor value. Defaults to 0.25.

    Returns:
        BM25Config with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_bm25_config()
        >>> config.k1
        1.5
        >>> config.b
        0.75

        >>> config = create_bm25_config(k1=2.0, b=0.8)
        >>> config.k1
        2.0

        >>> create_bm25_config(k1=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: k1 must be non-negative
    """
    config = BM25Config(k1=k1, b=b, epsilon=epsilon)
    validate_bm25_config(config)
    return config


def create_sparse_config(
    method: str = "bm25",
    bm25_config: BM25Config | None = None,
    vocab_size: int = 30000,
) -> SparseConfig:
    """Create a sparse retrieval configuration.

    Args:
        method: Sparse method to use. Defaults to "bm25".
        bm25_config: BM25 configuration. Defaults to None (uses defaults).
        vocab_size: Vocabulary size. Defaults to 30000.

    Returns:
        SparseConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_sparse_config()
        >>> config.method
        <SparseMethod.BM25: 'bm25'>
        >>> config.vocab_size
        30000

        >>> config = create_sparse_config(method="tfidf", vocab_size=50000)
        >>> config.method
        <SparseMethod.TFIDF: 'tfidf'>

        >>> create_sparse_config(method="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if method not in VALID_SPARSE_METHODS:
        msg = f"method must be one of {VALID_SPARSE_METHODS}, got '{method}'"
        raise ValueError(msg)

    if bm25_config is None:
        bm25_config = create_bm25_config()

    config = SparseConfig(
        method=SparseMethod(method),
        bm25_config=bm25_config,
        vocab_size=vocab_size,
    )
    validate_sparse_config(config)
    return config


def create_hybrid_config(
    sparse_config: SparseConfig | None = None,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    fusion_method: str = "rrf",
    top_k: int = 10,
) -> HybridConfig:
    """Create a hybrid search configuration.

    Args:
        sparse_config: Sparse retrieval config. Defaults to None (uses defaults).
        dense_weight: Weight for dense scores. Defaults to 0.7.
        sparse_weight: Weight for sparse scores. Defaults to 0.3.
        fusion_method: Score fusion method. Defaults to "rrf".
        top_k: Number of results to return. Defaults to 10.

    Returns:
        HybridConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_hybrid_config()
        >>> config.dense_weight
        0.7
        >>> config.sparse_weight
        0.3
        >>> config.fusion_method
        <FusionMethod.RRF: 'rrf'>

        >>> config = create_hybrid_config(dense_weight=0.6, sparse_weight=0.4)
        >>> config.dense_weight
        0.6

        >>> create_hybrid_config(fusion_method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: fusion_method must be one of
    """
    if fusion_method not in VALID_FUSION_METHODS:
        msg = (
            f"fusion_method must be one of {VALID_FUSION_METHODS}, "
            f"got '{fusion_method}'"
        )
        raise ValueError(msg)

    if sparse_config is None:
        sparse_config = create_sparse_config()

    config = HybridConfig(
        sparse_config=sparse_config,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        fusion_method=FusionMethod(fusion_method),
        top_k=top_k,
    )
    validate_hybrid_config(config)
    return config


def create_hybrid_search_result(
    doc_ids: tuple[str, ...],
    scores: tuple[float, ...],
    dense_scores: tuple[float, ...] | None = None,
    sparse_scores: tuple[float, ...] | None = None,
) -> HybridSearchResult:
    """Create a hybrid search result.

    Args:
        doc_ids: Document identifiers.
        scores: Final fused scores.
        dense_scores: Dense retrieval scores. Defaults to same as scores.
        sparse_scores: Sparse retrieval scores. Defaults to same as scores.

    Returns:
        HybridSearchResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_hybrid_search_result(
        ...     doc_ids=("doc1", "doc2"),
        ...     scores=(0.92, 0.85),
        ... )
        >>> result.doc_ids
        ('doc1', 'doc2')

        >>> result = create_hybrid_search_result(
        ...     ("doc1",), (0.9,), (0.85,), (0.95,)
        ... )
        >>> result.dense_scores
        (0.85,)

        >>> create_hybrid_search_result(
        ...     ("doc1", "doc2"), (0.9,)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: doc_ids and scores must have the same length
    """
    if dense_scores is None:
        dense_scores = scores
    if sparse_scores is None:
        sparse_scores = scores

    result = HybridSearchResult(
        doc_ids=doc_ids,
        scores=scores,
        dense_scores=dense_scores,
        sparse_scores=sparse_scores,
    )
    validate_hybrid_search_result(result)
    return result


def list_sparse_methods() -> list[str]:
    """List supported sparse retrieval methods.

    Returns:
        Sorted list of sparse method names.

    Examples:
        >>> methods = list_sparse_methods()
        >>> "bm25" in methods
        True
        >>> "splade" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_SPARSE_METHODS)


def list_fusion_methods() -> list[str]:
    """List supported fusion methods.

    Returns:
        Sorted list of fusion method names.

    Examples:
        >>> methods = list_fusion_methods()
        >>> "rrf" in methods
        True
        >>> "convex" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_FUSION_METHODS)


def list_retrieval_modes() -> list[str]:
    """List supported retrieval modes.

    Returns:
        Sorted list of retrieval mode names.

    Examples:
        >>> modes = list_retrieval_modes()
        >>> "hybrid" in modes
        True
        >>> "dense_only" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_RETRIEVAL_MODES)


def get_sparse_method(name: str) -> SparseMethod:
    """Get sparse method from name.

    Args:
        name: Sparse method name.

    Returns:
        SparseMethod enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_sparse_method("bm25")
        <SparseMethod.BM25: 'bm25'>

        >>> get_sparse_method("splade")
        <SparseMethod.SPLADE: 'splade'>

        >>> get_sparse_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sparse_method must be one of
    """
    if name not in VALID_SPARSE_METHODS:
        msg = f"sparse_method must be one of {VALID_SPARSE_METHODS}, got '{name}'"
        raise ValueError(msg)
    return SparseMethod(name)


def get_fusion_method(name: str) -> FusionMethod:
    """Get fusion method from name.

    Args:
        name: Fusion method name.

    Returns:
        FusionMethod enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_fusion_method("rrf")
        <FusionMethod.RRF: 'rrf'>

        >>> get_fusion_method("convex")
        <FusionMethod.CONVEX: 'convex'>

        >>> get_fusion_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: fusion_method must be one of
    """
    if name not in VALID_FUSION_METHODS:
        msg = f"fusion_method must be one of {VALID_FUSION_METHODS}, got '{name}'"
        raise ValueError(msg)
    return FusionMethod(name)


def get_retrieval_mode(name: str) -> RetrievalMode:
    """Get retrieval mode from name.

    Args:
        name: Retrieval mode name.

    Returns:
        RetrievalMode enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_retrieval_mode("hybrid")
        <RetrievalMode.HYBRID: 'hybrid'>

        >>> get_retrieval_mode("dense_only")
        <RetrievalMode.DENSE_ONLY: 'dense_only'>

        >>> get_retrieval_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: retrieval_mode must be one of
    """
    if name not in VALID_RETRIEVAL_MODES:
        msg = f"retrieval_mode must be one of {VALID_RETRIEVAL_MODES}, got '{name}'"
        raise ValueError(msg)
    return RetrievalMode(name)


def calculate_bm25_score(
    tf: int,
    df: int,
    doc_len: int,
    avg_doc_len: float,
    num_docs: int,
    config: BM25Config | None = None,
) -> float:
    """Calculate BM25 score for a term.

    BM25 formula:
    score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

    Args:
        tf: Term frequency in document.
        df: Document frequency (number of docs containing term).
        doc_len: Length of current document.
        avg_doc_len: Average document length in corpus.
        num_docs: Total number of documents in corpus.
        config: BM25 configuration. Defaults to None (uses defaults).

    Returns:
        BM25 score for the term.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> score = calculate_bm25_score(
        ...     tf=3, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
        ... )
        >>> score > 0
        True

        >>> score = calculate_bm25_score(
        ...     tf=0, df=100, doc_len=500, avg_doc_len=400.0, num_docs=10000
        ... )
        >>> score
        0.0

        >>> calculate_bm25_score(
        ...     tf=3, df=0, doc_len=500, avg_doc_len=400.0, num_docs=10000
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: df must be positive when tf is positive
    """
    if tf < 0:
        msg = f"tf must be non-negative, got {tf}"
        raise ValueError(msg)

    if tf == 0:
        return 0.0

    if df <= 0:
        msg = f"df must be positive when tf is positive, got {df}"
        raise ValueError(msg)

    if doc_len <= 0:
        msg = f"doc_len must be positive, got {doc_len}"
        raise ValueError(msg)

    if avg_doc_len <= 0.0:
        msg = f"avg_doc_len must be positive, got {avg_doc_len}"
        raise ValueError(msg)

    if num_docs <= 0:
        msg = f"num_docs must be positive, got {num_docs}"
        raise ValueError(msg)

    if df > num_docs:
        msg = f"df ({df}) cannot exceed num_docs ({num_docs})"
        raise ValueError(msg)

    if config is None:
        config = create_bm25_config()

    # Calculate IDF with floor
    idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
    idf = max(idf, config.epsilon)

    # Calculate normalized term frequency
    k1 = config.k1
    b = config.b
    len_norm = 1.0 - b + b * (doc_len / avg_doc_len)
    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * len_norm)

    return idf * tf_norm


def calculate_rrf_score(
    ranks: tuple[int, ...],
    k: int = 60,
) -> float:
    """Calculate Reciprocal Rank Fusion score.

    RRF combines rankings from multiple systems using the formula:
    RRF(d) = sum(1 / (k + rank_i)) for each ranker i.

    Args:
        ranks: Tuple of ranks from different rankers (1-indexed).
        k: Constant to prevent high scores for top ranks. Defaults to 60.

    Returns:
        Combined RRF score.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_rrf_score((1, 2))
        0.03252247488101534

        >>> calculate_rrf_score((1, 1, 1), k=60)
        0.04918032786885246

        >>> score1 = calculate_rrf_score((1,), k=60)
        >>> score2 = calculate_rrf_score((10,), k=60)
        >>> score1 > score2
        True

        >>> calculate_rrf_score(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ranks cannot be empty

        >>> calculate_rrf_score((0, 1))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: all ranks must be positive
    """
    if not ranks:
        msg = "ranks cannot be empty"
        raise ValueError(msg)

    if any(r < 1 for r in ranks):
        msg = "all ranks must be positive (1-indexed)"
        raise ValueError(msg)

    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)

    return sum(1.0 / (k + rank) for rank in ranks)


def fuse_rankings(
    dense_ranks: dict[str, int],
    sparse_ranks: dict[str, int],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    fusion_method: FusionMethod = FusionMethod.RRF,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse rankings from dense and sparse retrieval.

    Args:
        dense_ranks: Document ID to rank mapping from dense retrieval.
        sparse_ranks: Document ID to rank mapping from sparse retrieval.
        dense_weight: Weight for dense rankings. Defaults to 0.7.
        sparse_weight: Weight for sparse rankings. Defaults to 0.3.
        fusion_method: Method for combining rankings. Defaults to RRF.
        k: RRF constant. Defaults to 60.

    Returns:
        List of (doc_id, score) tuples sorted by score descending.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> dense = {"doc1": 1, "doc2": 2, "doc3": 3}
        >>> sparse = {"doc1": 2, "doc2": 1, "doc4": 3}
        >>> results = fuse_rankings(dense, sparse)
        >>> results[0][0] in ("doc1", "doc2")
        True

        >>> results = fuse_rankings({}, {})
        >>> results
        []

        >>> fuse_rankings({"doc1": 1}, {"doc1": 1}, dense_weight=-0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dense_weight must be non-negative
    """
    if dense_weight < 0.0:
        msg = f"dense_weight must be non-negative, got {dense_weight}"
        raise ValueError(msg)

    if sparse_weight < 0.0:
        msg = f"sparse_weight must be non-negative, got {sparse_weight}"
        raise ValueError(msg)

    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)

    # Get all unique document IDs
    all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    if not all_docs:
        return []

    scores: dict[str, float] = {}

    for doc_id in all_docs:
        dense_rank = dense_ranks.get(doc_id)
        sparse_rank = sparse_ranks.get(doc_id)

        if fusion_method == FusionMethod.RRF:
            # RRF scoring
            score = 0.0
            if dense_rank is not None:
                score += dense_weight * (1.0 / (k + dense_rank))
            if sparse_rank is not None:
                score += sparse_weight * (1.0 / (k + sparse_rank))
            scores[doc_id] = score

        elif fusion_method in (FusionMethod.LINEAR, FusionMethod.CONVEX):
            # Linear/Convex combination of inverse ranks
            max_rank = max(len(dense_ranks), len(sparse_ranks)) + 1
            dense_score = 1.0 / dense_rank if dense_rank else 1.0 / max_rank
            sparse_score = 1.0 / sparse_rank if sparse_rank else 1.0 / max_rank

            total_weight = dense_weight + sparse_weight
            if total_weight > 0:
                scores[doc_id] = (
                    dense_weight * dense_score + sparse_weight * sparse_score
                ) / total_weight
            else:
                scores[doc_id] = 0.0

        else:  # LEARNED - use equal weights as fallback
            score = 0.0
            if dense_rank is not None:
                score += 0.5 * (1.0 / (k + dense_rank))
            if sparse_rank is not None:
                score += 0.5 * (1.0 / (k + sparse_rank))
            scores[doc_id] = score

    # Sort by score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def optimize_fusion_weights(
    dense_recalls: tuple[float, ...],
    sparse_recalls: tuple[float, ...],
    hybrid_recalls: tuple[tuple[float, float, float], ...],
) -> tuple[float, float]:
    """Optimize fusion weights based on recall metrics.

    Finds the weight combination that maximizes hybrid recall
    improvement over the better of dense or sparse alone.

    Args:
        dense_recalls: Tuple of dense recall values for different queries.
        sparse_recalls: Tuple of sparse recall values for different queries.
        hybrid_recalls: Tuple of (dense_weight, sparse_weight, recall) tuples
            for different weight combinations.

    Returns:
        Tuple of (optimal_dense_weight, optimal_sparse_weight).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> dense = (0.7, 0.8, 0.75)
        >>> sparse = (0.65, 0.72, 0.80)
        >>> hybrid = (
        ...     (0.7, 0.3, 0.82),
        ...     (0.6, 0.4, 0.85),
        ...     (0.5, 0.5, 0.83),
        ... )
        >>> weights = optimize_fusion_weights(dense, sparse, hybrid)
        >>> weights[0] + weights[1]
        1.0

        >>> optimize_fusion_weights((), (), ())
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: hybrid_recalls cannot be empty
    """
    if not hybrid_recalls:
        msg = "hybrid_recalls cannot be empty"
        raise ValueError(msg)

    # Find the weight combination with best recall
    best_recall = -1.0
    best_weights = (0.5, 0.5)

    for dense_w, sparse_w, recall in hybrid_recalls:
        if recall > best_recall:
            best_recall = recall
            best_weights = (dense_w, sparse_w)

    # Normalize weights to sum to 1
    total = best_weights[0] + best_weights[1]
    if total > 0:
        return (best_weights[0] / total, best_weights[1] / total)

    return (0.5, 0.5)


def format_hybrid_stats(stats: HybridStats) -> str:
    """Format hybrid search statistics for display.

    Args:
        stats: Hybrid search statistics.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = HybridStats(0.75, 0.70, 0.85, 0.10)
        >>> formatted = format_hybrid_stats(stats)
        >>> "dense=0.750" in formatted
        True
        >>> "sparse=0.700" in formatted
        True
        >>> "hybrid=0.850" in formatted
        True
        >>> "+10.0%" in formatted
        True
    """
    improvement_pct = stats.fusion_improvement * 100

    return (
        f"Recall: dense={stats.dense_recall:.3f}, "
        f"sparse={stats.sparse_recall:.3f}, "
        f"hybrid={stats.hybrid_recall:.3f} "
        f"({'+' if improvement_pct >= 0 else ''}{improvement_pct:.1f}%)"
    )


def get_recommended_hybrid_config(
    corpus_size: int = 100000,
    avg_doc_length: int = 500,
    use_case: str = "general",
) -> HybridConfig:
    """Get recommended hybrid search configuration.

    Args:
        corpus_size: Number of documents in corpus. Defaults to 100000.
        avg_doc_length: Average document length in tokens. Defaults to 500.
        use_case: Use case type ("general", "qa", "semantic", "lexical").

    Returns:
        HybridConfig optimized for the use case.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_hybrid_config()
        >>> config.fusion_method
        <FusionMethod.RRF: 'rrf'>

        >>> config = get_recommended_hybrid_config(use_case="semantic")
        >>> config.dense_weight > config.sparse_weight
        True

        >>> config = get_recommended_hybrid_config(use_case="lexical")
        >>> config.sparse_weight > config.dense_weight
        True

        >>> get_recommended_hybrid_config(corpus_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: corpus_size must be positive
    """
    if corpus_size <= 0:
        msg = f"corpus_size must be positive, got {corpus_size}"
        raise ValueError(msg)

    if avg_doc_length <= 0:
        msg = f"avg_doc_length must be positive, got {avg_doc_length}"
        raise ValueError(msg)

    # Adjust BM25 parameters based on document length
    if avg_doc_length < 200:
        k1 = 1.2
        b = 0.5  # Less length normalization for short docs
    elif avg_doc_length > 1000:
        k1 = 2.0
        b = 0.9  # More length normalization for long docs
    else:
        k1 = 1.5
        b = 0.75

    bm25_config = create_bm25_config(k1=k1, b=b)

    # Adjust vocab size based on corpus size
    vocab_size = min(50000, max(10000, corpus_size // 10))

    sparse_config = create_sparse_config(
        method="bm25",
        bm25_config=bm25_config,
        vocab_size=vocab_size,
    )

    # Set weights based on use case
    use_case_weights = {
        "general": (0.6, 0.4),
        "qa": (0.7, 0.3),  # Semantic matching more important
        "semantic": (0.8, 0.2),  # Heavy semantic focus
        "lexical": (0.3, 0.7),  # Keyword matching more important
    }

    dense_w, sparse_w = use_case_weights.get(use_case, (0.6, 0.4))

    # Larger corpora benefit more from sparse retrieval
    if corpus_size > 500000:
        sparse_w = min(sparse_w + 0.1, 0.5)
        dense_w = 1.0 - sparse_w

    # Choose fusion method
    fusion_method = "rrf" if corpus_size > 10000 else "linear"

    # Set top_k based on corpus size
    top_k = min(20, max(5, corpus_size // 10000))

    return create_hybrid_config(
        sparse_config=sparse_config,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
        fusion_method=fusion_method,
        top_k=top_k,
    )
