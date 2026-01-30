"""Reranking and hybrid search utilities.

This module provides functions for reranking search results and
implementing hybrid search with score fusion techniques.

Examples:
    >>> from hf_gtc.rag.reranking import create_reranker_config
    >>> config = create_reranker_config(model_id="BAAI/bge-reranker-base")
    >>> config.model_id
    'BAAI/bge-reranker-base'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RerankerType(Enum):
    """Supported reranker model types.

    Attributes:
        CROSS_ENCODER: Cross-encoder models that jointly encode query and document.
        BI_ENCODER: Bi-encoder models with separate query/document encoders.
        COLBERT: ColBERT late interaction models.
        BM25: Traditional BM25 sparse retrieval scoring.

    Examples:
        >>> RerankerType.CROSS_ENCODER.value
        'cross_encoder'
        >>> RerankerType.COLBERT.value
        'colbert'
    """

    CROSS_ENCODER = "cross_encoder"
    BI_ENCODER = "bi_encoder"
    COLBERT = "colbert"
    BM25 = "bm25"


VALID_RERANKER_TYPES = frozenset(t.value for t in RerankerType)


class FusionMethod(Enum):
    """Score fusion methods for hybrid search.

    Attributes:
        RRF: Reciprocal Rank Fusion - combines ranks from multiple rankers.
        LINEAR: Linear combination of normalized scores.
        WEIGHTED: Weighted sum of scores with custom weights.

    Examples:
        >>> FusionMethod.RRF.value
        'rrf'
        >>> FusionMethod.LINEAR.value
        'linear'
    """

    RRF = "rrf"
    LINEAR = "linear"
    WEIGHTED = "weighted"


VALID_FUSION_METHODS = frozenset(m.value for m in FusionMethod)


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Configuration for a reranker model.

    Attributes:
        model_id: HuggingFace model identifier.
        reranker_type: Type of reranker model.
        max_length: Maximum sequence length for encoding.
        batch_size: Batch size for inference.

    Examples:
        >>> config = RerankerConfig(
        ...     model_id="BAAI/bge-reranker-base",
        ...     reranker_type=RerankerType.CROSS_ENCODER,
        ...     max_length=512,
        ...     batch_size=32,
        ... )
        >>> config.model_id
        'BAAI/bge-reranker-base'
    """

    model_id: str
    reranker_type: RerankerType
    max_length: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Configuration for score fusion.

    Attributes:
        method: Fusion method to use.
        weights: Tuple of weights for weighted fusion.
        k: RRF constant (default 60).

    Examples:
        >>> config = FusionConfig(
        ...     method=FusionMethod.RRF,
        ...     weights=(0.5, 0.5),
        ...     k=60,
        ... )
        >>> config.method
        <FusionMethod.RRF: 'rrf'>
    """

    method: FusionMethod
    weights: tuple[float, ...]
    k: int


@dataclass(frozen=True, slots=True)
class RerankerResult:
    """Result from reranking a document.

    Attributes:
        document_id: Unique document identifier.
        original_score: Score before reranking.
        reranked_score: Score after reranking.
        rank: Position in reranked results (1-indexed).

    Examples:
        >>> result = RerankerResult(
        ...     document_id="doc_1",
        ...     original_score=0.75,
        ...     reranked_score=0.92,
        ...     rank=1,
        ... )
        >>> result.reranked_score
        0.92
    """

    document_id: str
    original_score: float
    reranked_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class HybridSearchConfig:
    """Configuration for hybrid search.

    Attributes:
        dense_weight: Weight for dense retrieval scores.
        sparse_weight: Weight for sparse retrieval scores.
        fusion_method: Method for combining scores.

    Examples:
        >>> config = HybridSearchConfig(
        ...     dense_weight=0.7,
        ...     sparse_weight=0.3,
        ...     fusion_method=FusionMethod.WEIGHTED,
        ... )
        >>> config.dense_weight
        0.7
    """

    dense_weight: float
    sparse_weight: float
    fusion_method: FusionMethod


def validate_reranker_config(config: RerankerConfig) -> None:
    """Validate reranker configuration.

    Args:
        config: Reranker configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = RerankerConfig(
        ...     "BAAI/bge-reranker-base", RerankerType.CROSS_ENCODER, 512, 32
        ... )
        >>> validate_reranker_config(config)  # No error

        >>> bad = RerankerConfig("", RerankerType.CROSS_ENCODER, 512, 32)
        >>> validate_reranker_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id cannot be empty
    """
    if not config.model_id:
        msg = "model_id cannot be empty"
        raise ValueError(msg)

    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)


def validate_fusion_config(config: FusionConfig) -> None:
    """Validate fusion configuration.

    Args:
        config: Fusion configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = FusionConfig(FusionMethod.RRF, (0.5, 0.5), 60)
        >>> validate_fusion_config(config)  # No error

        >>> bad = FusionConfig(FusionMethod.RRF, (), 60)
        >>> validate_fusion_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: weights cannot be empty
    """
    if not config.weights:
        msg = "weights cannot be empty"
        raise ValueError(msg)

    for i, weight in enumerate(config.weights):
        if weight < 0.0:
            msg = f"weight at index {i} must be non-negative, got {weight}"
            raise ValueError(msg)

    if config.k <= 0:
        msg = f"k must be positive, got {config.k}"
        raise ValueError(msg)


def validate_hybrid_search_config(config: HybridSearchConfig) -> None:
    """Validate hybrid search configuration.

    Args:
        config: Hybrid search configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = HybridSearchConfig(0.7, 0.3, FusionMethod.WEIGHTED)
        >>> validate_hybrid_search_config(config)  # No error

        >>> bad = HybridSearchConfig(-0.1, 0.3, FusionMethod.WEIGHTED)
        >>> validate_hybrid_search_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dense_weight must be non-negative
    """
    if config.dense_weight < 0.0:
        msg = f"dense_weight must be non-negative, got {config.dense_weight}"
        raise ValueError(msg)

    if config.sparse_weight < 0.0:
        msg = f"sparse_weight must be non-negative, got {config.sparse_weight}"
        raise ValueError(msg)

    total = config.dense_weight + config.sparse_weight
    if total <= 0.0:
        msg = "dense_weight + sparse_weight must be positive"
        raise ValueError(msg)


def create_reranker_config(
    model_id: str = "BAAI/bge-reranker-base",
    reranker_type: str = "cross_encoder",
    max_length: int = 512,
    batch_size: int = 32,
) -> RerankerConfig:
    """Create a reranker configuration.

    Args:
        model_id: HuggingFace model ID. Defaults to "BAAI/bge-reranker-base".
        reranker_type: Type of reranker. Defaults to "cross_encoder".
        max_length: Maximum sequence length. Defaults to 512.
        batch_size: Inference batch size. Defaults to 32.

    Returns:
        RerankerConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_reranker_config(model_id="ms-marco-MiniLM-L-6")
        >>> config.model_id
        'ms-marco-MiniLM-L-6'

        >>> config = create_reranker_config(reranker_type="colbert")
        >>> config.reranker_type
        <RerankerType.COLBERT: 'colbert'>

        >>> create_reranker_config(model_id="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id cannot be empty
    """
    if reranker_type not in VALID_RERANKER_TYPES:
        msg = (
            f"reranker_type must be one of {VALID_RERANKER_TYPES}, "
            f"got '{reranker_type}'"
        )
        raise ValueError(msg)

    config = RerankerConfig(
        model_id=model_id,
        reranker_type=RerankerType(reranker_type),
        max_length=max_length,
        batch_size=batch_size,
    )
    validate_reranker_config(config)
    return config


def create_fusion_config(
    method: str = "rrf",
    weights: tuple[float, ...] = (0.5, 0.5),
    k: int = 60,
) -> FusionConfig:
    """Create a fusion configuration.

    Args:
        method: Fusion method. Defaults to "rrf".
        weights: Weights for fusion. Defaults to (0.5, 0.5).
        k: RRF constant. Defaults to 60.

    Returns:
        FusionConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_fusion_config(method="linear", weights=(0.6, 0.4))
        >>> config.method
        <FusionMethod.LINEAR: 'linear'>

        >>> config = create_fusion_config(k=100)
        >>> config.k
        100

        >>> create_fusion_config(method="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if method not in VALID_FUSION_METHODS:
        msg = f"method must be one of {VALID_FUSION_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = FusionConfig(
        method=FusionMethod(method),
        weights=weights,
        k=k,
    )
    validate_fusion_config(config)
    return config


def create_hybrid_search_config(
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    fusion_method: str = "weighted",
) -> HybridSearchConfig:
    """Create a hybrid search configuration.

    Args:
        dense_weight: Weight for dense retrieval. Defaults to 0.7.
        sparse_weight: Weight for sparse retrieval. Defaults to 0.3.
        fusion_method: Fusion method. Defaults to "weighted".

    Returns:
        HybridSearchConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_hybrid_search_config(dense_weight=0.8, sparse_weight=0.2)
        >>> config.dense_weight
        0.8

        >>> config = create_hybrid_search_config(fusion_method="rrf")
        >>> config.fusion_method
        <FusionMethod.RRF: 'rrf'>

        >>> create_hybrid_search_config(dense_weight=-0.1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dense_weight must be non-negative
    """
    if fusion_method not in VALID_FUSION_METHODS:
        msg = (
            f"fusion_method must be one of {VALID_FUSION_METHODS}, "
            f"got '{fusion_method}'"
        )
        raise ValueError(msg)

    config = HybridSearchConfig(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        fusion_method=FusionMethod(fusion_method),
    )
    validate_hybrid_search_config(config)
    return config


def create_reranker_result(
    document_id: str,
    original_score: float,
    reranked_score: float,
    rank: int,
) -> RerankerResult:
    """Create a reranker result.

    Args:
        document_id: Document identifier.
        original_score: Score before reranking.
        reranked_score: Score after reranking.
        rank: Position in reranked results.

    Returns:
        RerankerResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_reranker_result("doc_1", 0.75, 0.92, 1)
        >>> result.reranked_score
        0.92

        >>> create_reranker_result("", 0.75, 0.92, 1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: document_id cannot be empty
    """
    if not document_id:
        msg = "document_id cannot be empty"
        raise ValueError(msg)

    if rank < 1:
        msg = f"rank must be positive, got {rank}"
        raise ValueError(msg)

    return RerankerResult(
        document_id=document_id,
        original_score=original_score,
        reranked_score=reranked_score,
        rank=rank,
    )


def list_reranker_types() -> list[str]:
    """List supported reranker types.

    Returns:
        Sorted list of reranker type names.

    Examples:
        >>> types = list_reranker_types()
        >>> "cross_encoder" in types
        True
        >>> "colbert" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_RERANKER_TYPES)


def list_fusion_methods() -> list[str]:
    """List supported fusion methods.

    Returns:
        Sorted list of fusion method names.

    Examples:
        >>> methods = list_fusion_methods()
        >>> "rrf" in methods
        True
        >>> "linear" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_FUSION_METHODS)


def get_reranker_type(name: str) -> RerankerType:
    """Get reranker type from name.

    Args:
        name: Reranker type name.

    Returns:
        RerankerType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_reranker_type("cross_encoder")
        <RerankerType.CROSS_ENCODER: 'cross_encoder'>

        >>> get_reranker_type("colbert")
        <RerankerType.COLBERT: 'colbert'>

        >>> get_reranker_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: reranker_type must be one of
    """
    if name not in VALID_RERANKER_TYPES:
        msg = f"reranker_type must be one of {VALID_RERANKER_TYPES}, got '{name}'"
        raise ValueError(msg)
    return RerankerType(name)


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

        >>> get_fusion_method("linear")
        <FusionMethod.LINEAR: 'linear'>

        >>> get_fusion_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: fusion_method must be one of
    """
    if name not in VALID_FUSION_METHODS:
        msg = f"fusion_method must be one of {VALID_FUSION_METHODS}, got '{name}'"
        raise ValueError(msg)
    return FusionMethod(name)


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


def calculate_linear_fusion(
    scores: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
) -> float:
    """Calculate linear fusion of scores.

    Combines scores using weighted sum with optional normalization.

    Args:
        scores: Tuple of scores from different sources.
        weights: Tuple of weights (optional). If None, equal weights used.

    Returns:
        Combined score.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_linear_fusion((0.8, 0.6))
        0.7

        >>> calculate_linear_fusion((0.8, 0.6), weights=(0.7, 0.3))
        0.74

        >>> calculate_linear_fusion((1.0, 0.5, 0.3), weights=(0.5, 0.3, 0.2))
        0.71

        >>> calculate_linear_fusion(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores cannot be empty

        >>> calculate_linear_fusion((0.8, 0.6), weights=(0.7,))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores and weights must have the same length
    """
    if not scores:
        msg = "scores cannot be empty"
        raise ValueError(msg)

    if weights is None:
        # Equal weights
        weights = tuple(1.0 / len(scores) for _ in scores)

    if len(scores) != len(weights):
        msg = (
            f"scores and weights must have the same length, "
            f"got {len(scores)} scores and {len(weights)} weights"
        )
        raise ValueError(msg)

    if any(w < 0.0 for w in weights):
        msg = "all weights must be non-negative"
        raise ValueError(msg)

    total_weight = sum(weights)
    if total_weight <= 0.0:
        msg = "sum of weights must be positive"
        raise ValueError(msg)

    # Normalize weights and compute weighted sum
    weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
    return weighted_sum / total_weight
