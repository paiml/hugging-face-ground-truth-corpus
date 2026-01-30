"""Embedding models and pooling strategies for semantic search and similarity.

This module provides comprehensive utilities for creating and using sentence embeddings
with various pooling strategies, normalization methods, and similarity metrics.

Examples:
    >>> from hf_gtc.inference.embeddings import create_embedding_config
    >>> config = create_embedding_config(normalization="l2")
    >>> config.normalization
    <EmbeddingNormalization.L2: 'l2'>

    >>> from hf_gtc.inference.embeddings import calculate_similarity
    >>> sim = calculate_similarity((1.0, 0.0), (1.0, 0.0), metric="cosine")
    >>> round(sim, 4)
    1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class PoolingStrategy(Enum):
    """Strategy for pooling token embeddings into sentence embeddings.

    Attributes:
        CLS: Use CLS token embedding as sentence representation.
        MEAN: Mean pooling of all token embeddings.
        MAX: Max pooling of token embeddings.
        WEIGHTED_MEAN: Weighted mean pooling with attention weights.
        LAST_TOKEN: Use last token embedding (useful for GPT-style models).

    Examples:
        >>> PoolingStrategy.CLS.value
        'cls'
        >>> PoolingStrategy.MEAN.value
        'mean'
        >>> PoolingStrategy.WEIGHTED_MEAN.value
        'weighted_mean'
    """

    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    WEIGHTED_MEAN = "weighted_mean"
    LAST_TOKEN = "last_token"


VALID_POOLING_STRATEGIES = frozenset(s.value for s in PoolingStrategy)


class EmbeddingNormalization(Enum):
    """Normalization method for embeddings.

    Attributes:
        L2: L2 (Euclidean) normalization to unit length.
        UNIT: Alias for L2 normalization (unit vectors).
        NONE: No normalization applied.

    Examples:
        >>> EmbeddingNormalization.L2.value
        'l2'
        >>> EmbeddingNormalization.NONE.value
        'none'
    """

    L2 = "l2"
    UNIT = "unit"
    NONE = "none"


VALID_EMBEDDING_NORMALIZATIONS = frozenset(n.value for n in EmbeddingNormalization)


class SimilarityMetric(Enum):
    """Similarity metric for comparing embeddings.

    Attributes:
        COSINE: Cosine similarity (normalized dot product).
        DOT_PRODUCT: Raw dot product similarity.
        EUCLIDEAN: Euclidean distance (converted to similarity).
        MANHATTAN: Manhattan distance (converted to similarity).

    Examples:
        >>> SimilarityMetric.COSINE.value
        'cosine'
        >>> SimilarityMetric.DOT_PRODUCT.value
        'dot_product'
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


VALID_SIMILARITY_METRICS = frozenset(m.value for m in SimilarityMetric)

# Type aliases
PoolingStrategyStr = Literal["cls", "mean", "max", "weighted_mean", "last_token"]
EmbeddingNormalizationStr = Literal["l2", "unit", "none"]
SimilarityMetricStr = Literal["cosine", "dot_product", "euclidean", "manhattan"]
TaskTypeStr = Literal["general", "qa", "clustering", "classification", "retrieval"]
PrecisionStr = Literal["fp32", "fp16"]


@dataclass(frozen=True, slots=True)
class PoolingConfig:
    """Configuration for embedding pooling.

    Attributes:
        strategy: Pooling strategy to use.
        layer_weights: Weights for combining layers (for multi-layer pooling).
        attention_mask: Whether to use attention mask during pooling.

    Examples:
        >>> config = PoolingConfig(
        ...     strategy=PoolingStrategy.MEAN,
        ...     layer_weights=None,
        ...     attention_mask=True,
        ... )
        >>> config.strategy
        <PoolingStrategy.MEAN: 'mean'>
        >>> config.attention_mask
        True
    """

    strategy: PoolingStrategy
    layer_weights: tuple[float, ...] | None
    attention_mask: bool


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for embedding models.

    Attributes:
        model_name: Name of the embedding model.
        pooling_config: Pooling configuration.
        normalization: Embedding normalization method.
        dimension: Embedding dimension.
        max_length: Maximum sequence length.

    Examples:
        >>> pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        >>> config = EmbeddingConfig(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     pooling_config=pooling,
        ...     normalization=EmbeddingNormalization.L2,
        ...     dimension=384,
        ...     max_length=512,
        ... )
        >>> config.dimension
        384
        >>> config.normalization
        <EmbeddingNormalization.L2: 'l2'>
    """

    model_name: str
    pooling_config: PoolingConfig
    normalization: EmbeddingNormalization
    dimension: int
    max_length: int


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result from embedding computation.

    Attributes:
        embeddings: The computed embedding vectors.
        tokens: Token IDs from tokenization.
        attention_mask: Attention mask for the tokens.
        pooled: Pooled embedding (if pooling was applied).

    Examples:
        >>> result = EmbeddingResult(
        ...     embeddings=((0.1, 0.2), (0.3, 0.4)),
        ...     tokens=(101, 2003, 102),
        ...     attention_mask=(1, 1, 1),
        ...     pooled=(0.2, 0.3),
        ... )
        >>> len(result.embeddings)
        2
        >>> result.pooled
        (0.2, 0.3)
    """

    embeddings: tuple[tuple[float, ...], ...]
    tokens: tuple[int, ...]
    attention_mask: tuple[int, ...]
    pooled: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class EmbeddingStats:
    """Statistics about embedding quality and characteristics.

    Attributes:
        dimension: Embedding dimension.
        vocab_coverage: Fraction of vocabulary covered (0-1).
        avg_magnitude: Average magnitude of embeddings.
        isotropy_score: Measure of embedding space uniformity (0-1).

    Examples:
        >>> stats = EmbeddingStats(
        ...     dimension=384,
        ...     vocab_coverage=0.95,
        ...     avg_magnitude=1.0,
        ...     isotropy_score=0.85,
        ... )
        >>> stats.dimension
        384
        >>> stats.isotropy_score
        0.85
    """

    dimension: int
    vocab_coverage: float
    avg_magnitude: float
    isotropy_score: float


def validate_pooling_config(config: PoolingConfig) -> None:
    """Validate pooling configuration.

    Args:
        config: PoolingConfig to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PoolingConfig(PoolingStrategy.MEAN, None, True)
        >>> validate_pooling_config(config)  # No error

        >>> config = PoolingConfig(PoolingStrategy.MEAN, (0.5, 0.5), True)
        >>> validate_pooling_config(config)  # No error

        >>> config = PoolingConfig(PoolingStrategy.MEAN, (), True)
        >>> validate_pooling_config(config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_weights cannot be empty if provided
    """
    if config.layer_weights is not None:
        if len(config.layer_weights) == 0:
            msg = "layer_weights cannot be empty if provided"
            raise ValueError(msg)

        weight_sum = sum(config.layer_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            msg = f"layer_weights must sum to 1.0, got {weight_sum}"
            raise ValueError(msg)

        for i, w in enumerate(config.layer_weights):
            if w < 0:
                msg = f"layer_weights[{i}] cannot be negative, got {w}"
                raise ValueError(msg)


def validate_embedding_config(config: EmbeddingConfig) -> None:
    """Validate embedding configuration.

    Args:
        config: EmbeddingConfig to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> pooling = PoolingConfig(PoolingStrategy.MEAN, None, True)
        >>> config = EmbeddingConfig(
        ...     model_name="all-MiniLM-L6-v2",
        ...     pooling_config=pooling,
        ...     normalization=EmbeddingNormalization.L2,
        ...     dimension=384,
        ...     max_length=512,
        ... )
        >>> validate_embedding_config(config)  # No error

        >>> bad_config = EmbeddingConfig(
        ...     model_name="",
        ...     pooling_config=pooling,
        ...     normalization=EmbeddingNormalization.L2,
        ...     dimension=384,
        ...     max_length=512,
        ... )
        >>> validate_embedding_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not config.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if config.dimension <= 0:
        msg = f"dimension must be positive, got {config.dimension}"
        raise ValueError(msg)

    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    validate_pooling_config(config.pooling_config)


def validate_embedding_result(result: EmbeddingResult) -> None:
    """Validate embedding result.

    Args:
        result: EmbeddingResult to validate.

    Raises:
        ValueError: If result is invalid.

    Examples:
        >>> result = EmbeddingResult(
        ...     embeddings=((0.1, 0.2),),
        ...     tokens=(101, 102),
        ...     attention_mask=(1, 1),
        ...     pooled=(0.1, 0.2),
        ... )
        >>> validate_embedding_result(result)  # No error

        >>> bad_result = EmbeddingResult(
        ...     embeddings=(),
        ...     tokens=(101,),
        ...     attention_mask=(1,),
        ...     pooled=None,
        ... )
        >>> validate_embedding_result(bad_result)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: embeddings cannot be empty
    """
    if len(result.embeddings) == 0:
        msg = "embeddings cannot be empty"
        raise ValueError(msg)

    if len(result.tokens) == 0:
        msg = "tokens cannot be empty"
        raise ValueError(msg)

    if len(result.attention_mask) != len(result.tokens):
        msg = (
            f"attention_mask length ({len(result.attention_mask)}) "
            f"must match tokens length ({len(result.tokens)})"
        )
        raise ValueError(msg)


def validate_embedding_stats(stats: EmbeddingStats) -> None:
    """Validate embedding statistics.

    Args:
        stats: EmbeddingStats to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = EmbeddingStats(384, 0.95, 1.0, 0.85)
        >>> validate_embedding_stats(stats)  # No error

        >>> bad_stats = EmbeddingStats(0, 0.95, 1.0, 0.85)
        >>> validate_embedding_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dimension must be positive
    """
    if stats.dimension <= 0:
        msg = f"dimension must be positive, got {stats.dimension}"
        raise ValueError(msg)

    if not 0.0 <= stats.vocab_coverage <= 1.0:
        msg = f"vocab_coverage must be in [0, 1], got {stats.vocab_coverage}"
        raise ValueError(msg)

    if stats.avg_magnitude < 0:
        msg = f"avg_magnitude cannot be negative, got {stats.avg_magnitude}"
        raise ValueError(msg)

    if not 0.0 <= stats.isotropy_score <= 1.0:
        msg = f"isotropy_score must be in [0, 1], got {stats.isotropy_score}"
        raise ValueError(msg)


def create_pooling_config(
    strategy: PoolingStrategyStr = "mean",
    layer_weights: tuple[float, ...] | None = None,
    attention_mask: bool = True,
) -> PoolingConfig:
    """Create a pooling configuration.

    Args:
        strategy: Pooling strategy. Defaults to "mean".
        layer_weights: Weights for combining layers. Defaults to None.
        attention_mask: Whether to use attention mask. Defaults to True.

    Returns:
        PoolingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_pooling_config(strategy="cls")
        >>> config.strategy
        <PoolingStrategy.CLS: 'cls'>

        >>> config = create_pooling_config(attention_mask=False)
        >>> config.attention_mask
        False

        >>> create_pooling_config(
        ...     strategy="invalid",
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if strategy not in VALID_POOLING_STRATEGIES:
        msg = (
            f"strategy must be one of {VALID_POOLING_STRATEGIES}, "
            f"got '{strategy}'"
        )
        raise ValueError(msg)

    config = PoolingConfig(
        strategy=PoolingStrategy(strategy),
        layer_weights=layer_weights,
        attention_mask=attention_mask,
    )
    validate_pooling_config(config)
    return config


def create_embedding_config(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    pooling_strategy: PoolingStrategyStr = "mean",
    normalization: EmbeddingNormalizationStr = "l2",
    dimension: int = 384,
    max_length: int = 512,
    attention_mask: bool = True,
) -> EmbeddingConfig:
    """Create an embedding configuration.

    Args:
        model_name: Model identifier.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        pooling_strategy: Pooling strategy. Defaults to "mean".
        normalization: Normalization method. Defaults to "l2".
        dimension: Embedding dimension. Defaults to 384.
        max_length: Maximum sequence length. Defaults to 512.
        attention_mask: Whether to use attention mask. Defaults to True.

    Returns:
        EmbeddingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_embedding_config(normalization="l2")
        >>> config.normalization
        <EmbeddingNormalization.L2: 'l2'>

        >>> config = create_embedding_config(pooling_strategy="cls")
        >>> config.pooling_config.strategy
        <PoolingStrategy.CLS: 'cls'>

        >>> create_embedding_config(model_name="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if normalization not in VALID_EMBEDDING_NORMALIZATIONS:
        msg = (
            f"normalization must be one of {VALID_EMBEDDING_NORMALIZATIONS}, "
            f"got '{normalization}'"
        )
        raise ValueError(msg)

    pooling_config = create_pooling_config(
        strategy=pooling_strategy,
        layer_weights=None,
        attention_mask=attention_mask,
    )

    config = EmbeddingConfig(
        model_name=model_name,
        pooling_config=pooling_config,
        normalization=EmbeddingNormalization(normalization),
        dimension=dimension,
        max_length=max_length,
    )
    validate_embedding_config(config)
    return config


def create_embedding_result(
    embeddings: tuple[tuple[float, ...], ...],
    tokens: tuple[int, ...],
    attention_mask: tuple[int, ...],
    pooled: tuple[float, ...] | None = None,
) -> EmbeddingResult:
    """Create an embedding result.

    Args:
        embeddings: Computed embedding vectors.
        tokens: Token IDs from tokenization.
        attention_mask: Attention mask for tokens.
        pooled: Pooled embedding. Defaults to None.

    Returns:
        EmbeddingResult with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = create_embedding_result(
        ...     embeddings=((0.1, 0.2), (0.3, 0.4)),
        ...     tokens=(101, 2003, 102),
        ...     attention_mask=(1, 1, 1),
        ...     pooled=(0.2, 0.3),
        ... )
        >>> len(result.embeddings)
        2

        >>> create_embedding_result(
        ...     (), (101,), (1,),
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: embeddings cannot be empty
    """
    result = EmbeddingResult(
        embeddings=embeddings,
        tokens=tokens,
        attention_mask=attention_mask,
        pooled=pooled,
    )
    validate_embedding_result(result)
    return result


def create_embedding_stats(
    dimension: int = 384,
    vocab_coverage: float = 1.0,
    avg_magnitude: float = 1.0,
    isotropy_score: float = 0.5,
) -> EmbeddingStats:
    """Create embedding statistics.

    Args:
        dimension: Embedding dimension. Defaults to 384.
        vocab_coverage: Vocabulary coverage (0-1). Defaults to 1.0.
        avg_magnitude: Average embedding magnitude. Defaults to 1.0.
        isotropy_score: Isotropy score (0-1). Defaults to 0.5.

    Returns:
        EmbeddingStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_embedding_stats(dimension=768)
        >>> stats.dimension
        768

        >>> stats = create_embedding_stats(isotropy_score=0.9)
        >>> stats.isotropy_score
        0.9

        >>> create_embedding_stats(dimension=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dimension must be positive
    """
    stats = EmbeddingStats(
        dimension=dimension,
        vocab_coverage=vocab_coverage,
        avg_magnitude=avg_magnitude,
        isotropy_score=isotropy_score,
    )
    validate_embedding_stats(stats)
    return stats


def list_pooling_strategies() -> list[str]:
    """List all available pooling strategies.

    Returns:
        Sorted list of pooling strategy names.

    Examples:
        >>> strategies = list_pooling_strategies()
        >>> "mean" in strategies
        True
        >>> "cls" in strategies
        True
        >>> "weighted_mean" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_POOLING_STRATEGIES)


def list_embedding_normalizations() -> list[str]:
    """List all available embedding normalizations.

    Returns:
        Sorted list of normalization names.

    Examples:
        >>> norms = list_embedding_normalizations()
        >>> "l2" in norms
        True
        >>> "none" in norms
        True
        >>> norms == sorted(norms)
        True
    """
    return sorted(VALID_EMBEDDING_NORMALIZATIONS)


def list_similarity_metrics() -> list[str]:
    """List all available similarity metrics.

    Returns:
        Sorted list of similarity metric names.

    Examples:
        >>> metrics = list_similarity_metrics()
        >>> "cosine" in metrics
        True
        >>> "euclidean" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_SIMILARITY_METRICS)


def get_pooling_strategy(name: str) -> PoolingStrategy:
    """Get a pooling strategy by name.

    Args:
        name: Name of the pooling strategy.

    Returns:
        The corresponding PoolingStrategy enum value.

    Raises:
        ValueError: If name is not a valid pooling strategy.

    Examples:
        >>> get_pooling_strategy("mean")
        <PoolingStrategy.MEAN: 'mean'>
        >>> get_pooling_strategy("cls")
        <PoolingStrategy.CLS: 'cls'>

        >>> get_pooling_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown pooling strategy
    """
    if name not in VALID_POOLING_STRATEGIES:
        msg = (
            f"Unknown pooling strategy: '{name}'. "
            f"Valid: {VALID_POOLING_STRATEGIES}"
        )
        raise ValueError(msg)
    return PoolingStrategy(name)


def get_embedding_normalization(name: str) -> EmbeddingNormalization:
    """Get an embedding normalization by name.

    Args:
        name: Name of the normalization.

    Returns:
        The corresponding EmbeddingNormalization enum value.

    Raises:
        ValueError: If name is not a valid normalization.

    Examples:
        >>> get_embedding_normalization("l2")
        <EmbeddingNormalization.L2: 'l2'>
        >>> get_embedding_normalization("none")
        <EmbeddingNormalization.NONE: 'none'>

        >>> get_embedding_normalization("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown embedding normalization
    """
    if name not in VALID_EMBEDDING_NORMALIZATIONS:
        msg = (
            f"Unknown embedding normalization: '{name}'. "
            f"Valid: {VALID_EMBEDDING_NORMALIZATIONS}"
        )
        raise ValueError(msg)
    return EmbeddingNormalization(name)


def get_similarity_metric(name: str) -> SimilarityMetric:
    """Get a similarity metric by name.

    Args:
        name: Name of the similarity metric.

    Returns:
        The corresponding SimilarityMetric enum value.

    Raises:
        ValueError: If name is not a valid similarity metric.

    Examples:
        >>> get_similarity_metric("cosine")
        <SimilarityMetric.COSINE: 'cosine'>
        >>> get_similarity_metric("dot_product")
        <SimilarityMetric.DOT_PRODUCT: 'dot_product'>

        >>> get_similarity_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown similarity metric
    """
    if name not in VALID_SIMILARITY_METRICS:
        msg = (
            f"Unknown similarity metric: '{name}'. "
            f"Valid: {VALID_SIMILARITY_METRICS}"
        )
        raise ValueError(msg)
    return SimilarityMetric(name)


def calculate_similarity(
    vec_a: tuple[float, ...],
    vec_b: tuple[float, ...],
    metric: SimilarityMetricStr = "cosine",
) -> float:
    """Calculate similarity between two vectors using the specified metric.

    Args:
        vec_a: First vector.
        vec_b: Second vector.
        metric: Similarity metric. Defaults to "cosine".

    Returns:
        Similarity score. For cosine: [-1, 1]. For dot_product: unbounded.
        For euclidean/manhattan: non-negative (lower means more similar).

    Raises:
        ValueError: If vectors are empty or have different lengths.
        ValueError: If metric is invalid.

    Examples:
        >>> a = (1.0, 0.0, 0.0)
        >>> b = (1.0, 0.0, 0.0)
        >>> round(calculate_similarity(a, b, "cosine"), 4)
        1.0

        >>> a = (1.0, 0.0)
        >>> b = (0.0, 1.0)
        >>> round(calculate_similarity(a, b, "cosine"), 4)
        0.0

        >>> a = (1.0, 2.0, 3.0)
        >>> b = (4.0, 5.0, 6.0)
        >>> round(calculate_similarity(a, b, "dot_product"), 4)
        32.0

        >>> calculate_similarity((), (), "cosine")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vectors cannot be empty
    """
    if len(vec_a) == 0 or len(vec_b) == 0:
        msg = "vectors cannot be empty"
        raise ValueError(msg)

    if len(vec_a) != len(vec_b):
        msg = f"vectors must have same length: {len(vec_a)} != {len(vec_b)}"
        raise ValueError(msg)

    if metric not in VALID_SIMILARITY_METRICS:
        msg = f"metric must be one of {VALID_SIMILARITY_METRICS}, got '{metric}'"
        raise ValueError(msg)

    if metric == "cosine":
        return _cosine_similarity(vec_a, vec_b)
    elif metric == "dot_product":
        return _dot_product(vec_a, vec_b)
    elif metric == "euclidean":
        return _euclidean_distance(vec_a, vec_b)
    else:  # manhattan
        return _manhattan_distance(vec_a, vec_b)


def _cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _dot_product(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate dot product between two vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))


def _euclidean_distance(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b, strict=True)) ** 0.5


def _manhattan_distance(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate Manhattan distance between two vectors."""
    return sum(abs(a - b) for a, b in zip(vec_a, vec_b, strict=True))


def compute_pooled_embedding(
    embeddings: tuple[tuple[float, ...], ...],
    attention_mask: tuple[int, ...] | None = None,
    strategy: PoolingStrategyStr = "mean",
    weights: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Compute pooled embedding from token embeddings.

    Args:
        embeddings: Token embeddings as tuple of tuples.
        attention_mask: Mask indicating valid tokens (1) vs padding (0).
        strategy: Pooling strategy. Defaults to "mean".
        weights: Optional weights for weighted_mean pooling.

    Returns:
        Pooled embedding vector.

    Raises:
        ValueError: If embeddings are empty or parameters are invalid.

    Examples:
        >>> emb = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        >>> pooled = compute_pooled_embedding(emb, strategy="mean")
        >>> round(pooled[0], 4)
        3.0
        >>> round(pooled[1], 4)
        4.0

        >>> pooled = compute_pooled_embedding(emb, strategy="cls")
        >>> pooled
        (1.0, 2.0)

        >>> pooled = compute_pooled_embedding(emb, strategy="max")
        >>> pooled
        (5.0, 6.0)

        >>> emb = ((1.0, 2.0), (3.0, 4.0))
        >>> pooled = compute_pooled_embedding(emb, strategy="last_token")
        >>> pooled
        (3.0, 4.0)

        >>> compute_pooled_embedding(
        ...     (), strategy="mean",
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: embeddings cannot be empty
    """
    if len(embeddings) == 0:
        msg = "embeddings cannot be empty"
        raise ValueError(msg)

    if strategy not in VALID_POOLING_STRATEGIES:
        msg = f"strategy must be one of {VALID_POOLING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    dim = len(embeddings[0])

    if strategy == "cls":
        return embeddings[0]

    if strategy == "last_token":
        return embeddings[-1]

    if strategy == "max":
        result = list(embeddings[0])
        for emb in embeddings[1:]:
            for i, val in enumerate(emb):
                if val > result[i]:
                    result[i] = val
        return tuple(result)

    if strategy == "mean":
        if attention_mask is not None:
            # Masked mean pooling
            masked_sum = [0.0] * dim
            mask_count = 0
            for emb, mask in zip(embeddings, attention_mask, strict=False):
                if mask == 1:
                    for i, val in enumerate(emb):
                        masked_sum[i] += val
                    mask_count += 1
            if mask_count == 0:
                return tuple(0.0 for _ in range(dim))
            return tuple(s / mask_count for s in masked_sum)
        else:
            # Simple mean pooling
            sums = [0.0] * dim
            for emb in embeddings:
                for i, val in enumerate(emb):
                    sums[i] += val
            return tuple(s / len(embeddings) for s in sums)

    # weighted_mean strategy
    if weights is None:
        msg = "weights required for weighted_mean strategy"
        raise ValueError(msg)

    if len(weights) != len(embeddings):
        msg = (
            f"weights length ({len(weights)}) must match "
            f"embeddings length ({len(embeddings)})"
        )
        raise ValueError(msg)

    weighted_sum = [0.0] * dim
    weight_total = sum(weights)
    if weight_total == 0:
        return tuple(0.0 for _ in range(dim))

    for emb, w in zip(embeddings, weights, strict=True):
        for i, val in enumerate(emb):
            weighted_sum[i] += val * w

    return tuple(s / weight_total for s in weighted_sum)


def normalize_embeddings(
    embeddings: tuple[tuple[float, ...], ...],
    normalization: EmbeddingNormalizationStr = "l2",
) -> tuple[tuple[float, ...], ...]:
    """Normalize embedding vectors.

    Args:
        embeddings: Embedding vectors to normalize.
        normalization: Normalization method. Defaults to "l2".

    Returns:
        Normalized embedding vectors.

    Raises:
        ValueError: If embeddings are empty or normalization is invalid.

    Examples:
        >>> emb = ((3.0, 4.0),)
        >>> norm = normalize_embeddings(emb, "l2")
        >>> round(norm[0][0], 4)
        0.6
        >>> round(norm[0][1], 4)
        0.8

        >>> emb = ((1.0, 2.0),)
        >>> norm = normalize_embeddings(emb, "none")
        >>> norm
        ((1.0, 2.0),)

        >>> normalize_embeddings((), "l2")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: embeddings cannot be empty
    """
    if len(embeddings) == 0:
        msg = "embeddings cannot be empty"
        raise ValueError(msg)

    if normalization not in VALID_EMBEDDING_NORMALIZATIONS:
        msg = (
            f"normalization must be one of {VALID_EMBEDDING_NORMALIZATIONS}, "
            f"got '{normalization}'"
        )
        raise ValueError(msg)

    if normalization == "none":
        return embeddings

    # L2 or unit normalization (same thing)
    normalized = []
    for emb in embeddings:
        magnitude = sum(x * x for x in emb) ** 0.5
        if magnitude == 0:
            normalized.append(emb)
        else:
            normalized.append(tuple(x / magnitude for x in emb))

    return tuple(normalized)


def estimate_embedding_quality(
    embeddings: tuple[tuple[float, ...], ...],
) -> EmbeddingStats:
    """Estimate quality metrics for a set of embeddings.

    Computes dimension, average magnitude, and isotropy score.
    Isotropy measures how uniformly distributed embeddings are in space.

    Args:
        embeddings: Embedding vectors to analyze.

    Returns:
        EmbeddingStats with quality metrics.

    Raises:
        ValueError: If embeddings are empty.

    Examples:
        >>> emb = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        >>> stats = estimate_embedding_quality(emb)
        >>> stats.dimension
        2
        >>> stats.avg_magnitude
        1.0

        >>> estimate_embedding_quality(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: embeddings cannot be empty
    """
    if len(embeddings) == 0:
        msg = "embeddings cannot be empty"
        raise ValueError(msg)

    dimension = len(embeddings[0])

    # Calculate average magnitude
    magnitudes = []
    for emb in embeddings:
        mag = sum(x * x for x in emb) ** 0.5
        magnitudes.append(mag)
    avg_magnitude = sum(magnitudes) / len(magnitudes)

    # Calculate isotropy score (simplified: based on embedding variance)
    # Higher isotropy means embeddings are more uniformly distributed
    if len(embeddings) < 2:
        isotropy_score = 0.5
    else:
        # Calculate centroid
        centroid = [0.0] * dimension
        for emb in embeddings:
            for i, val in enumerate(emb):
                centroid[i] += val
        centroid = [c / len(embeddings) for c in centroid]

        # Calculate average cosine similarity to centroid
        # Low similarity = high isotropy
        centroid_mag = sum(c * c for c in centroid) ** 0.5
        if centroid_mag < 1e-10:
            isotropy_score = 1.0  # Perfect isotropy (centroid at origin)
        else:
            total_cos = 0.0
            for emb in embeddings:
                emb_mag = sum(e * e for e in emb) ** 0.5
                if emb_mag > 0:
                    dot = sum(e * c for e, c in zip(emb, centroid, strict=True))
                    cos = dot / (emb_mag * centroid_mag)
                    total_cos += abs(cos)
            avg_cos = total_cos / len(embeddings)
            # Convert: low avg_cos = high isotropy
            isotropy_score = max(0.0, min(1.0, 1.0 - avg_cos))

    return EmbeddingStats(
        dimension=dimension,
        vocab_coverage=1.0,  # Cannot compute without vocabulary
        avg_magnitude=avg_magnitude,
        isotropy_score=isotropy_score,
    )


def format_embedding_stats(stats: EmbeddingStats) -> str:
    """Format embedding statistics as a human-readable string.

    Args:
        stats: EmbeddingStats to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = EmbeddingStats(384, 0.95, 1.0, 0.85)
        >>> formatted = format_embedding_stats(stats)
        >>> "Dimension: 384" in formatted
        True
        >>> "Vocab Coverage: 95.00%" in formatted
        True
        >>> "Isotropy Score: 0.85" in formatted
        True

        >>> stats = EmbeddingStats(768, 1.0, 0.98, 0.72)
        >>> "Dimension: 768" in format_embedding_stats(stats)
        True
    """
    lines = [
        f"Dimension: {stats.dimension}",
        f"Vocab Coverage: {stats.vocab_coverage * 100:.2f}%",
        f"Avg Magnitude: {stats.avg_magnitude:.4f}",
        f"Isotropy Score: {stats.isotropy_score:.2f}",
    ]
    return "\n".join(lines)


def get_recommended_embedding_config(
    task: TaskTypeStr = "general",
) -> EmbeddingConfig:
    """Get recommended embedding configuration for a task.

    Args:
        task: Task type. Defaults to "general".

    Returns:
        Recommended EmbeddingConfig for the task.

    Raises:
        ValueError: If task is not recognized.

    Examples:
        >>> config = get_recommended_embedding_config("general")
        >>> config.model_name
        'sentence-transformers/all-MiniLM-L6-v2'
        >>> config.dimension
        384

        >>> config = get_recommended_embedding_config("retrieval")
        >>> config.model_name
        'sentence-transformers/all-mpnet-base-v2'
        >>> config.dimension
        768

        >>> get_recommended_embedding_config(
        ...     "invalid",
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown task
    """
    valid_tasks = {"general", "qa", "clustering", "classification", "retrieval"}
    if task not in valid_tasks:
        msg = f"Unknown task: '{task}'. Valid: {valid_tasks}"
        raise ValueError(msg)

    task_configs: dict[str, dict[str, object]] = {
        "general": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "pooling_strategy": "mean",
            "normalization": "l2",
            "dimension": 384,
            "max_length": 256,
        },
        "qa": {
            "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "pooling_strategy": "mean",
            "normalization": "l2",
            "dimension": 384,
            "max_length": 512,
        },
        "clustering": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "pooling_strategy": "mean",
            "normalization": "l2",
            "dimension": 768,
            "max_length": 384,
        },
        "classification": {
            "model_name": "sentence-transformers/all-MiniLM-L12-v2",
            "pooling_strategy": "cls",
            "normalization": "l2",
            "dimension": 384,
            "max_length": 256,
        },
        "retrieval": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "pooling_strategy": "mean",
            "normalization": "l2",
            "dimension": 768,
            "max_length": 512,
        },
    }

    cfg = task_configs[task]
    return create_embedding_config(
        model_name=str(cfg["model_name"]),
        pooling_strategy=cfg["pooling_strategy"],  # type: ignore[arg-type]
        normalization=cfg["normalization"],  # type: ignore[arg-type]
        dimension=int(cfg["dimension"]),  # type: ignore[arg-type]
        max_length=int(cfg["max_length"]),  # type: ignore[arg-type]
    )


# Backward compatibility aliases
PoolingMode = PoolingStrategy
VALID_POOLING_MODES = VALID_POOLING_STRATEGIES
DistanceMetric = Literal["cosine", "euclidean", "dot_product"]
VALID_DISTANCE_METRICS = frozenset({"cosine", "euclidean", "dot_product"})


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Result of similarity search.

    Attributes:
        indices: Indices of similar items.
        scores: Similarity scores.
        texts: Optional matched texts.

    Examples:
        >>> result = SimilarityResult(
        ...     indices=(0, 5, 2),
        ...     scores=(0.95, 0.87, 0.82),
        ...     texts=("text1", "text2", "text3"),
        ... )
        >>> len(result.indices)
        3
    """

    indices: tuple[int, ...]
    scores: tuple[float, ...]
    texts: tuple[str, ...] | None


def list_pooling_modes() -> list[str]:
    """List all supported pooling modes (alias for list_pooling_strategies).

    Returns:
        Sorted list of pooling mode names.

    Examples:
        >>> modes = list_pooling_modes()
        >>> "mean" in modes
        True
        >>> "cls" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return list_pooling_strategies()


def list_distance_metrics() -> list[str]:
    """List all supported distance metrics.

    Returns:
        Sorted list of distance metric names.

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


def cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity score in range [-1, 1].

    Raises:
        ValueError: If vectors have different lengths or are empty.

    Examples:
        >>> a = (1.0, 0.0, 0.0)
        >>> b = (1.0, 0.0, 0.0)
        >>> round(cosine_similarity(a, b), 4)
        1.0

        >>> a = (1.0, 0.0)
        >>> b = (0.0, 1.0)
        >>> round(cosine_similarity(a, b), 4)
        0.0

        >>> cosine_similarity((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vectors cannot be empty
    """
    return calculate_similarity(vec_a, vec_b, metric="cosine")


def euclidean_distance(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate Euclidean distance between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Euclidean distance (always non-negative).

    Raises:
        ValueError: If vectors have different lengths or are empty.

    Examples:
        >>> a = (0.0, 0.0)
        >>> b = (3.0, 4.0)
        >>> round(euclidean_distance(a, b), 4)
        5.0

        >>> a = (1.0, 1.0)
        >>> b = (1.0, 1.0)
        >>> round(euclidean_distance(a, b), 4)
        0.0

        >>> euclidean_distance((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vectors cannot be empty
    """
    return calculate_similarity(vec_a, vec_b, metric="euclidean")


def dot_product_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calculate dot product between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Dot product value.

    Raises:
        ValueError: If vectors have different lengths or are empty.

    Examples:
        >>> a = (1.0, 2.0, 3.0)
        >>> b = (4.0, 5.0, 6.0)
        >>> round(dot_product_similarity(a, b), 4)
        32.0

        >>> dot_product_similarity((), ())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vectors cannot be empty
    """
    return calculate_similarity(vec_a, vec_b, metric="dot_product")


def normalize_vector(vec: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize a vector to unit length.

    Args:
        vec: Vector to normalize.

    Returns:
        Normalized vector with unit length.

    Raises:
        ValueError: If vector is empty or has zero magnitude.

    Examples:
        >>> v = (3.0, 4.0)
        >>> norm = normalize_vector(v)
        >>> round(norm[0], 4)
        0.6
        >>> round(norm[1], 4)
        0.8

        >>> normalize_vector(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: vector cannot be empty
    """
    if len(vec) == 0:
        msg = "vector cannot be empty"
        raise ValueError(msg)

    magnitude = sum(x * x for x in vec) ** 0.5

    if magnitude == 0:
        msg = "cannot normalize zero vector"
        raise ValueError(msg)

    return tuple(x / magnitude for x in vec)


def get_recommended_model(task: str) -> str:
    """Get recommended embedding model for a task.

    Args:
        task: Task type ("general", "qa", "clustering", "classification").

    Returns:
        Recommended model name.

    Raises:
        ValueError: If task is not recognized.

    Examples:
        >>> get_recommended_model("general")
        'all-MiniLM-L6-v2'
        >>> get_recommended_model("qa")
        'multi-qa-MiniLM-L6-cos-v1'
    """
    valid_tasks = {"general", "qa", "clustering", "classification"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    model_map = {
        "general": "all-MiniLM-L6-v2",
        "qa": "multi-qa-MiniLM-L6-cos-v1",
        "clustering": "all-mpnet-base-v2",
        "classification": "all-MiniLM-L12-v2",
    }
    return model_map[task]


def estimate_embedding_memory(
    num_vectors: int,
    embedding_dim: int = 384,
    precision: PrecisionStr = "fp32",
) -> float:
    """Estimate memory usage for storing embeddings.

    Args:
        num_vectors: Number of embedding vectors.
        embedding_dim: Dimension of embeddings. Defaults to 384.
        precision: Storage precision. Defaults to "fp32".

    Returns:
        Estimated memory usage in megabytes.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> mem = estimate_embedding_memory(1000, embedding_dim=384)
        >>> mem > 0
        True

        >>> mem_fp16 = estimate_embedding_memory(1000, precision="fp16")
        >>> mem_fp32 = estimate_embedding_memory(1000, precision="fp32")
        >>> mem_fp16 < mem_fp32
        True

        >>> estimate_embedding_memory(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_vectors must be positive
    """
    if num_vectors <= 0:
        msg = f"num_vectors must be positive, got {num_vectors}"
        raise ValueError(msg)

    if embedding_dim <= 0:
        msg = f"embedding_dim must be positive, got {embedding_dim}"
        raise ValueError(msg)

    valid_precisions = {"fp32", "fp16"}
    if precision not in valid_precisions:
        msg = f"precision must be one of {valid_precisions}, got '{precision}'"
        raise ValueError(msg)

    bytes_per_element = 4 if precision == "fp32" else 2
    total_bytes = num_vectors * embedding_dim * bytes_per_element

    return total_bytes / (1024 * 1024)  # Convert to MB


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Args:
        text: Text to chunk.
        chunk_size: Maximum tokens per chunk. Defaults to 256.
        overlap: Token overlap between chunks. Defaults to 50.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> text = " ".join(["word"] * 100)
        >>> chunks = chunk_text(text, chunk_size=30, overlap=5)
        >>> len(chunks) > 1
        True

        >>> chunk_text("short text", chunk_size=100)
        ['short text']

        >>> chunk_text("", chunk_size=10, overlap=0)
        []
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)

    if overlap < 0:
        msg = f"overlap cannot be negative, got {overlap}"
        raise ValueError(msg)

    if overlap >= chunk_size:
        msg = f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        raise ValueError(msg)

    if not text:
        return []

    # Simple word-based chunking (approximate tokens)
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks
