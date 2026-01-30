"""Sentence embedding utilities for semantic search and similarity.

This module provides functions for creating and using sentence embeddings
with the sentence-transformers library for semantic search and similarity.

Examples:
    >>> from hf_gtc.inference.embeddings import create_embedding_config
    >>> config = create_embedding_config(normalize=True)
    >>> config.normalize
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class PoolingMode(Enum):
    """Supported pooling modes for sentence embeddings.

    Attributes:
        MEAN: Mean pooling of token embeddings.
        CLS: Use CLS token embedding.
        MAX: Max pooling of token embeddings.
        MEAN_SQRT_LEN: Mean pooling with sqrt(length) normalization.

    Examples:
        >>> PoolingMode.MEAN.value
        'mean'
        >>> PoolingMode.CLS.value
        'cls'
    """

    MEAN = "mean"
    CLS = "cls"
    MAX = "max"
    MEAN_SQRT_LEN = "mean_sqrt_len"


VALID_POOLING_MODES = frozenset(p.value for p in PoolingMode)

# Distance metrics for similarity search
DistanceMetric = Literal["cosine", "euclidean", "dot_product"]
VALID_DISTANCE_METRICS = frozenset({"cosine", "euclidean", "dot_product"})


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for sentence embeddings.

    Attributes:
        model_name: Name of the embedding model.
        max_seq_length: Maximum sequence length.
        normalize: Whether to normalize embeddings.
        pooling_mode: Pooling strategy for token embeddings.
        device: Device to use for inference.
        batch_size: Batch size for encoding.

    Examples:
        >>> config = EmbeddingConfig(
        ...     model_name="all-MiniLM-L6-v2",
        ...     max_seq_length=256,
        ...     normalize=True,
        ...     pooling_mode=PoolingMode.MEAN,
        ...     device="cpu",
        ...     batch_size=32,
        ... )
        >>> config.normalize
        True
    """

    model_name: str
    max_seq_length: int
    normalize: bool
    pooling_mode: PoolingMode
    device: str
    batch_size: int


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


def validate_embedding_config(config: EmbeddingConfig) -> None:
    """Validate embedding configuration parameters.

    Args:
        config: Embedding configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = EmbeddingConfig(
        ...     model_name="all-MiniLM-L6-v2",
        ...     max_seq_length=256,
        ...     normalize=True,
        ...     pooling_mode=PoolingMode.MEAN,
        ...     device="cpu",
        ...     batch_size=32,
        ... )
        >>> validate_embedding_config(config)  # No error

        >>> bad_config = EmbeddingConfig(
        ...     model_name="",
        ...     max_seq_length=256,
        ...     normalize=True,
        ...     pooling_mode=PoolingMode.MEAN,
        ...     device="cpu",
        ...     batch_size=32,
        ... )
        >>> validate_embedding_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not config.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if config.max_seq_length <= 0:
        msg = f"max_seq_length must be positive, got {config.max_seq_length}"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)


def create_embedding_config(
    model_name: str = "all-MiniLM-L6-v2",
    max_seq_length: int = 256,
    normalize: bool = True,
    pooling_mode: str = "mean",
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingConfig:
    """Create an embedding configuration.

    Args:
        model_name: Model identifier. Defaults to "all-MiniLM-L6-v2".
        max_seq_length: Maximum sequence length. Defaults to 256.
        normalize: Whether to normalize embeddings. Defaults to True.
        pooling_mode: Pooling strategy. Defaults to "mean".
        device: Device for inference. Defaults to "cpu".
        batch_size: Encoding batch size. Defaults to 32.

    Returns:
        EmbeddingConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_embedding_config(normalize=True)
        >>> config.normalize
        True

        >>> config = create_embedding_config(pooling_mode="cls")
        >>> config.pooling_mode
        <PoolingMode.CLS: 'cls'>

        >>> create_embedding_config(model_name="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if pooling_mode not in VALID_POOLING_MODES:
        msg = f"pooling_mode must be one of {VALID_POOLING_MODES}, got '{pooling_mode}'"
        raise ValueError(msg)

    config = EmbeddingConfig(
        model_name=model_name,
        max_seq_length=max_seq_length,
        normalize=normalize,
        pooling_mode=PoolingMode(pooling_mode),
        device=device,
        batch_size=batch_size,
    )
    validate_embedding_config(config)
    return config


def list_pooling_modes() -> list[str]:
    """List all supported pooling modes.

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
    return sorted(VALID_POOLING_MODES)


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
    if len(vec_a) == 0 or len(vec_b) == 0:
        msg = "vectors cannot be empty"
        raise ValueError(msg)

    if len(vec_a) != len(vec_b):
        msg = f"vectors must have same length: {len(vec_a)} != {len(vec_b)}"
        raise ValueError(msg)

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


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
    if len(vec_a) == 0 or len(vec_b) == 0:
        msg = "vectors cannot be empty"
        raise ValueError(msg)

    if len(vec_a) != len(vec_b):
        msg = f"vectors must have same length: {len(vec_a)} != {len(vec_b)}"
        raise ValueError(msg)

    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b, strict=True)) ** 0.5


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
    if len(vec_a) == 0 or len(vec_b) == 0:
        msg = "vectors cannot be empty"
        raise ValueError(msg)

    if len(vec_a) != len(vec_b):
        msg = f"vectors must have same length: {len(vec_a)} != {len(vec_b)}"
        raise ValueError(msg)

    return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))


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
    precision: Literal["fp32", "fp16"] = "fp32",
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

        >>> chunk_text("", chunk_size=10)
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
