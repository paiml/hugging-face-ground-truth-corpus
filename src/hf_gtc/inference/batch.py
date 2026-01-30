"""Batch inference utilities for efficient model inference.

This module provides utilities for performing batch inference with
HuggingFace models, including batching strategies and memory management.

Examples:
    >>> from hf_gtc.inference.batch import BatchConfig
    >>> config = BatchConfig(batch_size=32)
    >>> config.batch_size
    32
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Generic type variable for batch operations
T = TypeVar("T")


class PaddingStrategy(Enum):
    """Strategy for padding sequences in a batch.

    Attributes:
        LONGEST: Pad to the longest sequence in the batch.
        MAX_LENGTH: Pad to a fixed maximum length.
        DO_NOT_PAD: Do not pad sequences.

    Examples:
        >>> PaddingStrategy.LONGEST.value
        'longest'
        >>> PaddingStrategy.MAX_LENGTH.value
        'max_length'
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


VALID_PADDING_STRATEGIES = frozenset(p.value for p in PaddingStrategy)


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Configuration for batch inference.

    Attributes:
        batch_size: Number of samples per batch. Defaults to 32.
        max_length: Maximum sequence length. Defaults to 512.
        padding: Padding strategy. Defaults to LONGEST.
        truncation: Whether to truncate sequences. Defaults to True.
        return_tensors: Return format ("pt", "tf", "np"). Defaults to "pt".

    Examples:
        >>> config = BatchConfig(batch_size=16, max_length=256)
        >>> config.batch_size
        16
        >>> config.max_length
        256
    """

    batch_size: int = 32
    max_length: int = 512
    padding: PaddingStrategy = PaddingStrategy.LONGEST
    truncation: bool = True
    return_tensors: str = "pt"


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Result of batch inference.

    Attributes:
        predictions: List of model predictions.
        batch_index: Index of the batch in the sequence.
        num_samples: Number of samples in the batch.
        processing_time_ms: Time taken to process the batch in milliseconds.

    Examples:
        >>> result = BatchResult(
        ...     predictions=["label1", "label2"],
        ...     batch_index=0,
        ...     num_samples=2,
        ...     processing_time_ms=100.5,
        ... )
        >>> result.num_samples
        2
    """

    predictions: list[Any]
    batch_index: int
    num_samples: int
    processing_time_ms: float


@dataclass(frozen=True, slots=True)
class BatchStats:
    """Statistics from batch inference.

    Attributes:
        total_samples: Total number of samples processed.
        total_batches: Total number of batches processed.
        avg_batch_time_ms: Average time per batch in milliseconds.
        total_time_ms: Total processing time in milliseconds.
        samples_per_second: Throughput in samples per second.

    Examples:
        >>> stats = BatchStats(
        ...     total_samples=100,
        ...     total_batches=4,
        ...     avg_batch_time_ms=250.0,
        ...     total_time_ms=1000.0,
        ...     samples_per_second=100.0,
        ... )
        >>> stats.total_batches
        4
    """

    total_samples: int
    total_batches: int
    avg_batch_time_ms: float
    total_time_ms: float
    samples_per_second: float


def validate_batch_config(config: BatchConfig) -> None:
    """Validate batch configuration.

    Args:
        config: BatchConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If batch_size is not positive.
        ValueError: If max_length is not positive.
        ValueError: If return_tensors is not valid.

    Examples:
        >>> config = BatchConfig(batch_size=32, max_length=512)
        >>> validate_batch_config(config)  # No error

        >>> validate_batch_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = BatchConfig(batch_size=0)
        >>> validate_batch_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    valid_tensors = {"pt", "tf", "np", "jax"}
    if config.return_tensors not in valid_tensors:
        msg = (
            f"return_tensors must be one of {valid_tensors}, "
            f"got {config.return_tensors!r}"
        )
        raise ValueError(msg)


def create_batches(
    items: Sequence[T],
    batch_size: int,
) -> Iterator[list[T]]:
    """Split items into batches of specified size.

    Args:
        items: Sequence of items to batch.
        batch_size: Maximum number of items per batch.

    Yields:
        Lists of items, each with at most batch_size items.

    Raises:
        ValueError: If items is None.
        ValueError: If batch_size is not positive.

    Examples:
        >>> list(create_batches([1, 2, 3, 4, 5], batch_size=2))
        [[1, 2], [3, 4], [5]]
        >>> list(create_batches(["a", "b", "c"], batch_size=3))
        [['a', 'b', 'c']]
        >>> list(create_batches([], batch_size=10))
        []

        >>> list(create_batches(None, batch_size=2))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: items cannot be None
    """
    if items is None:
        msg = "items cannot be None"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def compute_num_batches(num_samples: int, batch_size: int) -> int:
    """Compute the number of batches needed for given samples.

    Args:
        num_samples: Total number of samples.
        batch_size: Number of samples per batch.

    Returns:
        Number of batches (ceiling division).

    Raises:
        ValueError: If num_samples is negative.
        ValueError: If batch_size is not positive.

    Examples:
        >>> compute_num_batches(100, 32)
        4
        >>> compute_num_batches(64, 32)
        2
        >>> compute_num_batches(65, 32)
        3
        >>> compute_num_batches(0, 32)
        0

        >>> compute_num_batches(-1, 32)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples cannot be negative
    """
    if num_samples < 0:
        msg = f"num_samples cannot be negative, got {num_samples}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if num_samples == 0:
        return 0

    return (num_samples + batch_size - 1) // batch_size


def compute_batch_stats(
    total_samples: int,
    total_batches: int,
    total_time_ms: float,
) -> BatchStats:
    """Compute batch inference statistics.

    Args:
        total_samples: Total number of samples processed.
        total_batches: Total number of batches processed.
        total_time_ms: Total processing time in milliseconds.

    Returns:
        BatchStats with computed metrics.

    Raises:
        ValueError: If total_samples is negative.
        ValueError: If total_batches is negative.
        ValueError: If total_time_ms is negative.

    Examples:
        >>> stats = compute_batch_stats(100, 4, 1000.0)
        >>> stats.total_samples
        100
        >>> stats.avg_batch_time_ms
        250.0
        >>> stats.samples_per_second
        100.0

        >>> compute_batch_stats(-1, 4, 1000.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_samples cannot be negative
    """
    if total_samples < 0:
        msg = f"total_samples cannot be negative, got {total_samples}"
        raise ValueError(msg)

    if total_batches < 0:
        msg = f"total_batches cannot be negative, got {total_batches}"
        raise ValueError(msg)

    if total_time_ms < 0:
        msg = f"total_time_ms cannot be negative, got {total_time_ms}"
        raise ValueError(msg)

    # Avoid division by zero
    avg_batch_time = total_time_ms / total_batches if total_batches > 0 else 0.0
    samples_per_sec = (
        total_samples / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0
    )

    return BatchStats(
        total_samples=total_samples,
        total_batches=total_batches,
        avg_batch_time_ms=avg_batch_time,
        total_time_ms=total_time_ms,
        samples_per_second=samples_per_sec,
    )


def estimate_memory_per_batch(
    batch_size: int,
    max_length: int,
    hidden_size: int = 768,
    dtype_bytes: int = 4,
) -> float:
    """Estimate memory usage per batch in megabytes.

    This provides a rough estimate based on sequence length and hidden size.
    Actual memory usage may vary based on model architecture.

    Args:
        batch_size: Number of samples per batch.
        max_length: Maximum sequence length.
        hidden_size: Model hidden dimension. Defaults to 768 (BERT base).
        dtype_bytes: Bytes per element. Defaults to 4 (float32).

    Returns:
        Estimated memory in megabytes.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> round(estimate_memory_per_batch(32, 512, 768, 4), 2)
        48.0
        >>> round(estimate_memory_per_batch(1, 512, 768, 4), 2)
        1.5

        >>> estimate_memory_per_batch(0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    # Rough estimate: batch_size * seq_len * hidden_size * dtype_bytes
    bytes_total = batch_size * max_length * hidden_size * dtype_bytes
    return bytes_total / (1024 * 1024)  # Convert to MB


def get_optimal_batch_size(
    available_memory_mb: float,
    max_length: int,
    hidden_size: int = 768,
    dtype_bytes: int = 4,
    memory_fraction: float = 0.8,
) -> int:
    """Calculate optimal batch size given available memory.

    Args:
        available_memory_mb: Available GPU/CPU memory in megabytes.
        max_length: Maximum sequence length.
        hidden_size: Model hidden dimension. Defaults to 768.
        dtype_bytes: Bytes per element. Defaults to 4.
        memory_fraction: Fraction of memory to use. Defaults to 0.8.

    Returns:
        Recommended batch size (at least 1).

    Raises:
        ValueError: If available_memory_mb is not positive.
        ValueError: If max_length is not positive.
        ValueError: If memory_fraction is not in (0, 1].

    Examples:
        >>> get_optimal_batch_size(30.0, 512, 768, 4)
        16
        >>> get_optimal_batch_size(1.0, 512, 768, 4)
        1

        >>> get_optimal_batch_size(0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: available_memory_mb must be positive
    """
    if available_memory_mb <= 0:
        msg = f"available_memory_mb must be positive, got {available_memory_mb}"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if not 0 < memory_fraction <= 1:
        msg = f"memory_fraction must be in (0, 1], got {memory_fraction}"
        raise ValueError(msg)

    usable_memory = available_memory_mb * memory_fraction

    # Calculate memory per sample
    memory_per_sample = estimate_memory_per_batch(
        1, max_length, hidden_size, dtype_bytes
    )

    # Calculate batch size
    batch_size = int(usable_memory / memory_per_sample)

    # Ensure at least batch size of 1
    return max(1, batch_size)


def list_padding_strategies() -> list[str]:
    """List all available padding strategies.

    Returns:
        Sorted list of padding strategy names.

    Examples:
        >>> strategies = list_padding_strategies()
        >>> "longest" in strategies
        True
        >>> "max_length" in strategies
        True
    """
    return sorted(VALID_PADDING_STRATEGIES)
