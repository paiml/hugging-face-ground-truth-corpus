"""Dataset streaming utilities for large-scale data processing.

This module provides utilities for streaming datasets from HuggingFace Hub,
enabling processing of datasets that don't fit in memory.

Examples:
    >>> from hf_gtc.preprocessing.streaming import StreamConfig
    >>> config = StreamConfig(batch_size=1000)
    >>> config.batch_size
    1000
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Generic type variables for streaming functions
T = TypeVar("T")
U = TypeVar("U")


class ShuffleMode(Enum):
    """Mode for shuffling streamed data.

    Attributes:
        DISABLED: No shuffling.
        BUFFER: Shuffle within a buffer window.
        FULL: Full shuffle (requires loading into memory).

    Examples:
        >>> ShuffleMode.DISABLED.value
        'disabled'
        >>> ShuffleMode.BUFFER.value
        'buffer'
    """

    DISABLED = "disabled"
    BUFFER = "buffer"
    FULL = "full"


VALID_SHUFFLE_MODES = frozenset(s.value for s in ShuffleMode)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Configuration for dataset streaming.

    Attributes:
        batch_size: Number of samples per batch. Defaults to 1000.
        buffer_size: Size of shuffle buffer. Defaults to 10000.
        shuffle_mode: How to shuffle data. Defaults to BUFFER.
        prefetch_batches: Number of batches to prefetch. Defaults to 2.
        num_workers: Number of worker processes. Defaults to 0 (main process).

    Examples:
        >>> config = StreamConfig(batch_size=500, buffer_size=5000)
        >>> config.batch_size
        500
        >>> config.buffer_size
        5000
    """

    batch_size: int = 1000
    buffer_size: int = 10000
    shuffle_mode: ShuffleMode = ShuffleMode.BUFFER
    prefetch_batches: int = 2
    num_workers: int = 0


@dataclass(frozen=True, slots=True)
class StreamStats:
    """Statistics from streaming operation.

    Attributes:
        total_samples: Total samples processed.
        total_batches: Total batches processed.
        bytes_read: Total bytes read from source.
        samples_per_second: Processing throughput.

    Examples:
        >>> stats = StreamStats(
        ...     total_samples=10000,
        ...     total_batches=10,
        ...     bytes_read=1024000,
        ...     samples_per_second=5000.0,
        ... )
        >>> stats.total_samples
        10000
    """

    total_samples: int
    total_batches: int
    bytes_read: int
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class StreamProgress:
    """Progress information during streaming.

    Attributes:
        samples_processed: Number of samples processed so far.
        batches_processed: Number of batches processed so far.
        estimated_total: Estimated total samples (if known).
        percent_complete: Completion percentage (if total known).

    Examples:
        >>> progress = StreamProgress(
        ...     samples_processed=500,
        ...     batches_processed=5,
        ...     estimated_total=1000,
        ...     percent_complete=50.0,
        ... )
        >>> progress.percent_complete
        50.0
    """

    samples_processed: int
    batches_processed: int
    estimated_total: int | None
    percent_complete: float | None


def validate_stream_config(config: StreamConfig) -> None:
    """Validate streaming configuration.

    Args:
        config: StreamConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If batch_size is not positive.
        ValueError: If buffer_size is not positive.
        ValueError: If prefetch_batches is negative.
        ValueError: If num_workers is negative.

    Examples:
        >>> config = StreamConfig(batch_size=100, buffer_size=1000)
        >>> validate_stream_config(config)  # No error

        >>> validate_stream_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = StreamConfig(batch_size=0)
        >>> validate_stream_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.buffer_size <= 0:
        msg = f"buffer_size must be positive, got {config.buffer_size}"
        raise ValueError(msg)

    if config.prefetch_batches < 0:
        msg = f"prefetch_batches cannot be negative, got {config.prefetch_batches}"
        raise ValueError(msg)

    if config.num_workers < 0:
        msg = f"num_workers cannot be negative, got {config.num_workers}"
        raise ValueError(msg)


def create_stream_iterator(
    items: Iterator[T],
    *,
    batch_size: int = 1000,
    drop_last: bool = False,
) -> Iterator[list[T]]:
    """Create a batched iterator from a stream of items.

    Args:
        items: Iterator of items to batch.
        batch_size: Number of items per batch. Defaults to 1000.
        drop_last: Whether to drop the last incomplete batch. Defaults to False.

    Yields:
        Lists of items, each with at most batch_size items.

    Raises:
        ValueError: If items is None.
        ValueError: If batch_size is not positive.

    Examples:
        >>> list(create_stream_iterator(iter([1, 2, 3, 4, 5]), batch_size=2))
        [[1, 2], [3, 4], [5]]

        >>> batches = create_stream_iterator(
        ...     iter([1, 2, 3, 4, 5]), batch_size=2, drop_last=True
        ... )
        >>> list(batches)
        [[1, 2], [3, 4]]

        >>> list(create_stream_iterator(iter([]), batch_size=10))
        []

        >>> list(create_stream_iterator(None, batch_size=2))
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

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch and not drop_last:
        yield batch


def stream_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    config: StreamConfig | None = None,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream a dataset from HuggingFace Hub.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub.
        split: Dataset split to stream. Defaults to "train".
        config: Streaming configuration. Defaults to None (uses defaults).
        columns: Columns to include. Defaults to None (all columns).

    Yields:
        Dictionary rows from the dataset.

    Raises:
        ValueError: If dataset_name is empty.
        ValueError: If split is empty.

    Examples:
        >>> next(stream_dataset(""))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_name cannot be empty

        >>> next(stream_dataset("test", split=""))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: split cannot be empty
    """
    if not dataset_name:
        msg = "dataset_name cannot be empty"
        raise ValueError(msg)

    if not split:
        msg = "split cannot be empty"
        raise ValueError(msg)

    if config is not None:
        validate_stream_config(config)

    from datasets import load_dataset

    # Load dataset in streaming mode
    ds = load_dataset(dataset_name, split=split, streaming=True)

    # Select columns if specified
    if columns:
        ds = ds.select_columns(columns)

    # Apply shuffle if configured
    if config and config.shuffle_mode == ShuffleMode.BUFFER:
        ds = ds.shuffle(buffer_size=config.buffer_size)

    yield from ds


def stream_batches(
    dataset_name: str,
    *,
    split: str = "train",
    config: StreamConfig | None = None,
    columns: list[str] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Stream batched data from a HuggingFace dataset.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub.
        split: Dataset split to stream. Defaults to "train".
        config: Streaming configuration. Defaults to None.
        columns: Columns to include. Defaults to None.

    Yields:
        Lists of dictionary rows (batches).

    Raises:
        ValueError: If dataset_name is empty.

    Examples:
        >>> next(stream_batches(""))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_name cannot be empty
    """
    if not dataset_name:
        msg = "dataset_name cannot be empty"
        raise ValueError(msg)

    effective_config = config or StreamConfig()
    rows = stream_dataset(
        dataset_name,
        split=split,
        config=effective_config,
        columns=columns,
    )

    yield from create_stream_iterator(rows, batch_size=effective_config.batch_size)


def map_stream(
    items: Iterator[T],
    fn: Callable[[T], U],
) -> Iterator[U]:
    """Apply a function to each item in a stream.

    Args:
        items: Iterator of items.
        fn: Function to apply to each item.

    Yields:
        Transformed items.

    Raises:
        ValueError: If items is None.
        ValueError: If fn is None.

    Examples:
        >>> list(map_stream(iter([1, 2, 3]), lambda x: x * 2))
        [2, 4, 6]

        >>> list(map_stream(iter(["a", "b"]), str.upper))
        ['A', 'B']

        >>> list(map_stream(None, lambda x: x))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: items cannot be None

        >>> list(map_stream(iter([1]), None))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: fn cannot be None
    """
    if items is None:
        msg = "items cannot be None"
        raise ValueError(msg)

    if fn is None:
        msg = "fn cannot be None"
        raise ValueError(msg)

    for item in items:
        yield fn(item)


def filter_stream(
    items: Iterator[T],
    predicate: Callable[[T], bool],
) -> Iterator[T]:
    """Filter items from a stream based on a predicate.

    Args:
        items: Iterator of items.
        predicate: Function returning True for items to keep.

    Yields:
        Items that satisfy the predicate.

    Raises:
        ValueError: If items is None.
        ValueError: If predicate is None.

    Examples:
        >>> list(filter_stream(iter([1, 2, 3, 4, 5]), lambda x: x > 2))
        [3, 4, 5]

        >>> list(filter_stream(iter(["", "a", "", "b"]), bool))
        ['a', 'b']

        >>> list(filter_stream(None, lambda x: True))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: items cannot be None

        >>> list(filter_stream(iter([1]), None))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predicate cannot be None
    """
    if items is None:
        msg = "items cannot be None"
        raise ValueError(msg)

    if predicate is None:
        msg = "predicate cannot be None"
        raise ValueError(msg)

    for item in items:
        if predicate(item):
            yield item


def take_stream(items: Iterator[T], n: int) -> Iterator[T]:
    """Take the first n items from a stream.

    Args:
        items: Iterator of items.
        n: Maximum number of items to take.

    Yields:
        Up to n items from the stream.

    Raises:
        ValueError: If items is None.
        ValueError: If n is negative.

    Examples:
        >>> list(take_stream(iter([1, 2, 3, 4, 5]), 3))
        [1, 2, 3]

        >>> list(take_stream(iter([1, 2]), 10))
        [1, 2]

        >>> list(take_stream(iter([1, 2, 3]), 0))
        []

        >>> list(take_stream(None, 5))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: items cannot be None

        >>> list(take_stream(iter([1]), -1))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: n cannot be negative
    """
    if items is None:
        msg = "items cannot be None"
        raise ValueError(msg)

    if n < 0:
        msg = f"n cannot be negative, got {n}"
        raise ValueError(msg)

    for count, item in enumerate(items):
        if count >= n:
            break
        yield item


def skip_stream(items: Iterator[T], n: int) -> Iterator[T]:
    """Skip the first n items from a stream.

    Args:
        items: Iterator of items.
        n: Number of items to skip.

    Yields:
        Items after skipping the first n.

    Raises:
        ValueError: If items is None.
        ValueError: If n is negative.

    Examples:
        >>> list(skip_stream(iter([1, 2, 3, 4, 5]), 2))
        [3, 4, 5]

        >>> list(skip_stream(iter([1, 2]), 10))
        []

        >>> list(skip_stream(iter([1, 2, 3]), 0))
        [1, 2, 3]

        >>> list(skip_stream(None, 5))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: items cannot be None

        >>> list(skip_stream(iter([1]), -1))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: n cannot be negative
    """
    if items is None:
        msg = "items cannot be None"
        raise ValueError(msg)

    if n < 0:
        msg = f"n cannot be negative, got {n}"
        raise ValueError(msg)

    count = 0
    for item in items:
        if count < n:
            count += 1
            continue
        yield item


def compute_stream_stats(
    total_samples: int,
    total_batches: int,
    bytes_read: int,
    elapsed_seconds: float,
) -> StreamStats:
    """Compute statistics from a streaming operation.

    Args:
        total_samples: Total samples processed.
        total_batches: Total batches processed.
        bytes_read: Total bytes read.
        elapsed_seconds: Total elapsed time in seconds.

    Returns:
        StreamStats with computed metrics.

    Raises:
        ValueError: If any count is negative.
        ValueError: If elapsed_seconds is negative.

    Examples:
        >>> stats = compute_stream_stats(10000, 10, 1024000, 2.0)
        >>> stats.samples_per_second
        5000.0

        >>> compute_stream_stats(-1, 10, 0, 1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_samples cannot be negative
    """
    if total_samples < 0:
        msg = f"total_samples cannot be negative, got {total_samples}"
        raise ValueError(msg)

    if total_batches < 0:
        msg = f"total_batches cannot be negative, got {total_batches}"
        raise ValueError(msg)

    if bytes_read < 0:
        msg = f"bytes_read cannot be negative, got {bytes_read}"
        raise ValueError(msg)

    if elapsed_seconds < 0:
        msg = f"elapsed_seconds cannot be negative, got {elapsed_seconds}"
        raise ValueError(msg)

    samples_per_sec = total_samples / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return StreamStats(
        total_samples=total_samples,
        total_batches=total_batches,
        bytes_read=bytes_read,
        samples_per_second=samples_per_sec,
    )


def list_shuffle_modes() -> list[str]:
    """List all available shuffle modes.

    Returns:
        Sorted list of shuffle mode names.

    Examples:
        >>> modes = list_shuffle_modes()
        >>> "buffer" in modes
        True
        >>> "disabled" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_SHUFFLE_MODES)


def validate_shuffle_mode(mode: str) -> bool:
    """Validate if a string is a valid shuffle mode.

    Args:
        mode: The mode string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_shuffle_mode("buffer")
        True
        >>> validate_shuffle_mode("disabled")
        True
        >>> validate_shuffle_mode("invalid")
        False
        >>> validate_shuffle_mode("")
        False
    """
    return mode in VALID_SHUFFLE_MODES
