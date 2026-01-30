"""Continuous batching utilities for efficient inference.

This module provides utilities for implementing continuous batching in
transformer model inference, including batching strategies, queue management,
latency SLOs, and throughput estimation.

Examples:
    >>> from hf_gtc.inference.batching import create_batch_config
    >>> config = create_batch_config(max_batch_size=32)
    >>> config.max_batch_size
    32
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class BatchingStrategy(Enum):
    """Strategy for batching inference requests.

    Attributes:
        STATIC: Fixed batch size, wait for batch to fill.
        DYNAMIC: Variable batch size based on queue state.
        CONTINUOUS: Continuous batching with iteration-level scheduling.

    Examples:
        >>> BatchingStrategy.STATIC.value
        'static'
        >>> BatchingStrategy.CONTINUOUS.value
        'continuous'
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTINUOUS = "continuous"


VALID_BATCHING_STRATEGIES = frozenset(s.value for s in BatchingStrategy)


class SchedulingPolicy(Enum):
    """Policy for scheduling requests in the queue.

    Attributes:
        FCFS: First-come-first-served scheduling.
        SJF: Shortest-job-first scheduling.
        PRIORITY: Priority-based scheduling.
        FAIR_SHARE: Fair share scheduling across clients.

    Examples:
        >>> SchedulingPolicy.FCFS.value
        'fcfs'
        >>> SchedulingPolicy.FAIR_SHARE.value
        'fair_share'
    """

    FCFS = "fcfs"
    SJF = "sjf"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"


VALID_SCHEDULING_POLICIES = frozenset(p.value for p in SchedulingPolicy)


class QueueOverflowPolicy(Enum):
    """Policy for handling queue overflow.

    Attributes:
        REJECT: Reject new requests when queue is full.
        WAIT: Block and wait for queue space.
        PREEMPT: Preempt lower priority requests.

    Examples:
        >>> QueueOverflowPolicy.REJECT.value
        'reject'
        >>> QueueOverflowPolicy.PREEMPT.value
        'preempt'
    """

    REJECT = "reject"
    WAIT = "wait"
    PREEMPT = "preempt"


VALID_QUEUE_OVERFLOW_POLICIES = frozenset(p.value for p in QueueOverflowPolicy)


# Type aliases for string literal types
BatchingStrategyStr = Literal["static", "dynamic", "continuous"]
SchedulingPolicyStr = Literal["fcfs", "sjf", "priority", "fair_share"]
QueueOverflowPolicyStr = Literal["reject", "wait", "preempt"]
PaddingStrategyStr = Literal["longest", "max_length", "do_not_pad"]
HardwareTypeStr = Literal["cpu", "gpu_consumer", "gpu_datacenter", "tpu"]


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Configuration for batch inference.

    Attributes:
        strategy: Batching strategy to use.
        max_batch_size: Maximum number of requests per batch.
        max_tokens_per_batch: Maximum total tokens per batch.
        padding_strategy: Strategy for padding sequences.

    Examples:
        >>> config = BatchConfig(
        ...     strategy=BatchingStrategy.CONTINUOUS,
        ...     max_batch_size=32,
        ...     max_tokens_per_batch=4096,
        ...     padding_strategy="longest",
        ... )
        >>> config.max_batch_size
        32
        >>> config.strategy
        <BatchingStrategy.CONTINUOUS: 'continuous'>
    """

    strategy: BatchingStrategy
    max_batch_size: int
    max_tokens_per_batch: int
    padding_strategy: PaddingStrategyStr


@dataclass(frozen=True, slots=True)
class QueueConfig:
    """Configuration for request queue management.

    Attributes:
        max_queue_size: Maximum number of pending requests.
        overflow_policy: Policy for handling overflow.
        timeout_seconds: Request timeout in seconds.

    Examples:
        >>> config = QueueConfig(
        ...     max_queue_size=1000,
        ...     overflow_policy=QueueOverflowPolicy.REJECT,
        ...     timeout_seconds=30.0,
        ... )
        >>> config.max_queue_size
        1000
        >>> config.overflow_policy
        <QueueOverflowPolicy.REJECT: 'reject'>
    """

    max_queue_size: int
    overflow_policy: QueueOverflowPolicy
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class LatencySLO:
    """Latency service level objectives.

    Attributes:
        p50_ms: 50th percentile latency target in milliseconds.
        p90_ms: 90th percentile latency target in milliseconds.
        p99_ms: 99th percentile latency target in milliseconds.
        max_ms: Maximum allowed latency in milliseconds.

    Examples:
        >>> slo = LatencySLO(
        ...     p50_ms=50.0,
        ...     p90_ms=100.0,
        ...     p99_ms=200.0,
        ...     max_ms=500.0,
        ... )
        >>> slo.p50_ms
        50.0
        >>> slo.p99_ms
        200.0
    """

    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class BatchingStats:
    """Statistics from batched inference.

    Attributes:
        requests_processed: Total number of requests processed.
        avg_batch_size: Average batch size.
        avg_wait_time_ms: Average wait time in milliseconds.
        slo_violations: Number of SLO violations.

    Examples:
        >>> stats = BatchingStats(
        ...     requests_processed=10000,
        ...     avg_batch_size=24.5,
        ...     avg_wait_time_ms=15.3,
        ...     slo_violations=42,
        ... )
        >>> stats.requests_processed
        10000
        >>> stats.avg_batch_size
        24.5
    """

    requests_processed: int
    avg_batch_size: float
    avg_wait_time_ms: float
    slo_violations: int


def validate_batch_config(config: BatchConfig) -> None:
    """Validate batch configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = BatchConfig(
        ...     strategy=BatchingStrategy.CONTINUOUS,
        ...     max_batch_size=32,
        ...     max_tokens_per_batch=4096,
        ...     padding_strategy="longest",
        ... )
        >>> validate_batch_config(config)  # No error

        >>> bad_config = BatchConfig(
        ...     strategy=BatchingStrategy.CONTINUOUS,
        ...     max_batch_size=0,
        ...     max_tokens_per_batch=4096,
        ...     padding_strategy="longest",
        ... )
        >>> validate_batch_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_batch_size must be positive
    """
    if config.max_batch_size <= 0:
        msg = f"max_batch_size must be positive, got {config.max_batch_size}"
        raise ValueError(msg)

    if config.max_tokens_per_batch <= 0:
        msg = (
            f"max_tokens_per_batch must be positive, got {config.max_tokens_per_batch}"
        )
        raise ValueError(msg)

    valid_padding = {"longest", "max_length", "do_not_pad"}
    if config.padding_strategy not in valid_padding:
        msg = (
            f"padding_strategy must be one of {valid_padding}, "
            f"got '{config.padding_strategy}'"
        )
        raise ValueError(msg)


def validate_queue_config(config: QueueConfig) -> None:
    """Validate queue configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = QueueConfig(
        ...     max_queue_size=1000,
        ...     overflow_policy=QueueOverflowPolicy.REJECT,
        ...     timeout_seconds=30.0,
        ... )
        >>> validate_queue_config(config)  # No error

        >>> bad_config = QueueConfig(
        ...     max_queue_size=0,
        ...     overflow_policy=QueueOverflowPolicy.REJECT,
        ...     timeout_seconds=30.0,
        ... )
        >>> validate_queue_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_queue_size must be positive
    """
    if config.max_queue_size <= 0:
        msg = f"max_queue_size must be positive, got {config.max_queue_size}"
        raise ValueError(msg)

    if config.timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {config.timeout_seconds}"
        raise ValueError(msg)


def validate_latency_slo(slo: LatencySLO) -> None:
    """Validate latency SLO configuration.

    Args:
        slo: Latency SLO to validate.

    Raises:
        ValueError: If SLO is invalid.

    Examples:
        >>> slo = LatencySLO(p50_ms=50.0, p90_ms=100.0, p99_ms=200.0, max_ms=500.0)
        >>> validate_latency_slo(slo)  # No error

        >>> bad_slo = LatencySLO(p50_ms=-1.0, p90_ms=100.0, p99_ms=200.0, max_ms=500.0)
        >>> validate_latency_slo(bad_slo)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: p50_ms must be positive

        >>> bad_slo2 = LatencySLO(100.0, 50.0, 200.0, 500.0)
        >>> validate_latency_slo(bad_slo2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: p90_ms must be >= p50_ms
    """
    if slo.p50_ms <= 0:
        msg = f"p50_ms must be positive, got {slo.p50_ms}"
        raise ValueError(msg)

    if slo.p90_ms <= 0:
        msg = f"p90_ms must be positive, got {slo.p90_ms}"
        raise ValueError(msg)

    if slo.p99_ms <= 0:
        msg = f"p99_ms must be positive, got {slo.p99_ms}"
        raise ValueError(msg)

    if slo.max_ms <= 0:
        msg = f"max_ms must be positive, got {slo.max_ms}"
        raise ValueError(msg)

    # Check ordering: p50 <= p90 <= p99 <= max
    if slo.p90_ms < slo.p50_ms:
        msg = f"p90_ms must be >= p50_ms, got p90={slo.p90_ms}, p50={slo.p50_ms}"
        raise ValueError(msg)

    if slo.p99_ms < slo.p90_ms:
        msg = f"p99_ms must be >= p90_ms, got p99={slo.p99_ms}, p90={slo.p90_ms}"
        raise ValueError(msg)

    if slo.max_ms < slo.p99_ms:
        msg = f"max_ms must be >= p99_ms, got max={slo.max_ms}, p99={slo.p99_ms}"
        raise ValueError(msg)


def validate_batching_stats(stats: BatchingStats) -> None:
    """Validate batching statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = BatchingStats(
        ...     requests_processed=10000,
        ...     avg_batch_size=24.5,
        ...     avg_wait_time_ms=15.3,
        ...     slo_violations=42,
        ... )
        >>> validate_batching_stats(stats)  # No error

        >>> bad_stats = BatchingStats(
        ...     requests_processed=-1,
        ...     avg_batch_size=24.5,
        ...     avg_wait_time_ms=15.3,
        ...     slo_violations=42,
        ... )
        >>> validate_batching_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: requests_processed cannot be negative
    """
    if stats.requests_processed < 0:
        msg = f"requests_processed cannot be negative, got {stats.requests_processed}"
        raise ValueError(msg)

    if stats.avg_batch_size < 0:
        msg = f"avg_batch_size cannot be negative, got {stats.avg_batch_size}"
        raise ValueError(msg)

    if stats.avg_wait_time_ms < 0:
        msg = f"avg_wait_time_ms cannot be negative, got {stats.avg_wait_time_ms}"
        raise ValueError(msg)

    if stats.slo_violations < 0:
        msg = f"slo_violations cannot be negative, got {stats.slo_violations}"
        raise ValueError(msg)


def create_batch_config(
    strategy: BatchingStrategyStr = "continuous",
    max_batch_size: int = 32,
    max_tokens_per_batch: int = 4096,
    padding_strategy: PaddingStrategyStr = "longest",
) -> BatchConfig:
    """Create a batch configuration.

    Args:
        strategy: Batching strategy. Defaults to "continuous".
        max_batch_size: Maximum batch size. Defaults to 32.
        max_tokens_per_batch: Maximum tokens per batch. Defaults to 4096.
        padding_strategy: Padding strategy. Defaults to "longest".

    Returns:
        BatchConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_batch_config(max_batch_size=64)
        >>> config.max_batch_size
        64

        >>> config = create_batch_config(strategy="dynamic")
        >>> config.strategy
        <BatchingStrategy.DYNAMIC: 'dynamic'>

        >>> create_batch_config(max_batch_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_batch_size must be positive
    """
    if strategy not in VALID_BATCHING_STRATEGIES:
        msg = f"strategy must be one of {VALID_BATCHING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = BatchConfig(
        strategy=BatchingStrategy(strategy),
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        padding_strategy=padding_strategy,
    )
    validate_batch_config(config)
    return config


def create_queue_config(
    max_queue_size: int = 1000,
    overflow_policy: QueueOverflowPolicyStr = "reject",
    timeout_seconds: float = 30.0,
) -> QueueConfig:
    """Create a queue configuration.

    Args:
        max_queue_size: Maximum queue size. Defaults to 1000.
        overflow_policy: Overflow policy. Defaults to "reject".
        timeout_seconds: Request timeout. Defaults to 30.0.

    Returns:
        QueueConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_queue_config(max_queue_size=500)
        >>> config.max_queue_size
        500

        >>> config = create_queue_config(overflow_policy="wait")
        >>> config.overflow_policy
        <QueueOverflowPolicy.WAIT: 'wait'>

        >>> create_queue_config(max_queue_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_queue_size must be positive
    """
    if overflow_policy not in VALID_QUEUE_OVERFLOW_POLICIES:
        msg = (
            f"overflow_policy must be one of {VALID_QUEUE_OVERFLOW_POLICIES}, "
            f"got '{overflow_policy}'"
        )
        raise ValueError(msg)

    config = QueueConfig(
        max_queue_size=max_queue_size,
        overflow_policy=QueueOverflowPolicy(overflow_policy),
        timeout_seconds=timeout_seconds,
    )
    validate_queue_config(config)
    return config


def create_latency_slo(
    p50_ms: float = 50.0,
    p90_ms: float = 100.0,
    p99_ms: float = 200.0,
    max_ms: float = 500.0,
) -> LatencySLO:
    """Create a latency SLO configuration.

    Args:
        p50_ms: 50th percentile target. Defaults to 50.0.
        p90_ms: 90th percentile target. Defaults to 100.0.
        p99_ms: 99th percentile target. Defaults to 200.0.
        max_ms: Maximum latency. Defaults to 500.0.

    Returns:
        LatencySLO with the specified targets.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> slo = create_latency_slo(p50_ms=25.0)
        >>> slo.p50_ms
        25.0

        >>> slo = create_latency_slo(p99_ms=150.0)
        >>> slo.p99_ms
        150.0

        >>> create_latency_slo(p50_ms=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: p50_ms must be positive
    """
    slo = LatencySLO(
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        max_ms=max_ms,
    )
    validate_latency_slo(slo)
    return slo


def create_batching_stats(
    requests_processed: int = 0,
    avg_batch_size: float = 0.0,
    avg_wait_time_ms: float = 0.0,
    slo_violations: int = 0,
) -> BatchingStats:
    """Create batching statistics.

    Args:
        requests_processed: Total requests processed. Defaults to 0.
        avg_batch_size: Average batch size. Defaults to 0.0.
        avg_wait_time_ms: Average wait time. Defaults to 0.0.
        slo_violations: Number of SLO violations. Defaults to 0.

    Returns:
        BatchingStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_batching_stats(requests_processed=1000)
        >>> stats.requests_processed
        1000

        >>> stats = create_batching_stats(avg_batch_size=28.5)
        >>> stats.avg_batch_size
        28.5

        >>> create_batching_stats(requests_processed=-1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: requests_processed cannot be negative
    """
    stats = BatchingStats(
        requests_processed=requests_processed,
        avg_batch_size=avg_batch_size,
        avg_wait_time_ms=avg_wait_time_ms,
        slo_violations=slo_violations,
    )
    validate_batching_stats(stats)
    return stats


def list_batching_strategies() -> list[str]:
    """List available batching strategies.

    Returns:
        Sorted list of batching strategy names.

    Examples:
        >>> strategies = list_batching_strategies()
        >>> "continuous" in strategies
        True
        >>> "static" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_BATCHING_STRATEGIES)


def list_scheduling_policies() -> list[str]:
    """List available scheduling policies.

    Returns:
        Sorted list of scheduling policy names.

    Examples:
        >>> policies = list_scheduling_policies()
        >>> "fcfs" in policies
        True
        >>> "fair_share" in policies
        True
        >>> policies == sorted(policies)
        True
    """
    return sorted(VALID_SCHEDULING_POLICIES)


def list_queue_overflow_policies() -> list[str]:
    """List available queue overflow policies.

    Returns:
        Sorted list of queue overflow policy names.

    Examples:
        >>> policies = list_queue_overflow_policies()
        >>> "reject" in policies
        True
        >>> "preempt" in policies
        True
        >>> policies == sorted(policies)
        True
    """
    return sorted(VALID_QUEUE_OVERFLOW_POLICIES)


def get_batching_strategy(name: str) -> BatchingStrategy:
    """Get a batching strategy by name.

    Args:
        name: Name of the batching strategy.

    Returns:
        The corresponding BatchingStrategy enum value.

    Raises:
        ValueError: If name is not a valid batching strategy.

    Examples:
        >>> get_batching_strategy("continuous")
        <BatchingStrategy.CONTINUOUS: 'continuous'>
        >>> get_batching_strategy("static")
        <BatchingStrategy.STATIC: 'static'>

        >>> get_batching_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown batching strategy
    """
    if name not in VALID_BATCHING_STRATEGIES:
        msg = f"Unknown batching strategy: '{name}'. Valid: {VALID_BATCHING_STRATEGIES}"
        raise ValueError(msg)
    return BatchingStrategy(name)


def get_scheduling_policy(name: str) -> SchedulingPolicy:
    """Get a scheduling policy by name.

    Args:
        name: Name of the scheduling policy.

    Returns:
        The corresponding SchedulingPolicy enum value.

    Raises:
        ValueError: If name is not a valid scheduling policy.

    Examples:
        >>> get_scheduling_policy("fcfs")
        <SchedulingPolicy.FCFS: 'fcfs'>
        >>> get_scheduling_policy("fair_share")
        <SchedulingPolicy.FAIR_SHARE: 'fair_share'>

        >>> get_scheduling_policy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown scheduling policy
    """
    if name not in VALID_SCHEDULING_POLICIES:
        msg = f"Unknown scheduling policy: '{name}'. Valid: {VALID_SCHEDULING_POLICIES}"
        raise ValueError(msg)
    return SchedulingPolicy(name)


def get_queue_overflow_policy(name: str) -> QueueOverflowPolicy:
    """Get a queue overflow policy by name.

    Args:
        name: Name of the queue overflow policy.

    Returns:
        The corresponding QueueOverflowPolicy enum value.

    Raises:
        ValueError: If name is not a valid queue overflow policy.

    Examples:
        >>> get_queue_overflow_policy("reject")
        <QueueOverflowPolicy.REJECT: 'reject'>
        >>> get_queue_overflow_policy("preempt")
        <QueueOverflowPolicy.PREEMPT: 'preempt'>

        >>> get_queue_overflow_policy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown queue overflow policy
    """
    if name not in VALID_QUEUE_OVERFLOW_POLICIES:
        msg = (
            f"Unknown queue overflow policy: '{name}'. "
            f"Valid: {VALID_QUEUE_OVERFLOW_POLICIES}"
        )
        raise ValueError(msg)
    return QueueOverflowPolicy(name)


def calculate_optimal_batch_size(
    available_memory_mb: float,
    avg_sequence_length: int,
    hidden_size: int = 768,
    dtype_bytes: int = 2,
    memory_fraction: float = 0.8,
) -> int:
    """Calculate optimal batch size given available memory.

    Args:
        available_memory_mb: Available GPU memory in megabytes.
        avg_sequence_length: Average sequence length.
        hidden_size: Model hidden dimension. Defaults to 768.
        dtype_bytes: Bytes per element. Defaults to 2 (float16).
        memory_fraction: Fraction of memory to use. Defaults to 0.8.

    Returns:
        Recommended batch size (at least 1).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> batch_size = calculate_optimal_batch_size(
        ...     available_memory_mb=1000.0,
        ...     avg_sequence_length=512,
        ...     hidden_size=768,
        ... )
        >>> batch_size > 0
        True

        >>> calculate_optimal_batch_size(0.0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: available_memory_mb must be positive
    """
    if available_memory_mb <= 0:
        msg = f"available_memory_mb must be positive, got {available_memory_mb}"
        raise ValueError(msg)

    if avg_sequence_length <= 0:
        msg = f"avg_sequence_length must be positive, got {avg_sequence_length}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    if not 0 < memory_fraction <= 1:
        msg = f"memory_fraction must be in (0, 1], got {memory_fraction}"
        raise ValueError(msg)

    # Memory per sample: seq_len * hidden_size * dtype_bytes
    # Plus factor of 3 for activations (forward pass intermediate values)
    memory_per_sample_bytes = avg_sequence_length * hidden_size * dtype_bytes * 3
    memory_per_sample_mb = memory_per_sample_bytes / (1024 * 1024)

    usable_memory = available_memory_mb * memory_fraction
    if memory_per_sample_mb > 0:
        batch_size = int(usable_memory / memory_per_sample_mb)
    else:
        batch_size = 1

    return max(1, batch_size)


def estimate_throughput(
    batch_size: int,
    avg_sequence_length: int,
    inference_time_ms: float,
) -> float:
    """Estimate throughput in tokens per second.

    Args:
        batch_size: Number of sequences per batch.
        avg_sequence_length: Average sequence length.
        inference_time_ms: Time per batch inference in milliseconds.

    Returns:
        Estimated throughput in tokens per second.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> throughput = estimate_throughput(
        ...     batch_size=32,
        ...     avg_sequence_length=512,
        ...     inference_time_ms=100.0,
        ... )
        >>> throughput > 0
        True
        >>> throughput == 163840.0
        True

        >>> estimate_throughput(0, 512, 100.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if avg_sequence_length <= 0:
        msg = f"avg_sequence_length must be positive, got {avg_sequence_length}"
        raise ValueError(msg)

    if inference_time_ms <= 0:
        msg = f"inference_time_ms must be positive, got {inference_time_ms}"
        raise ValueError(msg)

    tokens_per_batch = batch_size * avg_sequence_length
    batches_per_second = 1000.0 / inference_time_ms
    return tokens_per_batch * batches_per_second


def calculate_token_budget(
    max_batch_size: int,
    max_sequence_length: int,
    budget_fraction: float = 0.9,
) -> int:
    """Calculate token budget for a batch.

    Args:
        max_batch_size: Maximum number of sequences.
        max_sequence_length: Maximum sequence length.
        budget_fraction: Fraction of theoretical max to use. Defaults to 0.9.

    Returns:
        Token budget for the batch.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> budget = calculate_token_budget(32, 512)
        >>> budget > 0
        True
        >>> budget == 14745
        True

        >>> budget = calculate_token_budget(32, 512, budget_fraction=1.0)
        >>> budget == 16384
        True

        >>> calculate_token_budget(0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_batch_size must be positive
    """
    if max_batch_size <= 0:
        msg = f"max_batch_size must be positive, got {max_batch_size}"
        raise ValueError(msg)

    if max_sequence_length <= 0:
        msg = f"max_sequence_length must be positive, got {max_sequence_length}"
        raise ValueError(msg)

    if not 0 < budget_fraction <= 1:
        msg = f"budget_fraction must be in (0, 1], got {budget_fraction}"
        raise ValueError(msg)

    theoretical_max = max_batch_size * max_sequence_length
    return int(theoretical_max * budget_fraction)


def check_slo_compliance(
    actual_latency_ms: float,
    slo: LatencySLO,
    percentile: Literal["p50", "p90", "p99", "max"] = "p99",
) -> bool:
    """Check if actual latency meets SLO target.

    Args:
        actual_latency_ms: Actual latency in milliseconds.
        slo: Latency SLO to check against.
        percentile: Which percentile target to check. Defaults to "p99".

    Returns:
        True if latency meets SLO, False otherwise.

    Raises:
        ValueError: If actual_latency_ms is negative.

    Examples:
        >>> slo = LatencySLO(p50_ms=50.0, p90_ms=100.0, p99_ms=200.0, max_ms=500.0)
        >>> check_slo_compliance(150.0, slo, "p99")
        True
        >>> check_slo_compliance(250.0, slo, "p99")
        False
        >>> check_slo_compliance(75.0, slo, "p50")
        False

        >>> check_slo_compliance(-1.0, slo)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: actual_latency_ms cannot be negative
    """
    if actual_latency_ms < 0:
        msg = f"actual_latency_ms cannot be negative, got {actual_latency_ms}"
        raise ValueError(msg)

    target_map = {
        "p50": slo.p50_ms,
        "p90": slo.p90_ms,
        "p99": slo.p99_ms,
        "max": slo.max_ms,
    }

    target = target_map[percentile]
    return actual_latency_ms <= target


def format_batching_stats(stats: BatchingStats) -> str:
    """Format batching statistics as a human-readable string.

    Args:
        stats: Batching statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = BatchingStats(
        ...     requests_processed=10000,
        ...     avg_batch_size=24.5,
        ...     avg_wait_time_ms=15.3,
        ...     slo_violations=42,
        ... )
        >>> formatted = format_batching_stats(stats)
        >>> "Requests Processed: 10000" in formatted
        True
        >>> "Avg Batch Size: 24.50" in formatted
        True

        >>> empty_stats = BatchingStats(0, 0.0, 0.0, 0)
        >>> "Requests Processed: 0" in format_batching_stats(empty_stats)
        True
    """
    lines = [
        f"Requests Processed: {stats.requests_processed}",
        f"Avg Batch Size: {stats.avg_batch_size:.2f}",
        f"Avg Wait Time: {stats.avg_wait_time_ms:.2f} ms",
        f"SLO Violations: {stats.slo_violations}",
    ]
    return "\n".join(lines)


def get_recommended_batching_config(
    hardware: HardwareTypeStr = "gpu_datacenter",
) -> BatchConfig:
    """Get recommended batching configuration for hardware type.

    Args:
        hardware: Target hardware type. Defaults to "gpu_datacenter".

    Returns:
        Recommended BatchConfig for the hardware.

    Raises:
        ValueError: If hardware type is invalid.

    Examples:
        >>> config = get_recommended_batching_config("gpu_datacenter")
        >>> config.strategy
        <BatchingStrategy.CONTINUOUS: 'continuous'>
        >>> config.max_batch_size
        64

        >>> config = get_recommended_batching_config("cpu")
        >>> config.strategy
        <BatchingStrategy.STATIC: 'static'>
        >>> config.max_batch_size
        8

        >>> get_recommended_batching_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown hardware type
    """
    valid_hardware = {"cpu", "gpu_consumer", "gpu_datacenter", "tpu"}
    if hardware not in valid_hardware:
        msg = f"Unknown hardware type: '{hardware}'. Valid: {valid_hardware}"
        raise ValueError(msg)

    # Hardware-specific recommendations
    if hardware == "cpu":
        return BatchConfig(
            strategy=BatchingStrategy.STATIC,
            max_batch_size=8,
            max_tokens_per_batch=1024,
            padding_strategy="longest",
        )
    elif hardware == "gpu_consumer":
        return BatchConfig(
            strategy=BatchingStrategy.DYNAMIC,
            max_batch_size=16,
            max_tokens_per_batch=2048,
            padding_strategy="longest",
        )
    elif hardware == "gpu_datacenter":
        return BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=64,
            max_tokens_per_batch=8192,
            padding_strategy="longest",
        )
    else:  # tpu
        return BatchConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            max_batch_size=128,
            max_tokens_per_batch=16384,
            padding_strategy="max_length",
        )
