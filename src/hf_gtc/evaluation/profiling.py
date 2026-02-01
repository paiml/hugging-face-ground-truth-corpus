"""Model profiling and benchmarking utilities.

This module provides utilities for profiling ML model performance,
including latency, throughput, memory usage, and FLOPS estimation.

Examples:
    >>> from hf_gtc.evaluation.profiling import ProfileMetric, ProfilingLevel
    >>> ProfileMetric.LATENCY.value
    'latency'
    >>> ProfilingLevel.BASIC.value
    'basic'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from hf_gtc._validation import validate_not_none


class ProfileMetric(Enum):
    """Types of profiling metrics to collect.

    Attributes:
        LATENCY: Execution time measurement.
        THROUGHPUT: Samples processed per unit time.
        MEMORY: Memory consumption metrics.
        FLOPS: Floating point operations count.
        PARAMETERS: Model parameter count.

    Examples:
        >>> ProfileMetric.LATENCY.value
        'latency'
        >>> ProfileMetric.THROUGHPUT.value
        'throughput'
        >>> ProfileMetric.MEMORY.value
        'memory'
    """

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    FLOPS = "flops"
    PARAMETERS = "parameters"


VALID_PROFILE_METRICS = frozenset(m.value for m in ProfileMetric)


class BottleneckType(Enum):
    """Types of performance bottlenecks.

    Attributes:
        COMPUTE: Compute-bound bottleneck (limited by computation).
        MEMORY: Memory-bound bottleneck (limited by memory bandwidth).
        IO: I/O-bound bottleneck (limited by data loading).
        COMMUNICATION: Communication-bound (limited by data transfer).

    Examples:
        >>> BottleneckType.COMPUTE.value
        'compute'
        >>> BottleneckType.MEMORY.value
        'memory'
        >>> BottleneckType.IO.value
        'io'
    """

    COMPUTE = "compute"
    MEMORY = "memory"
    IO = "io"
    COMMUNICATION = "communication"


VALID_BOTTLENECK_TYPES = frozenset(b.value for b in BottleneckType)


class ProfilingLevel(Enum):
    """Profiling detail level.

    Attributes:
        BASIC: Basic profiling (timing only).
        DETAILED: Detailed profiling (timing + memory).
        TRACE: Full trace profiling (all metrics + call stack).

    Examples:
        >>> ProfilingLevel.BASIC.value
        'basic'
        >>> ProfilingLevel.DETAILED.value
        'detailed'
        >>> ProfilingLevel.TRACE.value
        'trace'
    """

    BASIC = "basic"
    DETAILED = "detailed"
    TRACE = "trace"


VALID_PROFILING_LEVELS = frozenset(lvl.value for lvl in ProfilingLevel)


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    """Configuration for model profiling.

    Attributes:
        metrics: List of metrics to collect.
        level: Profiling detail level. Defaults to BASIC.
        warmup_runs: Number of warmup iterations. Defaults to 3.
        num_runs: Number of measurement runs. Defaults to 10.
        include_backward: Whether to include backward pass. Defaults to False.

    Examples:
        >>> config = ProfilingConfig(
        ...     metrics=[ProfileMetric.LATENCY, ProfileMetric.MEMORY],
        ... )
        >>> config.level
        <ProfilingLevel.BASIC: 'basic'>
        >>> config.warmup_runs
        3
    """

    metrics: tuple[ProfileMetric, ...]
    level: ProfilingLevel = ProfilingLevel.BASIC
    warmup_runs: int = 3
    num_runs: int = 10
    include_backward: bool = False


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """Breakdown of latency components.

    Attributes:
        forward_ms: Forward pass latency in milliseconds.
        backward_ms: Backward pass latency in milliseconds.
        data_loading_ms: Data loading latency in milliseconds.
        optimizer_ms: Optimizer step latency in milliseconds.

    Examples:
        >>> breakdown = LatencyBreakdown(
        ...     forward_ms=10.5,
        ...     backward_ms=15.2,
        ...     data_loading_ms=2.1,
        ...     optimizer_ms=1.3,
        ... )
        >>> breakdown.forward_ms
        10.5
        >>> breakdown.total_ms
        29.1
    """

    forward_ms: float
    backward_ms: float
    data_loading_ms: float
    optimizer_ms: float

    @property
    def total_ms(self) -> float:
        """Calculate total latency in milliseconds.

        Returns:
            Sum of all latency components.

        Examples:
            >>> breakdown = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
            >>> breakdown.total_ms
            28.0
        """
        return (
            self.forward_ms
            + self.backward_ms
            + self.data_loading_ms
            + self.optimizer_ms
        )


@dataclass(frozen=True, slots=True)
class MemoryBreakdown:
    """Breakdown of memory usage components.

    Attributes:
        parameters_mb: Memory for model parameters in MB.
        gradients_mb: Memory for gradients in MB.
        activations_mb: Memory for activations in MB.
        optimizer_state_mb: Memory for optimizer state in MB.

    Examples:
        >>> breakdown = MemoryBreakdown(
        ...     parameters_mb=500.0,
        ...     gradients_mb=500.0,
        ...     activations_mb=1000.0,
        ...     optimizer_state_mb=1000.0,
        ... )
        >>> breakdown.parameters_mb
        500.0
        >>> breakdown.total_mb
        3000.0
    """

    parameters_mb: float
    gradients_mb: float
    activations_mb: float
    optimizer_state_mb: float

    @property
    def total_mb(self) -> float:
        """Calculate total memory usage in MB.

        Returns:
            Sum of all memory components.

        Examples:
            >>> breakdown = MemoryBreakdown(100.0, 100.0, 200.0, 200.0)
            >>> breakdown.total_mb
            600.0
        """
        return (
            self.parameters_mb
            + self.gradients_mb
            + self.activations_mb
            + self.optimizer_state_mb
        )


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    """Results from model profiling.

    Attributes:
        latency: Latency breakdown or None.
        memory: Memory breakdown or None.
        throughput_tokens_per_sec: Throughput in tokens per second.
        bottlenecks: Identified performance bottlenecks.

    Examples:
        >>> latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        >>> memory = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        >>> result = ProfilingResult(
        ...     latency=latency,
        ...     memory=memory,
        ...     throughput_tokens_per_sec=1000.0,
        ...     bottlenecks=[BottleneckType.MEMORY],
        ... )
        >>> result.throughput_tokens_per_sec
        1000.0
    """

    latency: LatencyBreakdown | None
    memory: MemoryBreakdown | None
    throughput_tokens_per_sec: float
    bottlenecks: tuple[BottleneckType, ...]


def validate_profiling_config(config: ProfilingConfig) -> None:
    """Validate profiling configuration.

    Args:
        config: ProfilingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If metrics is empty.
        ValueError: If warmup_runs is negative.
        ValueError: If num_runs is not positive.

    Examples:
        >>> config = ProfilingConfig(metrics=[ProfileMetric.LATENCY])
        >>> validate_profiling_config(config)  # No error

        >>> validate_profiling_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ProfilingConfig(metrics=[], num_runs=1)
        >>> validate_profiling_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metrics cannot be empty
    """
    validate_not_none(config, "config")

    if not config.metrics:
        msg = "metrics cannot be empty"
        raise ValueError(msg)

    if config.warmup_runs < 0:
        msg = f"warmup_runs cannot be negative, got {config.warmup_runs}"
        raise ValueError(msg)

    if config.num_runs <= 0:
        msg = f"num_runs must be positive, got {config.num_runs}"
        raise ValueError(msg)


def validate_latency_breakdown(breakdown: LatencyBreakdown) -> None:
    """Validate latency breakdown values.

    Args:
        breakdown: LatencyBreakdown to validate.

    Raises:
        ValueError: If breakdown is None.
        ValueError: If any latency value is negative.

    Examples:
        >>> breakdown = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        >>> validate_latency_breakdown(breakdown)  # No error

        >>> validate_latency_breakdown(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: breakdown cannot be None

        >>> bad = LatencyBreakdown(-1.0, 15.0, 2.0, 1.0)
        >>> validate_latency_breakdown(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: forward_ms cannot be negative
    """
    if breakdown is None:
        msg = "breakdown cannot be None"
        raise ValueError(msg)

    if breakdown.forward_ms < 0:
        msg = f"forward_ms cannot be negative, got {breakdown.forward_ms}"
        raise ValueError(msg)

    if breakdown.backward_ms < 0:
        msg = f"backward_ms cannot be negative, got {breakdown.backward_ms}"
        raise ValueError(msg)

    if breakdown.data_loading_ms < 0:
        msg = f"data_loading_ms cannot be negative, got {breakdown.data_loading_ms}"
        raise ValueError(msg)

    if breakdown.optimizer_ms < 0:
        msg = f"optimizer_ms cannot be negative, got {breakdown.optimizer_ms}"
        raise ValueError(msg)


def validate_memory_breakdown(breakdown: MemoryBreakdown) -> None:
    """Validate memory breakdown values.

    Args:
        breakdown: MemoryBreakdown to validate.

    Raises:
        ValueError: If breakdown is None.
        ValueError: If any memory value is negative.

    Examples:
        >>> breakdown = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        >>> validate_memory_breakdown(breakdown)  # No error

        >>> validate_memory_breakdown(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: breakdown cannot be None

        >>> bad = MemoryBreakdown(-100.0, 500.0, 1000.0, 1000.0)
        >>> validate_memory_breakdown(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: parameters_mb cannot be negative
    """
    if breakdown is None:
        msg = "breakdown cannot be None"
        raise ValueError(msg)

    if breakdown.parameters_mb < 0:
        msg = f"parameters_mb cannot be negative, got {breakdown.parameters_mb}"
        raise ValueError(msg)

    if breakdown.gradients_mb < 0:
        msg = f"gradients_mb cannot be negative, got {breakdown.gradients_mb}"
        raise ValueError(msg)

    if breakdown.activations_mb < 0:
        msg = f"activations_mb cannot be negative, got {breakdown.activations_mb}"
        raise ValueError(msg)

    if breakdown.optimizer_state_mb < 0:
        msg = (
            f"optimizer_state_mb cannot be negative, got {breakdown.optimizer_state_mb}"
        )
        raise ValueError(msg)


def validate_profiling_result(result: ProfilingResult) -> None:
    """Validate profiling result values.

    Args:
        result: ProfilingResult to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If throughput is negative.

    Examples:
        >>> latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        >>> result = ProfilingResult(
        ...     latency=latency,
        ...     memory=None,
        ...     throughput_tokens_per_sec=1000.0,
        ...     bottlenecks=[],
        ... )
        >>> validate_profiling_result(result)  # No error

        >>> validate_profiling_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    validate_not_none(result, "result")

    if result.throughput_tokens_per_sec < 0:
        msg = (
            f"throughput_tokens_per_sec cannot be negative, "
            f"got {result.throughput_tokens_per_sec}"
        )
        raise ValueError(msg)

    if result.latency is not None:
        validate_latency_breakdown(result.latency)

    if result.memory is not None:
        validate_memory_breakdown(result.memory)


def create_profiling_config(
    metrics: Sequence[ProfileMetric],
    *,
    level: ProfilingLevel = ProfilingLevel.BASIC,
    warmup_runs: int = 3,
    num_runs: int = 10,
    include_backward: bool = False,
) -> ProfilingConfig:
    """Create and validate a profiling configuration.

    Args:
        metrics: List of metrics to collect.
        level: Profiling detail level. Defaults to BASIC.
        warmup_runs: Number of warmup runs. Defaults to 3.
        num_runs: Number of measurement runs. Defaults to 10.
        include_backward: Whether to include backward pass. Defaults to False.

    Returns:
        Validated ProfilingConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_profiling_config([ProfileMetric.LATENCY])
        >>> config.metrics
        (<ProfileMetric.LATENCY: 'latency'>,)

        >>> config = create_profiling_config(
        ...     [ProfileMetric.LATENCY, ProfileMetric.MEMORY],
        ...     level=ProfilingLevel.DETAILED,
        ...     num_runs=20,
        ... )
        >>> config.level
        <ProfilingLevel.DETAILED: 'detailed'>
        >>> config.num_runs
        20
    """
    config = ProfilingConfig(
        metrics=tuple(metrics),
        level=level,
        warmup_runs=warmup_runs,
        num_runs=num_runs,
        include_backward=include_backward,
    )
    validate_profiling_config(config)
    return config


def create_latency_breakdown(
    forward_ms: float,
    backward_ms: float = 0.0,
    data_loading_ms: float = 0.0,
    optimizer_ms: float = 0.0,
) -> LatencyBreakdown:
    """Create and validate a latency breakdown.

    Args:
        forward_ms: Forward pass latency in milliseconds.
        backward_ms: Backward pass latency in milliseconds. Defaults to 0.0.
        data_loading_ms: Data loading latency in milliseconds. Defaults to 0.0.
        optimizer_ms: Optimizer step latency in milliseconds. Defaults to 0.0.

    Returns:
        Validated LatencyBreakdown instance.

    Raises:
        ValueError: If any latency value is negative.

    Examples:
        >>> breakdown = create_latency_breakdown(10.5)
        >>> breakdown.forward_ms
        10.5
        >>> breakdown.backward_ms
        0.0

        >>> breakdown = create_latency_breakdown(10.0, 15.0, 2.0, 1.0)
        >>> breakdown.total_ms
        28.0
    """
    breakdown = LatencyBreakdown(
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        data_loading_ms=data_loading_ms,
        optimizer_ms=optimizer_ms,
    )
    validate_latency_breakdown(breakdown)
    return breakdown


def create_memory_breakdown(
    parameters_mb: float,
    gradients_mb: float = 0.0,
    activations_mb: float = 0.0,
    optimizer_state_mb: float = 0.0,
) -> MemoryBreakdown:
    """Create and validate a memory breakdown.

    Args:
        parameters_mb: Memory for parameters in MB.
        gradients_mb: Memory for gradients in MB. Defaults to 0.0.
        activations_mb: Memory for activations in MB. Defaults to 0.0.
        optimizer_state_mb: Memory for optimizer state in MB. Defaults to 0.0.

    Returns:
        Validated MemoryBreakdown instance.

    Raises:
        ValueError: If any memory value is negative.

    Examples:
        >>> breakdown = create_memory_breakdown(500.0)
        >>> breakdown.parameters_mb
        500.0
        >>> breakdown.gradients_mb
        0.0

        >>> breakdown = create_memory_breakdown(500.0, 500.0, 1000.0, 1000.0)
        >>> breakdown.total_mb
        3000.0
    """
    breakdown = MemoryBreakdown(
        parameters_mb=parameters_mb,
        gradients_mb=gradients_mb,
        activations_mb=activations_mb,
        optimizer_state_mb=optimizer_state_mb,
    )
    validate_memory_breakdown(breakdown)
    return breakdown


def create_profiling_result(
    throughput_tokens_per_sec: float,
    *,
    latency: LatencyBreakdown | None = None,
    memory: MemoryBreakdown | None = None,
    bottlenecks: Sequence[BottleneckType] | None = None,
) -> ProfilingResult:
    """Create and validate a profiling result.

    Args:
        throughput_tokens_per_sec: Throughput in tokens per second.
        latency: Latency breakdown or None.
        memory: Memory breakdown or None.
        bottlenecks: List of identified bottlenecks or None.

    Returns:
        Validated ProfilingResult instance.

    Raises:
        ValueError: If throughput is negative.

    Examples:
        >>> result = create_profiling_result(1000.0)
        >>> result.throughput_tokens_per_sec
        1000.0
        >>> result.latency is None
        True

        >>> latency = create_latency_breakdown(10.0, 15.0)
        >>> result = create_profiling_result(
        ...     1500.0,
        ...     latency=latency,
        ...     bottlenecks=[BottleneckType.COMPUTE],
        ... )
        >>> result.bottlenecks
        (<BottleneckType.COMPUTE: 'compute'>,)
    """
    result = ProfilingResult(
        latency=latency,
        memory=memory,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        bottlenecks=tuple(bottlenecks) if bottlenecks else (),
    )
    validate_profiling_result(result)
    return result


def list_profile_metrics() -> list[str]:
    """List all available profile metrics.

    Returns:
        Sorted list of profile metric names.

    Examples:
        >>> metrics = list_profile_metrics()
        >>> "latency" in metrics
        True
        >>> "throughput" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_PROFILE_METRICS)


def get_profile_metric(name: str) -> ProfileMetric:
    """Get ProfileMetric enum from string name.

    Args:
        name: Name of the profile metric.

    Returns:
        Corresponding ProfileMetric enum value.

    Raises:
        ValueError: If name is not a valid profile metric.

    Examples:
        >>> get_profile_metric("latency")
        <ProfileMetric.LATENCY: 'latency'>

        >>> get_profile_metric("memory")
        <ProfileMetric.MEMORY: 'memory'>

        >>> get_profile_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid profile metric: invalid
    """
    if name not in VALID_PROFILE_METRICS:
        msg = f"invalid profile metric: {name}"
        raise ValueError(msg)

    return ProfileMetric(name)


def validate_profile_metric(metric: str) -> bool:
    """Validate if a string is a valid profile metric.

    Args:
        metric: The metric string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_profile_metric("latency")
        True
        >>> validate_profile_metric("throughput")
        True
        >>> validate_profile_metric("invalid")
        False
        >>> validate_profile_metric("")
        False
    """
    return metric in VALID_PROFILE_METRICS


def list_bottleneck_types() -> list[str]:
    """List all available bottleneck types.

    Returns:
        Sorted list of bottleneck type names.

    Examples:
        >>> types = list_bottleneck_types()
        >>> "compute" in types
        True
        >>> "memory" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_BOTTLENECK_TYPES)


def get_bottleneck_type(name: str) -> BottleneckType:
    """Get BottleneckType enum from string name.

    Args:
        name: Name of the bottleneck type.

    Returns:
        Corresponding BottleneckType enum value.

    Raises:
        ValueError: If name is not a valid bottleneck type.

    Examples:
        >>> get_bottleneck_type("compute")
        <BottleneckType.COMPUTE: 'compute'>

        >>> get_bottleneck_type("memory")
        <BottleneckType.MEMORY: 'memory'>

        >>> get_bottleneck_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid bottleneck type: invalid
    """
    if name not in VALID_BOTTLENECK_TYPES:
        msg = f"invalid bottleneck type: {name}"
        raise ValueError(msg)

    return BottleneckType(name)


def validate_bottleneck_type(bottleneck: str) -> bool:
    """Validate if a string is a valid bottleneck type.

    Args:
        bottleneck: The bottleneck string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_bottleneck_type("compute")
        True
        >>> validate_bottleneck_type("memory")
        True
        >>> validate_bottleneck_type("invalid")
        False
        >>> validate_bottleneck_type("")
        False
    """
    return bottleneck in VALID_BOTTLENECK_TYPES


def list_profiling_levels() -> list[str]:
    """List all available profiling levels.

    Returns:
        Sorted list of profiling level names.

    Examples:
        >>> levels = list_profiling_levels()
        >>> "basic" in levels
        True
        >>> "detailed" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_PROFILING_LEVELS)


def get_profiling_level(name: str) -> ProfilingLevel:
    """Get ProfilingLevel enum from string name.

    Args:
        name: Name of the profiling level.

    Returns:
        Corresponding ProfilingLevel enum value.

    Raises:
        ValueError: If name is not a valid profiling level.

    Examples:
        >>> get_profiling_level("basic")
        <ProfilingLevel.BASIC: 'basic'>

        >>> get_profiling_level("detailed")
        <ProfilingLevel.DETAILED: 'detailed'>

        >>> get_profiling_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid profiling level: invalid
    """
    if name not in VALID_PROFILING_LEVELS:
        msg = f"invalid profiling level: {name}"
        raise ValueError(msg)

    return ProfilingLevel(name)


def validate_profiling_level(level: str) -> bool:
    """Validate if a string is a valid profiling level.

    Args:
        level: The level string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_profiling_level("basic")
        True
        >>> validate_profiling_level("trace")
        True
        >>> validate_profiling_level("invalid")
        False
        >>> validate_profiling_level("")
        False
    """
    return level in VALID_PROFILING_LEVELS


def calculate_flops(
    num_parameters: int,
    sequence_length: int,
    batch_size: int = 1,
    *,
    is_training: bool = False,
) -> int:
    """Estimate FLOPs for a transformer model forward pass.

    Uses the approximation: FLOPs = 2 * num_parameters * sequence_length * batch_size
    For training, multiplies by 3 (forward + backward + optimizer).

    Args:
        num_parameters: Number of model parameters.
        sequence_length: Input sequence length.
        batch_size: Batch size. Defaults to 1.
        is_training: Whether this is a training step. Defaults to False.

    Returns:
        Estimated number of FLOPs.

    Raises:
        ValueError: If any input is not positive.

    Examples:
        >>> calculate_flops(1_000_000, 512)
        1024000000

        >>> calculate_flops(1_000_000, 512, batch_size=8)
        8192000000

        >>> calculate_flops(1_000_000, 512, is_training=True)
        3072000000

        >>> calculate_flops(0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive
    """
    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    # Base FLOPs estimation for forward pass
    flops = 2 * num_parameters * sequence_length * batch_size

    # Training includes backward pass and optimizer step
    if is_training:
        flops *= 3

    return flops


def estimate_memory_footprint(
    num_parameters: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    *,
    bytes_per_param: int = 4,
    is_training: bool = False,
    optimizer_state_factor: float = 2.0,
) -> MemoryBreakdown:
    """Estimate memory footprint for a transformer model.

    Args:
        num_parameters: Number of model parameters.
        batch_size: Batch size.
        sequence_length: Input sequence length.
        hidden_size: Hidden dimension size.
        num_layers: Number of transformer layers.
        bytes_per_param: Bytes per parameter (4=fp32, 2=fp16). Defaults to 4.
        is_training: Whether this is training mode. Defaults to False.
        optimizer_state_factor: Factor for optimizer state (2=Adam). Defaults to 2.0.

    Returns:
        MemoryBreakdown with estimated memory usage.

    Raises:
        ValueError: If any input is not positive.

    Examples:
        >>> breakdown = estimate_memory_footprint(
        ...     num_parameters=1_000_000,
        ...     batch_size=1,
        ...     sequence_length=512,
        ...     hidden_size=768,
        ...     num_layers=12,
        ... )
        >>> breakdown.parameters_mb > 0
        True
        >>> breakdown.gradients_mb
        0.0

        >>> breakdown = estimate_memory_footprint(
        ...     num_parameters=1_000_000,
        ...     batch_size=8,
        ...     sequence_length=512,
        ...     hidden_size=768,
        ...     num_layers=12,
        ...     is_training=True,
        ... )
        >>> breakdown.gradients_mb > 0
        True
    """
    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if bytes_per_param <= 0:
        msg = f"bytes_per_param must be positive, got {bytes_per_param}"
        raise ValueError(msg)

    # Convert to MB
    bytes_to_mb = 1 / (1024 * 1024)

    # Parameters memory
    parameters_mb = num_parameters * bytes_per_param * bytes_to_mb

    # Gradients memory (only during training)
    gradients_mb = 0.0
    if is_training:
        gradients_mb = num_parameters * bytes_per_param * bytes_to_mb

    # Activations memory (rough estimate for transformer)
    # Each layer stores activations of shape (batch, seq, hidden)
    activation_size = batch_size * sequence_length * hidden_size * bytes_per_param
    activations_mb = num_layers * activation_size * bytes_to_mb

    # Optimizer state memory (only during training)
    optimizer_state_mb = 0.0
    if is_training:
        optimizer_state_mb = (
            num_parameters * bytes_per_param * optimizer_state_factor * bytes_to_mb
        )

    return MemoryBreakdown(
        parameters_mb=parameters_mb,
        gradients_mb=gradients_mb,
        activations_mb=activations_mb,
        optimizer_state_mb=optimizer_state_mb,
    )


def identify_bottlenecks(
    latency: LatencyBreakdown,
    memory: MemoryBreakdown,
    *,
    io_threshold: float = 0.3,
    memory_threshold_mb: float = 8000.0,
) -> list[BottleneckType]:
    """Identify performance bottlenecks from profiling data.

    Args:
        latency: Latency breakdown from profiling.
        memory: Memory breakdown from profiling.
        io_threshold: Threshold for I/O bottleneck (loading/total). Defaults to 0.3.
        memory_threshold_mb: Threshold for memory bottleneck in MB. Defaults to 8000.

    Returns:
        List of identified bottleneck types.

    Raises:
        ValueError: If latency is None.
        ValueError: If memory is None.

    Examples:
        >>> latency = LatencyBreakdown(10.0, 15.0, 20.0, 1.0)  # High data loading
        >>> memory = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        >>> bottlenecks = identify_bottlenecks(latency, memory)
        >>> BottleneckType.IO in bottlenecks
        True

        >>> latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)  # Normal
        >>> memory = MemoryBreakdown(5000.0, 5000.0, 5000.0, 5000.0)  # High memory
        >>> bottlenecks = identify_bottlenecks(latency, memory)
        >>> BottleneckType.MEMORY in bottlenecks
        True
    """
    if latency is None:
        msg = "latency cannot be None"
        raise ValueError(msg)

    if memory is None:
        msg = "memory cannot be None"
        raise ValueError(msg)

    validate_latency_breakdown(latency)
    validate_memory_breakdown(memory)

    bottlenecks = []
    total_latency = latency.total_ms

    # Check for I/O bottleneck
    if total_latency > 0 and latency.data_loading_ms / total_latency > io_threshold:
        bottlenecks.append(BottleneckType.IO)

    # Check for memory bottleneck
    if memory.total_mb > memory_threshold_mb:
        bottlenecks.append(BottleneckType.MEMORY)

    # Check for compute bottleneck (forward + backward dominates)
    if total_latency > 0:
        compute_ratio = (latency.forward_ms + latency.backward_ms) / total_latency
        if compute_ratio > 0.7 and BottleneckType.IO not in bottlenecks:
            bottlenecks.append(BottleneckType.COMPUTE)

    # Default to compute if no other bottleneck identified
    if not bottlenecks:
        bottlenecks.append(BottleneckType.COMPUTE)

    return bottlenecks


def compare_profiles(
    profiles: Sequence[ProfilingResult],
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare multiple profiling results.

    Args:
        profiles: Sequence of profiling results to compare.
        names: Optional names for each profile. Defaults to profile_0, profile_1, etc.

    Returns:
        Dictionary with comparison statistics.

    Raises:
        ValueError: If profiles is None or empty.
        ValueError: If names length doesn't match profiles length.

    Examples:
        >>> result1 = create_profiling_result(1000.0)
        >>> result2 = create_profiling_result(1500.0)
        >>> comparison = compare_profiles([result1, result2], ["model_a", "model_b"])
        >>> comparison["fastest"]
        'model_b'
        >>> comparison["throughput_range"]["max"]
        1500.0

        >>> compare_profiles([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: profiles cannot be empty

        >>> compare_profiles(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: profiles cannot be None
    """
    if profiles is None:
        msg = "profiles cannot be None"
        raise ValueError(msg)

    if len(profiles) == 0:
        msg = "profiles cannot be empty"
        raise ValueError(msg)

    if names is None:
        names = [f"profile_{i}" for i in range(len(profiles))]

    if len(names) != len(profiles):
        msg = (
            f"names length ({len(names)}) must match profiles length ({len(profiles)})"
        )
        raise ValueError(msg)

    # Find fastest (highest throughput)
    throughputs = [p.throughput_tokens_per_sec for p in profiles]
    max_throughput_idx = throughputs.index(max(throughputs))
    min_throughput_idx = throughputs.index(min(throughputs))

    comparison: dict[str, Any] = {
        "num_profiles": len(profiles),
        "fastest": names[max_throughput_idx],
        "slowest": names[min_throughput_idx],
        "throughput_range": {
            "min": min(throughputs),
            "max": max(throughputs),
            "mean": sum(throughputs) / len(throughputs),
        },
    }

    # Compare latencies if available
    latencies = [
        (n, p.latency)
        for n, p in zip(names, profiles, strict=True)
        if p.latency is not None
    ]
    if latencies:
        total_latencies = [(n, lat.total_ms) for n, lat in latencies]
        min_lat = min(total_latencies, key=lambda x: x[1])
        max_lat = max(total_latencies, key=lambda x: x[1])
        comparison["latency_comparison"] = {
            "lowest_latency": min_lat[0],
            "lowest_latency_ms": min_lat[1],
            "highest_latency": max_lat[0],
            "highest_latency_ms": max_lat[1],
        }

    # Compare memory if available
    memories = [
        (n, p.memory)
        for n, p in zip(names, profiles, strict=True)
        if p.memory is not None
    ]
    if memories:
        total_memories = [(n, mem.total_mb) for n, mem in memories]
        min_mem = min(total_memories, key=lambda x: x[1])
        max_mem = max(total_memories, key=lambda x: x[1])
        comparison["memory_comparison"] = {
            "lowest_memory": min_mem[0],
            "lowest_memory_mb": min_mem[1],
            "highest_memory": max_mem[0],
            "highest_memory_mb": max_mem[1],
        }

    # Aggregate bottlenecks
    all_bottlenecks: dict[str, int] = {}
    for profile in profiles:
        for bottleneck in profile.bottlenecks:
            name = bottleneck.value
            all_bottlenecks[name] = all_bottlenecks.get(name, 0) + 1

    comparison["bottleneck_counts"] = all_bottlenecks

    return comparison


def format_profiling_result(result: ProfilingResult) -> str:
    """Format a profiling result as a human-readable string.

    Args:
        result: Profiling result to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> latency = create_latency_breakdown(10.0, 15.0, 2.0, 1.0)
        >>> memory = create_memory_breakdown(500.0, 500.0, 1000.0, 1000.0)
        >>> result = create_profiling_result(
        ...     1000.0,
        ...     latency=latency,
        ...     memory=memory,
        ...     bottlenecks=[BottleneckType.COMPUTE],
        ... )
        >>> formatted = format_profiling_result(result)
        >>> "Throughput" in formatted
        True
        >>> "1000.0" in formatted
        True

        >>> format_profiling_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    validate_not_none(result, "result")

    lines = [
        "Model Profiling Results",
        "=" * 40,
        f"Throughput: {result.throughput_tokens_per_sec:.1f} tokens/sec",
    ]

    if result.latency is not None:
        lines.extend(
            [
                "",
                "Latency Breakdown:",
                f"  Forward:      {result.latency.forward_ms:.2f} ms",
                f"  Backward:     {result.latency.backward_ms:.2f} ms",
                f"  Data Loading: {result.latency.data_loading_ms:.2f} ms",
                f"  Optimizer:    {result.latency.optimizer_ms:.2f} ms",
                f"  Total:        {result.latency.total_ms:.2f} ms",
            ]
        )

    if result.memory is not None:
        lines.extend(
            [
                "",
                "Memory Breakdown:",
                f"  Parameters:      {result.memory.parameters_mb:.1f} MB",
                f"  Gradients:       {result.memory.gradients_mb:.1f} MB",
                f"  Activations:     {result.memory.activations_mb:.1f} MB",
                f"  Optimizer State: {result.memory.optimizer_state_mb:.1f} MB",
                f"  Total:           {result.memory.total_mb:.1f} MB",
            ]
        )

    if result.bottlenecks:
        lines.extend(
            [
                "",
                "Identified Bottlenecks:",
            ]
        )
        for bottleneck in result.bottlenecks:
            lines.append(f"  - {bottleneck.value}")

    return "\n".join(lines)


def get_recommended_profiling_config(
    model_type: str,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Get recommended profiling configuration for a model type.

    Args:
        model_type: Type of model (e.g., "llm", "vision", "multimodal").
        quick: Whether to use quick profiling settings. Defaults to False.

    Returns:
        Dictionary with recommended configuration.

    Raises:
        ValueError: If model_type is None or empty.

    Examples:
        >>> config = get_recommended_profiling_config("llm")
        >>> "metrics" in config
        True
        >>> "level" in config
        True

        >>> config = get_recommended_profiling_config("llm", quick=True)
        >>> config["num_runs"]
        5

        >>> get_recommended_profiling_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty

        >>> get_recommended_profiling_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be None
    """
    if model_type is None:
        msg = "model_type cannot be None"
        raise ValueError(msg)

    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    # Base configuration
    base_config: dict[str, Any] = {
        "metrics": [
            ProfileMetric.LATENCY,
            ProfileMetric.THROUGHPUT,
            ProfileMetric.MEMORY,
        ],
        "level": ProfilingLevel.DETAILED,
        "warmup_runs": 5 if not quick else 2,
        "num_runs": 20 if not quick else 5,
        "include_backward": False,
    }

    # Model-specific configurations
    model_configs: dict[str, dict[str, Any]] = {
        "llm": {
            "metrics": [
                ProfileMetric.LATENCY,
                ProfileMetric.THROUGHPUT,
                ProfileMetric.MEMORY,
                ProfileMetric.PARAMETERS,
            ],
            "level": ProfilingLevel.DETAILED,
            "warmup_runs": 5 if not quick else 2,
            "num_runs": 20 if not quick else 5,
            "include_backward": False,
            "recommended_batch_sizes": [1, 4, 8, 16],
            "recommended_sequence_lengths": [128, 512, 1024, 2048],
        },
        "vision": {
            "metrics": [
                ProfileMetric.LATENCY,
                ProfileMetric.THROUGHPUT,
                ProfileMetric.MEMORY,
                ProfileMetric.FLOPS,
            ],
            "level": ProfilingLevel.DETAILED,
            "warmup_runs": 3 if not quick else 1,
            "num_runs": 15 if not quick else 5,
            "include_backward": False,
            "recommended_batch_sizes": [1, 8, 16, 32],
            "recommended_image_sizes": [224, 384, 512],
        },
        "multimodal": {
            "metrics": [
                ProfileMetric.LATENCY,
                ProfileMetric.THROUGHPUT,
                ProfileMetric.MEMORY,
                ProfileMetric.PARAMETERS,
            ],
            "level": ProfilingLevel.TRACE,
            "warmup_runs": 5 if not quick else 2,
            "num_runs": 15 if not quick else 5,
            "include_backward": False,
            "recommended_batch_sizes": [1, 2, 4, 8],
        },
        "training": {
            "metrics": [
                ProfileMetric.LATENCY,
                ProfileMetric.THROUGHPUT,
                ProfileMetric.MEMORY,
                ProfileMetric.FLOPS,
            ],
            "level": ProfilingLevel.TRACE,
            "warmup_runs": 3 if not quick else 1,
            "num_runs": 10 if not quick else 3,
            "include_backward": True,
        },
    }

    return model_configs.get(model_type.lower(), base_config)
