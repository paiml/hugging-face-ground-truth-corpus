"""Memory estimation and optimization utilities for inference.

This module provides utilities for estimating and optimizing memory usage
during model inference, including parameter memory, activation memory,
KV cache memory, and optimization strategies.

Examples:
    >>> from hf_gtc.inference.memory import estimate_model_memory
    >>> mem = estimate_model_memory(num_parameters=7_000_000_000, dtype_bytes=2)
    >>> mem > 0
    True
    >>> from hf_gtc.inference.memory import create_memory_budget
    >>> budget = create_memory_budget(gpu_memory_gb=24.0)
    >>> budget.gpu_memory_gb
    24.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class MemoryComponent(Enum):
    """Components of memory usage during inference/training.

    Attributes:
        PARAMETERS: Memory for model parameters (weights).
        GRADIENTS: Memory for gradients during training.
        OPTIMIZER_STATES: Memory for optimizer states (Adam momentum, etc.).
        ACTIVATIONS: Memory for activations during forward pass.
        KV_CACHE: Memory for key-value cache during generation.

    Examples:
        >>> MemoryComponent.PARAMETERS.value
        'parameters'
        >>> MemoryComponent.KV_CACHE.value
        'kv_cache'
        >>> MemoryComponent.ACTIVATIONS.value
        'activations'
    """

    PARAMETERS = "parameters"
    GRADIENTS = "gradients"
    OPTIMIZER_STATES = "optimizer_states"
    ACTIVATIONS = "activations"
    KV_CACHE = "kv_cache"


VALID_MEMORY_COMPONENTS = frozenset(c.value for c in MemoryComponent)


class MemoryUnit(Enum):
    """Units for memory measurement.

    Attributes:
        BYTES: Memory in bytes.
        KB: Memory in kilobytes.
        MB: Memory in megabytes.
        GB: Memory in gigabytes.

    Examples:
        >>> MemoryUnit.BYTES.value
        'bytes'
        >>> MemoryUnit.GB.value
        'gb'
        >>> MemoryUnit.MB.value
        'mb'
    """

    BYTES = "bytes"
    KB = "kb"
    MB = "mb"
    GB = "gb"


VALID_MEMORY_UNITS = frozenset(u.value for u in MemoryUnit)


class OptimizationStrategy(Enum):
    """Memory optimization strategies.

    Attributes:
        GRADIENT_CHECKPOINTING: Recompute activations to save memory.
        CPU_OFFLOAD: Offload tensors to CPU memory.
        DISK_OFFLOAD: Offload tensors to disk storage.
        QUANTIZATION: Reduce precision to save memory.

    Examples:
        >>> OptimizationStrategy.GRADIENT_CHECKPOINTING.value
        'gradient_checkpointing'
        >>> OptimizationStrategy.CPU_OFFLOAD.value
        'cpu_offload'
        >>> OptimizationStrategy.QUANTIZATION.value
        'quantization'
    """

    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    CPU_OFFLOAD = "cpu_offload"
    DISK_OFFLOAD = "disk_offload"
    QUANTIZATION = "quantization"


VALID_OPTIMIZATION_STRATEGIES = frozenset(s.value for s in OptimizationStrategy)


# Type aliases
MemoryComponentStr = Literal[
    "parameters", "gradients", "optimizer_states", "activations", "kv_cache"
]
MemoryUnitStr = Literal["bytes", "kb", "mb", "gb"]
OptimizationStrategyStr = Literal[
    "gradient_checkpointing", "cpu_offload", "disk_offload", "quantization"
]
ModelSizeStr = Literal["small", "medium", "large", "xlarge"]
BottleneckStr = Literal["parameters", "activations", "kv_cache", "total", "none"]


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Estimated memory usage for model components.

    Attributes:
        parameters_mb: Memory for model parameters in megabytes.
        activations_mb: Memory for activations in megabytes.
        kv_cache_mb: Memory for KV cache in megabytes.
        total_mb: Total memory usage in megabytes.

    Examples:
        >>> estimate = MemoryEstimate(
        ...     parameters_mb=14000.0,
        ...     activations_mb=2000.0,
        ...     kv_cache_mb=1000.0,
        ...     total_mb=17000.0,
        ... )
        >>> estimate.parameters_mb
        14000.0
        >>> estimate.total_mb
        17000.0
    """

    parameters_mb: float
    activations_mb: float
    kv_cache_mb: float
    total_mb: float


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """Memory budget constraints for inference.

    Attributes:
        gpu_memory_gb: Available GPU memory in gigabytes.
        cpu_memory_gb: Available CPU memory in gigabytes.
        allow_offload: Whether to allow offloading to CPU/disk.

    Examples:
        >>> budget = MemoryBudget(
        ...     gpu_memory_gb=24.0,
        ...     cpu_memory_gb=64.0,
        ...     allow_offload=True,
        ... )
        >>> budget.gpu_memory_gb
        24.0
        >>> budget.allow_offload
        True
    """

    gpu_memory_gb: float
    cpu_memory_gb: float
    allow_offload: bool


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Configuration for memory-aware inference.

    Attributes:
        budget: Memory budget constraints.
        optimization_strategies: Tuple of enabled optimization strategies.
        batch_size: Batch size for inference.
        sequence_length: Maximum sequence length.

    Examples:
        >>> budget = MemoryBudget(24.0, 64.0, True)
        >>> config = MemoryConfig(
        ...     budget=budget,
        ...     optimization_strategies=(OptimizationStrategy.GRADIENT_CHECKPOINTING,),
        ...     batch_size=4,
        ...     sequence_length=2048,
        ... )
        >>> config.batch_size
        4
        >>> config.sequence_length
        2048
    """

    budget: MemoryBudget
    optimization_strategies: tuple[OptimizationStrategy, ...]
    batch_size: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Runtime memory statistics.

    Attributes:
        peak_memory_mb: Peak memory usage in megabytes.
        allocated_mb: Currently allocated memory in megabytes.
        reserved_mb: Reserved memory in megabytes.
        utilization: Memory utilization as a ratio (0.0 to 1.0).

    Examples:
        >>> stats = MemoryStats(
        ...     peak_memory_mb=20000.0,
        ...     allocated_mb=18000.0,
        ...     reserved_mb=22000.0,
        ...     utilization=0.82,
        ... )
        >>> stats.peak_memory_mb
        20000.0
        >>> stats.utilization
        0.82
    """

    peak_memory_mb: float
    allocated_mb: float
    reserved_mb: float
    utilization: float


def validate_memory_estimate(estimate: MemoryEstimate) -> None:
    """Validate memory estimate.

    Args:
        estimate: Memory estimate to validate.

    Raises:
        ValueError: If estimate values are invalid.

    Examples:
        >>> estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        >>> validate_memory_estimate(estimate)  # No error

        >>> bad = MemoryEstimate(-100.0, 2000.0, 1000.0, 17000.0)
        >>> validate_memory_estimate(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: parameters_mb cannot be negative
    """
    if estimate.parameters_mb < 0:
        msg = f"parameters_mb cannot be negative, got {estimate.parameters_mb}"
        raise ValueError(msg)

    if estimate.activations_mb < 0:
        msg = f"activations_mb cannot be negative, got {estimate.activations_mb}"
        raise ValueError(msg)

    if estimate.kv_cache_mb < 0:
        msg = f"kv_cache_mb cannot be negative, got {estimate.kv_cache_mb}"
        raise ValueError(msg)

    if estimate.total_mb < 0:
        msg = f"total_mb cannot be negative, got {estimate.total_mb}"
        raise ValueError(msg)


def validate_memory_budget(budget: MemoryBudget) -> None:
    """Validate memory budget.

    Args:
        budget: Memory budget to validate.

    Raises:
        ValueError: If budget values are invalid.

    Examples:
        >>> budget = MemoryBudget(24.0, 64.0, True)
        >>> validate_memory_budget(budget)  # No error

        >>> bad = MemoryBudget(-1.0, 64.0, True)
        >>> validate_memory_budget(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gpu_memory_gb cannot be negative
    """
    if budget.gpu_memory_gb < 0:
        msg = f"gpu_memory_gb cannot be negative, got {budget.gpu_memory_gb}"
        raise ValueError(msg)

    if budget.cpu_memory_gb < 0:
        msg = f"cpu_memory_gb cannot be negative, got {budget.cpu_memory_gb}"
        raise ValueError(msg)


def validate_memory_config(config: MemoryConfig) -> None:
    """Validate memory configuration.

    Args:
        config: Memory configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> budget = MemoryBudget(24.0, 64.0, True)
        >>> config = MemoryConfig(budget, (), 4, 2048)
        >>> validate_memory_config(config)  # No error

        >>> bad_config = MemoryConfig(budget, (), 0, 2048)
        >>> validate_memory_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    validate_memory_budget(config.budget)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.sequence_length <= 0:
        msg = f"sequence_length must be positive, got {config.sequence_length}"
        raise ValueError(msg)


def validate_memory_stats(stats: MemoryStats) -> None:
    """Validate memory statistics.

    Args:
        stats: Memory statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        >>> validate_memory_stats(stats)  # No error

        >>> bad = MemoryStats(-100.0, 18000.0, 22000.0, 0.82)
        >>> validate_memory_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: peak_memory_mb cannot be negative
    """
    if stats.peak_memory_mb < 0:
        msg = f"peak_memory_mb cannot be negative, got {stats.peak_memory_mb}"
        raise ValueError(msg)

    if stats.allocated_mb < 0:
        msg = f"allocated_mb cannot be negative, got {stats.allocated_mb}"
        raise ValueError(msg)

    if stats.reserved_mb < 0:
        msg = f"reserved_mb cannot be negative, got {stats.reserved_mb}"
        raise ValueError(msg)

    if not 0.0 <= stats.utilization <= 1.0:
        msg = f"utilization must be between 0.0 and 1.0, got {stats.utilization}"
        raise ValueError(msg)


def create_memory_estimate(
    parameters_mb: float,
    activations_mb: float = 0.0,
    kv_cache_mb: float = 0.0,
    total_mb: float | None = None,
) -> MemoryEstimate:
    """Create a memory estimate.

    Args:
        parameters_mb: Memory for parameters in MB.
        activations_mb: Memory for activations in MB. Defaults to 0.0.
        kv_cache_mb: Memory for KV cache in MB. Defaults to 0.0.
        total_mb: Total memory in MB. Computed if not provided.

    Returns:
        MemoryEstimate with the specified values.

    Raises:
        ValueError: If values are invalid.

    Examples:
        >>> estimate = create_memory_estimate(14000.0)
        >>> estimate.parameters_mb
        14000.0
        >>> estimate.total_mb
        14000.0

        >>> estimate = create_memory_estimate(14000.0, 2000.0, 1000.0)
        >>> estimate.total_mb
        17000.0

        >>> create_memory_estimate(-100.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: parameters_mb cannot be negative
    """
    if total_mb is None:
        total_mb = parameters_mb + activations_mb + kv_cache_mb

    estimate = MemoryEstimate(
        parameters_mb=parameters_mb,
        activations_mb=activations_mb,
        kv_cache_mb=kv_cache_mb,
        total_mb=total_mb,
    )
    validate_memory_estimate(estimate)
    return estimate


def create_memory_budget(
    gpu_memory_gb: float = 24.0,
    cpu_memory_gb: float = 64.0,
    allow_offload: bool = True,
) -> MemoryBudget:
    """Create a memory budget.

    Args:
        gpu_memory_gb: GPU memory in GB. Defaults to 24.0.
        cpu_memory_gb: CPU memory in GB. Defaults to 64.0.
        allow_offload: Allow offloading. Defaults to True.

    Returns:
        MemoryBudget with the specified values.

    Raises:
        ValueError: If values are invalid.

    Examples:
        >>> budget = create_memory_budget(gpu_memory_gb=48.0)
        >>> budget.gpu_memory_gb
        48.0
        >>> budget.allow_offload
        True

        >>> budget = create_memory_budget(gpu_memory_gb=24.0, allow_offload=False)
        >>> budget.allow_offload
        False

        >>> create_memory_budget(gpu_memory_gb=-1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gpu_memory_gb cannot be negative
    """
    budget = MemoryBudget(
        gpu_memory_gb=gpu_memory_gb,
        cpu_memory_gb=cpu_memory_gb,
        allow_offload=allow_offload,
    )
    validate_memory_budget(budget)
    return budget


def create_memory_config(
    budget: MemoryBudget | None = None,
    optimization_strategies: tuple[str, ...] = (),
    batch_size: int = 1,
    sequence_length: int = 2048,
) -> MemoryConfig:
    """Create a memory configuration.

    Args:
        budget: Memory budget. Creates default if not provided.
        optimization_strategies: Tuple of strategy names. Defaults to ().
        batch_size: Batch size. Defaults to 1.
        sequence_length: Sequence length. Defaults to 2048.

    Returns:
        MemoryConfig with the specified values.

    Raises:
        ValueError: If values are invalid.

    Examples:
        >>> config = create_memory_config(batch_size=4)
        >>> config.batch_size
        4
        >>> config.sequence_length
        2048

        >>> config = create_memory_config(
        ...     optimization_strategies=("gradient_checkpointing",)
        ... )
        >>> len(config.optimization_strategies)
        1

        >>> create_memory_config(batch_size=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if budget is None:
        budget = create_memory_budget()

    # Convert strategy strings to enums
    strategy_enums: list[OptimizationStrategy] = []
    for strategy in optimization_strategies:
        if strategy not in VALID_OPTIMIZATION_STRATEGIES:
            msg = (
                f"optimization strategy must be one of "
                f"{VALID_OPTIMIZATION_STRATEGIES}, got '{strategy}'"
            )
            raise ValueError(msg)
        strategy_enums.append(OptimizationStrategy(strategy))

    config = MemoryConfig(
        budget=budget,
        optimization_strategies=tuple(strategy_enums),
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    validate_memory_config(config)
    return config


def create_memory_stats(
    peak_memory_mb: float,
    allocated_mb: float,
    reserved_mb: float,
    utilization: float | None = None,
) -> MemoryStats:
    """Create memory statistics.

    Args:
        peak_memory_mb: Peak memory in MB.
        allocated_mb: Allocated memory in MB.
        reserved_mb: Reserved memory in MB.
        utilization: Memory utilization. Computed if not provided.

    Returns:
        MemoryStats with the specified values.

    Raises:
        ValueError: If values are invalid.

    Examples:
        >>> stats = create_memory_stats(20000.0, 18000.0, 22000.0)
        >>> stats.peak_memory_mb
        20000.0
        >>> 0.0 <= stats.utilization <= 1.0
        True

        >>> stats = create_memory_stats(20000.0, 18000.0, 22000.0, 0.82)
        >>> stats.utilization
        0.82

        >>> create_memory_stats(-100.0, 18000.0, 22000.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: peak_memory_mb cannot be negative
    """
    if utilization is None:
        utilization = allocated_mb / reserved_mb if reserved_mb > 0 else 0.0

    stats = MemoryStats(
        peak_memory_mb=peak_memory_mb,
        allocated_mb=allocated_mb,
        reserved_mb=reserved_mb,
        utilization=utilization,
    )
    validate_memory_stats(stats)
    return stats


def list_memory_components() -> list[str]:
    """List available memory components.

    Returns:
        Sorted list of memory component names.

    Examples:
        >>> components = list_memory_components()
        >>> "parameters" in components
        True
        >>> "kv_cache" in components
        True
        >>> components == sorted(components)
        True
    """
    return sorted(VALID_MEMORY_COMPONENTS)


def list_memory_units() -> list[str]:
    """List available memory units.

    Returns:
        Sorted list of memory unit names.

    Examples:
        >>> units = list_memory_units()
        >>> "bytes" in units
        True
        >>> "gb" in units
        True
        >>> units == sorted(units)
        True
    """
    return sorted(VALID_MEMORY_UNITS)


def list_optimization_strategies() -> list[str]:
    """List available optimization strategies.

    Returns:
        Sorted list of optimization strategy names.

    Examples:
        >>> strategies = list_optimization_strategies()
        >>> "gradient_checkpointing" in strategies
        True
        >>> "cpu_offload" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_OPTIMIZATION_STRATEGIES)


def get_memory_component(name: str) -> MemoryComponent:
    """Get a memory component by name.

    Args:
        name: Name of the memory component.

    Returns:
        The corresponding MemoryComponent enum value.

    Raises:
        ValueError: If name is not a valid memory component.

    Examples:
        >>> get_memory_component("parameters")
        <MemoryComponent.PARAMETERS: 'parameters'>
        >>> get_memory_component("kv_cache")
        <MemoryComponent.KV_CACHE: 'kv_cache'>

        >>> get_memory_component("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory component must be one of
    """
    if name not in VALID_MEMORY_COMPONENTS:
        msg = f"memory component must be one of {VALID_MEMORY_COMPONENTS}, got '{name}'"
        raise ValueError(msg)
    return MemoryComponent(name)


def get_memory_unit(name: str) -> MemoryUnit:
    """Get a memory unit by name.

    Args:
        name: Name of the memory unit.

    Returns:
        The corresponding MemoryUnit enum value.

    Raises:
        ValueError: If name is not a valid memory unit.

    Examples:
        >>> get_memory_unit("bytes")
        <MemoryUnit.BYTES: 'bytes'>
        >>> get_memory_unit("gb")
        <MemoryUnit.GB: 'gb'>

        >>> get_memory_unit("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory unit must be one of
    """
    if name not in VALID_MEMORY_UNITS:
        msg = f"memory unit must be one of {VALID_MEMORY_UNITS}, got '{name}'"
        raise ValueError(msg)
    return MemoryUnit(name)


def get_optimization_strategy(name: str) -> OptimizationStrategy:
    """Get an optimization strategy by name.

    Args:
        name: Name of the optimization strategy.

    Returns:
        The corresponding OptimizationStrategy enum value.

    Raises:
        ValueError: If name is not a valid optimization strategy.

    Examples:
        >>> get_optimization_strategy("gradient_checkpointing")
        <OptimizationStrategy.GRADIENT_CHECKPOINTING: 'gradient_checkpointing'>
        >>> get_optimization_strategy("cpu_offload")
        <OptimizationStrategy.CPU_OFFLOAD: 'cpu_offload'>

        >>> get_optimization_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: optimization strategy must be one of
    """
    if name not in VALID_OPTIMIZATION_STRATEGIES:
        msg = (
            f"optimization strategy must be one of {VALID_OPTIMIZATION_STRATEGIES}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return OptimizationStrategy(name)


def estimate_model_memory(
    num_parameters: int,
    dtype_bytes: int = 2,
) -> float:
    """Estimate memory for model parameters.

    Args:
        num_parameters: Number of model parameters.
        dtype_bytes: Bytes per parameter. Defaults to 2 (FP16).

    Returns:
        Estimated memory in megabytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> mem = estimate_model_memory(7_000_000_000)  # 7B model
        >>> 13000 < mem < 14000  # ~13.3 GB in MB
        True

        >>> mem = estimate_model_memory(7_000_000_000, dtype_bytes=4)  # FP32
        >>> 26000 < mem < 27000  # ~26.6 GB in MB
        True

        >>> estimate_model_memory(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive
    """
    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    bytes_total = num_parameters * dtype_bytes
    return bytes_total / (1024 * 1024)


def estimate_inference_memory(
    num_parameters: int,
    batch_size: int = 1,
    sequence_length: int = 2048,
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_heads: int = 32,
    dtype_bytes: int = 2,
) -> MemoryEstimate:
    """Estimate total memory for inference.

    Args:
        num_parameters: Number of model parameters.
        batch_size: Batch size. Defaults to 1.
        sequence_length: Sequence length. Defaults to 2048.
        num_layers: Number of layers. Defaults to 32.
        hidden_size: Hidden size. Defaults to 4096.
        num_heads: Number of attention heads. Defaults to 32.
        dtype_bytes: Bytes per element. Defaults to 2 (FP16).

    Returns:
        MemoryEstimate with breakdown by component.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate = estimate_inference_memory(7_000_000_000)
        >>> estimate.parameters_mb > 0
        True
        >>> estimate.total_mb > estimate.parameters_mb
        True

        >>> estimate = estimate_inference_memory(
        ...     7_000_000_000, batch_size=4, sequence_length=4096
        ... )
        >>> estimate.kv_cache_mb > 0
        True

        >>> estimate_inference_memory(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive
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

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    # Parameter memory
    params_mb = estimate_model_memory(num_parameters, dtype_bytes)

    # Activation memory (simplified estimate)
    # Activations: batch * seq * hidden * layers * 2 (for forward pass intermediate)
    activation_elements = batch_size * sequence_length * hidden_size * num_layers * 2
    activations_mb = (activation_elements * dtype_bytes) / (1024 * 1024)

    # KV cache memory
    # KV cache: 2 * batch * layers * seq * heads * head_dim * dtype
    head_dim = hidden_size // num_heads
    kv_elements = 2 * batch_size * num_layers * sequence_length * num_heads * head_dim
    kv_cache_mb = (kv_elements * dtype_bytes) / (1024 * 1024)

    total_mb = params_mb + activations_mb + kv_cache_mb

    return create_memory_estimate(
        parameters_mb=params_mb,
        activations_mb=activations_mb,
        kv_cache_mb=kv_cache_mb,
        total_mb=total_mb,
    )


def estimate_training_memory(
    num_parameters: int,
    batch_size: int = 1,
    sequence_length: int = 2048,
    num_layers: int = 32,
    hidden_size: int = 4096,
    dtype_bytes: int = 2,
    optimizer_states: int = 2,
    gradient_checkpointing: bool = False,
) -> MemoryEstimate:
    """Estimate total memory for training.

    Args:
        num_parameters: Number of model parameters.
        batch_size: Batch size. Defaults to 1.
        sequence_length: Sequence length. Defaults to 2048.
        num_layers: Number of layers. Defaults to 32.
        hidden_size: Hidden size. Defaults to 4096.
        dtype_bytes: Bytes per element. Defaults to 2 (FP16).
        optimizer_states: Number of optimizer state tensors. Defaults to 2 (Adam).
        gradient_checkpointing: Whether gradient checkpointing is enabled.

    Returns:
        MemoryEstimate with breakdown by component.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate = estimate_training_memory(7_000_000_000)
        >>> estimate.parameters_mb > 0  # includes params + grads + opt states
        True
        >>> estimate.total_mb > estimate.parameters_mb  # total includes activations
        True

        >>> with_cp = estimate_training_memory(
        ...     7_000_000_000, gradient_checkpointing=True
        ... )
        >>> without_cp = estimate_training_memory(
        ...     7_000_000_000, gradient_checkpointing=False
        ... )
        >>> with_cp.activations_mb < without_cp.activations_mb
        True

        >>> estimate_training_memory(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive
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

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if dtype_bytes <= 0:
        msg = f"dtype_bytes must be positive, got {dtype_bytes}"
        raise ValueError(msg)

    if optimizer_states < 0:
        msg = f"optimizer_states cannot be negative, got {optimizer_states}"
        raise ValueError(msg)

    # Parameter memory (includes gradients and optimizer states)
    params_bytes = num_parameters * dtype_bytes
    grads_bytes = num_parameters * dtype_bytes
    opt_bytes = num_parameters * dtype_bytes * optimizer_states
    params_mb = (params_bytes + grads_bytes + opt_bytes) / (1024 * 1024)

    # Activation memory
    # Full activations: batch * seq * hidden * layers * 4 (for forward + backward)
    if gradient_checkpointing:
        # With checkpointing, only store sqrt(layers) checkpoints
        import math

        effective_layers = int(math.sqrt(num_layers)) + 1
    else:
        effective_layers = num_layers

    activation_elements = (
        batch_size * sequence_length * hidden_size * effective_layers * 4
    )
    activations_mb = (activation_elements * dtype_bytes) / (1024 * 1024)

    # No KV cache during training (use full attention)
    kv_cache_mb = 0.0

    total_mb = params_mb + activations_mb + kv_cache_mb

    return create_memory_estimate(
        parameters_mb=params_mb,
        activations_mb=activations_mb,
        kv_cache_mb=kv_cache_mb,
        total_mb=total_mb,
    )


def find_max_batch_size(
    num_parameters: int,
    gpu_memory_gb: float,
    sequence_length: int = 2048,
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_heads: int = 32,
    dtype_bytes: int = 2,
    memory_fraction: float = 0.9,
) -> int:
    """Find maximum batch size that fits in GPU memory.

    Args:
        num_parameters: Number of model parameters.
        gpu_memory_gb: Available GPU memory in GB.
        sequence_length: Sequence length. Defaults to 2048.
        num_layers: Number of layers. Defaults to 32.
        hidden_size: Hidden size. Defaults to 4096.
        num_heads: Number of attention heads. Defaults to 32.
        dtype_bytes: Bytes per element. Defaults to 2 (FP16).
        memory_fraction: Fraction of memory to use. Defaults to 0.9.

    Returns:
        Maximum batch size (at least 1).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> batch_size = find_max_batch_size(7_000_000_000, 24.0)
        >>> batch_size >= 1
        True

        >>> batch_size_80 = find_max_batch_size(7_000_000_000, 80.0)
        >>> batch_size_24 = find_max_batch_size(7_000_000_000, 24.0)
        >>> batch_size_80 >= batch_size_24
        True

        >>> find_max_batch_size(0, 24.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive
    """
    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    if gpu_memory_gb <= 0:
        msg = f"gpu_memory_gb must be positive, got {gpu_memory_gb}"
        raise ValueError(msg)

    if not 0.0 < memory_fraction <= 1.0:
        msg = f"memory_fraction must be between 0.0 and 1.0, got {memory_fraction}"
        raise ValueError(msg)

    # Available memory in MB
    available_mb = gpu_memory_gb * 1024 * memory_fraction

    # Binary search for max batch size
    low, high = 1, 1024
    result = 1

    while low <= high:
        mid = (low + high) // 2
        estimate = estimate_inference_memory(
            num_parameters=num_parameters,
            batch_size=mid,
            sequence_length=sequence_length,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dtype_bytes=dtype_bytes,
        )

        if estimate.total_mb <= available_mb:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result


def detect_memory_bottleneck(
    estimate: MemoryEstimate,
    budget: MemoryBudget,
) -> BottleneckStr:
    """Detect the main memory bottleneck.

    Args:
        estimate: Memory estimate for the model.
        budget: Memory budget constraints.

    Returns:
        The component causing the bottleneck, or "none" if no bottleneck.

    Examples:
        >>> estimate = create_memory_estimate(20000.0, 5000.0, 2000.0)
        >>> budget = create_memory_budget(gpu_memory_gb=24.0)  # 24GB = 24576 MB
        >>> detect_memory_bottleneck(estimate, budget)
        'parameters'

        >>> small_estimate = create_memory_estimate(5000.0, 1000.0, 500.0)
        >>> detect_memory_bottleneck(small_estimate, budget)
        'none'

        >>> kv_heavy = create_memory_estimate(5000.0, 1000.0, 20000.0)
        >>> detect_memory_bottleneck(kv_heavy, budget)
        'kv_cache'
    """
    budget_mb = budget.gpu_memory_gb * 1024

    if estimate.total_mb <= budget_mb:
        return "none"

    # Find the largest component exceeding reasonable thresholds
    components = [
        ("parameters", estimate.parameters_mb),
        ("activations", estimate.activations_mb),
        ("kv_cache", estimate.kv_cache_mb),
    ]

    # Sort by size descending
    components.sort(key=lambda x: x[1], reverse=True)

    # Return the largest component as the bottleneck
    if components[0][1] > budget_mb * 0.5:
        return components[0][0]  # type: ignore[return-value]

    return "total"


def format_memory_stats(stats: MemoryStats) -> str:
    """Format memory statistics as a human-readable string.

    Args:
        stats: Memory statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        >>> formatted = format_memory_stats(stats)
        >>> "Peak Memory: 20000.00 MB" in formatted
        True
        >>> "Utilization: 82.00%" in formatted
        True

        >>> stats_low = MemoryStats(1000.0, 500.0, 2000.0, 0.25)
        >>> "Utilization: 25.00%" in format_memory_stats(stats_low)
        True
    """
    lines = [
        f"Peak Memory: {stats.peak_memory_mb:.2f} MB",
        f"Allocated: {stats.allocated_mb:.2f} MB",
        f"Reserved: {stats.reserved_mb:.2f} MB",
        f"Utilization: {stats.utilization * 100:.2f}%",
    ]
    return "\n".join(lines)


def format_memory_estimate(estimate: MemoryEstimate) -> str:
    """Format memory estimate as a human-readable string.

    Args:
        estimate: Memory estimate to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        >>> formatted = format_memory_estimate(estimate)
        >>> "Parameters: 14000.00 MB" in formatted
        True
        >>> "Total: 17000.00 MB" in formatted
        True

        >>> estimate_gb = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        >>> "KV Cache: 1000.00 MB" in format_memory_estimate(estimate_gb)
        True
    """
    lines = [
        f"Parameters: {estimate.parameters_mb:.2f} MB",
        f"Activations: {estimate.activations_mb:.2f} MB",
        f"KV Cache: {estimate.kv_cache_mb:.2f} MB",
        f"Total: {estimate.total_mb:.2f} MB",
    ]
    return "\n".join(lines)


def get_recommended_memory_config(
    model_size: ModelSizeStr,
    gpu_memory_gb: float = 24.0,
    use_case: Literal["inference", "training"] = "inference",
) -> MemoryConfig:
    """Get recommended memory configuration for model and use case.

    Args:
        model_size: Model size category.
        gpu_memory_gb: Available GPU memory in GB. Defaults to 24.0.
        use_case: Use case ("inference" or "training"). Defaults to "inference".

    Returns:
        Recommended MemoryConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_memory_config("large", 24.0)
        >>> config.batch_size >= 1
        True
        >>> len(config.optimization_strategies) >= 0
        True

        >>> config = get_recommended_memory_config("xlarge", 24.0, "training")
        >>> strats = config.optimization_strategies
        >>> OptimizationStrategy.GRADIENT_CHECKPOINTING in strats
        True

        >>> get_recommended_memory_config("invalid", 24.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size must be one of
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    valid_use_cases = {"inference", "training"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    if gpu_memory_gb <= 0:
        msg = f"gpu_memory_gb must be positive, got {gpu_memory_gb}"
        raise ValueError(msg)

    # Model size to parameter count mapping (approximate)
    size_to_params = {
        "small": 1_000_000_000,  # 1B
        "medium": 7_000_000_000,  # 7B
        "large": 13_000_000_000,  # 13B
        "xlarge": 70_000_000_000,  # 70B
    }

    num_params = size_to_params[model_size]
    budget = create_memory_budget(gpu_memory_gb=gpu_memory_gb)

    strategies: list[str] = []

    if use_case == "training":
        # Training requires more memory optimizations
        if model_size in ("large", "xlarge"):
            strategies.append("gradient_checkpointing")

        if model_size == "xlarge" or gpu_memory_gb < 48:
            strategies.append("cpu_offload")

        batch_size = max(1, find_max_batch_size(num_params, gpu_memory_gb) // 4)
        sequence_length = 2048

    else:
        # Inference recommendations
        if model_size == "xlarge" and gpu_memory_gb < 80:
            strategies.append("quantization")
            strategies.append("cpu_offload")
        elif model_size == "large" and gpu_memory_gb < 24:
            strategies.append("quantization")

        batch_size = find_max_batch_size(num_params, gpu_memory_gb)
        sequence_length = 4096 if gpu_memory_gb >= 48 else 2048

    return create_memory_config(
        budget=budget,
        optimization_strategies=tuple(strategies),
        batch_size=batch_size,
        sequence_length=sequence_length,
    )


def convert_memory_units(
    value: float,
    from_unit: MemoryUnitStr,
    to_unit: MemoryUnitStr,
) -> float:
    """Convert memory value between units.

    Args:
        value: Memory value to convert.
        from_unit: Source unit.
        to_unit: Target unit.

    Returns:
        Converted memory value.

    Raises:
        ValueError: If units are invalid.

    Examples:
        >>> convert_memory_units(1024, "mb", "gb")
        1.0
        >>> convert_memory_units(1, "gb", "mb")
        1024.0
        >>> convert_memory_units(1, "kb", "bytes")
        1024.0

        >>> convert_memory_units(1, "invalid", "gb")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: from_unit must be one of
    """
    if from_unit not in VALID_MEMORY_UNITS:
        msg = f"from_unit must be one of {VALID_MEMORY_UNITS}, got '{from_unit}'"
        raise ValueError(msg)

    if to_unit not in VALID_MEMORY_UNITS:
        msg = f"to_unit must be one of {VALID_MEMORY_UNITS}, got '{to_unit}'"
        raise ValueError(msg)

    # Convert to bytes first
    unit_to_bytes = {
        "bytes": 1,
        "kb": 1024,
        "mb": 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
    }

    bytes_value = value * unit_to_bytes[from_unit]
    return bytes_value / unit_to_bytes[to_unit]
