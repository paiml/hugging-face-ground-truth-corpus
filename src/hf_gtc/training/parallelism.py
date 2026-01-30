"""Distributed training parallelism utilities.

This module provides functions for configuring distributed training
parallelism strategies including tensor parallelism, pipeline parallelism,
sequence parallelism, FSDP, and expert parallelism.

Examples:
    >>> from hf_gtc.training.parallelism import (
    ...     create_parallel_config,
    ...     ParallelismType,
    ... )
    >>> config = create_parallel_config()
    >>> config.dp_size
    1
    >>> config.tp_config.tp_size
    1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ParallelismType(Enum):
    """Types of distributed training parallelism.

    Attributes:
        DATA: Data parallelism - replicate model, shard data.
        TENSOR: Tensor parallelism - shard model layers horizontally.
        PIPELINE: Pipeline parallelism - shard model layers vertically.
        SEQUENCE: Sequence parallelism - shard sequence dimension.
        EXPERT: Expert parallelism - distribute MoE experts.

    Examples:
        >>> ParallelismType.DATA.value
        'data'
        >>> ParallelismType.TENSOR.value
        'tensor'
        >>> ParallelismType.PIPELINE.value
        'pipeline'
        >>> ParallelismType.SEQUENCE.value
        'sequence'
        >>> ParallelismType.EXPERT.value
        'expert'
    """

    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    SEQUENCE = "sequence"
    EXPERT = "expert"


class ParallelShardingStrategy(Enum):
    """FSDP sharding strategies for distributed training.

    Attributes:
        FULL_SHARD: Full sharding of parameters, gradients, and optimizer states.
        SHARD_GRAD_OP: Shard only gradients and optimizer states.
        NO_SHARD: No sharding (equivalent to DDP).
        HYBRID: Hybrid sharding within and across nodes.

    Examples:
        >>> ParallelShardingStrategy.FULL_SHARD.value
        'full_shard'
        >>> ParallelShardingStrategy.SHARD_GRAD_OP.value
        'shard_grad_op'
        >>> ParallelShardingStrategy.NO_SHARD.value
        'no_shard'
        >>> ParallelShardingStrategy.HYBRID.value
        'hybrid'
    """

    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID = "hybrid"


class CommunicationBackend(Enum):
    """Communication backends for distributed training.

    Attributes:
        NCCL: NVIDIA Collective Communications Library (GPU).
        GLOO: Facebook's Gloo library (CPU/GPU).
        MPI: Message Passing Interface.

    Examples:
        >>> CommunicationBackend.NCCL.value
        'nccl'
        >>> CommunicationBackend.GLOO.value
        'gloo'
        >>> CommunicationBackend.MPI.value
        'mpi'
    """

    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


VALID_PARALLELISM_TYPES = frozenset(pt.value for pt in ParallelismType)
VALID_PARALLEL_SHARDING_STRATEGIES = frozenset(
    ss.value for ss in ParallelShardingStrategy
)
VALID_COMMUNICATION_BACKENDS = frozenset(cb.value for cb in CommunicationBackend)


@dataclass(frozen=True, slots=True)
class TensorParallelConfig:
    """Configuration for tensor parallelism.

    Attributes:
        tp_size: Number of tensor parallel ranks.
        partition_dim: Dimension to partition (0=row, 1=column).
        sequence_parallel: Enable sequence parallelism with TP.
        async_comm: Enable asynchronous communication.

    Examples:
        >>> config = TensorParallelConfig(
        ...     tp_size=8,
        ...     partition_dim=1,
        ...     sequence_parallel=True,
        ...     async_comm=True,
        ... )
        >>> config.tp_size
        8
        >>> config.sequence_parallel
        True
    """

    tp_size: int
    partition_dim: int
    sequence_parallel: bool
    async_comm: bool


@dataclass(frozen=True, slots=True)
class PipelineParallelConfig:
    """Configuration for pipeline parallelism.

    Attributes:
        pp_size: Number of pipeline parallel stages.
        num_microbatches: Number of microbatches for pipelining.
        interleave: Enable interleaved pipeline scheduling.
        activation_checkpointing: Enable activation checkpointing.

    Examples:
        >>> config = PipelineParallelConfig(
        ...     pp_size=4,
        ...     num_microbatches=8,
        ...     interleave=True,
        ...     activation_checkpointing=True,
        ... )
        >>> config.pp_size
        4
        >>> config.num_microbatches
        8
    """

    pp_size: int
    num_microbatches: int
    interleave: bool
    activation_checkpointing: bool


@dataclass(frozen=True, slots=True)
class ParallelFSDPConfig:
    """Configuration for Fully Sharded Data Parallel.

    Attributes:
        sharding_strategy: Strategy for sharding parameters.
        cpu_offload: Offload parameters to CPU when not in use.
        backward_prefetch: Prefetch parameters during backward pass.
        mixed_precision: Enable mixed precision training.

    Examples:
        >>> config = ParallelFSDPConfig(
        ...     sharding_strategy=ParallelShardingStrategy.FULL_SHARD,
        ...     cpu_offload=False,
        ...     backward_prefetch=True,
        ...     mixed_precision=True,
        ... )
        >>> config.sharding_strategy
        <ParallelShardingStrategy.FULL_SHARD: 'full_shard'>
        >>> config.backward_prefetch
        True
    """

    sharding_strategy: ParallelShardingStrategy
    cpu_offload: bool
    backward_prefetch: bool
    mixed_precision: bool


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Main configuration for distributed training parallelism.

    Attributes:
        dp_size: Data parallel size.
        tp_config: Tensor parallelism configuration.
        pp_config: Pipeline parallelism configuration.
        fsdp_config: FSDP configuration.
        backend: Communication backend.

    Examples:
        >>> tp = TensorParallelConfig(2, 1, True, True)
        >>> pp = PipelineParallelConfig(2, 4, False, True)
        >>> strategy = ParallelShardingStrategy.FULL_SHARD
        >>> fsdp = ParallelFSDPConfig(strategy, False, True, True)
        >>> config = ParallelConfig(
        ...     dp_size=2,
        ...     tp_config=tp,
        ...     pp_config=pp,
        ...     fsdp_config=fsdp,
        ...     backend=CommunicationBackend.NCCL,
        ... )
        >>> config.dp_size
        2
        >>> config.backend
        <CommunicationBackend.NCCL: 'nccl'>
    """

    dp_size: int
    tp_config: TensorParallelConfig
    pp_config: PipelineParallelConfig
    fsdp_config: ParallelFSDPConfig
    backend: CommunicationBackend


@dataclass(frozen=True, slots=True)
class ParallelStats:
    """Statistics for parallel training configuration.

    Attributes:
        world_size: Total number of parallel processes.
        memory_per_device_gb: Estimated memory per device in GB.
        communication_overhead: Estimated communication overhead ratio.
        efficiency: Estimated parallel efficiency.

    Examples:
        >>> stats = ParallelStats(
        ...     world_size=16,
        ...     memory_per_device_gb=40.0,
        ...     communication_overhead=0.15,
        ...     efficiency=0.85,
        ... )
        >>> stats.world_size
        16
        >>> stats.efficiency
        0.85
    """

    world_size: int
    memory_per_device_gb: float
    communication_overhead: float
    efficiency: float


def validate_tensor_parallel_config(config: TensorParallelConfig) -> None:
    """Validate tensor parallel configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = TensorParallelConfig(8, 1, True, True)
        >>> validate_tensor_parallel_config(config)

        >>> bad_config = TensorParallelConfig(0, 1, True, True)
        >>> validate_tensor_parallel_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: tp_size must be positive, got 0
    """
    if config.tp_size <= 0:
        msg = f"tp_size must be positive, got {config.tp_size}"
        raise ValueError(msg)
    if config.partition_dim not in (0, 1):
        msg = f"partition_dim must be 0 or 1, got {config.partition_dim}"
        raise ValueError(msg)
    # Validate tp_size is power of 2 for efficient communication
    if config.tp_size > 1 and (config.tp_size & (config.tp_size - 1)) != 0:
        msg = f"tp_size must be a power of 2, got {config.tp_size}"
        raise ValueError(msg)


def validate_pipeline_parallel_config(config: PipelineParallelConfig) -> None:
    """Validate pipeline parallel configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PipelineParallelConfig(4, 8, True, True)
        >>> validate_pipeline_parallel_config(config)

        >>> bad_config = PipelineParallelConfig(0, 8, True, True)
        >>> validate_pipeline_parallel_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: pp_size must be positive, got 0
    """
    if config.pp_size <= 0:
        msg = f"pp_size must be positive, got {config.pp_size}"
        raise ValueError(msg)
    if config.num_microbatches <= 0:
        msg = f"num_microbatches must be positive, got {config.num_microbatches}"
        raise ValueError(msg)
    # Microbatches should be at least as many as pipeline stages
    if config.num_microbatches < config.pp_size:
        msg = (
            f"num_microbatches ({config.num_microbatches}) must be >= "
            f"pp_size ({config.pp_size})"
        )
        raise ValueError(msg)


def validate_fsdp_config(config: ParallelFSDPConfig) -> None:
    """Validate FSDP configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> strat = ParallelShardingStrategy.FULL_SHARD
        >>> config = ParallelFSDPConfig(strat, False, True, True)
        >>> validate_fsdp_config(config)

        >>> bad_strat = ParallelShardingStrategy.NO_SHARD
        >>> bad_config = ParallelFSDPConfig(bad_strat, True, True, True)
        >>> validate_fsdp_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: cpu_offload requires sharding (not NO_SHARD)
    """
    no_shard = ParallelShardingStrategy.NO_SHARD
    if config.cpu_offload and config.sharding_strategy == no_shard:
        msg = "cpu_offload requires sharding (not NO_SHARD)"
        raise ValueError(msg)


def validate_parallel_config(config: ParallelConfig) -> None:
    """Validate parallel configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> tp = TensorParallelConfig(2, 1, True, True)
        >>> pp = PipelineParallelConfig(2, 4, False, True)
        >>> strat = ParallelShardingStrategy.FULL_SHARD
        >>> fsdp = ParallelFSDPConfig(strat, False, True, True)
        >>> config = ParallelConfig(2, tp, pp, fsdp, CommunicationBackend.NCCL)
        >>> validate_parallel_config(config)

        >>> bad_tp = TensorParallelConfig(0, 1, True, True)
        >>> bad_config = ParallelConfig(2, bad_tp, pp, fsdp, CommunicationBackend.NCCL)
        >>> validate_parallel_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: tp_size must be positive, got 0
    """
    if config.dp_size <= 0:
        msg = f"dp_size must be positive, got {config.dp_size}"
        raise ValueError(msg)
    validate_tensor_parallel_config(config.tp_config)
    validate_pipeline_parallel_config(config.pp_config)
    validate_fsdp_config(config.fsdp_config)


def create_tensor_parallel_config(
    tp_size: int = 1,
    partition_dim: int = 1,
    sequence_parallel: bool = False,
    async_comm: bool = True,
) -> TensorParallelConfig:
    """Create a tensor parallel configuration.

    Args:
        tp_size: Number of tensor parallel ranks.
        partition_dim: Dimension to partition (0=row, 1=column).
        sequence_parallel: Enable sequence parallelism.
        async_comm: Enable asynchronous communication.

    Returns:
        Validated TensorParallelConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_tensor_parallel_config()
        >>> config.tp_size
        1

        >>> config = create_tensor_parallel_config(tp_size=8, sequence_parallel=True)
        >>> config.tp_size
        8
        >>> config.sequence_parallel
        True

        >>> create_tensor_parallel_config(tp_size=0)
        Traceback (most recent call last):
            ...
        ValueError: tp_size must be positive, got 0
    """
    config = TensorParallelConfig(
        tp_size=tp_size,
        partition_dim=partition_dim,
        sequence_parallel=sequence_parallel,
        async_comm=async_comm,
    )
    validate_tensor_parallel_config(config)
    return config


def create_pipeline_parallel_config(
    pp_size: int = 1,
    num_microbatches: int = 1,
    interleave: bool = False,
    activation_checkpointing: bool = True,
) -> PipelineParallelConfig:
    """Create a pipeline parallel configuration.

    Args:
        pp_size: Number of pipeline stages.
        num_microbatches: Number of microbatches.
        interleave: Enable interleaved scheduling.
        activation_checkpointing: Enable activation checkpointing.

    Returns:
        Validated PipelineParallelConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_pipeline_parallel_config()
        >>> config.pp_size
        1

        >>> config = create_pipeline_parallel_config(pp_size=4, num_microbatches=8)
        >>> config.pp_size
        4
        >>> config.num_microbatches
        8

        >>> create_pipeline_parallel_config(pp_size=0)
        Traceback (most recent call last):
            ...
        ValueError: pp_size must be positive, got 0
    """
    config = PipelineParallelConfig(
        pp_size=pp_size,
        num_microbatches=num_microbatches,
        interleave=interleave,
        activation_checkpointing=activation_checkpointing,
    )
    validate_pipeline_parallel_config(config)
    return config


def create_fsdp_config(
    sharding_strategy: str | ParallelShardingStrategy = (
        ParallelShardingStrategy.FULL_SHARD
    ),
    cpu_offload: bool = False,
    backward_prefetch: bool = True,
    mixed_precision: bool = True,
) -> ParallelFSDPConfig:
    """Create an FSDP configuration.

    Args:
        sharding_strategy: Strategy for sharding.
        cpu_offload: Enable CPU offloading.
        backward_prefetch: Enable backward prefetch.
        mixed_precision: Enable mixed precision.

    Returns:
        Validated ParallelFSDPConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_fsdp_config()
        >>> config.sharding_strategy
        <ParallelShardingStrategy.FULL_SHARD: 'full_shard'>

        >>> config = create_fsdp_config(sharding_strategy="shard_grad_op")
        >>> config.sharding_strategy
        <ParallelShardingStrategy.SHARD_GRAD_OP: 'shard_grad_op'>

        >>> create_fsdp_config(sharding_strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sharding_strategy must be one of
    """
    if isinstance(sharding_strategy, str):
        sharding_strategy = get_sharding_strategy(sharding_strategy)

    config = ParallelFSDPConfig(
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        mixed_precision=mixed_precision,
    )
    validate_fsdp_config(config)
    return config


def create_parallel_config(
    dp_size: int = 1,
    tp_config: TensorParallelConfig | None = None,
    pp_config: PipelineParallelConfig | None = None,
    fsdp_config: ParallelFSDPConfig | None = None,
    backend: str | CommunicationBackend = CommunicationBackend.NCCL,
) -> ParallelConfig:
    """Create a parallel configuration.

    Args:
        dp_size: Data parallel size.
        tp_config: Tensor parallelism config.
        pp_config: Pipeline parallelism config.
        fsdp_config: FSDP config.
        backend: Communication backend.

    Returns:
        Validated ParallelConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_parallel_config()
        >>> config.dp_size
        1

        >>> tp = create_tensor_parallel_config(tp_size=8)
        >>> config = create_parallel_config(dp_size=2, tp_config=tp)
        >>> config.dp_size
        2
        >>> config.tp_config.tp_size
        8

        >>> create_parallel_config(dp_size=0)
        Traceback (most recent call last):
            ...
        ValueError: dp_size must be positive, got 0
    """
    if tp_config is None:
        tp_config = create_tensor_parallel_config()
    if pp_config is None:
        pp_config = create_pipeline_parallel_config()
    if fsdp_config is None:
        fsdp_config = create_fsdp_config()
    if isinstance(backend, str):
        backend = get_communication_backend(backend)

    config = ParallelConfig(
        dp_size=dp_size,
        tp_config=tp_config,
        pp_config=pp_config,
        fsdp_config=fsdp_config,
        backend=backend,
    )
    validate_parallel_config(config)
    return config


def list_parallelism_types() -> list[str]:
    """List all available parallelism types.

    Returns:
        Sorted list of parallelism type names.

    Examples:
        >>> types = list_parallelism_types()
        >>> "tensor" in types
        True
        >>> "pipeline" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_PARALLELISM_TYPES)


def list_sharding_strategies() -> list[str]:
    """List all available sharding strategies.

    Returns:
        Sorted list of sharding strategy names.

    Examples:
        >>> strategies = list_sharding_strategies()
        >>> "full_shard" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_PARALLEL_SHARDING_STRATEGIES)


def list_communication_backends() -> list[str]:
    """List all available communication backends.

    Returns:
        Sorted list of backend names.

    Examples:
        >>> backends = list_communication_backends()
        >>> "nccl" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_COMMUNICATION_BACKENDS)


def get_parallelism_type(name: str) -> ParallelismType:
    """Get parallelism type from string name.

    Args:
        name: Name of the parallelism type.

    Returns:
        Corresponding ParallelismType enum.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_parallelism_type("tensor")
        <ParallelismType.TENSOR: 'tensor'>
        >>> get_parallelism_type("pipeline")
        <ParallelismType.PIPELINE: 'pipeline'>

        >>> get_parallelism_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: parallelism_type must be one of ...
    """
    if name not in VALID_PARALLELISM_TYPES:
        msg = f"parallelism_type must be one of {VALID_PARALLELISM_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ParallelismType(name)


def get_sharding_strategy(name: str) -> ParallelShardingStrategy:
    """Get sharding strategy from string name.

    Args:
        name: Name of the sharding strategy.

    Returns:
        Corresponding ParallelShardingStrategy enum.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_sharding_strategy("full_shard")
        <ParallelShardingStrategy.FULL_SHARD: 'full_shard'>
        >>> get_sharding_strategy("hybrid")
        <ParallelShardingStrategy.HYBRID: 'hybrid'>

        >>> get_sharding_strategy("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: sharding_strategy must be one of ...
    """
    if name not in VALID_PARALLEL_SHARDING_STRATEGIES:
        msg = (
            f"sharding_strategy must be one of "
            f"{VALID_PARALLEL_SHARDING_STRATEGIES}, got '{name}'"
        )
        raise ValueError(msg)
    return ParallelShardingStrategy(name)


def get_communication_backend(name: str) -> CommunicationBackend:
    """Get communication backend from string name.

    Args:
        name: Name of the communication backend.

    Returns:
        Corresponding CommunicationBackend enum.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_communication_backend("nccl")
        <CommunicationBackend.NCCL: 'nccl'>
        >>> get_communication_backend("gloo")
        <CommunicationBackend.GLOO: 'gloo'>

        >>> get_communication_backend("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: communication_backend must be one of ...
    """
    if name not in VALID_COMMUNICATION_BACKENDS:
        msg = (
            f"communication_backend must be one of {VALID_COMMUNICATION_BACKENDS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return CommunicationBackend(name)


def calculate_world_size(config: ParallelConfig) -> int:
    """Calculate total world size from parallel configuration.

    World size = dp_size * tp_size * pp_size

    Args:
        config: Parallel configuration.

    Returns:
        Total number of parallel processes.

    Examples:
        >>> tp = create_tensor_parallel_config(tp_size=4)
        >>> pp = create_pipeline_parallel_config(pp_size=2)
        >>> config = create_parallel_config(dp_size=2, tp_config=tp, pp_config=pp)
        >>> calculate_world_size(config)
        16

        >>> config = create_parallel_config()
        >>> calculate_world_size(config)
        1
    """
    return config.dp_size * config.tp_config.tp_size * config.pp_config.pp_size


def estimate_communication_overhead(
    config: ParallelConfig,
    model_params_billions: float = 7.0,
) -> float:
    """Estimate communication overhead ratio.

    Calculates approximate fraction of time spent in communication
    based on parallelism configuration.

    Args:
        config: Parallel configuration.
        model_params_billions: Model size in billions of parameters.

    Returns:
        Estimated communication overhead as ratio (0.0 to 1.0).

    Raises:
        ValueError: If model_params_billions is not positive.

    Examples:
        >>> config = create_parallel_config()
        >>> overhead = estimate_communication_overhead(config)
        >>> 0.0 <= overhead <= 1.0
        True

        >>> tp = create_tensor_parallel_config(tp_size=8)
        >>> config = create_parallel_config(tp_config=tp)
        >>> overhead = estimate_communication_overhead(config)
        >>> overhead > 0.0
        True

        >>> estimate_communication_overhead(config, model_params_billions=0)
        Traceback (most recent call last):
            ...
        ValueError: model_params_billions must be positive, got 0
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    overhead = 0.0

    # Data parallel: all-reduce gradients
    if config.dp_size > 1:
        # Communication scales with number of ranks
        overhead += 0.05 * math.log2(config.dp_size)

    # Tensor parallel: all-reduce per layer
    if config.tp_config.tp_size > 1:
        # TP has high communication overhead
        overhead += 0.10 * math.log2(config.tp_config.tp_size)
        # Sequence parallel reduces activation communication
        if config.tp_config.sequence_parallel:
            overhead *= 0.7

    # Pipeline parallel: point-to-point communication
    if config.pp_config.pp_size > 1:
        # PP has lower comm overhead but pipeline bubbles
        overhead += 0.03 * config.pp_config.pp_size
        # Interleaving reduces bubble overhead
        if config.pp_config.interleave:
            overhead *= 0.8

    # Larger models amortize communication better
    overhead *= min(1.0, 7.0 / model_params_billions)

    return min(overhead, 0.5)  # Cap at 50% overhead


def calculate_memory_per_device(
    config: ParallelConfig,
    model_params_billions: float = 7.0,
    batch_size: int = 8,
    sequence_length: int = 2048,
) -> float:
    """Calculate estimated memory per device in GB.

    Estimates GPU memory usage based on parallelism and model configuration.

    Args:
        config: Parallel configuration.
        model_params_billions: Model size in billions of parameters.
        batch_size: Per-device batch size.
        sequence_length: Sequence length.

    Returns:
        Estimated memory per device in GB.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_parallel_config()
        >>> memory = calculate_memory_per_device(config)
        >>> memory > 0
        True

        >>> tp = create_tensor_parallel_config(tp_size=8)
        >>> config = create_parallel_config(tp_config=tp)
        >>> memory_tp = calculate_memory_per_device(config)
        >>> memory_single = calculate_memory_per_device(create_parallel_config())
        >>> memory_tp < memory_single
        True

        >>> calculate_memory_per_device(config, model_params_billions=0)
        Traceback (most recent call last):
            ...
        ValueError: model_params_billions must be positive, got 0
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    # Base memory: 2 bytes per param (FP16) + 8 bytes for optimizer (FP32 Adam)
    params_per_device = model_params_billions / (
        config.tp_config.tp_size * config.pp_config.pp_size
    )
    param_memory_gb = params_per_device * 2  # FP16

    # Optimizer states: 8 bytes per param (momentum + variance in FP32)
    optimizer_memory_gb = params_per_device * 8

    # Gradients: 2 bytes per param
    gradient_memory_gb = params_per_device * 2

    # Apply FSDP sharding
    if config.fsdp_config.sharding_strategy == ParallelShardingStrategy.FULL_SHARD:
        # Full shard divides all states
        shard_factor = config.dp_size
        param_memory_gb /= shard_factor
        optimizer_memory_gb /= shard_factor
        gradient_memory_gb /= shard_factor
    elif config.fsdp_config.sharding_strategy == ParallelShardingStrategy.SHARD_GRAD_OP:
        # Only shard optimizer and gradients
        shard_factor = config.dp_size
        optimizer_memory_gb /= shard_factor
        gradient_memory_gb /= shard_factor

    # CPU offload reduces GPU memory
    if config.fsdp_config.cpu_offload:
        optimizer_memory_gb *= 0.1

    # Activation memory (rough estimate)
    # Activations scale with batch * seq * hidden
    hidden_dim_estimate = 4096 * (model_params_billions / 7.0) ** 0.5
    activation_memory_gb = (
        batch_size * sequence_length * hidden_dim_estimate * 2  # FP16
    ) / 1e9

    # Pipeline and tensor parallelism reduce activation memory
    activation_memory_gb /= config.tp_config.tp_size
    if config.pp_config.activation_checkpointing:
        activation_memory_gb *= 0.3  # Checkpointing reduces by ~70%

    total_memory_gb = (
        param_memory_gb
        + optimizer_memory_gb
        + gradient_memory_gb
        + activation_memory_gb
    )

    return round(total_memory_gb, 2)


def optimize_parallelism_strategy(
    model_params_billions: float,
    num_gpus: int,
    gpu_memory_gb: float = 80.0,
    target_batch_size: int = 1024,
) -> ParallelConfig:
    """Optimize parallelism strategy for given constraints.

    Finds an efficient parallelism configuration based on model size,
    available GPUs, and memory constraints.

    Args:
        model_params_billions: Model size in billions of parameters.
        num_gpus: Total number of available GPUs.
        gpu_memory_gb: Memory per GPU in GB.
        target_batch_size: Target global batch size.

    Returns:
        Optimized ParallelConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = optimize_parallelism_strategy(7.0, 8)
        >>> calculate_world_size(config) <= 8
        True

        >>> config = optimize_parallelism_strategy(70.0, 64)
        >>> config.tp_config.tp_size > 1
        True

        >>> optimize_parallelism_strategy(0, 8)
        Traceback (most recent call last):
            ...
        ValueError: model_params_billions must be positive, got 0
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)
    if num_gpus <= 0:
        msg = f"num_gpus must be positive, got {num_gpus}"
        raise ValueError(msg)
    if gpu_memory_gb <= 0:
        msg = f"gpu_memory_gb must be positive, got {gpu_memory_gb}"
        raise ValueError(msg)
    if target_batch_size <= 0:
        msg = f"target_batch_size must be positive, got {target_batch_size}"
        raise ValueError(msg)

    # Estimate memory needed for model training
    # 2 bytes param + 8 bytes optimizer + 2 bytes gradient = 12 bytes per param
    base_memory_gb = model_params_billions * 12

    # Determine tensor parallelism (for very large models)
    tp_size = 1
    while base_memory_gb / tp_size > gpu_memory_gb * 0.7 and tp_size < num_gpus:
        tp_size *= 2

    # Cap tp_size at 8 (diminishing returns beyond)
    tp_size = min(tp_size, 8, num_gpus)

    # Ensure tp_size is a power of 2
    tp_size = 2 ** int(math.log2(tp_size)) if tp_size > 1 else 1

    # Determine pipeline parallelism (for models that still don't fit)
    remaining_gpus = num_gpus // tp_size
    pp_size = 1

    memory_after_tp = base_memory_gb / tp_size
    while memory_after_tp / pp_size > gpu_memory_gb * 0.6 and pp_size < remaining_gpus:
        pp_size *= 2

    pp_size = min(pp_size, 8, remaining_gpus)

    # Data parallelism uses remaining GPUs
    dp_size = max(1, num_gpus // (tp_size * pp_size))

    # Determine microbatches for pipeline
    num_microbatches = max(pp_size, 4) if pp_size > 1 else 1

    # Configure FSDP based on data parallelism
    if dp_size > 1:
        sharding_strategy = ParallelShardingStrategy.FULL_SHARD
    else:
        sharding_strategy = ParallelShardingStrategy.NO_SHARD

    # Enable sequence parallel if using tensor parallel
    sequence_parallel = tp_size > 1

    tp_config = create_tensor_parallel_config(
        tp_size=tp_size,
        partition_dim=1,
        sequence_parallel=sequence_parallel,
        async_comm=True,
    )

    pp_config = create_pipeline_parallel_config(
        pp_size=pp_size,
        num_microbatches=num_microbatches,
        interleave=pp_size > 2,
        activation_checkpointing=True,
    )

    fsdp_config = create_fsdp_config(
        sharding_strategy=sharding_strategy,
        cpu_offload=False,
        backward_prefetch=True,
        mixed_precision=True,
    )

    return create_parallel_config(
        dp_size=dp_size,
        tp_config=tp_config,
        pp_config=pp_config,
        fsdp_config=fsdp_config,
        backend=CommunicationBackend.NCCL,
    )


def format_parallel_stats(stats: ParallelStats) -> str:
    """Format parallel stats as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = ParallelStats(16, 40.0, 0.15, 0.85)
        >>> formatted = format_parallel_stats(stats)
        >>> "World Size: 16" in formatted
        True
        >>> "Memory: 40.00 GB" in formatted
        True
        >>> "Efficiency: 85.0%" in formatted
        True
    """
    return (
        f"Parallel Stats:\n"
        f"  World Size: {stats.world_size}\n"
        f"  Memory: {stats.memory_per_device_gb:.2f} GB per device\n"
        f"  Communication Overhead: {stats.communication_overhead * 100:.1f}%\n"
        f"  Efficiency: {stats.efficiency * 100:.1f}%"
    )


def get_recommended_parallel_config(
    model_params_billions: float,
    num_gpus: int,
    gpu_memory_gb: float = 80.0,
) -> ParallelConfig:
    """Get recommended parallel configuration for model training.

    Provides sensible defaults based on model size and available resources.

    Args:
        model_params_billions: Model size in billions of parameters.
        num_gpus: Total number of available GPUs.
        gpu_memory_gb: Memory per GPU in GB.

    Returns:
        Recommended ParallelConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_parallel_config(7.0, 8)
        >>> config.dp_size >= 1
        True

        >>> config = get_recommended_parallel_config(70.0, 64, 80.0)
        >>> config.tp_config.tp_size >= 1
        True

        >>> get_recommended_parallel_config(0, 8)
        Traceback (most recent call last):
            ...
        ValueError: model_params_billions must be positive, got 0
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)
    if num_gpus <= 0:
        msg = f"num_gpus must be positive, got {num_gpus}"
        raise ValueError(msg)
    if gpu_memory_gb <= 0:
        msg = f"gpu_memory_gb must be positive, got {gpu_memory_gb}"
        raise ValueError(msg)

    return optimize_parallelism_strategy(
        model_params_billions=model_params_billions,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        target_batch_size=1024,
    )


def create_parallel_stats(
    config: ParallelConfig,
    model_params_billions: float = 7.0,
) -> ParallelStats:
    """Create parallel statistics from configuration.

    Args:
        config: Parallel configuration.
        model_params_billions: Model size in billions of parameters.

    Returns:
        ParallelStats with computed metrics.

    Raises:
        ValueError: If model_params_billions is not positive.

    Examples:
        >>> config = create_parallel_config()
        >>> stats = create_parallel_stats(config)
        >>> stats.world_size >= 1
        True

        >>> tp = create_tensor_parallel_config(tp_size=4)
        >>> pp = create_pipeline_parallel_config(pp_size=2)
        >>> config = create_parallel_config(dp_size=2, tp_config=tp, pp_config=pp)
        >>> stats = create_parallel_stats(config)
        >>> stats.world_size
        16

        >>> create_parallel_stats(config, model_params_billions=0)
        Traceback (most recent call last):
            ...
        ValueError: model_params_billions must be positive, got 0
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    world_size = calculate_world_size(config)
    memory = calculate_memory_per_device(config, model_params_billions)
    overhead = estimate_communication_overhead(config, model_params_billions)
    efficiency = 1.0 - overhead

    return ParallelStats(
        world_size=world_size,
        memory_per_device_gb=memory,
        communication_overhead=overhead,
        efficiency=efficiency,
    )
