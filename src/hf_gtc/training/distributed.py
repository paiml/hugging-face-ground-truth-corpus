"""Distributed training utilities.

This module provides functions for configuring distributed training
including FSDP, DeepSpeed, and multi-GPU training patterns.

Examples:
    >>> from hf_gtc.training.distributed import create_fsdp_config
    >>> config = create_fsdp_config(sharding_strategy="full_shard")
    >>> config.sharding_strategy
    <ShardingStrategy.FULL_SHARD: 'full_shard'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ShardingStrategy(Enum):
    """FSDP sharding strategies.

    Attributes:
        FULL_SHARD: Full sharding across all ranks.
        SHARD_GRAD_OP: Shard gradients and optimizer states.
        NO_SHARD: No sharding (DDP-like).
        HYBRID_SHARD: Hybrid sharding within nodes.

    Examples:
        >>> ShardingStrategy.FULL_SHARD.value
        'full_shard'
        >>> ShardingStrategy.SHARD_GRAD_OP.value
        'shard_grad_op'
    """

    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"


VALID_SHARDING_STRATEGIES = frozenset(s.value for s in ShardingStrategy)


class DeepSpeedStage(Enum):
    """DeepSpeed ZeRO optimization stages.

    Attributes:
        STAGE_0: No sharding (baseline).
        STAGE_1: Optimizer state partitioning.
        STAGE_2: Gradient partitioning.
        STAGE_3: Parameter partitioning.

    Examples:
        >>> DeepSpeedStage.STAGE_2.value
        'stage_2'
        >>> DeepSpeedStage.STAGE_3.value
        'stage_3'
    """

    STAGE_0 = "stage_0"
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"


VALID_DEEPSPEED_STAGES = frozenset(s.value for s in DeepSpeedStage)


class DistributedBackend(Enum):
    """Distributed training backends.

    Attributes:
        NCCL: NVIDIA Collective Communications Library.
        GLOO: Facebook's Gloo library.
        MPI: Message Passing Interface.

    Examples:
        >>> DistributedBackend.NCCL.value
        'nccl'
        >>> DistributedBackend.GLOO.value
        'gloo'
    """

    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


VALID_BACKENDS = frozenset(b.value for b in DistributedBackend)


class ActivationCheckpointing(Enum):
    """Activation checkpointing strategies.

    Attributes:
        NONE: No checkpointing.
        FULL: Checkpoint all layers.
        SELECTIVE: Checkpoint selected layers.

    Examples:
        >>> ActivationCheckpointing.FULL.value
        'full'
        >>> ActivationCheckpointing.SELECTIVE.value
        'selective'
    """

    NONE = "none"
    FULL = "full"
    SELECTIVE = "selective"


VALID_CHECKPOINTING = frozenset(c.value for c in ActivationCheckpointing)


@dataclass(frozen=True, slots=True)
class FSDPConfig:
    """Configuration for FSDP (Fully Sharded Data Parallel).

    Attributes:
        sharding_strategy: How to shard parameters.
        cpu_offload: Whether to offload to CPU.
        mixed_precision: Enable mixed precision.
        backward_prefetch: Prefetch during backward.
        forward_prefetch: Prefetch during forward.
        activation_checkpointing: Checkpointing strategy.

    Examples:
        >>> config = FSDPConfig(
        ...     sharding_strategy=ShardingStrategy.FULL_SHARD,
        ...     cpu_offload=False,
        ...     mixed_precision=True,
        ...     backward_prefetch=True,
        ...     forward_prefetch=False,
        ...     activation_checkpointing=ActivationCheckpointing.FULL,
        ... )
        >>> config.sharding_strategy
        <ShardingStrategy.FULL_SHARD: 'full_shard'>
    """

    sharding_strategy: ShardingStrategy
    cpu_offload: bool
    mixed_precision: bool
    backward_prefetch: bool
    forward_prefetch: bool
    activation_checkpointing: ActivationCheckpointing


@dataclass(frozen=True, slots=True)
class DeepSpeedConfig:
    """Configuration for DeepSpeed.

    Attributes:
        stage: ZeRO optimization stage.
        offload_optimizer: Offload optimizer to CPU.
        offload_param: Offload parameters to CPU.
        overlap_comm: Overlap communication with computation.
        contiguous_gradients: Use contiguous gradient buffer.
        reduce_bucket_size: Bucket size for all-reduce.

    Examples:
        >>> config = DeepSpeedConfig(
        ...     stage=DeepSpeedStage.STAGE_2,
        ...     offload_optimizer=False,
        ...     offload_param=False,
        ...     overlap_comm=True,
        ...     contiguous_gradients=True,
        ...     reduce_bucket_size=500000000,
        ... )
        >>> config.stage
        <DeepSpeedStage.STAGE_2: 'stage_2'>
    """

    stage: DeepSpeedStage
    offload_optimizer: bool
    offload_param: bool
    overlap_comm: bool
    contiguous_gradients: bool
    reduce_bucket_size: int


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """General distributed training configuration.

    Attributes:
        backend: Communication backend.
        world_size: Total number of processes.
        local_rank: Rank within local node.
        num_nodes: Number of nodes.
        gpus_per_node: GPUs per node.

    Examples:
        >>> config = DistributedConfig(
        ...     backend=DistributedBackend.NCCL,
        ...     world_size=8,
        ...     local_rank=0,
        ...     num_nodes=2,
        ...     gpus_per_node=4,
        ... )
        >>> config.world_size
        8
    """

    backend: DistributedBackend
    world_size: int
    local_rank: int
    num_nodes: int
    gpus_per_node: int


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Memory estimate for distributed training.

    Attributes:
        model_memory_gb: Model memory in GB.
        optimizer_memory_gb: Optimizer memory in GB.
        gradient_memory_gb: Gradient memory in GB.
        activation_memory_gb: Activation memory in GB.
        total_memory_gb: Total memory per GPU.

    Examples:
        >>> est = MemoryEstimate(
        ...     model_memory_gb=14.0,
        ...     optimizer_memory_gb=28.0,
        ...     gradient_memory_gb=14.0,
        ...     activation_memory_gb=8.0,
        ...     total_memory_gb=16.0,
        ... )
        >>> est.total_memory_gb
        16.0
    """

    model_memory_gb: float
    optimizer_memory_gb: float
    gradient_memory_gb: float
    activation_memory_gb: float
    total_memory_gb: float


@dataclass(frozen=True, slots=True)
class ScalingMetrics:
    """Scaling efficiency metrics.

    Attributes:
        throughput_samples_per_sec: Training throughput.
        scaling_efficiency: Efficiency vs linear scaling.
        communication_overhead: Communication overhead ratio.
        gpu_utilization: Average GPU utilization.

    Examples:
        >>> metrics = ScalingMetrics(
        ...     throughput_samples_per_sec=1000.0,
        ...     scaling_efficiency=0.85,
        ...     communication_overhead=0.15,
        ...     gpu_utilization=0.92,
        ... )
        >>> metrics.scaling_efficiency
        0.85
    """

    throughput_samples_per_sec: float
    scaling_efficiency: float
    communication_overhead: float
    gpu_utilization: float


def validate_fsdp_config(config: FSDPConfig) -> None:
    """Validate FSDP configuration.

    Args:
        config: FSDP configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = FSDPConfig(
        ...     ShardingStrategy.FULL_SHARD, False, True, True, False,
        ...     ActivationCheckpointing.FULL
        ... )
        >>> validate_fsdp_config(config)  # No error

        >>> bad = FSDPConfig(
        ...     ShardingStrategy.NO_SHARD, True, True, True, False,
        ...     ActivationCheckpointing.FULL
        ... )
        >>> validate_fsdp_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: cpu_offload requires sharding strategy other than NO_SHARD
    """
    if config.cpu_offload and config.sharding_strategy == ShardingStrategy.NO_SHARD:
        msg = "cpu_offload requires sharding strategy other than NO_SHARD"
        raise ValueError(msg)


def validate_deepspeed_config(config: DeepSpeedConfig) -> None:
    """Validate DeepSpeed configuration.

    Args:
        config: DeepSpeed configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = DeepSpeedConfig(
        ...     DeepSpeedStage.STAGE_2, False, False, True, True, 500000000
        ... )
        >>> validate_deepspeed_config(config)  # No error

        >>> bad = DeepSpeedConfig(
        ...     DeepSpeedStage.STAGE_1, False, True, True, True, 500000000
        ... )
        >>> validate_deepspeed_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: offload_param requires ZeRO stage 3
    """
    if config.offload_param and config.stage != DeepSpeedStage.STAGE_3:
        msg = "offload_param requires ZeRO stage 3"
        raise ValueError(msg)

    if config.reduce_bucket_size <= 0:
        msg = f"reduce_bucket_size must be positive, got {config.reduce_bucket_size}"
        raise ValueError(msg)


def validate_distributed_config(config: DistributedConfig) -> None:
    """Validate distributed configuration.

    Args:
        config: Distributed configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = DistributedConfig(
        ...     DistributedBackend.NCCL, 8, 0, 2, 4
        ... )
        >>> validate_distributed_config(config)  # No error

        >>> bad = DistributedConfig(
        ...     DistributedBackend.NCCL, 8, 0, 2, 3
        ... )
        >>> validate_distributed_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: world_size must equal num_nodes * gpus_per_node
    """
    if config.world_size <= 0:
        msg = f"world_size must be positive, got {config.world_size}"
        raise ValueError(msg)

    if config.num_nodes <= 0:
        msg = f"num_nodes must be positive, got {config.num_nodes}"
        raise ValueError(msg)

    if config.gpus_per_node <= 0:
        msg = f"gpus_per_node must be positive, got {config.gpus_per_node}"
        raise ValueError(msg)

    expected_world_size = config.num_nodes * config.gpus_per_node
    if config.world_size != expected_world_size:
        msg = (
            f"world_size must equal num_nodes * gpus_per_node "
            f"({config.world_size} != {expected_world_size})"
        )
        raise ValueError(msg)

    if config.local_rank < 0 or config.local_rank >= config.gpus_per_node:
        msg = f"local_rank must be in [0, gpus_per_node), got {config.local_rank}"
        raise ValueError(msg)


def create_fsdp_config(
    sharding_strategy: str = "full_shard",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    backward_prefetch: bool = True,
    forward_prefetch: bool = False,
    activation_checkpointing: str = "none",
) -> FSDPConfig:
    """Create an FSDP configuration.

    Args:
        sharding_strategy: Sharding strategy. Defaults to "full_shard".
        cpu_offload: Offload to CPU. Defaults to False.
        mixed_precision: Use mixed precision. Defaults to True.
        backward_prefetch: Prefetch backward. Defaults to True.
        forward_prefetch: Prefetch forward. Defaults to False.
        activation_checkpointing: Checkpointing. Defaults to "none".

    Returns:
        FSDPConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_fsdp_config()
        >>> config.sharding_strategy
        <ShardingStrategy.FULL_SHARD: 'full_shard'>

        >>> config = create_fsdp_config(activation_checkpointing="full")
        >>> config.activation_checkpointing
        <ActivationCheckpointing.FULL: 'full'>

        >>> create_fsdp_config(sharding_strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sharding_strategy must be one of
    """
    if sharding_strategy not in VALID_SHARDING_STRATEGIES:
        msg = (
            f"sharding_strategy must be one of {VALID_SHARDING_STRATEGIES}, "
            f"got '{sharding_strategy}'"
        )
        raise ValueError(msg)

    if activation_checkpointing not in VALID_CHECKPOINTING:
        msg = (
            f"activation_checkpointing must be one of {VALID_CHECKPOINTING}, "
            f"got '{activation_checkpointing}'"
        )
        raise ValueError(msg)

    config = FSDPConfig(
        sharding_strategy=ShardingStrategy(sharding_strategy),
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision,
        backward_prefetch=backward_prefetch,
        forward_prefetch=forward_prefetch,
        activation_checkpointing=ActivationCheckpointing(activation_checkpointing),
    )
    validate_fsdp_config(config)
    return config


def create_deepspeed_config(
    stage: str = "stage_2",
    offload_optimizer: bool = False,
    offload_param: bool = False,
    overlap_comm: bool = True,
    contiguous_gradients: bool = True,
    reduce_bucket_size: int = 500_000_000,
) -> DeepSpeedConfig:
    """Create a DeepSpeed configuration.

    Args:
        stage: ZeRO stage. Defaults to "stage_2".
        offload_optimizer: Offload optimizer. Defaults to False.
        offload_param: Offload params (stage 3 only). Defaults to False.
        overlap_comm: Overlap communication. Defaults to True.
        contiguous_gradients: Contiguous gradients. Defaults to True.
        reduce_bucket_size: Bucket size. Defaults to 500_000_000.

    Returns:
        DeepSpeedConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_deepspeed_config()
        >>> config.stage
        <DeepSpeedStage.STAGE_2: 'stage_2'>

        >>> config = create_deepspeed_config(stage="stage_3", offload_param=True)
        >>> config.offload_param
        True

        >>> create_deepspeed_config(stage="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stage must be one of
    """
    if stage not in VALID_DEEPSPEED_STAGES:
        msg = f"stage must be one of {VALID_DEEPSPEED_STAGES}, got '{stage}'"
        raise ValueError(msg)

    config = DeepSpeedConfig(
        stage=DeepSpeedStage(stage),
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
        overlap_comm=overlap_comm,
        contiguous_gradients=contiguous_gradients,
        reduce_bucket_size=reduce_bucket_size,
    )
    validate_deepspeed_config(config)
    return config


def create_distributed_config(
    backend: str = "nccl",
    world_size: int = 1,
    local_rank: int = 0,
    num_nodes: int = 1,
    gpus_per_node: int = 1,
) -> DistributedConfig:
    """Create a distributed training configuration.

    Args:
        backend: Communication backend. Defaults to "nccl".
        world_size: Total processes. Defaults to 1.
        local_rank: Local rank. Defaults to 0.
        num_nodes: Number of nodes. Defaults to 1.
        gpus_per_node: GPUs per node. Defaults to 1.

    Returns:
        DistributedConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_distributed_config()
        >>> config.world_size
        1

        >>> config = create_distributed_config(
        ...     world_size=8, num_nodes=2, gpus_per_node=4
        ... )
        >>> config.num_nodes
        2

        >>> create_distributed_config(backend="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: backend must be one of
    """
    if backend not in VALID_BACKENDS:
        msg = f"backend must be one of {VALID_BACKENDS}, got '{backend}'"
        raise ValueError(msg)

    config = DistributedConfig(
        backend=DistributedBackend(backend),
        world_size=world_size,
        local_rank=local_rank,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )
    validate_distributed_config(config)
    return config


def estimate_fsdp_memory(
    model_params_billions: float,
    sharding_strategy: str = "full_shard",
    world_size: int = 8,
    mixed_precision: bool = True,
    activation_checkpointing: bool = False,
) -> MemoryEstimate:
    """Estimate memory per GPU for FSDP training.

    Args:
        model_params_billions: Model size in billions of parameters.
        sharding_strategy: FSDP sharding strategy.
        world_size: Number of GPUs.
        mixed_precision: Whether using mixed precision.
        activation_checkpointing: Whether using activation checkpointing.

    Returns:
        MemoryEstimate with per-GPU memory breakdown.

    Examples:
        >>> est = estimate_fsdp_memory(7.0, world_size=8)
        >>> est.total_memory_gb > 0
        True

        >>> est_full = estimate_fsdp_memory(7.0, "full_shard", 8)
        >>> est_no = estimate_fsdp_memory(7.0, "no_shard", 8)
        >>> est_full.total_memory_gb < est_no.total_memory_gb
        True
    """
    # Base memory calculations (in GB)
    bytes_per_param = 2.0 if mixed_precision else 4.0
    model_memory = model_params_billions * bytes_per_param

    # Optimizer states (Adam: 2x model for momentum + variance)
    optimizer_memory = model_params_billions * 4.0 * 2  # Always FP32

    # Gradients
    gradient_memory = model_params_billions * bytes_per_param

    # Activation memory (rough estimate based on model size)
    activation_memory = model_params_billions * 2.0
    if activation_checkpointing:
        activation_memory *= 0.3  # Reduce by ~70%

    # Apply sharding based on strategy
    if sharding_strategy == "full_shard":
        model_memory /= world_size
        optimizer_memory /= world_size
        gradient_memory /= world_size
    elif sharding_strategy == "shard_grad_op":
        optimizer_memory /= world_size
        gradient_memory /= world_size
    # no_shard and hybrid_shard: minimal or partial reduction

    total = model_memory + optimizer_memory + gradient_memory + activation_memory

    return MemoryEstimate(
        model_memory_gb=round(model_memory, 2),
        optimizer_memory_gb=round(optimizer_memory, 2),
        gradient_memory_gb=round(gradient_memory, 2),
        activation_memory_gb=round(activation_memory, 2),
        total_memory_gb=round(total, 2),
    )


def estimate_deepspeed_memory(
    model_params_billions: float,
    stage: str = "stage_2",
    world_size: int = 8,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> MemoryEstimate:
    """Estimate memory per GPU for DeepSpeed training.

    Args:
        model_params_billions: Model size in billions of parameters.
        stage: ZeRO optimization stage.
        world_size: Number of GPUs.
        offload_optimizer: Whether offloading optimizer to CPU.
        offload_param: Whether offloading parameters to CPU.

    Returns:
        MemoryEstimate with per-GPU memory breakdown.

    Examples:
        >>> est = estimate_deepspeed_memory(7.0, "stage_2", world_size=8)
        >>> est.total_memory_gb > 0
        True

        >>> est_s2 = estimate_deepspeed_memory(7.0, "stage_2", 8)
        >>> est_s3 = estimate_deepspeed_memory(7.0, "stage_3", 8)
        >>> est_s3.total_memory_gb < est_s2.total_memory_gb
        True
    """
    # Base calculations
    model_memory = model_params_billions * 2.0  # FP16
    optimizer_memory = model_params_billions * 4.0 * 2  # FP32 Adam states
    gradient_memory = model_params_billions * 2.0  # FP16
    activation_memory = model_params_billions * 2.0

    # Apply ZeRO stage reductions
    if stage == "stage_1":
        optimizer_memory /= world_size
    elif stage == "stage_2":
        optimizer_memory /= world_size
        gradient_memory /= world_size
    elif stage == "stage_3":
        model_memory /= world_size
        optimizer_memory /= world_size
        gradient_memory /= world_size

    # CPU offload reduces GPU memory
    if offload_optimizer:
        optimizer_memory *= 0.1  # Most on CPU
    if offload_param:
        model_memory *= 0.1  # Most on CPU

    total = model_memory + optimizer_memory + gradient_memory + activation_memory

    return MemoryEstimate(
        model_memory_gb=round(model_memory, 2),
        optimizer_memory_gb=round(optimizer_memory, 2),
        gradient_memory_gb=round(gradient_memory, 2),
        activation_memory_gb=round(activation_memory, 2),
        total_memory_gb=round(total, 2),
    )


def calculate_scaling_efficiency(
    single_gpu_throughput: float,
    multi_gpu_throughput: float,
    num_gpus: int,
) -> float:
    """Calculate scaling efficiency.

    Args:
        single_gpu_throughput: Samples/sec with 1 GPU.
        multi_gpu_throughput: Samples/sec with multiple GPUs.
        num_gpus: Number of GPUs used.

    Returns:
        Scaling efficiency (1.0 = perfect linear scaling).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_scaling_efficiency(100.0, 750.0, 8)
        0.9375

        >>> calculate_scaling_efficiency(100.0, 800.0, 8)
        1.0

        >>> calculate_scaling_efficiency(0, 800.0, 8)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: single_gpu_throughput must be positive
    """
    if single_gpu_throughput <= 0:
        msg = f"single_gpu_throughput must be positive, got {single_gpu_throughput}"
        raise ValueError(msg)

    if multi_gpu_throughput <= 0:
        msg = f"multi_gpu_throughput must be positive, got {multi_gpu_throughput}"
        raise ValueError(msg)

    if num_gpus <= 0:
        msg = f"num_gpus must be positive, got {num_gpus}"
        raise ValueError(msg)

    ideal_throughput = single_gpu_throughput * num_gpus
    return multi_gpu_throughput / ideal_throughput


def get_recommended_strategy(
    model_params_billions: float,
    gpu_memory_gb: float = 80.0,
    num_gpus: int = 8,
) -> str:
    """Get recommended distributed training strategy.

    Args:
        model_params_billions: Model size in billions of parameters.
        gpu_memory_gb: Available GPU memory per device.
        num_gpus: Number of available GPUs.

    Returns:
        Recommended strategy name.

    Examples:
        >>> get_recommended_strategy(7.0, 80.0, 8)
        'fsdp_full_shard'

        >>> get_recommended_strategy(70.0, 80.0, 8)
        'deepspeed_stage_3'

        >>> get_recommended_strategy(1.0, 80.0, 4)
        'ddp'
    """
    # Estimate memory needs
    model_memory = model_params_billions * 2  # FP16
    optimizer_memory = model_params_billions * 4 * 2  # FP32 Adam
    total_memory_per_gpu = model_memory + optimizer_memory + model_params_billions * 4

    # If it fits on one GPU with room to spare, use DDP
    if total_memory_per_gpu < gpu_memory_gb * 0.5:
        return "ddp"

    # If sharding across GPUs makes it fit, use FSDP
    sharded_memory = total_memory_per_gpu / num_gpus
    if sharded_memory < gpu_memory_gb * 0.7:
        return "fsdp_full_shard"

    # For very large models, use DeepSpeed Stage 3
    return "deepspeed_stage_3"


def list_sharding_strategies() -> list[str]:
    """List supported FSDP sharding strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_sharding_strategies()
        >>> "full_shard" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SHARDING_STRATEGIES)


def list_deepspeed_stages() -> list[str]:
    """List supported DeepSpeed ZeRO stages.

    Returns:
        Sorted list of stage names.

    Examples:
        >>> stages = list_deepspeed_stages()
        >>> "stage_2" in stages
        True
        >>> "stage_3" in stages
        True
    """
    return sorted(VALID_DEEPSPEED_STAGES)


def list_backends() -> list[str]:
    """List supported distributed backends.

    Returns:
        Sorted list of backend names.

    Examples:
        >>> backends = list_backends()
        >>> "nccl" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_BACKENDS)


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Get sharding strategy from name.

    Args:
        name: Strategy name.

    Returns:
        ShardingStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_sharding_strategy("full_shard")
        <ShardingStrategy.FULL_SHARD: 'full_shard'>

        >>> get_sharding_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if name not in VALID_SHARDING_STRATEGIES:
        msg = f"strategy must be one of {VALID_SHARDING_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return ShardingStrategy(name)


def get_deepspeed_stage(name: str) -> DeepSpeedStage:
    """Get DeepSpeed stage from name.

    Args:
        name: Stage name.

    Returns:
        DeepSpeedStage enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_deepspeed_stage("stage_2")
        <DeepSpeedStage.STAGE_2: 'stage_2'>

        >>> get_deepspeed_stage("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stage must be one of
    """
    if name not in VALID_DEEPSPEED_STAGES:
        msg = f"stage must be one of {VALID_DEEPSPEED_STAGES}, got '{name}'"
        raise ValueError(msg)
    return DeepSpeedStage(name)


def format_memory_estimate(estimate: MemoryEstimate) -> str:
    """Format memory estimate for display.

    Args:
        estimate: Memory estimate to format.

    Returns:
        Formatted string.

    Examples:
        >>> est = MemoryEstimate(7.0, 14.0, 7.0, 4.0, 32.0)
        >>> formatted = format_memory_estimate(est)
        >>> "Model: 7.00 GB" in formatted
        True
        >>> "Total: 32.00 GB" in formatted
        True
    """
    lines = [
        f"Model: {estimate.model_memory_gb:.2f} GB",
        f"Optimizer: {estimate.optimizer_memory_gb:.2f} GB",
        f"Gradients: {estimate.gradient_memory_gb:.2f} GB",
        f"Activations: {estimate.activation_memory_gb:.2f} GB",
        f"Total: {estimate.total_memory_gb:.2f} GB",
    ]
    return "\n".join(lines)
