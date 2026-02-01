"""Gradient and activation checkpointing utilities.

This module provides functions for configuring gradient checkpointing,
activation checkpointing, and memory optimization strategies for training
large models.

Examples:
    >>> from hf_gtc.training.checkpointing import create_checkpoint_config
    >>> config = create_checkpoint_config(strategy="selective")
    >>> config.strategy
    <CheckpointStrategy.SELECTIVE: 'selective'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class CheckpointStrategy(Enum):
    """Checkpointing strategies for memory optimization.

    Attributes:
        FULL: Checkpoint all layers (maximum memory savings).
        SELECTIVE: Checkpoint only selected layers.
        OFFLOAD: Offload activations to CPU/disk.
        NONE: No checkpointing (baseline).

    Examples:
        >>> CheckpointStrategy.FULL.value
        'full'
        >>> CheckpointStrategy.SELECTIVE.value
        'selective'
        >>> CheckpointStrategy.OFFLOAD.value
        'offload'
        >>> CheckpointStrategy.NONE.value
        'none'
    """

    FULL = "full"
    SELECTIVE = "selective"
    OFFLOAD = "offload"
    NONE = "none"


VALID_CHECKPOINT_STRATEGIES = frozenset(s.value for s in CheckpointStrategy)


class CheckpointGranularity(Enum):
    """Granularity levels for selective checkpointing.

    Attributes:
        LAYER: Checkpoint at layer boundaries.
        BLOCK: Checkpoint at transformer block boundaries.
        ATTENTION: Checkpoint attention layers only.
        MLP: Checkpoint MLP/FFN layers only.

    Examples:
        >>> CheckpointGranularity.LAYER.value
        'layer'
        >>> CheckpointGranularity.BLOCK.value
        'block'
        >>> CheckpointGranularity.ATTENTION.value
        'attention'
        >>> CheckpointGranularity.MLP.value
        'mlp'
    """

    LAYER = "layer"
    BLOCK = "block"
    ATTENTION = "attention"
    MLP = "mlp"


VALID_CHECKPOINT_GRANULARITIES = frozenset(g.value for g in CheckpointGranularity)


class OffloadTarget(Enum):
    """Target locations for activation offloading.

    Attributes:
        CPU: Offload to CPU memory.
        DISK: Offload to disk storage.
        NVME: Offload to NVMe storage.

    Examples:
        >>> OffloadTarget.CPU.value
        'cpu'
        >>> OffloadTarget.DISK.value
        'disk'
        >>> OffloadTarget.NVME.value
        'nvme'
    """

    CPU = "cpu"
    DISK = "disk"
    NVME = "nvme"


VALID_OFFLOAD_TARGETS = frozenset(t.value for t in OffloadTarget)


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Configuration for gradient/activation checkpointing.

    Attributes:
        strategy: Checkpointing strategy to use.
        granularity: Granularity level for selective checkpointing.
        checkpoint_ratio: Fraction of layers to checkpoint (0.0-1.0).

    Examples:
        >>> config = CheckpointConfig(
        ...     strategy=CheckpointStrategy.SELECTIVE,
        ...     granularity=CheckpointGranularity.BLOCK,
        ...     checkpoint_ratio=0.5,
        ... )
        >>> config.strategy
        <CheckpointStrategy.SELECTIVE: 'selective'>
        >>> config.checkpoint_ratio
        0.5
    """

    strategy: CheckpointStrategy
    granularity: CheckpointGranularity
    checkpoint_ratio: float


@dataclass(frozen=True, slots=True)
class OffloadConfig:
    """Configuration for activation offloading.

    Attributes:
        target: Where to offload activations.
        pin_memory: Use pinned memory for faster CPU transfers.
        async_transfer: Use asynchronous transfers.

    Examples:
        >>> config = OffloadConfig(
        ...     target=OffloadTarget.CPU,
        ...     pin_memory=True,
        ...     async_transfer=True,
        ... )
        >>> config.target
        <OffloadTarget.CPU: 'cpu'>
        >>> config.pin_memory
        True
    """

    target: OffloadTarget
    pin_memory: bool
    async_transfer: bool


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Combined memory optimization configuration.

    Attributes:
        checkpoint_config: Checkpointing settings.
        offload_config: Offloading settings.
        cpu_offload: Enable CPU offloading for model parameters.

    Examples:
        >>> ckpt = CheckpointConfig(
        ...     CheckpointStrategy.FULL, CheckpointGranularity.LAYER, 1.0
        ... )
        >>> offload = OffloadConfig(OffloadTarget.CPU, True, True)
        >>> config = MemoryConfig(
        ...     checkpoint_config=ckpt,
        ...     offload_config=offload,
        ...     cpu_offload=False,
        ... )
        >>> config.checkpoint_config.strategy
        <CheckpointStrategy.FULL: 'full'>
    """

    checkpoint_config: CheckpointConfig
    offload_config: OffloadConfig
    cpu_offload: bool


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """Statistics for memory usage with checkpointing.

    Attributes:
        baseline_memory_gb: Memory without checkpointing.
        checkpointed_memory_gb: Memory with checkpointing.
        memory_saved_gb: Memory saved by checkpointing.
        savings_percentage: Percentage of memory saved.
        recomputation_overhead_pct: Recomputation overhead percentage.

    Examples:
        >>> stats = MemoryStats(
        ...     baseline_memory_gb=32.0,
        ...     checkpointed_memory_gb=12.0,
        ...     memory_saved_gb=20.0,
        ...     savings_percentage=62.5,
        ...     recomputation_overhead_pct=33.0,
        ... )
        >>> stats.savings_percentage
        62.5
    """

    baseline_memory_gb: float
    checkpointed_memory_gb: float
    memory_saved_gb: float
    savings_percentage: float
    recomputation_overhead_pct: float


def validate_checkpoint_config(config: CheckpointConfig) -> None:
    """Validate checkpoint configuration.

    Args:
        config: Checkpoint configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If checkpoint_ratio is not in [0.0, 1.0].
        ValueError: If checkpoint_ratio is 0 with non-NONE strategy.

    Examples:
        >>> config = CheckpointConfig(
        ...     CheckpointStrategy.SELECTIVE, CheckpointGranularity.BLOCK, 0.5
        ... )
        >>> validate_checkpoint_config(config)  # No error

        >>> validate_checkpoint_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = CheckpointConfig(
        ...     CheckpointStrategy.SELECTIVE, CheckpointGranularity.BLOCK, -0.1
        ... )
        >>> validate_checkpoint_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: checkpoint_ratio must be in [0.0, 1.0]
    """
    validate_not_none(config, "config")

    if not 0.0 <= config.checkpoint_ratio <= 1.0:
        msg = f"checkpoint_ratio must be in [0.0, 1.0], got {config.checkpoint_ratio}"
        raise ValueError(msg)

    if config.checkpoint_ratio == 0.0 and config.strategy != CheckpointStrategy.NONE:
        msg = "checkpoint_ratio cannot be 0.0 unless strategy is 'none'"
        raise ValueError(msg)


def validate_offload_config(config: OffloadConfig) -> None:
    """Validate offload configuration.

    Args:
        config: Offload configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If pin_memory is True for disk/nvme targets.

    Examples:
        >>> config = OffloadConfig(OffloadTarget.CPU, True, True)
        >>> validate_offload_config(config)  # No error

        >>> validate_offload_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = OffloadConfig(OffloadTarget.DISK, True, True)
        >>> validate_offload_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: pin_memory only applies to CPU offloading
    """
    validate_not_none(config, "config")

    if config.pin_memory and config.target != OffloadTarget.CPU:
        msg = "pin_memory only applies to CPU offloading"
        raise ValueError(msg)


def validate_memory_config(config: MemoryConfig) -> None:
    """Validate memory configuration.

    Args:
        config: Memory configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If sub-configs are invalid.

    Examples:
        >>> ckpt = CheckpointConfig(
        ...     CheckpointStrategy.FULL, CheckpointGranularity.LAYER, 1.0
        ... )
        >>> offload = OffloadConfig(OffloadTarget.CPU, True, True)
        >>> config = MemoryConfig(ckpt, offload, False)
        >>> validate_memory_config(config)  # No error

        >>> validate_memory_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    validate_checkpoint_config(config.checkpoint_config)
    validate_offload_config(config.offload_config)


def create_checkpoint_config(
    strategy: str = "selective",
    granularity: str = "block",
    checkpoint_ratio: float = 0.5,
) -> CheckpointConfig:
    """Create a checkpoint configuration.

    Args:
        strategy: Checkpointing strategy. Defaults to "selective".
        granularity: Checkpoint granularity. Defaults to "block".
        checkpoint_ratio: Ratio of layers to checkpoint. Defaults to 0.5.

    Returns:
        Validated CheckpointConfig instance.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If granularity is invalid.
        ValueError: If checkpoint_ratio is not in [0.0, 1.0].

    Examples:
        >>> config = create_checkpoint_config()
        >>> config.strategy
        <CheckpointStrategy.SELECTIVE: 'selective'>
        >>> config.checkpoint_ratio
        0.5

        >>> config = create_checkpoint_config(strategy="full", checkpoint_ratio=1.0)
        >>> config.strategy
        <CheckpointStrategy.FULL: 'full'>

        >>> create_checkpoint_config(strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if strategy not in VALID_CHECKPOINT_STRATEGIES:
        msg = f"strategy must be one of {VALID_CHECKPOINT_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    if granularity not in VALID_CHECKPOINT_GRANULARITIES:
        msg = (
            f"granularity must be one of {VALID_CHECKPOINT_GRANULARITIES}, "
            f"got '{granularity}'"
        )
        raise ValueError(msg)

    config = CheckpointConfig(
        strategy=CheckpointStrategy(strategy),
        granularity=CheckpointGranularity(granularity),
        checkpoint_ratio=checkpoint_ratio,
    )
    validate_checkpoint_config(config)
    return config


def create_offload_config(
    target: str = "cpu",
    pin_memory: bool = True,
    async_transfer: bool = True,
) -> OffloadConfig:
    """Create an offload configuration.

    Args:
        target: Offload target location. Defaults to "cpu".
        pin_memory: Use pinned memory. Defaults to True.
        async_transfer: Use async transfers. Defaults to True.

    Returns:
        Validated OffloadConfig instance.

    Raises:
        ValueError: If target is invalid.

    Examples:
        >>> config = create_offload_config()
        >>> config.target
        <OffloadTarget.CPU: 'cpu'>
        >>> config.pin_memory
        True

        >>> config = create_offload_config(target="disk", pin_memory=False)
        >>> config.target
        <OffloadTarget.DISK: 'disk'>

        >>> create_offload_config(target="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target must be one of
    """
    if target not in VALID_OFFLOAD_TARGETS:
        msg = f"target must be one of {VALID_OFFLOAD_TARGETS}, got '{target}'"
        raise ValueError(msg)

    config = OffloadConfig(
        target=OffloadTarget(target),
        pin_memory=pin_memory,
        async_transfer=async_transfer,
    )
    validate_offload_config(config)
    return config


def create_memory_config(
    checkpoint_config: CheckpointConfig | None = None,
    offload_config: OffloadConfig | None = None,
    cpu_offload: bool = False,
) -> MemoryConfig:
    """Create a memory optimization configuration.

    Args:
        checkpoint_config: Checkpointing settings. Defaults to selective.
        offload_config: Offloading settings. Defaults to CPU offload.
        cpu_offload: Enable CPU offloading for parameters. Defaults to False.

    Returns:
        Validated MemoryConfig instance.

    Examples:
        >>> config = create_memory_config()
        >>> config.checkpoint_config.strategy
        <CheckpointStrategy.SELECTIVE: 'selective'>
        >>> config.offload_config.target
        <OffloadTarget.CPU: 'cpu'>

        >>> ckpt = create_checkpoint_config(strategy="full", checkpoint_ratio=1.0)
        >>> config = create_memory_config(checkpoint_config=ckpt, cpu_offload=True)
        >>> config.cpu_offload
        True
    """
    if checkpoint_config is None:
        checkpoint_config = create_checkpoint_config()

    if offload_config is None:
        offload_config = create_offload_config()

    config = MemoryConfig(
        checkpoint_config=checkpoint_config,
        offload_config=offload_config,
        cpu_offload=cpu_offload,
    )
    validate_memory_config(config)
    return config


def calculate_memory_savings(
    model_params_billions: float,
    num_layers: int,
    strategy: str = "full",
    checkpoint_ratio: float = 1.0,
    batch_size: int = 1,
    sequence_length: int = 2048,
    hidden_size: int = 4096,
) -> tuple[float, float, float]:
    """Calculate memory savings from checkpointing.

    Args:
        model_params_billions: Model size in billions of parameters.
        num_layers: Number of transformer layers.
        strategy: Checkpointing strategy. Defaults to "full".
        checkpoint_ratio: Ratio of layers to checkpoint. Defaults to 1.0.
        batch_size: Training batch size. Defaults to 1.
        sequence_length: Sequence length. Defaults to 2048.
        hidden_size: Hidden dimension size. Defaults to 4096.

    Returns:
        Tuple of (baseline_gb, checkpointed_gb, savings_gb).

    Raises:
        ValueError: If model_params_billions is not positive.
        ValueError: If num_layers is not positive.
        ValueError: If checkpoint_ratio is not in [0.0, 1.0].

    Examples:
        >>> baseline, ckpt, saved = calculate_memory_savings(7.0, 32)
        >>> saved > 0
        True
        >>> ckpt < baseline
        True

        >>> baseline, ckpt, saved = calculate_memory_savings(7.0, 32, "none", 0.0)
        >>> saved
        0.0

        >>> calculate_memory_savings(0, 32)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params_billions must be positive
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if not 0.0 <= checkpoint_ratio <= 1.0:
        msg = f"checkpoint_ratio must be in [0.0, 1.0], got {checkpoint_ratio}"
        raise ValueError(msg)

    # Activation memory estimation (approximate)
    # Each layer stores activations proportional to batch * seq * hidden
    activation_per_layer_gb = (
        batch_size * sequence_length * hidden_size * 4 * 4  # 4 tensors, 4 bytes
    ) / (1024**3)

    baseline_activation_memory_gb = activation_per_layer_gb * num_layers

    # With checkpointing, only store activations for non-checkpointed layers
    # Checkpointed layers store only inputs, not intermediate activations
    if strategy == "none":
        effective_ratio = 0.0
    elif strategy == "full":
        effective_ratio = 1.0
    else:
        effective_ratio = checkpoint_ratio

    # Checkpointed layers reduce activation storage by ~60-70%
    reduction_factor = 0.35  # Keep 35% of activations for checkpointed layers
    checkpointed_layers = int(num_layers * effective_ratio)
    non_checkpointed_layers = num_layers - checkpointed_layers

    checkpointed_activation_memory_gb = (
        activation_per_layer_gb * non_checkpointed_layers
        + activation_per_layer_gb * checkpointed_layers * reduction_factor
    )

    savings_gb = baseline_activation_memory_gb - checkpointed_activation_memory_gb

    return (
        round(baseline_activation_memory_gb, 2),
        round(checkpointed_activation_memory_gb, 2),
        round(savings_gb, 2),
    )


def estimate_recomputation_overhead(
    strategy: str = "full",
    checkpoint_ratio: float = 1.0,
    num_layers: int = 32,
) -> float:
    """Estimate recomputation overhead from checkpointing.

    Args:
        strategy: Checkpointing strategy. Defaults to "full".
        checkpoint_ratio: Ratio of layers to checkpoint. Defaults to 1.0.
        num_layers: Number of transformer layers. Defaults to 32.

    Returns:
        Estimated overhead percentage (additional compute time).

    Raises:
        ValueError: If checkpoint_ratio is not in [0.0, 1.0].
        ValueError: If num_layers is not positive.

    Examples:
        >>> overhead = estimate_recomputation_overhead("full", 1.0, 32)
        >>> overhead > 0
        True
        >>> overhead <= 50.0
        True

        >>> overhead = estimate_recomputation_overhead("none", 0.0, 32)
        >>> overhead
        0.0

        >>> estimate_recomputation_overhead("full", 1.5, 32)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: checkpoint_ratio must be in [0.0, 1.0]
    """
    if not 0.0 <= checkpoint_ratio <= 1.0:
        msg = f"checkpoint_ratio must be in [0.0, 1.0], got {checkpoint_ratio}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if strategy == "none":
        return 0.0

    # Base overhead: checkpointing adds ~33% compute for full model
    # This is because we recompute forward pass during backward
    base_overhead = 33.0

    if strategy == "full":
        effective_ratio = 1.0
    elif strategy == "offload":
        # Offload has less recomputation but adds transfer overhead
        effective_ratio = checkpoint_ratio * 0.5  # Transfer adds ~50% of recompute
    else:
        effective_ratio = checkpoint_ratio

    return round(base_overhead * effective_ratio, 2)


def calculate_optimal_checkpoint_ratio(
    available_gpu_memory_gb: float,
    model_params_billions: float,
    num_layers: int,
    batch_size: int = 1,
    sequence_length: int = 2048,
    hidden_size: int = 4096,
) -> float:
    """Calculate optimal checkpoint ratio for given memory constraints.

    Args:
        available_gpu_memory_gb: Available GPU memory in GB.
        model_params_billions: Model size in billions of parameters.
        num_layers: Number of transformer layers.
        batch_size: Training batch size. Defaults to 1.
        sequence_length: Sequence length. Defaults to 2048.
        hidden_size: Hidden dimension size. Defaults to 4096.

    Returns:
        Recommended checkpoint ratio (0.0-1.0).

    Raises:
        ValueError: If available_gpu_memory_gb is not positive.
        ValueError: If model_params_billions is not positive.
        ValueError: If num_layers is not positive.

    Examples:
        >>> ratio = calculate_optimal_checkpoint_ratio(24.0, 7.0, 32)
        >>> 0.0 <= ratio <= 1.0
        True

        >>> ratio = calculate_optimal_checkpoint_ratio(80.0, 1.0, 32)
        >>> ratio == 0.0  # Plenty of memory for small model, no checkpointing
        True

        >>> calculate_optimal_checkpoint_ratio(0, 7.0, 32)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: available_gpu_memory_gb must be positive
    """
    if available_gpu_memory_gb <= 0:
        msg = f"available_gpu_memory_gb must be positive, got {available_gpu_memory_gb}"
        raise ValueError(msg)

    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    # Estimate model memory (FP16 weights + optimizer states)
    model_memory_gb = model_params_billions * 2  # FP16
    optimizer_memory_gb = model_params_billions * 4 * 2  # FP32 Adam

    # Estimate baseline activation memory
    baseline, _, _ = calculate_memory_savings(
        model_params_billions,
        num_layers,
        strategy="none",
        checkpoint_ratio=0.0,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
    )

    # Total baseline memory
    total_baseline = model_memory_gb + optimizer_memory_gb + baseline

    # Memory headroom needed
    if total_baseline <= available_gpu_memory_gb * 0.9:
        # Plenty of memory, minimal checkpointing
        return 0.0

    # Calculate how much memory we need to save
    memory_deficit = total_baseline - available_gpu_memory_gb * 0.85

    # Maximum savings possible with full checkpointing
    _, _checkpointed_full, max_savings = calculate_memory_savings(
        model_params_billions,
        num_layers,
        strategy="full",
        checkpoint_ratio=1.0,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
    )

    if max_savings <= 0:
        return 1.0

    # Calculate required ratio
    required_ratio = memory_deficit / max_savings
    return min(1.0, max(0.0, round(required_ratio, 2)))


def select_checkpoint_layers(
    num_layers: int,
    checkpoint_ratio: float,
    granularity: str = "block",
) -> list[int]:
    """Select which layers to checkpoint based on ratio and granularity.

    Args:
        num_layers: Total number of layers.
        checkpoint_ratio: Ratio of layers to checkpoint.
        granularity: Checkpoint granularity. Defaults to "block".

    Returns:
        List of layer indices to checkpoint.

    Raises:
        ValueError: If num_layers is not positive.
        ValueError: If checkpoint_ratio is not in [0.0, 1.0].

    Examples:
        >>> layers = select_checkpoint_layers(32, 0.5)
        >>> len(layers)
        16
        >>> all(0 <= l < 32 for l in layers)
        True

        >>> layers = select_checkpoint_layers(32, 1.0)
        >>> len(layers)
        32

        >>> layers = select_checkpoint_layers(32, 0.0)
        >>> layers
        []

        >>> select_checkpoint_layers(0, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_layers must be positive
    """
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if not 0.0 <= checkpoint_ratio <= 1.0:
        msg = f"checkpoint_ratio must be in [0.0, 1.0], got {checkpoint_ratio}"
        raise ValueError(msg)

    num_to_checkpoint = int(num_layers * checkpoint_ratio)

    if num_to_checkpoint == 0:
        return []

    if num_to_checkpoint >= num_layers:
        return list(range(num_layers))

    # Select layers evenly distributed
    # For attention/mlp granularity, prefer deeper layers (more expensive)
    if granularity in ("attention", "mlp"):
        # Checkpoint later layers first (more memory intensive)
        step = num_layers / num_to_checkpoint
        return [int(num_layers - 1 - i * step) for i in range(num_to_checkpoint)]
    else:
        # Evenly distributed for block/layer granularity
        step = num_layers / num_to_checkpoint
        return [int(i * step) for i in range(num_to_checkpoint)]


def format_memory_stats(stats: MemoryStats) -> str:
    """Format memory statistics for display.

    Args:
        stats: Memory statistics to format.

    Returns:
        Formatted string with statistics.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = MemoryStats(32.0, 12.0, 20.0, 62.5, 33.0)
        >>> formatted = format_memory_stats(stats)
        >>> "Baseline:" in formatted
        True
        >>> "Savings:" in formatted
        True

        >>> format_memory_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    lines = [
        f"Baseline: {stats.baseline_memory_gb:.2f} GB",
        f"Checkpointed: {stats.checkpointed_memory_gb:.2f} GB",
        f"Savings: {stats.memory_saved_gb:.2f} GB ({stats.savings_percentage:.1f}%)",
        f"Recomputation Overhead: {stats.recomputation_overhead_pct:.1f}%",
    ]
    return "\n".join(lines)


def get_recommended_checkpoint_config(
    model_params_billions: float,
    available_gpu_memory_gb: float = 24.0,
    num_layers: int = 32,
    batch_size: int = 1,
) -> CheckpointConfig:
    """Get recommended checkpointing configuration.

    Args:
        model_params_billions: Model size in billions of parameters.
        available_gpu_memory_gb: Available GPU memory in GB. Defaults to 24.0.
        num_layers: Number of transformer layers. Defaults to 32.
        batch_size: Training batch size. Defaults to 1.

    Returns:
        Recommended CheckpointConfig.

    Raises:
        ValueError: If model_params_billions is not positive.
        ValueError: If available_gpu_memory_gb is not positive.
        ValueError: If num_layers is not positive.

    Examples:
        >>> config = get_recommended_checkpoint_config(7.0, 24.0)
        >>> config.strategy in (CheckpointStrategy.FULL, CheckpointStrategy.SELECTIVE)
        True

        >>> config = get_recommended_checkpoint_config(1.0, 80.0)
        >>> config.strategy == CheckpointStrategy.NONE
        True

        >>> get_recommended_checkpoint_config(0, 24.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params_billions must be positive
    """
    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    if available_gpu_memory_gb <= 0:
        msg = f"available_gpu_memory_gb must be positive, got {available_gpu_memory_gb}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    # Calculate optimal ratio
    optimal_ratio = calculate_optimal_checkpoint_ratio(
        available_gpu_memory_gb=available_gpu_memory_gb,
        model_params_billions=model_params_billions,
        num_layers=num_layers,
        batch_size=batch_size,
    )

    # Determine strategy based on ratio
    if optimal_ratio == 0.0:
        return create_checkpoint_config(
            strategy="none",
            granularity="block",
            checkpoint_ratio=0.0,
        )
    elif optimal_ratio >= 0.9:
        return create_checkpoint_config(
            strategy="full",
            granularity="layer",
            checkpoint_ratio=1.0,
        )
    else:
        return create_checkpoint_config(
            strategy="selective",
            granularity="block",
            checkpoint_ratio=optimal_ratio,
        )


def list_checkpoint_strategies() -> list[str]:
    """List supported checkpoint strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_checkpoint_strategies()
        >>> "full" in strategies
        True
        >>> "selective" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_CHECKPOINT_STRATEGIES)


def list_checkpoint_granularities() -> list[str]:
    """List supported checkpoint granularities.

    Returns:
        Sorted list of granularity names.

    Examples:
        >>> granularities = list_checkpoint_granularities()
        >>> "block" in granularities
        True
        >>> "attention" in granularities
        True
        >>> granularities == sorted(granularities)
        True
    """
    return sorted(VALID_CHECKPOINT_GRANULARITIES)


def list_offload_targets() -> list[str]:
    """List supported offload targets.

    Returns:
        Sorted list of target names.

    Examples:
        >>> targets = list_offload_targets()
        >>> "cpu" in targets
        True
        >>> "disk" in targets
        True
        >>> targets == sorted(targets)
        True
    """
    return sorted(VALID_OFFLOAD_TARGETS)


def get_checkpoint_strategy(name: str) -> CheckpointStrategy:
    """Get checkpoint strategy from name.

    Args:
        name: Strategy name.

    Returns:
        CheckpointStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_checkpoint_strategy("full")
        <CheckpointStrategy.FULL: 'full'>

        >>> get_checkpoint_strategy("selective")
        <CheckpointStrategy.SELECTIVE: 'selective'>

        >>> get_checkpoint_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if name not in VALID_CHECKPOINT_STRATEGIES:
        msg = f"strategy must be one of {VALID_CHECKPOINT_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return CheckpointStrategy(name)


def get_checkpoint_granularity(name: str) -> CheckpointGranularity:
    """Get checkpoint granularity from name.

    Args:
        name: Granularity name.

    Returns:
        CheckpointGranularity enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_checkpoint_granularity("block")
        <CheckpointGranularity.BLOCK: 'block'>

        >>> get_checkpoint_granularity("attention")
        <CheckpointGranularity.ATTENTION: 'attention'>

        >>> get_checkpoint_granularity("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: granularity must be one of
    """
    if name not in VALID_CHECKPOINT_GRANULARITIES:
        msg = (
            f"granularity must be one of {VALID_CHECKPOINT_GRANULARITIES}, got '{name}'"
        )
        raise ValueError(msg)
    return CheckpointGranularity(name)


def get_offload_target(name: str) -> OffloadTarget:
    """Get offload target from name.

    Args:
        name: Target name.

    Returns:
        OffloadTarget enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_offload_target("cpu")
        <OffloadTarget.CPU: 'cpu'>

        >>> get_offload_target("disk")
        <OffloadTarget.DISK: 'disk'>

        >>> get_offload_target("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target must be one of
    """
    if name not in VALID_OFFLOAD_TARGETS:
        msg = f"target must be one of {VALID_OFFLOAD_TARGETS}, got '{name}'"
        raise ValueError(msg)
    return OffloadTarget(name)
