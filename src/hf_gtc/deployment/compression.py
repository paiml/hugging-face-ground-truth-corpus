"""Advanced model compression utilities.

This module provides functions for compressing neural network models
through pruning, layer fusion, weight sharing, and low-rank factorization
techniques.

Examples:
    >>> from hf_gtc.deployment.compression import (
    ...     create_pruning_config,
    ...     PruningMethod,
    ... )
    >>> config = create_pruning_config(sparsity=0.5)
    >>> config.sparsity
    0.5
    >>> config.method
    <PruningMethod.MAGNITUDE: 'magnitude'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PruningMethod(Enum):
    """Methods for neural network pruning.

    Attributes:
        MAGNITUDE: Prune weights with smallest absolute values.
        STRUCTURED: Prune entire structures (channels, heads).
        MOVEMENT: Prune based on weight movement during training.
        LOTTERY_TICKET: Iterative magnitude pruning with rewinding.
        GRADUAL: Gradually increase sparsity during training.

    Examples:
        >>> PruningMethod.MAGNITUDE.value
        'magnitude'
        >>> PruningMethod.STRUCTURED.value
        'structured'
        >>> PruningMethod.MOVEMENT.value
        'movement'
    """

    MAGNITUDE = "magnitude"
    STRUCTURED = "structured"
    MOVEMENT = "movement"
    LOTTERY_TICKET = "lottery_ticket"
    GRADUAL = "gradual"


class PruningScope(Enum):
    """Scope for applying pruning.

    Attributes:
        GLOBAL: Prune across entire model globally.
        LOCAL: Prune within each layer independently.
        LAYER_WISE: Different sparsity per layer.

    Examples:
        >>> PruningScope.GLOBAL.value
        'global'
        >>> PruningScope.LOCAL.value
        'local'
        >>> PruningScope.LAYER_WISE.value
        'layer_wise'
    """

    GLOBAL = "global"
    LOCAL = "local"
    LAYER_WISE = "layer_wise"


class StructuredPruningDim(Enum):
    """Dimensions for structured pruning.

    Attributes:
        CHANNEL: Prune output channels (filters).
        HEAD: Prune attention heads.
        LAYER: Prune entire layers.
        BLOCK: Prune transformer blocks.

    Examples:
        >>> StructuredPruningDim.CHANNEL.value
        'channel'
        >>> StructuredPruningDim.HEAD.value
        'head'
        >>> StructuredPruningDim.LAYER.value
        'layer'
    """

    CHANNEL = "channel"
    HEAD = "head"
    LAYER = "layer"
    BLOCK = "block"


class FusionType(Enum):
    """Types of layer fusion operations.

    Attributes:
        CONV_BN: Fuse convolution with batch normalization.
        LINEAR_BN: Fuse linear layer with batch normalization.
        ATTENTION: Fuse attention components.
        MLP: Fuse MLP layers.

    Examples:
        >>> FusionType.CONV_BN.value
        'conv_bn'
        >>> FusionType.LINEAR_BN.value
        'linear_bn'
        >>> FusionType.ATTENTION.value
        'attention'
    """

    CONV_BN = "conv_bn"
    LINEAR_BN = "linear_bn"
    ATTENTION = "attention"
    MLP = "mlp"


class DecompositionMethod(Enum):
    """Methods for low-rank matrix decomposition.

    Attributes:
        SVD: Singular Value Decomposition.
        TUCKER: Tucker decomposition for tensors.
        CP: CP (CANDECOMP/PARAFAC) decomposition.
        NMF: Non-negative Matrix Factorization.

    Examples:
        >>> DecompositionMethod.SVD.value
        'svd'
        >>> DecompositionMethod.TUCKER.value
        'tucker'
        >>> DecompositionMethod.CP.value
        'cp'
    """

    SVD = "svd"
    TUCKER = "tucker"
    CP = "cp"
    NMF = "nmf"


class ImportanceMetric(Enum):
    """Metrics for measuring weight importance.

    Attributes:
        L1_NORM: L1 norm of weights.
        L2_NORM: L2 norm of weights.
        GRADIENT: Gradient-based importance.
        TAYLOR: Taylor expansion importance.
        HESSIAN: Hessian-based importance.

    Examples:
        >>> ImportanceMetric.L1_NORM.value
        'l1_norm'
        >>> ImportanceMetric.GRADIENT.value
        'gradient'
        >>> ImportanceMetric.TAYLOR.value
        'taylor'
    """

    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    GRADIENT = "gradient"
    TAYLOR = "taylor"
    HESSIAN = "hessian"


VALID_PRUNING_METHODS = frozenset(m.value for m in PruningMethod)
VALID_PRUNING_SCOPES = frozenset(s.value for s in PruningScope)
VALID_STRUCTURED_DIMS = frozenset(d.value for d in StructuredPruningDim)
VALID_FUSION_TYPES = frozenset(f.value for f in FusionType)
VALID_DECOMPOSITION_METHODS = frozenset(d.value for d in DecompositionMethod)
VALID_IMPORTANCE_METRICS = frozenset(m.value for m in ImportanceMetric)


@dataclass(frozen=True, slots=True)
class PruningSchedule:
    """Schedule for gradual pruning.

    Attributes:
        initial_sparsity: Starting sparsity level.
        final_sparsity: Target sparsity level.
        begin_step: Step to begin pruning.
        end_step: Step to end pruning.
        frequency: Steps between pruning updates.

    Examples:
        >>> schedule = PruningSchedule(
        ...     initial_sparsity=0.0,
        ...     final_sparsity=0.9,
        ...     begin_step=0,
        ...     end_step=1000,
        ...     frequency=100,
        ... )
        >>> schedule.final_sparsity
        0.9
    """

    initial_sparsity: float
    final_sparsity: float
    begin_step: int
    end_step: int
    frequency: int


@dataclass(frozen=True, slots=True)
class PruningConfig:
    """Configuration for model pruning.

    Attributes:
        method: Pruning method to use.
        sparsity: Target sparsity level (0 to 1).
        scope: Scope for applying pruning.
        importance_metric: Metric for weight importance.
        schedule: Optional pruning schedule for gradual pruning.

    Examples:
        >>> config = PruningConfig(
        ...     method=PruningMethod.MAGNITUDE,
        ...     sparsity=0.5,
        ...     scope=PruningScope.GLOBAL,
        ...     importance_metric=ImportanceMetric.L1_NORM,
        ...     schedule=None,
        ... )
        >>> config.sparsity
        0.5
    """

    method: PruningMethod
    sparsity: float
    scope: PruningScope
    importance_metric: ImportanceMetric
    schedule: PruningSchedule | None


@dataclass(frozen=True, slots=True)
class StructuredPruningConfig:
    """Configuration for structured pruning.

    Attributes:
        dimension: Dimension to prune along.
        pruning_ratio: Ratio of structures to prune.
        importance_metric: Metric for structure importance.
        min_structures: Minimum structures to keep.

    Examples:
        >>> config = StructuredPruningConfig(
        ...     dimension=StructuredPruningDim.CHANNEL,
        ...     pruning_ratio=0.3,
        ...     importance_metric=ImportanceMetric.L2_NORM,
        ...     min_structures=1,
        ... )
        >>> config.pruning_ratio
        0.3
    """

    dimension: StructuredPruningDim
    pruning_ratio: float
    importance_metric: ImportanceMetric
    min_structures: int


@dataclass(frozen=True, slots=True)
class LayerFusionConfig:
    """Configuration for layer fusion.

    Attributes:
        fusion_type: Type of fusion to perform.
        fuse_activation: Whether to fuse activation function.
        optimize_for_inference: Optimize fused layers for inference.

    Examples:
        >>> config = LayerFusionConfig(
        ...     fusion_type=FusionType.CONV_BN,
        ...     fuse_activation=True,
        ...     optimize_for_inference=True,
        ... )
        >>> config.fusion_type
        <FusionType.CONV_BN: 'conv_bn'>
    """

    fusion_type: FusionType
    fuse_activation: bool
    optimize_for_inference: bool


@dataclass(frozen=True, slots=True)
class WeightSharingConfig:
    """Configuration for weight sharing/clustering.

    Attributes:
        num_clusters: Number of weight clusters.
        clustering_method: Method for clustering (kmeans, linear).
        fine_tune_centroids: Whether to fine-tune cluster centroids.
        bits_per_weight: Bits used to encode cluster indices.

    Examples:
        >>> config = WeightSharingConfig(
        ...     num_clusters=256,
        ...     clustering_method="kmeans",
        ...     fine_tune_centroids=True,
        ...     bits_per_weight=8,
        ... )
        >>> config.num_clusters
        256
    """

    num_clusters: int
    clustering_method: str
    fine_tune_centroids: bool
    bits_per_weight: int


@dataclass(frozen=True, slots=True)
class LowRankConfig:
    """Configuration for low-rank factorization.

    Attributes:
        rank: Target rank for decomposition.
        decomposition_method: Method for decomposition.
        rank_ratio: Ratio of original rank to keep (alternative to rank).
        energy_threshold: Energy threshold for automatic rank selection.

    Examples:
        >>> config = LowRankConfig(
        ...     rank=64,
        ...     decomposition_method=DecompositionMethod.SVD,
        ...     rank_ratio=None,
        ...     energy_threshold=0.99,
        ... )
        >>> config.rank
        64
    """

    rank: int | None
    decomposition_method: DecompositionMethod
    rank_ratio: float | None
    energy_threshold: float


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    """Main configuration for model compression.

    Attributes:
        enable_pruning: Whether to apply pruning.
        enable_fusion: Whether to apply layer fusion.
        enable_weight_sharing: Whether to apply weight sharing.
        enable_low_rank: Whether to apply low-rank factorization.
        target_speedup: Target inference speedup factor.
        target_size_reduction: Target model size reduction ratio.

    Examples:
        >>> config = CompressionConfig(
        ...     enable_pruning=True,
        ...     enable_fusion=True,
        ...     enable_weight_sharing=False,
        ...     enable_low_rank=False,
        ...     target_speedup=2.0,
        ...     target_size_reduction=0.5,
        ... )
        >>> config.target_speedup
        2.0
    """

    enable_pruning: bool
    enable_fusion: bool
    enable_weight_sharing: bool
    enable_low_rank: bool
    target_speedup: float
    target_size_reduction: float


@dataclass(frozen=True, slots=True)
class CompressionStats:
    """Statistics from model compression.

    Attributes:
        original_params: Original parameter count.
        compressed_params: Compressed parameter count.
        sparsity: Achieved sparsity level.
        compression_ratio: Size compression ratio.
        estimated_speedup: Estimated inference speedup.
        original_size_mb: Original model size in MB.
        compressed_size_mb: Compressed model size in MB.

    Examples:
        >>> stats = CompressionStats(
        ...     original_params=110_000_000,
        ...     compressed_params=55_000_000,
        ...     sparsity=0.5,
        ...     compression_ratio=2.0,
        ...     estimated_speedup=1.8,
        ...     original_size_mb=440.0,
        ...     compressed_size_mb=220.0,
        ... )
        >>> stats.compression_ratio
        2.0
    """

    original_params: int
    compressed_params: int
    sparsity: float
    compression_ratio: float
    estimated_speedup: float
    original_size_mb: float
    compressed_size_mb: float


def validate_pruning_config(config: PruningConfig) -> None:
    """Validate pruning configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = PruningConfig(
        ...     PruningMethod.MAGNITUDE, 0.5, PruningScope.GLOBAL,
        ...     ImportanceMetric.L1_NORM, None
        ... )
        >>> validate_pruning_config(config)

        >>> bad_config = PruningConfig(
        ...     PruningMethod.MAGNITUDE, 1.5, PruningScope.GLOBAL,
        ...     ImportanceMetric.L1_NORM, None
        ... )
        >>> validate_pruning_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: sparsity must be between 0 and 1, got 1.5
    """
    if not 0 <= config.sparsity <= 1:
        msg = f"sparsity must be between 0 and 1, got {config.sparsity}"
        raise ValueError(msg)

    if config.schedule is not None:
        if config.schedule.begin_step < 0:
            msg = f"begin_step must be non-negative, got {config.schedule.begin_step}"
            raise ValueError(msg)
        if config.schedule.end_step <= config.schedule.begin_step:
            msg = "end_step must be greater than begin_step"
            raise ValueError(msg)
        if config.schedule.frequency <= 0:
            msg = f"frequency must be positive, got {config.schedule.frequency}"
            raise ValueError(msg)


def validate_structured_pruning_config(config: StructuredPruningConfig) -> None:
    """Validate structured pruning configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = StructuredPruningConfig(
        ...     StructuredPruningDim.CHANNEL, 0.3, ImportanceMetric.L2_NORM, 1
        ... )
        >>> validate_structured_pruning_config(config)

        >>> bad_config = StructuredPruningConfig(
        ...     StructuredPruningDim.CHANNEL, 1.5, ImportanceMetric.L2_NORM, 1
        ... )
        >>> validate_structured_pruning_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: pruning_ratio must be between 0 and 1, got 1.5
    """
    if not 0 <= config.pruning_ratio <= 1:
        msg = f"pruning_ratio must be between 0 and 1, got {config.pruning_ratio}"
        raise ValueError(msg)
    if config.min_structures < 0:
        msg = f"min_structures must be non-negative, got {config.min_structures}"
        raise ValueError(msg)


def validate_weight_sharing_config(config: WeightSharingConfig) -> None:
    """Validate weight sharing configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = WeightSharingConfig(256, "kmeans", True, 8)
        >>> validate_weight_sharing_config(config)

        >>> bad_config = WeightSharingConfig(0, "kmeans", True, 8)
        >>> validate_weight_sharing_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: num_clusters must be positive, got 0
    """
    if config.num_clusters <= 0:
        msg = f"num_clusters must be positive, got {config.num_clusters}"
        raise ValueError(msg)
    if config.bits_per_weight <= 0:
        msg = f"bits_per_weight must be positive, got {config.bits_per_weight}"
        raise ValueError(msg)
    if config.num_clusters > 2**config.bits_per_weight:
        msg = (
            f"num_clusters ({config.num_clusters}) exceeds maximum for "
            f"{config.bits_per_weight} bits ({2**config.bits_per_weight})"
        )
        raise ValueError(msg)


def validate_low_rank_config(config: LowRankConfig) -> None:
    """Validate low-rank factorization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = LowRankConfig(64, DecompositionMethod.SVD, None, 0.99)
        >>> validate_low_rank_config(config)

        >>> bad_config = LowRankConfig(0, DecompositionMethod.SVD, None, 0.99)
        >>> validate_low_rank_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: rank must be positive, got 0
    """
    if config.rank is not None and config.rank <= 0:
        msg = f"rank must be positive, got {config.rank}"
        raise ValueError(msg)
    if config.rank_ratio is not None and not 0 < config.rank_ratio <= 1:
        msg = f"rank_ratio must be between 0 and 1, got {config.rank_ratio}"
        raise ValueError(msg)
    if not 0 < config.energy_threshold <= 1:
        msg = f"energy_threshold must be between 0 and 1, got {config.energy_threshold}"
        raise ValueError(msg)
    if config.rank is None and config.rank_ratio is None:
        msg = "either rank or rank_ratio must be specified"
        raise ValueError(msg)


def validate_compression_config(config: CompressionConfig) -> None:
    """Validate compression configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = CompressionConfig(True, True, False, False, 2.0, 0.5)
        >>> validate_compression_config(config)

        >>> bad_config = CompressionConfig(True, True, False, False, 0.5, 0.5)
        >>> validate_compression_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: target_speedup must be >= 1.0, got 0.5
    """
    if config.target_speedup < 1.0:
        msg = f"target_speedup must be >= 1.0, got {config.target_speedup}"
        raise ValueError(msg)
    if not 0 < config.target_size_reduction <= 1:
        msg = (
            f"target_size_reduction must be between 0 and 1, "
            f"got {config.target_size_reduction}"
        )
        raise ValueError(msg)


def create_pruning_schedule(
    initial_sparsity: float = 0.0,
    final_sparsity: float = 0.9,
    begin_step: int = 0,
    end_step: int = 1000,
    frequency: int = 100,
) -> PruningSchedule:
    """Create a pruning schedule for gradual pruning.

    Args:
        initial_sparsity: Starting sparsity level.
        final_sparsity: Target sparsity level.
        begin_step: Step to begin pruning.
        end_step: Step to end pruning.
        frequency: Steps between pruning updates.

    Returns:
        PruningSchedule instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> schedule = create_pruning_schedule(final_sparsity=0.8)
        >>> schedule.final_sparsity
        0.8

        >>> create_pruning_schedule(initial_sparsity=1.5)
        Traceback (most recent call last):
            ...
        ValueError: initial_sparsity must be between 0 and 1, got 1.5
    """
    if not 0 <= initial_sparsity <= 1:
        msg = f"initial_sparsity must be between 0 and 1, got {initial_sparsity}"
        raise ValueError(msg)
    if not 0 <= final_sparsity <= 1:
        msg = f"final_sparsity must be between 0 and 1, got {final_sparsity}"
        raise ValueError(msg)
    if begin_step < 0:
        msg = f"begin_step must be non-negative, got {begin_step}"
        raise ValueError(msg)
    if end_step <= begin_step:
        msg = "end_step must be greater than begin_step"
        raise ValueError(msg)
    if frequency <= 0:
        msg = f"frequency must be positive, got {frequency}"
        raise ValueError(msg)

    return PruningSchedule(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step,
        frequency=frequency,
    )


def create_pruning_config(
    method: str | PruningMethod = PruningMethod.MAGNITUDE,
    sparsity: float = 0.5,
    scope: str | PruningScope = PruningScope.GLOBAL,
    importance_metric: str | ImportanceMetric = ImportanceMetric.L1_NORM,
    schedule: PruningSchedule | None = None,
) -> PruningConfig:
    """Create a pruning configuration.

    Args:
        method: Pruning method to use.
        sparsity: Target sparsity level.
        scope: Scope for applying pruning.
        importance_metric: Metric for weight importance.
        schedule: Optional pruning schedule.

    Returns:
        Validated PruningConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_pruning_config()
        >>> config.sparsity
        0.5
        >>> config.method
        <PruningMethod.MAGNITUDE: 'magnitude'>

        >>> config = create_pruning_config(method="structured", sparsity=0.7)
        >>> config.method
        <PruningMethod.STRUCTURED: 'structured'>

        >>> create_pruning_config(sparsity=1.5)
        Traceback (most recent call last):
            ...
        ValueError: sparsity must be between 0 and 1, got 1.5
    """
    if isinstance(method, str):
        method = get_pruning_method(method)
    if isinstance(scope, str):
        scope = get_pruning_scope(scope)
    if isinstance(importance_metric, str):
        importance_metric = get_importance_metric(importance_metric)

    config = PruningConfig(
        method=method,
        sparsity=sparsity,
        scope=scope,
        importance_metric=importance_metric,
        schedule=schedule,
    )
    validate_pruning_config(config)
    return config


def create_structured_pruning_config(
    dimension: str | StructuredPruningDim = StructuredPruningDim.CHANNEL,
    pruning_ratio: float = 0.3,
    importance_metric: str | ImportanceMetric = ImportanceMetric.L2_NORM,
    min_structures: int = 1,
) -> StructuredPruningConfig:
    """Create a structured pruning configuration.

    Args:
        dimension: Dimension to prune along.
        pruning_ratio: Ratio of structures to prune.
        importance_metric: Metric for importance.
        min_structures: Minimum structures to keep.

    Returns:
        Validated StructuredPruningConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_structured_pruning_config()
        >>> config.dimension
        <StructuredPruningDim.CHANNEL: 'channel'>
        >>> config.pruning_ratio
        0.3

        >>> config = create_structured_pruning_config(dimension="head")
        >>> config.dimension
        <StructuredPruningDim.HEAD: 'head'>

        >>> create_structured_pruning_config(pruning_ratio=1.5)
        Traceback (most recent call last):
            ...
        ValueError: pruning_ratio must be between 0 and 1, got 1.5
    """
    if isinstance(dimension, str):
        dimension = get_structured_pruning_dim(dimension)
    if isinstance(importance_metric, str):
        importance_metric = get_importance_metric(importance_metric)

    config = StructuredPruningConfig(
        dimension=dimension,
        pruning_ratio=pruning_ratio,
        importance_metric=importance_metric,
        min_structures=min_structures,
    )
    validate_structured_pruning_config(config)
    return config


def create_layer_fusion_config(
    fusion_type: str | FusionType = FusionType.CONV_BN,
    fuse_activation: bool = True,
    optimize_for_inference: bool = True,
) -> LayerFusionConfig:
    """Create a layer fusion configuration.

    Args:
        fusion_type: Type of fusion to perform.
        fuse_activation: Whether to fuse activation.
        optimize_for_inference: Optimize for inference.

    Returns:
        LayerFusionConfig instance.

    Examples:
        >>> config = create_layer_fusion_config()
        >>> config.fusion_type
        <FusionType.CONV_BN: 'conv_bn'>
        >>> config.fuse_activation
        True

        >>> config = create_layer_fusion_config(fusion_type="linear_bn")
        >>> config.fusion_type
        <FusionType.LINEAR_BN: 'linear_bn'>
    """
    if isinstance(fusion_type, str):
        fusion_type = get_fusion_type(fusion_type)

    return LayerFusionConfig(
        fusion_type=fusion_type,
        fuse_activation=fuse_activation,
        optimize_for_inference=optimize_for_inference,
    )


def create_weight_sharing_config(
    num_clusters: int = 256,
    clustering_method: str = "kmeans",
    fine_tune_centroids: bool = True,
    bits_per_weight: int = 8,
) -> WeightSharingConfig:
    """Create a weight sharing configuration.

    Args:
        num_clusters: Number of weight clusters.
        clustering_method: Clustering method (kmeans, linear).
        fine_tune_centroids: Whether to fine-tune centroids.
        bits_per_weight: Bits for cluster indices.

    Returns:
        Validated WeightSharingConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_weight_sharing_config()
        >>> config.num_clusters
        256
        >>> config.bits_per_weight
        8

        >>> config = create_weight_sharing_config(num_clusters=16, bits_per_weight=4)
        >>> config.num_clusters
        16

        >>> create_weight_sharing_config(num_clusters=0)
        Traceback (most recent call last):
            ...
        ValueError: num_clusters must be positive, got 0
    """
    valid_methods = frozenset({"kmeans", "linear", "uniform"})
    if clustering_method not in valid_methods:
        msg = (
            f"clustering_method must be one of {valid_methods}, "
            f"got '{clustering_method}'"
        )
        raise ValueError(msg)

    config = WeightSharingConfig(
        num_clusters=num_clusters,
        clustering_method=clustering_method,
        fine_tune_centroids=fine_tune_centroids,
        bits_per_weight=bits_per_weight,
    )
    validate_weight_sharing_config(config)
    return config


def create_low_rank_config(
    rank: int | None = 64,
    decomposition_method: str | DecompositionMethod = DecompositionMethod.SVD,
    rank_ratio: float | None = None,
    energy_threshold: float = 0.99,
) -> LowRankConfig:
    """Create a low-rank factorization configuration.

    Args:
        rank: Target rank for decomposition.
        decomposition_method: Method for decomposition.
        rank_ratio: Ratio of original rank to keep.
        energy_threshold: Energy threshold for auto rank.

    Returns:
        Validated LowRankConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_low_rank_config()
        >>> config.rank
        64
        >>> config.decomposition_method
        <DecompositionMethod.SVD: 'svd'>

        >>> config = create_low_rank_config(rank_ratio=0.5, rank=None)
        >>> config.rank_ratio
        0.5

        >>> create_low_rank_config(rank=0)
        Traceback (most recent call last):
            ...
        ValueError: rank must be positive, got 0
    """
    if isinstance(decomposition_method, str):
        decomposition_method = get_decomposition_method(decomposition_method)

    config = LowRankConfig(
        rank=rank,
        decomposition_method=decomposition_method,
        rank_ratio=rank_ratio,
        energy_threshold=energy_threshold,
    )
    validate_low_rank_config(config)
    return config


def create_compression_config(
    enable_pruning: bool = True,
    enable_fusion: bool = True,
    enable_weight_sharing: bool = False,
    enable_low_rank: bool = False,
    target_speedup: float = 2.0,
    target_size_reduction: float = 0.5,
) -> CompressionConfig:
    """Create a main compression configuration.

    Args:
        enable_pruning: Whether to enable pruning.
        enable_fusion: Whether to enable layer fusion.
        enable_weight_sharing: Whether to enable weight sharing.
        enable_low_rank: Whether to enable low-rank factorization.
        target_speedup: Target inference speedup.
        target_size_reduction: Target size reduction ratio.

    Returns:
        Validated CompressionConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_compression_config()
        >>> config.target_speedup
        2.0
        >>> config.enable_pruning
        True

        >>> config = create_compression_config(target_speedup=3.0)
        >>> config.target_speedup
        3.0

        >>> create_compression_config(target_speedup=0.5)
        Traceback (most recent call last):
            ...
        ValueError: target_speedup must be >= 1.0, got 0.5
    """
    config = CompressionConfig(
        enable_pruning=enable_pruning,
        enable_fusion=enable_fusion,
        enable_weight_sharing=enable_weight_sharing,
        enable_low_rank=enable_low_rank,
        target_speedup=target_speedup,
        target_size_reduction=target_size_reduction,
    )
    validate_compression_config(config)
    return config


def create_compression_stats(
    original_params: int,
    compressed_params: int,
    original_size_mb: float | None = None,
    compressed_size_mb: float | None = None,
) -> CompressionStats:
    """Create compression statistics.

    Args:
        original_params: Original parameter count.
        compressed_params: Compressed parameter count.
        original_size_mb: Original size in MB (calculated if None).
        compressed_size_mb: Compressed size in MB (calculated if None).

    Returns:
        CompressionStats instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_compression_stats(100_000_000, 50_000_000)
        >>> stats.compression_ratio
        2.0
        >>> stats.sparsity
        0.5

        >>> create_compression_stats(0, 50_000_000)
        Traceback (most recent call last):
            ...
        ValueError: original_params must be positive, got 0
    """
    if original_params <= 0:
        msg = f"original_params must be positive, got {original_params}"
        raise ValueError(msg)
    if compressed_params < 0:
        msg = f"compressed_params must be non-negative, got {compressed_params}"
        raise ValueError(msg)
    if compressed_params > original_params:
        msg = (
            f"compressed_params ({compressed_params}) cannot exceed "
            f"original_params ({original_params})"
        )
        raise ValueError(msg)

    if original_size_mb is None:
        original_size_mb = original_params * 4 / (1024 * 1024)
    if compressed_size_mb is None:
        compressed_size_mb = compressed_params * 4 / (1024 * 1024)

    sparsity = 1 - (compressed_params / original_params)
    compression_ratio = (
        original_params / compressed_params if compressed_params > 0 else float("inf")
    )
    estimated_speedup = estimate_speedup_from_sparsity(sparsity)

    return CompressionStats(
        original_params=original_params,
        compressed_params=compressed_params,
        sparsity=sparsity,
        compression_ratio=compression_ratio,
        estimated_speedup=estimated_speedup,
        original_size_mb=original_size_mb,
        compressed_size_mb=compressed_size_mb,
    )


def list_pruning_methods() -> list[str]:
    """List all available pruning methods.

    Returns:
        Sorted list of pruning method names.

    Examples:
        >>> methods = list_pruning_methods()
        >>> "magnitude" in methods
        True
        >>> "structured" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_PRUNING_METHODS)


def list_pruning_scopes() -> list[str]:
    """List all available pruning scopes.

    Returns:
        Sorted list of pruning scope names.

    Examples:
        >>> scopes = list_pruning_scopes()
        >>> "global" in scopes
        True
        >>> "local" in scopes
        True
        >>> scopes == sorted(scopes)
        True
    """
    return sorted(VALID_PRUNING_SCOPES)


def list_structured_pruning_dims() -> list[str]:
    """List all available structured pruning dimensions.

    Returns:
        Sorted list of structured pruning dimension names.

    Examples:
        >>> dims = list_structured_pruning_dims()
        >>> "channel" in dims
        True
        >>> "head" in dims
        True
        >>> dims == sorted(dims)
        True
    """
    return sorted(VALID_STRUCTURED_DIMS)


def list_fusion_types() -> list[str]:
    """List all available fusion types.

    Returns:
        Sorted list of fusion type names.

    Examples:
        >>> types = list_fusion_types()
        >>> "conv_bn" in types
        True
        >>> "linear_bn" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_FUSION_TYPES)


def list_decomposition_methods() -> list[str]:
    """List all available decomposition methods.

    Returns:
        Sorted list of decomposition method names.

    Examples:
        >>> methods = list_decomposition_methods()
        >>> "svd" in methods
        True
        >>> "tucker" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_DECOMPOSITION_METHODS)


def list_importance_metrics() -> list[str]:
    """List all available importance metrics.

    Returns:
        Sorted list of importance metric names.

    Examples:
        >>> metrics = list_importance_metrics()
        >>> "l1_norm" in metrics
        True
        >>> "gradient" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_IMPORTANCE_METRICS)


def get_pruning_method(name: str) -> PruningMethod:
    """Get pruning method enum from string name.

    Args:
        name: Name of the pruning method.

    Returns:
        Corresponding PruningMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_pruning_method("magnitude")
        <PruningMethod.MAGNITUDE: 'magnitude'>
        >>> get_pruning_method("structured")
        <PruningMethod.STRUCTURED: 'structured'>

        >>> get_pruning_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_PRUNING_METHODS:
        msg = f"method must be one of {VALID_PRUNING_METHODS}, got '{name}'"
        raise ValueError(msg)
    return PruningMethod(name)


def get_pruning_scope(name: str) -> PruningScope:
    """Get pruning scope enum from string name.

    Args:
        name: Name of the pruning scope.

    Returns:
        Corresponding PruningScope enum.

    Raises:
        ValueError: If scope name is invalid.

    Examples:
        >>> get_pruning_scope("global")
        <PruningScope.GLOBAL: 'global'>
        >>> get_pruning_scope("local")
        <PruningScope.LOCAL: 'local'>

        >>> get_pruning_scope("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: scope must be one of ...
    """
    if name not in VALID_PRUNING_SCOPES:
        msg = f"scope must be one of {VALID_PRUNING_SCOPES}, got '{name}'"
        raise ValueError(msg)
    return PruningScope(name)


def get_structured_pruning_dim(name: str) -> StructuredPruningDim:
    """Get structured pruning dimension enum from string name.

    Args:
        name: Name of the pruning dimension.

    Returns:
        Corresponding StructuredPruningDim enum.

    Raises:
        ValueError: If dimension name is invalid.

    Examples:
        >>> get_structured_pruning_dim("channel")
        <StructuredPruningDim.CHANNEL: 'channel'>
        >>> get_structured_pruning_dim("head")
        <StructuredPruningDim.HEAD: 'head'>

        >>> get_structured_pruning_dim("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: dimension must be one of ...
    """
    if name not in VALID_STRUCTURED_DIMS:
        msg = f"dimension must be one of {VALID_STRUCTURED_DIMS}, got '{name}'"
        raise ValueError(msg)
    return StructuredPruningDim(name)


def get_fusion_type(name: str) -> FusionType:
    """Get fusion type enum from string name.

    Args:
        name: Name of the fusion type.

    Returns:
        Corresponding FusionType enum.

    Raises:
        ValueError: If fusion type name is invalid.

    Examples:
        >>> get_fusion_type("conv_bn")
        <FusionType.CONV_BN: 'conv_bn'>
        >>> get_fusion_type("linear_bn")
        <FusionType.LINEAR_BN: 'linear_bn'>

        >>> get_fusion_type("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: fusion_type must be one of ...
    """
    if name not in VALID_FUSION_TYPES:
        msg = f"fusion_type must be one of {VALID_FUSION_TYPES}, got '{name}'"
        raise ValueError(msg)
    return FusionType(name)


def get_decomposition_method(name: str) -> DecompositionMethod:
    """Get decomposition method enum from string name.

    Args:
        name: Name of the decomposition method.

    Returns:
        Corresponding DecompositionMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_decomposition_method("svd")
        <DecompositionMethod.SVD: 'svd'>
        >>> get_decomposition_method("tucker")
        <DecompositionMethod.TUCKER: 'tucker'>

        >>> get_decomposition_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_DECOMPOSITION_METHODS:
        msg = f"method must be one of {VALID_DECOMPOSITION_METHODS}, got '{name}'"
        raise ValueError(msg)
    return DecompositionMethod(name)


def get_importance_metric(name: str) -> ImportanceMetric:
    """Get importance metric enum from string name.

    Args:
        name: Name of the importance metric.

    Returns:
        Corresponding ImportanceMetric enum.

    Raises:
        ValueError: If metric name is invalid.

    Examples:
        >>> get_importance_metric("l1_norm")
        <ImportanceMetric.L1_NORM: 'l1_norm'>
        >>> get_importance_metric("gradient")
        <ImportanceMetric.GRADIENT: 'gradient'>

        >>> get_importance_metric("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: metric must be one of ...
    """
    if name not in VALID_IMPORTANCE_METRICS:
        msg = f"metric must be one of {VALID_IMPORTANCE_METRICS}, got '{name}'"
        raise ValueError(msg)
    return ImportanceMetric(name)


def calculate_sparsity_at_step(
    schedule: PruningSchedule,
    current_step: int,
) -> float:
    """Calculate sparsity level at a given training step.

    Args:
        schedule: Pruning schedule configuration.
        current_step: Current training step.

    Returns:
        Sparsity level at the current step.

    Examples:
        >>> schedule = create_pruning_schedule(
        ...     initial_sparsity=0.0,
        ...     final_sparsity=0.9,
        ...     begin_step=0,
        ...     end_step=1000,
        ... )
        >>> calculate_sparsity_at_step(schedule, 0)
        0.0
        >>> calculate_sparsity_at_step(schedule, 500)
        0.45
        >>> calculate_sparsity_at_step(schedule, 1000)
        0.9
    """
    if current_step < schedule.begin_step:
        return schedule.initial_sparsity
    if current_step >= schedule.end_step:
        return schedule.final_sparsity

    progress = (current_step - schedule.begin_step) / (
        schedule.end_step - schedule.begin_step
    )

    sparsity_range = schedule.final_sparsity - schedule.initial_sparsity
    return schedule.initial_sparsity + sparsity_range * progress


def estimate_speedup_from_sparsity(sparsity: float) -> float:
    """Estimate inference speedup from sparsity level.

    Note: Actual speedup depends on hardware and sparse implementation.
    This provides a theoretical upper bound.

    Args:
        sparsity: Sparsity level (0 to 1).

    Returns:
        Estimated speedup factor.

    Raises:
        ValueError: If sparsity is invalid.

    Examples:
        >>> estimate_speedup_from_sparsity(0.0)
        1.0
        >>> estimate_speedup_from_sparsity(0.5)
        2.0
        >>> round(estimate_speedup_from_sparsity(0.9), 1)
        10.0

        >>> estimate_speedup_from_sparsity(1.5)
        Traceback (most recent call last):
            ...
        ValueError: sparsity must be between 0 and 1, got 1.5
    """
    if not 0 <= sparsity < 1:
        if sparsity == 1.0:
            return float("inf")
        msg = f"sparsity must be between 0 and 1, got {sparsity}"
        raise ValueError(msg)

    return 1 / (1 - sparsity)


def estimate_compressed_size(
    original_params: int,
    sparsity: float,
    bits_per_param: int = 32,
) -> int:
    """Estimate compressed model size in bytes.

    Args:
        original_params: Original parameter count.
        sparsity: Target sparsity level.
        bits_per_param: Bits per parameter.

    Returns:
        Estimated size in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate_compressed_size(100_000_000, 0.5)
        200000000
        >>> estimate_compressed_size(100_000_000, 0.9)
        39999996

        >>> estimate_compressed_size(0, 0.5)
        Traceback (most recent call last):
            ...
        ValueError: original_params must be positive, got 0
    """
    if original_params <= 0:
        msg = f"original_params must be positive, got {original_params}"
        raise ValueError(msg)
    if not 0 <= sparsity <= 1:
        msg = f"sparsity must be between 0 and 1, got {sparsity}"
        raise ValueError(msg)
    if bits_per_param <= 0:
        msg = f"bits_per_param must be positive, got {bits_per_param}"
        raise ValueError(msg)

    remaining_params = int(original_params * (1 - sparsity))
    bytes_per_param = bits_per_param // 8
    return remaining_params * bytes_per_param


def calculate_low_rank_params(
    original_shape: tuple[int, int],
    rank: int,
) -> int:
    """Calculate parameters after low-rank factorization.

    For a matrix of shape (m, n), SVD produces U(m, r) and V(r, n).

    Args:
        original_shape: Original matrix shape (m, n).
        rank: Target rank for factorization.

    Returns:
        Number of parameters after factorization.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_low_rank_params((768, 768), 64)
        98304
        >>> calculate_low_rank_params((1024, 512), 32)
        49152

        >>> calculate_low_rank_params((768, 768), 0)
        Traceback (most recent call last):
            ...
        ValueError: rank must be positive, got 0
    """
    if len(original_shape) != 2:
        msg = f"original_shape must be 2D, got {len(original_shape)}D"
        raise ValueError(msg)
    if rank <= 0:
        msg = f"rank must be positive, got {rank}"
        raise ValueError(msg)

    m, n = original_shape
    if rank > min(m, n):
        msg = f"rank ({rank}) cannot exceed min dimension ({min(m, n)})"
        raise ValueError(msg)

    return m * rank + rank * n


def calculate_weight_sharing_bits(
    original_params: int,
    num_clusters: int,
    bits_per_weight: int,
) -> int:
    """Calculate total bits after weight sharing.

    Args:
        original_params: Original parameter count.
        num_clusters: Number of weight clusters.
        bits_per_weight: Bits for cluster indices.

    Returns:
        Total bits for compressed representation.

    Examples:
        >>> calculate_weight_sharing_bits(100_000, 256, 8)
        808192
        >>> calculate_weight_sharing_bits(100_000, 16, 4)
        400512
    """
    index_bits = original_params * bits_per_weight
    centroid_bits = num_clusters * 32
    return index_bits + centroid_bits


def get_recommended_compression_config(task: str) -> CompressionConfig:
    """Get recommended compression configuration for a task.

    Args:
        task: Task type (inference, mobile, edge, research).

    Returns:
        Recommended CompressionConfig for the task.

    Raises:
        ValueError: If task type is unknown.

    Examples:
        >>> config = get_recommended_compression_config("inference")
        >>> config.enable_fusion
        True

        >>> config = get_recommended_compression_config("mobile")
        >>> config.target_size_reduction
        0.25

        >>> get_recommended_compression_config("unknown")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: task must be one of ...
    """
    valid_tasks = frozenset({"inference", "mobile", "edge", "research"})

    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    if task == "inference":
        return create_compression_config(
            enable_pruning=True,
            enable_fusion=True,
            enable_weight_sharing=False,
            enable_low_rank=False,
            target_speedup=2.0,
            target_size_reduction=0.5,
        )
    elif task == "mobile":
        return create_compression_config(
            enable_pruning=True,
            enable_fusion=True,
            enable_weight_sharing=True,
            enable_low_rank=True,
            target_speedup=4.0,
            target_size_reduction=0.25,
        )
    elif task == "edge":
        return create_compression_config(
            enable_pruning=True,
            enable_fusion=True,
            enable_weight_sharing=True,
            enable_low_rank=True,
            target_speedup=8.0,
            target_size_reduction=0.1,
        )
    else:  # research
        return create_compression_config(
            enable_pruning=True,
            enable_fusion=False,
            enable_weight_sharing=False,
            enable_low_rank=False,
            target_speedup=1.5,
            target_size_reduction=0.7,
        )


def format_compression_stats(stats: CompressionStats) -> str:
    """Format compression statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_compression_stats(100_000_000, 50_000_000)
        >>> formatted = format_compression_stats(stats)
        >>> "Compression Ratio: 2.00x" in formatted
        True
        >>> "Sparsity: 50.0%" in formatted
        True
    """
    return (
        f"Compression Stats:\n"
        f"  Original Params: {stats.original_params:,}\n"
        f"  Compressed Params: {stats.compressed_params:,}\n"
        f"  Sparsity: {stats.sparsity * 100:.1f}%\n"
        f"  Compression Ratio: {stats.compression_ratio:.2f}x\n"
        f"  Est. Speedup: {stats.estimated_speedup:.2f}x\n"
        f"  Original Size: {stats.original_size_mb:.1f} MB\n"
        f"  Compressed Size: {stats.compressed_size_mb:.1f} MB"
    )


def calculate_flops_reduction(
    original_flops: int,
    sparsity: float,
    structured: bool = False,
) -> int:
    """Calculate FLOPs after compression.

    Args:
        original_flops: Original FLOPs count.
        sparsity: Sparsity level achieved.
        structured: Whether pruning is structured.

    Returns:
        Estimated FLOPs after compression.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_flops_reduction(1_000_000, 0.5)
        500000
        >>> calculate_flops_reduction(1_000_000, 0.5, structured=True)
        500000

        >>> calculate_flops_reduction(0, 0.5)
        Traceback (most recent call last):
            ...
        ValueError: original_flops must be positive, got 0
    """
    if original_flops <= 0:
        msg = f"original_flops must be positive, got {original_flops}"
        raise ValueError(msg)
    if not 0 <= sparsity <= 1:
        msg = f"sparsity must be between 0 and 1, got {sparsity}"
        raise ValueError(msg)

    base_reduction = 1 - sparsity
    reduction_factor = base_reduction if structured else max(base_reduction, 0.1)

    return int(original_flops * reduction_factor)
