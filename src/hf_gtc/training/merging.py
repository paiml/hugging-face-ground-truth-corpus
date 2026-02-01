"""Model merging training utilities.

This module provides functions for configuring model merging techniques
including TIES, DARE, SLERP, and Task Arithmetic. It focuses on the
training aspects of model merging such as weight interpolation, conflict
resolution, and task vector computation.

Examples:
    >>> from hf_gtc.training.merging import (
    ...     create_merge_config,
    ...     MergeMethod,
    ... )
    >>> config = create_merge_config(method="slerp", weights=(0.5, 0.5))
    >>> config.method
    <MergeMethod.SLERP: 'slerp'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class MergeMethod(Enum):
    """Supported model merging methods.

    Attributes:
        TIES: TIES merging (trim, elect, merge) for handling parameter conflicts.
        DARE: Drop And REscale merging for sparsity-aware merging.
        SLERP: Spherical linear interpolation for smooth merging.
        TASK_ARITHMETIC: Task arithmetic using task vectors.
        LINEAR: Simple linear interpolation between models.
        STOCK: Stock/passthrough merging without modification.

    Examples:
        >>> MergeMethod.TIES.value
        'ties'
        >>> MergeMethod.DARE.value
        'dare'
        >>> MergeMethod.SLERP.value
        'slerp'
        >>> MergeMethod.TASK_ARITHMETIC.value
        'task_arithmetic'
    """

    TIES = "ties"
    DARE = "dare"
    SLERP = "slerp"
    TASK_ARITHMETIC = "task_arithmetic"
    LINEAR = "linear"
    STOCK = "stock"


class SparsificationMethod(Enum):
    """Sparsification methods for TIES and DARE merging.

    Attributes:
        MAGNITUDE: Prune parameters with smallest magnitude.
        RANDOM: Randomly prune parameters.
        TOPK: Keep only top-k magnitude parameters.

    Examples:
        >>> SparsificationMethod.MAGNITUDE.value
        'magnitude'
        >>> SparsificationMethod.RANDOM.value
        'random'
        >>> SparsificationMethod.TOPK.value
        'topk'
    """

    MAGNITUDE = "magnitude"
    RANDOM = "random"
    TOPK = "topk"


class ConflictResolution(Enum):
    """Conflict resolution strategies for parameter merging.

    Attributes:
        SUM: Sum conflicting parameters.
        MEAN: Average conflicting parameters.
        MAX: Take maximum magnitude parameter.
        RANDOM: Randomly select from conflicting parameters.

    Examples:
        >>> ConflictResolution.SUM.value
        'sum'
        >>> ConflictResolution.MEAN.value
        'mean'
        >>> ConflictResolution.MAX.value
        'max'
        >>> ConflictResolution.RANDOM.value
        'random'
    """

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    RANDOM = "random"


VALID_MERGE_METHODS = frozenset(m.value for m in MergeMethod)
VALID_SPARSIFICATION_METHODS = frozenset(s.value for s in SparsificationMethod)
VALID_CONFLICT_RESOLUTIONS = frozenset(cr.value for cr in ConflictResolution)


@dataclass(frozen=True, slots=True)
class TIESConfig:
    """Configuration for TIES (Trim, Elect, Merge) merging.

    TIES resolves parameter interference by trimming low-magnitude deltas,
    electing signs via majority voting, and merging.

    Attributes:
        density: Fraction of parameters to retain (0.0-1.0).
        normalize: Whether to normalize merged parameters.
        conflict_resolution: Strategy for resolving sign conflicts.

    Examples:
        >>> config = TIESConfig(
        ...     density=0.5,
        ...     normalize=True,
        ...     conflict_resolution=ConflictResolution.SUM,
        ... )
        >>> config.density
        0.5
        >>> config.conflict_resolution
        <ConflictResolution.SUM: 'sum'>
    """

    density: float
    normalize: bool
    conflict_resolution: ConflictResolution


@dataclass(frozen=True, slots=True)
class DAREConfig:
    """Configuration for DARE (Drop And REscale) merging.

    DARE randomly drops delta parameters and rescales remaining ones
    to compensate, improving generalization.

    Attributes:
        drop_rate: Fraction of parameters to drop (0.0-1.0).
        rescale: Whether to rescale remaining parameters.

    Examples:
        >>> config = DAREConfig(
        ...     drop_rate=0.1,
        ...     rescale=True,
        ... )
        >>> config.drop_rate
        0.1
        >>> config.rescale
        True
    """

    drop_rate: float
    rescale: bool


@dataclass(frozen=True, slots=True)
class SLERPConfig:
    """Configuration for SLERP (Spherical Linear Interpolation) merging.

    SLERP interpolates model parameters along the shortest path on a
    hypersphere, preserving angular relationships.

    Attributes:
        interpolation_factor: Interpolation factor (0.0-1.0).

    Examples:
        >>> config = SLERPConfig(interpolation_factor=0.5)
        >>> config.interpolation_factor
        0.5
    """

    interpolation_factor: float


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Main configuration for model merging.

    Attributes:
        method: Merging method to use.
        ties_config: Configuration for TIES merging (if applicable).
        dare_config: Configuration for DARE merging (if applicable).
        slerp_config: Configuration for SLERP merging (if applicable).
        weights: Weights for each model being merged.

    Examples:
        >>> ties = TIESConfig(
        ...     density=0.5,
        ...     normalize=True,
        ...     conflict_resolution=ConflictResolution.SUM,
        ... )
        >>> config = MergeConfig(
        ...     method=MergeMethod.TIES,
        ...     ties_config=ties,
        ...     dare_config=None,
        ...     slerp_config=None,
        ...     weights=(0.5, 0.5),
        ... )
        >>> config.method
        <MergeMethod.TIES: 'ties'>
        >>> config.weights
        (0.5, 0.5)
    """

    method: MergeMethod
    ties_config: TIESConfig | None
    dare_config: DAREConfig | None
    slerp_config: SLERPConfig | None
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MergeStats:
    """Statistics from model merging operations.

    Attributes:
        num_models: Number of models merged.
        total_parameters: Total parameter count.
        retained_parameters: Parameters retained after sparsification.
        conflict_rate: Rate of sign conflicts encountered.
        method_used: Merge method that was used.

    Examples:
        >>> stats = MergeStats(
        ...     num_models=3,
        ...     total_parameters=7_000_000_000,
        ...     retained_parameters=3_500_000_000,
        ...     conflict_rate=0.15,
        ...     method_used=MergeMethod.TIES,
        ... )
        >>> stats.num_models
        3
        >>> stats.conflict_rate
        0.15
    """

    num_models: int
    total_parameters: int
    retained_parameters: int
    conflict_rate: float
    method_used: MergeMethod


def validate_ties_config(config: TIESConfig) -> None:
    """Validate TIES configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = TIESConfig(
        ...     density=0.5,
        ...     normalize=True,
        ...     conflict_resolution=ConflictResolution.SUM,
        ... )
        >>> validate_ties_config(config)

        >>> bad_config = TIESConfig(
        ...     density=-0.1,
        ...     normalize=True,
        ...     conflict_resolution=ConflictResolution.SUM,
        ... )
        >>> validate_ties_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: density must be between 0.0 and 1.0, got -0.1
    """
    if not 0.0 <= config.density <= 1.0:
        msg = f"density must be between 0.0 and 1.0, got {config.density}"
        raise ValueError(msg)


def validate_dare_config(config: DAREConfig) -> None:
    """Validate DARE configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = DAREConfig(drop_rate=0.1, rescale=True)
        >>> validate_dare_config(config)

        >>> bad_config = DAREConfig(drop_rate=1.5, rescale=True)
        >>> validate_dare_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: drop_rate must be between 0.0 and 1.0, got 1.5
    """
    if not 0.0 <= config.drop_rate <= 1.0:
        msg = f"drop_rate must be between 0.0 and 1.0, got {config.drop_rate}"
        raise ValueError(msg)


def validate_slerp_config(config: SLERPConfig) -> None:
    """Validate SLERP configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> config = SLERPConfig(interpolation_factor=0.5)
        >>> validate_slerp_config(config)

        >>> bad_config = SLERPConfig(interpolation_factor=-0.5)
        >>> validate_slerp_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: interpolation_factor must be between 0.0 and 1.0, got -0.5
    """
    if not 0.0 <= config.interpolation_factor <= 1.0:
        msg = (
            f"interpolation_factor must be between 0.0 and 1.0, "
            f"got {config.interpolation_factor}"
        )
        raise ValueError(msg)


def _validate_merge_method(config: MergeConfig) -> None:
    """Validate method-specific merge configuration."""
    method_validators = {
        MergeMethod.TIES: ("ties_config", "TIES", validate_ties_config),
        MergeMethod.DARE: ("dare_config", "DARE", validate_dare_config),
        MergeMethod.SLERP: ("slerp_config", "SLERP", validate_slerp_config),
    }
    entry = method_validators.get(config.method)
    if entry is None:
        return
    attr_name, method_label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} required for {method_label} method"
        raise ValueError(msg)
    validator(sub_config)
    if config.method == MergeMethod.SLERP and len(config.weights) != 2:
        msg = f"SLERP requires exactly 2 models, got {len(config.weights)}"
        raise ValueError(msg)


def validate_merge_config(config: MergeConfig) -> None:
    """Validate merge configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any configuration value is invalid.

    Examples:
        >>> ties = TIESConfig(
        ...     density=0.5,
        ...     normalize=True,
        ...     conflict_resolution=ConflictResolution.SUM,
        ... )
        >>> config = MergeConfig(
        ...     method=MergeMethod.TIES,
        ...     ties_config=ties,
        ...     dare_config=None,
        ...     slerp_config=None,
        ...     weights=(0.5, 0.5),
        ... )
        >>> validate_merge_config(config)

        >>> bad_config = MergeConfig(
        ...     method=MergeMethod.TIES,
        ...     ties_config=None,
        ...     dare_config=None,
        ...     slerp_config=None,
        ...     weights=(0.5, 0.5),
        ... )
        >>> validate_merge_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: ties_config required for TIES method
    """
    if not config.weights:
        msg = "weights cannot be empty"
        raise ValueError(msg)

    for w in config.weights:
        if w < 0:
            msg = f"weights must be non-negative, got {w}"
            raise ValueError(msg)

    # Validate method-specific configs via dispatch
    _validate_merge_method(config)


def create_ties_config(
    density: float = 0.5,
    normalize: bool = True,
    conflict_resolution: str | ConflictResolution = ConflictResolution.SUM,
) -> TIESConfig:
    """Create a TIES configuration with validation.

    Args:
        density: Fraction of parameters to retain.
        normalize: Whether to normalize merged parameters.
        conflict_resolution: Strategy for resolving conflicts.

    Returns:
        Validated TIESConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_ties_config()
        >>> config.density
        0.5
        >>> config.normalize
        True

        >>> config = create_ties_config(density=0.7, conflict_resolution="mean")
        >>> config.density
        0.7
        >>> config.conflict_resolution
        <ConflictResolution.MEAN: 'mean'>

        >>> create_ties_config(density=1.5)
        Traceback (most recent call last):
            ...
        ValueError: density must be between 0.0 and 1.0, got 1.5
    """
    if isinstance(conflict_resolution, str):
        conflict_resolution = get_conflict_resolution(conflict_resolution)

    config = TIESConfig(
        density=density,
        normalize=normalize,
        conflict_resolution=conflict_resolution,
    )
    validate_ties_config(config)
    return config


def create_dare_config(
    drop_rate: float = 0.1,
    rescale: bool = True,
) -> DAREConfig:
    """Create a DARE configuration with validation.

    Args:
        drop_rate: Fraction of parameters to drop.
        rescale: Whether to rescale remaining parameters.

    Returns:
        Validated DAREConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_dare_config()
        >>> config.drop_rate
        0.1
        >>> config.rescale
        True

        >>> config = create_dare_config(drop_rate=0.2, rescale=False)
        >>> config.drop_rate
        0.2
        >>> config.rescale
        False

        >>> create_dare_config(drop_rate=-0.1)
        Traceback (most recent call last):
            ...
        ValueError: drop_rate must be between 0.0 and 1.0, got -0.1
    """
    config = DAREConfig(drop_rate=drop_rate, rescale=rescale)
    validate_dare_config(config)
    return config


def create_slerp_config(
    interpolation_factor: float = 0.5,
) -> SLERPConfig:
    """Create a SLERP configuration with validation.

    Args:
        interpolation_factor: Interpolation factor (0.0-1.0).

    Returns:
        Validated SLERPConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_slerp_config()
        >>> config.interpolation_factor
        0.5

        >>> config = create_slerp_config(interpolation_factor=0.7)
        >>> config.interpolation_factor
        0.7

        >>> create_slerp_config(interpolation_factor=1.5)
        Traceback (most recent call last):
            ...
        ValueError: interpolation_factor must be between 0.0 and 1.0, got 1.5
    """
    config = SLERPConfig(interpolation_factor=interpolation_factor)
    validate_slerp_config(config)
    return config


def create_merge_config(
    method: str | MergeMethod = MergeMethod.LINEAR,
    ties_config: TIESConfig | None = None,
    dare_config: DAREConfig | None = None,
    slerp_config: SLERPConfig | None = None,
    weights: tuple[float, ...] = (0.5, 0.5),
) -> MergeConfig:
    """Create a merge configuration with validation.

    Args:
        method: Merging method to use.
        ties_config: TIES-specific configuration.
        dare_config: DARE-specific configuration.
        slerp_config: SLERP-specific configuration.
        weights: Weights for each model being merged.

    Returns:
        Validated MergeConfig.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_merge_config(method="linear")
        >>> config.method
        <MergeMethod.LINEAR: 'linear'>
        >>> config.weights
        (0.5, 0.5)

        >>> ties = create_ties_config(density=0.7)
        >>> config = create_merge_config(method="ties", ties_config=ties)
        >>> config.method
        <MergeMethod.TIES: 'ties'>

        >>> create_merge_config(weights=())
        Traceback (most recent call last):
            ...
        ValueError: weights cannot be empty
    """
    if isinstance(method, str):
        method = get_merge_method(method)

    # Auto-create default configs for methods that need them
    if method == MergeMethod.TIES and ties_config is None:
        ties_config = create_ties_config()
    elif method == MergeMethod.DARE and dare_config is None:
        dare_config = create_dare_config()
    elif method == MergeMethod.SLERP and slerp_config is None:
        slerp_config = create_slerp_config()

    config = MergeConfig(
        method=method,
        ties_config=ties_config,
        dare_config=dare_config,
        slerp_config=slerp_config,
        weights=weights,
    )
    validate_merge_config(config)
    return config


def create_merge_stats(
    num_models: int = 2,
    total_parameters: int = 0,
    retained_parameters: int = 0,
    conflict_rate: float = 0.0,
    method_used: MergeMethod = MergeMethod.LINEAR,
) -> MergeStats:
    """Create merge statistics.

    Args:
        num_models: Number of models merged.
        total_parameters: Total parameter count.
        retained_parameters: Parameters retained after sparsification.
        conflict_rate: Rate of sign conflicts encountered.
        method_used: Merge method that was used.

    Returns:
        MergeStats instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_merge_stats()
        >>> stats.num_models
        2
        >>> stats.conflict_rate
        0.0

        >>> stats = create_merge_stats(
        ...     num_models=3,
        ...     total_parameters=7_000_000_000,
        ...     method_used=MergeMethod.TIES,
        ... )
        >>> stats.num_models
        3

        >>> create_merge_stats(num_models=0)
        Traceback (most recent call last):
            ...
        ValueError: num_models must be positive, got 0
    """
    if num_models <= 0:
        msg = f"num_models must be positive, got {num_models}"
        raise ValueError(msg)
    if total_parameters < 0:
        msg = f"total_parameters must be non-negative, got {total_parameters}"
        raise ValueError(msg)
    if retained_parameters < 0:
        msg = f"retained_parameters must be non-negative, got {retained_parameters}"
        raise ValueError(msg)
    if not 0.0 <= conflict_rate <= 1.0:
        msg = f"conflict_rate must be between 0.0 and 1.0, got {conflict_rate}"
        raise ValueError(msg)

    return MergeStats(
        num_models=num_models,
        total_parameters=total_parameters,
        retained_parameters=retained_parameters,
        conflict_rate=conflict_rate,
        method_used=method_used,
    )


def list_merge_methods() -> list[str]:
    """List all available merge methods.

    Returns:
        Sorted list of merge method names.

    Examples:
        >>> methods = list_merge_methods()
        >>> "ties" in methods
        True
        >>> "slerp" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_MERGE_METHODS)


def list_sparsification_methods() -> list[str]:
    """List all available sparsification methods.

    Returns:
        Sorted list of sparsification method names.

    Examples:
        >>> methods = list_sparsification_methods()
        >>> "magnitude" in methods
        True
        >>> "topk" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_SPARSIFICATION_METHODS)


def list_conflict_resolutions() -> list[str]:
    """List all available conflict resolution strategies.

    Returns:
        Sorted list of conflict resolution names.

    Examples:
        >>> resolutions = list_conflict_resolutions()
        >>> "sum" in resolutions
        True
        >>> "mean" in resolutions
        True
        >>> resolutions == sorted(resolutions)
        True
    """
    return sorted(VALID_CONFLICT_RESOLUTIONS)


def get_merge_method(name: str) -> MergeMethod:
    """Get merge method enum from string name.

    Args:
        name: Name of the merge method.

    Returns:
        Corresponding MergeMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_merge_method("ties")
        <MergeMethod.TIES: 'ties'>
        >>> get_merge_method("slerp")
        <MergeMethod.SLERP: 'slerp'>

        >>> get_merge_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: method must be one of ...
    """
    if name not in VALID_MERGE_METHODS:
        msg = f"method must be one of {VALID_MERGE_METHODS}, got '{name}'"
        raise ValueError(msg)
    return MergeMethod(name)


def get_sparsification_method(name: str) -> SparsificationMethod:
    """Get sparsification method enum from string name.

    Args:
        name: Name of the sparsification method.

    Returns:
        Corresponding SparsificationMethod enum.

    Raises:
        ValueError: If method name is invalid.

    Examples:
        >>> get_sparsification_method("magnitude")
        <SparsificationMethod.MAGNITUDE: 'magnitude'>
        >>> get_sparsification_method("topk")
        <SparsificationMethod.TOPK: 'topk'>

        >>> get_sparsification_method("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: sparsification_method must be one of ...
    """
    if name not in VALID_SPARSIFICATION_METHODS:
        msg = (
            f"sparsification_method must be one of {VALID_SPARSIFICATION_METHODS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return SparsificationMethod(name)


def get_conflict_resolution(name: str) -> ConflictResolution:
    """Get conflict resolution enum from string name.

    Args:
        name: Name of the conflict resolution strategy.

    Returns:
        Corresponding ConflictResolution enum.

    Raises:
        ValueError: If resolution name is invalid.

    Examples:
        >>> get_conflict_resolution("sum")
        <ConflictResolution.SUM: 'sum'>
        >>> get_conflict_resolution("mean")
        <ConflictResolution.MEAN: 'mean'>

        >>> get_conflict_resolution("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: conflict_resolution must be one of ...
    """
    if name not in VALID_CONFLICT_RESOLUTIONS:
        msg = (
            f"conflict_resolution must be one of {VALID_CONFLICT_RESOLUTIONS}, "
            f"got '{name}'"
        )
        raise ValueError(msg)
    return ConflictResolution(name)


def calculate_merge_weights(
    model_scores: tuple[float, ...],
    normalize: bool = True,
) -> tuple[float, ...]:
    """Calculate merge weights from model performance scores.

    Higher-performing models get higher weights in the merge.

    Args:
        model_scores: Performance score for each model (higher is better).
        normalize: Whether to normalize weights to sum to 1.0.

    Returns:
        Tuple of merge weights for each model.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> weights = calculate_merge_weights((0.8, 0.9, 0.7))
        >>> round(sum(weights), 10)
        1.0
        >>> weights[1] > weights[0] > weights[2]  # Best model gets highest weight
        True

        >>> weights = calculate_merge_weights((1.0, 1.0), normalize=False)
        >>> weights
        (1.0, 1.0)

        >>> calculate_merge_weights(())
        Traceback (most recent call last):
            ...
        ValueError: model_scores cannot be empty
    """
    if not model_scores:
        msg = "model_scores cannot be empty"
        raise ValueError(msg)

    for score in model_scores:
        if score < 0:
            msg = f"model_scores must be non-negative, got {score}"
            raise ValueError(msg)

    total = sum(model_scores)
    if total == 0:
        # Equal weights if all scores are 0
        equal_weight = 1.0 / len(model_scores) if normalize else 1.0
        return tuple(equal_weight for _ in model_scores)

    if normalize:
        return tuple(score / total for score in model_scores)

    return model_scores


def estimate_merged_performance(
    model_performances: tuple[float, ...],
    weights: tuple[float, ...],
    method: MergeMethod = MergeMethod.LINEAR,
) -> float:
    """Estimate performance of merged model.

    This provides a rough estimate based on weighted combination
    of individual model performances with method-specific adjustments.

    Args:
        model_performances: Performance metrics for each model (0.0-1.0).
        weights: Merge weights for each model.
        method: Merge method being used.

    Returns:
        Estimated performance of merged model (0.0-1.0).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> perf = estimate_merged_performance((0.8, 0.9), (0.5, 0.5))
        >>> 0.8 <= perf <= 0.9
        True

        >>> perf = estimate_merged_performance(
        ...     (0.7, 0.8, 0.9),
        ...     (0.33, 0.33, 0.34),
        ...     method=MergeMethod.TIES,
        ... )
        >>> 0.7 <= perf <= 1.0
        True

        >>> estimate_merged_performance((), ())
        Traceback (most recent call last):
            ...
        ValueError: model_performances cannot be empty
    """
    if not model_performances:
        msg = "model_performances cannot be empty"
        raise ValueError(msg)
    if not weights:
        msg = "weights cannot be empty"
        raise ValueError(msg)
    if len(model_performances) != len(weights):
        msg = "model_performances and weights must have same length"
        raise ValueError(msg)

    for perf in model_performances:
        if not 0.0 <= perf <= 1.0:
            msg = f"performance values must be between 0.0 and 1.0, got {perf}"
            raise ValueError(msg)

    # Base weighted average
    weighted_sum = sum(p * w for p, w in zip(model_performances, weights, strict=True))
    total_weight = sum(weights)

    if total_weight == 0:
        return sum(model_performances) / len(model_performances)

    base_estimate = weighted_sum / total_weight

    # Method-specific adjustments
    if method == MergeMethod.TIES:
        # TIES typically improves upon simple averaging
        bonus = 0.02 * len(model_performances)
        return min(1.0, base_estimate + bonus)

    elif method == MergeMethod.SLERP:
        # SLERP works best for 2 models
        return base_estimate

    elif method == MergeMethod.DARE:
        # DARE may have slight generalization improvement
        return min(1.0, base_estimate + 0.01)

    return base_estimate


def resolve_parameter_conflicts(
    values: tuple[float, ...],
    signs: tuple[int, ...],
    resolution: ConflictResolution = ConflictResolution.SUM,
) -> float:
    """Resolve conflicting parameter values based on resolution strategy.

    Used in TIES merging to handle parameters with different signs
    across models.

    Args:
        values: Parameter values from different models.
        signs: Sign (+1, -1, or 0) of each parameter.
        resolution: Strategy for resolving conflicts.

    Returns:
        Resolved parameter value.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> resolve_parameter_conflicts((0.5, -0.3, 0.2), (1, -1, 1))
        0.4

        >>> resolve_parameter_conflicts(
        ...     (0.5, -0.3, 0.2),
        ...     (1, -1, 1),
        ...     resolution=ConflictResolution.MEAN,
        ... )
        0.13333333333333333

        >>> resolve_parameter_conflicts((), ())
        Traceback (most recent call last):
            ...
        ValueError: values cannot be empty
    """
    if not values:
        msg = "values cannot be empty"
        raise ValueError(msg)
    if not signs:
        msg = "signs cannot be empty"
        raise ValueError(msg)
    if len(values) != len(signs):
        msg = "values and signs must have same length"
        raise ValueError(msg)

    for sign in signs:
        if sign not in (-1, 0, 1):
            msg = f"signs must be -1, 0, or 1, got {sign}"
            raise ValueError(msg)

    if resolution == ConflictResolution.SUM:
        return sum(values)

    elif resolution == ConflictResolution.MEAN:
        return sum(values) / len(values)

    elif resolution == ConflictResolution.MAX:
        max_idx = max(range(len(values)), key=lambda i: abs(values[i]))
        return values[max_idx]

    elif resolution == ConflictResolution.RANDOM:
        # Deterministic "random" based on values for reproducibility
        idx = int(abs(sum(values) * 1000)) % len(values)
        return values[idx]

    return sum(values)


def calculate_task_vector(
    fine_tuned_params: tuple[float, ...],
    base_params: tuple[float, ...],
) -> tuple[float, ...]:
    """Calculate task vector as difference between fine-tuned and base models.

    The task vector represents the "direction" of fine-tuning and can
    be added to other models to transfer task-specific knowledge.

    Args:
        fine_tuned_params: Parameters from fine-tuned model.
        base_params: Parameters from base model.

    Returns:
        Task vector (fine_tuned - base).

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> fine_tuned = (1.5, 2.5, 3.5)
        >>> base = (1.0, 2.0, 3.0)
        >>> task_vec = calculate_task_vector(fine_tuned, base)
        >>> task_vec
        (0.5, 0.5, 0.5)

        >>> calculate_task_vector((), ())
        Traceback (most recent call last):
            ...
        ValueError: parameters cannot be empty

        >>> calculate_task_vector((1.0,), (1.0, 2.0))
        Traceback (most recent call last):
            ...
        ValueError: fine_tuned and base parameters must have same length
    """
    if not fine_tuned_params or not base_params:
        msg = "parameters cannot be empty"
        raise ValueError(msg)

    if len(fine_tuned_params) != len(base_params):
        msg = "fine_tuned and base parameters must have same length"
        raise ValueError(msg)

    return tuple(ft - b for ft, b in zip(fine_tuned_params, base_params, strict=True))


def apply_task_vector(
    base_params: tuple[float, ...],
    task_vector: tuple[float, ...],
    scaling_factor: float = 1.0,
) -> tuple[float, ...]:
    """Apply scaled task vector to base model parameters.

    Args:
        base_params: Parameters from target base model.
        task_vector: Task vector to apply.
        scaling_factor: Scaling factor for task vector.

    Returns:
        New parameters with task vector applied.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> base = (1.0, 2.0, 3.0)
        >>> task_vec = (0.5, 0.5, 0.5)
        >>> result = apply_task_vector(base, task_vec)
        >>> result
        (1.5, 2.5, 3.5)

        >>> result = apply_task_vector(base, task_vec, scaling_factor=0.5)
        >>> result
        (1.25, 2.25, 3.25)

        >>> apply_task_vector((), ())
        Traceback (most recent call last):
            ...
        ValueError: parameters cannot be empty
    """
    if not base_params or not task_vector:
        msg = "parameters cannot be empty"
        raise ValueError(msg)

    if len(base_params) != len(task_vector):
        msg = "base_params and task_vector must have same length"
        raise ValueError(msg)

    return tuple(
        b + scaling_factor * tv for b, tv in zip(base_params, task_vector, strict=True)
    )


def format_merge_stats(stats: MergeStats) -> str:
    """Format merge statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = create_merge_stats(
        ...     num_models=3,
        ...     total_parameters=7_000_000_000,
        ...     retained_parameters=3_500_000_000,
        ...     conflict_rate=0.15,
        ...     method_used=MergeMethod.TIES,
        ... )
        >>> formatted = format_merge_stats(stats)
        >>> "Models Merged: 3" in formatted
        True
        >>> "Conflict Rate: 15.0%" in formatted
        True
    """
    retention_rate = (
        (stats.retained_parameters / stats.total_parameters * 100)
        if stats.total_parameters > 0
        else 0.0
    )

    return (
        f"Merge Stats:\n"
        f"  Method: {stats.method_used.value}\n"
        f"  Models Merged: {stats.num_models}\n"
        f"  Total Parameters: {stats.total_parameters:,}\n"
        f"  Retained Parameters: {stats.retained_parameters:,}\n"
        f"  Retention Rate: {retention_rate:.1f}%\n"
        f"  Conflict Rate: {stats.conflict_rate * 100:.1f}%"
    )


def get_recommended_merge_config(
    num_models: int,
    task_type: str = "general",
) -> MergeConfig:
    """Get recommended merge configuration based on inputs.

    Provides sensible defaults for different merging scenarios.

    Args:
        num_models: Number of models to merge.
        task_type: Type of task (general, classification, generation).

    Returns:
        Recommended MergeConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_merge_config(2)
        >>> config.method
        <MergeMethod.SLERP: 'slerp'>

        >>> config = get_recommended_merge_config(3)
        >>> config.method
        <MergeMethod.TIES: 'ties'>

        >>> config = get_recommended_merge_config(2, task_type="generation")
        >>> config.method
        <MergeMethod.SLERP: 'slerp'>

        >>> get_recommended_merge_config(0)
        Traceback (most recent call last):
            ...
        ValueError: num_models must be at least 2, got 0
    """
    if num_models < 2:
        msg = f"num_models must be at least 2, got {num_models}"
        raise ValueError(msg)

    valid_task_types = frozenset({"general", "classification", "generation"})
    if task_type not in valid_task_types:
        msg = f"task_type must be one of {valid_task_types}, got '{task_type}'"
        raise ValueError(msg)

    # Equal weights for all models
    equal_weight = 1.0 / num_models
    weights = tuple(equal_weight for _ in range(num_models))

    # SLERP is best for 2 models
    if num_models == 2:
        return create_merge_config(
            method=MergeMethod.SLERP,
            slerp_config=create_slerp_config(interpolation_factor=0.5),
            weights=weights,
        )

    # TIES is best for multiple models
    if task_type == "classification":
        density = 0.5
    elif task_type == "generation":
        density = 0.7
    else:
        density = 0.6

    return create_merge_config(
        method=MergeMethod.TIES,
        ties_config=create_ties_config(
            density=density,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        ),
        weights=weights,
    )


def slerp_interpolate(
    v0: tuple[float, ...],
    v1: tuple[float, ...],
    t: float,
) -> tuple[float, ...]:
    """Perform spherical linear interpolation between two vectors.

    SLERP interpolates along the shortest arc on a hypersphere,
    preserving angular relationships between parameters.

    Args:
        v0: First parameter vector.
        v1: Second parameter vector.
        t: Interpolation factor (0.0 = v0, 1.0 = v1).

    Returns:
        Interpolated parameter vector.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> v0 = (1.0, 0.0)
        >>> v1 = (0.0, 1.0)
        >>> result = slerp_interpolate(v0, v1, 0.5)
        >>> round(result[0], 4)
        0.7071
        >>> round(result[1], 4)
        0.7071

        >>> slerp_interpolate((1.0, 0.0), (0.0, 1.0), 0.0)
        (1.0, 0.0)

        >>> slerp_interpolate((), (), 0.5)
        Traceback (most recent call last):
            ...
        ValueError: vectors cannot be empty
    """
    if not v0 or not v1:
        msg = "vectors cannot be empty"
        raise ValueError(msg)

    if len(v0) != len(v1):
        msg = f"vectors must have same length: {len(v0)} != {len(v1)}"
        raise ValueError(msg)

    if not 0.0 <= t <= 1.0:
        msg = f"t must be between 0.0 and 1.0, got {t}"
        raise ValueError(msg)

    # Handle edge cases
    if t == 0.0:
        return v0
    if t == 1.0:
        return v1

    # Compute norms
    norm0 = math.sqrt(sum(x * x for x in v0))
    norm1 = math.sqrt(sum(x * x for x in v1))

    # Handle zero vectors
    if norm0 == 0 or norm1 == 0:
        # Fall back to linear interpolation
        return tuple((1 - t) * a + t * b for a, b in zip(v0, v1, strict=True))

    # Normalize vectors
    v0_norm = tuple(x / norm0 for x in v0)
    v1_norm = tuple(x / norm1 for x in v1)

    # Compute dot product (cosine of angle)
    dot = sum(a * b for a, b in zip(v0_norm, v1_norm, strict=True))

    # Clamp to valid range for acos
    dot = max(-1.0, min(1.0, dot))

    # Compute angle
    omega = math.acos(dot)

    # Handle nearly parallel vectors
    if abs(omega) < 1e-10:
        return tuple((1 - t) * a + t * b for a, b in zip(v0, v1, strict=True))

    # SLERP formula
    sin_omega = math.sin(omega)
    s0 = math.sin((1 - t) * omega) / sin_omega
    s1 = math.sin(t * omega) / sin_omega

    # Interpolate both direction and magnitude
    interp_norm = (1 - t) * norm0 + t * norm1

    result = tuple(
        (s0 * a + s1 * b) * interp_norm for a, b in zip(v0_norm, v1_norm, strict=True)
    )

    return result


def linear_interpolate(
    v0: tuple[float, ...],
    v1: tuple[float, ...],
    t: float,
) -> tuple[float, ...]:
    """Perform linear interpolation between two vectors.

    Args:
        v0: First parameter vector.
        v1: Second parameter vector.
        t: Interpolation factor (0.0 = v0, 1.0 = v1).

    Returns:
        Interpolated parameter vector.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> v0 = (0.0, 0.0)
        >>> v1 = (1.0, 2.0)
        >>> linear_interpolate(v0, v1, 0.5)
        (0.5, 1.0)

        >>> linear_interpolate((1.0,), (3.0,), 0.25)
        (1.5,)

        >>> linear_interpolate((), (), 0.5)
        Traceback (most recent call last):
            ...
        ValueError: vectors cannot be empty
    """
    if not v0 or not v1:
        msg = "vectors cannot be empty"
        raise ValueError(msg)

    if len(v0) != len(v1):
        msg = f"vectors must have same length: {len(v0)} != {len(v1)}"
        raise ValueError(msg)

    if not 0.0 <= t <= 1.0:
        msg = f"t must be between 0.0 and 1.0, got {t}"
        raise ValueError(msg)

    return tuple((1 - t) * a + t * b for a, b in zip(v0, v1, strict=True))
