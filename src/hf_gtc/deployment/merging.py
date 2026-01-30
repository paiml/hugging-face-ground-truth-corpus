"""Model merging utilities.

This module provides functions for merging multiple models using
techniques like SLERP, TIES, DARE, and linear interpolation.

Examples:
    >>> from hf_gtc.deployment.merging import create_merge_config
    >>> config = create_merge_config(method="slerp", t=0.5)
    >>> config.method
    <MergeMethod.SLERP: 'slerp'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class MergeMethod(Enum):
    """Supported model merging methods.

    Attributes:
        LINEAR: Linear interpolation between models.
        SLERP: Spherical linear interpolation.
        TIES: TIES merging (trim, elect, merge).
        DARE: Drop And REscale merging.
        TASK_ARITHMETIC: Task arithmetic merging.
        PASSTHROUGH: Pass through layers without merging.

    Examples:
        >>> MergeMethod.SLERP.value
        'slerp'
        >>> MergeMethod.TIES.value
        'ties'
    """

    LINEAR = "linear"
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"
    TASK_ARITHMETIC = "task_arithmetic"
    PASSTHROUGH = "passthrough"


VALID_MERGE_METHODS = frozenset(m.value for m in MergeMethod)

# Density options for TIES/DARE
DensityType = Literal["low", "medium", "high"]
DENSITY_VALUES: dict[DensityType, float] = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
}


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Configuration for model merging.

    Attributes:
        method: Merging method to use.
        t: Interpolation parameter (0.0-1.0).
        density: Density for TIES/DARE methods.
        normalize: Whether to normalize weights.

    Examples:
        >>> config = MergeConfig(
        ...     method=MergeMethod.SLERP,
        ...     t=0.5,
        ...     density=0.5,
        ...     normalize=True,
        ... )
        >>> config.t
        0.5
    """

    method: MergeMethod
    t: float
    density: float
    normalize: bool


@dataclass(frozen=True, slots=True)
class ModelSlice:
    """Configuration for a model slice in layer-based merging.

    Attributes:
        model_id: Model identifier.
        start_layer: Starting layer index.
        end_layer: Ending layer index (exclusive).
        weight: Weight for this slice.

    Examples:
        >>> slice_cfg = ModelSlice(
        ...     model_id="model_a",
        ...     start_layer=0,
        ...     end_layer=16,
        ...     weight=1.0,
        ... )
        >>> slice_cfg.start_layer
        0
    """

    model_id: str
    start_layer: int
    end_layer: int
    weight: float


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result from a merge operation.

    Attributes:
        num_parameters: Number of parameters in merged model.
        method_used: Merge method that was used.
        models_merged: Number of models merged.
        metadata: Additional merge metadata.

    Examples:
        >>> result = MergeResult(
        ...     num_parameters=7_000_000_000,
        ...     method_used=MergeMethod.SLERP,
        ...     models_merged=2,
        ...     metadata={"t": "0.5"},
        ... )
        >>> result.models_merged
        2
    """

    num_parameters: int
    method_used: MergeMethod
    models_merged: int
    metadata: dict[str, str]


def validate_merge_config(config: MergeConfig) -> None:
    """Validate merge configuration.

    Args:
        config: Merge configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = MergeConfig(
        ...     method=MergeMethod.SLERP,
        ...     t=0.5,
        ...     density=0.5,
        ...     normalize=True,
        ... )
        >>> validate_merge_config(config)  # No error

        >>> bad_config = MergeConfig(
        ...     method=MergeMethod.SLERP,
        ...     t=1.5,
        ...     density=0.5,
        ...     normalize=True,
        ... )
        >>> validate_merge_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: t must be between 0.0 and 1.0
    """
    if not 0.0 <= config.t <= 1.0:
        msg = f"t must be between 0.0 and 1.0, got {config.t}"
        raise ValueError(msg)

    if not 0.0 <= config.density <= 1.0:
        msg = f"density must be between 0.0 and 1.0, got {config.density}"
        raise ValueError(msg)


def validate_model_slice(slice_cfg: ModelSlice) -> None:
    """Validate model slice configuration.

    Args:
        slice_cfg: Model slice to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> slice_cfg = ModelSlice("model_a", 0, 16, 1.0)
        >>> validate_model_slice(slice_cfg)  # No error

        >>> bad_slice = ModelSlice("model_a", 16, 8, 1.0)
        >>> validate_model_slice(bad_slice)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: start_layer must be less than end_layer
    """
    if not slice_cfg.model_id:
        msg = "model_id cannot be empty"
        raise ValueError(msg)

    if slice_cfg.start_layer < 0:
        msg = f"start_layer cannot be negative, got {slice_cfg.start_layer}"
        raise ValueError(msg)

    if slice_cfg.end_layer <= slice_cfg.start_layer:
        msg = (
            f"start_layer ({slice_cfg.start_layer}) must be less than "
            f"end_layer ({slice_cfg.end_layer})"
        )
        raise ValueError(msg)

    if slice_cfg.weight < 0:
        msg = f"weight cannot be negative, got {slice_cfg.weight}"
        raise ValueError(msg)


def create_merge_config(
    method: str = "slerp",
    t: float = 0.5,
    density: float = 0.5,
    normalize: bool = True,
) -> MergeConfig:
    """Create a merge configuration.

    Args:
        method: Merge method. Defaults to "slerp".
        t: Interpolation parameter. Defaults to 0.5.
        density: Density for TIES/DARE. Defaults to 0.5.
        normalize: Whether to normalize. Defaults to True.

    Returns:
        MergeConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_merge_config(method="slerp", t=0.5)
        >>> config.method
        <MergeMethod.SLERP: 'slerp'>

        >>> config = create_merge_config(t=0.7)
        >>> config.t
        0.7

        >>> create_merge_config(t=1.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: t must be between 0.0 and 1.0
    """
    if method not in VALID_MERGE_METHODS:
        msg = f"method must be one of {VALID_MERGE_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = MergeConfig(
        method=MergeMethod(method),
        t=t,
        density=density,
        normalize=normalize,
    )
    validate_merge_config(config)
    return config


def create_model_slice(
    model_id: str,
    start_layer: int,
    end_layer: int,
    weight: float = 1.0,
) -> ModelSlice:
    """Create a model slice configuration.

    Args:
        model_id: Model identifier.
        start_layer: Starting layer index.
        end_layer: Ending layer index (exclusive).
        weight: Weight for this slice. Defaults to 1.0.

    Returns:
        ModelSlice with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> slice_cfg = create_model_slice("model_a", 0, 16)
        >>> slice_cfg.start_layer
        0

        >>> create_model_slice("", 0, 16)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_id cannot be empty
    """
    slice_cfg = ModelSlice(
        model_id=model_id,
        start_layer=start_layer,
        end_layer=end_layer,
        weight=weight,
    )
    validate_model_slice(slice_cfg)
    return slice_cfg


def list_merge_methods() -> list[str]:
    """List supported merge methods.

    Returns:
        Sorted list of merge method names.

    Examples:
        >>> methods = list_merge_methods()
        >>> "slerp" in methods
        True
        >>> "ties" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_MERGE_METHODS)


def linear_interpolate(
    weight_a: float,
    weight_b: float,
    t: float,
) -> float:
    """Perform linear interpolation between two values.

    Args:
        weight_a: First value.
        weight_b: Second value.
        t: Interpolation parameter (0.0 = a, 1.0 = b).

    Returns:
        Interpolated value.

    Raises:
        ValueError: If t is out of range.

    Examples:
        >>> linear_interpolate(0.0, 1.0, 0.5)
        0.5

        >>> linear_interpolate(0.0, 10.0, 0.3)
        3.0

        >>> linear_interpolate(5.0, 5.0, 0.7)
        5.0
    """
    if not 0.0 <= t <= 1.0:
        msg = f"t must be between 0.0 and 1.0, got {t}"
        raise ValueError(msg)

    return weight_a * (1 - t) + weight_b * t


def slerp(
    v0: tuple[float, ...],
    v1: tuple[float, ...],
    t: float,
) -> tuple[float, ...]:
    """Perform spherical linear interpolation.

    Args:
        v0: First vector.
        v1: Second vector.
        t: Interpolation parameter (0.0 = v0, 1.0 = v1).

    Returns:
        Interpolated vector.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> v0 = (1.0, 0.0)
        >>> v1 = (0.0, 1.0)
        >>> result = slerp(v0, v1, 0.5)
        >>> round(result[0], 4)
        0.7071
        >>> round(result[1], 4)
        0.7071

        >>> slerp((), (), 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
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

    # Compute dot product
    dot = sum(a * b for a, b in zip(v0, v1, strict=True))

    # Clamp dot product to valid range
    dot = max(-1.0, min(1.0, dot))

    # Compute angle
    theta = math.acos(dot) * t

    # Compute orthogonal vector
    scale0 = math.cos(theta)
    scale1 = math.sin(theta)

    # Handle case where vectors are nearly parallel
    norm_v1 = sum(x * x for x in v1) ** 0.5
    if norm_v1 == 0:
        return v0

    # Compute result
    result = []
    for a, b in zip(v0, v1, strict=True):
        val = scale0 * a + scale1 * (b - dot * a) / max(norm_v1, 1e-10)
        result.append(val)

    return tuple(result)


def get_density_value(level: DensityType) -> float:
    """Get density value for a named level.

    Args:
        level: Density level name.

    Returns:
        Corresponding density value.

    Examples:
        >>> get_density_value("low")
        0.3
        >>> get_density_value("medium")
        0.5
        >>> get_density_value("high")
        0.7
    """
    return DENSITY_VALUES[level]


def calculate_merged_parameter_count(
    model_params: tuple[int, ...],
    method: MergeMethod,
) -> int:
    """Calculate parameter count of merged model.

    Args:
        model_params: Parameter counts of input models.
        method: Merge method used.

    Returns:
        Estimated parameter count of merged model.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> calculate_merged_parameter_count((7_000_000, 7_000_000), MergeMethod.SLERP)
        7000000

        >>> calculate_merged_parameter_count(
        ...     (7_000_000, 13_000_000), MergeMethod.PASSTHROUGH
        ... )
        20000000

        >>> calculate_merged_parameter_count((), MergeMethod.SLERP)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params cannot be empty
    """
    if not model_params:
        msg = "model_params cannot be empty"
        raise ValueError(msg)

    for params in model_params:
        if params <= 0:
            msg = f"parameter counts must be positive, got {params}"
            raise ValueError(msg)

    if method == MergeMethod.PASSTHROUGH:
        # Passthrough combines all layers
        return sum(model_params)

    # Other methods produce a model of similar size to the largest input
    return max(model_params)


def get_recommended_method(
    num_models: int,
    same_architecture: bool = True,
) -> MergeMethod:
    """Get recommended merge method based on inputs.

    Args:
        num_models: Number of models to merge.
        same_architecture: Whether models have same architecture.

    Returns:
        Recommended merge method.

    Raises:
        ValueError: If num_models is invalid.

    Examples:
        >>> get_recommended_method(2)
        <MergeMethod.SLERP: 'slerp'>

        >>> get_recommended_method(3)
        <MergeMethod.TIES: 'ties'>

        >>> get_recommended_method(2, same_architecture=False)
        <MergeMethod.PASSTHROUGH: 'passthrough'>
    """
    if num_models <= 0:
        msg = f"num_models must be positive, got {num_models}"
        raise ValueError(msg)

    if not same_architecture:
        return MergeMethod.PASSTHROUGH

    if num_models == 2:
        return MergeMethod.SLERP

    return MergeMethod.TIES


def estimate_merge_time(
    total_params: int,
    method: MergeMethod,
) -> str:
    """Estimate merge time category.

    Args:
        total_params: Total parameters across all models.
        method: Merge method to use.

    Returns:
        Time category ("fast", "moderate", "slow").

    Examples:
        >>> estimate_merge_time(1_000_000, MergeMethod.LINEAR)
        'fast'

        >>> estimate_merge_time(7_000_000_000, MergeMethod.TIES)
        'moderate'

        >>> estimate_merge_time(70_000_000_000, MergeMethod.DARE)
        'slow'
    """
    if total_params <= 0:
        msg = f"total_params must be positive, got {total_params}"
        raise ValueError(msg)

    # Simple methods are faster
    fast_methods = {MergeMethod.LINEAR, MergeMethod.PASSTHROUGH}

    if method in fast_methods:
        if total_params < 1_000_000_000:
            return "fast"
        if total_params < 20_000_000_000:
            return "moderate"
        return "slow"

    # Complex methods (SLERP, TIES, DARE)
    if total_params < 100_000_000:
        return "fast"
    if total_params < 10_000_000_000:
        return "moderate"
    return "slow"
