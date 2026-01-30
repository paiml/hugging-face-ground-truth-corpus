"""Mixed precision training utilities.

This module provides functions for configuring mixed precision training
including FP16, BF16, and FP8 precision modes with automatic loss scaling.

Examples:
    >>> from hf_gtc.training.mixed_precision import create_precision_config
    >>> config = create_precision_config(dtype="fp16")
    >>> config.dtype
    <PrecisionType.FP16: 'fp16'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PrecisionType(Enum):
    """Precision data types for mixed precision training.

    Attributes:
        FP32: Full precision 32-bit floating point.
        FP16: Half precision 16-bit floating point.
        BF16: Brain floating point 16-bit.
        FP8_E4M3: FP8 with 4 exponent, 3 mantissa bits.
        FP8_E5M2: FP8 with 5 exponent, 2 mantissa bits.

    Examples:
        >>> PrecisionType.FP16.value
        'fp16'
        >>> PrecisionType.BF16.value
        'bf16'
        >>> PrecisionType.FP8_E4M3.value
        'fp8_e4m3'
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


VALID_PRECISION_TYPES = frozenset(p.value for p in PrecisionType)


class ScalingStrategy(Enum):
    """Loss scaling strategies for mixed precision training.

    Attributes:
        STATIC: Fixed loss scale throughout training.
        DYNAMIC: Automatically adjusted loss scale.
        LOSS_SCALE: Manual loss scale with backoff.

    Examples:
        >>> ScalingStrategy.DYNAMIC.value
        'dynamic'
        >>> ScalingStrategy.STATIC.value
        'static'
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    LOSS_SCALE = "loss_scale"


VALID_SCALING_STRATEGIES = frozenset(s.value for s in ScalingStrategy)


class CastingPolicy(Enum):
    """Tensor casting policies for mixed precision.

    Attributes:
        ALL: Cast all operations to lower precision.
        COMPUTE_ONLY: Cast only compute operations.
        GRADIENTS: Cast only gradients to lower precision.

    Examples:
        >>> CastingPolicy.ALL.value
        'all'
        >>> CastingPolicy.COMPUTE_ONLY.value
        'compute_only'
    """

    ALL = "all"
    COMPUTE_ONLY = "compute_only"
    GRADIENTS = "gradients"


VALID_CASTING_POLICIES = frozenset(c.value for c in CastingPolicy)


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Configuration for precision settings.

    Attributes:
        dtype: Primary data type for model weights.
        compute_dtype: Data type for compute operations.
        storage_dtype: Data type for weight storage.

    Examples:
        >>> config = PrecisionConfig(
        ...     dtype=PrecisionType.FP16,
        ...     compute_dtype=PrecisionType.FP16,
        ...     storage_dtype=PrecisionType.FP32,
        ... )
        >>> config.dtype
        <PrecisionType.FP16: 'fp16'>
        >>> config.storage_dtype
        <PrecisionType.FP32: 'fp32'>
    """

    dtype: PrecisionType
    compute_dtype: PrecisionType
    storage_dtype: PrecisionType


@dataclass(frozen=True, slots=True)
class ScalerConfig:
    """Configuration for loss scaling.

    Attributes:
        strategy: Scaling strategy to use.
        initial_scale: Initial loss scale value.
        growth_factor: Factor to increase scale by.
        backoff_factor: Factor to decrease scale by.
        growth_interval: Steps between scale increases.
        max_scale: Maximum allowed scale.
        min_scale: Minimum allowed scale.

    Examples:
        >>> config = ScalerConfig(
        ...     strategy=ScalingStrategy.DYNAMIC,
        ...     initial_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000,
        ...     max_scale=2**24,
        ...     min_scale=1.0,
        ... )
        >>> config.strategy
        <ScalingStrategy.DYNAMIC: 'dynamic'>
        >>> config.initial_scale
        65536.0
    """

    strategy: ScalingStrategy
    initial_scale: float
    growth_factor: float
    backoff_factor: float
    growth_interval: int
    max_scale: float
    min_scale: float


@dataclass(frozen=True, slots=True)
class MixedPrecisionConfig:
    """Combined mixed precision training configuration.

    Attributes:
        precision_config: Precision settings.
        scaler_config: Loss scaler settings.
        casting_policy: Tensor casting policy.
        enabled: Whether mixed precision is enabled.
        autocast_enabled: Whether autocast is enabled.

    Examples:
        >>> precision = PrecisionConfig(
        ...     PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP32
        ... )
        >>> scaler = ScalerConfig(
        ...     ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 2**24, 1.0
        ... )
        >>> config = MixedPrecisionConfig(
        ...     precision_config=precision,
        ...     scaler_config=scaler,
        ...     casting_policy=CastingPolicy.ALL,
        ...     enabled=True,
        ...     autocast_enabled=True,
        ... )
        >>> config.enabled
        True
    """

    precision_config: PrecisionConfig
    scaler_config: ScalerConfig
    casting_policy: CastingPolicy
    enabled: bool
    autocast_enabled: bool


@dataclass(frozen=True, slots=True)
class PrecisionStats:
    """Statistics for mixed precision training.

    Attributes:
        current_scale: Current loss scale value.
        num_overflows: Count of overflow events.
        num_scale_updates: Count of scale updates.
        overflow_rate: Rate of overflow events.
        memory_reduction_pct: Memory reduction percentage.
        throughput_improvement_pct: Throughput improvement percentage.

    Examples:
        >>> stats = PrecisionStats(
        ...     current_scale=65536.0,
        ...     num_overflows=10,
        ...     num_scale_updates=5,
        ...     overflow_rate=0.001,
        ...     memory_reduction_pct=45.0,
        ...     throughput_improvement_pct=30.0,
        ... )
        >>> stats.current_scale
        65536.0
    """

    current_scale: float
    num_overflows: int
    num_scale_updates: int
    overflow_rate: float
    memory_reduction_pct: float
    throughput_improvement_pct: float


def validate_precision_config(config: PrecisionConfig) -> None:
    """Validate precision configuration.

    Args:
        config: Precision configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If FP8 is used as storage dtype.

    Examples:
        >>> config = PrecisionConfig(
        ...     PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP32
        ... )
        >>> validate_precision_config(config)  # No error

        >>> validate_precision_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = PrecisionConfig(
        ...     PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP8_E4M3
        ... )
        >>> validate_precision_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: FP8 types cannot be used for storage
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    fp8_types = {PrecisionType.FP8_E4M3, PrecisionType.FP8_E5M2}
    if config.storage_dtype in fp8_types:
        msg = "FP8 types cannot be used for storage"
        raise ValueError(msg)


def validate_scaler_config(config: ScalerConfig) -> None:
    """Validate scaler configuration.

    Args:
        config: Scaler configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If initial_scale is not positive.
        ValueError: If growth_factor is not greater than 1.
        ValueError: If backoff_factor is not between 0 and 1.
        ValueError: If growth_interval is not positive.
        ValueError: If max_scale < min_scale.

    Examples:
        >>> config = ScalerConfig(
        ...     ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 2**24, 1.0
        ... )
        >>> validate_scaler_config(config)  # No error

        >>> validate_scaler_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ScalerConfig(
        ...     ScalingStrategy.DYNAMIC, 0.0, 2.0, 0.5, 2000, 2**24, 1.0
        ... )
        >>> validate_scaler_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: initial_scale must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.initial_scale <= 0:
        msg = f"initial_scale must be positive, got {config.initial_scale}"
        raise ValueError(msg)

    if config.growth_factor <= 1:
        msg = f"growth_factor must be greater than 1, got {config.growth_factor}"
        raise ValueError(msg)

    if not 0 < config.backoff_factor < 1:
        msg = f"backoff_factor must be between 0 and 1, got {config.backoff_factor}"
        raise ValueError(msg)

    if config.growth_interval <= 0:
        msg = f"growth_interval must be positive, got {config.growth_interval}"
        raise ValueError(msg)

    if config.max_scale < config.min_scale:
        msg = (
            f"max_scale must be >= min_scale, "
            f"got max={config.max_scale}, min={config.min_scale}"
        )
        raise ValueError(msg)


def validate_mixed_precision_config(config: MixedPrecisionConfig) -> None:
    """Validate mixed precision configuration.

    Args:
        config: Mixed precision configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If sub-configs are invalid.

    Examples:
        >>> precision = PrecisionConfig(
        ...     PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP32
        ... )
        >>> scaler = ScalerConfig(
        ...     ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 2**24, 1.0
        ... )
        >>> config = MixedPrecisionConfig(
        ...     precision, scaler, CastingPolicy.ALL, True, True
        ... )
        >>> validate_mixed_precision_config(config)  # No error

        >>> validate_mixed_precision_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_precision_config(config.precision_config)
    validate_scaler_config(config.scaler_config)


def create_precision_config(
    dtype: str = "fp16",
    compute_dtype: str | None = None,
    storage_dtype: str = "fp32",
) -> PrecisionConfig:
    """Create a precision configuration.

    Args:
        dtype: Primary data type. Defaults to "fp16".
        compute_dtype: Compute data type. Defaults to dtype.
        storage_dtype: Storage data type. Defaults to "fp32".

    Returns:
        Validated PrecisionConfig instance.

    Raises:
        ValueError: If dtype is invalid.

    Examples:
        >>> config = create_precision_config()
        >>> config.dtype
        <PrecisionType.FP16: 'fp16'>

        >>> config = create_precision_config(dtype="bf16")
        >>> config.dtype
        <PrecisionType.BF16: 'bf16'>

        >>> create_precision_config(dtype="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dtype must be one of
    """
    if dtype not in VALID_PRECISION_TYPES:
        msg = f"dtype must be one of {VALID_PRECISION_TYPES}, got '{dtype}'"
        raise ValueError(msg)

    if compute_dtype is None:
        compute_dtype = dtype

    if compute_dtype not in VALID_PRECISION_TYPES:
        msg = (
            f"compute_dtype must be one of {VALID_PRECISION_TYPES}, "
            f"got '{compute_dtype}'"
        )
        raise ValueError(msg)

    if storage_dtype not in VALID_PRECISION_TYPES:
        msg = (
            f"storage_dtype must be one of {VALID_PRECISION_TYPES}, "
            f"got '{storage_dtype}'"
        )
        raise ValueError(msg)

    config = PrecisionConfig(
        dtype=PrecisionType(dtype),
        compute_dtype=PrecisionType(compute_dtype),
        storage_dtype=PrecisionType(storage_dtype),
    )
    validate_precision_config(config)
    return config


def create_scaler_config(
    strategy: str = "dynamic",
    initial_scale: float = 65536.0,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    max_scale: float = 2**24,
    min_scale: float = 1.0,
) -> ScalerConfig:
    """Create a scaler configuration.

    Args:
        strategy: Scaling strategy. Defaults to "dynamic".
        initial_scale: Initial loss scale. Defaults to 65536.0.
        growth_factor: Scale growth factor. Defaults to 2.0.
        backoff_factor: Scale backoff factor. Defaults to 0.5.
        growth_interval: Steps between increases. Defaults to 2000.
        max_scale: Maximum scale. Defaults to 2**24.
        min_scale: Minimum scale. Defaults to 1.0.

    Returns:
        Validated ScalerConfig instance.

    Raises:
        ValueError: If strategy is invalid.

    Examples:
        >>> config = create_scaler_config()
        >>> config.strategy
        <ScalingStrategy.DYNAMIC: 'dynamic'>

        >>> config = create_scaler_config(strategy="static", initial_scale=1024.0)
        >>> config.initial_scale
        1024.0

        >>> create_scaler_config(strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if strategy not in VALID_SCALING_STRATEGIES:
        msg = f"strategy must be one of {VALID_SCALING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = ScalerConfig(
        strategy=ScalingStrategy(strategy),
        initial_scale=initial_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval,
        max_scale=max_scale,
        min_scale=min_scale,
    )
    validate_scaler_config(config)
    return config


def create_mixed_precision_config(
    precision_config: PrecisionConfig | None = None,
    scaler_config: ScalerConfig | None = None,
    casting_policy: str = "all",
    enabled: bool = True,
    autocast_enabled: bool = True,
) -> MixedPrecisionConfig:
    """Create a mixed precision configuration.

    Args:
        precision_config: Precision settings. Defaults to new FP16 config.
        scaler_config: Scaler settings. Defaults to new dynamic config.
        casting_policy: Casting policy. Defaults to "all".
        enabled: Enable mixed precision. Defaults to True.
        autocast_enabled: Enable autocast. Defaults to True.

    Returns:
        Validated MixedPrecisionConfig instance.

    Raises:
        ValueError: If casting_policy is invalid.

    Examples:
        >>> config = create_mixed_precision_config()
        >>> config.enabled
        True
        >>> config.precision_config.dtype
        <PrecisionType.FP16: 'fp16'>

        >>> custom_precision = create_precision_config(dtype="bf16")
        >>> config = create_mixed_precision_config(precision_config=custom_precision)
        >>> config.precision_config.dtype
        <PrecisionType.BF16: 'bf16'>

        >>> create_mixed_precision_config(casting_policy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: casting_policy must be one of
    """
    if casting_policy not in VALID_CASTING_POLICIES:
        msg = (
            f"casting_policy must be one of {VALID_CASTING_POLICIES}, "
            f"got '{casting_policy}'"
        )
        raise ValueError(msg)

    if precision_config is None:
        precision_config = create_precision_config()

    if scaler_config is None:
        scaler_config = create_scaler_config()

    config = MixedPrecisionConfig(
        precision_config=precision_config,
        scaler_config=scaler_config,
        casting_policy=CastingPolicy(casting_policy),
        enabled=enabled,
        autocast_enabled=autocast_enabled,
    )
    validate_mixed_precision_config(config)
    return config


def calculate_memory_reduction(
    original_dtype: str = "fp32",
    target_dtype: str = "fp16",
    model_params_billions: float = 7.0,
) -> tuple[float, float]:
    """Calculate memory reduction from precision change.

    Args:
        original_dtype: Original precision type. Defaults to "fp32".
        target_dtype: Target precision type. Defaults to "fp16".
        model_params_billions: Model size in billions. Defaults to 7.0.

    Returns:
        Tuple of (memory_saved_gb, reduction_percentage).

    Raises:
        ValueError: If dtype is invalid.
        ValueError: If model_params_billions is not positive.

    Examples:
        >>> saved, pct = calculate_memory_reduction("fp32", "fp16", 7.0)
        >>> saved > 0
        True
        >>> pct == 50.0
        True

        >>> saved, pct = calculate_memory_reduction("fp32", "bf16", 7.0)
        >>> pct == 50.0
        True

        >>> calculate_memory_reduction("fp32", "fp16", 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params_billions must be positive
    """
    if original_dtype not in VALID_PRECISION_TYPES:
        msg = (
            f"original_dtype must be one of {VALID_PRECISION_TYPES}, "
            f"got '{original_dtype}'"
        )
        raise ValueError(msg)

    if target_dtype not in VALID_PRECISION_TYPES:
        msg = (
            f"target_dtype must be one of {VALID_PRECISION_TYPES}, got '{target_dtype}'"
        )
        raise ValueError(msg)

    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    # Bytes per element for each dtype
    bytes_per_dtype = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8_e4m3": 1.0,
        "fp8_e5m2": 1.0,
    }

    original_bytes = bytes_per_dtype[original_dtype]
    target_bytes = bytes_per_dtype[target_dtype]

    # Memory in GB (params * bytes / 1e9)
    original_memory_gb = model_params_billions * original_bytes
    target_memory_gb = model_params_billions * target_bytes

    memory_saved_gb = original_memory_gb - target_memory_gb
    reduction_percentage = (memory_saved_gb / original_memory_gb) * 100

    return round(memory_saved_gb, 2), round(reduction_percentage, 2)


def estimate_speedup(
    original_dtype: str = "fp32",
    target_dtype: str = "fp16",
    gpu_architecture: str = "ampere",
) -> float:
    """Estimate training speedup from precision change.

    Args:
        original_dtype: Original precision type. Defaults to "fp32".
        target_dtype: Target precision type. Defaults to "fp16".
        gpu_architecture: GPU architecture name. Defaults to "ampere".

    Returns:
        Estimated speedup factor (>1 means faster).

    Raises:
        ValueError: If dtype is invalid.

    Examples:
        >>> speedup = estimate_speedup("fp32", "fp16", "ampere")
        >>> speedup > 1.0
        True

        >>> speedup = estimate_speedup("fp32", "bf16", "hopper")
        >>> speedup > 1.0
        True

        >>> speedup = estimate_speedup("fp32", "fp8_e4m3", "hopper")
        >>> speedup > 2.0
        True
    """
    if original_dtype not in VALID_PRECISION_TYPES:
        msg = (
            f"original_dtype must be one of {VALID_PRECISION_TYPES}, "
            f"got '{original_dtype}'"
        )
        raise ValueError(msg)

    if target_dtype not in VALID_PRECISION_TYPES:
        msg = (
            f"target_dtype must be one of {VALID_PRECISION_TYPES}, got '{target_dtype}'"
        )
        raise ValueError(msg)

    # Base speedup ratios (approximate, real-world varies)
    # FP32 = 1.0 baseline
    base_speedup = {
        "fp32": 1.0,
        "fp16": 2.0,  # Tensor cores double throughput
        "bf16": 2.0,  # Same as FP16 on Ampere+
        "fp8_e4m3": 4.0,  # Hopper FP8 tensor cores
        "fp8_e5m2": 4.0,  # Hopper FP8 tensor cores
    }

    # Architecture multipliers
    arch_multiplier = {
        "volta": 0.8,  # Limited tensor core support
        "turing": 0.9,  # Good FP16 support
        "ampere": 1.0,  # Full BF16 support
        "hopper": 1.2,  # FP8 support, enhanced tensor cores
        "blackwell": 1.3,  # Next-gen improvements
    }

    # Default to ampere if unknown architecture
    multiplier = arch_multiplier.get(gpu_architecture.lower(), 1.0)

    original_speed = base_speedup[original_dtype]
    target_speed = base_speedup[target_dtype]

    speedup = (target_speed / original_speed) * multiplier
    return round(speedup, 2)


def check_overflow_risk(
    gradient_norm: float,
    loss_scale: float,
    dtype: str = "fp16",
) -> tuple[bool, float]:
    """Check risk of gradient overflow in mixed precision training.

    Args:
        gradient_norm: Current gradient norm.
        loss_scale: Current loss scale.
        dtype: Precision type. Defaults to "fp16".

    Returns:
        Tuple of (overflow_detected, headroom_ratio).

    Raises:
        ValueError: If gradient_norm is negative.
        ValueError: If loss_scale is not positive.

    Examples:
        >>> overflow, headroom = check_overflow_risk(0.1, 1000.0, "fp16")
        >>> overflow
        False
        >>> headroom > 1.0
        True

        >>> overflow, headroom = check_overflow_risk(10.0, 1000.0, "fp16")
        >>> headroom < 10.0
        True

        >>> check_overflow_risk(-1.0, 65536.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradient_norm cannot be negative
    """
    if gradient_norm < 0:
        msg = f"gradient_norm cannot be negative, got {gradient_norm}"
        raise ValueError(msg)

    if loss_scale <= 0:
        msg = f"loss_scale must be positive, got {loss_scale}"
        raise ValueError(msg)

    # Max representable values
    max_values = {
        "fp32": 3.4e38,
        "fp16": 65504.0,
        "bf16": 3.39e38,  # Similar to FP32 due to exponent range
        "fp8_e4m3": 448.0,
        "fp8_e5m2": 57344.0,
    }

    max_val = max_values.get(dtype, 65504.0)  # Default to FP16

    # Scaled gradient
    scaled_gradient = gradient_norm * loss_scale

    # Headroom ratio (how far from overflow)
    headroom = float("inf") if scaled_gradient == 0 else max_val / scaled_gradient

    overflow_detected = scaled_gradient > max_val

    rounded_headroom = round(headroom, 2) if headroom != float("inf") else float("inf")
    return overflow_detected, rounded_headroom


def calculate_optimal_scale(
    gradient_norm: float,
    dtype: str = "fp16",
    target_headroom: float = 10.0,
) -> float:
    """Calculate optimal loss scale for given gradient norm.

    Args:
        gradient_norm: Current gradient norm.
        dtype: Precision type. Defaults to "fp16".
        target_headroom: Desired safety margin. Defaults to 10.0.

    Returns:
        Recommended loss scale value.

    Raises:
        ValueError: If gradient_norm is not positive.
        ValueError: If target_headroom is not positive.

    Examples:
        >>> scale = calculate_optimal_scale(1.0, "fp16", 10.0)
        >>> scale > 0
        True

        >>> scale = calculate_optimal_scale(0.01, "fp16", 10.0)
        >>> scale > 65536.0
        True

        >>> calculate_optimal_scale(0, "fp16")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradient_norm must be positive
    """
    if gradient_norm <= 0:
        msg = f"gradient_norm must be positive, got {gradient_norm}"
        raise ValueError(msg)

    if target_headroom <= 0:
        msg = f"target_headroom must be positive, got {target_headroom}"
        raise ValueError(msg)

    max_values = {
        "fp32": 3.4e38,
        "fp16": 65504.0,
        "bf16": 3.39e38,
        "fp8_e4m3": 448.0,
        "fp8_e5m2": 57344.0,
    }

    max_val = max_values.get(dtype, 65504.0)

    # Scale that achieves target headroom
    optimal_scale = max_val / (gradient_norm * target_headroom)

    # Clamp to reasonable range
    optimal_scale = max(1.0, min(optimal_scale, 2**24))

    return round(optimal_scale, 2)


def format_precision_stats(stats: PrecisionStats) -> str:
    """Format precision statistics for display.

    Args:
        stats: Precision statistics to format.

    Returns:
        Formatted string with statistics.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = PrecisionStats(65536.0, 10, 5, 0.001, 45.0, 30.0)
        >>> formatted = format_precision_stats(stats)
        >>> "Current Scale:" in formatted
        True
        >>> "Memory Reduction:" in formatted
        True

        >>> format_precision_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        f"Current Scale: {stats.current_scale:,.0f}",
        f"Overflow Count: {stats.num_overflows}",
        f"Scale Updates: {stats.num_scale_updates}",
        f"Overflow Rate: {stats.overflow_rate:.4f}",
        f"Memory Reduction: {stats.memory_reduction_pct:.1f}%",
        f"Throughput Improvement: {stats.throughput_improvement_pct:.1f}%",
    ]
    return "\n".join(lines)


def get_recommended_precision_config(
    model_size: str = "7b",
    gpu_architecture: str = "ampere",
    use_case: str = "training",
) -> MixedPrecisionConfig:
    """Get recommended mixed precision configuration.

    Args:
        model_size: Model size string (e.g., "7b", "70b"). Defaults to "7b".
        gpu_architecture: GPU architecture. Defaults to "ampere".
        use_case: Use case ("training" or "inference"). Defaults to "training".

    Returns:
        Recommended MixedPrecisionConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_precision_config("7b", "ampere")
        >>> config.precision_config.dtype in (PrecisionType.FP16, PrecisionType.BF16)
        True

        >>> config = get_recommended_precision_config("70b", "hopper")
        >>> config.enabled
        True

        >>> config = get_recommended_precision_config("7b", "ampere", "inference")
        >>> config.scaler_config.strategy == ScalingStrategy.STATIC
        True
    """
    model_size = model_size.lower().strip()
    gpu_arch = gpu_architecture.lower().strip()
    use_case = use_case.lower().strip()

    valid_use_cases = {"training", "inference"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    # Determine dtype based on GPU architecture
    if gpu_arch in ("hopper", "blackwell"):
        # Newer GPUs have native BF16 support, FP8 for large models
        if model_size in ("70b", "70B", "70", "175b", "175B"):
            dtype = "fp8_e4m3"
        else:
            dtype = "bf16"
    else:
        # Ampere and earlier prefer BF16 where available
        dtype = "bf16" if gpu_arch in ("ampere",) else "fp16"

    # Create precision config
    precision_config = create_precision_config(
        dtype=dtype,
        compute_dtype=dtype,
        storage_dtype="fp32",
    )

    # Determine scaling strategy
    if use_case == "inference":
        # Static scaling for inference
        scaler_config = create_scaler_config(
            strategy="static",
            initial_scale=1.0,
        )
    else:
        # Dynamic scaling for training
        # Larger models benefit from higher initial scale
        if model_size in ("70b", "70B", "70", "175b", "175B"):
            initial_scale = 2**16
        else:
            initial_scale = 2**15

        scaler_config = create_scaler_config(
            strategy="dynamic",
            initial_scale=float(initial_scale),
        )

    # Create mixed precision config
    return create_mixed_precision_config(
        precision_config=precision_config,
        scaler_config=scaler_config,
        casting_policy="all",
        enabled=True,
        autocast_enabled=use_case == "training",
    )


def list_precision_types() -> list[str]:
    """List supported precision types.

    Returns:
        Sorted list of precision type names.

    Examples:
        >>> types = list_precision_types()
        >>> "fp16" in types
        True
        >>> "bf16" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_PRECISION_TYPES)


def list_scaling_strategies() -> list[str]:
    """List supported scaling strategies.

    Returns:
        Sorted list of scaling strategy names.

    Examples:
        >>> strategies = list_scaling_strategies()
        >>> "dynamic" in strategies
        True
        >>> "static" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SCALING_STRATEGIES)


def list_casting_policies() -> list[str]:
    """List supported casting policies.

    Returns:
        Sorted list of casting policy names.

    Examples:
        >>> policies = list_casting_policies()
        >>> "all" in policies
        True
        >>> "compute_only" in policies
        True
        >>> policies == sorted(policies)
        True
    """
    return sorted(VALID_CASTING_POLICIES)


def get_precision_type(name: str) -> PrecisionType:
    """Get precision type from name.

    Args:
        name: Precision type name.

    Returns:
        PrecisionType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_precision_type("fp16")
        <PrecisionType.FP16: 'fp16'>

        >>> get_precision_type("bf16")
        <PrecisionType.BF16: 'bf16'>

        >>> get_precision_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: precision type must be one of
    """
    if name not in VALID_PRECISION_TYPES:
        msg = f"precision type must be one of {VALID_PRECISION_TYPES}, got '{name}'"
        raise ValueError(msg)
    return PrecisionType(name)


def get_scaling_strategy(name: str) -> ScalingStrategy:
    """Get scaling strategy from name.

    Args:
        name: Strategy name.

    Returns:
        ScalingStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_scaling_strategy("dynamic")
        <ScalingStrategy.DYNAMIC: 'dynamic'>

        >>> get_scaling_strategy("static")
        <ScalingStrategy.STATIC: 'static'>

        >>> get_scaling_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scaling strategy must be one of
    """
    if name not in VALID_SCALING_STRATEGIES:
        msg = (
            f"scaling strategy must be one of {VALID_SCALING_STRATEGIES}, got '{name}'"
        )
        raise ValueError(msg)
    return ScalingStrategy(name)


def get_casting_policy(name: str) -> CastingPolicy:
    """Get casting policy from name.

    Args:
        name: Policy name.

    Returns:
        CastingPolicy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_casting_policy("all")
        <CastingPolicy.ALL: 'all'>

        >>> get_casting_policy("compute_only")
        <CastingPolicy.COMPUTE_ONLY: 'compute_only'>

        >>> get_casting_policy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: casting policy must be one of
    """
    if name not in VALID_CASTING_POLICIES:
        msg = f"casting policy must be one of {VALID_CASTING_POLICIES}, got '{name}'"
        raise ValueError(msg)
    return CastingPolicy(name)
