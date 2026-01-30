"""Model quantization utilities for inference.

This module provides utilities for configuring model quantization during inference,
including GPTQ, AWQ, GGUF, and dynamic quantization methods. It focuses on
quantization configuration, error estimation, and memory savings calculations.

Examples:
    >>> from hf_gtc.inference.quantization import create_gptq_config
    >>> config = create_gptq_config(bits=4)
    >>> config.bits
    4
    >>> config.group_size
    128
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class QuantizationMethod(Enum):
    """Quantization method types for inference.

    Attributes:
        GPTQ: GPTQ (GPT Quantization) method.
        AWQ: AWQ (Activation-aware Weight Quantization) method.
        GGUF: GGUF format quantization (for llama.cpp).
        DYNAMIC: Dynamic quantization at runtime.
        STATIC: Static quantization with calibration.
        INT8: 8-bit integer quantization.
        INT4: 4-bit integer quantization.

    Examples:
        >>> QuantizationMethod.GPTQ.value
        'gptq'
        >>> QuantizationMethod.AWQ.value
        'awq'
        >>> QuantizationMethod.DYNAMIC.value
        'dynamic'
    """

    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    DYNAMIC = "dynamic"
    STATIC = "static"
    INT8 = "int8"
    INT4 = "int4"


VALID_QUANTIZATION_METHODS = frozenset(m.value for m in QuantizationMethod)


class CalibrationStrategy(Enum):
    """Calibration strategies for quantization.

    Attributes:
        MINMAX: Min-max calibration strategy.
        PERCENTILE: Percentile-based calibration.
        MSE: Mean squared error minimization.
        ENTROPY: Entropy-based calibration (KL divergence).

    Examples:
        >>> CalibrationStrategy.MINMAX.value
        'minmax'
        >>> CalibrationStrategy.ENTROPY.value
        'entropy'
    """

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    MSE = "mse"
    ENTROPY = "entropy"


VALID_CALIBRATION_STRATEGIES = frozenset(s.value for s in CalibrationStrategy)


class QuantizationGranularity(Enum):
    """Quantization granularity levels.

    Attributes:
        PER_TENSOR: Single scale/zero-point per tensor.
        PER_CHANNEL: Scale/zero-point per output channel.
        PER_GROUP: Scale/zero-point per group of weights.

    Examples:
        >>> QuantizationGranularity.PER_TENSOR.value
        'per_tensor'
        >>> QuantizationGranularity.PER_GROUP.value
        'per_group'
    """

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


VALID_QUANTIZATION_GRANULARITIES = frozenset(g.value for g in QuantizationGranularity)


# Type aliases
QuantizationMethodStr = Literal[
    "gptq", "awq", "gguf", "dynamic", "static", "int8", "int4"
]
CalibrationStrategyStr = Literal["minmax", "percentile", "mse", "entropy"]
QuantizationGranularityStr = Literal["per_tensor", "per_channel", "per_group"]
GGUFQuantTypeStr = Literal[
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"
]
AWQVersionStr = Literal["gemm", "gemv", "marlin"]
ModelSizeStr = Literal["small", "medium", "large", "xlarge"]


@dataclass(frozen=True, slots=True)
class GPTQConfig:
    """Configuration for GPTQ quantization.

    Attributes:
        bits: Number of bits (typically 4 or 8).
        group_size: Group size for group-wise quantization.
        damp_percent: Dampening percentage for Hessian.
        desc_act: Use descending activation order.

    Examples:
        >>> config = GPTQConfig(
        ...     bits=4, group_size=128, damp_percent=0.01, desc_act=True
        ... )
        >>> config.bits
        4
        >>> config.group_size
        128
    """

    bits: int
    group_size: int
    damp_percent: float
    desc_act: bool


@dataclass(frozen=True, slots=True)
class AWQConfig:
    """Configuration for AWQ quantization.

    Attributes:
        bits: Number of bits (typically 4).
        group_size: Group size for group-wise quantization.
        zero_point: Use zero point for asymmetric quantization.
        version: AWQ implementation version.

    Examples:
        >>> config = AWQConfig(bits=4, group_size=128, zero_point=True, version="gemm")
        >>> config.bits
        4
        >>> config.version
        'gemm'
    """

    bits: int
    group_size: int
    zero_point: bool
    version: AWQVersionStr


@dataclass(frozen=True, slots=True)
class GGUFConfig:
    """Configuration for GGUF format quantization.

    Attributes:
        quantization_type: GGUF quantization type.
        allow_requantize: Allow requantization of already quantized models.

    Examples:
        >>> config = GGUFConfig(quantization_type="q4_0", allow_requantize=False)
        >>> config.quantization_type
        'q4_0'
        >>> config.allow_requantize
        False
    """

    quantization_type: GGUFQuantTypeStr
    allow_requantize: bool


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """General quantization configuration.

    Attributes:
        method: Quantization method to use.
        bits: Number of bits for quantization.
        calibration_strategy: Calibration strategy.
        calibration_samples: Number of calibration samples.
        granularity: Quantization granularity level.

    Examples:
        >>> config = QuantizationConfig(
        ...     method=QuantizationMethod.DYNAMIC,
        ...     bits=8,
        ...     calibration_strategy=CalibrationStrategy.MINMAX,
        ...     calibration_samples=128,
        ...     granularity=QuantizationGranularity.PER_TENSOR,
        ... )
        >>> config.bits
        8
        >>> config.method
        <QuantizationMethod.DYNAMIC: 'dynamic'>
    """

    method: QuantizationMethod
    bits: int
    calibration_strategy: CalibrationStrategy
    calibration_samples: int
    granularity: QuantizationGranularity


@dataclass(frozen=True, slots=True)
class QuantizationStats:
    """Statistics from model quantization.

    Attributes:
        original_size_mb: Original model size in megabytes.
        quantized_size_mb: Quantized model size in megabytes.
        compression_ratio: Compression ratio achieved.
        accuracy_drop: Estimated accuracy drop (percentage points).

    Examples:
        >>> stats = QuantizationStats(
        ...     original_size_mb=14000.0,
        ...     quantized_size_mb=4000.0,
        ...     compression_ratio=3.5,
        ...     accuracy_drop=0.5,
        ... )
        >>> stats.compression_ratio
        3.5
        >>> stats.original_size_mb
        14000.0
    """

    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    accuracy_drop: float


def validate_gptq_config(config: GPTQConfig) -> None:
    """Validate GPTQ configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = GPTQConfig(
        ...     bits=4, group_size=128, damp_percent=0.01, desc_act=True
        ... )
        >>> validate_gptq_config(config)  # No error

        >>> bad = GPTQConfig(bits=5, group_size=128, damp_percent=0.01, desc_act=True)
        >>> validate_gptq_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be 4 or 8 for GPTQ
    """
    if config.bits not in (4, 8):
        msg = f"bits must be 4 or 8 for GPTQ, got {config.bits}"
        raise ValueError(msg)

    if config.group_size <= 0:
        msg = f"group_size must be positive, got {config.group_size}"
        raise ValueError(msg)

    if not 0.0 <= config.damp_percent <= 1.0:
        msg = f"damp_percent must be between 0.0 and 1.0, got {config.damp_percent}"
        raise ValueError(msg)


def validate_awq_config(config: AWQConfig) -> None:
    """Validate AWQ configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = AWQConfig(bits=4, group_size=128, zero_point=True, version="gemm")
        >>> validate_awq_config(config)  # No error

        >>> bad = AWQConfig(bits=8, group_size=128, zero_point=True, version="gemm")
        >>> validate_awq_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: AWQ only supports 4-bit quantization
    """
    if config.bits != 4:
        msg = f"AWQ only supports 4-bit quantization, got {config.bits}"
        raise ValueError(msg)

    if config.group_size <= 0:
        msg = f"group_size must be positive, got {config.group_size}"
        raise ValueError(msg)

    valid_versions = {"gemm", "gemv", "marlin"}
    if config.version not in valid_versions:
        msg = f"version must be one of {valid_versions}, got '{config.version}'"
        raise ValueError(msg)


def validate_gguf_config(config: GGUFConfig) -> None:
    """Validate GGUF configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = GGUFConfig(quantization_type="q4_0", allow_requantize=False)
        >>> validate_gguf_config(config)  # No error

        >>> bad = GGUFConfig(quantization_type="invalid", allow_requantize=False)
        >>> validate_gguf_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quantization_type must be one of
    """
    valid_types = {
        "q4_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q8_0",
        "q2_k",
        "q3_k",
        "q4_k",
        "q5_k",
        "q6_k",
    }
    if config.quantization_type not in valid_types:
        msg = (
            f"quantization_type must be one of {valid_types}, "
            f"got '{config.quantization_type}'"
        )
        raise ValueError(msg)


def validate_quantization_config(config: QuantizationConfig) -> None:
    """Validate general quantization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = QuantizationConfig(
        ...     method=QuantizationMethod.DYNAMIC,
        ...     bits=8,
        ...     calibration_strategy=CalibrationStrategy.MINMAX,
        ...     calibration_samples=128,
        ...     granularity=QuantizationGranularity.PER_TENSOR,
        ... )
        >>> validate_quantization_config(config)  # No error

        >>> bad = QuantizationConfig(
        ...     method=QuantizationMethod.DYNAMIC,
        ...     bits=0,
        ...     calibration_strategy=CalibrationStrategy.MINMAX,
        ...     calibration_samples=128,
        ...     granularity=QuantizationGranularity.PER_TENSOR,
        ... )
        >>> validate_quantization_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be positive
    """
    valid_bits = {2, 3, 4, 8, 16}
    if config.bits not in valid_bits:
        msg = f"bits must be one of {valid_bits}, got {config.bits}"
        raise ValueError(msg)

    if config.calibration_samples <= 0:
        msg = f"calibration_samples must be positive, got {config.calibration_samples}"
        raise ValueError(msg)


def validate_quantization_stats(stats: QuantizationStats) -> None:
    """Validate quantization statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = QuantizationStats(
        ...     original_size_mb=14000.0,
        ...     quantized_size_mb=4000.0,
        ...     compression_ratio=3.5,
        ...     accuracy_drop=0.5,
        ... )
        >>> validate_quantization_stats(stats)  # No error

        >>> bad = QuantizationStats(
        ...     original_size_mb=-100.0,
        ...     quantized_size_mb=4000.0,
        ...     compression_ratio=3.5,
        ...     accuracy_drop=0.5,
        ... )
        >>> validate_quantization_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_size_mb must be positive
    """
    if stats.original_size_mb <= 0:
        msg = f"original_size_mb must be positive, got {stats.original_size_mb}"
        raise ValueError(msg)

    if stats.quantized_size_mb <= 0:
        msg = f"quantized_size_mb must be positive, got {stats.quantized_size_mb}"
        raise ValueError(msg)

    if stats.compression_ratio <= 0:
        msg = f"compression_ratio must be positive, got {stats.compression_ratio}"
        raise ValueError(msg)

    if stats.accuracy_drop < 0:
        msg = f"accuracy_drop cannot be negative, got {stats.accuracy_drop}"
        raise ValueError(msg)


def create_gptq_config(
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
    desc_act: bool = True,
) -> GPTQConfig:
    """Create a GPTQ configuration.

    Args:
        bits: Number of bits. Defaults to 4.
        group_size: Group size. Defaults to 128.
        damp_percent: Dampening percentage. Defaults to 0.01.
        desc_act: Descending activation order. Defaults to True.

    Returns:
        GPTQConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_gptq_config(bits=4)
        >>> config.bits
        4
        >>> config.group_size
        128

        >>> config = create_gptq_config(bits=8, group_size=64)
        >>> config.bits
        8
        >>> config.group_size
        64

        >>> create_gptq_config(bits=5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be 4 or 8 for GPTQ
    """
    config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
    )
    validate_gptq_config(config)
    return config


def create_awq_config(
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: AWQVersionStr = "gemm",
) -> AWQConfig:
    """Create an AWQ configuration.

    Args:
        bits: Number of bits. Defaults to 4.
        group_size: Group size. Defaults to 128.
        zero_point: Use zero point. Defaults to True.
        version: AWQ version. Defaults to "gemm".

    Returns:
        AWQConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_awq_config(bits=4)
        >>> config.bits
        4
        >>> config.version
        'gemm'

        >>> config = create_awq_config(group_size=64, version="marlin")
        >>> config.group_size
        64
        >>> config.version
        'marlin'

        >>> create_awq_config(bits=8)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: AWQ only supports 4-bit quantization
    """
    config = AWQConfig(
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
        version=version,
    )
    validate_awq_config(config)
    return config


def create_gguf_config(
    quantization_type: GGUFQuantTypeStr = "q4_0",
    allow_requantize: bool = False,
) -> GGUFConfig:
    """Create a GGUF configuration.

    Args:
        quantization_type: GGUF quantization type. Defaults to "q4_0".
        allow_requantize: Allow requantization. Defaults to False.

    Returns:
        GGUFConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_gguf_config(quantization_type="q4_0")
        >>> config.quantization_type
        'q4_0'

        >>> config = create_gguf_config(quantization_type="q8_0", allow_requantize=True)
        >>> config.quantization_type
        'q8_0'
        >>> config.allow_requantize
        True

        >>> create_gguf_config(quantization_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quantization_type must be one of
    """
    config = GGUFConfig(
        quantization_type=quantization_type,
        allow_requantize=allow_requantize,
    )
    validate_gguf_config(config)
    return config


def create_quantization_config(
    method: QuantizationMethodStr = "dynamic",
    bits: int = 8,
    calibration_strategy: CalibrationStrategyStr = "minmax",
    calibration_samples: int = 128,
    granularity: QuantizationGranularityStr = "per_tensor",
) -> QuantizationConfig:
    """Create a general quantization configuration.

    Args:
        method: Quantization method. Defaults to "dynamic".
        bits: Number of bits. Defaults to 8.
        calibration_strategy: Calibration strategy. Defaults to "minmax".
        calibration_samples: Number of calibration samples. Defaults to 128.
        granularity: Quantization granularity. Defaults to "per_tensor".

    Returns:
        QuantizationConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_quantization_config(method="dynamic", bits=8)
        >>> config.method
        <QuantizationMethod.DYNAMIC: 'dynamic'>
        >>> config.bits
        8

        >>> config = create_quantization_config(
        ...     method="static",
        ...     calibration_strategy="entropy",
        ...     calibration_samples=256,
        ... )
        >>> config.calibration_strategy
        <CalibrationStrategy.ENTROPY: 'entropy'>
        >>> config.calibration_samples
        256

        >>> create_quantization_config(bits=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be one of
    """
    if method not in VALID_QUANTIZATION_METHODS:
        msg = f"method must be one of {VALID_QUANTIZATION_METHODS}, got '{method}'"
        raise ValueError(msg)

    if calibration_strategy not in VALID_CALIBRATION_STRATEGIES:
        msg = (
            f"calibration_strategy must be one of {VALID_CALIBRATION_STRATEGIES}, "
            f"got '{calibration_strategy}'"
        )
        raise ValueError(msg)

    if granularity not in VALID_QUANTIZATION_GRANULARITIES:
        msg = (
            f"granularity must be one of {VALID_QUANTIZATION_GRANULARITIES}, "
            f"got '{granularity}'"
        )
        raise ValueError(msg)

    config = QuantizationConfig(
        method=QuantizationMethod(method),
        bits=bits,
        calibration_strategy=CalibrationStrategy(calibration_strategy),
        calibration_samples=calibration_samples,
        granularity=QuantizationGranularity(granularity),
    )
    validate_quantization_config(config)
    return config


def create_quantization_stats(
    original_size_mb: float,
    quantized_size_mb: float,
    compression_ratio: float | None = None,
    accuracy_drop: float = 0.0,
) -> QuantizationStats:
    """Create quantization statistics.

    Args:
        original_size_mb: Original model size in MB.
        quantized_size_mb: Quantized model size in MB.
        compression_ratio: Compression ratio. Computed if not provided.
        accuracy_drop: Estimated accuracy drop. Defaults to 0.0.

    Returns:
        QuantizationStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_quantization_stats(14000.0, 4000.0)
        >>> stats.compression_ratio
        3.5

        >>> stats = create_quantization_stats(14000.0, 4000.0, accuracy_drop=0.5)
        >>> stats.accuracy_drop
        0.5

        >>> create_quantization_stats(0.0, 4000.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_size_mb must be positive
    """
    if original_size_mb <= 0:
        msg = f"original_size_mb must be positive, got {original_size_mb}"
        raise ValueError(msg)

    if quantized_size_mb <= 0:
        msg = f"quantized_size_mb must be positive, got {quantized_size_mb}"
        raise ValueError(msg)

    if compression_ratio is None:
        compression_ratio = original_size_mb / quantized_size_mb

    stats = QuantizationStats(
        original_size_mb=original_size_mb,
        quantized_size_mb=quantized_size_mb,
        compression_ratio=compression_ratio,
        accuracy_drop=accuracy_drop,
    )
    validate_quantization_stats(stats)
    return stats


def list_quantization_methods() -> list[str]:
    """List available quantization methods.

    Returns:
        Sorted list of quantization method names.

    Examples:
        >>> methods = list_quantization_methods()
        >>> "gptq" in methods
        True
        >>> "awq" in methods
        True
        >>> "dynamic" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_QUANTIZATION_METHODS)


def list_calibration_strategies() -> list[str]:
    """List available calibration strategies.

    Returns:
        Sorted list of calibration strategy names.

    Examples:
        >>> strategies = list_calibration_strategies()
        >>> "minmax" in strategies
        True
        >>> "entropy" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_CALIBRATION_STRATEGIES)


def list_quantization_granularities() -> list[str]:
    """List available quantization granularities.

    Returns:
        Sorted list of quantization granularity names.

    Examples:
        >>> granularities = list_quantization_granularities()
        >>> "per_tensor" in granularities
        True
        >>> "per_channel" in granularities
        True
        >>> granularities == sorted(granularities)
        True
    """
    return sorted(VALID_QUANTIZATION_GRANULARITIES)


def get_quantization_method(name: str) -> QuantizationMethod:
    """Get a quantization method by name.

    Args:
        name: Name of the quantization method.

    Returns:
        The corresponding QuantizationMethod enum value.

    Raises:
        ValueError: If name is not a valid quantization method.

    Examples:
        >>> get_quantization_method("gptq")
        <QuantizationMethod.GPTQ: 'gptq'>
        >>> get_quantization_method("dynamic")
        <QuantizationMethod.DYNAMIC: 'dynamic'>

        >>> get_quantization_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown quantization method
    """
    if name not in VALID_QUANTIZATION_METHODS:
        msg = (
            f"Unknown quantization method: '{name}'. "
            f"Valid: {VALID_QUANTIZATION_METHODS}"
        )
        raise ValueError(msg)
    return QuantizationMethod(name)


def get_calibration_strategy(name: str) -> CalibrationStrategy:
    """Get a calibration strategy by name.

    Args:
        name: Name of the calibration strategy.

    Returns:
        The corresponding CalibrationStrategy enum value.

    Raises:
        ValueError: If name is not a valid calibration strategy.

    Examples:
        >>> get_calibration_strategy("minmax")
        <CalibrationStrategy.MINMAX: 'minmax'>
        >>> get_calibration_strategy("entropy")
        <CalibrationStrategy.ENTROPY: 'entropy'>

        >>> get_calibration_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown calibration strategy
    """
    if name not in VALID_CALIBRATION_STRATEGIES:
        msg = (
            f"Unknown calibration strategy: '{name}'. "
            f"Valid: {VALID_CALIBRATION_STRATEGIES}"
        )
        raise ValueError(msg)
    return CalibrationStrategy(name)


def get_quantization_granularity(name: str) -> QuantizationGranularity:
    """Get a quantization granularity by name.

    Args:
        name: Name of the quantization granularity.

    Returns:
        The corresponding QuantizationGranularity enum value.

    Raises:
        ValueError: If name is not a valid quantization granularity.

    Examples:
        >>> get_quantization_granularity("per_tensor")
        <QuantizationGranularity.PER_TENSOR: 'per_tensor'>
        >>> get_quantization_granularity("per_channel")
        <QuantizationGranularity.PER_CHANNEL: 'per_channel'>

        >>> get_quantization_granularity("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown quantization granularity
    """
    if name not in VALID_QUANTIZATION_GRANULARITIES:
        msg = (
            f"Unknown quantization granularity: '{name}'. "
            f"Valid: {VALID_QUANTIZATION_GRANULARITIES}"
        )
        raise ValueError(msg)
    return QuantizationGranularity(name)


def calculate_quantization_error(
    original_values: list[float],
    quantized_values: list[float],
    error_type: Literal["mse", "mae", "max"] = "mse",
) -> float:
    """Calculate quantization error between original and quantized values.

    Args:
        original_values: List of original floating-point values.
        quantized_values: List of quantized values.
        error_type: Type of error metric. Defaults to "mse".

    Returns:
        Calculated error value.

    Raises:
        ValueError: If lists are empty or have different lengths.

    Examples:
        >>> original = [1.0, 2.0, 3.0, 4.0]
        >>> quantized = [1.1, 2.1, 3.1, 4.1]
        >>> error = calculate_quantization_error(original, quantized, "mse")
        >>> abs(error - 0.01) < 0.001
        True

        >>> error_mae = calculate_quantization_error(original, quantized, "mae")
        >>> abs(error_mae - 0.1) < 0.001
        True

        >>> error_max = calculate_quantization_error(original, quantized, "max")
        >>> abs(error_max - 0.1) < 0.001
        True

        >>> calculate_quantization_error([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_values cannot be empty

        >>> calculate_quantization_error([1.0], [1.0, 2.0])
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_values and quantized_values must have the same length
    """
    if not original_values:
        msg = "original_values cannot be empty"
        raise ValueError(msg)

    if len(original_values) != len(quantized_values):
        msg = (
            f"original_values and quantized_values must have the same length, "
            f"got {len(original_values)} and {len(quantized_values)}"
        )
        raise ValueError(msg)

    errors = [
        abs(o - q) for o, q in zip(original_values, quantized_values, strict=True)
    ]

    if error_type == "mse":
        squared_errors = [e * e for e in errors]
        return sum(squared_errors) / len(squared_errors)
    elif error_type == "mae":
        return sum(errors) / len(errors)
    else:  # max
        return max(errors)


def estimate_memory_savings(
    original_bits: int,
    quantized_bits: int,
    model_params: int,
) -> tuple[float, float]:
    """Estimate memory savings from quantization.

    Args:
        original_bits: Original precision bits (e.g., 32 for fp32).
        quantized_bits: Quantized precision bits.
        model_params: Number of model parameters.

    Returns:
        Tuple of (original_size_mb, quantized_size_mb).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> orig, quant = estimate_memory_savings(16, 4, 7_000_000_000)
        >>> orig > quant
        True
        >>> abs(orig / quant - 4.0) < 0.01
        True

        >>> estimate_memory_savings(0, 4, 1000)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_bits must be positive

        >>> estimate_memory_savings(16, 0, 1000)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quantized_bits must be positive
    """
    if original_bits <= 0:
        msg = f"original_bits must be positive, got {original_bits}"
        raise ValueError(msg)

    if quantized_bits <= 0:
        msg = f"quantized_bits must be positive, got {quantized_bits}"
        raise ValueError(msg)

    if model_params <= 0:
        msg = f"model_params must be positive, got {model_params}"
        raise ValueError(msg)

    original_bytes = model_params * original_bits / 8
    quantized_bytes = model_params * quantized_bits / 8

    original_mb = original_bytes / (1024 * 1024)
    quantized_mb = quantized_bytes / (1024 * 1024)

    return (original_mb, quantized_mb)


def calculate_bits_per_weight(
    quantization_type: str,
) -> float:
    """Calculate effective bits per weight for a quantization type.

    Args:
        quantization_type: Quantization type string.

    Returns:
        Effective bits per weight.

    Raises:
        ValueError: If quantization_type is not recognized.

    Examples:
        >>> calculate_bits_per_weight("int8")
        8.0
        >>> calculate_bits_per_weight("int4")
        4.0
        >>> calculate_bits_per_weight("q4_0")
        4.5

        >>> calculate_bits_per_weight("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown quantization type
    """
    # Mapping of quantization types to effective bits per weight
    bits_map = {
        # Standard types
        "int8": 8.0,
        "int4": 4.0,
        "fp16": 16.0,
        "bf16": 16.0,
        "fp32": 32.0,
        # GGUF types (approximate effective bits including metadata)
        "q4_0": 4.5,
        "q4_1": 5.0,
        "q5_0": 5.5,
        "q5_1": 6.0,
        "q8_0": 8.5,
        "q2_k": 2.5,
        "q3_k": 3.4,
        "q4_k": 4.5,
        "q5_k": 5.5,
        "q6_k": 6.5,
        # GPTQ/AWQ
        "gptq_4bit": 4.0,
        "gptq_8bit": 8.0,
        "awq_4bit": 4.0,
    }

    if quantization_type not in bits_map:
        msg = f"Unknown quantization type: '{quantization_type}'"
        raise ValueError(msg)

    return bits_map[quantization_type]


def select_calibration_data(
    dataset_size: int,
    target_samples: int,
    seed: int = 42,
) -> list[int]:
    """Select indices for calibration data.

    Args:
        dataset_size: Total size of the dataset.
        target_samples: Number of samples to select.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        List of selected indices.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> indices = select_calibration_data(1000, 128, seed=42)
        >>> len(indices)
        128
        >>> all(0 <= i < 1000 for i in indices)
        True

        >>> indices2 = select_calibration_data(1000, 128, seed=42)
        >>> indices == indices2  # Deterministic with same seed
        True

        >>> select_calibration_data(0, 128)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_size must be positive

        >>> select_calibration_data(100, 200)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_samples cannot exceed dataset_size
    """
    if dataset_size <= 0:
        msg = f"dataset_size must be positive, got {dataset_size}"
        raise ValueError(msg)

    if target_samples <= 0:
        msg = f"target_samples must be positive, got {target_samples}"
        raise ValueError(msg)

    if target_samples > dataset_size:
        msg = (
            f"target_samples cannot exceed dataset_size, "
            f"got {target_samples} > {dataset_size}"
        )
        raise ValueError(msg)

    # Simple deterministic selection using linear congruential generator
    # This is reproducible and doesn't require external dependencies
    a = 1103515245
    c = 12345
    m = 2**31

    state = seed
    selected = set()
    indices = []

    while len(indices) < target_samples:
        state = (a * state + c) % m
        idx = state % dataset_size
        if idx not in selected:
            selected.add(idx)
            indices.append(idx)

    return indices


def format_quantization_stats(stats: QuantizationStats) -> str:
    """Format quantization statistics as a human-readable string.

    Args:
        stats: Quantization statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = QuantizationStats(
        ...     original_size_mb=14000.0,
        ...     quantized_size_mb=4000.0,
        ...     compression_ratio=3.5,
        ...     accuracy_drop=0.5,
        ... )
        >>> formatted = format_quantization_stats(stats)
        >>> "Original Size: 14000.00 MB" in formatted
        True
        >>> "Compression Ratio: 3.50x" in formatted
        True

        >>> stats_zero = QuantizationStats(100.0, 25.0, 4.0, 0.0)
        >>> "Accuracy Drop: 0.00%" in format_quantization_stats(stats_zero)
        True
    """
    lines = [
        f"Original Size: {stats.original_size_mb:.2f} MB",
        f"Quantized Size: {stats.quantized_size_mb:.2f} MB",
        f"Compression Ratio: {stats.compression_ratio:.2f}x",
        f"Accuracy Drop: {stats.accuracy_drop:.2f}%",
    ]
    return "\n".join(lines)


def get_recommended_quantization_config(
    model_size: ModelSizeStr,
    target_device: Literal["cpu", "gpu", "mobile"] = "gpu",
) -> QuantizationConfig:
    """Get recommended quantization configuration for model and device.

    Args:
        model_size: Model size category.
        target_device: Target device for inference. Defaults to "gpu".

    Returns:
        Recommended QuantizationConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_quantization_config("large", "gpu")
        >>> config.method
        <QuantizationMethod.GPTQ: 'gptq'>
        >>> config.bits
        4

        >>> config = get_recommended_quantization_config("small", "cpu")
        >>> config.method
        <QuantizationMethod.DYNAMIC: 'dynamic'>

        >>> config = get_recommended_quantization_config("medium", "mobile")
        >>> config.bits
        4

        >>> get_recommended_quantization_config("invalid", "gpu")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown model size
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"Unknown model size: '{model_size}'. Valid: {valid_sizes}"
        raise ValueError(msg)

    valid_devices = {"cpu", "gpu", "mobile"}
    if target_device not in valid_devices:
        msg = f"Unknown target device: '{target_device}'. Valid: {valid_devices}"
        raise ValueError(msg)

    # CPU recommendations: dynamic quantization works well
    if target_device == "cpu":
        if model_size in ("small", "medium"):
            return create_quantization_config(
                method="dynamic",
                bits=8,
                calibration_strategy="minmax",
                calibration_samples=128,
                granularity="per_tensor",
            )
        else:  # large, xlarge
            return create_quantization_config(
                method="int8",
                bits=8,
                calibration_strategy="minmax",
                calibration_samples=256,
                granularity="per_channel",
            )

    # Mobile recommendations: aggressive quantization for memory
    if target_device == "mobile":
        return create_quantization_config(
            method="int4",
            bits=4,
            calibration_strategy="percentile",
            calibration_samples=256,
            granularity="per_group",
        )

    # GPU recommendations: GPTQ/AWQ for larger models
    if model_size == "small":
        return create_quantization_config(
            method="dynamic",
            bits=8,
            calibration_strategy="minmax",
            calibration_samples=128,
            granularity="per_tensor",
        )
    elif model_size == "medium":
        return create_quantization_config(
            method="int8",
            bits=8,
            calibration_strategy="minmax",
            calibration_samples=128,
            granularity="per_channel",
        )
    else:  # large, xlarge
        return create_quantization_config(
            method="gptq",
            bits=4,
            calibration_strategy="entropy",
            calibration_samples=256,
            granularity="per_group",
        )
