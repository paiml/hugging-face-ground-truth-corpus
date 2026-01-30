"""Model quantization utilities for deployment.

This module provides utilities for quantizing models for efficient
inference, including GPTQ, AWQ, and dynamic quantization methods.

Examples:
    >>> from hf_gtc.deployment.quantization import QuantMethod, QuantProfile
    >>> profile = QuantProfile(method=QuantMethod.GPTQ, bits=4)
    >>> profile.bits
    4
    >>> profile.method.value
    'gptq'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class QuantMethod(Enum):
    """Quantization method types.

    Examples:
        >>> QuantMethod.GPTQ.value
        'gptq'
        >>> QuantMethod.AWQ.value
        'awq'
    """

    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    DYNAMIC = "dynamic"
    STATIC = "static"


class CalibrationMethod(Enum):
    """Calibration methods for quantization.

    Examples:
        >>> CalibrationMethod.MINMAX.value
        'minmax'
        >>> CalibrationMethod.ENTROPY.value
        'entropy'
    """

    MINMAX = "minmax"
    ENTROPY = "entropy"
    PERCENTILE = "percentile"
    MSE = "mse"


class QuantGranularity(Enum):
    """Quantization granularity levels.

    Examples:
        >>> QuantGranularity.PER_TENSOR.value
        'per_tensor'
        >>> QuantGranularity.PER_CHANNEL.value
        'per_channel'
    """

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


@dataclass(frozen=True, slots=True)
class QuantProfile:
    """Quantization profile configuration.

    Attributes:
        method: Quantization method to use.
        bits: Number of bits for quantization.
        group_size: Group size for group-wise quantization.
        desc_act: Use descending activation order.
        sym: Use symmetric quantization.
        true_sequential: Use true sequential quantization.

    Examples:
        >>> profile = QuantProfile(method=QuantMethod.GPTQ, bits=4)
        >>> profile.bits
        4
        >>> profile.group_size
        128

        >>> profile2 = QuantProfile(method=QuantMethod.AWQ, bits=4, group_size=64)
        >>> profile2.group_size
        64
    """

    method: QuantMethod = QuantMethod.GPTQ
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    sym: bool = True
    true_sequential: bool = True


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Configuration for calibration data.

    Attributes:
        method: Calibration method to use.
        num_samples: Number of calibration samples.
        sequence_length: Sequence length for calibration.
        batch_size: Batch size for calibration.

    Examples:
        >>> config = CalibrationConfig(num_samples=128)
        >>> config.num_samples
        128
        >>> config.method
        <CalibrationMethod.MINMAX: 'minmax'>
    """

    method: CalibrationMethod = CalibrationMethod.MINMAX
    num_samples: int = 128
    sequence_length: int = 2048
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class QuantResult:
    """Result of quantization operation.

    Attributes:
        original_size_mb: Original model size in MB.
        quantized_size_mb: Quantized model size in MB.
        compression_ratio: Compression ratio achieved.
        perplexity_before: Perplexity before quantization.
        perplexity_after: Perplexity after quantization.
        perplexity_degradation: Percentage perplexity degradation.

    Examples:
        >>> result = QuantResult(
        ...     original_size_mb=14000,
        ...     quantized_size_mb=4000,
        ...     compression_ratio=3.5,
        ... )
        >>> result.compression_ratio
        3.5
    """

    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    perplexity_before: float | None = None
    perplexity_after: float | None = None
    perplexity_degradation: float | None = None


@dataclass(frozen=True, slots=True)
class GPTQConfig:
    """GPTQ-specific quantization configuration.

    Attributes:
        bits: Number of bits (typically 4 or 8).
        group_size: Group size for groupwise quantization.
        desc_act: Descending activation order.
        sym: Symmetric quantization.
        damp_percent: Dampening percentage for Hessian.
        static_groups: Use static groups.

    Examples:
        >>> config = GPTQConfig(bits=4, group_size=128)
        >>> config.bits
        4
        >>> config.damp_percent
        0.01
    """

    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    sym: bool = True
    damp_percent: float = 0.01
    static_groups: bool = False


@dataclass(frozen=True, slots=True)
class AWQConfig:
    """AWQ-specific quantization configuration.

    Attributes:
        bits: Number of bits (typically 4).
        group_size: Group size for groupwise quantization.
        zero_point: Use zero point for quantization.
        version: AWQ version to use.

    Examples:
        >>> config = AWQConfig(bits=4, group_size=128)
        >>> config.bits
        4
        >>> config.version
        'gemm'
    """

    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"


def validate_quant_profile(profile: QuantProfile) -> None:
    """Validate quantization profile.

    Args:
        profile: Profile to validate.

    Raises:
        ValueError: If profile is None.
        ValueError: If bits is not valid.
        ValueError: If group_size is not positive.

    Examples:
        >>> profile = QuantProfile(method=QuantMethod.GPTQ, bits=4)
        >>> validate_quant_profile(profile)  # No error

        >>> validate_quant_profile(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: profile cannot be None

        >>> bad = QuantProfile(bits=5)
        >>> validate_quant_profile(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be 2, 3, 4, or 8
    """
    if profile is None:
        msg = "profile cannot be None"
        raise ValueError(msg)

    valid_bits = {2, 3, 4, 8}
    if profile.bits not in valid_bits:
        msg = f"bits must be 2, 3, 4, or 8, got {profile.bits}"
        raise ValueError(msg)

    if profile.group_size <= 0:
        msg = f"group_size must be positive, got {profile.group_size}"
        raise ValueError(msg)


def validate_calibration_config(config: CalibrationConfig) -> None:
    """Validate calibration configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_samples is not positive.
        ValueError: If sequence_length is not positive.

    Examples:
        >>> config = CalibrationConfig(num_samples=128)
        >>> validate_calibration_config(config)  # No error

        >>> validate_calibration_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_samples <= 0:
        msg = f"num_samples must be positive, got {config.num_samples}"
        raise ValueError(msg)

    if config.sequence_length <= 0:
        msg = f"sequence_length must be positive, got {config.sequence_length}"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)


def create_quant_profile(
    method: QuantMethod | str = QuantMethod.GPTQ,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    sym: bool = True,
) -> QuantProfile:
    """Create a quantization profile.

    Args:
        method: Quantization method. Defaults to GPTQ.
        bits: Number of bits. Defaults to 4.
        group_size: Group size. Defaults to 128.
        desc_act: Descending activation order. Defaults to True.
        sym: Symmetric quantization. Defaults to True.

    Returns:
        Validated QuantProfile instance.

    Raises:
        ValueError: If bits is invalid.

    Examples:
        >>> profile = create_quant_profile(method="gptq", bits=4)
        >>> profile.method
        <QuantMethod.GPTQ: 'gptq'>

        >>> profile2 = create_quant_profile(method=QuantMethod.AWQ)
        >>> profile2.method
        <QuantMethod.AWQ: 'awq'>
    """
    if isinstance(method, str):
        method = get_quant_method(method)

    profile = QuantProfile(
        method=method,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
    )
    validate_quant_profile(profile)
    return profile


def create_calibration_config(
    method: CalibrationMethod | str = CalibrationMethod.MINMAX,
    num_samples: int = 128,
    sequence_length: int = 2048,
    batch_size: int = 1,
) -> CalibrationConfig:
    """Create a calibration configuration.

    Args:
        method: Calibration method. Defaults to MINMAX.
        num_samples: Number of samples. Defaults to 128.
        sequence_length: Sequence length. Defaults to 2048.
        batch_size: Batch size. Defaults to 1.

    Returns:
        Validated CalibrationConfig instance.

    Examples:
        >>> config = create_calibration_config(num_samples=256)
        >>> config.num_samples
        256

        >>> config2 = create_calibration_config(method="entropy")
        >>> config2.method
        <CalibrationMethod.ENTROPY: 'entropy'>
    """
    if isinstance(method, str):
        method = get_calibration_method(method)

    config = CalibrationConfig(
        method=method,
        num_samples=num_samples,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    validate_calibration_config(config)
    return config


def create_gptq_config(
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    sym: bool = True,
    damp_percent: float = 0.01,
) -> GPTQConfig:
    """Create a GPTQ configuration.

    Args:
        bits: Number of bits. Defaults to 4.
        group_size: Group size. Defaults to 128.
        desc_act: Descending activation order. Defaults to True.
        sym: Symmetric quantization. Defaults to True.
        damp_percent: Dampening percentage. Defaults to 0.01.

    Returns:
        GPTQConfig instance.

    Raises:
        ValueError: If bits is not 4 or 8.

    Examples:
        >>> config = create_gptq_config(bits=4)
        >>> config.bits
        4
        >>> config.group_size
        128
    """
    if bits not in (4, 8):
        msg = f"bits must be 4 or 8, got {bits}"
        raise ValueError(msg)

    if group_size <= 0:
        msg = f"group_size must be positive, got {group_size}"
        raise ValueError(msg)

    return GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        damp_percent=damp_percent,
    )


def create_awq_config(
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "gemm",
) -> AWQConfig:
    """Create an AWQ configuration.

    Args:
        bits: Number of bits. Defaults to 4.
        group_size: Group size. Defaults to 128.
        zero_point: Use zero point. Defaults to True.
        version: AWQ version. Defaults to "gemm".

    Returns:
        AWQConfig instance.

    Raises:
        ValueError: If bits is not 4.

    Examples:
        >>> config = create_awq_config(bits=4)
        >>> config.bits
        4
        >>> config.version
        'gemm'
    """
    if bits != 4:
        msg = f"AWQ only supports 4-bit quantization, got {bits}"
        raise ValueError(msg)

    if group_size <= 0:
        msg = f"group_size must be positive, got {group_size}"
        raise ValueError(msg)

    valid_versions = {"gemm", "gemv", "marlin"}
    if version not in valid_versions:
        msg = f"version must be one of {valid_versions}, got {version}"
        raise ValueError(msg)

    return AWQConfig(
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
        version=version,
    )


def estimate_quantized_size(
    model_params: int,
    bits: int = 4,
    overhead_factor: float = 1.1,
) -> float:
    """Estimate quantized model size in MB.

    Args:
        model_params: Number of model parameters.
        bits: Quantization bits. Defaults to 4.
        overhead_factor: Overhead factor for metadata. Defaults to 1.1.

    Returns:
        Estimated size in MB.

    Raises:
        ValueError: If model_params is not positive.
        ValueError: If bits is not valid.

    Examples:
        >>> size = estimate_quantized_size(7_000_000_000, bits=4)
        >>> size > 0
        True
        >>> size < 7000  # Much smaller than FP32
        True

        >>> estimate_quantized_size(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    valid_bits = {2, 3, 4, 8, 16, 32}
    if bits not in valid_bits:
        msg = f"bits must be one of {valid_bits}, got {bits}"
        raise ValueError(msg)

    # bytes = params * bits / 8
    size_bytes = model_params * bits / 8
    size_mb = size_bytes / (1024 * 1024)

    return size_mb * overhead_factor


def compute_compression_ratio(
    original_bits: int,
    quantized_bits: int,
) -> float:
    """Compute compression ratio.

    Args:
        original_bits: Original precision bits (e.g., 16 or 32).
        quantized_bits: Quantized precision bits.

    Returns:
        Compression ratio.

    Raises:
        ValueError: If bits are not positive.

    Examples:
        >>> compute_compression_ratio(16, 4)
        4.0
        >>> compute_compression_ratio(32, 4)
        8.0
        >>> compute_compression_ratio(16, 8)
        2.0
    """
    if original_bits <= 0:
        msg = f"original_bits must be positive, got {original_bits}"
        raise ValueError(msg)

    if quantized_bits <= 0:
        msg = f"quantized_bits must be positive, got {quantized_bits}"
        raise ValueError(msg)

    return original_bits / quantized_bits


def create_quant_result(
    original_size_mb: float,
    quantized_size_mb: float,
    perplexity_before: float | None = None,
    perplexity_after: float | None = None,
) -> QuantResult:
    """Create a quantization result.

    Args:
        original_size_mb: Original size in MB.
        quantized_size_mb: Quantized size in MB.
        perplexity_before: Perplexity before quantization.
        perplexity_after: Perplexity after quantization.

    Returns:
        QuantResult instance.

    Raises:
        ValueError: If sizes are not positive.

    Examples:
        >>> result = create_quant_result(14000, 4000)
        >>> result.compression_ratio
        3.5

        >>> result2 = create_quant_result(14000, 4000, 5.0, 5.5)
        >>> result2.perplexity_degradation
        10.0
    """
    if original_size_mb <= 0:
        msg = f"original_size_mb must be positive, got {original_size_mb}"
        raise ValueError(msg)

    if quantized_size_mb <= 0:
        msg = f"quantized_size_mb must be positive, got {quantized_size_mb}"
        raise ValueError(msg)

    compression_ratio = original_size_mb / quantized_size_mb

    perplexity_degradation = None
    if perplexity_before is not None and perplexity_after is not None:
        perplexity_degradation = (
            (perplexity_after - perplexity_before) / perplexity_before * 100
        )

    return QuantResult(
        original_size_mb=original_size_mb,
        quantized_size_mb=quantized_size_mb,
        compression_ratio=compression_ratio,
        perplexity_before=perplexity_before,
        perplexity_after=perplexity_after,
        perplexity_degradation=perplexity_degradation,
    )


def get_gptq_dict(config: GPTQConfig) -> dict[str, Any]:
    """Convert GPTQConfig to transformers-compatible dict.

    Args:
        config: GPTQ configuration.

    Returns:
        Dictionary for GPTQConfig initialization.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_gptq_config(bits=4)
        >>> d = get_gptq_dict(config)
        >>> d["bits"]
        4
        >>> d["group_size"]
        128
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "bits": config.bits,
        "group_size": config.group_size,
        "desc_act": config.desc_act,
        "sym": config.sym,
        "damp_percent": config.damp_percent,
        "static_groups": config.static_groups,
    }


def get_awq_dict(config: AWQConfig) -> dict[str, Any]:
    """Convert AWQConfig to transformers-compatible dict.

    Args:
        config: AWQ configuration.

    Returns:
        Dictionary for AWQConfig initialization.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_awq_config(bits=4)
        >>> d = get_awq_dict(config)
        >>> d["bits"]
        4
        >>> d["zero_point"]
        True
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "bits": config.bits,
        "group_size": config.group_size,
        "zero_point": config.zero_point,
        "version": config.version,
    }


def format_quant_result(result: QuantResult) -> str:
    """Format quantization result for display.

    Args:
        result: Result to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> result = create_quant_result(14000, 4000)
        >>> formatted = format_quant_result(result)
        >>> "Original:" in formatted
        True
        >>> "Compression:" in formatted
        True
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    lines = [
        f"Original: {result.original_size_mb:.1f} MB",
        f"Quantized: {result.quantized_size_mb:.1f} MB",
        f"Compression: {result.compression_ratio:.2f}x",
    ]

    if result.perplexity_before is not None:
        lines.append(f"Perplexity (before): {result.perplexity_before:.2f}")
    if result.perplexity_after is not None:
        lines.append(f"Perplexity (after): {result.perplexity_after:.2f}")
    if result.perplexity_degradation is not None:
        lines.append(f"Perplexity degradation: {result.perplexity_degradation:.1f}%")

    return "\n".join(lines)


def list_quant_methods() -> list[str]:
    """List available quantization methods.

    Returns:
        Sorted list of method names.

    Examples:
        >>> methods = list_quant_methods()
        >>> "gptq" in methods
        True
        >>> "awq" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(m.value for m in QuantMethod)


def validate_quant_method(method: str) -> bool:
    """Check if quantization method is valid.

    Args:
        method: Method to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_quant_method("gptq")
        True
        >>> validate_quant_method("awq")
        True
        >>> validate_quant_method("invalid")
        False
    """
    valid_methods = {m.value for m in QuantMethod}
    return method in valid_methods


def get_quant_method(name: str) -> QuantMethod:
    """Get quantization method enum from string.

    Args:
        name: Method name.

    Returns:
        QuantMethod enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_quant_method("gptq")
        <QuantMethod.GPTQ: 'gptq'>
        >>> get_quant_method("awq")
        <QuantMethod.AWQ: 'awq'>

        >>> get_quant_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid quant method: invalid
    """
    for method in QuantMethod:
        if method.value == name:
            return method
    msg = f"invalid quant method: {name}"
    raise ValueError(msg)


def list_calibration_methods() -> list[str]:
    """List available calibration methods.

    Returns:
        Sorted list of method names.

    Examples:
        >>> methods = list_calibration_methods()
        >>> "minmax" in methods
        True
        >>> "entropy" in methods
        True
    """
    return sorted(m.value for m in CalibrationMethod)


def validate_calibration_method(method: str) -> bool:
    """Check if calibration method is valid.

    Args:
        method: Method to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_calibration_method("minmax")
        True
        >>> validate_calibration_method("entropy")
        True
        >>> validate_calibration_method("invalid")
        False
    """
    valid_methods = {m.value for m in CalibrationMethod}
    return method in valid_methods


def get_calibration_method(name: str) -> CalibrationMethod:
    """Get calibration method enum from string.

    Args:
        name: Method name.

    Returns:
        CalibrationMethod enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_calibration_method("minmax")
        <CalibrationMethod.MINMAX: 'minmax'>
        >>> get_calibration_method("entropy")
        <CalibrationMethod.ENTROPY: 'entropy'>

        >>> get_calibration_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid calibration method: invalid
    """
    for method in CalibrationMethod:
        if method.value == name:
            return method
    msg = f"invalid calibration method: {name}"
    raise ValueError(msg)


def list_quant_granularities() -> list[str]:
    """List available quantization granularities.

    Returns:
        Sorted list of granularity names.

    Examples:
        >>> granularities = list_quant_granularities()
        >>> "per_tensor" in granularities
        True
        >>> "per_channel" in granularities
        True
    """
    return sorted(g.value for g in QuantGranularity)


def validate_quant_granularity(granularity: str) -> bool:
    """Check if quantization granularity is valid.

    Args:
        granularity: Granularity to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_quant_granularity("per_tensor")
        True
        >>> validate_quant_granularity("per_channel")
        True
        >>> validate_quant_granularity("invalid")
        False
    """
    valid_granularities = {g.value for g in QuantGranularity}
    return granularity in valid_granularities


def get_quant_granularity(name: str) -> QuantGranularity:
    """Get quantization granularity enum from string.

    Args:
        name: Granularity name.

    Returns:
        QuantGranularity enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_quant_granularity("per_tensor")
        <QuantGranularity.PER_TENSOR: 'per_tensor'>
        >>> get_quant_granularity("per_channel")
        <QuantGranularity.PER_CHANNEL: 'per_channel'>

        >>> get_quant_granularity("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid granularity: invalid
    """
    for granularity in QuantGranularity:
        if granularity.value == name:
            return granularity
    msg = f"invalid granularity: {name}"
    raise ValueError(msg)


def get_recommended_profile(model_size: str) -> QuantProfile:
    """Get recommended quantization profile for model size.

    Args:
        model_size: Model size string (e.g., "7b", "13b", "70b").

    Returns:
        Recommended QuantProfile.

    Raises:
        ValueError: If model_size is not recognized.

    Examples:
        >>> profile = get_recommended_profile("7b")
        >>> profile.method
        <QuantMethod.GPTQ: 'gptq'>
        >>> profile.bits
        4

        >>> profile_70b = get_recommended_profile("70b")
        >>> profile_70b.group_size
        64
    """
    model_size = model_size.lower().strip()

    if model_size in ("7b", "7B", "7") or model_size in ("13b", "13B", "13"):
        return create_quant_profile(method=QuantMethod.GPTQ, bits=4, group_size=128)
    elif model_size in ("70b", "70B", "70"):
        return create_quant_profile(method=QuantMethod.GPTQ, bits=4, group_size=64)
    else:
        msg = f"unrecognized model size: {model_size}"
        raise ValueError(msg)
