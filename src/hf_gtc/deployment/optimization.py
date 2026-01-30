"""Model optimization utilities for deployment.

This module provides functions for model quantization, pruning,
and other optimization techniques for efficient deployment.

Examples:
    >>> from hf_gtc.deployment.optimization import get_quantization_config
    >>> config = get_quantization_config("int8")
    >>> config.load_in_8bit
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class QuantizationType(Enum):
    """Supported quantization types.

    Attributes:
        INT8: 8-bit integer quantization.
        INT4: 4-bit integer quantization.
        FP16: Half-precision floating point.
        BF16: Brain floating point 16.
        NONE: No quantization.

    Examples:
        >>> QuantizationType.INT8.value
        'int8'
        >>> QuantizationType.FP16.value
        'fp16'
    """

    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"
    NONE = "none"


VALID_QUANTIZATION_TYPES = frozenset(q.value for q in QuantizationType)


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        quantization_type: Type of quantization to apply.
        load_in_8bit: Whether to load model in 8-bit mode.
        load_in_4bit: Whether to load model in 4-bit mode.
        torch_dtype: PyTorch data type for model weights.
        device_map: Device placement strategy.

    Examples:
        >>> config = QuantizationConfig(
        ...     quantization_type=QuantizationType.INT8,
        ...     load_in_8bit=True,
        ...     load_in_4bit=False,
        ...     torch_dtype="float16",
        ...     device_map="auto",
        ... )
        >>> config.load_in_8bit
        True
    """

    quantization_type: QuantizationType
    load_in_8bit: bool
    load_in_4bit: bool
    torch_dtype: str
    device_map: str


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Result of model optimization.

    Attributes:
        original_size_mb: Original model size in megabytes.
        optimized_size_mb: Optimized model size in megabytes.
        compression_ratio: Ratio of original to optimized size.
        quantization_type: Type of quantization applied.

    Examples:
        >>> result = OptimizationResult(
        ...     original_size_mb=1000.0,
        ...     optimized_size_mb=250.0,
        ...     compression_ratio=4.0,
        ...     quantization_type=QuantizationType.INT8,
        ... )
        >>> result.compression_ratio
        4.0
    """

    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    quantization_type: QuantizationType


def get_quantization_config(
    quantization_type: str,
    device_map: str = "auto",
) -> QuantizationConfig:
    """Get quantization configuration for a given type.

    Args:
        quantization_type: Type of quantization ("int8", "int4", "fp16", "bf16", "none").
        device_map: Device placement strategy. Defaults to "auto".

    Returns:
        QuantizationConfig with appropriate settings.

    Raises:
        ValueError: If quantization_type is not valid.

    Examples:
        >>> config = get_quantization_config("int8")
        >>> config.load_in_8bit
        True
        >>> config.torch_dtype
        'float16'

        >>> config = get_quantization_config("int4")
        >>> config.load_in_4bit
        True

        >>> config = get_quantization_config("fp16")
        >>> config.torch_dtype
        'float16'

        >>> get_quantization_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quantization_type must be one of...
    """
    if quantization_type not in VALID_QUANTIZATION_TYPES:
        msg = f"quantization_type must be one of {VALID_QUANTIZATION_TYPES}, got '{quantization_type}'"
        raise ValueError(msg)

    qtype = QuantizationType(quantization_type)

    if qtype == QuantizationType.INT8:
        return QuantizationConfig(
            quantization_type=qtype,
            load_in_8bit=True,
            load_in_4bit=False,
            torch_dtype="float16",
            device_map=device_map,
        )
    elif qtype == QuantizationType.INT4:
        return QuantizationConfig(
            quantization_type=qtype,
            load_in_8bit=False,
            load_in_4bit=True,
            torch_dtype="float16",
            device_map=device_map,
        )
    elif qtype == QuantizationType.FP16:
        return QuantizationConfig(
            quantization_type=qtype,
            load_in_8bit=False,
            load_in_4bit=False,
            torch_dtype="float16",
            device_map=device_map,
        )
    elif qtype == QuantizationType.BF16:
        return QuantizationConfig(
            quantization_type=qtype,
            load_in_8bit=False,
            load_in_4bit=False,
            torch_dtype="bfloat16",
            device_map=device_map,
        )
    else:  # NONE
        return QuantizationConfig(
            quantization_type=qtype,
            load_in_8bit=False,
            load_in_4bit=False,
            torch_dtype="float32",
            device_map=device_map,
        )


def estimate_model_size(
    num_parameters: int,
    quantization_type: str = "none",
) -> float:
    """Estimate model size in megabytes based on parameter count.

    Args:
        num_parameters: Number of model parameters.
        quantization_type: Type of quantization. Defaults to "none".

    Returns:
        Estimated model size in megabytes.

    Raises:
        ValueError: If num_parameters is not positive.
        ValueError: If quantization_type is not valid.

    Examples:
        >>> estimate_model_size(1_000_000_000, "none")
        4000.0
        >>> estimate_model_size(1_000_000_000, "fp16")
        2000.0
        >>> estimate_model_size(1_000_000_000, "int8")
        1000.0
        >>> estimate_model_size(1_000_000_000, "int4")
        500.0

        >>> estimate_model_size(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive
    """
    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    if quantization_type not in VALID_QUANTIZATION_TYPES:
        msg = f"quantization_type must be one of {VALID_QUANTIZATION_TYPES}, got '{quantization_type}'"
        raise ValueError(msg)

    # Bytes per parameter based on quantization type
    bytes_per_param = {
        "none": 4,  # float32
        "fp16": 2,  # float16
        "bf16": 2,  # bfloat16
        "int8": 1,  # int8
        "int4": 0.5,  # int4
    }

    bytes_total = num_parameters * bytes_per_param[quantization_type]
    return bytes_total / (1024 * 1024)  # Convert to MB


def calculate_compression_ratio(
    original_size: float,
    optimized_size: float,
) -> float:
    """Calculate compression ratio between original and optimized sizes.

    Args:
        original_size: Original size (any unit).
        optimized_size: Optimized size (same unit).

    Returns:
        Compression ratio (original / optimized).

    Raises:
        ValueError: If either size is not positive.

    Examples:
        >>> calculate_compression_ratio(1000.0, 250.0)
        4.0
        >>> calculate_compression_ratio(100.0, 50.0)
        2.0

        >>> calculate_compression_ratio(0, 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_size must be positive
    """
    if original_size <= 0:
        msg = f"original_size must be positive, got {original_size}"
        raise ValueError(msg)

    if optimized_size <= 0:
        msg = f"optimized_size must be positive, got {optimized_size}"
        raise ValueError(msg)

    return original_size / optimized_size


def get_optimization_result(
    num_parameters: int,
    quantization_type: str,
) -> OptimizationResult:
    """Get optimization result for a model with given parameters.

    Args:
        num_parameters: Number of model parameters.
        quantization_type: Type of quantization to apply.

    Returns:
        OptimizationResult with size estimates and compression ratio.

    Raises:
        ValueError: If num_parameters is not positive.
        ValueError: If quantization_type is not valid.

    Examples:
        >>> result = get_optimization_result(1_000_000_000, "int8")
        >>> result.original_size_mb
        4000.0
        >>> result.optimized_size_mb
        1000.0
        >>> result.compression_ratio
        4.0
    """
    original_size = estimate_model_size(num_parameters, "none")
    optimized_size = estimate_model_size(num_parameters, quantization_type)
    compression = calculate_compression_ratio(original_size, optimized_size)

    return OptimizationResult(
        original_size_mb=original_size,
        optimized_size_mb=optimized_size,
        compression_ratio=compression,
        quantization_type=QuantizationType(quantization_type),
    )


def get_model_loading_kwargs(config: QuantizationConfig) -> dict[str, Any]:
    """Get keyword arguments for model loading based on quantization config.

    Args:
        config: Quantization configuration.

    Returns:
        Dictionary of kwargs to pass to model.from_pretrained().

    Examples:
        >>> config = get_quantization_config("int8")
        >>> kwargs = get_model_loading_kwargs(config)
        >>> kwargs["load_in_8bit"]
        True
        >>> kwargs["device_map"]
        'auto'
    """
    kwargs: dict[str, Any] = {
        "device_map": config.device_map,
    }

    if config.load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif config.load_in_4bit:
        kwargs["load_in_4bit"] = True

    # Only set torch_dtype if not using int8/int4 quantization
    if not config.load_in_8bit and not config.load_in_4bit:
        kwargs["torch_dtype"] = config.torch_dtype

    return kwargs


def list_quantization_types() -> list[str]:
    """List all supported quantization types.

    Returns:
        Sorted list of quantization type names.

    Examples:
        >>> types = list_quantization_types()
        >>> "int8" in types
        True
        >>> "fp16" in types
        True
        >>> len(types)
        5
    """
    return sorted(VALID_QUANTIZATION_TYPES)
