"""TensorFlow Lite conversion utilities for mobile and edge deployment.

This module provides utilities for converting HuggingFace models to
TensorFlow Lite format for deployment on mobile devices, edge hardware,
and embedded systems.

Examples:
    >>> from hf_gtc.deployment.tflite import TFLiteQuantization, TFLiteDelegate
    >>> config = create_convert_config(quantization=TFLiteQuantization.DYNAMIC)
    >>> config.quantization.value
    'dynamic'
    >>> config.optimization_target.value
    'default'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class TFLiteQuantization(Enum):
    """TFLite quantization methods.

    Quantization reduces model size and improves inference speed
    by converting floating point weights to lower precision formats.

    Examples:
        >>> TFLiteQuantization.NONE.value
        'none'
        >>> TFLiteQuantization.DYNAMIC.value
        'dynamic'
        >>> TFLiteQuantization.FULL_INTEGER.value
        'full_integer'
        >>> TFLiteQuantization.FLOAT16.value
        'float16'
    """

    NONE = "none"
    DYNAMIC = "dynamic"
    FULL_INTEGER = "full_integer"
    FLOAT16 = "float16"


class TFLiteDelegate(Enum):
    """TFLite hardware delegates for accelerated inference.

    Delegates enable hardware acceleration on specific platforms
    like GPUs, DSPs, and neural accelerators.

    Examples:
        >>> TFLiteDelegate.NONE.value
        'none'
        >>> TFLiteDelegate.GPU.value
        'gpu'
        >>> TFLiteDelegate.NNAPI.value
        'nnapi'
        >>> TFLiteDelegate.XNNPACK.value
        'xnnpack'
        >>> TFLiteDelegate.COREML.value
        'coreml'
        >>> TFLiteDelegate.HEXAGON.value
        'hexagon'
    """

    NONE = "none"
    GPU = "gpu"
    NNAPI = "nnapi"
    XNNPACK = "xnnpack"
    COREML = "coreml"
    HEXAGON = "hexagon"


class OptimizationTarget(Enum):
    """Optimization targets for TFLite conversion.

    These targets guide the converter to optimize for different
    deployment requirements.

    Examples:
        >>> OptimizationTarget.DEFAULT.value
        'default'
        >>> OptimizationTarget.SIZE.value
        'size'
        >>> OptimizationTarget.LATENCY.value
        'latency'
    """

    DEFAULT = "default"
    SIZE = "size"
    LATENCY = "latency"


@dataclass(frozen=True, slots=True)
class TFLiteConvertConfig:
    """Configuration for TFLite model conversion.

    Attributes:
        quantization: Quantization method to apply.
        optimization_target: Optimization priority.
        supported_ops: Set of supported operation types.
        allow_custom_ops: Allow custom TensorFlow operations.

    Examples:
        >>> config = TFLiteConvertConfig(
        ...     quantization=TFLiteQuantization.DYNAMIC
        ... )
        >>> config.quantization.value
        'dynamic'
        >>> config.allow_custom_ops
        False

        >>> config2 = TFLiteConvertConfig(
        ...     quantization=TFLiteQuantization.FULL_INTEGER,
        ...     optimization_target=OptimizationTarget.SIZE,
        ...     allow_custom_ops=True,
        ... )
        >>> config2.optimization_target.value
        'size'
    """

    quantization: TFLiteQuantization = TFLiteQuantization.NONE
    optimization_target: OptimizationTarget = OptimizationTarget.DEFAULT
    supported_ops: frozenset[str] = frozenset({"TFLITE_BUILTINS"})
    allow_custom_ops: bool = False


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Configuration for TFLite quantization calibration.

    Attributes:
        representative_dataset_size: Number of samples for calibration.
        num_calibration_steps: Number of calibration iterations.
        output_type: Output tensor data type.

    Examples:
        >>> config = QuantizationConfig(representative_dataset_size=100)
        >>> config.representative_dataset_size
        100
        >>> config.num_calibration_steps
        100
        >>> config.output_type
        'int8'

        >>> config2 = QuantizationConfig(
        ...     representative_dataset_size=200,
        ...     num_calibration_steps=50,
        ...     output_type="uint8",
        ... )
        >>> config2.output_type
        'uint8'
    """

    representative_dataset_size: int = 100
    num_calibration_steps: int = 100
    output_type: str = "int8"


@dataclass(frozen=True, slots=True)
class DelegateConfig:
    """Configuration for TFLite hardware delegate.

    Attributes:
        delegate_type: Type of hardware delegate.
        num_threads: Number of threads for CPU inference.
        enable_fallback: Fall back to CPU for unsupported ops.

    Examples:
        >>> config = DelegateConfig(delegate_type=TFLiteDelegate.XNNPACK)
        >>> config.delegate_type.value
        'xnnpack'
        >>> config.num_threads
        4
        >>> config.enable_fallback
        True

        >>> config2 = DelegateConfig(
        ...     delegate_type=TFLiteDelegate.GPU,
        ...     num_threads=8,
        ...     enable_fallback=False,
        ... )
        >>> config2.num_threads
        8
    """

    delegate_type: TFLiteDelegate = TFLiteDelegate.NONE
    num_threads: int = 4
    enable_fallback: bool = True


@dataclass(frozen=True, slots=True)
class TFLiteModelInfo:
    """Information about a TFLite model file.

    Attributes:
        model_path: Path to the TFLite model file.
        input_details: List of input tensor details.
        output_details: List of output tensor details.
        model_size_bytes: Model file size in bytes.

    Examples:
        >>> info = TFLiteModelInfo(
        ...     model_path="/models/model.tflite",
        ...     input_details=[{"name": "input", "shape": [1, 224, 224, 3]}],
        ...     output_details=[{"name": "output", "shape": [1, 1000]}],
        ...     model_size_bytes=4_000_000,
        ... )
        >>> info.model_path
        '/models/model.tflite'
        >>> info.model_size_bytes
        4000000

        >>> info2 = TFLiteModelInfo(
        ...     model_path="/models/bert.tflite",
        ...     input_details=[],
        ...     output_details=[],
        ...     model_size_bytes=100_000_000,
        ... )
        >>> len(info2.input_details)
        0
    """

    model_path: str
    input_details: list[dict[str, Any]]
    output_details: list[dict[str, Any]]
    model_size_bytes: int


@dataclass(frozen=True, slots=True)
class ConversionStats:
    """Statistics from TFLite model conversion.

    Attributes:
        conversion_time_seconds: Time taken for conversion.
        original_size: Original model size in bytes.
        converted_size: Converted model size in bytes.
        compression_ratio: Size reduction ratio.

    Examples:
        >>> stats = ConversionStats(
        ...     conversion_time_seconds=10.5,
        ...     original_size=100_000_000,
        ...     converted_size=25_000_000,
        ...     compression_ratio=4.0,
        ... )
        >>> stats.compression_ratio
        4.0
        >>> stats.conversion_time_seconds
        10.5

        >>> stats2 = ConversionStats(
        ...     conversion_time_seconds=5.0,
        ...     original_size=50_000_000,
        ...     converted_size=50_000_000,
        ...     compression_ratio=1.0,
        ... )
        >>> stats2.compression_ratio
        1.0
    """

    conversion_time_seconds: float
    original_size: int
    converted_size: int
    compression_ratio: float


def validate_convert_config(config: TFLiteConvertConfig) -> None:
    """Validate TFLite conversion configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If supported_ops is empty.

    Examples:
        >>> config = TFLiteConvertConfig()
        >>> validate_convert_config(config)  # No error

        >>> validate_convert_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = TFLiteConvertConfig(supported_ops=frozenset())
        >>> validate_convert_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: supported_ops cannot be empty
    """
    validate_not_none(config, "config")

    if not config.supported_ops:
        msg = "supported_ops cannot be empty"
        raise ValueError(msg)


def validate_quantization_config(config: QuantizationConfig) -> None:
    """Validate TFLite quantization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If representative_dataset_size is not positive.
        ValueError: If num_calibration_steps is not positive.
        ValueError: If output_type is not valid.

    Examples:
        >>> config = QuantizationConfig()
        >>> validate_quantization_config(config)  # No error

        >>> validate_quantization_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = QuantizationConfig(representative_dataset_size=0)
        >>> validate_quantization_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: representative_dataset_size must be positive

        >>> bad2 = QuantizationConfig(num_calibration_steps=-1)
        >>> validate_quantization_config(bad2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_calibration_steps must be positive

        >>> bad3 = QuantizationConfig(output_type="invalid")
        >>> validate_quantization_config(bad3)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: output_type must be one of {'int8', 'uint8', 'float16'}
    """
    validate_not_none(config, "config")

    if config.representative_dataset_size <= 0:
        msg = (
            f"representative_dataset_size must be positive, "
            f"got {config.representative_dataset_size}"
        )
        raise ValueError(msg)

    if config.num_calibration_steps <= 0:
        msg = (
            f"num_calibration_steps must be positive, "
            f"got {config.num_calibration_steps}"
        )
        raise ValueError(msg)

    valid_output_types = {"int8", "uint8", "float16"}
    if config.output_type not in valid_output_types:
        msg = (
            f"output_type must be one of {valid_output_types}, got {config.output_type}"
        )
        raise ValueError(msg)


def validate_delegate_config(config: DelegateConfig) -> None:
    """Validate TFLite delegate configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_threads is not positive.

    Examples:
        >>> config = DelegateConfig()
        >>> validate_delegate_config(config)  # No error

        >>> validate_delegate_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = DelegateConfig(num_threads=0)
        >>> validate_delegate_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_threads must be positive
    """
    validate_not_none(config, "config")

    if config.num_threads <= 0:
        msg = f"num_threads must be positive, got {config.num_threads}"
        raise ValueError(msg)


def create_convert_config(
    quantization: TFLiteQuantization | str = TFLiteQuantization.NONE,
    optimization_target: OptimizationTarget | str = OptimizationTarget.DEFAULT,
    supported_ops: frozenset[str] | None = None,
    allow_custom_ops: bool = False,
) -> TFLiteConvertConfig:
    """Create a TFLite conversion configuration.

    Args:
        quantization: Quantization method. Defaults to NONE.
        optimization_target: Optimization priority. Defaults to DEFAULT.
        supported_ops: Set of supported ops. Defaults to TFLITE_BUILTINS.
        allow_custom_ops: Allow custom ops. Defaults to False.

    Returns:
        Validated TFLiteConvertConfig instance.

    Raises:
        ValueError: If supported_ops is empty.

    Examples:
        >>> config = create_convert_config(quantization="dynamic")
        >>> config.quantization
        <TFLiteQuantization.DYNAMIC: 'dynamic'>

        >>> config2 = create_convert_config(
        ...     quantization=TFLiteQuantization.FULL_INTEGER,
        ...     optimization_target="size",
        ... )
        >>> config2.optimization_target
        <OptimizationTarget.SIZE: 'size'>

        >>> config3 = create_convert_config(
        ...     supported_ops=frozenset({"TFLITE_BUILTINS", "SELECT_TF_OPS"})
        ... )
        >>> "SELECT_TF_OPS" in config3.supported_ops
        True
    """
    if isinstance(quantization, str):
        quantization = get_quantization_type(quantization)
    if isinstance(optimization_target, str):
        optimization_target = get_optimization_target(optimization_target)

    if supported_ops is None:
        supported_ops = frozenset({"TFLITE_BUILTINS"})

    config = TFLiteConvertConfig(
        quantization=quantization,
        optimization_target=optimization_target,
        supported_ops=supported_ops,
        allow_custom_ops=allow_custom_ops,
    )
    validate_convert_config(config)
    return config


def create_quantization_config(
    representative_dataset_size: int = 100,
    num_calibration_steps: int = 100,
    output_type: str = "int8",
) -> QuantizationConfig:
    """Create a TFLite quantization configuration.

    Args:
        representative_dataset_size: Calibration samples. Defaults to 100.
        num_calibration_steps: Calibration iterations. Defaults to 100.
        output_type: Output data type. Defaults to "int8".

    Returns:
        Validated QuantizationConfig instance.

    Raises:
        ValueError: If representative_dataset_size is not positive.
        ValueError: If num_calibration_steps is not positive.
        ValueError: If output_type is invalid.

    Examples:
        >>> config = create_quantization_config(representative_dataset_size=200)
        >>> config.representative_dataset_size
        200

        >>> config2 = create_quantization_config(output_type="uint8")
        >>> config2.output_type
        'uint8'

        >>> create_quantization_config(representative_dataset_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: representative_dataset_size must be positive
    """
    config = QuantizationConfig(
        representative_dataset_size=representative_dataset_size,
        num_calibration_steps=num_calibration_steps,
        output_type=output_type,
    )
    validate_quantization_config(config)
    return config


def create_delegate_config(
    delegate_type: TFLiteDelegate | str = TFLiteDelegate.NONE,
    num_threads: int = 4,
    enable_fallback: bool = True,
) -> DelegateConfig:
    """Create a TFLite delegate configuration.

    Args:
        delegate_type: Hardware delegate type. Defaults to NONE.
        num_threads: CPU threads. Defaults to 4.
        enable_fallback: Enable CPU fallback. Defaults to True.

    Returns:
        Validated DelegateConfig instance.

    Raises:
        ValueError: If num_threads is not positive.

    Examples:
        >>> config = create_delegate_config(delegate_type="xnnpack")
        >>> config.delegate_type
        <TFLiteDelegate.XNNPACK: 'xnnpack'>

        >>> config2 = create_delegate_config(num_threads=8)
        >>> config2.num_threads
        8

        >>> create_delegate_config(num_threads=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_threads must be positive
    """
    if isinstance(delegate_type, str):
        delegate_type = get_delegate_type(delegate_type)

    config = DelegateConfig(
        delegate_type=delegate_type,
        num_threads=num_threads,
        enable_fallback=enable_fallback,
    )
    validate_delegate_config(config)
    return config


def list_quantization_types() -> list[str]:
    """List available TFLite quantization types.

    Returns:
        Sorted list of quantization type names.

    Examples:
        >>> types = list_quantization_types()
        >>> "dynamic" in types
        True
        >>> "full_integer" in types
        True
        >>> "none" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(q.value for q in TFLiteQuantization)


def list_delegate_types() -> list[str]:
    """List available TFLite delegate types.

    Returns:
        Sorted list of delegate type names.

    Examples:
        >>> types = list_delegate_types()
        >>> "gpu" in types
        True
        >>> "xnnpack" in types
        True
        >>> "nnapi" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(d.value for d in TFLiteDelegate)


def list_optimization_targets() -> list[str]:
    """List available optimization targets.

    Returns:
        Sorted list of optimization target names.

    Examples:
        >>> targets = list_optimization_targets()
        >>> "default" in targets
        True
        >>> "size" in targets
        True
        >>> "latency" in targets
        True
        >>> targets == sorted(targets)
        True
    """
    return sorted(o.value for o in OptimizationTarget)


def get_quantization_type(name: str) -> TFLiteQuantization:
    """Get TFLite quantization type enum from string.

    Args:
        name: Quantization type name.

    Returns:
        TFLiteQuantization enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_quantization_type("dynamic")
        <TFLiteQuantization.DYNAMIC: 'dynamic'>
        >>> get_quantization_type("full_integer")
        <TFLiteQuantization.FULL_INTEGER: 'full_integer'>
        >>> get_quantization_type("none")
        <TFLiteQuantization.NONE: 'none'>

        >>> get_quantization_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid quantization type: invalid
    """
    for qt in TFLiteQuantization:
        if qt.value == name:
            return qt
    msg = f"invalid quantization type: {name}"
    raise ValueError(msg)


def get_delegate_type(name: str) -> TFLiteDelegate:
    """Get TFLite delegate type enum from string.

    Args:
        name: Delegate type name.

    Returns:
        TFLiteDelegate enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_delegate_type("gpu")
        <TFLiteDelegate.GPU: 'gpu'>
        >>> get_delegate_type("xnnpack")
        <TFLiteDelegate.XNNPACK: 'xnnpack'>
        >>> get_delegate_type("nnapi")
        <TFLiteDelegate.NNAPI: 'nnapi'>

        >>> get_delegate_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid delegate type: invalid
    """
    for dt in TFLiteDelegate:
        if dt.value == name:
            return dt
    msg = f"invalid delegate type: {name}"
    raise ValueError(msg)


def get_optimization_target(name: str) -> OptimizationTarget:
    """Get optimization target enum from string.

    Args:
        name: Target name.

    Returns:
        OptimizationTarget enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_optimization_target("default")
        <OptimizationTarget.DEFAULT: 'default'>
        >>> get_optimization_target("size")
        <OptimizationTarget.SIZE: 'size'>
        >>> get_optimization_target("latency")
        <OptimizationTarget.LATENCY: 'latency'>

        >>> get_optimization_target("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid optimization target: invalid
    """
    for ot in OptimizationTarget:
        if ot.value == name:
            return ot
    msg = f"invalid optimization target: {name}"
    raise ValueError(msg)


def estimate_tflite_size(
    model_params: int,
    quantization: TFLiteQuantization = TFLiteQuantization.NONE,
    overhead_factor: float = 1.05,
) -> float:
    """Estimate TFLite model file size.

    Args:
        model_params: Number of model parameters.
        quantization: Quantization type to apply. Defaults to NONE.
        overhead_factor: Overhead for TFLite format. Defaults to 1.05.

    Returns:
        Estimated size in bytes.

    Raises:
        ValueError: If model_params is not positive.
        ValueError: If overhead_factor is less than 1.0.

    Examples:
        >>> size = estimate_tflite_size(1_000_000, TFLiteQuantization.NONE)
        >>> size > 0
        True

        >>> size_dynamic = estimate_tflite_size(
        ...     1_000_000, TFLiteQuantization.DYNAMIC
        ... )
        >>> size_dynamic < size  # Dynamic quantization reduces size
        True

        >>> size_int8 = estimate_tflite_size(
        ...     1_000_000, TFLiteQuantization.FULL_INTEGER
        ... )
        >>> size_int8 < size_dynamic  # Full integer is smallest
        True

        >>> estimate_tflite_size(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive

        >>> estimate_tflite_size(1000, overhead_factor=0.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: overhead_factor must be >= 1.0
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    if overhead_factor < 1.0:
        msg = f"overhead_factor must be >= 1.0, got {overhead_factor}"
        raise ValueError(msg)

    # Bytes per parameter based on quantization
    bytes_per_param = {
        TFLiteQuantization.NONE: 4.0,  # FP32
        TFLiteQuantization.FLOAT16: 2.0,  # FP16
        TFLiteQuantization.DYNAMIC: 1.5,  # Mixed INT8/FP32
        TFLiteQuantization.FULL_INTEGER: 1.0,  # INT8
    }

    bpp = bytes_per_param.get(quantization, 4.0)
    size_bytes = model_params * bpp * overhead_factor

    return size_bytes


def calculate_compression_ratio(
    original_size: int,
    converted_size: int,
) -> float:
    """Calculate compression ratio between original and converted sizes.

    Args:
        original_size: Original model size in bytes.
        converted_size: Converted model size in bytes.

    Returns:
        Compression ratio (original / converted).

    Raises:
        ValueError: If original_size is not positive.
        ValueError: If converted_size is not positive.

    Examples:
        >>> calculate_compression_ratio(100_000_000, 25_000_000)
        4.0
        >>> calculate_compression_ratio(50_000_000, 50_000_000)
        1.0
        >>> ratio = calculate_compression_ratio(100_000_000, 12_500_000)
        >>> ratio
        8.0

        >>> calculate_compression_ratio(0, 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_size must be positive

        >>> calculate_compression_ratio(100, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: converted_size must be positive
    """
    if original_size <= 0:
        msg = f"original_size must be positive, got {original_size}"
        raise ValueError(msg)

    if converted_size <= 0:
        msg = f"converted_size must be positive, got {converted_size}"
        raise ValueError(msg)

    return original_size / converted_size


def create_conversion_stats(
    conversion_time_seconds: float,
    original_size: int,
    converted_size: int,
) -> ConversionStats:
    """Create conversion statistics.

    Args:
        conversion_time_seconds: Conversion duration.
        original_size: Original model size in bytes.
        converted_size: Converted model size in bytes.

    Returns:
        ConversionStats instance.

    Raises:
        ValueError: If conversion_time_seconds is negative.
        ValueError: If original_size is not positive.
        ValueError: If converted_size is not positive.

    Examples:
        >>> stats = create_conversion_stats(10.5, 100_000_000, 25_000_000)
        >>> stats.compression_ratio
        4.0
        >>> stats.conversion_time_seconds
        10.5

        >>> create_conversion_stats(-1.0, 100, 50)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: conversion_time_seconds cannot be negative
    """
    if conversion_time_seconds < 0:
        msg = (
            f"conversion_time_seconds cannot be negative, got {conversion_time_seconds}"
        )
        raise ValueError(msg)

    compression_ratio = calculate_compression_ratio(original_size, converted_size)

    return ConversionStats(
        conversion_time_seconds=conversion_time_seconds,
        original_size=original_size,
        converted_size=converted_size,
        compression_ratio=compression_ratio,
    )


def create_model_info(
    model_path: str,
    input_details: list[dict[str, Any]],
    output_details: list[dict[str, Any]],
    model_size_bytes: int,
) -> TFLiteModelInfo:
    """Create TFLite model information.

    Args:
        model_path: Path to the model file.
        input_details: Input tensor details.
        output_details: Output tensor details.
        model_size_bytes: Model file size.

    Returns:
        TFLiteModelInfo instance.

    Raises:
        ValueError: If model_path is empty.
        ValueError: If model_size_bytes is not positive.

    Examples:
        >>> info = create_model_info(
        ...     "/models/model.tflite",
        ...     [{"name": "input", "shape": [1, 224, 224, 3]}],
        ...     [{"name": "output", "shape": [1, 1000]}],
        ...     4_000_000,
        ... )
        >>> info.model_path
        '/models/model.tflite'
        >>> len(info.input_details)
        1

        >>> create_model_info("", [], [], 100)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_path cannot be empty

        >>> create_model_info("/model.tflite", [], [], 0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size_bytes must be positive
    """
    if not model_path:
        msg = "model_path cannot be empty"
        raise ValueError(msg)

    if model_size_bytes <= 0:
        msg = f"model_size_bytes must be positive, got {model_size_bytes}"
        raise ValueError(msg)

    return TFLiteModelInfo(
        model_path=model_path,
        input_details=input_details,
        output_details=output_details,
        model_size_bytes=model_size_bytes,
    )


def format_model_info(info: TFLiteModelInfo) -> str:
    """Format TFLite model info for display.

    Args:
        info: Model info to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If info is None.

    Examples:
        >>> info = TFLiteModelInfo(
        ...     model_path="/models/model.tflite",
        ...     input_details=[{"name": "input"}],
        ...     output_details=[{"name": "output"}],
        ...     model_size_bytes=4_000_000,
        ... )
        >>> formatted = format_model_info(info)
        >>> "Path:" in formatted
        True
        >>> "Size:" in formatted
        True
        >>> "Inputs:" in formatted
        True

        >>> format_model_info(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: info cannot be None
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    size_mb = info.model_size_bytes / (1024 * 1024)
    lines = [
        f"Path: {info.model_path}",
        f"Size: {size_mb:.2f} MB",
        f"Inputs: {len(info.input_details)}",
        f"Outputs: {len(info.output_details)}",
    ]

    return "\n".join(lines)


def format_conversion_stats(stats: ConversionStats) -> str:
    """Format conversion statistics for display.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = ConversionStats(
        ...     conversion_time_seconds=10.5,
        ...     original_size=100_000_000,
        ...     converted_size=25_000_000,
        ...     compression_ratio=4.0,
        ... )
        >>> formatted = format_conversion_stats(stats)
        >>> "Conversion time:" in formatted
        True
        >>> "Compression:" in formatted
        True

        >>> format_conversion_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    original_mb = stats.original_size / (1024 * 1024)
    converted_mb = stats.converted_size / (1024 * 1024)

    lines = [
        f"Conversion time: {stats.conversion_time_seconds:.2f}s",
        f"Original size: {original_mb:.2f} MB",
        f"Converted size: {converted_mb:.2f} MB",
        f"Compression: {stats.compression_ratio:.2f}x",
    ]

    return "\n".join(lines)


def get_recommended_config(
    target_device: str = "mobile",
    optimize_for: str = "balanced",
) -> TFLiteConvertConfig:
    """Get recommended conversion config for target device.

    Args:
        target_device: Target device type ("mobile", "edge", "server").
        optimize_for: Optimization priority ("size", "latency", "balanced").

    Returns:
        Recommended TFLiteConvertConfig.

    Raises:
        ValueError: If target_device is not recognized.
        ValueError: If optimize_for is not recognized.

    Examples:
        >>> config = get_recommended_config("mobile", "size")
        >>> config.quantization
        <TFLiteQuantization.FULL_INTEGER: 'full_integer'>
        >>> config.optimization_target
        <OptimizationTarget.SIZE: 'size'>

        >>> config2 = get_recommended_config("edge", "latency")
        >>> config2.quantization
        <TFLiteQuantization.DYNAMIC: 'dynamic'>

        >>> get_recommended_config("unknown")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid target device: unknown

        >>> get_recommended_config("mobile", "unknown")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid optimization priority: unknown
    """
    valid_devices = {"mobile", "edge", "server"}
    if target_device not in valid_devices:
        msg = f"invalid target device: {target_device}"
        raise ValueError(msg)

    valid_priorities = {"size", "latency", "balanced"}
    if optimize_for not in valid_priorities:
        msg = f"invalid optimization priority: {optimize_for}"
        raise ValueError(msg)

    # Select quantization based on device and priority
    if target_device == "mobile":
        if optimize_for == "size":
            quantization = TFLiteQuantization.FULL_INTEGER
            opt_target = OptimizationTarget.SIZE
        elif optimize_for == "latency":
            quantization = TFLiteQuantization.DYNAMIC
            opt_target = OptimizationTarget.LATENCY
        else:  # balanced
            quantization = TFLiteQuantization.DYNAMIC
            opt_target = OptimizationTarget.DEFAULT
    elif target_device == "edge":
        if optimize_for == "size":
            quantization = TFLiteQuantization.FULL_INTEGER
            opt_target = OptimizationTarget.SIZE
        elif optimize_for == "latency":
            quantization = TFLiteQuantization.DYNAMIC
            opt_target = OptimizationTarget.LATENCY
        else:  # balanced
            quantization = TFLiteQuantization.FLOAT16
            opt_target = OptimizationTarget.DEFAULT
    else:  # server
        if optimize_for == "size":
            quantization = TFLiteQuantization.FLOAT16
            opt_target = OptimizationTarget.SIZE
        elif optimize_for == "latency":
            quantization = TFLiteQuantization.NONE
            opt_target = OptimizationTarget.LATENCY
        else:  # balanced
            quantization = TFLiteQuantization.FLOAT16
            opt_target = OptimizationTarget.DEFAULT

    return create_convert_config(
        quantization=quantization,
        optimization_target=opt_target,
    )


def get_recommended_delegate(target_device: str = "mobile") -> DelegateConfig:
    """Get recommended delegate config for target device.

    Args:
        target_device: Target device type ("mobile", "edge", "server").

    Returns:
        Recommended DelegateConfig.

    Raises:
        ValueError: If target_device is not recognized.

    Examples:
        >>> config = get_recommended_delegate("mobile")
        >>> config.delegate_type
        <TFLiteDelegate.GPU: 'gpu'>

        >>> config2 = get_recommended_delegate("edge")
        >>> config2.delegate_type
        <TFLiteDelegate.XNNPACK: 'xnnpack'>

        >>> get_recommended_delegate("unknown")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid target device: unknown
    """
    valid_devices = {"mobile", "edge", "server"}
    if target_device not in valid_devices:
        msg = f"invalid target device: {target_device}"
        raise ValueError(msg)

    if target_device == "mobile":
        return create_delegate_config(
            delegate_type=TFLiteDelegate.GPU,
            num_threads=4,
            enable_fallback=True,
        )
    elif target_device == "edge":
        return create_delegate_config(
            delegate_type=TFLiteDelegate.XNNPACK,
            num_threads=2,
            enable_fallback=True,
        )
    else:  # server
        return create_delegate_config(
            delegate_type=TFLiteDelegate.XNNPACK,
            num_threads=8,
            enable_fallback=False,
        )


def get_convert_config_dict(config: TFLiteConvertConfig) -> dict[str, Any]:
    """Convert TFLiteConvertConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_convert_config(quantization="dynamic")
        >>> d = get_convert_config_dict(config)
        >>> d["quantization"]
        'dynamic'
        >>> d["allow_custom_ops"]
        False

        >>> get_convert_config_dict(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    return {
        "quantization": config.quantization.value,
        "optimization_target": config.optimization_target.value,
        "supported_ops": list(config.supported_ops),
        "allow_custom_ops": config.allow_custom_ops,
    }


def get_delegate_config_dict(config: DelegateConfig) -> dict[str, Any]:
    """Convert DelegateConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_delegate_config(delegate_type="xnnpack")
        >>> d = get_delegate_config_dict(config)
        >>> d["delegate_type"]
        'xnnpack'
        >>> d["num_threads"]
        4

        >>> get_delegate_config_dict(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    return {
        "delegate_type": config.delegate_type.value,
        "num_threads": config.num_threads,
        "enable_fallback": config.enable_fallback,
    }
