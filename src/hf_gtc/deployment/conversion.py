"""Model format conversion utilities.

This module provides utilities for converting models between different
formats including HuggingFace, SafeTensors, GGUF, ONNX, TorchScript, and TFLite.

Examples:
    >>> from hf_gtc.deployment.conversion import ModelFormat, ConversionConfig
    >>> config = ConversionConfig(
    ...     source_format=ModelFormat.HUGGINGFACE,
    ...     target_format=ModelFormat.SAFETENSORS,
    ... )
    >>> config.source_format.value
    'huggingface'
    >>> config.target_format.value
    'safetensors'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class ModelFormat(Enum):
    """Supported model formats for conversion.

    Examples:
        >>> ModelFormat.HUGGINGFACE.value
        'huggingface'
        >>> ModelFormat.SAFETENSORS.value
        'safetensors'
        >>> ModelFormat.GGUF.value
        'gguf'
        >>> ModelFormat.ONNX.value
        'onnx'
    """

    HUGGINGFACE = "huggingface"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TFLITE = "tflite"


VALID_MODEL_FORMATS = frozenset(f.value for f in ModelFormat)


class ConversionPrecision(Enum):
    """Precision levels for model conversion.

    Examples:
        >>> ConversionPrecision.FP32.value
        'fp32'
        >>> ConversionPrecision.FP16.value
        'fp16'
        >>> ConversionPrecision.INT8.value
        'int8'
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


VALID_CONVERSION_PRECISIONS = frozenset(p.value for p in ConversionPrecision)


class ShardingStrategy(Enum):
    """Sharding strategies for large model conversion.

    Examples:
        >>> ShardingStrategy.NONE.value
        'none'
        >>> ShardingStrategy.LAYER.value
        'layer'
        >>> ShardingStrategy.TENSOR.value
        'tensor'
    """

    NONE = "none"
    LAYER = "layer"
    TENSOR = "tensor"
    HYBRID = "hybrid"


VALID_SHARDING_STRATEGIES = frozenset(s.value for s in ShardingStrategy)


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    """Configuration for model format conversion.

    Attributes:
        source_format: Source model format.
        target_format: Target model format.
        precision: Target precision level.
        shard_size_gb: Maximum shard size in gigabytes.

    Examples:
        >>> config = ConversionConfig(
        ...     source_format=ModelFormat.HUGGINGFACE,
        ...     target_format=ModelFormat.SAFETENSORS,
        ... )
        >>> config.source_format.value
        'huggingface'
        >>> config.precision
        <ConversionPrecision.FP16: 'fp16'>

        >>> config2 = ConversionConfig(
        ...     source_format=ModelFormat.HUGGINGFACE,
        ...     target_format=ModelFormat.GGUF,
        ...     precision=ConversionPrecision.INT4,
        ...     shard_size_gb=4.0,
        ... )
        >>> config2.shard_size_gb
        4.0
    """

    source_format: ModelFormat
    target_format: ModelFormat
    precision: ConversionPrecision = ConversionPrecision.FP16
    shard_size_gb: float = 5.0


@dataclass(frozen=True, slots=True)
class GGUFConversionConfig:
    """GGUF-specific conversion configuration.

    Attributes:
        quantization_type: GGUF quantization type (e.g., "q4_k_m").
        use_mmap: Use memory-mapped file for loading.
        vocab_only: Export vocabulary only.

    Examples:
        >>> config = GGUFConversionConfig(quantization_type="q4_k_m")
        >>> config.quantization_type
        'q4_k_m'
        >>> config.use_mmap
        True

        >>> config2 = GGUFConversionConfig(
        ...     quantization_type="q8_0",
        ...     use_mmap=False,
        ...     vocab_only=True,
        ... )
        >>> config2.vocab_only
        True
    """

    quantization_type: str = "q4_k_m"
    use_mmap: bool = True
    vocab_only: bool = False


@dataclass(frozen=True, slots=True)
class SafeTensorsConversionConfig:
    """SafeTensors-specific conversion configuration.

    Attributes:
        metadata: Metadata to include in the file.
        shard_size: Maximum shard size in bytes.
        strict: Enable strict validation mode.

    Examples:
        >>> config = SafeTensorsConversionConfig()
        >>> config.strict
        True

        >>> config2 = SafeTensorsConversionConfig(
        ...     metadata={"format": "pt"},
        ...     shard_size=5_000_000_000,
        ...     strict=False,
        ... )
        >>> config2.metadata
        {'format': 'pt'}
    """

    metadata: dict[str, str] | None = None
    shard_size: int = 5_000_000_000  # 5GB
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ConversionStats:
    """Statistics from a model conversion operation.

    Attributes:
        original_size_mb: Original model size in megabytes.
        converted_size_mb: Converted model size in megabytes.
        conversion_time_seconds: Time taken for conversion.
        precision_loss: Estimated precision loss percentage.

    Examples:
        >>> stats = ConversionStats(
        ...     original_size_mb=14000.0,
        ...     converted_size_mb=4000.0,
        ...     conversion_time_seconds=120.5,
        ...     precision_loss=0.05,
        ... )
        >>> stats.original_size_mb
        14000.0
        >>> stats.precision_loss
        0.05
    """

    original_size_mb: float
    converted_size_mb: float
    conversion_time_seconds: float
    precision_loss: float


def validate_conversion_config(config: ConversionConfig) -> None:
    """Validate conversion configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If source and target formats are the same.
        ValueError: If shard_size_gb is not positive.

    Examples:
        >>> config = ConversionConfig(
        ...     source_format=ModelFormat.HUGGINGFACE,
        ...     target_format=ModelFormat.SAFETENSORS,
        ... )
        >>> validate_conversion_config(config)  # No error

        >>> validate_conversion_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ConversionConfig(
        ...     source_format=ModelFormat.GGUF,
        ...     target_format=ModelFormat.GGUF,
        ... )
        >>> validate_conversion_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: source and target formats cannot be the same
    """
    validate_not_none(config, "config")

    if config.source_format == config.target_format:
        msg = "source and target formats cannot be the same"
        raise ValueError(msg)

    if config.shard_size_gb <= 0:
        msg = f"shard_size_gb must be positive, got {config.shard_size_gb}"
        raise ValueError(msg)


def validate_gguf_conversion_config(config: GGUFConversionConfig) -> None:
    """Validate GGUF conversion configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If quantization_type is empty.

    Examples:
        >>> config = GGUFConversionConfig(quantization_type="q4_k_m")
        >>> validate_gguf_conversion_config(config)  # No error

        >>> validate_gguf_conversion_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = GGUFConversionConfig(quantization_type="")
        >>> validate_gguf_conversion_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quantization_type cannot be empty
    """
    validate_not_none(config, "config")

    if not config.quantization_type:
        msg = "quantization_type cannot be empty"
        raise ValueError(msg)


def validate_safetensors_conversion_config(
    config: SafeTensorsConversionConfig,
) -> None:
    """Validate SafeTensors conversion configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If shard_size is not positive.

    Examples:
        >>> config = SafeTensorsConversionConfig()
        >>> validate_safetensors_conversion_config(config)  # No error

        >>> validate_safetensors_conversion_config(None)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SafeTensorsConversionConfig(shard_size=0)
        >>> validate_safetensors_conversion_config(bad)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: shard_size must be positive
    """
    validate_not_none(config, "config")

    if config.shard_size <= 0:
        msg = f"shard_size must be positive, got {config.shard_size}"
        raise ValueError(msg)

    if config.metadata is not None:
        for key, value in config.metadata.items():
            if not isinstance(key, str):
                msg = f"metadata keys must be strings, got {type(key)}"
                raise ValueError(msg)
            if not isinstance(value, str):
                msg = f"metadata values must be strings, got {type(value)}"
                raise ValueError(msg)


def create_conversion_config(
    source_format: ModelFormat | str,
    target_format: ModelFormat | str,
    precision: ConversionPrecision | str = ConversionPrecision.FP16,
    shard_size_gb: float = 5.0,
) -> ConversionConfig:
    """Create a conversion configuration.

    Args:
        source_format: Source model format.
        target_format: Target model format.
        precision: Target precision. Defaults to FP16.
        shard_size_gb: Maximum shard size in GB. Defaults to 5.0.

    Returns:
        Validated ConversionConfig instance.

    Raises:
        ValueError: If source and target formats are the same.
        ValueError: If shard_size_gb is not positive.

    Examples:
        >>> config = create_conversion_config("huggingface", "safetensors")
        >>> config.source_format
        <ModelFormat.HUGGINGFACE: 'huggingface'>

        >>> config2 = create_conversion_config(
        ...     ModelFormat.HUGGINGFACE,
        ...     ModelFormat.GGUF,
        ...     precision="int4",
        ... )
        >>> config2.precision
        <ConversionPrecision.INT4: 'int4'>
    """
    if isinstance(source_format, str):
        source_format = get_model_format(source_format)
    if isinstance(target_format, str):
        target_format = get_model_format(target_format)
    if isinstance(precision, str):
        precision = get_conversion_precision(precision)

    config = ConversionConfig(
        source_format=source_format,
        target_format=target_format,
        precision=precision,
        shard_size_gb=shard_size_gb,
    )
    validate_conversion_config(config)
    return config


def create_gguf_conversion_config(
    quantization_type: str = "q4_k_m",
    use_mmap: bool = True,
    vocab_only: bool = False,
) -> GGUFConversionConfig:
    """Create a GGUF conversion configuration.

    Args:
        quantization_type: GGUF quantization type. Defaults to "q4_k_m".
        use_mmap: Use memory-mapped files. Defaults to True.
        vocab_only: Export vocabulary only. Defaults to False.

    Returns:
        Validated GGUFConversionConfig instance.

    Raises:
        ValueError: If quantization_type is empty.

    Examples:
        >>> config = create_gguf_conversion_config(quantization_type="q8_0")
        >>> config.quantization_type
        'q8_0'

        >>> config2 = create_gguf_conversion_config(use_mmap=False)
        >>> config2.use_mmap
        False
    """
    config = GGUFConversionConfig(
        quantization_type=quantization_type,
        use_mmap=use_mmap,
        vocab_only=vocab_only,
    )
    validate_gguf_conversion_config(config)
    return config


def create_safetensors_conversion_config(
    metadata: dict[str, str] | None = None,
    shard_size: int = 5_000_000_000,
    strict: bool = True,
) -> SafeTensorsConversionConfig:
    """Create a SafeTensors conversion configuration.

    Args:
        metadata: Metadata dictionary. Defaults to None.
        shard_size: Shard size in bytes. Defaults to 5GB.
        strict: Strict validation mode. Defaults to True.

    Returns:
        Validated SafeTensorsConversionConfig instance.

    Raises:
        ValueError: If shard_size is not positive.

    Examples:
        >>> config = create_safetensors_conversion_config()
        >>> config.strict
        True

        >>> config2 = create_safetensors_conversion_config(
        ...     metadata={"author": "user"},
        ... )
        >>> config2.metadata
        {'author': 'user'}
    """
    config = SafeTensorsConversionConfig(
        metadata=metadata,
        shard_size=shard_size,
        strict=strict,
    )
    validate_safetensors_conversion_config(config)
    return config


def list_model_formats() -> list[str]:
    """List available model formats.

    Returns:
        Sorted list of format names.

    Examples:
        >>> formats = list_model_formats()
        >>> "huggingface" in formats
        True
        >>> "safetensors" in formats
        True
        >>> "gguf" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_MODEL_FORMATS)


def list_conversion_precisions() -> list[str]:
    """List available conversion precisions.

    Returns:
        Sorted list of precision names.

    Examples:
        >>> precisions = list_conversion_precisions()
        >>> "fp32" in precisions
        True
        >>> "fp16" in precisions
        True
        >>> "int8" in precisions
        True
        >>> precisions == sorted(precisions)
        True
    """
    return sorted(VALID_CONVERSION_PRECISIONS)


def list_sharding_strategies() -> list[str]:
    """List available sharding strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_sharding_strategies()
        >>> "none" in strategies
        True
        >>> "layer" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_SHARDING_STRATEGIES)


def get_model_format(name: str) -> ModelFormat:
    """Get model format enum from string.

    Args:
        name: Format name.

    Returns:
        ModelFormat enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_model_format("huggingface")
        <ModelFormat.HUGGINGFACE: 'huggingface'>
        >>> get_model_format("safetensors")
        <ModelFormat.SAFETENSORS: 'safetensors'>

        >>> get_model_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid model format: invalid
    """
    for fmt in ModelFormat:
        if fmt.value == name:
            return fmt
    msg = f"invalid model format: {name}"
    raise ValueError(msg)


def get_conversion_precision(name: str) -> ConversionPrecision:
    """Get conversion precision enum from string.

    Args:
        name: Precision name.

    Returns:
        ConversionPrecision enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_conversion_precision("fp32")
        <ConversionPrecision.FP32: 'fp32'>
        >>> get_conversion_precision("int8")
        <ConversionPrecision.INT8: 'int8'>

        >>> get_conversion_precision("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid conversion precision: invalid
    """
    for prec in ConversionPrecision:
        if prec.value == name:
            return prec
    msg = f"invalid conversion precision: {name}"
    raise ValueError(msg)


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Get sharding strategy enum from string.

    Args:
        name: Strategy name.

    Returns:
        ShardingStrategy enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_sharding_strategy("none")
        <ShardingStrategy.NONE: 'none'>
        >>> get_sharding_strategy("layer")
        <ShardingStrategy.LAYER: 'layer'>

        >>> get_sharding_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid sharding strategy: invalid
    """
    for strategy in ShardingStrategy:
        if strategy.value == name:
            return strategy
    msg = f"invalid sharding strategy: {name}"
    raise ValueError(msg)


def estimate_converted_size(
    model_params: int,
    source_precision: ConversionPrecision,
    target_precision: ConversionPrecision,
) -> float:
    """Estimate converted model size in megabytes.

    Args:
        model_params: Number of model parameters.
        source_precision: Source precision level.
        target_precision: Target precision level.

    Returns:
        Estimated size in megabytes.

    Raises:
        ValueError: If model_params is not positive.

    Examples:
        >>> size = estimate_converted_size(
        ...     7_000_000_000,
        ...     ConversionPrecision.FP32,
        ...     ConversionPrecision.FP16,
        ... )
        >>> size > 0
        True
        >>> size < 28000  # Smaller than FP32
        True

        >>> size_int4 = estimate_converted_size(
        ...     7_000_000_000,
        ...     ConversionPrecision.FP32,
        ...     ConversionPrecision.INT4,
        ... )
        >>> size_int4 < size
        True

        >>> estimate_converted_size(
        ...     0, ConversionPrecision.FP32, ConversionPrecision.FP16
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    # Bytes per parameter for each precision
    bytes_per_param = {
        ConversionPrecision.FP32: 4,
        ConversionPrecision.FP16: 2,
        ConversionPrecision.BF16: 2,
        ConversionPrecision.INT8: 1,
        ConversionPrecision.INT4: 0.5,
    }

    target_bytes = bytes_per_param[target_precision]
    size_bytes = model_params * target_bytes

    # Add ~5% overhead for format metadata
    size_bytes *= 1.05

    return size_bytes / (1024 * 1024)


def validate_format_compatibility(
    source_format: ModelFormat,
    target_format: ModelFormat,
) -> bool:
    """Validate that conversion between formats is supported.

    Args:
        source_format: Source model format.
        target_format: Target model format.

    Returns:
        True if conversion is supported, False otherwise.

    Examples:
        >>> validate_format_compatibility(
        ...     ModelFormat.HUGGINGFACE, ModelFormat.SAFETENSORS
        ... )
        True
        >>> validate_format_compatibility(
        ...     ModelFormat.HUGGINGFACE, ModelFormat.GGUF
        ... )
        True
        >>> validate_format_compatibility(
        ...     ModelFormat.TFLITE, ModelFormat.GGUF
        ... )
        False
        >>> validate_format_compatibility(
        ...     ModelFormat.HUGGINGFACE, ModelFormat.HUGGINGFACE
        ... )
        False
    """
    if source_format == target_format:
        return False

    # Define supported conversion paths
    supported_conversions: dict[ModelFormat, set[ModelFormat]] = {
        ModelFormat.HUGGINGFACE: {
            ModelFormat.SAFETENSORS,
            ModelFormat.GGUF,
            ModelFormat.ONNX,
            ModelFormat.TORCHSCRIPT,
            ModelFormat.TFLITE,
        },
        ModelFormat.SAFETENSORS: {
            ModelFormat.HUGGINGFACE,
            ModelFormat.GGUF,
            ModelFormat.ONNX,
            ModelFormat.TORCHSCRIPT,
        },
        ModelFormat.GGUF: {
            ModelFormat.HUGGINGFACE,
        },
        ModelFormat.ONNX: {
            ModelFormat.TFLITE,
        },
        ModelFormat.TORCHSCRIPT: {
            ModelFormat.ONNX,
        },
        ModelFormat.TFLITE: set(),  # TFLite is typically an end format
    }

    return target_format in supported_conversions.get(source_format, set())


def calculate_precision_loss(
    source_precision: ConversionPrecision,
    target_precision: ConversionPrecision,
) -> float:
    """Calculate estimated precision loss from conversion.

    Args:
        source_precision: Source precision level.
        target_precision: Target precision level.

    Returns:
        Estimated precision loss as a percentage (0.0 to 100.0).

    Examples:
        >>> calculate_precision_loss(ConversionPrecision.FP32, ConversionPrecision.FP32)
        0.0
        >>> calculate_precision_loss(ConversionPrecision.FP32, ConversionPrecision.FP16)
        0.1
        >>> calculate_precision_loss(ConversionPrecision.FP32, ConversionPrecision.INT8)
        1.0
        >>> calculate_precision_loss(ConversionPrecision.FP32, ConversionPrecision.INT4)
        5.0
        >>> calculate_precision_loss(ConversionPrecision.FP16, ConversionPrecision.FP32)
        0.0
    """
    # Precision ranking (higher = more precise)
    precision_rank = {
        ConversionPrecision.FP32: 5,
        ConversionPrecision.BF16: 4,
        ConversionPrecision.FP16: 4,
        ConversionPrecision.INT8: 2,
        ConversionPrecision.INT4: 1,
    }

    # Estimated loss percentages for each conversion
    loss_matrix = {
        (ConversionPrecision.FP32, ConversionPrecision.FP16): 0.1,
        (ConversionPrecision.FP32, ConversionPrecision.BF16): 0.1,
        (ConversionPrecision.FP32, ConversionPrecision.INT8): 1.0,
        (ConversionPrecision.FP32, ConversionPrecision.INT4): 5.0,
        (ConversionPrecision.FP16, ConversionPrecision.INT8): 0.9,
        (ConversionPrecision.FP16, ConversionPrecision.INT4): 4.9,
        (ConversionPrecision.BF16, ConversionPrecision.INT8): 0.9,
        (ConversionPrecision.BF16, ConversionPrecision.INT4): 4.9,
        (ConversionPrecision.INT8, ConversionPrecision.INT4): 3.0,
    }

    # No loss if going to same or higher precision
    if precision_rank[target_precision] >= precision_rank[source_precision]:
        return 0.0

    return loss_matrix.get((source_precision, target_precision), 0.0)


def estimate_conversion_time(
    model_params: int,
    source_format: ModelFormat,
    target_format: ModelFormat,
    target_precision: ConversionPrecision = ConversionPrecision.FP16,
) -> float:
    """Estimate conversion time in seconds.

    Args:
        model_params: Number of model parameters.
        source_format: Source model format.
        target_format: Target model format.
        target_precision: Target precision level. Defaults to FP16.

    Returns:
        Estimated conversion time in seconds.

    Raises:
        ValueError: If model_params is not positive.

    Examples:
        >>> time = estimate_conversion_time(
        ...     7_000_000_000,
        ...     ModelFormat.HUGGINGFACE,
        ...     ModelFormat.SAFETENSORS,
        ... )
        >>> time > 0
        True
        >>> time < 300  # Should be under 5 minutes for simple conversion
        True

        >>> time_gguf = estimate_conversion_time(
        ...     7_000_000_000,
        ...     ModelFormat.HUGGINGFACE,
        ...     ModelFormat.GGUF,
        ...     ConversionPrecision.INT4,
        ... )
        >>> time_gguf > time  # GGUF quantization takes longer
        True

        >>> estimate_conversion_time(0, ModelFormat.HUGGINGFACE, ModelFormat.GGUF)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    # Base time: ~1 second per billion parameters for simple conversion
    base_time = model_params / 1_000_000_000

    # Format-specific multipliers
    format_multipliers = {
        (ModelFormat.HUGGINGFACE, ModelFormat.SAFETENSORS): 1.0,
        (ModelFormat.SAFETENSORS, ModelFormat.HUGGINGFACE): 1.0,
        (ModelFormat.HUGGINGFACE, ModelFormat.GGUF): 5.0,
        (ModelFormat.SAFETENSORS, ModelFormat.GGUF): 5.0,
        (ModelFormat.HUGGINGFACE, ModelFormat.ONNX): 3.0,
        (ModelFormat.SAFETENSORS, ModelFormat.ONNX): 3.0,
        (ModelFormat.HUGGINGFACE, ModelFormat.TORCHSCRIPT): 2.0,
        (ModelFormat.SAFETENSORS, ModelFormat.TORCHSCRIPT): 2.0,
        (ModelFormat.HUGGINGFACE, ModelFormat.TFLITE): 4.0,
        (ModelFormat.ONNX, ModelFormat.TFLITE): 2.0,
        (ModelFormat.TORCHSCRIPT, ModelFormat.ONNX): 2.0,
        (ModelFormat.GGUF, ModelFormat.HUGGINGFACE): 3.0,
    }

    multiplier = format_multipliers.get((source_format, target_format), 2.0)

    # Additional time for quantization
    quantization_multipliers = {
        ConversionPrecision.FP32: 1.0,
        ConversionPrecision.FP16: 1.0,
        ConversionPrecision.BF16: 1.0,
        ConversionPrecision.INT8: 1.5,
        ConversionPrecision.INT4: 2.0,
    }

    quant_multiplier = quantization_multipliers[target_precision]

    return base_time * multiplier * quant_multiplier


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
        ...     original_size_mb=14000.0,
        ...     converted_size_mb=4000.0,
        ...     conversion_time_seconds=120.5,
        ...     precision_loss=0.05,
        ... )
        >>> formatted = format_conversion_stats(stats)
        >>> "Original size:" in formatted
        True
        >>> "Converted size:" in formatted
        True
        >>> "Compression ratio:" in formatted
        True
    """
    validate_not_none(stats, "stats")

    compression_ratio = stats.original_size_mb / stats.converted_size_mb

    lines = [
        f"Original size: {stats.original_size_mb:.1f} MB",
        f"Converted size: {stats.converted_size_mb:.1f} MB",
        f"Compression ratio: {compression_ratio:.2f}x",
        f"Conversion time: {stats.conversion_time_seconds:.1f}s",
        f"Precision loss: {stats.precision_loss:.2f}%",
    ]
    return "\n".join(lines)


def get_recommended_conversion_config(
    model_size: str,
    target_format: ModelFormat | str,
    optimize_for: str = "balanced",
) -> ConversionConfig:
    """Get recommended conversion configuration for a model.

    Args:
        model_size: Model size string (e.g., "7b", "13b", "70b").
        target_format: Target model format.
        optimize_for: Optimization goal ("speed", "quality", "balanced", "size").

    Returns:
        Recommended ConversionConfig.

    Raises:
        ValueError: If model_size is not recognized.
        ValueError: If optimize_for is not valid.

    Examples:
        >>> config = get_recommended_conversion_config("7b", "safetensors")
        >>> config.precision
        <ConversionPrecision.FP16: 'fp16'>

        >>> config_gguf = get_recommended_conversion_config(
        ...     "7b", ModelFormat.GGUF, optimize_for="size"
        ... )
        >>> config_gguf.precision
        <ConversionPrecision.INT4: 'int4'>

        >>> config_quality = get_recommended_conversion_config(
        ...     "70b", "safetensors", optimize_for="quality"
        ... )
        >>> config_quality.precision
        <ConversionPrecision.FP32: 'fp32'>

        >>> get_recommended_conversion_config("invalid", "safetensors")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: unrecognized model size: invalid
    """
    if isinstance(target_format, str):
        target_format = get_model_format(target_format)

    model_size = model_size.lower().strip()
    valid_sizes = {"7b", "13b", "70b", "small", "medium", "large"}
    if model_size not in valid_sizes:
        msg = f"unrecognized model size: {model_size}"
        raise ValueError(msg)

    valid_optimize_for = {"speed", "quality", "balanced", "size"}
    if optimize_for not in valid_optimize_for:
        msg = f"optimize_for must be one of {valid_optimize_for}, got {optimize_for}"
        raise ValueError(msg)

    # Determine precision based on optimization goal
    if optimize_for == "quality":
        precision = ConversionPrecision.FP32
    elif optimize_for == "speed":
        precision = ConversionPrecision.FP16
    elif optimize_for == "size":
        if target_format == ModelFormat.GGUF:
            precision = ConversionPrecision.INT4
        else:
            precision = ConversionPrecision.INT8
    else:  # balanced
        precision = ConversionPrecision.FP16

    # Determine shard size based on model size
    shard_sizes = {
        "7b": 5.0,
        "13b": 5.0,
        "70b": 10.0,
        "small": 2.0,
        "medium": 5.0,
        "large": 10.0,
    }
    shard_size = shard_sizes.get(model_size, 5.0)

    return create_conversion_config(
        source_format=ModelFormat.HUGGINGFACE,
        target_format=target_format,
        precision=precision,
        shard_size_gb=shard_size,
    )


def get_conversion_config_dict(config: ConversionConfig) -> dict[str, Any]:
    """Convert ConversionConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_conversion_config("huggingface", "safetensors")
        >>> d = get_conversion_config_dict(config)
        >>> d["source_format"]
        'huggingface'
        >>> d["target_format"]
        'safetensors'
    """
    validate_not_none(config, "config")

    return {
        "source_format": config.source_format.value,
        "target_format": config.target_format.value,
        "precision": config.precision.value,
        "shard_size_gb": config.shard_size_gb,
    }


def get_gguf_conversion_config_dict(config: GGUFConversionConfig) -> dict[str, Any]:
    """Convert GGUFConversionConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_gguf_conversion_config(quantization_type="q4_k_m")
        >>> d = get_gguf_conversion_config_dict(config)
        >>> d["quantization_type"]
        'q4_k_m'
        >>> d["use_mmap"]
        True
    """
    validate_not_none(config, "config")

    return {
        "quantization_type": config.quantization_type,
        "use_mmap": config.use_mmap,
        "vocab_only": config.vocab_only,
    }


def get_safetensors_conversion_config_dict(
    config: SafeTensorsConversionConfig,
) -> dict[str, Any]:
    """Convert SafeTensorsConversionConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_safetensors_conversion_config()
        >>> d = get_safetensors_conversion_config_dict(config)
        >>> d["strict"]
        True
        >>> d["shard_size"]
        5000000000
    """
    validate_not_none(config, "config")

    return {
        "metadata": config.metadata,
        "shard_size": config.shard_size,
        "strict": config.strict,
    }


def create_conversion_stats(
    original_size_mb: float,
    converted_size_mb: float,
    conversion_time_seconds: float,
    precision_loss: float,
) -> ConversionStats:
    """Create conversion statistics.

    Args:
        original_size_mb: Original model size in MB.
        converted_size_mb: Converted model size in MB.
        conversion_time_seconds: Conversion time in seconds.
        precision_loss: Precision loss percentage.

    Returns:
        ConversionStats instance.

    Raises:
        ValueError: If sizes are not positive.
        ValueError: If conversion_time_seconds is negative.
        ValueError: If precision_loss is negative.

    Examples:
        >>> stats = create_conversion_stats(14000.0, 4000.0, 120.5, 0.05)
        >>> stats.original_size_mb
        14000.0

        >>> create_conversion_stats(0, 4000.0, 120.5, 0.05)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_size_mb must be positive
    """
    if original_size_mb <= 0:
        msg = f"original_size_mb must be positive, got {original_size_mb}"
        raise ValueError(msg)

    if converted_size_mb <= 0:
        msg = f"converted_size_mb must be positive, got {converted_size_mb}"
        raise ValueError(msg)

    if conversion_time_seconds < 0:
        msg = (
            f"conversion_time_seconds cannot be negative, got {conversion_time_seconds}"
        )
        raise ValueError(msg)

    if precision_loss < 0:
        msg = f"precision_loss cannot be negative, got {precision_loss}"
        raise ValueError(msg)

    return ConversionStats(
        original_size_mb=original_size_mb,
        converted_size_mb=converted_size_mb,
        conversion_time_seconds=conversion_time_seconds,
        precision_loss=precision_loss,
    )
