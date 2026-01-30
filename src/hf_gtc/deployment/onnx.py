"""ONNX export and runtime utilities for model deployment.

This module provides utilities for exporting HuggingFace models to ONNX format,
optimizing ONNX models, and configuring ONNX Runtime for inference.

Examples:
    >>> from hf_gtc.deployment.onnx import ONNXOpset, ExecutionProvider
    >>> ONNXOpset.OPSET_14.value
    14
    >>> ExecutionProvider.CPU.value
    'cpu'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ONNXOpset(Enum):
    """ONNX opset versions.

    ONNX opsets define the set of operators available for model export.
    Higher opset versions generally provide more operators and better
    optimization opportunities.

    Attributes:
        OPSET_11: Opset version 11 (ONNX 1.6).
        OPSET_12: Opset version 12 (ONNX 1.7).
        OPSET_13: Opset version 13 (ONNX 1.8).
        OPSET_14: Opset version 14 (ONNX 1.9).
        OPSET_15: Opset version 15 (ONNX 1.10).
        OPSET_16: Opset version 16 (ONNX 1.11).
        OPSET_17: Opset version 17 (ONNX 1.12).

    Examples:
        >>> ONNXOpset.OPSET_14.value
        14
        >>> ONNXOpset.OPSET_17.value
        17
    """

    OPSET_11 = 11
    OPSET_12 = 12
    OPSET_13 = 13
    OPSET_14 = 14
    OPSET_15 = 15
    OPSET_16 = 16
    OPSET_17 = 17


VALID_OPSET_VERSIONS = frozenset(o.value for o in ONNXOpset)
# Alias for consistency with module naming convention
VALID_ONNX_OPSET_VERSIONS = VALID_OPSET_VERSIONS


class OptimizationLevel(Enum):
    """ONNX Runtime graph optimization levels.

    Optimization levels control the extent of graph transformations
    applied during ONNX Runtime session initialization.

    Attributes:
        DISABLE: No graph optimizations.
        BASIC: Basic optimizations (constant folding, redundant node elimination).
        EXTENDED: Extended optimizations (includes operator fusions).
        ALL: All optimizations including layout transformations.

    Examples:
        >>> OptimizationLevel.BASIC.value
        'basic'
        >>> OptimizationLevel.ALL.value
        'all'
    """

    DISABLE = "disable"
    BASIC = "basic"
    EXTENDED = "extended"
    ALL = "all"


VALID_OPTIMIZATION_LEVELS = frozenset(o.value for o in OptimizationLevel)


class ExecutionProvider(Enum):
    """ONNX Runtime execution providers.

    Execution providers define the hardware backend used for inference.
    Different providers offer optimized execution on specific hardware.

    Attributes:
        CPU: CPU execution provider (default, always available).
        CUDA: NVIDIA CUDA execution provider for GPU inference.
        TENSORRT: NVIDIA TensorRT execution provider for optimized GPU inference.
        COREML: Apple CoreML execution provider for macOS/iOS.
        DIRECTML: DirectML execution provider for Windows GPU inference.

    Examples:
        >>> ExecutionProvider.CPU.value
        'cpu'
        >>> ExecutionProvider.CUDA.value
        'cuda'
    """

    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    DIRECTML = "directml"


VALID_EXECUTION_PROVIDERS = frozenset(e.value for e in ExecutionProvider)


@dataclass(frozen=True, slots=True)
class ONNXExportConfig:
    """Configuration for ONNX model export.

    Attributes:
        opset_version: ONNX opset version. Defaults to 14.
        do_constant_folding: Whether to fold constants during export.
        dynamic_axes: Dictionary mapping input/output names to dynamic axes.
        input_names: List of input tensor names.
        output_names: List of output tensor names.

    Examples:
        >>> config = ONNXExportConfig(
        ...     opset_version=14,
        ...     input_names=("input_ids", "attention_mask"),
        ...     output_names=("logits",),
        ... )
        >>> config.opset_version
        14
        >>> config.input_names
        ('input_ids', 'attention_mask')

        >>> config2 = ONNXExportConfig(dynamic_axes={"input_ids": {0: "batch"}})
        >>> config2.dynamic_axes
        {'input_ids': {0: 'batch'}}
    """

    opset_version: int = 14
    do_constant_folding: bool = True
    dynamic_axes: dict[str, dict[int, str]] | None = None
    input_names: tuple[str, ...] = ("input_ids",)
    output_names: tuple[str, ...] = ("logits",)


@dataclass(frozen=True, slots=True)
class ONNXOptimizeConfig:
    """Configuration for ONNX model optimization.

    Attributes:
        level: Optimization level. Defaults to BASIC.
        enable_gelu_fusion: Whether to fuse GELU patterns. Defaults to True.
        enable_layer_norm_fusion: Whether to fuse LayerNorm patterns.
        enable_attention_fusion: Whether to fuse attention patterns.

    Examples:
        >>> config = ONNXOptimizeConfig(level=OptimizationLevel.EXTENDED)
        >>> config.level
        <OptimizationLevel.EXTENDED: 'extended'>
        >>> config.enable_gelu_fusion
        True

        >>> config2 = ONNXOptimizeConfig(enable_attention_fusion=True)
        >>> config2.enable_attention_fusion
        True
    """

    level: OptimizationLevel = OptimizationLevel.BASIC
    enable_gelu_fusion: bool = True
    enable_layer_norm_fusion: bool = True
    enable_attention_fusion: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Configuration for ONNX Runtime session.

    Attributes:
        provider: Execution provider. Defaults to CPU.
        num_threads: Number of intra-op threads. Defaults to 0 (auto).
        graph_optimization_level: Graph optimization level. Defaults to BASIC.
        enable_profiling: Whether to enable profiling. Defaults to False.

    Examples:
        >>> config = RuntimeConfig(provider=ExecutionProvider.CUDA)
        >>> config.provider
        <ExecutionProvider.CUDA: 'cuda'>
        >>> config.num_threads
        0

        >>> config2 = RuntimeConfig(num_threads=4, enable_profiling=True)
        >>> config2.num_threads
        4
        >>> config2.enable_profiling
        True
    """

    provider: ExecutionProvider = ExecutionProvider.CPU
    num_threads: int = 0
    graph_optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    enable_profiling: bool = False


@dataclass(frozen=True, slots=True)
class ONNXModelInfo:
    """Information about an ONNX model.

    Attributes:
        model_path: Path to the ONNX model file.
        opset_version: ONNX opset version of the model.
        input_shapes: Dictionary mapping input names to shapes.
        output_shapes: Dictionary mapping output names to shapes.
        num_parameters: Estimated number of model parameters.

    Examples:
        >>> info = ONNXModelInfo(
        ...     model_path="/path/to/model.onnx",
        ...     opset_version=14,
        ...     input_shapes={"input_ids": [1, 512]},
        ...     output_shapes={"logits": [1, 512, 768]},
        ...     num_parameters=110_000_000,
        ... )
        >>> info.opset_version
        14
        >>> info.num_parameters
        110000000
    """

    model_path: str
    opset_version: int
    input_shapes: dict[str, list[int | str]]
    output_shapes: dict[str, list[int | str]]
    num_parameters: int


@dataclass(frozen=True, slots=True)
class ExportStats:
    """Statistics from ONNX model export.

    Attributes:
        export_time_seconds: Time taken to export the model.
        model_size_bytes: Size of the exported ONNX file in bytes.
        num_nodes: Number of nodes in the ONNX graph.
        num_initializers: Number of initializers (weights) in the model.

    Examples:
        >>> stats = ExportStats(
        ...     export_time_seconds=5.5,
        ...     model_size_bytes=440_000_000,
        ...     num_nodes=1500,
        ...     num_initializers=200,
        ... )
        >>> stats.export_time_seconds
        5.5
        >>> stats.model_size_bytes
        440000000
    """

    export_time_seconds: float
    model_size_bytes: int
    num_nodes: int
    num_initializers: int


def validate_export_config(config: ONNXExportConfig) -> None:
    """Validate ONNX export configuration.

    Args:
        config: ONNXExportConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If opset_version is not valid.
        ValueError: If input_names is empty.
        ValueError: If output_names is empty.

    Examples:
        >>> config = ONNXExportConfig(opset_version=14)
        >>> validate_export_config(config)  # No error

        >>> validate_export_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ONNXExportConfig(opset_version=10)
        >>> validate_export_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: opset_version must be one of...

        >>> bad2 = ONNXExportConfig(input_names=())
        >>> validate_export_config(bad2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: input_names cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.opset_version not in VALID_OPSET_VERSIONS:
        msg = (
            f"opset_version must be one of {sorted(VALID_OPSET_VERSIONS)}, "
            f"got {config.opset_version}"
        )
        raise ValueError(msg)

    if not config.input_names:
        msg = "input_names cannot be empty"
        raise ValueError(msg)

    if not config.output_names:
        msg = "output_names cannot be empty"
        raise ValueError(msg)


def validate_runtime_config(config: RuntimeConfig) -> None:
    """Validate ONNX Runtime configuration.

    Args:
        config: RuntimeConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_threads is negative.

    Examples:
        >>> config = RuntimeConfig(num_threads=4)
        >>> validate_runtime_config(config)  # No error

        >>> validate_runtime_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = RuntimeConfig(num_threads=-1)
        >>> validate_runtime_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_threads cannot be negative
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_threads < 0:
        msg = f"num_threads cannot be negative, got {config.num_threads}"
        raise ValueError(msg)


def create_export_config(
    opset_version: int | ONNXOpset = 14,
    do_constant_folding: bool = True,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: tuple[str, ...] | list[str] = ("input_ids",),
    output_names: tuple[str, ...] | list[str] = ("logits",),
) -> ONNXExportConfig:
    """Create an ONNX export configuration.

    Args:
        opset_version: ONNX opset version. Defaults to 14.
        do_constant_folding: Whether to fold constants. Defaults to True.
        dynamic_axes: Dictionary mapping input/output names to dynamic axes.
        input_names: List of input tensor names. Defaults to ("input_ids",).
        output_names: List of output tensor names. Defaults to ("logits",).

    Returns:
        Validated ONNXExportConfig instance.

    Raises:
        ValueError: If opset_version is invalid.
        ValueError: If input_names or output_names is empty.

    Examples:
        >>> config = create_export_config(opset_version=14)
        >>> config.opset_version
        14
        >>> config.do_constant_folding
        True

        >>> config2 = create_export_config(
        ...     opset_version=ONNXOpset.OPSET_17,
        ...     input_names=["input_ids", "attention_mask"],
        ... )
        >>> config2.opset_version
        17

        >>> create_export_config(opset_version=10)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: opset_version must be one of...
    """
    if isinstance(opset_version, ONNXOpset):
        opset_version = opset_version.value

    # Convert list to tuple if needed
    if isinstance(input_names, list):
        input_names = tuple(input_names)
    if isinstance(output_names, list):
        output_names = tuple(output_names)

    config = ONNXExportConfig(
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
    )
    validate_export_config(config)
    return config


def create_optimize_config(
    level: OptimizationLevel | str = OptimizationLevel.BASIC,
    enable_gelu_fusion: bool = True,
    enable_layer_norm_fusion: bool = True,
    enable_attention_fusion: bool = False,
) -> ONNXOptimizeConfig:
    """Create an ONNX optimization configuration.

    Args:
        level: Optimization level. Defaults to BASIC.
        enable_gelu_fusion: Whether to fuse GELU patterns. Defaults to True.
        enable_layer_norm_fusion: Whether to fuse LayerNorm. Defaults to True.
        enable_attention_fusion: Whether to fuse attention. Defaults to False.

    Returns:
        ONNXOptimizeConfig instance.

    Examples:
        >>> config = create_optimize_config(level="extended")
        >>> config.level
        <OptimizationLevel.EXTENDED: 'extended'>

        >>> config2 = create_optimize_config(
        ...     level=OptimizationLevel.ALL,
        ...     enable_attention_fusion=True,
        ... )
        >>> config2.enable_attention_fusion
        True

        >>> create_optimize_config(level="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid optimization level: invalid
    """
    if isinstance(level, str):
        level = get_optimization_level(level)

    return ONNXOptimizeConfig(
        level=level,
        enable_gelu_fusion=enable_gelu_fusion,
        enable_layer_norm_fusion=enable_layer_norm_fusion,
        enable_attention_fusion=enable_attention_fusion,
    )


def create_runtime_config(
    provider: ExecutionProvider | str = ExecutionProvider.CPU,
    num_threads: int = 0,
    graph_optimization_level: OptimizationLevel | str = OptimizationLevel.BASIC,
    enable_profiling: bool = False,
) -> RuntimeConfig:
    """Create an ONNX Runtime configuration.

    Args:
        provider: Execution provider. Defaults to CPU.
        num_threads: Number of intra-op threads. Defaults to 0 (auto).
        graph_optimization_level: Graph optimization level. Defaults to BASIC.
        enable_profiling: Whether to enable profiling. Defaults to False.

    Returns:
        Validated RuntimeConfig instance.

    Raises:
        ValueError: If provider is invalid.
        ValueError: If num_threads is negative.

    Examples:
        >>> config = create_runtime_config(provider="cuda", num_threads=4)
        >>> config.provider
        <ExecutionProvider.CUDA: 'cuda'>
        >>> config.num_threads
        4

        >>> config2 = create_runtime_config(
        ...     provider=ExecutionProvider.TENSORRT,
        ...     enable_profiling=True,
        ... )
        >>> config2.provider
        <ExecutionProvider.TENSORRT: 'tensorrt'>

        >>> create_runtime_config(num_threads=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_threads cannot be negative
    """
    if isinstance(provider, str):
        provider = get_execution_provider(provider)

    if isinstance(graph_optimization_level, str):
        graph_optimization_level = get_optimization_level(graph_optimization_level)

    config = RuntimeConfig(
        provider=provider,
        num_threads=num_threads,
        graph_optimization_level=graph_optimization_level,
        enable_profiling=enable_profiling,
    )
    validate_runtime_config(config)
    return config


def list_opset_versions() -> list[int]:
    """List available ONNX opset versions.

    Returns:
        Sorted list of opset version numbers.

    Examples:
        >>> versions = list_opset_versions()
        >>> 14 in versions
        True
        >>> 17 in versions
        True
        >>> versions == sorted(versions)
        True
    """
    return sorted(VALID_OPSET_VERSIONS)


def list_optimization_levels() -> list[str]:
    """List available optimization levels.

    Returns:
        Sorted list of optimization level names.

    Examples:
        >>> levels = list_optimization_levels()
        >>> "basic" in levels
        True
        >>> "all" in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_OPTIMIZATION_LEVELS)


def list_execution_providers() -> list[str]:
    """List available execution providers.

    Returns:
        Sorted list of execution provider names.

    Examples:
        >>> providers = list_execution_providers()
        >>> "cpu" in providers
        True
        >>> "cuda" in providers
        True
        >>> providers == sorted(providers)
        True
    """
    return sorted(VALID_EXECUTION_PROVIDERS)


def get_opset_version(version: int | str) -> ONNXOpset:
    """Get ONNXOpset enum from version number or string.

    Args:
        version: Opset version number or string.

    Returns:
        Corresponding ONNXOpset enum value.

    Raises:
        ValueError: If version is not valid.

    Examples:
        >>> get_opset_version(14)
        <ONNXOpset.OPSET_14: 14>
        >>> get_opset_version("17")
        <ONNXOpset.OPSET_17: 17>

        >>> get_opset_version(10)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid opset version: 10
    """
    if isinstance(version, str):
        version = int(version)

    if version not in VALID_OPSET_VERSIONS:
        msg = f"invalid opset version: {version}"
        raise ValueError(msg)

    return ONNXOpset(version)


def get_optimization_level(name: str) -> OptimizationLevel:
    """Get OptimizationLevel enum from string name.

    Args:
        name: Name of the optimization level.

    Returns:
        Corresponding OptimizationLevel enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_optimization_level("basic")
        <OptimizationLevel.BASIC: 'basic'>
        >>> get_optimization_level("all")
        <OptimizationLevel.ALL: 'all'>

        >>> get_optimization_level("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid optimization level: invalid
    """
    if name not in VALID_OPTIMIZATION_LEVELS:
        msg = f"invalid optimization level: {name}"
        raise ValueError(msg)

    return OptimizationLevel(name)


def get_execution_provider(name: str) -> ExecutionProvider:
    """Get ExecutionProvider enum from string name.

    Args:
        name: Name of the execution provider.

    Returns:
        Corresponding ExecutionProvider enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_execution_provider("cpu")
        <ExecutionProvider.CPU: 'cpu'>
        >>> get_execution_provider("cuda")
        <ExecutionProvider.CUDA: 'cuda'>

        >>> get_execution_provider("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid execution provider: invalid
    """
    if name not in VALID_EXECUTION_PROVIDERS:
        msg = f"invalid execution provider: {name}"
        raise ValueError(msg)

    return ExecutionProvider(name)


def estimate_onnx_model_size(
    num_parameters: int,
    precision: str = "fp32",
    overhead_factor: float = 1.1,
) -> int:
    """Estimate the size of an exported ONNX model in bytes.

    The estimate includes a configurable overhead factor to account for
    graph structure, metadata, and other ONNX-specific overhead.

    Args:
        num_parameters: Number of model parameters.
        precision: Precision of weights ("fp32", "fp16", "int8"). Defaults to "fp32".
        overhead_factor: Multiplier for overhead. Defaults to 1.1.

    Returns:
        Estimated model size in bytes.

    Raises:
        ValueError: If num_parameters is not positive.
        ValueError: If precision is not valid.
        ValueError: If overhead_factor is less than 1.0.

    Examples:
        >>> size = estimate_onnx_model_size(1_000_000, precision="fp32")
        >>> size > 0
        True
        >>> size_fp16 = estimate_onnx_model_size(1_000_000, precision="fp16")
        >>> size_fp16 < size  # FP16 should be smaller
        True

        >>> estimate_onnx_model_size(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_parameters must be positive

        >>> estimate_onnx_model_size(100, precision="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: precision must be one of...
    """
    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    valid_precisions = {"fp32", "fp16", "int8"}
    if precision not in valid_precisions:
        msg = f"precision must be one of {valid_precisions}, got '{precision}'"
        raise ValueError(msg)

    if overhead_factor < 1.0:
        msg = f"overhead_factor must be >= 1.0, got {overhead_factor}"
        raise ValueError(msg)

    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
    }

    base_size = num_parameters * bytes_per_param[precision]
    return int(base_size * overhead_factor)


def get_recommended_opset(model_type: str) -> ONNXOpset:
    """Get recommended ONNX opset version for a model type.

    Different model architectures may require specific opset versions
    to support all their operations.

    Args:
        model_type: Type of model (e.g., "bert", "gpt2", "llama", "t5").

    Returns:
        Recommended ONNXOpset for the model type.

    Raises:
        ValueError: If model_type is empty.

    Examples:
        >>> get_recommended_opset("bert")
        <ONNXOpset.OPSET_14: 14>
        >>> get_recommended_opset("llama")
        <ONNXOpset.OPSET_17: 17>
        >>> get_recommended_opset("t5")
        <ONNXOpset.OPSET_14: 14>

        >>> get_recommended_opset("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty
    """
    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    model_type = model_type.lower().strip()

    # Modern LLMs typically need higher opset versions
    high_opset_models = {"llama", "mistral", "falcon", "phi", "qwen", "gemma"}

    # Models that work well with opset 14
    medium_opset_models = {
        "bert", "roberta", "gpt2", "t5", "bart", "mbart", "distilbert",
    }

    if model_type in high_opset_models:
        return ONNXOpset.OPSET_17
    elif model_type in medium_opset_models:
        return ONNXOpset.OPSET_14
    else:
        # Default to opset 14 for unknown models
        return ONNXOpset.OPSET_14


def get_export_config_dict(config: ONNXExportConfig) -> dict[str, Any]:
    """Convert ONNXExportConfig to a dictionary for torch.onnx.export.

    Args:
        config: Export configuration.

    Returns:
        Dictionary with export arguments.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_export_config(opset_version=14)
        >>> d = get_export_config_dict(config)
        >>> d["opset_version"]
        14
        >>> d["do_constant_folding"]
        True

        >>> get_export_config_dict(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    result: dict[str, Any] = {
        "opset_version": config.opset_version,
        "do_constant_folding": config.do_constant_folding,
        "input_names": list(config.input_names),
        "output_names": list(config.output_names),
    }

    if config.dynamic_axes is not None:
        result["dynamic_axes"] = config.dynamic_axes

    return result


def get_runtime_session_options(config: RuntimeConfig) -> dict[str, Any]:
    """Convert RuntimeConfig to ONNX Runtime session options.

    Args:
        config: Runtime configuration.

    Returns:
        Dictionary with session option values.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_runtime_config(num_threads=4, enable_profiling=True)
        >>> opts = get_runtime_session_options(config)
        >>> opts["intra_op_num_threads"]
        4
        >>> opts["enable_profiling"]
        True

        >>> get_runtime_session_options(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    # Map OptimizationLevel to ORT GraphOptimizationLevel values
    opt_level_map = {
        OptimizationLevel.DISABLE: 0,
        OptimizationLevel.BASIC: 1,
        OptimizationLevel.EXTENDED: 2,
        OptimizationLevel.ALL: 99,
    }

    return {
        "intra_op_num_threads": config.num_threads,
        "graph_optimization_level": opt_level_map[config.graph_optimization_level],
        "enable_profiling": config.enable_profiling,
    }


def format_model_info(info: ONNXModelInfo) -> str:
    """Format ONNX model information as a human-readable string.

    Args:
        info: Model information to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If info is None.

    Examples:
        >>> info = ONNXModelInfo(
        ...     model_path="/path/to/model.onnx",
        ...     opset_version=14,
        ...     input_shapes={"input_ids": [1, 512]},
        ...     output_shapes={"logits": [1, 512, 768]},
        ...     num_parameters=110_000_000,
        ... )
        >>> formatted = format_model_info(info)
        >>> "Opset: 14" in formatted
        True
        >>> "Parameters: 110,000,000" in formatted
        True

        >>> format_model_info(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: info cannot be None
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    lines = [
        f"Model: {info.model_path}",
        f"Opset: {info.opset_version}",
        f"Parameters: {info.num_parameters:,}",
        "Inputs:",
    ]

    for name, shape in info.input_shapes.items():
        lines.append(f"  {name}: {shape}")

    lines.append("Outputs:")
    for name, shape in info.output_shapes.items():
        lines.append(f"  {name}: {shape}")

    return "\n".join(lines)


def format_export_stats(stats: ExportStats) -> str:
    """Format export statistics as a human-readable string.

    Args:
        stats: Export statistics to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = ExportStats(
        ...     export_time_seconds=5.5,
        ...     model_size_bytes=440_000_000,
        ...     num_nodes=1500,
        ...     num_initializers=200,
        ... )
        >>> formatted = format_export_stats(stats)
        >>> "Export Time:" in formatted
        True
        >>> "Nodes: 1,500" in formatted
        True

        >>> format_export_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    # Format size in appropriate units
    size_mb = stats.model_size_bytes / (1024 * 1024)

    return "\n".join([
        f"Export Time: {stats.export_time_seconds:.2f}s",
        f"Model Size: {size_mb:.1f} MB",
        f"Nodes: {stats.num_nodes:,}",
        f"Initializers: {stats.num_initializers:,}",
    ])
