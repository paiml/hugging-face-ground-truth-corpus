"""TorchScript model compilation utilities.

This module provides utilities for compiling PyTorch models to TorchScript
for optimized inference, including tracing, scripting, and optimization.

Examples:
    >>> from hf_gtc.deployment.torchscript import ScriptMode, TorchScriptConfig
    >>> config = TorchScriptConfig(mode=ScriptMode.TRACE)
    >>> config.mode.value
    'trace'
    >>> config.strict
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ScriptMode(Enum):
    """TorchScript compilation modes.

    Examples:
        >>> ScriptMode.TRACE.value
        'trace'
        >>> ScriptMode.SCRIPT.value
        'script'
        >>> ScriptMode.HYBRID.value
        'hybrid'
    """

    TRACE = "trace"
    SCRIPT = "script"
    HYBRID = "hybrid"


class OptimizeFor(Enum):
    """Optimization target options.

    Examples:
        >>> OptimizeFor.INFERENCE.value
        'inference'
        >>> OptimizeFor.MOBILE.value
        'mobile'
        >>> OptimizeFor.TRAINING.value
        'training'
    """

    INFERENCE = "inference"
    MOBILE = "mobile"
    TRAINING = "training"


class FreezeMode(Enum):
    """Model freezing modes.

    Examples:
        >>> FreezeMode.NONE.value
        'none'
        >>> FreezeMode.PRESERVE_PARAMETERS.value
        'preserve_parameters'
        >>> FreezeMode.FULL.value
        'full'
    """

    NONE = "none"
    PRESERVE_PARAMETERS = "preserve_parameters"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class TorchScriptConfig:
    """TorchScript compilation configuration.

    Attributes:
        mode: Compilation mode (trace, script, or hybrid).
        strict: Enable strict mode for tracing.
        optimize_for: Target optimization (inference, mobile, training).
        check_trace: Verify traced model output matches original.

    Examples:
        >>> config = TorchScriptConfig(mode=ScriptMode.TRACE)
        >>> config.mode
        <ScriptMode.TRACE: 'trace'>
        >>> config.strict
        True

        >>> config2 = TorchScriptConfig(
        ...     mode=ScriptMode.SCRIPT,
        ...     optimize_for=OptimizeFor.MOBILE,
        ... )
        >>> config2.optimize_for
        <OptimizeFor.MOBILE: 'mobile'>
    """

    mode: ScriptMode = ScriptMode.TRACE
    strict: bool = True
    optimize_for: OptimizeFor = OptimizeFor.INFERENCE
    check_trace: bool = True


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Configuration for model tracing.

    Attributes:
        example_inputs: Description of example inputs for tracing.
        check_inputs: Whether to check inputs during tracing.
        check_tolerance: Tolerance for output comparison.

    Examples:
        >>> config = TraceConfig(example_inputs="tensor([1, 2, 3])")
        >>> config.example_inputs
        'tensor([1, 2, 3])'
        >>> config.check_tolerance
        1e-05

        >>> config2 = TraceConfig(
        ...     example_inputs="batch tensor",
        ...     check_inputs=False,
        ...     check_tolerance=1e-4,
        ... )
        >>> config2.check_inputs
        False
    """

    example_inputs: str = ""
    check_inputs: bool = True
    check_tolerance: float = 1e-5


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Configuration for TorchScript optimization passes.

    Attributes:
        freeze_mode: How to freeze the model.
        fuse_operations: Enable operator fusion.
        inline_functions: Inline function calls.
        remove_dropout: Remove dropout layers for inference.

    Examples:
        >>> config = OptimizationConfig(freeze_mode=FreezeMode.FULL)
        >>> config.freeze_mode
        <FreezeMode.FULL: 'full'>
        >>> config.fuse_operations
        True

        >>> config2 = OptimizationConfig(
        ...     freeze_mode=FreezeMode.PRESERVE_PARAMETERS,
        ...     remove_dropout=False,
        ... )
        >>> config2.remove_dropout
        False
    """

    freeze_mode: FreezeMode = FreezeMode.NONE
    fuse_operations: bool = True
    inline_functions: bool = True
    remove_dropout: bool = True


@dataclass(frozen=True, slots=True)
class MobileConfig:
    """Configuration for mobile optimization.

    Attributes:
        optimize_for_mobile: Enable mobile-specific optimizations.
        backend: Target mobile backend (cpu, vulkan, metal).
        preserve_dtype: Preserve original data types.

    Examples:
        >>> config = MobileConfig(optimize_for_mobile=True)
        >>> config.optimize_for_mobile
        True
        >>> config.backend
        'cpu'

        >>> config2 = MobileConfig(backend="vulkan", preserve_dtype=False)
        >>> config2.backend
        'vulkan'
    """

    optimize_for_mobile: bool = False
    backend: str = "cpu"
    preserve_dtype: bool = True


@dataclass(frozen=True, slots=True)
class TorchScriptInfo:
    """Information about a compiled TorchScript model.

    Attributes:
        model_path: Path to the compiled model file.
        script_mode: Compilation mode used.
        graph_size: Number of nodes in the computation graph.
        num_parameters: Total number of model parameters.

    Examples:
        >>> info = TorchScriptInfo(
        ...     model_path="/models/model.pt",
        ...     script_mode=ScriptMode.TRACE,
        ...     graph_size=1500,
        ...     num_parameters=7_000_000_000,
        ... )
        >>> info.graph_size
        1500
        >>> info.num_parameters
        7000000000
    """

    model_path: str
    script_mode: ScriptMode
    graph_size: int
    num_parameters: int


@dataclass(frozen=True, slots=True)
class CompilationStats:
    """Statistics from TorchScript compilation.

    Attributes:
        compile_time_seconds: Time taken to compile.
        graph_nodes: Number of nodes in the final graph.
        fused_ops: Number of fused operations.
        model_size_bytes: Size of compiled model in bytes.

    Examples:
        >>> stats = CompilationStats(
        ...     compile_time_seconds=15.5,
        ...     graph_nodes=1200,
        ...     fused_ops=350,
        ...     model_size_bytes=4_000_000_000,
        ... )
        >>> stats.compile_time_seconds
        15.5
        >>> stats.fused_ops
        350
    """

    compile_time_seconds: float
    graph_nodes: int
    fused_ops: int
    model_size_bytes: int


def validate_torchscript_config(config: TorchScriptConfig) -> None:
    """Validate TorchScript configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = TorchScriptConfig(mode=ScriptMode.TRACE)
        >>> validate_torchscript_config(config)  # No error

        >>> validate_torchscript_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)


def validate_trace_config(config: TraceConfig) -> None:
    """Validate trace configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If check_tolerance is not positive.

    Examples:
        >>> config = TraceConfig(example_inputs="tensor([1])")
        >>> validate_trace_config(config)  # No error

        >>> validate_trace_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = TraceConfig(check_tolerance=-1.0)
        >>> validate_trace_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: check_tolerance must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.check_tolerance <= 0:
        msg = f"check_tolerance must be positive, got {config.check_tolerance}"
        raise ValueError(msg)


def validate_optimization_config(config: OptimizationConfig) -> None:
    """Validate optimization configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = OptimizationConfig(freeze_mode=FreezeMode.FULL)
        >>> validate_optimization_config(config)  # No error

        >>> validate_optimization_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)


def validate_mobile_config(config: MobileConfig) -> None:
    """Validate mobile configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If backend is not valid.

    Examples:
        >>> config = MobileConfig(backend="cpu")
        >>> validate_mobile_config(config)  # No error

        >>> validate_mobile_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = MobileConfig(backend="invalid")
        >>> validate_mobile_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: backend must be one of
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    valid_backends = {"cpu", "vulkan", "metal", "nnapi"}
    if config.backend not in valid_backends:
        msg = f"backend must be one of {valid_backends}, got {config.backend}"
        raise ValueError(msg)


def create_torchscript_config(
    mode: ScriptMode | str = ScriptMode.TRACE,
    strict: bool = True,
    optimize_for: OptimizeFor | str = OptimizeFor.INFERENCE,
    check_trace: bool = True,
) -> TorchScriptConfig:
    """Create a TorchScript configuration.

    Args:
        mode: Compilation mode. Defaults to TRACE.
        strict: Enable strict mode. Defaults to True.
        optimize_for: Target optimization. Defaults to INFERENCE.
        check_trace: Verify traced output. Defaults to True.

    Returns:
        Validated TorchScriptConfig instance.

    Examples:
        >>> config = create_torchscript_config(mode="trace")
        >>> config.mode
        <ScriptMode.TRACE: 'trace'>

        >>> config2 = create_torchscript_config(
        ...     mode=ScriptMode.SCRIPT,
        ...     optimize_for="mobile",
        ... )
        >>> config2.optimize_for
        <OptimizeFor.MOBILE: 'mobile'>
    """
    if isinstance(mode, str):
        mode = get_script_mode(mode)
    if isinstance(optimize_for, str):
        optimize_for = get_optimize_for(optimize_for)

    config = TorchScriptConfig(
        mode=mode,
        strict=strict,
        optimize_for=optimize_for,
        check_trace=check_trace,
    )
    validate_torchscript_config(config)
    return config


def create_trace_config(
    example_inputs: str = "",
    check_inputs: bool = True,
    check_tolerance: float = 1e-5,
) -> TraceConfig:
    """Create a trace configuration.

    Args:
        example_inputs: Description of example inputs. Defaults to "".
        check_inputs: Check inputs during tracing. Defaults to True.
        check_tolerance: Tolerance for comparison. Defaults to 1e-5.

    Returns:
        Validated TraceConfig instance.

    Raises:
        ValueError: If check_tolerance is not positive.

    Examples:
        >>> config = create_trace_config(example_inputs="input tensor")
        >>> config.example_inputs
        'input tensor'

        >>> config2 = create_trace_config(check_tolerance=1e-4)
        >>> config2.check_tolerance
        0.0001

        >>> create_trace_config(check_tolerance=-1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: check_tolerance must be positive
    """
    config = TraceConfig(
        example_inputs=example_inputs,
        check_inputs=check_inputs,
        check_tolerance=check_tolerance,
    )
    validate_trace_config(config)
    return config


def create_optimization_config(
    freeze_mode: FreezeMode | str = FreezeMode.NONE,
    fuse_operations: bool = True,
    inline_functions: bool = True,
    remove_dropout: bool = True,
) -> OptimizationConfig:
    """Create an optimization configuration.

    Args:
        freeze_mode: Freezing mode. Defaults to NONE.
        fuse_operations: Enable fusion. Defaults to True.
        inline_functions: Enable inlining. Defaults to True.
        remove_dropout: Remove dropout. Defaults to True.

    Returns:
        Validated OptimizationConfig instance.

    Examples:
        >>> config = create_optimization_config(freeze_mode="full")
        >>> config.freeze_mode
        <FreezeMode.FULL: 'full'>

        >>> config2 = create_optimization_config(
        ...     freeze_mode=FreezeMode.PRESERVE_PARAMETERS,
        ...     fuse_operations=False,
        ... )
        >>> config2.fuse_operations
        False
    """
    if isinstance(freeze_mode, str):
        freeze_mode = get_freeze_mode(freeze_mode)

    config = OptimizationConfig(
        freeze_mode=freeze_mode,
        fuse_operations=fuse_operations,
        inline_functions=inline_functions,
        remove_dropout=remove_dropout,
    )
    validate_optimization_config(config)
    return config


def create_mobile_config(
    optimize_for_mobile: bool = False,
    backend: str = "cpu",
    preserve_dtype: bool = True,
) -> MobileConfig:
    """Create a mobile configuration.

    Args:
        optimize_for_mobile: Enable mobile optimizations. Defaults to False.
        backend: Target backend. Defaults to "cpu".
        preserve_dtype: Preserve data types. Defaults to True.

    Returns:
        Validated MobileConfig instance.

    Raises:
        ValueError: If backend is not valid.

    Examples:
        >>> config = create_mobile_config(optimize_for_mobile=True)
        >>> config.optimize_for_mobile
        True

        >>> config2 = create_mobile_config(backend="vulkan")
        >>> config2.backend
        'vulkan'

        >>> create_mobile_config(backend="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: backend must be one of
    """
    config = MobileConfig(
        optimize_for_mobile=optimize_for_mobile,
        backend=backend,
        preserve_dtype=preserve_dtype,
    )
    validate_mobile_config(config)
    return config


def list_script_modes() -> list[str]:
    """List available script modes.

    Returns:
        Sorted list of mode names.

    Examples:
        >>> modes = list_script_modes()
        >>> "trace" in modes
        True
        >>> "script" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(m.value for m in ScriptMode)


def list_optimize_for_options() -> list[str]:
    """List available optimization targets.

    Returns:
        Sorted list of target names.

    Examples:
        >>> options = list_optimize_for_options()
        >>> "inference" in options
        True
        >>> "mobile" in options
        True
        >>> options == sorted(options)
        True
    """
    return sorted(o.value for o in OptimizeFor)


def list_freeze_modes() -> list[str]:
    """List available freeze modes.

    Returns:
        Sorted list of freeze mode names.

    Examples:
        >>> modes = list_freeze_modes()
        >>> "none" in modes
        True
        >>> "full" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(f.value for f in FreezeMode)


def get_script_mode(name: str) -> ScriptMode:
    """Get script mode enum from string.

    Args:
        name: Mode name.

    Returns:
        ScriptMode enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_script_mode("trace")
        <ScriptMode.TRACE: 'trace'>
        >>> get_script_mode("script")
        <ScriptMode.SCRIPT: 'script'>

        >>> get_script_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid script mode: invalid
    """
    for mode in ScriptMode:
        if mode.value == name:
            return mode
    msg = f"invalid script mode: {name}"
    raise ValueError(msg)


def get_optimize_for(name: str) -> OptimizeFor:
    """Get optimization target enum from string.

    Args:
        name: Target name.

    Returns:
        OptimizeFor enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_optimize_for("inference")
        <OptimizeFor.INFERENCE: 'inference'>
        >>> get_optimize_for("mobile")
        <OptimizeFor.MOBILE: 'mobile'>

        >>> get_optimize_for("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid optimize_for value: invalid
    """
    for opt in OptimizeFor:
        if opt.value == name:
            return opt
    msg = f"invalid optimize_for value: {name}"
    raise ValueError(msg)


def get_freeze_mode(name: str) -> FreezeMode:
    """Get freeze mode enum from string.

    Args:
        name: Mode name.

    Returns:
        FreezeMode enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_freeze_mode("none")
        <FreezeMode.NONE: 'none'>
        >>> get_freeze_mode("full")
        <FreezeMode.FULL: 'full'>

        >>> get_freeze_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid freeze mode: invalid
    """
    for mode in FreezeMode:
        if mode.value == name:
            return mode
    msg = f"invalid freeze mode: {name}"
    raise ValueError(msg)


def estimate_script_size(
    model_params: int,
    optimize_for: OptimizeFor = OptimizeFor.INFERENCE,
    freeze_mode: FreezeMode = FreezeMode.NONE,
) -> float:
    """Estimate compiled TorchScript model size in MB.

    Args:
        model_params: Number of model parameters.
        optimize_for: Optimization target. Defaults to INFERENCE.
        freeze_mode: Freezing mode. Defaults to NONE.

    Returns:
        Estimated size in MB.

    Raises:
        ValueError: If model_params is not positive.

    Examples:
        >>> size = estimate_script_size(7_000_000_000)
        >>> size > 0
        True
        >>> size < 30000  # Reasonable for 7B params
        True

        >>> estimate_script_size(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive

        >>> size_mobile = estimate_script_size(
        ...     7_000_000_000,
        ...     optimize_for=OptimizeFor.MOBILE,
        ... )
        >>> size_training = estimate_script_size(
        ...     7_000_000_000,
        ...     optimize_for=OptimizeFor.TRAINING,
        ... )
        >>> size_mobile < size_training  # Mobile uses INT8, smaller than FP32
        True
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    # Base: FP32 = 4 bytes per param
    base_bytes = model_params * 4

    # Optimize for mobile typically uses INT8 quantization
    if optimize_for == OptimizeFor.MOBILE:
        base_bytes = model_params * 1  # INT8
    elif optimize_for == OptimizeFor.INFERENCE:
        # FP16 for inference optimization
        base_bytes = model_params * 2

    # Frozen models have ~5% metadata overhead reduction
    if freeze_mode == FreezeMode.FULL:
        base_bytes = int(base_bytes * 0.95)
    elif freeze_mode == FreezeMode.PRESERVE_PARAMETERS:
        base_bytes = int(base_bytes * 0.98)

    # Add ~2% overhead for TorchScript metadata
    size_bytes = int(base_bytes * 1.02)
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def check_scriptable(
    model_type: str,
    has_control_flow: bool = False,
    has_dynamic_shapes: bool = False,
) -> bool:
    """Check if a model can be scripted.

    Evaluates whether a model with the given characteristics can be
    successfully compiled to TorchScript.

    Args:
        model_type: Type of model (e.g., "transformer", "cnn", "rnn").
        has_control_flow: Model has data-dependent control flow.
        has_dynamic_shapes: Model has dynamic tensor shapes.

    Returns:
        True if model can be scripted, False otherwise.

    Examples:
        >>> check_scriptable("transformer")
        True
        >>> check_scriptable("cnn", has_control_flow=False)
        True
        >>> check_scriptable("custom", has_control_flow=True, has_dynamic_shapes=True)
        False

        >>> check_scriptable("")  # Empty type
        False
    """
    if not model_type:
        return False

    # Known scriptable model types
    scriptable_types = {
        "transformer",
        "cnn",
        "rnn",
        "lstm",
        "gru",
        "mlp",
        "resnet",
        "vgg",
        "bert",
        "gpt",
    }

    model_type_lower = model_type.lower()

    # Check if type is in known scriptable list
    if model_type_lower not in scriptable_types:
        # Unknown types are potentially scriptable if no dynamic features
        return not (has_control_flow and has_dynamic_shapes)

    # Known types with dynamic features may still have issues
    return not (has_control_flow and has_dynamic_shapes)


def get_torchscript_config_dict(config: TorchScriptConfig) -> dict[str, Any]:
    """Convert TorchScriptConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_torchscript_config(mode="trace")
        >>> d = get_torchscript_config_dict(config)
        >>> d["mode"]
        'trace'
        >>> d["strict"]
        True
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "mode": config.mode.value,
        "strict": config.strict,
        "optimize_for": config.optimize_for.value,
        "check_trace": config.check_trace,
    }


def get_trace_config_dict(config: TraceConfig) -> dict[str, Any]:
    """Convert TraceConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_trace_config(example_inputs="tensor")
        >>> d = get_trace_config_dict(config)
        >>> d["example_inputs"]
        'tensor'
        >>> d["check_inputs"]
        True
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "example_inputs": config.example_inputs,
        "check_inputs": config.check_inputs,
        "check_tolerance": config.check_tolerance,
    }


def get_optimization_config_dict(config: OptimizationConfig) -> dict[str, Any]:
    """Convert OptimizationConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_optimization_config(freeze_mode="full")
        >>> d = get_optimization_config_dict(config)
        >>> d["freeze_mode"]
        'full'
        >>> d["fuse_operations"]
        True
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "freeze_mode": config.freeze_mode.value,
        "fuse_operations": config.fuse_operations,
        "inline_functions": config.inline_functions,
        "remove_dropout": config.remove_dropout,
    }


def get_mobile_config_dict(config: MobileConfig) -> dict[str, Any]:
    """Convert MobileConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_mobile_config(backend="vulkan")
        >>> d = get_mobile_config_dict(config)
        >>> d["backend"]
        'vulkan'
        >>> d["preserve_dtype"]
        True
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "optimize_for_mobile": config.optimize_for_mobile,
        "backend": config.backend,
        "preserve_dtype": config.preserve_dtype,
    }


def format_torchscript_info(info: TorchScriptInfo) -> str:
    """Format TorchScript model info for display.

    Args:
        info: Info to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If info is None.

    Examples:
        >>> info = TorchScriptInfo(
        ...     model_path="/models/model.pt",
        ...     script_mode=ScriptMode.TRACE,
        ...     graph_size=1500,
        ...     num_parameters=7_000_000_000,
        ... )
        >>> formatted = format_torchscript_info(info)
        >>> "Path:" in formatted
        True
        >>> "Mode:" in formatted
        True
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    lines = [
        f"Path: {info.model_path}",
        f"Mode: {info.script_mode.value}",
        f"Graph size: {info.graph_size} nodes",
        f"Parameters: {info.num_parameters:,}",
    ]
    return "\n".join(lines)


def format_compilation_stats(stats: CompilationStats) -> str:
    """Format compilation statistics for display.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = CompilationStats(
        ...     compile_time_seconds=15.5,
        ...     graph_nodes=1200,
        ...     fused_ops=350,
        ...     model_size_bytes=4_000_000_000,
        ... )
        >>> formatted = format_compilation_stats(stats)
        >>> "Compile time:" in formatted
        True
        >>> "Fused ops:" in formatted
        True
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    size_mb = stats.model_size_bytes / (1024 * 1024)
    lines = [
        f"Compile time: {stats.compile_time_seconds:.2f}s",
        f"Graph nodes: {stats.graph_nodes}",
        f"Fused ops: {stats.fused_ops}",
        f"Model size: {size_mb:.1f} MB",
    ]
    return "\n".join(lines)


def create_torchscript_info(
    model_path: str,
    script_mode: ScriptMode | str,
    graph_size: int,
    num_parameters: int,
) -> TorchScriptInfo:
    """Create a TorchScriptInfo instance.

    Args:
        model_path: Path to the model file.
        script_mode: Compilation mode used.
        graph_size: Number of graph nodes.
        num_parameters: Number of parameters.

    Returns:
        TorchScriptInfo instance.

    Raises:
        ValueError: If model_path is empty.
        ValueError: If graph_size is negative.
        ValueError: If num_parameters is not positive.

    Examples:
        >>> info = create_torchscript_info(
        ...     "/models/model.pt", "trace", 1500, 7_000_000_000
        ... )
        >>> info.script_mode
        <ScriptMode.TRACE: 'trace'>

        >>> create_torchscript_info("", "trace", 100, 1000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_path cannot be empty
    """
    if not model_path:
        msg = "model_path cannot be empty"
        raise ValueError(msg)

    if graph_size < 0:
        msg = f"graph_size cannot be negative, got {graph_size}"
        raise ValueError(msg)

    if num_parameters <= 0:
        msg = f"num_parameters must be positive, got {num_parameters}"
        raise ValueError(msg)

    if isinstance(script_mode, str):
        script_mode = get_script_mode(script_mode)

    return TorchScriptInfo(
        model_path=model_path,
        script_mode=script_mode,
        graph_size=graph_size,
        num_parameters=num_parameters,
    )


def create_compilation_stats(
    compile_time_seconds: float,
    graph_nodes: int,
    fused_ops: int,
    model_size_bytes: int,
) -> CompilationStats:
    """Create a CompilationStats instance.

    Args:
        compile_time_seconds: Compilation time.
        graph_nodes: Number of graph nodes.
        fused_ops: Number of fused operations.
        model_size_bytes: Model size in bytes.

    Returns:
        CompilationStats instance.

    Raises:
        ValueError: If compile_time_seconds is negative.
        ValueError: If graph_nodes is negative.
        ValueError: If fused_ops is negative.
        ValueError: If model_size_bytes is not positive.

    Examples:
        >>> stats = create_compilation_stats(15.5, 1200, 350, 4_000_000_000)
        >>> stats.compile_time_seconds
        15.5

        >>> create_compilation_stats(-1.0, 100, 10, 1000)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: compile_time_seconds cannot be negative
    """
    if compile_time_seconds < 0:
        msg = f"compile_time_seconds cannot be negative, got {compile_time_seconds}"
        raise ValueError(msg)

    if graph_nodes < 0:
        msg = f"graph_nodes cannot be negative, got {graph_nodes}"
        raise ValueError(msg)

    if fused_ops < 0:
        msg = f"fused_ops cannot be negative, got {fused_ops}"
        raise ValueError(msg)

    if model_size_bytes <= 0:
        msg = f"model_size_bytes must be positive, got {model_size_bytes}"
        raise ValueError(msg)

    return CompilationStats(
        compile_time_seconds=compile_time_seconds,
        graph_nodes=graph_nodes,
        fused_ops=fused_ops,
        model_size_bytes=model_size_bytes,
    )


def get_recommended_config(
    model_type: str,
    target_platform: str = "server",
) -> TorchScriptConfig:
    """Get recommended TorchScript config for model and platform.

    Args:
        model_type: Type of model (e.g., "transformer", "cnn").
        target_platform: Target platform ("server", "mobile", "edge").

    Returns:
        Recommended TorchScriptConfig.

    Raises:
        ValueError: If model_type is empty.
        ValueError: If target_platform is not recognized.

    Examples:
        >>> config = get_recommended_config("transformer")
        >>> config.mode
        <ScriptMode.TRACE: 'trace'>

        >>> config_mobile = get_recommended_config("cnn", "mobile")
        >>> config_mobile.optimize_for
        <OptimizeFor.MOBILE: 'mobile'>

        >>> get_recommended_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type cannot be empty
    """
    if not model_type:
        msg = "model_type cannot be empty"
        raise ValueError(msg)

    valid_platforms = {"server", "mobile", "edge"}
    if target_platform not in valid_platforms:
        msg = f"target_platform must be one of {valid_platforms}, got {target_platform}"
        raise ValueError(msg)

    model_type_lower = model_type.lower()

    # Determine best mode based on model type
    # RNNs and models with control flow work better with script
    if model_type_lower in ("rnn", "lstm", "gru"):
        mode = ScriptMode.SCRIPT
    else:
        mode = ScriptMode.TRACE

    # Determine optimization target based on platform
    if target_platform == "mobile":
        optimize_for = OptimizeFor.MOBILE
    elif target_platform == "edge":
        optimize_for = OptimizeFor.INFERENCE
    else:
        optimize_for = OptimizeFor.INFERENCE

    return create_torchscript_config(
        mode=mode,
        strict=True,
        optimize_for=optimize_for,
        check_trace=True,
    )
