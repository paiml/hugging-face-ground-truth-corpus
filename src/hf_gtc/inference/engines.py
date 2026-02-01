"""Inference engine configurations for LLM serving.

This module provides utilities for configuring various inference engines
including vLLM, TGI, llama.cpp, TensorRT, CTranslate2, and ONNX Runtime.
It focuses on engine configuration, performance estimation, and compatibility
checking.

Examples:
    >>> from hf_gtc.inference.engines import create_vllm_config
    >>> config = create_vllm_config(tensor_parallel_size=4)
    >>> config.tensor_parallel_size
    4
    >>> config.gpu_memory_utilization
    0.9
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


class EngineType(Enum):
    """Inference engine types.

    Attributes:
        VLLM: vLLM high-throughput LLM serving engine.
        TGI: Text Generation Inference by HuggingFace.
        LLAMACPP: llama.cpp for efficient CPU/GPU inference.
        TENSORRT: NVIDIA TensorRT for optimized inference.
        CTRANSLATE2: CTranslate2 for efficient transformer inference.
        ONNXRUNTIME: ONNX Runtime for cross-platform inference.

    Examples:
        >>> EngineType.VLLM.value
        'vllm'
        >>> EngineType.TGI.value
        'tgi'
        >>> EngineType.LLAMACPP.value
        'llamacpp'
    """

    VLLM = "vllm"
    TGI = "tgi"
    LLAMACPP = "llamacpp"
    TENSORRT = "tensorrt"
    CTRANSLATE2 = "ctranslate2"
    ONNXRUNTIME = "onnxruntime"


VALID_ENGINE_TYPES = frozenset(e.value for e in EngineType)


class QuantizationBackend(Enum):
    """Quantization backend types.

    Attributes:
        NONE: No quantization applied.
        BITSANDBYTES: bitsandbytes library for 4/8-bit quantization.
        GPTQ: GPTQ quantization method.
        AWQ: Activation-aware Weight Quantization.
        GGML: GGML quantization (for llama.cpp).

    Examples:
        >>> QuantizationBackend.NONE.value
        'none'
        >>> QuantizationBackend.BITSANDBYTES.value
        'bitsandbytes'
        >>> QuantizationBackend.GPTQ.value
        'gptq'
    """

    NONE = "none"
    BITSANDBYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    GGML = "ggml"


VALID_QUANTIZATION_BACKENDS = frozenset(q.value for q in QuantizationBackend)


class EngineFeature(Enum):
    """Inference engine feature flags.

    Attributes:
        STREAMING: Supports streaming token generation.
        BATCHING: Supports continuous batching.
        SPECULATIVE: Supports speculative decoding.
        PREFIX_CACHING: Supports prefix/prompt caching.

    Examples:
        >>> EngineFeature.STREAMING.value
        'streaming'
        >>> EngineFeature.BATCHING.value
        'batching'
        >>> EngineFeature.SPECULATIVE.value
        'speculative'
    """

    STREAMING = "streaming"
    BATCHING = "batching"
    SPECULATIVE = "speculative"
    PREFIX_CACHING = "prefix_caching"


VALID_ENGINE_FEATURES = frozenset(f.value for f in EngineFeature)


# Type aliases
EngineTypeStr = Literal[
    "vllm", "tgi", "llamacpp", "tensorrt", "ctranslate2", "onnxruntime"
]
QuantizationBackendStr = Literal["none", "bitsandbytes", "gptq", "awq", "ggml"]
EngineFeatureStr = Literal["streaming", "batching", "speculative", "prefix_caching"]
ModelSizeStr = Literal["small", "medium", "large", "xlarge"]
HardwareTypeStr = Literal["cpu", "gpu_consumer", "gpu_datacenter"]


@dataclass(frozen=True, slots=True)
class VLLMConfig:
    """Configuration for vLLM inference engine.

    Attributes:
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
        max_model_len: Maximum model context length.
        enforce_eager: Disable CUDA graphs for debugging.

    Examples:
        >>> config = VLLMConfig(
        ...     tensor_parallel_size=4,
        ...     gpu_memory_utilization=0.9,
        ...     max_model_len=4096,
        ...     enforce_eager=False,
        ... )
        >>> config.tensor_parallel_size
        4
        >>> config.gpu_memory_utilization
        0.9
    """

    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool


@dataclass(frozen=True, slots=True)
class TGIConfig:
    """Configuration for Text Generation Inference.

    Attributes:
        max_concurrent_requests: Maximum concurrent requests.
        max_input_length: Maximum input sequence length.
        max_total_tokens: Maximum total tokens (input + output).

    Examples:
        >>> config = TGIConfig(
        ...     max_concurrent_requests=128,
        ...     max_input_length=2048,
        ...     max_total_tokens=4096,
        ... )
        >>> config.max_concurrent_requests
        128
        >>> config.max_input_length
        2048
    """

    max_concurrent_requests: int
    max_input_length: int
    max_total_tokens: int


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    """Configuration for llama.cpp inference.

    Attributes:
        n_ctx: Context window size.
        n_batch: Batch size for prompt processing.
        n_threads: Number of CPU threads to use.
        use_mmap: Use memory-mapped files for model loading.

    Examples:
        >>> config = LlamaCppConfig(
        ...     n_ctx=4096,
        ...     n_batch=512,
        ...     n_threads=8,
        ...     use_mmap=True,
        ... )
        >>> config.n_ctx
        4096
        >>> config.n_threads
        8
    """

    n_ctx: int
    n_batch: int
    n_threads: int
    use_mmap: bool


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """General engine configuration.

    Attributes:
        engine_type: Type of inference engine.
        vllm_config: vLLM-specific configuration (optional).
        tgi_config: TGI-specific configuration (optional).
        llamacpp_config: llama.cpp-specific configuration (optional).
        quantization: Quantization backend to use.

    Examples:
        >>> vllm = VLLMConfig(4, 0.9, 4096, False)
        >>> config = EngineConfig(
        ...     engine_type=EngineType.VLLM,
        ...     vllm_config=vllm,
        ...     tgi_config=None,
        ...     llamacpp_config=None,
        ...     quantization=QuantizationBackend.NONE,
        ... )
        >>> config.engine_type
        <EngineType.VLLM: 'vllm'>
        >>> config.vllm_config.tensor_parallel_size
        4
    """

    engine_type: EngineType
    vllm_config: VLLMConfig | None
    tgi_config: TGIConfig | None
    llamacpp_config: LlamaCppConfig | None
    quantization: QuantizationBackend


@dataclass(frozen=True, slots=True)
class EngineStats:
    """Statistics from inference engine.

    Attributes:
        throughput_tokens_per_sec: Token generation throughput.
        latency_ms: Average latency in milliseconds.
        memory_usage_gb: GPU/CPU memory usage in gigabytes.

    Examples:
        >>> stats = EngineStats(
        ...     throughput_tokens_per_sec=1500.0,
        ...     latency_ms=50.0,
        ...     memory_usage_gb=24.0,
        ... )
        >>> stats.throughput_tokens_per_sec
        1500.0
        >>> stats.latency_ms
        50.0
    """

    throughput_tokens_per_sec: float
    latency_ms: float
    memory_usage_gb: float


def validate_vllm_config(config: VLLMConfig) -> None:
    """Validate vLLM configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = VLLMConfig(4, 0.9, 4096, False)
        >>> validate_vllm_config(config)  # No error

        >>> bad = VLLMConfig(0, 0.9, 4096, False)
        >>> validate_vllm_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tensor_parallel_size must be positive
    """
    if config.tensor_parallel_size <= 0:
        msg = (
            f"tensor_parallel_size must be positive, got {config.tensor_parallel_size}"
        )
        raise ValueError(msg)

    if not 0.0 < config.gpu_memory_utilization <= 1.0:
        msg = (
            f"gpu_memory_utilization must be in (0.0, 1.0], "
            f"got {config.gpu_memory_utilization}"
        )
        raise ValueError(msg)

    if config.max_model_len <= 0:
        msg = f"max_model_len must be positive, got {config.max_model_len}"
        raise ValueError(msg)


def validate_tgi_config(config: TGIConfig) -> None:
    """Validate TGI configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = TGIConfig(128, 2048, 4096)
        >>> validate_tgi_config(config)  # No error

        >>> bad = TGIConfig(0, 2048, 4096)
        >>> validate_tgi_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_concurrent_requests must be positive
    """
    if config.max_concurrent_requests <= 0:
        msg = (
            f"max_concurrent_requests must be positive, "
            f"got {config.max_concurrent_requests}"
        )
        raise ValueError(msg)

    if config.max_input_length <= 0:
        msg = f"max_input_length must be positive, got {config.max_input_length}"
        raise ValueError(msg)

    if config.max_total_tokens <= 0:
        msg = f"max_total_tokens must be positive, got {config.max_total_tokens}"
        raise ValueError(msg)

    if config.max_total_tokens < config.max_input_length:
        msg = (
            f"max_total_tokens must be >= max_input_length, "
            f"got {config.max_total_tokens} < {config.max_input_length}"
        )
        raise ValueError(msg)


def validate_llamacpp_config(config: LlamaCppConfig) -> None:
    """Validate llama.cpp configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = LlamaCppConfig(4096, 512, 8, True)
        >>> validate_llamacpp_config(config)  # No error

        >>> bad = LlamaCppConfig(0, 512, 8, True)
        >>> validate_llamacpp_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: n_ctx must be positive
    """
    if config.n_ctx <= 0:
        msg = f"n_ctx must be positive, got {config.n_ctx}"
        raise ValueError(msg)

    if config.n_batch <= 0:
        msg = f"n_batch must be positive, got {config.n_batch}"
        raise ValueError(msg)

    if config.n_threads <= 0:
        msg = f"n_threads must be positive, got {config.n_threads}"
        raise ValueError(msg)

    if config.n_batch > config.n_ctx:
        msg = f"n_batch must be <= n_ctx, got {config.n_batch} > {config.n_ctx}"
        raise ValueError(msg)


def _validate_engine_type_config(config: EngineConfig) -> None:
    """Validate engine-specific sub-config based on engine type."""
    engine_validators: dict[EngineType, tuple[str, str, Callable[[Any], None]]] = {
        EngineType.VLLM: ("vllm_config", "VLLM engine", validate_vllm_config),
        EngineType.TGI: ("tgi_config", "TGI engine", validate_tgi_config),
        EngineType.LLAMACPP: (
            "llamacpp_config",
            "LLAMACPP engine",
            validate_llamacpp_config,
        ),
    }
    entry = engine_validators.get(config.engine_type)
    if entry is None:
        return
    attr_name, label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} is required for {label}"
        raise ValueError(msg)
    validator(sub_config)


def validate_engine_config(config: EngineConfig) -> None:
    """Validate engine configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> vllm = VLLMConfig(4, 0.9, 4096, False)
        >>> config = EngineConfig(
        ...     EngineType.VLLM, vllm, None, None,
        ...     QuantizationBackend.NONE,
        ... )
        >>> validate_engine_config(config)  # No error

        >>> bad_vllm = VLLMConfig(0, 0.9, 4096, False)
        >>> bad_config = EngineConfig(
        ...     EngineType.VLLM, bad_vllm, None, None,
        ...     QuantizationBackend.NONE,
        ... )
        >>> validate_engine_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tensor_parallel_size must be positive
    """
    # Validate engine-specific config based on type
    _validate_engine_type_config(config)

    # Check quantization compatibility
    if config.engine_type == EngineType.LLAMACPP and config.quantization not in (
        QuantizationBackend.NONE,
        QuantizationBackend.GGML,
    ):
        msg = (
            f"LLAMACPP only supports NONE or GGML quantization, "
            f"got {config.quantization.value}"
        )
        raise ValueError(msg)


def validate_engine_stats(stats: EngineStats) -> None:
    """Validate engine statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = EngineStats(1500.0, 50.0, 24.0)
        >>> validate_engine_stats(stats)  # No error

        >>> bad = EngineStats(-1.0, 50.0, 24.0)
        >>> validate_engine_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: throughput_tokens_per_sec cannot be negative
    """
    if stats.throughput_tokens_per_sec < 0:
        msg = (
            f"throughput_tokens_per_sec cannot be negative, "
            f"got {stats.throughput_tokens_per_sec}"
        )
        raise ValueError(msg)

    if stats.latency_ms < 0:
        msg = f"latency_ms cannot be negative, got {stats.latency_ms}"
        raise ValueError(msg)

    if stats.memory_usage_gb < 0:
        msg = f"memory_usage_gb cannot be negative, got {stats.memory_usage_gb}"
        raise ValueError(msg)


def create_vllm_config(
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
) -> VLLMConfig:
    """Create a vLLM configuration.

    Args:
        tensor_parallel_size: Number of GPUs. Defaults to 1.
        gpu_memory_utilization: Memory fraction. Defaults to 0.9.
        max_model_len: Maximum context length. Defaults to 4096.
        enforce_eager: Disable CUDA graphs. Defaults to False.

    Returns:
        VLLMConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_vllm_config(tensor_parallel_size=4)
        >>> config.tensor_parallel_size
        4
        >>> config.gpu_memory_utilization
        0.9

        >>> config = create_vllm_config(max_model_len=8192)
        >>> config.max_model_len
        8192

        >>> create_vllm_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     tensor_parallel_size=0
        ... )
        Traceback (most recent call last):
        ValueError: tensor_parallel_size must be positive
    """
    config = VLLMConfig(
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )
    validate_vllm_config(config)
    return config


def create_tgi_config(
    max_concurrent_requests: int = 128,
    max_input_length: int = 2048,
    max_total_tokens: int = 4096,
) -> TGIConfig:
    """Create a TGI configuration.

    Args:
        max_concurrent_requests: Concurrent requests. Defaults to 128.
        max_input_length: Input length limit. Defaults to 2048.
        max_total_tokens: Total token limit. Defaults to 4096.

    Returns:
        TGIConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_tgi_config(max_concurrent_requests=256)
        >>> config.max_concurrent_requests
        256
        >>> config.max_input_length
        2048

        >>> config = create_tgi_config(max_total_tokens=8192)
        >>> config.max_total_tokens
        8192

        >>> create_tgi_config(max_concurrent_requests=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_concurrent_requests must be positive
    """
    config = TGIConfig(
        max_concurrent_requests=max_concurrent_requests,
        max_input_length=max_input_length,
        max_total_tokens=max_total_tokens,
    )
    validate_tgi_config(config)
    return config


def create_llamacpp_config(
    n_ctx: int = 4096,
    n_batch: int = 512,
    n_threads: int = 8,
    use_mmap: bool = True,
) -> LlamaCppConfig:
    """Create a llama.cpp configuration.

    Args:
        n_ctx: Context window size. Defaults to 4096.
        n_batch: Batch size. Defaults to 512.
        n_threads: CPU threads. Defaults to 8.
        use_mmap: Use memory mapping. Defaults to True.

    Returns:
        LlamaCppConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_llamacpp_config(n_ctx=8192)
        >>> config.n_ctx
        8192
        >>> config.n_batch
        512

        >>> config = create_llamacpp_config(n_threads=16)
        >>> config.n_threads
        16

        >>> create_llamacpp_config(n_ctx=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: n_ctx must be positive
    """
    config = LlamaCppConfig(
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads,
        use_mmap=use_mmap,
    )
    validate_llamacpp_config(config)
    return config


def create_engine_config(
    engine_type: EngineTypeStr = "vllm",
    vllm_config: VLLMConfig | None = None,
    tgi_config: TGIConfig | None = None,
    llamacpp_config: LlamaCppConfig | None = None,
    quantization: QuantizationBackendStr = "none",
) -> EngineConfig:
    """Create a general engine configuration.

    Args:
        engine_type: Type of engine. Defaults to "vllm".
        vllm_config: vLLM configuration. Created if needed for VLLM.
        tgi_config: TGI configuration. Created if needed for TGI.
        llamacpp_config: llama.cpp configuration. Created if needed for LLAMACPP.
        quantization: Quantization backend. Defaults to "none".

    Returns:
        EngineConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_engine_config(engine_type="vllm")
        >>> config.engine_type
        <EngineType.VLLM: 'vllm'>
        >>> config.vllm_config is not None
        True

        >>> vllm = create_vllm_config(tensor_parallel_size=4)
        >>> config = create_engine_config(engine_type="vllm", vllm_config=vllm)
        >>> config.vllm_config.tensor_parallel_size
        4

        >>> config = create_engine_config(engine_type="tgi")
        >>> config.engine_type
        <EngineType.TGI: 'tgi'>

        >>> create_engine_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     engine_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: engine_type must be one of
    """
    if engine_type not in VALID_ENGINE_TYPES:
        msg = f"engine_type must be one of {VALID_ENGINE_TYPES}, got '{engine_type}'"
        raise ValueError(msg)

    if quantization not in VALID_QUANTIZATION_BACKENDS:
        msg = (
            f"quantization must be one of {VALID_QUANTIZATION_BACKENDS}, "
            f"got '{quantization}'"
        )
        raise ValueError(msg)

    engine = EngineType(engine_type)
    quant = QuantizationBackend(quantization)

    # Create default configs if needed
    if engine == EngineType.VLLM and vllm_config is None:
        vllm_config = create_vllm_config()
    elif engine == EngineType.TGI and tgi_config is None:
        tgi_config = create_tgi_config()
    elif engine == EngineType.LLAMACPP and llamacpp_config is None:
        llamacpp_config = create_llamacpp_config()

    config = EngineConfig(
        engine_type=engine,
        vllm_config=vllm_config,
        tgi_config=tgi_config,
        llamacpp_config=llamacpp_config,
        quantization=quant,
    )
    validate_engine_config(config)
    return config


def create_engine_stats(
    throughput_tokens_per_sec: float = 0.0,
    latency_ms: float = 0.0,
    memory_usage_gb: float = 0.0,
) -> EngineStats:
    """Create engine statistics.

    Args:
        throughput_tokens_per_sec: Token throughput. Defaults to 0.0.
        latency_ms: Average latency. Defaults to 0.0.
        memory_usage_gb: Memory usage. Defaults to 0.0.

    Returns:
        EngineStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_engine_stats(throughput_tokens_per_sec=1500.0)
        >>> stats.throughput_tokens_per_sec
        1500.0

        >>> stats = create_engine_stats(latency_ms=50.0, memory_usage_gb=24.0)
        >>> stats.latency_ms
        50.0
        >>> stats.memory_usage_gb
        24.0

        >>> create_engine_stats(throughput_tokens_per_sec=-1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: throughput_tokens_per_sec cannot be negative
    """
    stats = EngineStats(
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        latency_ms=latency_ms,
        memory_usage_gb=memory_usage_gb,
    )
    validate_engine_stats(stats)
    return stats


def list_engine_types() -> list[str]:
    """List available engine types.

    Returns:
        Sorted list of engine type names.

    Examples:
        >>> engines = list_engine_types()
        >>> "vllm" in engines
        True
        >>> "tgi" in engines
        True
        >>> engines == sorted(engines)
        True
    """
    return sorted(VALID_ENGINE_TYPES)


def list_quantization_backends() -> list[str]:
    """List available quantization backends.

    Returns:
        Sorted list of quantization backend names.

    Examples:
        >>> backends = list_quantization_backends()
        >>> "none" in backends
        True
        >>> "gptq" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_QUANTIZATION_BACKENDS)


def list_engine_features() -> list[str]:
    """List available engine features.

    Returns:
        Sorted list of engine feature names.

    Examples:
        >>> features = list_engine_features()
        >>> "streaming" in features
        True
        >>> "batching" in features
        True
        >>> features == sorted(features)
        True
    """
    return sorted(VALID_ENGINE_FEATURES)


def get_engine_type(name: str) -> EngineType:
    """Get an engine type by name.

    Args:
        name: Name of the engine type.

    Returns:
        The corresponding EngineType enum value.

    Raises:
        ValueError: If name is not a valid engine type.

    Examples:
        >>> get_engine_type("vllm")
        <EngineType.VLLM: 'vllm'>
        >>> get_engine_type("tgi")
        <EngineType.TGI: 'tgi'>

        >>> get_engine_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown engine type
    """
    if name not in VALID_ENGINE_TYPES:
        msg = f"Unknown engine type: '{name}'. Valid: {VALID_ENGINE_TYPES}"
        raise ValueError(msg)
    return EngineType(name)


def get_quantization_backend(name: str) -> QuantizationBackend:
    """Get a quantization backend by name.

    Args:
        name: Name of the quantization backend.

    Returns:
        The corresponding QuantizationBackend enum value.

    Raises:
        ValueError: If name is not a valid quantization backend.

    Examples:
        >>> get_quantization_backend("none")
        <QuantizationBackend.NONE: 'none'>
        >>> get_quantization_backend("gptq")
        <QuantizationBackend.GPTQ: 'gptq'>

        >>> get_quantization_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown quantization backend
    """
    if name not in VALID_QUANTIZATION_BACKENDS:
        msg = (
            f"Unknown quantization backend: '{name}'. "
            f"Valid: {VALID_QUANTIZATION_BACKENDS}"
        )
        raise ValueError(msg)
    return QuantizationBackend(name)


def get_engine_feature(name: str) -> EngineFeature:
    """Get an engine feature by name.

    Args:
        name: Name of the engine feature.

    Returns:
        The corresponding EngineFeature enum value.

    Raises:
        ValueError: If name is not a valid engine feature.

    Examples:
        >>> get_engine_feature("streaming")
        <EngineFeature.STREAMING: 'streaming'>
        >>> get_engine_feature("batching")
        <EngineFeature.BATCHING: 'batching'>

        >>> get_engine_feature("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown engine feature
    """
    if name not in VALID_ENGINE_FEATURES:
        msg = f"Unknown engine feature: '{name}'. Valid: {VALID_ENGINE_FEATURES}"
        raise ValueError(msg)
    return EngineFeature(name)


def estimate_engine_throughput(
    engine_type: EngineTypeStr,
    model_params_billions: float,
    batch_size: int,
    gpu_count: int = 1,
    quantization: QuantizationBackendStr = "none",
) -> float:
    """Estimate throughput for an inference engine.

    Args:
        engine_type: Type of inference engine.
        model_params_billions: Model size in billions of parameters.
        batch_size: Batch size for inference.
        gpu_count: Number of GPUs. Defaults to 1.
        quantization: Quantization backend. Defaults to "none".

    Returns:
        Estimated throughput in tokens per second.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> throughput = estimate_engine_throughput(
        ...     engine_type="vllm",
        ...     model_params_billions=7.0,
        ...     batch_size=16,
        ...     gpu_count=1,
        ... )
        >>> throughput > 0
        True

        >>> # vLLM typically has higher throughput with more GPUs
        >>> t1 = estimate_engine_throughput("vllm", 7.0, 16, 1)
        >>> t2 = estimate_engine_throughput("vllm", 7.0, 16, 4)
        >>> t2 > t1
        True

        >>> estimate_engine_throughput("invalid", 7.0, 16)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: engine_type must be one of
    """
    if engine_type not in VALID_ENGINE_TYPES:
        msg = f"engine_type must be one of {VALID_ENGINE_TYPES}, got '{engine_type}'"
        raise ValueError(msg)

    if model_params_billions <= 0:
        msg = f"model_params_billions must be positive, got {model_params_billions}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if gpu_count <= 0:
        msg = f"gpu_count must be positive, got {gpu_count}"
        raise ValueError(msg)

    if quantization not in VALID_QUANTIZATION_BACKENDS:
        msg = (
            f"quantization must be one of {VALID_QUANTIZATION_BACKENDS}, "
            f"got '{quantization}'"
        )
        raise ValueError(msg)

    # Base throughput factors per engine (relative efficiency)
    engine_factors = {
        "vllm": 1.0,  # Baseline
        "tgi": 0.9,  # Slightly lower
        "llamacpp": 0.5,  # CPU-focused, lower GPU throughput
        "tensorrt": 1.2,  # Optimized for NVIDIA
        "ctranslate2": 0.7,
        "onnxruntime": 0.6,
    }

    # Quantization throughput multipliers
    quant_multipliers = {
        "none": 1.0,
        "bitsandbytes": 1.5,  # Memory savings allow larger batches
        "gptq": 1.6,
        "awq": 1.7,
        "ggml": 1.4,
    }

    # Base calculation: larger models are slower
    # Roughly inverse relationship with model size
    base_tokens_per_sec = 1000.0 * (7.0 / model_params_billions)

    # Apply batch size scaling (roughly sqrt scaling due to memory bandwidth)
    batch_factor = (batch_size / 8) ** 0.5

    # Apply GPU scaling (roughly linear with some overhead)
    gpu_factor = gpu_count * 0.85  # 85% efficiency for multi-GPU

    # Apply engine and quantization factors
    engine_factor = engine_factors[engine_type]
    quant_factor = quant_multipliers[quantization]

    throughput = (
        base_tokens_per_sec * batch_factor * gpu_factor * engine_factor * quant_factor
    )
    return round(throughput, 2)


def compare_engine_performance(
    engine_a: EngineTypeStr,
    engine_b: EngineTypeStr,
    model_params_billions: float,
    batch_size: int,
) -> dict[str, float]:
    """Compare performance between two engines.

    Args:
        engine_a: First engine type.
        engine_b: Second engine type.
        model_params_billions: Model size in billions of parameters.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with throughput for each engine and ratio.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> result = compare_engine_performance("vllm", "tgi", 7.0, 16)
        >>> "engine_a_throughput" in result
        True
        >>> "engine_b_throughput" in result
        True
        >>> "ratio_a_to_b" in result
        True
        >>> result["ratio_a_to_b"] > 0
        True

        >>> compare_engine_performance("invalid", "tgi", 7.0, 16)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: engine_type must be one of
    """
    throughput_a = estimate_engine_throughput(
        engine_a, model_params_billions, batch_size
    )
    throughput_b = estimate_engine_throughput(
        engine_b, model_params_billions, batch_size
    )

    ratio = throughput_a / throughput_b if throughput_b > 0 else 0.0

    return {
        "engine_a_throughput": throughput_a,
        "engine_b_throughput": throughput_b,
        "ratio_a_to_b": round(ratio, 3),
    }


def check_engine_compatibility(
    engine_type: EngineTypeStr,
    model_format: Literal["transformers", "gguf", "onnx", "tensorrt"],
    quantization: QuantizationBackendStr = "none",
) -> tuple[bool, str]:
    """Check if an engine is compatible with a model format and quantization.

    Args:
        engine_type: Type of inference engine.
        model_format: Format of the model.
        quantization: Quantization backend.

    Returns:
        Tuple of (is_compatible, message).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> compatible, msg = check_engine_compatibility("vllm", "transformers")
        >>> compatible
        True
        >>> "compatible" in msg.lower()
        True

        >>> compatible, msg = check_engine_compatibility("llamacpp", "onnx")
        >>> compatible
        False
        >>> "not compatible" in msg.lower() or "requires" in msg.lower()
        True

        >>> check_engine_compatibility("invalid", "transformers")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: engine_type must be one of
    """
    if engine_type not in VALID_ENGINE_TYPES:
        msg = f"engine_type must be one of {VALID_ENGINE_TYPES}, got '{engine_type}'"
        raise ValueError(msg)

    if quantization not in VALID_QUANTIZATION_BACKENDS:
        msg = (
            f"quantization must be one of {VALID_QUANTIZATION_BACKENDS}, "
            f"got '{quantization}'"
        )
        raise ValueError(msg)

    valid_formats = {"transformers", "gguf", "onnx", "tensorrt"}
    if model_format not in valid_formats:
        msg = f"model_format must be one of {valid_formats}, got '{model_format}'"
        raise ValueError(msg)

    # Compatibility matrix
    format_compatibility = {
        "vllm": {"transformers"},
        "tgi": {"transformers"},
        "llamacpp": {"gguf"},
        "tensorrt": {"tensorrt", "onnx"},
        "ctranslate2": {"transformers", "onnx"},
        "onnxruntime": {"onnx"},
    }

    quant_compatibility = {
        "vllm": {"none", "bitsandbytes", "gptq", "awq"},
        "tgi": {"none", "bitsandbytes", "gptq", "awq"},
        "llamacpp": {"none", "ggml"},
        "tensorrt": {"none"},
        "ctranslate2": {"none"},
        "onnxruntime": {"none"},
    }

    # Check format compatibility
    if model_format not in format_compatibility[engine_type]:
        return (
            False,
            f"{engine_type} requires format in {format_compatibility[engine_type]}, "
            f"got '{model_format}'",
        )

    # Check quantization compatibility
    if quantization not in quant_compatibility[engine_type]:
        return (
            False,
            f"{engine_type} supports quantization in "
            f"{quant_compatibility[engine_type]}, "
            f"got '{quantization}'",
        )

    return (
        True,
        f"{engine_type} is compatible with {model_format} and {quantization}",
    )


def get_engine_features(engine_type: EngineTypeStr) -> frozenset[str]:
    """Get supported features for an engine type.

    Args:
        engine_type: Type of inference engine.

    Returns:
        Frozenset of supported feature names.

    Raises:
        ValueError: If engine_type is invalid.

    Examples:
        >>> features = get_engine_features("vllm")
        >>> "streaming" in features
        True
        >>> "batching" in features
        True

        >>> features = get_engine_features("llamacpp")
        >>> "streaming" in features
        True

        >>> get_engine_features("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: engine_type must be one of
    """
    if engine_type not in VALID_ENGINE_TYPES:
        msg = f"engine_type must be one of {VALID_ENGINE_TYPES}, got '{engine_type}'"
        raise ValueError(msg)

    feature_matrix = {
        "vllm": frozenset({"streaming", "batching", "speculative", "prefix_caching"}),
        "tgi": frozenset({"streaming", "batching", "prefix_caching"}),
        "llamacpp": frozenset({"streaming"}),
        "tensorrt": frozenset({"batching"}),
        "ctranslate2": frozenset({"batching", "streaming"}),
        "onnxruntime": frozenset({"batching"}),
    }

    return feature_matrix[engine_type]


def format_engine_stats(stats: EngineStats) -> str:
    """Format engine statistics as a human-readable string.

    Args:
        stats: Engine statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = EngineStats(
        ...     throughput_tokens_per_sec=1500.0,
        ...     latency_ms=50.0,
        ...     memory_usage_gb=24.0,
        ... )
        >>> formatted = format_engine_stats(stats)
        >>> "Throughput: 1500.00 tokens/sec" in formatted
        True
        >>> "Latency: 50.00 ms" in formatted
        True
        >>> "Memory: 24.00 GB" in formatted
        True

        >>> zero_stats = EngineStats(0.0, 0.0, 0.0)
        >>> "Throughput: 0.00 tokens/sec" in format_engine_stats(zero_stats)
        True
    """
    lines = [
        f"Throughput: {stats.throughput_tokens_per_sec:.2f} tokens/sec",
        f"Latency: {stats.latency_ms:.2f} ms",
        f"Memory: {stats.memory_usage_gb:.2f} GB",
    ]
    return "\n".join(lines)


def get_recommended_engine_config(
    model_size: ModelSizeStr,
    hardware: HardwareTypeStr = "gpu_datacenter",
) -> EngineConfig:
    """Get recommended engine configuration for model size and hardware.

    Args:
        model_size: Model size category (small/medium/large/xlarge).
        hardware: Target hardware. Defaults to "gpu_datacenter".

    Returns:
        Recommended EngineConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_engine_config("large", "gpu_datacenter")
        >>> config.engine_type
        <EngineType.VLLM: 'vllm'>
        >>> config.quantization
        <QuantizationBackend.AWQ: 'awq'>

        >>> config = get_recommended_engine_config("small", "cpu")
        >>> config.engine_type
        <EngineType.LLAMACPP: 'llamacpp'>

        >>> config = get_recommended_engine_config("medium", "gpu_consumer")
        >>> config.engine_type
        <EngineType.VLLM: 'vllm'>

        >>> get_recommended_engine_config("invalid", "cpu")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown model size
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"Unknown model size: '{model_size}'. Valid: {valid_sizes}"
        raise ValueError(msg)

    valid_hardware = {"cpu", "gpu_consumer", "gpu_datacenter"}
    if hardware not in valid_hardware:
        msg = f"Unknown hardware type: '{hardware}'. Valid: {valid_hardware}"
        raise ValueError(msg)

    # CPU recommendations: use llama.cpp
    if hardware == "cpu":
        llamacpp = LlamaCppConfig(
            n_ctx=2048 if model_size == "small" else 4096,
            n_batch=256 if model_size == "small" else 512,
            n_threads=8,
            use_mmap=True,
        )
        return EngineConfig(
            engine_type=EngineType.LLAMACPP,
            vllm_config=None,
            tgi_config=None,
            llamacpp_config=llamacpp,
            quantization=QuantizationBackend.GGML,
        )

    # GPU consumer recommendations: vLLM with quantization for larger models
    if hardware == "gpu_consumer":
        if model_size in ("small", "medium"):
            vllm = VLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                enforce_eager=False,
            )
            quant = (
                QuantizationBackend.NONE
                if model_size == "small"
                else QuantizationBackend.BITSANDBYTES
            )
        else:
            vllm = VLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95,
                max_model_len=2048,
                enforce_eager=False,
            )
            quant = QuantizationBackend.GPTQ

        return EngineConfig(
            engine_type=EngineType.VLLM,
            vllm_config=vllm,
            tgi_config=None,
            llamacpp_config=None,
            quantization=quant,
        )

    # GPU datacenter recommendations: vLLM with tensor parallelism for large models
    if model_size == "small":
        vllm = VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            enforce_eager=False,
        )
        quant = QuantizationBackend.NONE
    elif model_size == "medium":
        vllm = VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=False,
        )
        quant = QuantizationBackend.NONE
    elif model_size == "large":
        vllm = VLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=False,
        )
        quant = QuantizationBackend.AWQ
    else:  # xlarge
        vllm = VLLMConfig(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=False,
        )
        quant = QuantizationBackend.AWQ

    return EngineConfig(
        engine_type=EngineType.VLLM,
        vllm_config=vllm,
        tgi_config=None,
        llamacpp_config=None,
        quantization=quant,
    )
