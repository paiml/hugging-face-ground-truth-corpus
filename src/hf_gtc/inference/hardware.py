"""Hardware detection and optimization utilities for ML inference.

This module provides utilities for detecting available hardware devices,
creating device configurations, and optimizing inference for different
GPU architectures and compute capabilities.

Examples:
    >>> from hf_gtc.inference.hardware import detect_available_devices
    >>> devices = detect_available_devices()
    >>> "cpu" in devices
    True

    >>> from hf_gtc.inference.hardware import create_gpu_info
    >>> gpu = create_gpu_info(name="A100", memory_gb=80.0, compute_capability="sm_80")
    >>> gpu.tensor_cores
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class DeviceType(Enum):
    """Device types for ML inference.

    Attributes:
        CPU: Central Processing Unit.
        CUDA: NVIDIA CUDA GPU.
        MPS: Apple Metal Performance Shaders.
        TPU: Google Tensor Processing Unit.
        XPU: Intel XPU (oneAPI).

    Examples:
        >>> DeviceType.CPU.value
        'cpu'
        >>> DeviceType.CUDA.value
        'cuda'
        >>> DeviceType.MPS.value
        'mps'
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    TPU = "tpu"
    XPU = "xpu"


VALID_DEVICE_TYPES = frozenset(d.value for d in DeviceType)


class GPUArchitecture(Enum):
    """NVIDIA GPU architecture generations.

    Attributes:
        VOLTA: Volta architecture (V100, sm_70).
        TURING: Turing architecture (RTX 20xx, sm_75).
        AMPERE: Ampere architecture (A100, RTX 30xx, sm_80/sm_86).
        HOPPER: Hopper architecture (H100, sm_90).
        ADA_LOVELACE: Ada Lovelace architecture (RTX 40xx, sm_89).

    Examples:
        >>> GPUArchitecture.VOLTA.value
        'volta'
        >>> GPUArchitecture.AMPERE.value
        'ampere'
        >>> GPUArchitecture.HOPPER.value
        'hopper'
    """

    VOLTA = "volta"
    TURING = "turing"
    AMPERE = "ampere"
    HOPPER = "hopper"
    ADA_LOVELACE = "ada_lovelace"


VALID_GPU_ARCHITECTURES = frozenset(a.value for a in GPUArchitecture)


class ComputeCapability(Enum):
    """CUDA compute capability versions.

    Attributes:
        SM_70: Volta (V100).
        SM_75: Turing (RTX 20xx).
        SM_80: Ampere (A100).
        SM_86: Ampere (RTX 30xx).
        SM_89: Ada Lovelace (RTX 40xx).
        SM_90: Hopper (H100).

    Examples:
        >>> ComputeCapability.SM_70.value
        'sm_70'
        >>> ComputeCapability.SM_80.value
        'sm_80'
        >>> ComputeCapability.SM_90.value
        'sm_90'
    """

    SM_70 = "sm_70"
    SM_75 = "sm_75"
    SM_80 = "sm_80"
    SM_86 = "sm_86"
    SM_89 = "sm_89"
    SM_90 = "sm_90"


VALID_COMPUTE_CAPABILITIES = frozenset(c.value for c in ComputeCapability)


# Type aliases
DeviceTypeStr = Literal["cpu", "cuda", "mps", "tpu", "xpu"]
GPUArchitectureStr = Literal["volta", "turing", "ampere", "hopper", "ada_lovelace"]
ComputeCapabilityStr = Literal["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"]
HardwareTargetStr = Literal["cpu", "gpu_consumer", "gpu_datacenter", "tpu", "edge"]


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Information about a GPU device.

    Attributes:
        name: GPU model name (e.g., "A100", "RTX 4090").
        memory_gb: Total memory in gigabytes.
        compute_capability: CUDA compute capability.
        tensor_cores: Whether the GPU has tensor cores.

    Examples:
        >>> gpu = GPUInfo(
        ...     name="A100",
        ...     memory_gb=80.0,
        ...     compute_capability=ComputeCapability.SM_80,
        ...     tensor_cores=True,
        ... )
        >>> gpu.name
        'A100'
        >>> gpu.memory_gb
        80.0
        >>> gpu.tensor_cores
        True
    """

    name: str
    memory_gb: float
    compute_capability: ComputeCapability
    tensor_cores: bool


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Configuration for device placement.

    Attributes:
        device_type: Type of device to use.
        device_ids: List of device IDs for multi-device setups.
        memory_fraction: Fraction of device memory to use.
        allow_growth: Allow memory growth instead of pre-allocation.

    Examples:
        >>> config = DeviceConfig(
        ...     device_type=DeviceType.CUDA,
        ...     device_ids=(0, 1),
        ...     memory_fraction=0.9,
        ...     allow_growth=True,
        ... )
        >>> config.device_type
        <DeviceType.CUDA: 'cuda'>
        >>> config.memory_fraction
        0.9
    """

    device_type: DeviceType
    device_ids: tuple[int, ...]
    memory_fraction: float
    allow_growth: bool


@dataclass(frozen=True, slots=True)
class DeviceMap:
    """Device mapping configuration for model sharding.

    Attributes:
        layer_mapping: Mapping of layer names to device IDs.
        offload_folder: Folder for CPU offloading.
        offload_buffers: Whether to offload buffers to CPU.

    Examples:
        >>> device_map = DeviceMap(
        ...     layer_mapping={"model.embed_tokens": 0, "model.layers.0": 0},
        ...     offload_folder="/tmp/offload",
        ...     offload_buffers=True,
        ... )
        >>> device_map.offload_buffers
        True
    """

    layer_mapping: dict[str, int]
    offload_folder: str | None
    offload_buffers: bool


@dataclass(frozen=True, slots=True)
class HardwareStats:
    """Hardware statistics summary.

    Attributes:
        device_count: Number of available devices.
        total_memory_gb: Total device memory in GB.
        compute_tflops: Estimated compute throughput in TFLOPS.
        memory_bandwidth_gbps: Memory bandwidth in GB/s.

    Examples:
        >>> stats = HardwareStats(
        ...     device_count=8,
        ...     total_memory_gb=640.0,
        ...     compute_tflops=156.0,
        ...     memory_bandwidth_gbps=2039.0,
        ... )
        >>> stats.device_count
        8
        >>> stats.total_memory_gb
        640.0
    """

    device_count: int
    total_memory_gb: float
    compute_tflops: float
    memory_bandwidth_gbps: float


def validate_gpu_info(info: GPUInfo) -> None:
    """Validate GPU information.

    Args:
        info: GPU info to validate.

    Raises:
        ValueError: If GPU info is invalid.

    Examples:
        >>> gpu = GPUInfo(
        ...     name="A100",
        ...     memory_gb=80.0,
        ...     compute_capability=ComputeCapability.SM_80,
        ...     tensor_cores=True,
        ... )
        >>> validate_gpu_info(gpu)  # No error

        >>> bad = GPUInfo("", 80.0, ComputeCapability.SM_80, True)
        >>> validate_gpu_info(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not info.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if info.memory_gb <= 0:
        msg = f"memory_gb must be positive, got {info.memory_gb}"
        raise ValueError(msg)


def validate_device_config(config: DeviceConfig) -> None:
    """Validate device configuration.

    Args:
        config: Device config to validate.

    Raises:
        ValueError: If device config is invalid.

    Examples:
        >>> config = DeviceConfig(
        ...     device_type=DeviceType.CUDA,
        ...     device_ids=(0,),
        ...     memory_fraction=0.9,
        ...     allow_growth=True,
        ... )
        >>> validate_device_config(config)  # No error

        >>> bad = DeviceConfig(DeviceType.CUDA, (0,), 1.5, True)
        >>> validate_device_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory_fraction must be between 0 and 1
    """
    if not 0 < config.memory_fraction <= 1:
        msg = f"memory_fraction must be between 0 and 1, got {config.memory_fraction}"
        raise ValueError(msg)

    if not config.device_ids:
        msg = "device_ids cannot be empty"
        raise ValueError(msg)

    for device_id in config.device_ids:
        if device_id < 0:
            msg = f"device_ids must be non-negative, got {device_id}"
            raise ValueError(msg)


def validate_device_map(device_map: DeviceMap) -> None:
    """Validate device map configuration.

    Args:
        device_map: Device map to validate.

    Raises:
        ValueError: If device map is invalid.

    Examples:
        >>> dm = DeviceMap({"layer0": 0}, None, False)
        >>> validate_device_map(dm)  # No error

        >>> bad = DeviceMap({}, None, False)
        >>> validate_device_map(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_mapping cannot be empty
    """
    if not device_map.layer_mapping:
        msg = "layer_mapping cannot be empty"
        raise ValueError(msg)

    for layer_name, device_id in device_map.layer_mapping.items():
        if not layer_name:
            msg = "layer names cannot be empty"
            raise ValueError(msg)
        if device_id < 0:
            msg = f"device_id must be non-negative, got {device_id}"
            raise ValueError(msg)


def validate_hardware_stats(stats: HardwareStats) -> None:
    """Validate hardware statistics.

    Args:
        stats: Hardware stats to validate.

    Raises:
        ValueError: If hardware stats are invalid.

    Examples:
        >>> stats = HardwareStats(8, 640.0, 156.0, 2039.0)
        >>> validate_hardware_stats(stats)  # No error

        >>> bad = HardwareStats(-1, 640.0, 156.0, 2039.0)
        >>> validate_hardware_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: device_count cannot be negative
    """
    if stats.device_count < 0:
        msg = f"device_count cannot be negative, got {stats.device_count}"
        raise ValueError(msg)

    if stats.total_memory_gb < 0:
        msg = f"total_memory_gb cannot be negative, got {stats.total_memory_gb}"
        raise ValueError(msg)

    if stats.compute_tflops < 0:
        msg = f"compute_tflops cannot be negative, got {stats.compute_tflops}"
        raise ValueError(msg)

    if stats.memory_bandwidth_gbps < 0:
        msg = (
            f"memory_bandwidth_gbps cannot be negative, "
            f"got {stats.memory_bandwidth_gbps}"
        )
        raise ValueError(msg)


def create_gpu_info(
    name: str,
    memory_gb: float,
    compute_capability: ComputeCapabilityStr = "sm_80",
    tensor_cores: bool | None = None,
) -> GPUInfo:
    """Create GPU information.

    Args:
        name: GPU model name.
        memory_gb: Total memory in GB.
        compute_capability: CUDA compute capability. Defaults to "sm_80".
        tensor_cores: Whether GPU has tensor cores. Auto-detected if None.

    Returns:
        GPUInfo with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> gpu = create_gpu_info("A100", 80.0, "sm_80")
        >>> gpu.name
        'A100'
        >>> gpu.tensor_cores
        True

        >>> gpu = create_gpu_info("RTX 4090", 24.0, "sm_89")
        >>> gpu.compute_capability
        <ComputeCapability.SM_89: 'sm_89'>

        >>> create_gpu_info("", 80.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if compute_capability not in VALID_COMPUTE_CAPABILITIES:
        msg = (
            f"compute_capability must be one of {VALID_COMPUTE_CAPABILITIES}, "
            f"got '{compute_capability}'"
        )
        raise ValueError(msg)

    # Auto-detect tensor cores based on compute capability
    # All modern architectures (sm_70+) have tensor cores
    if tensor_cores is None:
        tensor_cores = True

    info = GPUInfo(
        name=name,
        memory_gb=memory_gb,
        compute_capability=ComputeCapability(compute_capability),
        tensor_cores=tensor_cores,
    )
    validate_gpu_info(info)
    return info


def create_device_config(
    device_type: DeviceTypeStr = "cuda",
    device_ids: tuple[int, ...] | None = None,
    memory_fraction: float = 0.9,
    allow_growth: bool = True,
) -> DeviceConfig:
    """Create device configuration.

    Args:
        device_type: Type of device. Defaults to "cuda".
        device_ids: Device IDs to use. Defaults to (0,).
        memory_fraction: Fraction of memory to use. Defaults to 0.9.
        allow_growth: Allow memory growth. Defaults to True.

    Returns:
        DeviceConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_device_config()
        >>> config.device_type
        <DeviceType.CUDA: 'cuda'>
        >>> config.memory_fraction
        0.9

        >>> config = create_device_config("cpu")
        >>> config.device_type
        <DeviceType.CPU: 'cpu'>

        >>> config = create_device_config(device_ids=(0, 1))
        >>> config.device_ids
        (0, 1)

        >>> create_device_config(memory_fraction=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory_fraction must be between 0 and 1
    """
    if device_type not in VALID_DEVICE_TYPES:
        msg = f"device_type must be one of {VALID_DEVICE_TYPES}, got '{device_type}'"
        raise ValueError(msg)

    if device_ids is None:
        device_ids = (0,)

    config = DeviceConfig(
        device_type=DeviceType(device_type),
        device_ids=device_ids,
        memory_fraction=memory_fraction,
        allow_growth=allow_growth,
    )
    validate_device_config(config)
    return config


def create_device_map(
    layer_mapping: dict[str, int],
    offload_folder: str | None = None,
    offload_buffers: bool = False,
) -> DeviceMap:
    """Create device map configuration.

    Args:
        layer_mapping: Mapping of layer names to device IDs.
        offload_folder: Folder for CPU offloading. Defaults to None.
        offload_buffers: Whether to offload buffers. Defaults to False.

    Returns:
        DeviceMap with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> dm = create_device_map({"model.embed_tokens": 0, "model.layers.0": 0})
        >>> dm.layer_mapping["model.embed_tokens"]
        0

        >>> dm = create_device_map({"layer0": 0}, "/tmp/offload", True)
        >>> dm.offload_folder
        '/tmp/offload'
        >>> dm.offload_buffers
        True

        >>> create_device_map({})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_mapping cannot be empty
    """
    device_map = DeviceMap(
        layer_mapping=layer_mapping,
        offload_folder=offload_folder,
        offload_buffers=offload_buffers,
    )
    validate_device_map(device_map)
    return device_map


def create_hardware_stats(
    device_count: int = 1,
    total_memory_gb: float = 0.0,
    compute_tflops: float = 0.0,
    memory_bandwidth_gbps: float = 0.0,
) -> HardwareStats:
    """Create hardware statistics.

    Args:
        device_count: Number of devices. Defaults to 1.
        total_memory_gb: Total memory in GB. Defaults to 0.0.
        compute_tflops: Compute throughput in TFLOPS. Defaults to 0.0.
        memory_bandwidth_gbps: Memory bandwidth in GB/s. Defaults to 0.0.

    Returns:
        HardwareStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_hardware_stats(8, 640.0, 156.0, 2039.0)
        >>> stats.device_count
        8
        >>> stats.total_memory_gb
        640.0

        >>> stats = create_hardware_stats()
        >>> stats.device_count
        1

        >>> create_hardware_stats(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: device_count cannot be negative
    """
    stats = HardwareStats(
        device_count=device_count,
        total_memory_gb=total_memory_gb,
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
    )
    validate_hardware_stats(stats)
    return stats


def list_device_types() -> list[str]:
    """List available device types.

    Returns:
        Sorted list of device type names.

    Examples:
        >>> types = list_device_types()
        >>> "cpu" in types
        True
        >>> "cuda" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_DEVICE_TYPES)


def list_gpu_architectures() -> list[str]:
    """List available GPU architectures.

    Returns:
        Sorted list of GPU architecture names.

    Examples:
        >>> archs = list_gpu_architectures()
        >>> "ampere" in archs
        True
        >>> "hopper" in archs
        True
        >>> archs == sorted(archs)
        True
    """
    return sorted(VALID_GPU_ARCHITECTURES)


def list_compute_capabilities() -> list[str]:
    """List available compute capabilities.

    Returns:
        Sorted list of compute capability names.

    Examples:
        >>> caps = list_compute_capabilities()
        >>> "sm_80" in caps
        True
        >>> "sm_90" in caps
        True
        >>> caps == sorted(caps)
        True
    """
    return sorted(VALID_COMPUTE_CAPABILITIES)


def get_device_type(name: str) -> DeviceType:
    """Get a device type by name.

    Args:
        name: Name of the device type.

    Returns:
        The corresponding DeviceType enum value.

    Raises:
        ValueError: If name is not a valid device type.

    Examples:
        >>> get_device_type("cpu")
        <DeviceType.CPU: 'cpu'>
        >>> get_device_type("cuda")
        <DeviceType.CUDA: 'cuda'>

        >>> get_device_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown device type
    """
    if name not in VALID_DEVICE_TYPES:
        msg = f"Unknown device type: '{name}'. Valid: {VALID_DEVICE_TYPES}"
        raise ValueError(msg)
    return DeviceType(name)


def get_gpu_architecture(name: str) -> GPUArchitecture:
    """Get a GPU architecture by name.

    Args:
        name: Name of the GPU architecture.

    Returns:
        The corresponding GPUArchitecture enum value.

    Raises:
        ValueError: If name is not a valid GPU architecture.

    Examples:
        >>> get_gpu_architecture("ampere")
        <GPUArchitecture.AMPERE: 'ampere'>
        >>> get_gpu_architecture("hopper")
        <GPUArchitecture.HOPPER: 'hopper'>

        >>> get_gpu_architecture("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown GPU architecture
    """
    if name not in VALID_GPU_ARCHITECTURES:
        msg = f"Unknown GPU architecture: '{name}'. Valid: {VALID_GPU_ARCHITECTURES}"
        raise ValueError(msg)
    return GPUArchitecture(name)


def get_compute_capability(name: str) -> ComputeCapability:
    """Get a compute capability by name.

    Args:
        name: Name of the compute capability.

    Returns:
        The corresponding ComputeCapability enum value.

    Raises:
        ValueError: If name is not a valid compute capability.

    Examples:
        >>> get_compute_capability("sm_80")
        <ComputeCapability.SM_80: 'sm_80'>
        >>> get_compute_capability("sm_90")
        <ComputeCapability.SM_90: 'sm_90'>

        >>> get_compute_capability("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown compute capability
    """
    if name not in VALID_COMPUTE_CAPABILITIES:
        msg = (
            f"Unknown compute capability: '{name}'. Valid: {VALID_COMPUTE_CAPABILITIES}"
        )
        raise ValueError(msg)
    return ComputeCapability(name)


def detect_available_devices() -> list[str]:
    """Detect available compute devices.

    Returns:
        List of available device types.

    Examples:
        >>> devices = detect_available_devices()
        >>> "cpu" in devices
        True
        >>> isinstance(devices, list)
        True
    """
    available: list[str] = ["cpu"]  # CPU always available

    try:
        import torch

        if torch.cuda.is_available():
            available.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available.append("mps")
    except ImportError:
        pass

    # TPU detection (JAX)
    try:
        import jax

        devices = jax.devices("tpu")
        if devices:
            available.append("tpu")
    except (ImportError, RuntimeError):
        pass

    # XPU detection (Intel)
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            available.append("xpu")
    except (ImportError, AttributeError):
        pass

    return available


def get_optimal_device_map(
    model_layers: int,
    layer_memory_gb: float,
    gpu_count: int = 1,
    gpu_memory_gb: float = 24.0,
) -> DeviceMap:
    """Get optimal device map for model sharding.

    Creates an even distribution of layers across available GPUs,
    with overflow to CPU offloading if necessary.

    Args:
        model_layers: Number of model layers.
        layer_memory_gb: Memory per layer in GB.
        gpu_count: Number of available GPUs. Defaults to 1.
        gpu_memory_gb: Memory per GPU in GB. Defaults to 24.0.

    Returns:
        DeviceMap with optimal layer distribution.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> dm = get_optimal_device_map(32, 0.5, gpu_count=2, gpu_memory_gb=24.0)
        >>> len(dm.layer_mapping) == 32
        True

        >>> dm = get_optimal_device_map(4, 5.0, gpu_count=2, gpu_memory_gb=10.0)
        >>> dm.layer_mapping
        {'layer_0': 0, 'layer_1': 1, 'layer_2': -1, 'layer_3': -1}

        >>> get_optimal_device_map(0, 1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_layers must be positive
    """
    if model_layers <= 0:
        msg = f"model_layers must be positive, got {model_layers}"
        raise ValueError(msg)

    if layer_memory_gb <= 0:
        msg = f"layer_memory_gb must be positive, got {layer_memory_gb}"
        raise ValueError(msg)

    if gpu_count <= 0:
        msg = f"gpu_count must be positive, got {gpu_count}"
        raise ValueError(msg)

    if gpu_memory_gb <= 0:
        msg = f"gpu_memory_gb must be positive, got {gpu_memory_gb}"
        raise ValueError(msg)

    # Calculate how many layers fit per GPU (with 90% memory usage)
    usable_memory = gpu_memory_gb * 0.9
    layers_per_gpu = int(usable_memory / layer_memory_gb)

    layer_mapping: dict[str, int] = {}
    offload_folder: str | None = None

    for i in range(model_layers):
        gpu_idx = i // max(layers_per_gpu, 1)
        if gpu_idx < gpu_count:
            layer_mapping[f"layer_{i}"] = gpu_idx
        else:
            # Offload to CPU
            layer_mapping[f"layer_{i}"] = -1  # -1 indicates CPU offload
            offload_folder = "/tmp/offload"

    return DeviceMap(
        layer_mapping=layer_mapping,
        offload_folder=offload_folder,
        offload_buffers=offload_folder is not None,
    )


def estimate_throughput(
    device_type: DeviceTypeStr,
    batch_size: int = 32,
    sequence_length: int = 512,
    hidden_size: int = 4096,
) -> float:
    """Estimate inference throughput in tokens per second.

    Args:
        device_type: Type of device.
        batch_size: Batch size. Defaults to 32.
        sequence_length: Sequence length. Defaults to 512.
        hidden_size: Hidden size. Defaults to 4096.

    Returns:
        Estimated tokens per second.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> throughput = estimate_throughput("cuda", 32, 512)
        >>> throughput > 0
        True

        >>> cpu_throughput = estimate_throughput("cpu", 32, 512)
        >>> cuda_throughput = estimate_throughput("cuda", 32, 512)
        >>> cuda_throughput > cpu_throughput
        True

        >>> estimate_throughput("invalid", 32, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: device_type must be one of
    """
    if device_type not in VALID_DEVICE_TYPES:
        msg = f"device_type must be one of {VALID_DEVICE_TYPES}, got '{device_type}'"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    # Base throughput estimates (tokens/second) for different devices
    # These are rough estimates based on typical hardware performance
    base_throughput: dict[str, float] = {
        "cpu": 100.0,
        "cuda": 10000.0,
        "mps": 3000.0,
        "tpu": 25000.0,
        "xpu": 8000.0,
    }

    base = base_throughput[device_type]

    # Scale by batch size (larger batches improve throughput)
    batch_factor = min(batch_size / 32.0, 4.0)

    # Penalize for longer sequences (memory bandwidth limited)
    sequence_factor = 512.0 / sequence_length

    # Penalize for larger hidden sizes
    hidden_factor = 4096.0 / hidden_size

    return base * batch_factor * sequence_factor * hidden_factor


def calculate_memory_bandwidth(
    device_type: DeviceTypeStr,
    compute_capability: ComputeCapabilityStr | None = None,
) -> float:
    """Calculate memory bandwidth in GB/s for a device.

    Args:
        device_type: Type of device.
        compute_capability: CUDA compute capability (for CUDA devices).

    Returns:
        Memory bandwidth in GB/s.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> bw = calculate_memory_bandwidth("cuda", "sm_80")
        >>> bw > 0
        True

        >>> bw_cpu = calculate_memory_bandwidth("cpu")
        >>> bw_gpu = calculate_memory_bandwidth("cuda", "sm_80")
        >>> bw_gpu > bw_cpu
        True

        >>> calculate_memory_bandwidth("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: device_type must be one of
    """
    if device_type not in VALID_DEVICE_TYPES:
        msg = f"device_type must be one of {VALID_DEVICE_TYPES}, got '{device_type}'"
        raise ValueError(msg)

    # Memory bandwidth estimates in GB/s
    if device_type == "cpu":
        return 50.0  # DDR4/DDR5 typical

    if device_type == "mps":
        return 400.0  # Apple unified memory

    if device_type == "tpu":
        return 900.0  # TPU v4

    if device_type == "xpu":
        return 600.0  # Intel Max Series

    # CUDA - varies by compute capability
    if compute_capability is None:
        return 900.0  # Default A100-like

    if compute_capability not in VALID_COMPUTE_CAPABILITIES:
        msg = (
            f"compute_capability must be one of {VALID_COMPUTE_CAPABILITIES}, "
            f"got '{compute_capability}'"
        )
        raise ValueError(msg)

    bandwidth_map: dict[str, float] = {
        "sm_70": 900.0,  # V100 HBM2
        "sm_75": 616.0,  # RTX 2080 Ti GDDR6
        "sm_80": 2039.0,  # A100 HBM2e
        "sm_86": 936.0,  # RTX 3090 GDDR6X
        "sm_89": 1008.0,  # RTX 4090 GDDR6X
        "sm_90": 3350.0,  # H100 HBM3
    }

    return bandwidth_map[compute_capability]


def check_hardware_compatibility(
    model_memory_gb: float,
    model_compute_tflops: float,
    device_memory_gb: float,
    device_compute_tflops: float,
) -> tuple[bool, str]:
    """Check if hardware is compatible with model requirements.

    Args:
        model_memory_gb: Model memory requirement in GB.
        model_compute_tflops: Model compute requirement in TFLOPS.
        device_memory_gb: Available device memory in GB.
        device_compute_tflops: Available device compute in TFLOPS.

    Returns:
        Tuple of (is_compatible, message).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> compat, msg = check_hardware_compatibility(10.0, 50.0, 24.0, 100.0)
        >>> compat
        True
        >>> "compatible" in msg.lower()
        True

        >>> compat, msg = check_hardware_compatibility(50.0, 50.0, 24.0, 100.0)
        >>> compat
        False
        >>> "memory" in msg.lower()
        True

        >>> check_hardware_compatibility(-1.0, 50.0, 24.0, 100.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_memory_gb must be positive
    """
    if model_memory_gb <= 0:
        msg = f"model_memory_gb must be positive, got {model_memory_gb}"
        raise ValueError(msg)

    if model_compute_tflops <= 0:
        msg = f"model_compute_tflops must be positive, got {model_compute_tflops}"
        raise ValueError(msg)

    if device_memory_gb <= 0:
        msg = f"device_memory_gb must be positive, got {device_memory_gb}"
        raise ValueError(msg)

    if device_compute_tflops <= 0:
        msg = f"device_compute_tflops must be positive, got {device_compute_tflops}"
        raise ValueError(msg)

    # Check memory compatibility (allow 90% utilization)
    if model_memory_gb > device_memory_gb * 0.9:
        shortage = model_memory_gb - device_memory_gb * 0.9
        return (
            False,
            f"Insufficient memory: need {model_memory_gb:.1f} GB, "
            f"have {device_memory_gb:.1f} GB ({shortage:.1f} GB shortage)",
        )

    # Check compute compatibility
    if model_compute_tflops > device_compute_tflops:
        return (
            False,
            f"Insufficient compute: need {model_compute_tflops:.1f} TFLOPS, "
            f"have {device_compute_tflops:.1f} TFLOPS",
        )

    return (
        True,
        f"Hardware compatible: {device_memory_gb:.1f} GB memory, "
        f"{device_compute_tflops:.1f} TFLOPS compute",
    )


def format_hardware_stats(stats: HardwareStats) -> str:
    """Format hardware statistics as a human-readable string.

    Args:
        stats: Hardware statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = HardwareStats(8, 640.0, 156.0, 2039.0)
        >>> formatted = format_hardware_stats(stats)
        >>> "Device Count: 8" in formatted
        True
        >>> "Total Memory: 640.00 GB" in formatted
        True

        >>> stats_zero = HardwareStats(0, 0.0, 0.0, 0.0)
        >>> "Device Count: 0" in format_hardware_stats(stats_zero)
        True
    """
    lines = [
        f"Device Count: {stats.device_count}",
        f"Total Memory: {stats.total_memory_gb:.2f} GB",
        f"Compute: {stats.compute_tflops:.2f} TFLOPS",
        f"Memory Bandwidth: {stats.memory_bandwidth_gbps:.2f} GB/s",
    ]
    return "\n".join(lines)


def get_recommended_hardware_config(
    target: HardwareTargetStr = "gpu_datacenter",
) -> DeviceConfig:
    """Get recommended hardware configuration for a target environment.

    Args:
        target: Target environment. Defaults to "gpu_datacenter".

    Returns:
        Recommended DeviceConfig.

    Raises:
        ValueError: If target is invalid.

    Examples:
        >>> config = get_recommended_hardware_config("gpu_datacenter")
        >>> config.device_type
        <DeviceType.CUDA: 'cuda'>
        >>> config.memory_fraction
        0.9

        >>> config = get_recommended_hardware_config("cpu")
        >>> config.device_type
        <DeviceType.CPU: 'cpu'>

        >>> config = get_recommended_hardware_config("edge")
        >>> config.memory_fraction
        0.7

        >>> get_recommended_hardware_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown target
    """
    valid_targets = {"cpu", "gpu_consumer", "gpu_datacenter", "tpu", "edge"}
    if target not in valid_targets:
        msg = f"Unknown target: '{target}'. Valid: {valid_targets}"
        raise ValueError(msg)

    if target == "cpu":
        return create_device_config(
            device_type="cpu",
            device_ids=(0,),
            memory_fraction=1.0,
            allow_growth=False,
        )

    if target == "gpu_consumer":
        return create_device_config(
            device_type="cuda",
            device_ids=(0,),
            memory_fraction=0.8,
            allow_growth=True,
        )

    if target == "gpu_datacenter":
        return create_device_config(
            device_type="cuda",
            device_ids=(0,),
            memory_fraction=0.9,
            allow_growth=False,
        )

    if target == "tpu":
        return create_device_config(
            device_type="tpu",
            device_ids=(0,),
            memory_fraction=0.95,
            allow_growth=False,
        )

    # edge
    return create_device_config(
        device_type="cpu",
        device_ids=(0,),
        memory_fraction=0.7,
        allow_growth=True,
    )


def get_architecture_for_compute_capability(
    compute_capability: ComputeCapabilityStr,
) -> GPUArchitecture:
    """Get GPU architecture for a compute capability.

    Args:
        compute_capability: CUDA compute capability.

    Returns:
        Corresponding GPUArchitecture.

    Raises:
        ValueError: If compute capability is invalid.

    Examples:
        >>> get_architecture_for_compute_capability("sm_80")
        <GPUArchitecture.AMPERE: 'ampere'>
        >>> get_architecture_for_compute_capability("sm_90")
        <GPUArchitecture.HOPPER: 'hopper'>
        >>> get_architecture_for_compute_capability("sm_89")
        <GPUArchitecture.ADA_LOVELACE: 'ada_lovelace'>

        >>> get_architecture_for_compute_capability("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown compute capability
    """
    if compute_capability not in VALID_COMPUTE_CAPABILITIES:
        msg = (
            f"Unknown compute capability: '{compute_capability}'. "
            f"Valid: {VALID_COMPUTE_CAPABILITIES}"
        )
        raise ValueError(msg)

    arch_map: dict[str, GPUArchitecture] = {
        "sm_70": GPUArchitecture.VOLTA,
        "sm_75": GPUArchitecture.TURING,
        "sm_80": GPUArchitecture.AMPERE,
        "sm_86": GPUArchitecture.AMPERE,
        "sm_89": GPUArchitecture.ADA_LOVELACE,
        "sm_90": GPUArchitecture.HOPPER,
    }

    return arch_map[compute_capability]
