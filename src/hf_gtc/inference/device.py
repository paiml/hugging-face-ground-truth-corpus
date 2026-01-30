"""Device detection and management utilities.

This module provides functions for detecting available compute devices
and managing GPU memory, with cross-reference to candle-core/device.rs.

Examples:
    >>> from hf_gtc.inference.device import get_device
    >>> device = get_device()
    >>> device in ("cuda", "mps", "cpu")
    True
"""

from __future__ import annotations

from typing import Literal

DeviceType = Literal["cuda", "mps", "cpu"]


def get_device() -> DeviceType:
    """Detect the best available compute device.

    Detection order (matches candle-core/device.rs):
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        The device type string: "cuda", "mps", or "cpu".

    Examples:
        >>> device = get_device()
        >>> device in ("cuda", "mps", "cpu")
        True
        >>> isinstance(device, str)
        True
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_map(
    model_size_gb: float,
    *,
    max_memory_fraction: float = 0.9,
) -> str | dict[str, str]:
    """Get optimal device map for model loading.

    For models that fit in GPU memory, returns "auto".
    For larger models, returns a memory-aware device map.

    Args:
        model_size_gb: Estimated model size in gigabytes.
        max_memory_fraction: Maximum fraction of GPU memory to use.
            Defaults to 0.9 (90%).

    Returns:
        Device map string or dictionary for model loading.

    Raises:
        ValueError: If model_size_gb is not positive.
        ValueError: If max_memory_fraction is not between 0 and 1.

    Examples:
        >>> # Small model fits on any device
        >>> get_device_map(0.5)
        'auto'

        >>> # Invalid inputs
        >>> get_device_map(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size_gb must be positive, got -1

        >>> get_device_map(1.0, max_memory_fraction=1.5)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: max_memory_fraction must be between 0 and 1...
    """
    if model_size_gb <= 0:
        msg = f"model_size_gb must be positive, got {model_size_gb}"
        raise ValueError(msg)

    if not 0 < max_memory_fraction <= 1:
        msg = f"max_memory_fraction must be between 0 and 1, got {max_memory_fraction}"
        raise ValueError(msg)

    device = get_device()

    if device == "cpu":
        return "cpu"

    # For GPU devices, check available memory
    memory_info = get_gpu_memory_info()
    available_gb = memory_info.get("free_gb", 0)

    # If model fits with margin, use auto
    if model_size_gb * 1.2 <= available_gb * max_memory_fraction:
        return "auto"

    # Otherwise, use balanced device map for multi-GPU or CPU offload
    return "balanced"


def get_gpu_memory_info() -> dict[str, float]:
    """Get GPU memory information.

    Returns:
        Dictionary with memory info:
        - total_gb: Total GPU memory in GB
        - used_gb: Used GPU memory in GB
        - free_gb: Free GPU memory in GB

        Returns zeros if no GPU is available.

    Examples:
        >>> info = get_gpu_memory_info()
        >>> "total_gb" in info and "used_gb" in info and "free_gb" in info
        True
        >>> all(isinstance(v, float) for v in info.values())
        True
        >>> all(v >= 0 for v in info.values())
        True
    """
    import torch

    if not torch.cuda.is_available():
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = total - reserved

    bytes_to_gb = 1 / (1024**3)

    return {
        "total_gb": total * bytes_to_gb,
        "used_gb": allocated * bytes_to_gb,
        "free_gb": free * bytes_to_gb,
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache.

    This is useful for freeing memory between model loads
    or when encountering OOM errors.

    Examples:
        >>> clear_gpu_memory()  # Should not raise
    """
    import gc

    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
