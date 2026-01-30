"""Inference recipes for HuggingFace models.

This module provides utilities for creating pipelines and
managing device placement for inference.

Examples:
    >>> from hf_gtc.inference import get_device
    >>> device = get_device()
    >>> device in ("cuda", "mps", "cpu")
    True
"""

from __future__ import annotations

from hf_gtc.inference.device import (
    clear_gpu_memory,
    get_device,
    get_device_map,
    get_gpu_memory_info,
)
from hf_gtc.inference.pipelines import create_pipeline, list_supported_tasks

__all__ = [
    "clear_gpu_memory",
    "create_pipeline",
    "get_device",
    "get_device_map",
    "get_gpu_memory_info",
    "list_supported_tasks",
]
