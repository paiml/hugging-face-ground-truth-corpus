"""Deployment recipes for HuggingFace models.

This module provides utilities for model quantization,
format conversion, and serving.
"""

from __future__ import annotations

from hf_gtc.deployment.optimization import (
    OptimizationResult,
    QuantizationConfig,
    QuantizationType,
    calculate_compression_ratio,
    estimate_model_size,
    get_model_loading_kwargs,
    get_optimization_result,
    get_quantization_config,
    list_quantization_types,
)

__all__: list[str] = [
    "OptimizationResult",
    "QuantizationConfig",
    "QuantizationType",
    "calculate_compression_ratio",
    "estimate_model_size",
    "get_model_loading_kwargs",
    "get_optimization_result",
    "get_quantization_config",
    "list_quantization_types",
]
