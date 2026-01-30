"""Deployment recipes for HuggingFace models.

This module provides utilities for model quantization,
format conversion, and serving.

Examples:
    >>> from hf_gtc.deployment import ServerConfig, QuantizationType
    >>> config = ServerConfig(port=8080)
    >>> config.port
    8080
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
from hf_gtc.deployment.serving import (
    HealthStatus,
    InferenceBackend,
    InferenceRequest,
    InferenceResponse,
    ModelServer,
    ServerConfig,
    ServerStatus,
    compute_server_metrics,
    create_server,
    format_server_info,
    get_health_status,
    get_inference_backend,
    get_server_status,
    list_inference_backends,
    list_server_statuses,
    process_batch,
    process_request,
    start_server,
    stop_server,
    validate_inference_backend,
    validate_server_config,
    validate_server_status,
)

__all__: list[str] = [
    "HealthStatus",
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResponse",
    "ModelServer",
    "OptimizationResult",
    "QuantizationConfig",
    "QuantizationType",
    "ServerConfig",
    "ServerStatus",
    "calculate_compression_ratio",
    "compute_server_metrics",
    "create_server",
    "estimate_model_size",
    "format_server_info",
    "get_health_status",
    "get_inference_backend",
    "get_model_loading_kwargs",
    "get_optimization_result",
    "get_quantization_config",
    "get_server_status",
    "list_inference_backends",
    "list_quantization_types",
    "list_server_statuses",
    "process_batch",
    "process_request",
    "start_server",
    "stop_server",
    "validate_inference_backend",
    "validate_server_config",
    "validate_server_status",
]
