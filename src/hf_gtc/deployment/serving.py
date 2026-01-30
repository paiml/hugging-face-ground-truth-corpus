"""Model serving utilities for production deployments.

This module provides utilities for serving HuggingFace models in
production environments, including configuration, health checks,
and endpoint management.

Examples:
    >>> from hf_gtc.deployment.serving import ServerConfig
    >>> config = ServerConfig(host="0.0.0.0", port=8000)
    >>> config.port
    8000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class ServerStatus(Enum):
    """Status of a model server.

    Attributes:
        STARTING: Server is starting up.
        READY: Server is ready to accept requests.
        BUSY: Server is processing requests.
        DEGRADED: Server is running but with reduced capacity.
        STOPPING: Server is shutting down.
        STOPPED: Server has stopped.

    Examples:
        >>> ServerStatus.READY.value
        'ready'
        >>> ServerStatus.STOPPED.value
        'stopped'
    """

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"


VALID_SERVER_STATUSES = frozenset(s.value for s in ServerStatus)


class InferenceBackend(Enum):
    """Backend for model inference.

    Attributes:
        PYTORCH: PyTorch backend.
        ONNX: ONNX Runtime backend.
        TENSORRT: TensorRT backend.
        VLLM: vLLM backend.
        TGI: Text Generation Inference backend.

    Examples:
        >>> InferenceBackend.PYTORCH.value
        'pytorch'
        >>> InferenceBackend.VLLM.value
        'vllm'
    """

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    VLLM = "vllm"
    TGI = "tgi"


VALID_BACKENDS = frozenset(b.value for b in InferenceBackend)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Configuration for model server.

    Attributes:
        host: Server host address. Defaults to "0.0.0.0".
        port: Server port number. Defaults to 8000.
        model_path: Path to the model. Defaults to None.
        backend: Inference backend. Defaults to PYTORCH.
        max_batch_size: Maximum batch size. Defaults to 32.
        max_concurrent_requests: Maximum concurrent requests. Defaults to 100.
        timeout_seconds: Request timeout in seconds. Defaults to 30.

    Examples:
        >>> config = ServerConfig(host="localhost", port=9000)
        >>> config.host
        'localhost'
        >>> config.port
        9000
    """

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str | None = None
    backend: InferenceBackend = InferenceBackend.PYTORCH
    max_batch_size: int = 32
    max_concurrent_requests: int = 100
    timeout_seconds: int = 30


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Health status of a model server.

    Attributes:
        status: Current server status.
        model_loaded: Whether the model is loaded.
        memory_used_mb: Memory usage in MB.
        requests_pending: Number of pending requests.
        uptime_seconds: Server uptime in seconds.

    Examples:
        >>> health = HealthStatus(
        ...     status=ServerStatus.READY,
        ...     model_loaded=True,
        ...     memory_used_mb=1024,
        ...     requests_pending=5,
        ...     uptime_seconds=3600,
        ... )
        >>> health.model_loaded
        True
    """

    status: ServerStatus
    model_loaded: bool
    memory_used_mb: int
    requests_pending: int
    uptime_seconds: float


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Request for model inference.

    Attributes:
        request_id: Unique request identifier.
        inputs: Input data for inference.
        parameters: Optional inference parameters.

    Examples:
        >>> request = InferenceRequest(
        ...     request_id="req-001",
        ...     inputs="Hello, world!",
        ...     parameters={"max_length": 100},
        ... )
        >>> request.request_id
        'req-001'
    """

    request_id: str
    inputs: Any
    parameters: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """Response from model inference.

    Attributes:
        request_id: Original request identifier.
        outputs: Model outputs.
        latency_ms: Inference latency in milliseconds.
        success: Whether inference succeeded.
        error_message: Error message if failed.

    Examples:
        >>> response = InferenceResponse(
        ...     request_id="req-001",
        ...     outputs={"generated_text": "Response"},
        ...     latency_ms=50.0,
        ...     success=True,
        ...     error_message=None,
        ... )
        >>> response.success
        True
    """

    request_id: str
    outputs: Any
    latency_ms: float
    success: bool
    error_message: str | None


@dataclass
class ModelServer:
    """Model server instance.

    Attributes:
        config: Server configuration.
        status: Current server status.
        requests_processed: Total requests processed.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = ModelServer(config)
        >>> server.config.port
        8080
    """

    config: ServerConfig
    status: ServerStatus = ServerStatus.STOPPED
    requests_processed: int = 0
    _handlers: dict[str, Callable[..., Any]] = field(default_factory=dict)


def validate_server_config(config: ServerConfig) -> None:
    """Validate server configuration.

    Args:
        config: ServerConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If port is invalid (not 1-65535).
        ValueError: If max_batch_size is not positive.
        ValueError: If timeout_seconds is not positive.

    Examples:
        >>> config = ServerConfig(port=8000, max_batch_size=16)
        >>> validate_server_config(config)  # No error

        >>> validate_server_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ServerConfig(port=0)
        >>> validate_server_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: port must be between 1 and 65535
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 1 <= config.port <= 65535:
        msg = f"port must be between 1 and 65535, got {config.port}"
        raise ValueError(msg)

    if config.max_batch_size <= 0:
        msg = f"max_batch_size must be positive, got {config.max_batch_size}"
        raise ValueError(msg)

    if config.max_concurrent_requests <= 0:
        val = config.max_concurrent_requests
        msg = f"max_concurrent_requests must be positive, got {val}"
        raise ValueError(msg)

    if config.timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {config.timeout_seconds}"
        raise ValueError(msg)


def create_server(config: ServerConfig) -> ModelServer:
    """Create a model server with the given configuration.

    Args:
        config: Server configuration.

    Returns:
        Configured ModelServer instance.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> server.status
        <ServerStatus.STOPPED: 'stopped'>

        >>> create_server(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_server_config(config)
    return ModelServer(config=config)


def start_server(server: ModelServer) -> bool:
    """Start the model server.

    Args:
        server: Server to start.

    Returns:
        True if server started successfully.

    Raises:
        ValueError: If server is None.
        RuntimeError: If server is already running.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> start_server(server)
        True
        >>> server.status
        <ServerStatus.READY: 'ready'>

        >>> start_server(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    if server.status in (ServerStatus.READY, ServerStatus.BUSY, ServerStatus.STARTING):
        msg = "server is already running"
        raise RuntimeError(msg)

    server.status = ServerStatus.STARTING
    # In a real implementation, this would start the actual server
    server.status = ServerStatus.READY
    return True


def stop_server(server: ModelServer) -> bool:
    """Stop the model server.

    Args:
        server: Server to stop.

    Returns:
        True if server stopped successfully.

    Raises:
        ValueError: If server is None.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> start_server(server)
        True
        >>> stop_server(server)
        True
        >>> server.status
        <ServerStatus.STOPPED: 'stopped'>

        >>> stop_server(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    if server.status == ServerStatus.STOPPED:
        return True

    server.status = ServerStatus.STOPPING
    # In a real implementation, this would stop the actual server
    server.status = ServerStatus.STOPPED
    return True


def get_health_status(server: ModelServer) -> HealthStatus:
    """Get health status of a server.

    Args:
        server: Server to check.

    Returns:
        HealthStatus with current server state.

    Raises:
        ValueError: If server is None.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> start_server(server)
        True
        >>> health = get_health_status(server)
        >>> health.status
        <ServerStatus.READY: 'ready'>

        >>> get_health_status(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    return HealthStatus(
        status=server.status,
        model_loaded=server.status == ServerStatus.READY,
        memory_used_mb=0,  # Would be actual memory in real implementation
        requests_pending=0,
        uptime_seconds=0.0,
    )


def process_request(
    server: ModelServer,
    request: InferenceRequest,
    inference_fn: Callable[[Any], Any],
) -> InferenceResponse:
    """Process an inference request.

    Args:
        server: Server to process the request.
        request: Inference request.
        inference_fn: Function to perform inference.

    Returns:
        InferenceResponse with outputs or error.

    Raises:
        ValueError: If server is None.
        ValueError: If request is None.
        ValueError: If inference_fn is None.
        RuntimeError: If server is not ready.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> start_server(server)
        True
        >>> request = InferenceRequest("req-001", "test input")
        >>> response = process_request(server, request, lambda x: f"output: {x}")
        >>> response.success
        True

        >>> process_request(None, request, lambda x: x)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    if request is None:
        msg = "request cannot be None"
        raise ValueError(msg)

    if inference_fn is None:
        msg = "inference_fn cannot be None"
        raise ValueError(msg)

    if server.status != ServerStatus.READY:
        msg = f"server is not ready, current status: {server.status.value}"
        raise RuntimeError(msg)

    import time

    start = time.perf_counter()

    try:
        outputs = inference_fn(request.inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        server.requests_processed += 1

        return InferenceResponse(
            request_id=request.request_id,
            outputs=outputs,
            latency_ms=elapsed_ms,
            success=True,
            error_message=None,
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return InferenceResponse(
            request_id=request.request_id,
            outputs=None,
            latency_ms=elapsed_ms,
            success=False,
            error_message=str(e),
        )


def process_batch(
    server: ModelServer,
    requests: Sequence[InferenceRequest],
    inference_fn: Callable[[Sequence[Any]], Sequence[Any]],
) -> list[InferenceResponse]:
    """Process a batch of inference requests.

    Args:
        server: Server to process the requests.
        requests: Sequence of inference requests.
        inference_fn: Function to perform batch inference.

    Returns:
        List of InferenceResponse for each request.

    Raises:
        ValueError: If server is None.
        ValueError: If requests is None.
        ValueError: If inference_fn is None.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> start_server(server)
        True
        >>> reqs = [InferenceRequest("req-001", "a"), InferenceRequest("req-002", "b")]
        >>> responses = process_batch(server, reqs, lambda x: [f"out-{i}" for i in x])
        >>> len(responses)
        2

        >>> process_batch(None, [], lambda x: x)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    if requests is None:
        msg = "requests cannot be None"
        raise ValueError(msg)

    if inference_fn is None:
        msg = "inference_fn cannot be None"
        raise ValueError(msg)

    if not requests:
        return []

    import time

    start = time.perf_counter()

    try:
        inputs = [r.inputs for r in requests]
        outputs = inference_fn(inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_request_ms = elapsed_ms / len(requests)

        server.requests_processed += len(requests)

        return [
            InferenceResponse(
                request_id=req.request_id,
                outputs=out,
                latency_ms=per_request_ms,
                success=True,
                error_message=None,
            )
            for req, out in zip(requests, outputs, strict=True)
        ]

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return [
            InferenceResponse(
                request_id=req.request_id,
                outputs=None,
                latency_ms=elapsed_ms,
                success=False,
                error_message=str(e),
            )
            for req in requests
        ]


def list_server_statuses() -> list[str]:
    """List all available server statuses.

    Returns:
        Sorted list of server status names.

    Examples:
        >>> statuses = list_server_statuses()
        >>> "ready" in statuses
        True
        >>> "stopped" in statuses
        True
        >>> statuses == sorted(statuses)
        True
    """
    return sorted(VALID_SERVER_STATUSES)


def validate_server_status(status: str) -> bool:
    """Validate if a string is a valid server status.

    Args:
        status: The status string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_server_status("ready")
        True
        >>> validate_server_status("stopped")
        True
        >>> validate_server_status("invalid")
        False
        >>> validate_server_status("")
        False
    """
    return status in VALID_SERVER_STATUSES


def get_server_status(name: str) -> ServerStatus:
    """Get ServerStatus enum from string name.

    Args:
        name: Name of the server status.

    Returns:
        Corresponding ServerStatus enum value.

    Raises:
        ValueError: If name is not a valid server status.

    Examples:
        >>> get_server_status("ready")
        <ServerStatus.READY: 'ready'>

        >>> get_server_status("stopped")
        <ServerStatus.STOPPED: 'stopped'>

        >>> get_server_status("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid server status: invalid
    """
    if not validate_server_status(name):
        msg = f"invalid server status: {name}"
        raise ValueError(msg)

    return ServerStatus(name)


def list_inference_backends() -> list[str]:
    """List all available inference backends.

    Returns:
        Sorted list of backend names.

    Examples:
        >>> backends = list_inference_backends()
        >>> "pytorch" in backends
        True
        >>> "vllm" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_BACKENDS)


def validate_inference_backend(backend: str) -> bool:
    """Validate if a string is a valid inference backend.

    Args:
        backend: The backend string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_inference_backend("pytorch")
        True
        >>> validate_inference_backend("vllm")
        True
        >>> validate_inference_backend("invalid")
        False
        >>> validate_inference_backend("")
        False
    """
    return backend in VALID_BACKENDS


def get_inference_backend(name: str) -> InferenceBackend:
    """Get InferenceBackend enum from string name.

    Args:
        name: Name of the inference backend.

    Returns:
        Corresponding InferenceBackend enum value.

    Raises:
        ValueError: If name is not a valid backend.

    Examples:
        >>> get_inference_backend("pytorch")
        <InferenceBackend.PYTORCH: 'pytorch'>

        >>> get_inference_backend("vllm")
        <InferenceBackend.VLLM: 'vllm'>

        >>> get_inference_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid inference backend: invalid
    """
    if not validate_inference_backend(name):
        msg = f"invalid inference backend: {name}"
        raise ValueError(msg)

    return InferenceBackend(name)


def format_server_info(server: ModelServer) -> str:
    """Format server information as a human-readable string.

    Args:
        server: Server to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If server is None.

    Examples:
        >>> config = ServerConfig(host="localhost", port=8080)
        >>> server = create_server(config)
        >>> "localhost:8080" in format_server_info(server)
        True

        >>> format_server_info(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    lines = [
        f"Server: {server.config.host}:{server.config.port}",
        f"Status: {server.status.value}",
        f"Backend: {server.config.backend.value}",
        f"Max Batch Size: {server.config.max_batch_size}",
        f"Max Concurrent: {server.config.max_concurrent_requests}",
        f"Timeout: {server.config.timeout_seconds}s",
        f"Requests Processed: {server.requests_processed}",
    ]

    if server.config.model_path:
        lines.insert(1, f"Model: {server.config.model_path}")

    return "\n".join(lines)


def compute_server_metrics(server: ModelServer) -> dict[str, Any]:
    """Compute metrics for a server.

    Args:
        server: Server to compute metrics for.

    Returns:
        Dictionary with server metrics.

    Raises:
        ValueError: If server is None.

    Examples:
        >>> config = ServerConfig(port=8080)
        >>> server = create_server(config)
        >>> metrics = compute_server_metrics(server)
        >>> metrics["requests_processed"]
        0

        >>> compute_server_metrics(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: server cannot be None
    """
    if server is None:
        msg = "server cannot be None"
        raise ValueError(msg)

    return {
        "status": server.status.value,
        "is_running": server.status in (ServerStatus.READY, ServerStatus.BUSY),
        "requests_processed": server.requests_processed,
        "host": server.config.host,
        "port": server.config.port,
        "backend": server.config.backend.value,
    }
