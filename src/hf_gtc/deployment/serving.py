"""Model serving utilities for production deployments.

This module provides utilities for serving HuggingFace models in
production environments, including configuration, health checks,
endpoint management, scaling, and load balancing.

Examples:
    >>> from hf_gtc.deployment.serving import ServerConfig
    >>> config = ServerConfig(host="0.0.0.0", port=8000)
    >>> config.port
    8000

    >>> from hf_gtc.deployment.serving import create_endpoint_config
    >>> endpoint = create_endpoint_config(host="0.0.0.0", port=8080)
    >>> endpoint.port
    8080
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


class ServingBackend(Enum):
    """Backend systems for model serving.

    Attributes:
        VLLM: vLLM for high-throughput LLM inference.
        TGI: Text Generation Inference from HuggingFace.
        TRITON: NVIDIA Triton Inference Server.
        RAY_SERVE: Ray Serve for distributed serving.
        FASTAPI: FastAPI for simple REST deployments.

    Examples:
        >>> ServingBackend.VLLM.value
        'vllm'
        >>> ServingBackend.TGI.value
        'tgi'
        >>> ServingBackend.TRITON.value
        'triton'
    """

    VLLM = "vllm"
    TGI = "tgi"
    TRITON = "triton"
    RAY_SERVE = "ray_serve"
    FASTAPI = "fastapi"


VALID_SERVING_BACKENDS = frozenset(b.value for b in ServingBackend)


class LoadBalancing(Enum):
    """Load balancing strategies for distributed serving.

    Attributes:
        ROUND_ROBIN: Distribute requests in round-robin fashion.
        LEAST_CONNECTIONS: Route to replica with fewest connections.
        RANDOM: Random replica selection.
        WEIGHTED: Weighted distribution based on replica capacity.

    Examples:
        >>> LoadBalancing.ROUND_ROBIN.value
        'round_robin'
        >>> LoadBalancing.LEAST_CONNECTIONS.value
        'least_connections'
        >>> LoadBalancing.RANDOM.value
        'random'
    """

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"


VALID_LOAD_BALANCING = frozenset(lb.value for lb in LoadBalancing)


class HealthCheckType(Enum):
    """Types of health checks for serving endpoints.

    Attributes:
        LIVENESS: Check if the service is alive.
        READINESS: Check if the service is ready to accept requests.
        STARTUP: Check during service startup phase.

    Examples:
        >>> HealthCheckType.LIVENESS.value
        'liveness'
        >>> HealthCheckType.READINESS.value
        'readiness'
        >>> HealthCheckType.STARTUP.value
        'startup'
    """

    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


VALID_HEALTH_CHECK_TYPES = frozenset(h.value for h in HealthCheckType)


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

    host: str = "127.0.0.1"  # Secure default: localhost only
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


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Configuration for a serving endpoint.

    Attributes:
        host: Host address to bind to.
        port: Port number for the endpoint.
        workers: Number of worker processes.
        timeout_seconds: Request timeout in seconds.
        max_batch_size: Maximum batch size for batched inference.

    Examples:
        >>> config = EndpointConfig(
        ...     host="0.0.0.0",
        ...     port=8080,
        ...     workers=4,
        ...     timeout_seconds=30,
        ...     max_batch_size=32,
        ... )
        >>> config.port
        8080
        >>> config.workers
        4
    """

    host: str
    port: int
    workers: int
    timeout_seconds: int
    max_batch_size: int


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    """Configuration for auto-scaling serving replicas.

    Attributes:
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        target_utilization: Target GPU/CPU utilization for scaling.
        scale_down_delay: Seconds to wait before scaling down.

    Examples:
        >>> config = ScalingConfig(
        ...     min_replicas=1,
        ...     max_replicas=10,
        ...     target_utilization=0.7,
        ...     scale_down_delay=300,
        ... )
        >>> config.min_replicas
        1
        >>> config.target_utilization
        0.7
    """

    min_replicas: int
    max_replicas: int
    target_utilization: float
    scale_down_delay: int


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """Complete serving configuration.

    Attributes:
        backend: Serving backend to use.
        endpoint_config: Endpoint configuration.
        scaling_config: Auto-scaling configuration.
        load_balancing: Load balancing strategy.

    Examples:
        >>> endpoint = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        >>> scaling = ScalingConfig(1, 10, 0.7, 300)
        >>> config = ServingConfig(
        ...     backend=ServingBackend.VLLM,
        ...     endpoint_config=endpoint,
        ...     scaling_config=scaling,
        ...     load_balancing=LoadBalancing.ROUND_ROBIN,
        ... )
        >>> config.backend
        <ServingBackend.VLLM: 'vllm'>
    """

    backend: ServingBackend
    endpoint_config: EndpointConfig
    scaling_config: ScalingConfig
    load_balancing: LoadBalancing


@dataclass(frozen=True, slots=True)
class ServingStats:
    """Statistics for serving performance monitoring.

    Attributes:
        requests_per_second: Current throughput in requests/second.
        avg_latency_ms: Average latency in milliseconds.
        p99_latency_ms: 99th percentile latency in milliseconds.
        gpu_utilization: GPU utilization (0.0 to 1.0).

    Examples:
        >>> stats = ServingStats(
        ...     requests_per_second=150.0,
        ...     avg_latency_ms=25.5,
        ...     p99_latency_ms=75.0,
        ...     gpu_utilization=0.85,
        ... )
        >>> stats.requests_per_second
        150.0
        >>> stats.gpu_utilization
        0.85
    """

    requests_per_second: float
    avg_latency_ms: float
    p99_latency_ms: float
    gpu_utilization: float


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


# New enum list/get/validate functions


def list_serving_backends() -> list[str]:
    """List all available serving backends.

    Returns:
        Sorted list of serving backend names.

    Examples:
        >>> backends = list_serving_backends()
        >>> "vllm" in backends
        True
        >>> "tgi" in backends
        True
        >>> "triton" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_SERVING_BACKENDS)


def validate_serving_backend(backend: str) -> bool:
    """Validate if a string is a valid serving backend.

    Args:
        backend: The backend string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_serving_backend("vllm")
        True
        >>> validate_serving_backend("tgi")
        True
        >>> validate_serving_backend("invalid")
        False
        >>> validate_serving_backend("")
        False
    """
    return backend in VALID_SERVING_BACKENDS


def get_serving_backend(name: str) -> ServingBackend:
    """Get ServingBackend enum from string name.

    Args:
        name: Name of the serving backend.

    Returns:
        Corresponding ServingBackend enum value.

    Raises:
        ValueError: If name is not a valid backend.

    Examples:
        >>> get_serving_backend("vllm")
        <ServingBackend.VLLM: 'vllm'>

        >>> get_serving_backend("tgi")
        <ServingBackend.TGI: 'tgi'>

        >>> get_serving_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid serving backend: invalid
    """
    if not validate_serving_backend(name):
        msg = f"invalid serving backend: {name}"
        raise ValueError(msg)

    return ServingBackend(name)


def list_load_balancing_strategies() -> list[str]:
    """List all available load balancing strategies.

    Returns:
        Sorted list of load balancing strategy names.

    Examples:
        >>> strategies = list_load_balancing_strategies()
        >>> "round_robin" in strategies
        True
        >>> "least_connections" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_LOAD_BALANCING)


def validate_load_balancing(strategy: str) -> bool:
    """Validate if a string is a valid load balancing strategy.

    Args:
        strategy: The strategy string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_load_balancing("round_robin")
        True
        >>> validate_load_balancing("least_connections")
        True
        >>> validate_load_balancing("invalid")
        False
        >>> validate_load_balancing("")
        False
    """
    return strategy in VALID_LOAD_BALANCING


def get_load_balancing(name: str) -> LoadBalancing:
    """Get LoadBalancing enum from string name.

    Args:
        name: Name of the load balancing strategy.

    Returns:
        Corresponding LoadBalancing enum value.

    Raises:
        ValueError: If name is not a valid strategy.

    Examples:
        >>> get_load_balancing("round_robin")
        <LoadBalancing.ROUND_ROBIN: 'round_robin'>

        >>> get_load_balancing("least_connections")
        <LoadBalancing.LEAST_CONNECTIONS: 'least_connections'>

        >>> get_load_balancing("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid load balancing strategy: invalid
    """
    if not validate_load_balancing(name):
        msg = f"invalid load balancing strategy: {name}"
        raise ValueError(msg)

    return LoadBalancing(name)


def list_health_check_types() -> list[str]:
    """List all available health check types.

    Returns:
        Sorted list of health check type names.

    Examples:
        >>> types = list_health_check_types()
        >>> "liveness" in types
        True
        >>> "readiness" in types
        True
        >>> "startup" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_HEALTH_CHECK_TYPES)


def validate_health_check_type(check_type: str) -> bool:
    """Validate if a string is a valid health check type.

    Args:
        check_type: The health check type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_health_check_type("liveness")
        True
        >>> validate_health_check_type("readiness")
        True
        >>> validate_health_check_type("invalid")
        False
        >>> validate_health_check_type("")
        False
    """
    return check_type in VALID_HEALTH_CHECK_TYPES


def get_health_check_type(name: str) -> HealthCheckType:
    """Get HealthCheckType enum from string name.

    Args:
        name: Name of the health check type.

    Returns:
        Corresponding HealthCheckType enum value.

    Raises:
        ValueError: If name is not a valid health check type.

    Examples:
        >>> get_health_check_type("liveness")
        <HealthCheckType.LIVENESS: 'liveness'>

        >>> get_health_check_type("readiness")
        <HealthCheckType.READINESS: 'readiness'>

        >>> get_health_check_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid health check type: invalid
    """
    if not validate_health_check_type(name):
        msg = f"invalid health check type: {name}"
        raise ValueError(msg)

    return HealthCheckType(name)


# Factory functions


def create_endpoint_config(
    host: str = "127.0.0.1",
    port: int = 8080,
    workers: int = 4,
    timeout_seconds: int = 30,
    max_batch_size: int = 32,
) -> EndpointConfig:
    """Create an endpoint configuration.

    Args:
        host: Host address to bind to. Defaults to "127.0.0.1".
        port: Port number for the endpoint. Defaults to 8080.
        workers: Number of worker processes. Defaults to 4.
        timeout_seconds: Request timeout in seconds. Defaults to 30.
        max_batch_size: Maximum batch size. Defaults to 32.

    Returns:
        Validated EndpointConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_endpoint_config()
        >>> config.host
        '127.0.0.1'
        >>> config.port
        8080

        >>> config = create_endpoint_config(host="0.0.0.0", port=9000)
        >>> config.host
        '0.0.0.0'
        >>> config.port
        9000

        >>> create_endpoint_config(port=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: port must be between 1 and 65535, got 0
    """
    config = EndpointConfig(
        host=host,
        port=port,
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_batch_size=max_batch_size,
    )
    validate_endpoint_config(config)
    return config


def create_scaling_config(
    min_replicas: int = 1,
    max_replicas: int = 10,
    target_utilization: float = 0.7,
    scale_down_delay: int = 300,
) -> ScalingConfig:
    """Create a scaling configuration.

    Args:
        min_replicas: Minimum number of replicas. Defaults to 1.
        max_replicas: Maximum number of replicas. Defaults to 10.
        target_utilization: Target utilization for scaling. Defaults to 0.7.
        scale_down_delay: Seconds before scaling down. Defaults to 300.

    Returns:
        Validated ScalingConfig instance.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_scaling_config()
        >>> config.min_replicas
        1
        >>> config.max_replicas
        10

        >>> config = create_scaling_config(min_replicas=2, max_replicas=20)
        >>> config.min_replicas
        2

        >>> create_scaling_config(min_replicas=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: min_replicas must be positive, got 0
    """
    config = ScalingConfig(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        target_utilization=target_utilization,
        scale_down_delay=scale_down_delay,
    )
    validate_scaling_config(config)
    return config


def create_serving_config(
    backend: str | ServingBackend = ServingBackend.VLLM,
    endpoint_config: EndpointConfig | None = None,
    scaling_config: ScalingConfig | None = None,
    load_balancing: str | LoadBalancing = LoadBalancing.ROUND_ROBIN,
) -> ServingConfig:
    """Create a complete serving configuration.

    Args:
        backend: Serving backend to use. Defaults to VLLM.
        endpoint_config: Endpoint configuration. Creates default if None.
        scaling_config: Scaling configuration. Creates default if None.
        load_balancing: Load balancing strategy. Defaults to ROUND_ROBIN.

    Returns:
        Validated ServingConfig instance.

    Examples:
        >>> config = create_serving_config()
        >>> config.backend
        <ServingBackend.VLLM: 'vllm'>
        >>> config.load_balancing
        <LoadBalancing.ROUND_ROBIN: 'round_robin'>

        >>> config = create_serving_config(backend="tgi")
        >>> config.backend
        <ServingBackend.TGI: 'tgi'>

        >>> config = create_serving_config(load_balancing="least_connections")
        >>> config.load_balancing
        <LoadBalancing.LEAST_CONNECTIONS: 'least_connections'>
    """
    if isinstance(backend, str):
        backend = get_serving_backend(backend)
    if isinstance(load_balancing, str):
        load_balancing = get_load_balancing(load_balancing)

    if endpoint_config is None:
        endpoint_config = create_endpoint_config()
    if scaling_config is None:
        scaling_config = create_scaling_config()

    config = ServingConfig(
        backend=backend,
        endpoint_config=endpoint_config,
        scaling_config=scaling_config,
        load_balancing=load_balancing,
    )
    validate_serving_config(config)
    return config


# Validation functions


def validate_endpoint_config(config: EndpointConfig) -> None:
    """Validate endpoint configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        >>> validate_endpoint_config(config)

        >>> bad_config = EndpointConfig("0.0.0.0", 0, 4, 30, 32)
        >>> validate_endpoint_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: port must be between 1 and 65535, got 0

        >>> bad_config = EndpointConfig("0.0.0.0", 8080, 0, 30, 32)
        >>> validate_endpoint_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: workers must be positive, got 0
    """
    if not 1 <= config.port <= 65535:
        msg = f"port must be between 1 and 65535, got {config.port}"
        raise ValueError(msg)
    if config.workers <= 0:
        msg = f"workers must be positive, got {config.workers}"
        raise ValueError(msg)
    if config.timeout_seconds <= 0:
        msg = f"timeout_seconds must be positive, got {config.timeout_seconds}"
        raise ValueError(msg)
    if config.max_batch_size <= 0:
        msg = f"max_batch_size must be positive, got {config.max_batch_size}"
        raise ValueError(msg)


def validate_scaling_config(config: ScalingConfig) -> None:
    """Validate scaling configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ScalingConfig(1, 10, 0.7, 300)
        >>> validate_scaling_config(config)

        >>> bad_config = ScalingConfig(0, 10, 0.7, 300)
        >>> validate_scaling_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: min_replicas must be positive, got 0

        >>> bad_config = ScalingConfig(10, 5, 0.7, 300)
        >>> validate_scaling_config(bad_config)
        Traceback (most recent call last):
            ...
        ValueError: max_replicas must be >= min_replicas
    """
    if config.min_replicas <= 0:
        msg = f"min_replicas must be positive, got {config.min_replicas}"
        raise ValueError(msg)
    if config.max_replicas < config.min_replicas:
        msg = "max_replicas must be >= min_replicas"
        raise ValueError(msg)
    if not 0 < config.target_utilization <= 1:
        msg = (
            f"target_utilization must be between 0 and 1, "
            f"got {config.target_utilization}"
        )
        raise ValueError(msg)
    if config.scale_down_delay < 0:
        msg = f"scale_down_delay must be non-negative, got {config.scale_down_delay}"
        raise ValueError(msg)


def validate_serving_config(config: ServingConfig) -> None:
    """Validate serving configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> endpoint = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        >>> scaling = ScalingConfig(1, 10, 0.7, 300)
        >>> config = ServingConfig(
        ...     ServingBackend.VLLM, endpoint, scaling, LoadBalancing.ROUND_ROBIN
        ... )
        >>> validate_serving_config(config)
    """
    validate_endpoint_config(config.endpoint_config)
    validate_scaling_config(config.scaling_config)


# Core functions


def calculate_throughput_capacity(
    workers: int,
    batch_size: int,
    avg_latency_ms: float,
) -> float:
    """Calculate theoretical throughput capacity.

    Args:
        workers: Number of worker processes.
        batch_size: Maximum batch size.
        avg_latency_ms: Average inference latency in milliseconds.

    Returns:
        Theoretical requests per second.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_throughput_capacity(4, 32, 50.0)
        2560.0

        >>> calculate_throughput_capacity(1, 1, 100.0)
        10.0

        >>> calculate_throughput_capacity(0, 32, 50.0)
        Traceback (most recent call last):
            ...
        ValueError: workers must be positive, got 0
    """
    if workers <= 0:
        msg = f"workers must be positive, got {workers}"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if avg_latency_ms <= 0:
        msg = f"avg_latency_ms must be positive, got {avg_latency_ms}"
        raise ValueError(msg)

    requests_per_batch_per_second = 1000.0 / avg_latency_ms
    return workers * batch_size * requests_per_batch_per_second


def estimate_latency(
    model_size_gb: float,
    sequence_length: int,
    batch_size: int,
    gpu_memory_bandwidth_gbps: float = 900.0,
) -> float:
    """Estimate inference latency in milliseconds.

    This provides a rough estimate based on memory bandwidth.
    Actual latency depends on many factors including model architecture.

    Args:
        model_size_gb: Model size in gigabytes.
        sequence_length: Input sequence length.
        batch_size: Batch size.
        gpu_memory_bandwidth_gbps: GPU memory bandwidth in GB/s.

    Returns:
        Estimated latency in milliseconds.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> latency = estimate_latency(7.0, 512, 1)
        >>> 5.0 < latency < 20.0
        True

        >>> estimate_latency(0, 512, 1)
        Traceback (most recent call last):
            ...
        ValueError: model_size_gb must be positive, got 0

        >>> estimate_latency(7.0, 0, 1)
        Traceback (most recent call last):
            ...
        ValueError: sequence_length must be positive, got 0
    """
    if model_size_gb <= 0:
        msg = f"model_size_gb must be positive, got {model_size_gb}"
        raise ValueError(msg)
    if sequence_length <= 0:
        msg = f"sequence_length must be positive, got {sequence_length}"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if gpu_memory_bandwidth_gbps <= 0:
        msg = (
            f"gpu_memory_bandwidth_gbps must be positive, "
            f"got {gpu_memory_bandwidth_gbps}"
        )
        raise ValueError(msg)

    # Simplified model: time to read model weights + overhead for compute
    memory_read_time_ms = (model_size_gb / gpu_memory_bandwidth_gbps) * 1000
    # Scale by sequence length and batch size (simplified linear scaling)
    scaling_factor = (sequence_length / 512) * (batch_size / 1)
    # Add compute overhead (roughly 0.5x memory time for modern GPUs)
    compute_overhead = 0.5

    return memory_read_time_ms * scaling_factor * (1 + compute_overhead)


def calculate_cost_per_request(
    gpu_cost_per_hour: float,
    requests_per_second: float,
) -> float:
    """Calculate cost per inference request.

    Args:
        gpu_cost_per_hour: GPU cost in dollars per hour.
        requests_per_second: Throughput in requests per second.

    Returns:
        Cost per request in dollars.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> cost = calculate_cost_per_request(2.0, 100.0)
        >>> round(cost, 8)
        5.56e-06

        >>> calculate_cost_per_request(0, 100.0)
        Traceback (most recent call last):
            ...
        ValueError: gpu_cost_per_hour must be positive, got 0

        >>> calculate_cost_per_request(2.0, 0)
        Traceback (most recent call last):
            ...
        ValueError: requests_per_second must be positive, got 0
    """
    if gpu_cost_per_hour <= 0:
        msg = f"gpu_cost_per_hour must be positive, got {gpu_cost_per_hour}"
        raise ValueError(msg)
    if requests_per_second <= 0:
        msg = f"requests_per_second must be positive, got {requests_per_second}"
        raise ValueError(msg)

    cost_per_second = gpu_cost_per_hour / 3600
    return cost_per_second / requests_per_second


def validate_endpoint_health(
    check_type: str | HealthCheckType,
    is_alive: bool,
    is_ready: bool,
    startup_complete: bool,
) -> bool:
    """Validate endpoint health based on check type.

    Args:
        check_type: Type of health check to perform.
        is_alive: Whether the service is alive.
        is_ready: Whether the service is ready for requests.
        startup_complete: Whether startup has completed.

    Returns:
        True if the endpoint passes the health check.

    Examples:
        >>> validate_endpoint_health("liveness", True, False, False)
        True

        >>> validate_endpoint_health("readiness", True, True, True)
        True

        >>> validate_endpoint_health("readiness", True, False, True)
        False

        >>> validate_endpoint_health("startup", True, False, True)
        True

        >>> validate_endpoint_health(HealthCheckType.LIVENESS, True, False, False)
        True
    """
    if isinstance(check_type, str):
        check_type = get_health_check_type(check_type)

    if check_type == HealthCheckType.LIVENESS:
        return is_alive
    elif check_type == HealthCheckType.READINESS:
        return is_alive and is_ready
    else:  # STARTUP
        return startup_complete


def format_serving_stats(stats: ServingStats) -> str:
    """Format serving statistics as a human-readable string.

    Args:
        stats: Statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = ServingStats(150.0, 25.5, 75.0, 0.85)
        >>> formatted = format_serving_stats(stats)
        >>> "Throughput: 150.00 req/s" in formatted
        True
        >>> "Avg Latency: 25.50 ms" in formatted
        True
        >>> "P99 Latency: 75.00 ms" in formatted
        True
        >>> "GPU Utilization: 85.0%" in formatted
        True
    """
    return (
        f"Serving Stats:\n"
        f"  Throughput: {stats.requests_per_second:.2f} req/s\n"
        f"  Avg Latency: {stats.avg_latency_ms:.2f} ms\n"
        f"  P99 Latency: {stats.p99_latency_ms:.2f} ms\n"
        f"  GPU Utilization: {stats.gpu_utilization * 100:.1f}%"
    )


def get_recommended_serving_config(
    model_size_gb: float,
    expected_qps: float,
    gpu_memory_gb: float = 24.0,
) -> ServingConfig:
    """Get recommended serving configuration for a model.

    Args:
        model_size_gb: Model size in gigabytes.
        expected_qps: Expected queries per second.
        gpu_memory_gb: Available GPU memory in GB. Defaults to 24.0.

    Returns:
        Recommended ServingConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_serving_config(7.0, 100.0)
        >>> config.backend == ServingBackend.FASTAPI  # Small model uses FastAPI
        True

        >>> config = get_recommended_serving_config(15.0, 100.0)
        >>> config.backend == ServingBackend.TGI  # Medium model uses TGI
        True

        >>> config = get_recommended_serving_config(70.0, 10.0)
        >>> config.backend == ServingBackend.VLLM  # Large model uses vLLM
        True
        >>> config.scaling_config.min_replicas >= 1
        True

        >>> get_recommended_serving_config(0, 100.0)
        Traceback (most recent call last):
            ...
        ValueError: model_size_gb must be positive, got 0
    """
    if model_size_gb <= 0:
        msg = f"model_size_gb must be positive, got {model_size_gb}"
        raise ValueError(msg)
    if expected_qps <= 0:
        msg = f"expected_qps must be positive, got {expected_qps}"
        raise ValueError(msg)
    if gpu_memory_gb <= 0:
        msg = f"gpu_memory_gb must be positive, got {gpu_memory_gb}"
        raise ValueError(msg)

    # Determine backend based on model size
    if model_size_gb > 30:
        backend = ServingBackend.VLLM  # Best for large models
    elif model_size_gb > 10:
        backend = ServingBackend.TGI  # Good general purpose
    else:
        backend = ServingBackend.FASTAPI  # Simple for small models

    # Calculate workers based on memory constraints
    gpus_needed = max(1, int(model_size_gb / (gpu_memory_gb * 0.8)))
    workers = max(1, gpus_needed)

    # Calculate replicas based on expected QPS
    # Assume ~50 QPS per replica as baseline
    min_replicas = max(1, int(expected_qps / 100))
    max_replicas = max(min_replicas * 2, min_replicas + 5)

    # Batch size based on model size
    if model_size_gb > 30:
        batch_size = 8
    elif model_size_gb > 10:
        batch_size = 16
    else:
        batch_size = 32

    endpoint_config = create_endpoint_config(
        host="0.0.0.0",
        port=8080,
        workers=workers,
        timeout_seconds=60 if model_size_gb > 30 else 30,
        max_batch_size=batch_size,
    )

    scaling_config = create_scaling_config(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        target_utilization=0.7,
        scale_down_delay=300,
    )

    # Use least_connections for high QPS, round_robin otherwise
    load_balancing = (
        LoadBalancing.LEAST_CONNECTIONS
        if expected_qps > 100
        else LoadBalancing.ROUND_ROBIN
    )

    return ServingConfig(
        backend=backend,
        endpoint_config=endpoint_config,
        scaling_config=scaling_config,
        load_balancing=load_balancing,
    )
