"""Tests for model serving functionality."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.serving import (
    VALID_HEALTH_CHECK_TYPES,
    VALID_LOAD_BALANCING,
    VALID_SERVING_BACKENDS,
    EndpointConfig,
    HealthCheckType,
    HealthStatus,
    InferenceBackend,
    InferenceRequest,
    InferenceResponse,
    LoadBalancing,
    ModelServer,
    ScalingConfig,
    ServerConfig,
    ServerStatus,
    ServingBackend,
    ServingConfig,
    ServingStats,
    calculate_cost_per_request,
    calculate_throughput_capacity,
    compute_server_metrics,
    create_endpoint_config,
    create_scaling_config,
    create_server,
    create_serving_config,
    estimate_latency,
    format_server_info,
    format_serving_stats,
    get_health_check_type,
    get_health_status,
    get_inference_backend,
    get_load_balancing,
    get_recommended_serving_config,
    get_server_status,
    get_serving_backend,
    list_health_check_types,
    list_inference_backends,
    list_load_balancing_strategies,
    list_server_statuses,
    list_serving_backends,
    process_batch,
    process_request,
    start_server,
    stop_server,
    validate_endpoint_config,
    validate_endpoint_health,
    validate_health_check_type,
    validate_inference_backend,
    validate_load_balancing,
    validate_scaling_config,
    validate_server_config,
    validate_server_status,
    validate_serving_backend,
    validate_serving_config,
)


class TestServerStatus:
    """Tests for ServerStatus enum."""

    def test_starting_value(self) -> None:
        """Test STARTING value."""
        assert ServerStatus.STARTING.value == "starting"

    def test_ready_value(self) -> None:
        """Test READY value."""
        assert ServerStatus.READY.value == "ready"

    def test_busy_value(self) -> None:
        """Test BUSY value."""
        assert ServerStatus.BUSY.value == "busy"

    def test_degraded_value(self) -> None:
        """Test DEGRADED value."""
        assert ServerStatus.DEGRADED.value == "degraded"

    def test_stopping_value(self) -> None:
        """Test STOPPING value."""
        assert ServerStatus.STOPPING.value == "stopping"

    def test_stopped_value(self) -> None:
        """Test STOPPED value."""
        assert ServerStatus.STOPPED.value == "stopped"


class TestInferenceBackend:
    """Tests for InferenceBackend enum."""

    def test_pytorch_value(self) -> None:
        """Test PYTORCH value."""
        assert InferenceBackend.PYTORCH.value == "pytorch"

    def test_onnx_value(self) -> None:
        """Test ONNX value."""
        assert InferenceBackend.ONNX.value == "onnx"

    def test_tensorrt_value(self) -> None:
        """Test TENSORRT value."""
        assert InferenceBackend.TENSORRT.value == "tensorrt"

    def test_vllm_value(self) -> None:
        """Test VLLM value."""
        assert InferenceBackend.VLLM.value == "vllm"

    def test_tgi_value(self) -> None:
        """Test TGI value."""
        assert InferenceBackend.TGI.value == "tgi"


class TestServingBackend:
    """Tests for ServingBackend enum."""

    def test_vllm_value(self) -> None:
        """Test VLLM value."""
        assert ServingBackend.VLLM.value == "vllm"

    def test_tgi_value(self) -> None:
        """Test TGI value."""
        assert ServingBackend.TGI.value == "tgi"

    def test_triton_value(self) -> None:
        """Test TRITON value."""
        assert ServingBackend.TRITON.value == "triton"

    def test_ray_serve_value(self) -> None:
        """Test RAY_SERVE value."""
        assert ServingBackend.RAY_SERVE.value == "ray_serve"

    def test_fastapi_value(self) -> None:
        """Test FASTAPI value."""
        assert ServingBackend.FASTAPI.value == "fastapi"


class TestLoadBalancing:
    """Tests for LoadBalancing enum."""

    def test_round_robin_value(self) -> None:
        """Test ROUND_ROBIN value."""
        assert LoadBalancing.ROUND_ROBIN.value == "round_robin"

    def test_least_connections_value(self) -> None:
        """Test LEAST_CONNECTIONS value."""
        assert LoadBalancing.LEAST_CONNECTIONS.value == "least_connections"

    def test_random_value(self) -> None:
        """Test RANDOM value."""
        assert LoadBalancing.RANDOM.value == "random"

    def test_weighted_value(self) -> None:
        """Test WEIGHTED value."""
        assert LoadBalancing.WEIGHTED.value == "weighted"


class TestHealthCheckType:
    """Tests for HealthCheckType enum."""

    def test_liveness_value(self) -> None:
        """Test LIVENESS value."""
        assert HealthCheckType.LIVENESS.value == "liveness"

    def test_readiness_value(self) -> None:
        """Test READINESS value."""
        assert HealthCheckType.READINESS.value == "readiness"

    def test_startup_value(self) -> None:
        """Test STARTUP value."""
        assert HealthCheckType.STARTUP.value == "startup"


class TestValidFrozensets:
    """Tests for VALID_* frozensets."""

    def test_valid_serving_backends(self) -> None:
        """Test VALID_SERVING_BACKENDS contains all backends."""
        assert "vllm" in VALID_SERVING_BACKENDS
        assert "tgi" in VALID_SERVING_BACKENDS
        assert "triton" in VALID_SERVING_BACKENDS
        assert "ray_serve" in VALID_SERVING_BACKENDS
        assert "fastapi" in VALID_SERVING_BACKENDS
        assert len(VALID_SERVING_BACKENDS) == 5

    def test_valid_load_balancing(self) -> None:
        """Test VALID_LOAD_BALANCING contains all strategies."""
        assert "round_robin" in VALID_LOAD_BALANCING
        assert "least_connections" in VALID_LOAD_BALANCING
        assert "random" in VALID_LOAD_BALANCING
        assert "weighted" in VALID_LOAD_BALANCING
        assert len(VALID_LOAD_BALANCING) == 4

    def test_valid_health_check_types(self) -> None:
        """Test VALID_HEALTH_CHECK_TYPES contains all types."""
        assert "liveness" in VALID_HEALTH_CHECK_TYPES
        assert "readiness" in VALID_HEALTH_CHECK_TYPES
        assert "startup" in VALID_HEALTH_CHECK_TYPES
        assert len(VALID_HEALTH_CHECK_TYPES) == 3


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ServerConfig()
        assert config.host == "127.0.0.1"  # Secure default: localhost only
        assert config.port == 8000
        assert config.model_path is None
        assert config.backend == InferenceBackend.PYTORCH
        assert config.max_batch_size == 32
        assert config.max_concurrent_requests == 100
        assert config.timeout_seconds == 30

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ServerConfig(
            host="localhost",
            port=9000,
            model_path="/models/bert",
            backend=InferenceBackend.VLLM,
            max_batch_size=64,
        )
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.model_path == "/models/bert"
        assert config.backend == InferenceBackend.VLLM

    def test_frozen(self) -> None:
        """Test that ServerConfig is immutable."""
        config = ServerConfig()
        with pytest.raises(AttributeError):
            config.port = 9000  # type: ignore[misc]


class TestEndpointConfig:
    """Tests for EndpointConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating EndpointConfig instance."""
        config = EndpointConfig(
            host="0.0.0.0",
            port=8080,
            workers=4,
            timeout_seconds=30,
            max_batch_size=32,
        )
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.workers == 4
        assert config.timeout_seconds == 30
        assert config.max_batch_size == 32

    def test_frozen(self) -> None:
        """Test that EndpointConfig is immutable."""
        config = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        with pytest.raises(AttributeError):
            config.port = 9000  # type: ignore[misc]


class TestScalingConfig:
    """Tests for ScalingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ScalingConfig instance."""
        config = ScalingConfig(
            min_replicas=1,
            max_replicas=10,
            target_utilization=0.7,
            scale_down_delay=300,
        )
        assert config.min_replicas == 1
        assert config.max_replicas == 10
        assert config.target_utilization == 0.7
        assert config.scale_down_delay == 300

    def test_frozen(self) -> None:
        """Test that ScalingConfig is immutable."""
        config = ScalingConfig(1, 10, 0.7, 300)
        with pytest.raises(AttributeError):
            config.min_replicas = 2  # type: ignore[misc]


class TestServingConfig:
    """Tests for ServingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ServingConfig instance."""
        endpoint = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        scaling = ScalingConfig(1, 10, 0.7, 300)
        config = ServingConfig(
            backend=ServingBackend.VLLM,
            endpoint_config=endpoint,
            scaling_config=scaling,
            load_balancing=LoadBalancing.ROUND_ROBIN,
        )
        assert config.backend == ServingBackend.VLLM
        assert config.load_balancing == LoadBalancing.ROUND_ROBIN

    def test_frozen(self) -> None:
        """Test that ServingConfig is immutable."""
        endpoint = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        scaling = ScalingConfig(1, 10, 0.7, 300)
        config = ServingConfig(
            ServingBackend.VLLM, endpoint, scaling, LoadBalancing.ROUND_ROBIN
        )
        with pytest.raises(AttributeError):
            config.backend = ServingBackend.TGI  # type: ignore[misc]


class TestServingStats:
    """Tests for ServingStats dataclass."""

    def test_creation(self) -> None:
        """Test creating ServingStats instance."""
        stats = ServingStats(
            requests_per_second=150.0,
            avg_latency_ms=25.5,
            p99_latency_ms=75.0,
            gpu_utilization=0.85,
        )
        assert stats.requests_per_second == 150.0
        assert stats.avg_latency_ms == 25.5
        assert stats.p99_latency_ms == 75.0
        assert stats.gpu_utilization == 0.85

    def test_frozen(self) -> None:
        """Test that ServingStats is immutable."""
        stats = ServingStats(150.0, 25.5, 75.0, 0.85)
        with pytest.raises(AttributeError):
            stats.requests_per_second = 200.0  # type: ignore[misc]


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_creation(self) -> None:
        """Test creating HealthStatus instance."""
        health = HealthStatus(
            status=ServerStatus.READY,
            model_loaded=True,
            memory_used_mb=1024,
            requests_pending=5,
            uptime_seconds=3600,
        )
        assert health.status == ServerStatus.READY
        assert health.model_loaded is True
        assert health.memory_used_mb == 1024

    def test_frozen(self) -> None:
        """Test that HealthStatus is immutable."""
        health = HealthStatus(ServerStatus.READY, True, 1024, 0, 100.0)
        with pytest.raises(AttributeError):
            health.model_loaded = False  # type: ignore[misc]


class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""

    def test_creation(self) -> None:
        """Test creating InferenceRequest instance."""
        request = InferenceRequest(
            request_id="req-001",
            inputs="Hello, world!",
            parameters={"max_length": 100},
        )
        assert request.request_id == "req-001"
        assert request.inputs == "Hello, world!"
        assert request.parameters == {"max_length": 100}

    def test_default_parameters(self) -> None:
        """Test default parameters value."""
        request = InferenceRequest("req-001", "test")
        assert request.parameters is None

    def test_frozen(self) -> None:
        """Test that InferenceRequest is immutable."""
        request = InferenceRequest("req-001", "test")
        with pytest.raises(AttributeError):
            request.inputs = "new"  # type: ignore[misc]


class TestInferenceResponse:
    """Tests for InferenceResponse dataclass."""

    def test_success_response(self) -> None:
        """Test creating successful response."""
        response = InferenceResponse(
            request_id="req-001",
            outputs={"text": "output"},
            latency_ms=50.0,
            success=True,
            error_message=None,
        )
        assert response.success is True
        assert response.outputs == {"text": "output"}

    def test_failed_response(self) -> None:
        """Test creating failed response."""
        response = InferenceResponse(
            request_id="req-001",
            outputs=None,
            latency_ms=10.0,
            success=False,
            error_message="Model error",
        )
        assert response.success is False
        assert response.error_message == "Model error"


class TestModelServer:
    """Tests for ModelServer dataclass."""

    def test_creation(self) -> None:
        """Test creating ModelServer instance."""
        config = ServerConfig(port=8080)
        server = ModelServer(config)
        assert server.config.port == 8080
        assert server.status == ServerStatus.STOPPED
        assert server.requests_processed == 0


class TestValidateServerConfig:
    """Tests for validate_server_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ServerConfig(port=8000, max_batch_size=16)
        validate_server_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_server_config(None)  # type: ignore[arg-type]

    def test_port_zero_raises_error(self) -> None:
        """Test that port 0 raises ValueError."""
        config = ServerConfig(port=0)
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            validate_server_config(config)

    def test_port_too_high_raises_error(self) -> None:
        """Test that port > 65535 raises ValueError."""
        config = ServerConfig(port=70000)
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            validate_server_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero max_batch_size raises ValueError."""
        config = ServerConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            validate_server_config(config)

    def test_zero_concurrent_raises_error(self) -> None:
        """Test that zero max_concurrent_requests raises ValueError."""
        config = ServerConfig(max_concurrent_requests=0)
        with pytest.raises(ValueError, match="max_concurrent_requests must be"):
            validate_server_config(config)

    def test_zero_timeout_raises_error(self) -> None:
        """Test that zero timeout_seconds raises ValueError."""
        config = ServerConfig(timeout_seconds=0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_server_config(config)


class TestValidateEndpointConfig:
    """Tests for validate_endpoint_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        validate_endpoint_config(config)  # Should not raise

    def test_port_zero_raises_error(self) -> None:
        """Test that port 0 raises ValueError."""
        config = EndpointConfig("0.0.0.0", 0, 4, 30, 32)
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            validate_endpoint_config(config)

    def test_workers_zero_raises_error(self) -> None:
        """Test that workers 0 raises ValueError."""
        config = EndpointConfig("0.0.0.0", 8080, 0, 30, 32)
        with pytest.raises(ValueError, match="workers must be positive"):
            validate_endpoint_config(config)

    def test_timeout_zero_raises_error(self) -> None:
        """Test that timeout 0 raises ValueError."""
        config = EndpointConfig("0.0.0.0", 8080, 4, 0, 32)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_endpoint_config(config)

    def test_batch_size_zero_raises_error(self) -> None:
        """Test that batch_size 0 raises ValueError."""
        config = EndpointConfig("0.0.0.0", 8080, 4, 30, 0)
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            validate_endpoint_config(config)


class TestValidateScalingConfig:
    """Tests for validate_scaling_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ScalingConfig(1, 10, 0.7, 300)
        validate_scaling_config(config)  # Should not raise

    def test_min_replicas_zero_raises_error(self) -> None:
        """Test that min_replicas 0 raises ValueError."""
        config = ScalingConfig(0, 10, 0.7, 300)
        with pytest.raises(ValueError, match="min_replicas must be positive"):
            validate_scaling_config(config)

    def test_max_less_than_min_raises_error(self) -> None:
        """Test that max < min raises ValueError."""
        config = ScalingConfig(10, 5, 0.7, 300)
        with pytest.raises(ValueError, match="max_replicas must be >= min_replicas"):
            validate_scaling_config(config)

    def test_utilization_zero_raises_error(self) -> None:
        """Test that utilization 0 raises ValueError."""
        config = ScalingConfig(1, 10, 0.0, 300)
        with pytest.raises(ValueError, match="target_utilization must be between"):
            validate_scaling_config(config)

    def test_utilization_over_one_raises_error(self) -> None:
        """Test that utilization > 1 raises ValueError."""
        config = ScalingConfig(1, 10, 1.5, 300)
        with pytest.raises(ValueError, match="target_utilization must be between"):
            validate_scaling_config(config)

    def test_negative_delay_raises_error(self) -> None:
        """Test that negative delay raises ValueError."""
        config = ScalingConfig(1, 10, 0.7, -1)
        with pytest.raises(ValueError, match="scale_down_delay must be non-negative"):
            validate_scaling_config(config)


class TestValidateServingConfig:
    """Tests for validate_serving_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        endpoint = EndpointConfig("0.0.0.0", 8080, 4, 30, 32)
        scaling = ScalingConfig(1, 10, 0.7, 300)
        config = ServingConfig(
            ServingBackend.VLLM, endpoint, scaling, LoadBalancing.ROUND_ROBIN
        )
        validate_serving_config(config)  # Should not raise


class TestCreateServer:
    """Tests for create_server function."""

    def test_creates_server(self) -> None:
        """Test that function creates a server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        assert isinstance(server, ModelServer)
        assert server.status == ServerStatus.STOPPED

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_server(None)  # type: ignore[arg-type]


class TestCreateEndpointConfig:
    """Tests for create_endpoint_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_endpoint_config()
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.workers == 4
        assert config.timeout_seconds == 30
        assert config.max_batch_size == 32

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = create_endpoint_config(host="0.0.0.0", port=9000, workers=8)
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.workers == 8

    def test_invalid_port_raises_error(self) -> None:
        """Test that invalid port raises ValueError."""
        with pytest.raises(ValueError, match="port must be between"):
            create_endpoint_config(port=0)


class TestCreateScalingConfig:
    """Tests for create_scaling_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_scaling_config()
        assert config.min_replicas == 1
        assert config.max_replicas == 10
        assert config.target_utilization == 0.7
        assert config.scale_down_delay == 300

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = create_scaling_config(min_replicas=2, max_replicas=20)
        assert config.min_replicas == 2
        assert config.max_replicas == 20

    def test_invalid_min_replicas_raises_error(self) -> None:
        """Test that invalid min_replicas raises ValueError."""
        with pytest.raises(ValueError, match="min_replicas must be positive"):
            create_scaling_config(min_replicas=0)


class TestCreateServingConfig:
    """Tests for create_serving_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_serving_config()
        assert config.backend == ServingBackend.VLLM
        assert config.load_balancing == LoadBalancing.ROUND_ROBIN

    def test_string_backend(self) -> None:
        """Test creating with string backend."""
        config = create_serving_config(backend="tgi")
        assert config.backend == ServingBackend.TGI

    def test_string_load_balancing(self) -> None:
        """Test creating with string load balancing."""
        config = create_serving_config(load_balancing="least_connections")
        assert config.load_balancing == LoadBalancing.LEAST_CONNECTIONS

    def test_custom_endpoint_config(self) -> None:
        """Test with custom endpoint config."""
        endpoint = create_endpoint_config(port=9000)
        config = create_serving_config(endpoint_config=endpoint)
        assert config.endpoint_config.port == 9000

    def test_custom_scaling_config(self) -> None:
        """Test with custom scaling config."""
        scaling = create_scaling_config(min_replicas=2)
        config = create_serving_config(scaling_config=scaling)
        assert config.scaling_config.min_replicas == 2


class TestStartServer:
    """Tests for start_server function."""

    def test_starts_server(self) -> None:
        """Test starting a server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        result = start_server(server)
        assert result is True
        assert server.status == ServerStatus.READY

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            start_server(None)  # type: ignore[arg-type]

    def test_already_running_raises_error(self) -> None:
        """Test that starting already running server raises error."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        with pytest.raises(RuntimeError, match="already running"):
            start_server(server)


class TestStopServer:
    """Tests for stop_server function."""

    def test_stops_running_server(self) -> None:
        """Test stopping a running server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        result = stop_server(server)
        assert result is True
        assert server.status == ServerStatus.STOPPED

    def test_stops_stopped_server(self) -> None:
        """Test stopping an already stopped server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        result = stop_server(server)
        assert result is True

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            stop_server(None)  # type: ignore[arg-type]


class TestGetHealthStatus:
    """Tests for get_health_status function."""

    def test_ready_server_health(self) -> None:
        """Test health status of ready server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        health = get_health_status(server)
        assert health.status == ServerStatus.READY
        assert health.model_loaded is True

    def test_stopped_server_health(self) -> None:
        """Test health status of stopped server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        health = get_health_status(server)
        assert health.status == ServerStatus.STOPPED
        assert health.model_loaded is False

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            get_health_status(None)  # type: ignore[arg-type]


class TestProcessRequest:
    """Tests for process_request function."""

    def test_successful_request(self) -> None:
        """Test processing successful request."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        request = InferenceRequest("req-001", "test input")
        response = process_request(server, request, lambda x: f"output: {x}")
        assert response.success is True
        assert response.outputs == "output: test input"
        assert response.latency_ms > 0

    def test_failed_request(self) -> None:
        """Test processing request with error."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        request = InferenceRequest("req-001", "test")

        def failing_fn(x: str) -> None:
            msg = "Model error"
            raise RuntimeError(msg)

        response = process_request(server, request, failing_fn)
        assert response.success is False
        assert "Model error" in (response.error_message or "")

    def test_increments_requests_processed(self) -> None:
        """Test that successful request increments counter."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        request = InferenceRequest("req-001", "test")
        process_request(server, request, lambda x: x)
        assert server.requests_processed == 1

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        request = InferenceRequest("req-001", "test")
        with pytest.raises(ValueError, match="server cannot be None"):
            process_request(None, request, lambda x: x)  # type: ignore[arg-type]

    def test_none_request_raises_error(self) -> None:
        """Test that None request raises ValueError."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        with pytest.raises(ValueError, match="request cannot be None"):
            process_request(server, None, lambda x: x)  # type: ignore[arg-type]

    def test_none_inference_fn_raises_error(self) -> None:
        """Test that None inference_fn raises ValueError."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        request = InferenceRequest("req-001", "test")
        with pytest.raises(ValueError, match="inference_fn cannot be None"):
            process_request(server, request, None)  # type: ignore[arg-type]

    def test_not_ready_server_raises_error(self) -> None:
        """Test that non-ready server raises RuntimeError."""
        config = ServerConfig(port=8080)
        server = create_server(config)  # Not started
        request = InferenceRequest("req-001", "test")
        with pytest.raises(RuntimeError, match="not ready"):
            process_request(server, request, lambda x: x)


class TestProcessBatch:
    """Tests for process_batch function."""

    def test_successful_batch(self) -> None:
        """Test processing successful batch."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        requests = [
            InferenceRequest("req-001", "a"),
            InferenceRequest("req-002", "b"),
        ]
        responses = process_batch(server, requests, lambda x: [f"out-{i}" for i in x])
        assert len(responses) == 2
        assert all(r.success for r in responses)

    def test_empty_batch(self) -> None:
        """Test processing empty batch."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        responses = process_batch(server, [], lambda x: x)
        assert responses == []

    def test_failed_batch(self) -> None:
        """Test processing batch with error."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        requests = [InferenceRequest("req-001", "test")]

        def failing_fn(x: list) -> None:
            msg = "Batch error"
            raise RuntimeError(msg)

        responses = process_batch(server, requests, failing_fn)
        assert all(not r.success for r in responses)

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        with pytest.raises(ValueError, match="server cannot be None"):
            process_batch(None, [], lambda x: x)  # type: ignore[arg-type]

    def test_none_requests_raises_error(self) -> None:
        """Test that None requests raises ValueError."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        with pytest.raises(ValueError, match="requests cannot be None"):
            process_batch(server, None, lambda x: x)  # type: ignore[arg-type]

    def test_none_inference_fn_raises_error(self) -> None:
        """Test that None inference_fn raises ValueError."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        with pytest.raises(ValueError, match="inference_fn cannot be None"):
            process_batch(server, [], None)  # type: ignore[arg-type]


class TestListServerStatuses:
    """Tests for list_server_statuses function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        statuses = list_server_statuses()
        assert isinstance(statuses, list)

    def test_contains_expected_statuses(self) -> None:
        """Test that list contains expected statuses."""
        statuses = list_server_statuses()
        assert "ready" in statuses
        assert "stopped" in statuses
        assert "starting" in statuses

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        statuses = list_server_statuses()
        assert statuses == sorted(statuses)


class TestListServingBackends:
    """Tests for list_serving_backends function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        backends = list_serving_backends()
        assert isinstance(backends, list)

    def test_contains_expected_backends(self) -> None:
        """Test that list contains expected backends."""
        backends = list_serving_backends()
        assert "vllm" in backends
        assert "tgi" in backends
        assert "triton" in backends

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        backends = list_serving_backends()
        assert backends == sorted(backends)


class TestListLoadBalancingStrategies:
    """Tests for list_load_balancing_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_load_balancing_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_load_balancing_strategies()
        assert "round_robin" in strategies
        assert "least_connections" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_load_balancing_strategies()
        assert strategies == sorted(strategies)


class TestListHealthCheckTypes:
    """Tests for list_health_check_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_health_check_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_health_check_types()
        assert "liveness" in types
        assert "readiness" in types
        assert "startup" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_health_check_types()
        assert types == sorted(types)


class TestValidateServerStatus:
    """Tests for validate_server_status function."""

    def test_valid_ready(self) -> None:
        """Test validation of ready status."""
        assert validate_server_status("ready") is True

    def test_valid_stopped(self) -> None:
        """Test validation of stopped status."""
        assert validate_server_status("stopped") is True

    def test_invalid_status(self) -> None:
        """Test validation of invalid status."""
        assert validate_server_status("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_server_status("") is False


class TestValidateServingBackend:
    """Tests for validate_serving_backend function."""

    def test_valid_vllm(self) -> None:
        """Test validation of vllm backend."""
        assert validate_serving_backend("vllm") is True

    def test_valid_tgi(self) -> None:
        """Test validation of tgi backend."""
        assert validate_serving_backend("tgi") is True

    def test_invalid_backend(self) -> None:
        """Test validation of invalid backend."""
        assert validate_serving_backend("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_serving_backend("") is False


class TestValidateLoadBalancing:
    """Tests for validate_load_balancing function."""

    def test_valid_round_robin(self) -> None:
        """Test validation of round_robin."""
        assert validate_load_balancing("round_robin") is True

    def test_valid_least_connections(self) -> None:
        """Test validation of least_connections."""
        assert validate_load_balancing("least_connections") is True

    def test_invalid_strategy(self) -> None:
        """Test validation of invalid strategy."""
        assert validate_load_balancing("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_load_balancing("") is False


class TestValidateHealthCheckType:
    """Tests for validate_health_check_type function."""

    def test_valid_liveness(self) -> None:
        """Test validation of liveness."""
        assert validate_health_check_type("liveness") is True

    def test_valid_readiness(self) -> None:
        """Test validation of readiness."""
        assert validate_health_check_type("readiness") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_health_check_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_health_check_type("") is False


class TestGetServerStatus:
    """Tests for get_server_status function."""

    def test_get_ready(self) -> None:
        """Test getting READY status."""
        result = get_server_status("ready")
        assert result == ServerStatus.READY

    def test_get_stopped(self) -> None:
        """Test getting STOPPED status."""
        result = get_server_status("stopped")
        assert result == ServerStatus.STOPPED

    def test_invalid_status_raises_error(self) -> None:
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="invalid server status"):
            get_server_status("invalid")


class TestGetServingBackend:
    """Tests for get_serving_backend function."""

    def test_get_vllm(self) -> None:
        """Test getting VLLM backend."""
        result = get_serving_backend("vllm")
        assert result == ServingBackend.VLLM

    def test_get_tgi(self) -> None:
        """Test getting TGI backend."""
        result = get_serving_backend("tgi")
        assert result == ServingBackend.TGI

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="invalid serving backend"):
            get_serving_backend("invalid")


class TestGetLoadBalancing:
    """Tests for get_load_balancing function."""

    def test_get_round_robin(self) -> None:
        """Test getting ROUND_ROBIN strategy."""
        result = get_load_balancing("round_robin")
        assert result == LoadBalancing.ROUND_ROBIN

    def test_get_least_connections(self) -> None:
        """Test getting LEAST_CONNECTIONS strategy."""
        result = get_load_balancing("least_connections")
        assert result == LoadBalancing.LEAST_CONNECTIONS

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid load balancing strategy"):
            get_load_balancing("invalid")


class TestGetHealthCheckType:
    """Tests for get_health_check_type function."""

    def test_get_liveness(self) -> None:
        """Test getting LIVENESS type."""
        result = get_health_check_type("liveness")
        assert result == HealthCheckType.LIVENESS

    def test_get_readiness(self) -> None:
        """Test getting READINESS type."""
        result = get_health_check_type("readiness")
        assert result == HealthCheckType.READINESS

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid health check type"):
            get_health_check_type("invalid")


class TestListInferenceBackends:
    """Tests for list_inference_backends function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        backends = list_inference_backends()
        assert isinstance(backends, list)

    def test_contains_expected_backends(self) -> None:
        """Test that list contains expected backends."""
        backends = list_inference_backends()
        assert "pytorch" in backends
        assert "vllm" in backends
        assert "onnx" in backends

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        backends = list_inference_backends()
        assert backends == sorted(backends)


class TestValidateInferenceBackend:
    """Tests for validate_inference_backend function."""

    def test_valid_pytorch(self) -> None:
        """Test validation of pytorch backend."""
        assert validate_inference_backend("pytorch") is True

    def test_valid_vllm(self) -> None:
        """Test validation of vllm backend."""
        assert validate_inference_backend("vllm") is True

    def test_invalid_backend(self) -> None:
        """Test validation of invalid backend."""
        assert validate_inference_backend("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_inference_backend("") is False


class TestGetInferenceBackend:
    """Tests for get_inference_backend function."""

    def test_get_pytorch(self) -> None:
        """Test getting PYTORCH backend."""
        result = get_inference_backend("pytorch")
        assert result == InferenceBackend.PYTORCH

    def test_get_vllm(self) -> None:
        """Test getting VLLM backend."""
        result = get_inference_backend("vllm")
        assert result == InferenceBackend.VLLM

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="invalid inference backend"):
            get_inference_backend("invalid")


class TestFormatServerInfo:
    """Tests for format_server_info function."""

    def test_format_basic_info(self) -> None:
        """Test formatting basic server info."""
        config = ServerConfig(host="localhost", port=8080)
        server = create_server(config)
        formatted = format_server_info(server)
        assert "localhost:8080" in formatted
        assert "stopped" in formatted

    def test_format_with_model_path(self) -> None:
        """Test formatting with model path."""
        config = ServerConfig(port=8080, model_path="/models/bert")
        server = create_server(config)
        formatted = format_server_info(server)
        assert "/models/bert" in formatted

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_server_info(None)  # type: ignore[arg-type]


class TestFormatServingStats:
    """Tests for format_serving_stats function."""

    def test_format_stats(self) -> None:
        """Test formatting serving stats."""
        stats = ServingStats(150.0, 25.5, 75.0, 0.85)
        formatted = format_serving_stats(stats)
        assert "Throughput: 150.00 req/s" in formatted
        assert "Avg Latency: 25.50 ms" in formatted
        assert "P99 Latency: 75.00 ms" in formatted
        assert "GPU Utilization: 85.0%" in formatted


class TestComputeServerMetrics:
    """Tests for compute_server_metrics function."""

    def test_compute_metrics(self) -> None:
        """Test computing server metrics."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        metrics = compute_server_metrics(server)
        assert metrics["status"] == "stopped"
        assert metrics["is_running"] is False
        assert metrics["requests_processed"] == 0

    def test_running_server_metrics(self) -> None:
        """Test metrics for running server."""
        config = ServerConfig(port=8080)
        server = create_server(config)
        start_server(server)
        metrics = compute_server_metrics(server)
        assert metrics["status"] == "ready"
        assert metrics["is_running"] is True

    def test_none_server_raises_error(self) -> None:
        """Test that None server raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compute_server_metrics(None)  # type: ignore[arg-type]


class TestCalculateThroughputCapacity:
    """Tests for calculate_throughput_capacity function."""

    def test_basic_calculation(self) -> None:
        """Test basic throughput calculation."""
        result = calculate_throughput_capacity(4, 32, 50.0)
        assert result == 2560.0

    def test_single_worker(self) -> None:
        """Test with single worker."""
        result = calculate_throughput_capacity(1, 1, 100.0)
        assert result == 10.0

    def test_zero_workers_raises_error(self) -> None:
        """Test that zero workers raises ValueError."""
        with pytest.raises(ValueError, match="workers must be positive"):
            calculate_throughput_capacity(0, 32, 50.0)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_throughput_capacity(4, 0, 50.0)

    def test_zero_latency_raises_error(self) -> None:
        """Test that zero latency raises ValueError."""
        with pytest.raises(ValueError, match="avg_latency_ms must be positive"):
            calculate_throughput_capacity(4, 32, 0)


class TestEstimateLatency:
    """Tests for estimate_latency function."""

    def test_basic_estimation(self) -> None:
        """Test basic latency estimation."""
        latency = estimate_latency(7.0, 512, 1)
        assert 5.0 < latency < 20.0

    def test_zero_model_size_raises_error(self) -> None:
        """Test that zero model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            estimate_latency(0, 512, 1)

    def test_zero_sequence_length_raises_error(self) -> None:
        """Test that zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_latency(7.0, 0, 1)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_latency(7.0, 512, 0)

    def test_zero_bandwidth_raises_error(self) -> None:
        """Test that zero bandwidth raises ValueError."""
        with pytest.raises(
            ValueError, match="gpu_memory_bandwidth_gbps must be positive"
        ):
            estimate_latency(7.0, 512, 1, gpu_memory_bandwidth_gbps=0)


class TestCalculateCostPerRequest:
    """Tests for calculate_cost_per_request function."""

    def test_basic_calculation(self) -> None:
        """Test basic cost calculation."""
        cost = calculate_cost_per_request(2.0, 100.0)
        # $2/hour = $0.000555/second, / 100 requests = $5.55e-06
        assert round(cost, 8) == 5.56e-06

    def test_zero_cost_raises_error(self) -> None:
        """Test that zero cost raises ValueError."""
        with pytest.raises(ValueError, match="gpu_cost_per_hour must be positive"):
            calculate_cost_per_request(0, 100.0)

    def test_zero_requests_raises_error(self) -> None:
        """Test that zero requests raises ValueError."""
        with pytest.raises(ValueError, match="requests_per_second must be positive"):
            calculate_cost_per_request(2.0, 0)


class TestValidateEndpointHealth:
    """Tests for validate_endpoint_health function."""

    def test_liveness_check_alive(self) -> None:
        """Test liveness check when alive."""
        result = validate_endpoint_health("liveness", True, False, False)
        assert result is True

    def test_liveness_check_not_alive(self) -> None:
        """Test liveness check when not alive."""
        result = validate_endpoint_health("liveness", False, False, False)
        assert result is False

    def test_readiness_check_ready(self) -> None:
        """Test readiness check when ready."""
        result = validate_endpoint_health("readiness", True, True, True)
        assert result is True

    def test_readiness_check_not_ready(self) -> None:
        """Test readiness check when not ready."""
        result = validate_endpoint_health("readiness", True, False, True)
        assert result is False

    def test_startup_check_complete(self) -> None:
        """Test startup check when complete."""
        result = validate_endpoint_health("startup", True, False, True)
        assert result is True

    def test_startup_check_not_complete(self) -> None:
        """Test startup check when not complete."""
        result = validate_endpoint_health("startup", True, True, False)
        assert result is False

    def test_with_enum_type(self) -> None:
        """Test with HealthCheckType enum."""
        result = validate_endpoint_health(HealthCheckType.LIVENESS, True, False, False)
        assert result is True


class TestGetRecommendedServingConfig:
    """Tests for get_recommended_serving_config function."""

    def test_small_model(self) -> None:
        """Test config for small model."""
        config = get_recommended_serving_config(7.0, 100.0)
        valid_backends = (
            ServingBackend.VLLM, ServingBackend.TGI, ServingBackend.FASTAPI
        )
        assert config.backend in valid_backends

    def test_large_model(self) -> None:
        """Test config for large model."""
        config = get_recommended_serving_config(70.0, 10.0)
        assert config.backend == ServingBackend.VLLM
        assert config.scaling_config.min_replicas >= 1

    def test_high_qps(self) -> None:
        """Test config for high QPS."""
        config = get_recommended_serving_config(7.0, 200.0)
        assert config.load_balancing == LoadBalancing.LEAST_CONNECTIONS

    def test_zero_model_size_raises_error(self) -> None:
        """Test that zero model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            get_recommended_serving_config(0, 100.0)

    def test_zero_qps_raises_error(self) -> None:
        """Test that zero expected_qps raises ValueError."""
        with pytest.raises(ValueError, match="expected_qps must be positive"):
            get_recommended_serving_config(7.0, 0)

    def test_zero_gpu_memory_raises_error(self) -> None:
        """Test that zero gpu_memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb must be positive"):
            get_recommended_serving_config(7.0, 100.0, gpu_memory_gb=0)
