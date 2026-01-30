"""Tests for model serving functionality."""

from __future__ import annotations

import pytest

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
