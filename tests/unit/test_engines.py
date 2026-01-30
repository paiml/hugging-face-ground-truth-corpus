"""Tests for inference.engines module."""

from __future__ import annotations

import pytest

from hf_gtc.inference.engines import (
    VALID_ENGINE_FEATURES,
    VALID_ENGINE_TYPES,
    VALID_QUANTIZATION_BACKENDS,
    EngineConfig,
    EngineFeature,
    EngineStats,
    EngineType,
    LlamaCppConfig,
    QuantizationBackend,
    TGIConfig,
    VLLMConfig,
    check_engine_compatibility,
    compare_engine_performance,
    create_engine_config,
    create_engine_stats,
    create_llamacpp_config,
    create_tgi_config,
    create_vllm_config,
    estimate_engine_throughput,
    format_engine_stats,
    get_engine_feature,
    get_engine_features,
    get_engine_type,
    get_quantization_backend,
    get_recommended_engine_config,
    list_engine_features,
    list_engine_types,
    list_quantization_backends,
    validate_engine_config,
    validate_engine_stats,
    validate_llamacpp_config,
    validate_tgi_config,
    validate_vllm_config,
)


class TestEngineType:
    """Tests for EngineType enum."""

    def test_all_types_have_values(self) -> None:
        """All engine types have string values."""
        for engine in EngineType:
            assert isinstance(engine.value, str)

    def test_vllm_value(self) -> None:
        """VLLM has correct value."""
        assert EngineType.VLLM.value == "vllm"

    def test_tgi_value(self) -> None:
        """TGI has correct value."""
        assert EngineType.TGI.value == "tgi"

    def test_llamacpp_value(self) -> None:
        """LLAMACPP has correct value."""
        assert EngineType.LLAMACPP.value == "llamacpp"

    def test_tensorrt_value(self) -> None:
        """TENSORRT has correct value."""
        assert EngineType.TENSORRT.value == "tensorrt"

    def test_ctranslate2_value(self) -> None:
        """CTRANSLATE2 has correct value."""
        assert EngineType.CTRANSLATE2.value == "ctranslate2"

    def test_onnxruntime_value(self) -> None:
        """ONNXRUNTIME has correct value."""
        assert EngineType.ONNXRUNTIME.value == "onnxruntime"

    def test_valid_types_frozenset(self) -> None:
        """VALID_ENGINE_TYPES is a frozenset."""
        assert isinstance(VALID_ENGINE_TYPES, frozenset)

    def test_valid_types_count(self) -> None:
        """VALID_ENGINE_TYPES has correct count."""
        assert len(VALID_ENGINE_TYPES) == 6


class TestQuantizationBackend:
    """Tests for QuantizationBackend enum."""

    def test_all_backends_have_values(self) -> None:
        """All backends have string values."""
        for backend in QuantizationBackend:
            assert isinstance(backend.value, str)

    def test_none_value(self) -> None:
        """NONE has correct value."""
        assert QuantizationBackend.NONE.value == "none"

    def test_bitsandbytes_value(self) -> None:
        """BITSANDBYTES has correct value."""
        assert QuantizationBackend.BITSANDBYTES.value == "bitsandbytes"

    def test_gptq_value(self) -> None:
        """GPTQ has correct value."""
        assert QuantizationBackend.GPTQ.value == "gptq"

    def test_awq_value(self) -> None:
        """AWQ has correct value."""
        assert QuantizationBackend.AWQ.value == "awq"

    def test_ggml_value(self) -> None:
        """GGML has correct value."""
        assert QuantizationBackend.GGML.value == "ggml"

    def test_valid_backends_frozenset(self) -> None:
        """VALID_QUANTIZATION_BACKENDS is a frozenset."""
        assert isinstance(VALID_QUANTIZATION_BACKENDS, frozenset)

    def test_valid_backends_count(self) -> None:
        """VALID_QUANTIZATION_BACKENDS has correct count."""
        assert len(VALID_QUANTIZATION_BACKENDS) == 5


class TestEngineFeature:
    """Tests for EngineFeature enum."""

    def test_all_features_have_values(self) -> None:
        """All features have string values."""
        for feature in EngineFeature:
            assert isinstance(feature.value, str)

    def test_streaming_value(self) -> None:
        """STREAMING has correct value."""
        assert EngineFeature.STREAMING.value == "streaming"

    def test_batching_value(self) -> None:
        """BATCHING has correct value."""
        assert EngineFeature.BATCHING.value == "batching"

    def test_speculative_value(self) -> None:
        """SPECULATIVE has correct value."""
        assert EngineFeature.SPECULATIVE.value == "speculative"

    def test_prefix_caching_value(self) -> None:
        """PREFIX_CACHING has correct value."""
        assert EngineFeature.PREFIX_CACHING.value == "prefix_caching"

    def test_valid_features_frozenset(self) -> None:
        """VALID_ENGINE_FEATURES is a frozenset."""
        assert isinstance(VALID_ENGINE_FEATURES, frozenset)

    def test_valid_features_count(self) -> None:
        """VALID_ENGINE_FEATURES has correct count."""
        assert len(VALID_ENGINE_FEATURES) == 4


class TestVLLMConfig:
    """Tests for VLLMConfig dataclass."""

    def test_create_config(self) -> None:
        """Create vLLM config."""
        config = VLLMConfig(
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=False,
        )
        assert config.tensor_parallel_size == 4
        assert config.gpu_memory_utilization == 0.9

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = VLLMConfig(4, 0.9, 4096, False)
        with pytest.raises(AttributeError):
            config.tensor_parallel_size = 8  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = VLLMConfig(2, 0.85, 8192, True)
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 8192
        assert config.enforce_eager is True


class TestTGIConfig:
    """Tests for TGIConfig dataclass."""

    def test_create_config(self) -> None:
        """Create TGI config."""
        config = TGIConfig(
            max_concurrent_requests=128,
            max_input_length=2048,
            max_total_tokens=4096,
        )
        assert config.max_concurrent_requests == 128
        assert config.max_input_length == 2048

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = TGIConfig(128, 2048, 4096)
        with pytest.raises(AttributeError):
            config.max_concurrent_requests = 256  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = TGIConfig(64, 1024, 2048)
        assert config.max_concurrent_requests == 64
        assert config.max_input_length == 1024
        assert config.max_total_tokens == 2048


class TestLlamaCppConfig:
    """Tests for LlamaCppConfig dataclass."""

    def test_create_config(self) -> None:
        """Create llama.cpp config."""
        config = LlamaCppConfig(
            n_ctx=4096,
            n_batch=512,
            n_threads=8,
            use_mmap=True,
        )
        assert config.n_ctx == 4096
        assert config.n_threads == 8

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LlamaCppConfig(4096, 512, 8, True)
        with pytest.raises(AttributeError):
            config.n_ctx = 8192  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = LlamaCppConfig(2048, 256, 4, False)
        assert config.n_ctx == 2048
        assert config.n_batch == 256
        assert config.n_threads == 4
        assert config.use_mmap is False


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_create_vllm_config(self) -> None:
        """Create engine config with vLLM."""
        vllm = VLLMConfig(4, 0.9, 4096, False)
        config = EngineConfig(
            engine_type=EngineType.VLLM,
            vllm_config=vllm,
            tgi_config=None,
            llamacpp_config=None,
            quantization=QuantizationBackend.NONE,
        )
        assert config.engine_type == EngineType.VLLM
        assert config.vllm_config is not None

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        vllm = VLLMConfig(4, 0.9, 4096, False)
        config = EngineConfig(
            EngineType.VLLM, vllm, None, None, QuantizationBackend.NONE
        )
        with pytest.raises(AttributeError):
            config.engine_type = EngineType.TGI  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        tgi = TGIConfig(128, 2048, 4096)
        config = EngineConfig(EngineType.TGI, None, tgi, None, QuantizationBackend.GPTQ)
        assert config.engine_type == EngineType.TGI
        assert config.vllm_config is None
        assert config.tgi_config is not None
        assert config.llamacpp_config is None
        assert config.quantization == QuantizationBackend.GPTQ


class TestEngineStats:
    """Tests for EngineStats dataclass."""

    def test_create_stats(self) -> None:
        """Create engine stats."""
        stats = EngineStats(
            throughput_tokens_per_sec=1500.0,
            latency_ms=50.0,
            memory_usage_gb=24.0,
        )
        assert stats.throughput_tokens_per_sec == 1500.0
        assert stats.latency_ms == 50.0

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = EngineStats(1500.0, 50.0, 24.0)
        with pytest.raises(AttributeError):
            stats.throughput_tokens_per_sec = 2000.0  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        """All fields are accessible."""
        stats = EngineStats(1000.0, 100.0, 16.0)
        assert stats.throughput_tokens_per_sec == 1000.0
        assert stats.latency_ms == 100.0
        assert stats.memory_usage_gb == 16.0


class TestValidateVLLMConfig:
    """Tests for validate_vllm_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = VLLMConfig(4, 0.9, 4096, False)
        validate_vllm_config(config)

    def test_zero_tensor_parallel_raises(self) -> None:
        """Zero tensor_parallel_size raises ValueError."""
        config = VLLMConfig(0, 0.9, 4096, False)
        with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
            validate_vllm_config(config)

    def test_negative_tensor_parallel_raises(self) -> None:
        """Negative tensor_parallel_size raises ValueError."""
        config = VLLMConfig(-1, 0.9, 4096, False)
        with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
            validate_vllm_config(config)

    def test_zero_memory_utilization_raises(self) -> None:
        """Zero gpu_memory_utilization raises ValueError."""
        config = VLLMConfig(4, 0.0, 4096, False)
        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            validate_vllm_config(config)

    def test_over_one_memory_utilization_raises(self) -> None:
        """Over 1.0 gpu_memory_utilization raises ValueError."""
        config = VLLMConfig(4, 1.5, 4096, False)
        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            validate_vllm_config(config)

    def test_negative_memory_utilization_raises(self) -> None:
        """Negative gpu_memory_utilization raises ValueError."""
        config = VLLMConfig(4, -0.1, 4096, False)
        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            validate_vllm_config(config)

    def test_zero_max_model_len_raises(self) -> None:
        """Zero max_model_len raises ValueError."""
        config = VLLMConfig(4, 0.9, 0, False)
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            validate_vllm_config(config)

    def test_exactly_one_utilization_valid(self) -> None:
        """Exactly 1.0 gpu_memory_utilization is valid."""
        config = VLLMConfig(4, 1.0, 4096, False)
        validate_vllm_config(config)


class TestValidateTGIConfig:
    """Tests for validate_tgi_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = TGIConfig(128, 2048, 4096)
        validate_tgi_config(config)

    def test_zero_concurrent_requests_raises(self) -> None:
        """Zero max_concurrent_requests raises ValueError."""
        config = TGIConfig(0, 2048, 4096)
        with pytest.raises(
            ValueError, match="max_concurrent_requests must be positive"
        ):
            validate_tgi_config(config)

    def test_negative_concurrent_requests_raises(self) -> None:
        """Negative max_concurrent_requests raises ValueError."""
        config = TGIConfig(-1, 2048, 4096)
        with pytest.raises(
            ValueError, match="max_concurrent_requests must be positive"
        ):
            validate_tgi_config(config)

    def test_zero_input_length_raises(self) -> None:
        """Zero max_input_length raises ValueError."""
        config = TGIConfig(128, 0, 4096)
        with pytest.raises(ValueError, match="max_input_length must be positive"):
            validate_tgi_config(config)

    def test_zero_total_tokens_raises(self) -> None:
        """Zero max_total_tokens raises ValueError."""
        config = TGIConfig(128, 2048, 0)
        with pytest.raises(ValueError, match="max_total_tokens must be positive"):
            validate_tgi_config(config)

    def test_total_less_than_input_raises(self) -> None:
        """max_total_tokens < max_input_length raises ValueError."""
        config = TGIConfig(128, 4096, 2048)
        with pytest.raises(
            ValueError, match="max_total_tokens must be >= max_input_length"
        ):
            validate_tgi_config(config)

    def test_equal_input_and_total_valid(self) -> None:
        """Equal max_input_length and max_total_tokens is valid."""
        config = TGIConfig(128, 2048, 2048)
        validate_tgi_config(config)


class TestValidateLlamaCppConfig:
    """Tests for validate_llamacpp_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LlamaCppConfig(4096, 512, 8, True)
        validate_llamacpp_config(config)

    def test_zero_n_ctx_raises(self) -> None:
        """Zero n_ctx raises ValueError."""
        config = LlamaCppConfig(0, 512, 8, True)
        with pytest.raises(ValueError, match="n_ctx must be positive"):
            validate_llamacpp_config(config)

    def test_negative_n_ctx_raises(self) -> None:
        """Negative n_ctx raises ValueError."""
        config = LlamaCppConfig(-1, 512, 8, True)
        with pytest.raises(ValueError, match="n_ctx must be positive"):
            validate_llamacpp_config(config)

    def test_zero_n_batch_raises(self) -> None:
        """Zero n_batch raises ValueError."""
        config = LlamaCppConfig(4096, 0, 8, True)
        with pytest.raises(ValueError, match="n_batch must be positive"):
            validate_llamacpp_config(config)

    def test_zero_n_threads_raises(self) -> None:
        """Zero n_threads raises ValueError."""
        config = LlamaCppConfig(4096, 512, 0, True)
        with pytest.raises(ValueError, match="n_threads must be positive"):
            validate_llamacpp_config(config)

    def test_batch_greater_than_ctx_raises(self) -> None:
        """n_batch > n_ctx raises ValueError."""
        config = LlamaCppConfig(512, 1024, 8, True)
        with pytest.raises(ValueError, match="n_batch must be <= n_ctx"):
            validate_llamacpp_config(config)

    def test_equal_batch_and_ctx_valid(self) -> None:
        """n_batch == n_ctx is valid."""
        config = LlamaCppConfig(512, 512, 8, True)
        validate_llamacpp_config(config)


class TestValidateEngineConfig:
    """Tests for validate_engine_config function."""

    def test_valid_vllm_config(self) -> None:
        """Valid vLLM config passes validation."""
        vllm = VLLMConfig(4, 0.9, 4096, False)
        config = EngineConfig(
            EngineType.VLLM, vllm, None, None, QuantizationBackend.NONE
        )
        validate_engine_config(config)

    def test_valid_tgi_config(self) -> None:
        """Valid TGI config passes validation."""
        tgi = TGIConfig(128, 2048, 4096)
        config = EngineConfig(EngineType.TGI, None, tgi, None, QuantizationBackend.NONE)
        validate_engine_config(config)

    def test_valid_llamacpp_config(self) -> None:
        """Valid llama.cpp config passes validation."""
        llamacpp = LlamaCppConfig(4096, 512, 8, True)
        config = EngineConfig(
            EngineType.LLAMACPP, None, None, llamacpp, QuantizationBackend.GGML
        )
        validate_engine_config(config)

    def test_missing_vllm_config_raises(self) -> None:
        """Missing vllm_config for VLLM raises ValueError."""
        config = EngineConfig(
            EngineType.VLLM, None, None, None, QuantizationBackend.NONE
        )
        with pytest.raises(ValueError, match="vllm_config is required for VLLM"):
            validate_engine_config(config)

    def test_missing_tgi_config_raises(self) -> None:
        """Missing tgi_config for TGI raises ValueError."""
        config = EngineConfig(
            EngineType.TGI, None, None, None, QuantizationBackend.NONE
        )
        with pytest.raises(ValueError, match="tgi_config is required for TGI"):
            validate_engine_config(config)

    def test_missing_llamacpp_config_raises(self) -> None:
        """Missing llamacpp_config for LLAMACPP raises ValueError."""
        config = EngineConfig(
            EngineType.LLAMACPP, None, None, None, QuantizationBackend.NONE
        )
        with pytest.raises(
            ValueError, match="llamacpp_config is required for LLAMACPP"
        ):
            validate_engine_config(config)

    def test_invalid_llamacpp_quantization_raises(self) -> None:
        """Invalid quantization for LLAMACPP raises ValueError."""
        llamacpp = LlamaCppConfig(4096, 512, 8, True)
        config = EngineConfig(
            EngineType.LLAMACPP, None, None, llamacpp, QuantizationBackend.GPTQ
        )
        with pytest.raises(ValueError, match="LLAMACPP only supports NONE or GGML"):
            validate_engine_config(config)

    def test_invalid_nested_vllm_config_raises(self) -> None:
        """Invalid nested vllm_config raises ValueError."""
        bad_vllm = VLLMConfig(0, 0.9, 4096, False)
        config = EngineConfig(
            EngineType.VLLM, bad_vllm, None, None, QuantizationBackend.NONE
        )
        with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
            validate_engine_config(config)


class TestValidateEngineStats:
    """Tests for validate_engine_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = EngineStats(1500.0, 50.0, 24.0)
        validate_engine_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        stats = EngineStats(0.0, 0.0, 0.0)
        validate_engine_stats(stats)

    def test_negative_throughput_raises(self) -> None:
        """Negative throughput raises ValueError."""
        stats = EngineStats(-1.0, 50.0, 24.0)
        with pytest.raises(
            ValueError, match="throughput_tokens_per_sec cannot be negative"
        ):
            validate_engine_stats(stats)

    def test_negative_latency_raises(self) -> None:
        """Negative latency raises ValueError."""
        stats = EngineStats(1500.0, -1.0, 24.0)
        with pytest.raises(ValueError, match="latency_ms cannot be negative"):
            validate_engine_stats(stats)

    def test_negative_memory_raises(self) -> None:
        """Negative memory raises ValueError."""
        stats = EngineStats(1500.0, 50.0, -1.0)
        with pytest.raises(ValueError, match="memory_usage_gb cannot be negative"):
            validate_engine_stats(stats)


class TestCreateVLLMConfig:
    """Tests for create_vllm_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_vllm_config()
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 4096
        assert config.enforce_eager is False

    def test_custom_tensor_parallel(self) -> None:
        """Create config with custom tensor_parallel_size."""
        config = create_vllm_config(tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_custom_memory_utilization(self) -> None:
        """Create config with custom gpu_memory_utilization."""
        config = create_vllm_config(gpu_memory_utilization=0.8)
        assert config.gpu_memory_utilization == 0.8

    def test_custom_max_model_len(self) -> None:
        """Create config with custom max_model_len."""
        config = create_vllm_config(max_model_len=8192)
        assert config.max_model_len == 8192

    def test_custom_enforce_eager(self) -> None:
        """Create config with custom enforce_eager."""
        config = create_vllm_config(enforce_eager=True)
        assert config.enforce_eager is True

    def test_zero_tensor_parallel_raises(self) -> None:
        """Zero tensor_parallel_size raises ValueError."""
        with pytest.raises(ValueError, match="tensor_parallel_size must be positive"):
            create_vllm_config(tensor_parallel_size=0)


class TestCreateTGIConfig:
    """Tests for create_tgi_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_tgi_config()
        assert config.max_concurrent_requests == 128
        assert config.max_input_length == 2048
        assert config.max_total_tokens == 4096

    def test_custom_concurrent_requests(self) -> None:
        """Create config with custom max_concurrent_requests."""
        config = create_tgi_config(max_concurrent_requests=256)
        assert config.max_concurrent_requests == 256

    def test_custom_input_length(self) -> None:
        """Create config with custom max_input_length."""
        config = create_tgi_config(max_input_length=4096)
        assert config.max_input_length == 4096

    def test_custom_total_tokens(self) -> None:
        """Create config with custom max_total_tokens."""
        config = create_tgi_config(max_total_tokens=8192)
        assert config.max_total_tokens == 8192

    def test_zero_concurrent_requests_raises(self) -> None:
        """Zero max_concurrent_requests raises ValueError."""
        with pytest.raises(
            ValueError, match="max_concurrent_requests must be positive"
        ):
            create_tgi_config(max_concurrent_requests=0)


class TestCreateLlamaCppConfig:
    """Tests for create_llamacpp_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_llamacpp_config()
        assert config.n_ctx == 4096
        assert config.n_batch == 512
        assert config.n_threads == 8
        assert config.use_mmap is True

    def test_custom_n_ctx(self) -> None:
        """Create config with custom n_ctx."""
        config = create_llamacpp_config(n_ctx=8192)
        assert config.n_ctx == 8192

    def test_custom_n_batch(self) -> None:
        """Create config with custom n_batch."""
        config = create_llamacpp_config(n_batch=256)
        assert config.n_batch == 256

    def test_custom_n_threads(self) -> None:
        """Create config with custom n_threads."""
        config = create_llamacpp_config(n_threads=16)
        assert config.n_threads == 16

    def test_custom_use_mmap(self) -> None:
        """Create config with custom use_mmap."""
        config = create_llamacpp_config(use_mmap=False)
        assert config.use_mmap is False

    def test_zero_n_ctx_raises(self) -> None:
        """Zero n_ctx raises ValueError."""
        with pytest.raises(ValueError, match="n_ctx must be positive"):
            create_llamacpp_config(n_ctx=0)


class TestCreateEngineConfig:
    """Tests for create_engine_config function."""

    def test_default_config(self) -> None:
        """Create default config (vLLM)."""
        config = create_engine_config()
        assert config.engine_type == EngineType.VLLM
        assert config.vllm_config is not None
        assert config.quantization == QuantizationBackend.NONE

    def test_vllm_engine(self) -> None:
        """Create vLLM engine config."""
        config = create_engine_config(engine_type="vllm")
        assert config.engine_type == EngineType.VLLM
        assert config.vllm_config is not None

    def test_tgi_engine(self) -> None:
        """Create TGI engine config."""
        config = create_engine_config(engine_type="tgi")
        assert config.engine_type == EngineType.TGI
        assert config.tgi_config is not None

    def test_llamacpp_engine(self) -> None:
        """Create llama.cpp engine config."""
        config = create_engine_config(engine_type="llamacpp")
        assert config.engine_type == EngineType.LLAMACPP
        assert config.llamacpp_config is not None

    def test_custom_vllm_config(self) -> None:
        """Create config with custom vLLM config."""
        vllm = create_vllm_config(tensor_parallel_size=4)
        config = create_engine_config(engine_type="vllm", vllm_config=vllm)
        assert config.vllm_config.tensor_parallel_size == 4

    def test_custom_quantization(self) -> None:
        """Create config with custom quantization."""
        config = create_engine_config(quantization="gptq")
        assert config.quantization == QuantizationBackend.GPTQ

    def test_invalid_engine_type_raises(self) -> None:
        """Invalid engine_type raises ValueError."""
        with pytest.raises(ValueError, match="engine_type must be one of"):
            create_engine_config(engine_type="invalid")  # type: ignore[arg-type]

    def test_invalid_quantization_raises(self) -> None:
        """Invalid quantization raises ValueError."""
        with pytest.raises(ValueError, match="quantization must be one of"):
            create_engine_config(quantization="invalid")  # type: ignore[arg-type]


class TestCreateEngineStats:
    """Tests for create_engine_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_engine_stats()
        assert stats.throughput_tokens_per_sec == 0.0
        assert stats.latency_ms == 0.0
        assert stats.memory_usage_gb == 0.0

    def test_custom_throughput(self) -> None:
        """Create stats with custom throughput."""
        stats = create_engine_stats(throughput_tokens_per_sec=1500.0)
        assert stats.throughput_tokens_per_sec == 1500.0

    def test_custom_latency(self) -> None:
        """Create stats with custom latency."""
        stats = create_engine_stats(latency_ms=50.0)
        assert stats.latency_ms == 50.0

    def test_custom_memory(self) -> None:
        """Create stats with custom memory."""
        stats = create_engine_stats(memory_usage_gb=24.0)
        assert stats.memory_usage_gb == 24.0

    def test_negative_throughput_raises(self) -> None:
        """Negative throughput raises ValueError."""
        with pytest.raises(
            ValueError, match="throughput_tokens_per_sec cannot be negative"
        ):
            create_engine_stats(throughput_tokens_per_sec=-1.0)


class TestListEngineTypes:
    """Tests for list_engine_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        engines = list_engine_types()
        assert engines == sorted(engines)

    def test_contains_vllm(self) -> None:
        """Contains vllm."""
        engines = list_engine_types()
        assert "vllm" in engines

    def test_contains_tgi(self) -> None:
        """Contains tgi."""
        engines = list_engine_types()
        assert "tgi" in engines

    def test_contains_all_types(self) -> None:
        """Contains all engine types."""
        engines = list_engine_types()
        assert len(engines) == len(VALID_ENGINE_TYPES)


class TestListQuantizationBackends:
    """Tests for list_quantization_backends function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        backends = list_quantization_backends()
        assert backends == sorted(backends)

    def test_contains_none(self) -> None:
        """Contains none."""
        backends = list_quantization_backends()
        assert "none" in backends

    def test_contains_gptq(self) -> None:
        """Contains gptq."""
        backends = list_quantization_backends()
        assert "gptq" in backends

    def test_contains_all_backends(self) -> None:
        """Contains all backends."""
        backends = list_quantization_backends()
        assert len(backends) == len(VALID_QUANTIZATION_BACKENDS)


class TestListEngineFeatures:
    """Tests for list_engine_features function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        features = list_engine_features()
        assert features == sorted(features)

    def test_contains_streaming(self) -> None:
        """Contains streaming."""
        features = list_engine_features()
        assert "streaming" in features

    def test_contains_batching(self) -> None:
        """Contains batching."""
        features = list_engine_features()
        assert "batching" in features

    def test_contains_all_features(self) -> None:
        """Contains all features."""
        features = list_engine_features()
        assert len(features) == len(VALID_ENGINE_FEATURES)


class TestGetEngineType:
    """Tests for get_engine_type function."""

    def test_get_vllm(self) -> None:
        """Get vllm type."""
        engine = get_engine_type("vllm")
        assert engine == EngineType.VLLM

    def test_get_tgi(self) -> None:
        """Get tgi type."""
        engine = get_engine_type("tgi")
        assert engine == EngineType.TGI

    def test_get_llamacpp(self) -> None:
        """Get llamacpp type."""
        engine = get_engine_type("llamacpp")
        assert engine == EngineType.LLAMACPP

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown engine type"):
            get_engine_type("invalid")


class TestGetQuantizationBackend:
    """Tests for get_quantization_backend function."""

    def test_get_none(self) -> None:
        """Get none backend."""
        backend = get_quantization_backend("none")
        assert backend == QuantizationBackend.NONE

    def test_get_gptq(self) -> None:
        """Get gptq backend."""
        backend = get_quantization_backend("gptq")
        assert backend == QuantizationBackend.GPTQ

    def test_get_awq(self) -> None:
        """Get awq backend."""
        backend = get_quantization_backend("awq")
        assert backend == QuantizationBackend.AWQ

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization backend"):
            get_quantization_backend("invalid")


class TestGetEngineFeature:
    """Tests for get_engine_feature function."""

    def test_get_streaming(self) -> None:
        """Get streaming feature."""
        feature = get_engine_feature("streaming")
        assert feature == EngineFeature.STREAMING

    def test_get_batching(self) -> None:
        """Get batching feature."""
        feature = get_engine_feature("batching")
        assert feature == EngineFeature.BATCHING

    def test_get_speculative(self) -> None:
        """Get speculative feature."""
        feature = get_engine_feature("speculative")
        assert feature == EngineFeature.SPECULATIVE

    def test_invalid_feature_raises(self) -> None:
        """Invalid feature raises ValueError."""
        with pytest.raises(ValueError, match="Unknown engine feature"):
            get_engine_feature("invalid")


class TestEstimateEngineThroughput:
    """Tests for estimate_engine_throughput function."""

    def test_basic_estimation(self) -> None:
        """Basic throughput estimation."""
        throughput = estimate_engine_throughput(
            engine_type="vllm",
            model_params_billions=7.0,
            batch_size=16,
        )
        assert throughput > 0

    def test_more_gpus_higher_throughput(self) -> None:
        """More GPUs give higher throughput."""
        t1 = estimate_engine_throughput("vllm", 7.0, 16, gpu_count=1)
        t4 = estimate_engine_throughput("vllm", 7.0, 16, gpu_count=4)
        assert t4 > t1

    def test_larger_batch_higher_throughput(self) -> None:
        """Larger batch gives higher throughput."""
        t8 = estimate_engine_throughput("vllm", 7.0, 8)
        t32 = estimate_engine_throughput("vllm", 7.0, 32)
        assert t32 > t8

    def test_larger_model_lower_throughput(self) -> None:
        """Larger model gives lower throughput."""
        t7b = estimate_engine_throughput("vllm", 7.0, 16)
        t70b = estimate_engine_throughput("vllm", 70.0, 16)
        assert t7b > t70b

    def test_quantization_increases_throughput(self) -> None:
        """Quantization increases throughput."""
        t_none = estimate_engine_throughput("vllm", 7.0, 16, quantization="none")
        t_awq = estimate_engine_throughput("vllm", 7.0, 16, quantization="awq")
        assert t_awq > t_none

    def test_invalid_engine_type_raises(self) -> None:
        """Invalid engine_type raises ValueError."""
        with pytest.raises(ValueError, match="engine_type must be one of"):
            estimate_engine_throughput("invalid", 7.0, 16)

    def test_zero_model_params_raises(self) -> None:
        """Zero model_params_billions raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            estimate_engine_throughput("vllm", 0.0, 16)

    def test_negative_model_params_raises(self) -> None:
        """Negative model_params_billions raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            estimate_engine_throughput("vllm", -7.0, 16)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_engine_throughput("vllm", 7.0, 0)

    def test_zero_gpu_count_raises(self) -> None:
        """Zero gpu_count raises ValueError."""
        with pytest.raises(ValueError, match="gpu_count must be positive"):
            estimate_engine_throughput("vllm", 7.0, 16, gpu_count=0)

    def test_invalid_quantization_raises(self) -> None:
        """Invalid quantization raises ValueError."""
        with pytest.raises(ValueError, match="quantization must be one of"):
            estimate_engine_throughput("vllm", 7.0, 16, quantization="invalid")


class TestCompareEnginePerformance:
    """Tests for compare_engine_performance function."""

    def test_basic_comparison(self) -> None:
        """Basic performance comparison."""
        result = compare_engine_performance("vllm", "tgi", 7.0, 16)
        assert "engine_a_throughput" in result
        assert "engine_b_throughput" in result
        assert "ratio_a_to_b" in result

    def test_vllm_faster_than_tgi(self) -> None:
        """VLLM is faster than TGI."""
        result = compare_engine_performance("vllm", "tgi", 7.0, 16)
        assert result["engine_a_throughput"] > result["engine_b_throughput"]
        assert result["ratio_a_to_b"] > 1.0

    def test_same_engine_equal_performance(self) -> None:
        """Same engine has equal performance."""
        result = compare_engine_performance("vllm", "vllm", 7.0, 16)
        assert result["ratio_a_to_b"] == 1.0

    def test_invalid_engine_a_raises(self) -> None:
        """Invalid engine_a raises ValueError."""
        with pytest.raises(ValueError, match="engine_type must be one of"):
            compare_engine_performance("invalid", "tgi", 7.0, 16)

    def test_invalid_engine_b_raises(self) -> None:
        """Invalid engine_b raises ValueError."""
        with pytest.raises(ValueError, match="engine_type must be one of"):
            compare_engine_performance("vllm", "invalid", 7.0, 16)


class TestCheckEngineCompatibility:
    """Tests for check_engine_compatibility function."""

    def test_vllm_transformers_compatible(self) -> None:
        """VLLM is compatible with transformers."""
        compatible, msg = check_engine_compatibility("vllm", "transformers")
        assert compatible is True
        assert "compatible" in msg.lower()

    def test_vllm_gguf_incompatible(self) -> None:
        """VLLM is incompatible with gguf."""
        compatible, msg = check_engine_compatibility("vllm", "gguf")
        assert compatible is False
        assert "requires" in msg.lower()

    def test_llamacpp_gguf_compatible(self) -> None:
        """llama.cpp is compatible with gguf."""
        compatible, _msg = check_engine_compatibility("llamacpp", "gguf")
        assert compatible is True

    def test_llamacpp_onnx_incompatible(self) -> None:
        """llama.cpp is incompatible with onnx."""
        compatible, _msg = check_engine_compatibility("llamacpp", "onnx")
        assert compatible is False

    def test_onnxruntime_onnx_compatible(self) -> None:
        """ONNX Runtime is compatible with onnx."""
        compatible, _msg = check_engine_compatibility("onnxruntime", "onnx")
        assert compatible is True

    def test_vllm_gptq_compatible(self) -> None:
        """VLLM is compatible with gptq quantization."""
        compatible, _msg = check_engine_compatibility("vllm", "transformers", "gptq")
        assert compatible is True

    def test_llamacpp_gptq_incompatible(self) -> None:
        """llama.cpp is incompatible with gptq quantization."""
        compatible, _msg = check_engine_compatibility("llamacpp", "gguf", "gptq")
        assert compatible is False

    def test_llamacpp_ggml_compatible(self) -> None:
        """llama.cpp is compatible with ggml quantization."""
        compatible, _msg = check_engine_compatibility("llamacpp", "gguf", "ggml")
        assert compatible is True

    def test_invalid_engine_type_raises(self) -> None:
        """Invalid engine_type raises ValueError."""
        with pytest.raises(ValueError, match="engine_type must be one of"):
            check_engine_compatibility("invalid", "transformers")

    def test_invalid_model_format_raises(self) -> None:
        """Invalid model_format raises ValueError."""
        with pytest.raises(ValueError, match="model_format must be one of"):
            check_engine_compatibility("vllm", "invalid")

    def test_invalid_quantization_raises(self) -> None:
        """Invalid quantization raises ValueError."""
        with pytest.raises(ValueError, match="quantization must be one of"):
            check_engine_compatibility("vllm", "transformers", "invalid")


class TestGetEngineFeatures:
    """Tests for get_engine_features function."""

    def test_vllm_features(self) -> None:
        """VLLM has all features."""
        features = get_engine_features("vllm")
        assert "streaming" in features
        assert "batching" in features
        assert "speculative" in features
        assert "prefix_caching" in features

    def test_tgi_features(self) -> None:
        """TGI has streaming, batching, prefix_caching."""
        features = get_engine_features("tgi")
        assert "streaming" in features
        assert "batching" in features
        assert "prefix_caching" in features
        assert "speculative" not in features

    def test_llamacpp_features(self) -> None:
        """llama.cpp has only streaming."""
        features = get_engine_features("llamacpp")
        assert "streaming" in features
        assert "batching" not in features

    def test_returns_frozenset(self) -> None:
        """Returns a frozenset."""
        features = get_engine_features("vllm")
        assert isinstance(features, frozenset)

    def test_invalid_engine_type_raises(self) -> None:
        """Invalid engine_type raises ValueError."""
        with pytest.raises(ValueError, match="engine_type must be one of"):
            get_engine_features("invalid")


class TestFormatEngineStats:
    """Tests for format_engine_stats function."""

    def test_basic_formatting(self) -> None:
        """Basic stats formatting."""
        stats = EngineStats(1500.0, 50.0, 24.0)
        formatted = format_engine_stats(stats)
        assert "Throughput: 1500.00 tokens/sec" in formatted
        assert "Latency: 50.00 ms" in formatted
        assert "Memory: 24.00 GB" in formatted

    def test_zero_values(self) -> None:
        """Zero values format correctly."""
        stats = EngineStats(0.0, 0.0, 0.0)
        formatted = format_engine_stats(stats)
        assert "Throughput: 0.00 tokens/sec" in formatted
        assert "Latency: 0.00 ms" in formatted
        assert "Memory: 0.00 GB" in formatted

    def test_multiline_output(self) -> None:
        """Output contains multiple lines."""
        stats = EngineStats(1500.0, 50.0, 24.0)
        formatted = format_engine_stats(stats)
        lines = formatted.split("\n")
        assert len(lines) == 3


class TestGetRecommendedEngineConfig:
    """Tests for get_recommended_engine_config function."""

    def test_small_cpu(self) -> None:
        """Small model on CPU uses llama.cpp."""
        config = get_recommended_engine_config("small", "cpu")
        assert config.engine_type == EngineType.LLAMACPP
        assert config.quantization == QuantizationBackend.GGML

    def test_large_cpu(self) -> None:
        """Large model on CPU uses llama.cpp with GGML."""
        config = get_recommended_engine_config("large", "cpu")
        assert config.engine_type == EngineType.LLAMACPP
        assert config.quantization == QuantizationBackend.GGML

    def test_small_gpu_consumer(self) -> None:
        """Small model on consumer GPU uses vLLM without quantization."""
        config = get_recommended_engine_config("small", "gpu_consumer")
        assert config.engine_type == EngineType.VLLM
        assert config.quantization == QuantizationBackend.NONE

    def test_large_gpu_consumer(self) -> None:
        """Large model on consumer GPU uses vLLM with GPTQ."""
        config = get_recommended_engine_config("large", "gpu_consumer")
        assert config.engine_type == EngineType.VLLM
        assert config.quantization == QuantizationBackend.GPTQ

    def test_small_gpu_datacenter(self) -> None:
        """Small model on datacenter GPU uses vLLM without quantization."""
        config = get_recommended_engine_config("small", "gpu_datacenter")
        assert config.engine_type == EngineType.VLLM
        assert config.quantization == QuantizationBackend.NONE

    def test_large_gpu_datacenter(self) -> None:
        """Large model on datacenter GPU uses vLLM with AWQ."""
        config = get_recommended_engine_config("large", "gpu_datacenter")
        assert config.engine_type == EngineType.VLLM
        assert config.quantization == QuantizationBackend.AWQ
        assert config.vllm_config.tensor_parallel_size == 2

    def test_xlarge_gpu_datacenter(self) -> None:
        """XLarge model on datacenter GPU uses 4-way tensor parallelism."""
        config = get_recommended_engine_config("xlarge", "gpu_datacenter")
        assert config.engine_type == EngineType.VLLM
        assert config.quantization == QuantizationBackend.AWQ
        assert config.vllm_config.tensor_parallel_size == 4

    def test_default_is_gpu_datacenter(self) -> None:
        """Default hardware is gpu_datacenter."""
        config = get_recommended_engine_config("medium")
        assert config.engine_type == EngineType.VLLM

    def test_invalid_model_size_raises(self) -> None:
        """Invalid model_size raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model size"):
            get_recommended_engine_config("invalid")

    def test_invalid_hardware_raises(self) -> None:
        """Invalid hardware raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hardware type"):
            get_recommended_engine_config("small", "invalid")  # type: ignore[arg-type]


class TestAllEngineTypes:
    """Test all engine types can be retrieved."""

    @pytest.mark.parametrize("engine_type", list(VALID_ENGINE_TYPES))
    def test_get_engine_type(self, engine_type: str) -> None:
        """Engine type can be retrieved."""
        result = get_engine_type(engine_type)
        assert result.value == engine_type


class TestAllQuantizationBackends:
    """Test all quantization backends can be retrieved."""

    @pytest.mark.parametrize("backend", list(VALID_QUANTIZATION_BACKENDS))
    def test_get_quantization_backend(self, backend: str) -> None:
        """Quantization backend can be retrieved."""
        result = get_quantization_backend(backend)
        assert result.value == backend


class TestAllEngineFeatures:
    """Test all engine features can be retrieved."""

    @pytest.mark.parametrize("feature", list(VALID_ENGINE_FEATURES))
    def test_get_engine_feature(self, feature: str) -> None:
        """Engine feature can be retrieved."""
        result = get_engine_feature(feature)
        assert result.value == feature
