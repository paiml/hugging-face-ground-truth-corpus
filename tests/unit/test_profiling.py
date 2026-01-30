"""Tests for model profiling functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.profiling import (
    VALID_BOTTLENECK_TYPES,
    VALID_PROFILE_METRICS,
    VALID_PROFILING_LEVELS,
    BottleneckType,
    LatencyBreakdown,
    MemoryBreakdown,
    ProfileMetric,
    ProfilingConfig,
    ProfilingLevel,
    ProfilingResult,
    calculate_flops,
    compare_profiles,
    create_latency_breakdown,
    create_memory_breakdown,
    create_profiling_config,
    create_profiling_result,
    estimate_memory_footprint,
    format_profiling_result,
    get_bottleneck_type,
    get_profile_metric,
    get_profiling_level,
    get_recommended_profiling_config,
    identify_bottlenecks,
    list_bottleneck_types,
    list_profile_metrics,
    list_profiling_levels,
    validate_bottleneck_type,
    validate_latency_breakdown,
    validate_memory_breakdown,
    validate_profile_metric,
    validate_profiling_config,
    validate_profiling_level,
    validate_profiling_result,
)


class TestProfileMetricEnum:
    """Tests for ProfileMetric enum."""

    def test_latency_value(self) -> None:
        """Test LATENCY value."""
        assert ProfileMetric.LATENCY.value == "latency"

    def test_throughput_value(self) -> None:
        """Test THROUGHPUT value."""
        assert ProfileMetric.THROUGHPUT.value == "throughput"

    def test_memory_value(self) -> None:
        """Test MEMORY value."""
        assert ProfileMetric.MEMORY.value == "memory"

    def test_flops_value(self) -> None:
        """Test FLOPS value."""
        assert ProfileMetric.FLOPS.value == "flops"

    def test_parameters_value(self) -> None:
        """Test PARAMETERS value."""
        assert ProfileMetric.PARAMETERS.value == "parameters"

    def test_valid_profile_metrics_frozenset(self) -> None:
        """Test that VALID_PROFILE_METRICS is a frozenset."""
        assert isinstance(VALID_PROFILE_METRICS, frozenset)
        assert len(VALID_PROFILE_METRICS) == 5


class TestBottleneckTypeEnum:
    """Tests for BottleneckType enum."""

    def test_compute_value(self) -> None:
        """Test COMPUTE value."""
        assert BottleneckType.COMPUTE.value == "compute"

    def test_memory_value(self) -> None:
        """Test MEMORY value."""
        assert BottleneckType.MEMORY.value == "memory"

    def test_io_value(self) -> None:
        """Test IO value."""
        assert BottleneckType.IO.value == "io"

    def test_communication_value(self) -> None:
        """Test COMMUNICATION value."""
        assert BottleneckType.COMMUNICATION.value == "communication"

    def test_valid_bottleneck_types_frozenset(self) -> None:
        """Test that VALID_BOTTLENECK_TYPES is a frozenset."""
        assert isinstance(VALID_BOTTLENECK_TYPES, frozenset)
        assert len(VALID_BOTTLENECK_TYPES) == 4


class TestProfilingLevelEnum:
    """Tests for ProfilingLevel enum."""

    def test_basic_value(self) -> None:
        """Test BASIC value."""
        assert ProfilingLevel.BASIC.value == "basic"

    def test_detailed_value(self) -> None:
        """Test DETAILED value."""
        assert ProfilingLevel.DETAILED.value == "detailed"

    def test_trace_value(self) -> None:
        """Test TRACE value."""
        assert ProfilingLevel.TRACE.value == "trace"

    def test_valid_profiling_levels_frozenset(self) -> None:
        """Test that VALID_PROFILING_LEVELS is a frozenset."""
        assert isinstance(VALID_PROFILING_LEVELS, frozenset)
        assert len(VALID_PROFILING_LEVELS) == 3


class TestProfilingConfig:
    """Tests for ProfilingConfig dataclass."""

    def test_creation_with_metrics(self) -> None:
        """Test creating ProfilingConfig with metrics."""
        config = ProfilingConfig(metrics=(ProfileMetric.LATENCY,))
        assert ProfileMetric.LATENCY in config.metrics

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ProfilingConfig(metrics=(ProfileMetric.LATENCY,))
        assert config.level == ProfilingLevel.BASIC
        assert config.warmup_runs == 3
        assert config.num_runs == 10
        assert config.include_backward is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ProfilingConfig(
            metrics=(ProfileMetric.LATENCY, ProfileMetric.MEMORY),
            level=ProfilingLevel.DETAILED,
            warmup_runs=5,
            num_runs=20,
            include_backward=True,
        )
        assert len(config.metrics) == 2
        assert config.level == ProfilingLevel.DETAILED
        assert config.warmup_runs == 5
        assert config.num_runs == 20
        assert config.include_backward is True

    def test_frozen(self) -> None:
        """Test that ProfilingConfig is immutable."""
        config = ProfilingConfig(metrics=(ProfileMetric.LATENCY,))
        with pytest.raises(AttributeError):
            config.level = ProfilingLevel.TRACE  # type: ignore[misc]


class TestLatencyBreakdown:
    """Tests for LatencyBreakdown dataclass."""

    def test_creation(self) -> None:
        """Test creating LatencyBreakdown instance."""
        breakdown = LatencyBreakdown(
            forward_ms=10.5,
            backward_ms=15.2,
            data_loading_ms=2.1,
            optimizer_ms=1.3,
        )
        assert breakdown.forward_ms == pytest.approx(10.5)
        assert breakdown.backward_ms == pytest.approx(15.2)
        assert breakdown.data_loading_ms == pytest.approx(2.1)
        assert breakdown.optimizer_ms == pytest.approx(1.3)

    def test_total_ms_property(self) -> None:
        """Test total_ms property calculation."""
        breakdown = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        assert breakdown.total_ms == pytest.approx(28.0)

    def test_total_ms_with_zeros(self) -> None:
        """Test total_ms with zero values."""
        breakdown = LatencyBreakdown(10.0, 0.0, 0.0, 0.0)
        assert breakdown.total_ms == pytest.approx(10.0)

    def test_frozen(self) -> None:
        """Test that LatencyBreakdown is immutable."""
        breakdown = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        with pytest.raises(AttributeError):
            breakdown.forward_ms = 20.0  # type: ignore[misc]


class TestMemoryBreakdown:
    """Tests for MemoryBreakdown dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryBreakdown instance."""
        breakdown = MemoryBreakdown(
            parameters_mb=500.0,
            gradients_mb=500.0,
            activations_mb=1000.0,
            optimizer_state_mb=1000.0,
        )
        assert breakdown.parameters_mb == pytest.approx(500.0)
        assert breakdown.gradients_mb == pytest.approx(500.0)

    def test_total_mb_property(self) -> None:
        """Test total_mb property calculation."""
        breakdown = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        assert breakdown.total_mb == pytest.approx(3000.0)

    def test_total_mb_with_zeros(self) -> None:
        """Test total_mb with zero values."""
        breakdown = MemoryBreakdown(500.0, 0.0, 0.0, 0.0)
        assert breakdown.total_mb == pytest.approx(500.0)

    def test_frozen(self) -> None:
        """Test that MemoryBreakdown is immutable."""
        breakdown = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        with pytest.raises(AttributeError):
            breakdown.parameters_mb = 1000.0  # type: ignore[misc]


class TestProfilingResult:
    """Tests for ProfilingResult dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating ProfilingResult with minimal data."""
        result = ProfilingResult(
            latency=None,
            memory=None,
            throughput_tokens_per_sec=1000.0,
            bottlenecks=(),
        )
        assert result.latency is None
        assert result.memory is None
        assert result.throughput_tokens_per_sec == pytest.approx(1000.0)
        assert result.bottlenecks == ()

    def test_creation_full(self) -> None:
        """Test creating ProfilingResult with all data."""
        latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        memory = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        result = ProfilingResult(
            latency=latency,
            memory=memory,
            throughput_tokens_per_sec=1500.0,
            bottlenecks=(BottleneckType.COMPUTE,),
        )
        assert result.latency is not None
        assert result.memory is not None
        assert BottleneckType.COMPUTE in result.bottlenecks

    def test_frozen(self) -> None:
        """Test that ProfilingResult is immutable."""
        result = ProfilingResult(None, None, 1000.0, ())
        with pytest.raises(AttributeError):
            result.throughput_tokens_per_sec = 2000.0  # type: ignore[misc]


class TestValidateProfilingConfig:
    """Tests for validate_profiling_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ProfilingConfig(metrics=(ProfileMetric.LATENCY,))
        validate_profiling_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_profiling_config(None)  # type: ignore[arg-type]

    def test_empty_metrics_raises_error(self) -> None:
        """Test that empty metrics raises ValueError."""
        config = ProfilingConfig(metrics=())
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            validate_profiling_config(config)

    def test_negative_warmup_raises_error(self) -> None:
        """Test that negative warmup_runs raises ValueError."""
        config = ProfilingConfig(metrics=(ProfileMetric.LATENCY,), warmup_runs=-1)
        with pytest.raises(ValueError, match="warmup_runs cannot be negative"):
            validate_profiling_config(config)

    def test_zero_num_runs_raises_error(self) -> None:
        """Test that zero num_runs raises ValueError."""
        config = ProfilingConfig(metrics=(ProfileMetric.LATENCY,), num_runs=0)
        with pytest.raises(ValueError, match="num_runs must be positive"):
            validate_profiling_config(config)


class TestValidateLatencyBreakdown:
    """Tests for validate_latency_breakdown function."""

    def test_valid_breakdown(self) -> None:
        """Test validation of valid breakdown."""
        breakdown = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        validate_latency_breakdown(breakdown)  # Should not raise

    def test_none_breakdown_raises_error(self) -> None:
        """Test that None breakdown raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_latency_breakdown(None)  # type: ignore[arg-type]

    def test_negative_forward_ms_raises_error(self) -> None:
        """Test that negative forward_ms raises ValueError."""
        breakdown = LatencyBreakdown(-1.0, 15.0, 2.0, 1.0)
        with pytest.raises(ValueError, match="forward_ms cannot be negative"):
            validate_latency_breakdown(breakdown)

    def test_negative_backward_ms_raises_error(self) -> None:
        """Test that negative backward_ms raises ValueError."""
        breakdown = LatencyBreakdown(10.0, -1.0, 2.0, 1.0)
        with pytest.raises(ValueError, match="backward_ms cannot be negative"):
            validate_latency_breakdown(breakdown)

    def test_negative_data_loading_ms_raises_error(self) -> None:
        """Test that negative data_loading_ms raises ValueError."""
        breakdown = LatencyBreakdown(10.0, 15.0, -1.0, 1.0)
        with pytest.raises(ValueError, match="data_loading_ms cannot be negative"):
            validate_latency_breakdown(breakdown)

    def test_negative_optimizer_ms_raises_error(self) -> None:
        """Test that negative optimizer_ms raises ValueError."""
        breakdown = LatencyBreakdown(10.0, 15.0, 2.0, -1.0)
        with pytest.raises(ValueError, match="optimizer_ms cannot be negative"):
            validate_latency_breakdown(breakdown)


class TestValidateMemoryBreakdown:
    """Tests for validate_memory_breakdown function."""

    def test_valid_breakdown(self) -> None:
        """Test validation of valid breakdown."""
        breakdown = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        validate_memory_breakdown(breakdown)  # Should not raise

    def test_none_breakdown_raises_error(self) -> None:
        """Test that None breakdown raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_memory_breakdown(None)  # type: ignore[arg-type]

    def test_negative_parameters_mb_raises_error(self) -> None:
        """Test that negative parameters_mb raises ValueError."""
        breakdown = MemoryBreakdown(-100.0, 500.0, 1000.0, 1000.0)
        with pytest.raises(ValueError, match="parameters_mb cannot be negative"):
            validate_memory_breakdown(breakdown)

    def test_negative_gradients_mb_raises_error(self) -> None:
        """Test that negative gradients_mb raises ValueError."""
        breakdown = MemoryBreakdown(500.0, -100.0, 1000.0, 1000.0)
        with pytest.raises(ValueError, match="gradients_mb cannot be negative"):
            validate_memory_breakdown(breakdown)

    def test_negative_activations_mb_raises_error(self) -> None:
        """Test that negative activations_mb raises ValueError."""
        breakdown = MemoryBreakdown(500.0, 500.0, -100.0, 1000.0)
        with pytest.raises(ValueError, match="activations_mb cannot be negative"):
            validate_memory_breakdown(breakdown)

    def test_negative_optimizer_state_mb_raises_error(self) -> None:
        """Test that negative optimizer_state_mb raises ValueError."""
        breakdown = MemoryBreakdown(500.0, 500.0, 1000.0, -100.0)
        with pytest.raises(ValueError, match="optimizer_state_mb cannot be negative"):
            validate_memory_breakdown(breakdown)


class TestValidateProfilingResult:
    """Tests for validate_profiling_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = ProfilingResult(None, None, 1000.0, ())
        validate_profiling_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_profiling_result(None)  # type: ignore[arg-type]

    def test_negative_throughput_raises_error(self) -> None:
        """Test that negative throughput raises ValueError."""
        result = ProfilingResult(None, None, -1.0, ())
        with pytest.raises(ValueError, match="throughput_tokens_per_sec cannot be"):
            validate_profiling_result(result)

    def test_validates_latency_if_present(self) -> None:
        """Test that latency is validated if present."""
        latency = LatencyBreakdown(-1.0, 15.0, 2.0, 1.0)
        result = ProfilingResult(latency, None, 1000.0, ())
        with pytest.raises(ValueError, match="forward_ms cannot be negative"):
            validate_profiling_result(result)

    def test_validates_memory_if_present(self) -> None:
        """Test that memory is validated if present."""
        memory = MemoryBreakdown(-100.0, 500.0, 1000.0, 1000.0)
        result = ProfilingResult(None, memory, 1000.0, ())
        with pytest.raises(ValueError, match="parameters_mb cannot be negative"):
            validate_profiling_result(result)


class TestCreateProfilingConfig:
    """Tests for create_profiling_config function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_profiling_config([ProfileMetric.LATENCY])
        assert isinstance(config, ProfilingConfig)
        assert ProfileMetric.LATENCY in config.metrics

    def test_with_custom_options(self) -> None:
        """Test config creation with custom options."""
        config = create_profiling_config(
            [ProfileMetric.LATENCY, ProfileMetric.MEMORY],
            level=ProfilingLevel.DETAILED,
            warmup_runs=5,
            num_runs=20,
            include_backward=True,
        )
        assert config.level == ProfilingLevel.DETAILED
        assert config.warmup_runs == 5
        assert config.num_runs == 20
        assert config.include_backward is True

    def test_empty_metrics_raises_error(self) -> None:
        """Test that empty metrics raises ValueError."""
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            create_profiling_config([])


class TestCreateLatencyBreakdown:
    """Tests for create_latency_breakdown function."""

    def test_creates_breakdown(self) -> None:
        """Test that function creates a breakdown."""
        breakdown = create_latency_breakdown(10.5)
        assert isinstance(breakdown, LatencyBreakdown)
        assert breakdown.forward_ms == pytest.approx(10.5)
        assert breakdown.backward_ms == 0.0

    def test_with_all_values(self) -> None:
        """Test breakdown creation with all values."""
        breakdown = create_latency_breakdown(10.0, 15.0, 2.0, 1.0)
        assert breakdown.forward_ms == pytest.approx(10.0)
        assert breakdown.backward_ms == pytest.approx(15.0)
        assert breakdown.data_loading_ms == pytest.approx(2.0)
        assert breakdown.optimizer_ms == pytest.approx(1.0)

    def test_negative_forward_raises_error(self) -> None:
        """Test that negative forward_ms raises ValueError."""
        with pytest.raises(ValueError, match="forward_ms cannot be negative"):
            create_latency_breakdown(-1.0)


class TestCreateMemoryBreakdown:
    """Tests for create_memory_breakdown function."""

    def test_creates_breakdown(self) -> None:
        """Test that function creates a breakdown."""
        breakdown = create_memory_breakdown(500.0)
        assert isinstance(breakdown, MemoryBreakdown)
        assert breakdown.parameters_mb == pytest.approx(500.0)
        assert breakdown.gradients_mb == 0.0

    def test_with_all_values(self) -> None:
        """Test breakdown creation with all values."""
        breakdown = create_memory_breakdown(500.0, 500.0, 1000.0, 1000.0)
        assert breakdown.parameters_mb == pytest.approx(500.0)
        assert breakdown.gradients_mb == pytest.approx(500.0)
        assert breakdown.activations_mb == pytest.approx(1000.0)
        assert breakdown.optimizer_state_mb == pytest.approx(1000.0)

    def test_negative_parameters_raises_error(self) -> None:
        """Test that negative parameters_mb raises ValueError."""
        with pytest.raises(ValueError, match="parameters_mb cannot be negative"):
            create_memory_breakdown(-100.0)


class TestCreateProfilingResult:
    """Tests for create_profiling_result function."""

    def test_creates_result(self) -> None:
        """Test that function creates a result."""
        result = create_profiling_result(1000.0)
        assert isinstance(result, ProfilingResult)
        assert result.throughput_tokens_per_sec == pytest.approx(1000.0)
        assert result.latency is None

    def test_with_all_options(self) -> None:
        """Test result creation with all options."""
        latency = create_latency_breakdown(10.0, 15.0)
        memory = create_memory_breakdown(500.0, 500.0)
        result = create_profiling_result(
            1500.0,
            latency=latency,
            memory=memory,
            bottlenecks=[BottleneckType.COMPUTE],
        )
        assert result.latency is not None
        assert result.memory is not None
        assert BottleneckType.COMPUTE in result.bottlenecks

    def test_negative_throughput_raises_error(self) -> None:
        """Test that negative throughput raises ValueError."""
        with pytest.raises(ValueError, match="throughput_tokens_per_sec cannot be"):
            create_profiling_result(-1.0)


class TestListProfileMetrics:
    """Tests for list_profile_metrics function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        metrics = list_profile_metrics()
        assert isinstance(metrics, list)

    def test_contains_expected_metrics(self) -> None:
        """Test that list contains expected metrics."""
        metrics = list_profile_metrics()
        assert "latency" in metrics
        assert "throughput" in metrics
        assert "memory" in metrics

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        metrics = list_profile_metrics()
        assert metrics == sorted(metrics)


class TestGetProfileMetric:
    """Tests for get_profile_metric function."""

    def test_get_latency(self) -> None:
        """Test getting LATENCY."""
        result = get_profile_metric("latency")
        assert result == ProfileMetric.LATENCY

    def test_get_memory(self) -> None:
        """Test getting MEMORY."""
        result = get_profile_metric("memory")
        assert result == ProfileMetric.MEMORY

    def test_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="invalid profile metric"):
            get_profile_metric("invalid")


class TestValidateProfileMetric:
    """Tests for validate_profile_metric function."""

    def test_valid_latency(self) -> None:
        """Test validation of latency."""
        assert validate_profile_metric("latency") is True

    def test_valid_throughput(self) -> None:
        """Test validation of throughput."""
        assert validate_profile_metric("throughput") is True

    def test_invalid_metric(self) -> None:
        """Test validation of invalid metric."""
        assert validate_profile_metric("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_profile_metric("") is False


class TestListBottleneckTypes:
    """Tests for list_bottleneck_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_bottleneck_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_bottleneck_types()
        assert "compute" in types
        assert "memory" in types
        assert "io" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_bottleneck_types()
        assert types == sorted(types)


class TestGetBottleneckType:
    """Tests for get_bottleneck_type function."""

    def test_get_compute(self) -> None:
        """Test getting COMPUTE."""
        result = get_bottleneck_type("compute")
        assert result == BottleneckType.COMPUTE

    def test_get_memory(self) -> None:
        """Test getting MEMORY."""
        result = get_bottleneck_type("memory")
        assert result == BottleneckType.MEMORY

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid bottleneck type"):
            get_bottleneck_type("invalid")


class TestValidateBottleneckType:
    """Tests for validate_bottleneck_type function."""

    def test_valid_compute(self) -> None:
        """Test validation of compute."""
        assert validate_bottleneck_type("compute") is True

    def test_valid_io(self) -> None:
        """Test validation of io."""
        assert validate_bottleneck_type("io") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_bottleneck_type("invalid") is False


class TestListProfilingLevels:
    """Tests for list_profiling_levels function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        levels = list_profiling_levels()
        assert isinstance(levels, list)

    def test_contains_expected_levels(self) -> None:
        """Test that list contains expected levels."""
        levels = list_profiling_levels()
        assert "basic" in levels
        assert "detailed" in levels
        assert "trace" in levels

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        levels = list_profiling_levels()
        assert levels == sorted(levels)


class TestGetProfilingLevel:
    """Tests for get_profiling_level function."""

    def test_get_basic(self) -> None:
        """Test getting BASIC."""
        result = get_profiling_level("basic")
        assert result == ProfilingLevel.BASIC

    def test_get_detailed(self) -> None:
        """Test getting DETAILED."""
        result = get_profiling_level("detailed")
        assert result == ProfilingLevel.DETAILED

    def test_invalid_level_raises_error(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="invalid profiling level"):
            get_profiling_level("invalid")


class TestValidateProfilingLevel:
    """Tests for validate_profiling_level function."""

    def test_valid_basic(self) -> None:
        """Test validation of basic."""
        assert validate_profiling_level("basic") is True

    def test_valid_trace(self) -> None:
        """Test validation of trace."""
        assert validate_profiling_level("trace") is True

    def test_invalid_level(self) -> None:
        """Test validation of invalid level."""
        assert validate_profiling_level("invalid") is False


class TestCalculateFlops:
    """Tests for calculate_flops function."""

    def test_basic_calculation(self) -> None:
        """Test basic FLOPS calculation."""
        flops = calculate_flops(1_000_000, 512)
        assert flops == 2 * 1_000_000 * 512

    def test_with_batch_size(self) -> None:
        """Test FLOPS calculation with batch size."""
        flops = calculate_flops(1_000_000, 512, batch_size=8)
        assert flops == 2 * 1_000_000 * 512 * 8

    def test_with_training(self) -> None:
        """Test FLOPS calculation for training."""
        inference_flops = calculate_flops(1_000_000, 512)
        training_flops = calculate_flops(1_000_000, 512, is_training=True)
        assert training_flops == 3 * inference_flops

    def test_zero_parameters_raises_error(self) -> None:
        """Test that zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            calculate_flops(0, 512)

    def test_zero_sequence_length_raises_error(self) -> None:
        """Test that zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            calculate_flops(1_000_000, 0)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_flops(1_000_000, 512, batch_size=0)


class TestEstimateMemoryFootprint:
    """Tests for estimate_memory_footprint function."""

    def test_basic_estimation(self) -> None:
        """Test basic memory estimation."""
        breakdown = estimate_memory_footprint(
            num_parameters=1_000_000,
            batch_size=1,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
        )
        assert isinstance(breakdown, MemoryBreakdown)
        assert breakdown.parameters_mb > 0
        assert breakdown.gradients_mb == 0.0  # Not training
        assert breakdown.activations_mb > 0
        assert breakdown.optimizer_state_mb == 0.0  # Not training

    def test_training_mode(self) -> None:
        """Test memory estimation in training mode."""
        breakdown = estimate_memory_footprint(
            num_parameters=1_000_000,
            batch_size=8,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            is_training=True,
        )
        assert breakdown.gradients_mb > 0
        assert breakdown.optimizer_state_mb > 0

    def test_fp16_precision(self) -> None:
        """Test memory estimation with FP16 precision."""
        fp32_breakdown = estimate_memory_footprint(
            num_parameters=1_000_000,
            batch_size=1,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            bytes_per_param=4,
        )
        fp16_breakdown = estimate_memory_footprint(
            num_parameters=1_000_000,
            batch_size=1,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            bytes_per_param=2,
        )
        assert fp16_breakdown.parameters_mb < fp32_breakdown.parameters_mb

    def test_zero_parameters_raises_error(self) -> None:
        """Test that zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_memory_footprint(0, 1, 512, 768, 12)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_memory_footprint(1_000_000, 0, 512, 768, 12)

    def test_zero_sequence_length_raises_error(self) -> None:
        """Test that zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_memory_footprint(1_000_000, 1, 0, 768, 12)

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_memory_footprint(1_000_000, 1, 512, 0, 12)

    def test_zero_num_layers_raises_error(self) -> None:
        """Test that zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_memory_footprint(1_000_000, 1, 512, 768, 0)

    def test_zero_bytes_per_param_raises_error(self) -> None:
        """Test that zero bytes_per_param raises ValueError."""
        with pytest.raises(ValueError, match="bytes_per_param must be positive"):
            estimate_memory_footprint(1_000_000, 1, 512, 768, 12, bytes_per_param=0)


class TestIdentifyBottlenecks:
    """Tests for identify_bottlenecks function."""

    def test_io_bottleneck(self) -> None:
        """Test identification of I/O bottleneck."""
        latency = LatencyBreakdown(10.0, 15.0, 20.0, 1.0)  # High data loading
        memory = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        bottlenecks = identify_bottlenecks(latency, memory)
        assert BottleneckType.IO in bottlenecks

    def test_memory_bottleneck(self) -> None:
        """Test identification of memory bottleneck."""
        latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        memory = MemoryBreakdown(5000.0, 5000.0, 5000.0, 5000.0)  # High memory
        bottlenecks = identify_bottlenecks(latency, memory)
        assert BottleneckType.MEMORY in bottlenecks

    def test_compute_bottleneck(self) -> None:
        """Test identification of compute bottleneck."""
        latency = LatencyBreakdown(50.0, 60.0, 2.0, 1.0)  # Compute dominates
        memory = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        bottlenecks = identify_bottlenecks(latency, memory)
        assert BottleneckType.COMPUTE in bottlenecks

    def test_none_latency_raises_error(self) -> None:
        """Test that None latency raises ValueError."""
        memory = MemoryBreakdown(500.0, 500.0, 1000.0, 1000.0)
        with pytest.raises(ValueError, match="latency cannot be None"):
            identify_bottlenecks(None, memory)  # type: ignore[arg-type]

    def test_none_memory_raises_error(self) -> None:
        """Test that None memory raises ValueError."""
        latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)
        with pytest.raises(ValueError, match="memory cannot be None"):
            identify_bottlenecks(latency, None)  # type: ignore[arg-type]

    def test_default_to_compute(self) -> None:
        """Test that compute is default when no other bottleneck."""
        latency = LatencyBreakdown(10.0, 15.0, 2.0, 1.0)  # Normal
        memory = MemoryBreakdown(100.0, 100.0, 200.0, 200.0)  # Low memory
        bottlenecks = identify_bottlenecks(latency, memory)
        assert BottleneckType.COMPUTE in bottlenecks

    def test_zero_latency_no_compute_check(self) -> None:
        """Test behavior when total latency is zero."""
        latency = LatencyBreakdown(0.0, 0.0, 0.0, 0.0)  # All zeros
        memory = MemoryBreakdown(100.0, 100.0, 200.0, 200.0)  # Low memory
        bottlenecks = identify_bottlenecks(latency, memory)
        # Should default to compute when no bottleneck identified
        assert BottleneckType.COMPUTE in bottlenecks

    def test_io_bottleneck_excludes_compute(self) -> None:
        """Test that IO bottleneck excludes compute from being added."""
        # Data loading dominates (40%), so IO bottleneck identified
        # But compute is not added because IO is present
        latency = LatencyBreakdown(20.0, 20.0, 50.0, 5.0)  # High data loading
        memory = MemoryBreakdown(100.0, 100.0, 200.0, 200.0)  # Low memory
        bottlenecks = identify_bottlenecks(latency, memory)
        assert BottleneckType.IO in bottlenecks
        # Compute should not be added when IO is present
        assert BottleneckType.COMPUTE not in bottlenecks


class TestCompareProfiles:
    """Tests for compare_profiles function."""

    def test_compare_two_profiles(self) -> None:
        """Test comparing two profiling results."""
        result1 = create_profiling_result(1000.0)
        result2 = create_profiling_result(1500.0)
        comparison = compare_profiles([result1, result2], ["model_a", "model_b"])
        assert comparison["fastest"] == "model_b"
        assert comparison["slowest"] == "model_a"
        assert comparison["num_profiles"] == 2

    def test_throughput_range(self) -> None:
        """Test throughput range in comparison."""
        result1 = create_profiling_result(1000.0)
        result2 = create_profiling_result(1500.0)
        comparison = compare_profiles([result1, result2])
        assert comparison["throughput_range"]["min"] == pytest.approx(1000.0)
        assert comparison["throughput_range"]["max"] == pytest.approx(1500.0)
        assert comparison["throughput_range"]["mean"] == pytest.approx(1250.0)

    def test_with_latency_comparison(self) -> None:
        """Test comparison with latency data."""
        latency1 = create_latency_breakdown(10.0, 15.0)
        latency2 = create_latency_breakdown(8.0, 12.0)
        result1 = create_profiling_result(1000.0, latency=latency1)
        result2 = create_profiling_result(1500.0, latency=latency2)
        comparison = compare_profiles([result1, result2], ["model_a", "model_b"])
        assert "latency_comparison" in comparison
        assert comparison["latency_comparison"]["lowest_latency"] == "model_b"

    def test_with_memory_comparison(self) -> None:
        """Test comparison with memory data."""
        memory1 = create_memory_breakdown(500.0, 500.0)
        memory2 = create_memory_breakdown(400.0, 400.0)
        result1 = create_profiling_result(1000.0, memory=memory1)
        result2 = create_profiling_result(1500.0, memory=memory2)
        comparison = compare_profiles([result1, result2], ["model_a", "model_b"])
        assert "memory_comparison" in comparison
        assert comparison["memory_comparison"]["lowest_memory"] == "model_b"

    def test_bottleneck_counts(self) -> None:
        """Test bottleneck aggregation."""
        result1 = create_profiling_result(1000.0, bottlenecks=[BottleneckType.COMPUTE])
        result2 = create_profiling_result(1500.0, bottlenecks=[BottleneckType.COMPUTE])
        comparison = compare_profiles([result1, result2])
        assert comparison["bottleneck_counts"]["compute"] == 2

    def test_none_profiles_raises_error(self) -> None:
        """Test that None profiles raises ValueError."""
        with pytest.raises(ValueError, match="profiles cannot be None"):
            compare_profiles(None)  # type: ignore[arg-type]

    def test_empty_profiles_raises_error(self) -> None:
        """Test that empty profiles raises ValueError."""
        with pytest.raises(ValueError, match="profiles cannot be empty"):
            compare_profiles([])

    def test_names_length_mismatch_raises_error(self) -> None:
        """Test that mismatched names length raises ValueError."""
        result1 = create_profiling_result(1000.0)
        with pytest.raises(ValueError, match="must match profiles length"):
            compare_profiles([result1], ["a", "b"])

    def test_default_names(self) -> None:
        """Test that default names are generated."""
        result1 = create_profiling_result(1000.0)
        result2 = create_profiling_result(1500.0)
        comparison = compare_profiles([result1, result2])
        assert comparison["fastest"] == "profile_1"
        assert comparison["slowest"] == "profile_0"


class TestFormatProfilingResult:
    """Tests for format_profiling_result function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        result = create_profiling_result(1000.0)
        formatted = format_profiling_result(result)
        assert "Model Profiling Results" in formatted
        assert "1000.0" in formatted
        assert "Throughput" in formatted

    def test_format_with_latency(self) -> None:
        """Test formatting with latency."""
        latency = create_latency_breakdown(10.0, 15.0, 2.0, 1.0)
        result = create_profiling_result(1000.0, latency=latency)
        formatted = format_profiling_result(result)
        assert "Latency Breakdown" in formatted
        assert "Forward" in formatted
        assert "Backward" in formatted

    def test_format_with_memory(self) -> None:
        """Test formatting with memory."""
        memory = create_memory_breakdown(500.0, 500.0, 1000.0, 1000.0)
        result = create_profiling_result(1000.0, memory=memory)
        formatted = format_profiling_result(result)
        assert "Memory Breakdown" in formatted
        assert "Parameters" in formatted
        assert "Activations" in formatted

    def test_format_with_bottlenecks(self) -> None:
        """Test formatting with bottlenecks."""
        result = create_profiling_result(
            1000.0,
            bottlenecks=[BottleneckType.COMPUTE, BottleneckType.MEMORY],
        )
        formatted = format_profiling_result(result)
        assert "Identified Bottlenecks" in formatted
        assert "compute" in formatted
        assert "memory" in formatted

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            format_profiling_result(None)  # type: ignore[arg-type]


class TestGetRecommendedProfilingConfig:
    """Tests for get_recommended_profiling_config function."""

    def test_llm_config(self) -> None:
        """Test LLM recommended config."""
        config = get_recommended_profiling_config("llm")
        assert "metrics" in config
        assert "level" in config
        assert "recommended_batch_sizes" in config

    def test_vision_config(self) -> None:
        """Test vision recommended config."""
        config = get_recommended_profiling_config("vision")
        assert "metrics" in config
        assert "recommended_image_sizes" in config

    def test_multimodal_config(self) -> None:
        """Test multimodal recommended config."""
        config = get_recommended_profiling_config("multimodal")
        assert "metrics" in config
        assert config["level"] == ProfilingLevel.TRACE

    def test_training_config(self) -> None:
        """Test training recommended config."""
        config = get_recommended_profiling_config("training")
        assert config["include_backward"] is True

    def test_quick_mode(self) -> None:
        """Test quick profiling mode."""
        normal_config = get_recommended_profiling_config("llm")
        quick_config = get_recommended_profiling_config("llm", quick=True)
        assert quick_config["num_runs"] < normal_config["num_runs"]
        assert quick_config["warmup_runs"] < normal_config["warmup_runs"]

    def test_unknown_model_type_returns_base(self) -> None:
        """Test that unknown model type returns base config."""
        config = get_recommended_profiling_config("unknown")
        assert "metrics" in config
        assert "level" in config

    def test_case_insensitive(self) -> None:
        """Test that model type is case insensitive."""
        config1 = get_recommended_profiling_config("LLM")
        config2 = get_recommended_profiling_config("llm")
        assert config1 == config2

    def test_none_model_type_raises_error(self) -> None:
        """Test that None model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be None"):
            get_recommended_profiling_config(None)  # type: ignore[arg-type]

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_profiling_config("")


class TestPropertyBased:
    """Property-based tests for profiling functions."""

    @given(
        st.integers(min_value=1, max_value=1_000_000_000),
        st.integers(min_value=1, max_value=8192),
        st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=10)
    def test_flops_scales_with_inputs(
        self,
        num_params: int,
        seq_len: int,
        batch_size: int,
    ) -> None:
        """Test that FLOPS scales linearly with inputs."""
        flops = calculate_flops(num_params, seq_len, batch_size)
        assert flops > 0
        # Verify linear relationship
        assert flops == 2 * num_params * seq_len * batch_size

    @given(
        st.floats(min_value=0.0, max_value=1000.0),
        st.floats(min_value=0.0, max_value=1000.0),
        st.floats(min_value=0.0, max_value=1000.0),
        st.floats(min_value=0.0, max_value=1000.0),
    )
    @settings(max_examples=10)
    def test_latency_total_is_sum(
        self,
        forward: float,
        backward: float,
        loading: float,
        optimizer: float,
    ) -> None:
        """Test that total_ms is the sum of all components."""
        breakdown = LatencyBreakdown(forward, backward, loading, optimizer)
        expected = forward + backward + loading + optimizer
        assert breakdown.total_ms == pytest.approx(expected)

    @given(
        st.floats(min_value=0.0, max_value=10000.0),
        st.floats(min_value=0.0, max_value=10000.0),
        st.floats(min_value=0.0, max_value=10000.0),
        st.floats(min_value=0.0, max_value=10000.0),
    )
    @settings(max_examples=10)
    def test_memory_total_is_sum(
        self,
        params: float,
        grads: float,
        acts: float,
        opt_state: float,
    ) -> None:
        """Test that total_mb is the sum of all components."""
        breakdown = MemoryBreakdown(params, grads, acts, opt_state)
        expected = params + grads + acts + opt_state
        assert breakdown.total_mb == pytest.approx(expected)

    @given(st.floats(min_value=0.0, max_value=100000.0))
    @settings(max_examples=10)
    def test_throughput_nonnegative(self, throughput: float) -> None:
        """Test that profiling result accepts non-negative throughput."""
        result = create_profiling_result(throughput)
        assert result.throughput_tokens_per_sec == pytest.approx(throughput)
