"""Tests for inference.memory module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.inference.memory import (
    VALID_MEMORY_COMPONENTS,
    VALID_MEMORY_UNITS,
    VALID_OPTIMIZATION_STRATEGIES,
    MemoryBudget,
    MemoryComponent,
    MemoryConfig,
    MemoryEstimate,
    MemoryStats,
    MemoryUnit,
    OptimizationStrategy,
    convert_memory_units,
    create_memory_budget,
    create_memory_config,
    create_memory_estimate,
    create_memory_stats,
    detect_memory_bottleneck,
    estimate_inference_memory,
    estimate_model_memory,
    estimate_training_memory,
    find_max_batch_size,
    format_memory_estimate,
    format_memory_stats,
    get_memory_component,
    get_memory_unit,
    get_optimization_strategy,
    get_recommended_memory_config,
    list_memory_components,
    list_memory_units,
    list_optimization_strategies,
    validate_memory_budget,
    validate_memory_config,
    validate_memory_estimate,
    validate_memory_stats,
)


class TestMemoryComponent:
    """Tests for MemoryComponent enum."""

    def test_parameters_value(self) -> None:
        """Test PARAMETERS value."""
        assert MemoryComponent.PARAMETERS.value == "parameters"

    def test_gradients_value(self) -> None:
        """Test GRADIENTS value."""
        assert MemoryComponent.GRADIENTS.value == "gradients"

    def test_optimizer_states_value(self) -> None:
        """Test OPTIMIZER_STATES value."""
        assert MemoryComponent.OPTIMIZER_STATES.value == "optimizer_states"

    def test_activations_value(self) -> None:
        """Test ACTIVATIONS value."""
        assert MemoryComponent.ACTIVATIONS.value == "activations"

    def test_kv_cache_value(self) -> None:
        """Test KV_CACHE value."""
        assert MemoryComponent.KV_CACHE.value == "kv_cache"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [c.value for c in MemoryComponent]
        assert len(values) == len(set(values))

    def test_all_components_have_string_values(self) -> None:
        """All components have string values."""
        for component in MemoryComponent:
            assert isinstance(component.value, str)


class TestValidMemoryComponents:
    """Tests for VALID_MEMORY_COMPONENTS frozenset."""

    def test_is_frozenset(self) -> None:
        """Test VALID_MEMORY_COMPONENTS is a frozenset."""
        assert isinstance(VALID_MEMORY_COMPONENTS, frozenset)

    def test_contains_all_enums(self) -> None:
        """Test VALID_MEMORY_COMPONENTS contains all enum values."""
        for component in MemoryComponent:
            assert component.value in VALID_MEMORY_COMPONENTS

    def test_is_immutable(self) -> None:
        """Test that frozenset is immutable."""
        with pytest.raises(AttributeError):
            VALID_MEMORY_COMPONENTS.add("new")  # type: ignore[attr-defined]


class TestMemoryUnit:
    """Tests for MemoryUnit enum."""

    def test_bytes_value(self) -> None:
        """Test BYTES value."""
        assert MemoryUnit.BYTES.value == "bytes"

    def test_kb_value(self) -> None:
        """Test KB value."""
        assert MemoryUnit.KB.value == "kb"

    def test_mb_value(self) -> None:
        """Test MB value."""
        assert MemoryUnit.MB.value == "mb"

    def test_gb_value(self) -> None:
        """Test GB value."""
        assert MemoryUnit.GB.value == "gb"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [u.value for u in MemoryUnit]
        assert len(values) == len(set(values))


class TestValidMemoryUnits:
    """Tests for VALID_MEMORY_UNITS frozenset."""

    def test_is_frozenset(self) -> None:
        """Test VALID_MEMORY_UNITS is a frozenset."""
        assert isinstance(VALID_MEMORY_UNITS, frozenset)

    def test_contains_all_enums(self) -> None:
        """Test VALID_MEMORY_UNITS contains all enum values."""
        for unit in MemoryUnit:
            assert unit.value in VALID_MEMORY_UNITS


class TestOptimizationStrategy:
    """Tests for OptimizationStrategy enum."""

    def test_gradient_checkpointing_value(self) -> None:
        """Test GRADIENT_CHECKPOINTING value."""
        assert (
            OptimizationStrategy.GRADIENT_CHECKPOINTING.value
            == "gradient_checkpointing"
        )

    def test_cpu_offload_value(self) -> None:
        """Test CPU_OFFLOAD value."""
        assert OptimizationStrategy.CPU_OFFLOAD.value == "cpu_offload"

    def test_disk_offload_value(self) -> None:
        """Test DISK_OFFLOAD value."""
        assert OptimizationStrategy.DISK_OFFLOAD.value == "disk_offload"

    def test_quantization_value(self) -> None:
        """Test QUANTIZATION value."""
        assert OptimizationStrategy.QUANTIZATION.value == "quantization"

    def test_all_values_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [s.value for s in OptimizationStrategy]
        assert len(values) == len(set(values))


class TestValidOptimizationStrategies:
    """Tests for VALID_OPTIMIZATION_STRATEGIES frozenset."""

    def test_is_frozenset(self) -> None:
        """Test VALID_OPTIMIZATION_STRATEGIES is a frozenset."""
        assert isinstance(VALID_OPTIMIZATION_STRATEGIES, frozenset)

    def test_contains_all_enums(self) -> None:
        """Test VALID_OPTIMIZATION_STRATEGIES contains all enum values."""
        for strategy in OptimizationStrategy:
            assert strategy.value in VALID_OPTIMIZATION_STRATEGIES


class TestMemoryEstimate:
    """Tests for MemoryEstimate dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryEstimate instance."""
        estimate = MemoryEstimate(
            parameters_mb=14000.0,
            activations_mb=2000.0,
            kv_cache_mb=1000.0,
            total_mb=17000.0,
        )
        assert estimate.parameters_mb == 14000.0
        assert estimate.activations_mb == 2000.0
        assert estimate.kv_cache_mb == 1000.0
        assert estimate.total_mb == 17000.0

    def test_frozen(self) -> None:
        """Test that MemoryEstimate is immutable."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        with pytest.raises(AttributeError):
            estimate.parameters_mb = 10000.0  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that MemoryEstimate uses slots."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        assert not hasattr(estimate, "__dict__")


class TestMemoryBudget:
    """Tests for MemoryBudget dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryBudget instance."""
        budget = MemoryBudget(
            gpu_memory_gb=24.0,
            cpu_memory_gb=64.0,
            allow_offload=True,
        )
        assert budget.gpu_memory_gb == 24.0
        assert budget.cpu_memory_gb == 64.0
        assert budget.allow_offload is True

    def test_frozen(self) -> None:
        """Test that MemoryBudget is immutable."""
        budget = MemoryBudget(24.0, 64.0, True)
        with pytest.raises(AttributeError):
            budget.gpu_memory_gb = 48.0  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that MemoryBudget uses slots."""
        budget = MemoryBudget(24.0, 64.0, True)
        assert not hasattr(budget, "__dict__")


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryConfig instance."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(
            budget=budget,
            optimization_strategies=(OptimizationStrategy.GRADIENT_CHECKPOINTING,),
            batch_size=4,
            sequence_length=2048,
        )
        assert config.budget == budget
        assert len(config.optimization_strategies) == 1
        assert config.batch_size == 4
        assert config.sequence_length == 2048

    def test_frozen(self) -> None:
        """Test that MemoryConfig is immutable."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(budget, (), 4, 2048)
        with pytest.raises(AttributeError):
            config.batch_size = 8  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that MemoryConfig uses slots."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(budget, (), 4, 2048)
        assert not hasattr(config, "__dict__")


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_creation(self) -> None:
        """Test creating MemoryStats instance."""
        stats = MemoryStats(
            peak_memory_mb=20000.0,
            allocated_mb=18000.0,
            reserved_mb=22000.0,
            utilization=0.82,
        )
        assert stats.peak_memory_mb == 20000.0
        assert stats.allocated_mb == 18000.0
        assert stats.reserved_mb == 22000.0
        assert stats.utilization == 0.82

    def test_frozen(self) -> None:
        """Test that MemoryStats is immutable."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        with pytest.raises(AttributeError):
            stats.peak_memory_mb = 25000.0  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test that MemoryStats uses slots."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        assert not hasattr(stats, "__dict__")


class TestValidateMemoryEstimate:
    """Tests for validate_memory_estimate function."""

    def test_valid_estimate(self) -> None:
        """Valid estimate passes validation."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        validate_memory_estimate(estimate)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        estimate = MemoryEstimate(0.0, 0.0, 0.0, 0.0)
        validate_memory_estimate(estimate)

    def test_negative_parameters_raises(self) -> None:
        """Negative parameters_mb raises ValueError."""
        estimate = MemoryEstimate(-100.0, 2000.0, 1000.0, 17000.0)
        with pytest.raises(ValueError, match="parameters_mb cannot be negative"):
            validate_memory_estimate(estimate)

    def test_negative_activations_raises(self) -> None:
        """Negative activations_mb raises ValueError."""
        estimate = MemoryEstimate(14000.0, -100.0, 1000.0, 17000.0)
        with pytest.raises(ValueError, match="activations_mb cannot be negative"):
            validate_memory_estimate(estimate)

    def test_negative_kv_cache_raises(self) -> None:
        """Negative kv_cache_mb raises ValueError."""
        estimate = MemoryEstimate(14000.0, 2000.0, -100.0, 17000.0)
        with pytest.raises(ValueError, match="kv_cache_mb cannot be negative"):
            validate_memory_estimate(estimate)

    def test_negative_total_raises(self) -> None:
        """Negative total_mb raises ValueError."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, -100.0)
        with pytest.raises(ValueError, match="total_mb cannot be negative"):
            validate_memory_estimate(estimate)


class TestValidateMemoryBudget:
    """Tests for validate_memory_budget function."""

    def test_valid_budget(self) -> None:
        """Valid budget passes validation."""
        budget = MemoryBudget(24.0, 64.0, True)
        validate_memory_budget(budget)

    def test_zero_gpu_memory_valid(self) -> None:
        """Zero GPU memory is valid (CPU-only)."""
        budget = MemoryBudget(0.0, 64.0, True)
        validate_memory_budget(budget)

    def test_negative_gpu_memory_raises(self) -> None:
        """Negative GPU memory raises ValueError."""
        budget = MemoryBudget(-1.0, 64.0, True)
        with pytest.raises(ValueError, match="gpu_memory_gb cannot be negative"):
            validate_memory_budget(budget)

    def test_negative_cpu_memory_raises(self) -> None:
        """Negative CPU memory raises ValueError."""
        budget = MemoryBudget(24.0, -1.0, True)
        with pytest.raises(ValueError, match="cpu_memory_gb cannot be negative"):
            validate_memory_budget(budget)


class TestValidateMemoryConfig:
    """Tests for validate_memory_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(budget, (), 4, 2048)
        validate_memory_config(config)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(budget, (), 0, 2048)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_memory_config(config)

    def test_negative_batch_size_raises(self) -> None:
        """Negative batch_size raises ValueError."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(budget, (), -1, 2048)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_memory_config(config)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence_length raises ValueError."""
        budget = MemoryBudget(24.0, 64.0, True)
        config = MemoryConfig(budget, (), 4, 0)
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            validate_memory_config(config)


class TestValidateMemoryStats:
    """Tests for validate_memory_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        validate_memory_stats(stats)

    def test_zero_utilization_valid(self) -> None:
        """Zero utilization is valid."""
        stats = MemoryStats(0.0, 0.0, 0.0, 0.0)
        validate_memory_stats(stats)

    def test_full_utilization_valid(self) -> None:
        """Full utilization (1.0) is valid."""
        stats = MemoryStats(20000.0, 20000.0, 20000.0, 1.0)
        validate_memory_stats(stats)

    def test_negative_peak_raises(self) -> None:
        """Negative peak_memory_mb raises ValueError."""
        stats = MemoryStats(-100.0, 18000.0, 22000.0, 0.82)
        with pytest.raises(ValueError, match="peak_memory_mb cannot be negative"):
            validate_memory_stats(stats)

    def test_negative_allocated_raises(self) -> None:
        """Negative allocated_mb raises ValueError."""
        stats = MemoryStats(20000.0, -100.0, 22000.0, 0.82)
        with pytest.raises(ValueError, match="allocated_mb cannot be negative"):
            validate_memory_stats(stats)

    def test_negative_reserved_raises(self) -> None:
        """Negative reserved_mb raises ValueError."""
        stats = MemoryStats(20000.0, 18000.0, -100.0, 0.82)
        with pytest.raises(ValueError, match="reserved_mb cannot be negative"):
            validate_memory_stats(stats)

    def test_utilization_below_zero_raises(self) -> None:
        """Utilization below 0 raises ValueError."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, -0.1)
        with pytest.raises(ValueError, match="utilization must be between"):
            validate_memory_stats(stats)

    def test_utilization_above_one_raises(self) -> None:
        """Utilization above 1 raises ValueError."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 1.1)
        with pytest.raises(ValueError, match="utilization must be between"):
            validate_memory_stats(stats)


class TestCreateMemoryEstimate:
    """Tests for create_memory_estimate function."""

    def test_default_estimate(self) -> None:
        """Create estimate with defaults."""
        estimate = create_memory_estimate(14000.0)
        assert estimate.parameters_mb == 14000.0
        assert estimate.activations_mb == 0.0
        assert estimate.kv_cache_mb == 0.0
        assert estimate.total_mb == 14000.0

    def test_custom_estimate(self) -> None:
        """Create estimate with custom values."""
        estimate = create_memory_estimate(14000.0, 2000.0, 1000.0)
        assert estimate.parameters_mb == 14000.0
        assert estimate.activations_mb == 2000.0
        assert estimate.kv_cache_mb == 1000.0
        assert estimate.total_mb == 17000.0

    def test_explicit_total(self) -> None:
        """Create estimate with explicit total."""
        estimate = create_memory_estimate(14000.0, 2000.0, 1000.0, total_mb=20000.0)
        assert estimate.total_mb == 20000.0

    def test_negative_params_raises(self) -> None:
        """Negative parameters raises ValueError."""
        with pytest.raises(ValueError, match="parameters_mb cannot be negative"):
            create_memory_estimate(-100.0)


class TestCreateMemoryBudget:
    """Tests for create_memory_budget function."""

    def test_default_budget(self) -> None:
        """Create budget with defaults."""
        budget = create_memory_budget()
        assert budget.gpu_memory_gb == 24.0
        assert budget.cpu_memory_gb == 64.0
        assert budget.allow_offload is True

    def test_custom_budget(self) -> None:
        """Create budget with custom values."""
        budget = create_memory_budget(gpu_memory_gb=48.0, allow_offload=False)
        assert budget.gpu_memory_gb == 48.0
        assert budget.allow_offload is False

    def test_negative_gpu_raises(self) -> None:
        """Negative GPU memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb cannot be negative"):
            create_memory_budget(gpu_memory_gb=-1.0)


class TestCreateMemoryConfig:
    """Tests for create_memory_config function."""

    def test_default_config(self) -> None:
        """Create config with defaults."""
        config = create_memory_config()
        assert config.batch_size == 1
        assert config.sequence_length == 2048
        assert len(config.optimization_strategies) == 0

    def test_custom_config(self) -> None:
        """Create config with custom values."""
        config = create_memory_config(
            batch_size=4,
            sequence_length=4096,
            optimization_strategies=("gradient_checkpointing",),
        )
        assert config.batch_size == 4
        assert config.sequence_length == 4096
        assert len(config.optimization_strategies) == 1

    def test_with_budget(self) -> None:
        """Create config with custom budget."""
        budget = create_memory_budget(gpu_memory_gb=48.0)
        config = create_memory_config(budget=budget)
        assert config.budget.gpu_memory_gb == 48.0

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="optimization strategy must be one of"):
            create_memory_config(optimization_strategies=("invalid",))

    def test_zero_batch_raises(self) -> None:
        """Zero batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_memory_config(batch_size=0)


class TestCreateMemoryStats:
    """Tests for create_memory_stats function."""

    def test_basic_stats(self) -> None:
        """Create basic stats."""
        stats = create_memory_stats(20000.0, 18000.0, 22000.0)
        assert stats.peak_memory_mb == 20000.0
        assert stats.allocated_mb == 18000.0
        assert stats.reserved_mb == 22000.0

    def test_auto_utilization(self) -> None:
        """Utilization is computed if not provided."""
        stats = create_memory_stats(20000.0, 18000.0, 20000.0)
        assert stats.utilization == 0.9

    def test_explicit_utilization(self) -> None:
        """Explicit utilization is used."""
        stats = create_memory_stats(20000.0, 18000.0, 22000.0, utilization=0.5)
        assert stats.utilization == 0.5

    def test_zero_reserved_utilization(self) -> None:
        """Zero reserved gives zero utilization."""
        stats = create_memory_stats(0.0, 0.0, 0.0)
        assert stats.utilization == 0.0

    def test_negative_peak_raises(self) -> None:
        """Negative peak raises ValueError."""
        with pytest.raises(ValueError, match="peak_memory_mb cannot be negative"):
            create_memory_stats(-100.0, 18000.0, 22000.0)


class TestListFunctions:
    """Tests for list functions."""

    def test_list_memory_components(self) -> None:
        """List memory components."""
        components = list_memory_components()
        assert "parameters" in components
        assert "kv_cache" in components
        assert components == sorted(components)

    def test_list_memory_units(self) -> None:
        """List memory units."""
        units = list_memory_units()
        assert "bytes" in units
        assert "gb" in units
        assert units == sorted(units)

    def test_list_optimization_strategies(self) -> None:
        """List optimization strategies."""
        strategies = list_optimization_strategies()
        assert "gradient_checkpointing" in strategies
        assert "cpu_offload" in strategies
        assert strategies == sorted(strategies)


class TestGetFunctions:
    """Tests for get functions."""

    def test_get_memory_component(self) -> None:
        """Get memory component by name."""
        assert get_memory_component("parameters") == MemoryComponent.PARAMETERS
        assert get_memory_component("kv_cache") == MemoryComponent.KV_CACHE

    def test_get_memory_component_invalid(self) -> None:
        """Invalid component name raises ValueError."""
        with pytest.raises(ValueError, match="memory component must be one of"):
            get_memory_component("invalid")

    def test_get_memory_unit(self) -> None:
        """Get memory unit by name."""
        assert get_memory_unit("bytes") == MemoryUnit.BYTES
        assert get_memory_unit("gb") == MemoryUnit.GB

    def test_get_memory_unit_invalid(self) -> None:
        """Invalid unit name raises ValueError."""
        with pytest.raises(ValueError, match="memory unit must be one of"):
            get_memory_unit("invalid")

    def test_get_optimization_strategy(self) -> None:
        """Get optimization strategy by name."""
        assert (
            get_optimization_strategy("gradient_checkpointing")
            == OptimizationStrategy.GRADIENT_CHECKPOINTING
        )
        assert (
            get_optimization_strategy("cpu_offload") == OptimizationStrategy.CPU_OFFLOAD
        )

    def test_get_optimization_strategy_invalid(self) -> None:
        """Invalid strategy name raises ValueError."""
        with pytest.raises(ValueError, match="optimization strategy must be one of"):
            get_optimization_strategy("invalid")


class TestEstimateModelMemory:
    """Tests for estimate_model_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic memory estimate."""
        mem = estimate_model_memory(1_000_000_000)  # 1B params
        assert mem > 0
        # 1B * 2 bytes = 2GB = ~1907 MB
        assert 1900 < mem < 2000

    def test_fp32_estimate(self) -> None:
        """FP32 memory estimate."""
        mem_fp16 = estimate_model_memory(1_000_000_000, dtype_bytes=2)
        mem_fp32 = estimate_model_memory(1_000_000_000, dtype_bytes=4)
        assert mem_fp32 == mem_fp16 * 2

    def test_zero_params_raises(self) -> None:
        """Zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_model_memory(0)

    def test_negative_params_raises(self) -> None:
        """Negative parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_model_memory(-1)

    def test_zero_dtype_raises(self) -> None:
        """Zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_model_memory(1_000_000_000, dtype_bytes=0)


class TestEstimateInferenceMemory:
    """Tests for estimate_inference_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic inference memory estimate."""
        estimate = estimate_inference_memory(7_000_000_000)
        assert estimate.parameters_mb > 0
        assert estimate.activations_mb > 0
        assert estimate.kv_cache_mb > 0
        assert estimate.total_mb > estimate.parameters_mb

    def test_larger_batch_more_memory(self) -> None:
        """Larger batch uses more memory."""
        est1 = estimate_inference_memory(7_000_000_000, batch_size=1)
        est4 = estimate_inference_memory(7_000_000_000, batch_size=4)
        assert est4.total_mb > est1.total_mb

    def test_longer_sequence_more_memory(self) -> None:
        """Longer sequence uses more memory."""
        est_short = estimate_inference_memory(7_000_000_000, sequence_length=1024)
        est_long = estimate_inference_memory(7_000_000_000, sequence_length=4096)
        assert est_long.total_mb > est_short.total_mb

    def test_zero_params_raises(self) -> None:
        """Zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_inference_memory(0)

    def test_zero_batch_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_inference_memory(7_000_000_000, batch_size=0)

    def test_zero_sequence_raises(self) -> None:
        """Zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_inference_memory(7_000_000_000, sequence_length=0)

    def test_zero_layers_raises(self) -> None:
        """Zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_inference_memory(7_000_000_000, num_layers=0)

    def test_zero_hidden_raises(self) -> None:
        """Zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_inference_memory(7_000_000_000, hidden_size=0)

    def test_zero_heads_raises(self) -> None:
        """Zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            estimate_inference_memory(7_000_000_000, num_heads=0)

    def test_zero_dtype_raises(self) -> None:
        """Zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_inference_memory(7_000_000_000, dtype_bytes=0)


class TestEstimateTrainingMemory:
    """Tests for estimate_training_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic training memory estimate."""
        estimate = estimate_training_memory(7_000_000_000)
        assert estimate.parameters_mb > 0
        assert estimate.activations_mb > 0
        # Training has no KV cache
        assert estimate.kv_cache_mb == 0.0

    def test_training_more_than_inference(self) -> None:
        """Training uses more memory than inference."""
        train_est = estimate_training_memory(7_000_000_000)
        infer_est = estimate_inference_memory(7_000_000_000)
        assert train_est.total_mb > infer_est.total_mb

    def test_gradient_checkpointing_reduces_memory(self) -> None:
        """Gradient checkpointing reduces activation memory."""
        no_cp = estimate_training_memory(7_000_000_000, gradient_checkpointing=False)
        with_cp = estimate_training_memory(7_000_000_000, gradient_checkpointing=True)
        assert with_cp.activations_mb < no_cp.activations_mb

    def test_zero_params_raises(self) -> None:
        """Zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            estimate_training_memory(0)

    def test_zero_batch_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_training_memory(7_000_000_000, batch_size=0)

    def test_negative_optimizer_states_raises(self) -> None:
        """Negative optimizer_states raises ValueError."""
        with pytest.raises(ValueError, match="optimizer_states cannot be negative"):
            estimate_training_memory(7_000_000_000, optimizer_states=-1)

    def test_zero_sequence_raises(self) -> None:
        """Zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_training_memory(7_000_000_000, sequence_length=0)

    def test_zero_layers_raises(self) -> None:
        """Zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_training_memory(7_000_000_000, num_layers=0)

    def test_zero_hidden_raises(self) -> None:
        """Zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_training_memory(7_000_000_000, hidden_size=0)

    def test_zero_dtype_raises(self) -> None:
        """Zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_training_memory(7_000_000_000, dtype_bytes=0)


class TestFindMaxBatchSize:
    """Tests for find_max_batch_size function."""

    def test_basic_find(self) -> None:
        """Find max batch size."""
        batch_size = find_max_batch_size(7_000_000_000, 24.0)
        assert batch_size >= 1

    def test_more_memory_larger_batch(self) -> None:
        """More memory allows larger batch."""
        batch_24 = find_max_batch_size(7_000_000_000, 24.0)
        batch_80 = find_max_batch_size(7_000_000_000, 80.0)
        assert batch_80 >= batch_24

    def test_zero_params_raises(self) -> None:
        """Zero parameters raises ValueError."""
        with pytest.raises(ValueError, match="num_parameters must be positive"):
            find_max_batch_size(0, 24.0)

    def test_zero_memory_raises(self) -> None:
        """Zero GPU memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb must be positive"):
            find_max_batch_size(7_000_000_000, 0.0)

    def test_invalid_fraction_raises(self) -> None:
        """Invalid memory fraction raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction must be between"):
            find_max_batch_size(7_000_000_000, 24.0, memory_fraction=0.0)

    def test_fraction_above_one_raises(self) -> None:
        """Fraction above 1 raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction must be between"):
            find_max_batch_size(7_000_000_000, 24.0, memory_fraction=1.1)


class TestDetectMemoryBottleneck:
    """Tests for detect_memory_bottleneck function."""

    def test_no_bottleneck(self) -> None:
        """No bottleneck when under budget."""
        estimate = create_memory_estimate(5000.0, 1000.0, 500.0)
        budget = create_memory_budget(gpu_memory_gb=24.0)
        assert detect_memory_bottleneck(estimate, budget) == "none"

    def test_parameters_bottleneck(self) -> None:
        """Parameters are the bottleneck."""
        estimate = create_memory_estimate(20000.0, 1000.0, 500.0)
        budget = create_memory_budget(gpu_memory_gb=20.0)
        assert detect_memory_bottleneck(estimate, budget) == "parameters"

    def test_kv_cache_bottleneck(self) -> None:
        """KV cache is the bottleneck."""
        estimate = create_memory_estimate(5000.0, 1000.0, 20000.0)
        budget = create_memory_budget(gpu_memory_gb=24.0)
        assert detect_memory_bottleneck(estimate, budget) == "kv_cache"

    def test_activations_bottleneck(self) -> None:
        """Activations are the bottleneck."""
        estimate = create_memory_estimate(5000.0, 20000.0, 500.0)
        budget = create_memory_budget(gpu_memory_gb=24.0)
        assert detect_memory_bottleneck(estimate, budget) == "activations"

    def test_total_bottleneck(self) -> None:
        """Total is the bottleneck when no component dominates."""
        # Create an estimate where total exceeds budget but no single
        # component > 50% of budget. Budget is 8GB = 8192 MB, so 50% = 4096 MB.
        # Each component should be < 4096 but total > 8192.
        estimate = create_memory_estimate(3000.0, 3000.0, 3000.0, 9000.0)
        budget = create_memory_budget(gpu_memory_gb=8.0)
        assert detect_memory_bottleneck(estimate, budget) == "total"


class TestFormatMemoryStats:
    """Tests for format_memory_stats function."""

    def test_format_contains_peak(self) -> None:
        """Formatted string contains peak memory."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        formatted = format_memory_stats(stats)
        assert "Peak Memory: 20000.00 MB" in formatted

    def test_format_contains_utilization(self) -> None:
        """Formatted string contains utilization."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        formatted = format_memory_stats(stats)
        assert "Utilization: 82.00%" in formatted

    def test_format_contains_allocated(self) -> None:
        """Formatted string contains allocated memory."""
        stats = MemoryStats(20000.0, 18000.0, 22000.0, 0.82)
        formatted = format_memory_stats(stats)
        assert "Allocated: 18000.00 MB" in formatted


class TestFormatMemoryEstimate:
    """Tests for format_memory_estimate function."""

    def test_format_contains_parameters(self) -> None:
        """Formatted string contains parameters."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        formatted = format_memory_estimate(estimate)
        assert "Parameters: 14000.00 MB" in formatted

    def test_format_contains_total(self) -> None:
        """Formatted string contains total."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        formatted = format_memory_estimate(estimate)
        assert "Total: 17000.00 MB" in formatted

    def test_format_contains_kv_cache(self) -> None:
        """Formatted string contains KV cache."""
        estimate = MemoryEstimate(14000.0, 2000.0, 1000.0, 17000.0)
        formatted = format_memory_estimate(estimate)
        assert "KV Cache: 1000.00 MB" in formatted


class TestGetRecommendedMemoryConfig:
    """Tests for get_recommended_memory_config function."""

    def test_small_inference(self) -> None:
        """Recommended config for small model inference."""
        config = get_recommended_memory_config("small", 24.0)
        assert config.batch_size >= 1
        assert config.sequence_length > 0

    def test_large_training(self) -> None:
        """Recommended config for large model training."""
        config = get_recommended_memory_config("large", 24.0, "training")
        assert (
            OptimizationStrategy.GRADIENT_CHECKPOINTING
            in config.optimization_strategies
        )

    def test_xlarge_inference(self) -> None:
        """Recommended config for xlarge model inference."""
        config = get_recommended_memory_config("xlarge", 24.0)
        assert OptimizationStrategy.QUANTIZATION in config.optimization_strategies

    def test_invalid_size_raises(self) -> None:
        """Invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_memory_config("invalid", 24.0)

    def test_invalid_use_case_raises(self) -> None:
        """Invalid use case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_memory_config("small", 24.0, "invalid")  # type: ignore[arg-type]

    def test_zero_memory_raises(self) -> None:
        """Zero GPU memory raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_gb must be positive"):
            get_recommended_memory_config("small", 0.0)

    def test_xlarge_training(self) -> None:
        """Recommended config for xlarge model training."""
        config = get_recommended_memory_config("xlarge", 24.0, "training")
        assert (
            OptimizationStrategy.GRADIENT_CHECKPOINTING
            in config.optimization_strategies
        )
        assert OptimizationStrategy.CPU_OFFLOAD in config.optimization_strategies

    def test_medium_training_low_memory(self) -> None:
        """Recommended config for medium model training with low memory."""
        config = get_recommended_memory_config("medium", 24.0, "training")
        # medium model with < 48 GB should get cpu_offload
        assert OptimizationStrategy.CPU_OFFLOAD in config.optimization_strategies

    def test_large_inference_low_memory(self) -> None:
        """Recommended config for large model inference with low GPU memory."""
        config = get_recommended_memory_config("large", 16.0)
        assert OptimizationStrategy.QUANTIZATION in config.optimization_strategies

    def test_medium_inference_high_memory(self) -> None:
        """Recommended config for medium model inference with high GPU memory."""
        config = get_recommended_memory_config("medium", 80.0)
        # High memory should get longer sequence length
        assert config.sequence_length == 4096

    def test_xlarge_training_high_memory(self) -> None:
        """Recommended config for xlarge model training with high GPU memory."""
        # xlarge training with 80GB GPU should still get cpu_offload due to xlarge
        # but hits the "xlarge" part of the condition, not the "< 48" part
        config = get_recommended_memory_config("xlarge", 80.0, "training")
        assert (
            OptimizationStrategy.GRADIENT_CHECKPOINTING
            in config.optimization_strategies
        )
        # xlarge always gets cpu_offload, regardless of memory
        assert OptimizationStrategy.CPU_OFFLOAD in config.optimization_strategies


class TestConvertMemoryUnits:
    """Tests for convert_memory_units function."""

    def test_mb_to_gb(self) -> None:
        """Convert MB to GB."""
        assert convert_memory_units(1024, "mb", "gb") == 1.0

    def test_gb_to_mb(self) -> None:
        """Convert GB to MB."""
        assert convert_memory_units(1, "gb", "mb") == 1024.0

    def test_kb_to_bytes(self) -> None:
        """Convert KB to bytes."""
        assert convert_memory_units(1, "kb", "bytes") == 1024.0

    def test_bytes_to_kb(self) -> None:
        """Convert bytes to KB."""
        assert convert_memory_units(1024, "bytes", "kb") == 1.0

    def test_same_unit(self) -> None:
        """Same unit returns same value."""
        assert convert_memory_units(100, "mb", "mb") == 100.0

    def test_invalid_from_unit_raises(self) -> None:
        """Invalid from_unit raises ValueError."""
        with pytest.raises(ValueError, match="from_unit must be one of"):
            convert_memory_units(100, "invalid", "gb")

    def test_invalid_to_unit_raises(self) -> None:
        """Invalid to_unit raises ValueError."""
        with pytest.raises(ValueError, match="to_unit must be one of"):
            convert_memory_units(100, "mb", "invalid")


class TestHypothesis:
    """Property-based tests using Hypothesis."""

    @given(st.floats(min_value=0.0, max_value=100000.0))
    @settings(max_examples=50)
    def test_valid_parameters_mb_accepted(self, parameters_mb: float) -> None:
        """Valid parameters_mb values are accepted."""
        estimate = create_memory_estimate(parameters_mb)
        assert estimate.parameters_mb == parameters_mb

    @given(st.floats(min_value=0.0, max_value=1000.0))
    @settings(max_examples=50)
    def test_valid_gpu_memory_accepted(self, gpu_memory_gb: float) -> None:
        """Valid GPU memory values are accepted."""
        budget = create_memory_budget(gpu_memory_gb=gpu_memory_gb)
        assert budget.gpu_memory_gb == gpu_memory_gb

    @given(st.integers(min_value=1, max_value=128))
    @settings(max_examples=50)
    def test_valid_batch_size_accepted(self, batch_size: int) -> None:
        """Valid batch sizes are accepted."""
        config = create_memory_config(batch_size=batch_size)
        assert config.batch_size == batch_size

    @given(st.integers(min_value=1, max_value=32768))
    @settings(max_examples=50)
    def test_valid_sequence_length_accepted(self, sequence_length: int) -> None:
        """Valid sequence lengths are accepted."""
        config = create_memory_config(sequence_length=sequence_length)
        assert config.sequence_length == sequence_length

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=30)
    def test_valid_utilization_accepted(self, utilization: float) -> None:
        """Valid utilization values are accepted."""
        stats = create_memory_stats(1000.0, 500.0, 1000.0, utilization=utilization)
        assert stats.utilization == utilization

    @given(st.sampled_from(list(MemoryComponent)))
    def test_all_memory_components_retrievable(
        self, component: MemoryComponent
    ) -> None:
        """All memory components can be retrieved by name."""
        result = get_memory_component(component.value)
        assert result == component

    @given(st.sampled_from(list(MemoryUnit)))
    def test_all_memory_units_retrievable(self, unit: MemoryUnit) -> None:
        """All memory units can be retrieved by name."""
        result = get_memory_unit(unit.value)
        assert result == unit

    @given(st.sampled_from(list(OptimizationStrategy)))
    def test_all_strategies_retrievable(self, strategy: OptimizationStrategy) -> None:
        """All optimization strategies can be retrieved by name."""
        result = get_optimization_strategy(strategy.value)
        assert result == strategy

    @given(
        st.integers(min_value=1, max_value=100_000_000_000),
        st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=50)
    def test_model_memory_scales_with_params(
        self, num_parameters: int, dtype_bytes: int
    ) -> None:
        """Model memory scales linearly with parameters."""
        mem = estimate_model_memory(num_parameters, dtype_bytes)
        expected = (num_parameters * dtype_bytes) / (1024 * 1024)
        assert abs(mem - expected) < 0.01
