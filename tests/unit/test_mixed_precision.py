"""Tests for training.mixed_precision module."""

from __future__ import annotations

import pytest

from hf_gtc.training.mixed_precision import (
    VALID_CASTING_POLICIES,
    VALID_PRECISION_TYPES,
    VALID_SCALING_STRATEGIES,
    CastingPolicy,
    MixedPrecisionConfig,
    PrecisionConfig,
    PrecisionStats,
    PrecisionType,
    ScalerConfig,
    ScalingStrategy,
    calculate_memory_reduction,
    calculate_optimal_scale,
    check_overflow_risk,
    create_mixed_precision_config,
    create_precision_config,
    create_scaler_config,
    estimate_speedup,
    format_precision_stats,
    get_casting_policy,
    get_precision_type,
    get_recommended_precision_config,
    get_scaling_strategy,
    list_casting_policies,
    list_precision_types,
    list_scaling_strategies,
    validate_mixed_precision_config,
    validate_precision_config,
    validate_scaler_config,
)


class TestPrecisionType:
    """Tests for PrecisionType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for pt in PrecisionType:
            assert isinstance(pt.value, str)

    def test_fp16_value(self) -> None:
        """FP16 has correct value."""
        assert PrecisionType.FP16.value == "fp16"

    def test_bf16_value(self) -> None:
        """BF16 has correct value."""
        assert PrecisionType.BF16.value == "bf16"

    def test_fp8_e4m3_value(self) -> None:
        """FP8_E4M3 has correct value."""
        assert PrecisionType.FP8_E4M3.value == "fp8_e4m3"

    def test_fp8_e5m2_value(self) -> None:
        """FP8_E5M2 has correct value."""
        assert PrecisionType.FP8_E5M2.value == "fp8_e5m2"

    def test_fp32_value(self) -> None:
        """FP32 has correct value."""
        assert PrecisionType.FP32.value == "fp32"

    def test_valid_precision_types_frozenset(self) -> None:
        """VALID_PRECISION_TYPES is a frozenset."""
        assert isinstance(VALID_PRECISION_TYPES, frozenset)
        assert len(VALID_PRECISION_TYPES) == 5


class TestScalingStrategy:
    """Tests for ScalingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ScalingStrategy:
            assert isinstance(strategy.value, str)

    def test_dynamic_value(self) -> None:
        """Dynamic has correct value."""
        assert ScalingStrategy.DYNAMIC.value == "dynamic"

    def test_static_value(self) -> None:
        """Static has correct value."""
        assert ScalingStrategy.STATIC.value == "static"

    def test_loss_scale_value(self) -> None:
        """Loss scale has correct value."""
        assert ScalingStrategy.LOSS_SCALE.value == "loss_scale"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_SCALING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_SCALING_STRATEGIES, frozenset)
        assert len(VALID_SCALING_STRATEGIES) == 3


class TestCastingPolicy:
    """Tests for CastingPolicy enum."""

    def test_all_policies_have_values(self) -> None:
        """All policies have string values."""
        for policy in CastingPolicy:
            assert isinstance(policy.value, str)

    def test_all_value(self) -> None:
        """All has correct value."""
        assert CastingPolicy.ALL.value == "all"

    def test_compute_only_value(self) -> None:
        """Compute only has correct value."""
        assert CastingPolicy.COMPUTE_ONLY.value == "compute_only"

    def test_gradients_value(self) -> None:
        """Gradients has correct value."""
        assert CastingPolicy.GRADIENTS.value == "gradients"

    def test_valid_policies_frozenset(self) -> None:
        """VALID_CASTING_POLICIES is a frozenset."""
        assert isinstance(VALID_CASTING_POLICIES, frozenset)
        assert len(VALID_CASTING_POLICIES) == 3


class TestPrecisionConfig:
    """Tests for PrecisionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create precision config."""
        config = PrecisionConfig(
            dtype=PrecisionType.FP16,
            compute_dtype=PrecisionType.FP16,
            storage_dtype=PrecisionType.FP32,
        )
        assert config.dtype == PrecisionType.FP16
        assert config.compute_dtype == PrecisionType.FP16
        assert config.storage_dtype == PrecisionType.FP32

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PrecisionConfig(
            dtype=PrecisionType.FP16,
            compute_dtype=PrecisionType.FP16,
            storage_dtype=PrecisionType.FP32,
        )
        with pytest.raises(AttributeError):
            config.dtype = PrecisionType.BF16  # type: ignore[misc]


class TestScalerConfig:
    """Tests for ScalerConfig dataclass."""

    def test_create_config(self) -> None:
        """Create scaler config."""
        config = ScalerConfig(
            strategy=ScalingStrategy.DYNAMIC,
            initial_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            max_scale=2**24,
            min_scale=1.0,
        )
        assert config.strategy == ScalingStrategy.DYNAMIC
        assert config.initial_scale == 65536.0
        assert config.growth_factor == 2.0

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ScalerConfig(
            strategy=ScalingStrategy.DYNAMIC,
            initial_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            max_scale=2**24,
            min_scale=1.0,
        )
        with pytest.raises(AttributeError):
            config.strategy = ScalingStrategy.STATIC  # type: ignore[misc]


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create mixed precision config."""
        precision = PrecisionConfig(
            PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP32
        )
        scaler = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 2**24, 1.0
        )
        config = MixedPrecisionConfig(
            precision_config=precision,
            scaler_config=scaler,
            casting_policy=CastingPolicy.ALL,
            enabled=True,
            autocast_enabled=True,
        )
        assert config.enabled is True
        assert config.casting_policy == CastingPolicy.ALL


class TestPrecisionStats:
    """Tests for PrecisionStats dataclass."""

    def test_create_stats(self) -> None:
        """Create precision stats."""
        stats = PrecisionStats(
            current_scale=65536.0,
            num_overflows=10,
            num_scale_updates=5,
            overflow_rate=0.001,
            memory_reduction_pct=45.0,
            throughput_improvement_pct=30.0,
        )
        assert stats.current_scale == 65536.0
        assert stats.num_overflows == 10


class TestValidatePrecisionConfig:
    """Tests for validate_precision_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PrecisionConfig(
            PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP32
        )
        validate_precision_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_precision_config(None)  # type: ignore[arg-type]

    def test_fp8_storage_raises(self) -> None:
        """FP8 storage raises ValueError."""
        config = PrecisionConfig(
            PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP8_E4M3
        )
        with pytest.raises(ValueError, match="FP8 types cannot be used for storage"):
            validate_precision_config(config)

    def test_fp8_e5m2_storage_raises(self) -> None:
        """FP8_E5M2 storage raises ValueError."""
        config = PrecisionConfig(
            PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP8_E5M2
        )
        with pytest.raises(ValueError, match="FP8 types cannot be used for storage"):
            validate_precision_config(config)


class TestValidateScalerConfig:
    """Tests for validate_scaler_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 2**24, 1.0
        )
        validate_scaler_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_scaler_config(None)  # type: ignore[arg-type]

    def test_zero_initial_scale_raises(self) -> None:
        """Zero initial scale raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 0.0, 2.0, 0.5, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="initial_scale must be positive"):
            validate_scaler_config(config)

    def test_negative_initial_scale_raises(self) -> None:
        """Negative initial scale raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, -1.0, 2.0, 0.5, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="initial_scale must be positive"):
            validate_scaler_config(config)

    def test_growth_factor_one_raises(self) -> None:
        """Growth factor of 1 raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 1.0, 0.5, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="growth_factor must be greater than 1"):
            validate_scaler_config(config)

    def test_growth_factor_less_than_one_raises(self) -> None:
        """Growth factor less than 1 raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 0.5, 0.5, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="growth_factor must be greater than 1"):
            validate_scaler_config(config)

    def test_backoff_factor_zero_raises(self) -> None:
        """Backoff factor of 0 raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.0, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="backoff_factor must be between 0 and 1"):
            validate_scaler_config(config)

    def test_backoff_factor_one_raises(self) -> None:
        """Backoff factor of 1 raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 1.0, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="backoff_factor must be between 0 and 1"):
            validate_scaler_config(config)

    def test_backoff_factor_greater_than_one_raises(self) -> None:
        """Backoff factor > 1 raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 1.5, 2000, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="backoff_factor must be between 0 and 1"):
            validate_scaler_config(config)

    def test_zero_growth_interval_raises(self) -> None:
        """Zero growth interval raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 0, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="growth_interval must be positive"):
            validate_scaler_config(config)

    def test_negative_growth_interval_raises(self) -> None:
        """Negative growth interval raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, -100, 2**24, 1.0
        )
        with pytest.raises(ValueError, match="growth_interval must be positive"):
            validate_scaler_config(config)

    def test_max_less_than_min_raises(self) -> None:
        """Max scale less than min raises ValueError."""
        config = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 1.0, 100.0
        )
        with pytest.raises(ValueError, match="max_scale must be >= min_scale"):
            validate_scaler_config(config)


class TestValidateMixedPrecisionConfig:
    """Tests for validate_mixed_precision_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        precision = PrecisionConfig(
            PrecisionType.FP16, PrecisionType.FP16, PrecisionType.FP32
        )
        scaler = ScalerConfig(
            ScalingStrategy.DYNAMIC, 65536.0, 2.0, 0.5, 2000, 2**24, 1.0
        )
        config = MixedPrecisionConfig(
            precision, scaler, CastingPolicy.ALL, True, True
        )
        validate_mixed_precision_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_mixed_precision_config(None)  # type: ignore[arg-type]


class TestCreatePrecisionConfig:
    """Tests for create_precision_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_precision_config()
        assert config.dtype == PrecisionType.FP16
        assert config.compute_dtype == PrecisionType.FP16
        assert config.storage_dtype == PrecisionType.FP32

    def test_bf16_config(self) -> None:
        """Create BF16 config."""
        config = create_precision_config(dtype="bf16")
        assert config.dtype == PrecisionType.BF16

    def test_fp8_config(self) -> None:
        """Create FP8 config."""
        config = create_precision_config(dtype="fp8_e4m3", compute_dtype="fp8_e4m3")
        assert config.dtype == PrecisionType.FP8_E4M3

    def test_different_compute_dtype(self) -> None:
        """Create config with different compute dtype."""
        config = create_precision_config(dtype="fp32", compute_dtype="fp16")
        assert config.dtype == PrecisionType.FP32
        assert config.compute_dtype == PrecisionType.FP16

    def test_invalid_dtype_raises(self) -> None:
        """Invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be one of"):
            create_precision_config(dtype="invalid")

    def test_invalid_compute_dtype_raises(self) -> None:
        """Invalid compute dtype raises ValueError."""
        with pytest.raises(ValueError, match="compute_dtype must be one of"):
            create_precision_config(dtype="fp16", compute_dtype="invalid")

    def test_invalid_storage_dtype_raises(self) -> None:
        """Invalid storage dtype raises ValueError."""
        with pytest.raises(ValueError, match="storage_dtype must be one of"):
            create_precision_config(dtype="fp16", storage_dtype="invalid")


class TestCreateScalerConfig:
    """Tests for create_scaler_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_scaler_config()
        assert config.strategy == ScalingStrategy.DYNAMIC
        assert config.initial_scale == 65536.0

    def test_static_config(self) -> None:
        """Create static config."""
        config = create_scaler_config(strategy="static", initial_scale=1024.0)
        assert config.strategy == ScalingStrategy.STATIC
        assert config.initial_scale == 1024.0

    def test_custom_growth_factor(self) -> None:
        """Create config with custom growth factor."""
        config = create_scaler_config(growth_factor=4.0)
        assert config.growth_factor == 4.0

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_scaler_config(strategy="invalid")


class TestCreateMixedPrecisionConfig:
    """Tests for create_mixed_precision_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_mixed_precision_config()
        assert config.enabled is True
        assert config.precision_config.dtype == PrecisionType.FP16

    def test_custom_precision_config(self) -> None:
        """Create config with custom precision."""
        precision = create_precision_config(dtype="bf16")
        config = create_mixed_precision_config(precision_config=precision)
        assert config.precision_config.dtype == PrecisionType.BF16

    def test_custom_scaler_config(self) -> None:
        """Create config with custom scaler."""
        scaler = create_scaler_config(strategy="static", initial_scale=1.0)
        config = create_mixed_precision_config(scaler_config=scaler)
        assert config.scaler_config.strategy == ScalingStrategy.STATIC

    def test_disabled_config(self) -> None:
        """Create disabled config."""
        config = create_mixed_precision_config(enabled=False)
        assert config.enabled is False

    def test_invalid_casting_policy_raises(self) -> None:
        """Invalid casting policy raises ValueError."""
        with pytest.raises(ValueError, match="casting_policy must be one of"):
            create_mixed_precision_config(casting_policy="invalid")

    def test_compute_only_casting(self) -> None:
        """Create config with compute_only casting."""
        config = create_mixed_precision_config(casting_policy="compute_only")
        assert config.casting_policy == CastingPolicy.COMPUTE_ONLY

    def test_gradients_casting(self) -> None:
        """Create config with gradients casting."""
        config = create_mixed_precision_config(casting_policy="gradients")
        assert config.casting_policy == CastingPolicy.GRADIENTS


class TestCalculateMemoryReduction:
    """Tests for calculate_memory_reduction function."""

    def test_fp32_to_fp16(self) -> None:
        """Calculate FP32 to FP16 reduction."""
        saved, pct = calculate_memory_reduction("fp32", "fp16", 7.0)
        assert saved == 14.0
        assert pct == 50.0

    def test_fp32_to_bf16(self) -> None:
        """Calculate FP32 to BF16 reduction."""
        _, pct = calculate_memory_reduction("fp32", "bf16", 7.0)
        assert pct == 50.0

    def test_fp32_to_fp8(self) -> None:
        """Calculate FP32 to FP8 reduction."""
        _, pct = calculate_memory_reduction("fp32", "fp8_e4m3", 7.0)
        assert pct == 75.0

    def test_fp16_to_fp8(self) -> None:
        """Calculate FP16 to FP8 reduction."""
        _, pct = calculate_memory_reduction("fp16", "fp8_e4m3", 7.0)
        assert pct == 50.0

    def test_same_precision_no_reduction(self) -> None:
        """Same precision has no reduction."""
        saved, pct = calculate_memory_reduction("fp16", "fp16", 7.0)
        assert saved == 0.0
        assert pct == 0.0

    def test_invalid_original_dtype_raises(self) -> None:
        """Invalid original dtype raises ValueError."""
        with pytest.raises(ValueError, match="original_dtype must be one of"):
            calculate_memory_reduction("invalid", "fp16", 7.0)

    def test_invalid_target_dtype_raises(self) -> None:
        """Invalid target dtype raises ValueError."""
        with pytest.raises(ValueError, match="target_dtype must be one of"):
            calculate_memory_reduction("fp32", "invalid", 7.0)

    def test_zero_model_size_raises(self) -> None:
        """Zero model size raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            calculate_memory_reduction("fp32", "fp16", 0)

    def test_negative_model_size_raises(self) -> None:
        """Negative model size raises ValueError."""
        with pytest.raises(ValueError, match="model_params_billions must be positive"):
            calculate_memory_reduction("fp32", "fp16", -1.0)


class TestEstimateSpeedup:
    """Tests for estimate_speedup function."""

    def test_fp32_to_fp16_ampere(self) -> None:
        """FP32 to FP16 on Ampere."""
        speedup = estimate_speedup("fp32", "fp16", "ampere")
        assert speedup >= 1.5

    def test_fp32_to_bf16_ampere(self) -> None:
        """FP32 to BF16 on Ampere."""
        speedup = estimate_speedup("fp32", "bf16", "ampere")
        assert speedup >= 1.5

    def test_fp32_to_fp8_hopper(self) -> None:
        """FP32 to FP8 on Hopper."""
        speedup = estimate_speedup("fp32", "fp8_e4m3", "hopper")
        assert speedup > 2.0

    def test_unknown_architecture(self) -> None:
        """Unknown architecture uses default multiplier."""
        speedup = estimate_speedup("fp32", "fp16", "unknown_arch")
        assert speedup > 1.0

    def test_invalid_original_dtype_raises(self) -> None:
        """Invalid original dtype raises ValueError."""
        with pytest.raises(ValueError, match="original_dtype must be one of"):
            estimate_speedup("invalid", "fp16", "ampere")

    def test_invalid_target_dtype_raises(self) -> None:
        """Invalid target dtype raises ValueError."""
        with pytest.raises(ValueError, match="target_dtype must be one of"):
            estimate_speedup("fp32", "invalid", "ampere")

    def test_volta_architecture(self) -> None:
        """Volta architecture has lower speedup."""
        speedup_volta = estimate_speedup("fp32", "fp16", "volta")
        speedup_ampere = estimate_speedup("fp32", "fp16", "ampere")
        assert speedup_volta < speedup_ampere

    def test_turing_architecture(self) -> None:
        """Turing architecture speedup."""
        speedup = estimate_speedup("fp32", "fp16", "turing")
        assert speedup > 1.0

    def test_blackwell_architecture(self) -> None:
        """Blackwell architecture has higher speedup."""
        speedup_blackwell = estimate_speedup("fp32", "fp16", "blackwell")
        speedup_ampere = estimate_speedup("fp32", "fp16", "ampere")
        assert speedup_blackwell > speedup_ampere


class TestCheckOverflowRisk:
    """Tests for check_overflow_risk function."""

    def test_no_overflow_low_gradient(self) -> None:
        """Low gradient has no overflow."""
        overflow, headroom = check_overflow_risk(0.1, 1000.0, "fp16")
        assert overflow is False
        assert headroom > 1.0

    def test_high_overflow_risk(self) -> None:
        """High gradient has low headroom."""
        _, headroom = check_overflow_risk(100.0, 65536.0, "fp16")
        assert headroom < 100.0

    def test_overflow_detected(self) -> None:
        """Overflow is detected for extreme values."""
        overflow, _ = check_overflow_risk(1000.0, 65536.0, "fp16")
        assert overflow is True

    def test_fp8_lower_max(self) -> None:
        """FP8 has lower max value."""
        _, headroom_fp16 = check_overflow_risk(1.0, 1024.0, "fp16")
        _, headroom_fp8 = check_overflow_risk(1.0, 1024.0, "fp8_e4m3")
        assert headroom_fp16 > headroom_fp8

    def test_negative_gradient_raises(self) -> None:
        """Negative gradient raises ValueError."""
        with pytest.raises(ValueError, match="gradient_norm cannot be negative"):
            check_overflow_risk(-1.0, 65536.0)

    def test_zero_loss_scale_raises(self) -> None:
        """Zero loss scale raises ValueError."""
        with pytest.raises(ValueError, match="loss_scale must be positive"):
            check_overflow_risk(1.0, 0.0)

    def test_negative_loss_scale_raises(self) -> None:
        """Negative loss scale raises ValueError."""
        with pytest.raises(ValueError, match="loss_scale must be positive"):
            check_overflow_risk(1.0, -1.0)

    def test_zero_gradient_infinite_headroom(self) -> None:
        """Zero gradient has infinite headroom."""
        overflow, headroom = check_overflow_risk(0.0, 65536.0, "fp16")
        assert overflow is False
        assert headroom == float("inf")

    def test_bf16_higher_max(self) -> None:
        """BF16 has higher max value than FP16."""
        _, headroom_fp16 = check_overflow_risk(1.0, 1000.0, "fp16")
        _, headroom_bf16 = check_overflow_risk(1.0, 1000.0, "bf16")
        assert headroom_bf16 > headroom_fp16

    def test_fp8_e5m2_max(self) -> None:
        """FP8_E5M2 has specific max value."""
        _, headroom = check_overflow_risk(1.0, 100.0, "fp8_e5m2")
        assert headroom > 0


class TestCalculateOptimalScale:
    """Tests for calculate_optimal_scale function."""

    def test_small_gradient(self) -> None:
        """Small gradient results in high scale."""
        scale = calculate_optimal_scale(0.01, "fp16", 10.0)
        assert scale > 65536.0

    def test_large_gradient(self) -> None:
        """Large gradient results in lower scale."""
        scale = calculate_optimal_scale(100.0, "fp16", 10.0)
        assert scale < 65536.0

    def test_zero_gradient_raises(self) -> None:
        """Zero gradient raises ValueError."""
        with pytest.raises(ValueError, match="gradient_norm must be positive"):
            calculate_optimal_scale(0, "fp16")

    def test_negative_gradient_raises(self) -> None:
        """Negative gradient raises ValueError."""
        with pytest.raises(ValueError, match="gradient_norm must be positive"):
            calculate_optimal_scale(-1.0, "fp16")

    def test_zero_headroom_raises(self) -> None:
        """Zero headroom raises ValueError."""
        with pytest.raises(ValueError, match="target_headroom must be positive"):
            calculate_optimal_scale(1.0, "fp16", 0.0)

    def test_negative_headroom_raises(self) -> None:
        """Negative headroom raises ValueError."""
        with pytest.raises(ValueError, match="target_headroom must be positive"):
            calculate_optimal_scale(1.0, "fp16", -1.0)

    def test_scale_clamped_to_max(self) -> None:
        """Scale is clamped to max value."""
        scale = calculate_optimal_scale(0.000001, "fp16", 1.0)
        assert scale <= 2**24

    def test_scale_clamped_to_min(self) -> None:
        """Scale is clamped to min value."""
        scale = calculate_optimal_scale(100000.0, "fp16", 100.0)
        assert scale >= 1.0


class TestFormatPrecisionStats:
    """Tests for format_precision_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = PrecisionStats(65536.0, 10, 5, 0.001, 45.0, 30.0)
        formatted = format_precision_stats(stats)
        assert "Current Scale:" in formatted
        assert "65,536" in formatted
        assert "Overflow Count: 10" in formatted

    def test_format_contains_all_fields(self) -> None:
        """Format contains all fields."""
        stats = PrecisionStats(32768.0, 5, 3, 0.002, 50.0, 25.0)
        formatted = format_precision_stats(stats)
        assert "Scale Updates:" in formatted
        assert "Overflow Rate:" in formatted
        assert "Memory Reduction:" in formatted
        assert "Throughput Improvement:" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_precision_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedPrecisionConfig:
    """Tests for get_recommended_precision_config function."""

    def test_7b_ampere(self) -> None:
        """Get config for 7B on Ampere."""
        config = get_recommended_precision_config("7b", "ampere")
        assert config.precision_config.dtype in (PrecisionType.FP16, PrecisionType.BF16)

    def test_70b_hopper(self) -> None:
        """Get config for 70B on Hopper."""
        config = get_recommended_precision_config("70b", "hopper")
        assert config.enabled is True

    def test_inference_use_case(self) -> None:
        """Inference uses static scaling."""
        config = get_recommended_precision_config("7b", "ampere", "inference")
        assert config.scaler_config.strategy == ScalingStrategy.STATIC

    def test_training_use_case(self) -> None:
        """Training uses dynamic scaling."""
        config = get_recommended_precision_config("7b", "ampere", "training")
        assert config.scaler_config.strategy == ScalingStrategy.DYNAMIC

    def test_invalid_use_case_raises(self) -> None:
        """Invalid use case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_precision_config("7b", "ampere", "invalid")

    def test_hopper_fp8_for_large_models(self) -> None:
        """Hopper uses FP8 for large models."""
        config = get_recommended_precision_config("70b", "hopper")
        assert config.precision_config.dtype == PrecisionType.FP8_E4M3

    def test_blackwell_fp8_for_large_models(self) -> None:
        """Blackwell uses FP8 for large models."""
        config = get_recommended_precision_config("175b", "blackwell")
        assert config.precision_config.dtype == PrecisionType.FP8_E4M3

    def test_older_gpu_uses_fp16(self) -> None:
        """Older GPUs use FP16."""
        config = get_recommended_precision_config("7b", "volta")
        assert config.precision_config.dtype == PrecisionType.FP16

    def test_hopper_small_model_uses_bf16(self) -> None:
        """Hopper uses BF16 for small models."""
        config = get_recommended_precision_config("7b", "hopper")
        assert config.precision_config.dtype == PrecisionType.BF16


class TestListPrecisionTypes:
    """Tests for list_precision_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_precision_types()
        assert types == sorted(types)

    def test_contains_fp16(self) -> None:
        """Contains fp16."""
        types = list_precision_types()
        assert "fp16" in types

    def test_contains_bf16(self) -> None:
        """Contains bf16."""
        types = list_precision_types()
        assert "bf16" in types

    def test_contains_fp8(self) -> None:
        """Contains fp8 types."""
        types = list_precision_types()
        assert "fp8_e4m3" in types
        assert "fp8_e5m2" in types


class TestListScalingStrategies:
    """Tests for list_scaling_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_scaling_strategies()
        assert strategies == sorted(strategies)

    def test_contains_dynamic(self) -> None:
        """Contains dynamic."""
        strategies = list_scaling_strategies()
        assert "dynamic" in strategies

    def test_contains_static(self) -> None:
        """Contains static."""
        strategies = list_scaling_strategies()
        assert "static" in strategies


class TestListCastingPolicies:
    """Tests for list_casting_policies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        policies = list_casting_policies()
        assert policies == sorted(policies)

    def test_contains_all(self) -> None:
        """Contains all."""
        policies = list_casting_policies()
        assert "all" in policies

    def test_contains_compute_only(self) -> None:
        """Contains compute_only."""
        policies = list_casting_policies()
        assert "compute_only" in policies


class TestGetPrecisionType:
    """Tests for get_precision_type function."""

    def test_get_fp16(self) -> None:
        """Get fp16."""
        assert get_precision_type("fp16") == PrecisionType.FP16

    def test_get_bf16(self) -> None:
        """Get bf16."""
        assert get_precision_type("bf16") == PrecisionType.BF16

    def test_get_fp8_e4m3(self) -> None:
        """Get fp8_e4m3."""
        assert get_precision_type("fp8_e4m3") == PrecisionType.FP8_E4M3

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="precision type must be one of"):
            get_precision_type("invalid")


class TestGetScalingStrategy:
    """Tests for get_scaling_strategy function."""

    def test_get_dynamic(self) -> None:
        """Get dynamic."""
        assert get_scaling_strategy("dynamic") == ScalingStrategy.DYNAMIC

    def test_get_static(self) -> None:
        """Get static."""
        assert get_scaling_strategy("static") == ScalingStrategy.STATIC

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="scaling strategy must be one of"):
            get_scaling_strategy("invalid")


class TestGetCastingPolicy:
    """Tests for get_casting_policy function."""

    def test_get_all(self) -> None:
        """Get all."""
        assert get_casting_policy("all") == CastingPolicy.ALL

    def test_get_compute_only(self) -> None:
        """Get compute_only."""
        assert get_casting_policy("compute_only") == CastingPolicy.COMPUTE_ONLY

    def test_get_gradients(self) -> None:
        """Get gradients."""
        assert get_casting_policy("gradients") == CastingPolicy.GRADIENTS

    def test_invalid_policy_raises(self) -> None:
        """Invalid policy raises ValueError."""
        with pytest.raises(ValueError, match="casting policy must be one of"):
            get_casting_policy("invalid")
