"""Tests for training.gradient module."""

from __future__ import annotations

import pytest

from hf_gtc.training.gradient import (
    VALID_ACCUMULATION_STRATEGIES,
    VALID_CLIPPING_METHODS,
    VALID_SCALING_METHODS,
    AccumulationConfig,
    AccumulationStrategy,
    ClippingConfig,
    ClippingMethod,
    GradientConfig,
    GradientStats,
    ScalingConfig,
    ScalingMethod,
    accumulate_gradients,
    calculate_grad_norm,
    clip_gradients,
    create_accumulation_config,
    create_clipping_config,
    create_gradient_config,
    create_gradient_stats,
    create_scaling_config,
    detect_overflow,
    format_gradient_stats,
    get_accumulation_strategy,
    get_clipping_method,
    get_recommended_gradient_config,
    get_scaling_method,
    list_accumulation_strategies,
    list_clipping_methods,
    list_scaling_methods,
    scale_gradients,
    validate_accumulation_config,
    validate_clipping_config,
    validate_gradient_config,
    validate_gradient_stats,
    validate_scaling_config,
)


class TestClippingMethod:
    """Tests for ClippingMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in ClippingMethod:
            assert isinstance(method.value, str)

    def test_norm_value(self) -> None:
        """Norm has correct value."""
        assert ClippingMethod.NORM.value == "norm"

    def test_value_value(self) -> None:
        """Value has correct value."""
        assert ClippingMethod.VALUE.value == "value"

    def test_adaptive_value(self) -> None:
        """Adaptive has correct value."""
        assert ClippingMethod.ADAPTIVE.value == "adaptive"

    def test_none_value(self) -> None:
        """None has correct value."""
        assert ClippingMethod.NONE.value == "none"

    def test_valid_clipping_methods_frozenset(self) -> None:
        """VALID_CLIPPING_METHODS is a frozenset."""
        assert isinstance(VALID_CLIPPING_METHODS, frozenset)
        assert len(VALID_CLIPPING_METHODS) == 4


class TestScalingMethod:
    """Tests for ScalingMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in ScalingMethod:
            assert isinstance(method.value, str)

    def test_static_value(self) -> None:
        """Static has correct value."""
        assert ScalingMethod.STATIC.value == "static"

    def test_dynamic_value(self) -> None:
        """Dynamic has correct value."""
        assert ScalingMethod.DYNAMIC.value == "dynamic"

    def test_loss_scale_value(self) -> None:
        """Loss scale has correct value."""
        assert ScalingMethod.LOSS_SCALE.value == "loss_scale"

    def test_valid_scaling_methods_frozenset(self) -> None:
        """VALID_SCALING_METHODS is a frozenset."""
        assert isinstance(VALID_SCALING_METHODS, frozenset)
        assert len(VALID_SCALING_METHODS) == 3


class TestAccumulationStrategy:
    """Tests for AccumulationStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in AccumulationStrategy:
            assert isinstance(strategy.value, str)

    def test_mean_value(self) -> None:
        """Mean has correct value."""
        assert AccumulationStrategy.MEAN.value == "mean"

    def test_sum_value(self) -> None:
        """Sum has correct value."""
        assert AccumulationStrategy.SUM.value == "sum"

    def test_weighted_value(self) -> None:
        """Weighted has correct value."""
        assert AccumulationStrategy.WEIGHTED.value == "weighted"

    def test_valid_accumulation_strategies_frozenset(self) -> None:
        """VALID_ACCUMULATION_STRATEGIES is a frozenset."""
        assert isinstance(VALID_ACCUMULATION_STRATEGIES, frozenset)
        assert len(VALID_ACCUMULATION_STRATEGIES) == 3


class TestClippingConfig:
    """Tests for ClippingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create clipping config."""
        config = ClippingConfig(
            method=ClippingMethod.NORM,
            max_norm=1.0,
            max_value=1.0,
            norm_type=2.0,
        )
        assert config.method == ClippingMethod.NORM
        assert config.max_norm == pytest.approx(1.0)
        assert config.max_value == pytest.approx(1.0)
        assert config.norm_type == pytest.approx(2.0)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ClippingConfig(
            method=ClippingMethod.NORM,
            max_norm=1.0,
            max_value=1.0,
            norm_type=2.0,
        )
        with pytest.raises(AttributeError):
            config.method = ClippingMethod.VALUE  # type: ignore[misc]


class TestScalingConfig:
    """Tests for ScalingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create scaling config."""
        config = ScalingConfig(
            method=ScalingMethod.DYNAMIC,
            initial_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
        )
        assert config.method == ScalingMethod.DYNAMIC
        assert config.initial_scale == pytest.approx(65536.0)
        assert config.growth_factor == pytest.approx(2.0)
        assert config.backoff_factor == pytest.approx(0.5)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ScalingConfig(
            method=ScalingMethod.DYNAMIC,
            initial_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
        )
        with pytest.raises(AttributeError):
            config.method = ScalingMethod.STATIC  # type: ignore[misc]


class TestAccumulationConfig:
    """Tests for AccumulationConfig dataclass."""

    def test_create_config(self) -> None:
        """Create accumulation config."""
        config = AccumulationConfig(
            steps=4,
            strategy=AccumulationStrategy.MEAN,
            sync_grads=True,
        )
        assert config.steps == 4
        assert config.strategy == AccumulationStrategy.MEAN
        assert config.sync_grads is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = AccumulationConfig(
            steps=4,
            strategy=AccumulationStrategy.MEAN,
            sync_grads=True,
        )
        with pytest.raises(AttributeError):
            config.steps = 8  # type: ignore[misc]


class TestGradientConfig:
    """Tests for GradientConfig dataclass."""

    def test_create_config(self) -> None:
        """Create gradient config."""
        clipping = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        scaling = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        accumulation = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        config = GradientConfig(clipping, scaling, accumulation)
        assert config.clipping_config.method == ClippingMethod.NORM
        assert config.scaling_config.method == ScalingMethod.DYNAMIC
        assert config.accumulation_config.steps == 4

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        clipping = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        scaling = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        accumulation = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        config = GradientConfig(clipping, scaling, accumulation)
        with pytest.raises(AttributeError):
            config.clipping_config = clipping  # type: ignore[misc]


class TestGradientStats:
    """Tests for GradientStats dataclass."""

    def test_create_stats(self) -> None:
        """Create gradient stats."""
        stats = GradientStats(
            grad_norm=0.5,
            clipped_ratio=0.1,
            overflow_count=2,
            effective_batch_size=32,
        )
        assert stats.grad_norm == pytest.approx(0.5)
        assert stats.clipped_ratio == pytest.approx(0.1)
        assert stats.overflow_count == 2
        assert stats.effective_batch_size == 32

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = GradientStats(
            grad_norm=0.5,
            clipped_ratio=0.1,
            overflow_count=2,
            effective_batch_size=32,
        )
        with pytest.raises(AttributeError):
            stats.grad_norm = 1.0  # type: ignore[misc]


class TestValidateClippingConfig:
    """Tests for validate_clipping_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        validate_clipping_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_clipping_config(None)  # type: ignore[arg-type]

    def test_zero_max_norm_raises(self) -> None:
        """Zero max_norm for norm clipping raises ValueError."""
        config = ClippingConfig(ClippingMethod.NORM, 0.0, 1.0, 2.0)
        with pytest.raises(ValueError, match="max_norm must be positive"):
            validate_clipping_config(config)

    def test_negative_max_norm_raises(self) -> None:
        """Negative max_norm for norm clipping raises ValueError."""
        config = ClippingConfig(ClippingMethod.NORM, -1.0, 1.0, 2.0)
        with pytest.raises(ValueError, match="max_norm must be positive"):
            validate_clipping_config(config)

    def test_zero_max_value_raises(self) -> None:
        """Zero max_value for value clipping raises ValueError."""
        config = ClippingConfig(ClippingMethod.VALUE, 1.0, 0.0, 2.0)
        with pytest.raises(ValueError, match="max_value must be positive"):
            validate_clipping_config(config)

    def test_negative_max_value_raises(self) -> None:
        """Negative max_value for value clipping raises ValueError."""
        config = ClippingConfig(ClippingMethod.VALUE, 1.0, -1.0, 2.0)
        with pytest.raises(ValueError, match="max_value must be positive"):
            validate_clipping_config(config)

    def test_zero_norm_type_raises(self) -> None:
        """Zero norm_type raises ValueError."""
        config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="norm_type must be positive"):
            validate_clipping_config(config)

    def test_negative_norm_type_raises(self) -> None:
        """Negative norm_type raises ValueError."""
        config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, -1.0)
        with pytest.raises(ValueError, match="norm_type must be positive"):
            validate_clipping_config(config)

    def test_none_clipping_valid(self) -> None:
        """None clipping method is valid with any max values."""
        config = ClippingConfig(ClippingMethod.NONE, 0.0, 0.0, 1.0)
        validate_clipping_config(config)


class TestValidateScalingConfig:
    """Tests for validate_scaling_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        validate_scaling_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_scaling_config(None)  # type: ignore[arg-type]

    def test_zero_initial_scale_raises(self) -> None:
        """Zero initial_scale raises ValueError."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, 0.0, 2.0, 0.5)
        with pytest.raises(ValueError, match="initial_scale must be positive"):
            validate_scaling_config(config)

    def test_negative_initial_scale_raises(self) -> None:
        """Negative initial_scale raises ValueError."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, -1.0, 2.0, 0.5)
        with pytest.raises(ValueError, match="initial_scale must be positive"):
            validate_scaling_config(config)

    def test_growth_factor_one_raises(self) -> None:
        """Growth factor of 1 raises ValueError for dynamic."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 1.0, 0.5)
        with pytest.raises(ValueError, match="growth_factor must be greater than 1"):
            validate_scaling_config(config)

    def test_growth_factor_less_than_one_raises(self) -> None:
        """Growth factor < 1 raises ValueError for dynamic."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 0.5, 0.5)
        with pytest.raises(ValueError, match="growth_factor must be greater than 1"):
            validate_scaling_config(config)

    def test_backoff_factor_zero_raises(self) -> None:
        """Backoff factor of 0 raises ValueError for dynamic."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.0)
        with pytest.raises(ValueError, match="backoff_factor must be between 0 and 1"):
            validate_scaling_config(config)

    def test_backoff_factor_one_raises(self) -> None:
        """Backoff factor of 1 raises ValueError for dynamic."""
        config = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 1.0)
        with pytest.raises(ValueError, match="backoff_factor must be between 0 and 1"):
            validate_scaling_config(config)

    def test_static_allows_any_growth_factor(self) -> None:
        """Static scaling allows any growth factor."""
        config = ScalingConfig(ScalingMethod.STATIC, 1.0, 0.5, 0.0)
        validate_scaling_config(config)


class TestValidateAccumulationConfig:
    """Tests for validate_accumulation_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        validate_accumulation_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_accumulation_config(None)  # type: ignore[arg-type]

    def test_zero_steps_raises(self) -> None:
        """Zero steps raises ValueError."""
        config = AccumulationConfig(0, AccumulationStrategy.MEAN, True)
        with pytest.raises(ValueError, match="steps must be positive"):
            validate_accumulation_config(config)

    def test_negative_steps_raises(self) -> None:
        """Negative steps raises ValueError."""
        config = AccumulationConfig(-1, AccumulationStrategy.MEAN, True)
        with pytest.raises(ValueError, match="steps must be positive"):
            validate_accumulation_config(config)


class TestValidateGradientConfig:
    """Tests for validate_gradient_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        clipping = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        scaling = ScalingConfig(ScalingMethod.DYNAMIC, 65536.0, 2.0, 0.5)
        accumulation = AccumulationConfig(4, AccumulationStrategy.MEAN, True)
        config = GradientConfig(clipping, scaling, accumulation)
        validate_gradient_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_gradient_config(None)  # type: ignore[arg-type]


class TestValidateGradientStats:
    """Tests for validate_gradient_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = GradientStats(0.5, 0.1, 2, 32)
        validate_gradient_stats(stats)

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_gradient_stats(None)  # type: ignore[arg-type]

    def test_negative_grad_norm_raises(self) -> None:
        """Negative grad_norm raises ValueError."""
        stats = GradientStats(-0.5, 0.1, 2, 32)
        with pytest.raises(ValueError, match="grad_norm cannot be negative"):
            validate_gradient_stats(stats)

    def test_clipped_ratio_less_than_zero_raises(self) -> None:
        """Clipped ratio < 0 raises ValueError."""
        stats = GradientStats(0.5, -0.1, 2, 32)
        with pytest.raises(ValueError, match="clipped_ratio must be in"):
            validate_gradient_stats(stats)

    def test_clipped_ratio_greater_than_one_raises(self) -> None:
        """Clipped ratio > 1 raises ValueError."""
        stats = GradientStats(0.5, 1.5, 2, 32)
        with pytest.raises(ValueError, match="clipped_ratio must be in"):
            validate_gradient_stats(stats)

    def test_negative_overflow_count_raises(self) -> None:
        """Negative overflow_count raises ValueError."""
        stats = GradientStats(0.5, 0.1, -1, 32)
        with pytest.raises(ValueError, match="overflow_count cannot be negative"):
            validate_gradient_stats(stats)

    def test_zero_effective_batch_size_raises(self) -> None:
        """Zero effective_batch_size raises ValueError."""
        stats = GradientStats(0.5, 0.1, 2, 0)
        with pytest.raises(ValueError, match="effective_batch_size must be positive"):
            validate_gradient_stats(stats)

    def test_negative_effective_batch_size_raises(self) -> None:
        """Negative effective_batch_size raises ValueError."""
        stats = GradientStats(0.5, 0.1, 2, -1)
        with pytest.raises(ValueError, match="effective_batch_size must be positive"):
            validate_gradient_stats(stats)


class TestCreateClippingConfig:
    """Tests for create_clipping_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_clipping_config()
        assert config.method == ClippingMethod.NORM
        assert config.max_norm == pytest.approx(1.0)
        assert config.max_value == pytest.approx(1.0)
        assert config.norm_type == pytest.approx(2.0)

    def test_value_clipping(self) -> None:
        """Create value clipping config."""
        config = create_clipping_config(method="value", max_value=0.5)
        assert config.method == ClippingMethod.VALUE
        assert config.max_value == pytest.approx(0.5)

    def test_adaptive_clipping(self) -> None:
        """Create adaptive clipping config."""
        config = create_clipping_config(method="adaptive", max_norm=2.0)
        assert config.method == ClippingMethod.ADAPTIVE
        assert config.max_norm == pytest.approx(2.0)

    def test_no_clipping(self) -> None:
        """Create no clipping config."""
        config = create_clipping_config(method="none")
        assert config.method == ClippingMethod.NONE

    def test_custom_norm_type(self) -> None:
        """Create config with custom norm type."""
        config = create_clipping_config(norm_type=1.0)
        assert config.norm_type == pytest.approx(1.0)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_clipping_config(method="invalid")


class TestCreateScalingConfig:
    """Tests for create_scaling_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_scaling_config()
        assert config.method == ScalingMethod.DYNAMIC
        assert config.initial_scale == pytest.approx(65536.0)
        assert config.growth_factor == pytest.approx(2.0)
        assert config.backoff_factor == pytest.approx(0.5)

    def test_static_config(self) -> None:
        """Create static config."""
        config = create_scaling_config(method="static", initial_scale=1.0)
        assert config.method == ScalingMethod.STATIC
        assert config.initial_scale == pytest.approx(1.0)

    def test_loss_scale_config(self) -> None:
        """Create loss scale config."""
        config = create_scaling_config(method="loss_scale")
        assert config.method == ScalingMethod.LOSS_SCALE

    def test_custom_growth_factor(self) -> None:
        """Create config with custom growth factor."""
        config = create_scaling_config(growth_factor=4.0)
        assert config.growth_factor == pytest.approx(4.0)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_scaling_config(method="invalid")


class TestCreateAccumulationConfig:
    """Tests for create_accumulation_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_accumulation_config()
        assert config.steps == 1
        assert config.strategy == AccumulationStrategy.MEAN
        assert config.sync_grads is True

    def test_sum_strategy(self) -> None:
        """Create sum strategy config."""
        config = create_accumulation_config(strategy="sum")
        assert config.strategy == AccumulationStrategy.SUM

    def test_weighted_strategy(self) -> None:
        """Create weighted strategy config."""
        config = create_accumulation_config(strategy="weighted")
        assert config.strategy == AccumulationStrategy.WEIGHTED

    def test_custom_steps(self) -> None:
        """Create config with custom steps."""
        config = create_accumulation_config(steps=8)
        assert config.steps == 8

    def test_sync_grads_false(self) -> None:
        """Create config with sync_grads=False."""
        config = create_accumulation_config(sync_grads=False)
        assert config.sync_grads is False

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_accumulation_config(strategy="invalid")


class TestCreateGradientConfig:
    """Tests for create_gradient_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_gradient_config()
        assert config.clipping_config.method == ClippingMethod.NORM
        assert config.scaling_config.method == ScalingMethod.DYNAMIC
        assert config.accumulation_config.steps == 1

    def test_custom_clipping(self) -> None:
        """Create config with custom clipping."""
        clipping = create_clipping_config(method="value", max_value=0.5)
        config = create_gradient_config(clipping_config=clipping)
        assert config.clipping_config.method == ClippingMethod.VALUE

    def test_custom_scaling(self) -> None:
        """Create config with custom scaling."""
        scaling = create_scaling_config(method="static", initial_scale=1.0)
        config = create_gradient_config(scaling_config=scaling)
        assert config.scaling_config.method == ScalingMethod.STATIC

    def test_custom_accumulation(self) -> None:
        """Create config with custom accumulation."""
        accumulation = create_accumulation_config(steps=4, strategy="sum")
        config = create_gradient_config(accumulation_config=accumulation)
        assert config.accumulation_config.steps == 4
        assert config.accumulation_config.strategy == AccumulationStrategy.SUM


class TestCreateGradientStats:
    """Tests for create_gradient_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_gradient_stats()
        assert stats.grad_norm == pytest.approx(0.0)
        assert stats.clipped_ratio == pytest.approx(0.0)
        assert stats.overflow_count == 0
        assert stats.effective_batch_size == 1

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_gradient_stats(
            grad_norm=0.5,
            clipped_ratio=0.1,
            overflow_count=2,
            effective_batch_size=32,
        )
        assert stats.grad_norm == pytest.approx(0.5)
        assert stats.clipped_ratio == pytest.approx(0.1)
        assert stats.overflow_count == 2
        assert stats.effective_batch_size == 32


class TestCalculateGradNorm:
    """Tests for calculate_grad_norm function."""

    def test_l2_norm_3_4(self) -> None:
        """Calculate L2 norm of [3, 4]."""
        result = calculate_grad_norm([3.0, 4.0], 2.0)
        assert result == pytest.approx(5.0)

    def test_l1_norm(self) -> None:
        """Calculate L1 norm."""
        result = calculate_grad_norm([1.0, 2.0, 3.0], 1.0)
        assert result == pytest.approx(6.0)

    def test_linf_norm(self) -> None:
        """Calculate Linf norm."""
        result = calculate_grad_norm([1.0, 5.0, 2.0], float("inf"))
        assert result == pytest.approx(5.0)

    def test_l2_norm_single_element(self) -> None:
        """Calculate L2 norm of single element."""
        result = calculate_grad_norm([5.0], 2.0)
        assert result == pytest.approx(5.0)

    def test_negative_gradients(self) -> None:
        """Calculate norm with negative gradients."""
        result = calculate_grad_norm([-3.0, -4.0], 2.0)
        assert result == pytest.approx(5.0)

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            calculate_grad_norm([], 2.0)

    def test_zero_norm_type_raises(self) -> None:
        """Zero norm_type raises ValueError."""
        with pytest.raises(ValueError, match="norm_type must be positive"):
            calculate_grad_norm([1.0, 2.0], 0.0)

    def test_negative_norm_type_raises(self) -> None:
        """Negative norm_type raises ValueError."""
        with pytest.raises(ValueError, match="norm_type must be positive"):
            calculate_grad_norm([1.0, 2.0], -1.0)


class TestClipGradients:
    """Tests for clip_gradients function."""

    def test_norm_clipping_under_threshold(self) -> None:
        """Norm clipping when under threshold."""
        config = ClippingConfig(ClippingMethod.NORM, 10.0, 1.0, 2.0)
        grads = [0.6, 0.8]  # norm = 1.0
        clipped, ratio = clip_gradients(grads, config)
        assert clipped == [0.6, 0.8]
        assert ratio == pytest.approx(0.0)

    def test_norm_clipping_over_threshold(self) -> None:
        """Norm clipping when over threshold."""
        config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        grads = [3.0, 4.0]  # norm = 5.0
        clipped, ratio = clip_gradients(grads, config)
        clipped_norm = calculate_grad_norm(clipped, 2.0)
        assert abs(clipped_norm - 1.0) < 0.01
        assert ratio == pytest.approx(1.0)

    def test_value_clipping(self) -> None:
        """Value clipping clips individual values."""
        config = ClippingConfig(ClippingMethod.VALUE, 1.0, 0.5, 2.0)
        grads = [1.0, -2.0, 0.3]
        clipped, ratio = clip_gradients(grads, config)
        assert all(-0.5 <= g <= 0.5 for g in clipped)
        assert ratio == 2 / 3

    def test_no_clipping(self) -> None:
        """No clipping returns original gradients."""
        config = ClippingConfig(ClippingMethod.NONE, 1.0, 1.0, 2.0)
        grads = [1.0, 2.0, 3.0]
        clipped, ratio = clip_gradients(grads, config)
        assert clipped == [1.0, 2.0, 3.0]
        assert ratio == pytest.approx(0.0)

    def test_adaptive_clipping(self) -> None:
        """Adaptive clipping uses stricter threshold."""
        config = ClippingConfig(ClippingMethod.ADAPTIVE, 1.0, 1.0, 2.0)
        grads = [3.0, 4.0]  # norm = 5.0
        clipped, ratio = clip_gradients(grads, config)
        clipped_norm = calculate_grad_norm(clipped, 2.0)
        assert clipped_norm < 1.0  # Stricter threshold (0.9 * 1.0)
        assert ratio == pytest.approx(1.0)

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        config = ClippingConfig(ClippingMethod.NORM, 1.0, 1.0, 2.0)
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            clip_gradients([], config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            clip_gradients([1.0, 2.0], None)  # type: ignore[arg-type]


class TestScaleGradients:
    """Tests for scale_gradients function."""

    def test_scale_up(self) -> None:
        """Scale gradients up."""
        result = scale_gradients([1.0, 2.0, 3.0], 2.0)
        assert result == [2.0, 4.0, 6.0]

    def test_scale_down(self) -> None:
        """Scale gradients down."""
        result = scale_gradients([1.0, 2.0], 0.5)
        assert result == [0.5, 1.0]

    def test_scale_identity(self) -> None:
        """Scale by 1 returns same values."""
        result = scale_gradients([1.0, 2.0, 3.0], 1.0)
        assert result == [1.0, 2.0, 3.0]

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            scale_gradients([], 1.0)

    def test_zero_scale_raises(self) -> None:
        """Zero scale raises ValueError."""
        with pytest.raises(ValueError, match="scale must be positive"):
            scale_gradients([1.0, 2.0], 0.0)

    def test_negative_scale_raises(self) -> None:
        """Negative scale raises ValueError."""
        with pytest.raises(ValueError, match="scale must be positive"):
            scale_gradients([1.0, 2.0], -1.0)


class TestAccumulateGradients:
    """Tests for accumulate_gradients function."""

    def test_mean_accumulation(self) -> None:
        """Mean accumulation averages gradients."""
        batches = [[1.0, 2.0], [3.0, 4.0]]
        result = accumulate_gradients(batches, AccumulationStrategy.MEAN)
        assert result == [2.0, 3.0]

    def test_sum_accumulation(self) -> None:
        """Sum accumulation sums gradients."""
        batches = [[1.0, 2.0], [3.0, 4.0]]
        result = accumulate_gradients(batches, AccumulationStrategy.SUM)
        assert result == [4.0, 6.0]

    def test_weighted_accumulation(self) -> None:
        """Weighted accumulation applies weights."""
        batches = [[1.0, 2.0], [3.0, 4.0]]
        weights = [0.25, 0.75]
        result = accumulate_gradients(batches, AccumulationStrategy.WEIGHTED, weights)
        assert result == [2.5, 3.5]

    def test_single_batch_mean(self) -> None:
        """Single batch mean returns same values."""
        batches = [[1.0, 2.0, 3.0]]
        result = accumulate_gradients(batches, AccumulationStrategy.MEAN)
        assert result == [1.0, 2.0, 3.0]

    def test_single_batch_sum(self) -> None:
        """Single batch sum returns same values."""
        batches = [[1.0, 2.0, 3.0]]
        result = accumulate_gradients(batches, AccumulationStrategy.SUM)
        assert result == [1.0, 2.0, 3.0]

    def test_empty_batches_raises(self) -> None:
        """Empty batches raises ValueError."""
        with pytest.raises(ValueError, match="gradient_batches cannot be empty"):
            accumulate_gradients([], AccumulationStrategy.MEAN)

    def test_different_lengths_raises(self) -> None:
        """Different batch lengths raises ValueError."""
        batches = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        with pytest.raises(ValueError, match="all gradient batches must have the same"):
            accumulate_gradients(batches, AccumulationStrategy.MEAN)

    def test_weighted_without_weights_raises(self) -> None:
        """Weighted without weights raises ValueError."""
        batches = [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(ValueError, match="weights must be provided"):
            accumulate_gradients(batches, AccumulationStrategy.WEIGHTED)

    def test_weighted_wrong_weights_length_raises(self) -> None:
        """Weighted with wrong weights length raises ValueError."""
        batches = [[1.0, 2.0], [3.0, 4.0]]
        weights = [0.5]
        with pytest.raises(ValueError, match="weights length"):
            accumulate_gradients(batches, AccumulationStrategy.WEIGHTED, weights)


class TestDetectOverflow:
    """Tests for detect_overflow function."""

    def test_no_overflow(self) -> None:
        """No overflow in normal gradients."""
        overflow, count = detect_overflow([1.0, 2.0, 3.0])
        assert overflow is False
        assert count == 0

    def test_overflow_by_threshold(self) -> None:
        """Detect overflow by threshold."""
        overflow, count = detect_overflow([1.0, 70000.0, 3.0])
        assert overflow is True
        assert count == 1

    def test_overflow_inf(self) -> None:
        """Detect inf as overflow."""
        overflow, count = detect_overflow([float("inf"), 1.0])
        assert overflow is True
        assert count == 1

    def test_overflow_nan(self) -> None:
        """Detect NaN as overflow."""
        overflow, count = detect_overflow([float("nan"), 1.0])
        assert overflow is True
        assert count == 1

    def test_overflow_multiple(self) -> None:
        """Count multiple overflows."""
        overflow, count = detect_overflow([float("inf"), float("nan"), 70000.0, 1.0])
        assert overflow is True
        assert count == 3

    def test_custom_threshold(self) -> None:
        """Use custom threshold."""
        overflow, count = detect_overflow([100.0, 200.0], threshold=150.0)
        assert overflow is True
        assert count == 1

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            detect_overflow([])

    def test_zero_threshold_raises(self) -> None:
        """Zero threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            detect_overflow([1.0, 2.0], threshold=0.0)

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            detect_overflow([1.0, 2.0], threshold=-1.0)


class TestFormatGradientStats:
    """Tests for format_gradient_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = GradientStats(0.5, 0.1, 2, 32)
        formatted = format_gradient_stats(stats)
        assert "Gradient Norm:" in formatted
        assert "0.500000" in formatted
        assert "Clipped Ratio:" in formatted
        assert "10.00%" in formatted
        assert "Overflow Count: 2" in formatted
        assert "Effective Batch Size: 32" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_gradient_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedGradientConfig:
    """Tests for get_recommended_gradient_config function."""

    def test_fine_tuning_7b(self) -> None:
        """Get config for 7B fine tuning."""
        config = get_recommended_gradient_config("7b", "fine_tuning")
        assert config.clipping_config.method == ClippingMethod.NORM
        assert config.clipping_config.max_norm == pytest.approx(1.0)

    def test_fine_tuning_70b(self) -> None:
        """Get config for 70B fine tuning."""
        config = get_recommended_gradient_config("70b", "fine_tuning")
        assert config.clipping_config.max_norm == pytest.approx(2.0)

    def test_pretraining(self) -> None:
        """Get config for pretraining."""
        config = get_recommended_gradient_config("7b", "pretraining")
        assert config.scaling_config.initial_scale == 2**16

    def test_rlhf(self) -> None:
        """Get config for RLHF."""
        config = get_recommended_gradient_config("7b", "rlhf")
        assert config.clipping_config.max_norm == pytest.approx(0.5)

    def test_custom_accumulation_steps(self) -> None:
        """Get config with custom accumulation steps."""
        config = get_recommended_gradient_config("7b", "fine_tuning", 4)
        assert config.accumulation_config.steps == 4

    def test_invalid_training_type_raises(self) -> None:
        """Invalid training type raises ValueError."""
        with pytest.raises(ValueError, match="training_type must be one of"):
            get_recommended_gradient_config("7b", "invalid")

    def test_zero_accumulation_steps_raises(self) -> None:
        """Zero accumulation steps raises ValueError."""
        with pytest.raises(ValueError, match="accumulation_steps must be positive"):
            get_recommended_gradient_config("7b", "fine_tuning", 0)

    def test_negative_accumulation_steps_raises(self) -> None:
        """Negative accumulation steps raises ValueError."""
        with pytest.raises(ValueError, match="accumulation_steps must be positive"):
            get_recommended_gradient_config("7b", "fine_tuning", -1)


class TestListClippingMethods:
    """Tests for list_clipping_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_clipping_methods()
        assert methods == sorted(methods)

    def test_contains_norm(self) -> None:
        """Contains norm."""
        methods = list_clipping_methods()
        assert "norm" in methods

    def test_contains_value(self) -> None:
        """Contains value."""
        methods = list_clipping_methods()
        assert "value" in methods

    def test_contains_adaptive(self) -> None:
        """Contains adaptive."""
        methods = list_clipping_methods()
        assert "adaptive" in methods

    def test_contains_none(self) -> None:
        """Contains none."""
        methods = list_clipping_methods()
        assert "none" in methods


class TestListScalingMethods:
    """Tests for list_scaling_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        methods = list_scaling_methods()
        assert methods == sorted(methods)

    def test_contains_static(self) -> None:
        """Contains static."""
        methods = list_scaling_methods()
        assert "static" in methods

    def test_contains_dynamic(self) -> None:
        """Contains dynamic."""
        methods = list_scaling_methods()
        assert "dynamic" in methods

    def test_contains_loss_scale(self) -> None:
        """Contains loss_scale."""
        methods = list_scaling_methods()
        assert "loss_scale" in methods


class TestListAccumulationStrategies:
    """Tests for list_accumulation_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_accumulation_strategies()
        assert strategies == sorted(strategies)

    def test_contains_mean(self) -> None:
        """Contains mean."""
        strategies = list_accumulation_strategies()
        assert "mean" in strategies

    def test_contains_sum(self) -> None:
        """Contains sum."""
        strategies = list_accumulation_strategies()
        assert "sum" in strategies

    def test_contains_weighted(self) -> None:
        """Contains weighted."""
        strategies = list_accumulation_strategies()
        assert "weighted" in strategies


class TestGetClippingMethod:
    """Tests for get_clipping_method function."""

    def test_get_norm(self) -> None:
        """Get norm."""
        assert get_clipping_method("norm") == ClippingMethod.NORM

    def test_get_value(self) -> None:
        """Get value."""
        assert get_clipping_method("value") == ClippingMethod.VALUE

    def test_get_adaptive(self) -> None:
        """Get adaptive."""
        assert get_clipping_method("adaptive") == ClippingMethod.ADAPTIVE

    def test_get_none(self) -> None:
        """Get none."""
        assert get_clipping_method("none") == ClippingMethod.NONE

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="clipping method must be one of"):
            get_clipping_method("invalid")


class TestGetScalingMethod:
    """Tests for get_scaling_method function."""

    def test_get_static(self) -> None:
        """Get static."""
        assert get_scaling_method("static") == ScalingMethod.STATIC

    def test_get_dynamic(self) -> None:
        """Get dynamic."""
        assert get_scaling_method("dynamic") == ScalingMethod.DYNAMIC

    def test_get_loss_scale(self) -> None:
        """Get loss_scale."""
        assert get_scaling_method("loss_scale") == ScalingMethod.LOSS_SCALE

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="scaling method must be one of"):
            get_scaling_method("invalid")


class TestGetAccumulationStrategy:
    """Tests for get_accumulation_strategy function."""

    def test_get_mean(self) -> None:
        """Get mean."""
        assert get_accumulation_strategy("mean") == AccumulationStrategy.MEAN

    def test_get_sum(self) -> None:
        """Get sum."""
        assert get_accumulation_strategy("sum") == AccumulationStrategy.SUM

    def test_get_weighted(self) -> None:
        """Get weighted."""
        assert get_accumulation_strategy("weighted") == AccumulationStrategy.WEIGHTED

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="accumulation strategy must be one of"):
            get_accumulation_strategy("invalid")
