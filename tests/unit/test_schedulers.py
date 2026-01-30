"""Tests for training.schedulers module."""

from __future__ import annotations

import pytest

from hf_gtc.training.schedulers import (
    VALID_DECAY_TYPES,
    VALID_SCHEDULER_TYPES,
    VALID_WARMUP_TYPES,
    CosineConfig,
    DecayType,
    LRSchedulerConfig,
    LRSchedulerType,
    PolynomialConfig,
    SchedulerStats,
    WarmupConfig,
    WarmupType,
    calculate_decay_lr,
    calculate_lr_at_step,
    calculate_warmup_lr,
    create_cosine_config,
    create_lr_scheduler_config,
    create_polynomial_config,
    create_scheduler_stats,
    create_warmup_config,
    format_scheduler_stats,
    get_decay_type,
    get_recommended_scheduler_config,
    get_scheduler_type,
    get_warmup_type,
    list_decay_types,
    list_scheduler_types,
    list_warmup_types,
    plot_lr_schedule,
    validate_cosine_config,
    validate_lr_scheduler_config,
    validate_polynomial_config,
    validate_scheduler_stats,
    validate_warmup_config,
)


class TestLRSchedulerType:
    """Tests for LRSchedulerType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for scheduler_type in LRSchedulerType:
            assert isinstance(scheduler_type.value, str)

    def test_constant_value(self) -> None:
        """Constant has correct value."""
        assert LRSchedulerType.CONSTANT.value == "constant"

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert LRSchedulerType.LINEAR.value == "linear"

    def test_cosine_value(self) -> None:
        """Cosine has correct value."""
        assert LRSchedulerType.COSINE.value == "cosine"

    def test_cosine_restarts_value(self) -> None:
        """Cosine restarts has correct value."""
        assert LRSchedulerType.COSINE_RESTARTS.value == "cosine_restarts"

    def test_polynomial_value(self) -> None:
        """Polynomial has correct value."""
        assert LRSchedulerType.POLYNOMIAL.value == "polynomial"

    def test_inverse_sqrt_value(self) -> None:
        """Inverse sqrt has correct value."""
        assert LRSchedulerType.INVERSE_SQRT.value == "inverse_sqrt"

    def test_one_cycle_value(self) -> None:
        """One cycle has correct value."""
        assert LRSchedulerType.ONE_CYCLE.value == "one_cycle"

    def test_valid_scheduler_types_frozenset(self) -> None:
        """VALID_SCHEDULER_TYPES is a frozenset."""
        assert isinstance(VALID_SCHEDULER_TYPES, frozenset)
        assert len(VALID_SCHEDULER_TYPES) == 7


class TestWarmupType:
    """Tests for WarmupType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for warmup_type in WarmupType:
            assert isinstance(warmup_type.value, str)

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert WarmupType.LINEAR.value == "linear"

    def test_exponential_value(self) -> None:
        """Exponential has correct value."""
        assert WarmupType.EXPONENTIAL.value == "exponential"

    def test_constant_value(self) -> None:
        """Constant has correct value."""
        assert WarmupType.CONSTANT.value == "constant"

    def test_valid_warmup_types_frozenset(self) -> None:
        """VALID_WARMUP_TYPES is a frozenset."""
        assert isinstance(VALID_WARMUP_TYPES, frozenset)
        assert len(VALID_WARMUP_TYPES) == 3


class TestDecayType:
    """Tests for DecayType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for decay_type in DecayType:
            assert isinstance(decay_type.value, str)

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert DecayType.LINEAR.value == "linear"

    def test_exponential_value(self) -> None:
        """Exponential has correct value."""
        assert DecayType.EXPONENTIAL.value == "exponential"

    def test_cosine_value(self) -> None:
        """Cosine has correct value."""
        assert DecayType.COSINE.value == "cosine"

    def test_valid_decay_types_frozenset(self) -> None:
        """VALID_DECAY_TYPES is a frozenset."""
        assert isinstance(VALID_DECAY_TYPES, frozenset)
        assert len(VALID_DECAY_TYPES) == 3


class TestWarmupConfig:
    """Tests for WarmupConfig dataclass."""

    def test_create_warmup_config(self) -> None:
        """Create warmup config."""
        config = WarmupConfig(
            warmup_steps=100,
            warmup_ratio=0.0,
            warmup_type=WarmupType.LINEAR,
        )
        assert config.warmup_steps == 100
        assert config.warmup_ratio == 0.0
        assert config.warmup_type == WarmupType.LINEAR

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = WarmupConfig(100, 0.0, WarmupType.LINEAR)
        with pytest.raises(AttributeError):
            config.warmup_steps = 200  # type: ignore[misc]


class TestCosineConfig:
    """Tests for CosineConfig dataclass."""

    def test_create_cosine_config(self) -> None:
        """Create cosine config."""
        config = CosineConfig(
            num_cycles=1.0,
            min_lr_ratio=0.0,
            eta_min=0.0,
        )
        assert config.num_cycles == 1.0
        assert config.min_lr_ratio == 0.0
        assert config.eta_min == 0.0

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = CosineConfig(1.0, 0.0, 0.0)
        with pytest.raises(AttributeError):
            config.num_cycles = 2.0  # type: ignore[misc]


class TestPolynomialConfig:
    """Tests for PolynomialConfig dataclass."""

    def test_create_polynomial_config(self) -> None:
        """Create polynomial config."""
        config = PolynomialConfig(power=2.0, lr_end=1e-6)
        assert config.power == 2.0
        assert config.lr_end == 1e-6

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PolynomialConfig(1.0, 0.0)
        with pytest.raises(AttributeError):
            config.power = 2.0  # type: ignore[misc]


class TestLRSchedulerConfig:
    """Tests for LRSchedulerConfig dataclass."""

    def test_create_scheduler_config(self) -> None:
        """Create scheduler config."""
        warmup = WarmupConfig(100, 0.0, WarmupType.LINEAR)
        cosine = CosineConfig(1.0, 0.0, 0.0)
        config = LRSchedulerConfig(
            scheduler_type=LRSchedulerType.COSINE,
            warmup_config=warmup,
            cosine_config=cosine,
            polynomial_config=None,
            total_steps=1000,
            num_epochs=3,
            base_lr=1e-4,
        )
        assert config.scheduler_type == LRSchedulerType.COSINE
        assert config.total_steps == 1000
        assert config.base_lr == 1e-4

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LRSchedulerConfig(
            LRSchedulerType.COSINE, None, None, None, 1000, 3, 1e-4
        )
        with pytest.raises(AttributeError):
            config.total_steps = 2000  # type: ignore[misc]


class TestSchedulerStats:
    """Tests for SchedulerStats dataclass."""

    def test_create_scheduler_stats(self) -> None:
        """Create scheduler stats."""
        stats = SchedulerStats(
            current_lr=5e-5,
            step=500,
            warmup_complete=True,
            decay_progress=0.5,
        )
        assert stats.current_lr == 5e-5
        assert stats.step == 500
        assert stats.warmup_complete is True
        assert stats.decay_progress == 0.5

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = SchedulerStats(1e-4, 100, True, 0.5)
        with pytest.raises(AttributeError):
            stats.step = 200  # type: ignore[misc]


class TestValidateWarmupConfig:
    """Tests for validate_warmup_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = WarmupConfig(100, 0.0, WarmupType.LINEAR)
        validate_warmup_config(config)

    def test_valid_ratio_config(self) -> None:
        """Valid ratio config passes validation."""
        config = WarmupConfig(0, 0.1, WarmupType.LINEAR)
        validate_warmup_config(config)

    def test_negative_warmup_steps_raises(self) -> None:
        """Negative warmup_steps raises ValueError."""
        config = WarmupConfig(-1, 0.0, WarmupType.LINEAR)
        with pytest.raises(ValueError, match="warmup_steps cannot be negative"):
            validate_warmup_config(config)

    def test_negative_warmup_ratio_raises(self) -> None:
        """Negative warmup_ratio raises ValueError."""
        config = WarmupConfig(0, -0.1, WarmupType.LINEAR)
        with pytest.raises(ValueError, match="warmup_ratio must be between"):
            validate_warmup_config(config)

    def test_warmup_ratio_above_one_raises(self) -> None:
        """Warmup ratio above 1.0 raises ValueError."""
        config = WarmupConfig(0, 1.5, WarmupType.LINEAR)
        with pytest.raises(ValueError, match="warmup_ratio must be between"):
            validate_warmup_config(config)


class TestValidateCosineConfig:
    """Tests for validate_cosine_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = CosineConfig(1.0, 0.0, 0.0)
        validate_cosine_config(config)

    def test_zero_num_cycles_raises(self) -> None:
        """Zero num_cycles raises ValueError."""
        config = CosineConfig(0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="num_cycles must be positive"):
            validate_cosine_config(config)

    def test_negative_num_cycles_raises(self) -> None:
        """Negative num_cycles raises ValueError."""
        config = CosineConfig(-1.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="num_cycles must be positive"):
            validate_cosine_config(config)

    def test_negative_min_lr_ratio_raises(self) -> None:
        """Negative min_lr_ratio raises ValueError."""
        config = CosineConfig(1.0, -0.1, 0.0)
        with pytest.raises(ValueError, match="min_lr_ratio must be between"):
            validate_cosine_config(config)

    def test_min_lr_ratio_above_one_raises(self) -> None:
        """Min LR ratio above 1.0 raises ValueError."""
        config = CosineConfig(1.0, 1.5, 0.0)
        with pytest.raises(ValueError, match="min_lr_ratio must be between"):
            validate_cosine_config(config)

    def test_negative_eta_min_raises(self) -> None:
        """Negative eta_min raises ValueError."""
        config = CosineConfig(1.0, 0.0, -1e-6)
        with pytest.raises(ValueError, match="eta_min cannot be negative"):
            validate_cosine_config(config)


class TestValidatePolynomialConfig:
    """Tests for validate_polynomial_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PolynomialConfig(1.0, 0.0)
        validate_polynomial_config(config)

    def test_zero_power_raises(self) -> None:
        """Zero power raises ValueError."""
        config = PolynomialConfig(0.0, 0.0)
        with pytest.raises(ValueError, match="power must be positive"):
            validate_polynomial_config(config)

    def test_negative_power_raises(self) -> None:
        """Negative power raises ValueError."""
        config = PolynomialConfig(-1.0, 0.0)
        with pytest.raises(ValueError, match="power must be positive"):
            validate_polynomial_config(config)

    def test_negative_lr_end_raises(self) -> None:
        """Negative lr_end raises ValueError."""
        config = PolynomialConfig(1.0, -1e-6)
        with pytest.raises(ValueError, match="lr_end cannot be negative"):
            validate_polynomial_config(config)


class TestValidateLRSchedulerConfig:
    """Tests for validate_lr_scheduler_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LRSchedulerConfig(
            LRSchedulerType.COSINE, None, None, None, 1000, 3, 1e-4
        )
        validate_lr_scheduler_config(config)

    def test_zero_total_steps_raises(self) -> None:
        """Zero total_steps raises ValueError."""
        config = LRSchedulerConfig(LRSchedulerType.COSINE, None, None, None, 0, 3, 1e-4)
        with pytest.raises(ValueError, match="total_steps must be positive"):
            validate_lr_scheduler_config(config)

    def test_zero_num_epochs_raises(self) -> None:
        """Zero num_epochs raises ValueError."""
        config = LRSchedulerConfig(
            LRSchedulerType.COSINE, None, None, None, 1000, 0, 1e-4
        )
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            validate_lr_scheduler_config(config)

    def test_zero_base_lr_raises(self) -> None:
        """Zero base_lr raises ValueError."""
        config = LRSchedulerConfig(
            LRSchedulerType.COSINE, None, None, None, 1000, 3, 0.0
        )
        with pytest.raises(ValueError, match="base_lr must be positive"):
            validate_lr_scheduler_config(config)

    def test_invalid_warmup_config_raises(self) -> None:
        """Invalid warmup config raises ValueError."""
        warmup = WarmupConfig(-1, 0.0, WarmupType.LINEAR)
        config = LRSchedulerConfig(
            LRSchedulerType.COSINE, warmup, None, None, 1000, 3, 1e-4
        )
        with pytest.raises(ValueError, match="warmup_steps cannot be negative"):
            validate_lr_scheduler_config(config)

    def test_invalid_cosine_config_raises(self) -> None:
        """Invalid cosine config raises ValueError."""
        cosine = CosineConfig(0.0, 0.0, 0.0)
        config = LRSchedulerConfig(
            LRSchedulerType.COSINE, None, cosine, None, 1000, 3, 1e-4
        )
        with pytest.raises(ValueError, match="num_cycles must be positive"):
            validate_lr_scheduler_config(config)

    def test_invalid_polynomial_config_raises(self) -> None:
        """Invalid polynomial config raises ValueError."""
        poly = PolynomialConfig(0.0, 0.0)
        config = LRSchedulerConfig(
            LRSchedulerType.POLYNOMIAL, None, None, poly, 1000, 3, 1e-4
        )
        with pytest.raises(ValueError, match="power must be positive"):
            validate_lr_scheduler_config(config)


class TestValidateSchedulerStats:
    """Tests for validate_scheduler_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = SchedulerStats(1e-4, 100, True, 0.5)
        validate_scheduler_stats(stats)

    def test_negative_current_lr_raises(self) -> None:
        """Negative current_lr raises ValueError."""
        stats = SchedulerStats(-1e-4, 100, True, 0.5)
        with pytest.raises(ValueError, match="current_lr cannot be negative"):
            validate_scheduler_stats(stats)

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        stats = SchedulerStats(1e-4, -1, True, 0.5)
        with pytest.raises(ValueError, match="step cannot be negative"):
            validate_scheduler_stats(stats)

    def test_negative_decay_progress_raises(self) -> None:
        """Negative decay_progress raises ValueError."""
        stats = SchedulerStats(1e-4, 100, True, -0.1)
        with pytest.raises(ValueError, match="decay_progress must be between"):
            validate_scheduler_stats(stats)

    def test_decay_progress_above_one_raises(self) -> None:
        """Decay progress above 1.0 raises ValueError."""
        stats = SchedulerStats(1e-4, 100, True, 1.5)
        with pytest.raises(ValueError, match="decay_progress must be between"):
            validate_scheduler_stats(stats)


class TestCreateWarmupConfig:
    """Tests for create_warmup_config function."""

    def test_default_config(self) -> None:
        """Create default warmup config."""
        config = create_warmup_config()
        assert config.warmup_steps == 0
        assert config.warmup_ratio == 0.0
        assert config.warmup_type == WarmupType.LINEAR

    def test_with_steps(self) -> None:
        """Create config with warmup steps."""
        config = create_warmup_config(warmup_steps=100)
        assert config.warmup_steps == 100

    def test_with_ratio(self) -> None:
        """Create config with warmup ratio."""
        config = create_warmup_config(warmup_ratio=0.1)
        assert config.warmup_ratio == 0.1

    def test_with_string_type(self) -> None:
        """Create config with string warmup type."""
        config = create_warmup_config(warmup_type="exponential")
        assert config.warmup_type == WarmupType.EXPONENTIAL

    def test_with_enum_type(self) -> None:
        """Create config with enum warmup type."""
        config = create_warmup_config(warmup_type=WarmupType.CONSTANT)
        assert config.warmup_type == WarmupType.CONSTANT

    def test_invalid_steps_raises(self) -> None:
        """Invalid warmup steps raises ValueError."""
        with pytest.raises(ValueError, match="warmup_steps cannot be negative"):
            create_warmup_config(warmup_steps=-1)


class TestCreateCosineConfig:
    """Tests for create_cosine_config function."""

    def test_default_config(self) -> None:
        """Create default cosine config."""
        config = create_cosine_config()
        assert config.num_cycles == 1.0
        assert config.min_lr_ratio == 0.0
        assert config.eta_min == 0.0

    def test_with_custom_values(self) -> None:
        """Create config with custom values."""
        config = create_cosine_config(num_cycles=3.0, min_lr_ratio=0.1, eta_min=1e-6)
        assert config.num_cycles == 3.0
        assert config.min_lr_ratio == 0.1
        assert config.eta_min == 1e-6

    def test_invalid_cycles_raises(self) -> None:
        """Invalid num_cycles raises ValueError."""
        with pytest.raises(ValueError, match="num_cycles must be positive"):
            create_cosine_config(num_cycles=0.0)


class TestCreatePolynomialConfig:
    """Tests for create_polynomial_config function."""

    def test_default_config(self) -> None:
        """Create default polynomial config."""
        config = create_polynomial_config()
        assert config.power == 1.0
        assert config.lr_end == 0.0

    def test_with_custom_values(self) -> None:
        """Create config with custom values."""
        config = create_polynomial_config(power=2.0, lr_end=1e-6)
        assert config.power == 2.0
        assert config.lr_end == 1e-6

    def test_invalid_power_raises(self) -> None:
        """Invalid power raises ValueError."""
        with pytest.raises(ValueError, match="power must be positive"):
            create_polynomial_config(power=0.0)


class TestCreateLRSchedulerConfig:
    """Tests for create_lr_scheduler_config function."""

    def test_default_config(self) -> None:
        """Create default scheduler config."""
        config = create_lr_scheduler_config()
        assert config.scheduler_type == LRSchedulerType.COSINE
        assert config.total_steps == 1000
        assert config.num_epochs == 3
        assert config.base_lr == 1e-4

    def test_with_string_type(self) -> None:
        """Create config with string scheduler type."""
        config = create_lr_scheduler_config(scheduler_type="linear")
        assert config.scheduler_type == LRSchedulerType.LINEAR

    def test_with_enum_type(self) -> None:
        """Create config with enum scheduler type."""
        config = create_lr_scheduler_config(scheduler_type=LRSchedulerType.POLYNOMIAL)
        assert config.scheduler_type == LRSchedulerType.POLYNOMIAL

    def test_with_warmup(self) -> None:
        """Create config with warmup."""
        warmup = create_warmup_config(warmup_steps=100)
        config = create_lr_scheduler_config(warmup_config=warmup)
        assert config.warmup_config is not None
        assert config.warmup_config.warmup_steps == 100

    def test_with_cosine_config(self) -> None:
        """Create config with cosine config."""
        cosine = create_cosine_config(num_cycles=3.0)
        config = create_lr_scheduler_config(cosine_config=cosine)
        assert config.cosine_config is not None
        assert config.cosine_config.num_cycles == 3.0

    def test_with_polynomial_config(self) -> None:
        """Create config with polynomial config."""
        poly = create_polynomial_config(power=2.0)
        config = create_lr_scheduler_config(polynomial_config=poly)
        assert config.polynomial_config is not None
        assert config.polynomial_config.power == 2.0

    def test_invalid_total_steps_raises(self) -> None:
        """Invalid total_steps raises ValueError."""
        with pytest.raises(ValueError, match="total_steps must be positive"):
            create_lr_scheduler_config(total_steps=0)


class TestCreateSchedulerStats:
    """Tests for create_scheduler_stats function."""

    def test_default_stats(self) -> None:
        """Create default scheduler stats."""
        stats = create_scheduler_stats()
        assert stats.current_lr == 0.0
        assert stats.step == 0
        assert stats.warmup_complete is False
        assert stats.decay_progress == 0.0

    def test_with_custom_values(self) -> None:
        """Create stats with custom values."""
        stats = create_scheduler_stats(
            current_lr=5e-5,
            step=500,
            warmup_complete=True,
            decay_progress=0.5,
        )
        assert stats.current_lr == 5e-5
        assert stats.step == 500
        assert stats.warmup_complete is True
        assert stats.decay_progress == 0.5

    def test_invalid_lr_raises(self) -> None:
        """Invalid current_lr raises ValueError."""
        with pytest.raises(ValueError, match="current_lr cannot be negative"):
            create_scheduler_stats(current_lr=-1e-4)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_scheduler_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_scheduler_types()
        assert types == sorted(types)
        assert "cosine" in types
        assert "linear" in types

    def test_list_warmup_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_warmup_types()
        assert types == sorted(types)
        assert "linear" in types
        assert "exponential" in types

    def test_list_decay_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_decay_types()
        assert types == sorted(types)
        assert "linear" in types
        assert "cosine" in types


class TestGetSchedulerType:
    """Tests for get_scheduler_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("constant", LRSchedulerType.CONSTANT),
            ("linear", LRSchedulerType.LINEAR),
            ("cosine", LRSchedulerType.COSINE),
            ("cosine_restarts", LRSchedulerType.COSINE_RESTARTS),
            ("polynomial", LRSchedulerType.POLYNOMIAL),
            ("inverse_sqrt", LRSchedulerType.INVERSE_SQRT),
            ("one_cycle", LRSchedulerType.ONE_CYCLE),
        ],
    )
    def test_all_types(self, name: str, expected: LRSchedulerType) -> None:
        """Test all valid scheduler types."""
        assert get_scheduler_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="scheduler_type must be one of"):
            get_scheduler_type("invalid")


class TestGetWarmupType:
    """Tests for get_warmup_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("linear", WarmupType.LINEAR),
            ("exponential", WarmupType.EXPONENTIAL),
            ("constant", WarmupType.CONSTANT),
        ],
    )
    def test_all_types(self, name: str, expected: WarmupType) -> None:
        """Test all valid warmup types."""
        assert get_warmup_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="warmup_type must be one of"):
            get_warmup_type("invalid")


class TestGetDecayType:
    """Tests for get_decay_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("linear", DecayType.LINEAR),
            ("exponential", DecayType.EXPONENTIAL),
            ("cosine", DecayType.COSINE),
        ],
    )
    def test_all_types(self, name: str, expected: DecayType) -> None:
        """Test all valid decay types."""
        assert get_decay_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="decay_type must be one of"):
            get_decay_type("invalid")


class TestCalculateWarmupLR:
    """Tests for calculate_warmup_lr function."""

    def test_linear_warmup_start(self) -> None:
        """Linear warmup starts at 0."""
        lr = calculate_warmup_lr(0, 100, 1e-4)
        assert lr == 0.0

    def test_linear_warmup_mid(self) -> None:
        """Linear warmup at midpoint."""
        lr = calculate_warmup_lr(50, 100, 1e-4)
        assert lr == pytest.approx(5e-5)

    def test_linear_warmup_end(self) -> None:
        """Linear warmup reaches base_lr at end."""
        lr = calculate_warmup_lr(100, 100, 1e-4)
        assert lr == pytest.approx(1e-4)

    def test_linear_warmup_past_end(self) -> None:
        """Linear warmup returns base_lr past end."""
        lr = calculate_warmup_lr(150, 100, 1e-4)
        assert lr == pytest.approx(1e-4)

    def test_exponential_warmup(self) -> None:
        """Exponential warmup produces increasing LR."""
        lr_25 = calculate_warmup_lr(25, 100, 1e-4, WarmupType.EXPONENTIAL)
        lr_50 = calculate_warmup_lr(50, 100, 1e-4, WarmupType.EXPONENTIAL)
        lr_75 = calculate_warmup_lr(75, 100, 1e-4, WarmupType.EXPONENTIAL)
        assert 0 < lr_25 < lr_50 < lr_75 < 1e-4

    def test_constant_warmup(self) -> None:
        """Constant warmup returns base_lr immediately."""
        lr = calculate_warmup_lr(50, 100, 1e-4, WarmupType.CONSTANT)
        assert lr == pytest.approx(1e-4)

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        with pytest.raises(ValueError, match="step cannot be negative"):
            calculate_warmup_lr(-1, 100, 1e-4)

    def test_zero_warmup_steps_raises(self) -> None:
        """Zero warmup_steps raises ValueError."""
        with pytest.raises(ValueError, match="warmup_steps must be positive"):
            calculate_warmup_lr(50, 0, 1e-4)

    def test_zero_base_lr_raises(self) -> None:
        """Zero base_lr raises ValueError."""
        with pytest.raises(ValueError, match="base_lr must be positive"):
            calculate_warmup_lr(50, 100, 0.0)


class TestCalculateDecayLR:
    """Tests for calculate_decay_lr function."""

    def test_cosine_decay_start(self) -> None:
        """Cosine decay starts at base_lr."""
        lr = calculate_decay_lr(0, 100, 1e-4)
        assert lr == pytest.approx(1e-4)

    def test_cosine_decay_mid(self) -> None:
        """Cosine decay at midpoint."""
        lr = calculate_decay_lr(50, 100, 1e-4)
        assert lr == pytest.approx(5e-5)

    def test_cosine_decay_end(self) -> None:
        """Cosine decay reaches min_lr at end."""
        lr = calculate_decay_lr(100, 100, 1e-4)
        assert lr == pytest.approx(0.0)

    def test_cosine_decay_past_end(self) -> None:
        """Cosine decay returns min_lr past end."""
        lr = calculate_decay_lr(150, 100, 1e-4)
        assert lr == pytest.approx(0.0)

    def test_cosine_decay_with_min_lr(self) -> None:
        """Cosine decay with non-zero min_lr."""
        lr = calculate_decay_lr(100, 100, 1e-4, min_lr=1e-5)
        assert lr == pytest.approx(1e-5)

    def test_linear_decay(self) -> None:
        """Linear decay at midpoint."""
        lr = calculate_decay_lr(50, 100, 1e-4, decay_type=DecayType.LINEAR)
        assert lr == pytest.approx(5e-5)

    def test_exponential_decay(self) -> None:
        """Exponential decay produces decreasing LR."""
        lr_25 = calculate_decay_lr(25, 100, 1e-4, decay_type=DecayType.EXPONENTIAL)
        lr_50 = calculate_decay_lr(50, 100, 1e-4, decay_type=DecayType.EXPONENTIAL)
        lr_75 = calculate_decay_lr(75, 100, 1e-4, decay_type=DecayType.EXPONENTIAL)
        assert 1e-4 > lr_25 > lr_50 > lr_75 > 0

    def test_exponential_decay_with_min_lr(self) -> None:
        """Exponential decay with non-zero min_lr."""
        lr = calculate_decay_lr(
            50, 100, 1e-4, min_lr=1e-5, decay_type=DecayType.EXPONENTIAL
        )
        assert lr > 1e-5

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        with pytest.raises(ValueError, match="step cannot be negative"):
            calculate_decay_lr(-1, 100, 1e-4)

    def test_zero_decay_steps_raises(self) -> None:
        """Zero decay_steps raises ValueError."""
        with pytest.raises(ValueError, match="decay_steps must be positive"):
            calculate_decay_lr(50, 0, 1e-4)

    def test_negative_base_lr_raises(self) -> None:
        """Negative base_lr raises ValueError."""
        with pytest.raises(ValueError, match="base_lr cannot be negative"):
            calculate_decay_lr(50, 100, -1e-4)

    def test_negative_min_lr_raises(self) -> None:
        """Negative min_lr raises ValueError."""
        with pytest.raises(ValueError, match="min_lr cannot be negative"):
            calculate_decay_lr(50, 100, 1e-4, min_lr=-1e-5)


class TestCalculateLRAtStep:
    """Tests for calculate_lr_at_step function."""

    def test_constant_scheduler(self) -> None:
        """Constant scheduler returns base_lr."""
        config = create_lr_scheduler_config(scheduler_type="constant", base_lr=1e-4)
        assert calculate_lr_at_step(0, config) == pytest.approx(1e-4)
        assert calculate_lr_at_step(500, config) == pytest.approx(1e-4)
        assert calculate_lr_at_step(1000, config) == pytest.approx(1e-4)

    def test_linear_scheduler(self) -> None:
        """Linear scheduler decays linearly."""
        config = create_lr_scheduler_config(
            scheduler_type="linear",
            total_steps=1000,
            base_lr=1e-4,
        )
        assert calculate_lr_at_step(0, config) == pytest.approx(1e-4)
        assert calculate_lr_at_step(500, config) == pytest.approx(5e-5)
        assert calculate_lr_at_step(1000, config) == pytest.approx(0.0)

    def test_cosine_scheduler(self) -> None:
        """Cosine scheduler follows cosine curve."""
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            total_steps=1000,
            base_lr=1e-4,
        )
        assert calculate_lr_at_step(0, config) == pytest.approx(1e-4)
        lr_mid = calculate_lr_at_step(500, config)
        assert lr_mid == pytest.approx(5e-5)
        assert calculate_lr_at_step(1000, config) == pytest.approx(0.0, abs=1e-10)

    def test_cosine_with_warmup(self) -> None:
        """Cosine scheduler with warmup."""
        warmup = create_warmup_config(warmup_steps=100)
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            warmup_config=warmup,
            total_steps=1000,
            base_lr=1e-4,
        )
        # During warmup
        assert calculate_lr_at_step(50, config) == pytest.approx(5e-5)
        # At warmup end
        assert calculate_lr_at_step(100, config) == pytest.approx(1e-4)
        # After warmup (decay phase)
        lr_mid = calculate_lr_at_step(550, config)
        assert 0 < lr_mid < 1e-4

    def test_cosine_with_warmup_ratio(self) -> None:
        """Cosine scheduler with warmup ratio."""
        warmup = create_warmup_config(warmup_ratio=0.1)  # 100 steps
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            warmup_config=warmup,
            total_steps=1000,
            base_lr=1e-4,
        )
        # During warmup (0.1 * 1000 = 100 warmup steps)
        assert calculate_lr_at_step(50, config) == pytest.approx(5e-5)

    def test_cosine_restarts(self) -> None:
        """Cosine restarts scheduler."""
        cosine = create_cosine_config(num_cycles=2.0)
        config = create_lr_scheduler_config(
            scheduler_type="cosine_restarts",
            cosine_config=cosine,
            total_steps=1000,
            base_lr=1e-4,
        )
        # Should restart at midpoint
        lr_start = calculate_lr_at_step(0, config)
        lr_quarter = calculate_lr_at_step(250, config)
        lr_mid = calculate_lr_at_step(500, config)
        assert lr_start == pytest.approx(1e-4)
        assert lr_quarter < lr_start  # Decaying
        # At restart point (500), should be back to peak
        assert lr_mid == pytest.approx(1e-4)

    def test_polynomial_scheduler(self) -> None:
        """Polynomial scheduler."""
        poly = create_polynomial_config(power=2.0, lr_end=1e-6)
        config = create_lr_scheduler_config(
            scheduler_type="polynomial",
            polynomial_config=poly,
            total_steps=1000,
            base_lr=1e-4,
        )
        assert calculate_lr_at_step(0, config) == pytest.approx(1e-4)
        # Quadratic decay
        lr_mid = calculate_lr_at_step(500, config)
        assert 0 < lr_mid < 1e-4
        # At end should be lr_end
        assert calculate_lr_at_step(1000, config) == pytest.approx(1e-6)

    def test_inverse_sqrt_scheduler(self) -> None:
        """Inverse sqrt scheduler."""
        config = create_lr_scheduler_config(
            scheduler_type="inverse_sqrt",
            total_steps=1000,
            base_lr=1e-4,
        )
        lr_start = calculate_lr_at_step(0, config)
        lr_mid = calculate_lr_at_step(500, config)
        lr_end = calculate_lr_at_step(1000, config)
        # Should decrease following inverse sqrt
        assert lr_start > lr_mid > lr_end

    def test_one_cycle_scheduler(self) -> None:
        """One cycle scheduler."""
        config = create_lr_scheduler_config(
            scheduler_type="one_cycle",
            total_steps=1000,
            base_lr=1e-4,
        )
        # First half: rise
        lr_quarter = calculate_lr_at_step(250, config)
        lr_mid = calculate_lr_at_step(500, config)
        # Second half: decay
        lr_three_quarter = calculate_lr_at_step(750, config)
        lr_end = calculate_lr_at_step(1000, config)

        assert lr_quarter < lr_mid  # Rising
        assert lr_mid > lr_three_quarter  # Decaying
        assert lr_three_quarter > lr_end  # Still decaying

    def test_cosine_with_min_lr_ratio(self) -> None:
        """Cosine with min_lr_ratio."""
        cosine = create_cosine_config(min_lr_ratio=0.1)  # 10% of base_lr
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            cosine_config=cosine,
            total_steps=1000,
            base_lr=1e-4,
        )
        lr_end = calculate_lr_at_step(1000, config)
        assert lr_end == pytest.approx(1e-5)

    def test_cosine_with_eta_min(self) -> None:
        """Cosine with eta_min."""
        cosine = create_cosine_config(eta_min=2e-5)  # Higher than ratio
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            cosine_config=cosine,
            total_steps=1000,
            base_lr=1e-4,
        )
        lr_end = calculate_lr_at_step(1000, config)
        assert lr_end == pytest.approx(2e-5)

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        config = create_lr_scheduler_config()
        with pytest.raises(ValueError, match="step cannot be negative"):
            calculate_lr_at_step(-1, config)


class TestPlotLRSchedule:
    """Tests for plot_lr_schedule function."""

    def test_returns_correct_length(self) -> None:
        """Returns correct number of points."""
        config = create_lr_scheduler_config(total_steps=100, base_lr=1e-3)
        steps, lrs = plot_lr_schedule(config, num_points=10)
        assert len(steps) == 10
        assert len(lrs) == 10

    def test_steps_start_at_zero(self) -> None:
        """Steps start at zero."""
        config = create_lr_scheduler_config(total_steps=100, base_lr=1e-3)
        steps, _lrs = plot_lr_schedule(config, num_points=10)
        assert steps[0] == 0

    def test_lrs_match_config(self) -> None:
        """LRs match expected values."""
        config = create_lr_scheduler_config(
            scheduler_type="constant",
            total_steps=100,
            base_lr=1e-3,
        )
        _steps, lrs = plot_lr_schedule(config, num_points=10)
        for lr in lrs:
            assert lr == pytest.approx(1e-3)

    def test_returns_tuples(self) -> None:
        """Returns tuples."""
        config = create_lr_scheduler_config()
        steps, lrs = plot_lr_schedule(config)
        assert isinstance(steps, tuple)
        assert isinstance(lrs, tuple)

    def test_zero_num_points_raises(self) -> None:
        """Zero num_points raises ValueError."""
        config = create_lr_scheduler_config()
        with pytest.raises(ValueError, match="num_points must be positive"):
            plot_lr_schedule(config, num_points=0)

    def test_negative_num_points_raises(self) -> None:
        """Negative num_points raises ValueError."""
        config = create_lr_scheduler_config()
        with pytest.raises(ValueError, match="num_points must be positive"):
            plot_lr_schedule(config, num_points=-10)


class TestFormatSchedulerStats:
    """Tests for format_scheduler_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_scheduler_stats(
            current_lr=5e-5,
            step=500,
            warmup_complete=True,
            decay_progress=0.5,
        )
        formatted = format_scheduler_stats(stats)
        assert "LR: 5.00e-05" in formatted
        assert "Step: 500" in formatted
        assert "Warmup: Complete" in formatted
        assert "Decay: 50.0%" in formatted

    def test_warmup_in_progress(self) -> None:
        """Format stats with warmup in progress."""
        stats = create_scheduler_stats(
            current_lr=1e-5,
            step=50,
            warmup_complete=False,
            decay_progress=0.0,
        )
        formatted = format_scheduler_stats(stats)
        assert "Warmup: In Progress" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_scheduler_stats(current_lr=1e-4, step=100)
        formatted = format_scheduler_stats(stats)
        assert "LR:" in formatted
        assert "Step:" in formatted
        assert "Warmup:" in formatted
        assert "Decay:" in formatted


class TestGetRecommendedSchedulerConfig:
    """Tests for get_recommended_scheduler_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_scheduler_config("classification")
        assert config.scheduler_type == LRSchedulerType.COSINE
        assert config.warmup_config is not None
        assert config.cosine_config is not None

    def test_generation_config(self) -> None:
        """Get config for generation task."""
        config = get_recommended_scheduler_config("generation")
        assert config.scheduler_type == LRSchedulerType.COSINE
        assert config.warmup_config is not None

    def test_fine_tuning_config(self) -> None:
        """Get config for fine-tuning task."""
        config = get_recommended_scheduler_config("fine_tuning")
        assert config.scheduler_type == LRSchedulerType.COSINE
        assert config.warmup_config is not None

    def test_pretraining_config(self) -> None:
        """Get config for pretraining task."""
        config = get_recommended_scheduler_config("pretraining")
        assert config.scheduler_type == LRSchedulerType.COSINE
        assert config.warmup_config is not None
        assert config.warmup_config.warmup_steps == 2000

    def test_rlhf_config(self) -> None:
        """Get config for RLHF task."""
        config = get_recommended_scheduler_config("rlhf")
        assert config.scheduler_type == LRSchedulerType.LINEAR
        assert config.warmup_config is not None

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_scheduler_config("unknown")

    @pytest.mark.parametrize(
        "task",
        ["classification", "generation", "fine_tuning", "pretraining", "rlhf"],
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All supported tasks return valid configs."""
        config = get_recommended_scheduler_config(task)
        validate_lr_scheduler_config(config)


class TestSchedulerIntegration:
    """Integration tests for scheduler functionality."""

    def test_full_training_schedule(self) -> None:
        """Test complete training schedule with warmup and decay."""
        warmup = create_warmup_config(warmup_steps=100)
        cosine = create_cosine_config(min_lr_ratio=0.1)
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            warmup_config=warmup,
            cosine_config=cosine,
            total_steps=1000,
            base_lr=1e-4,
        )

        # Warmup phase
        for step in range(0, 100, 10):
            lr = calculate_lr_at_step(step, config)
            assert 0 <= lr <= 1e-4

        # At warmup end
        lr_at_warmup_end = calculate_lr_at_step(100, config)
        assert lr_at_warmup_end == pytest.approx(1e-4)

        # Decay phase - should decrease
        prev_lr = lr_at_warmup_end
        for step in range(200, 1001, 100):
            lr = calculate_lr_at_step(step, config)
            assert lr < prev_lr
            prev_lr = lr

        # Final LR should be min_lr
        final_lr = calculate_lr_at_step(1000, config)
        assert final_lr == pytest.approx(1e-5)

    def test_polynomial_with_power_2(self) -> None:
        """Test polynomial decay with power=2 (quadratic)."""
        poly = create_polynomial_config(power=2.0, lr_end=0.0)
        config = create_lr_scheduler_config(
            scheduler_type="polynomial",
            polynomial_config=poly,
            total_steps=100,
            base_lr=1e-4,
        )

        # Quadratic decay: lr = base_lr * (1 - progress)^2
        for step in [0, 25, 50, 75, 100]:
            expected = 1e-4 * ((1 - step / 100) ** 2)
            actual = calculate_lr_at_step(step, config)
            assert actual == pytest.approx(expected, rel=1e-6)

    def test_exponential_warmup_curve(self) -> None:
        """Test exponential warmup produces smooth curve."""
        warmup = create_warmup_config(
            warmup_steps=100, warmup_type=WarmupType.EXPONENTIAL
        )
        config = create_lr_scheduler_config(
            scheduler_type="constant",
            warmup_config=warmup,
            total_steps=200,
            base_lr=1e-4,
        )

        lrs = [calculate_lr_at_step(s, config) for s in range(0, 101, 10)]
        # Should be monotonically increasing
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]

    def test_cosine_restarts_multiple_cycles(self) -> None:
        """Test cosine restarts with multiple cycles."""
        cosine = create_cosine_config(num_cycles=4.0)
        config = create_lr_scheduler_config(
            scheduler_type="cosine_restarts",
            cosine_config=cosine,
            total_steps=400,
            base_lr=1e-4,
        )

        # Each cycle is 100 steps
        # At start of each cycle, LR should be at peak
        for cycle_start in [0, 100, 200, 300]:
            lr = calculate_lr_at_step(cycle_start, config)
            assert lr == pytest.approx(1e-4)

        # At middle of each cycle, LR should be lower
        for cycle_mid in [50, 150, 250, 350]:
            lr = calculate_lr_at_step(cycle_mid, config)
            assert lr < 1e-4

    def test_lr_schedule_plot_data(self) -> None:
        """Test that plot data accurately represents schedule."""
        warmup = create_warmup_config(warmup_steps=10)
        config = create_lr_scheduler_config(
            scheduler_type="cosine",
            warmup_config=warmup,
            total_steps=100,
            base_lr=1e-4,
        )

        steps, lrs = plot_lr_schedule(config, num_points=20)

        # Verify each point matches calculate_lr_at_step
        for step, lr in zip(steps, lrs, strict=True):
            expected = calculate_lr_at_step(step, config)
            assert lr == pytest.approx(expected)
