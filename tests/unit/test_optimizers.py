"""Tests for training.optimizers module."""

from __future__ import annotations

import pytest

from hf_gtc.training.optimizers import (
    VALID_MOMENTUM_TYPES,
    VALID_OPTIMIZER_TYPES,
    VALID_WEIGHT_DECAY_TYPES,
    AdafactorConfig,
    AdamWConfig,
    LionConfig,
    MomentumType,
    OptimizerConfig,
    OptimizerStats,
    OptimizerType,
    SophiaConfig,
    WeightDecayType,
    calculate_optimizer_memory,
    compare_optimizers,
    create_adafactor_config,
    create_adamw_config,
    create_lion_config,
    create_optimizer_config,
    create_sophia_config,
    estimate_convergence_speed,
    format_optimizer_stats,
    get_momentum_type,
    get_optimizer_type,
    get_param_groups,
    get_recommended_optimizer_config,
    get_weight_decay_type,
    list_momentum_types,
    list_optimizer_types,
    list_weight_decay_types,
    validate_adafactor_config,
    validate_adamw_config,
    validate_lion_config,
    validate_optimizer_config,
    validate_sophia_config,
)


class TestOptimizerType:
    """Tests for OptimizerType enum."""

    def test_adamw_value(self) -> None:
        """AdamW has correct value."""
        assert OptimizerType.ADAMW.value == "adamw"

    def test_adam_value(self) -> None:
        """Adam has correct value."""
        assert OptimizerType.ADAM.value == "adam"

    def test_sgd_value(self) -> None:
        """SGD has correct value."""
        assert OptimizerType.SGD.value == "sgd"

    def test_lion_value(self) -> None:
        """Lion has correct value."""
        assert OptimizerType.LION.value == "lion"

    def test_sophia_value(self) -> None:
        """Sophia has correct value."""
        assert OptimizerType.SOPHIA.value == "sophia"

    def test_adafactor_value(self) -> None:
        """Adafactor has correct value."""
        assert OptimizerType.ADAFACTOR.value == "adafactor"

    def test_adam_8bit_value(self) -> None:
        """Adam 8bit has correct value."""
        assert OptimizerType.ADAM_8BIT.value == "adam_8bit"

    def test_paged_adamw_value(self) -> None:
        """Paged AdamW has correct value."""
        assert OptimizerType.PAGED_ADAMW.value == "paged_adamw"

    def test_valid_optimizer_types_frozenset(self) -> None:
        """VALID_OPTIMIZER_TYPES is a frozenset."""
        assert isinstance(VALID_OPTIMIZER_TYPES, frozenset)
        assert len(VALID_OPTIMIZER_TYPES) == 8

    def test_all_enum_values_in_valid_set(self) -> None:
        """All enum values are in VALID_OPTIMIZER_TYPES."""
        for opt_type in OptimizerType:
            assert opt_type.value in VALID_OPTIMIZER_TYPES


class TestWeightDecayType:
    """Tests for WeightDecayType enum."""

    def test_decoupled_value(self) -> None:
        """Decoupled has correct value."""
        assert WeightDecayType.DECOUPLED.value == "decoupled"

    def test_l2_value(self) -> None:
        """L2 has correct value."""
        assert WeightDecayType.L2.value == "l2"

    def test_none_value(self) -> None:
        """None has correct value."""
        assert WeightDecayType.NONE.value == "none"

    def test_valid_weight_decay_types_frozenset(self) -> None:
        """VALID_WEIGHT_DECAY_TYPES is a frozenset."""
        assert isinstance(VALID_WEIGHT_DECAY_TYPES, frozenset)
        assert len(VALID_WEIGHT_DECAY_TYPES) == 3

    def test_all_enum_values_in_valid_set(self) -> None:
        """All enum values are in VALID_WEIGHT_DECAY_TYPES."""
        for wd_type in WeightDecayType:
            assert wd_type.value in VALID_WEIGHT_DECAY_TYPES


class TestMomentumType:
    """Tests for MomentumType enum."""

    def test_standard_value(self) -> None:
        """Standard has correct value."""
        assert MomentumType.STANDARD.value == "standard"

    def test_nesterov_value(self) -> None:
        """Nesterov has correct value."""
        assert MomentumType.NESTEROV.value == "nesterov"

    def test_heavy_ball_value(self) -> None:
        """Heavy ball has correct value."""
        assert MomentumType.HEAVY_BALL.value == "heavy_ball"

    def test_valid_momentum_types_frozenset(self) -> None:
        """VALID_MOMENTUM_TYPES is a frozenset."""
        assert isinstance(VALID_MOMENTUM_TYPES, frozenset)
        assert len(VALID_MOMENTUM_TYPES) == 3

    def test_all_enum_values_in_valid_set(self) -> None:
        """All enum values are in VALID_MOMENTUM_TYPES."""
        for mom_type in MomentumType:
            assert mom_type.value in VALID_MOMENTUM_TYPES


class TestAdamWConfig:
    """Tests for AdamWConfig dataclass."""

    def test_create_config(self) -> None:
        """Create AdamW config."""
        config = AdamWConfig(
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
        )
        assert config.lr == pytest.approx(1e-4)
        assert config.betas == (0.9, 0.999)
        assert config.eps == pytest.approx(1e-8)
        assert config.weight_decay == pytest.approx(0.01)
        assert config.amsgrad is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        with pytest.raises(AttributeError):
            config.lr = 1e-3  # type: ignore[misc]


class TestLionConfig:
    """Tests for LionConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Lion config."""
        config = LionConfig(
            lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.01,
        )
        assert config.lr == pytest.approx(1e-4)
        assert config.betas == (0.9, 0.99)
        assert config.weight_decay == pytest.approx(0.01)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LionConfig(1e-4, (0.9, 0.99), 0.01)
        with pytest.raises(AttributeError):
            config.lr = 1e-3  # type: ignore[misc]


class TestSophiaConfig:
    """Tests for SophiaConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Sophia config."""
        config = SophiaConfig(
            lr=1e-4,
            betas=(0.965, 0.99),
            rho=0.04,
            weight_decay=0.1,
        )
        assert config.lr == pytest.approx(1e-4)
        assert config.betas == (0.965, 0.99)
        assert config.rho == pytest.approx(0.04)
        assert config.weight_decay == pytest.approx(0.1)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SophiaConfig(1e-4, (0.965, 0.99), 0.04, 0.1)
        with pytest.raises(AttributeError):
            config.rho = 0.05  # type: ignore[misc]


class TestAdafactorConfig:
    """Tests for AdafactorConfig dataclass."""

    def test_create_config(self) -> None:
        """Create Adafactor config."""
        config = AdafactorConfig(
            lr=None,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            scale_parameter=True,
        )
        assert config.lr is None
        assert config.eps == (1e-30, 1e-3)
        assert config.clip_threshold == pytest.approx(1.0)
        assert config.scale_parameter is True

    def test_config_with_lr(self) -> None:
        """Create Adafactor config with explicit lr."""
        config = AdafactorConfig(
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            scale_parameter=False,
        )
        assert config.lr == pytest.approx(1e-3)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = AdafactorConfig(None, (1e-30, 1e-3), 1.0, True)
        with pytest.raises(AttributeError):
            config.lr = 1e-3  # type: ignore[misc]


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_create_config(self) -> None:
        """Create optimizer config."""
        adamw = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW,
            adamw_config=adamw,
            lion_config=None,
            sophia_config=None,
            adafactor_config=None,
        )
        assert config.optimizer_type == OptimizerType.ADAMW
        assert config.adamw_config is not None
        assert config.lion_config is None


class TestOptimizerStats:
    """Tests for OptimizerStats dataclass."""

    def test_create_stats(self) -> None:
        """Create optimizer stats."""
        stats = OptimizerStats(
            memory_mb=1000.0,
            convergence_speed=1.0,
            stability_score=0.9,
            recommended_for=("fine_tuning", "classification"),
        )
        assert stats.memory_mb == pytest.approx(1000.0)
        assert stats.convergence_speed == pytest.approx(1.0)
        assert stats.stability_score == pytest.approx(0.9)
        assert "fine_tuning" in stats.recommended_for


class TestValidateAdamWConfig:
    """Tests for validate_adamw_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        validate_adamw_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_adamw_config(None)  # type: ignore[arg-type]

    def test_zero_lr_raises(self) -> None:
        """Zero lr raises ValueError."""
        config = AdamWConfig(0.0, (0.9, 0.999), 1e-8, 0.01, False)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_adamw_config(config)

    def test_negative_lr_raises(self) -> None:
        """Negative lr raises ValueError."""
        config = AdamWConfig(-1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_adamw_config(config)

    def test_beta1_zero_raises(self) -> None:
        """Beta1 of 0 raises ValueError."""
        config = AdamWConfig(1e-4, (0.0, 0.999), 1e-8, 0.01, False)
        with pytest.raises(ValueError, match="beta1 must be in"):
            validate_adamw_config(config)

    def test_beta1_one_raises(self) -> None:
        """Beta1 of 1 raises ValueError."""
        config = AdamWConfig(1e-4, (1.0, 0.999), 1e-8, 0.01, False)
        with pytest.raises(ValueError, match="beta1 must be in"):
            validate_adamw_config(config)

    def test_beta2_zero_raises(self) -> None:
        """Beta2 of 0 raises ValueError."""
        config = AdamWConfig(1e-4, (0.9, 0.0), 1e-8, 0.01, False)
        with pytest.raises(ValueError, match="beta2 must be in"):
            validate_adamw_config(config)

    def test_beta2_one_raises(self) -> None:
        """Beta2 of 1 raises ValueError."""
        config = AdamWConfig(1e-4, (0.9, 1.0), 1e-8, 0.01, False)
        with pytest.raises(ValueError, match="beta2 must be in"):
            validate_adamw_config(config)

    def test_zero_eps_raises(self) -> None:
        """Zero eps raises ValueError."""
        config = AdamWConfig(1e-4, (0.9, 0.999), 0.0, 0.01, False)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_adamw_config(config)

    def test_negative_eps_raises(self) -> None:
        """Negative eps raises ValueError."""
        config = AdamWConfig(1e-4, (0.9, 0.999), -1e-8, 0.01, False)
        with pytest.raises(ValueError, match="eps must be positive"):
            validate_adamw_config(config)

    def test_negative_weight_decay_raises(self) -> None:
        """Negative weight decay raises ValueError."""
        config = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, -0.01, False)
        with pytest.raises(ValueError, match="weight_decay cannot be negative"):
            validate_adamw_config(config)

    def test_zero_weight_decay_valid(self) -> None:
        """Zero weight decay is valid."""
        config = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.0, False)
        validate_adamw_config(config)

    def test_amsgrad_true_valid(self) -> None:
        """AMSGrad true is valid."""
        config = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, True)
        validate_adamw_config(config)


class TestValidateLionConfig:
    """Tests for validate_lion_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LionConfig(1e-4, (0.9, 0.99), 0.01)
        validate_lion_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_lion_config(None)  # type: ignore[arg-type]

    def test_zero_lr_raises(self) -> None:
        """Zero lr raises ValueError."""
        config = LionConfig(0.0, (0.9, 0.99), 0.01)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_lion_config(config)

    def test_negative_lr_raises(self) -> None:
        """Negative lr raises ValueError."""
        config = LionConfig(-1e-4, (0.9, 0.99), 0.01)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_lion_config(config)

    def test_beta1_invalid_raises(self) -> None:
        """Invalid beta1 raises ValueError."""
        config = LionConfig(1e-4, (0.0, 0.99), 0.01)
        with pytest.raises(ValueError, match="beta1 must be in"):
            validate_lion_config(config)

    def test_beta2_invalid_raises(self) -> None:
        """Invalid beta2 raises ValueError."""
        config = LionConfig(1e-4, (0.9, 1.0), 0.01)
        with pytest.raises(ValueError, match="beta2 must be in"):
            validate_lion_config(config)

    def test_negative_weight_decay_raises(self) -> None:
        """Negative weight decay raises ValueError."""
        config = LionConfig(1e-4, (0.9, 0.99), -0.01)
        with pytest.raises(ValueError, match="weight_decay cannot be negative"):
            validate_lion_config(config)


class TestValidateSophiaConfig:
    """Tests for validate_sophia_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SophiaConfig(1e-4, (0.965, 0.99), 0.04, 0.1)
        validate_sophia_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_sophia_config(None)  # type: ignore[arg-type]

    def test_zero_lr_raises(self) -> None:
        """Zero lr raises ValueError."""
        config = SophiaConfig(0.0, (0.965, 0.99), 0.04, 0.1)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_sophia_config(config)

    def test_negative_lr_raises(self) -> None:
        """Negative lr raises ValueError."""
        config = SophiaConfig(-1e-4, (0.965, 0.99), 0.04, 0.1)
        with pytest.raises(ValueError, match="lr must be positive"):
            validate_sophia_config(config)

    def test_beta1_invalid_raises(self) -> None:
        """Invalid beta1 raises ValueError."""
        config = SophiaConfig(1e-4, (0.0, 0.99), 0.04, 0.1)
        with pytest.raises(ValueError, match="beta1 must be in"):
            validate_sophia_config(config)

    def test_beta2_invalid_raises(self) -> None:
        """Invalid beta2 raises ValueError."""
        config = SophiaConfig(1e-4, (0.965, 0.0), 0.04, 0.1)
        with pytest.raises(ValueError, match="beta2 must be in"):
            validate_sophia_config(config)

    def test_zero_rho_raises(self) -> None:
        """Zero rho raises ValueError."""
        config = SophiaConfig(1e-4, (0.965, 0.99), 0.0, 0.1)
        with pytest.raises(ValueError, match="rho must be positive"):
            validate_sophia_config(config)

    def test_negative_rho_raises(self) -> None:
        """Negative rho raises ValueError."""
        config = SophiaConfig(1e-4, (0.965, 0.99), -0.04, 0.1)
        with pytest.raises(ValueError, match="rho must be positive"):
            validate_sophia_config(config)

    def test_negative_weight_decay_raises(self) -> None:
        """Negative weight decay raises ValueError."""
        config = SophiaConfig(1e-4, (0.965, 0.99), 0.04, -0.1)
        with pytest.raises(ValueError, match="weight_decay cannot be negative"):
            validate_sophia_config(config)


class TestValidateAdafactorConfig:
    """Tests for validate_adafactor_config function."""

    def test_valid_config_with_none_lr(self) -> None:
        """Valid config with None lr passes validation."""
        config = AdafactorConfig(None, (1e-30, 1e-3), 1.0, True)
        validate_adafactor_config(config)

    def test_valid_config_with_lr(self) -> None:
        """Valid config with lr passes validation."""
        config = AdafactorConfig(1e-3, (1e-30, 1e-3), 1.0, True)
        validate_adafactor_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_adafactor_config(None)  # type: ignore[arg-type]

    def test_zero_lr_raises(self) -> None:
        """Zero lr raises ValueError."""
        config = AdafactorConfig(0.0, (1e-30, 1e-3), 1.0, True)
        with pytest.raises(ValueError, match="lr must be positive when specified"):
            validate_adafactor_config(config)

    def test_negative_lr_raises(self) -> None:
        """Negative lr raises ValueError."""
        config = AdafactorConfig(-1e-3, (1e-30, 1e-3), 1.0, True)
        with pytest.raises(ValueError, match="lr must be positive when specified"):
            validate_adafactor_config(config)

    def test_zero_eps1_raises(self) -> None:
        """Zero eps1 raises ValueError."""
        config = AdafactorConfig(None, (0.0, 1e-3), 1.0, True)
        with pytest.raises(ValueError, match="eps1 must be positive"):
            validate_adafactor_config(config)

    def test_negative_eps1_raises(self) -> None:
        """Negative eps1 raises ValueError."""
        config = AdafactorConfig(None, (-1e-30, 1e-3), 1.0, True)
        with pytest.raises(ValueError, match="eps1 must be positive"):
            validate_adafactor_config(config)

    def test_zero_eps2_raises(self) -> None:
        """Zero eps2 raises ValueError."""
        config = AdafactorConfig(None, (1e-30, 0.0), 1.0, True)
        with pytest.raises(ValueError, match="eps2 must be positive"):
            validate_adafactor_config(config)

    def test_negative_eps2_raises(self) -> None:
        """Negative eps2 raises ValueError."""
        config = AdafactorConfig(None, (1e-30, -1e-3), 1.0, True)
        with pytest.raises(ValueError, match="eps2 must be positive"):
            validate_adafactor_config(config)

    def test_zero_clip_threshold_raises(self) -> None:
        """Zero clip threshold raises ValueError."""
        config = AdafactorConfig(None, (1e-30, 1e-3), 0.0, True)
        with pytest.raises(ValueError, match="clip_threshold must be positive"):
            validate_adafactor_config(config)

    def test_negative_clip_threshold_raises(self) -> None:
        """Negative clip threshold raises ValueError."""
        config = AdafactorConfig(None, (1e-30, 1e-3), -1.0, True)
        with pytest.raises(ValueError, match="clip_threshold must be positive"):
            validate_adafactor_config(config)


class TestValidateOptimizerConfig:
    """Tests for validate_optimizer_config function."""

    def test_valid_adamw_config(self) -> None:
        """Valid AdamW config passes validation."""
        adamw = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        config = OptimizerConfig(OptimizerType.ADAMW, adamw, None, None, None)
        validate_optimizer_config(config)

    def test_valid_adam_config(self) -> None:
        """Valid Adam config passes validation."""
        adamw = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.0, False)
        config = OptimizerConfig(OptimizerType.ADAM, adamw, None, None, None)
        validate_optimizer_config(config)

    def test_valid_paged_adamw_config(self) -> None:
        """Valid Paged AdamW config passes validation."""
        adamw = AdamWConfig(1e-4, (0.9, 0.999), 1e-8, 0.01, False)
        config = OptimizerConfig(OptimizerType.PAGED_ADAMW, adamw, None, None, None)
        validate_optimizer_config(config)

    def test_valid_lion_config(self) -> None:
        """Valid Lion config passes validation."""
        lion = LionConfig(1e-4, (0.9, 0.99), 0.01)
        config = OptimizerConfig(OptimizerType.LION, None, lion, None, None)
        validate_optimizer_config(config)

    def test_valid_sophia_config(self) -> None:
        """Valid Sophia config passes validation."""
        sophia = SophiaConfig(1e-4, (0.965, 0.99), 0.04, 0.1)
        config = OptimizerConfig(OptimizerType.SOPHIA, None, None, sophia, None)
        validate_optimizer_config(config)

    def test_valid_adafactor_config(self) -> None:
        """Valid Adafactor config passes validation."""
        adafactor = AdafactorConfig(None, (1e-30, 1e-3), 1.0, True)
        config = OptimizerConfig(OptimizerType.ADAFACTOR, None, None, None, adafactor)
        validate_optimizer_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_optimizer_config(None)  # type: ignore[arg-type]

    def test_missing_adamw_config_raises(self) -> None:
        """Missing AdamW config raises ValueError."""
        config = OptimizerConfig(OptimizerType.ADAMW, None, None, None, None)
        with pytest.raises(ValueError, match="adamw_config required"):
            validate_optimizer_config(config)

    def test_missing_lion_config_raises(self) -> None:
        """Missing Lion config raises ValueError."""
        config = OptimizerConfig(OptimizerType.LION, None, None, None, None)
        with pytest.raises(ValueError, match="lion_config required"):
            validate_optimizer_config(config)

    def test_missing_sophia_config_raises(self) -> None:
        """Missing Sophia config raises ValueError."""
        config = OptimizerConfig(OptimizerType.SOPHIA, None, None, None, None)
        with pytest.raises(ValueError, match="sophia_config required"):
            validate_optimizer_config(config)

    def test_missing_adafactor_config_raises(self) -> None:
        """Missing Adafactor config raises ValueError."""
        config = OptimizerConfig(OptimizerType.ADAFACTOR, None, None, None, None)
        with pytest.raises(ValueError, match="adafactor_config required"):
            validate_optimizer_config(config)


class TestCreateAdamWConfig:
    """Tests for create_adamw_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_adamw_config()
        assert config.lr == pytest.approx(1e-4)
        assert config.betas == (0.9, 0.999)
        assert config.eps == pytest.approx(1e-8)
        assert config.weight_decay == pytest.approx(0.01)
        assert config.amsgrad is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_adamw_config(
            lr=3e-4,
            betas=(0.95, 0.998),
            eps=1e-6,
            weight_decay=0.1,
            amsgrad=True,
        )
        assert config.lr == pytest.approx(3e-4)
        assert config.betas == (0.95, 0.998)
        assert config.eps == pytest.approx(1e-6)
        assert config.weight_decay == pytest.approx(0.1)
        assert config.amsgrad is True

    def test_zero_lr_raises(self) -> None:
        """Zero lr raises ValueError."""
        with pytest.raises(ValueError, match="lr must be positive"):
            create_adamw_config(lr=0.0)


class TestCreateLionConfig:
    """Tests for create_lion_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_lion_config()
        assert config.lr == pytest.approx(1e-4)
        assert config.betas == (0.9, 0.99)
        assert config.weight_decay == pytest.approx(0.01)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_lion_config(
            lr=3e-5,
            betas=(0.95, 0.98),
            weight_decay=0.0,
        )
        assert config.lr == pytest.approx(3e-5)
        assert config.betas == (0.95, 0.98)
        assert config.weight_decay == pytest.approx(0.0)

    def test_negative_lr_raises(self) -> None:
        """Negative lr raises ValueError."""
        with pytest.raises(ValueError, match="lr must be positive"):
            create_lion_config(lr=-1e-4)


class TestCreateSophiaConfig:
    """Tests for create_sophia_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_sophia_config()
        assert config.lr == pytest.approx(1e-4)
        assert config.betas == (0.965, 0.99)
        assert config.rho == pytest.approx(0.04)
        assert config.weight_decay == pytest.approx(0.1)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_sophia_config(
            lr=2e-4,
            betas=(0.97, 0.995),
            rho=0.05,
            weight_decay=0.2,
        )
        assert config.lr == pytest.approx(2e-4)
        assert config.rho == pytest.approx(0.05)

    def test_zero_rho_raises(self) -> None:
        """Zero rho raises ValueError."""
        with pytest.raises(ValueError, match="rho must be positive"):
            create_sophia_config(rho=0.0)


class TestCreateAdafactorConfig:
    """Tests for create_adafactor_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_adafactor_config()
        assert config.lr is None
        assert config.eps == (1e-30, 1e-3)
        assert config.clip_threshold == pytest.approx(1.0)
        assert config.scale_parameter is True

    def test_config_with_lr(self) -> None:
        """Create config with explicit lr."""
        config = create_adafactor_config(lr=1e-3)
        assert config.lr == pytest.approx(1e-3)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_adafactor_config(
            lr=5e-4,
            eps=(1e-28, 1e-2),
            clip_threshold=2.0,
            scale_parameter=False,
        )
        assert config.lr == pytest.approx(5e-4)
        assert config.clip_threshold == pytest.approx(2.0)
        assert config.scale_parameter is False

    def test_zero_clip_threshold_raises(self) -> None:
        """Zero clip threshold raises ValueError."""
        with pytest.raises(ValueError, match="clip_threshold must be positive"):
            create_adafactor_config(clip_threshold=0.0)


class TestCreateOptimizerConfig:
    """Tests for create_optimizer_config function."""

    def test_default_adamw_config(self) -> None:
        """Create default AdamW config."""
        config = create_optimizer_config()
        assert config.optimizer_type == OptimizerType.ADAMW
        assert config.adamw_config is not None

    def test_adamw_with_custom_config(self) -> None:
        """Create AdamW with custom config."""
        adamw = create_adamw_config(lr=3e-4)
        config = create_optimizer_config("adamw", adamw_config=adamw)
        assert config.adamw_config.lr == pytest.approx(3e-4)

    def test_adam_config(self) -> None:
        """Create Adam config."""
        config = create_optimizer_config("adam")
        assert config.optimizer_type == OptimizerType.ADAM
        assert config.adamw_config is not None

    def test_sgd_config(self) -> None:
        """Create SGD config."""
        config = create_optimizer_config("sgd")
        assert config.optimizer_type == OptimizerType.SGD
        assert config.adamw_config is not None

    def test_lion_config(self) -> None:
        """Create Lion config."""
        config = create_optimizer_config("lion")
        assert config.optimizer_type == OptimizerType.LION
        assert config.lion_config is not None

    def test_lion_with_custom_config(self) -> None:
        """Create Lion with custom config."""
        lion = create_lion_config(lr=3e-5)
        config = create_optimizer_config("lion", lion_config=lion)
        assert config.lion_config.lr == pytest.approx(3e-5)

    def test_sophia_config(self) -> None:
        """Create Sophia config."""
        config = create_optimizer_config("sophia")
        assert config.optimizer_type == OptimizerType.SOPHIA
        assert config.sophia_config is not None

    def test_adafactor_config(self) -> None:
        """Create Adafactor config."""
        config = create_optimizer_config("adafactor")
        assert config.optimizer_type == OptimizerType.ADAFACTOR
        assert config.adafactor_config is not None

    def test_adam_8bit_config(self) -> None:
        """Create Adam 8bit config."""
        config = create_optimizer_config("adam_8bit")
        assert config.optimizer_type == OptimizerType.ADAM_8BIT
        assert config.adamw_config is not None

    def test_paged_adamw_config(self) -> None:
        """Create Paged AdamW config."""
        config = create_optimizer_config("paged_adamw")
        assert config.optimizer_type == OptimizerType.PAGED_ADAMW
        assert config.adamw_config is not None

    def test_invalid_optimizer_type_raises(self) -> None:
        """Invalid optimizer type raises ValueError."""
        with pytest.raises(ValueError, match="optimizer_type must be one of"):
            create_optimizer_config("invalid")


class TestCalculateOptimizerMemory:
    """Tests for calculate_optimizer_memory function."""

    def test_adamw_fp32(self) -> None:
        """Calculate AdamW memory in FP32."""
        mem = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        # 1M params * 4 bytes * 2 (m+v) / 1MB = 8 MB
        assert mem == pytest.approx(7.63, rel=0.01)

    def test_adam_fp32(self) -> None:
        """Calculate Adam memory in FP32."""
        mem = calculate_optimizer_memory(1_000_000, "adam", "fp32")
        assert mem > 0

    def test_sgd_fp32(self) -> None:
        """SGD uses less memory than Adam."""
        mem_sgd = calculate_optimizer_memory(1_000_000, "sgd", "fp32")
        mem_adam = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        assert mem_sgd < mem_adam

    def test_lion_fp32(self) -> None:
        """Lion uses less memory than AdamW."""
        mem_lion = calculate_optimizer_memory(1_000_000, "lion", "fp32")
        mem_adamw = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        assert mem_lion < mem_adamw

    def test_adafactor_fp32(self) -> None:
        """Adafactor uses less memory than AdamW."""
        mem_adafactor = calculate_optimizer_memory(1_000_000, "adafactor", "fp32")
        mem_adamw = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        assert mem_adafactor < mem_adamw

    def test_adam_8bit(self) -> None:
        """8-bit Adam uses less memory than FP32 Adam."""
        mem_8bit = calculate_optimizer_memory(1_000_000, "adam_8bit", "8bit")
        mem_fp32 = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        assert mem_8bit < mem_fp32

    def test_fp16_precision(self) -> None:
        """FP16 uses less memory than FP32."""
        mem_fp16 = calculate_optimizer_memory(1_000_000, "adamw", "fp16")
        mem_fp32 = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        assert mem_fp16 < mem_fp32

    def test_zero_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            calculate_optimizer_memory(0, "adamw")

    def test_negative_params_raises(self) -> None:
        """Negative model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            calculate_optimizer_memory(-100, "adamw")

    def test_invalid_optimizer_raises(self) -> None:
        """Invalid optimizer type raises ValueError."""
        with pytest.raises(ValueError, match="optimizer_type must be one of"):
            calculate_optimizer_memory(1_000_000, "invalid")

    def test_sophia_memory(self) -> None:
        """Sophia memory calculation."""
        mem = calculate_optimizer_memory(1_000_000, "sophia", "fp32")
        assert mem > 0

    def test_paged_adamw_memory(self) -> None:
        """Paged AdamW memory calculation."""
        mem = calculate_optimizer_memory(1_000_000, "paged_adamw", "fp32")
        assert mem > 0

    def test_unknown_precision_uses_default(self) -> None:
        """Unknown precision uses FP32 default."""
        mem_unknown = calculate_optimizer_memory(1_000_000, "adamw", "unknown")
        mem_fp32 = calculate_optimizer_memory(1_000_000, "adamw", "fp32")
        assert mem_unknown == mem_fp32


class TestEstimateConvergenceSpeed:
    """Tests for estimate_convergence_speed function."""

    def test_adamw_baseline(self) -> None:
        """AdamW is baseline (1.0)."""
        speed = estimate_convergence_speed("adamw", "7b", "fine_tuning")
        assert speed > 0.9

    def test_sophia_faster(self) -> None:
        """Sophia converges faster than AdamW."""
        speed_sophia = estimate_convergence_speed("sophia", "7b", "fine_tuning")
        speed_adamw = estimate_convergence_speed("adamw", "7b", "fine_tuning")
        assert speed_sophia > speed_adamw

    def test_sgd_slower(self) -> None:
        """SGD converges slower than AdamW."""
        speed_sgd = estimate_convergence_speed("sgd", "7b", "fine_tuning")
        speed_adamw = estimate_convergence_speed("adamw", "7b", "fine_tuning")
        assert speed_sgd < speed_adamw

    def test_model_size_affects_speed(self) -> None:
        """Larger models converge slower."""
        speed_7b = estimate_convergence_speed("adamw", "7b", "fine_tuning")
        speed_70b = estimate_convergence_speed("adamw", "70b", "fine_tuning")
        assert speed_7b >= speed_70b

    def test_fine_tuning_faster_than_pretraining(self) -> None:
        """Fine-tuning converges faster than pretraining."""
        speed_ft = estimate_convergence_speed("adamw", "7b", "fine_tuning")
        speed_pt = estimate_convergence_speed("adamw", "7b", "pretraining")
        assert speed_ft >= speed_pt

    def test_sophia_pretraining_boost(self) -> None:
        """Sophia gets extra boost in pretraining."""
        speed_sophia = estimate_convergence_speed("sophia", "7b", "pretraining")
        assert speed_sophia > 1.5

    def test_invalid_optimizer_raises(self) -> None:
        """Invalid optimizer raises ValueError."""
        with pytest.raises(ValueError, match="optimizer_type must be one of"):
            estimate_convergence_speed("invalid")

    def test_unknown_model_size_uses_default(self) -> None:
        """Unknown model size uses default factor."""
        speed = estimate_convergence_speed("adamw", "unknown_size", "fine_tuning")
        assert speed > 0

    def test_unknown_task_type_uses_default(self) -> None:
        """Unknown task type uses default factor."""
        speed = estimate_convergence_speed("adamw", "7b", "unknown_task")
        assert speed > 0

    def test_small_model_speed(self) -> None:
        """Small model speed estimation."""
        speed = estimate_convergence_speed("adamw", "small", "fine_tuning")
        assert speed > 0

    def test_13b_model_speed(self) -> None:
        """13B model speed estimation."""
        speed = estimate_convergence_speed("adamw", "13b", "fine_tuning")
        assert speed > 0


class TestCompareOptimizers:
    """Tests for compare_optimizers function."""

    def test_compare_adamw_lion(self) -> None:
        """Compare AdamW and Lion."""
        stats = compare_optimizers(("adamw", "lion"))
        assert "adamw" in stats
        assert "lion" in stats
        assert stats["adamw"].memory_mb > 0
        assert stats["lion"].memory_mb > 0

    def test_compare_all_optimizers(self) -> None:
        """Compare all optimizer types."""
        stats = compare_optimizers(tuple(VALID_OPTIMIZER_TYPES))
        assert len(stats) == len(VALID_OPTIMIZER_TYPES)

    def test_stats_have_all_fields(self) -> None:
        """Stats have all required fields."""
        stats = compare_optimizers(("adamw",))
        adamw_stats = stats["adamw"]
        assert adamw_stats.memory_mb > 0
        assert adamw_stats.convergence_speed > 0
        assert 0 <= adamw_stats.stability_score <= 1
        assert len(adamw_stats.recommended_for) > 0

    def test_invalid_optimizer_raises(self) -> None:
        """Invalid optimizer raises ValueError."""
        with pytest.raises(ValueError, match="optimizer_type must be one of"):
            compare_optimizers(("invalid",))

    def test_zero_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            compare_optimizers(("adamw",), model_params=0)

    def test_negative_params_raises(self) -> None:
        """Negative model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            compare_optimizers(("adamw",), model_params=-100)

    def test_custom_model_params(self) -> None:
        """Compare with custom model params."""
        stats = compare_optimizers(("adamw",), model_params=1_000_000)
        assert stats["adamw"].memory_mb > 0


class TestGetParamGroups:
    """Tests for get_param_groups function."""

    def test_default_groups(self) -> None:
        """Get default parameter groups."""
        groups = get_param_groups(1_000_000, "adamw")
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == pytest.approx(0.01)
        assert groups[1]["weight_decay"] == pytest.approx(0.0)

    def test_custom_weight_decay(self) -> None:
        """Get groups with custom weight decay."""
        groups = get_param_groups(1_000_000, "adamw", weight_decay=0.1)
        assert groups[0]["weight_decay"] == pytest.approx(0.1)

    def test_custom_no_decay_keywords(self) -> None:
        """Get groups with custom no-decay keywords."""
        keywords = ("bias", "norm")
        groups = get_param_groups(1_000_000, "adamw", no_decay_keywords=keywords)
        assert groups[1]["no_decay_keywords"] == keywords

    def test_zero_params_raises(self) -> None:
        """Zero model params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            get_param_groups(0, "adamw")

    def test_negative_weight_decay_raises(self) -> None:
        """Negative weight decay raises ValueError."""
        with pytest.raises(ValueError, match="weight_decay cannot be negative"):
            get_param_groups(1_000_000, "adamw", weight_decay=-0.01)


class TestFormatOptimizerStats:
    """Tests for format_optimizer_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = OptimizerStats(1000.0, 1.0, 0.9, ("fine_tuning",))
        formatted = format_optimizer_stats(stats)
        assert "Memory:" in formatted
        assert "1,000.00 MB" in formatted
        assert "Convergence Speed:" in formatted
        assert "Stability Score:" in formatted
        assert "Recommended For:" in formatted

    def test_format_with_multiple_recommendations(self) -> None:
        """Format stats with multiple recommendations."""
        stats = OptimizerStats(500.0, 1.5, 0.85, ("pretraining", "large_models"))
        formatted = format_optimizer_stats(stats)
        assert "pretraining" in formatted
        assert "large_models" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_optimizer_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedOptimizerConfig:
    """Tests for get_recommended_optimizer_config function."""

    def test_7b_fine_tuning(self) -> None:
        """Get config for 7B fine-tuning."""
        config = get_recommended_optimizer_config("7b", "fine_tuning")
        assert config.optimizer_type == OptimizerType.ADAMW

    def test_70b_fine_tuning(self) -> None:
        """Get config for 70B fine-tuning."""
        config = get_recommended_optimizer_config("70b", "fine_tuning")
        assert config.optimizer_type == OptimizerType.LION

    def test_13b_pretraining(self) -> None:
        """Get config for 13B pretraining."""
        config = get_recommended_optimizer_config("13b", "pretraining")
        assert config.optimizer_type == OptimizerType.SOPHIA

    def test_7b_pretraining(self) -> None:
        """Get config for 7B pretraining."""
        config = get_recommended_optimizer_config("7b", "pretraining")
        assert config.optimizer_type == OptimizerType.ADAMW

    def test_small_fine_tuning(self) -> None:
        """Get config for small model fine-tuning."""
        config = get_recommended_optimizer_config("small", "fine_tuning")
        assert config.optimizer_type == OptimizerType.ADAMW

    def test_memory_constrained_70b(self) -> None:
        """Get config for 70B with memory constraints."""
        config = get_recommended_optimizer_config(
            "70b", "fine_tuning", memory_constrained=True
        )
        assert config.optimizer_type == OptimizerType.PAGED_ADAMW

    def test_memory_constrained_7b(self) -> None:
        """Get config for 7B with memory constraints."""
        config = get_recommended_optimizer_config(
            "7b", "fine_tuning", memory_constrained=True
        )
        assert config.optimizer_type == OptimizerType.ADAM_8BIT

    def test_memory_constrained_13b(self) -> None:
        """Get config for 13B with memory constraints."""
        config = get_recommended_optimizer_config(
            "13b", "fine_tuning", memory_constrained=True
        )
        assert config.optimizer_type == OptimizerType.PAGED_ADAMW

    def test_invalid_model_size_raises(self) -> None:
        """Invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_optimizer_config("invalid")

    def test_case_insensitive_model_size(self) -> None:
        """Model size is case insensitive."""
        config1 = get_recommended_optimizer_config("7B")
        config2 = get_recommended_optimizer_config("7b")
        assert config1.optimizer_type == config2.optimizer_type


class TestListOptimizerTypes:
    """Tests for list_optimizer_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_optimizer_types()
        assert types == sorted(types)

    def test_contains_adamw(self) -> None:
        """Contains adamw."""
        types = list_optimizer_types()
        assert "adamw" in types

    def test_contains_lion(self) -> None:
        """Contains lion."""
        types = list_optimizer_types()
        assert "lion" in types

    def test_contains_sophia(self) -> None:
        """Contains sophia."""
        types = list_optimizer_types()
        assert "sophia" in types

    def test_correct_count(self) -> None:
        """Returns correct count."""
        types = list_optimizer_types()
        assert len(types) == 8


class TestListWeightDecayTypes:
    """Tests for list_weight_decay_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_weight_decay_types()
        assert types == sorted(types)

    def test_contains_decoupled(self) -> None:
        """Contains decoupled."""
        types = list_weight_decay_types()
        assert "decoupled" in types

    def test_contains_l2(self) -> None:
        """Contains l2."""
        types = list_weight_decay_types()
        assert "l2" in types

    def test_correct_count(self) -> None:
        """Returns correct count."""
        types = list_weight_decay_types()
        assert len(types) == 3


class TestListMomentumTypes:
    """Tests for list_momentum_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_momentum_types()
        assert types == sorted(types)

    def test_contains_standard(self) -> None:
        """Contains standard."""
        types = list_momentum_types()
        assert "standard" in types

    def test_contains_nesterov(self) -> None:
        """Contains nesterov."""
        types = list_momentum_types()
        assert "nesterov" in types

    def test_correct_count(self) -> None:
        """Returns correct count."""
        types = list_momentum_types()
        assert len(types) == 3


class TestGetOptimizerType:
    """Tests for get_optimizer_type function."""

    def test_get_adamw(self) -> None:
        """Get adamw."""
        assert get_optimizer_type("adamw") == OptimizerType.ADAMW

    def test_get_lion(self) -> None:
        """Get lion."""
        assert get_optimizer_type("lion") == OptimizerType.LION

    def test_get_sophia(self) -> None:
        """Get sophia."""
        assert get_optimizer_type("sophia") == OptimizerType.SOPHIA

    def test_get_adafactor(self) -> None:
        """Get adafactor."""
        assert get_optimizer_type("adafactor") == OptimizerType.ADAFACTOR

    def test_get_adam_8bit(self) -> None:
        """Get adam_8bit."""
        assert get_optimizer_type("adam_8bit") == OptimizerType.ADAM_8BIT

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="optimizer type must be one of"):
            get_optimizer_type("invalid")


class TestGetWeightDecayType:
    """Tests for get_weight_decay_type function."""

    def test_get_decoupled(self) -> None:
        """Get decoupled."""
        assert get_weight_decay_type("decoupled") == WeightDecayType.DECOUPLED

    def test_get_l2(self) -> None:
        """Get l2."""
        assert get_weight_decay_type("l2") == WeightDecayType.L2

    def test_get_none(self) -> None:
        """Get none."""
        assert get_weight_decay_type("none") == WeightDecayType.NONE

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="weight decay type must be one of"):
            get_weight_decay_type("invalid")


class TestGetMomentumType:
    """Tests for get_momentum_type function."""

    def test_get_standard(self) -> None:
        """Get standard."""
        assert get_momentum_type("standard") == MomentumType.STANDARD

    def test_get_nesterov(self) -> None:
        """Get nesterov."""
        assert get_momentum_type("nesterov") == MomentumType.NESTEROV

    def test_get_heavy_ball(self) -> None:
        """Get heavy_ball."""
        assert get_momentum_type("heavy_ball") == MomentumType.HEAVY_BALL

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="momentum type must be one of"):
            get_momentum_type("invalid")
