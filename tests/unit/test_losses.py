"""Tests for training.losses module."""

from __future__ import annotations

import pytest

from hf_gtc.training.losses import (
    VALID_CONTRASTIVE_TYPES,
    VALID_LOSS_TYPES,
    VALID_REDUCTION_TYPES,
    ContrastiveLossConfig,
    ContrastiveType,
    FocalLossConfig,
    LabelSmoothingConfig,
    LossConfig,
    LossStats,
    LossType,
    ReductionType,
    apply_label_smoothing,
    calculate_contrastive_loss,
    calculate_focal_loss,
    compute_class_weights,
    create_contrastive_loss_config,
    create_focal_loss_config,
    create_label_smoothing_config,
    create_loss_config,
    create_loss_stats,
    format_loss_stats,
    get_contrastive_type,
    get_loss_type,
    get_recommended_loss_config,
    get_reduction_type,
    list_contrastive_types,
    list_loss_types,
    list_reduction_types,
    validate_contrastive_loss_config,
    validate_focal_loss_config,
    validate_label_smoothing_config,
    validate_loss_config,
    validate_loss_stats,
)


class TestLossType:
    """Tests for LossType enum."""

    def test_cross_entropy_value(self) -> None:
        """Cross entropy has correct value."""
        assert LossType.CROSS_ENTROPY.value == "cross_entropy"

    def test_focal_value(self) -> None:
        """Focal has correct value."""
        assert LossType.FOCAL.value == "focal"

    def test_contrastive_value(self) -> None:
        """Contrastive has correct value."""
        assert LossType.CONTRASTIVE.value == "contrastive"

    def test_triplet_value(self) -> None:
        """Triplet has correct value."""
        assert LossType.TRIPLET.value == "triplet"

    def test_cosine_value(self) -> None:
        """Cosine has correct value."""
        assert LossType.COSINE.value == "cosine"

    def test_kl_divergence_value(self) -> None:
        """KL divergence has correct value."""
        assert LossType.KL_DIVERGENCE.value == "kl_divergence"

    def test_mse_value(self) -> None:
        """MSE has correct value."""
        assert LossType.MSE.value == "mse"

    def test_mae_value(self) -> None:
        """MAE has correct value."""
        assert LossType.MAE.value == "mae"

    def test_valid_loss_types_frozenset(self) -> None:
        """VALID_LOSS_TYPES is a frozenset."""
        assert isinstance(VALID_LOSS_TYPES, frozenset)
        assert len(VALID_LOSS_TYPES) == 8

    def test_all_enum_values_in_valid_set(self) -> None:
        """All enum values are in VALID_LOSS_TYPES."""
        for loss_type in LossType:
            assert loss_type.value in VALID_LOSS_TYPES


class TestReductionType:
    """Tests for ReductionType enum."""

    def test_mean_value(self) -> None:
        """Mean has correct value."""
        assert ReductionType.MEAN.value == "mean"

    def test_sum_value(self) -> None:
        """Sum has correct value."""
        assert ReductionType.SUM.value == "sum"

    def test_none_value(self) -> None:
        """None has correct value."""
        assert ReductionType.NONE.value == "none"

    def test_batchmean_value(self) -> None:
        """Batchmean has correct value."""
        assert ReductionType.BATCHMEAN.value == "batchmean"

    def test_valid_reduction_types_frozenset(self) -> None:
        """VALID_REDUCTION_TYPES is a frozenset."""
        assert isinstance(VALID_REDUCTION_TYPES, frozenset)
        assert len(VALID_REDUCTION_TYPES) == 4

    def test_all_enum_values_in_valid_set(self) -> None:
        """All enum values are in VALID_REDUCTION_TYPES."""
        for reduction_type in ReductionType:
            assert reduction_type.value in VALID_REDUCTION_TYPES


class TestContrastiveType:
    """Tests for ContrastiveType enum."""

    def test_simclr_value(self) -> None:
        """SimCLR has correct value."""
        assert ContrastiveType.SIMCLR.value == "simclr"

    def test_infonce_value(self) -> None:
        """InfoNCE has correct value."""
        assert ContrastiveType.INFONCE.value == "infonce"

    def test_triplet_value(self) -> None:
        """Triplet has correct value."""
        assert ContrastiveType.TRIPLET.value == "triplet"

    def test_ntxent_value(self) -> None:
        """NT-Xent has correct value."""
        assert ContrastiveType.NTXENT.value == "ntxent"

    def test_valid_contrastive_types_frozenset(self) -> None:
        """VALID_CONTRASTIVE_TYPES is a frozenset."""
        assert isinstance(VALID_CONTRASTIVE_TYPES, frozenset)
        assert len(VALID_CONTRASTIVE_TYPES) == 4

    def test_all_enum_values_in_valid_set(self) -> None:
        """All enum values are in VALID_CONTRASTIVE_TYPES."""
        for contrastive_type in ContrastiveType:
            assert contrastive_type.value in VALID_CONTRASTIVE_TYPES


class TestFocalLossConfig:
    """Tests for FocalLossConfig dataclass."""

    def test_create_config(self) -> None:
        """Create focal loss config."""
        config = FocalLossConfig(
            alpha=0.25,
            gamma=2.0,
            reduction=ReductionType.MEAN,
        )
        assert config.alpha == pytest.approx(0.25)
        assert config.gamma == pytest.approx(2.0)
        assert config.reduction == ReductionType.MEAN

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        with pytest.raises(AttributeError):
            config.gamma = 3.0  # type: ignore[misc]


class TestContrastiveLossConfig:
    """Tests for ContrastiveLossConfig dataclass."""

    def test_create_config(self) -> None:
        """Create contrastive loss config."""
        config = ContrastiveLossConfig(
            temperature=0.07,
            margin=1.0,
            contrastive_type=ContrastiveType.SIMCLR,
        )
        assert config.temperature == pytest.approx(0.07)
        assert config.margin == pytest.approx(1.0)
        assert config.contrastive_type == ContrastiveType.SIMCLR

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ContrastiveLossConfig(0.07, 1.0, ContrastiveType.SIMCLR)
        with pytest.raises(AttributeError):
            config.temperature = 0.1  # type: ignore[misc]


class TestLabelSmoothingConfig:
    """Tests for LabelSmoothingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create label smoothing config."""
        config = LabelSmoothingConfig(
            smoothing=0.1,
            reduction=ReductionType.MEAN,
        )
        assert config.smoothing == pytest.approx(0.1)
        assert config.reduction == ReductionType.MEAN

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = LabelSmoothingConfig(0.1, ReductionType.MEAN)
        with pytest.raises(AttributeError):
            config.smoothing = 0.2  # type: ignore[misc]


class TestLossConfig:
    """Tests for LossConfig dataclass."""

    def test_create_config(self) -> None:
        """Create loss config."""
        focal = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        config = LossConfig(
            loss_type=LossType.FOCAL,
            focal_config=focal,
            contrastive_config=None,
            label_smoothing_config=None,
            weight=None,
        )
        assert config.loss_type == LossType.FOCAL
        assert config.focal_config is not None
        assert config.contrastive_config is None

    def test_config_with_weights(self) -> None:
        """Create loss config with weights."""
        config = LossConfig(
            loss_type=LossType.CROSS_ENTROPY,
            focal_config=None,
            contrastive_config=None,
            label_smoothing_config=None,
            weight=(1.0, 2.0, 1.5),
        )
        assert config.weight == (1.0, 2.0, 1.5)


class TestLossStats:
    """Tests for LossStats dataclass."""

    def test_create_stats(self) -> None:
        """Create loss stats."""
        stats = LossStats(
            loss_value=0.5,
            num_samples=32,
            per_class_loss=(0.3, 0.5, 0.7),
        )
        assert stats.loss_value == pytest.approx(0.5)
        assert stats.num_samples == 32
        assert stats.per_class_loss == (0.3, 0.5, 0.7)

    def test_stats_without_per_class(self) -> None:
        """Create stats without per-class loss."""
        stats = LossStats(
            loss_value=0.25,
            num_samples=64,
            per_class_loss=None,
        )
        assert stats.per_class_loss is None


class TestValidateFocalLossConfig:
    """Tests for validate_focal_loss_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        validate_focal_loss_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_focal_loss_config(None)  # type: ignore[arg-type]

    def test_negative_alpha_raises(self) -> None:
        """Negative alpha raises ValueError."""
        config = FocalLossConfig(-0.1, 2.0, ReductionType.MEAN)
        with pytest.raises(ValueError, match=r"alpha must be between 0\.0 and 1\.0"):
            validate_focal_loss_config(config)

    def test_alpha_above_one_raises(self) -> None:
        """Alpha above 1 raises ValueError."""
        config = FocalLossConfig(1.5, 2.0, ReductionType.MEAN)
        with pytest.raises(ValueError, match=r"alpha must be between 0\.0 and 1\.0"):
            validate_focal_loss_config(config)

    def test_negative_gamma_raises(self) -> None:
        """Negative gamma raises ValueError."""
        config = FocalLossConfig(0.25, -1.0, ReductionType.MEAN)
        with pytest.raises(ValueError, match="gamma cannot be negative"):
            validate_focal_loss_config(config)

    def test_zero_alpha_valid(self) -> None:
        """Zero alpha is valid."""
        config = FocalLossConfig(0.0, 2.0, ReductionType.MEAN)
        validate_focal_loss_config(config)

    def test_zero_gamma_valid(self) -> None:
        """Zero gamma is valid (becomes cross-entropy)."""
        config = FocalLossConfig(0.25, 0.0, ReductionType.MEAN)
        validate_focal_loss_config(config)


class TestValidateContrastiveLossConfig:
    """Tests for validate_contrastive_loss_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ContrastiveLossConfig(0.07, 1.0, ContrastiveType.SIMCLR)
        validate_contrastive_loss_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_contrastive_loss_config(None)  # type: ignore[arg-type]

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        config = ContrastiveLossConfig(0.0, 1.0, ContrastiveType.SIMCLR)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_contrastive_loss_config(config)

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        config = ContrastiveLossConfig(-0.1, 1.0, ContrastiveType.SIMCLR)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_contrastive_loss_config(config)

    def test_negative_margin_raises(self) -> None:
        """Negative margin raises ValueError."""
        config = ContrastiveLossConfig(0.07, -1.0, ContrastiveType.TRIPLET)
        with pytest.raises(ValueError, match="margin cannot be negative"):
            validate_contrastive_loss_config(config)

    def test_zero_margin_valid(self) -> None:
        """Zero margin is valid."""
        config = ContrastiveLossConfig(0.07, 0.0, ContrastiveType.SIMCLR)
        validate_contrastive_loss_config(config)


class TestValidateLabelSmoothingConfig:
    """Tests for validate_label_smoothing_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LabelSmoothingConfig(0.1, ReductionType.MEAN)
        validate_label_smoothing_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_label_smoothing_config(None)  # type: ignore[arg-type]

    def test_negative_smoothing_raises(self) -> None:
        """Negative smoothing raises ValueError."""
        config = LabelSmoothingConfig(-0.1, ReductionType.MEAN)
        with pytest.raises(ValueError, match=r"smoothing must be between"):
            validate_label_smoothing_config(config)

    def test_smoothing_above_one_raises(self) -> None:
        """Smoothing above 1 raises ValueError."""
        config = LabelSmoothingConfig(1.5, ReductionType.MEAN)
        with pytest.raises(ValueError, match=r"smoothing must be between"):
            validate_label_smoothing_config(config)

    def test_zero_smoothing_valid(self) -> None:
        """Zero smoothing is valid (no smoothing)."""
        config = LabelSmoothingConfig(0.0, ReductionType.MEAN)
        validate_label_smoothing_config(config)


class TestValidateLossConfig:
    """Tests for validate_loss_config function."""

    def test_valid_focal_config(self) -> None:
        """Valid focal config passes validation."""
        focal = FocalLossConfig(0.25, 2.0, ReductionType.MEAN)
        config = LossConfig(LossType.FOCAL, focal, None, None, None)
        validate_loss_config(config)

    def test_valid_contrastive_config(self) -> None:
        """Valid contrastive config passes validation."""
        contrastive = ContrastiveLossConfig(0.07, 1.0, ContrastiveType.SIMCLR)
        config = LossConfig(LossType.CONTRASTIVE, None, contrastive, None, None)
        validate_loss_config(config)

    def test_valid_triplet_config(self) -> None:
        """Valid triplet config passes validation."""
        contrastive = ContrastiveLossConfig(0.07, 1.0, ContrastiveType.TRIPLET)
        config = LossConfig(LossType.TRIPLET, None, contrastive, None, None)
        validate_loss_config(config)

    def test_valid_cross_entropy_config(self) -> None:
        """Valid cross entropy config passes validation."""
        config = LossConfig(LossType.CROSS_ENTROPY, None, None, None, None)
        validate_loss_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_loss_config(None)  # type: ignore[arg-type]

    def test_missing_focal_config_raises(self) -> None:
        """Missing focal config raises ValueError."""
        config = LossConfig(LossType.FOCAL, None, None, None, None)
        with pytest.raises(ValueError, match="focal_config required"):
            validate_loss_config(config)

    def test_missing_contrastive_config_raises(self) -> None:
        """Missing contrastive config raises ValueError."""
        config = LossConfig(LossType.CONTRASTIVE, None, None, None, None)
        with pytest.raises(ValueError, match="contrastive_config required"):
            validate_loss_config(config)

    def test_missing_triplet_config_raises(self) -> None:
        """Missing triplet config raises ValueError."""
        config = LossConfig(LossType.TRIPLET, None, None, None, None)
        with pytest.raises(ValueError, match="contrastive_config required"):
            validate_loss_config(config)

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        config = LossConfig(LossType.CROSS_ENTROPY, None, None, None, (1.0, -0.5))
        with pytest.raises(ValueError, match="weight cannot contain negative"):
            validate_loss_config(config)


class TestValidateLossStats:
    """Tests for validate_loss_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = LossStats(0.5, 32, None)
        validate_loss_stats(stats)

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_loss_stats(None)  # type: ignore[arg-type]

    def test_zero_samples_raises(self) -> None:
        """Zero samples raises ValueError."""
        stats = LossStats(0.5, 0, None)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            validate_loss_stats(stats)

    def test_negative_samples_raises(self) -> None:
        """Negative samples raises ValueError."""
        stats = LossStats(0.5, -1, None)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            validate_loss_stats(stats)


class TestCreateFocalLossConfig:
    """Tests for create_focal_loss_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_focal_loss_config()
        assert config.alpha == pytest.approx(0.25)
        assert config.gamma == pytest.approx(2.0)
        assert config.reduction == ReductionType.MEAN

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_focal_loss_config(
            alpha=0.5,
            gamma=1.0,
            reduction="sum",
        )
        assert config.alpha == pytest.approx(0.5)
        assert config.gamma == pytest.approx(1.0)
        assert config.reduction == ReductionType.SUM

    def test_string_reduction(self) -> None:
        """Accept string reduction type."""
        config = create_focal_loss_config(reduction="none")
        assert config.reduction == ReductionType.NONE

    def test_negative_alpha_raises(self) -> None:
        """Negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between"):
            create_focal_loss_config(alpha=-0.1)


class TestCreateContrastiveLossConfig:
    """Tests for create_contrastive_loss_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_contrastive_loss_config()
        assert config.temperature == pytest.approx(0.07)
        assert config.margin == pytest.approx(1.0)
        assert config.contrastive_type == ContrastiveType.SIMCLR

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_contrastive_loss_config(
            temperature=0.1,
            margin=0.5,
            contrastive_type="triplet",
        )
        assert config.temperature == pytest.approx(0.1)
        assert config.margin == pytest.approx(0.5)
        assert config.contrastive_type == ContrastiveType.TRIPLET

    def test_string_contrastive_type(self) -> None:
        """Accept string contrastive type."""
        config = create_contrastive_loss_config(contrastive_type="ntxent")
        assert config.contrastive_type == ContrastiveType.NTXENT

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            create_contrastive_loss_config(temperature=0.0)


class TestCreateLabelSmoothingConfig:
    """Tests for create_label_smoothing_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_label_smoothing_config()
        assert config.smoothing == pytest.approx(0.1)
        assert config.reduction == ReductionType.MEAN

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_label_smoothing_config(
            smoothing=0.2,
            reduction="sum",
        )
        assert config.smoothing == pytest.approx(0.2)
        assert config.reduction == ReductionType.SUM

    def test_string_reduction(self) -> None:
        """Accept string reduction type."""
        config = create_label_smoothing_config(reduction="batchmean")
        assert config.reduction == ReductionType.BATCHMEAN

    def test_invalid_smoothing_raises(self) -> None:
        """Invalid smoothing raises ValueError."""
        with pytest.raises(ValueError, match="smoothing must be between"):
            create_label_smoothing_config(smoothing=1.5)


class TestCreateLossConfig:
    """Tests for create_loss_config function."""

    def test_default_config(self) -> None:
        """Create default cross entropy config."""
        config = create_loss_config()
        assert config.loss_type == LossType.CROSS_ENTROPY

    def test_focal_config_auto_created(self) -> None:
        """Focal config auto-created for focal loss."""
        config = create_loss_config("focal")
        assert config.loss_type == LossType.FOCAL
        assert config.focal_config is not None

    def test_focal_with_custom_config(self) -> None:
        """Create focal with custom config."""
        focal = create_focal_loss_config(gamma=3.0)
        config = create_loss_config("focal", focal_config=focal)
        assert config.focal_config.gamma == pytest.approx(3.0)

    def test_contrastive_config_auto_created(self) -> None:
        """Contrastive config auto-created for contrastive loss."""
        config = create_loss_config("contrastive")
        assert config.loss_type == LossType.CONTRASTIVE
        assert config.contrastive_config is not None

    def test_triplet_config_auto_created(self) -> None:
        """Triplet config auto-created for triplet loss."""
        config = create_loss_config("triplet")
        assert config.loss_type == LossType.TRIPLET
        assert config.contrastive_config is not None
        assert config.contrastive_config.contrastive_type == ContrastiveType.TRIPLET

    def test_mse_config(self) -> None:
        """Create MSE config."""
        config = create_loss_config("mse")
        assert config.loss_type == LossType.MSE

    def test_mae_config(self) -> None:
        """Create MAE config."""
        config = create_loss_config("mae")
        assert config.loss_type == LossType.MAE

    def test_with_weights(self) -> None:
        """Create config with class weights."""
        config = create_loss_config("cross_entropy", weight=(1.0, 2.0, 1.5))
        assert config.weight == (1.0, 2.0, 1.5)

    def test_invalid_loss_type_raises(self) -> None:
        """Invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="loss_type must be one of"):
            create_loss_config("invalid")


class TestCreateLossStats:
    """Tests for create_loss_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_loss_stats()
        assert stats.loss_value == pytest.approx(0.0)
        assert stats.num_samples == 1
        assert stats.per_class_loss is None

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_loss_stats(
            loss_value=0.5,
            num_samples=32,
            per_class_loss=(0.3, 0.5, 0.7),
        )
        assert stats.loss_value == pytest.approx(0.5)
        assert stats.num_samples == 32
        assert stats.per_class_loss == (0.3, 0.5, 0.7)

    def test_zero_samples_raises(self) -> None:
        """Zero samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            create_loss_stats(num_samples=0)


class TestListLossTypes:
    """Tests for list_loss_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_loss_types()
        assert types == sorted(types)

    def test_contains_cross_entropy(self) -> None:
        """Contains cross_entropy."""
        types = list_loss_types()
        assert "cross_entropy" in types

    def test_contains_focal(self) -> None:
        """Contains focal."""
        types = list_loss_types()
        assert "focal" in types

    def test_correct_count(self) -> None:
        """Returns correct count."""
        types = list_loss_types()
        assert len(types) == 8


class TestListReductionTypes:
    """Tests for list_reduction_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_reduction_types()
        assert types == sorted(types)

    def test_contains_mean(self) -> None:
        """Contains mean."""
        types = list_reduction_types()
        assert "mean" in types

    def test_contains_sum(self) -> None:
        """Contains sum."""
        types = list_reduction_types()
        assert "sum" in types

    def test_correct_count(self) -> None:
        """Returns correct count."""
        types = list_reduction_types()
        assert len(types) == 4


class TestListContrastiveTypes:
    """Tests for list_contrastive_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_contrastive_types()
        assert types == sorted(types)

    def test_contains_simclr(self) -> None:
        """Contains simclr."""
        types = list_contrastive_types()
        assert "simclr" in types

    def test_contains_ntxent(self) -> None:
        """Contains ntxent."""
        types = list_contrastive_types()
        assert "ntxent" in types

    def test_correct_count(self) -> None:
        """Returns correct count."""
        types = list_contrastive_types()
        assert len(types) == 4


class TestGetLossType:
    """Tests for get_loss_type function."""

    def test_get_cross_entropy(self) -> None:
        """Get cross_entropy."""
        assert get_loss_type("cross_entropy") == LossType.CROSS_ENTROPY

    def test_get_focal(self) -> None:
        """Get focal."""
        assert get_loss_type("focal") == LossType.FOCAL

    def test_get_contrastive(self) -> None:
        """Get contrastive."""
        assert get_loss_type("contrastive") == LossType.CONTRASTIVE

    def test_get_triplet(self) -> None:
        """Get triplet."""
        assert get_loss_type("triplet") == LossType.TRIPLET

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="loss_type must be one of"):
            get_loss_type("invalid")


class TestGetReductionType:
    """Tests for get_reduction_type function."""

    def test_get_mean(self) -> None:
        """Get mean."""
        assert get_reduction_type("mean") == ReductionType.MEAN

    def test_get_sum(self) -> None:
        """Get sum."""
        assert get_reduction_type("sum") == ReductionType.SUM

    def test_get_none(self) -> None:
        """Get none."""
        assert get_reduction_type("none") == ReductionType.NONE

    def test_get_batchmean(self) -> None:
        """Get batchmean."""
        assert get_reduction_type("batchmean") == ReductionType.BATCHMEAN

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="reduction_type must be one of"):
            get_reduction_type("invalid")


class TestGetContrastiveType:
    """Tests for get_contrastive_type function."""

    def test_get_simclr(self) -> None:
        """Get simclr."""
        assert get_contrastive_type("simclr") == ContrastiveType.SIMCLR

    def test_get_infonce(self) -> None:
        """Get infonce."""
        assert get_contrastive_type("infonce") == ContrastiveType.INFONCE

    def test_get_triplet(self) -> None:
        """Get triplet."""
        assert get_contrastive_type("triplet") == ContrastiveType.TRIPLET

    def test_get_ntxent(self) -> None:
        """Get ntxent."""
        assert get_contrastive_type("ntxent") == ContrastiveType.NTXENT

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="contrastive_type must be one of"):
            get_contrastive_type("invalid")


class TestCalculateFocalLoss:
    """Tests for calculate_focal_loss function."""

    def test_basic_calculation(self) -> None:
        """Basic focal loss calculation."""
        config = create_focal_loss_config(alpha=0.25, gamma=2.0)
        loss = calculate_focal_loss((0.9, 0.8, 0.7), (1, 1, 1), config)
        assert loss > 0
        assert loss < 1

    def test_well_classified_low_loss(self) -> None:
        """Well classified samples have low loss."""
        config = create_focal_loss_config(alpha=0.25, gamma=2.0)
        loss_high_conf = calculate_focal_loss((0.99,), (1,), config)
        loss_low_conf = calculate_focal_loss((0.5,), (1,), config)
        assert loss_high_conf < loss_low_conf

    def test_negative_class(self) -> None:
        """Loss for negative class."""
        config = create_focal_loss_config(alpha=0.25, gamma=2.0)
        loss = calculate_focal_loss((0.1, 0.2), (0, 0), config)
        assert loss > 0
        assert loss < 1

    def test_gamma_zero_is_cross_entropy(self) -> None:
        """Gamma=0 degenerates to weighted cross-entropy."""
        config_focal = create_focal_loss_config(alpha=0.5, gamma=0.0)
        loss = calculate_focal_loss((0.5,), (1,), config_focal)
        assert loss > 0

    def test_sum_reduction(self) -> None:
        """Sum reduction."""
        config = create_focal_loss_config(alpha=0.25, gamma=2.0, reduction="sum")
        loss = calculate_focal_loss((0.9, 0.8), (1, 1), config)
        assert loss > 0

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        config = create_focal_loss_config()
        with pytest.raises(ValueError, match="same length"):
            calculate_focal_loss((0.5,), (1, 0), config)

    def test_empty_predictions_raises(self) -> None:
        """Empty predictions raises ValueError."""
        config = create_focal_loss_config()
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_focal_loss((), (), config)

    def test_prediction_out_of_range_raises(self) -> None:
        """Prediction out of range raises ValueError."""
        config = create_focal_loss_config()
        with pytest.raises(ValueError, match="must be in"):
            calculate_focal_loss((1.5,), (1,), config)


class TestCalculateContrastiveLoss:
    """Tests for calculate_contrastive_loss function."""

    def test_simclr_loss(self) -> None:
        """SimCLR contrastive loss calculation."""
        config = create_contrastive_loss_config(temperature=0.5)
        anchor = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        positive = ((0.9, 0.1, 0.0), (0.1, 0.9, 0.0))
        loss = calculate_contrastive_loss(anchor, positive, None, config)
        assert loss >= 0

    def test_triplet_loss(self) -> None:
        """Triplet loss calculation."""
        config = create_contrastive_loss_config(contrastive_type="triplet", margin=1.0)
        anchor = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        positive = ((0.9, 0.1, 0.0), (0.1, 0.9, 0.0))
        negative = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        loss = calculate_contrastive_loss(anchor, positive, negative, config)
        assert loss >= 0

    def test_triplet_zero_loss_when_satisfied(self) -> None:
        """Triplet loss is zero when constraint satisfied."""
        config = create_contrastive_loss_config(contrastive_type="triplet", margin=0.1)
        # Positive is very close, negative is far
        anchor = ((1.0, 0.0), (1.0, 0.0))
        positive = ((1.0, 0.0), (1.0, 0.0))  # Same as anchor
        negative = ((0.0, 1.0), (0.0, 1.0))  # Far from anchor
        loss = calculate_contrastive_loss(anchor, positive, negative, config)
        assert loss == pytest.approx(0.0, abs=0.01)

    def test_empty_embeddings_raises(self) -> None:
        """Empty embeddings raises ValueError."""
        config = create_contrastive_loss_config()
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_contrastive_loss((), (), None, config)

    def test_mismatched_batch_size_raises(self) -> None:
        """Mismatched batch size raises ValueError."""
        config = create_contrastive_loss_config()
        anchor = ((1.0, 0.0),)
        positive = ((0.9, 0.1), (0.1, 0.9))
        with pytest.raises(ValueError, match="same batch size"):
            calculate_contrastive_loss(anchor, positive, None, config)

    def test_triplet_missing_negative_raises(self) -> None:
        """Triplet loss without negative raises ValueError."""
        config = create_contrastive_loss_config(contrastive_type="triplet")
        anchor = ((1.0, 0.0),)
        positive = ((0.9, 0.1),)
        with pytest.raises(ValueError, match="negative_embeddings required"):
            calculate_contrastive_loss(anchor, positive, None, config)

    def test_triplet_mismatched_negative_raises(self) -> None:
        """Triplet loss with mismatched negative raises ValueError."""
        config = create_contrastive_loss_config(contrastive_type="triplet")
        anchor = ((1.0, 0.0),)
        positive = ((0.9, 0.1),)
        negative = ((0.0, 1.0), (0.0, 1.0))  # Different batch size
        with pytest.raises(ValueError, match="same batch size"):
            calculate_contrastive_loss(anchor, positive, negative, config)


class TestApplyLabelSmoothing:
    """Tests for apply_label_smoothing function."""

    def test_basic_smoothing(self) -> None:
        """Basic label smoothing."""
        config = create_label_smoothing_config(smoothing=0.1)
        soft_labels = apply_label_smoothing((0, 1, 2), 3, config)
        assert len(soft_labels) == 3
        assert len(soft_labels[0]) == 3

    def test_target_has_higher_prob(self) -> None:
        """Target class has higher probability."""
        config = create_label_smoothing_config(smoothing=0.1)
        soft_labels = apply_label_smoothing((0,), 3, config)
        assert soft_labels[0][0] > soft_labels[0][1]
        assert soft_labels[0][0] > soft_labels[0][2]

    def test_sums_to_one(self) -> None:
        """Soft labels sum to 1."""
        config = create_label_smoothing_config(smoothing=0.1)
        soft_labels = apply_label_smoothing((0, 1), 3, config)
        for label in soft_labels:
            assert sum(label) == pytest.approx(1.0)

    def test_zero_smoothing(self) -> None:
        """Zero smoothing gives hard labels."""
        config = create_label_smoothing_config(smoothing=0.0)
        soft_labels = apply_label_smoothing((0,), 3, config)
        assert soft_labels[0][0] == pytest.approx(1.0)
        assert soft_labels[0][1] == pytest.approx(0.0)

    def test_high_smoothing(self) -> None:
        """High smoothing distributes probability more evenly."""
        config_low = create_label_smoothing_config(smoothing=0.1)
        config_high = create_label_smoothing_config(smoothing=0.5)

        labels_low = apply_label_smoothing((0,), 3, config_low)
        labels_high = apply_label_smoothing((0,), 3, config_high)

        # High smoothing has more uniform distribution
        assert labels_high[0][0] < labels_low[0][0]

    def test_empty_targets_raises(self) -> None:
        """Empty targets raises ValueError."""
        config = create_label_smoothing_config()
        with pytest.raises(ValueError, match="cannot be empty"):
            apply_label_smoothing((), 3, config)

    def test_zero_classes_raises(self) -> None:
        """Zero classes raises ValueError."""
        config = create_label_smoothing_config()
        with pytest.raises(ValueError, match="num_classes must be positive"):
            apply_label_smoothing((0,), 0, config)

    def test_target_out_of_range_raises(self) -> None:
        """Target out of range raises ValueError."""
        config = create_label_smoothing_config()
        with pytest.raises(ValueError, match="out of range"):
            apply_label_smoothing((5,), 3, config)


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    def test_inverse_weighting(self) -> None:
        """Inverse weighting gives higher weight to minority."""
        weights = compute_class_weights((100, 50, 25), "inverse")
        assert len(weights) == 3
        assert weights[2] > weights[0]  # Minority has higher weight

    def test_inverse_sqrt_weighting(self) -> None:
        """Inverse sqrt weighting."""
        weights = compute_class_weights((100, 50, 25), "inverse_sqrt")
        assert len(weights) == 3
        assert weights[2] > weights[0]

    def test_effective_weighting(self) -> None:
        """Effective number weighting."""
        weights = compute_class_weights((100, 50, 25), "effective")
        assert len(weights) == 3
        assert weights[2] > weights[0]

    def test_balanced_classes(self) -> None:
        """Equal counts give equal weights."""
        weights = compute_class_weights((100, 100, 100), "inverse")
        assert all(abs(w - weights[0]) < 1e-6 for w in weights)

    def test_weights_sum_to_num_classes(self) -> None:
        """Normalized weights sum to num_classes."""
        weights = compute_class_weights((100, 50, 25), "inverse")
        assert sum(weights) == pytest.approx(3.0)

    def test_empty_counts_raises(self) -> None:
        """Empty counts raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_class_weights((), "inverse")

    def test_zero_count_raises(self) -> None:
        """Zero count raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_class_weights((100, 0), "inverse")

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            compute_class_weights((100, 50), "invalid")


class TestFormatLossStats:
    """Tests for format_loss_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_loss_stats(loss_value=0.5, num_samples=32)
        formatted = format_loss_stats(stats)
        assert "Loss: 0.5000" in formatted
        assert "Samples: 32" in formatted

    def test_format_with_per_class(self) -> None:
        """Format with per-class loss."""
        stats = create_loss_stats(
            loss_value=0.5,
            num_samples=32,
            per_class_loss=(0.3, 0.5, 0.7),
        )
        formatted = format_loss_stats(stats)
        assert "Per-class:" in formatted
        assert "0.3000" in formatted
        assert "0.5000" in formatted
        assert "0.7000" in formatted

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_loss_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedLossConfig:
    """Tests for get_recommended_loss_config function."""

    def test_classification(self) -> None:
        """Get config for classification."""
        config = get_recommended_loss_config("classification")
        assert config.loss_type == LossType.CROSS_ENTROPY
        assert config.label_smoothing_config is not None

    def test_imbalanced_classification(self) -> None:
        """Get config for imbalanced classification."""
        config = get_recommended_loss_config("imbalanced_classification")
        assert config.loss_type == LossType.FOCAL
        assert config.focal_config is not None

    def test_contrastive_learning(self) -> None:
        """Get config for contrastive learning."""
        config = get_recommended_loss_config("contrastive_learning")
        assert config.loss_type == LossType.CONTRASTIVE
        assert config.contrastive_config is not None

    def test_regression(self) -> None:
        """Get config for regression."""
        config = get_recommended_loss_config("regression")
        assert config.loss_type == LossType.MSE

    def test_generation(self) -> None:
        """Get config for generation."""
        config = get_recommended_loss_config("generation")
        assert config.loss_type == LossType.CROSS_ENTROPY
        assert config.label_smoothing_config is not None

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_loss_config("invalid")
