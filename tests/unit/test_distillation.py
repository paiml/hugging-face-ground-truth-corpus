"""Tests for training.distillation module."""

from __future__ import annotations

import pytest

from hf_gtc.training.distillation import (
    VALID_DISTILLATION_LOSSES,
    VALID_DISTILLATION_METHODS,
    VALID_STUDENT_INITIALIZATIONS,
    VALID_TEMPERATURE_SCHEDULES,
    DistillationConfig,
    DistillationLoss,
    DistillationLossConfig,
    DistillationMethod,
    DistillationStats,
    FeatureMatchingConfig,
    StudentConfig,
    StudentInitialization,
    TeacherConfig,
    TemperatureSchedule,
    calculate_combined_loss,
    calculate_soft_labels_loss,
    calculate_temperature_at_step,
    create_distillation_config,
    create_distillation_stats,
    create_feature_matching_config,
    create_loss_config,
    create_student_config,
    create_teacher_config,
    estimate_compression_ratio,
    estimate_model_parameters,
    format_distillation_stats,
    get_distillation_loss,
    get_distillation_method,
    get_layer_mapping_strategy,
    get_recommended_distillation_config,
    get_student_initialization,
    get_temperature_schedule,
    list_distillation_losses,
    list_distillation_methods,
    list_student_initializations,
    list_temperature_schedules,
    validate_distillation_config,
    validate_feature_matching_config,
    validate_loss_config,
    validate_student_config,
    validate_teacher_config,
)


class TestDistillationMethod:
    """Tests for DistillationMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in DistillationMethod:
            assert isinstance(method.value, str)

    def test_response_based_value(self) -> None:
        """Response-based has correct value."""
        assert DistillationMethod.RESPONSE_BASED.value == "response_based"

    def test_feature_based_value(self) -> None:
        """Feature-based has correct value."""
        assert DistillationMethod.FEATURE_BASED.value == "feature_based"

    def test_attention_transfer_value(self) -> None:
        """Attention transfer has correct value."""
        assert DistillationMethod.ATTENTION_TRANSFER.value == "attention_transfer"

    def test_self_distillation_value(self) -> None:
        """Self distillation has correct value."""
        assert DistillationMethod.SELF_DISTILLATION.value == "self_distillation"

    def test_progressive_value(self) -> None:
        """Progressive has correct value."""
        assert DistillationMethod.PROGRESSIVE.value == "progressive"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_DISTILLATION_METHODS is a frozenset."""
        assert isinstance(VALID_DISTILLATION_METHODS, frozenset)
        assert len(VALID_DISTILLATION_METHODS) == 5


class TestDistillationLoss:
    """Tests for DistillationLoss enum."""

    def test_all_losses_have_values(self) -> None:
        """All losses have string values."""
        for loss in DistillationLoss:
            assert isinstance(loss.value, str)

    def test_kl_divergence_value(self) -> None:
        """KL divergence has correct value."""
        assert DistillationLoss.KL_DIVERGENCE.value == "kl_divergence"

    def test_mse_value(self) -> None:
        """MSE has correct value."""
        assert DistillationLoss.MSE.value == "mse"

    def test_cosine_value(self) -> None:
        """Cosine has correct value."""
        assert DistillationLoss.COSINE.value == "cosine"

    def test_cross_entropy_value(self) -> None:
        """Cross entropy has correct value."""
        assert DistillationLoss.CROSS_ENTROPY.value == "cross_entropy"

    def test_combined_value(self) -> None:
        """Combined has correct value."""
        assert DistillationLoss.COMBINED.value == "combined"

    def test_valid_losses_frozenset(self) -> None:
        """VALID_DISTILLATION_LOSSES is a frozenset."""
        assert isinstance(VALID_DISTILLATION_LOSSES, frozenset)
        assert len(VALID_DISTILLATION_LOSSES) == 5


class TestTemperatureSchedule:
    """Tests for TemperatureSchedule enum."""

    def test_all_schedules_have_values(self) -> None:
        """All schedules have string values."""
        for schedule in TemperatureSchedule:
            assert isinstance(schedule.value, str)

    def test_constant_value(self) -> None:
        """Constant has correct value."""
        assert TemperatureSchedule.CONSTANT.value == "constant"

    def test_linear_decay_value(self) -> None:
        """Linear decay has correct value."""
        assert TemperatureSchedule.LINEAR_DECAY.value == "linear_decay"

    def test_cosine_decay_value(self) -> None:
        """Cosine decay has correct value."""
        assert TemperatureSchedule.COSINE_DECAY.value == "cosine_decay"

    def test_exponential_decay_value(self) -> None:
        """Exponential decay has correct value."""
        assert TemperatureSchedule.EXPONENTIAL_DECAY.value == "exponential_decay"

    def test_warmup_value(self) -> None:
        """Warmup has correct value."""
        assert TemperatureSchedule.WARMUP.value == "warmup"

    def test_valid_schedules_frozenset(self) -> None:
        """VALID_TEMPERATURE_SCHEDULES is a frozenset."""
        assert isinstance(VALID_TEMPERATURE_SCHEDULES, frozenset)
        assert len(VALID_TEMPERATURE_SCHEDULES) == 5


class TestStudentInitialization:
    """Tests for StudentInitialization enum."""

    def test_all_inits_have_values(self) -> None:
        """All initializations have string values."""
        for init in StudentInitialization:
            assert isinstance(init.value, str)

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert StudentInitialization.RANDOM.value == "random"

    def test_teacher_subset_value(self) -> None:
        """Teacher subset has correct value."""
        assert StudentInitialization.TEACHER_SUBSET.value == "teacher_subset"

    def test_pretrained_value(self) -> None:
        """Pretrained has correct value."""
        assert StudentInitialization.PRETRAINED.value == "pretrained"

    def test_pruned_value(self) -> None:
        """Pruned has correct value."""
        assert StudentInitialization.PRUNED.value == "pruned"

    def test_valid_inits_frozenset(self) -> None:
        """VALID_STUDENT_INITIALIZATIONS is a frozenset."""
        assert isinstance(VALID_STUDENT_INITIALIZATIONS, frozenset)
        assert len(VALID_STUDENT_INITIALIZATIONS) == 4


class TestTeacherConfig:
    """Tests for TeacherConfig dataclass."""

    def test_create_config(self) -> None:
        """Create teacher config."""
        config = TeacherConfig(
            model_name_or_path="bert-large-uncased",
            num_layers=24,
            hidden_size=1024,
            output_hidden_states=True,
            output_attentions=False,
        )
        assert config.model_name_or_path == "bert-large-uncased"
        assert config.num_layers == 24
        assert config.hidden_size == 1024

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = TeacherConfig("bert", 12, 768, True, False)
        with pytest.raises(AttributeError):
            config.num_layers = 6  # type: ignore[misc]


class TestStudentConfig:
    """Tests for StudentConfig dataclass."""

    def test_create_config(self) -> None:
        """Create student config."""
        config = StudentConfig(
            model_name_or_path="distilbert-base-uncased",
            num_layers=6,
            hidden_size=768,
            initialization=StudentInitialization.PRETRAINED,
            layer_mapping=(0, 2, 4, 6, 8, 10),
        )
        assert config.model_name_or_path == "distilbert-base-uncased"
        assert config.num_layers == 6
        assert len(config.layer_mapping) == 6

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = StudentConfig(
            "distilbert", 6, 768, StudentInitialization.RANDOM, (0, 1, 2, 3, 4, 5)
        )
        with pytest.raises(AttributeError):
            config.hidden_size = 512  # type: ignore[misc]


class TestDistillationLossConfig:
    """Tests for DistillationLossConfig dataclass."""

    def test_create_config(self) -> None:
        """Create loss config."""
        config = DistillationLossConfig(
            loss_type=DistillationLoss.KL_DIVERGENCE,
            temperature=4.0,
            alpha=0.7,
            beta=0.0,
            normalize_features=True,
        )
        assert config.loss_type == DistillationLoss.KL_DIVERGENCE
        assert config.temperature == pytest.approx(4.0)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = DistillationLossConfig(DistillationLoss.MSE, 4.0, 0.5, 0.0, True)
        with pytest.raises(AttributeError):
            config.alpha = 0.8  # type: ignore[misc]


class TestFeatureMatchingConfig:
    """Tests for FeatureMatchingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create feature matching config."""
        config = FeatureMatchingConfig(
            match_hidden_states=True,
            match_attention=False,
            hidden_layer_indices=(4, 8, 12),
            attention_layer_indices=(),
            projection_dim=768,
        )
        assert config.match_hidden_states is True
        assert len(config.hidden_layer_indices) == 3

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = FeatureMatchingConfig(True, False, (4,), (), 768)
        with pytest.raises(AttributeError):
            config.projection_dim = 512  # type: ignore[misc]


class TestDistillationConfig:
    """Tests for DistillationConfig dataclass."""

    def test_create_config(self) -> None:
        """Create distillation config."""
        config = DistillationConfig(
            method=DistillationMethod.RESPONSE_BASED,
            temperature=4.0,
            alpha=0.7,
            temperature_schedule=TemperatureSchedule.CONSTANT,
            final_temperature=1.0,
            warmup_steps=100,
        )
        assert config.method == DistillationMethod.RESPONSE_BASED
        assert config.temperature == pytest.approx(4.0)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            4.0,
            0.5,
            TemperatureSchedule.CONSTANT,
            1.0,
            0,
        )
        with pytest.raises(AttributeError):
            config.temperature = 2.0  # type: ignore[misc]


class TestDistillationStats:
    """Tests for DistillationStats dataclass."""

    def test_create_stats(self) -> None:
        """Create distillation stats."""
        stats = DistillationStats(
            total_steps=1000,
            distillation_loss=0.5,
            task_loss=0.3,
            combined_loss=0.44,
            temperature=4.0,
            compression_ratio=0.25,
        )
        assert stats.total_steps == 1000
        assert stats.compression_ratio == pytest.approx(0.25)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = DistillationStats(1000, 0.5, 0.3, 0.44, 4.0, 0.25)
        with pytest.raises(AttributeError):
            stats.total_steps = 2000  # type: ignore[misc]


class TestValidateDistillationConfig:
    """Tests for validate_distillation_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            4.0,
            0.5,
            TemperatureSchedule.CONSTANT,
            1.0,
            0,
        )
        validate_distillation_config(config)

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            0.0,
            0.5,
            TemperatureSchedule.CONSTANT,
            1.0,
            0,
        )
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_distillation_config(config)

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            -1.0,
            0.5,
            TemperatureSchedule.CONSTANT,
            1.0,
            0,
        )
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_distillation_config(config)

    def test_alpha_out_of_range_raises(self) -> None:
        """Alpha > 1 raises ValueError."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            4.0,
            1.5,
            TemperatureSchedule.CONSTANT,
            1.0,
            0,
        )
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            validate_distillation_config(config)

    def test_negative_alpha_raises(self) -> None:
        """Negative alpha raises ValueError."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            4.0,
            -0.1,
            TemperatureSchedule.CONSTANT,
            1.0,
            0,
        )
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            validate_distillation_config(config)

    def test_zero_final_temperature_raises(self) -> None:
        """Zero final_temperature raises ValueError."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            4.0,
            0.5,
            TemperatureSchedule.LINEAR_DECAY,
            0.0,
            0,
        )
        with pytest.raises(ValueError, match="final_temperature must be positive"):
            validate_distillation_config(config)

    def test_negative_warmup_raises(self) -> None:
        """Negative warmup_steps raises ValueError."""
        config = DistillationConfig(
            DistillationMethod.RESPONSE_BASED,
            4.0,
            0.5,
            TemperatureSchedule.CONSTANT,
            1.0,
            -1,
        )
        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            validate_distillation_config(config)


class TestValidateTeacherConfig:
    """Tests for validate_teacher_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = TeacherConfig("bert-base-uncased", 12, 768, True, False)
        validate_teacher_config(config)

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name_or_path raises ValueError."""
        config = TeacherConfig("", 12, 768, True, False)
        with pytest.raises(ValueError, match="model_name_or_path cannot be empty"):
            validate_teacher_config(config)

    def test_whitespace_model_name_raises(self) -> None:
        """Whitespace model_name_or_path raises ValueError."""
        config = TeacherConfig("   ", 12, 768, True, False)
        with pytest.raises(ValueError, match="model_name_or_path cannot be empty"):
            validate_teacher_config(config)

    def test_zero_layers_raises(self) -> None:
        """Zero num_layers raises ValueError."""
        config = TeacherConfig("bert", 0, 768, True, False)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_teacher_config(config)

    def test_zero_hidden_size_raises(self) -> None:
        """Zero hidden_size raises ValueError."""
        config = TeacherConfig("bert", 12, 0, True, False)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_teacher_config(config)


class TestValidateStudentConfig:
    """Tests for validate_student_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = StudentConfig(
            "distilbert", 6, 768, StudentInitialization.PRETRAINED, (0, 2, 4, 6, 8, 10)
        )
        validate_student_config(config)

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name_or_path raises ValueError."""
        config = StudentConfig(
            "", 6, 768, StudentInitialization.RANDOM, (0, 1, 2, 3, 4, 5)
        )
        with pytest.raises(ValueError, match="model_name_or_path cannot be empty"):
            validate_student_config(config)

    def test_layer_mapping_mismatch_raises(self) -> None:
        """Layer mapping length mismatch raises ValueError."""
        config = StudentConfig(
            "distilbert", 6, 768, StudentInitialization.PRETRAINED, (0, 4, 8)
        )
        with pytest.raises(ValueError, match="layer_mapping length"):
            validate_student_config(config)

    def test_zero_layers_raises(self) -> None:
        """Zero num_layers raises ValueError."""
        config = StudentConfig(
            "distilbert", 0, 768, StudentInitialization.PRETRAINED, ()
        )
        with pytest.raises(ValueError, match="num_layers must be positive"):
            validate_student_config(config)

    def test_zero_hidden_size_raises(self) -> None:
        """Zero hidden_size raises ValueError."""
        config = StudentConfig(
            "distilbert", 6, 0, StudentInitialization.PRETRAINED, (0, 1, 2, 3, 4, 5)
        )
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_student_config(config)


class TestValidateLossConfig:
    """Tests for validate_loss_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DistillationLossConfig(
            DistillationLoss.KL_DIVERGENCE, 4.0, 0.5, 0.0, True
        )
        validate_loss_config(config)

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        config = DistillationLossConfig(
            DistillationLoss.KL_DIVERGENCE, 0.0, 0.5, 0.0, True
        )
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_loss_config(config)

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        config = DistillationLossConfig(
            DistillationLoss.KL_DIVERGENCE, -1.0, 0.5, 0.0, True
        )
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_loss_config(config)

    def test_alpha_out_of_range_raises(self) -> None:
        """Alpha > 1 raises ValueError."""
        config = DistillationLossConfig(
            DistillationLoss.KL_DIVERGENCE, 4.0, 1.5, 0.0, True
        )
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            validate_loss_config(config)

    def test_negative_beta_raises(self) -> None:
        """Negative beta raises ValueError."""
        config = DistillationLossConfig(
            DistillationLoss.KL_DIVERGENCE, 4.0, 0.5, -0.1, True
        )
        with pytest.raises(ValueError, match="beta must be non-negative"):
            validate_loss_config(config)


class TestValidateFeatureMatchingConfig:
    """Tests for validate_feature_matching_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = FeatureMatchingConfig(True, False, (4, 8, 12), (), 768)
        validate_feature_matching_config(config)

    def test_zero_projection_dim_raises(self) -> None:
        """Zero projection_dim raises ValueError."""
        config = FeatureMatchingConfig(True, False, (4,), (), 0)
        with pytest.raises(ValueError, match="projection_dim must be positive"):
            validate_feature_matching_config(config)

    def test_hidden_states_without_indices_raises(self) -> None:
        """Match hidden states without indices raises ValueError."""
        config = FeatureMatchingConfig(True, False, (), (), 768)
        with pytest.raises(ValueError, match="hidden_layer_indices required"):
            validate_feature_matching_config(config)

    def test_attention_without_indices_raises(self) -> None:
        """Match attention without indices raises ValueError."""
        config = FeatureMatchingConfig(False, True, (), (), 768)
        with pytest.raises(ValueError, match="attention_layer_indices required"):
            validate_feature_matching_config(config)


class TestCreateDistillationConfig:
    """Tests for create_distillation_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_distillation_config()
        assert config.method == DistillationMethod.RESPONSE_BASED
        assert config.temperature == pytest.approx(4.0)
        assert config.alpha == pytest.approx(0.5)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_distillation_config(
            method="feature_based",
            temperature=2.0,
            alpha=0.8,
        )
        assert config.method == DistillationMethod.FEATURE_BASED
        assert config.temperature == pytest.approx(2.0)
        assert config.alpha == pytest.approx(0.8)

    def test_with_enum_method(self) -> None:
        """Create with enum method."""
        config = create_distillation_config(method=DistillationMethod.PROGRESSIVE)
        assert config.method == DistillationMethod.PROGRESSIVE

    def test_with_string_schedule(self) -> None:
        """Create with string schedule."""
        config = create_distillation_config(temperature_schedule="linear_decay")
        assert config.temperature_schedule == TemperatureSchedule.LINEAR_DECAY

    @pytest.mark.parametrize(
        "method",
        [
            "response_based",
            "feature_based",
            "attention_transfer",
            "self_distillation",
            "progressive",
        ],
    )
    def test_all_methods(self, method: str) -> None:
        """Test all distillation methods."""
        config = create_distillation_config(method=method)
        assert config.method.value == method

    def test_invalid_temperature_raises(self) -> None:
        """Invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            create_distillation_config(temperature=0)


class TestCreateTeacherConfig:
    """Tests for create_teacher_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_teacher_config("bert-base-uncased")
        assert config.num_layers == 12
        assert config.hidden_size == 768
        assert config.output_hidden_states is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_teacher_config(
            "bert-large-uncased",
            num_layers=24,
            hidden_size=1024,
            output_attentions=True,
        )
        assert config.num_layers == 24
        assert config.hidden_size == 1024
        assert config.output_attentions is True

    def test_empty_model_name_raises(self) -> None:
        """Empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name_or_path cannot be empty"):
            create_teacher_config("")


class TestCreateStudentConfig:
    """Tests for create_student_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_student_config("distilbert-base-uncased")
        assert config.num_layers == 6
        assert config.hidden_size == 768
        assert len(config.layer_mapping) == 6

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_student_config(
            "my-student",
            num_layers=4,
            hidden_size=512,
            initialization="random",
            layer_mapping=(0, 4, 8, 11),
        )
        assert config.num_layers == 4
        assert config.initialization == StudentInitialization.RANDOM
        assert config.layer_mapping == (0, 4, 8, 11)

    def test_auto_layer_mapping(self) -> None:
        """Auto-generates layer mapping when None."""
        config = create_student_config("student", num_layers=4)
        assert config.layer_mapping == (0, 1, 2, 3)

    def test_empty_model_name_raises(self) -> None:
        """Empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name_or_path cannot be empty"):
            create_student_config("")


class TestCreateLossConfig:
    """Tests for create_loss_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_loss_config()
        assert config.loss_type == DistillationLoss.KL_DIVERGENCE
        assert config.temperature == pytest.approx(4.0)
        assert config.alpha == pytest.approx(0.5)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_loss_config(
            loss_type="mse",
            temperature=2.0,
            beta=0.5,
        )
        assert config.loss_type == DistillationLoss.MSE
        assert config.beta == pytest.approx(0.5)

    def test_with_enum_loss_type(self) -> None:
        """Create with enum loss type."""
        config = create_loss_config(loss_type=DistillationLoss.COSINE)
        assert config.loss_type == DistillationLoss.COSINE

    def test_invalid_alpha_raises(self) -> None:
        """Invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            create_loss_config(alpha=1.5)


class TestCreateFeatureMatchingConfig:
    """Tests for create_feature_matching_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_feature_matching_config()
        assert config.match_hidden_states is True
        assert config.hidden_layer_indices == (4, 8, 12)
        assert config.projection_dim == 768

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_feature_matching_config(
            match_attention=True,
            attention_layer_indices=(6, 12),
            projection_dim=512,
        )
        assert config.match_attention is True
        assert config.attention_layer_indices == (6, 12)

    def test_invalid_projection_dim_raises(self) -> None:
        """Invalid projection_dim raises ValueError."""
        with pytest.raises(ValueError, match="projection_dim must be positive"):
            create_feature_matching_config(projection_dim=0)


class TestCreateDistillationStats:
    """Tests for create_distillation_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_distillation_stats()
        assert stats.total_steps == 0
        assert stats.compression_ratio == pytest.approx(1.0)

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_distillation_stats(
            total_steps=1000,
            distillation_loss=0.5,
            compression_ratio=0.25,
        )
        assert stats.total_steps == 1000
        assert stats.compression_ratio == pytest.approx(0.25)

    def test_negative_steps_raises(self) -> None:
        """Negative total_steps raises ValueError."""
        with pytest.raises(ValueError, match="total_steps must be non-negative"):
            create_distillation_stats(total_steps=-1)

    def test_zero_compression_ratio_raises(self) -> None:
        """Zero compression_ratio raises ValueError."""
        with pytest.raises(ValueError, match="compression_ratio must be positive"):
            create_distillation_stats(compression_ratio=0)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_distillation_methods_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_distillation_methods()
        assert methods == sorted(methods)
        assert "response_based" in methods

    def test_list_distillation_losses_sorted(self) -> None:
        """Returns sorted list."""
        losses = list_distillation_losses()
        assert losses == sorted(losses)
        assert "kl_divergence" in losses

    def test_list_temperature_schedules_sorted(self) -> None:
        """Returns sorted list."""
        schedules = list_temperature_schedules()
        assert schedules == sorted(schedules)
        assert "constant" in schedules

    def test_list_student_initializations_sorted(self) -> None:
        """Returns sorted list."""
        inits = list_student_initializations()
        assert inits == sorted(inits)
        assert "pretrained" in inits


class TestGetDistillationMethod:
    """Tests for get_distillation_method function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("response_based", DistillationMethod.RESPONSE_BASED),
            ("feature_based", DistillationMethod.FEATURE_BASED),
            ("attention_transfer", DistillationMethod.ATTENTION_TRANSFER),
            ("self_distillation", DistillationMethod.SELF_DISTILLATION),
            ("progressive", DistillationMethod.PROGRESSIVE),
        ],
    )
    def test_all_methods(self, name: str, expected: DistillationMethod) -> None:
        """Test all valid methods."""
        assert get_distillation_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_distillation_method("invalid")


class TestGetDistillationLoss:
    """Tests for get_distillation_loss function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("kl_divergence", DistillationLoss.KL_DIVERGENCE),
            ("mse", DistillationLoss.MSE),
            ("cosine", DistillationLoss.COSINE),
            ("cross_entropy", DistillationLoss.CROSS_ENTROPY),
            ("combined", DistillationLoss.COMBINED),
        ],
    )
    def test_all_losses(self, name: str, expected: DistillationLoss) -> None:
        """Test all valid losses."""
        assert get_distillation_loss(name) == expected

    def test_invalid_loss_raises(self) -> None:
        """Invalid loss raises ValueError."""
        with pytest.raises(ValueError, match="loss_type must be one of"):
            get_distillation_loss("invalid")


class TestGetTemperatureSchedule:
    """Tests for get_temperature_schedule function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("constant", TemperatureSchedule.CONSTANT),
            ("linear_decay", TemperatureSchedule.LINEAR_DECAY),
            ("cosine_decay", TemperatureSchedule.COSINE_DECAY),
            ("exponential_decay", TemperatureSchedule.EXPONENTIAL_DECAY),
            ("warmup", TemperatureSchedule.WARMUP),
        ],
    )
    def test_all_schedules(self, name: str, expected: TemperatureSchedule) -> None:
        """Test all valid schedules."""
        assert get_temperature_schedule(name) == expected

    def test_invalid_schedule_raises(self) -> None:
        """Invalid schedule raises ValueError."""
        with pytest.raises(ValueError, match="schedule must be one of"):
            get_temperature_schedule("invalid")


class TestGetStudentInitialization:
    """Tests for get_student_initialization function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("random", StudentInitialization.RANDOM),
            ("teacher_subset", StudentInitialization.TEACHER_SUBSET),
            ("pretrained", StudentInitialization.PRETRAINED),
            ("pruned", StudentInitialization.PRUNED),
        ],
    )
    def test_all_initializations(
        self, name: str, expected: StudentInitialization
    ) -> None:
        """Test all valid initializations."""
        assert get_student_initialization(name) == expected

    def test_invalid_initialization_raises(self) -> None:
        """Invalid initialization raises ValueError."""
        with pytest.raises(ValueError, match="initialization must be one of"):
            get_student_initialization("invalid")


class TestCalculateTemperatureAtStep:
    """Tests for calculate_temperature_at_step function."""

    def test_constant_schedule(self) -> None:
        """Constant schedule returns same temperature."""
        config = create_distillation_config(
            temperature=4.0,
            temperature_schedule="constant",
        )
        assert calculate_temperature_at_step(config, 0, 1000) == pytest.approx(4.0)
        assert calculate_temperature_at_step(config, 500, 1000) == pytest.approx(4.0)
        assert calculate_temperature_at_step(config, 1000, 1000) == pytest.approx(4.0)

    def test_linear_decay(self) -> None:
        """Linear decay decreases linearly."""
        config = create_distillation_config(
            temperature=4.0,
            final_temperature=1.0,
            temperature_schedule="linear_decay",
        )
        assert calculate_temperature_at_step(config, 0, 1000) == pytest.approx(4.0)
        assert calculate_temperature_at_step(config, 500, 1000) == pytest.approx(2.5)
        assert calculate_temperature_at_step(config, 1000, 1000) == pytest.approx(1.0)

    def test_cosine_decay(self) -> None:
        """Cosine decay follows cosine curve."""
        config = create_distillation_config(
            temperature=4.0,
            final_temperature=1.0,
            temperature_schedule="cosine_decay",
        )
        mid = calculate_temperature_at_step(config, 500, 1000)
        assert 1.0 < mid < 4.0

    def test_exponential_decay(self) -> None:
        """Exponential decay decreases exponentially."""
        config = create_distillation_config(
            temperature=4.0,
            final_temperature=1.0,
            temperature_schedule="exponential_decay",
        )
        mid = calculate_temperature_at_step(config, 500, 1000)
        assert 1.0 < mid < 4.0

    def test_warmup_schedule(self) -> None:
        """Warmup schedule warms up then holds."""
        config = create_distillation_config(
            temperature=4.0,
            final_temperature=1.0,
            temperature_schedule="warmup",
            warmup_steps=500,
        )
        early = calculate_temperature_at_step(config, 250, 1000)
        assert early > 1.0 and early < 4.0
        final = calculate_temperature_at_step(config, 1000, 1000)
        assert final == pytest.approx(4.0)

    def test_negative_step_raises(self) -> None:
        """Negative current_step raises ValueError."""
        config = create_distillation_config()
        with pytest.raises(ValueError, match="current_step must be non-negative"):
            calculate_temperature_at_step(config, -1, 1000)

    def test_zero_total_steps_raises(self) -> None:
        """Zero total_steps raises ValueError."""
        config = create_distillation_config()
        with pytest.raises(ValueError, match="total_steps must be positive"):
            calculate_temperature_at_step(config, 0, 0)

    def test_step_exceeds_total_raises(self) -> None:
        """current_step > total_steps raises ValueError."""
        config = create_distillation_config()
        with pytest.raises(ValueError, match=r"current_step.*cannot exceed"):
            calculate_temperature_at_step(config, 1001, 1000)


class TestCalculateSoftLabelsLoss:
    """Tests for calculate_soft_labels_loss function."""

    def test_basic_loss(self) -> None:
        """Calculate basic KL divergence loss."""
        student = (1.0, 2.0, 3.0)
        teacher = (1.5, 2.5, 3.5)
        loss = calculate_soft_labels_loss(student, teacher, 4.0)
        assert loss >= 0

    def test_identical_logits_near_zero(self) -> None:
        """Identical logits give near-zero loss."""
        logits = (1.0, 2.0, 3.0)
        loss = calculate_soft_labels_loss(logits, logits, 4.0)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_empty_logits_raises(self) -> None:
        """Empty logits raises ValueError."""
        with pytest.raises(ValueError, match="logits cannot be empty"):
            calculate_soft_labels_loss((), (), 4.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_soft_labels_loss((1.0,), (1.0, 2.0), 4.0)

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            calculate_soft_labels_loss((1.0, 2.0), (1.0, 2.0), 0.0)

    def test_higher_temp_smoother_loss(self) -> None:
        """Higher temperature leads to smoother distribution."""
        student = (1.0, 2.0, 3.0)
        teacher = (0.5, 2.5, 4.0)
        loss_low = calculate_soft_labels_loss(student, teacher, 1.0)
        loss_high = calculate_soft_labels_loss(student, teacher, 10.0)
        assert loss_high != loss_low


class TestCalculateCombinedLoss:
    """Tests for calculate_combined_loss function."""

    def test_basic_combined_loss(self) -> None:
        """Calculate basic combined loss."""
        combined = calculate_combined_loss(0.5, 0.3, 0.7)
        assert combined == pytest.approx(0.44)

    def test_alpha_one_ignores_task_loss(self) -> None:
        """Alpha=1 uses only distillation loss."""
        combined = calculate_combined_loss(0.5, 0.3, 1.0)
        assert combined == pytest.approx(0.5)

    def test_alpha_zero_ignores_distillation_loss(self) -> None:
        """Alpha=0 uses only task loss."""
        combined = calculate_combined_loss(0.5, 0.3, 0.0)
        assert combined == pytest.approx(0.3)

    def test_alpha_half_averages(self) -> None:
        """Alpha=0.5 averages losses."""
        combined = calculate_combined_loss(1.0, 0.0, 0.5)
        assert combined == pytest.approx(0.5)

    def test_invalid_alpha_raises(self) -> None:
        """Invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            calculate_combined_loss(0.5, 0.3, 1.5)


class TestEstimateCompressionRatio:
    """Tests for estimate_compression_ratio function."""

    def test_basic_ratio(self) -> None:
        """Calculate basic compression ratio."""
        ratio = estimate_compression_ratio(110_000_000, 66_000_000)
        assert ratio == pytest.approx(0.6)

    def test_same_size_is_one(self) -> None:
        """Same size gives ratio of 1."""
        ratio = estimate_compression_ratio(100_000, 100_000)
        assert ratio == pytest.approx(1.0)

    def test_smaller_student(self) -> None:
        """Smaller student gives ratio < 1."""
        ratio = estimate_compression_ratio(340_000_000, 66_000_000)
        assert ratio < 1.0

    def test_zero_teacher_raises(self) -> None:
        """Zero teacher_params raises ValueError."""
        with pytest.raises(ValueError, match="teacher_params must be positive"):
            estimate_compression_ratio(0, 66_000_000)

    def test_zero_student_raises(self) -> None:
        """Zero student_params raises ValueError."""
        with pytest.raises(ValueError, match="student_params must be positive"):
            estimate_compression_ratio(110_000_000, 0)


class TestEstimateModelParameters:
    """Tests for estimate_model_parameters function."""

    def test_basic_estimate(self) -> None:
        """Estimate parameters for BERT-base."""
        params = estimate_model_parameters(12, 768)
        assert params > 100_000_000

    def test_smaller_model(self) -> None:
        """Smaller model has fewer parameters."""
        params_base = estimate_model_parameters(12, 768)
        params_small = estimate_model_parameters(6, 768)
        assert params_small < params_base

    def test_custom_intermediate(self) -> None:
        """Custom intermediate size affects params."""
        params_default = estimate_model_parameters(12, 768)
        params_custom = estimate_model_parameters(12, 768, intermediate_size=2048)
        assert params_custom < params_default

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_model_parameters(0, 768)

    def test_zero_hidden_size_raises(self) -> None:
        """Zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_model_parameters(12, 0)

    def test_zero_vocab_size_raises(self) -> None:
        """Zero vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            estimate_model_parameters(12, 768, vocab_size=0)


class TestGetRecommendedDistillationConfig:
    """Tests for get_recommended_distillation_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_distillation_config("classification")
        assert config.temperature == pytest.approx(4.0)
        assert config.alpha == pytest.approx(0.7)
        assert config.method == DistillationMethod.RESPONSE_BASED

    def test_generation_config(self) -> None:
        """Get config for generation task."""
        config = get_recommended_distillation_config("generation")
        assert config.temperature == pytest.approx(2.0)
        assert config.alpha == pytest.approx(0.5)

    def test_embedding_config(self) -> None:
        """Get config for embedding task."""
        config = get_recommended_distillation_config("embedding")
        assert config.method == DistillationMethod.FEATURE_BASED
        assert config.alpha == pytest.approx(0.8)

    def test_qa_config(self) -> None:
        """Get config for QA task."""
        config = get_recommended_distillation_config("qa")
        assert config.temperature == pytest.approx(3.0)
        assert config.alpha == pytest.approx(0.6)

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_distillation_config("unknown")


class TestFormatDistillationStats:
    """Tests for format_distillation_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_distillation_stats(
            total_steps=1000,
            distillation_loss=0.5,
            task_loss=0.3,
            combined_loss=0.44,
            compression_ratio=0.25,
        )
        formatted = format_distillation_stats(stats)
        assert "Steps: 1000" in formatted
        assert "Distillation Loss: 0.5000" in formatted
        assert "Compression: 25.0%" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_distillation_stats(total_steps=500)
        formatted = format_distillation_stats(stats)
        assert "Steps:" in formatted
        assert "Distillation Loss:" in formatted
        assert "Task Loss:" in formatted
        assert "Combined Loss:" in formatted
        assert "Temperature:" in formatted
        assert "Compression:" in formatted


class TestGetLayerMappingStrategy:
    """Tests for get_layer_mapping_strategy function."""

    def test_uniform_mapping(self) -> None:
        """Uniform mapping distributes evenly."""
        mapping = get_layer_mapping_strategy(12, 6, "uniform")
        assert len(mapping) == 6
        assert mapping[0] == 0
        assert mapping[-1] == 11

    def test_skip_first_mapping(self) -> None:
        """Skip first mapping uses last layers."""
        mapping = get_layer_mapping_strategy(12, 6, "skip_first")
        assert mapping == (6, 7, 8, 9, 10, 11)

    def test_skip_last_mapping(self) -> None:
        """Skip last mapping uses first layers."""
        mapping = get_layer_mapping_strategy(12, 6, "skip_last")
        assert mapping == (0, 1, 2, 3, 4, 5)

    def test_single_student_layer(self) -> None:
        """Single student layer maps to last teacher layer."""
        mapping = get_layer_mapping_strategy(12, 1, "uniform")
        assert mapping == (11,)

    def test_zero_teacher_raises(self) -> None:
        """Zero teacher_layers raises ValueError."""
        with pytest.raises(ValueError, match="teacher_layers must be positive"):
            get_layer_mapping_strategy(0, 6, "uniform")

    def test_zero_student_raises(self) -> None:
        """Zero student_layers raises ValueError."""
        with pytest.raises(ValueError, match="student_layers must be positive"):
            get_layer_mapping_strategy(12, 0, "uniform")

    def test_student_exceeds_teacher_raises(self) -> None:
        """Student > teacher raises ValueError."""
        with pytest.raises(ValueError, match=r"student_layers.*cannot exceed"):
            get_layer_mapping_strategy(6, 12, "uniform")

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_layer_mapping_strategy(12, 6, "invalid")
