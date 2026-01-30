"""Tests for trainer management utilities."""

from __future__ import annotations

import pytest

from hf_gtc.training.trainer import (
    CheckpointConfig,
    EarlyStoppingConfig,
    EarlyStoppingMode,
    SchedulerConfig,
    SchedulerType,
    TrainerState,
    TrainingProgress,
    compute_warmup_steps,
    create_trainer_state,
    create_training_progress,
    format_training_progress,
    get_checkpoint_path,
    get_checkpoints_to_delete,
    get_early_stopping_mode,
    get_latest_checkpoint,
    get_scheduler_type,
    is_metric_improved,
    list_checkpoints,
    list_early_stopping_modes,
    list_scheduler_types,
    should_stop_early,
    update_trainer_state,
    validate_checkpoint_config,
    validate_early_stopping_config,
    validate_early_stopping_mode,
    validate_scheduler_config,
    validate_scheduler_type,
    validate_trainer_state,
)


class TestSchedulerType:
    """Tests for SchedulerType enum."""

    def test_linear_value(self) -> None:
        """Test LINEAR value."""
        assert SchedulerType.LINEAR.value == "linear"

    def test_cosine_value(self) -> None:
        """Test COSINE value."""
        assert SchedulerType.COSINE.value == "cosine"

    def test_cosine_with_restarts_value(self) -> None:
        """Test COSINE_WITH_RESTARTS value."""
        assert SchedulerType.COSINE_WITH_RESTARTS.value == "cosine_with_restarts"

    def test_polynomial_value(self) -> None:
        """Test POLYNOMIAL value."""
        assert SchedulerType.POLYNOMIAL.value == "polynomial"

    def test_constant_value(self) -> None:
        """Test CONSTANT value."""
        assert SchedulerType.CONSTANT.value == "constant"


class TestEarlyStoppingMode:
    """Tests for EarlyStoppingMode enum."""

    def test_min_value(self) -> None:
        """Test MIN value."""
        assert EarlyStoppingMode.MIN.value == "min"

    def test_max_value(self) -> None:
        """Test MAX value."""
        assert EarlyStoppingMode.MAX.value == "max"


class TestTrainerState:
    """Tests for TrainerState dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        state = TrainerState()
        assert state.global_step == 0
        assert state.epoch == 0.0
        assert state.best_metric == 0.0
        assert state.best_model_checkpoint is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        state = TrainerState(global_step=100, epoch=1.5, best_metric=0.95)
        assert state.global_step == 100
        assert state.epoch == 1.5
        assert state.best_metric == 0.95

    def test_frozen(self) -> None:
        """Test that TrainerState is immutable."""
        state = TrainerState(global_step=100)
        with pytest.raises(AttributeError):
            state.global_step = 200  # type: ignore[misc]


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = CheckpointConfig()
        assert config.save_total_limit is None
        assert config.save_on_each_node is False
        assert config.save_only_model is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = CheckpointConfig(save_total_limit=3, save_only_model=True)
        assert config.save_total_limit == 3
        assert config.save_only_model is True

    def test_frozen(self) -> None:
        """Test that CheckpointConfig is immutable."""
        config = CheckpointConfig()
        with pytest.raises(AttributeError):
            config.save_total_limit = 5  # type: ignore[misc]


class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = EarlyStoppingConfig()
        assert config.patience == 3
        assert config.threshold == 0.0
        assert config.mode == EarlyStoppingMode.MIN
        assert config.metric_name == "eval_loss"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = EarlyStoppingConfig(
            patience=5, mode=EarlyStoppingMode.MAX, metric_name="eval_accuracy"
        )
        assert config.patience == 5
        assert config.mode == EarlyStoppingMode.MAX
        assert config.metric_name == "eval_accuracy"

    def test_frozen(self) -> None:
        """Test that EarlyStoppingConfig is immutable."""
        config = EarlyStoppingConfig()
        with pytest.raises(AttributeError):
            config.patience = 10  # type: ignore[misc]


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = SchedulerConfig()
        assert config.scheduler_type == SchedulerType.LINEAR
        assert config.num_warmup_steps == 0
        assert config.num_training_steps is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        assert config.scheduler_type == SchedulerType.COSINE
        assert config.num_warmup_steps == 100
        assert config.num_training_steps == 1000


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_creation(self) -> None:
        """Test creating TrainingProgress."""
        progress = TrainingProgress(
            current_step=50, total_steps=100, current_epoch=1, total_epochs=3
        )
        assert progress.current_step == 50
        assert progress.total_steps == 100

    def test_percent_complete(self) -> None:
        """Test percent_complete property."""
        progress = TrainingProgress(current_step=25, total_steps=100)
        assert progress.percent_complete == 25.0

    def test_percent_complete_zero_total(self) -> None:
        """Test percent_complete with zero total steps."""
        progress = TrainingProgress(current_step=0, total_steps=0)
        assert progress.percent_complete == 0.0

    def test_steps_remaining(self) -> None:
        """Test steps_remaining property."""
        progress = TrainingProgress(current_step=30, total_steps=100)
        assert progress.steps_remaining == 70


class TestValidateTrainerState:
    """Tests for validate_trainer_state function."""

    def test_valid_state(self) -> None:
        """Test validating valid state."""
        state = TrainerState(global_step=100, epoch=1.5)
        validate_trainer_state(state)  # Should not raise

    def test_none_state_raises_error(self) -> None:
        """Test that None state raises ValueError."""
        with pytest.raises(ValueError, match="state cannot be None"):
            validate_trainer_state(None)  # type: ignore[arg-type]

    def test_negative_global_step_raises_error(self) -> None:
        """Test that negative global_step raises ValueError."""
        state = TrainerState(global_step=-1)
        with pytest.raises(ValueError, match="global_step cannot be negative"):
            validate_trainer_state(state)

    def test_negative_epoch_raises_error(self) -> None:
        """Test that negative epoch raises ValueError."""
        state = TrainerState(epoch=-1.0)
        with pytest.raises(ValueError, match="epoch cannot be negative"):
            validate_trainer_state(state)


class TestValidateCheckpointConfig:
    """Tests for validate_checkpoint_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = CheckpointConfig(save_total_limit=3)
        validate_checkpoint_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_checkpoint_config(None)  # type: ignore[arg-type]

    def test_zero_save_limit_raises_error(self) -> None:
        """Test that zero save_total_limit raises ValueError."""
        config = CheckpointConfig(save_total_limit=0)
        with pytest.raises(ValueError, match="save_total_limit must be positive"):
            validate_checkpoint_config(config)


class TestValidateEarlyStoppingConfig:
    """Tests for validate_early_stopping_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = EarlyStoppingConfig(patience=3, metric_name="eval_loss")
        validate_early_stopping_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_early_stopping_config(None)  # type: ignore[arg-type]

    def test_zero_patience_raises_error(self) -> None:
        """Test that zero patience raises ValueError."""
        config = EarlyStoppingConfig(patience=0)
        with pytest.raises(ValueError, match="patience must be positive"):
            validate_early_stopping_config(config)

    def test_negative_threshold_raises_error(self) -> None:
        """Test that negative threshold raises ValueError."""
        config = EarlyStoppingConfig(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold cannot be negative"):
            validate_early_stopping_config(config)

    def test_empty_metric_name_raises_error(self) -> None:
        """Test that empty metric_name raises ValueError."""
        config = EarlyStoppingConfig(metric_name="")
        with pytest.raises(ValueError, match="metric_name cannot be empty"):
            validate_early_stopping_config(config)


class TestValidateSchedulerConfig:
    """Tests for validate_scheduler_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = SchedulerConfig(scheduler_type=SchedulerType.LINEAR)
        validate_scheduler_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_scheduler_config(None)  # type: ignore[arg-type]

    def test_negative_warmup_steps_raises_error(self) -> None:
        """Test that negative num_warmup_steps raises ValueError."""
        config = SchedulerConfig(num_warmup_steps=-1)
        with pytest.raises(ValueError, match="num_warmup_steps cannot be negative"):
            validate_scheduler_config(config)

    def test_zero_training_steps_raises_error(self) -> None:
        """Test that zero num_training_steps raises ValueError."""
        config = SchedulerConfig(num_training_steps=0)
        with pytest.raises(ValueError, match="num_training_steps must be positive"):
            validate_scheduler_config(config)


class TestCreateTrainerState:
    """Tests for create_trainer_state function."""

    def test_creates_state(self) -> None:
        """Test creating trainer state."""
        state = create_trainer_state(global_step=100, epoch=1.5)
        assert state.global_step == 100
        assert state.epoch == 1.5

    def test_default_values(self) -> None:
        """Test default values."""
        state = create_trainer_state()
        assert state.global_step == 0
        assert state.epoch == 0.0

    def test_negative_step_raises_error(self) -> None:
        """Test that negative global_step raises ValueError."""
        with pytest.raises(ValueError, match="global_step cannot be negative"):
            create_trainer_state(global_step=-1)


class TestUpdateTrainerState:
    """Tests for update_trainer_state function."""

    def test_updates_state(self) -> None:
        """Test updating trainer state."""
        state = create_trainer_state(global_step=100, epoch=1.0)
        updated = update_trainer_state(state, global_step=200, epoch=2.0)
        assert updated.global_step == 200
        assert updated.epoch == 2.0

    def test_preserves_unchanged_values(self) -> None:
        """Test that unchanged values are preserved."""
        state = create_trainer_state(global_step=100, epoch=1.0, best_metric=0.9)
        updated = update_trainer_state(state, global_step=200)
        assert updated.epoch == 1.0
        assert updated.best_metric == 0.9

    def test_adds_log_entry(self) -> None:
        """Test adding log entry."""
        state = create_trainer_state()
        updated = update_trainer_state(state, new_log_entry={"loss": 0.5})
        assert len(updated.log_history) == 1
        assert updated.log_history[0]["loss"] == 0.5

    def test_none_state_raises_error(self) -> None:
        """Test that None state raises ValueError."""
        with pytest.raises(ValueError, match="state cannot be None"):
            update_trainer_state(None, global_step=100)  # type: ignore[arg-type]


class TestShouldStopEarly:
    """Tests for should_stop_early function."""

    def test_should_stop(self) -> None:
        """Test when should stop."""
        config = EarlyStoppingConfig(patience=3, mode=EarlyStoppingMode.MIN)
        assert should_stop_early(config, 0.5, 0.4, 3) is True

    def test_should_not_stop(self) -> None:
        """Test when should not stop."""
        config = EarlyStoppingConfig(patience=3, mode=EarlyStoppingMode.MIN)
        assert should_stop_early(config, 0.5, 0.4, 2) is False

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            should_stop_early(None, 0.5, 0.4, 3)  # type: ignore[arg-type]


class TestIsMetricImproved:
    """Tests for is_metric_improved function."""

    def test_improved_min_mode(self) -> None:
        """Test improvement in min mode."""
        config = EarlyStoppingConfig(mode=EarlyStoppingMode.MIN, threshold=0.01)
        assert is_metric_improved(config, 0.3, 0.4) is True

    def test_not_improved_min_mode(self) -> None:
        """Test no improvement in min mode (within threshold)."""
        config = EarlyStoppingConfig(mode=EarlyStoppingMode.MIN, threshold=0.01)
        assert is_metric_improved(config, 0.39, 0.4) is False

    def test_improved_max_mode(self) -> None:
        """Test improvement in max mode."""
        config = EarlyStoppingConfig(mode=EarlyStoppingMode.MAX, threshold=0.01)
        assert is_metric_improved(config, 0.9, 0.8) is True

    def test_not_improved_max_mode(self) -> None:
        """Test no improvement in max mode (within threshold)."""
        config = EarlyStoppingConfig(mode=EarlyStoppingMode.MAX, threshold=0.01)
        assert is_metric_improved(config, 0.81, 0.8) is False

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            is_metric_improved(None, 0.5, 0.4)  # type: ignore[arg-type]


class TestComputeWarmupSteps:
    """Tests for compute_warmup_steps function."""

    def test_computes_warmup_steps(self) -> None:
        """Test computing warmup steps."""
        assert compute_warmup_steps(1000, 0.1) == 100

    def test_zero_ratio(self) -> None:
        """Test with zero ratio."""
        assert compute_warmup_steps(1000, 0.0) == 0

    def test_full_ratio(self) -> None:
        """Test with full ratio."""
        assert compute_warmup_steps(1000, 1.0) == 1000

    def test_negative_total_raises_error(self) -> None:
        """Test that negative total_steps raises ValueError."""
        with pytest.raises(ValueError, match="total_steps cannot be negative"):
            compute_warmup_steps(-1, 0.1)

    def test_invalid_ratio_raises_error(self) -> None:
        """Test that invalid warmup_ratio raises ValueError."""
        with pytest.raises(ValueError, match="warmup_ratio must be between 0 and 1"):
            compute_warmup_steps(1000, 1.5)


class TestGetCheckpointPath:
    """Tests for get_checkpoint_path function."""

    def test_generates_path(self) -> None:
        """Test generating checkpoint path."""
        path = get_checkpoint_path("/models/my-model", 1000)
        assert path == "/models/my-model/checkpoint-1000"

    def test_empty_dir_raises_error(self) -> None:
        """Test that empty output_dir raises ValueError."""
        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            get_checkpoint_path("", 1000)

    def test_negative_step_raises_error(self) -> None:
        """Test that negative global_step raises ValueError."""
        with pytest.raises(ValueError, match="global_step cannot be negative"):
            get_checkpoint_path("/models", -1)


class TestListCheckpoints:
    """Tests for list_checkpoints function."""

    def test_lists_and_sorts_checkpoints(self) -> None:
        """Test listing and sorting checkpoints."""
        paths = [
            "/model/checkpoint-100",
            "/model/checkpoint-500",
            "/model/checkpoint-200",
        ]
        result = list_checkpoints(paths)
        assert result == [
            ("/model/checkpoint-500", 500),
            ("/model/checkpoint-200", 200),
            ("/model/checkpoint-100", 100),
        ]

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert list_checkpoints([]) == []

    def test_filters_invalid_paths(self) -> None:
        """Test filtering invalid paths."""
        paths = ["/model/checkpoint-100", "/invalid/path"]
        result = list_checkpoints(paths)
        assert len(result) == 1
        assert result[0][0] == "/model/checkpoint-100"

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_paths cannot be None"):
            list_checkpoints(None)  # type: ignore[arg-type]


class TestGetLatestCheckpoint:
    """Tests for get_latest_checkpoint function."""

    def test_gets_latest(self) -> None:
        """Test getting latest checkpoint."""
        paths = [
            "/model/checkpoint-100",
            "/model/checkpoint-500",
            "/model/checkpoint-200",
        ]
        assert get_latest_checkpoint(paths) == "/model/checkpoint-500"

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert get_latest_checkpoint([]) is None

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_paths cannot be None"):
            get_latest_checkpoint(None)  # type: ignore[arg-type]


class TestGetCheckpointsToDelete:
    """Tests for get_checkpoints_to_delete function."""

    def test_gets_checkpoints_to_delete(self) -> None:
        """Test getting checkpoints to delete."""
        paths = [
            "/model/checkpoint-100",
            "/model/checkpoint-200",
            "/model/checkpoint-300",
        ]
        to_delete = get_checkpoints_to_delete(paths, 2)
        assert to_delete == ["/model/checkpoint-100"]

    def test_no_deletion_needed(self) -> None:
        """Test when no deletion needed."""
        paths = ["/model/checkpoint-100", "/model/checkpoint-200"]
        to_delete = get_checkpoints_to_delete(paths, 5)
        assert to_delete == []

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_paths cannot be None"):
            get_checkpoints_to_delete(None, 2)  # type: ignore[arg-type]

    def test_zero_limit_raises_error(self) -> None:
        """Test that zero limit raises ValueError."""
        with pytest.raises(ValueError, match="save_total_limit must be positive"):
            get_checkpoints_to_delete(["/model/checkpoint-100"], 0)


class TestCreateTrainingProgress:
    """Tests for create_training_progress function."""

    def test_creates_progress(self) -> None:
        """Test creating training progress."""
        progress = create_training_progress(50, 100, 1, 3)
        assert progress.current_step == 50
        assert progress.total_steps == 100
        assert progress.percent_complete == 50.0

    def test_with_loss(self) -> None:
        """Test with loss value."""
        progress = create_training_progress(50, 100, 1, 3, loss=0.5)
        assert progress.loss == 0.5

    def test_negative_step_raises_error(self) -> None:
        """Test that negative current_step raises ValueError."""
        with pytest.raises(ValueError, match="current_step cannot be negative"):
            create_training_progress(-1, 100, 1, 3)

    def test_negative_total_raises_error(self) -> None:
        """Test that negative total_steps raises ValueError."""
        with pytest.raises(ValueError, match="total_steps cannot be negative"):
            create_training_progress(0, -1, 1, 3)

    def test_zero_epochs_raises_error(self) -> None:
        """Test that zero total_epochs raises ValueError."""
        with pytest.raises(ValueError, match="total_epochs must be positive"):
            create_training_progress(0, 100, 0, 0)


class TestFormatTrainingProgress:
    """Tests for format_training_progress function."""

    def test_formats_progress(self) -> None:
        """Test formatting progress."""
        progress = TrainingProgress(
            current_step=50,
            total_steps=100,
            current_epoch=1,
            total_epochs=3,
        )
        formatted = format_training_progress(progress)
        assert "50/100" in formatted
        assert "50.0%" in formatted
        assert "Epoch 1/3" in formatted

    def test_with_loss(self) -> None:
        """Test formatting with loss."""
        progress = TrainingProgress(
            current_step=50,
            total_steps=100,
            current_epoch=1,
            total_epochs=3,
            loss=0.5,
        )
        formatted = format_training_progress(progress)
        assert "Loss: 0.5000" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="progress cannot be None"):
            format_training_progress(None)  # type: ignore[arg-type]


class TestListSchedulerTypes:
    """Tests for list_scheduler_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_scheduler_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_scheduler_types()
        assert "linear" in types
        assert "cosine" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_scheduler_types()
        assert types == sorted(types)


class TestValidateSchedulerType:
    """Tests for validate_scheduler_type function."""

    def test_valid_linear(self) -> None:
        """Test validation of linear type."""
        assert validate_scheduler_type("linear") is True

    def test_valid_cosine(self) -> None:
        """Test validation of cosine type."""
        assert validate_scheduler_type("cosine") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_scheduler_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_scheduler_type("") is False


class TestGetSchedulerType:
    """Tests for get_scheduler_type function."""

    def test_get_linear(self) -> None:
        """Test getting LINEAR type."""
        assert get_scheduler_type("linear") == SchedulerType.LINEAR

    def test_get_cosine(self) -> None:
        """Test getting COSINE type."""
        assert get_scheduler_type("cosine") == SchedulerType.COSINE

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid scheduler type"):
            get_scheduler_type("invalid")


class TestListEarlyStoppingModes:
    """Tests for list_early_stopping_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        modes = list_early_stopping_modes()
        assert isinstance(modes, list)

    def test_contains_expected_modes(self) -> None:
        """Test that list contains expected modes."""
        modes = list_early_stopping_modes()
        assert "min" in modes
        assert "max" in modes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        modes = list_early_stopping_modes()
        assert modes == sorted(modes)


class TestValidateEarlyStoppingMode:
    """Tests for validate_early_stopping_mode function."""

    def test_valid_min(self) -> None:
        """Test validation of min mode."""
        assert validate_early_stopping_mode("min") is True

    def test_valid_max(self) -> None:
        """Test validation of max mode."""
        assert validate_early_stopping_mode("max") is True

    def test_invalid_mode(self) -> None:
        """Test validation of invalid mode."""
        assert validate_early_stopping_mode("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_early_stopping_mode("") is False


class TestGetEarlyStoppingMode:
    """Tests for get_early_stopping_mode function."""

    def test_get_min(self) -> None:
        """Test getting MIN mode."""
        assert get_early_stopping_mode("min") == EarlyStoppingMode.MIN

    def test_get_max(self) -> None:
        """Test getting MAX mode."""
        assert get_early_stopping_mode("max") == EarlyStoppingMode.MAX

    def test_invalid_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid early stopping mode"):
            get_early_stopping_mode("invalid")
