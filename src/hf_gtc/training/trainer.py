"""Trainer management utilities for HuggingFace models.

This module provides advanced utilities for managing training runs,
checkpointing, resumption, and progress tracking with HuggingFace Trainer.

Examples:
    >>> from hf_gtc.training.trainer import TrainerState, CheckpointConfig
    >>> state = TrainerState(global_step=100, epoch=1.5, best_metric=0.95)
    >>> state.global_step
    100
    >>> config = CheckpointConfig(save_total_limit=3)
    >>> config.save_total_limit
    3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class SchedulerType(Enum):
    """Learning rate scheduler types.

    Examples:
        >>> SchedulerType.LINEAR.value
        'linear'
        >>> SchedulerType.COSINE.value
        'cosine'
    """

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"


class EarlyStoppingMode(Enum):
    """Early stopping comparison modes.

    Examples:
        >>> EarlyStoppingMode.MIN.value
        'min'
        >>> EarlyStoppingMode.MAX.value
        'max'
    """

    MIN = "min"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class TrainerState:
    """State of a training run.

    Attributes:
        global_step: Current global step.
        epoch: Current epoch (can be fractional).
        best_metric: Best metric value seen.
        best_model_checkpoint: Path to best model checkpoint.
        is_world_process_zero: Whether this is the main process.
        log_history: History of logged metrics.

    Examples:
        >>> state = TrainerState(global_step=100, epoch=1.5)
        >>> state.global_step
        100
        >>> state.epoch
        1.5
        >>> state.best_metric
        0.0

        >>> state2 = TrainerState(global_step=200, epoch=2.0, best_metric=0.95)
        >>> state2.best_metric
        0.95
    """

    global_step: int = 0
    epoch: float = 0.0
    best_metric: float = 0.0
    best_model_checkpoint: str | None = None
    is_world_process_zero: bool = True
    log_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Configuration for checkpoint management.

    Attributes:
        save_total_limit: Maximum number of checkpoints to keep.
        save_on_each_node: Save on each node in distributed training.
        save_only_model: Only save model weights, not optimizer state.
        resume_from_checkpoint: Path to checkpoint to resume from.
        load_best_model_at_end: Load best model at end of training.

    Examples:
        >>> config = CheckpointConfig(save_total_limit=3)
        >>> config.save_total_limit
        3
        >>> config.save_only_model
        False

        >>> config2 = CheckpointConfig(save_only_model=True)
        >>> config2.save_only_model
        True
    """

    save_total_limit: int | None = None
    save_on_each_node: bool = False
    save_only_model: bool = False
    resume_from_checkpoint: str | None = None
    load_best_model_at_end: bool = False


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        patience: Number of evaluations to wait for improvement.
        threshold: Minimum change to qualify as improvement.
        mode: Whether to minimize or maximize the metric.
        metric_name: Name of the metric to monitor.

    Examples:
        >>> config = EarlyStoppingConfig(patience=3, metric_name="eval_loss")
        >>> config.patience
        3
        >>> config.mode
        <EarlyStoppingMode.MIN: 'min'>

        >>> config2 = EarlyStoppingConfig(patience=5, mode=EarlyStoppingMode.MAX)
        >>> config2.mode
        <EarlyStoppingMode.MAX: 'max'>
    """

    patience: int = 3
    threshold: float = 0.0
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN
    metric_name: str = "eval_loss"


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configuration for learning rate scheduler.

    Attributes:
        scheduler_type: Type of scheduler to use.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cycles for cosine scheduler.
        power: Power for polynomial scheduler.

    Examples:
        >>> config = SchedulerConfig(scheduler_type=SchedulerType.LINEAR)
        >>> config.scheduler_type.value
        'linear'
        >>> config.num_warmup_steps
        0

        >>> config2 = SchedulerConfig(
        ...     scheduler_type=SchedulerType.COSINE,
        ...     num_warmup_steps=100,
        ...     num_training_steps=1000,
        ... )
        >>> config2.num_warmup_steps
        100
    """

    scheduler_type: SchedulerType = SchedulerType.LINEAR
    num_warmup_steps: int = 0
    num_training_steps: int | None = None
    num_cycles: float = 0.5
    power: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Progress information for a training run.

    Attributes:
        current_step: Current training step.
        total_steps: Total training steps.
        current_epoch: Current epoch number.
        total_epochs: Total number of epochs.
        loss: Current loss value.
        learning_rate: Current learning rate.
        samples_seen: Number of samples processed.

    Examples:
        >>> progress = TrainingProgress(
        ...     current_step=50,
        ...     total_steps=100,
        ...     current_epoch=1,
        ...     total_epochs=3,
        ... )
        >>> progress.percent_complete
        50.0
        >>> progress.steps_remaining
        50
    """

    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float | None = None
    learning_rate: float | None = None
    samples_seen: int = 0

    @property
    def percent_complete(self) -> float:
        """Calculate percentage of training complete.

        Returns:
            Percentage complete (0-100).

        Examples:
            >>> TrainingProgress(current_step=25, total_steps=100).percent_complete
            25.0
            >>> TrainingProgress(current_step=0, total_steps=0).percent_complete
            0.0
        """
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0

    @property
    def steps_remaining(self) -> int:
        """Calculate remaining training steps.

        Returns:
            Number of steps remaining.

        Examples:
            >>> TrainingProgress(current_step=30, total_steps=100).steps_remaining
            70
        """
        return max(0, self.total_steps - self.current_step)


def validate_trainer_state(state: TrainerState) -> None:
    """Validate trainer state.

    Args:
        state: Trainer state to validate.

    Raises:
        ValueError: If state is None.
        ValueError: If global_step is negative.
        ValueError: If epoch is negative.

    Examples:
        >>> state = TrainerState(global_step=100, epoch=1.5)
        >>> validate_trainer_state(state)  # No error

        >>> validate_trainer_state(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: state cannot be None

        >>> bad_state = TrainerState(global_step=-1)
        >>> validate_trainer_state(bad_state)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: global_step cannot be negative
    """
    if state is None:
        msg = "state cannot be None"
        raise ValueError(msg)

    if state.global_step < 0:
        msg = "global_step cannot be negative"
        raise ValueError(msg)

    if state.epoch < 0:
        msg = "epoch cannot be negative"
        raise ValueError(msg)


def validate_checkpoint_config(config: CheckpointConfig) -> None:
    """Validate checkpoint configuration.

    Args:
        config: Checkpoint configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If save_total_limit is not positive when set.

    Examples:
        >>> config = CheckpointConfig(save_total_limit=3)
        >>> validate_checkpoint_config(config)  # No error

        >>> validate_checkpoint_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = CheckpointConfig(save_total_limit=0)
        >>> validate_checkpoint_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: save_total_limit must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.save_total_limit is not None and config.save_total_limit <= 0:
        msg = "save_total_limit must be positive"
        raise ValueError(msg)


def validate_early_stopping_config(config: EarlyStoppingConfig) -> None:
    """Validate early stopping configuration.

    Args:
        config: Early stopping configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If patience is not positive.
        ValueError: If metric_name is empty.

    Examples:
        >>> config = EarlyStoppingConfig(patience=3, metric_name="eval_loss")
        >>> validate_early_stopping_config(config)  # No error

        >>> validate_early_stopping_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = EarlyStoppingConfig(patience=0)
        >>> validate_early_stopping_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: patience must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.patience <= 0:
        msg = "patience must be positive"
        raise ValueError(msg)

    if config.threshold < 0:
        msg = "threshold cannot be negative"
        raise ValueError(msg)

    if not config.metric_name:
        msg = "metric_name cannot be empty"
        raise ValueError(msg)


def validate_scheduler_config(config: SchedulerConfig) -> None:
    """Validate scheduler configuration.

    Args:
        config: Scheduler configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_warmup_steps is negative.

    Examples:
        >>> config = SchedulerConfig(scheduler_type=SchedulerType.LINEAR)
        >>> validate_scheduler_config(config)  # No error

        >>> validate_scheduler_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = SchedulerConfig(num_warmup_steps=-1)
        >>> validate_scheduler_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_warmup_steps cannot be negative
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_warmup_steps < 0:
        msg = "num_warmup_steps cannot be negative"
        raise ValueError(msg)

    if config.num_training_steps is not None and config.num_training_steps <= 0:
        msg = "num_training_steps must be positive"
        raise ValueError(msg)


def create_trainer_state(
    global_step: int = 0,
    epoch: float = 0.0,
    best_metric: float = 0.0,
    best_model_checkpoint: str | None = None,
) -> TrainerState:
    """Create a new trainer state.

    Args:
        global_step: Current global step. Defaults to 0.
        epoch: Current epoch. Defaults to 0.0.
        best_metric: Best metric value. Defaults to 0.0.
        best_model_checkpoint: Path to best checkpoint. Defaults to None.

    Returns:
        New TrainerState instance.

    Raises:
        ValueError: If global_step is negative.
        ValueError: If epoch is negative.

    Examples:
        >>> state = create_trainer_state(global_step=100, epoch=1.5)
        >>> state.global_step
        100
        >>> state.epoch
        1.5

        >>> create_trainer_state(global_step=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: global_step cannot be negative
    """
    state = TrainerState(
        global_step=global_step,
        epoch=epoch,
        best_metric=best_metric,
        best_model_checkpoint=best_model_checkpoint,
    )
    validate_trainer_state(state)
    return state


def update_trainer_state(
    state: TrainerState,
    *,
    global_step: int | None = None,
    epoch: float | None = None,
    best_metric: float | None = None,
    best_model_checkpoint: str | None = None,
    new_log_entry: dict[str, Any] | None = None,
) -> TrainerState:
    """Update trainer state with new values.

    Args:
        state: Current trainer state.
        global_step: New global step. Defaults to None (no change).
        epoch: New epoch. Defaults to None (no change).
        best_metric: New best metric. Defaults to None (no change).
        best_model_checkpoint: New best checkpoint path. Defaults to None.
        new_log_entry: New log entry to append. Defaults to None.

    Returns:
        Updated TrainerState instance.

    Raises:
        ValueError: If state is None.

    Examples:
        >>> state = create_trainer_state(global_step=100, epoch=1.0)
        >>> updated = update_trainer_state(state, global_step=200, epoch=2.0)
        >>> updated.global_step
        200
        >>> updated.epoch
        2.0

        >>> update_trainer_state(None, global_step=100)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: state cannot be None
    """
    if state is None:
        msg = "state cannot be None"
        raise ValueError(msg)

    log_history = state.log_history
    if new_log_entry is not None:
        log_history = (*state.log_history, new_log_entry)

    new_state = TrainerState(
        global_step=global_step if global_step is not None else state.global_step,
        epoch=epoch if epoch is not None else state.epoch,
        best_metric=best_metric if best_metric is not None else state.best_metric,
        best_model_checkpoint=(
            best_model_checkpoint
            if best_model_checkpoint is not None
            else state.best_model_checkpoint
        ),
        is_world_process_zero=state.is_world_process_zero,
        log_history=log_history,
    )
    validate_trainer_state(new_state)
    return new_state


def should_stop_early(
    config: EarlyStoppingConfig,
    current_metric: float,
    best_metric: float,
    evaluations_without_improvement: int,
) -> bool:
    """Check if training should stop early.

    Args:
        config: Early stopping configuration.
        current_metric: Current metric value.
        best_metric: Best metric value seen.
        evaluations_without_improvement: Consecutive evaluations without improvement.

    Returns:
        True if training should stop, False otherwise.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = EarlyStoppingConfig(patience=3, mode=EarlyStoppingMode.MIN)
        >>> should_stop_early(config, 0.5, 0.4, 3)
        True
        >>> should_stop_early(config, 0.5, 0.4, 2)
        False
        >>> should_stop_early(config, 0.3, 0.4, 2)  # Improved
        False

        >>> should_stop_early(None, 0.5, 0.4, 3)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return evaluations_without_improvement >= config.patience


def is_metric_improved(
    config: EarlyStoppingConfig,
    current_metric: float,
    best_metric: float,
) -> bool:
    """Check if the current metric is an improvement.

    Args:
        config: Early stopping configuration.
        current_metric: Current metric value.
        best_metric: Best metric value seen.

    Returns:
        True if metric improved, False otherwise.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config_min = EarlyStoppingConfig(mode=EarlyStoppingMode.MIN, threshold=0.01)
        >>> is_metric_improved(config_min, 0.3, 0.4)
        True
        >>> is_metric_improved(config_min, 0.39, 0.4)  # Below threshold
        False

        >>> config_max = EarlyStoppingConfig(mode=EarlyStoppingMode.MAX, threshold=0.01)
        >>> is_metric_improved(config_max, 0.9, 0.8)
        True
        >>> is_metric_improved(config_max, 0.81, 0.8)  # Below threshold
        False

        >>> is_metric_improved(None, 0.5, 0.4)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.mode == EarlyStoppingMode.MIN:
        return current_metric < best_metric - config.threshold
    return current_metric > best_metric + config.threshold


def compute_warmup_steps(
    total_steps: int,
    warmup_ratio: float,
) -> int:
    """Compute number of warmup steps from ratio.

    Args:
        total_steps: Total training steps.
        warmup_ratio: Ratio of warmup steps.

    Returns:
        Number of warmup steps.

    Raises:
        ValueError: If total_steps is negative.
        ValueError: If warmup_ratio is not between 0 and 1.

    Examples:
        >>> compute_warmup_steps(1000, 0.1)
        100
        >>> compute_warmup_steps(1000, 0.0)
        0
        >>> compute_warmup_steps(500, 0.2)
        100

        >>> compute_warmup_steps(-1, 0.1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_steps cannot be negative

        >>> compute_warmup_steps(1000, 1.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: warmup_ratio must be between 0 and 1
    """
    if total_steps < 0:
        msg = "total_steps cannot be negative"
        raise ValueError(msg)

    if not 0 <= warmup_ratio <= 1:
        msg = "warmup_ratio must be between 0 and 1"
        raise ValueError(msg)

    return int(total_steps * warmup_ratio)


def get_checkpoint_path(
    output_dir: str,
    global_step: int,
) -> str:
    """Generate checkpoint path for a given step.

    Args:
        output_dir: Base output directory.
        global_step: Current global step.

    Returns:
        Path to checkpoint directory.

    Raises:
        ValueError: If output_dir is empty.
        ValueError: If global_step is negative.

    Examples:
        >>> get_checkpoint_path("/models/my-model", 1000)
        '/models/my-model/checkpoint-1000'
        >>> get_checkpoint_path("/tmp/training", 500)
        '/tmp/training/checkpoint-500'

        >>> get_checkpoint_path("", 1000)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: output_dir cannot be empty

        >>> get_checkpoint_path("/models", -1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: global_step cannot be negative
    """
    if not output_dir:
        msg = "output_dir cannot be empty"
        raise ValueError(msg)

    if global_step < 0:
        msg = "global_step cannot be negative"
        raise ValueError(msg)

    return f"{output_dir}/checkpoint-{global_step}"


def list_checkpoints(
    checkpoint_paths: Sequence[str],
) -> list[tuple[str, int]]:
    """Parse and sort checkpoint paths by step number.

    Args:
        checkpoint_paths: List of checkpoint directory paths.

    Returns:
        List of (path, step) tuples sorted by step descending.

    Raises:
        ValueError: If checkpoint_paths is None.

    Examples:
        >>> paths = ["/m/checkpoint-100", "/m/checkpoint-500"]
        >>> result = list_checkpoints(paths)
        >>> result[0]
        ('/m/checkpoint-500', 500)
        >>> len(result)
        2

        >>> list_checkpoints([])
        []

        >>> list_checkpoints(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: checkpoint_paths cannot be None
    """
    if checkpoint_paths is None:
        msg = "checkpoint_paths cannot be None"
        raise ValueError(msg)

    result = []
    for path in checkpoint_paths:
        if "checkpoint-" not in path:
            continue
        try:
            step = int(path.split("checkpoint-")[-1].split("/")[0])
            result.append((path, step))
        except ValueError:
            continue

    return sorted(result, key=lambda x: x[1], reverse=True)


def get_latest_checkpoint(
    checkpoint_paths: Sequence[str],
) -> str | None:
    """Get the path to the latest checkpoint.

    Args:
        checkpoint_paths: List of checkpoint directory paths.

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist.

    Raises:
        ValueError: If checkpoint_paths is None.

    Examples:
        >>> paths = ["/m/checkpoint-100", "/m/checkpoint-500"]
        >>> get_latest_checkpoint(paths)
        '/m/checkpoint-500'

        >>> get_latest_checkpoint([])
        >>> get_latest_checkpoint(["/invalid/path"])

        >>> get_latest_checkpoint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: checkpoint_paths cannot be None
    """
    if checkpoint_paths is None:
        msg = "checkpoint_paths cannot be None"
        raise ValueError(msg)

    checkpoints = list_checkpoints(checkpoint_paths)
    if not checkpoints:
        return None
    return checkpoints[0][0]


def get_checkpoints_to_delete(
    checkpoint_paths: Sequence[str],
    save_total_limit: int,
) -> list[str]:
    """Get checkpoint paths that should be deleted to maintain limit.

    Args:
        checkpoint_paths: List of checkpoint directory paths.
        save_total_limit: Maximum number of checkpoints to keep.

    Returns:
        List of checkpoint paths to delete.

    Raises:
        ValueError: If checkpoint_paths is None.
        ValueError: If save_total_limit is not positive.

    Examples:
        >>> paths = ["/m/checkpoint-100", "/m/checkpoint-200", "/m/checkpoint-300"]
        >>> get_checkpoints_to_delete(paths, 2)
        ['/m/checkpoint-100']
        >>> get_checkpoints_to_delete(paths, 5)
        []

        >>> get_checkpoints_to_delete(None, 2)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: checkpoint_paths cannot be None

        >>> get_checkpoints_to_delete(paths, 0)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: save_total_limit must be positive
    """
    if checkpoint_paths is None:
        msg = "checkpoint_paths cannot be None"
        raise ValueError(msg)

    if save_total_limit <= 0:
        msg = "save_total_limit must be positive"
        raise ValueError(msg)

    checkpoints = list_checkpoints(checkpoint_paths)
    if len(checkpoints) <= save_total_limit:
        return []

    # Return oldest checkpoints (at the end after sorting by step descending)
    return [path for path, _ in checkpoints[save_total_limit:]]


def create_training_progress(
    current_step: int,
    total_steps: int,
    current_epoch: int,
    total_epochs: int,
    loss: float | None = None,
    learning_rate: float | None = None,
) -> TrainingProgress:
    """Create a training progress snapshot.

    Args:
        current_step: Current training step.
        total_steps: Total training steps.
        current_epoch: Current epoch number.
        total_epochs: Total number of epochs.
        loss: Current loss value. Defaults to None.
        learning_rate: Current learning rate. Defaults to None.

    Returns:
        TrainingProgress instance.

    Raises:
        ValueError: If current_step is negative.
        ValueError: If total_steps is negative.
        ValueError: If current_epoch is negative.
        ValueError: If total_epochs is not positive.

    Examples:
        >>> progress = create_training_progress(50, 100, 1, 3)
        >>> progress.percent_complete
        50.0
        >>> progress.steps_remaining
        50

        >>> create_training_progress(-1, 100, 1, 3)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: current_step cannot be negative
    """
    if current_step < 0:
        msg = "current_step cannot be negative"
        raise ValueError(msg)

    if total_steps < 0:
        msg = "total_steps cannot be negative"
        raise ValueError(msg)

    if current_epoch < 0:
        msg = "current_epoch cannot be negative"
        raise ValueError(msg)

    if total_epochs <= 0:
        msg = "total_epochs must be positive"
        raise ValueError(msg)

    return TrainingProgress(
        current_step=current_step,
        total_steps=total_steps,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        loss=loss,
        learning_rate=learning_rate,
    )


def format_training_progress(progress: TrainingProgress) -> str:
    """Format training progress for display.

    Args:
        progress: Training progress to format.

    Returns:
        Formatted progress string.

    Raises:
        ValueError: If progress is None.

    Examples:
        >>> progress = TrainingProgress(
        ...     current_step=50,
        ...     total_steps=100,
        ...     current_epoch=1,
        ...     total_epochs=3,
        ...     loss=0.5,
        ... )
        >>> formatted = format_training_progress(progress)
        >>> "50/100" in formatted
        True
        >>> "50.0%" in formatted
        True

        >>> format_training_progress(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: progress cannot be None
    """
    if progress is None:
        msg = "progress cannot be None"
        raise ValueError(msg)

    parts = [
        f"Step {progress.current_step}/{progress.total_steps}",
        f"({progress.percent_complete:.1f}%)",
        f"Epoch {progress.current_epoch}/{progress.total_epochs}",
    ]

    if progress.loss is not None:
        parts.append(f"Loss: {progress.loss:.4f}")

    if progress.learning_rate is not None:
        parts.append(f"LR: {progress.learning_rate:.2e}")

    return " | ".join(parts)


def list_scheduler_types() -> list[str]:
    """List available scheduler types.

    Returns:
        Sorted list of scheduler type names.

    Examples:
        >>> types = list_scheduler_types()
        >>> "linear" in types
        True
        >>> "cosine" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(s.value for s in SchedulerType)


def validate_scheduler_type(scheduler_type: str) -> bool:
    """Check if a scheduler type is valid.

    Args:
        scheduler_type: Scheduler type to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_scheduler_type("linear")
        True
        >>> validate_scheduler_type("cosine")
        True
        >>> validate_scheduler_type("invalid")
        False
        >>> validate_scheduler_type("")
        False
    """
    valid_types = {s.value for s in SchedulerType}
    return scheduler_type in valid_types


def get_scheduler_type(name: str) -> SchedulerType:
    """Get scheduler type enum from string name.

    Args:
        name: Scheduler type name.

    Returns:
        SchedulerType enum value.

    Raises:
        ValueError: If name is not a valid scheduler type.

    Examples:
        >>> get_scheduler_type("linear")
        <SchedulerType.LINEAR: 'linear'>
        >>> get_scheduler_type("cosine")
        <SchedulerType.COSINE: 'cosine'>

        >>> get_scheduler_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid scheduler type: invalid
    """
    for stype in SchedulerType:
        if stype.value == name:
            return stype
    msg = f"invalid scheduler type: {name}"
    raise ValueError(msg)


def list_early_stopping_modes() -> list[str]:
    """List available early stopping modes.

    Returns:
        Sorted list of early stopping mode names.

    Examples:
        >>> modes = list_early_stopping_modes()
        >>> "min" in modes
        True
        >>> "max" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(m.value for m in EarlyStoppingMode)


def validate_early_stopping_mode(mode: str) -> bool:
    """Check if an early stopping mode is valid.

    Args:
        mode: Mode to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_early_stopping_mode("min")
        True
        >>> validate_early_stopping_mode("max")
        True
        >>> validate_early_stopping_mode("invalid")
        False
        >>> validate_early_stopping_mode("")
        False
    """
    valid_modes = {m.value for m in EarlyStoppingMode}
    return mode in valid_modes


def get_early_stopping_mode(name: str) -> EarlyStoppingMode:
    """Get early stopping mode enum from string name.

    Args:
        name: Mode name.

    Returns:
        EarlyStoppingMode enum value.

    Raises:
        ValueError: If name is not a valid mode.

    Examples:
        >>> get_early_stopping_mode("min")
        <EarlyStoppingMode.MIN: 'min'>
        >>> get_early_stopping_mode("max")
        <EarlyStoppingMode.MAX: 'max'>

        >>> get_early_stopping_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid early stopping mode: invalid
    """
    for mode in EarlyStoppingMode:
        if mode.value == name:
            return mode
    msg = f"invalid early stopping mode: {name}"
    raise ValueError(msg)
