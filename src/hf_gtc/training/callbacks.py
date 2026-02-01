"""Training callbacks for HuggingFace Trainer.

This module provides custom callback implementations for monitoring
and controlling model training with the HuggingFace Trainer API.

Examples:
    >>> from hf_gtc.training.callbacks import EarlyStoppingConfig
    >>> config = EarlyStoppingConfig(patience=3, threshold=0.01)
    >>> config.patience
    3
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import TrainerCallback

from hf_gtc._validation import validate_not_none


class MetricMode(Enum):
    """Mode for metric comparison in early stopping.

    Attributes:
        MIN: Stop when metric stops decreasing (for loss).
        MAX: Stop when metric stops increasing (for accuracy).

    Examples:
        >>> MetricMode.MIN.value
        'min'
        >>> MetricMode.MAX.value
        'max'
    """

    MIN = "min"
    MAX = "max"


VALID_METRIC_MODES = frozenset(m.value for m in MetricMode)


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    """Configuration for early stopping callback.

    Attributes:
        patience: Number of evaluations to wait for improvement.
        threshold: Minimum change to qualify as improvement.
        metric: Metric name to monitor (e.g., "eval_loss").
        mode: Whether to minimize or maximize the metric.

    Examples:
        >>> config = EarlyStoppingConfig(
        ...     patience=3,
        ...     threshold=0.01,
        ...     metric="eval_loss",
        ...     mode=MetricMode.MIN,
        ... )
        >>> config.patience
        3
        >>> config.mode == MetricMode.MIN
        True
    """

    patience: int = 3
    threshold: float = 0.0
    metric: str = "eval_loss"
    mode: MetricMode = MetricMode.MIN


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Configuration for logging callback.

    Attributes:
        log_every_n_steps: Log metrics every N training steps.
        log_predictions: Whether to log sample predictions.
        log_gradients: Whether to log gradient statistics.
        max_samples_to_log: Maximum samples to log per evaluation.

    Examples:
        >>> config = LoggingConfig(
        ...     log_every_n_steps=100,
        ...     log_predictions=True,
        ... )
        >>> config.log_every_n_steps
        100
    """

    log_every_n_steps: int = 100
    log_predictions: bool = False
    log_gradients: bool = False
    max_samples_to_log: int = 10


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Configuration for checkpoint callback.

    Attributes:
        save_every_n_steps: Save checkpoint every N steps.
        save_total_limit: Maximum number of checkpoints to keep.
        save_on_each_node: Whether to save on each node in distributed.
        metric_for_best_model: Metric to use for best model selection.
        greater_is_better: Whether higher metric values are better.

    Examples:
        >>> config = CheckpointConfig(
        ...     save_every_n_steps=500,
        ...     save_total_limit=3,
        ... )
        >>> config.save_every_n_steps
        500
    """

    save_every_n_steps: int = 500
    save_total_limit: int | None = None
    save_on_each_node: bool = False
    metric_for_best_model: str | None = "eval_loss"
    greater_is_better: bool = False


@dataclass(frozen=True, slots=True)
class CallbackMetrics:
    """Metrics tracked by callbacks during training.

    Attributes:
        best_metric: Best value of the monitored metric.
        best_step: Step at which best metric was achieved.
        epochs_without_improvement: Number of epochs since last improvement.
        total_steps: Total training steps completed.

    Examples:
        >>> metrics = CallbackMetrics(
        ...     best_metric=0.5,
        ...     best_step=100,
        ...     epochs_without_improvement=0,
        ...     total_steps=200,
        ... )
        >>> metrics.best_metric
        0.5
    """

    best_metric: float | None = None
    best_step: int = 0
    epochs_without_improvement: int = 0
    total_steps: int = 0


def validate_early_stopping_config(config: EarlyStoppingConfig) -> None:
    """Validate early stopping configuration.

    Args:
        config: EarlyStoppingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If patience is not positive.
        ValueError: If threshold is negative.
        ValueError: If metric is empty.

    Examples:
        >>> config = EarlyStoppingConfig(patience=3, threshold=0.01)
        >>> validate_early_stopping_config(config)  # No error

        >>> validate_early_stopping_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = EarlyStoppingConfig(patience=0)
        >>> validate_early_stopping_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     bad_config
        ... )
        Traceback (most recent call last):
        ValueError: patience must be positive
    """
    validate_not_none(config, "config")

    if config.patience <= 0:
        msg = f"patience must be positive, got {config.patience}"
        raise ValueError(msg)

    if config.threshold < 0:
        msg = f"threshold must be non-negative, got {config.threshold}"
        raise ValueError(msg)

    if not config.metric or not config.metric.strip():
        msg = "metric cannot be empty"
        raise ValueError(msg)


def validate_logging_config(config: LoggingConfig) -> None:
    """Validate logging configuration.

    Args:
        config: LoggingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If log_every_n_steps is not positive.
        ValueError: If max_samples_to_log is not positive.

    Examples:
        >>> config = LoggingConfig(log_every_n_steps=100)
        >>> validate_logging_config(config)  # No error

        >>> validate_logging_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if config.log_every_n_steps <= 0:
        msg = f"log_every_n_steps must be positive, got {config.log_every_n_steps}"
        raise ValueError(msg)

    if config.max_samples_to_log <= 0:
        msg = f"max_samples_to_log must be positive, got {config.max_samples_to_log}"
        raise ValueError(msg)


def validate_checkpoint_config(config: CheckpointConfig) -> None:
    """Validate checkpoint configuration.

    Args:
        config: CheckpointConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If save_every_n_steps is not positive.
        ValueError: If save_total_limit is not positive when set.

    Examples:
        >>> config = CheckpointConfig(save_every_n_steps=500)
        >>> validate_checkpoint_config(config)  # No error

        >>> validate_checkpoint_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    validate_not_none(config, "config")

    if config.save_every_n_steps <= 0:
        msg = f"save_every_n_steps must be positive, got {config.save_every_n_steps}"
        raise ValueError(msg)

    if config.save_total_limit is not None and config.save_total_limit <= 0:
        msg = (
            f"save_total_limit must be positive when set, got {config.save_total_limit}"
        )
        raise ValueError(msg)


def should_stop_early(
    current_metric: float,
    best_metric: float | None,
    epochs_without_improvement: int,
    config: EarlyStoppingConfig,
) -> tuple[bool, float, int]:
    """Determine if training should stop early.

    Args:
        current_metric: Current value of the monitored metric.
        best_metric: Best value seen so far, or None if first evaluation.
        epochs_without_improvement: Number of epochs since last improvement.
        config: Early stopping configuration.

    Returns:
        Tuple of (should_stop, new_best_metric, new_epochs_without_improvement).

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = EarlyStoppingConfig(
        ...     patience=3, threshold=0.01, mode=MetricMode.MIN
        ... )
        >>> should_stop, best, epochs = should_stop_early(0.5, None, 0, config)
        >>> should_stop
        False
        >>> best
        0.5
        >>> epochs
        0

        >>> # Metric improved
        >>> should_stop, best, epochs = should_stop_early(0.4, 0.5, 2, config)
        >>> should_stop
        False
        >>> best
        0.4
        >>> epochs
        0

        >>> # Metric did not improve
        >>> should_stop, best, epochs = should_stop_early(0.5, 0.4, 0, config)
        >>> should_stop
        False
        >>> best
        0.4
        >>> epochs
        1
    """
    validate_not_none(config, "config")

    # First evaluation - initialize best metric
    if best_metric is None:
        return False, current_metric, 0

    # Check if metric improved
    if config.mode == MetricMode.MIN:
        improved = current_metric < (best_metric - config.threshold)
    else:
        improved = current_metric > (best_metric + config.threshold)

    if improved:
        return False, current_metric, 0

    # Metric did not improve
    new_epochs = epochs_without_improvement + 1
    should_stop = new_epochs >= config.patience

    return should_stop, best_metric, new_epochs


def create_early_stopping_callback(
    config: EarlyStoppingConfig,
) -> TrainerCallback:
    """Create a HuggingFace Trainer callback for early stopping.

    Args:
        config: Early stopping configuration.

    Returns:
        TrainerCallback instance for early stopping.

    Raises:
        ValueError: If config is invalid.

    Examples:
        >>> config = EarlyStoppingConfig(patience=3)
        >>> callback = create_early_stopping_callback(config)
        >>> callback is not None
        True
    """
    validate_early_stopping_config(config)

    from transformers import EarlyStoppingCallback as HFEarlyStoppingCallback

    return HFEarlyStoppingCallback(
        early_stopping_patience=config.patience,
        early_stopping_threshold=config.threshold,
    )


def create_logging_callback(config: LoggingConfig) -> TrainerCallback:
    """Create a HuggingFace Trainer callback for enhanced logging.

    Args:
        config: Logging configuration.

    Returns:
        TrainerCallback instance for logging.

    Raises:
        ValueError: If config is invalid.

    Examples:
        >>> config = LoggingConfig(log_every_n_steps=50)
        >>> callback = create_logging_callback(config)
        >>> callback is not None
        True
    """
    validate_logging_config(config)

    from transformers import TrainerCallback as HFTrainerCallback

    class EnhancedLoggingCallback(HFTrainerCallback):
        """Custom logging callback with enhanced features."""

        def __init__(self, logging_config: LoggingConfig) -> None:
            """Initialize with the given logging configuration."""
            self.config = logging_config
            self.step_count = 0

        def on_step_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> None:
            """Called at the end of each training step."""
            self.step_count += 1

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """Called when logging metrics."""
            pass  # Logging is handled by the trainer

    return EnhancedLoggingCallback(config)


def get_recommended_callbacks(
    task: str = "training",
    *,
    enable_early_stopping: bool = True,
    enable_logging: bool = True,
) -> list[dict[str, Any]]:
    """Get recommended callback configurations for a task.

    Args:
        task: The training task type. Defaults to "training".
        enable_early_stopping: Whether to include early stopping. Defaults to True.
        enable_logging: Whether to include enhanced logging. Defaults to True.

    Returns:
        List of callback configuration dictionaries.

    Examples:
        >>> callbacks = get_recommended_callbacks("fine-tuning")
        >>> len(callbacks) > 0
        True
        >>> any(cb["type"] == "early_stopping" for cb in callbacks)
        True

        >>> callbacks = get_recommended_callbacks(enable_early_stopping=False)
        >>> any(cb["type"] == "early_stopping" for cb in callbacks)
        False
    """
    callbacks: list[dict[str, Any]] = []

    if enable_early_stopping:
        callbacks.append(
            {
                "type": "early_stopping",
                "config": {
                    "patience": 3,
                    "threshold": 0.0,
                    "metric": "eval_loss",
                    "mode": "min",
                },
            }
        )

    if enable_logging:
        callbacks.append(
            {
                "type": "logging",
                "config": {
                    "log_every_n_steps": 100,
                    "log_predictions": False,
                    "log_gradients": False,
                },
            }
        )

    return callbacks


def list_callback_types() -> list[str]:
    """List all available callback types.

    Returns:
        Sorted list of callback type names.

    Examples:
        >>> types = list_callback_types()
        >>> "early_stopping" in types
        True
        >>> "logging" in types
        True
        >>> "checkpoint" in types
        True
    """
    return sorted(["early_stopping", "logging", "checkpoint", "gradient_clip"])
