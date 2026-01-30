"""Training recipes for HuggingFace models.

This module provides utilities for model training using
the HuggingFace Trainer API and PEFT for parameter-efficient fine-tuning.

Examples:
    >>> from hf_gtc.training import TrainingConfig, TrainerState
    >>> config = TrainingConfig(output_dir="/tmp/test")
    >>> config.num_epochs
    3
    >>> state = TrainerState(global_step=100, epoch=1.5)
    >>> state.global_step
    100
"""

from __future__ import annotations

from hf_gtc.training.callbacks import (
    CallbackMetrics,
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    MetricMode,
    create_early_stopping_callback,
    create_logging_callback,
    get_recommended_callbacks,
    list_callback_types,
    should_stop_early,
    validate_checkpoint_config,
    validate_early_stopping_config,
    validate_logging_config,
)
from hf_gtc.training.fine_tuning import (
    TrainingConfig,
    compute_num_training_steps,
    create_trainer,
    create_training_args,
    validate_training_config,
)
from hf_gtc.training.lora import (
    LoRAConfig,
    TaskType,
    calculate_lora_memory_savings,
    create_lora_config,
    estimate_lora_parameters,
    get_peft_config,
    get_recommended_lora_config,
    list_task_types,
)
from hf_gtc.training.trainer import (
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
    update_trainer_state,
    validate_scheduler_config,
    validate_scheduler_type,
    validate_trainer_state,
)

__all__: list[str] = [
    "CallbackMetrics",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "LoRAConfig",
    "LoggingConfig",
    "MetricMode",
    "SchedulerConfig",
    "SchedulerType",
    "TaskType",
    "TrainerState",
    "TrainingConfig",
    "TrainingProgress",
    "calculate_lora_memory_savings",
    "compute_num_training_steps",
    "compute_warmup_steps",
    "create_early_stopping_callback",
    "create_logging_callback",
    "create_lora_config",
    "create_trainer",
    "create_trainer_state",
    "create_training_args",
    "create_training_progress",
    "estimate_lora_parameters",
    "format_training_progress",
    "get_checkpoint_path",
    "get_checkpoints_to_delete",
    "get_early_stopping_mode",
    "get_latest_checkpoint",
    "get_peft_config",
    "get_recommended_callbacks",
    "get_recommended_lora_config",
    "get_scheduler_type",
    "is_metric_improved",
    "list_callback_types",
    "list_checkpoints",
    "list_early_stopping_modes",
    "list_scheduler_types",
    "list_task_types",
    "should_stop_early",
    "update_trainer_state",
    "validate_checkpoint_config",
    "validate_early_stopping_config",
    "validate_logging_config",
    "validate_scheduler_config",
    "validate_scheduler_type",
    "validate_trainer_state",
    "validate_training_config",
]
