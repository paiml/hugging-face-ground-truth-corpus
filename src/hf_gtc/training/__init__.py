"""Training recipes for HuggingFace models.

This module provides utilities for model training using
the HuggingFace Trainer API and PEFT for parameter-efficient fine-tuning.
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

__all__: list[str] = [
    "CallbackMetrics",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "LoRAConfig",
    "LoggingConfig",
    "MetricMode",
    "TaskType",
    "TrainingConfig",
    "calculate_lora_memory_savings",
    "compute_num_training_steps",
    "create_early_stopping_callback",
    "create_logging_callback",
    "create_lora_config",
    "create_trainer",
    "create_training_args",
    "estimate_lora_parameters",
    "get_peft_config",
    "get_recommended_callbacks",
    "get_recommended_lora_config",
    "list_callback_types",
    "list_task_types",
    "should_stop_early",
    "validate_checkpoint_config",
    "validate_early_stopping_config",
    "validate_logging_config",
    "validate_training_config",
]
