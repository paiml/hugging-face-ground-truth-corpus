"""Training recipes for HuggingFace models.

This module provides utilities for model training using
the HuggingFace Trainer API and PEFT for parameter-efficient fine-tuning.
"""

from __future__ import annotations

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
    "LoRAConfig",
    "TaskType",
    "TrainingConfig",
    "calculate_lora_memory_savings",
    "compute_num_training_steps",
    "create_lora_config",
    "create_trainer",
    "create_training_args",
    "estimate_lora_parameters",
    "get_peft_config",
    "get_recommended_lora_config",
    "list_task_types",
    "validate_training_config",
]
