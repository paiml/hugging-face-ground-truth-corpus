"""Fine-tuning utilities for HuggingFace models.

This module provides functions for configuring and executing
fine-tuning of transformer models using the HuggingFace Trainer API.

Examples:
    >>> from hf_gtc.training.fine_tuning import create_training_args
    >>> args = create_training_args(output_dir="/tmp/test")
    >>> args.output_dir
    '/tmp/test'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformers import Trainer as _HFTrainer
from transformers import TrainingArguments as _TrainingArguments

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for model fine-tuning.

    Attributes:
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Training batch size per device.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay coefficient.
        warmup_ratio: Ratio of warmup steps.
        eval_strategy: Evaluation strategy ("epoch", "steps", "no").
        save_strategy: Save strategy ("epoch", "steps", "no").
        logging_steps: Number of steps between logging.
        fp16: Whether to use mixed precision training.

    Examples:
        >>> config = TrainingConfig(output_dir="/tmp/model")
        >>> config.num_epochs
        3
        >>> config.learning_rate
        5e-05
    """

    output_dir: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 100
    fp16: bool = False


VALID_EVAL_STRATEGIES = frozenset({"epoch", "steps", "no"})
VALID_SAVE_STRATEGIES = frozenset({"epoch", "steps", "no"})


def validate_training_config(config: TrainingConfig) -> None:
    """Validate training configuration parameters.

    Args:
        config: Training configuration to validate.

    Raises:
        ValueError: If any configuration parameter is invalid.

    Examples:
        >>> config = TrainingConfig(output_dir="/tmp/test")
        >>> validate_training_config(config)  # No error

        >>> bad_config = TrainingConfig(output_dir="", num_epochs=0)
        >>> validate_training_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: output_dir cannot be empty
    """
    if not config.output_dir:
        msg = "output_dir cannot be empty"
        raise ValueError(msg)

    if config.num_epochs <= 0:
        msg = f"num_epochs must be positive, got {config.num_epochs}"
        raise ValueError(msg)

    if config.batch_size <= 0:
        msg = f"batch_size must be positive, got {config.batch_size}"
        raise ValueError(msg)

    if config.learning_rate <= 0:
        msg = f"learning_rate must be positive, got {config.learning_rate}"
        raise ValueError(msg)

    if config.weight_decay < 0:
        msg = f"weight_decay cannot be negative, got {config.weight_decay}"
        raise ValueError(msg)

    if not 0 <= config.warmup_ratio <= 1:
        msg = f"warmup_ratio must be between 0 and 1, got {config.warmup_ratio}"
        raise ValueError(msg)

    if config.eval_strategy not in VALID_EVAL_STRATEGIES:
        msg = f"eval_strategy must be one of {VALID_EVAL_STRATEGIES}"
        raise ValueError(msg)

    if config.save_strategy not in VALID_SAVE_STRATEGIES:
        msg = f"save_strategy must be one of {VALID_SAVE_STRATEGIES}"
        raise ValueError(msg)

    if config.logging_steps <= 0:
        msg = f"logging_steps must be positive, got {config.logging_steps}"
        raise ValueError(msg)


def create_training_args(
    output_dir: str,
    *,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 100,
    fp16: bool = False,
) -> Any:
    """Create TrainingArguments for HuggingFace Trainer.

    Args:
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs. Defaults to 3.
        batch_size: Training batch size per device. Defaults to 8.
        learning_rate: Initial learning rate. Defaults to 5e-5.
        weight_decay: Weight decay coefficient. Defaults to 0.01.
        warmup_ratio: Ratio of warmup steps. Defaults to 0.1.
        eval_strategy: Evaluation strategy. Defaults to "epoch".
        save_strategy: Save strategy. Defaults to "epoch".
        logging_steps: Steps between logging. Defaults to 100.
        fp16: Whether to use FP16 training. Defaults to False.

    Returns:
        TrainingArguments instance configured with the parameters.

    Raises:
        ValueError: If output_dir is empty.
        ValueError: If num_epochs is not positive.
        ValueError: If batch_size is not positive.

    Examples:
        >>> args = create_training_args("/tmp/model")
        >>> args.output_dir
        '/tmp/model'
        >>> args.num_train_epochs
        3

        >>> create_training_args("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: output_dir cannot be empty
    """
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        fp16=fp16,
    )
    validate_training_config(config)

    return _TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_steps=config.logging_steps,
        fp16=config.fp16,
        load_best_model_at_end=True,
        report_to="none",
    )


def create_trainer(
    model: PreTrainedModel,
    training_args: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    compute_metrics: Callable[[Any], dict[str, float]] | None = None,
) -> Trainer:
    """Create a Trainer instance for fine-tuning.

    Args:
        model: Pre-trained model to fine-tune.
        training_args: TrainingArguments instance.
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        tokenizer: Optional tokenizer for data collation.
        compute_metrics: Optional function to compute metrics.

    Returns:
        Configured Trainer instance.

    Raises:
        ValueError: If model is None.
        ValueError: If train_dataset is None.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> model = MagicMock()
        >>> args = MagicMock()
        >>> dataset = MagicMock()
        >>> trainer = create_trainer(model, args, dataset)
        >>> trainer is not None
        True
    """
    if model is None:
        msg = "model cannot be None"
        raise ValueError(msg)

    if train_dataset is None:
        msg = "train_dataset cannot be None"
        raise ValueError(msg)

    return _HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def compute_num_training_steps(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Compute total number of training steps.

    Args:
        num_samples: Number of training samples.
        batch_size: Batch size per device.
        num_epochs: Number of training epochs.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Total number of training steps.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> compute_num_training_steps(1000, 8, 3)
        375
        >>> compute_num_training_steps(1000, 8, 3, gradient_accumulation_steps=2)
        186
        >>> compute_num_training_steps(100, 32, 1)
        3

        >>> compute_num_training_steps(0, 8, 3)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_samples must be positive, got 0
    """
    if num_samples <= 0:
        msg = f"num_samples must be positive, got {num_samples}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if num_epochs <= 0:
        msg = f"num_epochs must be positive, got {num_epochs}"
        raise ValueError(msg)

    if gradient_accumulation_steps <= 0:
        msg = f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}"
        raise ValueError(msg)

    steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
    return steps_per_epoch * num_epochs
