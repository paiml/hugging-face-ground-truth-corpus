"""LoRA (Low-Rank Adaptation) fine-tuning utilities.

This module provides functions for configuring and applying LoRA
adapters to HuggingFace models using the PEFT library.

Examples:
    >>> from hf_gtc.training.lora import create_lora_config
    >>> config = create_lora_config(r=8, lora_alpha=16)
    >>> config.r
    8
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

# Strict type for bias configuration matching PEFT's Literal expectation
BiasType = Literal["none", "all", "lora_only"]


class TaskType(Enum):
    """Supported task types for LoRA.

    Attributes:
        CAUSAL_LM: Causal language modeling (text generation).
        SEQ_CLS: Sequence classification.
        SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
        TOKEN_CLS: Token classification (NER, POS tagging).
        QUESTION_ANS: Question answering.
        FEATURE_EXTRACTION: Feature extraction.

    Examples:
        >>> TaskType.CAUSAL_LM.value
        'CAUSAL_LM'
        >>> TaskType.SEQ_CLS.value
        'SEQ_CLS'
    """

    CAUSAL_LM = "CAUSAL_LM"
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


VALID_TASK_TYPES = frozenset(t.value for t in TaskType)

# Default target modules for common architectures
DEFAULT_TARGET_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        r: Rank of the low-rank matrices.
        lora_alpha: Scaling factor for LoRA weights.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: Modules to apply LoRA to.
        task_type: Type of task for the model.
        bias: Bias configuration ("none", "all", "lora_only").
        modules_to_save: Additional modules to train fully.

    Examples:
        >>> config = LoRAConfig(
        ...     r=8,
        ...     lora_alpha=16,
        ...     lora_dropout=0.1,
        ...     target_modules=("q_proj", "v_proj"),
        ...     task_type=TaskType.CAUSAL_LM,
        ...     bias="none",
        ...     modules_to_save=None,
        ... )
        >>> config.r
        8
    """

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    task_type: TaskType
    bias: BiasType
    modules_to_save: tuple[str, ...] | None


VALID_BIAS_OPTIONS = frozenset({"none", "all", "lora_only"})


def validate_lora_config(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    bias: BiasType,
) -> None:
    """Validate LoRA configuration parameters.

    Args:
        r: Rank of low-rank matrices.
        lora_alpha: Scaling factor.
        lora_dropout: Dropout probability.
        bias: Bias configuration.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> validate_lora_config(8, 16, 0.1, "none")  # No error

        >>> validate_lora_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     0, 16, 0.1, "none"
        ... )
        Traceback (most recent call last):
        ValueError: r must be positive
    """
    if r <= 0:
        msg = f"r must be positive, got {r}"
        raise ValueError(msg)

    if lora_alpha <= 0:
        msg = f"lora_alpha must be positive, got {lora_alpha}"
        raise ValueError(msg)

    if not 0.0 <= lora_dropout < 1.0:
        msg = f"lora_dropout must be in [0.0, 1.0), got {lora_dropout}"
        raise ValueError(msg)

    if bias not in VALID_BIAS_OPTIONS:
        msg = f"bias must be one of {VALID_BIAS_OPTIONS}, got '{bias}'"
        raise ValueError(msg)


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: tuple[str, ...] | None = None,
    task_type: str = "CAUSAL_LM",
    bias: BiasType = "none",
    modules_to_save: tuple[str, ...] | None = None,
) -> LoRAConfig:
    """Create a LoRA configuration.

    Args:
        r: Rank of low-rank matrices. Defaults to 8.
        lora_alpha: Scaling factor. Defaults to 16.
        lora_dropout: Dropout probability. Defaults to 0.1.
        target_modules: Modules to apply LoRA to. Defaults to common attention modules.
        task_type: Task type. Defaults to "CAUSAL_LM".
        bias: Bias configuration. Defaults to "none".
        modules_to_save: Additional modules to train. Defaults to None.

    Returns:
        LoRAConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_lora_config(r=16, lora_alpha=32)
        >>> config.r
        16
        >>> config.lora_alpha
        32

        >>> config = create_lora_config(target_modules=("q_proj", "v_proj"))
        >>> "q_proj" in config.target_modules
        True

        >>> create_lora_config(r=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: r must be positive
    """
    validate_lora_config(r, lora_alpha, lora_dropout, bias)

    if task_type not in VALID_TASK_TYPES:
        msg = f"task_type must be one of {VALID_TASK_TYPES}, got '{task_type}'"
        raise ValueError(msg)

    if target_modules is None:
        target_modules = tuple(sorted(DEFAULT_TARGET_MODULES))

    return LoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType(task_type),
        bias=bias,
        modules_to_save=modules_to_save,
    )


def get_peft_config(lora_config: LoRAConfig) -> Any:
    """Convert LoRAConfig to PEFT LoraConfig.

    Args:
        lora_config: Our LoRA configuration.

    Returns:
        PEFT LoraConfig instance.

    Examples:
        >>> config = create_lora_config()
        >>> peft_config = get_peft_config(config)
        >>> peft_config.r
        8
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType as PeftTaskType

    task_type_map = {
        TaskType.CAUSAL_LM: PeftTaskType.CAUSAL_LM,
        TaskType.SEQ_CLS: PeftTaskType.SEQ_CLS,
        TaskType.SEQ_2_SEQ_LM: PeftTaskType.SEQ_2_SEQ_LM,
        TaskType.TOKEN_CLS: PeftTaskType.TOKEN_CLS,
        TaskType.QUESTION_ANS: PeftTaskType.QUESTION_ANS,
        TaskType.FEATURE_EXTRACTION: PeftTaskType.FEATURE_EXTRACTION,
    }

    modules_to_save = (
        list(lora_config.modules_to_save) if lora_config.modules_to_save else None
    )
    return PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=list(lora_config.target_modules),
        task_type=task_type_map[lora_config.task_type],
        bias=lora_config.bias,
        modules_to_save=modules_to_save,
    )


def estimate_lora_parameters(
    base_model_params: int,
    r: int,
    num_target_modules: int,
    hidden_size: int = 4096,
) -> int:
    """Estimate number of trainable LoRA parameters.

    Args:
        base_model_params: Number of parameters in base model.
        r: LoRA rank.
        num_target_modules: Number of modules with LoRA applied.
        hidden_size: Hidden dimension of the model. Defaults to 4096.

    Returns:
        Estimated number of trainable parameters.

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> estimate_lora_parameters(7_000_000_000, r=8, num_target_modules=7)
        458752
        >>> estimate_lora_parameters(7_000_000_000, r=16, num_target_modules=7)
        917504

        >>> estimate_lora_parameters(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     0, r=8, num_target_modules=7
        ... )
        Traceback (most recent call last):
        ValueError: base_model_params must be positive
    """
    if base_model_params <= 0:
        msg = f"base_model_params must be positive, got {base_model_params}"
        raise ValueError(msg)

    if r <= 0:
        msg = f"r must be positive, got {r}"
        raise ValueError(msg)

    if num_target_modules <= 0:
        msg = f"num_target_modules must be positive, got {num_target_modules}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    # Each LoRA layer adds 2 * hidden_size * r parameters (A and B matrices)
    params_per_module = 2 * hidden_size * r
    return params_per_module * num_target_modules


def calculate_lora_memory_savings(
    base_model_params: int,
    lora_params: int,
) -> float:
    """Calculate memory savings from using LoRA.

    Args:
        base_model_params: Number of parameters in base model.
        lora_params: Number of LoRA parameters.

    Returns:
        Percentage of parameters that are trainable (lower = more savings).

    Raises:
        ValueError: If any parameter is not positive.

    Examples:
        >>> savings = calculate_lora_memory_savings(7_000_000_000, 458752)
        >>> savings < 0.01  # Less than 1% of parameters
        True

        >>> calculate_lora_memory_savings(0, 1000)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: base_model_params must be positive
    """
    if base_model_params <= 0:
        msg = f"base_model_params must be positive, got {base_model_params}"
        raise ValueError(msg)

    if lora_params <= 0:
        msg = f"lora_params must be positive, got {lora_params}"
        raise ValueError(msg)

    return (lora_params / base_model_params) * 100


def list_task_types() -> list[str]:
    """List all supported task types.

    Returns:
        Sorted list of task type names.

    Examples:
        >>> types = list_task_types()
        >>> "CAUSAL_LM" in types
        True
        >>> "SEQ_CLS" in types
        True
    """
    return sorted(VALID_TASK_TYPES)


def get_recommended_lora_config(
    model_size: str,
    task_type: str = "CAUSAL_LM",
) -> LoRAConfig:
    """Get recommended LoRA configuration based on model size.

    Args:
        model_size: Model size category ("small", "medium", "large", "xlarge").
        task_type: Task type. Defaults to "CAUSAL_LM".

    Returns:
        Recommended LoRAConfig for the model size.

    Raises:
        ValueError: If model_size is not valid.

    Examples:
        >>> config = get_recommended_lora_config("small")
        >>> config.r
        4

        >>> config = get_recommended_lora_config("large")
        >>> config.r
        16

        >>> get_recommended_lora_config("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_size must be one of...
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    # Recommended configurations based on model size
    configs = {
        "small": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.05},
        "medium": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
        "large": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
        "xlarge": {"r": 32, "lora_alpha": 64, "lora_dropout": 0.05},
    }

    cfg = configs[model_size]
    return create_lora_config(
        r=int(cfg["r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        task_type=task_type,
    )
