"""QLoRA (Quantized Low-Rank Adaptation) fine-tuning utilities.

This module provides functions for configuring and applying QLoRA
adapters to HuggingFace models using 4-bit quantization with the
bitsandbytes library.

Examples:
    >>> from hf_gtc.training.qlora import QLoRAConfig, QuantConfig
    >>> config = QLoRAConfig(r=8, lora_alpha=16)
    >>> config.r
    8
    >>> quant = QuantConfig(bits=4)
    >>> quant.bits
    4
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class QuantBits(Enum):
    """Supported quantization bit widths.

    Examples:
        >>> QuantBits.FOUR.value
        4
        >>> QuantBits.EIGHT.value
        8
    """

    FOUR = 4
    EIGHT = 8


class ComputeType(Enum):
    """Compute dtype for quantized operations.

    Examples:
        >>> ComputeType.FLOAT16.value
        'float16'
        >>> ComputeType.BFLOAT16.value
        'bfloat16'
    """

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class QuantType(Enum):
    """Quantization type/method.

    Examples:
        >>> QuantType.NF4.value
        'nf4'
        >>> QuantType.FP4.value
        'fp4'
    """

    NF4 = "nf4"
    FP4 = "fp4"


# Default target modules for QLoRA (same as LoRA)
DEFAULT_QLORA_TARGET_MODULES = frozenset(
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
class QuantConfig:
    """Configuration for model quantization.

    Attributes:
        bits: Number of bits for quantization (4 or 8).
        quant_type: Quantization type (nf4 or fp4).
        compute_dtype: Dtype for compute operations.
        double_quant: Whether to use double quantization.
        nested_quant: Whether to use nested quantization.

    Examples:
        >>> config = QuantConfig(bits=4)
        >>> config.bits
        4
        >>> config.quant_type
        <QuantType.NF4: 'nf4'>

        >>> config2 = QuantConfig(bits=8, quant_type=QuantType.FP4)
        >>> config2.bits
        8
    """

    bits: int = 4
    quant_type: QuantType = QuantType.NF4
    compute_dtype: ComputeType = ComputeType.FLOAT16
    double_quant: bool = True
    nested_quant: bool = False


@dataclass(frozen=True, slots=True)
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning.

    Attributes:
        r: Rank of the low-rank matrices.
        lora_alpha: Scaling factor for LoRA weights.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: Modules to apply LoRA to.
        bias: Bias configuration ("none", "all", "lora_only").
        modules_to_save: Additional modules to train fully.

    Examples:
        >>> config = QLoRAConfig(r=8, lora_alpha=16)
        >>> config.r
        8
        >>> config.lora_alpha
        16
        >>> config.lora_dropout
        0.05

        >>> config2 = QLoRAConfig(r=16, lora_dropout=0.1)
        >>> config2.r
        16
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: frozenset[str] = DEFAULT_QLORA_TARGET_MODULES
    bias: str = "none"
    modules_to_save: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class QLoRATrainingConfig:
    """Combined configuration for QLoRA training.

    Attributes:
        qlora_config: QLoRA adapter configuration.
        quant_config: Quantization configuration.
        gradient_checkpointing: Enable gradient checkpointing.
        gradient_accumulation_steps: Steps for gradient accumulation.
        max_grad_norm: Maximum gradient norm for clipping.

    Examples:
        >>> qlora = QLoRAConfig(r=8)
        >>> quant = QuantConfig(bits=4)
        >>> config = QLoRATrainingConfig(qlora_config=qlora, quant_config=quant)
        >>> config.qlora_config.r
        8
        >>> config.gradient_checkpointing
        True
    """

    qlora_config: QLoRAConfig
    quant_config: QuantConfig
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Memory usage estimate for QLoRA training.

    Attributes:
        model_memory_mb: Memory for model weights in MB.
        adapter_memory_mb: Memory for LoRA adapters in MB.
        optimizer_memory_mb: Memory for optimizer states in MB.
        activation_memory_mb: Memory for activations in MB.
        total_memory_mb: Total estimated memory in MB.

    Examples:
        >>> est = MemoryEstimate(
        ...     model_memory_mb=4000,
        ...     adapter_memory_mb=50,
        ...     optimizer_memory_mb=100,
        ...     activation_memory_mb=500,
        ...     total_memory_mb=4650,
        ... )
        >>> est.total_memory_mb
        4650
    """

    model_memory_mb: float
    adapter_memory_mb: float
    optimizer_memory_mb: float
    activation_memory_mb: float
    total_memory_mb: float


def validate_quant_config(config: QuantConfig) -> None:
    """Validate quantization configuration.

    Args:
        config: Quantization configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If bits is not 4 or 8.

    Examples:
        >>> config = QuantConfig(bits=4)
        >>> validate_quant_config(config)  # No error

        >>> validate_quant_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = QuantConfig(bits=3)
        >>> validate_quant_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be 4 or 8
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.bits not in (4, 8):
        msg = f"bits must be 4 or 8, got {config.bits}"
        raise ValueError(msg)


def validate_qlora_config(config: QLoRAConfig) -> None:
    """Validate QLoRA configuration.

    Args:
        config: QLoRA configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If r is not positive.
        ValueError: If lora_alpha is not positive.
        ValueError: If lora_dropout is not between 0 and 1.

    Examples:
        >>> config = QLoRAConfig(r=8, lora_alpha=16)
        >>> validate_qlora_config(config)  # No error

        >>> validate_qlora_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = QLoRAConfig(r=0)
        >>> validate_qlora_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: r must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.r <= 0:
        msg = f"r must be positive, got {config.r}"
        raise ValueError(msg)

    if config.lora_alpha <= 0:
        msg = f"lora_alpha must be positive, got {config.lora_alpha}"
        raise ValueError(msg)

    if not 0 <= config.lora_dropout <= 1:
        msg = f"lora_dropout must be between 0 and 1, got {config.lora_dropout}"
        raise ValueError(msg)

    valid_bias = {"none", "all", "lora_only"}
    if config.bias not in valid_bias:
        msg = f"bias must be one of {valid_bias}, got {config.bias}"
        raise ValueError(msg)


def create_quant_config(
    bits: int = 4,
    quant_type: QuantType | str = QuantType.NF4,
    compute_dtype: ComputeType | str = ComputeType.FLOAT16,
    double_quant: bool = True,
) -> QuantConfig:
    """Create a quantization configuration.

    Args:
        bits: Number of bits (4 or 8). Defaults to 4.
        quant_type: Quantization type. Defaults to NF4.
        compute_dtype: Compute dtype. Defaults to FLOAT16.
        double_quant: Enable double quantization. Defaults to True.

    Returns:
        Validated QuantConfig instance.

    Raises:
        ValueError: If bits is not 4 or 8.

    Examples:
        >>> config = create_quant_config(bits=4)
        >>> config.bits
        4
        >>> config.quant_type
        <QuantType.NF4: 'nf4'>

        >>> config2 = create_quant_config(bits=8, quant_type="fp4")
        >>> config2.bits
        8
    """
    if isinstance(quant_type, str):
        quant_type = get_quant_type(quant_type)
    if isinstance(compute_dtype, str):
        compute_dtype = get_compute_type(compute_dtype)

    config = QuantConfig(
        bits=bits,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        double_quant=double_quant,
    )
    validate_quant_config(config)
    return config


def create_qlora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
    bias: str = "none",
) -> QLoRAConfig:
    """Create a QLoRA configuration.

    Args:
        r: Rank of low-rank matrices. Defaults to 8.
        lora_alpha: Scaling factor. Defaults to 16.
        lora_dropout: Dropout probability. Defaults to 0.05.
        target_modules: Modules to apply LoRA to. Defaults to standard.
        bias: Bias configuration. Defaults to "none".

    Returns:
        Validated QLoRAConfig instance.

    Raises:
        ValueError: If r is not positive.

    Examples:
        >>> config = create_qlora_config(r=8, lora_alpha=16)
        >>> config.r
        8

        >>> config2 = create_qlora_config(r=16, lora_dropout=0.1)
        >>> config2.r
        16
    """
    modules = (
        frozenset(target_modules)
        if target_modules is not None
        else DEFAULT_QLORA_TARGET_MODULES
    )

    config = QLoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias=bias,
    )
    validate_qlora_config(config)
    return config


def create_qlora_training_config(
    qlora_config: QLoRAConfig | None = None,
    quant_config: QuantConfig | None = None,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 0.3,
) -> QLoRATrainingConfig:
    """Create a combined QLoRA training configuration.

    Args:
        qlora_config: QLoRA adapter config. Defaults to new instance.
        quant_config: Quantization config. Defaults to new instance.
        gradient_checkpointing: Enable checkpointing. Defaults to True.
        gradient_accumulation_steps: Accumulation steps. Defaults to 4.
        max_grad_norm: Max gradient norm. Defaults to 0.3.

    Returns:
        QLoRATrainingConfig instance.

    Examples:
        >>> config = create_qlora_training_config()
        >>> config.qlora_config.r
        8
        >>> config.quant_config.bits
        4

        >>> custom = create_qlora_training_config(
        ...     qlora_config=QLoRAConfig(r=16),
        ...     gradient_accumulation_steps=8,
        ... )
        >>> custom.gradient_accumulation_steps
        8
    """
    if qlora_config is None:
        qlora_config = create_qlora_config()
    if quant_config is None:
        quant_config = create_quant_config()

    return QLoRATrainingConfig(
        qlora_config=qlora_config,
        quant_config=quant_config,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
    )


def estimate_qlora_memory(
    model_params: int,
    quant_bits: int = 4,
    r: int = 8,
    batch_size: int = 1,
    sequence_length: int = 2048,
) -> MemoryEstimate:
    """Estimate memory requirements for QLoRA training.

    Args:
        model_params: Number of model parameters.
        quant_bits: Quantization bits. Defaults to 4.
        r: LoRA rank. Defaults to 8.
        batch_size: Training batch size. Defaults to 1.
        sequence_length: Maximum sequence length. Defaults to 2048.

    Returns:
        MemoryEstimate with memory breakdown.

    Raises:
        ValueError: If model_params is not positive.
        ValueError: If quant_bits is not 4 or 8.

    Examples:
        >>> est = estimate_qlora_memory(7_000_000_000, quant_bits=4, r=8)
        >>> est.model_memory_mb > 0
        True
        >>> est.total_memory_mb > est.model_memory_mb
        True

        >>> estimate_qlora_memory(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    if quant_bits not in (4, 8):
        msg = f"quant_bits must be 4 or 8, got {quant_bits}"
        raise ValueError(msg)

    # Model memory: params * bits / 8 bytes / 1024^2 MB
    model_memory_mb = (model_params * quant_bits) / (8 * 1024 * 1024)

    # Adapter memory: roughly 2 * r * hidden_dim * num_layers
    # Approximate as 0.5% of model params at rank 8
    adapter_ratio = (r / 8) * 0.005
    adapter_memory_mb = (model_params * adapter_ratio * 2) / (1024 * 1024)

    # Optimizer states: 2x adapter memory for Adam
    optimizer_memory_mb = adapter_memory_mb * 2

    # Activation memory estimate
    activation_memory_mb = (batch_size * sequence_length * 4096 * 4) / (1024 * 1024)

    total_memory_mb = (
        model_memory_mb + adapter_memory_mb + optimizer_memory_mb + activation_memory_mb
    )

    return MemoryEstimate(
        model_memory_mb=model_memory_mb,
        adapter_memory_mb=adapter_memory_mb,
        optimizer_memory_mb=optimizer_memory_mb,
        activation_memory_mb=activation_memory_mb,
        total_memory_mb=total_memory_mb,
    )


def get_qlora_peft_config(config: QLoRAConfig) -> dict[str, Any]:
    """Convert QLoRAConfig to PEFT-compatible dict.

    Args:
        config: QLoRA configuration.

    Returns:
        Dictionary for PEFT LoraConfig.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_qlora_config(r=8, lora_alpha=16)
        >>> peft_dict = get_qlora_peft_config(config)
        >>> peft_dict["r"]
        8
        >>> peft_dict["lora_alpha"]
        16

        >>> get_qlora_peft_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    return {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(config.target_modules),
        "bias": config.bias,
        "task_type": "CAUSAL_LM",
    }


def get_bnb_config(config: QuantConfig) -> dict[str, Any]:
    """Convert QuantConfig to BitsAndBytes config dict.

    Args:
        config: Quantization configuration.

    Returns:
        Dictionary for BitsAndBytesConfig.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_quant_config(bits=4)
        >>> bnb_dict = get_bnb_config(config)
        >>> bnb_dict["load_in_4bit"]
        True
        >>> bnb_dict["bnb_4bit_quant_type"]
        'nf4'

        >>> get_bnb_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    result: dict[str, Any] = {}

    if config.bits == 4:
        result["load_in_4bit"] = True
        result["bnb_4bit_quant_type"] = config.quant_type.value
        result["bnb_4bit_compute_dtype"] = config.compute_dtype.value
        result["bnb_4bit_use_double_quant"] = config.double_quant
    else:
        result["load_in_8bit"] = True

    return result


def calculate_qlora_trainable_params(
    model_params: int,
    r: int = 8,
    num_target_modules: int = 7,
) -> tuple[int, float]:
    """Calculate trainable parameters for QLoRA.

    Args:
        model_params: Total model parameters.
        r: LoRA rank. Defaults to 8.
        num_target_modules: Number of target modules. Defaults to 7.

    Returns:
        Tuple of (trainable_params, percentage).

    Raises:
        ValueError: If model_params is not positive.
        ValueError: If r is not positive.

    Examples:
        >>> params, pct = calculate_qlora_trainable_params(7_000_000_000)
        >>> params > 0
        True
        >>> 0 < pct < 1
        True

        >>> calculate_qlora_trainable_params(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    if r <= 0:
        msg = "r must be positive"
        raise ValueError(msg)

    # Approximate hidden dimension based on model size
    # This is a rough estimate; actual varies by architecture
    hidden_dim = int((model_params / 32) ** 0.5)

    # Each target module has r * hidden_dim * 2 trainable params
    trainable_params = num_target_modules * r * hidden_dim * 2

    percentage = trainable_params / model_params

    return trainable_params, percentage


def get_recommended_qlora_config(
    model_size: str = "7b",
    memory_constraint_gb: float | None = None,
) -> QLoRATrainingConfig:
    """Get recommended QLoRA configuration for model size.

    Args:
        model_size: Model size string (e.g., "7b", "13b", "70b").
        memory_constraint_gb: GPU memory limit in GB. Defaults to None.

    Returns:
        Recommended QLoRATrainingConfig.

    Raises:
        ValueError: If model_size is not recognized.

    Examples:
        >>> config = get_recommended_qlora_config("7b")
        >>> config.qlora_config.r
        8
        >>> config.quant_config.bits
        4

        >>> config_70b = get_recommended_qlora_config("70b")
        >>> config_70b.gradient_accumulation_steps >= 4
        True
    """
    model_size = model_size.lower().strip()

    if model_size in ("7b", "7B", "7"):
        return create_qlora_training_config(
            qlora_config=create_qlora_config(r=8, lora_alpha=16),
            quant_config=create_quant_config(bits=4),
            gradient_accumulation_steps=4,
        )
    elif model_size in ("13b", "13B", "13"):
        return create_qlora_training_config(
            qlora_config=create_qlora_config(r=8, lora_alpha=16),
            quant_config=create_quant_config(bits=4),
            gradient_accumulation_steps=8,
        )
    elif model_size in ("70b", "70B", "70"):
        return create_qlora_training_config(
            qlora_config=create_qlora_config(r=16, lora_alpha=32),
            quant_config=create_quant_config(bits=4, double_quant=True),
            gradient_accumulation_steps=16,
        )
    else:
        msg = f"unrecognized model size: {model_size}"
        raise ValueError(msg)


def list_quant_types() -> list[str]:
    """List available quantization types.

    Returns:
        Sorted list of quantization type names.

    Examples:
        >>> types = list_quant_types()
        >>> "nf4" in types
        True
        >>> "fp4" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(q.value for q in QuantType)


def validate_quant_type(quant_type: str) -> bool:
    """Check if a quantization type is valid.

    Args:
        quant_type: Type to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_quant_type("nf4")
        True
        >>> validate_quant_type("fp4")
        True
        >>> validate_quant_type("invalid")
        False
        >>> validate_quant_type("")
        False
    """
    valid_types = {q.value for q in QuantType}
    return quant_type in valid_types


def get_quant_type(name: str) -> QuantType:
    """Get quantization type enum from string.

    Args:
        name: Type name.

    Returns:
        QuantType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_quant_type("nf4")
        <QuantType.NF4: 'nf4'>
        >>> get_quant_type("fp4")
        <QuantType.FP4: 'fp4'>

        >>> get_quant_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid quant type: invalid
    """
    for qt in QuantType:
        if qt.value == name:
            return qt
    msg = f"invalid quant type: {name}"
    raise ValueError(msg)


def list_compute_types() -> list[str]:
    """List available compute dtypes.

    Returns:
        Sorted list of compute dtype names.

    Examples:
        >>> types = list_compute_types()
        >>> "float16" in types
        True
        >>> "bfloat16" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(c.value for c in ComputeType)


def validate_compute_type(compute_type: str) -> bool:
    """Check if a compute dtype is valid.

    Args:
        compute_type: Type to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_compute_type("float16")
        True
        >>> validate_compute_type("bfloat16")
        True
        >>> validate_compute_type("invalid")
        False
        >>> validate_compute_type("")
        False
    """
    valid_types = {c.value for c in ComputeType}
    return compute_type in valid_types


def get_compute_type(name: str) -> ComputeType:
    """Get compute dtype enum from string.

    Args:
        name: Type name.

    Returns:
        ComputeType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_compute_type("float16")
        <ComputeType.FLOAT16: 'float16'>
        >>> get_compute_type("bfloat16")
        <ComputeType.BFLOAT16: 'bfloat16'>

        >>> get_compute_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid compute type: invalid
    """
    for ct in ComputeType:
        if ct.value == name:
            return ct
    msg = f"invalid compute type: {name}"
    raise ValueError(msg)


def list_quant_bits() -> list[int]:
    """List supported quantization bit widths.

    Returns:
        List of supported bit widths.

    Examples:
        >>> bits = list_quant_bits()
        >>> 4 in bits
        True
        >>> 8 in bits
        True
    """
    return sorted(q.value for q in QuantBits)


def format_memory_estimate(estimate: MemoryEstimate) -> str:
    """Format memory estimate for display.

    Args:
        estimate: Memory estimate to format.

    Returns:
        Formatted string with memory breakdown.

    Raises:
        ValueError: If estimate is None.

    Examples:
        >>> est = MemoryEstimate(
        ...     model_memory_mb=4000,
        ...     adapter_memory_mb=50,
        ...     optimizer_memory_mb=100,
        ...     activation_memory_mb=500,
        ...     total_memory_mb=4650,
        ... )
        >>> formatted = format_memory_estimate(est)
        >>> "Model:" in formatted
        True
        >>> "Total:" in formatted
        True

        >>> format_memory_estimate(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: estimate cannot be None
    """
    if estimate is None:
        msg = "estimate cannot be None"
        raise ValueError(msg)

    lines = [
        f"Model: {estimate.model_memory_mb:.1f} MB",
        f"Adapter: {estimate.adapter_memory_mb:.1f} MB",
        f"Optimizer: {estimate.optimizer_memory_mb:.1f} MB",
        f"Activations: {estimate.activation_memory_mb:.1f} MB",
        f"Total: {estimate.total_memory_mb:.1f} MB",
    ]
    return "\n".join(lines)
