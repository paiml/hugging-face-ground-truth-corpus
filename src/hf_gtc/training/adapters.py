"""Adapter-based fine-tuning utilities (LoRA, QLoRA, AdaLoRA, IA3).

This module provides unified configuration and utilities for adapter-based
parameter-efficient fine-tuning methods, including LoRA, QLoRA, AdaLoRA,
IA3, and Prefix Tuning.

Examples:
    >>> from hf_gtc.training.adapters import create_adapter_config, AdapterType
    >>> config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
    >>> config.adapter_type
    <AdapterType.LORA: 'lora'>
    >>> config.lora_config.r
    8
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class AdapterType(Enum):
    """Supported adapter types for parameter-efficient fine-tuning.

    Attributes:
        LORA: Low-Rank Adaptation.
        QLORA: Quantized Low-Rank Adaptation (4-bit).
        ADALORA: Adaptive Low-Rank Adaptation with rank allocation.
        IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations.
        PREFIX_TUNING: Prefix tuning with learnable prefix tokens.

    Examples:
        >>> AdapterType.LORA.value
        'lora'
        >>> AdapterType.QLORA.value
        'qlora'
        >>> AdapterType.ADALORA.value
        'adalora'
        >>> AdapterType.IA3.value
        'ia3'
        >>> AdapterType.PREFIX_TUNING.value
        'prefix_tuning'
    """

    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    IA3 = "ia3"
    PREFIX_TUNING = "prefix_tuning"


VALID_ADAPTER_TYPES = frozenset(t.value for t in AdapterType)


class TargetModules(Enum):
    """Target module groups for adapter application.

    Attributes:
        ATTENTION: Apply to attention layers only (q, k, v, o projections).
        MLP: Apply to MLP/FFN layers only (gate, up, down projections).
        ALL: Apply to all supported layers.

    Examples:
        >>> TargetModules.ATTENTION.value
        'attention'
        >>> TargetModules.MLP.value
        'mlp'
        >>> TargetModules.ALL.value
        'all'
    """

    ATTENTION = "attention"
    MLP = "mlp"
    ALL = "all"


VALID_TARGET_MODULES = frozenset(t.value for t in TargetModules)


class MergeStrategy(Enum):
    """Strategies for merging adapter weights.

    Attributes:
        CAT: Concatenate adapter weights.
        ADD: Add adapter weights element-wise.
        WEIGHTED: Weighted combination of adapter weights.

    Examples:
        >>> MergeStrategy.CAT.value
        'cat'
        >>> MergeStrategy.ADD.value
        'add'
        >>> MergeStrategy.WEIGHTED.value
        'weighted'
    """

    CAT = "cat"
    ADD = "add"
    WEIGHTED = "weighted"


VALID_MERGE_STRATEGIES = frozenset(s.value for s in MergeStrategy)


# Default target modules for common architectures
ATTENTION_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_MODULES = frozenset({"gate_proj", "up_proj", "down_proj"})
ALL_MODULES = ATTENTION_MODULES | MLP_MODULES


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        r: Rank of the low-rank matrices.
        alpha: Scaling factor for LoRA weights (lora_alpha).
        dropout: Dropout probability for LoRA layers.
        target_modules: Modules to apply LoRA to.
        bias: Bias configuration ("none", "all", "lora_only").
        use_rslora: Whether to use rank-stabilized LoRA.
        use_dora: Whether to use weight-decomposed LoRA (DoRA).

    Examples:
        >>> config = LoRAConfig(
        ...     r=8,
        ...     alpha=16,
        ...     dropout=0.1,
        ...     target_modules=("q_proj", "v_proj"),
        ...     bias="none",
        ...     use_rslora=False,
        ...     use_dora=False,
        ... )
        >>> config.r
        8
        >>> config.alpha
        16
    """

    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str
    use_rslora: bool
    use_dora: bool


@dataclass(frozen=True, slots=True)
class QLoRAConfig:
    """Configuration for QLoRA quantization settings.

    Attributes:
        bits: Number of bits for quantization (4 or 8).
        double_quant: Whether to use double quantization.
        compute_dtype: Data type for compute operations.
        quant_type: Quantization type (nf4 or fp4).

    Examples:
        >>> config = QLoRAConfig(
        ...     bits=4,
        ...     double_quant=True,
        ...     compute_dtype="float16",
        ...     quant_type="nf4",
        ... )
        >>> config.bits
        4
        >>> config.double_quant
        True
    """

    bits: int
    double_quant: bool
    compute_dtype: str
    quant_type: str


@dataclass(frozen=True, slots=True)
class AdaLoRAConfig:
    """Configuration for AdaLoRA with adaptive rank allocation.

    Attributes:
        init_r: Initial rank for all modules.
        target_r: Target average rank after adaptation.
        beta1: Coefficient for importance moving average.
        beta2: Coefficient for orthogonality regularization.
        tinit: Initial training steps before rank allocation.
        tfinal: Final step for rank allocation.
        delta_t: Interval between rank allocations.

    Examples:
        >>> config = AdaLoRAConfig(
        ...     init_r=12,
        ...     target_r=8,
        ...     beta1=0.85,
        ...     beta2=0.85,
        ...     tinit=200,
        ...     tfinal=1000,
        ...     delta_t=10,
        ... )
        >>> config.init_r
        12
        >>> config.target_r
        8
    """

    init_r: int
    target_r: int
    beta1: float
    beta2: float
    tinit: int
    tfinal: int
    delta_t: int


@dataclass(frozen=True, slots=True)
class IA3Config:
    """Configuration for IA3 adapters.

    Attributes:
        target_modules: Modules to apply IA3 to.
        feedforward_modules: Names of feedforward modules.
        init_weights: Whether to initialize weights to 1.0.

    Examples:
        >>> config = IA3Config(
        ...     target_modules=("q_proj", "v_proj", "down_proj"),
        ...     feedforward_modules=("down_proj",),
        ...     init_weights=True,
        ... )
        >>> config.init_weights
        True
    """

    target_modules: tuple[str, ...]
    feedforward_modules: tuple[str, ...]
    init_weights: bool


@dataclass(frozen=True, slots=True)
class PrefixTuningConfig:
    """Configuration for prefix tuning.

    Attributes:
        num_virtual_tokens: Number of prefix tokens to prepend.
        encoder_hidden_size: Hidden size of the prefix encoder.
        prefix_projection: Whether to use prefix projection.

    Examples:
        >>> config = PrefixTuningConfig(
        ...     num_virtual_tokens=20,
        ...     encoder_hidden_size=512,
        ...     prefix_projection=True,
        ... )
        >>> config.num_virtual_tokens
        20
    """

    num_virtual_tokens: int
    encoder_hidden_size: int
    prefix_projection: bool


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Unified adapter configuration.

    Attributes:
        adapter_type: Type of adapter to use.
        lora_config: LoRA-specific configuration (for LORA/QLORA/ADALORA).
        qlora_config: QLoRA quantization configuration (for QLORA).
        adalora_config: AdaLoRA-specific configuration (for ADALORA).
        ia3_config: IA3-specific configuration (for IA3).
        prefix_config: Prefix tuning configuration (for PREFIX_TUNING).
        trainable_params: Estimated number of trainable parameters.
        base_model_params: Number of parameters in base model.

    Examples:
        >>> from hf_gtc.training.adapters import create_adapter_config
        >>> config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        >>> config.adapter_type
        <AdapterType.LORA: 'lora'>
    """

    adapter_type: AdapterType
    lora_config: LoRAConfig | None
    qlora_config: QLoRAConfig | None
    adalora_config: AdaLoRAConfig | None
    ia3_config: IA3Config | None
    prefix_config: PrefixTuningConfig | None
    trainable_params: int
    base_model_params: int


@dataclass(frozen=True, slots=True)
class AdapterStats:
    """Statistics for adapter training.

    Attributes:
        adapter_type: Type of adapter used.
        trainable_params: Number of trainable parameters.
        total_params: Total model parameters.
        trainable_percent: Percentage of trainable parameters.
        memory_saved_mb: Estimated memory saved in MB.

    Examples:
        >>> stats = AdapterStats(
        ...     adapter_type=AdapterType.LORA,
        ...     trainable_params=4_000_000,
        ...     total_params=7_000_000_000,
        ...     trainable_percent=0.057,
        ...     memory_saved_mb=12000.0,
        ... )
        >>> stats.trainable_percent < 0.1
        True
    """

    adapter_type: AdapterType
    trainable_params: int
    total_params: int
    trainable_percent: float
    memory_saved_mb: float


def validate_lora_config(config: LoRAConfig) -> None:
    """Validate LoRA configuration.

    Args:
        config: LoRA configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If r is not positive.
        ValueError: If alpha is not positive.
        ValueError: If dropout is not in [0.0, 1.0).
        ValueError: If bias is not a valid option.

    Examples:
        >>> config = LoRAConfig(
        ...     r=8, alpha=16, dropout=0.1, target_modules=("q_proj",),
        ...     bias="none", use_rslora=False, use_dora=False
        ... )
        >>> validate_lora_config(config)  # No error

        >>> validate_lora_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = LoRAConfig(
        ...     r=0, alpha=16, dropout=0.1, target_modules=("q_proj",),
        ...     bias="none", use_rslora=False, use_dora=False
        ... )
        >>> validate_lora_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: r must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.r <= 0:
        msg = f"r must be positive, got {config.r}"
        raise ValueError(msg)

    if config.alpha <= 0:
        msg = f"alpha must be positive, got {config.alpha}"
        raise ValueError(msg)

    if not 0.0 <= config.dropout < 1.0:
        msg = f"dropout must be in [0.0, 1.0), got {config.dropout}"
        raise ValueError(msg)

    valid_bias = {"none", "all", "lora_only"}
    if config.bias not in valid_bias:
        msg = f"bias must be one of {valid_bias}, got '{config.bias}'"
        raise ValueError(msg)

    if len(config.target_modules) == 0:
        msg = "target_modules cannot be empty"
        raise ValueError(msg)


def validate_qlora_config(config: QLoRAConfig) -> None:
    """Validate QLoRA configuration.

    Args:
        config: QLoRA configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If bits is not 4 or 8.
        ValueError: If compute_dtype is not valid.
        ValueError: If quant_type is not valid.

    Examples:
        >>> config = QLoRAConfig(
        ...     bits=4, double_quant=True, compute_dtype="float16", quant_type="nf4"
        ... )
        >>> validate_qlora_config(config)  # No error

        >>> validate_qlora_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = QLoRAConfig(
        ...     bits=3, double_quant=True, compute_dtype="float16", quant_type="nf4"
        ... )
        >>> validate_qlora_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be 4 or 8
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.bits not in (4, 8):
        msg = f"bits must be 4 or 8, got {config.bits}"
        raise ValueError(msg)

    valid_dtypes = {"float16", "bfloat16", "float32"}
    if config.compute_dtype not in valid_dtypes:
        msg = (
            f"compute_dtype must be one of {valid_dtypes}, "
            f"got '{config.compute_dtype}'"
        )
        raise ValueError(msg)

    valid_quant_types = {"nf4", "fp4"}
    if config.quant_type not in valid_quant_types:
        msg = (
            f"quant_type must be one of {valid_quant_types}, "
            f"got '{config.quant_type}'"
        )
        raise ValueError(msg)


def validate_adalora_config(config: AdaLoRAConfig) -> None:
    """Validate AdaLoRA configuration.

    Args:
        config: AdaLoRA configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If init_r is not positive.
        ValueError: If target_r is not positive or greater than init_r.
        ValueError: If beta values are not in (0, 1).
        ValueError: If timing parameters are invalid.

    Examples:
        >>> config = AdaLoRAConfig(
        ...     init_r=12, target_r=8, beta1=0.85, beta2=0.85,
        ...     tinit=200, tfinal=1000, delta_t=10
        ... )
        >>> validate_adalora_config(config)  # No error

        >>> validate_adalora_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = AdaLoRAConfig(
        ...     init_r=0, target_r=8, beta1=0.85, beta2=0.85,
        ...     tinit=200, tfinal=1000, delta_t=10
        ... )
        >>> validate_adalora_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: init_r must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.init_r <= 0:
        msg = f"init_r must be positive, got {config.init_r}"
        raise ValueError(msg)

    if config.target_r <= 0:
        msg = f"target_r must be positive, got {config.target_r}"
        raise ValueError(msg)

    if config.target_r > config.init_r:
        msg = f"target_r ({config.target_r}) cannot exceed init_r ({config.init_r})"
        raise ValueError(msg)

    if not 0.0 < config.beta1 < 1.0:
        msg = f"beta1 must be in (0, 1), got {config.beta1}"
        raise ValueError(msg)

    if not 0.0 < config.beta2 < 1.0:
        msg = f"beta2 must be in (0, 1), got {config.beta2}"
        raise ValueError(msg)

    if config.tinit < 0:
        msg = f"tinit must be non-negative, got {config.tinit}"
        raise ValueError(msg)

    if config.tfinal <= config.tinit:
        msg = f"tfinal ({config.tfinal}) must be greater than tinit ({config.tinit})"
        raise ValueError(msg)

    if config.delta_t <= 0:
        msg = f"delta_t must be positive, got {config.delta_t}"
        raise ValueError(msg)


def validate_ia3_config(config: IA3Config) -> None:
    """Validate IA3 configuration.

    Args:
        config: IA3 configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If target_modules is empty.

    Examples:
        >>> config = IA3Config(
        ...     target_modules=("q_proj", "v_proj"),
        ...     feedforward_modules=("down_proj",),
        ...     init_weights=True
        ... )
        >>> validate_ia3_config(config)  # No error

        >>> validate_ia3_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = IA3Config(
        ...     target_modules=(), feedforward_modules=(), init_weights=True
        ... )
        >>> validate_ia3_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_modules cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if len(config.target_modules) == 0:
        msg = "target_modules cannot be empty"
        raise ValueError(msg)


def validate_prefix_config(config: PrefixTuningConfig) -> None:
    """Validate prefix tuning configuration.

    Args:
        config: Prefix tuning configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_virtual_tokens is not positive.
        ValueError: If encoder_hidden_size is not positive.

    Examples:
        >>> config = PrefixTuningConfig(
        ...     num_virtual_tokens=20, encoder_hidden_size=512, prefix_projection=True
        ... )
        >>> validate_prefix_config(config)  # No error

        >>> validate_prefix_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = PrefixTuningConfig(
        ...     num_virtual_tokens=0, encoder_hidden_size=512, prefix_projection=True
        ... )
        >>> validate_prefix_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_virtual_tokens must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_virtual_tokens <= 0:
        msg = f"num_virtual_tokens must be positive, got {config.num_virtual_tokens}"
        raise ValueError(msg)

    if config.encoder_hidden_size <= 0:
        msg = f"encoder_hidden_size must be positive, got {config.encoder_hidden_size}"
        raise ValueError(msg)


def validate_adapter_config(config: AdapterConfig) -> None:
    """Validate adapter configuration.

    Args:
        config: Adapter configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If required sub-config is missing for adapter type.

    Examples:
        >>> from hf_gtc.training.adapters import create_adapter_config
        >>> config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        >>> validate_adapter_config(config)  # No error

        >>> validate_adapter_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    lora_types = (AdapterType.LORA, AdapterType.QLORA, AdapterType.ADALORA)
    if config.adapter_type in lora_types:
        if config.lora_config is None:
            msg = f"lora_config required for {config.adapter_type.value}"
            raise ValueError(msg)
        validate_lora_config(config.lora_config)

    if config.adapter_type == AdapterType.QLORA:
        if config.qlora_config is None:
            msg = "qlora_config required for qlora adapter type"
            raise ValueError(msg)
        validate_qlora_config(config.qlora_config)

    if config.adapter_type == AdapterType.ADALORA:
        if config.adalora_config is None:
            msg = "adalora_config required for adalora adapter type"
            raise ValueError(msg)
        validate_adalora_config(config.adalora_config)

    if config.adapter_type == AdapterType.IA3:
        if config.ia3_config is None:
            msg = "ia3_config required for ia3 adapter type"
            raise ValueError(msg)
        validate_ia3_config(config.ia3_config)

    if config.adapter_type == AdapterType.PREFIX_TUNING:
        if config.prefix_config is None:
            msg = "prefix_config required for prefix_tuning adapter type"
            raise ValueError(msg)
        validate_prefix_config(config.prefix_config)

    if config.base_model_params <= 0:
        msg = f"base_model_params must be positive, got {config.base_model_params}"
        raise ValueError(msg)


def create_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: tuple[str, ...] | str = "all",
    bias: str = "none",
    use_rslora: bool = False,
    use_dora: bool = False,
) -> LoRAConfig:
    """Create a LoRA configuration.

    Args:
        r: Rank of low-rank matrices. Defaults to 8.
        alpha: Scaling factor. Defaults to 16.
        dropout: Dropout probability. Defaults to 0.1.
        target_modules: Modules to apply LoRA to, or preset ("attention", "mlp", "all").
            Defaults to "all".
        bias: Bias configuration. Defaults to "none".
        use_rslora: Enable rank-stabilized LoRA. Defaults to False.
        use_dora: Enable weight-decomposed LoRA (DoRA). Defaults to False.

    Returns:
        Validated LoRAConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_lora_config(r=16, alpha=32)
        >>> config.r
        16
        >>> config.alpha
        32

        >>> config = create_lora_config(target_modules="attention")
        >>> "q_proj" in config.target_modules
        True

        >>> config = create_lora_config(target_modules=("q_proj", "v_proj"))
        >>> len(config.target_modules)
        2

        >>> create_lora_config(r=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: r must be positive
    """
    if isinstance(target_modules, str):
        if target_modules == "attention":
            modules = tuple(sorted(ATTENTION_MODULES))
        elif target_modules == "mlp":
            modules = tuple(sorted(MLP_MODULES))
        elif target_modules == "all":
            modules = tuple(sorted(ALL_MODULES))
        else:
            msg = (
                f"target_modules preset must be 'attention', 'mlp', or 'all', "
                f"got '{target_modules}'"
            )
            raise ValueError(msg)
    else:
        modules = target_modules

    config = LoRAConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=modules,
        bias=bias,
        use_rslora=use_rslora,
        use_dora=use_dora,
    )
    validate_lora_config(config)
    return config


def create_qlora_config(
    bits: int = 4,
    double_quant: bool = True,
    compute_dtype: str = "float16",
    quant_type: str = "nf4",
) -> QLoRAConfig:
    """Create a QLoRA quantization configuration.

    Args:
        bits: Number of bits for quantization. Defaults to 4.
        double_quant: Enable double quantization. Defaults to True.
        compute_dtype: Data type for compute. Defaults to "float16".
        quant_type: Quantization type. Defaults to "nf4".

    Returns:
        Validated QLoRAConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_qlora_config(bits=4)
        >>> config.bits
        4
        >>> config.quant_type
        'nf4'

        >>> config = create_qlora_config(bits=8, quant_type="fp4")
        >>> config.bits
        8

        >>> create_qlora_config(bits=3)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: bits must be 4 or 8
    """
    config = QLoRAConfig(
        bits=bits,
        double_quant=double_quant,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
    )
    validate_qlora_config(config)
    return config


def create_adalora_config(
    init_r: int = 12,
    target_r: int = 8,
    beta1: float = 0.85,
    beta2: float = 0.85,
    tinit: int = 200,
    tfinal: int = 1000,
    delta_t: int = 10,
) -> AdaLoRAConfig:
    """Create an AdaLoRA configuration.

    Args:
        init_r: Initial rank for all modules. Defaults to 12.
        target_r: Target average rank. Defaults to 8.
        beta1: Importance moving average coefficient. Defaults to 0.85.
        beta2: Orthogonality regularization coefficient. Defaults to 0.85.
        tinit: Initial steps before rank allocation. Defaults to 200.
        tfinal: Final step for rank allocation. Defaults to 1000.
        delta_t: Interval between rank allocations. Defaults to 10.

    Returns:
        Validated AdaLoRAConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_adalora_config(init_r=16, target_r=8)
        >>> config.init_r
        16
        >>> config.target_r
        8

        >>> create_adalora_config(init_r=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: init_r must be positive

        >>> create_adalora_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     init_r=8, target_r=12
        ... )
        Traceback (most recent call last):
        ValueError: target_r (12) cannot exceed init_r (8)
    """
    config = AdaLoRAConfig(
        init_r=init_r,
        target_r=target_r,
        beta1=beta1,
        beta2=beta2,
        tinit=tinit,
        tfinal=tfinal,
        delta_t=delta_t,
    )
    validate_adalora_config(config)
    return config


def create_ia3_config(
    target_modules: tuple[str, ...] | str = "all",
    feedforward_modules: tuple[str, ...] | None = None,
    init_weights: bool = True,
) -> IA3Config:
    """Create an IA3 configuration.

    Args:
        target_modules: Modules to apply IA3 to. Defaults to "all".
        feedforward_modules: Names of feedforward modules. Defaults to auto-detect.
        init_weights: Initialize weights to 1.0. Defaults to True.

    Returns:
        Validated IA3Config instance.

    Raises:
        ValueError: If target_modules is empty.

    Examples:
        >>> config = create_ia3_config()
        >>> "q_proj" in config.target_modules
        True

        >>> config = create_ia3_config(target_modules=("k_proj", "v_proj", "down_proj"))
        >>> "k_proj" in config.target_modules
        True

        >>> create_ia3_config(target_modules=())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_modules cannot be empty
    """
    if isinstance(target_modules, str):
        if target_modules == "attention":
            modules = tuple(sorted(ATTENTION_MODULES))
        elif target_modules == "mlp":
            modules = tuple(sorted(MLP_MODULES))
        elif target_modules == "all":
            modules = tuple(sorted(ALL_MODULES))
        else:
            msg = (
                f"target_modules preset must be 'attention', 'mlp', or 'all', "
                f"got '{target_modules}'"
            )
            raise ValueError(msg)
    else:
        modules = target_modules

    if feedforward_modules is None:
        feedforward_modules = tuple(m for m in modules if m in MLP_MODULES)

    config = IA3Config(
        target_modules=modules,
        feedforward_modules=feedforward_modules,
        init_weights=init_weights,
    )
    validate_ia3_config(config)
    return config


def create_prefix_config(
    num_virtual_tokens: int = 20,
    encoder_hidden_size: int = 512,
    prefix_projection: bool = True,
) -> PrefixTuningConfig:
    """Create a prefix tuning configuration.

    Args:
        num_virtual_tokens: Number of prefix tokens. Defaults to 20.
        encoder_hidden_size: Hidden size of prefix encoder. Defaults to 512.
        prefix_projection: Use prefix projection. Defaults to True.

    Returns:
        Validated PrefixTuningConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_prefix_config(num_virtual_tokens=30)
        >>> config.num_virtual_tokens
        30

        >>> create_prefix_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     num_virtual_tokens=0
        ... )
        Traceback (most recent call last):
        ValueError: num_virtual_tokens must be positive
    """
    config = PrefixTuningConfig(
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        prefix_projection=prefix_projection,
    )
    validate_prefix_config(config)
    return config


def create_adapter_config(
    adapter_type: str = "lora",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: tuple[str, ...] | str = "all",
    bias: str = "none",
    use_rslora: bool = False,
    use_dora: bool = False,
    bits: int = 4,
    double_quant: bool = True,
    compute_dtype: str = "float16",
    quant_type: str = "nf4",
    init_r: int = 12,
    target_r: int = 8,
    num_virtual_tokens: int = 20,
    encoder_hidden_size: int = 512,
    prefix_projection: bool = True,
    base_model_params: int = 7_000_000_000,
) -> AdapterConfig:
    """Create a unified adapter configuration.

    Args:
        adapter_type: Type of adapter
            ("lora", "qlora", "adalora", "ia3", "prefix_tuning").
        r: LoRA rank. Defaults to 8.
        alpha: LoRA alpha. Defaults to 16.
        dropout: LoRA dropout. Defaults to 0.1.
        target_modules: Target modules or preset. Defaults to "all".
        bias: Bias config. Defaults to "none".
        use_rslora: Enable RS-LoRA. Defaults to False.
        use_dora: Enable DoRA. Defaults to False.
        bits: QLoRA bits. Defaults to 4.
        double_quant: QLoRA double quantization. Defaults to True.
        compute_dtype: QLoRA compute dtype. Defaults to "float16".
        quant_type: QLoRA quant type. Defaults to "nf4".
        init_r: AdaLoRA initial rank. Defaults to 12.
        target_r: AdaLoRA target rank. Defaults to 8.
        num_virtual_tokens: Prefix tuning tokens. Defaults to 20.
        encoder_hidden_size: Prefix encoder hidden size. Defaults to 512.
        prefix_projection: Use prefix projection. Defaults to True.
        base_model_params: Base model parameters. Defaults to 7B.

    Returns:
        Validated AdapterConfig instance.

    Raises:
        ValueError: If adapter_type is invalid or required params are invalid.

    Examples:
        >>> config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        >>> config.adapter_type
        <AdapterType.LORA: 'lora'>
        >>> config.lora_config.r
        8

        >>> config = create_adapter_config(adapter_type="qlora", bits=4)
        >>> config.adapter_type
        <AdapterType.QLORA: 'qlora'>
        >>> config.qlora_config.bits
        4

        >>> config = create_adapter_config(adapter_type="ia3")
        >>> config.adapter_type
        <AdapterType.IA3: 'ia3'>

        >>> create_adapter_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     adapter_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: adapter_type must be one of...
    """
    if adapter_type not in VALID_ADAPTER_TYPES:
        msg = f"adapter_type must be one of {VALID_ADAPTER_TYPES}, got '{adapter_type}'"
        raise ValueError(msg)

    adapter_type_enum = AdapterType(adapter_type)

    lora_config: LoRAConfig | None = None
    qlora_config: QLoRAConfig | None = None
    adalora_config: AdaLoRAConfig | None = None
    ia3_config: IA3Config | None = None
    prefix_config: PrefixTuningConfig | None = None

    if adapter_type_enum in (AdapterType.LORA, AdapterType.QLORA, AdapterType.ADALORA):
        lora_config = create_lora_config(
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            bias=bias,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    if adapter_type_enum == AdapterType.QLORA:
        qlora_config = create_qlora_config(
            bits=bits,
            double_quant=double_quant,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
        )

    if adapter_type_enum == AdapterType.ADALORA:
        adalora_config = create_adalora_config(
            init_r=init_r,
            target_r=target_r,
        )

    if adapter_type_enum == AdapterType.IA3:
        ia3_config = create_ia3_config(
            target_modules=target_modules,
        )

    if adapter_type_enum == AdapterType.PREFIX_TUNING:
        prefix_config = create_prefix_config(
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            prefix_projection=prefix_projection,
        )

    # Calculate trainable params
    trainable = calculate_trainable_params(
        adapter_type=adapter_type,
        base_model_params=base_model_params,
        r=r,
        num_target_modules=len(lora_config.target_modules) if lora_config else 7,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
    )

    config = AdapterConfig(
        adapter_type=adapter_type_enum,
        lora_config=lora_config,
        qlora_config=qlora_config,
        adalora_config=adalora_config,
        ia3_config=ia3_config,
        prefix_config=prefix_config,
        trainable_params=trainable,
        base_model_params=base_model_params,
    )
    return config


def calculate_trainable_params(
    adapter_type: str,
    base_model_params: int,
    r: int = 8,
    num_target_modules: int = 7,
    hidden_size: int = 4096,
    num_virtual_tokens: int = 20,
    encoder_hidden_size: int = 512,
    num_layers: int = 32,
) -> int:
    """Calculate estimated trainable parameters for adapter.

    Args:
        adapter_type: Type of adapter.
        base_model_params: Number of base model parameters.
        r: LoRA rank. Defaults to 8.
        num_target_modules: Number of target modules. Defaults to 7.
        hidden_size: Model hidden size. Defaults to 4096.
        num_virtual_tokens: Prefix tuning tokens. Defaults to 20.
        encoder_hidden_size: Prefix encoder hidden size. Defaults to 512.
        num_layers: Number of transformer layers. Defaults to 32.

    Returns:
        Estimated number of trainable parameters.

    Raises:
        ValueError: If base_model_params is not positive.
        ValueError: If adapter_type is not valid.

    Examples:
        >>> params = calculate_trainable_params("lora", 7_000_000_000, r=8)
        >>> params > 0
        True
        >>> params < 7_000_000_000
        True

        >>> params_ia3 = calculate_trainable_params("ia3", 7_000_000_000)
        >>> params_lora = calculate_trainable_params("lora", 7_000_000_000)
        >>> params_ia3 < params_lora
        True

        >>> calculate_trainable_params("lora", 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: base_model_params must be positive

        >>> calculate_trainable_params(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid", 7_000_000_000
        ... )
        Traceback (most recent call last):
        ValueError: adapter_type must be one of...
    """
    if base_model_params <= 0:
        msg = f"base_model_params must be positive, got {base_model_params}"
        raise ValueError(msg)

    if adapter_type not in VALID_ADAPTER_TYPES:
        msg = f"adapter_type must be one of {VALID_ADAPTER_TYPES}, got '{adapter_type}'"
        raise ValueError(msg)

    if adapter_type in ("lora", "qlora"):
        # LoRA: 2 * hidden_size * r * num_modules * num_layers
        return 2 * hidden_size * r * num_target_modules * num_layers

    if adapter_type == "adalora":
        # AdaLoRA similar to LoRA but with additional importance parameters
        lora_params = 2 * hidden_size * r * num_target_modules * num_layers
        importance_params = r * num_target_modules * num_layers
        return lora_params + importance_params

    if adapter_type == "ia3":
        # IA3: Only learned vectors, much fewer params
        # 1 vector per module per layer
        return hidden_size * num_target_modules * num_layers

    if adapter_type == "prefix_tuning":
        # Prefix tuning: prefix_length * hidden_size * num_layers * 2 (key + value)
        # Plus encoder if using projection
        prefix_params = num_virtual_tokens * hidden_size * num_layers * 2
        encoder_params = encoder_hidden_size * hidden_size * 2
        return prefix_params + encoder_params

    # Fallback (shouldn't reach here due to validation)
    return 0


def estimate_memory_savings(
    adapter_type: str,
    base_model_params: int,
    trainable_params: int,
    bits: int = 16,
) -> tuple[float, float]:
    """Estimate memory savings from using adapters.

    Args:
        adapter_type: Type of adapter being used.
        base_model_params: Number of parameters in base model.
        trainable_params: Number of trainable adapter parameters.
        bits: Bits per parameter for base model (16 for fp16, 4 for qlora).

    Returns:
        Tuple of (memory_saved_mb, percentage_saved).

    Raises:
        ValueError: If any parameter is not positive.
        ValueError: If adapter_type is not valid.

    Examples:
        >>> saved_mb, pct = estimate_memory_savings("lora", 7_000_000_000, 4_000_000)
        >>> saved_mb > 0
        True
        >>> pct > 90
        True

        >>> saved_mb, pct = estimate_memory_savings(
        ...     "qlora", 7_000_000_000, 4_000_000, bits=4
        ... )
        >>> pct > 70
        True

        >>> estimate_memory_savings(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "lora", 0, 1000
        ... )
        Traceback (most recent call last):
        ValueError: base_model_params must be positive
    """
    if base_model_params <= 0:
        msg = f"base_model_params must be positive, got {base_model_params}"
        raise ValueError(msg)

    if trainable_params <= 0:
        msg = f"trainable_params must be positive, got {trainable_params}"
        raise ValueError(msg)

    if adapter_type not in VALID_ADAPTER_TYPES:
        msg = f"adapter_type must be one of {VALID_ADAPTER_TYPES}, got '{adapter_type}'"
        raise ValueError(msg)

    if bits not in (4, 8, 16, 32):
        msg = f"bits must be 4, 8, 16, or 32, got {bits}"
        raise ValueError(msg)

    # Full fine-tuning memory: all params at fp16 + gradients + optimizer states
    # Approximately 16 bytes per parameter for Adam optimizer at fp16
    bytes_per_param_full = 16.0

    # Adapter fine-tuning: base model at bits precision (frozen) +
    # trainable params at fp16 with gradients and optimizer
    bytes_per_frozen_param = bits / 8
    bytes_per_trainable_param = 16.0

    full_memory_bytes = base_model_params * bytes_per_param_full
    adapter_memory_bytes = (
        base_model_params * bytes_per_frozen_param
        + trainable_params * bytes_per_trainable_param
    )

    memory_saved_bytes = full_memory_bytes - adapter_memory_bytes
    memory_saved_mb = memory_saved_bytes / (1024 * 1024)
    percentage_saved = (memory_saved_bytes / full_memory_bytes) * 100

    return memory_saved_mb, max(0, percentage_saved)


def merge_adapter_weights(
    base_weights: list[float],
    adapter_weights: list[float],
    strategy: str = "add",
    alpha: float = 1.0,
) -> list[float]:
    """Merge adapter weights into base model weights.

    Args:
        base_weights: Base model weights.
        adapter_weights: Adapter weights to merge.
        strategy: Merge strategy ("cat", "add", "weighted"). Defaults to "add".
        alpha: Scaling factor for weighted merge. Defaults to 1.0.

    Returns:
        Merged weights list.

    Raises:
        ValueError: If strategy is not valid.
        ValueError: If weights lengths don't match for add/weighted.
        ValueError: If alpha is not in (0, 1] for weighted strategy.

    Examples:
        >>> merge_adapter_weights([1.0, 2.0], [0.1, 0.2], strategy="add")
        [1.1, 2.2]

        >>> merge_adapter_weights(
        ...     [1.0, 2.0], [0.1, 0.2], strategy="weighted", alpha=0.5
        ... )
        [1.05, 2.1]

        >>> merge_adapter_weights([1.0], [2.0], strategy="cat")
        [1.0, 2.0]

        >>> merge_adapter_weights(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     [1.0], [2.0, 3.0], strategy="add"
        ... )
        Traceback (most recent call last):
        ValueError: weights must have same length for add strategy

        >>> merge_adapter_weights(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     [1.0], [2.0], strategy="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: strategy must be one of...
    """
    if strategy not in VALID_MERGE_STRATEGIES:
        msg = f"strategy must be one of {VALID_MERGE_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    if strategy == "cat":
        return base_weights + adapter_weights

    if len(base_weights) != len(adapter_weights):
        msg = f"weights must have same length for {strategy} strategy"
        raise ValueError(msg)

    if strategy == "add":
        return [b + a for b, a in zip(base_weights, adapter_weights, strict=True)]

    # weighted
    if not 0.0 < alpha <= 1.0:
        msg = f"alpha must be in (0, 1] for weighted strategy, got {alpha}"
        raise ValueError(msg)

    return [b + alpha * a for b, a in zip(base_weights, adapter_weights, strict=True)]


def calculate_lora_rank(
    base_model_params: int,
    memory_budget_mb: float,
    target_modules: int = 7,
    num_layers: int = 32,
    hidden_size: int = 4096,
) -> int:
    """Calculate optimal LoRA rank given memory budget.

    Args:
        base_model_params: Number of base model parameters.
        memory_budget_mb: Memory budget in MB for trainable params.
        target_modules: Number of target modules. Defaults to 7.
        num_layers: Number of transformer layers. Defaults to 32.
        hidden_size: Model hidden size. Defaults to 4096.

    Returns:
        Recommended LoRA rank.

    Raises:
        ValueError: If base_model_params is not positive.
        ValueError: If memory_budget_mb is not positive.

    Examples:
        >>> rank = calculate_lora_rank(7_000_000_000, memory_budget_mb=100)
        >>> 1 <= rank <= 64
        True

        >>> rank_small = calculate_lora_rank(7_000_000_000, memory_budget_mb=50)
        >>> rank_large = calculate_lora_rank(7_000_000_000, memory_budget_mb=200)
        >>> rank_small <= rank_large
        True

        >>> calculate_lora_rank(0, 100)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: base_model_params must be positive

        >>> calculate_lora_rank(7_000_000_000, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: memory_budget_mb must be positive
    """
    if base_model_params <= 0:
        msg = f"base_model_params must be positive, got {base_model_params}"
        raise ValueError(msg)

    if memory_budget_mb <= 0:
        msg = f"memory_budget_mb must be positive, got {memory_budget_mb}"
        raise ValueError(msg)

    # Memory per parameter: fp16 weights + fp16 gradients + fp32 optimizer states
    # = 2 + 2 + 8 = 12 bytes
    bytes_per_trainable_param = 12.0
    memory_budget_bytes = memory_budget_mb * 1024 * 1024

    # max_params = memory_budget_bytes / bytes_per_trainable_param
    max_params = memory_budget_bytes / bytes_per_trainable_param

    # params = 2 * hidden_size * r * target_modules * num_layers
    # r = params / (2 * hidden_size * target_modules * num_layers)
    divisor = 2 * hidden_size * target_modules * num_layers
    r = max_params / divisor

    # Clamp to reasonable range and round to nearest power of 2
    r = max(1, min(64, int(r)))

    # Round to nearest common value
    common_ranks = [1, 2, 4, 8, 16, 32, 64]
    return min(common_ranks, key=lambda x: abs(x - r))


def format_adapter_stats(stats: AdapterStats) -> str:
    """Format adapter statistics for display.

    Args:
        stats: Adapter statistics to format.

    Returns:
        Formatted string with stats breakdown.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = AdapterStats(
        ...     adapter_type=AdapterType.LORA,
        ...     trainable_params=4_000_000,
        ...     total_params=7_000_000_000,
        ...     trainable_percent=0.057,
        ...     memory_saved_mb=12000.0,
        ... )
        >>> formatted = format_adapter_stats(stats)
        >>> "Adapter Type: lora" in formatted
        True
        >>> "Trainable:" in formatted
        True

        >>> format_adapter_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    lines = [
        f"Adapter Type: {stats.adapter_type.value}",
        f"Trainable: {stats.trainable_params:,} / {stats.total_params:,}",
        f"Percentage: {stats.trainable_percent:.4f}%",
        f"Memory Saved: {stats.memory_saved_mb:.1f} MB",
    ]
    return "\n".join(lines)


def get_recommended_adapter_config(
    model_size: str = "7b",
    task: str = "causal_lm",
    memory_constraint_gb: float | None = None,
) -> AdapterConfig:
    """Get recommended adapter configuration for model size and task.

    Args:
        model_size: Model size category ("7b", "13b", "70b"). Defaults to "7b".
        task: Task type ("causal_lm", "classification", "seq2seq").
            Defaults to "causal_lm".
        memory_constraint_gb: GPU memory constraint in GB. Recommends qlora if tight.

    Returns:
        Recommended AdapterConfig.

    Raises:
        ValueError: If model_size is not recognized.

    Examples:
        >>> config = get_recommended_adapter_config("7b")
        >>> config.adapter_type
        <AdapterType.LORA: 'lora'>
        >>> config.lora_config.r
        8

        >>> config = get_recommended_adapter_config("70b", memory_constraint_gb=24)
        >>> config.adapter_type
        <AdapterType.QLORA: 'qlora'>

        >>> get_recommended_adapter_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: model_size must be one of...
    """
    valid_sizes = {"7b", "13b", "70b", "small", "medium", "large"}
    model_size = model_size.lower().strip()

    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    # Map sizes to params
    size_to_params = {
        "7b": 7_000_000_000,
        "13b": 13_000_000_000,
        "70b": 70_000_000_000,
        "small": 1_000_000_000,
        "medium": 7_000_000_000,
        "large": 70_000_000_000,
    }
    base_params = size_to_params[model_size]

    # Determine if we need quantization based on memory constraint
    use_qlora = False
    if memory_constraint_gb is not None:
        # Rough estimate: need ~2 bytes per param for 4-bit qlora
        min_memory_for_lora = (base_params * 2) / (1024**3)
        if memory_constraint_gb < min_memory_for_lora * 1.5:
            use_qlora = True

    # Size-specific recommendations
    if model_size in ("7b", "small", "medium", "13b"):
        r = 8
        alpha = 16
    else:  # 70b, large
        r = 16
        alpha = 32
        use_qlora = True  # Always use qlora for 70B+

    adapter_type = "qlora" if use_qlora else "lora"

    return create_adapter_config(
        adapter_type=adapter_type,
        r=r,
        alpha=alpha,
        base_model_params=base_params,
    )


def list_adapter_types() -> list[str]:
    """List all supported adapter types.

    Returns:
        Sorted list of adapter type names.

    Examples:
        >>> types = list_adapter_types()
        >>> "lora" in types
        True
        >>> "qlora" in types
        True
        >>> "ia3" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ADAPTER_TYPES)


def list_target_modules() -> list[str]:
    """List all supported target module presets.

    Returns:
        Sorted list of target module preset names.

    Examples:
        >>> modules = list_target_modules()
        >>> "attention" in modules
        True
        >>> "mlp" in modules
        True
        >>> "all" in modules
        True
    """
    return sorted(VALID_TARGET_MODULES)


def list_merge_strategies() -> list[str]:
    """List all supported merge strategies.

    Returns:
        Sorted list of merge strategy names.

    Examples:
        >>> strategies = list_merge_strategies()
        >>> "add" in strategies
        True
        >>> "cat" in strategies
        True
        >>> "weighted" in strategies
        True
    """
    return sorted(VALID_MERGE_STRATEGIES)


def get_adapter_type(name: str) -> AdapterType:
    """Get adapter type enum from string.

    Args:
        name: Adapter type name.

    Returns:
        AdapterType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_adapter_type("lora")
        <AdapterType.LORA: 'lora'>
        >>> get_adapter_type("qlora")
        <AdapterType.QLORA: 'qlora'>

        >>> get_adapter_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid adapter type: invalid
    """
    for at in AdapterType:
        if at.value == name:
            return at
    msg = f"invalid adapter type: {name}"
    raise ValueError(msg)


def get_target_modules(name: str) -> TargetModules:
    """Get target modules enum from string.

    Args:
        name: Target modules name.

    Returns:
        TargetModules enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_target_modules("attention")
        <TargetModules.ATTENTION: 'attention'>
        >>> get_target_modules("all")
        <TargetModules.ALL: 'all'>

        >>> get_target_modules("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid target modules: invalid
    """
    for tm in TargetModules:
        if tm.value == name:
            return tm
    msg = f"invalid target modules: {name}"
    raise ValueError(msg)


def get_merge_strategy(name: str) -> MergeStrategy:
    """Get merge strategy enum from string.

    Args:
        name: Merge strategy name.

    Returns:
        MergeStrategy enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_merge_strategy("add")
        <MergeStrategy.ADD: 'add'>
        >>> get_merge_strategy("cat")
        <MergeStrategy.CAT: 'cat'>

        >>> get_merge_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid merge strategy: invalid
    """
    for ms in MergeStrategy:
        if ms.value == name:
            return ms
    msg = f"invalid merge strategy: {name}"
    raise ValueError(msg)


def get_target_modules_list(preset: str) -> tuple[str, ...]:
    """Get list of target modules for a preset.

    Args:
        preset: Preset name ("attention", "mlp", "all").

    Returns:
        Tuple of module names.

    Raises:
        ValueError: If preset is not valid.

    Examples:
        >>> modules = get_target_modules_list("attention")
        >>> "q_proj" in modules
        True
        >>> "gate_proj" in modules
        False

        >>> modules = get_target_modules_list("mlp")
        >>> "down_proj" in modules
        True

        >>> get_target_modules_list("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: preset must be one of...
    """
    if preset not in VALID_TARGET_MODULES:
        msg = f"preset must be one of {VALID_TARGET_MODULES}, got '{preset}'"
        raise ValueError(msg)

    if preset == "attention":
        return tuple(sorted(ATTENTION_MODULES))
    elif preset == "mlp":
        return tuple(sorted(MLP_MODULES))
    else:  # all
        return tuple(sorted(ALL_MODULES))


def create_adapter_stats(
    adapter_config: AdapterConfig,
) -> AdapterStats:
    """Create adapter statistics from configuration.

    Args:
        adapter_config: Adapter configuration.

    Returns:
        AdapterStats with computed statistics.

    Raises:
        ValueError: If adapter_config is None.

    Examples:
        >>> config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        >>> stats = create_adapter_stats(config)
        >>> stats.adapter_type
        <AdapterType.LORA: 'lora'>
        >>> stats.trainable_percent < 1.0
        True

        >>> create_adapter_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: adapter_config cannot be None
    """
    if adapter_config is None:
        msg = "adapter_config cannot be None"
        raise ValueError(msg)

    trainable_pct = (
        adapter_config.trainable_params / adapter_config.base_model_params
    ) * 100

    bits = 16
    if adapter_config.qlora_config:
        bits = adapter_config.qlora_config.bits

    saved_mb, _ = estimate_memory_savings(
        adapter_type=adapter_config.adapter_type.value,
        base_model_params=adapter_config.base_model_params,
        trainable_params=adapter_config.trainable_params,
        bits=bits,
    )

    return AdapterStats(
        adapter_type=adapter_config.adapter_type,
        trainable_params=adapter_config.trainable_params,
        total_params=adapter_config.base_model_params,
        trainable_percent=trainable_pct,
        memory_saved_mb=saved_mb,
    )


def get_peft_config_dict(adapter_config: AdapterConfig) -> dict[str, Any]:
    """Convert AdapterConfig to PEFT-compatible dict.

    Args:
        adapter_config: Adapter configuration.

    Returns:
        Dictionary for PEFT configuration.

    Raises:
        ValueError: If adapter_config is None.

    Examples:
        >>> config = create_adapter_config(adapter_type="lora", r=8, alpha=16)
        >>> peft_dict = get_peft_config_dict(config)
        >>> peft_dict["r"]
        8
        >>> peft_dict["lora_alpha"]
        16

        >>> get_peft_config_dict(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: adapter_config cannot be None
    """
    if adapter_config is None:
        msg = "adapter_config cannot be None"
        raise ValueError(msg)

    result: dict[str, Any] = {
        "peft_type": adapter_config.adapter_type.value.upper(),
    }

    if adapter_config.lora_config:
        lora = adapter_config.lora_config
        result.update(
            {
                "r": lora.r,
                "lora_alpha": lora.alpha,
                "lora_dropout": lora.dropout,
                "target_modules": list(lora.target_modules),
                "bias": lora.bias,
                "use_rslora": lora.use_rslora,
                "use_dora": lora.use_dora,
            }
        )

    if adapter_config.adalora_config:
        adalora = adapter_config.adalora_config
        result.update(
            {
                "init_r": adalora.init_r,
                "target_r": adalora.target_r,
                "beta1": adalora.beta1,
                "beta2": adalora.beta2,
                "tinit": adalora.tinit,
                "tfinal": adalora.tfinal,
                "deltaT": adalora.delta_t,
            }
        )

    if adapter_config.ia3_config:
        ia3 = adapter_config.ia3_config
        result.update(
            {
                "target_modules": list(ia3.target_modules),
                "feedforward_modules": list(ia3.feedforward_modules),
                "init_ia3_weights": ia3.init_weights,
            }
        )

    if adapter_config.prefix_config:
        prefix = adapter_config.prefix_config
        result.update(
            {
                "num_virtual_tokens": prefix.num_virtual_tokens,
                "encoder_hidden_size": prefix.encoder_hidden_size,
                "prefix_projection": prefix.prefix_projection,
            }
        )

    return result
