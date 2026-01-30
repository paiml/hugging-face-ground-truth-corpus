"""Context length extension techniques for transformer models.

This module provides utilities for extending context length in transformer models,
including RoPE scaling, sliding window attention, ALiBi, and streaming approaches.

Examples:
    >>> from hf_gtc.inference.context import create_context_config
    >>> config = create_context_config(max_length=8192)
    >>> config.max_length
    8192
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class ExtensionMethod(Enum):
    """Context length extension method.

    Attributes:
        ROPE_SCALING: RoPE (Rotary Position Embedding) scaling techniques.
        ALIBI: Attention with Linear Biases.
        SLIDING_WINDOW: Sliding window attention.
        LANDMARK: Landmark attention for long contexts.
        STREAMING: Streaming attention for infinite context.

    Examples:
        >>> ExtensionMethod.ROPE_SCALING.value
        'rope_scaling'
        >>> ExtensionMethod.ALIBI.value
        'alibi'
    """

    ROPE_SCALING = "rope_scaling"
    ALIBI = "alibi"
    SLIDING_WINDOW = "sliding_window"
    LANDMARK = "landmark"
    STREAMING = "streaming"


VALID_EXTENSION_METHODS = frozenset(m.value for m in ExtensionMethod)


class RoPEScalingType(Enum):
    """RoPE scaling type for context extension.

    Attributes:
        LINEAR: Linear scaling of position embeddings.
        DYNAMIC: Dynamic NTK-aware scaling.
        YARN: YaRN (Yet another RoPE extensioN) scaling.
        NTK: Neural Tangent Kernel aware scaling.

    Examples:
        >>> RoPEScalingType.LINEAR.value
        'linear'
        >>> RoPEScalingType.YARN.value
        'yarn'
    """

    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    NTK = "ntk"


VALID_ROPE_SCALING_TYPES = frozenset(t.value for t in RoPEScalingType)


class AttentionPattern(Enum):
    """Attention pattern type for context handling.

    Attributes:
        FULL: Full quadratic attention.
        LOCAL: Local sliding window attention.
        GLOBAL_LOCAL: Combination of global and local attention.
        DILATED: Dilated attention pattern.

    Examples:
        >>> AttentionPattern.FULL.value
        'full'
        >>> AttentionPattern.GLOBAL_LOCAL.value
        'global_local'
    """

    FULL = "full"
    LOCAL = "local"
    GLOBAL_LOCAL = "global_local"
    DILATED = "dilated"


VALID_ATTENTION_PATTERNS = frozenset(p.value for p in AttentionPattern)


# Type aliases for string literal types
ExtensionMethodStr = Literal[
    "rope_scaling", "alibi", "sliding_window", "landmark", "streaming"
]
RoPEScalingTypeStr = Literal["linear", "dynamic", "yarn", "ntk"]
AttentionPatternStr = Literal["full", "local", "global_local", "dilated"]


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    """Configuration for RoPE scaling.

    Attributes:
        scaling_type: Type of RoPE scaling to apply.
        scaling_factor: Factor to scale positions by.
        original_max_length: Original model's maximum context length.

    Examples:
        >>> config = RoPEConfig(
        ...     scaling_type=RoPEScalingType.LINEAR,
        ...     scaling_factor=2.0,
        ...     original_max_length=4096,
        ... )
        >>> config.scaling_factor
        2.0
        >>> config.scaling_type
        <RoPEScalingType.LINEAR: 'linear'>
    """

    scaling_type: RoPEScalingType
    scaling_factor: float
    original_max_length: int


@dataclass(frozen=True, slots=True)
class SlidingWindowConfig:
    """Configuration for sliding window attention.

    Attributes:
        window_size: Size of the attention window.
        sink_tokens: Number of sink tokens at the beginning.
        overlap: Overlap between adjacent windows.

    Examples:
        >>> config = SlidingWindowConfig(
        ...     window_size=4096,
        ...     sink_tokens=4,
        ...     overlap=128,
        ... )
        >>> config.window_size
        4096
        >>> config.sink_tokens
        4
    """

    window_size: int
    sink_tokens: int
    overlap: int


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Configuration for context length extension.

    Attributes:
        extension_method: Method used for context extension.
        rope_config: Optional RoPE configuration.
        window_config: Optional sliding window configuration.
        max_length: Maximum target context length.

    Examples:
        >>> config = ContextConfig(
        ...     extension_method=ExtensionMethod.ROPE_SCALING,
        ...     rope_config=RoPEConfig(
        ...         scaling_type=RoPEScalingType.LINEAR,
        ...         scaling_factor=2.0,
        ...         original_max_length=4096,
        ...     ),
        ...     window_config=None,
        ...     max_length=8192,
        ... )
        >>> config.max_length
        8192
        >>> config.extension_method
        <ExtensionMethod.ROPE_SCALING: 'rope_scaling'>
    """

    extension_method: ExtensionMethod
    rope_config: RoPEConfig | None
    window_config: SlidingWindowConfig | None
    max_length: int


@dataclass(frozen=True, slots=True)
class ContextStats:
    """Statistics for context length extension.

    Attributes:
        effective_length: Effective context length achieved.
        memory_usage_mb: Memory usage in megabytes.
        attention_sparsity: Sparsity of attention pattern (0.0 to 1.0).

    Examples:
        >>> stats = ContextStats(
        ...     effective_length=8192,
        ...     memory_usage_mb=1024.0,
        ...     attention_sparsity=0.75,
        ... )
        >>> stats.effective_length
        8192
        >>> stats.attention_sparsity
        0.75
    """

    effective_length: int
    memory_usage_mb: float
    attention_sparsity: float


def validate_rope_config(config: RoPEConfig) -> None:
    """Validate RoPE configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = RoPEConfig(
        ...     scaling_type=RoPEScalingType.LINEAR,
        ...     scaling_factor=2.0,
        ...     original_max_length=4096,
        ... )
        >>> validate_rope_config(config)  # No error

        >>> bad_config = RoPEConfig(
        ...     scaling_type=RoPEScalingType.LINEAR,
        ...     scaling_factor=0.0,
        ...     original_max_length=4096,
        ... )
        >>> validate_rope_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scaling_factor must be positive
    """
    if config.scaling_factor <= 0:
        msg = f"scaling_factor must be positive, got {config.scaling_factor}"
        raise ValueError(msg)

    if config.original_max_length <= 0:
        msg = f"original_max_length must be positive, got {config.original_max_length}"
        raise ValueError(msg)


def validate_sliding_window_config(config: SlidingWindowConfig) -> None:
    """Validate sliding window configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SlidingWindowConfig(
        ...     window_size=4096,
        ...     sink_tokens=4,
        ...     overlap=128,
        ... )
        >>> validate_sliding_window_config(config)  # No error

        >>> bad_config = SlidingWindowConfig(
        ...     window_size=0,
        ...     sink_tokens=4,
        ...     overlap=128,
        ... )
        >>> validate_sliding_window_config(bad_config)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be positive
    """
    if config.window_size <= 0:
        msg = f"window_size must be positive, got {config.window_size}"
        raise ValueError(msg)

    if config.sink_tokens < 0:
        msg = f"sink_tokens cannot be negative, got {config.sink_tokens}"
        raise ValueError(msg)

    if config.overlap < 0:
        msg = f"overlap cannot be negative, got {config.overlap}"
        raise ValueError(msg)

    if config.overlap >= config.window_size:
        msg = (
            f"overlap must be less than window_size, got overlap={config.overlap} "
            f"with window_size={config.window_size}"
        )
        raise ValueError(msg)


def validate_context_config(config: ContextConfig) -> None:
    """Validate context configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ContextConfig(
        ...     extension_method=ExtensionMethod.ROPE_SCALING,
        ...     rope_config=RoPEConfig(
        ...         scaling_type=RoPEScalingType.LINEAR,
        ...         scaling_factor=2.0,
        ...         original_max_length=4096,
        ...     ),
        ...     window_config=None,
        ...     max_length=8192,
        ... )
        >>> validate_context_config(config)  # No error

        >>> bad_config = ContextConfig(
        ...     extension_method=ExtensionMethod.ROPE_SCALING,
        ...     rope_config=None,
        ...     window_config=None,
        ...     max_length=8192,
        ... )
        >>> validate_context_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: rope_config is required for rope_scaling method
    """
    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)

    if config.extension_method == ExtensionMethod.ROPE_SCALING:
        if config.rope_config is None:
            msg = "rope_config is required for rope_scaling method"
            raise ValueError(msg)
        validate_rope_config(config.rope_config)

    if config.extension_method == ExtensionMethod.SLIDING_WINDOW:
        if config.window_config is None:
            msg = "window_config is required for sliding_window method"
            raise ValueError(msg)
        validate_sliding_window_config(config.window_config)


def validate_context_stats(stats: ContextStats) -> None:
    """Validate context statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = ContextStats(
        ...     effective_length=8192,
        ...     memory_usage_mb=1024.0,
        ...     attention_sparsity=0.75,
        ... )
        >>> validate_context_stats(stats)  # No error

        >>> bad_stats = ContextStats(
        ...     effective_length=-1,
        ...     memory_usage_mb=1024.0,
        ...     attention_sparsity=0.75,
        ... )
        >>> validate_context_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: effective_length cannot be negative
    """
    if stats.effective_length < 0:
        msg = f"effective_length cannot be negative, got {stats.effective_length}"
        raise ValueError(msg)

    if stats.memory_usage_mb < 0:
        msg = f"memory_usage_mb cannot be negative, got {stats.memory_usage_mb}"
        raise ValueError(msg)

    if not 0.0 <= stats.attention_sparsity <= 1.0:
        msg = (
            f"attention_sparsity must be between 0.0 and 1.0, "
            f"got {stats.attention_sparsity}"
        )
        raise ValueError(msg)


def create_rope_config(
    scaling_type: RoPEScalingTypeStr = "linear",
    scaling_factor: float = 2.0,
    original_max_length: int = 4096,
) -> RoPEConfig:
    """Create a RoPE configuration.

    Args:
        scaling_type: Type of RoPE scaling. Defaults to "linear".
        scaling_factor: Factor to scale positions by. Defaults to 2.0.
        original_max_length: Original model's max context length. Defaults to 4096.

    Returns:
        RoPEConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_rope_config(scaling_factor=4.0)
        >>> config.scaling_factor
        4.0

        >>> config = create_rope_config(scaling_type="yarn")
        >>> config.scaling_type
        <RoPEScalingType.YARN: 'yarn'>

        >>> create_rope_config(scaling_factor=0.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scaling_factor must be positive
    """
    if scaling_type not in VALID_ROPE_SCALING_TYPES:
        msg = (
            f"scaling_type must be one of {VALID_ROPE_SCALING_TYPES}, "
            f"got '{scaling_type}'"
        )
        raise ValueError(msg)

    config = RoPEConfig(
        scaling_type=RoPEScalingType(scaling_type),
        scaling_factor=scaling_factor,
        original_max_length=original_max_length,
    )
    validate_rope_config(config)
    return config


def create_sliding_window_config(
    window_size: int = 4096,
    sink_tokens: int = 4,
    overlap: int = 128,
) -> SlidingWindowConfig:
    """Create a sliding window configuration.

    Args:
        window_size: Size of the attention window. Defaults to 4096.
        sink_tokens: Number of sink tokens. Defaults to 4.
        overlap: Overlap between adjacent windows. Defaults to 128.

    Returns:
        SlidingWindowConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_sliding_window_config(window_size=8192)
        >>> config.window_size
        8192

        >>> config = create_sliding_window_config(sink_tokens=8)
        >>> config.sink_tokens
        8

        >>> create_sliding_window_config(window_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: window_size must be positive
    """
    config = SlidingWindowConfig(
        window_size=window_size,
        sink_tokens=sink_tokens,
        overlap=overlap,
    )
    validate_sliding_window_config(config)
    return config


def create_context_config(
    extension_method: ExtensionMethodStr = "rope_scaling",
    rope_config: RoPEConfig | None = None,
    window_config: SlidingWindowConfig | None = None,
    max_length: int = 8192,
) -> ContextConfig:
    """Create a context configuration.

    Args:
        extension_method: Method for context extension. Defaults to "rope_scaling".
        rope_config: Optional RoPE configuration.
        window_config: Optional sliding window configuration.
        max_length: Maximum target context length. Defaults to 8192.

    Returns:
        ContextConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_context_config(max_length=16384)
        >>> config.max_length
        16384

        >>> config = create_context_config(extension_method="alibi")
        >>> config.extension_method
        <ExtensionMethod.ALIBI: 'alibi'>

        >>> create_context_config(max_length=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    if extension_method not in VALID_EXTENSION_METHODS:
        msg = (
            f"extension_method must be one of {VALID_EXTENSION_METHODS}, "
            f"got '{extension_method}'"
        )
        raise ValueError(msg)

    method = ExtensionMethod(extension_method)

    # Create default configs if needed
    final_rope_config = rope_config
    final_window_config = window_config

    if method == ExtensionMethod.ROPE_SCALING and rope_config is None:
        final_rope_config = create_rope_config()

    if method == ExtensionMethod.SLIDING_WINDOW and window_config is None:
        final_window_config = create_sliding_window_config()

    config = ContextConfig(
        extension_method=method,
        rope_config=final_rope_config,
        window_config=final_window_config,
        max_length=max_length,
    )
    validate_context_config(config)
    return config


def create_context_stats(
    effective_length: int = 0,
    memory_usage_mb: float = 0.0,
    attention_sparsity: float = 0.0,
) -> ContextStats:
    """Create context statistics.

    Args:
        effective_length: Effective context length achieved. Defaults to 0.
        memory_usage_mb: Memory usage in megabytes. Defaults to 0.0.
        attention_sparsity: Sparsity of attention (0.0 to 1.0). Defaults to 0.0.

    Returns:
        ContextStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_context_stats(effective_length=8192)
        >>> stats.effective_length
        8192

        >>> stats = create_context_stats(attention_sparsity=0.75)
        >>> stats.attention_sparsity
        0.75

        >>> create_context_stats(attention_sparsity=1.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: attention_sparsity must be between 0.0 and 1.0
    """
    stats = ContextStats(
        effective_length=effective_length,
        memory_usage_mb=memory_usage_mb,
        attention_sparsity=attention_sparsity,
    )
    validate_context_stats(stats)
    return stats


def list_extension_methods() -> list[str]:
    """List available context extension methods.

    Returns:
        Sorted list of extension method names.

    Examples:
        >>> methods = list_extension_methods()
        >>> "rope_scaling" in methods
        True
        >>> "sliding_window" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_EXTENSION_METHODS)


def list_rope_scaling_types() -> list[str]:
    """List available RoPE scaling types.

    Returns:
        Sorted list of RoPE scaling type names.

    Examples:
        >>> types = list_rope_scaling_types()
        >>> "linear" in types
        True
        >>> "yarn" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ROPE_SCALING_TYPES)


def list_attention_patterns() -> list[str]:
    """List available attention patterns.

    Returns:
        Sorted list of attention pattern names.

    Examples:
        >>> patterns = list_attention_patterns()
        >>> "full" in patterns
        True
        >>> "local" in patterns
        True
        >>> patterns == sorted(patterns)
        True
    """
    return sorted(VALID_ATTENTION_PATTERNS)


def get_extension_method(name: str) -> ExtensionMethod:
    """Get an extension method by name.

    Args:
        name: Name of the extension method.

    Returns:
        The corresponding ExtensionMethod enum value.

    Raises:
        ValueError: If name is not a valid extension method.

    Examples:
        >>> get_extension_method("rope_scaling")
        <ExtensionMethod.ROPE_SCALING: 'rope_scaling'>
        >>> get_extension_method("alibi")
        <ExtensionMethod.ALIBI: 'alibi'>

        >>> get_extension_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown extension method
    """
    if name not in VALID_EXTENSION_METHODS:
        msg = f"Unknown extension method: '{name}'. Valid: {VALID_EXTENSION_METHODS}"
        raise ValueError(msg)
    return ExtensionMethod(name)


def get_rope_scaling_type(name: str) -> RoPEScalingType:
    """Get a RoPE scaling type by name.

    Args:
        name: Name of the RoPE scaling type.

    Returns:
        The corresponding RoPEScalingType enum value.

    Raises:
        ValueError: If name is not a valid RoPE scaling type.

    Examples:
        >>> get_rope_scaling_type("linear")
        <RoPEScalingType.LINEAR: 'linear'>
        >>> get_rope_scaling_type("yarn")
        <RoPEScalingType.YARN: 'yarn'>

        >>> get_rope_scaling_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown RoPE scaling type
    """
    if name not in VALID_ROPE_SCALING_TYPES:
        msg = f"Unknown RoPE scaling type: '{name}'. Valid: {VALID_ROPE_SCALING_TYPES}"
        raise ValueError(msg)
    return RoPEScalingType(name)


def get_attention_pattern(name: str) -> AttentionPattern:
    """Get an attention pattern by name.

    Args:
        name: Name of the attention pattern.

    Returns:
        The corresponding AttentionPattern enum value.

    Raises:
        ValueError: If name is not a valid attention pattern.

    Examples:
        >>> get_attention_pattern("full")
        <AttentionPattern.FULL: 'full'>
        >>> get_attention_pattern("local")
        <AttentionPattern.LOCAL: 'local'>

        >>> get_attention_pattern("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unknown attention pattern
    """
    if name not in VALID_ATTENTION_PATTERNS:
        msg = f"Unknown attention pattern: '{name}'. Valid: {VALID_ATTENTION_PATTERNS}"
        raise ValueError(msg)
    return AttentionPattern(name)


def calculate_effective_length(
    config: ContextConfig,
) -> int:
    """Calculate effective context length for a configuration.

    Args:
        config: Context configuration.

    Returns:
        Effective context length achieved.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> rope_config = RoPEConfig(
        ...     scaling_type=RoPEScalingType.LINEAR,
        ...     scaling_factor=2.0,
        ...     original_max_length=4096,
        ... )
        >>> config = ContextConfig(
        ...     extension_method=ExtensionMethod.ROPE_SCALING,
        ...     rope_config=rope_config,
        ...     window_config=None,
        ...     max_length=8192,
        ... )
        >>> calculate_effective_length(config)
        8192

        >>> window_config = SlidingWindowConfig(
        ...     window_size=4096,
        ...     sink_tokens=4,
        ...     overlap=128,
        ... )
        >>> config = ContextConfig(
        ...     extension_method=ExtensionMethod.SLIDING_WINDOW,
        ...     rope_config=None,
        ...     window_config=window_config,
        ...     max_length=1000000,
        ... )
        >>> calculate_effective_length(config)
        1000000
    """
    validate_context_config(config)

    # For most methods, the effective length is the max_length
    # For sliding window, it's theoretically unlimited but capped at max_length
    return config.max_length


def estimate_memory_scaling(
    original_length: int,
    target_length: int,
    attention_pattern: AttentionPatternStr = "full",
    window_size: int | None = None,
) -> float:
    """Estimate memory scaling factor for context extension.

    Args:
        original_length: Original context length.
        target_length: Target context length.
        attention_pattern: Attention pattern type. Defaults to "full".
        window_size: Window size for local attention. Required for non-full patterns.

    Returns:
        Memory scaling factor (relative to original).

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> scale = estimate_memory_scaling(4096, 8192, attention_pattern="full")
        >>> scale
        4.0

        >>> scale = estimate_memory_scaling(
        ...     4096, 16384, attention_pattern="local", window_size=4096
        ... )
        >>> scale == 4.0
        True

        >>> estimate_memory_scaling(0, 8192)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: original_length must be positive
    """
    if original_length <= 0:
        msg = f"original_length must be positive, got {original_length}"
        raise ValueError(msg)

    if target_length <= 0:
        msg = f"target_length must be positive, got {target_length}"
        raise ValueError(msg)

    if attention_pattern not in VALID_ATTENTION_PATTERNS:
        msg = (
            f"attention_pattern must be one of {VALID_ATTENTION_PATTERNS}, "
            f"got '{attention_pattern}'"
        )
        raise ValueError(msg)

    if attention_pattern == "full":
        # Full attention scales quadratically with sequence length
        return (target_length / original_length) ** 2

    # For local/sliding window attention
    if window_size is None:
        msg = "window_size is required for non-full attention patterns"
        raise ValueError(msg)

    if window_size <= 0:
        msg = f"window_size must be positive, got {window_size}"
        raise ValueError(msg)

    if attention_pattern == "local":
        # Local attention scales linearly (window_size * seq_length)
        # Normalized to original quadratic: (target * window) / (original^2)
        original_memory = original_length**2
        target_memory = target_length * min(window_size, target_length)
        return target_memory / original_memory

    if attention_pattern == "global_local":
        # Global-local: full attention for some tokens + local for rest
        # Assume ~10% global tokens + local for the rest
        global_tokens = max(1, target_length // 10)
        local_tokens = target_length - global_tokens
        original_memory = original_length**2
        local_window = min(window_size, local_tokens)
        target_memory = global_tokens * target_length + local_tokens * local_window
        return target_memory / original_memory

    # Dilated attention
    # Dilated pattern reduces effective interactions
    dilation_factor = max(1, target_length // window_size)
    original_memory = original_length**2
    target_memory = (target_length * target_length) / dilation_factor
    return target_memory / original_memory


def calculate_attention_complexity(
    seq_length: int,
    attention_pattern: AttentionPatternStr = "full",
    window_size: int | None = None,
    num_heads: int = 32,
) -> int:
    """Calculate attention computation complexity (FLOPs approximation).

    Args:
        seq_length: Sequence length.
        attention_pattern: Attention pattern type. Defaults to "full".
        window_size: Window size for local attention.
        num_heads: Number of attention heads. Defaults to 32.

    Returns:
        Approximate FLOPs for attention computation.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> flops = calculate_attention_complexity(4096, attention_pattern="full")
        >>> flops > 0
        True

        >>> flops_full = calculate_attention_complexity(8192, attention_pattern="full")
        >>> flops_local = calculate_attention_complexity(
        ...     8192, attention_pattern="local", window_size=4096
        ... )
        >>> flops_local < flops_full
        True

        >>> calculate_attention_complexity(0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: seq_length must be positive
    """
    if seq_length <= 0:
        msg = f"seq_length must be positive, got {seq_length}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if attention_pattern not in VALID_ATTENTION_PATTERNS:
        msg = (
            f"attention_pattern must be one of {VALID_ATTENTION_PATTERNS}, "
            f"got '{attention_pattern}'"
        )
        raise ValueError(msg)

    # Base multiplier: 2 (Q*K and Attn*V) * num_heads
    base_multiplier = 2 * num_heads

    if attention_pattern == "full":
        # Full attention: O(n^2)
        return base_multiplier * seq_length * seq_length

    if window_size is None:
        msg = "window_size is required for non-full attention patterns"
        raise ValueError(msg)

    if window_size <= 0:
        msg = f"window_size must be positive, got {window_size}"
        raise ValueError(msg)

    effective_window = min(window_size, seq_length)

    if attention_pattern == "local":
        # Local attention: O(n * w)
        return base_multiplier * seq_length * effective_window

    if attention_pattern == "global_local":
        # Global-local: some full + some local
        global_tokens = max(1, seq_length // 10)
        local_tokens = seq_length - global_tokens
        global_flops = global_tokens * seq_length
        local_flops = local_tokens * effective_window
        return base_multiplier * (global_flops + local_flops)

    # Dilated attention
    dilation_factor = max(1, seq_length // effective_window)
    effective_interactions = seq_length * seq_length // dilation_factor
    return base_multiplier * effective_interactions


def validate_position_ids(
    position_ids: list[int],
    max_length: int,
    allow_duplicates: bool = True,
) -> bool:
    """Validate position IDs for context extension.

    Args:
        position_ids: List of position IDs to validate.
        max_length: Maximum allowed position value.
        allow_duplicates: Whether to allow duplicate position IDs. Defaults to True.

    Returns:
        True if position IDs are valid.

    Raises:
        ValueError: If position IDs are invalid.

    Examples:
        >>> validate_position_ids([0, 1, 2, 3], max_length=4096)
        True

        >>> validate_position_ids([0, 1, 1, 2], max_length=4096, allow_duplicates=True)
        True

        >>> validate_position_ids([0, 1, 1, 2], max_length=4096, allow_duplicates=False)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Duplicate position IDs found

        >>> validate_position_ids([0, 1, 5000], max_length=4096)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Position ID 5000 exceeds max_length 4096

        >>> validate_position_ids([-1, 0, 1], max_length=4096)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Position ID cannot be negative
    """
    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    for pos_id in position_ids:
        if pos_id < 0:
            msg = f"Position ID cannot be negative, got {pos_id}"
            raise ValueError(msg)
        if pos_id >= max_length:
            msg = f"Position ID {pos_id} exceeds max_length {max_length}"
            raise ValueError(msg)

    if not allow_duplicates and len(position_ids) != len(set(position_ids)):
        msg = "Duplicate position IDs found when allow_duplicates is False"
        raise ValueError(msg)

    return True


def format_context_stats(stats: ContextStats) -> str:
    """Format context statistics as a human-readable string.

    Args:
        stats: Context statistics to format.

    Returns:
        Formatted string representation.

    Examples:
        >>> stats = ContextStats(
        ...     effective_length=8192,
        ...     memory_usage_mb=1024.0,
        ...     attention_sparsity=0.75,
        ... )
        >>> formatted = format_context_stats(stats)
        >>> "Effective Length: 8192" in formatted
        True
        >>> "Memory Usage: 1024.00 MB" in formatted
        True

        >>> empty_stats = ContextStats(0, 0.0, 0.0)
        >>> "Effective Length: 0" in format_context_stats(empty_stats)
        True
    """
    lines = [
        f"Effective Length: {stats.effective_length}",
        f"Memory Usage: {stats.memory_usage_mb:.2f} MB",
        f"Attention Sparsity: {stats.attention_sparsity * 100:.1f}%",
    ]
    return "\n".join(lines)


def get_recommended_context_config(
    target_length: int,
    original_max_length: int = 4096,
    memory_constraint_gb: float | None = None,
) -> ContextConfig:
    """Get recommended context configuration based on target length.

    Args:
        target_length: Target context length.
        original_max_length: Original model's max context length. Defaults to 4096.
        memory_constraint_gb: Optional memory constraint in gigabytes.

    Returns:
        Recommended ContextConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_context_config(target_length=8192)
        >>> config.max_length
        8192
        >>> config.extension_method
        <ExtensionMethod.ROPE_SCALING: 'rope_scaling'>

        >>> config = get_recommended_context_config(
        ...     target_length=1000000,
        ...     memory_constraint_gb=8.0,
        ... )
        >>> config.extension_method
        <ExtensionMethod.SLIDING_WINDOW: 'sliding_window'>

        >>> get_recommended_context_config(target_length=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_length must be positive
    """
    if target_length <= 0:
        msg = f"target_length must be positive, got {target_length}"
        raise ValueError(msg)

    if original_max_length <= 0:
        msg = f"original_max_length must be positive, got {original_max_length}"
        raise ValueError(msg)

    if memory_constraint_gb is not None and memory_constraint_gb <= 0:
        msg = f"memory_constraint_gb must be positive, got {memory_constraint_gb}"
        raise ValueError(msg)

    scaling_factor = target_length / original_max_length

    # Determine best method based on scaling factor and memory constraints
    if memory_constraint_gb is not None:
        # Estimate if full attention would exceed memory
        # Very rough estimate: assume 2 bytes per attention score
        attention_memory_gb = (target_length**2 * 2) / (1024**3)

        if attention_memory_gb > memory_constraint_gb:
            # Need sparse attention - use sliding window
            # Choose window size based on memory constraint
            # window_size * target_length * 2 bytes <= memory
            max_window = int((memory_constraint_gb * (1024**3)) / (target_length * 2))
            window_size = min(max(1024, max_window), target_length)

            window_config = SlidingWindowConfig(
                window_size=window_size,
                sink_tokens=4,
                overlap=min(128, window_size // 4),
            )
            return ContextConfig(
                extension_method=ExtensionMethod.SLIDING_WINDOW,
                rope_config=None,
                window_config=window_config,
                max_length=target_length,
            )

    # Choose RoPE scaling method based on scaling factor
    if scaling_factor <= 2.0:
        scaling_type = RoPEScalingType.LINEAR
    elif scaling_factor <= 8.0:
        scaling_type = RoPEScalingType.DYNAMIC
    else:
        scaling_type = RoPEScalingType.YARN

    rope_config = RoPEConfig(
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
        original_max_length=original_max_length,
    )

    return ContextConfig(
        extension_method=ExtensionMethod.ROPE_SCALING,
        rope_config=rope_config,
        window_config=None,
        max_length=target_length,
    )
