"""Text generation sampling utilities.

This module provides functions for configuring text generation
sampling strategies including top-k, top-p, beam search, and more.

Examples:
    >>> from hf_gtc.generation.sampling import create_sampling_config
    >>> config = create_sampling_config(temperature=0.7, top_p=0.9)
    >>> config.temperature
    0.7
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class SamplingStrategy(Enum):
    """Supported sampling strategies.

    Attributes:
        GREEDY: Greedy decoding (always pick highest probability).
        TOP_K: Sample from top-k most likely tokens.
        TOP_P: Sample from tokens with cumulative probability <= p.
        BEAM_SEARCH: Beam search decoding.
        CONTRASTIVE: Contrastive search decoding.
        DIVERSE_BEAM: Diverse beam search.

    Examples:
        >>> SamplingStrategy.GREEDY.value
        'greedy'
        >>> SamplingStrategy.TOP_P.value
        'top_p'
    """

    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM_SEARCH = "beam_search"
    CONTRASTIVE = "contrastive"
    DIVERSE_BEAM = "diverse_beam"


VALID_STRATEGIES = frozenset(s.value for s in SamplingStrategy)

# Stopping criteria types
StopCriteriaType = Literal["max_length", "max_time", "eos_token", "custom"]
VALID_STOP_CRITERIA = frozenset({"max_length", "max_time", "eos_token", "custom"})


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Configuration for text generation sampling.

    Attributes:
        strategy: Sampling strategy to use.
        temperature: Sampling temperature (higher = more random).
        top_k: Number of top tokens to consider for top-k sampling.
        top_p: Cumulative probability threshold for nucleus sampling.
        repetition_penalty: Penalty for repeating tokens.
        length_penalty: Penalty for sequence length in beam search.
        no_repeat_ngram_size: Size of n-grams that cannot repeat.

    Examples:
        >>> config = SamplingConfig(
        ...     strategy=SamplingStrategy.TOP_P,
        ...     temperature=0.7,
        ...     top_k=50,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1,
        ...     length_penalty=1.0,
        ...     no_repeat_ngram_size=3,
        ... )
        >>> config.temperature
        0.7
    """

    strategy: SamplingStrategy
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    length_penalty: float
    no_repeat_ngram_size: int


@dataclass(frozen=True, slots=True)
class BeamSearchConfig:
    """Configuration for beam search decoding.

    Attributes:
        num_beams: Number of beams for beam search.
        num_beam_groups: Number of groups for diverse beam search.
        diversity_penalty: Penalty for diversity in diverse beam search.
        early_stopping: Whether to stop when num_beams sentences finish.
        length_penalty: Exponential penalty for sequence length.

    Examples:
        >>> config = BeamSearchConfig(
        ...     num_beams=4,
        ...     num_beam_groups=1,
        ...     diversity_penalty=0.0,
        ...     early_stopping=True,
        ...     length_penalty=1.0,
        ... )
        >>> config.num_beams
        4
    """

    num_beams: int
    num_beam_groups: int
    diversity_penalty: float
    early_stopping: bool
    length_penalty: float


@dataclass(frozen=True, slots=True)
class GenerationConstraints:
    """Constraints for text generation.

    Attributes:
        max_length: Maximum total length (prompt + generation).
        max_new_tokens: Maximum new tokens to generate.
        min_length: Minimum total length.
        min_new_tokens: Minimum new tokens to generate.
        max_time: Maximum generation time in seconds.

    Examples:
        >>> constraints = GenerationConstraints(
        ...     max_length=512,
        ...     max_new_tokens=256,
        ...     min_length=10,
        ...     min_new_tokens=5,
        ...     max_time=30.0,
        ... )
        >>> constraints.max_new_tokens
        256
    """

    max_length: int
    max_new_tokens: int
    min_length: int
    min_new_tokens: int
    max_time: float


@dataclass(frozen=True, slots=True)
class StoppingCriteria:
    r"""Stopping criteria for generation.

    Attributes:
        stop_strings: Strings that trigger stopping.
        stop_token_ids: Token IDs that trigger stopping.
        criteria_type: Type of stopping criteria.

    Examples:
        >>> criteria = StoppingCriteria(
        ...     stop_strings=("\\n\\n", "###"),
        ...     stop_token_ids=(50256,),
        ...     criteria_type="eos_token",
        ... )
        >>> criteria.criteria_type
        'eos_token'
    """

    stop_strings: tuple[str, ...]
    stop_token_ids: tuple[int, ...]
    criteria_type: StopCriteriaType


@dataclass(frozen=True, slots=True)
class ContrastiveConfig:
    """Configuration for contrastive search.

    Attributes:
        penalty_alpha: Degeneration penalty (0 = greedy, 1 = full contrast).
        top_k: Number of candidates for contrastive search.

    Examples:
        >>> config = ContrastiveConfig(
        ...     penalty_alpha=0.6,
        ...     top_k=4,
        ... )
        >>> config.penalty_alpha
        0.6
    """

    penalty_alpha: float
    top_k: int


def validate_sampling_config(config: SamplingConfig) -> None:
    """Validate sampling configuration.

    Args:
        config: Sampling configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SamplingConfig(
        ...     strategy=SamplingStrategy.TOP_P,
        ...     temperature=0.7,
        ...     top_k=50,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1,
        ...     length_penalty=1.0,
        ...     no_repeat_ngram_size=3,
        ... )
        >>> validate_sampling_config(config)  # No error

        >>> bad_config = SamplingConfig(
        ...     strategy=SamplingStrategy.TOP_P,
        ...     temperature=-0.5,
        ...     top_k=50,
        ...     top_p=0.9,
        ...     repetition_penalty=1.0,
        ...     length_penalty=1.0,
        ...     no_repeat_ngram_size=0,
        ... )
        >>> validate_sampling_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be non-negative
    """
    if config.temperature < 0:
        msg = f"temperature must be non-negative, got {config.temperature}"
        raise ValueError(msg)

    if config.top_k < 0:
        msg = f"top_k must be non-negative, got {config.top_k}"
        raise ValueError(msg)

    if not 0.0 <= config.top_p <= 1.0:
        msg = f"top_p must be between 0.0 and 1.0, got {config.top_p}"
        raise ValueError(msg)

    if config.repetition_penalty < 1.0:
        msg = f"repetition_penalty must be >= 1.0, got {config.repetition_penalty}"
        raise ValueError(msg)

    if config.no_repeat_ngram_size < 0:
        msg = (
            f"no_repeat_ngram_size must be non-negative, "
            f"got {config.no_repeat_ngram_size}"
        )
        raise ValueError(msg)


def validate_beam_search_config(config: BeamSearchConfig) -> None:
    """Validate beam search configuration.

    Args:
        config: Beam search configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = BeamSearchConfig(4, 1, 0.0, True, 1.0)
        >>> validate_beam_search_config(config)  # No error

        >>> bad_config = BeamSearchConfig(0, 1, 0.0, True, 1.0)
        >>> validate_beam_search_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_beams must be positive
    """
    if config.num_beams <= 0:
        msg = f"num_beams must be positive, got {config.num_beams}"
        raise ValueError(msg)

    if config.num_beam_groups <= 0:
        msg = f"num_beam_groups must be positive, got {config.num_beam_groups}"
        raise ValueError(msg)

    if config.num_beams % config.num_beam_groups != 0:
        msg = (
            f"num_beams ({config.num_beams}) must be divisible by "
            f"num_beam_groups ({config.num_beam_groups})"
        )
        raise ValueError(msg)

    if config.diversity_penalty < 0:
        msg = f"diversity_penalty must be non-negative, got {config.diversity_penalty}"
        raise ValueError(msg)


def validate_generation_constraints(constraints: GenerationConstraints) -> None:
    """Validate generation constraints.

    Args:
        constraints: Generation constraints to validate.

    Raises:
        ValueError: If constraints are invalid.

    Examples:
        >>> constraints = GenerationConstraints(512, 256, 10, 5, 30.0)
        >>> validate_generation_constraints(constraints)  # No error

        >>> bad = GenerationConstraints(512, 256, 600, 5, 30.0)
        >>> validate_generation_constraints(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: min_length cannot exceed max_length
    """
    if constraints.max_length <= 0:
        msg = f"max_length must be positive, got {constraints.max_length}"
        raise ValueError(msg)

    if constraints.max_new_tokens <= 0:
        msg = f"max_new_tokens must be positive, got {constraints.max_new_tokens}"
        raise ValueError(msg)

    if constraints.min_length < 0:
        msg = f"min_length must be non-negative, got {constraints.min_length}"
        raise ValueError(msg)

    if constraints.min_length > constraints.max_length:
        msg = (
            f"min_length ({constraints.min_length}) cannot exceed "
            f"max_length ({constraints.max_length})"
        )
        raise ValueError(msg)

    if constraints.max_time <= 0:
        msg = f"max_time must be positive, got {constraints.max_time}"
        raise ValueError(msg)


def create_sampling_config(
    strategy: str = "top_p",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> SamplingConfig:
    """Create a sampling configuration.

    Args:
        strategy: Sampling strategy. Defaults to "top_p".
        temperature: Sampling temperature. Defaults to 1.0.
        top_k: Top-k value. Defaults to 50.
        top_p: Top-p (nucleus) value. Defaults to 1.0.
        repetition_penalty: Repetition penalty. Defaults to 1.0.
        length_penalty: Length penalty. Defaults to 1.0.
        no_repeat_ngram_size: N-gram size to prevent repeating. Defaults to 0.

    Returns:
        SamplingConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_sampling_config(temperature=0.7, top_p=0.9)
        >>> config.temperature
        0.7

        >>> config = create_sampling_config(strategy="greedy")
        >>> config.strategy
        <SamplingStrategy.GREEDY: 'greedy'>

        >>> create_sampling_config(temperature=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be non-negative
    """
    if strategy not in VALID_STRATEGIES:
        msg = f"strategy must be one of {VALID_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = SamplingConfig(
        strategy=SamplingStrategy(strategy),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    validate_sampling_config(config)
    return config


def create_beam_search_config(
    num_beams: int = 4,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    early_stopping: bool = True,
    length_penalty: float = 1.0,
) -> BeamSearchConfig:
    """Create a beam search configuration.

    Args:
        num_beams: Number of beams. Defaults to 4.
        num_beam_groups: Number of beam groups. Defaults to 1.
        diversity_penalty: Diversity penalty. Defaults to 0.0.
        early_stopping: Whether to stop early. Defaults to True.
        length_penalty: Length penalty. Defaults to 1.0.

    Returns:
        BeamSearchConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_beam_search_config(num_beams=8)
        >>> config.num_beams
        8

        >>> create_beam_search_config(num_beams=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_beams must be positive
    """
    config = BeamSearchConfig(
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        early_stopping=early_stopping,
        length_penalty=length_penalty,
    )
    validate_beam_search_config(config)
    return config


def create_generation_constraints(
    max_length: int = 512,
    max_new_tokens: int = 256,
    min_length: int = 0,
    min_new_tokens: int = 0,
    max_time: float = 60.0,
) -> GenerationConstraints:
    """Create generation constraints.

    Args:
        max_length: Maximum total length. Defaults to 512.
        max_new_tokens: Maximum new tokens. Defaults to 256.
        min_length: Minimum total length. Defaults to 0.
        min_new_tokens: Minimum new tokens. Defaults to 0.
        max_time: Maximum time in seconds. Defaults to 60.0.

    Returns:
        GenerationConstraints with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> constraints = create_generation_constraints(max_new_tokens=128)
        >>> constraints.max_new_tokens
        128

        >>> create_generation_constraints(max_length=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_length must be positive
    """
    constraints = GenerationConstraints(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        min_new_tokens=min_new_tokens,
        max_time=max_time,
    )
    validate_generation_constraints(constraints)
    return constraints


def create_stopping_criteria(
    stop_strings: tuple[str, ...] = (),
    stop_token_ids: tuple[int, ...] = (),
    criteria_type: StopCriteriaType = "max_length",
) -> StoppingCriteria:
    r"""Create stopping criteria.

    Args:
        stop_strings: Strings that trigger stopping. Defaults to ().
        stop_token_ids: Token IDs that trigger stopping. Defaults to ().
        criteria_type: Type of criteria. Defaults to "max_length".

    Returns:
        StoppingCriteria with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> criteria = create_stopping_criteria(stop_strings=("\\n\\n",))
        >>> criteria.stop_strings
        ('\\n\\n',)

        >>> create_stopping_criteria(criteria_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: criteria_type must be one of
    """
    if criteria_type not in VALID_STOP_CRITERIA:
        msg = (
            f"criteria_type must be one of {VALID_STOP_CRITERIA}, got '{criteria_type}'"
        )
        raise ValueError(msg)

    return StoppingCriteria(
        stop_strings=stop_strings,
        stop_token_ids=stop_token_ids,
        criteria_type=criteria_type,
    )


def create_contrastive_config(
    penalty_alpha: float = 0.6,
    top_k: int = 4,
) -> ContrastiveConfig:
    """Create contrastive search configuration.

    Args:
        penalty_alpha: Degeneration penalty. Defaults to 0.6.
        top_k: Number of candidates. Defaults to 4.

    Returns:
        ContrastiveConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_contrastive_config(penalty_alpha=0.5)
        >>> config.penalty_alpha
        0.5

        >>> create_contrastive_config(penalty_alpha=-1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: penalty_alpha must be between 0.0 and 1.0
    """
    if not 0.0 <= penalty_alpha <= 1.0:
        msg = f"penalty_alpha must be between 0.0 and 1.0, got {penalty_alpha}"
        raise ValueError(msg)

    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)

    return ContrastiveConfig(penalty_alpha=penalty_alpha, top_k=top_k)


def list_sampling_strategies() -> list[str]:
    """List supported sampling strategies.

    Returns:
        Sorted list of strategy names.

    Examples:
        >>> strategies = list_sampling_strategies()
        >>> "greedy" in strategies
        True
        >>> "top_p" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_STRATEGIES)


def get_sampling_strategy(name: str) -> SamplingStrategy:
    """Get sampling strategy from name.

    Args:
        name: Strategy name.

    Returns:
        SamplingStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_sampling_strategy("greedy")
        <SamplingStrategy.GREEDY: 'greedy'>

        >>> get_sampling_strategy("top_k")
        <SamplingStrategy.TOP_K: 'top_k'>

        >>> get_sampling_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if name not in VALID_STRATEGIES:
        msg = f"strategy must be one of {VALID_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return SamplingStrategy(name)


def get_recommended_config(task: str) -> SamplingConfig:
    """Get recommended sampling config for a task.

    Args:
        task: Task type ("creative", "factual", "code", "chat").

    Returns:
        Recommended SamplingConfig for the task.

    Raises:
        ValueError: If task is invalid.

    Examples:
        >>> config = get_recommended_config("creative")
        >>> config.temperature > 0.5
        True

        >>> config = get_recommended_config("factual")
        >>> config.temperature
        0.1

        >>> config = get_recommended_config("code")
        >>> config.temperature
        0.2
    """
    valid_tasks = {"creative", "factual", "code", "chat"}
    if task not in valid_tasks:
        msg = f"task must be one of {valid_tasks}, got '{task}'"
        raise ValueError(msg)

    configs = {
        "creative": create_sampling_config(
            strategy="top_p",
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
        ),
        "factual": create_sampling_config(
            strategy="greedy",
            temperature=0.1,
            top_p=1.0,
            repetition_penalty=1.0,
        ),
        "code": create_sampling_config(
            strategy="top_p",
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.0,
        ),
        "chat": create_sampling_config(
            strategy="top_p",
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        ),
    }
    return configs[task]


def calculate_effective_vocab_size(
    vocab_size: int,
    top_k: int,
    top_p: float,
) -> int:
    """Calculate effective vocabulary size after filtering.

    Args:
        vocab_size: Total vocabulary size.
        top_k: Top-k filter value (0 = no filter).
        top_p: Top-p filter value (1.0 = no filter).

    Returns:
        Estimated effective vocabulary size.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_effective_vocab_size(50000, 50, 1.0)
        50

        >>> calculate_effective_vocab_size(50000, 0, 0.9)
        45000

        >>> calculate_effective_vocab_size(50000, 100, 0.5)
        100
    """
    if vocab_size <= 0:
        msg = f"vocab_size must be positive, got {vocab_size}"
        raise ValueError(msg)

    if top_k < 0:
        msg = f"top_k must be non-negative, got {top_k}"
        raise ValueError(msg)

    if not 0.0 <= top_p <= 1.0:
        msg = f"top_p must be between 0.0 and 1.0, got {top_p}"
        raise ValueError(msg)

    # Apply top_k filter
    effective_size = min(top_k, vocab_size) if top_k > 0 else vocab_size

    # Apply top_p filter (rough estimate)
    if top_p < 1.0:
        top_p_estimate = int(vocab_size * top_p)
        effective_size = min(effective_size, top_p_estimate)

    return max(1, effective_size)


def estimate_generation_memory(
    batch_size: int,
    max_length: int,
    hidden_size: int,
    num_layers: int,
    num_beams: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Estimate memory usage for text generation.

    Args:
        batch_size: Batch size.
        max_length: Maximum sequence length.
        hidden_size: Model hidden dimension.
        num_layers: Number of transformer layers.
        num_beams: Number of beams (for beam search).
        dtype_bytes: Bytes per element. Defaults to 2 (FP16).

    Returns:
        Estimated memory in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> mem = estimate_generation_memory(1, 512, 768, 12)
        >>> mem > 0
        True

        >>> estimate_generation_memory(0, 512, 768, 12)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: batch_size must be positive
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if max_length <= 0:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if num_beams <= 0:
        msg = f"num_beams must be positive, got {num_beams}"
        raise ValueError(msg)

    # KV cache memory: 2 (K + V) * layers * batch * beams * seq * hidden * dtype
    kv_cache = 2 * num_layers * batch_size * num_beams * max_length * hidden_size
    return kv_cache * dtype_bytes
