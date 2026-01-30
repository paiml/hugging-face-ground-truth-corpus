"""Synthetic data generation and self-instruct patterns.

This module provides utilities for synthetic data generation including
self-instruct, evol-instruct, backtranslation, and paraphrase patterns
for creating high-quality training data.

Examples:
    >>> from hf_gtc.preprocessing.synthetic import GenerationMethod
    >>> GenerationMethod.SELF_INSTRUCT.value
    'self_instruct'
    >>> from hf_gtc.preprocessing.synthetic import create_self_instruct_config
    >>> config = create_self_instruct_config(num_instructions=100)
    >>> config.num_instructions
    100
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class GenerationMethod(Enum):
    """Methods for generating synthetic data.

    Attributes:
        SELF_INSTRUCT: Self-instruction generation from seed tasks.
        EVOL_INSTRUCT: Evolutionary instruction complexity increase.
        BACKTRANSLATION: Back-translation augmentation.
        PARAPHRASE: Paraphrase-based generation.
        TEMPLATE: Template-based generation.

    Examples:
        >>> GenerationMethod.SELF_INSTRUCT.value
        'self_instruct'
        >>> GenerationMethod.EVOL_INSTRUCT.value
        'evol_instruct'
    """

    SELF_INSTRUCT = "self_instruct"
    EVOL_INSTRUCT = "evol_instruct"
    BACKTRANSLATION = "backtranslation"
    PARAPHRASE = "paraphrase"
    TEMPLATE = "template"


VALID_GENERATION_METHODS = frozenset(m.value for m in GenerationMethod)


class DiversityStrategy(Enum):
    """Strategies for ensuring diversity in generated data.

    Attributes:
        TOPIC: Diversity based on topic variation.
        STYLE: Diversity based on style variation.
        DIFFICULTY: Diversity based on difficulty levels.
        FORMAT: Diversity based on output format variation.

    Examples:
        >>> DiversityStrategy.TOPIC.value
        'topic'
        >>> DiversityStrategy.STYLE.value
        'style'
    """

    TOPIC = "topic"
    STYLE = "style"
    DIFFICULTY = "difficulty"
    FORMAT = "format"


VALID_DIVERSITY_STRATEGIES = frozenset(s.value for s in DiversityStrategy)


class QualityFilter(Enum):
    """Filters for assessing synthetic data quality.

    Attributes:
        PERPLEXITY: Filter based on language model perplexity.
        LENGTH: Filter based on text length constraints.
        SIMILARITY: Filter based on similarity to existing data.
        TOXICITY: Filter based on toxicity detection.

    Examples:
        >>> QualityFilter.PERPLEXITY.value
        'perplexity'
        >>> QualityFilter.SIMILARITY.value
        'similarity'
    """

    PERPLEXITY = "perplexity"
    LENGTH = "length"
    SIMILARITY = "similarity"
    TOXICITY = "toxicity"


VALID_QUALITY_FILTERS = frozenset(f.value for f in QualityFilter)


@dataclass(frozen=True, slots=True)
class SelfInstructConfig:
    """Configuration for self-instruction generation.

    Attributes:
        num_instructions: Number of instructions to generate.
        seed_tasks: Tuple of seed tasks to bootstrap generation.
        diversity_strategy: Strategy for ensuring diversity.
        temperature: Sampling temperature for generation.
        max_retries: Maximum retries for failed generations.

    Examples:
        >>> config = SelfInstructConfig(
        ...     num_instructions=100,
        ...     seed_tasks=("Write a poem", "Solve a math problem"),
        ...     diversity_strategy=DiversityStrategy.TOPIC,
        ...     temperature=0.7,
        ...     max_retries=3,
        ... )
        >>> config.num_instructions
        100
        >>> config.diversity_strategy
        <DiversityStrategy.TOPIC: 'topic'>
    """

    num_instructions: int
    seed_tasks: tuple[str, ...]
    diversity_strategy: DiversityStrategy
    temperature: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class EvolInstructConfig:
    """Configuration for evolutionary instruction generation.

    Attributes:
        evolution_steps: Number of evolution steps per instruction.
        mutation_types: Tuple of mutation types to apply.
        complexity_increase: Factor for complexity increase per step.
        preserve_semantics: Whether to preserve semantic meaning.
        max_length: Maximum length of evolved instructions.

    Examples:
        >>> config = EvolInstructConfig(
        ...     evolution_steps=3,
        ...     mutation_types=("add_constraint", "increase_depth"),
        ...     complexity_increase=1.2,
        ...     preserve_semantics=True,
        ...     max_length=512,
        ... )
        >>> config.evolution_steps
        3
        >>> config.complexity_increase
        1.2
    """

    evolution_steps: int
    mutation_types: tuple[str, ...]
    complexity_increase: float
    preserve_semantics: bool
    max_length: int


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Configuration for synthetic data generation pipeline.

    Attributes:
        method: Generation method to use.
        quality_filters: Tuple of quality filters to apply.
        dedup_threshold: Similarity threshold for deduplication.
        target_count: Target number of samples to generate.
        min_length: Minimum text length for generated samples.
        max_length: Maximum text length for generated samples.

    Examples:
        >>> config = SyntheticConfig(
        ...     method=GenerationMethod.SELF_INSTRUCT,
        ...     quality_filters=(QualityFilter.PERPLEXITY, QualityFilter.LENGTH),
        ...     dedup_threshold=0.9,
        ...     target_count=1000,
        ...     min_length=10,
        ...     max_length=512,
        ... )
        >>> config.method
        <GenerationMethod.SELF_INSTRUCT: 'self_instruct'>
        >>> config.dedup_threshold
        0.9
    """

    method: GenerationMethod
    quality_filters: tuple[QualityFilter, ...]
    dedup_threshold: float
    target_count: int
    min_length: int
    max_length: int


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    """A synthetic data sample with metadata.

    Attributes:
        text: The generated text content.
        source_method: Method used to generate this sample.
        quality_score: Overall quality score (0-1).
        diversity_score: Diversity contribution score (0-1).

    Examples:
        >>> sample = SyntheticSample(
        ...     text="Write a function to sort a list",
        ...     source_method=GenerationMethod.SELF_INSTRUCT,
        ...     quality_score=0.85,
        ...     diversity_score=0.72,
        ... )
        >>> sample.quality_score
        0.85
    """

    text: str
    source_method: GenerationMethod
    quality_score: float
    diversity_score: float


@dataclass(frozen=True, slots=True)
class GenerationStats:
    """Statistics from synthetic data generation.

    Attributes:
        total_generated: Total number of samples generated.
        passed_quality: Number passing quality filters.
        passed_dedup: Number passing deduplication.
        final_count: Final number of samples retained.
        avg_quality_score: Average quality score.
        avg_diversity_score: Average diversity score.

    Examples:
        >>> stats = GenerationStats(
        ...     total_generated=1500,
        ...     passed_quality=1200,
        ...     passed_dedup=1000,
        ...     final_count=1000,
        ...     avg_quality_score=0.82,
        ...     avg_diversity_score=0.75,
        ... )
        >>> stats.total_generated
        1500
        >>> stats.final_count
        1000
    """

    total_generated: int
    passed_quality: int
    passed_dedup: int
    final_count: int
    avg_quality_score: float
    avg_diversity_score: float


def validate_self_instruct_config(config: SelfInstructConfig) -> None:
    """Validate self-instruction configuration.

    Args:
        config: SelfInstructConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_instructions is not positive.
        ValueError: If seed_tasks is empty.
        ValueError: If temperature is not in valid range.
        ValueError: If max_retries is not positive.

    Examples:
        >>> config = SelfInstructConfig(
        ...     num_instructions=100,
        ...     seed_tasks=("task1",),
        ...     diversity_strategy=DiversityStrategy.TOPIC,
        ...     temperature=0.7,
        ...     max_retries=3,
        ... )
        >>> validate_self_instruct_config(config)  # No error

        >>> validate_self_instruct_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SelfInstructConfig(
        ...     num_instructions=0,
        ...     seed_tasks=("task1",),
        ...     diversity_strategy=DiversityStrategy.TOPIC,
        ...     temperature=0.7,
        ...     max_retries=3,
        ... )
        >>> validate_self_instruct_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_instructions must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_instructions <= 0:
        msg = f"num_instructions must be positive, got {config.num_instructions}"
        raise ValueError(msg)

    if not config.seed_tasks:
        msg = "seed_tasks cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.temperature <= 2.0:
        msg = f"temperature must be between 0 and 2, got {config.temperature}"
        raise ValueError(msg)

    if config.max_retries <= 0:
        msg = f"max_retries must be positive, got {config.max_retries}"
        raise ValueError(msg)


def validate_evol_instruct_config(config: EvolInstructConfig) -> None:
    """Validate evolutionary instruction configuration.

    Args:
        config: EvolInstructConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If evolution_steps is not positive.
        ValueError: If mutation_types is empty.
        ValueError: If complexity_increase is not >= 1.0.
        ValueError: If max_length is not positive.

    Examples:
        >>> config = EvolInstructConfig(
        ...     evolution_steps=3,
        ...     mutation_types=("add_constraint",),
        ...     complexity_increase=1.2,
        ...     preserve_semantics=True,
        ...     max_length=512,
        ... )
        >>> validate_evol_instruct_config(config)  # No error

        >>> validate_evol_instruct_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = EvolInstructConfig(
        ...     evolution_steps=0,
        ...     mutation_types=("add_constraint",),
        ...     complexity_increase=1.2,
        ...     preserve_semantics=True,
        ...     max_length=512,
        ... )
        >>> validate_evol_instruct_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: evolution_steps must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.evolution_steps <= 0:
        msg = f"evolution_steps must be positive, got {config.evolution_steps}"
        raise ValueError(msg)

    if not config.mutation_types:
        msg = "mutation_types cannot be empty"
        raise ValueError(msg)

    if config.complexity_increase < 1.0:
        msg = (
            f"complexity_increase must be >= 1.0, got {config.complexity_increase}"
        )
        raise ValueError(msg)

    if config.max_length <= 0:
        msg = f"max_length must be positive, got {config.max_length}"
        raise ValueError(msg)


def validate_synthetic_config(config: SyntheticConfig) -> None:
    """Validate synthetic data generation configuration.

    Args:
        config: SyntheticConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If quality_filters is empty.
        ValueError: If dedup_threshold is not in [0, 1].
        ValueError: If target_count is not positive.
        ValueError: If min_length >= max_length.

    Examples:
        >>> config = SyntheticConfig(
        ...     method=GenerationMethod.SELF_INSTRUCT,
        ...     quality_filters=(QualityFilter.PERPLEXITY,),
        ...     dedup_threshold=0.9,
        ...     target_count=1000,
        ...     min_length=10,
        ...     max_length=512,
        ... )
        >>> validate_synthetic_config(config)  # No error

        >>> validate_synthetic_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SyntheticConfig(
        ...     method=GenerationMethod.SELF_INSTRUCT,
        ...     quality_filters=(),
        ...     dedup_threshold=0.9,
        ...     target_count=1000,
        ...     min_length=10,
        ...     max_length=512,
        ... )
        >>> validate_synthetic_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quality_filters cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.quality_filters:
        msg = "quality_filters cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.dedup_threshold <= 1.0:
        msg = (
            f"dedup_threshold must be between 0 and 1, got {config.dedup_threshold}"
        )
        raise ValueError(msg)

    if config.target_count <= 0:
        msg = f"target_count must be positive, got {config.target_count}"
        raise ValueError(msg)

    if config.min_length >= config.max_length:
        msg = (
            f"min_length ({config.min_length}) must be less than "
            f"max_length ({config.max_length})"
        )
        raise ValueError(msg)


def create_self_instruct_config(
    num_instructions: int = 100,
    seed_tasks: tuple[str, ...] | None = None,
    diversity_strategy: str = "topic",
    temperature: float = 0.7,
    max_retries: int = 3,
) -> SelfInstructConfig:
    """Create a self-instruction configuration.

    Args:
        num_instructions: Number of instructions to generate. Defaults to 100.
        seed_tasks: Seed tasks for bootstrapping. Defaults to sample tasks.
        diversity_strategy: Diversity strategy name. Defaults to "topic".
        temperature: Generation temperature. Defaults to 0.7.
        max_retries: Maximum retries. Defaults to 3.

    Returns:
        SelfInstructConfig with the specified settings.

    Raises:
        ValueError: If diversity_strategy is not valid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_self_instruct_config(num_instructions=50)
        >>> config.num_instructions
        50
        >>> config.diversity_strategy
        <DiversityStrategy.TOPIC: 'topic'>

        >>> create_self_instruct_config(diversity_strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: diversity_strategy must be one of

        >>> create_self_instruct_config(num_instructions=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_instructions must be positive
    """
    if diversity_strategy not in VALID_DIVERSITY_STRATEGIES:
        msg = (
            f"diversity_strategy must be one of {VALID_DIVERSITY_STRATEGIES}, "
            f"got '{diversity_strategy}'"
        )
        raise ValueError(msg)

    effective_seed_tasks = seed_tasks or (
        "Write a short story about a robot",
        "Explain how photosynthesis works",
        "Create a recipe for chocolate cake",
        "Solve a quadratic equation",
        "Describe the water cycle",
    )

    config = SelfInstructConfig(
        num_instructions=num_instructions,
        seed_tasks=effective_seed_tasks,
        diversity_strategy=DiversityStrategy(diversity_strategy),
        temperature=temperature,
        max_retries=max_retries,
    )
    validate_self_instruct_config(config)
    return config


def create_evol_instruct_config(
    evolution_steps: int = 3,
    mutation_types: tuple[str, ...] | None = None,
    complexity_increase: float = 1.2,
    preserve_semantics: bool = True,
    max_length: int = 512,
) -> EvolInstructConfig:
    """Create an evolutionary instruction configuration.

    Args:
        evolution_steps: Number of evolution steps. Defaults to 3.
        mutation_types: Types of mutations to apply. Defaults to standard types.
        complexity_increase: Complexity increase factor. Defaults to 1.2.
        preserve_semantics: Whether to preserve semantics. Defaults to True.
        max_length: Maximum instruction length. Defaults to 512.

    Returns:
        EvolInstructConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_evol_instruct_config(evolution_steps=5)
        >>> config.evolution_steps
        5
        >>> config.preserve_semantics
        True

        >>> create_evol_instruct_config(evolution_steps=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: evolution_steps must be positive

        >>> create_evol_instruct_config(complexity_increase=0.5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: complexity_increase must be >= 1.0
    """
    effective_mutation_types = mutation_types or (
        "add_constraint",
        "increase_depth",
        "concretize",
        "add_reasoning",
    )

    config = EvolInstructConfig(
        evolution_steps=evolution_steps,
        mutation_types=effective_mutation_types,
        complexity_increase=complexity_increase,
        preserve_semantics=preserve_semantics,
        max_length=max_length,
    )
    validate_evol_instruct_config(config)
    return config


def create_synthetic_config(
    method: str = "self_instruct",
    quality_filters: tuple[str, ...] = ("perplexity", "length"),
    dedup_threshold: float = 0.9,
    target_count: int = 1000,
    min_length: int = 10,
    max_length: int = 512,
) -> SyntheticConfig:
    """Create a synthetic data generation configuration.

    Args:
        method: Generation method name. Defaults to "self_instruct".
        quality_filters: Quality filter names. Defaults to ("perplexity", "length").
        dedup_threshold: Deduplication threshold. Defaults to 0.9.
        target_count: Target sample count. Defaults to 1000.
        min_length: Minimum text length. Defaults to 10.
        max_length: Maximum text length. Defaults to 512.

    Returns:
        SyntheticConfig with the specified settings.

    Raises:
        ValueError: If method is not valid.
        ValueError: If any quality_filter is not valid.
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_synthetic_config(method="evol_instruct")
        >>> config.method
        <GenerationMethod.EVOL_INSTRUCT: 'evol_instruct'>

        >>> create_synthetic_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of

        >>> create_synthetic_config(quality_filters=("invalid",))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: quality_filter must be one of
    """
    if method not in VALID_GENERATION_METHODS:
        msg = f"method must be one of {VALID_GENERATION_METHODS}, got '{method}'"
        raise ValueError(msg)

    for qf in quality_filters:
        if qf not in VALID_QUALITY_FILTERS:
            msg = f"quality_filter must be one of {VALID_QUALITY_FILTERS}, got '{qf}'"
            raise ValueError(msg)

    filter_enums = tuple(QualityFilter(qf) for qf in quality_filters)

    config = SyntheticConfig(
        method=GenerationMethod(method),
        quality_filters=filter_enums,
        dedup_threshold=dedup_threshold,
        target_count=target_count,
        min_length=min_length,
        max_length=max_length,
    )
    validate_synthetic_config(config)
    return config


def list_generation_methods() -> list[str]:
    """List all available generation methods.

    Returns:
        Sorted list of generation method names.

    Examples:
        >>> methods = list_generation_methods()
        >>> "self_instruct" in methods
        True
        >>> "evol_instruct" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_GENERATION_METHODS)


def get_generation_method(name: str) -> GenerationMethod:
    """Get GenerationMethod enum from string name.

    Args:
        name: Name of the generation method.

    Returns:
        Corresponding GenerationMethod enum value.

    Raises:
        ValueError: If name is not a valid generation method.

    Examples:
        >>> get_generation_method("self_instruct")
        <GenerationMethod.SELF_INSTRUCT: 'self_instruct'>

        >>> get_generation_method("evol_instruct")
        <GenerationMethod.EVOL_INSTRUCT: 'evol_instruct'>

        >>> get_generation_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid generation method: invalid
    """
    if name not in VALID_GENERATION_METHODS:
        msg = f"invalid generation method: {name}"
        raise ValueError(msg)

    return GenerationMethod(name)


def list_diversity_strategies() -> list[str]:
    """List all available diversity strategies.

    Returns:
        Sorted list of diversity strategy names.

    Examples:
        >>> strategies = list_diversity_strategies()
        >>> "topic" in strategies
        True
        >>> "style" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_DIVERSITY_STRATEGIES)


def get_diversity_strategy(name: str) -> DiversityStrategy:
    """Get DiversityStrategy enum from string name.

    Args:
        name: Name of the diversity strategy.

    Returns:
        Corresponding DiversityStrategy enum value.

    Raises:
        ValueError: If name is not a valid diversity strategy.

    Examples:
        >>> get_diversity_strategy("topic")
        <DiversityStrategy.TOPIC: 'topic'>

        >>> get_diversity_strategy("style")
        <DiversityStrategy.STYLE: 'style'>

        >>> get_diversity_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid diversity strategy: invalid
    """
    if name not in VALID_DIVERSITY_STRATEGIES:
        msg = f"invalid diversity strategy: {name}"
        raise ValueError(msg)

    return DiversityStrategy(name)


def list_quality_filters() -> list[str]:
    """List all available quality filters.

    Returns:
        Sorted list of quality filter names.

    Examples:
        >>> filters = list_quality_filters()
        >>> "perplexity" in filters
        True
        >>> "similarity" in filters
        True
        >>> filters == sorted(filters)
        True
    """
    return sorted(VALID_QUALITY_FILTERS)


def get_quality_filter(name: str) -> QualityFilter:
    """Get QualityFilter enum from string name.

    Args:
        name: Name of the quality filter.

    Returns:
        Corresponding QualityFilter enum value.

    Raises:
        ValueError: If name is not a valid quality filter.

    Examples:
        >>> get_quality_filter("perplexity")
        <QualityFilter.PERPLEXITY: 'perplexity'>

        >>> get_quality_filter("similarity")
        <QualityFilter.SIMILARITY: 'similarity'>

        >>> get_quality_filter("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid quality filter: invalid
    """
    if name not in VALID_QUALITY_FILTERS:
        msg = f"invalid quality filter: {name}"
        raise ValueError(msg)

    return QualityFilter(name)


def estimate_generation_cost(
    config: SyntheticConfig,
    *,
    tokens_per_sample: int = 256,
    cost_per_1k_tokens: float = 0.002,
) -> dict[str, float]:
    """Estimate the cost of synthetic data generation.

    Args:
        config: Synthetic generation configuration.
        tokens_per_sample: Estimated tokens per generated sample. Defaults to 256.
        cost_per_1k_tokens: Cost per 1000 tokens. Defaults to 0.002.

    Returns:
        Dictionary with cost estimates including total_tokens, total_cost,
        and cost_per_sample.

    Raises:
        ValueError: If config is None.
        ValueError: If tokens_per_sample is not positive.
        ValueError: If cost_per_1k_tokens is negative.

    Examples:
        >>> config = create_synthetic_config(target_count=1000)
        >>> costs = estimate_generation_cost(config)
        >>> costs["total_tokens"] > 0
        True
        >>> costs["total_cost"] >= 0
        True

        >>> estimate_generation_cost(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> estimate_generation_cost(config, tokens_per_sample=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tokens_per_sample must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if tokens_per_sample <= 0:
        msg = f"tokens_per_sample must be positive, got {tokens_per_sample}"
        raise ValueError(msg)

    if cost_per_1k_tokens < 0:
        msg = f"cost_per_1k_tokens cannot be negative, got {cost_per_1k_tokens}"
        raise ValueError(msg)

    validate_synthetic_config(config)

    # Estimate generation overhead based on method
    overhead_factor = {
        GenerationMethod.SELF_INSTRUCT: 1.5,
        GenerationMethod.EVOL_INSTRUCT: 2.0,
        GenerationMethod.BACKTRANSLATION: 2.5,
        GenerationMethod.PARAPHRASE: 1.3,
        GenerationMethod.TEMPLATE: 1.1,
    }.get(config.method, 1.5)

    # Estimate quality filter overhead
    filter_overhead = 1.0 + (len(config.quality_filters) * 0.1)

    # Calculate total tokens needed
    effective_samples = int(config.target_count * overhead_factor * filter_overhead)
    total_tokens = effective_samples * tokens_per_sample

    # Calculate costs
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    cost_per_sample = total_cost / config.target_count if config.target_count > 0 else 0

    return {
        "total_tokens": float(total_tokens),
        "total_cost": total_cost,
        "cost_per_sample": cost_per_sample,
        "estimated_samples_needed": float(effective_samples),
    }


def calculate_diversity_score(
    texts: Sequence[str],
    *,
    ngram_size: int = 3,
) -> float:
    """Calculate diversity score for a collection of texts.

    Uses n-gram diversity to measure how varied the generated content is.
    Higher scores indicate more diverse text collections.

    Args:
        texts: Sequence of texts to evaluate.
        ngram_size: Size of n-grams to use. Defaults to 3.

    Returns:
        Diversity score between 0 (identical) and 1 (maximally diverse).

    Raises:
        ValueError: If texts is None.
        ValueError: If ngram_size is not positive.

    Examples:
        >>> diverse_texts = ["Hello world", "Goodbye moon", "Testing here"]
        >>> score = calculate_diversity_score(diverse_texts)
        >>> 0.0 <= score <= 1.0
        True

        >>> identical_texts = ["same text", "same text", "same text"]
        >>> score = calculate_diversity_score(identical_texts)
        >>> score < 0.5
        True

        >>> calculate_diversity_score([])
        0.0

        >>> calculate_diversity_score(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: texts cannot be None
    """
    if texts is None:
        msg = "texts cannot be None"
        raise ValueError(msg)

    if ngram_size <= 0:
        msg = f"ngram_size must be positive, got {ngram_size}"
        raise ValueError(msg)

    if len(texts) < 2:
        return 0.0

    # Collect all n-grams
    all_ngrams: set[tuple[str, ...]] = set()
    total_ngrams = 0

    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i : i + ngram_size])
            all_ngrams.add(ngram)
            total_ngrams += 1

    if total_ngrams == 0:
        return 0.0

    # Diversity = unique n-grams / total n-grams
    diversity = len(all_ngrams) / total_ngrams

    return min(1.0, diversity)


def filter_synthetic_samples(
    samples: Sequence[SyntheticSample],
    config: SyntheticConfig,
) -> list[SyntheticSample]:
    """Filter synthetic samples based on quality criteria.

    Args:
        samples: Sequence of synthetic samples to filter.
        config: Synthetic generation configuration with filter settings.

    Returns:
        List of samples that pass all quality filters.

    Raises:
        ValueError: If samples is None.
        ValueError: If config is None.

    Examples:
        >>> sample1 = SyntheticSample(
        ...     text="This is a good sample with enough length",
        ...     source_method=GenerationMethod.SELF_INSTRUCT,
        ...     quality_score=0.9,
        ...     diversity_score=0.8,
        ... )
        >>> sample2 = SyntheticSample(
        ...     text="Short",
        ...     source_method=GenerationMethod.SELF_INSTRUCT,
        ...     quality_score=0.3,
        ...     diversity_score=0.2,
        ... )
        >>> config = create_synthetic_config(min_length=10, max_length=100)
        >>> filtered = filter_synthetic_samples([sample1, sample2], config)
        >>> len(filtered) == 1
        True

        >>> filter_synthetic_samples(None, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: samples cannot be None
    """
    if samples is None:
        msg = "samples cannot be None"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_synthetic_config(config)

    filtered: list[SyntheticSample] = []

    for sample in samples:
        # Length filter
        if QualityFilter.LENGTH in config.quality_filters:
            text_len = len(sample.text)
            if text_len < config.min_length or text_len > config.max_length:
                continue

        # Quality score threshold (using perplexity as proxy)
        if (
            QualityFilter.PERPLEXITY in config.quality_filters
            and sample.quality_score < 0.5
        ):
            continue

        # Toxicity filter (placeholder - real implementation would use model)
        if QualityFilter.TOXICITY in config.quality_filters:
            # Placeholder: accept all for now
            pass

        filtered.append(sample)

    return filtered


def validate_synthetic_quality(
    samples: Sequence[SyntheticSample],
    *,
    min_quality_score: float = 0.5,
    min_diversity_score: float = 0.3,
) -> dict[str, object]:
    """Validate quality of synthetic samples.

    Args:
        samples: Sequence of synthetic samples to validate.
        min_quality_score: Minimum acceptable quality score. Defaults to 0.5.
        min_diversity_score: Minimum acceptable diversity score. Defaults to 0.3.

    Returns:
        Dictionary with validation results including is_valid, avg_quality,
        avg_diversity, and issues.

    Raises:
        ValueError: If samples is None.
        ValueError: If min_quality_score is not in [0, 1].
        ValueError: If min_diversity_score is not in [0, 1].

    Examples:
        >>> samples = [
        ...     SyntheticSample(
        ...         text="Good sample",
        ...         source_method=GenerationMethod.SELF_INSTRUCT,
        ...         quality_score=0.8,
        ...         diversity_score=0.6,
        ...     ),
        ... ]
        >>> result = validate_synthetic_quality(samples)
        >>> result["is_valid"]
        True

        >>> validate_synthetic_quality(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: samples cannot be None
    """
    if samples is None:
        msg = "samples cannot be None"
        raise ValueError(msg)

    if not 0.0 <= min_quality_score <= 1.0:
        msg = f"min_quality_score must be between 0 and 1, got {min_quality_score}"
        raise ValueError(msg)

    if not 0.0 <= min_diversity_score <= 1.0:
        msg = f"min_diversity_score must be between 0 and 1, got {min_diversity_score}"
        raise ValueError(msg)

    if not samples:
        return {
            "is_valid": False,
            "avg_quality": 0.0,
            "avg_diversity": 0.0,
            "issues": ["No samples provided"],
        }

    issues: list[str] = []

    # Calculate averages
    total_quality = sum(s.quality_score for s in samples)
    total_diversity = sum(s.diversity_score for s in samples)
    avg_quality = total_quality / len(samples)
    avg_diversity = total_diversity / len(samples)

    # Check thresholds
    if avg_quality < min_quality_score:
        issues.append(
            f"Average quality score {avg_quality:.2f} below threshold "
            f"{min_quality_score:.2f}"
        )

    if avg_diversity < min_diversity_score:
        issues.append(
            f"Average diversity score {avg_diversity:.2f} below threshold "
            f"{min_diversity_score:.2f}"
        )

    # Check for low-quality individual samples
    low_quality_count = sum(1 for s in samples if s.quality_score < min_quality_score)
    if low_quality_count > len(samples) * 0.2:
        issues.append(
            f"{low_quality_count} samples ({low_quality_count/len(samples)*100:.1f}%) "
            f"below quality threshold"
        )

    return {
        "is_valid": len(issues) == 0,
        "avg_quality": avg_quality,
        "avg_diversity": avg_diversity,
        "issues": issues,
    }


def format_generation_stats(stats: GenerationStats) -> str:
    """Format generation statistics as a human-readable string.

    Args:
        stats: GenerationStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = GenerationStats(
        ...     total_generated=1500,
        ...     passed_quality=1200,
        ...     passed_dedup=1000,
        ...     final_count=1000,
        ...     avg_quality_score=0.82,
        ...     avg_diversity_score=0.75,
        ... )
        >>> formatted = format_generation_stats(stats)
        >>> "1,500" in formatted or "1500" in formatted
        True
        >>> "82" in formatted
        True

        >>> format_generation_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    quality_pass_rate = (
        stats.passed_quality / stats.total_generated * 100
        if stats.total_generated > 0
        else 0.0
    )

    dedup_rate = (
        (stats.passed_quality - stats.passed_dedup) / stats.passed_quality * 100
        if stats.passed_quality > 0
        else 0.0
    )

    lines = [
        "Synthetic Generation Statistics",
        "=" * 40,
        f"Total generated:     {stats.total_generated:,}",
        f"Passed quality:      {stats.passed_quality:,}",
        f"Passed dedup:        {stats.passed_dedup:,}",
        f"Final count:         {stats.final_count:,}",
        "",
        f"Quality pass rate:   {quality_pass_rate:.1f}%",
        f"Dedup removal rate:  {dedup_rate:.1f}%",
        "",
        f"Avg quality score:   {stats.avg_quality_score:.2f}",
        f"Avg diversity score: {stats.avg_diversity_score:.2f}",
    ]

    return "\n".join(lines)


def get_recommended_synthetic_config(
    use_case: str,
    *,
    target_count: int = 1000,
) -> SyntheticConfig:
    """Get recommended synthetic configuration based on use case.

    Args:
        use_case: Use case identifier (instruction_tuning, qa_generation,
            summarization, code_generation).
        target_count: Target number of samples. Defaults to 1000.

    Returns:
        Recommended SyntheticConfig for the use case.

    Raises:
        ValueError: If use_case is not recognized.
        ValueError: If target_count is not positive.

    Examples:
        >>> config = get_recommended_synthetic_config("instruction_tuning")
        >>> config.method
        <GenerationMethod.SELF_INSTRUCT: 'self_instruct'>

        >>> config = get_recommended_synthetic_config("code_generation")
        >>> config.method
        <GenerationMethod.EVOL_INSTRUCT: 'evol_instruct'>

        >>> get_recommended_synthetic_config("invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: use_case must be one of

        >>> get_recommended_synthetic_config("instruction_tuning", target_count=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_count must be positive
    """
    valid_use_cases = {
        "instruction_tuning",
        "qa_generation",
        "summarization",
        "code_generation",
    }

    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    if target_count <= 0:
        msg = f"target_count must be positive, got {target_count}"
        raise ValueError(msg)

    if use_case == "instruction_tuning":
        return create_synthetic_config(
            method="self_instruct",
            quality_filters=("perplexity", "length", "similarity"),
            dedup_threshold=0.85,
            target_count=target_count,
            min_length=20,
            max_length=1024,
        )

    if use_case == "qa_generation":
        return create_synthetic_config(
            method="template",
            quality_filters=("perplexity", "length"),
            dedup_threshold=0.9,
            target_count=target_count,
            min_length=10,
            max_length=512,
        )

    if use_case == "summarization":
        return create_synthetic_config(
            method="backtranslation",
            quality_filters=("perplexity", "length", "similarity"),
            dedup_threshold=0.8,
            target_count=target_count,
            min_length=50,
            max_length=2048,
        )

    # code_generation
    return create_synthetic_config(
        method="evol_instruct",
        quality_filters=("perplexity", "length", "toxicity"),
        dedup_threshold=0.9,
        target_count=target_count,
        min_length=30,
        max_length=4096,
    )


def _compute_text_hash(text: str) -> str:
    """Compute hash for a text string."""
    return hashlib.md5(text.lower().encode(), usedforsecurity=False).hexdigest()


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def deduplicate_samples(
    samples: Sequence[SyntheticSample],
    threshold: float = 0.9,
) -> list[SyntheticSample]:
    """Remove duplicate samples based on similarity threshold.

    Args:
        samples: Sequence of synthetic samples.
        threshold: Similarity threshold for deduplication. Defaults to 0.9.

    Returns:
        List of deduplicated samples.

    Raises:
        ValueError: If samples is None.
        ValueError: If threshold is not in [0, 1].

    Examples:
        >>> samples = [
        ...     SyntheticSample(
        ...         text="Hello world",
        ...         source_method=GenerationMethod.SELF_INSTRUCT,
        ...         quality_score=0.8,
        ...         diversity_score=0.7,
        ...     ),
        ...     SyntheticSample(
        ...         text="Hello world",
        ...         source_method=GenerationMethod.SELF_INSTRUCT,
        ...         quality_score=0.75,
        ...         diversity_score=0.6,
        ...     ),
        ... ]
        >>> deduped = deduplicate_samples(samples)
        >>> len(deduped) == 1
        True

        >>> deduplicate_samples(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: samples cannot be None
    """
    if samples is None:
        msg = "samples cannot be None"
        raise ValueError(msg)

    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be between 0 and 1, got {threshold}"
        raise ValueError(msg)

    if not samples:
        return []

    # Use hash-based exact dedup first
    seen_hashes: set[str] = set()
    unique_samples: list[SyntheticSample] = []

    for sample in samples:
        text_hash = _compute_text_hash(sample.text)
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_samples.append(sample)

    # If threshold < 1.0, also do similarity-based dedup
    if threshold < 1.0 and len(unique_samples) > 1:
        final_samples: list[SyntheticSample] = []

        for sample in unique_samples:
            is_duplicate = False
            for existing in final_samples:
                similarity = _calculate_text_similarity(sample.text, existing.text)
                if similarity >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_samples.append(sample)

        return final_samples

    return unique_samples


def compute_generation_stats(
    generated: Sequence[SyntheticSample],
    after_quality: Sequence[SyntheticSample],
    after_dedup: Sequence[SyntheticSample],
) -> GenerationStats:
    """Compute statistics from generation pipeline stages.

    Args:
        generated: All generated samples.
        after_quality: Samples after quality filtering.
        after_dedup: Samples after deduplication.

    Returns:
        GenerationStats with computed metrics.

    Raises:
        ValueError: If any argument is None.

    Examples:
        >>> gen = [
        ...     SyntheticSample("a", GenerationMethod.SELF_INSTRUCT, 0.8, 0.7),
        ...     SyntheticSample("b", GenerationMethod.SELF_INSTRUCT, 0.6, 0.5),
        ... ]
        >>> qual = [gen[0]]
        >>> dedup = [gen[0]]
        >>> stats = compute_generation_stats(gen, qual, dedup)
        >>> stats.total_generated
        2
        >>> stats.final_count
        1

        >>> compute_generation_stats(None, [], [])
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: generated cannot be None
    """
    if generated is None:
        msg = "generated cannot be None"
        raise ValueError(msg)

    if after_quality is None:
        msg = "after_quality cannot be None"
        raise ValueError(msg)

    if after_dedup is None:
        msg = "after_dedup cannot be None"
        raise ValueError(msg)

    avg_quality = (
        sum(s.quality_score for s in after_dedup) / len(after_dedup)
        if after_dedup
        else 0.0
    )

    avg_diversity = (
        sum(s.diversity_score for s in after_dedup) / len(after_dedup)
        if after_dedup
        else 0.0
    )

    return GenerationStats(
        total_generated=len(generated),
        passed_quality=len(after_quality),
        passed_dedup=len(after_dedup),
        final_count=len(after_dedup),
        avg_quality_score=avg_quality,
        avg_diversity_score=avg_diversity,
    )
