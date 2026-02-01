"""Data sampling strategies for ML datasets.

This module provides utilities for data sampling including random,
stratified, importance, balanced, cluster, and curriculum sampling
strategies for machine learning preprocessing pipelines.

Examples:
    >>> from hf_gtc.preprocessing.sampling import SamplingMethod, WeightingScheme
    >>> SamplingMethod.RANDOM.value
    'random'
    >>> WeightingScheme.UNIFORM.value
    'uniform'
    >>> from hf_gtc.preprocessing.sampling import BalancingStrategy
    >>> BalancingStrategy.OVERSAMPLE.value
    'oversample'
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from hf_gtc._validation import validate_not_none


class SamplingMethod(Enum):
    """Methods for sampling data from datasets.

    Attributes:
        RANDOM: Simple random sampling.
        STRATIFIED: Stratified sampling preserving class distribution.
        IMPORTANCE: Importance-weighted sampling.
        BALANCED: Balanced sampling across classes.
        CLUSTER: Cluster-based sampling.
        CURRICULUM: Curriculum learning-based sampling.

    Examples:
        >>> SamplingMethod.RANDOM.value
        'random'
        >>> SamplingMethod.STRATIFIED.value
        'stratified'
        >>> SamplingMethod.IMPORTANCE.value
        'importance'
    """

    RANDOM = "random"
    STRATIFIED = "stratified"
    IMPORTANCE = "importance"
    BALANCED = "balanced"
    CLUSTER = "cluster"
    CURRICULUM = "curriculum"


VALID_SAMPLING_METHODS = frozenset(m.value for m in SamplingMethod)


class WeightingScheme(Enum):
    """Schemes for weighting samples in importance sampling.

    Attributes:
        UNIFORM: Equal weight for all samples.
        INVERSE_FREQUENCY: Weight inversely proportional to class frequency.
        SQRT_INVERSE: Square root of inverse frequency.
        CUSTOM: Custom user-defined weights.

    Examples:
        >>> WeightingScheme.UNIFORM.value
        'uniform'
        >>> WeightingScheme.INVERSE_FREQUENCY.value
        'inverse_frequency'
        >>> WeightingScheme.SQRT_INVERSE.value
        'sqrt_inverse'
    """

    UNIFORM = "uniform"
    INVERSE_FREQUENCY = "inverse_frequency"
    SQRT_INVERSE = "sqrt_inverse"
    CUSTOM = "custom"


VALID_WEIGHTING_SCHEMES = frozenset(w.value for w in WeightingScheme)


class BalancingStrategy(Enum):
    """Strategies for balancing class distributions.

    Attributes:
        OVERSAMPLE: Oversample minority classes.
        UNDERSAMPLE: Undersample majority classes.
        SMOTE: Synthetic Minority Over-sampling Technique.
        CLASS_WEIGHT: Use class weights during training.

    Examples:
        >>> BalancingStrategy.OVERSAMPLE.value
        'oversample'
        >>> BalancingStrategy.UNDERSAMPLE.value
        'undersample'
        >>> BalancingStrategy.SMOTE.value
        'smote'
    """

    OVERSAMPLE = "oversample"
    UNDERSAMPLE = "undersample"
    SMOTE = "smote"
    CLASS_WEIGHT = "class_weight"


VALID_BALANCING_STRATEGIES = frozenset(b.value for b in BalancingStrategy)


@dataclass(frozen=True, slots=True)
class StratifiedConfig:
    """Configuration for stratified sampling.

    Attributes:
        column: Column name to stratify by.
        preserve_distribution: Whether to preserve exact class distribution.

    Examples:
        >>> config = StratifiedConfig(column="label", preserve_distribution=True)
        >>> config.column
        'label'
        >>> config.preserve_distribution
        True
    """

    column: str
    preserve_distribution: bool


@dataclass(frozen=True, slots=True)
class ImportanceConfig:
    """Configuration for importance sampling.

    Attributes:
        weights_column: Column containing sample weights (or None for computed).
        temperature: Temperature for softmax over weights (higher = more uniform).
        normalize: Whether to normalize weights to sum to 1.

    Examples:
        >>> config = ImportanceConfig(
        ...     weights_column="weight",
        ...     temperature=1.0,
        ...     normalize=True,
        ... )
        >>> config.temperature
        1.0
        >>> config.normalize
        True
    """

    weights_column: str | None
    temperature: float
    normalize: bool


@dataclass(frozen=True, slots=True)
class BalancedConfig:
    """Configuration for balanced sampling.

    Attributes:
        strategy: Balancing strategy to use.
        target_ratio: Target ratio for class balance (1.0 = perfect balance).
        random_state: Random seed for reproducibility.

    Examples:
        >>> config = BalancedConfig(
        ...     strategy=BalancingStrategy.OVERSAMPLE,
        ...     target_ratio=1.0,
        ...     random_state=42,
        ... )
        >>> config.strategy
        <BalancingStrategy.OVERSAMPLE: 'oversample'>
        >>> config.target_ratio
        1.0
    """

    strategy: BalancingStrategy
    target_ratio: float
    random_state: int | None


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Combined configuration for data sampling operations.

    Attributes:
        method: Sampling method to use.
        stratified_config: Configuration for stratified sampling.
        importance_config: Configuration for importance sampling.
        balanced_config: Configuration for balanced sampling.
        sample_size: Number of samples to draw (or fraction if < 1.0).

    Examples:
        >>> config = SamplingConfig(
        ...     method=SamplingMethod.RANDOM,
        ...     stratified_config=None,
        ...     importance_config=None,
        ...     balanced_config=None,
        ...     sample_size=1000,
        ... )
        >>> config.method
        <SamplingMethod.RANDOM: 'random'>
        >>> config.sample_size
        1000
    """

    method: SamplingMethod
    stratified_config: StratifiedConfig | None
    importance_config: ImportanceConfig | None
    balanced_config: BalancedConfig | None
    sample_size: int | float


@dataclass(frozen=True, slots=True)
class SamplingStats:
    """Statistics from sampling operations.

    Attributes:
        original_size: Original dataset size.
        sampled_size: Size after sampling.
        class_distribution: Distribution of classes in sampled data.
        effective_ratio: Effective sampling ratio achieved.

    Examples:
        >>> stats = SamplingStats(
        ...     original_size=10000,
        ...     sampled_size=1000,
        ...     class_distribution={"A": 500, "B": 500},
        ...     effective_ratio=0.1,
        ... )
        >>> stats.original_size
        10000
        >>> stats.sampled_size
        1000
    """

    original_size: int
    sampled_size: int
    class_distribution: dict[str, int]
    effective_ratio: float


def validate_stratified_config(config: StratifiedConfig) -> None:
    """Validate stratified sampling configuration.

    Args:
        config: StratifiedConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If column is empty.

    Examples:
        >>> config = StratifiedConfig(column="label", preserve_distribution=True)
        >>> validate_stratified_config(config)  # No error

        >>> validate_stratified_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = StratifiedConfig(column="", preserve_distribution=True)
        >>> validate_stratified_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: column cannot be empty
    """
    validate_not_none(config, "config")

    if not config.column:
        msg = "column cannot be empty"
        raise ValueError(msg)


def validate_importance_config(config: ImportanceConfig) -> None:
    """Validate importance sampling configuration.

    Args:
        config: ImportanceConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If temperature is not positive.

    Examples:
        >>> config = ImportanceConfig(
        ...     weights_column="weight",
        ...     temperature=1.0,
        ...     normalize=True,
        ... )
        >>> validate_importance_config(config)  # No error

        >>> validate_importance_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ImportanceConfig(weights_column=None, temperature=0.0, normalize=True)
        >>> validate_importance_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive
    """
    validate_not_none(config, "config")

    if config.temperature <= 0:
        msg = f"temperature must be positive, got {config.temperature}"
        raise ValueError(msg)


def validate_balanced_config(config: BalancedConfig) -> None:
    """Validate balanced sampling configuration.

    Args:
        config: BalancedConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If target_ratio is not in (0, 1].

    Examples:
        >>> config = BalancedConfig(
        ...     strategy=BalancingStrategy.OVERSAMPLE,
        ...     target_ratio=1.0,
        ...     random_state=42,
        ... )
        >>> validate_balanced_config(config)  # No error

        >>> validate_balanced_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BalancedConfig(
        ...     strategy=BalancingStrategy.OVERSAMPLE,
        ...     target_ratio=0.0,
        ...     random_state=None,
        ... )
        >>> validate_balanced_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: target_ratio must be in (0, 1]
    """
    validate_not_none(config, "config")

    if not 0 < config.target_ratio <= 1.0:
        msg = f"target_ratio must be in (0, 1], got {config.target_ratio}"
        raise ValueError(msg)


def _validate_sampling_method_config(config: SamplingConfig) -> None:
    """Validate method-specific sampling sub-config."""
    method_validators: dict[SamplingMethod, tuple[str, str, Callable[[Any], None]]] = {
        SamplingMethod.STRATIFIED: (
            "stratified_config",
            "STRATIFIED sampling",
            validate_stratified_config,
        ),
        SamplingMethod.IMPORTANCE: (
            "importance_config",
            "IMPORTANCE sampling",
            validate_importance_config,
        ),
        SamplingMethod.BALANCED: (
            "balanced_config",
            "BALANCED sampling",
            validate_balanced_config,
        ),
    }
    entry = method_validators.get(config.method)
    if entry is None:
        return
    attr_name, label, validator = entry
    sub_config = getattr(config, attr_name)
    if sub_config is None:
        msg = f"{attr_name} required for {label}"
        raise ValueError(msg)
    validator(sub_config)


def validate_sampling_config(config: SamplingConfig) -> None:
    """Validate sampling configuration.

    Args:
        config: SamplingConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If sample_size is not positive.
        ValueError: If method is STRATIFIED but stratified_config is None.
        ValueError: If method is IMPORTANCE but importance_config is None.
        ValueError: If method is BALANCED but balanced_config is None.

    Examples:
        >>> config = SamplingConfig(
        ...     method=SamplingMethod.RANDOM,
        ...     stratified_config=None,
        ...     importance_config=None,
        ...     balanced_config=None,
        ...     sample_size=100,
        ... )
        >>> validate_sampling_config(config)  # No error

        >>> validate_sampling_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = SamplingConfig(
        ...     method=SamplingMethod.STRATIFIED,
        ...     stratified_config=None,
        ...     importance_config=None,
        ...     balanced_config=None,
        ...     sample_size=100,
        ... )
        >>> validate_sampling_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stratified_config required for STRATIFIED sampling
    """
    validate_not_none(config, "config")

    if config.sample_size <= 0:
        msg = f"sample_size must be positive, got {config.sample_size}"
        raise ValueError(msg)

    _validate_sampling_method_config(config)


def create_stratified_config(
    column: str = "label",
    preserve_distribution: bool = True,
) -> StratifiedConfig:
    """Create a stratified sampling configuration.

    Args:
        column: Column name to stratify by. Defaults to "label".
        preserve_distribution: Whether to preserve distribution. Defaults to True.

    Returns:
        StratifiedConfig with the specified settings.

    Raises:
        ValueError: If column is empty.

    Examples:
        >>> config = create_stratified_config(column="category")
        >>> config.column
        'category'

        >>> config = create_stratified_config(preserve_distribution=False)
        >>> config.preserve_distribution
        False

        >>> create_stratified_config(column="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: column cannot be empty
    """
    config = StratifiedConfig(
        column=column,
        preserve_distribution=preserve_distribution,
    )
    validate_stratified_config(config)
    return config


def create_importance_config(
    weights_column: str | None = None,
    temperature: float = 1.0,
    normalize: bool = True,
) -> ImportanceConfig:
    """Create an importance sampling configuration.

    Args:
        weights_column: Column containing weights. Defaults to None.
        temperature: Temperature for weight softmax. Defaults to 1.0.
        normalize: Whether to normalize weights. Defaults to True.

    Returns:
        ImportanceConfig with the specified settings.

    Raises:
        ValueError: If temperature is not positive.

    Examples:
        >>> config = create_importance_config(weights_column="score")
        >>> config.weights_column
        'score'

        >>> config = create_importance_config(temperature=0.5)
        >>> config.temperature
        0.5

        >>> create_importance_config(temperature=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: temperature must be positive
    """
    config = ImportanceConfig(
        weights_column=weights_column,
        temperature=temperature,
        normalize=normalize,
    )
    validate_importance_config(config)
    return config


def create_balanced_config(
    strategy: str = "oversample",
    target_ratio: float = 1.0,
    random_state: int | None = None,
) -> BalancedConfig:
    """Create a balanced sampling configuration.

    Args:
        strategy: Balancing strategy name. Defaults to "oversample".
        target_ratio: Target balance ratio. Defaults to 1.0.
        random_state: Random seed. Defaults to None.

    Returns:
        BalancedConfig with the specified settings.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If target_ratio is not in (0, 1].

    Examples:
        >>> config = create_balanced_config(strategy="undersample")
        >>> config.strategy
        <BalancingStrategy.UNDERSAMPLE: 'undersample'>

        >>> config = create_balanced_config(target_ratio=0.8)
        >>> config.target_ratio
        0.8

        >>> create_balanced_config(strategy="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if strategy not in VALID_BALANCING_STRATEGIES:
        msg = f"strategy must be one of {VALID_BALANCING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = BalancedConfig(
        strategy=BalancingStrategy(strategy),
        target_ratio=target_ratio,
        random_state=random_state,
    )
    validate_balanced_config(config)
    return config


def create_sampling_config(
    method: str = "random",
    stratified_config: StratifiedConfig | None = None,
    importance_config: ImportanceConfig | None = None,
    balanced_config: BalancedConfig | None = None,
    sample_size: int | float = 1000,
) -> SamplingConfig:
    """Create a sampling configuration.

    Args:
        method: Sampling method name. Defaults to "random".
        stratified_config: Config for stratified sampling. Defaults to None.
        importance_config: Config for importance sampling. Defaults to None.
        balanced_config: Config for balanced sampling. Defaults to None.
        sample_size: Number of samples or fraction. Defaults to 1000.

    Returns:
        SamplingConfig with the specified settings.

    Raises:
        ValueError: If method is invalid.
        ValueError: If sample_size is not positive.

    Examples:
        >>> config = create_sampling_config(method="random", sample_size=500)
        >>> config.method
        <SamplingMethod.RANDOM: 'random'>
        >>> config.sample_size
        500

        >>> strat_config = create_stratified_config()
        >>> config = create_sampling_config(
        ...     method="stratified",
        ...     stratified_config=strat_config,
        ... )
        >>> config.method
        <SamplingMethod.STRATIFIED: 'stratified'>

        >>> create_sampling_config(method="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: method must be one of
    """
    if method not in VALID_SAMPLING_METHODS:
        msg = f"method must be one of {VALID_SAMPLING_METHODS}, got '{method}'"
        raise ValueError(msg)

    config = SamplingConfig(
        method=SamplingMethod(method),
        stratified_config=stratified_config,
        importance_config=importance_config,
        balanced_config=balanced_config,
        sample_size=sample_size,
    )
    validate_sampling_config(config)
    return config


def list_sampling_methods() -> list[str]:
    """List all available sampling methods.

    Returns:
        Sorted list of sampling method names.

    Examples:
        >>> methods = list_sampling_methods()
        >>> "random" in methods
        True
        >>> "stratified" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_SAMPLING_METHODS)


def get_sampling_method(name: str) -> SamplingMethod:
    """Get SamplingMethod enum from string name.

    Args:
        name: Name of the sampling method.

    Returns:
        Corresponding SamplingMethod enum value.

    Raises:
        ValueError: If name is not a valid sampling method.

    Examples:
        >>> get_sampling_method("random")
        <SamplingMethod.RANDOM: 'random'>

        >>> get_sampling_method("stratified")
        <SamplingMethod.STRATIFIED: 'stratified'>

        >>> get_sampling_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid sampling method: invalid
    """
    if name not in VALID_SAMPLING_METHODS:
        msg = f"invalid sampling method: {name}"
        raise ValueError(msg)

    return SamplingMethod(name)


def list_weighting_schemes() -> list[str]:
    """List all available weighting schemes.

    Returns:
        Sorted list of weighting scheme names.

    Examples:
        >>> schemes = list_weighting_schemes()
        >>> "uniform" in schemes
        True
        >>> "inverse_frequency" in schemes
        True
        >>> schemes == sorted(schemes)
        True
    """
    return sorted(VALID_WEIGHTING_SCHEMES)


def get_weighting_scheme(name: str) -> WeightingScheme:
    """Get WeightingScheme enum from string name.

    Args:
        name: Name of the weighting scheme.

    Returns:
        Corresponding WeightingScheme enum value.

    Raises:
        ValueError: If name is not a valid weighting scheme.

    Examples:
        >>> get_weighting_scheme("uniform")
        <WeightingScheme.UNIFORM: 'uniform'>

        >>> get_weighting_scheme("inverse_frequency")
        <WeightingScheme.INVERSE_FREQUENCY: 'inverse_frequency'>

        >>> get_weighting_scheme("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid weighting scheme: invalid
    """
    if name not in VALID_WEIGHTING_SCHEMES:
        msg = f"invalid weighting scheme: {name}"
        raise ValueError(msg)

    return WeightingScheme(name)


def list_balancing_strategies() -> list[str]:
    """List all available balancing strategies.

    Returns:
        Sorted list of balancing strategy names.

    Examples:
        >>> strategies = list_balancing_strategies()
        >>> "oversample" in strategies
        True
        >>> "undersample" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_BALANCING_STRATEGIES)


def get_balancing_strategy(name: str) -> BalancingStrategy:
    """Get BalancingStrategy enum from string name.

    Args:
        name: Name of the balancing strategy.

    Returns:
        Corresponding BalancingStrategy enum value.

    Raises:
        ValueError: If name is not a valid balancing strategy.

    Examples:
        >>> get_balancing_strategy("oversample")
        <BalancingStrategy.OVERSAMPLE: 'oversample'>

        >>> get_balancing_strategy("smote")
        <BalancingStrategy.SMOTE: 'smote'>

        >>> get_balancing_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid balancing strategy: invalid
    """
    if name not in VALID_BALANCING_STRATEGIES:
        msg = f"invalid balancing strategy: {name}"
        raise ValueError(msg)

    return BalancingStrategy(name)


def calculate_sample_weights(
    labels: Sequence[str],
    scheme: WeightingScheme = WeightingScheme.INVERSE_FREQUENCY,
    custom_weights: Mapping[str, float] | None = None,
) -> list[float]:
    """Calculate sample weights based on label distribution.

    Args:
        labels: Sequence of class labels for each sample.
        scheme: Weighting scheme to use. Defaults to INVERSE_FREQUENCY.
        custom_weights: Custom weights per class (required if scheme is CUSTOM).

    Returns:
        List of weights, one per sample.

    Raises:
        ValueError: If labels is None or empty.
        ValueError: If scheme is CUSTOM but custom_weights is None.

    Examples:
        >>> weights = calculate_sample_weights(["A", "A", "A", "B"])
        >>> len(weights)
        4
        >>> weights[3] > weights[0]  # B (minority) has higher weight
        True

        >>> weights = calculate_sample_weights(
        ...     ["A", "B"],
        ...     scheme=WeightingScheme.UNIFORM,
        ... )
        >>> weights[0] == weights[1]
        True

        >>> calculate_sample_weights([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: labels cannot be empty

        >>> calculate_sample_weights(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: labels cannot be None
    """
    if labels is None:
        msg = "labels cannot be None"
        raise ValueError(msg)

    if not labels:
        msg = "labels cannot be empty"
        raise ValueError(msg)

    if scheme == WeightingScheme.CUSTOM and custom_weights is None:
        msg = "custom_weights required when scheme is CUSTOM"
        raise ValueError(msg)

    label_counts = Counter(labels)
    total = len(labels)
    weights: list[float] = []

    for label in labels:
        if scheme == WeightingScheme.UNIFORM:
            weights.append(1.0)
        elif scheme == WeightingScheme.INVERSE_FREQUENCY:
            # Weight = total / (num_classes * class_count)
            num_classes = len(label_counts)
            weights.append(total / (num_classes * label_counts[label]))
        elif scheme == WeightingScheme.SQRT_INVERSE:
            # Weight = sqrt(total / (num_classes * class_count))
            num_classes = len(label_counts)
            weights.append(math.sqrt(total / (num_classes * label_counts[label])))
        elif scheme == WeightingScheme.CUSTOM:
            # custom_weights is guaranteed to be not None here
            assert custom_weights is not None
            if label not in custom_weights:
                msg = f"custom_weights missing weight for label '{label}'"
                raise ValueError(msg)
            weights.append(custom_weights[label])

    return weights


def stratified_sample(
    data: Sequence[dict[str, object]],
    column: str,
    sample_size: int,
    preserve_distribution: bool = True,
    random_state: int | None = None,
) -> list[dict[str, object]]:
    """Perform stratified sampling on data.

    Args:
        data: Sequence of data records (dicts).
        column: Column to stratify by.
        sample_size: Number of samples to draw.
        preserve_distribution: Whether to preserve class distribution.
        random_state: Random seed for reproducibility.

    Returns:
        List of sampled records.

    Raises:
        ValueError: If data is None or empty.
        ValueError: If column is empty.
        ValueError: If sample_size is not positive.
        ValueError: If sample_size exceeds data length.

    Examples:
        >>> data = [
        ...     {"id": 1, "label": "A"},
        ...     {"id": 2, "label": "A"},
        ...     {"id": 3, "label": "B"},
        ...     {"id": 4, "label": "B"},
        ... ]
        >>> result = stratified_sample(data, "label", 2, random_state=42)
        >>> len(result)
        2

        >>> stratified_sample([], "label", 10)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be empty

        >>> stratified_sample(None, "label", 10)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be None
    """
    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    if not data:
        msg = "data cannot be empty"
        raise ValueError(msg)

    if not column:
        msg = "column cannot be empty"
        raise ValueError(msg)

    if sample_size <= 0:
        msg = f"sample_size must be positive, got {sample_size}"
        raise ValueError(msg)

    if sample_size > len(data):
        msg = f"sample_size ({sample_size}) exceeds data length ({len(data)})"
        raise ValueError(msg)

    if random_state is not None:
        random.seed(random_state)

    # Group data by class
    groups: dict[object, list[dict[str, object]]] = {}
    for record in data:
        label = record.get(column)
        if label not in groups:
            groups[label] = []
        groups[label].append(record)

    sampled: list[dict[str, object]] = []

    if preserve_distribution:
        # Sample proportionally from each class
        for _label, group in groups.items():
            proportion = len(group) / len(data)
            n_samples = max(1, int(sample_size * proportion))
            n_samples = min(n_samples, len(group))
            sampled.extend(random.sample(group, n_samples))
    else:
        # Equal samples from each class
        n_per_class = max(1, sample_size // len(groups))
        for _label, group in groups.items():
            n_samples = min(n_per_class, len(group))
            sampled.extend(random.sample(group, n_samples))

    # Trim to exact sample_size if over
    if len(sampled) > sample_size:
        sampled = random.sample(sampled, sample_size)

    return sampled


def importance_sample(
    data: Sequence[dict[str, object]],
    weights: Sequence[float],
    sample_size: int,
    temperature: float = 1.0,
    random_state: int | None = None,
) -> list[dict[str, object]]:
    """Perform importance-weighted sampling on data.

    Args:
        data: Sequence of data records (dicts).
        weights: Importance weight for each record.
        sample_size: Number of samples to draw.
        temperature: Temperature for softmax (higher = more uniform).
        random_state: Random seed for reproducibility.

    Returns:
        List of sampled records.

    Raises:
        ValueError: If data is None or empty.
        ValueError: If weights length doesn't match data length.
        ValueError: If sample_size is not positive.
        ValueError: If temperature is not positive.

    Examples:
        >>> data = [{"id": 1}, {"id": 2}, {"id": 3}]
        >>> weights = [0.1, 0.1, 0.8]
        >>> result = importance_sample(data, weights, 2, random_state=42)
        >>> len(result)
        2

        >>> importance_sample([], [0.5], 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be empty

        >>> importance_sample([{"a": 1}], [], 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: weights length must match data length
    """
    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    if not data:
        msg = "data cannot be empty"
        raise ValueError(msg)

    if len(weights) != len(data):
        msg = f"weights length ({len(weights)}) must match data length ({len(data)})"
        raise ValueError(msg)

    if sample_size <= 0:
        msg = f"sample_size must be positive, got {sample_size}"
        raise ValueError(msg)

    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    if random_state is not None:
        random.seed(random_state)

    # Apply temperature scaling and normalize
    scaled = [w / temperature for w in weights]
    max_scaled = max(scaled) if scaled else 0.0

    # Softmax normalization with numerical stability
    exp_weights = [math.exp(w - max_scaled) for w in scaled]
    total = sum(exp_weights)
    probs = (
        [w / total for w in exp_weights] if total > 0 else [1.0 / len(data)] * len(data)
    )

    # Sample with replacement using weighted probabilities
    sample_size = min(sample_size, len(data))
    indices = random.choices(range(len(data)), weights=probs, k=sample_size)

    return [data[i] for i in indices]


def balance_classes(
    data: Sequence[dict[str, object]],
    label_column: str,
    strategy: BalancingStrategy = BalancingStrategy.OVERSAMPLE,
    target_ratio: float = 1.0,
    random_state: int | None = None,
) -> list[dict[str, object]]:
    """Balance class distribution in data.

    Args:
        data: Sequence of data records (dicts).
        label_column: Column containing class labels.
        strategy: Balancing strategy to use.
        target_ratio: Target ratio for minority/majority (1.0 = equal).
        random_state: Random seed for reproducibility.

    Returns:
        List of balanced records.

    Raises:
        ValueError: If data is None or empty.
        ValueError: If label_column is empty.
        ValueError: If target_ratio is not in (0, 1].

    Examples:
        >>> data = [
        ...     {"id": 1, "label": "A"},
        ...     {"id": 2, "label": "A"},
        ...     {"id": 3, "label": "A"},
        ...     {"id": 4, "label": "B"},
        ... ]
        >>> result = balance_classes(data, "label", random_state=42)
        >>> len([r for r in result if r["label"] == "A"])  # doctest: +SKIP
        3
        >>> len([r for r in result if r["label"] == "B"]) >= 1  # doctest: +SKIP
        True

        >>> balance_classes([], "label")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be empty

        >>> balance_classes(None, "label")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be None
    """
    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    if not data:
        msg = "data cannot be empty"
        raise ValueError(msg)

    if not label_column:
        msg = "label_column cannot be empty"
        raise ValueError(msg)

    if not 0 < target_ratio <= 1.0:
        msg = f"target_ratio must be in (0, 1], got {target_ratio}"
        raise ValueError(msg)

    if random_state is not None:
        random.seed(random_state)

    # Group data by class
    groups: dict[object, list[dict[str, object]]] = {}
    for record in data:
        label = record.get(label_column)
        if label not in groups:
            groups[label] = []
        groups[label].append(record)

    class_counts = {label: len(records) for label, records in groups.items()}
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    balanced: list[dict[str, object]] = []

    if strategy == BalancingStrategy.OVERSAMPLE:
        # Oversample minority classes to match majority
        target_count = int(max_count * target_ratio)
        for _label, records in groups.items():
            if len(records) >= target_count:
                balanced.extend(records)
            else:
                # Sample with replacement to reach target
                balanced.extend(records)
                additional = target_count - len(records)
                balanced.extend(random.choices(records, k=additional))

    elif strategy == BalancingStrategy.UNDERSAMPLE:
        # Undersample majority classes to match minority
        target_count = max(1, int(min_count / target_ratio))
        for _label, records in groups.items():
            if len(records) <= target_count:
                balanced.extend(records)
            else:
                balanced.extend(random.sample(records, target_count))

    elif strategy == BalancingStrategy.SMOTE:
        # Simplified SMOTE: just oversample for now
        # Full SMOTE requires feature interpolation
        target_count = int(max_count * target_ratio)
        for _label, records in groups.items():
            if len(records) >= target_count:
                balanced.extend(records)
            else:
                balanced.extend(records)
                additional = target_count - len(records)
                balanced.extend(random.choices(records, k=additional))

    elif strategy == BalancingStrategy.CLASS_WEIGHT:
        # Return original data (weights computed separately)
        balanced.extend(list(data))

    return balanced


def estimate_effective_samples(
    class_counts: Mapping[str, int],
    beta: float = 0.9999,
) -> dict[str, float]:
    """Estimate effective number of samples per class.

    Uses the effective number formula: (1 - beta^n) / (1 - beta)
    where n is the number of samples and beta controls the decay.

    Args:
        class_counts: Mapping of class labels to sample counts.
        beta: Decay factor (higher = less downweighting). Defaults to 0.9999.

    Returns:
        Dictionary mapping class labels to effective sample counts.

    Raises:
        ValueError: If class_counts is None or empty.
        ValueError: If beta is not in (0, 1).

    Examples:
        >>> counts = {"A": 1000, "B": 100, "C": 10}
        >>> effective = estimate_effective_samples(counts)
        >>> effective["A"] > effective["B"] > effective["C"]
        True

        >>> estimate_effective_samples({})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: class_counts cannot be empty

        >>> estimate_effective_samples(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: class_counts cannot be None
    """
    if class_counts is None:
        msg = "class_counts cannot be None"
        raise ValueError(msg)

    if not class_counts:
        msg = "class_counts cannot be empty"
        raise ValueError(msg)

    if not 0 < beta < 1:
        msg = f"beta must be in (0, 1), got {beta}"
        raise ValueError(msg)

    effective: dict[str, float] = {}

    for label, count in class_counts.items():
        # Effective number formula
        if count == 0:
            effective[str(label)] = 0.0
        else:
            effective[str(label)] = (1.0 - beta**count) / (1.0 - beta)

    return effective


def format_sampling_stats(stats: SamplingStats) -> str:
    """Format sampling statistics as a human-readable string.

    Args:
        stats: SamplingStats to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = SamplingStats(
        ...     original_size=10000,
        ...     sampled_size=1000,
        ...     class_distribution={"A": 500, "B": 500},
        ...     effective_ratio=0.1,
        ... )
        >>> formatted = format_sampling_stats(stats)
        >>> "10,000" in formatted or "10000" in formatted
        True
        >>> "1,000" in formatted or "1000" in formatted
        True

        >>> format_sampling_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    validate_not_none(stats, "stats")

    sampling_rate = (
        stats.sampled_size / stats.original_size * 100
        if stats.original_size > 0
        else 0.0
    )

    lines = [
        "Sampling Statistics",
        "=" * 40,
        f"Original size:    {stats.original_size:,}",
        f"Sampled size:     {stats.sampled_size:,}",
        f"Sampling rate:    {sampling_rate:.1f}%",
        f"Effective ratio:  {stats.effective_ratio:.4f}",
        "",
        "Class Distribution:",
    ]

    total_sampled = sum(stats.class_distribution.values())
    for label, count in sorted(stats.class_distribution.items()):
        percentage = count / total_sampled * 100 if total_sampled > 0 else 0.0
        lines.append(f"  {label}: {count:,} ({percentage:.1f}%)")

    return "\n".join(lines)


def get_recommended_sampling_config(
    dataset_size: int,
    num_classes: int,
    imbalance_ratio: float,
) -> SamplingConfig:
    """Get recommended sampling configuration based on dataset characteristics.

    Args:
        dataset_size: Total number of samples.
        num_classes: Number of unique classes.
        imbalance_ratio: Ratio of largest to smallest class.

    Returns:
        Recommended SamplingConfig for the dataset.

    Raises:
        ValueError: If dataset_size is not positive.
        ValueError: If num_classes is not positive.
        ValueError: If imbalance_ratio is less than 1.

    Examples:
        >>> config = get_recommended_sampling_config(
        ...     dataset_size=10000,
        ...     num_classes=2,
        ...     imbalance_ratio=10.0,
        ... )
        >>> config.method
        <SamplingMethod.BALANCED: 'balanced'>

        >>> config = get_recommended_sampling_config(
        ...     dataset_size=10000,
        ...     num_classes=2,
        ...     imbalance_ratio=3.0,
        ... )
        >>> config.method
        <SamplingMethod.STRATIFIED: 'stratified'>

        >>> get_recommended_sampling_config(0, 2, 1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: dataset_size must be positive

        >>> get_recommended_sampling_config(100, 0, 1.0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_classes must be positive
    """
    if dataset_size <= 0:
        msg = f"dataset_size must be positive, got {dataset_size}"
        raise ValueError(msg)

    if num_classes <= 0:
        msg = f"num_classes must be positive, got {num_classes}"
        raise ValueError(msg)

    if imbalance_ratio < 1.0:
        msg = f"imbalance_ratio must be >= 1.0, got {imbalance_ratio}"
        raise ValueError(msg)

    # Highly imbalanced datasets: use balanced sampling
    if imbalance_ratio >= 5.0:
        balanced_config = create_balanced_config(
            strategy="oversample",
            target_ratio=0.8,
        )
        return create_sampling_config(
            method="balanced",
            balanced_config=balanced_config,
            sample_size=dataset_size,
        )

    # Moderately imbalanced: use stratified sampling
    if imbalance_ratio >= 2.0:
        stratified_config = create_stratified_config(
            column="label",
            preserve_distribution=True,
        )
        return create_sampling_config(
            method="stratified",
            stratified_config=stratified_config,
            sample_size=min(dataset_size, 10000),
        )

    # Well-balanced: use random sampling
    return create_sampling_config(
        method="random",
        sample_size=min(dataset_size, 10000),
    )


def compute_sampling_stats(
    data: Sequence[dict[str, object]],
    label_column: str,
    sampled: Sequence[dict[str, object]],
) -> SamplingStats:
    """Compute statistics for a sampling operation.

    Args:
        data: Original dataset.
        label_column: Column containing class labels.
        sampled: Sampled dataset.

    Returns:
        SamplingStats with computed metrics.

    Raises:
        ValueError: If data is None.
        ValueError: If sampled is None.
        ValueError: If label_column is empty.

    Examples:
        >>> data = [
        ...     {"id": 1, "label": "A"},
        ...     {"id": 2, "label": "A"},
        ...     {"id": 3, "label": "B"},
        ...     {"id": 4, "label": "B"},
        ... ]
        >>> sampled = [{"id": 1, "label": "A"}, {"id": 3, "label": "B"}]
        >>> stats = compute_sampling_stats(data, "label", sampled)
        >>> stats.original_size
        4
        >>> stats.sampled_size
        2

        >>> compute_sampling_stats(None, "label", [])
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: data cannot be None
    """
    if data is None:
        msg = "data cannot be None"
        raise ValueError(msg)

    if sampled is None:
        msg = "sampled cannot be None"
        raise ValueError(msg)

    if not label_column:
        msg = "label_column cannot be empty"
        raise ValueError(msg)

    # Count class distribution in sampled data
    class_dist: dict[str, int] = {}
    for record in sampled:
        label = str(record.get(label_column, "unknown"))
        class_dist[label] = class_dist.get(label, 0) + 1

    original_size = len(data)
    sampled_size = len(sampled)
    effective_ratio = sampled_size / original_size if original_size > 0 else 0.0

    return SamplingStats(
        original_size=original_size,
        sampled_size=sampled_size,
        class_distribution=class_dist,
        effective_ratio=effective_ratio,
    )
