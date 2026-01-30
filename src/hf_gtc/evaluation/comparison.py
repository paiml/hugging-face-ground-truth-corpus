"""Model comparison and statistical testing utilities.

This module provides tools for comparing model performance using statistical tests,
effect size calculations, and confidence intervals.

Examples:
    >>> from hf_gtc.evaluation.comparison import ComparisonMethod, SignificanceLevel
    >>> ComparisonMethod.PAIRED_TTEST.value
    'paired_ttest'
    >>> SignificanceLevel.P05.value
    0.05
    >>> from hf_gtc.evaluation.comparison import EffectSize, create_comparison_config
    >>> EffectSize.COHENS_D.value
    'cohens_d'
    >>> config = create_comparison_config()
    >>> config.method
    <ComparisonMethod.PAIRED_TTEST: 'paired_ttest'>
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class ComparisonMethod(Enum):
    """Methods for statistical comparison of models.

    Attributes:
        PAIRED_TTEST: Paired t-test for normally distributed differences.
        BOOTSTRAP: Bootstrap resampling for non-parametric comparison.
        PERMUTATION: Permutation test for non-parametric comparison.
        WILCOXON: Wilcoxon signed-rank test for non-normal distributions.
        MCNEMAR: McNemar's test for paired binary outcomes.

    Examples:
        >>> ComparisonMethod.PAIRED_TTEST.value
        'paired_ttest'
        >>> ComparisonMethod.BOOTSTRAP.value
        'bootstrap'
        >>> ComparisonMethod.PERMUTATION.value
        'permutation'
        >>> ComparisonMethod.WILCOXON.value
        'wilcoxon'
        >>> ComparisonMethod.MCNEMAR.value
        'mcnemar'
    """

    PAIRED_TTEST = "paired_ttest"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    WILCOXON = "wilcoxon"
    MCNEMAR = "mcnemar"


VALID_COMPARISON_METHODS = frozenset(m.value for m in ComparisonMethod)


class SignificanceLevel(Enum):
    """Standard significance levels for hypothesis testing.

    Attributes:
        P01: 0.01 significance level (99% confidence).
        P05: 0.05 significance level (95% confidence).
        P10: 0.10 significance level (90% confidence).

    Examples:
        >>> SignificanceLevel.P01.value
        0.01
        >>> SignificanceLevel.P05.value
        0.05
        >>> SignificanceLevel.P10.value
        0.1
    """

    P01 = 0.01
    P05 = 0.05
    P10 = 0.10


VALID_SIGNIFICANCE_LEVELS = frozenset(s.value for s in SignificanceLevel)


class EffectSize(Enum):
    """Effect size measures for quantifying practical significance.

    Attributes:
        COHENS_D: Cohen's d, standardized mean difference.
        HEDGES_G: Hedges' g, bias-corrected standardized mean difference.
        GLASS_DELTA: Glass's delta, uses control group SD.

    Examples:
        >>> EffectSize.COHENS_D.value
        'cohens_d'
        >>> EffectSize.HEDGES_G.value
        'hedges_g'
        >>> EffectSize.GLASS_DELTA.value
        'glass_delta'
    """

    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"


VALID_EFFECT_SIZES = frozenset(e.value for e in EffectSize)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Configuration for model comparison.

    Attributes:
        method: Statistical test method. Defaults to PAIRED_TTEST.
        significance_level: Alpha level for hypothesis testing. Defaults to P05.
        num_bootstrap: Number of bootstrap samples. Defaults to 1000.
        effect_size_type: Effect size measure to use. Defaults to COHENS_D.
        random_seed: Random seed for reproducibility. Defaults to None.

    Examples:
        >>> config = ComparisonConfig()
        >>> config.method
        <ComparisonMethod.PAIRED_TTEST: 'paired_ttest'>
        >>> config.significance_level
        <SignificanceLevel.P05: 0.05>
        >>> config.num_bootstrap
        1000
        >>> config = ComparisonConfig(
        ...     method=ComparisonMethod.BOOTSTRAP, num_bootstrap=5000
        ... )
        >>> config.num_bootstrap
        5000
    """

    method: ComparisonMethod = ComparisonMethod.PAIRED_TTEST
    significance_level: SignificanceLevel = SignificanceLevel.P05
    num_bootstrap: int = 1000
    effect_size_type: EffectSize = EffectSize.COHENS_D
    random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Results from a single model evaluation.

    Attributes:
        model_name: Identifier for the model.
        scores: Performance scores (e.g., accuracy per sample).
        metadata: Additional metadata about the model.

    Examples:
        >>> result = ModelResult(
        ...     model_name="model_a",
        ...     scores=(0.8, 0.9, 0.85),
        ...     metadata={"version": "1.0"},
        ... )
        >>> result.model_name
        'model_a'
        >>> len(result.scores)
        3
        >>> result.metadata["version"]
        '1.0'
    """

    model_name: str
    scores: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Result of a statistical comparison between two models.

    Attributes:
        model_a: Name of the first model.
        model_b: Name of the second model.
        p_value: P-value from the statistical test.
        effect_size: Computed effect size.
        is_significant: Whether the difference is statistically significant.
        confidence_interval: Confidence interval for the difference.
        test_statistic: The test statistic value.
        method: The comparison method used.

    Examples:
        >>> result = ComparisonResult(
        ...     model_a="model_1",
        ...     model_b="model_2",
        ...     p_value=0.03,
        ...     effect_size=0.5,
        ...     is_significant=True,
        ...     confidence_interval=(0.1, 0.9),
        ...     test_statistic=2.5,
        ...     method=ComparisonMethod.PAIRED_TTEST,
        ... )
        >>> result.is_significant
        True
        >>> result.effect_size
        0.5
    """

    model_a: str
    model_b: str
    p_value: float
    effect_size: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    test_statistic: float
    method: ComparisonMethod


@dataclass(frozen=True, slots=True)
class ComparisonStats:
    """Aggregate statistics from multiple model comparisons.

    Attributes:
        total_comparisons: Total number of comparisons made.
        significant_count: Number of statistically significant comparisons.
        avg_effect_size: Average absolute effect size across comparisons.
        max_effect_size: Maximum absolute effect size observed.
        min_p_value: Minimum p-value observed.

    Examples:
        >>> stats = ComparisonStats(
        ...     total_comparisons=10,
        ...     significant_count=3,
        ...     avg_effect_size=0.4,
        ...     max_effect_size=0.8,
        ...     min_p_value=0.001,
        ... )
        >>> stats.total_comparisons
        10
        >>> stats.significant_count
        3
    """

    total_comparisons: int
    significant_count: int
    avg_effect_size: float
    max_effect_size: float
    min_p_value: float


# Factory functions


def create_comparison_config(
    method: ComparisonMethod = ComparisonMethod.PAIRED_TTEST,
    significance_level: SignificanceLevel = SignificanceLevel.P05,
    num_bootstrap: int = 1000,
    effect_size_type: EffectSize = EffectSize.COHENS_D,
    random_seed: int | None = None,
) -> ComparisonConfig:
    """Create a validated ComparisonConfig.

    Args:
        method: Statistical test method. Defaults to PAIRED_TTEST.
        significance_level: Alpha level. Defaults to P05.
        num_bootstrap: Number of bootstrap samples. Defaults to 1000.
        effect_size_type: Effect size measure. Defaults to COHENS_D.
        random_seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Validated ComparisonConfig instance.

    Raises:
        ValueError: If num_bootstrap is not positive.

    Examples:
        >>> config = create_comparison_config()
        >>> config.method
        <ComparisonMethod.PAIRED_TTEST: 'paired_ttest'>

        >>> config = create_comparison_config(
        ...     method=ComparisonMethod.BOOTSTRAP,
        ...     num_bootstrap=5000,
        ... )
        >>> config.num_bootstrap
        5000

        >>> create_comparison_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     num_bootstrap=0
        ... )
        Traceback (most recent call last):
        ValueError: num_bootstrap must be positive
    """
    config = ComparisonConfig(
        method=method,
        significance_level=significance_level,
        num_bootstrap=num_bootstrap,
        effect_size_type=effect_size_type,
        random_seed=random_seed,
    )
    validate_comparison_config(config)
    return config


def create_model_result(
    model_name: str,
    scores: Sequence[float],
    metadata: dict[str, Any] | None = None,
) -> ModelResult:
    """Create a validated ModelResult.

    Args:
        model_name: Identifier for the model.
        scores: Performance scores.
        metadata: Additional metadata. Defaults to empty dict.

    Returns:
        Validated ModelResult instance.

    Raises:
        ValueError: If model_name is empty.
        ValueError: If scores is empty.

    Examples:
        >>> result = create_model_result("model_a", [0.8, 0.9, 0.85])
        >>> result.model_name
        'model_a'
        >>> len(result.scores)
        3

        >>> create_model_result("", [0.5])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty

        >>> create_model_result("model", [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores cannot be empty
    """
    result = ModelResult(
        model_name=model_name,
        scores=tuple(scores),
        metadata=metadata if metadata is not None else {},
    )
    validate_model_result(result)
    return result


def create_comparison_result(
    model_a: str,
    model_b: str,
    p_value: float,
    effect_size: float,
    is_significant: bool,
    confidence_interval: tuple[float, float],
    test_statistic: float,
    method: ComparisonMethod,
) -> ComparisonResult:
    """Create a validated ComparisonResult.

    Args:
        model_a: Name of the first model.
        model_b: Name of the second model.
        p_value: P-value from the statistical test.
        effect_size: Computed effect size.
        is_significant: Whether the difference is significant.
        confidence_interval: Confidence interval for the difference.
        test_statistic: The test statistic value.
        method: The comparison method used.

    Returns:
        Validated ComparisonResult instance.

    Raises:
        ValueError: If model_a or model_b is empty.
        ValueError: If p_value is not in [0, 1].

    Examples:
        >>> result = create_comparison_result(
        ...     model_a="model_1",
        ...     model_b="model_2",
        ...     p_value=0.03,
        ...     effect_size=0.5,
        ...     is_significant=True,
        ...     confidence_interval=(0.1, 0.9),
        ...     test_statistic=2.5,
        ...     method=ComparisonMethod.PAIRED_TTEST,
        ... )
        >>> result.is_significant
        True

        >>> create_comparison_result("", "b", 0.5, 0.1, False, (0, 1), 1.0,
        ...     ComparisonMethod.PAIRED_TTEST)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_a cannot be empty
    """
    result = ComparisonResult(
        model_a=model_a,
        model_b=model_b,
        p_value=p_value,
        effect_size=effect_size,
        is_significant=is_significant,
        confidence_interval=confidence_interval,
        test_statistic=test_statistic,
        method=method,
    )
    validate_comparison_result(result)
    return result


# Validation functions


def validate_comparison_config(config: ComparisonConfig) -> None:
    """Validate a ComparisonConfig.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If num_bootstrap is not positive.

    Examples:
        >>> config = ComparisonConfig()
        >>> validate_comparison_config(config)  # No error

        >>> validate_comparison_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = ComparisonConfig(num_bootstrap=0)
        >>> validate_comparison_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_bootstrap must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.num_bootstrap <= 0:
        msg = f"num_bootstrap must be positive, got {config.num_bootstrap}"
        raise ValueError(msg)


def validate_model_result(result: ModelResult) -> None:
    """Validate a ModelResult.

    Args:
        result: Result to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If model_name is empty.
        ValueError: If scores is empty.

    Examples:
        >>> result = ModelResult(model_name="model", scores=(0.5,))
        >>> validate_model_result(result)  # No error

        >>> validate_model_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = ModelResult(model_name="", scores=(0.5,))
        >>> validate_model_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    if not result.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if not result.scores:
        msg = "scores cannot be empty"
        raise ValueError(msg)


def validate_comparison_result(result: ComparisonResult) -> None:
    """Validate a ComparisonResult.

    Args:
        result: Result to validate.

    Raises:
        ValueError: If result is None.
        ValueError: If model_a or model_b is empty.
        ValueError: If p_value is not in [0, 1].

    Examples:
        >>> result = ComparisonResult(
        ...     model_a="a", model_b="b", p_value=0.05, effect_size=0.3,
        ...     is_significant=True, confidence_interval=(0.1, 0.5),
        ...     test_statistic=2.0, method=ComparisonMethod.PAIRED_TTEST,
        ... )
        >>> validate_comparison_result(result)  # No error

        >>> validate_comparison_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None

        >>> bad = ComparisonResult(
        ...     model_a="", model_b="b", p_value=0.05, effect_size=0.3,
        ...     is_significant=True, confidence_interval=(0.1, 0.5),
        ...     test_statistic=2.0, method=ComparisonMethod.PAIRED_TTEST,
        ... )
        >>> validate_comparison_result(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_a cannot be empty
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    if not result.model_a:
        msg = "model_a cannot be empty"
        raise ValueError(msg)

    if not result.model_b:
        msg = "model_b cannot be empty"
        raise ValueError(msg)

    if not 0 <= result.p_value <= 1:
        msg = f"p_value must be in [0, 1], got {result.p_value}"
        raise ValueError(msg)


def validate_comparison_stats(stats: ComparisonStats) -> None:
    """Validate ComparisonStats.

    Args:
        stats: Stats to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If total_comparisons is not positive.
        ValueError: If significant_count exceeds total_comparisons.

    Examples:
        >>> stats = ComparisonStats(
        ...     total_comparisons=10, significant_count=3,
        ...     avg_effect_size=0.4, max_effect_size=0.8, min_p_value=0.01,
        ... )
        >>> validate_comparison_stats(stats)  # No error

        >>> validate_comparison_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = ComparisonStats(
        ...     total_comparisons=0, significant_count=0,
        ...     avg_effect_size=0.0, max_effect_size=0.0, min_p_value=1.0,
        ... )
        >>> validate_comparison_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_comparisons must be positive
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.total_comparisons <= 0:
        msg = f"total_comparisons must be positive, got {stats.total_comparisons}"
        raise ValueError(msg)

    if stats.significant_count > stats.total_comparisons:
        msg = (
            f"significant_count ({stats.significant_count}) cannot exceed "
            f"total_comparisons ({stats.total_comparisons})"
        )
        raise ValueError(msg)

    if stats.significant_count < 0:
        msg = f"significant_count cannot be negative, got {stats.significant_count}"
        raise ValueError(msg)


# List/get functions for enums


def list_comparison_methods() -> list[str]:
    """List all available comparison methods.

    Returns:
        Sorted list of comparison method names.

    Examples:
        >>> methods = list_comparison_methods()
        >>> "paired_ttest" in methods
        True
        >>> "bootstrap" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_COMPARISON_METHODS)


def validate_comparison_method(method: str) -> bool:
    """Validate if a string is a valid comparison method.

    Args:
        method: The method string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_comparison_method("paired_ttest")
        True
        >>> validate_comparison_method("bootstrap")
        True
        >>> validate_comparison_method("invalid")
        False
        >>> validate_comparison_method("")
        False
    """
    return method in VALID_COMPARISON_METHODS


def get_comparison_method(name: str) -> ComparisonMethod:
    """Get ComparisonMethod enum from string name.

    Args:
        name: Name of the comparison method.

    Returns:
        Corresponding ComparisonMethod enum value.

    Raises:
        ValueError: If name is not a valid comparison method.

    Examples:
        >>> get_comparison_method("paired_ttest")
        <ComparisonMethod.PAIRED_TTEST: 'paired_ttest'>
        >>> get_comparison_method("bootstrap")
        <ComparisonMethod.BOOTSTRAP: 'bootstrap'>

        >>> get_comparison_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid comparison method: invalid
    """
    if not validate_comparison_method(name):
        msg = f"invalid comparison method: {name}"
        raise ValueError(msg)

    return ComparisonMethod(name)


def list_significance_levels() -> list[float]:
    """List all available significance levels.

    Returns:
        Sorted list of significance level values.

    Examples:
        >>> levels = list_significance_levels()
        >>> 0.01 in levels
        True
        >>> 0.05 in levels
        True
        >>> levels == sorted(levels)
        True
    """
    return sorted(VALID_SIGNIFICANCE_LEVELS)


def validate_significance_level(level: float) -> bool:
    """Validate if a value is a valid significance level.

    Args:
        level: The significance level to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_significance_level(0.05)
        True
        >>> validate_significance_level(0.01)
        True
        >>> validate_significance_level(0.03)
        False
    """
    return level in VALID_SIGNIFICANCE_LEVELS


def get_significance_level(value: float) -> SignificanceLevel:
    """Get SignificanceLevel enum from value.

    Args:
        value: Value of the significance level.

    Returns:
        Corresponding SignificanceLevel enum value.

    Raises:
        ValueError: If value is not a valid significance level.

    Examples:
        >>> get_significance_level(0.05)
        <SignificanceLevel.P05: 0.05>
        >>> get_significance_level(0.01)
        <SignificanceLevel.P01: 0.01>

        >>> get_significance_level(0.03)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid significance level: 0.03
    """
    if not validate_significance_level(value):
        msg = f"invalid significance level: {value}"
        raise ValueError(msg)

    return SignificanceLevel(value)


def list_effect_sizes() -> list[str]:
    """List all available effect size measures.

    Returns:
        Sorted list of effect size names.

    Examples:
        >>> sizes = list_effect_sizes()
        >>> "cohens_d" in sizes
        True
        >>> "hedges_g" in sizes
        True
        >>> sizes == sorted(sizes)
        True
    """
    return sorted(VALID_EFFECT_SIZES)


def validate_effect_size_type(effect_size: str) -> bool:
    """Validate if a string is a valid effect size type.

    Args:
        effect_size: The effect size string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_effect_size_type("cohens_d")
        True
        >>> validate_effect_size_type("hedges_g")
        True
        >>> validate_effect_size_type("invalid")
        False
        >>> validate_effect_size_type("")
        False
    """
    return effect_size in VALID_EFFECT_SIZES


def get_effect_size_type(name: str) -> EffectSize:
    """Get EffectSize enum from string name.

    Args:
        name: Name of the effect size measure.

    Returns:
        Corresponding EffectSize enum value.

    Raises:
        ValueError: If name is not a valid effect size type.

    Examples:
        >>> get_effect_size_type("cohens_d")
        <EffectSize.COHENS_D: 'cohens_d'>
        >>> get_effect_size_type("hedges_g")
        <EffectSize.HEDGES_G: 'hedges_g'>

        >>> get_effect_size_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid effect size type: invalid
    """
    if not validate_effect_size_type(name):
        msg = f"invalid effect size type: {name}"
        raise ValueError(msg)

    return EffectSize(name)


# Core statistical functions


def _compute_mean(values: Sequence[float]) -> float:
    """Compute the mean of a sequence of values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Mean value.
    """
    return sum(values) / len(values)


def _compute_variance(values: Sequence[float], ddof: int = 1) -> float:
    """Compute the variance of a sequence of values.

    Args:
        values: Sequence of numeric values.
        ddof: Delta degrees of freedom. Defaults to 1 for sample variance.

    Returns:
        Variance value.
    """
    n = len(values)
    if n <= ddof:
        return 0.0
    mean = _compute_mean(values)
    return sum((x - mean) ** 2 for x in values) / (n - ddof)


def _compute_std(values: Sequence[float], ddof: int = 1) -> float:
    """Compute the standard deviation of a sequence of values.

    Args:
        values: Sequence of numeric values.
        ddof: Delta degrees of freedom. Defaults to 1.

    Returns:
        Standard deviation.
    """
    return math.sqrt(_compute_variance(values, ddof))


def _compute_pooled_std(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Compute pooled standard deviation.

    Args:
        values_a: First sequence of values.
        values_b: Second sequence of values.

    Returns:
        Pooled standard deviation.
    """
    n1, n2 = len(values_a), len(values_b)
    var1 = _compute_variance(values_a)
    var2 = _compute_variance(values_b)

    if n1 + n2 <= 2:
        return 0.0

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return math.sqrt(pooled_var)


def calculate_effect_size(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    effect_type: EffectSize = EffectSize.COHENS_D,
) -> float:
    """Calculate effect size between two sets of scores.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.
        effect_type: Type of effect size to compute. Defaults to Cohen's d.

    Returns:
        Effect size value. Positive means A > B.

    Raises:
        ValueError: If scores_a or scores_b is None or empty.
        ValueError: If scores have different lengths.

    Examples:
        >>> scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        >>> scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        >>> effect = calculate_effect_size(scores_a, scores_b)
        >>> effect > 0
        True

        >>> calculate_effect_size([], [0.5])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores_a cannot be empty

        >>> calculate_effect_size([0.5], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scores_b cannot be empty
    """
    if scores_a is None:
        msg = "scores_a cannot be None"
        raise ValueError(msg)

    if len(scores_a) == 0:
        msg = "scores_a cannot be empty"
        raise ValueError(msg)

    if scores_b is None:
        msg = "scores_b cannot be None"
        raise ValueError(msg)

    if len(scores_b) == 0:
        msg = "scores_b cannot be empty"
        raise ValueError(msg)

    if len(scores_a) != len(scores_b):
        msg = (
            f"scores must have the same length, got {len(scores_a)} and {len(scores_b)}"
        )
        raise ValueError(msg)

    mean_a = _compute_mean(scores_a)
    mean_b = _compute_mean(scores_b)
    mean_diff = mean_a - mean_b

    if effect_type == EffectSize.COHENS_D:
        pooled_std = _compute_pooled_std(scores_a, scores_b)
        if pooled_std == 0:
            return (
                0.0 if mean_diff == 0 else float("inf") * (1 if mean_diff > 0 else -1)
            )
        return mean_diff / pooled_std

    elif effect_type == EffectSize.HEDGES_G:
        pooled_std = _compute_pooled_std(scores_a, scores_b)
        if pooled_std == 0:
            return (
                0.0 if mean_diff == 0 else float("inf") * (1 if mean_diff > 0 else -1)
            )
        d = mean_diff / pooled_std
        # Apply Hedges' correction factor
        n = len(scores_a) + len(scores_b)
        correction = 1 - (3 / (4 * n - 9))
        return d * correction

    elif effect_type == EffectSize.GLASS_DELTA:
        # Use scores_b (control group) standard deviation
        std_b = _compute_std(scores_b)
        if std_b == 0:
            return (
                0.0 if mean_diff == 0 else float("inf") * (1 if mean_diff > 0 else -1)
            )
        return mean_diff / std_b

    return 0.0


def bootstrap_confidence_interval(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean difference.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.
        num_bootstrap: Number of bootstrap samples. Defaults to 1000.
        confidence_level: Confidence level (0 to 1). Defaults to 0.95.
        random_seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.

    Raises:
        ValueError: If scores_a or scores_b is None or empty.
        ValueError: If num_bootstrap is not positive.
        ValueError: If confidence_level is not in (0, 1).

    Examples:
        >>> scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        >>> scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        >>> lower, upper = bootstrap_confidence_interval(
        ...     scores_a, scores_b, num_bootstrap=100, random_seed=42
        ... )
        >>> lower < upper
        True

        >>> bootstrap_confidence_interval(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     [], [0.5]
        ... )
        Traceback (most recent call last):
        ValueError: scores_a cannot be empty

        >>> bootstrap_confidence_interval(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     [0.5], [0.5], num_bootstrap=0
        ... )
        Traceback (most recent call last):
        ValueError: num_bootstrap must be positive
    """
    if scores_a is None:
        msg = "scores_a cannot be None"
        raise ValueError(msg)

    if len(scores_a) == 0:
        msg = "scores_a cannot be empty"
        raise ValueError(msg)

    if scores_b is None:
        msg = "scores_b cannot be None"
        raise ValueError(msg)

    if len(scores_b) == 0:
        msg = "scores_b cannot be empty"
        raise ValueError(msg)

    if num_bootstrap <= 0:
        msg = f"num_bootstrap must be positive, got {num_bootstrap}"
        raise ValueError(msg)

    if not 0 < confidence_level < 1:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(msg)

    if random_seed is not None:
        random.seed(random_seed)

    n = len(scores_a)
    bootstrap_diffs: list[float] = []

    for _ in range(num_bootstrap):
        # Resample with replacement
        indices = [random.randint(0, n - 1) for _ in range(n)]
        sample_a = [scores_a[i] for i in indices]
        sample_b = [scores_b[i] for i in indices]

        diff = _compute_mean(sample_a) - _compute_mean(sample_b)
        bootstrap_diffs.append(diff)

    bootstrap_diffs.sort()

    alpha = 1 - confidence_level
    lower_idx = int(alpha / 2 * num_bootstrap)
    upper_idx = int((1 - alpha / 2) * num_bootstrap) - 1

    lower_idx = max(0, min(lower_idx, num_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, num_bootstrap - 1))

    return (bootstrap_diffs[lower_idx], bootstrap_diffs[upper_idx])


def _paired_ttest(
    scores_a: Sequence[float], scores_b: Sequence[float]
) -> tuple[float, float]:
    """Perform paired t-test.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]

    mean_diff = _compute_mean(diffs)
    std_diff = _compute_std(diffs)

    if std_diff == 0:
        # All differences are identical
        if mean_diff == 0:
            return (0.0, 1.0)
        return (float("inf") * (1 if mean_diff > 0 else -1), 0.0)

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    # Approximate two-tailed p-value using t-distribution
    # Using normal approximation for large df
    if df >= 30:
        # Normal approximation
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else:
        # Use t-distribution approximation
        p_value = 2 * _t_cdf(-abs(t_stat), df)

    return (t_stat, p_value)


def _normal_cdf(x: float) -> float:
    """Compute the cumulative distribution function of standard normal.

    Args:
        x: Input value.

    Returns:
        CDF value at x.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_cdf(x: float, df: int) -> float:
    """Approximate t-distribution CDF.

    Args:
        x: Input value.
        df: Degrees of freedom.

    Returns:
        Approximate CDF value at x.
    """
    # Use normal approximation adjusted for degrees of freedom
    # This is a simplification; for precise values, use scipy.stats
    if df >= 30:
        return _normal_cdf(x)

    # Wilson-Hilferty transformation
    z = x * (1 - 1 / (4 * df)) * math.sqrt(1 + x**2 / (2 * df))
    return _normal_cdf(z)


def _wilcoxon_test(
    scores_a: Sequence[float], scores_b: Sequence[float]
) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.

    Returns:
        Tuple of (test_statistic, p_value).
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]

    # Remove zeros and track signs
    abs_diffs_with_signs = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]

    if not abs_diffs_with_signs:
        return (0.0, 1.0)

    # Rank the absolute differences
    abs_diffs_with_signs.sort(key=lambda x: x[0])
    n = len(abs_diffs_with_signs)

    # Compute signed ranks
    w_plus = 0.0
    w_minus = 0.0

    for rank, (_, sign) in enumerate(abs_diffs_with_signs, start=1):
        if sign > 0:
            w_plus += rank
        else:
            w_minus += rank

    w = min(w_plus, w_minus)

    # Normal approximation for p-value (valid for n >= 10)
    if n >= 10:
        mean_w = n * (n + 1) / 4
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if std_w == 0:
            return (w, 1.0)
        z = (w - mean_w) / std_w
        p_value = 2 * _normal_cdf(z)
    else:
        # For small samples, use exact test (simplified)
        # This is a rough approximation
        p_value = 2 * w / (n * (n + 1) / 2)
        p_value = min(1.0, p_value)

    return (w, p_value)


def _permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    num_permutations: int = 1000,
    random_seed: int | None = None,
) -> tuple[float, float]:
    """Perform permutation test.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.
        num_permutations: Number of permutations.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (observed_diff, p_value).
    """
    if random_seed is not None:
        random.seed(random_seed)

    n = len(scores_a)
    observed_diff = _compute_mean(scores_a) - _compute_mean(scores_b)

    # Count permutations with difference >= observed
    count = 0
    combined = list(scores_a) + list(scores_b)

    for _ in range(num_permutations):
        random.shuffle(combined)
        perm_a = combined[:n]
        perm_b = combined[n:]
        perm_diff = _compute_mean(perm_a) - _compute_mean(perm_b)

        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / num_permutations
    return (observed_diff, p_value)


def _mcnemar_test(
    scores_a: Sequence[float], scores_b: Sequence[float]
) -> tuple[float, float]:
    """Perform McNemar's test for paired binary outcomes.

    Args:
        scores_a: Binary scores from model A (0 or 1).
        scores_b: Binary scores from model B (0 or 1).

    Returns:
        Tuple of (chi_squared, p_value).
    """
    # Count discordant pairs
    b = sum(1 for a, bb in zip(scores_a, scores_b, strict=True) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(scores_a, scores_b, strict=True) if a == 0 and bb == 1)

    if b + c == 0:
        return (0.0, 1.0)

    # McNemar's test statistic with continuity correction
    chi_sq = ((abs(b - c) - 1) ** 2) / (b + c)

    # p-value from chi-squared distribution with 1 df
    # Using approximation
    p_value = 1 - _chi_squared_cdf(chi_sq, 1)

    return (chi_sq, p_value)


def _chi_squared_cdf(x: float, df: int) -> float:
    """Approximate chi-squared CDF.

    Args:
        x: Input value.
        df: Degrees of freedom.

    Returns:
        Approximate CDF value.
    """
    if x <= 0:
        return 0.0

    # Wilson-Hilferty transformation
    h = 2 / (9 * df)
    z = (x / df) ** (1 / 3) - (1 - h)
    z /= math.sqrt(h)

    return _normal_cdf(z)


def run_significance_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    config: ComparisonConfig,
) -> tuple[float, float]:
    """Run statistical significance test.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.
        config: Comparison configuration.

    Returns:
        Tuple of (test_statistic, p_value).

    Raises:
        ValueError: If scores are None or empty.
        ValueError: If config is None.

    Examples:
        >>> scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        >>> scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        >>> config = ComparisonConfig()
        >>> stat, pval = run_significance_test(scores_a, scores_b, config)
        >>> pval < 1.0
        True

        >>> run_significance_test(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     [], [0.5], config
        ... )
        Traceback (most recent call last):
        ValueError: scores_a cannot be empty
    """
    if scores_a is None:
        msg = "scores_a cannot be None"
        raise ValueError(msg)

    if len(scores_a) == 0:
        msg = "scores_a cannot be empty"
        raise ValueError(msg)

    if scores_b is None:
        msg = "scores_b cannot be None"
        raise ValueError(msg)

    if len(scores_b) == 0:
        msg = "scores_b cannot be empty"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.method == ComparisonMethod.PAIRED_TTEST:
        return _paired_ttest(scores_a, scores_b)

    elif config.method == ComparisonMethod.WILCOXON:
        return _wilcoxon_test(scores_a, scores_b)

    elif config.method == ComparisonMethod.PERMUTATION:
        return _permutation_test(
            scores_a, scores_b, config.num_bootstrap, config.random_seed
        )

    elif config.method == ComparisonMethod.BOOTSTRAP:
        # Bootstrap test: compute p-value from bootstrap distribution
        if config.random_seed is not None:
            random.seed(config.random_seed)

        n = len(scores_a)
        observed_diff = _compute_mean(scores_a) - _compute_mean(scores_b)

        # Bootstrap under null hypothesis (pooled samples)
        pooled = list(scores_a) + list(scores_b)
        count = 0

        for _ in range(config.num_bootstrap):
            sample_a = [random.choice(pooled) for _ in range(n)]
            sample_b = [random.choice(pooled) for _ in range(n)]
            boot_diff = _compute_mean(sample_a) - _compute_mean(sample_b)

            if abs(boot_diff) >= abs(observed_diff):
                count += 1

        p_value = count / config.num_bootstrap
        return (observed_diff, p_value)

    elif config.method == ComparisonMethod.MCNEMAR:
        return _mcnemar_test(scores_a, scores_b)

    return (0.0, 1.0)


def compare_models(
    result_a: ModelResult,
    result_b: ModelResult,
    config: ComparisonConfig | None = None,
) -> ComparisonResult:
    """Compare two models using statistical testing.

    Args:
        result_a: Results from first model.
        result_b: Results from second model.
        config: Comparison configuration. Uses defaults if None.

    Returns:
        ComparisonResult with statistical analysis.

    Raises:
        ValueError: If result_a or result_b is None.
        ValueError: If results have different lengths.

    Examples:
        >>> result_a = ModelResult(
        ...     model_name="model_a",
        ...     scores=(0.8, 0.85, 0.9, 0.82, 0.88),
        ... )
        >>> result_b = ModelResult(
        ...     model_name="model_b",
        ...     scores=(0.7, 0.75, 0.72, 0.68, 0.71),
        ... )
        >>> comparison = compare_models(result_a, result_b)
        >>> comparison.model_a
        'model_a'
        >>> comparison.model_b
        'model_b'

        >>> compare_models(None, result_b)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result_a cannot be None
    """
    if result_a is None:
        msg = "result_a cannot be None"
        raise ValueError(msg)

    if result_b is None:
        msg = "result_b cannot be None"
        raise ValueError(msg)

    validate_model_result(result_a)
    validate_model_result(result_b)

    if len(result_a.scores) != len(result_b.scores):
        msg = (
            f"results must have the same number of scores, "
            f"got {len(result_a.scores)} and {len(result_b.scores)}"
        )
        raise ValueError(msg)

    if config is None:
        config = create_comparison_config()

    # Run significance test
    test_stat, p_value = run_significance_test(result_a.scores, result_b.scores, config)

    # Calculate effect size
    effect_size = calculate_effect_size(
        result_a.scores, result_b.scores, config.effect_size_type
    )

    # Compute confidence interval
    ci = bootstrap_confidence_interval(
        result_a.scores,
        result_b.scores,
        config.num_bootstrap,
        1 - config.significance_level.value,
        config.random_seed,
    )

    # Determine significance
    is_significant = p_value < config.significance_level.value

    return ComparisonResult(
        model_a=result_a.model_name,
        model_b=result_b.model_name,
        p_value=p_value,
        effect_size=effect_size,
        is_significant=is_significant,
        confidence_interval=ci,
        test_statistic=test_stat,
        method=config.method,
    )


def format_comparison_table(results: Sequence[ComparisonResult]) -> str:
    """Format comparison results as a table.

    Args:
        results: Sequence of comparison results.

    Returns:
        Formatted table string.

    Raises:
        ValueError: If results is None.

    Examples:
        >>> result = ComparisonResult(
        ...     model_a="model_1", model_b="model_2", p_value=0.03,
        ...     effect_size=0.5, is_significant=True,
        ...     confidence_interval=(0.1, 0.9), test_statistic=2.5,
        ...     method=ComparisonMethod.PAIRED_TTEST,
        ... )
        >>> table = format_comparison_table([result])
        >>> "model_1" in table
        True
        >>> "model_2" in table
        True

        >>> format_comparison_table(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: results cannot be None
    """
    if results is None:
        msg = "results cannot be None"
        raise ValueError(msg)

    if len(results) == 0:
        return "No comparison results to display."

    lines = [
        "Model Comparison Results",
        "=" * 80,
        f"{'Model A':<15} {'Model B':<15} "
        f"{'p-value':<10} {'Effect':<10} {'Sig?':<6} {'CI'}",
        "-" * 80,
    ]

    for r in results:
        sig_marker = "*" if r.is_significant else ""
        ci_str = f"[{r.confidence_interval[0]:.3f}, {r.confidence_interval[1]:.3f}]"
        lines.append(
            f"{r.model_a:<15} {r.model_b:<15} {r.p_value:<10.4f} "
            f"{r.effect_size:<10.3f} {sig_marker:<6} {ci_str}"
        )

    lines.append("-" * 80)
    lines.append("* indicates statistical significance")

    return "\n".join(lines)


def format_comparison_stats(stats: ComparisonStats) -> str:
    """Format comparison statistics as a human-readable string.

    Args:
        stats: Comparison statistics to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If stats is None.

    Examples:
        >>> stats = ComparisonStats(
        ...     total_comparisons=10, significant_count=3,
        ...     avg_effect_size=0.4, max_effect_size=0.8, min_p_value=0.001,
        ... )
        >>> formatted = format_comparison_stats(stats)
        >>> "Total Comparisons" in formatted
        True
        >>> "10" in formatted
        True

        >>> format_comparison_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    sig_rate = stats.significant_count / stats.total_comparisons * 100

    lines = [
        "Comparison Statistics Summary",
        "=" * 40,
        f"Total Comparisons:    {stats.total_comparisons}",
        f"Significant Results:  {stats.significant_count} ({sig_rate:.1f}%)",
        f"Average Effect Size:  {stats.avg_effect_size:.4f}",
        f"Maximum Effect Size:  {stats.max_effect_size:.4f}",
        f"Minimum p-value:      {stats.min_p_value:.4e}",
    ]

    return "\n".join(lines)


def get_recommended_comparison_config(
    sample_size: int,
    distribution: str = "unknown",
) -> ComparisonConfig:
    """Get recommended comparison configuration based on data characteristics.

    Args:
        sample_size: Number of samples in each group.
        distribution: Distribution assumption
            ("normal", "non-normal", "binary", "unknown").

    Returns:
        Recommended ComparisonConfig.

    Raises:
        ValueError: If sample_size is not positive.
        ValueError: If distribution is empty.

    Examples:
        >>> config = get_recommended_comparison_config(100)
        >>> config.method
        <ComparisonMethod.PAIRED_TTEST: 'paired_ttest'>

        >>> config = get_recommended_comparison_config(10, "non-normal")
        >>> config.method
        <ComparisonMethod.WILCOXON: 'wilcoxon'>

        >>> config = get_recommended_comparison_config(50, "binary")
        >>> config.method
        <ComparisonMethod.MCNEMAR: 'mcnemar'>

        >>> get_recommended_comparison_config(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sample_size must be positive

        >>> get_recommended_comparison_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     10, ""
        ... )
        Traceback (most recent call last):
        ValueError: distribution cannot be empty
    """
    if sample_size <= 0:
        msg = f"sample_size must be positive, got {sample_size}"
        raise ValueError(msg)

    if not distribution:
        msg = "distribution cannot be empty"
        raise ValueError(msg)

    dist_lower = distribution.lower()

    # Choose method based on distribution and sample size
    if dist_lower == "binary":
        method = ComparisonMethod.MCNEMAR
    elif dist_lower == "normal" or (dist_lower == "unknown" and sample_size >= 30):
        method = ComparisonMethod.PAIRED_TTEST
    elif dist_lower == "non-normal" or sample_size < 30:
        if sample_size < 10:
            method = ComparisonMethod.PERMUTATION
        else:
            method = ComparisonMethod.WILCOXON
    else:
        method = ComparisonMethod.BOOTSTRAP

    # Adjust bootstrap samples based on sample size
    if sample_size < 50:
        num_bootstrap = 5000
    elif sample_size < 200:
        num_bootstrap = 2000
    else:
        num_bootstrap = 1000

    return ComparisonConfig(
        method=method,
        significance_level=SignificanceLevel.P05,
        num_bootstrap=num_bootstrap,
        effect_size_type=EffectSize.HEDGES_G
        if sample_size < 20
        else EffectSize.COHENS_D,
    )
