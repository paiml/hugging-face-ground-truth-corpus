"""Bias detection and fairness evaluation utilities.

This module provides tools for detecting bias in ML models, computing
fairness metrics, and generating recommendations for bias mitigation.

Examples:
    >>> from hf_gtc.evaluation.bias import BiasDetectionConfig, FairnessMetric
    >>> config = BiasDetectionConfig(
    ...     protected_attributes=["gender"],
    ...     metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
    ... )
    >>> config.protected_attributes
    ['gender']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class FairnessMetric(Enum):
    """Fairness metrics for bias evaluation.

    Attributes:
        DEMOGRAPHIC_PARITY: Equal positive prediction rates across groups.
        EQUALIZED_ODDS: Equal TPR and FPR across groups.
        CALIBRATION: Predicted probabilities match actual outcomes.
        PREDICTIVE_PARITY: Equal PPV across groups.

    Examples:
        >>> FairnessMetric.DEMOGRAPHIC_PARITY.value
        'demographic_parity'
        >>> FairnessMetric.EQUALIZED_ODDS.value
        'equalized_odds'
    """

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    PREDICTIVE_PARITY = "predictive_parity"


VALID_FAIRNESS_METRICS = frozenset(m.value for m in FairnessMetric)


class BiasType(Enum):
    """Types of bias that can be detected.

    Attributes:
        GENDER: Gender-related bias.
        RACE: Race-related bias.
        AGE: Age-related bias.
        RELIGION: Religion-related bias.
        NATIONALITY: Nationality-related bias.

    Examples:
        >>> BiasType.GENDER.value
        'gender'
        >>> BiasType.RACE.value
        'race'
    """

    GENDER = "gender"
    RACE = "race"
    AGE = "age"
    RELIGION = "religion"
    NATIONALITY = "nationality"


VALID_BIAS_TYPES = frozenset(b.value for b in BiasType)


class MitigationStrategy(Enum):
    """Strategies for mitigating detected bias.

    Attributes:
        REWEIGHTING: Adjust sample weights to balance groups.
        RESAMPLING: Resample data to balance groups.
        ADVERSARIAL: Use adversarial training to remove bias.
        CALIBRATION: Calibrate predictions across groups.

    Examples:
        >>> MitigationStrategy.REWEIGHTING.value
        'reweighting'
        >>> MitigationStrategy.ADVERSARIAL.value
        'adversarial'
    """

    REWEIGHTING = "reweighting"
    RESAMPLING = "resampling"
    ADVERSARIAL = "adversarial"
    CALIBRATION = "calibration"


VALID_MITIGATION_STRATEGIES = frozenset(s.value for s in MitigationStrategy)


@dataclass(frozen=True, slots=True)
class BiasDetectionConfig:
    """Configuration for bias detection.

    Attributes:
        protected_attributes: Attributes to check for bias (e.g., gender, race).
        metrics: Fairness metrics to compute.
        threshold: Disparity threshold for flagging bias. Defaults to 0.1.
        intersectional: Whether to check intersectional bias. Defaults to False.

    Examples:
        >>> config = BiasDetectionConfig(
        ...     protected_attributes=["gender", "race"],
        ...     metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        ... )
        >>> config.threshold
        0.1
        >>> config.intersectional
        False
    """

    protected_attributes: list[str]
    metrics: list[FairnessMetric]
    threshold: float = 0.1
    intersectional: bool = False


@dataclass(frozen=True, slots=True)
class FairnessConstraint:
    """A fairness constraint to enforce during training or evaluation.

    Attributes:
        metric: The fairness metric to constrain.
        threshold: Maximum allowed disparity. Defaults to 0.05.
        group_comparison: Type of group comparison ("pairwise" or "reference").
        slack: Allowed slack in the constraint. Defaults to 0.0.

    Examples:
        >>> constraint = FairnessConstraint(
        ...     metric=FairnessMetric.DEMOGRAPHIC_PARITY,
        ...     threshold=0.1,
        ...     group_comparison="pairwise",
        ... )
        >>> constraint.metric
        <FairnessMetric.DEMOGRAPHIC_PARITY: 'demographic_parity'>
        >>> constraint.slack
        0.0
    """

    metric: FairnessMetric
    threshold: float = 0.05
    group_comparison: str = "pairwise"
    slack: float = 0.0


@dataclass(frozen=True, slots=True)
class BiasAuditResult:
    """Results from a bias audit.

    Attributes:
        disparities: Dictionary mapping attribute to disparity scores.
        scores_by_group: Metric scores broken down by group.
        overall_fairness: Overall fairness score (0-1, higher is better).
        recommendations: List of recommendations for bias mitigation.

    Examples:
        >>> result = BiasAuditResult(
        ...     disparities={"gender": 0.15},
        ...     scores_by_group={"male": 0.8, "female": 0.65},
        ...     overall_fairness=0.75,
        ...     recommendations=["Consider reweighting training data"],
        ... )
        >>> result.overall_fairness
        0.75
    """

    disparities: dict[str, float]
    scores_by_group: dict[str, float]
    overall_fairness: float
    recommendations: list[str]


@dataclass(frozen=True, slots=True)
class StereotypeConfig:
    """Configuration for stereotype detection in text.

    Attributes:
        lexicon_path: Path to stereotype lexicon file. Defaults to None.
        categories: Categories of stereotypes to detect.
        sensitivity: Sensitivity level (0.0-1.0). Defaults to 0.5.
        context_window: Number of tokens for context. Defaults to 5.

    Examples:
        >>> config = StereotypeConfig(
        ...     categories=["gender", "race"],
        ...     sensitivity=0.7,
        ... )
        >>> config.sensitivity
        0.7
        >>> config.context_window
        5
    """

    lexicon_path: str | None = None
    categories: list[str] = field(default_factory=list)
    sensitivity: float = 0.5
    context_window: int = 5


def validate_bias_detection_config(config: BiasDetectionConfig) -> None:
    """Validate a BiasDetectionConfig.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If protected_attributes is empty.
        ValueError: If metrics is empty.
        ValueError: If threshold is not in (0, 1].

    Examples:
        >>> config = BiasDetectionConfig(
        ...     protected_attributes=["gender"],
        ...     metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        ... )
        >>> validate_bias_detection_config(config)  # No error

        >>> validate_bias_detection_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = BiasDetectionConfig(protected_attributes=[], metrics=[])
        >>> validate_bias_detection_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: protected_attributes cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.protected_attributes:
        msg = "protected_attributes cannot be empty"
        raise ValueError(msg)

    if not config.metrics:
        msg = "metrics cannot be empty"
        raise ValueError(msg)

    if not 0 < config.threshold <= 1:
        msg = f"threshold must be in (0, 1], got {config.threshold}"
        raise ValueError(msg)


def create_bias_detection_config(
    protected_attributes: list[str],
    metrics: list[FairnessMetric],
    *,
    threshold: float = 0.1,
    intersectional: bool = False,
) -> BiasDetectionConfig:
    """Create a validated BiasDetectionConfig.

    Args:
        protected_attributes: Attributes to check for bias.
        metrics: Fairness metrics to compute.
        threshold: Disparity threshold. Defaults to 0.1.
        intersectional: Whether to check intersectional bias. Defaults to False.

    Returns:
        Validated BiasDetectionConfig instance.

    Raises:
        ValueError: If protected_attributes is empty.
        ValueError: If metrics is empty.

    Examples:
        >>> config = create_bias_detection_config(
        ...     protected_attributes=["gender"],
        ...     metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        ... )
        >>> config.protected_attributes
        ['gender']

        >>> create_bias_detection_config([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: protected_attributes cannot be empty
    """
    config = BiasDetectionConfig(
        protected_attributes=protected_attributes,
        metrics=metrics,
        threshold=threshold,
        intersectional=intersectional,
    )
    validate_bias_detection_config(config)
    return config


def validate_fairness_constraint(constraint: FairnessConstraint) -> None:
    """Validate a FairnessConstraint.

    Args:
        constraint: Constraint to validate.

    Raises:
        ValueError: If constraint is None.
        ValueError: If threshold is not positive.
        ValueError: If group_comparison is invalid.
        ValueError: If slack is negative.

    Examples:
        >>> constraint = FairnessConstraint(
        ...     metric=FairnessMetric.DEMOGRAPHIC_PARITY,
        ...     group_comparison="pairwise",
        ... )
        >>> validate_fairness_constraint(constraint)  # No error

        >>> validate_fairness_constraint(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: constraint cannot be None
    """
    if constraint is None:
        msg = "constraint cannot be None"
        raise ValueError(msg)

    if constraint.threshold <= 0:
        msg = f"threshold must be positive, got {constraint.threshold}"
        raise ValueError(msg)

    valid_comparisons = {"pairwise", "reference"}
    if constraint.group_comparison not in valid_comparisons:
        msg = (
            f"group_comparison must be one of {valid_comparisons}, "
            f"got {constraint.group_comparison}"
        )
        raise ValueError(msg)

    if constraint.slack < 0:
        msg = f"slack cannot be negative, got {constraint.slack}"
        raise ValueError(msg)


def create_fairness_constraint(
    metric: FairnessMetric,
    *,
    threshold: float = 0.05,
    group_comparison: str = "pairwise",
    slack: float = 0.0,
) -> FairnessConstraint:
    """Create a validated FairnessConstraint.

    Args:
        metric: The fairness metric to constrain.
        threshold: Maximum allowed disparity. Defaults to 0.05.
        group_comparison: Type of group comparison. Defaults to "pairwise".
        slack: Allowed slack in the constraint. Defaults to 0.0.

    Returns:
        Validated FairnessConstraint instance.

    Raises:
        ValueError: If threshold is not positive.
        ValueError: If group_comparison is invalid.

    Examples:
        >>> constraint = create_fairness_constraint(
        ...     FairnessMetric.EQUALIZED_ODDS,
        ...     threshold=0.1,
        ... )
        >>> constraint.threshold
        0.1

        >>> create_fairness_constraint(
        ...     FairnessMetric.DEMOGRAPHIC_PARITY,
        ...     group_comparison="invalid",
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: group_comparison must be one of ...
    """
    constraint = FairnessConstraint(
        metric=metric,
        threshold=threshold,
        group_comparison=group_comparison,
        slack=slack,
    )
    validate_fairness_constraint(constraint)
    return constraint


def validate_stereotype_config(config: StereotypeConfig) -> None:
    """Validate a StereotypeConfig.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If sensitivity is not in [0, 1].
        ValueError: If context_window is not positive.

    Examples:
        >>> config = StereotypeConfig(categories=["gender"], sensitivity=0.5)
        >>> validate_stereotype_config(config)  # No error

        >>> validate_stereotype_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = StereotypeConfig(sensitivity=1.5)
        >>> validate_stereotype_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sensitivity must be in [0, 1]
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not 0 <= config.sensitivity <= 1:
        msg = f"sensitivity must be in [0, 1], got {config.sensitivity}"
        raise ValueError(msg)

    if config.context_window <= 0:
        msg = f"context_window must be positive, got {config.context_window}"
        raise ValueError(msg)


def create_stereotype_config(
    *,
    lexicon_path: str | None = None,
    categories: list[str] | None = None,
    sensitivity: float = 0.5,
    context_window: int = 5,
) -> StereotypeConfig:
    """Create a validated StereotypeConfig.

    Args:
        lexicon_path: Path to stereotype lexicon file. Defaults to None.
        categories: Categories of stereotypes to detect. Defaults to empty list.
        sensitivity: Sensitivity level (0.0-1.0). Defaults to 0.5.
        context_window: Number of tokens for context. Defaults to 5.

    Returns:
        Validated StereotypeConfig instance.

    Raises:
        ValueError: If sensitivity is not in [0, 1].
        ValueError: If context_window is not positive.

    Examples:
        >>> config = create_stereotype_config(
        ...     categories=["gender", "race"],
        ...     sensitivity=0.8,
        ... )
        >>> config.categories
        ['gender', 'race']

        >>> create_stereotype_config(
        ...     sensitivity=2.0,
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sensitivity must be in [0, 1]
    """
    config = StereotypeConfig(
        lexicon_path=lexicon_path,
        categories=categories if categories is not None else [],
        sensitivity=sensitivity,
        context_window=context_window,
    )
    validate_stereotype_config(config)
    return config


def list_fairness_metrics() -> list[str]:
    """List all available fairness metrics.

    Returns:
        Sorted list of fairness metric names.

    Examples:
        >>> metrics = list_fairness_metrics()
        >>> "demographic_parity" in metrics
        True
        >>> "equalized_odds" in metrics
        True
        >>> metrics == sorted(metrics)
        True
    """
    return sorted(VALID_FAIRNESS_METRICS)


def validate_fairness_metric(metric: str) -> bool:
    """Validate if a string is a valid fairness metric.

    Args:
        metric: The metric string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_fairness_metric("demographic_parity")
        True
        >>> validate_fairness_metric("equalized_odds")
        True
        >>> validate_fairness_metric("invalid")
        False
        >>> validate_fairness_metric("")
        False
    """
    return metric in VALID_FAIRNESS_METRICS


def get_fairness_metric(name: str) -> FairnessMetric:
    """Get FairnessMetric enum from string name.

    Args:
        name: Name of the fairness metric.

    Returns:
        Corresponding FairnessMetric enum value.

    Raises:
        ValueError: If name is not a valid fairness metric.

    Examples:
        >>> get_fairness_metric("demographic_parity")
        <FairnessMetric.DEMOGRAPHIC_PARITY: 'demographic_parity'>

        >>> get_fairness_metric("equalized_odds")
        <FairnessMetric.EQUALIZED_ODDS: 'equalized_odds'>

        >>> get_fairness_metric("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid fairness metric: invalid
    """
    if not validate_fairness_metric(name):
        msg = f"invalid fairness metric: {name}"
        raise ValueError(msg)

    return FairnessMetric(name)


def list_bias_types() -> list[str]:
    """List all available bias types.

    Returns:
        Sorted list of bias type names.

    Examples:
        >>> types = list_bias_types()
        >>> "gender" in types
        True
        >>> "race" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_BIAS_TYPES)


def validate_bias_type(bias_type: str) -> bool:
    """Validate if a string is a valid bias type.

    Args:
        bias_type: The bias type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_bias_type("gender")
        True
        >>> validate_bias_type("race")
        True
        >>> validate_bias_type("invalid")
        False
        >>> validate_bias_type("")
        False
    """
    return bias_type in VALID_BIAS_TYPES


def get_bias_type(name: str) -> BiasType:
    """Get BiasType enum from string name.

    Args:
        name: Name of the bias type.

    Returns:
        Corresponding BiasType enum value.

    Raises:
        ValueError: If name is not a valid bias type.

    Examples:
        >>> get_bias_type("gender")
        <BiasType.GENDER: 'gender'>

        >>> get_bias_type("age")
        <BiasType.AGE: 'age'>

        >>> get_bias_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid bias type: invalid
    """
    if not validate_bias_type(name):
        msg = f"invalid bias type: {name}"
        raise ValueError(msg)

    return BiasType(name)


def list_mitigation_strategies() -> list[str]:
    """List all available mitigation strategies.

    Returns:
        Sorted list of mitigation strategy names.

    Examples:
        >>> strategies = list_mitigation_strategies()
        >>> "reweighting" in strategies
        True
        >>> "adversarial" in strategies
        True
        >>> strategies == sorted(strategies)
        True
    """
    return sorted(VALID_MITIGATION_STRATEGIES)


def validate_mitigation_strategy(strategy: str) -> bool:
    """Validate if a string is a valid mitigation strategy.

    Args:
        strategy: The strategy string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_mitigation_strategy("reweighting")
        True
        >>> validate_mitigation_strategy("adversarial")
        True
        >>> validate_mitigation_strategy("invalid")
        False
        >>> validate_mitigation_strategy("")
        False
    """
    return strategy in VALID_MITIGATION_STRATEGIES


def get_mitigation_strategy(name: str) -> MitigationStrategy:
    """Get MitigationStrategy enum from string name.

    Args:
        name: Name of the mitigation strategy.

    Returns:
        Corresponding MitigationStrategy enum value.

    Raises:
        ValueError: If name is not a valid mitigation strategy.

    Examples:
        >>> get_mitigation_strategy("reweighting")
        <MitigationStrategy.REWEIGHTING: 'reweighting'>

        >>> get_mitigation_strategy("adversarial")
        <MitigationStrategy.ADVERSARIAL: 'adversarial'>

        >>> get_mitigation_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid mitigation strategy: invalid
    """
    if not validate_mitigation_strategy(name):
        msg = f"invalid mitigation strategy: {name}"
        raise ValueError(msg)

    return MitigationStrategy(name)


def calculate_demographic_parity(
    predictions: Sequence[int],
    groups: Sequence[str],
) -> dict[str, float]:
    """Calculate demographic parity metric across groups.

    Demographic parity requires equal positive prediction rates
    across different groups.

    Args:
        predictions: Binary predictions (0 or 1).
        groups: Group membership for each prediction.

    Returns:
        Dictionary with positive prediction rate per group.

    Raises:
        ValueError: If predictions is None or empty.
        ValueError: If groups is None or empty.
        ValueError: If lengths don't match.

    Examples:
        >>> predictions = [1, 0, 1, 1, 0, 1]
        >>> groups = ["A", "A", "A", "B", "B", "B"]
        >>> result = calculate_demographic_parity(predictions, groups)
        >>> round(result["A"], 4)
        0.6667
        >>> round(result["B"], 4)
        0.6667

        >>> calculate_demographic_parity([], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty

        >>> calculate_demographic_parity([1], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: groups cannot be empty
    """
    if predictions is None:
        msg = "predictions cannot be None"
        raise ValueError(msg)

    if len(predictions) == 0:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if groups is None:
        msg = "groups cannot be None"
        raise ValueError(msg)

    if len(groups) == 0:
        msg = "groups cannot be empty"
        raise ValueError(msg)

    if len(predictions) != len(groups):
        msg = (
            f"predictions and groups must have same length, "
            f"got {len(predictions)} and {len(groups)}"
        )
        raise ValueError(msg)

    # Calculate positive rate per group
    group_counts: dict[str, int] = {}
    group_positives: dict[str, int] = {}

    for pred, group in zip(predictions, groups, strict=True):
        group_counts[group] = group_counts.get(group, 0) + 1
        if pred == 1:
            group_positives[group] = group_positives.get(group, 0) + 1

    result: dict[str, float] = {}
    for group, count in group_counts.items():
        positives = group_positives.get(group, 0)
        result[group] = positives / count

    return result


def calculate_equalized_odds(
    predictions: Sequence[int],
    labels: Sequence[int],
    groups: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Calculate equalized odds metric across groups.

    Equalized odds requires equal true positive rates (TPR) and
    false positive rates (FPR) across different groups.

    Args:
        predictions: Binary predictions (0 or 1).
        labels: Ground truth labels (0 or 1).
        groups: Group membership for each sample.

    Returns:
        Dictionary with TPR and FPR per group.

    Raises:
        ValueError: If any input is None or empty.
        ValueError: If lengths don't match.

    Examples:
        >>> predictions = [1, 0, 1, 1, 0, 0]
        >>> labels = [1, 0, 0, 1, 0, 1]
        >>> groups = ["A", "A", "A", "B", "B", "B"]
        >>> result = calculate_equalized_odds(predictions, labels, groups)
        >>> "A" in result
        True
        >>> "tpr" in result["A"]
        True

        >>> calculate_equalized_odds([], [], [])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: predictions cannot be empty
    """
    if predictions is None:
        msg = "predictions cannot be None"
        raise ValueError(msg)

    if len(predictions) == 0:
        msg = "predictions cannot be empty"
        raise ValueError(msg)

    if labels is None:
        msg = "labels cannot be None"
        raise ValueError(msg)

    if len(labels) == 0:
        msg = "labels cannot be empty"
        raise ValueError(msg)

    if groups is None:
        msg = "groups cannot be None"
        raise ValueError(msg)

    if len(groups) == 0:
        msg = "groups cannot be empty"
        raise ValueError(msg)

    if not (len(predictions) == len(labels) == len(groups)):
        msg = "predictions, labels, and groups must have same length"
        raise ValueError(msg)

    # Calculate TPR and FPR per group
    group_stats: dict[str, dict[str, int]] = {}

    for pred, label, group in zip(predictions, labels, groups, strict=True):
        if group not in group_stats:
            group_stats[group] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        if pred == 1 and label == 1:
            group_stats[group]["tp"] += 1
        elif pred == 1 and label == 0:
            group_stats[group]["fp"] += 1
        elif pred == 0 and label == 0:
            group_stats[group]["tn"] += 1
        else:  # pred == 0 and label == 1
            group_stats[group]["fn"] += 1

    result: dict[str, dict[str, float]] = {}
    for group, stats in group_stats.items():
        tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]

        # TPR = TP / (TP + FN)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        result[group] = {"tpr": tpr, "fpr": fpr}

    return result


def detect_stereotypes(
    text: str,
    config: StereotypeConfig,
) -> list[dict[str, Any]]:
    """Detect potential stereotypes in text.

    Args:
        text: Text to analyze for stereotypes.
        config: Configuration for stereotype detection.

    Returns:
        List of detected stereotype instances with metadata.

    Raises:
        ValueError: If text is None.
        ValueError: If config is None.

    Examples:
        >>> config = StereotypeConfig(categories=["gender"], sensitivity=0.5)
        >>> result = detect_stereotypes("Some neutral text.", config)
        >>> isinstance(result, list)
        True

        >>> detect_stereotypes(None, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: text cannot be None

        >>> detect_stereotypes("text", None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if text is None:
        msg = "text cannot be None"
        raise ValueError(msg)

    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    validate_stereotype_config(config)

    # Basic stereotype detection patterns (simplified for demonstration)
    # In production, this would use more sophisticated NLP models
    stereotype_patterns: dict[str, list[str]] = {
        "gender": [
            "women are emotional",
            "men are strong",
            "women belong in",
            "men should be",
        ],
        "race": [
            "all asians are",
            "black people are",
            "white people are",
        ],
        "age": [
            "old people are",
            "young people are",
            "millennials are",
            "boomers are",
        ],
    }

    detections: list[dict[str, Any]] = []
    text_lower = text.lower()

    for category in config.categories:
        if category in stereotype_patterns:
            for pattern in stereotype_patterns[category]:
                if pattern in text_lower:
                    start_idx = text_lower.find(pattern)
                    detections.append(
                        {
                            "category": category,
                            "pattern": pattern,
                            "start_index": start_idx,
                            "end_index": start_idx + len(pattern),
                            "confidence": config.sensitivity,
                        }
                    )

    return detections


def calculate_disparity_score(
    group_rates: dict[str, float],
) -> float:
    """Calculate disparity score from group-wise rates.

    The disparity score is the maximum difference between any
    two groups' rates. A score of 0 indicates perfect parity.

    Args:
        group_rates: Dictionary mapping group names to rates.

    Returns:
        Maximum disparity between any two groups.

    Raises:
        ValueError: If group_rates is None.
        ValueError: If group_rates has fewer than 2 groups.

    Examples:
        >>> group_rates = {"A": 0.8, "B": 0.7, "C": 0.6}
        >>> round(calculate_disparity_score(group_rates), 4)
        0.2

        >>> group_rates = {"A": 0.5, "B": 0.5}
        >>> calculate_disparity_score(group_rates)
        0.0

        >>> calculate_disparity_score(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: group_rates cannot be None

        >>> calculate_disparity_score({"A": 0.5})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: group_rates must have at least 2 groups
    """
    if group_rates is None:
        msg = "group_rates cannot be None"
        raise ValueError(msg)

    if len(group_rates) < 2:
        msg = f"group_rates must have at least 2 groups, got {len(group_rates)}"
        raise ValueError(msg)

    rates = list(group_rates.values())
    return max(rates) - min(rates)


def format_bias_audit(result: BiasAuditResult) -> str:
    """Format a bias audit result as human-readable string.

    Args:
        result: Bias audit result to format.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> result = BiasAuditResult(
        ...     disparities={"gender": 0.15},
        ...     scores_by_group={"male": 0.8, "female": 0.65},
        ...     overall_fairness=0.75,
        ...     recommendations=["Consider reweighting"],
        ... )
        >>> "Fairness" in format_bias_audit(result)
        True

        >>> format_bias_audit(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    if result is None:
        msg = "result cannot be None"
        raise ValueError(msg)

    lines = [
        "Bias Audit Report",
        "=" * 40,
        f"Overall Fairness Score: {result.overall_fairness:.2%}",
        "",
        "Disparities by Attribute:",
    ]

    for attr, disparity in sorted(result.disparities.items()):
        lines.append(f"  {attr}: {disparity:.4f}")

    lines.extend(["", "Scores by Group:"])
    for group, score in sorted(result.scores_by_group.items()):
        lines.append(f"  {group}: {score:.4f}")

    if result.recommendations:
        lines.extend(["", "Recommendations:"])
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


def get_recommended_bias_config(task_type: str) -> BiasDetectionConfig:
    """Get recommended bias detection configuration for a task type.

    Args:
        task_type: Type of ML task (e.g., "classification", "generation", "hiring").

    Returns:
        Recommended BiasDetectionConfig for the task.

    Raises:
        ValueError: If task_type is None or empty.

    Examples:
        >>> config = get_recommended_bias_config("classification")
        >>> len(config.metrics) > 0
        True

        >>> config = get_recommended_bias_config("hiring")
        >>> FairnessMetric.DEMOGRAPHIC_PARITY in config.metrics
        True

        >>> get_recommended_bias_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_type cannot be empty

        >>> get_recommended_bias_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_type cannot be None
    """
    if task_type is None:
        msg = "task_type cannot be None"
        raise ValueError(msg)

    if not task_type:
        msg = "task_type cannot be empty"
        raise ValueError(msg)

    # Default configuration
    default_config = BiasDetectionConfig(
        protected_attributes=["gender", "race", "age"],
        metrics=[FairnessMetric.DEMOGRAPHIC_PARITY, FairnessMetric.EQUALIZED_ODDS],
        threshold=0.1,
        intersectional=False,
    )

    # Task-specific configurations
    task_configs: dict[str, BiasDetectionConfig] = {
        "classification": BiasDetectionConfig(
            protected_attributes=["gender", "race"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY, FairnessMetric.EQUALIZED_ODDS],
            threshold=0.1,
            intersectional=False,
        ),
        "generation": BiasDetectionConfig(
            protected_attributes=["gender", "race", "religion", "nationality"],
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
            threshold=0.15,
            intersectional=True,
        ),
        "hiring": BiasDetectionConfig(
            protected_attributes=["gender", "race", "age"],
            metrics=[
                FairnessMetric.DEMOGRAPHIC_PARITY,
                FairnessMetric.EQUALIZED_ODDS,
                FairnessMetric.PREDICTIVE_PARITY,
            ],
            threshold=0.05,
            intersectional=True,
        ),
        "credit": BiasDetectionConfig(
            protected_attributes=["gender", "race", "age"],
            metrics=[
                FairnessMetric.DEMOGRAPHIC_PARITY,
                FairnessMetric.EQUALIZED_ODDS,
                FairnessMetric.CALIBRATION,
            ],
            threshold=0.05,
            intersectional=True,
        ),
        "healthcare": BiasDetectionConfig(
            protected_attributes=["gender", "race", "age"],
            metrics=[
                FairnessMetric.EQUALIZED_ODDS,
                FairnessMetric.CALIBRATION,
                FairnessMetric.PREDICTIVE_PARITY,
            ],
            threshold=0.03,
            intersectional=True,
        ),
    }

    return task_configs.get(task_type.lower(), default_config)
