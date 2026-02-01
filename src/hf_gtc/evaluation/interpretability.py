"""Model interpretability utilities for HuggingFace models.

This module provides utilities for computing feature attributions,
visualizing attention patterns, and interpreting model predictions.

Examples:
    >>> from hf_gtc.evaluation.interpretability import AttributionMethod
    >>> AttributionMethod.INTEGRATED_GRADIENTS.value
    'integrated_gradients'
    >>> from hf_gtc.evaluation.interpretability import create_attribution_config
    >>> config = create_attribution_config()
    >>> config.method
    <AttributionMethod.INTEGRATED_GRADIENTS: 'integrated_gradients'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from hf_gtc._validation import validate_not_none


class AttributionMethod(Enum):
    """Methods for computing feature attributions.

    Attributes:
        INTEGRATED_GRADIENTS: Integrated gradients method.
        SALIENCY: Simple gradient saliency.
        GRAD_CAM: Gradient-weighted Class Activation Mapping.
        ATTENTION_ROLLOUT: Attention rollout across layers.
        LIME: Local Interpretable Model-agnostic Explanations.
        SHAP: SHapley Additive exPlanations.

    Examples:
        >>> AttributionMethod.INTEGRATED_GRADIENTS.value
        'integrated_gradients'
        >>> AttributionMethod.SHAP.value
        'shap'
    """

    INTEGRATED_GRADIENTS = "integrated_gradients"
    SALIENCY = "saliency"
    GRAD_CAM = "grad_cam"
    ATTENTION_ROLLOUT = "attention_rollout"
    LIME = "lime"
    SHAP = "shap"


VALID_ATTRIBUTION_METHODS = frozenset(m.value for m in AttributionMethod)


class VisualizationType(Enum):
    """Types of interpretability visualizations.

    Attributes:
        HEATMAP: 2D heatmap visualization.
        BAR_CHART: Bar chart for feature importance.
        TEXT_HIGHLIGHT: Highlighted text with attribution colors.
        ATTENTION_PATTERN: Attention pattern matrix.

    Examples:
        >>> VisualizationType.HEATMAP.value
        'heatmap'
        >>> VisualizationType.TEXT_HIGHLIGHT.value
        'text_highlight'
    """

    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"
    TEXT_HIGHLIGHT = "text_highlight"
    ATTENTION_PATTERN = "attention_pattern"


VALID_VISUALIZATION_TYPES = frozenset(v.value for v in VisualizationType)


class AggregationMethod(Enum):
    """Methods for aggregating attention or attribution scores.

    Attributes:
        MEAN: Average across dimension.
        MAX: Maximum across dimension.
        L2_NORM: L2 norm across dimension.
        SUM: Sum across dimension.

    Examples:
        >>> AggregationMethod.MEAN.value
        'mean'
        >>> AggregationMethod.L2_NORM.value
        'l2_norm'
    """

    MEAN = "mean"
    MAX = "max"
    L2_NORM = "l2_norm"
    SUM = "sum"


VALID_AGGREGATION_METHODS = frozenset(a.value for a in AggregationMethod)


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Configuration for attention weight extraction.

    Attributes:
        layer_indices: Indices of layers to extract attention from.
            None means all layers.
        head_indices: Indices of attention heads to extract.
            None means all heads.
        aggregation: Method to aggregate across heads.
        normalize: Whether to normalize attention weights.

    Examples:
        >>> config = AttentionConfig(layer_indices=[0, 1, 2])
        >>> config.layer_indices
        [0, 1, 2]
        >>> config.normalize
        True
    """

    layer_indices: list[int] | None = None
    head_indices: list[int] | None = None
    aggregation: AggregationMethod = AggregationMethod.MEAN
    normalize: bool = True


@dataclass(frozen=True, slots=True)
class AttributionConfig:
    """Configuration for attribution computation.

    Attributes:
        method: Attribution method to use.
        baseline_type: Type of baseline for comparison (zero, uniform, blur).
        n_steps: Number of interpolation steps for integrated gradients.
        internal_batch_size: Batch size for internal computation.

    Examples:
        >>> config = AttributionConfig(n_steps=100)
        >>> config.n_steps
        100
        >>> config.method
        <AttributionMethod.INTEGRATED_GRADIENTS: 'integrated_gradients'>
    """

    method: AttributionMethod = AttributionMethod.INTEGRATED_GRADIENTS
    baseline_type: str = "zero"
    n_steps: int = 50
    internal_batch_size: int = 32


@dataclass(frozen=True, slots=True)
class InterpretabilityResult:
    """Result of interpretability analysis.

    Attributes:
        attributions: Feature attribution scores per token/feature.
        attention_weights: Raw attention weights from model (optional).
        tokens: Token strings corresponding to attributions (optional).
        layer_info: Layer-wise attribution breakdown (optional).

    Examples:
        >>> result = InterpretabilityResult(
        ...     attributions=[0.1, 0.5, 0.2, 0.15, 0.05],
        ...     attention_weights=None,
        ...     tokens=["The", "quick", "brown", "fox", "jumps"],
        ...     layer_info=None,
        ... )
        >>> len(result.attributions)
        5
        >>> result.tokens[1]
        'quick'
    """

    attributions: list[float]
    attention_weights: list[list[float]] | None
    tokens: list[str] | None
    layer_info: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Configuration for interpretability visualization.

    Attributes:
        viz_type: Type of visualization to generate.
        colormap: Colormap name for heatmaps (e.g., 'viridis', 'coolwarm').
        figsize: Figure size as (width, height) tuple.
        show_values: Whether to show numerical values on visualization.

    Examples:
        >>> config = VisualizationConfig(colormap="coolwarm")
        >>> config.colormap
        'coolwarm'
        >>> config.viz_type
        <VisualizationType.HEATMAP: 'heatmap'>
    """

    viz_type: VisualizationType = VisualizationType.HEATMAP
    colormap: str = "viridis"
    figsize: tuple[int, int] = (10, 6)
    show_values: bool = False


def create_attention_config(
    layer_indices: list[int] | None = None,
    head_indices: list[int] | None = None,
    aggregation: AggregationMethod = AggregationMethod.MEAN,
    normalize: bool = True,
) -> AttentionConfig:
    """Create an attention extraction configuration.

    Args:
        layer_indices: Indices of layers to extract attention from.
        head_indices: Indices of attention heads to extract.
        aggregation: Method to aggregate across heads.
        normalize: Whether to normalize attention weights.

    Returns:
        Configured AttentionConfig instance.

    Examples:
        >>> config = create_attention_config()
        >>> config.aggregation
        <AggregationMethod.MEAN: 'mean'>

        >>> config = create_attention_config(layer_indices=[0, 1])
        >>> config.layer_indices
        [0, 1]

        >>> config = create_attention_config(aggregation=AggregationMethod.MAX)
        >>> config.aggregation
        <AggregationMethod.MAX: 'max'>
    """
    return AttentionConfig(
        layer_indices=layer_indices,
        head_indices=head_indices,
        aggregation=aggregation,
        normalize=normalize,
    )


def validate_attention_config(config: AttentionConfig) -> None:
    """Validate attention configuration.

    Args:
        config: AttentionConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If layer_indices contains negative values.
        ValueError: If head_indices contains negative values.

    Examples:
        >>> config = create_attention_config()
        >>> validate_attention_config(config)  # No error

        >>> validate_attention_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AttentionConfig(layer_indices=[-1])
        >>> validate_attention_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_indices cannot contain negative values
    """
    validate_not_none(config, "config")

    if config.layer_indices is not None and any(
        idx < 0 for idx in config.layer_indices
    ):
        msg = "layer_indices cannot contain negative values"
        raise ValueError(msg)

    if config.head_indices is not None and any(idx < 0 for idx in config.head_indices):
        msg = "head_indices cannot contain negative values"
        raise ValueError(msg)


def create_attribution_config(
    method: AttributionMethod = AttributionMethod.INTEGRATED_GRADIENTS,
    baseline_type: str = "zero",
    n_steps: int = 50,
    internal_batch_size: int = 32,
) -> AttributionConfig:
    """Create an attribution computation configuration.

    Args:
        method: Attribution method to use.
        baseline_type: Type of baseline for comparison.
        n_steps: Number of interpolation steps.
        internal_batch_size: Batch size for internal computation.

    Returns:
        Configured AttributionConfig instance.

    Examples:
        >>> config = create_attribution_config()
        >>> config.method
        <AttributionMethod.INTEGRATED_GRADIENTS: 'integrated_gradients'>

        >>> config = create_attribution_config(method=AttributionMethod.SHAP)
        >>> config.method
        <AttributionMethod.SHAP: 'shap'>

        >>> config = create_attribution_config(n_steps=100)
        >>> config.n_steps
        100
    """
    return AttributionConfig(
        method=method,
        baseline_type=baseline_type,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )


def validate_attribution_config(config: AttributionConfig) -> None:
    """Validate attribution configuration.

    Args:
        config: AttributionConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If n_steps is not positive.
        ValueError: If internal_batch_size is not positive.
        ValueError: If baseline_type is empty.

    Examples:
        >>> config = create_attribution_config()
        >>> validate_attribution_config(config)  # No error

        >>> validate_attribution_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AttributionConfig(n_steps=0)
        >>> validate_attribution_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: n_steps must be positive
    """
    validate_not_none(config, "config")

    if config.n_steps <= 0:
        msg = f"n_steps must be positive, got {config.n_steps}"
        raise ValueError(msg)

    if config.internal_batch_size <= 0:
        msg = f"internal_batch_size must be positive, got {config.internal_batch_size}"
        raise ValueError(msg)

    if not config.baseline_type:
        msg = "baseline_type cannot be empty"
        raise ValueError(msg)


def create_visualization_config(
    viz_type: VisualizationType = VisualizationType.HEATMAP,
    colormap: str = "viridis",
    figsize: tuple[int, int] = (10, 6),
    show_values: bool = False,
) -> VisualizationConfig:
    """Create a visualization configuration.

    Args:
        viz_type: Type of visualization to generate.
        colormap: Colormap name for heatmaps.
        figsize: Figure size as (width, height) tuple.
        show_values: Whether to show numerical values.

    Returns:
        Configured VisualizationConfig instance.

    Examples:
        >>> config = create_visualization_config()
        >>> config.viz_type
        <VisualizationType.HEATMAP: 'heatmap'>

        >>> config = create_visualization_config(viz_type=VisualizationType.BAR_CHART)
        >>> config.viz_type
        <VisualizationType.BAR_CHART: 'bar_chart'>

        >>> config = create_visualization_config(show_values=True)
        >>> config.show_values
        True
    """
    return VisualizationConfig(
        viz_type=viz_type,
        colormap=colormap,
        figsize=figsize,
        show_values=show_values,
    )


def validate_visualization_config(config: VisualizationConfig) -> None:
    """Validate visualization configuration.

    Args:
        config: VisualizationConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If colormap is empty.
        ValueError: If figsize dimensions are not positive.

    Examples:
        >>> config = create_visualization_config()
        >>> validate_visualization_config(config)  # No error

        >>> validate_visualization_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = VisualizationConfig(figsize=(0, 5))
        >>> validate_visualization_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: figsize dimensions must be positive
    """
    validate_not_none(config, "config")

    if not config.colormap:
        msg = "colormap cannot be empty"
        raise ValueError(msg)

    if config.figsize[0] <= 0 or config.figsize[1] <= 0:
        msg = f"figsize dimensions must be positive, got {config.figsize}"
        raise ValueError(msg)


def list_attribution_methods() -> list[str]:
    """List all available attribution methods.

    Returns:
        Sorted list of attribution method names.

    Examples:
        >>> methods = list_attribution_methods()
        >>> "integrated_gradients" in methods
        True
        >>> "shap" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_ATTRIBUTION_METHODS)


def validate_attribution_method(method: str) -> bool:
    """Validate if a string is a valid attribution method.

    Args:
        method: The method string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_attribution_method("integrated_gradients")
        True
        >>> validate_attribution_method("shap")
        True
        >>> validate_attribution_method("invalid")
        False
        >>> validate_attribution_method("")
        False
    """
    return method in VALID_ATTRIBUTION_METHODS


def get_attribution_method(name: str) -> AttributionMethod:
    """Get AttributionMethod enum from string name.

    Args:
        name: Name of the attribution method.

    Returns:
        Corresponding AttributionMethod enum value.

    Raises:
        ValueError: If name is not a valid attribution method.

    Examples:
        >>> get_attribution_method("integrated_gradients")
        <AttributionMethod.INTEGRATED_GRADIENTS: 'integrated_gradients'>

        >>> get_attribution_method("shap")
        <AttributionMethod.SHAP: 'shap'>

        >>> get_attribution_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid attribution method: invalid
    """
    if not validate_attribution_method(name):
        msg = f"invalid attribution method: {name}"
        raise ValueError(msg)

    return AttributionMethod(name)


def list_visualization_types() -> list[str]:
    """List all available visualization types.

    Returns:
        Sorted list of visualization type names.

    Examples:
        >>> types = list_visualization_types()
        >>> "heatmap" in types
        True
        >>> "text_highlight" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_VISUALIZATION_TYPES)


def validate_visualization_type(viz_type: str) -> bool:
    """Validate if a string is a valid visualization type.

    Args:
        viz_type: The visualization type string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_visualization_type("heatmap")
        True
        >>> validate_visualization_type("bar_chart")
        True
        >>> validate_visualization_type("invalid")
        False
        >>> validate_visualization_type("")
        False
    """
    return viz_type in VALID_VISUALIZATION_TYPES


def get_visualization_type(name: str) -> VisualizationType:
    """Get VisualizationType enum from string name.

    Args:
        name: Name of the visualization type.

    Returns:
        Corresponding VisualizationType enum value.

    Raises:
        ValueError: If name is not a valid visualization type.

    Examples:
        >>> get_visualization_type("heatmap")
        <VisualizationType.HEATMAP: 'heatmap'>

        >>> get_visualization_type("bar_chart")
        <VisualizationType.BAR_CHART: 'bar_chart'>

        >>> get_visualization_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid visualization type: invalid
    """
    if not validate_visualization_type(name):
        msg = f"invalid visualization type: {name}"
        raise ValueError(msg)

    return VisualizationType(name)


def list_aggregation_methods() -> list[str]:
    """List all available aggregation methods.

    Returns:
        Sorted list of aggregation method names.

    Examples:
        >>> methods = list_aggregation_methods()
        >>> "mean" in methods
        True
        >>> "l2_norm" in methods
        True
        >>> methods == sorted(methods)
        True
    """
    return sorted(VALID_AGGREGATION_METHODS)


def validate_aggregation_method(method: str) -> bool:
    """Validate if a string is a valid aggregation method.

    Args:
        method: The method string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_aggregation_method("mean")
        True
        >>> validate_aggregation_method("l2_norm")
        True
        >>> validate_aggregation_method("invalid")
        False
        >>> validate_aggregation_method("")
        False
    """
    return method in VALID_AGGREGATION_METHODS


def get_aggregation_method(name: str) -> AggregationMethod:
    """Get AggregationMethod enum from string name.

    Args:
        name: Name of the aggregation method.

    Returns:
        Corresponding AggregationMethod enum value.

    Raises:
        ValueError: If name is not a valid aggregation method.

    Examples:
        >>> get_aggregation_method("mean")
        <AggregationMethod.MEAN: 'mean'>

        >>> get_aggregation_method("l2_norm")
        <AggregationMethod.L2_NORM: 'l2_norm'>

        >>> get_aggregation_method("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid aggregation method: invalid
    """
    if not validate_aggregation_method(name):
        msg = f"invalid aggregation method: {name}"
        raise ValueError(msg)

    return AggregationMethod(name)


def calculate_attribution_scores(
    gradients: Sequence[float],
    inputs: Sequence[float],
    config: AttributionConfig,
) -> list[float]:
    """Calculate attribution scores from gradients and inputs.

    This function computes feature attributions based on the specified method.
    For integrated gradients, it multiplies gradients by (input - baseline).
    For saliency, it takes the absolute gradient values.

    Args:
        gradients: Gradient values for each feature.
        inputs: Input values for each feature.
        config: Attribution configuration.

    Returns:
        List of attribution scores for each feature.

    Raises:
        ValueError: If gradients is None or empty.
        ValueError: If inputs is None or empty.
        ValueError: If config is None.
        ValueError: If gradients and inputs have different lengths.

    Examples:
        >>> config = create_attribution_config()
        >>> grads = [0.5, -0.3, 0.8, 0.1]
        >>> inputs = [1.0, 2.0, 0.5, 1.5]
        >>> scores = calculate_attribution_scores(grads, inputs, config)
        >>> len(scores)
        4
        >>> all(isinstance(s, float) for s in scores)
        True

        >>> calculate_attribution_scores(
        ...     [], [], config
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradients cannot be empty

        >>> calculate_attribution_scores(
        ...     None, [1.0], config
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: gradients cannot be None
    """
    if gradients is None:
        msg = "gradients cannot be None"
        raise ValueError(msg)

    if len(gradients) == 0:
        msg = "gradients cannot be empty"
        raise ValueError(msg)

    if inputs is None:
        msg = "inputs cannot be None"
        raise ValueError(msg)

    if len(inputs) == 0:
        msg = "inputs cannot be empty"
        raise ValueError(msg)

    validate_not_none(config, "config")

    if len(gradients) != len(inputs):
        msg = (
            f"gradients and inputs must have the same length, "
            f"got {len(gradients)} and {len(inputs)}"
        )
        raise ValueError(msg)

    attributions: list[float] = []

    if config.method == AttributionMethod.SALIENCY:
        # Saliency: absolute value of gradients
        attributions = [abs(g) for g in gradients]
    elif config.method in (
        AttributionMethod.INTEGRATED_GRADIENTS,
        AttributionMethod.GRAD_CAM,
    ):
        # Integrated gradients: gradient * (input - baseline)
        # Assuming zero baseline
        attributions = [g * i for g, i in zip(gradients, inputs, strict=True)]
    else:
        # Default: gradient * input
        attributions = [g * i for g, i in zip(gradients, inputs, strict=True)]

    return attributions


def aggregate_attention_weights(
    attention_weights: Sequence[Sequence[float]],
    config: AttentionConfig,
) -> list[float]:
    """Aggregate attention weights across heads or layers.

    Applies the specified aggregation method to combine attention
    weights from multiple heads or layers into a single set of scores.

    Args:
        attention_weights: 2D array of attention weights (heads x positions).
        config: Attention configuration with aggregation method.

    Returns:
        Aggregated attention weights per position.

    Raises:
        ValueError: If attention_weights is None or empty.
        ValueError: If config is None.

    Examples:
        >>> config = create_attention_config(
        ...     aggregation=AggregationMethod.MEAN, normalize=False
        ... )
        >>> weights = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        >>> agg = aggregate_attention_weights(weights, config)
        >>> len(agg)
        3
        >>> 0.14 < agg[0] < 0.16
        True

        >>> config_max = create_attention_config(
        ...     aggregation=AggregationMethod.MAX, normalize=False
        ... )
        >>> agg_max = aggregate_attention_weights(weights, config_max)
        >>> agg_max[0]
        0.2

        >>> aggregate_attention_weights(
        ...     [], config
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: attention_weights cannot be empty
    """
    if attention_weights is None:
        msg = "attention_weights cannot be None"
        raise ValueError(msg)

    if len(attention_weights) == 0:
        msg = "attention_weights cannot be empty"
        raise ValueError(msg)

    validate_not_none(config, "config")

    num_positions = len(attention_weights[0])
    aggregated: list[float] = []

    for pos in range(num_positions):
        values = [head[pos] for head in attention_weights]

        if config.aggregation == AggregationMethod.MEAN:
            agg_value = sum(values) / len(values)
        elif config.aggregation == AggregationMethod.MAX:
            agg_value = max(values)
        elif config.aggregation == AggregationMethod.SUM:
            agg_value = sum(values)
        elif config.aggregation == AggregationMethod.L2_NORM:
            agg_value = math.sqrt(sum(v * v for v in values))
        else:
            agg_value = sum(values) / len(values)

        aggregated.append(agg_value)

    if config.normalize:
        total = sum(aggregated)
        if total > 0:
            aggregated = [v / total for v in aggregated]

    return aggregated


def estimate_interpretation_time(
    num_tokens: int,
    config: AttributionConfig,
    model_inference_time_ms: float = 50.0,
) -> float:
    """Estimate time required for interpretation analysis.

    Provides a rough estimate of computation time based on the
    attribution method and configuration.

    Args:
        num_tokens: Number of tokens to analyze.
        config: Attribution configuration.
        model_inference_time_ms: Base model inference time in milliseconds.

    Returns:
        Estimated time in seconds.

    Raises:
        ValueError: If num_tokens is not positive.
        ValueError: If config is None.
        ValueError: If model_inference_time_ms is not positive.

    Examples:
        >>> config = create_attribution_config(n_steps=50)
        >>> time_est = estimate_interpretation_time(100, config)
        >>> time_est > 0
        True

        >>> estimate_interpretation_time(0, config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_tokens must be positive

        >>> estimate_interpretation_time(100, None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None
    """
    if num_tokens <= 0:
        msg = f"num_tokens must be positive, got {num_tokens}"
        raise ValueError(msg)

    validate_not_none(config, "config")

    if model_inference_time_ms <= 0:
        msg = f"model_inference_time_ms must be positive, got {model_inference_time_ms}"
        raise ValueError(msg)

    # Base time per token
    base_time_ms = model_inference_time_ms * num_tokens

    # Method-specific multipliers
    method_multipliers = {
        AttributionMethod.SALIENCY: 1.5,
        AttributionMethod.INTEGRATED_GRADIENTS: config.n_steps,
        AttributionMethod.GRAD_CAM: 2.0,
        AttributionMethod.ATTENTION_ROLLOUT: 1.0,
        AttributionMethod.LIME: num_tokens * 5,
        AttributionMethod.SHAP: num_tokens * 10,
    }

    multiplier = method_multipliers.get(config.method, 1.0)
    total_time_ms = base_time_ms * multiplier

    # Account for batch processing
    batches = math.ceil(num_tokens / config.internal_batch_size)
    total_time_ms *= 1 + (batches - 1) * 0.1

    return total_time_ms / 1000.0


def format_interpretation_result(
    result: InterpretabilityResult,
    top_k: int = 5,
) -> str:
    """Format an interpretation result as a human-readable string.

    Args:
        result: Interpretation result to format.
        top_k: Number of top attributions to display.

    Returns:
        Formatted string representation.

    Raises:
        ValueError: If result is None.
        ValueError: If top_k is not positive.

    Examples:
        >>> result = InterpretabilityResult(
        ...     attributions=[0.5, 0.3, 0.1, 0.05, 0.05],
        ...     attention_weights=None,
        ...     tokens=["The", "quick", "brown", "fox", "jumps"],
        ...     layer_info=None,
        ... )
        >>> formatted = format_interpretation_result(result, top_k=3)
        >>> "quick" in formatted or "The" in formatted
        True

        >>> format_interpretation_result(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: result cannot be None
    """
    validate_not_none(result, "result")

    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)

    lines = ["Interpretability Analysis Results", "=" * 35]

    # Show top attributions
    if result.attributions:
        lines.append(f"\nTop {min(top_k, len(result.attributions))} Attributions:")

        # Sort by absolute value
        indexed = list(enumerate(result.attributions))
        sorted_attrs = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)

        for idx, score in sorted_attrs[:top_k]:
            if result.tokens and idx < len(result.tokens):
                token = result.tokens[idx]
            else:
                token = f"[{idx}]"
            lines.append(f"  {token}: {score:.4f}")

    # Summary statistics
    if result.attributions:
        lines.append("\nAttribution Statistics:")
        abs_attrs = [abs(a) for a in result.attributions]
        lines.append(f"  Max: {max(abs_attrs):.4f}")
        lines.append(f"  Mean: {sum(abs_attrs) / len(abs_attrs):.4f}")
        lines.append(f"  Total features: {len(result.attributions)}")

    # Attention info
    if result.attention_weights is not None:
        lines.append(f"\nAttention weights: {len(result.attention_weights)} layers")

    # Layer info
    if result.layer_info is not None:
        lines.append(f"\nLayer info: {len(result.layer_info)} entries")

    return "\n".join(lines)


def get_recommended_interpretability_config(
    task: str,
) -> tuple[AttributionConfig, AttentionConfig]:
    """Get recommended interpretability configuration for a task.

    Provides sensible default configurations based on the task type.

    Args:
        task: Task type (classification, qa, generation, vision).

    Returns:
        Tuple of (AttributionConfig, AttentionConfig).

    Raises:
        ValueError: If task is empty.

    Examples:
        >>> attr_cfg, attn_cfg = get_recommended_interpretability_config(
        ...     "classification"
        ... )
        >>> attr_cfg.method
        <AttributionMethod.INTEGRATED_GRADIENTS: 'integrated_gradients'>

        >>> attr_cfg, attn_cfg = get_recommended_interpretability_config("vision")
        >>> attr_cfg.method
        <AttributionMethod.GRAD_CAM: 'grad_cam'>

        >>> get_recommended_interpretability_config(
        ...     ""
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task cannot be empty
    """
    if not task:
        msg = "task cannot be empty"
        raise ValueError(msg)

    task_lower = task.lower()

    # Task-specific configurations
    if task_lower in ("classification", "sentiment", "text_classification"):
        attr_config = AttributionConfig(
            method=AttributionMethod.INTEGRATED_GRADIENTS,
            baseline_type="zero",
            n_steps=50,
            internal_batch_size=32,
        )
        attn_config = AttentionConfig(
            layer_indices=None,  # All layers
            head_indices=None,
            aggregation=AggregationMethod.MEAN,
            normalize=True,
        )
    elif task_lower in ("qa", "question_answering"):
        attr_config = AttributionConfig(
            method=AttributionMethod.ATTENTION_ROLLOUT,
            baseline_type="zero",
            n_steps=25,
            internal_batch_size=16,
        )
        attn_config = AttentionConfig(
            layer_indices=[-1, -2, -3],  # Last 3 layers
            head_indices=None,
            aggregation=AggregationMethod.MAX,
            normalize=True,
        )
    elif task_lower in ("generation", "text_generation", "lm"):
        attr_config = AttributionConfig(
            method=AttributionMethod.SALIENCY,
            baseline_type="zero",
            n_steps=1,
            internal_batch_size=64,
        )
        attn_config = AttentionConfig(
            layer_indices=None,
            head_indices=None,
            aggregation=AggregationMethod.MEAN,
            normalize=False,
        )
    elif task_lower in ("vision", "image_classification", "image"):
        attr_config = AttributionConfig(
            method=AttributionMethod.GRAD_CAM,
            baseline_type="blur",
            n_steps=100,
            internal_batch_size=8,
        )
        attn_config = AttentionConfig(
            layer_indices=[-1],  # Last layer
            head_indices=None,
            aggregation=AggregationMethod.L2_NORM,
            normalize=True,
        )
    else:
        # Default configuration
        attr_config = AttributionConfig(
            method=AttributionMethod.INTEGRATED_GRADIENTS,
            baseline_type="zero",
            n_steps=50,
            internal_batch_size=32,
        )
        attn_config = AttentionConfig(
            layer_indices=None,
            head_indices=None,
            aggregation=AggregationMethod.MEAN,
            normalize=True,
        )

    return attr_config, attn_config
