"""Model analysis and parameter statistics for HuggingFace models.

This module provides utilities for analyzing model architectures, computing
parameter statistics, layer-wise breakdowns, memory footprints, and generating
analysis reports.

Examples:
    >>> from hf_gtc.models.analysis import (
    ...     AnalysisType, create_parameter_stats, analyze_parameters
    ... )
    >>> stats = create_parameter_stats(
    ...     total_params=110_000_000,
    ...     trainable_params=110_000_000,
    ...     frozen_params=0,
    ... )
    >>> stats.total_params
    110000000

    >>> from hf_gtc.models.analysis import StatisticType, list_statistic_types
    >>> types = list_statistic_types()
    >>> "mean" in types
    True

    >>> from hf_gtc.models.analysis import VisualizationType, get_visualization_type
    >>> viz = get_visualization_type("histogram")
    >>> viz.value
    'histogram'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class AnalysisType(Enum):
    """Types of model analysis.

    Attributes:
        PARAMETER_COUNT: Total parameter count analysis.
        LAYER_WISE: Layer-by-layer parameter breakdown.
        ATTENTION_HEADS: Analysis of attention heads configuration.
        MEMORY_FOOTPRINT: Memory usage estimation.

    Examples:
        >>> AnalysisType.PARAMETER_COUNT.value
        'parameter_count'
        >>> AnalysisType.LAYER_WISE.value
        'layer_wise'
        >>> AnalysisType.ATTENTION_HEADS.value
        'attention_heads'
        >>> AnalysisType.MEMORY_FOOTPRINT.value
        'memory_footprint'
    """

    PARAMETER_COUNT = "parameter_count"
    LAYER_WISE = "layer_wise"
    ATTENTION_HEADS = "attention_heads"
    MEMORY_FOOTPRINT = "memory_footprint"


VALID_ANALYSIS_TYPES = frozenset(t.value for t in AnalysisType)


class StatisticType(Enum):
    """Types of statistics for parameter analysis.

    Attributes:
        MEAN: Mean value of parameters.
        STD: Standard deviation of parameters.
        MIN: Minimum parameter value.
        MAX: Maximum parameter value.
        SPARSITY: Ratio of zero parameters.
        NORM: L2 norm of parameters.

    Examples:
        >>> StatisticType.MEAN.value
        'mean'
        >>> StatisticType.STD.value
        'std'
        >>> StatisticType.MIN.value
        'min'
        >>> StatisticType.MAX.value
        'max'
        >>> StatisticType.SPARSITY.value
        'sparsity'
        >>> StatisticType.NORM.value
        'norm'
    """

    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    SPARSITY = "sparsity"
    NORM = "norm"


VALID_STATISTIC_TYPES = frozenset(t.value for t in StatisticType)


class VisualizationType(Enum):
    """Types of visualizations for model analysis.

    Attributes:
        HISTOGRAM: Parameter distribution histogram.
        HEATMAP: Layer-wise heatmap visualization.
        LINE_PLOT: Line plot for trends.
        BAR_CHART: Bar chart for comparisons.

    Examples:
        >>> VisualizationType.HISTOGRAM.value
        'histogram'
        >>> VisualizationType.HEATMAP.value
        'heatmap'
        >>> VisualizationType.LINE_PLOT.value
        'line_plot'
        >>> VisualizationType.BAR_CHART.value
        'bar_chart'
    """

    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    LINE_PLOT = "line_plot"
    BAR_CHART = "bar_chart"


VALID_VISUALIZATION_TYPES = frozenset(t.value for t in VisualizationType)


@dataclass(frozen=True, slots=True)
class ParameterStats:
    """Statistics about model parameters.

    Attributes:
        total_params: Total number of parameters.
        trainable_params: Number of trainable parameters.
        frozen_params: Number of frozen (non-trainable) parameters.
        dtype_breakdown: Mapping of dtype to parameter count.

    Examples:
        >>> stats = ParameterStats(
        ...     total_params=110_000_000,
        ...     trainable_params=110_000_000,
        ...     frozen_params=0,
        ...     dtype_breakdown={"float32": 110_000_000},
        ... )
        >>> stats.total_params
        110000000
        >>> stats.trainable_params
        110000000
        >>> stats.frozen_params
        0
        >>> stats.dtype_breakdown["float32"]
        110000000
    """

    total_params: int
    trainable_params: int
    frozen_params: int
    dtype_breakdown: dict[str, int]


@dataclass(frozen=True, slots=True)
class LayerAnalysis:
    """Analysis of a single model layer.

    Attributes:
        layer_name: Name or identifier of the layer.
        param_count: Number of parameters in the layer.
        input_shape: Expected input tensor shape.
        output_shape: Expected output tensor shape.
        flops: Floating point operations per forward pass.

    Examples:
        >>> analysis = LayerAnalysis(
        ...     layer_name="attention.self",
        ...     param_count=2359296,
        ...     input_shape=(1, 512, 768),
        ...     output_shape=(1, 512, 768),
        ...     flops=4718592,
        ... )
        >>> analysis.layer_name
        'attention.self'
        >>> analysis.param_count
        2359296
        >>> analysis.input_shape
        (1, 512, 768)
    """

    layer_name: str
    param_count: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    flops: int


@dataclass(frozen=True, slots=True)
class ModelAnalysis:
    """Complete model analysis results.

    Attributes:
        parameter_stats: Overall parameter statistics.
        layer_analyses: Per-layer analysis results.
        memory_estimate_mb: Estimated memory usage in megabytes.
        compute_flops: Total compute FLOPs for one forward pass.

    Examples:
        >>> param_stats = ParameterStats(
        ...     total_params=110_000_000,
        ...     trainable_params=110_000_000,
        ...     frozen_params=0,
        ...     dtype_breakdown={"float32": 110_000_000},
        ... )
        >>> analysis = ModelAnalysis(
        ...     parameter_stats=param_stats,
        ...     layer_analyses=[],
        ...     memory_estimate_mb=420.0,
        ...     compute_flops=22_000_000_000,
        ... )
        >>> analysis.memory_estimate_mb
        420.0
        >>> analysis.compute_flops
        22000000000
    """

    parameter_stats: ParameterStats
    layer_analyses: tuple[LayerAnalysis, ...]
    memory_estimate_mb: float
    compute_flops: int


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration for model analysis.

    Attributes:
        analysis_types: Types of analysis to perform.
        include_gradients: Whether to include gradient memory estimation.
        compute_flops: Whether to compute FLOPs.
        per_layer: Whether to perform per-layer analysis.

    Examples:
        >>> config = AnalysisConfig(
        ...     analysis_types=("parameter_count", "memory_footprint"),
        ...     include_gradients=True,
        ...     compute_flops=True,
        ...     per_layer=True,
        ... )
        >>> config.include_gradients
        True
        >>> "parameter_count" in config.analysis_types
        True
    """

    analysis_types: tuple[str, ...]
    include_gradients: bool
    compute_flops: bool
    per_layer: bool


def validate_parameter_stats(stats: ParameterStats) -> None:
    """Validate parameter statistics.

    Args:
        stats: Statistics to validate.

    Raises:
        ValueError: If stats is None.
        ValueError: If total_params is negative.
        ValueError: If trainable_params is negative.
        ValueError: If frozen_params is negative.
        ValueError: If trainable + frozen != total.
        ValueError: If dtype_breakdown is None.

    Examples:
        >>> stats = ParameterStats(
        ...     total_params=110_000_000,
        ...     trainable_params=110_000_000,
        ...     frozen_params=0,
        ...     dtype_breakdown={"float32": 110_000_000},
        ... )
        >>> validate_parameter_stats(stats)  # No error

        >>> validate_parameter_stats(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad_stats = ParameterStats(
        ...     total_params=-1,
        ...     trainable_params=0,
        ...     frozen_params=0,
        ...     dtype_breakdown={},
        ... )
        >>> validate_parameter_stats(bad_stats)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_params must be non-negative
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.total_params < 0:
        msg = f"total_params must be non-negative, got {stats.total_params}"
        raise ValueError(msg)

    if stats.trainable_params < 0:
        msg = f"trainable_params must be non-negative, got {stats.trainable_params}"
        raise ValueError(msg)

    if stats.frozen_params < 0:
        msg = f"frozen_params must be non-negative, got {stats.frozen_params}"
        raise ValueError(msg)

    if stats.trainable_params + stats.frozen_params != stats.total_params:
        msg = (
            f"trainable_params ({stats.trainable_params}) + frozen_params "
            f"({stats.frozen_params}) must equal total_params ({stats.total_params})"
        )
        raise ValueError(msg)

    if stats.dtype_breakdown is None:
        msg = "dtype_breakdown cannot be None"
        raise ValueError(msg)


def validate_layer_analysis(analysis: LayerAnalysis) -> None:
    """Validate layer analysis.

    Args:
        analysis: Analysis to validate.

    Raises:
        ValueError: If analysis is None.
        ValueError: If layer_name is empty.
        ValueError: If param_count is negative.
        ValueError: If flops is negative.
        ValueError: If input_shape is empty.
        ValueError: If output_shape is empty.

    Examples:
        >>> analysis = LayerAnalysis(
        ...     layer_name="attention",
        ...     param_count=2359296,
        ...     input_shape=(1, 512, 768),
        ...     output_shape=(1, 512, 768),
        ...     flops=4718592,
        ... )
        >>> validate_layer_analysis(analysis)  # No error

        >>> validate_layer_analysis(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: analysis cannot be None

        >>> bad = LayerAnalysis(
        ...     layer_name="",
        ...     param_count=0,
        ...     input_shape=(1, 512, 768),
        ...     output_shape=(1, 512, 768),
        ...     flops=0,
        ... )
        >>> validate_layer_analysis(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_name cannot be empty
    """
    if analysis is None:
        msg = "analysis cannot be None"
        raise ValueError(msg)

    if not analysis.layer_name:
        msg = "layer_name cannot be empty"
        raise ValueError(msg)

    if analysis.param_count < 0:
        msg = f"param_count must be non-negative, got {analysis.param_count}"
        raise ValueError(msg)

    if analysis.flops < 0:
        msg = f"flops must be non-negative, got {analysis.flops}"
        raise ValueError(msg)

    if not analysis.input_shape:
        msg = "input_shape cannot be empty"
        raise ValueError(msg)

    if not analysis.output_shape:
        msg = "output_shape cannot be empty"
        raise ValueError(msg)


def validate_model_analysis(analysis: ModelAnalysis) -> None:
    """Validate model analysis.

    Args:
        analysis: Analysis to validate.

    Raises:
        ValueError: If analysis is None.
        ValueError: If parameter_stats is invalid.
        ValueError: If memory_estimate_mb is negative.
        ValueError: If compute_flops is negative.

    Examples:
        >>> param_stats = ParameterStats(
        ...     total_params=110_000_000,
        ...     trainable_params=110_000_000,
        ...     frozen_params=0,
        ...     dtype_breakdown={"float32": 110_000_000},
        ... )
        >>> model_analysis = ModelAnalysis(
        ...     parameter_stats=param_stats,
        ...     layer_analyses=(),
        ...     memory_estimate_mb=420.0,
        ...     compute_flops=22_000_000_000,
        ... )
        >>> validate_model_analysis(model_analysis)  # No error

        >>> validate_model_analysis(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: analysis cannot be None
    """
    if analysis is None:
        msg = "analysis cannot be None"
        raise ValueError(msg)

    validate_parameter_stats(analysis.parameter_stats)

    if analysis.memory_estimate_mb < 0:
        msg = (
            f"memory_estimate_mb must be non-negative, "
            f"got {analysis.memory_estimate_mb}"
        )
        raise ValueError(msg)

    if analysis.compute_flops < 0:
        msg = f"compute_flops must be non-negative, got {analysis.compute_flops}"
        raise ValueError(msg)

    for layer_analysis in analysis.layer_analyses:
        validate_layer_analysis(layer_analysis)


def validate_analysis_config(config: AnalysisConfig) -> None:
    """Validate analysis configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If analysis_types is empty.
        ValueError: If any analysis_type is invalid.

    Examples:
        >>> config = AnalysisConfig(
        ...     analysis_types=("parameter_count",),
        ...     include_gradients=True,
        ...     compute_flops=True,
        ...     per_layer=True,
        ... )
        >>> validate_analysis_config(config)  # No error

        >>> validate_analysis_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = AnalysisConfig(
        ...     analysis_types=(),
        ...     include_gradients=True,
        ...     compute_flops=True,
        ...     per_layer=True,
        ... )
        >>> validate_analysis_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: analysis_types cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.analysis_types:
        msg = "analysis_types cannot be empty"
        raise ValueError(msg)

    for analysis_type in config.analysis_types:
        if analysis_type not in VALID_ANALYSIS_TYPES:
            msg = (
                f"invalid analysis_type '{analysis_type}', "
                f"must be one of {VALID_ANALYSIS_TYPES}"
            )
            raise ValueError(msg)


def create_parameter_stats(
    total_params: int = 0,
    trainable_params: int | None = None,
    frozen_params: int = 0,
    dtype_breakdown: dict[str, int] | None = None,
) -> ParameterStats:
    """Create parameter statistics.

    Args:
        total_params: Total number of parameters. Defaults to 0.
        trainable_params: Number of trainable params. Defaults to total - frozen.
        frozen_params: Number of frozen parameters. Defaults to 0.
        dtype_breakdown: Dtype to parameter count mapping. Defaults to float32.

    Returns:
        Validated ParameterStats instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> stats = create_parameter_stats(
        ...     total_params=110_000_000,
        ...     trainable_params=110_000_000,
        ...     frozen_params=0,
        ... )
        >>> stats.total_params
        110000000
        >>> stats.trainable_params
        110000000

        >>> stats = create_parameter_stats(total_params=1000, frozen_params=100)
        >>> stats.trainable_params
        900

        >>> create_parameter_stats(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     total_params=-1
        ... )
        Traceback (most recent call last):
        ValueError: total_params must be non-negative
    """
    if trainable_params is None:
        trainable_params = total_params - frozen_params

    if dtype_breakdown is None:
        dtype_breakdown = {"float32": total_params}

    stats = ParameterStats(
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        dtype_breakdown=dtype_breakdown,
    )
    validate_parameter_stats(stats)
    return stats


def create_layer_analysis(
    layer_name: str,
    param_count: int = 0,
    input_shape: tuple[int, ...] = (1, 512, 768),
    output_shape: tuple[int, ...] | None = None,
    flops: int = 0,
) -> LayerAnalysis:
    """Create layer analysis.

    Args:
        layer_name: Name or identifier of the layer.
        param_count: Number of parameters. Defaults to 0.
        input_shape: Input tensor shape. Defaults to (1, 512, 768).
        output_shape: Output tensor shape. Defaults to input_shape.
        flops: FLOPs for forward pass. Defaults to 0.

    Returns:
        Validated LayerAnalysis instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> analysis = create_layer_analysis(
        ...     layer_name="attention.self",
        ...     param_count=2359296,
        ...     flops=4718592,
        ... )
        >>> analysis.layer_name
        'attention.self'
        >>> analysis.param_count
        2359296

        >>> analysis = create_layer_analysis("layer_0")
        >>> analysis.output_shape
        (1, 512, 768)

        >>> create_layer_analysis("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: layer_name cannot be empty
    """
    if output_shape is None:
        output_shape = input_shape

    analysis = LayerAnalysis(
        layer_name=layer_name,
        param_count=param_count,
        input_shape=input_shape,
        output_shape=output_shape,
        flops=flops,
    )
    validate_layer_analysis(analysis)
    return analysis


def create_model_analysis(
    parameter_stats: ParameterStats | None = None,
    layer_analyses: tuple[LayerAnalysis, ...] | list[LayerAnalysis] | None = None,
    memory_estimate_mb: float = 0.0,
    compute_flops: int = 0,
) -> ModelAnalysis:
    """Create model analysis.

    Args:
        parameter_stats: Parameter statistics. Defaults to empty stats.
        layer_analyses: Per-layer analyses. Defaults to empty tuple.
        memory_estimate_mb: Memory estimate in MB. Defaults to 0.0.
        compute_flops: Total FLOPs. Defaults to 0.

    Returns:
        Validated ModelAnalysis instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> param_stats = create_parameter_stats(total_params=110_000_000)
        >>> analysis = create_model_analysis(
        ...     parameter_stats=param_stats,
        ...     memory_estimate_mb=420.0,
        ...     compute_flops=22_000_000_000,
        ... )
        >>> analysis.memory_estimate_mb
        420.0

        >>> analysis = create_model_analysis()
        >>> analysis.parameter_stats.total_params
        0

        >>> create_model_analysis(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     memory_estimate_mb=-1.0
        ... )
        Traceback (most recent call last):
        ValueError: memory_estimate_mb must be non-negative
    """
    if parameter_stats is None:
        parameter_stats = create_parameter_stats()

    if layer_analyses is None:
        layer_analyses = ()
    elif isinstance(layer_analyses, list):
        layer_analyses = tuple(layer_analyses)

    analysis = ModelAnalysis(
        parameter_stats=parameter_stats,
        layer_analyses=layer_analyses,
        memory_estimate_mb=memory_estimate_mb,
        compute_flops=compute_flops,
    )
    validate_model_analysis(analysis)
    return analysis


def create_analysis_config(
    analysis_types: tuple[str, ...] | list[str] | None = None,
    include_gradients: bool = True,
    compute_flops: bool = True,
    per_layer: bool = True,
) -> AnalysisConfig:
    """Create analysis configuration.

    Args:
        analysis_types: Types of analysis to perform. Defaults to all types.
        include_gradients: Include gradient memory. Defaults to True.
        compute_flops: Compute FLOPs. Defaults to True.
        per_layer: Perform per-layer analysis. Defaults to True.

    Returns:
        Validated AnalysisConfig instance.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_analysis_config()
        >>> len(config.analysis_types) == 4
        True
        >>> config.include_gradients
        True

        >>> config = create_analysis_config(
        ...     analysis_types=("parameter_count",),
        ...     include_gradients=False,
        ... )
        >>> config.analysis_types
        ('parameter_count',)

        >>> create_analysis_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     analysis_types=("invalid",)
        ... )
        Traceback (most recent call last):
        ValueError: invalid analysis_type 'invalid'
    """
    if analysis_types is None:
        analysis_types = tuple(VALID_ANALYSIS_TYPES)
    elif isinstance(analysis_types, list):
        analysis_types = tuple(analysis_types)

    config = AnalysisConfig(
        analysis_types=analysis_types,
        include_gradients=include_gradients,
        compute_flops=compute_flops,
        per_layer=per_layer,
    )
    validate_analysis_config(config)
    return config


def list_analysis_types() -> list[str]:
    """List available analysis types.

    Returns:
        Sorted list of analysis type names.

    Examples:
        >>> types = list_analysis_types()
        >>> "parameter_count" in types
        True
        >>> "layer_wise" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ANALYSIS_TYPES)


def list_statistic_types() -> list[str]:
    """List available statistic types.

    Returns:
        Sorted list of statistic type names.

    Examples:
        >>> types = list_statistic_types()
        >>> "mean" in types
        True
        >>> "std" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_STATISTIC_TYPES)


def list_visualization_types() -> list[str]:
    """List available visualization types.

    Returns:
        Sorted list of visualization type names.

    Examples:
        >>> types = list_visualization_types()
        >>> "histogram" in types
        True
        >>> "heatmap" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_VISUALIZATION_TYPES)


def get_analysis_type(name: str) -> AnalysisType:
    """Get analysis type enum from string.

    Args:
        name: Type name.

    Returns:
        AnalysisType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_analysis_type("parameter_count")
        <AnalysisType.PARAMETER_COUNT: 'parameter_count'>
        >>> get_analysis_type("layer_wise")
        <AnalysisType.LAYER_WISE: 'layer_wise'>

        >>> get_analysis_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: analysis type must be one of...
    """
    for t in AnalysisType:
        if t.value == name:
            return t
    msg = f"analysis type must be one of {VALID_ANALYSIS_TYPES}, got {name}"
    raise ValueError(msg)


def get_statistic_type(name: str) -> StatisticType:
    """Get statistic type enum from string.

    Args:
        name: Type name.

    Returns:
        StatisticType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_statistic_type("mean")
        <StatisticType.MEAN: 'mean'>
        >>> get_statistic_type("std")
        <StatisticType.STD: 'std'>

        >>> get_statistic_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: statistic type must be one of...
    """
    for t in StatisticType:
        if t.value == name:
            return t
    msg = f"statistic type must be one of {VALID_STATISTIC_TYPES}, got {name}"
    raise ValueError(msg)


def get_visualization_type(name: str) -> VisualizationType:
    """Get visualization type enum from string.

    Args:
        name: Type name.

    Returns:
        VisualizationType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_visualization_type("histogram")
        <VisualizationType.HISTOGRAM: 'histogram'>
        >>> get_visualization_type("heatmap")
        <VisualizationType.HEATMAP: 'heatmap'>

        >>> get_visualization_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: visualization type must be one of...
    """
    for t in VisualizationType:
        if t.value == name:
            return t
    msg = f"visualization type must be one of {VALID_VISUALIZATION_TYPES}, got {name}"
    raise ValueError(msg)


def analyze_parameters(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int | None = None,
    vocab_size: int = 30522,
    include_embeddings: bool = True,
) -> ParameterStats:
    """Analyze model parameters based on architecture.

    Computes the parameter count breakdown for a transformer model
    based on its architecture configuration.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension. Defaults to 4 * hidden_size.
        vocab_size: Vocabulary size. Defaults to 30522.
        include_embeddings: Include embedding parameters. Defaults to True.

    Returns:
        ParameterStats with computed parameter counts.

    Raises:
        ValueError: If num_layers is not positive.
        ValueError: If hidden_size is not positive.
        ValueError: If num_heads is not positive.
        ValueError: If hidden_size not divisible by num_heads.

    Examples:
        >>> stats = analyze_parameters(
        ...     num_layers=12, hidden_size=768, num_heads=12
        ... )
        >>> stats.total_params > 100_000_000
        True

        >>> stats = analyze_parameters(
        ...     num_layers=6, hidden_size=512, num_heads=8,
        ...     vocab_size=50000
        ... )
        >>> stats.trainable_params > 0
        True

        >>> analyze_parameters(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     num_layers=0, hidden_size=768, num_heads=12
        ... )
        Traceback (most recent call last):
        ValueError: num_layers must be positive
    """
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    if hidden_size % num_heads != 0:
        msg = (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
        raise ValueError(msg)

    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    h = hidden_size
    n_layers = num_layers
    v = vocab_size
    ff = intermediate_size

    # Per-layer attention: Q, K, V projections + output projection
    attention_params_per_layer = 4 * h * h

    # Per-layer FFN: up-projection + down-projection
    ffn_params_per_layer = 2 * h * ff

    # Layer norms: 2 per layer, each with gamma and beta
    norm_params_per_layer = 4 * h

    # Total per layer
    params_per_layer = (
        attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    )

    # All layers
    total_layer_params = n_layers * params_per_layer

    # Final layer norm
    final_norm_params = 2 * h

    # Embedding parameters
    embedding_params = v * h if include_embeddings else 0

    total_params = embedding_params + total_layer_params + final_norm_params

    dtype_breakdown = {
        "embeddings": embedding_params,
        "attention": n_layers * attention_params_per_layer,
        "ffn": n_layers * ffn_params_per_layer,
        "norm": n_layers * norm_params_per_layer + final_norm_params,
    }

    return create_parameter_stats(
        total_params=total_params,
        trainable_params=total_params,
        frozen_params=0,
        dtype_breakdown=dtype_breakdown,
    )


def compute_layer_statistics(
    layer_name: str,
    hidden_size: int,
    intermediate_size: int | None = None,
    layer_type: str = "attention",
    batch_size: int = 1,
    seq_length: int = 512,
) -> LayerAnalysis:
    """Compute statistics for a single layer.

    Args:
        layer_name: Name or identifier of the layer.
        hidden_size: Hidden dimension size.
        intermediate_size: FFN intermediate dimension. Defaults to 4 * hidden_size.
        layer_type: Type of layer ("attention", "ffn", "norm"). Defaults to "attention".
        batch_size: Batch size for shape calculation. Defaults to 1.
        seq_length: Sequence length for shape calculation. Defaults to 512.

    Returns:
        LayerAnalysis with computed statistics.

    Raises:
        ValueError: If layer_name is empty.
        ValueError: If hidden_size is not positive.
        ValueError: If layer_type is not valid.

    Examples:
        >>> analysis = compute_layer_statistics(
        ...     layer_name="layer_0.attention",
        ...     hidden_size=768,
        ...     layer_type="attention",
        ... )
        >>> analysis.param_count > 0
        True

        >>> analysis = compute_layer_statistics(
        ...     layer_name="layer_0.ffn",
        ...     hidden_size=768,
        ...     layer_type="ffn",
        ... )
        >>> analysis.flops > 0
        True

        >>> compute_layer_statistics(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     layer_name="",
        ...     hidden_size=768,
        ... )
        Traceback (most recent call last):
        ValueError: layer_name cannot be empty
    """
    if not layer_name:
        msg = "layer_name cannot be empty"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    valid_layer_types = {"attention", "ffn", "norm"}
    if layer_type not in valid_layer_types:
        msg = f"layer_type must be one of {valid_layer_types}, got '{layer_type}'"
        raise ValueError(msg)

    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    h = hidden_size
    tokens = batch_size * seq_length

    if layer_type == "attention":
        # Q, K, V, O projections
        param_count = 4 * h * h
        # 2 * for multiply-add per projection
        flops = 4 * (2 * tokens * h * h)

    elif layer_type == "ffn":
        # Up and down projections
        param_count = 2 * h * intermediate_size
        # 2 * for multiply-add
        flops = 2 * (2 * tokens * h * intermediate_size)

    else:  # norm
        # Gamma and beta
        param_count = 2 * h
        # Normalize operations
        flops = 5 * tokens * h

    input_shape = (batch_size, seq_length, hidden_size)
    output_shape = input_shape

    return create_layer_analysis(
        layer_name=layer_name,
        param_count=param_count,
        input_shape=input_shape,
        output_shape=output_shape,
        flops=flops,
    )


def estimate_model_flops(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int | None = None,
    vocab_size: int = 30522,
    batch_size: int = 1,
    seq_length: int = 512,
) -> int:
    """Estimate total FLOPs for a model forward pass.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension. Defaults to 4 * hidden_size.
        vocab_size: Vocabulary size. Defaults to 30522.
        batch_size: Batch size. Defaults to 1.
        seq_length: Sequence length. Defaults to 512.

    Returns:
        Estimated total FLOPs.

    Raises:
        ValueError: If num_layers is not positive.
        ValueError: If hidden_size is not positive.
        ValueError: If batch_size is not positive.
        ValueError: If seq_length is not positive.

    Examples:
        >>> flops = estimate_model_flops(
        ...     num_layers=12, hidden_size=768, num_heads=12
        ... )
        >>> flops > 0
        True
        >>> isinstance(flops, int)
        True

        >>> flops_large = estimate_model_flops(
        ...     num_layers=24, hidden_size=1024, num_heads=16
        ... )
        >>> flops_large > flops
        True

        >>> estimate_model_flops(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     num_layers=0, hidden_size=768, num_heads=12
        ... )
        Traceback (most recent call last):
        ValueError: num_layers must be positive
    """
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    if seq_length <= 0:
        msg = f"seq_length must be positive, got {seq_length}"
        raise ValueError(msg)

    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    h = hidden_size
    tokens = batch_size * seq_length
    head_dim = hidden_size // num_heads

    # Embedding lookup (negligible compared to matmuls)
    embedding_flops = 0

    # Per-layer FLOPs
    # Attention: Q, K, V projections
    qkv_flops = 3 * (2 * tokens * h * h)
    # Attention scores: Q @ K^T
    attn_score_flops = 2 * num_heads * tokens * seq_length * head_dim
    # Attention output: scores @ V
    attn_out_flops = 2 * num_heads * tokens * seq_length * head_dim
    # Output projection
    out_proj_flops = 2 * tokens * h * h

    attention_flops = qkv_flops + attn_score_flops + attn_out_flops + out_proj_flops

    # FFN: up and down projections
    ffn_flops = 2 * (2 * tokens * h * intermediate_size)

    # Layer norms (negligible compared to matmuls)
    norm_flops = 2 * (5 * tokens * h)

    layer_flops = attention_flops + ffn_flops + norm_flops
    total_layer_flops = num_layers * layer_flops

    # Final layer norm
    final_norm_flops = 5 * tokens * h

    # Output projection (if separate from embeddings)
    output_flops = 2 * tokens * h * vocab_size

    total_flops = embedding_flops + total_layer_flops + final_norm_flops + output_flops

    return int(total_flops)


def compare_model_architectures(
    configs: list[tuple[str, dict[str, int]]],
) -> dict[str, ModelAnalysis]:
    """Compare multiple model architectures.

    Args:
        configs: List of (name, config_dict) tuples to compare.
            Each config_dict must contain: num_layers, hidden_size, num_heads.
            Optional keys: intermediate_size, vocab_size.

    Returns:
        Dictionary mapping names to ModelAnalysis results.

    Raises:
        ValueError: If configs is empty.
        ValueError: If any config is missing required keys.

    Examples:
        >>> configs = [
        ...     ("small", {"num_layers": 6, "hidden_size": 512, "num_heads": 8}),
        ...     ("large", {"num_layers": 12, "hidden_size": 768, "num_heads": 12}),
        ... ]
        >>> results = compare_model_architectures(configs)
        >>> "small" in results
        True
        >>> "large" in results
        True
        >>> large_params = results["large"].parameter_stats.total_params
        >>> small_params = results["small"].parameter_stats.total_params
        >>> large_params > small_params
        True

        >>> compare_model_architectures([])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: configs cannot be empty
    """
    if not configs:
        msg = "configs cannot be empty"
        raise ValueError(msg)

    required_keys = {"num_layers", "hidden_size", "num_heads"}
    results: dict[str, ModelAnalysis] = {}

    for name, config in configs:
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            msg = f"config for '{name}' missing required keys: {missing_keys}"
            raise ValueError(msg)

        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]
        intermediate_size = config.get("intermediate_size")
        vocab_size = config.get("vocab_size", 30522)

        param_stats = analyze_parameters(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )

        flops = estimate_model_flops(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )

        # Estimate memory (parameters in fp16 + activations)
        dtype_bytes = 2
        param_memory = param_stats.total_params * dtype_bytes
        # Rough activation estimate
        activation_memory = num_layers * 512 * hidden_size * 4 * dtype_bytes
        memory_mb = (param_memory + activation_memory) / (1024 * 1024)

        results[name] = create_model_analysis(
            parameter_stats=param_stats,
            layer_analyses=(),
            memory_estimate_mb=memory_mb,
            compute_flops=flops,
        )

    return results


def format_analysis_report(
    analysis: ModelAnalysis,
    include_layers: bool = False,
) -> str:
    """Format model analysis as a human-readable report.

    Args:
        analysis: Model analysis to format.
        include_layers: Include per-layer breakdown. Defaults to False.

    Returns:
        Formatted report string.

    Raises:
        ValueError: If analysis is None.

    Examples:
        >>> param_stats = create_parameter_stats(total_params=110_000_000)
        >>> analysis = create_model_analysis(
        ...     parameter_stats=param_stats,
        ...     memory_estimate_mb=420.0,
        ...     compute_flops=22_000_000_000,
        ... )
        >>> report = format_analysis_report(analysis)
        >>> "Total Parameters:" in report
        True
        >>> "Memory:" in report
        True

        >>> format_analysis_report(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: analysis cannot be None
    """
    if analysis is None:
        msg = "analysis cannot be None"
        raise ValueError(msg)

    def format_params(count: int) -> str:
        """Format parameter count with appropriate unit."""
        if count >= 1e12:
            return f"{count / 1e12:.2f}T"
        if count >= 1e9:
            return f"{count / 1e9:.2f}B"
        if count >= 1e6:
            return f"{count / 1e6:.2f}M"
        if count >= 1e3:
            return f"{count / 1e3:.2f}K"
        return str(count)

    def format_flops(count: int) -> str:
        """Format FLOPs with appropriate unit."""
        if count >= 1e15:
            return f"{count / 1e15:.2f}P"
        if count >= 1e12:
            return f"{count / 1e12:.2f}T"
        if count >= 1e9:
            return f"{count / 1e9:.2f}G"
        if count >= 1e6:
            return f"{count / 1e6:.2f}M"
        return str(count)

    def format_memory(mb: float) -> str:
        """Format memory size with appropriate unit."""
        if mb >= 1024:
            return f"{mb / 1024:.2f} GB"
        return f"{mb:.2f} MB"

    stats = analysis.parameter_stats

    lines = [
        "Model Analysis Report",
        "=" * 40,
        "",
        "Parameter Statistics:",
        f"  Total Parameters: {format_params(stats.total_params)}",
        f"  Trainable Parameters: {format_params(stats.trainable_params)}",
        f"  Frozen Parameters: {format_params(stats.frozen_params)}",
        "",
        "Compute & Memory:",
        f"  FLOPs (forward): {format_flops(analysis.compute_flops)}",
        f"  Memory: {format_memory(analysis.memory_estimate_mb)}",
    ]

    if stats.dtype_breakdown:
        lines.extend(["", "Parameter Breakdown:"])
        for dtype, count in sorted(stats.dtype_breakdown.items()):
            lines.append(f"  {dtype}: {format_params(count)}")

    if include_layers and analysis.layer_analyses:
        lines.extend(["", "Layer-wise Analysis:"])
        for layer in analysis.layer_analyses:
            lines.append(f"  {layer.layer_name}:")
            lines.append(f"    Parameters: {format_params(layer.param_count)}")
            lines.append(f"    FLOPs: {format_flops(layer.flops)}")
            lines.append(f"    Input: {layer.input_shape}")
            lines.append(f"    Output: {layer.output_shape}")

    return "\n".join(lines)


def get_recommended_analysis_config(
    model_size: str = "base",
    detailed: bool = False,
) -> AnalysisConfig:
    """Get recommended analysis configuration.

    Args:
        model_size: Model size hint ("small", "base", "large"). Defaults to "base".
        detailed: Whether to include detailed analysis. Defaults to False.

    Returns:
        Recommended AnalysisConfig.

    Raises:
        ValueError: If model_size is not recognized.

    Examples:
        >>> config = get_recommended_analysis_config()
        >>> config.include_gradients
        True
        >>> len(config.analysis_types) > 0
        True

        >>> config = get_recommended_analysis_config("large", detailed=True)
        >>> config.per_layer
        True

        >>> get_recommended_analysis_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: model_size must be one of...
    """
    valid_sizes = {"small", "base", "large"}
    model_size = model_size.lower().strip()

    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    if model_size == "small":
        return create_analysis_config(
            analysis_types=("parameter_count", "memory_footprint"),
            include_gradients=True,
            compute_flops=True,
            per_layer=detailed,
        )

    if model_size == "base":
        return create_analysis_config(
            analysis_types=("parameter_count", "layer_wise", "memory_footprint"),
            include_gradients=True,
            compute_flops=True,
            per_layer=detailed,
        )

    # large
    return create_analysis_config(
        analysis_types=tuple(VALID_ANALYSIS_TYPES),
        include_gradients=True,
        compute_flops=True,
        per_layer=True,
    )
