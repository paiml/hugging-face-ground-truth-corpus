"""Tests for model analysis module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.models.analysis import (
    VALID_ANALYSIS_TYPES,
    VALID_STATISTIC_TYPES,
    VALID_VISUALIZATION_TYPES,
    AnalysisConfig,
    AnalysisType,
    LayerAnalysis,
    ModelAnalysis,
    ParameterStats,
    StatisticType,
    VisualizationType,
    analyze_parameters,
    compare_model_architectures,
    compute_layer_statistics,
    create_analysis_config,
    create_layer_analysis,
    create_model_analysis,
    create_parameter_stats,
    estimate_model_flops,
    format_analysis_report,
    get_analysis_type,
    get_recommended_analysis_config,
    get_statistic_type,
    get_visualization_type,
    list_analysis_types,
    list_statistic_types,
    list_visualization_types,
    validate_analysis_config,
    validate_layer_analysis,
    validate_model_analysis,
    validate_parameter_stats,
)


class TestAnalysisType:
    """Tests for AnalysisType enum."""

    def test_parameter_count_value(self) -> None:
        """Test PARAMETER_COUNT enum value."""
        assert AnalysisType.PARAMETER_COUNT.value == "parameter_count"

    def test_layer_wise_value(self) -> None:
        """Test LAYER_WISE enum value."""
        assert AnalysisType.LAYER_WISE.value == "layer_wise"

    def test_attention_heads_value(self) -> None:
        """Test ATTENTION_HEADS enum value."""
        assert AnalysisType.ATTENTION_HEADS.value == "attention_heads"

    def test_memory_footprint_value(self) -> None:
        """Test MEMORY_FOOTPRINT enum value."""
        assert AnalysisType.MEMORY_FOOTPRINT.value == "memory_footprint"

    def test_valid_analysis_types_frozenset(self) -> None:
        """Test VALID_ANALYSIS_TYPES is a frozenset."""
        assert isinstance(VALID_ANALYSIS_TYPES, frozenset)

    def test_valid_analysis_types_contains_all(self) -> None:
        """Test VALID_ANALYSIS_TYPES contains all enum values."""
        for t in AnalysisType:
            assert t.value in VALID_ANALYSIS_TYPES


class TestStatisticType:
    """Tests for StatisticType enum."""

    def test_mean_value(self) -> None:
        """Test MEAN enum value."""
        assert StatisticType.MEAN.value == "mean"

    def test_std_value(self) -> None:
        """Test STD enum value."""
        assert StatisticType.STD.value == "std"

    def test_min_value(self) -> None:
        """Test MIN enum value."""
        assert StatisticType.MIN.value == "min"

    def test_max_value(self) -> None:
        """Test MAX enum value."""
        assert StatisticType.MAX.value == "max"

    def test_sparsity_value(self) -> None:
        """Test SPARSITY enum value."""
        assert StatisticType.SPARSITY.value == "sparsity"

    def test_norm_value(self) -> None:
        """Test NORM enum value."""
        assert StatisticType.NORM.value == "norm"

    def test_valid_statistic_types_frozenset(self) -> None:
        """Test VALID_STATISTIC_TYPES is a frozenset."""
        assert isinstance(VALID_STATISTIC_TYPES, frozenset)

    def test_valid_statistic_types_contains_all(self) -> None:
        """Test VALID_STATISTIC_TYPES contains all enum values."""
        for t in StatisticType:
            assert t.value in VALID_STATISTIC_TYPES


class TestVisualizationType:
    """Tests for VisualizationType enum."""

    def test_histogram_value(self) -> None:
        """Test HISTOGRAM enum value."""
        assert VisualizationType.HISTOGRAM.value == "histogram"

    def test_heatmap_value(self) -> None:
        """Test HEATMAP enum value."""
        assert VisualizationType.HEATMAP.value == "heatmap"

    def test_line_plot_value(self) -> None:
        """Test LINE_PLOT enum value."""
        assert VisualizationType.LINE_PLOT.value == "line_plot"

    def test_bar_chart_value(self) -> None:
        """Test BAR_CHART enum value."""
        assert VisualizationType.BAR_CHART.value == "bar_chart"

    def test_valid_visualization_types_frozenset(self) -> None:
        """Test VALID_VISUALIZATION_TYPES is a frozenset."""
        assert isinstance(VALID_VISUALIZATION_TYPES, frozenset)

    def test_valid_visualization_types_contains_all(self) -> None:
        """Test VALID_VISUALIZATION_TYPES contains all enum values."""
        for t in VisualizationType:
            assert t.value in VALID_VISUALIZATION_TYPES


class TestParameterStats:
    """Tests for ParameterStats dataclass."""

    def test_create_stats(self) -> None:
        """Test basic creation."""
        stats = ParameterStats(
            total_params=110_000_000,
            trainable_params=110_000_000,
            frozen_params=0,
            dtype_breakdown={"float32": 110_000_000},
        )
        assert stats.total_params == 110_000_000
        assert stats.trainable_params == 110_000_000
        assert stats.frozen_params == 0
        assert stats.dtype_breakdown["float32"] == 110_000_000

    def test_with_frozen_params(self) -> None:
        """Test creation with frozen params."""
        stats = ParameterStats(
            total_params=110_000_000,
            trainable_params=100_000_000,
            frozen_params=10_000_000,
            dtype_breakdown={"float32": 110_000_000},
        )
        assert stats.frozen_params == 10_000_000
        assert stats.trainable_params + stats.frozen_params == stats.total_params

    def test_stats_is_frozen(self) -> None:
        """Test stats is immutable."""
        stats = ParameterStats(110_000_000, 110_000_000, 0, {})
        with pytest.raises(AttributeError):
            stats.total_params = 200_000_000  # type: ignore[misc]


class TestLayerAnalysis:
    """Tests for LayerAnalysis dataclass."""

    def test_create_analysis(self) -> None:
        """Test basic creation."""
        analysis = LayerAnalysis(
            layer_name="attention.self",
            param_count=2359296,
            input_shape=(1, 512, 768),
            output_shape=(1, 512, 768),
            flops=4718592,
        )
        assert analysis.layer_name == "attention.self"
        assert analysis.param_count == 2359296
        assert analysis.input_shape == (1, 512, 768)
        assert analysis.output_shape == (1, 512, 768)
        assert analysis.flops == 4718592

    def test_different_shapes(self) -> None:
        """Test with different input/output shapes."""
        analysis = LayerAnalysis(
            layer_name="pooler",
            param_count=590592,
            input_shape=(1, 512, 768),
            output_shape=(1, 768),
            flops=1181184,
        )
        assert analysis.input_shape != analysis.output_shape

    def test_analysis_is_frozen(self) -> None:
        """Test analysis is immutable."""
        analysis = LayerAnalysis("layer", 0, (1,), (1,), 0)
        with pytest.raises(AttributeError):
            analysis.layer_name = "new_name"  # type: ignore[misc]


class TestModelAnalysis:
    """Tests for ModelAnalysis dataclass."""

    def test_create_analysis(self) -> None:
        """Test basic creation."""
        param_stats = ParameterStats(110_000_000, 110_000_000, 0, {})
        analysis = ModelAnalysis(
            parameter_stats=param_stats,
            layer_analyses=(),
            memory_estimate_mb=420.0,
            compute_flops=22_000_000_000,
        )
        assert analysis.parameter_stats == param_stats
        assert analysis.layer_analyses == ()
        assert analysis.memory_estimate_mb == pytest.approx(420.0)
        assert analysis.compute_flops == 22_000_000_000

    def test_with_layer_analyses(self) -> None:
        """Test with layer analyses."""
        param_stats = ParameterStats(110_000_000, 110_000_000, 0, {})
        layer = LayerAnalysis("layer_0", 1000, (1, 512, 768), (1, 512, 768), 2000)
        analysis = ModelAnalysis(
            parameter_stats=param_stats,
            layer_analyses=(layer,),
            memory_estimate_mb=420.0,
            compute_flops=22_000_000_000,
        )
        assert len(analysis.layer_analyses) == 1
        assert analysis.layer_analyses[0].layer_name == "layer_0"

    def test_analysis_is_frozen(self) -> None:
        """Test analysis is immutable."""
        param_stats = ParameterStats(110_000_000, 110_000_000, 0, {})
        analysis = ModelAnalysis(param_stats, (), 420.0, 22_000_000_000)
        with pytest.raises(AttributeError):
            analysis.memory_estimate_mb = 500.0  # type: ignore[misc]


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_create_config(self) -> None:
        """Test basic creation."""
        config = AnalysisConfig(
            analysis_types=("parameter_count", "memory_footprint"),
            include_gradients=True,
            compute_flops=True,
            per_layer=True,
        )
        assert "parameter_count" in config.analysis_types
        assert config.include_gradients is True
        assert config.compute_flops is True
        assert config.per_layer is True

    def test_config_is_frozen(self) -> None:
        """Test config is immutable."""
        config = AnalysisConfig(("parameter_count",), True, True, True)
        with pytest.raises(AttributeError):
            config.include_gradients = False  # type: ignore[misc]


class TestValidateParameterStats:
    """Tests for validate_parameter_stats function."""

    def test_valid_stats(self) -> None:
        """Test valid stats pass validation."""
        stats = ParameterStats(110_000_000, 110_000_000, 0, {"float32": 110_000_000})
        validate_parameter_stats(stats)

    def test_none_stats_raises(self) -> None:
        """Test None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            validate_parameter_stats(None)  # type: ignore[arg-type]

    def test_negative_total_params_raises(self) -> None:
        """Test negative total_params raises ValueError."""
        stats = ParameterStats(-1, 0, 0, {})
        with pytest.raises(ValueError, match="total_params must be non-negative"):
            validate_parameter_stats(stats)

    def test_negative_trainable_params_raises(self) -> None:
        """Test negative trainable_params raises ValueError."""
        stats = ParameterStats(100, -1, 0, {})
        with pytest.raises(ValueError, match="trainable_params must be non-negative"):
            validate_parameter_stats(stats)

    def test_negative_frozen_params_raises(self) -> None:
        """Test negative frozen_params raises ValueError."""
        stats = ParameterStats(100, 100, -1, {})
        with pytest.raises(ValueError, match="frozen_params must be non-negative"):
            validate_parameter_stats(stats)

    def test_params_mismatch_raises(self) -> None:
        """Test trainable + frozen != total raises ValueError."""
        stats = ParameterStats(100, 50, 40, {})  # 50 + 40 != 100
        with pytest.raises(ValueError, match="must equal total_params"):
            validate_parameter_stats(stats)

    def test_zero_values_valid(self) -> None:
        """Test zero values are valid."""
        stats = ParameterStats(0, 0, 0, {})
        validate_parameter_stats(stats)


class TestValidateLayerAnalysis:
    """Tests for validate_layer_analysis function."""

    def test_valid_analysis(self) -> None:
        """Test valid analysis passes validation."""
        analysis = LayerAnalysis("attention", 2359296, (1, 512, 768), (1, 512, 768), 0)
        validate_layer_analysis(analysis)

    def test_none_analysis_raises(self) -> None:
        """Test None analysis raises ValueError."""
        with pytest.raises(ValueError, match="analysis cannot be None"):
            validate_layer_analysis(None)  # type: ignore[arg-type]

    def test_empty_layer_name_raises(self) -> None:
        """Test empty layer_name raises ValueError."""
        analysis = LayerAnalysis("", 0, (1,), (1,), 0)
        with pytest.raises(ValueError, match="layer_name cannot be empty"):
            validate_layer_analysis(analysis)

    def test_negative_param_count_raises(self) -> None:
        """Test negative param_count raises ValueError."""
        analysis = LayerAnalysis("layer", -1, (1,), (1,), 0)
        with pytest.raises(ValueError, match="param_count must be non-negative"):
            validate_layer_analysis(analysis)

    def test_negative_flops_raises(self) -> None:
        """Test negative flops raises ValueError."""
        analysis = LayerAnalysis("layer", 0, (1,), (1,), -1)
        with pytest.raises(ValueError, match="flops must be non-negative"):
            validate_layer_analysis(analysis)

    def test_empty_input_shape_raises(self) -> None:
        """Test empty input_shape raises ValueError."""
        analysis = LayerAnalysis("layer", 0, (), (1,), 0)
        with pytest.raises(ValueError, match="input_shape cannot be empty"):
            validate_layer_analysis(analysis)

    def test_empty_output_shape_raises(self) -> None:
        """Test empty output_shape raises ValueError."""
        analysis = LayerAnalysis("layer", 0, (1,), (), 0)
        with pytest.raises(ValueError, match="output_shape cannot be empty"):
            validate_layer_analysis(analysis)


class TestValidateModelAnalysis:
    """Tests for validate_model_analysis function."""

    def test_valid_analysis(self) -> None:
        """Test valid analysis passes validation."""
        param_stats = ParameterStats(110_000_000, 110_000_000, 0, {})
        analysis = ModelAnalysis(param_stats, (), 420.0, 22_000_000_000)
        validate_model_analysis(analysis)

    def test_none_analysis_raises(self) -> None:
        """Test None analysis raises ValueError."""
        with pytest.raises(ValueError, match="analysis cannot be None"):
            validate_model_analysis(None)  # type: ignore[arg-type]

    def test_negative_memory_raises(self) -> None:
        """Test negative memory_estimate_mb raises ValueError."""
        param_stats = ParameterStats(0, 0, 0, {})
        analysis = ModelAnalysis(param_stats, (), -1.0, 0)
        with pytest.raises(ValueError, match="memory_estimate_mb must be non-negative"):
            validate_model_analysis(analysis)

    def test_negative_flops_raises(self) -> None:
        """Test negative compute_flops raises ValueError."""
        param_stats = ParameterStats(0, 0, 0, {})
        analysis = ModelAnalysis(param_stats, (), 0.0, -1)
        with pytest.raises(ValueError, match="compute_flops must be non-negative"):
            validate_model_analysis(analysis)

    def test_invalid_layer_analysis_raises(self) -> None:
        """Test invalid layer analysis raises ValueError."""
        param_stats = ParameterStats(0, 0, 0, {})
        bad_layer = LayerAnalysis("", 0, (1,), (1,), 0)  # Empty name
        analysis = ModelAnalysis(param_stats, (bad_layer,), 0.0, 0)
        with pytest.raises(ValueError, match="layer_name cannot be empty"):
            validate_model_analysis(analysis)


class TestValidateAnalysisConfig:
    """Tests for validate_analysis_config function."""

    def test_valid_config(self) -> None:
        """Test valid config passes validation."""
        config = AnalysisConfig(("parameter_count",), True, True, True)
        validate_analysis_config(config)

    def test_none_config_raises(self) -> None:
        """Test None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_analysis_config(None)  # type: ignore[arg-type]

    def test_empty_analysis_types_raises(self) -> None:
        """Test empty analysis_types raises ValueError."""
        config = AnalysisConfig((), True, True, True)
        with pytest.raises(ValueError, match="analysis_types cannot be empty"):
            validate_analysis_config(config)

    def test_invalid_analysis_type_raises(self) -> None:
        """Test invalid analysis_type raises ValueError."""
        config = AnalysisConfig(("invalid",), True, True, True)
        with pytest.raises(ValueError, match="invalid analysis_type"):
            validate_analysis_config(config)


class TestCreateParameterStats:
    """Tests for create_parameter_stats function."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = create_parameter_stats()
        assert stats.total_params == 0
        assert stats.trainable_params == 0
        assert stats.frozen_params == 0

    def test_custom_total_params(self) -> None:
        """Test custom total_params."""
        stats = create_parameter_stats(total_params=110_000_000)
        assert stats.total_params == 110_000_000
        assert stats.trainable_params == 110_000_000
        assert stats.frozen_params == 0

    def test_with_frozen_params(self) -> None:
        """Test with frozen_params."""
        stats = create_parameter_stats(total_params=1000, frozen_params=100)
        assert stats.trainable_params == 900

    def test_explicit_trainable_params(self) -> None:
        """Test explicit trainable_params."""
        stats = create_parameter_stats(
            total_params=1000, trainable_params=800, frozen_params=200
        )
        assert stats.trainable_params == 800

    def test_custom_dtype_breakdown(self) -> None:
        """Test custom dtype_breakdown."""
        dtype = {"float16": 50_000_000, "float32": 60_000_000}
        stats = create_parameter_stats(total_params=110_000_000, dtype_breakdown=dtype)
        assert stats.dtype_breakdown == dtype

    def test_default_dtype_breakdown(self) -> None:
        """Test default dtype_breakdown."""
        stats = create_parameter_stats(total_params=1000)
        assert stats.dtype_breakdown == {"float32": 1000}

    def test_invalid_params_raises(self) -> None:
        """Test invalid params raises ValueError."""
        with pytest.raises(ValueError):
            create_parameter_stats(total_params=-1)


class TestCreateLayerAnalysis:
    """Tests for create_layer_analysis function."""

    def test_default_values(self) -> None:
        """Test default values."""
        analysis = create_layer_analysis("layer_0")
        assert analysis.layer_name == "layer_0"
        assert analysis.param_count == 0
        assert analysis.input_shape == (1, 512, 768)
        assert analysis.output_shape == (1, 512, 768)
        assert analysis.flops == 0

    def test_custom_values(self) -> None:
        """Test custom values."""
        analysis = create_layer_analysis(
            layer_name="attention",
            param_count=2359296,
            input_shape=(2, 1024, 1024),
            output_shape=(2, 1024, 1024),
            flops=4718592,
        )
        assert analysis.layer_name == "attention"
        assert analysis.param_count == 2359296
        assert analysis.input_shape == (2, 1024, 1024)

    def test_default_output_shape(self) -> None:
        """Test default output_shape equals input_shape."""
        analysis = create_layer_analysis("layer", input_shape=(4, 256, 512))
        assert analysis.output_shape == (4, 256, 512)

    def test_empty_name_raises(self) -> None:
        """Test empty layer_name raises ValueError."""
        with pytest.raises(ValueError, match="layer_name cannot be empty"):
            create_layer_analysis("")


class TestCreateModelAnalysis:
    """Tests for create_model_analysis function."""

    def test_default_values(self) -> None:
        """Test default values."""
        analysis = create_model_analysis()
        assert analysis.parameter_stats.total_params == 0
        assert analysis.layer_analyses == ()
        assert analysis.memory_estimate_mb == pytest.approx(0.0)
        assert analysis.compute_flops == 0

    def test_custom_parameter_stats(self) -> None:
        """Test custom parameter_stats."""
        param_stats = create_parameter_stats(total_params=110_000_000)
        analysis = create_model_analysis(parameter_stats=param_stats)
        assert analysis.parameter_stats.total_params == 110_000_000

    def test_with_layer_analyses_list(self) -> None:
        """Test with layer_analyses as list."""
        layer = create_layer_analysis("layer_0", param_count=1000)
        analysis = create_model_analysis(layer_analyses=[layer])
        assert len(analysis.layer_analyses) == 1
        assert analysis.layer_analyses[0].layer_name == "layer_0"

    def test_with_layer_analyses_tuple(self) -> None:
        """Test with layer_analyses as tuple."""
        layer = create_layer_analysis("layer_0", param_count=1000)
        analysis = create_model_analysis(layer_analyses=(layer,))
        assert len(analysis.layer_analyses) == 1

    def test_custom_memory_and_flops(self) -> None:
        """Test custom memory_estimate_mb and compute_flops."""
        analysis = create_model_analysis(
            memory_estimate_mb=420.0, compute_flops=22_000_000_000
        )
        assert analysis.memory_estimate_mb == pytest.approx(420.0)
        assert analysis.compute_flops == 22_000_000_000

    def test_negative_memory_raises(self) -> None:
        """Test negative memory_estimate_mb raises ValueError."""
        with pytest.raises(ValueError, match="memory_estimate_mb must be non-negative"):
            create_model_analysis(memory_estimate_mb=-1.0)


class TestCreateAnalysisConfig:
    """Tests for create_analysis_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_analysis_config()
        assert len(config.analysis_types) == 4
        assert config.include_gradients is True
        assert config.compute_flops is True
        assert config.per_layer is True

    def test_custom_analysis_types_list(self) -> None:
        """Test custom analysis_types as list."""
        config = create_analysis_config(
            analysis_types=["parameter_count", "memory_footprint"]
        )
        assert config.analysis_types == ("parameter_count", "memory_footprint")

    def test_custom_analysis_types_tuple(self) -> None:
        """Test custom analysis_types as tuple."""
        config = create_analysis_config(analysis_types=("parameter_count",))
        assert config.analysis_types == ("parameter_count",)

    def test_custom_flags(self) -> None:
        """Test custom flags."""
        config = create_analysis_config(
            include_gradients=False, compute_flops=False, per_layer=False
        )
        assert config.include_gradients is False
        assert config.compute_flops is False
        assert config.per_layer is False

    def test_invalid_analysis_type_raises(self) -> None:
        """Test invalid analysis_type raises ValueError."""
        with pytest.raises(ValueError, match="invalid analysis_type"):
            create_analysis_config(analysis_types=("invalid",))


class TestListAnalysisTypes:
    """Tests for list_analysis_types function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        types = list_analysis_types()
        assert types == sorted(types)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_analysis_types()
        assert "parameter_count" in types
        assert "layer_wise" in types
        assert "attention_heads" in types
        assert "memory_footprint" in types

    def test_contains_all_types(self) -> None:
        """Test contains all types."""
        types = list_analysis_types()
        for t in AnalysisType:
            assert t.value in types


class TestListStatisticTypes:
    """Tests for list_statistic_types function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        types = list_statistic_types()
        assert types == sorted(types)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_statistic_types()
        assert "mean" in types
        assert "std" in types
        assert "min" in types
        assert "max" in types

    def test_contains_all_types(self) -> None:
        """Test contains all types."""
        types = list_statistic_types()
        for t in StatisticType:
            assert t.value in types


class TestListVisualizationTypes:
    """Tests for list_visualization_types function."""

    def test_returns_sorted_list(self) -> None:
        """Test returns sorted list."""
        types = list_visualization_types()
        assert types == sorted(types)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_visualization_types()
        assert "histogram" in types
        assert "heatmap" in types
        assert "line_plot" in types
        assert "bar_chart" in types

    def test_contains_all_types(self) -> None:
        """Test contains all types."""
        types = list_visualization_types()
        for t in VisualizationType:
            assert t.value in types


class TestGetAnalysisType:
    """Tests for get_analysis_type function."""

    def test_get_parameter_count(self) -> None:
        """Test get parameter_count type."""
        assert get_analysis_type("parameter_count") == AnalysisType.PARAMETER_COUNT

    def test_get_layer_wise(self) -> None:
        """Test get layer_wise type."""
        assert get_analysis_type("layer_wise") == AnalysisType.LAYER_WISE

    def test_get_attention_heads(self) -> None:
        """Test get attention_heads type."""
        assert get_analysis_type("attention_heads") == AnalysisType.ATTENTION_HEADS

    def test_get_memory_footprint(self) -> None:
        """Test get memory_footprint type."""
        assert get_analysis_type("memory_footprint") == AnalysisType.MEMORY_FOOTPRINT

    def test_invalid_type_raises(self) -> None:
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="analysis type must be one of"):
            get_analysis_type("invalid")


class TestGetStatisticType:
    """Tests for get_statistic_type function."""

    def test_get_mean(self) -> None:
        """Test get mean type."""
        assert get_statistic_type("mean") == StatisticType.MEAN

    def test_get_std(self) -> None:
        """Test get std type."""
        assert get_statistic_type("std") == StatisticType.STD

    def test_get_sparsity(self) -> None:
        """Test get sparsity type."""
        assert get_statistic_type("sparsity") == StatisticType.SPARSITY

    def test_get_norm(self) -> None:
        """Test get norm type."""
        assert get_statistic_type("norm") == StatisticType.NORM

    def test_invalid_type_raises(self) -> None:
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="statistic type must be one of"):
            get_statistic_type("invalid")


class TestGetVisualizationType:
    """Tests for get_visualization_type function."""

    def test_get_histogram(self) -> None:
        """Test get histogram type."""
        assert get_visualization_type("histogram") == VisualizationType.HISTOGRAM

    def test_get_heatmap(self) -> None:
        """Test get heatmap type."""
        assert get_visualization_type("heatmap") == VisualizationType.HEATMAP

    def test_get_line_plot(self) -> None:
        """Test get line_plot type."""
        assert get_visualization_type("line_plot") == VisualizationType.LINE_PLOT

    def test_get_bar_chart(self) -> None:
        """Test get bar_chart type."""
        assert get_visualization_type("bar_chart") == VisualizationType.BAR_CHART

    def test_invalid_type_raises(self) -> None:
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="visualization type must be one of"):
            get_visualization_type("invalid")


class TestAnalyzeParameters:
    """Tests for analyze_parameters function."""

    def test_basic_analysis(self) -> None:
        """Test basic parameter analysis."""
        stats = analyze_parameters(num_layers=12, hidden_size=768, num_heads=12)
        assert stats.total_params > 100_000_000  # BERT-base ~110M
        assert stats.trainable_params == stats.total_params
        assert stats.frozen_params == 0

    def test_larger_model_more_params(self) -> None:
        """Test larger model has more parameters."""
        small = analyze_parameters(num_layers=6, hidden_size=512, num_heads=8)
        large = analyze_parameters(num_layers=12, hidden_size=768, num_heads=12)
        assert large.total_params > small.total_params

    def test_more_layers_more_params(self) -> None:
        """Test more layers means more parameters."""
        shallow = analyze_parameters(num_layers=6, hidden_size=768, num_heads=12)
        deep = analyze_parameters(num_layers=12, hidden_size=768, num_heads=12)
        assert deep.total_params > shallow.total_params

    def test_custom_intermediate_size(self) -> None:
        """Test custom intermediate_size."""
        default = analyze_parameters(num_layers=12, hidden_size=768, num_heads=12)
        custom = analyze_parameters(
            num_layers=12, hidden_size=768, num_heads=12, intermediate_size=4096
        )
        assert custom.total_params > default.total_params

    def test_larger_vocab_more_params(self) -> None:
        """Test larger vocab means more parameters."""
        small_vocab = analyze_parameters(
            num_layers=12, hidden_size=768, num_heads=12, vocab_size=30000
        )
        large_vocab = analyze_parameters(
            num_layers=12, hidden_size=768, num_heads=12, vocab_size=50000
        )
        assert large_vocab.total_params > small_vocab.total_params

    def test_exclude_embeddings(self) -> None:
        """Test excluding embeddings."""
        with_emb = analyze_parameters(
            num_layers=12, hidden_size=768, num_heads=12, include_embeddings=True
        )
        without_emb = analyze_parameters(
            num_layers=12, hidden_size=768, num_heads=12, include_embeddings=False
        )
        assert without_emb.total_params < with_emb.total_params

    def test_dtype_breakdown(self) -> None:
        """Test dtype_breakdown is populated."""
        stats = analyze_parameters(num_layers=12, hidden_size=768, num_heads=12)
        assert "embeddings" in stats.dtype_breakdown
        assert "attention" in stats.dtype_breakdown
        assert "ffn" in stats.dtype_breakdown
        assert "norm" in stats.dtype_breakdown

    def test_zero_num_layers_raises(self) -> None:
        """Test zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            analyze_parameters(num_layers=0, hidden_size=768, num_heads=12)

    def test_negative_num_layers_raises(self) -> None:
        """Test negative num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            analyze_parameters(num_layers=-1, hidden_size=768, num_heads=12)

    def test_zero_hidden_size_raises(self) -> None:
        """Test zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            analyze_parameters(num_layers=12, hidden_size=0, num_heads=12)

    def test_zero_num_heads_raises(self) -> None:
        """Test zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            analyze_parameters(num_layers=12, hidden_size=768, num_heads=0)

    def test_hidden_size_not_divisible_raises(self) -> None:
        """Test hidden_size not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            analyze_parameters(num_layers=12, hidden_size=100, num_heads=12)


class TestComputeLayerStatistics:
    """Tests for compute_layer_statistics function."""

    def test_attention_layer(self) -> None:
        """Test attention layer statistics."""
        analysis = compute_layer_statistics(
            layer_name="layer_0.attention", hidden_size=768, layer_type="attention"
        )
        assert analysis.param_count > 0
        assert analysis.flops > 0
        assert analysis.input_shape == (1, 512, 768)

    def test_ffn_layer(self) -> None:
        """Test FFN layer statistics."""
        analysis = compute_layer_statistics(
            layer_name="layer_0.ffn", hidden_size=768, layer_type="ffn"
        )
        assert analysis.param_count > 0
        assert analysis.flops > 0

    def test_norm_layer(self) -> None:
        """Test norm layer statistics."""
        analysis = compute_layer_statistics(
            layer_name="layer_0.norm", hidden_size=768, layer_type="norm"
        )
        assert analysis.param_count > 0
        assert analysis.flops > 0

    def test_custom_batch_size(self) -> None:
        """Test custom batch_size."""
        analysis = compute_layer_statistics(
            layer_name="layer", hidden_size=768, batch_size=4
        )
        assert analysis.input_shape[0] == 4

    def test_custom_seq_length(self) -> None:
        """Test custom seq_length."""
        analysis = compute_layer_statistics(
            layer_name="layer", hidden_size=768, seq_length=1024
        )
        assert analysis.input_shape[1] == 1024

    def test_custom_intermediate_size(self) -> None:
        """Test custom intermediate_size for FFN."""
        analysis = compute_layer_statistics(
            layer_name="ffn",
            hidden_size=768,
            intermediate_size=4096,
            layer_type="ffn",
        )
        assert analysis.param_count > 0

    def test_empty_name_raises(self) -> None:
        """Test empty layer_name raises ValueError."""
        with pytest.raises(ValueError, match="layer_name cannot be empty"):
            compute_layer_statistics(layer_name="", hidden_size=768)

    def test_zero_hidden_size_raises(self) -> None:
        """Test zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            compute_layer_statistics(layer_name="layer", hidden_size=0)

    def test_invalid_layer_type_raises(self) -> None:
        """Test invalid layer_type raises ValueError."""
        with pytest.raises(ValueError, match="layer_type must be one of"):
            compute_layer_statistics(
                layer_name="layer", hidden_size=768, layer_type="invalid"
            )


class TestEstimateModelFlops:
    """Tests for estimate_model_flops function."""

    def test_basic_estimation(self) -> None:
        """Test basic FLOPs estimation."""
        flops = estimate_model_flops(num_layers=12, hidden_size=768, num_heads=12)
        assert flops > 0
        assert isinstance(flops, int)

    def test_larger_model_more_flops(self) -> None:
        """Test larger model has more FLOPs."""
        small = estimate_model_flops(num_layers=6, hidden_size=512, num_heads=8)
        large = estimate_model_flops(num_layers=12, hidden_size=768, num_heads=12)
        assert large > small

    def test_more_layers_more_flops(self) -> None:
        """Test more layers means more FLOPs."""
        shallow = estimate_model_flops(num_layers=6, hidden_size=768, num_heads=12)
        deep = estimate_model_flops(num_layers=12, hidden_size=768, num_heads=12)
        assert deep > shallow

    def test_larger_batch_more_flops(self) -> None:
        """Test larger batch_size means more FLOPs."""
        small_batch = estimate_model_flops(
            num_layers=12, hidden_size=768, num_heads=12, batch_size=1
        )
        large_batch = estimate_model_flops(
            num_layers=12, hidden_size=768, num_heads=12, batch_size=4
        )
        assert large_batch > small_batch

    def test_longer_seq_more_flops(self) -> None:
        """Test longer seq_length means more FLOPs."""
        short_seq = estimate_model_flops(
            num_layers=12, hidden_size=768, num_heads=12, seq_length=256
        )
        long_seq = estimate_model_flops(
            num_layers=12, hidden_size=768, num_heads=12, seq_length=1024
        )
        assert long_seq > short_seq

    def test_zero_num_layers_raises(self) -> None:
        """Test zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_model_flops(num_layers=0, hidden_size=768, num_heads=12)

    def test_zero_hidden_size_raises(self) -> None:
        """Test zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_model_flops(num_layers=12, hidden_size=0, num_heads=12)

    def test_zero_batch_size_raises(self) -> None:
        """Test zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_model_flops(
                num_layers=12, hidden_size=768, num_heads=12, batch_size=0
            )

    def test_zero_seq_length_raises(self) -> None:
        """Test zero seq_length raises ValueError."""
        with pytest.raises(ValueError, match="seq_length must be positive"):
            estimate_model_flops(
                num_layers=12, hidden_size=768, num_heads=12, seq_length=0
            )


class TestCompareModelArchitectures:
    """Tests for compare_model_architectures function."""

    def test_basic_comparison(self) -> None:
        """Test basic architecture comparison."""
        configs = [
            ("small", {"num_layers": 6, "hidden_size": 512, "num_heads": 8}),
            ("large", {"num_layers": 12, "hidden_size": 768, "num_heads": 12}),
        ]
        results = compare_model_architectures(configs)
        assert "small" in results
        assert "large" in results
        large_params = results["large"].parameter_stats.total_params
        small_params = results["small"].parameter_stats.total_params
        assert large_params > small_params

    def test_single_config(self) -> None:
        """Test single config comparison."""
        configs = [("model", {"num_layers": 12, "hidden_size": 768, "num_heads": 12})]
        results = compare_model_architectures(configs)
        assert "model" in results
        assert results["model"].parameter_stats.total_params > 0

    def test_with_optional_keys(self) -> None:
        """Test with optional keys."""
        configs = [
            (
                "model",
                {
                    "num_layers": 12,
                    "hidden_size": 768,
                    "num_heads": 12,
                    "intermediate_size": 4096,
                    "vocab_size": 50000,
                },
            ),
        ]
        results = compare_model_architectures(configs)
        assert "model" in results

    def test_memory_estimate_positive(self) -> None:
        """Test memory estimate is positive."""
        configs = [("model", {"num_layers": 12, "hidden_size": 768, "num_heads": 12})]
        results = compare_model_architectures(configs)
        assert results["model"].memory_estimate_mb > 0

    def test_compute_flops_positive(self) -> None:
        """Test compute FLOPs is positive."""
        configs = [("model", {"num_layers": 12, "hidden_size": 768, "num_heads": 12})]
        results = compare_model_architectures(configs)
        assert results["model"].compute_flops > 0

    def test_empty_list_raises(self) -> None:
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="configs cannot be empty"):
            compare_model_architectures([])

    def test_missing_required_keys_raises(self) -> None:
        """Test missing required keys raises ValueError."""
        # Missing num_heads
        configs = [("model", {"num_layers": 12, "hidden_size": 768})]
        with pytest.raises(ValueError, match="missing required keys"):
            compare_model_architectures(configs)


class TestFormatAnalysisReport:
    """Tests for format_analysis_report function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        param_stats = create_parameter_stats(total_params=110_000_000)
        analysis = create_model_analysis(
            parameter_stats=param_stats,
            memory_estimate_mb=420.0,
            compute_flops=22_000_000_000,
        )
        report = format_analysis_report(analysis)
        assert "Total Parameters:" in report
        assert "Trainable Parameters:" in report
        assert "Memory:" in report
        assert "FLOPs" in report

    def test_millions_format(self) -> None:
        """Test millions formatting."""
        param_stats = create_parameter_stats(total_params=110_000_000)
        analysis = create_model_analysis(parameter_stats=param_stats)
        report = format_analysis_report(analysis)
        assert "M" in report

    def test_billions_format(self) -> None:
        """Test billions formatting."""
        param_stats = create_parameter_stats(total_params=7_000_000_000)
        analysis = create_model_analysis(parameter_stats=param_stats)
        report = format_analysis_report(analysis)
        assert "B" in report

    def test_with_dtype_breakdown(self) -> None:
        """Test with dtype breakdown."""
        param_stats = create_parameter_stats(
            total_params=110_000_000,
            dtype_breakdown={"embeddings": 50_000_000, "attention": 60_000_000},
        )
        analysis = create_model_analysis(parameter_stats=param_stats)
        report = format_analysis_report(analysis)
        assert "Parameter Breakdown:" in report

    def test_with_layers(self) -> None:
        """Test with layer analyses."""
        param_stats = create_parameter_stats(total_params=110_000_000)
        layer = create_layer_analysis("layer_0", param_count=1000, flops=2000)
        analysis = create_model_analysis(
            parameter_stats=param_stats, layer_analyses=(layer,)
        )
        report = format_analysis_report(analysis, include_layers=True)
        assert "Layer-wise Analysis:" in report
        assert "layer_0" in report

    def test_without_layers(self) -> None:
        """Test without layer analyses."""
        param_stats = create_parameter_stats(total_params=110_000_000)
        layer = create_layer_analysis("layer_0", param_count=1000, flops=2000)
        analysis = create_model_analysis(
            parameter_stats=param_stats, layer_analyses=(layer,)
        )
        report = format_analysis_report(analysis, include_layers=False)
        assert "Layer-wise Analysis:" not in report

    def test_none_analysis_raises(self) -> None:
        """Test None analysis raises ValueError."""
        with pytest.raises(ValueError, match="analysis cannot be None"):
            format_analysis_report(None)  # type: ignore[arg-type]


class TestGetRecommendedAnalysisConfig:
    """Tests for get_recommended_analysis_config function."""

    def test_default_config(self) -> None:
        """Test default config."""
        config = get_recommended_analysis_config()
        assert config.include_gradients is True
        assert config.compute_flops is True

    def test_small_model(self) -> None:
        """Test small model config."""
        config = get_recommended_analysis_config("small")
        assert len(config.analysis_types) == 2
        assert "parameter_count" in config.analysis_types

    def test_base_model(self) -> None:
        """Test base model config."""
        config = get_recommended_analysis_config("base")
        assert len(config.analysis_types) == 3
        assert "layer_wise" in config.analysis_types

    def test_large_model(self) -> None:
        """Test large model config."""
        config = get_recommended_analysis_config("large")
        assert len(config.analysis_types) == 4
        assert config.per_layer is True

    def test_detailed_flag(self) -> None:
        """Test detailed flag."""
        config = get_recommended_analysis_config("base", detailed=True)
        assert config.per_layer is True

    def test_case_insensitive(self) -> None:
        """Test model_size is case insensitive."""
        config = get_recommended_analysis_config("BASE")
        assert config is not None

    def test_invalid_model_size_raises(self) -> None:
        """Test invalid model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_analysis_config("invalid")


class TestPropertyBasedTests:
    """Property-based tests using hypothesis."""

    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=64).map(lambda x: x * 64),
        st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=20)
    def test_params_always_positive(
        self, num_layers: int, hidden_size: int, num_heads: int
    ) -> None:
        """Test parameter count is always positive."""
        if hidden_size % num_heads != 0:
            return  # Skip invalid configs
        stats = analyze_parameters(
            num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads
        )
        assert stats.total_params > 0

    @given(
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=1, max_value=32).map(lambda x: x * 64),
        st.integers(min_value=1, max_value=32),
    )
    @settings(max_examples=20)
    def test_flops_always_positive(
        self, num_layers: int, hidden_size: int, num_heads: int
    ) -> None:
        """Test FLOPs is always positive."""
        if hidden_size % num_heads != 0:
            return  # Skip invalid configs
        flops = estimate_model_flops(
            num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads
        )
        assert flops > 0

    @given(st.sampled_from(list(VALID_ANALYSIS_TYPES)))
    @settings(max_examples=10)
    def test_all_analysis_types_gettable(self, analysis_type: str) -> None:
        """Test all valid analysis types can be gotten."""
        result = get_analysis_type(analysis_type)
        assert result.value == analysis_type

    @given(st.sampled_from(list(VALID_STATISTIC_TYPES)))
    @settings(max_examples=10)
    def test_all_statistic_types_gettable(self, statistic_type: str) -> None:
        """Test all valid statistic types can be gotten."""
        result = get_statistic_type(statistic_type)
        assert result.value == statistic_type

    @given(st.sampled_from(list(VALID_VISUALIZATION_TYPES)))
    @settings(max_examples=10)
    def test_all_visualization_types_gettable(self, viz_type: str) -> None:
        """Test all valid visualization types can be gotten."""
        result = get_visualization_type(viz_type)
        assert result.value == viz_type

    @given(st.integers(min_value=1, max_value=64).map(lambda x: x * 64))
    @settings(max_examples=10)
    def test_layer_statistics_consistent(self, hidden_size: int) -> None:
        """Test layer statistics are consistent."""
        attn = compute_layer_statistics(
            layer_name="attention", hidden_size=hidden_size, layer_type="attention"
        )
        ffn = compute_layer_statistics(
            layer_name="ffn", hidden_size=hidden_size, layer_type="ffn"
        )
        norm = compute_layer_statistics(
            layer_name="norm", hidden_size=hidden_size, layer_type="norm"
        )
        # FFN typically has more params than attention per layer
        assert attn.param_count > 0
        assert ffn.param_count > 0
        assert norm.param_count > 0
