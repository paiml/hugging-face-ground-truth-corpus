"""Tests for interpretability functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.interpretability import (
    AggregationMethod,
    AttentionConfig,
    AttributionConfig,
    AttributionMethod,
    InterpretabilityResult,
    VisualizationConfig,
    VisualizationType,
    aggregate_attention_weights,
    calculate_attribution_scores,
    create_attention_config,
    create_attribution_config,
    create_visualization_config,
    estimate_interpretation_time,
    format_interpretation_result,
    get_aggregation_method,
    get_attribution_method,
    get_recommended_interpretability_config,
    get_visualization_type,
    list_aggregation_methods,
    list_attribution_methods,
    list_visualization_types,
    validate_aggregation_method,
    validate_attention_config,
    validate_attribution_config,
    validate_attribution_method,
    validate_visualization_config,
    validate_visualization_type,
)


class TestAttributionMethod:
    """Tests for AttributionMethod enum."""

    def test_integrated_gradients_value(self) -> None:
        """Test INTEGRATED_GRADIENTS value."""
        assert AttributionMethod.INTEGRATED_GRADIENTS.value == "integrated_gradients"

    def test_saliency_value(self) -> None:
        """Test SALIENCY value."""
        assert AttributionMethod.SALIENCY.value == "saliency"

    def test_grad_cam_value(self) -> None:
        """Test GRAD_CAM value."""
        assert AttributionMethod.GRAD_CAM.value == "grad_cam"

    def test_attention_rollout_value(self) -> None:
        """Test ATTENTION_ROLLOUT value."""
        assert AttributionMethod.ATTENTION_ROLLOUT.value == "attention_rollout"

    def test_lime_value(self) -> None:
        """Test LIME value."""
        assert AttributionMethod.LIME.value == "lime"

    def test_shap_value(self) -> None:
        """Test SHAP value."""
        assert AttributionMethod.SHAP.value == "shap"


class TestVisualizationType:
    """Tests for VisualizationType enum."""

    def test_heatmap_value(self) -> None:
        """Test HEATMAP value."""
        assert VisualizationType.HEATMAP.value == "heatmap"

    def test_bar_chart_value(self) -> None:
        """Test BAR_CHART value."""
        assert VisualizationType.BAR_CHART.value == "bar_chart"

    def test_text_highlight_value(self) -> None:
        """Test TEXT_HIGHLIGHT value."""
        assert VisualizationType.TEXT_HIGHLIGHT.value == "text_highlight"

    def test_attention_pattern_value(self) -> None:
        """Test ATTENTION_PATTERN value."""
        assert VisualizationType.ATTENTION_PATTERN.value == "attention_pattern"


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_mean_value(self) -> None:
        """Test MEAN value."""
        assert AggregationMethod.MEAN.value == "mean"

    def test_max_value(self) -> None:
        """Test MAX value."""
        assert AggregationMethod.MAX.value == "max"

    def test_l2_norm_value(self) -> None:
        """Test L2_NORM value."""
        assert AggregationMethod.L2_NORM.value == "l2_norm"

    def test_sum_value(self) -> None:
        """Test SUM value."""
        assert AggregationMethod.SUM.value == "sum"


class TestAttentionConfig:
    """Tests for AttentionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AttentionConfig()
        assert config.layer_indices is None
        assert config.head_indices is None
        assert config.aggregation == AggregationMethod.MEAN
        assert config.normalize is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AttentionConfig(
            layer_indices=[0, 1, 2],
            head_indices=[0, 4],
            aggregation=AggregationMethod.MAX,
            normalize=False,
        )
        assert config.layer_indices == [0, 1, 2]
        assert config.head_indices == [0, 4]
        assert config.aggregation == AggregationMethod.MAX
        assert config.normalize is False

    def test_frozen(self) -> None:
        """Test that AttentionConfig is immutable."""
        config = AttentionConfig()
        with pytest.raises(AttributeError):
            config.normalize = False  # type: ignore[misc]


class TestAttributionConfig:
    """Tests for AttributionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AttributionConfig()
        assert config.method == AttributionMethod.INTEGRATED_GRADIENTS
        assert config.baseline_type == "zero"
        assert config.n_steps == 50
        assert config.internal_batch_size == 32

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AttributionConfig(
            method=AttributionMethod.SHAP,
            baseline_type="uniform",
            n_steps=100,
            internal_batch_size=16,
        )
        assert config.method == AttributionMethod.SHAP
        assert config.baseline_type == "uniform"
        assert config.n_steps == 100
        assert config.internal_batch_size == 16

    def test_frozen(self) -> None:
        """Test that AttributionConfig is immutable."""
        config = AttributionConfig()
        with pytest.raises(AttributeError):
            config.n_steps = 100  # type: ignore[misc]


class TestInterpretabilityResult:
    """Tests for InterpretabilityResult dataclass."""

    def test_creation(self) -> None:
        """Test creating InterpretabilityResult instance."""
        result = InterpretabilityResult(
            attributions=[0.1, 0.5, 0.2],
            attention_weights=None,
            tokens=["a", "b", "c"],
            layer_info=None,
        )
        assert len(result.attributions) == 3
        assert result.tokens == ["a", "b", "c"]

    def test_with_all_fields(self) -> None:
        """Test result with all fields populated."""
        result = InterpretabilityResult(
            attributions=[0.5],
            attention_weights=[[0.1, 0.2]],
            tokens=["test"],
            layer_info={"layer_0": [0.5]},
        )
        assert result.attention_weights is not None
        assert result.layer_info is not None

    def test_frozen(self) -> None:
        """Test that InterpretabilityResult is immutable."""
        result = InterpretabilityResult([0.1], None, None, None)
        with pytest.raises(AttributeError):
            result.tokens = ["new"]  # type: ignore[misc]


class TestVisualizationConfig:
    """Tests for VisualizationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VisualizationConfig()
        assert config.viz_type == VisualizationType.HEATMAP
        assert config.colormap == "viridis"
        assert config.figsize == (10, 6)
        assert config.show_values is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = VisualizationConfig(
            viz_type=VisualizationType.BAR_CHART,
            colormap="coolwarm",
            figsize=(12, 8),
            show_values=True,
        )
        assert config.viz_type == VisualizationType.BAR_CHART
        assert config.colormap == "coolwarm"
        assert config.figsize == (12, 8)
        assert config.show_values is True

    def test_frozen(self) -> None:
        """Test that VisualizationConfig is immutable."""
        config = VisualizationConfig()
        with pytest.raises(AttributeError):
            config.colormap = "new"  # type: ignore[misc]


class TestCreateAttentionConfig:
    """Tests for create_attention_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating default config."""
        config = create_attention_config()
        assert config.layer_indices is None
        assert config.aggregation == AggregationMethod.MEAN

    def test_with_layer_indices(self) -> None:
        """Test with layer indices."""
        config = create_attention_config(layer_indices=[0, 1, 2])
        assert config.layer_indices == [0, 1, 2]

    def test_with_head_indices(self) -> None:
        """Test with head indices."""
        config = create_attention_config(head_indices=[0, 4, 8])
        assert config.head_indices == [0, 4, 8]

    def test_with_aggregation(self) -> None:
        """Test with custom aggregation."""
        config = create_attention_config(aggregation=AggregationMethod.MAX)
        assert config.aggregation == AggregationMethod.MAX

    def test_with_normalize_false(self) -> None:
        """Test with normalize=False."""
        config = create_attention_config(normalize=False)
        assert config.normalize is False


class TestValidateAttentionConfig:
    """Tests for validate_attention_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_attention_config()
        validate_attention_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_attention_config(None)  # type: ignore[arg-type]

    def test_negative_layer_index_raises_error(self) -> None:
        """Test that negative layer index raises ValueError."""
        config = AttentionConfig(layer_indices=[-1])
        with pytest.raises(ValueError, match="layer_indices cannot contain negative"):
            validate_attention_config(config)

    def test_negative_head_index_raises_error(self) -> None:
        """Test that negative head index raises ValueError."""
        config = AttentionConfig(head_indices=[-5])
        with pytest.raises(ValueError, match="head_indices cannot contain negative"):
            validate_attention_config(config)

    def test_valid_layer_indices(self) -> None:
        """Test valid layer indices."""
        config = AttentionConfig(layer_indices=[0, 1, 2])
        validate_attention_config(config)  # Should not raise


class TestCreateAttributionConfig:
    """Tests for create_attribution_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating default config."""
        config = create_attribution_config()
        assert config.method == AttributionMethod.INTEGRATED_GRADIENTS
        assert config.baseline_type == "zero"

    def test_with_method(self) -> None:
        """Test with custom method."""
        config = create_attribution_config(method=AttributionMethod.SHAP)
        assert config.method == AttributionMethod.SHAP

    def test_with_baseline_type(self) -> None:
        """Test with custom baseline type."""
        config = create_attribution_config(baseline_type="uniform")
        assert config.baseline_type == "uniform"

    def test_with_n_steps(self) -> None:
        """Test with custom n_steps."""
        config = create_attribution_config(n_steps=100)
        assert config.n_steps == 100

    def test_with_batch_size(self) -> None:
        """Test with custom batch size."""
        config = create_attribution_config(internal_batch_size=16)
        assert config.internal_batch_size == 16


class TestValidateAttributionConfig:
    """Tests for validate_attribution_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_attribution_config()
        validate_attribution_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_attribution_config(None)  # type: ignore[arg-type]

    def test_zero_n_steps_raises_error(self) -> None:
        """Test that zero n_steps raises ValueError."""
        config = AttributionConfig(n_steps=0)
        with pytest.raises(ValueError, match="n_steps must be positive"):
            validate_attribution_config(config)

    def test_negative_n_steps_raises_error(self) -> None:
        """Test that negative n_steps raises ValueError."""
        config = AttributionConfig(n_steps=-5)
        with pytest.raises(ValueError, match="n_steps must be positive"):
            validate_attribution_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch size raises ValueError."""
        config = AttributionConfig(internal_batch_size=0)
        with pytest.raises(ValueError, match="internal_batch_size must be positive"):
            validate_attribution_config(config)

    def test_empty_baseline_type_raises_error(self) -> None:
        """Test that empty baseline type raises ValueError."""
        config = AttributionConfig(baseline_type="")
        with pytest.raises(ValueError, match="baseline_type cannot be empty"):
            validate_attribution_config(config)


class TestCreateVisualizationConfig:
    """Tests for create_visualization_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating default config."""
        config = create_visualization_config()
        assert config.viz_type == VisualizationType.HEATMAP
        assert config.colormap == "viridis"

    def test_with_viz_type(self) -> None:
        """Test with custom visualization type."""
        config = create_visualization_config(viz_type=VisualizationType.BAR_CHART)
        assert config.viz_type == VisualizationType.BAR_CHART

    def test_with_colormap(self) -> None:
        """Test with custom colormap."""
        config = create_visualization_config(colormap="coolwarm")
        assert config.colormap == "coolwarm"

    def test_with_figsize(self) -> None:
        """Test with custom figsize."""
        config = create_visualization_config(figsize=(15, 10))
        assert config.figsize == (15, 10)

    def test_with_show_values(self) -> None:
        """Test with show_values=True."""
        config = create_visualization_config(show_values=True)
        assert config.show_values is True


class TestValidateVisualizationConfig:
    """Tests for validate_visualization_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_visualization_config()
        validate_visualization_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_visualization_config(None)  # type: ignore[arg-type]

    def test_empty_colormap_raises_error(self) -> None:
        """Test that empty colormap raises ValueError."""
        config = VisualizationConfig(colormap="")
        with pytest.raises(ValueError, match="colormap cannot be empty"):
            validate_visualization_config(config)

    def test_zero_width_figsize_raises_error(self) -> None:
        """Test that zero width raises ValueError."""
        config = VisualizationConfig(figsize=(0, 5))
        with pytest.raises(ValueError, match="figsize dimensions must be positive"):
            validate_visualization_config(config)

    def test_zero_height_figsize_raises_error(self) -> None:
        """Test that zero height raises ValueError."""
        config = VisualizationConfig(figsize=(10, 0))
        with pytest.raises(ValueError, match="figsize dimensions must be positive"):
            validate_visualization_config(config)


class TestListAttributionMethods:
    """Tests for list_attribution_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_attribution_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_attribution_methods()
        assert "integrated_gradients" in methods
        assert "shap" in methods
        assert "lime" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_attribution_methods()
        assert methods == sorted(methods)


class TestValidateAttributionMethod:
    """Tests for validate_attribution_method function."""

    def test_valid_integrated_gradients(self) -> None:
        """Test validation of integrated_gradients."""
        assert validate_attribution_method("integrated_gradients") is True

    def test_valid_shap(self) -> None:
        """Test validation of shap."""
        assert validate_attribution_method("shap") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_attribution_method("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_attribution_method("") is False


class TestGetAttributionMethod:
    """Tests for get_attribution_method function."""

    def test_get_integrated_gradients(self) -> None:
        """Test getting INTEGRATED_GRADIENTS."""
        result = get_attribution_method("integrated_gradients")
        assert result == AttributionMethod.INTEGRATED_GRADIENTS

    def test_get_shap(self) -> None:
        """Test getting SHAP."""
        result = get_attribution_method("shap")
        assert result == AttributionMethod.SHAP

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid attribution method"):
            get_attribution_method("invalid")


class TestListVisualizationTypes:
    """Tests for list_visualization_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_visualization_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_visualization_types()
        assert "heatmap" in types
        assert "bar_chart" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_visualization_types()
        assert types == sorted(types)


class TestValidateVisualizationType:
    """Tests for validate_visualization_type function."""

    def test_valid_heatmap(self) -> None:
        """Test validation of heatmap."""
        assert validate_visualization_type("heatmap") is True

    def test_valid_bar_chart(self) -> None:
        """Test validation of bar_chart."""
        assert validate_visualization_type("bar_chart") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_visualization_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_visualization_type("") is False


class TestGetVisualizationType:
    """Tests for get_visualization_type function."""

    def test_get_heatmap(self) -> None:
        """Test getting HEATMAP."""
        result = get_visualization_type("heatmap")
        assert result == VisualizationType.HEATMAP

    def test_get_bar_chart(self) -> None:
        """Test getting BAR_CHART."""
        result = get_visualization_type("bar_chart")
        assert result == VisualizationType.BAR_CHART

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid visualization type"):
            get_visualization_type("invalid")


class TestListAggregationMethods:
    """Tests for list_aggregation_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_aggregation_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_aggregation_methods()
        assert "mean" in methods
        assert "max" in methods
        assert "l2_norm" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_aggregation_methods()
        assert methods == sorted(methods)


class TestValidateAggregationMethod:
    """Tests for validate_aggregation_method function."""

    def test_valid_mean(self) -> None:
        """Test validation of mean."""
        assert validate_aggregation_method("mean") is True

    def test_valid_l2_norm(self) -> None:
        """Test validation of l2_norm."""
        assert validate_aggregation_method("l2_norm") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_aggregation_method("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_aggregation_method("") is False


class TestGetAggregationMethod:
    """Tests for get_aggregation_method function."""

    def test_get_mean(self) -> None:
        """Test getting MEAN."""
        result = get_aggregation_method("mean")
        assert result == AggregationMethod.MEAN

    def test_get_l2_norm(self) -> None:
        """Test getting L2_NORM."""
        result = get_aggregation_method("l2_norm")
        assert result == AggregationMethod.L2_NORM

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid aggregation method"):
            get_aggregation_method("invalid")


class TestCalculateAttributionScores:
    """Tests for calculate_attribution_scores function."""

    def test_integrated_gradients(self) -> None:
        """Test integrated gradients calculation."""
        config = create_attribution_config(
            method=AttributionMethod.INTEGRATED_GRADIENTS
        )
        gradients = [0.5, -0.3, 0.8]
        inputs = [1.0, 2.0, 0.5]
        scores = calculate_attribution_scores(gradients, inputs, config)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.5)  # 0.5 * 1.0
        assert scores[1] == pytest.approx(-0.6)  # -0.3 * 2.0
        assert scores[2] == pytest.approx(0.4)  # 0.8 * 0.5

    def test_saliency(self) -> None:
        """Test saliency calculation."""
        config = create_attribution_config(method=AttributionMethod.SALIENCY)
        gradients = [0.5, -0.3, 0.8]
        inputs = [1.0, 2.0, 0.5]
        scores = calculate_attribution_scores(gradients, inputs, config)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.5)  # abs(0.5)
        assert scores[1] == pytest.approx(0.3)  # abs(-0.3)
        assert scores[2] == pytest.approx(0.8)  # abs(0.8)

    def test_grad_cam(self) -> None:
        """Test grad_cam calculation."""
        config = create_attribution_config(method=AttributionMethod.GRAD_CAM)
        gradients = [0.5, 0.3]
        inputs = [1.0, 2.0]
        scores = calculate_attribution_scores(gradients, inputs, config)
        assert len(scores) == 2

    def test_none_gradients_raises_error(self) -> None:
        """Test that None gradients raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(ValueError, match="gradients cannot be None"):
            calculate_attribution_scores(None, [1.0], config)  # type: ignore[arg-type]

    def test_empty_gradients_raises_error(self) -> None:
        """Test that empty gradients raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            calculate_attribution_scores([], [1.0], config)

    def test_none_inputs_raises_error(self) -> None:
        """Test that None inputs raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(ValueError, match="inputs cannot be None"):
            calculate_attribution_scores([0.5], None, config)  # type: ignore[arg-type]

    def test_empty_inputs_raises_error(self) -> None:
        """Test that empty inputs raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(ValueError, match="inputs cannot be empty"):
            calculate_attribution_scores([0.5], [], config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            calculate_attribution_scores([0.5], [1.0], None)  # type: ignore[arg-type]

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_attribution_scores([0.5], [1.0, 2.0], config)


class TestAggregateAttentionWeights:
    """Tests for aggregate_attention_weights function."""

    def test_mean_aggregation(self) -> None:
        """Test mean aggregation."""
        config = create_attention_config(aggregation=AggregationMethod.MEAN)
        weights = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        agg = aggregate_attention_weights(weights, config)
        assert len(agg) == 3
        # Mean of [0.1, 0.2] = 0.15, normalized by sum
        assert all(v >= 0 for v in agg)

    def test_max_aggregation(self) -> None:
        """Test max aggregation."""
        config = create_attention_config(
            aggregation=AggregationMethod.MAX, normalize=False
        )
        weights = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        agg = aggregate_attention_weights(weights, config)
        assert agg[0] == pytest.approx(0.2)  # max(0.1, 0.2)
        assert agg[1] == pytest.approx(0.3)  # max(0.2, 0.3)
        assert agg[2] == pytest.approx(0.4)  # max(0.3, 0.4)

    def test_sum_aggregation(self) -> None:
        """Test sum aggregation."""
        config = create_attention_config(
            aggregation=AggregationMethod.SUM, normalize=False
        )
        weights = [[0.1, 0.2], [0.2, 0.3]]
        agg = aggregate_attention_weights(weights, config)
        assert agg[0] == pytest.approx(0.3)  # 0.1 + 0.2
        assert agg[1] == pytest.approx(0.5)  # 0.2 + 0.3

    def test_l2_norm_aggregation(self) -> None:
        """Test L2 norm aggregation."""
        config = create_attention_config(
            aggregation=AggregationMethod.L2_NORM, normalize=False
        )
        weights = [[0.3, 0.0], [0.4, 0.0]]
        agg = aggregate_attention_weights(weights, config)
        assert agg[0] == pytest.approx(0.5)  # sqrt(0.3^2 + 0.4^2)
        assert agg[1] == pytest.approx(0.0)

    def test_normalization(self) -> None:
        """Test that normalization works."""
        config = create_attention_config(normalize=True)
        weights = [[0.5, 0.5], [0.5, 0.5]]
        agg = aggregate_attention_weights(weights, config)
        assert sum(agg) == pytest.approx(1.0)

    def test_none_weights_raises_error(self) -> None:
        """Test that None weights raises ValueError."""
        config = create_attention_config()
        with pytest.raises(ValueError, match="attention_weights cannot be None"):
            aggregate_attention_weights(None, config)  # type: ignore[arg-type]

    def test_empty_weights_raises_error(self) -> None:
        """Test that empty weights raises ValueError."""
        config = create_attention_config()
        with pytest.raises(ValueError, match="attention_weights cannot be empty"):
            aggregate_attention_weights([], config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            aggregate_attention_weights([[0.5]], None)  # type: ignore[arg-type]


class TestEstimateInterpretationTime:
    """Tests for estimate_interpretation_time function."""

    def test_basic_estimation(self) -> None:
        """Test basic time estimation."""
        config = create_attribution_config(n_steps=50)
        time_est = estimate_interpretation_time(100, config)
        assert time_est > 0

    def test_saliency_faster_than_ig(self) -> None:
        """Test that saliency is faster than integrated gradients."""
        config_sal = create_attribution_config(method=AttributionMethod.SALIENCY)
        config_ig = create_attribution_config(
            method=AttributionMethod.INTEGRATED_GRADIENTS
        )
        time_sal = estimate_interpretation_time(100, config_sal)
        time_ig = estimate_interpretation_time(100, config_ig)
        assert time_sal < time_ig

    def test_more_tokens_more_time(self) -> None:
        """Test that more tokens takes more time."""
        config = create_attribution_config()
        time_100 = estimate_interpretation_time(100, config)
        time_200 = estimate_interpretation_time(200, config)
        assert time_200 > time_100

    def test_zero_tokens_raises_error(self) -> None:
        """Test that zero tokens raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(ValueError, match="num_tokens must be positive"):
            estimate_interpretation_time(0, config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            estimate_interpretation_time(100, None)  # type: ignore[arg-type]

    def test_zero_inference_time_raises_error(self) -> None:
        """Test that zero inference time raises ValueError."""
        config = create_attribution_config()
        with pytest.raises(
            ValueError, match="model_inference_time_ms must be positive"
        ):
            estimate_interpretation_time(100, config, model_inference_time_ms=0)


class TestFormatInterpretationResult:
    """Tests for format_interpretation_result function."""

    def test_basic_formatting(self) -> None:
        """Test basic result formatting."""
        result = InterpretabilityResult(
            attributions=[0.5, 0.3, 0.1],
            attention_weights=None,
            tokens=["The", "quick", "fox"],
            layer_info=None,
        )
        formatted = format_interpretation_result(result)
        assert "Interpretability Analysis Results" in formatted
        assert "The" in formatted or "quick" in formatted

    def test_format_with_all_fields(self) -> None:
        """Test formatting with all fields."""
        result = InterpretabilityResult(
            attributions=[0.5],
            attention_weights=[[0.1, 0.2]],
            tokens=["test"],
            layer_info={"layer": [0.1]},
        )
        formatted = format_interpretation_result(result)
        assert "Attention weights" in formatted
        assert "Layer info" in formatted

    def test_top_k_limits_output(self) -> None:
        """Test that top_k limits output."""
        result = InterpretabilityResult(
            attributions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            attention_weights=None,
            tokens=["a", "b", "c", "d", "e", "f"],
            layer_info=None,
        )
        formatted = format_interpretation_result(result, top_k=2)
        assert "Top 2" in formatted

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            format_interpretation_result(None)  # type: ignore[arg-type]

    def test_zero_top_k_raises_error(self) -> None:
        """Test that zero top_k raises ValueError."""
        result = InterpretabilityResult([0.1], None, None, None)
        with pytest.raises(ValueError, match="top_k must be positive"):
            format_interpretation_result(result, top_k=0)

    def test_format_without_tokens(self) -> None:
        """Test formatting without tokens."""
        result = InterpretabilityResult(
            attributions=[0.5, 0.3],
            attention_weights=None,
            tokens=None,
            layer_info=None,
        )
        formatted = format_interpretation_result(result)
        assert "[0]" in formatted or "[1]" in formatted


class TestGetRecommendedInterpretabilityConfig:
    """Tests for get_recommended_interpretability_config function."""

    def test_classification_task(self) -> None:
        """Test config for classification task."""
        attr_cfg, attn_cfg = get_recommended_interpretability_config("classification")
        assert attr_cfg.method == AttributionMethod.INTEGRATED_GRADIENTS
        assert attn_cfg.aggregation == AggregationMethod.MEAN

    def test_qa_task(self) -> None:
        """Test config for QA task."""
        attr_cfg, attn_cfg = get_recommended_interpretability_config("qa")
        assert attr_cfg.method == AttributionMethod.ATTENTION_ROLLOUT
        assert attn_cfg.aggregation == AggregationMethod.MAX

    def test_vision_task(self) -> None:
        """Test config for vision task."""
        attr_cfg, attn_cfg = get_recommended_interpretability_config("vision")
        assert attr_cfg.method == AttributionMethod.GRAD_CAM
        assert attn_cfg.aggregation == AggregationMethod.L2_NORM

    def test_generation_task(self) -> None:
        """Test config for generation task."""
        attr_cfg, _attn_cfg = get_recommended_interpretability_config("generation")
        assert attr_cfg.method == AttributionMethod.SALIENCY

    def test_unknown_task_uses_default(self) -> None:
        """Test that unknown task uses default config."""
        attr_cfg, _attn_cfg = get_recommended_interpretability_config("unknown")
        assert attr_cfg.method == AttributionMethod.INTEGRATED_GRADIENTS

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        attr_cfg1, _ = get_recommended_interpretability_config("CLASSIFICATION")
        attr_cfg2, _ = get_recommended_interpretability_config("classification")
        assert attr_cfg1.method == attr_cfg2.method

    def test_empty_task_raises_error(self) -> None:
        """Test that empty task raises ValueError."""
        with pytest.raises(ValueError, match="task cannot be empty"):
            get_recommended_interpretability_config("")


class TestPropertyBased:
    """Property-based tests for interpretability functions."""

    @given(
        st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=10)
    def test_saliency_all_positive(self, gradients: list[float]) -> None:
        """Test that saliency scores are all non-negative."""
        config = create_attribution_config(method=AttributionMethod.SALIENCY)
        inputs = [1.0] * len(gradients)
        scores = calculate_attribution_scores(gradients, inputs, config)
        assert all(s >= 0 for s in scores)

    @given(
        st.lists(
            st.floats(
                min_value=0.01,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=10)
    def test_normalized_sum_to_one(self, values: list[float]) -> None:
        """Test that normalized weights sum to 1."""
        config = create_attention_config(normalize=True)
        weights = [values, values]  # 2 heads with same values
        agg = aggregate_attention_weights(weights, config)
        assert sum(agg) == pytest.approx(1.0, rel=1e-5)

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=10)
    def test_time_estimate_positive(self, num_tokens: int) -> None:
        """Test that time estimates are always positive."""
        config = create_attribution_config()
        time_est = estimate_interpretation_time(num_tokens, config)
        assert time_est > 0
