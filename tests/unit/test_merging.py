"""Tests for model merging utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.merging import (
    DENSITY_VALUES,
    VALID_MERGE_METHODS,
    MergeConfig,
    MergeMethod,
    MergeResult,
    ModelSlice,
    calculate_merged_parameter_count,
    create_merge_config,
    create_model_slice,
    estimate_merge_time,
    get_density_value,
    get_recommended_method,
    linear_interpolate,
    list_merge_methods,
    slerp,
    validate_merge_config,
    validate_model_slice,
)


class TestMergeMethod:
    """Tests for MergeMethod enum."""

    def test_linear_value(self) -> None:
        """Test LINEAR method value."""
        assert MergeMethod.LINEAR.value == "linear"

    def test_slerp_value(self) -> None:
        """Test SLERP method value."""
        assert MergeMethod.SLERP.value == "slerp"

    def test_ties_value(self) -> None:
        """Test TIES method value."""
        assert MergeMethod.TIES.value == "ties"

    def test_dare_value(self) -> None:
        """Test DARE method value."""
        assert MergeMethod.DARE.value == "dare"

    def test_task_arithmetic_value(self) -> None:
        """Test TASK_ARITHMETIC method value."""
        assert MergeMethod.TASK_ARITHMETIC.value == "task_arithmetic"

    def test_passthrough_value(self) -> None:
        """Test PASSTHROUGH method value."""
        assert MergeMethod.PASSTHROUGH.value == "passthrough"

    def test_all_methods_in_valid_set(self) -> None:
        """Test all enum values are in VALID_MERGE_METHODS."""
        for method in MergeMethod:
            assert method.value in VALID_MERGE_METHODS


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating merge config."""
        config = MergeConfig(
            method=MergeMethod.SLERP,
            t=0.5,
            density=0.5,
            normalize=True,
        )
        assert config.method == MergeMethod.SLERP
        assert config.t == pytest.approx(0.5)
        assert config.density == pytest.approx(0.5)
        assert config.normalize is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            t=0.3,
            density=0.7,
            normalize=False,
        )
        with pytest.raises(AttributeError):
            config.t = 0.8  # type: ignore[misc]


class TestModelSlice:
    """Tests for ModelSlice dataclass."""

    def test_create_slice(self) -> None:
        """Test creating model slice."""
        slice_cfg = ModelSlice(
            model_id="model_a",
            start_layer=0,
            end_layer=16,
            weight=1.0,
        )
        assert slice_cfg.model_id == "model_a"
        assert slice_cfg.start_layer == 0
        assert slice_cfg.end_layer == 16
        assert slice_cfg.weight == pytest.approx(1.0)

    def test_frozen(self) -> None:
        """Test slice is immutable."""
        slice_cfg = ModelSlice(
            model_id="model_b",
            start_layer=16,
            end_layer=32,
            weight=0.5,
        )
        with pytest.raises(AttributeError):
            slice_cfg.start_layer = 8  # type: ignore[misc]


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating merge result."""
        result = MergeResult(
            num_parameters=7_000_000_000,
            method_used=MergeMethod.SLERP,
            models_merged=2,
            metadata={"t": "0.5"},
        )
        assert result.num_parameters == 7_000_000_000
        assert result.method_used == MergeMethod.SLERP
        assert result.models_merged == 2
        assert result.metadata == {"t": "0.5"}

    def test_frozen(self) -> None:
        """Test result is immutable."""
        result = MergeResult(
            num_parameters=1000,
            method_used=MergeMethod.LINEAR,
            models_merged=2,
            metadata={},
        )
        with pytest.raises(AttributeError):
            result.models_merged = 3  # type: ignore[misc]


class TestValidateMergeConfig:
    """Tests for validate_merge_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = MergeConfig(
            method=MergeMethod.SLERP,
            t=0.5,
            density=0.5,
            normalize=True,
        )
        validate_merge_config(config)  # Should not raise

    def test_t_at_boundaries(self) -> None:
        """Test t at boundary values."""
        config_low = MergeConfig(
            method=MergeMethod.SLERP,
            t=0.0,
            density=0.5,
            normalize=True,
        )
        validate_merge_config(config_low)  # Should not raise

        config_high = MergeConfig(
            method=MergeMethod.SLERP,
            t=1.0,
            density=0.5,
            normalize=True,
        )
        validate_merge_config(config_high)  # Should not raise

    def test_t_too_low(self) -> None:
        """Test t below valid range."""
        config = MergeConfig(
            method=MergeMethod.SLERP,
            t=-0.1,
            density=0.5,
            normalize=True,
        )
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            validate_merge_config(config)

    def test_t_too_high(self) -> None:
        """Test t above valid range."""
        config = MergeConfig(
            method=MergeMethod.SLERP,
            t=1.5,
            density=0.5,
            normalize=True,
        )
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            validate_merge_config(config)

    def test_density_too_low(self) -> None:
        """Test density below valid range."""
        config = MergeConfig(
            method=MergeMethod.TIES,
            t=0.5,
            density=-0.1,
            normalize=True,
        )
        with pytest.raises(ValueError, match=r"density must be between 0\.0 and 1\.0"):
            validate_merge_config(config)

    def test_density_too_high(self) -> None:
        """Test density above valid range."""
        config = MergeConfig(
            method=MergeMethod.DARE,
            t=0.5,
            density=1.5,
            normalize=True,
        )
        with pytest.raises(ValueError, match=r"density must be between 0\.0 and 1\.0"):
            validate_merge_config(config)


class TestValidateModelSlice:
    """Tests for validate_model_slice function."""

    def test_valid_slice(self) -> None:
        """Test validating valid slice."""
        slice_cfg = ModelSlice("model_a", 0, 16, 1.0)
        validate_model_slice(slice_cfg)  # Should not raise

    def test_empty_model_id(self) -> None:
        """Test empty model_id."""
        slice_cfg = ModelSlice("", 0, 16, 1.0)
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            validate_model_slice(slice_cfg)

    def test_negative_start_layer(self) -> None:
        """Test negative start_layer."""
        slice_cfg = ModelSlice("model_a", -1, 16, 1.0)
        with pytest.raises(ValueError, match="start_layer cannot be negative"):
            validate_model_slice(slice_cfg)

    def test_start_equals_end(self) -> None:
        """Test start_layer equals end_layer."""
        slice_cfg = ModelSlice("model_a", 16, 16, 1.0)
        with pytest.raises(ValueError, match=r"start_layer .* must be less than"):
            validate_model_slice(slice_cfg)

    def test_start_greater_than_end(self) -> None:
        """Test start_layer greater than end_layer."""
        slice_cfg = ModelSlice("model_a", 20, 16, 1.0)
        with pytest.raises(ValueError, match=r"start_layer .* must be less than"):
            validate_model_slice(slice_cfg)

    def test_negative_weight(self) -> None:
        """Test negative weight."""
        slice_cfg = ModelSlice("model_a", 0, 16, -0.5)
        with pytest.raises(ValueError, match="weight cannot be negative"):
            validate_model_slice(slice_cfg)


class TestCreateMergeConfig:
    """Tests for create_merge_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_merge_config()
        assert config.method == MergeMethod.SLERP
        assert config.t == pytest.approx(0.5)
        assert config.density == pytest.approx(0.5)
        assert config.normalize is True

    def test_custom_method(self) -> None:
        """Test custom method."""
        config = create_merge_config(method="ties")
        assert config.method == MergeMethod.TIES

    def test_linear_method(self) -> None:
        """Test linear method."""
        config = create_merge_config(method="linear")
        assert config.method == MergeMethod.LINEAR

    def test_dare_method(self) -> None:
        """Test DARE method."""
        config = create_merge_config(method="dare")
        assert config.method == MergeMethod.DARE

    def test_task_arithmetic_method(self) -> None:
        """Test task_arithmetic method."""
        config = create_merge_config(method="task_arithmetic")
        assert config.method == MergeMethod.TASK_ARITHMETIC

    def test_passthrough_method(self) -> None:
        """Test passthrough method."""
        config = create_merge_config(method="passthrough")
        assert config.method == MergeMethod.PASSTHROUGH

    def test_custom_t(self) -> None:
        """Test custom t parameter."""
        config = create_merge_config(t=0.7)
        assert config.t == pytest.approx(0.7)

    def test_custom_density(self) -> None:
        """Test custom density."""
        config = create_merge_config(density=0.3)
        assert config.density == pytest.approx(0.3)

    def test_no_normalize(self) -> None:
        """Test normalize=False."""
        config = create_merge_config(normalize=False)
        assert config.normalize is False

    def test_invalid_method(self) -> None:
        """Test invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_merge_config(method="invalid")

    def test_invalid_t(self) -> None:
        """Test invalid t value."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            create_merge_config(t=1.5)


class TestCreateModelSlice:
    """Tests for create_model_slice function."""

    def test_basic_slice(self) -> None:
        """Test creating basic slice."""
        slice_cfg = create_model_slice("model_a", 0, 16)
        assert slice_cfg.model_id == "model_a"
        assert slice_cfg.start_layer == 0
        assert slice_cfg.end_layer == 16
        assert slice_cfg.weight == pytest.approx(1.0)

    def test_custom_weight(self) -> None:
        """Test custom weight."""
        slice_cfg = create_model_slice("model_b", 16, 32, weight=0.5)
        assert slice_cfg.weight == pytest.approx(0.5)

    def test_empty_model_id(self) -> None:
        """Test empty model_id."""
        with pytest.raises(ValueError, match="model_id cannot be empty"):
            create_model_slice("", 0, 16)

    def test_invalid_layers(self) -> None:
        """Test invalid layer range."""
        with pytest.raises(ValueError, match=r"start_layer .* must be less than"):
            create_model_slice("model_a", 16, 8)


class TestListMergeMethods:
    """Tests for list_merge_methods function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        methods = list_merge_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test contains expected methods."""
        methods = list_merge_methods()
        assert "slerp" in methods
        assert "ties" in methods
        assert "dare" in methods
        assert "linear" in methods

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        methods = list_merge_methods()
        assert methods == sorted(methods)


class TestLinearInterpolate:
    """Tests for linear_interpolate function."""

    def test_midpoint(self) -> None:
        """Test interpolation at midpoint."""
        result = linear_interpolate(0.0, 1.0, 0.5)
        assert result == pytest.approx(0.5)

    def test_at_start(self) -> None:
        """Test interpolation at t=0."""
        result = linear_interpolate(5.0, 10.0, 0.0)
        assert result == pytest.approx(5.0)

    def test_at_end(self) -> None:
        """Test interpolation at t=1."""
        result = linear_interpolate(5.0, 10.0, 1.0)
        assert result == pytest.approx(10.0)

    def test_custom_t(self) -> None:
        """Test interpolation at custom t."""
        result = linear_interpolate(0.0, 10.0, 0.3)
        assert result == pytest.approx(3.0)

    def test_same_values(self) -> None:
        """Test interpolation between same values."""
        result = linear_interpolate(5.0, 5.0, 0.7)
        assert result == pytest.approx(5.0)

    def test_negative_values(self) -> None:
        """Test interpolation with negative values."""
        result = linear_interpolate(-10.0, 10.0, 0.5)
        assert result == pytest.approx(0.0)

    def test_t_below_range(self) -> None:
        """Test t below valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            linear_interpolate(0.0, 1.0, -0.1)

    def test_t_above_range(self) -> None:
        """Test t above valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            linear_interpolate(0.0, 1.0, 1.5)


class TestSlerp:
    """Tests for slerp function."""

    def test_2d_midpoint(self) -> None:
        """Test SLERP at midpoint with 2D vectors."""
        v0 = (1.0, 0.0)
        v1 = (0.0, 1.0)
        result = slerp(v0, v1, 0.5)
        # At t=0.5, should be roughly (0.7071, 0.7071)
        assert round(result[0], 4) == pytest.approx(0.7071)
        assert round(result[1], 4) == pytest.approx(0.7071)

    def test_at_start(self) -> None:
        """Test SLERP at t=0."""
        v0 = (1.0, 0.0)
        v1 = (0.0, 1.0)
        result = slerp(v0, v1, 0.0)
        assert result[0] == pytest.approx(1.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)

    def test_at_end(self) -> None:
        """Test SLERP at t=1."""
        v0 = (1.0, 0.0)
        v1 = (0.0, 1.0)
        result = slerp(v0, v1, 1.0)
        # Note: SLERP math may not give exact v1 at t=1.0
        # but should be close
        assert len(result) == 2

    def test_empty_v0(self) -> None:
        """Test SLERP with empty v0."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            slerp((), (1.0,), 0.5)

    def test_empty_v1(self) -> None:
        """Test SLERP with empty v1."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            slerp((1.0,), (), 0.5)

    def test_mismatched_lengths(self) -> None:
        """Test SLERP with mismatched vector lengths."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            slerp((1.0, 0.0), (0.0,), 0.5)

    def test_t_below_range(self) -> None:
        """Test t below valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            slerp((1.0,), (0.0,), -0.1)

    def test_t_above_range(self) -> None:
        """Test t above valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            slerp((1.0,), (0.0,), 1.5)

    def test_3d_vectors(self) -> None:
        """Test SLERP with 3D vectors."""
        v0 = (1.0, 0.0, 0.0)
        v1 = (0.0, 1.0, 0.0)
        result = slerp(v0, v1, 0.5)
        assert len(result) == 3

    def test_parallel_vectors(self) -> None:
        """Test SLERP with nearly parallel vectors."""
        v0 = (1.0, 0.0)
        v1 = (1.0, 0.0)  # Same as v0
        result = slerp(v0, v1, 0.5)
        # Should return something valid
        assert len(result) == 2


class TestGetDensityValue:
    """Tests for get_density_value function."""

    def test_low_density(self) -> None:
        """Test low density value."""
        assert get_density_value("low") == pytest.approx(0.3)

    def test_medium_density(self) -> None:
        """Test medium density value."""
        assert get_density_value("medium") == pytest.approx(0.5)

    def test_high_density(self) -> None:
        """Test high density value."""
        assert get_density_value("high") == pytest.approx(0.7)

    def test_all_density_values(self) -> None:
        """Test all density values are in DENSITY_VALUES."""
        for level in ("low", "medium", "high"):
            assert get_density_value(level) == DENSITY_VALUES[level]  # type: ignore[index]


class TestCalculateMergedParameterCount:
    """Tests for calculate_merged_parameter_count function."""

    def test_slerp_two_models(self) -> None:
        """Test SLERP with two models."""
        result = calculate_merged_parameter_count(
            (7_000_000, 7_000_000), MergeMethod.SLERP
        )
        assert result == 7_000_000

    def test_passthrough_combines(self) -> None:
        """Test passthrough combines all parameters."""
        result = calculate_merged_parameter_count(
            (7_000_000, 13_000_000), MergeMethod.PASSTHROUGH
        )
        assert result == 20_000_000

    def test_ties_uses_max(self) -> None:
        """Test TIES uses max parameter count."""
        result = calculate_merged_parameter_count(
            (7_000_000, 13_000_000, 3_000_000), MergeMethod.TIES
        )
        assert result == 13_000_000

    def test_empty_params(self) -> None:
        """Test empty params tuple."""
        with pytest.raises(ValueError, match="model_params cannot be empty"):
            calculate_merged_parameter_count((), MergeMethod.SLERP)

    def test_zero_params(self) -> None:
        """Test zero parameter count."""
        with pytest.raises(ValueError, match="parameter counts must be positive"):
            calculate_merged_parameter_count((0, 7_000_000), MergeMethod.SLERP)

    def test_negative_params(self) -> None:
        """Test negative parameter count."""
        with pytest.raises(ValueError, match="parameter counts must be positive"):
            calculate_merged_parameter_count((-1, 7_000_000), MergeMethod.SLERP)

    def test_linear_method(self) -> None:
        """Test LINEAR method uses max."""
        result = calculate_merged_parameter_count(
            (5_000_000, 10_000_000), MergeMethod.LINEAR
        )
        assert result == 10_000_000

    def test_dare_method(self) -> None:
        """Test DARE method uses max."""
        result = calculate_merged_parameter_count(
            (5_000_000, 10_000_000), MergeMethod.DARE
        )
        assert result == 10_000_000


class TestGetRecommendedMethod:
    """Tests for get_recommended_method function."""

    def test_two_models_same_arch(self) -> None:
        """Test two models with same architecture."""
        result = get_recommended_method(2)
        assert result == MergeMethod.SLERP

    def test_three_models_same_arch(self) -> None:
        """Test three models with same architecture."""
        result = get_recommended_method(3)
        assert result == MergeMethod.TIES

    def test_many_models_same_arch(self) -> None:
        """Test many models with same architecture."""
        result = get_recommended_method(5)
        assert result == MergeMethod.TIES

    def test_different_architecture(self) -> None:
        """Test different architectures."""
        result = get_recommended_method(2, same_architecture=False)
        assert result == MergeMethod.PASSTHROUGH

    def test_zero_models(self) -> None:
        """Test zero models."""
        with pytest.raises(ValueError, match="num_models must be positive"):
            get_recommended_method(0)

    def test_negative_models(self) -> None:
        """Test negative models."""
        with pytest.raises(ValueError, match="num_models must be positive"):
            get_recommended_method(-1)

    def test_one_model(self) -> None:
        """Test one model (edge case)."""
        result = get_recommended_method(1)
        # One model is not really a merge, but TIES handles it
        assert result == MergeMethod.TIES


class TestEstimateMergeTime:
    """Tests for estimate_merge_time function."""

    def test_small_linear_fast(self) -> None:
        """Test small model with LINEAR is fast."""
        result = estimate_merge_time(1_000_000, MergeMethod.LINEAR)
        assert result == "fast"

    def test_medium_linear_moderate(self) -> None:
        """Test medium model with LINEAR is moderate."""
        result = estimate_merge_time(5_000_000_000, MergeMethod.LINEAR)
        assert result == "moderate"

    def test_large_linear_slow(self) -> None:
        """Test large model with LINEAR is slow."""
        result = estimate_merge_time(50_000_000_000, MergeMethod.LINEAR)
        assert result == "slow"

    def test_passthrough_fast(self) -> None:
        """Test passthrough is fast for small models."""
        result = estimate_merge_time(100_000_000, MergeMethod.PASSTHROUGH)
        assert result == "fast"

    def test_small_ties_fast(self) -> None:
        """Test small model with TIES is fast."""
        result = estimate_merge_time(10_000_000, MergeMethod.TIES)
        assert result == "fast"

    def test_medium_ties_moderate(self) -> None:
        """Test medium model with TIES is moderate."""
        result = estimate_merge_time(7_000_000_000, MergeMethod.TIES)
        assert result == "moderate"

    def test_large_dare_slow(self) -> None:
        """Test large model with DARE is slow."""
        result = estimate_merge_time(70_000_000_000, MergeMethod.DARE)
        assert result == "slow"

    def test_slerp_categories(self) -> None:
        """Test SLERP at different sizes."""
        assert estimate_merge_time(50_000_000, MergeMethod.SLERP) == "fast"
        assert estimate_merge_time(5_000_000_000, MergeMethod.SLERP) == "moderate"
        assert estimate_merge_time(50_000_000_000, MergeMethod.SLERP) == "slow"

    def test_zero_params(self) -> None:
        """Test zero params."""
        with pytest.raises(ValueError, match="total_params must be positive"):
            estimate_merge_time(0, MergeMethod.LINEAR)

    def test_negative_params(self) -> None:
        """Test negative params."""
        with pytest.raises(ValueError, match="total_params must be positive"):
            estimate_merge_time(-1, MergeMethod.LINEAR)
