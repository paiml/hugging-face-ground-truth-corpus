"""Tests for model merging training utilities."""

from __future__ import annotations

import pytest

from hf_gtc.training.merging import (
    VALID_CONFLICT_RESOLUTIONS,
    VALID_MERGE_METHODS,
    VALID_SPARSIFICATION_METHODS,
    ConflictResolution,
    DAREConfig,
    MergeConfig,
    MergeMethod,
    MergeStats,
    SLERPConfig,
    SparsificationMethod,
    TIESConfig,
    apply_task_vector,
    calculate_merge_weights,
    calculate_task_vector,
    create_dare_config,
    create_merge_config,
    create_merge_stats,
    create_slerp_config,
    create_ties_config,
    estimate_merged_performance,
    format_merge_stats,
    get_conflict_resolution,
    get_merge_method,
    get_recommended_merge_config,
    get_sparsification_method,
    linear_interpolate,
    list_conflict_resolutions,
    list_merge_methods,
    list_sparsification_methods,
    resolve_parameter_conflicts,
    slerp_interpolate,
    validate_dare_config,
    validate_merge_config,
    validate_slerp_config,
    validate_ties_config,
)


class TestMergeMethod:
    """Tests for MergeMethod enum."""

    def test_ties_value(self) -> None:
        """Test TIES method value."""
        assert MergeMethod.TIES.value == "ties"

    def test_dare_value(self) -> None:
        """Test DARE method value."""
        assert MergeMethod.DARE.value == "dare"

    def test_slerp_value(self) -> None:
        """Test SLERP method value."""
        assert MergeMethod.SLERP.value == "slerp"

    def test_task_arithmetic_value(self) -> None:
        """Test TASK_ARITHMETIC method value."""
        assert MergeMethod.TASK_ARITHMETIC.value == "task_arithmetic"

    def test_linear_value(self) -> None:
        """Test LINEAR method value."""
        assert MergeMethod.LINEAR.value == "linear"

    def test_stock_value(self) -> None:
        """Test STOCK method value."""
        assert MergeMethod.STOCK.value == "stock"

    def test_all_methods_in_valid_set(self) -> None:
        """Test all enum values are in VALID_MERGE_METHODS."""
        for method in MergeMethod:
            assert method.value in VALID_MERGE_METHODS


class TestSparsificationMethod:
    """Tests for SparsificationMethod enum."""

    def test_magnitude_value(self) -> None:
        """Test MAGNITUDE method value."""
        assert SparsificationMethod.MAGNITUDE.value == "magnitude"

    def test_random_value(self) -> None:
        """Test RANDOM method value."""
        assert SparsificationMethod.RANDOM.value == "random"

    def test_topk_value(self) -> None:
        """Test TOPK method value."""
        assert SparsificationMethod.TOPK.value == "topk"

    def test_all_methods_in_valid_set(self) -> None:
        """Test all enum values are in VALID_SPARSIFICATION_METHODS."""
        for method in SparsificationMethod:
            assert method.value in VALID_SPARSIFICATION_METHODS


class TestConflictResolution:
    """Tests for ConflictResolution enum."""

    def test_sum_value(self) -> None:
        """Test SUM resolution value."""
        assert ConflictResolution.SUM.value == "sum"

    def test_mean_value(self) -> None:
        """Test MEAN resolution value."""
        assert ConflictResolution.MEAN.value == "mean"

    def test_max_value(self) -> None:
        """Test MAX resolution value."""
        assert ConflictResolution.MAX.value == "max"

    def test_random_value(self) -> None:
        """Test RANDOM resolution value."""
        assert ConflictResolution.RANDOM.value == "random"

    def test_all_resolutions_in_valid_set(self) -> None:
        """Test all enum values are in VALID_CONFLICT_RESOLUTIONS."""
        for resolution in ConflictResolution:
            assert resolution.value in VALID_CONFLICT_RESOLUTIONS


class TestTIESConfig:
    """Tests for TIESConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating TIES config."""
        config = TIESConfig(
            density=0.5,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        assert config.density == 0.5
        assert config.normalize is True
        assert config.conflict_resolution == ConflictResolution.SUM

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = TIESConfig(
            density=0.5,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        with pytest.raises(AttributeError):
            config.density = 0.7  # type: ignore[misc]


class TestDAREConfig:
    """Tests for DAREConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating DARE config."""
        config = DAREConfig(drop_rate=0.1, rescale=True)
        assert config.drop_rate == 0.1
        assert config.rescale is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = DAREConfig(drop_rate=0.1, rescale=True)
        with pytest.raises(AttributeError):
            config.drop_rate = 0.2  # type: ignore[misc]


class TestSLERPConfig:
    """Tests for SLERPConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating SLERP config."""
        config = SLERPConfig(interpolation_factor=0.5)
        assert config.interpolation_factor == 0.5

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = SLERPConfig(interpolation_factor=0.5)
        with pytest.raises(AttributeError):
            config.interpolation_factor = 0.7  # type: ignore[misc]


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating merge config."""
        ties = TIESConfig(
            density=0.5,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_config=ties,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        assert config.method == MergeMethod.TIES
        assert config.ties_config is not None
        assert config.weights == (0.5, 0.5)

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        with pytest.raises(AttributeError):
            config.method = MergeMethod.TIES  # type: ignore[misc]


class TestMergeStats:
    """Tests for MergeStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating merge stats."""
        stats = MergeStats(
            num_models=3,
            total_parameters=7_000_000_000,
            retained_parameters=3_500_000_000,
            conflict_rate=0.15,
            method_used=MergeMethod.TIES,
        )
        assert stats.num_models == 3
        assert stats.total_parameters == 7_000_000_000
        assert stats.conflict_rate == 0.15

    def test_frozen(self) -> None:
        """Test stats is immutable."""
        stats = MergeStats(
            num_models=2,
            total_parameters=1000,
            retained_parameters=500,
            conflict_rate=0.1,
            method_used=MergeMethod.LINEAR,
        )
        with pytest.raises(AttributeError):
            stats.num_models = 3  # type: ignore[misc]


class TestValidateTIESConfig:
    """Tests for validate_ties_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = TIESConfig(
            density=0.5,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        validate_ties_config(config)  # Should not raise

    def test_density_at_boundaries(self) -> None:
        """Test density at boundary values."""
        config_low = TIESConfig(
            density=0.0,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        validate_ties_config(config_low)  # Should not raise

        config_high = TIESConfig(
            density=1.0,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        validate_ties_config(config_high)  # Should not raise

    def test_density_too_low(self) -> None:
        """Test density below valid range."""
        config = TIESConfig(
            density=-0.1,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        with pytest.raises(ValueError, match=r"density must be between 0\.0 and 1\.0"):
            validate_ties_config(config)

    def test_density_too_high(self) -> None:
        """Test density above valid range."""
        config = TIESConfig(
            density=1.5,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        with pytest.raises(ValueError, match=r"density must be between 0\.0 and 1\.0"):
            validate_ties_config(config)


class TestValidateDAREConfig:
    """Tests for validate_dare_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = DAREConfig(drop_rate=0.1, rescale=True)
        validate_dare_config(config)  # Should not raise

    def test_drop_rate_at_boundaries(self) -> None:
        """Test drop_rate at boundary values."""
        config_low = DAREConfig(drop_rate=0.0, rescale=True)
        validate_dare_config(config_low)  # Should not raise

        config_high = DAREConfig(drop_rate=1.0, rescale=True)
        validate_dare_config(config_high)  # Should not raise

    def test_drop_rate_too_low(self) -> None:
        """Test drop_rate below valid range."""
        config = DAREConfig(drop_rate=-0.1, rescale=True)
        with pytest.raises(
            ValueError, match=r"drop_rate must be between 0\.0 and 1\.0"
        ):
            validate_dare_config(config)

    def test_drop_rate_too_high(self) -> None:
        """Test drop_rate above valid range."""
        config = DAREConfig(drop_rate=1.5, rescale=True)
        with pytest.raises(
            ValueError, match=r"drop_rate must be between 0\.0 and 1\.0"
        ):
            validate_dare_config(config)


class TestValidateSLERPConfig:
    """Tests for validate_slerp_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = SLERPConfig(interpolation_factor=0.5)
        validate_slerp_config(config)  # Should not raise

    def test_factor_at_boundaries(self) -> None:
        """Test interpolation_factor at boundary values."""
        config_low = SLERPConfig(interpolation_factor=0.0)
        validate_slerp_config(config_low)  # Should not raise

        config_high = SLERPConfig(interpolation_factor=1.0)
        validate_slerp_config(config_high)  # Should not raise

    def test_factor_too_low(self) -> None:
        """Test interpolation_factor below valid range."""
        config = SLERPConfig(interpolation_factor=-0.5)
        with pytest.raises(
            ValueError, match=r"interpolation_factor must be between 0\.0 and 1\.0"
        ):
            validate_slerp_config(config)

    def test_factor_too_high(self) -> None:
        """Test interpolation_factor above valid range."""
        config = SLERPConfig(interpolation_factor=1.5)
        with pytest.raises(
            ValueError, match=r"interpolation_factor must be between 0\.0 and 1\.0"
        ):
            validate_slerp_config(config)


class TestValidateMergeConfig:
    """Tests for validate_merge_config function."""

    def test_valid_linear_config(self) -> None:
        """Test validating valid linear config."""
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        validate_merge_config(config)  # Should not raise

    def test_valid_ties_config(self) -> None:
        """Test validating valid TIES config."""
        ties = TIESConfig(
            density=0.5,
            normalize=True,
            conflict_resolution=ConflictResolution.SUM,
        )
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_config=ties,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        validate_merge_config(config)  # Should not raise

    def test_valid_dare_config(self) -> None:
        """Test validating valid DARE config."""
        dare = DAREConfig(drop_rate=0.1, rescale=True)
        config = MergeConfig(
            method=MergeMethod.DARE,
            ties_config=None,
            dare_config=dare,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        validate_merge_config(config)  # Should not raise

    def test_valid_slerp_config(self) -> None:
        """Test validating valid SLERP config."""
        slerp = SLERPConfig(interpolation_factor=0.5)
        config = MergeConfig(
            method=MergeMethod.SLERP,
            ties_config=None,
            dare_config=None,
            slerp_config=slerp,
            weights=(0.5, 0.5),
        )
        validate_merge_config(config)  # Should not raise

    def test_empty_weights(self) -> None:
        """Test empty weights."""
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(),
        )
        with pytest.raises(ValueError, match="weights cannot be empty"):
            validate_merge_config(config)

    def test_negative_weight(self) -> None:
        """Test negative weight."""
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, -0.1),
        )
        with pytest.raises(ValueError, match="weights must be non-negative"):
            validate_merge_config(config)

    def test_ties_without_ties_config(self) -> None:
        """Test TIES method without ties_config."""
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        with pytest.raises(ValueError, match="ties_config required"):
            validate_merge_config(config)

    def test_dare_without_dare_config(self) -> None:
        """Test DARE method without dare_config."""
        config = MergeConfig(
            method=MergeMethod.DARE,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        with pytest.raises(ValueError, match="dare_config required"):
            validate_merge_config(config)

    def test_slerp_without_slerp_config(self) -> None:
        """Test SLERP method without slerp_config."""
        config = MergeConfig(
            method=MergeMethod.SLERP,
            ties_config=None,
            dare_config=None,
            slerp_config=None,
            weights=(0.5, 0.5),
        )
        with pytest.raises(ValueError, match="slerp_config required"):
            validate_merge_config(config)

    def test_slerp_with_wrong_model_count(self) -> None:
        """Test SLERP with non-2 models."""
        slerp = SLERPConfig(interpolation_factor=0.5)
        config = MergeConfig(
            method=MergeMethod.SLERP,
            ties_config=None,
            dare_config=None,
            slerp_config=slerp,
            weights=(0.33, 0.33, 0.34),
        )
        with pytest.raises(ValueError, match="SLERP requires exactly 2 models"):
            validate_merge_config(config)


class TestCreateTIESConfig:
    """Tests for create_ties_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_ties_config()
        assert config.density == 0.5
        assert config.normalize is True
        assert config.conflict_resolution == ConflictResolution.SUM

    def test_custom_density(self) -> None:
        """Test custom density."""
        config = create_ties_config(density=0.7)
        assert config.density == 0.7

    def test_custom_conflict_resolution_string(self) -> None:
        """Test custom conflict resolution from string."""
        config = create_ties_config(conflict_resolution="mean")
        assert config.conflict_resolution == ConflictResolution.MEAN

    def test_custom_conflict_resolution_enum(self) -> None:
        """Test custom conflict resolution from enum."""
        config = create_ties_config(conflict_resolution=ConflictResolution.MAX)
        assert config.conflict_resolution == ConflictResolution.MAX

    def test_invalid_density(self) -> None:
        """Test invalid density."""
        with pytest.raises(ValueError, match=r"density must be between 0\.0 and 1\.0"):
            create_ties_config(density=1.5)


class TestCreateDAREConfig:
    """Tests for create_dare_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_dare_config()
        assert config.drop_rate == 0.1
        assert config.rescale is True

    def test_custom_drop_rate(self) -> None:
        """Test custom drop_rate."""
        config = create_dare_config(drop_rate=0.2)
        assert config.drop_rate == 0.2

    def test_rescale_false(self) -> None:
        """Test rescale=False."""
        config = create_dare_config(rescale=False)
        assert config.rescale is False

    def test_invalid_drop_rate(self) -> None:
        """Test invalid drop_rate."""
        with pytest.raises(
            ValueError, match=r"drop_rate must be between 0\.0 and 1\.0"
        ):
            create_dare_config(drop_rate=-0.1)


class TestCreateSLERPConfig:
    """Tests for create_slerp_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_slerp_config()
        assert config.interpolation_factor == 0.5

    def test_custom_factor(self) -> None:
        """Test custom interpolation factor."""
        config = create_slerp_config(interpolation_factor=0.7)
        assert config.interpolation_factor == 0.7

    def test_invalid_factor(self) -> None:
        """Test invalid interpolation factor."""
        with pytest.raises(
            ValueError, match=r"interpolation_factor must be between 0\.0 and 1\.0"
        ):
            create_slerp_config(interpolation_factor=1.5)


class TestCreateMergeConfig:
    """Tests for create_merge_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_merge_config()
        assert config.method == MergeMethod.LINEAR
        assert config.weights == (0.5, 0.5)

    def test_method_from_string(self) -> None:
        """Test method from string."""
        config = create_merge_config(method="ties")
        assert config.method == MergeMethod.TIES
        assert config.ties_config is not None

    def test_auto_create_ties_config(self) -> None:
        """Test auto-creation of ties_config."""
        config = create_merge_config(method="ties")
        assert config.ties_config is not None
        assert config.ties_config.density == 0.5

    def test_auto_create_dare_config(self) -> None:
        """Test auto-creation of dare_config."""
        config = create_merge_config(method="dare")
        assert config.dare_config is not None
        assert config.dare_config.drop_rate == 0.1

    def test_auto_create_slerp_config(self) -> None:
        """Test auto-creation of slerp_config."""
        config = create_merge_config(method="slerp")
        assert config.slerp_config is not None
        assert config.slerp_config.interpolation_factor == 0.5

    def test_custom_weights(self) -> None:
        """Test custom weights."""
        config = create_merge_config(weights=(0.3, 0.7))
        assert config.weights == (0.3, 0.7)

    def test_invalid_method(self) -> None:
        """Test invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_merge_config(method="invalid")

    def test_empty_weights(self) -> None:
        """Test empty weights."""
        with pytest.raises(ValueError, match="weights cannot be empty"):
            create_merge_config(weights=())


class TestCreateMergeStats:
    """Tests for create_merge_stats function."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = create_merge_stats()
        assert stats.num_models == 2
        assert stats.total_parameters == 0
        assert stats.conflict_rate == 0.0

    def test_custom_values(self) -> None:
        """Test custom values."""
        stats = create_merge_stats(
            num_models=3,
            total_parameters=7_000_000_000,
            method_used=MergeMethod.TIES,
        )
        assert stats.num_models == 3
        assert stats.total_parameters == 7_000_000_000
        assert stats.method_used == MergeMethod.TIES

    def test_zero_num_models(self) -> None:
        """Test zero num_models."""
        with pytest.raises(ValueError, match="num_models must be positive"):
            create_merge_stats(num_models=0)

    def test_negative_parameters(self) -> None:
        """Test negative total_parameters."""
        with pytest.raises(ValueError, match="total_parameters must be non-negative"):
            create_merge_stats(total_parameters=-1)

    def test_invalid_conflict_rate(self) -> None:
        """Test invalid conflict_rate."""
        with pytest.raises(
            ValueError, match=r"conflict_rate must be between 0\.0 and 1\.0"
        ):
            create_merge_stats(conflict_rate=1.5)


class TestListMergeMethods:
    """Tests for list_merge_methods function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        methods = list_merge_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test contains expected methods."""
        methods = list_merge_methods()
        assert "ties" in methods
        assert "dare" in methods
        assert "slerp" in methods
        assert "linear" in methods

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        methods = list_merge_methods()
        assert methods == sorted(methods)


class TestListSparsificationMethods:
    """Tests for list_sparsification_methods function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        methods = list_sparsification_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test contains expected methods."""
        methods = list_sparsification_methods()
        assert "magnitude" in methods
        assert "random" in methods
        assert "topk" in methods

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        methods = list_sparsification_methods()
        assert methods == sorted(methods)


class TestListConflictResolutions:
    """Tests for list_conflict_resolutions function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        resolutions = list_conflict_resolutions()
        assert isinstance(resolutions, list)

    def test_contains_expected_resolutions(self) -> None:
        """Test contains expected resolutions."""
        resolutions = list_conflict_resolutions()
        assert "sum" in resolutions
        assert "mean" in resolutions
        assert "max" in resolutions

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        resolutions = list_conflict_resolutions()
        assert resolutions == sorted(resolutions)


class TestGetMergeMethod:
    """Tests for get_merge_method function."""

    def test_ties(self) -> None:
        """Test getting TIES method."""
        assert get_merge_method("ties") == MergeMethod.TIES

    def test_slerp(self) -> None:
        """Test getting SLERP method."""
        assert get_merge_method("slerp") == MergeMethod.SLERP

    def test_dare(self) -> None:
        """Test getting DARE method."""
        assert get_merge_method("dare") == MergeMethod.DARE

    def test_linear(self) -> None:
        """Test getting LINEAR method."""
        assert get_merge_method("linear") == MergeMethod.LINEAR

    def test_invalid(self) -> None:
        """Test invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_merge_method("invalid")


class TestGetSparsificationMethod:
    """Tests for get_sparsification_method function."""

    def test_magnitude(self) -> None:
        """Test getting MAGNITUDE method."""
        assert get_sparsification_method("magnitude") == SparsificationMethod.MAGNITUDE

    def test_random(self) -> None:
        """Test getting RANDOM method."""
        assert get_sparsification_method("random") == SparsificationMethod.RANDOM

    def test_topk(self) -> None:
        """Test getting TOPK method."""
        assert get_sparsification_method("topk") == SparsificationMethod.TOPK

    def test_invalid(self) -> None:
        """Test invalid method."""
        with pytest.raises(ValueError, match="sparsification_method must be one of"):
            get_sparsification_method("invalid")


class TestGetConflictResolution:
    """Tests for get_conflict_resolution function."""

    def test_sum(self) -> None:
        """Test getting SUM resolution."""
        assert get_conflict_resolution("sum") == ConflictResolution.SUM

    def test_mean(self) -> None:
        """Test getting MEAN resolution."""
        assert get_conflict_resolution("mean") == ConflictResolution.MEAN

    def test_max(self) -> None:
        """Test getting MAX resolution."""
        assert get_conflict_resolution("max") == ConflictResolution.MAX

    def test_invalid(self) -> None:
        """Test invalid resolution."""
        with pytest.raises(ValueError, match="conflict_resolution must be one of"):
            get_conflict_resolution("invalid")


class TestCalculateMergeWeights:
    """Tests for calculate_merge_weights function."""

    def test_equal_scores(self) -> None:
        """Test equal scores produce equal weights."""
        weights = calculate_merge_weights((1.0, 1.0, 1.0))
        assert len(weights) == 3
        assert all(abs(w - 1 / 3) < 0.001 for w in weights)

    def test_different_scores(self) -> None:
        """Test different scores produce proportional weights."""
        weights = calculate_merge_weights((0.8, 0.9, 0.7))
        assert sum(weights) == pytest.approx(1.0)
        assert weights[1] > weights[0] > weights[2]

    def test_normalize_false(self) -> None:
        """Test normalize=False."""
        weights = calculate_merge_weights((1.0, 2.0), normalize=False)
        assert weights == (1.0, 2.0)

    def test_zero_scores(self) -> None:
        """Test all zero scores."""
        weights = calculate_merge_weights((0.0, 0.0))
        assert weights == (0.5, 0.5)

    def test_empty_scores(self) -> None:
        """Test empty scores."""
        with pytest.raises(ValueError, match="model_scores cannot be empty"):
            calculate_merge_weights(())

    def test_negative_score(self) -> None:
        """Test negative score."""
        with pytest.raises(ValueError, match="model_scores must be non-negative"):
            calculate_merge_weights((0.8, -0.1))


class TestEstimateMergedPerformance:
    """Tests for estimate_merged_performance function."""

    def test_basic_estimate(self) -> None:
        """Test basic performance estimate."""
        perf = estimate_merged_performance((0.8, 0.9), (0.5, 0.5))
        assert 0.8 <= perf <= 0.9

    def test_weighted_estimate(self) -> None:
        """Test weighted performance estimate."""
        perf = estimate_merged_performance((0.8, 0.9), (0.2, 0.8))
        assert perf > 0.85  # Weighted toward higher performer

    def test_ties_bonus(self) -> None:
        """Test TIES method bonus."""
        linear_perf = estimate_merged_performance(
            (0.7, 0.8, 0.9), (0.33, 0.33, 0.34), method=MergeMethod.LINEAR
        )
        ties_perf = estimate_merged_performance(
            (0.7, 0.8, 0.9), (0.33, 0.33, 0.34), method=MergeMethod.TIES
        )
        assert ties_perf > linear_perf

    def test_empty_performances(self) -> None:
        """Test empty performances."""
        with pytest.raises(ValueError, match="model_performances cannot be empty"):
            estimate_merged_performance((), ())

    def test_mismatched_lengths(self) -> None:
        """Test mismatched lengths."""
        with pytest.raises(
            ValueError, match="model_performances and weights must have same length"
        ):
            estimate_merged_performance((0.8, 0.9), (0.5, 0.3, 0.2))

    def test_invalid_performance(self) -> None:
        """Test invalid performance value."""
        with pytest.raises(
            ValueError, match=r"performance values must be between 0\.0 and 1\.0"
        ):
            estimate_merged_performance((0.8, 1.5), (0.5, 0.5))


class TestResolveParameterConflicts:
    """Tests for resolve_parameter_conflicts function."""

    def test_sum_resolution(self) -> None:
        """Test SUM resolution."""
        result = resolve_parameter_conflicts(
            (0.5, -0.3, 0.2), (1, -1, 1), resolution=ConflictResolution.SUM
        )
        assert result == pytest.approx(0.4)

    def test_mean_resolution(self) -> None:
        """Test MEAN resolution."""
        result = resolve_parameter_conflicts(
            (0.6, 0.3, 0.0), (1, 1, 0), resolution=ConflictResolution.MEAN
        )
        assert result == pytest.approx(0.3)

    def test_max_resolution(self) -> None:
        """Test MAX resolution."""
        result = resolve_parameter_conflicts(
            (0.5, -0.8, 0.2), (1, -1, 1), resolution=ConflictResolution.MAX
        )
        assert result == -0.8  # Largest magnitude

    def test_random_resolution(self) -> None:
        """Test RANDOM resolution."""
        result = resolve_parameter_conflicts(
            (0.5, -0.3, 0.2), (1, -1, 1), resolution=ConflictResolution.RANDOM
        )
        assert result in (0.5, -0.3, 0.2)

    def test_empty_values(self) -> None:
        """Test empty values."""
        with pytest.raises(ValueError, match="values cannot be empty"):
            resolve_parameter_conflicts((), ())

    def test_mismatched_lengths(self) -> None:
        """Test mismatched lengths."""
        with pytest.raises(ValueError, match="values and signs must have same length"):
            resolve_parameter_conflicts((0.5, 0.3), (1,))

    def test_invalid_sign(self) -> None:
        """Test invalid sign value."""
        with pytest.raises(ValueError, match="signs must be -1, 0, or 1"):
            resolve_parameter_conflicts((0.5, 0.3), (1, 2))


class TestCalculateTaskVector:
    """Tests for calculate_task_vector function."""

    def test_basic_task_vector(self) -> None:
        """Test basic task vector calculation."""
        fine_tuned = (1.5, 2.5, 3.5)
        base = (1.0, 2.0, 3.0)
        task_vec = calculate_task_vector(fine_tuned, base)
        assert task_vec == (0.5, 0.5, 0.5)

    def test_negative_deltas(self) -> None:
        """Test negative deltas."""
        fine_tuned = (0.5, 1.5, 2.5)
        base = (1.0, 2.0, 3.0)
        task_vec = calculate_task_vector(fine_tuned, base)
        assert task_vec == (-0.5, -0.5, -0.5)

    def test_empty_params(self) -> None:
        """Test empty parameters."""
        with pytest.raises(ValueError, match="parameters cannot be empty"):
            calculate_task_vector((), ())

    def test_mismatched_lengths(self) -> None:
        """Test mismatched lengths."""
        with pytest.raises(
            ValueError, match="fine_tuned and base parameters must have same length"
        ):
            calculate_task_vector((1.0, 2.0), (1.0,))


class TestApplyTaskVector:
    """Tests for apply_task_vector function."""

    def test_basic_apply(self) -> None:
        """Test basic task vector application."""
        base = (1.0, 2.0, 3.0)
        task_vec = (0.5, 0.5, 0.5)
        result = apply_task_vector(base, task_vec)
        assert result == (1.5, 2.5, 3.5)

    def test_scaled_apply(self) -> None:
        """Test scaled task vector application."""
        base = (1.0, 2.0, 3.0)
        task_vec = (0.5, 0.5, 0.5)
        result = apply_task_vector(base, task_vec, scaling_factor=0.5)
        assert result == (1.25, 2.25, 3.25)

    def test_negative_scaling(self) -> None:
        """Test negative scaling factor."""
        base = (1.0, 2.0, 3.0)
        task_vec = (0.5, 0.5, 0.5)
        result = apply_task_vector(base, task_vec, scaling_factor=-1.0)
        assert result == (0.5, 1.5, 2.5)

    def test_empty_params(self) -> None:
        """Test empty parameters."""
        with pytest.raises(ValueError, match="parameters cannot be empty"):
            apply_task_vector((), ())

    def test_mismatched_lengths(self) -> None:
        """Test mismatched lengths."""
        with pytest.raises(
            ValueError, match="base_params and task_vector must have same length"
        ):
            apply_task_vector((1.0, 2.0), (0.5,))


class TestFormatMergeStats:
    """Tests for format_merge_stats function."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        stats = create_merge_stats(
            num_models=3,
            total_parameters=7_000_000_000,
            retained_parameters=3_500_000_000,
            conflict_rate=0.15,
            method_used=MergeMethod.TIES,
        )
        formatted = format_merge_stats(stats)
        assert "Models Merged: 3" in formatted
        assert "Conflict Rate: 15.0%" in formatted
        assert "ties" in formatted

    def test_zero_parameters(self) -> None:
        """Test formatting with zero parameters."""
        stats = create_merge_stats(
            num_models=2, total_parameters=0, retained_parameters=0
        )
        formatted = format_merge_stats(stats)
        assert "Retention Rate: 0.0%" in formatted


class TestGetRecommendedMergeConfig:
    """Tests for get_recommended_merge_config function."""

    def test_two_models(self) -> None:
        """Test recommendation for 2 models."""
        config = get_recommended_merge_config(2)
        assert config.method == MergeMethod.SLERP

    def test_three_models(self) -> None:
        """Test recommendation for 3 models."""
        config = get_recommended_merge_config(3)
        assert config.method == MergeMethod.TIES

    def test_classification_task(self) -> None:
        """Test recommendation for classification."""
        config = get_recommended_merge_config(3, task_type="classification")
        assert config.method == MergeMethod.TIES
        assert config.ties_config is not None
        assert config.ties_config.density == 0.5

    def test_generation_task(self) -> None:
        """Test recommendation for generation."""
        config = get_recommended_merge_config(3, task_type="generation")
        assert config.method == MergeMethod.TIES
        assert config.ties_config is not None
        assert config.ties_config.density == 0.7

    def test_too_few_models(self) -> None:
        """Test too few models."""
        with pytest.raises(ValueError, match="num_models must be at least 2"):
            get_recommended_merge_config(1)

    def test_invalid_task_type(self) -> None:
        """Test invalid task type."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_merge_config(2, task_type="invalid")


class TestSlerpInterpolate:
    """Tests for slerp_interpolate function."""

    def test_2d_midpoint(self) -> None:
        """Test SLERP at midpoint with 2D vectors."""
        v0 = (1.0, 0.0)
        v1 = (0.0, 1.0)
        result = slerp_interpolate(v0, v1, 0.5)
        assert round(result[0], 4) == 0.7071
        assert round(result[1], 4) == 0.7071

    def test_at_start(self) -> None:
        """Test SLERP at t=0."""
        v0 = (1.0, 0.0)
        v1 = (0.0, 1.0)
        result = slerp_interpolate(v0, v1, 0.0)
        assert result == v0

    def test_at_end(self) -> None:
        """Test SLERP at t=1."""
        v0 = (1.0, 0.0)
        v1 = (0.0, 1.0)
        result = slerp_interpolate(v0, v1, 1.0)
        assert result == v1

    def test_3d_vectors(self) -> None:
        """Test SLERP with 3D vectors."""
        v0 = (1.0, 0.0, 0.0)
        v1 = (0.0, 1.0, 0.0)
        result = slerp_interpolate(v0, v1, 0.5)
        assert len(result) == 3

    def test_empty_vectors(self) -> None:
        """Test SLERP with empty vectors."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            slerp_interpolate((), (1.0,), 0.5)

    def test_mismatched_lengths(self) -> None:
        """Test SLERP with mismatched vector lengths."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            slerp_interpolate((1.0, 0.0), (0.0,), 0.5)

    def test_t_below_range(self) -> None:
        """Test t below valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            slerp_interpolate((1.0,), (0.0,), -0.1)

    def test_t_above_range(self) -> None:
        """Test t above valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            slerp_interpolate((1.0,), (0.0,), 1.5)

    def test_parallel_vectors(self) -> None:
        """Test SLERP with parallel vectors."""
        v0 = (1.0, 0.0)
        v1 = (1.0, 0.0)
        result = slerp_interpolate(v0, v1, 0.5)
        assert len(result) == 2

    def test_zero_vector(self) -> None:
        """Test SLERP with zero vector."""
        v0 = (0.0, 0.0)
        v1 = (1.0, 1.0)
        result = slerp_interpolate(v0, v1, 0.5)
        # Should fall back to linear interpolation
        assert result == (0.5, 0.5)


class TestLinearInterpolate:
    """Tests for linear_interpolate function."""

    def test_midpoint(self) -> None:
        """Test interpolation at midpoint."""
        v0 = (0.0, 0.0)
        v1 = (1.0, 2.0)
        result = linear_interpolate(v0, v1, 0.5)
        assert result == (0.5, 1.0)

    def test_at_start(self) -> None:
        """Test interpolation at t=0."""
        v0 = (1.0, 2.0)
        v1 = (3.0, 4.0)
        result = linear_interpolate(v0, v1, 0.0)
        assert result == v0

    def test_at_end(self) -> None:
        """Test interpolation at t=1."""
        v0 = (1.0, 2.0)
        v1 = (3.0, 4.0)
        result = linear_interpolate(v0, v1, 1.0)
        assert result == v1

    def test_custom_t(self) -> None:
        """Test interpolation at custom t."""
        v0 = (0.0,)
        v1 = (10.0,)
        result = linear_interpolate(v0, v1, 0.3)
        assert result == (3.0,)

    def test_empty_vectors(self) -> None:
        """Test interpolation with empty vectors."""
        with pytest.raises(ValueError, match="vectors cannot be empty"):
            linear_interpolate((), (1.0,), 0.5)

    def test_mismatched_lengths(self) -> None:
        """Test interpolation with mismatched lengths."""
        with pytest.raises(ValueError, match="vectors must have same length"):
            linear_interpolate((1.0, 0.0), (0.0,), 0.5)

    def test_t_below_range(self) -> None:
        """Test t below valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            linear_interpolate((1.0,), (0.0,), -0.1)

    def test_t_above_range(self) -> None:
        """Test t above valid range."""
        with pytest.raises(ValueError, match=r"t must be between 0\.0 and 1\.0"):
            linear_interpolate((1.0,), (0.0,), 1.5)
