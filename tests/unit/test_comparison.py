"""Tests for model comparison and statistical testing functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.comparison import (
    VALID_COMPARISON_METHODS,
    VALID_EFFECT_SIZES,
    VALID_SIGNIFICANCE_LEVELS,
    ComparisonConfig,
    ComparisonMethod,
    ComparisonResult,
    ComparisonStats,
    EffectSize,
    ModelResult,
    SignificanceLevel,
    bootstrap_confidence_interval,
    calculate_effect_size,
    compare_models,
    create_comparison_config,
    create_comparison_result,
    create_model_result,
    format_comparison_stats,
    format_comparison_table,
    get_comparison_method,
    get_effect_size_type,
    get_recommended_comparison_config,
    get_significance_level,
    list_comparison_methods,
    list_effect_sizes,
    list_significance_levels,
    run_significance_test,
    validate_comparison_config,
    validate_comparison_method,
    validate_comparison_result,
    validate_comparison_stats,
    validate_effect_size_type,
    validate_model_result,
    validate_significance_level,
)


class TestComparisonMethod:
    """Tests for ComparisonMethod enum."""

    def test_paired_ttest_value(self) -> None:
        """Test PAIRED_TTEST value."""
        assert ComparisonMethod.PAIRED_TTEST.value == "paired_ttest"

    def test_bootstrap_value(self) -> None:
        """Test BOOTSTRAP value."""
        assert ComparisonMethod.BOOTSTRAP.value == "bootstrap"

    def test_permutation_value(self) -> None:
        """Test PERMUTATION value."""
        assert ComparisonMethod.PERMUTATION.value == "permutation"

    def test_wilcoxon_value(self) -> None:
        """Test WILCOXON value."""
        assert ComparisonMethod.WILCOXON.value == "wilcoxon"

    def test_mcnemar_value(self) -> None:
        """Test MCNEMAR value."""
        assert ComparisonMethod.MCNEMAR.value == "mcnemar"


class TestSignificanceLevel:
    """Tests for SignificanceLevel enum."""

    def test_p01_value(self) -> None:
        """Test P01 value."""
        assert SignificanceLevel.P01.value == pytest.approx(0.01)

    def test_p05_value(self) -> None:
        """Test P05 value."""
        assert SignificanceLevel.P05.value == pytest.approx(0.05)

    def test_p10_value(self) -> None:
        """Test P10 value."""
        assert SignificanceLevel.P10.value == pytest.approx(0.10)


class TestEffectSize:
    """Tests for EffectSize enum."""

    def test_cohens_d_value(self) -> None:
        """Test COHENS_D value."""
        assert EffectSize.COHENS_D.value == "cohens_d"

    def test_hedges_g_value(self) -> None:
        """Test HEDGES_G value."""
        assert EffectSize.HEDGES_G.value == "hedges_g"

    def test_glass_delta_value(self) -> None:
        """Test GLASS_DELTA value."""
        assert EffectSize.GLASS_DELTA.value == "glass_delta"


class TestComparisonConfig:
    """Tests for ComparisonConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ComparisonConfig instance."""
        config = ComparisonConfig()
        assert config.method == ComparisonMethod.PAIRED_TTEST
        assert config.significance_level == SignificanceLevel.P05
        assert config.num_bootstrap == 1000

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ComparisonConfig(
            method=ComparisonMethod.BOOTSTRAP,
            significance_level=SignificanceLevel.P01,
            num_bootstrap=5000,
            effect_size_type=EffectSize.HEDGES_G,
            random_seed=42,
        )
        assert config.method == ComparisonMethod.BOOTSTRAP
        assert config.significance_level == SignificanceLevel.P01
        assert config.num_bootstrap == 5000
        assert config.random_seed == 42

    def test_frozen(self) -> None:
        """Test that ComparisonConfig is immutable."""
        config = ComparisonConfig()
        with pytest.raises(AttributeError):
            config.num_bootstrap = 2000  # type: ignore[misc]


class TestModelResult:
    """Tests for ModelResult dataclass."""

    def test_creation(self) -> None:
        """Test creating ModelResult instance."""
        result = ModelResult(
            model_name="model_a",
            scores=(0.8, 0.9, 0.85),
        )
        assert result.model_name == "model_a"
        assert len(result.scores) == 3

    def test_with_metadata(self) -> None:
        """Test creating ModelResult with metadata."""
        result = ModelResult(
            model_name="model_a",
            scores=(0.8,),
            metadata={"version": "1.0"},
        )
        assert result.metadata["version"] == "1.0"

    def test_frozen(self) -> None:
        """Test that ModelResult is immutable."""
        result = ModelResult(model_name="model", scores=(0.5,))
        with pytest.raises(AttributeError):
            result.model_name = "new_name"  # type: ignore[misc]


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_creation(self) -> None:
        """Test creating ComparisonResult instance."""
        result = ComparisonResult(
            model_a="model_1",
            model_b="model_2",
            p_value=0.03,
            effect_size=0.5,
            is_significant=True,
            confidence_interval=(0.1, 0.9),
            test_statistic=2.5,
            method=ComparisonMethod.PAIRED_TTEST,
        )
        assert result.model_a == "model_1"
        assert result.is_significant is True
        assert result.p_value == pytest.approx(0.03)

    def test_frozen(self) -> None:
        """Test that ComparisonResult is immutable."""
        result = ComparisonResult(
            model_a="a",
            model_b="b",
            p_value=0.05,
            effect_size=0.3,
            is_significant=True,
            confidence_interval=(0.1, 0.5),
            test_statistic=2.0,
            method=ComparisonMethod.PAIRED_TTEST,
        )
        with pytest.raises(AttributeError):
            result.p_value = 0.1  # type: ignore[misc]


class TestComparisonStats:
    """Tests for ComparisonStats dataclass."""

    def test_creation(self) -> None:
        """Test creating ComparisonStats instance."""
        stats = ComparisonStats(
            total_comparisons=10,
            significant_count=3,
            avg_effect_size=0.4,
            max_effect_size=0.8,
            min_p_value=0.001,
        )
        assert stats.total_comparisons == 10
        assert stats.significant_count == 3

    def test_frozen(self) -> None:
        """Test that ComparisonStats is immutable."""
        stats = ComparisonStats(
            total_comparisons=10,
            significant_count=3,
            avg_effect_size=0.4,
            max_effect_size=0.8,
            min_p_value=0.001,
        )
        with pytest.raises(AttributeError):
            stats.total_comparisons = 20  # type: ignore[misc]


class TestCreateComparisonConfig:
    """Tests for create_comparison_config function."""

    def test_creates_default_config(self) -> None:
        """Test creating config with defaults."""
        config = create_comparison_config()
        assert isinstance(config, ComparisonConfig)
        assert config.method == ComparisonMethod.PAIRED_TTEST

    def test_creates_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = create_comparison_config(
            method=ComparisonMethod.BOOTSTRAP,
            num_bootstrap=5000,
        )
        assert config.method == ComparisonMethod.BOOTSTRAP
        assert config.num_bootstrap == 5000

    def test_invalid_bootstrap_raises_error(self) -> None:
        """Test that invalid num_bootstrap raises ValueError."""
        with pytest.raises(ValueError, match="num_bootstrap must be positive"):
            create_comparison_config(num_bootstrap=0)

    def test_negative_bootstrap_raises_error(self) -> None:
        """Test that negative num_bootstrap raises ValueError."""
        with pytest.raises(ValueError, match="num_bootstrap must be positive"):
            create_comparison_config(num_bootstrap=-100)


class TestCreateModelResult:
    """Tests for create_model_result function."""

    def test_creates_result(self) -> None:
        """Test creating model result."""
        result = create_model_result("model_a", [0.8, 0.9, 0.85])
        assert isinstance(result, ModelResult)
        assert result.model_name == "model_a"

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_model_result("", [0.5])

    def test_empty_scores_raises_error(self) -> None:
        """Test that empty scores raises ValueError."""
        with pytest.raises(ValueError, match="scores cannot be empty"):
            create_model_result("model", [])

    def test_with_metadata(self) -> None:
        """Test creating result with metadata."""
        result = create_model_result("model", [0.5], {"key": "value"})
        assert result.metadata["key"] == "value"


class TestCreateComparisonResult:
    """Tests for create_comparison_result function."""

    def test_creates_result(self) -> None:
        """Test creating comparison result."""
        result = create_comparison_result(
            model_a="a",
            model_b="b",
            p_value=0.05,
            effect_size=0.3,
            is_significant=True,
            confidence_interval=(0.1, 0.5),
            test_statistic=2.0,
            method=ComparisonMethod.PAIRED_TTEST,
        )
        assert isinstance(result, ComparisonResult)

    def test_empty_model_a_raises_error(self) -> None:
        """Test that empty model_a raises ValueError."""
        with pytest.raises(ValueError, match="model_a cannot be empty"):
            create_comparison_result(
                model_a="",
                model_b="b",
                p_value=0.05,
                effect_size=0.3,
                is_significant=True,
                confidence_interval=(0.1, 0.5),
                test_statistic=2.0,
                method=ComparisonMethod.PAIRED_TTEST,
            )

    def test_empty_model_b_raises_error(self) -> None:
        """Test that empty model_b raises ValueError."""
        with pytest.raises(ValueError, match="model_b cannot be empty"):
            create_comparison_result(
                model_a="a",
                model_b="",
                p_value=0.05,
                effect_size=0.3,
                is_significant=True,
                confidence_interval=(0.1, 0.5),
                test_statistic=2.0,
                method=ComparisonMethod.PAIRED_TTEST,
            )

    def test_invalid_p_value_raises_error(self) -> None:
        """Test that invalid p_value raises ValueError."""
        with pytest.raises(ValueError, match="p_value must be in"):
            create_comparison_result(
                model_a="a",
                model_b="b",
                p_value=1.5,
                effect_size=0.3,
                is_significant=True,
                confidence_interval=(0.1, 0.5),
                test_statistic=2.0,
                method=ComparisonMethod.PAIRED_TTEST,
            )


class TestValidateComparisonConfig:
    """Tests for validate_comparison_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ComparisonConfig()
        validate_comparison_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_comparison_config(None)  # type: ignore[arg-type]

    def test_zero_bootstrap_raises_error(self) -> None:
        """Test that zero bootstrap raises ValueError."""
        config = ComparisonConfig(num_bootstrap=0)
        with pytest.raises(ValueError, match="num_bootstrap must be positive"):
            validate_comparison_config(config)


class TestValidateModelResult:
    """Tests for validate_model_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = ModelResult(model_name="model", scores=(0.5,))
        validate_model_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_model_result(None)  # type: ignore[arg-type]

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        result = ModelResult(model_name="", scores=(0.5,))
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            validate_model_result(result)

    def test_empty_scores_raises_error(self) -> None:
        """Test that empty scores raises ValueError."""
        result = ModelResult(model_name="model", scores=())
        with pytest.raises(ValueError, match="scores cannot be empty"):
            validate_model_result(result)


class TestValidateComparisonResult:
    """Tests for validate_comparison_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = ComparisonResult(
            model_a="a",
            model_b="b",
            p_value=0.05,
            effect_size=0.3,
            is_significant=True,
            confidence_interval=(0.1, 0.5),
            test_statistic=2.0,
            method=ComparisonMethod.PAIRED_TTEST,
        )
        validate_comparison_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_comparison_result(None)  # type: ignore[arg-type]

    def test_negative_p_value_raises_error(self) -> None:
        """Test that negative p_value raises ValueError."""
        result = ComparisonResult(
            model_a="a",
            model_b="b",
            p_value=-0.1,
            effect_size=0.3,
            is_significant=True,
            confidence_interval=(0.1, 0.5),
            test_statistic=2.0,
            method=ComparisonMethod.PAIRED_TTEST,
        )
        with pytest.raises(ValueError, match="p_value must be in"):
            validate_comparison_result(result)


class TestValidateComparisonStats:
    """Tests for validate_comparison_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = ComparisonStats(
            total_comparisons=10,
            significant_count=3,
            avg_effect_size=0.4,
            max_effect_size=0.8,
            min_p_value=0.001,
        )
        validate_comparison_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_comparison_stats(None)  # type: ignore[arg-type]

    def test_zero_comparisons_raises_error(self) -> None:
        """Test that zero total_comparisons raises ValueError."""
        stats = ComparisonStats(
            total_comparisons=0,
            significant_count=0,
            avg_effect_size=0.0,
            max_effect_size=0.0,
            min_p_value=1.0,
        )
        with pytest.raises(ValueError, match="total_comparisons must be positive"):
            validate_comparison_stats(stats)

    def test_significant_exceeds_total_raises_error(self) -> None:
        """Test that significant_count > total raises ValueError."""
        stats = ComparisonStats(
            total_comparisons=5,
            significant_count=10,
            avg_effect_size=0.4,
            max_effect_size=0.8,
            min_p_value=0.001,
        )
        with pytest.raises(ValueError, match=r"significant_count.*cannot exceed"):
            validate_comparison_stats(stats)

    def test_negative_significant_raises_error(self) -> None:
        """Test that negative significant_count raises ValueError."""
        stats = ComparisonStats(
            total_comparisons=10,
            significant_count=-1,
            avg_effect_size=0.4,
            max_effect_size=0.8,
            min_p_value=0.001,
        )
        with pytest.raises(ValueError, match="significant_count cannot be negative"):
            validate_comparison_stats(stats)


class TestListComparisonMethods:
    """Tests for list_comparison_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_comparison_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_comparison_methods()
        assert "paired_ttest" in methods
        assert "bootstrap" in methods
        assert "wilcoxon" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_comparison_methods()
        assert methods == sorted(methods)


class TestValidateComparisonMethod:
    """Tests for validate_comparison_method function."""

    def test_valid_paired_ttest(self) -> None:
        """Test validation of paired_ttest."""
        assert validate_comparison_method("paired_ttest") is True

    def test_valid_bootstrap(self) -> None:
        """Test validation of bootstrap."""
        assert validate_comparison_method("bootstrap") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_comparison_method("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_comparison_method("") is False


class TestGetComparisonMethod:
    """Tests for get_comparison_method function."""

    def test_get_paired_ttest(self) -> None:
        """Test getting PAIRED_TTEST."""
        result = get_comparison_method("paired_ttest")
        assert result == ComparisonMethod.PAIRED_TTEST

    def test_get_bootstrap(self) -> None:
        """Test getting BOOTSTRAP."""
        result = get_comparison_method("bootstrap")
        assert result == ComparisonMethod.BOOTSTRAP

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid comparison method"):
            get_comparison_method("invalid")


class TestListSignificanceLevels:
    """Tests for list_significance_levels function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        levels = list_significance_levels()
        assert isinstance(levels, list)

    def test_contains_expected_levels(self) -> None:
        """Test that list contains expected levels."""
        levels = list_significance_levels()
        assert 0.01 in levels
        assert 0.05 in levels
        assert 0.10 in levels

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        levels = list_significance_levels()
        assert levels == sorted(levels)


class TestValidateSignificanceLevel:
    """Tests for validate_significance_level function."""

    def test_valid_p05(self) -> None:
        """Test validation of 0.05."""
        assert validate_significance_level(0.05) is True

    def test_valid_p01(self) -> None:
        """Test validation of 0.01."""
        assert validate_significance_level(0.01) is True

    def test_invalid_level(self) -> None:
        """Test validation of invalid level."""
        assert validate_significance_level(0.03) is False


class TestGetSignificanceLevel:
    """Tests for get_significance_level function."""

    def test_get_p05(self) -> None:
        """Test getting P05."""
        result = get_significance_level(0.05)
        assert result == SignificanceLevel.P05

    def test_get_p01(self) -> None:
        """Test getting P01."""
        result = get_significance_level(0.01)
        assert result == SignificanceLevel.P01

    def test_invalid_level_raises_error(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="invalid significance level"):
            get_significance_level(0.03)


class TestListEffectSizes:
    """Tests for list_effect_sizes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        sizes = list_effect_sizes()
        assert isinstance(sizes, list)

    def test_contains_expected_sizes(self) -> None:
        """Test that list contains expected sizes."""
        sizes = list_effect_sizes()
        assert "cohens_d" in sizes
        assert "hedges_g" in sizes
        assert "glass_delta" in sizes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        sizes = list_effect_sizes()
        assert sizes == sorted(sizes)


class TestValidateEffectSizeType:
    """Tests for validate_effect_size_type function."""

    def test_valid_cohens_d(self) -> None:
        """Test validation of cohens_d."""
        assert validate_effect_size_type("cohens_d") is True

    def test_valid_hedges_g(self) -> None:
        """Test validation of hedges_g."""
        assert validate_effect_size_type("hedges_g") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_effect_size_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_effect_size_type("") is False


class TestGetEffectSizeType:
    """Tests for get_effect_size_type function."""

    def test_get_cohens_d(self) -> None:
        """Test getting COHENS_D."""
        result = get_effect_size_type("cohens_d")
        assert result == EffectSize.COHENS_D

    def test_get_hedges_g(self) -> None:
        """Test getting HEDGES_G."""
        result = get_effect_size_type("hedges_g")
        assert result == EffectSize.HEDGES_G

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid effect size type"):
            get_effect_size_type("invalid")


class TestCalculateEffectSize:
    """Tests for calculate_effect_size function."""

    def test_cohens_d_basic(self) -> None:
        """Test basic Cohen's d calculation."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        effect = calculate_effect_size(scores_a, scores_b)
        assert effect > 0  # A is better than B

    def test_hedges_g(self) -> None:
        """Test Hedges' g calculation."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        effect = calculate_effect_size(scores_a, scores_b, EffectSize.HEDGES_G)
        assert effect > 0

    def test_glass_delta(self) -> None:
        """Test Glass's delta calculation."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        effect = calculate_effect_size(scores_a, scores_b, EffectSize.GLASS_DELTA)
        assert effect > 0

    def test_equal_scores(self) -> None:
        """Test with equal scores."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        effect = calculate_effect_size(scores, scores)
        assert effect == pytest.approx(0.0)

    def test_none_scores_a_raises_error(self) -> None:
        """Test that None scores_a raises ValueError."""
        with pytest.raises(ValueError, match="scores_a cannot be None"):
            calculate_effect_size(None, [0.5])  # type: ignore[arg-type]

    def test_empty_scores_a_raises_error(self) -> None:
        """Test that empty scores_a raises ValueError."""
        with pytest.raises(ValueError, match="scores_a cannot be empty"):
            calculate_effect_size([], [0.5])

    def test_none_scores_b_raises_error(self) -> None:
        """Test that None scores_b raises ValueError."""
        with pytest.raises(ValueError, match="scores_b cannot be None"):
            calculate_effect_size([0.5], None)  # type: ignore[arg-type]

    def test_empty_scores_b_raises_error(self) -> None:
        """Test that empty scores_b raises ValueError."""
        with pytest.raises(ValueError, match="scores_b cannot be empty"):
            calculate_effect_size([0.5], [])

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_effect_size([0.5, 0.6], [0.5])

    def test_zero_variance_different_means(self) -> None:
        """Test handling of zero variance with different means."""
        scores_a = [0.8, 0.8, 0.8, 0.8]
        scores_b = [0.5, 0.5, 0.5, 0.5]
        effect = calculate_effect_size(scores_a, scores_b)
        assert effect == float("inf") or effect > 1000


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval function."""

    def test_basic_bootstrap(self) -> None:
        """Test basic bootstrap CI."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        lower, upper = bootstrap_confidence_interval(
            scores_a, scores_b, num_bootstrap=100, random_seed=42
        )
        assert lower < upper

    def test_reproducibility_with_seed(self) -> None:
        """Test that results are reproducible with seed."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        ci1 = bootstrap_confidence_interval(
            scores_a, scores_b, num_bootstrap=100, random_seed=42
        )
        ci2 = bootstrap_confidence_interval(
            scores_a, scores_b, num_bootstrap=100, random_seed=42
        )
        assert ci1 == ci2

    def test_none_scores_a_raises_error(self) -> None:
        """Test that None scores_a raises ValueError."""
        with pytest.raises(ValueError, match="scores_a cannot be None"):
            bootstrap_confidence_interval(None, [0.5])  # type: ignore[arg-type]

    def test_empty_scores_a_raises_error(self) -> None:
        """Test that empty scores_a raises ValueError."""
        with pytest.raises(ValueError, match="scores_a cannot be empty"):
            bootstrap_confidence_interval([], [0.5])

    def test_invalid_num_bootstrap_raises_error(self) -> None:
        """Test that invalid num_bootstrap raises ValueError."""
        with pytest.raises(ValueError, match="num_bootstrap must be positive"):
            bootstrap_confidence_interval([0.5], [0.5], num_bootstrap=0)

    def test_invalid_confidence_level_raises_error(self) -> None:
        """Test that invalid confidence_level raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            bootstrap_confidence_interval([0.5], [0.5], confidence_level=1.5)


class TestRunSignificanceTest:
    """Tests for run_significance_test function."""

    def test_paired_ttest(self) -> None:
        """Test paired t-test."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        config = ComparisonConfig(method=ComparisonMethod.PAIRED_TTEST)
        _stat, pval = run_significance_test(scores_a, scores_b, config)
        assert pval < 1.0
        assert pval >= 0.0

    def test_wilcoxon(self) -> None:
        """Test Wilcoxon test."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.83, 0.89, 0.84, 0.86]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71, 0.73, 0.69, 0.74, 0.70, 0.72]
        config = ComparisonConfig(method=ComparisonMethod.WILCOXON)
        _stat, pval = run_significance_test(scores_a, scores_b, config)
        assert pval < 1.0
        assert pval >= 0.0

    def test_permutation(self) -> None:
        """Test permutation test."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        config = ComparisonConfig(
            method=ComparisonMethod.PERMUTATION, num_bootstrap=100, random_seed=42
        )
        _stat, pval = run_significance_test(scores_a, scores_b, config)
        assert pval >= 0.0
        assert pval <= 1.0

    def test_bootstrap(self) -> None:
        """Test bootstrap test."""
        scores_a = [0.8, 0.85, 0.9, 0.82, 0.88]
        scores_b = [0.7, 0.75, 0.72, 0.68, 0.71]
        config = ComparisonConfig(
            method=ComparisonMethod.BOOTSTRAP, num_bootstrap=100, random_seed=42
        )
        _stat, pval = run_significance_test(scores_a, scores_b, config)
        assert pval >= 0.0
        assert pval <= 1.0

    def test_mcnemar(self) -> None:
        """Test McNemar test."""
        scores_a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
        scores_b = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
        config = ComparisonConfig(method=ComparisonMethod.MCNEMAR)
        _stat, pval = run_significance_test(scores_a, scores_b, config)
        assert pval >= 0.0
        assert pval <= 1.0

    def test_none_scores_a_raises_error(self) -> None:
        """Test that None scores_a raises ValueError."""
        config = ComparisonConfig()
        with pytest.raises(ValueError, match="scores_a cannot be None"):
            run_significance_test(None, [0.5], config)  # type: ignore[arg-type]

    def test_empty_scores_a_raises_error(self) -> None:
        """Test that empty scores_a raises ValueError."""
        config = ComparisonConfig()
        with pytest.raises(ValueError, match="scores_a cannot be empty"):
            run_significance_test([], [0.5], config)

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            run_significance_test([0.5], [0.5], None)  # type: ignore[arg-type]


class TestCompareModels:
    """Tests for compare_models function."""

    def test_basic_comparison(self) -> None:
        """Test basic model comparison."""
        result_a = ModelResult(
            model_name="model_a", scores=(0.8, 0.85, 0.9, 0.82, 0.88)
        )
        result_b = ModelResult(
            model_name="model_b", scores=(0.7, 0.75, 0.72, 0.68, 0.71)
        )
        comparison = compare_models(result_a, result_b)
        assert comparison.model_a == "model_a"
        assert comparison.model_b == "model_b"
        assert 0 <= comparison.p_value <= 1

    def test_with_custom_config(self) -> None:
        """Test comparison with custom config."""
        result_a = ModelResult(
            model_name="model_a", scores=(0.8, 0.85, 0.9, 0.82, 0.88)
        )
        result_b = ModelResult(
            model_name="model_b", scores=(0.7, 0.75, 0.72, 0.68, 0.71)
        )
        config = ComparisonConfig(
            method=ComparisonMethod.WILCOXON,
            significance_level=SignificanceLevel.P01,
        )
        comparison = compare_models(result_a, result_b, config)
        assert comparison.method == ComparisonMethod.WILCOXON

    def test_none_result_a_raises_error(self) -> None:
        """Test that None result_a raises ValueError."""
        result_b = ModelResult(model_name="model_b", scores=(0.5,))
        with pytest.raises(ValueError, match="result_a cannot be None"):
            compare_models(None, result_b)  # type: ignore[arg-type]

    def test_none_result_b_raises_error(self) -> None:
        """Test that None result_b raises ValueError."""
        result_a = ModelResult(model_name="model_a", scores=(0.5,))
        with pytest.raises(ValueError, match="result_b cannot be None"):
            compare_models(result_a, None)  # type: ignore[arg-type]

    def test_length_mismatch_raises_error(self) -> None:
        """Test that length mismatch raises ValueError."""
        result_a = ModelResult(model_name="model_a", scores=(0.5, 0.6))
        result_b = ModelResult(model_name="model_b", scores=(0.5,))
        with pytest.raises(ValueError, match="must have the same number of scores"):
            compare_models(result_a, result_b)


class TestFormatComparisonTable:
    """Tests for format_comparison_table function."""

    def test_basic_formatting(self) -> None:
        """Test basic table formatting."""
        result = ComparisonResult(
            model_a="model_1",
            model_b="model_2",
            p_value=0.03,
            effect_size=0.5,
            is_significant=True,
            confidence_interval=(0.1, 0.9),
            test_statistic=2.5,
            method=ComparisonMethod.PAIRED_TTEST,
        )
        table = format_comparison_table([result])
        assert "model_1" in table
        assert "model_2" in table
        assert "*" in table  # Significance marker

    def test_empty_results(self) -> None:
        """Test with empty results."""
        table = format_comparison_table([])
        assert "No comparison results" in table

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_comparison_table(None)  # type: ignore[arg-type]

    def test_multiple_results(self) -> None:
        """Test formatting multiple results."""
        results = [
            ComparisonResult(
                model_a="model_1",
                model_b="model_2",
                p_value=0.03,
                effect_size=0.5,
                is_significant=True,
                confidence_interval=(0.1, 0.9),
                test_statistic=2.5,
                method=ComparisonMethod.PAIRED_TTEST,
            ),
            ComparisonResult(
                model_a="model_1",
                model_b="model_3",
                p_value=0.15,
                effect_size=0.2,
                is_significant=False,
                confidence_interval=(-0.1, 0.5),
                test_statistic=1.2,
                method=ComparisonMethod.PAIRED_TTEST,
            ),
        ]
        table = format_comparison_table(results)
        assert "model_1" in table
        assert "model_2" in table
        assert "model_3" in table


class TestFormatComparisonStats:
    """Tests for format_comparison_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = ComparisonStats(
            total_comparisons=10,
            significant_count=3,
            avg_effect_size=0.4,
            max_effect_size=0.8,
            min_p_value=0.001,
        )
        formatted = format_comparison_stats(stats)
        assert "Total Comparisons" in formatted
        assert "10" in formatted
        assert "30.0%" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_comparison_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedComparisonConfig:
    """Tests for get_recommended_comparison_config function."""

    def test_large_normal_sample(self) -> None:
        """Test recommendation for large normal sample."""
        config = get_recommended_comparison_config(100, "normal")
        assert config.method == ComparisonMethod.PAIRED_TTEST

    def test_small_sample(self) -> None:
        """Test recommendation for small sample."""
        config = get_recommended_comparison_config(10, "non-normal")
        assert config.method == ComparisonMethod.WILCOXON

    def test_very_small_sample(self) -> None:
        """Test recommendation for very small sample."""
        config = get_recommended_comparison_config(5, "non-normal")
        assert config.method == ComparisonMethod.PERMUTATION

    def test_binary_data(self) -> None:
        """Test recommendation for binary data."""
        config = get_recommended_comparison_config(50, "binary")
        assert config.method == ComparisonMethod.MCNEMAR

    def test_unknown_distribution_large_sample(self) -> None:
        """Test recommendation for unknown distribution with large sample."""
        config = get_recommended_comparison_config(50, "unknown")
        assert config.method == ComparisonMethod.PAIRED_TTEST

    def test_zero_sample_size_raises_error(self) -> None:
        """Test that zero sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size must be positive"):
            get_recommended_comparison_config(0)

    def test_empty_distribution_raises_error(self) -> None:
        """Test that empty distribution raises ValueError."""
        with pytest.raises(ValueError, match="distribution cannot be empty"):
            get_recommended_comparison_config(10, "")

    def test_bootstrap_samples_small_sample(self) -> None:
        """Test that small samples get more bootstrap samples."""
        config = get_recommended_comparison_config(20, "unknown")
        assert config.num_bootstrap >= 2000

    def test_effect_size_small_sample(self) -> None:
        """Test that small samples use Hedges' g."""
        config = get_recommended_comparison_config(15, "normal")
        assert config.effect_size_type == EffectSize.HEDGES_G


class TestPropertyBased:
    """Property-based tests for comparison functions."""

    @given(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=5,
            max_size=20,
        )
    )
    @settings(max_examples=20)
    def test_effect_size_sign(self, scores: list[float]) -> None:
        """Test that effect size sign is correct."""
        if len(scores) < 5:
            return

        # Create scores_b by subtracting a constant
        scores_a = scores
        scores_b = [s - 0.1 for s in scores]

        effect = calculate_effect_size(scores_a, scores_b)
        # A should be better than B
        assert effect >= 0 or abs(effect) < 0.001

    @given(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=5,
            max_size=20,
        )
    )
    @settings(max_examples=20)
    def test_bootstrap_ci_ordering(self, scores: list[float]) -> None:
        """Test that bootstrap CI is properly ordered."""
        if len(scores) < 5:
            return

        scores_b = [s + 0.1 for s in scores]
        lower, upper = bootstrap_confidence_interval(
            scores, scores_b, num_bootstrap=50, random_seed=42
        )
        assert lower <= upper

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=20)
    def test_recommended_config_valid(self, sample_size: int) -> None:
        """Test that recommended config is always valid."""
        config = get_recommended_comparison_config(sample_size)
        assert config.num_bootstrap > 0
        assert config.method in ComparisonMethod


class TestValidFrozensets:
    """Tests for VALID_* frozensets."""

    def test_valid_comparison_methods(self) -> None:
        """Test VALID_COMPARISON_METHODS contains all enum values."""
        assert len(VALID_COMPARISON_METHODS) == len(ComparisonMethod)
        for method in ComparisonMethod:
            assert method.value in VALID_COMPARISON_METHODS

    def test_valid_significance_levels(self) -> None:
        """Test VALID_SIGNIFICANCE_LEVELS contains all enum values."""
        assert len(VALID_SIGNIFICANCE_LEVELS) == len(SignificanceLevel)
        for level in SignificanceLevel:
            assert level.value in VALID_SIGNIFICANCE_LEVELS

    def test_valid_effect_sizes(self) -> None:
        """Test VALID_EFFECT_SIZES contains all enum values."""
        assert len(VALID_EFFECT_SIZES) == len(EffectSize)
        for size in EffectSize:
            assert size.value in VALID_EFFECT_SIZES


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_identical_scores_ttest(self) -> None:
        """Test t-test with identical scores."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        config = ComparisonConfig(method=ComparisonMethod.PAIRED_TTEST)
        _stat, pval = run_significance_test(scores, scores, config)
        assert pval == pytest.approx(1.0)

    def test_identical_binary_mcnemar(self) -> None:
        """Test McNemar with identical binary scores."""
        scores = [1, 0, 1, 0, 1]
        config = ComparisonConfig(method=ComparisonMethod.MCNEMAR)
        _stat, pval = run_significance_test(scores, scores, config)
        assert pval == pytest.approx(1.0)

    def test_wilcoxon_no_differences(self) -> None:
        """Test Wilcoxon with no differences."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        config = ComparisonConfig(method=ComparisonMethod.WILCOXON)
        _stat, pval = run_significance_test(scores, scores, config)
        assert pval == pytest.approx(1.0)

    def test_large_effect_size(self) -> None:
        """Test with large effect size."""
        scores_a = [0.9, 0.92, 0.91, 0.93, 0.9]
        scores_b = [0.1, 0.12, 0.11, 0.13, 0.1]
        effect = calculate_effect_size(scores_a, scores_b)
        assert effect > 1.0  # Large effect

    def test_negative_effect_size(self) -> None:
        """Test that effect size can be negative."""
        scores_a = [0.3, 0.32, 0.31, 0.33, 0.3]
        scores_b = [0.8, 0.82, 0.81, 0.83, 0.8]
        effect = calculate_effect_size(scores_a, scores_b)
        assert effect < 0  # A is worse than B


class TestIntegration:
    """Integration tests for the comparison module."""

    def test_full_workflow(self) -> None:
        """Test complete comparison workflow."""
        # Create model results
        result_a = create_model_result(
            "gpt-4", [0.92, 0.89, 0.94, 0.91, 0.93, 0.90, 0.88, 0.95, 0.91, 0.92]
        )
        result_b = create_model_result(
            "gpt-3.5",
            [0.85, 0.82, 0.88, 0.84, 0.86, 0.83, 0.81, 0.87, 0.84, 0.85],
        )

        # Get recommended config
        config = get_recommended_comparison_config(
            sample_size=10, distribution="normal"
        )

        # Run comparison
        comparison = compare_models(result_a, result_b, config)

        # Verify results
        assert comparison.model_a == "gpt-4"
        assert comparison.model_b == "gpt-3.5"
        assert comparison.effect_size > 0
        assert comparison.is_significant

        # Format results
        table = format_comparison_table([comparison])
        assert "gpt-4" in table
        assert "gpt-3.5" in table
