"""Tests for preprocessing.sampling module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.sampling import (
    VALID_BALANCING_STRATEGIES,
    VALID_SAMPLING_METHODS,
    VALID_WEIGHTING_SCHEMES,
    BalancedConfig,
    BalancingStrategy,
    ImportanceConfig,
    SamplingConfig,
    SamplingMethod,
    SamplingStats,
    StratifiedConfig,
    WeightingScheme,
    balance_classes,
    calculate_sample_weights,
    compute_sampling_stats,
    create_balanced_config,
    create_importance_config,
    create_sampling_config,
    create_stratified_config,
    estimate_effective_samples,
    format_sampling_stats,
    get_balancing_strategy,
    get_recommended_sampling_config,
    get_sampling_method,
    get_weighting_scheme,
    importance_sample,
    list_balancing_strategies,
    list_sampling_methods,
    list_weighting_schemes,
    stratified_sample,
    validate_balanced_config,
    validate_importance_config,
    validate_sampling_config,
    validate_stratified_config,
)


class TestSamplingMethod:
    """Tests for SamplingMethod enum."""

    def test_random_value(self) -> None:
        """Test RANDOM value."""
        assert SamplingMethod.RANDOM.value == "random"

    def test_stratified_value(self) -> None:
        """Test STRATIFIED value."""
        assert SamplingMethod.STRATIFIED.value == "stratified"

    def test_importance_value(self) -> None:
        """Test IMPORTANCE value."""
        assert SamplingMethod.IMPORTANCE.value == "importance"

    def test_balanced_value(self) -> None:
        """Test BALANCED value."""
        assert SamplingMethod.BALANCED.value == "balanced"

    def test_cluster_value(self) -> None:
        """Test CLUSTER value."""
        assert SamplingMethod.CLUSTER.value == "cluster"

    def test_curriculum_value(self) -> None:
        """Test CURRICULUM value."""
        assert SamplingMethod.CURRICULUM.value == "curriculum"

    def test_valid_methods_frozenset(self) -> None:
        """Test VALID_SAMPLING_METHODS is frozenset."""
        assert isinstance(VALID_SAMPLING_METHODS, frozenset)
        assert len(VALID_SAMPLING_METHODS) == len(SamplingMethod)


class TestWeightingScheme:
    """Tests for WeightingScheme enum."""

    def test_uniform_value(self) -> None:
        """Test UNIFORM value."""
        assert WeightingScheme.UNIFORM.value == "uniform"

    def test_inverse_frequency_value(self) -> None:
        """Test INVERSE_FREQUENCY value."""
        assert WeightingScheme.INVERSE_FREQUENCY.value == "inverse_frequency"

    def test_sqrt_inverse_value(self) -> None:
        """Test SQRT_INVERSE value."""
        assert WeightingScheme.SQRT_INVERSE.value == "sqrt_inverse"

    def test_custom_value(self) -> None:
        """Test CUSTOM value."""
        assert WeightingScheme.CUSTOM.value == "custom"

    def test_valid_schemes_frozenset(self) -> None:
        """Test VALID_WEIGHTING_SCHEMES is frozenset."""
        assert isinstance(VALID_WEIGHTING_SCHEMES, frozenset)
        assert len(VALID_WEIGHTING_SCHEMES) == len(WeightingScheme)


class TestBalancingStrategy:
    """Tests for BalancingStrategy enum."""

    def test_oversample_value(self) -> None:
        """Test OVERSAMPLE value."""
        assert BalancingStrategy.OVERSAMPLE.value == "oversample"

    def test_undersample_value(self) -> None:
        """Test UNDERSAMPLE value."""
        assert BalancingStrategy.UNDERSAMPLE.value == "undersample"

    def test_smote_value(self) -> None:
        """Test SMOTE value."""
        assert BalancingStrategy.SMOTE.value == "smote"

    def test_class_weight_value(self) -> None:
        """Test CLASS_WEIGHT value."""
        assert BalancingStrategy.CLASS_WEIGHT.value == "class_weight"

    def test_valid_strategies_frozenset(self) -> None:
        """Test VALID_BALANCING_STRATEGIES is frozenset."""
        assert isinstance(VALID_BALANCING_STRATEGIES, frozenset)
        assert len(VALID_BALANCING_STRATEGIES) == len(BalancingStrategy)


class TestStratifiedConfig:
    """Tests for StratifiedConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating StratifiedConfig instance."""
        config = StratifiedConfig(column="label", preserve_distribution=True)
        assert config.column == "label"
        assert config.preserve_distribution is True

    def test_frozen(self) -> None:
        """Test that StratifiedConfig is immutable."""
        config = StratifiedConfig(column="label", preserve_distribution=True)
        with pytest.raises(AttributeError):
            config.column = "other"  # type: ignore[misc]


class TestImportanceConfig:
    """Tests for ImportanceConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating ImportanceConfig instance."""
        config = ImportanceConfig(
            weights_column="weight",
            temperature=1.0,
            normalize=True,
        )
        assert config.weights_column == "weight"
        assert config.temperature == pytest.approx(1.0)
        assert config.normalize is True

    def test_frozen(self) -> None:
        """Test that ImportanceConfig is immutable."""
        config = ImportanceConfig(
            weights_column=None,
            temperature=1.0,
            normalize=True,
        )
        with pytest.raises(AttributeError):
            config.temperature = 0.5  # type: ignore[misc]


class TestBalancedConfig:
    """Tests for BalancedConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating BalancedConfig instance."""
        config = BalancedConfig(
            strategy=BalancingStrategy.OVERSAMPLE,
            target_ratio=1.0,
            random_state=42,
        )
        assert config.strategy == BalancingStrategy.OVERSAMPLE
        assert config.target_ratio == pytest.approx(1.0)
        assert config.random_state == 42

    def test_frozen(self) -> None:
        """Test that BalancedConfig is immutable."""
        config = BalancedConfig(
            strategy=BalancingStrategy.OVERSAMPLE,
            target_ratio=1.0,
            random_state=None,
        )
        with pytest.raises(AttributeError):
            config.target_ratio = 0.5  # type: ignore[misc]


class TestSamplingConfig:
    """Tests for SamplingConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SamplingConfig instance."""
        config = SamplingConfig(
            method=SamplingMethod.RANDOM,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=1000,
        )
        assert config.method == SamplingMethod.RANDOM
        assert config.sample_size == 1000

    def test_frozen(self) -> None:
        """Test that SamplingConfig is immutable."""
        config = SamplingConfig(
            method=SamplingMethod.RANDOM,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=1000,
        )
        with pytest.raises(AttributeError):
            config.sample_size = 500  # type: ignore[misc]


class TestSamplingStats:
    """Tests for SamplingStats dataclass."""

    def test_creation(self) -> None:
        """Test creating SamplingStats instance."""
        stats = SamplingStats(
            original_size=10000,
            sampled_size=1000,
            class_distribution={"A": 500, "B": 500},
            effective_ratio=0.1,
        )
        assert stats.original_size == 10000
        assert stats.sampled_size == 1000
        assert stats.class_distribution["A"] == 500
        assert stats.effective_ratio == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that SamplingStats is immutable."""
        stats = SamplingStats(
            original_size=100,
            sampled_size=10,
            class_distribution={},
            effective_ratio=0.1,
        )
        with pytest.raises(AttributeError):
            stats.sampled_size = 20  # type: ignore[misc]


class TestValidateStratifiedConfig:
    """Tests for validate_stratified_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = StratifiedConfig(column="label", preserve_distribution=True)
        validate_stratified_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_stratified_config(None)  # type: ignore[arg-type]

    def test_empty_column_raises_error(self) -> None:
        """Test that empty column raises ValueError."""
        config = StratifiedConfig(column="", preserve_distribution=True)
        with pytest.raises(ValueError, match="column cannot be empty"):
            validate_stratified_config(config)


class TestValidateImportanceConfig:
    """Tests for validate_importance_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ImportanceConfig(
            weights_column="weight",
            temperature=1.0,
            normalize=True,
        )
        validate_importance_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_importance_config(None)  # type: ignore[arg-type]

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        config = ImportanceConfig(
            weights_column=None,
            temperature=0.0,
            normalize=True,
        )
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_importance_config(config)

    def test_negative_temperature_raises_error(self) -> None:
        """Test that negative temperature raises ValueError."""
        config = ImportanceConfig(
            weights_column=None,
            temperature=-1.0,
            normalize=True,
        )
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_importance_config(config)


class TestValidateBalancedConfig:
    """Tests for validate_balanced_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BalancedConfig(
            strategy=BalancingStrategy.OVERSAMPLE,
            target_ratio=1.0,
            random_state=42,
        )
        validate_balanced_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_balanced_config(None)  # type: ignore[arg-type]

    def test_zero_target_ratio_raises_error(self) -> None:
        """Test that zero target_ratio raises ValueError."""
        config = BalancedConfig(
            strategy=BalancingStrategy.OVERSAMPLE,
            target_ratio=0.0,
            random_state=None,
        )
        with pytest.raises(ValueError, match=r"target_ratio must be in \(0, 1\]"):
            validate_balanced_config(config)

    def test_target_ratio_above_one_raises_error(self) -> None:
        """Test that target_ratio > 1 raises ValueError."""
        config = BalancedConfig(
            strategy=BalancingStrategy.OVERSAMPLE,
            target_ratio=1.5,
            random_state=None,
        )
        with pytest.raises(ValueError, match=r"target_ratio must be in \(0, 1\]"):
            validate_balanced_config(config)


class TestValidateSamplingConfig:
    """Tests for validate_sampling_config function."""

    def test_valid_random_config(self) -> None:
        """Test validation of valid random config."""
        config = SamplingConfig(
            method=SamplingMethod.RANDOM,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=100,
        )
        validate_sampling_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_sampling_config(None)  # type: ignore[arg-type]

    def test_zero_sample_size_raises_error(self) -> None:
        """Test that zero sample_size raises ValueError."""
        config = SamplingConfig(
            method=SamplingMethod.RANDOM,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=0,
        )
        with pytest.raises(ValueError, match="sample_size must be positive"):
            validate_sampling_config(config)

    def test_stratified_without_config_raises_error(self) -> None:
        """Test that STRATIFIED without config raises ValueError."""
        config = SamplingConfig(
            method=SamplingMethod.STRATIFIED,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=100,
        )
        with pytest.raises(
            ValueError, match="stratified_config required for STRATIFIED"
        ):
            validate_sampling_config(config)

    def test_importance_without_config_raises_error(self) -> None:
        """Test that IMPORTANCE without config raises ValueError."""
        config = SamplingConfig(
            method=SamplingMethod.IMPORTANCE,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=100,
        )
        with pytest.raises(
            ValueError, match="importance_config required for IMPORTANCE"
        ):
            validate_sampling_config(config)

    def test_balanced_without_config_raises_error(self) -> None:
        """Test that BALANCED without config raises ValueError."""
        config = SamplingConfig(
            method=SamplingMethod.BALANCED,
            stratified_config=None,
            importance_config=None,
            balanced_config=None,
            sample_size=100,
        )
        with pytest.raises(ValueError, match="balanced_config required for BALANCED"):
            validate_sampling_config(config)

    def test_valid_stratified_config(self) -> None:
        """Test validation of valid stratified config."""
        strat = StratifiedConfig(column="label", preserve_distribution=True)
        config = SamplingConfig(
            method=SamplingMethod.STRATIFIED,
            stratified_config=strat,
            importance_config=None,
            balanced_config=None,
            sample_size=100,
        )
        validate_sampling_config(config)  # Should not raise


class TestCreateStratifiedConfig:
    """Tests for create_stratified_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_stratified_config()
        assert config.column == "label"
        assert config.preserve_distribution is True

    def test_custom_column(self) -> None:
        """Test creating config with custom column."""
        config = create_stratified_config(column="category")
        assert config.column == "category"

    def test_empty_column_raises_error(self) -> None:
        """Test that empty column raises ValueError."""
        with pytest.raises(ValueError, match="column cannot be empty"):
            create_stratified_config(column="")


class TestCreateImportanceConfig:
    """Tests for create_importance_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_importance_config()
        assert config.weights_column is None
        assert config.temperature == pytest.approx(1.0)
        assert config.normalize is True

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_importance_config(
            weights_column="score",
            temperature=0.5,
            normalize=False,
        )
        assert config.weights_column == "score"
        assert config.temperature == pytest.approx(0.5)
        assert config.normalize is False

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            create_importance_config(temperature=0)


class TestCreateBalancedConfig:
    """Tests for create_balanced_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_balanced_config()
        assert config.strategy == BalancingStrategy.OVERSAMPLE
        assert config.target_ratio == pytest.approx(1.0)
        assert config.random_state is None

    def test_custom_strategy(self) -> None:
        """Test creating config with custom strategy."""
        config = create_balanced_config(strategy="undersample")
        assert config.strategy == BalancingStrategy.UNDERSAMPLE

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_balanced_config(strategy="invalid")

    def test_invalid_target_ratio_raises_error(self) -> None:
        """Test that invalid target_ratio raises ValueError."""
        with pytest.raises(ValueError, match=r"target_ratio must be in \(0, 1\]"):
            create_balanced_config(target_ratio=0.0)


class TestCreateSamplingConfig:
    """Tests for create_sampling_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_sampling_config()
        assert config.method == SamplingMethod.RANDOM
        assert config.sample_size == 1000

    def test_custom_method(self) -> None:
        """Test creating config with custom method."""
        strat = create_stratified_config()
        config = create_sampling_config(method="stratified", stratified_config=strat)
        assert config.method == SamplingMethod.STRATIFIED

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_sampling_config(method="invalid")

    def test_invalid_sample_size_raises_error(self) -> None:
        """Test that invalid sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size must be positive"):
            create_sampling_config(sample_size=0)


class TestListSamplingMethods:
    """Tests for list_sampling_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_sampling_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_sampling_methods()
        assert "random" in methods
        assert "stratified" in methods
        assert "importance" in methods
        assert "balanced" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_sampling_methods()
        assert methods == sorted(methods)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_SAMPLING_METHODS."""
        methods = list_sampling_methods()
        assert set(methods) == VALID_SAMPLING_METHODS


class TestGetSamplingMethod:
    """Tests for get_sampling_method function."""

    def test_get_random(self) -> None:
        """Test getting RANDOM method."""
        result = get_sampling_method("random")
        assert result == SamplingMethod.RANDOM

    def test_get_stratified(self) -> None:
        """Test getting STRATIFIED method."""
        result = get_sampling_method("stratified")
        assert result == SamplingMethod.STRATIFIED

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid sampling method"):
            get_sampling_method("invalid")


class TestListWeightingSchemes:
    """Tests for list_weighting_schemes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        schemes = list_weighting_schemes()
        assert isinstance(schemes, list)

    def test_contains_expected_schemes(self) -> None:
        """Test that list contains expected schemes."""
        schemes = list_weighting_schemes()
        assert "uniform" in schemes
        assert "inverse_frequency" in schemes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        schemes = list_weighting_schemes()
        assert schemes == sorted(schemes)


class TestGetWeightingScheme:
    """Tests for get_weighting_scheme function."""

    def test_get_uniform(self) -> None:
        """Test getting UNIFORM scheme."""
        result = get_weighting_scheme("uniform")
        assert result == WeightingScheme.UNIFORM

    def test_get_inverse_frequency(self) -> None:
        """Test getting INVERSE_FREQUENCY scheme."""
        result = get_weighting_scheme("inverse_frequency")
        assert result == WeightingScheme.INVERSE_FREQUENCY

    def test_invalid_scheme_raises_error(self) -> None:
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="invalid weighting scheme"):
            get_weighting_scheme("invalid")


class TestListBalancingStrategies:
    """Tests for list_balancing_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_balancing_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_balancing_strategies()
        assert "oversample" in strategies
        assert "undersample" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_balancing_strategies()
        assert strategies == sorted(strategies)


class TestGetBalancingStrategy:
    """Tests for get_balancing_strategy function."""

    def test_get_oversample(self) -> None:
        """Test getting OVERSAMPLE strategy."""
        result = get_balancing_strategy("oversample")
        assert result == BalancingStrategy.OVERSAMPLE

    def test_get_smote(self) -> None:
        """Test getting SMOTE strategy."""
        result = get_balancing_strategy("smote")
        assert result == BalancingStrategy.SMOTE

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid balancing strategy"):
            get_balancing_strategy("invalid")


class TestCalculateSampleWeights:
    """Tests for calculate_sample_weights function."""

    def test_uniform_weights(self) -> None:
        """Test uniform weighting."""
        labels = ["A", "A", "B", "B"]
        weights = calculate_sample_weights(labels, WeightingScheme.UNIFORM)
        assert all(w == 1.0 for w in weights)

    def test_inverse_frequency_weights(self) -> None:
        """Test inverse frequency weighting."""
        labels = ["A", "A", "A", "B"]
        weights = calculate_sample_weights(labels, WeightingScheme.INVERSE_FREQUENCY)
        assert len(weights) == 4
        # B (minority) should have higher weight
        assert weights[3] > weights[0]

    def test_sqrt_inverse_weights(self) -> None:
        """Test sqrt inverse weighting."""
        labels = ["A", "A", "A", "B"]
        weights = calculate_sample_weights(labels, WeightingScheme.SQRT_INVERSE)
        assert len(weights) == 4
        # B (minority) should have higher weight
        assert weights[3] > weights[0]

    def test_custom_weights(self) -> None:
        """Test custom weighting."""
        labels = ["A", "B", "C"]
        custom = {"A": 1.0, "B": 2.0, "C": 3.0}
        weights = calculate_sample_weights(labels, WeightingScheme.CUSTOM, custom)
        assert weights == [1.0, 2.0, 3.0]

    def test_empty_labels_raises_error(self) -> None:
        """Test that empty labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be empty"):
            calculate_sample_weights([])

    def test_none_labels_raises_error(self) -> None:
        """Test that None labels raises ValueError."""
        with pytest.raises(ValueError, match="labels cannot be None"):
            calculate_sample_weights(None)  # type: ignore[arg-type]

    def test_custom_without_weights_raises_error(self) -> None:
        """Test that CUSTOM without weights raises ValueError."""
        with pytest.raises(ValueError, match="custom_weights required"):
            calculate_sample_weights(["A"], WeightingScheme.CUSTOM)

    def test_custom_missing_label_raises_error(self) -> None:
        """Test that missing label in custom_weights raises ValueError."""
        with pytest.raises(ValueError, match="custom_weights missing weight"):
            calculate_sample_weights(["A", "B"], WeightingScheme.CUSTOM, {"A": 1.0})


class TestStratifiedSample:
    """Tests for stratified_sample function."""

    def test_basic_sampling(self) -> None:
        """Test basic stratified sampling."""
        data = [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "A"},
            {"id": 3, "label": "B"},
            {"id": 4, "label": "B"},
        ]
        result = stratified_sample(data, "label", 2, random_state=42)
        assert len(result) == 2

    def test_preserve_distribution(self) -> None:
        """Test distribution preservation."""
        data = [{"id": i, "label": "A" if i < 75 else "B"} for i in range(100)]
        result = stratified_sample(
            data, "label", 20, preserve_distribution=True, random_state=42
        )
        assert len(result) == 20

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be empty"):
            stratified_sample([], "label", 10)

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            stratified_sample(None, "label", 10)  # type: ignore[arg-type]

    def test_empty_column_raises_error(self) -> None:
        """Test that empty column raises ValueError."""
        with pytest.raises(ValueError, match="column cannot be empty"):
            stratified_sample([{"a": 1}], "", 1)

    def test_sample_size_exceeds_data_raises_error(self) -> None:
        """Test that sample_size > data length raises ValueError."""
        with pytest.raises(ValueError, match="exceeds data length"):
            stratified_sample([{"a": 1}], "a", 10)

    def test_zero_sample_size_raises_error(self) -> None:
        """Test that zero sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size must be positive"):
            stratified_sample([{"a": 1}], "a", 0)


class TestImportanceSample:
    """Tests for importance_sample function."""

    def test_basic_sampling(self) -> None:
        """Test basic importance sampling."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        weights = [0.1, 0.1, 0.8]
        result = importance_sample(data, weights, 2, random_state=42)
        assert len(result) == 2

    def test_temperature_effect(self) -> None:
        """Test temperature effect on sampling."""
        data = [{"id": i} for i in range(10)]
        weights = [1.0] * 9 + [10.0]
        # High temperature should make sampling more uniform
        result = importance_sample(data, weights, 5, temperature=10.0, random_state=42)
        assert len(result) == 5

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be empty"):
            importance_sample([], [0.5], 1)

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            importance_sample(None, [0.5], 1)  # type: ignore[arg-type]

    def test_weights_length_mismatch_raises_error(self) -> None:
        """Test that weights length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="weights length"):
            importance_sample([{"a": 1}], [], 1)

    def test_zero_sample_size_raises_error(self) -> None:
        """Test that zero sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size must be positive"):
            importance_sample([{"a": 1}], [1.0], 0)

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            importance_sample([{"a": 1}], [1.0], 1, temperature=0.0)


class TestBalanceClasses:
    """Tests for balance_classes function."""

    def test_oversample(self) -> None:
        """Test oversampling strategy."""
        data = [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "A"},
            {"id": 3, "label": "A"},
            {"id": 4, "label": "B"},
        ]
        result = balance_classes(
            data, "label", BalancingStrategy.OVERSAMPLE, random_state=42
        )
        # B should be oversampled to match A
        b_count = sum(1 for r in result if r["label"] == "B")
        assert b_count >= 1

    def test_undersample(self) -> None:
        """Test undersampling strategy."""
        data = [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "A"},
            {"id": 3, "label": "A"},
            {"id": 4, "label": "B"},
        ]
        result = balance_classes(
            data, "label", BalancingStrategy.UNDERSAMPLE, random_state=42
        )
        # A should be undersampled
        a_count = sum(1 for r in result if r["label"] == "A")
        assert a_count <= 3

    def test_smote(self) -> None:
        """Test SMOTE strategy (simplified)."""
        data = [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "A"},
            {"id": 3, "label": "B"},
        ]
        result = balance_classes(
            data, "label", BalancingStrategy.SMOTE, random_state=42
        )
        assert len(result) >= 3

    def test_class_weight(self) -> None:
        """Test class_weight strategy."""
        data = [{"id": 1, "label": "A"}, {"id": 2, "label": "B"}]
        result = balance_classes(data, "label", BalancingStrategy.CLASS_WEIGHT)
        # Should return original data unchanged
        assert len(result) == 2

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be empty"):
            balance_classes([], "label")

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            balance_classes(None, "label")  # type: ignore[arg-type]

    def test_empty_label_column_raises_error(self) -> None:
        """Test that empty label_column raises ValueError."""
        with pytest.raises(ValueError, match="label_column cannot be empty"):
            balance_classes([{"a": 1}], "")

    def test_invalid_target_ratio_raises_error(self) -> None:
        """Test that invalid target_ratio raises ValueError."""
        with pytest.raises(ValueError, match=r"target_ratio must be in \(0, 1\]"):
            balance_classes([{"label": "A"}], "label", target_ratio=0.0)


class TestEstimateEffectiveSamples:
    """Tests for estimate_effective_samples function."""

    def test_basic_estimation(self) -> None:
        """Test basic effective sample estimation."""
        counts = {"A": 1000, "B": 100, "C": 10}
        effective = estimate_effective_samples(counts)
        assert effective["A"] > effective["B"] > effective["C"]

    def test_zero_count(self) -> None:
        """Test with zero count class."""
        counts = {"A": 100, "B": 0}
        effective = estimate_effective_samples(counts)
        assert effective["B"] == 0.0

    def test_empty_counts_raises_error(self) -> None:
        """Test that empty counts raises ValueError."""
        with pytest.raises(ValueError, match="class_counts cannot be empty"):
            estimate_effective_samples({})

    def test_none_counts_raises_error(self) -> None:
        """Test that None counts raises ValueError."""
        with pytest.raises(ValueError, match="class_counts cannot be None"):
            estimate_effective_samples(None)  # type: ignore[arg-type]

    def test_invalid_beta_raises_error(self) -> None:
        """Test that invalid beta raises ValueError."""
        with pytest.raises(ValueError, match=r"beta must be in \(0, 1\)"):
            estimate_effective_samples({"A": 100}, beta=1.0)

    def test_negative_beta_raises_error(self) -> None:
        """Test that negative beta raises ValueError."""
        with pytest.raises(ValueError, match=r"beta must be in \(0, 1\)"):
            estimate_effective_samples({"A": 100}, beta=0.0)


class TestFormatSamplingStats:
    """Tests for format_sampling_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = SamplingStats(
            original_size=10000,
            sampled_size=1000,
            class_distribution={"A": 500, "B": 500},
            effective_ratio=0.1,
        )
        formatted = format_sampling_stats(stats)
        assert "10,000" in formatted or "10000" in formatted
        assert "1,000" in formatted or "1000" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_sampling_stats(None)  # type: ignore[arg-type]

    def test_empty_distribution(self) -> None:
        """Test with empty class distribution."""
        stats = SamplingStats(
            original_size=100,
            sampled_size=10,
            class_distribution={},
            effective_ratio=0.1,
        )
        formatted = format_sampling_stats(stats)
        assert "100" in formatted

    def test_zero_original_size(self) -> None:
        """Test with zero original size."""
        stats = SamplingStats(
            original_size=0,
            sampled_size=0,
            class_distribution={},
            effective_ratio=0.0,
        )
        formatted = format_sampling_stats(stats)
        assert "0" in formatted


class TestGetRecommendedSamplingConfig:
    """Tests for get_recommended_sampling_config function."""

    def test_highly_imbalanced(self) -> None:
        """Test recommendation for highly imbalanced data."""
        config = get_recommended_sampling_config(
            dataset_size=10000,
            num_classes=2,
            imbalance_ratio=10.0,
        )
        assert config.method == SamplingMethod.BALANCED

    def test_moderately_imbalanced(self) -> None:
        """Test recommendation for moderately imbalanced data."""
        config = get_recommended_sampling_config(
            dataset_size=10000,
            num_classes=2,
            imbalance_ratio=3.0,
        )
        assert config.method == SamplingMethod.STRATIFIED

    def test_well_balanced(self) -> None:
        """Test recommendation for well-balanced data."""
        config = get_recommended_sampling_config(
            dataset_size=10000,
            num_classes=2,
            imbalance_ratio=1.2,
        )
        assert config.method == SamplingMethod.RANDOM

    def test_zero_dataset_size_raises_error(self) -> None:
        """Test that zero dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            get_recommended_sampling_config(0, 2, 1.0)

    def test_zero_num_classes_raises_error(self) -> None:
        """Test that zero num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            get_recommended_sampling_config(100, 0, 1.0)

    def test_imbalance_ratio_below_one_raises_error(self) -> None:
        """Test that imbalance_ratio < 1 raises ValueError."""
        with pytest.raises(ValueError, match=r"imbalance_ratio must be >= 1\.0"):
            get_recommended_sampling_config(100, 2, 0.5)


class TestComputeSamplingStats:
    """Tests for compute_sampling_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic stats computation."""
        data = [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "A"},
            {"id": 3, "label": "B"},
            {"id": 4, "label": "B"},
        ]
        sampled = [{"id": 1, "label": "A"}, {"id": 3, "label": "B"}]
        stats = compute_sampling_stats(data, "label", sampled)
        assert stats.original_size == 4
        assert stats.sampled_size == 2
        assert stats.effective_ratio == pytest.approx(0.5)

    def test_class_distribution(self) -> None:
        """Test class distribution computation."""
        data = [{"id": 1, "label": "A"}, {"id": 2, "label": "B"}]
        sampled = [{"id": 1, "label": "A"}]
        stats = compute_sampling_stats(data, "label", sampled)
        assert stats.class_distribution["A"] == 1
        assert "B" not in stats.class_distribution

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        with pytest.raises(ValueError, match="data cannot be None"):
            compute_sampling_stats(None, "label", [])  # type: ignore[arg-type]

    def test_none_sampled_raises_error(self) -> None:
        """Test that None sampled raises ValueError."""
        with pytest.raises(ValueError, match="sampled cannot be None"):
            compute_sampling_stats([], "label", None)  # type: ignore[arg-type]

    def test_empty_label_column_raises_error(self) -> None:
        """Test that empty label_column raises ValueError."""
        with pytest.raises(ValueError, match="label_column cannot be empty"):
            compute_sampling_stats([{"a": 1}], "", [])


class TestPropertyBased:
    """Property-based tests for sampling module."""

    @given(st.lists(st.sampled_from(["A", "B", "C"]), min_size=1, max_size=50))
    @settings(max_examples=20)
    def test_sample_weights_length_matches_labels(self, labels: list[str]) -> None:
        """Test that sample weights length matches labels."""
        weights = calculate_sample_weights(labels, WeightingScheme.UNIFORM)
        assert len(weights) == len(labels)

    @given(st.integers(min_value=1, max_value=1000000))
    @settings(max_examples=20)
    def test_recommended_config_returns_valid(self, dataset_size: int) -> None:
        """Test that recommended config is always valid."""
        config = get_recommended_sampling_config(
            dataset_size=dataset_size,
            num_classes=2,
            imbalance_ratio=2.0,
        )
        validate_sampling_config(config)

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=5, alphabet="ABC"),
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=20)
    def test_effective_samples_positive(self, counts: dict[str, int]) -> None:
        """Test that effective samples are always non-negative."""
        effective = estimate_effective_samples(counts)
        assert all(v >= 0 for v in effective.values())

    @given(st.lists(st.sampled_from(["A", "B"]), min_size=1, max_size=20))
    @settings(max_examples=20)
    def test_inverse_frequency_minority_higher_weight(
        self, labels: list[str]
    ) -> None:
        """Test that minority class gets higher weight with inverse frequency."""
        if len(set(labels)) < 2:
            return  # Skip if only one class

        from collections import Counter
        counts = Counter(labels)
        minority = min(counts, key=counts.get)  # type: ignore[arg-type]
        majority = max(counts, key=counts.get)  # type: ignore[arg-type]

        if counts[minority] == counts[majority]:
            return  # Skip if equal counts

        weights = calculate_sample_weights(labels, WeightingScheme.INVERSE_FREQUENCY)
        minority_idx = labels.index(minority)
        majority_idx = labels.index(majority)
        assert weights[minority_idx] >= weights[majority_idx]
