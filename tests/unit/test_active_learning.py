"""Tests for training.active_learning module."""

from __future__ import annotations

import math

import pytest

from hf_gtc.training.active_learning import (
    VALID_POOLING_METHODS,
    VALID_QUERY_STRATEGIES,
    VALID_UNCERTAINTY_MEASURES,
    ActiveLearningConfig,
    ActiveLearningStats,
    PoolingMethod,
    QueryConfig,
    QueryResult,
    QueryStrategy,
    UncertaintyMeasure,
    calculate_query_efficiency,
    calculate_uncertainty,
    create_active_learning_config,
    create_active_learning_stats,
    create_query_config,
    create_query_result,
    estimate_labeling_cost,
    format_active_learning_stats,
    get_pooling_method,
    get_query_strategy,
    get_recommended_active_learning_config,
    get_uncertainty_measure,
    list_pooling_methods,
    list_query_strategies,
    list_uncertainty_measures,
    select_samples,
    validate_active_learning_config,
    validate_active_learning_stats,
    validate_query_config,
    validate_query_result,
)


class TestQueryStrategy:
    """Tests for QueryStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in QueryStrategy:
            assert isinstance(strategy.value, str)

    def test_uncertainty_value(self) -> None:
        """Uncertainty has correct value."""
        assert QueryStrategy.UNCERTAINTY.value == "uncertainty"

    def test_margin_value(self) -> None:
        """Margin has correct value."""
        assert QueryStrategy.MARGIN.value == "margin"

    def test_entropy_value(self) -> None:
        """Entropy has correct value."""
        assert QueryStrategy.ENTROPY.value == "entropy"

    def test_bald_value(self) -> None:
        """BALD has correct value."""
        assert QueryStrategy.BALD.value == "bald"

    def test_coreset_value(self) -> None:
        """Coreset has correct value."""
        assert QueryStrategy.CORESET.value == "coreset"

    def test_badge_value(self) -> None:
        """BADGE has correct value."""
        assert QueryStrategy.BADGE.value == "badge"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_QUERY_STRATEGIES is a frozenset."""
        assert isinstance(VALID_QUERY_STRATEGIES, frozenset)
        assert len(VALID_QUERY_STRATEGIES) == 6


class TestUncertaintyMeasure:
    """Tests for UncertaintyMeasure enum."""

    def test_all_measures_have_values(self) -> None:
        """All measures have string values."""
        for measure in UncertaintyMeasure:
            assert isinstance(measure.value, str)

    def test_predictive_entropy_value(self) -> None:
        """Predictive entropy has correct value."""
        assert UncertaintyMeasure.PREDICTIVE_ENTROPY.value == "predictive_entropy"

    def test_mutual_information_value(self) -> None:
        """Mutual information has correct value."""
        assert UncertaintyMeasure.MUTUAL_INFORMATION.value == "mutual_information"

    def test_variation_ratio_value(self) -> None:
        """Variation ratio has correct value."""
        assert UncertaintyMeasure.VARIATION_RATIO.value == "variation_ratio"

    def test_least_confidence_value(self) -> None:
        """Least confidence has correct value."""
        assert UncertaintyMeasure.LEAST_CONFIDENCE.value == "least_confidence"

    def test_valid_measures_frozenset(self) -> None:
        """VALID_UNCERTAINTY_MEASURES is a frozenset."""
        assert isinstance(VALID_UNCERTAINTY_MEASURES, frozenset)
        assert len(VALID_UNCERTAINTY_MEASURES) == 4


class TestPoolingMethod:
    """Tests for PoolingMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in PoolingMethod:
            assert isinstance(method.value, str)

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert PoolingMethod.RANDOM.value == "random"

    def test_cluster_value(self) -> None:
        """Cluster has correct value."""
        assert PoolingMethod.CLUSTER.value == "cluster"

    def test_diversity_value(self) -> None:
        """Diversity has correct value."""
        assert PoolingMethod.DIVERSITY.value == "diversity"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_POOLING_METHODS is a frozenset."""
        assert isinstance(VALID_POOLING_METHODS, frozenset)
        assert len(VALID_POOLING_METHODS) == 3


class TestQueryConfig:
    """Tests for QueryConfig dataclass."""

    def test_create_config(self) -> None:
        """Create query config."""
        config = QueryConfig(
            strategy=QueryStrategy.UNCERTAINTY,
            batch_size=32,
            uncertainty_measure=UncertaintyMeasure.PREDICTIVE_ENTROPY,
            diversity_weight=0.5,
        )
        assert config.strategy == QueryStrategy.UNCERTAINTY
        assert config.batch_size == 32
        assert config.diversity_weight == 0.5

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        with pytest.raises(AttributeError):
            config.batch_size = 64  # type: ignore[misc]


class TestActiveLearningConfig:
    """Tests for ActiveLearningConfig dataclass."""

    def test_create_config(self) -> None:
        """Create active learning config."""
        query_config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        config = ActiveLearningConfig(
            query_config=query_config,
            initial_pool_size=100,
            labeling_budget=1000,
            stopping_criterion=0.001,
        )
        assert config.query_config == query_config
        assert config.initial_pool_size == 100
        assert config.labeling_budget == 1000

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        query_config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        config = ActiveLearningConfig(query_config, 100, 1000, 0.001)
        with pytest.raises(AttributeError):
            config.initial_pool_size = 200  # type: ignore[misc]


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_create_result(self) -> None:
        """Create query result."""
        result = QueryResult(
            indices=(5, 12, 23),
            scores=(0.95, 0.92, 0.88),
            uncertainty_values=(0.9, 0.85, 0.8),
        )
        assert result.indices == (5, 12, 23)
        assert result.scores == (0.95, 0.92, 0.88)

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = QueryResult((5, 12), (0.95, 0.92), (0.9, 0.85))
        with pytest.raises(AttributeError):
            result.indices = (1, 2)  # type: ignore[misc]


class TestActiveLearningStats:
    """Tests for ActiveLearningStats dataclass."""

    def test_create_stats(self) -> None:
        """Create active learning stats."""
        stats = ActiveLearningStats(
            total_labeled=500,
            accuracy_history=(0.65, 0.72, 0.78),
            query_efficiency=0.015,
            labeling_cost=250.0,
        )
        assert stats.total_labeled == 500
        assert stats.query_efficiency == 0.015

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = ActiveLearningStats(500, (0.65, 0.72), 0.015, 250.0)
        with pytest.raises(AttributeError):
            stats.total_labeled = 600  # type: ignore[misc]


class TestValidateQueryConfig:
    """Tests for validate_query_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        validate_query_config(config)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            0,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_query_config(config)

    def test_negative_batch_size_raises(self) -> None:
        """Negative batch_size raises ValueError."""
        config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            -1,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_query_config(config)

    def test_diversity_weight_below_zero_raises(self) -> None:
        """Diversity weight < 0 raises ValueError."""
        config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            -0.1,
        )
        with pytest.raises(ValueError, match="diversity_weight must be between"):
            validate_query_config(config)

    def test_diversity_weight_above_one_raises(self) -> None:
        """Diversity weight > 1 raises ValueError."""
        config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            1.5,
        )
        with pytest.raises(ValueError, match="diversity_weight must be between"):
            validate_query_config(config)


class TestValidateActiveLearningConfig:
    """Tests for validate_active_learning_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = create_active_learning_config()
        validate_active_learning_config(config)

    def test_invalid_query_config_raises(self) -> None:
        """Invalid query config raises ValueError."""
        bad_query_config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            0,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        config = ActiveLearningConfig(bad_query_config, 100, 1000, 0.001)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_active_learning_config(config)

    def test_negative_initial_pool_size_raises(self) -> None:
        """Negative initial_pool_size raises ValueError."""
        query_config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        config = ActiveLearningConfig(query_config, -1, 1000, 0.001)
        with pytest.raises(ValueError, match="initial_pool_size must be non-negative"):
            validate_active_learning_config(config)

    def test_zero_labeling_budget_raises(self) -> None:
        """Zero labeling_budget raises ValueError."""
        query_config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        config = ActiveLearningConfig(query_config, 100, 0, 0.001)
        with pytest.raises(ValueError, match="labeling_budget must be positive"):
            validate_active_learning_config(config)

    def test_negative_stopping_criterion_raises(self) -> None:
        """Negative stopping_criterion raises ValueError."""
        query_config = QueryConfig(
            QueryStrategy.UNCERTAINTY,
            32,
            UncertaintyMeasure.PREDICTIVE_ENTROPY,
            0.5,
        )
        config = ActiveLearningConfig(query_config, 100, 1000, -0.001)
        with pytest.raises(ValueError, match="stopping_criterion must be non-negative"):
            validate_active_learning_config(config)


class TestValidateQueryResult:
    """Tests for validate_query_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes validation."""
        result = QueryResult(
            indices=(5, 12, 23),
            scores=(0.95, 0.92, 0.88),
            uncertainty_values=(0.9, 0.85, 0.8),
        )
        validate_query_result(result)

    def test_empty_result_valid(self) -> None:
        """Empty result passes validation."""
        result = QueryResult((), (), ())
        validate_query_result(result)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        result = QueryResult(
            indices=(5, 12),
            scores=(0.95, 0.92, 0.88),
            uncertainty_values=(0.9, 0.85, 0.8),
        )
        with pytest.raises(ValueError, match="must have same length"):
            validate_query_result(result)

    def test_negative_index_raises(self) -> None:
        """Negative index raises ValueError."""
        result = QueryResult(
            indices=(-1, 12),
            scores=(0.95, 0.92),
            uncertainty_values=(0.9, 0.85),
        )
        with pytest.raises(ValueError, match="all indices must be non-negative"):
            validate_query_result(result)


class TestValidateActiveLearningStats:
    """Tests for validate_active_learning_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = ActiveLearningStats(500, (0.65, 0.72), 0.015, 250.0)
        validate_active_learning_stats(stats)

    def test_negative_total_labeled_raises(self) -> None:
        """Negative total_labeled raises ValueError."""
        stats = ActiveLearningStats(-1, (0.65,), 0.015, 250.0)
        with pytest.raises(ValueError, match="total_labeled must be non-negative"):
            validate_active_learning_stats(stats)

    def test_negative_labeling_cost_raises(self) -> None:
        """Negative labeling_cost raises ValueError."""
        stats = ActiveLearningStats(500, (0.65,), 0.015, -10.0)
        with pytest.raises(ValueError, match="labeling_cost must be non-negative"):
            validate_active_learning_stats(stats)

    def test_accuracy_below_zero_raises(self) -> None:
        """Accuracy < 0 raises ValueError."""
        stats = ActiveLearningStats(500, (-0.1, 0.72), 0.015, 250.0)
        with pytest.raises(ValueError, match="all accuracy values must be between"):
            validate_active_learning_stats(stats)

    def test_accuracy_above_one_raises(self) -> None:
        """Accuracy > 1 raises ValueError."""
        stats = ActiveLearningStats(500, (0.65, 1.5), 0.015, 250.0)
        with pytest.raises(ValueError, match="all accuracy values must be between"):
            validate_active_learning_stats(stats)


class TestCreateQueryConfig:
    """Tests for create_query_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_query_config()
        assert config.strategy == QueryStrategy.UNCERTAINTY
        assert config.batch_size == 32
        assert config.uncertainty_measure == UncertaintyMeasure.PREDICTIVE_ENTROPY
        assert config.diversity_weight == 0.5

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_query_config(
            strategy="margin",
            batch_size=64,
            uncertainty_measure="least_confidence",
            diversity_weight=0.3,
        )
        assert config.strategy == QueryStrategy.MARGIN
        assert config.batch_size == 64
        assert config.uncertainty_measure == UncertaintyMeasure.LEAST_CONFIDENCE
        assert config.diversity_weight == 0.3

    def test_with_enum_strategy(self) -> None:
        """Create with enum strategy."""
        config = create_query_config(strategy=QueryStrategy.BALD)
        assert config.strategy == QueryStrategy.BALD

    def test_with_enum_measure(self) -> None:
        """Create with enum measure."""
        config = create_query_config(
            uncertainty_measure=UncertaintyMeasure.MUTUAL_INFORMATION
        )
        assert config.uncertainty_measure == UncertaintyMeasure.MUTUAL_INFORMATION

    @pytest.mark.parametrize(
        "strategy",
        ["uncertainty", "margin", "entropy", "bald", "coreset", "badge"],
    )
    def test_all_strategies(self, strategy: str) -> None:
        """Test all query strategies."""
        config = create_query_config(strategy=strategy)
        assert config.strategy.value == strategy

    @pytest.mark.parametrize(
        "measure",
        [
            "predictive_entropy",
            "mutual_information",
            "variation_ratio",
            "least_confidence",
        ],
    )
    def test_all_measures(self, measure: str) -> None:
        """Test all uncertainty measures."""
        config = create_query_config(uncertainty_measure=measure)
        assert config.uncertainty_measure.value == measure

    def test_invalid_batch_size_raises(self) -> None:
        """Invalid batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_query_config(batch_size=0)


class TestCreateActiveLearningConfig:
    """Tests for create_active_learning_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_active_learning_config()
        assert config.query_config.strategy == QueryStrategy.UNCERTAINTY
        assert config.initial_pool_size == 100
        assert config.labeling_budget == 1000
        assert config.stopping_criterion == 0.001

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_active_learning_config(
            strategy="bald",
            batch_size=64,
            initial_pool_size=50,
            labeling_budget=500,
            stopping_criterion=0.002,
        )
        assert config.query_config.strategy == QueryStrategy.BALD
        assert config.query_config.batch_size == 64
        assert config.initial_pool_size == 50
        assert config.labeling_budget == 500

    def test_invalid_labeling_budget_raises(self) -> None:
        """Invalid labeling_budget raises ValueError."""
        with pytest.raises(ValueError, match="labeling_budget must be positive"):
            create_active_learning_config(labeling_budget=0)

    def test_negative_initial_pool_raises(self) -> None:
        """Negative initial_pool_size raises ValueError."""
        with pytest.raises(ValueError, match="initial_pool_size must be non-negative"):
            create_active_learning_config(initial_pool_size=-1)


class TestCreateQueryResult:
    """Tests for create_query_result function."""

    def test_default_result(self) -> None:
        """Create default result."""
        result = create_query_result()
        assert result.indices == ()
        assert result.scores == ()
        assert result.uncertainty_values == ()

    def test_custom_result(self) -> None:
        """Create custom result."""
        result = create_query_result(
            indices=(5, 12, 23),
            scores=(0.95, 0.92, 0.88),
            uncertainty_values=(0.9, 0.85, 0.8),
        )
        assert result.indices == (5, 12, 23)
        assert len(result.scores) == 3

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            create_query_result(
                indices=(5, 12),
                scores=(0.95,),
                uncertainty_values=(0.9, 0.85),
            )


class TestCreateActiveLearningStats:
    """Tests for create_active_learning_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_active_learning_stats()
        assert stats.total_labeled == 0
        assert stats.accuracy_history == ()
        assert stats.query_efficiency == 0.0
        assert stats.labeling_cost == 0.0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_active_learning_stats(
            total_labeled=500,
            accuracy_history=(0.65, 0.72, 0.78),
            query_efficiency=0.015,
            labeling_cost=250.0,
        )
        assert stats.total_labeled == 500
        assert len(stats.accuracy_history) == 3

    def test_invalid_total_labeled_raises(self) -> None:
        """Invalid total_labeled raises ValueError."""
        with pytest.raises(ValueError, match="total_labeled must be non-negative"):
            create_active_learning_stats(total_labeled=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_query_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_query_strategies()
        assert strategies == sorted(strategies)
        assert "uncertainty" in strategies
        assert "bald" in strategies

    def test_list_uncertainty_measures_sorted(self) -> None:
        """Returns sorted list."""
        measures = list_uncertainty_measures()
        assert measures == sorted(measures)
        assert "predictive_entropy" in measures
        assert "mutual_information" in measures

    def test_list_pooling_methods_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_pooling_methods()
        assert methods == sorted(methods)
        assert "random" in methods
        assert "cluster" in methods


class TestGetQueryStrategy:
    """Tests for get_query_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("uncertainty", QueryStrategy.UNCERTAINTY),
            ("margin", QueryStrategy.MARGIN),
            ("entropy", QueryStrategy.ENTROPY),
            ("bald", QueryStrategy.BALD),
            ("coreset", QueryStrategy.CORESET),
            ("badge", QueryStrategy.BADGE),
        ],
    )
    def test_all_strategies(self, name: str, expected: QueryStrategy) -> None:
        """Test all valid strategies."""
        assert get_query_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_query_strategy("invalid")


class TestGetUncertaintyMeasure:
    """Tests for get_uncertainty_measure function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("predictive_entropy", UncertaintyMeasure.PREDICTIVE_ENTROPY),
            ("mutual_information", UncertaintyMeasure.MUTUAL_INFORMATION),
            ("variation_ratio", UncertaintyMeasure.VARIATION_RATIO),
            ("least_confidence", UncertaintyMeasure.LEAST_CONFIDENCE),
        ],
    )
    def test_all_measures(self, name: str, expected: UncertaintyMeasure) -> None:
        """Test all valid measures."""
        assert get_uncertainty_measure(name) == expected

    def test_invalid_measure_raises(self) -> None:
        """Invalid measure raises ValueError."""
        with pytest.raises(ValueError, match="uncertainty_measure must be one of"):
            get_uncertainty_measure("invalid")


class TestGetPoolingMethod:
    """Tests for get_pooling_method function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("random", PoolingMethod.RANDOM),
            ("cluster", PoolingMethod.CLUSTER),
            ("diversity", PoolingMethod.DIVERSITY),
        ],
    )
    def test_all_methods(self, name: str, expected: PoolingMethod) -> None:
        """Test all valid methods."""
        assert get_pooling_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="pooling_method must be one of"):
            get_pooling_method("invalid")


class TestCalculateUncertainty:
    """Tests for calculate_uncertainty function."""

    def test_maximum_uncertainty_uniform(self) -> None:
        """Maximum uncertainty for uniform distribution."""
        probs = (0.5, 0.5)
        uncertainty = calculate_uncertainty(probs)
        expected = -2 * 0.5 * math.log(0.5)
        assert uncertainty == pytest.approx(expected, rel=1e-4)

    def test_minimum_uncertainty_certain(self) -> None:
        """Minimum uncertainty for certain prediction."""
        probs = (1.0, 0.0)
        uncertainty = calculate_uncertainty(probs)
        assert uncertainty == 0.0

    def test_low_uncertainty_confident(self) -> None:
        """Low uncertainty for confident prediction."""
        probs = (0.9, 0.1)
        uncertainty = calculate_uncertainty(probs)
        # High confidence should give low entropy
        assert 0 < uncertainty < 0.5

    def test_least_confidence_measure(self) -> None:
        """Least confidence measure calculation."""
        probs = (0.7, 0.3)
        uncertainty = calculate_uncertainty(probs, UncertaintyMeasure.LEAST_CONFIDENCE)
        assert uncertainty == pytest.approx(0.3)

    def test_variation_ratio_measure(self) -> None:
        """Variation ratio measure calculation."""
        probs = (0.6, 0.4)
        uncertainty = calculate_uncertainty(probs, UncertaintyMeasure.VARIATION_RATIO)
        assert uncertainty == pytest.approx(0.4)

    def test_mutual_information_without_mc_samples(self) -> None:
        """MI without MC samples falls back to entropy."""
        probs = (0.5, 0.5)
        uncertainty = calculate_uncertainty(
            probs, UncertaintyMeasure.MUTUAL_INFORMATION
        )
        expected = -2 * 0.5 * math.log(0.5)
        assert uncertainty == pytest.approx(expected, rel=1e-4)

    def test_mutual_information_with_mc_samples(self) -> None:
        """MI with MC samples calculates properly."""
        probs = (0.5, 0.5)
        mc_samples = (
            (0.6, 0.4),
            (0.4, 0.6),
            (0.5, 0.5),
        )
        uncertainty = calculate_uncertainty(
            probs, UncertaintyMeasure.MUTUAL_INFORMATION, mc_samples
        )
        assert uncertainty >= 0

    def test_empty_probabilities_raises(self) -> None:
        """Empty probabilities raises ValueError."""
        with pytest.raises(ValueError, match="probabilities cannot be empty"):
            calculate_uncertainty(())

    def test_normalized_probabilities(self) -> None:
        """Unnormalized probabilities are normalized."""
        probs = (2.0, 2.0)  # Will be normalized to (0.5, 0.5)
        uncertainty = calculate_uncertainty(probs)
        expected = -2 * 0.5 * math.log(0.5)
        assert uncertainty == pytest.approx(expected, rel=1e-4)

    def test_zero_sum_probabilities_raises(self) -> None:
        """Zero sum probabilities raises ValueError."""
        with pytest.raises(ValueError, match="sum of probabilities must be positive"):
            calculate_uncertainty((0.0, 0.0))

    def test_multiclass_entropy(self) -> None:
        """Multi-class entropy calculation."""
        probs = (0.33, 0.33, 0.34)
        uncertainty = calculate_uncertainty(probs)
        # Close to log(3) for uniform 3-class
        assert uncertainty > 1.0


class TestSelectSamples:
    """Tests for select_samples function."""

    def test_basic_selection(self) -> None:
        """Select samples with highest uncertainty."""
        uncertainties = (0.9, 0.5, 0.8, 0.3, 0.7)
        indices = select_samples(uncertainties, 3)
        # Should select indices with highest uncertainties: 0, 2, 4
        assert indices == (0, 2, 4)

    def test_selection_with_smaller_batch(self) -> None:
        """Select fewer samples than available."""
        uncertainties = (0.9, 0.5, 0.8, 0.3, 0.7)
        indices = select_samples(uncertainties, 2)
        assert len(indices) == 2
        assert indices == (0, 2)

    def test_selection_exceeds_available(self) -> None:
        """Batch size larger than available samples."""
        uncertainties = (0.9, 0.5)
        indices = select_samples(uncertainties, 10)
        assert len(indices) == 2

    def test_selection_with_diversity(self) -> None:
        """Selection with diversity scores."""
        uncertainties = (0.5, 0.5, 0.5, 0.5)
        diversity_scores = (0.9, 0.1, 0.8, 0.2)
        indices = select_samples(
            uncertainties, 2, diversity_scores=diversity_scores, diversity_weight=0.8
        )
        # Higher diversity should be preferred
        assert 0 in indices or 2 in indices

    def test_empty_uncertainties_raises(self) -> None:
        """Empty uncertainties raises ValueError."""
        with pytest.raises(ValueError, match="uncertainties cannot be empty"):
            select_samples((), 3)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            select_samples((0.9, 0.5), 0)

    def test_negative_batch_size_raises(self) -> None:
        """Negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            select_samples((0.9, 0.5), -1)

    def test_margin_strategy(self) -> None:
        """Select samples with margin strategy."""
        uncertainties = (0.9, 0.5, 0.8)
        indices = select_samples(uncertainties, 2, strategy=QueryStrategy.MARGIN)
        assert len(indices) == 2


class TestEstimateLabelingCost:
    """Tests for estimate_labeling_cost function."""

    def test_basic_cost(self) -> None:
        """Basic cost calculation."""
        cost = estimate_labeling_cost(100)
        assert cost == 50.0

    def test_custom_cost_per_sample(self) -> None:
        """Custom cost per sample."""
        cost = estimate_labeling_cost(100, cost_per_sample=1.0)
        assert cost == 100.0

    def test_complexity_factor(self) -> None:
        """Complexity factor multiplier."""
        cost = estimate_labeling_cost(100, cost_per_sample=0.5, complexity_factor=2.0)
        assert cost == 100.0

    def test_zero_samples(self) -> None:
        """Zero samples costs nothing."""
        cost = estimate_labeling_cost(0)
        assert cost == 0.0

    def test_negative_samples_raises(self) -> None:
        """Negative samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be non-negative"):
            estimate_labeling_cost(-1)

    def test_negative_cost_per_sample_raises(self) -> None:
        """Negative cost_per_sample raises ValueError."""
        with pytest.raises(ValueError, match="cost_per_sample must be non-negative"):
            estimate_labeling_cost(100, cost_per_sample=-0.5)

    def test_negative_complexity_factor_raises(self) -> None:
        """Negative complexity_factor raises ValueError."""
        with pytest.raises(ValueError, match="complexity_factor must be non-negative"):
            estimate_labeling_cost(100, complexity_factor=-1.0)


class TestCalculateQueryEfficiency:
    """Tests for calculate_query_efficiency function."""

    def test_basic_efficiency(self) -> None:
        """Basic efficiency calculation."""
        accuracy_history = (0.5, 0.6, 0.7, 0.75)
        samples_per_round = (100, 50, 50, 50)
        efficiency = calculate_query_efficiency(accuracy_history, samples_per_round)
        # Gain of 0.25 over 150 samples (excluding initial)
        assert efficiency == pytest.approx(0.25 / 150, rel=1e-4)

    def test_no_improvement(self) -> None:
        """No accuracy improvement gives zero efficiency."""
        accuracy_history = (0.5, 0.5, 0.5)
        samples_per_round = (100, 50, 50)
        efficiency = calculate_query_efficiency(accuracy_history, samples_per_round)
        assert efficiency == 0.0

    def test_single_round(self) -> None:
        """Single round gives zero efficiency."""
        accuracy_history = (0.5,)
        samples_per_round = (100,)
        efficiency = calculate_query_efficiency(accuracy_history, samples_per_round)
        assert efficiency == 0.0

    def test_empty_history_raises(self) -> None:
        """Empty history raises ValueError."""
        with pytest.raises(ValueError, match="accuracy_history cannot be empty"):
            calculate_query_efficiency((), ())

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_query_efficiency((0.5, 0.6), (100,))

    def test_zero_samples_after_initial(self) -> None:
        """Zero samples after initial gives zero efficiency."""
        accuracy_history = (0.5, 0.6)
        samples_per_round = (100, 0)
        efficiency = calculate_query_efficiency(accuracy_history, samples_per_round)
        assert efficiency == 0.0


class TestFormatActiveLearningStats:
    """Tests for format_active_learning_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_active_learning_stats(
            total_labeled=500,
            accuracy_history=(0.65, 0.72, 0.78),
            query_efficiency=0.015,
            labeling_cost=250.0,
        )
        formatted = format_active_learning_stats(stats)
        assert "Total Labeled: 500" in formatted
        assert "Efficiency: 0.0150" in formatted
        assert "Cost: $250.00" in formatted

    def test_contains_accuracy_history(self) -> None:
        """Formatted string contains accuracy history."""
        stats = create_active_learning_stats(
            accuracy_history=(0.65, 0.72),
        )
        formatted = format_active_learning_stats(stats)
        assert "65.00%" in formatted
        assert "72.00%" in formatted

    def test_empty_accuracy_history(self) -> None:
        """Empty accuracy history shows N/A."""
        stats = create_active_learning_stats()
        formatted = format_active_learning_stats(stats)
        assert "N/A" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_active_learning_stats()
        formatted = format_active_learning_stats(stats)
        assert "Total Labeled:" in formatted
        assert "Accuracy History:" in formatted
        assert "Efficiency:" in formatted
        assert "Cost:" in formatted


class TestGetRecommendedActiveLearningConfig:
    """Tests for get_recommended_active_learning_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_active_learning_config("classification")
        assert config.query_config.strategy == QueryStrategy.UNCERTAINTY
        assert (
            config.query_config.uncertainty_measure
            == UncertaintyMeasure.PREDICTIVE_ENTROPY
        )
        assert config.query_config.diversity_weight == 0.3

    def test_ner_config(self) -> None:
        """Get config for NER task."""
        config = get_recommended_active_learning_config("ner")
        assert config.query_config.strategy == QueryStrategy.BALD
        assert (
            config.query_config.uncertainty_measure
            == UncertaintyMeasure.MUTUAL_INFORMATION
        )
        assert config.query_config.diversity_weight == 0.5

    def test_sentiment_config(self) -> None:
        """Get config for sentiment task."""
        config = get_recommended_active_learning_config("sentiment")
        assert config.query_config.strategy == QueryStrategy.MARGIN
        assert (
            config.query_config.uncertainty_measure
            == UncertaintyMeasure.LEAST_CONFIDENCE
        )

    def test_qa_config(self) -> None:
        """Get config for QA task."""
        config = get_recommended_active_learning_config("qa")
        assert config.query_config.strategy == QueryStrategy.BADGE
        assert (
            config.query_config.uncertainty_measure
            == UncertaintyMeasure.VARIATION_RATIO
        )

    def test_generation_config(self) -> None:
        """Get config for generation task."""
        config = get_recommended_active_learning_config("generation")
        assert config.query_config.strategy == QueryStrategy.CORESET
        assert config.query_config.diversity_weight == 0.7

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_active_learning_config("unknown")

    def test_data_size_affects_batch_size(self) -> None:
        """Data size affects batch size."""
        small_config = get_recommended_active_learning_config(
            "classification", data_size=1000
        )
        large_config = get_recommended_active_learning_config(
            "classification", data_size=100000
        )
        # Larger data should have larger batch size (up to limit)
        small_batch = small_config.query_config.batch_size
        large_batch = large_config.query_config.batch_size
        assert small_batch <= large_batch

    def test_data_size_affects_budget(self) -> None:
        """Data size affects labeling budget."""
        small_config = get_recommended_active_learning_config(
            "classification", data_size=1000
        )
        large_config = get_recommended_active_learning_config(
            "classification", data_size=100000
        )
        assert small_config.labeling_budget < large_config.labeling_budget

    @pytest.mark.parametrize(
        "task",
        ["classification", "ner", "sentiment", "qa", "generation"],
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All supported tasks return valid configs."""
        config = get_recommended_active_learning_config(task)
        validate_active_learning_config(config)
