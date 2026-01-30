"""Tests for training.nas module."""

from __future__ import annotations

import pytest

from hf_gtc.training.nas import (
    VALID_PERFORMANCE_PREDICTORS,
    VALID_SEARCH_SPACES,
    VALID_SEARCH_STRATEGIES,
    ArchitectureCandidate,
    NASStats,
    PerformancePredictor,
    SearchConfig,
    SearchSpace,
    SearchSpaceConfig,
    SearchStrategy,
    calculate_efficiency_score,
    calculate_search_space_size,
    create_architecture_candidate,
    create_nas_stats,
    create_search_config,
    create_search_space_config,
    estimate_architecture_cost,
    estimate_search_cost,
    evaluate_architecture,
    format_nas_stats,
    get_performance_predictor,
    get_recommended_nas_config,
    get_search_space,
    get_search_strategy,
    list_performance_predictors,
    list_search_spaces,
    list_search_strategies,
    select_pareto_optimal,
    validate_architecture_candidate,
    validate_nas_stats,
    validate_search_config,
    validate_search_space_config,
)


class TestSearchStrategy:
    """Tests for SearchStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in SearchStrategy:
            assert isinstance(strategy.value, str)

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert SearchStrategy.RANDOM.value == "random"

    def test_grid_value(self) -> None:
        """Grid has correct value."""
        assert SearchStrategy.GRID.value == "grid"

    def test_evolutionary_value(self) -> None:
        """Evolutionary has correct value."""
        assert SearchStrategy.EVOLUTIONARY.value == "evolutionary"

    def test_reinforcement_value(self) -> None:
        """Reinforcement has correct value."""
        assert SearchStrategy.REINFORCEMENT.value == "reinforcement"

    def test_differentiable_value(self) -> None:
        """Differentiable has correct value."""
        assert SearchStrategy.DIFFERENTIABLE.value == "differentiable"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_SEARCH_STRATEGIES is a frozenset."""
        assert isinstance(VALID_SEARCH_STRATEGIES, frozenset)
        assert len(VALID_SEARCH_STRATEGIES) == 5


class TestSearchSpace:
    """Tests for SearchSpace enum."""

    def test_all_spaces_have_values(self) -> None:
        """All spaces have string values."""
        for space in SearchSpace:
            assert isinstance(space.value, str)

    def test_micro_value(self) -> None:
        """Micro has correct value."""
        assert SearchSpace.MICRO.value == "micro"

    def test_macro_value(self) -> None:
        """Macro has correct value."""
        assert SearchSpace.MACRO.value == "macro"

    def test_hierarchical_value(self) -> None:
        """Hierarchical has correct value."""
        assert SearchSpace.HIERARCHICAL.value == "hierarchical"

    def test_valid_spaces_frozenset(self) -> None:
        """VALID_SEARCH_SPACES is a frozenset."""
        assert isinstance(VALID_SEARCH_SPACES, frozenset)
        assert len(VALID_SEARCH_SPACES) == 3


class TestPerformancePredictor:
    """Tests for PerformancePredictor enum."""

    def test_all_predictors_have_values(self) -> None:
        """All predictors have string values."""
        for predictor in PerformancePredictor:
            assert isinstance(predictor.value, str)

    def test_surrogate_value(self) -> None:
        """Surrogate has correct value."""
        assert PerformancePredictor.SURROGATE.value == "surrogate"

    def test_zero_cost_value(self) -> None:
        """Zero cost has correct value."""
        assert PerformancePredictor.ZERO_COST.value == "zero_cost"

    def test_weight_sharing_value(self) -> None:
        """Weight sharing has correct value."""
        assert PerformancePredictor.WEIGHT_SHARING.value == "weight_sharing"

    def test_valid_predictors_frozenset(self) -> None:
        """VALID_PERFORMANCE_PREDICTORS is a frozenset."""
        assert isinstance(VALID_PERFORMANCE_PREDICTORS, frozenset)
        assert len(VALID_PERFORMANCE_PREDICTORS) == 3


class TestSearchSpaceConfig:
    """Tests for SearchSpaceConfig dataclass."""

    def test_create_config(self) -> None:
        """Create search space config."""
        config = SearchSpaceConfig(
            space_type=SearchSpace.MICRO,
            num_layers_range=(4, 12),
            hidden_dims_range=(256, 1024),
            num_heads_range=(4, 16),
        )
        assert config.space_type == SearchSpace.MICRO
        assert config.num_layers_range == (4, 12)
        assert config.hidden_dims_range == (256, 1024)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16))
        with pytest.raises(AttributeError):
            config.space_type = SearchSpace.MACRO  # type: ignore[misc]


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_create_config(self) -> None:
        """Create search config."""
        space_config = SearchSpaceConfig(
            SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16)
        )
        config = SearchConfig(
            strategy=SearchStrategy.RANDOM,
            search_space=space_config,
            num_trials=100,
            early_stopping_patience=10,
        )
        assert config.strategy == SearchStrategy.RANDOM
        assert config.num_trials == 100
        assert config.early_stopping_patience == 10

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        space_config = SearchSpaceConfig(
            SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16)
        )
        config = SearchConfig(SearchStrategy.RANDOM, space_config, 100, 10)
        with pytest.raises(AttributeError):
            config.num_trials = 200  # type: ignore[misc]


class TestArchitectureCandidate:
    """Tests for ArchitectureCandidate dataclass."""

    def test_create_candidate(self) -> None:
        """Create architecture candidate."""
        candidate = ArchitectureCandidate(
            config={"num_layers": 6, "hidden_dim": 512},
            performance=0.92,
            cost=1.5e9,
            rank=1,
        )
        assert candidate.performance == 0.92
        assert candidate.cost == 1.5e9
        assert candidate.rank == 1
        assert candidate.config["num_layers"] == 6

    def test_candidate_is_frozen(self) -> None:
        """Candidate is immutable."""
        candidate = ArchitectureCandidate(
            config={"num_layers": 6}, performance=0.9, cost=1e9, rank=1
        )
        with pytest.raises(AttributeError):
            candidate.rank = 2  # type: ignore[misc]


class TestNASStats:
    """Tests for NASStats dataclass."""

    def test_create_stats(self) -> None:
        """Create NAS stats."""
        stats = NASStats(
            total_trials=100,
            best_performance=0.95,
            search_time_hours=24.5,
            pareto_front_size=5,
        )
        assert stats.total_trials == 100
        assert stats.best_performance == 0.95
        assert stats.search_time_hours == 24.5

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = NASStats(100, 0.95, 24.5, 5)
        with pytest.raises(AttributeError):
            stats.total_trials = 200  # type: ignore[misc]


class TestValidateSearchSpaceConfig:
    """Tests for validate_search_space_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16))
        validate_search_space_config(config)

    def test_inverted_layers_range_raises(self) -> None:
        """Inverted layers range raises ValueError."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (12, 4), (256, 1024), (4, 16))
        with pytest.raises(ValueError, match="num_layers_range min"):
            validate_search_space_config(config)

    def test_zero_min_layers_raises(self) -> None:
        """Zero min layers raises ValueError."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (0, 12), (256, 1024), (4, 16))
        with pytest.raises(ValueError, match="num_layers_range min must be positive"):
            validate_search_space_config(config)

    def test_inverted_hidden_dims_range_raises(self) -> None:
        """Inverted hidden dims range raises ValueError."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (4, 12), (1024, 256), (4, 16))
        with pytest.raises(ValueError, match="hidden_dims_range min"):
            validate_search_space_config(config)

    def test_zero_min_hidden_dims_raises(self) -> None:
        """Zero min hidden dims raises ValueError."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (4, 12), (0, 1024), (4, 16))
        with pytest.raises(ValueError, match="hidden_dims_range min must be positive"):
            validate_search_space_config(config)

    def test_inverted_heads_range_raises(self) -> None:
        """Inverted heads range raises ValueError."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (4, 12), (256, 1024), (16, 4))
        with pytest.raises(ValueError, match="num_heads_range min"):
            validate_search_space_config(config)

    def test_zero_min_heads_raises(self) -> None:
        """Zero min heads raises ValueError."""
        config = SearchSpaceConfig(SearchSpace.MICRO, (4, 12), (256, 1024), (0, 16))
        with pytest.raises(ValueError, match="num_heads_range min must be positive"):
            validate_search_space_config(config)


class TestValidateSearchConfig:
    """Tests for validate_search_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = create_search_config()
        validate_search_config(config)

    def test_zero_trials_raises(self) -> None:
        """Zero trials raises ValueError."""
        space_config = SearchSpaceConfig(
            SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16)
        )
        config = SearchConfig(SearchStrategy.RANDOM, space_config, 0, 10)
        with pytest.raises(ValueError, match="num_trials must be positive"):
            validate_search_config(config)

    def test_negative_trials_raises(self) -> None:
        """Negative trials raises ValueError."""
        space_config = SearchSpaceConfig(
            SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16)
        )
        config = SearchConfig(SearchStrategy.RANDOM, space_config, -1, 10)
        with pytest.raises(ValueError, match="num_trials must be positive"):
            validate_search_config(config)

    def test_negative_patience_raises(self) -> None:
        """Negative patience raises ValueError."""
        space_config = SearchSpaceConfig(
            SearchSpace.MICRO, (4, 12), (256, 1024), (4, 16)
        )
        config = SearchConfig(SearchStrategy.RANDOM, space_config, 100, -1)
        with pytest.raises(ValueError, match="early_stopping_patience must be"):
            validate_search_config(config)


class TestValidateArchitectureCandidate:
    """Tests for validate_architecture_candidate function."""

    def test_valid_candidate(self) -> None:
        """Valid candidate passes validation."""
        candidate = ArchitectureCandidate({"num_layers": 6}, 0.9, 1e9, 1)
        validate_architecture_candidate(candidate)

    def test_negative_performance_raises(self) -> None:
        """Negative performance raises ValueError."""
        candidate = ArchitectureCandidate({"num_layers": 6}, -0.1, 1e9, 1)
        with pytest.raises(ValueError, match="performance must be non-negative"):
            validate_architecture_candidate(candidate)

    def test_negative_cost_raises(self) -> None:
        """Negative cost raises ValueError."""
        candidate = ArchitectureCandidate({"num_layers": 6}, 0.9, -1e9, 1)
        with pytest.raises(ValueError, match="cost must be non-negative"):
            validate_architecture_candidate(candidate)

    def test_zero_rank_raises(self) -> None:
        """Zero rank raises ValueError."""
        candidate = ArchitectureCandidate({"num_layers": 6}, 0.9, 1e9, 0)
        with pytest.raises(ValueError, match="rank must be positive"):
            validate_architecture_candidate(candidate)

    def test_negative_rank_raises(self) -> None:
        """Negative rank raises ValueError."""
        candidate = ArchitectureCandidate({"num_layers": 6}, 0.9, 1e9, -1)
        with pytest.raises(ValueError, match="rank must be positive"):
            validate_architecture_candidate(candidate)


class TestValidateNASStats:
    """Tests for validate_nas_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = NASStats(100, 0.95, 24.5, 5)
        validate_nas_stats(stats)

    def test_negative_trials_raises(self) -> None:
        """Negative trials raises ValueError."""
        stats = NASStats(-1, 0.95, 24.5, 5)
        with pytest.raises(ValueError, match="total_trials must be non-negative"):
            validate_nas_stats(stats)

    def test_negative_performance_raises(self) -> None:
        """Negative performance raises ValueError."""
        stats = NASStats(100, -0.1, 24.5, 5)
        with pytest.raises(ValueError, match="best_performance must be non-negative"):
            validate_nas_stats(stats)

    def test_negative_search_time_raises(self) -> None:
        """Negative search time raises ValueError."""
        stats = NASStats(100, 0.95, -1.0, 5)
        with pytest.raises(ValueError, match="search_time_hours must be non-negative"):
            validate_nas_stats(stats)

    def test_negative_pareto_front_raises(self) -> None:
        """Negative pareto front size raises ValueError."""
        stats = NASStats(100, 0.95, 24.5, -1)
        with pytest.raises(ValueError, match="pareto_front_size must be non-negative"):
            validate_nas_stats(stats)


class TestCreateSearchSpaceConfig:
    """Tests for create_search_space_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_search_space_config()
        assert config.space_type == SearchSpace.MICRO
        assert config.num_layers_range == (4, 12)
        assert config.hidden_dims_range == (256, 1024)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_search_space_config(
            space_type="macro",
            num_layers_range=(6, 24),
            hidden_dims_range=(512, 2048),
            num_heads_range=(8, 32),
        )
        assert config.space_type == SearchSpace.MACRO
        assert config.num_layers_range == (6, 24)
        assert config.hidden_dims_range == (512, 2048)

    def test_with_enum_space_type(self) -> None:
        """Create with enum space type."""
        config = create_search_space_config(space_type=SearchSpace.HIERARCHICAL)
        assert config.space_type == SearchSpace.HIERARCHICAL

    @pytest.mark.parametrize("space_type", ["micro", "macro", "hierarchical"])
    def test_all_space_types(self, space_type: str) -> None:
        """Test all space types."""
        config = create_search_space_config(space_type=space_type)
        assert config.space_type.value == space_type

    def test_invalid_layers_range_raises(self) -> None:
        """Invalid layers range raises ValueError."""
        with pytest.raises(ValueError, match="num_layers_range min"):
            create_search_space_config(num_layers_range=(12, 4))


class TestCreateSearchConfig:
    """Tests for create_search_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_search_config()
        assert config.strategy == SearchStrategy.RANDOM
        assert config.num_trials == 100
        assert config.early_stopping_patience == 10

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_search_config(
            strategy="evolutionary",
            space_type="macro",
            num_trials=200,
            early_stopping_patience=20,
        )
        assert config.strategy == SearchStrategy.EVOLUTIONARY
        assert config.search_space.space_type == SearchSpace.MACRO
        assert config.num_trials == 200

    def test_with_enum_strategy(self) -> None:
        """Create with enum strategy."""
        config = create_search_config(strategy=SearchStrategy.DIFFERENTIABLE)
        assert config.strategy == SearchStrategy.DIFFERENTIABLE

    @pytest.mark.parametrize(
        "strategy",
        ["random", "grid", "evolutionary", "reinforcement", "differentiable"],
    )
    def test_all_strategies(self, strategy: str) -> None:
        """Test all strategies."""
        config = create_search_config(strategy=strategy)
        assert config.strategy.value == strategy

    def test_zero_trials_raises(self) -> None:
        """Zero trials raises ValueError."""
        with pytest.raises(ValueError, match="num_trials must be positive"):
            create_search_config(num_trials=0)


class TestCreateArchitectureCandidate:
    """Tests for create_architecture_candidate function."""

    def test_default_candidate(self) -> None:
        """Create default candidate."""
        candidate = create_architecture_candidate()
        assert candidate.performance == 0.0
        assert candidate.cost == 0.0
        assert candidate.rank == 1
        assert candidate.config == {}

    def test_custom_candidate(self) -> None:
        """Create custom candidate."""
        candidate = create_architecture_candidate(
            config={"num_layers": 6, "hidden_dim": 512},
            performance=0.92,
            cost=1.5e9,
            rank=3,
        )
        assert candidate.performance == 0.92
        assert candidate.cost == 1.5e9
        assert candidate.rank == 3

    def test_negative_performance_raises(self) -> None:
        """Negative performance raises ValueError."""
        with pytest.raises(ValueError, match="performance must be non-negative"):
            create_architecture_candidate(performance=-0.1)

    def test_negative_cost_raises(self) -> None:
        """Negative cost raises ValueError."""
        with pytest.raises(ValueError, match="cost must be non-negative"):
            create_architecture_candidate(cost=-1e9)

    def test_zero_rank_raises(self) -> None:
        """Zero rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            create_architecture_candidate(rank=0)


class TestCreateNASStats:
    """Tests for create_nas_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_nas_stats()
        assert stats.total_trials == 0
        assert stats.best_performance == 0.0
        assert stats.search_time_hours == 0.0
        assert stats.pareto_front_size == 0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_nas_stats(
            total_trials=100,
            best_performance=0.95,
            search_time_hours=24.5,
            pareto_front_size=5,
        )
        assert stats.total_trials == 100
        assert stats.best_performance == 0.95

    def test_negative_trials_raises(self) -> None:
        """Negative trials raises ValueError."""
        with pytest.raises(ValueError, match="total_trials must be non-negative"):
            create_nas_stats(total_trials=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_search_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_search_strategies()
        assert strategies == sorted(strategies)
        assert "random" in strategies
        assert "evolutionary" in strategies

    def test_list_search_spaces_sorted(self) -> None:
        """Returns sorted list."""
        spaces = list_search_spaces()
        assert spaces == sorted(spaces)
        assert "micro" in spaces
        assert "macro" in spaces

    def test_list_performance_predictors_sorted(self) -> None:
        """Returns sorted list."""
        predictors = list_performance_predictors()
        assert predictors == sorted(predictors)
        assert "surrogate" in predictors
        assert "zero_cost" in predictors


class TestGetSearchStrategy:
    """Tests for get_search_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("random", SearchStrategy.RANDOM),
            ("grid", SearchStrategy.GRID),
            ("evolutionary", SearchStrategy.EVOLUTIONARY),
            ("reinforcement", SearchStrategy.REINFORCEMENT),
            ("differentiable", SearchStrategy.DIFFERENTIABLE),
        ],
    )
    def test_all_strategies(self, name: str, expected: SearchStrategy) -> None:
        """Test all valid strategies."""
        assert get_search_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_search_strategy("invalid")


class TestGetSearchSpace:
    """Tests for get_search_space function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("micro", SearchSpace.MICRO),
            ("macro", SearchSpace.MACRO),
            ("hierarchical", SearchSpace.HIERARCHICAL),
        ],
    )
    def test_all_spaces(self, name: str, expected: SearchSpace) -> None:
        """Test all valid spaces."""
        assert get_search_space(name) == expected

    def test_invalid_space_raises(self) -> None:
        """Invalid space raises ValueError."""
        with pytest.raises(ValueError, match="search_space must be one of"):
            get_search_space("invalid")


class TestGetPerformancePredictor:
    """Tests for get_performance_predictor function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("surrogate", PerformancePredictor.SURROGATE),
            ("zero_cost", PerformancePredictor.ZERO_COST),
            ("weight_sharing", PerformancePredictor.WEIGHT_SHARING),
        ],
    )
    def test_all_predictors(self, name: str, expected: PerformancePredictor) -> None:
        """Test all valid predictors."""
        assert get_performance_predictor(name) == expected

    def test_invalid_predictor_raises(self) -> None:
        """Invalid predictor raises ValueError."""
        with pytest.raises(ValueError, match="performance_predictor must be one of"):
            get_performance_predictor("invalid")


class TestCalculateSearchSpaceSize:
    """Tests for calculate_search_space_size function."""

    def test_basic_calculation(self) -> None:
        """Basic search space size calculation."""
        config = create_search_space_config(
            num_layers_range=(4, 6),
            hidden_dims_range=(256, 512),
            num_heads_range=(4, 8),
        )
        # 3 * 257 * 5 = 3855
        assert calculate_search_space_size(config) == 3855

    def test_single_value_ranges(self) -> None:
        """Single value ranges give size 1."""
        config = create_search_space_config(
            num_layers_range=(6, 6),
            hidden_dims_range=(512, 512),
            num_heads_range=(8, 8),
        )
        assert calculate_search_space_size(config) == 1

    def test_larger_space(self) -> None:
        """Larger search space calculation."""
        config = create_search_space_config(
            num_layers_range=(4, 12),
            hidden_dims_range=(256, 1024),
            num_heads_range=(4, 16),
        )
        # 9 * 769 * 13 = 89973
        assert calculate_search_space_size(config) == 89973


class TestEstimateSearchCost:
    """Tests for estimate_search_cost function."""

    def test_basic_cost(self) -> None:
        """Basic cost estimation."""
        config = create_search_config(num_trials=100)
        cost = estimate_search_cost(config, hours_per_trial=0.5, gpu_cost_per_hour=1.0)
        assert cost == 50.0

    def test_higher_cost(self) -> None:
        """Higher cost estimation."""
        config = create_search_config(num_trials=50)
        cost = estimate_search_cost(config, hours_per_trial=1.0, gpu_cost_per_hour=2.0)
        assert cost == 100.0

    def test_zero_hours_raises(self) -> None:
        """Zero hours raises ValueError."""
        config = create_search_config()
        with pytest.raises(ValueError, match="hours_per_trial must be positive"):
            estimate_search_cost(config, hours_per_trial=0)

    def test_negative_hours_raises(self) -> None:
        """Negative hours raises ValueError."""
        config = create_search_config()
        with pytest.raises(ValueError, match="hours_per_trial must be positive"):
            estimate_search_cost(config, hours_per_trial=-1.0)

    def test_negative_gpu_cost_raises(self) -> None:
        """Negative GPU cost raises ValueError."""
        config = create_search_config()
        with pytest.raises(ValueError, match="gpu_cost_per_hour must be non-negative"):
            estimate_search_cost(config, gpu_cost_per_hour=-1.0)


class TestEvaluateArchitecture:
    """Tests for evaluate_architecture function."""

    def test_basic_evaluation(self) -> None:
        """Basic architecture evaluation."""
        result = evaluate_architecture(
            {"num_layers": 6, "hidden_dim": 512},
            performance_metric=0.92,
            cost_metric=1.5e9,
        )
        assert result.performance == 0.92
        assert result.cost == 1.5e9
        assert result.rank == 1
        assert result.config["num_layers"] == 6

    def test_empty_config(self) -> None:
        """Evaluation with empty config."""
        result = evaluate_architecture({}, 0.8, 1e9)
        assert result.config == {}
        assert result.performance == 0.8

    def test_negative_performance_raises(self) -> None:
        """Negative performance raises ValueError."""
        with pytest.raises(ValueError, match="performance_metric must be non-negative"):
            evaluate_architecture({}, performance_metric=-0.1, cost_metric=1.0)

    def test_negative_cost_raises(self) -> None:
        """Negative cost raises ValueError."""
        with pytest.raises(ValueError, match="cost_metric must be non-negative"):
            evaluate_architecture({}, performance_metric=0.9, cost_metric=-1.0)


class TestSelectParetoOptimal:
    """Tests for select_pareto_optimal function."""

    def test_basic_pareto_selection(self) -> None:
        """Basic Pareto optimal selection."""
        c1 = create_architecture_candidate(
            config={"id": 1}, performance=0.9, cost=100, rank=1
        )
        c2 = create_architecture_candidate(
            config={"id": 2}, performance=0.85, cost=80, rank=2
        )
        c3 = create_architecture_candidate(
            config={"id": 3}, performance=0.8, cost=120, rank=3
        )
        pareto = select_pareto_optimal((c1, c2, c3))
        assert len(pareto) == 2
        ids = sorted([c.config["id"] for c in pareto])
        assert ids == [1, 2]

    def test_single_candidate(self) -> None:
        """Single candidate is always Pareto optimal."""
        c = create_architecture_candidate(
            config={"id": 1}, performance=0.9, cost=100, rank=1
        )
        pareto = select_pareto_optimal((c,))
        assert len(pareto) == 1
        assert pareto[0].rank == 1

    def test_all_pareto_optimal(self) -> None:
        """All candidates can be Pareto optimal."""
        c1 = create_architecture_candidate(
            config={"id": 1}, performance=0.9, cost=100, rank=1
        )
        c2 = create_architecture_candidate(
            config={"id": 2}, performance=0.8, cost=50, rank=2
        )
        c3 = create_architecture_candidate(
            config={"id": 3}, performance=0.7, cost=25, rank=3
        )
        pareto = select_pareto_optimal((c1, c2, c3))
        assert len(pareto) == 3

    def test_ranks_are_updated(self) -> None:
        """Ranks are updated after selection."""
        c1 = create_architecture_candidate(
            config={"id": 1}, performance=0.9, cost=100, rank=10
        )
        c2 = create_architecture_candidate(
            config={"id": 2}, performance=0.8, cost=50, rank=20
        )
        pareto = select_pareto_optimal((c1, c2))
        assert pareto[0].rank == 1
        assert pareto[1].rank == 2

    def test_empty_candidates_raises(self) -> None:
        """Empty candidates raises ValueError."""
        with pytest.raises(ValueError, match="candidates cannot be empty"):
            select_pareto_optimal(())


class TestFormatNASStats:
    """Tests for format_nas_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_nas_stats(
            total_trials=100,
            best_performance=0.95,
            search_time_hours=24.5,
            pareto_front_size=5,
        )
        formatted = format_nas_stats(stats)
        assert "Total Trials: 100" in formatted
        assert "Best Performance: 95.0%" in formatted
        assert "Search Time: 24.5 hours" in formatted
        assert "Pareto Front Size: 5" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_nas_stats()
        formatted = format_nas_stats(stats)
        assert "Total Trials:" in formatted
        assert "Best Performance:" in formatted
        assert "Search Time:" in formatted
        assert "Pareto Front Size:" in formatted


class TestGetRecommendedNASConfig:
    """Tests for get_recommended_nas_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_nas_config("classification")
        assert config.strategy == SearchStrategy.EVOLUTIONARY
        assert config.search_space.space_type == SearchSpace.MICRO

    def test_generation_config(self) -> None:
        """Get config for generation task."""
        config = get_recommended_nas_config("generation")
        assert config.strategy == SearchStrategy.DIFFERENTIABLE
        assert config.search_space.space_type == SearchSpace.HIERARCHICAL

    def test_translation_config(self) -> None:
        """Get config for translation task."""
        config = get_recommended_nas_config("translation")
        assert config.strategy == SearchStrategy.REINFORCEMENT
        assert config.search_space.space_type == SearchSpace.MACRO

    def test_summarization_config(self) -> None:
        """Get config for summarization task."""
        config = get_recommended_nas_config("summarization")
        assert config.strategy == SearchStrategy.EVOLUTIONARY
        assert config.search_space.space_type == SearchSpace.HIERARCHICAL

    def test_qa_config(self) -> None:
        """Get config for QA task."""
        config = get_recommended_nas_config("qa")
        assert config.strategy == SearchStrategy.RANDOM
        assert config.search_space.space_type == SearchSpace.MICRO

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_nas_config("unknown")

    @pytest.mark.parametrize(
        "task",
        ["classification", "generation", "translation", "summarization", "qa"],
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All supported tasks return valid configs."""
        config = get_recommended_nas_config(task)
        validate_search_config(config)


class TestEstimateArchitectureCost:
    """Tests for estimate_architecture_cost function."""

    def test_basic_cost(self) -> None:
        """Basic cost estimation."""
        cost = estimate_architecture_cost(6, 512, 8)
        assert cost > 0

    def test_larger_model_higher_cost(self) -> None:
        """Larger model has higher cost."""
        cost_small = estimate_architecture_cost(6, 256, 4)
        cost_large = estimate_architecture_cost(12, 512, 8)
        assert cost_small < cost_large

    def test_more_layers_higher_cost(self) -> None:
        """More layers means higher cost."""
        cost_6 = estimate_architecture_cost(6, 512, 8)
        cost_12 = estimate_architecture_cost(12, 512, 8)
        assert cost_6 < cost_12

    def test_zero_layers_raises(self) -> None:
        """Zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_architecture_cost(0, 512, 8)

    def test_zero_hidden_dim_raises(self) -> None:
        """Zero hidden dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            estimate_architecture_cost(6, 0, 8)

    def test_zero_num_heads_raises(self) -> None:
        """Zero num heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            estimate_architecture_cost(6, 512, 0)

    def test_zero_sequence_length_raises(self) -> None:
        """Zero sequence length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_architecture_cost(6, 512, 8, sequence_length=0)


class TestCalculateEfficiencyScore:
    """Tests for calculate_efficiency_score function."""

    def test_basic_efficiency(self) -> None:
        """Basic efficiency calculation."""
        score = calculate_efficiency_score(0.9, 1e9)
        assert 0 < score < 1

    def test_efficient_is_better(self) -> None:
        """More efficient model scores better."""
        score_efficient = calculate_efficiency_score(0.85, 1e8)
        score_costly = calculate_efficiency_score(0.9, 1e10)
        assert score_efficient > score_costly

    def test_perfect_performance_high_cost(self) -> None:
        """Perfect performance with high cost."""
        score = calculate_efficiency_score(1.0, 1e12)
        assert 0 < score < 1

    def test_performance_above_one_raises(self) -> None:
        """Performance above 1 raises ValueError."""
        with pytest.raises(ValueError, match="performance must be between 0 and 1"):
            calculate_efficiency_score(1.5, 1e9)

    def test_negative_performance_raises(self) -> None:
        """Negative performance raises ValueError."""
        with pytest.raises(ValueError, match="performance must be between 0 and 1"):
            calculate_efficiency_score(-0.1, 1e9)

    def test_zero_cost_raises(self) -> None:
        """Zero cost raises ValueError."""
        with pytest.raises(ValueError, match="cost must be positive"):
            calculate_efficiency_score(0.9, 0)

    def test_negative_cost_raises(self) -> None:
        """Negative cost raises ValueError."""
        with pytest.raises(ValueError, match="cost must be positive"):
            calculate_efficiency_score(0.9, -1e9)

    def test_invalid_weight_raises(self) -> None:
        """Invalid weight raises ValueError."""
        with pytest.raises(ValueError, match="performance_weight must be between"):
            calculate_efficiency_score(0.9, 1e9, performance_weight=1.5)

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="performance_weight must be between"):
            calculate_efficiency_score(0.9, 1e9, performance_weight=-0.1)
