"""Tests for training.hyperopt module."""

from __future__ import annotations

import pytest

from hf_gtc.training.hyperopt import (
    VALID_OBJECTIVE_DIRECTIONS,
    VALID_PARAMETER_TYPES,
    VALID_SEARCH_ALGORITHMS,
    HyperoptConfig,
    HyperoptStats,
    ObjectiveDirection,
    ParameterSpace,
    ParameterType,
    SearchAlgorithm,
    TrialResult,
    calculate_improvement,
    create_hyperopt_config,
    create_hyperopt_stats,
    create_parameter_space,
    create_trial_result,
    estimate_remaining_trials,
    format_hyperopt_stats,
    get_objective_direction,
    get_parameter_type,
    get_recommended_hyperopt_config,
    get_search_algorithm,
    list_objective_directions,
    list_parameter_types,
    list_search_algorithms,
    sample_parameters,
    suggest_next_params,
    validate_hyperopt_config,
    validate_hyperopt_stats,
    validate_parameter_space,
    validate_trial_result,
)


class TestSearchAlgorithm:
    """Tests for SearchAlgorithm enum."""

    def test_all_algorithms_have_values(self) -> None:
        """All algorithms have string values."""
        for algorithm in SearchAlgorithm:
            assert isinstance(algorithm.value, str)

    def test_grid_value(self) -> None:
        """Grid has correct value."""
        assert SearchAlgorithm.GRID.value == "grid"

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert SearchAlgorithm.RANDOM.value == "random"

    def test_bayesian_value(self) -> None:
        """Bayesian has correct value."""
        assert SearchAlgorithm.BAYESIAN.value == "bayesian"

    def test_tpe_value(self) -> None:
        """TPE has correct value."""
        assert SearchAlgorithm.TPE.value == "tpe"

    def test_cmaes_value(self) -> None:
        """CMA-ES has correct value."""
        assert SearchAlgorithm.CMAES.value == "cmaes"

    def test_hyperband_value(self) -> None:
        """Hyperband has correct value."""
        assert SearchAlgorithm.HYPERBAND.value == "hyperband"

    def test_valid_algorithms_frozenset(self) -> None:
        """VALID_SEARCH_ALGORITHMS is a frozenset."""
        assert isinstance(VALID_SEARCH_ALGORITHMS, frozenset)
        assert len(VALID_SEARCH_ALGORITHMS) == 6


class TestParameterType:
    """Tests for ParameterType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for param_type in ParameterType:
            assert isinstance(param_type.value, str)

    def test_continuous_value(self) -> None:
        """Continuous has correct value."""
        assert ParameterType.CONTINUOUS.value == "continuous"

    def test_discrete_value(self) -> None:
        """Discrete has correct value."""
        assert ParameterType.DISCRETE.value == "discrete"

    def test_categorical_value(self) -> None:
        """Categorical has correct value."""
        assert ParameterType.CATEGORICAL.value == "categorical"

    def test_log_uniform_value(self) -> None:
        """Log uniform has correct value."""
        assert ParameterType.LOG_UNIFORM.value == "log_uniform"

    def test_valid_types_frozenset(self) -> None:
        """VALID_PARAMETER_TYPES is a frozenset."""
        assert isinstance(VALID_PARAMETER_TYPES, frozenset)
        assert len(VALID_PARAMETER_TYPES) == 4


class TestObjectiveDirection:
    """Tests for ObjectiveDirection enum."""

    def test_all_directions_have_values(self) -> None:
        """All directions have string values."""
        for direction in ObjectiveDirection:
            assert isinstance(direction.value, str)

    def test_minimize_value(self) -> None:
        """Minimize has correct value."""
        assert ObjectiveDirection.MINIMIZE.value == "minimize"

    def test_maximize_value(self) -> None:
        """Maximize has correct value."""
        assert ObjectiveDirection.MAXIMIZE.value == "maximize"

    def test_valid_directions_frozenset(self) -> None:
        """VALID_OBJECTIVE_DIRECTIONS is a frozenset."""
        assert isinstance(VALID_OBJECTIVE_DIRECTIONS, frozenset)
        assert len(VALID_OBJECTIVE_DIRECTIONS) == 2


class TestParameterSpace:
    """Tests for ParameterSpace dataclass."""

    def test_create_continuous_space(self) -> None:
        """Create continuous parameter space."""
        space = ParameterSpace(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            low=0.0,
            high=1.0,
            choices=None,
        )
        assert space.name == "learning_rate"
        assert space.param_type == ParameterType.CONTINUOUS
        assert space.low == pytest.approx(0.0)
        assert space.high == pytest.approx(1.0)

    def test_create_categorical_space(self) -> None:
        """Create categorical parameter space."""
        space = ParameterSpace(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            low=None,
            high=None,
            choices=("adam", "sgd"),
        )
        assert space.name == "optimizer"
        assert space.choices == ("adam", "sgd")

    def test_space_is_frozen(self) -> None:
        """Space is immutable."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        with pytest.raises(AttributeError):
            space.low = 0.1  # type: ignore[misc]


class TestHyperoptConfig:
    """Tests for HyperoptConfig dataclass."""

    def test_create_config(self) -> None:
        """Create hyperopt config."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            algorithm=SearchAlgorithm.RANDOM,
            parameter_spaces=(space,),
            n_trials=100,
            timeout_seconds=3600,
            direction=ObjectiveDirection.MINIMIZE,
        )
        assert config.algorithm == SearchAlgorithm.RANDOM
        assert config.n_trials == 100
        assert config.direction == ObjectiveDirection.MINIMIZE

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (space,), 100, 3600, ObjectiveDirection.MINIMIZE
        )
        with pytest.raises(AttributeError):
            config.n_trials = 50  # type: ignore[misc]


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_create_result(self) -> None:
        """Create trial result."""
        result = TrialResult(
            params={"lr": 0.001},
            objective_value=0.05,
            trial_number=1,
            duration_seconds=120.5,
        )
        assert result.params == {"lr": 0.001}
        assert result.objective_value == pytest.approx(0.05)
        assert result.trial_number == 1
        assert result.duration_seconds == pytest.approx(120.5)

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = TrialResult({"lr": 0.001}, 0.05, 1, 120.5)
        with pytest.raises(AttributeError):
            result.objective_value = 0.1  # type: ignore[misc]


class TestHyperoptStats:
    """Tests for HyperoptStats dataclass."""

    def test_create_stats(self) -> None:
        """Create hyperopt stats."""
        stats = HyperoptStats(
            total_trials=50,
            best_value=0.02,
            best_params={"lr": 0.0005},
            convergence_curve=(0.1, 0.05, 0.02),
        )
        assert stats.total_trials == 50
        assert stats.best_value == pytest.approx(0.02)
        assert stats.best_params == {"lr": 0.0005}
        assert stats.convergence_curve == (0.1, 0.05, 0.02)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = HyperoptStats(50, 0.02, {"lr": 0.0005}, (0.1, 0.05, 0.02))
        with pytest.raises(AttributeError):
            stats.total_trials = 100  # type: ignore[misc]


class TestValidateParameterSpace:
    """Tests for validate_parameter_space function."""

    def test_valid_continuous_space(self) -> None:
        """Valid continuous space passes validation."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        validate_parameter_space(space)

    def test_valid_categorical_space(self) -> None:
        """Valid categorical space passes validation."""
        space = ParameterSpace(
            "opt", ParameterType.CATEGORICAL, None, None, ("adam", "sgd")
        )
        validate_parameter_space(space)

    def test_valid_log_uniform_space(self) -> None:
        """Valid log_uniform space passes validation."""
        space = ParameterSpace("lr", ParameterType.LOG_UNIFORM, 1e-5, 1e-1, None)
        validate_parameter_space(space)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        space = ParameterSpace("", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_parameter_space(space)

    def test_categorical_without_choices_raises(self) -> None:
        """Categorical without choices raises ValueError."""
        space = ParameterSpace("opt", ParameterType.CATEGORICAL, None, None, None)
        with pytest.raises(ValueError, match="categorical parameter must have"):
            validate_parameter_space(space)

    def test_categorical_with_empty_choices_raises(self) -> None:
        """Categorical with empty choices raises ValueError."""
        space = ParameterSpace("opt", ParameterType.CATEGORICAL, None, None, ())
        with pytest.raises(ValueError, match="categorical parameter must have"):
            validate_parameter_space(space)

    def test_continuous_without_bounds_raises(self) -> None:
        """Continuous without bounds raises ValueError."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, None, None, None)
        with pytest.raises(ValueError, match="must have low and high bounds"):
            validate_parameter_space(space)

    def test_low_equals_high_raises(self) -> None:
        """Low equals high raises ValueError."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.5, 0.5, None)
        with pytest.raises(ValueError, match=r"low .* must be less than high"):
            validate_parameter_space(space)

    def test_low_greater_than_high_raises(self) -> None:
        """Low greater than high raises ValueError."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 1.0, 0.5, None)
        with pytest.raises(ValueError, match=r"low .* must be less than high"):
            validate_parameter_space(space)

    def test_log_uniform_with_zero_low_raises(self) -> None:
        """Log uniform with zero low raises ValueError."""
        space = ParameterSpace("lr", ParameterType.LOG_UNIFORM, 0.0, 1.0, None)
        with pytest.raises(
            ValueError, match="log_uniform parameter low must be positive"
        ):
            validate_parameter_space(space)

    def test_log_uniform_with_negative_low_raises(self) -> None:
        """Log uniform with negative low raises ValueError."""
        space = ParameterSpace("lr", ParameterType.LOG_UNIFORM, -0.1, 1.0, None)
        with pytest.raises(
            ValueError, match="log_uniform parameter low must be positive"
        ):
            validate_parameter_space(space)


class TestValidateHyperoptConfig:
    """Tests for validate_hyperopt_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (space,), 100, 3600, ObjectiveDirection.MINIMIZE
        )
        validate_hyperopt_config(config)

    def test_empty_spaces_raises(self) -> None:
        """Empty parameter spaces raises ValueError."""
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (), 100, 3600, ObjectiveDirection.MINIMIZE
        )
        with pytest.raises(ValueError, match="parameter_spaces cannot be empty"):
            validate_hyperopt_config(config)

    def test_zero_n_trials_raises(self) -> None:
        """Zero n_trials raises ValueError."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (space,), 0, 3600, ObjectiveDirection.MINIMIZE
        )
        with pytest.raises(ValueError, match="n_trials must be positive"):
            validate_hyperopt_config(config)

    def test_negative_n_trials_raises(self) -> None:
        """Negative n_trials raises ValueError."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (space,), -1, 3600, ObjectiveDirection.MINIMIZE
        )
        with pytest.raises(ValueError, match="n_trials must be positive"):
            validate_hyperopt_config(config)

    def test_zero_timeout_raises(self) -> None:
        """Zero timeout raises ValueError."""
        space = ParameterSpace("lr", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (space,), 100, 0, ObjectiveDirection.MINIMIZE
        )
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_hyperopt_config(config)

    def test_invalid_space_raises(self) -> None:
        """Invalid space in config raises ValueError."""
        space = ParameterSpace("", ParameterType.CONTINUOUS, 0.0, 1.0, None)
        config = HyperoptConfig(
            SearchAlgorithm.RANDOM, (space,), 100, 3600, ObjectiveDirection.MINIMIZE
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_hyperopt_config(config)


class TestValidateTrialResult:
    """Tests for validate_trial_result function."""

    def test_valid_result(self) -> None:
        """Valid result passes validation."""
        result = TrialResult({"lr": 0.001}, 0.05, 1, 120.5)
        validate_trial_result(result)

    def test_empty_params_raises(self) -> None:
        """Empty params raises ValueError."""
        result = TrialResult({}, 0.05, 1, 120.5)
        with pytest.raises(ValueError, match="params cannot be empty"):
            validate_trial_result(result)

    def test_zero_trial_number_raises(self) -> None:
        """Zero trial_number raises ValueError."""
        result = TrialResult({"lr": 0.001}, 0.05, 0, 120.5)
        with pytest.raises(ValueError, match="trial_number must be positive"):
            validate_trial_result(result)

    def test_negative_trial_number_raises(self) -> None:
        """Negative trial_number raises ValueError."""
        result = TrialResult({"lr": 0.001}, 0.05, -1, 120.5)
        with pytest.raises(ValueError, match="trial_number must be positive"):
            validate_trial_result(result)

    def test_negative_duration_raises(self) -> None:
        """Negative duration raises ValueError."""
        result = TrialResult({"lr": 0.001}, 0.05, 1, -1.0)
        with pytest.raises(ValueError, match="duration_seconds must be non-negative"):
            validate_trial_result(result)


class TestValidateHyperoptStats:
    """Tests for validate_hyperopt_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = HyperoptStats(50, 0.02, {"lr": 0.0005}, (0.1, 0.05, 0.02))
        validate_hyperopt_stats(stats)

    def test_negative_total_trials_raises(self) -> None:
        """Negative total_trials raises ValueError."""
        stats = HyperoptStats(-1, 0.02, {"lr": 0.0005}, (0.1,))
        with pytest.raises(ValueError, match="total_trials must be non-negative"):
            validate_hyperopt_stats(stats)

    def test_empty_best_params_raises(self) -> None:
        """Empty best_params raises ValueError."""
        stats = HyperoptStats(50, 0.02, {}, (0.1,))
        with pytest.raises(ValueError, match="best_params cannot be empty"):
            validate_hyperopt_stats(stats)


class TestCreateParameterSpace:
    """Tests for create_parameter_space function."""

    def test_create_continuous_space(self) -> None:
        """Create continuous space."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        assert space.name == "lr"
        assert space.param_type == ParameterType.CONTINUOUS
        assert space.low == pytest.approx(0.0)
        assert space.high == pytest.approx(1.0)

    def test_create_discrete_space(self) -> None:
        """Create discrete space."""
        space = create_parameter_space("batch_size", "discrete", 8, 128)
        assert space.param_type == ParameterType.DISCRETE
        assert space.low == 8
        assert space.high == 128

    def test_create_categorical_space(self) -> None:
        """Create categorical space."""
        space = create_parameter_space("opt", "categorical", choices=("adam", "sgd"))
        assert space.param_type == ParameterType.CATEGORICAL
        assert space.choices == ("adam", "sgd")

    def test_create_log_uniform_space(self) -> None:
        """Create log uniform space."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        assert space.param_type == ParameterType.LOG_UNIFORM
        assert space.low == 1e-5
        assert space.high == 1e-1

    def test_create_with_enum_type(self) -> None:
        """Create with enum type."""
        space = create_parameter_space("lr", ParameterType.CONTINUOUS, 0.0, 1.0)
        assert space.param_type == ParameterType.CONTINUOUS

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_parameter_space("", "continuous", 0.0, 1.0)

    @pytest.mark.parametrize(
        "param_type",
        ["continuous", "discrete", "categorical", "log_uniform"],
    )
    def test_all_types(self, param_type: str) -> None:
        """Test all parameter types."""
        if param_type == "categorical":
            space = create_parameter_space("p", param_type, choices=("a", "b"))
        elif param_type == "log_uniform":
            space = create_parameter_space("p", param_type, 0.001, 1.0)
        else:
            space = create_parameter_space("p", param_type, 0.0, 1.0)
        assert space.param_type.value == param_type


class TestCreateHyperoptConfig:
    """Tests for create_hyperopt_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_hyperopt_config()
        assert config.algorithm == SearchAlgorithm.RANDOM
        assert config.n_trials == 100
        assert config.timeout_seconds == 3600
        assert config.direction == ObjectiveDirection.MINIMIZE
        assert len(config.parameter_spaces) > 0

    def test_custom_config(self) -> None:
        """Create custom config."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
            n_trials=50,
            timeout_seconds=1800,
            direction="maximize",
        )
        assert config.algorithm == SearchAlgorithm.BAYESIAN
        assert config.n_trials == 50
        assert config.direction == ObjectiveDirection.MAXIMIZE

    def test_with_enum_algorithm(self) -> None:
        """Create with enum algorithm."""
        config = create_hyperopt_config(algorithm=SearchAlgorithm.TPE)
        assert config.algorithm == SearchAlgorithm.TPE

    def test_with_enum_direction(self) -> None:
        """Create with enum direction."""
        config = create_hyperopt_config(direction=ObjectiveDirection.MAXIMIZE)
        assert config.direction == ObjectiveDirection.MAXIMIZE

    def test_zero_n_trials_raises(self) -> None:
        """Zero n_trials raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            create_hyperopt_config(n_trials=0)

    @pytest.mark.parametrize(
        "algorithm",
        ["grid", "random", "bayesian", "tpe", "cmaes", "hyperband"],
    )
    def test_all_algorithms(self, algorithm: str) -> None:
        """Test all search algorithms."""
        config = create_hyperopt_config(algorithm=algorithm)
        assert config.algorithm.value == algorithm


class TestCreateTrialResult:
    """Tests for create_trial_result function."""

    def test_create_result(self) -> None:
        """Create trial result."""
        result = create_trial_result(
            params={"lr": 0.001, "batch_size": 32},
            objective_value=0.05,
            trial_number=1,
        )
        assert result.params == {"lr": 0.001, "batch_size": 32}
        assert result.objective_value == pytest.approx(0.05)
        assert result.trial_number == 1
        assert result.duration_seconds == pytest.approx(0.0)

    def test_with_duration(self) -> None:
        """Create result with duration."""
        result = create_trial_result(
            params={"lr": 0.001},
            objective_value=0.05,
            trial_number=1,
            duration_seconds=120.5,
        )
        assert result.duration_seconds == pytest.approx(120.5)

    def test_empty_params_raises(self) -> None:
        """Empty params raises ValueError."""
        with pytest.raises(ValueError, match="params cannot be empty"):
            create_trial_result({}, 0.05, 1)


class TestCreateHyperoptStats:
    """Tests for create_hyperopt_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_hyperopt_stats()
        assert stats.total_trials == 0
        assert stats.best_value == float("inf")
        assert stats.convergence_curve == ()

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_hyperopt_stats(
            total_trials=50,
            best_value=0.02,
            best_params={"lr": 0.0005},
            convergence_curve=(0.1, 0.05, 0.02),
        )
        assert stats.total_trials == 50
        assert stats.best_value == pytest.approx(0.02)
        assert stats.best_params == {"lr": 0.0005}

    def test_negative_total_trials_raises(self) -> None:
        """Negative total_trials raises ValueError."""
        with pytest.raises(ValueError, match="total_trials must be non-negative"):
            create_hyperopt_stats(total_trials=-1, best_params={"lr": 0.001})


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_search_algorithms_sorted(self) -> None:
        """Returns sorted list."""
        algorithms = list_search_algorithms()
        assert algorithms == sorted(algorithms)
        assert "random" in algorithms
        assert "bayesian" in algorithms

    def test_list_parameter_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_parameter_types()
        assert types == sorted(types)
        assert "continuous" in types
        assert "categorical" in types

    def test_list_objective_directions_sorted(self) -> None:
        """Returns sorted list."""
        directions = list_objective_directions()
        assert directions == sorted(directions)
        assert "minimize" in directions
        assert "maximize" in directions


class TestGetSearchAlgorithm:
    """Tests for get_search_algorithm function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("grid", SearchAlgorithm.GRID),
            ("random", SearchAlgorithm.RANDOM),
            ("bayesian", SearchAlgorithm.BAYESIAN),
            ("tpe", SearchAlgorithm.TPE),
            ("cmaes", SearchAlgorithm.CMAES),
            ("hyperband", SearchAlgorithm.HYPERBAND),
        ],
    )
    def test_all_algorithms(self, name: str, expected: SearchAlgorithm) -> None:
        """Test all valid algorithms."""
        assert get_search_algorithm(name) == expected

    def test_invalid_algorithm_raises(self) -> None:
        """Invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            get_search_algorithm("invalid")


class TestGetParameterType:
    """Tests for get_parameter_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("continuous", ParameterType.CONTINUOUS),
            ("discrete", ParameterType.DISCRETE),
            ("categorical", ParameterType.CATEGORICAL),
            ("log_uniform", ParameterType.LOG_UNIFORM),
        ],
    )
    def test_all_types(self, name: str, expected: ParameterType) -> None:
        """Test all valid types."""
        assert get_parameter_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="parameter_type must be one of"):
            get_parameter_type("invalid")


class TestGetObjectiveDirection:
    """Tests for get_objective_direction function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("minimize", ObjectiveDirection.MINIMIZE),
            ("maximize", ObjectiveDirection.MAXIMIZE),
        ],
    )
    def test_all_directions(self, name: str, expected: ObjectiveDirection) -> None:
        """Test all valid directions."""
        assert get_objective_direction(name) == expected

    def test_invalid_direction_raises(self) -> None:
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be one of"):
            get_objective_direction("invalid")


class TestSampleParameters:
    """Tests for sample_parameters function."""

    def test_sample_continuous(self) -> None:
        """Sample continuous parameter."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        params = sample_parameters((space,), seed=42)
        assert "lr" in params
        assert 0.0 <= params["lr"] <= 1.0

    def test_sample_discrete(self) -> None:
        """Sample discrete parameter."""
        space = create_parameter_space("batch_size", "discrete", 8, 128)
        params = sample_parameters((space,), seed=42)
        assert "batch_size" in params
        assert 8 <= params["batch_size"] <= 128
        assert isinstance(params["batch_size"], int)

    def test_sample_categorical(self) -> None:
        """Sample categorical parameter."""
        space = create_parameter_space("opt", "categorical", choices=("adam", "sgd"))
        params = sample_parameters((space,), seed=42)
        assert "opt" in params
        assert params["opt"] in ("adam", "sgd")

    def test_sample_log_uniform(self) -> None:
        """Sample log uniform parameter."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        params = sample_parameters((space,), seed=42)
        assert "lr" in params
        assert 1e-5 <= params["lr"] <= 1e-1

    def test_sample_multiple_spaces(self) -> None:
        """Sample from multiple spaces."""
        spaces = (
            create_parameter_space("lr", "continuous", 0.0, 1.0),
            create_parameter_space("batch_size", "discrete", 8, 128),
        )
        params = sample_parameters(spaces, seed=42)
        assert "lr" in params
        assert "batch_size" in params

    def test_reproducible_with_seed(self) -> None:
        """Samples are reproducible with same seed."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        params1 = sample_parameters((space,), seed=42)
        params2 = sample_parameters((space,), seed=42)
        assert params1["lr"] == params2["lr"]

    def test_empty_spaces_raises(self) -> None:
        """Empty spaces raises ValueError."""
        with pytest.raises(ValueError, match="spaces cannot be empty"):
            sample_parameters(())


class TestCalculateImprovement:
    """Tests for calculate_improvement function."""

    def test_improvement_minimize(self) -> None:
        """Calculate improvement for minimization."""
        assert calculate_improvement(
            0.05, 0.1, ObjectiveDirection.MINIMIZE
        ) == pytest.approx(0.5)
        assert calculate_improvement(
            0.15, 0.1, ObjectiveDirection.MINIMIZE
        ) == pytest.approx(-0.5)

    def test_improvement_maximize(self) -> None:
        """Calculate improvement for maximization."""
        assert calculate_improvement(
            0.9, 0.8, ObjectiveDirection.MAXIMIZE
        ) == pytest.approx(0.125)
        assert calculate_improvement(
            0.7, 0.8, ObjectiveDirection.MAXIMIZE
        ) == pytest.approx(-0.125)

    def test_zero_best_value(self) -> None:
        """Zero best value returns 0."""
        assert calculate_improvement(
            0.5, 0.0, ObjectiveDirection.MINIMIZE
        ) == pytest.approx(0.0)
        assert calculate_improvement(
            0.5, 0.0, ObjectiveDirection.MAXIMIZE
        ) == pytest.approx(0.0)

    def test_no_improvement(self) -> None:
        """Same value returns 0 improvement."""
        assert calculate_improvement(
            0.1, 0.1, ObjectiveDirection.MINIMIZE
        ) == pytest.approx(0.0)


class TestEstimateRemainingTrials:
    """Tests for estimate_remaining_trials function."""

    def test_basic_estimate(self) -> None:
        """Basic estimate with plenty of time."""
        remaining = estimate_remaining_trials(100.0, 10, 100, 3600)
        assert remaining == 90

    def test_time_limited(self) -> None:
        """Estimate when time-limited."""
        # 1800 seconds elapsed, 50 trials done
        # Avg 36 seconds per trial
        # 1800 seconds remaining = 50 more trials
        remaining = estimate_remaining_trials(1800.0, 50, 100, 3600)
        assert remaining == 50

    def test_near_timeout(self) -> None:
        """Estimate near timeout - time-limited to 2 trials."""
        # 3500 seconds elapsed, 90 trials done
        # Avg time = 3500/90 = 38.89 seconds per trial
        # Remaining time = 100 seconds
        # Remaining by time = 100 / 38.89 = 2 trials
        # Remaining by count = 10 trials
        # Result = min(10, 2) = 2
        remaining = estimate_remaining_trials(3500.0, 90, 100, 3600)
        assert remaining == 2

    def test_timeout_reached(self) -> None:
        """Returns 0 when timeout reached."""
        remaining = estimate_remaining_trials(3600.0, 50, 100, 3600)
        assert remaining == 0

    def test_no_trials_completed(self) -> None:
        """Returns full count when no trials completed."""
        remaining = estimate_remaining_trials(0.0, 0, 100, 3600)
        assert remaining == 100

    def test_negative_elapsed_raises(self) -> None:
        """Negative elapsed raises ValueError."""
        with pytest.raises(ValueError, match="elapsed_seconds must be non-negative"):
            estimate_remaining_trials(-1.0, 10, 100, 3600)

    def test_negative_completed_raises(self) -> None:
        """Negative completed raises ValueError."""
        with pytest.raises(ValueError, match="completed_trials must be non-negative"):
            estimate_remaining_trials(100.0, -1, 100, 3600)

    def test_zero_total_raises(self) -> None:
        """Zero total raises ValueError."""
        with pytest.raises(ValueError, match="total_trials must be positive"):
            estimate_remaining_trials(100.0, 10, 0, 3600)

    def test_zero_timeout_raises(self) -> None:
        """Zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            estimate_remaining_trials(100.0, 10, 100, 0)


class TestSuggestNextParams:
    """Tests for suggest_next_params function."""

    def test_random_algorithm(self) -> None:
        """Random algorithm samples parameters."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        config = create_hyperopt_config(algorithm="random", parameter_spaces=(space,))
        params = suggest_next_params(config, (), seed=42)
        assert "lr" in params
        assert 1e-5 <= params["lr"] <= 1e-1

    def test_grid_algorithm(self) -> None:
        """Grid algorithm uses structured sampling."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(algorithm="grid", parameter_spaces=(space,))
        params0 = suggest_next_params(config, ())
        assert "lr" in params0

    def test_bayesian_algorithm_no_history(self) -> None:
        """Bayesian algorithm samples randomly with no history."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        config = create_hyperopt_config(algorithm="bayesian", parameter_spaces=(space,))
        params = suggest_next_params(config, (), seed=42)
        assert "lr" in params

    def test_bayesian_algorithm_with_history(self) -> None:
        """Bayesian algorithm uses history."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(algorithm="bayesian", parameter_spaces=(space,))
        history = (
            create_trial_result({"lr": 0.5}, 0.1, 1),
            create_trial_result({"lr": 0.3}, 0.05, 2),
        )
        params = suggest_next_params(config, history, seed=42)
        assert "lr" in params

    def test_tpe_algorithm(self) -> None:
        """TPE algorithm works."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(algorithm="tpe", parameter_spaces=(space,))
        params = suggest_next_params(config, (), seed=42)
        assert "lr" in params

    def test_cmaes_algorithm(self) -> None:
        """CMA-ES algorithm works."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(algorithm="cmaes", parameter_spaces=(space,))
        params = suggest_next_params(config, (), seed=42)
        assert "lr" in params

    def test_hyperband_algorithm(self) -> None:
        """Hyperband algorithm works."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(
            algorithm="hyperband", parameter_spaces=(space,)
        )
        params = suggest_next_params(config, (), seed=42)
        assert "lr" in params

    def test_multiple_spaces(self) -> None:
        """Works with multiple parameter spaces."""
        spaces = (
            create_parameter_space("lr", "log_uniform", 1e-5, 1e-1),
            create_parameter_space("batch", "discrete", 8, 128),
            create_parameter_space("opt", "categorical", choices=("adam", "sgd")),
        )
        config = create_hyperopt_config(algorithm="random", parameter_spaces=spaces)
        params = suggest_next_params(config, (), seed=42)
        assert "lr" in params
        assert "batch" in params
        assert "opt" in params


class TestFormatHyperoptStats:
    """Tests for format_hyperopt_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_hyperopt_stats(
            total_trials=50,
            best_value=0.02,
            best_params={"lr": 0.0005, "batch_size": 64},
            convergence_curve=(0.1, 0.05, 0.02),
        )
        formatted = format_hyperopt_stats(stats)
        assert "Trials: 50" in formatted
        assert "Best Value: 0.0200" in formatted
        assert "lr" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_hyperopt_stats(
            total_trials=10,
            best_value=0.1,
            best_params={"lr": 0.001},
        )
        formatted = format_hyperopt_stats(stats)
        assert "Trials:" in formatted
        assert "Best Value:" in formatted
        assert "Best Params:" in formatted

    def test_convergence_curve_shown(self) -> None:
        """Convergence curve is shown when present."""
        stats = create_hyperopt_stats(
            total_trials=50,
            best_value=0.02,
            best_params={"lr": 0.0005},
            convergence_curve=(0.1, 0.08, 0.06, 0.04, 0.02),
        )
        formatted = format_hyperopt_stats(stats)
        assert "Convergence" in formatted

    def test_format_with_integer_params(self) -> None:
        """Format stats with integer parameters."""
        stats = create_hyperopt_stats(
            total_trials=20,
            best_value=0.05,
            best_params={"batch_size": 32, "epochs": 10},
        )
        formatted = format_hyperopt_stats(stats)
        assert "batch_size=32" in formatted
        assert "epochs=10" in formatted


class TestGetRecommendedHyperoptConfig:
    """Tests for get_recommended_hyperopt_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_hyperopt_config("classification")
        assert config.algorithm == SearchAlgorithm.TPE
        assert config.direction == ObjectiveDirection.MAXIMIZE
        assert any("learning_rate" in s.name for s in config.parameter_spaces)

    def test_generation_config(self) -> None:
        """Get config for generation task."""
        config = get_recommended_hyperopt_config("generation")
        assert config.algorithm == SearchAlgorithm.TPE
        assert config.direction == ObjectiveDirection.MINIMIZE

    def test_fine_tuning_config(self) -> None:
        """Get config for fine-tuning task."""
        config = get_recommended_hyperopt_config("fine_tuning")
        assert config.algorithm == SearchAlgorithm.BAYESIAN
        assert any("lora" in s.name for s in config.parameter_spaces)

    def test_pretraining_config(self) -> None:
        """Get config for pretraining task."""
        config = get_recommended_hyperopt_config("pretraining")
        assert config.algorithm == SearchAlgorithm.HYPERBAND
        assert config.direction == ObjectiveDirection.MINIMIZE

    def test_rl_config(self) -> None:
        """Get config for RL task."""
        config = get_recommended_hyperopt_config("rl")
        assert config.algorithm == SearchAlgorithm.CMAES
        assert config.direction == ObjectiveDirection.MAXIMIZE

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_hyperopt_config("unknown")

    @pytest.mark.parametrize(
        "task",
        ["classification", "generation", "fine_tuning", "pretraining", "rl"],
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All supported tasks return valid configs."""
        config = get_recommended_hyperopt_config(task)
        validate_hyperopt_config(config)


class TestGridSampling:
    """Tests for grid sampling behavior."""

    def test_grid_produces_different_values(self) -> None:
        """Grid search produces different values at different indices."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(algorithm="grid", parameter_spaces=(space,))

        params0 = suggest_next_params(config, ())
        params1 = suggest_next_params(config, (create_trial_result(params0, 0.1, 1),))

        assert params0["lr"] != params1["lr"]

    def test_grid_samples_categorical(self) -> None:
        """Grid search works with categorical."""
        space = create_parameter_space(
            "opt", "categorical", choices=("adam", "sgd", "adamw")
        )
        config = create_hyperopt_config(algorithm="grid", parameter_spaces=(space,))
        params = suggest_next_params(config, ())
        assert params["opt"] in ("adam", "sgd", "adamw")

    def test_grid_samples_discrete(self) -> None:
        """Grid search works with discrete."""
        space = create_parameter_space("batch", "discrete", 8, 128)
        config = create_hyperopt_config(algorithm="grid", parameter_spaces=(space,))
        params = suggest_next_params(config, ())
        assert 8 <= params["batch"] <= 128
        assert isinstance(params["batch"], int)

    def test_grid_samples_log_uniform(self) -> None:
        """Grid search works with log_uniform."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        config = create_hyperopt_config(algorithm="grid", parameter_spaces=(space,))
        params = suggest_next_params(config, ())
        # Use pytest.approx for floating point comparisons
        assert (
            params["lr"] == pytest.approx(1e-5, rel=1e-6) or 1e-5 < params["lr"] <= 1e-1
        )


class TestBayesianSampling:
    """Tests for Bayesian-like sampling behavior."""

    def test_bayesian_exploitation(self) -> None:
        """Bayesian method exploits good regions."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
            direction="minimize",
        )

        # Create history with best at lr=0.5
        history = (
            create_trial_result({"lr": 0.1}, 0.5, 1),
            create_trial_result({"lr": 0.5}, 0.1, 2),
            create_trial_result({"lr": 0.9}, 0.4, 3),
        )

        # With seed that doesn't trigger exploration
        params = suggest_next_params(config, history, seed=100)
        assert "lr" in params

    def test_bayesian_handles_missing_param(self) -> None:
        """Bayesian handles missing parameter in history."""
        spaces = (
            create_parameter_space("lr", "continuous", 0.0, 1.0),
            create_parameter_space("batch", "discrete", 8, 128),
        )
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=spaces,
        )

        # History only has lr
        history = (create_trial_result({"lr": 0.5}, 0.1, 1),)
        params = suggest_next_params(config, history, seed=100)
        assert "lr" in params
        assert "batch" in params

    def test_bayesian_categorical_random(self) -> None:
        """Bayesian samples categorical randomly."""
        space = create_parameter_space("opt", "categorical", choices=("adam", "sgd"))
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
        )
        history = (create_trial_result({"opt": "adam"}, 0.1, 1),)
        # Use seeds that trigger exploitation (not exploration)
        for seed in range(10, 20):
            params = suggest_next_params(config, history, seed=seed)
            assert params["opt"] in ("adam", "sgd")

    def test_bayesian_missing_param_non_categorical(self) -> None:
        """Bayesian falls back for non-categorical when param missing from history."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
        )
        # History has DIFFERENT param than what's in space
        history = (create_trial_result({"other_param": 0.5}, 0.1, 1),)
        # This should trigger the fallback for non-categorical params
        for seed in range(10, 20):
            params = suggest_next_params(config, history, seed=seed)
            assert "lr" in params
            assert 0.0 <= params["lr"] <= 1.0

    def test_bayesian_discrete_perturbation(self) -> None:
        """Bayesian perturbs discrete parameters."""
        space = create_parameter_space("batch", "discrete", 8, 128)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
        )
        history = (create_trial_result({"batch": 64}, 0.1, 1),)
        params = suggest_next_params(config, history, seed=100)
        assert 8 <= params["batch"] <= 128

    def test_bayesian_log_uniform_perturbation(self) -> None:
        """Bayesian perturbs log_uniform parameters."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
        )
        history = (create_trial_result({"lr": 1e-3}, 0.1, 1),)
        params = suggest_next_params(config, history, seed=100)
        assert 1e-5 <= params["lr"] <= 1e-1

    def test_bayesian_missing_param_fallback(self) -> None:
        """Bayesian falls back to random when param missing from history."""
        spaces = (
            create_parameter_space("lr", "continuous", 0.0, 1.0),
            create_parameter_space("new_param", "continuous", 0.0, 1.0),
        )
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=spaces,
        )
        # History only has lr, not new_param
        history = (create_trial_result({"lr": 0.5}, 0.1, 1),)
        params = suggest_next_params(config, history, seed=100)
        assert "new_param" in params
        assert 0.0 <= params["new_param"] <= 1.0

    def test_bayesian_maximize_direction(self) -> None:
        """Bayesian finds best trial for maximize direction."""
        space = create_parameter_space("lr", "continuous", 0.0, 1.0)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
            direction="maximize",
        )
        history = (
            create_trial_result({"lr": 0.3}, 0.5, 1),
            create_trial_result({"lr": 0.7}, 0.9, 2),  # Best for maximize
        )
        params = suggest_next_params(config, history, seed=100)
        assert "lr" in params

    def test_bayesian_exploitation_with_log_uniform(self) -> None:
        """Bayesian exploitation path with log_uniform parameter."""
        space = create_parameter_space("lr", "log_uniform", 1e-5, 1e-1)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
        )
        history = (create_trial_result({"lr": 1e-3}, 0.05, 1),)
        # Use different seeds to hit exploitation path (80% probability)
        for seed in range(10, 20):
            params = suggest_next_params(config, history, seed=seed)
            assert "lr" in params
            assert 1e-5 <= params["lr"] <= 1e-1

    def test_bayesian_exploitation_with_discrete(self) -> None:
        """Bayesian exploitation path with discrete parameter."""
        space = create_parameter_space("batch", "discrete", 8, 128)
        config = create_hyperopt_config(
            algorithm="bayesian",
            parameter_spaces=(space,),
        )
        history = (create_trial_result({"batch": 64}, 0.05, 1),)
        # Use different seeds to hit exploitation path
        for seed in range(10, 20):
            params = suggest_next_params(config, history, seed=seed)
            assert "batch" in params
            assert 8 <= params["batch"] <= 128
            assert isinstance(params["batch"], int)
