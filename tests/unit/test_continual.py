"""Tests for training.continual module."""

from __future__ import annotations

import pytest

from hf_gtc.training.continual import (
    VALID_CONTINUAL_METHODS,
    VALID_REGULARIZATION_TYPES,
    VALID_REPLAY_STRATEGIES,
    ContinualConfig,
    ContinualMethod,
    EWCConfig,
    ForgettingMetrics,
    RegularizationType,
    ReplayConfig,
    ReplayStrategy,
    calculate_backward_transfer,
    calculate_ewc_penalty,
    calculate_fisher_information,
    calculate_forgetting_measure,
    calculate_forward_transfer,
    calculate_plasticity_score,
    create_continual_config,
    create_ewc_config,
    create_forgetting_metrics,
    create_replay_config,
    estimate_importance_weights,
    format_forgetting_metrics,
    get_continual_method,
    get_recommended_continual_config,
    get_regularization_type,
    get_replay_strategy,
    list_continual_methods,
    list_regularization_types,
    list_replay_strategies,
    validate_continual_config,
    validate_ewc_config,
    validate_forgetting_metrics,
    validate_replay_config,
)


class TestContinualMethod:
    """Tests for ContinualMethod enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in ContinualMethod:
            assert isinstance(method.value, str)

    def test_ewc_value(self) -> None:
        """EWC has correct value."""
        assert ContinualMethod.EWC.value == "ewc"

    def test_si_value(self) -> None:
        """SI has correct value."""
        assert ContinualMethod.SI.value == "si"

    def test_mas_value(self) -> None:
        """MAS has correct value."""
        assert ContinualMethod.MAS.value == "mas"

    def test_gem_value(self) -> None:
        """GEM has correct value."""
        assert ContinualMethod.GEM.value == "gem"

    def test_agem_value(self) -> None:
        """AGEM has correct value."""
        assert ContinualMethod.AGEM.value == "agem"

    def test_replay_value(self) -> None:
        """REPLAY has correct value."""
        assert ContinualMethod.REPLAY.value == "replay"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_CONTINUAL_METHODS is a frozenset."""
        assert isinstance(VALID_CONTINUAL_METHODS, frozenset)
        assert len(VALID_CONTINUAL_METHODS) == 6


class TestReplayStrategy:
    """Tests for ReplayStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in ReplayStrategy:
            assert isinstance(strategy.value, str)

    def test_random_value(self) -> None:
        """Random has correct value."""
        assert ReplayStrategy.RANDOM.value == "random"

    def test_reservoir_value(self) -> None:
        """Reservoir has correct value."""
        assert ReplayStrategy.RESERVOIR.value == "reservoir"

    def test_herding_value(self) -> None:
        """Herding has correct value."""
        assert ReplayStrategy.HERDING.value == "herding"

    def test_mir_value(self) -> None:
        """MIR has correct value."""
        assert ReplayStrategy.MIR.value == "mir"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_REPLAY_STRATEGIES is a frozenset."""
        assert isinstance(VALID_REPLAY_STRATEGIES, frozenset)
        assert len(VALID_REPLAY_STRATEGIES) == 4


class TestRegularizationType:
    """Tests for RegularizationType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for reg_type in RegularizationType:
            assert isinstance(reg_type.value, str)

    def test_l2_value(self) -> None:
        """L2 has correct value."""
        assert RegularizationType.L2.value == "l2"

    def test_fisher_value(self) -> None:
        """Fisher has correct value."""
        assert RegularizationType.FISHER.value == "fisher"

    def test_importance_weighted_value(self) -> None:
        """Importance weighted has correct value."""
        assert RegularizationType.IMPORTANCE_WEIGHTED.value == "importance_weighted"

    def test_valid_types_frozenset(self) -> None:
        """VALID_REGULARIZATION_TYPES is a frozenset."""
        assert isinstance(VALID_REGULARIZATION_TYPES, frozenset)
        assert len(VALID_REGULARIZATION_TYPES) == 3


class TestEWCConfig:
    """Tests for EWCConfig dataclass."""

    def test_create_config(self) -> None:
        """Create EWC config."""
        config = EWCConfig(
            lambda_ewc=1000.0,
            fisher_samples=200,
            normalize_fisher=True,
            online=False,
        )
        assert config.lambda_ewc == 1000.0
        assert config.fisher_samples == 200
        assert config.normalize_fisher is True
        assert config.online is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = EWCConfig(1000.0, 200, True, False)
        with pytest.raises(AttributeError):
            config.lambda_ewc = 2000.0  # type: ignore[misc]


class TestReplayConfig:
    """Tests for ReplayConfig dataclass."""

    def test_create_config(self) -> None:
        """Create replay config."""
        config = ReplayConfig(
            strategy=ReplayStrategy.RESERVOIR,
            buffer_size=1000,
            samples_per_task=200,
            prioritized=False,
        )
        assert config.strategy == ReplayStrategy.RESERVOIR
        assert config.buffer_size == 1000
        assert config.samples_per_task == 200
        assert config.prioritized is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ReplayConfig(ReplayStrategy.RANDOM, 1000, 200, False)
        with pytest.raises(AttributeError):
            config.buffer_size = 2000  # type: ignore[misc]


class TestContinualConfig:
    """Tests for ContinualConfig dataclass."""

    def test_create_config(self) -> None:
        """Create continual config."""
        config = ContinualConfig(
            method=ContinualMethod.EWC,
            replay_config=None,
            regularization_strength=0.5,
            task_boundary=True,
        )
        assert config.method == ContinualMethod.EWC
        assert config.replay_config is None
        assert config.regularization_strength == 0.5
        assert config.task_boundary is True

    def test_config_with_replay(self) -> None:
        """Create config with replay."""
        replay = ReplayConfig(ReplayStrategy.RESERVOIR, 1000, 200, False)
        config = ContinualConfig(
            method=ContinualMethod.REPLAY,
            replay_config=replay,
            regularization_strength=0.1,
            task_boundary=False,
        )
        assert config.replay_config is not None
        assert config.replay_config.buffer_size == 1000

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ContinualConfig(ContinualMethod.EWC, None, 0.5, True)
        with pytest.raises(AttributeError):
            config.regularization_strength = 0.8  # type: ignore[misc]


class TestForgettingMetrics:
    """Tests for ForgettingMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Create forgetting metrics."""
        metrics = ForgettingMetrics(
            backward_transfer=-0.05,
            forward_transfer=0.10,
            forgetting_rate=0.03,
            plasticity=0.95,
        )
        assert metrics.backward_transfer == -0.05
        assert metrics.forward_transfer == 0.10
        assert metrics.forgetting_rate == 0.03
        assert metrics.plasticity == 0.95

    def test_metrics_is_frozen(self) -> None:
        """Metrics is immutable."""
        metrics = ForgettingMetrics(-0.05, 0.10, 0.03, 0.95)
        with pytest.raises(AttributeError):
            metrics.plasticity = 0.9  # type: ignore[misc]


class TestValidateEWCConfig:
    """Tests for validate_ewc_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = EWCConfig(1000.0, 200, True, False)
        validate_ewc_config(config)

    def test_negative_lambda_raises(self) -> None:
        """Negative lambda_ewc raises ValueError."""
        config = EWCConfig(-1.0, 200, True, False)
        with pytest.raises(ValueError, match="lambda_ewc must be non-negative"):
            validate_ewc_config(config)

    def test_zero_lambda_is_valid(self) -> None:
        """Zero lambda_ewc is valid (no regularization)."""
        config = EWCConfig(0.0, 200, True, False)
        validate_ewc_config(config)

    def test_zero_fisher_samples_raises(self) -> None:
        """Zero fisher_samples raises ValueError."""
        config = EWCConfig(1000.0, 0, True, False)
        with pytest.raises(ValueError, match="fisher_samples must be positive"):
            validate_ewc_config(config)

    def test_negative_fisher_samples_raises(self) -> None:
        """Negative fisher_samples raises ValueError."""
        config = EWCConfig(1000.0, -100, True, False)
        with pytest.raises(ValueError, match="fisher_samples must be positive"):
            validate_ewc_config(config)


class TestValidateReplayConfig:
    """Tests for validate_replay_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ReplayConfig(ReplayStrategy.RESERVOIR, 1000, 200, False)
        validate_replay_config(config)

    def test_zero_buffer_size_raises(self) -> None:
        """Zero buffer_size raises ValueError."""
        config = ReplayConfig(ReplayStrategy.RANDOM, 0, 200, False)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            validate_replay_config(config)

    def test_negative_buffer_size_raises(self) -> None:
        """Negative buffer_size raises ValueError."""
        config = ReplayConfig(ReplayStrategy.RANDOM, -100, 200, False)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            validate_replay_config(config)

    def test_zero_samples_per_task_raises(self) -> None:
        """Zero samples_per_task raises ValueError."""
        config = ReplayConfig(ReplayStrategy.RANDOM, 1000, 0, False)
        with pytest.raises(ValueError, match="samples_per_task must be positive"):
            validate_replay_config(config)

    def test_samples_exceeds_buffer_raises(self) -> None:
        """samples_per_task > buffer_size raises ValueError."""
        config = ReplayConfig(ReplayStrategy.RANDOM, 100, 200, False)
        with pytest.raises(ValueError, match=r"samples_per_task.*cannot exceed"):
            validate_replay_config(config)


class TestValidateContinualConfig:
    """Tests for validate_continual_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ContinualConfig(ContinualMethod.EWC, None, 0.5, True)
        validate_continual_config(config)

    def test_negative_regularization_strength_raises(self) -> None:
        """Negative regularization_strength raises ValueError."""
        config = ContinualConfig(ContinualMethod.EWC, None, -0.1, True)
        with pytest.raises(
            ValueError, match="regularization_strength must be non-negative"
        ):
            validate_continual_config(config)

    def test_zero_regularization_is_valid(self) -> None:
        """Zero regularization_strength is valid."""
        config = ContinualConfig(ContinualMethod.REPLAY, None, 0.0, False)
        validate_continual_config(config)

    def test_invalid_replay_config_raises(self) -> None:
        """Invalid replay_config raises ValueError."""
        bad_replay = ReplayConfig(ReplayStrategy.RANDOM, 100, 200, False)
        config = ContinualConfig(ContinualMethod.REPLAY, bad_replay, 0.1, False)
        with pytest.raises(ValueError, match=r"samples_per_task.*cannot exceed"):
            validate_continual_config(config)


class TestValidateForgettingMetrics:
    """Tests for validate_forgetting_metrics function."""

    def test_valid_metrics(self) -> None:
        """Valid metrics passes validation."""
        metrics = ForgettingMetrics(-0.05, 0.10, 0.03, 0.95)
        validate_forgetting_metrics(metrics)

    def test_negative_forgetting_rate_raises(self) -> None:
        """Negative forgetting_rate raises ValueError."""
        metrics = ForgettingMetrics(-0.05, 0.10, -0.1, 0.95)
        with pytest.raises(ValueError, match="forgetting_rate must be non-negative"):
            validate_forgetting_metrics(metrics)

    def test_plasticity_above_one_raises(self) -> None:
        """Plasticity > 1 raises ValueError."""
        metrics = ForgettingMetrics(-0.05, 0.10, 0.03, 1.5)
        with pytest.raises(ValueError, match="plasticity must be between 0 and 1"):
            validate_forgetting_metrics(metrics)

    def test_negative_plasticity_raises(self) -> None:
        """Negative plasticity raises ValueError."""
        metrics = ForgettingMetrics(-0.05, 0.10, 0.03, -0.1)
        with pytest.raises(ValueError, match="plasticity must be between 0 and 1"):
            validate_forgetting_metrics(metrics)

    def test_zero_values_valid(self) -> None:
        """Zero values are valid."""
        metrics = ForgettingMetrics(0.0, 0.0, 0.0, 0.0)
        validate_forgetting_metrics(metrics)


class TestCreateEWCConfig:
    """Tests for create_ewc_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_ewc_config()
        assert config.lambda_ewc == 1000.0
        assert config.fisher_samples == 200
        assert config.normalize_fisher is True
        assert config.online is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_ewc_config(
            lambda_ewc=5000.0,
            fisher_samples=500,
            normalize_fisher=False,
            online=True,
        )
        assert config.lambda_ewc == 5000.0
        assert config.fisher_samples == 500
        assert config.normalize_fisher is False
        assert config.online is True

    def test_negative_lambda_raises(self) -> None:
        """Negative lambda raises ValueError."""
        with pytest.raises(ValueError, match="lambda_ewc must be non-negative"):
            create_ewc_config(lambda_ewc=-1.0)

    def test_zero_fisher_samples_raises(self) -> None:
        """Zero fisher_samples raises ValueError."""
        with pytest.raises(ValueError, match="fisher_samples must be positive"):
            create_ewc_config(fisher_samples=0)


class TestCreateReplayConfig:
    """Tests for create_replay_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_replay_config()
        assert config.strategy == ReplayStrategy.RESERVOIR
        assert config.buffer_size == 1000
        assert config.samples_per_task == 200
        assert config.prioritized is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_replay_config(
            strategy="herding",
            buffer_size=5000,
            samples_per_task=500,
            prioritized=True,
        )
        assert config.strategy == ReplayStrategy.HERDING
        assert config.buffer_size == 5000
        assert config.prioritized is True

    def test_with_enum_strategy(self) -> None:
        """Create with enum strategy."""
        config = create_replay_config(strategy=ReplayStrategy.MIR)
        assert config.strategy == ReplayStrategy.MIR

    def test_zero_buffer_size_raises(self) -> None:
        """Zero buffer_size raises ValueError."""
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            create_replay_config(buffer_size=0)

    @pytest.mark.parametrize(
        "strategy",
        ["random", "reservoir", "herding", "mir"],
    )
    def test_all_strategies(self, strategy: str) -> None:
        """Test all replay strategies."""
        config = create_replay_config(strategy=strategy)
        assert config.strategy.value == strategy


class TestCreateContinualConfig:
    """Tests for create_continual_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_continual_config()
        assert config.method == ContinualMethod.EWC
        assert config.regularization_strength == 0.5
        assert config.task_boundary is True
        assert config.replay_config is None

    def test_custom_config(self) -> None:
        """Create custom config."""
        replay = create_replay_config()
        config = create_continual_config(
            method="replay",
            replay_config=replay,
            regularization_strength=0.1,
            task_boundary=False,
        )
        assert config.method == ContinualMethod.REPLAY
        assert config.replay_config is not None

    def test_with_enum_method(self) -> None:
        """Create with enum method."""
        config = create_continual_config(method=ContinualMethod.SI)
        assert config.method == ContinualMethod.SI

    def test_negative_regularization_raises(self) -> None:
        """Negative regularization raises ValueError."""
        with pytest.raises(
            ValueError, match="regularization_strength must be non-negative"
        ):
            create_continual_config(regularization_strength=-0.1)

    @pytest.mark.parametrize(
        "method",
        ["ewc", "si", "mas", "gem", "agem", "replay"],
    )
    def test_all_methods(self, method: str) -> None:
        """Test all continual methods."""
        config = create_continual_config(method=method)
        assert config.method.value == method


class TestCreateForgettingMetrics:
    """Tests for create_forgetting_metrics function."""

    def test_default_metrics(self) -> None:
        """Create default metrics."""
        metrics = create_forgetting_metrics()
        assert metrics.backward_transfer == 0.0
        assert metrics.forward_transfer == 0.0
        assert metrics.forgetting_rate == 0.0
        assert metrics.plasticity == 1.0

    def test_custom_metrics(self) -> None:
        """Create custom metrics."""
        metrics = create_forgetting_metrics(
            backward_transfer=-0.05,
            forward_transfer=0.10,
            forgetting_rate=0.03,
            plasticity=0.95,
        )
        assert metrics.backward_transfer == -0.05
        assert metrics.plasticity == 0.95

    def test_negative_forgetting_rate_raises(self) -> None:
        """Negative forgetting_rate raises ValueError."""
        with pytest.raises(ValueError, match="forgetting_rate must be non-negative"):
            create_forgetting_metrics(forgetting_rate=-0.1)

    def test_plasticity_out_of_range_raises(self) -> None:
        """Plasticity > 1 raises ValueError."""
        with pytest.raises(ValueError, match="plasticity must be between 0 and 1"):
            create_forgetting_metrics(plasticity=1.5)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_continual_methods_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_continual_methods()
        assert methods == sorted(methods)
        assert "ewc" in methods
        assert "replay" in methods

    def test_list_replay_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_replay_strategies()
        assert strategies == sorted(strategies)
        assert "reservoir" in strategies
        assert "random" in strategies

    def test_list_regularization_types_sorted(self) -> None:
        """Returns sorted list."""
        types = list_regularization_types()
        assert types == sorted(types)
        assert "fisher" in types
        assert "l2" in types


class TestGetContinualMethod:
    """Tests for get_continual_method function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("ewc", ContinualMethod.EWC),
            ("si", ContinualMethod.SI),
            ("mas", ContinualMethod.MAS),
            ("gem", ContinualMethod.GEM),
            ("agem", ContinualMethod.AGEM),
            ("replay", ContinualMethod.REPLAY),
        ],
    )
    def test_all_methods(self, name: str, expected: ContinualMethod) -> None:
        """Test all valid methods."""
        assert get_continual_method(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            get_continual_method("invalid")


class TestGetReplayStrategy:
    """Tests for get_replay_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("random", ReplayStrategy.RANDOM),
            ("reservoir", ReplayStrategy.RESERVOIR),
            ("herding", ReplayStrategy.HERDING),
            ("mir", ReplayStrategy.MIR),
        ],
    )
    def test_all_strategies(self, name: str, expected: ReplayStrategy) -> None:
        """Test all valid strategies."""
        assert get_replay_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_replay_strategy("invalid")


class TestGetRegularizationType:
    """Tests for get_regularization_type function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("l2", RegularizationType.L2),
            ("fisher", RegularizationType.FISHER),
            ("importance_weighted", RegularizationType.IMPORTANCE_WEIGHTED),
        ],
    )
    def test_all_types(self, name: str, expected: RegularizationType) -> None:
        """Test all valid types."""
        assert get_regularization_type(name) == expected

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="regularization_type must be one of"):
            get_regularization_type("invalid")


class TestCalculateFisherInformation:
    """Tests for calculate_fisher_information function."""

    def test_basic_calculation(self) -> None:
        """Basic Fisher calculation."""
        grads = (0.1, 0.2, 0.3)
        fisher = calculate_fisher_information(grads, 10)
        assert len(fisher) == 3
        assert all(f >= 0 for f in fisher)

    def test_fisher_is_squared_normalized(self) -> None:
        """Fisher is squared gradient normalized by samples."""
        grads = (1.0, 2.0)
        fisher = calculate_fisher_information(grads, 1)
        assert fisher[0] == pytest.approx(1.0)
        assert fisher[1] == pytest.approx(4.0)

    def test_scaling_by_samples(self) -> None:
        """Fisher scales inversely with sample count."""
        grads = (1.0,)
        fisher_1 = calculate_fisher_information(grads, 1)
        fisher_10 = calculate_fisher_information(grads, 10)
        assert fisher_1[0] == pytest.approx(fisher_10[0] * 10)

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            calculate_fisher_information((), 10)

    def test_zero_samples_raises(self) -> None:
        """Zero samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            calculate_fisher_information((0.1,), 0)

    def test_negative_samples_raises(self) -> None:
        """Negative samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            calculate_fisher_information((0.1,), -1)


class TestEstimateImportanceWeights:
    """Tests for estimate_importance_weights function."""

    def test_basic_estimation(self) -> None:
        """Basic importance estimation."""
        changes = (0.01, 0.05, 0.02)
        lrs = (0.001, 0.001, 0.001)
        weights = estimate_importance_weights(changes, lrs)
        assert len(weights) == 3
        assert all(w >= 0 for w in weights)

    def test_larger_change_higher_importance(self) -> None:
        """Larger parameter change gives higher importance."""
        changes = (0.01, 0.10)
        lrs = (0.001, 0.001)
        weights = estimate_importance_weights(changes, lrs)
        assert weights[1] > weights[0]

    def test_empty_changes_raises(self) -> None:
        """Empty parameter_changes raises ValueError."""
        with pytest.raises(ValueError, match="parameter_changes cannot be empty"):
            estimate_importance_weights((), (0.001,))

    def test_empty_learning_rates_raises(self) -> None:
        """Empty learning_rates raises ValueError."""
        with pytest.raises(ValueError, match="learning_rates cannot be empty"):
            estimate_importance_weights((0.01,), ())

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            estimate_importance_weights((0.01,), (0.001, 0.002))


class TestCalculateForgettingMeasure:
    """Tests for calculate_forgetting_measure function."""

    def test_basic_forgetting(self) -> None:
        """Calculate basic forgetting."""
        initial = (0.95, 0.90, 0.85)
        final = (0.90, 0.85, 0.85)
        forgetting = calculate_forgetting_measure(initial, final)
        assert 0 <= forgetting <= 1

    def test_no_forgetting(self) -> None:
        """No forgetting when final >= initial."""
        initial = (0.90, 0.85, 0.80)
        final = (0.92, 0.88, 0.85)
        forgetting = calculate_forgetting_measure(initial, final)
        assert forgetting == pytest.approx(0.0)

    def test_complete_forgetting(self) -> None:
        """Complete forgetting calculation."""
        initial = (1.0, 1.0)
        final = (0.0, 0.0)
        forgetting = calculate_forgetting_measure(initial, final)
        assert forgetting == pytest.approx(1.0)

    def test_empty_accuracies_raises(self) -> None:
        """Empty accuracies raises ValueError."""
        with pytest.raises(ValueError, match="accuracies cannot be empty"):
            calculate_forgetting_measure((), ())

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            calculate_forgetting_measure((0.9,), (0.8, 0.7))


class TestCalculatePlasticityScore:
    """Tests for calculate_plasticity_score function."""

    def test_basic_plasticity(self) -> None:
        """Calculate basic plasticity."""
        accuracies = (0.92, 0.88, 0.90)
        plasticity = calculate_plasticity_score(accuracies)
        assert 0 <= plasticity <= 1

    def test_full_plasticity(self) -> None:
        """Full plasticity when achieving baseline."""
        accuracies = (0.95, 0.95, 0.95)
        plasticity = calculate_plasticity_score(accuracies, baseline_accuracy=0.9)
        assert plasticity == pytest.approx(1.0)

    def test_partial_plasticity(self) -> None:
        """Partial plasticity below baseline."""
        accuracies = (0.45,)
        plasticity = calculate_plasticity_score(accuracies, baseline_accuracy=0.9)
        assert plasticity == pytest.approx(0.5)

    def test_capped_at_one(self) -> None:
        """Plasticity capped at 1.0."""
        accuracies = (1.0, 1.0)
        plasticity = calculate_plasticity_score(accuracies, baseline_accuracy=0.5)
        assert plasticity == pytest.approx(1.0)

    def test_empty_accuracies_raises(self) -> None:
        """Empty accuracies raises ValueError."""
        with pytest.raises(ValueError, match="task_accuracies cannot be empty"):
            calculate_plasticity_score(())

    def test_zero_baseline_raises(self) -> None:
        """Zero baseline raises ValueError."""
        with pytest.raises(ValueError, match="baseline_accuracy must be positive"):
            calculate_plasticity_score((0.9,), baseline_accuracy=0.0)


class TestFormatForgettingMetrics:
    """Tests for format_forgetting_metrics function."""

    def test_basic_format(self) -> None:
        """Format basic metrics."""
        metrics = create_forgetting_metrics(
            backward_transfer=-0.05,
            forward_transfer=0.10,
            forgetting_rate=0.03,
            plasticity=0.95,
        )
        formatted = format_forgetting_metrics(metrics)
        assert "Backward Transfer: -0.0500" in formatted
        assert "Forward Transfer: 0.1000" in formatted
        assert "Forgetting Rate: 0.0300" in formatted
        assert "Plasticity: 95.0%" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        metrics = create_forgetting_metrics()
        formatted = format_forgetting_metrics(metrics)
        assert "Backward Transfer:" in formatted
        assert "Forward Transfer:" in formatted
        assert "Forgetting Rate:" in formatted
        assert "Plasticity:" in formatted


class TestGetRecommendedContinualConfig:
    """Tests for get_recommended_continual_config function."""

    def test_few_tasks_uses_ewc(self) -> None:
        """Few tasks recommends EWC."""
        config = get_recommended_continual_config(2)
        assert config.method == ContinualMethod.EWC

    def test_three_tasks_uses_ewc(self) -> None:
        """Three tasks recommends EWC."""
        config = get_recommended_continual_config(3)
        assert config.method == ContinualMethod.EWC

    def test_moderate_tasks_uses_si(self) -> None:
        """Moderate tasks recommends SI."""
        config = get_recommended_continual_config(5)
        assert config.method == ContinualMethod.SI

    def test_ten_tasks_uses_si(self) -> None:
        """Ten tasks recommends SI."""
        config = get_recommended_continual_config(10)
        assert config.method == ContinualMethod.SI

    def test_many_tasks_uses_replay(self) -> None:
        """Many tasks recommends replay."""
        config = get_recommended_continual_config(15)
        assert config.method == ContinualMethod.REPLAY
        assert config.replay_config is not None

    def test_replay_buffer_scales(self) -> None:
        """Replay buffer size scales with tasks."""
        config_15 = get_recommended_continual_config(15)
        config_20 = get_recommended_continual_config(20)
        assert config_15.replay_config is not None
        assert config_20.replay_config is not None
        buffer_15 = config_15.replay_config.buffer_size
        buffer_20 = config_20.replay_config.buffer_size
        assert buffer_20 >= buffer_15

    def test_zero_tasks_raises(self) -> None:
        """Zero tasks raises ValueError."""
        with pytest.raises(ValueError, match="num_tasks must be positive"):
            get_recommended_continual_config(0)

    def test_negative_tasks_raises(self) -> None:
        """Negative tasks raises ValueError."""
        with pytest.raises(ValueError, match="num_tasks must be positive"):
            get_recommended_continual_config(-1)


class TestCalculateEWCPenalty:
    """Tests for calculate_ewc_penalty function."""

    def test_basic_penalty(self) -> None:
        """Calculate basic EWC penalty."""
        current = (0.5, 0.6, 0.7)
        optimal = (0.4, 0.5, 0.6)
        fisher = (1.0, 2.0, 3.0)
        penalty = calculate_ewc_penalty(current, optimal, fisher, 1000.0)
        assert penalty > 0

    def test_no_penalty_at_optimal(self) -> None:
        """No penalty when at optimal parameters."""
        params = (0.5, 0.6, 0.7)
        fisher = (1.0, 2.0, 3.0)
        penalty = calculate_ewc_penalty(params, params, fisher, 1000.0)
        assert penalty == pytest.approx(0.0)

    def test_zero_lambda_no_penalty(self) -> None:
        """Zero lambda gives no penalty."""
        current = (0.5, 0.6, 0.7)
        optimal = (0.4, 0.5, 0.6)
        fisher = (1.0, 2.0, 3.0)
        penalty = calculate_ewc_penalty(current, optimal, fisher, 0.0)
        assert penalty == pytest.approx(0.0)

    def test_penalty_scales_with_lambda(self) -> None:
        """Penalty scales with lambda."""
        current = (0.5,)
        optimal = (0.4,)
        fisher = (1.0,)
        penalty_1000 = calculate_ewc_penalty(current, optimal, fisher, 1000.0)
        penalty_2000 = calculate_ewc_penalty(current, optimal, fisher, 2000.0)
        assert penalty_2000 == pytest.approx(penalty_1000 * 2)

    def test_empty_params_raises(self) -> None:
        """Empty parameters raises ValueError."""
        with pytest.raises(ValueError, match="parameter tuples cannot be empty"):
            calculate_ewc_penalty((), (0.5,), (1.0,), 1000.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raises ValueError."""
        with pytest.raises(
            ValueError, match="all parameter tuples must have same length"
        ):
            calculate_ewc_penalty((0.5,), (0.4, 0.5), (1.0,), 1000.0)

    def test_negative_lambda_raises(self) -> None:
        """Negative lambda raises ValueError."""
        with pytest.raises(ValueError, match="lambda_ewc must be non-negative"):
            calculate_ewc_penalty((0.5,), (0.4,), (1.0,), -1.0)


class TestCalculateBackwardTransfer:
    """Tests for calculate_backward_transfer function."""

    def test_basic_backward_transfer(self) -> None:
        """Calculate basic backward transfer."""
        matrix = (
            (0.90, 0.00, 0.00),
            (0.85, 0.88, 0.00),
            (0.82, 0.85, 0.87),
        )
        bt = calculate_backward_transfer(matrix)
        assert bt < 0  # Some forgetting expected

    def test_no_forgetting(self) -> None:
        """No forgetting gives zero backward transfer."""
        matrix = (
            (0.90, 0.00),
            (0.90, 0.88),
        )
        bt = calculate_backward_transfer(matrix)
        assert bt == pytest.approx(0.0)

    def test_positive_backward_transfer(self) -> None:
        """Positive backward transfer (improvement)."""
        matrix = (
            (0.80, 0.00),
            (0.90, 0.88),  # Task 0 improved after learning task 1
        )
        bt = calculate_backward_transfer(matrix)
        assert bt > 0

    def test_single_task_returns_zero(self) -> None:
        """Single task returns zero backward transfer."""
        matrix = ((0.90,),)
        bt = calculate_backward_transfer(matrix)
        assert bt == pytest.approx(0.0)

    def test_empty_matrix_raises(self) -> None:
        """Empty matrix raises ValueError."""
        with pytest.raises(ValueError, match="accuracy_matrix cannot be empty"):
            calculate_backward_transfer(())

    def test_non_square_matrix_raises(self) -> None:
        """Non-square matrix raises ValueError."""
        matrix = (
            (0.90, 0.00),
            (0.85, 0.88, 0.00),
        )
        with pytest.raises(ValueError, match="accuracy_matrix must be square"):
            calculate_backward_transfer(matrix)


class TestCalculateForwardTransfer:
    """Tests for calculate_forward_transfer function."""

    def test_basic_forward_transfer(self) -> None:
        """Calculate basic forward transfer."""
        matrix = (
            (0.90, 0.30, 0.25),
            (0.85, 0.88, 0.35),
            (0.82, 0.85, 0.87),
        )
        baseline = (0.10, 0.10, 0.10)
        ft = calculate_forward_transfer(matrix, baseline)
        assert ft > 0  # Positive transfer expected

    def test_no_baseline_uses_zero(self) -> None:
        """No baseline uses zero."""
        matrix = (
            (0.90, 0.20),
            (0.85, 0.88),
        )
        ft = calculate_forward_transfer(matrix)
        assert ft == pytest.approx(0.20)

    def test_negative_forward_transfer(self) -> None:
        """Negative transfer when worse than baseline."""
        matrix = (
            (0.90, 0.05),
            (0.85, 0.88),
        )
        baseline = (0.10, 0.10)
        ft = calculate_forward_transfer(matrix, baseline)
        assert ft < 0

    def test_single_task_returns_zero(self) -> None:
        """Single task returns zero forward transfer."""
        matrix = ((0.90,),)
        ft = calculate_forward_transfer(matrix)
        assert ft == pytest.approx(0.0)

    def test_empty_matrix_raises(self) -> None:
        """Empty matrix raises ValueError."""
        with pytest.raises(ValueError, match="accuracy_matrix cannot be empty"):
            calculate_forward_transfer(())

    def test_baseline_length_mismatch_raises(self) -> None:
        """Baseline length mismatch raises ValueError."""
        matrix = (
            (0.90, 0.20),
            (0.85, 0.88),
        )
        baseline = (0.10,)
        with pytest.raises(ValueError, match="random_baseline length must match"):
            calculate_forward_transfer(matrix, baseline)
