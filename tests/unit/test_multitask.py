"""Tests for training.multitask module."""

from __future__ import annotations

import pytest

from hf_gtc.training.multitask import (
    VALID_GRADIENT_STRATEGIES,
    VALID_TASK_BALANCING,
    VALID_TASK_RELATIONS,
    GradientConfig,
    GradientStrategy,
    MultiTaskConfig,
    MultiTaskStats,
    TaskBalancing,
    TaskConfig,
    TaskRelation,
    calculate_task_weights,
    create_gradient_config,
    create_multitask_config,
    create_multitask_stats,
    create_task_config,
    detect_gradient_conflicts,
    estimate_task_difficulty,
    format_multitask_stats,
    get_gradient_strategy,
    get_recommended_multitask_config,
    get_task_balancing,
    get_task_relation,
    list_gradient_strategies,
    list_task_balancing_methods,
    list_task_relations,
    project_conflicting_gradients,
    validate_gradient_config,
    validate_multitask_config,
    validate_multitask_stats,
    validate_task_config,
)


class TestTaskBalancing:
    """Tests for TaskBalancing enum."""

    def test_all_methods_have_values(self) -> None:
        """All methods have string values."""
        for method in TaskBalancing:
            assert isinstance(method.value, str)

    def test_uniform_value(self) -> None:
        """Uniform has correct value."""
        assert TaskBalancing.UNIFORM.value == "uniform"

    def test_uncertainty_value(self) -> None:
        """Uncertainty has correct value."""
        assert TaskBalancing.UNCERTAINTY.value == "uncertainty"

    def test_gradnorm_value(self) -> None:
        """GradNorm has correct value."""
        assert TaskBalancing.GRADNORM.value == "gradnorm"

    def test_pcgrad_value(self) -> None:
        """PCGrad has correct value."""
        assert TaskBalancing.PCGRAD.value == "pcgrad"

    def test_cagrad_value(self) -> None:
        """CAGrad has correct value."""
        assert TaskBalancing.CAGRAD.value == "cagrad"

    def test_valid_methods_frozenset(self) -> None:
        """VALID_TASK_BALANCING is a frozenset."""
        assert isinstance(VALID_TASK_BALANCING, frozenset)
        assert len(VALID_TASK_BALANCING) == 5


class TestGradientStrategy:
    """Tests for GradientStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in GradientStrategy:
            assert isinstance(strategy.value, str)

    def test_sum_value(self) -> None:
        """Sum has correct value."""
        assert GradientStrategy.SUM.value == "sum"

    def test_pcgrad_value(self) -> None:
        """PCGrad has correct value."""
        assert GradientStrategy.PCGRAD.value == "pcgrad"

    def test_cagrad_value(self) -> None:
        """CAGrad has correct value."""
        assert GradientStrategy.CAGRAD.value == "cagrad"

    def test_mgda_value(self) -> None:
        """MGDA has correct value."""
        assert GradientStrategy.MGDA.value == "mgda"

    def test_nash_value(self) -> None:
        """Nash has correct value."""
        assert GradientStrategy.NASH.value == "nash"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_GRADIENT_STRATEGIES is a frozenset."""
        assert isinstance(VALID_GRADIENT_STRATEGIES, frozenset)
        assert len(VALID_GRADIENT_STRATEGIES) == 5


class TestTaskRelation:
    """Tests for TaskRelation enum."""

    def test_all_relations_have_values(self) -> None:
        """All relations have string values."""
        for relation in TaskRelation:
            assert isinstance(relation.value, str)

    def test_independent_value(self) -> None:
        """Independent has correct value."""
        assert TaskRelation.INDEPENDENT.value == "independent"

    def test_auxiliary_value(self) -> None:
        """Auxiliary has correct value."""
        assert TaskRelation.AUXILIARY.value == "auxiliary"

    def test_hierarchical_value(self) -> None:
        """Hierarchical has correct value."""
        assert TaskRelation.HIERARCHICAL.value == "hierarchical"

    def test_valid_relations_frozenset(self) -> None:
        """VALID_TASK_RELATIONS is a frozenset."""
        assert isinstance(VALID_TASK_RELATIONS, frozenset)
        assert len(VALID_TASK_RELATIONS) == 3


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_create_config(self) -> None:
        """Create task config."""
        config = TaskConfig(
            name="classification",
            weight=1.0,
            loss_type="cross_entropy",
            relation=TaskRelation.INDEPENDENT,
        )
        assert config.name == "classification"
        assert config.weight == 1.0
        assert config.loss_type == "cross_entropy"
        assert config.relation == TaskRelation.INDEPENDENT

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = TaskConfig("test", 1.0, "ce", TaskRelation.INDEPENDENT)
        with pytest.raises(AttributeError):
            config.weight = 2.0  # type: ignore[misc]


class TestGradientConfig:
    """Tests for GradientConfig dataclass."""

    def test_create_config(self) -> None:
        """Create gradient config."""
        config = GradientConfig(
            strategy=GradientStrategy.SUM,
            conflict_threshold=0.0,
            normalize=False,
        )
        assert config.strategy == GradientStrategy.SUM
        assert config.conflict_threshold == 0.0
        assert config.normalize is False

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = GradientConfig(GradientStrategy.SUM, 0.0, False)
        with pytest.raises(AttributeError):
            config.normalize = True  # type: ignore[misc]


class TestMultiTaskConfig:
    """Tests for MultiTaskConfig dataclass."""

    def test_create_config(self) -> None:
        """Create multi-task config."""
        task = TaskConfig("test", 1.0, "ce", TaskRelation.INDEPENDENT)
        gradient = GradientConfig(GradientStrategy.SUM, 0.0, False)
        config = MultiTaskConfig(
            tasks=(task,),
            gradient_config=gradient,
            balancing_method=TaskBalancing.UNIFORM,
            shared_layers=("encoder",),
        )
        assert len(config.tasks) == 1
        assert config.balancing_method == TaskBalancing.UNIFORM
        assert config.shared_layers == ("encoder",)

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        gradient = GradientConfig(GradientStrategy.SUM, 0.0, False)
        config = MultiTaskConfig((), gradient, TaskBalancing.UNIFORM, ())
        with pytest.raises(AttributeError):
            config.balancing_method = TaskBalancing.UNCERTAINTY  # type: ignore[misc]


class TestMultiTaskStats:
    """Tests for MultiTaskStats dataclass."""

    def test_create_stats(self) -> None:
        """Create multi-task stats."""
        stats = MultiTaskStats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": 0.9},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0},
        )
        assert stats.task_losses["task1"] == 0.5
        assert stats.task_accuracies["task1"] == 0.9
        assert stats.gradient_conflicts == 5

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = MultiTaskStats({}, {}, 0, {})
        with pytest.raises(AttributeError):
            stats.gradient_conflicts = 10  # type: ignore[misc]


class TestValidateTaskConfig:
    """Tests for validate_task_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = TaskConfig("test", 1.0, "ce", TaskRelation.INDEPENDENT)
        validate_task_config(config)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        config = TaskConfig("", 1.0, "ce", TaskRelation.INDEPENDENT)
        with pytest.raises(ValueError, match="task name cannot be empty"):
            validate_task_config(config)

    def test_whitespace_name_raises(self) -> None:
        """Whitespace name raises ValueError."""
        config = TaskConfig("   ", 1.0, "ce", TaskRelation.INDEPENDENT)
        with pytest.raises(ValueError, match="task name cannot be empty"):
            validate_task_config(config)

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        config = TaskConfig("test", -1.0, "ce", TaskRelation.INDEPENDENT)
        with pytest.raises(ValueError, match="weight must be non-negative"):
            validate_task_config(config)

    def test_empty_loss_type_raises(self) -> None:
        """Empty loss_type raises ValueError."""
        config = TaskConfig("test", 1.0, "", TaskRelation.INDEPENDENT)
        with pytest.raises(ValueError, match="loss_type cannot be empty"):
            validate_task_config(config)


class TestValidateGradientConfig:
    """Tests for validate_gradient_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = GradientConfig(GradientStrategy.SUM, 0.0, False)
        validate_gradient_config(config)

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold raises ValueError."""
        config = GradientConfig(GradientStrategy.PCGRAD, -0.5, False)
        with pytest.raises(ValueError, match="conflict_threshold must be non-negative"):
            validate_gradient_config(config)


class TestValidateMultiTaskConfig:
    """Tests for validate_multitask_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = create_multitask_config()
        validate_multitask_config(config)

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        bad_task = TaskConfig("", 1.0, "ce", TaskRelation.INDEPENDENT)
        gradient = GradientConfig(GradientStrategy.SUM, 0.0, False)
        config = MultiTaskConfig(
            tasks=(bad_task,),
            gradient_config=gradient,
            balancing_method=TaskBalancing.UNIFORM,
            shared_layers=(),
        )
        with pytest.raises(ValueError, match="task name cannot be empty"):
            validate_multitask_config(config)

    def test_invalid_gradient_config_raises(self) -> None:
        """Invalid gradient config raises ValueError."""
        task = TaskConfig("test", 1.0, "ce", TaskRelation.INDEPENDENT)
        bad_gradient = GradientConfig(GradientStrategy.PCGRAD, -1.0, False)
        config = MultiTaskConfig(
            tasks=(task,),
            gradient_config=bad_gradient,
            balancing_method=TaskBalancing.UNIFORM,
            shared_layers=(),
        )
        with pytest.raises(ValueError, match="conflict_threshold must be non-negative"):
            validate_multitask_config(config)

    def test_duplicate_task_names_raises(self) -> None:
        """Duplicate task names raises ValueError."""
        task1 = TaskConfig("test", 1.0, "ce", TaskRelation.INDEPENDENT)
        task2 = TaskConfig("test", 0.5, "mse", TaskRelation.AUXILIARY)
        gradient = GradientConfig(GradientStrategy.SUM, 0.0, False)
        config = MultiTaskConfig(
            tasks=(task1, task2),
            gradient_config=gradient,
            balancing_method=TaskBalancing.UNIFORM,
            shared_layers=(),
        )
        with pytest.raises(ValueError, match="duplicate task names found"):
            validate_multitask_config(config)


class TestValidateMultiTaskStats:
    """Tests for validate_multitask_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = MultiTaskStats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": 0.9},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0},
        )
        validate_multitask_stats(stats)

    def test_negative_loss_raises(self) -> None:
        """Negative loss raises ValueError."""
        stats = MultiTaskStats(
            task_losses={"task1": -0.5},
            task_accuracies={"task1": 0.9},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0},
        )
        with pytest.raises(ValueError, match="task losses must be non-negative"):
            validate_multitask_stats(stats)

    def test_accuracy_below_zero_raises(self) -> None:
        """Accuracy below 0 raises ValueError."""
        stats = MultiTaskStats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": -0.1},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0},
        )
        with pytest.raises(ValueError, match="task accuracies must be between 0 and 1"):
            validate_multitask_stats(stats)

    def test_accuracy_above_one_raises(self) -> None:
        """Accuracy above 1 raises ValueError."""
        stats = MultiTaskStats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": 1.5},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0},
        )
        with pytest.raises(ValueError, match="task accuracies must be between 0 and 1"):
            validate_multitask_stats(stats)

    def test_negative_conflicts_raises(self) -> None:
        """Negative conflicts raises ValueError."""
        stats = MultiTaskStats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": 0.9},
            gradient_conflicts=-1,
            effective_weights={"task1": 1.0},
        )
        with pytest.raises(ValueError, match="gradient_conflicts must be non-negative"):
            validate_multitask_stats(stats)

    def test_negative_weight_raises(self) -> None:
        """Negative effective weight raises ValueError."""
        stats = MultiTaskStats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": 0.9},
            gradient_conflicts=5,
            effective_weights={"task1": -1.0},
        )
        with pytest.raises(ValueError, match="effective weights must be non-negative"):
            validate_multitask_stats(stats)


class TestCreateTaskConfig:
    """Tests for create_task_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_task_config("classification")
        assert config.name == "classification"
        assert config.weight == 1.0
        assert config.loss_type == "cross_entropy"
        assert config.relation == TaskRelation.INDEPENDENT

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_task_config(
            "auxiliary",
            weight=0.5,
            loss_type="mse",
            relation="auxiliary",
        )
        assert config.name == "auxiliary"
        assert config.weight == 0.5
        assert config.loss_type == "mse"
        assert config.relation == TaskRelation.AUXILIARY

    def test_with_enum_relation(self) -> None:
        """Create with enum relation."""
        config = create_task_config("test", relation=TaskRelation.HIERARCHICAL)
        assert config.relation == TaskRelation.HIERARCHICAL

    @pytest.mark.parametrize(
        "relation",
        ["independent", "auxiliary", "hierarchical"],
    )
    def test_all_relations(self, relation: str) -> None:
        """Test all task relations."""
        config = create_task_config("test", relation=relation)
        assert config.relation.value == relation

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="task name cannot be empty"):
            create_task_config("")


class TestCreateGradientConfig:
    """Tests for create_gradient_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_gradient_config()
        assert config.strategy == GradientStrategy.SUM
        assert config.conflict_threshold == 0.0
        assert config.normalize is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_gradient_config(
            strategy="pcgrad",
            conflict_threshold=0.1,
            normalize=True,
        )
        assert config.strategy == GradientStrategy.PCGRAD
        assert config.conflict_threshold == 0.1
        assert config.normalize is True

    def test_with_enum_strategy(self) -> None:
        """Create with enum strategy."""
        config = create_gradient_config(strategy=GradientStrategy.CAGRAD)
        assert config.strategy == GradientStrategy.CAGRAD

    @pytest.mark.parametrize(
        "strategy",
        ["sum", "pcgrad", "cagrad", "mgda", "nash"],
    )
    def test_all_strategies(self, strategy: str) -> None:
        """Test all gradient strategies."""
        config = create_gradient_config(strategy=strategy)
        assert config.strategy.value == strategy

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="conflict_threshold must be non-negative"):
            create_gradient_config(conflict_threshold=-1.0)


class TestCreateMultiTaskConfig:
    """Tests for create_multitask_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_multitask_config()
        assert config.balancing_method == TaskBalancing.UNIFORM
        assert config.gradient_config.strategy == GradientStrategy.SUM
        assert len(config.tasks) == 0

    def test_with_tasks(self) -> None:
        """Create config with tasks."""
        task = create_task_config("classification")
        config = create_multitask_config(tasks=(task,))
        assert len(config.tasks) == 1
        assert config.tasks[0].name == "classification"

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_multitask_config(
            strategy="pcgrad",
            balancing_method="uncertainty",
            normalize=True,
            shared_layers=("encoder", "attention"),
        )
        assert config.gradient_config.strategy == GradientStrategy.PCGRAD
        assert config.balancing_method == TaskBalancing.UNCERTAINTY
        assert config.gradient_config.normalize is True
        assert config.shared_layers == ("encoder", "attention")

    def test_with_enum_balancing(self) -> None:
        """Create with enum balancing."""
        config = create_multitask_config(balancing_method=TaskBalancing.GRADNORM)
        assert config.balancing_method == TaskBalancing.GRADNORM

    @pytest.mark.parametrize(
        "method",
        ["uniform", "uncertainty", "gradnorm", "pcgrad", "cagrad"],
    )
    def test_all_balancing_methods(self, method: str) -> None:
        """Test all balancing methods."""
        config = create_multitask_config(balancing_method=method)
        assert config.balancing_method.value == method

    def test_invalid_threshold_raises(self) -> None:
        """Invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="conflict_threshold must be non-negative"):
            create_multitask_config(conflict_threshold=-0.5)


class TestCreateMultiTaskStats:
    """Tests for create_multitask_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_multitask_stats()
        assert stats.gradient_conflicts == 0
        assert len(stats.task_losses) == 0
        assert len(stats.task_accuracies) == 0
        assert len(stats.effective_weights) == 0

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_multitask_stats(
            task_losses={"task1": 0.5},
            task_accuracies={"task1": 0.9},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0},
        )
        assert stats.task_losses["task1"] == 0.5
        assert stats.task_accuracies["task1"] == 0.9
        assert stats.gradient_conflicts == 5

    def test_negative_conflicts_raises(self) -> None:
        """Negative conflicts raises ValueError."""
        with pytest.raises(ValueError, match="gradient_conflicts must be non-negative"):
            create_multitask_stats(gradient_conflicts=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_task_balancing_methods_sorted(self) -> None:
        """Returns sorted list."""
        methods = list_task_balancing_methods()
        assert methods == sorted(methods)
        assert "uniform" in methods
        assert "uncertainty" in methods

    def test_list_gradient_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_gradient_strategies()
        assert strategies == sorted(strategies)
        assert "sum" in strategies
        assert "pcgrad" in strategies

    def test_list_task_relations_sorted(self) -> None:
        """Returns sorted list."""
        relations = list_task_relations()
        assert relations == sorted(relations)
        assert "independent" in relations
        assert "auxiliary" in relations


class TestGetTaskBalancing:
    """Tests for get_task_balancing function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("uniform", TaskBalancing.UNIFORM),
            ("uncertainty", TaskBalancing.UNCERTAINTY),
            ("gradnorm", TaskBalancing.GRADNORM),
            ("pcgrad", TaskBalancing.PCGRAD),
            ("cagrad", TaskBalancing.CAGRAD),
        ],
    )
    def test_all_methods(self, name: str, expected: TaskBalancing) -> None:
        """Test all valid methods."""
        assert get_task_balancing(name) == expected

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="balancing_method must be one of"):
            get_task_balancing("invalid")


class TestGetGradientStrategy:
    """Tests for get_gradient_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("sum", GradientStrategy.SUM),
            ("pcgrad", GradientStrategy.PCGRAD),
            ("cagrad", GradientStrategy.CAGRAD),
            ("mgda", GradientStrategy.MGDA),
            ("nash", GradientStrategy.NASH),
        ],
    )
    def test_all_strategies(self, name: str, expected: GradientStrategy) -> None:
        """Test all valid strategies."""
        assert get_gradient_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="gradient_strategy must be one of"):
            get_gradient_strategy("invalid")


class TestGetTaskRelation:
    """Tests for get_task_relation function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("independent", TaskRelation.INDEPENDENT),
            ("auxiliary", TaskRelation.AUXILIARY),
            ("hierarchical", TaskRelation.HIERARCHICAL),
        ],
    )
    def test_all_relations(self, name: str, expected: TaskRelation) -> None:
        """Test all valid relations."""
        assert get_task_relation(name) == expected

    def test_invalid_relation_raises(self) -> None:
        """Invalid relation raises ValueError."""
        with pytest.raises(ValueError, match="task_relation must be one of"):
            get_task_relation("invalid")


class TestCalculateTaskWeights:
    """Tests for calculate_task_weights function."""

    def test_uniform_weights(self) -> None:
        """Uniform method gives equal weights."""
        weights = calculate_task_weights((0.5, 0.3), TaskBalancing.UNIFORM)
        assert weights == (1.0, 1.0)

    def test_uniform_three_tasks(self) -> None:
        """Uniform works with three tasks."""
        weights = calculate_task_weights((0.5, 0.3, 0.7), TaskBalancing.UNIFORM)
        assert weights == (1.0, 1.0, 1.0)

    def test_uncertainty_weights(self) -> None:
        """Uncertainty method weights by inverse variance."""
        weights = calculate_task_weights(
            (0.5, 0.3),
            TaskBalancing.UNCERTAINTY,
            uncertainties=(0.1, 0.2),
        )
        # 1/0.1^2 = 100, 1/0.2^2 = 25
        assert weights[0] > weights[1]
        assert weights[0] == pytest.approx(100.0)
        assert weights[1] == pytest.approx(25.0)

    def test_uncertainty_missing_raises(self) -> None:
        """Missing uncertainties raises ValueError."""
        with pytest.raises(ValueError, match="uncertainties required"):
            calculate_task_weights((0.5, 0.3), TaskBalancing.UNCERTAINTY)

    def test_uncertainty_wrong_length_raises(self) -> None:
        """Wrong length uncertainties raises ValueError."""
        with pytest.raises(ValueError, match="uncertainties must match number"):
            calculate_task_weights(
                (0.5, 0.3),
                TaskBalancing.UNCERTAINTY,
                uncertainties=(0.1,),
            )

    def test_uncertainty_zero_raises(self) -> None:
        """Zero uncertainty raises ValueError."""
        with pytest.raises(ValueError, match="uncertainties must be positive"):
            calculate_task_weights(
                (0.5, 0.3),
                TaskBalancing.UNCERTAINTY,
                uncertainties=(0.0, 0.2),
            )

    def test_gradnorm_weights(self) -> None:
        """GradNorm method uses loss ratios."""
        weights = calculate_task_weights(
            (0.5, 0.2),
            TaskBalancing.GRADNORM,
            initial_losses=(1.0, 1.0),
        )
        # Ratios: 0.5, 0.2 -> normalized
        assert len(weights) == 2
        assert all(w > 0 for w in weights)

    def test_gradnorm_missing_raises(self) -> None:
        """Missing initial_losses raises ValueError."""
        with pytest.raises(ValueError, match="initial_losses required"):
            calculate_task_weights((0.5, 0.3), TaskBalancing.GRADNORM)

    def test_gradnorm_wrong_length_raises(self) -> None:
        """Wrong length initial_losses raises ValueError."""
        with pytest.raises(ValueError, match="initial_losses must match number"):
            calculate_task_weights(
                (0.5, 0.3),
                TaskBalancing.GRADNORM,
                initial_losses=(1.0,),
            )

    def test_pcgrad_returns_uniform(self) -> None:
        """PCGrad returns uniform weights (gradients modified instead)."""
        weights = calculate_task_weights((0.5, 0.3), TaskBalancing.PCGRAD)
        assert weights == (1.0, 1.0)

    def test_cagrad_returns_uniform(self) -> None:
        """CAGrad returns uniform weights (gradients modified instead)."""
        weights = calculate_task_weights((0.5, 0.3), TaskBalancing.CAGRAD)
        assert weights == (1.0, 1.0)

    def test_empty_losses_raises(self) -> None:
        """Empty losses raises ValueError."""
        with pytest.raises(ValueError, match="losses cannot be empty"):
            calculate_task_weights((), TaskBalancing.UNIFORM)

    def test_negative_loss_raises(self) -> None:
        """Negative loss raises ValueError."""
        with pytest.raises(ValueError, match="losses must be non-negative"):
            calculate_task_weights((-0.5, 0.3), TaskBalancing.UNIFORM)


class TestDetectGradientConflicts:
    """Tests for detect_gradient_conflicts function."""

    def test_opposite_gradients(self) -> None:
        """Opposite gradients are detected as conflicts."""
        conflicts = detect_gradient_conflicts(
            ((1.0, 0.0), (-1.0, 0.0)),
            threshold=0.0,
        )
        assert (0, 1) in conflicts

    def test_same_direction_no_conflict(self) -> None:
        """Same direction gradients have no conflict."""
        conflicts = detect_gradient_conflicts(
            ((1.0, 0.0), (1.0, 0.0)),
            threshold=0.0,
        )
        assert len(conflicts) == 0

    def test_orthogonal_gradients(self) -> None:
        """Orthogonal gradients with zero threshold are not conflicts."""
        conflicts = detect_gradient_conflicts(
            ((1.0, 0.0), (0.0, 1.0)),
            threshold=0.0,
        )
        assert len(conflicts) == 0

    def test_orthogonal_with_positive_threshold(self) -> None:
        """Orthogonal gradients with positive threshold are conflicts."""
        conflicts = detect_gradient_conflicts(
            ((1.0, 0.0), (0.0, 1.0)),
            threshold=0.5,
        )
        assert (0, 1) in conflicts

    def test_three_tasks(self) -> None:
        """Multiple task pairs detected."""
        conflicts = detect_gradient_conflicts(
            ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)),
            threshold=0.0,
        )
        assert (0, 1) in conflicts  # Opposite
        assert (0, 2) not in conflicts  # Orthogonal
        assert (1, 2) not in conflicts  # Orthogonal

    def test_single_task_no_conflicts(self) -> None:
        """Single task has no conflicts."""
        conflicts = detect_gradient_conflicts(((1.0, 0.0),), threshold=0.0)
        assert len(conflicts) == 0

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            detect_gradient_conflicts((), threshold=0.0)

    def test_threshold_out_of_range_raises(self) -> None:
        """Threshold out of range raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between -1 and 1"):
            detect_gradient_conflicts(((1.0,), (1.0,)), threshold=1.5)

    def test_mismatched_dimensions_raises(self) -> None:
        """Mismatched dimensions raises ValueError."""
        match_str = "all gradients must have the same dimension"
        with pytest.raises(ValueError, match=match_str):
            detect_gradient_conflicts(((1.0, 0.0), (1.0,)), threshold=0.0)

    def test_zero_gradient(self) -> None:
        """Zero gradient has no conflict detection (cosine is 0)."""
        conflicts = detect_gradient_conflicts(
            ((1.0, 0.0), (0.0, 0.0)),
            threshold=-0.5,  # Low threshold to avoid conflict
        )
        # Cosine with zero gradient is 0
        assert len(conflicts) == 0


class TestProjectConflictingGradients:
    """Tests for project_conflicting_gradients function."""

    def test_sum_strategy(self) -> None:
        """Sum strategy adds gradients."""
        result = project_conflicting_gradients(
            ((1.0, 0.0), (2.0, 0.0)),
            strategy=GradientStrategy.SUM,
        )
        assert result == (3.0, 0.0)

    def test_sum_opposite_cancel(self) -> None:
        """Opposite gradients cancel with sum."""
        result = project_conflicting_gradients(
            ((1.0, 0.0), (-1.0, 0.0)),
            strategy=GradientStrategy.SUM,
        )
        assert abs(result[0]) < 1e-10
        assert abs(result[1]) < 1e-10

    def test_pcgrad_strategy(self) -> None:
        """PCGrad projects out conflicts."""
        result = project_conflicting_gradients(
            ((1.0, 0.0), (-0.5, 1.0)),
            strategy=GradientStrategy.PCGRAD,
        )
        # Result should be non-conflicting
        assert len(result) == 2

    def test_cagrad_strategy(self) -> None:
        """CAGrad finds conflict-averse direction."""
        result = project_conflicting_gradients(
            ((1.0, 0.0), (0.0, 1.0)),
            strategy=GradientStrategy.CAGRAD,
        )
        assert len(result) == 2

    def test_mgda_strategy(self) -> None:
        """MGDA averages gradients."""
        result = project_conflicting_gradients(
            ((2.0, 0.0), (0.0, 2.0)),
            strategy=GradientStrategy.MGDA,
        )
        assert result == pytest.approx((1.0, 1.0))

    def test_nash_strategy(self) -> None:
        """Nash averages gradients."""
        result = project_conflicting_gradients(
            ((2.0, 0.0), (0.0, 2.0)),
            strategy=GradientStrategy.NASH,
        )
        assert result == pytest.approx((1.0, 1.0))

    def test_empty_gradients_raises(self) -> None:
        """Empty gradients raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot be empty"):
            project_conflicting_gradients(())

    def test_empty_dimension_raises(self) -> None:
        """Empty dimension raises ValueError."""
        with pytest.raises(ValueError, match="gradients cannot have zero dimension"):
            project_conflicting_gradients(((),))

    def test_mismatched_dimensions_raises(self) -> None:
        """Mismatched dimensions raises ValueError."""
        match_str = "all gradients must have the same dimension"
        with pytest.raises(ValueError, match=match_str):
            project_conflicting_gradients(((1.0, 0.0), (1.0,)))

    def test_three_gradients(self) -> None:
        """Works with three gradients."""
        result = project_conflicting_gradients(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            strategy=GradientStrategy.SUM,
        )
        assert result == (1.0, 1.0, 1.0)


class TestEstimateTaskDifficulty:
    """Tests for estimate_task_difficulty function."""

    def test_basic_difficulty(self) -> None:
        """Higher loss and lower accuracy means harder task."""
        difficulties = estimate_task_difficulty(
            (0.5, 0.1),
            (0.8, 0.95),
        )
        assert difficulties[0] > difficulties[1]

    def test_normalized_output(self) -> None:
        """Output is normalized to [0, 1]."""
        difficulties = estimate_task_difficulty(
            (0.5, 0.1, 0.3),
            (0.8, 0.95, 0.9),
        )
        assert all(0 <= d <= 1 for d in difficulties)
        assert max(difficulties) == pytest.approx(1.0)

    def test_perfect_accuracy(self) -> None:
        """Perfect accuracy gives low difficulty."""
        difficulties = estimate_task_difficulty(
            (0.5, 0.5),
            (1.0, 0.5),
        )
        assert difficulties[0] < difficulties[1]

    def test_empty_raises(self) -> None:
        """Empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="losses and accuracies cannot be empty"):
            estimate_task_difficulty((), ())

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths raise ValueError."""
        match_str = "losses and accuracies must have same length"
        with pytest.raises(ValueError, match=match_str):
            estimate_task_difficulty((0.5,), (0.8, 0.9))

    def test_negative_loss_raises(self) -> None:
        """Negative loss raises ValueError."""
        with pytest.raises(ValueError, match="losses must be non-negative"):
            estimate_task_difficulty((-0.5, 0.1), (0.8, 0.9))

    def test_accuracy_out_of_range_raises(self) -> None:
        """Accuracy out of range raises ValueError."""
        with pytest.raises(ValueError, match="accuracies must be between 0 and 1"):
            estimate_task_difficulty((0.5, 0.1), (0.8, 1.5))


class TestFormatMultiTaskStats:
    """Tests for format_multitask_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_multitask_stats(
            task_losses={"task1": 0.5, "task2": 0.3},
            task_accuracies={"task1": 0.9, "task2": 0.85},
            gradient_conflicts=5,
            effective_weights={"task1": 1.0, "task2": 1.2},
        )
        formatted = format_multitask_stats(stats)
        assert "Gradient Conflicts: 5" in formatted
        assert "task1" in formatted
        assert "task2" in formatted

    def test_empty_stats(self) -> None:
        """Format empty stats."""
        stats = create_multitask_stats()
        formatted = format_multitask_stats(stats)
        assert "Multi-Task Stats:" in formatted
        assert "Gradient Conflicts: 0" in formatted

    def test_contains_losses(self) -> None:
        """Formatted string contains losses."""
        stats = create_multitask_stats(task_losses={"task1": 0.5})
        formatted = format_multitask_stats(stats)
        assert "Task Losses:" in formatted
        assert "0.5" in formatted

    def test_contains_accuracies(self) -> None:
        """Formatted string contains accuracies."""
        stats = create_multitask_stats(task_accuracies={"task1": 0.9})
        formatted = format_multitask_stats(stats)
        assert "Task Accuracies:" in formatted
        assert "90.0%" in formatted


class TestGetRecommendedMultiTaskConfig:
    """Tests for get_recommended_multitask_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification tasks."""
        config = get_recommended_multitask_config("classification")
        assert config.balancing_method == TaskBalancing.UNIFORM
        assert config.gradient_config.strategy == GradientStrategy.SUM

    def test_generation_config(self) -> None:
        """Get config for generation tasks."""
        config = get_recommended_multitask_config("generation")
        assert config.balancing_method == TaskBalancing.GRADNORM
        assert config.gradient_config.strategy == GradientStrategy.CAGRAD

    def test_mixed_config(self) -> None:
        """Get config for mixed tasks."""
        config = get_recommended_multitask_config("mixed")
        assert config.balancing_method == TaskBalancing.UNCERTAINTY
        assert config.gradient_config.strategy == GradientStrategy.PCGRAD
        assert config.gradient_config.normalize is True

    def test_hierarchical_config(self) -> None:
        """Get config for hierarchical tasks."""
        config = get_recommended_multitask_config("hierarchical")
        assert config.balancing_method == TaskBalancing.UNCERTAINTY
        assert config.gradient_config.strategy == GradientStrategy.MGDA

    def test_invalid_task_type_raises(self) -> None:
        """Invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_multitask_config("unknown")

    @pytest.mark.parametrize(
        "task_type",
        ["classification", "generation", "mixed", "hierarchical"],
    )
    def test_all_task_types_return_valid_config(self, task_type: str) -> None:
        """All supported task types return valid configs."""
        config = get_recommended_multitask_config(task_type)
        validate_multitask_config(config)
