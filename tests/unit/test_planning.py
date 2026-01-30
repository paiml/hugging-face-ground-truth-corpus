"""Tests for agents.planning module."""

from __future__ import annotations

import math

import pytest

from hf_gtc.agents.planning import (
    VALID_PLANNING_STRATEGIES,
    VALID_TASK_STATUSES,
    ExecutionPlan,
    PlanConfig,
    PlanningStats,
    PlanningStrategy,
    PlanStep,
    TaskNode,
    TaskStatus,
    calculate_plan_progress,
    create_execution_plan,
    create_plan_config,
    create_plan_step,
    create_task_node,
    estimate_plan_complexity,
    get_planning_strategy,
    get_task_status,
    list_planning_strategies,
    list_task_statuses,
    validate_plan_config,
    validate_task_node,
)


class TestPlanningStrategy:
    """Tests for PlanningStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in PlanningStrategy:
            assert isinstance(strategy.value, str)

    def test_chain_of_thought_value(self) -> None:
        """Chain of thought has correct value."""
        assert PlanningStrategy.CHAIN_OF_THOUGHT.value == "chain_of_thought"

    def test_tree_of_thought_value(self) -> None:
        """Tree of thought has correct value."""
        assert PlanningStrategy.TREE_OF_THOUGHT.value == "tree_of_thought"

    def test_plan_and_solve_value(self) -> None:
        """Plan and solve has correct value."""
        assert PlanningStrategy.PLAN_AND_SOLVE.value == "plan_and_solve"

    def test_decomposition_value(self) -> None:
        """Decomposition has correct value."""
        assert PlanningStrategy.DECOMPOSITION.value == "decomposition"

    def test_reflexion_value(self) -> None:
        """Reflexion has correct value."""
        assert PlanningStrategy.REFLEXION.value == "reflexion"

    def test_valid_planning_strategies_frozenset(self) -> None:
        """VALID_PLANNING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_PLANNING_STRATEGIES, frozenset)

    def test_valid_planning_strategies_contains_all(self) -> None:
        """VALID_PLANNING_STRATEGIES contains all enum values."""
        for strategy in PlanningStrategy:
            assert strategy.value in VALID_PLANNING_STRATEGIES


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_have_values(self) -> None:
        """All statuses have string values."""
        for status in TaskStatus:
            assert isinstance(status.value, str)

    def test_pending_value(self) -> None:
        """Pending has correct value."""
        assert TaskStatus.PENDING.value == "pending"

    def test_in_progress_value(self) -> None:
        """In progress has correct value."""
        assert TaskStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_value(self) -> None:
        """Completed has correct value."""
        assert TaskStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        """Failed has correct value."""
        assert TaskStatus.FAILED.value == "failed"

    def test_skipped_value(self) -> None:
        """Skipped has correct value."""
        assert TaskStatus.SKIPPED.value == "skipped"

    def test_valid_task_statuses_frozenset(self) -> None:
        """VALID_TASK_STATUSES is a frozenset."""
        assert isinstance(VALID_TASK_STATUSES, frozenset)

    def test_valid_task_statuses_contains_all(self) -> None:
        """VALID_TASK_STATUSES contains all enum values."""
        for status in TaskStatus:
            assert status.value in VALID_TASK_STATUSES


class TestPlanConfig:
    """Tests for PlanConfig dataclass."""

    def test_create_config(self) -> None:
        """Create plan config."""
        config = PlanConfig(
            strategy=PlanningStrategy.CHAIN_OF_THOUGHT,
            max_steps=10,
            max_retries=3,
            allow_replanning=True,
        )
        assert config.strategy == PlanningStrategy.CHAIN_OF_THOUGHT
        assert config.max_steps == 10
        assert config.max_retries == 3
        assert config.allow_replanning is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 10, 3, True)
        with pytest.raises(AttributeError):
            config.max_steps = 20  # type: ignore[misc]


class TestTaskNode:
    """Tests for TaskNode dataclass."""

    def test_create_task_node(self) -> None:
        """Create task node."""
        task = TaskNode(
            task_id="task_001",
            description="Fetch user data",
            status=TaskStatus.PENDING,
            dependencies=(),
            result=None,
        )
        assert task.task_id == "task_001"
        assert task.description == "Fetch user data"
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == ()
        assert task.result is None

    def test_task_node_with_dependencies(self) -> None:
        """Create task node with dependencies."""
        task = TaskNode(
            task_id="task_002",
            description="Process data",
            status=TaskStatus.PENDING,
            dependencies=("task_001",),
            result=None,
        )
        assert task.dependencies == ("task_001",)

    def test_task_node_with_result(self) -> None:
        """Create task node with result."""
        task = TaskNode(
            task_id="task_001",
            description="Fetch user data",
            status=TaskStatus.COMPLETED,
            dependencies=(),
            result="User data fetched successfully",
        )
        assert task.result == "User data fetched successfully"

    def test_task_node_is_frozen(self) -> None:
        """Task node is immutable."""
        task = TaskNode("t1", "Description", TaskStatus.PENDING, (), None)
        with pytest.raises(AttributeError):
            task.status = TaskStatus.COMPLETED  # type: ignore[misc]


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_create_plan_step(self) -> None:
        """Create plan step."""
        step = PlanStep(
            step_number=1,
            action="Search for relevant documents",
            reasoning="Need information to answer the query",
            expected_outcome="List of relevant documents",
        )
        assert step.step_number == 1
        assert step.action == "Search for relevant documents"
        assert step.reasoning == "Need information to answer the query"
        assert step.expected_outcome == "List of relevant documents"

    def test_plan_step_is_frozen(self) -> None:
        """Plan step is immutable."""
        step = PlanStep(1, "Action", "Reasoning", "Outcome")
        with pytest.raises(AttributeError):
            step.action = "New action"  # type: ignore[misc]


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_create_execution_plan(self) -> None:
        """Create execution plan."""
        step1 = PlanStep(1, "Search", "Need info", "Results")
        step2 = PlanStep(2, "Analyze", "Process data", "Insights")
        plan = ExecutionPlan(
            goal="Answer user question",
            steps=(step1, step2),
            current_step=0,
            is_complete=False,
        )
        assert plan.goal == "Answer user question"
        assert len(plan.steps) == 2
        assert plan.current_step == 0
        assert plan.is_complete is False

    def test_execution_plan_is_frozen(self) -> None:
        """Execution plan is immutable."""
        plan = ExecutionPlan("Goal", (), 0, False)
        with pytest.raises(AttributeError):
            plan.goal = "New goal"  # type: ignore[misc]


class TestPlanningStats:
    """Tests for PlanningStats dataclass."""

    def test_create_planning_stats(self) -> None:
        """Create planning stats."""
        stats = PlanningStats(
            total_steps=10,
            completed_steps=7,
            failed_steps=1,
            replanning_count=2,
        )
        assert stats.total_steps == 10
        assert stats.completed_steps == 7
        assert stats.failed_steps == 1
        assert stats.replanning_count == 2

    def test_planning_stats_is_frozen(self) -> None:
        """Planning stats is immutable."""
        stats = PlanningStats(10, 5, 1, 0)
        with pytest.raises(AttributeError):
            stats.total_steps = 20  # type: ignore[misc]


class TestValidatePlanConfig:
    """Tests for validate_plan_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 10, 3, True)
        validate_plan_config(config)

    def test_none_config_raises(self) -> None:
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_plan_config(None)  # type: ignore[arg-type]

    def test_zero_max_steps_raises(self) -> None:
        """Zero max_steps raises ValueError."""
        config = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 0, 3, True)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            validate_plan_config(config)

    def test_negative_max_steps_raises(self) -> None:
        """Negative max_steps raises ValueError."""
        config = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, -5, 3, True)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            validate_plan_config(config)

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_retries raises ValueError."""
        config = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 10, -1, True)
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            validate_plan_config(config)

    def test_zero_max_retries_allowed(self) -> None:
        """Zero max_retries is allowed."""
        config = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 10, 0, True)
        validate_plan_config(config)  # Should not raise


class TestValidateTaskNode:
    """Tests for validate_task_node function."""

    def test_valid_task(self) -> None:
        """Valid task passes validation."""
        task = TaskNode("t1", "Do something", TaskStatus.PENDING, (), None)
        validate_task_node(task)

    def test_none_task_raises(self) -> None:
        """None task raises ValueError."""
        with pytest.raises(ValueError, match="task cannot be None"):
            validate_task_node(None)  # type: ignore[arg-type]

    def test_empty_task_id_raises(self) -> None:
        """Empty task_id raises ValueError."""
        task = TaskNode("", "Description", TaskStatus.PENDING, (), None)
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            validate_task_node(task)

    def test_whitespace_task_id_raises(self) -> None:
        """Whitespace task_id raises ValueError."""
        task = TaskNode("   ", "Description", TaskStatus.PENDING, (), None)
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            validate_task_node(task)

    def test_empty_description_raises(self) -> None:
        """Empty description raises ValueError."""
        task = TaskNode("t1", "", TaskStatus.PENDING, (), None)
        with pytest.raises(ValueError, match="description cannot be empty"):
            validate_task_node(task)

    def test_whitespace_description_raises(self) -> None:
        """Whitespace description raises ValueError."""
        task = TaskNode("t1", "   ", TaskStatus.PENDING, (), None)
        with pytest.raises(ValueError, match="description cannot be empty"):
            validate_task_node(task)


class TestCreatePlanConfig:
    """Tests for create_plan_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_plan_config()
        assert config.strategy == PlanningStrategy.CHAIN_OF_THOUGHT
        assert config.max_steps == 10
        assert config.max_retries == 3
        assert config.allow_replanning is True

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("chain_of_thought", PlanningStrategy.CHAIN_OF_THOUGHT),
            ("tree_of_thought", PlanningStrategy.TREE_OF_THOUGHT),
            ("plan_and_solve", PlanningStrategy.PLAN_AND_SOLVE),
            ("decomposition", PlanningStrategy.DECOMPOSITION),
            ("reflexion", PlanningStrategy.REFLEXION),
        ],
    )
    def test_all_strategies(self, strategy: str, expected: PlanningStrategy) -> None:
        """Create config with each strategy."""
        config = create_plan_config(strategy=strategy)
        assert config.strategy == expected

    def test_custom_max_steps(self) -> None:
        """Create config with custom max_steps."""
        config = create_plan_config(max_steps=20)
        assert config.max_steps == 20

    def test_custom_max_retries(self) -> None:
        """Create config with custom max_retries."""
        config = create_plan_config(max_retries=5)
        assert config.max_retries == 5

    def test_custom_allow_replanning(self) -> None:
        """Create config with allow_replanning False."""
        config = create_plan_config(allow_replanning=False)
        assert config.allow_replanning is False

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_plan_config(strategy="invalid")

    def test_zero_max_steps_raises(self) -> None:
        """Zero max_steps raises ValueError."""
        with pytest.raises(ValueError, match="max_steps must be positive"):
            create_plan_config(max_steps=0)

    def test_negative_max_steps_raises(self) -> None:
        """Negative max_steps raises ValueError."""
        with pytest.raises(ValueError, match="max_steps must be positive"):
            create_plan_config(max_steps=-1)

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            create_plan_config(max_retries=-1)


class TestCreateTaskNode:
    """Tests for create_task_node function."""

    def test_default_task(self) -> None:
        """Create default task."""
        task = create_task_node("t1", "Fetch data")
        assert task.task_id == "t1"
        assert task.description == "Fetch data"
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == ()
        assert task.result is None

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("pending", TaskStatus.PENDING),
            ("in_progress", TaskStatus.IN_PROGRESS),
            ("completed", TaskStatus.COMPLETED),
            ("failed", TaskStatus.FAILED),
            ("skipped", TaskStatus.SKIPPED),
        ],
    )
    def test_all_statuses(self, status: str, expected: TaskStatus) -> None:
        """Create task with each status."""
        task = create_task_node("t1", "Description", status=status)
        assert task.status == expected

    def test_with_dependencies(self) -> None:
        """Create task with dependencies."""
        task = create_task_node("t2", "Process data", dependencies=("t1",))
        assert task.dependencies == ("t1",)

    def test_with_multiple_dependencies(self) -> None:
        """Create task with multiple dependencies."""
        task = create_task_node("t3", "Aggregate", dependencies=("t1", "t2"))
        assert task.dependencies == ("t1", "t2")

    def test_with_result(self) -> None:
        """Create task with result."""
        task = create_task_node("t1", "Fetch", status="completed", result="Success")
        assert task.result == "Success"

    def test_empty_task_id_raises(self) -> None:
        """Empty task_id raises ValueError."""
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            create_task_node("", "Description")

    def test_empty_description_raises(self) -> None:
        """Empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            create_task_node("t1", "")

    def test_invalid_status_raises(self) -> None:
        """Invalid status raises ValueError."""
        with pytest.raises(ValueError, match="status must be one of"):
            create_task_node("t1", "Description", status="bad")


class TestCreatePlanStep:
    """Tests for create_plan_step function."""

    def test_minimal_step(self) -> None:
        """Create minimal step."""
        step = create_plan_step(1, "Search for information")
        assert step.step_number == 1
        assert step.action == "Search for information"
        assert step.reasoning == ""
        assert step.expected_outcome == ""

    def test_full_step(self) -> None:
        """Create step with all fields."""
        step = create_plan_step(
            2,
            "Analyze results",
            reasoning="Need to extract insights",
            expected_outcome="Summary of findings",
        )
        assert step.step_number == 2
        assert step.reasoning == "Need to extract insights"
        assert step.expected_outcome == "Summary of findings"

    def test_zero_step_number_raises(self) -> None:
        """Zero step_number raises ValueError."""
        with pytest.raises(ValueError, match="step_number must be positive"):
            create_plan_step(0, "Action")

    def test_negative_step_number_raises(self) -> None:
        """Negative step_number raises ValueError."""
        with pytest.raises(ValueError, match="step_number must be positive"):
            create_plan_step(-1, "Action")

    def test_empty_action_raises(self) -> None:
        """Empty action raises ValueError."""
        with pytest.raises(ValueError, match="action cannot be empty"):
            create_plan_step(1, "")

    def test_whitespace_action_raises(self) -> None:
        """Whitespace action raises ValueError."""
        with pytest.raises(ValueError, match="action cannot be empty"):
            create_plan_step(1, "   ")


class TestCreateExecutionPlan:
    """Tests for create_execution_plan function."""

    def test_minimal_plan(self) -> None:
        """Create minimal plan."""
        plan = create_execution_plan("Answer a question")
        assert plan.goal == "Answer a question"
        assert plan.steps == ()
        assert plan.current_step == 0
        assert plan.is_complete is False

    def test_plan_with_steps(self) -> None:
        """Create plan with steps."""
        step1 = create_plan_step(1, "Search")
        step2 = create_plan_step(2, "Analyze")
        plan = create_execution_plan(
            "Find answer",
            steps=(step1, step2),
            current_step=1,
        )
        assert len(plan.steps) == 2
        assert plan.current_step == 1

    def test_complete_plan(self) -> None:
        """Create complete plan."""
        step = create_plan_step(1, "Done")
        plan = create_execution_plan(
            "Goal", steps=(step,), current_step=1, is_complete=True
        )
        assert plan.is_complete is True

    def test_empty_goal_raises(self) -> None:
        """Empty goal raises ValueError."""
        with pytest.raises(ValueError, match="goal cannot be empty"):
            create_execution_plan("")

    def test_whitespace_goal_raises(self) -> None:
        """Whitespace goal raises ValueError."""
        with pytest.raises(ValueError, match="goal cannot be empty"):
            create_execution_plan("   ")

    def test_negative_current_step_raises(self) -> None:
        """Negative current_step raises ValueError."""
        with pytest.raises(ValueError, match="current_step cannot be negative"):
            create_execution_plan("Goal", current_step=-1)

    def test_current_step_exceeds_steps_raises(self) -> None:
        """current_step exceeding number of steps raises ValueError."""
        step = create_plan_step(1, "Action")
        with pytest.raises(ValueError, match="current_step exceeds number of steps"):
            create_execution_plan("Goal", steps=(step,), current_step=5)

    def test_current_step_at_boundary_allowed(self) -> None:
        """current_step at number of steps is allowed."""
        step1 = create_plan_step(1, "Action1")
        step2 = create_plan_step(2, "Action2")
        plan = create_execution_plan("Goal", steps=(step1, step2), current_step=2)
        assert plan.current_step == 2

    def test_empty_steps_any_current_step_allowed(self) -> None:
        """With empty steps, any non-negative current_step is allowed."""
        plan = create_execution_plan("Goal", steps=(), current_step=5)
        assert plan.current_step == 5


class TestListPlanningStrategies:
    """Tests for list_planning_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_planning_strategies()
        assert strategies == sorted(strategies)

    def test_contains_all_strategies(self) -> None:
        """Contains all planning strategies."""
        strategies = list_planning_strategies()
        assert "chain_of_thought" in strategies
        assert "tree_of_thought" in strategies
        assert "plan_and_solve" in strategies
        assert "decomposition" in strategies
        assert "reflexion" in strategies

    def test_returns_exactly_five(self) -> None:
        """Returns exactly 5 strategies."""
        strategies = list_planning_strategies()
        assert len(strategies) == 5


class TestListTaskStatuses:
    """Tests for list_task_statuses function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        statuses = list_task_statuses()
        assert statuses == sorted(statuses)

    def test_contains_all_statuses(self) -> None:
        """Contains all task statuses."""
        statuses = list_task_statuses()
        assert "pending" in statuses
        assert "in_progress" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "skipped" in statuses

    def test_returns_exactly_five(self) -> None:
        """Returns exactly 5 statuses."""
        statuses = list_task_statuses()
        assert len(statuses) == 5


class TestGetPlanningStrategy:
    """Tests for get_planning_strategy function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("chain_of_thought", PlanningStrategy.CHAIN_OF_THOUGHT),
            ("tree_of_thought", PlanningStrategy.TREE_OF_THOUGHT),
            ("plan_and_solve", PlanningStrategy.PLAN_AND_SOLVE),
            ("decomposition", PlanningStrategy.DECOMPOSITION),
            ("reflexion", PlanningStrategy.REFLEXION),
        ],
    )
    def test_all_strategies(self, name: str, expected: PlanningStrategy) -> None:
        """Get all strategies by name."""
        assert get_planning_strategy(name) == expected

    def test_invalid_name_raises(self) -> None:
        """Invalid name raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_planning_strategy("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_planning_strategy("")


class TestGetTaskStatus:
    """Tests for get_task_status function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("pending", TaskStatus.PENDING),
            ("in_progress", TaskStatus.IN_PROGRESS),
            ("completed", TaskStatus.COMPLETED),
            ("failed", TaskStatus.FAILED),
            ("skipped", TaskStatus.SKIPPED),
        ],
    )
    def test_all_statuses(self, name: str, expected: TaskStatus) -> None:
        """Get all statuses by name."""
        assert get_task_status(name) == expected

    def test_invalid_name_raises(self) -> None:
        """Invalid name raises ValueError."""
        with pytest.raises(ValueError, match="status must be one of"):
            get_task_status("invalid")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="status must be one of"):
            get_task_status("")


class TestCalculatePlanProgress:
    """Tests for calculate_plan_progress function."""

    def test_zero_progress(self) -> None:
        """Calculate zero progress."""
        stats = PlanningStats(10, 0, 0, 0)
        assert calculate_plan_progress(stats) == 0.0

    def test_full_progress(self) -> None:
        """Calculate full progress."""
        stats = PlanningStats(10, 10, 0, 0)
        assert calculate_plan_progress(stats) == 100.0

    def test_partial_progress(self) -> None:
        """Calculate partial progress."""
        stats = PlanningStats(10, 5, 0, 0)
        assert calculate_plan_progress(stats) == 50.0

    def test_progress_with_failures(self) -> None:
        """Failed steps count toward progress."""
        stats = PlanningStats(10, 7, 3, 0)
        assert calculate_plan_progress(stats) == 100.0

    def test_progress_with_only_failures(self) -> None:
        """Progress with only failed steps."""
        stats = PlanningStats(10, 0, 5, 0)
        assert calculate_plan_progress(stats) == 50.0

    def test_none_stats_raises(self) -> None:
        """None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            calculate_plan_progress(None)  # type: ignore[arg-type]

    def test_zero_total_steps_raises(self) -> None:
        """Zero total_steps raises ValueError."""
        stats = PlanningStats(0, 0, 0, 0)
        with pytest.raises(ValueError, match="total_steps must be positive"):
            calculate_plan_progress(stats)

    def test_negative_total_steps_raises(self) -> None:
        """Negative total_steps raises ValueError."""
        stats = PlanningStats(-1, 0, 0, 0)
        with pytest.raises(ValueError, match="total_steps must be positive"):
            calculate_plan_progress(stats)

    def test_replanning_count_ignored(self) -> None:
        """Replanning count doesn't affect progress."""
        stats = PlanningStats(10, 5, 0, 100)
        assert calculate_plan_progress(stats) == 50.0


class TestEstimatePlanComplexity:
    """Tests for estimate_plan_complexity function."""

    def test_empty_plan_complexity(self) -> None:
        """Empty plan has zero complexity."""
        plan = create_execution_plan("Goal", steps=())
        assert estimate_plan_complexity(plan) == 0.0

    def test_single_step_complexity(self) -> None:
        """Single step plan complexity."""
        step = create_plan_step(1, "Action")
        plan = create_execution_plan("Goal", steps=(step,))
        complexity = estimate_plan_complexity(plan)
        # Complexity = 1 * (1.0 + 0.1 * sqrt(1)) = 1 * 1.1 = 1.1
        assert complexity == pytest.approx(1.1)

    def test_four_step_complexity(self) -> None:
        """Four step plan complexity."""
        steps = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 5))
        plan = create_execution_plan("Goal", steps=steps)
        complexity = estimate_plan_complexity(plan)
        # Complexity = 4 * (1.0 + 0.1 * sqrt(4)) = 4 * 1.2 = 4.8
        assert complexity == pytest.approx(4.8)

    def test_complexity_increases_with_steps(self) -> None:
        """Complexity increases with more steps."""
        step1 = create_plan_step(1, "Action1")
        plan1 = create_execution_plan("Goal", steps=(step1,))

        steps3 = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 4))
        plan3 = create_execution_plan("Goal", steps=steps3)

        assert estimate_plan_complexity(plan3) > estimate_plan_complexity(plan1)

    def test_complexity_grows_with_multiplier(self) -> None:
        """Complexity grows with a multiplier based on sqrt(steps)."""
        steps4 = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 5))
        plan4 = create_execution_plan("Goal", steps=steps4)

        steps16 = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 17))
        plan16 = create_execution_plan("Goal", steps=steps16)

        c4 = estimate_plan_complexity(plan4)
        c16 = estimate_plan_complexity(plan16)

        # Verify the formula: multiplier grows with sqrt of steps
        # c4 = 4 * (1.0 + 0.1 * sqrt(4)) = 4 * 1.2 = 4.8
        # c16 = 16 * (1.0 + 0.1 * sqrt(16)) = 16 * 1.4 = 22.4
        # Ratio = 22.4 / 4.8 = ~4.67 (more than 4 due to growing multiplier)
        assert c16 > c4 * 4  # Growth is faster than linear due to sqrt multiplier

    def test_none_plan_raises(self) -> None:
        """None plan raises ValueError."""
        with pytest.raises(ValueError, match="plan cannot be None"):
            estimate_plan_complexity(None)  # type: ignore[arg-type]

    def test_complexity_formula(self) -> None:
        """Verify complexity formula."""
        steps9 = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 10))
        plan = create_execution_plan("Goal", steps=steps9)
        complexity = estimate_plan_complexity(plan)
        # Complexity = 9 * (1.0 + 0.1 * sqrt(9)) = 9 * 1.3 = 11.7
        expected = 9 * (1.0 + 0.1 * math.sqrt(9))
        assert complexity == pytest.approx(expected)

    def test_current_step_doesnt_affect_complexity(self) -> None:
        """current_step doesn't affect complexity calculation."""
        steps = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 4))
        plan1 = create_execution_plan("Goal", steps=steps, current_step=0)
        plan2 = create_execution_plan("Goal", steps=steps, current_step=2)
        assert estimate_plan_complexity(plan1) == estimate_plan_complexity(plan2)

    def test_is_complete_doesnt_affect_complexity(self) -> None:
        """is_complete doesn't affect complexity calculation."""
        steps = tuple(create_plan_step(i, f"Action {i}") for i in range(1, 4))
        plan1 = create_execution_plan("Goal", steps=steps, is_complete=False)
        plan2 = create_execution_plan("Goal", steps=steps, is_complete=True)
        assert estimate_plan_complexity(plan1) == estimate_plan_complexity(plan2)
