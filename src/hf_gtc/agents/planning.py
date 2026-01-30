"""Agent planning and task decomposition utilities.

This module provides functions for planning, task decomposition, and
execution tracking for agentic workflows.

Examples:
    >>> from hf_gtc.agents.planning import create_plan_config, PlanningStrategy
    >>> config = create_plan_config(strategy="chain_of_thought")
    >>> config.strategy
    <PlanningStrategy.CHAIN_OF_THOUGHT: 'chain_of_thought'>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PlanningStrategy(Enum):
    """Supported planning strategies for agents.

    Attributes:
        CHAIN_OF_THOUGHT: Sequential reasoning through steps.
        TREE_OF_THOUGHT: Branching exploration of alternatives.
        PLAN_AND_SOLVE: Upfront planning before execution.
        DECOMPOSITION: Breaking complex tasks into subtasks.
        REFLEXION: Iterative refinement with self-reflection.

    Examples:
        >>> PlanningStrategy.CHAIN_OF_THOUGHT.value
        'chain_of_thought'
        >>> PlanningStrategy.TREE_OF_THOUGHT.value
        'tree_of_thought'
        >>> PlanningStrategy.REFLEXION.value
        'reflexion'
    """

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    PLAN_AND_SOLVE = "plan_and_solve"
    DECOMPOSITION = "decomposition"
    REFLEXION = "reflexion"


VALID_PLANNING_STRATEGIES = frozenset(s.value for s in PlanningStrategy)


class TaskStatus(Enum):
    """Status of a task in a plan.

    Attributes:
        PENDING: Task has not started.
        IN_PROGRESS: Task is currently executing.
        COMPLETED: Task finished successfully.
        FAILED: Task failed during execution.
        SKIPPED: Task was skipped (e.g., due to dependency failure).

    Examples:
        >>> TaskStatus.PENDING.value
        'pending'
        >>> TaskStatus.COMPLETED.value
        'completed'
        >>> TaskStatus.SKIPPED.value
        'skipped'
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


VALID_TASK_STATUSES = frozenset(s.value for s in TaskStatus)


@dataclass(frozen=True, slots=True)
class PlanConfig:
    """Configuration for plan execution.

    Attributes:
        strategy: Planning strategy to use.
        max_steps: Maximum number of execution steps.
        max_retries: Maximum retries per failed step.
        allow_replanning: Whether to allow dynamic replanning.

    Examples:
        >>> config = PlanConfig(
        ...     strategy=PlanningStrategy.CHAIN_OF_THOUGHT,
        ...     max_steps=10,
        ...     max_retries=3,
        ...     allow_replanning=True,
        ... )
        >>> config.max_steps
        10
        >>> config.allow_replanning
        True
    """

    strategy: PlanningStrategy
    max_steps: int
    max_retries: int
    allow_replanning: bool


@dataclass(frozen=True, slots=True)
class TaskNode:
    """Represents a task in a plan graph.

    Attributes:
        task_id: Unique identifier for the task.
        description: Human-readable task description.
        status: Current status of the task.
        dependencies: Tuple of task IDs this task depends on.
        result: Result of the task execution, if completed.

    Examples:
        >>> task = TaskNode(
        ...     task_id="task_001",
        ...     description="Fetch user data",
        ...     status=TaskStatus.PENDING,
        ...     dependencies=(),
        ...     result=None,
        ... )
        >>> task.task_id
        'task_001'
        >>> task.status
        <TaskStatus.PENDING: 'pending'>
    """

    task_id: str
    description: str
    status: TaskStatus
    dependencies: tuple[str, ...]
    result: str | None


@dataclass(frozen=True, slots=True)
class PlanStep:
    """Represents a single step in an execution plan.

    Attributes:
        step_number: Sequential step number (1-indexed).
        action: Action to perform in this step.
        reasoning: Explanation of why this step is needed.
        expected_outcome: What the step should produce.

    Examples:
        >>> step = PlanStep(
        ...     step_number=1,
        ...     action="Search for relevant documents",
        ...     reasoning="Need information to answer the query",
        ...     expected_outcome="List of relevant documents",
        ... )
        >>> step.step_number
        1
        >>> step.action
        'Search for relevant documents'
    """

    step_number: int
    action: str
    reasoning: str
    expected_outcome: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """A complete execution plan for achieving a goal.

    Attributes:
        goal: The objective to achieve.
        steps: Tuple of plan steps in execution order.
        current_step: Index of the current step (0-indexed).
        is_complete: Whether the plan has finished execution.

    Examples:
        >>> step1 = PlanStep(1, "Search", "Need info", "Results")
        >>> step2 = PlanStep(2, "Analyze", "Process data", "Insights")
        >>> plan = ExecutionPlan(
        ...     goal="Answer user question",
        ...     steps=(step1, step2),
        ...     current_step=0,
        ...     is_complete=False,
        ... )
        >>> plan.goal
        'Answer user question'
        >>> len(plan.steps)
        2
    """

    goal: str
    steps: tuple[PlanStep, ...]
    current_step: int
    is_complete: bool


@dataclass(frozen=True, slots=True)
class PlanningStats:
    """Statistics for plan execution.

    Attributes:
        total_steps: Total number of steps in the plan.
        completed_steps: Number of successfully completed steps.
        failed_steps: Number of steps that failed.
        replanning_count: Number of times the plan was revised.

    Examples:
        >>> stats = PlanningStats(
        ...     total_steps=10,
        ...     completed_steps=7,
        ...     failed_steps=1,
        ...     replanning_count=2,
        ... )
        >>> stats.total_steps
        10
        >>> stats.completed_steps
        7
    """

    total_steps: int
    completed_steps: int
    failed_steps: int
    replanning_count: int


def validate_plan_config(config: PlanConfig) -> None:
    """Validate plan configuration.

    Args:
        config: PlanConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_steps is not positive.
        ValueError: If max_retries is negative.

    Examples:
        >>> config = PlanConfig(
        ...     PlanningStrategy.CHAIN_OF_THOUGHT, 10, 3, True
        ... )
        >>> validate_plan_config(config)  # No error

        >>> validate_plan_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 0, 3, True)
        >>> validate_plan_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_steps must be positive

        >>> bad = PlanConfig(PlanningStrategy.CHAIN_OF_THOUGHT, 10, -1, True)
        >>> validate_plan_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_retries cannot be negative
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.max_steps <= 0:
        msg = f"max_steps must be positive, got {config.max_steps}"
        raise ValueError(msg)

    if config.max_retries < 0:
        msg = f"max_retries cannot be negative, got {config.max_retries}"
        raise ValueError(msg)


def validate_task_node(task: TaskNode) -> None:
    """Validate a task node.

    Args:
        task: TaskNode to validate.

    Raises:
        ValueError: If task is None.
        ValueError: If task_id is empty.
        ValueError: If description is empty.

    Examples:
        >>> task = TaskNode("t1", "Do something", TaskStatus.PENDING, (), None)
        >>> validate_task_node(task)  # No error

        >>> validate_task_node(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task cannot be None

        >>> bad = TaskNode("", "Description", TaskStatus.PENDING, (), None)
        >>> validate_task_node(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_id cannot be empty

        >>> bad = TaskNode("t1", "", TaskStatus.PENDING, (), None)
        >>> validate_task_node(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: description cannot be empty
    """
    if task is None:
        msg = "task cannot be None"
        raise ValueError(msg)

    if not task.task_id or not task.task_id.strip():
        msg = "task_id cannot be empty"
        raise ValueError(msg)

    if not task.description or not task.description.strip():
        msg = "description cannot be empty"
        raise ValueError(msg)


def create_plan_config(
    strategy: str = "chain_of_thought",
    max_steps: int = 10,
    max_retries: int = 3,
    allow_replanning: bool = True,
) -> PlanConfig:
    """Create a plan configuration.

    Args:
        strategy: Planning strategy name. Defaults to "chain_of_thought".
        max_steps: Maximum execution steps. Defaults to 10.
        max_retries: Maximum retries per step. Defaults to 3.
        allow_replanning: Allow dynamic replanning. Defaults to True.

    Returns:
        PlanConfig with the specified settings.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If max_steps is not positive.
        ValueError: If max_retries is negative.

    Examples:
        >>> config = create_plan_config()
        >>> config.strategy
        <PlanningStrategy.CHAIN_OF_THOUGHT: 'chain_of_thought'>
        >>> config.max_steps
        10

        >>> config = create_plan_config(strategy="reflexion", max_steps=20)
        >>> config.strategy
        <PlanningStrategy.REFLEXION: 'reflexion'>

        >>> create_plan_config(strategy="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of

        >>> create_plan_config(max_steps=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_steps must be positive
    """
    if strategy not in VALID_PLANNING_STRATEGIES:
        msg = f"strategy must be one of {VALID_PLANNING_STRATEGIES}, got '{strategy}'"
        raise ValueError(msg)

    config = PlanConfig(
        strategy=PlanningStrategy(strategy),
        max_steps=max_steps,
        max_retries=max_retries,
        allow_replanning=allow_replanning,
    )
    validate_plan_config(config)
    return config


def create_task_node(
    task_id: str,
    description: str,
    status: str = "pending",
    dependencies: tuple[str, ...] = (),
    result: str | None = None,
) -> TaskNode:
    """Create a task node.

    Args:
        task_id: Unique task identifier.
        description: Task description.
        status: Task status name. Defaults to "pending".
        dependencies: Task IDs this depends on. Defaults to ().
        result: Task result. Defaults to None.

    Returns:
        TaskNode with the specified settings.

    Raises:
        ValueError: If task_id is empty.
        ValueError: If description is empty.
        ValueError: If status is invalid.

    Examples:
        >>> task = create_task_node("t1", "Fetch data")
        >>> task.task_id
        't1'
        >>> task.status
        <TaskStatus.PENDING: 'pending'>

        >>> task = create_task_node(
        ...     "t2", "Process data", status="completed",
        ...     dependencies=("t1",), result="Success"
        ... )
        >>> task.status
        <TaskStatus.COMPLETED: 'completed'>
        >>> task.dependencies
        ('t1',)

        >>> create_task_node("", "desc")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task_id cannot be empty

        >>> create_task_node("t1", "")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: description cannot be empty

        >>> create_task_node("t1", "d", status="bad")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: status must be one of
    """
    if status not in VALID_TASK_STATUSES:
        msg = f"status must be one of {VALID_TASK_STATUSES}, got '{status}'"
        raise ValueError(msg)

    task = TaskNode(
        task_id=task_id,
        description=description,
        status=TaskStatus(status),
        dependencies=dependencies,
        result=result,
    )
    validate_task_node(task)
    return task


def create_plan_step(
    step_number: int,
    action: str,
    reasoning: str = "",
    expected_outcome: str = "",
) -> PlanStep:
    """Create a plan step.

    Args:
        step_number: Step number (1-indexed).
        action: Action to perform.
        reasoning: Why this step is needed. Defaults to "".
        expected_outcome: Expected result. Defaults to "".

    Returns:
        PlanStep with the specified settings.

    Raises:
        ValueError: If step_number is not positive.
        ValueError: If action is empty.

    Examples:
        >>> step = create_plan_step(1, "Search for information")
        >>> step.step_number
        1
        >>> step.action
        'Search for information'

        >>> step = create_plan_step(
        ...     2, "Analyze results",
        ...     reasoning="Need to extract insights",
        ...     expected_outcome="Summary of findings"
        ... )
        >>> step.reasoning
        'Need to extract insights'

        >>> create_plan_step(0, "action")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: step_number must be positive

        >>> create_plan_step(1, "")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: action cannot be empty
    """
    if step_number <= 0:
        msg = f"step_number must be positive, got {step_number}"
        raise ValueError(msg)

    if not action or not action.strip():
        msg = "action cannot be empty"
        raise ValueError(msg)

    return PlanStep(
        step_number=step_number,
        action=action,
        reasoning=reasoning,
        expected_outcome=expected_outcome,
    )


def create_execution_plan(
    goal: str,
    steps: tuple[PlanStep, ...] = (),
    current_step: int = 0,
    is_complete: bool = False,
) -> ExecutionPlan:
    """Create an execution plan.

    Args:
        goal: The objective to achieve.
        steps: Plan steps in order. Defaults to ().
        current_step: Current step index. Defaults to 0.
        is_complete: Whether plan is complete. Defaults to False.

    Returns:
        ExecutionPlan with the specified settings.

    Raises:
        ValueError: If goal is empty.
        ValueError: If current_step is negative.
        ValueError: If current_step exceeds number of steps.

    Examples:
        >>> plan = create_execution_plan("Answer a question")
        >>> plan.goal
        'Answer a question'
        >>> plan.is_complete
        False

        >>> step1 = create_plan_step(1, "Search")
        >>> step2 = create_plan_step(2, "Analyze")
        >>> plan = create_execution_plan(
        ...     "Find answer",
        ...     steps=(step1, step2),
        ...     current_step=1
        ... )
        >>> len(plan.steps)
        2
        >>> plan.current_step
        1

        >>> create_execution_plan("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: goal cannot be empty

        >>> create_execution_plan("goal", current_step=-1)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: current_step cannot be negative

        >>> step = create_plan_step(1, "action")
        >>> create_execution_plan("goal", steps=(step,), current_step=5)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: current_step exceeds number of steps
    """
    if not goal or not goal.strip():
        msg = "goal cannot be empty"
        raise ValueError(msg)

    if current_step < 0:
        msg = f"current_step cannot be negative, got {current_step}"
        raise ValueError(msg)

    if len(steps) > 0 and current_step > len(steps):
        msg = f"current_step exceeds number of steps: {current_step} > {len(steps)}"
        raise ValueError(msg)

    return ExecutionPlan(
        goal=goal,
        steps=steps,
        current_step=current_step,
        is_complete=is_complete,
    )


def list_planning_strategies() -> list[str]:
    """List all available planning strategies.

    Returns:
        Sorted list of planning strategy names.

    Examples:
        >>> strategies = list_planning_strategies()
        >>> "chain_of_thought" in strategies
        True
        >>> "tree_of_thought" in strategies
        True
        >>> "reflexion" in strategies
        True
        >>> len(strategies)
        5
    """
    return sorted(VALID_PLANNING_STRATEGIES)


def list_task_statuses() -> list[str]:
    """List all available task statuses.

    Returns:
        Sorted list of task status names.

    Examples:
        >>> statuses = list_task_statuses()
        >>> "pending" in statuses
        True
        >>> "completed" in statuses
        True
        >>> "failed" in statuses
        True
        >>> len(statuses)
        5
    """
    return sorted(VALID_TASK_STATUSES)


def get_planning_strategy(name: str) -> PlanningStrategy:
    """Get planning strategy by name.

    Args:
        name: Strategy name.

    Returns:
        PlanningStrategy enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_planning_strategy("chain_of_thought")
        <PlanningStrategy.CHAIN_OF_THOUGHT: 'chain_of_thought'>

        >>> get_planning_strategy("reflexion")
        <PlanningStrategy.REFLEXION: 'reflexion'>

        >>> get_planning_strategy("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: strategy must be one of
    """
    if name not in VALID_PLANNING_STRATEGIES:
        msg = f"strategy must be one of {VALID_PLANNING_STRATEGIES}, got '{name}'"
        raise ValueError(msg)
    return PlanningStrategy(name)


def get_task_status(name: str) -> TaskStatus:
    """Get task status by name.

    Args:
        name: Status name.

    Returns:
        TaskStatus enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_task_status("pending")
        <TaskStatus.PENDING: 'pending'>

        >>> get_task_status("completed")
        <TaskStatus.COMPLETED: 'completed'>

        >>> get_task_status("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: status must be one of
    """
    if name not in VALID_TASK_STATUSES:
        msg = f"status must be one of {VALID_TASK_STATUSES}, got '{name}'"
        raise ValueError(msg)
    return TaskStatus(name)


def calculate_plan_progress(stats: PlanningStats) -> float:
    """Calculate plan completion percentage.

    Args:
        stats: PlanningStats to calculate progress from.

    Returns:
        Progress as a percentage (0.0 to 100.0).

    Raises:
        ValueError: If stats is None.
        ValueError: If total_steps is not positive.

    Examples:
        >>> stats = PlanningStats(10, 5, 0, 0)
        >>> calculate_plan_progress(stats)
        50.0

        >>> stats = PlanningStats(10, 10, 0, 0)
        >>> calculate_plan_progress(stats)
        100.0

        >>> stats = PlanningStats(10, 0, 0, 0)
        >>> calculate_plan_progress(stats)
        0.0

        >>> stats = PlanningStats(4, 3, 1, 0)
        >>> calculate_plan_progress(stats)
        100.0

        >>> calculate_plan_progress(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stats cannot be None

        >>> bad = PlanningStats(0, 0, 0, 0)
        >>> calculate_plan_progress(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_steps must be positive
    """
    if stats is None:
        msg = "stats cannot be None"
        raise ValueError(msg)

    if stats.total_steps <= 0:
        msg = f"total_steps must be positive, got {stats.total_steps}"
        raise ValueError(msg)

    # Both completed and failed steps count as "processed"
    processed = stats.completed_steps + stats.failed_steps
    return (processed / stats.total_steps) * 100.0


def estimate_plan_complexity(plan: ExecutionPlan) -> float:
    """Estimate plan complexity based on steps and structure.

    Complexity is calculated as:
    - Base: number of steps
    - Multiplier: 1.0 + 0.1 * sqrt(number of steps)

    This provides a complexity score that grows sub-linearly with
    the number of steps, reflecting that longer plans are harder
    to execute but not proportionally so.

    Args:
        plan: ExecutionPlan to estimate complexity for.

    Returns:
        Complexity score as a float (>= 0.0).

    Raises:
        ValueError: If plan is None.

    Examples:
        >>> step1 = create_plan_step(1, "Search")
        >>> plan = create_execution_plan("Goal", steps=(step1,))
        >>> complexity = estimate_plan_complexity(plan)
        >>> complexity > 0
        True

        >>> step2 = create_plan_step(2, "Analyze")
        >>> step3 = create_plan_step(3, "Synthesize")
        >>> plan2 = create_execution_plan("Goal", steps=(step1, step2, step3))
        >>> c2 = estimate_plan_complexity(plan2)
        >>> c2 > complexity
        True

        >>> empty_plan = create_execution_plan("Goal", steps=())
        >>> estimate_plan_complexity(empty_plan)
        0.0

        >>> estimate_plan_complexity(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: plan cannot be None
    """
    if plan is None:
        msg = "plan cannot be None"
        raise ValueError(msg)

    num_steps = len(plan.steps)

    if num_steps == 0:
        return 0.0

    # Base complexity is number of steps
    # Multiply by a factor that grows with sqrt of steps
    multiplier = 1.0 + 0.1 * math.sqrt(num_steps)

    return num_steps * multiplier
