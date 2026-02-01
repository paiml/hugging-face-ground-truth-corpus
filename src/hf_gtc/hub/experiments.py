"""Experiment tracking utilities for HuggingFace Hub.

This module provides dataclasses and functions for tracking ML experiments,
including metrics logging, artifact management, and experiment comparison.

Examples:
    >>> from hf_gtc.hub.experiments import create_experiment_config
    >>> config = create_experiment_config(name="bert-finetuning")
    >>> config.name
    'bert-finetuning'
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    pass


class ExperimentStatus(Enum):
    """Status of an experiment run.

    Attributes:
        PENDING: Experiment is queued but not started.
        RUNNING: Experiment is currently running.
        COMPLETED: Experiment finished successfully.
        FAILED: Experiment failed with an error.
        CANCELLED: Experiment was cancelled by user.

    Examples:
        >>> ExperimentStatus.PENDING.value
        'pending'
        >>> ExperimentStatus.RUNNING.value
        'running'
        >>> ExperimentStatus.COMPLETED.value
        'completed'
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


VALID_STATUSES = frozenset(s.value for s in ExperimentStatus)


class ArtifactType(Enum):
    """Type of artifact logged during an experiment.

    Attributes:
        MODEL: Model weights or checkpoint.
        CHECKPOINT: Training checkpoint.
        CONFIG: Configuration file.
        METRICS: Metrics file (JSON, CSV, etc.).
        LOGS: Log files.

    Examples:
        >>> ArtifactType.MODEL.value
        'model'
        >>> ArtifactType.CHECKPOINT.value
        'checkpoint'
        >>> ArtifactType.LOGS.value
        'logs'
    """

    MODEL = "model"
    CHECKPOINT = "checkpoint"
    CONFIG = "config"
    METRICS = "metrics"
    LOGS = "logs"


VALID_ARTIFACT_TYPES = frozenset(a.value for a in ArtifactType)


class MetricAggregation(Enum):
    """Aggregation method for metrics across steps.

    Attributes:
        LAST: Use the last logged value.
        BEST: Use the best value (depends on higher_is_better).
        MEAN: Use the mean of all logged values.
        MIN: Use the minimum value.
        MAX: Use the maximum value.

    Examples:
        >>> MetricAggregation.LAST.value
        'last'
        >>> MetricAggregation.BEST.value
        'best'
        >>> MetricAggregation.MEAN.value
        'mean'
    """

    LAST = "last"
    BEST = "best"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


VALID_AGGREGATIONS = frozenset(a.value for a in MetricAggregation)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for an experiment.

    Attributes:
        name: Experiment name.
        description: Human-readable description.
        tags: Tags for categorization.
        hyperparameters: Dictionary of hyperparameters.
        parent_run_id: ID of parent run (for nested experiments).

    Examples:
        >>> config = ExperimentConfig(
        ...     name="bert-finetuning",
        ...     description="Fine-tune BERT on MNLI",
        ...     tags=("nlp", "classification"),
        ...     hyperparameters={"lr": 2e-5, "epochs": 3},
        ...     parent_run_id=None,
        ... )
        >>> config.name
        'bert-finetuning'
        >>> config.hyperparameters["lr"]
        2e-05
    """

    name: str
    description: str
    tags: tuple[str, ...]
    hyperparameters: dict[str, Any]
    parent_run_id: str | None


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Configuration for a tracked metric.

    Attributes:
        name: Metric name (e.g., "accuracy", "loss").
        aggregation: How to aggregate values across steps.
        higher_is_better: True if higher values are better.
        log_frequency: Logging frequency in steps (0 = log every step).

    Examples:
        >>> config = MetricConfig(
        ...     name="accuracy",
        ...     aggregation=MetricAggregation.BEST,
        ...     higher_is_better=True,
        ...     log_frequency=100,
        ... )
        >>> config.name
        'accuracy'
        >>> config.higher_is_better
        True
    """

    name: str
    aggregation: MetricAggregation
    higher_is_better: bool
    log_frequency: int


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Configuration for a logged artifact.

    Attributes:
        artifact_type: Type of artifact.
        path: Path to the artifact file.
        metadata: Additional metadata about the artifact.
        versioned: Whether to version the artifact.

    Examples:
        >>> config = ArtifactConfig(
        ...     artifact_type=ArtifactType.MODEL,
        ...     path="outputs/model.safetensors",
        ...     metadata={"format": "safetensors"},
        ...     versioned=True,
        ... )
        >>> config.path
        'outputs/model.safetensors'
        >>> config.versioned
        True
    """

    artifact_type: ArtifactType
    path: str
    metadata: dict[str, Any]
    versioned: bool


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Represents a single experiment run.

    Attributes:
        run_id: Unique identifier for the run.
        config: Experiment configuration.
        status: Current run status.
        metrics: Dictionary mapping metric names to lists of (step, value).
        artifacts: List of artifact configurations.
        start_time: When the run started.
        end_time: When the run ended (None if still running).

    Examples:
        >>> from datetime import datetime, timezone
        >>> config = ExperimentConfig(
        ...     name="test",
        ...     description="Test run",
        ...     tags=(),
        ...     hyperparameters={},
        ...     parent_run_id=None,
        ... )
        >>> run = ExperimentRun(
        ...     run_id="run-123",
        ...     config=config,
        ...     status=ExperimentStatus.RUNNING,
        ...     metrics={},
        ...     artifacts=(),
        ...     start_time=datetime.now(timezone.utc),
        ...     end_time=None,
        ... )
        >>> run.run_id
        'run-123'
    """

    run_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    metrics: dict[str, list[tuple[int, float]]]
    artifacts: tuple[ArtifactConfig, ...]
    start_time: datetime
    end_time: datetime | None


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration.

    Args:
        config: Experiment configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ExperimentConfig(
        ...     "test", "desc", (), {}, None
        ... )
        >>> validate_experiment_config(config)  # No error

        >>> bad = ExperimentConfig("", "desc", (), {}, None)
        >>> validate_experiment_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not config.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if len(config.name) > 128:
        msg = f"name cannot exceed 128 characters, got {len(config.name)}"
        raise ValueError(msg)

    if len(config.description) > 1024:
        desc_len = len(config.description)
        msg = f"description cannot exceed 1024 characters, got {desc_len}"
        raise ValueError(msg)

    for tag in config.tags:
        if not tag:
            msg = "tags cannot contain empty strings"
            raise ValueError(msg)
        if len(tag) > 64:
            msg = f"tag cannot exceed 64 characters, got '{tag}'"
            raise ValueError(msg)


def validate_metric_config(config: MetricConfig) -> None:
    """Validate metric configuration.

    Args:
        config: Metric configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = MetricConfig("loss", MetricAggregation.LAST, False, 100)
        >>> validate_metric_config(config)  # No error

        >>> bad = MetricConfig("", MetricAggregation.LAST, False, 100)
        >>> validate_metric_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not config.name:
        msg = "name cannot be empty"
        raise ValueError(msg)

    if len(config.name) > 64:
        msg = f"name cannot exceed 64 characters, got {len(config.name)}"
        raise ValueError(msg)

    if config.log_frequency < 0:
        msg = f"log_frequency must be non-negative, got {config.log_frequency}"
        raise ValueError(msg)


def validate_artifact_config(config: ArtifactConfig) -> None:
    """Validate artifact configuration.

    Args:
        config: Artifact configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ArtifactConfig(
        ...     ArtifactType.MODEL, "model.pt", {}, True
        ... )
        >>> validate_artifact_config(config)  # No error

        >>> bad = ArtifactConfig(ArtifactType.MODEL, "", {}, True)
        >>> validate_artifact_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: path cannot be empty
    """
    if not config.path:
        msg = "path cannot be empty"
        raise ValueError(msg)

    if len(config.path) > 512:
        msg = f"path cannot exceed 512 characters, got {len(config.path)}"
        raise ValueError(msg)


def validate_experiment_run(run: ExperimentRun) -> None:
    """Validate experiment run.

    Args:
        run: Experiment run to validate.

    Raises:
        ValueError: If run is invalid.

    Examples:
        >>> from datetime import UTC, datetime
        >>> config = ExperimentConfig("test", "", (), {}, None)
        >>> run = ExperimentRun(
        ...     "run-1", config, ExperimentStatus.RUNNING,
        ...     {}, (), datetime.now(UTC), None
        ... )
        >>> validate_experiment_run(run)  # No error

        >>> bad_run = ExperimentRun(
        ...     "", config, ExperimentStatus.RUNNING,
        ...     {}, (), datetime.now(UTC), None
        ... )
        >>> validate_experiment_run(bad_run)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: run_id cannot be empty
    """
    if not run.run_id:
        msg = "run_id cannot be empty"
        raise ValueError(msg)

    validate_experiment_config(run.config)

    if run.end_time is not None and run.end_time < run.start_time:
        msg = "end_time cannot be before start_time"
        raise ValueError(msg)


def create_experiment_config(
    name: str,
    description: str = "",
    tags: tuple[str, ...] | list[str] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> ExperimentConfig:
    """Create an experiment configuration.

    Args:
        name: Experiment name.
        description: Human-readable description. Defaults to "".
        tags: Tags for categorization. Defaults to None.
        hyperparameters: Hyperparameters dict. Defaults to None.
        parent_run_id: Parent run ID. Defaults to None.

    Returns:
        ExperimentConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_experiment_config("bert-finetuning")
        >>> config.name
        'bert-finetuning'

        >>> config = create_experiment_config(
        ...     "training",
        ...     tags=["nlp", "bert"],
        ...     hyperparameters={"lr": 1e-4},
        ... )
        >>> config.tags
        ('nlp', 'bert')

        >>> create_experiment_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    tags_tuple = tuple(tags) if tags is not None else ()
    hyperparams = hyperparameters if hyperparameters is not None else {}

    config = ExperimentConfig(
        name=name,
        description=description,
        tags=tags_tuple,
        hyperparameters=hyperparams,
        parent_run_id=parent_run_id,
    )
    validate_experiment_config(config)
    return config


def create_metric_config(
    name: str,
    aggregation: str = "last",
    higher_is_better: bool = True,
    log_frequency: int = 0,
) -> MetricConfig:
    """Create a metric configuration.

    Args:
        name: Metric name.
        aggregation: Aggregation method. Defaults to "last".
        higher_is_better: Whether higher values are better. Defaults to True.
        log_frequency: Logging frequency in steps. Defaults to 0.

    Returns:
        MetricConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_metric_config("accuracy")
        >>> config.name
        'accuracy'

        >>> config = create_metric_config(
        ...     "loss", aggregation="min", higher_is_better=False
        ... )
        >>> config.aggregation
        <MetricAggregation.MIN: 'min'>
        >>> config.higher_is_better
        False

        >>> create_metric_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty

        >>> create_metric_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "x", aggregation="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: aggregation must be one of
    """
    if aggregation not in VALID_AGGREGATIONS:
        msg = f"aggregation must be one of {VALID_AGGREGATIONS}, got '{aggregation}'"
        raise ValueError(msg)

    config = MetricConfig(
        name=name,
        aggregation=MetricAggregation(aggregation),
        higher_is_better=higher_is_better,
        log_frequency=log_frequency,
    )
    validate_metric_config(config)
    return config


def create_artifact_config(
    artifact_type: str,
    path: str,
    metadata: dict[str, Any] | None = None,
    versioned: bool = True,
) -> ArtifactConfig:
    """Create an artifact configuration.

    Args:
        artifact_type: Type of artifact.
        path: Path to the artifact.
        metadata: Additional metadata. Defaults to None.
        versioned: Whether to version the artifact. Defaults to True.

    Returns:
        ArtifactConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_artifact_config("model", "outputs/model.pt")
        >>> config.path
        'outputs/model.pt'

        >>> config = create_artifact_config(
        ...     "checkpoint",
        ...     "ckpt/step-1000.pt",
        ...     metadata={"step": 1000},
        ... )
        >>> config.metadata["step"]
        1000

        >>> create_artifact_config("model", "")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: path cannot be empty

        >>> create_artifact_config("invalid", "x")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: artifact_type must be one of
    """
    if artifact_type not in VALID_ARTIFACT_TYPES:
        msg = f"artifact_type must be one of {VALID_ARTIFACT_TYPES}, got '{artifact_type}'"  # noqa: E501
        raise ValueError(msg)

    config = ArtifactConfig(
        artifact_type=ArtifactType(artifact_type),
        path=path,
        metadata=metadata if metadata is not None else {},
        versioned=versioned,
    )
    validate_artifact_config(config)
    return config


def create_experiment_run(
    config: ExperimentConfig,
    run_id: str | None = None,
) -> ExperimentRun:
    """Create a new experiment run.

    Args:
        config: Experiment configuration.
        run_id: Optional run ID. If None, generates a UUID.

    Returns:
        ExperimentRun with PENDING status.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = create_experiment_config("test")
        >>> run = create_experiment_run(config)
        >>> run.status
        <ExperimentStatus.PENDING: 'pending'>
        >>> run.config.name
        'test'

        >>> run = create_experiment_run(config, run_id="custom-id")
        >>> run.run_id
        'custom-id'
    """
    validate_experiment_config(config)

    return ExperimentRun(
        run_id=run_id if run_id is not None else str(uuid4()),
        config=config,
        status=ExperimentStatus.PENDING,
        metrics={},
        artifacts=(),
        start_time=datetime.now(UTC),
        end_time=None,
    )


def log_metric(
    run: ExperimentRun,
    metric_name: str,
    value: float,
    step: int = 0,
) -> ExperimentRun:
    """Log a metric value for an experiment run.

    Args:
        run: The experiment run.
        metric_name: Name of the metric.
        value: Metric value.
        step: Training step. Defaults to 0.

    Returns:
        Updated ExperimentRun with the new metric value.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_experiment_config("test")
        >>> run = create_experiment_run(config)
        >>> run = log_metric(run, "loss", 0.5, step=100)
        >>> run.metrics["loss"]
        [(100, 0.5)]

        >>> run = log_metric(run, "loss", 0.3, step=200)
        >>> len(run.metrics["loss"])
        2

        >>> log_metric(run, "", 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metric_name cannot be empty
    """
    if not metric_name:
        msg = "metric_name cannot be empty"
        raise ValueError(msg)

    if step < 0:
        msg = f"step must be non-negative, got {step}"
        raise ValueError(msg)

    # Create new metrics dict with the updated value
    new_metrics = dict(run.metrics)
    if metric_name not in new_metrics:
        new_metrics[metric_name] = []
    new_metrics[metric_name] = [*new_metrics[metric_name], (step, value)]

    return ExperimentRun(
        run_id=run.run_id,
        config=run.config,
        status=run.status,
        metrics=new_metrics,
        artifacts=run.artifacts,
        start_time=run.start_time,
        end_time=run.end_time,
    )


def log_artifact(
    run: ExperimentRun,
    artifact_config: ArtifactConfig,
) -> ExperimentRun:
    """Log an artifact for an experiment run.

    Args:
        run: The experiment run.
        artifact_config: Artifact configuration.

    Returns:
        Updated ExperimentRun with the new artifact.

    Examples:
        >>> config = create_experiment_config("test")
        >>> run = create_experiment_run(config)
        >>> artifact = create_artifact_config("model", "model.pt")
        >>> run = log_artifact(run, artifact)
        >>> len(run.artifacts)
        1
        >>> run.artifacts[0].path
        'model.pt'
    """
    validate_artifact_config(artifact_config)

    return ExperimentRun(
        run_id=run.run_id,
        config=run.config,
        status=run.status,
        metrics=run.metrics,
        artifacts=(*run.artifacts, artifact_config),
        start_time=run.start_time,
        end_time=run.end_time,
    )


def list_statuses() -> list[str]:
    """List all valid experiment statuses.

    Returns:
        Sorted list of status names.

    Examples:
        >>> statuses = list_statuses()
        >>> "running" in statuses
        True
        >>> "completed" in statuses
        True
        >>> statuses == sorted(statuses)
        True
    """
    return sorted(VALID_STATUSES)


def list_artifact_types() -> list[str]:
    """List all valid artifact types.

    Returns:
        Sorted list of artifact type names.

    Examples:
        >>> types = list_artifact_types()
        >>> "model" in types
        True
        >>> "checkpoint" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_ARTIFACT_TYPES)


def list_aggregations() -> list[str]:
    """List all valid metric aggregation methods.

    Returns:
        Sorted list of aggregation names.

    Examples:
        >>> aggs = list_aggregations()
        >>> "last" in aggs
        True
        >>> "best" in aggs
        True
        >>> aggs == sorted(aggs)
        True
    """
    return sorted(VALID_AGGREGATIONS)


def get_status(name: str) -> ExperimentStatus:
    """Get experiment status from string name.

    Args:
        name: Status name.

    Returns:
        ExperimentStatus enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_status("running")
        <ExperimentStatus.RUNNING: 'running'>

        >>> get_status("completed")
        <ExperimentStatus.COMPLETED: 'completed'>

        >>> get_status("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: status must be one of
    """
    if name not in VALID_STATUSES:
        msg = f"status must be one of {VALID_STATUSES}, got '{name}'"
        raise ValueError(msg)
    return ExperimentStatus(name)


def get_artifact_type(name: str) -> ArtifactType:
    """Get artifact type from string name.

    Args:
        name: Artifact type name.

    Returns:
        ArtifactType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_artifact_type("model")
        <ArtifactType.MODEL: 'model'>

        >>> get_artifact_type("checkpoint")
        <ArtifactType.CHECKPOINT: 'checkpoint'>

        >>> get_artifact_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: artifact_type must be one of
    """
    if name not in VALID_ARTIFACT_TYPES:
        msg = f"artifact_type must be one of {VALID_ARTIFACT_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ArtifactType(name)


def get_aggregation(name: str) -> MetricAggregation:
    """Get metric aggregation from string name.

    Args:
        name: Aggregation name.

    Returns:
        MetricAggregation enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_aggregation("last")
        <MetricAggregation.LAST: 'last'>

        >>> get_aggregation("best")
        <MetricAggregation.BEST: 'best'>

        >>> get_aggregation("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: aggregation must be one of
    """
    if name not in VALID_AGGREGATIONS:
        msg = f"aggregation must be one of {VALID_AGGREGATIONS}, got '{name}'"
        raise ValueError(msg)
    return MetricAggregation(name)


def _aggregate_metric(
    values: list[tuple[int, float]],
    aggregation: MetricAggregation,
    higher_is_better: bool,
) -> float:
    """Aggregate metric values according to aggregation method.

    Args:
        values: List of (step, value) tuples.
        aggregation: Aggregation method.
        higher_is_better: Whether higher is better (for BEST).

    Returns:
        Aggregated value.

    Raises:
        ValueError: If values list is empty.

    Examples:
        >>> vals = [(0, 1.0), (1, 2.0), (2, 3.0)]
        >>> _aggregate_metric(vals, MetricAggregation.LAST, True)
        3.0
        >>> _aggregate_metric(vals, MetricAggregation.MEAN, True)
        2.0
        >>> _aggregate_metric(vals, MetricAggregation.MIN, True)
        1.0
        >>> _aggregate_metric(vals, MetricAggregation.MAX, True)
        3.0
        >>> _aggregate_metric([(0, 3.0), (1, 1.0)], MetricAggregation.BEST, True)
        3.0
        >>> _aggregate_metric([(0, 3.0), (1, 1.0)], MetricAggregation.BEST, False)
        1.0
    """
    if not values:
        msg = "values list cannot be empty"
        raise ValueError(msg)

    just_values = [v for _, v in values]

    if aggregation == MetricAggregation.LAST:
        # Sort by step and get last value
        sorted_values = sorted(values, key=lambda x: x[0])
        return sorted_values[-1][1]
    elif aggregation == MetricAggregation.MEAN:
        return sum(just_values) / len(just_values)
    elif aggregation == MetricAggregation.MIN:
        return min(just_values)
    elif aggregation == MetricAggregation.MAX:
        return max(just_values)
    elif aggregation == MetricAggregation.BEST:
        return max(just_values) if higher_is_better else min(just_values)

    # Should not reach here
    msg = f"Unknown aggregation: {aggregation}"
    raise ValueError(msg)


def get_best_run(
    runs: list[ExperimentRun],
    metric_name: str,
    higher_is_better: bool = True,
    aggregation: str = "best",
) -> ExperimentRun | None:
    """Get the best run based on a metric.

    Args:
        runs: List of experiment runs.
        metric_name: Metric to compare.
        higher_is_better: Whether higher values are better. Defaults to True.
        aggregation: How to aggregate metric values. Defaults to "best".

    Returns:
        Best run or None if no runs have the metric.

    Raises:
        ValueError: If aggregation is invalid.

    Examples:
        >>> config = create_experiment_config("test")
        >>> run1 = create_experiment_run(config, run_id="run-1")
        >>> run1 = log_metric(run1, "accuracy", 0.8)
        >>> run2 = create_experiment_run(config, run_id="run-2")
        >>> run2 = log_metric(run2, "accuracy", 0.9)
        >>> best = get_best_run([run1, run2], "accuracy")
        >>> best.run_id
        'run-2'

        >>> best = get_best_run([run1, run2], "accuracy", higher_is_better=False)
        >>> best.run_id
        'run-1'

        >>> get_best_run([], "accuracy") is None
        True

        >>> get_best_run([run1], "nonexistent") is None
        True
    """
    if aggregation not in VALID_AGGREGATIONS:
        msg = f"aggregation must be one of {VALID_AGGREGATIONS}, got '{aggregation}'"
        raise ValueError(msg)

    agg = MetricAggregation(aggregation)

    runs_with_metric: list[tuple[ExperimentRun, float]] = []
    for run in runs:
        metric_values = run.metrics.get(metric_name)
        if metric_values:
            agg_value = _aggregate_metric(metric_values, agg, higher_is_better)
            runs_with_metric.append((run, agg_value))

    if not runs_with_metric:
        return None

    if higher_is_better:
        return max(runs_with_metric, key=lambda x: x[1])[0]
    return min(runs_with_metric, key=lambda x: x[1])[0]


def compare_runs(
    runs: list[ExperimentRun],
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compare multiple runs across metrics.

    Args:
        runs: List of experiment runs.
        metric_names: Metrics to compare. If None, uses all metrics.

    Returns:
        Dict mapping run_id to dict of metric_name to last value.

    Examples:
        >>> config = create_experiment_config("test")
        >>> run1 = create_experiment_run(config, run_id="run-1")
        >>> run1 = log_metric(run1, "accuracy", 0.8)
        >>> run1 = log_metric(run1, "loss", 0.2)
        >>> run2 = create_experiment_run(config, run_id="run-2")
        >>> run2 = log_metric(run2, "accuracy", 0.9)
        >>> comparison = compare_runs([run1, run2])
        >>> comparison["run-1"]["accuracy"]
        0.8
        >>> comparison["run-2"]["accuracy"]
        0.9
        >>> comparison["run-2"]["loss"] is None
        True

        >>> compare_runs([])
        {}
    """
    if not runs:
        return {}

    # Collect all metric names if not specified
    if metric_names is None:
        all_metrics: set[str] = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())
        metric_names = sorted(all_metrics)

    result: dict[str, dict[str, float | None]] = {}
    for run in runs:
        result[run.run_id] = {}
        for metric_name in metric_names:
            metric_values = run.metrics.get(metric_name)
            if metric_values:
                # Get last value (sorted by step)
                sorted_values = sorted(metric_values, key=lambda x: x[0])
                result[run.run_id][metric_name] = sorted_values[-1][1]
            else:
                result[run.run_id][metric_name] = None

    return result


@dataclass(frozen=True, slots=True)
class ExperimentStats:
    """Statistics for a set of experiment runs.

    Attributes:
        total_runs: Total number of runs.
        completed_runs: Number of completed runs.
        failed_runs: Number of failed runs.
        avg_duration_seconds: Average run duration in seconds.
        metric_stats: Dict mapping metric name to (min, max, mean).

    Examples:
        >>> stats = ExperimentStats(
        ...     total_runs=10,
        ...     completed_runs=8,
        ...     failed_runs=2,
        ...     avg_duration_seconds=3600.0,
        ...     metric_stats={"accuracy": (0.7, 0.95, 0.85)},
        ... )
        >>> stats.total_runs
        10
    """

    total_runs: int
    completed_runs: int
    failed_runs: int
    avg_duration_seconds: float | None
    metric_stats: dict[str, tuple[float, float, float]]


def _compute_avg_duration(runs: list[ExperimentRun]) -> float | None:
    """Compute average duration for runs with end_time."""
    durations: list[float] = []
    for run in runs:
        if run.end_time is not None:
            delta = run.end_time - run.start_time
            durations.append(delta.total_seconds())
    return sum(durations) / len(durations) if durations else None


def _compute_metric_stats(
    runs: list[ExperimentRun],
) -> dict[str, tuple[float, float, float]]:
    """Compute per-metric min/max/mean across runs."""
    metric_values: dict[str, list[float]] = {}
    for run in runs:
        for metric_name, values in run.metrics.items():
            if not values:
                continue
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            sorted_values = sorted(values, key=lambda x: x[0])
            metric_values[metric_name].append(sorted_values[-1][1])

    return {
        name: (min(vals), max(vals), sum(vals) / len(vals))
        for name, vals in metric_values.items()
        if vals
    }


def calculate_experiment_stats(runs: list[ExperimentRun]) -> ExperimentStats:
    """Calculate statistics for a set of experiment runs.

    Args:
        runs: List of experiment runs.

    Returns:
        ExperimentStats with computed statistics.

    Examples:
        >>> config = create_experiment_config("test")
        >>> run1 = create_experiment_run(config, run_id="run-1")
        >>> run1 = log_metric(run1, "accuracy", 0.8)
        >>> run2 = create_experiment_run(config, run_id="run-2")
        >>> run2 = log_metric(run2, "accuracy", 0.9)
        >>> stats = calculate_experiment_stats([run1, run2])
        >>> stats.total_runs
        2
        >>> min_v, max_v, mean_v = stats.metric_stats["accuracy"]
        >>> (min_v, max_v, round(mean_v, 2))
        (0.8, 0.9, 0.85)

        >>> empty_stats = calculate_experiment_stats([])
        >>> empty_stats.total_runs
        0
    """
    if not runs:
        return ExperimentStats(
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            avg_duration_seconds=None,
            metric_stats={},
        )

    completed = sum(1 for r in runs if r.status == ExperimentStatus.COMPLETED)
    failed = sum(1 for r in runs if r.status == ExperimentStatus.FAILED)
    avg_duration = _compute_avg_duration(runs)
    metric_stats = _compute_metric_stats(runs)

    return ExperimentStats(
        total_runs=len(runs),
        completed_runs=completed,
        failed_runs=failed,
        avg_duration_seconds=avg_duration,
        metric_stats=metric_stats,
    )


def format_experiment_summary(run: ExperimentRun) -> str:
    """Format a human-readable summary of an experiment run.

    Args:
        run: The experiment run.

    Returns:
        Formatted summary string.

    Examples:
        >>> config = create_experiment_config("bert-finetuning", tags=["nlp"])
        >>> run = create_experiment_run(config, run_id="run-123")
        >>> run = log_metric(run, "accuracy", 0.95)
        >>> summary = format_experiment_summary(run)
        >>> "bert-finetuning" in summary
        True
        >>> "accuracy" in summary
        True
        >>> "0.95" in summary
        True
    """
    lines = [
        f"Experiment: {run.config.name}",
        f"Run ID: {run.run_id}",
        f"Status: {run.status.value}",
    ]

    if run.config.description:
        lines.append(f"Description: {run.config.description}")

    if run.config.tags:
        lines.append(f"Tags: {', '.join(run.config.tags)}")

    _append_hyperparameters(lines, run.config.hyperparameters)
    _append_metrics_summary(lines, run.metrics)
    _append_artifacts_summary(lines, run.artifacts)

    return "\n".join(lines)


def _append_hyperparameters(
    lines: list[str], hyperparameters: dict[str, object]
) -> None:
    """Append hyperparameters section if present."""
    if not hyperparameters:
        return
    lines.append("Hyperparameters:")
    for key, value in hyperparameters.items():
        lines.append(f"  {key}: {value}")


def _append_metrics_summary(
    lines: list[str], metrics: dict[str, list[tuple[int, float]]]
) -> None:
    """Append metrics summary section if present."""
    if not metrics:
        return
    lines.append("Metrics:")
    for metric_name, values in metrics.items():
        if values:
            sorted_values = sorted(values, key=lambda x: x[0])
            lines.append(f"  {metric_name}: {sorted_values[-1][1]}")


def _append_artifacts_summary(
    lines: list[str], artifacts: tuple[ArtifactConfig, ...]
) -> None:
    """Append artifacts section if present."""
    if not artifacts:
        return
    lines.append(f"Artifacts: {len(artifacts)}")
    for artifact in artifacts:
        lines.append(f"  [{artifact.artifact_type.value}] {artifact.path}")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Recommended tracking configuration.

    Attributes:
        metrics: List of recommended metric configurations.
        artifacts: List of recommended artifact types.
        log_system_metrics: Whether to log system metrics.
        checkpoint_frequency: How often to save checkpoints (epochs).

    Examples:
        >>> config = TrackingConfig(
        ...     metrics=(
        ...         MetricConfig("loss", MetricAggregation.MIN, False, 100),
        ...     ),
        ...     artifacts=("model", "checkpoint"),
        ...     log_system_metrics=True,
        ...     checkpoint_frequency=1,
        ... )
        >>> config.log_system_metrics
        True
    """

    metrics: tuple[MetricConfig, ...]
    artifacts: tuple[str, ...]
    log_system_metrics: bool
    checkpoint_frequency: int


def get_recommended_tracking_config(
    task_type: str = "classification",
) -> TrackingConfig:
    """Get recommended tracking configuration for a task type.

    Args:
        task_type: Type of ML task (classification, generation, regression).
            Defaults to "classification".

    Returns:
        TrackingConfig with recommended settings.

    Raises:
        ValueError: If task_type is unknown.

    Examples:
        >>> config = get_recommended_tracking_config("classification")
        >>> any(m.name == "accuracy" for m in config.metrics)
        True

        >>> config = get_recommended_tracking_config("generation")
        >>> any(m.name == "perplexity" for m in config.metrics)
        True

        >>> config = get_recommended_tracking_config("regression")
        >>> any(m.name == "mse" for m in config.metrics)
        True

        >>> get_recommended_tracking_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: task_type must be one of
    """
    valid_tasks = {"classification", "generation", "regression"}
    if task_type not in valid_tasks:
        msg = f"task_type must be one of {valid_tasks}, got '{task_type}'"
        raise ValueError(msg)

    base_metrics = [
        MetricConfig("loss", MetricAggregation.MIN, False, 100),
        MetricConfig("learning_rate", MetricAggregation.LAST, True, 100),
    ]

    if task_type == "classification":
        task_metrics = [
            MetricConfig("accuracy", MetricAggregation.BEST, True, 100),
            MetricConfig("f1", MetricAggregation.BEST, True, 100),
        ]
    elif task_type == "generation":
        task_metrics = [
            MetricConfig("perplexity", MetricAggregation.MIN, False, 100),
            MetricConfig("bleu", MetricAggregation.BEST, True, 500),
        ]
    else:  # regression
        task_metrics = [
            MetricConfig("mse", MetricAggregation.MIN, False, 100),
            MetricConfig("r2", MetricAggregation.BEST, True, 100),
        ]

    return TrackingConfig(
        metrics=tuple(base_metrics + task_metrics),
        artifacts=("model", "checkpoint", "config", "metrics"),
        log_system_metrics=True,
        checkpoint_frequency=1,
    )
