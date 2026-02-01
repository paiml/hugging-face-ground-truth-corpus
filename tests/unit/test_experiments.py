"""Tests for hub.experiments module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hf_gtc.hub.experiments import (
    VALID_AGGREGATIONS,
    VALID_ARTIFACT_TYPES,
    VALID_STATUSES,
    ArtifactConfig,
    ArtifactType,
    ExperimentConfig,
    ExperimentRun,
    ExperimentStats,
    ExperimentStatus,
    MetricAggregation,
    MetricConfig,
    TrackingConfig,
    _aggregate_metric,
    calculate_experiment_stats,
    compare_runs,
    create_artifact_config,
    create_experiment_config,
    create_experiment_run,
    create_metric_config,
    format_experiment_summary,
    get_aggregation,
    get_artifact_type,
    get_best_run,
    get_recommended_tracking_config,
    get_status,
    list_aggregations,
    list_artifact_types,
    list_statuses,
    log_artifact,
    log_metric,
    validate_artifact_config,
    validate_experiment_config,
    validate_experiment_run,
    validate_metric_config,
)


class TestExperimentStatus:
    """Tests for ExperimentStatus enum."""

    def test_all_statuses_have_values(self) -> None:
        """All statuses have string values."""
        for status in ExperimentStatus:
            assert isinstance(status.value, str)

    def test_pending_value(self) -> None:
        """Pending has correct value."""
        assert ExperimentStatus.PENDING.value == "pending"

    def test_running_value(self) -> None:
        """Running has correct value."""
        assert ExperimentStatus.RUNNING.value == "running"

    def test_completed_value(self) -> None:
        """Completed has correct value."""
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        """Failed has correct value."""
        assert ExperimentStatus.FAILED.value == "failed"

    def test_cancelled_value(self) -> None:
        """Cancelled has correct value."""
        assert ExperimentStatus.CANCELLED.value == "cancelled"

    def test_valid_statuses_frozenset(self) -> None:
        """VALID_STATUSES is a frozenset."""
        assert isinstance(VALID_STATUSES, frozenset)

    def test_valid_statuses_contains_all(self) -> None:
        """VALID_STATUSES contains all enum values."""
        for status in ExperimentStatus:
            assert status.value in VALID_STATUSES


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_all_types_have_values(self) -> None:
        """All types have string values."""
        for artifact_type in ArtifactType:
            assert isinstance(artifact_type.value, str)

    def test_model_value(self) -> None:
        """Model has correct value."""
        assert ArtifactType.MODEL.value == "model"

    def test_checkpoint_value(self) -> None:
        """Checkpoint has correct value."""
        assert ArtifactType.CHECKPOINT.value == "checkpoint"

    def test_config_value(self) -> None:
        """Config has correct value."""
        assert ArtifactType.CONFIG.value == "config"

    def test_metrics_value(self) -> None:
        """Metrics has correct value."""
        assert ArtifactType.METRICS.value == "metrics"

    def test_logs_value(self) -> None:
        """Logs has correct value."""
        assert ArtifactType.LOGS.value == "logs"

    def test_valid_artifact_types_frozenset(self) -> None:
        """VALID_ARTIFACT_TYPES is a frozenset."""
        assert isinstance(VALID_ARTIFACT_TYPES, frozenset)


class TestMetricAggregation:
    """Tests for MetricAggregation enum."""

    def test_all_aggregations_have_values(self) -> None:
        """All aggregations have string values."""
        for agg in MetricAggregation:
            assert isinstance(agg.value, str)

    def test_last_value(self) -> None:
        """Last has correct value."""
        assert MetricAggregation.LAST.value == "last"

    def test_best_value(self) -> None:
        """Best has correct value."""
        assert MetricAggregation.BEST.value == "best"

    def test_mean_value(self) -> None:
        """Mean has correct value."""
        assert MetricAggregation.MEAN.value == "mean"

    def test_min_value(self) -> None:
        """Min has correct value."""
        assert MetricAggregation.MIN.value == "min"

    def test_max_value(self) -> None:
        """Max has correct value."""
        assert MetricAggregation.MAX.value == "max"

    def test_valid_aggregations_frozenset(self) -> None:
        """VALID_AGGREGATIONS is a frozenset."""
        assert isinstance(VALID_AGGREGATIONS, frozenset)


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_create_config(self) -> None:
        """Create experiment config."""
        config = ExperimentConfig(
            name="bert-finetuning",
            description="Fine-tune BERT",
            tags=("nlp", "classification"),
            hyperparameters={"lr": 2e-5},
            parent_run_id=None,
        )
        assert config.name == "bert-finetuning"
        assert config.description == "Fine-tune BERT"
        assert config.tags == ("nlp", "classification")
        assert config.hyperparameters["lr"] == 2e-5
        assert config.parent_run_id is None

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ExperimentConfig("test", "", (), {}, None)
        with pytest.raises(AttributeError):
            config.name = "new"  # type: ignore[misc]

    def test_config_with_parent(self) -> None:
        """Config with parent run ID."""
        config = ExperimentConfig("test", "", (), {}, "parent-123")
        assert config.parent_run_id == "parent-123"


class TestMetricConfig:
    """Tests for MetricConfig dataclass."""

    def test_create_config(self) -> None:
        """Create metric config."""
        config = MetricConfig(
            name="accuracy",
            aggregation=MetricAggregation.BEST,
            higher_is_better=True,
            log_frequency=100,
        )
        assert config.name == "accuracy"
        assert config.aggregation == MetricAggregation.BEST
        assert config.higher_is_better is True
        assert config.log_frequency == 100

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = MetricConfig("loss", MetricAggregation.LAST, False, 0)
        with pytest.raises(AttributeError):
            config.name = "new"  # type: ignore[misc]


class TestArtifactConfig:
    """Tests for ArtifactConfig dataclass."""

    def test_create_config(self) -> None:
        """Create artifact config."""
        config = ArtifactConfig(
            artifact_type=ArtifactType.MODEL,
            path="outputs/model.pt",
            metadata={"format": "pytorch"},
            versioned=True,
        )
        assert config.artifact_type == ArtifactType.MODEL
        assert config.path == "outputs/model.pt"
        assert config.metadata["format"] == "pytorch"
        assert config.versioned is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = ArtifactConfig(ArtifactType.MODEL, "model.pt", {}, True)
        with pytest.raises(AttributeError):
            config.path = "new.pt"  # type: ignore[misc]


class TestExperimentRun:
    """Tests for ExperimentRun dataclass."""

    def test_create_run(self) -> None:
        """Create experiment run."""
        config = ExperimentConfig("test", "", (), {}, None)
        now = datetime.now(UTC)
        run = ExperimentRun(
            run_id="run-123",
            config=config,
            status=ExperimentStatus.RUNNING,
            metrics={"loss": [(0, 0.5)]},
            artifacts=(),
            start_time=now,
            end_time=None,
        )
        assert run.run_id == "run-123"
        assert run.config == config
        assert run.status == ExperimentStatus.RUNNING
        assert run.metrics == {"loss": [(0, 0.5)]}
        assert run.artifacts == ()
        assert run.start_time == now
        assert run.end_time is None

    def test_run_is_frozen(self) -> None:
        """Run is immutable."""
        config = ExperimentConfig("test", "", (), {}, None)
        run = ExperimentRun(
            "run-1", config, ExperimentStatus.RUNNING, {}, (), datetime.now(UTC), None
        )
        with pytest.raises(AttributeError):
            run.run_id = "new"  # type: ignore[misc]


class TestValidateExperimentConfig:
    """Tests for validate_experiment_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ExperimentConfig("test", "desc", ("tag1",), {"lr": 1e-4}, None)
        validate_experiment_config(config)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        config = ExperimentConfig("", "desc", (), {}, None)
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_experiment_config(config)

    def test_name_too_long_raises(self) -> None:
        """Name too long raises ValueError."""
        config = ExperimentConfig("x" * 129, "desc", (), {}, None)
        with pytest.raises(ValueError, match="name cannot exceed 128 characters"):
            validate_experiment_config(config)

    def test_description_too_long_raises(self) -> None:
        """Description too long raises ValueError."""
        config = ExperimentConfig("test", "x" * 1025, (), {}, None)
        with pytest.raises(ValueError, match="description cannot exceed 1024"):
            validate_experiment_config(config)

    def test_empty_tag_raises(self) -> None:
        """Empty tag raises ValueError."""
        config = ExperimentConfig("test", "", ("valid", ""), {}, None)
        with pytest.raises(ValueError, match="tags cannot contain empty strings"):
            validate_experiment_config(config)

    def test_tag_too_long_raises(self) -> None:
        """Tag too long raises ValueError."""
        config = ExperimentConfig("test", "", ("x" * 65,), {}, None)
        with pytest.raises(ValueError, match="tag cannot exceed 64 characters"):
            validate_experiment_config(config)


class TestValidateMetricConfig:
    """Tests for validate_metric_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = MetricConfig("accuracy", MetricAggregation.BEST, True, 100)
        validate_metric_config(config)

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        config = MetricConfig("", MetricAggregation.LAST, False, 0)
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_metric_config(config)

    def test_name_too_long_raises(self) -> None:
        """Name too long raises ValueError."""
        config = MetricConfig("x" * 65, MetricAggregation.LAST, False, 0)
        with pytest.raises(ValueError, match="name cannot exceed 64 characters"):
            validate_metric_config(config)

    def test_negative_frequency_raises(self) -> None:
        """Negative log_frequency raises ValueError."""
        config = MetricConfig("test", MetricAggregation.LAST, False, -1)
        with pytest.raises(ValueError, match="log_frequency must be non-negative"):
            validate_metric_config(config)


class TestValidateArtifactConfig:
    """Tests for validate_artifact_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ArtifactConfig(ArtifactType.MODEL, "model.pt", {}, True)
        validate_artifact_config(config)

    def test_empty_path_raises(self) -> None:
        """Empty path raises ValueError."""
        config = ArtifactConfig(ArtifactType.MODEL, "", {}, True)
        with pytest.raises(ValueError, match="path cannot be empty"):
            validate_artifact_config(config)

    def test_path_too_long_raises(self) -> None:
        """Path too long raises ValueError."""
        config = ArtifactConfig(ArtifactType.MODEL, "x" * 513, {}, True)
        with pytest.raises(ValueError, match="path cannot exceed 512 characters"):
            validate_artifact_config(config)


class TestValidateExperimentRun:
    """Tests for validate_experiment_run function."""

    def test_valid_run(self) -> None:
        """Valid run passes validation."""
        config = ExperimentConfig("test", "", (), {}, None)
        run = ExperimentRun(
            "run-1", config, ExperimentStatus.RUNNING, {}, (), datetime.now(UTC), None
        )
        validate_experiment_run(run)

    def test_empty_run_id_raises(self) -> None:
        """Empty run_id raises ValueError."""
        config = ExperimentConfig("test", "", (), {}, None)
        run = ExperimentRun(
            "", config, ExperimentStatus.RUNNING, {}, (), datetime.now(UTC), None
        )
        with pytest.raises(ValueError, match="run_id cannot be empty"):
            validate_experiment_run(run)

    def test_invalid_config_raises(self) -> None:
        """Invalid config raises ValueError."""
        config = ExperimentConfig("", "", (), {}, None)
        run = ExperimentRun(
            "run-1", config, ExperimentStatus.RUNNING, {}, (), datetime.now(UTC), None
        )
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_experiment_run(run)

    def test_end_before_start_raises(self) -> None:
        """End time before start time raises ValueError."""
        config = ExperimentConfig("test", "", (), {}, None)
        now = datetime.now(UTC)
        run = ExperimentRun(
            "run-1",
            config,
            ExperimentStatus.COMPLETED,
            {},
            (),
            now,
            now - timedelta(hours=1),
        )
        with pytest.raises(ValueError, match="end_time cannot be before start_time"):
            validate_experiment_run(run)


class TestCreateExperimentConfig:
    """Tests for create_experiment_config function."""

    def test_minimal_config(self) -> None:
        """Create minimal config."""
        config = create_experiment_config("test")
        assert config.name == "test"
        assert config.description == ""
        assert config.tags == ()
        assert config.hyperparameters == {}
        assert config.parent_run_id is None

    def test_full_config(self) -> None:
        """Create full config."""
        config = create_experiment_config(
            "bert",
            description="Fine-tune BERT",
            tags=["nlp", "bert"],
            hyperparameters={"lr": 1e-4},
            parent_run_id="parent-1",
        )
        assert config.name == "bert"
        assert config.description == "Fine-tune BERT"
        assert config.tags == ("nlp", "bert")
        assert config.hyperparameters["lr"] == 1e-4
        assert config.parent_run_id == "parent-1"

    def test_tags_as_tuple(self) -> None:
        """Tags can be passed as tuple."""
        config = create_experiment_config("test", tags=("a", "b"))
        assert config.tags == ("a", "b")

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_experiment_config("")


class TestCreateMetricConfig:
    """Tests for create_metric_config function."""

    def test_minimal_config(self) -> None:
        """Create minimal config."""
        config = create_metric_config("accuracy")
        assert config.name == "accuracy"
        assert config.aggregation == MetricAggregation.LAST
        assert config.higher_is_better is True
        assert config.log_frequency == 0

    def test_full_config(self) -> None:
        """Create full config."""
        config = create_metric_config(
            "loss",
            aggregation="min",
            higher_is_better=False,
            log_frequency=100,
        )
        assert config.name == "loss"
        assert config.aggregation == MetricAggregation.MIN
        assert config.higher_is_better is False
        assert config.log_frequency == 100

    def test_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_metric_config("")

    def test_invalid_aggregation_raises(self) -> None:
        """Invalid aggregation raises ValueError."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            create_metric_config("x", aggregation="invalid")


class TestCreateArtifactConfig:
    """Tests for create_artifact_config function."""

    def test_minimal_config(self) -> None:
        """Create minimal config."""
        config = create_artifact_config("model", "model.pt")
        assert config.artifact_type == ArtifactType.MODEL
        assert config.path == "model.pt"
        assert config.metadata == {}
        assert config.versioned is True

    def test_full_config(self) -> None:
        """Create full config."""
        config = create_artifact_config(
            "checkpoint",
            "ckpt/step-1000.pt",
            metadata={"step": 1000},
            versioned=False,
        )
        assert config.artifact_type == ArtifactType.CHECKPOINT
        assert config.path == "ckpt/step-1000.pt"
        assert config.metadata["step"] == 1000
        assert config.versioned is False

    def test_empty_path_raises(self) -> None:
        """Empty path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be empty"):
            create_artifact_config("model", "")

    def test_invalid_type_raises(self) -> None:
        """Invalid artifact type raises ValueError."""
        with pytest.raises(ValueError, match="artifact_type must be one of"):
            create_artifact_config("invalid", "path")


class TestCreateExperimentRun:
    """Tests for create_experiment_run function."""

    def test_auto_run_id(self) -> None:
        """Run ID is auto-generated."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        assert run.run_id is not None
        assert len(run.run_id) > 0

    def test_custom_run_id(self) -> None:
        """Custom run ID is used."""
        config = create_experiment_config("test")
        run = create_experiment_run(config, run_id="custom-123")
        assert run.run_id == "custom-123"

    def test_initial_status_pending(self) -> None:
        """Initial status is PENDING."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        assert run.status == ExperimentStatus.PENDING

    def test_initial_metrics_empty(self) -> None:
        """Initial metrics are empty."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        assert run.metrics == {}

    def test_initial_artifacts_empty(self) -> None:
        """Initial artifacts are empty."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        assert run.artifacts == ()

    def test_start_time_set(self) -> None:
        """Start time is set."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        assert run.start_time is not None
        assert run.end_time is None


class TestLogMetric:
    """Tests for log_metric function."""

    def test_log_first_metric(self) -> None:
        """Log first metric value."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run = log_metric(run, "loss", 0.5, step=0)
        assert "loss" in run.metrics
        assert run.metrics["loss"] == [(0, 0.5)]

    def test_log_multiple_values(self) -> None:
        """Log multiple values for same metric."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run = log_metric(run, "loss", 0.5, step=0)
        run = log_metric(run, "loss", 0.3, step=100)
        run = log_metric(run, "loss", 0.2, step=200)
        assert len(run.metrics["loss"]) == 3
        assert run.metrics["loss"][-1] == (200, 0.2)

    def test_log_multiple_metrics(self) -> None:
        """Log multiple different metrics."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run = log_metric(run, "loss", 0.5)
        run = log_metric(run, "accuracy", 0.9)
        assert "loss" in run.metrics
        assert "accuracy" in run.metrics

    def test_empty_name_raises(self) -> None:
        """Empty metric name raises ValueError."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        with pytest.raises(ValueError, match="metric_name cannot be empty"):
            log_metric(run, "", 0.5)

    def test_negative_step_raises(self) -> None:
        """Negative step raises ValueError."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        with pytest.raises(ValueError, match="step must be non-negative"):
            log_metric(run, "loss", 0.5, step=-1)

    def test_run_immutability(self) -> None:
        """Original run is not modified."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run2 = log_metric(run, "loss", 0.5)
        assert run.metrics == {}
        assert "loss" in run2.metrics


class TestLogArtifact:
    """Tests for log_artifact function."""

    def test_log_artifact(self) -> None:
        """Log an artifact."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        artifact = create_artifact_config("model", "model.pt")
        run = log_artifact(run, artifact)
        assert len(run.artifacts) == 1
        assert run.artifacts[0].path == "model.pt"

    def test_log_multiple_artifacts(self) -> None:
        """Log multiple artifacts."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run = log_artifact(run, create_artifact_config("model", "model.pt"))
        run = log_artifact(run, create_artifact_config("checkpoint", "ckpt.pt"))
        assert len(run.artifacts) == 2

    def test_run_immutability(self) -> None:
        """Original run is not modified."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        artifact = create_artifact_config("model", "model.pt")
        run2 = log_artifact(run, artifact)
        assert run.artifacts == ()
        assert len(run2.artifacts) == 1


class TestListStatuses:
    """Tests for list_statuses function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        statuses = list_statuses()
        assert statuses == sorted(statuses)

    def test_contains_running(self) -> None:
        """Contains running."""
        statuses = list_statuses()
        assert "running" in statuses

    def test_contains_completed(self) -> None:
        """Contains completed."""
        statuses = list_statuses()
        assert "completed" in statuses


class TestListArtifactTypes:
    """Tests for list_artifact_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_artifact_types()
        assert types == sorted(types)

    def test_contains_model(self) -> None:
        """Contains model."""
        types = list_artifact_types()
        assert "model" in types

    def test_contains_checkpoint(self) -> None:
        """Contains checkpoint."""
        types = list_artifact_types()
        assert "checkpoint" in types


class TestListAggregations:
    """Tests for list_aggregations function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        aggs = list_aggregations()
        assert aggs == sorted(aggs)

    def test_contains_last(self) -> None:
        """Contains last."""
        aggs = list_aggregations()
        assert "last" in aggs

    def test_contains_best(self) -> None:
        """Contains best."""
        aggs = list_aggregations()
        assert "best" in aggs


class TestGetStatus:
    """Tests for get_status function."""

    def test_get_running(self) -> None:
        """Get running status."""
        assert get_status("running") == ExperimentStatus.RUNNING

    def test_get_completed(self) -> None:
        """Get completed status."""
        assert get_status("completed") == ExperimentStatus.COMPLETED

    def test_get_failed(self) -> None:
        """Get failed status."""
        assert get_status("failed") == ExperimentStatus.FAILED

    def test_invalid_status_raises(self) -> None:
        """Invalid status raises ValueError."""
        with pytest.raises(ValueError, match="status must be one of"):
            get_status("invalid")


class TestGetArtifactType:
    """Tests for get_artifact_type function."""

    def test_get_model(self) -> None:
        """Get model type."""
        assert get_artifact_type("model") == ArtifactType.MODEL

    def test_get_checkpoint(self) -> None:
        """Get checkpoint type."""
        assert get_artifact_type("checkpoint") == ArtifactType.CHECKPOINT

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="artifact_type must be one of"):
            get_artifact_type("invalid")


class TestGetAggregation:
    """Tests for get_aggregation function."""

    def test_get_last(self) -> None:
        """Get last aggregation."""
        assert get_aggregation("last") == MetricAggregation.LAST

    def test_get_best(self) -> None:
        """Get best aggregation."""
        assert get_aggregation("best") == MetricAggregation.BEST

    def test_get_mean(self) -> None:
        """Get mean aggregation."""
        assert get_aggregation("mean") == MetricAggregation.MEAN

    def test_invalid_aggregation_raises(self) -> None:
        """Invalid aggregation raises ValueError."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            get_aggregation("invalid")


class TestAggregateMetric:
    """Tests for _aggregate_metric function."""

    def test_last_aggregation(self) -> None:
        """Last aggregation returns last value by step."""
        values = [(0, 1.0), (2, 3.0), (1, 2.0)]
        result = _aggregate_metric(values, MetricAggregation.LAST, True)
        assert result == pytest.approx(3.0)

    def test_mean_aggregation(self) -> None:
        """Mean aggregation returns average."""
        values = [(0, 1.0), (1, 2.0), (2, 3.0)]
        result = _aggregate_metric(values, MetricAggregation.MEAN, True)
        assert result == pytest.approx(2.0)

    def test_min_aggregation(self) -> None:
        """Min aggregation returns minimum."""
        values = [(0, 3.0), (1, 1.0), (2, 2.0)]
        result = _aggregate_metric(values, MetricAggregation.MIN, True)
        assert result == pytest.approx(1.0)

    def test_max_aggregation(self) -> None:
        """Max aggregation returns maximum."""
        values = [(0, 1.0), (1, 3.0), (2, 2.0)]
        result = _aggregate_metric(values, MetricAggregation.MAX, True)
        assert result == pytest.approx(3.0)

    def test_best_higher_is_better(self) -> None:
        """Best with higher_is_better=True returns max."""
        values = [(0, 1.0), (1, 3.0), (2, 2.0)]
        result = _aggregate_metric(values, MetricAggregation.BEST, True)
        assert result == pytest.approx(3.0)

    def test_best_lower_is_better(self) -> None:
        """Best with higher_is_better=False returns min."""
        values = [(0, 1.0), (1, 3.0), (2, 2.0)]
        result = _aggregate_metric(values, MetricAggregation.BEST, False)
        assert result == pytest.approx(1.0)

    def test_empty_values_raises(self) -> None:
        """Empty values raises ValueError."""
        with pytest.raises(ValueError, match="values list cannot be empty"):
            _aggregate_metric([], MetricAggregation.LAST, True)


class TestGetBestRun:
    """Tests for get_best_run function."""

    def test_best_run_higher_is_better(self) -> None:
        """Get best run when higher is better."""
        config = create_experiment_config("test")
        run1 = create_experiment_run(config, run_id="run-1")
        run1 = log_metric(run1, "accuracy", 0.8)
        run2 = create_experiment_run(config, run_id="run-2")
        run2 = log_metric(run2, "accuracy", 0.9)
        best = get_best_run([run1, run2], "accuracy", higher_is_better=True)
        assert best is not None
        assert best.run_id == "run-2"

    def test_best_run_lower_is_better(self) -> None:
        """Get best run when lower is better."""
        config = create_experiment_config("test")
        run1 = create_experiment_run(config, run_id="run-1")
        run1 = log_metric(run1, "loss", 0.2)
        run2 = create_experiment_run(config, run_id="run-2")
        run2 = log_metric(run2, "loss", 0.5)
        best = get_best_run([run1, run2], "loss", higher_is_better=False)
        assert best is not None
        assert best.run_id == "run-1"

    def test_empty_runs_returns_none(self) -> None:
        """Empty runs list returns None."""
        assert get_best_run([], "accuracy") is None

    def test_no_metric_returns_none(self) -> None:
        """Runs without metric returns None."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        assert get_best_run([run], "nonexistent") is None

    def test_invalid_aggregation_raises(self) -> None:
        """Invalid aggregation raises ValueError."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            get_best_run([], "accuracy", aggregation="invalid")


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_compare_runs(self) -> None:
        """Compare multiple runs."""
        config = create_experiment_config("test")
        run1 = create_experiment_run(config, run_id="run-1")
        run1 = log_metric(run1, "accuracy", 0.8)
        run1 = log_metric(run1, "loss", 0.2)
        run2 = create_experiment_run(config, run_id="run-2")
        run2 = log_metric(run2, "accuracy", 0.9)
        comparison = compare_runs([run1, run2])
        assert comparison["run-1"]["accuracy"] == pytest.approx(0.8)
        assert comparison["run-1"]["loss"] == pytest.approx(0.2)
        assert comparison["run-2"]["accuracy"] == pytest.approx(0.9)
        assert comparison["run-2"]["loss"] is None

    def test_compare_specific_metrics(self) -> None:
        """Compare specific metrics only."""
        config = create_experiment_config("test")
        run1 = create_experiment_run(config, run_id="run-1")
        run1 = log_metric(run1, "accuracy", 0.8)
        run1 = log_metric(run1, "loss", 0.2)
        comparison = compare_runs([run1], metric_names=["accuracy"])
        assert "accuracy" in comparison["run-1"]
        assert "loss" not in comparison["run-1"]

    def test_empty_runs(self) -> None:
        """Empty runs returns empty dict."""
        assert compare_runs([]) == {}

    def test_last_value_used(self) -> None:
        """Last value by step is used."""
        config = create_experiment_config("test")
        run = create_experiment_run(config, run_id="run-1")
        run = log_metric(run, "accuracy", 0.5, step=0)
        run = log_metric(run, "accuracy", 0.9, step=100)
        comparison = compare_runs([run])
        assert comparison["run-1"]["accuracy"] == pytest.approx(0.9)


class TestExperimentStats:
    """Tests for ExperimentStats dataclass."""

    def test_create_stats(self) -> None:
        """Create experiment stats."""
        stats = ExperimentStats(
            total_runs=10,
            completed_runs=8,
            failed_runs=2,
            avg_duration_seconds=3600.0,
            metric_stats={"accuracy": (0.7, 0.95, 0.85)},
        )
        assert stats.total_runs == 10
        assert stats.completed_runs == 8
        assert stats.failed_runs == 2
        assert stats.avg_duration_seconds == pytest.approx(3600.0)
        assert stats.metric_stats["accuracy"] == (0.7, 0.95, 0.85)


class TestCalculateExperimentStats:
    """Tests for calculate_experiment_stats function."""

    def test_basic_stats(self) -> None:
        """Calculate basic stats."""
        config = create_experiment_config("test")
        run1 = create_experiment_run(config, run_id="run-1")
        run1 = log_metric(run1, "accuracy", 0.8)
        run2 = create_experiment_run(config, run_id="run-2")
        run2 = log_metric(run2, "accuracy", 0.9)
        stats = calculate_experiment_stats([run1, run2])
        assert stats.total_runs == 2
        min_val, max_val, mean_val = stats.metric_stats["accuracy"]
        assert min_val == pytest.approx(0.8)
        assert max_val == pytest.approx(0.9)
        assert abs(mean_val - 0.85) < 1e-10

    def test_empty_runs(self) -> None:
        """Empty runs returns zero stats."""
        stats = calculate_experiment_stats([])
        assert stats.total_runs == 0
        assert stats.completed_runs == 0
        assert stats.failed_runs == 0
        assert stats.avg_duration_seconds is None
        assert stats.metric_stats == {}

    def test_counts_statuses(self) -> None:
        """Counts completed and failed runs."""
        config = create_experiment_config("test")
        now = datetime.now(UTC)
        run1 = ExperimentRun(
            "run-1", config, ExperimentStatus.COMPLETED, {}, (), now, now
        )
        run2 = ExperimentRun("run-2", config, ExperimentStatus.FAILED, {}, (), now, now)
        run3 = ExperimentRun(
            "run-3", config, ExperimentStatus.RUNNING, {}, (), now, None
        )
        stats = calculate_experiment_stats([run1, run2, run3])
        assert stats.completed_runs == 1
        assert stats.failed_runs == 1

    def test_calculates_duration(self) -> None:
        """Calculates average duration."""
        config = create_experiment_config("test")
        now = datetime.now(UTC)
        run = ExperimentRun(
            "run-1",
            config,
            ExperimentStatus.COMPLETED,
            {},
            (),
            now,
            now + timedelta(hours=1),
        )
        stats = calculate_experiment_stats([run])
        assert stats.avg_duration_seconds == pytest.approx(3600.0)


class TestFormatExperimentSummary:
    """Tests for format_experiment_summary function."""

    def test_basic_summary(self) -> None:
        """Format basic summary."""
        config = create_experiment_config("bert-finetuning")
        run = create_experiment_run(config, run_id="run-123")
        summary = format_experiment_summary(run)
        assert "bert-finetuning" in summary
        assert "run-123" in summary
        assert "pending" in summary

    def test_summary_with_description(self) -> None:
        """Summary includes description."""
        config = create_experiment_config("test", description="Test description")
        run = create_experiment_run(config)
        summary = format_experiment_summary(run)
        assert "Test description" in summary

    def test_summary_with_tags(self) -> None:
        """Summary includes tags."""
        config = create_experiment_config("test", tags=["nlp", "bert"])
        run = create_experiment_run(config)
        summary = format_experiment_summary(run)
        assert "nlp" in summary
        assert "bert" in summary

    def test_summary_with_hyperparameters(self) -> None:
        """Summary includes hyperparameters."""
        config = create_experiment_config(
            "test", hyperparameters={"lr": 1e-4, "epochs": 3}
        )
        run = create_experiment_run(config)
        summary = format_experiment_summary(run)
        assert "lr" in summary
        assert "epochs" in summary

    def test_summary_with_metrics(self) -> None:
        """Summary includes metrics."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run = log_metric(run, "accuracy", 0.95)
        summary = format_experiment_summary(run)
        assert "accuracy" in summary
        assert "0.95" in summary

    def test_summary_with_artifacts(self) -> None:
        """Summary includes artifacts."""
        config = create_experiment_config("test")
        run = create_experiment_run(config)
        run = log_artifact(run, create_artifact_config("model", "model.pt"))
        summary = format_experiment_summary(run)
        assert "model.pt" in summary
        assert "[model]" in summary


class TestTrackingConfig:
    """Tests for TrackingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create tracking config."""
        metric = MetricConfig("loss", MetricAggregation.MIN, False, 100)
        config = TrackingConfig(
            metrics=(metric,),
            artifacts=("model", "checkpoint"),
            log_system_metrics=True,
            checkpoint_frequency=1,
        )
        assert len(config.metrics) == 1
        assert config.artifacts == ("model", "checkpoint")
        assert config.log_system_metrics is True
        assert config.checkpoint_frequency == 1


class TestGetRecommendedTrackingConfig:
    """Tests for get_recommended_tracking_config function."""

    def test_classification_config(self) -> None:
        """Get classification config."""
        config = get_recommended_tracking_config("classification")
        metric_names = [m.name for m in config.metrics]
        assert "accuracy" in metric_names
        assert "f1" in metric_names
        assert "loss" in metric_names

    def test_generation_config(self) -> None:
        """Get generation config."""
        config = get_recommended_tracking_config("generation")
        metric_names = [m.name for m in config.metrics]
        assert "perplexity" in metric_names
        assert "bleu" in metric_names

    def test_regression_config(self) -> None:
        """Get regression config."""
        config = get_recommended_tracking_config("regression")
        metric_names = [m.name for m in config.metrics]
        assert "mse" in metric_names
        assert "r2" in metric_names

    def test_invalid_task_raises(self) -> None:
        """Invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_tracking_config("invalid")

    def test_has_artifacts(self) -> None:
        """Config has artifacts."""
        config = get_recommended_tracking_config()
        assert "model" in config.artifacts
        assert "checkpoint" in config.artifacts

    def test_log_system_metrics_enabled(self) -> None:
        """System metrics logging is enabled."""
        config = get_recommended_tracking_config()
        assert config.log_system_metrics is True
