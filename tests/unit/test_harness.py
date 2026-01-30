"""Tests for evaluation harness functionality."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.harness import (
    VALID_AGGREGATION_METHODS,
    VALID_OUTPUT_FORMATS,
    VALID_TASK_TYPES,
    AggregationMethod,
    EvaluationResult,
    HarnessConfig,
    HarnessStats,
    OutputFormat,
    TaskConfig,
    TaskType,
    aggregate_results,
    create_evaluation_result,
    create_harness_config,
    create_task_config,
    estimate_evaluation_time,
    format_harness_stats,
    format_results_table,
    get_aggregation_method,
    get_output_format,
    get_recommended_harness_config,
    get_task_type,
    list_aggregation_methods,
    list_output_formats,
    list_task_types,
    run_evaluation_task,
    validate_aggregation_method,
    validate_evaluation_result,
    validate_harness_config,
    validate_harness_stats,
    validate_output_format,
    validate_task_config,
    validate_task_type,
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_multiple_choice_value(self) -> None:
        """Test MULTIPLE_CHOICE value."""
        assert TaskType.MULTIPLE_CHOICE.value == "multiple_choice"

    def test_generation_value(self) -> None:
        """Test GENERATION value."""
        assert TaskType.GENERATION.value == "generation"

    def test_classification_value(self) -> None:
        """Test CLASSIFICATION value."""
        assert TaskType.CLASSIFICATION.value == "classification"

    def test_extraction_value(self) -> None:
        """Test EXTRACTION value."""
        assert TaskType.EXTRACTION.value == "extraction"

    def test_ranking_value(self) -> None:
        """Test RANKING value."""
        assert TaskType.RANKING.value == "ranking"

    def test_valid_task_types_frozenset(self) -> None:
        """Test VALID_TASK_TYPES frozenset."""
        assert isinstance(VALID_TASK_TYPES, frozenset)
        assert "multiple_choice" in VALID_TASK_TYPES
        assert "generation" in VALID_TASK_TYPES
        assert len(VALID_TASK_TYPES) == 5


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_json_value(self) -> None:
        """Test JSON value."""
        assert OutputFormat.JSON.value == "json"

    def test_csv_value(self) -> None:
        """Test CSV value."""
        assert OutputFormat.CSV.value == "csv"

    def test_markdown_value(self) -> None:
        """Test MARKDOWN value."""
        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_latex_value(self) -> None:
        """Test LATEX value."""
        assert OutputFormat.LATEX.value == "latex"

    def test_valid_output_formats_frozenset(self) -> None:
        """Test VALID_OUTPUT_FORMATS frozenset."""
        assert isinstance(VALID_OUTPUT_FORMATS, frozenset)
        assert "json" in VALID_OUTPUT_FORMATS
        assert "csv" in VALID_OUTPUT_FORMATS
        assert len(VALID_OUTPUT_FORMATS) == 4


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_mean_value(self) -> None:
        """Test MEAN value."""
        assert AggregationMethod.MEAN.value == "mean"

    def test_weighted_mean_value(self) -> None:
        """Test WEIGHTED_MEAN value."""
        assert AggregationMethod.WEIGHTED_MEAN.value == "weighted_mean"

    def test_median_value(self) -> None:
        """Test MEDIAN value."""
        assert AggregationMethod.MEDIAN.value == "median"

    def test_max_value(self) -> None:
        """Test MAX value."""
        assert AggregationMethod.MAX.value == "max"

    def test_valid_aggregation_methods_frozenset(self) -> None:
        """Test VALID_AGGREGATION_METHODS frozenset."""
        assert isinstance(VALID_AGGREGATION_METHODS, frozenset)
        assert "mean" in VALID_AGGREGATION_METHODS
        assert "weighted_mean" in VALID_AGGREGATION_METHODS
        assert len(VALID_AGGREGATION_METHODS) == 4


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = TaskConfig(task_type=TaskType.MULTIPLE_CHOICE)
        assert config.task_type == TaskType.MULTIPLE_CHOICE
        assert config.num_fewshot == 0
        assert config.batch_size == 32
        assert config.limit is None

    def test_creation_with_all_values(self) -> None:
        """Test creating config with all values."""
        config = TaskConfig(
            task_type=TaskType.GENERATION,
            num_fewshot=5,
            batch_size=16,
            limit=100,
        )
        assert config.task_type == TaskType.GENERATION
        assert config.num_fewshot == 5
        assert config.batch_size == 16
        assert config.limit == 100

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = TaskConfig(task_type=TaskType.MULTIPLE_CHOICE)
        with pytest.raises(AttributeError):
            config.num_fewshot = 5  # type: ignore[misc]


class TestHarnessConfig:
    """Tests for HarnessConfig dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = HarnessConfig(tasks=("mmlu", "hellaswag"))
        assert config.tasks == ("mmlu", "hellaswag")
        assert config.output_format == OutputFormat.JSON
        assert config.aggregation == AggregationMethod.MEAN
        assert config.log_samples is False
        assert config.cache_requests is True

    def test_creation_with_all_values(self) -> None:
        """Test creating config with all values."""
        config = HarnessConfig(
            tasks=("arc", "winogrande"),
            output_format=OutputFormat.MARKDOWN,
            aggregation=AggregationMethod.WEIGHTED_MEAN,
            log_samples=True,
            cache_requests=False,
        )
        assert config.tasks == ("arc", "winogrande")
        assert config.output_format == OutputFormat.MARKDOWN
        assert config.aggregation == AggregationMethod.WEIGHTED_MEAN
        assert config.log_samples is True
        assert config.cache_requests is False

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = HarnessConfig(tasks=("mmlu",))
        with pytest.raises(AttributeError):
            config.log_samples = True  # type: ignore[misc]


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating result."""
        result = EvaluationResult(
            task_name="mmlu",
            metrics={"accuracy": 0.65, "f1": 0.63},
            num_samples=1000,
            duration_seconds=120.5,
        )
        assert result.task_name == "mmlu"
        assert result.metrics["accuracy"] == pytest.approx(0.65)
        assert result.metrics["f1"] == pytest.approx(0.63)
        assert result.num_samples == 1000
        assert result.duration_seconds == pytest.approx(120.5)

    def test_frozen(self) -> None:
        """Test that result is immutable."""
        result = EvaluationResult(
            task_name="mmlu",
            metrics={},
            num_samples=100,
            duration_seconds=10.0,
        )
        with pytest.raises(AttributeError):
            result.task_name = "other"  # type: ignore[misc]


class TestHarnessStats:
    """Tests for HarnessStats dataclass."""

    def test_creation(self) -> None:
        """Test creating stats."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            total_samples=5000,
            avg_score=0.72,
        )
        assert stats.total_tasks == 5
        assert stats.completed_tasks == 4
        assert stats.failed_tasks == 1
        assert stats.total_samples == 5000
        assert stats.avg_score == pytest.approx(0.72)

    def test_frozen(self) -> None:
        """Test that stats is immutable."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            total_samples=5000,
            avg_score=0.72,
        )
        with pytest.raises(AttributeError):
            stats.total_tasks = 10  # type: ignore[misc]


class TestValidateTaskConfig:
    """Tests for validate_task_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = TaskConfig(task_type=TaskType.MULTIPLE_CHOICE)
        validate_task_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_task_config(None)  # type: ignore[arg-type]

    def test_negative_num_fewshot_raises_error(self) -> None:
        """Test that negative num_fewshot raises ValueError."""
        config = TaskConfig(
            task_type=TaskType.GENERATION,
            num_fewshot=-1,
        )
        with pytest.raises(ValueError, match="num_fewshot cannot be negative"):
            validate_task_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = TaskConfig(
            task_type=TaskType.CLASSIFICATION,
            batch_size=0,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_task_config(config)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch_size raises ValueError."""
        config = TaskConfig(
            task_type=TaskType.EXTRACTION,
            batch_size=-1,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_task_config(config)

    def test_zero_limit_raises_error(self) -> None:
        """Test that zero limit raises ValueError."""
        config = TaskConfig(
            task_type=TaskType.RANKING,
            limit=0,
        )
        with pytest.raises(ValueError, match="limit must be positive"):
            validate_task_config(config)

    def test_negative_limit_raises_error(self) -> None:
        """Test that negative limit raises ValueError."""
        config = TaskConfig(
            task_type=TaskType.MULTIPLE_CHOICE,
            limit=-1,
        )
        with pytest.raises(ValueError, match="limit must be positive"):
            validate_task_config(config)


class TestValidateHarnessConfig:
    """Tests for validate_harness_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = HarnessConfig(tasks=("mmlu",))
        validate_harness_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_harness_config(None)  # type: ignore[arg-type]

    def test_empty_tasks_raises_error(self) -> None:
        """Test that empty tasks raises ValueError."""
        config = HarnessConfig(tasks=())
        with pytest.raises(ValueError, match="tasks cannot be empty"):
            validate_harness_config(config)

    def test_empty_task_name_raises_error(self) -> None:
        """Test that empty task name raises ValueError."""
        config = HarnessConfig(tasks=("mmlu", ""))
        with pytest.raises(ValueError, match="tasks cannot contain empty strings"):
            validate_harness_config(config)


class TestValidateEvaluationResult:
    """Tests for validate_evaluation_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = EvaluationResult(
            task_name="mmlu",
            metrics={"accuracy": 0.65},
            num_samples=100,
            duration_seconds=10.0,
        )
        validate_evaluation_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_evaluation_result(None)  # type: ignore[arg-type]

    def test_empty_task_name_raises_error(self) -> None:
        """Test that empty task_name raises ValueError."""
        result = EvaluationResult(
            task_name="",
            metrics={},
            num_samples=100,
            duration_seconds=10.0,
        )
        with pytest.raises(ValueError, match="task_name cannot be empty"):
            validate_evaluation_result(result)

    def test_zero_num_samples_raises_error(self) -> None:
        """Test that zero num_samples raises ValueError."""
        result = EvaluationResult(
            task_name="mmlu",
            metrics={},
            num_samples=0,
            duration_seconds=10.0,
        )
        with pytest.raises(ValueError, match="num_samples must be positive"):
            validate_evaluation_result(result)

    def test_negative_duration_raises_error(self) -> None:
        """Test that negative duration_seconds raises ValueError."""
        result = EvaluationResult(
            task_name="mmlu",
            metrics={},
            num_samples=100,
            duration_seconds=-1.0,
        )
        with pytest.raises(ValueError, match="duration_seconds cannot be negative"):
            validate_evaluation_result(result)


class TestValidateHarnessStats:
    """Tests for validate_harness_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            total_samples=5000,
            avg_score=0.72,
        )
        validate_harness_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_harness_stats(None)  # type: ignore[arg-type]

    def test_zero_total_tasks_raises_error(self) -> None:
        """Test that zero total_tasks raises ValueError."""
        stats = HarnessStats(
            total_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            total_samples=0,
            avg_score=0.0,
        )
        with pytest.raises(ValueError, match="total_tasks must be positive"):
            validate_harness_stats(stats)

    def test_negative_completed_tasks_raises_error(self) -> None:
        """Test that negative completed_tasks raises ValueError."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=-1,
            failed_tasks=0,
            total_samples=0,
            avg_score=0.0,
        )
        with pytest.raises(ValueError, match="completed_tasks cannot be negative"):
            validate_harness_stats(stats)

    def test_negative_failed_tasks_raises_error(self) -> None:
        """Test that negative failed_tasks raises ValueError."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=0,
            failed_tasks=-1,
            total_samples=0,
            avg_score=0.0,
        )
        with pytest.raises(ValueError, match="failed_tasks cannot be negative"):
            validate_harness_stats(stats)

    def test_tasks_exceeding_total_raises_error(self) -> None:
        """Test that completed + failed > total raises ValueError."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=3,
            total_samples=0,
            avg_score=0.0,
        )
        with pytest.raises(ValueError, match="cannot exceed total_tasks"):
            validate_harness_stats(stats)

    def test_negative_total_samples_raises_error(self) -> None:
        """Test that negative total_samples raises ValueError."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            total_samples=-1,
            avg_score=0.0,
        )
        with pytest.raises(ValueError, match="total_samples cannot be negative"):
            validate_harness_stats(stats)


class TestCreateTaskConfig:
    """Tests for create_task_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_task_config(TaskType.MULTIPLE_CHOICE)
        assert config.task_type == TaskType.MULTIPLE_CHOICE
        assert config.num_fewshot == 0
        assert config.batch_size == 32
        assert config.limit is None

    def test_create_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_task_config(
            TaskType.GENERATION,
            num_fewshot=5,
            batch_size=16,
            limit=100,
        )
        assert config.task_type == TaskType.GENERATION
        assert config.num_fewshot == 5
        assert config.batch_size == 16
        assert config.limit == 100

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        with pytest.raises(ValueError, match="num_fewshot cannot be negative"):
            create_task_config(TaskType.CLASSIFICATION, num_fewshot=-1)


class TestCreateHarnessConfig:
    """Tests for create_harness_config function."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_harness_config(("mmlu", "hellaswag"))
        assert config.tasks == ("mmlu", "hellaswag")
        assert config.output_format == OutputFormat.JSON
        assert config.aggregation == AggregationMethod.MEAN

    def test_create_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_harness_config(
            ("arc",),
            output_format=OutputFormat.MARKDOWN,
            aggregation=AggregationMethod.WEIGHTED_MEAN,
        )
        assert config.tasks == ("arc",)
        assert config.output_format == OutputFormat.MARKDOWN
        assert config.aggregation == AggregationMethod.WEIGHTED_MEAN

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        with pytest.raises(ValueError, match="tasks cannot be empty"):
            create_harness_config(())


class TestCreateEvaluationResult:
    """Tests for create_evaluation_result function."""

    def test_create_result(self) -> None:
        """Test creating result."""
        result = create_evaluation_result(
            "mmlu",
            {"accuracy": 0.65},
            1000,
            120.5,
        )
        assert result.task_name == "mmlu"
        assert result.metrics["accuracy"] == pytest.approx(0.65)
        assert result.num_samples == 1000
        assert result.duration_seconds == pytest.approx(120.5)

    def test_invalid_result_raises_error(self) -> None:
        """Test that invalid result raises ValueError."""
        with pytest.raises(ValueError, match="task_name cannot be empty"):
            create_evaluation_result("", {}, 100, 10.0)


class TestListTaskTypes:
    """Tests for list_task_types function."""

    def test_returns_sorted_list(self) -> None:
        """Test that returns sorted list."""
        types = list_task_types()
        assert types == sorted(types)

    def test_contains_all_types(self) -> None:
        """Test that contains all types."""
        types = list_task_types()
        assert "multiple_choice" in types
        assert "generation" in types
        assert "classification" in types
        assert "extraction" in types
        assert "ranking" in types


class TestGetTaskType:
    """Tests for get_task_type function."""

    def test_get_multiple_choice(self) -> None:
        """Test getting MULTIPLE_CHOICE."""
        assert get_task_type("multiple_choice") == TaskType.MULTIPLE_CHOICE

    def test_get_generation(self) -> None:
        """Test getting GENERATION."""
        assert get_task_type("generation") == TaskType.GENERATION

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid task type"):
            get_task_type("invalid")


class TestValidateTaskType:
    """Tests for validate_task_type function."""

    def test_valid_types(self) -> None:
        """Test valid task types."""
        assert validate_task_type("multiple_choice") is True
        assert validate_task_type("generation") is True
        assert validate_task_type("classification") is True

    def test_invalid_types(self) -> None:
        """Test invalid task types."""
        assert validate_task_type("invalid") is False
        assert validate_task_type("") is False


class TestListOutputFormats:
    """Tests for list_output_formats function."""

    def test_returns_sorted_list(self) -> None:
        """Test that returns sorted list."""
        formats = list_output_formats()
        assert formats == sorted(formats)

    def test_contains_all_formats(self) -> None:
        """Test that contains all formats."""
        formats = list_output_formats()
        assert "json" in formats
        assert "csv" in formats
        assert "markdown" in formats
        assert "latex" in formats


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    def test_get_json(self) -> None:
        """Test getting JSON."""
        assert get_output_format("json") == OutputFormat.JSON

    def test_get_markdown(self) -> None:
        """Test getting MARKDOWN."""
        assert get_output_format("markdown") == OutputFormat.MARKDOWN

    def test_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="invalid output format"):
            get_output_format("invalid")


class TestValidateOutputFormat:
    """Tests for validate_output_format function."""

    def test_valid_formats(self) -> None:
        """Test valid output formats."""
        assert validate_output_format("json") is True
        assert validate_output_format("csv") is True
        assert validate_output_format("markdown") is True

    def test_invalid_formats(self) -> None:
        """Test invalid output formats."""
        assert validate_output_format("invalid") is False
        assert validate_output_format("") is False


class TestListAggregationMethods:
    """Tests for list_aggregation_methods function."""

    def test_returns_sorted_list(self) -> None:
        """Test that returns sorted list."""
        methods = list_aggregation_methods()
        assert methods == sorted(methods)

    def test_contains_all_methods(self) -> None:
        """Test that contains all methods."""
        methods = list_aggregation_methods()
        assert "mean" in methods
        assert "weighted_mean" in methods
        assert "median" in methods
        assert "max" in methods


class TestGetAggregationMethod:
    """Tests for get_aggregation_method function."""

    def test_get_mean(self) -> None:
        """Test getting MEAN."""
        assert get_aggregation_method("mean") == AggregationMethod.MEAN

    def test_get_weighted_mean(self) -> None:
        """Test getting WEIGHTED_MEAN."""
        assert (
            get_aggregation_method("weighted_mean") == AggregationMethod.WEIGHTED_MEAN
        )

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid aggregation method"):
            get_aggregation_method("invalid")


class TestValidateAggregationMethod:
    """Tests for validate_aggregation_method function."""

    def test_valid_methods(self) -> None:
        """Test valid aggregation methods."""
        assert validate_aggregation_method("mean") is True
        assert validate_aggregation_method("weighted_mean") is True
        assert validate_aggregation_method("median") is True

    def test_invalid_methods(self) -> None:
        """Test invalid aggregation methods."""
        assert validate_aggregation_method("invalid") is False
        assert validate_aggregation_method("") is False


class TestRunEvaluationTask:
    """Tests for run_evaluation_task function."""

    def test_basic_evaluation(self) -> None:
        """Test basic evaluation task."""
        config = create_task_config(TaskType.CLASSIFICATION)
        data = [(1, 1), (2, 0), (3, 1)]

        def inference(inputs):
            return [1 for _ in inputs]

        result = run_evaluation_task("test_task", config, inference, data)
        assert result.task_name == "test_task"
        assert result.num_samples == 3
        assert "accuracy" in result.metrics

    def test_with_custom_metrics(self) -> None:
        """Test evaluation with custom metrics function."""
        config = create_task_config(TaskType.CLASSIFICATION)
        data = [(1, 1), (2, 0), (3, 1)]

        def inference(inputs):
            return [1 for _ in inputs]

        def metrics(preds, labels):
            correct = sum(p == label for p, label in zip(preds, labels, strict=False))
            return {"custom_accuracy": correct / len(preds)}

        result = run_evaluation_task("test_task", config, inference, data, metrics)
        assert "custom_accuracy" in result.metrics

    def test_with_limit(self) -> None:
        """Test evaluation with sample limit."""
        config = create_task_config(TaskType.CLASSIFICATION, limit=2)
        data = [(1, 1), (2, 0), (3, 1), (4, 0)]

        def inference(inputs):
            return [1 for _ in inputs]

        result = run_evaluation_task("test_task", config, inference, data)
        assert result.num_samples == 2

    def test_empty_task_name_raises_error(self) -> None:
        """Test that empty task_name raises ValueError."""
        config = create_task_config(TaskType.CLASSIFICATION)
        with pytest.raises(ValueError, match="task_name cannot be empty"):
            run_evaluation_task("", config, lambda x: x, [])

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            run_evaluation_task("task", None, lambda x: x, [])  # type: ignore[arg-type]

    def test_none_inference_fn_raises_error(self) -> None:
        """Test that None inference_fn raises ValueError."""
        config = create_task_config(TaskType.CLASSIFICATION)
        with pytest.raises(ValueError, match="inference_fn cannot be None"):
            run_evaluation_task("task", config, None, [])  # type: ignore[arg-type]

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        config = create_task_config(TaskType.CLASSIFICATION)
        with pytest.raises(ValueError, match="data cannot be None"):
            run_evaluation_task("task", config, lambda x: x, None)  # type: ignore[arg-type]

    def test_batching(self) -> None:
        """Test that batching works correctly."""
        config = create_task_config(TaskType.CLASSIFICATION, batch_size=2)
        data = [(i, 1) for i in range(5)]
        batch_sizes = []

        def inference(inputs):
            batch_sizes.append(len(inputs))
            return [1 for _ in inputs]

        run_evaluation_task("test", config, inference, data)
        assert batch_sizes == [2, 2, 1]

    def test_without_metrics_fn_computes_default_accuracy(self) -> None:
        """Test that default accuracy is computed when no metrics_fn provided."""
        config = create_task_config(TaskType.CLASSIFICATION)
        data = [(1, 1), (2, 0), (3, 1), (4, 1)]

        def inference(inputs):
            return [1 for _ in inputs]

        # Without metrics_fn, should compute default accuracy
        result = run_evaluation_task("test_task", config, inference, data)
        assert "accuracy" in result.metrics
        # 3 out of 4 are correct (1==1, 0!=1, 1==1, 1==1)
        assert result.metrics["accuracy"] == pytest.approx(0.75)

    def test_with_non_tuple_data(self) -> None:
        """Test evaluation with non-tuple data (no labels)."""
        config = create_task_config(TaskType.GENERATION)
        data = [1, 2, 3, 4, 5]

        def inference(inputs):
            return [x * 2 for x in inputs]

        result = run_evaluation_task("test_task", config, inference, data)
        assert result.task_name == "test_task"
        assert result.num_samples == 5
        # No labels means no metrics computed by default
        assert result.metrics == {}

    def test_exception_handling(self) -> None:
        """Test that exceptions are handled gracefully."""
        config = create_task_config(TaskType.CLASSIFICATION)
        data = [(1, 1), (2, 0)]

        def failing_inference(inputs):
            raise RuntimeError("Inference failed!")

        result = run_evaluation_task("test_task", config, failing_inference, data)
        assert result.task_name == "test_task"
        assert "error" in result.metrics
        assert result.metrics["error"] == 0.0


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_mean_aggregation(self) -> None:
        """Test mean aggregation."""
        r1 = create_evaluation_result("task1", {"accuracy": 0.8}, 100, 10.0)
        r2 = create_evaluation_result("task2", {"accuracy": 0.9}, 200, 20.0)
        agg = aggregate_results([r1, r2], AggregationMethod.MEAN)
        assert agg["accuracy"] == pytest.approx(0.85)

    def test_weighted_mean_aggregation(self) -> None:
        """Test weighted mean aggregation."""
        r1 = create_evaluation_result("task1", {"accuracy": 0.8}, 100, 10.0)
        r2 = create_evaluation_result("task2", {"accuracy": 0.9}, 200, 20.0)
        agg = aggregate_results([r1, r2], AggregationMethod.WEIGHTED_MEAN)
        # (0.8 * 100 + 0.9 * 200) / 300 = 260 / 300 = 0.8667
        assert agg["accuracy"] == pytest.approx(0.8667, rel=0.01)

    def test_median_aggregation(self) -> None:
        """Test median aggregation."""
        r1 = create_evaluation_result("task1", {"accuracy": 0.7}, 100, 10.0)
        r2 = create_evaluation_result("task2", {"accuracy": 0.8}, 100, 10.0)
        r3 = create_evaluation_result("task3", {"accuracy": 0.9}, 100, 10.0)
        agg = aggregate_results([r1, r2, r3], AggregationMethod.MEDIAN)
        assert agg["accuracy"] == pytest.approx(0.8)

    def test_median_even_count(self) -> None:
        """Test median with even number of results."""
        r1 = create_evaluation_result("task1", {"accuracy": 0.7}, 100, 10.0)
        r2 = create_evaluation_result("task2", {"accuracy": 0.9}, 100, 10.0)
        agg = aggregate_results([r1, r2], AggregationMethod.MEDIAN)
        assert agg["accuracy"] == pytest.approx(0.8)

    def test_max_aggregation(self) -> None:
        """Test max aggregation."""
        r1 = create_evaluation_result("task1", {"accuracy": 0.8}, 100, 10.0)
        r2 = create_evaluation_result("task2", {"accuracy": 0.9}, 200, 20.0)
        agg = aggregate_results([r1, r2], AggregationMethod.MAX)
        assert agg["accuracy"] == pytest.approx(0.9)

    def test_multiple_metrics(self) -> None:
        """Test aggregation of multiple metrics."""
        r1 = create_evaluation_result("task1", {"accuracy": 0.8, "f1": 0.75}, 100, 10.0)
        r2 = create_evaluation_result("task2", {"accuracy": 0.9, "f1": 0.85}, 100, 10.0)
        agg = aggregate_results([r1, r2])
        assert agg["accuracy"] == pytest.approx(0.85)
        assert agg["f1"] == pytest.approx(0.8)

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            aggregate_results(None)  # type: ignore[arg-type]

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_results([])

    def test_weighted_mean_fallback_with_zero_samples(self) -> None:
        """Test weighted mean fallback when total_samples is 0."""
        # Create results with 0 samples (edge case)
        # This tests the fallback branch in weighted_mean
        r1 = EvaluationResult(
            task_name="task1",
            metrics={"accuracy": 0.8},
            num_samples=0,  # Invalid but tests edge case
            duration_seconds=10.0,
        )
        r2 = EvaluationResult(
            task_name="task2",
            metrics={"accuracy": 0.9},
            num_samples=0,
            duration_seconds=10.0,
        )
        # With 0 total samples, should fall back to mean
        agg = aggregate_results([r1, r2], AggregationMethod.WEIGHTED_MEAN)
        assert agg["accuracy"] == pytest.approx(0.85)


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    def test_markdown_format(self) -> None:
        """Test markdown format output."""
        r1 = create_evaluation_result("mmlu", {"accuracy": 0.65}, 1000, 120.0)
        r2 = create_evaluation_result("hellaswag", {"accuracy": 0.79}, 500, 60.0)
        table = format_results_table([r1, r2], OutputFormat.MARKDOWN)
        assert "mmlu" in table
        assert "hellaswag" in table
        assert "|" in table

    def test_json_format(self) -> None:
        """Test JSON format output."""
        r1 = create_evaluation_result("mmlu", {"accuracy": 0.65}, 1000, 120.0)
        table = format_results_table([r1], OutputFormat.JSON)
        data = json.loads(table)
        assert len(data) == 1
        assert data[0]["task_name"] == "mmlu"
        assert data[0]["accuracy"] == pytest.approx(0.65)

    def test_csv_format(self) -> None:
        """Test CSV format output."""
        r1 = create_evaluation_result("mmlu", {"accuracy": 0.65}, 1000, 120.0)
        table = format_results_table([r1], OutputFormat.CSV)
        assert "mmlu" in table
        assert "," in table
        lines = table.strip().split("\n")
        assert len(lines) == 2  # Header + data

    def test_latex_format(self) -> None:
        """Test LaTeX format output."""
        r1 = create_evaluation_result("mmlu", {"accuracy": 0.65}, 1000, 120.0)
        table = format_results_table([r1], OutputFormat.LATEX)
        assert "\\begin{tabular}" in table
        assert "\\end{tabular}" in table
        assert "mmlu" in table

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_results_table(None)  # type: ignore[arg-type]

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            format_results_table([])


class TestEstimateEvaluationTime:
    """Tests for estimate_evaluation_time function."""

    def test_basic_estimation(self) -> None:
        """Test basic time estimation."""
        time_estimate = estimate_evaluation_time(1000, 32, 0.5)
        # ceil(1000/32) = 32 batches * 0.5 = 16.0
        assert time_estimate == pytest.approx(16.0)

    def test_exact_batches(self) -> None:
        """Test with exact batch division."""
        time_estimate = estimate_evaluation_time(100, 10, 1.0)
        assert time_estimate == pytest.approx(10.0)

    def test_partial_batch(self) -> None:
        """Test with partial final batch."""
        time_estimate = estimate_evaluation_time(15, 10, 1.0)
        # ceil(15/10) = 2 batches * 1.0 = 2.0
        assert time_estimate == pytest.approx(2.0)

    def test_zero_num_samples_raises_error(self) -> None:
        """Test that zero num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            estimate_evaluation_time(0, 32, 0.5)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_evaluation_time(100, 0, 0.5)

    def test_zero_avg_inference_time_raises_error(self) -> None:
        """Test that zero avg_inference_time raises ValueError."""
        with pytest.raises(ValueError, match="avg_inference_time must be positive"):
            estimate_evaluation_time(100, 32, 0)


class TestFormatHarnessStats:
    """Tests for format_harness_stats function."""

    def test_format_stats(self) -> None:
        """Test formatting stats."""
        stats = HarnessStats(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            total_samples=5000,
            avg_score=0.72,
        )
        formatted = format_harness_stats(stats)
        assert "5" in formatted
        assert "4" in formatted
        assert "1" in formatted
        assert "5000" in formatted
        assert "0.72" in formatted
        assert "Success Rate" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_harness_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedHarnessConfig:
    """Tests for get_recommended_harness_config function."""

    def test_base_model(self) -> None:
        """Test configuration for base model."""
        config = get_recommended_harness_config("base")
        assert "tasks" in config
        assert len(config["tasks"]) > 0
        assert config["num_fewshot"] == 5

    def test_instruction_model(self) -> None:
        """Test configuration for instruction model."""
        config = get_recommended_harness_config("instruction")
        assert "tasks" in config
        assert config["num_fewshot"] == 0

    def test_code_model(self) -> None:
        """Test configuration for code model."""
        config = get_recommended_harness_config("code")
        assert "tasks" in config
        assert "humaneval" in config["tasks"] or "mbpp" in config["tasks"]

    def test_chat_model(self) -> None:
        """Test configuration for chat model."""
        config = get_recommended_harness_config("chat")
        assert "tasks" in config
        assert config["batch_size"] == 1

    def test_reasoning_model(self) -> None:
        """Test configuration for reasoning model."""
        config = get_recommended_harness_config("reasoning")
        assert "tasks" in config
        assert config["num_fewshot"] == 8

    def test_unknown_model_type_uses_default(self) -> None:
        """Test that unknown model type uses default (base) config."""
        config = get_recommended_harness_config("unknown_type")
        assert "tasks" in config
        assert len(config["tasks"]) > 0

    def test_num_tasks_limit(self) -> None:
        """Test that num_tasks limits the task list."""
        config = get_recommended_harness_config("base", num_tasks=2)
        assert len(config["tasks"]) == 2

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_harness_config("")


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        num_fewshot=st.integers(min_value=0, max_value=100),
        batch_size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=50)
    def test_task_config_valid_range(self, num_fewshot: int, batch_size: int) -> None:
        """Test that valid parameter ranges create valid configs."""
        config = create_task_config(
            TaskType.MULTIPLE_CHOICE,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
        )
        assert config.num_fewshot == num_fewshot
        assert config.batch_size == batch_size

    @given(
        num_samples=st.integers(min_value=1, max_value=10000),
        batch_size=st.integers(min_value=1, max_value=128),
        avg_time=st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_time_estimation_positive(
        self, num_samples: int, batch_size: int, avg_time: float
    ) -> None:
        """Test that time estimation is always positive."""
        time_estimate = estimate_evaluation_time(num_samples, batch_size, avg_time)
        assert time_estimate > 0

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=10,
        ),
        samples=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_aggregation_bounds(self, scores: list[float], samples: list[int]) -> None:
        """Test that aggregation stays within bounds."""
        # Ensure same length
        min_len = min(len(scores), len(samples))
        scores = scores[:min_len]
        samples = samples[:min_len]

        results = [
            create_evaluation_result(
                f"task{i}",
                {"accuracy": score},
                sample_count,
                1.0,
            )
            for i, (score, sample_count) in enumerate(
                zip(scores, samples, strict=False)
            )
        ]

        for method in AggregationMethod:
            agg = aggregate_results(results, method)
            assert 0.0 <= agg["accuracy"] <= 1.0


class TestDocstrings:
    """Tests to verify doctests pass."""

    def test_task_type_doctest(self) -> None:
        """Verify TaskType doctest examples."""
        assert TaskType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert TaskType.GENERATION.value == "generation"
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.EXTRACTION.value == "extraction"
        assert TaskType.RANKING.value == "ranking"

    def test_output_format_doctest(self) -> None:
        """Verify OutputFormat doctest examples."""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.CSV.value == "csv"
        assert OutputFormat.MARKDOWN.value == "markdown"
        assert OutputFormat.LATEX.value == "latex"

    def test_aggregation_method_doctest(self) -> None:
        """Verify AggregationMethod doctest examples."""
        assert AggregationMethod.MEAN.value == "mean"
        assert AggregationMethod.WEIGHTED_MEAN.value == "weighted_mean"
        assert AggregationMethod.MEDIAN.value == "median"
        assert AggregationMethod.MAX.value == "max"
