"""Tests for benchmark functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTask,
    TimingResult,
    aggregate_benchmark_results,
    compare_benchmark_results,
    compute_percentile,
    compute_timing_stats,
    create_benchmark_runner,
    format_benchmark_result,
    get_benchmark_task,
    list_benchmark_tasks,
    run_benchmark,
    validate_benchmark_config,
    validate_benchmark_task,
)


class TestBenchmarkTask:
    """Tests for BenchmarkTask enum."""

    def test_text_classification_value(self) -> None:
        """Test TEXT_CLASSIFICATION value."""
        assert BenchmarkTask.TEXT_CLASSIFICATION.value == "text_classification"

    def test_question_answering_value(self) -> None:
        """Test QUESTION_ANSWERING value."""
        assert BenchmarkTask.QUESTION_ANSWERING.value == "question_answering"

    def test_summarization_value(self) -> None:
        """Test SUMMARIZATION value."""
        assert BenchmarkTask.SUMMARIZATION.value == "summarization"

    def test_translation_value(self) -> None:
        """Test TRANSLATION value."""
        assert BenchmarkTask.TRANSLATION.value == "translation"

    def test_ner_value(self) -> None:
        """Test NER value."""
        assert BenchmarkTask.NER.value == "ner"

    def test_sentiment_value(self) -> None:
        """Test SENTIMENT value."""
        assert BenchmarkTask.SENTIMENT.value == "sentiment"

    def test_custom_value(self) -> None:
        """Test CUSTOM value."""
        assert BenchmarkTask.CUSTOM.value == "custom"


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_required_name(self) -> None:
        """Test that name is required."""
        config = BenchmarkConfig(name="test")
        assert config.name == "test"

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BenchmarkConfig(name="test")
        assert config.task == BenchmarkTask.CUSTOM
        assert config.num_samples is None
        assert config.batch_size == 32
        assert config.warmup_runs == 1
        assert config.num_runs == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BenchmarkConfig(
            name="custom",
            task=BenchmarkTask.TEXT_CLASSIFICATION,
            num_samples=100,
            batch_size=64,
            warmup_runs=2,
            num_runs=5,
        )
        assert config.name == "custom"
        assert config.task == BenchmarkTask.TEXT_CLASSIFICATION
        assert config.num_samples == 100
        assert config.batch_size == 64

    def test_frozen(self) -> None:
        """Test that BenchmarkConfig is immutable."""
        config = BenchmarkConfig(name="test")
        with pytest.raises(AttributeError):
            config.name = "new"  # type: ignore[misc]


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_creation(self) -> None:
        """Test creating TimingResult instance."""
        result = TimingResult(
            total_time=1.5,
            samples_per_second=1000.0,
            latency_p50=0.8,
            latency_p90=1.2,
            latency_p99=1.8,
        )
        assert result.total_time == pytest.approx(1.5)
        assert result.samples_per_second == pytest.approx(1000.0)
        assert result.latency_p50 == pytest.approx(0.8)

    def test_frozen(self) -> None:
        """Test that TimingResult is immutable."""
        result = TimingResult(1.0, 100.0, 0.5, 0.8, 1.0)
        with pytest.raises(AttributeError):
            result.total_time = 2.0  # type: ignore[misc]


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self) -> None:
        """Test creating BenchmarkResult instance."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(1.0, 100.0, 0.5, 0.8, 1.0)
        result = BenchmarkResult(
            config=config,
            timing=timing,
            metrics={"accuracy": 0.95},
            samples_evaluated=100,
            success=True,
            error_message=None,
        )
        assert result.success is True
        assert result.metrics["accuracy"] == pytest.approx(0.95)

    def test_failed_result(self) -> None:
        """Test creating failed BenchmarkResult."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        result = BenchmarkResult(
            config=config,
            timing=timing,
            metrics={},
            samples_evaluated=0,
            success=False,
            error_message="Test error",
        )
        assert result.success is False
        assert result.error_message == "Test error"


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner dataclass."""

    def test_creation(self) -> None:
        """Test creating BenchmarkRunner instance."""
        config = BenchmarkConfig(name="test")
        runner = BenchmarkRunner(config)
        assert runner.config.name == "test"
        assert runner.latencies == []


class TestValidateBenchmarkConfig:
    """Tests for validate_benchmark_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BenchmarkConfig(name="test", batch_size=16)
        validate_benchmark_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_benchmark_config(None)  # type: ignore[arg-type]

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        config = BenchmarkConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_benchmark_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = BenchmarkConfig(name="test", batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_benchmark_config(config)

    def test_negative_warmup_raises_error(self) -> None:
        """Test that negative warmup_runs raises ValueError."""
        config = BenchmarkConfig(name="test", warmup_runs=-1)
        with pytest.raises(ValueError, match="warmup_runs cannot be negative"):
            validate_benchmark_config(config)

    def test_zero_num_runs_raises_error(self) -> None:
        """Test that zero num_runs raises ValueError."""
        config = BenchmarkConfig(name="test", num_runs=0)
        with pytest.raises(ValueError, match="num_runs must be positive"):
            validate_benchmark_config(config)


class TestComputePercentile:
    """Tests for compute_percentile function."""

    def test_median(self) -> None:
        """Test computing median (50th percentile)."""
        result = compute_percentile([1, 2, 3, 4, 5], 50)
        assert result == pytest.approx(3.0)

    def test_zero_percentile(self) -> None:
        """Test 0th percentile (minimum)."""
        result = compute_percentile([1, 2, 3, 4, 5], 0)
        assert result == pytest.approx(1.0)

    def test_100_percentile(self) -> None:
        """Test 100th percentile (maximum)."""
        result = compute_percentile([1, 2, 3, 4, 5], 100)
        assert result == pytest.approx(5.0)

    def test_25th_percentile(self) -> None:
        """Test 25th percentile."""
        result = compute_percentile([1, 2, 3, 4, 5], 25)
        assert 1.0 <= result <= 2.0

    def test_75th_percentile(self) -> None:
        """Test 75th percentile."""
        result = compute_percentile([1, 2, 3, 4, 5], 75)
        assert 4.0 <= result <= 5.0

    def test_single_value(self) -> None:
        """Test with single value."""
        result = compute_percentile([5], 50)
        assert result == pytest.approx(5.0)

    def test_none_values_raises_error(self) -> None:
        """Test that None values raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compute_percentile(None, 50)  # type: ignore[arg-type]

    def test_empty_values_raises_error(self) -> None:
        """Test that empty values raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_percentile([], 50)

    def test_invalid_percentile_raises_error(self) -> None:
        """Test that invalid percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
            compute_percentile([1, 2], 150)

    def test_negative_percentile_raises_error(self) -> None:
        """Test that negative percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be between 0 and 100"):
            compute_percentile([1, 2], -10)


class TestComputeTimingStats:
    """Tests for compute_timing_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic timing statistics."""
        latencies = [0.1, 0.15, 0.12, 0.11, 0.13]
        result = compute_timing_stats(latencies, 500)
        assert result.total_time > 0
        assert result.samples_per_second > 0
        assert result.latency_p50 > 0

    def test_none_latencies_raises_error(self) -> None:
        """Test that None latencies raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compute_timing_stats(None, 100)  # type: ignore[arg-type]

    def test_empty_latencies_raises_error(self) -> None:
        """Test that empty latencies raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_timing_stats([], 100)

    def test_zero_samples_raises_error(self) -> None:
        """Test that zero total_samples raises ValueError."""
        with pytest.raises(ValueError, match="total_samples must be positive"):
            compute_timing_stats([0.1], 0)


class TestCreateBenchmarkRunner:
    """Tests for create_benchmark_runner function."""

    def test_creates_runner(self) -> None:
        """Test that function creates a runner."""
        config = BenchmarkConfig(name="test")
        runner = create_benchmark_runner(config)
        assert isinstance(runner, BenchmarkRunner)
        assert runner.config.name == "test"

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_benchmark_runner(None)  # type: ignore[arg-type]

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        config = BenchmarkConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_benchmark_runner(config)


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    def test_basic_run(self) -> None:
        """Test basic benchmark run."""
        config = BenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
        runner = create_benchmark_runner(config)
        data = list(range(100))

        result = run_benchmark(runner, data, lambda x: x)
        assert result.success is True
        assert result.samples_evaluated == 100

    def test_with_num_samples(self) -> None:
        """Test benchmark with num_samples limit."""
        config = BenchmarkConfig(name="test", num_samples=50, warmup_runs=0, num_runs=1)
        runner = create_benchmark_runner(config)
        data = list(range(100))

        result = run_benchmark(runner, data, lambda x: x)
        assert result.samples_evaluated == 50

    def test_with_metrics_fn(self) -> None:
        """Test benchmark with metrics function."""
        config = BenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
        runner = create_benchmark_runner(config)
        data = list(range(10))

        def metrics_fn(predictions: list, samples: list) -> dict[str, float]:
            return {"accuracy": 0.95}

        result = run_benchmark(runner, data, lambda x: x, metrics_fn)
        assert result.metrics["accuracy"] == pytest.approx(0.95)

    def test_handles_error(self) -> None:
        """Test that errors are handled."""
        config = BenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
        runner = create_benchmark_runner(config)
        data = list(range(10))

        def failing_fn(x: list) -> None:
            msg = "Test error"
            raise RuntimeError(msg)

        result = run_benchmark(runner, data, failing_fn)
        assert result.success is False
        assert "Test error" in (result.error_message or "")

    def test_none_runner_raises_error(self) -> None:
        """Test that None runner raises ValueError."""
        with pytest.raises(ValueError, match="runner cannot be None"):
            run_benchmark(None, [], lambda x: x)  # type: ignore[arg-type]

    def test_none_data_raises_error(self) -> None:
        """Test that None data raises ValueError."""
        config = BenchmarkConfig(name="test")
        runner = create_benchmark_runner(config)
        with pytest.raises(ValueError, match="data cannot be None"):
            run_benchmark(runner, None, lambda x: x)  # type: ignore[arg-type]

    def test_none_inference_fn_raises_error(self) -> None:
        """Test that None inference_fn raises ValueError."""
        config = BenchmarkConfig(name="test")
        runner = create_benchmark_runner(config)
        with pytest.raises(ValueError, match="inference_fn cannot be None"):
            run_benchmark(runner, [], None)  # type: ignore[arg-type]

    def test_latencies_recorded(self) -> None:
        """Test that latencies are recorded in runner."""
        config = BenchmarkConfig(name="test", warmup_runs=0, num_runs=3)
        runner = create_benchmark_runner(config)
        data = list(range(10))

        run_benchmark(runner, data, lambda x: x)
        assert len(runner.latencies) == 3


class TestCompareBenchmarkResults:
    """Tests for compare_benchmark_results function."""

    def test_compare_two_results(self) -> None:
        """Test comparing two benchmark results."""
        config1 = BenchmarkConfig(name="model1")
        config2 = BenchmarkConfig(name="model2")
        timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        timing2 = TimingResult(0.8, 125.0, 4.0, 6.0, 8.0)
        r1 = BenchmarkResult(config1, timing1, {"accuracy": 0.9}, 100, True, None)
        r2 = BenchmarkResult(config2, timing2, {"accuracy": 0.95}, 100, True, None)

        comparison = compare_benchmark_results([r1, r2])
        assert comparison["fastest"] == "model2"
        assert comparison["slowest"] == "model1"
        assert comparison["successful"] == 2

    def test_compare_with_failed(self) -> None:
        """Test comparing with failed result."""
        config1 = BenchmarkConfig(name="model1")
        config2 = BenchmarkConfig(name="model2")
        timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        timing2 = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        r1 = BenchmarkResult(config1, timing1, {}, 100, True, None)
        r2 = BenchmarkResult(config2, timing2, {}, 0, False, "Error")

        comparison = compare_benchmark_results([r1, r2])
        assert comparison["successful"] == 1
        assert comparison["failed"] == 1

    def test_all_failed(self) -> None:
        """Test comparing when all results failed."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        r = BenchmarkResult(config, timing, {}, 0, False, "Error")

        comparison = compare_benchmark_results([r])
        assert comparison["fastest"] is None
        assert comparison["slowest"] is None

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compare_benchmark_results(None)  # type: ignore[arg-type]

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compare_benchmark_results([])


class TestListBenchmarkTasks:
    """Tests for list_benchmark_tasks function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        tasks = list_benchmark_tasks()
        assert isinstance(tasks, list)

    def test_contains_expected_tasks(self) -> None:
        """Test that list contains expected tasks."""
        tasks = list_benchmark_tasks()
        assert "text_classification" in tasks
        assert "question_answering" in tasks
        assert "custom" in tasks

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        tasks = list_benchmark_tasks()
        assert tasks == sorted(tasks)


class TestValidateBenchmarkTask:
    """Tests for validate_benchmark_task function."""

    def test_valid_text_classification(self) -> None:
        """Test validation of text_classification."""
        assert validate_benchmark_task("text_classification") is True

    def test_valid_custom(self) -> None:
        """Test validation of custom."""
        assert validate_benchmark_task("custom") is True

    def test_invalid_task(self) -> None:
        """Test validation of invalid task."""
        assert validate_benchmark_task("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_benchmark_task("") is False


class TestGetBenchmarkTask:
    """Tests for get_benchmark_task function."""

    def test_get_text_classification(self) -> None:
        """Test getting TEXT_CLASSIFICATION."""
        result = get_benchmark_task("text_classification")
        assert result == BenchmarkTask.TEXT_CLASSIFICATION

    def test_get_custom(self) -> None:
        """Test getting CUSTOM."""
        result = get_benchmark_task("custom")
        assert result == BenchmarkTask.CUSTOM

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="invalid benchmark task"):
            get_benchmark_task("invalid")


class TestFormatBenchmarkResult:
    """Tests for format_benchmark_result function."""

    def test_format_success(self) -> None:
        """Test formatting successful result."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        result = BenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)

        formatted = format_benchmark_result(result)
        assert "test" in formatted
        assert "100" in formatted
        assert "0.9500" in formatted

    def test_format_failure(self) -> None:
        """Test formatting failed result."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        result = BenchmarkResult(config, timing, {}, 0, False, "Test error")

        formatted = format_benchmark_result(result)
        assert "Test error" in formatted

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_benchmark_result(None)  # type: ignore[arg-type]


class TestAggregateBenchmarkResults:
    """Tests for aggregate_benchmark_results function."""

    def test_aggregate_metrics(self) -> None:
        """Test aggregating metrics from multiple results."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        r1 = BenchmarkResult(config, timing, {"acc": 0.9}, 100, True, None)
        r2 = BenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)

        agg = aggregate_benchmark_results([r1, r2])
        assert 0.92 < agg["acc_mean"] < 0.93
        assert agg["acc_min"] == pytest.approx(0.9)
        assert agg["acc_max"] == pytest.approx(0.95)

    def test_aggregate_timing(self) -> None:
        """Test aggregating timing from multiple results."""
        config = BenchmarkConfig(name="test")
        timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        timing2 = TimingResult(2.0, 50.0, 5.0, 8.0, 10.0)
        r1 = BenchmarkResult(config, timing1, {}, 100, True, None)
        r2 = BenchmarkResult(config, timing2, {}, 100, True, None)

        agg = aggregate_benchmark_results([r1, r2])
        assert agg["total_time_mean"] == pytest.approx(1.5)
        assert agg["throughput_mean"] == pytest.approx(75.0)

    def test_aggregate_empty(self) -> None:
        """Test aggregating empty results."""
        agg = aggregate_benchmark_results([])
        assert agg == {}

    def test_aggregate_all_failed(self) -> None:
        """Test aggregating when all failed."""
        config = BenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        r = BenchmarkResult(config, timing, {}, 0, False, "Error")

        agg = aggregate_benchmark_results([r])
        assert agg == {}

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            aggregate_benchmark_results(None)  # type: ignore[arg-type]


class TestPropertyBased:
    """Property-based tests for benchmark functions."""

    @given(
        st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=1, max_size=50)
    )
    @settings(max_examples=10)
    def test_percentile_bounds(self, values: list[float]) -> None:
        """Test that percentiles are within value bounds."""
        p50 = compute_percentile(values, 50)
        assert min(values) <= p50 <= max(values)

    @given(
        st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=1, max_size=50)
    )
    @settings(max_examples=10)
    def test_percentile_ordering(self, values: list[float]) -> None:
        """Test that higher percentiles yield higher or equal values."""
        p25 = compute_percentile(values, 25)
        p50 = compute_percentile(values, 50)
        p75 = compute_percentile(values, 75)
        assert p25 <= p50 <= p75
