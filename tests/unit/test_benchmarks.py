"""Tests for benchmark functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.evaluation.benchmarks import (
    VALID_BENCHMARK_TASKS,
    VALID_BENCHMARK_TYPES,
    VALID_EVALUATION_MODES,
    VALID_SCORING_METHODS,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkStats,
    BenchmarkTask,
    BenchmarkType,
    EvaluationMode,
    HumanEvalConfig,
    LegacyBenchmarkConfig,
    LegacyBenchmarkResult,
    MMLUConfig,
    ScoringMethod,
    TimingResult,
    aggregate_benchmark_results,
    aggregate_results,
    calculate_benchmark_score,
    calculate_confidence_interval,
    compare_benchmark_results,
    compare_benchmarks,
    compute_percentile,
    compute_timing_stats,
    create_benchmark_config,
    create_benchmark_result,
    create_benchmark_runner,
    create_humaneval_config,
    create_mmlu_config,
    format_benchmark_result,
    format_benchmark_stats,
    get_benchmark_task,
    get_benchmark_type,
    get_evaluation_mode,
    get_recommended_benchmark_config,
    get_scoring_method,
    list_benchmark_tasks,
    list_benchmark_types,
    list_evaluation_modes,
    list_scoring_methods,
    run_benchmark,
    validate_benchmark_config,
    validate_benchmark_result,
    validate_benchmark_stats,
    validate_benchmark_task,
    validate_benchmark_type,
    validate_evaluation_mode,
    validate_humaneval_config,
    validate_legacy_benchmark_config,
    validate_mmlu_config,
    validate_scoring_method,
)


class TestBenchmarkType:
    """Tests for BenchmarkType enum."""

    def test_mmlu_value(self) -> None:
        """Test MMLU value."""
        assert BenchmarkType.MMLU.value == "mmlu"

    def test_hellaswag_value(self) -> None:
        """Test HELLASWAG value."""
        assert BenchmarkType.HELLASWAG.value == "hellaswag"

    def test_truthfulqa_value(self) -> None:
        """Test TRUTHFULQA value."""
        assert BenchmarkType.TRUTHFULQA.value == "truthfulqa"

    def test_arc_value(self) -> None:
        """Test ARC value."""
        assert BenchmarkType.ARC.value == "arc"

    def test_winogrande_value(self) -> None:
        """Test WINOGRANDE value."""
        assert BenchmarkType.WINOGRANDE.value == "winogrande"

    def test_gsm8k_value(self) -> None:
        """Test GSM8K value."""
        assert BenchmarkType.GSM8K.value == "gsm8k"

    def test_humaneval_value(self) -> None:
        """Test HUMANEVAL value."""
        assert BenchmarkType.HUMANEVAL.value == "humaneval"

    def test_mbpp_value(self) -> None:
        """Test MBPP value."""
        assert BenchmarkType.MBPP.value == "mbpp"

    def test_valid_benchmark_types_frozenset(self) -> None:
        """Test VALID_BENCHMARK_TYPES frozenset."""
        assert isinstance(VALID_BENCHMARK_TYPES, frozenset)
        assert "mmlu" in VALID_BENCHMARK_TYPES
        assert "hellaswag" in VALID_BENCHMARK_TYPES


class TestEvaluationMode:
    """Tests for EvaluationMode enum."""

    def test_zero_shot_value(self) -> None:
        """Test ZERO_SHOT value."""
        assert EvaluationMode.ZERO_SHOT.value == "zero_shot"

    def test_few_shot_value(self) -> None:
        """Test FEW_SHOT value."""
        assert EvaluationMode.FEW_SHOT.value == "few_shot"

    def test_chain_of_thought_value(self) -> None:
        """Test CHAIN_OF_THOUGHT value."""
        assert EvaluationMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"

    def test_valid_evaluation_modes_frozenset(self) -> None:
        """Test VALID_EVALUATION_MODES frozenset."""
        assert isinstance(VALID_EVALUATION_MODES, frozenset)
        assert "zero_shot" in VALID_EVALUATION_MODES


class TestScoringMethod:
    """Tests for ScoringMethod enum."""

    def test_exact_match_value(self) -> None:
        """Test EXACT_MATCH value."""
        assert ScoringMethod.EXACT_MATCH.value == "exact_match"

    def test_f1_value(self) -> None:
        """Test F1 value."""
        assert ScoringMethod.F1.value == "f1"

    def test_accuracy_value(self) -> None:
        """Test ACCURACY value."""
        assert ScoringMethod.ACCURACY.value == "accuracy"

    def test_pass_at_k_value(self) -> None:
        """Test PASS_AT_K value."""
        assert ScoringMethod.PASS_AT_K.value == "pass_at_k"

    def test_valid_scoring_methods_frozenset(self) -> None:
        """Test VALID_SCORING_METHODS frozenset."""
        assert isinstance(VALID_SCORING_METHODS, frozenset)
        assert "exact_match" in VALID_SCORING_METHODS


class TestBenchmarkTask:
    """Tests for BenchmarkTask enum (legacy)."""

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

    def test_valid_benchmark_tasks_frozenset(self) -> None:
        """Test VALID_BENCHMARK_TASKS frozenset."""
        assert isinstance(VALID_BENCHMARK_TASKS, frozenset)
        assert "text_classification" in VALID_BENCHMARK_TASKS


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = BenchmarkConfig(benchmark_type=BenchmarkType.MMLU)
        assert config.benchmark_type == BenchmarkType.MMLU
        assert config.num_few_shot == 0
        assert config.evaluation_mode == EvaluationMode.ZERO_SHOT
        assert config.subset is None

    def test_creation_with_all_values(self) -> None:
        """Test creating config with all values."""
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.MMLU,
            num_few_shot=5,
            evaluation_mode=EvaluationMode.FEW_SHOT,
            subset="abstract_algebra",
        )
        assert config.benchmark_type == BenchmarkType.MMLU
        assert config.num_few_shot == 5
        assert config.evaluation_mode == EvaluationMode.FEW_SHOT
        assert config.subset == "abstract_algebra"

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = BenchmarkConfig(benchmark_type=BenchmarkType.MMLU)
        with pytest.raises(AttributeError):
            config.num_few_shot = 5  # type: ignore[misc]


class TestMMLUConfig:
    """Tests for MMLUConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = MMLUConfig()
        assert config.subjects is None
        assert config.num_few_shot == 5

    def test_with_subjects(self) -> None:
        """Test with subjects specified."""
        config = MMLUConfig(subjects=("abstract_algebra", "anatomy"))
        assert config.subjects == ("abstract_algebra", "anatomy")

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = MMLUConfig()
        with pytest.raises(AttributeError):
            config.num_few_shot = 3  # type: ignore[misc]


class TestHumanEvalConfig:
    """Tests for HumanEvalConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = HumanEvalConfig()
        assert config.k_values == (1, 10, 100)
        assert config.timeout_seconds == 10.0

    def test_custom_values(self) -> None:
        """Test with custom values."""
        config = HumanEvalConfig(k_values=(1, 5), timeout_seconds=5.0)
        assert config.k_values == (1, 5)
        assert config.timeout_seconds == 5.0

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = HumanEvalConfig()
        with pytest.raises(AttributeError):
            config.timeout_seconds = 5.0  # type: ignore[misc]


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self) -> None:
        """Test creating result."""
        result = BenchmarkResult(
            benchmark=BenchmarkType.MMLU,
            score=0.65,
            num_samples=1000,
            breakdown={"abstract_algebra": 0.7},
        )
        assert result.benchmark == BenchmarkType.MMLU
        assert result.score == pytest.approx(0.65)
        assert result.num_samples == 1000
        assert result.breakdown["abstract_algebra"] == pytest.approx(0.7)

    def test_frozen(self) -> None:
        """Test that result is immutable."""
        result = BenchmarkResult(
            benchmark=BenchmarkType.MMLU,
            score=0.65,
            num_samples=1000,
            breakdown={},
        )
        with pytest.raises(AttributeError):
            result.score = 0.7  # type: ignore[misc]


class TestBenchmarkStats:
    """Tests for BenchmarkStats dataclass."""

    def test_creation(self) -> None:
        """Test creating stats."""
        stats = BenchmarkStats(
            overall_score=0.72,
            per_task_scores={"mmlu": 0.65, "hellaswag": 0.79},
            confidence_interval=(0.70, 0.74),
        )
        assert stats.overall_score == pytest.approx(0.72)
        assert stats.per_task_scores["mmlu"] == pytest.approx(0.65)
        assert stats.confidence_interval[0] == pytest.approx(0.70)

    def test_frozen(self) -> None:
        """Test that stats is immutable."""
        stats = BenchmarkStats(
            overall_score=0.72,
            per_task_scores={},
            confidence_interval=(0.70, 0.74),
        )
        with pytest.raises(AttributeError):
            stats.overall_score = 0.8  # type: ignore[misc]


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


class TestLegacyBenchmarkConfig:
    """Tests for LegacyBenchmarkConfig dataclass."""

    def test_required_name(self) -> None:
        """Test that name is required."""
        config = LegacyBenchmarkConfig(name="test")
        assert config.name == "test"

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LegacyBenchmarkConfig(name="test")
        assert config.task == BenchmarkTask.CUSTOM
        assert config.num_samples is None
        assert config.batch_size == 32
        assert config.warmup_runs == 1
        assert config.num_runs == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LegacyBenchmarkConfig(
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
        """Test that LegacyBenchmarkConfig is immutable."""
        config = LegacyBenchmarkConfig(name="test")
        with pytest.raises(AttributeError):
            config.name = "new"  # type: ignore[misc]


class TestLegacyBenchmarkResult:
    """Tests for LegacyBenchmarkResult dataclass."""

    def test_creation(self) -> None:
        """Test creating LegacyBenchmarkResult instance."""
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(1.0, 100.0, 0.5, 0.8, 1.0)
        result = LegacyBenchmarkResult(
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
        """Test creating failed LegacyBenchmarkResult."""
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        result = LegacyBenchmarkResult(
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
        config = LegacyBenchmarkConfig(name="test")
        runner = BenchmarkRunner(config)
        assert runner.config.name == "test"
        assert runner.latencies == []


class TestValidateBenchmarkConfig:
    """Tests for validate_benchmark_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BenchmarkConfig(benchmark_type=BenchmarkType.MMLU)
        validate_benchmark_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_benchmark_config(None)  # type: ignore[arg-type]

    def test_negative_num_few_shot_raises_error(self) -> None:
        """Test that negative num_few_shot raises ValueError."""
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.MMLU,
            num_few_shot=-1,
        )
        with pytest.raises(ValueError, match="num_few_shot cannot be negative"):
            validate_benchmark_config(config)

    def test_few_shot_mode_with_zero_examples_raises_error(self) -> None:
        """Test that few_shot mode with zero examples raises ValueError."""
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.MMLU,
            num_few_shot=0,
            evaluation_mode=EvaluationMode.FEW_SHOT,
        )
        with pytest.raises(ValueError, match="num_few_shot must be > 0"):
            validate_benchmark_config(config)


class TestValidateMMLUConfig:
    """Tests for validate_mmlu_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = MMLUConfig()
        validate_mmlu_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_mmlu_config(None)  # type: ignore[arg-type]

    def test_negative_num_few_shot_raises_error(self) -> None:
        """Test that negative num_few_shot raises ValueError."""
        config = MMLUConfig(num_few_shot=-1)
        with pytest.raises(ValueError, match="num_few_shot cannot be negative"):
            validate_mmlu_config(config)

    def test_empty_subject_raises_error(self) -> None:
        """Test that empty subject string raises ValueError."""
        config = MMLUConfig(subjects=("abstract_algebra", ""))
        with pytest.raises(ValueError, match="cannot contain empty strings"):
            validate_mmlu_config(config)


class TestValidateHumanEvalConfig:
    """Tests for validate_humaneval_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = HumanEvalConfig()
        validate_humaneval_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_humaneval_config(None)  # type: ignore[arg-type]

    def test_empty_k_values_raises_error(self) -> None:
        """Test that empty k_values raises ValueError."""
        config = HumanEvalConfig(k_values=())
        with pytest.raises(ValueError, match="k_values cannot be empty"):
            validate_humaneval_config(config)

    def test_negative_k_raises_error(self) -> None:
        """Test that negative k value raises ValueError."""
        config = HumanEvalConfig(k_values=(1, -5))
        with pytest.raises(ValueError, match="k values must be positive"):
            validate_humaneval_config(config)

    def test_zero_k_raises_error(self) -> None:
        """Test that zero k value raises ValueError."""
        config = HumanEvalConfig(k_values=(0,))
        with pytest.raises(ValueError, match="k values must be positive"):
            validate_humaneval_config(config)

    def test_negative_timeout_raises_error(self) -> None:
        """Test that negative timeout raises ValueError."""
        config = HumanEvalConfig(timeout_seconds=-1.0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            validate_humaneval_config(config)


class TestValidateBenchmarkResult:
    """Tests for validate_benchmark_result function."""

    def test_valid_result(self) -> None:
        """Test validation of valid result."""
        result = BenchmarkResult(
            benchmark=BenchmarkType.MMLU,
            score=0.65,
            num_samples=100,
            breakdown={},
        )
        validate_benchmark_result(result)  # Should not raise

    def test_none_result_raises_error(self) -> None:
        """Test that None result raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_benchmark_result(None)  # type: ignore[arg-type]

    def test_score_above_one_raises_error(self) -> None:
        """Test that score above 1 raises ValueError."""
        result = BenchmarkResult(
            benchmark=BenchmarkType.MMLU,
            score=1.5,
            num_samples=100,
            breakdown={},
        )
        with pytest.raises(ValueError, match="score must be between 0 and 1"):
            validate_benchmark_result(result)

    def test_negative_score_raises_error(self) -> None:
        """Test that negative score raises ValueError."""
        result = BenchmarkResult(
            benchmark=BenchmarkType.MMLU,
            score=-0.1,
            num_samples=100,
            breakdown={},
        )
        with pytest.raises(ValueError, match="score must be between 0 and 1"):
            validate_benchmark_result(result)

    def test_zero_num_samples_raises_error(self) -> None:
        """Test that zero num_samples raises ValueError."""
        result = BenchmarkResult(
            benchmark=BenchmarkType.MMLU,
            score=0.65,
            num_samples=0,
            breakdown={},
        )
        with pytest.raises(ValueError, match="num_samples must be positive"):
            validate_benchmark_result(result)


class TestValidateBenchmarkStats:
    """Tests for validate_benchmark_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = BenchmarkStats(
            overall_score=0.72,
            per_task_scores={},
            confidence_interval=(0.70, 0.74),
        )
        validate_benchmark_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_benchmark_stats(None)  # type: ignore[arg-type]

    def test_score_above_one_raises_error(self) -> None:
        """Test that overall_score above 1 raises ValueError."""
        stats = BenchmarkStats(
            overall_score=1.5,
            per_task_scores={},
            confidence_interval=(0.70, 0.74),
        )
        with pytest.raises(ValueError, match="overall_score must be between 0 and 1"):
            validate_benchmark_stats(stats)

    def test_invalid_confidence_interval_raises_error(self) -> None:
        """Test that invalid confidence interval raises ValueError."""
        stats = BenchmarkStats(
            overall_score=0.72,
            per_task_scores={},
            confidence_interval=(0.80, 0.70),  # lower > upper
        )
        with pytest.raises(ValueError, match="cannot be greater than upper bound"):
            validate_benchmark_stats(stats)


class TestValidateLegacyBenchmarkConfig:
    """Tests for validate_legacy_benchmark_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = LegacyBenchmarkConfig(name="test", batch_size=16)
        validate_legacy_benchmark_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_legacy_benchmark_config(None)  # type: ignore[arg-type]

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        config = LegacyBenchmarkConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            validate_legacy_benchmark_config(config)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = LegacyBenchmarkConfig(name="test", batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_legacy_benchmark_config(config)

    def test_negative_warmup_raises_error(self) -> None:
        """Test that negative warmup_runs raises ValueError."""
        config = LegacyBenchmarkConfig(name="test", warmup_runs=-1)
        with pytest.raises(ValueError, match="warmup_runs cannot be negative"):
            validate_legacy_benchmark_config(config)

    def test_zero_num_runs_raises_error(self) -> None:
        """Test that zero num_runs raises ValueError."""
        config = LegacyBenchmarkConfig(name="test", num_runs=0)
        with pytest.raises(ValueError, match="num_runs must be positive"):
            validate_legacy_benchmark_config(config)


class TestCreateBenchmarkConfig:
    """Tests for create_benchmark_config factory function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_benchmark_config(BenchmarkType.MMLU)
        assert config.benchmark_type == BenchmarkType.MMLU

    def test_with_all_parameters(self) -> None:
        """Test with all parameters."""
        config = create_benchmark_config(
            BenchmarkType.HELLASWAG,
            num_few_shot=10,
            evaluation_mode=EvaluationMode.FEW_SHOT,
            subset="test",
        )
        assert config.num_few_shot == 10
        assert config.evaluation_mode == EvaluationMode.FEW_SHOT
        assert config.subset == "test"


class TestCreateMMLUConfig:
    """Tests for create_mmlu_config factory function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_mmlu_config()
        assert config.num_few_shot == 5

    def test_with_subjects(self) -> None:
        """Test with subjects."""
        config = create_mmlu_config(subjects=("math", "science"))
        assert config.subjects == ("math", "science")


class TestCreateHumanEvalConfig:
    """Tests for create_humaneval_config factory function."""

    def test_creates_config(self) -> None:
        """Test that function creates a config."""
        config = create_humaneval_config()
        assert config.k_values == (1, 10, 100)

    def test_with_custom_values(self) -> None:
        """Test with custom values."""
        config = create_humaneval_config(k_values=(1, 5), timeout_seconds=5.0)
        assert config.k_values == (1, 5)
        assert config.timeout_seconds == 5.0


class TestCreateBenchmarkResult:
    """Tests for create_benchmark_result factory function."""

    def test_creates_result(self) -> None:
        """Test that function creates a result."""
        result = create_benchmark_result(
            BenchmarkType.MMLU,
            0.65,
            1000,
        )
        assert result.score == pytest.approx(0.65)
        assert result.breakdown == {}

    def test_with_breakdown(self) -> None:
        """Test with breakdown."""
        result = create_benchmark_result(
            BenchmarkType.MMLU,
            0.65,
            1000,
            {"math": 0.7, "science": 0.6},
        )
        assert result.breakdown["math"] == pytest.approx(0.7)


class TestListBenchmarkTypes:
    """Tests for list_benchmark_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_benchmark_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_benchmark_types()
        assert "mmlu" in types
        assert "hellaswag" in types
        assert "truthfulqa" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_benchmark_types()
        assert types == sorted(types)


class TestGetBenchmarkType:
    """Tests for get_benchmark_type function."""

    def test_get_mmlu(self) -> None:
        """Test getting MMLU."""
        result = get_benchmark_type("mmlu")
        assert result == BenchmarkType.MMLU

    def test_get_hellaswag(self) -> None:
        """Test getting HELLASWAG."""
        result = get_benchmark_type("hellaswag")
        assert result == BenchmarkType.HELLASWAG

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid benchmark type"):
            get_benchmark_type("invalid")


class TestValidateBenchmarkType:
    """Tests for validate_benchmark_type function."""

    def test_valid_mmlu(self) -> None:
        """Test validation of mmlu."""
        assert validate_benchmark_type("mmlu") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_benchmark_type("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_benchmark_type("") is False


class TestListEvaluationModes:
    """Tests for list_evaluation_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        modes = list_evaluation_modes()
        assert isinstance(modes, list)

    def test_contains_expected_modes(self) -> None:
        """Test that list contains expected modes."""
        modes = list_evaluation_modes()
        assert "zero_shot" in modes
        assert "few_shot" in modes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        modes = list_evaluation_modes()
        assert modes == sorted(modes)


class TestGetEvaluationMode:
    """Tests for get_evaluation_mode function."""

    def test_get_zero_shot(self) -> None:
        """Test getting ZERO_SHOT."""
        result = get_evaluation_mode("zero_shot")
        assert result == EvaluationMode.ZERO_SHOT

    def test_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid evaluation mode"):
            get_evaluation_mode("invalid")


class TestValidateEvaluationMode:
    """Tests for validate_evaluation_mode function."""

    def test_valid_zero_shot(self) -> None:
        """Test validation of zero_shot."""
        assert validate_evaluation_mode("zero_shot") is True

    def test_invalid_mode(self) -> None:
        """Test validation of invalid mode."""
        assert validate_evaluation_mode("invalid") is False


class TestListScoringMethods:
    """Tests for list_scoring_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_scoring_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_scoring_methods()
        assert "exact_match" in methods
        assert "accuracy" in methods


class TestGetScoringMethod:
    """Tests for get_scoring_method function."""

    def test_get_exact_match(self) -> None:
        """Test getting EXACT_MATCH."""
        result = get_scoring_method("exact_match")
        assert result == ScoringMethod.EXACT_MATCH

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid scoring method"):
            get_scoring_method("invalid")


class TestValidateScoringMethod:
    """Tests for validate_scoring_method function."""

    def test_valid_exact_match(self) -> None:
        """Test validation of exact_match."""
        assert validate_scoring_method("exact_match") is True

    def test_invalid_method(self) -> None:
        """Test validation of invalid method."""
        assert validate_scoring_method("invalid") is False


class TestListBenchmarkTasks:
    """Tests for list_benchmark_tasks function (legacy)."""

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


class TestCalculateBenchmarkScore:
    """Tests for calculate_benchmark_score function."""

    def test_basic_calculation(self) -> None:
        """Test basic score calculation."""
        assert calculate_benchmark_score(80, 100) == pytest.approx(0.8)

    def test_zero_correct(self) -> None:
        """Test zero correct."""
        assert calculate_benchmark_score(0, 100) == pytest.approx(0.0)

    def test_all_correct(self) -> None:
        """Test all correct."""
        assert calculate_benchmark_score(100, 100) == pytest.approx(1.0)

    def test_negative_correct_raises_error(self) -> None:
        """Test that negative correct raises ValueError."""
        with pytest.raises(ValueError, match="correct cannot be negative"):
            calculate_benchmark_score(-1, 100)

    def test_zero_total_raises_error(self) -> None:
        """Test that zero total raises ValueError."""
        with pytest.raises(ValueError, match="total must be positive"):
            calculate_benchmark_score(50, 0)

    def test_correct_greater_than_total_raises_error(self) -> None:
        """Test that correct > total raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater than total"):
            calculate_benchmark_score(101, 100)


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_two_results(self) -> None:
        """Test aggregating two results."""
        r1 = BenchmarkResult(BenchmarkType.MMLU, 0.65, 100, {})
        r2 = BenchmarkResult(BenchmarkType.HELLASWAG, 0.79, 100, {})
        stats = aggregate_results([r1, r2])
        assert 0.70 <= stats.overall_score <= 0.75

    def test_none_results_raises_error(self) -> None:
        """Test that None results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            aggregate_results(None)  # type: ignore[arg-type]

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_results([])


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval function."""

    def test_basic_interval(self) -> None:
        """Test basic confidence interval."""
        lower, upper = calculate_confidence_interval([0.7, 0.8, 0.75])
        assert 0.65 <= lower <= 0.75
        assert 0.75 <= upper <= 0.85

    def test_single_value(self) -> None:
        """Test with single value."""
        lower, upper = calculate_confidence_interval([0.7])
        assert lower == upper == pytest.approx(0.7)

    def test_none_scores_raises_error(self) -> None:
        """Test that None scores raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_confidence_interval(None)  # type: ignore[arg-type]

    def test_empty_scores_raises_error(self) -> None:
        """Test that empty scores raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_confidence_interval([])

    def test_invalid_confidence_level_raises_error(self) -> None:
        """Test that invalid confidence level raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            calculate_confidence_interval([0.7, 0.8], confidence_level=1.5)


class TestCompareBenchmarks:
    """Tests for compare_benchmarks function."""

    def test_compare_two_sets(self) -> None:
        """Test comparing two result sets."""
        r1 = [BenchmarkResult(BenchmarkType.MMLU, 0.65, 100, {})]
        r2 = [BenchmarkResult(BenchmarkType.MMLU, 0.70, 100, {})]
        comparison = compare_benchmarks(r1, r2)
        assert comparison["better"] == "b"

    def test_none_results_a_raises_error(self) -> None:
        """Test that None results_a raises ValueError."""
        r2 = [BenchmarkResult(BenchmarkType.MMLU, 0.70, 100, {})]
        with pytest.raises(ValueError, match="results_a cannot be None"):
            compare_benchmarks(None, r2)  # type: ignore[arg-type]

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raises ValueError."""
        r2 = [BenchmarkResult(BenchmarkType.MMLU, 0.70, 100, {})]
        with pytest.raises(ValueError, match="results_a cannot be empty"):
            compare_benchmarks([], r2)


class TestFormatBenchmarkStats:
    """Tests for format_benchmark_stats function."""

    def test_format_stats(self) -> None:
        """Test formatting stats."""
        stats = BenchmarkStats(
            overall_score=0.72,
            per_task_scores={"mmlu": 0.65, "hellaswag": 0.79},
            confidence_interval=(0.70, 0.74),
        )
        formatted = format_benchmark_stats(stats)
        assert "0.72" in formatted
        assert "mmlu" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            format_benchmark_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedBenchmarkConfig:
    """Tests for get_recommended_benchmark_config function."""

    def test_llm_config(self) -> None:
        """Test LLM config."""
        config = get_recommended_benchmark_config("llm")
        assert "benchmarks" in config
        assert BenchmarkType.MMLU in config["benchmarks"]

    def test_code_config(self) -> None:
        """Test code config."""
        config = get_recommended_benchmark_config("code")
        assert BenchmarkType.HUMANEVAL in config["benchmarks"]

    def test_math_config(self) -> None:
        """Test math config."""
        config = get_recommended_benchmark_config("math")
        assert BenchmarkType.GSM8K in config["benchmarks"]

    def test_empty_model_type_raises_error(self) -> None:
        """Test that empty model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            get_recommended_benchmark_config("")

    def test_none_model_type_raises_error(self) -> None:
        """Test that None model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type cannot be None"):
            get_recommended_benchmark_config(None)  # type: ignore[arg-type]

    def test_with_task_type(self) -> None:
        """Test with task_type specified."""
        config = get_recommended_benchmark_config("llm", task_type="zero_shot")
        assert config["evaluation_mode"] == EvaluationMode.ZERO_SHOT


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
        config = LegacyBenchmarkConfig(name="test")
        runner = create_benchmark_runner(config)
        assert isinstance(runner, BenchmarkRunner)
        assert runner.config.name == "test"

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_benchmark_runner(None)  # type: ignore[arg-type]

    def test_invalid_config_raises_error(self) -> None:
        """Test that invalid config raises ValueError."""
        config = LegacyBenchmarkConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_benchmark_runner(config)


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    def test_basic_run(self) -> None:
        """Test basic benchmark run."""
        config = LegacyBenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
        runner = create_benchmark_runner(config)
        data = list(range(100))

        result = run_benchmark(runner, data, lambda x: x)
        assert result.success is True
        assert result.samples_evaluated == 100

    def test_with_num_samples(self) -> None:
        """Test benchmark with num_samples limit."""
        config = LegacyBenchmarkConfig(
            name="test", num_samples=50, warmup_runs=0, num_runs=1
        )
        runner = create_benchmark_runner(config)
        data = list(range(100))

        result = run_benchmark(runner, data, lambda x: x)
        assert result.samples_evaluated == 50

    def test_with_metrics_fn(self) -> None:
        """Test benchmark with metrics function."""
        config = LegacyBenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
        runner = create_benchmark_runner(config)
        data = list(range(10))

        def metrics_fn(predictions: list, samples: list) -> dict[str, float]:
            return {"accuracy": 0.95}

        result = run_benchmark(runner, data, lambda x: x, metrics_fn)
        assert result.metrics["accuracy"] == pytest.approx(0.95)

    def test_handles_error(self) -> None:
        """Test that errors are handled."""
        config = LegacyBenchmarkConfig(name="test", warmup_runs=0, num_runs=1)
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
        config = LegacyBenchmarkConfig(name="test")
        runner = create_benchmark_runner(config)
        with pytest.raises(ValueError, match="data cannot be None"):
            run_benchmark(runner, None, lambda x: x)  # type: ignore[arg-type]

    def test_none_inference_fn_raises_error(self) -> None:
        """Test that None inference_fn raises ValueError."""
        config = LegacyBenchmarkConfig(name="test")
        runner = create_benchmark_runner(config)
        with pytest.raises(ValueError, match="inference_fn cannot be None"):
            run_benchmark(runner, [], None)  # type: ignore[arg-type]

    def test_latencies_recorded(self) -> None:
        """Test that latencies are recorded in runner."""
        config = LegacyBenchmarkConfig(name="test", warmup_runs=0, num_runs=3)
        runner = create_benchmark_runner(config)
        data = list(range(10))

        run_benchmark(runner, data, lambda x: x)
        assert len(runner.latencies) == 3


class TestCompareBenchmarkResults:
    """Tests for compare_benchmark_results function."""

    def test_compare_two_results(self) -> None:
        """Test comparing two benchmark results."""
        config1 = LegacyBenchmarkConfig(name="model1")
        config2 = LegacyBenchmarkConfig(name="model2")
        timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        timing2 = TimingResult(0.8, 125.0, 4.0, 6.0, 8.0)
        r1 = LegacyBenchmarkResult(config1, timing1, {"accuracy": 0.9}, 100, True, None)
        r2 = LegacyBenchmarkResult(
            config2, timing2, {"accuracy": 0.95}, 100, True, None
        )

        comparison = compare_benchmark_results([r1, r2])
        assert comparison["fastest"] == "model2"
        assert comparison["slowest"] == "model1"
        assert comparison["successful"] == 2

    def test_compare_with_failed(self) -> None:
        """Test comparing with failed result."""
        config1 = LegacyBenchmarkConfig(name="model1")
        config2 = LegacyBenchmarkConfig(name="model2")
        timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        timing2 = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        r1 = LegacyBenchmarkResult(config1, timing1, {}, 100, True, None)
        r2 = LegacyBenchmarkResult(config2, timing2, {}, 0, False, "Error")

        comparison = compare_benchmark_results([r1, r2])
        assert comparison["successful"] == 1
        assert comparison["failed"] == 1

    def test_all_failed(self) -> None:
        """Test comparing when all results failed."""
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        r = LegacyBenchmarkResult(config, timing, {}, 0, False, "Error")

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


class TestFormatBenchmarkResult:
    """Tests for format_benchmark_result function."""

    def test_format_success(self) -> None:
        """Test formatting successful result."""
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        result = LegacyBenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)

        formatted = format_benchmark_result(result)
        assert "test" in formatted
        assert "100" in formatted
        assert "0.9500" in formatted

    def test_format_failure(self) -> None:
        """Test formatting failed result."""
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        result = LegacyBenchmarkResult(config, timing, {}, 0, False, "Test error")

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
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        r1 = LegacyBenchmarkResult(config, timing, {"acc": 0.9}, 100, True, None)
        r2 = LegacyBenchmarkResult(config, timing, {"acc": 0.95}, 100, True, None)

        agg = aggregate_benchmark_results([r1, r2])
        assert 0.92 < agg["acc_mean"] < 0.93
        assert agg["acc_min"] == pytest.approx(0.9)
        assert agg["acc_max"] == pytest.approx(0.95)

    def test_aggregate_timing(self) -> None:
        """Test aggregating timing from multiple results."""
        config = LegacyBenchmarkConfig(name="test")
        timing1 = TimingResult(1.0, 100.0, 5.0, 8.0, 10.0)
        timing2 = TimingResult(2.0, 50.0, 5.0, 8.0, 10.0)
        r1 = LegacyBenchmarkResult(config, timing1, {}, 100, True, None)
        r2 = LegacyBenchmarkResult(config, timing2, {}, 100, True, None)

        agg = aggregate_benchmark_results([r1, r2])
        assert agg["total_time_mean"] == pytest.approx(1.5)
        assert agg["throughput_mean"] == pytest.approx(75.0)

    def test_aggregate_empty(self) -> None:
        """Test aggregating empty results."""
        agg = aggregate_benchmark_results([])
        assert agg == {}

    def test_aggregate_all_failed(self) -> None:
        """Test aggregating when all failed."""
        config = LegacyBenchmarkConfig(name="test")
        timing = TimingResult(0.0, 0.0, 0.0, 0.0, 0.0)
        r = LegacyBenchmarkResult(config, timing, {}, 0, False, "Error")

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

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_benchmark_score_bounds(self, correct: int, total: int) -> None:
        """Test that benchmark score is always between 0 and 1."""
        if correct > total:
            return  # Skip invalid cases
        score = calculate_benchmark_score(correct, total)
        assert 0.0 <= score <= 1.0

    @given(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=10)
    def test_confidence_interval_contains_mean(self, scores: list[float]) -> None:
        """Test that confidence interval contains the mean."""
        lower, upper = calculate_confidence_interval(scores)
        mean = sum(scores) / len(scores)
        assert lower <= mean <= upper
