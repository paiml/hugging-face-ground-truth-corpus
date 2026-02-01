"""Tests for training.curriculum module."""

from __future__ import annotations

import pytest

from hf_gtc.training.curriculum import (
    VALID_DIFFICULTY_METRICS,
    VALID_PACING_FUNCTIONS,
    VALID_SAMPLING_STRATEGIES,
    CurriculumConfig,
    CurriculumStats,
    DifficultyConfig,
    DifficultyMetric,
    PacingConfig,
    PacingFunction,
    SamplingStrategy,
    calculate_competence_score,
    calculate_sample_difficulty,
    calculate_sample_weights,
    create_curriculum_config,
    create_curriculum_stats,
    create_difficulty_config,
    create_pacing_config,
    format_curriculum_stats,
    get_difficulty_at_step,
    get_difficulty_metric,
    get_pacing_function,
    get_recommended_curriculum_config,
    get_sampling_strategy,
    list_difficulty_metrics,
    list_pacing_functions,
    list_sampling_strategies,
    validate_curriculum_config,
    validate_curriculum_stats,
    validate_difficulty_config,
    validate_pacing_config,
)


class TestDifficultyMetric:
    """Tests for DifficultyMetric enum."""

    def test_all_metrics_have_values(self) -> None:
        """All metrics have string values."""
        for metric in DifficultyMetric:
            assert isinstance(metric.value, str)

    def test_length_value(self) -> None:
        """Length has correct value."""
        assert DifficultyMetric.LENGTH.value == "length"

    def test_perplexity_value(self) -> None:
        """Perplexity has correct value."""
        assert DifficultyMetric.PERPLEXITY.value == "perplexity"

    def test_loss_value(self) -> None:
        """Loss has correct value."""
        assert DifficultyMetric.LOSS.value == "loss"

    def test_confidence_value(self) -> None:
        """Confidence has correct value."""
        assert DifficultyMetric.CONFIDENCE.value == "confidence"

    def test_manual_value(self) -> None:
        """Manual has correct value."""
        assert DifficultyMetric.MANUAL.value == "manual"

    def test_valid_metrics_frozenset(self) -> None:
        """VALID_DIFFICULTY_METRICS is a frozenset."""
        assert isinstance(VALID_DIFFICULTY_METRICS, frozenset)
        assert len(VALID_DIFFICULTY_METRICS) == 5


class TestPacingFunction:
    """Tests for PacingFunction enum."""

    def test_all_functions_have_values(self) -> None:
        """All functions have string values."""
        for func in PacingFunction:
            assert isinstance(func.value, str)

    def test_linear_value(self) -> None:
        """Linear has correct value."""
        assert PacingFunction.LINEAR.value == "linear"

    def test_exponential_value(self) -> None:
        """Exponential has correct value."""
        assert PacingFunction.EXPONENTIAL.value == "exponential"

    def test_step_value(self) -> None:
        """Step has correct value."""
        assert PacingFunction.STEP.value == "step"

    def test_self_paced_value(self) -> None:
        """Self-paced has correct value."""
        assert PacingFunction.SELF_PACED.value == "self_paced"

    def test_competence_value(self) -> None:
        """Competence has correct value."""
        assert PacingFunction.COMPETENCE.value == "competence"

    def test_valid_functions_frozenset(self) -> None:
        """VALID_PACING_FUNCTIONS is a frozenset."""
        assert isinstance(VALID_PACING_FUNCTIONS, frozenset)
        assert len(VALID_PACING_FUNCTIONS) == 5


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in SamplingStrategy:
            assert isinstance(strategy.value, str)

    def test_threshold_value(self) -> None:
        """Threshold has correct value."""
        assert SamplingStrategy.THRESHOLD.value == "threshold"

    def test_weighted_value(self) -> None:
        """Weighted has correct value."""
        assert SamplingStrategy.WEIGHTED.value == "weighted"

    def test_probabilistic_value(self) -> None:
        """Probabilistic has correct value."""
        assert SamplingStrategy.PROBABILISTIC.value == "probabilistic"

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_SAMPLING_STRATEGIES is a frozenset."""
        assert isinstance(VALID_SAMPLING_STRATEGIES, frozenset)
        assert len(VALID_SAMPLING_STRATEGIES) == 3


class TestDifficultyConfig:
    """Tests for DifficultyConfig dataclass."""

    def test_create_config(self) -> None:
        """Create difficulty config."""
        config = DifficultyConfig(
            metric=DifficultyMetric.LENGTH,
            normalize=True,
            buckets=10,
            ascending=True,
        )
        assert config.metric == DifficultyMetric.LENGTH
        assert config.normalize is True
        assert config.buckets == 10

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = DifficultyConfig(DifficultyMetric.LENGTH, True, 10, True)
        with pytest.raises(AttributeError):
            config.buckets = 5  # type: ignore[misc]


class TestPacingConfig:
    """Tests for PacingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create pacing config."""
        config = PacingConfig(
            function=PacingFunction.LINEAR,
            initial_difficulty=0.0,
            target_difficulty=1.0,
            warmup_steps=1000,
        )
        assert config.function == PacingFunction.LINEAR
        assert config.initial_difficulty == pytest.approx(0.0)
        assert config.warmup_steps == 1000

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.0, 0)
        with pytest.raises(AttributeError):
            config.warmup_steps = 500  # type: ignore[misc]


class TestCurriculumConfig:
    """Tests for CurriculumConfig dataclass."""

    def test_create_config(self) -> None:
        """Create curriculum config."""
        diff_config = DifficultyConfig(DifficultyMetric.LENGTH, True, 10, True)
        pacing_config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.0, 0)
        config = CurriculumConfig(
            difficulty_config=diff_config,
            pacing_config=pacing_config,
            sampling_strategy=SamplingStrategy.THRESHOLD,
        )
        assert config.difficulty_config == diff_config
        assert config.pacing_config == pacing_config
        assert config.sampling_strategy == SamplingStrategy.THRESHOLD

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        diff_config = DifficultyConfig(DifficultyMetric.LENGTH, True, 10, True)
        pacing_config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.0, 0)
        config = CurriculumConfig(
            diff_config, pacing_config, SamplingStrategy.THRESHOLD
        )
        with pytest.raises(AttributeError):
            config.sampling_strategy = SamplingStrategy.WEIGHTED  # type: ignore[misc]


class TestCurriculumStats:
    """Tests for CurriculumStats dataclass."""

    def test_create_stats(self) -> None:
        """Create curriculum stats."""
        stats = CurriculumStats(
            current_difficulty=0.5,
            samples_seen=10000,
            competence_score=0.8,
            curriculum_progress=0.5,
        )
        assert stats.current_difficulty == pytest.approx(0.5)
        assert stats.samples_seen == 10000
        assert stats.competence_score == pytest.approx(0.8)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = CurriculumStats(0.5, 10000, 0.8, 0.5)
        with pytest.raises(AttributeError):
            stats.samples_seen = 20000  # type: ignore[misc]


class TestValidateDifficultyConfig:
    """Tests for validate_difficulty_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DifficultyConfig(DifficultyMetric.LENGTH, True, 10, True)
        validate_difficulty_config(config)

    def test_zero_buckets_raises(self) -> None:
        """Zero buckets raises ValueError."""
        config = DifficultyConfig(DifficultyMetric.LENGTH, True, 0, True)
        with pytest.raises(ValueError, match="buckets must be positive"):
            validate_difficulty_config(config)

    def test_negative_buckets_raises(self) -> None:
        """Negative buckets raises ValueError."""
        config = DifficultyConfig(DifficultyMetric.LENGTH, True, -1, True)
        with pytest.raises(ValueError, match="buckets must be positive"):
            validate_difficulty_config(config)


class TestValidatePacingConfig:
    """Tests for validate_pacing_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.0, 1000)
        validate_pacing_config(config)

    def test_negative_initial_difficulty_raises(self) -> None:
        """Negative initial_difficulty raises ValueError."""
        config = PacingConfig(PacingFunction.LINEAR, -0.1, 1.0, 0)
        with pytest.raises(ValueError, match="initial_difficulty must be between"):
            validate_pacing_config(config)

    def test_initial_difficulty_above_one_raises(self) -> None:
        """initial_difficulty > 1 raises ValueError."""
        config = PacingConfig(PacingFunction.LINEAR, 1.5, 1.0, 0)
        with pytest.raises(ValueError, match="initial_difficulty must be between"):
            validate_pacing_config(config)

    def test_negative_target_difficulty_raises(self) -> None:
        """Negative target_difficulty raises ValueError."""
        config = PacingConfig(PacingFunction.LINEAR, 0.0, -0.1, 0)
        with pytest.raises(ValueError, match="target_difficulty must be between"):
            validate_pacing_config(config)

    def test_target_difficulty_above_one_raises(self) -> None:
        """target_difficulty > 1 raises ValueError."""
        config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.5, 0)
        with pytest.raises(ValueError, match="target_difficulty must be between"):
            validate_pacing_config(config)

    def test_negative_warmup_steps_raises(self) -> None:
        """Negative warmup_steps raises ValueError."""
        config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.0, -1)
        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            validate_pacing_config(config)


class TestValidateCurriculumConfig:
    """Tests for validate_curriculum_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = create_curriculum_config()
        validate_curriculum_config(config)

    def test_invalid_difficulty_config_raises(self) -> None:
        """Invalid difficulty config raises ValueError."""
        bad_diff_config = DifficultyConfig(DifficultyMetric.LENGTH, True, -1, True)
        pacing_config = PacingConfig(PacingFunction.LINEAR, 0.0, 1.0, 0)
        config = CurriculumConfig(
            bad_diff_config, pacing_config, SamplingStrategy.THRESHOLD
        )
        with pytest.raises(ValueError, match="buckets must be positive"):
            validate_curriculum_config(config)

    def test_invalid_pacing_config_raises(self) -> None:
        """Invalid pacing config raises ValueError."""
        diff_config = DifficultyConfig(DifficultyMetric.LENGTH, True, 10, True)
        bad_pacing_config = PacingConfig(PacingFunction.LINEAR, -0.1, 1.0, 0)
        config = CurriculumConfig(
            diff_config, bad_pacing_config, SamplingStrategy.THRESHOLD
        )
        with pytest.raises(ValueError, match="initial_difficulty must be between"):
            validate_curriculum_config(config)


class TestValidateCurriculumStats:
    """Tests for validate_curriculum_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = CurriculumStats(0.5, 10000, 0.8, 0.5)
        validate_curriculum_stats(stats)

    def test_negative_current_difficulty_raises(self) -> None:
        """Negative current_difficulty raises ValueError."""
        stats = CurriculumStats(-0.1, 10000, 0.8, 0.5)
        with pytest.raises(ValueError, match="current_difficulty must be between"):
            validate_curriculum_stats(stats)

    def test_current_difficulty_above_one_raises(self) -> None:
        """current_difficulty > 1 raises ValueError."""
        stats = CurriculumStats(1.5, 10000, 0.8, 0.5)
        with pytest.raises(ValueError, match="current_difficulty must be between"):
            validate_curriculum_stats(stats)

    def test_negative_samples_seen_raises(self) -> None:
        """Negative samples_seen raises ValueError."""
        stats = CurriculumStats(0.5, -1, 0.8, 0.5)
        with pytest.raises(ValueError, match="samples_seen must be non-negative"):
            validate_curriculum_stats(stats)

    def test_negative_competence_score_raises(self) -> None:
        """Negative competence_score raises ValueError."""
        stats = CurriculumStats(0.5, 10000, -0.1, 0.5)
        with pytest.raises(ValueError, match="competence_score must be between"):
            validate_curriculum_stats(stats)

    def test_competence_score_above_one_raises(self) -> None:
        """competence_score > 1 raises ValueError."""
        stats = CurriculumStats(0.5, 10000, 1.5, 0.5)
        with pytest.raises(ValueError, match="competence_score must be between"):
            validate_curriculum_stats(stats)

    def test_negative_curriculum_progress_raises(self) -> None:
        """Negative curriculum_progress raises ValueError."""
        stats = CurriculumStats(0.5, 10000, 0.8, -0.1)
        with pytest.raises(ValueError, match="curriculum_progress must be between"):
            validate_curriculum_stats(stats)

    def test_curriculum_progress_above_one_raises(self) -> None:
        """curriculum_progress > 1 raises ValueError."""
        stats = CurriculumStats(0.5, 10000, 0.8, 1.5)
        with pytest.raises(ValueError, match="curriculum_progress must be between"):
            validate_curriculum_stats(stats)


class TestCreateDifficultyConfig:
    """Tests for create_difficulty_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_difficulty_config()
        assert config.metric == DifficultyMetric.LENGTH
        assert config.normalize is True
        assert config.buckets == 10

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_difficulty_config(
            metric="perplexity",
            normalize=False,
            buckets=5,
            ascending=False,
        )
        assert config.metric == DifficultyMetric.PERPLEXITY
        assert config.normalize is False
        assert config.buckets == 5
        assert config.ascending is False

    def test_with_enum_metric(self) -> None:
        """Create with enum metric."""
        config = create_difficulty_config(metric=DifficultyMetric.LOSS)
        assert config.metric == DifficultyMetric.LOSS

    @pytest.mark.parametrize(
        "metric",
        ["length", "perplexity", "loss", "confidence", "manual"],
    )
    def test_all_metrics(self, metric: str) -> None:
        """Test all difficulty metrics."""
        config = create_difficulty_config(metric=metric)
        assert config.metric.value == metric

    def test_invalid_buckets_raises(self) -> None:
        """Invalid buckets raises ValueError."""
        with pytest.raises(ValueError, match="buckets must be positive"):
            create_difficulty_config(buckets=0)


class TestCreatePacingConfig:
    """Tests for create_pacing_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_pacing_config()
        assert config.function == PacingFunction.LINEAR
        assert config.initial_difficulty == pytest.approx(0.0)
        assert config.target_difficulty == pytest.approx(1.0)

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_pacing_config(
            function="exponential",
            initial_difficulty=0.2,
            target_difficulty=0.8,
            warmup_steps=500,
        )
        assert config.function == PacingFunction.EXPONENTIAL
        assert config.initial_difficulty == pytest.approx(0.2)
        assert config.warmup_steps == 500

    def test_with_enum_function(self) -> None:
        """Create with enum function."""
        config = create_pacing_config(function=PacingFunction.STEP)
        assert config.function == PacingFunction.STEP

    @pytest.mark.parametrize(
        "function",
        ["linear", "exponential", "step", "self_paced", "competence"],
    )
    def test_all_functions(self, function: str) -> None:
        """Test all pacing functions."""
        config = create_pacing_config(function=function)
        assert config.function.value == function

    def test_invalid_initial_difficulty_raises(self) -> None:
        """Invalid initial_difficulty raises ValueError."""
        with pytest.raises(ValueError, match="initial_difficulty must be between"):
            create_pacing_config(initial_difficulty=1.5)


class TestCreateCurriculumConfig:
    """Tests for create_curriculum_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_curriculum_config()
        assert config.difficulty_config.metric == DifficultyMetric.LENGTH
        assert config.pacing_config.function == PacingFunction.LINEAR
        assert config.sampling_strategy == SamplingStrategy.THRESHOLD

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_curriculum_config(
            metric="perplexity",
            function="exponential",
            sampling_strategy="weighted",
        )
        assert config.difficulty_config.metric == DifficultyMetric.PERPLEXITY
        assert config.pacing_config.function == PacingFunction.EXPONENTIAL
        assert config.sampling_strategy == SamplingStrategy.WEIGHTED

    def test_with_enum_sampling_strategy(self) -> None:
        """Create with enum sampling strategy."""
        config = create_curriculum_config(
            sampling_strategy=SamplingStrategy.PROBABILISTIC
        )
        assert config.sampling_strategy == SamplingStrategy.PROBABILISTIC

    @pytest.mark.parametrize(
        "strategy",
        ["threshold", "weighted", "probabilistic"],
    )
    def test_all_strategies(self, strategy: str) -> None:
        """Test all sampling strategies."""
        config = create_curriculum_config(sampling_strategy=strategy)
        assert config.sampling_strategy.value == strategy

    def test_invalid_buckets_raises(self) -> None:
        """Invalid buckets raises ValueError."""
        with pytest.raises(ValueError, match="buckets must be positive"):
            create_curriculum_config(buckets=-1)


class TestCreateCurriculumStats:
    """Tests for create_curriculum_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_curriculum_stats()
        assert stats.current_difficulty == pytest.approx(0.0)
        assert stats.samples_seen == 0
        assert stats.competence_score == pytest.approx(0.0)
        assert stats.curriculum_progress == pytest.approx(0.0)

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_curriculum_stats(
            current_difficulty=0.5,
            samples_seen=10000,
            competence_score=0.8,
            curriculum_progress=0.5,
        )
        assert stats.current_difficulty == pytest.approx(0.5)
        assert stats.samples_seen == 10000

    def test_invalid_current_difficulty_raises(self) -> None:
        """Invalid current_difficulty raises ValueError."""
        with pytest.raises(ValueError, match="current_difficulty must be between"):
            create_curriculum_stats(current_difficulty=1.5)

    def test_negative_samples_seen_raises(self) -> None:
        """Negative samples_seen raises ValueError."""
        with pytest.raises(ValueError, match="samples_seen must be non-negative"):
            create_curriculum_stats(samples_seen=-1)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_difficulty_metrics_sorted(self) -> None:
        """Returns sorted list."""
        metrics = list_difficulty_metrics()
        assert metrics == sorted(metrics)
        assert "length" in metrics
        assert "perplexity" in metrics

    def test_list_pacing_functions_sorted(self) -> None:
        """Returns sorted list."""
        functions = list_pacing_functions()
        assert functions == sorted(functions)
        assert "linear" in functions
        assert "exponential" in functions

    def test_list_sampling_strategies_sorted(self) -> None:
        """Returns sorted list."""
        strategies = list_sampling_strategies()
        assert strategies == sorted(strategies)
        assert "threshold" in strategies
        assert "weighted" in strategies


class TestGetDifficultyMetric:
    """Tests for get_difficulty_metric function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("length", DifficultyMetric.LENGTH),
            ("perplexity", DifficultyMetric.PERPLEXITY),
            ("loss", DifficultyMetric.LOSS),
            ("confidence", DifficultyMetric.CONFIDENCE),
            ("manual", DifficultyMetric.MANUAL),
        ],
    )
    def test_all_metrics(self, name: str, expected: DifficultyMetric) -> None:
        """Test all valid metrics."""
        assert get_difficulty_metric(name) == expected

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            get_difficulty_metric("invalid")


class TestGetPacingFunction:
    """Tests for get_pacing_function function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("linear", PacingFunction.LINEAR),
            ("exponential", PacingFunction.EXPONENTIAL),
            ("step", PacingFunction.STEP),
            ("self_paced", PacingFunction.SELF_PACED),
            ("competence", PacingFunction.COMPETENCE),
        ],
    )
    def test_all_functions(self, name: str, expected: PacingFunction) -> None:
        """Test all valid functions."""
        assert get_pacing_function(name) == expected

    def test_invalid_function_raises(self) -> None:
        """Invalid function raises ValueError."""
        with pytest.raises(ValueError, match="pacing_function must be one of"):
            get_pacing_function("invalid")


class TestGetSamplingStrategy:
    """Tests for get_sampling_strategy function."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("threshold", SamplingStrategy.THRESHOLD),
            ("weighted", SamplingStrategy.WEIGHTED),
            ("probabilistic", SamplingStrategy.PROBABILISTIC),
        ],
    )
    def test_all_strategies(self, name: str, expected: SamplingStrategy) -> None:
        """Test all valid strategies."""
        assert get_sampling_strategy(name) == expected

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="sampling_strategy must be one of"):
            get_sampling_strategy("invalid")


class TestCalculateSampleDifficulty:
    """Tests for calculate_sample_difficulty function."""

    def test_basic_normalization(self) -> None:
        """Basic normalization calculation."""
        assert calculate_sample_difficulty(50.0, 0.0, 100.0) == pytest.approx(0.5)
        assert calculate_sample_difficulty(0.0, 0.0, 100.0) == pytest.approx(0.0)
        assert calculate_sample_difficulty(100.0, 0.0, 100.0) == pytest.approx(1.0)

    def test_descending_order(self) -> None:
        """Descending order inverts values."""
        assert calculate_sample_difficulty(0.0, 0.0, 100.0, ascending=False) == 1.0
        assert calculate_sample_difficulty(100.0, 0.0, 100.0, ascending=False) == 0.0

    def test_clamping(self) -> None:
        """Values outside range are clamped."""
        assert calculate_sample_difficulty(-10.0, 0.0, 100.0) == pytest.approx(0.0)
        assert calculate_sample_difficulty(110.0, 0.0, 100.0) == pytest.approx(1.0)

    def test_equal_min_max_raises(self) -> None:
        """Equal min and max raises ValueError."""
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            calculate_sample_difficulty(50.0, 100.0, 100.0)

    def test_min_greater_than_max_raises(self) -> None:
        """Min greater than max raises ValueError."""
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            calculate_sample_difficulty(50.0, 100.0, 0.0)


class TestCalculateCompetenceScore:
    """Tests for calculate_competence_score function."""

    def test_basic_competence(self) -> None:
        """Basic competence calculation."""
        score = calculate_competence_score((0.5, 0.4, 0.3))
        assert 0 < score < 1

    def test_perfect_competence(self) -> None:
        """Perfect performance gives high competence."""
        score = calculate_competence_score((0.1,), target_loss=0.1)
        assert score == pytest.approx(1.0)

    def test_poor_competence(self) -> None:
        """Poor performance gives low competence."""
        score = calculate_competence_score((10.0,))
        assert score < 0.1

    def test_improving_losses(self) -> None:
        """Improving losses gives reasonable competence."""
        score = calculate_competence_score((1.0, 0.8, 0.6, 0.4))
        assert 0 < score < 1

    def test_empty_losses_raises(self) -> None:
        """Empty losses raises ValueError."""
        with pytest.raises(ValueError, match="recent_losses cannot be empty"):
            calculate_competence_score(())

    def test_zero_target_loss_raises(self) -> None:
        """Zero target_loss raises ValueError."""
        with pytest.raises(ValueError, match="target_loss must be positive"):
            calculate_competence_score((0.5,), target_loss=0)

    def test_negative_target_loss_raises(self) -> None:
        """Negative target_loss raises ValueError."""
        with pytest.raises(ValueError, match="target_loss must be positive"):
            calculate_competence_score((0.5,), target_loss=-0.1)

    def test_invalid_smoothing_raises(self) -> None:
        """Invalid smoothing raises ValueError."""
        with pytest.raises(ValueError, match="smoothing must be in"):
            calculate_competence_score((0.5,), smoothing=0)

    def test_smoothing_above_one_raises(self) -> None:
        """Smoothing > 1 raises ValueError."""
        with pytest.raises(ValueError, match="smoothing must be in"):
            calculate_competence_score((0.5,), smoothing=1.5)


class TestGetDifficultyAtStep:
    """Tests for get_difficulty_at_step function."""

    def test_linear_pacing(self) -> None:
        """Linear pacing increases linearly."""
        config = create_pacing_config(function="linear")
        assert get_difficulty_at_step(config, 0, 1000) == pytest.approx(0.0)
        assert get_difficulty_at_step(config, 500, 1000) == pytest.approx(0.5)
        assert get_difficulty_at_step(config, 1000, 1000) == pytest.approx(1.0)

    def test_exponential_pacing(self) -> None:
        """Exponential pacing increases faster early."""
        config = create_pacing_config(function="exponential")
        mid = get_difficulty_at_step(config, 500, 1000)
        # Exponential pacing should be above 0.5 at midpoint (faster early)
        assert 0.5 < mid < 1.0

    def test_step_pacing(self) -> None:
        """Step pacing increases in discrete steps."""
        config = create_pacing_config(function="step")
        early = get_difficulty_at_step(config, 100, 1000)
        mid = get_difficulty_at_step(config, 500, 1000)
        assert early < mid

    def test_self_paced(self) -> None:
        """Self-paced uses competence score."""
        config = create_pacing_config(function="self_paced")
        low_comp = get_difficulty_at_step(config, 500, 1000, competence_score=0.2)
        high_comp = get_difficulty_at_step(config, 500, 1000, competence_score=0.8)
        assert low_comp < high_comp

    def test_competence_based(self) -> None:
        """Competence-based uses progress and competence."""
        config = create_pacing_config(function="competence")
        low_comp = get_difficulty_at_step(config, 500, 1000, competence_score=0.2)
        high_comp = get_difficulty_at_step(config, 500, 1000, competence_score=0.8)
        assert low_comp < high_comp

    def test_warmup_period(self) -> None:
        """Warmup period returns initial difficulty."""
        config = create_pacing_config(warmup_steps=500, initial_difficulty=0.1)
        assert get_difficulty_at_step(config, 250, 1000) == pytest.approx(0.1)

    def test_negative_step_raises(self) -> None:
        """Negative current_step raises ValueError."""
        config = create_pacing_config()
        with pytest.raises(ValueError, match="current_step must be non-negative"):
            get_difficulty_at_step(config, -1, 1000)

    def test_zero_total_steps_raises(self) -> None:
        """Zero total_steps raises ValueError."""
        config = create_pacing_config()
        with pytest.raises(ValueError, match="total_steps must be positive"):
            get_difficulty_at_step(config, 0, 0)

    def test_step_exceeds_total_raises(self) -> None:
        """Current step exceeds total steps raises ValueError."""
        config = create_pacing_config()
        with pytest.raises(ValueError, match=r"current_step.*cannot exceed"):
            get_difficulty_at_step(config, 1001, 1000)

    def test_warmup_exceeds_total(self) -> None:
        """When warmup >= total, returns target difficulty."""
        config = create_pacing_config(warmup_steps=1000, target_difficulty=0.8)
        assert get_difficulty_at_step(config, 500, 1000) == pytest.approx(0.0)
        assert get_difficulty_at_step(config, 1000, 1000) == pytest.approx(0.8)


class TestCalculateSampleWeights:
    """Tests for calculate_sample_weights function."""

    def test_threshold_strategy(self) -> None:
        """Threshold strategy gives binary weights."""
        difficulties = (0.1, 0.3, 0.5, 0.7, 0.9)
        weights = calculate_sample_weights(
            difficulties, 0.5, SamplingStrategy.THRESHOLD
        )
        assert weights == (1.0, 1.0, 1.0, 0.0, 0.0)

    def test_weighted_strategy(self) -> None:
        """Weighted strategy gives smooth weights."""
        difficulties = (0.1, 0.3, 0.5, 0.7, 0.9)
        weights = calculate_sample_weights(difficulties, 0.5, SamplingStrategy.WEIGHTED)
        assert all(w >= 0 for w in weights)
        assert weights[0] >= weights[-1]

    def test_probabilistic_strategy(self) -> None:
        """Probabilistic strategy gives sigmoid-like weights."""
        difficulties = (0.1, 0.3, 0.5, 0.7, 0.9)
        weights = calculate_sample_weights(
            difficulties, 0.5, SamplingStrategy.PROBABILISTIC
        )
        assert all(0 < w < 1 for w in weights)
        assert weights[0] > weights[-1]

    def test_temperature_affects_weights(self) -> None:
        """Higher temperature smooths weights."""
        difficulties = (0.1, 0.5, 0.9)
        weights_low = calculate_sample_weights(
            difficulties, 0.5, SamplingStrategy.WEIGHTED, temperature=0.1
        )
        weights_high = calculate_sample_weights(
            difficulties, 0.5, SamplingStrategy.WEIGHTED, temperature=1.0
        )
        assert weights_low[-1] < weights_high[-1]

    def test_empty_difficulties_raises(self) -> None:
        """Empty difficulties raises ValueError."""
        with pytest.raises(ValueError, match="difficulties cannot be empty"):
            calculate_sample_weights((), 0.5, SamplingStrategy.THRESHOLD)

    def test_invalid_current_difficulty_raises(self) -> None:
        """Invalid current_difficulty raises ValueError."""
        with pytest.raises(ValueError, match="current_difficulty must be between"):
            calculate_sample_weights((0.1, 0.5), 1.5, SamplingStrategy.THRESHOLD)

    def test_negative_current_difficulty_raises(self) -> None:
        """Negative current_difficulty raises ValueError."""
        with pytest.raises(ValueError, match="current_difficulty must be between"):
            calculate_sample_weights((0.1, 0.5), -0.1, SamplingStrategy.THRESHOLD)

    def test_zero_temperature_raises(self) -> None:
        """Zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            calculate_sample_weights(
                (0.1, 0.5), 0.5, SamplingStrategy.WEIGHTED, temperature=0
            )

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            calculate_sample_weights(
                (0.1, 0.5), 0.5, SamplingStrategy.WEIGHTED, temperature=-1.0
            )


class TestFormatCurriculumStats:
    """Tests for format_curriculum_stats function."""

    def test_basic_format(self) -> None:
        """Format basic stats."""
        stats = create_curriculum_stats(
            current_difficulty=0.5,
            samples_seen=10000,
            competence_score=0.8,
            curriculum_progress=0.5,
        )
        formatted = format_curriculum_stats(stats)
        assert "Difficulty: 50.0%" in formatted
        assert "Samples: 10000" in formatted
        assert "Competence: 80.0%" in formatted
        assert "Progress: 50.0%" in formatted

    def test_contains_all_fields(self) -> None:
        """Formatted string contains all fields."""
        stats = create_curriculum_stats()
        formatted = format_curriculum_stats(stats)
        assert "Difficulty:" in formatted
        assert "Samples:" in formatted
        assert "Competence:" in formatted
        assert "Progress:" in formatted


class TestGetRecommendedCurriculumConfig:
    """Tests for get_recommended_curriculum_config function."""

    def test_classification_config(self) -> None:
        """Get config for classification task."""
        config = get_recommended_curriculum_config("classification")
        assert config.difficulty_config.metric == DifficultyMetric.CONFIDENCE
        assert config.pacing_config.function == PacingFunction.LINEAR
        assert config.sampling_strategy == SamplingStrategy.THRESHOLD

    def test_generation_config(self) -> None:
        """Get config for generation task."""
        config = get_recommended_curriculum_config("generation")
        assert config.difficulty_config.metric == DifficultyMetric.LENGTH
        assert config.pacing_config.function == PacingFunction.EXPONENTIAL
        assert config.sampling_strategy == SamplingStrategy.WEIGHTED

    def test_translation_config(self) -> None:
        """Get config for translation task."""
        config = get_recommended_curriculum_config("translation")
        assert config.difficulty_config.metric == DifficultyMetric.LENGTH
        assert config.pacing_config.function == PacingFunction.COMPETENCE
        assert config.sampling_strategy == SamplingStrategy.PROBABILISTIC

    def test_summarization_config(self) -> None:
        """Get config for summarization task."""
        config = get_recommended_curriculum_config("summarization")
        assert config.difficulty_config.metric == DifficultyMetric.PERPLEXITY
        assert config.pacing_config.function == PacingFunction.SELF_PACED
        assert config.sampling_strategy == SamplingStrategy.WEIGHTED

    def test_qa_config(self) -> None:
        """Get config for QA task."""
        config = get_recommended_curriculum_config("qa")
        assert config.difficulty_config.metric == DifficultyMetric.LOSS
        assert config.pacing_config.function == PacingFunction.STEP
        assert config.sampling_strategy == SamplingStrategy.THRESHOLD

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_curriculum_config("unknown")

    @pytest.mark.parametrize(
        "task",
        ["classification", "generation", "translation", "summarization", "qa"],
    )
    def test_all_tasks_return_valid_config(self, task: str) -> None:
        """All supported tasks return valid configs."""
        config = get_recommended_curriculum_config(task)
        validate_curriculum_config(config)
