"""Tests for synthetic data generation functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.synthetic import (
    VALID_DIVERSITY_STRATEGIES,
    VALID_GENERATION_METHODS,
    VALID_QUALITY_FILTERS,
    DiversityStrategy,
    EvolInstructConfig,
    GenerationMethod,
    GenerationStats,
    QualityFilter,
    SelfInstructConfig,
    SyntheticConfig,
    SyntheticSample,
    calculate_diversity_score,
    compute_generation_stats,
    create_evol_instruct_config,
    create_self_instruct_config,
    create_synthetic_config,
    deduplicate_samples,
    estimate_generation_cost,
    filter_synthetic_samples,
    format_generation_stats,
    get_diversity_strategy,
    get_generation_method,
    get_quality_filter,
    get_recommended_synthetic_config,
    list_diversity_strategies,
    list_generation_methods,
    list_quality_filters,
    validate_evol_instruct_config,
    validate_self_instruct_config,
    validate_synthetic_config,
    validate_synthetic_quality,
)


class TestGenerationMethod:
    """Tests for GenerationMethod enum."""

    def test_self_instruct_value(self) -> None:
        """Test SELF_INSTRUCT value."""
        assert GenerationMethod.SELF_INSTRUCT.value == "self_instruct"

    def test_evol_instruct_value(self) -> None:
        """Test EVOL_INSTRUCT value."""
        assert GenerationMethod.EVOL_INSTRUCT.value == "evol_instruct"

    def test_backtranslation_value(self) -> None:
        """Test BACKTRANSLATION value."""
        assert GenerationMethod.BACKTRANSLATION.value == "backtranslation"

    def test_paraphrase_value(self) -> None:
        """Test PARAPHRASE value."""
        assert GenerationMethod.PARAPHRASE.value == "paraphrase"

    def test_template_value(self) -> None:
        """Test TEMPLATE value."""
        assert GenerationMethod.TEMPLATE.value == "template"


class TestDiversityStrategy:
    """Tests for DiversityStrategy enum."""

    def test_topic_value(self) -> None:
        """Test TOPIC value."""
        assert DiversityStrategy.TOPIC.value == "topic"

    def test_style_value(self) -> None:
        """Test STYLE value."""
        assert DiversityStrategy.STYLE.value == "style"

    def test_difficulty_value(self) -> None:
        """Test DIFFICULTY value."""
        assert DiversityStrategy.DIFFICULTY.value == "difficulty"

    def test_format_value(self) -> None:
        """Test FORMAT value."""
        assert DiversityStrategy.FORMAT.value == "format"


class TestQualityFilter:
    """Tests for QualityFilter enum."""

    def test_perplexity_value(self) -> None:
        """Test PERPLEXITY value."""
        assert QualityFilter.PERPLEXITY.value == "perplexity"

    def test_length_value(self) -> None:
        """Test LENGTH value."""
        assert QualityFilter.LENGTH.value == "length"

    def test_similarity_value(self) -> None:
        """Test SIMILARITY value."""
        assert QualityFilter.SIMILARITY.value == "similarity"

    def test_toxicity_value(self) -> None:
        """Test TOXICITY value."""
        assert QualityFilter.TOXICITY.value == "toxicity"


class TestSelfInstructConfig:
    """Tests for SelfInstructConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SelfInstructConfig instance."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=("task1", "task2"),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=3,
        )
        assert config.num_instructions == 100
        assert len(config.seed_tasks) == 2
        assert config.diversity_strategy == DiversityStrategy.TOPIC
        assert config.temperature == pytest.approx(0.7)
        assert config.max_retries == 3

    def test_frozen(self) -> None:
        """Test that SelfInstructConfig is immutable."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=3,
        )
        with pytest.raises(AttributeError):
            config.num_instructions = 200  # type: ignore[misc]


class TestEvolInstructConfig:
    """Tests for EvolInstructConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating EvolInstructConfig instance."""
        config = EvolInstructConfig(
            evolution_steps=3,
            mutation_types=("add_constraint", "increase_depth"),
            complexity_increase=1.2,
            preserve_semantics=True,
            max_length=512,
        )
        assert config.evolution_steps == 3
        assert len(config.mutation_types) == 2
        assert config.complexity_increase == pytest.approx(1.2)
        assert config.preserve_semantics is True
        assert config.max_length == 512

    def test_frozen(self) -> None:
        """Test that EvolInstructConfig is immutable."""
        config = EvolInstructConfig(
            evolution_steps=3,
            mutation_types=("add_constraint",),
            complexity_increase=1.2,
            preserve_semantics=True,
            max_length=512,
        )
        with pytest.raises(AttributeError):
            config.evolution_steps = 5  # type: ignore[misc]


class TestSyntheticConfig:
    """Tests for SyntheticConfig dataclass."""

    def test_creation(self) -> None:
        """Test creating SyntheticConfig instance."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.PERPLEXITY, QualityFilter.LENGTH),
            dedup_threshold=0.9,
            target_count=1000,
            min_length=10,
            max_length=512,
        )
        assert config.method == GenerationMethod.SELF_INSTRUCT
        assert len(config.quality_filters) == 2
        assert config.dedup_threshold == pytest.approx(0.9)
        assert config.target_count == 1000
        assert config.min_length == 10
        assert config.max_length == 512

    def test_frozen(self) -> None:
        """Test that SyntheticConfig is immutable."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.LENGTH,),
            dedup_threshold=0.9,
            target_count=1000,
            min_length=10,
            max_length=512,
        )
        with pytest.raises(AttributeError):
            config.target_count = 2000  # type: ignore[misc]


class TestSyntheticSample:
    """Tests for SyntheticSample dataclass."""

    def test_creation(self) -> None:
        """Test creating SyntheticSample instance."""
        sample = SyntheticSample(
            text="Test sample text",
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.85,
            diversity_score=0.72,
        )
        assert sample.text == "Test sample text"
        assert sample.source_method == GenerationMethod.SELF_INSTRUCT
        assert sample.quality_score == pytest.approx(0.85)
        assert sample.diversity_score == pytest.approx(0.72)

    def test_frozen(self) -> None:
        """Test that SyntheticSample is immutable."""
        sample = SyntheticSample(
            text="Test",
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.8,
            diversity_score=0.7,
        )
        with pytest.raises(AttributeError):
            sample.quality_score = 0.9  # type: ignore[misc]


class TestGenerationStats:
    """Tests for GenerationStats dataclass."""

    def test_creation(self) -> None:
        """Test creating GenerationStats instance."""
        stats = GenerationStats(
            total_generated=1500,
            passed_quality=1200,
            passed_dedup=1000,
            final_count=1000,
            avg_quality_score=0.82,
            avg_diversity_score=0.75,
        )
        assert stats.total_generated == 1500
        assert stats.passed_quality == 1200
        assert stats.passed_dedup == 1000
        assert stats.final_count == 1000
        assert stats.avg_quality_score == pytest.approx(0.82)
        assert stats.avg_diversity_score == pytest.approx(0.75)

    def test_frozen(self) -> None:
        """Test that GenerationStats is immutable."""
        stats = GenerationStats(
            total_generated=1500,
            passed_quality=1200,
            passed_dedup=1000,
            final_count=1000,
            avg_quality_score=0.82,
            avg_diversity_score=0.75,
        )
        with pytest.raises(AttributeError):
            stats.total_generated = 2000  # type: ignore[misc]


class TestValidateSelfInstructConfig:
    """Tests for validate_self_instruct_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=3,
        )
        validate_self_instruct_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_self_instruct_config(None)  # type: ignore[arg-type]

    def test_zero_instructions_raises_error(self) -> None:
        """Test that zero instructions raises ValueError."""
        config = SelfInstructConfig(
            num_instructions=0,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=3,
        )
        with pytest.raises(ValueError, match="num_instructions must be positive"):
            validate_self_instruct_config(config)

    def test_negative_instructions_raises_error(self) -> None:
        """Test that negative instructions raises ValueError."""
        config = SelfInstructConfig(
            num_instructions=-10,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=3,
        )
        with pytest.raises(ValueError, match="num_instructions must be positive"):
            validate_self_instruct_config(config)

    def test_empty_seed_tasks_raises_error(self) -> None:
        """Test that empty seed_tasks raises ValueError."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=(),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=3,
        )
        with pytest.raises(ValueError, match="seed_tasks cannot be empty"):
            validate_self_instruct_config(config)

    def test_temperature_above_two_raises_error(self) -> None:
        """Test that temperature above 2 raises ValueError."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=2.5,
            max_retries=3,
        )
        with pytest.raises(ValueError, match="temperature must be between"):
            validate_self_instruct_config(config)

    def test_temperature_below_zero_raises_error(self) -> None:
        """Test that temperature below 0 raises ValueError."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=-0.5,
            max_retries=3,
        )
        with pytest.raises(ValueError, match="temperature must be between"):
            validate_self_instruct_config(config)

    def test_zero_max_retries_raises_error(self) -> None:
        """Test that zero max_retries raises ValueError."""
        config = SelfInstructConfig(
            num_instructions=100,
            seed_tasks=("task1",),
            diversity_strategy=DiversityStrategy.TOPIC,
            temperature=0.7,
            max_retries=0,
        )
        with pytest.raises(ValueError, match="max_retries must be positive"):
            validate_self_instruct_config(config)


class TestValidateEvolInstructConfig:
    """Tests for validate_evol_instruct_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = EvolInstructConfig(
            evolution_steps=3,
            mutation_types=("add_constraint",),
            complexity_increase=1.2,
            preserve_semantics=True,
            max_length=512,
        )
        validate_evol_instruct_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_evol_instruct_config(None)  # type: ignore[arg-type]

    def test_zero_evolution_steps_raises_error(self) -> None:
        """Test that zero evolution_steps raises ValueError."""
        config = EvolInstructConfig(
            evolution_steps=0,
            mutation_types=("add_constraint",),
            complexity_increase=1.2,
            preserve_semantics=True,
            max_length=512,
        )
        with pytest.raises(ValueError, match="evolution_steps must be positive"):
            validate_evol_instruct_config(config)

    def test_empty_mutation_types_raises_error(self) -> None:
        """Test that empty mutation_types raises ValueError."""
        config = EvolInstructConfig(
            evolution_steps=3,
            mutation_types=(),
            complexity_increase=1.2,
            preserve_semantics=True,
            max_length=512,
        )
        with pytest.raises(ValueError, match="mutation_types cannot be empty"):
            validate_evol_instruct_config(config)

    def test_complexity_below_one_raises_error(self) -> None:
        """Test that complexity_increase below 1.0 raises ValueError."""
        config = EvolInstructConfig(
            evolution_steps=3,
            mutation_types=("add_constraint",),
            complexity_increase=0.5,
            preserve_semantics=True,
            max_length=512,
        )
        with pytest.raises(ValueError, match=r"complexity_increase must be >= 1\.0"):
            validate_evol_instruct_config(config)

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        config = EvolInstructConfig(
            evolution_steps=3,
            mutation_types=("add_constraint",),
            complexity_increase=1.2,
            preserve_semantics=True,
            max_length=0,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_evol_instruct_config(config)


class TestValidateSyntheticConfig:
    """Tests for validate_synthetic_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.PERPLEXITY,),
            dedup_threshold=0.9,
            target_count=1000,
            min_length=10,
            max_length=512,
        )
        validate_synthetic_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_synthetic_config(None)  # type: ignore[arg-type]

    def test_empty_quality_filters_raises_error(self) -> None:
        """Test that empty quality_filters raises ValueError."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(),
            dedup_threshold=0.9,
            target_count=1000,
            min_length=10,
            max_length=512,
        )
        with pytest.raises(ValueError, match="quality_filters cannot be empty"):
            validate_synthetic_config(config)

    def test_dedup_threshold_above_one_raises_error(self) -> None:
        """Test that dedup_threshold above 1 raises ValueError."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.LENGTH,),
            dedup_threshold=1.5,
            target_count=1000,
            min_length=10,
            max_length=512,
        )
        with pytest.raises(ValueError, match="dedup_threshold must be between"):
            validate_synthetic_config(config)

    def test_dedup_threshold_below_zero_raises_error(self) -> None:
        """Test that dedup_threshold below 0 raises ValueError."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.LENGTH,),
            dedup_threshold=-0.1,
            target_count=1000,
            min_length=10,
            max_length=512,
        )
        with pytest.raises(ValueError, match="dedup_threshold must be between"):
            validate_synthetic_config(config)

    def test_zero_target_count_raises_error(self) -> None:
        """Test that zero target_count raises ValueError."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.LENGTH,),
            dedup_threshold=0.9,
            target_count=0,
            min_length=10,
            max_length=512,
        )
        with pytest.raises(ValueError, match="target_count must be positive"):
            validate_synthetic_config(config)

    def test_min_length_gte_max_length_raises_error(self) -> None:
        """Test that min_length >= max_length raises ValueError."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.LENGTH,),
            dedup_threshold=0.9,
            target_count=1000,
            min_length=512,
            max_length=100,
        )
        with pytest.raises(ValueError, match=r"min_length .* must be less than"):
            validate_synthetic_config(config)

    def test_min_length_equals_max_length_raises_error(self) -> None:
        """Test that min_length == max_length raises ValueError."""
        config = SyntheticConfig(
            method=GenerationMethod.SELF_INSTRUCT,
            quality_filters=(QualityFilter.LENGTH,),
            dedup_threshold=0.9,
            target_count=1000,
            min_length=100,
            max_length=100,
        )
        with pytest.raises(ValueError, match=r"min_length .* must be less than"):
            validate_synthetic_config(config)


class TestCreateSelfInstructConfig:
    """Tests for create_self_instruct_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_self_instruct_config()
        assert config.num_instructions == 100
        assert config.diversity_strategy == DiversityStrategy.TOPIC
        assert config.temperature == pytest.approx(0.7)
        assert config.max_retries == 3
        assert len(config.seed_tasks) > 0

    def test_custom_instructions(self) -> None:
        """Test creating config with custom num_instructions."""
        config = create_self_instruct_config(num_instructions=50)
        assert config.num_instructions == 50

    def test_custom_diversity_strategy(self) -> None:
        """Test creating config with custom diversity strategy."""
        config = create_self_instruct_config(diversity_strategy="style")
        assert config.diversity_strategy == DiversityStrategy.STYLE

    def test_invalid_diversity_strategy_raises_error(self) -> None:
        """Test that invalid diversity_strategy raises ValueError."""
        with pytest.raises(ValueError, match="diversity_strategy must be one of"):
            create_self_instruct_config(diversity_strategy="invalid")

    def test_invalid_num_instructions_raises_error(self) -> None:
        """Test that invalid num_instructions raises ValueError."""
        with pytest.raises(ValueError, match="num_instructions must be positive"):
            create_self_instruct_config(num_instructions=0)

    def test_custom_seed_tasks(self) -> None:
        """Test creating config with custom seed tasks."""
        config = create_self_instruct_config(seed_tasks=("custom task",))
        assert config.seed_tasks == ("custom task",)


class TestCreateEvolInstructConfig:
    """Tests for create_evol_instruct_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_evol_instruct_config()
        assert config.evolution_steps == 3
        assert config.complexity_increase == pytest.approx(1.2)
        assert config.preserve_semantics is True
        assert config.max_length == 512
        assert len(config.mutation_types) > 0

    def test_custom_evolution_steps(self) -> None:
        """Test creating config with custom evolution steps."""
        config = create_evol_instruct_config(evolution_steps=5)
        assert config.evolution_steps == 5

    def test_invalid_evolution_steps_raises_error(self) -> None:
        """Test that invalid evolution_steps raises ValueError."""
        with pytest.raises(ValueError, match="evolution_steps must be positive"):
            create_evol_instruct_config(evolution_steps=0)

    def test_invalid_complexity_increase_raises_error(self) -> None:
        """Test that invalid complexity_increase raises ValueError."""
        with pytest.raises(ValueError, match=r"complexity_increase must be >= 1\.0"):
            create_evol_instruct_config(complexity_increase=0.5)

    def test_custom_mutation_types(self) -> None:
        """Test creating config with custom mutation types."""
        config = create_evol_instruct_config(mutation_types=("custom",))
        assert config.mutation_types == ("custom",)


class TestCreateSyntheticConfig:
    """Tests for create_synthetic_config function."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = create_synthetic_config()
        assert config.method == GenerationMethod.SELF_INSTRUCT
        assert config.dedup_threshold == pytest.approx(0.9)
        assert config.target_count == 1000
        assert config.min_length == 10
        assert config.max_length == 512

    def test_custom_method(self) -> None:
        """Test creating config with custom method."""
        config = create_synthetic_config(method="evol_instruct")
        assert config.method == GenerationMethod.EVOL_INSTRUCT

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_synthetic_config(method="invalid")

    def test_invalid_quality_filter_raises_error(self) -> None:
        """Test that invalid quality_filter raises ValueError."""
        with pytest.raises(ValueError, match="quality_filter must be one of"):
            create_synthetic_config(quality_filters=("invalid",))

    def test_custom_quality_filters(self) -> None:
        """Test creating config with custom quality filters."""
        config = create_synthetic_config(quality_filters=("similarity", "toxicity"))
        assert QualityFilter.SIMILARITY in config.quality_filters
        assert QualityFilter.TOXICITY in config.quality_filters


class TestListGenerationMethods:
    """Tests for list_generation_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_generation_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_generation_methods()
        assert "self_instruct" in methods
        assert "evol_instruct" in methods
        assert "backtranslation" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_generation_methods()
        assert methods == sorted(methods)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_GENERATION_METHODS."""
        methods = list_generation_methods()
        assert set(methods) == VALID_GENERATION_METHODS


class TestGetGenerationMethod:
    """Tests for get_generation_method function."""

    def test_get_self_instruct(self) -> None:
        """Test getting SELF_INSTRUCT method."""
        result = get_generation_method("self_instruct")
        assert result == GenerationMethod.SELF_INSTRUCT

    def test_get_evol_instruct(self) -> None:
        """Test getting EVOL_INSTRUCT method."""
        result = get_generation_method("evol_instruct")
        assert result == GenerationMethod.EVOL_INSTRUCT

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid generation method"):
            get_generation_method("invalid")


class TestListDiversityStrategies:
    """Tests for list_diversity_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_diversity_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_diversity_strategies()
        assert "topic" in strategies
        assert "style" in strategies
        assert "difficulty" in strategies

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_diversity_strategies()
        assert strategies == sorted(strategies)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_DIVERSITY_STRATEGIES."""
        strategies = list_diversity_strategies()
        assert set(strategies) == VALID_DIVERSITY_STRATEGIES


class TestGetDiversityStrategy:
    """Tests for get_diversity_strategy function."""

    def test_get_topic(self) -> None:
        """Test getting TOPIC strategy."""
        result = get_diversity_strategy("topic")
        assert result == DiversityStrategy.TOPIC

    def test_get_style(self) -> None:
        """Test getting STYLE strategy."""
        result = get_diversity_strategy("style")
        assert result == DiversityStrategy.STYLE

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="invalid diversity strategy"):
            get_diversity_strategy("invalid")


class TestListQualityFilters:
    """Tests for list_quality_filters function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        filters = list_quality_filters()
        assert isinstance(filters, list)

    def test_contains_expected_filters(self) -> None:
        """Test that list contains expected filters."""
        filters = list_quality_filters()
        assert "perplexity" in filters
        assert "length" in filters
        assert "similarity" in filters

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        filters = list_quality_filters()
        assert filters == sorted(filters)

    def test_matches_valid_set(self) -> None:
        """Test that list matches VALID_QUALITY_FILTERS."""
        filters = list_quality_filters()
        assert set(filters) == VALID_QUALITY_FILTERS


class TestGetQualityFilter:
    """Tests for get_quality_filter function."""

    def test_get_perplexity(self) -> None:
        """Test getting PERPLEXITY filter."""
        result = get_quality_filter("perplexity")
        assert result == QualityFilter.PERPLEXITY

    def test_get_similarity(self) -> None:
        """Test getting SIMILARITY filter."""
        result = get_quality_filter("similarity")
        assert result == QualityFilter.SIMILARITY

    def test_invalid_filter_raises_error(self) -> None:
        """Test that invalid filter raises ValueError."""
        with pytest.raises(ValueError, match="invalid quality filter"):
            get_quality_filter("invalid")


class TestEstimateGenerationCost:
    """Tests for estimate_generation_cost function."""

    def test_basic_cost_estimate(self) -> None:
        """Test basic cost estimation."""
        config = create_synthetic_config(target_count=1000)
        costs = estimate_generation_cost(config)
        assert costs["total_tokens"] > 0
        assert costs["total_cost"] >= 0
        assert costs["cost_per_sample"] >= 0
        assert costs["estimated_samples_needed"] > 0

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            estimate_generation_cost(None)  # type: ignore[arg-type]

    def test_zero_tokens_per_sample_raises_error(self) -> None:
        """Test that zero tokens_per_sample raises ValueError."""
        config = create_synthetic_config()
        with pytest.raises(ValueError, match="tokens_per_sample must be positive"):
            estimate_generation_cost(config, tokens_per_sample=0)

    def test_negative_cost_per_token_raises_error(self) -> None:
        """Test that negative cost_per_1k_tokens raises ValueError."""
        config = create_synthetic_config()
        with pytest.raises(ValueError, match="cost_per_1k_tokens cannot be negative"):
            estimate_generation_cost(config, cost_per_1k_tokens=-0.001)

    def test_different_methods_have_different_costs(self) -> None:
        """Test that different methods have different cost estimates."""
        config_self = create_synthetic_config(method="self_instruct")
        config_evol = create_synthetic_config(method="evol_instruct")

        cost_self = estimate_generation_cost(config_self)
        cost_evol = estimate_generation_cost(config_evol)

        # evol_instruct has higher overhead
        assert cost_evol["total_cost"] > cost_self["total_cost"]


class TestCalculateDiversityScore:
    """Tests for calculate_diversity_score function."""

    def test_diverse_texts(self) -> None:
        """Test with diverse texts."""
        texts = ["Hello world", "Goodbye moon", "Testing here now"]
        score = calculate_diversity_score(texts)
        assert 0.0 <= score <= 1.0

    def test_identical_texts(self) -> None:
        """Test with identical texts."""
        texts = ["same text here", "same text here", "same text here"]
        score = calculate_diversity_score(texts)
        assert score < 0.5

    def test_empty_texts(self) -> None:
        """Test with empty texts list."""
        score = calculate_diversity_score([])
        assert score == 0.0

    def test_single_text(self) -> None:
        """Test with single text."""
        score = calculate_diversity_score(["only one"])
        assert score == 0.0

    def test_none_texts_raises_error(self) -> None:
        """Test that None texts raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be None"):
            calculate_diversity_score(None)  # type: ignore[arg-type]

    def test_zero_ngram_size_raises_error(self) -> None:
        """Test that zero ngram_size raises ValueError."""
        with pytest.raises(ValueError, match="ngram_size must be positive"):
            calculate_diversity_score(["test"], ngram_size=0)

    def test_negative_ngram_size_raises_error(self) -> None:
        """Test that negative ngram_size raises ValueError."""
        with pytest.raises(ValueError, match="ngram_size must be positive"):
            calculate_diversity_score(["test"], ngram_size=-1)

    def test_texts_without_words(self) -> None:
        """Test with texts that have no words after splitting."""
        texts = ["", "", ""]
        score = calculate_diversity_score(texts)
        assert score == 0.0


class TestFilterSyntheticSamples:
    """Tests for filter_synthetic_samples function."""

    def test_filter_by_length(self) -> None:
        """Test filtering samples by length."""
        sample1 = SyntheticSample(
            text="This is a good sample with enough length for filtering",
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.9,
            diversity_score=0.8,
        )
        sample2 = SyntheticSample(
            text="Short",
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.9,
            diversity_score=0.8,
        )
        config = create_synthetic_config(min_length=10, max_length=100)
        filtered = filter_synthetic_samples([sample1, sample2], config)
        assert len(filtered) == 1
        assert filtered[0].text == sample1.text

    def test_filter_by_quality_score(self) -> None:
        """Test filtering samples by quality score."""
        sample1 = SyntheticSample(
            text="A good quality sample text here",
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.9,
            diversity_score=0.8,
        )
        sample2 = SyntheticSample(
            text="A low quality sample text here",
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.3,
            diversity_score=0.8,
        )
        config = create_synthetic_config(
            quality_filters=("perplexity",),
            min_length=5,
            max_length=100,
        )
        filtered = filter_synthetic_samples([sample1, sample2], config)
        assert len(filtered) == 1
        assert filtered[0].quality_score == pytest.approx(0.9)

    def test_none_samples_raises_error(self) -> None:
        """Test that None samples raises ValueError."""
        config = create_synthetic_config()
        with pytest.raises(ValueError, match="samples cannot be None"):
            filter_synthetic_samples(None, config)  # type: ignore[arg-type]

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            filter_synthetic_samples([], None)  # type: ignore[arg-type]

    def test_empty_samples(self) -> None:
        """Test with empty samples list."""
        config = create_synthetic_config()
        filtered = filter_synthetic_samples([], config)
        assert filtered == []

    def test_filter_max_length(self) -> None:
        """Test filtering by max length."""
        sample = SyntheticSample(
            text="x" * 1000,  # Very long sample
            source_method=GenerationMethod.SELF_INSTRUCT,
            quality_score=0.9,
            diversity_score=0.8,
        )
        config = create_synthetic_config(min_length=5, max_length=100)
        filtered = filter_synthetic_samples([sample], config)
        assert len(filtered) == 0


class TestValidateSyntheticQuality:
    """Tests for validate_synthetic_quality function."""

    def test_valid_samples(self) -> None:
        """Test validation with valid samples."""
        samples = [
            SyntheticSample(
                text="Good sample",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.8,
                diversity_score=0.6,
            ),
            SyntheticSample(
                text="Another good sample",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.7,
                diversity_score=0.5,
            ),
        ]
        result = validate_synthetic_quality(samples)
        assert result["is_valid"] is True
        assert result["avg_quality"] == pytest.approx(0.75)
        assert result["avg_diversity"] == pytest.approx(0.55)
        assert result["issues"] == []

    def test_low_quality_samples(self) -> None:
        """Test validation with low quality samples."""
        samples = [
            SyntheticSample(
                text="Low quality",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.2,
                diversity_score=0.1,
            ),
        ]
        result = validate_synthetic_quality(samples)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_none_samples_raises_error(self) -> None:
        """Test that None samples raises ValueError."""
        with pytest.raises(ValueError, match="samples cannot be None"):
            validate_synthetic_quality(None)  # type: ignore[arg-type]

    def test_empty_samples(self) -> None:
        """Test with empty samples list."""
        result = validate_synthetic_quality([])
        assert result["is_valid"] is False
        assert "No samples provided" in result["issues"]

    def test_invalid_min_quality_score_raises_error(self) -> None:
        """Test that invalid min_quality_score raises ValueError."""
        with pytest.raises(ValueError, match="min_quality_score must be between"):
            validate_synthetic_quality([], min_quality_score=1.5)

    def test_invalid_min_diversity_score_raises_error(self) -> None:
        """Test that invalid min_diversity_score raises ValueError."""
        with pytest.raises(ValueError, match="min_diversity_score must be between"):
            validate_synthetic_quality([], min_diversity_score=-0.1)

    def test_many_low_quality_samples_reported(self) -> None:
        """Test that many low quality samples are reported."""
        samples = [
            SyntheticSample(
                text=f"Sample {i}",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.3 if i < 5 else 0.9,
                diversity_score=0.5,
            )
            for i in range(10)
        ]
        result = validate_synthetic_quality(samples, min_quality_score=0.5)
        # 50% are below threshold, should be flagged
        issues_str = " ".join(result["issues"])
        assert "below quality threshold" in issues_str


class TestFormatGenerationStats:
    """Tests for format_generation_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = GenerationStats(
            total_generated=1500,
            passed_quality=1200,
            passed_dedup=1000,
            final_count=1000,
            avg_quality_score=0.82,
            avg_diversity_score=0.75,
        )
        formatted = format_generation_stats(stats)
        assert "1,500" in formatted or "1500" in formatted
        assert "1,000" in formatted or "1000" in formatted
        assert "0.82" in formatted
        assert "0.75" in formatted

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_generation_stats(None)  # type: ignore[arg-type]

    def test_zero_values(self) -> None:
        """Test with zero values."""
        stats = GenerationStats(
            total_generated=0,
            passed_quality=0,
            passed_dedup=0,
            final_count=0,
            avg_quality_score=0.0,
            avg_diversity_score=0.0,
        )
        formatted = format_generation_stats(stats)
        assert "0" in formatted

    def test_contains_section_headers(self) -> None:
        """Test that formatted output contains section headers."""
        stats = GenerationStats(
            total_generated=100,
            passed_quality=90,
            passed_dedup=85,
            final_count=85,
            avg_quality_score=0.8,
            avg_diversity_score=0.7,
        )
        formatted = format_generation_stats(stats)
        assert "Synthetic Generation Statistics" in formatted
        assert "Quality pass rate" in formatted


class TestGetRecommendedSyntheticConfig:
    """Tests for get_recommended_synthetic_config function."""

    def test_instruction_tuning(self) -> None:
        """Test recommendation for instruction tuning."""
        config = get_recommended_synthetic_config("instruction_tuning")
        assert config.method == GenerationMethod.SELF_INSTRUCT
        assert QualityFilter.SIMILARITY in config.quality_filters

    def test_qa_generation(self) -> None:
        """Test recommendation for QA generation."""
        config = get_recommended_synthetic_config("qa_generation")
        assert config.method == GenerationMethod.TEMPLATE

    def test_summarization(self) -> None:
        """Test recommendation for summarization."""
        config = get_recommended_synthetic_config("summarization")
        assert config.method == GenerationMethod.BACKTRANSLATION

    def test_code_generation(self) -> None:
        """Test recommendation for code generation."""
        config = get_recommended_synthetic_config("code_generation")
        assert config.method == GenerationMethod.EVOL_INSTRUCT

    def test_invalid_use_case_raises_error(self) -> None:
        """Test that invalid use_case raises ValueError."""
        with pytest.raises(ValueError, match="use_case must be one of"):
            get_recommended_synthetic_config("invalid")

    def test_custom_target_count(self) -> None:
        """Test with custom target count."""
        config = get_recommended_synthetic_config(
            "instruction_tuning",
            target_count=5000,
        )
        assert config.target_count == 5000

    def test_zero_target_count_raises_error(self) -> None:
        """Test that zero target_count raises ValueError."""
        with pytest.raises(ValueError, match="target_count must be positive"):
            get_recommended_synthetic_config("instruction_tuning", target_count=0)


class TestDeduplicateSamples:
    """Tests for deduplicate_samples function."""

    def test_exact_duplicates(self) -> None:
        """Test removal of exact duplicates."""
        samples = [
            SyntheticSample(
                text="Hello world",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.8,
                diversity_score=0.7,
            ),
            SyntheticSample(
                text="Hello world",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.75,
                diversity_score=0.6,
            ),
            SyntheticSample(
                text="Different text",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.8,
                diversity_score=0.7,
            ),
        ]
        deduped = deduplicate_samples(samples)
        assert len(deduped) == 2

    def test_similar_texts_with_low_threshold(self) -> None:
        """Test deduplication with similarity threshold."""
        samples = [
            SyntheticSample(
                text="The quick brown fox jumps over",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.8,
                diversity_score=0.7,
            ),
            SyntheticSample(
                text="The quick brown fox leaps over",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.75,
                diversity_score=0.6,
            ),
        ]
        deduped = deduplicate_samples(samples, threshold=0.5)
        # With low threshold, similar texts should be deduped
        assert len(deduped) == 1

    def test_none_samples_raises_error(self) -> None:
        """Test that None samples raises ValueError."""
        with pytest.raises(ValueError, match="samples cannot be None"):
            deduplicate_samples(None)  # type: ignore[arg-type]

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            deduplicate_samples([], threshold=1.5)

    def test_negative_threshold_raises_error(self) -> None:
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            deduplicate_samples([], threshold=-0.1)

    def test_empty_samples(self) -> None:
        """Test with empty samples list."""
        deduped = deduplicate_samples([])
        assert deduped == []

    def test_threshold_one_keeps_unique_only(self) -> None:
        """Test with threshold=1.0 keeps only hash-unique texts."""
        samples = [
            SyntheticSample(
                text="Text one",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.8,
                diversity_score=0.7,
            ),
            SyntheticSample(
                text="Text two",
                source_method=GenerationMethod.SELF_INSTRUCT,
                quality_score=0.8,
                diversity_score=0.7,
            ),
        ]
        deduped = deduplicate_samples(samples, threshold=1.0)
        assert len(deduped) == 2


class TestComputeGenerationStats:
    """Tests for compute_generation_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic stats computation."""
        gen = [
            SyntheticSample("a", GenerationMethod.SELF_INSTRUCT, 0.8, 0.7),
            SyntheticSample("b", GenerationMethod.SELF_INSTRUCT, 0.6, 0.5),
            SyntheticSample("c", GenerationMethod.SELF_INSTRUCT, 0.4, 0.3),
        ]
        qual = [gen[0], gen[1]]
        dedup = [gen[0]]

        stats = compute_generation_stats(gen, qual, dedup)
        assert stats.total_generated == 3
        assert stats.passed_quality == 2
        assert stats.passed_dedup == 1
        assert stats.final_count == 1
        assert stats.avg_quality_score == pytest.approx(0.8)
        assert stats.avg_diversity_score == pytest.approx(0.7)

    def test_none_generated_raises_error(self) -> None:
        """Test that None generated raises ValueError."""
        with pytest.raises(ValueError, match="generated cannot be None"):
            compute_generation_stats(None, [], [])  # type: ignore[arg-type]

    def test_none_after_quality_raises_error(self) -> None:
        """Test that None after_quality raises ValueError."""
        with pytest.raises(ValueError, match="after_quality cannot be None"):
            compute_generation_stats([], None, [])  # type: ignore[arg-type]

    def test_none_after_dedup_raises_error(self) -> None:
        """Test that None after_dedup raises ValueError."""
        with pytest.raises(ValueError, match="after_dedup cannot be None"):
            compute_generation_stats([], [], None)  # type: ignore[arg-type]

    def test_empty_dedup(self) -> None:
        """Test with empty after_dedup list."""
        gen = [SyntheticSample("a", GenerationMethod.SELF_INSTRUCT, 0.8, 0.7)]
        stats = compute_generation_stats(gen, gen, [])
        assert stats.final_count == 0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_diversity_score == 0.0


class TestPropertyBased:
    """Property-based tests for synthetic module."""

    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=20)
    def test_create_self_instruct_config_valid(self, num_instructions: int) -> None:
        """Test that create_self_instruct_config returns valid config."""
        config = create_self_instruct_config(num_instructions=num_instructions)
        assert config.num_instructions == num_instructions
        validate_self_instruct_config(config)

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_create_evol_instruct_config_valid(self, evolution_steps: int) -> None:
        """Test that create_evol_instruct_config returns valid config."""
        config = create_evol_instruct_config(evolution_steps=evolution_steps)
        assert config.evolution_steps == evolution_steps
        validate_evol_instruct_config(config)

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20)
    def test_dedup_threshold_valid_range(self, threshold: float) -> None:
        """Test that dedup_threshold in valid range creates valid config."""
        config = create_synthetic_config(
            dedup_threshold=threshold,
            min_length=10,
            max_length=100,
        )
        assert config.dedup_threshold == pytest.approx(threshold)
        validate_synthetic_config(config)

    @given(
        st.lists(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("L", "N", "P")),
            ),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=20)
    def test_diversity_score_valid_range(self, texts: list[str]) -> None:
        """Test that diversity score is always in valid range."""
        # Filter out empty texts
        non_empty_texts = [t for t in texts if t.strip()]
        score = calculate_diversity_score(non_empty_texts)
        assert 0.0 <= score <= 1.0

    @given(
        st.sampled_from(
            ["instruction_tuning", "qa_generation", "summarization", "code_generation"]
        )
    )
    @settings(max_examples=10)
    def test_recommended_config_valid(self, use_case: str) -> None:
        """Test that recommended configs are always valid."""
        config = get_recommended_synthetic_config(use_case)
        validate_synthetic_config(config)
