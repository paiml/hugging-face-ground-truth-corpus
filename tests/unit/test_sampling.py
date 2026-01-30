"""Tests for generation.sampling module."""

from __future__ import annotations

import pytest

from hf_gtc.generation.sampling import (
    VALID_STOP_CRITERIA,
    VALID_STRATEGIES,
    BeamSearchConfig,
    ContrastiveConfig,
    GenerationConstraints,
    SamplingConfig,
    SamplingStrategy,
    StoppingCriteria,
    calculate_effective_vocab_size,
    create_beam_search_config,
    create_contrastive_config,
    create_generation_constraints,
    create_sampling_config,
    create_stopping_criteria,
    estimate_generation_memory,
    get_recommended_config,
    get_sampling_strategy,
    list_sampling_strategies,
    validate_beam_search_config,
    validate_generation_constraints,
    validate_sampling_config,
)


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        """All strategies have string values."""
        for strategy in SamplingStrategy:
            assert isinstance(strategy.value, str)

    def test_valid_strategies_frozenset(self) -> None:
        """VALID_STRATEGIES is a frozenset of all strategy values."""
        assert isinstance(VALID_STRATEGIES, frozenset)
        assert len(VALID_STRATEGIES) == len(SamplingStrategy)

    def test_greedy_value(self) -> None:
        """Greedy strategy has correct value."""
        assert SamplingStrategy.GREEDY.value == "greedy"

    def test_top_k_value(self) -> None:
        """Top-k strategy has correct value."""
        assert SamplingStrategy.TOP_K.value == "top_k"

    def test_top_p_value(self) -> None:
        """Top-p strategy has correct value."""
        assert SamplingStrategy.TOP_P.value == "top_p"


class TestSamplingConfig:
    """Tests for SamplingConfig dataclass."""

    def test_create_config(self) -> None:
        """Create sampling config."""
        config = SamplingConfig(
            strategy=SamplingStrategy.TOP_P,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, 0.7, 50, 0.9, 1.1, 1.0, 3
        )
        with pytest.raises(AttributeError):
            config.temperature = 0.5  # type: ignore[misc]


class TestValidateSamplingConfig:
    """Tests for validate_sampling_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, 0.7, 50, 0.9, 1.1, 1.0, 3
        )
        validate_sampling_config(config)

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature raises ValueError."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, -0.1, 50, 0.9, 1.0, 1.0, 0
        )
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            validate_sampling_config(config)

    def test_negative_top_k_raises(self) -> None:
        """Negative top_k raises ValueError."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, 0.7, -1, 0.9, 1.0, 1.0, 0
        )
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            validate_sampling_config(config)

    def test_top_p_out_of_range_raises(self) -> None:
        """Top_p out of range raises ValueError."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, 0.7, 50, 1.5, 1.0, 1.0, 0
        )
        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_config(config)

    def test_low_repetition_penalty_raises(self) -> None:
        """Repetition penalty < 1.0 raises ValueError."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, 0.7, 50, 0.9, 0.5, 1.0, 0
        )
        with pytest.raises(ValueError, match="repetition_penalty must be"):
            validate_sampling_config(config)

    def test_negative_ngram_size_raises(self) -> None:
        """Negative ngram size raises ValueError."""
        config = SamplingConfig(
            SamplingStrategy.TOP_P, 0.7, 50, 0.9, 1.0, 1.0, -1
        )
        with pytest.raises(ValueError, match="no_repeat_ngram_size must be"):
            validate_sampling_config(config)


class TestBeamSearchConfig:
    """Tests for BeamSearchConfig dataclass."""

    def test_create_config(self) -> None:
        """Create beam search config."""
        config = BeamSearchConfig(
            num_beams=4,
            num_beam_groups=1,
            diversity_penalty=0.0,
            early_stopping=True,
            length_penalty=1.0,
        )
        assert config.num_beams == 4

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = BeamSearchConfig(4, 1, 0.0, True, 1.0)
        with pytest.raises(AttributeError):
            config.num_beams = 8  # type: ignore[misc]


class TestValidateBeamSearchConfig:
    """Tests for validate_beam_search_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = BeamSearchConfig(4, 1, 0.0, True, 1.0)
        validate_beam_search_config(config)

    def test_zero_beams_raises(self) -> None:
        """Zero beams raises ValueError."""
        config = BeamSearchConfig(0, 1, 0.0, True, 1.0)
        with pytest.raises(ValueError, match="num_beams must be positive"):
            validate_beam_search_config(config)

    def test_zero_beam_groups_raises(self) -> None:
        """Zero beam groups raises ValueError."""
        config = BeamSearchConfig(4, 0, 0.0, True, 1.0)
        with pytest.raises(ValueError, match="num_beam_groups must be positive"):
            validate_beam_search_config(config)

    def test_beams_not_divisible_by_groups_raises(self) -> None:
        """Beams not divisible by groups raises ValueError."""
        config = BeamSearchConfig(5, 2, 0.0, True, 1.0)
        with pytest.raises(ValueError, match="must be divisible by"):
            validate_beam_search_config(config)

    def test_negative_diversity_penalty_raises(self) -> None:
        """Negative diversity penalty raises ValueError."""
        config = BeamSearchConfig(4, 1, -0.5, True, 1.0)
        with pytest.raises(ValueError, match="diversity_penalty must be"):
            validate_beam_search_config(config)


class TestGenerationConstraints:
    """Tests for GenerationConstraints dataclass."""

    def test_create_constraints(self) -> None:
        """Create generation constraints."""
        constraints = GenerationConstraints(
            max_length=512,
            max_new_tokens=256,
            min_length=10,
            min_new_tokens=5,
            max_time=30.0,
        )
        assert constraints.max_new_tokens == 256


class TestValidateGenerationConstraints:
    """Tests for validate_generation_constraints function."""

    def test_valid_constraints(self) -> None:
        """Valid constraints pass validation."""
        constraints = GenerationConstraints(512, 256, 10, 5, 30.0)
        validate_generation_constraints(constraints)

    def test_zero_max_length_raises(self) -> None:
        """Zero max_length raises ValueError."""
        constraints = GenerationConstraints(0, 256, 10, 5, 30.0)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_generation_constraints(constraints)

    def test_zero_max_new_tokens_raises(self) -> None:
        """Zero max_new_tokens raises ValueError."""
        constraints = GenerationConstraints(512, 0, 10, 5, 30.0)
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            validate_generation_constraints(constraints)

    def test_negative_min_length_raises(self) -> None:
        """Negative min_length raises ValueError."""
        constraints = GenerationConstraints(512, 256, -1, 5, 30.0)
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            validate_generation_constraints(constraints)

    def test_min_exceeds_max_raises(self) -> None:
        """Min exceeding max raises ValueError."""
        constraints = GenerationConstraints(512, 256, 600, 5, 30.0)
        with pytest.raises(ValueError, match=r"min_length.*cannot exceed"):
            validate_generation_constraints(constraints)

    def test_zero_max_time_raises(self) -> None:
        """Zero max_time raises ValueError."""
        constraints = GenerationConstraints(512, 256, 10, 5, 0.0)
        with pytest.raises(ValueError, match="max_time must be positive"):
            validate_generation_constraints(constraints)


class TestCreateSamplingConfig:
    """Tests for create_sampling_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_sampling_config()
        assert config.strategy == SamplingStrategy.TOP_P
        assert config.temperature == 1.0

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_sampling_config(
            strategy="greedy",
            temperature=0.5,
            top_p=0.8,
        )
        assert config.strategy == SamplingStrategy.GREEDY
        assert config.temperature == 0.5
        assert config.top_p == 0.8

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_sampling_config(strategy="invalid")

    def test_invalid_temperature_raises(self) -> None:
        """Invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be"):
            create_sampling_config(temperature=-1)


class TestCreateBeamSearchConfig:
    """Tests for create_beam_search_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_beam_search_config()
        assert config.num_beams == 4
        assert config.early_stopping is True

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_beam_search_config(num_beams=8, length_penalty=1.5)
        assert config.num_beams == 8
        assert config.length_penalty == 1.5

    def test_zero_beams_raises(self) -> None:
        """Zero beams raises ValueError."""
        with pytest.raises(ValueError, match="num_beams must be positive"):
            create_beam_search_config(num_beams=0)


class TestCreateGenerationConstraints:
    """Tests for create_generation_constraints function."""

    def test_default_constraints(self) -> None:
        """Create default constraints."""
        constraints = create_generation_constraints()
        assert constraints.max_length == 512
        assert constraints.max_new_tokens == 256

    def test_custom_constraints(self) -> None:
        """Create custom constraints."""
        constraints = create_generation_constraints(max_new_tokens=128)
        assert constraints.max_new_tokens == 128

    def test_invalid_max_length_raises(self) -> None:
        """Invalid max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_generation_constraints(max_length=0)


class TestCreateStoppingCriteria:
    """Tests for create_stopping_criteria function."""

    def test_default_criteria(self) -> None:
        """Create default criteria."""
        criteria = create_stopping_criteria()
        assert criteria.criteria_type == "max_length"
        assert criteria.stop_strings == ()

    def test_with_stop_strings(self) -> None:
        """Create criteria with stop strings."""
        criteria = create_stopping_criteria(stop_strings=("\n\n",))
        assert "\n\n" in criteria.stop_strings

    def test_invalid_type_raises(self) -> None:
        """Invalid criteria type raises ValueError."""
        with pytest.raises(ValueError, match="criteria_type must be one of"):
            create_stopping_criteria(criteria_type="invalid")


class TestCreateContrastiveConfig:
    """Tests for create_contrastive_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_contrastive_config()
        assert config.penalty_alpha == 0.6
        assert config.top_k == 4

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_contrastive_config(penalty_alpha=0.5, top_k=6)
        assert config.penalty_alpha == 0.5
        assert config.top_k == 6

    def test_invalid_alpha_raises(self) -> None:
        """Invalid penalty_alpha raises ValueError."""
        with pytest.raises(ValueError, match="penalty_alpha must be"):
            create_contrastive_config(penalty_alpha=1.5)

    def test_invalid_top_k_raises(self) -> None:
        """Invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            create_contrastive_config(top_k=0)


class TestListSamplingStrategies:
    """Tests for list_sampling_strategies function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        strategies = list_sampling_strategies()
        assert strategies == sorted(strategies)

    def test_contains_all_strategies(self) -> None:
        """Contains all strategy values."""
        strategies = list_sampling_strategies()
        assert "greedy" in strategies
        assert "top_p" in strategies
        assert "beam_search" in strategies


class TestGetSamplingStrategy:
    """Tests for get_sampling_strategy function."""

    def test_get_greedy(self) -> None:
        """Get greedy strategy."""
        assert get_sampling_strategy("greedy") == SamplingStrategy.GREEDY

    def test_get_top_k(self) -> None:
        """Get top_k strategy."""
        assert get_sampling_strategy("top_k") == SamplingStrategy.TOP_K

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            get_sampling_strategy("invalid")


class TestGetRecommendedConfig:
    """Tests for get_recommended_config function."""

    def test_creative_task(self) -> None:
        """Get config for creative task."""
        config = get_recommended_config("creative")
        assert config.temperature > 0.5

    def test_factual_task(self) -> None:
        """Get config for factual task."""
        config = get_recommended_config("factual")
        assert config.temperature == 0.1

    def test_code_task(self) -> None:
        """Get config for code task."""
        config = get_recommended_config("code")
        assert config.temperature == 0.2

    def test_chat_task(self) -> None:
        """Get config for chat task."""
        config = get_recommended_config("chat")
        assert 0.5 < config.temperature < 1.0

    def test_invalid_task_raises(self) -> None:
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_config("invalid")


class TestCalculateEffectiveVocabSize:
    """Tests for calculate_effective_vocab_size function."""

    def test_top_k_only(self) -> None:
        """With top_k only."""
        result = calculate_effective_vocab_size(50000, 50, 1.0)
        assert result == 50

    def test_top_p_only(self) -> None:
        """With top_p only."""
        result = calculate_effective_vocab_size(50000, 0, 0.9)
        assert result == 45000

    def test_both_filters(self) -> None:
        """With both filters."""
        result = calculate_effective_vocab_size(50000, 100, 0.5)
        assert result == 100

    def test_invalid_vocab_size_raises(self) -> None:
        """Invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            calculate_effective_vocab_size(0, 50, 0.9)

    def test_negative_top_k_raises(self) -> None:
        """Negative top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            calculate_effective_vocab_size(50000, -1, 0.9)

    def test_invalid_top_p_raises(self) -> None:
        """Invalid top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between"):
            calculate_effective_vocab_size(50000, 50, 1.5)


class TestEstimateGenerationMemory:
    """Tests for estimate_generation_memory function."""

    def test_basic_estimate(self) -> None:
        """Basic memory estimate."""
        mem = estimate_generation_memory(1, 512, 768, 12)
        assert mem > 0

    def test_larger_batch(self) -> None:
        """Larger batch uses more memory."""
        mem1 = estimate_generation_memory(1, 512, 768, 12)
        mem8 = estimate_generation_memory(8, 512, 768, 12)
        assert mem8 == mem1 * 8

    def test_beam_search_memory(self) -> None:
        """Beam search uses more memory."""
        mem1 = estimate_generation_memory(1, 512, 768, 12, num_beams=1)
        mem4 = estimate_generation_memory(1, 512, 768, 12, num_beams=4)
        assert mem4 == mem1 * 4

    def test_invalid_batch_size_raises(self) -> None:
        """Invalid batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_generation_memory(0, 512, 768, 12)

    def test_invalid_max_length_raises(self) -> None:
        """Invalid max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            estimate_generation_memory(1, 0, 768, 12)

    def test_invalid_hidden_size_raises(self) -> None:
        """Invalid hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_generation_memory(1, 512, 0, 12)

    def test_invalid_num_layers_raises(self) -> None:
        """Invalid num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            estimate_generation_memory(1, 512, 768, 0)

    def test_invalid_num_beams_raises(self) -> None:
        """Invalid num_beams raises ValueError."""
        with pytest.raises(ValueError, match="num_beams must be positive"):
            estimate_generation_memory(1, 512, 768, 12, num_beams=0)


class TestStoppingCriteria:
    """Tests for StoppingCriteria dataclass."""

    def test_create_criteria(self) -> None:
        """Create stopping criteria."""
        criteria = StoppingCriteria(
            stop_strings=("\n\n",),
            stop_token_ids=(50256,),
            criteria_type="eos_token",
        )
        assert criteria.criteria_type == "eos_token"

    def test_valid_stop_criteria(self) -> None:
        """VALID_STOP_CRITERIA contains expected values."""
        assert "max_length" in VALID_STOP_CRITERIA
        assert "eos_token" in VALID_STOP_CRITERIA


class TestContrastiveConfig:
    """Tests for ContrastiveConfig dataclass."""

    def test_create_config(self) -> None:
        """Create contrastive config."""
        config = ContrastiveConfig(penalty_alpha=0.6, top_k=4)
        assert config.penalty_alpha == 0.6
        assert config.top_k == 4
