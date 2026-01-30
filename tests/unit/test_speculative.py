"""Tests for speculative decoding functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.inference.speculative import (
    VALID_ACCEPTANCE_CRITERIA,
    VALID_DRAFT_MODEL_TYPES,
    VALID_VERIFICATION_STRATEGIES,
    AcceptanceCriteria,
    DraftModelConfig,
    DraftModelType,
    SpeculativeConfig,
    SpeculativeStats,
    VerificationConfig,
    VerificationStrategy,
    calculate_expected_speedup,
    calculate_speculation_efficiency,
    create_draft_model_config,
    create_speculative_config,
    create_verification_config,
    estimate_acceptance_rate,
    format_speculative_stats,
    get_acceptance_criteria,
    get_draft_model_type,
    get_recommended_speculative_config,
    get_verification_strategy,
    list_acceptance_criteria,
    list_draft_model_types,
    list_verification_strategies,
    validate_draft_model_config,
    validate_speculative_config,
    validate_verification_config,
)


class TestVerificationStrategy:
    """Tests for VerificationStrategy enum."""

    def test_greedy_value(self) -> None:
        """Test GREEDY verification value."""
        assert VerificationStrategy.GREEDY.value == "greedy"

    def test_sampling_value(self) -> None:
        """Test SAMPLING verification value."""
        assert VerificationStrategy.SAMPLING.value == "sampling"

    def test_nucleus_value(self) -> None:
        """Test NUCLEUS verification value."""
        assert VerificationStrategy.NUCLEUS.value == "nucleus"

    def test_valid_frozenset_matches_enum(self) -> None:
        """Test that VALID_VERIFICATION_STRATEGIES matches enum."""
        expected = frozenset(v.value for v in VerificationStrategy)
        assert expected == VALID_VERIFICATION_STRATEGIES


class TestDraftModelType:
    """Tests for DraftModelType enum."""

    def test_smaller_same_family_value(self) -> None:
        """Test SMALLER_SAME_FAMILY value."""
        assert DraftModelType.SMALLER_SAME_FAMILY.value == "smaller_same_family"

    def test_distilled_value(self) -> None:
        """Test DISTILLED value."""
        assert DraftModelType.DISTILLED.value == "distilled"

    def test_ngram_value(self) -> None:
        """Test NGRAM value."""
        assert DraftModelType.NGRAM.value == "ngram"

    def test_medusa_value(self) -> None:
        """Test MEDUSA value."""
        assert DraftModelType.MEDUSA.value == "medusa"

    def test_valid_frozenset_matches_enum(self) -> None:
        """Test that VALID_DRAFT_MODEL_TYPES matches enum."""
        expected = frozenset(d.value for d in DraftModelType)
        assert expected == VALID_DRAFT_MODEL_TYPES


class TestAcceptanceCriteria:
    """Tests for AcceptanceCriteria enum."""

    def test_threshold_value(self) -> None:
        """Test THRESHOLD value."""
        assert AcceptanceCriteria.THRESHOLD.value == "threshold"

    def test_top_k_value(self) -> None:
        """Test TOP_K value."""
        assert AcceptanceCriteria.TOP_K.value == "top_k"

    def test_adaptive_value(self) -> None:
        """Test ADAPTIVE value."""
        assert AcceptanceCriteria.ADAPTIVE.value == "adaptive"

    def test_valid_frozenset_matches_enum(self) -> None:
        """Test that VALID_ACCEPTANCE_CRITERIA matches enum."""
        expected = frozenset(a.value for a in AcceptanceCriteria)
        assert expected == VALID_ACCEPTANCE_CRITERIA


class TestDraftModelConfig:
    """Tests for DraftModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DraftModelConfig()
        assert config.model_type == DraftModelType.SMALLER_SAME_FAMILY
        assert config.model_name == ""
        assert config.gamma_tokens == 5
        assert config.temperature == 1.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DraftModelConfig(
            model_type=DraftModelType.DISTILLED,
            model_name="distilgpt2",
            gamma_tokens=8,
            temperature=0.7,
        )
        assert config.model_type == DraftModelType.DISTILLED
        assert config.model_name == "distilgpt2"
        assert config.gamma_tokens == 8
        assert config.temperature == pytest.approx(0.7)

    def test_frozen(self) -> None:
        """Test that DraftModelConfig is immutable."""
        config = DraftModelConfig()
        with pytest.raises(AttributeError):
            config.gamma_tokens = 10  # type: ignore[misc]


class TestVerificationConfig:
    """Tests for VerificationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VerificationConfig()
        assert config.strategy == VerificationStrategy.GREEDY
        assert config.acceptance_criteria == AcceptanceCriteria.THRESHOLD
        assert config.threshold == pytest.approx(0.9)
        assert config.fallback_to_target is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = VerificationConfig(
            strategy=VerificationStrategy.SAMPLING,
            acceptance_criteria=AcceptanceCriteria.ADAPTIVE,
            threshold=0.85,
            fallback_to_target=False,
        )
        assert config.strategy == VerificationStrategy.SAMPLING
        assert config.acceptance_criteria == AcceptanceCriteria.ADAPTIVE
        assert config.threshold == pytest.approx(0.85)
        assert config.fallback_to_target is False

    def test_frozen(self) -> None:
        """Test that VerificationConfig is immutable."""
        config = VerificationConfig()
        with pytest.raises(AttributeError):
            config.threshold = 0.5  # type: ignore[misc]


class TestSpeculativeConfig:
    """Tests for SpeculativeConfig dataclass."""

    def test_default_max_speculation_length(self) -> None:
        """Test default max_speculation_length value."""
        draft = DraftModelConfig()
        verification = VerificationConfig()
        config = SpeculativeConfig(draft, verification)
        assert config.max_speculation_length == 128

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        draft = DraftModelConfig(model_name="gpt2", gamma_tokens=8)
        verification = VerificationConfig(threshold=0.95)
        config = SpeculativeConfig(draft, verification, max_speculation_length=256)
        assert config.draft_config.model_name == "gpt2"
        assert config.draft_config.gamma_tokens == 8
        assert config.verification_config.threshold == pytest.approx(0.95)
        assert config.max_speculation_length == 256

    def test_frozen(self) -> None:
        """Test that SpeculativeConfig is immutable."""
        draft = DraftModelConfig()
        verification = VerificationConfig()
        config = SpeculativeConfig(draft, verification)
        with pytest.raises(AttributeError):
            config.max_speculation_length = 512  # type: ignore[misc]


class TestSpeculativeStats:
    """Tests for SpeculativeStats dataclass."""

    def test_creation(self) -> None:
        """Test creating SpeculativeStats instance."""
        stats = SpeculativeStats(
            accepted_tokens=80,
            rejected_tokens=20,
            speedup_factor=2.5,
            acceptance_rate=0.8,
        )
        assert stats.accepted_tokens == 80
        assert stats.rejected_tokens == 20
        assert stats.speedup_factor == pytest.approx(2.5)
        assert stats.acceptance_rate == pytest.approx(0.8)

    def test_frozen(self) -> None:
        """Test that SpeculativeStats is immutable."""
        stats = SpeculativeStats(
            accepted_tokens=80,
            rejected_tokens=20,
            speedup_factor=2.5,
            acceptance_rate=0.8,
        )
        with pytest.raises(AttributeError):
            stats.accepted_tokens = 100  # type: ignore[misc]


class TestValidateDraftModelConfig:
    """Tests for validate_draft_model_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = DraftModelConfig(model_name="gpt2", gamma_tokens=5)
        validate_draft_model_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_draft_model_config(None)  # type: ignore[arg-type]

    def test_zero_gamma_tokens_raises_error(self) -> None:
        """Test that zero gamma_tokens raises ValueError."""
        config = DraftModelConfig(gamma_tokens=0)
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            validate_draft_model_config(config)

    def test_negative_gamma_tokens_raises_error(self) -> None:
        """Test that negative gamma_tokens raises ValueError."""
        config = DraftModelConfig(gamma_tokens=-1)
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            validate_draft_model_config(config)

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        config = DraftModelConfig(temperature=0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_draft_model_config(config)

    def test_negative_temperature_raises_error(self) -> None:
        """Test that negative temperature raises ValueError."""
        config = DraftModelConfig(temperature=-0.5)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_draft_model_config(config)


class TestValidateVerificationConfig:
    """Tests for validate_verification_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = VerificationConfig(threshold=0.9)
        validate_verification_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_verification_config(None)  # type: ignore[arg-type]

    def test_negative_threshold_raises_error(self) -> None:
        """Test that negative threshold raises ValueError."""
        config = VerificationConfig(threshold=-0.1)
        with pytest.raises(ValueError, match=r"threshold must be between"):
            validate_verification_config(config)

    def test_threshold_above_one_raises_error(self) -> None:
        """Test that threshold > 1.0 raises ValueError."""
        config = VerificationConfig(threshold=1.5)
        with pytest.raises(ValueError, match=r"threshold must be between"):
            validate_verification_config(config)

    def test_threshold_at_boundaries(self) -> None:
        """Test threshold at valid boundaries."""
        config_zero = VerificationConfig(threshold=0.0)
        validate_verification_config(config_zero)  # Should not raise

        config_one = VerificationConfig(threshold=1.0)
        validate_verification_config(config_one)  # Should not raise


class TestValidateSpeculativeConfig:
    """Tests for validate_speculative_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        draft = DraftModelConfig(model_name="gpt2", gamma_tokens=5)
        verification = VerificationConfig(threshold=0.9)
        config = SpeculativeConfig(draft, verification, 256)
        validate_speculative_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_speculative_config(None)  # type: ignore[arg-type]

    def test_zero_max_speculation_length_raises_error(self) -> None:
        """Test that zero max_speculation_length raises ValueError."""
        draft = DraftModelConfig()
        verification = VerificationConfig()
        config = SpeculativeConfig(draft, verification, max_speculation_length=0)
        with pytest.raises(ValueError, match="max_speculation_length must be positive"):
            validate_speculative_config(config)

    def test_negative_max_speculation_length_raises_error(self) -> None:
        """Test that negative max_speculation_length raises ValueError."""
        draft = DraftModelConfig()
        verification = VerificationConfig()
        config = SpeculativeConfig(draft, verification, max_speculation_length=-1)
        with pytest.raises(ValueError, match="max_speculation_length must be positive"):
            validate_speculative_config(config)

    def test_invalid_draft_config_raises_error(self) -> None:
        """Test that invalid draft_config raises ValueError."""
        draft = DraftModelConfig(gamma_tokens=0)
        verification = VerificationConfig()
        config = SpeculativeConfig(draft, verification)
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            validate_speculative_config(config)

    def test_invalid_verification_config_raises_error(self) -> None:
        """Test that invalid verification_config raises ValueError."""
        draft = DraftModelConfig()
        verification = VerificationConfig(threshold=-0.1)
        config = SpeculativeConfig(draft, verification)
        with pytest.raises(ValueError, match="threshold must be between"):
            validate_speculative_config(config)


class TestCreateDraftModelConfig:
    """Tests for create_draft_model_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_draft_model_config()
        assert config.model_type == DraftModelType.SMALLER_SAME_FAMILY
        assert config.model_name == ""
        assert config.gamma_tokens == 5
        assert config.temperature == 1.0

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = create_draft_model_config(
            model_type="distilled",
            model_name="distilgpt2",
            gamma_tokens=8,
            temperature=0.7,
        )
        assert config.model_type == DraftModelType.DISTILLED
        assert config.model_name == "distilgpt2"
        assert config.gamma_tokens == 8
        assert config.temperature == pytest.approx(0.7)

    def test_invalid_model_type_raises_error(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_draft_model_config(model_type="invalid")

    def test_zero_gamma_tokens_raises_error(self) -> None:
        """Test that zero gamma_tokens raises ValueError."""
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            create_draft_model_config(gamma_tokens=0)

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            create_draft_model_config(temperature=0)

    def test_all_model_types(self) -> None:
        """Test creation with all valid model types."""
        for model_type in VALID_DRAFT_MODEL_TYPES:
            config = create_draft_model_config(model_type=model_type)
            assert config.model_type.value == model_type


class TestCreateVerificationConfig:
    """Tests for create_verification_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_verification_config()
        assert config.strategy == VerificationStrategy.GREEDY
        assert config.acceptance_criteria == AcceptanceCriteria.THRESHOLD
        assert config.threshold == pytest.approx(0.9)
        assert config.fallback_to_target is True

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = create_verification_config(
            strategy="sampling",
            acceptance_criteria="adaptive",
            threshold=0.85,
            fallback_to_target=False,
        )
        assert config.strategy == VerificationStrategy.SAMPLING
        assert config.acceptance_criteria == AcceptanceCriteria.ADAPTIVE
        assert config.threshold == pytest.approx(0.85)
        assert config.fallback_to_target is False

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            create_verification_config(strategy="invalid")

    def test_invalid_acceptance_criteria_raises_error(self) -> None:
        """Test that invalid acceptance_criteria raises ValueError."""
        with pytest.raises(ValueError, match="acceptance_criteria must be one of"):
            create_verification_config(acceptance_criteria="invalid")

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            create_verification_config(threshold=1.5)

    def test_all_strategies(self) -> None:
        """Test creation with all valid strategies."""
        for strategy in VALID_VERIFICATION_STRATEGIES:
            config = create_verification_config(strategy=strategy)
            assert config.strategy.value == strategy

    def test_all_acceptance_criteria(self) -> None:
        """Test creation with all valid acceptance criteria."""
        for criteria in VALID_ACCEPTANCE_CRITERIA:
            config = create_verification_config(acceptance_criteria=criteria)
            assert config.acceptance_criteria.value == criteria


class TestCreateSpeculativeConfig:
    """Tests for create_speculative_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_speculative_config()
        assert config.draft_config.model_name == ""
        assert config.draft_config.gamma_tokens == 5
        assert config.verification_config.threshold == pytest.approx(0.9)
        assert config.max_speculation_length == 128

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = create_speculative_config(
            model_name="gpt2",
            model_type="distilled",
            gamma_tokens=8,
            temperature=0.7,
            strategy="sampling",
            acceptance_criteria="adaptive",
            threshold=0.85,
            fallback_to_target=False,
            max_speculation_length=256,
        )
        assert config.draft_config.model_name == "gpt2"
        assert config.draft_config.model_type == DraftModelType.DISTILLED
        assert config.draft_config.gamma_tokens == 8
        assert config.draft_config.temperature == pytest.approx(0.7)
        assert config.verification_config.strategy == VerificationStrategy.SAMPLING
        assert (
            config.verification_config.acceptance_criteria
            == AcceptanceCriteria.ADAPTIVE
        )
        assert config.verification_config.threshold == pytest.approx(0.85)
        assert config.verification_config.fallback_to_target is False
        assert config.max_speculation_length == 256

    def test_invalid_model_type_raises_error(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_speculative_config(model_type="invalid")

    def test_invalid_max_speculation_length_raises_error(self) -> None:
        """Test that invalid max_speculation_length raises ValueError."""
        with pytest.raises(ValueError, match="max_speculation_length must be positive"):
            create_speculative_config(max_speculation_length=0)


class TestListVerificationStrategies:
    """Tests for list_verification_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_verification_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_verification_strategies()
        assert "greedy" in strategies
        assert "sampling" in strategies
        assert "nucleus" in strategies

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_verification_strategies()
        assert strategies == sorted(strategies)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        strategies = list_verification_strategies()
        assert all(isinstance(s, str) for s in strategies)


class TestListDraftModelTypes:
    """Tests for list_draft_model_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_draft_model_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_draft_model_types()
        assert "smaller_same_family" in types
        assert "distilled" in types
        assert "ngram" in types
        assert "medusa" in types

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_draft_model_types()
        assert types == sorted(types)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        types = list_draft_model_types()
        assert all(isinstance(t, str) for t in types)


class TestListAcceptanceCriteria:
    """Tests for list_acceptance_criteria function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        criteria = list_acceptance_criteria()
        assert isinstance(criteria, list)

    def test_contains_expected_criteria(self) -> None:
        """Test that list contains expected criteria."""
        criteria = list_acceptance_criteria()
        assert "threshold" in criteria
        assert "top_k" in criteria
        assert "adaptive" in criteria

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        criteria = list_acceptance_criteria()
        assert criteria == sorted(criteria)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        criteria = list_acceptance_criteria()
        assert all(isinstance(c, str) for c in criteria)


class TestGetVerificationStrategy:
    """Tests for get_verification_strategy function."""

    def test_get_greedy(self) -> None:
        """Test getting GREEDY strategy."""
        assert get_verification_strategy("greedy") == VerificationStrategy.GREEDY

    def test_get_sampling(self) -> None:
        """Test getting SAMPLING strategy."""
        assert get_verification_strategy("sampling") == VerificationStrategy.SAMPLING

    def test_get_nucleus(self) -> None:
        """Test getting NUCLEUS strategy."""
        assert get_verification_strategy("nucleus") == VerificationStrategy.NUCLEUS

    def test_invalid_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown verification strategy"):
            get_verification_strategy("invalid")

    def test_empty_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown verification strategy"):
            get_verification_strategy("")


class TestGetDraftModelType:
    """Tests for get_draft_model_type function."""

    def test_get_smaller_same_family(self) -> None:
        """Test getting SMALLER_SAME_FAMILY type."""
        assert (
            get_draft_model_type("smaller_same_family")
            == DraftModelType.SMALLER_SAME_FAMILY
        )

    def test_get_distilled(self) -> None:
        """Test getting DISTILLED type."""
        assert get_draft_model_type("distilled") == DraftModelType.DISTILLED

    def test_get_ngram(self) -> None:
        """Test getting NGRAM type."""
        assert get_draft_model_type("ngram") == DraftModelType.NGRAM

    def test_get_medusa(self) -> None:
        """Test getting MEDUSA type."""
        assert get_draft_model_type("medusa") == DraftModelType.MEDUSA

    def test_invalid_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown draft model type"):
            get_draft_model_type("invalid")

    def test_empty_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown draft model type"):
            get_draft_model_type("")


class TestGetAcceptanceCriteria:
    """Tests for get_acceptance_criteria function."""

    def test_get_threshold(self) -> None:
        """Test getting THRESHOLD criteria."""
        assert get_acceptance_criteria("threshold") == AcceptanceCriteria.THRESHOLD

    def test_get_top_k(self) -> None:
        """Test getting TOP_K criteria."""
        assert get_acceptance_criteria("top_k") == AcceptanceCriteria.TOP_K

    def test_get_adaptive(self) -> None:
        """Test getting ADAPTIVE criteria."""
        assert get_acceptance_criteria("adaptive") == AcceptanceCriteria.ADAPTIVE

    def test_invalid_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown acceptance criteria"):
            get_acceptance_criteria("invalid")

    def test_empty_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown acceptance criteria"):
            get_acceptance_criteria("")


class TestCalculateExpectedSpeedup:
    """Tests for calculate_expected_speedup function."""

    def test_high_acceptance_rate(self) -> None:
        """Test speedup with high acceptance rate."""
        speedup = calculate_expected_speedup(0.8, 5, 0.1)
        assert speedup > 1.0
        assert speedup == pytest.approx(3.33)

    def test_zero_acceptance_rate(self) -> None:
        """Test speedup with zero acceptance rate."""
        speedup = calculate_expected_speedup(0.0, 5, 0.1)
        assert speedup == pytest.approx(0.67)

    def test_full_acceptance_rate(self) -> None:
        """Test speedup with full acceptance rate."""
        speedup = calculate_expected_speedup(1.0, 5, 0.1)
        assert speedup == pytest.approx(4.0)

    def test_negative_acceptance_rate_raises_error(self) -> None:
        """Test that negative acceptance_rate raises ValueError."""
        with pytest.raises(ValueError, match="acceptance_rate must be between"):
            calculate_expected_speedup(-0.1, 5, 0.1)

    def test_acceptance_rate_above_one_raises_error(self) -> None:
        """Test that acceptance_rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="acceptance_rate must be between"):
            calculate_expected_speedup(1.1, 5, 0.1)

    def test_zero_gamma_tokens_raises_error(self) -> None:
        """Test that zero gamma_tokens raises ValueError."""
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            calculate_expected_speedup(0.8, 0, 0.1)

    def test_negative_gamma_tokens_raises_error(self) -> None:
        """Test that negative gamma_tokens raises ValueError."""
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            calculate_expected_speedup(0.8, -1, 0.1)

    def test_zero_latency_ratio_raises_error(self) -> None:
        """Test that zero draft_latency_ratio raises ValueError."""
        with pytest.raises(
            ValueError, match=r"draft_latency_ratio must be between 0\.0 and 1\.0"
        ):
            calculate_expected_speedup(0.8, 5, 0.0)

    def test_latency_ratio_at_one_raises_error(self) -> None:
        """Test that draft_latency_ratio = 1.0 raises ValueError."""
        with pytest.raises(
            ValueError, match=r"draft_latency_ratio must be between 0\.0 and 1\.0"
        ):
            calculate_expected_speedup(0.8, 5, 1.0)

    def test_latency_ratio_above_one_raises_error(self) -> None:
        """Test that draft_latency_ratio > 1.0 raises ValueError."""
        with pytest.raises(
            ValueError, match=r"draft_latency_ratio must be between 0\.0 and 1\.0"
        ):
            calculate_expected_speedup(0.8, 5, 1.5)

    @given(
        acceptance_rate=st.floats(min_value=0.0, max_value=1.0),
        gamma_tokens=st.integers(min_value=1, max_value=20),
        draft_latency_ratio=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20)
    def test_speedup_always_positive(
        self, acceptance_rate: float, gamma_tokens: int, draft_latency_ratio: float
    ) -> None:
        """Test that speedup is always positive for valid inputs."""
        speedup = calculate_expected_speedup(
            acceptance_rate, gamma_tokens, draft_latency_ratio
        )
        assert speedup >= 0.0


class TestEstimateAcceptanceRate:
    """Tests for estimate_acceptance_rate function."""

    def test_equal_perplexities(self) -> None:
        """Test acceptance rate with equal perplexities."""
        rate = estimate_acceptance_rate(15.0, 15.0)
        assert rate == pytest.approx(1.0)

    def test_higher_draft_perplexity(self) -> None:
        """Test acceptance rate when draft has higher perplexity."""
        rate = estimate_acceptance_rate(20.0, 15.0)
        assert rate == pytest.approx(0.75)

    def test_much_higher_draft_perplexity(self) -> None:
        """Test acceptance rate with much higher draft perplexity."""
        rate = estimate_acceptance_rate(30.0, 10.0)
        assert rate == pytest.approx(0.33)

    def test_lower_draft_perplexity(self) -> None:
        """Test acceptance rate when draft has lower perplexity (capped at 1.0)."""
        rate = estimate_acceptance_rate(10.0, 15.0)
        assert rate == pytest.approx(1.0)

    def test_zero_draft_perplexity_raises_error(self) -> None:
        """Test that zero draft_perplexity raises ValueError."""
        with pytest.raises(ValueError, match="draft_perplexity must be positive"):
            estimate_acceptance_rate(0, 15.0)

    def test_negative_draft_perplexity_raises_error(self) -> None:
        """Test that negative draft_perplexity raises ValueError."""
        with pytest.raises(ValueError, match="draft_perplexity must be positive"):
            estimate_acceptance_rate(-5.0, 15.0)

    def test_zero_target_perplexity_raises_error(self) -> None:
        """Test that zero target_perplexity raises ValueError."""
        with pytest.raises(ValueError, match="target_perplexity must be positive"):
            estimate_acceptance_rate(20.0, 0)

    def test_negative_target_perplexity_raises_error(self) -> None:
        """Test that negative target_perplexity raises ValueError."""
        with pytest.raises(ValueError, match="target_perplexity must be positive"):
            estimate_acceptance_rate(20.0, -5.0)

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            estimate_acceptance_rate(20.0, 15.0, temperature=0)

    def test_negative_temperature_raises_error(self) -> None:
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            estimate_acceptance_rate(20.0, 15.0, temperature=-0.5)

    @given(
        draft_perplexity=st.floats(
            min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        target_perplexity=st.floats(
            min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        temperature=st.floats(
            min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20)
    def test_acceptance_rate_bounded(
        self, draft_perplexity: float, target_perplexity: float, temperature: float
    ) -> None:
        """Test that acceptance rate is always in [0, 1]."""
        rate = estimate_acceptance_rate(
            draft_perplexity, target_perplexity, temperature
        )
        assert 0.0 <= rate <= 1.0


class TestCalculateSpeculationEfficiency:
    """Tests for calculate_speculation_efficiency function."""

    def test_high_acceptance(self) -> None:
        """Test efficiency with high acceptance."""
        efficiency = calculate_speculation_efficiency(80, 20, 5)
        assert efficiency == pytest.approx(0.8)

    def test_zero_accepted(self) -> None:
        """Test efficiency with zero accepted."""
        efficiency = calculate_speculation_efficiency(0, 100, 5)
        assert efficiency == pytest.approx(0.0)

    def test_full_acceptance(self) -> None:
        """Test efficiency with full acceptance."""
        efficiency = calculate_speculation_efficiency(100, 0, 5)
        assert efficiency == pytest.approx(1.0)

    def test_zero_total(self) -> None:
        """Test efficiency with zero total (edge case)."""
        efficiency = calculate_speculation_efficiency(0, 0, 5)
        assert efficiency == pytest.approx(0.0)

    def test_negative_accepted_raises_error(self) -> None:
        """Test that negative accepted_tokens raises ValueError."""
        with pytest.raises(ValueError, match="accepted_tokens cannot be negative"):
            calculate_speculation_efficiency(-1, 20, 5)

    def test_negative_rejected_raises_error(self) -> None:
        """Test that negative rejected_tokens raises ValueError."""
        with pytest.raises(ValueError, match="rejected_tokens cannot be negative"):
            calculate_speculation_efficiency(80, -1, 5)

    def test_zero_gamma_tokens_raises_error(self) -> None:
        """Test that zero gamma_tokens raises ValueError."""
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            calculate_speculation_efficiency(80, 20, 0)

    def test_negative_gamma_tokens_raises_error(self) -> None:
        """Test that negative gamma_tokens raises ValueError."""
        with pytest.raises(ValueError, match="gamma_tokens must be positive"):
            calculate_speculation_efficiency(80, 20, -1)

    @given(
        accepted=st.integers(min_value=0, max_value=1000),
        rejected=st.integers(min_value=0, max_value=1000),
        gamma=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=20)
    def test_efficiency_bounded(self, accepted: int, rejected: int, gamma: int) -> None:
        """Test that efficiency is always in [0, 1]."""
        efficiency = calculate_speculation_efficiency(accepted, rejected, gamma)
        assert 0.0 <= efficiency <= 1.0


class TestFormatSpeculativeStats:
    """Tests for format_speculative_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = SpeculativeStats(
            accepted_tokens=80,
            rejected_tokens=20,
            speedup_factor=2.5,
            acceptance_rate=0.8,
        )
        output = format_speculative_stats(stats)
        assert "Speculative Decoding Statistics:" in output
        assert "Accepted: 80" in output
        assert "Rejected: 20" in output
        assert "Total: 100" in output
        assert "Speedup: 2.50x" in output
        assert "Acceptance Rate: 80.0%" in output

    def test_zero_stats(self) -> None:
        """Test formatting with zero stats."""
        stats = SpeculativeStats(
            accepted_tokens=0,
            rejected_tokens=0,
            speedup_factor=0.0,
            acceptance_rate=0.0,
        )
        output = format_speculative_stats(stats)
        assert "Accepted: 0" in output
        assert "Total: 0" in output
        assert "Acceptance Rate: 0.0%" in output

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_speculative_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedSpeculativeConfig:
    """Tests for get_recommended_speculative_config function."""

    def test_small_model_config(self) -> None:
        """Test recommended config for small model."""
        config = get_recommended_speculative_config("small")
        assert config.draft_config.gamma_tokens == 3
        assert config.verification_config.threshold == pytest.approx(0.95)
        assert config.max_speculation_length == 64

    def test_medium_model_config(self) -> None:
        """Test recommended config for medium model."""
        config = get_recommended_speculative_config("medium")
        assert config.draft_config.gamma_tokens == 4
        assert config.verification_config.threshold == pytest.approx(0.9)
        assert config.max_speculation_length == 128

    def test_large_model_config(self) -> None:
        """Test recommended config for large model."""
        config = get_recommended_speculative_config("large")
        assert config.draft_config.gamma_tokens == 5
        assert config.verification_config.threshold == pytest.approx(0.85)
        assert config.max_speculation_length == 256

    def test_xlarge_model_config(self) -> None:
        """Test recommended config for xlarge model."""
        config = get_recommended_speculative_config("xlarge")
        assert config.draft_config.gamma_tokens == 8
        assert config.verification_config.threshold == pytest.approx(0.8)
        assert config.max_speculation_length == 512

    def test_invalid_size_raises_error(self) -> None:
        """Test that invalid model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_speculative_config("invalid")

    def test_empty_size_raises_error(self) -> None:
        """Test that empty model_size raises ValueError."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_speculative_config("")

    def test_all_valid_sizes(self) -> None:
        """Test that all valid sizes return valid configs."""
        for size in ["small", "medium", "large", "xlarge"]:
            config = get_recommended_speculative_config(size)
            # Validate the returned config
            validate_speculative_config(config)

    def test_gamma_increases_with_size(self) -> None:
        """Test that gamma_tokens increases with model size."""
        configs = {
            size: get_recommended_speculative_config(size)
            for size in ["small", "medium", "large", "xlarge"]
        }
        assert (
            configs["small"].draft_config.gamma_tokens
            < configs["medium"].draft_config.gamma_tokens
        )
        assert (
            configs["medium"].draft_config.gamma_tokens
            < configs["large"].draft_config.gamma_tokens
        )
        assert (
            configs["large"].draft_config.gamma_tokens
            < configs["xlarge"].draft_config.gamma_tokens
        )

    def test_threshold_decreases_with_size(self) -> None:
        """Test that threshold decreases with model size."""
        configs = {
            size: get_recommended_speculative_config(size)
            for size in ["small", "medium", "large", "xlarge"]
        }
        assert (
            configs["small"].verification_config.threshold
            > configs["medium"].verification_config.threshold
        )
        assert (
            configs["medium"].verification_config.threshold
            > configs["large"].verification_config.threshold
        )
        assert (
            configs["large"].verification_config.threshold
            > configs["xlarge"].verification_config.threshold
        )
