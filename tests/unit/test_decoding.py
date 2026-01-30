"""Tests for decoding strategies functionality."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.inference.decoding import (
    VALID_DECODING_METHODS,
    VALID_REPETITION_PENALTY_TYPES,
    VALID_STOPPING_CRITERIA,
    BeamConfig,
    ContrastiveConfig,
    DecodingConfig,
    DecodingMethod,
    DecodingStats,
    RepetitionPenaltyType,
    SamplingConfig,
    StoppingCriteria,
    apply_repetition_penalty,
    calculate_beam_memory,
    calculate_entropy,
    create_beam_config,
    create_contrastive_config,
    create_decoding_config,
    create_sampling_config,
    estimate_generation_time,
    format_decoding_stats,
    get_decoding_method,
    get_recommended_decoding_config,
    get_repetition_penalty_type,
    get_stopping_criteria,
    list_decoding_methods,
    list_repetition_penalty_types,
    list_stopping_criteria,
    validate_beam_config,
    validate_contrastive_config,
    validate_decoding_config,
    validate_decoding_stats,
    validate_sampling_config,
)


class TestDecodingMethod:
    """Tests for DecodingMethod enum."""

    def test_greedy_value(self) -> None:
        """Test GREEDY decoding value."""
        assert DecodingMethod.GREEDY.value == "greedy"

    def test_beam_value(self) -> None:
        """Test BEAM decoding value."""
        assert DecodingMethod.BEAM.value == "beam"

    def test_sampling_value(self) -> None:
        """Test SAMPLING decoding value."""
        assert DecodingMethod.SAMPLING.value == "sampling"

    def test_nucleus_value(self) -> None:
        """Test NUCLEUS decoding value."""
        assert DecodingMethod.NUCLEUS.value == "nucleus"

    def test_top_k_value(self) -> None:
        """Test TOP_K decoding value."""
        assert DecodingMethod.TOP_K.value == "top_k"

    def test_contrastive_value(self) -> None:
        """Test CONTRASTIVE decoding value."""
        assert DecodingMethod.CONTRASTIVE.value == "contrastive"

    def test_typical_value(self) -> None:
        """Test TYPICAL decoding value."""
        assert DecodingMethod.TYPICAL.value == "typical"

    def test_valid_frozenset_matches_enum(self) -> None:
        """Test that VALID_DECODING_METHODS matches enum."""
        expected = frozenset(m.value for m in DecodingMethod)
        assert expected == VALID_DECODING_METHODS


class TestStoppingCriteria:
    """Tests for StoppingCriteria enum."""

    def test_max_length_value(self) -> None:
        """Test MAX_LENGTH value."""
        assert StoppingCriteria.MAX_LENGTH.value == "max_length"

    def test_eos_token_value(self) -> None:
        """Test EOS_TOKEN value."""
        assert StoppingCriteria.EOS_TOKEN.value == "eos_token"

    def test_max_time_value(self) -> None:
        """Test MAX_TIME value."""
        assert StoppingCriteria.MAX_TIME.value == "max_time"

    def test_valid_frozenset_matches_enum(self) -> None:
        """Test that VALID_STOPPING_CRITERIA matches enum."""
        expected = frozenset(c.value for c in StoppingCriteria)
        assert expected == VALID_STOPPING_CRITERIA


class TestRepetitionPenaltyType:
    """Tests for RepetitionPenaltyType enum."""

    def test_multiplicative_value(self) -> None:
        """Test MULTIPLICATIVE value."""
        assert RepetitionPenaltyType.MULTIPLICATIVE.value == "multiplicative"

    def test_additive_value(self) -> None:
        """Test ADDITIVE value."""
        assert RepetitionPenaltyType.ADDITIVE.value == "additive"

    def test_presence_value(self) -> None:
        """Test PRESENCE value."""
        assert RepetitionPenaltyType.PRESENCE.value == "presence"

    def test_valid_frozenset_matches_enum(self) -> None:
        """Test that VALID_REPETITION_PENALTY_TYPES matches enum."""
        expected = frozenset(t.value for t in RepetitionPenaltyType)
        assert expected == VALID_REPETITION_PENALTY_TYPES


class TestBeamConfig:
    """Tests for BeamConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BeamConfig()
        assert config.num_beams == 4
        assert config.length_penalty == pytest.approx(1.0)
        assert config.early_stopping is False
        assert config.num_return_sequences == 1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BeamConfig(
            num_beams=8,
            length_penalty=0.8,
            early_stopping=True,
            num_return_sequences=4,
        )
        assert config.num_beams == 8
        assert config.length_penalty == pytest.approx(0.8)
        assert config.early_stopping is True
        assert config.num_return_sequences == 4

    def test_frozen(self) -> None:
        """Test that BeamConfig is immutable."""
        config = BeamConfig()
        with pytest.raises(AttributeError):
            config.num_beams = 10  # type: ignore[misc]


class TestSamplingConfig:
    """Tests for SamplingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SamplingConfig()
        assert config.temperature == pytest.approx(1.0)
        assert config.top_k == 50
        assert config.top_p == pytest.approx(1.0)
        assert config.typical_p == pytest.approx(1.0)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SamplingConfig(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            typical_p=0.95,
        )
        assert config.temperature == pytest.approx(0.7)
        assert config.top_k == 40
        assert config.top_p == pytest.approx(0.9)
        assert config.typical_p == pytest.approx(0.95)

    def test_frozen(self) -> None:
        """Test that SamplingConfig is immutable."""
        config = SamplingConfig()
        with pytest.raises(AttributeError):
            config.temperature = 0.5  # type: ignore[misc]


class TestContrastiveConfig:
    """Tests for ContrastiveConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ContrastiveConfig()
        assert config.penalty_alpha == pytest.approx(0.6)
        assert config.top_k == 4
        assert config.amateur_model == ""

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ContrastiveConfig(
            penalty_alpha=0.5,
            top_k=6,
            amateur_model="gpt2-small",
        )
        assert config.penalty_alpha == pytest.approx(0.5)
        assert config.top_k == 6
        assert config.amateur_model == "gpt2-small"

    def test_frozen(self) -> None:
        """Test that ContrastiveConfig is immutable."""
        config = ContrastiveConfig()
        with pytest.raises(AttributeError):
            config.penalty_alpha = 0.5  # type: ignore[misc]


class TestDecodingConfig:
    """Tests for DecodingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DecodingConfig()
        assert config.method == DecodingMethod.GREEDY
        assert config.beam_config is None
        assert config.sampling_config is None
        assert config.contrastive_config is None
        assert config.max_length == 128
        assert config.repetition_penalty == pytest.approx(1.0)
        assert config.repetition_penalty_type == RepetitionPenaltyType.MULTIPLICATIVE

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        beam = BeamConfig(num_beams=5)
        config = DecodingConfig(
            method=DecodingMethod.BEAM,
            beam_config=beam,
            max_length=256,
            repetition_penalty=1.2,
        )
        assert config.method == DecodingMethod.BEAM
        assert config.beam_config is not None
        assert config.beam_config.num_beams == 5
        assert config.max_length == 256
        assert config.repetition_penalty == pytest.approx(1.2)

    def test_frozen(self) -> None:
        """Test that DecodingConfig is immutable."""
        config = DecodingConfig()
        with pytest.raises(AttributeError):
            config.max_length = 512  # type: ignore[misc]


class TestDecodingStats:
    """Tests for DecodingStats dataclass."""

    def test_creation(self) -> None:
        """Test creating DecodingStats instance."""
        stats = DecodingStats(
            tokens_generated=100,
            time_per_token_ms=25.5,
            tokens_per_second=39.2,
        )
        assert stats.tokens_generated == 100
        assert stats.time_per_token_ms == pytest.approx(25.5)
        assert stats.tokens_per_second == pytest.approx(39.2)

    def test_frozen(self) -> None:
        """Test that DecodingStats is immutable."""
        stats = DecodingStats(
            tokens_generated=100,
            time_per_token_ms=25.5,
            tokens_per_second=39.2,
        )
        with pytest.raises(AttributeError):
            stats.tokens_generated = 200  # type: ignore[misc]


class TestValidateBeamConfig:
    """Tests for validate_beam_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BeamConfig(num_beams=5, num_return_sequences=3)
        validate_beam_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_beam_config(None)  # type: ignore[arg-type]

    def test_zero_num_beams_raises_error(self) -> None:
        """Test that zero num_beams raises ValueError."""
        config = BeamConfig(num_beams=0)
        with pytest.raises(ValueError, match="num_beams must be positive"):
            validate_beam_config(config)

    def test_negative_num_beams_raises_error(self) -> None:
        """Test that negative num_beams raises ValueError."""
        config = BeamConfig(num_beams=-1)
        with pytest.raises(ValueError, match="num_beams must be positive"):
            validate_beam_config(config)

    def test_negative_length_penalty_raises_error(self) -> None:
        """Test that negative length_penalty raises ValueError."""
        config = BeamConfig(length_penalty=-0.5)
        with pytest.raises(ValueError, match="length_penalty cannot be negative"):
            validate_beam_config(config)

    def test_zero_num_return_sequences_raises_error(self) -> None:
        """Test that zero num_return_sequences raises ValueError."""
        config = BeamConfig(num_return_sequences=0)
        with pytest.raises(ValueError, match="num_return_sequences must be positive"):
            validate_beam_config(config)

    def test_num_return_exceeds_num_beams_raises_error(self) -> None:
        """Test that num_return_sequences > num_beams raises ValueError."""
        config = BeamConfig(num_beams=3, num_return_sequences=5)
        with pytest.raises(ValueError, match="num_return_sequences cannot exceed"):
            validate_beam_config(config)


class TestValidateSamplingConfig:
    """Tests for validate_sampling_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = SamplingConfig(temperature=0.7, top_p=0.9)
        validate_sampling_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_sampling_config(None)  # type: ignore[arg-type]

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        config = SamplingConfig(temperature=0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_sampling_config(config)

    def test_negative_temperature_raises_error(self) -> None:
        """Test that negative temperature raises ValueError."""
        config = SamplingConfig(temperature=-0.5)
        with pytest.raises(ValueError, match="temperature must be positive"):
            validate_sampling_config(config)

    def test_negative_top_k_raises_error(self) -> None:
        """Test that negative top_k raises ValueError."""
        config = SamplingConfig(top_k=-1)
        with pytest.raises(ValueError, match="top_k cannot be negative"):
            validate_sampling_config(config)

    def test_zero_top_p_raises_error(self) -> None:
        """Test that zero top_p raises ValueError."""
        config = SamplingConfig(top_p=0)
        with pytest.raises(ValueError, match=r"top_p must be in \(0\.0, 1\.0\]"):
            validate_sampling_config(config)

    def test_top_p_above_one_raises_error(self) -> None:
        """Test that top_p > 1.0 raises ValueError."""
        config = SamplingConfig(top_p=1.5)
        with pytest.raises(ValueError, match=r"top_p must be in \(0\.0, 1\.0\]"):
            validate_sampling_config(config)

    def test_zero_typical_p_raises_error(self) -> None:
        """Test that zero typical_p raises ValueError."""
        config = SamplingConfig(typical_p=0)
        with pytest.raises(ValueError, match=r"typical_p must be in \(0\.0, 1\.0\]"):
            validate_sampling_config(config)

    def test_typical_p_above_one_raises_error(self) -> None:
        """Test that typical_p > 1.0 raises ValueError."""
        config = SamplingConfig(typical_p=1.5)
        with pytest.raises(ValueError, match=r"typical_p must be in \(0\.0, 1\.0\]"):
            validate_sampling_config(config)

    def test_top_k_zero_is_valid(self) -> None:
        """Test that top_k = 0 is valid (disables top-k filtering)."""
        config = SamplingConfig(top_k=0)
        validate_sampling_config(config)  # Should not raise


class TestValidateContrastiveConfig:
    """Tests for validate_contrastive_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = ContrastiveConfig(penalty_alpha=0.5, top_k=6)
        validate_contrastive_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_contrastive_config(None)  # type: ignore[arg-type]

    def test_negative_penalty_alpha_raises_error(self) -> None:
        """Test that negative penalty_alpha raises ValueError."""
        config = ContrastiveConfig(penalty_alpha=-0.1)
        with pytest.raises(
            ValueError, match=r"penalty_alpha must be in \[0\.0, 1\.0\]"
        ):
            validate_contrastive_config(config)

    def test_penalty_alpha_above_one_raises_error(self) -> None:
        """Test that penalty_alpha > 1.0 raises ValueError."""
        config = ContrastiveConfig(penalty_alpha=1.5)
        with pytest.raises(
            ValueError, match=r"penalty_alpha must be in \[0\.0, 1\.0\]"
        ):
            validate_contrastive_config(config)

    def test_zero_top_k_raises_error(self) -> None:
        """Test that zero top_k raises ValueError."""
        config = ContrastiveConfig(top_k=0)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_contrastive_config(config)

    def test_negative_top_k_raises_error(self) -> None:
        """Test that negative top_k raises ValueError."""
        config = ContrastiveConfig(top_k=-1)
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_contrastive_config(config)

    def test_boundary_penalty_alpha_values(self) -> None:
        """Test boundary values for penalty_alpha."""
        config_zero = ContrastiveConfig(penalty_alpha=0.0)
        validate_contrastive_config(config_zero)  # Should not raise

        config_one = ContrastiveConfig(penalty_alpha=1.0)
        validate_contrastive_config(config_one)  # Should not raise


class TestValidateDecodingConfig:
    """Tests for validate_decoding_config function."""

    def test_valid_greedy_config(self) -> None:
        """Test validation of valid greedy config."""
        config = DecodingConfig(max_length=256)
        validate_decoding_config(config)  # Should not raise

    def test_valid_beam_config(self) -> None:
        """Test validation of valid beam config."""
        beam = BeamConfig(num_beams=5)
        config = DecodingConfig(method=DecodingMethod.BEAM, beam_config=beam)
        validate_decoding_config(config)  # Should not raise

    def test_valid_sampling_config(self) -> None:
        """Test validation of valid sampling config."""
        sampling = SamplingConfig(temperature=0.7)
        config = DecodingConfig(
            method=DecodingMethod.SAMPLING, sampling_config=sampling
        )
        validate_decoding_config(config)  # Should not raise

    def test_valid_contrastive_config(self) -> None:
        """Test validation of valid contrastive config."""
        contrastive = ContrastiveConfig(penalty_alpha=0.5)
        config = DecodingConfig(
            method=DecodingMethod.CONTRASTIVE, contrastive_config=contrastive
        )
        validate_decoding_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_decoding_config(None)  # type: ignore[arg-type]

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        config = DecodingConfig(max_length=0)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_decoding_config(config)

    def test_negative_max_length_raises_error(self) -> None:
        """Test that negative max_length raises ValueError."""
        config = DecodingConfig(max_length=-1)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_decoding_config(config)

    def test_zero_repetition_penalty_raises_error(self) -> None:
        """Test that zero repetition_penalty raises ValueError."""
        config = DecodingConfig(repetition_penalty=0)
        with pytest.raises(ValueError, match="repetition_penalty must be positive"):
            validate_decoding_config(config)

    def test_missing_beam_config_raises_error(self) -> None:
        """Test that missing beam_config raises ValueError."""
        config = DecodingConfig(method=DecodingMethod.BEAM)
        with pytest.raises(ValueError, match="beam_config required"):
            validate_decoding_config(config)

    def test_missing_sampling_config_raises_error(self) -> None:
        """Test that missing sampling_config raises ValueError."""
        config = DecodingConfig(method=DecodingMethod.SAMPLING)
        with pytest.raises(ValueError, match="sampling_config required"):
            validate_decoding_config(config)

    def test_missing_sampling_config_for_nucleus_raises_error(self) -> None:
        """Test that missing sampling_config for nucleus raises ValueError."""
        config = DecodingConfig(method=DecodingMethod.NUCLEUS)
        with pytest.raises(ValueError, match="sampling_config required"):
            validate_decoding_config(config)

    def test_missing_sampling_config_for_top_k_raises_error(self) -> None:
        """Test that missing sampling_config for top_k raises ValueError."""
        config = DecodingConfig(method=DecodingMethod.TOP_K)
        with pytest.raises(ValueError, match="sampling_config required"):
            validate_decoding_config(config)

    def test_missing_sampling_config_for_typical_raises_error(self) -> None:
        """Test that missing sampling_config for typical raises ValueError."""
        config = DecodingConfig(method=DecodingMethod.TYPICAL)
        with pytest.raises(ValueError, match="sampling_config required"):
            validate_decoding_config(config)

    def test_missing_contrastive_config_raises_error(self) -> None:
        """Test that missing contrastive_config raises ValueError."""
        config = DecodingConfig(method=DecodingMethod.CONTRASTIVE)
        with pytest.raises(ValueError, match="contrastive_config required"):
            validate_decoding_config(config)


class TestValidateDecodingStats:
    """Tests for validate_decoding_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = DecodingStats(100, 25.5, 39.2)
        validate_decoding_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_decoding_stats(None)  # type: ignore[arg-type]

    def test_negative_tokens_generated_raises_error(self) -> None:
        """Test that negative tokens_generated raises ValueError."""
        stats = DecodingStats(-1, 25.5, 39.2)
        with pytest.raises(ValueError, match="tokens_generated cannot be negative"):
            validate_decoding_stats(stats)

    def test_negative_time_per_token_raises_error(self) -> None:
        """Test that negative time_per_token_ms raises ValueError."""
        stats = DecodingStats(100, -1.0, 39.2)
        with pytest.raises(ValueError, match="time_per_token_ms cannot be negative"):
            validate_decoding_stats(stats)

    def test_negative_tokens_per_second_raises_error(self) -> None:
        """Test that negative tokens_per_second raises ValueError."""
        stats = DecodingStats(100, 25.5, -1.0)
        with pytest.raises(ValueError, match="tokens_per_second cannot be negative"):
            validate_decoding_stats(stats)

    def test_zero_values_are_valid(self) -> None:
        """Test that zero values are valid."""
        stats = DecodingStats(0, 0.0, 0.0)
        validate_decoding_stats(stats)  # Should not raise


class TestCreateBeamConfig:
    """Tests for create_beam_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_beam_config()
        assert config.num_beams == 4
        assert config.length_penalty == pytest.approx(1.0)
        assert config.early_stopping is False
        assert config.num_return_sequences == 1

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = create_beam_config(
            num_beams=8,
            length_penalty=0.8,
            early_stopping=True,
            num_return_sequences=4,
        )
        assert config.num_beams == 8
        assert config.length_penalty == pytest.approx(0.8)
        assert config.early_stopping is True
        assert config.num_return_sequences == 4

    def test_zero_num_beams_raises_error(self) -> None:
        """Test that zero num_beams raises ValueError."""
        with pytest.raises(ValueError, match="num_beams must be positive"):
            create_beam_config(num_beams=0)

    def test_num_return_exceeds_num_beams_raises_error(self) -> None:
        """Test that num_return_sequences > num_beams raises ValueError."""
        with pytest.raises(ValueError, match="num_return_sequences cannot exceed"):
            create_beam_config(num_beams=3, num_return_sequences=5)


class TestCreateSamplingConfig:
    """Tests for create_sampling_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_sampling_config()
        assert config.temperature == pytest.approx(1.0)
        assert config.top_k == 50
        assert config.top_p == pytest.approx(1.0)
        assert config.typical_p == pytest.approx(1.0)

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = create_sampling_config(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            typical_p=0.95,
        )
        assert config.temperature == pytest.approx(0.7)
        assert config.top_k == 40
        assert config.top_p == pytest.approx(0.9)
        assert config.typical_p == pytest.approx(0.95)

    def test_zero_temperature_raises_error(self) -> None:
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            create_sampling_config(temperature=0)

    def test_zero_top_p_raises_error(self) -> None:
        """Test that zero top_p raises ValueError."""
        with pytest.raises(ValueError, match=r"top_p must be in \(0\.0, 1\.0\]"):
            create_sampling_config(top_p=0)


class TestCreateContrastiveConfig:
    """Tests for create_contrastive_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_contrastive_config()
        assert config.penalty_alpha == pytest.approx(0.6)
        assert config.top_k == 4
        assert config.amateur_model == ""

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = create_contrastive_config(
            penalty_alpha=0.5,
            top_k=6,
            amateur_model="gpt2-small",
        )
        assert config.penalty_alpha == pytest.approx(0.5)
        assert config.top_k == 6
        assert config.amateur_model == "gpt2-small"

    def test_negative_penalty_alpha_raises_error(self) -> None:
        """Test that negative penalty_alpha raises ValueError."""
        with pytest.raises(
            ValueError, match=r"penalty_alpha must be in \[0\.0, 1\.0\]"
        ):
            create_contrastive_config(penalty_alpha=-0.1)

    def test_zero_top_k_raises_error(self) -> None:
        """Test that zero top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            create_contrastive_config(top_k=0)


class TestCreateDecodingConfig:
    """Tests for create_decoding_config function."""

    def test_default_values(self) -> None:
        """Test creation with default values."""
        config = create_decoding_config()
        assert config.method == DecodingMethod.GREEDY
        assert config.max_length == 128
        assert config.repetition_penalty == pytest.approx(1.0)

    def test_beam_config(self) -> None:
        """Test creation with beam config."""
        beam = create_beam_config(num_beams=5)
        config = create_decoding_config(method="beam", beam_config=beam)
        assert config.method == DecodingMethod.BEAM
        assert config.beam_config is not None
        assert config.beam_config.num_beams == 5

    def test_sampling_config(self) -> None:
        """Test creation with sampling config."""
        sampling = create_sampling_config(temperature=0.7)
        config = create_decoding_config(method="sampling", sampling_config=sampling)
        assert config.method == DecodingMethod.SAMPLING
        assert config.sampling_config is not None
        assert config.sampling_config.temperature == pytest.approx(0.7)

    def test_contrastive_config(self) -> None:
        """Test creation with contrastive config."""
        contrastive = create_contrastive_config(penalty_alpha=0.5)
        config = create_decoding_config(
            method="contrastive", contrastive_config=contrastive
        )
        assert config.method == DecodingMethod.CONTRASTIVE
        assert config.contrastive_config is not None
        assert config.contrastive_config.penalty_alpha == pytest.approx(0.5)

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_decoding_config(method="invalid")  # type: ignore[arg-type]

    def test_invalid_repetition_penalty_type_raises_error(self) -> None:
        """Test that invalid repetition_penalty_type raises ValueError."""
        with pytest.raises(ValueError, match="repetition_penalty_type must be one of"):
            create_decoding_config(repetition_penalty_type="invalid")  # type: ignore[arg-type]

    def test_missing_beam_config_raises_error(self) -> None:
        """Test that missing beam_config raises ValueError."""
        with pytest.raises(ValueError, match="beam_config required"):
            create_decoding_config(method="beam")

    def test_all_decoding_methods_valid(self) -> None:
        """Test creation with all valid methods that don't need config."""
        # Greedy doesn't need any config
        config = create_decoding_config(method="greedy")
        assert config.method == DecodingMethod.GREEDY


class TestListDecodingMethods:
    """Tests for list_decoding_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_decoding_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_decoding_methods()
        assert "greedy" in methods
        assert "beam" in methods
        assert "sampling" in methods
        assert "nucleus" in methods
        assert "top_k" in methods
        assert "contrastive" in methods
        assert "typical" in methods

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_decoding_methods()
        assert methods == sorted(methods)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        methods = list_decoding_methods()
        assert all(isinstance(m, str) for m in methods)


class TestListStoppingCriteria:
    """Tests for list_stopping_criteria function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        criteria = list_stopping_criteria()
        assert isinstance(criteria, list)

    def test_contains_expected_criteria(self) -> None:
        """Test that list contains expected criteria."""
        criteria = list_stopping_criteria()
        assert "max_length" in criteria
        assert "eos_token" in criteria
        assert "max_time" in criteria

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        criteria = list_stopping_criteria()
        assert criteria == sorted(criteria)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        criteria = list_stopping_criteria()
        assert all(isinstance(c, str) for c in criteria)


class TestListRepetitionPenaltyTypes:
    """Tests for list_repetition_penalty_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_repetition_penalty_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_repetition_penalty_types()
        assert "multiplicative" in types
        assert "additive" in types
        assert "presence" in types

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_repetition_penalty_types()
        assert types == sorted(types)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        types = list_repetition_penalty_types()
        assert all(isinstance(t, str) for t in types)


class TestGetDecodingMethod:
    """Tests for get_decoding_method function."""

    def test_get_greedy(self) -> None:
        """Test getting GREEDY method."""
        assert get_decoding_method("greedy") == DecodingMethod.GREEDY

    def test_get_beam(self) -> None:
        """Test getting BEAM method."""
        assert get_decoding_method("beam") == DecodingMethod.BEAM

    def test_get_sampling(self) -> None:
        """Test getting SAMPLING method."""
        assert get_decoding_method("sampling") == DecodingMethod.SAMPLING

    def test_get_nucleus(self) -> None:
        """Test getting NUCLEUS method."""
        assert get_decoding_method("nucleus") == DecodingMethod.NUCLEUS

    def test_get_top_k(self) -> None:
        """Test getting TOP_K method."""
        assert get_decoding_method("top_k") == DecodingMethod.TOP_K

    def test_get_contrastive(self) -> None:
        """Test getting CONTRASTIVE method."""
        assert get_decoding_method("contrastive") == DecodingMethod.CONTRASTIVE

    def test_get_typical(self) -> None:
        """Test getting TYPICAL method."""
        assert get_decoding_method("typical") == DecodingMethod.TYPICAL

    def test_invalid_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown decoding method"):
            get_decoding_method("invalid")

    def test_empty_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown decoding method"):
            get_decoding_method("")


class TestGetStoppingCriteria:
    """Tests for get_stopping_criteria function."""

    def test_get_max_length(self) -> None:
        """Test getting MAX_LENGTH criteria."""
        assert get_stopping_criteria("max_length") == StoppingCriteria.MAX_LENGTH

    def test_get_eos_token(self) -> None:
        """Test getting EOS_TOKEN criteria."""
        assert get_stopping_criteria("eos_token") == StoppingCriteria.EOS_TOKEN

    def test_get_max_time(self) -> None:
        """Test getting MAX_TIME criteria."""
        assert get_stopping_criteria("max_time") == StoppingCriteria.MAX_TIME

    def test_invalid_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown stopping criteria"):
            get_stopping_criteria("invalid")

    def test_empty_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown stopping criteria"):
            get_stopping_criteria("")


class TestGetRepetitionPenaltyType:
    """Tests for get_repetition_penalty_type function."""

    def test_get_multiplicative(self) -> None:
        """Test getting MULTIPLICATIVE type."""
        assert (
            get_repetition_penalty_type("multiplicative")
            == RepetitionPenaltyType.MULTIPLICATIVE
        )

    def test_get_additive(self) -> None:
        """Test getting ADDITIVE type."""
        assert get_repetition_penalty_type("additive") == RepetitionPenaltyType.ADDITIVE

    def test_get_presence(self) -> None:
        """Test getting PRESENCE type."""
        assert get_repetition_penalty_type("presence") == RepetitionPenaltyType.PRESENCE

    def test_invalid_raises_error(self) -> None:
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown repetition penalty type"):
            get_repetition_penalty_type("invalid")

    def test_empty_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown repetition penalty type"):
            get_repetition_penalty_type("")


class TestCalculateBeamMemory:
    """Tests for calculate_beam_memory function."""

    def test_basic_calculation(self) -> None:
        """Test basic memory calculation."""
        mem = calculate_beam_memory(
            num_beams=4,
            batch_size=1,
            sequence_length=512,
            hidden_size=4096,
            num_layers=32,
            dtype_bytes=2,
        )
        assert mem > 0
        assert mem == pytest.approx(1.0)

    def test_larger_batch_increases_memory(self) -> None:
        """Test that larger batch increases memory."""
        mem_small = calculate_beam_memory(4, 1, 512, 4096, 32)
        mem_large = calculate_beam_memory(4, 2, 512, 4096, 32)
        assert mem_large > mem_small
        assert mem_large == pytest.approx(mem_small * 2)

    def test_more_beams_increases_memory(self) -> None:
        """Test that more beams increases memory."""
        mem_few = calculate_beam_memory(2, 1, 512, 4096, 32)
        mem_many = calculate_beam_memory(8, 1, 512, 4096, 32)
        assert mem_many > mem_few
        assert mem_many == pytest.approx(mem_few * 4)

    def test_zero_num_beams_raises_error(self) -> None:
        """Test that zero num_beams raises ValueError."""
        with pytest.raises(ValueError, match="num_beams must be positive"):
            calculate_beam_memory(0, 1, 512, 4096, 32)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate_beam_memory(4, 0, 512, 4096, 32)

    def test_zero_sequence_length_raises_error(self) -> None:
        """Test that zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            calculate_beam_memory(4, 1, 0, 4096, 32)

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            calculate_beam_memory(4, 1, 512, 0, 32)

    def test_zero_num_layers_raises_error(self) -> None:
        """Test that zero num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            calculate_beam_memory(4, 1, 512, 4096, 0)

    def test_zero_dtype_bytes_raises_error(self) -> None:
        """Test that zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            calculate_beam_memory(4, 1, 512, 4096, 32, dtype_bytes=0)

    @given(
        num_beams=st.integers(min_value=1, max_value=16),
        batch_size=st.integers(min_value=1, max_value=8),
        sequence_length=st.integers(min_value=64, max_value=2048),
        hidden_size=st.integers(min_value=256, max_value=8192),
        num_layers=st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=20)
    def test_memory_always_positive(
        self,
        num_beams: int,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        """Test that memory is always positive for valid inputs."""
        mem = calculate_beam_memory(
            num_beams, batch_size, sequence_length, hidden_size, num_layers
        )
        assert mem > 0


class TestEstimateGenerationTime:
    """Tests for estimate_generation_time function."""

    def test_basic_calculation(self) -> None:
        """Test basic time estimation."""
        time = estimate_generation_time(100, 25.0)
        assert time == pytest.approx(2.5)

    def test_beam_search_overhead(self) -> None:
        """Test beam search adds overhead."""
        time_greedy = estimate_generation_time(100, 25.0, num_beams=1)
        time_beam = estimate_generation_time(100, 25.0, num_beams=4)
        assert time_beam > time_greedy
        assert time_beam == pytest.approx(6.25)

    def test_zero_num_tokens_raises_error(self) -> None:
        """Test that zero num_tokens raises ValueError."""
        with pytest.raises(ValueError, match="num_tokens must be positive"):
            estimate_generation_time(0, 25.0)

    def test_negative_num_tokens_raises_error(self) -> None:
        """Test that negative num_tokens raises ValueError."""
        with pytest.raises(ValueError, match="num_tokens must be positive"):
            estimate_generation_time(-1, 25.0)

    def test_zero_time_per_token_raises_error(self) -> None:
        """Test that zero time_per_token_ms raises ValueError."""
        with pytest.raises(ValueError, match="time_per_token_ms must be positive"):
            estimate_generation_time(100, 0)

    def test_negative_time_per_token_raises_error(self) -> None:
        """Test that negative time_per_token_ms raises ValueError."""
        with pytest.raises(ValueError, match="time_per_token_ms must be positive"):
            estimate_generation_time(100, -1.0)

    def test_zero_num_beams_raises_error(self) -> None:
        """Test that zero num_beams raises ValueError."""
        with pytest.raises(ValueError, match="num_beams must be positive"):
            estimate_generation_time(100, 25.0, num_beams=0)

    @given(
        num_tokens=st.integers(min_value=1, max_value=1000),
        time_per_token_ms=st.floats(min_value=0.1, max_value=100.0),
        num_beams=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=20)
    def test_time_always_positive(
        self, num_tokens: int, time_per_token_ms: float, num_beams: int
    ) -> None:
        """Test that time is always positive for valid inputs."""
        time = estimate_generation_time(num_tokens, time_per_token_ms, num_beams)
        assert time > 0


class TestApplyRepetitionPenalty:
    """Tests for apply_repetition_penalty function."""

    def test_multiplicative_positive_logits(self) -> None:
        """Test multiplicative penalty on positive logits."""
        logits = [1.0, 2.0, 3.0, 4.0]
        generated = [2]
        result = apply_repetition_penalty(logits, generated, 1.2, "multiplicative")
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(2.5)  # 3.0 / 1.2
        assert result[3] == pytest.approx(4.0)

    def test_multiplicative_negative_logits(self) -> None:
        """Test multiplicative penalty on negative logits."""
        logits = [1.0, -2.0, 3.0]
        generated = [1]
        result = apply_repetition_penalty(logits, generated, 1.2, "multiplicative")
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(-2.4)  # -2.0 * 1.2
        assert result[2] == pytest.approx(3.0)

    def test_additive_penalty(self) -> None:
        """Test additive penalty."""
        logits = [1.0, 2.0, 3.0]
        generated = [2]
        result = apply_repetition_penalty(logits, generated, 0.5, "additive")
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(2.5)  # 3.0 - 0.5

    def test_presence_penalty(self) -> None:
        """Test presence penalty."""
        logits = [1.0, 2.0, 3.0]
        generated = [1, 2]
        result = apply_repetition_penalty(logits, generated, 0.5, "presence")
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.5)  # 2.0 - 0.5
        assert result[2] == pytest.approx(2.5)  # 3.0 - 0.5

    def test_no_generated_tokens(self) -> None:
        """Test with no generated tokens."""
        logits = [1.0, 2.0, 3.0]
        generated: list[int] = []
        result = apply_repetition_penalty(logits, generated, 1.2)
        assert result == logits

    def test_out_of_bounds_token_ignored(self) -> None:
        """Test that out-of-bounds tokens are ignored."""
        logits = [1.0, 2.0, 3.0]
        generated = [10]  # Out of bounds
        result = apply_repetition_penalty(logits, generated, 1.2)
        assert result == logits

    def test_invalid_penalty_type_raises_error(self) -> None:
        """Test that invalid penalty_type raises ValueError."""
        with pytest.raises(ValueError, match="penalty_type must be one of"):
            apply_repetition_penalty([1.0], [0], 1.2, "invalid")  # type: ignore[arg-type]

    def test_zero_penalty_multiplicative_raises_error(self) -> None:
        """Test that zero penalty raises ValueError for multiplicative."""
        with pytest.raises(
            ValueError, match="penalty must be positive for multiplicative"
        ):
            apply_repetition_penalty([1.0], [0], 0, "multiplicative")

    def test_negative_penalty_multiplicative_raises_error(self) -> None:
        """Test that negative penalty raises ValueError for multiplicative."""
        with pytest.raises(
            ValueError, match="penalty must be positive for multiplicative"
        ):
            apply_repetition_penalty([1.0], [0], -1.0, "multiplicative")

    def test_does_not_modify_input(self) -> None:
        """Test that input logits are not modified."""
        logits = [1.0, 2.0, 3.0]
        generated = [1]
        _ = apply_repetition_penalty(logits, generated, 1.2)
        assert logits == [1.0, 2.0, 3.0]


class TestCalculateEntropy:
    """Tests for calculate_entropy function."""

    def test_uniform_distribution_two_elements(self) -> None:
        """Test entropy of uniform distribution with 2 elements."""
        probs = [0.5, 0.5]
        entropy = calculate_entropy(probs)
        assert entropy == pytest.approx(math.log(2))

    def test_uniform_distribution_four_elements(self) -> None:
        """Test entropy of uniform distribution with 4 elements."""
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = calculate_entropy(probs)
        assert entropy == pytest.approx(math.log(4))

    def test_deterministic_distribution(self) -> None:
        """Test entropy of deterministic distribution."""
        probs = [1.0]
        entropy = calculate_entropy(probs)
        assert entropy == pytest.approx(0.0)

    def test_skewed_distribution(self) -> None:
        """Test entropy of skewed distribution."""
        probs = [0.9, 0.1]
        entropy = calculate_entropy(probs)
        expected = -0.9 * math.log(0.9) - 0.1 * math.log(0.1)
        assert entropy == pytest.approx(expected)

    def test_empty_probabilities_raises_error(self) -> None:
        """Test that empty probabilities raises ValueError."""
        with pytest.raises(ValueError, match="probabilities cannot be empty"):
            calculate_entropy([])

    def test_negative_probability_raises_error(self) -> None:
        """Test that negative probability raises ValueError."""
        with pytest.raises(ValueError, match="probabilities cannot be negative"):
            calculate_entropy([-0.5, 1.5])

    def test_zero_probability_handled(self) -> None:
        """Test that zero probability is handled correctly."""
        probs = [0.5, 0.0, 0.5]
        entropy = calculate_entropy(probs)
        # Should only consider non-zero probabilities
        expected = -0.5 * math.log(0.5) - 0.5 * math.log(0.5)
        assert entropy == pytest.approx(expected)

    @given(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=20)
    def test_entropy_always_non_negative(self, probs: list[float]) -> None:
        """Test that entropy is always non-negative."""
        entropy = calculate_entropy(probs)
        assert entropy >= 0


class TestFormatDecodingStats:
    """Tests for format_decoding_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = DecodingStats(
            tokens_generated=100,
            time_per_token_ms=25.5,
            tokens_per_second=39.2,
        )
        output = format_decoding_stats(stats)
        assert "Decoding Statistics:" in output
        assert "Tokens Generated: 100" in output
        assert "Time per Token: 25.50 ms" in output
        assert "Throughput: 39.20 tokens/sec" in output

    def test_zero_stats(self) -> None:
        """Test formatting with zero stats."""
        stats = DecodingStats(
            tokens_generated=0,
            time_per_token_ms=0.0,
            tokens_per_second=0.0,
        )
        output = format_decoding_stats(stats)
        assert "Tokens Generated: 0" in output
        assert "Time per Token: 0.00 ms" in output
        assert "Throughput: 0.00 tokens/sec" in output

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="stats cannot be None"):
            format_decoding_stats(None)  # type: ignore[arg-type]


class TestGetRecommendedDecodingConfig:
    """Tests for get_recommended_decoding_config function."""

    def test_creative_config(self) -> None:
        """Test recommended config for creative task."""
        config = get_recommended_decoding_config("creative")
        assert config.method == DecodingMethod.NUCLEUS
        assert config.sampling_config is not None
        assert config.sampling_config.temperature == pytest.approx(0.9)
        assert config.sampling_config.top_p == pytest.approx(0.95)

    def test_translation_quality_config(self) -> None:
        """Test recommended config for translation with quality priority."""
        config = get_recommended_decoding_config("translation", quality_priority=True)
        assert config.method == DecodingMethod.BEAM
        assert config.beam_config is not None
        assert config.beam_config.num_beams == 5

    def test_translation_speed_config(self) -> None:
        """Test recommended config for translation with speed priority."""
        config = get_recommended_decoding_config("translation", quality_priority=False)
        assert config.method == DecodingMethod.GREEDY

    def test_qa_quality_config(self) -> None:
        """Test recommended config for QA with quality priority."""
        config = get_recommended_decoding_config("qa", quality_priority=True)
        assert config.method == DecodingMethod.BEAM
        assert config.beam_config is not None
        assert config.beam_config.num_beams == 4

    def test_qa_speed_config(self) -> None:
        """Test recommended config for QA with speed priority."""
        config = get_recommended_decoding_config("qa", quality_priority=False)
        assert config.method == DecodingMethod.GREEDY

    def test_code_config(self) -> None:
        """Test recommended config for code generation."""
        config = get_recommended_decoding_config("code")
        assert config.method == DecodingMethod.SAMPLING
        assert config.sampling_config is not None
        assert config.sampling_config.temperature == pytest.approx(0.2)

    def test_chat_config(self) -> None:
        """Test recommended config for chat."""
        config = get_recommended_decoding_config("chat")
        assert config.method == DecodingMethod.NUCLEUS
        assert config.sampling_config is not None
        assert config.sampling_config.temperature == pytest.approx(0.7)

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_decoding_config("invalid")

    def test_empty_task_raises_error(self) -> None:
        """Test that empty task raises ValueError."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_recommended_decoding_config("")

    def test_all_valid_tasks(self) -> None:
        """Test that all valid tasks return valid configs."""
        for task in ["creative", "translation", "qa", "code", "chat"]:
            config = get_recommended_decoding_config(task)
            validate_decoding_config(config)

    def test_configs_are_valid(self) -> None:
        """Test that all returned configs pass validation."""
        for task in ["creative", "translation", "qa", "code", "chat"]:
            for quality in [True, False]:
                config = get_recommended_decoding_config(task, quality_priority=quality)
                validate_decoding_config(config)
