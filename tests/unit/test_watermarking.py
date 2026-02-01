"""Tests for watermarking utilities."""

from __future__ import annotations

import pytest

from hf_gtc.safety.watermarking import (
    DetectionConfig,
    DetectionMethod,
    EmbeddingConfig,
    WatermarkConfig,
    WatermarkResult,
    WatermarkStats,
    WatermarkStrength,
    WatermarkType,
    calculate_z_score,
    create_detection_config,
    create_embedding_config,
    create_watermark_config,
    estimate_detectability,
    get_detection_method,
    get_watermark_strength,
    get_watermark_type,
    list_detection_methods,
    list_watermark_strengths,
    list_watermark_types,
    validate_detection_config,
    validate_watermark_config,
)


class TestWatermarkType:
    """Tests for WatermarkType enum."""

    def test_soft_value(self) -> None:
        """Test SOFT value."""
        assert WatermarkType.SOFT.value == "soft"

    def test_hard_value(self) -> None:
        """Test HARD value."""
        assert WatermarkType.HARD.value == "hard"

    def test_semantic_value(self) -> None:
        """Test SEMANTIC value."""
        assert WatermarkType.SEMANTIC.value == "semantic"

    def test_statistical_value(self) -> None:
        """Test STATISTICAL value."""
        assert WatermarkType.STATISTICAL.value == "statistical"

    def test_all_types_have_unique_values(self) -> None:
        """Test that all types have unique values."""
        values = [t.value for t in WatermarkType]
        assert len(values) == len(set(values))


class TestWatermarkStrength:
    """Tests for WatermarkStrength enum."""

    def test_low_value(self) -> None:
        """Test LOW value."""
        assert WatermarkStrength.LOW.value == "low"

    def test_medium_value(self) -> None:
        """Test MEDIUM value."""
        assert WatermarkStrength.MEDIUM.value == "medium"

    def test_high_value(self) -> None:
        """Test HIGH value."""
        assert WatermarkStrength.HIGH.value == "high"

    def test_all_strengths_have_unique_values(self) -> None:
        """Test that all strengths have unique values."""
        values = [s.value for s in WatermarkStrength]
        assert len(values) == len(set(values))


class TestDetectionMethod:
    """Tests for DetectionMethod enum."""

    def test_z_score_value(self) -> None:
        """Test Z_SCORE value."""
        assert DetectionMethod.Z_SCORE.value == "z_score"

    def test_log_likelihood_value(self) -> None:
        """Test LOG_LIKELIHOOD value."""
        assert DetectionMethod.LOG_LIKELIHOOD.value == "log_likelihood"

    def test_perplexity_value(self) -> None:
        """Test PERPLEXITY value."""
        assert DetectionMethod.PERPLEXITY.value == "perplexity"

    def test_entropy_value(self) -> None:
        """Test ENTROPY value."""
        assert DetectionMethod.ENTROPY.value == "entropy"

    def test_all_methods_have_unique_values(self) -> None:
        """Test that all methods have unique values."""
        values = [m.value for m in DetectionMethod]
        assert len(values) == len(set(values))


class TestWatermarkConfig:
    """Tests for WatermarkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = WatermarkConfig()
        assert config.watermark_type == WatermarkType.SOFT
        assert config.strength == WatermarkStrength.MEDIUM
        assert config.gamma == pytest.approx(0.25)
        assert config.delta == pytest.approx(2.0)
        assert config.seeding_scheme == "selfhash"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = WatermarkConfig(
            watermark_type=WatermarkType.HARD,
            strength=WatermarkStrength.HIGH,
            gamma=0.5,
            delta=3.0,
            seeding_scheme="custom",
        )
        assert config.watermark_type == WatermarkType.HARD
        assert config.strength == WatermarkStrength.HIGH
        assert config.gamma == pytest.approx(0.5)
        assert config.delta == pytest.approx(3.0)
        assert config.seeding_scheme == "custom"

    def test_frozen(self) -> None:
        """Test that WatermarkConfig is immutable."""
        config = WatermarkConfig()
        with pytest.raises(AttributeError):
            config.gamma = 0.5  # type: ignore[misc]


class TestDetectionConfig:
    """Tests for DetectionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = DetectionConfig()
        assert config.method == DetectionMethod.Z_SCORE
        assert config.threshold == pytest.approx(4.0)
        assert config.window_size == 256
        assert config.ignore_repeated is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = DetectionConfig(
            method=DetectionMethod.LOG_LIKELIHOOD,
            threshold=5.0,
            window_size=512,
            ignore_repeated=False,
        )
        assert config.method == DetectionMethod.LOG_LIKELIHOOD
        assert config.threshold == pytest.approx(5.0)
        assert config.window_size == 512
        assert config.ignore_repeated is False

    def test_frozen(self) -> None:
        """Test that DetectionConfig is immutable."""
        config = DetectionConfig()
        with pytest.raises(AttributeError):
            config.threshold = 5.0  # type: ignore[misc]


class TestWatermarkResult:
    """Tests for WatermarkResult dataclass."""

    def test_creation_watermarked(self) -> None:
        """Test creating watermarked result."""
        result = WatermarkResult(
            is_watermarked=True,
            confidence=0.95,
            z_score=5.2,
            p_value=0.00001,
        )
        assert result.is_watermarked is True
        assert result.confidence == pytest.approx(0.95)
        assert result.z_score == pytest.approx(5.2)
        assert result.p_value == pytest.approx(0.00001)

    def test_creation_not_watermarked(self) -> None:
        """Test creating non-watermarked result."""
        result = WatermarkResult(
            is_watermarked=False,
            confidence=0.1,
            z_score=1.2,
            p_value=0.23,
        )
        assert result.is_watermarked is False
        assert result.confidence == pytest.approx(0.1)
        assert result.z_score == pytest.approx(1.2)
        assert result.p_value == pytest.approx(0.23)

    def test_frozen(self) -> None:
        """Test that WatermarkResult is immutable."""
        result = WatermarkResult(
            is_watermarked=True,
            confidence=0.95,
            z_score=5.2,
            p_value=0.00001,
        )
        with pytest.raises(AttributeError):
            result.is_watermarked = False  # type: ignore[misc]


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = EmbeddingConfig()
        assert config.vocab_fraction == pytest.approx(0.5)
        assert config.hash_key == 15485863
        assert config.context_width == 1

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = EmbeddingConfig(
            vocab_fraction=0.3,
            hash_key=42,
            context_width=2,
        )
        assert config.vocab_fraction == pytest.approx(0.3)
        assert config.hash_key == 42
        assert config.context_width == 2

    def test_frozen(self) -> None:
        """Test that EmbeddingConfig is immutable."""
        config = EmbeddingConfig()
        with pytest.raises(AttributeError):
            config.vocab_fraction = 0.3  # type: ignore[misc]


class TestWatermarkStats:
    """Tests for WatermarkStats dataclass."""

    def test_creation(self) -> None:
        """Test creating stats."""
        stats = WatermarkStats(
            tokens_processed=1000,
            tokens_watermarked=750,
            detection_rate=0.75,
        )
        assert stats.tokens_processed == 1000
        assert stats.tokens_watermarked == 750
        assert stats.detection_rate == pytest.approx(0.75)

    def test_frozen(self) -> None:
        """Test that WatermarkStats is immutable."""
        stats = WatermarkStats(
            tokens_processed=1000,
            tokens_watermarked=750,
            detection_rate=0.75,
        )
        with pytest.raises(AttributeError):
            stats.tokens_processed = 2000  # type: ignore[misc]


class TestValidateWatermarkConfig:
    """Tests for validate_watermark_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = WatermarkConfig(gamma=0.25, delta=2.0)
        validate_watermark_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_watermark_config(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gamma", [0.0, 1.0, -0.5, 1.5, 2.0])
    def test_invalid_gamma_raises_error(self, gamma: float) -> None:
        """Test that invalid gamma raises ValueError."""
        config = WatermarkConfig(gamma=gamma)
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            validate_watermark_config(config)

    @pytest.mark.parametrize("delta", [0.0, -1.0, -0.5])
    def test_invalid_delta_raises_error(self, delta: float) -> None:
        """Test that non-positive delta raises ValueError."""
        config = WatermarkConfig(delta=delta)
        with pytest.raises(ValueError, match="delta must be positive"):
            validate_watermark_config(config)

    @pytest.mark.parametrize("seeding_scheme", ["", "   ", "\t", "\n"])
    def test_empty_seeding_scheme_raises_error(self, seeding_scheme: str) -> None:
        """Test that empty seeding scheme raises ValueError."""
        config = WatermarkConfig(seeding_scheme=seeding_scheme)
        with pytest.raises(ValueError, match="seeding_scheme cannot be empty"):
            validate_watermark_config(config)


class TestValidateDetectionConfig:
    """Tests for validate_detection_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = DetectionConfig(threshold=4.0, window_size=256)
        validate_detection_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_detection_config(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", [0.0, -1.0, -0.5])
    def test_invalid_threshold_raises_error(self, threshold: float) -> None:
        """Test that non-positive threshold raises ValueError."""
        config = DetectionConfig(threshold=threshold)
        with pytest.raises(ValueError, match="threshold must be positive"):
            validate_detection_config(config)

    @pytest.mark.parametrize("window_size", [0, -1, -10])
    def test_invalid_window_size_raises_error(self, window_size: int) -> None:
        """Test that non-positive window size raises ValueError."""
        config = DetectionConfig(window_size=window_size)
        with pytest.raises(ValueError, match="window_size must be positive"):
            validate_detection_config(config)


class TestCreateWatermarkConfig:
    """Tests for create_watermark_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_watermark_config()
        assert config.watermark_type == WatermarkType.SOFT
        assert config.strength == WatermarkStrength.MEDIUM
        assert config.gamma == pytest.approx(0.25)
        assert config.delta == pytest.approx(2.0)
        assert config.seeding_scheme == "selfhash"

    def test_creates_config_with_string_type(self) -> None:
        """Test creating config with string watermark type."""
        config = create_watermark_config(watermark_type="hard")
        assert config.watermark_type == WatermarkType.HARD

    def test_creates_config_with_string_strength(self) -> None:
        """Test creating config with string strength."""
        config = create_watermark_config(strength="high")
        assert config.strength == WatermarkStrength.HIGH

    def test_creates_config_with_enum_type(self) -> None:
        """Test creating config with enum watermark type."""
        config = create_watermark_config(watermark_type=WatermarkType.SEMANTIC)
        assert config.watermark_type == WatermarkType.SEMANTIC

    def test_creates_config_with_enum_strength(self) -> None:
        """Test creating config with enum strength."""
        config = create_watermark_config(strength=WatermarkStrength.LOW)
        assert config.strength == WatermarkStrength.LOW

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_watermark_config(
            watermark_type="statistical",
            strength="low",
            gamma=0.3,
            delta=3.0,
            seeding_scheme="custom",
        )
        assert config.watermark_type == WatermarkType.STATISTICAL
        assert config.strength == WatermarkStrength.LOW
        assert config.gamma == pytest.approx(0.3)
        assert config.delta == pytest.approx(3.0)
        assert config.seeding_scheme == "custom"

    def test_invalid_gamma_raises_error(self) -> None:
        """Test that invalid gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            create_watermark_config(gamma=2.0)

    def test_invalid_delta_raises_error(self) -> None:
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be positive"):
            create_watermark_config(delta=-1.0)


class TestCreateDetectionConfig:
    """Tests for create_detection_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_detection_config()
        assert config.method == DetectionMethod.Z_SCORE
        assert config.threshold == pytest.approx(4.0)
        assert config.window_size == 256
        assert config.ignore_repeated is True

    def test_creates_config_with_string_method(self) -> None:
        """Test creating config with string method."""
        config = create_detection_config(method="log_likelihood")
        assert config.method == DetectionMethod.LOG_LIKELIHOOD

    def test_creates_config_with_enum_method(self) -> None:
        """Test creating config with enum method."""
        config = create_detection_config(method=DetectionMethod.PERPLEXITY)
        assert config.method == DetectionMethod.PERPLEXITY

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_detection_config(
            method="entropy",
            threshold=5.0,
            window_size=512,
            ignore_repeated=False,
        )
        assert config.method == DetectionMethod.ENTROPY
        assert config.threshold == pytest.approx(5.0)
        assert config.window_size == 512
        assert config.ignore_repeated is False

    def test_invalid_threshold_raises_error(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            create_detection_config(threshold=-1.0)

    def test_invalid_window_size_raises_error(self) -> None:
        """Test that invalid window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            create_detection_config(window_size=0)


class TestCreateEmbeddingConfig:
    """Tests for create_embedding_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Test creating config with defaults."""
        config = create_embedding_config()
        assert config.vocab_fraction == pytest.approx(0.5)
        assert config.hash_key == 15485863
        assert config.context_width == 1

    def test_creates_config_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = create_embedding_config(
            vocab_fraction=0.3,
            hash_key=42,
            context_width=2,
        )
        assert config.vocab_fraction == pytest.approx(0.3)
        assert config.hash_key == 42
        assert config.context_width == 2

    @pytest.mark.parametrize("vocab_fraction", [0.0, 1.0, -0.5, 1.5, 2.0])
    def test_invalid_vocab_fraction_raises_error(self, vocab_fraction: float) -> None:
        """Test that invalid vocab_fraction raises ValueError."""
        with pytest.raises(ValueError, match="vocab_fraction must be between 0 and 1"):
            create_embedding_config(vocab_fraction=vocab_fraction)

    @pytest.mark.parametrize("context_width", [0, -1, -10])
    def test_invalid_context_width_raises_error(self, context_width: int) -> None:
        """Test that non-positive context_width raises ValueError."""
        with pytest.raises(ValueError, match="context_width must be positive"):
            create_embedding_config(context_width=context_width)


class TestListWatermarkTypes:
    """Tests for list_watermark_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_watermark_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_watermark_types()
        assert "soft" in types
        assert "hard" in types
        assert "semantic" in types
        assert "statistical" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_watermark_types()
        assert types == sorted(types)

    def test_contains_all_enum_values(self) -> None:
        """Test that list contains all enum values."""
        types = list_watermark_types()
        enum_values = [t.value for t in WatermarkType]
        assert len(types) == len(enum_values)
        for v in enum_values:
            assert v in types


class TestListWatermarkStrengths:
    """Tests for list_watermark_strengths function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strengths = list_watermark_strengths()
        assert isinstance(strengths, list)

    def test_contains_expected_strengths(self) -> None:
        """Test that list contains expected strengths."""
        strengths = list_watermark_strengths()
        assert "low" in strengths
        assert "medium" in strengths
        assert "high" in strengths

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        strengths = list_watermark_strengths()
        assert strengths == sorted(strengths)

    def test_contains_all_enum_values(self) -> None:
        """Test that list contains all enum values."""
        strengths = list_watermark_strengths()
        enum_values = [s.value for s in WatermarkStrength]
        assert len(strengths) == len(enum_values)
        for v in enum_values:
            assert v in strengths


class TestListDetectionMethods:
    """Tests for list_detection_methods function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        methods = list_detection_methods()
        assert isinstance(methods, list)

    def test_contains_expected_methods(self) -> None:
        """Test that list contains expected methods."""
        methods = list_detection_methods()
        assert "z_score" in methods
        assert "log_likelihood" in methods
        assert "perplexity" in methods
        assert "entropy" in methods

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        methods = list_detection_methods()
        assert methods == sorted(methods)

    def test_contains_all_enum_values(self) -> None:
        """Test that list contains all enum values."""
        methods = list_detection_methods()
        enum_values = [m.value for m in DetectionMethod]
        assert len(methods) == len(enum_values)
        for v in enum_values:
            assert v in methods


class TestGetWatermarkType:
    """Tests for get_watermark_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("soft", WatermarkType.SOFT),
            ("hard", WatermarkType.HARD),
            ("semantic", WatermarkType.SEMANTIC),
            ("statistical", WatermarkType.STATISTICAL),
        ],
    )
    def test_get_valid_type(self, name: str, expected: WatermarkType) -> None:
        """Test getting valid watermark types."""
        assert get_watermark_type(name) == expected

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid watermark type: invalid"):
            get_watermark_type("invalid")

    def test_empty_type_raises_error(self) -> None:
        """Test that empty type raises ValueError."""
        with pytest.raises(ValueError, match="invalid watermark type:"):
            get_watermark_type("")


class TestGetWatermarkStrength:
    """Tests for get_watermark_strength function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("low", WatermarkStrength.LOW),
            ("medium", WatermarkStrength.MEDIUM),
            ("high", WatermarkStrength.HIGH),
        ],
    )
    def test_get_valid_strength(self, name: str, expected: WatermarkStrength) -> None:
        """Test getting valid watermark strengths."""
        assert get_watermark_strength(name) == expected

    def test_invalid_strength_raises_error(self) -> None:
        """Test that invalid strength raises ValueError."""
        with pytest.raises(ValueError, match="invalid watermark strength: invalid"):
            get_watermark_strength("invalid")

    def test_empty_strength_raises_error(self) -> None:
        """Test that empty strength raises ValueError."""
        with pytest.raises(ValueError, match="invalid watermark strength:"):
            get_watermark_strength("")


class TestGetDetectionMethod:
    """Tests for get_detection_method function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("z_score", DetectionMethod.Z_SCORE),
            ("log_likelihood", DetectionMethod.LOG_LIKELIHOOD),
            ("perplexity", DetectionMethod.PERPLEXITY),
            ("entropy", DetectionMethod.ENTROPY),
        ],
    )
    def test_get_valid_method(self, name: str, expected: DetectionMethod) -> None:
        """Test getting valid detection methods."""
        assert get_detection_method(name) == expected

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="invalid detection method: invalid"):
            get_detection_method("invalid")

    def test_empty_method_raises_error(self) -> None:
        """Test that empty method raises ValueError."""
        with pytest.raises(ValueError, match="invalid detection method:"):
            get_detection_method("")


class TestCalculateZScore:
    """Tests for calculate_z_score function."""

    def test_expected_count_equals_observed(self) -> None:
        """Test z-score when observed equals expected."""
        z = calculate_z_score(50, 100, 0.5)
        assert z == pytest.approx(0.0)

    def test_positive_z_score(self) -> None:
        """Test positive z-score."""
        z = calculate_z_score(75, 100, 0.5)
        assert z == pytest.approx(5.0)

    def test_negative_z_score(self) -> None:
        """Test negative z-score."""
        z = calculate_z_score(25, 100, 0.5)
        assert z == pytest.approx(-5.0)

    def test_intermediate_z_score(self) -> None:
        """Test intermediate z-score."""
        z = calculate_z_score(60, 100, 0.5)
        assert z == pytest.approx(2.0)

    def test_different_gamma(self) -> None:
        """Test z-score with different gamma value."""
        # Expected = 100 * 0.25 = 25
        # Variance = 100 * 0.25 * 0.75 = 18.75
        # Std = sqrt(18.75) = 4.33
        # z = (50 - 25) / 4.33 = 5.77
        z = calculate_z_score(50, 100, 0.25)
        assert z == pytest.approx(5.773502691896258)

    @pytest.mark.parametrize("total_tokens", [0, -1, -10])
    def test_zero_or_negative_total_tokens_raises_error(
        self, total_tokens: int
    ) -> None:
        """Test that zero or negative total_tokens raises ValueError."""
        with pytest.raises(ValueError, match="total_tokens must be positive"):
            calculate_z_score(10, total_tokens, 0.5)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, -0.5, 1.5, 2.0])
    def test_invalid_gamma_raises_error(self, gamma: float) -> None:
        """Test that invalid gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            calculate_z_score(10, 100, gamma)

    @pytest.mark.parametrize("green_count", [-1, -10, -100])
    def test_negative_green_token_count_raises_error(self, green_count: int) -> None:
        """Test that negative green_token_count raises ValueError."""
        with pytest.raises(ValueError, match="green_token_count cannot be negative"):
            calculate_z_score(green_count, 100, 0.5)

    def test_green_token_count_exceeds_total_raises_error(self) -> None:
        """Test that green_token_count exceeding total raises ValueError."""
        with pytest.raises(ValueError, match="green_token_count cannot exceed"):
            calculate_z_score(150, 100, 0.5)

    def test_boundary_green_token_count(self) -> None:
        """Test boundary cases for green_token_count."""
        # Zero green tokens
        z = calculate_z_score(0, 100, 0.5)
        assert z == pytest.approx(-10.0)

        # All tokens are green
        z = calculate_z_score(100, 100, 0.5)
        assert z == pytest.approx(10.0)


class TestEstimateDetectability:
    """Tests for estimate_detectability function."""

    def test_returns_probability_in_range(self) -> None:
        """Test that result is between 0 and 1."""
        prob = estimate_detectability(100, 0.5, 2.0)
        assert 0 <= prob <= 1

    def test_more_tokens_increases_detectability(self) -> None:
        """Test that more tokens increases detectability."""
        prob_short = estimate_detectability(50, 0.5, 2.0)
        prob_long = estimate_detectability(500, 0.5, 2.0)
        assert prob_long >= prob_short

    def test_higher_delta_increases_detectability(self) -> None:
        """Test that higher delta increases detectability."""
        prob_low_delta = estimate_detectability(100, 0.5, 1.0)
        prob_high_delta = estimate_detectability(100, 0.5, 4.0)
        assert prob_high_delta >= prob_low_delta

    def test_higher_threshold_decreases_detectability(self) -> None:
        """Test that higher threshold decreases detectability."""
        prob_low_threshold = estimate_detectability(100, 0.5, 2.0, threshold=2.0)
        prob_high_threshold = estimate_detectability(100, 0.5, 2.0, threshold=8.0)
        assert prob_low_threshold >= prob_high_threshold

    @pytest.mark.parametrize("num_tokens", [0, -1, -10])
    def test_zero_or_negative_num_tokens_raises_error(self, num_tokens: int) -> None:
        """Test that zero or negative num_tokens raises ValueError."""
        with pytest.raises(ValueError, match="num_tokens must be positive"):
            estimate_detectability(num_tokens, 0.5, 2.0)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, -0.5, 1.5, 2.0])
    def test_invalid_gamma_raises_error(self, gamma: float) -> None:
        """Test that invalid gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            estimate_detectability(100, gamma, 2.0)

    @pytest.mark.parametrize("delta", [0.0, -1.0, -0.5])
    def test_invalid_delta_raises_error(self, delta: float) -> None:
        """Test that non-positive delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be positive"):
            estimate_detectability(100, 0.5, delta)

    @pytest.mark.parametrize("threshold", [0.0, -1.0, -0.5])
    def test_invalid_threshold_raises_error(self, threshold: float) -> None:
        """Test that non-positive threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            estimate_detectability(100, 0.5, 2.0, threshold=threshold)

    def test_very_large_num_tokens(self) -> None:
        """Test with very large number of tokens."""
        prob = estimate_detectability(10000, 0.5, 2.0)
        assert prob == pytest.approx(1.0, abs=0.01)

    def test_very_small_num_tokens(self) -> None:
        """Test with very small number of tokens."""
        prob = estimate_detectability(1, 0.5, 2.0)
        assert 0 <= prob <= 1

    def test_very_high_threshold(self) -> None:
        """Test with very high threshold."""
        prob = estimate_detectability(100, 0.5, 2.0, threshold=100.0)
        assert prob == pytest.approx(0.0, abs=0.01)
