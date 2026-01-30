"""Tests for training collators functionality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.training.collators import (
    VALID_COLLATOR_TYPES,
    VALID_PADDING_SIDES,
    VALID_PADDING_STRATEGIES,
    VALID_TRUNCATION_STRATEGIES,
    CollatorConfig,
    CollatorStats,
    CollatorType,
    PaddingConfig,
    PaddingSide,
    PaddingStrategy,
    TruncationConfig,
    TruncationStrategy,
    calculate_padding_stats,
    create_collator_config,
    create_collator_stats,
    create_padding_config,
    create_truncation_config,
    estimate_memory_per_batch,
    format_collator_stats,
    get_collator_type,
    get_padding_side,
    get_padding_strategy,
    get_recommended_collator_config,
    get_truncation_strategy,
    list_collator_types,
    list_padding_sides,
    list_padding_strategies,
    list_truncation_strategies,
    optimize_batch_padding,
    validate_collator_config,
    validate_collator_output,
    validate_collator_stats,
    validate_padding_config,
    validate_truncation_config,
)


class TestPaddingStrategy:
    """Tests for PaddingStrategy enum."""

    def test_longest_value(self) -> None:
        """Test LONGEST enum value."""
        assert PaddingStrategy.LONGEST.value == "longest"

    def test_max_length_value(self) -> None:
        """Test MAX_LENGTH enum value."""
        assert PaddingStrategy.MAX_LENGTH.value == "max_length"

    def test_do_not_pad_value(self) -> None:
        """Test DO_NOT_PAD enum value."""
        assert PaddingStrategy.DO_NOT_PAD.value == "do_not_pad"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_PADDING_STRATEGIES."""
        for strategy in PaddingStrategy:
            assert strategy.value in VALID_PADDING_STRATEGIES


class TestTruncationStrategy:
    """Tests for TruncationStrategy enum."""

    def test_longest_first_value(self) -> None:
        """Test LONGEST_FIRST enum value."""
        assert TruncationStrategy.LONGEST_FIRST.value == "longest_first"

    def test_only_first_value(self) -> None:
        """Test ONLY_FIRST enum value."""
        assert TruncationStrategy.ONLY_FIRST.value == "only_first"

    def test_only_second_value(self) -> None:
        """Test ONLY_SECOND enum value."""
        assert TruncationStrategy.ONLY_SECOND.value == "only_second"

    def test_do_not_truncate_value(self) -> None:
        """Test DO_NOT_TRUNCATE enum value."""
        assert TruncationStrategy.DO_NOT_TRUNCATE.value == "do_not_truncate"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_TRUNCATION_STRATEGIES."""
        for strategy in TruncationStrategy:
            assert strategy.value in VALID_TRUNCATION_STRATEGIES


class TestCollatorType:
    """Tests for CollatorType enum."""

    def test_default_value(self) -> None:
        """Test DEFAULT enum value."""
        assert CollatorType.DEFAULT.value == "default"

    def test_language_modeling_value(self) -> None:
        """Test LANGUAGE_MODELING enum value."""
        assert CollatorType.LANGUAGE_MODELING.value == "language_modeling"

    def test_seq2seq_value(self) -> None:
        """Test SEQ2SEQ enum value."""
        assert CollatorType.SEQ2SEQ.value == "seq2seq"

    def test_completion_only_value(self) -> None:
        """Test COMPLETION_ONLY enum value."""
        assert CollatorType.COMPLETION_ONLY.value == "completion_only"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_COLLATOR_TYPES."""
        for collator_type in CollatorType:
            assert collator_type.value in VALID_COLLATOR_TYPES


class TestPaddingSide:
    """Tests for PaddingSide enum."""

    def test_left_value(self) -> None:
        """Test LEFT enum value."""
        assert PaddingSide.LEFT.value == "left"

    def test_right_value(self) -> None:
        """Test RIGHT enum value."""
        assert PaddingSide.RIGHT.value == "right"

    def test_all_values_in_valid_set(self) -> None:
        """Test all enum values are in VALID_PADDING_SIDES."""
        for side in PaddingSide:
            assert side.value in VALID_PADDING_SIDES


class TestPaddingConfig:
    """Tests for PaddingConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a PaddingConfig."""
        config = PaddingConfig(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
            pad_to_multiple_of=8,
            padding_side=PaddingSide.RIGHT,
        )
        assert config.strategy == PaddingStrategy.LONGEST
        assert config.max_length == 512
        assert config.pad_to_multiple_of == 8
        assert config.padding_side == PaddingSide.RIGHT

    def test_frozen(self) -> None:
        """Test that PaddingConfig is immutable."""
        config = PaddingConfig(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
            pad_to_multiple_of=8,
            padding_side=PaddingSide.RIGHT,
        )
        with pytest.raises(AttributeError):
            config.strategy = PaddingStrategy.MAX_LENGTH  # type: ignore[misc]


class TestTruncationConfig:
    """Tests for TruncationConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a TruncationConfig."""
        config = TruncationConfig(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
            stride=0,
        )
        assert config.strategy == TruncationStrategy.LONGEST_FIRST
        assert config.max_length == 512
        assert config.stride == 0

    def test_frozen(self) -> None:
        """Test that TruncationConfig is immutable."""
        config = TruncationConfig(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
            stride=0,
        )
        with pytest.raises(AttributeError):
            config.strategy = TruncationStrategy.ONLY_FIRST  # type: ignore[misc]


class TestCollatorConfig:
    """Tests for CollatorConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a CollatorConfig."""
        padding = PaddingConfig(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
            pad_to_multiple_of=8,
            padding_side=PaddingSide.RIGHT,
        )
        truncation = TruncationConfig(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
            stride=0,
        )
        config = CollatorConfig(
            collator_type=CollatorType.DEFAULT,
            padding_config=padding,
            truncation_config=truncation,
            mlm_probability=0.0,
            return_tensors="pt",
        )
        assert config.collator_type == CollatorType.DEFAULT
        assert config.mlm_probability == pytest.approx(0.0)
        assert config.return_tensors == "pt"

    def test_frozen(self) -> None:
        """Test that CollatorConfig is immutable."""
        config = create_collator_config()
        with pytest.raises(AttributeError):
            config.collator_type = CollatorType.SEQ2SEQ  # type: ignore[misc]


class TestCollatorStats:
    """Tests for CollatorStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating CollatorStats."""
        stats = CollatorStats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        assert stats.avg_length == pytest.approx(256.5)
        assert stats.max_length == 512
        assert stats.padding_ratio == pytest.approx(0.25)
        assert stats.truncation_ratio == pytest.approx(0.1)

    def test_frozen(self) -> None:
        """Test that CollatorStats is immutable."""
        stats = CollatorStats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        with pytest.raises(AttributeError):
            stats.avg_length = 300.0  # type: ignore[misc]


class TestValidatePaddingConfig:
    """Tests for validate_padding_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_padding_config()
        validate_padding_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_padding_config(None)  # type: ignore[arg-type]

    def test_max_length_required_for_max_length_strategy(self) -> None:
        """Test that max_length is required for MAX_LENGTH strategy."""
        config = PaddingConfig(
            strategy=PaddingStrategy.MAX_LENGTH,
            max_length=None,
            pad_to_multiple_of=None,
            padding_side=PaddingSide.RIGHT,
        )
        with pytest.raises(ValueError, match="max_length is required"):
            validate_padding_config(config)

    def test_negative_max_length_raises_error(self) -> None:
        """Test that negative max_length raises ValueError."""
        config = PaddingConfig(
            strategy=PaddingStrategy.LONGEST,
            max_length=-1,
            pad_to_multiple_of=None,
            padding_side=PaddingSide.RIGHT,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_padding_config(config)

    def test_negative_pad_to_multiple_raises_error(self) -> None:
        """Test that negative pad_to_multiple_of raises ValueError."""
        config = PaddingConfig(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
            pad_to_multiple_of=-1,
            padding_side=PaddingSide.RIGHT,
        )
        with pytest.raises(ValueError, match="pad_to_multiple_of must be positive"):
            validate_padding_config(config)

    def test_zero_pad_to_multiple_raises_error(self) -> None:
        """Test that zero pad_to_multiple_of raises ValueError."""
        config = PaddingConfig(
            strategy=PaddingStrategy.LONGEST,
            max_length=512,
            pad_to_multiple_of=0,
            padding_side=PaddingSide.RIGHT,
        )
        with pytest.raises(ValueError, match="pad_to_multiple_of must be positive"):
            validate_padding_config(config)


class TestValidateTruncationConfig:
    """Tests for validate_truncation_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_truncation_config()
        validate_truncation_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_truncation_config(None)  # type: ignore[arg-type]

    def test_negative_max_length_raises_error(self) -> None:
        """Test that negative max_length raises ValueError."""
        config = TruncationConfig(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=-1,
            stride=0,
        )
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_truncation_config(config)

    def test_negative_stride_raises_error(self) -> None:
        """Test that negative stride raises ValueError."""
        config = TruncationConfig(
            strategy=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
            stride=-1,
        )
        with pytest.raises(ValueError, match="stride cannot be negative"):
            validate_truncation_config(config)

    def test_do_not_truncate_with_any_max_length(self) -> None:
        """Test that DO_NOT_TRUNCATE allows any max_length."""
        config = TruncationConfig(
            strategy=TruncationStrategy.DO_NOT_TRUNCATE,
            max_length=None,
            stride=0,
        )
        validate_truncation_config(config)  # Should not raise


class TestValidateCollatorConfig:
    """Tests for validate_collator_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = create_collator_config()
        validate_collator_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_collator_config(None)  # type: ignore[arg-type]

    def test_invalid_mlm_probability_too_high(self) -> None:
        """Test that mlm_probability > 1.0 raises ValueError."""
        padding = create_padding_config()
        truncation = create_truncation_config()
        config = CollatorConfig(
            collator_type=CollatorType.LANGUAGE_MODELING,
            padding_config=padding,
            truncation_config=truncation,
            mlm_probability=1.5,
            return_tensors="pt",
        )
        with pytest.raises(ValueError, match="mlm_probability must be between"):
            validate_collator_config(config)

    def test_invalid_mlm_probability_negative(self) -> None:
        """Test that negative mlm_probability raises ValueError."""
        padding = create_padding_config()
        truncation = create_truncation_config()
        config = CollatorConfig(
            collator_type=CollatorType.LANGUAGE_MODELING,
            padding_config=padding,
            truncation_config=truncation,
            mlm_probability=-0.1,
            return_tensors="pt",
        )
        with pytest.raises(ValueError, match="mlm_probability must be between"):
            validate_collator_config(config)

    def test_invalid_return_tensors(self) -> None:
        """Test that invalid return_tensors raises ValueError."""
        padding = create_padding_config()
        truncation = create_truncation_config()
        config = CollatorConfig(
            collator_type=CollatorType.DEFAULT,
            padding_config=padding,
            truncation_config=truncation,
            mlm_probability=0.0,
            return_tensors="invalid",
        )
        with pytest.raises(ValueError, match="return_tensors must be one of"):
            validate_collator_config(config)

    def test_valid_return_tensors_none(self) -> None:
        """Test that None is a valid return_tensors value."""
        config = create_collator_config(return_tensors=None)
        validate_collator_config(config)  # Should not raise

    def test_valid_return_tensors_options(self) -> None:
        """Test all valid return_tensors options."""
        for tensors in ["pt", "tf", "np", None]:
            config = create_collator_config(return_tensors=tensors)
            validate_collator_config(config)  # Should not raise


class TestValidateCollatorStats:
    """Tests for validate_collator_stats function."""

    def test_valid_stats(self) -> None:
        """Test validation of valid stats."""
        stats = create_collator_stats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        validate_collator_stats(stats)  # Should not raise

    def test_none_stats_raises_error(self) -> None:
        """Test that None stats raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_collator_stats(None)  # type: ignore[arg-type]

    def test_negative_avg_length_raises_error(self) -> None:
        """Test that negative avg_length raises ValueError."""
        stats = CollatorStats(
            avg_length=-10.0,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        with pytest.raises(ValueError, match="avg_length cannot be negative"):
            validate_collator_stats(stats)

    def test_negative_max_length_raises_error(self) -> None:
        """Test that negative max_length raises ValueError."""
        stats = CollatorStats(
            avg_length=256.5,
            max_length=-1,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        with pytest.raises(ValueError, match="max_length cannot be negative"):
            validate_collator_stats(stats)

    def test_invalid_padding_ratio_high(self) -> None:
        """Test that padding_ratio > 1.0 raises ValueError."""
        stats = CollatorStats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=1.5,
            truncation_ratio=0.1,
        )
        with pytest.raises(ValueError, match="padding_ratio must be between"):
            validate_collator_stats(stats)

    def test_invalid_padding_ratio_negative(self) -> None:
        """Test that negative padding_ratio raises ValueError."""
        stats = CollatorStats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=-0.1,
            truncation_ratio=0.1,
        )
        with pytest.raises(ValueError, match="padding_ratio must be between"):
            validate_collator_stats(stats)

    def test_invalid_truncation_ratio_high(self) -> None:
        """Test that truncation_ratio > 1.0 raises ValueError."""
        stats = CollatorStats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=1.5,
        )
        with pytest.raises(ValueError, match="truncation_ratio must be between"):
            validate_collator_stats(stats)


class TestCreatePaddingConfig:
    """Tests for create_padding_config function."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = create_padding_config()
        assert config.strategy == PaddingStrategy.LONGEST
        assert config.max_length == 512
        assert config.pad_to_multiple_of == 8
        assert config.padding_side == PaddingSide.RIGHT

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = create_padding_config(
            strategy=PaddingStrategy.MAX_LENGTH,
            max_length=1024,
            pad_to_multiple_of=16,
            padding_side=PaddingSide.LEFT,
        )
        assert config.strategy == PaddingStrategy.MAX_LENGTH
        assert config.max_length == 1024
        assert config.pad_to_multiple_of == 16
        assert config.padding_side == PaddingSide.LEFT

    def test_string_strategy(self) -> None:
        """Test with string strategy."""
        config = create_padding_config(strategy="max_length")
        assert config.strategy == PaddingStrategy.MAX_LENGTH

    def test_string_padding_side(self) -> None:
        """Test with string padding_side."""
        config = create_padding_config(padding_side="left")
        assert config.padding_side == PaddingSide.LEFT

    def test_invalid_max_length_raises_error(self) -> None:
        """Test that invalid max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            create_padding_config(max_length=-1)


class TestCreateTruncationConfig:
    """Tests for create_truncation_config function."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = create_truncation_config()
        assert config.strategy == TruncationStrategy.LONGEST_FIRST
        assert config.max_length == 512
        assert config.stride == 0

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = create_truncation_config(
            strategy=TruncationStrategy.ONLY_FIRST,
            max_length=256,
            stride=128,
        )
        assert config.strategy == TruncationStrategy.ONLY_FIRST
        assert config.max_length == 256
        assert config.stride == 128

    def test_string_strategy(self) -> None:
        """Test with string strategy."""
        config = create_truncation_config(strategy="only_second")
        assert config.strategy == TruncationStrategy.ONLY_SECOND

    def test_invalid_stride_raises_error(self) -> None:
        """Test that invalid stride raises ValueError."""
        with pytest.raises(ValueError, match="stride cannot be negative"):
            create_truncation_config(stride=-1)


class TestCreateCollatorConfig:
    """Tests for create_collator_config function."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = create_collator_config()
        assert config.collator_type == CollatorType.DEFAULT
        assert config.mlm_probability == pytest.approx(0.0)
        assert config.return_tensors == "pt"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = create_collator_config(
            collator_type=CollatorType.LANGUAGE_MODELING,
            mlm_probability=0.15,
            return_tensors="tf",
        )
        assert config.collator_type == CollatorType.LANGUAGE_MODELING
        assert config.mlm_probability == pytest.approx(0.15)
        assert config.return_tensors == "tf"

    def test_string_collator_type(self) -> None:
        """Test with string collator_type."""
        config = create_collator_config(collator_type="seq2seq")
        assert config.collator_type == CollatorType.SEQ2SEQ

    def test_custom_padding_config(self) -> None:
        """Test with custom padding config."""
        padding = create_padding_config(max_length=1024)
        config = create_collator_config(padding_config=padding)
        assert config.padding_config.max_length == 1024

    def test_custom_truncation_config(self) -> None:
        """Test with custom truncation config."""
        truncation = create_truncation_config(stride=64)
        config = create_collator_config(truncation_config=truncation)
        assert config.truncation_config.stride == 64

    def test_invalid_mlm_probability_raises_error(self) -> None:
        """Test that invalid mlm_probability raises ValueError."""
        with pytest.raises(ValueError, match="mlm_probability must be between"):
            create_collator_config(mlm_probability=1.5)


class TestCreateCollatorStats:
    """Tests for create_collator_stats function."""

    def test_defaults(self) -> None:
        """Test default values."""
        stats = create_collator_stats()
        assert stats.avg_length == pytest.approx(0.0)
        assert stats.max_length == 0
        assert stats.padding_ratio == pytest.approx(0.0)
        assert stats.truncation_ratio == pytest.approx(0.0)

    def test_custom_values(self) -> None:
        """Test custom values."""
        stats = create_collator_stats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        assert stats.avg_length == pytest.approx(256.5)
        assert stats.max_length == 512
        assert stats.padding_ratio == pytest.approx(0.25)
        assert stats.truncation_ratio == pytest.approx(0.1)

    def test_invalid_avg_length_raises_error(self) -> None:
        """Test that invalid avg_length raises ValueError."""
        with pytest.raises(ValueError, match="avg_length cannot be negative"):
            create_collator_stats(avg_length=-10.0)


class TestListFunctions:
    """Tests for list_* functions."""

    def test_list_padding_strategies(self) -> None:
        """Test list_padding_strategies returns all strategies."""
        strategies = list_padding_strategies()
        assert isinstance(strategies, list)
        assert "longest" in strategies
        assert "max_length" in strategies
        assert "do_not_pad" in strategies
        assert strategies == sorted(strategies)

    def test_list_truncation_strategies(self) -> None:
        """Test list_truncation_strategies returns all strategies."""
        strategies = list_truncation_strategies()
        assert isinstance(strategies, list)
        assert "longest_first" in strategies
        assert "do_not_truncate" in strategies
        assert strategies == sorted(strategies)

    def test_list_collator_types(self) -> None:
        """Test list_collator_types returns all types."""
        types = list_collator_types()
        assert isinstance(types, list)
        assert "default" in types
        assert "language_modeling" in types
        assert types == sorted(types)

    def test_list_padding_sides(self) -> None:
        """Test list_padding_sides returns all sides."""
        sides = list_padding_sides()
        assert isinstance(sides, list)
        assert "left" in sides
        assert "right" in sides
        assert sides == sorted(sides)


class TestGetFunctions:
    """Tests for get_* functions."""

    def test_get_padding_strategy_valid(self) -> None:
        """Test get_padding_strategy with valid input."""
        assert get_padding_strategy("longest") == PaddingStrategy.LONGEST
        assert get_padding_strategy("max_length") == PaddingStrategy.MAX_LENGTH

    def test_get_padding_strategy_invalid(self) -> None:
        """Test get_padding_strategy with invalid input."""
        with pytest.raises(ValueError, match="padding_strategy must be one of"):
            get_padding_strategy("invalid")

    def test_get_truncation_strategy_valid(self) -> None:
        """Test get_truncation_strategy with valid input."""
        assert (
            get_truncation_strategy("longest_first") == TruncationStrategy.LONGEST_FIRST
        )
        assert get_truncation_strategy("only_first") == TruncationStrategy.ONLY_FIRST

    def test_get_truncation_strategy_invalid(self) -> None:
        """Test get_truncation_strategy with invalid input."""
        with pytest.raises(ValueError, match="truncation_strategy must be one of"):
            get_truncation_strategy("invalid")

    def test_get_collator_type_valid(self) -> None:
        """Test get_collator_type with valid input."""
        assert get_collator_type("default") == CollatorType.DEFAULT
        assert get_collator_type("seq2seq") == CollatorType.SEQ2SEQ

    def test_get_collator_type_invalid(self) -> None:
        """Test get_collator_type with invalid input."""
        with pytest.raises(ValueError, match="collator_type must be one of"):
            get_collator_type("invalid")

    def test_get_padding_side_valid(self) -> None:
        """Test get_padding_side with valid input."""
        assert get_padding_side("left") == PaddingSide.LEFT
        assert get_padding_side("right") == PaddingSide.RIGHT

    def test_get_padding_side_invalid(self) -> None:
        """Test get_padding_side with invalid input."""
        with pytest.raises(ValueError, match="padding_side must be one of"):
            get_padding_side("invalid")


class TestCalculatePaddingStats:
    """Tests for calculate_padding_stats function."""

    def test_basic_calculation(self) -> None:
        """Test basic padding stats calculation."""
        stats = calculate_padding_stats([100, 200, 300], 300)
        assert stats.avg_length == pytest.approx(200.0)
        assert stats.max_length == 300
        assert stats.padding_ratio == pytest.approx(1 / 3)

    def test_no_padding_needed(self) -> None:
        """Test when no padding is needed."""
        stats = calculate_padding_stats([128, 128, 128], 128)
        assert stats.padding_ratio == pytest.approx(0.0)

    def test_with_truncation(self) -> None:
        """Test with truncation detection."""
        stats = calculate_padding_stats([100, 200, 300], 150, original_max_length=300)
        assert stats.truncation_ratio == pytest.approx(2 / 3)  # 200, 300 > 150

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="sequence_lengths cannot be empty"):
            calculate_padding_stats([], 100)

    def test_zero_padded_length_raises_error(self) -> None:
        """Test that zero padded_length raises ValueError."""
        with pytest.raises(ValueError, match="padded_length must be positive"):
            calculate_padding_stats([100, 200], 0)

    def test_negative_padded_length_raises_error(self) -> None:
        """Test that negative padded_length raises ValueError."""
        with pytest.raises(ValueError, match="padded_length must be positive"):
            calculate_padding_stats([100, 200], -1)

    @given(
        lengths=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=20)
    def test_padding_ratio_bounds(self, lengths: list[int]) -> None:
        """Test that padding ratio is always between 0 and 1."""
        max_len = max(lengths)
        stats = calculate_padding_stats(lengths, max_len)
        assert 0.0 <= stats.padding_ratio <= 1.0


class TestEstimateMemoryPerBatch:
    """Tests for estimate_memory_per_batch function."""

    def test_basic_estimate(self) -> None:
        """Test basic memory estimation."""
        mem = estimate_memory_per_batch(32, 512, 768, 4)
        assert mem > 0

    def test_fp16_smaller_than_fp32(self) -> None:
        """Test that FP16 uses less memory than FP32."""
        mem_fp32 = estimate_memory_per_batch(32, 512, 768, 4)
        mem_fp16 = estimate_memory_per_batch(32, 512, 768, 2)
        assert mem_fp16 < mem_fp32

    def test_without_activations(self) -> None:
        """Test memory without activations."""
        mem_with = estimate_memory_per_batch(32, 512, 768, 4, include_activations=True)
        mem_without = estimate_memory_per_batch(
            32, 512, 768, 4, include_activations=False
        )
        assert mem_without < mem_with

    def test_larger_batch_uses_more_memory(self) -> None:
        """Test that larger batch uses more memory."""
        mem_small = estimate_memory_per_batch(16, 512, 768, 4)
        mem_large = estimate_memory_per_batch(32, 512, 768, 4)
        assert mem_large > mem_small

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_memory_per_batch(0, 512, 768, 4)

    def test_zero_sequence_length_raises_error(self) -> None:
        """Test that zero sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            estimate_memory_per_batch(32, 0, 768, 4)

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_memory_per_batch(32, 512, 0, 4)

    def test_zero_dtype_bytes_raises_error(self) -> None:
        """Test that zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_memory_per_batch(32, 512, 768, 0)


class TestOptimizeBatchPadding:
    """Tests for optimize_batch_padding function."""

    def test_basic_optimization(self) -> None:
        """Test basic batch padding optimization."""
        length, waste = optimize_batch_padding([100, 200, 300, 500], 512)
        assert length <= 512
        assert 0.0 <= waste <= 1.0

    def test_uniform_lengths(self) -> None:
        """Test with uniform lengths."""
        length, waste = optimize_batch_padding([128, 128, 128], 512)
        assert length == 128
        assert waste == pytest.approx(0.0)

    def test_respects_pad_to_multiple(self) -> None:
        """Test that result respects pad_to_multiple_of."""
        length, _ = optimize_batch_padding([100, 200], 512, pad_to_multiple_of=64)
        assert length % 64 == 0 or length == 512

    def test_no_pad_to_multiple(self) -> None:
        """Test without pad_to_multiple_of."""
        length, _ = optimize_batch_padding([100, 200], 512, pad_to_multiple_of=None)
        assert length == 200

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="sequence_lengths cannot be empty"):
            optimize_batch_padding([], 512)

    def test_zero_max_allowed_raises_error(self) -> None:
        """Test that zero max_allowed_length raises ValueError."""
        with pytest.raises(ValueError, match="max_allowed_length must be positive"):
            optimize_batch_padding([100, 200], 0)

    def test_caps_at_max_allowed(self) -> None:
        """Test that result is capped at max_allowed_length."""
        length, _ = optimize_batch_padding([1000, 2000], 512)
        assert length <= 512

    @given(
        lengths=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=20,
        ),
        max_len=st.integers(min_value=1, max_value=2000),
    )
    @settings(max_examples=20)
    def test_waste_ratio_bounds(self, lengths: list[int], max_len: int) -> None:
        """Test that waste ratio is always between 0 and 1."""
        _, waste = optimize_batch_padding(lengths, max_len)
        assert 0.0 <= waste <= 1.0


class TestValidateCollatorOutput:
    """Tests for validate_collator_output function."""

    def test_valid_batch(self) -> None:
        """Test validation of valid batch."""
        batch = {
            "input_ids": np.zeros((4, 128)),
            "attention_mask": np.ones((4, 128)),
        }
        assert validate_collator_output(batch) is True

    def test_with_expected_keys(self) -> None:
        """Test with expected keys."""
        batch = {
            "input_ids": np.zeros((4, 128)),
            "attention_mask": np.ones((4, 128)),
        }
        assert validate_collator_output(
            batch, expected_keys=["input_ids", "attention_mask"]
        )

    def test_missing_keys_raises_error(self) -> None:
        """Test that missing keys raises ValueError."""
        batch = {"input_ids": np.zeros((4, 128))}
        with pytest.raises(ValueError, match="Missing expected keys"):
            validate_collator_output(batch, expected_keys=["labels"])

    def test_with_expected_batch_size(self) -> None:
        """Test with expected batch size."""
        batch = {"input_ids": np.zeros((4, 128))}
        assert validate_collator_output(batch, expected_batch_size=4)

    def test_wrong_batch_size_raises_error(self) -> None:
        """Test that wrong batch size raises ValueError."""
        batch = {"input_ids": np.zeros((4, 128))}
        with pytest.raises(ValueError, match="Expected batch size"):
            validate_collator_output(batch, expected_batch_size=8)

    def test_with_expected_sequence_length(self) -> None:
        """Test with expected sequence length."""
        batch = {"input_ids": np.zeros((4, 128))}
        assert validate_collator_output(batch, expected_sequence_length=128)

    def test_wrong_sequence_length_raises_error(self) -> None:
        """Test that wrong sequence length raises ValueError."""
        batch = {"input_ids": np.zeros((4, 128))}
        with pytest.raises(ValueError, match="Expected sequence length"):
            validate_collator_output(batch, expected_sequence_length=256)

    def test_none_batch_raises_error(self) -> None:
        """Test that None batch raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_collator_output(None)  # type: ignore[arg-type]

    def test_empty_batch_raises_error(self) -> None:
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_collator_output({})


class TestFormatCollatorStats:
    """Tests for format_collator_stats function."""

    def test_format_stats(self) -> None:
        """Test formatting of stats."""
        stats = create_collator_stats(
            avg_length=256.5,
            max_length=512,
            padding_ratio=0.25,
            truncation_ratio=0.1,
        )
        formatted = format_collator_stats(stats)
        assert "Avg Length: 256.50" in formatted
        assert "Max Length: 512" in formatted
        assert "Padding: 25.0%" in formatted
        assert "Truncation: 10.0%" in formatted

    def test_format_zero_stats(self) -> None:
        """Test formatting of zero stats."""
        stats = create_collator_stats()
        formatted = format_collator_stats(stats)
        assert "Avg Length: 0.00" in formatted
        assert "Max Length: 0" in formatted
        assert "Padding: 0.0%" in formatted


class TestGetRecommendedCollatorConfig:
    """Tests for get_recommended_collator_config function."""

    def test_classification_config(self) -> None:
        """Test recommended config for classification."""
        config = get_recommended_collator_config("classification")
        assert config.collator_type == CollatorType.DEFAULT
        assert config.padding_config.strategy == PaddingStrategy.LONGEST

    def test_generation_config(self) -> None:
        """Test recommended config for generation."""
        config = get_recommended_collator_config("generation")
        assert config.collator_type == CollatorType.LANGUAGE_MODELING
        assert config.padding_config.padding_side == PaddingSide.LEFT

    def test_seq2seq_config(self) -> None:
        """Test recommended config for seq2seq."""
        config = get_recommended_collator_config("seq2seq")
        assert config.collator_type == CollatorType.SEQ2SEQ

    def test_masked_lm_config(self) -> None:
        """Test recommended config for masked_lm."""
        config = get_recommended_collator_config("masked_lm")
        assert config.collator_type == CollatorType.LANGUAGE_MODELING
        assert config.mlm_probability == pytest.approx(0.15)

    def test_causal_lm_config(self) -> None:
        """Test recommended config for causal_lm."""
        config = get_recommended_collator_config("causal_lm")
        assert config.collator_type == CollatorType.LANGUAGE_MODELING
        assert config.mlm_probability == pytest.approx(0.0)

    def test_completion_config(self) -> None:
        """Test recommended config for completion."""
        config = get_recommended_collator_config("completion")
        assert config.collator_type == CollatorType.COMPLETION_ONLY

    def test_invalid_task_type_raises_error(self) -> None:
        """Test that invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            get_recommended_collator_config("unknown")

    def test_all_configs_are_valid(self) -> None:
        """Test that all recommended configs pass validation."""
        task_types = [
            "classification",
            "generation",
            "seq2seq",
            "masked_lm",
            "causal_lm",
            "completion",
        ]
        for task_type in task_types:
            config = get_recommended_collator_config(task_type)
            validate_collator_config(config)  # Should not raise


class TestPropertyBasedTests:
    """Property-based tests using Hypothesis."""

    @given(
        max_length=st.integers(min_value=1, max_value=10000),
        pad_to_multiple_of=st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=20)
    def test_padding_config_creation(
        self, max_length: int, pad_to_multiple_of: int
    ) -> None:
        """Test padding config creation with various inputs."""
        config = create_padding_config(
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        assert config.max_length == max_length
        assert config.pad_to_multiple_of == pad_to_multiple_of

    @given(
        max_length=st.integers(min_value=1, max_value=10000),
        stride=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20)
    def test_truncation_config_creation(self, max_length: int, stride: int) -> None:
        """Test truncation config creation with various inputs."""
        config = create_truncation_config(
            max_length=max_length,
            stride=stride,
        )
        assert config.max_length == max_length
        assert config.stride == stride

    @given(mlm_probability=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20)
    def test_collator_config_mlm_probability(self, mlm_probability: float) -> None:
        """Test collator config with various mlm probabilities."""
        config = create_collator_config(mlm_probability=mlm_probability)
        assert config.mlm_probability == pytest.approx(mlm_probability)

    @given(
        avg_length=st.floats(min_value=0.0, max_value=10000.0),
        max_length=st.integers(min_value=0, max_value=10000),
        padding_ratio=st.floats(min_value=0.0, max_value=1.0),
        truncation_ratio=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20)
    def test_collator_stats_creation(
        self,
        avg_length: float,
        max_length: int,
        padding_ratio: float,
        truncation_ratio: float,
    ) -> None:
        """Test collator stats creation with various inputs."""
        stats = create_collator_stats(
            avg_length=avg_length,
            max_length=max_length,
            padding_ratio=padding_ratio,
            truncation_ratio=truncation_ratio,
        )
        assert stats.avg_length == pytest.approx(avg_length)
        assert stats.max_length == max_length
        assert stats.padding_ratio == pytest.approx(padding_ratio)
        assert stats.truncation_ratio == pytest.approx(truncation_ratio)
