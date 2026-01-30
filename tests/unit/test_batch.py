"""Tests for batch inference functionality."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.inference.batch import (
    BatchConfig,
    BatchResult,
    BatchStats,
    PaddingStrategy,
    compute_batch_stats,
    compute_num_batches,
    create_batches,
    estimate_memory_per_batch,
    get_optimal_batch_size,
    list_padding_strategies,
    validate_batch_config,
)


class TestPaddingStrategy:
    """Tests for PaddingStrategy enum."""

    def test_longest_value(self) -> None:
        """Test LONGEST padding value."""
        assert PaddingStrategy.LONGEST.value == "longest"

    def test_max_length_value(self) -> None:
        """Test MAX_LENGTH padding value."""
        assert PaddingStrategy.MAX_LENGTH.value == "max_length"

    def test_do_not_pad_value(self) -> None:
        """Test DO_NOT_PAD padding value."""
        assert PaddingStrategy.DO_NOT_PAD.value == "do_not_pad"


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BatchConfig()
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.padding == PaddingStrategy.LONGEST
        assert config.truncation is True
        assert config.return_tensors == "pt"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BatchConfig(
            batch_size=16,
            max_length=256,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=False,
            return_tensors="np",
        )
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.padding == PaddingStrategy.MAX_LENGTH
        assert config.truncation is False
        assert config.return_tensors == "np"

    def test_frozen(self) -> None:
        """Test that BatchConfig is immutable."""
        config = BatchConfig()
        with pytest.raises(AttributeError):
            config.batch_size = 64  # type: ignore[misc]


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_creation(self) -> None:
        """Test creating BatchResult instance."""
        result = BatchResult(
            predictions=["a", "b", "c"],
            batch_index=0,
            num_samples=3,
            processing_time_ms=100.0,
        )
        assert result.predictions == ["a", "b", "c"]
        assert result.batch_index == 0
        assert result.num_samples == 3
        assert result.processing_time_ms == 100.0

    def test_frozen(self) -> None:
        """Test that BatchResult is immutable."""
        result = BatchResult(
            predictions=[],
            batch_index=0,
            num_samples=0,
            processing_time_ms=0.0,
        )
        with pytest.raises(AttributeError):
            result.batch_index = 1  # type: ignore[misc]


class TestBatchStats:
    """Tests for BatchStats dataclass."""

    def test_creation(self) -> None:
        """Test creating BatchStats instance."""
        stats = BatchStats(
            total_samples=100,
            total_batches=4,
            avg_batch_time_ms=250.0,
            total_time_ms=1000.0,
            samples_per_second=100.0,
        )
        assert stats.total_samples == 100
        assert stats.total_batches == 4
        assert stats.avg_batch_time_ms == 250.0
        assert stats.total_time_ms == 1000.0
        assert stats.samples_per_second == 100.0

    def test_frozen(self) -> None:
        """Test that BatchStats is immutable."""
        stats = BatchStats(
            total_samples=100,
            total_batches=4,
            avg_batch_time_ms=250.0,
            total_time_ms=1000.0,
            samples_per_second=100.0,
        )
        with pytest.raises(AttributeError):
            stats.total_samples = 200  # type: ignore[misc]


class TestValidateBatchConfig:
    """Tests for validate_batch_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = BatchConfig(batch_size=32, max_length=512)
        validate_batch_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_batch_config(None)  # type: ignore[arg-type]

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = BatchConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_batch_config(config)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch_size raises ValueError."""
        config = BatchConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_batch_config(config)

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        config = BatchConfig(max_length=0)
        with pytest.raises(ValueError, match="max_length must be positive"):
            validate_batch_config(config)

    def test_invalid_return_tensors_raises_error(self) -> None:
        """Test that invalid return_tensors raises ValueError."""
        config = BatchConfig(return_tensors="invalid")
        with pytest.raises(ValueError, match="return_tensors must be one of"):
            validate_batch_config(config)

    def test_valid_return_tensors(self) -> None:
        """Test all valid return_tensors options."""
        for tensors in ["pt", "tf", "np", "jax"]:
            config = BatchConfig(return_tensors=tensors)
            validate_batch_config(config)  # Should not raise


class TestCreateBatches:
    """Tests for create_batches function."""

    def test_basic_batching(self) -> None:
        """Test basic batch creation."""
        batches = list(create_batches([1, 2, 3, 4, 5], batch_size=2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_exact_batch_size(self) -> None:
        """Test batching when items divide evenly."""
        batches = list(create_batches([1, 2, 3, 4], batch_size=2))
        assert batches == [[1, 2], [3, 4]]

    def test_single_batch(self) -> None:
        """Test when all items fit in one batch."""
        batches = list(create_batches([1, 2, 3], batch_size=10))
        assert batches == [[1, 2, 3]]

    def test_empty_items(self) -> None:
        """Test batching empty list."""
        batches = list(create_batches([], batch_size=10))
        assert batches == []

    def test_none_items_raises_error(self) -> None:
        """Test that None items raises ValueError."""
        with pytest.raises(ValueError, match="items cannot be None"):
            list(create_batches(None, batch_size=2))  # type: ignore[arg-type]

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(create_batches([1, 2, 3], batch_size=0))

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(create_batches([1, 2, 3], batch_size=-1))

    def test_string_items(self) -> None:
        """Test batching with string items."""
        batches = list(create_batches(["a", "b", "c"], batch_size=2))
        assert batches == [["a", "b"], ["c"]]

    @given(
        items=st.lists(st.integers(), min_size=1, max_size=100),
        batch_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20)
    def test_all_items_preserved(
        self, items: list[int], batch_size: int
    ) -> None:
        """Test that all items are preserved after batching."""
        batches = list(create_batches(items, batch_size=batch_size))
        flattened = [item for batch in batches for item in batch]
        assert flattened == items


class TestComputeNumBatches:
    """Tests for compute_num_batches function."""

    def test_exact_division(self) -> None:
        """Test when samples divide evenly into batches."""
        assert compute_num_batches(64, 32) == 2

    def test_remainder(self) -> None:
        """Test when there's a remainder."""
        assert compute_num_batches(65, 32) == 3

    def test_single_batch(self) -> None:
        """Test when samples fit in one batch."""
        assert compute_num_batches(10, 32) == 1

    def test_zero_samples(self) -> None:
        """Test with zero samples."""
        assert compute_num_batches(0, 32) == 0

    def test_negative_samples_raises_error(self) -> None:
        """Test that negative samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples cannot be negative"):
            compute_num_batches(-1, 32)

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            compute_num_batches(100, 0)

    @given(
        num_samples=st.integers(min_value=0, max_value=10000),
        batch_size=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_batch_count_correct(self, num_samples: int, batch_size: int) -> None:
        """Test that batch count is correct."""
        num_batches = compute_num_batches(num_samples, batch_size)
        if num_samples == 0:
            assert num_batches == 0
        else:
            # Should hold:
            # (num_batches-1)*batch_size < num_samples <= num_batches*batch_size
            assert (num_batches - 1) * batch_size < num_samples
            assert num_samples <= num_batches * batch_size


class TestComputeBatchStats:
    """Tests for compute_batch_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic statistics computation."""
        stats = compute_batch_stats(100, 4, 1000.0)
        assert stats.total_samples == 100
        assert stats.total_batches == 4
        assert stats.avg_batch_time_ms == 250.0
        assert stats.total_time_ms == 1000.0
        assert stats.samples_per_second == 100.0

    def test_zero_batches(self) -> None:
        """Test with zero batches (edge case)."""
        stats = compute_batch_stats(0, 0, 0.0)
        assert stats.avg_batch_time_ms == 0.0
        assert stats.samples_per_second == 0.0

    def test_zero_time(self) -> None:
        """Test with zero time (edge case)."""
        stats = compute_batch_stats(100, 4, 0.0)
        assert stats.samples_per_second == 0.0

    def test_negative_samples_raises_error(self) -> None:
        """Test that negative samples raises ValueError."""
        with pytest.raises(ValueError, match="total_samples cannot be negative"):
            compute_batch_stats(-1, 4, 1000.0)

    def test_negative_batches_raises_error(self) -> None:
        """Test that negative batches raises ValueError."""
        with pytest.raises(ValueError, match="total_batches cannot be negative"):
            compute_batch_stats(100, -1, 1000.0)

    def test_negative_time_raises_error(self) -> None:
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="total_time_ms cannot be negative"):
            compute_batch_stats(100, 4, -1.0)


class TestEstimateMemoryPerBatch:
    """Tests for estimate_memory_per_batch function."""

    def test_basic_estimate(self) -> None:
        """Test basic memory estimation."""
        memory = estimate_memory_per_batch(32, 512, 768, 4)
        # 32 * 512 * 768 * 4 = 50_331_648 bytes = 48 MB
        assert round(memory, 2) == 48.0

    def test_single_sample(self) -> None:
        """Test memory for single sample."""
        memory = estimate_memory_per_batch(1, 512, 768, 4)
        # 1 * 512 * 768 * 4 = 1_572_864 bytes = 1.5 MB
        assert round(memory, 2) == 1.5

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            estimate_memory_per_batch(0, 512)

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            estimate_memory_per_batch(32, 0)

    def test_zero_hidden_size_raises_error(self) -> None:
        """Test that zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            estimate_memory_per_batch(32, 512, 0)

    def test_zero_dtype_bytes_raises_error(self) -> None:
        """Test that zero dtype_bytes raises ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            estimate_memory_per_batch(32, 512, 768, 0)

    def test_fp16_estimate(self) -> None:
        """Test memory estimate with fp16 (2 bytes)."""
        memory_fp32 = estimate_memory_per_batch(32, 512, 768, 4)
        memory_fp16 = estimate_memory_per_batch(32, 512, 768, 2)
        assert memory_fp16 == memory_fp32 / 2


class TestGetOptimalBatchSize:
    """Tests for get_optimal_batch_size function."""

    def test_basic_calculation(self) -> None:
        """Test basic batch size calculation."""
        batch_size = get_optimal_batch_size(1000.0, 512, 768, 4)
        assert batch_size >= 1
        # With 1000 MB * 0.8 = 800 MB usable, and ~1.5 MB per sample
        # batch_size = 800 / 1.5 ≈ 533
        assert batch_size == 533

    def test_low_memory(self) -> None:
        """Test with very low memory returns at least 1."""
        batch_size = get_optimal_batch_size(1.0, 512, 768, 4)
        assert batch_size == 1

    def test_zero_memory_raises_error(self) -> None:
        """Test that zero memory raises ValueError."""
        with pytest.raises(ValueError, match="available_memory_mb must be positive"):
            get_optimal_batch_size(0, 512)

    def test_zero_max_length_raises_error(self) -> None:
        """Test that zero max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            get_optimal_batch_size(1000.0, 0)

    def test_invalid_memory_fraction_raises_error(self) -> None:
        """Test that invalid memory_fraction raises ValueError."""
        with pytest.raises(ValueError, match="memory_fraction must be in"):
            get_optimal_batch_size(1000.0, 512, memory_fraction=0)
        with pytest.raises(ValueError, match="memory_fraction must be in"):
            get_optimal_batch_size(1000.0, 512, memory_fraction=1.5)

    def test_custom_memory_fraction(self) -> None:
        """Test with custom memory fraction."""
        batch_full = get_optimal_batch_size(1000.0, 512, memory_fraction=1.0)
        batch_half = get_optimal_batch_size(1000.0, 512, memory_fraction=0.5)
        # Half memory fraction should give roughly half batch size
        assert batch_half < batch_full


class TestListPaddingStrategies:
    """Tests for list_padding_strategies function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        strategies = list_padding_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self) -> None:
        """Test that list contains expected strategies."""
        strategies = list_padding_strategies()
        assert "longest" in strategies
        assert "max_length" in strategies
        assert "do_not_pad" in strategies

    def test_list_is_sorted(self) -> None:
        """Test that list is sorted."""
        strategies = list_padding_strategies()
        assert strategies == sorted(strategies)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        strategies = list_padding_strategies()
        assert all(isinstance(s, str) for s in strategies)
