"""Tests for dataset streaming functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.streaming import (
    ShuffleMode,
    StreamConfig,
    StreamProgress,
    StreamStats,
    compute_stream_stats,
    create_stream_iterator,
    filter_stream,
    list_shuffle_modes,
    map_stream,
    skip_stream,
    stream_batches,
    stream_dataset,
    take_stream,
    validate_shuffle_mode,
    validate_stream_config,
)


class TestShuffleMode:
    """Tests for ShuffleMode enum."""

    def test_disabled_value(self) -> None:
        """Test DISABLED mode value."""
        assert ShuffleMode.DISABLED.value == "disabled"

    def test_buffer_value(self) -> None:
        """Test BUFFER mode value."""
        assert ShuffleMode.BUFFER.value == "buffer"

    def test_full_value(self) -> None:
        """Test FULL mode value."""
        assert ShuffleMode.FULL.value == "full"


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = StreamConfig()
        assert config.batch_size == 1000
        assert config.buffer_size == 10000
        assert config.shuffle_mode == ShuffleMode.BUFFER
        assert config.prefetch_batches == 2
        assert config.num_workers == 0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = StreamConfig(
            batch_size=500,
            buffer_size=5000,
            shuffle_mode=ShuffleMode.DISABLED,
            prefetch_batches=4,
            num_workers=2,
        )
        assert config.batch_size == 500
        assert config.buffer_size == 5000
        assert config.shuffle_mode == ShuffleMode.DISABLED
        assert config.prefetch_batches == 4
        assert config.num_workers == 2

    def test_frozen(self) -> None:
        """Test that StreamConfig is immutable."""
        config = StreamConfig()
        with pytest.raises(AttributeError):
            config.batch_size = 500  # type: ignore[misc]


class TestStreamStats:
    """Tests for StreamStats dataclass."""

    def test_creation(self) -> None:
        """Test creating StreamStats instance."""
        stats = StreamStats(
            total_samples=10000,
            total_batches=10,
            bytes_read=1024000,
            samples_per_second=5000.0,
        )
        assert stats.total_samples == 10000
        assert stats.total_batches == 10
        assert stats.bytes_read == 1024000
        assert stats.samples_per_second == pytest.approx(5000.0)

    def test_frozen(self) -> None:
        """Test that StreamStats is immutable."""
        stats = StreamStats(
            total_samples=100,
            total_batches=1,
            bytes_read=1000,
            samples_per_second=100.0,
        )
        with pytest.raises(AttributeError):
            stats.total_samples = 200  # type: ignore[misc]


class TestStreamProgress:
    """Tests for StreamProgress dataclass."""

    def test_creation(self) -> None:
        """Test creating StreamProgress instance."""
        progress = StreamProgress(
            samples_processed=500,
            batches_processed=5,
            estimated_total=1000,
            percent_complete=50.0,
        )
        assert progress.samples_processed == 500
        assert progress.batches_processed == 5
        assert progress.estimated_total == 1000
        assert progress.percent_complete == pytest.approx(50.0)

    def test_none_values(self) -> None:
        """Test StreamProgress with None values."""
        progress = StreamProgress(
            samples_processed=100,
            batches_processed=1,
            estimated_total=None,
            percent_complete=None,
        )
        assert progress.estimated_total is None
        assert progress.percent_complete is None

    def test_frozen(self) -> None:
        """Test that StreamProgress is immutable."""
        progress = StreamProgress(
            samples_processed=100,
            batches_processed=1,
            estimated_total=1000,
            percent_complete=10.0,
        )
        with pytest.raises(AttributeError):
            progress.samples_processed = 200  # type: ignore[misc]


class TestValidateStreamConfig:
    """Tests for validate_stream_config function."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = StreamConfig(batch_size=100, buffer_size=1000)
        validate_stream_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_stream_config(None)  # type: ignore[arg-type]

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        config = StreamConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_stream_config(config)

    def test_negative_batch_size_raises_error(self) -> None:
        """Test that negative batch_size raises ValueError."""
        config = StreamConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_stream_config(config)

    def test_zero_buffer_size_raises_error(self) -> None:
        """Test that zero buffer_size raises ValueError."""
        config = StreamConfig(buffer_size=0)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            validate_stream_config(config)

    def test_negative_prefetch_raises_error(self) -> None:
        """Test that negative prefetch_batches raises ValueError."""
        config = StreamConfig(prefetch_batches=-1)
        with pytest.raises(ValueError, match="prefetch_batches cannot be negative"):
            validate_stream_config(config)

    def test_negative_workers_raises_error(self) -> None:
        """Test that negative num_workers raises ValueError."""
        config = StreamConfig(num_workers=-1)
        with pytest.raises(ValueError, match="num_workers cannot be negative"):
            validate_stream_config(config)


class TestCreateStreamIterator:
    """Tests for create_stream_iterator function."""

    def test_basic_batching(self) -> None:
        """Test basic stream batching."""
        batches = list(create_stream_iterator(iter([1, 2, 3, 4, 5]), batch_size=2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_exact_batch_size(self) -> None:
        """Test batching when items divide evenly."""
        batches = list(create_stream_iterator(iter([1, 2, 3, 4]), batch_size=2))
        assert batches == [[1, 2], [3, 4]]

    def test_drop_last(self) -> None:
        """Test dropping last incomplete batch."""
        batches = list(
            create_stream_iterator(iter([1, 2, 3, 4, 5]), batch_size=2, drop_last=True)
        )
        assert batches == [[1, 2], [3, 4]]

    def test_empty_iterator(self) -> None:
        """Test batching empty iterator."""
        batches = list(create_stream_iterator(iter([]), batch_size=10))
        assert batches == []

    def test_none_items_raises_error(self) -> None:
        """Test that None items raises ValueError."""
        with pytest.raises(ValueError, match="items cannot be None"):
            list(create_stream_iterator(None, batch_size=2))  # type: ignore[arg-type]

    def test_zero_batch_size_raises_error(self) -> None:
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(create_stream_iterator(iter([1, 2, 3]), batch_size=0))

    @given(
        items=st.lists(st.integers(), min_size=1, max_size=100),
        batch_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20)
    def test_all_items_preserved(self, items: list[int], batch_size: int) -> None:
        """Test that all items are preserved after batching."""
        batches = list(create_stream_iterator(iter(items), batch_size=batch_size))
        flattened = [item for batch in batches for item in batch]
        assert flattened == items


class TestMapStream:
    """Tests for map_stream function."""

    def test_basic_map(self) -> None:
        """Test basic stream mapping."""
        result = list(map_stream(iter([1, 2, 3]), lambda x: x * 2))
        assert result == [2, 4, 6]

    def test_string_map(self) -> None:
        """Test mapping with string function."""
        result = list(map_stream(iter(["a", "b", "c"]), str.upper))
        assert result == ["A", "B", "C"]

    def test_empty_stream(self) -> None:
        """Test mapping empty stream."""
        result = list(map_stream(iter([]), lambda x: x))
        assert result == []

    def test_none_items_raises_error(self) -> None:
        """Test that None items raises ValueError."""
        with pytest.raises(ValueError, match="items cannot be None"):
            list(map_stream(None, lambda x: x))  # type: ignore[arg-type]

    def test_none_fn_raises_error(self) -> None:
        """Test that None fn raises ValueError."""
        with pytest.raises(ValueError, match="fn cannot be None"):
            list(map_stream(iter([1]), None))  # type: ignore[arg-type]


class TestFilterStream:
    """Tests for filter_stream function."""

    def test_basic_filter(self) -> None:
        """Test basic stream filtering."""
        result = list(filter_stream(iter([1, 2, 3, 4, 5]), lambda x: x > 2))
        assert result == [3, 4, 5]

    def test_filter_all(self) -> None:
        """Test filtering all items."""
        result = list(filter_stream(iter([1, 2, 3]), lambda x: x > 10))
        assert result == []

    def test_filter_none(self) -> None:
        """Test filtering no items."""
        result = list(filter_stream(iter([1, 2, 3]), lambda x: True))
        assert result == [1, 2, 3]

    def test_empty_stream(self) -> None:
        """Test filtering empty stream."""
        result = list(filter_stream(iter([]), lambda x: True))
        assert result == []

    def test_none_items_raises_error(self) -> None:
        """Test that None items raises ValueError."""
        with pytest.raises(ValueError, match="items cannot be None"):
            list(filter_stream(None, lambda x: True))  # type: ignore[arg-type]

    def test_none_predicate_raises_error(self) -> None:
        """Test that None predicate raises ValueError."""
        with pytest.raises(ValueError, match="predicate cannot be None"):
            list(filter_stream(iter([1]), None))  # type: ignore[arg-type]


class TestTakeStream:
    """Tests for take_stream function."""

    def test_basic_take(self) -> None:
        """Test basic stream taking."""
        result = list(take_stream(iter([1, 2, 3, 4, 5]), 3))
        assert result == [1, 2, 3]

    def test_take_more_than_available(self) -> None:
        """Test taking more items than available."""
        result = list(take_stream(iter([1, 2]), 10))
        assert result == [1, 2]

    def test_take_zero(self) -> None:
        """Test taking zero items."""
        result = list(take_stream(iter([1, 2, 3]), 0))
        assert result == []

    def test_empty_stream(self) -> None:
        """Test taking from empty stream."""
        result = list(take_stream(iter([]), 5))
        assert result == []

    def test_none_items_raises_error(self) -> None:
        """Test that None items raises ValueError."""
        with pytest.raises(ValueError, match="items cannot be None"):
            list(take_stream(None, 5))  # type: ignore[arg-type]

    def test_negative_n_raises_error(self) -> None:
        """Test that negative n raises ValueError."""
        with pytest.raises(ValueError, match="n cannot be negative"):
            list(take_stream(iter([1]), -1))


class TestSkipStream:
    """Tests for skip_stream function."""

    def test_basic_skip(self) -> None:
        """Test basic stream skipping."""
        result = list(skip_stream(iter([1, 2, 3, 4, 5]), 2))
        assert result == [3, 4, 5]

    def test_skip_more_than_available(self) -> None:
        """Test skipping more items than available."""
        result = list(skip_stream(iter([1, 2]), 10))
        assert result == []

    def test_skip_zero(self) -> None:
        """Test skipping zero items."""
        result = list(skip_stream(iter([1, 2, 3]), 0))
        assert result == [1, 2, 3]

    def test_empty_stream(self) -> None:
        """Test skipping from empty stream."""
        result = list(skip_stream(iter([]), 5))
        assert result == []

    def test_none_items_raises_error(self) -> None:
        """Test that None items raises ValueError."""
        with pytest.raises(ValueError, match="items cannot be None"):
            list(skip_stream(None, 5))  # type: ignore[arg-type]

    def test_negative_n_raises_error(self) -> None:
        """Test that negative n raises ValueError."""
        with pytest.raises(ValueError, match="n cannot be negative"):
            list(skip_stream(iter([1]), -1))


class TestStreamDataset:
    """Tests for stream_dataset function."""

    def test_empty_dataset_name_raises_error(self) -> None:
        """Test that empty dataset_name raises ValueError."""
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            next(stream_dataset(""))

    def test_empty_split_raises_error(self) -> None:
        """Test that empty split raises ValueError."""
        with pytest.raises(ValueError, match="split cannot be empty"):
            next(stream_dataset("test", split=""))

    @patch("datasets.load_dataset")
    def test_calls_load_dataset_with_streaming(self, mock_load: MagicMock) -> None:
        """Test that load_dataset is called with streaming=True."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([]))
        mock_load.return_value = mock_ds

        list(stream_dataset("test/dataset"))

        mock_load.assert_called_once_with(
            "test/dataset", split="train", streaming=True, revision=None
        )

    @patch("datasets.load_dataset")
    def test_yields_rows(self, mock_load: MagicMock) -> None:
        """Test that stream_dataset yields rows."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([{"text": "a"}, {"text": "b"}]))
        mock_load.return_value = mock_ds

        results = list(stream_dataset("test/dataset"))

        assert results == [{"text": "a"}, {"text": "b"}]


class TestStreamBatches:
    """Tests for stream_batches function."""

    def test_empty_dataset_name_raises_error(self) -> None:
        """Test that empty dataset_name raises ValueError."""
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            next(stream_batches(""))

    @patch("datasets.load_dataset")
    def test_yields_batches(self, mock_load: MagicMock) -> None:
        """Test that stream_batches yields batches."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([{"text": "a"}, {"text": "b"}, {"text": "c"}])
        )
        # Handle shuffle() returning a dataset with same iterator
        mock_ds.shuffle.return_value = mock_ds
        mock_load.return_value = mock_ds

        config = StreamConfig(batch_size=2)
        results = list(stream_batches("test/dataset", config=config))

        assert len(results) == 2
        assert results[0] == [{"text": "a"}, {"text": "b"}]
        assert results[1] == [{"text": "c"}]


class TestComputeStreamStats:
    """Tests for compute_stream_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic statistics computation."""
        stats = compute_stream_stats(10000, 10, 1024000, 2.0)
        assert stats.total_samples == 10000
        assert stats.total_batches == 10
        assert stats.bytes_read == 1024000
        assert stats.samples_per_second == pytest.approx(5000.0)

    def test_zero_elapsed_time(self) -> None:
        """Test with zero elapsed time."""
        stats = compute_stream_stats(100, 1, 1000, 0.0)
        assert stats.samples_per_second == pytest.approx(0.0)

    def test_negative_samples_raises_error(self) -> None:
        """Test that negative samples raises ValueError."""
        with pytest.raises(ValueError, match="total_samples cannot be negative"):
            compute_stream_stats(-1, 10, 0, 1.0)

    def test_negative_batches_raises_error(self) -> None:
        """Test that negative batches raises ValueError."""
        with pytest.raises(ValueError, match="total_batches cannot be negative"):
            compute_stream_stats(100, -1, 0, 1.0)

    def test_negative_bytes_raises_error(self) -> None:
        """Test that negative bytes raises ValueError."""
        with pytest.raises(ValueError, match="bytes_read cannot be negative"):
            compute_stream_stats(100, 10, -1, 1.0)

    def test_negative_time_raises_error(self) -> None:
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="elapsed_seconds cannot be negative"):
            compute_stream_stats(100, 10, 0, -1.0)


class TestListShuffleModes:
    """Tests for list_shuffle_modes function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        modes = list_shuffle_modes()
        assert isinstance(modes, list)

    def test_contains_expected_modes(self) -> None:
        """Test that list contains expected modes."""
        modes = list_shuffle_modes()
        assert "disabled" in modes
        assert "buffer" in modes
        assert "full" in modes

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        modes = list_shuffle_modes()
        assert modes == sorted(modes)

    def test_all_strings(self) -> None:
        """Test that all items are strings."""
        modes = list_shuffle_modes()
        assert all(isinstance(m, str) for m in modes)


class TestValidateShuffleMode:
    """Tests for validate_shuffle_mode function."""

    def test_valid_disabled(self) -> None:
        """Test validation of disabled mode."""
        assert validate_shuffle_mode("disabled") is True

    def test_valid_buffer(self) -> None:
        """Test validation of buffer mode."""
        assert validate_shuffle_mode("buffer") is True

    def test_valid_full(self) -> None:
        """Test validation of full mode."""
        assert validate_shuffle_mode("full") is True

    def test_invalid_mode(self) -> None:
        """Test validation of invalid mode."""
        assert validate_shuffle_mode("invalid") is False

    def test_empty_string(self) -> None:
        """Test validation of empty string."""
        assert validate_shuffle_mode("") is False

    def test_case_sensitive(self) -> None:
        """Test that validation is case sensitive."""
        assert validate_shuffle_mode("BUFFER") is False
        assert validate_shuffle_mode("Buffer") is False
