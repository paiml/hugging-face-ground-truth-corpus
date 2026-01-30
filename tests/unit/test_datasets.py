"""Tests for dataset utilities functionality."""

from __future__ import annotations

import pytest
from datasets import Dataset

from hf_gtc.preprocessing.datasets import (
    DatasetInfo,
    create_train_test_split,
    create_train_val_test_split,
    filter_by_length,
    get_dataset_info,
    rename_columns,
    sample_dataset,
    select_columns,
    validate_split_sizes,
)


@pytest.fixture
def sample_dataset_fixture() -> Dataset:
    """Create a sample dataset for testing."""
    return Dataset.from_dict({
        "text": ["hello", "world", "foo", "bar", "baz"] * 20,
        "label": [0, 1, 0, 1, 0] * 20,
    })


@pytest.fixture
def text_length_dataset() -> Dataset:
    """Create a dataset with varying text lengths."""
    return Dataset.from_dict({
        "text": ["a", "ab", "abc", "abcd", "abcde"],
    })


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating DatasetInfo instance."""
        info = DatasetInfo(
            name="test",
            num_rows=100,
            columns=("text", "label"),
            splits=("train", "test"),
        )
        assert info.name == "test"
        assert info.num_rows == 100
        assert "text" in info.columns
        assert "train" in info.splits

    def test_frozen(self) -> None:
        """Test that DatasetInfo is immutable."""
        info = DatasetInfo(
            name="test",
            num_rows=100,
            columns=("text",),
            splits=("train",),
        )
        with pytest.raises(AttributeError):
            info.num_rows = 200  # type: ignore[misc]


class TestValidateSplitSizes:
    """Tests for validate_split_sizes function."""

    def test_valid_sizes(self) -> None:
        """Test validation passes for valid sizes."""
        validate_split_sizes(0.8, 0.2, None)  # Should not raise
        validate_split_sizes(0.7, 0.15, 0.15)  # Should not raise

    def test_sizes_sum_over_one_raises_error(self) -> None:
        """Test that sizes summing over 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must sum to <= 1.0"):
            validate_split_sizes(0.8, 0.3, None)

    def test_zero_size_raises_error(self) -> None:
        """Test that zero size raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_split_sizes(0.0, 0.2, None)

    def test_negative_size_raises_error(self) -> None:
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_split_sizes(-0.1, 0.2, None)

    def test_size_equals_one_raises_error(self) -> None:
        """Test that size of 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be less than 1.0"):
            validate_split_sizes(1.0, None, None)


class TestCreateTrainTestSplit:
    """Tests for create_train_test_split function."""

    def test_basic_split(self, sample_dataset_fixture: Dataset) -> None:
        """Test basic train/test split."""
        train, test = create_train_test_split(sample_dataset_fixture, test_size=0.2)
        total = len(train) + len(test)
        assert total == len(sample_dataset_fixture)

    def test_split_proportions(self, sample_dataset_fixture: Dataset) -> None:
        """Test that split proportions are approximately correct."""
        train, test = create_train_test_split(sample_dataset_fixture, test_size=0.2)
        test_ratio = len(test) / len(sample_dataset_fixture)
        assert 0.15 <= test_ratio <= 0.25  # Allow some variance

    def test_reproducible_with_seed(self, sample_dataset_fixture: Dataset) -> None:
        """Test that same seed gives same split."""
        train1, test1 = create_train_test_split(sample_dataset_fixture, seed=42)
        train2, test2 = create_train_test_split(sample_dataset_fixture, seed=42)
        assert list(train1["text"]) == list(train2["text"])
        assert list(test1["text"]) == list(test2["text"])

    def test_different_seed_gives_different_split(
        self, sample_dataset_fixture: Dataset
    ) -> None:
        """Test that different seeds give different splits."""
        train1, _ = create_train_test_split(sample_dataset_fixture, seed=42)
        train2, _ = create_train_test_split(sample_dataset_fixture, seed=123)
        # Should have different ordering
        assert list(train1["text"]) != list(train2["text"])

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            create_train_test_split(None, test_size=0.2)  # type: ignore[arg-type]

    def test_invalid_test_size_raises_error(self, sample_dataset_fixture: Dataset) -> None:
        """Test that invalid test_size raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            create_train_test_split(sample_dataset_fixture, test_size=0)


class TestCreateTrainValTestSplit:
    """Tests for create_train_val_test_split function."""

    def test_three_way_split(self, sample_dataset_fixture: Dataset) -> None:
        """Test three-way split."""
        train, val, test = create_train_val_test_split(
            sample_dataset_fixture, test_size=0.1, validation_size=0.1
        )
        total = len(train) + len(val) + len(test)
        assert total == len(sample_dataset_fixture)

    def test_split_proportions(self, sample_dataset_fixture: Dataset) -> None:
        """Test that split proportions are approximately correct."""
        train, val, test = create_train_val_test_split(
            sample_dataset_fixture, test_size=0.1, validation_size=0.1
        )
        total = len(sample_dataset_fixture)
        assert abs(len(test) / total - 0.1) < 0.05
        assert abs(len(val) / total - 0.1) < 0.05

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            create_train_val_test_split(None)  # type: ignore[arg-type]


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""

    def test_basic_info(self, sample_dataset_fixture: Dataset) -> None:
        """Test getting basic dataset info."""
        info = get_dataset_info(sample_dataset_fixture, "test_ds")
        assert info.name == "test_ds"
        assert info.num_rows == 100
        assert "text" in info.columns
        assert "label" in info.columns

    def test_dataset_dict_info(self, sample_dataset_fixture: Dataset) -> None:
        """Test getting info from DatasetDict."""
        from datasets import DatasetDict

        ds_dict = DatasetDict({
            "train": sample_dataset_fixture,
            "test": sample_dataset_fixture.select(range(20)),
        })
        info = get_dataset_info(ds_dict, "test_dict")
        assert info.name == "test_dict"
        assert info.num_rows == 120  # 100 + 20
        assert "train" in info.splits
        assert "test" in info.splits

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            get_dataset_info(None)  # type: ignore[arg-type]


class TestSelectColumns:
    """Tests for select_columns function."""

    def test_select_single_column(self, sample_dataset_fixture: Dataset) -> None:
        """Test selecting a single column."""
        selected = select_columns(sample_dataset_fixture, ["text"])
        assert list(selected.column_names) == ["text"]

    def test_select_multiple_columns(self, sample_dataset_fixture: Dataset) -> None:
        """Test selecting multiple columns."""
        selected = select_columns(sample_dataset_fixture, ["text", "label"])
        assert set(selected.column_names) == {"text", "label"}

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            select_columns(None, ["text"])  # type: ignore[arg-type]

    def test_empty_columns_raises_error(self, sample_dataset_fixture: Dataset) -> None:
        """Test that empty columns raises ValueError."""
        with pytest.raises(ValueError, match="columns cannot be empty"):
            select_columns(sample_dataset_fixture, [])

    def test_nonexistent_column_raises_error(
        self, sample_dataset_fixture: Dataset
    ) -> None:
        """Test that nonexistent column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            select_columns(sample_dataset_fixture, ["nonexistent"])


class TestRenameColumns:
    """Tests for rename_columns function."""

    def test_rename_single_column(self, sample_dataset_fixture: Dataset) -> None:
        """Test renaming a single column."""
        renamed = rename_columns(sample_dataset_fixture, {"text": "content"})
        assert "content" in renamed.column_names
        assert "text" not in renamed.column_names

    def test_rename_multiple_columns(self, sample_dataset_fixture: Dataset) -> None:
        """Test renaming multiple columns."""
        renamed = rename_columns(
            sample_dataset_fixture, {"text": "content", "label": "category"}
        )
        assert "content" in renamed.column_names
        assert "category" in renamed.column_names

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            rename_columns(None, {"text": "content"})  # type: ignore[arg-type]

    def test_empty_mapping_raises_error(self, sample_dataset_fixture: Dataset) -> None:
        """Test that empty mapping raises ValueError."""
        with pytest.raises(ValueError, match="column_mapping cannot be empty"):
            rename_columns(sample_dataset_fixture, {})

    def test_nonexistent_column_raises_error(
        self, sample_dataset_fixture: Dataset
    ) -> None:
        """Test that nonexistent column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            rename_columns(sample_dataset_fixture, {"nonexistent": "new_name"})


class TestFilterByLength:
    """Tests for filter_by_length function."""

    def test_filter_min_length(self, text_length_dataset: Dataset) -> None:
        """Test filtering by minimum length."""
        filtered = filter_by_length(text_length_dataset, "text", min_length=3)
        assert len(filtered) == 3  # "abc", "abcd", "abcde"

    def test_filter_max_length(self, text_length_dataset: Dataset) -> None:
        """Test filtering by maximum length."""
        filtered = filter_by_length(text_length_dataset, "text", max_length=2)
        assert len(filtered) == 2  # "a", "ab"

    def test_filter_min_and_max_length(self, text_length_dataset: Dataset) -> None:
        """Test filtering by both min and max length."""
        filtered = filter_by_length(
            text_length_dataset, "text", min_length=2, max_length=4
        )
        assert len(filtered) == 3  # "ab", "abc", "abcd"

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            filter_by_length(None, "text", min_length=1)  # type: ignore[arg-type]

    def test_nonexistent_column_raises_error(
        self, text_length_dataset: Dataset
    ) -> None:
        """Test that nonexistent column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            filter_by_length(text_length_dataset, "nonexistent", min_length=1)

    def test_no_constraints_raises_error(self, text_length_dataset: Dataset) -> None:
        """Test that no min/max constraints raises ValueError."""
        with pytest.raises(ValueError, match="At least one of"):
            filter_by_length(text_length_dataset, "text")


class TestSampleDataset:
    """Tests for sample_dataset function."""

    def test_sample_correct_size(self, sample_dataset_fixture: Dataset) -> None:
        """Test that sample returns correct number of rows."""
        sampled = sample_dataset(sample_dataset_fixture, n=10)
        assert len(sampled) == 10

    def test_reproducible_with_seed(self, sample_dataset_fixture: Dataset) -> None:
        """Test that same seed gives same sample."""
        sample1 = sample_dataset(sample_dataset_fixture, n=10, seed=42)
        sample2 = sample_dataset(sample_dataset_fixture, n=10, seed=42)
        assert list(sample1["text"]) == list(sample2["text"])

    def test_none_dataset_raises_error(self) -> None:
        """Test that None dataset raises ValueError."""
        with pytest.raises(ValueError, match="dataset cannot be None"):
            sample_dataset(None, n=10)  # type: ignore[arg-type]

    def test_zero_n_raises_error(self, sample_dataset_fixture: Dataset) -> None:
        """Test that n=0 raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive"):
            sample_dataset(sample_dataset_fixture, n=0)

    def test_negative_n_raises_error(self, sample_dataset_fixture: Dataset) -> None:
        """Test that negative n raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive"):
            sample_dataset(sample_dataset_fixture, n=-1)

    def test_n_exceeds_size_raises_error(self, sample_dataset_fixture: Dataset) -> None:
        """Test that n > dataset size raises ValueError."""
        with pytest.raises(ValueError, match="exceeds dataset size"):
            sample_dataset(sample_dataset_fixture, n=1000)
