"""Dataset loading and processing utilities.

This module provides functions for loading, splitting, and processing
datasets using the HuggingFace datasets library.

Examples:
    >>> from datasets import Dataset
    >>> from hf_gtc.preprocessing.datasets import create_train_test_split
    >>> ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
    >>> train, test = create_train_test_split(ds, test_size=0.2, seed=42)
    >>> len(train) + len(test) == 5
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Information about a loaded dataset.

    Attributes:
        name: Dataset name.
        num_rows: Total number of rows.
        columns: List of column names.
        splits: Available splits in the dataset.

    Examples:
        >>> info = DatasetInfo(
        ...     name="test_dataset",
        ...     num_rows=1000,
        ...     columns=["text", "label"],
        ...     splits=("train", "test"),
        ... )
        >>> info.num_rows
        1000
    """

    name: str
    num_rows: int
    columns: tuple[str, ...]
    splits: tuple[str, ...]


def validate_split_sizes(
    train_size: float | None,
    test_size: float | None,
    validation_size: float | None,
) -> None:
    """Validate that split sizes are valid.

    Args:
        train_size: Proportion for training set.
        test_size: Proportion for test set.
        validation_size: Proportion for validation set.

    Raises:
        ValueError: If any size is invalid or total exceeds 1.0.

    Examples:
        >>> validate_split_sizes(0.8, 0.2, None)  # No error

        >>> validate_split_sizes(0.8, 0.3, None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Split sizes must sum to <= 1.0
    """
    sizes = [s for s in [train_size, test_size, validation_size] if s is not None]

    for size in sizes:
        if size <= 0:
            msg = f"Split size must be positive, got {size}"
            raise ValueError(msg)
        if size >= 1.0:
            msg = f"Split size must be less than 1.0, got {size}"
            raise ValueError(msg)

    total = sum(sizes)
    if total > 1.0:
        msg = f"Split sizes must sum to <= 1.0, got {total}"
        raise ValueError(msg)


def create_train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and test sets.

    Args:
        dataset: Dataset to split.
        test_size: Proportion of data for test set. Defaults to 0.2.
        seed: Random seed for reproducibility. Defaults to 42.
        shuffle: Whether to shuffle before splitting. Defaults to True.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If test_size is not in (0, 1).
        ValueError: If dataset is None.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        >>> train, test = create_train_test_split(ds, test_size=0.2, seed=42)
        >>> len(train) + len(test) == 5
        True
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    validate_split_sizes(None, test_size, None)

    split = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=shuffle,
    )
    return split["train"], split["test"]


def create_train_val_test_split(
    dataset: Dataset,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split a dataset into train, validation, and test sets.

    Args:
        dataset: Dataset to split.
        test_size: Proportion for test set. Defaults to 0.1.
        validation_size: Proportion for validation set. Defaults to 0.1.
        seed: Random seed for reproducibility. Defaults to 42.
        shuffle: Whether to shuffle before splitting. Defaults to True.

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset).

    Raises:
        ValueError: If sizes are invalid.
        ValueError: If dataset is None.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"text": list(range(100))})
        >>> train, val, test = create_train_val_test_split(
        ...     ds, test_size=0.1, validation_size=0.1
        ... )
        >>> len(train) + len(val) + len(test) == 100
        True
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    validate_split_sizes(None, test_size, validation_size)

    # First split off test set
    split1 = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=shuffle,
    )
    test_dataset = split1["test"]

    # Then split remaining into train/validation
    # Adjust validation size to account for already removed test data
    adjusted_val_size = validation_size / (1 - test_size)
    split2 = split1["train"].train_test_split(
        test_size=adjusted_val_size,
        seed=seed,
        shuffle=shuffle,
    )

    return split2["train"], split2["test"], test_dataset


def get_dataset_info(
    dataset: Dataset | DatasetDict, name: str = "dataset"
) -> DatasetInfo:
    """Get information about a dataset.

    Args:
        dataset: Dataset or DatasetDict to inspect.
        name: Name to use for the dataset. Defaults to "dataset".

    Returns:
        DatasetInfo with dataset metadata.

    Raises:
        ValueError: If dataset is None.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        >>> info = get_dataset_info(ds, "my_dataset")
        >>> info.num_rows
        2
        >>> "text" in info.columns
        True
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    # Handle DatasetDict
    if hasattr(dataset, "keys"):
        splits = tuple(dataset.keys())  # type: ignore[call-non-callable]
        # Use first split to get column info
        first_split = dataset[splits[0]]
        columns = tuple(first_split.column_names)
        num_rows = sum(len(dataset[s]) for s in splits)
    else:
        splits = ("train",)
        columns = tuple(dataset.column_names)
        num_rows = len(dataset)

    return DatasetInfo(
        name=name,
        num_rows=num_rows,
        columns=columns,
        splits=splits,
    )


def select_columns(
    dataset: Dataset,
    columns: list[str],
) -> Dataset:
    """Select specific columns from a dataset.

    Args:
        dataset: Dataset to select from.
        columns: List of column names to keep.

    Returns:
        Dataset with only the specified columns.

    Raises:
        ValueError: If dataset is None.
        ValueError: If columns is empty.
        ValueError: If any column doesn't exist.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        >>> selected = select_columns(ds, ["a", "b"])
        >>> list(selected.column_names)
        ['a', 'b']
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    if not columns:
        msg = "columns cannot be empty"
        raise ValueError(msg)

    existing_columns = set(dataset.column_names)
    for col in columns:
        if col not in existing_columns:
            msg = f"Column '{col}' not found in dataset. Available: {existing_columns}"
            raise ValueError(msg)

    # Remove columns not in the list
    columns_to_remove = [c for c in dataset.column_names if c not in columns]
    return dataset.remove_columns(columns_to_remove)


def rename_columns(
    dataset: Dataset,
    column_mapping: dict[str, str],
) -> Dataset:
    """Rename columns in a dataset.

    Args:
        dataset: Dataset to rename columns in.
        column_mapping: Dictionary mapping old names to new names.

    Returns:
        Dataset with renamed columns.

    Raises:
        ValueError: If dataset is None.
        ValueError: If column_mapping is empty.
        ValueError: If any source column doesn't exist.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"old_name": [1, 2, 3]})
        >>> renamed = rename_columns(ds, {"old_name": "new_name"})
        >>> "new_name" in renamed.column_names
        True
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    if not column_mapping:
        msg = "column_mapping cannot be empty"
        raise ValueError(msg)

    existing_columns = set(dataset.column_names)
    for old_name in column_mapping:
        if old_name not in existing_columns:
            msg = (
                f"Column '{old_name}' not found in dataset. "
                f"Available: {existing_columns}"
            )
            raise ValueError(msg)

    return dataset.rename_columns(column_mapping)


def filter_by_length(
    dataset: Dataset,
    column: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> Dataset:
    """Filter dataset rows by string length in a column.

    Args:
        dataset: Dataset to filter.
        column: Column name to check length.
        min_length: Minimum length (inclusive). Defaults to None.
        max_length: Maximum length (inclusive). Defaults to None.

    Returns:
        Filtered dataset.

    Raises:
        ValueError: If dataset is None.
        ValueError: If column doesn't exist.
        ValueError: If both min and max are None.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"text": ["a", "ab", "abc", "abcd"]})
        >>> filtered = filter_by_length(ds, "text", min_length=2, max_length=3)
        >>> len(filtered)
        2
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    if column not in dataset.column_names:
        msg = (
            f"Column '{column}' not found in dataset. Available: {dataset.column_names}"
        )
        raise ValueError(msg)

    if min_length is None and max_length is None:
        msg = "At least one of min_length or max_length must be specified"
        raise ValueError(msg)

    def filter_fn(example: dict[str, Any]) -> bool:
        length = len(example[column])
        if min_length is not None and length < min_length:
            return False
        return not (max_length is not None and length > max_length)

    return dataset.filter(filter_fn)


def sample_dataset(
    dataset: Dataset,
    n: int,
    seed: int = 42,
) -> Dataset:
    """Sample n rows from a dataset.

    Args:
        dataset: Dataset to sample from.
        n: Number of rows to sample.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        Sampled dataset.

    Raises:
        ValueError: If dataset is None.
        ValueError: If n is not positive.
        ValueError: If n exceeds dataset size.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"x": list(range(100))})
        >>> sampled = sample_dataset(ds, n=10, seed=42)
        >>> len(sampled)
        10
    """
    if dataset is None:
        msg = "dataset cannot be None"
        raise ValueError(msg)

    if n <= 0:
        msg = f"n must be positive, got {n}"
        raise ValueError(msg)

    if n > len(dataset):
        msg = f"n ({n}) exceeds dataset size ({len(dataset)})"
        raise ValueError(msg)

    return dataset.shuffle(seed=seed).select(range(n))
