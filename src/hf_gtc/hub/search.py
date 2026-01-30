"""Search utilities for HuggingFace Hub.

This module provides functions for searching models and datasets
on the HuggingFace Hub with filtering and sorting capabilities.

Examples:
    >>> from hf_gtc.hub.search import search_models, search_datasets
    >>> # Functions are callable
    >>> callable(search_models) and callable(search_datasets)
    True
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import DatasetInfo as HFDatasetInfo
    from huggingface_hub import ModelInfo as HFModelInfo


@dataclass(frozen=True)
class ModelInfo:
    """Information about a HuggingFace model.

    Attributes:
        model_id: The model identifier (e.g., "bert-base-uncased").
        downloads: Number of downloads.
        likes: Number of likes.
        pipeline_tag: The pipeline tag (e.g., "text-classification").
        library_name: The library name (e.g., "transformers").

    Examples:
        >>> info = ModelInfo(
        ...     model_id="test/model",
        ...     downloads=1000,
        ...     likes=50,
        ...     pipeline_tag="text-classification",
        ...     library_name="transformers",
        ... )
        >>> info.model_id
        'test/model'
        >>> info.downloads
        1000
    """

    model_id: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    library_name: str | None


@dataclass(frozen=True)
class DatasetInfo:
    """Information about a HuggingFace dataset.

    Attributes:
        dataset_id: The dataset identifier (e.g., "imdb").
        downloads: Number of downloads.
        likes: Number of likes.
        task_categories: List of task categories.

    Examples:
        >>> info = DatasetInfo(
        ...     dataset_id="test/dataset",
        ...     downloads=500,
        ...     likes=25,
        ...     task_categories=["text-classification"],
        ... )
        >>> info.dataset_id
        'test/dataset'
    """

    dataset_id: str
    downloads: int
    likes: int
    task_categories: list[str]


def _convert_model_info(hf_info: HFModelInfo) -> ModelInfo:
    """Convert HuggingFace ModelInfo to our ModelInfo dataclass.

    Args:
        hf_info: HuggingFace ModelInfo object.

    Returns:
        Converted ModelInfo dataclass.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> mock = MagicMock()
        >>> mock.id = "test/model"
        >>> mock.downloads = 100
        >>> mock.likes = 10
        >>> mock.pipeline_tag = "text-generation"
        >>> mock.library_name = "transformers"
        >>> result = _convert_model_info(mock)
        >>> result.model_id
        'test/model'
    """
    return ModelInfo(
        model_id=hf_info.id,
        downloads=hf_info.downloads or 0,
        likes=hf_info.likes or 0,
        pipeline_tag=hf_info.pipeline_tag,
        library_name=hf_info.library_name,
    )


def _convert_dataset_info(hf_info: HFDatasetInfo) -> DatasetInfo:
    """Convert HuggingFace DatasetInfo to our DatasetInfo dataclass.

    Args:
        hf_info: HuggingFace DatasetInfo object.

    Returns:
        Converted DatasetInfo dataclass.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> mock = MagicMock()
        >>> mock.id = "test/dataset"
        >>> mock.downloads = 50
        >>> mock.likes = 5
        >>> mock.task_categories = ["text-classification"]
        >>> result = _convert_dataset_info(mock)
        >>> result.dataset_id
        'test/dataset'
    """
    return DatasetInfo(
        dataset_id=hf_info.id,
        downloads=hf_info.downloads or 0,
        likes=hf_info.likes or 0,
        task_categories=getattr(hf_info, "task_categories", None) or [],
    )


def search_models(
    query: str | None = None,
    *,
    task: str | None = None,
    library: str | None = None,
    limit: int = 10,
    sort: str = "downloads",
) -> list[ModelInfo]:
    """Search for models on the HuggingFace Hub.

    Args:
        query: Search query string. Defaults to None.
        task: Filter by pipeline task (e.g., "text-classification").
        library: Filter by library (e.g., "transformers", "pytorch").
        limit: Maximum number of results. Defaults to 10. Must be 1-100.
        sort: Sort field ("downloads", "likes", "created"). Defaults to "downloads".

    Returns:
        List of ModelInfo objects matching the search criteria.

    Raises:
        ValueError: If limit is not between 1 and 100.
        ValueError: If sort is not a valid sort field.

    Examples:
        >>> # Validate parameter constraints
        >>> search_models(limit=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: limit must be between 1 and 100, got 0

        >>> search_models(limit=101)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: limit must be between 1 and 100, got 101

        >>> search_models(sort="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sort must be one of ...
    """
    # Validate parameters
    if not 1 <= limit <= 100:
        msg = f"limit must be between 1 and 100, got {limit}"
        raise ValueError(msg)

    valid_sorts = {"downloads", "likes", "created", "lastModified"}
    if sort not in valid_sorts:
        msg = f"sort must be one of {valid_sorts}, got {sort!r}"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(
        search=query,
        pipeline_tag=task,
        library=library,
        limit=limit,
        sort=sort,
        direction=-1,  # Descending
    )

    return [_convert_model_info(m) for m in models]


def search_datasets(
    query: str | None = None,
    *,
    task: str | None = None,
    limit: int = 10,
    sort: str = "downloads",
) -> list[DatasetInfo]:
    """Search for datasets on the HuggingFace Hub.

    Args:
        query: Search query string. Defaults to None.
        task: Filter by task category (e.g., "text-classification").
        limit: Maximum number of results. Defaults to 10. Must be 1-100.
        sort: Sort field ("downloads", "likes", "created"). Defaults to "downloads".

    Returns:
        List of DatasetInfo objects matching the search criteria.

    Raises:
        ValueError: If limit is not between 1 and 100.
        ValueError: If sort is not a valid sort field.

    Examples:
        >>> # Validate parameter constraints
        >>> search_datasets(limit=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: limit must be between 1 and 100, got 0

        >>> search_datasets(sort="bad")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sort must be one of ...
    """
    # Validate parameters
    if not 1 <= limit <= 100:
        msg = f"limit must be between 1 and 100, got {limit}"
        raise ValueError(msg)

    valid_sorts = {"downloads", "likes", "created", "lastModified"}
    if sort not in valid_sorts:
        msg = f"sort must be one of {valid_sorts}, got {sort!r}"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()
    datasets = api.list_datasets(
        search=query,
        task_categories=task,
        limit=limit,
        sort=sort,
        direction=-1,  # Descending
    )

    return [_convert_dataset_info(d) for d in datasets]


def iter_models(
    *,
    task: str | None = None,
    library: str | None = None,
) -> Iterator[ModelInfo]:
    """Iterate over all models on the HuggingFace Hub.

    This is a streaming API that yields models one at a time,
    suitable for processing large numbers of models without
    loading them all into memory.

    Args:
        task: Filter by pipeline task (e.g., "text-classification").
        library: Filter by library (e.g., "transformers").

    Yields:
        ModelInfo objects matching the filter criteria.

    Examples:
        >>> # Just verify it returns an iterator
        >>> it = iter_models()
        >>> hasattr(it, '__iter__') and hasattr(it, '__next__')
        True
    """
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(
        pipeline_tag=task,
        library=library,
    )

    for model in models:
        yield _convert_model_info(model)
