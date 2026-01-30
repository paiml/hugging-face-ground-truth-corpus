"""Tests for hub search functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hf_gtc.hub.search import (
    DatasetInfo,
    ModelInfo,
    _convert_dataset_info,
    _convert_model_info,
    search_datasets,
    search_models,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self) -> None:
        """Test creating ModelInfo instance."""
        info = ModelInfo(
            model_id="test/model",
            downloads=1000,
            likes=50,
            pipeline_tag="text-classification",
            library_name="transformers",
        )
        assert info.model_id == "test/model"
        assert info.downloads == 1000
        assert info.likes == 50
        assert info.pipeline_tag == "text-classification"
        assert info.library_name == "transformers"

    def test_model_info_frozen(self) -> None:
        """Test that ModelInfo is immutable."""
        info = ModelInfo(
            model_id="test/model",
            downloads=1000,
            likes=50,
            pipeline_tag=None,
            library_name=None,
        )
        with pytest.raises(AttributeError):
            info.model_id = "changed"  # type: ignore[misc]


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_dataset_info_creation(self) -> None:
        """Test creating DatasetInfo instance."""
        info = DatasetInfo(
            dataset_id="test/dataset",
            downloads=500,
            likes=25,
            task_categories=["text-classification", "sentiment-analysis"],
        )
        assert info.dataset_id == "test/dataset"
        assert info.downloads == 500
        assert info.likes == 25
        assert "text-classification" in info.task_categories


class TestConvertModelInfo:
    """Tests for _convert_model_info function."""

    def test_convert_with_all_fields(self) -> None:
        """Test conversion with all fields populated."""
        mock = MagicMock()
        mock.id = "org/model"
        mock.downloads = 100
        mock.likes = 10
        mock.pipeline_tag = "text-generation"
        mock.library_name = "transformers"

        result = _convert_model_info(mock)

        assert result.model_id == "org/model"
        assert result.downloads == 100
        assert result.likes == 10
        assert result.pipeline_tag == "text-generation"
        assert result.library_name == "transformers"

    def test_convert_with_none_fields(self) -> None:
        """Test conversion with None fields."""
        mock = MagicMock()
        mock.id = "org/model"
        mock.downloads = None
        mock.likes = None
        mock.pipeline_tag = None
        mock.library_name = None

        result = _convert_model_info(mock)

        assert result.model_id == "org/model"
        assert result.downloads == 0
        assert result.likes == 0
        assert result.pipeline_tag is None
        assert result.library_name is None


class TestConvertDatasetInfo:
    """Tests for _convert_dataset_info function."""

    def test_convert_with_all_fields(self) -> None:
        """Test conversion with all fields populated."""
        mock = MagicMock()
        mock.id = "org/dataset"
        mock.downloads = 50
        mock.likes = 5
        mock.task_categories = ["qa", "text-classification"]

        result = _convert_dataset_info(mock)

        assert result.dataset_id == "org/dataset"
        assert result.downloads == 50
        assert result.likes == 5
        assert result.task_categories == ["qa", "text-classification"]

    def test_convert_with_none_fields(self) -> None:
        """Test conversion with None fields."""
        mock = MagicMock()
        mock.id = "org/dataset"
        mock.downloads = None
        mock.likes = None
        mock.task_categories = None

        result = _convert_dataset_info(mock)

        assert result.downloads == 0
        assert result.likes == 0
        assert result.task_categories == []


class TestSearchModels:
    """Tests for search_models function."""

    def test_search_models_success(self, mock_hf_api: MagicMock) -> None:
        """Test successful model search."""
        results = search_models(query="test", limit=5)

        assert len(results) == 1
        assert results[0].model_id == "test-org/test-model"
        assert results[0].downloads == 1000

    def test_search_models_with_filters(self, mock_hf_api: MagicMock) -> None:
        """Test model search with filters."""
        results = search_models(
            task="text-classification",
            library="transformers",
            limit=10,
        )

        assert len(results) == 1
        mock_hf_api.list_models.assert_called_once()

    def test_search_models_invalid_limit_zero(self) -> None:
        """Test that limit=0 raises ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 100"):
            search_models(limit=0)

    def test_search_models_invalid_limit_too_high(self) -> None:
        """Test that limit>100 raises ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 100"):
            search_models(limit=101)

    def test_search_models_invalid_sort(self) -> None:
        """Test that invalid sort raises ValueError."""
        with pytest.raises(ValueError, match="sort must be one of"):
            search_models(sort="invalid")

    def test_search_models_valid_sorts(self, mock_hf_api: MagicMock) -> None:
        """Test all valid sort options."""
        for sort in ["downloads", "likes", "created", "lastModified"]:
            results = search_models(sort=sort)
            assert isinstance(results, list)


class TestSearchDatasets:
    """Tests for search_datasets function."""

    def test_search_datasets_success(self, mock_hf_api: MagicMock) -> None:
        """Test successful dataset search."""
        results = search_datasets(query="test", limit=5)

        assert len(results) == 1
        assert results[0].dataset_id == "test-org/test-dataset"
        assert results[0].downloads == 500

    def test_search_datasets_invalid_limit(self) -> None:
        """Test that invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be between 1 and 100"):
            search_datasets(limit=0)

    def test_search_datasets_invalid_sort(self) -> None:
        """Test that invalid sort raises ValueError."""
        with pytest.raises(ValueError, match="sort must be one of"):
            search_datasets(sort="bad_sort")
