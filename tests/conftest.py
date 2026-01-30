"""Pytest configuration and fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_hf_api() -> Generator[MagicMock, None, None]:
    """Mock HuggingFace API for testing without network."""
    from unittest.mock import patch

    mock_api = MagicMock()

    # Create mock model info
    mock_model = MagicMock()
    mock_model.id = "test-org/test-model"
    mock_model.downloads = 1000
    mock_model.likes = 100
    mock_model.pipeline_tag = "text-classification"
    mock_model.library_name = "transformers"

    # Create mock dataset info
    mock_dataset = MagicMock()
    mock_dataset.id = "test-org/test-dataset"
    mock_dataset.downloads = 500
    mock_dataset.likes = 50
    mock_dataset.task_categories = ["text-classification"]

    mock_api.list_models.return_value = [mock_model]
    mock_api.list_datasets.return_value = [mock_dataset]

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        yield mock_api


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer for testing without models."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": [[101, 2023, 2003, 1037, 3231, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]],
    }
    return tokenizer
