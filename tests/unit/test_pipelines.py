"""Tests for pipeline creation functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_gtc.inference.pipelines import (
    SUPPORTED_TASKS,
    create_pipeline,
    list_supported_tasks,
)


class TestListSupportedTasks:
    """Tests for list_supported_tasks function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        result = list_supported_tasks()
        assert isinstance(result, list)

    def test_returns_non_empty(self) -> None:
        """Test that list is not empty."""
        result = list_supported_tasks()
        assert len(result) > 0

    def test_returns_sorted(self) -> None:
        """Test that list is sorted."""
        result = list_supported_tasks()
        assert result == sorted(result)

    def test_contains_common_tasks(self) -> None:
        """Test that common tasks are included."""
        result = list_supported_tasks()
        assert "text-classification" in result
        assert "sentiment-analysis" in result
        assert "text-generation" in result
        assert "question-answering" in result

    def test_all_strings(self) -> None:
        """Test that all elements are strings."""
        result = list_supported_tasks()
        assert all(isinstance(t, str) for t in result)


class TestSupportedTasks:
    """Tests for SUPPORTED_TASKS constant."""

    def test_is_frozenset(self) -> None:
        """Test that SUPPORTED_TASKS is a frozenset."""
        assert isinstance(SUPPORTED_TASKS, frozenset)

    def test_contains_expected_tasks(self) -> None:
        """Test that expected tasks are present."""
        expected = {
            "text-classification",
            "text-generation",
            "sentiment-analysis",
            "question-answering",
            "summarization",
            "translation",
            "fill-mask",
            "ner",
            "token-classification",
        }
        assert expected.issubset(SUPPORTED_TASKS)


class TestCreatePipeline:
    """Tests for create_pipeline function."""

    def test_invalid_task_raises_error(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported task"):
            create_pipeline("definitely-not-a-real-task")

    def test_valid_task_accepted(self) -> None:
        """Test that valid tasks don't raise on task validation."""
        # We can test that validation passes by mocking the pipeline call
        # Must patch where it's imported as _hf_pipeline
        with (
            patch("hf_gtc.inference.pipelines._hf_pipeline") as mock_pipeline,
            patch("hf_gtc.inference.device.get_device", return_value="cpu"),
        ):
            mock_pipeline.return_value = MagicMock()
            result = create_pipeline("text-classification")
            mock_pipeline.assert_called_once()
            assert result is not None

    def test_auto_device_detection(self) -> None:
        """Test that device is auto-detected when not specified."""
        with (
            patch("hf_gtc.inference.pipelines._hf_pipeline") as mock_pipeline,
            patch("hf_gtc.inference.device.get_device", return_value="cuda"),
        ):
            mock_pipeline.return_value = MagicMock()
            create_pipeline("text-classification")

            # Should pass device=0 for CUDA
            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs["device"] == 0

    def test_explicit_device(self) -> None:
        """Test that explicit device is used when provided."""
        with patch("hf_gtc.inference.pipelines._hf_pipeline") as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            create_pipeline("text-classification", device="cpu")

            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs["device"] == "cpu"

    def test_model_passed_through(self) -> None:
        """Test that model parameter is passed through."""
        with (
            patch("hf_gtc.inference.pipelines._hf_pipeline") as mock_pipeline,
            patch("hf_gtc.inference.device.get_device", return_value="cpu"),
        ):
            mock_pipeline.return_value = MagicMock()
            create_pipeline("text-classification", model="my-model")

            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs["model"] == "my-model"

    def test_kwargs_passed_through(self) -> None:
        """Test that additional kwargs are passed through."""
        with (
            patch("hf_gtc.inference.pipelines._hf_pipeline") as mock_pipeline,
            patch("hf_gtc.inference.device.get_device", return_value="cpu"),
        ):
            mock_pipeline.return_value = MagicMock()
            create_pipeline(
                "text-classification",
                batch_size=8,
                framework="pt",
            )

            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs["batch_size"] == 8
            assert call_kwargs["framework"] == "pt"

    def test_mps_device_handling(self) -> None:
        """Test MPS device is passed correctly."""
        with (
            patch("hf_gtc.inference.pipelines._hf_pipeline") as mock_pipeline,
            patch("hf_gtc.inference.device.get_device", return_value="mps"),
        ):
            mock_pipeline.return_value = MagicMock()
            create_pipeline("text-classification")

            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs["device"] == "mps"
