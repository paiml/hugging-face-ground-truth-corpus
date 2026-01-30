"""Tests for device detection and management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_gtc.inference.device import (
    clear_gpu_memory,
    get_device,
    get_device_map,
    get_gpu_memory_info,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_returns_valid_type(self) -> None:
        """Test that get_device returns a valid device type."""
        device = get_device()
        assert device in ("cuda", "mps", "cpu")

    def test_get_device_cuda_available(self) -> None:
        """Test CUDA detection when available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device()
            assert device == "cuda"

    def test_get_device_mps_available(self) -> None:
        """Test MPS detection when available (no CUDA)."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            device = get_device()
            assert device == "mps"

    def test_get_device_cpu_fallback(self) -> None:
        """Test CPU fallback when no GPU available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = get_device()
            assert device == "cpu"


class TestGetDeviceMap:
    """Tests for get_device_map function."""

    def test_small_model_returns_auto(self) -> None:
        """Test that small models get 'auto' device map."""
        with (
            patch("hf_gtc.inference.device.get_device", return_value="cuda"),
            patch(
                "hf_gtc.inference.device.get_gpu_memory_info",
                return_value={"free_gb": 16.0},
            ),
        ):
            result = get_device_map(1.0)
            assert result == "auto"

    def test_cpu_device_returns_cpu(self) -> None:
        """Test that CPU device returns 'cpu' map."""
        with patch("hf_gtc.inference.device.get_device", return_value="cpu"):
            result = get_device_map(1.0)
            assert result == "cpu"

    def test_large_model_returns_balanced(self) -> None:
        """Test that large models get 'balanced' device map."""
        with (
            patch("hf_gtc.inference.device.get_device", return_value="cuda"),
            patch(
                "hf_gtc.inference.device.get_gpu_memory_info",
                return_value={"free_gb": 8.0},
            ),
        ):
            result = get_device_map(10.0)
            assert result == "balanced"

    def test_invalid_model_size_negative(self) -> None:
        """Test that negative model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            get_device_map(-1.0)

    def test_invalid_model_size_zero(self) -> None:
        """Test that zero model size raises ValueError."""
        with pytest.raises(ValueError, match="model_size_gb must be positive"):
            get_device_map(0.0)

    def test_invalid_memory_fraction_too_high(self) -> None:
        """Test that memory fraction > 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_memory_fraction must be between"):
            get_device_map(1.0, max_memory_fraction=1.5)

    def test_invalid_memory_fraction_zero(self) -> None:
        """Test that memory fraction = 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_memory_fraction must be between"):
            get_device_map(1.0, max_memory_fraction=0.0)

    def test_invalid_memory_fraction_negative(self) -> None:
        """Test that negative memory fraction raises ValueError."""
        with pytest.raises(ValueError, match="max_memory_fraction must be between"):
            get_device_map(1.0, max_memory_fraction=-0.5)


class TestGetGpuMemoryInfo:
    """Tests for get_gpu_memory_info function."""

    def test_returns_dict_with_required_keys(self) -> None:
        """Test that return dict has required keys."""
        info = get_gpu_memory_info()
        assert "total_gb" in info
        assert "used_gb" in info
        assert "free_gb" in info

    def test_returns_float_values(self) -> None:
        """Test that all values are floats."""
        info = get_gpu_memory_info()
        assert all(isinstance(v, float) for v in info.values())

    def test_returns_non_negative_values(self) -> None:
        """Test that all values are non-negative."""
        info = get_gpu_memory_info()
        assert all(v >= 0 for v in info.values())

    def test_no_cuda_returns_zeros(self) -> None:
        """Test that no CUDA returns zeros."""
        with patch("torch.cuda.is_available", return_value=False):
            info = get_gpu_memory_info()
            assert info == {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

    def test_cuda_available_returns_real_values(self) -> None:
        """Test CUDA memory info retrieval."""
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16 GB

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.memory_reserved", return_value=4 * 1024**3),
            patch("torch.cuda.memory_allocated", return_value=2 * 1024**3),
        ):
            info = get_gpu_memory_info()
            assert info["total_gb"] == pytest.approx(16.0, rel=0.01)
            assert info["used_gb"] == pytest.approx(2.0, rel=0.01)
            assert info["free_gb"] == pytest.approx(12.0, rel=0.01)


class TestClearGpuMemory:
    """Tests for clear_gpu_memory function."""

    def test_clear_gpu_memory_no_cuda(self) -> None:
        """Test that clear works when no CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            # Should not raise
            clear_gpu_memory()

    def test_clear_gpu_memory_with_cuda(self) -> None:
        """Test that clear calls CUDA functions."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty,
            patch("torch.cuda.synchronize") as mock_sync,
        ):
            clear_gpu_memory()
            mock_empty.assert_called_once()
            mock_sync.assert_called_once()
