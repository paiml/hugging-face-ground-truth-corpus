"""Tests for safetensors serialization utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.safetensors import (
    VALID_DTYPES,
    VALID_TENSOR_FORMATS,
    FileInfo,
    LoadConfig,
    SaveConfig,
    TensorFormat,
    TensorInfo,
    calculate_memory_savings,
    create_load_config,
    create_metadata,
    create_save_config,
    estimate_file_size,
    format_size,
    get_recommended_dtype,
    list_dtypes,
    list_tensor_formats,
    validate_load_config,
    validate_save_config,
    validate_tensor_name,
)


class TestTensorFormat:
    """Tests for TensorFormat enum."""

    def test_pytorch_value(self) -> None:
        """Test PyTorch format value."""
        assert TensorFormat.PYTORCH.value == "pt"

    def test_numpy_value(self) -> None:
        """Test NumPy format value."""
        assert TensorFormat.NUMPY.value == "np"

    def test_tensorflow_value(self) -> None:
        """Test TensorFlow format value."""
        assert TensorFormat.TENSORFLOW.value == "tf"

    def test_flax_value(self) -> None:
        """Test Flax format value."""
        assert TensorFormat.FLAX.value == "flax"

    def test_all_formats_in_valid_set(self) -> None:
        """Test all enum values are in VALID_TENSOR_FORMATS."""
        for fmt in TensorFormat:
            assert fmt.value in VALID_TENSOR_FORMATS


class TestSaveConfig:
    """Tests for SaveConfig dataclass."""

    def test_create_with_metadata(self) -> None:
        """Test creating config with metadata."""
        config = SaveConfig(
            metadata={"format": "pt"},
            force_contiguous=True,
            shared_memory=False,
        )
        assert config.metadata == {"format": "pt"}
        assert config.force_contiguous is True
        assert config.shared_memory is False

    def test_create_without_metadata(self) -> None:
        """Test creating config without metadata."""
        config = SaveConfig(
            metadata=None,
            force_contiguous=False,
            shared_memory=True,
        )
        assert config.metadata is None
        assert config.force_contiguous is False
        assert config.shared_memory is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = SaveConfig(
            metadata=None,
            force_contiguous=True,
            shared_memory=False,
        )
        with pytest.raises(AttributeError):
            config.force_contiguous = False  # type: ignore[misc]


class TestLoadConfig:
    """Tests for LoadConfig dataclass."""

    def test_create_load_config(self) -> None:
        """Test creating load config."""
        config = LoadConfig(
            device="cuda:0",
            framework=TensorFormat.PYTORCH,
            strict=True,
        )
        assert config.device == "cuda:0"
        assert config.framework == TensorFormat.PYTORCH
        assert config.strict is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = LoadConfig(
            device="cpu",
            framework=TensorFormat.NUMPY,
            strict=False,
        )
        with pytest.raises(AttributeError):
            config.device = "cuda"  # type: ignore[misc]


class TestTensorInfo:
    """Tests for TensorInfo dataclass."""

    def test_create_tensor_info(self) -> None:
        """Test creating tensor info."""
        info = TensorInfo(
            name="weight",
            shape=(768, 768),
            dtype="F16",
            size_bytes=1179648,
        )
        assert info.name == "weight"
        assert info.shape == (768, 768)
        assert info.dtype == "F16"
        assert info.size_bytes == 1179648

    def test_frozen(self) -> None:
        """Test info is immutable."""
        info = TensorInfo(
            name="bias",
            shape=(768,),
            dtype="F32",
            size_bytes=3072,
        )
        with pytest.raises(AttributeError):
            info.name = "new_name"  # type: ignore[misc]


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_create_file_info(self) -> None:
        """Test creating file info."""
        tensor = TensorInfo("weight", (768, 768), "F16", 1179648)
        info = FileInfo(
            path="model.safetensors",
            total_size_bytes=1000000,
            num_tensors=1,
            metadata={"format": "pt"},
            tensors=(tensor,),
        )
        assert info.path == "model.safetensors"
        assert info.total_size_bytes == 1000000
        assert info.num_tensors == 1
        assert info.metadata == {"format": "pt"}
        assert len(info.tensors) == 1

    def test_empty_tensors(self) -> None:
        """Test file info with no tensors."""
        info = FileInfo(
            path="empty.safetensors",
            total_size_bytes=0,
            num_tensors=0,
            metadata={},
            tensors=(),
        )
        assert info.num_tensors == 0
        assert len(info.tensors) == 0


class TestValidateSaveConfig:
    """Tests for validate_save_config function."""

    def test_valid_config_with_metadata(self) -> None:
        """Test validating config with valid metadata."""
        config = SaveConfig(
            metadata={"key": "value"},
            force_contiguous=True,
            shared_memory=False,
        )
        validate_save_config(config)  # Should not raise

    def test_valid_config_without_metadata(self) -> None:
        """Test validating config without metadata."""
        config = SaveConfig(
            metadata=None,
            force_contiguous=True,
            shared_memory=False,
        )
        validate_save_config(config)  # Should not raise

    def test_invalid_metadata_key_type(self) -> None:
        """Test validation fails with non-string metadata key."""
        config = SaveConfig(
            metadata={123: "value"},  # type: ignore[dict-item]
            force_contiguous=True,
            shared_memory=False,
        )
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_save_config(config)

    def test_invalid_metadata_value_type(self) -> None:
        """Test validation fails with non-string metadata value."""
        config = SaveConfig(
            metadata={"key": 123},  # type: ignore[dict-item]
            force_contiguous=True,
            shared_memory=False,
        )
        with pytest.raises(ValueError, match="values must be strings"):
            validate_save_config(config)


class TestValidateLoadConfig:
    """Tests for validate_load_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = LoadConfig(
            device="cpu",
            framework=TensorFormat.PYTORCH,
            strict=True,
        )
        validate_load_config(config)  # Should not raise

    def test_empty_device(self) -> None:
        """Test validation fails with empty device."""
        config = LoadConfig(
            device="",
            framework=TensorFormat.PYTORCH,
            strict=True,
        )
        with pytest.raises(ValueError, match="device cannot be empty"):
            validate_load_config(config)


class TestCreateSaveConfig:
    """Tests for create_save_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_save_config()
        assert config.metadata is None
        assert config.force_contiguous is True
        assert config.shared_memory is False

    def test_with_metadata(self) -> None:
        """Test with metadata."""
        config = create_save_config(metadata={"format": "pt"})
        assert config.metadata == {"format": "pt"}

    def test_custom_flags(self) -> None:
        """Test with custom flags."""
        config = create_save_config(force_contiguous=False, shared_memory=True)
        assert config.force_contiguous is False
        assert config.shared_memory is True


class TestCreateLoadConfig:
    """Tests for create_load_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_load_config()
        assert config.device == "cpu"
        assert config.framework == TensorFormat.PYTORCH
        assert config.strict is True

    def test_custom_device(self) -> None:
        """Test with custom device."""
        config = create_load_config(device="cuda:0")
        assert config.device == "cuda:0"

    def test_numpy_framework(self) -> None:
        """Test with numpy framework."""
        config = create_load_config(framework="np")
        assert config.framework == TensorFormat.NUMPY

    def test_tensorflow_framework(self) -> None:
        """Test with tensorflow framework."""
        config = create_load_config(framework="tf")
        assert config.framework == TensorFormat.TENSORFLOW

    def test_flax_framework(self) -> None:
        """Test with flax framework."""
        config = create_load_config(framework="flax")
        assert config.framework == TensorFormat.FLAX

    def test_non_strict(self) -> None:
        """Test with non-strict mode."""
        config = create_load_config(strict=False)
        assert config.strict is False

    def test_invalid_framework(self) -> None:
        """Test with invalid framework."""
        with pytest.raises(ValueError, match="framework must be one of"):
            create_load_config(framework="invalid")

    def test_empty_device(self) -> None:
        """Test with empty device."""
        with pytest.raises(ValueError, match="device cannot be empty"):
            create_load_config(device="")


class TestListTensorFormats:
    """Tests for list_tensor_formats function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        formats = list_tensor_formats()
        assert isinstance(formats, list)

    def test_contains_expected_formats(self) -> None:
        """Test contains expected formats."""
        formats = list_tensor_formats()
        assert "pt" in formats
        assert "np" in formats
        assert "tf" in formats
        assert "flax" in formats

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        formats = list_tensor_formats()
        assert formats == sorted(formats)


class TestListDtypes:
    """Tests for list_dtypes function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        dtypes = list_dtypes()
        assert isinstance(dtypes, list)

    def test_contains_expected_dtypes(self) -> None:
        """Test contains expected dtypes."""
        dtypes = list_dtypes()
        assert "F32" in dtypes
        assert "F16" in dtypes
        assert "BF16" in dtypes
        assert "I64" in dtypes
        assert "BOOL" in dtypes

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        dtypes = list_dtypes()
        assert dtypes == sorted(dtypes)

    def test_all_valid_dtypes_present(self) -> None:
        """Test all VALID_DTYPES are in the list."""
        dtypes = list_dtypes()
        for dtype in VALID_DTYPES:
            assert dtype in dtypes


class TestEstimateFileSize:
    """Tests for estimate_file_size function."""

    def test_single_2d_tensor_f16(self) -> None:
        """Test single 2D tensor with F16."""
        # 768 * 768 = 589824 elements
        # Plus 768 elements = 590592 total
        # 590592 * 2 bytes = 1181184
        size = estimate_file_size(((768, 768), (768,)), dtype="F16")
        assert size == 1181184

    def test_single_tensor_f32(self) -> None:
        """Test single tensor with F32."""
        # 1000 * 1000 = 1000000 elements * 4 bytes = 4000000
        size = estimate_file_size(((1000, 1000),), dtype="F32")
        assert size == 4000000

    def test_bf16_dtype(self) -> None:
        """Test BF16 dtype."""
        size = estimate_file_size(((100, 100),), dtype="BF16")
        assert size == 20000  # 10000 * 2

    def test_i64_dtype(self) -> None:
        """Test I64 dtype."""
        size = estimate_file_size(((100,),), dtype="I64")
        assert size == 800  # 100 * 8

    def test_i32_dtype(self) -> None:
        """Test I32 dtype."""
        size = estimate_file_size(((100,),), dtype="I32")
        assert size == 400  # 100 * 4

    def test_i16_dtype(self) -> None:
        """Test I16 dtype."""
        size = estimate_file_size(((100,),), dtype="I16")
        assert size == 200  # 100 * 2

    def test_i8_dtype(self) -> None:
        """Test I8 dtype."""
        size = estimate_file_size(((100,),), dtype="I8")
        assert size == 100  # 100 * 1

    def test_u8_dtype(self) -> None:
        """Test U8 dtype."""
        size = estimate_file_size(((100,),), dtype="U8")
        assert size == 100  # 100 * 1

    def test_bool_dtype(self) -> None:
        """Test BOOL dtype."""
        size = estimate_file_size(((100,),), dtype="BOOL")
        assert size == 100  # 100 * 1

    def test_empty_shapes(self) -> None:
        """Test empty shapes tuple."""
        with pytest.raises(ValueError, match="shapes cannot be empty"):
            estimate_file_size(())

    def test_invalid_dtype(self) -> None:
        """Test invalid dtype."""
        with pytest.raises(ValueError, match="dtype must be one of"):
            estimate_file_size(((10, 10),), dtype="INVALID")  # type: ignore[arg-type]

    def test_negative_dimension(self) -> None:
        """Test negative dimension."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            estimate_file_size(((10, -5),))

    def test_zero_dimension(self) -> None:
        """Test zero dimension."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            estimate_file_size(((10, 0),))

    def test_empty_shape_skipped(self) -> None:
        """Test empty shape is skipped."""
        size = estimate_file_size(((), (10,)), dtype="F32")
        assert size == 40  # 10 * 4

    def test_3d_tensor(self) -> None:
        """Test 3D tensor shape."""
        size = estimate_file_size(((2, 3, 4),), dtype="F32")
        assert size == 96  # 2 * 3 * 4 * 4 = 96


class TestCalculateMemorySavings:
    """Tests for calculate_memory_savings function."""

    def test_f32_to_f16(self) -> None:
        """Test F32 to F16 conversion."""
        savings = calculate_memory_savings("F32", "F16")
        assert savings == pytest.approx(0.5)

    def test_f32_to_f32(self) -> None:
        """Test same dtype (no change)."""
        savings = calculate_memory_savings("F32", "F32")
        assert savings == pytest.approx(1.0)

    def test_f16_to_f32(self) -> None:
        """Test F16 to F32 (increase)."""
        savings = calculate_memory_savings("F16", "F32")
        assert savings == pytest.approx(2.0)

    def test_f32_to_i8(self) -> None:
        """Test F32 to I8 conversion."""
        savings = calculate_memory_savings("F32", "I8")
        assert savings == pytest.approx(0.25)

    def test_i64_to_i32(self) -> None:
        """Test I64 to I32 conversion."""
        savings = calculate_memory_savings("I64", "I32")
        assert savings == pytest.approx(0.5)

    def test_invalid_original_dtype(self) -> None:
        """Test invalid original dtype."""
        with pytest.raises(ValueError, match="original_dtype must be one of"):
            calculate_memory_savings("INVALID", "F16")  # type: ignore[arg-type]

    def test_invalid_target_dtype(self) -> None:
        """Test invalid target dtype."""
        with pytest.raises(ValueError, match="target_dtype must be one of"):
            calculate_memory_savings("F32", "INVALID")  # type: ignore[arg-type]


class TestFormatSize:
    """Tests for format_size function."""

    def test_bytes(self) -> None:
        """Test formatting bytes."""
        assert format_size(500) == "500 B"

    def test_zero_bytes(self) -> None:
        """Test formatting zero bytes."""
        assert format_size(0) == "0 B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.00 KB"

    def test_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert format_size(1048576) == "1.00 MB"

    def test_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        assert format_size(1073741824) == "1.00 GB"

    def test_terabytes(self) -> None:
        """Test formatting terabytes."""
        size = 1024**4
        assert format_size(size) == "1.00 TB"

    def test_petabytes(self) -> None:
        """Test formatting petabytes."""
        size = 1024**5
        assert format_size(size) == "1.00 PB"

    def test_negative_size(self) -> None:
        """Test negative size raises error."""
        with pytest.raises(ValueError, match="size_bytes cannot be negative"):
            format_size(-1)

    def test_fractional_kb(self) -> None:
        """Test fractional KB."""
        assert format_size(1536) == "1.50 KB"

    def test_large_bytes_under_kb(self) -> None:
        """Test 1023 bytes (just under 1KB)."""
        assert format_size(1023) == "1023 B"


class TestGetRecommendedDtype:
    """Tests for get_recommended_dtype function."""

    def test_small_model(self) -> None:
        """Test small model recommendation."""
        assert get_recommended_dtype("small") == "F32"

    def test_medium_model(self) -> None:
        """Test medium model recommendation."""
        assert get_recommended_dtype("medium") == "F16"

    def test_large_model(self) -> None:
        """Test large model recommendation."""
        assert get_recommended_dtype("large") == "F16"

    def test_xlarge_model(self) -> None:
        """Test xlarge model recommendation."""
        assert get_recommended_dtype("xlarge") == "BF16"

    def test_invalid_size(self) -> None:
        """Test invalid model size."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            get_recommended_dtype("invalid")


class TestValidateTensorName:
    """Tests for validate_tensor_name function."""

    def test_valid_name(self) -> None:
        """Test valid tensor name."""
        assert validate_tensor_name("model.layers.0.weight") is True

    def test_simple_name(self) -> None:
        """Test simple tensor name."""
        assert validate_tensor_name("weight") is True

    def test_empty_name(self) -> None:
        """Test empty name raises error."""
        with pytest.raises(ValueError, match="tensor name cannot be empty"):
            validate_tensor_name("")

    def test_null_char(self) -> None:
        """Test name with null character."""
        with pytest.raises(ValueError, match="invalid character"):
            validate_tensor_name("weight\x00")

    def test_newline_char(self) -> None:
        """Test name with newline character."""
        with pytest.raises(ValueError, match="invalid character"):
            validate_tensor_name("weight\n")

    def test_carriage_return_char(self) -> None:
        """Test name with carriage return character."""
        with pytest.raises(ValueError, match="invalid character"):
            validate_tensor_name("weight\r")


class TestCreateMetadata:
    """Tests for create_metadata function."""

    def test_default_values(self) -> None:
        """Test default metadata values."""
        meta = create_metadata()
        assert meta["format_version"] == "1.0"
        assert meta["framework"] == "pt"

    def test_custom_framework(self) -> None:
        """Test custom framework."""
        meta = create_metadata(framework="np")
        assert meta["framework"] == "np"

    def test_custom_version(self) -> None:
        """Test custom version."""
        meta = create_metadata(format_version="2.0")
        assert meta["format_version"] == "2.0"

    def test_additional_kwargs(self) -> None:
        """Test additional keyword arguments."""
        meta = create_metadata(author="user", model="gpt2")
        assert "author" in meta
        assert meta["author"] == "user"
        assert "model" in meta
        assert meta["model"] == "gpt2"

    def test_combined_kwargs(self) -> None:
        """Test combining default and custom kwargs."""
        meta = create_metadata(framework="tf", custom_key="custom_value")
        assert meta["framework"] == "tf"
        assert meta["format_version"] == "1.0"
        assert meta["custom_key"] == "custom_value"
