"""Tests for GGUF export utilities."""

from __future__ import annotations

import pytest

from hf_gtc.deployment.gguf import (
    GGUFArchitecture,
    GGUFConfig,
    GGUFEndian,
    GGUFExportResult,
    GGUFMetadata,
    GGUFModelInfo,
    GGUFQuantType,
    create_gguf_config,
    create_gguf_export_result,
    create_gguf_metadata,
    estimate_gguf_size,
    format_gguf_export_result,
    format_gguf_model_info,
    get_gguf_architecture,
    get_gguf_config_dict,
    get_gguf_filename,
    get_gguf_metadata_dict,
    get_gguf_quant_type,
    get_recommended_gguf_quant,
    list_gguf_architectures,
    list_gguf_quant_types,
    validate_gguf_architecture,
    validate_gguf_config,
    validate_gguf_metadata,
    validate_gguf_quant_type,
)


class TestGGUFQuantType:
    """Tests for GGUFQuantType enum."""

    def test_q4_k_m_value(self) -> None:
        """Test Q4_K_M value."""
        assert GGUFQuantType.Q4_K_M.value == "q4_k_m"

    def test_q8_0_value(self) -> None:
        """Test Q8_0 value."""
        assert GGUFQuantType.Q8_0.value == "q8_0"

    def test_f16_value(self) -> None:
        """Test F16 value."""
        assert GGUFQuantType.F16.value == "f16"


class TestGGUFArchitecture:
    """Tests for GGUFArchitecture enum."""

    def test_llama_value(self) -> None:
        """Test LLAMA value."""
        assert GGUFArchitecture.LLAMA.value == "llama"

    def test_mistral_value(self) -> None:
        """Test MISTRAL value."""
        assert GGUFArchitecture.MISTRAL.value == "mistral"

    def test_falcon_value(self) -> None:
        """Test FALCON value."""
        assert GGUFArchitecture.FALCON.value == "falcon"


class TestGGUFEndian:
    """Tests for GGUFEndian enum."""

    def test_little_value(self) -> None:
        """Test LITTLE value."""
        assert GGUFEndian.LITTLE.value == "little"

    def test_big_value(self) -> None:
        """Test BIG value."""
        assert GGUFEndian.BIG.value == "big"


class TestGGUFConfig:
    """Tests for GGUFConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = GGUFConfig()
        assert config.quant_type == GGUFQuantType.Q4_K_M
        assert config.architecture is None
        assert config.vocab_only is False
        assert config.concurrency == 4

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = GGUFConfig(
            quant_type=GGUFQuantType.Q8_0, architecture=GGUFArchitecture.LLAMA
        )
        assert config.quant_type == GGUFQuantType.Q8_0
        assert config.architecture == GGUFArchitecture.LLAMA

    def test_frozen(self) -> None:
        """Test that GGUFConfig is immutable."""
        config = GGUFConfig()
        with pytest.raises(AttributeError):
            config.concurrency = 8  # type: ignore[misc]


class TestGGUFMetadata:
    """Tests for GGUFMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        meta = GGUFMetadata()
        assert meta.name == ""
        assert meta.author == ""

    def test_custom_values(self) -> None:
        """Test custom values."""
        meta = GGUFMetadata(name="my-model", author="me")
        assert meta.name == "my-model"
        assert meta.author == "me"


class TestGGUFExportResult:
    """Tests for GGUFExportResult dataclass."""

    def test_creation(self) -> None:
        """Test creating result."""
        result = GGUFExportResult(
            output_path="/model.gguf",
            original_size_mb=14000,
            exported_size_mb=4000,
            compression_ratio=3.5,
            quant_type=GGUFQuantType.Q4_K_M,
        )
        assert result.compression_ratio == pytest.approx(3.5)


class TestGGUFModelInfo:
    """Tests for GGUFModelInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating info."""
        info = GGUFModelInfo(
            path="/model.gguf", size_mb=4000, quant_type=GGUFQuantType.Q4_K_M
        )
        assert info.size_mb == 4000


class TestValidateGGUFConfig:
    """Tests for validate_gguf_config function."""

    def test_valid_config(self) -> None:
        """Test validating valid config."""
        config = GGUFConfig(quant_type=GGUFQuantType.Q4_K_M)
        validate_gguf_config(config)  # Should not raise

    def test_none_config_raises_error(self) -> None:
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_gguf_config(None)  # type: ignore[arg-type]

    def test_zero_concurrency_raises_error(self) -> None:
        """Test that zero concurrency raises ValueError."""
        config = GGUFConfig(concurrency=0)
        with pytest.raises(ValueError, match="concurrency must be positive"):
            validate_gguf_config(config)


class TestValidateGGUFMetadata:
    """Tests for validate_gguf_metadata function."""

    def test_valid_metadata(self) -> None:
        """Test validating valid metadata."""
        meta = GGUFMetadata(name="my-model")
        validate_gguf_metadata(meta)  # Should not raise

    def test_none_metadata_raises_error(self) -> None:
        """Test that None metadata raises ValueError."""
        with pytest.raises(ValueError, match="metadata cannot be None"):
            validate_gguf_metadata(None)  # type: ignore[arg-type]


class TestCreateGGUFConfig:
    """Tests for create_gguf_config function."""

    def test_creates_config(self) -> None:
        """Test creating config."""
        config = create_gguf_config(quant_type="q4_k_m")
        assert config.quant_type == GGUFQuantType.Q4_K_M

    def test_with_architecture(self) -> None:
        """Test with architecture."""
        config = create_gguf_config(architecture="llama")
        assert config.architecture == GGUFArchitecture.LLAMA

    def test_enum_quant_type(self) -> None:
        """Test with enum quant type."""
        config = create_gguf_config(quant_type=GGUFQuantType.Q8_0)
        assert config.quant_type == GGUFQuantType.Q8_0


class TestCreateGGUFMetadata:
    """Tests for create_gguf_metadata function."""

    def test_creates_metadata(self) -> None:
        """Test creating metadata."""
        meta = create_gguf_metadata(name="my-model", author="me")
        assert meta.name == "my-model"
        assert meta.author == "me"

    def test_default_values(self) -> None:
        """Test default values."""
        meta = create_gguf_metadata()
        assert meta.name == ""


class TestCreateGGUFExportResult:
    """Tests for create_gguf_export_result function."""

    def test_creates_result(self) -> None:
        """Test creating result."""
        result = create_gguf_export_result(
            "/model.gguf", 14000, 4000, GGUFQuantType.Q4_K_M
        )
        assert result.compression_ratio == pytest.approx(3.5)

    def test_empty_path_raises_error(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            create_gguf_export_result("", 14000, 4000, GGUFQuantType.Q4_K_M)

    def test_zero_original_raises_error(self) -> None:
        """Test that zero original raises ValueError."""
        with pytest.raises(ValueError, match="original_size_mb must be positive"):
            create_gguf_export_result("/model.gguf", 0, 4000, GGUFQuantType.Q4_K_M)

    def test_zero_exported_raises_error(self) -> None:
        """Test that zero exported raises ValueError."""
        with pytest.raises(ValueError, match="exported_size_mb must be positive"):
            create_gguf_export_result("/model.gguf", 14000, 0, GGUFQuantType.Q4_K_M)


class TestEstimateGGUFSize:
    """Tests for estimate_gguf_size function."""

    def test_estimates_size(self) -> None:
        """Test estimating size."""
        size = estimate_gguf_size(7_000_000_000, GGUFQuantType.Q4_K_M)
        assert size > 0
        assert size < 14000

    def test_zero_params_raises_error(self) -> None:
        """Test that zero params raises ValueError."""
        with pytest.raises(ValueError, match="model_params must be positive"):
            estimate_gguf_size(0, GGUFQuantType.Q4_K_M)

    def test_f16_larger_than_q4(self) -> None:
        """Test that F16 is larger than Q4."""
        size_f16 = estimate_gguf_size(7_000_000_000, GGUFQuantType.F16)
        size_q4 = estimate_gguf_size(7_000_000_000, GGUFQuantType.Q4_K_M)
        assert size_f16 > size_q4


class TestGetGGUFFilename:
    """Tests for get_gguf_filename function."""

    def test_generates_filename(self) -> None:
        """Test generating filename."""
        filename = get_gguf_filename("llama-7b", GGUFQuantType.Q4_K_M)
        assert filename == "llama-7b-q4_k_m.gguf"

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            get_gguf_filename("", GGUFQuantType.Q4_K_M)


class TestFormatGGUFExportResult:
    """Tests for format_gguf_export_result function."""

    def test_formats_result(self) -> None:
        """Test formatting result."""
        result = GGUFExportResult(
            output_path="/model.gguf",
            original_size_mb=14000,
            exported_size_mb=4000,
            compression_ratio=3.5,
            quant_type=GGUFQuantType.Q4_K_M,
        )
        formatted = format_gguf_export_result(result)
        assert "Output:" in formatted
        assert "Compression:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="result cannot be None"):
            format_gguf_export_result(None)  # type: ignore[arg-type]


class TestFormatGGUFModelInfo:
    """Tests for format_gguf_model_info function."""

    def test_formats_info(self) -> None:
        """Test formatting info."""
        info = GGUFModelInfo(
            path="/model.gguf", size_mb=4000, quant_type=GGUFQuantType.Q4_K_M
        )
        formatted = format_gguf_model_info(info)
        assert "Path:" in formatted
        assert "Size:" in formatted

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="info cannot be None"):
            format_gguf_model_info(None)  # type: ignore[arg-type]


class TestListGGUFQuantTypes:
    """Tests for list_gguf_quant_types function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        types = list_gguf_quant_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test that list contains expected types."""
        types = list_gguf_quant_types()
        assert "q4_k_m" in types
        assert "q8_0" in types

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        types = list_gguf_quant_types()
        assert types == sorted(types)


class TestValidateGGUFQuantType:
    """Tests for validate_gguf_quant_type function."""

    def test_valid_q4_k_m(self) -> None:
        """Test validation of q4_k_m type."""
        assert validate_gguf_quant_type("q4_k_m") is True

    def test_valid_q8_0(self) -> None:
        """Test validation of q8_0 type."""
        assert validate_gguf_quant_type("q8_0") is True

    def test_invalid_type(self) -> None:
        """Test validation of invalid type."""
        assert validate_gguf_quant_type("invalid") is False


class TestGetGGUFQuantType:
    """Tests for get_gguf_quant_type function."""

    def test_get_q4_k_m(self) -> None:
        """Test getting Q4_K_M type."""
        assert get_gguf_quant_type("q4_k_m") == GGUFQuantType.Q4_K_M

    def test_get_q8_0(self) -> None:
        """Test getting Q8_0 type."""
        assert get_gguf_quant_type("q8_0") == GGUFQuantType.Q8_0

    def test_invalid_raises_error(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="invalid GGUF quant type"):
            get_gguf_quant_type("invalid")


class TestListGGUFArchitectures:
    """Tests for list_gguf_architectures function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        archs = list_gguf_architectures()
        assert isinstance(archs, list)

    def test_contains_expected_archs(self) -> None:
        """Test that list contains expected architectures."""
        archs = list_gguf_architectures()
        assert "llama" in archs
        assert "mistral" in archs

    def test_is_sorted(self) -> None:
        """Test that list is sorted."""
        archs = list_gguf_architectures()
        assert archs == sorted(archs)


class TestValidateGGUFArchitecture:
    """Tests for validate_gguf_architecture function."""

    def test_valid_llama(self) -> None:
        """Test validation of llama architecture."""
        assert validate_gguf_architecture("llama") is True

    def test_valid_mistral(self) -> None:
        """Test validation of mistral architecture."""
        assert validate_gguf_architecture("mistral") is True

    def test_invalid_arch(self) -> None:
        """Test validation of invalid architecture."""
        assert validate_gguf_architecture("invalid") is False


class TestGetGGUFArchitecture:
    """Tests for get_gguf_architecture function."""

    def test_get_llama(self) -> None:
        """Test getting LLAMA architecture."""
        assert get_gguf_architecture("llama") == GGUFArchitecture.LLAMA

    def test_get_mistral(self) -> None:
        """Test getting MISTRAL architecture."""
        assert get_gguf_architecture("mistral") == GGUFArchitecture.MISTRAL

    def test_invalid_raises_error(self) -> None:
        """Test that invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="invalid GGUF architecture"):
            get_gguf_architecture("invalid")


class TestGetRecommendedGGUFQuant:
    """Tests for get_recommended_gguf_quant function."""

    def test_7b_default(self) -> None:
        """Test 7B model default."""
        assert get_recommended_gguf_quant("7b") == GGUFQuantType.Q4_K_M

    def test_7b_quality(self) -> None:
        """Test 7B model with quality priority."""
        result = get_recommended_gguf_quant("7b", quality_priority=True)
        assert result == GGUFQuantType.Q5_K_M

    def test_70b_default(self) -> None:
        """Test 70B model default."""
        assert get_recommended_gguf_quant("70b") == GGUFQuantType.Q4_K_S

    def test_invalid_size_raises_error(self) -> None:
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="unrecognized model size"):
            get_recommended_gguf_quant("invalid")


class TestGetGGUFConfigDict:
    """Tests for get_gguf_config_dict function."""

    def test_converts_config(self) -> None:
        """Test converting config."""
        config = create_gguf_config(quant_type="q4_k_m")
        d = get_gguf_config_dict(config)
        assert d["quant_type"] == "q4_k_m"
        assert d["concurrency"] == 4

    def test_with_architecture(self) -> None:
        """Test with architecture."""
        config = create_gguf_config(architecture="llama")
        d = get_gguf_config_dict(config)
        assert d["architecture"] == "llama"

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            get_gguf_config_dict(None)  # type: ignore[arg-type]


class TestGetGGUFMetadataDict:
    """Tests for get_gguf_metadata_dict function."""

    def test_converts_metadata(self) -> None:
        """Test converting metadata."""
        meta = create_gguf_metadata(name="my-model", author="me")
        d = get_gguf_metadata_dict(meta)
        assert d["name"] == "my-model"
        assert d["author"] == "me"

    def test_excludes_empty_values(self) -> None:
        """Test that empty values are excluded."""
        meta = create_gguf_metadata(name="my-model")
        d = get_gguf_metadata_dict(meta)
        assert "description" not in d

    def test_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="metadata cannot be None"):
            get_gguf_metadata_dict(None)  # type: ignore[arg-type]
