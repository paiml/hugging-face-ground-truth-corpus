"""Tests for document processing utilities."""

from __future__ import annotations

import pytest

from hf_gtc.multimodal.document import (
    VALID_BBOX_FORMATS,
    VALID_DOCUMENT_MODELS,
    VALID_DOCUMENT_TASKS,
    VALID_OCR_BACKENDS,
    VALID_RESIZE_MODES,
    BoundingBoxConfig,
    DocumentConfig,
    DocumentModelType,
    DocumentProcessingConfig,
    DocumentStats,
    DocumentTask,
    OCRBackend,
    OCRConfig,
    create_bounding_box_config,
    create_document_config,
    create_document_processing_config,
    create_ocr_config,
    estimate_document_tokens,
    get_default_document_config,
    get_default_ocr_config,
    get_document_model_type,
    get_document_task,
    get_ocr_backend,
    list_document_model_types,
    list_document_tasks,
    list_ocr_backends,
    normalize_bounding_box,
    validate_document_config,
    validate_ocr_config,
)


class TestDocumentModelType:
    """Tests for DocumentModelType enum."""

    def test_layoutlm_value(self) -> None:
        """Test LAYOUTLM value."""
        assert DocumentModelType.LAYOUTLM.value == "layoutlm"

    def test_layoutlmv2_value(self) -> None:
        """Test LAYOUTLMV2 value."""
        assert DocumentModelType.LAYOUTLMV2.value == "layoutlmv2"

    def test_layoutlmv3_value(self) -> None:
        """Test LAYOUTLMV3 value."""
        assert DocumentModelType.LAYOUTLMV3.value == "layoutlmv3"

    def test_donut_value(self) -> None:
        """Test DONUT value."""
        assert DocumentModelType.DONUT.value == "donut"

    def test_udop_value(self) -> None:
        """Test UDOP value."""
        assert DocumentModelType.UDOP.value == "udop"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_DOCUMENT_MODELS."""
        for model in DocumentModelType:
            assert model.value in VALID_DOCUMENT_MODELS


class TestDocumentTask:
    """Tests for DocumentTask enum."""

    def test_classification_value(self) -> None:
        """Test CLASSIFICATION value."""
        assert DocumentTask.CLASSIFICATION.value == "classification"

    def test_ner_value(self) -> None:
        """Test NER value."""
        assert DocumentTask.NER.value == "ner"

    def test_qa_value(self) -> None:
        """Test QA value."""
        assert DocumentTask.QA.value == "qa"

    def test_kie_value(self) -> None:
        """Test KIE value."""
        assert DocumentTask.KIE.value == "kie"

    def test_vqa_value(self) -> None:
        """Test VQA value."""
        assert DocumentTask.VQA.value == "vqa"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_DOCUMENT_TASKS."""
        for task in DocumentTask:
            assert task.value in VALID_DOCUMENT_TASKS


class TestOCRBackend:
    """Tests for OCRBackend enum."""

    def test_tesseract_value(self) -> None:
        """Test TESSERACT value."""
        assert OCRBackend.TESSERACT.value == "tesseract"

    def test_easyocr_value(self) -> None:
        """Test EASYOCR value."""
        assert OCRBackend.EASYOCR.value == "easyocr"

    def test_paddleocr_value(self) -> None:
        """Test PADDLEOCR value."""
        assert OCRBackend.PADDLEOCR.value == "paddleocr"

    def test_doctr_value(self) -> None:
        """Test DOCTR value."""
        assert OCRBackend.DOCTR.value == "doctr"

    def test_all_in_valid_set(self) -> None:
        """Test all enum values are in VALID_OCR_BACKENDS."""
        for backend in OCRBackend:
            assert backend.value in VALID_OCR_BACKENDS


class TestValidSets:
    """Tests for valid frozensets."""

    def test_valid_bbox_formats(self) -> None:
        """Test VALID_BBOX_FORMATS contents."""
        assert "xyxy" in VALID_BBOX_FORMATS
        assert "xywh" in VALID_BBOX_FORMATS
        assert "cxcywh" in VALID_BBOX_FORMATS
        assert len(VALID_BBOX_FORMATS) == 3

    def test_valid_resize_modes(self) -> None:
        """Test VALID_RESIZE_MODES contents."""
        assert "resize" in VALID_RESIZE_MODES
        assert "pad" in VALID_RESIZE_MODES
        assert "crop" in VALID_RESIZE_MODES
        assert len(VALID_RESIZE_MODES) == 3


class TestDocumentConfig:
    """Tests for DocumentConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating document config."""
        config = DocumentConfig(
            model_type=DocumentModelType.LAYOUTLMV3,
            max_seq_length=512,
            image_size=(224, 224),
            use_ocr=True,
        )
        assert config.model_type == DocumentModelType.LAYOUTLMV3
        assert config.max_seq_length == 512
        assert config.image_size == (224, 224)
        assert config.use_ocr is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 512, (224, 224), True)
        with pytest.raises(AttributeError):
            config.max_seq_length = 256  # type: ignore[misc]


class TestOCRConfig:
    """Tests for OCRConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating OCR config."""
        config = OCRConfig(
            backend=OCRBackend.TESSERACT,
            language="en",
            confidence_threshold=0.5,
            dpi=300,
        )
        assert config.backend == OCRBackend.TESSERACT
        assert config.language == "en"
        assert config.confidence_threshold == pytest.approx(0.5)
        assert config.dpi == 300

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 0.5, 300)
        with pytest.raises(AttributeError):
            config.dpi = 600  # type: ignore[misc]


class TestBoundingBoxConfig:
    """Tests for BoundingBoxConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating bounding box config."""
        config = BoundingBoxConfig(
            format="xyxy",
            normalize=True,
            include_text=True,
        )
        assert config.format == "xyxy"
        assert config.normalize is True
        assert config.include_text is True

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = BoundingBoxConfig("xyxy", True, True)
        with pytest.raises(AttributeError):
            config.normalize = False  # type: ignore[misc]


class TestDocumentProcessingConfig:
    """Tests for DocumentProcessingConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating document processing config."""
        config = DocumentProcessingConfig(
            resize_mode="resize",
            normalize_boxes=True,
            max_pages=10,
        )
        assert config.resize_mode == "resize"
        assert config.normalize_boxes is True
        assert config.max_pages == 10

    def test_frozen(self) -> None:
        """Test config is immutable."""
        config = DocumentProcessingConfig("resize", True, 10)
        with pytest.raises(AttributeError):
            config.max_pages = 5  # type: ignore[misc]


class TestDocumentStats:
    """Tests for DocumentStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating document stats."""
        stats = DocumentStats(
            num_pages=5,
            num_words=1000,
            num_boxes=250,
            processing_time=2.5,
        )
        assert stats.num_pages == 5
        assert stats.num_words == 1000
        assert stats.num_boxes == 250
        assert stats.processing_time == pytest.approx(2.5)

    def test_frozen(self) -> None:
        """Test stats is immutable."""
        stats = DocumentStats(5, 1000, 250, 2.5)
        with pytest.raises(AttributeError):
            stats.num_pages = 10  # type: ignore[misc]


class TestValidateDocumentConfig:
    """Tests for validate_document_config function."""

    def test_valid_config(self) -> None:
        """Test valid config."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 512, (224, 224), True)
        validate_document_config(config)  # Should not raise

    def test_none_config(self) -> None:
        """Test None config."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_document_config(None)  # type: ignore[arg-type]

    def test_zero_max_seq_length(self) -> None:
        """Test zero max_seq_length."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 0, (224, 224), True)
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_document_config(config)

    def test_negative_max_seq_length(self) -> None:
        """Test negative max_seq_length."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, -1, (224, 224), True)
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            validate_document_config(config)

    def test_zero_image_width(self) -> None:
        """Test zero image width."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 512, (0, 224), True)
        with pytest.raises(ValueError, match="image_size width must be positive"):
            validate_document_config(config)

    def test_negative_image_width(self) -> None:
        """Test negative image width."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 512, (-1, 224), True)
        with pytest.raises(ValueError, match="image_size width must be positive"):
            validate_document_config(config)

    def test_zero_image_height(self) -> None:
        """Test zero image height."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 512, (224, 0), True)
        with pytest.raises(ValueError, match="image_size height must be positive"):
            validate_document_config(config)

    def test_negative_image_height(self) -> None:
        """Test negative image height."""
        config = DocumentConfig(DocumentModelType.LAYOUTLMV3, 512, (224, -1), True)
        with pytest.raises(ValueError, match="image_size height must be positive"):
            validate_document_config(config)


class TestValidateOCRConfig:
    """Tests for validate_ocr_config function."""

    def test_valid_config(self) -> None:
        """Test valid config."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 0.5, 300)
        validate_ocr_config(config)  # Should not raise

    def test_none_config(self) -> None:
        """Test None config."""
        with pytest.raises(ValueError, match="config cannot be None"):
            validate_ocr_config(None)  # type: ignore[arg-type]

    def test_empty_language(self) -> None:
        """Test empty language."""
        config = OCRConfig(OCRBackend.TESSERACT, "", 0.5, 300)
        with pytest.raises(ValueError, match="language cannot be empty"):
            validate_ocr_config(config)

    def test_negative_confidence_threshold(self) -> None:
        """Test negative confidence_threshold."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", -0.1, 300)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            validate_ocr_config(config)

    def test_confidence_threshold_above_one(self) -> None:
        """Test confidence_threshold above 1."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 1.1, 300)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            validate_ocr_config(config)

    def test_boundary_confidence_zero(self) -> None:
        """Test confidence_threshold at 0."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 0.0, 300)
        validate_ocr_config(config)  # Should not raise

    def test_boundary_confidence_one(self) -> None:
        """Test confidence_threshold at 1."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 1.0, 300)
        validate_ocr_config(config)  # Should not raise

    def test_zero_dpi(self) -> None:
        """Test zero dpi."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 0.5, 0)
        with pytest.raises(ValueError, match="dpi must be positive"):
            validate_ocr_config(config)

    def test_negative_dpi(self) -> None:
        """Test negative dpi."""
        config = OCRConfig(OCRBackend.TESSERACT, "en", 0.5, -1)
        with pytest.raises(ValueError, match="dpi must be positive"):
            validate_ocr_config(config)


class TestCreateDocumentConfig:
    """Tests for create_document_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_document_config()
        assert config.model_type == DocumentModelType.LAYOUTLMV3
        assert config.max_seq_length == 512
        assert config.image_size == (224, 224)
        assert config.use_ocr is True

    @pytest.mark.parametrize(
        "model_type",
        ["layoutlm", "layoutlmv2", "layoutlmv3", "donut", "udop"],
    )
    def test_all_model_types(self, model_type: str) -> None:
        """Test all valid model types."""
        config = create_document_config(model_type=model_type)
        assert config.model_type.value == model_type

    def test_custom_max_seq_length(self) -> None:
        """Test custom max_seq_length."""
        config = create_document_config(max_seq_length=256)
        assert config.max_seq_length == 256

    def test_custom_image_size(self) -> None:
        """Test custom image_size."""
        config = create_document_config(image_size=(336, 336))
        assert config.image_size == (336, 336)

    def test_use_ocr_false(self) -> None:
        """Test use_ocr=False."""
        config = create_document_config(use_ocr=False)
        assert config.use_ocr is False

    def test_invalid_model_type(self) -> None:
        """Test invalid model_type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            create_document_config(model_type="invalid")

    def test_invalid_max_seq_length(self) -> None:
        """Test invalid max_seq_length."""
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            create_document_config(max_seq_length=0)

    def test_invalid_image_width(self) -> None:
        """Test invalid image width."""
        with pytest.raises(ValueError, match="image_size width must be positive"):
            create_document_config(image_size=(0, 224))

    def test_invalid_image_height(self) -> None:
        """Test invalid image height."""
        with pytest.raises(ValueError, match="image_size height must be positive"):
            create_document_config(image_size=(224, 0))


class TestCreateOCRConfig:
    """Tests for create_ocr_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_ocr_config()
        assert config.backend == OCRBackend.TESSERACT
        assert config.language == "en"
        assert config.confidence_threshold == pytest.approx(0.5)
        assert config.dpi == 300

    @pytest.mark.parametrize(
        "backend",
        ["tesseract", "easyocr", "paddleocr", "doctr"],
    )
    def test_all_backends(self, backend: str) -> None:
        """Test all valid backends."""
        config = create_ocr_config(backend=backend)
        assert config.backend.value == backend

    def test_custom_language(self) -> None:
        """Test custom language."""
        config = create_ocr_config(language="zh")
        assert config.language == "zh"

    def test_custom_confidence_threshold(self) -> None:
        """Test custom confidence_threshold."""
        config = create_ocr_config(confidence_threshold=0.8)
        assert config.confidence_threshold == pytest.approx(0.8)

    def test_custom_dpi(self) -> None:
        """Test custom dpi."""
        config = create_ocr_config(dpi=600)
        assert config.dpi == 600

    def test_invalid_backend(self) -> None:
        """Test invalid backend."""
        with pytest.raises(ValueError, match="backend must be one of"):
            create_ocr_config(backend="invalid")

    def test_empty_language(self) -> None:
        """Test empty language."""
        with pytest.raises(ValueError, match="language cannot be empty"):
            create_ocr_config(language="")

    def test_invalid_confidence_threshold(self) -> None:
        """Test invalid confidence_threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            create_ocr_config(confidence_threshold=1.5)

    def test_invalid_dpi(self) -> None:
        """Test invalid dpi."""
        with pytest.raises(ValueError, match="dpi must be positive"):
            create_ocr_config(dpi=0)


class TestCreateBoundingBoxConfig:
    """Tests for create_bounding_box_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_bounding_box_config()
        assert config.format == "xyxy"
        assert config.normalize is True
        assert config.include_text is True

    @pytest.mark.parametrize("fmt", ["xyxy", "xywh", "cxcywh"])
    def test_all_formats(self, fmt: str) -> None:
        """Test all valid formats."""
        config = create_bounding_box_config(format=fmt)
        assert config.format == fmt

    def test_normalize_false(self) -> None:
        """Test normalize=False."""
        config = create_bounding_box_config(normalize=False)
        assert config.normalize is False

    def test_include_text_false(self) -> None:
        """Test include_text=False."""
        config = create_bounding_box_config(include_text=False)
        assert config.include_text is False

    def test_invalid_format(self) -> None:
        """Test invalid format."""
        with pytest.raises(ValueError, match="format must be one of"):
            create_bounding_box_config(format="invalid")


class TestCreateDocumentProcessingConfig:
    """Tests for create_document_processing_config function."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = create_document_processing_config()
        assert config.resize_mode == "resize"
        assert config.normalize_boxes is True
        assert config.max_pages == 10

    @pytest.mark.parametrize("mode", ["resize", "pad", "crop"])
    def test_all_resize_modes(self, mode: str) -> None:
        """Test all valid resize modes."""
        config = create_document_processing_config(resize_mode=mode)
        assert config.resize_mode == mode

    def test_normalize_boxes_false(self) -> None:
        """Test normalize_boxes=False."""
        config = create_document_processing_config(normalize_boxes=False)
        assert config.normalize_boxes is False

    def test_custom_max_pages(self) -> None:
        """Test custom max_pages."""
        config = create_document_processing_config(max_pages=5)
        assert config.max_pages == 5

    def test_invalid_resize_mode(self) -> None:
        """Test invalid resize_mode."""
        with pytest.raises(ValueError, match="resize_mode must be one of"):
            create_document_processing_config(resize_mode="invalid")

    def test_zero_max_pages(self) -> None:
        """Test zero max_pages."""
        with pytest.raises(ValueError, match="max_pages must be positive"):
            create_document_processing_config(max_pages=0)

    def test_negative_max_pages(self) -> None:
        """Test negative max_pages."""
        with pytest.raises(ValueError, match="max_pages must be positive"):
            create_document_processing_config(max_pages=-1)


class TestListDocumentModelTypes:
    """Tests for list_document_model_types function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        types = list_document_model_types()
        assert isinstance(types, list)

    def test_contains_expected_types(self) -> None:
        """Test contains expected types."""
        types = list_document_model_types()
        assert "layoutlmv3" in types
        assert "donut" in types

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        types = list_document_model_types()
        assert types == sorted(types)


class TestListDocumentTasks:
    """Tests for list_document_tasks function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        tasks = list_document_tasks()
        assert isinstance(tasks, list)

    def test_contains_expected_tasks(self) -> None:
        """Test contains expected tasks."""
        tasks = list_document_tasks()
        assert "classification" in tasks
        assert "ner" in tasks

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        tasks = list_document_tasks()
        assert tasks == sorted(tasks)


class TestListOCRBackends:
    """Tests for list_ocr_backends function."""

    def test_returns_list(self) -> None:
        """Test returns a list."""
        backends = list_ocr_backends()
        assert isinstance(backends, list)

    def test_contains_expected_backends(self) -> None:
        """Test contains expected backends."""
        backends = list_ocr_backends()
        assert "tesseract" in backends
        assert "paddleocr" in backends

    def test_is_sorted(self) -> None:
        """Test list is sorted."""
        backends = list_ocr_backends()
        assert backends == sorted(backends)


class TestGetDocumentModelType:
    """Tests for get_document_model_type function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("layoutlm", DocumentModelType.LAYOUTLM),
            ("layoutlmv2", DocumentModelType.LAYOUTLMV2),
            ("layoutlmv3", DocumentModelType.LAYOUTLMV3),
            ("donut", DocumentModelType.DONUT),
            ("udop", DocumentModelType.UDOP),
        ],
    )
    def test_valid_model_types(self, name: str, expected: DocumentModelType) -> None:
        """Test getting valid model types."""
        assert get_document_model_type(name) == expected

    def test_invalid_model_type(self) -> None:
        """Test invalid model type."""
        with pytest.raises(ValueError, match="model type must be one of"):
            get_document_model_type("invalid")


class TestGetDocumentTask:
    """Tests for get_document_task function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("classification", DocumentTask.CLASSIFICATION),
            ("ner", DocumentTask.NER),
            ("qa", DocumentTask.QA),
            ("kie", DocumentTask.KIE),
            ("vqa", DocumentTask.VQA),
        ],
    )
    def test_valid_tasks(self, name: str, expected: DocumentTask) -> None:
        """Test getting valid tasks."""
        assert get_document_task(name) == expected

    def test_invalid_task(self) -> None:
        """Test invalid task."""
        with pytest.raises(ValueError, match="task must be one of"):
            get_document_task("invalid")


class TestGetOCRBackend:
    """Tests for get_ocr_backend function."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("tesseract", OCRBackend.TESSERACT),
            ("easyocr", OCRBackend.EASYOCR),
            ("paddleocr", OCRBackend.PADDLEOCR),
            ("doctr", OCRBackend.DOCTR),
        ],
    )
    def test_valid_backends(self, name: str, expected: OCRBackend) -> None:
        """Test getting valid backends."""
        assert get_ocr_backend(name) == expected

    def test_invalid_backend(self) -> None:
        """Test invalid backend."""
        with pytest.raises(ValueError, match="backend must be one of"):
            get_ocr_backend("invalid")


class TestNormalizeBoundingBox:
    """Tests for normalize_bounding_box function."""

    def test_1000x1000_image(self) -> None:
        """Test normalization with 1000x1000 image."""
        result = normalize_bounding_box((100, 200, 300, 400), 1000, 1000)
        assert result == (100, 200, 300, 400)

    def test_500x500_image(self) -> None:
        """Test normalization with 500x500 image."""
        result = normalize_bounding_box((50, 100, 150, 200), 500, 500)
        assert result == (100, 200, 300, 400)

    def test_full_image_box(self) -> None:
        """Test full image bounding box."""
        result = normalize_bounding_box((0, 0, 500, 500), 500, 500)
        assert result == (0, 0, 1000, 1000)

    def test_custom_target_range(self) -> None:
        """Test custom target_range."""
        result = normalize_bounding_box((0, 0, 100, 100), 100, 100, target_range=100)
        assert result == (0, 0, 100, 100)

    def test_clamping_upper(self) -> None:
        """Test clamping to upper bound."""
        # Box exceeds image bounds
        result = normalize_bounding_box((900, 900, 1100, 1100), 1000, 1000)
        assert result == (900, 900, 1000, 1000)

    def test_zero_image_width(self) -> None:
        """Test zero image_width."""
        with pytest.raises(ValueError, match="image_width must be positive"):
            normalize_bounding_box((0, 0, 100, 100), 0, 100)

    def test_negative_image_width(self) -> None:
        """Test negative image_width."""
        with pytest.raises(ValueError, match="image_width must be positive"):
            normalize_bounding_box((0, 0, 100, 100), -1, 100)

    def test_zero_image_height(self) -> None:
        """Test zero image_height."""
        with pytest.raises(ValueError, match="image_height must be positive"):
            normalize_bounding_box((0, 0, 100, 100), 100, 0)

    def test_negative_image_height(self) -> None:
        """Test negative image_height."""
        with pytest.raises(ValueError, match="image_height must be positive"):
            normalize_bounding_box((0, 0, 100, 100), 100, -1)

    def test_zero_target_range(self) -> None:
        """Test zero target_range."""
        with pytest.raises(ValueError, match="target_range must be positive"):
            normalize_bounding_box((0, 0, 100, 100), 100, 100, target_range=0)

    def test_negative_target_range(self) -> None:
        """Test negative target_range."""
        with pytest.raises(ValueError, match="target_range must be positive"):
            normalize_bounding_box((0, 0, 100, 100), 100, 100, target_range=-1)

    def test_negative_x1(self) -> None:
        """Test negative x1 coordinate."""
        with pytest.raises(ValueError, match="box coordinates cannot be negative"):
            normalize_bounding_box((-1, 0, 100, 100), 100, 100)

    def test_negative_y1(self) -> None:
        """Test negative y1 coordinate."""
        with pytest.raises(ValueError, match="box coordinates cannot be negative"):
            normalize_bounding_box((0, -1, 100, 100), 100, 100)

    def test_negative_x2(self) -> None:
        """Test negative x2 coordinate."""
        with pytest.raises(ValueError, match="box coordinates cannot be negative"):
            normalize_bounding_box((0, 0, -1, 100), 100, 100)

    def test_negative_y2(self) -> None:
        """Test negative y2 coordinate."""
        with pytest.raises(ValueError, match="box coordinates cannot be negative"):
            normalize_bounding_box((0, 0, 100, -1), 100, 100)

    def test_rectangular_image(self) -> None:
        """Test normalization with rectangular image."""
        result = normalize_bounding_box((0, 0, 400, 300), 800, 600)
        assert result == (0, 0, 500, 500)


class TestEstimateDocumentTokens:
    """Tests for estimate_document_tokens function."""

    def test_basic_estimation(self) -> None:
        """Test basic token estimation."""
        tokens = estimate_document_tokens(100, 50)
        # 100 text tokens + 196 image tokens (224/16 * 224/16)
        assert tokens == 296

    def test_zero_words_and_boxes(self) -> None:
        """Test with zero words and boxes."""
        tokens = estimate_document_tokens(0, 0)
        # Only image tokens: 196
        assert tokens == 196

    def test_without_image_tokens(self) -> None:
        """Test without image tokens."""
        tokens = estimate_document_tokens(100, 50, include_image_tokens=False)
        assert tokens == 100

    def test_layoutlm_no_image_tokens(self) -> None:
        """Test LAYOUTLM model (no vision encoder)."""
        tokens = estimate_document_tokens(
            100, 50, model_type=DocumentModelType.LAYOUTLM
        )
        # LAYOUTLM doesn't have vision encoder
        assert tokens == 100

    @pytest.mark.parametrize(
        "model_type",
        [
            DocumentModelType.LAYOUTLMV2,
            DocumentModelType.LAYOUTLMV3,
            DocumentModelType.DONUT,
            DocumentModelType.UDOP,
        ],
    )
    def test_vision_models_include_image_tokens(
        self, model_type: DocumentModelType
    ) -> None:
        """Test vision models include image tokens."""
        tokens = estimate_document_tokens(100, 50, model_type=model_type)
        assert tokens > 100  # More than text tokens alone

    def test_custom_image_size(self) -> None:
        """Test custom image size."""
        tokens = estimate_document_tokens(100, 50, image_size=(336, 336), patch_size=16)
        # 100 text + (336/16)*(336/16) = 100 + 441
        assert tokens == 541

    def test_custom_patch_size(self) -> None:
        """Test custom patch size."""
        tokens = estimate_document_tokens(100, 50, image_size=(224, 224), patch_size=14)
        # 100 text + (224/14)*(224/14) = 100 + 256
        assert tokens == 356

    def test_negative_num_words(self) -> None:
        """Test negative num_words."""
        with pytest.raises(ValueError, match="num_words cannot be negative"):
            estimate_document_tokens(-1, 50)

    def test_negative_num_boxes(self) -> None:
        """Test negative num_boxes."""
        with pytest.raises(ValueError, match="num_boxes cannot be negative"):
            estimate_document_tokens(100, -1)

    def test_zero_image_width(self) -> None:
        """Test zero image width."""
        with pytest.raises(ValueError, match="image_size width must be positive"):
            estimate_document_tokens(100, 50, image_size=(0, 224))

    def test_negative_image_width(self) -> None:
        """Test negative image width."""
        with pytest.raises(ValueError, match="image_size width must be positive"):
            estimate_document_tokens(100, 50, image_size=(-1, 224))

    def test_zero_image_height(self) -> None:
        """Test zero image height."""
        with pytest.raises(ValueError, match="image_size height must be positive"):
            estimate_document_tokens(100, 50, image_size=(224, 0))

    def test_negative_image_height(self) -> None:
        """Test negative image height."""
        with pytest.raises(ValueError, match="image_size height must be positive"):
            estimate_document_tokens(100, 50, image_size=(224, -1))

    def test_zero_patch_size(self) -> None:
        """Test zero patch_size."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            estimate_document_tokens(100, 50, patch_size=0)

    def test_negative_patch_size(self) -> None:
        """Test negative patch_size."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            estimate_document_tokens(100, 50, patch_size=-1)


class TestGetDefaultDocumentConfig:
    """Tests for get_default_document_config function."""

    @pytest.mark.parametrize(
        "model_type",
        list(DocumentModelType),
    )
    def test_all_model_types_have_defaults(self, model_type: DocumentModelType) -> None:
        """Test all model types have default configs."""
        config = get_default_document_config(model_type)
        assert config.model_type == model_type

    def test_layoutlmv3_defaults(self) -> None:
        """Test LAYOUTLMV3 defaults."""
        config = get_default_document_config(DocumentModelType.LAYOUTLMV3)
        assert config.max_seq_length == 512
        assert config.image_size == (224, 224)
        assert config.use_ocr is True

    def test_donut_defaults(self) -> None:
        """Test DONUT defaults."""
        config = get_default_document_config(DocumentModelType.DONUT)
        assert config.max_seq_length == 2048
        assert config.image_size == (1280, 960)
        assert config.use_ocr is False

    def test_layoutlm_defaults(self) -> None:
        """Test LAYOUTLM defaults."""
        config = get_default_document_config(DocumentModelType.LAYOUTLM)
        assert config.max_seq_length == 512
        assert config.use_ocr is True

    def test_layoutlmv2_defaults(self) -> None:
        """Test LAYOUTLMV2 defaults."""
        config = get_default_document_config(DocumentModelType.LAYOUTLMV2)
        assert config.max_seq_length == 512
        assert config.use_ocr is True

    def test_udop_defaults(self) -> None:
        """Test UDOP defaults."""
        config = get_default_document_config(DocumentModelType.UDOP)
        assert config.max_seq_length == 512
        assert config.use_ocr is True


class TestGetDefaultOCRConfig:
    """Tests for get_default_ocr_config function."""

    @pytest.mark.parametrize(
        "backend",
        list(OCRBackend),
    )
    def test_all_backends_have_defaults(self, backend: OCRBackend) -> None:
        """Test all backends have default configs."""
        config = get_default_ocr_config(backend)
        assert config.backend == backend

    def test_tesseract_defaults(self) -> None:
        """Test TESSERACT defaults."""
        config = get_default_ocr_config(OCRBackend.TESSERACT)
        assert config.language == "en"
        assert config.confidence_threshold == pytest.approx(0.5)
        assert config.dpi == 300

    def test_easyocr_defaults(self) -> None:
        """Test EASYOCR defaults."""
        config = get_default_ocr_config(OCRBackend.EASYOCR)
        assert config.language == "en"
        assert config.dpi == 300

    def test_paddleocr_defaults(self) -> None:
        """Test PADDLEOCR defaults."""
        config = get_default_ocr_config(OCRBackend.PADDLEOCR)
        assert config.language == "en"
        assert config.dpi == 300

    def test_doctr_defaults(self) -> None:
        """Test DOCTR defaults."""
        config = get_default_ocr_config(OCRBackend.DOCTR)
        assert config.language == "en"
        assert config.dpi == 300
