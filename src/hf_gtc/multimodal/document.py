"""Document processing utilities for HuggingFace models.

This module provides functions for working with document understanding models,
OCR processing, and bounding box normalization for document AI tasks.

Examples:
    >>> from hf_gtc.multimodal.document import create_document_config
    >>> config = create_document_config(model_type="layoutlmv3")
    >>> config.model_type
    <DocumentModelType.LAYOUTLMV3: 'layoutlmv3'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class DocumentModelType(Enum):
    """Supported document model types.

    Attributes:
        LAYOUTLM: LayoutLM model for document understanding.
        LAYOUTLMV2: LayoutLMv2 with visual features.
        LAYOUTLMV3: LayoutLMv3 unified model.
        DONUT: Document Understanding Transformer (OCR-free).
        UDOP: Unified Document Processing model.

    Examples:
        >>> DocumentModelType.LAYOUTLM.value
        'layoutlm'
        >>> DocumentModelType.DONUT.value
        'donut'
    """

    LAYOUTLM = "layoutlm"
    LAYOUTLMV2 = "layoutlmv2"
    LAYOUTLMV3 = "layoutlmv3"
    DONUT = "donut"
    UDOP = "udop"


VALID_DOCUMENT_MODELS = frozenset(m.value for m in DocumentModelType)


class DocumentTask(Enum):
    """Supported document processing tasks.

    Attributes:
        CLASSIFICATION: Document classification.
        NER: Named entity recognition on documents.
        QA: Document question answering.
        KIE: Key information extraction.
        VQA: Visual question answering on documents.

    Examples:
        >>> DocumentTask.CLASSIFICATION.value
        'classification'
        >>> DocumentTask.KIE.value
        'kie'
    """

    CLASSIFICATION = "classification"
    NER = "ner"
    QA = "qa"
    KIE = "kie"
    VQA = "vqa"


VALID_DOCUMENT_TASKS = frozenset(t.value for t in DocumentTask)


class OCRBackend(Enum):
    """Supported OCR backend engines.

    Attributes:
        TESSERACT: Tesseract OCR engine.
        EASYOCR: EasyOCR engine.
        PADDLEOCR: PaddleOCR engine.
        DOCTR: docTR (Document Text Recognition) engine.

    Examples:
        >>> OCRBackend.TESSERACT.value
        'tesseract'
        >>> OCRBackend.PADDLEOCR.value
        'paddleocr'
    """

    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    DOCTR = "doctr"


VALID_OCR_BACKENDS = frozenset(b.value for b in OCRBackend)


# Bounding box format types
BoundingBoxFormat = Literal["xyxy", "xywh", "cxcywh"]
VALID_BBOX_FORMATS: frozenset[str] = frozenset({"xyxy", "xywh", "cxcywh"})

# Resize mode types
ResizeMode = Literal["resize", "pad", "crop"]
VALID_RESIZE_MODES: frozenset[str] = frozenset({"resize", "pad", "crop"})


@dataclass(frozen=True, slots=True)
class DocumentConfig:
    """Configuration for document processing models.

    Attributes:
        model_type: Type of document model.
        max_seq_length: Maximum sequence length for tokenization.
        image_size: Image size for vision encoder (width, height).
        use_ocr: Whether to use OCR for text extraction.

    Examples:
        >>> config = DocumentConfig(
        ...     model_type=DocumentModelType.LAYOUTLMV3,
        ...     max_seq_length=512,
        ...     image_size=(224, 224),
        ...     use_ocr=True,
        ... )
        >>> config.max_seq_length
        512
    """

    model_type: DocumentModelType
    max_seq_length: int
    image_size: tuple[int, int]
    use_ocr: bool


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Configuration for OCR processing.

    Attributes:
        backend: OCR backend engine to use.
        language: Language code for OCR (e.g., "en", "zh").
        confidence_threshold: Minimum confidence for OCR results.
        dpi: DPI for image processing.

    Examples:
        >>> config = OCRConfig(
        ...     backend=OCRBackend.TESSERACT,
        ...     language="en",
        ...     confidence_threshold=0.5,
        ...     dpi=300,
        ... )
        >>> config.backend
        <OCRBackend.TESSERACT: 'tesseract'>
    """

    backend: OCRBackend
    language: str
    confidence_threshold: float
    dpi: int


@dataclass(frozen=True, slots=True)
class BoundingBoxConfig:
    """Configuration for bounding box processing.

    Attributes:
        format: Bounding box format (xyxy, xywh, cxcywh).
        normalize: Whether to normalize boxes to 0-1000 range.
        include_text: Whether to include text with boxes.

    Examples:
        >>> config = BoundingBoxConfig(
        ...     format="xyxy",
        ...     normalize=True,
        ...     include_text=True,
        ... )
        >>> config.normalize
        True
    """

    format: str
    normalize: bool
    include_text: bool


@dataclass(frozen=True, slots=True)
class DocumentProcessingConfig:
    """Configuration for document processing pipeline.

    Attributes:
        resize_mode: How to resize document images.
        normalize_boxes: Whether to normalize bounding boxes.
        max_pages: Maximum number of pages to process.

    Examples:
        >>> config = DocumentProcessingConfig(
        ...     resize_mode="resize",
        ...     normalize_boxes=True,
        ...     max_pages=10,
        ... )
        >>> config.max_pages
        10
    """

    resize_mode: str
    normalize_boxes: bool
    max_pages: int


@dataclass(frozen=True, slots=True)
class DocumentStats:
    """Statistics from document processing.

    Attributes:
        num_pages: Number of pages in the document.
        num_words: Total number of words extracted.
        num_boxes: Total number of bounding boxes.
        processing_time: Processing time in seconds.

    Examples:
        >>> stats = DocumentStats(
        ...     num_pages=5,
        ...     num_words=1000,
        ...     num_boxes=250,
        ...     processing_time=2.5,
        ... )
        >>> stats.num_pages
        5
    """

    num_pages: int
    num_words: int
    num_boxes: int
    processing_time: float


def validate_document_config(config: DocumentConfig) -> None:
    """Validate document configuration.

    Args:
        config: DocumentConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If max_seq_length is not positive.
        ValueError: If image_size dimensions are not positive.

    Examples:
        >>> config = DocumentConfig(
        ...     model_type=DocumentModelType.LAYOUTLMV3,
        ...     max_seq_length=512,
        ...     image_size=(224, 224),
        ...     use_ocr=True,
        ... )
        >>> validate_document_config(config)  # No error

        >>> validate_document_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = DocumentConfig(
        ...     model_type=DocumentModelType.LAYOUTLMV3,
        ...     max_seq_length=0,
        ...     image_size=(224, 224),
        ...     use_ocr=True,
        ... )
        >>> validate_document_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_seq_length must be positive
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if config.max_seq_length <= 0:
        msg = f"max_seq_length must be positive, got {config.max_seq_length}"
        raise ValueError(msg)

    width, height = config.image_size
    if width <= 0:
        msg = f"image_size width must be positive, got {width}"
        raise ValueError(msg)

    if height <= 0:
        msg = f"image_size height must be positive, got {height}"
        raise ValueError(msg)


def validate_ocr_config(config: OCRConfig) -> None:
    """Validate OCR configuration.

    Args:
        config: OCRConfig to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If language is empty.
        ValueError: If confidence_threshold is not in [0, 1].
        ValueError: If dpi is not positive.

    Examples:
        >>> config = OCRConfig(
        ...     backend=OCRBackend.TESSERACT,
        ...     language="en",
        ...     confidence_threshold=0.5,
        ...     dpi=300,
        ... )
        >>> validate_ocr_config(config)  # No error

        >>> validate_ocr_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = OCRConfig(
        ...     backend=OCRBackend.TESSERACT,
        ...     language="",
        ...     confidence_threshold=0.5,
        ...     dpi=300,
        ... )
        >>> validate_ocr_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: language cannot be empty
    """
    if config is None:
        msg = "config cannot be None"
        raise ValueError(msg)

    if not config.language:
        msg = "language cannot be empty"
        raise ValueError(msg)

    if not 0.0 <= config.confidence_threshold <= 1.0:
        msg = (
            f"confidence_threshold must be in [0, 1], "
            f"got {config.confidence_threshold}"
        )
        raise ValueError(msg)

    if config.dpi <= 0:
        msg = f"dpi must be positive, got {config.dpi}"
        raise ValueError(msg)


def create_document_config(
    model_type: str = "layoutlmv3",
    max_seq_length: int = 512,
    image_size: tuple[int, int] = (224, 224),
    use_ocr: bool = True,
) -> DocumentConfig:
    """Create a document processing configuration.

    Args:
        model_type: Document model type. Defaults to "layoutlmv3".
        max_seq_length: Maximum sequence length. Defaults to 512.
        image_size: Image size (width, height). Defaults to (224, 224).
        use_ocr: Whether to use OCR. Defaults to True.

    Returns:
        DocumentConfig with the specified settings.

    Raises:
        ValueError: If model_type is invalid.
        ValueError: If max_seq_length is not positive.
        ValueError: If image_size dimensions are not positive.

    Examples:
        >>> config = create_document_config(model_type="layoutlmv3")
        >>> config.model_type
        <DocumentModelType.LAYOUTLMV3: 'layoutlmv3'>

        >>> config = create_document_config(max_seq_length=256)
        >>> config.max_seq_length
        256

        >>> create_document_config(model_type="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_type must be one of ...
    """
    if model_type not in VALID_DOCUMENT_MODELS:
        msg = f"model_type must be one of {VALID_DOCUMENT_MODELS}, got '{model_type}'"
        raise ValueError(msg)

    config = DocumentConfig(
        model_type=DocumentModelType(model_type),
        max_seq_length=max_seq_length,
        image_size=image_size,
        use_ocr=use_ocr,
    )
    validate_document_config(config)
    return config


def create_ocr_config(
    backend: str = "tesseract",
    language: str = "en",
    confidence_threshold: float = 0.5,
    dpi: int = 300,
) -> OCRConfig:
    """Create an OCR configuration.

    Args:
        backend: OCR backend engine. Defaults to "tesseract".
        language: Language code. Defaults to "en".
        confidence_threshold: Minimum confidence. Defaults to 0.5.
        dpi: DPI for processing. Defaults to 300.

    Returns:
        OCRConfig with the specified settings.

    Raises:
        ValueError: If backend is invalid.
        ValueError: If language is empty.
        ValueError: If confidence_threshold is not in [0, 1].
        ValueError: If dpi is not positive.

    Examples:
        >>> config = create_ocr_config(backend="tesseract")
        >>> config.backend
        <OCRBackend.TESSERACT: 'tesseract'>

        >>> config = create_ocr_config(language="zh")
        >>> config.language
        'zh'

        >>> create_ocr_config(backend="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: backend must be one of ...
    """
    if backend not in VALID_OCR_BACKENDS:
        msg = f"backend must be one of {VALID_OCR_BACKENDS}, got '{backend}'"
        raise ValueError(msg)

    config = OCRConfig(
        backend=OCRBackend(backend),
        language=language,
        confidence_threshold=confidence_threshold,
        dpi=dpi,
    )
    validate_ocr_config(config)
    return config


def create_bounding_box_config(
    format: str = "xyxy",
    normalize: bool = True,
    include_text: bool = True,
) -> BoundingBoxConfig:
    """Create a bounding box configuration.

    Args:
        format: Box format (xyxy, xywh, cxcywh). Defaults to "xyxy".
        normalize: Whether to normalize. Defaults to True.
        include_text: Whether to include text. Defaults to True.

    Returns:
        BoundingBoxConfig with the specified settings.

    Raises:
        ValueError: If format is invalid.

    Examples:
        >>> config = create_bounding_box_config(format="xyxy")
        >>> config.format
        'xyxy'

        >>> config = create_bounding_box_config(normalize=False)
        >>> config.normalize
        False

        >>> create_bounding_box_config(format="invalid")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: format must be one of ...
    """
    if format not in VALID_BBOX_FORMATS:
        msg = f"format must be one of {VALID_BBOX_FORMATS}, got '{format}'"
        raise ValueError(msg)

    return BoundingBoxConfig(
        format=format,
        normalize=normalize,
        include_text=include_text,
    )


def create_document_processing_config(
    resize_mode: str = "resize",
    normalize_boxes: bool = True,
    max_pages: int = 10,
) -> DocumentProcessingConfig:
    """Create a document processing pipeline configuration.

    Args:
        resize_mode: How to resize images. Defaults to "resize".
        normalize_boxes: Whether to normalize boxes. Defaults to True.
        max_pages: Maximum pages to process. Defaults to 10.

    Returns:
        DocumentProcessingConfig with the specified settings.

    Raises:
        ValueError: If resize_mode is invalid.
        ValueError: If max_pages is not positive.

    Examples:
        >>> config = create_document_processing_config(resize_mode="pad")
        >>> config.resize_mode
        'pad'

        >>> config = create_document_processing_config(max_pages=5)
        >>> config.max_pages
        5

        >>> create_document_processing_config(max_pages=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_pages must be positive
    """
    if resize_mode not in VALID_RESIZE_MODES:
        msg = f"resize_mode must be one of {VALID_RESIZE_MODES}, got '{resize_mode}'"
        raise ValueError(msg)

    if max_pages <= 0:
        msg = f"max_pages must be positive, got {max_pages}"
        raise ValueError(msg)

    return DocumentProcessingConfig(
        resize_mode=resize_mode,
        normalize_boxes=normalize_boxes,
        max_pages=max_pages,
    )


def list_document_model_types() -> list[str]:
    """List supported document model types.

    Returns:
        Sorted list of document model type names.

    Examples:
        >>> types = list_document_model_types()
        >>> "layoutlmv3" in types
        True
        >>> "donut" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_DOCUMENT_MODELS)


def list_document_tasks() -> list[str]:
    """List supported document tasks.

    Returns:
        Sorted list of document task names.

    Examples:
        >>> tasks = list_document_tasks()
        >>> "classification" in tasks
        True
        >>> "ner" in tasks
        True
        >>> tasks == sorted(tasks)
        True
    """
    return sorted(VALID_DOCUMENT_TASKS)


def list_ocr_backends() -> list[str]:
    """List supported OCR backends.

    Returns:
        Sorted list of OCR backend names.

    Examples:
        >>> backends = list_ocr_backends()
        >>> "tesseract" in backends
        True
        >>> "paddleocr" in backends
        True
        >>> backends == sorted(backends)
        True
    """
    return sorted(VALID_OCR_BACKENDS)


def get_document_model_type(name: str) -> DocumentModelType:
    """Get document model type from name.

    Args:
        name: Model type name.

    Returns:
        DocumentModelType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_document_model_type("layoutlmv3")
        <DocumentModelType.LAYOUTLMV3: 'layoutlmv3'>

        >>> get_document_model_type("donut")
        <DocumentModelType.DONUT: 'donut'>

        >>> get_document_model_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model type must be one of ...
    """
    if name not in VALID_DOCUMENT_MODELS:
        msg = f"model type must be one of {VALID_DOCUMENT_MODELS}, got '{name}'"
        raise ValueError(msg)
    return DocumentModelType(name)


def get_document_task(name: str) -> DocumentTask:
    """Get document task from name.

    Args:
        name: Task name.

    Returns:
        DocumentTask enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_document_task("classification")
        <DocumentTask.CLASSIFICATION: 'classification'>

        >>> get_document_task("ner")
        <DocumentTask.NER: 'ner'>

        >>> get_document_task("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: task must be one of ...
    """
    if name not in VALID_DOCUMENT_TASKS:
        msg = f"task must be one of {VALID_DOCUMENT_TASKS}, got '{name}'"
        raise ValueError(msg)
    return DocumentTask(name)


def get_ocr_backend(name: str) -> OCRBackend:
    """Get OCR backend from name.

    Args:
        name: Backend name.

    Returns:
        OCRBackend enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_ocr_backend("tesseract")
        <OCRBackend.TESSERACT: 'tesseract'>

        >>> get_ocr_backend("easyocr")
        <OCRBackend.EASYOCR: 'easyocr'>

        >>> get_ocr_backend("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: backend must be one of ...
    """
    if name not in VALID_OCR_BACKENDS:
        msg = f"backend must be one of {VALID_OCR_BACKENDS}, got '{name}'"
        raise ValueError(msg)
    return OCRBackend(name)


def normalize_bounding_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    target_range: int = 1000,
) -> tuple[int, int, int, int]:
    """Normalize bounding box to target range (default 0-1000).

    LayoutLM models use 0-1000 normalized coordinates for bounding boxes.
    This function converts pixel coordinates to the normalized range.

    Args:
        box: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        target_range: Target normalization range. Defaults to 1000.

    Returns:
        Normalized bounding box as (x1, y1, x2, y2).

    Raises:
        ValueError: If image dimensions are not positive.
        ValueError: If target_range is not positive.
        ValueError: If box coordinates are negative.

    Examples:
        >>> normalize_bounding_box((100, 200, 300, 400), 1000, 1000)
        (100, 200, 300, 400)

        >>> normalize_bounding_box((50, 100, 150, 200), 500, 500)
        (100, 200, 300, 400)

        >>> normalize_bounding_box((0, 0, 500, 500), 500, 500)
        (0, 0, 1000, 1000)

        >>> normalize_bounding_box((0, 0, 100, 100), 0, 100)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: image_width must be positive
    """
    if image_width <= 0:
        msg = f"image_width must be positive, got {image_width}"
        raise ValueError(msg)

    if image_height <= 0:
        msg = f"image_height must be positive, got {image_height}"
        raise ValueError(msg)

    if target_range <= 0:
        msg = f"target_range must be positive, got {target_range}"
        raise ValueError(msg)

    x1, y1, x2, y2 = box

    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        msg = f"box coordinates cannot be negative, got {box}"
        raise ValueError(msg)

    # Normalize to target range
    norm_x1 = int(x1 * target_range / image_width)
    norm_y1 = int(y1 * target_range / image_height)
    norm_x2 = int(x2 * target_range / image_width)
    norm_y2 = int(y2 * target_range / image_height)

    # Clamp to target range
    norm_x1 = min(max(norm_x1, 0), target_range)
    norm_y1 = min(max(norm_y1, 0), target_range)
    norm_x2 = min(max(norm_x2, 0), target_range)
    norm_y2 = min(max(norm_y2, 0), target_range)

    return (norm_x1, norm_y1, norm_x2, norm_y2)


def estimate_document_tokens(
    num_words: int,
    num_boxes: int,
    model_type: DocumentModelType = DocumentModelType.LAYOUTLMV3,
    include_image_tokens: bool = True,
    image_size: tuple[int, int] = (224, 224),
    patch_size: int = 16,
) -> int:
    """Estimate token count for a document.

    This estimates the total tokens needed to process a document,
    including text tokens, position embeddings, and image tokens
    (for vision-enabled models).

    Args:
        num_words: Number of words in the document.
        num_boxes: Number of bounding boxes.
        model_type: Document model type. Defaults to LAYOUTLMV3.
        include_image_tokens: Whether to include image tokens. Defaults to True.
        image_size: Image size (width, height). Defaults to (224, 224).
        patch_size: Vision encoder patch size. Defaults to 16.

    Returns:
        Estimated total token count.

    Raises:
        ValueError: If num_words is negative.
        ValueError: If num_boxes is negative.
        ValueError: If image dimensions are not positive.
        ValueError: If patch_size is not positive.

    Examples:
        >>> estimate_document_tokens(100, 50)
        296

        >>> estimate_document_tokens(0, 0)
        196

        >>> estimate_document_tokens(100, 50, include_image_tokens=False)
        100

        >>> estimate_document_tokens(-1, 50)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_words cannot be negative
    """
    if num_words < 0:
        msg = f"num_words cannot be negative, got {num_words}"
        raise ValueError(msg)

    if num_boxes < 0:
        msg = f"num_boxes cannot be negative, got {num_boxes}"
        raise ValueError(msg)

    width, height = image_size
    if width <= 0:
        msg = f"image_size width must be positive, got {width}"
        raise ValueError(msg)

    if height <= 0:
        msg = f"image_size height must be positive, got {height}"
        raise ValueError(msg)

    if patch_size <= 0:
        msg = f"patch_size must be positive, got {patch_size}"
        raise ValueError(msg)

    # Estimate text tokens (roughly 1 token per word, simplified)
    text_tokens = num_words

    # Calculate image tokens if needed
    image_tokens = 0
    if include_image_tokens and model_type in (
        DocumentModelType.LAYOUTLMV2,
        DocumentModelType.LAYOUTLMV3,
        DocumentModelType.DONUT,
        DocumentModelType.UDOP,
    ):
        # Number of patches in the image
        patches_w = width // patch_size
        patches_h = height // patch_size
        image_tokens = patches_w * patches_h

    return text_tokens + image_tokens


def get_default_document_config(model_type: DocumentModelType) -> DocumentConfig:
    """Get default configuration for a document model type.

    Args:
        model_type: Document model type.

    Returns:
        Default DocumentConfig for the model.

    Examples:
        >>> config = get_default_document_config(DocumentModelType.LAYOUTLMV3)
        >>> config.max_seq_length
        512

        >>> config = get_default_document_config(DocumentModelType.DONUT)
        >>> config.use_ocr
        False
    """
    # Default configurations for each model type
    defaults: dict[DocumentModelType, DocumentConfig] = {
        DocumentModelType.LAYOUTLM: DocumentConfig(
            model_type=DocumentModelType.LAYOUTLM,
            max_seq_length=512,
            image_size=(224, 224),
            use_ocr=True,
        ),
        DocumentModelType.LAYOUTLMV2: DocumentConfig(
            model_type=DocumentModelType.LAYOUTLMV2,
            max_seq_length=512,
            image_size=(224, 224),
            use_ocr=True,
        ),
        DocumentModelType.LAYOUTLMV3: DocumentConfig(
            model_type=DocumentModelType.LAYOUTLMV3,
            max_seq_length=512,
            image_size=(224, 224),
            use_ocr=True,
        ),
        DocumentModelType.DONUT: DocumentConfig(
            model_type=DocumentModelType.DONUT,
            max_seq_length=2048,
            image_size=(1280, 960),
            use_ocr=False,  # Donut is OCR-free
        ),
        DocumentModelType.UDOP: DocumentConfig(
            model_type=DocumentModelType.UDOP,
            max_seq_length=512,
            image_size=(224, 224),
            use_ocr=True,
        ),
    }
    return defaults[model_type]


def get_default_ocr_config(backend: OCRBackend) -> OCRConfig:
    """Get default configuration for an OCR backend.

    Args:
        backend: OCR backend type.

    Returns:
        Default OCRConfig for the backend.

    Examples:
        >>> config = get_default_ocr_config(OCRBackend.TESSERACT)
        >>> config.dpi
        300

        >>> config = get_default_ocr_config(OCRBackend.PADDLEOCR)
        >>> config.language
        'en'
    """
    # Default configurations for each OCR backend
    defaults: dict[OCRBackend, OCRConfig] = {
        OCRBackend.TESSERACT: OCRConfig(
            backend=OCRBackend.TESSERACT,
            language="en",
            confidence_threshold=0.5,
            dpi=300,
        ),
        OCRBackend.EASYOCR: OCRConfig(
            backend=OCRBackend.EASYOCR,
            language="en",
            confidence_threshold=0.5,
            dpi=300,
        ),
        OCRBackend.PADDLEOCR: OCRConfig(
            backend=OCRBackend.PADDLEOCR,
            language="en",
            confidence_threshold=0.5,
            dpi=300,
        ),
        OCRBackend.DOCTR: OCRConfig(
            backend=OCRBackend.DOCTR,
            language="en",
            confidence_threshold=0.5,
            dpi=300,
        ),
    }
    return defaults[backend]
