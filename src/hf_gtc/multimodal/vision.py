"""Vision and multimodal model utilities.

This module provides functions for working with vision-language models,
image processing, and multimodal inference configurations.

Examples:
    >>> from hf_gtc.multimodal.vision import create_image_config
    >>> config = create_image_config(width=224, height=224)
    >>> config.width
    224
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class VisionModelType(Enum):
    """Supported vision model types.

    Attributes:
        CLIP: CLIP vision encoder.
        SIGLIP: SigLIP vision encoder.
        VIT: Vision Transformer.
        DINO: DINO self-supervised vision model.
        EVA: EVA vision model.
        CONVNEXT: ConvNeXt vision model.

    Examples:
        >>> VisionModelType.CLIP.value
        'clip'
        >>> VisionModelType.VIT.value
        'vit'
    """

    CLIP = "clip"
    SIGLIP = "siglip"
    VIT = "vit"
    DINO = "dino"
    EVA = "eva"
    CONVNEXT = "convnext"


VALID_VISION_MODELS = frozenset(m.value for m in VisionModelType)


class ModalityType(Enum):
    """Supported modality types.

    Attributes:
        TEXT: Text-only modality.
        IMAGE: Image-only modality.
        AUDIO: Audio-only modality.
        VIDEO: Video modality.
        TEXT_IMAGE: Text and image multimodal.
        TEXT_AUDIO: Text and audio multimodal.
        TEXT_VIDEO: Text and video multimodal.

    Examples:
        >>> ModalityType.TEXT_IMAGE.value
        'text_image'
        >>> ModalityType.IMAGE.value
        'image'
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT_IMAGE = "text_image"
    TEXT_AUDIO = "text_audio"
    TEXT_VIDEO = "text_video"


VALID_MODALITIES = frozenset(m.value for m in ModalityType)

# Common image sizes for vision models
ImageSizePreset = Literal["small", "medium", "large", "xlarge"]
IMAGE_SIZE_PRESETS: dict[ImageSizePreset, tuple[int, int]] = {
    "small": (224, 224),
    "medium": (336, 336),
    "large": (448, 448),
    "xlarge": (672, 672),
}


@dataclass(frozen=True, slots=True)
class ImageConfig:
    """Configuration for image processing.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels.
        normalize: Whether to normalize pixel values.
        resize_mode: Resize mode (crop, pad, stretch).

    Examples:
        >>> config = ImageConfig(
        ...     width=224,
        ...     height=224,
        ...     channels=3,
        ...     normalize=True,
        ...     resize_mode="crop",
        ... )
        >>> config.width
        224
    """

    width: int
    height: int
    channels: int
    normalize: bool
    resize_mode: str


@dataclass(frozen=True, slots=True)
class VisionEncoderConfig:
    """Configuration for vision encoder.

    Attributes:
        model_type: Type of vision model.
        patch_size: Patch size for ViT-style models.
        hidden_size: Hidden dimension size.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.

    Examples:
        >>> config = VisionEncoderConfig(
        ...     model_type=VisionModelType.CLIP,
        ...     patch_size=14,
        ...     hidden_size=768,
        ...     num_layers=12,
        ...     num_heads=12,
        ... )
        >>> config.patch_size
        14
    """

    model_type: VisionModelType
    patch_size: int
    hidden_size: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True, slots=True)
class ImageInput:
    """Represents an image input for inference.

    Attributes:
        path: Path to image file.
        width: Original image width.
        height: Original image height.
        format: Image format (png, jpg, etc).

    Examples:
        >>> img = ImageInput(
        ...     path="image.png",
        ...     width=1024,
        ...     height=768,
        ...     format="png",
        ... )
        >>> img.format
        'png'
    """

    path: str
    width: int
    height: int
    format: str


@dataclass(frozen=True, slots=True)
class MultimodalInput:
    """Represents a multimodal input.

    Attributes:
        text: Text input.
        images: Tuple of image inputs.
        modality: Modality type.

    Examples:
        >>> inp = MultimodalInput(
        ...     text="Describe this image",
        ...     images=(),
        ...     modality=ModalityType.TEXT_IMAGE,
        ... )
        >>> inp.modality
        <ModalityType.TEXT_IMAGE: 'text_image'>
    """

    text: str
    images: tuple[ImageInput, ...]
    modality: ModalityType


@dataclass(frozen=True, slots=True)
class MultimodalConfig:
    """Configuration for multimodal model.

    Attributes:
        image_config: Image processing configuration.
        vision_encoder: Vision encoder configuration.
        max_images: Maximum number of images.
        modality: Modality type.

    Examples:
        >>> img_cfg = ImageConfig(224, 224, 3, True, "crop")
        >>> vis_cfg = VisionEncoderConfig(
        ...     VisionModelType.CLIP, 14, 768, 12, 12
        ... )
        >>> config = MultimodalConfig(
        ...     image_config=img_cfg,
        ...     vision_encoder=vis_cfg,
        ...     max_images=5,
        ...     modality=ModalityType.TEXT_IMAGE,
        ... )
        >>> config.max_images
        5
    """

    image_config: ImageConfig
    vision_encoder: VisionEncoderConfig
    max_images: int
    modality: ModalityType


@dataclass(frozen=True, slots=True)
class ImageProcessor:
    """Image processor configuration.

    Attributes:
        name: Processor name.
        image_mean: Normalization mean values.
        image_std: Normalization std values.
        do_resize: Whether to resize images.
        do_normalize: Whether to normalize images.

    Examples:
        >>> proc = ImageProcessor(
        ...     name="clip",
        ...     image_mean=(0.485, 0.456, 0.406),
        ...     image_std=(0.229, 0.224, 0.225),
        ...     do_resize=True,
        ...     do_normalize=True,
        ... )
        >>> proc.do_resize
        True
    """

    name: str
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    do_resize: bool
    do_normalize: bool


def validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions.

    Args:
        width: Image width.
        height: Image height.

    Raises:
        ValueError: If dimensions are invalid.

    Examples:
        >>> validate_image_dimensions(224, 224)  # No error

        >>> validate_image_dimensions(0, 224)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: width must be positive
    """
    if width <= 0:
        msg = f"width must be positive, got {width}"
        raise ValueError(msg)

    if height <= 0:
        msg = f"height must be positive, got {height}"
        raise ValueError(msg)


def validate_image_config(config: ImageConfig) -> None:
    """Validate image configuration.

    Args:
        config: Image configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = ImageConfig(224, 224, 3, True, "crop")
        >>> validate_image_config(config)  # No error

        >>> bad_config = ImageConfig(-1, 224, 3, True, "crop")
        >>> validate_image_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: width must be positive
    """
    validate_image_dimensions(config.width, config.height)

    if config.channels <= 0:
        msg = f"channels must be positive, got {config.channels}"
        raise ValueError(msg)

    valid_resize_modes = {"crop", "pad", "stretch"}
    if config.resize_mode not in valid_resize_modes:
        msg = (
            f"resize_mode must be one of {valid_resize_modes}, "
            f"got '{config.resize_mode}'"
        )
        raise ValueError(msg)


def validate_multimodal_config(config: MultimodalConfig) -> None:
    """Validate multimodal configuration.

    Args:
        config: Multimodal configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> img_cfg = ImageConfig(224, 224, 3, True, "crop")
        >>> vis_cfg = VisionEncoderConfig(
        ...     VisionModelType.CLIP, 14, 768, 12, 12
        ... )
        >>> config = MultimodalConfig(img_cfg, vis_cfg, 5, ModalityType.TEXT_IMAGE)
        >>> validate_multimodal_config(config)  # No error
    """
    validate_image_config(config.image_config)

    if config.max_images <= 0:
        msg = f"max_images must be positive, got {config.max_images}"
        raise ValueError(msg)

    if config.vision_encoder.patch_size <= 0:
        msg = f"patch_size must be positive, got {config.vision_encoder.patch_size}"
        raise ValueError(msg)


def create_image_config(
    width: int = 224,
    height: int = 224,
    channels: int = 3,
    normalize: bool = True,
    resize_mode: str = "crop",
) -> ImageConfig:
    """Create an image configuration.

    Args:
        width: Image width. Defaults to 224.
        height: Image height. Defaults to 224.
        channels: Number of channels. Defaults to 3.
        normalize: Whether to normalize. Defaults to True.
        resize_mode: Resize mode. Defaults to "crop".

    Returns:
        ImageConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_image_config(width=224, height=224)
        >>> config.width
        224

        >>> config = create_image_config(resize_mode="pad")
        >>> config.resize_mode
        'pad'

        >>> create_image_config(width=-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: width must be positive
    """
    config = ImageConfig(
        width=width,
        height=height,
        channels=channels,
        normalize=normalize,
        resize_mode=resize_mode,
    )
    validate_image_config(config)
    return config


def create_vision_encoder_config(
    model_type: str = "clip",
    patch_size: int = 14,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> VisionEncoderConfig:
    """Create a vision encoder configuration.

    Args:
        model_type: Vision model type. Defaults to "clip".
        patch_size: Patch size. Defaults to 14.
        hidden_size: Hidden dimension. Defaults to 768.
        num_layers: Number of layers. Defaults to 12.
        num_heads: Number of attention heads. Defaults to 12.

    Returns:
        VisionEncoderConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_vision_encoder_config(model_type="clip")
        >>> config.model_type
        <VisionModelType.CLIP: 'clip'>

        >>> config = create_vision_encoder_config(patch_size=16)
        >>> config.patch_size
        16

        >>> create_vision_encoder_config(patch_size=0)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: patch_size must be positive
    """
    if model_type not in VALID_VISION_MODELS:
        msg = f"model_type must be one of {VALID_VISION_MODELS}, got '{model_type}'"
        raise ValueError(msg)

    if patch_size <= 0:
        msg = f"patch_size must be positive, got {patch_size}"
        raise ValueError(msg)

    if hidden_size <= 0:
        msg = f"hidden_size must be positive, got {hidden_size}"
        raise ValueError(msg)

    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)

    if num_heads <= 0:
        msg = f"num_heads must be positive, got {num_heads}"
        raise ValueError(msg)

    return VisionEncoderConfig(
        model_type=VisionModelType(model_type),
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def create_image_input(
    path: str,
    width: int,
    height: int,
    format: str = "png",
) -> ImageInput:
    """Create an image input.

    Args:
        path: Path to image file.
        width: Image width.
        height: Image height.
        format: Image format. Defaults to "png".

    Returns:
        ImageInput with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> img = create_image_input("test.png", 1024, 768)
        >>> img.path
        'test.png'

        >>> create_image_input("", 1024, 768)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: path cannot be empty
    """
    if not path:
        msg = "path cannot be empty"
        raise ValueError(msg)

    validate_image_dimensions(width, height)

    valid_formats = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
    if format.lower() not in valid_formats:
        msg = f"format must be one of {valid_formats}, got '{format}'"
        raise ValueError(msg)

    return ImageInput(
        path=path,
        width=width,
        height=height,
        format=format.lower(),
    )


def create_multimodal_config(
    image_width: int = 224,
    image_height: int = 224,
    vision_model: str = "clip",
    patch_size: int = 14,
    max_images: int = 5,
) -> MultimodalConfig:
    """Create a multimodal configuration.

    Args:
        image_width: Image width. Defaults to 224.
        image_height: Image height. Defaults to 224.
        vision_model: Vision model type. Defaults to "clip".
        patch_size: Patch size. Defaults to 14.
        max_images: Maximum images. Defaults to 5.

    Returns:
        MultimodalConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_multimodal_config(image_width=336)
        >>> config.image_config.width
        336

        >>> create_multimodal_config(max_images=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_images must be positive
    """
    image_config = create_image_config(width=image_width, height=image_height)
    vision_config = create_vision_encoder_config(
        model_type=vision_model, patch_size=patch_size
    )

    config = MultimodalConfig(
        image_config=image_config,
        vision_encoder=vision_config,
        max_images=max_images,
        modality=ModalityType.TEXT_IMAGE,
    )
    validate_multimodal_config(config)
    return config


def list_vision_model_types() -> list[str]:
    """List supported vision model types.

    Returns:
        Sorted list of vision model type names.

    Examples:
        >>> types = list_vision_model_types()
        >>> "clip" in types
        True
        >>> "vit" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_VISION_MODELS)


def list_modality_types() -> list[str]:
    """List supported modality types.

    Returns:
        Sorted list of modality type names.

    Examples:
        >>> types = list_modality_types()
        >>> "text_image" in types
        True
        >>> "text" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_MODALITIES)


def get_vision_model_type(name: str) -> VisionModelType:
    """Get vision model type from name.

    Args:
        name: Model type name.

    Returns:
        VisionModelType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_vision_model_type("clip")
        <VisionModelType.CLIP: 'clip'>

        >>> get_vision_model_type("vit")
        <VisionModelType.VIT: 'vit'>

        >>> get_vision_model_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model type must be one of ...
    """
    if name not in VALID_VISION_MODELS:
        msg = f"model type must be one of {VALID_VISION_MODELS}, got '{name}'"
        raise ValueError(msg)
    return VisionModelType(name)


def get_modality_type(name: str) -> ModalityType:
    """Get modality type from name.

    Args:
        name: Modality type name.

    Returns:
        ModalityType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_modality_type("text_image")
        <ModalityType.TEXT_IMAGE: 'text_image'>

        >>> get_modality_type("text")
        <ModalityType.TEXT: 'text'>

        >>> get_modality_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: modality must be one of ...
    """
    if name not in VALID_MODALITIES:
        msg = f"modality must be one of {VALID_MODALITIES}, got '{name}'"
        raise ValueError(msg)
    return ModalityType(name)


def get_default_image_size(preset: ImageSizePreset) -> tuple[int, int]:
    """Get default image size for a preset.

    Args:
        preset: Size preset name.

    Returns:
        Tuple of (width, height).

    Examples:
        >>> get_default_image_size("small")
        (224, 224)
        >>> get_default_image_size("large")
        (448, 448)
    """
    return IMAGE_SIZE_PRESETS[preset]


def calculate_patch_count(
    width: int,
    height: int,
    patch_size: int,
) -> int:
    """Calculate number of patches for an image.

    Args:
        width: Image width.
        height: Image height.
        patch_size: Size of each patch.

    Returns:
        Number of patches.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_patch_count(224, 224, 14)
        256

        >>> calculate_patch_count(336, 336, 14)
        576

        >>> calculate_patch_count(224, 224, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: patch_size must be positive
    """
    validate_image_dimensions(width, height)

    if patch_size <= 0:
        msg = f"patch_size must be positive, got {patch_size}"
        raise ValueError(msg)

    patches_w = width // patch_size
    patches_h = height // patch_size
    return patches_w * patches_h


def calculate_image_tokens(
    width: int,
    height: int,
    patch_size: int,
    include_cls: bool = True,
) -> int:
    """Calculate number of tokens for an image.

    Args:
        width: Image width.
        height: Image height.
        patch_size: Size of each patch.
        include_cls: Whether to include CLS token. Defaults to True.

    Returns:
        Number of tokens.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_image_tokens(224, 224, 14, include_cls=True)
        257

        >>> calculate_image_tokens(224, 224, 14, include_cls=False)
        256
    """
    patches = calculate_patch_count(width, height, patch_size)
    return patches + (1 if include_cls else 0)


def estimate_multimodal_memory(
    num_images: int,
    image_width: int,
    image_height: int,
    patch_size: int = 14,
    hidden_size: int = 768,
    dtype_bytes: int = 2,
) -> int:
    """Estimate memory usage for multimodal input.

    Args:
        num_images: Number of images.
        image_width: Image width.
        image_height: Image height.
        patch_size: Patch size. Defaults to 14.
        hidden_size: Hidden dimension. Defaults to 768.
        dtype_bytes: Bytes per element. Defaults to 2 (FP16).

    Returns:
        Estimated memory in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> mem = estimate_multimodal_memory(1, 224, 224)
        >>> mem > 0
        True

        >>> estimate_multimodal_memory(0, 224, 224)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_images must be positive
    """
    if num_images <= 0:
        msg = f"num_images must be positive, got {num_images}"
        raise ValueError(msg)

    tokens_per_image = calculate_image_tokens(
        image_width, image_height, patch_size, include_cls=True
    )
    total_tokens = num_images * tokens_per_image
    return total_tokens * hidden_size * dtype_bytes


def format_image_size(width: int, height: int) -> str:
    """Format image size as string.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        Formatted size string.

    Examples:
        >>> format_image_size(224, 224)
        '224x224'

        >>> format_image_size(1920, 1080)
        '1920x1080'
    """
    return f"{width}x{height}"


def get_recommended_processor(model_type: VisionModelType) -> ImageProcessor:
    """Get recommended image processor for a model type.

    Args:
        model_type: Vision model type.

    Returns:
        Recommended ImageProcessor configuration.

    Examples:
        >>> proc = get_recommended_processor(VisionModelType.CLIP)
        >>> proc.name
        'clip'

        >>> proc = get_recommended_processor(VisionModelType.DINO)
        >>> proc.name
        'dino'
    """
    # Standard ImageNet normalization values
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # CLIP uses different normalization
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    processors: dict[VisionModelType, ImageProcessor] = {
        VisionModelType.CLIP: ImageProcessor(
            name="clip",
            image_mean=clip_mean,
            image_std=clip_std,
            do_resize=True,
            do_normalize=True,
        ),
        VisionModelType.SIGLIP: ImageProcessor(
            name="siglip",
            image_mean=clip_mean,
            image_std=clip_std,
            do_resize=True,
            do_normalize=True,
        ),
        VisionModelType.VIT: ImageProcessor(
            name="vit",
            image_mean=imagenet_mean,
            image_std=imagenet_std,
            do_resize=True,
            do_normalize=True,
        ),
        VisionModelType.DINO: ImageProcessor(
            name="dino",
            image_mean=imagenet_mean,
            image_std=imagenet_std,
            do_resize=True,
            do_normalize=True,
        ),
        VisionModelType.EVA: ImageProcessor(
            name="eva",
            image_mean=imagenet_mean,
            image_std=imagenet_std,
            do_resize=True,
            do_normalize=True,
        ),
        VisionModelType.CONVNEXT: ImageProcessor(
            name="convnext",
            image_mean=imagenet_mean,
            image_std=imagenet_std,
            do_resize=True,
            do_normalize=True,
        ),
    }
    return processors[model_type]
