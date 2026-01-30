"""Multimodal recipes for HuggingFace models.

This module provides utilities for vision-language models,
image processing, and multimodal inference.

Examples:
    >>> from hf_gtc.multimodal import ImageConfig, VisionModelType
    >>> config = ImageConfig(width=224, height=224)
    >>> config.width
    224
"""

from __future__ import annotations

from hf_gtc.multimodal.vision import (
    ImageConfig,
    ImageInput,
    ImageProcessor,
    ModalityType,
    MultimodalConfig,
    MultimodalInput,
    VisionEncoderConfig,
    VisionModelType,
    calculate_image_tokens,
    calculate_patch_count,
    create_image_config,
    create_image_input,
    create_multimodal_config,
    create_vision_encoder_config,
    estimate_multimodal_memory,
    format_image_size,
    get_default_image_size,
    get_modality_type,
    get_recommended_processor,
    get_vision_model_type,
    list_modality_types,
    list_vision_model_types,
    validate_image_config,
    validate_image_dimensions,
    validate_multimodal_config,
)

__all__: list[str] = [
    "ImageConfig",
    "ImageInput",
    "ImageProcessor",
    "ModalityType",
    "MultimodalConfig",
    "MultimodalInput",
    "VisionEncoderConfig",
    "VisionModelType",
    "calculate_image_tokens",
    "calculate_patch_count",
    "create_image_config",
    "create_image_input",
    "create_multimodal_config",
    "create_vision_encoder_config",
    "estimate_multimodal_memory",
    "format_image_size",
    "get_default_image_size",
    "get_modality_type",
    "get_recommended_processor",
    "get_vision_model_type",
    "list_modality_types",
    "list_vision_model_types",
    "validate_image_config",
    "validate_image_dimensions",
    "validate_multimodal_config",
]
