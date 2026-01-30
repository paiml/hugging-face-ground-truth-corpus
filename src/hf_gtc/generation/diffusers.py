"""Diffusion model utilities for image generation.

This module provides functions for configuring and running inference
with HuggingFace diffusers models including Stable Diffusion and SDXL.

Examples:
    >>> from hf_gtc.generation.diffusers import create_generation_config
    >>> config = create_generation_config(num_inference_steps=20)
    >>> config.num_inference_steps
    20
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


class SchedulerType(Enum):
    """Supported scheduler types for diffusion models.

    Attributes:
        DDIM: Denoising Diffusion Implicit Models scheduler.
        DDPM: Denoising Diffusion Probabilistic Models scheduler.
        PNDM: Pseudo Numerical Methods for Diffusion Models.
        LMS: Linear Multi-Step scheduler.
        EULER: Euler scheduler.
        EULER_ANCESTRAL: Euler Ancestral scheduler.
        DPM_SOLVER: DPM-Solver scheduler.
        DPM_SOLVER_MULTISTEP: DPM-Solver++ multistep scheduler.

    Examples:
        >>> SchedulerType.EULER.value
        'euler'
        >>> SchedulerType.DPM_SOLVER_MULTISTEP.value
        'dpm_solver_multistep'
    """

    DDIM = "ddim"
    DDPM = "ddpm"
    PNDM = "pndm"
    LMS = "lms"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    DPM_SOLVER = "dpm_solver"
    DPM_SOLVER_MULTISTEP = "dpm_solver_multistep"


VALID_SCHEDULER_TYPES = frozenset(s.value for s in SchedulerType)

# Supported image sizes for common models
SUPPORTED_SIZES = frozenset({256, 512, 768, 1024, 2048})

# Guidance scale type
GuidanceScaleType = Literal["none", "low", "medium", "high", "very_high"]
GUIDANCE_SCALE_MAP: dict[GuidanceScaleType, float] = {
    "none": 1.0,
    "low": 3.5,
    "medium": 7.5,
    "high": 12.0,
    "very_high": 20.0,
}


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Configuration for image generation.

    Attributes:
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        height: Output image height in pixels.
        width: Output image width in pixels.
        scheduler: Scheduler type to use.
        seed: Random seed for reproducibility.
        negative_prompt: Negative prompt for guidance.

    Examples:
        >>> config = GenerationConfig(
        ...     num_inference_steps=50,
        ...     guidance_scale=7.5,
        ...     height=512,
        ...     width=512,
        ...     scheduler=SchedulerType.EULER,
        ...     seed=42,
        ...     negative_prompt=None,
        ... )
        >>> config.num_inference_steps
        50
    """

    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    scheduler: SchedulerType
    seed: int | None
    negative_prompt: str | None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of image generation.

    Attributes:
        images: List of generated images (as bytes or PIL Images).
        seeds: Seeds used for each image.
        nsfw_detected: Whether NSFW content was detected.
        inference_time_ms: Time taken for inference in milliseconds.

    Examples:
        >>> result = GenerationResult(
        ...     images=[b"fake_image_data"],
        ...     seeds=[42],
        ...     nsfw_detected=[False],
        ...     inference_time_ms=1500.0,
        ... )
        >>> len(result.images)
        1
    """

    images: tuple[Any, ...]
    seeds: tuple[int, ...]
    nsfw_detected: tuple[bool, ...]
    inference_time_ms: float


def validate_generation_config(config: GenerationConfig) -> None:
    """Validate generation configuration parameters.

    Args:
        config: Generation configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = GenerationConfig(
        ...     num_inference_steps=50,
        ...     guidance_scale=7.5,
        ...     height=512,
        ...     width=512,
        ...     scheduler=SchedulerType.EULER,
        ...     seed=42,
        ...     negative_prompt=None,
        ... )
        >>> validate_generation_config(config)  # No error

        >>> bad_config = GenerationConfig(
        ...     num_inference_steps=0,
        ...     guidance_scale=7.5,
        ...     height=512,
        ...     width=512,
        ...     scheduler=SchedulerType.EULER,
        ...     seed=42,
        ...     negative_prompt=None,
        ... )
        >>> validate_generation_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_inference_steps must be positive
    """
    if config.num_inference_steps <= 0:
        msg = f"num_inference_steps must be positive, got {config.num_inference_steps}"
        raise ValueError(msg)

    if config.guidance_scale < 1.0:
        msg = f"guidance_scale must be >= 1.0, got {config.guidance_scale}"
        raise ValueError(msg)

    if config.height <= 0:
        msg = f"height must be positive, got {config.height}"
        raise ValueError(msg)

    if config.width <= 0:
        msg = f"width must be positive, got {config.width}"
        raise ValueError(msg)

    if config.height % 8 != 0:
        msg = f"height must be divisible by 8, got {config.height}"
        raise ValueError(msg)

    if config.width % 8 != 0:
        msg = f"width must be divisible by 8, got {config.width}"
        raise ValueError(msg)


def create_generation_config(
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    scheduler: str = "euler",
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> GenerationConfig:
    """Create a generation configuration.

    Args:
        num_inference_steps: Number of denoising steps. Defaults to 50.
        guidance_scale: Classifier-free guidance scale. Defaults to 7.5.
        height: Output image height. Defaults to 512.
        width: Output image width. Defaults to 512.
        scheduler: Scheduler type name. Defaults to "euler".
        seed: Random seed for reproducibility. Defaults to None.
        negative_prompt: Negative prompt for guidance. Defaults to None.

    Returns:
        GenerationConfig with the specified settings.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> config = create_generation_config(num_inference_steps=20)
        >>> config.num_inference_steps
        20

        >>> config = create_generation_config(guidance_scale=12.0)
        >>> config.guidance_scale
        12.0

        >>> create_generation_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     num_inference_steps=0
        ... )
        Traceback (most recent call last):
        ValueError: num_inference_steps must be positive
    """
    if scheduler not in VALID_SCHEDULER_TYPES:
        msg = f"scheduler must be one of {VALID_SCHEDULER_TYPES}, got '{scheduler}'"
        raise ValueError(msg)

    config = GenerationConfig(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        scheduler=SchedulerType(scheduler),
        seed=seed,
        negative_prompt=negative_prompt,
    )
    validate_generation_config(config)
    return config


def get_guidance_scale(level: GuidanceScaleType) -> float:
    """Get guidance scale value for a named level.

    Args:
        level: Named guidance level.

    Returns:
        Corresponding guidance scale value.

    Examples:
        >>> get_guidance_scale("none")
        1.0
        >>> get_guidance_scale("medium")
        7.5
        >>> get_guidance_scale("high")
        12.0
    """
    return GUIDANCE_SCALE_MAP[level]


def list_scheduler_types() -> list[str]:
    """List all supported scheduler types.

    Returns:
        Sorted list of scheduler type names.

    Examples:
        >>> types = list_scheduler_types()
        >>> "euler" in types
        True
        >>> "ddim" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_SCHEDULER_TYPES)


def calculate_latent_size(
    height: int, width: int, scale_factor: int = 8
) -> tuple[int, int]:
    """Calculate latent space dimensions from image dimensions.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        scale_factor: VAE scale factor. Defaults to 8.

    Returns:
        Tuple of (latent_height, latent_width).

    Raises:
        ValueError: If dimensions are not divisible by scale factor.

    Examples:
        >>> calculate_latent_size(512, 512)
        (64, 64)
        >>> calculate_latent_size(1024, 768)
        (128, 96)

        >>> calculate_latent_size(513, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: height must be divisible by scale_factor
    """
    if height <= 0:
        msg = f"height must be positive, got {height}"
        raise ValueError(msg)

    if width <= 0:
        msg = f"width must be positive, got {width}"
        raise ValueError(msg)

    if scale_factor <= 0:
        msg = f"scale_factor must be positive, got {scale_factor}"
        raise ValueError(msg)

    if height % scale_factor != 0:
        msg = f"height must be divisible by scale_factor ({scale_factor}), got {height}"
        raise ValueError(msg)

    if width % scale_factor != 0:
        msg = f"width must be divisible by scale_factor ({scale_factor}), got {width}"
        raise ValueError(msg)

    return height // scale_factor, width // scale_factor


def estimate_vram_usage(
    height: int,
    width: int,
    batch_size: int = 1,
    precision: Literal["fp32", "fp16", "bf16"] = "fp16",
) -> float:
    """Estimate VRAM usage for image generation.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        batch_size: Number of images to generate. Defaults to 1.
        precision: Model precision. Defaults to "fp16".

    Returns:
        Estimated VRAM usage in gigabytes.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> usage = estimate_vram_usage(512, 512)
        >>> 4.0 < usage < 8.0  # Typical SD 1.5 range
        True

        >>> usage_high = estimate_vram_usage(1024, 1024)
        >>> usage_high > usage
        True

        >>> estimate_vram_usage(0, 512)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: height must be positive
    """
    if height <= 0:
        msg = f"height must be positive, got {height}"
        raise ValueError(msg)

    if width <= 0:
        msg = f"width must be positive, got {width}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    valid_precisions = {"fp32", "fp16", "bf16"}
    if precision not in valid_precisions:
        msg = f"precision must be one of {valid_precisions}, got '{precision}'"
        raise ValueError(msg)

    # Base model size (approximate for SD 1.5)
    base_vram_gb = 4.0 if precision == "fp32" else 2.0

    # Latent dimensions
    latent_h = height // 8
    latent_w = width // 8

    # Memory for latents (4 channels, batch_size, fp16/fp32)
    bytes_per_element = 4 if precision == "fp32" else 2
    latent_memory_bytes = latent_h * latent_w * 4 * batch_size * bytes_per_element

    # Memory for intermediate activations (rough estimate)
    activation_multiplier = 3.0

    total_latent_gb = (latent_memory_bytes * activation_multiplier) / (1024**3)

    return base_vram_gb + total_latent_gb


def get_recommended_steps(
    model_type: Literal["sd15", "sd21", "sdxl", "sdxl_turbo"],
) -> int:
    """Get recommended inference steps for a model type.

    Args:
        model_type: Type of diffusion model.

    Returns:
        Recommended number of inference steps.

    Examples:
        >>> get_recommended_steps("sd15")
        50
        >>> get_recommended_steps("sdxl")
        30
        >>> get_recommended_steps("sdxl_turbo")
        4
    """
    steps_map = {
        "sd15": 50,
        "sd21": 50,
        "sdxl": 30,
        "sdxl_turbo": 4,
    }
    return steps_map[model_type]


def get_recommended_size(
    model_type: Literal["sd15", "sd21", "sdxl", "sdxl_turbo"],
) -> tuple[int, int]:
    """Get recommended image size for a model type.

    Args:
        model_type: Type of diffusion model.

    Returns:
        Tuple of (height, width) in pixels.

    Examples:
        >>> get_recommended_size("sd15")
        (512, 512)
        >>> get_recommended_size("sdxl")
        (1024, 1024)
    """
    size_map = {
        "sd15": (512, 512),
        "sd21": (768, 768),
        "sdxl": (1024, 1024),
        "sdxl_turbo": (512, 512),
    }
    return size_map[model_type]


def validate_prompt(prompt: str, max_tokens: int = 77) -> bool:
    """Validate that a prompt is within token limits.

    This is a simple word-based approximation. For accurate token
    counting, use the actual tokenizer.

    Args:
        prompt: Text prompt to validate.
        max_tokens: Maximum allowed tokens. Defaults to 77.

    Returns:
        True if prompt is likely within limits.

    Raises:
        ValueError: If prompt is None or max_tokens is invalid.

    Examples:
        >>> validate_prompt("A cat sitting on a mat")
        True
        >>> validate_prompt("word " * 100)
        False

        >>> validate_prompt("")
        True
    """
    if prompt is None:
        msg = "prompt cannot be None"
        raise ValueError(msg)

    if max_tokens <= 0:
        msg = f"max_tokens must be positive, got {max_tokens}"
        raise ValueError(msg)

    # Rough approximation: ~1.3 tokens per word
    word_count = len(prompt.split())
    estimated_tokens = int(word_count * 1.3)

    return estimated_tokens <= max_tokens


def create_negative_prompt(
    categories: tuple[str, ...] = (),
) -> str:
    """Create a negative prompt from categories.

    Args:
        categories: Categories to include in negative prompt.

    Returns:
        Formatted negative prompt string.

    Examples:
        >>> create_negative_prompt(("blurry", "low quality"))
        'blurry, low quality'

        >>> create_negative_prompt(())
        ''

        >>> prompt = create_negative_prompt(("nsfw", "watermark", "text"))
        >>> "nsfw" in prompt
        True
    """
    return ", ".join(categories)


def get_default_negative_prompt() -> str:
    """Get a sensible default negative prompt.

    Returns:
        Default negative prompt for general use.

    Examples:
        >>> prompt = get_default_negative_prompt()
        >>> "blurry" in prompt
        True
        >>> "low quality" in prompt
        True
    """
    return create_negative_prompt(
        (
            "blurry",
            "low quality",
            "distorted",
            "deformed",
            "disfigured",
            "bad anatomy",
            "watermark",
            "text",
            "signature",
        )
    )
