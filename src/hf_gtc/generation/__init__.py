"""Generation utilities for image and video generation.

This module provides utilities for working with diffusion models
including Stable Diffusion, SDXL, and other image generation models.

Examples:
    >>> from hf_gtc.generation import diffusers
    >>> config = diffusers.create_generation_config(num_inference_steps=20)
    >>> config.num_inference_steps
    20
"""

from __future__ import annotations

from hf_gtc.generation import diffusers

__all__ = ["diffusers"]
