"""Generation utilities for image, video, and text generation.

This module provides utilities for working with diffusion models
including Stable Diffusion, SDXL, and text generation sampling.

Examples:
    >>> from hf_gtc.generation import diffusers, sampling
    >>> config = diffusers.create_generation_config(num_inference_steps=20)
    >>> config.num_inference_steps
    20
    >>> samp = sampling.create_sampling_config(temperature=0.7)
    >>> samp.temperature
    0.7
"""

from __future__ import annotations

from hf_gtc.generation import diffusers
from hf_gtc.generation.sampling import (
    VALID_STOP_CRITERIA,
    VALID_STRATEGIES,
    BeamSearchConfig,
    ContrastiveConfig,
    GenerationConstraints,
    SamplingConfig,
    SamplingStrategy,
    StoppingCriteria,
    calculate_effective_vocab_size,
    create_beam_search_config,
    create_contrastive_config,
    create_generation_constraints,
    create_sampling_config,
    create_stopping_criteria,
    estimate_generation_memory,
    get_recommended_config,
    get_sampling_strategy,
    list_sampling_strategies,
    validate_beam_search_config,
    validate_generation_constraints,
    validate_sampling_config,
)

__all__: list[str] = [
    "VALID_STOP_CRITERIA",
    "VALID_STRATEGIES",
    "BeamSearchConfig",
    "ContrastiveConfig",
    "GenerationConstraints",
    "SamplingConfig",
    "SamplingStrategy",
    "StoppingCriteria",
    "calculate_effective_vocab_size",
    "create_beam_search_config",
    "create_contrastive_config",
    "create_generation_constraints",
    "create_sampling_config",
    "create_stopping_criteria",
    "diffusers",
    "estimate_generation_memory",
    "get_recommended_config",
    "get_sampling_strategy",
    "list_sampling_strategies",
    "validate_beam_search_config",
    "validate_generation_constraints",
    "validate_sampling_config",
]
