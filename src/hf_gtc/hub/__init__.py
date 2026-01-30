"""Hub API recipes for HuggingFace model and dataset discovery.

This module provides utilities for searching, parsing, and interacting with
the HuggingFace Hub API, including models, datasets, and Spaces.

Examples:
    >>> from hf_gtc.hub import search_models, parse_model_card, search_spaces
    >>> # Results depend on network, so just verify import works
    >>> callable(search_models) and callable(parse_model_card)
    True
    >>> callable(search_spaces)
    True
"""

from __future__ import annotations

from hf_gtc.hub.cards import (
    ModelCard,
    ModelCardMetadata,
    ModelCardSection,
    ValidationResult,
    extract_model_description,
    get_model_card,
    list_model_card_sections,
    parse_model_card,
    validate_model_card,
)
from hf_gtc.hub.search import search_datasets, search_models
from hf_gtc.hub.spaces import (
    SpaceHardware,
    SpaceInfo,
    SpaceRuntime,
    SpaceSDK,
    SpaceStage,
    get_space_info,
    get_space_runtime,
    iter_spaces,
    list_hardware_tiers,
    list_sdks,
    search_spaces,
    validate_hardware,
    validate_sdk,
)

__all__ = [
    "ModelCard",
    "ModelCardMetadata",
    "ModelCardSection",
    "SpaceHardware",
    "SpaceInfo",
    "SpaceRuntime",
    "SpaceSDK",
    "SpaceStage",
    "ValidationResult",
    "extract_model_description",
    "get_model_card",
    "get_space_info",
    "get_space_runtime",
    "iter_spaces",
    "list_hardware_tiers",
    "list_model_card_sections",
    "list_sdks",
    "parse_model_card",
    "search_datasets",
    "search_models",
    "search_spaces",
    "validate_hardware",
    "validate_model_card",
    "validate_sdk",
]
