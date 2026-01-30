"""Hub API recipes for HuggingFace model and dataset discovery.

This module provides utilities for searching, parsing, and interacting with
the HuggingFace Hub API.

Examples:
    >>> from hf_gtc.hub import search_models, parse_model_card
    >>> # Results depend on network, so just verify import works
    >>> callable(search_models) and callable(parse_model_card)
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

__all__ = [
    "ModelCard",
    "ModelCardMetadata",
    "ModelCardSection",
    "ValidationResult",
    "extract_model_description",
    "get_model_card",
    "list_model_card_sections",
    "parse_model_card",
    "search_datasets",
    "search_models",
    "validate_model_card",
]
