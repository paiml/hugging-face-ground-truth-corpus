"""Hub API recipes for HuggingFace model and dataset discovery.

This module provides utilities for searching and interacting with
the HuggingFace Hub API.

Examples:
    >>> from hf_gtc.hub import search_models
    >>> # Results depend on network, so just verify import works
    >>> callable(search_models)
    True
"""

from __future__ import annotations

from hf_gtc.hub.search import search_datasets, search_models

__all__ = ["search_datasets", "search_models"]
