"""RAG (Retrieval-Augmented Generation) recipes for HuggingFace models.

This module provides utilities for building RAG systems including
document chunking, embedding, and retrieval patterns.

Examples:
    >>> from hf_gtc.rag import ChunkingStrategy, create_chunking_config
    >>> config = create_chunking_config(chunk_size=512, overlap=50)
    >>> config.chunk_size
    512
"""

from __future__ import annotations

from hf_gtc.rag.retrieval import (
    VALID_CHUNKING_STRATEGIES,
    VALID_DISTANCE_METRICS,
    VALID_RETRIEVAL_METHODS,
    ChunkingConfig,
    ChunkingStrategy,
    ChunkResult,
    DistanceMetric,
    DocumentChunk,
    RAGConfig,
    RetrievalMethod,
    RetrievalResult,
    RetrievalStats,
    calculate_chunk_count,
    calculate_overlap_ratio,
    create_chunking_config,
    create_document_chunk,
    create_rag_config,
    create_retrieval_result,
    estimate_retrieval_latency,
    format_retrieval_stats,
    get_chunking_strategy,
    get_distance_metric,
    get_recommended_chunk_size,
    get_retrieval_method,
    list_chunking_strategies,
    list_distance_metrics,
    list_retrieval_methods,
    validate_chunking_config,
    validate_rag_config,
)

__all__: list[str] = [
    "VALID_CHUNKING_STRATEGIES",
    "VALID_DISTANCE_METRICS",
    "VALID_RETRIEVAL_METHODS",
    "ChunkResult",
    "ChunkingConfig",
    "ChunkingStrategy",
    "DistanceMetric",
    "DocumentChunk",
    "RAGConfig",
    "RetrievalMethod",
    "RetrievalResult",
    "RetrievalStats",
    "calculate_chunk_count",
    "calculate_overlap_ratio",
    "create_chunking_config",
    "create_document_chunk",
    "create_rag_config",
    "create_retrieval_result",
    "estimate_retrieval_latency",
    "format_retrieval_stats",
    "get_chunking_strategy",
    "get_distance_metric",
    "get_recommended_chunk_size",
    "get_retrieval_method",
    "list_chunking_strategies",
    "list_distance_metrics",
    "list_retrieval_methods",
    "validate_chunking_config",
    "validate_rag_config",
]
