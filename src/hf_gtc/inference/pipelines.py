"""Pipeline creation utilities for HuggingFace models.

This module provides functions for creating inference pipelines
with automatic device placement and configuration.

Examples:
    >>> from hf_gtc.inference.pipelines import list_supported_tasks
    >>> tasks = list_supported_tasks()
    >>> "text-classification" in tasks
    True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from transformers import Pipeline

# Literal type for supported tasks - enables compile-time validation
SupportedTask = Literal[
    "audio-classification",
    "automatic-speech-recognition",
    "depth-estimation",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "image-classification",
    "image-feature-extraction",
    "image-segmentation",
    "image-to-text",
    "mask-generation",
    "ner",
    "object-detection",
    "question-answering",
    "sentiment-analysis",
    "summarization",
    "table-question-answering",
    "text-classification",
    "text-generation",
    "text2text-generation",
    "token-classification",
    "translation",
    "video-classification",
    "visual-question-answering",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
]

# Supported pipeline tasks (subset of transformers pipeline tasks)
SUPPORTED_TASKS: frozenset[str] = frozenset(
    {
        "audio-classification",
        "automatic-speech-recognition",
        "depth-estimation",
        "document-question-answering",
        "feature-extraction",
        "fill-mask",
        "image-classification",
        "image-feature-extraction",
        "image-segmentation",
        "image-to-text",
        "mask-generation",
        "ner",
        "object-detection",
        "question-answering",
        "sentiment-analysis",
        "summarization",
        "table-question-answering",
        "text-classification",
        "text-generation",
        "text2text-generation",
        "token-classification",
        "translation",
        "video-classification",
        "visual-question-answering",
        "zero-shot-classification",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
    }
)


def list_supported_tasks() -> list[str]:
    """List all supported pipeline tasks.

    Returns:
        Sorted list of supported task names.

    Examples:
        >>> tasks = list_supported_tasks()
        >>> isinstance(tasks, list)
        True
        >>> len(tasks) > 0
        True
        >>> "text-classification" in tasks
        True
        >>> "sentiment-analysis" in tasks
        True
        >>> tasks == sorted(tasks)  # Sorted
        True
    """
    return sorted(SUPPORTED_TASKS)


def _get_pipeline_factory() -> Any:
    """Get the pipeline factory function without type stub interference.

    This uses importlib to get the pipeline function in a way that
    bypasses the complex overload type stubs that don't work well
    with union Literal types.

    Returns:
        The transformers.pipeline function.
    """
    import importlib

    module = importlib.import_module("transformers")
    return module.pipeline


def create_pipeline(
    task: SupportedTask,
    model: str | None = None,
    *,
    device: str | int | None = None,
    torch_dtype: Any = None,
    **kwargs: Any,
) -> Pipeline:
    """Create a HuggingFace pipeline for inference.

    This function wraps transformers.pipeline() with automatic
    device detection and sensible defaults.

    Args:
        task: The task type (e.g., "text-classification").
        model: Model identifier or path. If None, uses task default.
        device: Device to use. If None, auto-detects best device.
        torch_dtype: Torch dtype for model weights. If None, uses default.
        **kwargs: Additional arguments passed to pipeline().

    Returns:
        Configured Pipeline object ready for inference.

    Raises:
        ValueError: If task is not supported.

    Examples:
        >>> # Validate task parameter
        >>> create_pipeline("invalid-task")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Unsupported task: 'invalid-task'...

        >>> # Valid task names are accepted
        >>> "text-classification" in SUPPORTED_TASKS
        True
    """
    from hf_gtc.inference.device import get_device

    # Runtime validation for task parameter
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS)[:5]) + "..."
        msg = f"Unsupported task: {task!r}. Supported tasks include: {supported}"
        raise ValueError(msg)

    # Auto-detect device if not specified
    if device is None:
        detected = get_device()
        device = 0 if detected == "cuda" else detected

    # Get pipeline factory and call it
    # Using importlib avoids type stub overload resolution issues
    pipeline_fn = _get_pipeline_factory()
    return pipeline_fn(
        task=task,
        model=model,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs,
    )
