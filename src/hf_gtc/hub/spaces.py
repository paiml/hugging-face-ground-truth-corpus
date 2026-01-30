"""Spaces API utilities for HuggingFace Hub.

This module provides functions for searching and retrieving information
about Spaces (ML demo apps) on the HuggingFace Hub.

Examples:
    >>> from hf_gtc.hub.spaces import search_spaces, SpaceInfo
    >>> # Functions are callable
    >>> callable(search_spaces)
    True
    >>> # SpaceInfo is a class
    >>> SpaceInfo.__name__
    'SpaceInfo'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from huggingface_hub import SpaceInfo as HFSpaceInfo


class SpaceSDK(Enum):
    """SDK type for HuggingFace Spaces.

    Attributes:
        GRADIO: Gradio SDK for interactive ML demos.
        STREAMLIT: Streamlit SDK for data apps.
        DOCKER: Docker SDK for custom containers.
        STATIC: Static HTML pages.

    Examples:
        >>> SpaceSDK.GRADIO.value
        'gradio'
        >>> SpaceSDK.STREAMLIT.value
        'streamlit'
    """

    GRADIO = "gradio"
    STREAMLIT = "streamlit"
    DOCKER = "docker"
    STATIC = "static"


class SpaceStage(Enum):
    """Runtime stage of a HuggingFace Space.

    Attributes:
        RUNNING: Space is running and accessible.
        PAUSED: Space is paused (can be restarted).
        STOPPED: Space is stopped.
        BUILDING: Space is being built.
        NO_APP_FILE: Space has no app file.
        CONFIG_ERROR: Space has a configuration error.

    Examples:
        >>> SpaceStage.RUNNING.value
        'RUNNING'
        >>> SpaceStage.PAUSED.value
        'PAUSED'
    """

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    BUILDING = "BUILDING"
    NO_APP_FILE = "NO_APP_FILE"
    CONFIG_ERROR = "CONFIG_ERROR"


class SpaceHardware(Enum):
    """Hardware tier for HuggingFace Spaces.

    Attributes:
        CPU_BASIC: Basic CPU (free tier).
        CPU_UPGRADE: Upgraded CPU.
        T4_SMALL: Small T4 GPU.
        T4_MEDIUM: Medium T4 GPU.
        A10G_SMALL: Small A10G GPU.
        A10G_LARGE: Large A10G GPU.
        A100_LARGE: Large A100 GPU.

    Examples:
        >>> SpaceHardware.CPU_BASIC.value
        'cpu-basic'
        >>> SpaceHardware.T4_SMALL.value
        't4-small'
    """

    CPU_BASIC = "cpu-basic"
    CPU_UPGRADE = "cpu-upgrade"
    T4_SMALL = "t4-small"
    T4_MEDIUM = "t4-medium"
    A10G_SMALL = "a10g-small"
    A10G_LARGE = "a10g-large"
    A100_LARGE = "a100-large"


VALID_SDKS = frozenset(s.value for s in SpaceSDK)
VALID_HARDWARE = frozenset(h.value for h in SpaceHardware)


@dataclass(frozen=True, slots=True)
class SpaceInfo:
    """Information about a HuggingFace Space.

    Attributes:
        space_id: The Space identifier (e.g., "gradio/hello-world").
        author: The Space author username.
        title: Human-readable title of the Space.
        sdk: The SDK used (gradio, streamlit, docker, static).
        likes: Number of likes.
        emoji: The emoji icon for the Space.
        hardware: The hardware tier.

    Examples:
        >>> info = SpaceInfo(
        ...     space_id="test/space",
        ...     author="test",
        ...     title="Test Space",
        ...     sdk="gradio",
        ...     likes=100,
        ...     emoji="🚀",
        ...     hardware="cpu-basic",
        ... )
        >>> info.space_id
        'test/space'
        >>> info.author
        'test'
        >>> info.sdk
        'gradio'
    """

    space_id: str
    author: str
    title: str | None
    sdk: str | None
    likes: int
    emoji: str | None
    hardware: str | None


@dataclass(frozen=True, slots=True)
class SpaceRuntime:
    """Runtime information for a HuggingFace Space.

    Attributes:
        stage: Current runtime stage (RUNNING, PAUSED, etc.).
        hardware: Hardware tier being used.
        sdk_version: Version of the SDK.

    Examples:
        >>> runtime = SpaceRuntime(
        ...     stage="RUNNING",
        ...     hardware="cpu-basic",
        ...     sdk_version="3.50.0",
        ... )
        >>> runtime.stage
        'RUNNING'
        >>> runtime.hardware
        'cpu-basic'
    """

    stage: str | None
    hardware: str | None
    sdk_version: str | None


def _convert_space_info(hf_info: HFSpaceInfo) -> SpaceInfo:
    """Convert HuggingFace SpaceInfo to our SpaceInfo dataclass.

    Args:
        hf_info: HuggingFace SpaceInfo object.

    Returns:
        Converted SpaceInfo dataclass.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> mock = MagicMock()
        >>> mock.id = "test/space"
        >>> mock.author = "test"
        >>> mock.title = "Test Space"
        >>> mock.sdk = "gradio"
        >>> mock.likes = 50
        >>> mock.emoji = "🤗"
        >>> mock.hardware = "cpu-basic"
        >>> result = _convert_space_info(mock)
        >>> result.space_id
        'test/space'
        >>> result.author
        'test'
    """
    return SpaceInfo(
        space_id=hf_info.id,
        author=hf_info.author,
        title=getattr(hf_info, "title", None),
        sdk=hf_info.sdk,
        likes=hf_info.likes or 0,
        emoji=getattr(hf_info, "emoji", None),
        hardware=getattr(hf_info, "hardware", None),
    )


def search_spaces(
    query: str | None = None,
    *,
    sdk: str | None = None,
    author: str | None = None,
    limit: int = 10,
    sort: str = "likes",
) -> list[SpaceInfo]:
    """Search for Spaces on the HuggingFace Hub.

    Args:
        query: Search query string. Defaults to None.
        sdk: Filter by SDK (gradio, streamlit, docker, static).
        author: Filter by author username.
        limit: Maximum number of results. Defaults to 10. Must be 1-100.
        sort: Sort field ("likes", "created", "lastModified").

    Returns:
        List of SpaceInfo objects matching the search criteria.

    Raises:
        ValueError: If limit is not between 1 and 100.
        ValueError: If sort is not a valid sort field.
        ValueError: If sdk is not a valid SDK type.

    Examples:
        >>> # Validate parameter constraints
        >>> search_spaces(limit=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: limit must be between 1 and 100, got 0

        >>> search_spaces(limit=101)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: limit must be between 1 and 100, got 101

        >>> search_spaces(sort="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sort must be one of ...

        >>> search_spaces(sdk="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sdk must be one of ...
    """
    if not 1 <= limit <= 100:
        msg = f"limit must be between 1 and 100, got {limit}"
        raise ValueError(msg)

    valid_sorts = {"likes", "created", "lastModified"}
    if sort not in valid_sorts:
        msg = f"sort must be one of {valid_sorts}, got {sort!r}"
        raise ValueError(msg)

    if sdk is not None and sdk not in VALID_SDKS:
        msg = f"sdk must be one of {VALID_SDKS}, got {sdk!r}"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()

    # Build filter parameters
    filter_params = {}
    if sdk:
        filter_params["filter"] = sdk

    spaces = api.list_spaces(
        search=query,
        author=author,
        limit=limit,
        sort=sort,
        direction=-1,
        **filter_params,
    )

    return [_convert_space_info(s) for s in spaces]


def iter_spaces(
    *,
    sdk: str | None = None,
    author: str | None = None,
) -> Iterator[SpaceInfo]:
    """Iterate over all Spaces on the HuggingFace Hub.

    This is a streaming API that yields Spaces one at a time,
    suitable for processing large numbers of Spaces without
    loading them all into memory.

    Args:
        sdk: Filter by SDK (gradio, streamlit, docker, static).
        author: Filter by author username.

    Yields:
        SpaceInfo objects matching the filter criteria.

    Raises:
        ValueError: If sdk is not a valid SDK type.

    Examples:
        >>> # Just verify it returns an iterator
        >>> it = iter_spaces()
        >>> hasattr(it, '__iter__') and hasattr(it, '__next__')
        True

        >>> iter_spaces(sdk="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: sdk must be one of ...
    """
    if sdk is not None and sdk not in VALID_SDKS:
        msg = f"sdk must be one of {VALID_SDKS}, got {sdk!r}"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()

    filter_params = {}
    if sdk:
        filter_params["filter"] = sdk

    spaces = api.list_spaces(
        author=author,
        **filter_params,
    )

    for space in spaces:
        yield _convert_space_info(space)


def get_space_info(space_id: str) -> SpaceInfo:
    """Get information about a specific Space.

    Args:
        space_id: The Space identifier (e.g., "gradio/hello-world").

    Returns:
        SpaceInfo object with Space details.

    Raises:
        ValueError: If space_id is empty.

    Examples:
        >>> get_space_info("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: space_id cannot be empty
    """
    if not space_id:
        msg = "space_id cannot be empty"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()
    space = api.space_info(space_id)
    return _convert_space_info(space)


def get_space_runtime(space_id: str) -> SpaceRuntime:
    """Get runtime information about a specific Space.

    Args:
        space_id: The Space identifier (e.g., "gradio/hello-world").

    Returns:
        SpaceRuntime object with runtime details.

    Raises:
        ValueError: If space_id is empty.

    Examples:
        >>> get_space_runtime("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: space_id cannot be empty
    """
    if not space_id:
        msg = "space_id cannot be empty"
        raise ValueError(msg)

    from huggingface_hub import HfApi

    api = HfApi()
    runtime = api.get_space_runtime(space_id)

    return SpaceRuntime(
        stage=runtime.stage,
        hardware=runtime.hardware,
        sdk_version=getattr(runtime, "sdk_version", None),
    )


def list_sdks() -> list[str]:
    """List all available SDK types for Spaces.

    Returns:
        Sorted list of SDK type names.

    Examples:
        >>> sdks = list_sdks()
        >>> "gradio" in sdks
        True
        >>> "streamlit" in sdks
        True
        >>> sdks == sorted(sdks)
        True
    """
    return sorted(VALID_SDKS)


def list_hardware_tiers() -> list[str]:
    """List all available hardware tiers for Spaces.

    Returns:
        Sorted list of hardware tier names.

    Examples:
        >>> hardware = list_hardware_tiers()
        >>> "cpu-basic" in hardware
        True
        >>> "t4-small" in hardware
        True
        >>> hardware == sorted(hardware)
        True
    """
    return sorted(VALID_HARDWARE)


def validate_sdk(sdk: str) -> bool:
    """Validate if a string is a valid SDK type.

    Args:
        sdk: The SDK string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_sdk("gradio")
        True
        >>> validate_sdk("streamlit")
        True
        >>> validate_sdk("invalid")
        False
        >>> validate_sdk("")
        False
    """
    return sdk in VALID_SDKS


def validate_hardware(hardware: str) -> bool:
    """Validate if a string is a valid hardware tier.

    Args:
        hardware: The hardware string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_hardware("cpu-basic")
        True
        >>> validate_hardware("t4-small")
        True
        >>> validate_hardware("invalid")
        False
        >>> validate_hardware("")
        False
    """
    return hardware in VALID_HARDWARE
