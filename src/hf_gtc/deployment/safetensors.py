"""SafeTensors serialization utilities.

This module provides functions for working with the safetensors format
for safe and efficient tensor serialization.

Examples:
    >>> from hf_gtc.deployment.safetensors import create_save_config
    >>> config = create_save_config(metadata={"format": "pt"})
    >>> config.metadata
    {'format': 'pt'}
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


class TensorFormat(Enum):
    """Supported tensor formats.

    Attributes:
        PYTORCH: PyTorch tensor format.
        NUMPY: NumPy array format.
        TENSORFLOW: TensorFlow tensor format.
        FLAX: Flax/JAX array format.

    Examples:
        >>> TensorFormat.PYTORCH.value
        'pt'
        >>> TensorFormat.NUMPY.value
        'np'
    """

    PYTORCH = "pt"
    NUMPY = "np"
    TENSORFLOW = "tf"
    FLAX = "flax"


VALID_TENSOR_FORMATS = frozenset(f.value for f in TensorFormat)

# Data types supported by safetensors
DType = Literal["F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"]
VALID_DTYPES = frozenset(
    {"F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"}
)


@dataclass(frozen=True, slots=True)
class SaveConfig:
    """Configuration for saving tensors.

    Attributes:
        metadata: Optional metadata dictionary.
        force_contiguous: Force tensors to be contiguous.
        shared_memory: Use shared memory for loading.

    Examples:
        >>> config = SaveConfig(
        ...     metadata={"format": "pt"},
        ...     force_contiguous=True,
        ...     shared_memory=False,
        ... )
        >>> config.force_contiguous
        True
    """

    metadata: dict[str, str] | None
    force_contiguous: bool
    shared_memory: bool


@dataclass(frozen=True, slots=True)
class LoadConfig:
    """Configuration for loading tensors.

    Attributes:
        device: Device to load tensors to.
        framework: Target framework (pt, np, tf, flax).
        strict: Strict loading mode.

    Examples:
        >>> config = LoadConfig(
        ...     device="cpu",
        ...     framework=TensorFormat.PYTORCH,
        ...     strict=True,
        ... )
        >>> config.device
        'cpu'
    """

    device: str
    framework: TensorFormat
    strict: bool


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Information about a tensor in a safetensors file.

    Attributes:
        name: Tensor name/key.
        shape: Tensor shape.
        dtype: Data type string.
        size_bytes: Size in bytes.

    Examples:
        >>> info = TensorInfo(
        ...     name="weight",
        ...     shape=(768, 768),
        ...     dtype="F16",
        ...     size_bytes=1179648,
        ... )
        >>> info.name
        'weight'
    """

    name: str
    shape: tuple[int, ...]
    dtype: DType
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Information about a safetensors file.

    Attributes:
        path: File path.
        total_size_bytes: Total file size.
        num_tensors: Number of tensors.
        metadata: File metadata.
        tensors: Tuple of tensor information.

    Examples:
        >>> info = FileInfo(
        ...     path="model.safetensors",
        ...     total_size_bytes=1000000,
        ...     num_tensors=10,
        ...     metadata={"format": "pt"},
        ...     tensors=(),
        ... )
        >>> info.num_tensors
        10
    """

    path: str
    total_size_bytes: int
    num_tensors: int
    metadata: dict[str, str]
    tensors: tuple[TensorInfo, ...]


def validate_save_config(config: SaveConfig) -> None:
    """Validate save configuration.

    Args:
        config: Save configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = SaveConfig(
        ...     metadata={"format": "pt"},
        ...     force_contiguous=True,
        ...     shared_memory=False,
        ... )
        >>> validate_save_config(config)  # No error
    """
    if config.metadata is not None:
        for key, value in config.metadata.items():
            if not isinstance(key, str):
                msg = f"metadata keys must be strings, got {type(key)}"
                raise ValueError(msg)
            if not isinstance(value, str):
                msg = f"metadata values must be strings, got {type(value)}"
                raise ValueError(msg)


def validate_load_config(config: LoadConfig) -> None:
    """Validate load configuration.

    Args:
        config: Load configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = LoadConfig(
        ...     device="cpu",
        ...     framework=TensorFormat.PYTORCH,
        ...     strict=True,
        ... )
        >>> validate_load_config(config)  # No error

        >>> bad_config = LoadConfig(
        ...     device="",
        ...     framework=TensorFormat.PYTORCH,
        ...     strict=True,
        ... )
        >>> validate_load_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: device cannot be empty
    """
    if not config.device:
        msg = "device cannot be empty"
        raise ValueError(msg)


def create_save_config(
    metadata: dict[str, str] | None = None,
    force_contiguous: bool = True,
    shared_memory: bool = False,
) -> SaveConfig:
    """Create a save configuration.

    Args:
        metadata: Optional metadata dictionary. Defaults to None.
        force_contiguous: Force tensors to be contiguous. Defaults to True.
        shared_memory: Use shared memory. Defaults to False.

    Returns:
        SaveConfig with the specified settings.

    Examples:
        >>> config = create_save_config(metadata={"format": "pt"})
        >>> config.metadata
        {'format': 'pt'}

        >>> config = create_save_config(force_contiguous=False)
        >>> config.force_contiguous
        False
    """
    config = SaveConfig(
        metadata=metadata,
        force_contiguous=force_contiguous,
        shared_memory=shared_memory,
    )
    validate_save_config(config)
    return config


def create_load_config(
    device: str = "cpu",
    framework: str = "pt",
    strict: bool = True,
) -> LoadConfig:
    """Create a load configuration.

    Args:
        device: Device to load to. Defaults to "cpu".
        framework: Target framework. Defaults to "pt".
        strict: Strict loading mode. Defaults to True.

    Returns:
        LoadConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_load_config(device="cuda:0")
        >>> config.device
        'cuda:0'

        >>> config = create_load_config(framework="np")
        >>> config.framework
        <TensorFormat.NUMPY: 'np'>

        >>> create_load_config(device="")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: device cannot be empty
    """
    if framework not in VALID_TENSOR_FORMATS:
        msg = f"framework must be one of {VALID_TENSOR_FORMATS}, got '{framework}'"
        raise ValueError(msg)

    config = LoadConfig(
        device=device,
        framework=TensorFormat(framework),
        strict=strict,
    )
    validate_load_config(config)
    return config


def list_tensor_formats() -> list[str]:
    """List supported tensor formats.

    Returns:
        Sorted list of format names.

    Examples:
        >>> formats = list_tensor_formats()
        >>> "pt" in formats
        True
        >>> "np" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_TENSOR_FORMATS)


def list_dtypes() -> list[str]:
    """List supported data types.

    Returns:
        Sorted list of dtype names.

    Examples:
        >>> dtypes = list_dtypes()
        >>> "F32" in dtypes
        True
        >>> "F16" in dtypes
        True
        >>> "BF16" in dtypes
        True
    """
    return sorted(VALID_DTYPES)


def estimate_file_size(
    shapes: tuple[tuple[int, ...], ...],
    dtype: DType = "F16",
) -> int:
    """Estimate safetensors file size.

    Args:
        shapes: Tuple of tensor shapes.
        dtype: Data type. Defaults to "F16".

    Returns:
        Estimated file size in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate_file_size(((768, 768), (768,)), dtype="F16")
        1181184

        >>> estimate_file_size(((1000, 1000),), dtype="F32")
        4000000

        >>> estimate_file_size(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: shapes cannot be empty
    """
    if not shapes:
        msg = "shapes cannot be empty"
        raise ValueError(msg)

    if dtype not in VALID_DTYPES:
        msg = f"dtype must be one of {VALID_DTYPES}, got '{dtype}'"
        raise ValueError(msg)

    bytes_per_element = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }

    total_elements = 0
    for shape in shapes:
        if not shape:
            continue
        elements = 1
        for dim in shape:
            if dim <= 0:
                msg = f"dimensions must be positive, got {dim}"
                raise ValueError(msg)
            elements *= dim
        total_elements += elements

    return total_elements * bytes_per_element[dtype]


def calculate_memory_savings(
    original_dtype: DType,
    target_dtype: DType,
) -> float:
    """Calculate memory savings from dtype conversion.

    Args:
        original_dtype: Original data type.
        target_dtype: Target data type.

    Returns:
        Memory savings as a ratio (1.0 = no change, 0.5 = 50% reduction).

    Raises:
        ValueError: If dtypes are invalid.

    Examples:
        >>> calculate_memory_savings("F32", "F16")
        0.5

        >>> calculate_memory_savings("F32", "F32")
        1.0

        >>> calculate_memory_savings("F16", "F32")
        2.0
    """
    if original_dtype not in VALID_DTYPES:
        msg = f"original_dtype must be one of {VALID_DTYPES}, got '{original_dtype}'"
        raise ValueError(msg)

    if target_dtype not in VALID_DTYPES:
        msg = f"target_dtype must be one of {VALID_DTYPES}, got '{target_dtype}'"
        raise ValueError(msg)

    bytes_per_element = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }

    return bytes_per_element[target_dtype] / bytes_per_element[original_dtype]


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.

    Raises:
        ValueError: If size is negative.

    Examples:
        >>> format_size(1024)
        '1.00 KB'

        >>> format_size(1048576)
        '1.00 MB'

        >>> format_size(1073741824)
        '1.00 GB'

        >>> format_size(500)
        '500 B'
    """
    if size_bytes < 0:
        msg = f"size_bytes cannot be negative, got {size_bytes}"
        raise ValueError(msg)

    if size_bytes < 1024:
        return f"{size_bytes} B"

    size: float = float(size_bytes)
    for unit in ("KB", "MB", "GB", "TB", "PB"):
        size /= 1024
        if size < 1024:
            return f"{size:.2f} {unit}"

    return f"{size:.2f} EB"


def get_recommended_dtype(model_size: str) -> DType:
    """Get recommended dtype for a model size.

    Args:
        model_size: Model size category ("small", "medium", "large", "xlarge").

    Returns:
        Recommended dtype.

    Raises:
        ValueError: If model_size is invalid.

    Examples:
        >>> get_recommended_dtype("small")
        'F32'
        >>> get_recommended_dtype("large")
        'F16'
        >>> get_recommended_dtype("xlarge")
        'BF16'
    """
    valid_sizes = {"small", "medium", "large", "xlarge"}
    if model_size not in valid_sizes:
        msg = f"model_size must be one of {valid_sizes}, got '{model_size}'"
        raise ValueError(msg)

    dtype_map: dict[str, DType] = {
        "small": "F32",
        "medium": "F16",
        "large": "F16",
        "xlarge": "BF16",
    }
    return dtype_map[model_size]


def validate_tensor_name(name: str) -> bool:
    """Validate a tensor name for safetensors compatibility.

    Args:
        name: Tensor name to validate.

    Returns:
        True if name is valid.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> validate_tensor_name("model.layers.0.weight")
        True

        >>> validate_tensor_name("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: tensor name cannot be empty
    """
    if not name:
        msg = "tensor name cannot be empty"
        raise ValueError(msg)

    # Check for invalid characters
    invalid_chars = {"\x00", "\n", "\r"}
    for char in invalid_chars:
        if char in name:
            msg = f"tensor name contains invalid character: {char!r}"
            raise ValueError(msg)

    return True


def create_metadata(
    format_version: str = "1.0",
    framework: str = "pt",
    **kwargs: str,
) -> dict[str, str]:
    """Create standard metadata for safetensors file.

    Args:
        format_version: Format version. Defaults to "1.0".
        framework: Framework identifier. Defaults to "pt".
        **kwargs: Additional metadata key-value pairs.

    Returns:
        Metadata dictionary.

    Examples:
        >>> meta = create_metadata(framework="pt")
        >>> meta["framework"]
        'pt'

        >>> meta = create_metadata(author="user", model="gpt2")
        >>> "author" in meta
        True
    """
    metadata = {
        "format_version": format_version,
        "framework": framework,
    }
    metadata.update(kwargs)
    return metadata
