"""GGUF model export utilities.

This module provides utilities for exporting HuggingFace models to
GGUF format for use with llama.cpp and other inference engines.

Examples:
    >>> from hf_gtc.deployment.gguf import GGUFConfig, GGUFQuantType
    >>> config = GGUFConfig(quant_type=GGUFQuantType.Q4_K_M)
    >>> config.quant_type.value
    'q4_k_m'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hf_gtc._validation import validate_not_none


class GGUFQuantType(Enum):
    """GGUF quantization types.

    Examples:
        >>> GGUFQuantType.Q4_0.value
        'q4_0'
        >>> GGUFQuantType.Q4_K_M.value
        'q4_k_m'
    """

    F32 = "f32"
    F16 = "f16"
    Q8_0 = "q8_0"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q4_K_S = "q4_k_s"
    Q4_K_M = "q4_k_m"
    Q5_K_S = "q5_k_s"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q2_K = "q2_k"
    Q3_K_S = "q3_k_s"
    Q3_K_M = "q3_k_m"
    Q3_K_L = "q3_k_l"
    IQ2_XXS = "iq2_xxs"
    IQ2_XS = "iq2_xs"


class GGUFArchitecture(Enum):
    """GGUF model architectures.

    Examples:
        >>> GGUFArchitecture.LLAMA.value
        'llama'
        >>> GGUFArchitecture.MISTRAL.value
        'mistral'
    """

    LLAMA = "llama"
    MISTRAL = "mistral"
    FALCON = "falcon"
    GPT2 = "gpt2"
    GPTJ = "gptj"
    GPTNEOX = "gptneox"
    BLOOM = "bloom"
    MPT = "mpt"
    STABLELM = "stablelm"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"


class GGUFEndian(Enum):
    """GGUF endianness options.

    Examples:
        >>> GGUFEndian.LITTLE.value
        'little'
        >>> GGUFEndian.BIG.value
        'big'
    """

    LITTLE = "little"
    BIG = "big"


@dataclass(frozen=True, slots=True)
class GGUFConfig:
    """Configuration for GGUF export.

    Attributes:
        quant_type: Quantization type.
        architecture: Model architecture.
        vocab_only: Export vocabulary only.
        outtype: Output type for conversion.
        concurrency: Number of threads for quantization.

    Examples:
        >>> config = GGUFConfig(quant_type=GGUFQuantType.Q4_K_M)
        >>> config.quant_type.value
        'q4_k_m'
        >>> config.vocab_only
        False

        >>> config2 = GGUFConfig(
        ...     quant_type=GGUFQuantType.Q8_0,
        ...     architecture=GGUFArchitecture.MISTRAL,
        ... )
        >>> config2.architecture.value
        'mistral'
    """

    quant_type: GGUFQuantType = GGUFQuantType.Q4_K_M
    architecture: GGUFArchitecture | None = None
    vocab_only: bool = False
    outtype: str = "f16"
    concurrency: int = 4


@dataclass(frozen=True, slots=True)
class GGUFMetadata:
    """GGUF file metadata.

    Attributes:
        name: Model name.
        author: Model author.
        version: Model version.
        description: Model description.
        license: License identifier.
        url: Model URL.

    Examples:
        >>> meta = GGUFMetadata(name="my-model", author="me")
        >>> meta.name
        'my-model'
    """

    name: str = ""
    author: str = ""
    version: str = ""
    description: str = ""
    license: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class GGUFExportResult:
    """Result of GGUF export operation.

    Attributes:
        output_path: Path to exported file.
        original_size_mb: Original model size in MB.
        exported_size_mb: Exported model size in MB.
        compression_ratio: Compression ratio achieved.
        quant_type: Quantization type used.

    Examples:
        >>> result = GGUFExportResult(
        ...     output_path="/models/model.gguf",
        ...     original_size_mb=14000,
        ...     exported_size_mb=4000,
        ...     compression_ratio=3.5,
        ...     quant_type=GGUFQuantType.Q4_K_M,
        ... )
        >>> result.compression_ratio
        3.5
    """

    output_path: str
    original_size_mb: float
    exported_size_mb: float
    compression_ratio: float
    quant_type: GGUFQuantType


@dataclass(frozen=True, slots=True)
class GGUFModelInfo:
    """Information about a GGUF model file.

    Attributes:
        path: Path to the GGUF file.
        size_mb: File size in MB.
        quant_type: Quantization type.
        architecture: Model architecture.
        context_length: Maximum context length.
        vocab_size: Vocabulary size.

    Examples:
        >>> info = GGUFModelInfo(
        ...     path="/models/model.gguf",
        ...     size_mb=4000,
        ...     quant_type=GGUFQuantType.Q4_K_M,
        ... )
        >>> info.size_mb
        4000
    """

    path: str
    size_mb: float
    quant_type: GGUFQuantType | None = None
    architecture: GGUFArchitecture | None = None
    context_length: int | None = None
    vocab_size: int | None = None


def validate_gguf_config(config: GGUFConfig) -> None:
    """Validate GGUF configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If config is None.
        ValueError: If concurrency is not positive.

    Examples:
        >>> config = GGUFConfig(quant_type=GGUFQuantType.Q4_K_M)
        >>> validate_gguf_config(config)  # No error

        >>> validate_gguf_config(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: config cannot be None

        >>> bad_config = GGUFConfig(concurrency=0)
        >>> validate_gguf_config(bad_config)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: concurrency must be positive
    """
    validate_not_none(config, "config")

    if config.concurrency <= 0:
        msg = f"concurrency must be positive, got {config.concurrency}"
        raise ValueError(msg)


def validate_gguf_metadata(metadata: GGUFMetadata) -> None:
    """Validate GGUF metadata.

    Args:
        metadata: Metadata to validate.

    Raises:
        ValueError: If metadata is None.

    Examples:
        >>> meta = GGUFMetadata(name="my-model")
        >>> validate_gguf_metadata(meta)  # No error

        >>> validate_gguf_metadata(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: metadata cannot be None
    """
    if metadata is None:
        msg = "metadata cannot be None"
        raise ValueError(msg)


def create_gguf_config(
    quant_type: GGUFQuantType | str = GGUFQuantType.Q4_K_M,
    architecture: GGUFArchitecture | str | None = None,
    vocab_only: bool = False,
    concurrency: int = 4,
) -> GGUFConfig:
    """Create a GGUF configuration.

    Args:
        quant_type: Quantization type. Defaults to Q4_K_M.
        architecture: Model architecture. Defaults to None (auto-detect).
        vocab_only: Export vocabulary only. Defaults to False.
        concurrency: Number of threads. Defaults to 4.

    Returns:
        Validated GGUFConfig instance.

    Raises:
        ValueError: If concurrency is not positive.

    Examples:
        >>> config = create_gguf_config(quant_type="q4_k_m")
        >>> config.quant_type
        <GGUFQuantType.Q4_K_M: 'q4_k_m'>

        >>> config2 = create_gguf_config(architecture="llama")
        >>> config2.architecture
        <GGUFArchitecture.LLAMA: 'llama'>
    """
    if isinstance(quant_type, str):
        quant_type = get_gguf_quant_type(quant_type)
    if isinstance(architecture, str):
        architecture = get_gguf_architecture(architecture)

    config = GGUFConfig(
        quant_type=quant_type,
        architecture=architecture,
        vocab_only=vocab_only,
        concurrency=concurrency,
    )
    validate_gguf_config(config)
    return config


def create_gguf_metadata(
    name: str = "",
    author: str = "",
    version: str = "",
    description: str = "",
    license: str = "",
    url: str = "",
) -> GGUFMetadata:
    """Create GGUF metadata.

    Args:
        name: Model name. Defaults to "".
        author: Model author. Defaults to "".
        version: Model version. Defaults to "".
        description: Model description. Defaults to "".
        license: License identifier. Defaults to "".
        url: Model URL. Defaults to "".

    Returns:
        GGUFMetadata instance.

    Examples:
        >>> meta = create_gguf_metadata(name="my-model", author="me")
        >>> meta.name
        'my-model'
        >>> meta.author
        'me'
    """
    return GGUFMetadata(
        name=name,
        author=author,
        version=version,
        description=description,
        license=license,
        url=url,
    )


def create_gguf_export_result(
    output_path: str,
    original_size_mb: float,
    exported_size_mb: float,
    quant_type: GGUFQuantType,
) -> GGUFExportResult:
    """Create a GGUF export result.

    Args:
        output_path: Path to exported file.
        original_size_mb: Original size in MB.
        exported_size_mb: Exported size in MB.
        quant_type: Quantization type used.

    Returns:
        GGUFExportResult instance.

    Raises:
        ValueError: If output_path is empty.
        ValueError: If sizes are not positive.

    Examples:
        >>> result = create_gguf_export_result(
        ...     "/models/model.gguf", 14000, 4000, GGUFQuantType.Q4_K_M
        ... )
        >>> result.compression_ratio
        3.5

        >>> create_gguf_export_result("", 14000, 4000, GGUFQuantType.Q4_K_M)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: output_path cannot be empty
    """
    if not output_path:
        msg = "output_path cannot be empty"
        raise ValueError(msg)

    if original_size_mb <= 0:
        msg = f"original_size_mb must be positive, got {original_size_mb}"
        raise ValueError(msg)

    if exported_size_mb <= 0:
        msg = f"exported_size_mb must be positive, got {exported_size_mb}"
        raise ValueError(msg)

    compression_ratio = original_size_mb / exported_size_mb

    return GGUFExportResult(
        output_path=output_path,
        original_size_mb=original_size_mb,
        exported_size_mb=exported_size_mb,
        compression_ratio=compression_ratio,
        quant_type=quant_type,
    )


def estimate_gguf_size(
    model_params: int,
    quant_type: GGUFQuantType,
) -> float:
    """Estimate GGUF file size for a model.

    Args:
        model_params: Number of model parameters.
        quant_type: Quantization type.

    Returns:
        Estimated size in MB.

    Raises:
        ValueError: If model_params is not positive.

    Examples:
        >>> size = estimate_gguf_size(7_000_000_000, GGUFQuantType.Q4_K_M)
        >>> size > 0
        True
        >>> size < 14000  # Much smaller than FP32
        True

        >>> estimate_gguf_size(0, GGUFQuantType.Q4_K_M)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_params must be positive
    """
    if model_params <= 0:
        msg = "model_params must be positive"
        raise ValueError(msg)

    # Bits per weight for each quant type
    bits_per_weight = {
        GGUFQuantType.F32: 32,
        GGUFQuantType.F16: 16,
        GGUFQuantType.Q8_0: 8,
        GGUFQuantType.Q6_K: 6,
        GGUFQuantType.Q5_0: 5,
        GGUFQuantType.Q5_1: 5,
        GGUFQuantType.Q5_K_S: 5,
        GGUFQuantType.Q5_K_M: 5,
        GGUFQuantType.Q4_0: 4,
        GGUFQuantType.Q4_1: 4,
        GGUFQuantType.Q4_K_S: 4,
        GGUFQuantType.Q4_K_M: 4.5,  # K-quants have overhead
        GGUFQuantType.Q3_K_S: 3,
        GGUFQuantType.Q3_K_M: 3.5,
        GGUFQuantType.Q3_K_L: 3.8,
        GGUFQuantType.Q2_K: 2.5,
        GGUFQuantType.IQ2_XXS: 2.06,
        GGUFQuantType.IQ2_XS: 2.3,
    }

    bits = bits_per_weight.get(quant_type, 4.5)
    size_bytes = model_params * bits / 8
    size_mb = size_bytes / (1024 * 1024)

    # Add overhead for metadata (~1%)
    return size_mb * 1.01


def get_gguf_filename(
    model_name: str,
    quant_type: GGUFQuantType,
) -> str:
    """Generate standard GGUF filename.

    Args:
        model_name: Model name.
        quant_type: Quantization type.

    Returns:
        Standard filename.

    Raises:
        ValueError: If model_name is empty.

    Examples:
        >>> get_gguf_filename("llama-7b", GGUFQuantType.Q4_K_M)
        'llama-7b-q4_k_m.gguf'
        >>> get_gguf_filename("mistral", GGUFQuantType.Q8_0)
        'mistral-q8_0.gguf'

        >>> get_gguf_filename("", GGUFQuantType.Q4_K_M)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    return f"{model_name}-{quant_type.value}.gguf"


def format_gguf_export_result(result: GGUFExportResult) -> str:
    """Format export result for display.

    Args:
        result: Result to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If result is None.

    Examples:
        >>> result = GGUFExportResult(
        ...     output_path="/models/model.gguf",
        ...     original_size_mb=14000,
        ...     exported_size_mb=4000,
        ...     compression_ratio=3.5,
        ...     quant_type=GGUFQuantType.Q4_K_M,
        ... )
        >>> formatted = format_gguf_export_result(result)
        >>> "Output:" in formatted
        True
        >>> "Compression:" in formatted
        True
    """
    validate_not_none(result, "result")

    lines = [
        f"Output: {result.output_path}",
        f"Original: {result.original_size_mb:.1f} MB",
        f"Exported: {result.exported_size_mb:.1f} MB",
        f"Compression: {result.compression_ratio:.2f}x",
        f"Quantization: {result.quant_type.value}",
    ]
    return "\n".join(lines)


def format_gguf_model_info(info: GGUFModelInfo) -> str:
    """Format model info for display.

    Args:
        info: Info to format.

    Returns:
        Formatted string.

    Raises:
        ValueError: If info is None.

    Examples:
        >>> info = GGUFModelInfo(
        ...     path="/models/model.gguf",
        ...     size_mb=4000,
        ...     quant_type=GGUFQuantType.Q4_K_M,
        ... )
        >>> formatted = format_gguf_model_info(info)
        >>> "Path:" in formatted
        True
        >>> "Size:" in formatted
        True
    """
    if info is None:
        msg = "info cannot be None"
        raise ValueError(msg)

    lines = [
        f"Path: {info.path}",
        f"Size: {info.size_mb:.1f} MB",
    ]

    if info.quant_type is not None:
        lines.append(f"Quantization: {info.quant_type.value}")
    if info.architecture is not None:
        lines.append(f"Architecture: {info.architecture.value}")
    if info.context_length is not None:
        lines.append(f"Context length: {info.context_length}")
    if info.vocab_size is not None:
        lines.append(f"Vocab size: {info.vocab_size}")

    return "\n".join(lines)


def list_gguf_quant_types() -> list[str]:
    """List available GGUF quantization types.

    Returns:
        Sorted list of quant type names.

    Examples:
        >>> types = list_gguf_quant_types()
        >>> "q4_k_m" in types
        True
        >>> "q8_0" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(q.value for q in GGUFQuantType)


def validate_gguf_quant_type(quant_type: str) -> bool:
    """Check if GGUF quant type is valid.

    Args:
        quant_type: Type to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_gguf_quant_type("q4_k_m")
        True
        >>> validate_gguf_quant_type("q8_0")
        True
        >>> validate_gguf_quant_type("invalid")
        False
    """
    valid_types = {q.value for q in GGUFQuantType}
    return quant_type in valid_types


def get_gguf_quant_type(name: str) -> GGUFQuantType:
    """Get GGUF quant type enum from string.

    Args:
        name: Type name.

    Returns:
        GGUFQuantType enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_gguf_quant_type("q4_k_m")
        <GGUFQuantType.Q4_K_M: 'q4_k_m'>
        >>> get_gguf_quant_type("q8_0")
        <GGUFQuantType.Q8_0: 'q8_0'>

        >>> get_gguf_quant_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid GGUF quant type: invalid
    """
    for qt in GGUFQuantType:
        if qt.value == name:
            return qt
    msg = f"invalid GGUF quant type: {name}"
    raise ValueError(msg)


def list_gguf_architectures() -> list[str]:
    """List available GGUF architectures.

    Returns:
        Sorted list of architecture names.

    Examples:
        >>> archs = list_gguf_architectures()
        >>> "llama" in archs
        True
        >>> "mistral" in archs
        True
        >>> archs == sorted(archs)
        True
    """
    return sorted(a.value for a in GGUFArchitecture)


def validate_gguf_architecture(architecture: str) -> bool:
    """Check if GGUF architecture is valid.

    Args:
        architecture: Architecture to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_gguf_architecture("llama")
        True
        >>> validate_gguf_architecture("mistral")
        True
        >>> validate_gguf_architecture("invalid")
        False
    """
    valid_archs = {a.value for a in GGUFArchitecture}
    return architecture in valid_archs


def get_gguf_architecture(name: str) -> GGUFArchitecture:
    """Get GGUF architecture enum from string.

    Args:
        name: Architecture name.

    Returns:
        GGUFArchitecture enum value.

    Raises:
        ValueError: If name is not valid.

    Examples:
        >>> get_gguf_architecture("llama")
        <GGUFArchitecture.LLAMA: 'llama'>
        >>> get_gguf_architecture("mistral")
        <GGUFArchitecture.MISTRAL: 'mistral'>

        >>> get_gguf_architecture("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid GGUF architecture: invalid
    """
    for arch in GGUFArchitecture:
        if arch.value == name:
            return arch
    msg = f"invalid GGUF architecture: {name}"
    raise ValueError(msg)


def get_recommended_gguf_quant(
    model_size: str,
    quality_priority: bool = False,
) -> GGUFQuantType:
    """Get recommended GGUF quantization type.

    Args:
        model_size: Model size string (e.g., "7b", "13b", "70b").
        quality_priority: Prioritize quality over size. Defaults to False.

    Returns:
        Recommended GGUFQuantType.

    Raises:
        ValueError: If model_size is not recognized.

    Examples:
        >>> get_recommended_gguf_quant("7b")
        <GGUFQuantType.Q4_K_M: 'q4_k_m'>
        >>> get_recommended_gguf_quant("7b", quality_priority=True)
        <GGUFQuantType.Q5_K_M: 'q5_k_m'>
        >>> get_recommended_gguf_quant("70b")
        <GGUFQuantType.Q4_K_S: 'q4_k_s'>
    """
    model_size = model_size.lower().strip()

    if model_size in ("7b", "7B", "7", "13b", "13B", "13"):
        if quality_priority:
            return GGUFQuantType.Q5_K_M
        return GGUFQuantType.Q4_K_M
    elif model_size in ("70b", "70B", "70"):
        if quality_priority:
            return GGUFQuantType.Q4_K_M
        return GGUFQuantType.Q4_K_S
    else:
        msg = f"unrecognized model size: {model_size}"
        raise ValueError(msg)


def get_gguf_config_dict(config: GGUFConfig) -> dict[str, Any]:
    """Convert GGUFConfig to dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Dictionary representation.

    Raises:
        ValueError: If config is None.

    Examples:
        >>> config = create_gguf_config(quant_type="q4_k_m")
        >>> d = get_gguf_config_dict(config)
        >>> d["quant_type"]
        'q4_k_m'
        >>> d["concurrency"]
        4
    """
    validate_not_none(config, "config")

    result: dict[str, Any] = {
        "quant_type": config.quant_type.value,
        "vocab_only": config.vocab_only,
        "outtype": config.outtype,
        "concurrency": config.concurrency,
    }

    if config.architecture is not None:
        result["architecture"] = config.architecture.value

    return result


def get_gguf_metadata_dict(metadata: GGUFMetadata) -> dict[str, str]:
    """Convert GGUFMetadata to dictionary.

    Args:
        metadata: Metadata to convert.

    Returns:
        Dictionary with non-empty values.

    Raises:
        ValueError: If metadata is None.

    Examples:
        >>> meta = create_gguf_metadata(name="my-model", author="me")
        >>> d = get_gguf_metadata_dict(meta)
        >>> d["name"]
        'my-model'
        >>> "description" not in d  # Empty values excluded
        True
    """
    if metadata is None:
        msg = "metadata cannot be None"
        raise ValueError(msg)

    fields = ("name", "author", "version", "description", "license", "url")
    return {
        field: getattr(metadata, field) for field in fields if getattr(metadata, field)
    }
