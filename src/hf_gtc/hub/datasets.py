"""Dataset management utilities for HuggingFace Hub.

This module provides functions for managing, validating, and working with
datasets on the HuggingFace Hub, including configuration, download, and upload
utilities.

Examples:
    >>> from hf_gtc.hub.datasets import create_dataset_config, DatasetFormat
    >>> config = create_dataset_config("squad")
    >>> config.name
    'squad'
    >>> DatasetFormat.PARQUET.value
    'parquet'
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DatasetFormat(Enum):
    """Supported dataset file formats.

    Attributes:
        PARQUET: Apache Parquet format (columnar, efficient).
        JSON: JSON Lines format.
        CSV: Comma-separated values format.
        ARROW: Apache Arrow format (in-memory).
        WEBDATASET: WebDataset format (tar archives).

    Examples:
        >>> DatasetFormat.PARQUET.value
        'parquet'
        >>> DatasetFormat.JSON.value
        'json'
        >>> DatasetFormat.WEBDATASET.value
        'webdataset'
    """

    PARQUET = "parquet"
    JSON = "json"
    CSV = "csv"
    ARROW = "arrow"
    WEBDATASET = "webdataset"


VALID_FORMATS = frozenset(f.value for f in DatasetFormat)


class SplitType(Enum):
    """Standard dataset split types.

    Attributes:
        TRAIN: Training split.
        VALIDATION: Validation/development split.
        TEST: Test/evaluation split.
        ALL: All splits combined.

    Examples:
        >>> SplitType.TRAIN.value
        'train'
        >>> SplitType.VALIDATION.value
        'validation'
        >>> SplitType.ALL.value
        'all'
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


VALID_SPLITS = frozenset(s.value for s in SplitType)


class StreamingMode(Enum):
    """Dataset streaming modes.

    Attributes:
        DISABLED: No streaming, load entire dataset.
        BASIC: Basic streaming without shuffling.
        SHUFFLED: Streaming with shuffle buffer.

    Examples:
        >>> StreamingMode.DISABLED.value
        'disabled'
        >>> StreamingMode.BASIC.value
        'basic'
        >>> StreamingMode.SHUFFLED.value
        'shuffled'
    """

    DISABLED = "disabled"
    BASIC = "basic"
    SHUFFLED = "shuffled"


VALID_STREAMING_MODES = frozenset(m.value for m in StreamingMode)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for loading a dataset.

    Attributes:
        name: Dataset name or path (e.g., "squad", "user/dataset").
        subset: Dataset subset/configuration name.
        split: Split to load (train, validation, test, all).
        streaming: Streaming mode for the dataset.
        trust_remote_code: Whether to trust remote code in dataset scripts.

    Examples:
        >>> config = DatasetConfig(
        ...     name="squad",
        ...     subset=None,
        ...     split=SplitType.TRAIN,
        ...     streaming=StreamingMode.DISABLED,
        ...     trust_remote_code=False,
        ... )
        >>> config.name
        'squad'
        >>> config.split
        <SplitType.TRAIN: 'train'>
    """

    name: str
    subset: str | None
    split: SplitType
    streaming: StreamingMode
    trust_remote_code: bool


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    """Configuration for downloading datasets.

    Attributes:
        cache_dir: Directory to cache downloaded files.
        force_download: Force re-download even if cached.
        resume_download: Resume interrupted downloads.
        max_retries: Maximum number of download retries.

    Examples:
        >>> config = DownloadConfig(
        ...     cache_dir="/tmp/cache",
        ...     force_download=False,
        ...     resume_download=True,
        ...     max_retries=3,
        ... )
        >>> config.cache_dir
        '/tmp/cache'
        >>> config.max_retries
        3
    """

    cache_dir: str | None
    force_download: bool
    resume_download: bool
    max_retries: int


@dataclass(frozen=True, slots=True)
class UploadConfig:
    """Configuration for uploading datasets to Hub.

    Attributes:
        repo_id: Repository ID for upload (e.g., "user/dataset").
        private: Whether the dataset should be private.
        commit_message: Commit message for the upload.
        create_pr: Whether to create a pull request instead of direct push.

    Examples:
        >>> config = UploadConfig(
        ...     repo_id="user/my-dataset",
        ...     private=True,
        ...     commit_message="Add training data",
        ...     create_pr=False,
        ... )
        >>> config.repo_id
        'user/my-dataset'
        >>> config.private
        True
    """

    repo_id: str
    private: bool
    commit_message: str
    create_pr: bool


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Statistics about a dataset.

    Attributes:
        num_rows: Total number of rows in the dataset.
        num_columns: Number of columns/features.
        size_bytes: Total size in bytes.
        features: Dictionary of feature names to types.

    Examples:
        >>> stats = DatasetStats(
        ...     num_rows=10000,
        ...     num_columns=5,
        ...     size_bytes=1048576,
        ...     features={"text": "string", "label": "int64"},
        ... )
        >>> stats.num_rows
        10000
        >>> stats.size_bytes
        1048576
    """

    num_rows: int
    num_columns: int
    size_bytes: int
    features: dict[str, str]


def validate_dataset_config(config: DatasetConfig) -> None:
    """Validate a dataset configuration.

    Args:
        config: Dataset configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = DatasetConfig(
        ...     "squad", None, SplitType.TRAIN, StreamingMode.DISABLED, False
        ... )
        >>> validate_dataset_config(config)  # No error

        >>> bad = DatasetConfig(
        ...     "", None, SplitType.TRAIN, StreamingMode.DISABLED, False
        ... )
        >>> validate_dataset_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty
    """
    if not config.name:
        msg = "name cannot be empty"
        raise ValueError(msg)


def validate_download_config(config: DownloadConfig) -> None:
    """Validate a download configuration.

    Args:
        config: Download configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = DownloadConfig("/tmp", False, True, 3)
        >>> validate_download_config(config)  # No error

        >>> bad = DownloadConfig(None, False, True, -1)
        >>> validate_download_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: max_retries must be non-negative
    """
    if config.max_retries < 0:
        msg = f"max_retries must be non-negative, got {config.max_retries}"
        raise ValueError(msg)


def validate_upload_config(config: UploadConfig) -> None:
    """Validate an upload configuration.

    Args:
        config: Upload configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = UploadConfig("user/dataset", False, "Initial commit", False)
        >>> validate_upload_config(config)  # No error

        >>> bad = UploadConfig("", False, "msg", False)
        >>> validate_upload_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: repo_id cannot be empty

        >>> bad = UploadConfig("user/ds", False, "", False)
        >>> validate_upload_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: commit_message cannot be empty
    """
    if not config.repo_id:
        msg = "repo_id cannot be empty"
        raise ValueError(msg)

    if not config.commit_message:
        msg = "commit_message cannot be empty"
        raise ValueError(msg)


def validate_dataset_stats(stats: DatasetStats) -> None:
    """Validate dataset statistics.

    Args:
        stats: Dataset statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = DatasetStats(1000, 5, 10240, {"text": "string"})
        >>> validate_dataset_stats(stats)  # No error

        >>> bad = DatasetStats(-1, 5, 10240, {})
        >>> validate_dataset_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_rows must be non-negative

        >>> bad = DatasetStats(100, -1, 10240, {})
        >>> validate_dataset_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_columns must be non-negative

        >>> bad = DatasetStats(100, 5, -1, {})
        >>> validate_dataset_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: size_bytes must be non-negative
    """
    if stats.num_rows < 0:
        msg = f"num_rows must be non-negative, got {stats.num_rows}"
        raise ValueError(msg)

    if stats.num_columns < 0:
        msg = f"num_columns must be non-negative, got {stats.num_columns}"
        raise ValueError(msg)

    if stats.size_bytes < 0:
        msg = f"size_bytes must be non-negative, got {stats.size_bytes}"
        raise ValueError(msg)


def create_dataset_config(
    name: str,
    subset: str | None = None,
    split: str = "train",
    streaming: str = "disabled",
    trust_remote_code: bool = False,
) -> DatasetConfig:
    """Create a dataset configuration.

    Args:
        name: Dataset name or path.
        subset: Dataset subset/configuration. Defaults to None.
        split: Split to load. Defaults to "train".
        streaming: Streaming mode. Defaults to "disabled".
        trust_remote_code: Trust remote code. Defaults to False.

    Returns:
        DatasetConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_dataset_config("squad")
        >>> config.name
        'squad'
        >>> config.split
        <SplitType.TRAIN: 'train'>

        >>> config = create_dataset_config(
        ...     "glue",
        ...     subset="mrpc",
        ...     split="validation",
        ...     streaming="basic",
        ... )
        >>> config.subset
        'mrpc'
        >>> config.streaming
        <StreamingMode.BASIC: 'basic'>

        >>> create_dataset_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: name cannot be empty

        >>> create_dataset_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "ds", split="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: split must be one of
    """
    if split not in VALID_SPLITS:
        msg = f"split must be one of {VALID_SPLITS}, got '{split}'"
        raise ValueError(msg)

    if streaming not in VALID_STREAMING_MODES:
        msg = f"streaming must be one of {VALID_STREAMING_MODES}, got '{streaming}'"
        raise ValueError(msg)

    config = DatasetConfig(
        name=name,
        subset=subset,
        split=SplitType(split),
        streaming=StreamingMode(streaming),
        trust_remote_code=trust_remote_code,
    )
    validate_dataset_config(config)
    return config


def create_download_config(
    cache_dir: str | None = None,
    force_download: bool = False,
    resume_download: bool = True,
    max_retries: int = 3,
) -> DownloadConfig:
    """Create a download configuration.

    Args:
        cache_dir: Cache directory. Defaults to None (system default).
        force_download: Force re-download. Defaults to False.
        resume_download: Resume interrupted downloads. Defaults to True.
        max_retries: Maximum retries. Defaults to 3.

    Returns:
        DownloadConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_download_config()
        >>> config.resume_download
        True
        >>> config.max_retries
        3

        >>> config = create_download_config(
        ...     cache_dir="/data/cache",
        ...     force_download=True,
        ... )
        >>> config.cache_dir
        '/data/cache'

        >>> create_download_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     max_retries=-1
        ... )
        Traceback (most recent call last):
        ValueError: max_retries must be non-negative
    """
    config = DownloadConfig(
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        max_retries=max_retries,
    )
    validate_download_config(config)
    return config


def create_upload_config(
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload dataset",
    create_pr: bool = False,
) -> UploadConfig:
    """Create an upload configuration.

    Args:
        repo_id: Repository ID for upload.
        private: Whether dataset is private. Defaults to False.
        commit_message: Commit message. Defaults to "Upload dataset".
        create_pr: Create PR instead of direct push. Defaults to False.

    Returns:
        UploadConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_upload_config("user/my-dataset")
        >>> config.repo_id
        'user/my-dataset'
        >>> config.private
        False

        >>> config = create_upload_config(
        ...     "org/dataset",
        ...     private=True,
        ...     commit_message="Add v2 data",
        ...     create_pr=True,
        ... )
        >>> config.create_pr
        True

        >>> create_upload_config("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: repo_id cannot be empty
    """
    config = UploadConfig(
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        create_pr=create_pr,
    )
    validate_upload_config(config)
    return config


def create_dataset_stats(
    num_rows: int = 0,
    num_columns: int = 0,
    size_bytes: int = 0,
    features: dict[str, str] | None = None,
) -> DatasetStats:
    """Create dataset statistics.

    Args:
        num_rows: Number of rows. Defaults to 0.
        num_columns: Number of columns. Defaults to 0.
        size_bytes: Size in bytes. Defaults to 0.
        features: Feature names to types. Defaults to empty dict.

    Returns:
        DatasetStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_dataset_stats(num_rows=1000)
        >>> stats.num_rows
        1000

        >>> stats = create_dataset_stats(
        ...     num_rows=5000,
        ...     num_columns=3,
        ...     size_bytes=1048576,
        ...     features={"text": "string", "label": "int64"},
        ... )
        >>> stats.features["text"]
        'string'

        >>> create_dataset_stats(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     num_rows=-1
        ... )
        Traceback (most recent call last):
        ValueError: num_rows must be non-negative
    """
    if features is None:
        features = {}

    stats = DatasetStats(
        num_rows=num_rows,
        num_columns=num_columns,
        size_bytes=size_bytes,
        features=features,
    )
    validate_dataset_stats(stats)
    return stats


def list_formats() -> list[str]:
    """List all available dataset formats.

    Returns:
        Sorted list of format names.

    Examples:
        >>> formats = list_formats()
        >>> "parquet" in formats
        True
        >>> "json" in formats
        True
        >>> formats == sorted(formats)
        True
    """
    return sorted(VALID_FORMATS)


def list_splits() -> list[str]:
    """List all available split types.

    Returns:
        Sorted list of split type names.

    Examples:
        >>> splits = list_splits()
        >>> "train" in splits
        True
        >>> "validation" in splits
        True
        >>> splits == sorted(splits)
        True
    """
    return sorted(VALID_SPLITS)


def list_streaming_modes() -> list[str]:
    """List all available streaming modes.

    Returns:
        Sorted list of streaming mode names.

    Examples:
        >>> modes = list_streaming_modes()
        >>> "disabled" in modes
        True
        >>> "basic" in modes
        True
        >>> modes == sorted(modes)
        True
    """
    return sorted(VALID_STREAMING_MODES)


def get_format(name: str) -> DatasetFormat:
    """Get dataset format from name.

    Args:
        name: Format name.

    Returns:
        DatasetFormat enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_format("parquet")
        <DatasetFormat.PARQUET: 'parquet'>

        >>> get_format("json")
        <DatasetFormat.JSON: 'json'>

        >>> get_format("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: format must be one of
    """
    if name not in VALID_FORMATS:
        msg = f"format must be one of {VALID_FORMATS}, got '{name}'"
        raise ValueError(msg)
    return DatasetFormat(name)


def get_split(name: str) -> SplitType:
    """Get split type from name.

    Args:
        name: Split name.

    Returns:
        SplitType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_split("train")
        <SplitType.TRAIN: 'train'>

        >>> get_split("validation")
        <SplitType.VALIDATION: 'validation'>

        >>> get_split("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: split must be one of
    """
    if name not in VALID_SPLITS:
        msg = f"split must be one of {VALID_SPLITS}, got '{name}'"
        raise ValueError(msg)
    return SplitType(name)


def get_streaming_mode(name: str) -> StreamingMode:
    """Get streaming mode from name.

    Args:
        name: Streaming mode name.

    Returns:
        StreamingMode enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_streaming_mode("disabled")
        <StreamingMode.DISABLED: 'disabled'>

        >>> get_streaming_mode("basic")
        <StreamingMode.BASIC: 'basic'>

        >>> get_streaming_mode("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: streaming_mode must be one of
    """
    if name not in VALID_STREAMING_MODES:
        msg = f"streaming_mode must be one of {VALID_STREAMING_MODES}, got '{name}'"
        raise ValueError(msg)
    return StreamingMode(name)


def estimate_download_size(
    num_rows: int,
    avg_row_size_bytes: int = 1024,
    compression_ratio: float = 0.3,
) -> int:
    """Estimate download size for a dataset.

    Args:
        num_rows: Number of rows in the dataset.
        avg_row_size_bytes: Average size per row in bytes. Defaults to 1024.
        compression_ratio: Compression ratio (0-1). Defaults to 0.3.

    Returns:
        Estimated download size in bytes.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> estimate_download_size(1000)
        307200

        >>> estimate_download_size(1000, avg_row_size_bytes=2048)
        614400

        >>> estimate_download_size(1000, compression_ratio=0.5)
        512000

        >>> estimate_download_size(-1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: num_rows must be non-negative

        >>> estimate_download_size(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     100, compression_ratio=1.5
        ... )
        Traceback (most recent call last):
        ValueError: compression_ratio must be between 0 and 1
    """
    if num_rows < 0:
        msg = f"num_rows must be non-negative, got {num_rows}"
        raise ValueError(msg)

    if avg_row_size_bytes < 0:
        msg = f"avg_row_size_bytes must be non-negative, got {avg_row_size_bytes}"
        raise ValueError(msg)

    if not 0 <= compression_ratio <= 1:
        msg = f"compression_ratio must be between 0 and 1, got {compression_ratio}"
        raise ValueError(msg)

    raw_size = num_rows * avg_row_size_bytes
    return int(raw_size * compression_ratio)


def calculate_dataset_hash(
    data: bytes | str,
    algorithm: str = "sha256",
) -> str:
    """Calculate hash of dataset content.

    Args:
        data: Data to hash (bytes or string).
        algorithm: Hash algorithm (sha256, md5, sha1). Defaults to "sha256".

    Returns:
        Hex digest of the hash.

    Raises:
        ValueError: If algorithm is invalid.

    Examples:
        >>> calculate_dataset_hash(b"test data")
        '916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9'

        >>> calculate_dataset_hash("test data")
        '916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9'

        >>> calculate_dataset_hash(b"test", algorithm="md5")
        '098f6bcd4621d373cade4e832627b4f6'

        >>> calculate_dataset_hash(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     b"data", algorithm="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: algorithm must be one of
    """
    valid_algorithms = {"sha256", "md5", "sha1"}
    if algorithm not in valid_algorithms:
        msg = f"algorithm must be one of {valid_algorithms}, got '{algorithm}'"
        raise ValueError(msg)

    if isinstance(data, str):
        data = data.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def validate_dataset_format(
    file_path: str,
    expected_format: str,
) -> bool:
    """Validate that a file matches the expected dataset format.

    Args:
        file_path: Path to the dataset file.
        expected_format: Expected format (parquet, json, csv, arrow, webdataset).

    Returns:
        True if format matches, False otherwise.

    Raises:
        ValueError: If expected_format is invalid.

    Examples:
        >>> validate_dataset_format("data.parquet", "parquet")
        True

        >>> validate_dataset_format("data.json", "parquet")
        False

        >>> validate_dataset_format("data.csv", "csv")
        True

        >>> validate_dataset_format("data.arrow", "arrow")
        True

        >>> validate_dataset_format("archive.tar", "webdataset")
        True

        >>> validate_dataset_format(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "data.txt", "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: expected_format must be one of
    """
    if expected_format not in VALID_FORMATS:
        msg = f"expected_format must be one of {VALID_FORMATS}, got '{expected_format}'"
        raise ValueError(msg)

    # Map formats to expected file extensions
    format_extensions: dict[str, tuple[str, ...]] = {
        "parquet": (".parquet", ".pq"),
        "json": (".json", ".jsonl"),
        "csv": (".csv", ".tsv"),
        "arrow": (".arrow", ".feather"),
        "webdataset": (".tar", ".tar.gz", ".tgz"),
    }

    file_lower = file_path.lower()
    expected_exts = format_extensions[expected_format]

    return any(file_lower.endswith(ext) for ext in expected_exts)


def compare_dataset_schemas(
    schema_a: dict[str, str],
    schema_b: dict[str, str],
) -> dict[str, dict[str, str | None]]:
    """Compare two dataset schemas.

    Args:
        schema_a: First schema (feature name to type).
        schema_b: Second schema (feature name to type).

    Returns:
        Dictionary with comparison results for each field.

    Examples:
        >>> schema_a = {"text": "string", "label": "int64"}
        >>> schema_b = {"text": "string", "score": "float32"}
        >>> result = compare_dataset_schemas(schema_a, schema_b)
        >>> result["text"]["status"]
        'match'
        >>> result["label"]["status"]
        'only_in_a'
        >>> result["score"]["status"]
        'only_in_b'

        >>> # Type mismatch detection
        >>> schema_a = {"value": "int64"}
        >>> schema_b = {"value": "float32"}
        >>> result = compare_dataset_schemas(schema_a, schema_b)
        >>> result["value"]["status"]
        'type_mismatch'
    """
    all_fields = set(schema_a.keys()) | set(schema_b.keys())

    result: dict[str, dict[str, str | None]] = {}
    for field in all_fields:
        type_a = schema_a.get(field)
        type_b = schema_b.get(field)

        if type_a is None:
            result[field] = {
                "status": "only_in_b",
                "type_a": None,
                "type_b": type_b,
            }
        elif type_b is None:
            result[field] = {
                "status": "only_in_a",
                "type_a": type_a,
                "type_b": None,
            }
        elif type_a == type_b:
            result[field] = {
                "status": "match",
                "type_a": type_a,
                "type_b": type_b,
            }
        else:
            result[field] = {
                "status": "type_mismatch",
                "type_a": type_a,
                "type_b": type_b,
            }

    return result


def format_dataset_stats(stats: DatasetStats) -> str:
    """Format dataset statistics as a human-readable string.

    Args:
        stats: Dataset statistics to format.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = create_dataset_stats(
        ...     num_rows=10000,
        ...     num_columns=5,
        ...     size_bytes=1048576,
        ...     features={"text": "string", "label": "int64"},
        ... )
        >>> output = format_dataset_stats(stats)
        >>> "10,000 rows" in output
        True
        >>> "1.00 MB" in output
        True

        >>> stats = create_dataset_stats()
        >>> output = format_dataset_stats(stats)
        >>> "0 rows" in output
        True
    """
    # Format size with appropriate unit
    size = stats.size_bytes
    if size >= 1024 * 1024 * 1024:
        size_str = f"{size / (1024 * 1024 * 1024):.2f} GB"
    elif size >= 1024 * 1024:
        size_str = f"{size / (1024 * 1024):.2f} MB"
    elif size >= 1024:
        size_str = f"{size / 1024:.2f} KB"
    else:
        size_str = f"{size} bytes"

    lines = [
        "Dataset Statistics:",
        f"  Rows: {stats.num_rows:,} rows",
        f"  Columns: {stats.num_columns}",
        f"  Size: {size_str}",
    ]

    if stats.features:
        lines.append("  Features:")
        for name, dtype in sorted(stats.features.items()):
            lines.append(f"    {name}: {dtype}")

    return "\n".join(lines)


def get_recommended_dataset_config(
    dataset_size: str = "small",
    use_case: str = "training",
) -> DatasetConfig:
    """Get recommended dataset configuration based on requirements.

    Args:
        dataset_size: Size category ("small", "medium", "large").
            Defaults to "small".
        use_case: Use case ("training", "evaluation", "inference").
            Defaults to "training".

    Returns:
        Recommended DatasetConfig.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = get_recommended_dataset_config()
        >>> config.streaming
        <StreamingMode.DISABLED: 'disabled'>

        >>> config = get_recommended_dataset_config(dataset_size="large")
        >>> config.streaming
        <StreamingMode.SHUFFLED: 'shuffled'>

        >>> config = get_recommended_dataset_config(use_case="evaluation")
        >>> config.split
        <SplitType.VALIDATION: 'validation'>

        >>> config = get_recommended_dataset_config(use_case="inference")
        >>> config.split
        <SplitType.TEST: 'test'>

        >>> get_recommended_dataset_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     dataset_size="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: dataset_size must be one of

        >>> get_recommended_dataset_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     use_case="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: use_case must be one of
    """
    valid_sizes = {"small", "medium", "large"}
    if dataset_size not in valid_sizes:
        msg = f"dataset_size must be one of {valid_sizes}, got '{dataset_size}'"
        raise ValueError(msg)

    valid_use_cases = {"training", "evaluation", "inference"}
    if use_case not in valid_use_cases:
        msg = f"use_case must be one of {valid_use_cases}, got '{use_case}'"
        raise ValueError(msg)

    # Determine streaming mode based on size
    if dataset_size == "large":
        streaming = "shuffled"
    elif dataset_size == "medium":
        streaming = "basic"
    else:
        streaming = "disabled"

    # Determine split based on use case
    if use_case == "training":
        split = "train"
    elif use_case == "evaluation":
        split = "validation"
    else:
        split = "test"

    return create_dataset_config(
        name="__placeholder__",
        split=split,
        streaming=streaming,
        trust_remote_code=False,
    )
