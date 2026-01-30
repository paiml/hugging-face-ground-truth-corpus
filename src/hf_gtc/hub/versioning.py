"""Model versioning and tracking utilities for HuggingFace Hub.

This module provides dataclasses and functions for managing model versions,
tracking changes between versions, and computing version statistics.

Examples:
    >>> from hf_gtc.hub.versioning import create_version_info, parse_version
    >>> version = create_version_info(1, 0, 0)
    >>> version.major
    1
    >>> parsed = parse_version("2.1.0-alpha+build.123")
    >>> parsed.major, parsed.minor, parsed.patch
    (2, 1, 0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class VersionScheme(Enum):
    """Version numbering schemes.

    Attributes:
        SEMANTIC: Semantic versioning (major.minor.patch).
        DATE_BASED: Date-based versioning (YYYY.MM.DD).
        INCREMENTAL: Simple incremental versioning (1, 2, 3...).
        GIT_HASH: Git commit hash-based versioning.

    Examples:
        >>> VersionScheme.SEMANTIC.value
        'semantic'
        >>> VersionScheme.DATE_BASED.value
        'date_based'
        >>> VersionScheme.INCREMENTAL.value
        'incremental'
        >>> VersionScheme.GIT_HASH.value
        'git_hash'
    """

    SEMANTIC = "semantic"
    DATE_BASED = "date_based"
    INCREMENTAL = "incremental"
    GIT_HASH = "git_hash"


VALID_VERSION_SCHEMES = frozenset(s.value for s in VersionScheme)


class ChangeType(Enum):
    """Types of changes in a version.

    Attributes:
        MAJOR: Breaking changes that require major version bump.
        MINOR: New features that are backwards compatible.
        PATCH: Bug fixes and minor improvements.
        PRE_RELEASE: Pre-release versions (alpha, beta, rc).

    Examples:
        >>> ChangeType.MAJOR.value
        'major'
        >>> ChangeType.MINOR.value
        'minor'
        >>> ChangeType.PATCH.value
        'patch'
        >>> ChangeType.PRE_RELEASE.value
        'pre_release'
    """

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRE_RELEASE = "pre_release"


VALID_CHANGE_TYPES = frozenset(c.value for c in ChangeType)


class DiffType(Enum):
    """Types of differences to track between model versions.

    Attributes:
        CONFIG: Configuration file changes.
        WEIGHTS: Model weight/parameter changes.
        TOKENIZER: Tokenizer changes.
        ALL: All types of changes.

    Examples:
        >>> DiffType.CONFIG.value
        'config'
        >>> DiffType.WEIGHTS.value
        'weights'
        >>> DiffType.TOKENIZER.value
        'tokenizer'
        >>> DiffType.ALL.value
        'all'
    """

    CONFIG = "config"
    WEIGHTS = "weights"
    TOKENIZER = "tokenizer"
    ALL = "all"


VALID_DIFF_TYPES = frozenset(d.value for d in DiffType)


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """Semantic version information.

    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        pre_release: Pre-release identifier (e.g., "alpha", "beta.1").
        build_metadata: Build metadata (e.g., "build.123").

    Examples:
        >>> version = VersionInfo(
        ...     major=1,
        ...     minor=2,
        ...     patch=3,
        ...     pre_release="alpha",
        ...     build_metadata="build.456",
        ... )
        >>> version.major
        1
        >>> version.pre_release
        'alpha'
    """

    major: int
    minor: int
    patch: int
    pre_release: str | None
    build_metadata: str | None


@dataclass(frozen=True, slots=True)
class VersionConfig:
    """Configuration for version management.

    Attributes:
        scheme: Version numbering scheme to use.
        auto_increment: Automatically increment version on changes.
        track_changes: Track what changed between versions.
        require_changelog: Require changelog entries for new versions.

    Examples:
        >>> config = VersionConfig(
        ...     scheme=VersionScheme.SEMANTIC,
        ...     auto_increment=True,
        ...     track_changes=True,
        ...     require_changelog=False,
        ... )
        >>> config.scheme
        <VersionScheme.SEMANTIC: 'semantic'>
        >>> config.auto_increment
        True
    """

    scheme: VersionScheme
    auto_increment: bool
    track_changes: bool
    require_changelog: bool


@dataclass(frozen=True, slots=True)
class ModelDiff:
    """Differences between two model versions.

    Attributes:
        changed_files: Tuple of file paths that changed.
        added_params: Number of parameters added.
        removed_params: Number of parameters removed.
        config_changes: Dictionary of config key to (old_value, new_value).

    Examples:
        >>> diff = ModelDiff(
        ...     changed_files=("config.json", "model.safetensors"),
        ...     added_params=1000,
        ...     removed_params=0,
        ...     config_changes={"hidden_size": (768, 1024)},
        ... )
        >>> diff.added_params
        1000
        >>> "config.json" in diff.changed_files
        True
    """

    changed_files: tuple[str, ...]
    added_params: int
    removed_params: int
    config_changes: dict[str, tuple[object, object]]


@dataclass(frozen=True, slots=True)
class VersionStats:
    """Statistics for model versioning.

    Attributes:
        total_versions: Total number of versions.
        latest_version: Latest version string.
        release_frequency_days: Average days between releases.

    Examples:
        >>> stats = VersionStats(
        ...     total_versions=10,
        ...     latest_version="2.1.0",
        ...     release_frequency_days=14.5,
        ... )
        >>> stats.total_versions
        10
        >>> stats.latest_version
        '2.1.0'
    """

    total_versions: int
    latest_version: str
    release_frequency_days: float | None


def validate_version_info(info: VersionInfo) -> None:
    """Validate version information.

    Args:
        info: Version information to validate.

    Raises:
        ValueError: If version information is invalid.

    Examples:
        >>> info = VersionInfo(1, 2, 3, None, None)
        >>> validate_version_info(info)  # No error

        >>> bad = VersionInfo(-1, 0, 0, None, None)
        >>> validate_version_info(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: major must be non-negative
    """
    if info.major < 0:
        msg = f"major must be non-negative, got {info.major}"
        raise ValueError(msg)

    if info.minor < 0:
        msg = f"minor must be non-negative, got {info.minor}"
        raise ValueError(msg)

    if info.patch < 0:
        msg = f"patch must be non-negative, got {info.patch}"
        raise ValueError(msg)

    if info.pre_release is not None and not info.pre_release:
        msg = "pre_release cannot be an empty string"
        raise ValueError(msg)

    if info.build_metadata is not None and not info.build_metadata:
        msg = "build_metadata cannot be an empty string"
        raise ValueError(msg)


def validate_version_config(config: VersionConfig) -> None:
    """Validate version configuration.

    Args:
        config: Version configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = VersionConfig(
        ...     VersionScheme.SEMANTIC, True, True, False
        ... )
        >>> validate_version_config(config)  # No error
    """
    # Currently all valid enum combinations are acceptable
    # This function exists for future validation rules
    pass


def validate_model_diff(diff: ModelDiff) -> None:
    """Validate model diff.

    Args:
        diff: Model diff to validate.

    Raises:
        ValueError: If diff is invalid.

    Examples:
        >>> diff = ModelDiff(("config.json",), 100, 0, {})
        >>> validate_model_diff(diff)  # No error

        >>> bad = ModelDiff(("",), 0, 0, {})
        >>> validate_model_diff(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: changed_files cannot contain empty strings
    """
    for f in diff.changed_files:
        if not f:
            msg = "changed_files cannot contain empty strings"
            raise ValueError(msg)

    if diff.added_params < 0:
        msg = f"added_params must be non-negative, got {diff.added_params}"
        raise ValueError(msg)

    if diff.removed_params < 0:
        msg = f"removed_params must be non-negative, got {diff.removed_params}"
        raise ValueError(msg)


def validate_version_stats(stats: VersionStats) -> None:
    """Validate version statistics.

    Args:
        stats: Version statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = VersionStats(10, "1.0.0", 7.0)
        >>> validate_version_stats(stats)  # No error

        >>> bad = VersionStats(-1, "1.0.0", 7.0)
        >>> validate_version_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_versions must be non-negative
    """
    if stats.total_versions < 0:
        msg = f"total_versions must be non-negative, got {stats.total_versions}"
        raise ValueError(msg)

    if not stats.latest_version:
        msg = "latest_version cannot be empty"
        raise ValueError(msg)

    if stats.release_frequency_days is not None and stats.release_frequency_days < 0:
        msg = (
            f"release_frequency_days must be non-negative, "
            f"got {stats.release_frequency_days}"
        )
        raise ValueError(msg)


def create_version_info(
    major: int = 0,
    minor: int = 0,
    patch: int = 0,
    pre_release: str | None = None,
    build_metadata: str | None = None,
) -> VersionInfo:
    """Create version information.

    Args:
        major: Major version number. Defaults to 0.
        minor: Minor version number. Defaults to 0.
        patch: Patch version number. Defaults to 0.
        pre_release: Pre-release identifier. Defaults to None.
        build_metadata: Build metadata. Defaults to None.

    Returns:
        VersionInfo with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> info = create_version_info(1, 2, 3)
        >>> info.major, info.minor, info.patch
        (1, 2, 3)

        >>> info = create_version_info(1, 0, 0, pre_release="alpha")
        >>> info.pre_release
        'alpha'

        >>> create_version_info(-1, 0, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: major must be non-negative
    """
    info = VersionInfo(
        major=major,
        minor=minor,
        patch=patch,
        pre_release=pre_release,
        build_metadata=build_metadata,
    )
    validate_version_info(info)
    return info


def create_version_config(
    scheme: str = "semantic",
    auto_increment: bool = True,
    track_changes: bool = True,
    require_changelog: bool = False,
) -> VersionConfig:
    """Create version configuration.

    Args:
        scheme: Version numbering scheme. Defaults to "semantic".
        auto_increment: Auto-increment on changes. Defaults to True.
        track_changes: Track changes between versions. Defaults to True.
        require_changelog: Require changelog entries. Defaults to False.

    Returns:
        VersionConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_version_config()
        >>> config.scheme
        <VersionScheme.SEMANTIC: 'semantic'>

        >>> config = create_version_config(scheme="date_based")
        >>> config.scheme
        <VersionScheme.DATE_BASED: 'date_based'>

        >>> create_version_config(scheme="invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: scheme must be one of
    """
    if scheme not in VALID_VERSION_SCHEMES:
        msg = f"scheme must be one of {VALID_VERSION_SCHEMES}, got '{scheme}'"
        raise ValueError(msg)

    config = VersionConfig(
        scheme=VersionScheme(scheme),
        auto_increment=auto_increment,
        track_changes=track_changes,
        require_changelog=require_changelog,
    )
    validate_version_config(config)
    return config


def create_model_diff(
    changed_files: tuple[str, ...] | list[str] | None = None,
    added_params: int = 0,
    removed_params: int = 0,
    config_changes: dict[str, tuple[object, object]] | None = None,
) -> ModelDiff:
    """Create model diff.

    Args:
        changed_files: Files that changed. Defaults to empty tuple.
        added_params: Number of parameters added. Defaults to 0.
        removed_params: Number of parameters removed. Defaults to 0.
        config_changes: Config changes dict. Defaults to empty dict.

    Returns:
        ModelDiff with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> diff = create_model_diff(
        ...     changed_files=["config.json"],
        ...     added_params=1000,
        ... )
        >>> diff.added_params
        1000

        >>> diff = create_model_diff()
        >>> diff.changed_files
        ()

        >>> create_model_diff(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     added_params=-1
        ... )
        Traceback (most recent call last):
        ValueError: added_params must be non-negative
    """
    if changed_files is None:
        files_tuple: tuple[str, ...] = ()
    elif isinstance(changed_files, list):
        files_tuple = tuple(changed_files)
    else:
        files_tuple = changed_files

    if config_changes is None:
        config_changes = {}

    diff = ModelDiff(
        changed_files=files_tuple,
        added_params=added_params,
        removed_params=removed_params,
        config_changes=config_changes,
    )
    validate_model_diff(diff)
    return diff


def create_version_stats(
    total_versions: int = 0,
    latest_version: str = "0.0.0",
    release_frequency_days: float | None = None,
) -> VersionStats:
    """Create version statistics.

    Args:
        total_versions: Total number of versions. Defaults to 0.
        latest_version: Latest version string. Defaults to "0.0.0".
        release_frequency_days: Average days between releases. Defaults to None.

    Returns:
        VersionStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_version_stats(total_versions=5)
        >>> stats.total_versions
        5

        >>> stats = create_version_stats(
        ...     total_versions=10,
        ...     latest_version="2.0.0",
        ...     release_frequency_days=7.0,
        ... )
        >>> stats.release_frequency_days
        7.0

        >>> create_version_stats(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     total_versions=-1
        ... )
        Traceback (most recent call last):
        ValueError: total_versions must be non-negative
    """
    stats = VersionStats(
        total_versions=total_versions,
        latest_version=latest_version,
        release_frequency_days=release_frequency_days,
    )
    validate_version_stats(stats)
    return stats


def list_version_schemes() -> list[str]:
    """List all valid version schemes.

    Returns:
        Sorted list of version scheme names.

    Examples:
        >>> schemes = list_version_schemes()
        >>> "semantic" in schemes
        True
        >>> "date_based" in schemes
        True
        >>> schemes == sorted(schemes)
        True
    """
    return sorted(VALID_VERSION_SCHEMES)


def list_change_types() -> list[str]:
    """List all valid change types.

    Returns:
        Sorted list of change type names.

    Examples:
        >>> types = list_change_types()
        >>> "major" in types
        True
        >>> "patch" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_CHANGE_TYPES)


def list_diff_types() -> list[str]:
    """List all valid diff types.

    Returns:
        Sorted list of diff type names.

    Examples:
        >>> types = list_diff_types()
        >>> "config" in types
        True
        >>> "weights" in types
        True
        >>> types == sorted(types)
        True
    """
    return sorted(VALID_DIFF_TYPES)


def get_version_scheme(name: str) -> VersionScheme:
    """Get version scheme from string name.

    Args:
        name: Version scheme name.

    Returns:
        VersionScheme enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_version_scheme("semantic")
        <VersionScheme.SEMANTIC: 'semantic'>

        >>> get_version_scheme("date_based")
        <VersionScheme.DATE_BASED: 'date_based'>

        >>> get_version_scheme("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: version_scheme must be one of
    """
    if name not in VALID_VERSION_SCHEMES:
        msg = f"version_scheme must be one of {VALID_VERSION_SCHEMES}, got '{name}'"
        raise ValueError(msg)
    return VersionScheme(name)


def get_change_type(name: str) -> ChangeType:
    """Get change type from string name.

    Args:
        name: Change type name.

    Returns:
        ChangeType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_change_type("major")
        <ChangeType.MAJOR: 'major'>

        >>> get_change_type("patch")
        <ChangeType.PATCH: 'patch'>

        >>> get_change_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: change_type must be one of
    """
    if name not in VALID_CHANGE_TYPES:
        msg = f"change_type must be one of {VALID_CHANGE_TYPES}, got '{name}'"
        raise ValueError(msg)
    return ChangeType(name)


def get_diff_type(name: str) -> DiffType:
    """Get diff type from string name.

    Args:
        name: Diff type name.

    Returns:
        DiffType enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_diff_type("config")
        <DiffType.CONFIG: 'config'>

        >>> get_diff_type("weights")
        <DiffType.WEIGHTS: 'weights'>

        >>> get_diff_type("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: diff_type must be one of
    """
    if name not in VALID_DIFF_TYPES:
        msg = f"diff_type must be one of {VALID_DIFF_TYPES}, got '{name}'"
        raise ValueError(msg)
    return DiffType(name)


# Regex for parsing semantic versions
_SEMVER_PATTERN = re.compile(
    r"^v?(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def parse_version(version_string: str) -> VersionInfo:
    """Parse a version string into VersionInfo.

    Supports semantic versioning format: major.minor.patch[-prerelease][+build]

    Args:
        version_string: Version string to parse.

    Returns:
        Parsed VersionInfo.

    Raises:
        ValueError: If version string is invalid.

    Examples:
        >>> info = parse_version("1.2.3")
        >>> info.major, info.minor, info.patch
        (1, 2, 3)

        >>> info = parse_version("v2.0.0-alpha")
        >>> info.pre_release
        'alpha'

        >>> info = parse_version("1.0.0+build.123")
        >>> info.build_metadata
        'build.123'

        >>> info = parse_version("3.2.1-beta.2+build.456")
        >>> (info.major, info.pre_release, info.build_metadata)
        (3, 'beta.2', 'build.456')

        >>> parse_version("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: version_string cannot be empty

        >>> parse_version("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid version format
    """
    if not version_string:
        msg = "version_string cannot be empty"
        raise ValueError(msg)

    match = _SEMVER_PATTERN.match(version_string)
    if not match:
        msg = f"invalid version format: '{version_string}'"
        raise ValueError(msg)

    return VersionInfo(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch")),
        pre_release=match.group("prerelease"),
        build_metadata=match.group("buildmetadata"),
    )


def compare_versions(version_a: VersionInfo, version_b: VersionInfo) -> int:
    """Compare two versions.

    Compares versions according to semantic versioning rules.
    Pre-release versions are compared alphanumerically.

    Args:
        version_a: First version.
        version_b: Second version.

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b.

    Examples:
        >>> v1 = create_version_info(1, 0, 0)
        >>> v2 = create_version_info(2, 0, 0)
        >>> compare_versions(v1, v2)
        -1

        >>> v1 = create_version_info(1, 2, 0)
        >>> v2 = create_version_info(1, 1, 0)
        >>> compare_versions(v1, v2)
        1

        >>> v1 = create_version_info(1, 0, 0)
        >>> v2 = create_version_info(1, 0, 0)
        >>> compare_versions(v1, v2)
        0

        >>> v1 = create_version_info(1, 0, 0, pre_release="alpha")
        >>> v2 = create_version_info(1, 0, 0)
        >>> compare_versions(v1, v2)
        -1

        >>> v1 = create_version_info(1, 0, 0, pre_release="alpha")
        >>> v2 = create_version_info(1, 0, 0, pre_release="beta")
        >>> compare_versions(v1, v2)
        -1
    """
    # Compare major.minor.patch
    if version_a.major != version_b.major:
        return -1 if version_a.major < version_b.major else 1
    if version_a.minor != version_b.minor:
        return -1 if version_a.minor < version_b.minor else 1
    if version_a.patch != version_b.patch:
        return -1 if version_a.patch < version_b.patch else 1

    # Pre-release versions have lower precedence than normal versions
    if version_a.pre_release is None and version_b.pre_release is None:
        return 0
    if version_a.pre_release is None:
        return 1  # a is release, b is pre-release
    if version_b.pre_release is None:
        return -1  # a is pre-release, b is release

    # Compare pre-release identifiers
    if version_a.pre_release < version_b.pre_release:
        return -1
    if version_a.pre_release > version_b.pre_release:
        return 1
    return 0


def increment_version(version: VersionInfo, change_type: str) -> VersionInfo:
    """Increment a version based on change type.

    Args:
        version: Version to increment.
        change_type: Type of change ("major", "minor", "patch", "pre_release").

    Returns:
        New incremented VersionInfo.

    Raises:
        ValueError: If change_type is invalid.

    Examples:
        >>> v = create_version_info(1, 2, 3)
        >>> new_v = increment_version(v, "major")
        >>> new_v.major, new_v.minor, new_v.patch
        (2, 0, 0)

        >>> v = create_version_info(1, 2, 3)
        >>> new_v = increment_version(v, "minor")
        >>> new_v.major, new_v.minor, new_v.patch
        (1, 3, 0)

        >>> v = create_version_info(1, 2, 3)
        >>> new_v = increment_version(v, "patch")
        >>> new_v.major, new_v.minor, new_v.patch
        (1, 2, 4)

        >>> v = create_version_info(1, 0, 0)
        >>> new_v = increment_version(v, "pre_release")
        >>> new_v.pre_release
        'alpha.1'

        >>> increment_version(v, "invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: change_type must be one of
    """
    if change_type not in VALID_CHANGE_TYPES:
        msg = f"change_type must be one of {VALID_CHANGE_TYPES}, got '{change_type}'"
        raise ValueError(msg)

    ctype = ChangeType(change_type)

    if ctype == ChangeType.MAJOR:
        return VersionInfo(
            major=version.major + 1,
            minor=0,
            patch=0,
            pre_release=None,
            build_metadata=None,
        )
    elif ctype == ChangeType.MINOR:
        return VersionInfo(
            major=version.major,
            minor=version.minor + 1,
            patch=0,
            pre_release=None,
            build_metadata=None,
        )
    elif ctype == ChangeType.PATCH:
        return VersionInfo(
            major=version.major,
            minor=version.minor,
            patch=version.patch + 1,
            pre_release=None,
            build_metadata=None,
        )
    else:  # PRE_RELEASE
        # Increment pre-release or start new one
        if version.pre_release is None:
            new_pre = "alpha.1"
        else:
            # Try to increment numeric suffix
            parts = version.pre_release.rsplit(".", 1)
            if len(parts) == 2 and parts[1].isdigit():
                new_pre = f"{parts[0]}.{int(parts[1]) + 1}"
            else:
                new_pre = f"{version.pre_release}.1"

        return VersionInfo(
            major=version.major,
            minor=version.minor,
            patch=version.patch,
            pre_release=new_pre,
            build_metadata=version.build_metadata,
        )


def calculate_model_diff(
    old_files: dict[str, str],
    new_files: dict[str, str],
    old_params: int = 0,
    new_params: int = 0,
    old_config: dict[str, object] | None = None,
    new_config: dict[str, object] | None = None,
) -> ModelDiff:
    """Calculate differences between two model versions.

    Args:
        old_files: Dictionary of filename to hash for old version.
        new_files: Dictionary of filename to hash for new version.
        old_params: Parameter count for old version. Defaults to 0.
        new_params: Parameter count for new version. Defaults to 0.
        old_config: Configuration dict for old version. Defaults to None.
        new_config: Configuration dict for new version. Defaults to None.

    Returns:
        ModelDiff describing the changes.

    Examples:
        >>> diff = calculate_model_diff(
        ...     old_files={"config.json": "abc123"},
        ...     new_files={"config.json": "def456", "model.bin": "xyz789"},
        ...     old_params=1000,
        ...     new_params=1500,
        ... )
        >>> "config.json" in diff.changed_files
        True
        >>> "model.bin" in diff.changed_files
        True
        >>> diff.added_params
        500

        >>> diff = calculate_model_diff(
        ...     old_files={},
        ...     new_files={},
        ...     old_config={"hidden_size": 768},
        ...     new_config={"hidden_size": 1024},
        ... )
        >>> diff.config_changes["hidden_size"]
        (768, 1024)

        >>> diff = calculate_model_diff({}, {})
        >>> diff.changed_files
        ()
    """
    if old_config is None:
        old_config = {}
    if new_config is None:
        new_config = {}

    # Find changed files
    changed: set[str] = set()
    all_files = set(old_files.keys()) | set(new_files.keys())
    for filename in all_files:
        old_hash = old_files.get(filename)
        new_hash = new_files.get(filename)
        if old_hash != new_hash:
            changed.add(filename)

    # Calculate parameter changes
    added_params = max(0, new_params - old_params)
    removed_params = max(0, old_params - new_params)

    # Find config changes
    config_changes: dict[str, tuple[object, object]] = {}
    all_keys = set(old_config.keys()) | set(new_config.keys())
    for key in all_keys:
        old_val = old_config.get(key)
        new_val = new_config.get(key)
        if old_val != new_val:
            config_changes[key] = (old_val, new_val)

    return ModelDiff(
        changed_files=tuple(sorted(changed)),
        added_params=added_params,
        removed_params=removed_params,
        config_changes=config_changes,
    )


def validate_version_bump(
    current: VersionInfo,
    proposed: VersionInfo,
    change_type: str,
) -> bool:
    """Validate that a proposed version bump is correct.

    Args:
        current: Current version.
        proposed: Proposed new version.
        change_type: Type of change being made.

    Returns:
        True if the bump is valid, False otherwise.

    Raises:
        ValueError: If change_type is invalid.

    Examples:
        >>> current = create_version_info(1, 2, 3)
        >>> proposed = create_version_info(2, 0, 0)
        >>> validate_version_bump(current, proposed, "major")
        True

        >>> current = create_version_info(1, 2, 3)
        >>> proposed = create_version_info(1, 3, 0)
        >>> validate_version_bump(current, proposed, "minor")
        True

        >>> current = create_version_info(1, 2, 3)
        >>> proposed = create_version_info(1, 2, 4)
        >>> validate_version_bump(current, proposed, "patch")
        True

        >>> current = create_version_info(1, 2, 3)
        >>> proposed = create_version_info(1, 2, 5)
        >>> validate_version_bump(current, proposed, "patch")
        False

        >>> current = create_version_info(1, 0, 0)
        >>> proposed = create_version_info(0, 9, 0)
        >>> validate_version_bump(current, proposed, "major")
        False
    """
    if change_type not in VALID_CHANGE_TYPES:
        msg = f"change_type must be one of {VALID_CHANGE_TYPES}, got '{change_type}'"
        raise ValueError(msg)

    expected = increment_version(current, change_type)

    # For non-pre-release changes, compare major.minor.patch exactly
    ctype = ChangeType(change_type)
    if ctype != ChangeType.PRE_RELEASE:
        return (
            proposed.major == expected.major
            and proposed.minor == expected.minor
            and proposed.patch == expected.patch
        )

    # For pre-release, just check it's the same base version with a pre-release
    return (
        proposed.major == current.major
        and proposed.minor == current.minor
        and proposed.patch == current.patch
        and proposed.pre_release is not None
    )


def format_version_stats(stats: VersionStats) -> str:
    """Format version statistics as a human-readable string.

    Args:
        stats: Version statistics to format.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = create_version_stats(
        ...     total_versions=10,
        ...     latest_version="2.1.0",
        ...     release_frequency_days=14.5,
        ... )
        >>> output = format_version_stats(stats)
        >>> "10 versions" in output
        True
        >>> "2.1.0" in output
        True
        >>> "14.5 days" in output
        True

        >>> stats = create_version_stats()
        >>> output = format_version_stats(stats)
        >>> "0 versions" in output
        True
    """
    lines = [
        "Version Statistics:",
        f"  Total versions: {stats.total_versions} versions",
        f"  Latest version: {stats.latest_version}",
    ]

    if stats.release_frequency_days is not None:
        lines.append(f"  Release frequency: {stats.release_frequency_days} days")
    else:
        lines.append("  Release frequency: N/A")

    return "\n".join(lines)


def format_version(version: VersionInfo) -> str:
    """Format VersionInfo as a version string.

    Args:
        version: Version information to format.

    Returns:
        Formatted version string.

    Examples:
        >>> v = create_version_info(1, 2, 3)
        >>> format_version(v)
        '1.2.3'

        >>> v = create_version_info(1, 0, 0, pre_release="alpha")
        >>> format_version(v)
        '1.0.0-alpha'

        >>> v = create_version_info(2, 0, 0, build_metadata="build.123")
        >>> format_version(v)
        '2.0.0+build.123'

        >>> v = create_version_info(1, 0, 0, pre_release="beta", build_metadata="456")
        >>> format_version(v)
        '1.0.0-beta+456'
    """
    result = f"{version.major}.{version.minor}.{version.patch}"

    if version.pre_release:
        result = f"{result}-{version.pre_release}"

    if version.build_metadata:
        result = f"{result}+{version.build_metadata}"

    return result


def get_recommended_version_config(
    project_type: str = "model",
    team_size: int = 1,
) -> VersionConfig:
    """Get recommended version configuration based on project requirements.

    Args:
        project_type: Type of project ("model", "dataset", "library").
            Defaults to "model".
        team_size: Number of team members. Defaults to 1.

    Returns:
        Recommended VersionConfig.

    Raises:
        ValueError: If project_type is invalid or team_size is non-positive.

    Examples:
        >>> config = get_recommended_version_config()
        >>> config.scheme
        <VersionScheme.SEMANTIC: 'semantic'>

        >>> config = get_recommended_version_config(project_type="dataset")
        >>> config.scheme
        <VersionScheme.DATE_BASED: 'date_based'>

        >>> config = get_recommended_version_config(team_size=5)
        >>> config.require_changelog
        True

        >>> get_recommended_version_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     project_type="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: project_type must be one of

        >>> get_recommended_version_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     team_size=0
        ... )
        Traceback (most recent call last):
        ValueError: team_size must be positive
    """
    valid_types = {"model", "dataset", "library"}
    if project_type not in valid_types:
        msg = f"project_type must be one of {valid_types}, got '{project_type}'"
        raise ValueError(msg)

    if team_size <= 0:
        msg = f"team_size must be positive, got {team_size}"
        raise ValueError(msg)

    # Datasets benefit from date-based versioning
    scheme = "date_based" if project_type == "dataset" else "semantic"

    # Larger teams benefit from requiring changelogs
    require_changelog = team_size > 3

    return create_version_config(
        scheme=scheme,
        auto_increment=True,
        track_changes=True,
        require_changelog=require_changelog,
    )
