"""HuggingFace Model Registry utilities.

This module provides functions for managing model versions, stages, and transitions
in a HuggingFace-compatible model registry pattern.

Examples:
    >>> from hf_gtc.hub.registry import create_registry_config, register_model
    >>> config = create_registry_config()
    >>> config.versioning_scheme
    <VersioningScheme.SEMANTIC: 'semantic'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ModelStage(Enum):
    """Model lifecycle stages.

    Attributes:
        DEVELOPMENT: Model under active development.
        STAGING: Model in staging/validation.
        PRODUCTION: Model deployed to production.
        ARCHIVED: Model archived (no longer active).

    Examples:
        >>> ModelStage.DEVELOPMENT.value
        'development'
        >>> ModelStage.PRODUCTION.value
        'production'
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


VALID_STAGES = frozenset(s.value for s in ModelStage)


class VersioningScheme(Enum):
    """Version numbering schemes.

    Attributes:
        SEMANTIC: Semantic versioning (major.minor.patch).
        TIMESTAMP: Timestamp-based versioning.
        INCREMENTAL: Simple incremental versioning.
        HASH: Hash-based versioning.

    Examples:
        >>> VersioningScheme.SEMANTIC.value
        'semantic'
        >>> VersioningScheme.TIMESTAMP.value
        'timestamp'
    """

    SEMANTIC = "semantic"
    TIMESTAMP = "timestamp"
    INCREMENTAL = "incremental"
    HASH = "hash"


VALID_VERSIONING_SCHEMES = frozenset(v.value for v in VersioningScheme)


class TransitionAction(Enum):
    """Stage transition actions.

    Attributes:
        PROMOTE: Move model to next stage.
        DEMOTE: Move model to previous stage.
        ARCHIVE: Archive the model.
        ROLLBACK: Rollback to previous version.

    Examples:
        >>> TransitionAction.PROMOTE.value
        'promote'
        >>> TransitionAction.ROLLBACK.value
        'rollback'
    """

    PROMOTE = "promote"
    DEMOTE = "demote"
    ARCHIVE = "archive"
    ROLLBACK = "rollback"


VALID_TRANSITION_ACTIONS = frozenset(a.value for a in TransitionAction)


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Represents a model version in the registry.

    Attributes:
        version_id: Unique version identifier.
        stage: Current lifecycle stage.
        created_at: Creation timestamp (ISO format).
        metrics: Performance metrics dictionary.
        tags: Version tags.
        description: Version description.

    Examples:
        >>> version = ModelVersion(
        ...     version_id="v1.0.0",
        ...     stage=ModelStage.DEVELOPMENT,
        ...     created_at="2024-01-01T00:00:00Z",
        ...     metrics={"accuracy": 0.95},
        ...     tags=("baseline",),
        ...     description="Initial release",
        ... )
        >>> version.version_id
        'v1.0.0'
        >>> version.stage
        <ModelStage.DEVELOPMENT: 'development'>
    """

    version_id: str
    stage: ModelStage
    created_at: str
    metrics: dict[str, float]
    tags: tuple[str, ...]
    description: str


@dataclass(frozen=True, slots=True)
class RegistryConfig:
    """Configuration for model registry.

    Attributes:
        versioning_scheme: How versions are numbered.
        auto_archive: Automatically archive old versions.
        retention_policy: Days to retain archived versions.
        require_approval: Require approval for stage transitions.

    Examples:
        >>> config = RegistryConfig(
        ...     versioning_scheme=VersioningScheme.SEMANTIC,
        ...     auto_archive=True,
        ...     retention_policy=90,
        ...     require_approval=True,
        ... )
        >>> config.versioning_scheme
        <VersioningScheme.SEMANTIC: 'semantic'>
        >>> config.auto_archive
        True
    """

    versioning_scheme: VersioningScheme
    auto_archive: bool
    retention_policy: int
    require_approval: bool


@dataclass(frozen=True, slots=True)
class TransitionRequest:
    """Request to transition a model between stages.

    Attributes:
        model_name: Name of the model.
        from_stage: Current stage.
        to_stage: Target stage.
        action: Transition action type.
        approval_required: Whether approval is needed.

    Examples:
        >>> request = TransitionRequest(
        ...     model_name="my-model",
        ...     from_stage=ModelStage.STAGING,
        ...     to_stage=ModelStage.PRODUCTION,
        ...     action=TransitionAction.PROMOTE,
        ...     approval_required=True,
        ... )
        >>> request.model_name
        'my-model'
        >>> request.action
        <TransitionAction.PROMOTE: 'promote'>
    """

    model_name: str
    from_stage: ModelStage
    to_stage: ModelStage
    action: TransitionAction
    approval_required: bool


@dataclass(frozen=True, slots=True)
class RegistryStats:
    """Statistics for a model registry.

    Attributes:
        total_models: Total number of registered models.
        versions_by_stage: Count of versions in each stage.
        storage_used_gb: Total storage used in GB.
        transitions_count: Total number of stage transitions.

    Examples:
        >>> stats = RegistryStats(
        ...     total_models=100,
        ...     versions_by_stage={"development": 50, "production": 30},
        ...     storage_used_gb=250.5,
        ...     transitions_count=500,
        ... )
        >>> stats.total_models
        100
        >>> stats.storage_used_gb
        250.5
    """

    total_models: int
    versions_by_stage: dict[str, int]
    storage_used_gb: float
    transitions_count: int


def validate_model_version(version: ModelVersion) -> None:
    """Validate a model version.

    Args:
        version: Model version to validate.

    Raises:
        ValueError: If version is invalid.

    Examples:
        >>> version = ModelVersion(
        ...     "v1.0.0", ModelStage.DEVELOPMENT, "2024-01-01T00:00:00Z",
        ...     {"accuracy": 0.95}, ("baseline",), "Initial"
        ... )
        >>> validate_model_version(version)  # No error

        >>> bad = ModelVersion(
        ...     "", ModelStage.DEVELOPMENT, "2024-01-01T00:00:00Z",
        ...     {}, (), ""
        ... )
        >>> validate_model_version(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: version_id cannot be empty
    """
    if not version.version_id:
        msg = "version_id cannot be empty"
        raise ValueError(msg)

    if not version.created_at:
        msg = "created_at cannot be empty"
        raise ValueError(msg)


def validate_registry_config(config: RegistryConfig) -> None:
    """Validate registry configuration.

    Args:
        config: Registry configuration to validate.

    Raises:
        ValueError: If configuration is invalid.

    Examples:
        >>> config = RegistryConfig(
        ...     VersioningScheme.SEMANTIC, True, 90, True
        ... )
        >>> validate_registry_config(config)  # No error

        >>> bad = RegistryConfig(
        ...     VersioningScheme.SEMANTIC, True, -1, True
        ... )
        >>> validate_registry_config(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: retention_policy must be non-negative
    """
    if config.retention_policy < 0:
        msg = f"retention_policy must be non-negative, got {config.retention_policy}"
        raise ValueError(msg)


def validate_transition_request(request: TransitionRequest) -> None:
    """Validate a stage transition request.

    Args:
        request: Transition request to validate.

    Raises:
        ValueError: If request is invalid.

    Examples:
        >>> request = TransitionRequest(
        ...     "my-model", ModelStage.STAGING, ModelStage.PRODUCTION,
        ...     TransitionAction.PROMOTE, True
        ... )
        >>> validate_transition_request(request)  # No error

        >>> bad = TransitionRequest(
        ...     "", ModelStage.STAGING, ModelStage.PRODUCTION,
        ...     TransitionAction.PROMOTE, True
        ... )
        >>> validate_transition_request(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not request.model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if request.from_stage == request.to_stage:
        msg = "from_stage and to_stage must be different"
        raise ValueError(msg)


def validate_registry_stats(stats: RegistryStats) -> None:
    """Validate registry statistics.

    Args:
        stats: Registry statistics to validate.

    Raises:
        ValueError: If statistics are invalid.

    Examples:
        >>> stats = RegistryStats(100, {"development": 50}, 250.5, 500)
        >>> validate_registry_stats(stats)  # No error

        >>> bad = RegistryStats(-1, {}, 0.0, 0)
        >>> validate_registry_stats(bad)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: total_models must be non-negative
    """
    if stats.total_models < 0:
        msg = f"total_models must be non-negative, got {stats.total_models}"
        raise ValueError(msg)

    if stats.storage_used_gb < 0:
        msg = f"storage_used_gb must be non-negative, got {stats.storage_used_gb}"
        raise ValueError(msg)

    if stats.transitions_count < 0:
        msg = f"transitions_count must be non-negative, got {stats.transitions_count}"
        raise ValueError(msg)

    for stage, count in stats.versions_by_stage.items():
        if count < 0:
            msg = f"version count for '{stage}' must be non-negative, got {count}"
            raise ValueError(msg)


def create_model_version(
    version_id: str,
    stage: str = "development",
    created_at: str = "",
    metrics: dict[str, float] | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
    description: str = "",
) -> ModelVersion:
    """Create a model version.

    Args:
        version_id: Unique version identifier.
        stage: Lifecycle stage. Defaults to "development".
        created_at: Creation timestamp. Defaults to current time.
        metrics: Performance metrics. Defaults to empty dict.
        tags: Version tags. Defaults to empty tuple.
        description: Version description. Defaults to "".

    Returns:
        ModelVersion with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> version = create_model_version("v1.0.0")
        >>> version.version_id
        'v1.0.0'
        >>> version.stage
        <ModelStage.DEVELOPMENT: 'development'>

        >>> version = create_model_version(
        ...     "v2.0.0",
        ...     stage="production",
        ...     metrics={"accuracy": 0.98},
        ...     tags=["optimized", "quantized"],
        ... )
        >>> version.stage
        <ModelStage.PRODUCTION: 'production'>
        >>> version.tags
        ('optimized', 'quantized')

        >>> create_model_version("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: version_id cannot be empty
    """
    if stage not in VALID_STAGES:
        msg = f"stage must be one of {VALID_STAGES}, got '{stage}'"
        raise ValueError(msg)

    if metrics is None:
        metrics = {}

    if tags is None:
        tags_tuple: tuple[str, ...] = ()
    elif isinstance(tags, list):
        tags_tuple = tuple(tags)
    else:
        tags_tuple = tags

    if not created_at:
        from datetime import UTC, datetime

        created_at = datetime.now(UTC).isoformat()

    version = ModelVersion(
        version_id=version_id,
        stage=ModelStage(stage),
        created_at=created_at,
        metrics=metrics,
        tags=tags_tuple,
        description=description,
    )
    validate_model_version(version)
    return version


def create_registry_config(
    versioning_scheme: str = "semantic",
    auto_archive: bool = True,
    retention_policy: int = 90,
    require_approval: bool = False,
) -> RegistryConfig:
    """Create registry configuration.

    Args:
        versioning_scheme: Version numbering scheme. Defaults to "semantic".
        auto_archive: Auto-archive old versions. Defaults to True.
        retention_policy: Days to retain archived versions. Defaults to 90.
        require_approval: Require approval for transitions. Defaults to False.

    Returns:
        RegistryConfig with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> config = create_registry_config()
        >>> config.versioning_scheme
        <VersioningScheme.SEMANTIC: 'semantic'>
        >>> config.retention_policy
        90

        >>> config = create_registry_config(
        ...     versioning_scheme="timestamp",
        ...     require_approval=True,
        ... )
        >>> config.versioning_scheme
        <VersioningScheme.TIMESTAMP: 'timestamp'>

        >>> create_registry_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     versioning_scheme="invalid"
        ... )
        Traceback (most recent call last):
        ValueError: versioning_scheme must be one of
    """
    if versioning_scheme not in VALID_VERSIONING_SCHEMES:
        msg = (
            f"versioning_scheme must be one of {VALID_VERSIONING_SCHEMES}, "
            f"got '{versioning_scheme}'"
        )
        raise ValueError(msg)

    config = RegistryConfig(
        versioning_scheme=VersioningScheme(versioning_scheme),
        auto_archive=auto_archive,
        retention_policy=retention_policy,
        require_approval=require_approval,
    )
    validate_registry_config(config)
    return config


def create_transition_request(
    model_name: str,
    from_stage: str,
    to_stage: str,
    action: str = "promote",
    approval_required: bool = False,
) -> TransitionRequest:
    """Create a stage transition request.

    Args:
        model_name: Name of the model.
        from_stage: Current stage.
        to_stage: Target stage.
        action: Transition action. Defaults to "promote".
        approval_required: Whether approval is needed. Defaults to False.

    Returns:
        TransitionRequest with the specified settings.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> request = create_transition_request(
        ...     "my-model", "staging", "production"
        ... )
        >>> request.model_name
        'my-model'
        >>> request.action
        <TransitionAction.PROMOTE: 'promote'>

        >>> create_transition_request(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "", "staging", "production"
        ... )
        Traceback (most recent call last):
        ValueError: model_name cannot be empty

        >>> create_transition_request(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "model", "staging", "staging"
        ... )
        Traceback (most recent call last):
        ValueError: from_stage and to_stage must be different
    """
    if from_stage not in VALID_STAGES:
        msg = f"from_stage must be one of {VALID_STAGES}, got '{from_stage}'"
        raise ValueError(msg)

    if to_stage not in VALID_STAGES:
        msg = f"to_stage must be one of {VALID_STAGES}, got '{to_stage}'"
        raise ValueError(msg)

    if action not in VALID_TRANSITION_ACTIONS:
        msg = f"action must be one of {VALID_TRANSITION_ACTIONS}, got '{action}'"
        raise ValueError(msg)

    request = TransitionRequest(
        model_name=model_name,
        from_stage=ModelStage(from_stage),
        to_stage=ModelStage(to_stage),
        action=TransitionAction(action),
        approval_required=approval_required,
    )
    validate_transition_request(request)
    return request


def create_registry_stats(
    total_models: int = 0,
    versions_by_stage: dict[str, int] | None = None,
    storage_used_gb: float = 0.0,
    transitions_count: int = 0,
) -> RegistryStats:
    """Create registry statistics.

    Args:
        total_models: Total number of models. Defaults to 0.
        versions_by_stage: Version counts by stage. Defaults to empty dict.
        storage_used_gb: Storage used in GB. Defaults to 0.0.
        transitions_count: Total transitions. Defaults to 0.

    Returns:
        RegistryStats with the specified values.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> stats = create_registry_stats(total_models=50)
        >>> stats.total_models
        50

        >>> stats = create_registry_stats(
        ...     total_models=100,
        ...     versions_by_stage={"production": 30, "staging": 20},
        ...     storage_used_gb=500.0,
        ... )
        >>> stats.versions_by_stage["production"]
        30

        >>> create_registry_stats(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     total_models=-1
        ... )
        Traceback (most recent call last):
        ValueError: total_models must be non-negative
    """
    if versions_by_stage is None:
        versions_by_stage = {}

    stats = RegistryStats(
        total_models=total_models,
        versions_by_stage=versions_by_stage,
        storage_used_gb=storage_used_gb,
        transitions_count=transitions_count,
    )
    validate_registry_stats(stats)
    return stats


def list_stages() -> list[str]:
    """List all available model stages.

    Returns:
        Sorted list of stage names.

    Examples:
        >>> stages = list_stages()
        >>> "development" in stages
        True
        >>> "production" in stages
        True
        >>> stages == sorted(stages)
        True
    """
    return sorted(VALID_STAGES)


def list_versioning_schemes() -> list[str]:
    """List all available versioning schemes.

    Returns:
        Sorted list of versioning scheme names.

    Examples:
        >>> schemes = list_versioning_schemes()
        >>> "semantic" in schemes
        True
        >>> "timestamp" in schemes
        True
        >>> schemes == sorted(schemes)
        True
    """
    return sorted(VALID_VERSIONING_SCHEMES)


def list_transition_actions() -> list[str]:
    """List all available transition actions.

    Returns:
        Sorted list of transition action names.

    Examples:
        >>> actions = list_transition_actions()
        >>> "promote" in actions
        True
        >>> "rollback" in actions
        True
        >>> actions == sorted(actions)
        True
    """
    return sorted(VALID_TRANSITION_ACTIONS)


def get_stage(name: str) -> ModelStage:
    """Get model stage from name.

    Args:
        name: Stage name.

    Returns:
        ModelStage enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_stage("development")
        <ModelStage.DEVELOPMENT: 'development'>

        >>> get_stage("production")
        <ModelStage.PRODUCTION: 'production'>

        >>> get_stage("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: stage must be one of
    """
    if name not in VALID_STAGES:
        msg = f"stage must be one of {VALID_STAGES}, got '{name}'"
        raise ValueError(msg)
    return ModelStage(name)


def get_versioning_scheme(name: str) -> VersioningScheme:
    """Get versioning scheme from name.

    Args:
        name: Versioning scheme name.

    Returns:
        VersioningScheme enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_versioning_scheme("semantic")
        <VersioningScheme.SEMANTIC: 'semantic'>

        >>> get_versioning_scheme("timestamp")
        <VersioningScheme.TIMESTAMP: 'timestamp'>

        >>> get_versioning_scheme("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: versioning_scheme must be one of
    """
    if name not in VALID_VERSIONING_SCHEMES:
        msg = (
            f"versioning_scheme must be one of {VALID_VERSIONING_SCHEMES}, got '{name}'"
        )
        raise ValueError(msg)
    return VersioningScheme(name)


def get_transition_action(name: str) -> TransitionAction:
    """Get transition action from name.

    Args:
        name: Transition action name.

    Returns:
        TransitionAction enum value.

    Raises:
        ValueError: If name is invalid.

    Examples:
        >>> get_transition_action("promote")
        <TransitionAction.PROMOTE: 'promote'>

        >>> get_transition_action("rollback")
        <TransitionAction.ROLLBACK: 'rollback'>

        >>> get_transition_action("invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: transition_action must be one of
    """
    if name not in VALID_TRANSITION_ACTIONS:
        msg = (
            f"transition_action must be one of {VALID_TRANSITION_ACTIONS}, got '{name}'"
        )
        raise ValueError(msg)
    return TransitionAction(name)


# In-memory registry storage for demonstration
_registry: dict[str, list[ModelVersion]] = {}


def register_model(
    model_name: str,
    version: ModelVersion,
) -> ModelVersion:
    """Register a model version in the registry.

    Args:
        model_name: Name of the model.
        version: Model version to register.

    Returns:
        The registered ModelVersion.

    Raises:
        ValueError: If model_name is empty or version is invalid.

    Examples:
        >>> version = create_model_version("v1.0.0")
        >>> registered = register_model("test-model", version)
        >>> registered.version_id
        'v1.0.0'

        >>> register_model("", version)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    validate_model_version(version)

    if model_name not in _registry:
        _registry[model_name] = []

    _registry[model_name].append(version)
    return version


def get_model_version(model_name: str, version_id: str) -> ModelVersion | None:
    """Get a specific model version from the registry.

    Args:
        model_name: Name of the model.
        version_id: Version identifier.

    Returns:
        ModelVersion if found, None otherwise.

    Raises:
        ValueError: If model_name or version_id is empty.

    Examples:
        >>> version = create_model_version("v1.0.0")
        >>> _ = register_model("get-test-model", version)
        >>> result = get_model_version("get-test-model", "v1.0.0")
        >>> result is not None
        True
        >>> result.version_id
        'v1.0.0'

        >>> get_model_version("nonexistent", "v1.0.0") is None
        True

        >>> get_model_version("", "v1.0.0")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty

        >>> get_model_version("model", "")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: version_id cannot be empty
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if not version_id:
        msg = "version_id cannot be empty"
        raise ValueError(msg)

    versions = _registry.get(model_name, [])
    for version in versions:
        if version.version_id == version_id:
            return version
    return None


def list_model_versions(
    model_name: str,
    stage: str | None = None,
) -> list[ModelVersion]:
    """List all versions of a model.

    Args:
        model_name: Name of the model.
        stage: Filter by stage. Defaults to None (all stages).

    Returns:
        List of ModelVersion objects.

    Raises:
        ValueError: If model_name is empty or stage is invalid.

    Examples:
        >>> v1 = create_model_version("v1.0.0", stage="development")
        >>> v2 = create_model_version("v2.0.0", stage="production")
        >>> _ = register_model("list-test-model", v1)
        >>> _ = register_model("list-test-model", v2)
        >>> versions = list_model_versions("list-test-model")
        >>> len(versions) >= 2
        True

        >>> prod_versions = list_model_versions("list-test-model", stage="production")
        >>> all(v.stage == ModelStage.PRODUCTION for v in prod_versions)
        True

        >>> list_model_versions("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if stage is not None and stage not in VALID_STAGES:
        msg = f"stage must be one of {VALID_STAGES}, got '{stage}'"
        raise ValueError(msg)

    versions = _registry.get(model_name, [])

    if stage is not None:
        target_stage = ModelStage(stage)
        versions = [v for v in versions if v.stage == target_stage]

    return versions


def transition_stage(
    model_name: str,
    version_id: str,
    to_stage: str,
    action: str = "promote",
) -> ModelVersion:
    """Transition a model version to a new stage.

    Args:
        model_name: Name of the model.
        version_id: Version identifier.
        to_stage: Target stage.
        action: Transition action. Defaults to "promote".

    Returns:
        New ModelVersion with updated stage.

    Raises:
        ValueError: If parameters are invalid or version not found.

    Examples:
        >>> v = create_model_version("v1.0.0", stage="development")
        >>> _ = register_model("transition-test", v)
        >>> new_v = transition_stage("transition-test", "v1.0.0", "staging")
        >>> new_v.stage
        <ModelStage.STAGING: 'staging'>

        >>> transition_stage(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "", "v1.0.0", "staging"
        ... )
        Traceback (most recent call last):
        ValueError: model_name cannot be empty

        >>> transition_stage(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "model", "v1.0.0", "invalid"
        ... )
        Traceback (most recent call last):
        ValueError: to_stage must be one of
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if not version_id:
        msg = "version_id cannot be empty"
        raise ValueError(msg)

    if to_stage not in VALID_STAGES:
        msg = f"to_stage must be one of {VALID_STAGES}, got '{to_stage}'"
        raise ValueError(msg)

    if action not in VALID_TRANSITION_ACTIONS:
        msg = f"action must be one of {VALID_TRANSITION_ACTIONS}, got '{action}'"
        raise ValueError(msg)

    version = get_model_version(model_name, version_id)
    if version is None:
        msg = f"version '{version_id}' not found for model '{model_name}'"
        raise ValueError(msg)

    # Create new version with updated stage
    new_version = ModelVersion(
        version_id=version.version_id,
        stage=ModelStage(to_stage),
        created_at=version.created_at,
        metrics=version.metrics,
        tags=version.tags,
        description=version.description,
    )

    # Update registry
    versions = _registry.get(model_name, [])
    _registry[model_name] = [
        new_version if v.version_id == version_id else v for v in versions
    ]

    return new_version


def rollback_version(
    model_name: str,
    to_version_id: str,
) -> ModelVersion:
    """Rollback to a previous model version.

    Args:
        model_name: Name of the model.
        to_version_id: Version ID to rollback to.

    Returns:
        The ModelVersion that was rolled back to.

    Raises:
        ValueError: If parameters are invalid or version not found.

    Examples:
        >>> v1 = create_model_version("v1.0.0", stage="production")
        >>> v2 = create_model_version("v2.0.0", stage="production")
        >>> _ = register_model("rollback-test", v1)
        >>> _ = register_model("rollback-test", v2)
        >>> rolled = rollback_version("rollback-test", "v1.0.0")
        >>> rolled.version_id
        'v1.0.0'

        >>> rollback_version("", "v1.0.0")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: model_name cannot be empty

        >>> rollback_version(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     "nonexistent", "v1.0.0"
        ... )
        Traceback (most recent call last):
        ValueError: version 'v1.0.0' not found
    """
    if not model_name:
        msg = "model_name cannot be empty"
        raise ValueError(msg)

    if not to_version_id:
        msg = "to_version_id cannot be empty"
        raise ValueError(msg)

    version = get_model_version(model_name, to_version_id)
    if version is None:
        msg = f"version '{to_version_id}' not found for model '{model_name}'"
        raise ValueError(msg)

    return version


def compare_versions(
    version_a: ModelVersion,
    version_b: ModelVersion,
) -> dict[str, dict[str, float]]:
    """Compare metrics between two model versions.

    Args:
        version_a: First model version.
        version_b: Second model version.

    Returns:
        Dictionary with metric comparisons (diff = b - a).

    Examples:
        >>> v1 = create_model_version(
        ...     "v1.0.0",
        ...     metrics={"accuracy": 0.85, "f1": 0.80},
        ... )
        >>> v2 = create_model_version(
        ...     "v2.0.0",
        ...     metrics={"accuracy": 0.90, "f1": 0.85},
        ... )
        >>> comparison = compare_versions(v1, v2)
        >>> comparison["accuracy"]["diff"]
        0.05
        >>> comparison["f1"]["diff"]
        0.05

        >>> # Metrics only in one version are included
        >>> v3 = create_model_version("v3.0.0", metrics={"loss": 0.1})
        >>> comparison = compare_versions(v1, v3)
        >>> "loss" in comparison
        True
    """
    all_metrics = set(version_a.metrics.keys()) | set(version_b.metrics.keys())

    result: dict[str, dict[str, float]] = {}
    for metric in all_metrics:
        val_a = version_a.metrics.get(metric, 0.0)
        val_b = version_b.metrics.get(metric, 0.0)
        result[metric] = {
            "version_a": val_a,
            "version_b": val_b,
            "diff": round(val_b - val_a, 6),
        }

    return result


def format_registry_stats(stats: RegistryStats) -> str:
    """Format registry statistics as a human-readable string.

    Args:
        stats: Registry statistics to format.

    Returns:
        Formatted statistics string.

    Examples:
        >>> stats = create_registry_stats(
        ...     total_models=100,
        ...     versions_by_stage={"production": 30, "staging": 20},
        ...     storage_used_gb=250.5,
        ...     transitions_count=500,
        ... )
        >>> output = format_registry_stats(stats)
        >>> "100 models" in output
        True
        >>> "250.5 GB" in output
        True

        >>> stats = create_registry_stats()
        >>> output = format_registry_stats(stats)
        >>> "0 models" in output
        True
    """
    lines = [
        "Registry Statistics:",
        f"  Total models: {stats.total_models} models",
        f"  Storage used: {stats.storage_used_gb} GB",
        f"  Total transitions: {stats.transitions_count}",
    ]

    if stats.versions_by_stage:
        lines.append("  Versions by stage:")
        for stage, count in sorted(stats.versions_by_stage.items()):
            lines.append(f"    {stage}: {count}")

    return "\n".join(lines)


def get_recommended_registry_config(
    team_size: int = 1,
    compliance_required: bool = False,
) -> RegistryConfig:
    """Get recommended registry configuration based on team requirements.

    Args:
        team_size: Number of team members. Defaults to 1.
        compliance_required: Whether compliance/audit is required. Defaults to False.

    Returns:
        Recommended RegistryConfig.

    Examples:
        >>> config = get_recommended_registry_config()
        >>> config.require_approval
        False
        >>> config.versioning_scheme
        <VersioningScheme.SEMANTIC: 'semantic'>

        >>> config = get_recommended_registry_config(team_size=10)
        >>> config.require_approval
        True

        >>> config = get_recommended_registry_config(compliance_required=True)
        >>> config.retention_policy
        365
        >>> config.require_approval
        True

        >>> get_recommended_registry_config(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     team_size=0
        ... )
        Traceback (most recent call last):
        ValueError: team_size must be positive
    """
    if team_size <= 0:
        msg = f"team_size must be positive, got {team_size}"
        raise ValueError(msg)

    # Large teams benefit from approval workflows
    require_approval = team_size > 5 or compliance_required

    # Compliance requires longer retention
    retention_policy = 365 if compliance_required else 90

    # Auto-archive helps with compliance
    auto_archive = True

    # Semantic versioning is recommended for most cases
    versioning_scheme = "semantic"

    return create_registry_config(
        versioning_scheme=versioning_scheme,
        auto_archive=auto_archive,
        retention_policy=retention_policy,
        require_approval=require_approval,
    )


def clear_registry() -> None:
    """Clear the in-memory registry.

    This is primarily useful for testing.

    Examples:
        >>> v = create_model_version("v1.0.0")
        >>> _ = register_model("clear-test", v)
        >>> clear_registry()
        >>> list_model_versions("clear-test")
        []
    """
    _registry.clear()
