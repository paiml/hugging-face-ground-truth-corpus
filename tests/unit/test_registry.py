"""Tests for hub.registry module."""

from __future__ import annotations

import pytest

from hf_gtc.hub.registry import (
    VALID_STAGES,
    VALID_TRANSITION_ACTIONS,
    VALID_VERSIONING_SCHEMES,
    ModelStage,
    ModelVersion,
    RegistryConfig,
    RegistryStats,
    TransitionAction,
    TransitionRequest,
    VersioningScheme,
    clear_registry,
    compare_versions,
    create_model_version,
    create_registry_config,
    create_registry_stats,
    create_transition_request,
    format_registry_stats,
    get_model_version,
    get_recommended_registry_config,
    get_stage,
    get_transition_action,
    get_versioning_scheme,
    list_model_versions,
    list_stages,
    list_transition_actions,
    list_versioning_schemes,
    register_model,
    rollback_version,
    transition_stage,
    validate_model_version,
    validate_registry_config,
    validate_registry_stats,
    validate_transition_request,
)


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clear registry before each test."""
    clear_registry()


class TestModelStage:
    """Tests for ModelStage enum."""

    def test_all_stages_have_values(self) -> None:
        """All stages have string values."""
        for stage in ModelStage:
            assert isinstance(stage.value, str)

    def test_development_value(self) -> None:
        """Development has correct value."""
        assert ModelStage.DEVELOPMENT.value == "development"

    def test_staging_value(self) -> None:
        """Staging has correct value."""
        assert ModelStage.STAGING.value == "staging"

    def test_production_value(self) -> None:
        """Production has correct value."""
        assert ModelStage.PRODUCTION.value == "production"

    def test_archived_value(self) -> None:
        """Archived has correct value."""
        assert ModelStage.ARCHIVED.value == "archived"

    def test_valid_stages_frozenset(self) -> None:
        """VALID_STAGES is a frozenset."""
        assert isinstance(VALID_STAGES, frozenset)

    def test_valid_stages_contains_all(self) -> None:
        """VALID_STAGES contains all enum values."""
        for stage in ModelStage:
            assert stage.value in VALID_STAGES


class TestVersioningScheme:
    """Tests for VersioningScheme enum."""

    def test_all_schemes_have_values(self) -> None:
        """All schemes have string values."""
        for scheme in VersioningScheme:
            assert isinstance(scheme.value, str)

    def test_semantic_value(self) -> None:
        """Semantic has correct value."""
        assert VersioningScheme.SEMANTIC.value == "semantic"

    def test_timestamp_value(self) -> None:
        """Timestamp has correct value."""
        assert VersioningScheme.TIMESTAMP.value == "timestamp"

    def test_incremental_value(self) -> None:
        """Incremental has correct value."""
        assert VersioningScheme.INCREMENTAL.value == "incremental"

    def test_hash_value(self) -> None:
        """Hash has correct value."""
        assert VersioningScheme.HASH.value == "hash"

    def test_valid_schemes_frozenset(self) -> None:
        """VALID_VERSIONING_SCHEMES is a frozenset."""
        assert isinstance(VALID_VERSIONING_SCHEMES, frozenset)


class TestTransitionAction:
    """Tests for TransitionAction enum."""

    def test_all_actions_have_values(self) -> None:
        """All actions have string values."""
        for action in TransitionAction:
            assert isinstance(action.value, str)

    def test_promote_value(self) -> None:
        """Promote has correct value."""
        assert TransitionAction.PROMOTE.value == "promote"

    def test_demote_value(self) -> None:
        """Demote has correct value."""
        assert TransitionAction.DEMOTE.value == "demote"

    def test_archive_value(self) -> None:
        """Archive has correct value."""
        assert TransitionAction.ARCHIVE.value == "archive"

    def test_rollback_value(self) -> None:
        """Rollback has correct value."""
        assert TransitionAction.ROLLBACK.value == "rollback"

    def test_valid_actions_frozenset(self) -> None:
        """VALID_TRANSITION_ACTIONS is a frozenset."""
        assert isinstance(VALID_TRANSITION_ACTIONS, frozenset)


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_create_version(self) -> None:
        """Create model version."""
        version = ModelVersion(
            version_id="v1.0.0",
            stage=ModelStage.DEVELOPMENT,
            created_at="2024-01-01T00:00:00Z",
            metrics={"accuracy": 0.95},
            tags=("baseline",),
            description="Initial release",
        )
        assert version.version_id == "v1.0.0"
        assert version.stage == ModelStage.DEVELOPMENT

    def test_version_is_frozen(self) -> None:
        """Version is immutable."""
        version = ModelVersion(
            "v1.0.0",
            ModelStage.DEVELOPMENT,
            "2024-01-01T00:00:00Z",
            {},
            (),
            "",
        )
        with pytest.raises(AttributeError):
            version.version_id = "v2.0.0"  # type: ignore[misc]

    def test_version_has_slots(self) -> None:
        """Version uses __slots__."""
        version = ModelVersion(
            "v1.0.0",
            ModelStage.DEVELOPMENT,
            "2024-01-01T00:00:00Z",
            {},
            (),
            "",
        )
        assert not hasattr(version, "__dict__")


class TestRegistryConfig:
    """Tests for RegistryConfig dataclass."""

    def test_create_config(self) -> None:
        """Create registry config."""
        config = RegistryConfig(
            versioning_scheme=VersioningScheme.SEMANTIC,
            auto_archive=True,
            retention_policy=90,
            require_approval=True,
        )
        assert config.versioning_scheme == VersioningScheme.SEMANTIC
        assert config.auto_archive is True

    def test_config_is_frozen(self) -> None:
        """Config is immutable."""
        config = RegistryConfig(
            VersioningScheme.SEMANTIC,
            True,
            90,
            True,
        )
        with pytest.raises(AttributeError):
            config.auto_archive = False  # type: ignore[misc]


class TestTransitionRequest:
    """Tests for TransitionRequest dataclass."""

    def test_create_request(self) -> None:
        """Create transition request."""
        request = TransitionRequest(
            model_name="my-model",
            from_stage=ModelStage.STAGING,
            to_stage=ModelStage.PRODUCTION,
            action=TransitionAction.PROMOTE,
            approval_required=True,
        )
        assert request.model_name == "my-model"
        assert request.action == TransitionAction.PROMOTE

    def test_request_is_frozen(self) -> None:
        """Request is immutable."""
        request = TransitionRequest(
            "model",
            ModelStage.STAGING,
            ModelStage.PRODUCTION,
            TransitionAction.PROMOTE,
            True,
        )
        with pytest.raises(AttributeError):
            request.model_name = "other"  # type: ignore[misc]


class TestRegistryStats:
    """Tests for RegistryStats dataclass."""

    def test_create_stats(self) -> None:
        """Create registry stats."""
        stats = RegistryStats(
            total_models=100,
            versions_by_stage={"development": 50},
            storage_used_gb=250.5,
            transitions_count=500,
        )
        assert stats.total_models == 100
        assert stats.storage_used_gb == pytest.approx(250.5)

    def test_stats_is_frozen(self) -> None:
        """Stats is immutable."""
        stats = RegistryStats(100, {}, 0.0, 0)
        with pytest.raises(AttributeError):
            stats.total_models = 200  # type: ignore[misc]


class TestValidateModelVersion:
    """Tests for validate_model_version function."""

    def test_valid_version(self) -> None:
        """Valid version passes validation."""
        version = ModelVersion(
            "v1.0.0",
            ModelStage.DEVELOPMENT,
            "2024-01-01T00:00:00Z",
            {},
            (),
            "",
        )
        validate_model_version(version)

    def test_empty_version_id_raises(self) -> None:
        """Empty version_id raises ValueError."""
        version = ModelVersion(
            "",
            ModelStage.DEVELOPMENT,
            "2024-01-01T00:00:00Z",
            {},
            (),
            "",
        )
        with pytest.raises(ValueError, match="version_id cannot be empty"):
            validate_model_version(version)

    def test_empty_created_at_raises(self) -> None:
        """Empty created_at raises ValueError."""
        version = ModelVersion(
            "v1.0.0",
            ModelStage.DEVELOPMENT,
            "",
            {},
            (),
            "",
        )
        with pytest.raises(ValueError, match="created_at cannot be empty"):
            validate_model_version(version)


class TestValidateRegistryConfig:
    """Tests for validate_registry_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = RegistryConfig(
            VersioningScheme.SEMANTIC,
            True,
            90,
            True,
        )
        validate_registry_config(config)

    def test_negative_retention_raises(self) -> None:
        """Negative retention_policy raises ValueError."""
        config = RegistryConfig(
            VersioningScheme.SEMANTIC,
            True,
            -1,
            True,
        )
        with pytest.raises(ValueError, match="retention_policy must be non-negative"):
            validate_registry_config(config)


class TestValidateTransitionRequest:
    """Tests for validate_transition_request function."""

    def test_valid_request(self) -> None:
        """Valid request passes validation."""
        request = TransitionRequest(
            "my-model",
            ModelStage.STAGING,
            ModelStage.PRODUCTION,
            TransitionAction.PROMOTE,
            True,
        )
        validate_transition_request(request)

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        request = TransitionRequest(
            "",
            ModelStage.STAGING,
            ModelStage.PRODUCTION,
            TransitionAction.PROMOTE,
            True,
        )
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            validate_transition_request(request)

    def test_same_stages_raises(self) -> None:
        """Same from_stage and to_stage raises ValueError."""
        request = TransitionRequest(
            "model",
            ModelStage.STAGING,
            ModelStage.STAGING,
            TransitionAction.PROMOTE,
            True,
        )
        with pytest.raises(
            ValueError, match="from_stage and to_stage must be different"
        ):
            validate_transition_request(request)


class TestValidateRegistryStats:
    """Tests for validate_registry_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = RegistryStats(100, {"development": 50}, 250.5, 500)
        validate_registry_stats(stats)

    def test_negative_total_models_raises(self) -> None:
        """Negative total_models raises ValueError."""
        stats = RegistryStats(-1, {}, 0.0, 0)
        with pytest.raises(ValueError, match="total_models must be non-negative"):
            validate_registry_stats(stats)

    def test_negative_storage_raises(self) -> None:
        """Negative storage_used_gb raises ValueError."""
        stats = RegistryStats(0, {}, -1.0, 0)
        with pytest.raises(ValueError, match="storage_used_gb must be non-negative"):
            validate_registry_stats(stats)

    def test_negative_transitions_raises(self) -> None:
        """Negative transitions_count raises ValueError."""
        stats = RegistryStats(0, {}, 0.0, -1)
        with pytest.raises(ValueError, match="transitions_count must be non-negative"):
            validate_registry_stats(stats)

    def test_negative_version_count_raises(self) -> None:
        """Negative version count in versions_by_stage raises ValueError."""
        stats = RegistryStats(0, {"development": -1}, 0.0, 0)
        with pytest.raises(
            ValueError, match="version count for 'development' must be non-negative"
        ):
            validate_registry_stats(stats)


class TestCreateModelVersion:
    """Tests for create_model_version function."""

    def test_default_version(self) -> None:
        """Create default version."""
        version = create_model_version("v1.0.0")
        assert version.version_id == "v1.0.0"
        assert version.stage == ModelStage.DEVELOPMENT

    def test_custom_version(self) -> None:
        """Create custom version."""
        version = create_model_version(
            "v2.0.0",
            stage="production",
            metrics={"accuracy": 0.98},
            tags=["optimized"],
            description="Production release",
        )
        assert version.stage == ModelStage.PRODUCTION
        assert version.metrics["accuracy"] == pytest.approx(0.98)
        assert version.tags == ("optimized",)

    def test_tuple_tags(self) -> None:
        """Tags can be passed as tuple."""
        version = create_model_version("v1.0.0", tags=("a", "b"))
        assert version.tags == ("a", "b")

    def test_empty_version_id_raises(self) -> None:
        """Empty version_id raises ValueError."""
        with pytest.raises(ValueError, match="version_id cannot be empty"):
            create_model_version("")

    def test_invalid_stage_raises(self) -> None:
        """Invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="stage must be one of"):
            create_model_version("v1.0.0", stage="invalid")

    def test_auto_created_at(self) -> None:
        """created_at is auto-generated when not provided."""
        version = create_model_version("v1.0.0")
        assert version.created_at != ""
        assert "T" in version.created_at  # ISO format


class TestCreateRegistryConfig:
    """Tests for create_registry_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_registry_config()
        assert config.versioning_scheme == VersioningScheme.SEMANTIC
        assert config.retention_policy == 90

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_registry_config(
            versioning_scheme="timestamp",
            auto_archive=False,
            retention_policy=180,
            require_approval=True,
        )
        assert config.versioning_scheme == VersioningScheme.TIMESTAMP
        assert config.auto_archive is False
        assert config.retention_policy == 180
        assert config.require_approval is True

    def test_invalid_scheme_raises(self) -> None:
        """Invalid versioning_scheme raises ValueError."""
        with pytest.raises(ValueError, match="versioning_scheme must be one of"):
            create_registry_config(versioning_scheme="invalid")

    def test_negative_retention_raises(self) -> None:
        """Negative retention_policy raises ValueError."""
        with pytest.raises(ValueError, match="retention_policy must be non-negative"):
            create_registry_config(retention_policy=-1)


class TestCreateTransitionRequest:
    """Tests for create_transition_request function."""

    def test_default_request(self) -> None:
        """Create default transition request."""
        request = create_transition_request(
            "my-model",
            "staging",
            "production",
        )
        assert request.model_name == "my-model"
        assert request.action == TransitionAction.PROMOTE

    def test_custom_request(self) -> None:
        """Create custom transition request."""
        request = create_transition_request(
            "my-model",
            "production",
            "staging",
            action="demote",
            approval_required=True,
        )
        assert request.action == TransitionAction.DEMOTE
        assert request.approval_required is True

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            create_transition_request("", "staging", "production")

    def test_invalid_from_stage_raises(self) -> None:
        """Invalid from_stage raises ValueError."""
        with pytest.raises(ValueError, match="from_stage must be one of"):
            create_transition_request("model", "invalid", "production")

    def test_invalid_to_stage_raises(self) -> None:
        """Invalid to_stage raises ValueError."""
        with pytest.raises(ValueError, match="to_stage must be one of"):
            create_transition_request("model", "staging", "invalid")

    def test_invalid_action_raises(self) -> None:
        """Invalid action raises ValueError."""
        with pytest.raises(ValueError, match="action must be one of"):
            create_transition_request(
                "model", "staging", "production", action="invalid"
            )

    def test_same_stages_raises(self) -> None:
        """Same from_stage and to_stage raises ValueError."""
        with pytest.raises(
            ValueError, match="from_stage and to_stage must be different"
        ):
            create_transition_request("model", "staging", "staging")


class TestCreateRegistryStats:
    """Tests for create_registry_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_registry_stats()
        assert stats.total_models == 0
        assert stats.storage_used_gb == pytest.approx(0.0)

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_registry_stats(
            total_models=50,
            versions_by_stage={"production": 20},
            storage_used_gb=100.5,
            transitions_count=200,
        )
        assert stats.total_models == 50
        assert stats.versions_by_stage["production"] == 20

    def test_negative_models_raises(self) -> None:
        """Negative total_models raises ValueError."""
        with pytest.raises(ValueError, match="total_models must be non-negative"):
            create_registry_stats(total_models=-1)

    def test_negative_storage_raises(self) -> None:
        """Negative storage_used_gb raises ValueError."""
        with pytest.raises(ValueError, match="storage_used_gb must be non-negative"):
            create_registry_stats(storage_used_gb=-1.0)

    def test_negative_transitions_raises(self) -> None:
        """Negative transitions_count raises ValueError."""
        with pytest.raises(ValueError, match="transitions_count must be non-negative"):
            create_registry_stats(transitions_count=-1)


class TestListStages:
    """Tests for list_stages function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        stages = list_stages()
        assert stages == sorted(stages)

    def test_contains_development(self) -> None:
        """Contains development."""
        stages = list_stages()
        assert "development" in stages

    def test_contains_production(self) -> None:
        """Contains production."""
        stages = list_stages()
        assert "production" in stages


class TestListVersioningSchemes:
    """Tests for list_versioning_schemes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        schemes = list_versioning_schemes()
        assert schemes == sorted(schemes)

    def test_contains_semantic(self) -> None:
        """Contains semantic."""
        schemes = list_versioning_schemes()
        assert "semantic" in schemes

    def test_contains_timestamp(self) -> None:
        """Contains timestamp."""
        schemes = list_versioning_schemes()
        assert "timestamp" in schemes


class TestListTransitionActions:
    """Tests for list_transition_actions function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        actions = list_transition_actions()
        assert actions == sorted(actions)

    def test_contains_promote(self) -> None:
        """Contains promote."""
        actions = list_transition_actions()
        assert "promote" in actions

    def test_contains_rollback(self) -> None:
        """Contains rollback."""
        actions = list_transition_actions()
        assert "rollback" in actions


class TestGetStage:
    """Tests for get_stage function."""

    def test_get_development(self) -> None:
        """Get development stage."""
        assert get_stage("development") == ModelStage.DEVELOPMENT

    def test_get_production(self) -> None:
        """Get production stage."""
        assert get_stage("production") == ModelStage.PRODUCTION

    def test_get_staging(self) -> None:
        """Get staging stage."""
        assert get_stage("staging") == ModelStage.STAGING

    def test_get_archived(self) -> None:
        """Get archived stage."""
        assert get_stage("archived") == ModelStage.ARCHIVED

    def test_invalid_stage_raises(self) -> None:
        """Invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="stage must be one of"):
            get_stage("invalid")


class TestGetVersioningScheme:
    """Tests for get_versioning_scheme function."""

    def test_get_semantic(self) -> None:
        """Get semantic scheme."""
        assert get_versioning_scheme("semantic") == VersioningScheme.SEMANTIC

    def test_get_timestamp(self) -> None:
        """Get timestamp scheme."""
        assert get_versioning_scheme("timestamp") == VersioningScheme.TIMESTAMP

    def test_get_incremental(self) -> None:
        """Get incremental scheme."""
        assert get_versioning_scheme("incremental") == VersioningScheme.INCREMENTAL

    def test_get_hash(self) -> None:
        """Get hash scheme."""
        assert get_versioning_scheme("hash") == VersioningScheme.HASH

    def test_invalid_scheme_raises(self) -> None:
        """Invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="versioning_scheme must be one of"):
            get_versioning_scheme("invalid")


class TestGetTransitionAction:
    """Tests for get_transition_action function."""

    def test_get_promote(self) -> None:
        """Get promote action."""
        assert get_transition_action("promote") == TransitionAction.PROMOTE

    def test_get_demote(self) -> None:
        """Get demote action."""
        assert get_transition_action("demote") == TransitionAction.DEMOTE

    def test_get_archive(self) -> None:
        """Get archive action."""
        assert get_transition_action("archive") == TransitionAction.ARCHIVE

    def test_get_rollback(self) -> None:
        """Get rollback action."""
        assert get_transition_action("rollback") == TransitionAction.ROLLBACK

    def test_invalid_action_raises(self) -> None:
        """Invalid action raises ValueError."""
        with pytest.raises(ValueError, match="transition_action must be one of"):
            get_transition_action("invalid")


class TestRegisterModel:
    """Tests for register_model function."""

    def test_register_version(self) -> None:
        """Register a model version."""
        version = create_model_version("v1.0.0")
        registered = register_model("test-model", version)
        assert registered.version_id == "v1.0.0"

    def test_register_multiple_versions(self) -> None:
        """Register multiple versions for same model."""
        v1 = create_model_version("v1.0.0")
        v2 = create_model_version("v2.0.0")
        register_model("test-model", v1)
        register_model("test-model", v2)
        versions = list_model_versions("test-model")
        assert len(versions) == 2

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        version = create_model_version("v1.0.0")
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            register_model("", version)


class TestGetModelVersion:
    """Tests for get_model_version function."""

    def test_get_existing_version(self) -> None:
        """Get existing version."""
        version = create_model_version("v1.0.0")
        register_model("test-model", version)
        result = get_model_version("test-model", "v1.0.0")
        assert result is not None
        assert result.version_id == "v1.0.0"

    def test_get_nonexistent_version(self) -> None:
        """Get nonexistent version returns None."""
        result = get_model_version("nonexistent", "v1.0.0")
        assert result is None

    def test_get_nonexistent_model(self) -> None:
        """Get nonexistent model returns None."""
        version = create_model_version("v1.0.0")
        register_model("test-model", version)
        result = get_model_version("test-model", "v999.0.0")
        assert result is None

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            get_model_version("", "v1.0.0")

    def test_empty_version_id_raises(self) -> None:
        """Empty version_id raises ValueError."""
        with pytest.raises(ValueError, match="version_id cannot be empty"):
            get_model_version("model", "")


class TestListModelVersions:
    """Tests for list_model_versions function."""

    def test_list_all_versions(self) -> None:
        """List all versions."""
        v1 = create_model_version("v1.0.0", stage="development")
        v2 = create_model_version("v2.0.0", stage="production")
        register_model("test-model", v1)
        register_model("test-model", v2)
        versions = list_model_versions("test-model")
        assert len(versions) == 2

    def test_filter_by_stage(self) -> None:
        """Filter versions by stage."""
        v1 = create_model_version("v1.0.0", stage="development")
        v2 = create_model_version("v2.0.0", stage="production")
        register_model("test-model", v1)
        register_model("test-model", v2)
        versions = list_model_versions("test-model", stage="production")
        assert len(versions) == 1
        assert versions[0].stage == ModelStage.PRODUCTION

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            list_model_versions("")

    def test_invalid_stage_raises(self) -> None:
        """Invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="stage must be one of"):
            list_model_versions("model", stage="invalid")

    def test_nonexistent_model_returns_empty(self) -> None:
        """Nonexistent model returns empty list."""
        versions = list_model_versions("nonexistent")
        assert versions == []


class TestTransitionStage:
    """Tests for transition_stage function."""

    def test_transition_to_staging(self) -> None:
        """Transition version to staging."""
        v = create_model_version("v1.0.0", stage="development")
        register_model("test-model", v)
        new_v = transition_stage("test-model", "v1.0.0", "staging")
        assert new_v.stage == ModelStage.STAGING

    def test_transition_to_production(self) -> None:
        """Transition version to production."""
        v = create_model_version("v1.0.0", stage="staging")
        register_model("test-model", v)
        new_v = transition_stage("test-model", "v1.0.0", "production")
        assert new_v.stage == ModelStage.PRODUCTION

    def test_transition_with_action(self) -> None:
        """Transition with specific action."""
        v = create_model_version("v1.0.0", stage="production")
        register_model("test-model", v)
        new_v = transition_stage("test-model", "v1.0.0", "archived", action="archive")
        assert new_v.stage == ModelStage.ARCHIVED

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            transition_stage("", "v1.0.0", "staging")

    def test_empty_version_id_raises(self) -> None:
        """Empty version_id raises ValueError."""
        with pytest.raises(ValueError, match="version_id cannot be empty"):
            transition_stage("model", "", "staging")

    def test_invalid_stage_raises(self) -> None:
        """Invalid to_stage raises ValueError."""
        with pytest.raises(ValueError, match="to_stage must be one of"):
            transition_stage("model", "v1.0.0", "invalid")

    def test_invalid_action_raises(self) -> None:
        """Invalid action raises ValueError."""
        with pytest.raises(ValueError, match="action must be one of"):
            transition_stage("model", "v1.0.0", "staging", action="invalid")

    def test_nonexistent_version_raises(self) -> None:
        """Nonexistent version raises ValueError."""
        with pytest.raises(ValueError, match=r"version 'v1\.0\.0' not found"):
            transition_stage("nonexistent", "v1.0.0", "staging")


class TestRollbackVersion:
    """Tests for rollback_version function."""

    def test_rollback_to_version(self) -> None:
        """Rollback to previous version."""
        v1 = create_model_version("v1.0.0", stage="production")
        v2 = create_model_version("v2.0.0", stage="production")
        register_model("test-model", v1)
        register_model("test-model", v2)
        rolled = rollback_version("test-model", "v1.0.0")
        assert rolled.version_id == "v1.0.0"

    def test_empty_model_name_raises(self) -> None:
        """Empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            rollback_version("", "v1.0.0")

    def test_empty_version_id_raises(self) -> None:
        """Empty to_version_id raises ValueError."""
        with pytest.raises(ValueError, match="to_version_id cannot be empty"):
            rollback_version("model", "")

    def test_nonexistent_version_raises(self) -> None:
        """Nonexistent version raises ValueError."""
        with pytest.raises(ValueError, match=r"version 'v1\.0\.0' not found"):
            rollback_version("nonexistent", "v1.0.0")


class TestCompareVersions:
    """Tests for compare_versions function."""

    def test_compare_metrics(self) -> None:
        """Compare metrics between versions."""
        v1 = create_model_version(
            "v1.0.0",
            metrics={"accuracy": 0.85, "f1": 0.80},
        )
        v2 = create_model_version(
            "v2.0.0",
            metrics={"accuracy": 0.90, "f1": 0.85},
        )
        comparison = compare_versions(v1, v2)
        assert comparison["accuracy"]["diff"] == pytest.approx(0.05)
        assert comparison["f1"]["diff"] == pytest.approx(0.05)

    def test_compare_with_missing_metrics(self) -> None:
        """Compare when metrics differ between versions."""
        v1 = create_model_version("v1.0.0", metrics={"accuracy": 0.85})
        v2 = create_model_version("v2.0.0", metrics={"loss": 0.1})
        comparison = compare_versions(v1, v2)
        assert "accuracy" in comparison
        assert "loss" in comparison
        assert comparison["accuracy"]["version_b"] == pytest.approx(0.0)
        assert comparison["loss"]["version_a"] == pytest.approx(0.0)

    def test_compare_empty_metrics(self) -> None:
        """Compare versions with empty metrics."""
        v1 = create_model_version("v1.0.0", metrics={})
        v2 = create_model_version("v2.0.0", metrics={})
        comparison = compare_versions(v1, v2)
        assert comparison == {}

    def test_compare_same_metrics(self) -> None:
        """Compare versions with same metrics."""
        v1 = create_model_version("v1.0.0", metrics={"accuracy": 0.90})
        v2 = create_model_version("v2.0.0", metrics={"accuracy": 0.90})
        comparison = compare_versions(v1, v2)
        assert comparison["accuracy"]["diff"] == pytest.approx(0.0)


class TestFormatRegistryStats:
    """Tests for format_registry_stats function."""

    def test_format_basic_stats(self) -> None:
        """Format basic stats."""
        stats = create_registry_stats(
            total_models=100,
            storage_used_gb=250.5,
            transitions_count=500,
        )
        output = format_registry_stats(stats)
        assert "100 models" in output
        assert "250.5 GB" in output
        assert "500" in output

    def test_format_with_stages(self) -> None:
        """Format stats with version counts by stage."""
        stats = create_registry_stats(
            total_models=50,
            versions_by_stage={"production": 30, "staging": 20},
        )
        output = format_registry_stats(stats)
        assert "production: 30" in output
        assert "staging: 20" in output

    def test_format_empty_stats(self) -> None:
        """Format empty stats."""
        stats = create_registry_stats()
        output = format_registry_stats(stats)
        assert "0 models" in output


class TestGetRecommendedRegistryConfig:
    """Tests for get_recommended_registry_config function."""

    def test_default_config(self) -> None:
        """Get default recommended config."""
        config = get_recommended_registry_config()
        assert config.require_approval is False
        assert config.versioning_scheme == VersioningScheme.SEMANTIC

    def test_large_team_requires_approval(self) -> None:
        """Large teams get approval requirement."""
        config = get_recommended_registry_config(team_size=10)
        assert config.require_approval is True

    def test_compliance_requires_approval(self) -> None:
        """Compliance requirement enables approval."""
        config = get_recommended_registry_config(compliance_required=True)
        assert config.require_approval is True

    def test_compliance_increases_retention(self) -> None:
        """Compliance requirement increases retention."""
        config = get_recommended_registry_config(compliance_required=True)
        assert config.retention_policy == 365

    def test_zero_team_size_raises(self) -> None:
        """Zero team_size raises ValueError."""
        with pytest.raises(ValueError, match="team_size must be positive"):
            get_recommended_registry_config(team_size=0)

    def test_negative_team_size_raises(self) -> None:
        """Negative team_size raises ValueError."""
        with pytest.raises(ValueError, match="team_size must be positive"):
            get_recommended_registry_config(team_size=-1)


class TestClearRegistry:
    """Tests for clear_registry function."""

    def test_clear_removes_all(self) -> None:
        """Clear removes all registered models."""
        v1 = create_model_version("v1.0.0")
        register_model("model-1", v1)
        register_model("model-2", v1)
        clear_registry()
        assert list_model_versions("model-1") == []
        assert list_model_versions("model-2") == []
