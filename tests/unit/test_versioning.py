"""Tests for hub.versioning module."""

from __future__ import annotations

import pytest

from hf_gtc.hub.versioning import (
    VALID_CHANGE_TYPES,
    VALID_DIFF_TYPES,
    VALID_VERSION_SCHEMES,
    ChangeType,
    DiffType,
    ModelDiff,
    VersionConfig,
    VersionInfo,
    VersionScheme,
    VersionStats,
    calculate_model_diff,
    compare_versions,
    create_model_diff,
    create_version_config,
    create_version_info,
    create_version_stats,
    format_version,
    format_version_stats,
    get_change_type,
    get_diff_type,
    get_recommended_version_config,
    get_version_scheme,
    increment_version,
    list_change_types,
    list_diff_types,
    list_version_schemes,
    parse_version,
    validate_model_diff,
    validate_version_bump,
    validate_version_config,
    validate_version_info,
    validate_version_stats,
)


class TestVersionScheme:
    """Tests for VersionScheme enum."""

    def test_all_schemes_have_values(self) -> None:
        """All schemes have string values."""
        for scheme in VersionScheme:
            assert isinstance(scheme.value, str)

    def test_semantic_value(self) -> None:
        """Semantic has correct value."""
        assert VersionScheme.SEMANTIC.value == "semantic"

    def test_date_based_value(self) -> None:
        """Date-based has correct value."""
        assert VersionScheme.DATE_BASED.value == "date_based"

    def test_incremental_value(self) -> None:
        """Incremental has correct value."""
        assert VersionScheme.INCREMENTAL.value == "incremental"

    def test_git_hash_value(self) -> None:
        """Git hash has correct value."""
        assert VersionScheme.GIT_HASH.value == "git_hash"

    def test_valid_schemes_frozenset(self) -> None:
        """VALID_VERSION_SCHEMES is a frozenset."""
        assert isinstance(VALID_VERSION_SCHEMES, frozenset)

    def test_valid_schemes_contains_all(self) -> None:
        """VALID_VERSION_SCHEMES contains all enum values."""
        for scheme in VersionScheme:
            assert scheme.value in VALID_VERSION_SCHEMES


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_have_values(self) -> None:
        """All change types have string values."""
        for ctype in ChangeType:
            assert isinstance(ctype.value, str)

    def test_major_value(self) -> None:
        """Major has correct value."""
        assert ChangeType.MAJOR.value == "major"

    def test_minor_value(self) -> None:
        """Minor has correct value."""
        assert ChangeType.MINOR.value == "minor"

    def test_patch_value(self) -> None:
        """Patch has correct value."""
        assert ChangeType.PATCH.value == "patch"

    def test_pre_release_value(self) -> None:
        """Pre-release has correct value."""
        assert ChangeType.PRE_RELEASE.value == "pre_release"

    def test_valid_types_frozenset(self) -> None:
        """VALID_CHANGE_TYPES is a frozenset."""
        assert isinstance(VALID_CHANGE_TYPES, frozenset)


class TestDiffType:
    """Tests for DiffType enum."""

    def test_all_types_have_values(self) -> None:
        """All diff types have string values."""
        for dtype in DiffType:
            assert isinstance(dtype.value, str)

    def test_config_value(self) -> None:
        """Config has correct value."""
        assert DiffType.CONFIG.value == "config"

    def test_weights_value(self) -> None:
        """Weights has correct value."""
        assert DiffType.WEIGHTS.value == "weights"

    def test_tokenizer_value(self) -> None:
        """Tokenizer has correct value."""
        assert DiffType.TOKENIZER.value == "tokenizer"

    def test_all_value(self) -> None:
        """All has correct value."""
        assert DiffType.ALL.value == "all"

    def test_valid_types_frozenset(self) -> None:
        """VALID_DIFF_TYPES is a frozenset."""
        assert isinstance(VALID_DIFF_TYPES, frozenset)


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_create_version_info(self) -> None:
        """Create version info."""
        info = VersionInfo(
            major=1,
            minor=2,
            patch=3,
            pre_release="alpha",
            build_metadata="build.123",
        )
        assert info.major == 1
        assert info.minor == 2
        assert info.patch == 3
        assert info.pre_release == "alpha"
        assert info.build_metadata == "build.123"

    def test_version_info_is_frozen(self) -> None:
        """VersionInfo is immutable."""
        info = VersionInfo(1, 0, 0, None, None)
        with pytest.raises(AttributeError):
            info.major = 2  # type: ignore[misc]

    def test_version_info_has_slots(self) -> None:
        """VersionInfo uses __slots__."""
        info = VersionInfo(1, 0, 0, None, None)
        assert not hasattr(info, "__dict__")


class TestVersionConfig:
    """Tests for VersionConfig dataclass."""

    def test_create_version_config(self) -> None:
        """Create version config."""
        config = VersionConfig(
            scheme=VersionScheme.SEMANTIC,
            auto_increment=True,
            track_changes=True,
            require_changelog=False,
        )
        assert config.scheme == VersionScheme.SEMANTIC
        assert config.auto_increment is True

    def test_config_is_frozen(self) -> None:
        """VersionConfig is immutable."""
        config = VersionConfig(
            VersionScheme.SEMANTIC,
            True,
            True,
            False,
        )
        with pytest.raises(AttributeError):
            config.auto_increment = False  # type: ignore[misc]

    def test_config_has_slots(self) -> None:
        """VersionConfig uses __slots__."""
        config = VersionConfig(VersionScheme.SEMANTIC, True, True, False)
        assert not hasattr(config, "__dict__")


class TestModelDiff:
    """Tests for ModelDiff dataclass."""

    def test_create_model_diff(self) -> None:
        """Create model diff."""
        diff = ModelDiff(
            changed_files=("config.json", "model.bin"),
            added_params=1000,
            removed_params=0,
            config_changes={"hidden_size": (768, 1024)},
        )
        assert "config.json" in diff.changed_files
        assert diff.added_params == 1000

    def test_diff_is_frozen(self) -> None:
        """ModelDiff is immutable."""
        diff = ModelDiff((), 0, 0, {})
        with pytest.raises(AttributeError):
            diff.added_params = 100  # type: ignore[misc]

    def test_diff_has_slots(self) -> None:
        """ModelDiff uses __slots__."""
        diff = ModelDiff((), 0, 0, {})
        assert not hasattr(diff, "__dict__")


class TestVersionStats:
    """Tests for VersionStats dataclass."""

    def test_create_version_stats(self) -> None:
        """Create version stats."""
        stats = VersionStats(
            total_versions=10,
            latest_version="2.0.0",
            release_frequency_days=14.0,
        )
        assert stats.total_versions == 10
        assert stats.latest_version == "2.0.0"
        assert stats.release_frequency_days == pytest.approx(14.0)

    def test_stats_is_frozen(self) -> None:
        """VersionStats is immutable."""
        stats = VersionStats(10, "1.0.0", 7.0)
        with pytest.raises(AttributeError):
            stats.total_versions = 20  # type: ignore[misc]

    def test_stats_has_slots(self) -> None:
        """VersionStats uses __slots__."""
        stats = VersionStats(10, "1.0.0", 7.0)
        assert not hasattr(stats, "__dict__")


class TestValidateVersionInfo:
    """Tests for validate_version_info function."""

    def test_valid_info(self) -> None:
        """Valid info passes validation."""
        info = VersionInfo(1, 2, 3, "alpha", "build.1")
        validate_version_info(info)

    def test_valid_info_no_pre_release(self) -> None:
        """Info without pre_release passes validation."""
        info = VersionInfo(1, 0, 0, None, None)
        validate_version_info(info)

    def test_negative_major_raises(self) -> None:
        """Negative major raises ValueError."""
        info = VersionInfo(-1, 0, 0, None, None)
        with pytest.raises(ValueError, match="major must be non-negative"):
            validate_version_info(info)

    def test_negative_minor_raises(self) -> None:
        """Negative minor raises ValueError."""
        info = VersionInfo(0, -1, 0, None, None)
        with pytest.raises(ValueError, match="minor must be non-negative"):
            validate_version_info(info)

    def test_negative_patch_raises(self) -> None:
        """Negative patch raises ValueError."""
        info = VersionInfo(0, 0, -1, None, None)
        with pytest.raises(ValueError, match="patch must be non-negative"):
            validate_version_info(info)

    def test_empty_pre_release_raises(self) -> None:
        """Empty pre_release string raises ValueError."""
        info = VersionInfo(1, 0, 0, "", None)
        with pytest.raises(ValueError, match="pre_release cannot be an empty string"):
            validate_version_info(info)

    def test_empty_build_metadata_raises(self) -> None:
        """Empty build_metadata string raises ValueError."""
        info = VersionInfo(1, 0, 0, None, "")
        with pytest.raises(ValueError, match="build_metadata cannot be an empty"):
            validate_version_info(info)


class TestValidateVersionConfig:
    """Tests for validate_version_config function."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = VersionConfig(VersionScheme.SEMANTIC, True, True, False)
        validate_version_config(config)

    def test_all_schemes_valid(self) -> None:
        """All schemes pass validation."""
        for scheme in VersionScheme:
            config = VersionConfig(scheme, True, False, False)
            validate_version_config(config)


class TestValidateModelDiff:
    """Tests for validate_model_diff function."""

    def test_valid_diff(self) -> None:
        """Valid diff passes validation."""
        diff = ModelDiff(("config.json",), 100, 0, {})
        validate_model_diff(diff)

    def test_empty_files_valid(self) -> None:
        """Empty changed_files passes validation."""
        diff = ModelDiff((), 0, 0, {})
        validate_model_diff(diff)

    def test_empty_file_string_raises(self) -> None:
        """Empty string in changed_files raises ValueError."""
        diff = ModelDiff(("",), 0, 0, {})
        with pytest.raises(ValueError, match="changed_files cannot contain empty"):
            validate_model_diff(diff)

    def test_negative_added_params_raises(self) -> None:
        """Negative added_params raises ValueError."""
        diff = ModelDiff((), -1, 0, {})
        with pytest.raises(ValueError, match="added_params must be non-negative"):
            validate_model_diff(diff)

    def test_negative_removed_params_raises(self) -> None:
        """Negative removed_params raises ValueError."""
        diff = ModelDiff((), 0, -1, {})
        with pytest.raises(ValueError, match="removed_params must be non-negative"):
            validate_model_diff(diff)


class TestValidateVersionStats:
    """Tests for validate_version_stats function."""

    def test_valid_stats(self) -> None:
        """Valid stats passes validation."""
        stats = VersionStats(10, "1.0.0", 7.0)
        validate_version_stats(stats)

    def test_none_frequency_valid(self) -> None:
        """None release_frequency_days passes validation."""
        stats = VersionStats(5, "1.0.0", None)
        validate_version_stats(stats)

    def test_negative_total_raises(self) -> None:
        """Negative total_versions raises ValueError."""
        stats = VersionStats(-1, "1.0.0", 7.0)
        with pytest.raises(ValueError, match="total_versions must be non-negative"):
            validate_version_stats(stats)

    def test_empty_latest_raises(self) -> None:
        """Empty latest_version raises ValueError."""
        stats = VersionStats(10, "", 7.0)
        with pytest.raises(ValueError, match="latest_version cannot be empty"):
            validate_version_stats(stats)

    def test_negative_frequency_raises(self) -> None:
        """Negative release_frequency_days raises ValueError."""
        stats = VersionStats(10, "1.0.0", -1.0)
        with pytest.raises(ValueError, match="release_frequency_days must be non-neg"):
            validate_version_stats(stats)


class TestCreateVersionInfo:
    """Tests for create_version_info function."""

    def test_default_info(self) -> None:
        """Create default version info."""
        info = create_version_info()
        assert info.major == 0
        assert info.minor == 0
        assert info.patch == 0

    def test_custom_info(self) -> None:
        """Create custom version info."""
        info = create_version_info(1, 2, 3, pre_release="alpha", build_metadata="123")
        assert info.major == 1
        assert info.minor == 2
        assert info.patch == 3
        assert info.pre_release == "alpha"
        assert info.build_metadata == "123"

    def test_negative_major_raises(self) -> None:
        """Negative major raises ValueError."""
        with pytest.raises(ValueError, match="major must be non-negative"):
            create_version_info(-1, 0, 0)

    def test_negative_minor_raises(self) -> None:
        """Negative minor raises ValueError."""
        with pytest.raises(ValueError, match="minor must be non-negative"):
            create_version_info(0, -1, 0)

    def test_negative_patch_raises(self) -> None:
        """Negative patch raises ValueError."""
        with pytest.raises(ValueError, match="patch must be non-negative"):
            create_version_info(0, 0, -1)


class TestCreateVersionConfig:
    """Tests for create_version_config function."""

    def test_default_config(self) -> None:
        """Create default config."""
        config = create_version_config()
        assert config.scheme == VersionScheme.SEMANTIC
        assert config.auto_increment is True
        assert config.track_changes is True
        assert config.require_changelog is False

    def test_custom_config(self) -> None:
        """Create custom config."""
        config = create_version_config(
            scheme="date_based",
            auto_increment=False,
            track_changes=False,
            require_changelog=True,
        )
        assert config.scheme == VersionScheme.DATE_BASED
        assert config.auto_increment is False
        assert config.require_changelog is True

    def test_invalid_scheme_raises(self) -> None:
        """Invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="scheme must be one of"):
            create_version_config(scheme="invalid")


class TestCreateModelDiff:
    """Tests for create_model_diff function."""

    def test_default_diff(self) -> None:
        """Create default diff."""
        diff = create_model_diff()
        assert diff.changed_files == ()
        assert diff.added_params == 0
        assert diff.removed_params == 0
        assert diff.config_changes == {}

    def test_custom_diff(self) -> None:
        """Create custom diff."""
        diff = create_model_diff(
            changed_files=["config.json", "model.bin"],
            added_params=500,
            removed_params=100,
            config_changes={"hidden": (768, 1024)},
        )
        assert diff.changed_files == ("config.json", "model.bin")
        assert diff.added_params == 500
        assert diff.removed_params == 100

    def test_tuple_files_accepted(self) -> None:
        """Tuple for changed_files is accepted."""
        diff = create_model_diff(changed_files=("a.json", "b.json"))
        assert diff.changed_files == ("a.json", "b.json")

    def test_negative_added_raises(self) -> None:
        """Negative added_params raises ValueError."""
        with pytest.raises(ValueError, match="added_params must be non-negative"):
            create_model_diff(added_params=-1)

    def test_negative_removed_raises(self) -> None:
        """Negative removed_params raises ValueError."""
        with pytest.raises(ValueError, match="removed_params must be non-negative"):
            create_model_diff(removed_params=-1)

    def test_empty_file_raises(self) -> None:
        """Empty string in changed_files raises ValueError."""
        with pytest.raises(ValueError, match="changed_files cannot contain empty"):
            create_model_diff(changed_files=[""])


class TestCreateVersionStats:
    """Tests for create_version_stats function."""

    def test_default_stats(self) -> None:
        """Create default stats."""
        stats = create_version_stats()
        assert stats.total_versions == 0
        assert stats.latest_version == "0.0.0"
        assert stats.release_frequency_days is None

    def test_custom_stats(self) -> None:
        """Create custom stats."""
        stats = create_version_stats(
            total_versions=20,
            latest_version="3.0.0",
            release_frequency_days=7.5,
        )
        assert stats.total_versions == 20
        assert stats.latest_version == "3.0.0"
        assert stats.release_frequency_days == pytest.approx(7.5)

    def test_negative_total_raises(self) -> None:
        """Negative total_versions raises ValueError."""
        with pytest.raises(ValueError, match="total_versions must be non-negative"):
            create_version_stats(total_versions=-1)


class TestListVersionSchemes:
    """Tests for list_version_schemes function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        schemes = list_version_schemes()
        assert schemes == sorted(schemes)

    def test_contains_semantic(self) -> None:
        """Contains semantic."""
        schemes = list_version_schemes()
        assert "semantic" in schemes

    def test_contains_date_based(self) -> None:
        """Contains date_based."""
        schemes = list_version_schemes()
        assert "date_based" in schemes


class TestListChangeTypes:
    """Tests for list_change_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_change_types()
        assert types == sorted(types)

    def test_contains_major(self) -> None:
        """Contains major."""
        types = list_change_types()
        assert "major" in types

    def test_contains_patch(self) -> None:
        """Contains patch."""
        types = list_change_types()
        assert "patch" in types


class TestListDiffTypes:
    """Tests for list_diff_types function."""

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list."""
        types = list_diff_types()
        assert types == sorted(types)

    def test_contains_config(self) -> None:
        """Contains config."""
        types = list_diff_types()
        assert "config" in types

    def test_contains_weights(self) -> None:
        """Contains weights."""
        types = list_diff_types()
        assert "weights" in types


class TestGetVersionScheme:
    """Tests for get_version_scheme function."""

    def test_get_semantic(self) -> None:
        """Get semantic scheme."""
        assert get_version_scheme("semantic") == VersionScheme.SEMANTIC

    def test_get_date_based(self) -> None:
        """Get date_based scheme."""
        assert get_version_scheme("date_based") == VersionScheme.DATE_BASED

    def test_get_incremental(self) -> None:
        """Get incremental scheme."""
        assert get_version_scheme("incremental") == VersionScheme.INCREMENTAL

    def test_get_git_hash(self) -> None:
        """Get git_hash scheme."""
        assert get_version_scheme("git_hash") == VersionScheme.GIT_HASH

    def test_invalid_scheme_raises(self) -> None:
        """Invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="version_scheme must be one of"):
            get_version_scheme("invalid")


class TestGetChangeType:
    """Tests for get_change_type function."""

    def test_get_major(self) -> None:
        """Get major change type."""
        assert get_change_type("major") == ChangeType.MAJOR

    def test_get_minor(self) -> None:
        """Get minor change type."""
        assert get_change_type("minor") == ChangeType.MINOR

    def test_get_patch(self) -> None:
        """Get patch change type."""
        assert get_change_type("patch") == ChangeType.PATCH

    def test_get_pre_release(self) -> None:
        """Get pre_release change type."""
        assert get_change_type("pre_release") == ChangeType.PRE_RELEASE

    def test_invalid_type_raises(self) -> None:
        """Invalid change type raises ValueError."""
        with pytest.raises(ValueError, match="change_type must be one of"):
            get_change_type("invalid")


class TestGetDiffType:
    """Tests for get_diff_type function."""

    def test_get_config(self) -> None:
        """Get config diff type."""
        assert get_diff_type("config") == DiffType.CONFIG

    def test_get_weights(self) -> None:
        """Get weights diff type."""
        assert get_diff_type("weights") == DiffType.WEIGHTS

    def test_get_tokenizer(self) -> None:
        """Get tokenizer diff type."""
        assert get_diff_type("tokenizer") == DiffType.TOKENIZER

    def test_get_all(self) -> None:
        """Get all diff type."""
        assert get_diff_type("all") == DiffType.ALL

    def test_invalid_type_raises(self) -> None:
        """Invalid diff type raises ValueError."""
        with pytest.raises(ValueError, match="diff_type must be one of"):
            get_diff_type("invalid")


class TestParseVersion:
    """Tests for parse_version function."""

    def test_parse_simple_version(self) -> None:
        """Parse simple version string."""
        info = parse_version("1.2.3")
        assert info.major == 1
        assert info.minor == 2
        assert info.patch == 3
        assert info.pre_release is None
        assert info.build_metadata is None

    def test_parse_with_v_prefix(self) -> None:
        """Parse version with v prefix."""
        info = parse_version("v1.0.0")
        assert info.major == 1
        assert info.minor == 0
        assert info.patch == 0

    def test_parse_with_pre_release(self) -> None:
        """Parse version with pre-release."""
        info = parse_version("2.0.0-alpha")
        assert info.major == 2
        assert info.pre_release == "alpha"

    def test_parse_with_complex_pre_release(self) -> None:
        """Parse version with complex pre-release."""
        info = parse_version("1.0.0-beta.1")
        assert info.pre_release == "beta.1"

    def test_parse_with_build_metadata(self) -> None:
        """Parse version with build metadata."""
        info = parse_version("1.0.0+build.123")
        assert info.build_metadata == "build.123"

    def test_parse_full_version(self) -> None:
        """Parse version with all components."""
        info = parse_version("3.2.1-rc.1+build.456")
        assert info.major == 3
        assert info.minor == 2
        assert info.patch == 1
        assert info.pre_release == "rc.1"
        assert info.build_metadata == "build.456"

    def test_parse_zero_version(self) -> None:
        """Parse zero version."""
        info = parse_version("0.0.0")
        assert info.major == 0
        assert info.minor == 0
        assert info.patch == 0

    def test_empty_string_raises(self) -> None:
        """Empty version string raises ValueError."""
        with pytest.raises(ValueError, match="version_string cannot be empty"):
            parse_version("")

    def test_invalid_format_raises(self) -> None:
        """Invalid version format raises ValueError."""
        with pytest.raises(ValueError, match="invalid version format"):
            parse_version("invalid")

    def test_partial_version_raises(self) -> None:
        """Partial version string raises ValueError."""
        with pytest.raises(ValueError, match="invalid version format"):
            parse_version("1.2")

    def test_leading_zeros_invalid(self) -> None:
        """Leading zeros in version numbers are invalid (except 0 itself)."""
        with pytest.raises(ValueError, match="invalid version format"):
            parse_version("01.0.0")


class TestCompareVersions:
    """Tests for compare_versions function."""

    def test_equal_versions(self) -> None:
        """Equal versions return 0."""
        v1 = create_version_info(1, 0, 0)
        v2 = create_version_info(1, 0, 0)
        assert compare_versions(v1, v2) == 0

    def test_major_less(self) -> None:
        """Lower major version returns -1."""
        v1 = create_version_info(1, 0, 0)
        v2 = create_version_info(2, 0, 0)
        assert compare_versions(v1, v2) == -1

    def test_major_greater(self) -> None:
        """Higher major version returns 1."""
        v1 = create_version_info(2, 0, 0)
        v2 = create_version_info(1, 0, 0)
        assert compare_versions(v1, v2) == 1

    def test_minor_less(self) -> None:
        """Lower minor version returns -1."""
        v1 = create_version_info(1, 1, 0)
        v2 = create_version_info(1, 2, 0)
        assert compare_versions(v1, v2) == -1

    def test_minor_greater(self) -> None:
        """Higher minor version returns 1."""
        v1 = create_version_info(1, 2, 0)
        v2 = create_version_info(1, 1, 0)
        assert compare_versions(v1, v2) == 1

    def test_patch_less(self) -> None:
        """Lower patch version returns -1."""
        v1 = create_version_info(1, 0, 1)
        v2 = create_version_info(1, 0, 2)
        assert compare_versions(v1, v2) == -1

    def test_patch_greater(self) -> None:
        """Higher patch version returns 1."""
        v1 = create_version_info(1, 0, 2)
        v2 = create_version_info(1, 0, 1)
        assert compare_versions(v1, v2) == 1

    def test_pre_release_lower_than_release(self) -> None:
        """Pre-release version is lower than release."""
        v1 = create_version_info(1, 0, 0, pre_release="alpha")
        v2 = create_version_info(1, 0, 0)
        assert compare_versions(v1, v2) == -1

    def test_release_higher_than_pre_release(self) -> None:
        """Release version is higher than pre-release."""
        v1 = create_version_info(1, 0, 0)
        v2 = create_version_info(1, 0, 0, pre_release="alpha")
        assert compare_versions(v1, v2) == 1

    def test_pre_release_comparison(self) -> None:
        """Pre-release versions compared alphabetically."""
        v1 = create_version_info(1, 0, 0, pre_release="alpha")
        v2 = create_version_info(1, 0, 0, pre_release="beta")
        assert compare_versions(v1, v2) == -1

    def test_pre_release_equal(self) -> None:
        """Equal pre-release versions return 0."""
        v1 = create_version_info(1, 0, 0, pre_release="alpha")
        v2 = create_version_info(1, 0, 0, pre_release="alpha")
        assert compare_versions(v1, v2) == 0

    def test_pre_release_greater(self) -> None:
        """Higher pre-release returns 1."""
        v1 = create_version_info(1, 0, 0, pre_release="beta")
        v2 = create_version_info(1, 0, 0, pre_release="alpha")
        assert compare_versions(v1, v2) == 1


class TestIncrementVersion:
    """Tests for increment_version function."""

    def test_increment_major(self) -> None:
        """Increment major version."""
        v = create_version_info(1, 2, 3)
        new_v = increment_version(v, "major")
        assert new_v.major == 2
        assert new_v.minor == 0
        assert new_v.patch == 0

    def test_increment_minor(self) -> None:
        """Increment minor version."""
        v = create_version_info(1, 2, 3)
        new_v = increment_version(v, "minor")
        assert new_v.major == 1
        assert new_v.minor == 3
        assert new_v.patch == 0

    def test_increment_patch(self) -> None:
        """Increment patch version."""
        v = create_version_info(1, 2, 3)
        new_v = increment_version(v, "patch")
        assert new_v.major == 1
        assert new_v.minor == 2
        assert new_v.patch == 4

    def test_increment_pre_release_new(self) -> None:
        """Create new pre-release when none exists."""
        v = create_version_info(1, 0, 0)
        new_v = increment_version(v, "pre_release")
        assert new_v.pre_release == "alpha.1"

    def test_increment_pre_release_existing(self) -> None:
        """Increment existing pre-release."""
        v = create_version_info(1, 0, 0, pre_release="alpha.1")
        new_v = increment_version(v, "pre_release")
        assert new_v.pre_release == "alpha.2"

    def test_increment_pre_release_no_number(self) -> None:
        """Increment pre-release without numeric suffix."""
        v = create_version_info(1, 0, 0, pre_release="alpha")
        new_v = increment_version(v, "pre_release")
        assert new_v.pre_release == "alpha.1"

    def test_increment_clears_pre_release(self) -> None:
        """Major/minor/patch increment clears pre-release."""
        v = create_version_info(1, 0, 0, pre_release="alpha")
        new_v = increment_version(v, "major")
        assert new_v.pre_release is None

    def test_invalid_change_type_raises(self) -> None:
        """Invalid change_type raises ValueError."""
        v = create_version_info(1, 0, 0)
        with pytest.raises(ValueError, match="change_type must be one of"):
            increment_version(v, "invalid")


class TestCalculateModelDiff:
    """Tests for calculate_model_diff function."""

    def test_no_changes(self) -> None:
        """No changes returns empty diff."""
        diff = calculate_model_diff(
            old_files={"a.json": "hash1"},
            new_files={"a.json": "hash1"},
        )
        assert diff.changed_files == ()
        assert diff.added_params == 0
        assert diff.removed_params == 0

    def test_file_changed(self) -> None:
        """Changed file detected."""
        diff = calculate_model_diff(
            old_files={"config.json": "hash1"},
            new_files={"config.json": "hash2"},
        )
        assert "config.json" in diff.changed_files

    def test_file_added(self) -> None:
        """Added file detected."""
        diff = calculate_model_diff(
            old_files={},
            new_files={"new.json": "hash1"},
        )
        assert "new.json" in diff.changed_files

    def test_file_removed(self) -> None:
        """Removed file detected."""
        diff = calculate_model_diff(
            old_files={"old.json": "hash1"},
            new_files={},
        )
        assert "old.json" in diff.changed_files

    def test_params_added(self) -> None:
        """Parameters added calculated."""
        diff = calculate_model_diff(
            old_files={},
            new_files={},
            old_params=1000,
            new_params=1500,
        )
        assert diff.added_params == 500
        assert diff.removed_params == 0

    def test_params_removed(self) -> None:
        """Parameters removed calculated."""
        diff = calculate_model_diff(
            old_files={},
            new_files={},
            old_params=1500,
            new_params=1000,
        )
        assert diff.added_params == 0
        assert diff.removed_params == 500

    def test_config_changes(self) -> None:
        """Config changes detected."""
        diff = calculate_model_diff(
            old_files={},
            new_files={},
            old_config={"hidden_size": 768},
            new_config={"hidden_size": 1024},
        )
        assert diff.config_changes["hidden_size"] == (768, 1024)

    def test_config_added(self) -> None:
        """Config key added detected."""
        diff = calculate_model_diff(
            old_files={},
            new_files={},
            old_config={},
            new_config={"new_key": "value"},
        )
        assert diff.config_changes["new_key"] == (None, "value")

    def test_config_removed(self) -> None:
        """Config key removed detected."""
        diff = calculate_model_diff(
            old_files={},
            new_files={},
            old_config={"old_key": "value"},
            new_config={},
        )
        assert diff.config_changes["old_key"] == ("value", None)

    def test_changed_files_sorted(self) -> None:
        """Changed files are sorted."""
        diff = calculate_model_diff(
            old_files={"z.json": "1", "a.json": "1"},
            new_files={"z.json": "2", "a.json": "2"},
        )
        assert diff.changed_files == ("a.json", "z.json")


class TestValidateVersionBump:
    """Tests for validate_version_bump function."""

    def test_valid_major_bump(self) -> None:
        """Valid major version bump."""
        current = create_version_info(1, 2, 3)
        proposed = create_version_info(2, 0, 0)
        assert validate_version_bump(current, proposed, "major") is True

    def test_valid_minor_bump(self) -> None:
        """Valid minor version bump."""
        current = create_version_info(1, 2, 3)
        proposed = create_version_info(1, 3, 0)
        assert validate_version_bump(current, proposed, "minor") is True

    def test_valid_patch_bump(self) -> None:
        """Valid patch version bump."""
        current = create_version_info(1, 2, 3)
        proposed = create_version_info(1, 2, 4)
        assert validate_version_bump(current, proposed, "patch") is True

    def test_invalid_major_bump(self) -> None:
        """Invalid major version bump."""
        current = create_version_info(1, 2, 3)
        proposed = create_version_info(3, 0, 0)
        assert validate_version_bump(current, proposed, "major") is False

    def test_invalid_minor_bump(self) -> None:
        """Invalid minor version bump."""
        current = create_version_info(1, 2, 3)
        proposed = create_version_info(1, 4, 0)
        assert validate_version_bump(current, proposed, "minor") is False

    def test_invalid_patch_bump(self) -> None:
        """Invalid patch version bump."""
        current = create_version_info(1, 2, 3)
        proposed = create_version_info(1, 2, 5)
        assert validate_version_bump(current, proposed, "patch") is False

    def test_valid_pre_release_bump(self) -> None:
        """Valid pre-release version bump."""
        current = create_version_info(1, 0, 0)
        proposed = create_version_info(1, 0, 0, pre_release="alpha")
        assert validate_version_bump(current, proposed, "pre_release") is True

    def test_invalid_pre_release_bump(self) -> None:
        """Invalid pre-release without identifier."""
        current = create_version_info(1, 0, 0)
        proposed = create_version_info(1, 0, 0)
        assert validate_version_bump(current, proposed, "pre_release") is False

    def test_invalid_change_type_raises(self) -> None:
        """Invalid change_type raises ValueError."""
        current = create_version_info(1, 0, 0)
        proposed = create_version_info(2, 0, 0)
        with pytest.raises(ValueError, match="change_type must be one of"):
            validate_version_bump(current, proposed, "invalid")


class TestFormatVersionStats:
    """Tests for format_version_stats function."""

    def test_format_full_stats(self) -> None:
        """Format stats with all fields."""
        stats = create_version_stats(
            total_versions=10,
            latest_version="2.1.0",
            release_frequency_days=14.5,
        )
        output = format_version_stats(stats)
        assert "10 versions" in output
        assert "2.1.0" in output
        assert "14.5 days" in output

    def test_format_empty_stats(self) -> None:
        """Format empty stats."""
        stats = create_version_stats()
        output = format_version_stats(stats)
        assert "0 versions" in output
        assert "0.0.0" in output
        assert "N/A" in output

    def test_format_no_frequency(self) -> None:
        """Format stats without frequency."""
        stats = create_version_stats(
            total_versions=5,
            latest_version="1.0.0",
        )
        output = format_version_stats(stats)
        assert "N/A" in output


class TestFormatVersion:
    """Tests for format_version function."""

    def test_format_simple_version(self) -> None:
        """Format simple version."""
        v = create_version_info(1, 2, 3)
        assert format_version(v) == "1.2.3"

    def test_format_with_pre_release(self) -> None:
        """Format version with pre-release."""
        v = create_version_info(1, 0, 0, pre_release="alpha")
        assert format_version(v) == "1.0.0-alpha"

    def test_format_with_build_metadata(self) -> None:
        """Format version with build metadata."""
        v = create_version_info(2, 0, 0, build_metadata="build.123")
        assert format_version(v) == "2.0.0+build.123"

    def test_format_full_version(self) -> None:
        """Format version with all components."""
        v = create_version_info(1, 0, 0, pre_release="beta", build_metadata="456")
        assert format_version(v) == "1.0.0-beta+456"

    def test_format_zero_version(self) -> None:
        """Format zero version."""
        v = create_version_info(0, 0, 0)
        assert format_version(v) == "0.0.0"


class TestGetRecommendedVersionConfig:
    """Tests for get_recommended_version_config function."""

    def test_default_config(self) -> None:
        """Get default recommended config."""
        config = get_recommended_version_config()
        assert config.scheme == VersionScheme.SEMANTIC
        assert config.auto_increment is True
        assert config.track_changes is True
        assert config.require_changelog is False

    def test_dataset_project(self) -> None:
        """Dataset projects get date-based versioning."""
        config = get_recommended_version_config(project_type="dataset")
        assert config.scheme == VersionScheme.DATE_BASED

    def test_library_project(self) -> None:
        """Library projects get semantic versioning."""
        config = get_recommended_version_config(project_type="library")
        assert config.scheme == VersionScheme.SEMANTIC

    def test_large_team_requires_changelog(self) -> None:
        """Large teams require changelog."""
        config = get_recommended_version_config(team_size=5)
        assert config.require_changelog is True

    def test_small_team_no_changelog(self) -> None:
        """Small teams don't require changelog."""
        config = get_recommended_version_config(team_size=2)
        assert config.require_changelog is False

    def test_invalid_project_type_raises(self) -> None:
        """Invalid project_type raises ValueError."""
        with pytest.raises(ValueError, match="project_type must be one of"):
            get_recommended_version_config(project_type="invalid")

    def test_zero_team_size_raises(self) -> None:
        """Zero team_size raises ValueError."""
        with pytest.raises(ValueError, match="team_size must be positive"):
            get_recommended_version_config(team_size=0)

    def test_negative_team_size_raises(self) -> None:
        """Negative team_size raises ValueError."""
        with pytest.raises(ValueError, match="team_size must be positive"):
            get_recommended_version_config(team_size=-1)
